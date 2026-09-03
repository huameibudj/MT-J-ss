import os
import sys
import yaml
import csv
import contextlib

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from models.vit_backbone import CompatibleViTBackbone
from models.eomt_head import EoMT


CLASS_NAMES = {
    0: "Background",
    1: "Yellow",
    2: "Red",
    3: "Cyan",
    4: "Blue",
    5: "Green",
}

TASK_NAME = "glassai_6class_segmentation_pretrained_frozen"
ABLATION_MODE = "pretrained_backbone_fully_frozen"


# -------------------------
# YAML 读取
# -------------------------
def load_yaml_any_encoding(path: str):
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return yaml.safe_load(f)


# -------------------------
# GlassAI 协议
# -------------------------
def enforce_glassai_protocol(cfg):
    old_dataset_name = str(cfg.get("dataset_name", "glassai")).lower()
    old_num_classes = int(cfg.get("num_classes", 6))
    old_ignore_index = int(cfg.get("ignore_index", 255))

    if old_dataset_name != "glassai":
        print(
            f"[WARN] config dataset_name={old_dataset_name}, "
            f"but this evaluate.py is for GlassAI. Override dataset_name -> glassai."
        )

    if old_num_classes != 6:
        print(
            f"[WARN] config num_classes={old_num_classes}, "
            f"but GlassAI requires num_classes=6. Override num_classes -> 6."
        )

    if old_ignore_index != 255:
        print(
            f"[WARN] config ignore_index={old_ignore_index}, "
            f"but GlassAI unknown colors use ignore_index=255. Override ignore_index -> 255."
        )

    cfg["dataset_name"] = "glassai"
    cfg["num_classes"] = 6
    cfg["ignore_index"] = 255

    return cfg


# -------------------------
# AMP dtype
# -------------------------
def pick_amp_dtype():
    if (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16

    return torch.float16


def configure_fast_runtime(device, cfg):
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))

    if device.type == "cuda":
        allow_tf32 = bool(cfg.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(
                str(cfg.get("float32_matmul_precision", "high"))
            )


def autocast_context(device, enabled, dtype):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=dtype)

    return torch.cuda.amp.autocast(dtype=dtype)


def make_progress(iterable, desc, cfg, leave=False, total=None):
    explicit_disable = cfg.get("tqdm_disable", None)

    if explicit_disable is None:
        auto_disable = bool(cfg.get("tqdm_auto_disable_non_tty", True))
        disable = auto_disable and not sys.stderr.isatty()
    else:
        disable = bool(explicit_disable)

    return tqdm(
        iterable,
        total=total,
        desc=desc,
        dynamic_ncols=True,
        mininterval=float(cfg.get("tqdm_mininterval", 0.8)),
        leave=leave,
        position=0,
        disable=disable,
        file=sys.stderr,
    )


# -------------------------
# 从 cfg 构建 dataset
# -------------------------
def build_dataset_from_cfg(cfg, img_list, lbl_list):
    dataset_name = str(cfg.get("dataset_name", "glassai")).lower()

    resize_hw = cfg.get("resize_hw", None)

    if isinstance(resize_hw, list) and len(resize_hw) == 2:
        resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
    else:
        resize_hw = None

    ignore_index = int(cfg.get("ignore_index", 255))

    if dataset_name == "glassai":
        try:
            from datasets.glassai_dataset import GlassAIDataset, FIXED_PALETTE
        except ImportError:
            from glassai_dataset import GlassAIDataset, FIXED_PALETTE

        ds = GlassAIDataset(
            image_paths=img_list,
            label_paths=lbl_list,
            palette=FIXED_PALETTE,
            resize_hw=resize_hw,
            ignore_index=ignore_index,
        )

        return ds, FIXED_PALETTE, dataset_name

    raise ValueError(f"Unsupported dataset_name: {dataset_name}. This evaluate.py only supports glassai.")


# -------------------------
# 读取 splits/test.txt
# -------------------------
def read_split_list(split_file: str):
    img_list, lbl_list = [], []

    with open(split_file, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()

            if not line:
                continue

            if "\t" in line:
                parts = line.split("\t")
            elif "|||" in line:
                parts = line.split("|||")
            else:
                raise ValueError(
                    f"[ERROR] split file line {line_no} format error: {line}\n"
                    f"Expected format: 'img_path\\tlabel_path' or 'img_path|||label_path'"
                )

            if len(parts) != 2:
                raise ValueError(
                    f"[ERROR] split file line {line_no} cannot be parsed into two columns: {line}"
                )

            img_p = parts[0].strip()
            lbl_p = parts[1].strip()

            if not img_p or not lbl_p:
                raise ValueError(
                    f"[ERROR] split file line {line_no} contains empty path: {line}"
                )

            img_list.append(img_p)
            lbl_list.append(lbl_p)

    return img_list, lbl_list


# -------------------------
# 混淆矩阵更新
# -------------------------
@torch.inference_mode()
def update_confusion_matrix(conf_mat, pred, target, num_classes, ignore_index=255):
    """Vectorized update for a complete batch on CPU or GPU."""
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    valid = valid & (pred >= 0) & (pred < num_classes)

    if not bool(valid.any()):
        return conf_mat

    t = target[valid].reshape(-1).to(torch.int64)
    p = pred[valid].reshape(-1).to(torch.int64)
    idx = t * num_classes + p
    bins = torch.bincount(idx, minlength=num_classes * num_classes)
    conf_mat.add_(bins.reshape(num_classes, num_classes).to(conf_mat.device))
    return conf_mat


# -------------------------
# 计算分割指标
# -------------------------
def compute_metrics_from_confmat(conf_mat: torch.Tensor, foreground_class_ids=None, eps=1e-12):
    """
    默认：
    - macro mean_iou / mean_dice：对全部类别 0..5 求平均
    - foreground_mean_iou / foreground_mean_dice：只对前景类 1..5 求平均
    """
    conf = conf_mat.to(torch.float64)
    num_classes = conf.shape[0]

    if foreground_class_ids is None:
        foreground_class_ids = list(range(1, num_classes))

    foreground_class_ids = [
        int(c) for c in foreground_class_ids
        if 0 <= int(c) < num_classes
    ]

    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    tn = conf.sum() - tp - fp - fn

    support = conf.sum(dim=1)
    pred_count = conf.sum(dim=0)

    dice = torch.where(
        2 * tp + fp + fn > 0,
        (2 * tp) / (2 * tp + fp + fn + eps),
        torch.zeros_like(tp),
    )

    f1 = dice.clone()

    iou = torch.where(
        tp + fp + fn > 0,
        tp / (tp + fp + fn + eps),
        torch.zeros_like(tp),
    )

    precision = torch.where(
        tp + fp > 0,
        tp / (tp + fp + eps),
        torch.zeros_like(tp),
    )

    recall = torch.where(
        tp + fn > 0,
        tp / (tp + fn + eps),
        torch.zeros_like(tp),
    )

    sensitivity = recall.clone()

    specificity = torch.where(
        tn + fp > 0,
        tn / (tn + fp + eps),
        torch.zeros_like(tp),
    )

    fpr = torch.where(
        fp + tn > 0,
        fp / (fp + tn + eps),
        torch.zeros_like(tp),
    )

    fnr = torch.where(
        fn + tp > 0,
        fn / (fn + tp + eps),
        torch.zeros_like(tp),
    )

    accuracy_per_class = torch.where(
        tp + tn + fp + fn > 0,
        (tp + tn) / (tp + tn + fp + fn + eps),
        torch.zeros_like(tp),
    )

    total = conf.sum()
    overall_accuracy = float(tp.sum().item() / (total.item() + eps)) if total.item() > 0 else 0.0
    mean_accuracy = float(recall.mean().item())

    freq = torch.where(
        total > 0,
        support / (total + eps),
        torch.zeros_like(support),
    )

    fw_iou = float((freq * iou).sum().item())

    if len(foreground_class_ids) > 0:
        fg_idx = torch.tensor(foreground_class_ids, dtype=torch.long)
        foreground_mean_iou = float(iou[fg_idx].mean().item())
        foreground_mean_dice = float(dice[fg_idx].mean().item())
        foreground_mean_f1 = float(f1[fg_idx].mean().item())
        foreground_mean_precision = float(precision[fg_idx].mean().item())
        foreground_mean_recall = float(recall[fg_idx].mean().item())
    else:
        foreground_mean_iou = 0.0
        foreground_mean_dice = 0.0
        foreground_mean_f1 = 0.0
        foreground_mean_precision = 0.0
        foreground_mean_recall = 0.0

    macro = {
        "mean_iou": float(iou.mean().item()),
        "mean_dice": float(dice.mean().item()),
        "mean_f1": float(f1.mean().item()),
        "mean_precision": float(precision.mean().item()),
        "mean_recall": float(recall.mean().item()),
        "mean_sensitivity": float(sensitivity.mean().item()),
        "mean_specificity": float(specificity.mean().item()),
        "mean_accuracy": mean_accuracy,
        "overall_accuracy": overall_accuracy,
        "fw_iou": fw_iou,

        # 重点新增：前景指标，默认 class 1..5
        "foreground_mean_iou": foreground_mean_iou,
        "foreground_mean_dice": foreground_mean_dice,
        "foreground_mean_f1": foreground_mean_f1,
        "foreground_mean_precision": foreground_mean_precision,
        "foreground_mean_recall": foreground_mean_recall,
    }

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": support,
        "pred_count": pred_count,
        "iou": iou,
        "dice": dice,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "accuracy_per_class": accuracy_per_class,
        "macro": macro,
        "foreground_class_ids": foreground_class_ids,
    }


# -------------------------
# GlassAI 可视化调色板
# -------------------------
def make_glassai_vis_palette():
    return {
        0: (255, 255, 255),  # Background
        1: (255, 255, 0),    # Yellow
        2: (255, 0, 0),      # Red
        3: (0, 255, 255),    # Cyan
        4: (0, 0, 255),      # Blue
        5: (0, 255, 0),      # Green
        255: (0, 0, 0),      # Ignore
    }


def decode_mask_to_rgb(index_mask: np.ndarray, palette_idx_to_rgb: dict, unknown_color=(255, 0, 255)):
    h, w = index_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    valid_ids = []

    for idx, color in palette_idx_to_rgb.items():
        rgb[index_mask == int(idx)] = color
        valid_ids.append(int(idx))

    valid_ids = np.array(valid_ids)
    rgb[~np.isin(index_mask, valid_ids)] = unknown_color

    return rgb


def overlay_multiclass_mask_on_image(img_pil, index_mask: np.ndarray, palette_idx_to_rgb: dict, alpha=0.45):
    """
    多类别彩色 overlay。
    ignore_index=255 不叠加。
    """
    img_np = np.array(img_pil.convert("RGB")).astype(np.float32)
    out = img_np.copy()

    for cls_id, color in palette_idx_to_rgb.items():
        cls_id = int(cls_id)

        if cls_id == 255:
            continue

        mask = index_mask == cls_id

        if not np.any(mask):
            continue

        color_np = np.array(color, dtype=np.float32)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color_np

    return out.astype(np.uint8)


# -------------------------
# 自动选择 checkpoint
# -------------------------
def resolve_checkpoint_path(cfg, dataset_name: str):
    ckpt_dir = str(cfg.get("ckpt_dir", "checkpoints"))

    preferred_best = os.path.join(
        ckpt_dir,
        f"{dataset_name}_6class_pretrained_frozen_best.pth",
    )

    preferred_last = os.path.join(
        ckpt_dir,
        f"{dataset_name}_6class_pretrained_frozen_last.pth",
    )

    eval_ckpt = str(cfg.get("eval_ckpt", "best")).strip()

    if eval_ckpt and eval_ckpt not in ("best", "auto", "last", "final"):
        if os.path.exists(eval_ckpt):
            return eval_ckpt

        if not os.path.isabs(eval_ckpt):
            candidate = os.path.join(ckpt_dir, eval_ckpt)
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            f"[ERROR] eval_ckpt was set to '{eval_ckpt}', but file was not found."
        )

    if eval_ckpt in ("best", "auto") and os.path.exists(preferred_best):
        return preferred_best

    if eval_ckpt in ("last", "final") and os.path.exists(preferred_last):
        return preferred_last

    if os.path.exists(preferred_best):
        return preferred_best

    if os.path.exists(preferred_last):
        return preferred_last

    raise FileNotFoundError(
        "[ERROR] Cannot find GlassAI 6-class checkpoint.\n"
        f"Expected best: {preferred_best}\n"
        f"Expected last: {preferred_last}\n"
        "Please run the GlassAI train.py first, or set eval_ckpt to a valid path."
    )


# -------------------------
# 提取模型 state_dict
# -------------------------
def extract_model_state_dict(state):
    if not isinstance(state, dict):
        raise TypeError(f"[ERROR] checkpoint type error: {type(state)}")

    if "model" in state and isinstance(state["model"], dict):
        return state["model"], "checkpoint_dict:model"

    if "raw_model" in state and isinstance(state["raw_model"], dict):
        return state["raw_model"], "checkpoint_dict:raw_model"

    return state, "state_dict_only"


# -------------------------
# 构建模型
# -------------------------
def build_model(cfg, device):
    vit = CompatibleViTBackbone(cfg).float()

    model = EoMT(
        vit_model=vit,
        num_classes=int(cfg["num_classes"]),
        num_queries=int(cfg.get("num_queries", 16)),
        L1=int(cfg.get("L1", 9)),
        L2=int(cfg.get("L2", 3)),
        joint_query_blocks=int(cfg.get("joint_query_blocks", 1)),
        mask_gate_floor=float(cfg.get("mask_gate_floor", 0.30)),
        aux_return_all=bool(cfg.get("aux_return_all", True)),
        fpn_dim=int(cfg.get("fpn_dim", 256)),
        fpn_layers=int(cfg.get("fpn_layers", 3)),
        out_upsample_x4=bool(cfg.get("out_upsample_x4", True)),
        final_upsample_to_input=bool(cfg.get("final_upsample_to_input", True)),
    )

    model = model.float().to(device)

    return model


# -------------------------
# 保存指标 CSV
# -------------------------
def save_metrics_csv(save_path, metrics):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rows = []

    num_classes = len(metrics["iou"])

    for c in range(num_classes):
        class_name = CLASS_NAMES.get(c, f"Class{c}")

        rows.append(
            {
                "class_id": c,
                "class_name": class_name,
                "tp": int(metrics["tp"][c].item()),
                "fp": int(metrics["fp"][c].item()),
                "fn": int(metrics["fn"][c].item()),
                "tn": int(metrics["tn"][c].item()),
                "support_gt_pixels": int(metrics["support"][c].item()),
                "pred_pixels": int(metrics["pred_count"][c].item()),
                "iou": float(metrics["iou"][c].item()),
                "dice": float(metrics["dice"][c].item()),
                "f1": float(metrics["f1"][c].item()),
                "precision": float(metrics["precision"][c].item()),
                "recall": float(metrics["recall"][c].item()),
                "sensitivity": float(metrics["sensitivity"][c].item()),
                "specificity": float(metrics["specificity"][c].item()),
                "fpr": float(metrics["fpr"][c].item()),
                "fnr": float(metrics["fnr"][c].item()),
                "accuracy_per_class": float(metrics["accuracy_per_class"][c].item()),
                "is_foreground": int(c in metrics["foreground_class_ids"]),
            }
        )

    fieldnames = list(rows[0].keys())

    with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_txt(save_path, metrics, conf_mat, ckpt_path, test_list_file):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    macro = metrics["macro"]
    foreground_class_ids = metrics["foreground_class_ids"]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("GlassAI 6-class segmentation evaluation\n")
        f.write("=" * 80 + "\n")
        f.write(f"Task name: {TASK_NAME}\n")

        for c, name in CLASS_NAMES.items():
            f.write(f"Class {c} = {name}\n")

        f.write("Label 255 = Unknown color / Ignored pixel\n")
        f.write(f"Foreground classes = {foreground_class_ids}\n")
        f.write("Foreground metrics exclude Background class 0.\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Test split: {test_list_file}\n")
        f.write("\n")

        f.write("Confusion matrix, rows=GT, cols=Pred\n")
        f.write(str(conf_mat.numpy()))
        f.write("\n\n")

        f.write("Macro / global metrics\n")
        f.write("-" * 80 + "\n")
        for k, v in macro.items():
            f.write(f"{k}: {v:.6f}\n")

        f.write("\nPer-class metrics\n")
        f.write("-" * 80 + "\n")

        for c in range(len(metrics["iou"])):
            name = CLASS_NAMES.get(c, f"Class{c}")
            f.write(
                f"Class {c} ({name}): "
                f"IoU={float(metrics['iou'][c]):.6f}, "
                f"Dice={float(metrics['dice'][c]):.6f}, "
                f"F1={float(metrics['f1'][c]):.6f}, "
                f"Precision={float(metrics['precision'][c]):.6f}, "
                f"Recall={float(metrics['recall'][c]):.6f}, "
                f"Sensitivity={float(metrics['sensitivity'][c]):.6f}, "
                f"Specificity={float(metrics['specificity'][c]):.6f}, "
                f"FPR={float(metrics['fpr'][c]):.6f}, "
                f"FNR={float(metrics['fnr'][c]):.6f}, "
                f"GT_pixels={int(metrics['support'][c])}, "
                f"Pred_pixels={int(metrics['pred_count'][c])}, "
                f"IsForeground={int(c in foreground_class_ids)}\n"
            )


def main():
    cfg = load_yaml_any_encoding("configs/train_config_glassai.yaml")
    cfg = enforce_glassai_protocol(cfg)

    cfg["lr"] = float(cfg.get("lr", 0.0))
    cfg["load_coach"] = False
    cfg.setdefault("eval_amp", True)
    cfg.setdefault("allow_tf32", True)
    cfg.setdefault("tqdm_auto_disable_non_tty", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_fast_runtime(device, cfg)

    print("=" * 100)
    print("[INFO] Starting evaluation for pretrained-backbone FULL-FREEZE ablation")
    print("[INFO] Task protocol:")
    for c, name in CLASS_NAMES.items():
        print(f"       Class {c} = {name}")
    print("       Label 255 = Unknown color / Ignored pixel")
    print("[INFO] Pixels with label=255 are ignored in all metrics.")
    print("[INFO] Foreground metrics are calculated over classes 1..5, excluding Background=0.")
    print("=" * 100)

    print(f"[INFO] Using device: {device}")

    dataset_name = str(cfg.get("dataset_name", "glassai")).lower()
    splits_root = str(cfg.get("splits_root", "splits"))
    splits_dir = os.path.join(splits_root, dataset_name)

    test_list_file = os.path.join(splits_dir, "test.txt")

    if not os.path.exists(test_list_file):
        raise FileNotFoundError(
            f"[ERROR] Cannot find test split file: {test_list_file}\n"
            f"Please run train.py first to generate splits."
        )

    img_list, lbl_list = read_split_list(test_list_file)

    print(f"[INFO] Loaded test split: {len(img_list)} samples")
    print(f"[INFO] Test split file: {test_list_file}")

    dataset, original_palette, dataset_name = build_dataset_from_cfg(cfg, img_list, lbl_list)

    bs = int(cfg.get("eval_batch_size", 1))
    default_workers = min(4, max(0, (os.cpu_count() or 1) - 1))
    num_workers = int(cfg.get("num_workers", default_workers))
    pin_memory = bool(cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(cfg.get("persistent_workers", True)) and num_workers > 0

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    num_classes = int(cfg["num_classes"])
    ignore_index = int(cfg.get("ignore_index", 255))

    print(f"[INFO] dataset_name={dataset_name}")
    print(f"[INFO] num_classes={num_classes}")
    print(f"[INFO] ignore_index={ignore_index}")
    print(f"[INFO] eval_batch_size={bs}")
    print(f"[INFO] DataLoader workers={num_workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}")

    if num_classes != 6:
        raise ValueError(
            f"[ERROR] This evaluate.py requires num_classes=6, got {num_classes}."
        )

    # 前景类别，默认 1..5
    foreground_class_ids = cfg.get("foreground_class_ids", [1, 2, 3, 4, 5])

    if isinstance(foreground_class_ids, str):
        foreground_class_ids = [
            int(x.strip())
            for x in foreground_class_ids.split(",")
            if x.strip()
        ]

    foreground_class_ids = [
        int(c) for c in foreground_class_ids
        if 0 <= int(c) < num_classes
    ]

    if len(foreground_class_ids) == 0:
        foreground_class_ids = [1, 2, 3, 4, 5]

    print(f"[INFO] Foreground classes for foreground Dice/IoU: {foreground_class_ids}")

    model = build_model(cfg, device)

    ckpt_path = resolve_checkpoint_path(cfg, dataset_name)

    print("[INFO] Loading checkpoint:")
    print(f"       {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict):
        ckpt_num_classes = state.get("num_classes", None)
        ckpt_ignore_index = state.get("ignore_index", None)
        ckpt_task_name = state.get("task_name", None)
        ckpt_ablation_mode = state.get("ablation_mode", None)

        print(f"[INFO] checkpoint task_name={ckpt_task_name}")
        print(f"[INFO] checkpoint ablation_mode={ckpt_ablation_mode}")
        print(f"[INFO] checkpoint num_classes={ckpt_num_classes}")
        print(f"[INFO] checkpoint ignore_index={ckpt_ignore_index}")

        if ckpt_ablation_mode not in (None, ABLATION_MODE):
            print(
                f"[WARN] checkpoint ablation_mode={ckpt_ablation_mode}, "
                f"expected {ABLATION_MODE}."
            )

        if ckpt_num_classes is not None and int(ckpt_num_classes) != 6:
            raise RuntimeError(
                f"[ERROR] This checkpoint has num_classes={ckpt_num_classes}, "
                f"but current evaluation requires num_classes=6. "
                f"Do not evaluate an incompatible checkpoint with this script."
            )

    state_dict, ckpt_style = extract_model_state_dict(state)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[INFO] checkpoint format = {ckpt_style}")

    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {len(missing)}")
        for k in missing[:20]:
            print(f"       missing: {k}")
        if len(missing) > 20:
            print("       ...")

    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {len(unexpected)}")
        for k in unexpected[:20]:
            print(f"       unexpected: {k}")
        if len(unexpected) > 20:
            print("       ...")

    model.eval()

    use_amp = bool(cfg.get("eval_amp", True)) and device.type == "cuda"
    amp_dtype = pick_amp_dtype()

    print(f"[INFO] AMP evaluation: {'enabled' if use_amp else 'disabled'}, dtype={amp_dtype}")

    enable_vis = bool(cfg.get("eval_vis_enable", True))
    vis_max = int(cfg.get("eval_vis_max", 5))
    vis_dir = str(cfg.get("eval_vis_dir", f"eval_vis_{dataset_name}_{TASK_NAME}"))

    if enable_vis:
        os.makedirs(vis_dir, exist_ok=True)
        print(f"[INFO] Visualization enabled. Save dir: {vis_dir}, max samples: {vis_max}")
    else:
        print("[INFO] Visualization disabled.")

    result_dir = os.path.join("eval_results", dataset_name, TASK_NAME)
    os.makedirs(result_dir, exist_ok=True)

    print(f"[INFO] Evaluation result dir: {result_dir}")

    vis_palette = make_glassai_vis_palette()

    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    saved = 0
    total_pixels = 0
    valid_pixels = 0
    ignored_pixels = 0

    with torch.inference_mode():
        prog = make_progress(
            loader, "Evaluating", cfg=cfg, leave=False, total=len(loader)
        )

        for batch in prog:
            img, label = batch[0], batch[1]

            img = img.to(device, non_blocking=True)
            label_cpu = label.long()

            total_pixels += int(label_cpu.numel())
            valid_pixels += int((label_cpu != ignore_index).sum().item())
            ignored_pixels += int((label_cpu == ignore_index).sum().item())

            with autocast_context(device, use_amp, amp_dtype):
                out = model(img)
                masks = out[0]

                if masks.shape[-2:] != label_cpu.shape[-2:]:
                    masks_up = F.interpolate(
                        masks,
                        size=label_cpu.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    masks_up = masks

            pred = masks_up.argmax(dim=1)
            label_device = label_cpu.to(device, non_blocking=True)
            update_confusion_matrix(
                conf_mat=conf_mat,
                pred=pred,
                target=label_device,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
            pred_cpu = pred.detach().cpu().long() if enable_vis and saved < vis_max else None

            if enable_vis and saved < vis_max:
                for i in range(pred_cpu.size(0)):
                    if saved >= vis_max:
                        break

                    img_pil = TF.to_pil_image(img[i].detach().cpu())

                    gt_np = label_cpu[i].numpy()
                    pr_np = pred_cpu[i].numpy()

                    vis_alpha = float(cfg.get("alpha", 0.45))

                    gt_rgb = decode_mask_to_rgb(gt_np, vis_palette)
                    pred_rgb = decode_mask_to_rgb(pr_np, vis_palette)

                    gt_overlay = overlay_multiclass_mask_on_image(
                        img_pil,
                        gt_np,
                        palette_idx_to_rgb=vis_palette,
                        alpha=vis_alpha,
                    )

                    pred_overlay = overlay_multiclass_mask_on_image(
                        img_pil,
                        pr_np,
                        palette_idx_to_rgb=vis_palette,
                        alpha=vis_alpha,
                    )

                    fig, axs = plt.subplots(1, 5, figsize=(22, 4))

                    axs[0].imshow(img_pil)
                    axs[0].set_title("Image")

                    axs[1].imshow(gt_rgb)
                    axs[1].set_title("GT mask")

                    axs[2].imshow(pred_rgb)
                    axs[2].set_title("Pred mask")

                    axs[3].imshow(gt_overlay)
                    axs[3].set_title(f"GT overlay\nalpha={vis_alpha}")

                    axs[4].imshow(pred_overlay)
                    axs[4].set_title(f"Pred overlay\nalpha={vis_alpha}")

                    for ax in axs:
                        ax.axis("off")

                    plt.tight_layout()

                    save_path = os.path.join(vis_dir, f"sample_{saved + 1:03d}.png")
                    plt.savefig(save_path, dpi=150)
                    plt.close()

                    saved += 1

            del masks, masks_up

    conf_mat = conf_mat.cpu()
    metrics = compute_metrics_from_confmat(
        conf_mat,
        foreground_class_ids=foreground_class_ids,
    )

    macro = metrics["macro"]

    valid_ratio = valid_pixels / max(1, total_pixels)
    ignored_ratio = ignored_pixels / max(1, total_pixels)

    print("\n" + "=" * 100)
    print("[Result] GlassAI 6-class segmentation metrics")
    print("=" * 100)

    print("[INFO] Confusion matrix, rows=GT, cols=Pred")
    print(conf_mat.numpy())

    print("\n[INFO] Pixel statistics")
    print(f"       total_pixels   = {total_pixels}")
    print(f"       valid_pixels   = {valid_pixels}")
    print(f"       ignored_pixels = {ignored_pixels}")
    print(f"       valid_ratio    = {valid_ratio:.6f}")
    print(f"       ignored_ratio  = {ignored_ratio:.6f}")

    print("\n[Per-class metrics]")
    for c in range(num_classes):
        name = CLASS_NAMES.get(c, f"Class{c}")

        print(
            f"Class {c:02d} ({name:10s}) | "
            f"IoU={float(metrics['iou'][c]):.4f} | "
            f"Dice={float(metrics['dice'][c]):.4f} | "
            f"F1={float(metrics['f1'][c]):.4f} | "
            f"Precision={float(metrics['precision'][c]):.4f} | "
            f"Recall={float(metrics['recall'][c]):.4f} | "
            f"Sensitivity={float(metrics['sensitivity'][c]):.4f} | "
            f"Specificity={float(metrics['specificity'][c]):.4f} | "
            f"FPR={float(metrics['fpr'][c]):.4f} | "
            f"FNR={float(metrics['fnr'][c]):.4f} | "
            f"Acc={float(metrics['accuracy_per_class'][c]):.4f} | "
            f"GT_pixels={int(metrics['support'][c])} | "
            f"Pred_pixels={int(metrics['pred_count'][c])} | "
            f"Foreground={int(c in foreground_class_ids)}"
        )

    print("\n[Macro / global metrics]")
    print(f"Mean IoU              = {macro['mean_iou']:.4f}")
    print(f"Mean Dice             = {macro['mean_dice']:.4f}")
    print(f"Mean F1               = {macro['mean_f1']:.4f}")
    print(f"Mean Precision        = {macro['mean_precision']:.4f}")
    print(f"Mean Recall           = {macro['mean_recall']:.4f}")
    print(f"Mean Sensitivity      = {macro['mean_sensitivity']:.4f}")
    print(f"Mean Specificity      = {macro['mean_specificity']:.4f}")
    print(f"Mean Accuracy         = {macro['mean_accuracy']:.4f}")
    print(f"Overall Accuracy      = {macro['overall_accuracy']:.4f}")
    print(f"FWIoU                 = {macro['fw_iou']:.4f}")

    print("\n[Foreground metrics]")
    print(f"Foreground classes    = {foreground_class_ids}")
    print(f"Foreground Mean IoU   = {macro['foreground_mean_iou']:.4f}")
    print(f"Foreground Mean Dice  = {macro['foreground_mean_dice']:.4f}")
    print(f"Foreground Mean F1    = {macro['foreground_mean_f1']:.4f}")
    print(f"Foreground Precision  = {macro['foreground_mean_precision']:.4f}")
    print(f"Foreground Recall     = {macro['foreground_mean_recall']:.4f}")

    csv_path = os.path.join(result_dir, "metrics_per_class.csv")
    txt_path = os.path.join(result_dir, "summary.txt")
    conf_path = os.path.join(result_dir, "confusion_matrix.npy")

    save_metrics_csv(csv_path, metrics)
    save_summary_txt(txt_path, metrics, conf_mat, ckpt_path, test_list_file)
    np.save(conf_path, conf_mat.numpy())

    print("\n[INFO] Saved evaluation files:")
    print(f"       per-class metrics CSV: {csv_path}")
    print(f"       summary TXT:           {txt_path}")
    print(f"       confusion matrix NPY:  {conf_path}")

    if enable_vis:
        print(f"       visualization dir:     {vis_dir}")

    print("=" * 100)


if __name__ == "__main__":
    main()