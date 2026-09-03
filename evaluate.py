import os
import yaml
import csv
import contextlib

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 可视化
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from models.vit_backbone import CompatibleViTBackbone
from models.eomt_head import EoMT


CLASS_NAMES = {
    0: "Tumor",
    1: "Normal",
}

TASK_NAME = "2class_ignore_other"


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
# 强制使用方案 A
# -------------------------
def enforce_2class_ignore_protocol(cfg):
    old_num_classes = int(cfg.get("num_classes", 2))
    old_ignore_index = int(cfg.get("ignore_index", 255))

    if old_num_classes != 2:
        print(
            f"[WARN] config num_classes={old_num_classes}, "
            f"but this evaluate.py is for 2-class ignore-other protocol. "
            f"Override num_classes -> 2."
        )

    if old_ignore_index != 255:
        print(
            f"[WARN] config ignore_index={old_ignore_index}, "
            f"but this evaluate.py is for ignore_index=255. "
            f"Override ignore_index -> 255."
        )

    cfg["num_classes"] = 2
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


# -------------------------
# 从 cfg 构建 dataset
# -------------------------
def build_dataset_from_cfg(cfg, img_list, lbl_list):
    dataset_name = str(cfg.get("dataset_name", "lungcancer")).lower()

    resize_hw = cfg.get("resize_hw", None)

    if isinstance(resize_hw, list) and len(resize_hw) == 2:
        resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
    else:
        resize_hw = None

    ignore_index = int(cfg.get("ignore_index", 255))

    if dataset_name == "glassai":
        from datasets.glassai_dataset import GlassAIDataset, FIXED_PALETTE

        try:
            ds = GlassAIDataset(
                image_paths=img_list,
                label_paths=lbl_list,
                palette=FIXED_PALETTE,
                resize_hw=resize_hw,
                ignore_index=ignore_index,
            )
        except TypeError:
            ds = GlassAIDataset(img_list, lbl_list, palette=FIXED_PALETTE)

        return ds, FIXED_PALETTE, dataset_name

    if dataset_name == "lungcancer":
        from datasets.lungcancer_dataset import LungCancerDataset, FIXED_PALETTE

        try:
            ds = LungCancerDataset(
                image_paths=img_list,
                label_paths=lbl_list,
                palette=FIXED_PALETTE,
                resize_hw=resize_hw,
                ignore_index=ignore_index,
            )
        except TypeError:
            ds = LungCancerDataset(img_list, lbl_list, palette=FIXED_PALETTE)

        return ds, FIXED_PALETTE, dataset_name

    raise ValueError(f"Unsupported dataset_name: {dataset_name}. Use glassai or lungcancer.")


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
    """
    conf_mat[target, pred]
    pred/target: CPU LongTensor, shape=(H, W)
    """
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)

    if valid.sum().item() == 0:
        return conf_mat

    t = target[valid].view(-1)
    p = pred[valid].view(-1)

    valid_p = (p >= 0) & (p < num_classes)
    t = t[valid_p]
    p = p[valid_p]

    if t.numel() == 0:
        return conf_mat

    idx = (t * num_classes + p).to(torch.int64)
    bins = torch.bincount(idx, minlength=num_classes * num_classes)
    conf_mat += bins.view(num_classes, num_classes)

    return conf_mat


# -------------------------
# 计算常见分割指标
# -------------------------
def compute_metrics_from_confmat(conf_mat: torch.Tensor, eps=1e-12):
    conf = conf_mat.to(torch.float64)

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
    }


# -------------------------
# 可视化调色板
# -------------------------
def make_2class_vis_palette():
    return {
        0: (128, 0, 0),      # Tumor
        1: (0, 130, 200),    # Normal
        255: (0, 0, 0),      # Ignore / Other
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


def make_binary_mask_rgb(index_mask: np.ndarray, class_id=0, color=(255, 0, 0)):
    """
    只显示指定类别的二值 mask。
    class_id 区域显示为 color，其余区域为黑色。
    默认 class_id=0，即 Tumor。
    """
    h, w = index_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[index_mask == class_id] = color
    return rgb


def overlay_mask_on_image(img_pil, index_mask: np.ndarray, class_id=0, color=(255, 0, 0), alpha=0.45):
    """
    在原图上叠加指定类别的半透明蒙版。
    alpha 越大，红色蒙版越明显；alpha 越小，原图越清晰。
    默认只叠加 Tumor=0 区域。
    """
    img_np = np.array(img_pil.convert("RGB")).astype(np.float32)

    mask = index_mask == class_id
    overlay_color = np.array(color, dtype=np.float32)

    img_np[mask] = (1 - alpha) * img_np[mask] + alpha * overlay_color

    return img_np.astype(np.uint8)


# -------------------------
# 自动选择 checkpoint
# -------------------------
def resolve_checkpoint_path(cfg, dataset_name: str):
    ckpt_dir = str(cfg.get("ckpt_dir", "checkpoints"))

    preferred_best = os.path.join(
        ckpt_dir,
        f"{dataset_name}_2class_ignore_other_best.pth",
    )

    preferred_last = os.path.join(
        ckpt_dir,
        f"{dataset_name}_2class_ignore_other_last.pth",
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
        "[ERROR] Cannot find 2-class ignore-other checkpoint.\n"
        f"Expected best: {preferred_best}\n"
        f"Expected last: {preferred_last}\n"
        "Please run the modified train.py first, or set eval_ckpt to a valid path."
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
        aux_return_all=True,
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

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("2-class ignore-other segmentation evaluation\n")
        f.write("=" * 80 + "\n")
        f.write(f"Task name: {TASK_NAME}\n")
        f.write("Class 0 = Tumor\n")
        f.write("Class 1 = Normal\n")
        f.write("Label 255 = Other / Background / Ignored region\n")
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
                f"Pred_pixels={int(metrics['pred_count'][c])}\n"
            )


def main():
    cfg = load_yaml_any_encoding("configs/train_config.yaml")
    cfg = enforce_2class_ignore_protocol(cfg)

    cfg["lr"] = float(cfg.get("lr", 0.0))
    cfg["load_coach"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("[INFO] Starting evaluation with 2-class ignore-other protocol")
    print("[INFO] Task protocol:")
    print("       Class 0 = Tumor")
    print("       Class 1 = Normal")
    print("       Label 255 = Other / Background / Ignored region")
    print("[INFO] Pixels with label=255 are ignored in all metrics.")
    print("=" * 100)

    print(f"[INFO] Using device: {device}")

    dataset_name = str(cfg.get("dataset_name", "lungcancer")).lower()
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
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    num_classes = int(cfg["num_classes"])
    ignore_index = int(cfg.get("ignore_index", 255))

    print(f"[INFO] dataset_name={dataset_name}")
    print(f"[INFO] num_classes={num_classes}")
    print(f"[INFO] ignore_index={ignore_index}")
    print(f"[INFO] eval_batch_size={bs}")

    if num_classes != 2:
        raise ValueError(
            f"[ERROR] This evaluate.py requires num_classes=2, got {num_classes}."
        )

    model = build_model(cfg, device)

    ckpt_path = resolve_checkpoint_path(cfg, dataset_name)

    print("[INFO] Loading checkpoint:")
    print(f"       {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict):
        ckpt_num_classes = state.get("num_classes", None)
        ckpt_ignore_index = state.get("ignore_index", None)
        ckpt_task_name = state.get("task_name", None)

        print(f"[INFO] checkpoint task_name={ckpt_task_name}")
        print(f"[INFO] checkpoint num_classes={ckpt_num_classes}")
        print(f"[INFO] checkpoint ignore_index={ckpt_ignore_index}")

        if ckpt_num_classes is not None and int(ckpt_num_classes) != 2:
            raise RuntimeError(
                f"[ERROR] This checkpoint has num_classes={ckpt_num_classes}, "
                f"but current evaluation requires num_classes=2. "
                f"Do not evaluate an old 3-class checkpoint with this script."
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

    vis_palette = make_2class_vis_palette()

    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    saved = 0
    total_pixels = 0
    valid_pixels = 0
    ignored_pixels = 0

    with torch.inference_mode():
        prog = tqdm(loader, desc="Evaluating", ncols=120)

        for batch in prog:
            img, label = batch[0], batch[1]

            img = img.to(device, non_blocking=True)
            label_cpu = label.long()

            total_pixels += int(label_cpu.numel())
            valid_pixels += int((label_cpu != ignore_index).sum().item())
            ignored_pixels += int((label_cpu == ignore_index).sum().item())

            ctx = (
                torch.cuda.amp.autocast(dtype=amp_dtype)
                if use_amp
                else contextlib.nullcontext()
            )

            with ctx:
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

            pred_cpu = masks_up.argmax(dim=1).detach().cpu().long()

            for i in range(pred_cpu.size(0)):
                conf_mat = update_confusion_matrix(
                    conf_mat=conf_mat,
                    pred=pred_cpu[i],
                    target=label_cpu[i],
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                )

            if enable_vis and saved < vis_max:
                for i in range(pred_cpu.size(0)):
                    if saved >= vis_max:
                        break

                    img_pil = TF.to_pil_image(img[i].detach().cpu())

                    gt_np = label_cpu[i].numpy()
                    pr_np = pred_cpu[i].numpy()

                    # -------------------------------------------------
                    # 新版可视化：
                    # Image | GT | Pred | GT overlay | Pred overlay
                    #
                    # 默认只把 Tumor=0 区域作为红色半透明蒙版叠加。
                    # alpha 可在 configs/train_config.yaml 中设置：
                    # alpha: 0.45
                    #
                    # 如果你想叠加 Normal=1，可在配置里加：
                    # vis_overlay_class_id: 1
                    # -------------------------------------------------
                    vis_alpha = float(cfg.get("alpha", 0.45))
                    overlay_class_id = int(cfg.get("vis_overlay_class_id", 0))
                    overlay_color = (255, 0, 0)

                    gt_mask_rgb = make_binary_mask_rgb(
                        gt_np,
                        class_id=overlay_class_id,
                        color=overlay_color,
                    )

                    pred_mask_rgb = make_binary_mask_rgb(
                        pr_np,
                        class_id=overlay_class_id,
                        color=overlay_color,
                    )

                    gt_overlay = overlay_mask_on_image(
                        img_pil,
                        gt_np,
                        class_id=overlay_class_id,
                        color=overlay_color,
                        alpha=vis_alpha,
                    )

                    pred_overlay = overlay_mask_on_image(
                        img_pil,
                        pr_np,
                        class_id=overlay_class_id,
                        color=overlay_color,
                        alpha=vis_alpha,
                    )

                    fig, axs = plt.subplots(1, 5, figsize=(22, 4))

                    axs[0].imshow(img_pil)
                    axs[0].set_title("Image")

                    axs[1].imshow(gt_mask_rgb)
                    axs[1].set_title("GT")

                    axs[2].imshow(pred_mask_rgb)
                    axs[2].set_title("Pred")

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

            if device.type == "cuda":
                torch.cuda.empty_cache()

    metrics = compute_metrics_from_confmat(conf_mat)
    macro = metrics["macro"]

    valid_ratio = valid_pixels / max(1, total_pixels)
    ignored_ratio = ignored_pixels / max(1, total_pixels)

    print("\n" + "=" * 100)
    print("[Result] 2-class ignore-other segmentation metrics")
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
            f"Class {c:02d} ({name:6s}) | "
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
            f"Pred_pixels={int(metrics['pred_count'][c])}"
        )

    print("\n[Macro / global metrics]")
    print(f"Mean IoU          = {macro['mean_iou']:.4f}")
    print(f"Mean Dice         = {macro['mean_dice']:.4f}")
    print(f"Mean F1           = {macro['mean_f1']:.4f}")
    print(f"Mean Precision    = {macro['mean_precision']:.4f}")
    print(f"Mean Recall       = {macro['mean_recall']:.4f}")
    print(f"Mean Sensitivity  = {macro['mean_sensitivity']:.4f}")
    print(f"Mean Specificity  = {macro['mean_specificity']:.4f}")
    print(f"Mean Accuracy     = {macro['mean_accuracy']:.4f}")
    print(f"Overall Accuracy  = {macro['overall_accuracy']:.4f}")
    print(f"FWIoU             = {macro['fw_iou']:.4f}")

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