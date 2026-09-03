import os
import csv
import yaml
import contextlib
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 可视化（可选）
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from models.vit_backbone import CompatibleViTBackbone
from models.eomt_head import EoMT
from datasets.glas_dataset import GlasDataset


# -------------------------
# YAML 读取（兼容不同编码）
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
# 混淆矩阵更新
# conf_mat 行=GT，列=Pred
# -------------------------
@torch.inference_mode()
def update_confusion_matrix(conf_mat, pred, target, num_classes, ignore_index=255, ignore_background_pixels=False):
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)

    if ignore_background_pixels:
        valid = valid & (target != 0)

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
# 由混淆矩阵计算 IoU / F1
# -------------------------
def compute_iou_f1_from_confmat(conf_mat: torch.Tensor):
    conf = conf_mat.to(torch.float64)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp

    denom_iou = tp + fp + fn
    iou = torch.where(denom_iou > 0, tp / (denom_iou + 1e-12), torch.zeros_like(tp))

    denom_f1 = 2 * tp + fp + fn
    f1 = torch.where(denom_f1 > 0, (2 * tp) / (denom_f1 + 1e-12), torch.zeros_like(tp))

    return iou, f1


# -------------------------
# GlaS 可视化：0背景黑，1前景绿
# -------------------------
def decode_glas_mask_to_rgb(index_mask: np.ndarray):
    h, w = index_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[index_mask == 0] = [0, 0, 0]
    rgb[index_mask == 1] = [0, 255, 0]
    return rgb


# -------------------------
# 选择 checkpoint
# 默认优先 best，其次最后 epoch
# -------------------------
def resolve_checkpoint_path(cfg, dataset_name: str):
    ckpt_dir = str(cfg.get("ckpt_dir", "checkpoints"))
    epochs = int(cfg["epochs"])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir_abs = os.path.join(script_dir, ckpt_dir)

    ckpt_preference = str(cfg.get("eval_ckpt", "best")).lower()
    best_ckpt = os.path.join(ckpt_dir_abs, f"{dataset_name}_best.pth")
    final_ckpt = os.path.join(ckpt_dir_abs, f"{dataset_name}_epoch_{epochs}.pth")

    if ckpt_preference in ("best", "auto") and os.path.exists(best_ckpt):
        return best_ckpt
    if ckpt_preference in ("last", "final") and os.path.exists(final_ckpt):
        return final_ckpt
    if os.path.exists(best_ckpt):
        return best_ckpt
    if os.path.exists(final_ckpt):
        return final_ckpt

    raise FileNotFoundError(
        f"[ERROR] 找不到 checkpoint：\n  best: {best_ckpt}\n  final: {final_ckpt}"
    )


def _forward_logits_with_optional_tta(model, img, label_hw, use_amp, amp_dtype, tta_mode="none"):
    """
    返回和 label_hw 对齐后的 logits。
    支持:
      - none: 不做 TTA
      - hflip: 原图 + 水平翻转
      - hvflip: 原图 + 水平翻转 + 垂直翻转 + 水平垂直翻转
    """
    device_type = img.device.type
    ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if (use_amp and device_type == "cuda") else contextlib.nullcontext()

    def _predict(x):
        out = model(x)
        logits = out[0]
        if logits.shape[-2:] != label_hw:
            logits = F.interpolate(logits, size=label_hw, mode="bilinear", align_corners=False)
        return logits

    with ctx:
        logits_sum = _predict(img).float()
        n = 1

        if tta_mode in ("hflip", "hvflip"):
            logits_h = _predict(torch.flip(img, dims=[3])).float()
            logits_sum += torch.flip(logits_h, dims=[3])
            n += 1

        if tta_mode == "hvflip":
            logits_v = _predict(torch.flip(img, dims=[2])).float()
            logits_sum += torch.flip(logits_v, dims=[2])
            n += 1

            logits_hv = _predict(torch.flip(img, dims=[2, 3])).float()
            logits_sum += torch.flip(logits_hv, dims=[2, 3])
            n += 1

    return logits_sum / n


def main():
    cfg = load_yaml_any_encoding("configs/trainconfig_glas.yaml")
    cfg["lr"] = float(cfg["lr"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset_name = str(cfg.get("dataset_name", "glas")).lower()

    # -------------------------
    # Dataset
    # -------------------------
    ds = GlasDataset(
        image_dir=os.path.join(cfg["data_root"], "test", "images"),
        label_dir=os.path.join(cfg["data_root"], "test", "masks"),
        resize_hw=tuple(cfg["resize_hw"]) if isinstance(cfg.get("resize_hw"), list) else cfg.get("resize_hw", None),
        ignore_index=int(cfg.get("ignore_index", 255)),
    )

    bs = int(cfg.get("eval_batch_size", 1))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # -------------------------
    # Model（和训练脚本一致）
    # -------------------------
    num_classes = int(cfg["num_classes"])
    ignore_index = int(cfg.get("ignore_index", 255))

    vit = CompatibleViTBackbone(cfg)
    vit = vit.float()

    model = EoMT(
        vit_model=vit,
        num_classes=num_classes,
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

    ckpt_path = resolve_checkpoint_path(cfg, dataset_name)

    state = torch.load(ckpt_path, map_location=device)

    # 兼容两种 checkpoint:
    # 1. 直接 state_dict
    # 2. {"model_state": state_dict, ...}
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    strict_load = bool(cfg.get("eval_strict_state_dict", True))
    if strict_load:
        model.load_state_dict(state, strict=True)
        missing, unexpected = [], []
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    print(f"[INFO] load_state_dict strict={strict_load} | missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("[WARN] missing examples:", missing[:20])
    if len(unexpected) > 0:
        print("[WARN] unexpected examples:", unexpected[:20])

    model.eval()

    use_amp = bool(cfg.get("eval_amp", True)) and (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    ) else torch.float16

    ignore_background_pixels = bool(cfg.get("ignore_background_pixels", False))
    macro_exclude_bg_class = bool(cfg.get("macro_exclude_bg_class", False))

    eval_tta = str(cfg.get("eval_tta", "none")).lower()
    if eval_tta not in ("none", "hflip", "hvflip"):
        raise ValueError("eval_tta must be one of: none, hflip, hvflip")

    enable_vis = bool(cfg.get("eval_vis_enable", True))
    vis_max = int(cfg.get("eval_vis_max", 5))
    vis_dir = str(cfg.get("eval_vis_dir", f"eval_vis_{dataset_name}"))
    if enable_vis:
        os.makedirs(vis_dir, exist_ok=True)

    print(f"[INFO] num_classes={num_classes}, ignore_index={ignore_index}")
    print(f"[INFO] ignore_background_pixels={ignore_background_pixels}, macro_exclude_bg_class={macro_exclude_bg_class}")
    print(f"[INFO] eval_batch_size={bs}, amp={use_amp}, tta={eval_tta}")
    if enable_vis:
        print(f"[INFO] Visualization enabled: save first {vis_max} samples to {vis_dir}")

    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    saved = 0
    with torch.inference_mode():
        prog = tqdm(loader, desc="Evaluating", ncols=100)
        for batch_idx, batch in enumerate(prog):
            img, label = batch[0], batch[1]
            img = img.to(device, non_blocking=True)
            label_cpu = label.long()

            masks = _forward_logits_with_optional_tta(
                model=model,
                img=img,
                label_hw=label_cpu.shape[-2:],
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                tta_mode=eval_tta,
            )

            pred_cpu = masks.argmax(dim=1).detach().cpu().long()

            for i in range(pred_cpu.size(0)):
                conf_mat = update_confusion_matrix(
                    conf_mat,
                    pred_cpu[i],
                    label_cpu[i],
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    ignore_background_pixels=ignore_background_pixels,
                )

            if enable_vis and saved < vis_max:
                for i in range(pred_cpu.size(0)):
                    if saved >= vis_max:
                        break

                    img_pil = TF.to_pil_image(img[i].detach().cpu())
                    gt_np = label_cpu[i].numpy()
                    pr_np = pred_cpu[i].numpy()

                    gt_rgb = decode_glas_mask_to_rgb(gt_np)
                    pr_rgb = decode_glas_mask_to_rgb(pr_np)

                    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
                    axs[0].imshow(img_pil)
                    axs[0].set_title("Input")
                    axs[1].imshow(gt_rgb)
                    axs[1].set_title("Ground Truth")
                    axs[2].imshow(pr_rgb)
                    axs[2].set_title("Prediction")
                    for ax in axs:
                        ax.axis("off")
                    plt.tight_layout()

                    save_path = os.path.join(vis_dir, f"sample_{saved+1:03d}.png")
                    plt.savefig(save_path, dpi=150)
                    plt.close()
                    saved += 1

            del masks
            if device.type == "cuda":
                torch.cuda.empty_cache()

    per_iou, per_f1 = compute_iou_f1_from_confmat(conf_mat)

    conf = conf_mat.to(torch.float64)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    tn = conf.sum() - (tp + fp + fn)

    per_precision = torch.where(tp + fp > 0, tp / (tp + fp + 1e-12), torch.zeros_like(tp))
    per_recall = torch.where(tp + fn > 0, tp / (tp + fn + 1e-12), torch.zeros_like(tp))
    per_specificity = torch.where(tn + fp > 0, tn / (tn + fp + 1e-12), torch.zeros_like(tp))

    if macro_exclude_bg_class and num_classes > 1:
        macro_iou = float(per_iou[1:].mean().item())
        macro_f1 = float(per_f1[1:].mean().item())
    else:
        macro_iou = float(per_iou.mean().item())
        macro_f1 = float(per_f1.mean().item())

    print("\n====================")
    print("[Result] Confusion-matrix based metrics (IoU & F1)")
    print("====================")

    for c in range(num_classes):
        print(f"Class {c:02d}: IoU={per_iou[c]:.4f} | F1={per_f1[c]:.4f}")

    print("\n--------------------")
    print(f"Macro mIoU = {macro_iou:.4f}  (macro_exclude_bg_class={macro_exclude_bg_class})")
    print(f"Macro F1   = {macro_f1:.4f}  (macro_exclude_bg_class={macro_exclude_bg_class})")

    if num_classes == 2:
        csv_path = str(cfg.get("eval_csv_path", f"{dataset_name}_eval_comparable.csv"))
        row = {
            "model_name": str(cfg.get("model_name", "my_model")),
            "checkpoint": ckpt_path,
            "test_mean_dice": macro_f1,
            "test_mean_iou": macro_iou,
            "background_dice": float(per_f1[0].item()),
            "background_iou": float(per_iou[0].item()),
            "background_precision": float(per_precision[0].item()),
            "background_recall": float(per_recall[0].item()),
            "background_specificity": float(per_specificity[0].item()),
            "gland_dice": float(per_f1[1].item()),
            "gland_iou": float(per_iou[1].item()),
            "gland_precision": float(per_precision[1].item()),
            "gland_recall": float(per_recall[1].item()),
            "gland_specificity": float(per_specificity[1].item()),
            "ignore_background_pixels": ignore_background_pixels,
            "macro_exclude_bg_class": macro_exclude_bg_class,
            "eval_tta": eval_tta,
        }

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        print(f"[INFO] Comparable CSV saved to: {csv_path}")

    print("--------------------")


if __name__ == "__main__":
    main()