import os
import contextlib

import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.vit_backbone import CompatibleViTBackbone
from models.eomt_head import EoMT


def load_yaml_any_encoding(path: str):
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return yaml.safe_load(f)


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
                raise ValueError(f"[ERROR] split line {line_no} format error: {line}")
            if len(parts) != 2:
                raise ValueError(f"[ERROR] split line {line_no} cannot be parsed: {line}")
            img_p, lbl_p = parts[0].strip(), parts[1].strip()
            if not img_p or not lbl_p:
                raise ValueError(f"[ERROR] split line {line_no} contains empty path: {line}")
            img_list.append(img_p)
            lbl_list.append(lbl_p)
    return img_list, lbl_list


def build_dataset_from_cfg(cfg, img_list, lbl_list):
    dataset_name = str(cfg.get("dataset_name", "glassai")).lower()
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
        return ds, dataset_name

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
        return ds, dataset_name

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


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
    ).float().to(device)
    return model


def resolve_checkpoint_path(cfg, dataset_name: str):
    ckpt_dir = str(cfg.get("ckpt_dir", "checkpoints"))
    pref = str(cfg.get("eval_ckpt", "best")).lower()
    best_ckpt = os.path.join(ckpt_dir, f"{dataset_name}_best_mean_dice.pth")
    last_ckpt = os.path.join(ckpt_dir, str(cfg.get("last_ckpt_name", "last.pth")))

    if pref in ("best", "auto") and os.path.exists(best_ckpt):
        return best_ckpt
    if pref in ("last", "final") and os.path.exists(last_ckpt):
        return last_ckpt
    if os.path.exists(best_ckpt):
        return best_ckpt
    if os.path.exists(last_ckpt):
        return last_ckpt
    raise FileNotFoundError(f"[ERROR] checkpoint not found:\n  best: {best_ckpt}\n  last: {last_ckpt}")


def extract_model_state_dict(state):
    if not isinstance(state, dict):
        raise TypeError(f"[ERROR] invalid checkpoint type: {type(state)}")
    if "model" in state and isinstance(state["model"], dict):
        return state["model"], "checkpoint_model"
    return state, "state_dict_only"


@torch.inference_mode()
def update_confusion_matrix(conf_mat, pred, target, num_classes, ignore_index=255):
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


def compute_dice_iou_from_confmat(conf_mat: torch.Tensor):
    conf = conf_mat.to(torch.float64)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    dice = torch.where(2 * tp + fp + fn > 0, (2 * tp) / (2 * tp + fp + fn + 1e-8), torch.zeros_like(tp))
    iou = torch.where(tp + fp + fn > 0, tp / (tp + fp + fn + 1e-8), torch.zeros_like(tp))
    return dice, iou


def forward_logits_with_tta(model, img, out_size, cfg):
    logits_list = []

    out = model(img)
    masks = out[0]
    logits_list.append(F.interpolate(masks, size=out_size, mode="bilinear", align_corners=False))

    if bool(cfg.get("eval_tta_hflip", True)):
        img_f = torch.flip(img, dims=(-1,))
        masks_f = model(img_f)[0]
        masks_f = F.interpolate(masks_f, size=out_size, mode="bilinear", align_corners=False)
        logits_list.append(torch.flip(masks_f, dims=(-1,)))

    if bool(cfg.get("eval_tta_vflip", True)):
        img_f = torch.flip(img, dims=(-2,))
        masks_f = model(img_f)[0]
        masks_f = F.interpolate(masks_f, size=out_size, mode="bilinear", align_corners=False)
        logits_list.append(torch.flip(masks_f, dims=(-2,)))

    return torch.stack(logits_list, dim=0).mean(dim=0)


def main():
    cfg = load_yaml_any_encoding("configs/train_config.yaml")
    cfg["lr"] = float(cfg["lr"])
    cfg["load_coach"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset_name = str(cfg.get("dataset_name", "glassai")).lower()
    splits_dir = os.path.join(str(cfg.get("splits_root", "splits")), dataset_name)
    test_list_file = os.path.join(splits_dir, "test.txt")
    if not os.path.exists(test_list_file):
        raise FileNotFoundError(f"[ERROR] test split file not found: {test_list_file}. Run training first.")

    img_list, lbl_list = read_split_list(test_list_file)
    dataset, dataset_name = build_dataset_from_cfg(cfg, img_list, lbl_list)
    print(f"[INFO] Loaded test split: {len(dataset)} samples from {test_list_file}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.get("eval_batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_memory", False)),
        drop_last=False,
    )

    num_classes = int(cfg["num_classes"])
    ignore_index = int(cfg.get("ignore_index", 255))
    model = build_model(cfg, device)

    ckpt_path = resolve_checkpoint_path(cfg, dataset_name)
    state = torch.load(ckpt_path, map_location=device)
    state_dict, ckpt_style = extract_model_state_dict(state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    print(f"[INFO] checkpoint format = {ckpt_style}")
    if isinstance(state, dict):
        print(f"[INFO] checkpoint epoch = {state.get('epoch', 'NA')}")
        print(f"[INFO] best_metric = {state.get('best_metric', 'NA')}")
        print(f"[INFO] best_metric_name = {state.get('best_metric_name', 'NA')}")
        print(f"[INFO] ema_saved_as_model = {state.get('ema_saved_as_model', 'NA')}")
    print(f"[INFO] load_state_dict strict=False | missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[WARN] missing examples:", missing[:20])
    if unexpected:
        print("[WARN] unexpected examples:", unexpected[:20])

    model.eval()
    use_amp = bool(cfg.get("eval_amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    ) else torch.float16

    print(f"[INFO] num_classes={num_classes}, ignore_index={ignore_index}, eval_amp={use_amp}")
    print(f"[INFO] TTA: hflip={bool(cfg.get('eval_tta_hflip', True))}, vflip={bool(cfg.get('eval_tta_vflip', True))}")
    print("[INFO] metric protocol: include background for mean; also report foreground mean")

    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.inference_mode():
        prog = tqdm(loader, desc="Evaluating", ncols=100)
        for batch in prog:
            img, label = batch[0], batch[1]
            img = img.to(device, non_blocking=True)
            label_cpu = label.long()
            out_size = label_cpu.shape[-2:]

            ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if use_amp else contextlib.nullcontext()
            with ctx:
                logits = forward_logits_with_tta(model, img, out_size, cfg)
            pred_cpu = logits.argmax(dim=1).detach().cpu().long()

            for i in range(pred_cpu.size(0)):
                conf_mat = update_confusion_matrix(conf_mat, pred_cpu[i], label_cpu[i], num_classes, ignore_index)

    per_dice, per_iou = compute_dice_iou_from_confmat(conf_mat)
    mean_dice = float(per_dice.mean().item())
    mean_iou = float(per_iou.mean().item())
    fg_mean_dice = float(per_dice[1:].mean().item()) if num_classes > 1 else mean_dice
    fg_mean_iou = float(per_iou[1:].mean().item()) if num_classes > 1 else mean_iou

    print("\n====================")
    print("[Result] Confusion-matrix metrics")
    print("====================")
    for c in range(num_classes):
        print(f"Class {c:02d}: Dice={per_dice[c]:.4f} | IoU={per_iou[c]:.4f}")
    print("\n--------------------")
    print(f"Mean Dice    = {mean_dice:.4f}")
    print(f"Mean IoU     = {mean_iou:.4f}")
    print(f"FG Mean Dice = {fg_mean_dice:.4f}")
    print(f"FG Mean IoU  = {fg_mean_iou:.4f}")
    print("--------------------")


if __name__ == "__main__":
    main()
