import os
import yaml
import random
import contextlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.backboneglas import CompatibleViTBackbone
from models.eomt_head import EoMT


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


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SubsetWithPaths(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        if not hasattr(dataset, "image_paths") or not hasattr(dataset, "label_paths"):
            raise AttributeError("[ERROR] Dataset 必须有 image_paths / label_paths 才能定位坏样本")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        img, label = self.dataset[i]
        return img, label, self.dataset.image_paths[i], self.dataset.label_paths[i]


def build_dataset_from_cfg(cfg, split: str):
    dataset_name = str(cfg.get("dataset_name", "glas")).lower()
    data_root = cfg["data_root"]

    resize_hw = cfg.get("resize_hw", None)
    if isinstance(resize_hw, list) and len(resize_hw) == 2:
        resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
    elif isinstance(resize_hw, tuple) and len(resize_hw) == 2:
        resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
    else:
        resize_hw = None

    ignore_index = int(cfg.get("ignore_index", 255))
    use_train_aug = bool(cfg.get("train_aug", True)) and (split == "train")

    if dataset_name == "glas":
        from datasets.glas_dataset import GlasDataset

        split_root = os.path.join(data_root, split)
        image_dir = os.path.join(split_root, "images")
        label_dir = os.path.join(split_root, "masks")

        ds = GlasDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            resize_hw=resize_hw,
            ignore_index=ignore_index,
            augment=use_train_aug,
            hflip_prob=float(cfg.get("aug_hflip_prob", 0.5)),
            vflip_prob=float(cfg.get("aug_vflip_prob", 0.5)),
            rotate90_prob=float(cfg.get("aug_rotate90_prob", 0.5)),
            color_jitter_prob=float(cfg.get("aug_color_jitter_prob", 0.5)),
            blur_prob=float(cfg.get("aug_blur_prob", 0.15)),
        )
        return ds, None, dataset_name

    elif dataset_name == "glassai":
        from datasets.glassai_dataset import GlassAIDataset, FIXED_PALETTE

        split_root = os.path.join(data_root, split)
        image_dir = os.path.join(split_root, "images")
        label_dir = os.path.join(split_root, "masks")
        strict_match = bool(cfg.get("glassai_strict_match", True))

        ds = GlassAIDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            palette=FIXED_PALETTE,
            resize_hw=resize_hw,
            ignore_index=ignore_index,
            strict_match=strict_match,
        )
        return ds, FIXED_PALETTE, dataset_name

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name} (use glas or glassai)")


def save_split_files(train_ds, val_ds, test_ds, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def _write(name, dataset):
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            for img_p, lbl_p in zip(dataset.image_paths, dataset.label_paths):
                f.write(f"{img_p}\t{lbl_p}\n")

    _write("train.txt", train_ds)
    _write("val.txt", val_ds)
    _write("test.txt", test_ds)


def _pick_amp_dtype(cfg):
    dtype_str = str(cfg.get("train_amp_dtype", "auto")).lower()

    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16

    if dtype_str in ("fp16", "float16"):
        return torch.float16

    if (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16

    return torch.float16


def dice_loss_from_logits(logits, target, num_classes, ignore_index=255, eps=1e-6):
    valid = target != ignore_index

    target_safe = torch.where(valid, target, torch.zeros_like(target))

    prob = F.softmax(logits.float(), dim=1)
    onehot = F.one_hot(target_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()

    valid = valid.unsqueeze(1).float()

    prob = prob * valid
    onehot = onehot * valid

    inter = (prob * onehot).sum(dim=(0, 2, 3))
    union = prob.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))

    dice = (2 * inter + eps) / (union + eps)

    # 二分类 GlaS：训练 Dice 默认只看前景，避免背景过大主导 loss
    if num_classes > 1:
        dice = dice[1:]

    return 1.0 - dice.mean()


def boundary_loss_from_logits(logits, target, ignore_index=255):
    """
    简单边界 loss。
    用 GT mask 的边缘图监督预测前景概率的边缘。
    适合 GlaS 二分类分割。

    注意：
    - 只作用于前景 class=1。
    - ignore_index 区域不参与。
    """
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    prob = torch.softmax(logits.float(), dim=1)[:, 1:2]

    valid = (target != ignore_index).float().unsqueeze(1)
    target_fg = (target == 1).float().unsqueeze(1)

    prob = prob * valid
    target_fg = target_fg * valid

    def gradient_map(x):
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))

        edge = torch.clamp(dx + dy, 0.0, 1.0)
        return edge

    pred_edge = gradient_map(prob)
    target_edge = gradient_map(target_fg)

    loss = F.binary_cross_entropy(pred_edge, target_edge)
    return loss


def build_class_presence_target(label, num_classes, ignore_index=255):
    bsz = label.shape[0]
    device = label.device

    target = torch.zeros((bsz, num_classes), dtype=torch.float32, device=device)

    for b in range(bsz):
        valid = label[b] != ignore_index

        if valid.sum().item() == 0:
            continue

        present = torch.unique(label[b][valid])
        present = present[(present >= 0) & (present < num_classes)]

        if present.numel() > 0:
            target[b, present.long()] = 1.0

    return target


def unpack_model_output(model, out):
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise RuntimeError("[ERROR] model(img) 必须返回 (masks, class_logits_img)")

    masks = out[0]
    class_logits_img = out[1]

    query_class_logits = getattr(model, "last_query_class_logits", None)
    aux_outputs = getattr(model, "last_aux_outputs", None)

    return masks, class_logits_img, query_class_logits, aux_outputs


@torch.no_grad()
def compute_dataset_metrics_from_confmat(
    model,
    loader,
    device,
    num_classes,
    ignore_index=255,
    use_amp=True,
):
    model.eval()

    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    amp_on = bool(use_amp) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (
            torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        else torch.float16
    )

    for batch in loader:
        img, label = batch[0], batch[1]

        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True).long()

        ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if amp_on else contextlib.nullcontext()

        with ctx:
            out = model(img)
            masks, _, _, _ = unpack_model_output(model, out)

            if masks.shape[-2:] != label.shape[-2:]:
                masks = F.interpolate(
                    masks,
                    size=label.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        pred = masks.argmax(dim=1)

        valid = (label != ignore_index) & (label >= 0) & (label < num_classes)
        if valid.sum().item() == 0:
            continue

        t = label[valid].view(-1)
        p = pred[valid].view(-1)

        valid_p = (p >= 0) & (p < num_classes)
        t = t[valid_p]
        p = p[valid_p]

        if t.numel() == 0:
            continue

        idx = (t * num_classes + p).to(torch.int64)
        bins = torch.bincount(idx, minlength=num_classes * num_classes)
        conf_mat += bins.view(num_classes, num_classes)

    conf = conf_mat.to(torch.float64)

    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp

    iou = tp / (tp + fp + fn + 1e-12)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-12)

    mean_iou = float(iou.mean().item())
    mean_dice = float(dice.mean().item())

    bg_iou = float(iou[0].item()) if num_classes > 0 else 0.0
    fg_miou = float(iou[1:].mean().item()) if num_classes > 1 else mean_iou

    bg_dice = float(dice[0].item()) if num_classes > 0 else 0.0
    fg_dice = float(dice[1:].mean().item()) if num_classes > 1 else mean_dice

    return {
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "bg_iou": bg_iou,
        "fg_miou": fg_miou,
        "bg_dice": bg_dice,
        "fg_dice": fg_dice,
        "per_iou": [float(x.item()) for x in iou],
        "per_dice": [float(x.item()) for x in dice],
        "conf_mat": conf_mat.detach().cpu(),
    }


def build_scheduler_lambda(total_epochs: int, warmup_epochs: int, min_lr_ratio: float):
    total_epochs = max(1, int(total_epochs))
    warmup_epochs = max(0, int(warmup_epochs))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(epoch: int):
        e = epoch + 1

        if warmup_epochs > 0 and e <= warmup_epochs:
            return max(1e-8, e / warmup_epochs)

        if total_epochs <= warmup_epochs:
            return 1.0

        progress = (e - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def compute_single_pred_losses(
    masks: torch.Tensor,
    class_logits_img: torch.Tensor,
    query_class_logits: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    ce_w: float,
    dice_w: float,
    boundary_w: float,
    cls_slot_w: float,
    cls_presence_w: float,
    cls_presence_exclude_bg: bool,
):
    if masks.shape[-2:] != label.shape[-2:]:
        masks_up = F.interpolate(
            masks,
            size=label.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).float()
    else:
        masks_up = masks.float()

    class_logits_img = class_logits_img.float()
    query_class_logits = None if query_class_logits is None else query_class_logits.float()

    ce = F.cross_entropy(masks_up, label, ignore_index=ignore_index)

    dice = dice_loss_from_logits(
        masks_up,
        label,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    if boundary_w > 0:
        boundary = boundary_loss_from_logits(
            masks_up,
            label,
            ignore_index=ignore_index,
        )
    else:
        boundary = masks_up.new_tensor(0.0)

    cls_slot = masks_up.new_tensor(0.0)

    if cls_slot_w > 0 and query_class_logits is not None:
        q_used = min(num_classes, query_class_logits.shape[1])
        slot_logits = query_class_logits[:, :q_used, :]

        slot_targets = torch.arange(
            q_used,
            device=slot_logits.device,
        ).unsqueeze(0).expand(slot_logits.size(0), -1)

        cls_slot = F.cross_entropy(
            slot_logits.reshape(-1, num_classes),
            slot_targets.reshape(-1),
        )

    cls_presence = masks_up.new_tensor(0.0)

    if cls_presence_w > 0:
        presence_target = build_class_presence_target(
            label,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        if cls_presence_exclude_bg and num_classes > 1:
            cls_presence = F.binary_cross_entropy_with_logits(
                class_logits_img[:, 1:],
                presence_target[:, 1:],
            )
        else:
            cls_presence = F.binary_cross_entropy_with_logits(
                class_logits_img,
                presence_target,
            )

    loss = (
        ce_w * ce
        + dice_w * dice
        + boundary_w * boundary
        + cls_slot_w * cls_slot
        + cls_presence_w * cls_presence
    )

    stat = {
        "ce": float(ce.item()),
        "dice": float(dice.item()),
        "boundary": float(boundary.item()),
        "cls_slot": float(cls_slot.item()),
        "cls_presence": float(cls_presence.item()),
    }

    aux = {
        "masks_up": masks_up.detach(),
        "class_logits_img": class_logits_img.detach(),
        "query_class_logits": None if query_class_logits is None else query_class_logits.detach(),
    }

    return loss, stat, aux


def _tensor_stats(x: torch.Tensor):
    x = x.detach().float()
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "absmax": float(x.abs().max().item()),
    }


def _print_logit_stats(name: str, x: torch.Tensor, step: int, warn_absmax: float):
    s = _tensor_stats(x)

    print(
        f"[{name}] step={step} "
        f"min={s['min']:.4f} max={s['max']:.4f} "
        f"mean={s['mean']:.4f} std={s['std']:.4f} "
        f"absmax={s['absmax']:.4f}"
    )

    if s["absmax"] > warn_absmax:
        print(
            f"[WARN] huge {name.lower()} detected: "
            f"absmax={s['absmax']:.4f} "
            f"(threshold={warn_absmax:.4f})"
        )


def main():
    cfg = load_yaml_any_encoding("configs/trainconfig_glas.yaml")
    cfg["lr"] = float(cfg["lr"])

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -------------------------
    # 1) Dataset
    # -------------------------
    train_base_ds, palette, dataset_name = build_dataset_from_cfg(cfg, split="train")
    val_base_ds, _, _ = build_dataset_from_cfg(cfg, split="val")
    test_base_ds, _, _ = build_dataset_from_cfg(cfg, split="test")

    print(f"[INFO] Using dataset: {dataset_name}")
    print(f"[INFO] Train/Val/Test = {len(train_base_ds)}/{len(val_base_ds)}/{len(test_base_ds)}")

    ignore_index = int(cfg.get("ignore_index", 255))
    print(f"[INFO] ignore_index={ignore_index}")

    print(
        f"[INFO] img_size={cfg.get('img_size', 224)}, "
        f"resize_hw={cfg.get('resize_hw', None)}, "
        f"train_aug={bool(cfg.get('train_aug', True))}"
    )

    splits_root = str(cfg.get("splits_root", "splits"))
    splits_dir = os.path.join(splits_root, dataset_name)
    save_split_files(train_base_ds, val_base_ds, test_base_ds, splits_dir)

    print(f"[INFO] Saved splits -> {splits_dir}/train.txt, val.txt, test.txt")

    train_ds = SubsetWithPaths(train_base_ds)
    val_ds = SubsetWithPaths(val_base_ds)

    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("val_batch_size", cfg["batch_size"])),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # -------------------------
    # 2) Model
    # -------------------------
    vit = CompatibleViTBackbone(cfg)
    vit = vit.float()

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

    # -------------------------
    # 3) Optimizer
    # -------------------------
    backbone_params = []
    head_params = []
    joint_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if "queries" in n:
            joint_params.append(p)
        elif ("vit_model" in n) or ("backbone" in n) or ("vit." in n):
            backbone_params.append(p)
        else:
            head_params.append(p)

    base_lr = float(cfg["lr"])
    backbone_lr_mult = float(cfg.get("backbone_lr_mult", 0.1))
    joint_lr_mult = float(cfg.get("joint_lr_mult", 0.5))
    weight_decay = float(cfg.get("weight_decay", 0.05))

    param_groups = []

    if len(backbone_params) > 0:
        param_groups.append({
            "params": backbone_params,
            "lr": base_lr * backbone_lr_mult,
        })

    if len(joint_params) > 0:
        param_groups.append({
            "params": joint_params,
            "lr": base_lr * joint_lr_mult,
        })

    if len(head_params) > 0:
        param_groups.append({
            "params": head_params,
            "lr": base_lr,
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
    )

    epochs = int(cfg["epochs"])
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    min_lr_ratio = float(cfg.get("min_lr_ratio", 0.01))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_scheduler_lambda(
            total_epochs=epochs,
            warmup_epochs=warmup_epochs,
            min_lr_ratio=min_lr_ratio,
        ),
    )

    num_classes = int(cfg["num_classes"])

    use_amp = bool(cfg.get("train_amp", False)) and device.type == "cuda"
    amp_dtype = _pick_amp_dtype(cfg) if use_amp else None

    scaler_enabled = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))

    ce_w = float(cfg.get("loss_ce_weight", 0.5))
    dice_w = float(cfg.get("loss_dice_weight", 0.5))
    boundary_w = float(cfg.get("loss_boundary_weight", 0.0))

    cls_slot_w = float(cfg.get("loss_cls_slot_weight", 0.0))
    cls_presence_w = float(cfg.get("loss_cls_presence_weight", 0.0))
    cls_presence_exclude_bg = bool(cfg.get("loss_cls_presence_exclude_bg", True))

    aux_pred_weight = float(cfg.get("aux_pred_weight", 0.10))

    logit_monitor_interval = int(cfg.get("logit_monitor_interval", 200))
    logit_warn_absmax = float(cfg.get("logit_warn_absmax", 30.0))

    ckpt_dir = str(cfg.get("ckpt_dir", "checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)

    save_each_epoch = bool(cfg.get("save_each_epoch", False))
    metric_for_best = str(cfg.get("metric_for_best", "mean_dice")).lower()

    best_metric = -1.0
    best_ckpt = os.path.join(ckpt_dir, f"{dataset_name}_best.pth")

    print("[INFO] Starting training.")

    print(
        f"[INFO] Loss weights: "
        f"CE={ce_w}, Dice={dice_w}, Boundary={boundary_w}, "
        f"ClsSlot={cls_slot_w}, ClsPresence={cls_presence_w}, "
        f"AuxPred={aux_pred_weight}"
    )

    print(
        f"[INFO] Optimizer lr: "
        f"backbone={base_lr * backbone_lr_mult}, "
        f"joint={base_lr * joint_lr_mult}, "
        f"head={base_lr}"
    )

    print(
        f"[INFO] Joint query blocks={int(cfg.get('joint_query_blocks', 1))}, "
        f"mask_gate_floor={float(cfg.get('mask_gate_floor', 0.30))}"
    )

    print(
        f"[INFO] FPN config: "
        f"fpn_dim={int(cfg.get('fpn_dim', 256))}, "
        f"fpn_layers={int(cfg.get('fpn_layers', 3))}, "
        f"out_upsample_x4={bool(cfg.get('out_upsample_x4', True))}, "
        f"final_upsample_to_input={bool(cfg.get('final_upsample_to_input', True))}"
    )

    print(f"[INFO] Scheduler: warmup_epochs={warmup_epochs}, min_lr_ratio={min_lr_ratio}")
    print(f"[INFO] metric_for_best={metric_for_best}")

    if use_amp:
        print(f"[INFO] AMP enabled. dtype={amp_dtype} (scaler={scaler_enabled})")
    else:
        print("[INFO] AMP disabled (full fp32).")

    def forward_and_loss(img, label, amp_on: bool):
        if amp_on and use_amp:
            ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            out = model(img)
            masks, class_logits_img, query_class_logits, aux_outputs = unpack_model_output(model, out)

        final_loss, final_stat, final_aux = compute_single_pred_losses(
            masks=masks,
            class_logits_img=class_logits_img,
            query_class_logits=query_class_logits,
            label=label,
            num_classes=num_classes,
            ignore_index=ignore_index,
            ce_w=ce_w,
            dice_w=dice_w,
            boundary_w=boundary_w,
            cls_slot_w=cls_slot_w,
            cls_presence_w=cls_presence_w,
            cls_presence_exclude_bg=cls_presence_exclude_bg,
        )

        total_loss = final_loss

        aux_ce = 0.0
        aux_dice = 0.0
        aux_boundary = 0.0
        aux_cls_slot = 0.0
        aux_cls_presence = 0.0
        n_aux_used = 0

        if aux_outputs is not None and len(aux_outputs) > 1 and aux_pred_weight > 0:
            aux_list = aux_outputs[:-1]

            for pred in aux_list:
                loss_i, stat_i, _ = compute_single_pred_losses(
                    masks=pred["masks"],
                    class_logits_img=pred["class_logits_img"],
                    query_class_logits=pred["query_class_logits"],
                    label=label,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    ce_w=ce_w,
                    dice_w=dice_w,
                    boundary_w=boundary_w,
                    cls_slot_w=cls_slot_w,
                    cls_presence_w=cls_presence_w,
                    cls_presence_exclude_bg=cls_presence_exclude_bg,
                )

                total_loss = total_loss + aux_pred_weight * loss_i

                aux_ce += stat_i["ce"]
                aux_dice += stat_i["dice"]
                aux_boundary += stat_i["boundary"]
                aux_cls_slot += stat_i["cls_slot"]
                aux_cls_presence += stat_i["cls_presence"]
                n_aux_used += 1

        if n_aux_used > 0:
            aux_ce /= n_aux_used
            aux_dice /= n_aux_used
            aux_boundary /= n_aux_used
            aux_cls_slot /= n_aux_used
            aux_cls_presence /= n_aux_used

        stat = {
            "ce": final_stat["ce"],
            "dice": final_stat["dice"],
            "boundary": final_stat["boundary"],
            "cls_slot": final_stat["cls_slot"],
            "cls_presence": final_stat["cls_presence"],
            "aux_ce": aux_ce,
            "aux_dice": aux_dice,
            "aux_boundary": aux_boundary,
            "aux_cls_slot": aux_cls_slot,
            "aux_cls_presence": aux_cls_presence,
            "n_aux_used": n_aux_used,
        }

        aux = final_aux

        return total_loss, stat, aux

    # -------------------------
    # 4) Train
    # -------------------------
    global_step = 0

    for epoch in range(epochs):
        model.train()

        loss_sum = 0.0
        loss_ce_sum = 0.0
        loss_dice_sum = 0.0
        loss_boundary_sum = 0.0
        loss_cls_slot_sum = 0.0
        loss_cls_presence_sum = 0.0
        loss_aux_ce_sum = 0.0
        loss_aux_dice_sum = 0.0
        loss_aux_boundary_sum = 0.0

        update_steps = 0
        skipped_all_ignore = 0
        skipped_nonfinite = 0
        fallback_fp32 = 0

        total_pixels = 0
        valid_pixels = 0

        bad_examples = []

        prog = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            ncols=150,
        )

        for batch in prog:
            global_step += 1

            img, label = batch[0], batch[1]
            lbl_paths = batch[3]

            label_cpu = label.long()
            valid_mask = label_cpu != ignore_index

            v = int(valid_mask.sum().item())
            total = int(label_cpu.numel())

            total_pixels += total
            valid_pixels += v

            if v == 0:
                skipped_all_ignore += 1
                if len(bad_examples) < 10:
                    bad_examples.append(lbl_paths[0])
                continue

            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            loss, stat, aux = forward_and_loss(img, label, amp_on=True)

            if use_amp and not torch.isfinite(loss):
                loss, stat, aux = forward_and_loss(img, label, amp_on=False)
                fallback_fp32 += 1

            if logit_monitor_interval > 0 and (
                global_step == 1 or global_step % logit_monitor_interval == 0
            ):
                _print_logit_stats("LOGITS", aux["masks_up"], global_step, logit_warn_absmax)
                _print_logit_stats("CLASS_LOGITS_IMG", aux["class_logits_img"], global_step, logit_warn_absmax)

                if aux["query_class_logits"] is not None:
                    _print_logit_stats(
                        "QUERY_CLASS_LOGITS",
                        aux["query_class_logits"],
                        global_step,
                        logit_warn_absmax,
                    )

            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                if len(bad_examples) < 10:
                    bad_examples.append(lbl_paths[0])
                continue

            if scaler_enabled:
                scaler.scale(loss).backward()

                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip_norm,
                    )

                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()

                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip_norm,
                    )

                optimizer.step()

            loss_sum += float(loss.item())
            loss_ce_sum += stat["ce"]
            loss_dice_sum += stat["dice"]
            loss_boundary_sum += stat["boundary"]
            loss_cls_slot_sum += stat["cls_slot"]
            loss_cls_presence_sum += stat["cls_presence"]
            loss_aux_ce_sum += stat["aux_ce"]
            loss_aux_dice_sum += stat["aux_dice"]
            loss_aux_boundary_sum += stat["aux_boundary"]

            update_steps += 1

            prog.set_postfix({
                "loss": f"{loss_sum / max(1, update_steps):.4f}",
                "ce": f"{loss_ce_sum / max(1, update_steps):.4f}",
                "dice": f"{loss_dice_sum / max(1, update_steps):.4f}",
                "bd": f"{loss_boundary_sum / max(1, update_steps):.4f}",
                "auxBD": f"{loss_aux_boundary_sum / max(1, update_steps):.4f}",
            })

        avg_loss = loss_sum / max(1, update_steps)
        valid_ratio = valid_pixels / max(1, total_pixels)

        val_metrics = compute_dataset_metrics_from_confmat(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
            use_amp=bool(cfg.get("eval_amp", True)),
        )

        val_mean_dice = val_metrics["mean_dice"]
        val_mean_iou = val_metrics["mean_iou"]
        val_fg_miou = val_metrics["fg_miou"]
        val_bg_iou = val_metrics["bg_iou"]

        print(
            f"[Epoch {epoch + 1}] "
            f"Loss={avg_loss:.4f} | "
            f"CE={loss_ce_sum / max(1, update_steps):.4f} | "
            f"Dice={loss_dice_sum / max(1, update_steps):.4f} | "
            f"Boundary={loss_boundary_sum / max(1, update_steps):.4f} | "
            f"ClsSlot={loss_cls_slot_sum / max(1, update_steps):.4f} | "
            f"ClsPresence={loss_cls_presence_sum / max(1, update_steps):.4f} | "
            f"AuxCE={loss_aux_ce_sum / max(1, update_steps):.4f} | "
            f"AuxDice={loss_aux_dice_sum / max(1, update_steps):.4f} | "
            f"AuxBoundary={loss_aux_boundary_sum / max(1, update_steps):.4f} | "
            f"UpdateSteps={update_steps}/{len(train_loader)} | "
            f"TrainValidRatio={valid_ratio:.4f} | "
            f"SkipAllIgnore={skipped_all_ignore} | "
            f"SkipNonFinite={skipped_nonfinite} | "
            f"FallbackFP32={fallback_fp32} | "
            f"Val_meanDice={val_mean_dice:.4f} | "
            f"Val_meanIoU={val_mean_iou:.4f} | "
            f"Val_FG_mIoU={val_fg_miou:.4f} | "
            f"Val_BG_IoU={val_bg_iou:.4f}"
        )

        if update_steps == 0:
            print("[FATAL] 本 epoch 没有任何一次有效更新（全部 loss 非有限或全 ignore）。")
            if len(bad_examples) > 0:
                print("[FATAL] 示例 label 路径（仅供排查）：")
                for p in bad_examples[:5]:
                    print("  -", p)
            break

        if metric_for_best in ("mean_dice", "dice", "macro_dice"):
            current_metric = val_mean_dice
            metric_print_name = "mean_dice"
        elif metric_for_best in ("mean_iou", "miou", "macro_iou"):
            current_metric = val_mean_iou
            metric_print_name = "mean_iou"
        elif metric_for_best in ("fg_miou", "foreground_miou"):
            current_metric = val_fg_miou
            metric_print_name = "fg_miou"
        else:
            raise ValueError(
                f"[ERROR] Unsupported metric_for_best={metric_for_best}. "
                f"Use mean_dice / mean_iou / fg_miou."
            )

        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(model.state_dict(), best_ckpt)
            print(
                f"[SAVE] best checkpoint -> {best_ckpt}, "
                f"{metric_print_name}={best_metric:.4f}"
            )

        if save_each_epoch:
            ckpt_path = os.path.join(
                ckpt_dir,
                f"{dataset_name}_epoch_{epoch + 1}.pth",
            )
            torch.save(model.state_dict(), ckpt_path)

        scheduler.step()

    final_ckpt = os.path.join(ckpt_dir, f"{dataset_name}_epoch_{epochs}.pth")
    torch.save(model.state_dict(), final_ckpt)

    print(f"[INFO] Training finished. Final checkpoint: {final_ckpt}")
    print(f"[INFO] Best checkpoint: {best_ckpt} | best Val_{metric_for_best}={best_metric:.4f}")
    print(f"[INFO] Test set kept for evaluate.py: {splits_dir}/test.txt")


if __name__ == "__main__":
    main()