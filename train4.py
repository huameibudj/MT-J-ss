
import os
import yaml
import random
import contextlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.vit_backbone import CompatibleViTBackbone
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


def split_indices(n, val_test_split=0.3, test_split=0.5, seed=42):
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)

    n_valtest = int(round(n * float(val_test_split)))
    n_train = n - n_valtest

    train_idx = idx[:n_train]
    valtest_idx = idx[n_train:]

    n_test = int(round(len(valtest_idx) * float(test_split)))
    test_idx = valtest_idx[:n_test]
    val_idx = valtest_idx[n_test:]
    return train_idx, val_idx, test_idx


class SubsetWithPaths(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
        if not hasattr(dataset, "image_paths") or not hasattr(dataset, "label_paths"):
            raise AttributeError("[ERROR] Dataset 必须有 image_paths / label_paths 才能定位坏样本")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, label = self.dataset[idx]
        return img, label, self.dataset.image_paths[idx], self.dataset.label_paths[idx]


def build_dataset_from_cfg(cfg):
    dataset_name = str(cfg.get("dataset_name", "glassai")).lower()
    data_root = cfg["data_root"]
    image_dir = os.path.join(data_root, "Images")
    label_dir = os.path.join(data_root, "Labels")

    resize_hw = cfg.get("resize_hw", None)
    if isinstance(resize_hw, list) and len(resize_hw) == 2:
        resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
    else:
        resize_hw = None

    ignore_index = int(cfg.get("ignore_index", 255))

    if dataset_name == "glassai":
        from datasets.glassai_dataset import GlassAIDataset, FIXED_PALETTE
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

    elif dataset_name == "lungcancer":
        from datasets.lungcancer_dataset import LungCancerDataset, FIXED_PALETTE
        ds = LungCancerDataset(
            image_input=image_dir,
            label_input=label_dir,
            palette=FIXED_PALETTE
        )
        return ds, FIXED_PALETTE, dataset_name

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name} (use glassai or lungcancer)")


def save_split_files(dataset, train_idx, val_idx, test_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def _write(name, indices):
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            for idx in indices:
                img_p = dataset.image_paths[idx]
                lbl_p = dataset.label_paths[idx]
                f.write(f"{img_p}\t{lbl_p}\n")

    _write("train.txt", train_idx)
    _write("val.txt", val_idx)
    _write("test.txt", test_idx)


def _pick_amp_dtype(cfg):
    dtype_str = str(cfg.get("train_amp_dtype", "auto")).lower()
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str in ("fp16", "float16"):
        return torch.float16
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def dice_loss_from_logits(logits, target, num_classes, ignore_index=255, eps=1e-6):
    valid = (target != ignore_index)
    target = torch.where(valid, target, torch.zeros_like(target))
    prob = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1)
    prob = prob * valid
    onehot = onehot * valid
    inter = (prob * onehot).sum(dim=(0, 2, 3))
    union = prob.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    if num_classes > 1:
        dice = dice[1:]
    return 1.0 - dice.mean()


def build_class_presence_target(label, num_classes, ignore_index=255):
    """
    label: [B, H, W]
    return: [B, C] float tensor
    """
    bsz = label.shape[0]
    device = label.device
    target = torch.zeros((bsz, num_classes), dtype=torch.float32, device=device)

    for b in range(bsz):
        valid = (label[b] != ignore_index)
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
def compute_dataset_mIoU(model, loader, device, num_classes, ignore_index=255, use_amp=True):
    model.eval()

    total_inter = torch.zeros(num_classes, device=device, dtype=torch.float64)
    total_union = torch.zeros(num_classes, device=device, dtype=torch.float64)

    amp_on = bool(use_amp) and (device.type == "cuda")
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
                masks = F.interpolate(masks, size=label.shape[-2:], mode="bilinear", align_corners=False)

        pred = masks.argmax(dim=1)
        valid = (label != ignore_index)

        for c in range(num_classes):
            pred_c = (pred == c) & valid
            label_c = (label == c) & valid
            inter = (pred_c & label_c).sum()
            union = (pred_c | label_c).sum()
            total_inter[c] += inter
            total_union[c] += union

    iou = total_inter / (total_union + 1e-6)
    bg_iou = float(iou[0].item()) if num_classes > 0 else 0.0
    fg_miou = float(iou[1:].mean().item()) if num_classes > 1 else float(iou.mean().item())
    return bg_iou, fg_miou


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
        f"min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} std={s['std']:.4f} absmax={s['absmax']:.4f}"
    )
    if s["absmax"] > warn_absmax:
        print(f"[WARN] huge {name.lower()} detected: absmax={s['absmax']:.4f} (threshold={warn_absmax:.4f})")


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
    cls_slot_w: float,
    cls_presence_w: float,
    cls_presence_exclude_bg: bool,
):
    if masks.shape[-2:] != label.shape[-2:]:
        masks_up = F.interpolate(
            masks,
            size=label.shape[-2:],
            mode="bilinear",
            align_corners=False
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
        ignore_index=ignore_index
    )

    cls_slot = masks_up.new_tensor(0.0)
    if query_class_logits is not None:
        q_used = min(num_classes, query_class_logits.shape[1])
        slot_logits = query_class_logits[:, :q_used, :]
        slot_targets = torch.arange(
            q_used, device=slot_logits.device
        ).unsqueeze(0).expand(slot_logits.size(0), -1)
        cls_slot = F.cross_entropy(
            slot_logits.reshape(-1, num_classes),
            slot_targets.reshape(-1)
        )

    cls_presence = masks_up.new_tensor(0.0)
    presence_target = build_class_presence_target(
        label, num_classes=num_classes, ignore_index=ignore_index
    )
    if cls_presence_exclude_bg and num_classes > 1:
        cls_presence = F.binary_cross_entropy_with_logits(
            class_logits_img[:, 1:],
            presence_target[:, 1:]
        )
    else:
        cls_presence = F.binary_cross_entropy_with_logits(
            class_logits_img,
            presence_target
        )

    loss = ce_w * ce + dice_w * dice + cls_slot_w * cls_slot + cls_presence_w * cls_presence

    stat = {
        "ce": float(ce.item()),
        "dice": float(dice.item()),
        "cls_slot": float(cls_slot.item()),
        "cls_presence": float(cls_presence.item()),
    }
    aux = {
        "masks_up": masks_up.detach(),
        "class_logits_img": class_logits_img.detach(),
        "query_class_logits": None if query_class_logits is None else query_class_logits.detach(),
    }
    return loss, stat, aux


def main():
    cfg = load_yaml_any_encoding("configs/train_config.yaml")
    cfg["lr"] = float(cfg["lr"])

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -------------------------
    # 1) Dataset + split
    # -------------------------
    dataset, palette, dataset_name = build_dataset_from_cfg(cfg)
    print(f"[INFO] Using dataset: {dataset_name}, size={len(dataset)}")

    ignore_index = int(cfg.get("ignore_index", 255))
    print(f"[INFO] ignore_index={ignore_index}")

    train_idx, val_idx, test_idx = split_indices(
        len(dataset),
        val_test_split=float(cfg.get("val_test_split", 0.3)),
        test_split=float(cfg.get("test_split", 0.5)),
        seed=seed
    )

    splits_root = str(cfg.get("splits_root", "splits"))
    splits_dir = os.path.join(splits_root, dataset_name)
    save_split_files(dataset, train_idx, val_idx, test_idx, splits_dir)

    print(f"[INFO] Split dir: {splits_dir}")
    print(f"[INFO] Train/Val/Test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"[INFO] Saved splits -> {splits_dir}/train.txt, val.txt, test.txt")

    train_ds = SubsetWithPaths(dataset, train_idx)
    val_ds = SubsetWithPaths(dataset, val_idx)

    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("val_batch_size", cfg["batch_size"])),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
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
    backbone_params, head_params = [], []
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
        param_groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_mult})
    if len(joint_params) > 0:
        param_groups.append({"params": joint_params, "lr": base_lr * joint_lr_mult})
    if len(head_params) > 0:
        param_groups.append({"params": head_params, "lr": base_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    epochs = int(cfg["epochs"])
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    min_lr_ratio = float(cfg.get("min_lr_ratio", 0.01))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_scheduler_lambda(epochs, warmup_epochs, min_lr_ratio)
    )

    num_classes = int(cfg["num_classes"])
    use_amp = bool(cfg.get("train_amp", False)) and (device.type == "cuda")
    amp_dtype = _pick_amp_dtype(cfg) if use_amp else None
    scaler_enabled = use_amp and (amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))
    ce_w = float(cfg.get("loss_ce_weight", 0.6))
    dice_w = float(cfg.get("loss_dice_weight", 0.4))

    cls_slot_w = float(cfg.get("loss_cls_slot_weight", 0.02))
    cls_presence_w = float(cfg.get("loss_cls_presence_weight", 0.02))
    cls_presence_exclude_bg = bool(cfg.get("loss_cls_presence_exclude_bg", True))

    aux_pred_weight = float(cfg.get("aux_pred_weight", 0.20))

    logit_monitor_interval = int(cfg.get("logit_monitor_interval", 200))
    logit_warn_absmax = float(cfg.get("logit_warn_absmax", 30.0))

    ckpt_dir = str(cfg.get("ckpt_dir", "checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)
    save_each_epoch = bool(cfg.get("save_each_epoch", False))
    best_fg_miou = -1.0
    best_ckpt = os.path.join(ckpt_dir, f"{dataset_name}_best.pth")

    print("[INFO] Starting training.")
    print(
        f"[INFO] Loss weights: CE={ce_w}, Dice={dice_w}, "
        f"ClsSlot={cls_slot_w}, ClsPresence={cls_presence_w}, AuxPred={aux_pred_weight}"
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
            cls_slot_w=cls_slot_w,
            cls_presence_w=cls_presence_w,
            cls_presence_exclude_bg=cls_presence_exclude_bg,
        )

        total_loss = final_loss

        aux_ce = 0.0
        aux_dice = 0.0
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
                    cls_slot_w=cls_slot_w,
                    cls_presence_w=cls_presence_w,
                    cls_presence_exclude_bg=cls_presence_exclude_bg,
                )
                total_loss = total_loss + aux_pred_weight * loss_i
                aux_ce += stat_i["ce"]
                aux_dice += stat_i["dice"]
                aux_cls_slot += stat_i["cls_slot"]
                aux_cls_presence += stat_i["cls_presence"]
                n_aux_used += 1

        if n_aux_used > 0:
            aux_ce /= n_aux_used
            aux_dice /= n_aux_used
            aux_cls_slot /= n_aux_used
            aux_cls_presence /= n_aux_used

        stat = {
            "ce": final_stat["ce"],
            "dice": final_stat["dice"],
            "cls_slot": final_stat["cls_slot"],
            "cls_presence": final_stat["cls_presence"],
            "aux_ce": aux_ce,
            "aux_dice": aux_dice,
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
        loss_cls_slot_sum = 0.0
        loss_cls_presence_sum = 0.0
        loss_aux_ce_sum = 0.0
        loss_aux_dice_sum = 0.0
        update_steps = 0
        skipped_all_ignore = 0
        skipped_nonfinite = 0
        fallback_fp32 = 0
        total_pixels = 0
        valid_pixels = 0
        bad_examples = []

        prog = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=140)

        for batch in prog:
            global_step += 1

            img, label = batch[0], batch[1]
            lbl_paths = batch[3]

            label_cpu = label.long()
            valid_mask = (label_cpu != ignore_index)
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
            if use_amp and (not torch.isfinite(loss)):
                loss, stat, aux = forward_and_loss(img, label, amp_on=False)
                fallback_fp32 += 1

            if logit_monitor_interval > 0 and (global_step == 1 or global_step % logit_monitor_interval == 0):
                _print_logit_stats("LOGITS", aux["masks_up"], global_step, logit_warn_absmax)
                _print_logit_stats("CLASS_LOGITS_IMG", aux["class_logits_img"], global_step, logit_warn_absmax)
                if aux["query_class_logits"] is not None:
                    _print_logit_stats("QUERY_CLASS_LOGITS", aux["query_class_logits"], global_step, logit_warn_absmax)

            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                if len(bad_examples) < 10:
                    bad_examples.append(lbl_paths[0])
                continue

            if scaler_enabled:
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            loss_sum += float(loss.item())
            loss_ce_sum += stat["ce"]
            loss_dice_sum += stat["dice"]
            loss_cls_slot_sum += stat["cls_slot"]
            loss_cls_presence_sum += stat["cls_presence"]
            loss_aux_ce_sum += stat["aux_ce"]
            loss_aux_dice_sum += stat["aux_dice"]
            update_steps += 1

            prog.set_postfix({
                "loss": f"{loss_sum / max(1, update_steps):.4f}",
                "ce": f"{loss_ce_sum / max(1, update_steps):.4f}",
                "dice": f"{loss_dice_sum / max(1, update_steps):.4f}",
                "clsS": f"{loss_cls_slot_sum / max(1, update_steps):.4f}",
                "clsP": f"{loss_cls_presence_sum / max(1, update_steps):.4f}",
                "auxCE": f"{loss_aux_ce_sum / max(1, update_steps):.4f}",
            })

        avg_loss = loss_sum / max(1, update_steps)
        valid_ratio = valid_pixels / max(1, total_pixels)

        bg_iou, fg_miou = compute_dataset_mIoU(
            model,
            val_loader,
            device,
            num_classes,
            ignore_index=ignore_index,
            use_amp=bool(cfg.get("eval_amp", True)),
        )

        print(
            f"[Epoch {epoch + 1}] Loss={avg_loss:.4f} | "
            f"CE={loss_ce_sum / max(1, update_steps):.4f} | "
            f"Dice={loss_dice_sum / max(1, update_steps):.4f} | "
            f"ClsSlot={loss_cls_slot_sum / max(1, update_steps):.4f} | "
            f"ClsPresence={loss_cls_presence_sum / max(1, update_steps):.4f} | "
            f"AuxCE={loss_aux_ce_sum / max(1, update_steps):.4f} | "
            f"AuxDice={loss_aux_dice_sum / max(1, update_steps):.4f} | "
            f"UpdateSteps={update_steps}/{len(train_loader)} | "
            f"TrainValidRatio={valid_ratio:.4f} | "
            f"SkipAllIgnore={skipped_all_ignore} | "
            f"SkipNonFinite={skipped_nonfinite} | "
            f"FallbackFP32={fallback_fp32} | "
            f"Val_FG_mIoU={fg_miou:.4f} | "
            f"Val_BG_IoU={bg_iou:.4f}"
        )

        if update_steps == 0:
            print("[FATAL] 本 epoch 没有任何一次有效更新（全部 loss 非有限或全 ignore）。")
            if len(bad_examples) > 0:
                print("[FATAL] 示例 label 路径（仅供排查）：")
                for p in bad_examples[:5]:
                    print("  -", p)
            break

        if fg_miou > best_fg_miou:
            best_fg_miou = fg_miou
            torch.save(model.state_dict(), best_ckpt)
            print(f"[SAVE] best checkpoint -> {best_ckpt}, fg_mIoU={best_fg_miou:.4f}")

        if save_each_epoch:
            ckpt_path = os.path.join(ckpt_dir, f"{dataset_name}_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)

        scheduler.step()

    final_ckpt = os.path.join(ckpt_dir, f"{dataset_name}_epoch_{epochs}.pth")
    torch.save(model.state_dict(), final_ckpt)
    print(f"[INFO] Training finished. Final checkpoint: {final_ckpt}")
    print(f"[INFO] Best checkpoint: {best_ckpt} | best Val_FG_mIoU={best_fg_miou:.4f}")
    print(f"[INFO] Test set kept for evaluate.py: {splits_dir}/test.txt")


if __name__ == "__main__":
    main()
