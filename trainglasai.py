import os
import sys
import random
import contextlib
from copy import deepcopy

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.vit_backbone import CompatibleViTBackbone
from models.eomt_head import EoMT


CLASS_NAMES_GLASSAI = {
    0: "Background",   # RGB (255,255,255)
    1: "Yellow",       # RGB (255,255,0)
    2: "Red",          # RGB (255,0,0)
    3: "Cyan",         # RGB (0,255,255)
    4: "Blue",         # RGB (0,0,255)
    5: "Green",        # RGB (0,255,0)
}

TASK_NAME = "glassai_6class_segmentation_pretrained_frozen"
ABLATION_MODE = "pretrained_backbone_fully_frozen"


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


def configure_fast_runtime(device, cfg):
    """Enable safe CUDA speedups without changing the model architecture."""
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


def make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass

    return torch.cuda.amp.GradScaler(enabled=enabled)


def make_progress(iterable, desc, cfg, leave=False, total=None):
    """
    A tqdm wrapper that avoids repeated lines in redirected logs/IDE consoles.
    Set tqdm_disable explicitly in YAML to override auto detection.
    """
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


def enforce_glassai_protocol(cfg):
    """
    GlassAI 训练协议：
    - 使用 glassai_dataset.py 中的 FIXED_PALETTE
    - 6 个有效类别：0..5
    - ignore_index=255 只用于未知颜色/无效像素，不参与 loss 和 metrics
    """
    old_dataset_name = str(cfg.get("dataset_name", "glassai")).lower()
    old_num_classes = int(cfg.get("num_classes", 6))
    old_ignore_index = int(cfg.get("ignore_index", 255))

    if old_dataset_name != "glassai":
        print(
            f"[WARN] dataset_name in config is {old_dataset_name}, "
            f"but this training script is for GlassAI. Override dataset_name -> glassai."
        )

    if old_num_classes != 6:
        print(
            f"[WARN] num_classes in config is {old_num_classes}, "
            f"but GlassAI FIXED_PALETTE has 6 valid classes. "
            f"Override num_classes -> 6."
        )

    if old_ignore_index != 255:
        print(
            f"[WARN] ignore_index in config is {old_ignore_index}. "
            f"Unknown colors in GlassAI labels are mapped to 255. "
            f"Override ignore_index -> 255."
        )

    cfg["dataset_name"] = "glassai"
    cfg["num_classes"] = 6
    cfg["ignore_index"] = 255

    return cfg


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
    def __init__(self, dataset, indices, train_aug=False, cfg=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.train_aug = bool(train_aug)
        self.cfg = cfg or {}

        if not hasattr(dataset, "image_paths") or not hasattr(dataset, "label_paths"):
            raise AttributeError("[ERROR] Dataset must have image_paths / label_paths")

    def __len__(self):
        return len(self.indices)

    def _augment(self, img, label):
        if bool(self.cfg.get("aug_hflip", True)) and random.random() < float(
            self.cfg.get("aug_hflip_p", 0.5)
        ):
            img = torch.flip(img, dims=(-1,))
            label = torch.flip(label, dims=(-1,))

        if bool(self.cfg.get("aug_vflip", True)) and random.random() < float(
            self.cfg.get("aug_vflip_p", 0.5)
        ):
            img = torch.flip(img, dims=(-2,))
            label = torch.flip(label, dims=(-2,))

        if bool(self.cfg.get("aug_rotate90", True)):
            k = random.randint(0, 3)

            if k:
                img = torch.rot90(img, k=k, dims=(-2, -1))
                label = torch.rot90(label, k=k, dims=(-2, -1))

        noise_std = float(self.cfg.get("aug_noise_std", 0.0))

        if noise_std > 0:
            img = img + torch.randn_like(img) * noise_std
            img = img.clamp(0.0, 1.0)

        return img, label

    def __getitem__(self, i):
        idx = self.indices[i]
        img, label = self.dataset[idx]
        label = label.long()

        if self.train_aug:
            img, label = self._augment(img, label)

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
        try:
            from datasets.glassai_dataset import GlassAIDataset, FIXED_PALETTE
        except ImportError:
            from glassai_dataset import GlassAIDataset, FIXED_PALETTE

        ds = GlassAIDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            palette=FIXED_PALETTE,
            resize_hw=resize_hw,
            ignore_index=ignore_index,
            strict_match=bool(cfg.get("glassai_strict_match", True)),
        )

        return ds, FIXED_PALETTE, dataset_name

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def save_split_files(dataset, train_idx, val_idx, test_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def _write(name, indices):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            for idx in indices:
                f.write(f"{dataset.image_paths[idx]}\t{dataset.label_paths[idx]}\n")

    _write("train.txt", train_idx)
    _write("val.txt", val_idx)
    _write("test.txt", test_idx)


@torch.inference_mode()
def update_confusion_matrix(conf_mat, pred, target, num_classes, ignore_index=255):
    """Vectorized confusion-matrix update for a full batch on CPU or GPU."""
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


def compute_dice_iou_from_confmat(conf_mat: torch.Tensor):
    conf = conf_mat.to(torch.float64)

    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp

    dice = torch.where(
        2 * tp + fp + fn > 0,
        (2 * tp) / (2 * tp + fp + fn + 1e-8),
        torch.zeros_like(tp),
    )

    iou = torch.where(
        tp + fp + fn > 0,
        tp / (tp + fp + fn + 1e-8),
        torch.zeros_like(tp),
    )

    return dice, iou


def summarize_confmat(conf_mat: torch.Tensor):
    per_dice, per_iou = compute_dice_iou_from_confmat(conf_mat)

    mean_dice = float(per_dice.mean().item())
    mean_iou = float(per_iou.mean().item())

    return {
        "per_dice": per_dice,
        "per_iou": per_iou,
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
    }


def format_per_class_metric(values, prefix):
    parts = []

    for i, x in enumerate(values):
        class_name = CLASS_NAMES_GLASSAI.get(i, f"Class{i}")
        parts.append(f"{prefix}{i}({class_name})={float(x):.4f}")

    return " ".join(parts)


@torch.no_grad()
def evaluate_dataset_metrics(
    model,
    loader,
    device,
    num_classes,
    ignore_index=255,
    use_amp=True,
    amp_dtype=torch.float16,
):
    was_training = model.training
    model.eval()

    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    amp_on = bool(use_amp) and device.type == "cuda"

    for batch in loader:
        img, label = batch[0], batch[1]

        img = img.to(device, non_blocking=True)
        label_cpu = label.long()

        with autocast_context(device, amp_on, amp_dtype):
            out = model(img)
            masks = out[0]

            if masks.shape[-2:] != label_cpu.shape[-2:]:
                masks = F.interpolate(
                    masks,
                    size=label_cpu.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        pred = masks.argmax(dim=1)
        label_device = label_cpu.to(device, non_blocking=True)
        update_confusion_matrix(
            conf_mat, pred, label_device, num_classes, ignore_index
        )

    if was_training:
        model.train()

    return summarize_confmat(conf_mat.cpu())


def unpack_model_output(model, out):
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise RuntimeError("[ERROR] model(img) must return at least (masks, class_logits_img)")

    masks = out[0]
    class_logits_img = out[1]
    aux_outputs = getattr(model, "last_aux_outputs", None)

    return masks, class_logits_img, aux_outputs


def get_backbone_module(model):
    for name in ("vit_model", "backbone", "vit"):
        if hasattr(model, name):
            module = getattr(model, name)

            if module is not None:
                return module

    return None


def set_backbone_trainable(model, trainable: bool):
    backbone = get_backbone_module(model)

    if backbone is None:
        print("[WARN] Could not find backbone module to freeze/unfreeze.")
        return 0

    n_params = 0

    for p in backbone.parameters():
        p.requires_grad = bool(trainable)
        n_params += p.numel()

    backbone.train(bool(trainable))

    return n_params


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
    ).float().to(device)

    return model


def make_optimizer(model, cfg):
    base_lr = float(cfg["lr"])
    backbone_lr_mult = float(cfg.get("backbone_lr_mult", 0.03))
    joint_lr_mult = float(cfg.get("joint_lr_mult", 0.5))
    weight_decay = float(cfg.get("weight_decay", 0.12))

    backbone_params = []
    joint_params = []
    head_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if "queries" in n:
            joint_params.append(p)
        elif ("vit_model" in n) or ("backbone" in n) or ("vit." in n):
            backbone_params.append(p)
        else:
            head_params.append(p)

    groups = []

    if backbone_params:
        groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_mult, "name": "backbone"})
    if joint_params:
        groups.append({"params": joint_params, "lr": base_lr * joint_lr_mult, "name": "joint"})
    if head_params:
        groups.append({"params": head_params, "lr": base_lr, "name": "head"})

    if not groups:
        raise RuntimeError("[ERROR] No trainable parameters remain after freezing the backbone.")

    use_fused = bool(cfg.get("fused_adamw", True)) and torch.cuda.is_available()
    optimizer = None

    if use_fused:
        try:
            optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay, fused=True)
            print("[INFO] Using fused AdamW optimizer.")
        except (TypeError, RuntimeError):
            optimizer = None

    if optimizer is None:
        optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
        print("[INFO] Using standard AdamW optimizer.")

    for g in optimizer.param_groups:
        g["base_lr"] = float(g["lr"])
        g["min_lr"] = float(g["base_lr"]) * float(cfg.get("min_lr_ratio", 0.05))

    return optimizer


def set_warmup_lr(optimizer, epoch, warmup_epochs):
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return

    factor = float(epoch + 1) / float(warmup_epochs)

    for g in optimizer.param_groups:
        g["lr"] = max(g["min_lr"], g["base_lr"] * factor)


def reduce_lr_on_plateau(optimizer, factor=0.5):
    changed = False

    for g in optimizer.param_groups:
        old = float(g["lr"])
        new = max(float(g["min_lr"]), old * factor)

        if new < old - 1e-12:
            g["lr"] = new
            changed = True

    return changed


def format_lrs(optimizer):
    return ", ".join(
        [
            f"{g.get('name', i)}={g['lr']:.3e}"
            for i, g in enumerate(optimizer.param_groups)
        ]
    )


@torch.no_grad()
def estimate_class_weights(dataset, indices, num_classes, ignore_index, cfg):
    if not bool(cfg.get("class_weight_enable", True)):
        print("[INFO] Class weights disabled.")
        return None

    max_scan = int(cfg.get("class_weight_max_scan", 12000))
    scan_indices = list(indices)

    random.Random(int(cfg.get("seed", 42))).shuffle(scan_indices)
    scan_indices = scan_indices[: min(len(scan_indices), max_scan)]

    counts = torch.zeros(num_classes, dtype=torch.float64)

    print(f"[INFO] Estimating class weights from {len(scan_indices)} training masks...")
    print("[INFO] Valid classes for weight scan: 0..5 = GlassAI palette classes")
    print(f"[INFO] Pixels with label={ignore_index} are ignored during class-weight scan.")

    for idx in make_progress(scan_indices, "ClassWeightScan", cfg, leave=False):
        _, label = dataset[idx]
        label = label.long().view(-1)

        valid = (label != ignore_index) & (label >= 0) & (label < num_classes)

        if valid.any():
            counts += torch.bincount(label[valid], minlength=num_classes).double()

    counts = counts.clamp_min(1.0)
    freq = counts / counts.sum()

    weights = torch.sqrt(freq.mean() / freq).float()
    weights = weights / weights.mean().clamp_min(1e-8)

    weights = weights.clamp(
        min=float(cfg.get("class_weight_min", 0.35)),
        max=float(cfg.get("class_weight_max", 3.0)),
    )

    weights[0] *= float(cfg.get("class0_weight_mult", 1.0))
    weights = weights / weights.mean().clamp_min(1e-8)

    print("[INFO] class pixel counts:")
    for i in range(num_classes):
        class_name = CLASS_NAMES_GLASSAI.get(i, f"Class{i}")
        print(f"       C{i}({class_name}) = {int(counts[i].item())}")

    print("[INFO] Pixels with unknown colors should have label=255 and are not included above.")

    print("[INFO] class loss weights:")
    for i in range(num_classes):
        class_name = CLASS_NAMES_GLASSAI.get(i, f"Class{i}")
        print(f"       C{i}({class_name}) weight = {float(weights[i]):.4f}")

    return weights


def dice_loss_from_logits(
    logits,
    target,
    num_classes,
    ignore_index=255,
    class_weights=None,
    eps=1e-6,
):
    valid = target != ignore_index
    safe_target = torch.where(valid, target, torch.zeros_like(target))

    prob = F.softmax(logits.float(), dim=1)

    onehot = F.one_hot(
        safe_target,
        num_classes=num_classes,
    ).permute(0, 3, 1, 2).float()

    valid = valid.unsqueeze(1)

    prob = prob * valid
    onehot = onehot * valid

    inter = (prob * onehot).sum(dim=(0, 2, 3))
    union = prob.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))

    dice = (2 * inter + eps) / (union + eps)
    loss_per_class = 1.0 - dice

    if class_weights is None:
        return loss_per_class.mean()

    w = class_weights.to(logits.device).float()

    return (loss_per_class * w).sum() / w.sum().clamp_min(eps)


def build_class_presence_target(label, num_classes, ignore_index=255):
    bsz = label.shape[0]
    device = label.device

    target = torch.zeros(
        (bsz, num_classes),
        dtype=torch.float32,
        device=device,
    )

    for b in range(bsz):
        valid = label[b] != ignore_index

        if valid.sum().item() == 0:
            continue

        present = torch.unique(label[b][valid])
        present = present[(present >= 0) & (present < num_classes)]

        if present.numel() > 0:
            target[b, present.long()] = 1.0

    return target


def compute_pred_loss(
    masks,
    class_logits_img,
    label,
    num_classes,
    ignore_index,
    ce_w,
    dice_w,
    cls_presence_w,
    class_weights,
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

    ce_weight = None if class_weights is None else class_weights.to(masks_up.device).float()

    ce = F.cross_entropy(
        masks_up,
        label,
        ignore_index=ignore_index,
        weight=ce_weight,
    )

    dice = dice_loss_from_logits(
        masks_up,
        label,
        num_classes,
        ignore_index,
        class_weights=class_weights,
    )

    presence_target = build_class_presence_target(
        label,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    cls_presence = F.binary_cross_entropy_with_logits(
        class_logits_img.float(),
        presence_target,
    )

    loss = ce_w * ce + dice_w * dice + cls_presence_w * cls_presence

    stat = {
        "ce": ce.detach(),
        "dice_loss": dice.detach(),
        "cls_presence": cls_presence.detach(),
    }

    return loss, stat, masks_up.detach()


def make_forward_loss(
    model,
    cfg,
    num_classes,
    ignore_index,
    class_weights,
    amp_dtype,
    use_amp,
):
    ce_w = float(cfg.get("loss_ce_weight", 0.40))
    dice_w = float(cfg.get("loss_dice_weight", 0.60))
    cls_presence_w = float(cfg.get("loss_cls_presence_weight", 0.01))
    aux_w = float(cfg.get("aux_pred_weight", 0.05))

    def forward_and_loss(img, label, amp_on=True):
        with autocast_context(img.device, amp_on and use_amp, amp_dtype):
            out = model(img)
            masks, class_logits_img, aux_outputs = unpack_model_output(model, out)

        final_loss, stat, masks_up = compute_pred_loss(
            masks=masks,
            class_logits_img=class_logits_img,
            label=label,
            num_classes=num_classes,
            ignore_index=ignore_index,
            ce_w=ce_w,
            dice_w=dice_w,
            cls_presence_w=cls_presence_w,
            class_weights=class_weights,
        )

        total_loss = final_loss

        aux_ce = torch.zeros((), device=masks.device, dtype=torch.float32)
        aux_dice = torch.zeros((), device=masks.device, dtype=torch.float32)
        aux_presence = torch.zeros((), device=masks.device, dtype=torch.float32)
        n_aux = 0

        if aux_outputs is not None and aux_w > 0:
            for pred in aux_outputs[:-1]:
                if not isinstance(pred, dict) or "masks" not in pred:
                    continue

                pred_class_logits_img = pred.get("class_logits_img", class_logits_img)

                li, si, _ = compute_pred_loss(
                    masks=pred["masks"],
                    class_logits_img=pred_class_logits_img,
                    label=label,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    ce_w=ce_w,
                    dice_w=dice_w,
                    cls_presence_w=cls_presence_w,
                    class_weights=class_weights,
                )

                total_loss = total_loss + aux_w * li

                aux_ce += si["ce"]
                aux_dice += si["dice_loss"]
                aux_presence += si["cls_presence"]
                n_aux += 1

        if n_aux > 0:
            aux_ce /= n_aux
            aux_dice /= n_aux
            aux_presence /= n_aux

        stat.update(
            {
                "aux_ce": aux_ce,
                "aux_dice_loss": aux_dice,
                "aux_cls_presence": aux_presence,
                "n_aux_used": n_aux,
            }
        )

        return total_loss, stat, masks_up

    return forward_and_loss


@torch.no_grad()
def update_ema(ema_model, model, decay, copy_buffers=False):
    """Update only trainable parameters; cache parameter maps to reduce per-step overhead."""
    source_trainable = getattr(model, "_ema_source_trainable", None)
    if source_trainable is None:
        source_trainable = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        model._ema_source_trainable = source_trainable

    ema_params = getattr(ema_model, "_ema_param_map", None)
    if ema_params is None:
        ema_params = dict(ema_model.named_parameters())
        ema_model._ema_param_map = ema_params

    for name, src in source_trainable:
        dst = ema_params[name]
        if torch.is_floating_point(dst):
            dst.mul_(decay).add_(src.detach(), alpha=1.0 - decay)
        else:
            dst.copy_(src.detach())

    if copy_buffers:
        source_buffers = getattr(model, "_ema_source_buffers", None)
        if source_buffers is None:
            source_buffers = list(model.named_buffers())
            model._ema_source_buffers = source_buffers

        ema_buffers = getattr(ema_model, "_ema_buffer_map", None)
        if ema_buffers is None:
            ema_buffers = dict(ema_model.named_buffers())
            ema_model._ema_buffer_map = ema_buffers

        for name, src in source_buffers:
            if name in ema_buffers:
                ema_buffers[name].copy_(src.detach())


def save_checkpoint(
    path,
    model,
    ema_model,
    optimizer,
    scaler,
    epoch,
    best_metric,
    cfg,
    dataset_name,
    best_metric_name,
):
    use_ema = ema_model is not None and bool(cfg.get("ema_save_as_model", True))

    ckpt = {
        "epoch": int(epoch),
        "model": (ema_model if use_ema else model).state_dict(),
        "raw_model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_metric": float(best_metric),
        "best_metric_name": str(best_metric_name),
        "dataset_name": dataset_name,
        "task_name": TASK_NAME,
        "num_classes": int(cfg["num_classes"]),
        "ignore_index": int(cfg.get("ignore_index", 255)),
        "class_names": CLASS_NAMES_GLASSAI,
        "cfg": dict(cfg),
        "ema_saved_as_model": bool(use_ema),
        "ablation_mode": ABLATION_MODE,
        "pretrained_backbone_fully_frozen": True,
    }

    torch.save(ckpt, path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scaler=None,
    device="cpu",
    strict=True,
    expected_num_classes=6,
):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict):
        ckpt_num_classes = ckpt.get("num_classes", None)

        if ckpt_num_classes is not None and int(ckpt_num_classes) != int(expected_num_classes):
            raise RuntimeError(
                f"[ERROR] Checkpoint num_classes={ckpt_num_classes}, "
                f"but current GlassAI task requires num_classes={expected_num_classes}. "
                f"Please do not resume from an incompatible checkpoint."
            )

    if isinstance(ckpt, dict) and "raw_model" in ckpt:
        state = ckpt["raw_model"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=strict)

    if optimizer is not None and isinstance(ckpt, dict) and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and isinstance(ckpt, dict) and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return {
        "start_epoch": int(ckpt.get("epoch", -1)) + 1 if isinstance(ckpt, dict) else 0,
        "best_metric": float(ckpt.get("best_metric", -1.0)) if isinstance(ckpt, dict) else -1.0,
    }


def count_label_values(dataset, indices, ignore_index=255, max_scan=50):
    scan_indices = list(indices)[: min(len(indices), max_scan)]
    counter = {}

    for idx in scan_indices:
        _, label = dataset[idx]
        vals = torch.unique(label.long())

        for v in vals.tolist():
            counter[int(v)] = counter.get(int(v), 0) + 1

    print(f"[INFO] Label value quick check from {len(scan_indices)} samples:")
    for k in sorted(counter.keys()):
        meaning = "Ignore/Other" if k == ignore_index else CLASS_NAMES_GLASSAI.get(k, "Unexpected")
        print(f"       label={k}: appears in {counter[k]} scanned mask(s), meaning={meaning}")

    unexpected = [
        k
        for k in counter.keys()
        if k != ignore_index and k not in CLASS_NAMES_GLASSAI
    ]

    if unexpected:
        print(
            f"[WARN] Unexpected label values detected: {unexpected}. "
            f"For this task, valid labels should be 0..5 and {ignore_index}."
        )


def main():
    cfg = load_yaml_any_encoding("configs/train_config_glassai.yaml")
    cfg = enforce_glassai_protocol(cfg)

    cfg["lr"] = float(cfg["lr"])
    cfg["freeze_pretrained_backbone"] = True
    cfg.setdefault("train_amp", True)
    cfg.setdefault("eval_amp", True)
    cfg.setdefault("allow_tf32", True)
    cfg.setdefault("fused_adamw", True)
    cfg.setdefault("tqdm_auto_disable_non_tty", True)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_fast_runtime(device, cfg)

    print("=" * 100)
    print("[INFO] Starting pretrained-backbone FULL-FREEZE ablation training")
    print("[INFO] Task protocol:")
    for k, v in CLASS_NAMES_GLASSAI.items():
        print(f"       Class {k} = {v}")
    print("       Label 255 = Unknown color / Ignored pixel")
    print("[INFO] Pixels with label=255 do NOT participate in CE loss, Dice loss, presence loss, or metrics.")
    print("=" * 100)

    print(f"[INFO] Using device: {device}")

    dataset, _, dataset_name = build_dataset_from_cfg(cfg)
    print(f"[INFO] Using dataset: {dataset_name}, size={len(dataset)}")

    ignore_index = int(cfg.get("ignore_index", 255))
    num_classes = int(cfg["num_classes"])

    print(f"[INFO] num_classes={num_classes}, ignore_index={ignore_index}")
    print(f"[INFO] Expected output channels: {num_classes}")
    print("[INFO] Metric protocol: mean_dice / mean_iou = average over all 6 GlassAI classes.")

    train_idx, val_idx, test_idx = split_indices(
        len(dataset),
        val_test_split=float(cfg.get("val_test_split", 0.3)),
        test_split=float(cfg.get("test_split", 0.5)),
        seed=seed,
    )

    splits_dir = os.path.join(
        str(cfg.get("splits_root", "splits")),
        dataset_name,
    )

    save_split_files(dataset, train_idx, val_idx, test_idx, splits_dir)

    print(f"[INFO] Split dir: {splits_dir}")
    print(f"[INFO] Train/Val/Test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"[INFO] Saved splits -> {splits_dir}/train.txt, val.txt, test.txt")

    count_label_values(
        dataset=dataset,
        indices=train_idx,
        ignore_index=ignore_index,
        max_scan=int(cfg.get("label_quick_check_max_scan", 50)),
    )

    train_ds = SubsetWithPaths(
        dataset,
        train_idx,
        train_aug=bool(cfg.get("train_aug_enable", True)),
        cfg=cfg,
    )

    val_ds = SubsetWithPaths(
        dataset,
        val_idx,
        train_aug=False,
        cfg=cfg,
    )

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

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        drop_last=bool(cfg.get("train_drop_last", False)),
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("val_batch_size", cfg["batch_size"])),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    print(
        f"[INFO] DataLoader: workers={num_workers}, pin_memory={pin_memory}, "
        f"persistent_workers={persistent_workers}"
    )

    class_weights = estimate_class_weights(
        dataset,
        train_idx,
        num_classes,
        ignore_index,
        cfg,
    )

    model = build_model(cfg, device)

    frozen_backbone_params = set_backbone_trainable(model, trainable=False)
    backbone = get_backbone_module(model)
    if backbone is None:
        raise RuntimeError("[ERROR] Full-freeze ablation requires a detectable ViT backbone.")
    backbone.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[INFO] Full-freeze ablation active: frozen backbone params={frozen_backbone_params:,}; "
        f"trainable head/query params={trainable_params:,}/{total_params:,}."
    )

    optimizer = make_optimizer(model, cfg)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    backbone_param_ids = {id(p) for p in backbone.parameters()}
    optimizer_param_ids = {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    overlap = backbone_param_ids & optimizer_param_ids
    if overlap:
        raise RuntimeError(
            "[ERROR] Frozen backbone parameters unexpectedly entered the optimizer."
        )
    print("[INFO] Freeze verification passed: optimizer contains no backbone parameters.")

    use_amp = bool(cfg.get("train_amp", True)) and device.type == "cuda"
    amp_dtype = _pick_amp_dtype(cfg) if use_amp else torch.float16

    scaler = make_grad_scaler(
        enabled=(use_amp and amp_dtype == torch.float16)
    )

    forward_and_loss = make_forward_loss(
        model,
        cfg,
        num_classes,
        ignore_index,
        class_weights,
        amp_dtype,
        use_amp,
    )

    use_ema = bool(cfg.get("ema_enable", True))
    ema_model = None

    if use_ema:
        ema_model = deepcopy(model).eval()

        for p in ema_model.parameters():
            p.requires_grad_(False)

        print(f"[INFO] EMA enabled: decay={float(cfg.get('ema_decay', 0.999))}")
    else:
        print("[INFO] EMA disabled.")

    ckpt_dir = str(cfg.get("ckpt_dir", "checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)

    best_ckpt = os.path.join(
        ckpt_dir,
        f"{dataset_name}_6class_pretrained_frozen_best.pth",
    )

    last_ckpt = os.path.join(
        ckpt_dir,
        f"{dataset_name}_6class_pretrained_frozen_last.pth",
    )

    print("[INFO] Checkpoint directory is unchanged:")
    print(f"       ckpt_dir = {ckpt_dir}")
    print("[INFO] Checkpoint filenames are isolated for the full-freeze ablation:")
    print(f"       best_ckpt = {best_ckpt}")
    print(f"       last_ckpt = {last_ckpt}")

    metric_for_best = str(cfg.get("metric_for_best", "mean_dice")).lower()

    if metric_for_best not in ("mean_dice", "mean_iou"):
        print(f"[WARN] unsupported metric_for_best={metric_for_best}; fallback to mean_dice")
        metric_for_best = "mean_dice"

    start_epoch = 0
    best_metric = -1.0

    resume_path = str(cfg.get("resume", "")).strip()

    if resume_path and os.path.exists(resume_path):
        print(f"[INFO] Trying to resume from: {resume_path}")

        info = load_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            strict=bool(cfg.get("resume_strict", True)),
            expected_num_classes=num_classes,
        )

        start_epoch = info["start_epoch"]
        best_metric = info["best_metric"]

        if ema_model is not None:
            ema_model.load_state_dict(model.state_dict(), strict=True)

        print(
            f"[INFO] Resume from {resume_path}: "
            f"start_epoch={start_epoch}, best={best_metric:.4f}"
        )

    elif resume_path:
        print(f"[WARN] resume path not found: {resume_path}; training from scratch")
    else:
        print("[INFO] No resume checkpoint provided. Training from scratch.")

    epochs = int(cfg["epochs"])
    warmup_epochs = int(cfg.get("warmup_epochs", 4))
    grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))
    early_stop_patience = int(cfg.get("early_stop_patience", 8))
    min_delta = float(cfg.get("early_stop_min_delta", 1e-4))
    lr_patience = int(cfg.get("lr_reduce_patience", 3))
    lr_factor = float(cfg.get("lr_reduce_factor", 0.5))

    ce_w = float(cfg.get("loss_ce_weight", 0.40))
    dice_w = float(cfg.get("loss_dice_weight", 0.60))
    cls_presence_w = float(cfg.get("loss_cls_presence_weight", 0.01))
    aux_w = float(cfg.get("aux_pred_weight", 0.05))

    print("=" * 100)
    print("[INFO] Training configuration summary")
    print(f"[INFO] Dataset name: {dataset_name}")
    print(f"[INFO] Task name: {TASK_NAME}")
    print(f"[INFO] Epochs: {epochs}")
    print(f"[INFO] Batch size: train={int(cfg['batch_size'])}, val={int(cfg.get('val_batch_size', cfg['batch_size']))}")
    print(f"[INFO] Loss weights: CE={ce_w}, Dice={dice_w}, ClsPresence={cls_presence_w}, AuxPred={aux_w}")
    print(f"[INFO] Optimizer lr: {format_lrs(optimizer)}")
    print(f"[INFO] Weight decay: {float(cfg.get('weight_decay', 0.12))}")
    print(f"[INFO] pretrained backbone: fully frozen for all {epochs} epochs; warmup_epochs={warmup_epochs}")
    print(f"[INFO] best metric: {metric_for_best}, early_stop_patience={early_stop_patience}")
    print(f"[INFO] AMP {'enabled' if use_amp else 'disabled'}, dtype={amp_dtype}")
    print(f"[INFO] Train augmentation: {bool(cfg.get('train_aug_enable', True))}")
    print(
        "[INFO] Aug settings: "
        f"hflip={bool(cfg.get('aug_hflip', True))}, "
        f"vflip={bool(cfg.get('aug_vflip', True))}, "
        f"rotate90={bool(cfg.get('aug_rotate90', True))}, "
        f"noise_std={float(cfg.get('aug_noise_std', 0.0))}"
    )
    print("=" * 100)

    global_step = 0
    bad_epochs = 0
    plateau_epochs = 0
    progress_update_every = max(1, int(cfg.get("tqdm_update_every", 20)))
    ema_buffer_interval = max(1, int(cfg.get("ema_buffer_interval", 100)))

    for epoch in range(start_epoch, epochs):
        model.train()
        backbone = get_backbone_module(model)
        if backbone is None:
            raise RuntimeError("[ERROR] Backbone disappeared after model construction.")
        backbone.eval()

        set_warmup_lr(optimizer, epoch, warmup_epochs)

        conf_mat = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64, device=device
        )

        loss_sum = torch.zeros((), device=device)
        ce_sum = torch.zeros((), device=device)
        dice_sum = torch.zeros((), device=device)
        cls_presence_sum = torch.zeros((), device=device)
        aux_ce_sum = torch.zeros((), device=device)
        aux_dice_sum = torch.zeros((), device=device)
        aux_presence_sum = torch.zeros((), device=device)

        update_steps = 0
        skipped_all_ignore = 0
        skipped_nonfinite = 0
        fallback_fp32 = 0

        total_pixels = 0
        valid_pixels = 0
        ignored_pixels = 0
        bad_examples = []

        prog = make_progress(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            cfg=cfg,
            leave=False,
            total=len(train_loader),
        )

        for batch_idx, batch in enumerate(prog, start=1):
            global_step += 1

            img, label, _, lbl_paths = batch

            label_cpu = label.long()
            valid_mask = label_cpu != ignore_index

            v = int(valid_mask.sum().item())
            total = int(label_cpu.numel())
            ignored = total - v

            total_pixels += total
            valid_pixels += v
            ignored_pixels += ignored

            if v == 0:
                skipped_all_ignore += 1

                if len(bad_examples) < 5:
                    bad_examples.append(lbl_paths[0])

                continue

            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            loss, stat, masks_up = forward_and_loss(
                img,
                label,
                amp_on=True,
            )

            if use_amp and not torch.isfinite(loss):
                loss, stat, masks_up = forward_and_loss(
                    img,
                    label,
                    amp_on=False,
                )
                fallback_fp32 += 1

            if not torch.isfinite(loss):
                skipped_nonfinite += 1

                if len(bad_examples) < 5:
                    bad_examples.append(lbl_paths[0])

                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()

                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=grad_clip_norm,
                    )

                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()

                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=grad_clip_norm,
                    )

                optimizer.step()

            if ema_model is not None:
                update_ema(
                    ema_model,
                    model,
                    decay=float(cfg.get("ema_decay", 0.999)),
                    copy_buffers=(global_step % ema_buffer_interval == 0),
                )

            pred = masks_up.argmax(dim=1)
            update_confusion_matrix(
                conf_mat, pred, label, num_classes, ignore_index
            )

            loss_sum.add_(loss.detach())
            ce_sum.add_(stat["ce"])
            dice_sum.add_(stat["dice_loss"])
            cls_presence_sum.add_(stat["cls_presence"])
            aux_ce_sum.add_(stat["aux_ce"])
            aux_dice_sum.add_(stat["aux_dice_loss"])
            aux_presence_sum.add_(stat["aux_cls_presence"])

            update_steps += 1

            if batch_idx % progress_update_every == 0 or batch_idx == len(train_loader):
                prog.set_postfix_str(
                    f"loss={float(loss_sum.item()) / max(1, update_steps):.4f}"
                )

        if update_steps == 0:
            print("[FATAL] No valid optimizer update in this epoch.")
            print("[FATAL] This usually means all pixels are ignore_index=255.")

            for p in bad_examples:
                print("  -", p)

            break

        train_metrics = summarize_confmat(conf_mat.cpu())

        avg_loss = float(loss_sum.item()) / update_steps
        avg_ce = float(ce_sum.item()) / update_steps
        avg_dice_loss = float(dice_sum.item()) / update_steps
        avg_cls_presence = float(cls_presence_sum.item()) / update_steps
        avg_aux_ce = float(aux_ce_sum.item()) / update_steps
        avg_aux_dice = float(aux_dice_sum.item()) / update_steps
        avg_aux_presence = float(aux_presence_sum.item()) / update_steps

        eval_model = ema_model if ema_model is not None else model
        if ema_model is not None:
            update_ema(ema_model, model, decay=1.0, copy_buffers=True)

        val_metrics = evaluate_dataset_metrics(
            eval_model,
            val_loader,
            device,
            num_classes,
            ignore_index,
            bool(cfg.get("eval_amp", True)),
            amp_dtype,
        )

        current_metric = float(val_metrics[metric_for_best])

        valid_ratio = valid_pixels / max(1, total_pixels)
        ignored_ratio = ignored_pixels / max(1, total_pixels)

        print(
            f"[Epoch {epoch + 1}] "
            f"Loss={avg_loss:.4f} | "
            f"Train_Dice={train_metrics['mean_dice']:.4f} | "
            f"Train_IoU={train_metrics['mean_iou']:.4f} | "
            f"Val_Dice={val_metrics['mean_dice']:.4f} | "
            f"Val_IoU={val_metrics['mean_iou']:.4f} | "
            f"CE={avg_ce:.4f} | "
            f"DiceLoss={avg_dice_loss:.4f} | "
            f"ClsPresence={avg_cls_presence:.4f} | "
            f"AuxCE={avg_aux_ce:.4f} | "
            f"AuxDiceLoss={avg_aux_dice:.4f} | "
            f"AuxClsPresence={avg_aux_presence:.4f} | "
            f"UpdateSteps={update_steps}/{len(train_loader)} | "
            f"TrainValidRatio={valid_ratio:.4f} | "
            f"TrainIgnoreRatio={ignored_ratio:.4f} | "
            f"SkipAllIgnore={skipped_all_ignore} | "
            f"SkipNonFinite={skipped_nonfinite} | "
            f"FallbackFP32={fallback_fp32} | "
            f"LR={format_lrs(optimizer)}"
        )

        print(
            "[TRAIN per-class Dice] "
            + format_per_class_metric(train_metrics["per_dice"], "C")
        )

        print(
            "[TRAIN per-class IoU] "
            + format_per_class_metric(train_metrics["per_iou"], "C")
        )

        print(
            "[VAL per-class Dice] "
            + format_per_class_metric(val_metrics["per_dice"], "C")
        )

        print(
            "[VAL per-class IoU] "
            + format_per_class_metric(val_metrics["per_iou"], "C")
        )

        improved = current_metric > best_metric + min_delta

        if improved:
            best_metric = current_metric
            bad_epochs = 0
            plateau_epochs = 0

            save_checkpoint(
                best_ckpt,
                model,
                ema_model,
                optimizer,
                scaler,
                epoch,
                best_metric,
                cfg,
                dataset_name,
                metric_for_best,
            )

            print(
                f"[SAVE] best checkpoint -> {best_ckpt}, "
                f"{metric_for_best}={best_metric:.4f}"
            )

        else:
            bad_epochs += 1
            plateau_epochs += 1

            print(
                f"[INFO] No improvement: "
                f"bad_epochs={bad_epochs}/{early_stop_patience}, "
                f"plateau={plateau_epochs}/{lr_patience}"
            )

        if plateau_epochs >= lr_patience and epoch >= warmup_epochs:
            if reduce_lr_on_plateau(optimizer, factor=lr_factor):
                print(f"[INFO] ReduceLROnPlateau: lr -> {format_lrs(optimizer)}")

            plateau_epochs = 0

        if bool(cfg.get("save_last_ckpt", True)):
            save_checkpoint(
                last_ckpt,
                model,
                ema_model,
                optimizer,
                scaler,
                epoch,
                best_metric,
                cfg,
                dataset_name,
                metric_for_best,
            )

            print(f"[SAVE] last checkpoint -> {last_ckpt}")

        if bool(cfg.get("save_each_epoch", False)):
            epoch_ckpt = os.path.join(
                ckpt_dir,
                f"{dataset_name}_6class_pretrained_frozen_epoch_{epoch + 1:03d}.pth",
            )

            save_checkpoint(
                epoch_ckpt,
                model,
                ema_model,
                optimizer,
                scaler,
                epoch,
                best_metric,
                cfg,
                dataset_name,
                metric_for_best,
            )

            print(f"[SAVE] epoch checkpoint -> {epoch_ckpt}")

        if early_stop_patience > 0 and bad_epochs >= early_stop_patience:
            print(
                f"[EARLY STOP] {metric_for_best} did not improve for "
                f"{early_stop_patience} epochs. Best={best_metric:.4f}"
            )
            break

    print("=" * 100)
    print("[INFO] Full-freeze ablation training finished.")
    print(f"[INFO] Best checkpoint: {best_ckpt}")
    print(f"[INFO] Last checkpoint: {last_ckpt}")
    print(f"[INFO] Best {metric_for_best}={best_metric:.4f}")
    print(f"[INFO] Test set kept for evaluate.py: {splits_dir}/test.txt")
    print("[INFO] Reminder: evaluate.py / predict.py should also use num_classes=6 and ignore_index=255.")
    print("=" * 100)


if __name__ == "__main__":
    main()