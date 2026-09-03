import os
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


CLASS_NAMES_2CLASS = {
    0: "Tumor",
    1: "Normal",
}

TASK_NAME = "2class_ignore_other"


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


def enforce_2class_ignore_protocol(cfg):
    """
    方案 A：
    Class 0 = 肿瘤区域
    Class 1 = 正常区域
    ignore_index = 255 = 其他区域 / 背景 / 不参与训练区域
    """
    old_num_classes = int(cfg.get("num_classes", 2))
    old_ignore_index = int(cfg.get("ignore_index", 255))

    if old_num_classes != 2:
        print(
            f"[WARN] num_classes in config is {old_num_classes}, "
            f"but 2-class ignore-other protocol requires num_classes=2. "
            f"Override num_classes -> 2."
        )

    if old_ignore_index != 255:
        print(
            f"[WARN] ignore_index in config is {old_ignore_index}. "
            f"This script is designed for ignore_index=255. "
            f"Override ignore_index -> 255."
        )

    cfg["num_classes"] = 2
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
        from datasets.glassai_dataset import GlassAIDataset, FIXED_PALETTE

        ds = GlassAIDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            palette=FIXED_PALETTE,
            resize_hw=resize_hw,
            ignore_index=ignore_index,
            strict_match=bool(cfg.get("glassai_strict_match", True)),
        )

        return ds, FIXED_PALETTE, dataset_name

    if dataset_name == "lungcancer":
        from datasets.lungcancer_dataset import LungCancerDataset, FIXED_PALETTE

        ds = LungCancerDataset(
            image_input=image_dir,
            label_input=label_dir,
            palette=FIXED_PALETTE,
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

    num_classes = conf_mat.shape[0]

    if num_classes == 2:
        mean_dice = float(per_dice.mean().item())
        mean_iou = float(per_iou.mean().item())
    elif num_classes > 2:
        mean_dice = float(per_dice[:2].mean().item())
        mean_iou = float(per_iou[:2].mean().item())
    else:
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
        class_name = CLASS_NAMES_2CLASS.get(i, f"Class{i}")
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

    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    amp_on = bool(use_amp) and device.type == "cuda"

    for batch in loader:
        img, label = batch[0], batch[1]

        img = img.to(device, non_blocking=True)
        label_cpu = label.long()

        ctx = (
            torch.cuda.amp.autocast(dtype=amp_dtype)
            if amp_on
            else contextlib.nullcontext()
        )

        with ctx:
            out = model(img)
            masks = out[0]

            if masks.shape[-2:] != label_cpu.shape[-2:]:
                masks = F.interpolate(
                    masks,
                    size=label_cpu.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        pred_cpu = masks.argmax(dim=1).detach().cpu().long()

        for i in range(pred_cpu.size(0)):
            conf_mat = update_confusion_matrix(
                conf_mat,
                pred_cpu[i],
                label_cpu[i],
                num_classes,
                ignore_index,
            )

    if was_training:
        model.train()

    return summarize_confmat(conf_mat)


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
    """
    Freeze / unfreeze the ViT backbone.

    Important for ablation experiments:
    - When trainable=False, every parameter inside model.vit_model/backbone/vit is set
      to requires_grad=False.
    - The backbone is also switched to eval() so Dropout / stochastic layers do not
      introduce training-time randomness while the pretrained representation is frozen.
    """
    backbone = get_backbone_module(model)

    if backbone is None:
        print("[WARN] Could not find backbone module to freeze/unfreeze.")
        return 0

    n_params = 0

    for p in backbone.parameters():
        p.requires_grad_(bool(trainable))
        n_params += p.numel()

    if trainable:
        backbone.train()
    else:
        backbone.eval()

    return n_params


def get_backbone_trainable_stats(model):
    backbone = get_backbone_module(model)

    if backbone is None:
        return {
            "found": False,
            "total_params": 0,
            "trainable_params": 0,
            "total_tensors": 0,
            "trainable_tensors": 0,
        }

    total_params = 0
    trainable_params = 0
    total_tensors = 0
    trainable_tensors = 0

    for p in backbone.parameters():
        total_tensors += 1
        total_params += p.numel()

        if p.requires_grad:
            trainable_tensors += 1
            trainable_params += p.numel()

    return {
        "found": True,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "total_tensors": int(total_tensors),
        "trainable_tensors": int(trainable_tensors),
    }


def assert_backbone_frozen(model, where=""):
    stats = get_backbone_trainable_stats(model)

    if not stats["found"]:
        print("[WARN] Could not verify frozen backbone because backbone module was not found.")
        return

    if stats["trainable_params"] != 0:
        suffix = f" at {where}" if where else ""
        raise RuntimeError(
            "[ERROR] ViT backbone is expected to be fully frozen"
            f"{suffix}, but {stats['trainable_params']} / {stats['total_params']} "
            "backbone parameters still have requires_grad=True."
        )


def print_trainable_parameter_summary(model):
    total = 0
    trainable = 0

    for p in model.parameters():
        total += p.numel()

        if p.requires_grad:
            trainable += p.numel()

    bb = get_backbone_trainable_stats(model)

    print(
        "[INFO] Trainable parameter summary: "
        f"model_trainable={trainable}/{total}, "
        f"backbone_trainable={bb['trainable_params']}/{bb['total_params']} "
        f"({bb['trainable_tensors']}/{bb['total_tensors']} tensors)"
    )


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


def make_optimizer(model, cfg):
    base_lr = float(cfg["lr"])
    backbone_lr_mult = float(cfg.get("backbone_lr_mult", 0.0))
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
        groups.append(
            {
                "params": backbone_params,
                "lr": base_lr * backbone_lr_mult,
                "name": "backbone",
            }
        )

    if joint_params:
        groups.append(
            {
                "params": joint_params,
                "lr": base_lr * joint_lr_mult,
                "name": "joint",
            }
        )

    if head_params:
        groups.append(
            {
                "params": head_params,
                "lr": base_lr,
                "name": "head",
            }
        )

    if not groups:
        raise RuntimeError(
            "[ERROR] No trainable parameters were found for the optimizer. "
            "If the ViT backbone is frozen, make sure the decoder/head/query modules remain trainable."
        )

    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)

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
    print("[INFO] Valid classes for weight scan: 0=Tumor, 1=Normal")
    print(f"[INFO] Pixels with label={ignore_index} are ignored during class-weight scan.")

    for idx in tqdm(scan_indices, desc="ClassWeightScan", ncols=100):
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
        class_name = CLASS_NAMES_2CLASS.get(i, f"Class{i}")
        print(f"       C{i}({class_name}) = {int(counts[i].item())}")

    ignored_note = (
        "Other/background pixels should have label=255 and are not included above."
    )
    print(f"[INFO] {ignored_note}")

    print("[INFO] class loss weights:")
    for i in range(num_classes):
        class_name = CLASS_NAMES_2CLASS.get(i, f"Class{i}")
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
        "ce": float(ce.item()),
        "dice_loss": float(dice.item()),
        "cls_presence": float(cls_presence.item()),
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
        ctx = (
            torch.cuda.amp.autocast(dtype=amp_dtype)
            if amp_on and use_amp
            else contextlib.nullcontext()
        )

        with ctx:
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

        aux_ce = 0.0
        aux_dice = 0.0
        aux_presence = 0.0
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
def update_ema(ema_model, model, decay):
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()

    for k, v in ema_state.items():
        src = model_state[k]

        if torch.is_floating_point(v):
            v.mul_(decay).add_(src.detach(), alpha=1.0 - decay)
        else:
            v.copy_(src)


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
        "class_names": CLASS_NAMES_2CLASS,
        "cfg": dict(cfg),
        "ema_saved_as_model": bool(use_ema),
    }

    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu", strict=True):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict):
        ckpt_num_classes = ckpt.get("num_classes", None)

        if ckpt_num_classes is not None and int(ckpt_num_classes) != 2:
            raise RuntimeError(
                f"[ERROR] Checkpoint num_classes={ckpt_num_classes}, "
                f"but current task requires num_classes=2. "
                f"Please do not resume from an old 3-class checkpoint."
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
    """
    只用于训练开始时做轻量检查。
    不参与训练。
    """
    scan_indices = list(indices)[: min(len(indices), max_scan)]
    counter = {}

    for idx in scan_indices:
        _, label = dataset[idx]
        vals = torch.unique(label.long())

        for v in vals.tolist():
            counter[int(v)] = counter.get(int(v), 0) + 1

    print(f"[INFO] Label value quick check from {len(scan_indices)} samples:")
    for k in sorted(counter.keys()):
        meaning = "Ignore/Other" if k == ignore_index else CLASS_NAMES_2CLASS.get(k, "Unexpected")
        print(f"       label={k}: appears in {counter[k]} scanned mask(s), meaning={meaning}")

    unexpected = [
        k
        for k in counter.keys()
        if k != ignore_index and k not in CLASS_NAMES_2CLASS
    ]

    if unexpected:
        print(
            f"[WARN] Unexpected label values detected: {unexpected}. "
            f"For this task, valid labels should be 0, 1, and {ignore_index}."
        )


def main():
    cfg = load_yaml_any_encoding("configs/train_config.yaml")
    cfg = enforce_2class_ignore_protocol(cfg)

    cfg["lr"] = float(cfg["lr"])

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("[INFO] Starting training with 2-class ignore-other protocol")
    print("[INFO] Task protocol:")
    print("       Class 0 = Tumor")
    print("       Class 1 = Normal")
    print("       Label 255 = Other / Background / Ignored region")
    print("[INFO] Other / ignored regions do NOT participate in CE loss, Dice loss, presence loss, or metrics.")
    print("=" * 100)

    print(f"[INFO] Using device: {device}")

    dataset, _, dataset_name = build_dataset_from_cfg(cfg)
    print(f"[INFO] Using dataset: {dataset_name}, size={len(dataset)}")

    ignore_index = int(cfg.get("ignore_index", 255))
    num_classes = int(cfg["num_classes"])

    print(f"[INFO] num_classes={num_classes}, ignore_index={ignore_index}")
    print("[INFO] Expected output channels: 2")
    print("[INFO] Metric protocol: mean_dice / mean_iou = average of Tumor and Normal only.")

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

    loader_kwargs = dict(
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_memory", False)),
        drop_last=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("val_batch_size", cfg["batch_size"])),
        shuffle=False,
        **loader_kwargs,
    )

    class_weights = estimate_class_weights(
        dataset,
        train_idx,
        num_classes,
        ignore_index,
        cfg,
    )

    epochs = int(cfg["epochs"])

    # ==========================================================================
    # ViT backbone freezing for ablation
    # --------------------------------------------------------------------------
    # 默认开启全程冻结。
    # 目的：保证 ViT 预训练权重不参与反向传播、不进入优化器、不被 AdamW 更新。
    #
    # 如果你之后想跑“不冻结 ViT”的对照组，在 configs/train_config.yaml 中加入：
    # freeze_vit_backbone: false
    # freeze_backbone_all_training: false
    # backbone_lr_mult: 0.03
    #
    # 如果要严格冻结 ViT，保持：
    # freeze_vit_backbone: true
    # freeze_backbone_all_training: true
    # backbone_lr_mult: 0.0
    # ==========================================================================
    freeze_backbone_all_training = bool(cfg.get("freeze_backbone_all_training", True))

    if bool(cfg.get("freeze_vit_backbone", True)):
        freeze_backbone_all_training = True

    freeze_backbone_epochs = int(
        cfg.get(
            "freeze_backbone_epochs",
            epochs if freeze_backbone_all_training else 8,
        )
    )

    model = build_model(cfg, device)

    # 严格消融关键点：
    # 必须在 make_optimizer 之前冻结 backbone。
    # 这样 frozen ViT 参数不会进入 optimizer param_groups。
    if freeze_backbone_all_training:
        n_backbone = set_backbone_trainable(model, trainable=False)
        assert_backbone_frozen(model, where="before optimizer creation")
        print(
            f"[INFO] Full-training ViT backbone freeze enabled: "
            f"{n_backbone} backbone params are frozen before optimizer creation."
        )
    else:
        print("[INFO] Full-training ViT backbone freeze disabled.")

    print_trainable_parameter_summary(model)

    optimizer = make_optimizer(model, cfg)

    use_amp = bool(cfg.get("train_amp", False)) and device.type == "cuda"
    amp_dtype = _pick_amp_dtype(cfg) if use_amp else torch.float16

    scaler = torch.cuda.amp.GradScaler(
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
        f"{dataset_name}_2class_ignore_other_best.pth",
    )

    last_ckpt = os.path.join(
        ckpt_dir,
        f"{dataset_name}_2class_ignore_other_last.pth",
    )

    print("[INFO] Checkpoint directory is unchanged:")
    print(f"       ckpt_dir = {ckpt_dir}")
    print("[INFO] Checkpoint filenames are changed for 2-class ignore-other training:")
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

        resume_optimizer = bool(cfg.get("resume_optimizer", True))

        if freeze_backbone_all_training and not bool(
            cfg.get("resume_optimizer_when_backbone_frozen", False)
        ):
            resume_optimizer = False
            print(
                "[INFO] Full-training backbone freeze is enabled; "
                "skip loading optimizer state by default to avoid old param-group mismatch."
            )

        info = load_checkpoint(
            resume_path,
            model,
            optimizer=optimizer if resume_optimizer else None,
            scaler=scaler,
            device=device,
            strict=bool(cfg.get("resume_strict", True)),
        )

        if freeze_backbone_all_training:
            set_backbone_trainable(model, trainable=False)
            assert_backbone_frozen(model, where="after resume checkpoint load")

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
    print(f"[INFO] freeze_backbone_all_training={freeze_backbone_all_training}")
    print(f"[INFO] freeze_backbone_epochs={freeze_backbone_epochs}, warmup_epochs={warmup_epochs}")
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
    last_backbone_frozen = None
    bad_epochs = 0
    plateau_epochs = 0

    for epoch in range(start_epoch, epochs):
        model.train()

        set_warmup_lr(optimizer, epoch, warmup_epochs)

        backbone_frozen = freeze_backbone_all_training or (
            freeze_backbone_epochs > 0 and epoch < freeze_backbone_epochs
        )

        if backbone_frozen != last_backbone_frozen:
            n_backbone = set_backbone_trainable(
                model,
                trainable=not backbone_frozen,
            )

            state = "frozen" if backbone_frozen else "unfrozen"

            print(
                f"[INFO] Epoch {epoch + 1}: backbone {state} "
                f"({n_backbone} params). lr: {format_lrs(optimizer)}"
            )

            last_backbone_frozen = backbone_frozen

        elif backbone_frozen:
            backbone = get_backbone_module(model)

            if backbone is not None:
                backbone.eval()

        if freeze_backbone_all_training:
            assert_backbone_frozen(model, where=f"epoch {epoch + 1}")

        conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

        loss_sum = 0.0
        ce_sum = 0.0
        dice_sum = 0.0
        cls_presence_sum = 0.0
        aux_ce_sum = 0.0
        aux_dice_sum = 0.0
        aux_presence_sum = 0.0

        update_steps = 0
        skipped_all_ignore = 0
        skipped_nonfinite = 0
        fallback_fp32 = 0

        total_pixels = 0
        valid_pixels = 0
        ignored_pixels = 0
        bad_examples = []

        prog = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            ncols=150,
            leave=True,
        )

        for batch in prog:
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

            if ema_model is not None:
                update_ema(
                    ema_model,
                    model,
                    decay=float(cfg.get("ema_decay", 0.999)),
                )

            pred_cpu = masks_up.argmax(dim=1).detach().cpu().long()
            label_cpu_after = label.detach().cpu().long()

            for i in range(pred_cpu.size(0)):
                conf_mat = update_confusion_matrix(
                    conf_mat,
                    pred_cpu[i],
                    label_cpu_after[i],
                    num_classes,
                    ignore_index,
                )

            loss_sum += float(loss.item())
            ce_sum += stat["ce"]
            dice_sum += stat["dice_loss"]
            cls_presence_sum += stat["cls_presence"]
            aux_ce_sum += stat["aux_ce"]
            aux_dice_sum += stat["aux_dice_loss"]
            aux_presence_sum += stat["aux_cls_presence"]

            update_steps += 1

            train_metrics = summarize_confmat(conf_mat)

            prog.set_postfix(
                {
                    "loss": f"{loss_sum / max(1, update_steps):.4f}",
                    "dice": f"{train_metrics['mean_dice']:.4f}",
                    "iou": f"{train_metrics['mean_iou']:.4f}",
                }
            )

        if update_steps == 0:
            print("[FATAL] No valid optimizer update in this epoch.")
            print("[FATAL] This usually means all pixels are ignore_index=255.")

            for p in bad_examples:
                print("  -", p)

            break

        train_metrics = summarize_confmat(conf_mat)

        eval_model = ema_model if ema_model is not None else model

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
            f"Loss={loss_sum / update_steps:.4f} | "
            f"Train_Dice={train_metrics['mean_dice']:.4f} | "
            f"Train_IoU={train_metrics['mean_iou']:.4f} | "
            f"Val_Dice={val_metrics['mean_dice']:.4f} | "
            f"Val_IoU={val_metrics['mean_iou']:.4f} | "
            f"CE={ce_sum / update_steps:.4f} | "
            f"DiceLoss={dice_sum / update_steps:.4f} | "
            f"ClsPresence={cls_presence_sum / update_steps:.4f} | "
            f"AuxCE={aux_ce_sum / update_steps:.4f} | "
            f"AuxDiceLoss={aux_dice_sum / update_steps:.4f} | "
            f"AuxClsPresence={aux_presence_sum / update_steps:.4f} | "
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
                f"{dataset_name}_2class_ignore_other_epoch_{epoch + 1:03d}.pth",
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
    print("[INFO] Training finished.")
    print(f"[INFO] Best checkpoint: {best_ckpt}")
    print(f"[INFO] Last checkpoint: {last_ckpt}")
    print(f"[INFO] Best {metric_for_best}={best_metric:.4f}")
    print(f"[INFO] Test set kept for evaluate.py: {splits_dir}/test.txt")
    print("[INFO] Reminder: evaluate.py / predict.py should also use num_classes=2 and ignore_index=255.")
    print("=" * 100)


if __name__ == "__main__":
    main()