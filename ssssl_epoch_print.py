import os
import sys
import math
import glob
import yaml
import random
import copy
import shutil
import argparse
from dataclasses import dataclass, fields
from typing import Optional, Dict

import numpy as np
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 1024 * 1024 * 100
PngImagePlugin.MAX_TEXT_MEMORY = 1024 * 1024 * 500

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from models.vit_backbone import CompatibleViTBackbone


# =========================================================
# 1) 工具函数
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def get_lr_scheduler(optimizer, warmup_epochs: int, total_epochs: int, steps_per_epoch: int):
    """线性预热 + 余弦退火调度器；对 optimizer 中所有 param_group 等比例缩放。"""
    warmup_steps = max(0, int(warmup_epochs) * int(steps_per_epoch))
    total_steps = max(1, int(total_epochs) * int(steps_per_epoch))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def infer_patch_and_embed(backbone: nn.Module, image_size: int, device: torch.device):
    """用 student_backbone 的一次 dummy forward 自动推断 patch 数量和 embed_dim。"""
    backbone.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, image_size, image_size, device=device)
        out = backbone(x)
        patch = out.last_hidden_state[:, 1:, :] if hasattr(out, "last_hidden_state") else out
        if patch.ndim != 3:
            raise RuntimeError(f"Backbone 输出维度异常，期望 [B, N, C]，实际得到 {tuple(patch.shape)}")
        _, num_patches, embed_dim = patch.shape
    return num_patches, embed_dim


def build_vit_cfg(cfg: "CFG") -> Dict[str, object]:
    return {
        "backbone_impl": cfg.backbone_impl,
        "img_size": cfg.image_size,

        # HF / local
        "vit_path": cfg.vit_path,
        "hf_model_name_or_path": cfg.hf_model_name_or_path,
        "vit_weight_path": cfg.vit_weight_path,

        # timm
        "timm_model_name": cfg.timm_model_name,
        "coach_bin": cfg.coach_bin,
        "timm_pretrained": cfg.timm_pretrained,
    }


def freeze_teacher(teacher: nn.Module):
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)


def build_optimizer(student: "SSLWrapper", cfg: "CFG"):
    """
    backbone_lr 真正生效：
    - student.vit 使用 cfg.backbone_lr
    - projector / reconstruction head 使用 cfg.lr
    """
    return torch.optim.AdamW(
        [
            {
                "name": "backbone",
                "params": student.vit.parameters(),
                "lr": cfg.backbone_lr,
            },
            {
                "name": "heads",
                "params": list(student.proj.parameters()) + list(student.recon.parameters()),
                "lr": cfg.lr,
            },
        ],
        weight_decay=cfg.weight_decay,
    )


def active_loss_weights(cfg: "CFG", t_cur: float):
    """只对已开启的目标归一化权重，保证 ablation 后 loss 尺度稳定。"""
    raw = {
        "pcl": ((0.25 + 0.15 * t_cur) * cfg.w_pcl) if cfg.use_pcl else 0.0,
        "mpr": ((0.50 - 0.20 * t_cur) * cfg.w_mpr) if cfg.use_mpr else 0.0,
        "struct": ((0.25 + 0.05 * t_cur) * cfg.w_struct) if cfg.use_struct else 0.0,
    }
    total = float(sum(raw.values()))
    if total <= 0:
        raise ValueError("至少需要开启一个训练目标：use_pcl / use_mpr / use_struct")
    return {k: v / total for k, v in raw.items()}


# =========================================================
# 2) 数据增强与加载
# =========================================================
class SharedCropTwoView:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.weak_photo = T.Compose([
            T.ColorJitter(0.2, 0.2, 0.2, 0.05)
        ])
        self.strong_photo = T.Compose([
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(9, (0.1, 2.0)),
        ])

    def __call__(self, img: Image.Image):
        i, j, h, w = T.RandomResizedCrop.get_params(
            img, scale=(0.5, 1.0), ratio=(0.75, 1.33)
        )
        img_crop = TF.resized_crop(
            img,
            i, j, h, w,
            size=(self.image_size, self.image_size),
            interpolation=Image.BILINEAR,
        )
        if random.random() < 0.5:
            img_crop = TF.hflip(img_crop)
        if random.random() < 0.5:
            img_crop = TF.vflip(img_crop)
        return TF.to_tensor(self.weak_photo(img_crop)), TF.to_tensor(self.strong_photo(img_crop))


class UnlabeledImageFolder(Dataset):
    def __init__(self, root: str, image_size: int = 224):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.webp")
        self.paths = []
        for e in exts:
            self.paths += glob.glob(os.path.join(root, "**", e), recursive=True)
        self.paths = sorted(set(self.paths))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"未找到图片: {root}")
        self.tf = SharedCropTwoView(image_size=image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)


# =========================================================
# 3) 模型架构与损失函数
# =========================================================
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SSLWrapper(nn.Module):
    def __init__(self, vit: nn.Module, embed_dim: int, proj_dim: int):
        super().__init__()
        self.vit = vit
        self.proj = MLP(embed_dim, embed_dim, proj_dim)
        self.recon = MLP(embed_dim, embed_dim, embed_dim)

    def encode(self, x):
        out = self.vit(x)
        patch = out.last_hidden_state[:, 1:, :] if hasattr(out, "last_hidden_state") else out
        if patch.ndim != 3:
            raise RuntimeError(f"SSLWrapper.encode 期望 patch 为 [B, N, C]，实际为 {tuple(patch.shape)}")
        return patch

    def forward(self, x, return_proj: bool = True):
        patch = self.encode(x)
        proj = self.proj(patch) if return_proj else None
        return patch, proj

    def reconstruct(self, patch):
        return self.recon(patch)


def make_random_mask(B: int, N: int, mask_ratio: float, device: torch.device):
    n_mask = int(round(N * mask_ratio))
    n_mask = min(max(n_mask, 1), N)
    mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    noise = torch.rand(B, N, device=device)
    idx = noise.topk(k=n_mask, dim=1, largest=False).indices
    mask.scatter_(1, idx, True)
    return mask


def mask_image_patches(x, mask, patch_size: int, fill_mode: str = "mean"):
    B, C, H, W = x.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"输入图像尺寸 {(H, W)} 不能被 patch_size={patch_size} 整除")

    gh, gw = H // patch_size, W // patch_size
    if mask.shape != (B, gh * gw):
        raise ValueError(f"mask 形状应为 {(B, gh * gw)}，实际为 {tuple(mask.shape)}")

    if fill_mode == "mean":
        fill = x.mean(dim=(2, 3), keepdim=True)
    elif fill_mode == "zero":
        fill = torch.zeros_like(x[:, :, :1, :1])
    else:
        raise ValueError(f"不支持的 mask_fill_mode: {fill_mode}")

    mask_2d = mask.view(B, gh, gw, 1, 1, 1).to(dtype=x.dtype)

    patches = (
        x.view(B, C, gh, patch_size, gw, patch_size)
         .permute(0, 2, 4, 1, 3, 5)
         .contiguous()
    )

    fill_patch = fill.view(B, 1, 1, C, 1, 1)
    patches = patches * (1.0 - mask_2d) + fill_patch * mask_2d

    x_masked = (
        patches.permute(0, 3, 1, 4, 2, 5)
               .contiguous()
               .view(B, C, H, W)
    )
    return x_masked


def patch_kl_contrastive(z_s, z_t, img_ids, tau_s: float, tau_t: float, conf_power: float = 1.5):
    z_s = F.normalize(z_s, dim=-1)
    z_t = F.normalize(z_t, dim=-1)

    l_s = z_s @ z_t.t() / tau_s
    l_t = z_t @ z_t.t() / tau_t

    same_img = img_ids[:, None].eq(img_ids[None, :])
    eye = torch.eye(z_s.size(0), device=z_s.device, dtype=torch.bool)
    mask = same_img & (~eye)

    l_s = l_s.masked_fill(mask, -1e4)
    l_t = l_t.masked_fill(mask, -1e4)

    with torch.no_grad():
        prob_t = F.softmax(l_t, dim=-1)
        entropy = -(prob_t * torch.log(prob_t.clamp_min(1e-8))).sum(-1)
        conf = (1.0 - entropy / math.log(prob_t.size(-1))).clamp(0, 1).pow(conf_power)

    loss = -(prob_t * F.log_softmax(l_s, dim=-1)).sum(-1)
    return (loss * conf).sum() / conf.sum().clamp_min(1.0)


def masked_patch_reconstruction_loss(recon, target, mask, mse_alpha: float = 0.5, cos_beta: float = 0.5):
    recon_m = recon[mask]
    target_m = target[mask]
    if recon_m.numel() == 0:
        return recon.sum() * 0.0

    l_mse = F.mse_loss(recon_m, target_m)
    l_cos = 1.0 - F.cosine_similarity(recon_m, target_m, dim=-1).mean()
    return mse_alpha * l_mse + cos_beta * l_cos


def multi_scale_structure_loss(s_patch, t_patch, grid: int, num_regions: int):
    def region_pool(p, g, nr):
        B, N, C = p.shape
        if N != g * g:
            raise ValueError(f"patch 数量 N={N} 与 grid={g} 不匹配，期望 N={g*g}")
        rs = max(1, int(round(math.sqrt(nr))))
        b = g // rs
        if b * rs != g:
            return p.mean(dim=1, keepdim=True)
        x = p.view(B, g, g, C)
        pooled = []
        for i in range(rs):
            for j in range(rs):
                pooled.append(x[:, i*b:(i+1)*b, j*b:(j+1)*b, :].mean(dim=(1, 2)))
        return torch.stack(pooled, dim=1)

    l_f = F.mse_loss(region_pool(s_patch, grid, num_regions), region_pool(t_patch, grid, num_regions))
    l_c = F.mse_loss(
        region_pool(s_patch, grid, max(4, num_regions // 4)),
        region_pool(t_patch, grid, max(4, num_regions // 4)),
    )

    s = s_patch.view(-1, grid, grid, s_patch.size(-1))
    t = t_patch.view(-1, grid, grid, t_patch.size(-1))

    l_g = (
        F.smooth_l1_loss(s[:, :, 1:, :] - s[:, :, :-1, :], t[:, :, 1:, :] - t[:, :, :-1, :]) +
        F.smooth_l1_loss(s[:, 1:, :, :] - s[:, :-1, :, :], t[:, 1:, :, :] - t[:, :-1, :, :])
    )
    return 0.4 * l_f + 0.2 * l_c + 0.4 * l_g


# =========================================================
# 4) 配置解析器
# =========================================================
@dataclass
class CFG:
    data_root: str = ""
    image_size: int = 224
    patch_size: int = 16

    backbone_impl: str = "timm"

    # HF / 本地离线加载
    vit_path: str = ""
    hf_model_name_or_path: str = ""
    vit_weight_path: str = ""

    # timm
    timm_model_name: str = "vit_base_patch16_224"
    timm_pretrained: bool = False
    coach_bin: str = ""

    batch_size: int = 8
    epochs: int = 40
    lr: float = 1e-4
    backbone_lr: float = 1e-5
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"
    use_amp: bool = True

    ema_m_base: float = 0.996
    ema_m_final: float = 0.9999

    mask_ratio: float = 0.3
    mask_fill_mode: str = "mean"

    proj_dim: int = 256
    temperature: float = 0.2
    pcl_teacher_temperature: float = 0.1
    pcl_conf_power: float = 1.5
    pcl_num_sampled_patches: int = 64

    num_regions: int = 16
    w_pcl: float = 1.0
    w_mpr: float = 1.5
    w_struct: float = 5.0

    # Ablation switches：关闭后不仅 loss=0，而且跳过对应模块前向
    use_pcl: bool = True
    use_mpr: bool = True
    use_struct: bool = True

    mpr_mse_alpha: float = 0.5
    mpr_cos_beta: float = 0.5

    out_dir: str = "ssl_checkpoints_v2"
    save_every: int = 10
    resume_from: str = ""


def _convert_cfg_value(name: str, value, target_type):
    try:
        if target_type == float:
            return float(value)
        if target_type == int:
            return int(value)
        if target_type == bool:
            if isinstance(value, str):
                v = value.strip().lower()
                if v in {"true", "1", "yes", "y", "on"}:
                    return True
                if v in {"false", "0", "no", "n", "off"}:
                    return False
                raise ValueError
            return bool(value)
        if target_type == str:
            return "" if value is None else str(value)
        return value
    except Exception as exc:
        raise ValueError(f"配置项 [{name}] 类型错误，当前值={value}, 期望类型={target_type}") from exc


def load_cfg(cfg_path: str) -> CFG:
    with open(cfg_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f) or {}

    cfg = CFG()
    valid_names = {field.name for field in fields(CFG)}
    unknown = sorted(set(yaml_data.keys()) - valid_names)
    if unknown:
        print(f"[WARN] 配置文件中存在未使用字段，已忽略: {unknown}")

    for field in fields(CFG):
        if field.name not in yaml_data:
            continue
        value = _convert_cfg_value(field.name, yaml_data[field.name], field.type)
        setattr(cfg, field.name, value)

    return cfg


def resolve_cfg_paths(cfg: CFG, cfg_path: str) -> CFG:
    cfg_base_dir = os.path.dirname(os.path.abspath(cfg_path))
    cfg.data_root = resolve_path(cfg.data_root, cfg_base_dir)
    cfg.coach_bin = resolve_path(cfg.coach_bin, cfg_base_dir) if cfg.coach_bin else cfg.coach_bin
    cfg.vit_path = resolve_path(cfg.vit_path, cfg_base_dir) if cfg.vit_path else cfg.vit_path
    cfg.hf_model_name_or_path = (
        resolve_path(cfg.hf_model_name_or_path, cfg_base_dir)
        if cfg.hf_model_name_or_path and os.path.exists(resolve_path(cfg.hf_model_name_or_path, cfg_base_dir))
        else cfg.hf_model_name_or_path
    )
    cfg.vit_weight_path = resolve_path(cfg.vit_weight_path, cfg_base_dir) if cfg.vit_weight_path else cfg.vit_weight_path
    cfg.resume_from = resolve_path(cfg.resume_from, cfg_base_dir) if cfg.resume_from else cfg.resume_from
    cfg.out_dir = resolve_path(cfg.out_dir, cfg_base_dir)
    return cfg


def validate_cfg(cfg: CFG):
    if not any([cfg.use_pcl, cfg.use_mpr, cfg.use_struct]):
        raise ValueError("use_pcl / use_mpr / use_struct 不能全部为 False")
    if cfg.image_size <= 0 or cfg.patch_size <= 0:
        raise ValueError("image_size 和 patch_size 必须为正整数")
    if (cfg.use_mpr or cfg.use_struct) and cfg.image_size % cfg.patch_size != 0:
        raise ValueError(f"image_size={cfg.image_size} 不能被 patch_size={cfg.patch_size} 整除")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if cfg.epochs <= 0:
        raise ValueError("epochs 必须大于 0")
    if cfg.lr <= 0 or cfg.backbone_lr <= 0:
        raise ValueError("lr 和 backbone_lr 必须大于 0")
    if cfg.use_mpr and not (0 < cfg.mask_ratio <= 1):
        raise ValueError("mask_ratio 必须在 (0, 1] 范围内")
    if cfg.use_pcl and cfg.pcl_num_sampled_patches <= 0:
        raise ValueError("pcl_num_sampled_patches 必须大于 0")


# =========================================================
# 5) 主训练入口
# =========================================================
def main(cfg_path: Optional[str] = None):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if cfg_path is None:
        cfg_path = os.path.join(script_dir, "configs", "ssssl.yaml")
    else:
        cfg_path = resolve_path(cfg_path, script_dir)

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    cfg = resolve_cfg_paths(load_cfg(cfg_path), cfg_path)
    validate_cfg(cfg)
    set_seed(cfg.seed)

    use_cuda = torch.cuda.is_available() and str(cfg.device).lower().startswith("cuda")
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_enabled = bool(cfg.use_amp and use_cuda)

    os.makedirs(cfg.out_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(cfg.out_dir, "config_active.yaml"))

    print(f"[INFO] 配置文件: {cfg_path}")
    print(f"[INFO] 数据目录: {cfg.data_root}")
    print(f"[INFO] 输出目录: {cfg.out_dir}")
    print(f"[INFO] 训练设备: {device}")
    print(
        "[INFO] 消融开关: "
        f"PCL={cfg.use_pcl}, MPR={cfg.use_mpr}, STRUCT={cfg.use_struct}"
    )
    print(f"[INFO] 学习率: backbone_lr={cfg.backbone_lr:.3e}, head_lr={cfg.lr:.3e}")

    ds = UnlabeledImageFolder(cfg.data_root, cfg.image_size)
    if len(ds) == 0:
        raise RuntimeError("数据集为空")

    effective_drop_last = len(ds) >= cfg.batch_size
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_cuda,
        drop_last=effective_drop_last,
        persistent_workers=(cfg.num_workers > 0),
    )

    if len(dl) == 0:
        raise RuntimeError(
            f"DataLoader 长度为 0。当前数据集大小={len(ds)}，batch_size={cfg.batch_size}，"
            f"请减小 batch_size 或检查数据集。"
        )

    vit_cfg = build_vit_cfg(cfg)

    # -----------------------------------------------------
    # 核心构造顺序：
    # CompatibleViTBackbone() -> infer() -> SSLWrapper(student) -> deepcopy() -> teacher
    # 这样 backbone 只实例化一次，预训练权重也只加载一次。
    # -----------------------------------------------------
    student_backbone = CompatibleViTBackbone(vit_cfg).to(device)
    num_patches, embed_dim = infer_patch_and_embed(student_backbone, cfg.image_size, device)

    expected_grid = cfg.image_size // cfg.patch_size
    if cfg.use_mpr or cfg.use_struct:
        expected_patches = expected_grid * expected_grid
        if num_patches != expected_patches:
            raise ValueError(
                f"Backbone 输出 patch 数={num_patches}，但根据 image_size={cfg.image_size} 和 "
                f"patch_size={cfg.patch_size} 推导应为 {expected_patches}。"
                f"请检查 patch_size 是否与 backbone 的真实 patch size 一致。"
            )

    student = SSLWrapper(student_backbone, embed_dim, cfg.proj_dim).to(device)
    teacher = copy.deepcopy(student).to(device)
    freeze_teacher(teacher)

    print(f"[INFO] Backbone 探测完成: num_patches={num_patches}, embed_dim={embed_dim}")

    optim = build_optimizer(student, cfg)
    scheduler = get_lr_scheduler(optim, cfg.warmup_epochs, cfg.epochs, len(dl))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    global_step = 0

    if cfg.resume_from and os.path.exists(cfg.resume_from):
        print(f"[RESUME] 正在加载: {cfg.resume_from}")
        ckpt = torch.load(cfg.resume_from, map_location=device)
        student.load_state_dict(ckpt["student"])
        teacher.load_state_dict(ckpt["teacher"])
        optim.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt["global_step"])
        freeze_teacher(teacher)

    grid = cfg.image_size // cfg.patch_size
    max_steps = max(1, cfg.epochs * len(dl))

    sampled_patches = min(cfg.pcl_num_sampled_patches, num_patches)
    if cfg.use_pcl and sampled_patches != cfg.pcl_num_sampled_patches:
        print(f"[WARN] pcl_num_sampled_patches 从 {cfg.pcl_num_sampled_patches} 自动裁剪到 {sampled_patches}")

    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            student.train()
            teacher.eval()

            t_cur = (epoch - 1) / max(1, cfg.epochs - 1)
            weights = active_loss_weights(cfg, t_cur)

            # 每个 epoch 只输出一次汇总，不使用实时进度条
            epoch_loss_sum = 0.0
            epoch_pcl_sum = 0.0
            epoch_mpr_sum = 0.0
            epoch_struct_sum = 0.0
            epoch_batches = 0

            for _, (xw, xs) in enumerate(dl, start=1):
                xw = xw.to(device, non_blocking=True)
                xs = xs.to(device, non_blocking=True)

                need_student_clean = cfg.use_pcl or cfg.use_struct
                need_student_proj = cfg.use_pcl
                need_teacher_proj = cfg.use_pcl

                with torch.no_grad():
                    patch_t, proj_t = teacher(xw, return_proj=need_teacher_proj)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    patch_s = None
                    proj_s = None

                    if need_student_clean:
                        patch_s, proj_s = student(xs, return_proj=need_student_proj)
                        B, N, _ = patch_s.shape
                    else:
                        B, N, _ = patch_t.shape

                    if N != num_patches:
                        raise RuntimeError(f"当前 batch 的 patch 数 N={N} 与初始化探测值 {num_patches} 不一致")

                    l_pcl = torch.zeros((), device=device)
                    l_mpr = torch.zeros((), device=device)
                    l_struct = torch.zeros((), device=device)

                    if cfg.use_pcl:
                        _, _, P = proj_s.shape
                        idx = torch.randint(0, N, (B, sampled_patches), device=device)
                        ps = torch.gather(proj_s, 1, idx.unsqueeze(-1).expand(-1, -1, P)).reshape(-1, P)
                        pt = torch.gather(proj_t, 1, idx.unsqueeze(-1).expand(-1, -1, P)).reshape(-1, P)
                        ids = torch.arange(B, device=device).unsqueeze(1).expand(-1, sampled_patches).reshape(-1)
                        l_pcl = patch_kl_contrastive(
                            ps,
                            pt,
                            ids,
                            cfg.temperature,
                            cfg.pcl_teacher_temperature,
                            cfg.pcl_conf_power,
                        )

                    if cfg.use_mpr:
                        mask = make_random_mask(B, N, cfg.mask_ratio, device)
                        xs_m = mask_image_patches(xs, mask, cfg.patch_size, cfg.mask_fill_mode)
                        p_sm, _ = student(xs_m, return_proj=False)
                        recon = student.reconstruct(p_sm)
                        l_mpr = masked_patch_reconstruction_loss(
                            recon,
                            patch_t,
                            mask,
                            cfg.mpr_mse_alpha,
                            cfg.mpr_cos_beta,
                        )

                    if cfg.use_struct:
                        l_struct = multi_scale_structure_loss(patch_s, patch_t, grid, cfg.num_regions)

                    loss = (
                        weights["pcl"] * l_pcl +
                        weights["mpr"] * l_mpr +
                        weights["struct"] * l_struct
                    )

                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                scheduler.step()

                ema_progress = min(max(global_step / max_steps, 0.0), 1.0)
                m = cfg.ema_m_base + (cfg.ema_m_final - cfg.ema_m_base) * ema_progress
                m = min(max(m, 0.0), 0.999999)

                with torch.no_grad():
                    for pt_param, ps_param in zip(teacher.parameters(), student.parameters()):
                        pt_param.data.mul_(m).add_(ps_param.data, alpha=1.0 - m)

                global_step += 1

                # 累计当前 epoch 的损失，结束后统一打印平均值
                epoch_loss_sum += float(loss.detach().item())
                epoch_pcl_sum += float(l_pcl.detach().item())
                epoch_mpr_sum += float(l_mpr.detach().item())
                epoch_struct_sum += float(l_struct.detach().item())
                epoch_batches += 1

            saved_checkpoint = None
            if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
                state = {
                    "epoch": epoch,
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": global_step,
                    "cfg_path": cfg_path,
                }
                latest_ckpt = os.path.join(cfg.out_dir, "latest_checkpoint.pth")
                backbone_ckpt = os.path.join(cfg.out_dir, f"backbone_ep{epoch}.pth")

                torch.save(state, latest_ckpt)
                torch.save(student.vit.state_dict(), backbone_ckpt)
                saved_checkpoint = latest_ckpt

            denom = max(1, epoch_batches)
            pcl_text = f"{epoch_pcl_sum / denom:.4f}" if cfg.use_pcl else "off"
            mpr_text = f"{epoch_mpr_sum / denom:.4f}" if cfg.use_mpr else "off"
            struct_text = f"{epoch_struct_sum / denom:.4f}" if cfg.use_struct else "off"
            save_text = f", saved={saved_checkpoint}" if saved_checkpoint else ""

            print(
                f"[EPOCH {epoch}/{cfg.epochs}] "
                f"loss={epoch_loss_sum / denom:.4f}, "
                f"pcl={pcl_text}, mpr={mpr_text}, struct={struct_text}, "
                f"lr_bb={optim.param_groups[0]['lr']:.2e}, "
                f"lr_hd={optim.param_groups[1]['lr']:.2e}, "
                f"ema_m={m:.6f}"
                f"{save_text}",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\n[EXIT] 检测到手动中断，正在保存紧急存盘...")
        emergency_path = os.path.join(cfg.out_dir, "interrupted_state.pth")
        torch.save({
            "epoch": epoch if "epoch" in locals() else 0,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "cfg_path": cfg_path,
        }, emergency_path)
        print(f"[EXIT] 已保存至: {emergency_path}")
        sys.exit(0)


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="configs/ssssl.yaml",
    )
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args.config)
