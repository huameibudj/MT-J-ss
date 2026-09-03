import os
import sys
import math
import glob
import yaml
import random
import time
import shutil
import argparse
from dataclasses import dataclass, fields
from typing import Dict, Any, Tuple, Optional

import tqdm
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


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """线性预热 + 余弦退火调度器"""
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)
    total_steps = max(1, total_epochs * steps_per_epoch)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def infer_patch_and_embed(backbone: nn.Module, image_size: int, device: torch.device):
    """用一次 dummy forward 自动推断 patch 数量和 embed_dim"""
    backbone.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, image_size, image_size, device=device)
        out = backbone(x)
        patch = out.last_hidden_state[:, 1:, :] if hasattr(out, "last_hidden_state") else out
        if patch.ndim != 3:
            raise RuntimeError(f"Backbone 输出维度异常，期望 [B, N, C]，实际得到 {tuple(patch.shape)}")
        _, num_patches, embed_dim = patch.shape
    return num_patches, embed_dim


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
            interpolation=Image.BILINEAR
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
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class SSLWrapper(nn.Module):
    def __init__(self, vit, embed_dim, proj_dim):
        super().__init__()
        self.vit = vit
        self.proj = MLP(embed_dim, embed_dim, proj_dim)
        self.recon = MLP(embed_dim, embed_dim, embed_dim)

    def forward(self, x):
        out = self.vit(x)
        patch = out.last_hidden_state[:, 1:, :] if hasattr(out, "last_hidden_state") else out
        if patch.ndim != 3:
            raise RuntimeError(f"SSLWrapper.forward 期望 patch 为 [B, N, C]，实际为 {tuple(patch.shape)}")
        return patch, self.proj(patch)


def make_random_mask(B, N, mask_ratio, device):
    n_mask = int(round(N * mask_ratio))
    n_mask = min(max(n_mask, 1), N)
    mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    noise = torch.rand(B, N, device=device)
    idx = noise.topk(k=n_mask, dim=1, largest=False).indices
    mask.scatter_(1, idx, True)
    return mask


def mask_image_patches(x, mask, patch_size, fill_mode="mean"):
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
    )  # [B, gh, gw, C, p, p]

    fill_patch = fill.view(B, 1, 1, C, 1, 1)
    patches = patches * (1.0 - mask_2d) + fill_patch * mask_2d

    x_masked = (
        patches.permute(0, 3, 1, 4, 2, 5)
               .contiguous()
               .view(B, C, H, W)
    )
    return x_masked


def patch_kl_contrastive(z_s, z_t, img_ids, tau_s, tau_t, conf_power=1.5):
    z_s = F.normalize(z_s, dim=-1)
    z_t = F.normalize(z_t, dim=-1)

    l_s = z_s @ z_t.t() / tau_s
    l_t = z_t @ z_t.t() / tau_t

    # 屏蔽来自同一张图的其他 patch，避免伪正样本污染
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


def multi_scale_structure_loss(s_patch, t_patch, grid, num_regions):
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
        region_pool(t_patch, grid, max(4, num_regions // 4))
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

    mpr_mse_alpha: float = 0.5
    mpr_cos_beta: float = 0.5

    out_dir: str = "ssl_checkpoints_v2"
    save_every: int = 10
    log_interval: int = 20
    resume_from: str = ""


def load_cfg(cfg_path: str) -> CFG:
    with open(cfg_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f) or {}

    valid_keys = {f.name for f in fields(CFG)}
    filtered = {k: v for k, v in yaml_data.items() if k in valid_keys}
    cfg = CFG(**filtered)
    return cfg


# =========================================================
# 5) 主训练入口
# =========================================================
def main(cfg_path: Optional[str] = None):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if cfg_path is None:
        cfg_path = os.path.join(script_dir, "configs", "ssl_pretrain_patch_pcl_v2.yaml")
    else:
        cfg_path = resolve_path(cfg_path, script_dir)

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    cfg = load_cfg(cfg_path)

    # 把配置中的相对路径统一解析到“配置文件所在目录”
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

    set_seed(cfg.seed)

    use_cuda = torch.cuda.is_available() and str(cfg.device).lower().startswith("cuda")
    device = torch.device("cuda" if use_cuda else "cpu")

    os.makedirs(cfg.out_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(cfg.out_dir, "config_active.yaml"))

    print(f"[INFO] 配置文件: {cfg_path}")
    print(f"[INFO] 数据目录: {cfg.data_root}")
    print(f"[INFO] 输出目录: {cfg.out_dir}")
    print(f"[INFO] 训练设备: {device}")

    ds = UnlabeledImageFolder(cfg.data_root, cfg.image_size)
    if len(ds) == 0:
        raise RuntimeError("数据集为空")

    # 小数据集时避免 drop_last 直接把整个 epoch 丢没
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

    vit_cfg = {
        "backbone_impl": cfg.backbone_impl,
        "img_size": cfg.image_size,

        # HF
        "vit_path": cfg.vit_path,
        "hf_model_name_or_path": cfg.hf_model_name_or_path,
        "vit_weight_path": cfg.vit_weight_path,

        # timm
        "timm_model_name": cfg.timm_model_name,
        "coach_bin": cfg.coach_bin,
        "timm_pretrained": cfg.timm_pretrained,
    }

    print(f"[DEBUG] vit_cfg = {vit_cfg}")

    # 先实例化 backbone，再自动探测输出维度
    probe_backbone = CompatibleViTBackbone(vit_cfg).to(device)
    num_patches, embed_dim = infer_patch_and_embed(probe_backbone, cfg.image_size, device)

    expected_grid = cfg.image_size // cfg.patch_size
    expected_patches = expected_grid * expected_grid
    if num_patches != expected_patches:
        raise ValueError(
            f"Backbone 输出 patch 数={num_patches}，但根据 image_size={cfg.image_size} 和 "
            f"patch_size={cfg.patch_size} 推导应为 {expected_patches}。"
            f"请检查 patch_size 是否与 backbone 的真实 patch size 一致。"
        )

    print(f"[INFO] 自动探测: num_patches={num_patches}, embed_dim={embed_dim}")

    student = SSLWrapper(CompatibleViTBackbone(vit_cfg), embed_dim, cfg.proj_dim).to(device)
    teacher = SSLWrapper(CompatibleViTBackbone(vit_cfg), embed_dim, cfg.proj_dim).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_lr_scheduler(optim, cfg.warmup_epochs, cfg.epochs, len(dl))
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and use_cuda))

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

    w_base = torch.tensor([cfg.w_pcl, cfg.w_mpr, cfg.w_struct], dtype=torch.float32)
    w_base = w_base / w_base.sum()
    grid = cfg.image_size // cfg.patch_size
    max_steps = max(1, cfg.epochs * len(dl))

    sampled_patches = min(cfg.pcl_num_sampled_patches, num_patches)
    if sampled_patches != cfg.pcl_num_sampled_patches:
        print(f"[WARN] pcl_num_sampled_patches 从 {cfg.pcl_num_sampled_patches} 自动裁剪到 {sampled_patches}")

    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            student.train()
            teacher.eval()

            t_cur = (epoch - 1) / max(1, cfg.epochs - 1)
            w = [
                (0.25 + 0.15 * t_cur) * w_base[0].item(),
                (0.50 - 0.20 * t_cur) * w_base[1].item(),
                (0.25 + 0.05 * t_cur) * w_base[2].item(),
            ]
            w_sum = sum(w)
            w = [v / w_sum for v in w]

            pbar = tqdm.tqdm(dl, desc=f"Epoch {epoch}/{cfg.epochs}", ncols=155)

            for step, (xw, xs) in enumerate(pbar, start=1):
                xw = xw.to(device, non_blocking=True)
                xs = xs.to(device, non_blocking=True)

                with torch.no_grad():
                    patch_t, proj_t = teacher(xw)

                with torch.cuda.amp.autocast(enabled=(cfg.use_amp and use_cuda)):
                    patch_s, proj_s = student(xs)

                    B, N, P = proj_s.shape
                    if N != num_patches:
                        raise RuntimeError(f"当前 batch 的 patch 数 N={N} 与初始化探测值 {num_patches} 不一致")

                    idx = torch.randint(0, N, (B, sampled_patches), device=device)
                    ps = torch.gather(proj_s, 1, idx.unsqueeze(-1).expand(-1, -1, P)).reshape(-1, P)
                    pt = torch.gather(proj_t, 1, idx.unsqueeze(-1).expand(-1, -1, P)).reshape(-1, P)
                    ids = torch.arange(B, device=device).unsqueeze(1).expand(-1, sampled_patches).reshape(-1)

                    l_pcl = patch_kl_contrastive(
                        ps, pt, ids,
                        cfg.temperature,
                        cfg.pcl_teacher_temperature,
                        cfg.pcl_conf_power
                    )

                    mask = make_random_mask(B, N, cfg.mask_ratio, device)
                    xs_m = mask_image_patches(xs, mask, cfg.patch_size, cfg.mask_fill_mode)
                    p_sm, _ = student(xs_m)

                    recon = student.recon(p_sm)
                    l_mpr = F.smooth_l1_loss(recon[mask], patch_t[mask])

                    l_struct = multi_scale_structure_loss(patch_s, patch_t, grid, cfg.num_regions)

                    loss = w[0] * l_pcl + w[1] * l_mpr + w[2] * l_struct

                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                scheduler.step()

                m = cfg.ema_m_base + (cfg.ema_m_final - cfg.ema_m_base) * (global_step / max_steps)
                m = min(max(m, 0.0), 0.999999)

                with torch.no_grad():
                    for pt_param, ps_param in zip(teacher.parameters(), student.parameters()):
                        pt_param.data.mul_(m).add_(ps_param.data, alpha=1.0 - m)

                global_step += 1

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "pcl": f"{l_pcl.item():.4f}",
                    "mpr": f"{l_mpr.item():.4f}",
                    "struct": f"{l_struct.item():.4f}",
                    "lr": f"{optim.param_groups[0]['lr']:.2e}",
                })

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
                print(f"[SAVE] checkpoint: {latest_ckpt}")
                print(f"[SAVE] backbone:   {backbone_ckpt}")

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
        help="配置文件路径，例如 configs/ssl_pretrain_patch_pcl_v2.yaml"
    )
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args.config)