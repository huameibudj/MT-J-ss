import os
import math
import glob
import yaml
import random
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from models.vit_backbone import CompatibleViTBackbone


# =========================================================
# 0) 通用工具
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_any_encoding(path: str):
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return yaml.safe_load(f)


def _clean_empty_to_none(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and v.strip() == "":
            out[k] = None
        else:
            out[k] = v
    return out


def build_vit_backbone_cfg(cfg) -> Dict[str, Any]:
    vit_cfg = {
        "backbone_impl": str(getattr(cfg, "backbone_impl", "hf")).lower(),
        "img_size": int(getattr(cfg, "image_size", 224)),
        "timm_model_name": getattr(cfg, "timm_model_name", "vit_base_patch16_224"),
        "timm_pretrained": bool(getattr(cfg, "timm_pretrained", False)),
        "coach_bin": getattr(cfg, "coach_bin", None),
        "vit_weight_path": getattr(cfg, "vit_weight_path", None),
        "vit_path": getattr(cfg, "vit_path", None),
        "hf_model_name_or_path": getattr(cfg, "hf_model_name_or_path", None),
    }
    return _clean_empty_to_none(vit_cfg)


def backbone_to_patch_tokens(out):
    if torch.is_tensor(out):
        return out
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state[:, 1:, :]
    raise TypeError(f"Unsupported backbone output type: {type(out)}")


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, m: float):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))


def cosine_schedule(base: float, final: float, step: int, max_steps: int):
    if max_steps <= 1:
        return final
    t = step / (max_steps - 1)
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * t))


def normalize_loss_weights(w_pcl: float, w_mpr: float, w_struct: float):
    w = torch.tensor([w_pcl, w_mpr, w_struct], dtype=torch.float32)
    w = torch.clamp(w, min=0.0)
    s = float(w.sum().item())
    if s <= 0:
        raise ValueError("三个损失权重不能全为 0，请至少开启一个 loss。")
    w = w / s
    return float(w[0].item()), float(w[1].item()), float(w[2].item())


def get_dynamic_weights(epoch_idx: int, epochs: int, base_weights: Tuple[float, float, float]):
    bw_pcl, bw_mpr, bw_struct = base_weights
    t = float(epoch_idx) / max(1.0, float(epochs - 1))
    dyn_pcl = (0.25 + 0.15 * t) * bw_pcl
    dyn_mpr = (0.50 - 0.20 * t) * bw_mpr
    dyn_struct = (0.25 + 0.05 * t) * bw_struct
    return normalize_loss_weights(dyn_pcl, dyn_mpr, dyn_struct)


# =========================================================
# 1) 共享几何增强（保证 patch-level 对齐）+ 强弱颜色增强
# =========================================================
class SharedCropTwoView:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.weak_photo = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])
        self.strong_photo = T.Compose([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        ])

    def __call__(self, img: Image.Image):
        i, j, h, w = T.RandomResizedCrop.get_params(img, scale=(0.5, 1.0), ratio=(0.75, 1.33))
        img_crop = TF.resized_crop(
            img, i, j, h, w,
            size=(self.image_size, self.image_size),
            interpolation=Image.BILINEAR
        )
        if random.random() < 0.5:
            img_crop = TF.hflip(img_crop)
        if random.random() < 0.5:
            img_crop = TF.vflip(img_crop)
        weak = TF.to_tensor(self.weak_photo(img_crop))
        strong = TF.to_tensor(self.strong_photo(img_crop))
        return weak, strong


class UnlabeledImageFolder(Dataset):
    def __init__(self, root: str, image_size: int = 224):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.webp")
        self.paths = []
        for e in exts:
            self.paths += glob.glob(os.path.join(root, "**", e), recursive=True)
        if len(self.paths) == 0:
            raise FileNotFoundError(f"在目录下未找到图片: {root}")
        self.tf = SharedCropTwoView(image_size=image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        xw, xs = self.tf(img)
        return xw, xs


# =========================================================
# 2) Head：patch-level 投影头 + MPR 重建头 + mask token
# =========================================================
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SSLWrapper(nn.Module):
    def __init__(self, vit: CompatibleViTBackbone, embed_dim: int, proj_dim: int):
        super().__init__()
        self.vit = vit
        self.proj = MLP(embed_dim, embed_dim, proj_dim, num_layers=2)
        self.recon = MLP(embed_dim, embed_dim, embed_dim, num_layers=2)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x):
        out = self.vit(x)
        patch = backbone_to_patch_tokens(out)
        proj_patch = self.proj(patch)
        return patch, proj_patch


# =========================================================
# 3) Mask / Loss
# =========================================================
def make_random_mask(B: int, N: int, mask_ratio: float, device):
    n_mask = int(round(N * mask_ratio))
    mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    if n_mask <= 0:
        return mask
    noise = torch.rand(B, N, device=device)
    idx = noise.topk(k=n_mask, dim=1, largest=False).indices
    mask.scatter_(1, idx, True)
    return mask


def mask_image_patches(x: torch.Tensor, mask: torch.Tensor, patch_size: int, fill_mode: str = "mean"):
    B, C, H, W = x.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"图像尺寸 {H}x{W} 不能被 patch_size={patch_size} 整除")

    gh = H // patch_size
    gw = W // patch_size
    if mask.shape != (B, gh * gw):
        raise ValueError(f"mask 形状应为 {(B, gh * gw)}，实际得到 {tuple(mask.shape)}")

    if fill_mode == "mean":
        fill = x.mean(dim=(2, 3), keepdim=True)
    elif fill_mode == "zero":
        fill = torch.zeros((B, C, 1, 1), device=x.device, dtype=x.dtype)
    else:
        raise ValueError(f"Unsupported fill_mode: {fill_mode}")

    mask_2d = mask.view(B, gh, gw, 1, 1, 1).to(dtype=x.dtype)
    patches = x.view(B, C, gh, patch_size, gw, patch_size).permute(0, 2, 4, 1, 3, 5).contiguous()
    fill_patch = fill.view(B, 1, 1, C, 1, 1)
    patches = patches * (1.0 - mask_2d) + fill_patch * mask_2d
    return patches.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)


def patch_kl_contrastive(z_s: torch.Tensor, z_t: torch.Tensor, img_ids: torch.Tensor,
                         tau_s: float = 0.2, tau_t: float = 0.07, conf_power: float = 1.5):
    z_s = F.normalize(z_s, dim=-1)
    z_t = F.normalize(z_t, dim=-1)

    logits_s = z_s @ z_t.t() / tau_s
    logits_t = z_t @ z_t.t() / tau_t

    same_img = img_ids[:, None].eq(img_ids[None, :])
    eye = torch.eye(z_s.size(0), device=z_s.device, dtype=torch.bool)
    neg_mask = same_img & (~eye)

    logits_s = logits_s.masked_fill(neg_mask, -1e4)
    logits_t = logits_t.masked_fill(neg_mask, -1e4)

    with torch.no_grad():
        target_prob = F.softmax(logits_t, dim=-1)
        entropy = -(target_prob * torch.log(target_prob.clamp_min(1e-8))).sum(dim=-1)
        conf = (1.0 - entropy / math.log(target_prob.size(-1))).clamp(0, 1)
        conf = conf.pow(conf_power)

    log_prob_s = F.log_softmax(logits_s, dim=-1)
    loss = -(target_prob * log_prob_s).sum(dim=-1)
    return (loss * conf).sum() / conf.sum().clamp_min(1.0)


def masked_recon_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                      alpha: float = 0.5, beta: float = 0.5):
    if mask.sum().item() == 0:
        return pred.sum() * 0.0
    pred_m = pred[mask]
    targ_m = target[mask]
    loss_mse = F.smooth_l1_loss(pred_m, targ_m)
    loss_cos = 1.0 - F.cosine_similarity(pred_m, targ_m, dim=-1).mean()
    return alpha * loss_mse + beta * loss_cos


def region_pool(patch_tokens: torch.Tensor, grid: int, num_regions: int):
    B, N, C = patch_tokens.shape
    assert N == grid * grid, f"N={N} 与 grid^2={grid*grid} 不一致"
    r_side = int(round(math.sqrt(num_regions)))
    r_side = max(1, r_side)
    block = grid // r_side
    if block * r_side != grid:
        return patch_tokens.mean(dim=1, keepdim=True)
    x = patch_tokens.view(B, grid, grid, C)
    regions = []
    for i in range(r_side):
        for j in range(r_side):
            blk = x[:, i * block:(i + 1) * block, j * block:(j + 1) * block, :]
            regions.append(blk.mean(dim=(1, 2)))
    return torch.stack(regions, dim=1)


def structure_mse(student_patch: torch.Tensor, teacher_patch: torch.Tensor, grid: int, num_regions: int):
    rs = region_pool(student_patch, grid=grid, num_regions=num_regions)
    rt = region_pool(teacher_patch, grid=grid, num_regions=num_regions)
    return F.mse_loss(rs, rt)


def token_gradient_loss(student_patch: torch.Tensor, teacher_patch: torch.Tensor, grid: int):
    B, N, C = student_patch.shape
    s = student_patch.view(B, grid, grid, C)
    t = teacher_patch.view(B, grid, grid, C)
    s_dx = s[:, :, 1:, :] - s[:, :, :-1, :]
    s_dy = s[:, 1:, :, :] - s[:, :-1, :, :]
    t_dx = t[:, :, 1:, :] - t[:, :, :-1, :]
    t_dy = t[:, 1:, :, :] - t[:, :-1, :, :]
    return F.smooth_l1_loss(s_dx, t_dx) + F.smooth_l1_loss(s_dy, t_dy)


def multi_scale_structure_loss(student_patch: torch.Tensor, teacher_patch: torch.Tensor, grid: int, num_regions: int):
    loss_region_fine = structure_mse(student_patch, teacher_patch, grid=grid, num_regions=num_regions)
    loss_region_coarse = structure_mse(student_patch, teacher_patch, grid=grid, num_regions=max(4, num_regions // 4))
    loss_grad = token_gradient_loss(student_patch, teacher_patch, grid=grid)
    return 0.4 * loss_region_fine + 0.2 * loss_region_coarse + 0.4 * loss_grad


# =========================================================
# 4) 配置
# =========================================================
@dataclass
class CFG:
    data_root: str = "C:/D/Data/GlassAI_unlabeled"
    image_size: int = 224
    patch_size: int = 16

    backbone_impl: str = "hf"
    vit_path: str = "C:/D/Project/vit-base-patch16-224"
    hf_model_name_or_path: str = ""
    vit_weight_path: str = ""

    timm_model_name: str = "vit_base_patch16_224"
    timm_pretrained: bool = False
    coach_bin: str = ""

    batch_size: int = 8
    epochs: int = 80
    lr: float = 1e-4
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
    pcl_teacher_temperature: float = 0.07
    pcl_conf_power: float = 1.5
    pcl_num_sampled_patches: int = 64

    num_regions: int = 16

    w_pcl: float = 1.0
    w_mpr: float = 1.5
    w_struct: float = 1.0

    mpr_mse_alpha: float = 0.5
    mpr_cos_beta: float = 0.5

    out_dir: str = "ssl_checkpoints_v2"
    save_every: int = 10
    log_interval: int = 20


def infer_embed_dim_and_grid(vit: CompatibleViTBackbone, image_size: int, patch_size: int, device):
    x = torch.randn(2, 3, image_size, image_size, device=device)
    out = vit(x)
    patch = backbone_to_patch_tokens(out)
    embed_dim = patch.size(-1)
    num_patches = patch.size(1)
    grid = int(round(math.sqrt(num_patches)))
    return embed_dim, num_patches, grid


# =========================================================
# 5) 主训练逻辑
# =========================================================
def main(cfg_path="configs/ssl_pretrain_patch_pcl_v2.yaml"):
    raw = load_yaml_any_encoding(cfg_path) or {}
    cfg = CFG(**raw)

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    os.makedirs(cfg.out_dir, exist_ok=True)

    print(f"[INFO] torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"[INFO] actual device = {device}")
    if device.type == "cuda":
        print(f"[INFO] gpu = {torch.cuda.get_device_name(0)}")

    ds = UnlabeledImageFolder(cfg.data_root, image_size=cfg.image_size)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    vit_cfg = build_vit_backbone_cfg(cfg)
    vit_s = CompatibleViTBackbone(vit_cfg).to(device)
    vit_t = CompatibleViTBackbone(vit_cfg).to(device)
    vit_t.eval()
    for p in vit_t.parameters():
        p.requires_grad_(False)

    embed_dim, num_patches, grid = infer_embed_dim_and_grid(vit_s, cfg.image_size, cfg.patch_size, device)
    print(f"[INFO] embed_dim={embed_dim}, num_patches={num_patches}, grid={grid}x{grid}")
    print(f"[INFO] backbone_impl={vit_cfg.get('backbone_impl')}")

    student = SSLWrapper(vit_s, embed_dim=embed_dim, proj_dim=cfg.proj_dim).to(device)
    teacher = SSLWrapper(vit_t, embed_dim=embed_dim, proj_dim=cfg.proj_dim).to(device)

    teacher.load_state_dict(student.state_dict(), strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    base_weights = normalize_loss_weights(cfg.w_pcl, cfg.w_mpr, cfg.w_struct)
    print(f"[INFO] base loss weights: pcl={base_weights[0]:.4f}, mpr={base_weights[1]:.4f}, struct={base_weights[2]:.4f}")

    max_steps = cfg.epochs * len(dl)
    global_step = 0
    print("[INFO] 开始自监督预训练（SegAlign-SSL v2: PCL + RealMPR + MultiScaleStructure）...")

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        total = 0.0
        total_pcl = 0.0
        total_mpr = 0.0
        total_struct = 0.0
        epoch_start = time.time()
        log_time_acc = 0.0
        log_img_acc = 0
        w_pcl, w_mpr, w_struct = get_dynamic_weights(epoch - 1, cfg.epochs, base_weights)

        pbar = tqdm.tqdm(dl, desc=f"Epoch {epoch}/{cfg.epochs}", ncols=130)
        for step_idx, (xw, xs) in enumerate(pbar, start=1):
            step_start = time.time()

            xw = xw.to(device, non_blocking=True)
            xs = xs.to(device, non_blocking=True)

            with torch.no_grad():
                patch_t, proj_t = teacher(xw)
                patch_t = patch_t.detach()
                proj_t = proj_t.detach()

            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                patch_s, proj_s = student(xs)

                B, N, P = proj_s.shape
                K = min(cfg.pcl_num_sampled_patches, N)
                idx = torch.randint(low=0, high=N, size=(B, K), device=device)
                ps = torch.gather(proj_s, dim=1, index=idx.unsqueeze(-1).expand(B, K, P))
                pt = torch.gather(proj_t, dim=1, index=idx.unsqueeze(-1).expand(B, K, P))
                img_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, K).reshape(-1)

                loss_pcl = patch_kl_contrastive(
                    ps.reshape(-1, P),
                    pt.reshape(-1, P),
                    img_ids=img_ids,
                    tau_s=cfg.temperature,
                    tau_t=cfg.pcl_teacher_temperature,
                    conf_power=cfg.pcl_conf_power,
                ) if w_pcl > 0 else ps.sum() * 0.0

                mask = make_random_mask(B, N, cfg.mask_ratio, device=device)
                xs_masked = mask_image_patches(xs, mask, patch_size=cfg.patch_size, fill_mode=cfg.mask_fill_mode)
                patch_s_masked, _ = student(xs_masked)
                recon_s = student.recon(patch_s_masked)
                loss_mpr = masked_recon_loss(
                    recon_s, patch_t, mask,
                    alpha=cfg.mpr_mse_alpha,
                    beta=cfg.mpr_cos_beta,
                ) if w_mpr > 0 else recon_s.sum() * 0.0

                loss_struct = multi_scale_structure_loss(
                    patch_s, patch_t, grid=grid, num_regions=cfg.num_regions
                ) if w_struct > 0 else patch_s.sum() * 0.0

                loss = w_pcl * loss_pcl + w_mpr * loss_mpr + w_struct * loss_struct

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            m = cosine_schedule(cfg.ema_m_base, cfg.ema_m_final, global_step, max_steps)
            ema_update(teacher, student, m=m)

            total += float(loss.item())
            total_pcl += float(loss_pcl.item())
            total_mpr += float(loss_mpr.item())
            total_struct += float(loss_struct.item())
            global_step += 1

            step_time = time.time() - step_start
            log_time_acc += step_time
            log_img_acc += xw.size(0)

            if step_idx % cfg.log_interval == 0:
                imgs_per_sec = log_img_acc / max(log_time_acc, 1e-6)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "img/s": f"{imgs_per_sec:.2f}",
                    "mask": f"{cfg.mask_ratio:.2f}",
                })
                log_time_acc = 0.0
                log_img_acc = 0

        denom = max(1, len(dl))
        epoch_time = time.time() - epoch_start
        epoch_imgs = denom * cfg.batch_size
        epoch_imgps = epoch_imgs / max(epoch_time, 1e-6)
        print(
            f"[Epoch {epoch}] total={total / denom:.4f} | pcl={total_pcl / denom:.4f} | "
            f"mpr={total_mpr / denom:.4f} | struct={total_struct / denom:.4f} | "
            f"weights=({w_pcl:.3f},{w_mpr:.3f},{w_struct:.3f}) | "
            f"epoch_time={epoch_time:.1f}s | img/s={epoch_imgps:.2f}"
        )

        if (epoch % cfg.save_every == 0) or (epoch == cfg.epochs):
            out_backbone = os.path.join(cfg.out_dir, f"vit_ssl_patchpcl_v2_{epoch}.pth")
            model_obj = getattr(student.vit, "model", student.vit)
            torch.save(model_obj.state_dict(), out_backbone)

            out_meta = os.path.join(cfg.out_dir, f"vit_ssl_patchpcl_v2_{epoch}.meta.pth")
            torch.save({
                "backbone_impl": vit_cfg.get("backbone_impl"),
                "vit_cfg": vit_cfg,
                "version": "SegAlign-SSL-v2-fast",
            }, out_meta)

            out_full = os.path.join(cfg.out_dir, f"ssl_full_v2_{epoch}.pth")
            torch.save({
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "cfg": raw,
                "version": "SegAlign-SSL-v2-fast",
            }, out_full)

            print(f"[SAVE] backbone(state_dict) -> {out_backbone}")
            print(f"[SAVE] backbone(meta)       -> {out_meta}")
            print(f"[SAVE] full                -> {out_full}")

    print("[INFO] 自监督预训练完成。")
    print("[INFO] 将 vit_ssl_patchpcl_v2_*.pth 作为 backbone 权重迁移即可（注意 timm/hf 要一致）。")


if __name__ == "__main__":
    main()
