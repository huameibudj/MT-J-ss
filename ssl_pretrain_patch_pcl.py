import os
import math
import glob
import yaml
import random
from dataclasses import dataclass
from typing import Dict, Any

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
    """把配置里的空字符串/空白字符串转成 None，避免被当成路径/模型名。"""
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and v.strip() == "":
            out[k] = None
        else:
            out[k] = v
    return out


def build_vit_backbone_cfg(cfg) -> Dict[str, Any]:
    """
    将 SSL CFG 字段整理成 CompatibleViTBackbone 需要的 dict。
    """
    vit_cfg = {
        "backbone_impl": str(getattr(cfg, "backbone_impl", "hf")).lower(),
        "img_size": int(getattr(cfg, "image_size", 224)),

        # timm
        "timm_model_name": getattr(cfg, "timm_model_name", "vit_base_patch16_224"),
        "timm_pretrained": bool(getattr(cfg, "timm_pretrained", False)),
        "coach_bin": getattr(cfg, "coach_bin", None),

        # 通用覆盖权重（timm/hf 都支持；hf 表示覆盖 load_state_dict）
        "vit_weight_path": getattr(cfg, "vit_weight_path", None),

        # hf
        "vit_path": getattr(cfg, "vit_path", None),
        "hf_model_name_or_path": getattr(cfg, "hf_model_name_or_path", None),
    }
    return _clean_empty_to_none(vit_cfg)


def backbone_to_patch_tokens(out):
    """
    兼容 timm/hf 两种 backbone 输出：

    - timm: CompatibleViTBackbone 默认 forward 返回 patch tokens (B, N, C)（不含 CLS）
    - hf  : CompatibleViTBackbone forward 返回 HF BaseModelOutputWithPooling
           取 out.last_hidden_state[:, 1:, :] 作为 patch tokens
    """
    if torch.is_tensor(out):
        return out  # (B, N, C)

    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        # (B, 1+N, C), 0 是 CLS
        return out.last_hidden_state[:, 1:, :]

    raise TypeError(f"Unsupported backbone output type: {type(out)}")


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, m: float):
    """EMA 更新：teacher = m*teacher + (1-m)*student"""
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))


def cosine_schedule(base: float, final: float, step: int, max_steps: int):
    """余弦调度（用于 EMA momentum）"""
    if max_steps <= 1:
        return final
    t = step / (max_steps - 1)
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * t))


def normalize_loss_weights(w_pcl: float, w_mpr: float, w_struct: float):
    """
    将三个损失权重绑定：有效项权重和为 1
    允许用户在 config 里设 0 来关闭某个 loss（用于对照实验）
    """
    w = torch.tensor([w_pcl, w_mpr, w_struct], dtype=torch.float32)
    w = torch.clamp(w, min=0.0)

    s = float(w.sum().item())
    if s <= 0:
        raise ValueError("三个损失权重不能全为 0，请至少开启一个 loss。")

    w = w / s
    return float(w[0].item()), float(w[1].item()), float(w[2].item())


# =========================================================
# 1) 共享几何增强（保证 patch-level 对齐）+ 强弱颜色增强
# =========================================================
class SharedCropTwoView:
    """
    同一张图生成 weak/strong 两个视图：
    - 几何部分（RandomResizedCrop + Flip）共享同一组随机参数 -> 保证空间对齐
    - photometric 部分分别强弱不同 -> 保证表观差异
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

        # photometric (弱)
        self.weak_photo = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])

        # photometric (强)
        self.strong_photo = T.Compose([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        ])

    def __call__(self, img: Image.Image):
        # 1) 共享随机裁剪参数
        i, j, h, w = T.RandomResizedCrop.get_params(img, scale=(0.5, 1.0), ratio=(0.75, 1.33))
        img_crop = TF.resized_crop(
            img, i, j, h, w,
            size=(self.image_size, self.image_size),
            interpolation=Image.BILINEAR
        )

        # 2) 共享随机翻转
        if random.random() < 0.5:
            img_crop = TF.hflip(img_crop)
        if random.random() < 0.5:
            img_crop = TF.vflip(img_crop)

        # 3) 生成 weak / strong（仅颜色/模糊不同）
        weak = self.weak_photo(img_crop)
        strong = self.strong_photo(img_crop)

        # 4) 转 tensor
        weak = TF.to_tensor(weak)
        strong = TF.to_tensor(strong)
        return weak, strong


class UnlabeledImageFolder(Dataset):
    """无标注数据集：扫描目录下所有图片"""
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
    """简单 MLP"""
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
    """
    包装 ViT Backbone，使其输出 patch tokens，并附加：
    - patch-level projection head（用于 PCL）
    - recon head（用于 MPR，重建 teacher patch 特征）
    """
    def __init__(self, vit: CompatibleViTBackbone, embed_dim: int, proj_dim: int):
        super().__init__()
        self.vit = vit
        self.proj = MLP(embed_dim, embed_dim, proj_dim, num_layers=2)
        self.recon = MLP(embed_dim, embed_dim, embed_dim, num_layers=2)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x):
        out = self.vit(x)
        patch = backbone_to_patch_tokens(out)  # (B, N, C)
        proj_patch = self.proj(patch)          # (B, N, P)
        return patch, proj_patch


# =========================================================
# 3) Mask / Loss
# =========================================================
def make_random_mask(B: int, N: int, mask_ratio: float, device):
    """生成随机 patch mask：True 表示被 mask"""
    n_mask = int(round(N * mask_ratio))
    mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(N, device=device)[:n_mask]
        mask[b, idx] = True
    return mask


def apply_mask(tokens: torch.Tensor, mask: torch.Tensor, mask_token: torch.Tensor):
    """将 mask 的位置替换成 mask_token"""
    B, N, C = tokens.shape
    mt = mask_token.expand(B, N, C)
    return torch.where(mask.unsqueeze(-1), mt, tokens)


def info_nce(z_s: torch.Tensor, z_t: torch.Tensor, temperature: float):
    """
    InfoNCE（批内负样本）
    z_s/z_t: (M, D) 已归一化
    """
    z_s = F.normalize(z_s, dim=-1)
    z_t = F.normalize(z_t, dim=-1)
    logits = (z_s @ z_t.t()) / temperature
    labels = torch.arange(z_s.size(0), device=z_s.device)
    return F.cross_entropy(logits, labels)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """仅在 mask==True 的 patch 上做 MSE"""
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=pred.device)
    diff = (pred - target).pow(2).mean(dim=-1)  # (B, N)
    return diff[mask].mean()


def region_pool(patch_tokens: torch.Tensor, grid: int, num_regions: int):
    """
    patch_tokens: (B, N, C), N=grid*grid
    将 patch 网格划分为 sqrt(R) x sqrt(R) 个区域，做区域平均池化
    返回: (B, R, C)
    """
    B, N, C = patch_tokens.shape
    assert N == grid * grid, f"N={N} 与 grid^2={grid*grid} 不一致"

    r_side = int(round(math.sqrt(num_regions)))
    r_side = max(1, r_side)
    block = grid // r_side
    if block * r_side != grid:
        # 不能整除就退化为整体平均
        return patch_tokens.mean(dim=1, keepdim=True)

    x = patch_tokens.view(B, grid, grid, C)
    regions = []
    for i in range(r_side):
        for j in range(r_side):
            blk = x[:, i*block:(i+1)*block, j*block:(j+1)*block, :]  # (B, block, block, C)
            regions.append(blk.mean(dim=(1, 2)))
    return torch.stack(regions, dim=1)  # (B, R, C)


def structure_mse(student_patch: torch.Tensor, teacher_patch: torch.Tensor, grid: int, num_regions: int):
    """结构损失：区域级特征一致性（teacher stop-grad）"""
    rs = region_pool(student_patch, grid=grid, num_regions=num_regions)
    rt = region_pool(teacher_patch, grid=grid, num_regions=num_regions)
    return F.mse_loss(rs, rt)


# =========================================================
# 4) 配置
# =========================================================
@dataclass
class CFG:
    # 数据
    data_root: str = "C:/D/Data/GlassAI_unlabeled"
    image_size: int = 224
    patch_size: int = 16  # vit-base-patch16

    # Backbone 切换
    backbone_impl: str = "hf"  # "timm" or "hf"

    # HF 风格
    vit_path: str = "C:/D/Project/vit-base-patch16-224"  # 本地目录：config.json + pytorch_model.bin
    hf_model_name_or_path: str = ""  # 可留空；若不提供 vit_path 才会用它
    vit_weight_path: str = ""        # 可选覆盖权重（hf/timm 都支持）

    # timm 风格
    timm_model_name: str = "vit_base_patch16_224"
    timm_pretrained: bool = False
    coach_bin: str = ""  # COACH 整包 bin（含 visual.trunk.*）

    # 训练
    batch_size: int = 8
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.05
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    use_amp: bool = True

    # EMA
    ema_m_base: float = 0.996
    ema_m_final: float = 0.9999

    # Mask
    mask_ratio: float = 0.4

    # PCL（patch-level）
    proj_dim: int = 256
    temperature: float = 0.2
    pcl_num_sampled_patches: int = 64

    # Structure
    num_regions: int = 16

    # loss 权重（会自动归一化到和=1，可用 0 关闭某个 loss）
    w_pcl: float = 1.0
    w_mpr: float = 1.0
    w_struct: float = 1.0

    # 保存
    out_dir: str = "ssl_checkpoints"
    save_every: int = 10


def infer_embed_dim_and_grid(vit: CompatibleViTBackbone, image_size: int, patch_size: int, device):
    """推断 embed_dim, num_patches, grid_size"""
    x = torch.randn(2, 3, image_size, image_size, device=device)
    out = vit(x)
    patch = backbone_to_patch_tokens(out)  # (B, N, C)
    embed_dim = patch.size(-1)
    num_patches = patch.size(1)
    grid = int(round(math.sqrt(num_patches)))
    return embed_dim, num_patches, grid


# =========================================================
# 5) 主训练逻辑
# =========================================================
def main(cfg_path="configs/ssl_pretrain_patch_pcl.yaml"):
    raw = load_yaml_any_encoding(cfg_path) or {}
    cfg = CFG(**raw)

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 数据
    ds = UnlabeledImageFolder(cfg.data_root, image_size=cfg.image_size)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True
    )

    # 构建 student/teacher backbone（完全同结构）
    vit_cfg = build_vit_backbone_cfg(cfg)
    vit_s = CompatibleViTBackbone(vit_cfg).to(device)
    vit_t = CompatibleViTBackbone(vit_cfg).to(device)

    # teacher 不参与梯度
    vit_t.eval()
    for p in vit_t.parameters():
        p.requires_grad_(False)

    # 推断维度与 patch 网格
    embed_dim, num_patches, grid = infer_embed_dim_and_grid(vit_s, cfg.image_size, cfg.patch_size, device)
    print(f"[INFO] embed_dim={embed_dim}, num_patches={num_patches}, grid={grid}x{grid}")
    print(f"[INFO] backbone_impl={vit_cfg.get('backbone_impl')}")

    # wrapper
    student = SSLWrapper(vit_s, embed_dim=embed_dim, proj_dim=cfg.proj_dim).to(device)
    teacher = SSLWrapper(vit_t, embed_dim=embed_dim, proj_dim=cfg.proj_dim).to(device)

    # 初始化 teacher=student
    teacher.load_state_dict(student.state_dict(), strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 优化器
    optim = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    # 三个损失权重：自动归一化（并允许 0 关闭）
    w_pcl, w_mpr, w_struct = normalize_loss_weights(cfg.w_pcl, cfg.w_mpr, cfg.w_struct)
    print(f"[INFO] 归一化后损失权重: w_pcl={w_pcl:.4f}, w_mpr={w_mpr:.4f}, w_struct={w_struct:.4f}")

    max_steps = cfg.epochs * len(dl)
    global_step = 0

    print("[INFO] 开始自监督预训练（patch-level PCL + MPR + Structure）...")

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        total = 0.0

        for xw, xs in tqdm.tqdm(dl, desc=f"Epoch {epoch}/{cfg.epochs}", ncols=100):
            xw = xw.to(device, non_blocking=True)
            xs = xs.to(device, non_blocking=True)

            # teacher 输出（弱增强），stop-grad
            with torch.no_grad():
                patch_t, proj_t = teacher(xw)
                patch_t = patch_t.detach()
                proj_t = proj_t.detach()

            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                # student 输出（强增强）
                patch_s, proj_s = student(xs)

                # ========== (1) patch-level PCL ==========
                B, N, P = proj_s.shape
                K = min(cfg.pcl_num_sampled_patches, N)
                idx = torch.randint(low=0, high=N, size=(B, K), device=device)

                ps = torch.gather(proj_s, dim=1, index=idx.unsqueeze(-1).expand(B, K, P))  # (B,K,P)
                pt = torch.gather(proj_t, dim=1, index=idx.unsqueeze(-1).expand(B, K, P))  # (B,K,P)

                loss_pcl = info_nce(ps.reshape(-1, P), pt.reshape(-1, P), temperature=cfg.temperature) if w_pcl > 0 else ps.sum() * 0.0

                # ========== (2) MPR ==========
                mask = make_random_mask(B, N, cfg.mask_ratio, device=device)
                patch_s_masked = apply_mask(patch_s, mask, student.mask_token)
                recon_s = student.recon(patch_s_masked)  # (B,N,C)
                loss_mpr = masked_mse(recon_s, patch_t, mask) if w_mpr > 0 else recon_s.sum() * 0.0

                # ========== (3) Structure ==========
                loss_struct = structure_mse(patch_s, patch_t, grid=grid, num_regions=cfg.num_regions) if w_struct > 0 else patch_s.sum() * 0.0

                # 总损失（权重已归一化和=1）
                loss = w_pcl * loss_pcl + w_mpr * loss_mpr + w_struct * loss_struct

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            # EMA 更新 teacher
            m = cosine_schedule(cfg.ema_m_base, cfg.ema_m_final, global_step, max_steps)
            ema_update(teacher, student, m=m)

            total += float(loss.item())
            global_step += 1

        avg = total / max(1, len(dl))
        print(f"[Epoch {epoch}] total_loss={avg:.4f}")

        # 保存
        if (epoch % cfg.save_every == 0) or (epoch == cfg.epochs):
            out_backbone = os.path.join(cfg.out_dir, f"vit_ssl_patchpcl_{epoch}.pth")
            torch.save(student.vit.model.state_dict(), out_backbone)

            out_meta = os.path.join(cfg.out_dir, f"vit_ssl_patchpcl_{epoch}.meta.pth")
            torch.save({
                "backbone_impl": vit_cfg.get("backbone_impl"),
                "vit_cfg": vit_cfg,
            }, out_meta)

            out_full = os.path.join(cfg.out_dir, f"ssl_full_{epoch}.pth")
            torch.save({
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "cfg": raw,
            }, out_full)

            print(f"[SAVE] backbone(state_dict) -> {out_backbone}")
            print(f"[SAVE] backbone(meta)       -> {out_meta}")
            print(f"[SAVE] full                -> {out_full}")

    print("[INFO] 自监督预训练完成。")
    print("[INFO] 将 vit_ssl_patchpcl_*.pth 作为 backbone 权重迁移即可（注意 timm/hf 要一致）。")


if __name__ == "__main__":
    main()
