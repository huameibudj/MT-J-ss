import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:
    timm = None

try:
    from transformers import ViTModel, ViTConfig
except Exception:
    ViTModel, ViTConfig = None, None


class ViTOutput:
    """模拟 HuggingFace ViTModel 的输出：last_hidden_state / hidden_states"""
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class DummyConfig:
    """兼容：model.config.hidden_size"""
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size


class CompatibleViTBackbone(nn.Module):
    """
    最终整合版 ViT Backbone（只改 config 就能跑）：

    支持：
    - backbone_impl = "timm"（推荐，支持 coach_bin）
    - backbone_impl = "hf"（支持本地 vit_path 离线加载权重）

    forward 兼容：
    - backbone(pixel_values=x, output_hidden_states=True) -> HF 输出
    - backbone(x) -> patch tokens (B,196,C)

    关键：
    - timm + coach_bin：自动抽 visual.trunk.*，自动插值 pos_embed（适配 224）
    """

    def __init__(self, cfg: dict):
        super().__init__()

        self.cfg = cfg
        self.backbone_impl = str(cfg.get("backbone_impl", "timm")).lower()
        self.img_size = int(cfg.get("img_size", 224))
        assert self.img_size == 224, "本工程固定 224 输入（ViT-B/16@224）"

        if self.backbone_impl == "timm":
            if timm is None:
                raise ImportError("未安装 timm：请 pip install timm")
            self._init_timm()

        elif self.backbone_impl == "hf":
            if ViTModel is None:
                raise ImportError("未安装 transformers：请 pip install transformers")
            self._init_hf()

        else:
            raise ValueError("backbone_impl 仅支持 'timm' 或 'hf'")

    # ============================================================
    # 1) timm 实现（原功能不变）
    # ============================================================
    def _init_timm(self):
        timm_model_name = self.cfg.get("timm_model_name", "vit_base_patch16_224")
        timm_pretrained = bool(self.cfg.get("timm_pretrained", False))
        coach_bin = self.cfg.get("coach_bin", None)
        vit_weight_path = self.cfg.get("vit_weight_path", None)

        print("[INFO] backbone_impl = timm")
        print(f"[INFO] timm_model_name={timm_model_name}, timm_pretrained={timm_pretrained}")

        # 构建 timm ViT（去掉分类头）
        self.model = timm.create_model(
            timm_model_name,
            pretrained=timm_pretrained,
            num_classes=0,
            global_pool=""
        )

        # 提供 config.hidden_size 兼容 EoMT
        hidden = int(getattr(self.model, "embed_dim", 768))
        self.model.config = DummyConfig(hidden)
        print(f"[INFO] 兼容层: self.model.config.hidden_size={hidden}")

        # 权重优先级：coach_bin > vit_weight_path > 不加载（或 timm_pretrained）
        if coach_bin:
            self._load_coach_bin_to_timm(coach_bin)
        elif vit_weight_path:
            self._load_timm_state_dict(vit_weight_path)
        else:
            print("[INFO] 未指定 coach_bin / vit_weight_path，将使用 timm_pretrained 或随机初始化")

    def _load_timm_state_dict(self, weight_path: str):
        assert os.path.exists(weight_path), f"[ERROR] vit_weight_path 不存在: {weight_path}"
        print(f"[INFO] 加载 timm 风格视觉权重: {weight_path}")

        sd = torch.load(weight_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        # 兼容 model. 前缀
        if any(k.startswith("model.") for k in sd.keys()):
            sd = {k.replace("model.", ""): v for k, v in sd.items()}

        sd = self._resize_pos_embed_if_needed(sd, self.model.pos_embed)

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"[INFO] timm 权重加载完成 | missing={len(missing)} unexpected={len(unexpected)}")

    def _load_coach_bin_to_timm(self, coach_bin: str):
        assert os.path.exists(coach_bin), f"[ERROR] coach_bin 不存在: {coach_bin}"
        print(f"[INFO] 加载 coach 权重（整包 bin）: {coach_bin}")

        sd = torch.load(coach_bin, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        trunk = {k.replace("visual.trunk.", ""): v for k, v in sd.items() if k.startswith("visual.trunk.")}
        if len(trunk) == 0:
            raise RuntimeError("[ERROR] coach bin 中未找到 visual.trunk.*，请确认文件是否正确")

        trunk = self._resize_pos_embed_if_needed(trunk, self.model.pos_embed)

        missing, unexpected = self.model.load_state_dict(trunk, strict=False)
        print(f"[INFO] 权重加载完成 | missing={len(missing)} unexpected={len(unexpected)}")

    # ============================================================
    # 2) HuggingFace 实现（离线本地 vit_path 会真正加载预训练）
    # ============================================================
    def _init_hf(self):
        hf_model_name_or_path = self.cfg.get("hf_model_name_or_path", None)
        vit_path = self.cfg.get("vit_path", None)
        weight_path = self.cfg.get("vit_weight_path", None)

        print("[INFO] backbone_impl = hf")

        # 优先本地 vit_path（目录含 config.json + pytorch_model.bin），否则用 hf_model_name_or_path
        model_src = vit_path if vit_path else hf_model_name_or_path
        if model_src is None:
            raise KeyError("使用 backbone_impl=hf 时，需要提供 vit_path（含config.json+pytorch_model.bin）或 hf_model_name_or_path")

        print(f"[INFO] HF ViT 来源: {model_src}")

        # 情况 A：提供 vit_path（本地目录）=> 直接 from_pretrained 读取 config + 权重（离线）
        if vit_path:
            # 保险检查：本地目录应存在
            if not os.path.isdir(vit_path):
                raise FileNotFoundError(f"[ERROR] vit_path 不是目录或不存在: {vit_path}")

            # 强制离线：只从本地读，避免任何 huggingface.co 探测
            self.model = ViTModel.from_pretrained(
                vit_path,
                add_pooling_layer=False,
                local_files_only=True,
            )
            print("[INFO] 已从 vit_path 加载 HF 预训练权重（local_files_only=True）")

        # 情况 B：没给 vit_path，只给了模型名/路径 => 维持你原来的行为（可能联网）
        else:
            print("[INFO] 未提供 vit_path，将使用 config 构建模型（默认随机初始化；如需预训练请提供 vit_path 或设置离线缓存）")
            config = ViTConfig.from_pretrained(model_src)
            config.add_pooling_layer = False
            self.model = ViTModel(config)

        # 权重覆盖加载（可选）：如果你显式给 vit_weight_path，就按你的原逻辑覆盖
        if weight_path:
            assert os.path.exists(weight_path), f"[ERROR] vit_weight_path 不存在: {weight_path}"
            print(f"[INFO] 加载 HF 权重(覆盖): {weight_path}")

            sd = torch.load(weight_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]

            # 去掉 classifier / vit. 前缀（兼容 ViTForImageClassification）
            new_sd = {}
            for k, v in sd.items():
                if k.startswith("classifier"):
                    continue
                if k.startswith("vit."):
                    k = k.replace("vit.", "")
                new_sd[k] = v

            missing, unexpected = self.model.load_state_dict(new_sd, strict=False)
            # 忽略 pooler 缺失
            missing = [k for k in missing if not k.startswith("pooler")]
            print(f"[INFO] HF 权重覆盖完成 | missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print("[INFO] 未指定 HF 权重覆盖：将使用已加载的预训练（若 vit_path）或随机初始化（若仅 config）")

    # ============================================================
    # 通用：pos_embed 插值（用于 timm 的 pos_embed）
    # ============================================================
    def _resize_pos_embed_if_needed(self, sd: dict, pos_model: torch.Tensor):
        if "pos_embed" not in sd:
            return sd
        pos_ckpt = sd["pos_embed"]
        if pos_ckpt.shape == pos_model.shape:
            return sd

        print(f"[INFO] pos_embed 尺寸不匹配，执行插值: {tuple(pos_ckpt.shape)} -> {tuple(pos_model.shape)}")

        cls_ckpt = pos_ckpt[:, :1, :]
        patch_ckpt = pos_ckpt[:, 1:, :]

        N1 = patch_ckpt.shape[1]
        N2 = pos_model.shape[1] - 1
        C = patch_ckpt.shape[2]

        gs1 = int(math.sqrt(N1))
        gs2 = int(math.sqrt(N2))
        assert gs1 * gs1 == N1, f"[ERROR] checkpoint patch token 数不是平方数: {N1}"
        assert gs2 * gs2 == N2, f"[ERROR] model patch token 数不是平方数: {N2}"

        patch_ckpt = patch_ckpt.reshape(1, gs1, gs1, C).permute(0, 3, 1, 2)
        patch_ckpt = F.interpolate(patch_ckpt, size=(gs2, gs2), mode="bilinear", align_corners=False)
        patch_ckpt = patch_ckpt.permute(0, 2, 3, 1).reshape(1, gs2 * gs2, C)

        sd["pos_embed"] = torch.cat([cls_ckpt, patch_ckpt], dim=1)
        return sd

    # ============================================================
    # forward：对外统一 HF 风格接口（原行为不变）
    # ============================================================
    def forward(self, x=None, pixel_values=None, output_hidden_states=False):
        if pixel_values is not None:
            x = pixel_values
        if x is None:
            raise TypeError("必须提供输入 x 或 pixel_values")

        if self.backbone_impl == "hf":
            # 直接走 HF 的 forward
            return self.model(pixel_values=x, output_hidden_states=output_hidden_states)

        # timm：手动构造 HF 风格输出
        if output_hidden_states:
            return self._timm_forward_hidden_states(x)

        # 默认返回 patch tokens（不含 CLS）
        out = self._timm_forward_hidden_states(x, output_hidden_states=False)
        patch_tokens = out.last_hidden_state[:, 1:, :]
        return patch_tokens

    def _timm_forward_hidden_states(self, x, output_hidden_states=True):
        """timm ViT：手动收集 hidden_states，返回 ViTOutput"""
        B = x.shape[0]
        t = self.model.patch_embed(x)                 # (B,196,C)
        cls = self.model.cls_token.expand(B, -1, -1)  # (B,1,C)
        t = torch.cat([cls, t], dim=1)                # (B,197,C)
        t = t + self.model.pos_embed
        t = self.model.pos_drop(t)

        hidden_states = []
        if output_hidden_states:
            hidden_states.append(t)

        for blk in self.model.blocks:
            t = blk(t)
            if output_hidden_states:
                hidden_states.append(t)

        if hasattr(self.model, "norm") and self.model.norm is not None:
            t = self.model.norm(t)
            if output_hidden_states and len(hidden_states) > 0:
                hidden_states[-1] = t

        return ViTOutput(
            last_hidden_state=t,
            hidden_states=tuple(hidden_states) if output_hidden_states else None
        )
