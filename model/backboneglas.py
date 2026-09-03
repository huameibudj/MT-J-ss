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
    支持 timm / HF 的 ViT Backbone。

    关键修改：
    1. 不再强制 img_size == 224。
    2. timm.create_model 显式传入 img_size。
    3. 加载 224 预训练权重时，如果 pos_embed 不匹配，自动插值。
    """

    def __init__(self, cfg: dict):
        super().__init__()

        self.cfg = cfg
        self.backbone_impl = str(cfg.get("backbone_impl", "timm")).lower()
        self.img_size = int(cfg.get("img_size", 224))

        if self.img_size % 16 != 0:
            raise ValueError(
                f"[ERROR] img_size={self.img_size} 不能被 patch_size=16 整除。"
                f"建议使用 224 / 320 / 384。"
            )

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
    # 1) timm 实现
    # ============================================================
    def _init_timm(self):
        timm_model_name = self.cfg.get("timm_model_name", "vit_base_patch16_224")
        timm_pretrained = bool(self.cfg.get("timm_pretrained", False))
        coach_bin = self.cfg.get("coach_bin", None)
        vit_weight_path = self.cfg.get("vit_weight_path", None)

        print("[INFO] backbone_impl = timm")
        print(f"[INFO] timm_model_name={timm_model_name}, timm_pretrained={timm_pretrained}")
        print(f"[INFO] img_size={self.img_size}")

        # 关键：这里必须传 img_size，否则模型内部 pos_embed 仍然是 224 对应的 14x14
        self.model = timm.create_model(
            timm_model_name,
            pretrained=timm_pretrained,
            num_classes=0,
            global_pool="",
            img_size=self.img_size,
        )

        hidden = int(getattr(self.model, "embed_dim", 768))
        self.model.config = DummyConfig(hidden)
        print(f"[INFO] 兼容层: self.model.config.hidden_size={hidden}")

        if coach_bin:
            self._load_coach_bin_to_timm(coach_bin)
        elif vit_weight_path:
            self._load_timm_state_dict(vit_weight_path)
        else:
            print("[INFO] 未指定 coach_bin / vit_weight_path，将使用 timm_pretrained 或随机初始化")

    def _normalize_state_dict(self, sd):
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        elif isinstance(sd, dict) and "model_state" in sd:
            sd = sd["model_state"]

        new_sd = {}
        for k, v in sd.items():
            nk = k

            for prefix in ("module.", "model.", "vit_model.", "backbone."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]

            # 如果是分类头，直接丢掉
            if nk.startswith("head.") or nk.startswith("fc.") or nk.startswith("classifier."):
                continue

            new_sd[nk] = v

        return new_sd

    def _load_timm_state_dict(self, weight_path: str):
        assert os.path.exists(weight_path), f"[ERROR] vit_weight_path 不存在: {weight_path}"
        print(f"[INFO] 加载 timm 风格视觉权重: {weight_path}")

        sd = torch.load(weight_path, map_location="cpu")
        sd = self._normalize_state_dict(sd)

        sd = self._resize_pos_embed_if_needed(sd, self.model.pos_embed)

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"[INFO] timm 权重加载完成 | missing={len(missing)} unexpected={len(unexpected)}")

        if len(missing) > 0:
            print("[WARN] timm missing examples:", missing[:20])
        if len(unexpected) > 0:
            print("[WARN] timm unexpected examples:", unexpected[:20])

    def _load_coach_bin_to_timm(self, coach_bin: str):
        assert os.path.exists(coach_bin), f"[ERROR] coach_bin 不存在: {coach_bin}"
        print(f"[INFO] 加载 coach 权重（整包 bin）: {coach_bin}")

        sd = torch.load(coach_bin, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        trunk = {
            k.replace("visual.trunk.", ""): v
            for k, v in sd.items()
            if k.startswith("visual.trunk.")
        }

        if len(trunk) == 0:
            raise RuntimeError("[ERROR] coach bin 中未找到 visual.trunk.*，请确认文件是否正确")

        trunk = self._resize_pos_embed_if_needed(trunk, self.model.pos_embed)

        missing, unexpected = self.model.load_state_dict(trunk, strict=False)
        print(f"[INFO] 权重加载完成 | missing={len(missing)} unexpected={len(unexpected)}")

        if len(missing) > 0:
            print("[WARN] coach missing examples:", missing[:20])
        if len(unexpected) > 0:
            print("[WARN] coach unexpected examples:", unexpected[:20])

    # ============================================================
    # 2) HuggingFace 实现
    # ============================================================
    def _init_hf(self):
        hf_model_name_or_path = self.cfg.get("hf_model_name_or_path", None)
        vit_path = self.cfg.get("vit_path", None)
        weight_path = self.cfg.get("vit_weight_path", None)

        print("[INFO] backbone_impl = hf")
        print(f"[INFO] img_size={self.img_size}")

        model_src = vit_path if vit_path else hf_model_name_or_path
        if model_src is None:
            raise KeyError(
                "使用 backbone_impl=hf 时，需要提供 vit_path 或 hf_model_name_or_path"
            )

        print(f"[INFO] HF ViT 来源: {model_src}")

        if vit_path:
            if not os.path.isdir(vit_path):
                raise FileNotFoundError(f"[ERROR] vit_path 不是目录或不存在: {vit_path}")

            self.model = ViTModel.from_pretrained(
                vit_path,
                add_pooling_layer=False,
                local_files_only=True,
                ignore_mismatched_sizes=True,
            )
            print("[INFO] 已从 vit_path 加载 HF 预训练权重（local_files_only=True）")
        else:
            config = ViTConfig.from_pretrained(model_src)
            config.add_pooling_layer = False
            config.image_size = self.img_size
            self.model = ViTModel(config)

        if weight_path:
            assert os.path.exists(weight_path), f"[ERROR] vit_weight_path 不存在: {weight_path}"
            print(f"[INFO] 加载 HF 权重(覆盖): {weight_path}")

            sd = torch.load(weight_path, map_location="cpu")
            sd = self._normalize_state_dict(sd)

            new_sd = {}
            for k, v in sd.items():
                if k.startswith("classifier"):
                    continue
                if k.startswith("vit."):
                    k = k.replace("vit.", "")
                new_sd[k] = v

            missing, unexpected = self.model.load_state_dict(new_sd, strict=False)
            missing = [k for k in missing if not k.startswith("pooler")]
            print(f"[INFO] HF 权重覆盖完成 | missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print("[INFO] 未指定 HF 权重覆盖：将使用已加载的预训练或随机初始化")

    # ============================================================
    # 通用：pos_embed 插值
    # ============================================================
    def _resize_pos_embed_if_needed(self, sd: dict, pos_model: torch.Tensor):
        if "pos_embed" not in sd:
            print("[WARN] checkpoint 中没有 pos_embed，跳过位置编码插值")
            return sd

        pos_ckpt = sd["pos_embed"]

        if pos_ckpt.shape == pos_model.shape:
            print(f"[INFO] pos_embed 尺寸匹配: {tuple(pos_ckpt.shape)}")
            return sd

        print(f"[INFO] pos_embed 尺寸不匹配，执行插值: {tuple(pos_ckpt.shape)} -> {tuple(pos_model.shape)}")

        # timm ViT 常见格式：[1, 1 + H*W, C]
        cls_ckpt = pos_ckpt[:, :1, :]
        patch_ckpt = pos_ckpt[:, 1:, :]

        old_n = patch_ckpt.shape[1]
        new_n = pos_model.shape[1] - 1
        c = patch_ckpt.shape[2]

        old_g = int(math.sqrt(old_n))
        new_g = int(math.sqrt(new_n))

        if old_g * old_g != old_n:
            raise ValueError(f"[ERROR] checkpoint patch token 数不是平方数: {old_n}")
        if new_g * new_g != new_n:
            raise ValueError(f"[ERROR] model patch token 数不是平方数: {new_n}")

        patch_ckpt = patch_ckpt.reshape(1, old_g, old_g, c).permute(0, 3, 1, 2)

        patch_ckpt = F.interpolate(
            patch_ckpt,
            size=(new_g, new_g),
            mode="bicubic",
            align_corners=False,
        )

        patch_ckpt = patch_ckpt.permute(0, 2, 3, 1).reshape(1, new_g * new_g, c)

        sd["pos_embed"] = torch.cat([cls_ckpt, patch_ckpt], dim=1)

        print(f"[INFO] pos_embed 插值完成: {old_g}x{old_g} -> {new_g}x{new_g}")

        return sd

    # ============================================================
    # forward：对外统一 HF 风格接口
    # ============================================================
    def forward(self, x=None, pixel_values=None, output_hidden_states=False):
        if pixel_values is not None:
            x = pixel_values
        if x is None:
            raise TypeError("必须提供输入 x 或 pixel_values")

        if self.backbone_impl == "hf":
            return self.model(pixel_values=x, output_hidden_states=output_hidden_states)

        if output_hidden_states:
            return self._timm_forward_hidden_states(x)

        out = self._timm_forward_hidden_states(x, output_hidden_states=False)
        patch_tokens = out.last_hidden_state[:, 1:, :]
        return patch_tokens

    def _timm_forward_hidden_states(self, x, output_hidden_states=True):
        """timm ViT：手动收集 hidden_states，返回 ViTOutput"""
        b = x.shape[0]

        t = self.model.patch_embed(x)

        cls = self.model.cls_token.expand(b, -1, -1)
        t = torch.cat([cls, t], dim=1)

        if t.shape[1] != self.model.pos_embed.shape[1]:
            raise RuntimeError(
                f"[ERROR] token 数与 pos_embed 不匹配: "
                f"tokens={t.shape}, pos_embed={self.model.pos_embed.shape}. "
                f"请确认 img_size / resize_hw / patch_size 是否一致。"
            )

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
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
        )