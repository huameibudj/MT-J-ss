
import math
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ViTFPNHead(nn.Module):
    """
    将多层 ViT patch token 融合成较高分辨率的 pixel feature。
    输入默认是 3 个尺度相同(14x14)但语义层次不同的 token map：
      [low_level, mid_level, high_level]
    做法：
      1) 每层先 1x1 lateral 投影到统一通道
      2) top-down 累加融合
      3) 3x3 平滑
      4) 两次上采样: 14->28->56
    """
    def __init__(self, embed_dim: int, fpn_dim: int = 256, num_scales: int = 3, out_upsample_x4: bool = True):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.fpn_dim = int(fpn_dim)
        self.num_scales = int(num_scales)
        self.out_upsample_x4 = bool(out_upsample_x4)

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim, self.fpn_dim, kernel_size=1, bias=False)
            for _ in range(self.num_scales)
        ])
        self.lateral_norms = nn.ModuleList([
            nn.GroupNorm(1, self.fpn_dim) for _ in range(self.num_scales)
        ])

        self.smooth_convs = nn.ModuleList([
            ConvGNAct(self.fpn_dim, self.fpn_dim, k=3, s=1, p=1)
            for _ in range(self.num_scales)
        ])

        self.up_block1 = nn.Sequential(
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(1, self.fpn_dim),
            nn.GELU(),
            ConvGNAct(self.fpn_dim, self.fpn_dim, k=3, s=1, p=1),
        )
        self.up_block2 = nn.Sequential(
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(1, self.fpn_dim),
            nn.GELU(),
            ConvGNAct(self.fpn_dim, self.fpn_dim, k=3, s=1, p=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")

    def _tokens_to_map(self, patch_tokens: torch.Tensor, patch_hw: Tuple[int, int]) -> torch.Tensor:
        b, n, c = patch_tokens.shape
        h, w = patch_hw
        if h * w != n:
            raise RuntimeError(f"[ERROR] patch_hw={patch_hw} 与 token 数 N={n} 不匹配")
        return patch_tokens.transpose(1, 2).contiguous().view(b, c, h, w)

    def forward(self, multi_scale_patch_tokens: List[torch.Tensor], patch_hw: Tuple[int, int]) -> torch.Tensor:
        if len(multi_scale_patch_tokens) != self.num_scales:
            raise ValueError(
                f"[ERROR] 期望 {self.num_scales} 个尺度特征, 实际得到 {len(multi_scale_patch_tokens)} 个"
            )

        feats = []
        for i, tok in enumerate(multi_scale_patch_tokens):
            x = self._tokens_to_map(tok, patch_hw)
            x = self.lateral_convs[i](x)
            x = self.lateral_norms[i](x)
            feats.append(x)

        p = feats[-1]
        outs = [None] * self.num_scales
        outs[-1] = self.smooth_convs[-1](p)

        for i in range(self.num_scales - 2, -1, -1):
            p = feats[i] + p
            outs[i] = self.smooth_convs[i](p)

        fused = outs[0] + outs[1] + outs[2]

        if self.out_upsample_x4:
            fused = self.up_block1(fused)
            fused = self.up_block2(fused)

        return fused


class MaskHead(nn.Module):
    """
    FPN 版：
    - query -> query class logits
    - query -> mask embedding
    - 多层 patch tokens -> ViT-FPN -> high-res pixel feature
    - einsum(mask_head(q), pixel_feat) -> query masks
    """
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        fpn_dim: int = 256,
        fpn_layers: int = 3,
        out_upsample_x4: bool = True,
        logit_scale_init: Optional[float] = None,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.fpn_dim = int(fpn_dim)
        self.fpn_layers = int(fpn_layers)

        self.class_embed = nn.Linear(self.embed_dim, self.num_classes)

        self.mask_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.fpn_dim),
        )

        self.pixel_head = ViTFPNHead(
            embed_dim=self.embed_dim,
            fpn_dim=self.fpn_dim,
            num_scales=self.fpn_layers,
            out_upsample_x4=out_upsample_x4,
        )

        if logit_scale_init is None:
            logit_scale_init = math.sqrt(self.fpn_dim)

        self.logit_scale_log = nn.Parameter(
            torch.tensor(math.log(float(logit_scale_init)), dtype=torch.float32)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        multi_scale_patch_tokens: List[torch.Tensor],
        patch_hw: Tuple[int, int],
    ):
        """
        query:       [B, Q, D]
        multi_scale_patch_tokens:
                     List[[B, N, D], ...]
        patch_hw:    (H, W)

        return:
          mask_logits:      [B, C, H', W']
          class_logits_img: [B, C]
          class_logits_q:   [B, Q, C]
          mask_logits_q:    [B, Q, H', W']
        """
        class_logits_q = self.class_embed(query)  # [B, Q, C]

        class_slots = class_logits_q[:, :self.num_classes, :]  # [B, C, C]
        class_logits_img = torch.diagonal(class_slots, offset=0, dim1=1, dim2=2)  # [B, C]

        mask_embed = self.mask_embed(query)  # [B, Q, fpn_dim]
        mask_embed = F.normalize(mask_embed, dim=-1, eps=1e-6)

        pixel_feat = self.pixel_head(multi_scale_patch_tokens, patch_hw=patch_hw)  # [B, fpn_dim, H', W']
        pixel_feat = F.normalize(pixel_feat, dim=1, eps=1e-6)

        mask_logits_q = torch.einsum("bqd,bdhw->bqhw", mask_embed, pixel_feat)

        scale = torch.exp(
            self.logit_scale_log.clamp(min=math.log(1.0), max=math.log(100.0))
        )
        mask_logits_q = mask_logits_q * scale

        mask_logits = mask_logits_q[:, :self.num_classes, :, :]  # [B, C, H', W']

        return mask_logits, class_logits_img, class_logits_q, mask_logits_q


class EoMT(nn.Module):
    def __init__(
        self,
        vit_model,
        num_classes: int = 6,
        num_queries: int = 16,
        L1: int = 9,
        L2: int = 3,
        joint_query_blocks: int = 1,
        mask_gate_floor: float = 0.30,
        aux_return_all: bool = True,
        fpn_dim: int = 256,
        fpn_layers: int = 3,
        out_upsample_x4: bool = True,
        final_upsample_to_input: bool = True,
    ):
        super().__init__()
        self.backbone = vit_model
        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.L1 = int(L1)
        self.L2 = int(L2)

        self.joint_query_blocks = max(1, int(joint_query_blocks))
        self.mask_gate_floor = float(mask_gate_floor)
        self.aux_return_all = bool(aux_return_all)

        self.fpn_dim = int(fpn_dim)
        self.fpn_layers = int(fpn_layers)
        self.out_upsample_x4 = bool(out_upsample_x4)
        self.final_upsample_to_input = bool(final_upsample_to_input)

        self.embed_dim = int(self.backbone.model.config.hidden_size)

        if self.num_queries < self.num_classes:
            raise ValueError(
                f"[ERROR] num_queries({self.num_queries}) 必须 >= num_classes({self.num_classes})"
            )

        self.queries = nn.Parameter(
            torch.randn(1, self.num_queries, self.embed_dim) * 0.02
        )

        self.mask_head = MaskHead(
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            fpn_dim=self.fpn_dim,
            fpn_layers=self.fpn_layers,
            out_upsample_x4=self.out_upsample_x4,
            logit_scale_init=math.sqrt(self.fpn_dim),
        )

        self.last_query_class_logits = None
        self.last_aux_outputs: List[Dict[str, torch.Tensor]] = []

    def _apply_block(self, blk, x: torch.Tensor) -> torch.Tensor:
        out = blk(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def _get_blocks(self):
        m = self.backbone.model
        if hasattr(m, "encoder") and hasattr(m.encoder, "layer"):
            return m.encoder.layer
        if hasattr(m, "blocks"):
            return m.blocks
        raise AttributeError(
            "[ERROR] backbone.model 未找到 encoder.layer 或 blocks，无法取 transformer blocks"
        )

    def _tokens_to_patch(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise RuntimeError(f"[ERROR] tokens 维度错误，期望 [B,1+N,D] 或 [B,N,D]，得到 {tokens.shape}")
        if tokens.shape[1] >= 2:
            return tokens[:, 1:, :]
        return tokens

    def _make_patch_gate_from_masks(
        self,
        mask_logits: torch.Tensor,
        patch_hw: Tuple[int, int],
    ) -> torch.Tensor:
        b, c, hm, wm = mask_logits.shape
        hp, wp = patch_hw

        prob = F.softmax(mask_logits, dim=1)

        if c > 1:
            fg_prob = 1.0 - prob[:, :1, :, :]
        else:
            fg_prob = prob.max(dim=1, keepdim=True)[0]

        gate = F.interpolate(
            fg_prob,
            size=(hp, wp),
            mode="bilinear",
            align_corners=False
        )

        gate = gate.clamp(0.0, 1.0)
        gate = self.mask_gate_floor + (1.0 - self.mask_gate_floor) * gate
        gate = gate.flatten(2).transpose(1, 2).contiguous()
        return gate

    def _build_multi_scale_tokens(
        self,
        hidden_states,
        patch_tokens_i: torch.Tensor,
    ) -> List[torch.Tensor]:
        idx0 = self.L1
        idx1 = min(self.L1 + 1, len(hidden_states) - 1)

        patch_tokens_l1 = self._tokens_to_patch(hidden_states[idx0])
        patch_tokens_mid = self._tokens_to_patch(hidden_states[idx1])

        return [patch_tokens_l1, patch_tokens_mid, patch_tokens_i]

    def forward(self, x: torch.Tensor):
        vit_out = self.backbone(pixel_values=x, output_hidden_states=True)
        hidden_states = vit_out.hidden_states
        if hidden_states is None:
            raise RuntimeError("[ERROR] backbone 未返回 hidden_states，请检查 vit_backbone")

        blocks = self._get_blocks()
        n_blocks = len(blocks)

        if self.L1 < 0 or self.L1 >= len(hidden_states):
            raise ValueError(
                f"[ERROR] L1={self.L1} 超出 hidden_states 长度={len(hidden_states)}"
            )

        tokens = hidden_states[self.L1]
        patch_tokens = tokens[:, 1:, :]
        b, n, d = patch_tokens.shape

        h = int(math.sqrt(n))
        if h * h != n:
            raise RuntimeError(f"[ERROR] patch 数 N={n} 不是平方数，无法恢复二维网格")
        w = h

        queries = self.queries.repeat(b, 1, 1)

        start = min(max(self.L1, 0), n_blocks - 1)
        end = min(start + self.joint_query_blocks, n_blocks)

        aux_outputs: List[Dict[str, torch.Tensor]] = []

        combined = torch.cat([patch_tokens, queries], dim=1)

        for i in range(start, end):
            combined = self._apply_block(blocks[i], combined)

            patch_tokens_i = combined[:, :n, :]
            queries_i = combined[:, n:, :]

            multi_scale_patch_tokens = self._build_multi_scale_tokens(hidden_states, patch_tokens_i)

            masks_i, class_logits_img_i, class_logits_q_i, _ = self.mask_head(
                queries_i,
                multi_scale_patch_tokens,
                patch_hw=(h, w)
            )

            if self.final_upsample_to_input and masks_i.shape[-2:] != x.shape[-2:]:
                masks_i = F.interpolate(
                    masks_i,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            aux_outputs.append({
                "masks": masks_i,
                "class_logits_img": class_logits_img_i,
                "query_class_logits": class_logits_q_i,
            })

            if i < end - 1:
                gate = self._make_patch_gate_from_masks(masks_i, patch_hw=(h, w))
                patch_tokens_i = patch_tokens_i * gate
                combined = torch.cat([patch_tokens_i, queries_i], dim=1)

        final = aux_outputs[-1]

        self.last_query_class_logits = final["query_class_logits"]
        self.last_aux_outputs = aux_outputs if self.aux_return_all else [final]

        return final["masks"], final["class_logits_img"]
