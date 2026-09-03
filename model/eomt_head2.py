import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelUpscaleHead(nn.Module):
    def __init__(self, embed_dim, up_scale=2):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.up_scale = int(up_scale)

        self.proj = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.embed_dim),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, self.embed_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")

    def forward(self, feat_2d):
        """
        feat_2d: [B, D, H, W]
        return : [B, D, H', W']
        """
        x = self.proj(feat_2d)
        if self.up_scale > 1:
            x = F.interpolate(
                x,
                scale_factor=float(self.up_scale),
                mode="bilinear",
                align_corners=False
            )
        return x


class MaskHead(nn.Module):
    def __init__(self, embed_dim, num_classes, up_scale=2, logit_scale_init=None):
        """
        稳定优先版：
        - query -> query-level class logits
        - query -> mask embedding
        - patch features -> 2D pixel embedding -> upscale
        - einsum(mask_head(q), upscale(x)) 生成 mask logits
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)

        self.class_embed = nn.Linear(self.embed_dim, self.num_classes)

        self.mask_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.pixel_head = PixelUpscaleHead(
            embed_dim=self.embed_dim,
            up_scale=up_scale
        )

        if logit_scale_init is None:
            logit_scale_init = math.sqrt(self.embed_dim)

        # 用 log-param 保证 scale 为正，且更平滑
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

    def forward(self, query, patch_tokens, patch_hw):
        """
        query:       [B, Q, D]
        patch_tokens:[B, N, D]
        patch_hw:    (H, W)

        return:
          mask_logits:      [B, C, H', W']
          class_logits_img: [B, C]
          class_logits_q:   [B, Q, C]
        """
        B, Q, D = query.shape
        H, W = patch_hw
        N = patch_tokens.shape[1]
        if H * W != N:
            raise RuntimeError(f"[ERROR] patch_hw={patch_hw} 与 N={N} 不匹配")

        # -------- class branch --------
        class_logits_q = self.class_embed(query)  # [B, Q, C]

        # 用前 C 个 query 的“对角类别得分”形成图像级 class logits
        # 即第 i 个 class-slot query 对第 i 类的响应
        class_slots = class_logits_q[:, :self.num_classes, :]  # [B, C, C]
        class_logits_img = torch.diagonal(class_slots, offset=0, dim1=1, dim2=2)  # [B, C]

        # -------- mask branch --------
        # query -> mask embedding
        mask_embed = self.mask_embed(query)  # [B, Q, D]
        mask_embed = F.normalize(mask_embed, dim=-1, eps=1e-6)

        # patch token -> 2D feature -> upscale
        feat_2d = patch_tokens.transpose(1, 2).contiguous().view(B, D, H, W)  # [B, D, H, W]
        pixel_feat = self.pixel_head(feat_2d)  # [B, D, H', W']
        pixel_feat = F.normalize(pixel_feat, dim=1, eps=1e-6)

        # einsum(mask_head(q), upscale(x))
        mask_logits_q = torch.einsum("bqd,bdhw->bqhw", mask_embed, pixel_feat)

        scale = torch.exp(
            self.logit_scale_log.clamp(min=math.log(1.0), max=math.log(100.0))
        )
        mask_logits_q = mask_logits_q * scale

        # 前 C 个 query 作为类别槽位对应的 mask
        mask_logits = mask_logits_q[:, :self.num_classes, :, :]  # [B, C, H', W']

        return mask_logits, class_logits_img, class_logits_q


class EoMT(nn.Module):
    def __init__(self, vit_model, num_classes=6, num_queries=10, L1=9, L2=3, up_scale=2):
        super().__init__()
        self.backbone = vit_model
        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.L1 = int(L1)
        self.L2 = int(L2)

        self.embed_dim = int(self.backbone.model.config.hidden_size)

        assert self.num_queries >= self.num_classes, (
            f"[ERROR] num_queries({self.num_queries}) 必须 >= num_classes({self.num_classes})"
        )

        # 更保守的 query 初始化
        self.queries = nn.Parameter(
            torch.randn(1, self.num_queries, self.embed_dim) * 0.02
        )

        self.mask_head = MaskHead(
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            up_scale=up_scale,
            logit_scale_init=math.sqrt(self.embed_dim),
        )

        # 训练时供外部读取，不改变主 forward 接口
        self.last_query_class_logits = None

    def _apply_block(self, blk, x):
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

    def forward(self, x):
        """
        x: [B, 3, H, W]

        return:
          masks: [B, C, h, w]
          class_logits_img: [B, C]
        """
        vit_out = self.backbone(pixel_values=x, output_hidden_states=True)
        hidden_states = vit_out.hidden_states
        if hidden_states is None:
            raise RuntimeError("[ERROR] backbone 未返回 hidden_states，请检查 vit_backbone")

        if self.L1 >= len(hidden_states):
            raise ValueError(
                f"[ERROR] L1={self.L1} 超出 hidden_states 长度={len(hidden_states)}"
            )

        tokens = hidden_states[self.L1]  # [B, 1+N, D]
        patch_tokens = tokens[:, 1:, :]
        B, N, D = patch_tokens.shape

        H = int(math.sqrt(N))
        if H * H != N:
            raise RuntimeError(f"[ERROR] N={N} 不是平方数，无法还原 (H,W)")
        W = H

        queries = self.queries.repeat(B, 1, 1)               # [B, Q, D]
        combined = torch.cat([patch_tokens, queries], dim=1) # [B, N+Q, D]

        blocks = self._get_blocks()
        start = self.L1
        end = min(self.L1 + self.L2, len(blocks))
        for i in range(start, end):
            combined = self._apply_block(blocks[i], combined)

        patch_tokens = combined[:, :N, :]   # [B, N, D]
        queries = combined[:, N:, :]        # [B, Q, D]

        masks, class_logits_img, class_logits_q = self.mask_head(
            queries, patch_tokens, patch_hw=(H, W)
        )

        # 给训练脚本读取 query-level class logits
        self.last_query_class_logits = class_logits_q

        return masks, class_logits_img