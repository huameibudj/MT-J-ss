import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskHead(nn.Module):
    def __init__(self, embed_dim, num_classes, logit_scale_init=None):
        """
        稳定版掩膜头：
        - query -> class logits
        - normalized query / feature similarity -> mask logits
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

        # 可学习 logit scale，初值设成 sqrt(D) 更接近常见 attention / cosine-sim 量级
        if logit_scale_init is None:
            logit_scale_init = math.sqrt(self.embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init), dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        # 更保守的初始化，避免 head 一开始输出过大
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, query, features):
        """
        query:    [B, Q, D]
        features: [B, N, D]

        返回：
          mask_logits:  [B, C, N]
          class_logits: [B, C]
        """
        # [B, Q, C]
        class_logits_q = self.class_embed(query)

        # [B, Q, D]
        mask_embed = self.mask_embed(query)

        # ===== 稳定关键：L2 normalize 后做相似度 =====
        mask_embed = F.normalize(mask_embed, dim=-1, eps=1e-6)
        features = F.normalize(features, dim=-1, eps=1e-6)

        # [B, Q, N]
        mask_logits_q = torch.einsum("bqd,bnd->bqn", mask_embed, features)

        # 限制 scale 为正，避免训练中出现反号/异常放大
        scale = torch.clamp(self.logit_scale, min=1.0, max=100.0)
        mask_logits_q = mask_logits_q * scale
        # ==========================================

        C = class_logits_q.shape[-1]

        # 取前 C 个 query 作为每类 mask 生成器
        mask_logits = mask_logits_q[:, :C, :]   # [B, C, N]

        # class 分支仍然保留均值聚合
        class_logits = class_logits_q.mean(dim=1)  # [B, C]

        return mask_logits, class_logits


class EoMT(nn.Module):
    def __init__(self, vit_model, num_classes=6, num_queries=10, L1=9, L2=3):
        super().__init__()
        self.backbone = vit_model
        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.L1 = int(L1)
        self.L2 = int(L2)

        # 兼容：HF/timm 都提供 model.config.hidden_size
        self.embed_dim = int(self.backbone.model.config.hidden_size)

        assert self.num_queries >= self.num_classes, (
            f"[ERROR] num_queries({self.num_queries}) 必须 >= num_classes({self.num_classes})"
        )

        # learnable queries
        # 原版是 randn 直接初始化，容易初始范数偏大；这里缩小到 0.02
        self.queries = nn.Parameter(
            torch.randn(1, self.num_queries, self.embed_dim) * 0.02
        )

        # 掩膜头
        self.mask_head = MaskHead(
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            logit_scale_init=math.sqrt(self.embed_dim),
        )

    def _apply_block(self, blk, x):
        """兼容 HF block 返回 tuple / timm 返回 tensor"""
        out = blk(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def _get_blocks(self):
        """HF: encoder.layer, timm: blocks"""
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
        x: [B, 3, 224, 224]

        返回:
          masks: [B, C, 14, 14]
          class_logits: [B, C]
        """
        # 统一走 backbone 的 HF 风格输出（timm/hf 都兼容）
        vit_out = self.backbone(pixel_values=x, output_hidden_states=True)
        hidden_states = vit_out.hidden_states
        if hidden_states is None:
            raise RuntimeError("[ERROR] backbone 未返回 hidden_states，请检查 vit_backbone")

        if self.L1 >= len(hidden_states):
            raise ValueError(
                f"[ERROR] L1={self.L1} 超出 hidden_states 长度={len(hidden_states)}"
            )

        # [B, 197, D]
        tokens = hidden_states[self.L1]

        # 去掉 cls token -> [B, 196, D]
        patch_tokens = tokens[:, 1:, :]
        B, N, D = patch_tokens.shape

        # 拼接 learnable queries
        queries = self.queries.repeat(B, 1, 1)               # [B, Q, D]
        combined = torch.cat([patch_tokens, queries], dim=1) # [B, N+Q, D]

        # 用后续 L2 个 block 做交互（HF/timm 都可）
        blocks = self._get_blocks()
        start = self.L1
        end = min(self.L1 + self.L2, len(blocks))
        for i in range(start, end):
            combined = self._apply_block(blocks[i], combined)

        # 拆回 patch / queries
        patch_tokens = combined[:, :N, :]   # [B, N, D]
        queries = combined[:, N:, :]        # [B, Q, D]

        # mask + class
        mask_logits, class_logits = self.mask_head(queries, patch_tokens)  # [B, C, N], [B, C]

        # 还原空间结构：N=14*14
        H = int(math.sqrt(N))
        if H * H != N:
            raise RuntimeError(f"[ERROR] N={N} 不是平方数，无法还原 (H,W)")

        mask_logits = mask_logits.view(B, self.num_classes, H, H)  # [B, C, H, H]

        return mask_logits, class_logits