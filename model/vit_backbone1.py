# === models/vit_backbone_tv2hf_v2.py ===
# Robust HuggingFace ViTModel backbone loader that STRICTLY aligns weights.
#
# Supports:
#   (1) torchvision ViT-B/16 style state_dict (your ssl_pretrain vit_backbone_best.bin)
#       keys like: class_token, conv_proj.*, encoder.pos_embedding,
#                  encoder.layers.encoder_layer_{i}.*
#   (2) HuggingFace ViTModel / ViTForImageClassification style state_dict
#       keys like: embeddings.*, encoder.layer.*, layernorm.*
#       or prefixed: vit.embeddings.*, vit.encoder.layer.*, vit.layernorm.*
#
# Behavior:
#   - Always prints a strict-load report (missing/unexpected, with examples)
#   - Any mismatch -> raises RuntimeError (NO silent mismatch)
#
# Note:
#   This module requires `transformers` in YOUR environment.

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from transformers import ViTModel, ViTConfig
except Exception as e:
    raise ImportError("transformers is required. Please `pip install transformers`.") from e


def _split_qkv(w: torch.Tensor, dim: int = 0):
    """Split (3*D, D) or (3*D,) into q,k,v along dim."""
    assert w.size(dim) % 3 == 0
    d = w.size(dim) // 3
    if dim == 0:
        return w[:d].clone(), w[d:2*d].clone(), w[2*d:].clone()
    return w[:, :d].clone(), w[:, d:2*d].clone(), w[:, 2*d:].clone()


def convert_torchvision_vit_to_hf(
    sd_tv: dict,
    *,
    num_layers: int = 12,
    hidden: int = 768,
) -> dict[str, torch.Tensor]:
    """
    Convert torchvision.models.vit_* state_dict -> transformers.ViTModel state_dict.

    This is tailored for ViT-Base patch16 224 (hidden=768, layers=12),
    but works for same-structure variants if (hidden, num_layers) match.
    """
    sd_hf: dict[str, torch.Tensor] = {}

    # ---- embeddings ----
    sd_hf["embeddings.cls_token"] = sd_tv["class_token"].clone()
    sd_hf["embeddings.patch_embeddings.projection.weight"] = sd_tv["conv_proj.weight"].clone()
    sd_hf["embeddings.patch_embeddings.projection.bias"] = sd_tv["conv_proj.bias"].clone()
    sd_hf["embeddings.position_embeddings"] = sd_tv["encoder.pos_embedding"].clone()

    # ---- encoder blocks ----
    for i in range(num_layers):
        tv_prefix = f"encoder.layers.encoder_layer_{i}."
        hf_prefix = f"encoder.layer.{i}."

        # layer norms
        sd_hf[hf_prefix + "layernorm_before.weight"] = sd_tv[tv_prefix + "ln_1.weight"].clone()
        sd_hf[hf_prefix + "layernorm_before.bias"] = sd_tv[tv_prefix + "ln_1.bias"].clone()
        sd_hf[hf_prefix + "layernorm_after.weight"] = sd_tv[tv_prefix + "ln_2.weight"].clone()
        sd_hf[hf_prefix + "layernorm_after.bias"] = sd_tv[tv_prefix + "ln_2.bias"].clone()

        # attention qkv
        in_w = sd_tv[tv_prefix + "self_attention.in_proj_weight"]
        in_b = sd_tv[tv_prefix + "self_attention.in_proj_bias"]
        qw, kw, vw = _split_qkv(in_w, dim=0)
        qb, kb, vb = _split_qkv(in_b, dim=0)

        sd_hf[hf_prefix + "attention.attention.query.weight"] = qw
        sd_hf[hf_prefix + "attention.attention.key.weight"] = kw
        sd_hf[hf_prefix + "attention.attention.value.weight"] = vw
        sd_hf[hf_prefix + "attention.attention.query.bias"] = qb
        sd_hf[hf_prefix + "attention.attention.key.bias"] = kb
        sd_hf[hf_prefix + "attention.attention.value.bias"] = vb

        # attention output proj
        sd_hf[hf_prefix + "attention.output.dense.weight"] = sd_tv[tv_prefix + "self_attention.out_proj.weight"].clone()
        sd_hf[hf_prefix + "attention.output.dense.bias"] = sd_tv[tv_prefix + "self_attention.out_proj.bias"].clone()

        # MLP
        sd_hf[hf_prefix + "intermediate.dense.weight"] = sd_tv[tv_prefix + "mlp.0.weight"].clone()
        sd_hf[hf_prefix + "intermediate.dense.bias"] = sd_tv[tv_prefix + "mlp.0.bias"].clone()
        sd_hf[hf_prefix + "output.dense.weight"] = sd_tv[tv_prefix + "mlp.3.weight"].clone()
        sd_hf[hf_prefix + "output.dense.bias"] = sd_tv[tv_prefix + "mlp.3.bias"].clone()

    # ---- final layernorm ----
    if "encoder.ln.weight" in sd_tv:
        sd_hf["layernorm.weight"] = sd_tv["encoder.ln.weight"].clone()
        sd_hf["layernorm.bias"] = sd_tv["encoder.ln.bias"].clone()

    return sd_hf


def _strip_common_prefixes(sd: dict) -> dict:
    def strip(k: str) -> str:
        for pref in ("model.", "module."):
            if k.startswith(pref):
                return k[len(pref):]
        return k
    return {strip(k): v for k, v in sd.items()}


def _maybe_unwrap(sd):
    # lightning/common wrappers
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        return sd["state_dict"]
    return sd


def _print_report(missing, unexpected, max_show: int = 40):
    print(f"[STRICT-LOAD REPORT] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("[STRICT-LOAD REPORT] missing (first):")
        for k in list(missing)[:max_show]:
            print("  -", k)
    if len(unexpected) > 0:
        print("[STRICT-LOAD REPORT] unexpected (first):")
        for k in list(unexpected)[:max_show]:
            print("  -", k)


def _align_keyspace_to_model(sd: dict[str, torch.Tensor], model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Make small, safe key-space adjustments based on target model keys:
      - if model keys are prefixed with 'vit.' but sd is not, prepend it
      - if sd is prefixed with 'vit.' but model is not, strip it
      - drop keys not in model (avoid unexpected keys)
    """
    model_keys = set(model.state_dict().keys())
    has_vit_prefix_model = any(k.startswith("vit.") for k in model_keys)
    has_vit_prefix_sd = any(k.startswith("vit.") for k in sd.keys())

    if has_vit_prefix_model and (not has_vit_prefix_sd):
        sd = {("vit." + k): v for k, v in sd.items()}
    elif (not has_vit_prefix_model) and has_vit_prefix_sd:
        sd = {k[len("vit."):]: v for k, v in sd.items() if k.startswith("vit.")}

    # Drop classifier/head/pooler if any (common when exporting from ViTForImageClassification)
    drop_prefixes = ("classifier.", "head.", "heads.", "pooler.", "fc.", "pre_logits.")
    sd = {k: v for k, v in sd.items() if not k.startswith(drop_prefixes)}

    # Finally drop anything not in model to avoid unexpected keys
    sd = {k: v for k, v in sd.items() if k in model_keys}

    # If the HF model tracks position_ids as a persistent buffer, inject it.
    # This avoids a common strict-load 'missing key' issue across transformers versions.
    pos_id_keys = [k for k in model_keys if k.endswith('embeddings.position_ids')]
    for pk in pos_id_keys:
        if pk not in sd:
            # position_embeddings is [1, num_positions, hidden]
            pe_key = pk.replace('position_ids', 'position_embeddings')
            if pe_key in model_keys and pe_key in sd:
                num_pos = sd[pe_key].shape[1]
            else:
                # fallback to standard ViT (1+196)
                num_pos = 197
            sd[pk] = torch.arange(num_pos, dtype=torch.long).unsqueeze(0)

    # Re-drop any keys not in model (in case we injected wrong)
    sd = {k: v for k, v in sd.items() if k in model_keys}
    return sd


def strict_load_into_vitmodel(model: nn.Module, sd_in: dict, *, tag: str = "weights"):
    """
    Print missing/unexpected (strict=False), then raise if mismatch, else strict=True load.
    """
    missing, unexpected = model.load_state_dict(sd_in, strict=False)
    print(f"[STRICT-LOAD] tag={tag}")
    _print_report(missing, unexpected)
    if len(missing) > 0 or len(unexpected) > 0:
        raise RuntimeError(
            f"Backbone weight mismatch for {tag}. "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    model.load_state_dict(sd_in, strict=True)
    print("[STRICT-LOAD] ✅ perfect match (strict=True)")


class CompatibleViTBackbone(nn.Module):
    """
    HuggingFace ViTModel backbone.

    forward(x) returns patch tokens: [B, N, C] by default.
    """
    def __init__(
        self,
        model_name_or_path: str = "google/vit-base-patch16-224-in21k",
        weight_path: str | None = None,
        image_size: int = 224,
        use_cls_token: bool = False,
        force_from_config: bool = False,
        strict: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.use_cls_token = bool(use_cls_token)
        self.verbose = bool(verbose)

        # Build model
        if force_from_config:
            config = ViTConfig(
                image_size=image_size,
                patch_size=16,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            self.model = ViTModel(config, add_pooling_layer=False)
        else:
            config = ViTConfig.from_pretrained(model_name_or_path)
            config.image_size = image_size
            self.model = ViTModel(config, add_pooling_layer=False)

        if not weight_path:
            if self.verbose:
                print(f"[INFO] Using HuggingFace pretrained: {model_name_or_path}")
            return

        # --- load sd (safe if your file is a raw state_dict) ---
        try:
            sd = torch.load(weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            # older torch
            sd = torch.load(weight_path, map_location="cpu")

        sd = _maybe_unwrap(sd)
        if not isinstance(sd, dict):
            raise TypeError(f"Unsupported weight format: {type(sd)}")

        sd = _strip_common_prefixes(sd)

        # detect torchvision ViT keys
        is_tv = ("class_token" in sd) and ("encoder.pos_embedding" in sd) and ("conv_proj.weight" in sd)
        if is_tv:
            if self.verbose:
                print(f"[INFO] Detected torchvision-style ViT weights -> converting: {weight_path}")
            sd = convert_torchvision_vit_to_hf(
                sd,
                num_layers=self.model.config.num_hidden_layers,
                hidden=self.model.config.hidden_size,
            )
        else:
            if self.verbose:
                print(f"[INFO] Detected HuggingFace-style weights (or other) -> using as-is: {weight_path}")

        # align prefixes and drop non-model keys
        sd = _align_keyspace_to_model(sd, self.model)

        if strict:
            strict_load_into_vitmodel(self.model, sd, tag=weight_path)
        else:
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if self.verbose:
                print(f"[INFO] Loaded with strict=False: {weight_path}")
                _print_report(missing, unexpected)

    def forward(self, x: torch.Tensor):
        """Forward for EoMT.

        Returns:
            patch_tokens: (B, N, D)
            grid_size: (Hp, Wp) where Hp=H/patch_size, Wp=W/patch_size

        Notes:
            - This wrapper assumes your training pipeline resizes inputs to a
              fixed square size (e.g. 224). If you pass other sizes, grid_size
              is computed from the runtime tensor shape.
        """
        B, C, H, W = x.shape
        ps = int(getattr(self.model.config, 'patch_size', 16))
        Hp, Wp = H // ps, W // ps

        outputs = self.model(pixel_values=x, output_hidden_states=False)
        hidden = outputs.last_hidden_state  # (B, 1+N, D)

        if self.use_cls_token:
            # For segmentation, you almost always want patch tokens, so keep
            # use_cls_token=False. We still return a tuple to keep API stable.
            return hidden[:, 0], (Hp, Wp)

        return hidden[:, 1:], (Hp, Wp)
