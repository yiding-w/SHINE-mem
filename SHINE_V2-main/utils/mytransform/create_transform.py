"""
Factory function for creating W-Transform modules.

All code that needs to create a transform module should call
``create_transform()`` from this file. This ensures a single entry point
for instantiation and makes it easy to add new transform types in the future.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def create_transform(
    cfg: dict,
    model_cfg=None,
    num_layers: int = 1,
    tp_mode: bool = False,
    tp_rank: int = 0,
    tp_world: int = 1,
    tp_group=None,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
    llm_model=None,
) -> nn.Module:
    """Factory function to create a W-Transform module from config.

    Returns the appropriate transform module based on cfg["method"]:
      - "identity": returns IdentityTransform (no parameters, zero overhead)
      - "zero": returns ZeroTransform (returns None, disabling wdict injection)
      - "compressed_mlp": returns CompressedMLPTransform (learned transform)

    All external code should use this function to create transforms.

    Args:
        cfg: Transform config dict. Must contain "method" key.
            For "identity": no additional keys needed.
            For "compressed_mlp": additional keys: k, mlp_ratio, activation.
        model_cfg: The model config (has .path, .lora_ranks, etc.).
            Required for "compressed_mlp" to determine projection dimensions.
        num_layers: Number of LLM layers. Each layer gets its own independent
            set of CompressMLP instances.
        tp_mode: Whether in TP mode.
        tp_rank: TP rank.
        tp_world: TP world size.
        tp_group: TP process group.
        device: Device to place parameters on.
        dtype: Parameter dtype.
        llm_model: Optional LLM model instance. If provided, projection
            dimensions are auto-detected from the model's init_lora_dict()
            method, making the transform model-agnostic.

    Returns:
        An nn.Module with forward(layer_wdict, layer_idx) -> transformed_wdict.
    """
    method = cfg.get("method", "identity")

    if method == "identity":
        from utils.mytransform.identity import IdentityTransform
        return IdentityTransform()

    elif method == "zero":
        from utils.mytransform.zero import ZeroTransform
        return ZeroTransform()

    elif method == "compressed_mlp":
        return _create_compressed_mlp(
            cfg=cfg,
            model_cfg=model_cfg,
            num_layers=num_layers,
            tp_mode=tp_mode,
            tp_rank=tp_rank,
            tp_world=tp_world,
            tp_group=tp_group,
            device=device,
            dtype=dtype,
            llm_model=llm_model,
        )

    else:
        raise ValueError(
            f"Unknown w_transform method: '{method}'. "
            f"Supported methods: 'identity', 'zero', 'compressed_mlp'."
        )


def _create_compressed_mlp(
    cfg: dict,
    model_cfg,
    num_layers: int = 1,
    tp_mode: bool = False,
    tp_rank: int = 0,
    tp_world: int = 1,
    tp_group=None,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
    llm_model=None,
) -> nn.Module:
    """Internal helper to create a CompressedMLPTransform.

    Auto-detects projection dimensions from the LLM model instance by calling
    init_lora_dict() on a full_attention layer. This makes the transform
    completely model-agnostic — no hardcoded dimension calculations or key
    name mappings are needed.
    """
    from omegaconf import OmegaConf
    from utils.mytransform.compressed_mlp import CompressedMLPTransform

    if model_cfg is None:
        raise ValueError(
            "model_cfg is required for 'compressed_mlp' w_transform method "
            "(needed to determine projection dimensions from LLM config)."
        )

    # Get lora_ranks dict
    if hasattr(model_cfg, "lora_ranks"):
        lr_raw = model_cfg.lora_ranks
        if hasattr(lr_raw, "_metadata"):
            lora_ranks_dict = OmegaConf.to_container(lr_raw, resolve=True)
        else:
            lora_ranks_dict = dict(lr_raw)
    else:
        lora_ranks_dict = {}

    # --- Auto-detect proj_dims from LLM model instance ---
    if llm_model is None:
        raise ValueError(
            "llm_model is required for 'compressed_mlp' w_transform method "
            "(needed to auto-detect projection dimensions from the model)."
        )

    active_proj_dims = _probe_proj_dims_from_model(
        llm_model, lora_ranks_dict
    )

    if not active_proj_dims:
        logger.warning(
            "[create_transform] No active projections found for compressed_mlp. "
            "Falling back to IdentityTransform."
        )
        from utils.mytransform.identity import IdentityTransform
        return IdentityTransform()

    module = CompressedMLPTransform(
        cfg=cfg,
        proj_dims=active_proj_dims,
        num_layers=num_layers,
        tp_mode=tp_mode,
        tp_rank=tp_rank,
        tp_world=tp_world,
        tp_group=tp_group,
    )

    if device is not None:
        module = module.to(device)
    if dtype is not None:
        module = module.to(dtype)

    # Log parameter count
    total_params = sum(p.numel() for p in module.parameters())
    logger.info(
        f"[create_transform] Created CompressedMLPTransform with "
        f"{total_params:,} parameters ({total_params * 2 / 1024 / 1024:.1f} MB at bf16), "
        f"active projections: {list(active_proj_dims.keys())}"
    )

    return module


def _probe_proj_dims_from_model(
    llm_model,
    lora_ranks_dict: dict,
) -> Dict[str, Tuple[int, int]]:
    """Auto-detect projection dimensions by probing the LLM model.

    Calls init_lora_dict() on the first full_attention layer to get the
    actual wdict structure and extract (d_in, d_out) for each projection.
    This is completely model-agnostic — works for any model that implements
    init_lora_dict().

    The wdict structure returned by init_lora_dict() is:
        {"attention": {"q_query": {"A":..,"B":..}, "k": {...}, ...},
         "mlp": {"gate": {"A":..,"B":..}, ...}}

    From each leaf: d_in = A.shape[1], d_out = B.shape[2].

    Args:
        llm_model: The LLM model instance (e.g. LoraQwen3_5ForCausalLM).
        lora_ranks_dict: Dict of lora_ranks from config (used as input to
            init_lora_dict).

    Returns:
        Dict mapping flat proj_key -> (d_in_full, d_out_full).
        Keys are the same as those used in wdict (e.g. "q_query", "k", "gate").
    """
    # Find the first full_attention layer
    layers = llm_model.model.layers
    target_layer = None
    for layer in layers:
        layer_type = getattr(layer, "layer_type", None)
        if layer_type == "full_attention":
            target_layer = layer
            break

    if target_layer is None:
        # Fallback: try the first layer (some models don't have layer_type)
        target_layer = layers[0]

    if not hasattr(target_layer, "init_lora_dict"):
        logger.warning(
            "[create_transform] LLM layer does not have init_lora_dict(). "
            "Cannot auto-detect proj_dims."
        )
        return {}

    # Call init_lora_dict with a dummy scale to get the structure
    dummy_device = torch.device("cpu")
    dummy_dtype = torch.float32
    dummy_scale = 1.0

    with torch.no_grad():
        sample_loradict = target_layer.init_lora_dict(
            lora_ranks_dict, dummy_scale, dummy_device, dummy_dtype
        )

    # Extract proj_dims from the sample loradict
    # Structure: {"attention": {"q_query": {"A":[1,in,r],"B":[1,r,out]}, ...}, "mlp": {"gate": {...}, ...}}
    active_proj_dims = {}
    _extract_dims_recursive(sample_loradict, active_proj_dims, path_keys=[])

    # Clean up dummy tensors
    del sample_loradict

    logger.info(
        f"[create_transform] Auto-detected proj_dims from model: "
        f"{{{', '.join(f'{k}: ({v[0]}, {v[1]})' for k, v in active_proj_dims.items())}}}"
    )

    return active_proj_dims


def _extract_dims_recursive(
    d: Optional[dict],
    result: Dict[str, Tuple[int, int]],
    path_keys: list,
) -> None:
    """Recursively extract (d_in, d_out) from a loradict structure.

    Leaf nodes have "A" and "B" keys. The projection name is the last
    path key (e.g. "q_query", "k", "gate", "down").
    """
    if d is None:
        return

    # Leaf: has "A" and "B" keys
    if "A" in d and "B" in d:
        proj_name = path_keys[-1] if path_keys else "unknown"
        A = d["A"]  # [1, d_in, r]
        B = d["B"]  # [1, r, d_out]
        d_in = A.shape[1]
        d_out = B.shape[2]
        result[proj_name] = (d_in, d_out)
        return

    # Recurse into sub-dicts
    for key, value in d.items():
        if value is None:
            continue
        if isinstance(value, dict):
            _extract_dims_recursive(value, result, path_keys + [key])



