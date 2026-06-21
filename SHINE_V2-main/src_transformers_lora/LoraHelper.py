"""
LoRA helper utilities shared across model families.

Contains:
  - ``_lora_linear_numel_detail``: compute LoRA params for one linear layer.
  - ``SUPPORTED_MODELS``: registry mapping model config name → model family string.
  - ``compute_layer_lora_params_numel``: dispatch per-layer LoRA param count
    to the appropriate model-family implementation.
"""

from typing import Dict


# ---- Supported model families ----
# Maps model config name → model family string.
# When adding a new model, register it here.
SUPPORTED_MODELS: Dict[str, str] = {
    "Qwen3_6-35B-A3B": "qwen3_5moe",
    "Qwen3_6-27B": "qwen3_5",
    "Qwen3-30B-A3B-Instruct-2507": "qwen3moe",
}


def _lora_linear_numel_detail(name: str, in_features: int, out_features: int, r: int, bias: bool) -> tuple:
    """
    Compute LoRA params for one linear layer and return (numel, detail_string).

    The detail string shows the arithmetic breakdown, e.g.:
        "q_query: in=2048 * r=1 + out=4096 * r=1 + bias=0 = 6144"
    """
    a_numel = in_features * r
    b_numel = out_features * r
    c_numel = (out_features if bias else 0) if r > 0 else 0
    total = a_numel + b_numel + c_numel
    detail = (
        f"  {name:12s}: A({in_features}×{r})={a_numel} + "
        f"B({out_features}×{r})={b_numel}"
    )
    if bias:
        detail += f" + C({out_features})={c_numel}"
    detail += f"  =>  {total}"
    return total, detail


def compute_layer_lora_params_numel(config, lora_ranks: dict, layer_idx: int = 0, verbose: bool = False):
    """
    Compute per-layer LoRA parameter count directly from config values.

    This is a pure-arithmetic equivalent of ``decoder_layer.lora_params_numel(lora_ranks)``
    that avoids instantiating any model — much faster than building a fake model
    on the meta device.

    Dispatches to the appropriate model-family implementation:
      * **Qwen3_5Moe** (``layer_types`` in config) — via ``LoraQwen3_5Moe``.
      * **Qwen3Moe** (``decoder_sparse_step`` in config) — via ``LoraQwen3Moe``.

    Args:
        config: Model config object.
        lora_ranks: Dict mapping component name to its LoRA rank, e.g.
            {"q_query": 16, "k_proj": 8, "expert_gate": 4, ...}.
        layer_idx: Which layer to compute for.
        verbose: If True, return (total, detail_string) instead of just total.

    Returns:
        int if verbose=False, or (int, str) if verbose=True.
    """
    # Qwen3_5Moe family (has layer_types AND num_experts)
    if hasattr(config, "layer_types") and hasattr(config, "num_experts"):
        from src_transformers_lora.LoraQwen3_5Moe import compute_qwen3_5moe_layer_lora_numel
        return compute_qwen3_5moe_layer_lora_numel(config, lora_ranks, layer_idx, verbose)

    # Qwen3_5 family (has layer_types but no num_experts — dense MLP)
    if hasattr(config, "layer_types"):
        from src_transformers_lora.LoraQwen3_5 import compute_qwen3_5_layer_lora_numel
        return compute_qwen3_5_layer_lora_numel(config, lora_ranks, layer_idx, verbose)

    # Qwen3Moe family (has decoder_sparse_step / mlp_only_layers)
    if hasattr(config, "decoder_sparse_step"):
        from src_transformers_lora.LoraQwen3Moe import compute_qwen3moe_layer_lora_numel
        return compute_qwen3moe_layer_lora_numel(config, lora_ranks, layer_idx, verbose)

    # Unknown model family — require explicit implementation
    raise NotImplementedError(
        f"compute_layer_lora_params_numel: unsupported model config type "
        f"'{type(config).__name__}'. Currently only Qwen3_5Moe (config with "
        f"'layer_types') and Qwen3Moe (config with 'decoder_sparse_step') are "
        f"supported. Please add a new branch for this model family."
    )
