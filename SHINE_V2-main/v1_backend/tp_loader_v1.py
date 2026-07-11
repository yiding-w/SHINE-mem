from __future__ import annotations

import gc
import logging
import os
import sys
from typing import Optional

import torch

from utils.myparallel import is_main_process_per_node
from v1_backend.tp_linear_v1 import V1ColwiseLoraLinear, V1RowwiseLoraLinear


logger = logging.getLogger(__name__)


def _ensure_repo_root_on_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    v2_root = os.path.dirname(here)
    repo_root = os.path.dirname(v2_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def _patch_transformers_qwen3_for_v1_import() -> None:
    """Patch small transformers API moves before importing the v1 Qwen file.

    The root SHINE-v1 LoraQwen.py was generated against a slightly different
    transformers snapshot than the SHINE-v2 environment. Keep the compatibility
    shim local to this backend so the normal v2 model path is untouched.
    """
    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod

    def _identity_decorator(*args, **_kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def decorator(fn):
            return fn

        return decorator

    decorator_fallbacks = {
        "auto_docstring": _identity_decorator,
        "can_return_tuple": _identity_decorator,
        "check_model_inputs": _identity_decorator,
        "use_kernel_forward_from_hub": _identity_decorator,
    }
    for name, fallback in decorator_fallbacks.items():
        if not hasattr(qwen3_mod, name):
            setattr(qwen3_mod, name, fallback)

    if not hasattr(qwen3_mod, "deprecate_kwarg"):
        try:
            from transformers.utils.deprecation import deprecate_kwarg
        except Exception:
            try:
                from transformers.utils import deprecate_kwarg
            except Exception:
                deprecate_kwarg = _identity_decorator
        qwen3_mod.deprecate_kwarg = deprecate_kwarg

    required_names = (
        "ACT2FN", "Cache", "DynamicCache", "GenerationMixin",
        "use_kernel_forward_from_hub", "create_causal_mask",
        "create_sliding_window_causal_mask", "FlashAttentionKwargs",
        "GenericForQuestionAnswering", "GenericForSequenceClassification",
        "GenericForTokenClassification", "GradientCheckpointingLayer",
        "BaseModelOutputWithPast", "CausalLMOutputWithPast",
        "ROPE_INIT_FUNCTIONS", "dynamic_rope_update",
        "ALL_ATTENTION_FUNCTIONS", "PreTrainedModel", "Unpack",
        "TransformersKwargs", "auto_docstring", "can_return_tuple",
        "deprecate_kwarg", "check_model_inputs", "Qwen3Config",
        "Qwen3RMSNorm", "Qwen3MLP", "Qwen3Attention",
        "apply_rotary_pos_emb", "eager_attention_forward",
        "Qwen3RotaryEmbedding",
    )
    missing = [name for name in required_names if not hasattr(qwen3_mod, name)]
    if missing:
        raise ImportError(
            "Current transformers qwen3 module is missing symbols required by "
            f"SHINE-v1 LoraQwen.py after compatibility patch: {missing}"
        )


def compute_v1_num_mem_token(config, lora_r: int, mean_pool_size: int = 1) -> int:
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    head_dim = int(getattr(config, "head_dim", hidden // config.num_attention_heads))
    q_out = int(config.num_attention_heads) * head_dim
    kv_out = int(config.num_key_value_heads) * head_dim
    attn_bias = bool(getattr(config, "attention_bias", False))

    def linear_numel(in_f: int, out_f: int, bias: bool) -> int:
        return in_f * lora_r + out_f * lora_r + (out_f if bias else 0)

    per_layer = (
        linear_numel(hidden, q_out, attn_bias)
        + linear_numel(hidden, kv_out, attn_bias)
        + linear_numel(hidden, kv_out, attn_bias)
        + linear_numel(q_out, hidden, attn_bias)
        + linear_numel(hidden, intermediate, False)
        + linear_numel(hidden, intermediate, False)
        + linear_numel(intermediate, hidden, False)
    )
    if (per_layer * mean_pool_size) % hidden != 0:
        raise ValueError(
            f"V1 lora param count {per_layer} * mean_pool_size={mean_pool_size} "
            f"is not divisible by hidden_size={hidden}"
        )
    return per_layer * mean_pool_size // hidden


def _replace_linear_col(old, *, tp_rank: int, tp_world: int, tp_group) -> V1ColwiseLoraLinear:
    new = V1ColwiseLoraLinear(
        old.in_features,
        old.out_features,
        bias=old.bias is not None,
        tp_rank=tp_rank,
        tp_world=tp_world,
        tp_process_group=tp_group,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    new.load_full_weight(old.weight.data, old.bias.data if old.bias is not None else None)
    return new


def _replace_linear_row(old, *, tp_rank: int, tp_world: int, tp_group) -> V1RowwiseLoraLinear:
    new = V1RowwiseLoraLinear(
        old.in_features,
        old.out_features,
        bias=old.bias is not None,
        tp_rank=tp_rank,
        tp_world=tp_world,
        tp_process_group=tp_group,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    new.load_full_weight(old.weight.data, old.bias.data if old.bias is not None else None)
    return new


def convert_v1_llm_to_tp_inplace(model, *, tp_rank: int, tp_world: int, tp_group):
    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp

        old_q, old_k, old_v, old_o = attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
        attn.q_proj = _replace_linear_col(old_q, tp_rank=tp_rank, tp_world=tp_world, tp_group=tp_group)
        attn.k_proj = _replace_linear_col(old_k, tp_rank=tp_rank, tp_world=tp_world, tp_group=tp_group)
        attn.v_proj = _replace_linear_col(old_v, tp_rank=tp_rank, tp_world=tp_world, tp_group=tp_group)
        attn.o_proj = _replace_linear_row(old_o, tp_rank=tp_rank, tp_world=tp_world, tp_group=tp_group)
        del old_q, old_k, old_v, old_o

        old_gate, old_up, old_down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        mlp.gate_proj = _replace_linear_col(old_gate, tp_rank=tp_rank, tp_world=tp_world, tp_group=tp_group)
        mlp.up_proj = _replace_linear_col(old_up, tp_rank=tp_rank, tp_world=tp_world, tp_group=tp_group)
        mlp.down_proj = _replace_linear_row(old_down, tp_rank=tp_rank, tp_world=tp_world, tp_group=tp_group)
        del old_gate, old_up, old_down

    model.tp_rank = tp_rank
    model.tp_world = tp_world
    model.tp_process_group = tp_group
    model.model.tp_rank = tp_rank
    model.model.tp_world = tp_world
    model.model.tp_process_group = tp_group
    gc.collect()
    return model


def load_v1_qwen3_for_tp(
    model_path: str,
    *,
    lora_r: int,
    mean_pool_size: int,
    tp_rank: int,
    tp_world: int,
    tp_group,
    dtype: torch.dtype = torch.bfloat16,
    freeze: bool = True,
    device: Optional[torch.device] = None,
):
    _ensure_repo_root_on_path()
    _patch_transformers_qwen3_for_v1_import()
    from LoraQwen import LoraQwen3ForCausalLM, Qwen3Config

    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    config = Qwen3Config.from_pretrained(model_path)
    config.num_mem_token = compute_v1_num_mem_token(
        config,
        lora_r=lora_r,
        mean_pool_size=mean_pool_size,
    )
    if is_main_process_per_node():
        logger.info(
            f"[v1_backend] num_mem_token={config.num_mem_token} "
            f"(lora_r={lora_r}, mean_pool_size={mean_pool_size})"
        )

    llm = LoraQwen3ForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        device_map=device,
    )
    llm.reset_mem_tokens()
    llm.config.use_cache = False
    llm = convert_v1_llm_to_tp_inplace(
        llm,
        tp_rank=tp_rank,
        tp_world=tp_world,
        tp_group=tp_group,
    )
    if freeze:
        for p in llm.parameters():
            p.requires_grad_(False)
    return llm
