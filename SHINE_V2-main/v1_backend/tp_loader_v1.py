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

