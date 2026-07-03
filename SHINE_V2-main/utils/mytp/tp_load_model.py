"""
Tensor-parallel LLM loader.

Loads a pretrained ``LoraQwen3_5ForCausalLM`` from disk and converts
every ``full_attention`` decoder layer into a ``TPLoraQwen3_5DecoderLayer``
in place, freeing the original layer's full-width tensors as we go.
``linear_attention`` layers (``Qwen3_5GatedDeltaNet`` + ``Qwen3_5MLP``)
stay replicated on every TP rank per the design doc.

Strategy — minimise peak memory:

  1. Load the full model on **CPU** with ``device_map={'': 'cpu'}``.
     Host RAM holds one full copy (~54 GB for Qwen3.6-27B in bf16).
     **All ranks read from the same safetensors files** — fine over an
     NFS or a local cached copy because the read is one-shot.
  2. For each full_attention layer ``i``:
       * Build a TPLoraQwen3_5DecoderLayer on the local GPU
         (~1/W of one layer's params live on each rank).
       * Use load_decoder_layer_weights_from_full to slice each linear's
         weight from the CPU tensor into the GPU TP layer.
       * Replace ``model.model.layers[i]`` with the TP layer; drop the
         reference to the original CPU layer so Python can GC its
         tensors.
  3. For each linear_attention layer ``i``:
       * Move the whole layer to the local GPU (replicated). Smaller
         than full_attention but still non-trivial (~0.3 B params
         in the production model).
  4. Move embed_tokens + final norm + lm_head + rotary_emb to GPU
     (replicated on every rank). lm_head is *not* TP-sharded for now —
     vocab-parallel CE is a future optimisation.
  5. Free the CPU copy completely.

After this returns the model is functionally identical to a
``TPLoraQwen3_5ForCausalLM`` but keeps the original ``LoraQwen3_5ForCausalLM``
class so the existing forward / generate / lm_head wiring just works.
"""
from __future__ import annotations

import gc
import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from utils.myparallel import is_main_process_per_node
from utils.mytp.tp_decoder_layer import TPLoraQwen3_5DecoderLayer, load_decoder_layer_weights_from_full

logger = logging.getLogger(__name__)


__all__ = ["load_pretrained_llm_for_tp"]


def _import_class(dotted_path: str):
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"_import_class: expected 'module.Class', got '{dotted_path}'")
    mod = __import__(parts[0], fromlist=[parts[1]])
    return getattr(mod, parts[1])


def load_pretrained_llm_for_tp(
    model_cfg,
    tp_rank: int,
    tp_world: int,
    tp_process_group,
    dtype: torch.dtype = torch.bfloat16,
    freeze: bool = True,
    num_mem_token: Optional[int] = None,
    sp_group=None,
    sp_world: int = 1,
) -> nn.Module:
    """Load a pretrained LLM in TP form on the current rank.

    Args:
        model_cfg: Hydra config with ``path`` and ``lora_class``.
        tp_rank: TP rank within the group (0..tp_world-1).
        tp_world: TP group size.
        tp_process_group: torch ProcessGroup for the TP collectives.
        dtype: LLM weight dtype (production: torch.bfloat16).
        freeze: Set requires_grad=False on every LLM param.
        num_mem_token: If provided, set ``text_config.num_mem_token`` before
            loading so the model allocates ``mem_tokens`` of the right size.
        sp_group: Sequence parallel process group (None if sp_world=1).
        sp_world: Sequence parallel group size (default 1 = no SP).

    Returns:
        The loaded model with full_attention decoder layers replaced by
        TP variants and everything placed on the current rank's GPU.
    """
    from hydra.utils import get_original_cwd
    from transformers import AutoConfig

    model_path = str(model_cfg.path)
    if not os.path.isabs(model_path):
        model_path = os.path.join(get_original_cwd(), model_path)

    class_path = getattr(model_cfg, "lora_class", None) or getattr(model_cfg, "base_class", None)
    if class_path is None:
        raise ValueError("model_cfg must specify lora_class or base_class")
    model_class = _import_class(str(class_path))

    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # ----------------------------------------------------------------
    # 0. Pre-load config so we can inject num_mem_token before the LLM
    #    builds its mem_tokens nn.Parameter at the requested width.
    #    LoraQwen3_5ForCausalLM expects a Qwen3_5TextConfig (the text
    #    sub-config of the multi-modal Qwen3_5Config saved on disk).
    # ----------------------------------------------------------------
    base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(base_config, "text_config") and base_config.text_config is not None:
        text_config = base_config.text_config
    elif hasattr(base_config, "get_text_config"):
        text_config = base_config.get_text_config()
    else:
        text_config = base_config
    if num_mem_token is not None and num_mem_token > 0:
        text_config.num_mem_token = int(num_mem_token)
        if is_main_process_per_node():
            logger.info(f"[tp_load_model] Set text_config.num_mem_token = {num_mem_token}")

    # Attention backend. Default to SDPA (same as PP) — when
    # attention_mask=None + is_causal=True, PyTorch SDPA automatically
    # selects the Flash Attention kernel. This matches PP's code path
    # exactly, eliminating numerical differences from different attention
    # implementations. Override with TP_ATTN_IMPL env var if needed.
    attn_impl = os.environ.get("TP_ATTN_IMPL", "sdpa").strip()
    text_config._attn_implementation = attn_impl
    text_config._attn_implementation_internal = attn_impl

    # Liger-Kernel: swap Qwen3_5RMSNorm for a fused Triton equivalent
    # (Gemma casting + offset=1.0 + zero init → bit-equivalent at bf16
    # noise, verified vs reference). 64×2+1 = 129 RMSNorms/step become
    # single-kernel calls. Set SHINE_LIGER=0 to disable for A/B.
    if os.environ.get("SHINE_LIGER", "1") not in ("0", "", "false"):
        from utils.liger_patch import apply_liger_rmsnorm_patch
        apply_liger_rmsnorm_patch()

    if is_main_process_per_node():
        logger.info(f"[tp_load_model] _attn_implementation = {attn_impl}")

    # ----------------------------------------------------------------
    # 1. Load full model on CPU
    # ----------------------------------------------------------------
    if is_main_process_per_node():
        logger.info(f"[tp_load_model] Loading full model from '{model_path}' on CPU…")

    # Each rank loads the FULL model directly onto its local GPU. Peak
    # memory per rank is ~54 GB for Qwen3.6-27B in bf16; after TP
    # conversion drops to ~14 GB (TP-sharded full_attn) + ~44 GB
    # (replicated linear_attn) ≈ 58 GB. Loading on CPU then moving
    # per-layer was attempted first but produced NaNs at forward time —
    # buffers / params end up in an inconsistent state that's hard to
    # reproduce in isolation. Direct-to-GPU mirrors the vanilla load
    # that's known to work.
    cpu_model = model_class.from_pretrained(
        model_path,
        config=text_config,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )

    if is_main_process_per_node():
        logger.info(
            f"[tp_load_model] CPU load done — "
            f"{sum(p.numel() for p in cpu_model.parameters()) / 1e9:.2f} B params"
        )

    # ----------------------------------------------------------------
    # 2. Per-layer TP conversion
    # ----------------------------------------------------------------
    config = cpu_model.config
    text_model = cpu_model.model  # LoraQwen3_5TextModel
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config

    num_layers = len(text_model.layers)
    if is_main_process_per_node():
        logger.info(f"[tp_load_model] Converting {num_layers} decoder layers…")

    for i in range(num_layers):
        full_layer = text_model.layers[i]
        tp_layer = TPLoraQwen3_5DecoderLayer(
            text_config, i,
            tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
            sp_group=sp_group, sp_world=sp_world,
        ).to(device=device, dtype=dtype)
        load_decoder_layer_weights_from_full(tp_layer, full_layer)
        text_model.layers[i] = tp_layer
        del full_layer
        if (i + 1) % 16 == 0 and is_main_process_per_node():
            logger.info(f"[tp_load_model]   converted {i + 1}/{num_layers} layers")

    # ----------------------------------------------------------------
    # 3. Replicated extras: embed_tokens, norm, rotary, mem_tokens, lm_head
    # ----------------------------------------------------------------
    # embed_tokens / norm / rotary_emb / mem_tokens / lm_head are already on
    # the local GPU from the device_map=device load — nothing further to do.

    # ----------------------------------------------------------------
    # 3.5 Enable SP on full_attention layers (ring attention)
    # ----------------------------------------------------------------
    if sp_group is not None and sp_world > 1:
        sp_mode = os.environ.get("SP_ATTN_MODE", "alltoall_zigzag")
        for i in range(num_layers):
            layer = text_model.layers[i]
            if layer.layer_type == "full_attention":
                layer.self_attn.enable_sp(sp_group, sp_world, sp_mode=sp_mode)
        if is_main_process_per_node():
            logger.info(
                f"[tp_load_model] SP enabled on full_attention layers: "
                f"sp_world={sp_world}, mode={sp_mode}"
            )

    # ----------------------------------------------------------------
    # 4. Force GC of any leftover CPU tensors
    # ----------------------------------------------------------------
    gc.collect()

    # ----------------------------------------------------------------
    # 5. Freeze
    # ----------------------------------------------------------------
    if freeze:
        for p in cpu_model.parameters():
            p.requires_grad_(False)

    # ----------------------------------------------------------------
    # 6. Stamp TP metadata on the model object for downstream consumers
    # ----------------------------------------------------------------
    cpu_model.tp_rank = tp_rank
    cpu_model.tp_world = tp_world
    cpu_model.tp_process_group = tp_process_group
    cpu_model.sp_group = sp_group
    cpu_model.sp_world = sp_world
    text_model.tp_rank = tp_rank
    text_model.tp_world = tp_world
    text_model.tp_process_group = tp_process_group
    text_model.sp_group = sp_group
    text_model.sp_world = sp_world

    if is_main_process_per_node():
        free, total = torch.cuda.mem_get_info(device)
        logger.info(
            f"[tp_load_model] TP load complete on rank {tp_rank}/{tp_world}: "
            f"GPU mem free={free / 1e9:.1f} GB / total={total / 1e9:.1f} GB"
        )

    return cpu_model
