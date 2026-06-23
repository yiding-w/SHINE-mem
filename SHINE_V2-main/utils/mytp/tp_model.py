"""
Tensor-parallel wrapper around ``LoraQwen3_5TextModel``.

Provides:

* ``TPLoraQwen3_5TextModel`` — subclass that replaces every decoder layer
  with a ``TPLoraQwen3_5DecoderLayer``. Embed + final norm + rotary stay
  replicated on every rank (under our TP plan the residual stream is
  always full ``hidden_size`` between layers, so anything operating on
  ``hidden_size`` is replicated).

* ``load_text_model_weights_from_full`` — copy weights from a full
  ``LoraQwen3_5TextModel`` into a ``TPLoraQwen3_5TextModel``, slicing
  each decoder layer's projection weights along the right axis.

* ``convert_text_model_to_tp_inplace`` — for production / large models:
  walk an existing full ``LoraQwen3_5TextModel`` and replace each
  ``LoraQwen3_5DecoderLayer`` with a ``TPLoraQwen3_5DecoderLayer`` in
  place, freeing the old layer's full-weight tensors as we go. Memory
  peak stays at full model + one TP layer (vs. full model + entire TP
  model with the subclass path).

The lm_head / embed sharding (item from the design doc) is deferred —
embeddings stay replicated for now. Once forward+loss are proven, we'll
revisit vocab-parallel CE to avoid the ~248k gather.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src_transformers_lora.LoraQwen3_5 import LoraQwen3_5TextModel, LoraQwen3_5DecoderLayer
from utils.mytp.tp_decoder_layer import (
    TPLoraQwen3_5DecoderLayer,
    load_decoder_layer_weights_from_full,
)


__all__ = [
    "TPLoraQwen3_5TextModel",
    "load_text_model_weights_from_full",
    "convert_text_model_to_tp_inplace",
]


class TPLoraQwen3_5TextModel(LoraQwen3_5TextModel):
    """LoraQwen3_5TextModel whose decoder layers are TP-sharded.

    Construction strategy: call the parent constructor (which allocates
    full-width decoder layers), then replace ``self.layers`` with TP
    variants. The full layers' weights are discarded immediately — fine
    for tests but wasteful for production-sized models. Use
    ``convert_text_model_to_tp_inplace`` instead when memory matters.

    Embeddings + final norm + rotary embedding are inherited from the
    parent unchanged.
    """

    def __init__(self, config, tp_rank: int, tp_world: int, tp_process_group):
        super().__init__(config)
        self.layers = nn.ModuleList([
            TPLoraQwen3_5DecoderLayer(
                config, i,
                tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
            )
            for i in range(config.num_hidden_layers)
        ])
        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.tp_process_group = tp_process_group


def load_text_model_weights_from_full(
    tp_model: TPLoraQwen3_5TextModel,
    full_model: LoraQwen3_5TextModel,
) -> None:
    """Copy weights from a full text model into a TP one.

    Decoder layers are sliced per the TP convention; everything else
    (embed_tokens, norm, rotary buffers, optional mem_tokens) is copied
    verbatim because it lives on the un-sharded ``hidden_size`` axis.
    """
    tp_model.embed_tokens.load_state_dict(full_model.embed_tokens.state_dict())
    tp_model.norm.load_state_dict(full_model.norm.state_dict())
    # rotary_emb may have only buffers; load_state_dict handles that.
    tp_model.rotary_emb.load_state_dict(full_model.rotary_emb.state_dict())

    if getattr(tp_model, "has_mem_token", False):
        with torch.no_grad():
            tp_model.mem_tokens.copy_(full_model.mem_tokens)

    assert len(tp_model.layers) == len(full_model.layers), (
        f"layer count mismatch: tp={len(tp_model.layers)} full={len(full_model.layers)}"
    )
    for tp_layer, full_layer in zip(tp_model.layers, full_model.layers):
        load_decoder_layer_weights_from_full(tp_layer, full_layer)


def convert_text_model_to_tp_inplace(
    full_model: LoraQwen3_5TextModel,
    tp_rank: int,
    tp_world: int,
    tp_process_group,
) -> LoraQwen3_5TextModel:
    """Replace each decoder layer in ``full_model`` with a TP variant in
    place, copying weights one layer at a time so the full layer's
    tensors can be freed before the next allocation.

    Returns the same model object (mutated). After this returns, the model
    is functionally a TP model; the only difference from
    ``TPLoraQwen3_5TextModel`` is the class identity of the container.

    Suitable for production use where the full model is loaded from HF
    safetensors first and TP conversion must avoid doubling memory.
    """
    config = full_model.config
    for i in range(config.num_hidden_layers):
        full_layer = full_model.layers[i]
        tp_layer = TPLoraQwen3_5DecoderLayer(
            config, i,
            tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
        ).to(device=full_layer.input_layernorm.weight.device, dtype=full_layer.input_layernorm.weight.dtype)
        load_decoder_layer_weights_from_full(tp_layer, full_layer)
        full_model.layers[i] = tp_layer
        del full_layer
    full_model.tp_rank = tp_rank
    full_model.tp_world = tp_world
    full_model.tp_process_group = tp_process_group
    return full_model
