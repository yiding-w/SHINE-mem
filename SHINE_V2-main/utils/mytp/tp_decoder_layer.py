"""
Tensor-parallel decoder layer for the tp_experiment branch.

Subclasses ``LoraQwen3_5DecoderLayer`` and replaces the per-rank
sub-modules:

  * ``full_attention`` layer: ``self_attn`` → ``TPLoraQwen3_5Attention``,
    ``mlp`` → ``TPLoraQwen3_5MLP``.
  * ``linear_attention`` layer: kept **replicated** on every TP rank
    (``Qwen3_5GatedDeltaNet`` + ``Qwen3_5MLP``). The fused in_proj +
    recurrence does not decompose cleanly along feature dim; the design
    doc accepts the 8× memory hit on these layers per TP rank.

RMSNorms (``input_layernorm`` / ``post_attention_layernorm``) are over
``hidden_size`` — that axis is **never sharded** under our TP plan (the
RowwiseLoraLinear all-reduce produces full hidden_size on every rank), so
the norms are replicated unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src_transformers_lora.LoraQwen3_5 import LoraQwen3_5DecoderLayer
from utils.mytp.tp_attention import TPLoraQwen3_5Attention, load_attention_weights_from_full
from utils.mytp.tp_mlp import TPLoraQwen3_5MLP, load_mlp_weights_from_full
from utils.mytp.tp_gated_deltanet import (
    TPQwen3_5GatedDeltaNet,
    load_gated_deltanet_weights_from_full,
)
from utils.mytp.tp_linear import ColwiseLoraLinear, RowwiseLoraLinear


__all__ = ["TPLoraQwen3_5DecoderLayer", "load_decoder_layer_weights_from_full"]


def _replace_mlp_with_tp(mlp: nn.Module, tp_rank: int, tp_world: int, tp_process_group) -> nn.Module:
    """Swap an existing ``Qwen3_5MLP`` (used inside linear_attention
    decoder layers) for TP-sharded gate/up (Colwise) + down (Rowwise).
    The MLP has no LoRA on linear_attention layers, so we don't need the
    LoraQwen3_5MLP wrapping — TP linears with ``loradict=None`` are
    equivalent to plain Linear forwards.
    """
    hidden = mlp.gate_proj.in_features
    intermediate = mlp.gate_proj.out_features
    device = mlp.gate_proj.weight.device
    dtype = mlp.gate_proj.weight.dtype

    mlp.gate_proj = ColwiseLoraLinear(
        in_features=hidden, out_features=intermediate, bias=False,
        tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
        device=device, dtype=dtype,
    )
    mlp.up_proj = ColwiseLoraLinear(
        in_features=hidden, out_features=intermediate, bias=False,
        tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
        device=device, dtype=dtype,
    )
    mlp.down_proj = RowwiseLoraLinear(
        in_features=intermediate, out_features=hidden, bias=False,
        tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
        device=device, dtype=dtype,
    )
    return mlp


def _load_mlp_weights_into_tp(tp_mlp: nn.Module, full_mlp: nn.Module) -> None:
    tp_mlp.gate_proj.load_full_weight(full_mlp.gate_proj.weight.data, None)
    tp_mlp.up_proj.load_full_weight(full_mlp.up_proj.weight.data, None)
    tp_mlp.down_proj.load_full_weight(full_mlp.down_proj.weight.data, None)


class TPLoraQwen3_5DecoderLayer(LoraQwen3_5DecoderLayer):
    def __init__(self, config, layer_idx: int, tp_rank: int, tp_world: int, tp_process_group):
        super().__init__(config, layer_idx)

        if self.layer_type == "full_attention":
            self.self_attn = TPLoraQwen3_5Attention(
                config, layer_idx,
                tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
            )
            self.mlp = TPLoraQwen3_5MLP(
                config, config.intermediate_size,
                tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
            )
        else:
            # linear_attention layer: TP-shard the GatedDeltaNet AND the
            # MLP. Before this change those two together accounted for
            # ~30 GB / 45 GB of per-rank memory; sharding them brings it
            # down to ~7 GB and frees the activation budget for mb >= 2.
            self.linear_attn = TPQwen3_5GatedDeltaNet(
                config, layer_idx,
                tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
            )
            self.mlp = _replace_mlp_with_tp(
                self.mlp, tp_rank, tp_world, tp_process_group,
            )

        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.tp_process_group = tp_process_group


def load_decoder_layer_weights_from_full(
    tp_layer: TPLoraQwen3_5DecoderLayer,
    full_layer,
) -> None:
    """Copy weights from a full ``LoraQwen3_5DecoderLayer`` into a TP one.

    full_attention layers slice attention + MLP weights along their TP
    axes; the two RMSNorms and (for linear_attention) the entire
    sub-module are replicated.
    """
    if tp_layer.layer_type == "full_attention":
        load_attention_weights_from_full(tp_layer.self_attn, full_layer.self_attn)
        load_mlp_weights_from_full(tp_layer.mlp, full_layer.mlp)
    else:
        # linear_attention: TP-shard both the GatedDeltaNet and the MLP.
        load_gated_deltanet_weights_from_full(tp_layer.linear_attn, full_layer.linear_attn)
        _load_mlp_weights_into_tp(tp_layer.mlp, full_layer.mlp)

    tp_layer.input_layernorm.load_state_dict(full_layer.input_layernorm.state_dict())
    tp_layer.post_attention_layernorm.load_state_dict(full_layer.post_attention_layernorm.state_dict())
