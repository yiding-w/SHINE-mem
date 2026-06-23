"""
Tensor-parallel MLP module for the tp_experiment branch.

Subclasses ``LoraQwen3_5MLP`` (from the LoRA fork) and replaces its three
``LoraLinear`` projections with the matching column- / row-wise sharded
primitives from ``utils.mytp.tp_linear``:

    gate_proj -> ColwiseLoraLinear   (hidden -> intermediate)
    up_proj   -> ColwiseLoraLinear   (hidden -> intermediate)
    down_proj -> RowwiseLoraLinear   (intermediate -> hidden, one all-reduce)

No merged-linear / half-LoRA wrinkle here — the parent ``forward`` works
unchanged because every per-rank tensor sees the same shape semantics
with ``intermediate_local = intermediate // tp_world`` substituted for the
full intermediate width.

Requires ``config.intermediate_size % tp_world == 0``.
"""

from __future__ import annotations

from typing import Optional

import torch

from src_transformers_lora.LoraQwen3_5 import LoraQwen3_5MLP
from utils.mytp.tp_linear import ColwiseLoraLinear, RowwiseLoraLinear


__all__ = ["TPLoraQwen3_5MLP", "load_mlp_weights_from_full"]


class TPLoraQwen3_5MLP(LoraQwen3_5MLP):
    """LoraQwen3_5MLP with TP-sharded projections.

    The intermediate axis (``intermediate_size``) is the only thing
    sharded; ``hidden_size`` (in for gate/up, out for down) is replicated
    across the TP group. The all-reduce inside ``down_proj`` produces the
    full hidden_size output on every rank.
    """

    def __init__(
        self,
        config,
        intermediate_size: int,
        tp_rank: int,
        tp_world: int,
        tp_process_group,
    ):
        super().__init__(config, intermediate_size)

        if intermediate_size % tp_world != 0:
            raise ValueError(
                f"TPLoraQwen3_5MLP: intermediate_size={intermediate_size} not divisible by tp_world={tp_world}"
            )

        hidden = self.hidden_size
        # Reuse dtype/device from the parent's freshly initialised params.
        ref_weight = self.gate_proj.weight
        device = ref_weight.device
        dtype = ref_weight.dtype

        self.gate_proj = ColwiseLoraLinear(
            in_features=hidden,
            out_features=intermediate_size,
            bias=False,
            tp_rank=tp_rank,
            tp_world=tp_world,
            tp_process_group=tp_process_group,
            device=device,
            dtype=dtype,
        )
        self.up_proj = ColwiseLoraLinear(
            in_features=hidden,
            out_features=intermediate_size,
            bias=False,
            tp_rank=tp_rank,
            tp_world=tp_world,
            tp_process_group=tp_process_group,
            device=device,
            dtype=dtype,
        )
        self.down_proj = RowwiseLoraLinear(
            in_features=intermediate_size,
            out_features=hidden,
            bias=False,
            tp_rank=tp_rank,
            tp_world=tp_world,
            tp_process_group=tp_process_group,
            device=device,
            dtype=dtype,
        )

        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.tp_process_group = tp_process_group


def load_mlp_weights_from_full(
    tp_mlp: TPLoraQwen3_5MLP,
    full_mlp,
) -> None:
    """Copy gate/up/down weights from a full ``LoraQwen3_5MLP`` into a
    ``TPLoraQwen3_5MLP``, slicing along the right axis. Biases are absent
    (bias=False everywhere) so we pass None.
    """
    tp_mlp.gate_proj.load_full_weight(full_mlp.gate_proj.weight.data, None)
    tp_mlp.up_proj.load_full_weight(full_mlp.up_proj.weight.data, None)
    tp_mlp.down_proj.load_full_weight(full_mlp.down_proj.weight.data, None)
