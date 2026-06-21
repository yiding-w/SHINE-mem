"""
Tensor-parallel attention module for the tp_experiment branch.

Subclasses ``LoraQwen3_5Attention`` (from the LoRA fork) and replaces its
four ``LoraLinear`` projections with the matching column- / row-wise
sharded primitives from ``utils.mytp.tp_linear``:

    q_proj   -> ColwiseLoraLinear  (merged query+gate, 2*H*head_dim out)
    k_proj   -> ColwiseLoraLinear  (kvH*head_dim out)
    v_proj   -> ColwiseLoraLinear  (kvH*head_dim out)
    o_proj   -> RowwiseLoraLinear  (H*head_dim in, hidden out, one all-reduce)

The merged q_proj keeps the [head0_q | head0_g | head1_q | head1_g | ...]
per-head interleaved layout (see ``LoraQwen3_5Attention.forward``: the
output is viewed as ``[..., H, head_dim*2]`` and chunked along the last
dim). A contiguous out-dim slice therefore lands on whole-head boundaries
(``H_local = H // tp_world`` heads per rank), which is the only sharding
that lets the existing ``view`` + ``chunk`` produce the per-rank query and
gate without any reshuffling.

The two independent half-LoRAs for the merged q_proj
(``q_query_lora`` / ``q_gate_lora``) keep their ``LoraHelper`` interface,
but the per-rank delta has to be a slice of the full unsharded delta.
``_TPLoraHelper`` does that slicing on ``B`` / ``C`` then defers to the
plain ``LoraHelper.lora_delta`` with a local ``out_features``.

q_norm / k_norm operate over ``head_dim`` (not sharded) — their weights
are replicated across TP ranks unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch

from src_transformers_lora.LoraQwen3_5 import LoraQwen3_5Attention, LoraHelper
from utils.mytp.tp_linear import ColwiseLoraLinear, RowwiseLoraLinear


__all__ = ["TPLoraQwen3_5Attention", "load_attention_weights_from_full"]


class _TPLoraHelper(LoraHelper):
    """LoraHelper whose ``lora_delta`` returns the **column-wise sharded**
    output of an unsharded ``A @ B (+ C)`` LoRA.

    The caller passes the full / unsharded loradict; this helper slices
    ``B`` along the last dim and ``C`` along the only dim before delegating
    to ``LoraHelper.lora_delta`` (which uses ``self.out_features`` to do
    the final reshape — so we set that to the local out width).
    """

    def __init__(
        self,
        in_features: int,
        out_features_full: int,
        bias: bool,
        tp_rank: int,
        tp_world: int,
    ):
        if out_features_full % tp_world != 0:
            raise ValueError(
                f"_TPLoraHelper: out_features_full={out_features_full} not divisible by tp_world={tp_world}"
            )
        out_features_local = out_features_full // tp_world
        super().__init__(in_features=in_features, out_features=out_features_local, bias=bias)
        self.out_features_full = out_features_full
        self.out_features_local = out_features_local
        self.tp_rank = tp_rank
        self.tp_world = tp_world
        # Full-dim helper so hypernetwork-facing methods report full counts.
        self._full_helper = LoraHelper(in_features=in_features, out_features=out_features_full, bias=bias)

    def lora_params_numel(self, r: int) -> int:
        return self._full_helper.lora_params_numel(r)

    def set_generate_func(self, method: str) -> None:
        self._full_helper.set_generate_func(method)

    def generate_lora_dict(self, r: int, scale: float, plain_tensor: torch.Tensor):
        if r == 0:
            return None
        return self._full_helper.generate_lora_dict(r, scale, plain_tensor)

    def init_lora_dict(self, r: int, scale: float, device, dtype):
        return self._full_helper.init_lora_dict(r, scale, device, dtype)

    def lora_delta(self, input: torch.Tensor, loradict: dict) -> torch.Tensor:
        s = self.tp_rank * self.out_features_local
        e = s + self.out_features_local
        A = loradict["A"]                       # [Lb, in, r]   replicated
        B_local = loradict["B"][:, :, s:e]      # [Lb, r, out_local]
        C_full = loradict.get("C")
        C_local = C_full[:, s:e] if C_full is not None else None
        local_loradict = {"A": A, "B": B_local, "C": C_local}
        return super().lora_delta(input, local_loradict)


class TPLoraQwen3_5Attention(LoraQwen3_5Attention):
    """LoraQwen3_5Attention with TP-sharded projections.

    Inherits the original ``forward`` unchanged — it already supports any
    number of heads in the per-rank slice because the post-q_proj reshape
    uses ``view(*input_shape, -1, head_dim * 2)`` (``-1`` becomes
    ``H_local`` per rank) and every subsequent op (q/k/v norm, RoPE, SDPA,
    o_proj) sees the local head count consistently.

    Requirements:
      * ``config.num_attention_heads % tp_world == 0``
      * ``config.num_key_value_heads % tp_world == 0``

    The attention layer's collectives:
      * No collective inside q/k/v (Colwise — local output is what we want).
      * One all-reduce inside ``o_proj`` (Rowwise) before adding the bias /
        LoRA C, summing the partial outputs from each rank's heads.
    """

    def __init__(self, config, layer_idx: int, tp_rank: int, tp_world: int, tp_process_group):
        super().__init__(config, layer_idx)

        H = config.num_attention_heads
        kvH = config.num_key_value_heads
        if H % tp_world != 0:
            raise ValueError(
                f"TPLoraQwen3_5Attention: num_attention_heads={H} not divisible by tp_world={tp_world}"
            )
        # KV-head replication: if tp_world > kvH, each KV head is shared
        # by tp_world / kvH adjacent ranks. shard_world for k_proj /
        # v_proj is min(tp_world, kvH); shard_rank is rank // (tp_world / kvH).
        if tp_world > kvH:
            if tp_world % kvH != 0:
                raise ValueError(
                    f"TPLoraQwen3_5Attention: tp_world={tp_world} must be a multiple of "
                    f"num_key_value_heads={kvH} for KV-head replication"
                )
            kv_shard_world = kvH
            kv_replicas = tp_world // kvH
            kv_shard_rank = tp_rank // kv_replicas
            # num_kv_groups (per-rank) = q_heads_local / kv_heads_local = (H/tp_world) / 1
            # because each rank holds 1 KV head's worth (replicated with its pair-mates).
            kv_heads_local = 1
        else:
            if kvH % tp_world != 0:
                raise ValueError(
                    f"TPLoraQwen3_5Attention: num_key_value_heads={kvH} not divisible by tp_world={tp_world}"
                )
            kv_shard_world = tp_world
            kv_shard_rank = tp_rank
            kv_heads_local = kvH // tp_world
        # Locally each rank has H/tp_world Q heads and kv_heads_local KV heads.
        # repeat_kv expands KV by (q_local / kv_local).
        q_heads_local = H // tp_world
        self.num_key_value_groups = q_heads_local // kv_heads_local

        head_dim = self.head_dim
        hidden = config.hidden_size
        bias = config.attention_bias

        # Pull dtype / device from the parameters the parent just built so
        # the replacement linears match (we will overwrite the values via
        # load_full_weight afterwards anyway).
        ref_weight = self.q_proj.weight
        device = ref_weight.device
        dtype = ref_weight.dtype

        self.q_proj = ColwiseLoraLinear(
            in_features=hidden,
            out_features=H * head_dim * 2,
            bias=bias,
            tp_rank=tp_rank,
            tp_world=tp_world,
            tp_process_group=tp_process_group,
            device=device,
            dtype=dtype,
        )
        self.k_proj = ColwiseLoraLinear(
            in_features=hidden,
            out_features=kvH * head_dim,
            bias=bias,
            tp_rank=tp_rank,
            tp_world=tp_world,
            tp_process_group=tp_process_group,
            shard_rank=kv_shard_rank,
            shard_world=kv_shard_world,
            device=device,
            dtype=dtype,
        )
        self.v_proj = ColwiseLoraLinear(
            in_features=hidden,
            out_features=kvH * head_dim,
            bias=bias,
            tp_rank=tp_rank,
            tp_world=tp_world,
            tp_process_group=tp_process_group,
            shard_rank=kv_shard_rank,
            shard_world=kv_shard_world,
            device=device,
            dtype=dtype,
        )
        self.o_proj = RowwiseLoraLinear(
            in_features=H * head_dim,
            out_features=hidden,
            bias=bias,
            tp_rank=tp_rank,
            tp_world=tp_world,
            tp_process_group=tp_process_group,
            device=device,
            dtype=dtype,
        )

        # q_query / q_gate helpers — slice the full unsharded LoRA output
        # along the last dim to this rank's share of heads.
        self.q_query_lora = _TPLoraHelper(
            in_features=hidden,
            out_features_full=H * head_dim,
            bias=bias,
            tp_rank=tp_rank,
            tp_world=tp_world,
        )
        self.q_gate_lora = _TPLoraHelper(
            in_features=hidden,
            out_features_full=H * head_dim,
            bias=bias,
            tp_rank=tp_rank,
            tp_world=tp_world,
        )

        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.tp_process_group = tp_process_group


def load_attention_weights_from_full(
    tp_attn: TPLoraQwen3_5Attention,
    full_attn,
) -> None:
    """Copy the four projection weights/biases and q/k norm weights from a
    full (un-sharded) ``LoraQwen3_5Attention`` into a ``TPLoraQwen3_5Attention``,
    slicing each linear weight along the correct axis on the way in.

    All copies use the TP module's own dtype/device. ``full_attn`` is
    expected to live on every rank with **identical** weights (typically
    constructed under a shared seed).
    """
    def _bias_or_none(linear):
        b = getattr(linear, "bias", None)
        return b.data if b is not None else None

    tp_attn.q_proj.load_full_weight(full_attn.q_proj.weight.data, _bias_or_none(full_attn.q_proj))
    tp_attn.k_proj.load_full_weight(full_attn.k_proj.weight.data, _bias_or_none(full_attn.k_proj))
    tp_attn.v_proj.load_full_weight(full_attn.v_proj.weight.data, _bias_or_none(full_attn.v_proj))
    tp_attn.o_proj.load_full_weight(full_attn.o_proj.weight.data, _bias_or_none(full_attn.o_proj))

    # q_norm / k_norm are RMSNorm over head_dim — replicated, no sharding.
    tp_attn.q_norm.load_state_dict(full_attn.q_norm.state_dict())
    tp_attn.k_norm.load_state_dict(full_attn.k_norm.state_dict())
