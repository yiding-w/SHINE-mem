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

        # SP (Sequence Parallelism) support
        self.sp_group = None
        self.sp_world = 1
        self._use_sp = False
        self._sp_attn_fn = None

    def enable_sp(self, sp_group, sp_world: int, sp_mode: str = "alltoall_zigzag"):
        """Enable sequence parallelism on this attention layer.

        Called after __init__ (during model loading) to avoid import
        issues and keep __init__ signature compatible.
        """
        from utils.mytp.ring_attention import get_sp_attention_fn
        self.sp_group = sp_group
        self.sp_world = sp_world
        self._use_sp = (sp_group is not None and sp_world > 1)
        if self._use_sp:
            self._sp_attn_fn = get_sp_attention_fn(sp_mode)

    @torch.compiler.disable
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        loradict=None,
        nograd_loradict=None,
        nograd_wdict=None,
        **kwargs,
    ):
        """Forward with optional SP ring attention.

        When SP is disabled (_use_sp=False), delegates to parent forward
        (identical to before). When SP is enabled, replaces the attention
        computation with ring flash attention.
        """
        if not self._use_sp:
            return super().forward(
                hidden_states, position_embeddings, attention_mask,
                past_key_values=past_key_values,
                loradict=loradict, nograd_loradict=nograd_loradict,
                nograd_wdict=nograd_wdict, **kwargs,
            )

        # --- SP-aware forward (ring attention) ---
        from src_transformers_lora.LoraQwen3_5 import apply_rotary_pos_emb

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if loradict is None and nograd_loradict is None and nograd_wdict is None:
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)
            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        else:
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            )
            if loradict is not None and loradict.get("q_query") is not None:
                query_states = query_states + self.q_query_lora.lora_delta(hidden_states, loradict["q_query"]).view(*input_shape, -1, self.head_dim)
            if loradict is not None and loradict.get("q_gate") is not None:
                gate = gate + self.q_gate_lora.lora_delta(hidden_states, loradict["q_gate"]).view(*input_shape, -1, self.head_dim)
            if nograd_loradict is not None and nograd_loradict.get("q_query") is not None:
                query_states = query_states + self.q_query_lora.lora_delta(hidden_states, nograd_loradict["q_query"]).view(*input_shape, -1, self.head_dim)
            if nograd_loradict is not None and nograd_loradict.get("q_gate") is not None:
                gate = gate + self.q_gate_lora.lora_delta(hidden_states, nograd_loradict["q_gate"]).view(*input_shape, -1, self.head_dim)
            if nograd_wdict is not None and nograd_wdict.get("q_query") is not None:
                query_states = query_states + self._compute_helper_w_delta(hidden_states, nograd_wdict["q_query"], self.q_query_lora).view(*input_shape, -1, self.head_dim)
            if nograd_wdict is not None and nograd_wdict.get("q_gate") is not None:
                gate = gate + self._compute_helper_w_delta(hidden_states, nograd_wdict["q_gate"], self.q_gate_lora).view(*input_shape, -1, self.head_dim)
            gate = gate.reshape(*input_shape, -1)

            k_lora = loradict.get("k") if loradict else None
            k_nograd = nograd_loradict.get("k") if nograd_loradict else None
            k_w = nograd_wdict.get("k") if nograd_wdict else None
            v_lora = loradict.get("v") if loradict else None
            v_nograd = nograd_loradict.get("v") if nograd_loradict else None
            v_w = nograd_wdict.get("v") if nograd_wdict else None

            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states, k_lora, k_nograd, k_w).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states, v_lora, v_nograd, v_w).view(hidden_shape).transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Ring attention (SP): expects [B, S, H, D] layout
        # query_states/key_states/value_states are [B, H, S, D] after transpose(1,2)
        q = query_states.transpose(1, 2)  # [B, S_local, Hq_local, D]
        k = key_states.transpose(1, 2)    # [B, S_local, Hkv_local, D]
        v = value_states.transpose(1, 2)  # [B, S_local, Hkv_local, D]

        attn_output = self._sp_attn_fn(q, k, v, sp_group=self.sp_group, causal=True)
        # attn_output: [B, S_local, Hq_local, D]

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        if loradict is None and nograd_loradict is None and nograd_wdict is None:
            attn_output = self.o_proj(attn_output)
        else:
            o_lora = loradict.get("o") if loradict else None
            o_nograd = nograd_loradict.get("o") if nograd_loradict else None
            o_w = nograd_wdict.get("o") if nograd_wdict else None
            attn_output = self.o_proj(attn_output, o_lora, o_nograd, o_w)
        return attn_output, None


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
