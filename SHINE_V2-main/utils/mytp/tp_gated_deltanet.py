"""
Tensor-parallel ``Qwen3_5GatedDeltaNet`` (the linear_attention layer).

This was the largest remaining memory drag on the TP path: with TP=4 the
16 full_attention layers shard to ~3 GB each but the 48 linear_attention
layers (~660 M params each) stayed fully replicated, eating ~30 GB
per rank by themselves. Sharding them along the head dim drops that to
~7 GB and frees up the activation budget for ``mb >= 2``.

What gets sharded
-----------------
The model has::

    num_v_heads = 32          # value heads
    num_k_heads = 16          # key heads (GQA factor = 2)
    head_v_dim = 128
    head_k_dim = 128
    key_dim   = num_k_heads * head_k_dim = 2048
    value_dim = num_v_heads * head_v_dim = 4096
    conv_dim  = 2*key_dim + value_dim   = 8192

With TP=W, each rank holds ``num_v_heads // W`` value heads and
``num_k_heads // W`` key heads. The sharded modules:

  * ``in_proj_qkv``  — Colwise on its output, **but the output layout
    is [query|key|value] concatenated**, so a naive contiguous slice
    would give rank 0 all of query and none of key/value. We use a
    per-head interleaved shard so each rank ends up with its local
    slice of all three. See ``_shard_merged_qkv_weight``.
  * ``conv1d``       — depthwise (groups=conv_dim) over the same
    [query|key|value] channel layout. Same per-head interleaved
    shard applies to the weight.
  * ``in_proj_z``    — Colwise on out (value_dim → value_dim/W).
  * ``in_proj_b`` / ``in_proj_a`` — Colwise on out (num_v → num_v/W).
  * ``dt_bias`` / ``A_log`` — sliced along the only dim (num_v_heads).
  * ``norm`` (RMSNormGated over head_v_dim) — replicated; the norm
    weight has shape ``[head_v_dim]`` independent of head count.
  * ``out_proj``     — Rowwise (value_dim/W → hidden, one all-reduce).

The forward is inherited from ``Qwen3_5GatedDeltaNet`` and references
``self.key_dim``, ``self.value_dim``, ``self.num_v_heads``,
``self.num_k_heads`` — we mutate these to local values so the parent's
``torch.split`` and ``view`` operations produce the right local shapes
without any forward override.

Gradient correctness
--------------------
The LLM is frozen so dL/d(linear_attn params) is never needed. The
gradient flow that matters is for the residual stream and the LoRA
tensors used by the full_attention layers:

  * ``in_proj_*`` (Colwise) inputs are the layer's hidden_states (full,
    replicated). The Megatron ``copy_to_tp_region`` autograd Function
    inside ``ColwiseLoraLinear`` already all-reduce-SUMs dL/dinput in
    backward, so the gradient flowing upstream is full per rank.
  * ``out_proj`` (Rowwise) forward all-reduce + identity backward
    correctly propagates dL/dout_partial = dL/dout on every rank.

Net: replacing the linear_attn layers in place leaves the gradient
flow correct without any extra plumbing.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from fla.modules.conv.causal_conv1d import causal_conv1d as fla_causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from utils.myparallel import is_main_process_per_node
from utils.mytp.fla_cp import build_sp_cp_context
from utils.mytp.tp_linear import ColwiseLoraLinear, RowwiseLoraLinear


logger = logging.getLogger(__name__)


__all__ = ["TPQwen3_5GatedDeltaNet", "load_gated_deltanet_weights_from_full"]


# ---------------------------------------------------------------------------
# Custom sharding for the merged [query | key | value] layout
# ---------------------------------------------------------------------------


def _shard_merged_qkv(full: Tensor, key_dim: int, value_dim: int,
                      tp_rank: int, tp_world: int) -> Tensor:
    """Slice a tensor whose first dim is laid out as
    ``[query (key_dim) | key (key_dim) | value (value_dim)]`` so the
    returned local tensor is
    ``[query_local | key_local | value_local]`` with the same layout
    but each chunk holding 1/tp_world of its full size.
    """
    if key_dim % tp_world != 0:
        raise ValueError(f"key_dim={key_dim} not divisible by tp_world={tp_world}")
    if value_dim % tp_world != 0:
        raise ValueError(f"value_dim={value_dim} not divisible by tp_world={tp_world}")
    qd = key_dim // tp_world
    vd = value_dim // tp_world

    q_part = full[tp_rank * qd : (tp_rank + 1) * qd]
    k_part = full[key_dim + tp_rank * qd : key_dim + (tp_rank + 1) * qd]
    v_part = full[2 * key_dim + tp_rank * vd : 2 * key_dim + (tp_rank + 1) * vd]
    return torch.cat([q_part, k_part, v_part], dim=0).contiguous()


class _MergedQKVColwiseLinear(ColwiseLoraLinear):
    """ColwiseLoraLinear with a custom ``load_full_weight`` that respects
    the merged ``[query|key|value]`` output layout of ``in_proj_qkv``."""

    def __init__(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        tp_rank: int,
        tp_world: int,
        tp_process_group,
        device=None,
        dtype=None,
    ):
        out_features = 2 * key_dim + value_dim
        super().__init__(
            in_features=hidden_size,
            out_features=out_features,
            bias=False,
            tp_rank=tp_rank,
            tp_world=tp_world,
            tp_process_group=tp_process_group,
            device=device,
            dtype=dtype,
        )
        self._merged_key_dim = key_dim
        self._merged_value_dim = value_dim

    def load_full_weight(self, full_weight: Tensor, full_bias: Optional[Tensor]):
        local_w = _shard_merged_qkv(
            full_weight, self._merged_key_dim, self._merged_value_dim,
            self.tp_rank, self.tp_world,
        )
        with torch.no_grad():
            self.weight.copy_(local_w.to(self.weight.dtype).to(self.weight.device))
        if full_bias is not None:
            raise ValueError("in_proj_qkv is expected to be bias-free")


# ---------------------------------------------------------------------------
# TP GatedDeltaNet
# ---------------------------------------------------------------------------


class TPQwen3_5GatedDeltaNet(nn.Module):
    """TP-sharded replacement for ``Qwen3_5GatedDeltaNet``.

    Inherits the parent class's *forward* by composition (we construct a
    parent instance, replace its sub-modules with TP variants, mutate
    ``self.key_dim`` / ``self.value_dim`` / ``self.num_*_heads`` to local
    values, and then expose its bound forward as our own).
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        tp_rank: int,
        tp_world: int,
        tp_process_group,
        sp_group: Optional[ProcessGroup] = None,
        sp_world: int = 1,
    ):
        # We don't actually want our own nn.Module — we want the parent's
        # forward bound to a parent instance whose internals we replace.
        # Defer to a parent instance held as ``self._inner``.
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
        inner = Qwen3_5GatedDeltaNet(config, layer_idx)

        num_v = inner.num_v_heads
        num_k = inner.num_k_heads
        if num_v % tp_world != 0:
            raise ValueError(
                f"TPQwen3_5GatedDeltaNet: linear_num_value_heads={num_v} not divisible by tp_world={tp_world}"
            )
        if num_k % tp_world != 0:
            raise ValueError(
                f"TPQwen3_5GatedDeltaNet: linear_num_key_heads={num_k} not divisible by tp_world={tp_world}"
            )

        hidden = inner.hidden_size
        head_k_dim = inner.head_k_dim
        head_v_dim = inner.head_v_dim
        key_dim_full = inner.key_dim
        value_dim_full = inner.value_dim
        conv_kernel = inner.conv_kernel_size

        num_v_local = num_v // tp_world
        num_k_local = num_k // tp_world
        key_dim_local = num_k_local * head_k_dim
        value_dim_local = num_v_local * head_v_dim
        conv_dim_local = 2 * key_dim_local + value_dim_local

        device = inner.in_proj_qkv.weight.device  # CPU / GPU as constructed
        dtype = inner.in_proj_qkv.weight.dtype

        # ---- Replace projections ----
        inner.in_proj_qkv = _MergedQKVColwiseLinear(
            hidden_size=hidden, key_dim=key_dim_full, value_dim=value_dim_full,
            tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
            device=device, dtype=dtype,
        )
        inner.in_proj_z = ColwiseLoraLinear(
            in_features=hidden, out_features=value_dim_full, bias=False,
            tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
            device=device, dtype=dtype,
        )
        inner.in_proj_b = ColwiseLoraLinear(
            in_features=hidden, out_features=num_v, bias=False,
            tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
            device=device, dtype=dtype,
        )
        inner.in_proj_a = ColwiseLoraLinear(
            in_features=hidden, out_features=num_v, bias=False,
            tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
            device=device, dtype=dtype,
        )
        inner.out_proj = RowwiseLoraLinear(
            in_features=value_dim_full, out_features=hidden, bias=False,
            tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
            device=device, dtype=dtype,
        )

        # ---- Replace conv1d with a local one ----
        local_conv1d = nn.Conv1d(
            in_channels=conv_dim_local,
            out_channels=conv_dim_local,
            bias=False,
            kernel_size=conv_kernel,
            groups=conv_dim_local,
            padding=conv_kernel - 1,
        ).to(device=device, dtype=dtype)
        inner.conv1d = local_conv1d

        # ---- Replace dt_bias and A_log with local-sized parameters ----
        inner.dt_bias = nn.Parameter(
            torch.ones(num_v_local, device=device, dtype=inner.dt_bias.dtype)
        )
        A = torch.empty(num_v_local, dtype=inner.A_log.dtype, device=device).uniform_(0, 16)
        inner.A_log = nn.Parameter(torch.log(A))

        # ---- Norm: replicate (operates on head_v_dim independent of head count) ----
        # Already constructed correctly by parent __init__ with the right
        # head_v_dim. Leave as-is.

        # ---- Mutate head counts so the parent's forward uses local dims ----
        inner.num_v_heads = num_v_local
        inner.num_k_heads = num_k_local
        inner.key_dim = key_dim_local
        inner.value_dim = value_dim_local
        inner.conv_dim = conv_dim_local

        # ---- Store metadata ----
        self._inner = inner
        self._key_dim_full = key_dim_full
        self._value_dim_full = value_dim_full
        self._num_v_heads_full = num_v
        self._num_k_heads_full = num_k
        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.tp_process_group = tp_process_group
        self.layer_idx = layer_idx

        # ---- SP (Sequence Parallelism) support ----
        self.sp_group = sp_group
        self.sp_world = sp_world
        self._use_sp = (sp_group is not None and sp_world > 1)

        # Cache local dims needed by forward_sp
        self._num_v_local = num_v_local
        self._num_k_local = num_k_local
        self._head_k_dim = head_k_dim
        self._head_v_dim = head_v_dim
        self._key_dim_local = key_dim_local
        self._value_dim_local = value_dim_local
        self._conv_kernel_size = conv_kernel

        # Pre-extract conv1d weight in [D, W] format for FLA's causal_conv1d
        # (will be populated after weight loading via _prepare_sp_weights)
        self._conv1d_weight_for_fla: Optional[Tensor] = None

        # cp_context is lazily built on first forward_sp call (needs device)
        self._cp_context = None
        self._cp_context_seq_len = None

        if is_main_process_per_node():
            sp_info = f" SP={sp_world}" if self._use_sp else ""
            logger.info(
                f"[TPQwen3_5GatedDeltaNet] layer {layer_idx}:{sp_info} "
                f"num_v={num_v}→{num_v_local} num_k={num_k}→{num_k_local} "
                f"key_dim={key_dim_full}→{key_dim_local} value_dim={value_dim_full}→{value_dim_local} "
                f"conv_dim={inner.conv_dim} (was {2*key_dim_full + value_dim_full})"
            )

    # ------------------------------------------------------------------
    # SP forward path
    # ------------------------------------------------------------------

    def _get_cp_context(self, seq_len_local: int, device: torch.device):
        """Lazily build and cache the FLACPContext."""
        if self._cp_context is None or self._cp_context_seq_len != seq_len_local:
            self._cp_context = build_sp_cp_context(
                seq_len_local=seq_len_local,
                sp_group=self.sp_group,
                conv1d_kernel_size=self._conv_kernel_size,
                device=device,
            )
            self._cp_context_seq_len = seq_len_local
        return self._cp_context

    def _get_conv1d_weight(self) -> Tensor:
        """Get conv1d weight in [D, W] format for FLA's causal_conv1d."""
        if self._conv1d_weight_for_fla is None:
            # conv1d.weight shape: [conv_dim_local, 1, kernel_size]
            # FLA expects: [D, W] where D=conv_dim_local, W=kernel_size
            self._conv1d_weight_for_fla = self._inner.conv1d.weight.squeeze(1)
        return self._conv1d_weight_for_fla

    @torch.compiler.disable
    def forward_sp(self, hidden_states: Tensor) -> Tensor:
        """SP-aware forward using FLA's native CP support.

        Uses fla.causal_conv1d (with halo exchange) and
        chunk_gated_delta_rule (with state merge) — both handle
        cross-rank communication internally via cp_context.

        Args:
            hidden_states: [B, S_local, H]

        Returns:
            output: [B, S_local, H]
        """
        batch_size, seq_len, _ = hidden_states.shape
        inner = self._inner

        # Build/retrieve cp_context
        cp_context = self._get_cp_context(seq_len, hidden_states.device)

        # 1. Projections (local, no communication — TP all-reduce in backward
        #    is handled by ColwiseLoraLinear / RowwiseLoraLinear)
        mixed_qkv = inner.in_proj_qkv(hidden_states)  # [B, S_local, conv_dim_local]
        z = inner.in_proj_z(hidden_states)              # [B, S_local, value_dim_local]
        b = inner.in_proj_b(hidden_states)              # [B, S_local, num_v_local]
        a = inner.in_proj_a(hidden_states)              # [B, S_local, num_v_local]

        # 2. Conv1d with CP halo exchange (FLA handles internally)
        #    FLA's causal_conv1d expects x: [B, T, D], weight: [D, W]
        conv_weight = self._get_conv1d_weight()
        mixed_qkv, _ = fla_causal_conv1d(
            x=mixed_qkv,
            weight=conv_weight,
            bias=inner.conv1d.bias,
            activation=inner.activation,
            cp_context=cp_context,
        )

        # 3. Split Q/K/V, compute g, beta (all local)
        query, key, value = torch.split(
            mixed_qkv,
            [self._key_dim_local, self._key_dim_local, self._value_dim_local],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, self._num_k_local, self._head_k_dim)
        key = key.reshape(batch_size, seq_len, self._num_k_local, self._head_k_dim)
        value = value.reshape(batch_size, seq_len, self._num_v_local, self._head_v_dim)

        beta = b.sigmoid()
        # float() to avoid -inf in fp16 for A_log.exp()
        g = -inner.A_log.float().exp() * F.softplus(a.float() + inner.dt_bias)

        # GQA: repeat k heads to match v heads
        gqa_factor = self._num_v_local // self._num_k_local
        if gqa_factor > 1:
            query = query.repeat_interleave(gqa_factor, dim=2)
            key = key.repeat_interleave(gqa_factor, dim=2)

        # 4. chunk_gated_delta_rule with CP (FLA handles state merge internally)
        output, _ = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cp_context=cp_context,
        )

        # 5. Norm + out_proj (local)
        z = z.reshape(batch_size, seq_len, self._num_v_local, self._head_v_dim)
        output = inner.norm(
            output.reshape(-1, self._head_v_dim),
            z.reshape(-1, self._head_v_dim),
        )

        output = inner.out_proj(output.reshape(batch_size, seq_len, -1))
        return output

    # ------------------------------------------------------------------
    # Unified forward: route based on init-time flag (no runtime branch cost)
    # ------------------------------------------------------------------

    def forward(self, hidden_states: Tensor, **kwargs) -> Tensor:
        if self._use_sp:
            return self.forward_sp(hidden_states)
        return self._inner(hidden_states, **kwargs)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def load_gated_deltanet_weights_from_full(
    tp_mod: TPQwen3_5GatedDeltaNet,
    full_mod,
) -> None:
    """Copy weights from a full ``Qwen3_5GatedDeltaNet`` (loaded from HF
    safetensors) into a TP variant, slicing per the conventions above.

    full_mod may be on a different device than tp_mod; the slicing
    happens at the source dtype/device and the result is cast on copy.
    """
    inner = tp_mod._inner
    W = tp_mod.tp_world
    r = tp_mod.tp_rank

    # in_proj_qkv: merged [q|k|v] custom shard.
    inner.in_proj_qkv.load_full_weight(full_mod.in_proj_qkv.weight.data, None)

    # in_proj_z: standard Colwise on value_dim.
    inner.in_proj_z.load_full_weight(full_mod.in_proj_z.weight.data, None)

    # in_proj_b / in_proj_a: Colwise on num_v_heads.
    inner.in_proj_b.load_full_weight(full_mod.in_proj_b.weight.data, None)
    inner.in_proj_a.load_full_weight(full_mod.in_proj_a.weight.data, None)

    # out_proj: Rowwise on value_dim.
    inner.out_proj.load_full_weight(full_mod.out_proj.weight.data, None)

    # conv1d: depthwise weight [conv_dim, 1, kernel]. Same per-head
    # interleaved shard as in_proj_qkv.
    full_conv_w = full_mod.conv1d.weight.data  # [conv_dim, 1, kernel]
    local_conv_w = _shard_merged_qkv(
        full_conv_w,
        tp_mod._key_dim_full,
        tp_mod._value_dim_full,
        r, W,
    )
    with torch.no_grad():
        inner.conv1d.weight.copy_(
            local_conv_w.to(inner.conv1d.weight.dtype).to(inner.conv1d.weight.device)
        )

    # dt_bias, A_log: simple slice on num_v_heads.
    num_v_local = inner.num_v_heads
    s = r * num_v_local
    e = s + num_v_local
    with torch.no_grad():
        inner.dt_bias.copy_(
            full_mod.dt_bias.data[s:e].to(inner.dt_bias.dtype).to(inner.dt_bias.device)
        )
        inner.A_log.copy_(
            full_mod.A_log.data[s:e].to(inner.A_log.dtype).to(inner.A_log.device)
        )

    # norm: replicated, copy full state.
    inner.norm.load_state_dict(full_mod.norm.state_dict())
