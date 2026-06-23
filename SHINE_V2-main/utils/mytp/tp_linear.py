"""
TP-aware linear primitives for the tp_experiment branch.

Two classes replace the forked ``LoraLinear`` (`src_transformers_lora.Lora*`):

    ColwiseLoraLinear  – out_features sharded across the TP group.
    RowwiseLoraLinear  – in_features  sharded across the TP group.

Both keep ``LoraLinear``'s call signature ``forward(input, loradict=None)``
so the surrounding decoder-layer code does not need to change. The
loradict tensors (``A``, ``B``, optional ``C``) arrive **full / unsharded**;
each rank slices its own piece inside the forward.

The nograd_wdict tensors (``W``, optional ``C``) arrive **pre-sliced** to
the local TP shard. This is because W is a full-rank matrix [Lb, in, out]
which is much larger than LoRA's low-rank factors; pre-slicing saves memory.
The DetachState is responsible for slicing wdict before returning it.

Sharding conventions inside one TP group of size ``W``::

    Colwise (q_proj, k_proj, v_proj, gate_proj, up_proj):
        weight     : [out/W, in]               local
        bias       : [out/W]                   local
        A          : [Lb, in, r]               replicated
        B          : [Lb, r, out]              sliced  [:, :, rank*out/W : (rank+1)*out/W]
        C          : [Lb, out]                 sliced  [:,    rank*out/W : (rank+1)*out/W]
        input      : [..., in]                 replicated
        output     : [..., out/W]              local — caller is expected to
                                               consume this as a shard
                                               (e.g. attention heads /
                                               MLP intermediate dim)

    Rowwise (o_proj, down_proj):
        weight     : [out, in/W]               local
        bias       : [out]                     replicated, added AFTER all-reduce
        A          : [Lb, in, r]               sliced  [:, rank*in/W : (rank+1)*in/W, :]
        B          : [Lb, r, out]              replicated
        C          : [Lb, out]                 replicated, added AFTER all-reduce
        input      : [..., in/W]               local
        output     : [..., out]                full — base_partial + lora_partial
                                               are summed across TP via all-reduce,
                                               then bias/C are added once.

    nograd_wdict (pre-sliced, NOT sliced inside forward):
      Colwise:
        W          : [Lb, in, out/W]           pre-sliced on output dim
        C          : [Lb, out/W]               pre-sliced on output dim
      Rowwise:
        W          : [Lb, in/W, out]           pre-sliced on input dim
        C          : [Lb, out]                 replicated, added pre-reduce scaled by 1/W

The base linear and the LoRA delta are summed into a single ``out_partial``
tensor on every rank before the all-reduce, so RowwiseLoraLinear only
issues one collective per call instead of two.
"""

from __future__ import annotations

from math import sqrt
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = ["ColwiseLoraLinear", "RowwiseLoraLinear", "shard_linear_weight"]


# ---------------------------------------------------------------------------
# LoRA-dict API helpers shared with the hypernetwork
# ---------------------------------------------------------------------------
#
# The hypernetwork produces a flat plain_tensor of LoRA parameters and the
# decoder layers / linears chop it up via lora_params_numel + generate_lora_dict
# at each level. For TP linears the loradict must be **full / unsharded** —
# the per-rank slicing happens inside ``forward``. So lora_params_numel /
# generate_lora_dict / init_lora_dict / set_generate_func must report and
# produce full-dim values, identical between the TP linears and stock
# LoraLinear. We get that by composing a ``LoraHelper`` (which carries no
# nn.Parameter — pure dim metadata) with full dims and delegating.


def _make_full_helper(in_features: int, out_features_full: int, bias: bool):
    """Return a fresh ``LoraHelper`` with full / unsharded dims used purely
    for the lora_params_numel / generate_lora_dict / init_lora_dict /
    set_generate_func calls. Imported lazily so utils.mytp.tp_linear stays free
    of src_transformers_lora dependencies at import time.
    """
    from src_transformers_lora.LoraQwen3_5 import LoraHelper
    return LoraHelper(in_features=in_features, out_features=out_features_full, bias=bias)


# ---------------------------------------------------------------------------
# TP autograd region helper: identity forward, all-reduce-SUM backward
# ---------------------------------------------------------------------------
#
# Used to wrap the **replicated input** of a ColwiseLoraLinear so backward
# correctly sums the per-rank partial ``dL/dinput`` across the TP group
# before passing it upstream. Without this, downstream (previous-layer)
# gradients are missing the cross-rank contributions and any parameter
# whose grad path runs through inter-layer hidden states sees a partial
# (wrong) gradient. Megatron calls this the ``f`` function.
#
# RowwiseLoraLinear does not need a corresponding ``g`` autograd Function
# on its output: the all-reduce-SUM in its forward, combined with the
# default identity-backward PyTorch falls back to for ``dist.all_reduce``,
# already implements ``g`` (all-reduce forward, identity backward) — but
# only because the in-place sum-reduce makes every rank's output equal
# and every rank's upstream gradient identical, which is exactly what
# identity-backward propagates.


class _CopyToTPRegion(torch.autograd.Function):
    """Identity in forward; all-reduce-SUM across the TP group in backward."""

    @staticmethod
    def forward(ctx, x, tp_group, tp_world):
        ctx.tp_group = tp_group
        ctx.tp_world = tp_world
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.tp_group is not None and ctx.tp_world > 1:
            # Clone first because grad_output may be a view that the
            # autograd engine still references.
            g = grad_output.contiguous()
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=ctx.tp_group)
            return g, None, None
        return grad_output, None, None


# Mark the TP-collective entry as opaque to torch.compile: dynamo can't
# safely trace through dist.all_reduce, but allowing it to compile *around*
# this hop (the per-layer attn/MLP regions in between) is what we want.
# Without these wrappers compile sees the autograd.Function and refuses to
# trace at all (prior attempts found compile to be a no-op for LLM layers).
@torch.compiler.disable
def copy_to_tp_region(x: Tensor, tp_group, tp_world: int) -> Tensor:
    """Public wrapper around ``_CopyToTPRegion.apply``."""
    return _CopyToTPRegion.apply(x, tp_group, tp_world)


class _ReduceFromTPRegion(torch.autograd.Function):
    """All-reduce-SUM in forward; identity in backward.

    This is the counterpart of ``_CopyToTPRegion`` (identity fwd,
    all-reduce bwd). Together they form the Megatron f/g pair:
      * Colwise input:  f = copy_to_tp_region (identity fwd, all-reduce bwd)
      * Rowwise output: g = reduce_from_tp_region (all-reduce fwd, identity bwd)
    """

    @staticmethod
    def forward(ctx, x, tp_group, tp_world):
        if tp_group is not None and tp_world > 1:
            # Use contiguous clone to avoid in-place modification of the
            # input tensor which can confuse autograd's version tracking.
            out = x.clone()
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
            return out
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Identity backward: each rank's upstream gradient is already
        # correct because after forward all ranks hold the same value,
        # so the loss function produces the same gradient on every rank.
        return grad_output, None, None


@torch.compiler.disable
def reduce_from_tp_region(x: Tensor, tp_group, tp_world: int) -> Tensor:
    """Public wrapper around ``_ReduceFromTPRegion.apply``."""
    return _ReduceFromTPRegion.apply(x, tp_group, tp_world)


@torch.compiler.disable
def _all_reduce_tp(t: Tensor, tp_group) -> None:
    """In-place all-reduce-SUM (legacy, no autograd support).

    DEPRECATED: Use ``reduce_from_tp_region`` instead for correct backward.
    Kept only for non-differentiable paths (e.g. inference-only)."""
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=tp_group)


# ---------------------------------------------------------------------------
# Weight-loading helper
# ---------------------------------------------------------------------------

def shard_linear_weight(
    full_weight: Tensor,
    axis: str,
    tp_rank: int,
    tp_world: int,
) -> Tensor:
    """
    Slice a full ``[out, in]`` weight tensor into this rank's TP shard.

    Args:
        full_weight: ``[out, in]`` weight matrix as found in HF state_dict.
        axis: ``"col"`` for column-wise (shard ``out``) or ``"row"`` for
            row-wise (shard ``in``).
        tp_rank: 0-based rank within the TP group.
        tp_world: TP group size.

    Returns:
        Contiguous local slice. For col: ``[out/W, in]``. For row:
        ``[out, in/W]``.
    """
    if axis == "col":
        per = full_weight.shape[0] // tp_world
        if full_weight.shape[0] != per * tp_world:
            raise ValueError(
                f"shard_linear_weight(col): out_features={full_weight.shape[0]} "
                f"not divisible by tp_world={tp_world}"
            )
        return full_weight[tp_rank * per : (tp_rank + 1) * per].contiguous()
    elif axis == "row":
        per = full_weight.shape[1] // tp_world
        if full_weight.shape[1] != per * tp_world:
            raise ValueError(
                f"shard_linear_weight(row): in_features={full_weight.shape[1]} "
                f"not divisible by tp_world={tp_world}"
            )
        return full_weight[:, tp_rank * per : (tp_rank + 1) * per].contiguous()
    else:
        raise ValueError(f"shard_linear_weight: axis must be 'col' or 'row', got {axis!r}")


# ---------------------------------------------------------------------------
# Internal: LoRA delta math (shared between Col and Row variants)
# ---------------------------------------------------------------------------

def _lora_delta(
    input_view_flat: Tensor,
    A: Tensor,
    B: Tensor,
    C: Optional[Tensor],
) -> Tensor:
    """Compute ``input @ A @ B (+ C)`` with the standard ``Lb, beams, S, d``
    reshape used by the forked LoraLinear. Caller passes ``A``, ``B``, ``C``
    already sliced/sharded as appropriate.
    """
    Lb = A.shape[0]
    in_dim = A.shape[1]
    out_dim = B.shape[2]
    torch._assert(
        input_view_flat.shape[-1] == in_dim,
        "lora_delta: input last dim must match A's in dim",
    )
    torch._assert(
        input_view_flat.shape[0] % Lb == 0,
        "lora_delta: input batch must be a multiple of Lb",
    )
    num_beams = input_view_flat.shape[0] // Lb
    x = input_view_flat.reshape(Lb, num_beams, -1, in_dim)
    tmp = torch.matmul(x, A[:, None, :, :])       # [Lb, beams, S, r]
    lora_out = torch.matmul(tmp, B[:, None, :, :]) # [Lb, beams, S, out]
    if C is not None:
        lora_out = lora_out + C[:, None, None, :]
    return lora_out.reshape(*input_view_flat.shape[:-1], out_dim)


# ---------------------------------------------------------------------------
# Colwise variant
# ---------------------------------------------------------------------------

class ColwiseLoraLinear(nn.Module):
    """Column-wise sharded linear with optional LoRA delta.

    Replaces a stock ``LoraLinear(in, out)`` whose ``out`` is being
    sharded across ``tp_world`` ranks (q_proj, k_proj, v_proj, gate_proj,
    up_proj).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        tp_rank: int,
        tp_world: int,
        tp_process_group=None,
        device=None,
        dtype=None,
        shard_rank: Optional[int] = None,
        shard_world: Optional[int] = None,
    ):
        """``shard_rank`` / ``shard_world`` decouple **weight sharding** from
        the **TP-collective** identity. By default both equal
        ``tp_rank`` / ``tp_world`` — the normal case where each TP rank
        holds a unique slice of the output. For KV-head replication
        (full_attention k_proj / v_proj at TP=8 with num_kv_heads=4),
        pass ``shard_rank = tp_rank // (tp_world // num_kv_heads)`` and
        ``shard_world = num_kv_heads`` so pairs of TP ranks share the
        same weight slice.

        The TP collective for backward (``copy_to_tp_region``) still uses
        the full ``tp_world`` / ``tp_group`` — every TP rank's
        ``dL/dinput`` contribution is summed, which is correct
        irrespective of whether the linear's weights are KV-replicated:
        each rank's contribution is per its local q heads, and summing
        across all q heads reconstructs the full gradient.
        """
        super().__init__()
        if shard_rank is None:
            shard_rank = tp_rank
        if shard_world is None:
            shard_world = tp_world
        if out_features % shard_world != 0:
            raise ValueError(
                f"ColwiseLoraLinear: out_features={out_features} not divisible by shard_world={shard_world}"
            )
        self.in_features = in_features
        self.out_features_total = out_features
        self.out_features_local = out_features // shard_world
        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.tp_group = tp_process_group
        self.shard_rank = shard_rank
        self.shard_world = shard_world
        self.weight = nn.Parameter(
            torch.empty(self.out_features_local, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_local, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        # Full-dim helper so lora_params_numel / generate_lora_dict / etc.
        # behave identically to a stock LoraLinear from the hypernetwork's
        # point of view.
        self._full_helper = _make_full_helper(in_features, out_features, bias)

    def lora_params_numel(self, r: int) -> int:
        return self._full_helper.lora_params_numel(r)

    def set_generate_func(self, method: str) -> None:
        self._full_helper.set_generate_func(method)

    def generate_lora_dict(self, r: int, scale: float, plain_tensor: Tensor) -> Optional[dict]:
        if r == 0:
            return None
        return self._full_helper.generate_lora_dict(r, scale, plain_tensor)

    def init_lora_dict(self, r: int, scale: float, device, dtype) -> Optional[dict]:
        return self._full_helper.init_lora_dict(r, scale, device, dtype)

    def forward(
        self,
        input: Tensor,
        loradict: Optional[dict] = None,
        nograd_loradict: Optional[dict] = None,
        nograd_wdict: Optional[dict] = None,
    ) -> Tensor:
        # Wrap input so backward sums dL/dinput across TP (every rank used the
        # same input with a different W slice, so each rank's autograd only
        # captures its slice's contribution — without this all-reduce-SUM the
        # upstream layer / hypernetwork would see a partial gradient).
        input = copy_to_tp_region(input, self.tp_group, self.tp_world)
        # Base output is sharded along the last dim.
        out = F.linear(input, self.weight, self.bias)  # [..., out_local]
        if loradict is None and nograd_loradict is None and nograd_wdict is None:
            return out
        s, e = self.shard_rank * self.out_features_local, (self.shard_rank + 1) * self.out_features_local
        for ld in (loradict, nograd_loradict):
            if ld is None:
                continue
            A = ld["A"]                      # [Lb, in, r] — full
            B_local = ld["B"][:, :, s:e]     # [Lb, r, out_local]
            C_full = ld.get("C")
            C_local = C_full[:, s:e] if C_full is not None else None
            out = out + _lora_delta(input, A, B_local, C_local)
        # nograd_wdict: pre-sliced W delta (already column-sharded)
        if nograd_wdict is not None:
            out = out + self._compute_w_delta_colwise(input, nograd_wdict)
        return out

    def _compute_w_delta_colwise(self, input: Tensor, wdict: dict) -> Tensor:
        """Compute x @ W_local (+ C_local) for a pre-sliced column-wise wdict.

        Convention: In TP mode, nograd_wdict is always pre-sliced to the
        local TP shard before being passed in. So:
            W: [Lb, in, out_local]  (already sliced on output dim)
            C: [Lb, out_local] or None  (already sliced on output dim)
        """
        W_local = wdict["W"]        # [Lb, in, out_local]
        C_local = wdict.get("C", None)  # [Lb, out_local] or None
        Lb = W_local.shape[0]
        # Flatten to [Lb, beams*S, in]
        x = input.reshape(Lb, -1, input.shape[-1])
        w_out = torch.matmul(x, W_local)  # [Lb, beams*S, out_local]
        if C_local is not None:
            w_out = w_out + C_local[:, None, :]
        return w_out.reshape(*input.shape[:-1], self.out_features_local)

    def load_full_weight(self, full_weight: Tensor, full_bias: Optional[Tensor]):
        """Copy the slice corresponding to this rank from ``full_weight``
        / ``full_bias`` into this module's parameters. With KV
        replication (shard_world < tp_world), pairs of TP ranks load
        the same slice."""
        local_w = shard_linear_weight(full_weight, "col", self.shard_rank, self.shard_world)
        with torch.no_grad():
            self.weight.copy_(local_w.to(self.weight.dtype).to(self.weight.device))
        if self.bias is not None:
            if full_bias is None:
                raise ValueError("ColwiseLoraLinear has bias but full_bias is None")
            s = self.shard_rank * self.out_features_local
            e = s + self.out_features_local
            with torch.no_grad():
                self.bias.copy_(full_bias[s:e].to(self.bias.dtype).to(self.bias.device))


# ---------------------------------------------------------------------------
# Rowwise variant
# ---------------------------------------------------------------------------

class RowwiseLoraLinear(nn.Module):
    """Row-wise sharded linear with optional LoRA delta.

    Replaces a stock ``LoraLinear(in, out)`` whose ``in`` is being sharded
    across ``tp_world`` ranks (o_proj, down_proj).

    Performs **one** all-reduce on the combined ``base_partial + lora_partial``
    so calling this once per layer costs one collective, not two.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        tp_rank: int,
        tp_world: int,
        tp_process_group,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if in_features % tp_world != 0:
            raise ValueError(
                f"RowwiseLoraLinear: in_features={in_features} not divisible by tp_world={tp_world}"
            )
        self.in_features_total = in_features
        self.in_features_local = in_features // tp_world
        self.out_features = out_features
        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.tp_group = tp_process_group
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_local, device=device, dtype=dtype)
        )
        # Bias is replicated (added once after the all-reduce). We still
        # store it locally for convenience; the value is identical on
        # every TP rank.
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self._full_helper = _make_full_helper(in_features, out_features, bias)

    def lora_params_numel(self, r: int) -> int:
        return self._full_helper.lora_params_numel(r)

    def set_generate_func(self, method: str) -> None:
        self._full_helper.set_generate_func(method)

    def generate_lora_dict(self, r: int, scale: float, plain_tensor: Tensor) -> Optional[dict]:
        if r == 0:
            return None
        return self._full_helper.generate_lora_dict(r, scale, plain_tensor)

    def init_lora_dict(self, r: int, scale: float, device, dtype) -> Optional[dict]:
        return self._full_helper.init_lora_dict(r, scale, device, dtype)

    def forward(
        self,
        input: Tensor,
        loradict: Optional[dict] = None,
        nograd_loradict: Optional[dict] = None,
        nograd_wdict: Optional[dict] = None,
    ) -> Tensor:
        # Base partial + per-loradict deltas summed pre-all-reduce.
        # Bias and LoRA C are added pre-reduce scaled by 1/tp_world so the
        # post-reduce sum gives back exactly one contribution (and backward
        # gradients on those replicated params match the non-sharded ref).
        out_partial = F.linear(input, self.weight, None)
        s = self.tp_rank * self.in_features_local
        e = s + self.in_features_local
        for ld in (loradict, nograd_loradict):
            if ld is None:
                continue
            A_local = ld["A"][:, s:e, :]   # [Lb, in_local, r]
            B_full = ld["B"]                # [Lb, r, out]
            out_partial = out_partial + _lora_delta(input, A_local, B_full, None)
        # nograd_wdict: pre-sliced W delta (already row-sharded on input dim)
        if nograd_wdict is not None:
            out_partial = out_partial + self._compute_w_delta_rowwise(input, nograd_wdict)

        W = max(self.tp_world, 1)

        if self.bias is not None:
            out_partial = out_partial + self.bias / W

        for ld in (loradict, nograd_loradict):
            if ld is None:
                continue
            C_full = ld.get("C")
            if C_full is None:
                continue
            Lb = C_full.shape[0]
            C_scaled = C_full / W
            if out_partial.shape[0] != Lb:
                out_partial_view = out_partial.reshape(Lb, -1, *out_partial.shape[1:])
                out_partial_view = out_partial_view + C_scaled.view(
                    Lb, *(1,) * (out_partial_view.dim() - 2), self.out_features
                )
                out_partial = out_partial_view.reshape(*out_partial.shape)
            else:
                out_partial = out_partial + C_scaled.view(
                    Lb, *(1,) * (out_partial.dim() - 2), self.out_features
                )

        # nograd_wdict C: added pre-reduce, scaled by 1/tp_world (same as lora C)
        if nograd_wdict is not None:
            C_w = nograd_wdict.get("C", None)
            if C_w is not None:
                Lb = C_w.shape[0]
                C_scaled = C_w / W
                if out_partial.shape[0] != Lb:
                    out_partial_view = out_partial.reshape(Lb, -1, *out_partial.shape[1:])
                    out_partial_view = out_partial_view + C_scaled.view(
                        Lb, *(1,) * (out_partial_view.dim() - 2), self.out_features
                    )
                    out_partial = out_partial_view.reshape(*out_partial.shape)
                else:
                    out_partial = out_partial + C_scaled.view(
                        Lb, *(1,) * (out_partial.dim() - 2), self.out_features
                    )

        if self.tp_group is not None and self.tp_world > 1:
            out_partial = reduce_from_tp_region(out_partial, self.tp_group, self.tp_world)
        return out_partial

    def _compute_w_delta_rowwise(self, input: Tensor, wdict: dict) -> Tensor:
        """Compute x @ W_local for a pre-sliced row-wise wdict.

        Convention: In TP mode, nograd_wdict is always pre-sliced to the
        local TP shard before being passed in. So:
            W: [Lb, in_local, out]  (already sliced on input dim)
        C is handled separately after all lora C's, pre-reduce.
        """
        W_local = wdict["W"]        # [Lb, in_local, out]
        Lb = W_local.shape[0]
        x = input.reshape(Lb, -1, input.shape[-1])  # [Lb, beams*S, in_local]
        w_out = torch.matmul(x, W_local)  # [Lb, beams*S, out]
        return w_out.reshape(*input.shape[:-1], self.out_features)

    def load_full_weight(self, full_weight: Tensor, full_bias: Optional[Tensor]):
        local_w = shard_linear_weight(full_weight, "row", self.tp_rank, self.tp_world)
        with torch.no_grad():
            self.weight.copy_(local_w.to(self.weight.dtype).to(self.weight.device))
        if self.bias is not None:
            if full_bias is None:
                raise ValueError("RowwiseLoraLinear has bias but full_bias is None")
            with torch.no_grad():
                self.bias.copy_(full_bias.to(self.bias.dtype).to(self.bias.device))
