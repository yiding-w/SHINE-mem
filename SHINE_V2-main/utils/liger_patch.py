"""Liger-Kernel drop-in replacements for Qwen3_5 RMSNorm + lm_head loss + SwiGLU.

Three fused-kernel swaps that preserve training math (verified at bf16
numerical noise — see commit log):

  1. ``Qwen3_5RMSNorm`` → ``LigerRMSNorm`` (Gemma casting, offset=1.0,
     init zeros). Matches Qwen3_5's exact formula
     ``output_fp32 * (1 + w.float()).type_as(x)``. 64 × 2 = 128 calls/step
     on the LLM plus a final norm; each becomes a single fused Triton
     kernel instead of the 4 sequential PyTorch ops.

  2. ``self.lm_head(hs) → F.cross_entropy(logits, labels)`` →
     ``LigerFusedLinearCrossEntropyFunction`` — fuses the lm_head GEMM,
     log_softmax, NLL, and backward into one chunked Triton kernel. Never
     materialises the [B*T, V=248k] logits tensor. The existing chunked
     loop already trades speed for memory; FLCE recovers the speed too.

  3. ``silu(gate_proj(x)) * up_proj(x)`` →
     ``LigerSiLUMulFunction.apply(gate, up)`` — fuses the SiLU activation
     and element-wise multiply into one Triton kernel. Saves one
     intermediate activation tensor (silu output) in the backward pass
     via recomputation, and eliminates one kernel launch per MLP call.

Neither op is LoRA-wrapped, so this is a pure leaf-module swap that does
not touch the LoRA delta path.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_LIGER_RMSNORM_APPLIED = False


class LigerQwen3_5RMSNorm(nn.Module):
    """Liger-backed drop-in for ``Qwen3_5RMSNorm``.

    Qwen3_5 formula: ``out = (x.float() * rsqrt(mean(x^2)+eps)) * (1 + w.float())``
    then cast to x's dtype.  Gemma casting mode in LigerRMSNorm does the
    full computation in fp32 and casts back at the end, which matches
    exactly.  Weight init is zeros (the offset of 1.0 supplies the "+1").
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # Lazy import — keeps liger optional for environments without it.
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        self._fn = LigerRMSNormFunction
        self.weight = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = eps
        self.offset = 1.0
        # "gemma" mode: full fp32 compute, cast back at end (matches Qwen3_5).
        self._casting_mode = 1  # _CASTING_MODE_GEMMA
        self._in_place = True
        self._row_mode = None

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The Triton kernel inside LigerRMSNormFunction.apply is opaque to
        # dynamo and triggers a SymNodeVariable bug when traced from inside
        # a compiled region (the grid uses a symbolic n_rows derived from
        # the dynamic outer shape). Marking the whole forward as
        # compiler.disable keeps the kernel running in eager and lets the
        # enclosing @torch.compile region fuse around it.
        return self._fn.apply(
            x, self.weight, self.variance_epsilon, self.offset,
            self._casting_mode, self._in_place, self._row_mode,
        )

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def apply_liger_rmsnorm_patch() -> None:
    """Monkey-patch ``Qwen3_5RMSNorm`` in the installed transformers module
    AND every other module that has already done ``from ... import
    Qwen3_5RMSNorm`` (which captures a name binding by value — patching only
    the source module leaves stale references in importer namespaces).

    Idempotent. Must be called before ``from_pretrained`` so newly
    constructed decoder layers pick up the Liger class.
    """
    global _LIGER_RMSNORM_APPLIED
    if _LIGER_RMSNORM_APPLIED:
        return
    import sys
    from transformers.models.qwen3_5 import modeling_qwen3_5 as _m
    original = _m.Qwen3_5RMSNorm
    _m.Qwen3_5RMSNorm = LigerQwen3_5RMSNorm
    # Walk sys.modules and rebind every name that still points at the
    # original class. Covers LoraQwen3_5 etc. which already did a
    # ``from transformers...modeling_qwen3_5 import Qwen3_5RMSNorm`` at
    # import time.
    rebound = 0
    for modname, mod in list(sys.modules.items()):
        if mod is None or mod is _m:
            continue
        try:
            sym = getattr(mod, "Qwen3_5RMSNorm", None)
        except Exception:
            continue
        if sym is original:
            setattr(mod, "Qwen3_5RMSNorm", LigerQwen3_5RMSNorm)
            rebound += 1
    _LIGER_RMSNORM_APPLIED = True
    logger.info(
        f"[liger_patch] Qwen3_5RMSNorm → LigerQwen3_5RMSNorm "
        f"(source module + {rebound} importer modules rebound)"
    )


def fused_lm_head_loss(
    hidden_states: torch.Tensor,        # [B*T, H], grad-required, bf16
    lm_head_weight: torch.Tensor,       # [V, H]
    labels: torch.Tensor,               # [B*T], int64, -100 = ignore
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Fused linear + cross-entropy via Liger.

    Returns scalar loss (in hidden_states' dtype). Internally never
    materialises [B*T, V] logits; computes the lm_head GEMM, softmax, NLL,
    and gradient in a single Triton kernel.
    """
    from liger_kernel.ops.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyFunction,
    )
    # Signature: (_input, weight, target, bias, ce_weight, ignore_index,
    #             lse_square_scale, label_smoothing, reduction, ...)
    out = LigerFusedLinearCrossEntropyFunction.apply(
        hidden_states, lm_head_weight, labels,
        None,            # bias
        None,            # ce_weight
        ignore_index,
        0.0,             # lse_square_scale
        0.0,             # label_smoothing
        reduction,
    )
    # FLCE returns (loss, z_loss, token_accuracy); we only want loss.
    return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
# 3. Liger SwiGLU: fuse silu(gate) * up into a single Triton kernel
# ---------------------------------------------------------------------------

_LIGER_SWIGLU_APPLIED = False


@torch.compiler.disable
def _liger_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fused silu(a) * b via Liger Triton kernel.

    Marked @torch.compiler.disable because the Triton kernel accesses raw
    tensor data pointers which is incompatible with FakeTensor tracing.
    The PP compile region must NOT use fullgraph=True so that this graph
    break is allowed.
    """
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    return LigerSiLUMulFunction.apply(a, b)


def apply_liger_swiglu_patch() -> None:
    """Monkey-patch ``Qwen3_5MLP.forward`` and ``LoraQwen3_5MLP.forward``
    to use Liger's fused SwiGLU kernel (``silu(gate) * up`` in one Triton
    kernel instead of two separate PyTorch ops).

    Saves one intermediate activation tensor in the backward pass and
    eliminates one kernel launch per MLP call.

    Idempotent. Must be called before ``from_pretrained`` so newly
    constructed decoder layers pick up the patched forward.
    """
    global _LIGER_SWIGLU_APPLIED
    if _LIGER_SWIGLU_APPLIED:
        return

    from transformers.models.qwen3_5 import modeling_qwen3_5 as _m

    # --- Patch Qwen3_5MLP.forward (used in linear_attention layers) ---
    def _qwen3_5_mlp_forward(self, x):
        return self.down_proj(_liger_swiglu(self.gate_proj(x), self.up_proj(x)))

    _m.Qwen3_5MLP.forward = _qwen3_5_mlp_forward

    # --- Patch LoraQwen3_5MLP.forward (used in full_attention layers) ---
    import sys
    _lora_mod = sys.modules.get("src_transformers_lora.LoraQwen3_5", None)
    if _lora_mod is not None and hasattr(_lora_mod, "LoraQwen3_5MLP"):
        def _lora_mlp_forward(self, x, loradict=None, nograd_loradict=None, nograd_wdict=None):
            if loradict is None and nograd_loradict is None and nograd_wdict is None:
                return self.down_proj(
                    _liger_swiglu(self.gate_proj(x), self.up_proj(x))
                )
            else:
                gate_lora = loradict.get("gate") if loradict else None
                gate_nograd = nograd_loradict.get("gate") if nograd_loradict else None
                gate_w = nograd_wdict.get("gate") if nograd_wdict else None
                up_lora = loradict.get("up") if loradict else None
                up_nograd = nograd_loradict.get("up") if nograd_loradict else None
                up_w = nograd_wdict.get("up") if nograd_wdict else None
                down_lora = loradict.get("down") if loradict else None
                down_nograd = nograd_loradict.get("down") if nograd_loradict else None
                down_w = nograd_wdict.get("down") if nograd_wdict else None
                return self.down_proj(
                    _liger_swiglu(
                        self.gate_proj(x, gate_lora, gate_nograd, gate_w),
                        self.up_proj(x, up_lora, up_nograd, up_w),
                    ),
                    down_lora, down_nograd, down_w,
                )

        _lora_mod.LoraQwen3_5MLP.forward = _lora_mlp_forward
        logger.info("[liger_patch] LoraQwen3_5MLP.forward → Liger SwiGLU")

    _LIGER_SWIGLU_APPLIED = True
    logger.info("[liger_patch] Qwen3_5MLP.forward → Liger SwiGLU")
