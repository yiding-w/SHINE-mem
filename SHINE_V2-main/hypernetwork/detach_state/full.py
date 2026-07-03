"""
FullDetachState: maintains a wdict as persistent state (PP mode only).

The wdict has batch dimension = local_batch_size, distributed across
PP stages (each GPU stores only its own layers' wdict).

Write uses torch.baddbmm for fused A@B accumulation without intermediate allocation.
Only PP mode is supported; TP mode raises NotImplementedError at creation time.

All tensors created and stored internally have requires_grad=False.
"""

from __future__ import annotations

import logging
import torch
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

from hypernetwork.detach_state.base import BaseDetachState

logger = logging.getLogger(__name__)

class FullDetachState(BaseDetachState):
    """
    Full DetachState: maintains a wdict (W = accumulated A@B) as persistent state.

    The wdict batch dimension equals local_batch_size. Each PP stage stores
    only the wdict for its own layers. Write uses baddbmm to fuse the
    A@B computation and accumulation into a single operation.

    Single-buffer design: read and write operate on the same _wdict.
    Write happens after Phase C (and Phase C' if distill) for each micro-batch,
    ensuring all reads for the current step complete before any writes.

    All internal tensors have requires_grad=False.
    """

    def __init__(self, cfg, *, local_batch_size: int, micro_batch_size: int,
                 parallel_mode: str, total_stages: int, data_parallel_size: int,
                 total_gpus_per_node: int, my_stage: int,
                 my_layer_indices: List[int], num_llm_layers: int):
        super().__init__(cfg)
        self._local_batch_size = local_batch_size
        self._micro_batch_size = micro_batch_size
        self._wdict = None  # Lazy init on first write
        self._last_sq_norms = None  # Per-sample sq_norms from previous step (after node all_reduce)
        self._num_mb = local_batch_size // micro_batch_size
        # Per-sample update step counters: one counter per sample in local_batch_size
        self._update_steps = [0] * self._local_batch_size

        # Frozen config: all settings that must match on checkpoint resume.
        # Changing any of these after a checkpoint would corrupt the wdict.
        self._frozen_cfg = {
            "type": "full",
            "parallel_mode": parallel_mode,
            "local_batch_size": local_batch_size,
            "micro_batch_size": micro_batch_size,
            "total_stages": total_stages,
            "data_parallel_size": data_parallel_size,
            "total_gpus_per_node": total_gpus_per_node,
            "my_stage": my_stage,
            "my_layer_indices": tuple(my_layer_indices),
            "num_llm_layers": num_llm_layers,
            # "wdict_shapes" is added lazily on first write
        }

        logger.info(
            f"[DetachState] Created FullDetachState "
            f"(local_batch_size={local_batch_size}, "
            f"micro_batch_size={micro_batch_size}, "
            f"stage={my_stage}, layers={my_layer_indices})"
        )

    # ------------------------------------------------------------------
    # Read interface
    # ------------------------------------------------------------------

    def read(self, mb_idx: Optional[int] = None) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Return (None, wdict) or (None, wdict_slice).

        Args:
            mb_idx: If None, returns the full wdict (all batch positions).
                    If provided, returns the slice for that micro-batch:
                    wdict[mb_idx*mbs : (mb_idx+1)*mbs] along batch dim.

        Returns:
            (None, wdict_or_slice). First element is always None (no loradict).
        """
        if self._wdict is None:
            return None, None
        if mb_idx is None:
            return None, self._wdict
        start = mb_idx * self._micro_batch_size
        end = start + self._micro_batch_size
        return None, self._slice_wdict(self._wdict, start, end)

    # ------------------------------------------------------------------
    # Write interface
    # ------------------------------------------------------------------

    def _write_impl(self, loradict: Optional[Dict], mb_idx: Optional[int] = None,
                    precomputed_wdict: Optional[Dict] = None) -> None:
        """
        Accumulate loradict into wdict at the batch position for mb_idx.

        If precomputed_wdict is provided (from compute_regu_loss), directly
        copies it into the appropriate batch position, skipping A@B computation.
        Otherwise, uses torch.baddbmm to fuse A@B + accumulate in one operation:
            wdict.W[start:end] += A @ B  (via baddbmm with beta=1, alpha=1)

        This method is called by base.write() which wraps this call in
        torch.no_grad(). When precomputed_wdict is None, base.write() also
        detaches the loradict before passing it here.

        Args:
            loradict: The detached loradict from hypernetwork_forward_with_grad.
                      Structure: {layer_idx: {linear_group: {proj: {"A":.., "B":.., "C":..}}}}
                      Leaf tensors have batch dim = micro_batch_size.
            mb_idx: The micro-batch index (0-based). Required for FullDetachState.
            precomputed_wdict: Optional precomputed new wdict slice (batch dim =
                      micro_batch_size). If provided, directly copies into the
                      batch position instead of computing A@B.

        Raises:
            ValueError: If mb_idx is None (required for full mode).
        """
        if loradict is None and precomputed_wdict is None:
            return

        if mb_idx is None:
            raise ValueError(
                "FullDetachState._write_impl requires mb_idx. "
                "Call write(loradict, mb_idx=<int>) instead of write(loradict)."
            )

        # Lazy init: create zero wdict on first write
        if self._wdict is None:
            ref = loradict if loradict is not None else precomputed_wdict
            if ref is None:
                return
            if loradict is not None:
                self._wdict = self._create_zero_wdict_from_loradict(loradict)
            else:
                self._wdict = self._create_zero_wdict_from_wdict(precomputed_wdict)
            # Freeze wdict shapes for checkpoint validation
            self._frozen_cfg["wdict_shapes"] = self._extract_wdict_shapes(self._wdict)

        batch_start = mb_idx * self._micro_batch_size

        if precomputed_wdict is not None:
            # Directly copy precomputed new wdict slice into position
            self._copy_wdict_slice(self._wdict, precomputed_wdict, batch_start)
        else:
            # Accumulate using baddbmm (fused A@B + add)
            self._baddbmm_accumulate(self._wdict, loradict, batch_start)

    # ------------------------------------------------------------------
    # Reset / state management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset wdict to zeros (if initialized) or None."""
        if self._wdict is not None:
            self._zero_wdict(self._wdict)

    def state_dict(self) -> Dict:
        """Serialize wdict + frozen config + update_steps for checkpointing."""
        return {
            "frozen_cfg": self._frozen_cfg,
            "wdict": self._wdict,
            "update_steps": list(self._update_steps),
        }

    def load_state_dict(self, state: Dict) -> None:
        """Restore wdict + update_steps from checkpoint, with strict config validation."""
        saved_cfg = state.get("frozen_cfg", {})
        self._validate_frozen_cfg(saved_cfg)
        self._wdict = state.get("wdict", None)
        self._last_sq_norms = None  # Not persisted; will be recomputed
        # Ensure all loaded tensors have requires_grad=False
        if self._wdict is not None:
            self._ensure_no_grad(self._wdict)
        # Restore wdict_shapes if present in saved state but not yet in ours
        if "wdict_shapes" in saved_cfg and "wdict_shapes" not in self._frozen_cfg:
            self._frozen_cfg["wdict_shapes"] = saved_cfg["wdict_shapes"]
        # Restore per-sample update step counters
        self._update_steps = list(state["update_steps"])
        # Pad or truncate to match current local_batch_size
        while len(self._update_steps) < self._local_batch_size:
            self._update_steps.append(0)
        self._update_steps = self._update_steps[:self._local_batch_size]

    # ------------------------------------------------------------------
    # Frozen config validation
    # ------------------------------------------------------------------

    def _validate_frozen_cfg(self, saved_cfg: Dict) -> None:
        """Validate that saved config matches current config exactly.

        Raises ValueError if any critical setting differs between the
        checkpoint and the current runtime configuration.
        """
        mismatches = []
        for key, current_val in self._frozen_cfg.items():
            if key not in saved_cfg:
                continue  # New key added after checkpoint was saved — skip
            saved_val = saved_cfg[key]
            if saved_val != current_val:
                mismatches.append(
                    f"  '{key}': checkpoint={saved_val}, current={current_val}"
                )
        if mismatches:
            raise ValueError(
                f"FullDetachState config mismatch — cannot resume:\n"
                + "\n".join(mismatches)
                + "\n\nThe wdict was saved with different parallelism/batch settings. "
                + "Either use the same settings or start fresh (delete the checkpoint)."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_zero_wdict_from_loradict(self, loradict: Dict) -> Dict:
        """
        Create a zero-initialized wdict with batch dim = local_batch_size,
        using the loradict structure as reference for dimensions.

        All created tensors have requires_grad=False (guaranteed by
        torch.no_grad() context in base.write() and detach in the caller).

        loradict leaf: {"A": [Lb, in, r], "B": [Lb, r, out], "C": [Lb, out] | None}
        wdict leaf:    {"W": [local_batch_size, in, out], "C": [local_batch_size, out] | None}
        """
        return self._build_zero_wdict_recursive(loradict)

    def _build_zero_wdict_recursive(self, loradict) -> Optional[Dict]:
        if loradict is None:
            return None
        # Check if this is a loradict leaf
        if "A" in loradict and "B" in loradict:
            A = loradict["A"]  # [Lb, in, r]
            B = loradict["B"]  # [Lb, r, out]
            C = loradict.get("C", None)
            _, in_dim, _ = A.shape
            _, _, out_dim = B.shape
            device = A.device
            dtype = A.dtype
            W_zero = torch.zeros(
                self._local_batch_size, in_dim, out_dim,
                device=device, dtype=dtype, requires_grad=False,
            )
            C_zero = None
            if C is not None:
                C_zero = torch.zeros(
                    self._local_batch_size, out_dim,
                    device=device, dtype=dtype, requires_grad=False,
                )
            return {"W": W_zero, "C": C_zero}
        # Recurse into sub-dicts
        result = {}
        for key, value in loradict.items():
            if value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._build_zero_wdict_recursive(value)
            else:
                result[key] = value
        return result

    def _baddbmm_accumulate(self, wdict: Dict, loradict: Dict, batch_start: int) -> None:
        """
        In-place accumulate loradict's A@B into wdict using baddbmm.

        For each leaf:
            wdict["W"][start:end] = baddbmm(wdict["W"][start:end], A, B)
            wdict["C"][start:end] += C  (if C is not None)

        baddbmm computes: out = beta * input + alpha * batch1 @ batch2
        With beta=1, alpha=1: out = input + A @ B

        The loradict is already detached by base.write() before reaching here
        (only called when precomputed_wdict is None).
        """
        if wdict is None or loradict is None:
            return

        # wdict leaf
        if "W" in wdict:
            # loradict leaf
            if "A" in loradict and "B" in loradict:
                A = loradict["A"]  # [Lb, in, r]
                B = loradict["B"]  # [Lb, r, out]
                Lb = A.shape[0]
                W_slice = wdict["W"][batch_start:batch_start + Lb]  # [Lb, in, out]
                # Fused: W_slice = W_slice + A @ B
                torch.baddbmm(W_slice, A, B, beta=1, alpha=1, out=W_slice)

                C = loradict.get("C", None)
                if C is not None and wdict["C"] is not None:
                    wdict["C"][batch_start:batch_start + Lb].add_(C)
            return

        # Recurse into sub-dicts
        for key in wdict:
            if wdict[key] is None or key not in loradict or loradict[key] is None:
                continue
            if isinstance(wdict[key], dict) and isinstance(loradict[key], dict):
                self._baddbmm_accumulate(wdict[key], loradict[key], batch_start)

    def _slice_wdict(self, wdict: Dict, start: int, end: int) -> Optional[Dict]:
        """Return a view/slice of wdict along the batch dimension."""
        if wdict is None:
            return None
        if "W" in wdict:  # leaf
            W_slice = wdict["W"][start:end]
            C_slice = wdict["C"][start:end] if wdict.get("C") is not None else None
            return {"W": W_slice, "C": C_slice}
        result = {}
        for key, value in wdict.items():
            if value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._slice_wdict(value, start, end)
            else:
                result[key] = value
        return result

    def _zero_wdict(self, wdict: Dict) -> None:
        """In-place zero all tensors in wdict."""
        if wdict is None:
            return
        if "W" in wdict:
            wdict["W"].zero_()
            if wdict.get("C") is not None:
                wdict["C"].zero_()
            return
        for key, value in wdict.items():
            if isinstance(value, dict):
                self._zero_wdict(value)

    def _ensure_no_grad(self, wdict: Dict) -> None:
        """Ensure all tensors in wdict have requires_grad=False.

        This is a defensive check after loading from checkpoint to prevent
        any accidental gradient tracking on stored state.
        """
        if wdict is None:
            return
        if "W" in wdict:
            if wdict["W"].requires_grad:
                wdict["W"].requires_grad_(False)
            if wdict.get("C") is not None and wdict["C"].requires_grad:
                wdict["C"].requires_grad_(False)
            return
        for key, value in wdict.items():
            if isinstance(value, dict):
                self._ensure_no_grad(value)

    def _extract_wdict_shapes(self, wdict: Dict) -> Optional[Dict]:
        """Extract tensor shapes from wdict for checkpoint validation."""
        if wdict is None:
            return None
        if "W" in wdict:
            shapes = {"W": list(wdict["W"].shape)}
            if wdict.get("C") is not None:
                shapes["C"] = list(wdict["C"].shape)
            else:
                shapes["C"] = None
            return shapes
        result = {}
        for key, value in wdict.items():
            if value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._extract_wdict_shapes(value)
            else:
                result[key] = None
        return result

    def __repr__(self) -> str:
        return (
            f"FullDetachState(local_batch_size={self._local_batch_size}, "
            f"micro_batch_size={self._micro_batch_size}, "
            f"stage={self._frozen_cfg['my_stage']}, "
            f"initialized={self._wdict is not None})"
        )

    # ------------------------------------------------------------------
    # Evaluation context manager
    # ------------------------------------------------------------------

    @contextmanager
    def eval_context(self, eval_local_batch_size: int = 0):
        """Context manager for evaluation: provides a fresh zero-initialized
        wdict independent of training state.

        On entry: saves training state (_wdict, _last_sq_norms, _update_steps,
                  _local_batch_size, _num_mb) and replaces with fresh zero state.
        On exit: restores training state and releases eval state (GC).

        Args:
            eval_local_batch_size: The local batch size used during evaluation.
                If 0 or not provided, uses the training local_batch_size.
        """
        # Save training state
        train_wdict = self._wdict
        train_last_sq_norms = self._last_sq_norms
        train_update_steps = self._update_steps
        train_num_mb = self._num_mb
        train_local_batch_size = self._local_batch_size

        # Determine eval batch size
        eval_bs = eval_local_batch_size if eval_local_batch_size > 0 else train_local_batch_size
        eval_num_mb = eval_bs // self._micro_batch_size

        # Replace with fresh eval state (None = lazy init from zero on first write)
        self._wdict = None
        self._last_sq_norms = None
        self._local_batch_size = eval_bs
        self._num_mb = eval_num_mb
        self._update_steps = [0] * eval_bs

        try:
            yield
        finally:
            # Release eval state explicitly
            del self._wdict
            # Restore training state
            self._wdict = train_wdict
            self._last_sq_norms = train_last_sq_norms
            self._update_steps = train_update_steps
            self._num_mb = train_num_mb
            self._local_batch_size = train_local_batch_size

    # ------------------------------------------------------------------
    # Regularization loss
    # ------------------------------------------------------------------

    def compute_regu_loss(self, loradict: Optional[Dict], mb_idx: int,
                          num_mb: int, grad_accum_steps: int) -> Tuple[Optional[torch.Tensor], None, Optional[Dict]]:
        """
        Compute regu_c * || W_old[mb_slice] + A @ B ||_F^2, register hooks
        on loradict's A/B/C tensors to inject regu gradients during the
        subsequent CE/distill backward pass, and return (unscaled_sq_norm, None, precomputed).

        Hook-based approach:
          1. Compute new_wdict = W_old_slice + A@B (with grad on A, B, C)
          2. Compute sq_norm = ||new_wdict||_F^2
          3. Use torch.autograd.grad to get d(scaled_sq_norm)/d(A,B,C)
          4. Register one-shot hooks on A, B, C that add these grads
          5. When CE/distill backward flows through A, B, C, the hooks fire
             and inject the regu gradients — no separate backward() needed.

        This avoids calling regu_loss.backward(retain_graph=True), which
        would trigger PipelineRecv.backward and cause deadlocks in PP mode.

        Args:
            loradict: The loradict from hypernetwork (with grad on A, B).
            mb_idx: The micro-batch index (0-based).
            num_mb: Total number of micro-batches.
            grad_accum_steps: Gradient accumulation steps for loss scaling.

        Returns:
            (unscaled_sq_norm, None, precomputed_wdict_slice):
                - unscaled_sq_norm: Scalar float (not tensor) of ||W_old + A@B||²,
                  or None if not computed. Used for logging only.
                - regu_loss_tensor: Always None (PP uses hook-based approach).
                - precomputed_wdict_slice: Detached new wdict slice, or None.
                  Can be passed to write(precomputed_wdict=...) to skip recomputation.
        """
        regu_c = self._cfg.get("regu_c", 0.0)
        if loradict is None:
            return None, None, None

        start = mb_idx * self._micro_batch_size
        end = start + self._micro_batch_size

        # Compute W_old_slice + A @ B (with grad on A, B)
        # When wdict is None (not yet initialized), treat W_old as zero → new_W = A@B
        if self._wdict is not None:
            new_wdict_slice = self._compute_new_wdict_slice(self._wdict, loradict, start, end)
        else:
            new_wdict_slice = self._compute_new_wdict_slice_zero(loradict)
        if new_wdict_slice is None:
            return None, None, None

        # Detach new_wdict_slice for write reuse (do this before potential grad computation)
        precomputed = self._detach_wdict_slice(new_wdict_slice)

        # Compute loss and register hooks if regu_c > 0
        unscaled_sq_norm = None
        if regu_c > 0:
            sq_norm = self._compute_sq_norm(new_wdict_slice)
            if sq_norm is not None:
                unscaled_sq_norm = sq_norm.item()
                scaled_loss = regu_c * sq_norm / (num_mb * grad_accum_steps)

                # Collect all leaf-level A, B, C tensors from loradict that require grad
                leaf_tensors = self._collect_grad_tensors(loradict)

                if leaf_tensors:
                    # Compute gradients of scaled_loss w.r.t. loradict tensors
                    # This does NOT trigger PipelineRecv.backward because we only
                    # differentiate to the loradict level (A, B, C), not further back.
                    grads = torch.autograd.grad(
                        scaled_loss, leaf_tensors,
                        retain_graph=True,  # Keep graph for CE/distill backward
                        allow_unused=True,
                    )

                    # Register one-shot hooks on each tensor to inject regu grads
                    for tensor, grad in zip(leaf_tensors, grads):
                        if grad is not None:
                            self._register_oneshot_hook(tensor, grad)

        return unscaled_sq_norm, None, precomputed

    @staticmethod
    def _collect_grad_tensors(loradict: Dict) -> List[torch.Tensor]:
        """Collect all A, B, C tensors from loradict that require grad."""
        tensors = []
        if loradict is None:
            return tensors

        # Leaf check
        if "A" in loradict and "B" in loradict:
            A = loradict["A"]
            B = loradict["B"]
            if A.requires_grad:
                tensors.append(A)
            if B.requires_grad:
                tensors.append(B)
            C = loradict.get("C", None)
            if C is not None and C.requires_grad:
                tensors.append(C)
            return tensors

        # Recurse
        for key, value in loradict.items():
            if isinstance(value, dict):
                tensors.extend(FullDetachState._collect_grad_tensors(value))
        return tensors

    @staticmethod
    def _register_oneshot_hook(tensor: torch.Tensor, regu_grad: torch.Tensor) -> None:
        """Register a hook that adds regu_grad to the tensor's gradient once, then removes itself."""
        handle = None

        def hook_fn(grad):
            nonlocal handle
            # Remove self after first invocation to avoid double-counting
            # (Phase C' backward and Phase C backward both flow through loradict)
            if handle is not None:
                handle.remove()
                handle = None
            return grad + regu_grad

        handle = tensor.register_hook(hook_fn)

    def set_last_sq_norms(self, sq_norms: List[float]) -> None:
        """Store per-sample squared norms (after node-level all_reduce SUM).

        These represent the full (all-stage) ||W_old + A@B||² for each
        sample position. Used by maybe_reset_slice() before the next read.
        """
        self._last_sq_norms = list(sq_norms)

    def maybe_reset_slice(self, sample_idx: int) -> bool:
        """Check threshold-based reset for a single sample's wdict slice.

        Called at the END of each step (after write, set_last_sq_norms,
        and step increment).

        IMPORTANT: The caller must increment _update_steps BEFORE calling
        this method, and read get_reset_stats() between the increment and
        this call. This ensures mean_update_step >= 1 at recording time.

        If reset: the sample's slice is zeroed, _update_steps[sample_idx] set to 0.
        If no reset: no change to _update_steps.

        Args:
            sample_idx: The sample index (0-based) within local_batch_size.

        Returns:
            True if the slice was reset, False otherwise.
        """
        # Dynamically extend _update_steps if sample_idx exceeds current size
        # (can happen during evaluation with different batch sizes)
        while sample_idx >= len(self._update_steps):
            self._update_steps.append(0)

        reset_threshold = self._cfg.get("reset_threshold", None)
        if reset_threshold is None:
            return False

        # Determine whether this sample should be reset
        should_reset = False
        if reset_threshold <= 0:
            # Always reset every step
            should_reset = True
        elif self._last_sq_norms is not None and sample_idx < len(self._last_sq_norms):
            should_reset = self._last_sq_norms[sample_idx] > reset_threshold

        if should_reset:
            # Zero out wdict slice if it exists
            if self._wdict is not None:
                self._zero_wdict_slice(self._wdict, sample_idx, sample_idx + 1)
            # Always reset counter (even when _wdict is None)
            self._update_steps[sample_idx] = 0
            return True
        else:
            return False

    def get_reset_stats(self) -> Tuple[float, float]:
        """Return (reset_ratio, mean_update_step) for this node.

        reset_ratio: fraction of sample positions whose sq_norm exceeds threshold
                     (i.e., will be reset at the start of the NEXT step).
                     Computed from _last_sq_norms and reset_threshold.
        mean_update_step: average of _update_steps across all sample positions.
                          Represents how many steps each sample's wdict slice has been
                          updated since its last reset (including current step).

        Returns:
            (reset_ratio, mean_update_step)
        """
        reset_threshold = self._cfg.get("reset_threshold", None)
        num_samples = self._local_batch_size

        # Compute reset_ratio from current sq_norms
        if reset_threshold is None or num_samples == 0:
            reset_ratio = 0.0
        elif reset_threshold <= 0:
            # Always reset every step
            reset_ratio = 1.0
        elif self._last_sq_norms is None:
            reset_ratio = 0.0
        else:
            num_resets = sum(
                1 for sq in self._last_sq_norms[:num_samples] if sq > reset_threshold
            )
            reset_ratio = num_resets / num_samples

        # Compute mean_update_step
        if num_samples == 0:
            mean_update_step = 0.0
        else:
            mean_update_step = sum(self._update_steps[:num_samples]) / num_samples

        return reset_ratio, mean_update_step

    def init_steps(self) -> None:
        """Reset _update_steps to initial state (all sample positions to 0)."""
        self._update_steps = [0] * self._local_batch_size

    def reset_slice(self, sample_idx: int) -> None:
        """Reset the wdict slice and step counter for a single sample.

        Zeros only the wdict slice at the given sample position (batch dim
        index) and resets the corresponding step counter to 0.

        Args:
            sample_idx: The sample index (0-based) within local_batch_size.
        """
        if self._wdict is not None:
            self._zero_wdict_slice(self._wdict, sample_idx, sample_idx + 1)
        while sample_idx >= len(self._update_steps):
            self._update_steps.append(0)
        self._update_steps[sample_idx] = 0

    def update_steps(self, sample_idx: int) -> None:
        """Increment the update step counter for the given sample index."""
        while sample_idx >= len(self._update_steps):
            self._update_steps.append(0)
        self._update_steps[sample_idx] += 1

    def _zero_wdict_slice(self, wdict: Dict, start: int, end: int) -> None:
        """Zero out a specific batch slice of wdict."""
        if wdict is None:
            return
        if "W" in wdict:
            wdict["W"][start:end].zero_()
            if wdict.get("C") is not None:
                wdict["C"][start:end].zero_()
            return
        for key, value in wdict.items():
            if isinstance(value, dict):
                self._zero_wdict_slice(value, start, end)

    def _compute_new_wdict_slice(self, wdict: Dict, loradict: Dict,
                                  start: int, end: int) -> Optional[Dict]:
        """
        Compute W_old[start:end] + A @ B recursively.

        Returns a new dict with the same structure as a wdict slice, where
        leaf["W"] = W_old_slice + A @ B (has grad from A, B).
        leaf["C"] = C_old_slice + C (has grad from C) if C exists.
        """
        if wdict is None or loradict is None:
            return None

        # wdict leaf
        if "W" in wdict:
            if "A" not in loradict or "B" not in loradict:
                return None
            W_slice = wdict["W"][start:end]  # [mbs, in, out], detached (no grad)
            A = loradict["A"]  # [mbs, in, r], has grad
            B = loradict["B"]  # [mbs, r, out], has grad
            AB = torch.bmm(A, B)  # [mbs, in, out], has grad
            new_W = W_slice + AB  # W_slice is detached, result has grad from AB

            C_lora = loradict.get("C", None)
            new_C = None
            if C_lora is not None and wdict.get("C") is not None:
                C_slice = wdict["C"][start:end]  # [mbs, out], detached
                new_C = C_slice + C_lora  # has grad from C_lora
            elif wdict.get("C") is not None:
                new_C = wdict["C"][start:end]  # detached, no change

            return {"W": new_W, "C": new_C}

        # Recurse into sub-dicts
        result = {}
        for key in wdict:
            if wdict[key] is None:
                result[key] = None
            elif isinstance(wdict[key], dict):
                if key in loradict and isinstance(loradict[key], dict):
                    result[key] = self._compute_new_wdict_slice(
                        wdict[key], loradict[key], start, end
                    )
                else:
                    # No corresponding loradict entry — just slice the old wdict
                    result[key] = self._slice_wdict(wdict[key], start, end)
            else:
                result[key] = wdict[key]
        return result

    def _compute_new_wdict_slice_zero(self, loradict: Dict) -> Optional[Dict]:
        """
        Compute A @ B recursively (treating W_old as zero).

        Used when wdict is not yet initialized. Returns the same structure
        as _compute_new_wdict_slice but without adding W_old.

        leaf["W"] = A @ B (has grad from A, B).
        leaf["C"] = C (has grad from C) if C exists, else None.
        """
        if loradict is None:
            return None

        # loradict leaf
        if "A" in loradict and "B" in loradict:
            A = loradict["A"]  # [mbs, in, r], has grad
            B = loradict["B"]  # [mbs, r, out], has grad
            new_W = torch.bmm(A, B)  # [mbs, in, out], has grad

            C_lora = loradict.get("C", None)
            new_C = C_lora  # has grad from C_lora, or None

            return {"W": new_W, "C": new_C}

        # Recurse into sub-dicts
        result = {}
        for key, value in loradict.items():
            if value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._compute_new_wdict_slice_zero(value)
            else:
                result[key] = value
        return result

    def _compute_sq_norm(self, wdict_slice: Optional[Dict]) -> Optional[torch.Tensor]:
        """Compute sum of squared Frobenius norms over all leaves."""
        if wdict_slice is None:
            return None

        # Leaf
        if "W" in wdict_slice:
            W = wdict_slice["W"]
            sq = (W * W).sum()
            C = wdict_slice.get("C", None)
            if C is not None:
                sq = sq + (C * C).sum()
            return sq

        # Recurse
        total = None
        for key, value in wdict_slice.items():
            if value is None or not isinstance(value, dict):
                continue
            part = self._compute_sq_norm(value)
            if part is not None:
                total = part if total is None else total + part
        return total

    def _detach_wdict_slice(self, wdict_slice: Optional[Dict]) -> Optional[Dict]:
        """Detach all tensors in a wdict slice (for write reuse)."""
        if wdict_slice is None:
            return None

        # Leaf
        if "W" in wdict_slice:
            W_det = wdict_slice["W"].detach()
            C_det = wdict_slice["C"].detach() if wdict_slice.get("C") is not None else None
            return {"W": W_det, "C": C_det}

        # Recurse
        result = {}
        for key, value in wdict_slice.items():
            if value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._detach_wdict_slice(value)
            else:
                result[key] = value
        return result

    def _copy_wdict_slice(self, wdict: Dict, src_slice: Dict, batch_start: int) -> None:
        """
        Copy src_slice into wdict at batch_start position.

        src_slice has batch dim = micro_batch_size.
        wdict has batch dim = local_batch_size.
        """
        if wdict is None or src_slice is None:
            return

        # Leaf
        if "W" in wdict:
            if "W" in src_slice:
                Lb = src_slice["W"].shape[0]
                wdict["W"][batch_start:batch_start + Lb].copy_(src_slice["W"])
                if src_slice.get("C") is not None and wdict.get("C") is not None:
                    wdict["C"][batch_start:batch_start + Lb].copy_(src_slice["C"])
            return

        # Recurse
        for key in wdict:
            if wdict[key] is None or key not in src_slice or src_slice[key] is None:
                continue
            if isinstance(wdict[key], dict) and isinstance(src_slice[key], dict):
                self._copy_wdict_slice(wdict[key], src_slice[key], batch_start)

    def _create_zero_wdict_from_wdict(self, wdict_slice: Dict) -> Optional[Dict]:
        """
        Create a zero-initialized wdict with batch dim = local_batch_size,
        using a wdict slice as reference for dimensions.
        """
        if wdict_slice is None:
            return None

        # Leaf
        if "W" in wdict_slice:
            W = wdict_slice["W"]  # [mbs, in, out]
            _, in_dim, out_dim = W.shape
            device = W.device
            dtype = W.dtype
            W_zero = torch.zeros(
                self._local_batch_size, in_dim, out_dim,
                device=device, dtype=dtype, requires_grad=False,
            )
            C_zero = None
            if wdict_slice.get("C") is not None:
                C_zero = torch.zeros(
                    self._local_batch_size, out_dim,
                    device=device, dtype=dtype, requires_grad=False,
                )
            return {"W": W_zero, "C": C_zero}

        # Recurse
        result = {}
        for key, value in wdict_slice.items():
            if value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._create_zero_wdict_from_wdict(value)
            else:
                result[key] = value
        return result
