"""
FullTPDetachState: maintains a TP-sharded wdict as persistent state (TP mode).

In TP mode, each rank stores only its own shard of the wdict:
  - Colwise projections (q, k, v, gate, up): W = [local_batch_size, in, out/W]
  - Rowwise projections (o, down):           W = [local_batch_size, in/W, out]

The loradict arrives full/unsharded from the hypernetwork. During write,
we compute A @ B and slice the result to the local TP shard before storing.
More efficiently, we compute only the local shard directly:
  - Colwise: W_local += A @ B_local  (B_local = B[:, :, s:e])
  - Rowwise: W_local += A_local @ B  (A_local = A[:, s:e, :])

During read, the stored wdict is already pre-sliced and can be passed
directly to the TP linear layers (which expect pre-sliced nograd_wdict).

Step counters and reset are managed at per-sample granularity:
  - _update_steps has length = local_batch_size (one counter per sample)
  - reset_slice(sample_idx) zeros only that single sample's wdict position

All tensors created and stored internally have requires_grad=False.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.distributed as dist

from hypernetwork.detach_state.base import BaseDetachState

logger = logging.getLogger(__name__)

# TP sharding plan: which projections are Colwise vs Rowwise.
# Colwise: output dim is sharded (q_query, q_gate, k, v, gate, up)
# Rowwise: input dim is sharded (o, down)
_COLWISE_PROJS = frozenset({"q_query", "q_gate", "k", "v", "gate", "up"})
_ROWWISE_PROJS = frozenset({"o", "down"})


class FullTPDetachState(BaseDetachState):
    """
    Full DetachState for TP mode: maintains a TP-sharded wdict.

    Each TP rank stores only its local shard of W (and C). The sharding
    follows the same convention as tp_linear.py:
      - Colwise: W[:, :, rank*out_local:(rank+1)*out_local]
      - Rowwise: W[:, rank*in_local:(rank+1)*in_local, :]

    The wdict batch dimension = local_batch_size. Step counters and reset
    are managed at per-sample granularity (one counter per sample in the batch).
    """

    def __init__(self, cfg, *, local_batch_size: int, micro_batch_size: int,
                 parallel_mode: str, tp_rank: int, tp_world: int,
                 tp_process_group, num_llm_layers: int,
                 data_parallel_size: int = 1):
        super().__init__(cfg)
        assert parallel_mode == "tp", f"FullTPDetachState requires mode='tp', got '{parallel_mode}'"

        self._local_batch_size = local_batch_size
        self._micro_batch_size = micro_batch_size
        self._tp_rank = tp_rank
        self._tp_world = tp_world
        self._tp_group = tp_process_group
        self._num_llm_layers = num_llm_layers

        self._wdict = None  # Lazy init on first write
        self._last_sq_norms = None  # Sq norms from previous step (after TP all_reduce)
        # Per-sample update step counters: one counter per sample in local_batch_size
        self._update_steps = [0] * local_batch_size

        # Frozen config for checkpoint validation
        self._frozen_cfg = {
            "type": "full_tp",
            "parallel_mode": parallel_mode,
            "local_batch_size": local_batch_size,
            "micro_batch_size": micro_batch_size,
            "tp_rank": tp_rank,
            "tp_world": tp_world,
            "num_llm_layers": num_llm_layers,
            "data_parallel_size": data_parallel_size,
            # "wdict_shapes" added lazily on first write
        }

        logger.info(
            f"[DetachState] Created FullTPDetachState "
            f"(local_batch_size={local_batch_size}, "
            f"tp_rank={tp_rank}/{tp_world}, "
            f"num_llm_layers={num_llm_layers})"
        )

    # ------------------------------------------------------------------
    # Read interface
    # ------------------------------------------------------------------

    def read(self, mb_idx: Optional[int] = None) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Return (None, wdict) or (None, wdict_slice).

        In TP mode, the stored wdict is already pre-sliced to the local
        TP shard. If mb_idx is provided, returns the slice for that
        micro-batch position along the batch dimension.

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
        Accumulate loradict into the TP-sharded wdict.

        If precomputed_wdict is provided (from compute_regu_loss), directly
        copies it into the wdict (already TP-sharded).
        Otherwise, computes the local shard of A@B and accumulates:
          - Colwise: W_local += A @ B_local
          - Rowwise: W_local += A_local @ B

        Args:
            loradict: The detached loradict from hypernetwork.
                      Structure: {layer_idx: {group: {proj: {"A":.., "B":.., "C":..}}}}
                      Leaf tensors are full/unsharded.
            mb_idx: Ignored in TP mode.
            precomputed_wdict: Optional precomputed TP-sharded wdict.
                      If provided, directly copies into the stored wdict.
        """
        if loradict is None and precomputed_wdict is None:
            return

        # Lazy init: create zero wdict on first write
        if self._wdict is None:
            ref = loradict if loradict is not None else precomputed_wdict
            if ref is None:
                return
            if loradict is not None:
                self._wdict = self._create_zero_wdict_from_loradict(loradict)
            else:
                self._wdict = self._create_zero_wdict_from_wdict(precomputed_wdict)
            self._frozen_cfg["wdict_shapes"] = self._extract_wdict_shapes(self._wdict)

        if precomputed_wdict is not None:
            self._copy_wdict(self._wdict, precomputed_wdict)
        else:
            self._tp_accumulate(self._wdict, loradict)

    # ------------------------------------------------------------------
    # Reset / state management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset wdict to zeros (if initialized)."""
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
        self._last_sq_norms = None
        if self._wdict is not None:
            self._ensure_no_grad(self._wdict)
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
        """Validate that saved config matches current config exactly."""
        mismatches = []
        for key, current_val in self._frozen_cfg.items():
            if key not in saved_cfg:
                continue
            saved_val = saved_cfg[key]
            if saved_val != current_val:
                mismatches.append(
                    f"  '{key}': checkpoint={saved_val}, current={current_val}"
                )
        if mismatches:
            raise ValueError(
                f"FullTPDetachState config mismatch — cannot resume:\n"
                + "\n".join(mismatches)
                + "\n\nThe wdict was saved with different TP/batch settings. "
                + "Either use the same settings or start fresh (delete the checkpoint)."
            )

    # ------------------------------------------------------------------
    # Evaluation context manager
    # ------------------------------------------------------------------

    @contextmanager
    def eval_context(self, eval_local_batch_size: int = 0):
        """Context manager for evaluation: provides a fresh zero-initialized
        wdict independent of training state.

        On entry: saves training state and replaces with fresh zero state.
        On exit: restores training state and releases eval state.

        Args:
            eval_local_batch_size: The local batch size used during evaluation.
                If 0 or not provided, uses the training local_batch_size.
        """
        train_wdict = self._wdict
        train_last_sq_norms = self._last_sq_norms
        train_update_steps = self._update_steps
        train_local_batch_size = self._local_batch_size

        # Determine eval batch size
        eval_bs = eval_local_batch_size if eval_local_batch_size > 0 else train_local_batch_size

        # Replace with fresh eval state
        self._wdict = None
        self._last_sq_norms = None
        self._local_batch_size = eval_bs
        self._update_steps = [0] * eval_bs

        try:
            yield
        finally:
            del self._wdict
            self._wdict = train_wdict
            self._last_sq_norms = train_last_sq_norms
            self._update_steps = train_update_steps
            self._local_batch_size = train_local_batch_size

    # ------------------------------------------------------------------
    # Regularization loss
    # ------------------------------------------------------------------

    def compute_regu_loss(self, loradict: Optional[Dict], mb_idx: int,
                          num_mb: int, grad_accum_steps: int) -> Tuple[Optional[Any], Optional[Any], Optional[Dict]]:
        """
        Compute regu_c * || W_old + A @ B ||_F^2 and return as a differentiable
        loss tensor that can be added to the main loss (instead of using hooks).

        This avoids the inplace modification issue: the caller adds regu_loss
        to the main loss, then does backward (which frees the graph), then
        calls _write_detach_state (which does inplace copy_ on self._wdict).

        The sq_norm is computed locally on this rank's shard. The caller
        should all_reduce the unscaled_sq_norm across the TP group for
        logging/threshold purposes (done via set_last_sq_norms).

        Args:
            loradict: The loradict from hypernetwork (with grad on A, B).
            mb_idx: Ignored in TP mode (always 0).
            num_mb: Always 1 in TP mode.
            grad_accum_steps: Gradient accumulation steps for loss scaling.

        Returns:
            (unscaled_sq_norm, regu_loss_tensor, precomputed_wdict):
                - unscaled_sq_norm: Local sq_norm float (this rank's shard only).
                  Caller should all_reduce across TP group for full norm.
                - regu_loss_tensor: Differentiable scalar loss tensor, or None
                  if regu_c == 0 or loradict is None.
                - precomputed_wdict: Detached TP-sharded new wdict, or None.
                  Can be passed to write(precomputed_wdict=...) to skip recomputation.
        """
        regu_c = self._cfg.get("regu_c", 0.0)
        if loradict is None:
            return None, None, None

        # Compute new_wdict = W_old + shard(A @ B) with grad on A, B
        if self._wdict is not None:
            new_wdict = self._compute_new_wdict(self._wdict, loradict)
        else:
            new_wdict = self._compute_new_wdict_zero(loradict)
        if new_wdict is None:
            return None, None, None

        # Detach for write reuse (shares storage with new_wdict tensors but
        # detached from graph — safe to use after backward frees the graph)
        precomputed = self._detach_wdict(new_wdict)

        # Compute loss if regu_c > 0
        # NOTE: Do NOT divide by grad_accum_steps here. The caller (_train_step)
        # already divides the entire backward_loss (CE + regu) by grad_accum_steps.
        # Dividing here would double-scale the regu gradient.
        # Only divide by num_mb to average across micro-batches within a single
        # forward call (always 1 in TP mode).
        unscaled_sq_norm = None
        regu_loss_tensor = None
        if regu_c > 0:
            sq_norm = self._compute_sq_norm(new_wdict)
            if sq_norm is not None:
                unscaled_sq_norm = sq_norm.detach().item()
                regu_loss_tensor = regu_c * sq_norm / num_mb

        return unscaled_sq_norm, regu_loss_tensor, precomputed

    # ------------------------------------------------------------------
    # Threshold-based reset
    # ------------------------------------------------------------------

    def set_last_sq_norms(self, sq_norms: List[float]) -> None:
        """Store per-sample squared norms (after TP all_reduce SUM).

        sq_norms should have length = local_batch_size (one per sample).
        Each value is the full (all-rank) ||W_old + A@B||² for that sample.
        """
        self._last_sq_norms = list(sq_norms)

    def maybe_reset_slice(self, sample_idx: int) -> bool:
        """Check threshold-based reset for a single sample's wdict slice.

        Called at the END of each step (after write, set_last_sq_norms,
        and step increment).

        IMPORTANT: The caller must increment _update_steps BEFORE calling
        this method, and read get_reset_stats() between the increment and
        this call. This ensures mean_update_step >= 1 at recording time.

        If reset: the sample's wdict slice is zeroed, _update_steps set to 0.
        If no reset: no change to _update_steps.

        Args:
            sample_idx: The sample index (0-based) within local_batch_size.

        Returns:
            True if the slice was reset, False otherwise.
        """
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
        """Return (reset_ratio, mean_update_step).

        reset_ratio: fraction of sample positions whose sq_norm exceeds threshold.
        mean_update_step: average of _update_steps across all sample positions.
        """
        reset_threshold = self._cfg.get("reset_threshold", None)
        num_samples = self._local_batch_size

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

    # ------------------------------------------------------------------
    # TP-aware accumulation
    # ------------------------------------------------------------------

    def _tp_accumulate(self, wdict: Dict, loradict: Dict) -> None:
        """
        Accumulate loradict's A@B into the TP-sharded wdict.

        For Colwise projections: W_local += A @ B_local
            where B_local = B[:, :, rank*out_local:(rank+1)*out_local]
        For Rowwise projections: W_local += A_local @ B
            where A_local = A[:, rank*in_local:(rank+1)*in_local, :]

        The recursion follows the wdict structure:
            wdict[layer_idx] → {"attention": {...}, "mlp": {...}}
            attention → {"q_query": leaf, "q_gate": leaf, "k": leaf, "v": leaf, "o": leaf}
            mlp → {"gate": leaf, "up": leaf, "down": leaf}
        """
        if wdict is None or loradict is None:
            return
        self._tp_accumulate_recursive(wdict, loradict, path_keys=[])

    def _tp_accumulate_recursive(self, wdict: Dict, loradict: Dict,
                                  path_keys: list) -> None:
        """Recursively accumulate, tracking path to determine Colwise/Rowwise."""
        if wdict is None or loradict is None:
            return

        # Leaf: wdict has "W" key
        if "W" in wdict:
            if "A" not in loradict or "B" not in loradict:
                return
            # Determine sharding type from the last path key (projection name)
            proj_name = path_keys[-1] if path_keys else ""
            A = loradict["A"]  # [Lb, in_full, r]
            B = loradict["B"]  # [Lb, r, out_full]
            C = loradict.get("C", None)

            if proj_name in _COLWISE_PROJS:
                # Colwise: slice B on output dim, compute A @ B_local
                out_full = B.shape[2]
                out_local = out_full // self._tp_world
                s = self._tp_rank * out_local
                e = s + out_local
                B_local = B[:, :, s:e]  # [Lb, r, out_local]
                # W_local += A @ B_local
                W_local = wdict["W"]  # [Lb, in, out_local]
                torch.baddbmm(W_local, A, B_local, beta=1, alpha=1, out=W_local)
                # C_local += C_local_slice
                if C is not None and wdict["C"] is not None:
                    wdict["C"].add_(C[:, s:e])
            elif proj_name in _ROWWISE_PROJS:
                # Rowwise: slice A on input dim, compute A_local @ B
                in_full = A.shape[1]
                in_local = in_full // self._tp_world
                s = self._tp_rank * in_local
                e = s + in_local
                A_local = A[:, s:e, :]  # [Lb, in_local, r]
                # W_local += A_local @ B
                W_local = wdict["W"]  # [Lb, in_local, out]
                torch.baddbmm(W_local, A_local, B, beta=1, alpha=1, out=W_local)
                # C for Rowwise is replicated (not sharded), add full C
                if C is not None and wdict["C"] is not None:
                    wdict["C"].add_(C)
            else:
                # Unknown projection — fallback: full A @ B, no slicing
                # This shouldn't happen with the standard model architecture
                logger.warning(
                    f"[FullTPDetachState] Unknown projection '{proj_name}' at "
                    f"path {path_keys}. Using full A@B (no TP slicing)."
                )
                torch.baddbmm(wdict["W"], A, B, beta=1, alpha=1, out=wdict["W"])
                if C is not None and wdict["C"] is not None:
                    wdict["C"].add_(C)
            return

        # Recurse into sub-dicts
        for key in wdict:
            if wdict[key] is None or key not in loradict or loradict[key] is None:
                continue
            if isinstance(wdict[key], dict) and isinstance(loradict[key], dict):
                self._tp_accumulate_recursive(
                    wdict[key], loradict[key], path_keys + [key]
                )

    # ------------------------------------------------------------------
    # Zero-init wdict from loradict (TP-sharded)
    # ------------------------------------------------------------------

    def _create_zero_wdict_from_loradict(self, loradict: Dict) -> Dict:
        """
        Create a zero-initialized TP-sharded wdict using loradict as reference.

        loradict leaf: {"A": [Lb, in_full, r], "B": [Lb, r, out_full], "C": ...}
        wdict leaf (Colwise): {"W": [Lb, in_full, out_local], "C": [Lb, out_local] | None}
        wdict leaf (Rowwise): {"W": [Lb, in_local, out_full], "C": [Lb, out_full] | None}
        """
        return self._build_zero_wdict_recursive(loradict, path_keys=[])

    def _build_zero_wdict_recursive(self, loradict, path_keys: list) -> Optional[Dict]:
        if loradict is None:
            return None
        # Leaf check
        if "A" in loradict and "B" in loradict:
            A = loradict["A"]  # [Lb, in_full, r]
            B = loradict["B"]  # [Lb, r, out_full]
            C = loradict.get("C", None)
            Lb = A.shape[0]
            in_full = A.shape[1]
            out_full = B.shape[2]
            device = A.device
            dtype = A.dtype

            proj_name = path_keys[-1] if path_keys else ""

            if proj_name in _COLWISE_PROJS:
                out_local = out_full // self._tp_world
                W_zero = torch.zeros(
                    Lb, in_full, out_local,
                    device=device, dtype=dtype, requires_grad=False,
                )
                C_zero = None
                if C is not None:
                    C_zero = torch.zeros(
                        Lb, out_local,
                        device=device, dtype=dtype, requires_grad=False,
                    )
            elif proj_name in _ROWWISE_PROJS:
                in_local = in_full // self._tp_world
                W_zero = torch.zeros(
                    Lb, in_local, out_full,
                    device=device, dtype=dtype, requires_grad=False,
                )
                C_zero = None
                if C is not None:
                    # Rowwise C is replicated (full out dim)
                    C_zero = torch.zeros(
                        Lb, out_full,
                        device=device, dtype=dtype, requires_grad=False,
                    )
            else:
                # Fallback: full dims
                W_zero = torch.zeros(
                    Lb, in_full, out_full,
                    device=device, dtype=dtype, requires_grad=False,
                )
                C_zero = None
                if C is not None:
                    C_zero = torch.zeros(
                        Lb, out_full,
                        device=device, dtype=dtype, requires_grad=False,
                    )
            return {"W": W_zero, "C": C_zero}

        # Recurse into sub-dicts
        result = {}
        for key, value in loradict.items():
            if value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._build_zero_wdict_recursive(value, path_keys + [key])
            else:
                result[key] = value
        return result

    # ------------------------------------------------------------------
    # Compute new wdict (with grad) for regu_loss
    # ------------------------------------------------------------------

    def _compute_new_wdict(self, wdict: Dict, loradict: Dict) -> Optional[Dict]:
        """
        Compute W_old + shard(A @ B) with grad on A, B.

        Returns a new dict with the same structure as wdict, where each leaf
        has grad flowing through A, B (for regu_loss backward).
        """
        return self._compute_new_wdict_recursive(wdict, loradict, path_keys=[])

    def _compute_new_wdict_recursive(self, wdict: Dict, loradict: Dict,
                                      path_keys: list) -> Optional[Dict]:
        if wdict is None or loradict is None:
            return None

        # Leaf
        if "W" in wdict:
            if "A" not in loradict or "B" not in loradict:
                return None
            proj_name = path_keys[-1] if path_keys else ""
            A = loradict["A"]  # [Lb, in_full, r], has grad
            B = loradict["B"]  # [Lb, r, out_full], has grad
            C = loradict.get("C", None)
            W_old = wdict["W"]  # [Lb, in_local_or_full, out_local_or_full], no grad

            if proj_name in _COLWISE_PROJS:
                out_full = B.shape[2]
                out_local = out_full // self._tp_world
                s = self._tp_rank * out_local
                e = s + out_local
                B_local = B[:, :, s:e]  # [Lb, r, out_local]
                AB_local = torch.bmm(A, B_local)  # [Lb, in_full, out_local], has grad
                new_W = W_old + AB_local  # has grad from AB_local

                new_C = None
                if C is not None and wdict.get("C") is not None:
                    new_C = wdict["C"] + C[:, s:e]  # has grad from C
                elif wdict.get("C") is not None:
                    new_C = wdict["C"]  # no change, detached
            elif proj_name in _ROWWISE_PROJS:
                in_full = A.shape[1]
                in_local = in_full // self._tp_world
                s = self._tp_rank * in_local
                e = s + in_local
                A_local = A[:, s:e, :]  # [Lb, in_local, r], has grad
                AB_local = torch.bmm(A_local, B)  # [Lb, in_local, out_full], has grad
                new_W = W_old + AB_local  # has grad from AB_local

                new_C = None
                if C is not None and wdict.get("C") is not None:
                    new_C = wdict["C"] + C  # Rowwise C is replicated
                elif wdict.get("C") is not None:
                    new_C = wdict["C"]
            else:
                # Fallback
                AB = torch.bmm(A, B)
                new_W = W_old + AB
                new_C = None
                if C is not None and wdict.get("C") is not None:
                    new_C = wdict["C"] + C
                elif wdict.get("C") is not None:
                    new_C = wdict["C"]

            return {"W": new_W, "C": new_C}

        # Recurse
        result = {}
        for key in wdict:
            if wdict[key] is None:
                result[key] = None
            elif isinstance(wdict[key], dict):
                if key in loradict and isinstance(loradict[key], dict):
                    result[key] = self._compute_new_wdict_recursive(
                        wdict[key], loradict[key], path_keys + [key]
                    )
                else:
                    result[key] = wdict[key]  # No loradict entry, keep old
            else:
                result[key] = wdict[key]
        return result

    def _compute_new_wdict_zero(self, loradict: Dict) -> Optional[Dict]:
        """
        Compute shard(A @ B) with grad (treating W_old as zero).
        Used when wdict is not yet initialized.
        """
        return self._compute_new_wdict_zero_recursive(loradict, path_keys=[])

    def _compute_new_wdict_zero_recursive(self, loradict: Dict,
                                           path_keys: list) -> Optional[Dict]:
        if loradict is None:
            return None

        # Leaf
        if "A" in loradict and "B" in loradict:
            proj_name = path_keys[-1] if path_keys else ""
            A = loradict["A"]  # [Lb, in_full, r], has grad
            B = loradict["B"]  # [Lb, r, out_full], has grad
            C = loradict.get("C", None)

            if proj_name in _COLWISE_PROJS:
                out_full = B.shape[2]
                out_local = out_full // self._tp_world
                s = self._tp_rank * out_local
                e = s + out_local
                B_local = B[:, :, s:e]
                new_W = torch.bmm(A, B_local)  # [Lb, in_full, out_local]
                new_C = C[:, s:e] if C is not None else None
            elif proj_name in _ROWWISE_PROJS:
                in_full = A.shape[1]
                in_local = in_full // self._tp_world
                s = self._tp_rank * in_local
                e = s + in_local
                A_local = A[:, s:e, :]
                new_W = torch.bmm(A_local, B)  # [Lb, in_local, out_full]
                new_C = C if C is not None else None  # Rowwise C is replicated
            else:
                new_W = torch.bmm(A, B)
                new_C = C

            return {"W": new_W, "C": new_C}

        # Recurse
        result = {}
        for key, value in loradict.items():
            if value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._compute_new_wdict_zero_recursive(
                    value, path_keys + [key]
                )
            else:
                result[key] = value
        return result

    # ------------------------------------------------------------------
    # Shared utility methods
    # ------------------------------------------------------------------

    def _compute_sq_norm(self, wdict: Optional[Dict]) -> Optional[torch.Tensor]:
        """Compute sum of squared Frobenius norms over all leaves (local shard)."""
        if wdict is None:
            return None

        if "W" in wdict:
            W = wdict["W"]
            sq = (W * W).sum()
            C = wdict.get("C", None)
            if C is not None:
                sq = sq + (C * C).sum()
            return sq

        total = None
        for key, value in wdict.items():
            if value is None or not isinstance(value, dict):
                continue
            part = self._compute_sq_norm(value)
            if part is not None:
                total = part if total is None else total + part
        return total

    def _detach_wdict(self, wdict: Optional[Dict]) -> Optional[Dict]:
        """Detach all tensors in a wdict (for write reuse)."""
        if wdict is None:
            return None

        if "W" in wdict:
            W_det = wdict["W"].detach()
            C_det = wdict["C"].detach() if wdict.get("C") is not None else None
            return {"W": W_det, "C": C_det}

        result = {}
        for key, value in wdict.items():
            if value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._detach_wdict(value)
            else:
                result[key] = value
        return result

    def _copy_wdict(self, dst: Dict, src: Dict) -> None:
        """Copy src wdict into dst wdict (same shape, in-place)."""
        if dst is None or src is None:
            return

        if "W" in dst:
            if "W" in src:
                dst["W"].copy_(src["W"])
                if src.get("C") is not None and dst.get("C") is not None:
                    dst["C"].copy_(src["C"])
            return

        for key in dst:
            if dst[key] is None or key not in src or src[key] is None:
                continue
            if isinstance(dst[key], dict) and isinstance(src[key], dict):
                self._copy_wdict(dst[key], src[key])

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

    def _ensure_no_grad(self, wdict: Dict) -> None:
        """Ensure all tensors in wdict have requires_grad=False."""
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

    def _create_zero_wdict_from_wdict(self, wdict_ref: Dict) -> Optional[Dict]:
        """Create a zero-initialized wdict using an existing wdict as reference."""
        if wdict_ref is None:
            return None

        if "W" in wdict_ref:
            W = wdict_ref["W"]
            W_zero = torch.zeros_like(W, requires_grad=False)
            C_zero = None
            if wdict_ref.get("C") is not None:
                C_zero = torch.zeros_like(wdict_ref["C"], requires_grad=False)
            return {"W": W_zero, "C": C_zero}

        result = {}
        for key, value in wdict_ref.items():
            if value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._create_zero_wdict_from_wdict(value)
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return (
            f"FullTPDetachState(local_batch_size={self._local_batch_size}, "
            f"tp_rank={self._tp_rank}/{self._tp_world}, "
            f"initialized={self._wdict is not None})"
        )
