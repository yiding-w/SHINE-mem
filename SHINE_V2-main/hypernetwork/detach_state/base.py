"""
Base class for all DetachState implementations.

DetachState maintains a persistent, detached (no-grad) state that accumulates
the effect of generated LoRA dicts over time. Bound 1:1 with a
ModelHypernetwork (PP or TP).

All tensors created and stored by DetachState subclasses must have
requires_grad=False. The write() method enforces this by detaching the
input loradict and wrapping _write_impl in torch.no_grad().
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, Dict, List, Tuple, Any

import torch

from utils.myloradict import detach_loradict

logger = logging.getLogger(__name__)


class BaseDetachState(ABC):
    """
    Abstract base class for DetachState implementations.

    Subclasses must implement: read, _write_impl, reset, state_dict, load_state_dict.

    The internal storage consists of:
      - A detached nograd_loradict (same structure as a normal loradict, but
        all tensors are detached with no gradient). Always full/unsharded —
        TP linears slice it on-the-fly inside their forward.
      - A detached nograd_wdict (same structure as a wdict: leaf is
        {"W": [Lb, in, out], "C": [Lb, out] | None}).

    At most one of these is non-None at any time. For EmptyDetachState,
    both are always None.

    All tensors stored internally MUST have requires_grad=False.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: Configuration dict/DictConfig. Subclasses read their own
                 specific parameters from this.
        """
        self._cfg = cfg

    @abstractmethod
    def read(self, mb_idx: Optional[int] = None) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Read / export current state for use in the forward pass.

        Args:
            mb_idx: Optional micro-batch index. If provided, returns only
                    the slice corresponding to that micro-batch. If None,
                    returns the full state.

        Returns:
            (nograd_loradict, nograd_wdict) — at most one is non-None.
            Both are None for EmptyDetachState.
            nograd_loradict: dict mapping layer_idx → leaf loradict (detached).
            nograd_wdict: dict mapping layer_idx → leaf wdict (detached).
        """
        ...

    def write(self, loradict: Optional[Dict], mb_idx: Optional[int] = None,
              precomputed_wdict: Optional[Dict] = None) -> None:
        """Write / update internal state with a newly generated loradict.

        This method forcibly detaches the loradict before passing it to the
        subclass implementation, ensuring no gradient leaks into the stored
        state regardless of what the caller passes in. All operations in
        _write_impl run under torch.no_grad().

        If precomputed_wdict is provided (e.g., from compute_regu_loss), the
        subclass can use it directly to avoid recomputing W_old + A@B.
        In this case, loradict detach is skipped as an optimization since
        _write_impl will use precomputed_wdict directly.

        Args:
            loradict: The loradict produced by the hypernetwork in this step.
                      May be None (e.g., on PP stages that don't own layers).
            mb_idx: Optional micro-batch index. If provided, writes to the
                    specific batch position. Required for FullDetachState.
            precomputed_wdict: Optional precomputed new wdict slice (already
                    detached, batch dim = micro_batch_size). If provided,
                    _write_impl can skip the A@B computation and directly
                    copy this into the appropriate batch position.
        """
        # Skip expensive detach+clone when precomputed_wdict is provided,
        # since _write_impl will use precomputed_wdict directly.
        detached = None if precomputed_wdict is not None else detach_loradict(loradict)
        with torch.no_grad():
            self._write_impl(detached, mb_idx=mb_idx, precomputed_wdict=precomputed_wdict)

    @abstractmethod
    def _write_impl(self, loradict: Optional[Dict], mb_idx: Optional[int] = None,
                    precomputed_wdict: Optional[Dict] = None) -> None:
        """Subclass-specific write logic. Called with an already-detached loradict.

        All operations here run under torch.no_grad() (enforced by write()).

        Args:
            loradict: The detached loradict (all tensors have requires_grad=False
                      and no grad_fn). May be None.
            mb_idx: Optional micro-batch index for batch-indexed writes.
            precomputed_wdict: Optional precomputed new wdict slice. If provided,
                    the implementation can skip A@B and directly use this.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state to zero / empty."""
        ...

    @abstractmethod
    def state_dict(self) -> Dict:
        """Serialize internal state for checkpointing."""
        ...

    @abstractmethod
    def load_state_dict(self, state: Dict) -> None:
        """Restore internal state from a checkpoint.

        Args:
            state: The dict returned by a previous call to state_dict().
        """
        ...

    @abstractmethod
    def compute_regu_loss(self, loradict: Optional[Dict], mb_idx: int,
                          num_mb: int, grad_accum_steps: int) -> Tuple[Optional[Any], Optional[Any], Optional[Dict]]:
        """Compute weight regularization loss.

        Computes: regu_c * || W_old[mb_slice] + A @ B ||_F^2
        where W_old is detached (from internal state) and A, B are from loradict
        (with grad).

        Implementation strategies differ by subclass:
          - PP (FullDetachState): Uses hooks to inject regu gradients during
            the subsequent CE/distill backward. Returns regu_loss_tensor=None.
          - TP (FullTPDetachState): Returns a differentiable regu_loss_tensor
            that the caller adds to the main loss before backward.

        Returns the precomputed new_wdict_slice (detached) so that the caller
        can pass it to write(precomputed_wdict=...) to avoid redundant A@B.

        Args:
            loradict: The loradict from hypernetwork (with grad on A, B).
                      May be None.
            mb_idx: The micro-batch index (0-based).
            num_mb: Total number of micro-batches.
            grad_accum_steps: Gradient accumulation steps for loss scaling.

        Returns:
            (unscaled_sq_norm, regu_loss_tensor, precomputed_wdict_slice):
                - unscaled_sq_norm: Float value of ||W_old + A@B||² (unscaled),
                  or None if disabled/not applicable. Used for logging only.
                - regu_loss_tensor: Differentiable scalar loss tensor, or None
                  if using hook-based approach (PP) or if regu is disabled.
                - precomputed_wdict_slice: Detached wdict slice (batch dim = micro_batch_size)
                  representing W_old + A@B, or None if not computed.
                  Can be passed to write(precomputed_wdict=...) to skip recomputation.
        """
        ...

    @abstractmethod
    def set_last_sq_norms(self, sq_norms: List[float]) -> None:
        """Store per-sample squared norms (after node-level all_reduce SUM).

        These norms represent the full (all-stage/all-rank) ||W_old + A@B||² for each
        sample position. They are used by maybe_reset_slice() at the
        end of the step to decide whether to reset specific slices.

        Args:
            sq_norms: List of floats, one per sample in local_batch_size.
                      Each value is the full ||W_new[sample]||² after all_reduce.
        """
        ...

    @abstractmethod
    def maybe_reset_slice(self, sample_idx: int) -> bool:
        """Check if the wdict slice for sample_idx should be reset based on threshold.

        Called at the END of each step (after write and set_last_sq_norms).
        Behavior depends on reset_threshold config value:
          - None: never reset (return False).
          - <= 0: always reset every step unconditionally.
          - > 0: reset only if the stored sq_norm exceeds reset_threshold.

        Also maintains _update_steps counter:
          - On reset: set to 0 (nothing accumulated in wdict after reset).
          - On no reset: no change.

        Args:
            sample_idx: The sample index (0-based) within local_batch_size.

        Returns:
            True if the slice was reset, False otherwise.
        """
        ...

    @abstractmethod
    def get_reset_stats(self) -> Tuple[float, float]:
        """Return (reset_ratio, mean_update_step) for this node.

        reset_ratio: fraction of sample positions whose sq_norm exceeds threshold
                     (will be reset at the start of the NEXT step). Range [0, 1].
        mean_update_step: average number of steps each sample's wdict slice has been
                          updated since its last reset (including current step).

        Returns:
            (reset_ratio, mean_update_step)
        """
        ...

    @abstractmethod
    def init_steps(self) -> None:
        """Reset the _update_steps counter to initial state.

        Called when the detach_state is reset (e.g., on repo change or skip step).
        Subclasses must implement this to reset their step counters appropriately.
        """
        ...

    @abstractmethod
    def reset_slice(self, sample_idx: int) -> None:
        """Reset the wdict slice and step counter for a single sample.

        Called on repo-change to reset a single sample position without
        affecting other positions. Zeros the wdict slice and resets the
        corresponding step counter to 0.

        Args:
            sample_idx: The sample index (0-based) within local_batch_size.
        """
        ...

    @abstractmethod
    def update_steps(self, sample_idx: int) -> None:
        """Increment the update step counter for the given sample index.

        Called after each successful training step to track how many steps
        have been accumulated since the last reset.

        Args:
            sample_idx: The sample index (0-based) within local_batch_size.
        """
        ...

    @contextmanager
    def eval_context(self, eval_local_batch_size: int = 0):
        """Context manager for evaluation: provides a fresh zero-initialized
        state independent of training state.

        On entry: saves training state and replaces with fresh zero state.
        On exit: restores training state and releases eval state.

        Args:
            eval_local_batch_size: The local batch size used during evaluation.
                If 0 or not provided, uses the training local_batch_size.

        Default implementation is a no-op (for EmptyDetachState).
        Subclasses with actual state (FullDetachState) should override.
        """
        yield
