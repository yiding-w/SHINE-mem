"""
EmptyDetachState: no-op implementation.

Equivalent to the original behavior — detach_state does nothing.
All methods are no-ops or return None.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

from hypernetwork.detach_state.base import BaseDetachState

logger = logging.getLogger(__name__)


class EmptyDetachState(BaseDetachState):
    """No-op DetachState. Equivalent to not using detach_state at all."""

    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info("[DetachState] Created EmptyDetachState (no-op)")

    def read(self, mb_idx: Optional[int] = None) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Always returns (None, None) — no detached state to apply."""
        return None, None

    def _write_impl(self, loradict: Optional[Dict], mb_idx: Optional[int] = None,
                    precomputed_wdict: Optional[Dict] = None) -> None:
        """No-op — nothing to accumulate."""
        pass

    def reset(self) -> None:
        """No-op — nothing to reset."""
        pass

    def state_dict(self) -> Dict:
        """Returns empty dict — nothing to checkpoint."""
        return {}

    def load_state_dict(self, state: Dict) -> None:
        """No-op — nothing to restore."""
        pass

    def compute_regu_loss(self, loradict: Optional[Dict], mb_idx: int,
                          num_mb: int, grad_accum_steps: int) -> Tuple[Optional[float], None, None]:
        """No-op — always returns (None, None, None). No hooks registered."""
        return None, None, None

    def set_last_sq_norms(self, sq_norms: List[float]) -> None:
        """No-op — nothing to store."""
        pass

    def maybe_reset_slice(self, sample_idx: int) -> bool:
        """No-op — never resets."""
        return False

    def get_reset_stats(self) -> Tuple[float, float]:
        """No-op — returns (0.0, 0.0)."""
        return 0.0, 0.0

    def init_steps(self) -> None:
        """No-op — nothing to reset."""
        pass

    def reset_slice(self, sample_idx: int) -> None:
        """No-op — nothing to reset."""
        pass

    def update_steps(self, sample_idx: int) -> None:
        """No-op — nothing to increment."""
        pass

    def __repr__(self) -> str:
        return "EmptyDetachState()"
