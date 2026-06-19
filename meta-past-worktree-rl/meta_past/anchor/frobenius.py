"""Soft L2 anchor pulling phi back toward the pretrained checkpoint.

After each ES update:  phi <- phi - lr * lambda(t) * (phi - phi_pretrained)

The snapshot is kept on CPU (pinned if available) so we don't burn HBM on a
second copy of the hypernetwork. It's streamed to the GPU per-tensor in the
step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class AnchorSchedule:
    coef_start: float = 1.0
    coef_end: float = 0.1
    decay_steps: int = 300

    def at(self, step: int) -> float:
        if step <= 0:
            return self.coef_start
        if step >= self.decay_steps:
            return self.coef_end
        frac = step / self.decay_steps
        return self.coef_start + (self.coef_end - self.coef_start) * frac


class FrobeniusAnchor:
    """Snapshot pretrained phi and apply the anchor gradient in place.

    The snapshot must be taken immediately after ``ShineHypernet`` construction
    and before ES starts modifying any tensors. Use ``FrobeniusAnchor.snapshot``
    to build it cleanly.
    """

    def __init__(
        self,
        params: Sequence[tuple[str, torch.Tensor]],
        schedule: AnchorSchedule | None = None,
    ):
        self.schedule = schedule or AnchorSchedule()
        self._snap: dict[str, torch.Tensor] = {}
        self.snapshot(params)
        self._rollback_mul = 1.0

    def snapshot(self, params: Sequence[tuple[str, torch.Tensor]]) -> None:
        self._snap = {}
        for name, p in params:
            # CPU copy in float32 — memory-cheap vs keeping a full bf16 duplicate
            # on GPU, and precision-safe for computing (phi - phi_pretrained).
            self._snap[name] = p.detach().to("cpu", dtype=torch.float32).clone()

    def apply_step(
        self,
        params: Sequence[tuple[str, torch.Tensor]],
        lr: float,
        step: int,
    ) -> None:
        coef = self.schedule.at(step) * self._rollback_mul
        if coef <= 0:
            return
        scale = lr * coef
        for name, p in params:
            snap = self._snap.get(name)
            if snap is None:
                raise KeyError(
                    f"Anchor snapshot missing for {name!r}; did you add a "
                    f"parameter after constructing the anchor?"
                )
            target = snap.to(p.device, dtype=p.dtype, non_blocking=True)
            p.data.add_(target - p.data, alpha=scale)

    # -- adaptive rollback -----------------------------------------------------

    def bump_coef(self, factor: float = 2.0, cap: float = 4.0) -> None:
        """Multiply the schedule's output by ``factor`` (bounded by ``cap``).

        Call this when held-out reward regresses — the anchor pulls harder
        until the model recovers.
        """
        self._rollback_mul = min(cap, self._rollback_mul * factor)

    def reset_rollback(self) -> None:
        self._rollback_mul = 1.0
