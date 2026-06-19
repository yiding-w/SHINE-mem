"""Protocol for rollout functions used by the ES trainer."""

from __future__ import annotations

from typing import Any, Protocol


class RolloutFn(Protocol):
    """Evaluate a batch of contexts under current hypernet params → scalar reward."""

    def __call__(self, contexts: list[Any]) -> float: ...
