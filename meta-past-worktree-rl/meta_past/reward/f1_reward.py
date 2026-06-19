"""SQuAD-style F1 reward. Wraps SHINE's calculate_f1.compute_f1."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


def _ensure_shine_on_path() -> None:
    shine_root = Path(__file__).resolve().parents[2] / "third_party" / "SHINE"
    p = str(shine_root)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_shine_on_path()
from calculate_f1 import compute_f1 as _compute_f1  # type: ignore


def f1_reward(pred: str, references: Sequence[str], *, question: str = "") -> float:
    """Max F1 across the reference answers. SQuAD scoring convention.

    ``question`` is accepted (and ignored) so the call signature matches
    the LLM-judge reward — both can be plugged in interchangeably.
    """
    del question  # unused by F1
    if not references:
        return 0.0
    return float(max(_compute_f1(ref, pred) for ref in references))


def f1_reward_batch(
    preds: Sequence[str],
    references_list: Sequence[Sequence[str]],
    *,
    questions: Sequence[str] | None = None,
) -> list[float]:
    del questions
    return [f1_reward(p, r) for p, r in zip(preds, references_list)]


# Attach the batch helper as an attribute so callers can do
# ``getattr(reward_fn, "batch_compute", None)`` uniformly across reward types.
f1_reward.batch_compute = f1_reward_batch  # type: ignore[attr-defined]
