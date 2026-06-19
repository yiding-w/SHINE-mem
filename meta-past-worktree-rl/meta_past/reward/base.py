"""Reward-function interface shared by the F1 / judge / learned reward modules.

The base ``RewardFn`` protocol is ``(pred, references) -> float`` and is what
F1 needs. Judge-style rewards additionally need the question (and possibly
the context) to score, and benefit from concurrent batched calls when
many samples need scoring per training step. Reward implementations that
support those expose ``batch_compute(...)`` and accept a ``question``
keyword; the rollout dispatches accordingly.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class RewardFn(Protocol):
    """A reward callable returns a scalar in [0, 1] given a prediction + refs.

    ``question`` is an optional kwarg — F1 ignores it; LLM-judge rewards use
    it. Reward implementations that ignore the question should accept and
    discard it silently.
    """

    def __call__(
        self, pred: str, references: Sequence[str], *, question: str = "",
    ) -> float:  # noqa: D401
        ...

    def batch_compute(
        self,
        preds: Sequence[str],
        references_list: Sequence[Sequence[str]],
        *,
        questions: Sequence[str] | None = None,
    ) -> list[float]:
        """Optional batched entry point.

        Default implementation (provided by helpers) loops sequentially
        over the inputs; specialized implementations (HTTP-judge with a
        thread pool) override this for concurrent calls.
        """
        ...


def sequential_batch_compute(
    fn: RewardFn,
    preds: Sequence[str],
    references_list: Sequence[Sequence[str]],
    *,
    questions: Sequence[str] | None = None,
) -> list[float]:
    """Default ``batch_compute`` implementation that just iterates."""
    qs = list(questions) if questions is not None else [""] * len(preds)
    if len(qs) != len(preds):
        raise ValueError(
            f"questions length {len(qs)} != preds length {len(preds)}"
        )
    return [
        float(fn(p, r, question=q))
        for p, r, q in zip(preds, references_list, qs)
    ]
