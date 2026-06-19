"""MMLU zero-shot — bucket C, side-effect probe.

Same items as bucket-B's MMLU but with empty context.
"""

from __future__ import annotations

from ._mcq import fmt_mcq_query
from ..items import EvalItem
from . import DatasetSpec, register
from .mmlu import _MMLU_SUBJECTS


def load(*, limit: int | None = None,
         subjects: tuple[str, ...] | None = None, **_) -> list[EvalItem]:
    from datasets import load_dataset
    use_subjects = subjects or _MMLU_SUBJECTS
    items: list[EvalItem] = []
    for subj in use_subjects:
        try:
            test = load_dataset("cais/mmlu", subj, split="test")
        except Exception:
            continue
        for i, ex in enumerate(test):
            items.append(EvalItem(
                context="",
                question=fmt_mcq_query(ex["question"], ex["choices"]),
                references=[chr(65 + ex["answer"])],
                metadata={"subject": subj, "n_options": 4},
                item_id=f"mmlu_zs/{subj}/{i}",
            ))
            if limit is not None and len(items) >= limit:
                return items
    return items


register(DatasetSpec(
    name="mmlu_zeroshot",
    bucket="C",
    load=load,
    scorer="mcq_letter",
    default_modes=("zero",),
    scorer_kwargs={"n_options": 4},
    notes="MMLU with no in-context demos. Compare against bucket-B `mmlu` k=5.",
))
