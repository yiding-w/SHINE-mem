"""TruthfulQA Generation — bucket C, free-form truthfulness probe.

Scoring is intentionally cheap (F1 over the union of ``correct_answers``)
because the official setup uses a GPT-judge that isn't always available.
Treat the number as an indication, not a definitive truthfulness score —
upgrade to judge scoring later if/when needed.
"""

from __future__ import annotations

from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, **_) -> list[EvalItem]:
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation",
                      split="validation")
    items: list[EvalItem] = []
    for i, ex in enumerate(ds):
        items.append(EvalItem(
            context="",
            question=ex["question"],
            references=list(ex["correct_answers"]),
            metadata={"incorrect_answers": list(ex.get("incorrect_answers", []))},
            item_id=f"truthfulqa_gen/{i}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="truthfulqa_gen",
    bucket="C",
    load=load,
    scorer="f1",
    default_modes=("zero",),
    notes="F1 vs correct_answers as a cheap proxy; official scoring uses GPT-judge.",
))
