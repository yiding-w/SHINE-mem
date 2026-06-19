"""PIQA — bucket B, 2-option physical commonsense MCQ."""

from __future__ import annotations

import random

from ..items import EvalItem
from . import DatasetSpec, register


def _fmt(ex, with_answer: bool = False) -> str:
    s = (f"Goal: {ex['goal']}\n"
         f"Options:\n(A) {ex['sol1']}\n(B) {ex['sol2']}\n"
         f"Answer:")
    if with_answer:
        s += f" ({'A' if ex['label']==0 else 'B'})"
    return s


def load(*, limit: int | None = None, shots: int = 5, seed: int = 42,
         **_) -> list[EvalItem]:
    from datasets import load_dataset
    train = load_dataset("lighteval/piqa", split="train")
    val = load_dataset("lighteval/piqa", split="validation")
    train_indices = list(range(len(train)))
    items: list[EvalItem] = []
    for i, ex in enumerate(val):
        rng = random.Random(f"{seed}-{i}")
        demo_idx = rng.sample(train_indices, k=min(shots, len(train_indices)))
        demos = [_fmt(train[j], with_answer=True) for j in demo_idx]
        items.append(EvalItem(
            context="\n\n".join(demos),
            question=_fmt(ex, with_answer=False),
            references=["A" if ex["label"] == 0 else "B"],
            metadata={"n_options": 2, "shots": int(shots)},
            item_id=f"piqa/{i}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="piqa",
    bucket="B",
    load=load,
    scorer="mcq_letter",
    default_modes=("shine", "icl", "zero"),
    scorer_kwargs={"n_options": 2},
    notes="Physical commonsense — simple 2-option choice.",
))
