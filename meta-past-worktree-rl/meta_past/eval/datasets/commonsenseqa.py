"""CommonsenseQA — bucket B, 5-option MCQ."""

from __future__ import annotations

import random

from ._mcq import fmt_mcq_demo, fmt_mcq_query
from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, shots: int = 5, seed: int = 42,
         **_) -> list[EvalItem]:
    from datasets import load_dataset
    train = load_dataset("tau/commonsense_qa", split="train")
    val = load_dataset("tau/commonsense_qa", split="validation")
    train_indices = list(range(len(train)))
    items: list[EvalItem] = []
    for i, ex in enumerate(val):
        rng = random.Random(f"{seed}-{i}")
        demo_idx = rng.sample(train_indices, k=min(shots, len(train_indices)))
        demos = [
            fmt_mcq_demo(train[j]["question"], train[j]["choices"]["text"],
                         train[j]["answerKey"])
            for j in demo_idx
        ]
        items.append(EvalItem(
            context="\n\n".join(demos),
            question=fmt_mcq_query(ex["question"], ex["choices"]["text"]),
            references=[ex["answerKey"]],
            metadata={"n_options": 5, "shots": int(shots)},
            item_id=f"csqa/{ex['id']}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="commonsenseqa",
    bucket="B",
    load=load,
    scorer="mcq_letter",
    default_modes=("shine", "icl", "zero"),
    scorer_kwargs={"n_options": 5},
    notes="ConceptNet-grounded concept associations.",
))
