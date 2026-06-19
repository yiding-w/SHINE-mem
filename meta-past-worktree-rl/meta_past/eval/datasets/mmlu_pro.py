"""MMLU-Pro — bucket B, 10-option MCQ, harder.

Demos drawn from the SAME ``category`` as the query, mimicking MMLU's
same-subject protocol.
"""

from __future__ import annotations

import random
from collections import defaultdict

from ._mcq import fmt_mcq_demo, fmt_mcq_query
from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, shots: int = 5, seed: int = 42,
         **_) -> list[EvalItem]:
    from datasets import load_dataset
    test = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    # Bucket by category to pool same-category demos.
    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(test):
        by_cat[ex["category"]].append(i)

    items: list[EvalItem] = []
    for i, ex in enumerate(test):
        cat = ex["category"]
        pool = [j for j in by_cat[cat] if j != i]
        if not pool:
            continue
        rng = random.Random(f"{seed}-{cat}-{i}")
        demo_idx = rng.sample(pool, k=min(shots, len(pool)))
        demos = [
            fmt_mcq_demo(test[j]["question"], test[j]["options"],
                         test[j]["answer"])
            for j in demo_idx
        ]
        items.append(EvalItem(
            context="\n\n".join(demos),
            question=fmt_mcq_query(ex["question"], ex["options"]),
            references=[ex["answer"]],
            metadata={"category": cat, "n_options": len(ex["options"]),
                      "shots": int(shots)},
            item_id=f"mmlu_pro/{cat}/{i}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="mmlu_pro",
    bucket="B",
    load=load,
    scorer="mcq_letter",
    default_modes=("shine", "icl", "zero"),
    scorer_kwargs={"n_options": 10},
    notes="Harder MMLU successor with 10 options per item.",
))
