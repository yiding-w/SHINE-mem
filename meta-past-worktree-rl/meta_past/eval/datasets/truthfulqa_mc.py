"""TruthfulQA MC1 — bucket B, MCQ with truthfulness focus.

MC1 = exactly one correct answer per item; we pick demos from other
items (cross-category) to avoid leaking the correct answer pattern.
"""

from __future__ import annotations

import random

from ._mcq import fmt_mcq_demo, fmt_mcq_query
from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, shots: int = 3, seed: int = 42,
         **_) -> list[EvalItem]:
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice",
                      split="validation")
    n = len(ds)
    items: list[EvalItem] = []
    for i, ex in enumerate(ds):
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        if 1 not in labels:
            continue
        correct_idx = labels.index(1)
        rng = random.Random(f"{seed}-{i}")
        pool = [j for j in range(n) if j != i]
        demo_idx = rng.sample(pool, k=min(shots, len(pool)))
        demos: list[str] = []
        for j in demo_idx:
            d = ds[j]
            d_labels = d["mc1_targets"]["labels"]
            if 1 not in d_labels:
                continue
            d_choices = d["mc1_targets"]["choices"]
            d_correct = d_labels.index(1)
            demos.append(fmt_mcq_demo(d["question"], d_choices,
                                      chr(65 + d_correct)))
        items.append(EvalItem(
            context="\n\n".join(demos),
            question=fmt_mcq_query(ex["question"], choices),
            references=[chr(65 + correct_idx)],
            metadata={"n_options": len(choices), "shots": int(shots)},
            item_id=f"truthfulqa_mc/{i}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="truthfulqa_mc",
    bucket="B",
    load=load,
    scorer="mcq_letter",
    default_modes=("shine", "icl", "zero"),
    scorer_kwargs={"n_options": 10},  # max across items; scorer ignores extras
    notes="MC1 single-correct; hallucination probe.",
))
