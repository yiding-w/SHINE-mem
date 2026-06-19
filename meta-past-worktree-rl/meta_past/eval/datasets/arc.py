"""ARC-Challenge — bucket B, 4-option MCQ science.

K-shot demos are drawn from the train split, query from test. We
do **not** filter by topic the way MMLU per-subject splits do; ARC
items are independent.
"""

from __future__ import annotations

import random

from ..items import EvalItem
from . import DatasetSpec, register


def _fmt_choices(choices: list[str]) -> str:
    return "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))


def _fmt_demo(question: str, choices: list[str], answer_letter: str) -> str:
    return (f"Question: {question}\n"
            f"Options:\n{_fmt_choices(choices)}\n"
            f"Answer: ({answer_letter})")


def _fmt_query(question: str, choices: list[str]) -> str:
    return (f"Question: {question}\n"
            f"Options:\n{_fmt_choices(choices)}\n"
            f"Answer:")


def load(*, limit: int | None = None, shots: int = 5,
         seed: int = 42, **_) -> list[EvalItem]:
    from datasets import load_dataset
    train = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    test = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    rng = random.Random(seed)
    train_indices = list(range(len(train)))

    items: list[EvalItem] = []
    for i, ex in enumerate(test):
        n_options = len(ex["choices"]["text"])
        # Sample K demos. Different random subset per item — gives the
        # K-shot a fair shake at variety. Fix seed via (seed, i) so
        # the same item gets the same demos across runs.
        item_rng = random.Random(f"{seed}-{i}")
        demo_idx = item_rng.sample(train_indices, k=min(shots, len(train_indices)))
        demos = []
        for j in demo_idx:
            d = train[j]
            demos.append(_fmt_demo(d["question"], d["choices"]["text"],
                                   d["answerKey"]))

        items.append(EvalItem(
            context="\n\n".join(demos),
            question=_fmt_query(ex["question"], ex["choices"]["text"]),
            references=[ex["answerKey"]],
            metadata={
                "n_options": n_options,
                "shots": int(shots),
                "id": ex["id"],
            },
            item_id=f"arc-challenge/{ex['id']}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="arc_challenge",
    bucket="B",
    load=load,
    scorer="mcq_letter",
    default_modes=("shine", "icl", "zero"),
    scorer_kwargs={"n_options": 4},
    notes="Small, clean 4-option MCQ; good cheap bucket-B test.",
))
