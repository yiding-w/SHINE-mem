"""HellaSwag — bucket B, 4-option completion MCQ."""

from __future__ import annotations

import random

from ..items import EvalItem
from . import DatasetSpec, register


def _fmt(ex, with_answer: bool = False) -> str:
    stem = ex["ctx"] + " ___"
    opts = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(ex["endings"]))
    s = f"{stem}\nOptions:\n{opts}\nAnswer:"
    if with_answer:
        s += f" ({chr(65 + int(ex['label']))})"
    return s


def load(*, limit: int | None = None, shots: int = 5, seed: int = 42,
         **_) -> list[EvalItem]:
    from datasets import load_dataset
    train = load_dataset("Rowan/hellaswag", split="train")
    val = load_dataset("Rowan/hellaswag", split="validation")
    train_indices = list(range(len(train)))
    items: list[EvalItem] = []
    for i, ex in enumerate(val):
        if not str(ex.get("label", "")).strip():
            continue
        rng = random.Random(f"{seed}-{i}")
        demo_idx = rng.sample(train_indices, k=min(shots, len(train_indices)))
        demos = [_fmt(train[j], with_answer=True) for j in demo_idx
                 if str(train[j].get("label", "")).strip()]
        items.append(EvalItem(
            context="\n\n".join(demos),
            question=_fmt(ex, with_answer=False),
            references=[chr(65 + int(ex["label"]))],
            metadata={"n_options": 4, "shots": int(shots)},
            item_id=f"hellaswag/{ex.get('ind', i)}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="hellaswag",
    bucket="B",
    load=load,
    scorer="mcq_letter",
    default_modes=("shine", "icl", "zero"),
    scorer_kwargs={"n_options": 4},
    notes="Continuation-style MCQ; tests narrative commonsense.",
))
