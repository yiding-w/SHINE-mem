"""StrategyQA — bucket B, implicit multi-step reasoning yes/no."""

from __future__ import annotations

import random

from ..items import EvalItem
from . import DatasetSpec, register


def _norm_ans(a) -> str:
    if a is True:
        return "yes"
    if a is False:
        return "no"
    return str(a).strip().lower()


def _fmt(question: str, ans: str | None = None) -> str:
    s = f"Question: {question}\nAnswer:"
    if ans is not None:
        s += f" {ans}"
    return s


def load(*, limit: int | None = None, shots: int = 4, seed: int = 42,
         **_) -> list[EvalItem]:
    from datasets import load_dataset
    # ChilleD/StrategyQA exposes train + test.
    train = load_dataset("ChilleD/StrategyQA", split="train")
    test = load_dataset("ChilleD/StrategyQA", split="test")
    train_indices = list(range(len(train)))
    items: list[EvalItem] = []
    for i, ex in enumerate(test):
        rng = random.Random(f"{seed}-{i}")
        demo_idx = rng.sample(train_indices, k=min(shots, len(train_indices)))
        demos = [
            _fmt(train[j].get("question", ""),
                 _norm_ans(train[j].get("answer")))
            for j in demo_idx
        ]
        items.append(EvalItem(
            context="\n\n".join(demos),
            question=_fmt(ex.get("question", "")),
            references=[_norm_ans(ex.get("answer"))],
            metadata={"shots": int(shots)},
            item_id=f"strategyqa/{i}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="strategyqa",
    bucket="B",
    load=load,
    scorer="em",
    default_modes=("shine", "icl", "zero"),
    notes="Yes/no on implicit multi-step reasoning.",
))
