"""GSM8K — bucket B, grade-school math with CoT few-shot.

Demos include the full chain-of-thought (the dataset's gold solution,
which ends with ``#### <number>``). The scorer pulls the predicted
final number from the model's completion.
"""

from __future__ import annotations

import random

from ..items import EvalItem
from . import DatasetSpec, register


def _fmt_demo(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def _fmt_query(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def load(*, limit: int | None = None, shots: int = 8, seed: int = 42,
         **_) -> list[EvalItem]:
    from datasets import load_dataset
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")

    train_indices = list(range(len(train)))
    items: list[EvalItem] = []
    for i, ex in enumerate(test):
        item_rng = random.Random(f"{seed}-{i}")
        demo_idx = item_rng.sample(train_indices, k=min(shots, len(train_indices)))
        demos = [_fmt_demo(train[j]["question"], train[j]["answer"])
                 for j in demo_idx]
        gold = ex["answer"].split("####")[-1].strip()
        items.append(EvalItem(
            context="\n\n".join(demos),
            question=_fmt_query(ex["question"]),
            references=[gold, ex["answer"]],
            metadata={"shots": int(shots)},
            item_id=f"gsm8k/{i}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="gsm8k",
    bucket="B",
    load=load,
    scorer="numeric_em",
    default_modes=("shine", "icl", "zero"),
    notes=("CoT demos compress 'math reasoning' into LoRA. "
           "Numeric extraction scores the final '#### N'."),
))
