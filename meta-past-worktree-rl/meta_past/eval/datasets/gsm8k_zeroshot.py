"""GSM8K zero-shot — bucket C, side-effect probe.

Same GSM8K test items as the bucket-B adapter, but ``context = ''``
and only ``zero`` mode is run. Pair with bucket-B's K-shot results to
see how much the LoRA-compiled CoT helps vs. nothing.
"""

from __future__ import annotations

from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, **_) -> list[EvalItem]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items: list[EvalItem] = []
    for i, ex in enumerate(ds):
        gold = ex["answer"].split("####")[-1].strip()
        items.append(EvalItem(
            context="",
            question=f"Question: {ex['question']}\nAnswer:",
            references=[gold, ex["answer"]],
            metadata={},
            item_id=f"gsm8k_zs/{i}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="gsm8k_zeroshot",
    bucket="C",
    load=load,
    scorer="numeric_em",
    default_modes=("zero",),
    notes="Same GSM8K data, no demos / no LoRA. Side-effect probe.",
))
