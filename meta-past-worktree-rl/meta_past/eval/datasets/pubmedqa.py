"""PubMedQA (labeled) — bucket A, biomedical yes/no/maybe."""

from __future__ import annotations

from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, **_) -> list[EvalItem]:
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    items: list[EvalItem] = []
    for ex in ds:
        ctx = "\n\n".join(ex["context"]["contexts"])
        q = (ex["question"] +
             " Answer with just 'yes', 'no', or 'maybe'.")
        items.append(EvalItem(
            context=ctx,
            question=q,
            references=[ex["final_decision"]],
            metadata={"pubid": ex["pubid"]},
            item_id=f"pubmedqa/{ex['pubid']}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="pubmedqa",
    bucket="A",
    load=load,
    scorer="em",
    default_modes=("shine", "icl", "zero"),
    notes="Strong domain shift to biomedical literature.",
))
