"""NarrativeQA — bucket A, domain shift (fiction summaries) + free-form short answer."""

from __future__ import annotations

from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, **_) -> list[EvalItem]:
    from datasets import load_dataset
    ds = load_dataset("deepmind/narrativeqa", split="validation")
    items: list[EvalItem] = []
    for ex in ds:
        items.append(EvalItem(
            context=ex["document"]["summary"]["text"],
            question=ex["question"]["text"],
            references=[a["text"] for a in ex["answers"]],
            metadata={"doc_kind": ex["document"]["kind"]},
            item_id=f"narrativeqa/{ex['document']['id']}/{ex['question']['text'][:32]}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="narrativeqa",
    bucket="A",
    load=load,
    scorer="f1",
    default_modes=("shine", "icl", "zero"),
    notes="Uses summary as context; full text would be 50k-100k tokens.",
))
