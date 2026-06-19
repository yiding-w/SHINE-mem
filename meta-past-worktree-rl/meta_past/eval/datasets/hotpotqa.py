"""HotpotQA distractor — bucket A, multi-hop QA over 10 paragraphs.

For eval context we use only the *supporting* paragraphs (the gold
2-hop chain) to keep context length manageable. Distractor paragraphs
remain available via metadata if you want to test retrieval robustness
in a later sweep.
"""

from __future__ import annotations

from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, **_) -> list[EvalItem]:
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation",
                      trust_remote_code=True)
    items: list[EvalItem] = []
    for i, ex in enumerate(ds):
        sup_titles = set(ex["supporting_facts"]["title"])
        blocks = []
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            if title in sup_titles:
                blocks.append(f"# {title}\n{''.join(sents)}")
        items.append(EvalItem(
            context="\n\n".join(blocks),
            question=ex["question"],
            references=[ex["answer"]],
            metadata={"id": ex["id"], "type": ex["type"], "level": ex["level"]},
            item_id=f"hotpotqa/{ex['id']}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="hotpotqa",
    bucket="A",
    load=load,
    scorer="f1",
    default_modes=("shine", "icl", "zero"),
    notes="Multi-hop QA; closest to MuSiQue but different release.",
))
