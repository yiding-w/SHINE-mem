"""2WikiMultihopQA — bucket A, compositional / bridge-comparison QA."""

from __future__ import annotations

import ast
import json

from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, **_) -> list[EvalItem]:
    from datasets import load_dataset
    ds = load_dataset("xanhho/2WikiMultiHopQA", split="validation")
    items: list[EvalItem] = []
    for ex in ds:
        # context / supporting_facts are JSON-ish strings.
        try:
            paras = json.loads(ex["context"])
        except Exception:
            paras = ast.literal_eval(ex["context"])
        sup = ex.get("supporting_facts", [])
        try:
            sup = json.loads(sup) if isinstance(sup, str) else sup
        except Exception:
            sup = ast.literal_eval(sup) if isinstance(sup, str) else sup
        sup_titles = {t for t, _ in sup} if sup else set()
        blocks = []
        for title, sents in paras:
            if (not sup_titles) or (title in sup_titles):
                blocks.append(f"# {title}\n{''.join(sents)}")
        items.append(EvalItem(
            context="\n\n".join(blocks),
            question=ex["question"],
            references=[ex["answer"]],
            metadata={"id": ex["_id"], "type": ex["type"]},
            item_id=f"2wikimulti/{ex['_id']}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="2wikimulti",
    bucket="A",
    load=load,
    scorer="f1",
    default_modes=("shine", "icl", "zero"),
    notes="Multi-hop with explicit reasoning-chain templates.",
))
