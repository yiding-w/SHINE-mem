"""TriviaQA (RC, Wikipedia) — bucket A, factual QA with wiki evidence.

We use the with-context split; ``context`` = wiki entity pages
concatenated as ``# Title\\nbody`` blocks. Each page is typically
long (~30k chars), so eval truncation to ``context_max_length`` is
expected.
"""

from __future__ import annotations

from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, **_) -> list[EvalItem]:
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia",
                      split="validation")
    items: list[EvalItem] = []
    for ex in ds:
        pages = ex.get("entity_pages", {}) or {}
        titles = pages.get("title", []) if isinstance(pages, dict) else []
        bodies = pages.get("wiki_context", []) if isinstance(pages, dict) else []
        blocks = [f"# {t}\n{b}" for t, b in zip(titles, bodies) if b]
        context = "\n\n".join(blocks)
        ans = ex.get("answer", {}) or {}
        refs: list[str] = []
        if ans.get("value"):
            refs.append(ans["value"])
        refs.extend(ans.get("aliases", []) or [])
        items.append(EvalItem(
            context=context,
            question=ex["question"],
            references=refs,
            metadata={"question_id": ex.get("question_id", "")},
            item_id=f"triviaqa/{ex.get('question_id', '')}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="triviaqa",
    bucket="A",
    load=load,
    scorer="f1",
    default_modes=("shine", "icl", "zero"),
    notes="Wiki evidence pages are long; truncation expected.",
))
