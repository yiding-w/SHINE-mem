#!/usr/bin/env python
"""Convert official CounterFact records into ControlSHINE three-source JSONL.

CounterFact provides a true target and one counterfactual target.  ControlSHINE
needs a third, distinct target.  This converter draws it from another record
with the same rewrite prompt template, which keeps relation/type compatibility
better than sampling from the complete answer pool.

This is a construction step, not the final benchmark filter.  The resulting
records still need checkpoint-specific Base/Memory/Context recoverability runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controlshine.schema import ControlSample, SourceFact  # noqa: E402


def _text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("str", "")
    return str(value).strip()


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _rewrite(record: dict[str, Any]) -> dict[str, Any]:
    rewrite = record.get("requested_rewrite")
    if isinstance(rewrite, list):
        if len(rewrite) != 1:
            raise ValueError("expected exactly one requested_rewrite")
        rewrite = rewrite[0]
    if not isinstance(rewrite, dict):
        raise ValueError("missing requested_rewrite")
    return rewrite


def _render_fact(template: str, subject: str, answer: str) -> str:
    stem = template.format(subject) if "{}" in template else f"{template} {subject}"
    return f"{stem.rstrip()} {answer}".strip()


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("CounterFact input must be a JSON list or JSONL records")
    return payload


def convert_records(
    records: Iterable[dict[str, Any]], *, seed: int = 42, limit: int | None = None
) -> tuple[list[ControlSample], dict[str, int]]:
    records = list(records)
    rng = random.Random(seed)
    pools: dict[str, list[str]] = defaultdict(list)
    parsed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    stats = defaultdict(int)

    for record in records:
        try:
            rewrite = _rewrite(record)
            template = _text(rewrite["prompt"])
            true_answer = _text(rewrite["target_true"])
            new_answer = _text(rewrite["target_new"])
            subject = _text(rewrite["subject"])
            if not all((template, true_answer, new_answer, subject)):
                raise ValueError("empty core field")
        except (KeyError, TypeError, ValueError):
            stats["malformed"] += 1
            continue
        parsed.append((record, rewrite))
        pools[template].extend((true_answer, new_answer))

    for template, values in pools.items():
        pools[template] = sorted(set(values), key=lambda x: (_norm(x), x))

    output: list[ControlSample] = []
    for record, rewrite in parsed:
        template = _text(rewrite["prompt"])
        subject = _text(rewrite["subject"])
        true_answer = _text(rewrite["target_true"])
        new_answer = _text(rewrite["target_new"])
        forbidden = {_norm(true_answer), _norm(new_answer)}
        donors = [value for value in pools[template] if _norm(value) not in forbidden]
        if not donors:
            stats["no_same_relation_third_answer"] += 1
            continue
        context_answer = rng.choice(donors)

        generation_prompts = list(record.get("paraphrase_prompts") or [])
        rewrite_prompt = template.format(subject) if "{}" in template else template
        if not generation_prompts:
            generation_prompts = [rewrite_prompt]

        case_id = record.get("case_id", len(output))
        sample = ControlSample(
            sample_id=f"counterfact-{case_id}",
            question=_text(generation_prompts[0]),
            entity=subject,
            relation=template,
            base=SourceFact(_render_fact(template, subject, true_answer), true_answer),
            memory=SourceFact(_render_fact(template, subject, new_answer), new_answer),
            context=SourceFact(_render_fact(template, subject, context_answer), context_answer),
            provenance={
                "dataset": "CounterFact",
                "case_id": case_id,
                "third_answer_strategy": "same_rewrite_template_donor",
                "seed": seed,
            },
            prompts={
                "rewrite": rewrite_prompt,
                "paraphrases": generation_prompts,
                "neighborhood": list(record.get("neighborhood_prompts") or []),
                "attribute": list(record.get("attribute_prompts") or []),
            },
        )
        try:
            sample.validate()
        except ValueError:
            stats["invalid"] += 1
            continue
        output.append(sample)
        if limit is not None and len(output) >= limit:
            break

    stats["input"] = len(records)
    stats["parsed"] = len(parsed)
    stats["written"] = len(output)
    return output, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "counterfact_three_source.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    samples, stats = convert_records(
        _load_records(args.input), seed=args.seed, limit=args.limit
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
