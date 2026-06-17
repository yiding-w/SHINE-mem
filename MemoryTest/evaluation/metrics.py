from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Iterable


MATCH_MODE = "case_insensitive_answer_substring"


def normalize_text(value: object) -> str:
    return str(value).strip().casefold()


def answer_matches(expected_answer: object, model_answer: object) -> bool:
    expected = normalize_text(expected_answer)
    answer = normalize_text(model_answer)
    return bool(expected) and expected in answer


def summarize_examples(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def wrong_examples(rows: list[dict], limit: int = 10) -> list[dict]:
    return [row for row in rows if not row.get("correct")][:limit]


def relation_breakdown(rows: Iterable[dict]) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("attribute", row.get("relation", "unknown")))].append(row)
    return {relation: summarize_examples(items) for relation, items in sorted(buckets.items())}


def aggregate_trial_results(results: list[dict], metric_key: str = "accuracy") -> dict:
    values = [float(result.get(metric_key, 0.0)) for result in results]
    return {
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "num_trials": len(values),
    }


def make_eval_row(index: int, fact: dict, model_answer: str, raw: str = "") -> dict:
    correct = answer_matches(fact["answer"], model_answer)
    return {
        "index": index,
        "id": fact["id"],
        "person": fact["person"],
        "attribute": fact.get("attribute", fact.get("relation", "unknown")),
        "question": fact["question"],
        "expected_answer": fact["answer"],
        "model_answer": model_answer,
        "raw": raw,
        "match_mode": MATCH_MODE,
        "correct": correct,
    }
