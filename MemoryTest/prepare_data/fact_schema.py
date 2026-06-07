from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class MemoryFact:
    id: str
    person: str
    attribute: str
    text: str
    question: str
    answer: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MemoryFact":
        missing = [key for key in ("id", "person", "text", "question", "answer") if key not in row]
        if missing:
            raise ValueError(f"Fact row is missing required keys {missing}: {row!r}")
        return cls(
            id=str(row["id"]),
            person=str(row["person"]),
            attribute=str(row.get("attribute", row.get("relation", "unknown"))),
            text=str(row["text"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
        )

    def to_row(self) -> dict[str, str]:
        return {
            "id": self.id,
            "person": self.person,
            "attribute": self.attribute,
            "text": self.text,
            "question": self.question,
            "answer": self.answer,
        }

    @property
    def triple_key(self) -> tuple[str, str, str]:
        return (self.person.casefold(), self.attribute.casefold(), self.answer.casefold())


def normalize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    return [MemoryFact.from_row(row).to_row() for row in rows]


def relation_distribution(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("attribute", row.get("relation", "unknown"))) for row in rows)
    return dict(sorted(counts.items()))


def assert_person_disjoint(*splits: Iterable[dict[str, Any]]) -> None:
    person_sets = [set(str(row["person"]) for row in split) for split in splits]
    for left_idx, left in enumerate(person_sets):
        for right_idx, right in enumerate(person_sets[left_idx + 1 :], start=left_idx + 1):
            overlap = left & right
            if overlap:
                sample = sorted(overlap)[:10]
                raise ValueError(f"Person leakage between split {left_idx} and {right_idx}: {sample}")


def assert_no_test_triple_leakage(train_rows: Iterable[dict[str, Any]], protected_rows: Iterable[dict[str, Any]]) -> None:
    train_triples = {MemoryFact.from_row(row).triple_key for row in train_rows}
    protected_triples = {MemoryFact.from_row(row).triple_key for row in protected_rows}
    overlap = train_triples & protected_triples
    if overlap:
        sample = sorted(overlap)[:10]
        raise ValueError(f"Train/protected triple leakage: {sample}")


def group_by_person(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in normalize_rows(rows):
        grouped.setdefault(row["person"], []).append(row)
    return grouped
