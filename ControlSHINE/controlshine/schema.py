"""Canonical data schema for three-source ControlSHINE examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceFact:
    text: str
    answer: str
    aliases: tuple[str, ...] = ()

    def validate(self, source: str) -> None:
        if not self.text.strip():
            raise ValueError(f"{source}.text must not be empty")
        if not self.answer.strip():
            raise ValueError(f"{source}.answer must not be empty")


@dataclass(frozen=True)
class ControlSample:
    sample_id: str
    question: str
    base: SourceFact
    memory: SourceFact
    context: SourceFact
    relation: str | None = None
    entity: str | None = None
    split: str = "dev"
    provenance: dict[str, Any] = field(default_factory=dict)
    prompts: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.sample_id.strip() or not self.question.strip():
            raise ValueError("sample_id and question must not be empty")
        for name in ("base", "memory", "context"):
            getattr(self, name).validate(name)
        normalized = {
            self.base.answer.strip().casefold(),
            self.memory.answer.strip().casefold(),
            self.context.answer.strip().casefold(),
        }
        if len(normalized) != 3:
            raise ValueError("base, memory, and context answers must be distinct")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        for source in ("base", "memory", "context"):
            result[source]["aliases"] = list(result[source]["aliases"])
        return result
