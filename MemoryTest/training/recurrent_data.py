from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from MemoryTest.prepare_data.prompt_templates import build_context


SCHEMA_NAME = "shine_recurrent_v1"


@dataclass(frozen=True)
class QARecord:
    qa_id: str
    question: str
    answer: str
    source_turn_id: str

    def as_row(self) -> dict[str, str]:
        return {
            "id": self.qa_id,
            "question": self.question,
            "answer": self.answer,
            "source_turn_id": self.source_turn_id,
        }


@dataclass(frozen=True)
class TrainingTurn:
    turn_id: str
    text: str
    qa: tuple[QARecord, ...]
    fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingStream:
    stream_id: str
    turns: tuple[TrainingTurn, ...]
    source_kind: str


@dataclass(frozen=True)
class FactRecord:
    fact_id: str
    text: str
    qa: tuple[QARecord, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class RecurrentDataset:
    streams: tuple[TrainingStream, ...]
    facts: tuple[FactRecord, ...]

    @property
    def has_streams(self) -> bool:
        return bool(self.streams)

    @property
    def has_facts(self) -> bool:
        return bool(self.facts)


def _read_payload(path: str | Path) -> Any:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_qa(container: dict[str, Any], source_id: str) -> tuple[QARecord, ...]:
    qa_items = container.get("qa")
    if qa_items is None and "question" in container and "answer" in container:
        qa_items = [{"question": container["question"], "answer": container["answer"]}]
    if qa_items is None:
        return ()
    if not isinstance(qa_items, list):
        raise ValueError(f"{source_id}.qa must be a list")
    normalized = []
    for index, item in enumerate(qa_items):
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            raise ValueError(f"{source_id}.qa[{index}] must contain question and answer")
        normalized.append(
            QARecord(
                qa_id=str(item.get("id", f"{source_id}:qa:{index}")),
                question=str(item["question"]),
                answer=str(item["answer"]),
                source_turn_id=source_id,
            )
        )
    return tuple(normalized)


def _normalize_fact(row: dict[str, Any], index: int) -> FactRecord:
    if not isinstance(row, dict) or "text" not in row:
        raise ValueError(f"facts[{index}] must be an object containing text")
    fact_id = str(row.get("id", f"fact:{index}"))
    return FactRecord(
        fact_id=fact_id,
        text=str(row["text"]),
        qa=_normalize_qa(row, fact_id),
        raw=dict(row),
    )


def _normalize_stream(row: dict[str, Any], index: int) -> TrainingStream:
    if not isinstance(row, dict) or not isinstance(row.get("turns"), list) or not row["turns"]:
        raise ValueError(f"streams[{index}] must contain a non-empty turns list")
    stream_id = str(row.get("stream_id", row.get("id", f"stream:{index}")))
    turns = []
    seen_turn_ids = set()
    for turn_index, turn in enumerate(row["turns"]):
        if not isinstance(turn, dict) or "text" not in turn:
            raise ValueError(f"{stream_id}.turns[{turn_index}] must contain text")
        turn_id = str(turn.get("turn_id", turn.get("id", f"{stream_id}:turn:{turn_index}")))
        if turn_id in seen_turn_ids:
            raise ValueError(f"Duplicate turn_id {turn_id!r} in stream {stream_id!r}")
        seen_turn_ids.add(turn_id)
        turns.append(
            TrainingTurn(
                turn_id=turn_id,
                text=str(turn["text"]),
                qa=_normalize_qa(turn, turn_id),
            )
        )
    return TrainingStream(stream_id=stream_id, turns=tuple(turns), source_kind="ordered_stream")


def load_recurrent_dataset(path: str | Path) -> RecurrentDataset:
    """Load the canonical schema or transparently adapt a legacy flat fact list."""
    payload = _read_payload(path)
    if isinstance(payload, list):
        # A list of stream objects is accepted; every other legacy list is a fact pool.
        if payload and all(isinstance(row, dict) and "turns" in row for row in payload):
            stream_rows, fact_rows = payload, []
        elif any(isinstance(row, dict) and row.get("stream_id") is not None for row in payload):
            # Compatibility with the temporary flat stream_id/turn representation.
            grouped: dict[str, list[dict[str, Any]]] = {}
            fact_rows = []
            for row in payload:
                if not isinstance(row, dict) or row.get("stream_id") is None:
                    fact_rows.append(row)
                    continue
                grouped.setdefault(str(row["stream_id"]), []).append(row)

            def turn_sort_key(row: dict[str, Any]):
                value = row.get("turn", row.get("turn_id", 0))
                try:
                    return 0, int(value)
                except (TypeError, ValueError):
                    return 1, str(value)

            stream_rows = []
            for stream_id, turns in grouped.items():
                turns.sort(key=turn_sort_key)
                stream_rows.append({"stream_id": stream_id, "turns": turns})
        else:
            stream_rows, fact_rows = [], payload
    elif isinstance(payload, dict):
        schema = payload.get("schema", SCHEMA_NAME)
        if schema != SCHEMA_NAME:
            raise ValueError(f"Unsupported recurrent data schema: {schema!r}")
        stream_rows = payload.get("streams", [])
        fact_rows = payload.get("facts", [])
        if not isinstance(stream_rows, list) or not isinstance(fact_rows, list):
            raise ValueError("streams and facts must both be lists")
    else:
        raise ValueError("Training data must be a JSON object or array")

    streams = tuple(_normalize_stream(row, index) for index, row in enumerate(stream_rows))
    facts = tuple(_normalize_fact(row, index) for index, row in enumerate(fact_rows))
    if not streams and not facts:
        raise ValueError("Training data contains neither streams nor facts")
    return RecurrentDataset(streams=streams, facts=facts)


def _sample_ordered_stream(
    streams: tuple[TrainingStream, ...],
    recurrent_steps: int,
    window_policy: str,
    rng: random.Random,
) -> TrainingStream:
    stream = rng.choice(streams)
    if window_policy == "full" or len(stream.turns) <= recurrent_steps:
        turns = stream.turns
    elif window_policy == "prefix":
        turns = stream.turns[:recurrent_steps]
    elif window_policy == "contiguous":
        start = rng.randint(0, len(stream.turns) - recurrent_steps)
        turns = stream.turns[start : start + recurrent_steps]
    else:
        raise ValueError(f"Unknown stream window policy: {window_policy}")
    return TrainingStream(stream_id=stream.stream_id, turns=turns, source_kind=stream.source_kind)


def _sample_fact_stream(
    facts: tuple[FactRecord, ...],
    recurrent_steps: int,
    fact_counts: list[int],
    context_format: str,
    rng: random.Random,
) -> TrainingStream:
    remaining = list(facts)
    turns = []
    for turn_index in range(recurrent_steps):
        usable_counts = [count for count in fact_counts if 0 < count <= len(remaining)]
        if not usable_counts:
            remaining = list(facts)
            usable_counts = [count for count in fact_counts if 0 < count <= len(remaining)]
        if not usable_counts:
            raise ValueError(f"No fact count in {fact_counts} is usable for a pool of {len(facts)} facts")
        count = rng.choice(usable_counts)
        selected = rng.sample(remaining, count)
        selected_ids = {fact.fact_id for fact in selected}
        remaining = [fact for fact in remaining if fact.fact_id not in selected_ids]
        raw_rows = [fact.raw for fact in selected]
        turn_context_format = rng.choice(["natural", "structured"]) if context_format == "mixed" else context_format
        if turn_context_format == "structured" and any(
            "person" not in row or "answer" not in row for row in raw_rows
        ):
            turn_context_format = "natural"
        turns.append(
            TrainingTurn(
                turn_id=f"synthetic:turn:{turn_index}",
                text=build_context(raw_rows, context_format=turn_context_format),
                qa=tuple(qa for fact in selected for qa in fact.qa),
                fact_ids=tuple(fact.fact_id for fact in selected),
            )
        )
    return TrainingStream(
        stream_id=f"synthetic:{rng.getrandbits(64):016x}",
        turns=tuple(turns),
        source_kind="fact_pool",
    )


def sample_training_stream(
    dataset: RecurrentDataset,
    recurrent_steps: int,
    fact_counts: list[int],
    context_format: str,
    ordered_stream_probability: float,
    window_policy: str,
    rng: random.Random,
) -> TrainingStream:
    if recurrent_steps < 1:
        raise ValueError("recurrent_steps must be at least 1")
    if not 0.0 <= ordered_stream_probability <= 1.0:
        raise ValueError("ordered_stream_probability must be between 0 and 1")
    use_ordered = dataset.has_streams and (
        not dataset.has_facts or rng.random() < ordered_stream_probability
    )
    if use_ordered:
        return _sample_ordered_stream(dataset.streams, recurrent_steps, window_policy, rng)
    return _sample_fact_stream(dataset.facts, recurrent_steps, fact_counts, context_format, rng)


def accumulated_qa(turns: tuple[TrainingTurn, ...] | list[TrainingTurn]) -> list[dict[str, str]]:
    return [qa.as_row() for turn in turns for qa in turn.qa]


def sample_turn_qa(
    turns: tuple[TrainingTurn, ...] | list[TrainingTurn],
    query_count: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    """Prefer one QA from each observed turn, then fill from the accumulated QA pool."""
    available_turns = [turn for turn in turns if turn.qa]
    rng.shuffle(available_turns)
    selected: list[QARecord] = []
    selected_ids = set()
    for turn in available_turns:
        if len(selected) >= query_count:
            break
        qa = rng.choice(turn.qa)
        selected.append(qa)
        selected_ids.add(qa.qa_id)
    remaining = [qa for turn in turns for qa in turn.qa if qa.qa_id not in selected_ids]
    fill_count = min(query_count - len(selected), len(remaining))
    if fill_count > 0:
        selected.extend(rng.sample(remaining, fill_count))
    return [qa.as_row() for qa in selected]
