#!/usr/bin/env python3
"""Convert the official Multi-Session Chat data to shine_recurrent_v1.

The official dialogue files contain cumulative snapshots.  This converter keeps
only the longest snapshot for each ``initial_data_id`` and recovers its ordered
sessions from ``previous_dialogs + dialog``.  Persona fields, ``newfact``, and
``followup`` are never inserted into the recurrent input text.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol


LOGGER = logging.getLogger("prepare_msc_recurrent")
SCHEMA_NAME = "shine_recurrent_v1"


NEXT_TURN_PROMPT = (
    "Continue the dialogue naturally. The previous utterance is:\n"
    "{previous}\n"
    "Write only {speaker}'s next utterance."
)
PERSONA_EXTRACTION_PROMPT = (
    "Extract the new persona facts about {speaker} that are revealed in this "
    "dialogue chunk. Return one fact per line and no explanation."
)
PERSONA_SUMMARY_PROMPT = (
    "Summarize the persona facts about {speaker} accumulated during this "
    "session. Return one fact per line and no explanation."
)


class TextTokenizer(Protocol):
    name: str

    def encode(self, text: str) -> list[Any]: ...

    def decode(self, tokens: list[Any]) -> str: ...


class WhitespaceTokenizer:
    """Dependency-free approximate counter used only when no HF tokenizer is given."""

    name = "whitespace-approximation"

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class HuggingFaceTokenizer:
    def __init__(self, model_name_or_path: str, trust_remote_code: bool = False):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required with --tokenizer; install it in the server environment"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.name = model_name_or_path

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()


@dataclass(frozen=True)
class DialogueChunk:
    text: str
    utterance_indices: tuple[int, ...]
    token_count: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    # Stream encoding to the file instead of materializing another large JSON
    # string; the training split contains more than 150k optional QA records.
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _find_task_root(input_dir: Path, task_name: str, required: bool = True) -> Path | None:
    candidates = [
        input_dir / task_name,
        input_dir / "msc" / task_name,
        input_dir if input_dir.name == task_name else input_dir / "__missing__",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    if required:
        raise FileNotFoundError(
            f"Could not find {task_name}/ below {input_dir}. Point --input-dir at the "
            "extracted official msc_v0.1 archive or its msc/ directory."
        )
    return None


def _session_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid MSC session directory: {path}") from exc


def select_longest_snapshots(dialogue_root: Path, split: str) -> dict[str, dict[str, Any]]:
    """Select one cumulative snapshot per trajectory, avoiding repeated history."""
    selected: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
    for session_dir in sorted(dialogue_root.glob("session_*"), key=_session_number):
        path = session_dir / f"{split}.txt"
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            metadata = row.get("metadata", {})
            trajectory_id = str(metadata.get("initial_data_id", ""))
            if not trajectory_id:
                raise ValueError(f"Missing metadata.initial_data_id in {path}")
            previous = row.get("previous_dialogs")
            current = row.get("dialog")
            if not isinstance(previous, list) or not isinstance(current, list):
                raise ValueError(f"Malformed dialogue snapshot {trajectory_id!r} in {path}")
            score = (len(previous) + 1, _session_number(session_dir))
            if trajectory_id not in selected or score > selected[trajectory_id][0]:
                selected[trajectory_id] = (score, row)
    return {trajectory_id: row for trajectory_id, (_, row) in selected.items()}


def load_persona_annotations(
    persona_root: Path | None,
    split: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Map (trajectory id, zero-based timeline session) to summary annotations."""
    annotations: dict[tuple[str, int], dict[str, Any]] = {}
    if persona_root is None:
        return annotations
    for session_dir in sorted(persona_root.glob("session_*"), key=_session_number):
        path = session_dir / f"{split}.txt"
        if not path.exists():
            continue
        # PersonaSummary session_1 annotates timeline session 0 (PersonaChat).
        timeline_session = _session_number(session_dir) - 1
        for row in _read_jsonl(path):
            trajectory_id = str(row.get("initial_data_id", ""))
            if trajectory_id:
                annotations[(trajectory_id, timeline_session)] = row
    return annotations


def _speaker(index: int) -> str:
    return f"Speaker {index % 2 + 1}"


def _split_long_utterance(
    text: str,
    speaker: str,
    tokenizer: TextTokenizer,
    max_tokens: int,
) -> list[str]:
    prefix = f"{speaker}: "
    if len(tokenizer.encode(prefix)) >= max_tokens:
        raise ValueError(f"--max-turn-tokens={max_tokens} is too small for speaker prefixes")
    remaining = tokenizer.encode(str(text).strip())
    if not remaining:
        return [prefix.rstrip()]
    pieces = []
    while remaining:
        # Reserve the prefix, then shrink if decoding changes token boundaries.
        budget = max(1, max_tokens - len(tokenizer.encode(prefix)))
        take = min(budget, len(remaining))
        while take > 0:
            piece = prefix + tokenizer.decode(remaining[:take])
            if len(tokenizer.encode(piece)) <= max_tokens:
                break
            take -= 1
        if take == 0:
            raise ValueError(f"Unable to fit an utterance into {max_tokens} tokens")
        pieces.append(piece)
        remaining = remaining[take:]
    return pieces


def chunk_dialogue(
    dialog: list[dict[str, Any]],
    tokenizer: TextTokenizer,
    max_tokens: int,
) -> list[DialogueChunk]:
    """Pack utterances up to max_tokens while retaining utterance/session boundaries."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    chunks: list[DialogueChunk] = []
    current_lines: list[str] = []
    current_indices: list[int] = []

    def flush() -> None:
        if not current_lines:
            return
        text = "\n".join(current_lines)
        chunks.append(
            DialogueChunk(
                text=text,
                utterance_indices=tuple(current_indices),
                token_count=len(tokenizer.encode(text)),
            )
        )
        current_lines.clear()
        current_indices.clear()

    for utterance_index, utterance in enumerate(dialog):
        if not isinstance(utterance, dict) or "text" not in utterance:
            raise ValueError(f"dialog[{utterance_index}] must contain text")
        speaker = _speaker(utterance_index)
        line = f"{speaker}: {str(utterance['text']).strip()}"
        lines = (
            [line]
            if len(tokenizer.encode(line)) <= max_tokens
            else _split_long_utterance(utterance["text"], speaker, tokenizer, max_tokens)
        )
        for piece in lines:
            candidate = "\n".join([*current_lines, piece])
            if current_lines and len(tokenizer.encode(candidate)) > max_tokens:
                flush()
                candidate = piece
            if len(tokenizer.encode(candidate)) > max_tokens:
                raise AssertionError("Internal error: oversized MSC chunk")
            current_lines.append(piece)
            if not current_indices or current_indices[-1] != utterance_index:
                current_indices.append(utterance_index)
    flush()
    return chunks


def _qa_row(qa_id: str, task: str, question: str, answer: str) -> dict[str, str]:
    return {
        "id": qa_id,
        "task": task,
        "question": question,
        "answer": answer,
    }


def _facts_for_speaker(
    annotation_dialog: list[dict[str, Any]],
    utterance_indices: Iterable[int],
    speaker_index: int,
) -> list[str]:
    facts = []
    seen = set()
    for index in utterance_indices:
        if index % 2 != speaker_index or index >= len(annotation_dialog):
            continue
        fact = annotation_dialog[index].get("persona_text")
        if fact and str(fact).strip().casefold() not in seen:
            facts.append(str(fact).strip())
            seen.add(str(fact).strip().casefold())
    return facts


def _session_summary(
    annotation_dialog: list[dict[str, Any]],
    speaker_index: int,
) -> list[str]:
    for index in range(len(annotation_dialog) - 1, -1, -1):
        if index % 2 == speaker_index:
            facts = annotation_dialog[index].get("agg_persona_list", [])
            if isinstance(facts, list):
                return [str(fact).strip() for fact in facts if str(fact).strip()]
    return []


def build_chunk_qa(
    trajectory_id: str,
    session_index: int,
    chunk_index: int,
    chunk: DialogueChunk,
    dialog: list[dict[str, Any]],
    annotation: dict[str, Any] | None,
    qa_tasks: set[str],
    is_final_chunk: bool,
) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []
    base_id = f"{trajectory_id}:session:{session_index}:chunk:{chunk_index}"
    if "next_turn" in qa_tasks:
        for index in chunk.utterance_indices:
            if index == 0:
                continue
            previous = f"{_speaker(index - 1)}: {str(dialog[index - 1]['text']).strip()}"
            qa.append(
                _qa_row(
                    f"{base_id}:next_turn:{index}",
                    "next_turn",
                    NEXT_TURN_PROMPT.format(previous=previous, speaker=_speaker(index)),
                    str(dialog[index]["text"]).strip(),
                )
            )

    annotation_dialog = annotation.get("dialog", []) if annotation else []
    if "persona_extraction" in qa_tasks and isinstance(annotation_dialog, list):
        for speaker_index in (0, 1):
            facts = _facts_for_speaker(
                annotation_dialog,
                chunk.utterance_indices,
                speaker_index,
            )
            if facts:
                speaker = _speaker(speaker_index)
                qa.append(
                    _qa_row(
                        f"{base_id}:persona_extraction:{speaker_index + 1}",
                        "persona_extraction",
                        PERSONA_EXTRACTION_PROMPT.format(speaker=speaker),
                        "\n".join(facts),
                    )
                )

    if "persona_summary" in qa_tasks and is_final_chunk and isinstance(annotation_dialog, list):
        for speaker_index in (0, 1):
            facts = _session_summary(annotation_dialog, speaker_index)
            if facts:
                speaker = _speaker(speaker_index)
                qa.append(
                    _qa_row(
                        f"{base_id}:persona_summary:{speaker_index + 1}",
                        "persona_summary",
                        PERSONA_SUMMARY_PROMPT.format(speaker=speaker),
                        "\n".join(facts),
                    )
                )
    return qa


def convert_split(
    dialogue_root: Path,
    persona_root: Path | None,
    split: str,
    tokenizer: TextTokenizer,
    max_turn_tokens: int,
    qa_tasks: set[str],
) -> dict[str, Any]:
    snapshots = select_longest_snapshots(dialogue_root, split)
    annotations = load_persona_annotations(persona_root, split)
    streams = []
    turn_count = 0
    qa_count: dict[str, int] = {task: 0 for task in sorted(qa_tasks)}
    session_histogram: dict[str, int] = {}
    max_turn_tokens_observed = 0
    max_stream_tokens_observed = 0

    for trajectory_id in sorted(snapshots):
        snapshot = snapshots[trajectory_id]
        sessions = [*snapshot["previous_dialogs"], {"dialog": snapshot["dialog"]}]
        session_histogram[str(len(sessions))] = session_histogram.get(str(len(sessions)), 0) + 1
        turns = []
        stream_token_count = 0
        for session_index, session in enumerate(sessions):
            dialog = session.get("dialog", [])
            if not isinstance(dialog, list) or not dialog:
                continue
            chunks = chunk_dialogue(dialog, tokenizer, max_turn_tokens)
            annotation = annotations.get((trajectory_id, session_index))
            for chunk_index, chunk in enumerate(chunks):
                qa = build_chunk_qa(
                    trajectory_id=trajectory_id,
                    session_index=session_index,
                    chunk_index=chunk_index,
                    chunk=chunk,
                    dialog=dialog,
                    annotation=annotation,
                    qa_tasks=qa_tasks,
                    is_final_chunk=chunk_index + 1 == len(chunks),
                )
                for row in qa:
                    qa_count[row["task"]] += 1
                turn = {
                    "turn_id": f"{trajectory_id}:session:{session_index}:chunk:{chunk_index}",
                    "text": chunk.text,
                    "qa": qa,
                    "metadata": {
                        "session_index": session_index,
                        "chunk_index": chunk_index,
                        "chunks_in_session": len(chunks),
                        "token_count": chunk.token_count,
                    },
                }
                turns.append(turn)
                turn_count += 1
                stream_token_count += chunk.token_count
                max_turn_tokens_observed = max(max_turn_tokens_observed, chunk.token_count)
        max_stream_tokens_observed = max(max_stream_tokens_observed, stream_token_count)
        streams.append(
            {
                "stream_id": trajectory_id,
                "turns": turns,
                "metadata": {
                    "source": "Multi-Session Chat v0.1",
                    "split": split,
                    "session_count": len(sessions),
                    "token_count": stream_token_count,
                },
            }
        )

    return {
        "schema": SCHEMA_NAME,
        "dataset": {
            "name": "Multi-Session Chat",
            "version": "v0.1",
            "split": split,
            "max_turn_tokens": max_turn_tokens,
            "tokenizer": tokenizer.name,
            "input_policy": "longest_snapshot_per_initial_data_id",
            "excluded_from_input": [
                "personas",
                "init_personas",
                "newfact",
                "followup",
            ],
            "qa_tasks": sorted(qa_tasks),
        },
        "streams": streams,
        "facts": [],
        "statistics": {
            "streams": len(streams),
            "turns": turn_count,
            "trajectory_session_count": session_histogram,
            "max_turn_tokens_observed": max_turn_tokens_observed,
            "max_stream_tokens_observed": max_stream_tokens_observed,
            "qa": qa_count,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert official MSC v0.1 snapshots into ordered SHINE recurrent streams."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "valid", "test"], default=["train", "valid", "test"])
    parser.add_argument("--max-turn-tokens", type=int, default=2048)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Qwen tokenizer path/name for exact token limits; omitted uses an approximate whitespace counter.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--qa-tasks",
        nargs="*",
        choices=["next_turn", "persona_extraction", "persona_summary"],
        default=["next_turn", "persona_extraction", "persona_summary"],
        help="Store future supervision as QA. These records are ignored by reconstruction-only training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    dialogue_root = _find_task_root(args.input_dir, "msc_dialogue")
    persona_root = _find_task_root(args.input_dir, "msc_personasummary", required=False)
    if args.tokenizer:
        tokenizer: TextTokenizer = HuggingFaceTokenizer(args.tokenizer, args.trust_remote_code)
    else:
        LOGGER.warning(
            "No --tokenizer supplied; --max-turn-tokens uses approximate whitespace counts. "
            "Pass the Qwen3 tokenizer path for exact model-token limits."
        )
        tokenizer = WhitespaceTokenizer()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA_NAME, "splits": {}}
    for split in args.splits:
        payload = convert_split(
            dialogue_root=dialogue_root,
            persona_root=persona_root,
            split=split,
            tokenizer=tokenizer,
            max_turn_tokens=args.max_turn_tokens,
            qa_tasks=set(args.qa_tasks),
        )
        output_path = args.output_dir / f"msc_{split}.json"
        _write_json(output_path, payload)
        manifest["splits"][split] = {
            "path": output_path.name,
            **payload["statistics"],
        }
        LOGGER.info("Wrote %s: %s", output_path, payload["statistics"])
    _write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
