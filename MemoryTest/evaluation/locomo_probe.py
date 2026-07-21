"""Small, deterministic LoCoMo probe used during recurrent SHINE training.

The full LoCoMo benchmark is deliberately expensive.  This module selects a
fixed, category-balanced subset from one conversation so that the same
questions can be evaluated at every checkpoint.  It implements the public
LoCoMo short-answer protocol locally; no delta-Mem runtime is required.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import string
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from nltk.stem import PorterStemmer
except ImportError:  # pragma: no cover - stemming is optional
    PorterStemmer = None


CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

OFFICIAL_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant whose job is to understand "
    "the following conversation and answer questions based on the conversation. "
    "If you don't know the answer to a question, please don't share false information."
)

OFFICIAL_QA_PROMPT = (
    "Based on the above conversations, write a short answer for the following question "
    "in a few words. Do not write complete and lengthy sentences. "
    "Answer with exact words from the conversations whenever possible.\n\n"
    "Question: {}"
)

_STEMMER = PorterStemmer() if PorterStemmer is not None else None
_EVIDENCE_SESSION_RE = re.compile(r"^[dD](\d+):")


@dataclass(frozen=True)
class LoCoMoQuestionSpec:
    prompt_text: str
    category: int
    option_answers: dict[str, str] | None = None


def load_locomo_sample(path: str | Path, sample_index: int) -> dict:
    path = Path(path)
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"LoCoMo file must contain a non-empty JSON list: {path}")
    if not -len(samples) <= sample_index < len(samples):
        raise IndexError(
            f"--locomo-eval-sample-index={sample_index} is outside a file with "
            f"{len(samples)} conversations"
        )
    sample = samples[sample_index]
    for key in ("sample_id", "conversation", "qa"):
        if key not in sample:
            raise KeyError(f"LoCoMo sample {sample_index} is missing {key!r}")
    return sample


def session_numbers(sample: dict) -> list[int]:
    conversation = sample["conversation"]
    numbers = []
    for key, value in conversation.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        suffix = key.removeprefix("session_")
        if suffix.isdigit() and isinstance(value, list) and value:
            numbers.append(int(suffix))
    return sorted(numbers)


def render_turn(dialog: dict) -> str:
    turn = f'{dialog["speaker"]} said, "{dialog["text"]}"\n'
    if dialog.get("blip_caption"):
        turn += f' and shared {dialog["blip_caption"]}.'
    return turn + "\n"


def build_session_texts(sample: dict) -> list[dict]:
    conversation = sample["conversation"]
    sessions = []
    for session_num in session_numbers(sample):
        session_key = f"session_{session_num}"
        date_key = f"{session_key}_date_time"
        if date_key not in conversation:
            raise KeyError(f"LoCoMo sample is missing {date_key!r}")
        turns = "".join(render_turn(dialog) for dialog in conversation[session_key]).rstrip()
        sessions.append(
            {
                "session_number": session_num,
                "text": f"DATE: {conversation[date_key]}\nCONVERSATION:\n{turns}",
            }
        )
    if not sessions:
        raise ValueError(f"LoCoMo sample {sample.get('sample_id')!r} has no non-empty sessions")
    return sessions


def _stable_seed(base_seed: int, sample_id: str, category: int) -> int:
    digest = hashlib.sha256(f"{base_seed}|{sample_id}|{category}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def select_probe_questions(
    sample: dict,
    categories: list[int] | tuple[int, ...],
    questions_per_category: int,
    seed: int,
    allowed_session_numbers: set[int] | None = None,
) -> list[dict]:
    """Select a stable category-balanced subset while retaining source indices."""
    selected = []
    for category in categories:
        candidates = [
            {**question, "_question_index": index}
            for index, question in enumerate(sample["qa"])
            if int(question["category"]) == int(category)
            and _question_is_supported_by_window(question, allowed_session_numbers)
        ]
        rng = random.Random(_stable_seed(seed, str(sample["sample_id"]), int(category)))
        rng.shuffle(candidates)
        if questions_per_category > 0:
            candidates = candidates[:questions_per_category]
        selected.extend(candidates)
    return selected


def _question_evidence_sessions(question: dict) -> list[int]:
    sessions = []
    for evidence in question.get("evidence", []):
        match = _EVIDENCE_SESSION_RE.match(str(evidence))
        if match:
            sessions.append(int(match.group(1)))
    return sessions


def _question_is_supported_by_window(
    question: dict,
    allowed_session_numbers: set[int] | None,
) -> bool:
    if allowed_session_numbers is None:
        return True
    evidence_sessions = _question_evidence_sessions(question)
    return bool(evidence_sessions) and set(evidence_sessions).issubset(allowed_session_numbers)


def select_probe_session_window(
    sample: dict,
    categories: list[int] | tuple[int, ...],
    max_sessions: int,
) -> list[int]:
    """Choose the contiguous window with the best minimum category coverage."""
    numbers = session_numbers(sample)
    if max_sessions >= len(numbers):
        return numbers
    candidates = []
    for start_index in range(len(numbers) - max_sessions + 1):
        window = numbers[start_index : start_index + max_sessions]
        allowed = set(window)
        counts = {
            int(category): sum(
                int(question["category"]) == int(category)
                and _question_is_supported_by_window(question, allowed)
                for question in sample["qa"]
            )
            for category in categories
        }
        # Prefer balanced category support, then more questions, then earlier history.
        score = (min(counts.values()), sum(counts.values()), -window[0])
        candidates.append((score, window))
    return max(candidates, key=lambda item: item[0])[1]


def prepare_question(question: dict, sample_id: str, seed: int) -> LoCoMoQuestionSpec:
    category = int(question["category"])
    if category == 2:
        return LoCoMoQuestionSpec(
            prompt_text=(
                question["question"]
                + " Use DATE of CONVERSATION to answer with an approximate date."
            ),
            category=category,
        )
    if category == 5:
        distractor = str(question.get("answer", question.get("adversarial_answer", "No information available")))
        payload = f"{seed}|{sample_id}|{question['_question_index']}|cat5"
        flip = (int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % 2) == 0
        options = (
            {"a": "No information available", "b": distractor}
            if flip
            else {"a": distractor, "b": "No information available"}
        )
        return LoCoMoQuestionSpec(
            prompt_text=(
                question["question"]
                + f" (a) {options['a']} (b) {options['b']}. "
                + "Select the correct answer by writing (a) or (b)."
            ),
            category=category,
            option_answers=options,
        )
    return LoCoMoQuestionSpec(prompt_text=question["question"], category=category)


def build_question_messages(spec: LoCoMoQuestionSpec) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
        {"role": "user", "content": OFFICIAL_QA_PROMPT.format(spec.prompt_text)},
    ]


def canonicalize_prediction(raw_prediction: str, spec: LoCoMoQuestionSpec) -> str:
    prediction = raw_prediction.replace('\\"', "'").strip()
    lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    if lines:
        prediction = lines[0]
    lowered = prediction.lower()
    if spec.category == 5 and spec.option_answers is not None:
        if "(a)" in lowered or lowered == "a" or lowered.startswith("a)"):
            return spec.option_answers["a"]
        if "(b)" in lowered or lowered == "b" or lowered.startswith("b)"):
            return spec.option_answers["b"]
        if "no information available" in lowered or "not mentioned" in lowered:
            return "No information available"
    return (
        lowered.replace("(a)", "")
        .replace("(b)", "")
        .replace("a)", "")
        .replace("b)", "")
        .replace("answer:", "")
        .strip()
    )


def normalize_answer(text: str) -> str:
    text = text.replace(",", "")
    normalized = unicodedata.normalize("NFD", text).lower()
    normalized = "".join(char for char in normalized if char not in set(string.punctuation))
    normalized = re.sub(r"\b(a|an|the|and)\b", " ", normalized)
    return " ".join(normalized.split())


def _stem_tokens(text: str) -> list[str]:
    tokens = normalize_answer(text).split()
    return tokens if _STEMMER is None else [_STEMMER.stem(token) for token in tokens]


def single_answer_f1(prediction: str, answer: str) -> float:
    prediction_tokens = _stem_tokens(prediction)
    answer_tokens = _stem_tokens(answer)
    if not prediction_tokens or not answer_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(answer_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(question: dict, prediction: str) -> float:
    category = int(question["category"])
    if category == 5:
        normalized = prediction.lower()
        return 1.0 if ("no information available" in normalized or "not mentioned" in normalized) else 0.0
    answer = str(question["answer"])
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category == 1:
        predictions = [item.strip() for item in prediction.split(",") if item.strip()]
        answers = [item.strip() for item in answer.split(",") if item.strip()]
        if not predictions or not answers:
            return 0.0
        return sum(max(single_answer_f1(candidate, gold) for candidate in predictions) for gold in answers) / len(answers)
    return single_answer_f1(prediction, answer)


def evidence_distance(question: dict, final_session_number: int) -> int | None:
    evidence_sessions = _question_evidence_sessions(question)
    if not evidence_sessions:
        return None
    return max(0, final_session_number - max(evidence_sessions))


def distance_bucket(distance: int | None) -> str | None:
    if distance is None:
        return None
    if distance <= 2:
        return "near_0_2"
    if distance <= 9:
        return "middle_3_9"
    return "far_10_plus"


def summarize_records(records: list[dict], condition_names: list[str]) -> dict:
    conditions = {}
    for condition_name in condition_names:
        category_scores = defaultdict(float)
        category_counts = defaultdict(int)
        distance_scores = defaultdict(float)
        distance_counts = defaultdict(int)
        total_score = 0.0
        total = 0
        for record in records:
            result = record["conditions"].get(condition_name)
            if result is None:
                continue
            score = float(result["score"])
            category = int(record["category"])
            bucket = record.get("evidence_distance_bucket")
            total_score += score
            total += 1
            category_scores[category] += score
            category_counts[category] += 1
            if bucket is not None:
                distance_scores[bucket] += score
                distance_counts[bucket] += 1
        category_summary = {
            str(category): {
                "name": CATEGORY_NAMES.get(category, "unknown"),
                "score": category_scores[category] / category_counts[category],
                "count": category_counts[category],
            }
            for category in sorted(category_counts)
        }
        conditions[condition_name] = {
            "overall_score": 0.0 if total == 0 else total_score / total,
            "macro_category_score": (
                0.0
                if not category_summary
                else sum(value["score"] for value in category_summary.values())
                / len(category_summary)
            ),
            "num_questions": total,
            "category_scores": category_summary,
            "distance_scores": {
                bucket: {
                    "score": distance_scores[bucket] / distance_counts[bucket],
                    "count": distance_counts[bucket],
                }
                for bucket in ("near_0_2", "middle_3_9", "far_10_plus")
                if distance_counts[bucket]
            },
        }
    return conditions
