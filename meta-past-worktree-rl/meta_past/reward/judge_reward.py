"""LLM-as-judge binary-correctness reward.

Two transports:

1. ``HttpJudgeReward`` — HTTP POST to a judge server. Protocol matches
   ``Long-Digestor-Experiments/reward_server.py``:

       POST /evaluate
       { "question": ..., "reference": ..., "pred": ... }
       → { "result": "True" | "False" }

   The server itself can be anything behind that endpoint (OpenAI
   gpt-4o-mini, a local Qwen3-32B served by vLLM, etc.). Concurrent
   calls via a thread pool — ~1k samples per training step finish in
   seconds rather than minutes.

2. ``SyncJudgeReward`` — direct OpenAI Chat Completions calls, one at
   a time. Kept for ES / batch eval where call volume is low.

Both expose ``__call__(pred, references, *, question) -> float`` and
a ``batch_compute(preds, refs_list, *, questions) -> list[float]`` hook
the rollout uses for batched concurrency.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


# Default judge prompt. Mirrors Long-Digestor-Experiments/prompts.py exactly
# so the same server can be reused across both projects.
JUDGE_SYSTEM_PROMPT = (
    "You are a precise evaluator. Your task is to determine if the "
    "'Predicted Answer' is semantically the same as the 'Ground Truth' "
    "for the given 'Question'. Your entire response MUST be only the "
    "single word 'True' or the single word 'False'. Do not provide any "
    "explanation or punctuation."
)
JUDGE_USER_TEMPLATE = (
    "Question: {question}\nGround Truth: {reference}\nPredicted Answer: {pred}"
)


def _parse_judge_text(text: str) -> float:
    """Long-Digestor-style verdict parsing: True → 1.0, False → 0.0."""
    s = text.strip().lower()
    if s.startswith("true"):
        return 1.0
    if s.startswith("false"):
        return 0.0
    # vintage 0/1 tokens still accepted for back-compat with the older
    # prompt below.
    for tok in s.split():
        cleaned = tok.strip(".,()<>[]`")
        if cleaned in ("0", "1"):
            return float(cleaned)
    return 0.0


# --- HTTP judge (recommended for RL training) -------------------------------


@dataclass
class HttpJudgeReward:
    """Judge reward that POSTs to a Long-Digestor-style ``/evaluate`` server.

    Bring up the server first (cheapest: copy
    ``Long-Digestor-Experiments/reward_server.py`` and point it at OpenAI
    or a local vllm-served Qwen3-32B). Then in this project's yaml:

        reward:
          type: judge
          judge_url: http://127.0.0.1:8124
          judge_concurrency: 32
          judge_timeout_s: 30

    Returns 0.0 on persistent network failure (after ``max_retries``)
    rather than crashing — gradient signal degrades gracefully if the
    judge server hiccups.
    """

    base_url: str = "http://127.0.0.1:8124"
    timeout_s: float = 30.0
    max_retries: int = 3
    initial_retry_delay_s: float = 2.0
    backoff: float = 2.0
    concurrency: int = 32
    # Best-effort flag: if True, single-prompt calls also use the executor
    # so they don't block the calling thread (still synchronous wrt caller).
    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # requests is already a transitive dep of vllm/datasets; no extra install.
        import requests  # noqa: F401  (validate availability)
        self._evaluate_url = self.base_url.rstrip("/") + "/evaluate"
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(self.concurrency)),
            thread_name_prefix="judge-rew",
        )

    def __call__(
        self, pred: str, references: Sequence[str], *, question: str = "",
    ) -> float:
        # Use the first reference as the "ground truth" the way the
        # Long-Digestor server expects. Servers that look at multiple refs
        # can iterate the response themselves; for binary correctness we
        # pick the canonical one.
        ref = references[0] if references else ""
        return self._call_one(question=question, reference=ref, pred=pred)

    def batch_compute(
        self,
        preds: Sequence[str],
        references_list: Sequence[Sequence[str]],
        *,
        questions: Sequence[str] | None = None,
    ) -> list[float]:
        if questions is None:
            questions = [""] * len(preds)
        if len(questions) != len(preds):
            raise ValueError(
                f"questions length {len(questions)} != preds length {len(preds)}"
            )
        # Submit all calls to the executor; futures resolve concurrently.
        assert self._executor is not None
        refs_first = [r[0] if r else "" for r in references_list]
        futures = [
            self._executor.submit(
                self._call_one, question=q, reference=ref, pred=p,
            )
            for p, ref, q in zip(preds, refs_first, questions)
        ]
        return [f.result() for f in futures]

    def _call_one(self, *, question: str, reference: str, pred: str) -> float:
        import requests
        payload = {"question": question, "reference": reference, "pred": pred}
        delay = self.initial_retry_delay_s
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    self._evaluate_url,
                    json=payload,
                    timeout=self.timeout_s,
                )
                r.raise_for_status()
                data = r.json()
                return _parse_judge_text(str(data.get("result", "False")))
            except (
                requests.exceptions.RequestException,
                json.JSONDecodeError,
                ValueError,
            ) as e:
                if attempt >= self.max_retries:
                    logging.getLogger(__name__).warning(
                        "Judge call failed after %d attempts (%s). "
                        "Defaulting reward to 0.",
                        self.max_retries, e,
                    )
                    return 0.0
                time.sleep(delay)
                delay *= self.backoff
        return 0.0

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


# --- Direct OpenAI judge (legacy / batch eval) ------------------------------


_SYSTEM = (
    "You are a strict grader for extractive question answering. "
    "Given the context, question, one or more reference answers, and a "
    "candidate answer, decide if the candidate is correct. A candidate is "
    "correct when it matches the semantic content of any reference answer, "
    "ignoring punctuation, articles, and minor paraphrasing. If the candidate "
    "contains extra incorrect content beyond the reference, it is wrong. "
    "Respond with exactly one token: 1 for correct or 0 for incorrect."
)


def _user_message(context: str, question: str, references: Sequence[str],
                  prediction: str) -> str:
    refs_block = "\n".join(f"- {r}" for r in references)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Reference answer(s):\n{refs_block}\n\n"
        f"Candidate answer: {prediction}\n\n"
        f"Respond with 1 if the candidate is correct, else 0."
    )


def _parse_verdict(text: str) -> float:
    """Pull the first 0/1 token out of the response, defaulting to 0."""
    for token in text.strip().split():
        cleaned = token.strip(".,()<>[]`")
        if cleaned in {"0", "1"}:
            return float(cleaned)
    return 0.0


@dataclass
class SyncJudgeReward:
    """Callable: (pred, references) -> {0.0, 1.0} via sync Chat Completions.

    Caller supplies ``context`` + ``question`` at bind time because the
    ``RewardFn`` protocol only passes (pred, refs). Wrap this per-(ctx, q)
    inside the rollout rather than reusing one instance across questions.
    """
    context: str
    question: str
    model: str = "gpt-4.1"
    temperature: float = 0.0
    max_retries: int = 3
    client: object | None = None

    def __post_init__(self) -> None:
        if OpenAI is None:
            raise RuntimeError(
                "openai package not installed. pip install openai>=1.0"
            )
        if self.client is None:
            self.client = OpenAI()

    def __call__(
        self, pred: str, references: Sequence[str], *, question: str = "",
    ) -> float:
        # Use the bound question/context unless an override is passed
        # (kept for backward compat — earlier callers wrapped per-(ctx, q)).
        q = question or self.question
        msgs = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _user_message(
                self.context, q, references, pred
            )},
        ]
        delay = 1.0
        for _ in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=2,
                )
                return _parse_verdict(resp.choices[0].message.content or "")
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "judge call failed, retrying in %.1fs: %s", delay, e
                )
                time.sleep(delay)
                delay *= 2
        return 0.0

    def batch_compute(
        self,
        preds: Sequence[str],
        references_list: Sequence[Sequence[str]],
        *,
        questions: Sequence[str] | None = None,
    ) -> list[float]:
        # Sequential — OpenAI sync clients aren't thread-safe across calls
        # and ES use cases have small N. For high-throughput RL use
        # ``HttpJudgeReward`` instead.
        if questions is None:
            questions = [self.question] * len(preds)
        return [
            float(self(p, r, question=q))
            for p, r, q in zip(preds, references_list, questions)
        ]


# --- Batch API path (bulk pre-judging) ---------------------------------------


def build_batch_jsonl(
    items: Sequence[dict],
    model: str = "gpt-4.1",
) -> str:
    """Render a list of {id, context, question, refs, pred} into OpenAI's
    Batch API JSONL format. Returns the serialized JSONL text; caller writes
    it to a file and uploads via ``client.files.create(purpose='batch')``.
    """
    lines = []
    for item in items:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _user_message(
                    item["context"], item["question"],
                    item["refs"], item["pred"],
                )},
            ],
            "max_tokens": 2,
            "temperature": 0.0,
        }
        lines.append(json.dumps({
            "custom_id": item["id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }))
    return "\n".join(lines)


def submit_batch(jsonl_path: str, *, completion_window: str = "24h") -> str:
    """Upload JSONL and start a batch job. Returns the batch id."""
    if OpenAI is None:
        raise RuntimeError("openai not installed.")
    client = OpenAI()
    file_obj = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    return batch.id


def collect_batch_verdicts(batch_id: str) -> dict[str, float]:
    """Poll a batch until complete, then parse verdicts into {custom_id: 0/1}."""
    if OpenAI is None:
        raise RuntimeError("openai not installed.")
    client = OpenAI()
    while True:
        b = client.batches.retrieve(batch_id)
        if b.status in {"completed", "failed", "cancelled", "expired"}:
            break
        time.sleep(30)
    if b.status != "completed":
        raise RuntimeError(f"Batch {batch_id} ended with status {b.status}")

    out_file = client.files.content(b.output_file_id)
    verdicts: dict[str, float] = {}
    for raw in out_file.text.splitlines():
        rec = json.loads(raw)
        body = rec.get("response", {}).get("body", {})
        choices = body.get("choices") or []
        text = choices[0]["message"]["content"] if choices else ""
        verdicts[rec["custom_id"]] = _parse_verdict(text)
    return verdicts
