"""SQuAD single-passage rollout.

For a list of SquadContext items:
  1. Tokenize the passage once -> evidence_(ids, mask).
  2. Call ShineHypernet.generate_lora(...) -> lora dict.
  3. For each question (up to questions_per_context), build a chat message,
     run the LoRA-conditioned base model, extract the answer, compute reward
     against the gold references.
  4. Return the mean reward across all (context, question) pairs.

This is the inner loop that ES calls N*2 times per step for antithetic
evaluation.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from ..data.squad_contexts import SquadContext
from ..reward.base import RewardFn
from ..shine_adapter import ShineHypernet


def _ensure_shine_on_path() -> None:
    shine_root = Path(__file__).resolve().parents[2] / "third_party" / "SHINE"
    p = str(shine_root)
    if p not in sys.path:
        sys.path.insert(0, p)


_SYS_PROMPT = (
    "You are a helpful assistant. Answer the question concisely with short "
    "words or phrases. Answer the question directly and output nothing else. "
    "Never say you don't know the answer. Never enter think mode."
)


def _extract_answer(text: str) -> str:
    """Strip Qwen3's <think>...</think> block and common answer prefixes.

    Handles three completion shapes:
      * ``...<think>X</think>Y``  — model emitted its own thinking block.
      * ``X</think>Y``            — prompt prefilled ``<think>\\n`` (force_think),
                                    so only the closing tag appears in the
                                    completion.
      * ``Y``                     — no thinking, just the answer.
    """
    if "<think>" in text:
        rest = text.split("<think>", 1)[1]
        if "</think>" in rest:
            text = rest.split("</think>", 1)[1]
        else:
            text = ""
    elif "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"^(final answer|answer)\s*:\s*", "", text.strip(),
                  flags=re.IGNORECASE)
    return text.strip()


@dataclass
class SquadRolloutConfig:
    context_max_length: int = 1024
    question_max_length: int = 256
    max_new_tokens: int = 64
    questions_per_context: int = 4
    # See RLRolloutConfig.enable_thinking / force_think. Must match the
    # train-side values so eval prompts are tokenized identically.
    enable_thinking: bool | None = True
    force_think: bool = False


class SquadRollout:
    def __init__(
        self,
        hypernet: ShineHypernet,
        reward_fn: RewardFn,
        cfg: SquadRolloutConfig | None = None,
    ):
        self.hypernet = hypernet
        self.reward_fn = reward_fn
        self.cfg = cfg or SquadRolloutConfig()

    def _encode_context(self, context: str) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.hypernet.tokenizer
        enc = tok(
            context,
            max_length=self.cfg.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        return (
            enc["input_ids"].to(self.hypernet.device),
            enc["attention_mask"].to(self.hypernet.device),
        )

    def _encode_question(
        self, question: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.hypernet.tokenizer
        messages = [
            {"role": "system", "content": _SYS_PROMPT},
            {"role": "user", "content": question},
        ]
        extra = {} if self.cfg.enable_thinking is None else \
            {"enable_thinking": bool(self.cfg.enable_thinking)}
        # Render to text first so we can optionally append the force_think
        # prefix before tokenizing + padding.
        text = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **extra,
        )
        if self.cfg.force_think:
            text = text + "<think>\n"
        enc = tok(
            text,
            return_tensors="pt",
            max_length=self.cfg.question_max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=False,  # chat template already wrapped it
        )
        return (
            enc["input_ids"].to(self.hypernet.device),
            enc["attention_mask"].to(self.hypernet.device),
        )

    @torch.no_grad()
    def answer_single(
        self, loradict: dict, question: str
    ) -> str:
        input_ids, attention_mask = self._encode_question(question)
        out = self.hypernet.answer(
            loradict,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.cfg.max_new_tokens,
        )
        new_tokens = out[0, input_ids.shape[1]:]
        text = self.hypernet.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _extract_answer(text)

    @torch.no_grad()
    def score_context(self, ctx: SquadContext) -> list[float]:
        """Rollout all (or up to K) questions for a single context; return rewards."""
        evidence_ids, evidence_mask = self._encode_context(ctx.context)
        loradict = self.hypernet.generate_lora(evidence_ids, evidence_mask)

        rewards: list[float] = []
        for qa in ctx.qa[: self.cfg.questions_per_context]:
            pred = self.answer_single(loradict, qa.question)
            rewards.append(self.reward_fn(pred, qa.references))
        return rewards

    @torch.no_grad()
    def __call__(self, contexts: Sequence[SquadContext]) -> float:
        """Mean reward across all (context, question) pairs."""
        all_r: list[float] = []
        for ctx in contexts:
            all_r.extend(self.score_context(ctx))
        if not all_r:
            return 0.0
        return float(sum(all_r) / len(all_r))
