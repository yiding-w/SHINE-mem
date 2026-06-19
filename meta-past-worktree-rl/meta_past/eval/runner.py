"""vLLM-backed batched runner for the three eval modes.

The whole point of routing through vLLM is **continuous batching**: for
N eval items, all N prompts (or N LoRA-bound prompts) go into one
``LLM.generate`` call, and vLLM schedules them across attention
batches automatically. That gives ~order-of-magnitude speedups over
the per-item ``hypernet.answer`` loop ``SquadRollout`` uses for
training-time heldout (which generates one prompt at a time).

Three entry points:

  - ``run_shine(items)`` — hypernet builds one LoRA per item, ring-buffer
    of size ``max_loras`` cycled across chunks, all prompts in one
    generate call per chunk.
  - ``run_icl(items)`` — pack each item's ``context`` into the prompt
    and run them all through base Qwen3 in one big generate call. No
    LoRA. This is the bucket-B "vanilla ICL" baseline.
  - ``run_zero(items)`` — like ``run_icl`` but with no context — just
    the question. Side-effect probe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch

from ..rl.lora_format import (
    QWEN3_TARGET_MODULES,
    peft_meta_for_qwen3,
    shine_loradict_to_peft_batch,
)
from ..rl.vllm_engine import VLLMEngine
from ..shine_adapter import ShineHypernet
from .items import EvalItem
from .prompts import (
    extract_answer,
    format_icl_prompt,
    format_shine_prompt,
    format_zero_prompt,
)


logger = logging.getLogger("meta_past.eval.runner")


@dataclass
class EvalRunnerConfig:
    # vLLM
    max_prompt_tokens: int = 7500      # safety margin under vLLM's
                                       # max_model_len=8192; we trim icl
                                       # prompts that pack many demos.
    max_loras: int = 8                 # ring-buffer + hypernet chunk size.
                                       # 8 matches training (B_local=8 on 8-GPU
                                       # DDP), known to fit on 80GB with
                                       # gpu_memory_utilization=0.40. Larger
                                       # OOMs in the hypernet forward.
    max_new_tokens: int = 128
    temperature: float = 0.0           # greedy by default for eval reproducibility
    top_p: float = 1.0
    # Hypernet
    context_max_length: int = 1024
    # Qwen3 chat-template thinking mode (stock template). ``True`` (default) /
    # ``None`` → no stub; model decides. ``False`` → inject empty
    # <think></think> stub, forcing direct answer. Must match what the model
    # was trained with.
    enable_thinking: bool | None = True
    # If True, append "<think>\n" to the prompt so completions start mid-
    # thinking. Must match what the model was trained with.
    force_think: bool = False
    # Hypernet forward microbatching (HBM-bound on long contexts).
    # 2 matches training; the M2P FFN intermediate is the bottleneck.
    hypernet_microbatch_contexts: int = 2


class EvalRunner:
    def __init__(self, hypernet: ShineHypernet, engine: VLLMEngine,
                 cfg: EvalRunnerConfig):
        self.hypernet = hypernet
        self.engine = engine
        self.cfg = cfg
        self._peft_meta = peft_meta_for_qwen3(
            lora_r=hypernet.lora_r,
            target_modules=QWEN3_TARGET_MODULES,
        )
        # Monotonic LoRA id allocator. vLLM reserves 0; we start at 1.
        # Ring-buffer naturally evicts via vLLM's own LRU cache.
        self._next_lora_id = 1

    # -- helpers ---------------------------------------------------------------

    def _encode_contexts(
        self, contexts: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.hypernet.tokenizer
        enc = tok(
            list(contexts),
            max_length=self.cfg.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        return (
            enc["input_ids"].to(self.hypernet.device),
            enc["attention_mask"].to(self.hypernet.device),
        )

    def _alloc_lora_ids(self, n: int) -> list[int]:
        ids = list(range(self._next_lora_id, self._next_lora_id + n))
        self._next_lora_id += n
        return ids

    # -- generation paths ------------------------------------------------------

    @torch.no_grad()
    def run_shine(self, items: Sequence[EvalItem]) -> list[str]:
        """Hypernet builds one LoRA per item; push in chunks of
        ``max_loras``; one batched ``LLM.generate`` per chunk.
        """
        if not items:
            return []
        tok = self.hypernet.tokenizer
        cfg = self.cfg
        preds: list[str] = [""] * len(items)

        # Wake engine once for the whole eval pass.
        self.engine.wake_up(tags=["weights", "kv_cache"])

        chunk_size = max(1, cfg.max_loras)
        for start in range(0, len(items), chunk_size):
            chunk = items[start:start + chunk_size]
            B = len(chunk)

            # 1. Hypernet forward → batched loradict.
            ev_ids, ev_mask = self._encode_contexts([it.context for it in chunk])
            # Microbatch the hypernet forward if the chunk is big.
            mb = cfg.hypernet_microbatch_contexts
            if mb <= 0 or mb >= B:
                loradict = self.hypernet.generate_lora(ev_ids, ev_mask)
            else:
                # generate_lora has no built-in microbatching; do it manually
                # by concatenating per-MB dicts along the batch dim.
                slices: list[dict] = []
                for s in range(0, B, mb):
                    e = min(s + mb, B)
                    slices.append(self.hypernet.generate_lora(
                        ev_ids[s:e], ev_mask[s:e]
                    ))
                # Concat each leaf tensor along dim 0 (Lb / B axis).
                loradict = _concat_loradict(slices)

            # 2. Split → per-context PEFT dicts → push to vLLM (CPU copy).
            per_b = shine_loradict_to_peft_batch(loradict)
            assert len(per_b) == B, f"got {len(per_b)} dicts, expected {B}"
            per_b_cpu = [
                {k: v.detach().to("cpu", copy=True) for k, v in d.items()}
                for d in per_b
            ]
            lora_ids = self._alloc_lora_ids(B)

            if torch.cuda.is_available() and self.hypernet.device.type == "cuda":
                torch.cuda.empty_cache()
            self.engine.push_lora_batch(per_b_cpu, lora_ids, self._peft_meta)

            # 3. Build prompts (one per item, LoRA-bound) and generate.
            prompts = [
                format_shine_prompt(tok, it.question,
                                    enable_thinking=cfg.enable_thinking,
                                    force_think=cfg.force_think)
                for it in chunk
            ]
            out = self.engine.complete(
                prompts=prompts,
                lora_ids=lora_ids,
                n=1,
                temperature=cfg.temperature,
                max_tokens=cfg.max_new_tokens,
                top_p=cfg.top_p,
            )

            for i, samples in enumerate(out):
                preds[start + i] = extract_answer(samples[0].text)

            logger.info("[shine] chunk %d-%d / %d done",
                        start, start + B, len(items))

        return preds

    def _maybe_trim_icl_prompt(self, prompt: str) -> str:
        """If a packed-demos prompt exceeds ``max_prompt_tokens``,
        head-truncate it (keep the tail, which has the held-out
        question). This protects K-sweep eval from outlier-long demos
        crashing vLLM with 'decoder prompt longer than max_model_len'.
        """
        tok = self.hypernet.tokenizer
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        cap = int(self.cfg.max_prompt_tokens)
        if len(ids) <= cap:
            return prompt
        kept = ids[-cap:]
        return tok.decode(kept, skip_special_tokens=False)

    def run_icl(self, items: Sequence[EvalItem]) -> list[str]:
        """No LoRA. One big batched generate over all items."""
        if not items:
            return []
        tok = self.hypernet.tokenizer
        cfg = self.cfg
        self.engine.wake_up(tags=["weights", "kv_cache"])
        prompts = [
            self._maybe_trim_icl_prompt(
                format_icl_prompt(tok, it.context, it.question,
                                  enable_thinking=cfg.enable_thinking,
                                  force_think=cfg.force_think)
            )
            for it in items
        ]
        out = self.engine.complete(
            prompts=prompts,
            lora_ids=None,              # base model only
            n=1,
            temperature=cfg.temperature,
            max_tokens=cfg.max_new_tokens,
            top_p=cfg.top_p,
        )
        return [extract_answer(samples[0].text) for samples in out]

    def run_zero(self, items: Sequence[EvalItem]) -> list[str]:
        """No LoRA, no context. Strips ``it.context`` from the prompt."""
        if not items:
            return []
        tok = self.hypernet.tokenizer
        cfg = self.cfg
        self.engine.wake_up(tags=["weights", "kv_cache"])
        prompts = [
            format_zero_prompt(tok, it.question,
                               enable_thinking=cfg.enable_thinking,
                               force_think=cfg.force_think)
            for it in items
        ]
        out = self.engine.complete(
            prompts=prompts,
            lora_ids=None,
            n=1,
            temperature=cfg.temperature,
            max_tokens=cfg.max_new_tokens,
            top_p=cfg.top_p,
        )
        return [extract_answer(samples[0].text) for samples in out]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _concat_loradict(dicts: Sequence[dict]) -> dict:
    """Concat each leaf tensor along the batch dim (dim 0)."""
    if not dicts:
        return {}
    if len(dicts) == 1:
        return dicts[0]
    keys = dicts[0].keys()
    out: dict = {}
    for k in keys:
        leaves = [d[k] for d in dicts]
        if isinstance(leaves[0], torch.Tensor):
            out[k] = torch.cat(leaves, dim=0)
        elif isinstance(leaves[0], dict):
            out[k] = _concat_loradict(leaves)
        else:
            # Scalar / non-tensor; assume all equal.
            out[k] = leaves[0]
    return out
