"""Per-rank batched RL rollout: B/N contexts → B/N LoRAs → local vLLM sampling.

Used inside a HybridEngine (verl-style) where each DDP rank co-locates a
training process and a ``vllm.LLM(enable_sleep_mode=True)`` on the same GPU.
The rollout runs on a per-rank batch slice (``contexts_per_step / world_size``
contexts per rank) and uses the rank-local ``VLLMEngine`` for sampling.

Per-step pipeline (each rank, in lockstep):

  1. Tokenize this rank's slice of contexts into ``[B_local, T_ctx]``.
  2. ``hypernet.generate_lora_grad(...)`` → loradict with ``Lb = B_local``,
     autograd-live for the eventual backward.
  3. ``engine.wake_up`` → push LoRAs in-process via
     ``collective_rpc(register, ...)`` → sample ``B_local*Q`` prompts × K
     completions → ``engine.sleep`` (frees vLLM weights from GPU).
  4. Re-tokenize completions; truncate at first OOB-vocab id (vLLM's
     151,936-row LM head can sample reserved tokens beyond SHINE's
     151,672-row embedding).
  5. Pack a context-major ``[B_local*Q*K, T_max]`` rescore batch and
     return it to the trainer (rescore + DDP all-reduce live in trainer).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import torch

from ..data.squad_contexts import SquadContext
from ..reward.base import RewardFn
from ..rollout.squad_rollout import _SYS_PROMPT, _extract_answer
from ..shine_adapter import ShineHypernet
from . import set_phase
from .lora_format import (
    QWEN3_TARGET_MODULES,
    peft_meta_for_qwen3,
    shine_loradict_to_peft_batch,
)
from .vllm_engine import VLLMEngine


@dataclass
class RolloutGroup:
    """Data needed to compute the policy loss, with the rescore deferred.

    Joint-microbatch architecture:
    ``rollout_step`` runs the hypernet under ``no_grad`` to generate a
    loradict, pushes it to vLLM, samples completions, and packs the
    rescore inputs. The loradict is **discarded** afterwards — the
    trainer re-runs the hypernet *with grad* per microbatch, immediately
    followed by the rescore + backward for that microbatch's samples,
    so each chunk's full autograd graph is released before the next
    chunk starts. This bounds peak memory by chunk size, independent of
    the global ``contexts_per_step``.

    Cost: one extra (no_grad) hypernet forward per step. Cheap relative
    to the rescore + backward.
    """
    rewards: torch.Tensor          # [N]
    response_mask: torch.Tensor    # [N, T-1] 1 on answer positions
    group_ids: torch.Tensor        # [N] — GRPO group id per sample
    # Deferred-rescore inputs (no loradict — trainer re-forwards hypernet):
    evidence_ids: torch.Tensor     # [B_local, T_ctx] for hypernet re-forward
    evidence_mask: torch.Tensor    # [B_local, T_ctx]
    input_ids: torch.Tensor        # [N, T_max] long
    attention_mask: torch.Tensor   # [N, T_max] long
    answer_mask: torch.Tensor      # [N, T_max] long (target positions)
    meta: dict = field(default_factory=dict)


@dataclass
class RLRolloutConfig:
    context_max_length: int = 1024
    question_max_length: int = 256
    max_new_tokens: int = 64
    temperature: float = 1.0
    contexts_per_step: int = 8        # B
    questions_per_context: int = 4    # Q
    rollouts_per_question: int = 16   # K
    use_gradient_checkpoint: bool = True
    # Ring-buffer window for LoRA IDs sent to vLLM. Must be >= contexts_per_step
    # (default = 4 * B so older steps can coexist with the current one in the
    # LRU cache without immediate eviction churn).
    lora_id_window: int = 0           # 0 → resolve to 4*B at init
    # Microbatch sizes (in contexts) for hypernet forward and re-score forward.
    # 0 = run the whole batch at once (no chunking). Pick the largest value
    # that fits HBM. Hypernet bottleneck is the M2P FFN intermediate
    # (~1.4 GB / chunk-of-8 / layer); rescore bottleneck is Qwen3-8B forward
    # over chunk * Q * K sequences. Both retain autograd graphs across chunks
    # for one backward at the end — savings are on peak working memory, not
    # stored activations.
    hypernet_microbatch_contexts: int = 0
    rescore_microbatch_contexts: int = 0
    # Qwen3 chat-template thinking mode (stock template, post-shine-override
    # removal). ``True`` (default) / ``None`` → no stub injected; model decides
    # whether to emit <think>...</think>. ``False`` → template inserts an empty
    # <think></think> stub, forcing direct-answer mode (use when
    # ``max_new_tokens`` is too small to fit thinking + answer).
    enable_thinking: bool | None = True
    # Force the model to start every completion inside a <think> block by
    # appending ``<think>\n`` to the prompt after apply_chat_template. The
    # model then must close </think> before producing the answer. Only
    # meaningful when ``enable_thinking`` is True/None (the False stub is a
    # closed <think></think> pair, so a prefilled opener would be redundant).
    force_think: bool = False


class RLRollout:
    def __init__(
        self,
        hypernet: ShineHypernet,
        reward_fn: RewardFn,
        engine: VLLMEngine,
        cfg: RLRolloutConfig,
    ):
        self.hypernet = hypernet
        self.reward_fn = reward_fn
        self.engine = engine
        self.cfg = cfg
        if cfg.lora_id_window <= 0:
            self._lora_window = max(1, 4 * cfg.contexts_per_step)
        else:
            self._lora_window = int(cfg.lora_id_window)
        self._peft_meta = peft_meta_for_qwen3(
            lora_r=hypernet.lora_r,
            target_modules=QWEN3_TARGET_MODULES,
        )
        # vLLM's Qwen3-8B LM head has 151,936 output rows (hardware-aligned),
        # but SHINE resized its input embedding / lm_head to len(tokenizer)
        # = 151,672. Tokens sampled by vLLM with IDs in [151672, 151936) —
        # mostly reserved special tokens like <|fim_pad|> — cannot be
        # embedded by SHINE's rescore forward. We truncate completions at
        # the first such ID and treat it as an early EOS.
        self._embed_rows = (
            hypernet.metanetwork.metamodel.get_input_embeddings().num_embeddings
        )
        # Token ids for the "<think>\n" force_think prefix. Cached once; for
        # Qwen3 this is exactly [151667, 198] (single special-token + newline).
        self._think_prefix_ids: list[int] = list(
            hypernet.tokenizer.encode("<think>\n", add_special_tokens=False)
        )

    # -- encoding helpers ------------------------------------------------------

    def _encode_contexts(
        self, contexts: Sequence[SquadContext]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.hypernet.tokenizer
        texts = [c.context for c in contexts]
        enc = tok(
            texts,
            max_length=self.cfg.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        return (
            enc["input_ids"].to(self.hypernet.device),
            enc["attention_mask"].to(self.hypernet.device),
        )

    def _chat_kwargs(self) -> dict:
        """Extra kwargs for apply_chat_template — currently just thinking mode."""
        if self.cfg.enable_thinking is None:
            return {}
        return {"enable_thinking": bool(self.cfg.enable_thinking)}

    def _format_prompt_text(self, question: str) -> str:
        """Qwen3 chat template → string. vLLM tokenizes server-side."""
        tok = self.hypernet.tokenizer
        messages = [
            {"role": "system", "content": _SYS_PROMPT},
            {"role": "user", "content": question},
        ]
        text = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            **self._chat_kwargs(),
        )
        if self.cfg.force_think:
            text = text + "<think>\n"
        return text

    def _encode_prompt_ids(self, question: str) -> list[int]:
        tok = self.hypernet.tokenizer
        messages = [
            {"role": "system", "content": _SYS_PROMPT},
            {"role": "user", "content": question},
        ]
        ids = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            max_length=self.cfg.question_max_length,
            truncation=True,
            **self._chat_kwargs(),
        )
        if self.cfg.force_think:
            ids = list(ids) + self._think_prefix_ids
        return ids

    def _tokenize_completion(self, text: str) -> list[int]:
        tok = self.hypernet.tokenizer
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return list(ids)

    # -- LoRA id ring buffer ---------------------------------------------------

    def _mem_snapshot(self) -> dict:
        """Snapshot of current/peak GPU memory in GiB on the training device."""
        if not torch.cuda.is_available() or self.hypernet.device.type != "cuda":
            return {}
        d = self.hypernet.device
        return {
            "alloc_GiB": round(torch.cuda.memory_allocated(d) / 2**30, 2),
            "reserved_GiB": round(torch.cuda.memory_reserved(d) / 2**30, 2),
            "peak_alloc_GiB": round(torch.cuda.max_memory_allocated(d) / 2**30, 2),
        }

    def _log_mem(self, label: str) -> None:
        """Log GPU memory at a boundary; flushed immediately so it survives OOM."""
        snap = self._mem_snapshot()
        if not snap:
            return
        import logging as _logging
        _logging.getLogger("meta_past.rl.rollout.mem").warning(
            "[mem %s] alloc=%.2f GiB  reserved=%.2f GiB  peak=%.2f GiB",
            label, snap["alloc_GiB"], snap["reserved_GiB"], snap["peak_alloc_GiB"],
        )

    def _lora_ids_for_step(self, step_idx: int, B: int) -> list[int]:
        """Stable, unique-within-window IDs starting from 1.

        ``lora_int_id == 0`` is reserved by vLLM (treated as no-LoRA). We add 1
        to keep all IDs ≥ 1 and ring-buffer over ``self._lora_window`` so that
        IDs from recent steps don't collide.
        """
        base = (step_idx * B) % self._lora_window
        return [(base + b) + 1 for b in range(B)]

    # -- main entry ------------------------------------------------------------

    def rollout_step(
        self,
        contexts: Sequence[SquadContext],
        step_idx: int,
        seed: int | None = None,
        global_lora_id_offset: int = 0,
    ) -> RolloutGroup:
        """Rollout for this rank's slice of contexts (length ``B_local``).

        ``global_lora_id_offset``: when running under DDP, each rank uses a
        disjoint slice of the LoRA-id ring buffer so registered LoRAs don't
        clash with peer ranks (each rank has its own LLM, but LoRA IDs are
        rank-local; we still keep them disjoint for log clarity).
        """
        cfg = self.cfg
        tok = self.hypernet.tokenizer
        device = self.hypernet.device
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        B = len(contexts)
        Q = cfg.questions_per_context
        K = cfg.rollouts_per_question

        if torch.cuda.is_available() and self.hypernet.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.hypernet.device)
        mem_pre = self._mem_snapshot()

        # 1 & 2: evidence → loradict for vLLM sampling (no_grad — trainer
        # will re-forward hypernet with grad per microbatch chunk so each
        # chunk's autograd graph is small and freed at chunk-end).
        set_phase("hyper")
        t0 = time.perf_counter()
        ev_ids, ev_mask = self._encode_contexts(contexts)
        with torch.no_grad():
            loradict = self.hypernet.generate_lora_grad(
                ev_ids, ev_mask,
                use_gradient_checkpoint=False,  # no_grad → grad ckpt is moot
                microbatch_size=cfg.hypernet_microbatch_contexts,
            )
        t_lora = time.perf_counter() - t0
        mem_post_hypernet = self._mem_snapshot()

        # 3: split into per-context PEFT-named tensor dicts
        t0 = time.perf_counter()
        per_b = shine_loradict_to_peft_batch(loradict)
        if len(per_b) != B:
            raise RuntimeError(
                f"shine_loradict_to_peft_batch returned {len(per_b)} dicts; expected B={B}."
            )
        local_ids = self._lora_ids_for_step(step_idx, B)
        lora_ids = [lid + global_lora_id_offset for lid in local_ids]
        # Detach + CPU copies for vLLM; originals stay on-device with autograd
        # graph for the deferred rescore.
        per_b_cpu = [
            {k: v.detach().to("cpu", copy=True) for k, v in d.items()}
            for d in per_b
        ]
        # Wake vLLM only for the rollout window; sleep right after sampling
        # so the GPU is free for the rescore + backward.
        # ``empty_cache`` first: PyTorch's caching allocator holds onto free
        # slabs (reserved >> allocated). vLLM's CuMemAllocator pulls from
        # the driver-level pool, which can't borrow from PyTorch's cache.
        # Without this, wake_up OOMs even when PyTorch's *allocated* footprint
        # is well under the GPU capacity.
        if torch.cuda.is_available() and self.hypernet.device.type == "cuda":
            torch.cuda.empty_cache()
        set_phase("wake")
        self.engine.wake_up(tags=["weights", "kv_cache"])
        set_phase("push")
        self.engine.push_lora_batch(per_b_cpu, lora_ids, self._peft_meta)
        t_push = time.perf_counter() - t0

        # 4: build prompts (B*Q in context-major order) and per-prompt lora_ids
        prompt_texts: list[str] = []
        prompt_id_lists: list[list[int]] = []
        prompt_qa: list[tuple[int, int, list[str]]] = []
        prompt_lora_ids: list[int] = []
        for ctx_idx, ctx in enumerate(contexts):
            qs = ctx.qa[:Q]
            if len(qs) < Q:
                raise ValueError(
                    f"Context {ctx.context_id} has only {len(qs)} questions; "
                    f"need {Q} (questions_per_context). Filter contexts upstream."
                )
            for q_idx, qa in enumerate(qs):
                prompt_texts.append(self._format_prompt_text(qa.question))
                prompt_id_lists.append(self._encode_prompt_ids(qa.question))
                prompt_qa.append((ctx_idx, q_idx, list(qa.references)))
                prompt_lora_ids.append(lora_ids[ctx_idx])

        # 5: sample K per prompt
        set_phase("sample")
        t0 = time.perf_counter()
        sampled = self.engine.complete(
            prompts=prompt_texts,
            lora_ids=prompt_lora_ids,
            n=K,
            temperature=cfg.temperature,
            max_tokens=cfg.max_new_tokens,
            seed=seed,
        )
        t_sample = time.perf_counter() - t0
        # Free vLLM HBM before rescore + backward.
        set_phase("sleep")
        self.engine.sleep(level=1)

        # 6: rewards + re-tokenize completions
        rewards_list: list[float] = []
        full_ids: list[list[int]] = []
        answer_masks: list[list[int]] = []
        group_ids_list: list[int] = []
        sample_lens: list[int] = []

        # First pass: extract predictions and re-tokenize for the rescore
        # tensor. Reward computation is deferred to a single batched call
        # so HTTP-judge rewards can fire all calls concurrently instead of
        # 1024× sequential round trips.
        n_oob_truncated = 0
        preds_for_reward: list[str] = []
        refs_for_reward: list[list[str]] = []
        questions_for_reward: list[str] = []
        for prompt_pos, ((ctx_idx, q_idx, refs), prompt_ids) in enumerate(
            zip(prompt_qa, prompt_id_lists)
        ):
            samples = sampled[prompt_pos]
            for samp in samples:
                y_ids = list(samp.token_ids) if samp.token_ids else self._tokenize_completion(samp.text)
                # Filter tokens that exceed SHINE's embedding rows (vLLM can
                # emit reserved Qwen3 special-token IDs above SHINE's resized
                # vocab). Truncate at first OOB; rest of the sequence would
                # be conditioned on an unembed-able token anyway.
                clean: list[int] = []
                for tid in y_ids:
                    if 0 <= tid < self._embed_rows:
                        clean.append(int(tid))
                    else:
                        n_oob_truncated += 1
                        break
                y_ids = clean
                if not y_ids:
                    y_ids = [tok.eos_token_id or pad_id]
                pred = _extract_answer(samp.text)
                preds_for_reward.append(pred)
                refs_for_reward.append(list(refs))
                # Look up the question via prompt_qa (ctx_idx, q_idx, refs).
                questions_for_reward.append(prompt_qa[prompt_pos][2] and "")
                # ^ placeholder; replaced below by the actual question text.
                full_ids.append(list(prompt_ids) + y_ids)
                answer_masks.append([0] * len(prompt_ids) + [1] * len(y_ids))
                group_ids_list.append(ctx_idx * Q + q_idx)
                sample_lens.append(len(y_ids))

        # Rebuild questions list from the (ctx, q_idx) → question text map.
        # ``ctx.qa[q_idx].question`` is the canonical text.
        questions_by_pair: dict[tuple[int, int], str] = {}
        for ctx_idx, ctx in enumerate(contexts):
            for q_idx, qa in enumerate(ctx.qa[:Q]):
                questions_by_pair[(ctx_idx, q_idx)] = qa.question
        # n samples per (ctx, q) prompt = K; group_ids_list[i] = ctx_idx*Q+q_idx
        questions_for_reward = []
        for gid in group_ids_list:
            ctx_idx, q_idx = divmod(gid, Q)
            questions_for_reward.append(questions_by_pair[(ctx_idx, q_idx)])

        # Batched reward call. ``HttpJudgeReward.batch_compute`` fires all
        # requests through a thread pool concurrently; ``f1_reward`` is
        # CPU-bound and just loops sequentially.
        set_phase("reward")
        t_reward_0 = time.perf_counter()
        batch_compute = getattr(self.reward_fn, "batch_compute", None)
        if batch_compute is not None:
            rewards_list = list(batch_compute(
                preds_for_reward, refs_for_reward,
                questions=questions_for_reward,
            ))
        else:
            rewards_list = [
                float(self.reward_fn(p, r, question=q))
                for p, r, q in zip(
                    preds_for_reward, refs_for_reward, questions_for_reward,
                )
            ]
        t_reward = time.perf_counter() - t_reward_0

        # 7: pad + re-score under grad. Order is context-major, then
        # question-major, then sample-major — so SHINE's
        # ``num_beams = N // Lb`` per-row LoRA selection lines up.
        N = len(full_ids)
        if N != B * Q * K:
            raise RuntimeError(
                f"Expected B*Q*K = {B*Q*K} samples, got {N}. Did vLLM drop any?"
            )
        T_max = max(len(s) for s in full_ids)
        input_ids = torch.full((N, T_max), pad_id, device=device, dtype=torch.long)
        attn_mask = torch.zeros((N, T_max), device=device, dtype=torch.long)
        ans_mask = torch.zeros((N, T_max), device=device, dtype=torch.long)
        for i, (seq, am) in enumerate(zip(full_ids, answer_masks)):
            L = len(seq)
            input_ids[i, :L] = torch.tensor(seq, device=device, dtype=torch.long)
            attn_mask[i, :L] = 1
            ans_mask[i, :L] = torch.tensor(am, device=device, dtype=torch.long)

        # Pre-compute response_mask in shifted form so the trainer can compute
        # advantages without needing to run the rescore first. This matches
        # what ``score_answer_logprobs`` returns as its second tensor.
        response_mask = ans_mask[:, 1:].to(torch.float32)

        rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)
        group_ids_t = torch.tensor(group_ids_list, device=device, dtype=torch.long)

        # The vLLM-side loradict has served its purpose; the trainer will
        # re-forward hypernet on chunks for the rescore+backward.
        del loradict, per_b, per_b_cpu
        return RolloutGroup(
            rewards=rewards,
            response_mask=response_mask,
            group_ids=group_ids_t,
            evidence_ids=ev_ids,
            evidence_mask=ev_mask,
            input_ids=input_ids,
            attention_mask=attn_mask,
            answer_mask=ans_mask,
            meta={
                "t_lora_s": t_lora,
                "t_push_s": t_push,
                "t_sample_s": t_sample,
                "t_reward_s": t_reward,
                "n_samples": N,
                "T_max": T_max,
                "B": B,
                "Q": Q,
                "K": K,
                "sample_lens_mean": float(sum(sample_lens) / len(sample_lens)),
                "sample_lens_min": int(min(sample_lens)),
                "sample_lens_max": int(max(sample_lens)),
                "n_oob_truncated": int(n_oob_truncated),
                "mem_pre": mem_pre,
                "mem_post_hypernet": mem_post_hypernet,
            },
        )
