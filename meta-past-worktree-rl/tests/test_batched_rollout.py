"""Unit test for RLRollout's batched-context orchestration.

Mocks ``ShineHypernet`` and ``VLLMPool`` so the test runs CPU-only and in
seconds. Asserts:
- B*Q prompts are dispatched, in context-major order
- per-prompt LoRA ids match the per-step LoRA id assignment
- group_ids = ctx_idx * Q + q_idx, broadcast across K samples
- re-score input ordering is context-major (so SHINE's
  ``num_beams = N // Lb`` per-row LoRA selection lines up with ``Lb=B``)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from unittest.mock import MagicMock

import torch

from meta_past.rl.rollout import RLRollout, RLRolloutConfig
from meta_past.data.squad_contexts import QA, SquadContext
from meta_past.rl.vllm_engine import SampledSequence


# A trivial hypernet stand-in: returns a fake loradict and records calls.
class _FakeHypernet:
    def __init__(self):
        self.device = torch.device("cpu")
        self.lora_r = 4
        # Fake nested ``metanetwork.metamodel.get_input_embeddings()`` that
        # exposes ``num_embeddings`` so RLRollout can read the vocab cap.
        emb = MagicMock()
        emb.num_embeddings = 200
        meta_model = MagicMock()
        meta_model.get_input_embeddings = lambda: emb
        meta = MagicMock()
        meta.metamodel = meta_model
        self.metanetwork = meta
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 1
        # Tokenizer call: return token ids of length context_max_length.
        def _tok(texts, **kw):
            if isinstance(texts, str):
                texts = [texts]
            T = kw.get("max_length", 16)
            n = len(texts)
            return {
                "input_ids": torch.zeros(n, T, dtype=torch.long),
                "attention_mask": torch.ones(n, T, dtype=torch.long),
            }
        self.tokenizer.side_effect = _tok
        self.tokenizer.apply_chat_template = MagicMock(
            side_effect=lambda *a, **kw: (
                # tokenize=True returns ids; tokenize=False returns text
                [10, 11, 12] if kw.get("tokenize", False) else "PROMPT"
            )
        )
        self.last_ev_ids_shape = None
        self.score_input_shape = None
        self.score_loradict_lb = None

    def generate_lora_grad(self, ev_ids, ev_mask, **kw):
        # Record shape, return a minimal loradict with Lb = ev_ids.shape[0].
        Lb = int(ev_ids.shape[0])
        self.last_ev_ids_shape = tuple(ev_ids.shape)
        in_f, out_f, r = 8, 8, self.lora_r
        leaf_AB = lambda: {
            "A": torch.zeros(Lb, in_f, r, requires_grad=True),
            "B": torch.zeros(Lb, r, out_f, requires_grad=True),
            "C": None,
        }
        return {
            0: {
                "attention": {k: leaf_AB() for k in ("q", "k", "v", "o")},
                "mlp": {k: leaf_AB() for k in ("gate", "up", "down")},
            }
        }

    def score_answer_logprobs(self, *, loradict, input_ids, attention_mask,
                              answer_mask, use_gradient_checkpoint, **kw):
        # Capture shapes; return logprobs that depend on the loradict so
        # autograd would be possible (but we don't backward in this test).
        first_leaf = loradict[0]["attention"]["q"]["A"]
        self.score_loradict_lb = int(first_leaf.shape[0])
        self.score_input_shape = tuple(input_ids.shape)
        N, T = input_ids.shape
        logprobs = first_leaf.sum() * torch.zeros(N, T - 1)
        mask = answer_mask[:, 1:].float()
        return logprobs, mask


class _FakeEngine:
    """Stand-in for ``VLLMEngine`` with the methods rollout uses."""
    def __init__(self):
        self.pushed_lora_ids: list[int] = []
        self.pushed_per_b_count = 0
        self.last_complete_args: dict = {}
        self.wake_calls = 0
        self.sleep_calls = 0

    def wake_up(self, tags=None):
        self.wake_calls += 1

    def sleep(self, level=1):
        self.sleep_calls += 1

    def push_lora_batch(self, per_b_tensors, lora_ids, peft_meta):
        self.pushed_per_b_count = len(per_b_tensors)
        self.pushed_lora_ids = list(lora_ids)

    def complete(self, prompts, lora_ids, n, temperature, max_tokens,
                 seed=None, top_p=1.0, stop=None):
        self.last_complete_args = {
            "prompts": list(prompts),
            "lora_ids": list(lora_ids),
            "n": n,
        }
        out: list[list[SampledSequence]] = []
        for i in range(len(prompts)):
            out.append([
                SampledSequence(text=f"ans-{i}-{k}", token_ids=[42, 43])
                for k in range(n)
            ])
        return out


def _ctx(cid: str, n_q: int) -> SquadContext:
    return SquadContext(
        context_id=cid,
        context=f"CONTEXT-{cid}",
        qa=[QA(question=f"q{j}", references=[f"ref{j}"]) for j in range(n_q)],
    )


def test_rollout_step_batched_shapes_and_grouping():
    B, Q, K = 2, 2, 4
    contexts = [_ctx("a", Q), _ctx("b", Q)]

    hyp = _FakeHypernet()
    pool = _FakeEngine()

    rollout = RLRollout(
        hypernet=hyp,  # type: ignore[arg-type]
        reward_fn=lambda pred, refs, **kw: 1.0 if pred in refs else 0.0,
        engine=pool,  # type: ignore[arg-type]
        cfg=RLRolloutConfig(
            context_max_length=16,
            question_max_length=8,
            max_new_tokens=4,
            temperature=1.0,
            contexts_per_step=B,
            questions_per_context=Q,
            rollouts_per_question=K,
            use_gradient_checkpoint=False,
        ),
    )

    group = rollout.rollout_step(contexts, step_idx=3, seed=7)

    # 1) push_lora_batch saw B per-context dicts and B lora_ids.
    assert pool.pushed_per_b_count == B
    assert len(pool.pushed_lora_ids) == B
    assert len(set(pool.pushed_lora_ids)) == B  # unique within step
    assert all(lid >= 1 for lid in pool.pushed_lora_ids)

    # 2) complete_many got B*Q prompts in context-major order and per-prompt lora ids.
    assert len(pool.last_complete_args["prompts"]) == B * Q
    expected_per_prompt_lora = [
        pool.pushed_lora_ids[ctx_idx]
        for ctx_idx in range(B)
        for _ in range(Q)
    ]
    assert pool.last_complete_args["lora_ids"] == expected_per_prompt_lora
    assert pool.last_complete_args["n"] == K

    # 3) hypernet evidence shape is [B, T_ctx]
    assert hyp.last_ev_ids_shape == (B, 16)

    # 4) deferred rescore inputs: shapes line up. With joint-microbatch the
    # rollout no longer ships a loradict — it ships the encoded evidence so
    # the trainer can re-forward hypernet (with grad) per chunk.
    assert group.input_ids.shape[0] == B * Q * K
    assert group.attention_mask.shape == group.input_ids.shape
    assert group.answer_mask.shape == group.input_ids.shape
    assert group.evidence_ids.shape[0] == B
    assert group.evidence_mask.shape == group.evidence_ids.shape
    assert not hasattr(group, "loradict") or group.loradict is None  # removed

    # 5) group_ids[i] = ctx_idx * Q + q_idx, broadcast across K samples.
    expected_group_ids = []
    for ctx_idx in range(B):
        for q_idx in range(Q):
            expected_group_ids.extend([ctx_idx * Q + q_idx] * K)
    assert group.group_ids.tolist() == expected_group_ids

    # 6) reward count = N
    assert group.rewards.shape[0] == B * Q * K

    # 7) ring-buffer sanity: same step idx → same lora_ids
    pool2 = _FakeEngine()
    rollout2 = RLRollout(
        hypernet=_FakeHypernet(),  # type: ignore[arg-type]
        reward_fn=lambda pred, refs, **kw: 0.0,
        engine=pool2,               # type: ignore[arg-type]
        cfg=rollout.cfg,
    )
    rollout2.rollout_step(contexts, step_idx=3, seed=7)
    assert pool2.pushed_lora_ids == pool.pushed_lora_ids
