#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Free-form memory-QA inference for SHINE_V2 (greedy decode), invoked from
meta_train_tp.tp_main when env MEMORY_QA_GEN=1 (after model build + checkpoint load).

Per test history (segmented memory_stream format):
  1. reset detach_state W;
  2. for each segment in order: build the fresh per-segment LoRA from its context
     (compute_memory_states -> hypernetwork), GREEDY-DECODE answers to that segment's
     QA (and the cross-segment final_qa on the last segment) using fresh-LoRA + metalora
     + accumulated detached W, then ACCUMULATE this segment into W (_write_detach_state);
  3. compare decoded answer to gold (lenient: gold substring; strict: exact after <think>).

This mirrors the training information flow exactly (W holds segments 0..i-1 when
answering segment i's questions). Single-process (main rank); TP=1 has no collectives.

Run (after training):
  EXP_NAME=mem_qwen35_4b ANNEALING_NAME=memstream_v1 MEMORY_QA_GEN=1 \
  MEMORY_QA_TEST_FILE=data/mem_synth/val.jsonl MEMORY_QA_NUM=5 \
  torchrun --nproc_per_node=1 meta_train.py --config-name=main_pretrain_annealing \
    model=Qwen3_5-4B m2p_transformer=full_prenorm_gatedlastnorm_4b \
    data=pretrain_annealing/memory_stream detach_state=full \
    parallel.mode=tp parallel.tensor_parallel_size=1 parallel.total_gpus=1 \
    training.resume_from=checkpoint/mem_qwen35_4b/pretrain_annealing/memstream_v1/final
"""

from __future__ import annotations

import json
import os
import re
import collections

import torch

from utils.mytokenizer import create_tokenizer, NOTHINKING_CHAT_TEMPLATE
from utils.myloradict import concat_loradict
from mydatasets.pretrain_annealing.memory_stream import _encode_turns, _qa_to_messages  # noqa: F401


def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _strip_think(text: str) -> str:
    # NOTHINKING template still emits an empty <think>..</think> before content.
    m = re.split(r"</think>", text, maxsplit=1)
    return (m[1] if len(m) > 1 else m[0]).strip()


@torch.no_grad()
def _gen_eval_core(model, cfg, my_device, *, n_hist, max_new, seg_sample, use_kv, recall, test_file):
    """Per-type generation accuracy. Returns (hit, total, examples) Counters.

    recall=False (default): mirrors training info flow — answer segment i's QA
        right after building its fresh per-segment LoRA, with W holding 0..i-1.
        The fact is in the CURRENT segment, so this barely needs W.
    recall=True (deferred): FIRST stream+write ALL K segments into W (no decode),
        THEN answer each sampled segment's QA from the FULL accumulated W using
        metalora ONLY (no fresh per-segment LoRA) -> the fact for segment i now
        lives only in W, so this directly tests 'do earlier segments help later
        answers'. The clean detach_state probe.

    Main-process only; caller handles is_main_process gating, detach_state
    isolation (eval_context), barrier, and model.train() restore.
    """
    n_layers = int(model._num_llm_layers)
    num_mem = int(getattr(model, "_num_mem_token", 0))
    ctx_len_cap = int(cfg.data.get("context_seq_length", 2048))
    tok = create_tokenizer(cfg.model.path, tokenizer_cfg=cfg.get("tokenizer", None),
                           chat_template=NOTHINKING_CHAT_TEMPLATE)
    pad_id = tok.pad_token_id or 0
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_id = tok.eos_token_id
    records = _load_jsonl(test_file)[:n_hist]
    model.eval()
    hit, total = collections.Counter(), collections.Counter()
    examples = []
    kv_state = {"ok": use_kv}

    def _build_context(seg):
        ids = _encode_turns(tok, seg["history_turns"])[:ctx_len_cap]
        ctx = torch.full((1, ctx_len_cap + num_mem), pad_id, dtype=torch.long, device=my_device)
        ctx[0, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=my_device)
        return ctx, torch.tensor([len(ids)], dtype=torch.long, device=my_device)

    def _greedy(question, new_loradict, ds_l, ds_w):
        msgs = [{"role": "user", "content": question}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=False, enable_thinking=False)
        full = torch.tensor([ids], dtype=torch.long, device=my_device)
        gen = []
        model._active_w_transform = model.w_transform_conversation
        try:
            if not kv_state["ok"]:
                raise RuntimeError("kv disabled")
            pkv = None
            cur = full
            for _ in range(max_new):
                out = model.llm.model(input_ids=cur, attention_mask=None, loradict=new_loradict,
                                      nograd_loradict=ds_l, nograd_wdict=ds_w,
                                      use_cache=True, past_key_values=pkv)
                pkv = out.past_key_values
                nxt = int(model.llm.lm_head(out.last_hidden_state[:, -1, :]).argmax(-1).item())
                if nxt == im_end or nxt == eos_id:
                    break
                gen.append(nxt)
                cur = torch.tensor([[nxt]], dtype=torch.long, device=my_device)
        except Exception as e:
            if kv_state["ok"]:
                print(f"[memory_qa_gen] KV-cache decode failed ({e}); falling back to no-cache.", flush=True)
                kv_state["ok"] = False
            gen = []
            cur = full
            for _ in range(max_new):
                out = model.llm.model(input_ids=cur, attention_mask=None, loradict=new_loradict,
                                      nograd_loradict=ds_l, nograd_wdict=ds_w, use_cache=False)
                nxt = int(model.llm.lm_head(out.last_hidden_state[:, -1, :]).argmax(-1).item())
                if nxt == im_end or nxt == eos_id:
                    break
                gen.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]], device=my_device)], dim=1)
        model._active_w_transform = None
        return _strip_think(tok.decode(gen, skip_special_tokens=True))

    def _score(qas, new_loradict, ds_l, ds_w, type_suffix=""):
        for qa in qas:
            pred = _greedy(qa["question"], new_loradict, ds_l, ds_w)
            gold = str(qa["answer"])
            ok = gold.lower() in pred.lower()
            t = qa.get("type", "?") + type_suffix
            total[t] += 1
            hit[t] += int(ok)
            if len(examples) < 12:
                examples.append((t, qa["question"], gold, pred, ok))

    for rec in records:
        segs = rec["segments"]
        final_qa = rec.get("final_qa", [])
        K = len(segs)
        if seg_sample <= 0 or seg_sample >= K:
            sample_idx = set(range(K))
        else:
            step = max(1, K // seg_sample)
            sample_idx = set(range(0, K, step))
        if model.detach_state is not None:
            model.detach_state.reset()

        if not recall:
            for i, seg in enumerate(segs):
                ds_l, ds_w = model._read_detach_state()
                ctx_ids, ctx_lens = _build_context(seg)
                mem = model.compute_memory_states(ctx_ids, None, ctx_lens,
                                                  nograd_loradict=ds_l, nograd_wdict=ds_w)
                loradict = model.generate_loradict(mem)
                new_loradict = {l: concat_loradict([loradict[l], model.metalora[l]]) for l in range(n_layers)}
                qas = list(seg.get("qa", [])) if i in sample_idx else []
                if i == K - 1 and final_qa:
                    qas = qas + final_qa
                _score(qas, new_loradict, ds_l, ds_w)
                if model.detach_state is not None:
                    model._write_detach_state(loradict, mb_idx=0)
        else:
            # Phase 1: write ALL segments into W (no decode).
            for i, seg in enumerate(segs):
                ds_l, ds_w = model._read_detach_state()
                ctx_ids, ctx_lens = _build_context(seg)
                mem = model.compute_memory_states(ctx_ids, None, ctx_lens,
                                                  nograd_loradict=ds_l, nograd_wdict=ds_w)
                loradict = model.generate_loradict(mem)
                if model.detach_state is not None:
                    model._write_detach_state(loradict, mb_idx=0)
            # Phase 2: answer from the FULL accumulated W with metalora ONLY.
            ds_l, ds_w = model._read_detach_state()
            meta_only = {l: model.metalora[l] for l in range(n_layers)}
            for i, seg in enumerate(segs):
                if i not in sample_idx:
                    continue
                # "_recall" suffix keeps these distinct from the immediate-answer
                # numbers; an early-segment fact correct here came purely from W.
                _score(list(seg.get("qa", [])), meta_only, ds_l, ds_w, type_suffix="_recall")
            if final_qa:
                _score(final_qa, meta_only, ds_l, ds_w, type_suffix="_recall")

    return hit, total, examples


@torch.no_grad()
def run_memory_qa_gen(model, cfg, tp_cfg, my_device):
    """Standalone entry (MEMORY_QA_GEN=1, no ICL): print per-type accuracy.
    Set MEMORY_QA_RECALL=1 for the deferred-recall probe."""
    from utils.myparallel import is_main_process, barrier
    if not is_main_process():
        barrier()
        return
    test_file = os.environ.get("MEMORY_QA_TEST_FILE") or os.path.join(
        cfg.data.get("data_path", "data/mem_synth"), cfg.data.get("val_file", "val.jsonl"))
    recall = os.environ.get("MEMORY_QA_RECALL", "") == "1"
    hit, total, examples = _gen_eval_core(
        model, cfg, my_device,
        n_hist=int(os.environ.get("MEMORY_QA_NUM", "5")),
        max_new=int(os.environ.get("MEMORY_QA_MAX_NEW", "16")),
        seg_sample=int(os.environ.get("MEMORY_QA_SEG_SAMPLE", "4")),
        use_kv=os.environ.get("MEMORY_QA_KV", "1") != "0",
        recall=recall, test_file=test_file,
    )
    print(f"\n[memory_qa_gen] recall={recall} from {test_file}", flush=True)
    print("\n===== sample QA (✓/✗  type | Q -> gold || pred) =====")
    for t, q, g, p, ok in examples:
        print(f"[{'✓' if ok else '✗'}] {t} | {q}\n      gold: {g}\n      pred: {p}")
    print("\n===== accuracy by type (lenient: gold substring of pred) =====")
    for t in sorted(total):
        print(f"  {t:22s}: {hit[t]}/{total[t]} = {hit[t]/max(1,total[t]):.3f}")
    print(f"  {'OVERALL':22s}: {sum(hit.values())}/{sum(total.values())} = "
          f"{sum(hit.values())/max(1,sum(total.values())):.3f}", flush=True)
    barrier()


@torch.no_grad()
def run_memory_qa_gen_inloop(model, cfg, tp_cfg, my_device, *, recall=False,
                             n_hist=8, max_new=16, seg_sample=4):
    """In-training generation eval. Isolates detach_state (eval_context) so the
    training W is untouched, computes per-type accuracy on the MAIN process,
    restores model.train(), and barriers all ranks. Returns (hit, total)
    Counters on the main process, (None, None) elsewhere. Caller logs to wandb.
    """
    from utils.myparallel import is_main_process, barrier
    from contextlib import nullcontext
    was_training = model.training
    _ctx = model.detach_state.eval_context() if getattr(model, "detach_state", None) is not None else nullcontext()
    _ctx.__enter__()
    hit = total = None
    try:
        if is_main_process():
            test_file = os.environ.get("MEMORY_QA_TEST_FILE") or os.path.join(
                cfg.data.get("data_path", "data/mem_synth"), cfg.data.get("val_file", "val.jsonl"))
            hit, total, _ = _gen_eval_core(
                model, cfg, my_device, n_hist=n_hist, max_new=max_new,
                seg_sample=seg_sample, use_kv=os.environ.get("MEMORY_QA_KV", "1") != "0",
                recall=recall, test_file=test_file,
            )
    finally:
        _ctx.__exit__(None, None, None)
        if was_training:
            model.train()
    barrier()
    return hit, total


@torch.no_grad()
def run_memory_qa_icl(model, cfg, tp_cfg, my_device):
    """ICL baseline (NO SHINE memory): put the cumulative history of all
    segments 0..i directly in the context window and let the PLAIN base model
    answer (loradict / nograd_wdict all None — no hypernetwork, no metalora,
    no accumulated W). Same data, same QA, same lenient scoring as
    run_memory_qa_gen, so the two are directly comparable.

    When the cumulative history exceeds MEMORY_QA_ICL_MAXLEN tokens we keep the
    most-recent tokens (front-truncate, question stays at the tail) — this is
    exactly where ICL degrades and SHINE is supposed to win.
    """
    from utils.myparallel import is_main_process, barrier

    if not is_main_process():
        barrier()
        return

    test_file = os.environ.get("MEMORY_QA_TEST_FILE") or os.path.join(
        cfg.data.get("data_path", "data/mem_synth"), cfg.data.get("val_file", "val.jsonl"))
    n_hist = int(os.environ.get("MEMORY_QA_NUM", "5"))
    max_new = int(os.environ.get("MEMORY_QA_MAX_NEW", "16"))
    seg_sample = int(os.environ.get("MEMORY_QA_SEG_SAMPLE", "4"))
    use_kv = os.environ.get("MEMORY_QA_KV", "1") != "0"
    # Base LLM context budget for ICL (NOT the SHINE per-segment ctx cap).
    max_len = int(os.environ.get("MEMORY_QA_ICL_MAXLEN", "32768"))

    tok = create_tokenizer(cfg.model.path, tokenizer_cfg=cfg.get("tokenizer", None),
                           chat_template=NOTHINKING_CHAT_TEMPLATE)
    pad_id = tok.pad_token_id or 0
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_id = tok.eos_token_id

    records = _load_jsonl(test_file)[:n_hist]
    print(f"\n[memory_qa_icl] ICL baseline (no SHINE): {len(records)} histories from "
          f"{test_file} (max_len={max_len}, max_new={max_new})", flush=True)

    model.eval()
    hit, total = collections.Counter(), collections.Counter()
    examples = []
    n_truncated = 0
    kv_state = {"ok": use_kv}

    def _prompt_ids(hist_turns, question):
        """Chat-template [all history turns] + [user question] + gen prompt."""
        msgs = list(hist_turns) + [{"role": "user", "content": question}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=False, enable_thinking=False)
        return ids

    def _greedy(hist_turns, question):
        nonlocal n_truncated
        ids = _prompt_ids(hist_turns, question)
        if len(ids) > max_len:
            ids = ids[-max_len:]            # keep most-recent + the question tail
            n_truncated += 1
        full = torch.tensor([ids], dtype=torch.long, device=my_device)
        gen = []
        try:
            if not kv_state["ok"]:
                raise RuntimeError("kv disabled")
            pkv = None
            cur = full
            for _ in range(max_new):
                out = model.llm.model(input_ids=cur, attention_mask=None, loradict=None,
                                      nograd_loradict=None, nograd_wdict=None,
                                      use_cache=True, past_key_values=pkv)
                pkv = out.past_key_values
                nxt = int(model.llm.lm_head(out.last_hidden_state[:, -1, :]).argmax(-1).item())
                if nxt == im_end or nxt == eos_id:
                    break
                gen.append(nxt)
                cur = torch.tensor([[nxt]], dtype=torch.long, device=my_device)
        except Exception as e:
            if kv_state["ok"]:
                print(f"[memory_qa_icl] KV-cache decode failed ({e}); falling back to no-cache.", flush=True)
                kv_state["ok"] = False
            gen = []
            cur = full
            for _ in range(max_new):
                out = model.llm.model(input_ids=cur, attention_mask=None, loradict=None,
                                      nograd_loradict=None, nograd_wdict=None, use_cache=False)
                nxt = int(model.llm.lm_head(out.last_hidden_state[:, -1, :]).argmax(-1).item())
                if nxt == im_end or nxt == eos_id:
                    break
                gen.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]], device=my_device)], dim=1)
        return _strip_think(tok.decode(gen, skip_special_tokens=True))

    for rec in records:
        segs = rec["segments"]
        final_qa = rec.get("final_qa", [])
        K = len(segs)
        if seg_sample <= 0 or seg_sample >= K:
            sample_idx = set(range(K))
        else:
            step = max(1, K // seg_sample)
            sample_idx = set(range(0, K, step))
        # cumulative history turns up to and including each segment
        cum_turns = []
        for i, seg in enumerate(segs):
            cum_turns = cum_turns + list(seg.get("history_turns", []))
            qas = list(seg.get("qa", [])) if i in sample_idx else []
            if i == K - 1 and final_qa:
                qas = qas + final_qa          # cross-segment QA sees the FULL history
            for qa in qas:
                pred = _greedy(cum_turns, qa["question"])
                gold = str(qa["answer"])
                ok = gold.lower() in pred.lower()
                t = qa.get("type", "?")
                total[t] += 1
                hit[t] += int(ok)
                if len(examples) < 12:
                    examples.append((t, qa["question"], gold, pred, ok))

    print("\n===== sample QA (✓/✗  type | Q -> gold || pred) =====")
    for t, q, g, p, ok in examples:
        print(f"[{'✓' if ok else '✗'}] {t} | {q}\n      gold: {g}\n      pred: {p}")
    print(f"\n===== ICL accuracy by type (lenient; {n_truncated} prompts front-truncated to {max_len}) =====")
    for t in sorted(total):
        print(f"  {t:18s}: {hit[t]}/{total[t]} = {hit[t]/max(1,total[t]):.3f}")
    print(f"  {'OVERALL':18s}: {sum(hit.values())}/{sum(total.values())} = "
          f"{sum(hit.values())/max(1,sum(total.values())):.3f}", flush=True)
    barrier()
