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
def run_memory_qa_gen(model, cfg, tp_cfg, my_device):
    from utils.myddp import is_main_process, barrier

    if not is_main_process():
        barrier()
        return

    test_file = os.environ.get("MEMORY_QA_TEST_FILE") or os.path.join(
        cfg.data.get("data_path", "data/mem_synth"), cfg.data.get("val_file", "val.jsonl"))
    n_hist = int(os.environ.get("MEMORY_QA_NUM", "5"))
    max_new = int(os.environ.get("MEMORY_QA_MAX_NEW", "32"))
    ctx_len_cap = int(cfg.data.get("context_seq_length", 2048))
    num_mem = int(getattr(model, "_num_mem_token", 0))
    n_layers = int(model._num_llm_layers)

    tok = create_tokenizer(cfg.model.path, tokenizer_cfg=cfg.get("tokenizer", None),
                           chat_template=NOTHINKING_CHAT_TEMPLATE)
    pad_id = tok.pad_token_id or 0
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_id = tok.eos_token_id

    records = _load_jsonl(test_file)[:n_hist]
    print(f"\n[memory_qa_gen] {len(records)} histories from {test_file} "
          f"(num_mem={num_mem}, ctx_cap={ctx_len_cap}, max_new={max_new})", flush=True)

    model.eval()
    hit, total = collections.Counter(), collections.Counter()
    examples = []

    def _build_context(seg):
        ids = _encode_turns(tok, seg["history_turns"])[:ctx_len_cap]
        ctx = torch.full((1, ctx_len_cap + num_mem), pad_id, dtype=torch.long, device=my_device)
        ctx[0, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=my_device)
        return ctx, torch.tensor([len(ids)], dtype=torch.long, device=my_device)

    def _greedy(question, new_loradict, ds_l, ds_w):
        msgs = [{"role": "user", "content": question}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=False, enable_thinking=False)
        ids = torch.tensor([ids], dtype=torch.long, device=my_device)
        gen = []
        model._active_w_transform = model.w_transform_conversation
        for _ in range(max_new):
            out = model.llm.model(input_ids=ids, attention_mask=None, loradict=new_loradict,
                                  nograd_loradict=ds_l, nograd_wdict=ds_w, use_cache=False)
            logits = model.llm.lm_head(out.last_hidden_state[:, -1, :])
            nxt = int(logits.argmax(-1).item())
            if nxt == im_end or nxt == eos_id:
                break
            gen.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=my_device)], dim=1)
        model._active_w_transform = None
        return _strip_think(tok.decode(gen, skip_special_tokens=True))

    for rec in records:
        segs = rec["segments"]
        final_qa = rec.get("final_qa", [])
        if model.detach_state is not None:
            model.detach_state.reset()
        for i, seg in enumerate(segs):
            ds_l, ds_w = model._read_detach_state()
            ctx_ids, ctx_lens = _build_context(seg)
            mem = model.compute_memory_states(ctx_ids, None, ctx_lens,
                                              nograd_loradict=ds_l, nograd_wdict=ds_w)
            loradict = model.generate_loradict(mem)
            new_loradict = {l: concat_loradict([loradict[l], model.metalora[l]]) for l in range(n_layers)}

            qas = list(seg.get("qa", []))
            if i == len(segs) - 1 and final_qa:
                qas = qas + final_qa
            for qa in qas:
                pred = _greedy(qa["question"], new_loradict, ds_l, ds_w)
                gold = str(qa["answer"])
                ok = gold.lower() in pred.lower()
                t = qa.get("type", "?")
                total[t] += 1
                hit[t] += int(ok)
                if len(examples) < 12:
                    examples.append((t, qa["question"], gold, pred, ok))

            if model.detach_state is not None:
                model._write_detach_state(loradict, mb_idx=0)

    print("\n===== sample QA (✓/✗  type | Q -> gold || pred) =====")
    for t, q, g, p, ok in examples:
        print(f"[{'✓' if ok else '✗'}] {t} | {q}\n      gold: {g}\n      pred: {p}")
    print("\n===== accuracy by type (lenient: gold substring of pred) =====")
    for t in sorted(total):
        print(f"  {t:18s}: {hit[t]}/{total[t]} = {hit[t]/max(1,total[t]):.3f}")
    print(f"  {'OVERALL':18s}: {sum(hit.values())}/{sum(total.values())} = "
          f"{sum(hit.values())/max(1,sum(total.values())):.3f}", flush=True)
    barrier()
