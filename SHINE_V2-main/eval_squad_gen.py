#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Free-form SQuAD evaluation for SHINE_V2.

Invoked from meta_train_tp.tp_main when env SQUAD_QA_GEN=1, after the V2 model
has been built and checkpoint weights have been loaded. For each SQuAD example:
  1. encode the context as the memory-writing context;
  2. generate a fresh LoRA from that context;
  3. answer the question with only the question prompt plus generated LoRA.

This mirrors the original SHINE comparison setting: context is compiled into
parameters, and the query side can omit the context.
"""

from __future__ import annotations

import collections
import json
import os
import re
import string
from typing import Any, Dict, Iterable, List

import torch

from utils.mytokenizer import create_tokenizer, NOTHINKING_CHAT_TEMPLATE
from utils.myloradict import concat_loradict


def _strip_think(text: str) -> str:
    m = re.split(r"</think>", text, maxsplit=1)
    out = (m[1] if len(m) > 1 else m[0]).strip()
    out = re.sub(r"^(final answer|answer)\s*:\s*", "", out, flags=re.IGNORECASE).strip()
    return out.splitlines()[0].strip() if out else ""


def _normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def _answers_text(example: Dict[str, Any]) -> List[str]:
    answers = example.get("answers", {})
    if isinstance(answers, dict):
        texts = answers.get("text", [])
        if isinstance(texts, str):
            return [texts]
        return [str(x) for x in texts] if texts else [""]
    if isinstance(answers, list):
        return [str(x) for x in answers] if answers else [""]
    return [str(answers)] if answers else [""]


def _load_squad_records(path: str | None, split: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset, load_from_disk

    if path and os.path.exists(path):
        try:
            ds = load_from_disk(path)
            if hasattr(ds, "keys"):
                return list(ds[split])
            return list(ds)
        except Exception:
            ds = load_dataset(path)
            return list(ds[split])

    ds = load_dataset("rajpurkar/squad")
    return list(ds[split])


@torch.no_grad()
def run_squad_qa_gen(model, cfg, tp_cfg, my_device):
    from utils.myparallel import is_main_process, barrier

    if not is_main_process():
        barrier()
        return

    data_path = os.environ.get("SQUAD_DATA_PATH", "data/squad")
    split = os.environ.get("SQUAD_SPLIT", "validation")
    limit = int(os.environ.get("SQUAD_NUM", "-1"))
    max_new = int(os.environ.get("SQUAD_MAX_NEW", "32"))
    out_path = os.environ.get(
        "SQUAD_OUT",
        os.path.join("outputs", "squad_shine_v2", f"squad_{split}.json"),
    )
    query_include_context = os.environ.get("SQUAD_QUERY_INCLUDE_CONTEXT", "0") == "1"

    ctx_len_cap = int(cfg.data.get("context_seq_length", 2048))
    num_mem = int(getattr(model, "_num_mem_token", 0))
    n_layers = int(model._num_llm_layers)

    tok = create_tokenizer(
        cfg.model.path,
        tokenizer_cfg=cfg.get("tokenizer", None),
        chat_template=NOTHINKING_CHAT_TEMPLATE,
    )
    pad_id = tok.pad_token_id or 0
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_id = tok.eos_token_id

    records = _load_squad_records(data_path, split)
    if limit > 0:
        records = records[:limit]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(
        f"\n[squad_qa_gen] {len(records)} examples split={split} data={data_path} "
        f"ctx_cap={ctx_len_cap} max_new={max_new} query_include_context={query_include_context}",
        flush=True,
    )

    model.eval()

    def _context_tensor(context: str):
        # Match original SHINE's SquadCollator: evidence is tokenized as raw text,
        # while only the query side uses the chat template.
        ids = tok(
            context,
            add_special_tokens=True,
            truncation=True,
            max_length=ctx_len_cap,
            return_attention_mask=False,
        )["input_ids"]
        ctx = torch.full((1, ctx_len_cap + num_mem), pad_id, dtype=torch.long, device=my_device)
        ctx[0, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=my_device)
        return ctx, torch.tensor([len(ids)], dtype=torch.long, device=my_device)

    def _greedy(question: str, context: str, new_loradict):
        if query_include_context:
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        else:
            prompt = question
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            enable_thinking=False,
        )
        cur = torch.tensor([ids], dtype=torch.long, device=my_device)
        gen: List[int] = []
        pkv = None
        model._active_w_transform = model.w_transform_conversation
        try:
            for _ in range(max_new):
                out = model.llm.model(
                    input_ids=cur,
                    attention_mask=None,
                    loradict=new_loradict,
                    nograd_loradict=None,
                    nograd_wdict=None,
                    use_cache=True,
                    past_key_values=pkv,
                )
                pkv = out.past_key_values
                nxt = int(model.llm.lm_head(out.last_hidden_state[:, -1, :]).argmax(-1).item())
                if nxt == im_end or nxt == eos_id:
                    break
                gen.append(nxt)
                cur = torch.tensor([[nxt]], dtype=torch.long, device=my_device)
        finally:
            model._active_w_transform = None
        return _strip_think(tok.decode(gen, skip_special_tokens=True))

    outputs = []
    total_f1 = 0.0
    total_em = 0.0

    for idx, ex in enumerate(records):
        context = str(ex["context"]).strip()
        question = str(ex["question"]).strip()
        golds = _answers_text(ex)

        ctx_ids, ctx_lens = _context_tensor(context)
        mem = model.compute_memory_states(
            ctx_ids,
            None,
            ctx_lens,
            nograd_loradict=None,
            nograd_wdict=None,
        )
        loradict = model.generate_loradict(mem)
        new_loradict = {
            layer: concat_loradict([loradict[layer], model.metalora[layer]])
            for layer in range(n_layers)
        }
        pred = _greedy(question, context, new_loradict)

        f1 = max(_f1_score(pred, gold) for gold in golds)
        em = max(_exact_match(pred, gold) for gold in golds)
        total_f1 += f1
        total_em += em
        outputs.append({
            "id": ex.get("id", str(idx)),
            "context": context,
            "question": question,
            "answers": golds,
            "prediction": pred,
            "f1": f1,
            "em": em,
        })

        if (idx + 1) % 50 == 0 or idx + 1 == len(records):
            print(
                f"[squad_qa_gen] {idx + 1}/{len(records)} "
                f"F1={total_f1 / (idx + 1):.4f} EM={total_em / (idx + 1):.4f}",
                flush=True,
            )

    summary = {
        "num_examples": len(outputs),
        "avg_f1": total_f1 / max(1, len(outputs)),
        "avg_em": total_em / max(1, len(outputs)),
        "split": split,
        "query_include_context": query_include_context,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "examples": outputs}, f, ensure_ascii=False, indent=2)

    print(f"\n[squad_qa_gen] summary: {summary}", flush=True)
    print(f"[squad_qa_gen] saved -> {out_path}", flush=True)
    barrier()
