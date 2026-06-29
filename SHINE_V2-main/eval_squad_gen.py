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


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _strip_think(text: str) -> str:
    text = text.strip()
    if text.startswith("<think>") and "</think>" not in text:
        return ""
    m = re.split(r"</think>", text, maxsplit=1)
    out = (m[1] if len(m) > 1 else m[0]).strip()
    out = re.sub(r"^<think>\s*", "", out, flags=re.IGNORECASE).strip()
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
    plain_baseline = os.environ.get("SQUAD_PLAIN_BASELINE", "0") == "1"
    context_baseline = os.environ.get("SQUAD_CONTEXT_BASELINE", "0") == "1"
    eval_mode = os.environ.get("SQUAD_EVAL_MODE", "").strip().lower()
    if not eval_mode:
        eval_mode = "both" if plain_baseline else "shine"
        if context_baseline:
            eval_mode = "all" if eval_mode == "both" else "shine_context"
    valid_modes = {
        "shine",
        "plain",
        "context",
        "both",
        "all",
        "shine_context",
        "lora_context",
        "shine_lora_context",
    }
    if eval_mode not in valid_modes:
        raise ValueError(f"Unknown SQUAD_EVAL_MODE='{eval_mode}'. Expected one of {sorted(valid_modes)}")

    run_shine = eval_mode in {
        "shine",
        "both",
        "all",
        "shine_context",
        "lora_context",
        "shine_lora_context",
    }
    run_plain = eval_mode in {"plain", "both", "all"}
    run_context = eval_mode in {"context", "all", "shine_context"}
    shine_include_context = query_include_context or eval_mode in {"lora_context", "shine_lora_context"}

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
    banned_token_ids = set()
    for token in ("<think>", "</think>"):
        tid = tok.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            banned_token_ids.add(tid)
        ids = tok(token, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        if len(ids) == 1:
            banned_token_ids.add(int(ids[0]))

    records = _load_squad_records(data_path, split)
    if limit > 0:
        records = records[:limit]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(
        f"\n[squad_qa_gen] {len(records)} examples split={split} data={data_path} "
        f"ctx_cap={ctx_len_cap} max_new={max_new} "
        f"eval_mode={eval_mode} prompt_context={run_context} "
        f"shine_query_include_context={shine_include_context}",
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

    system_prompt = (
        "You are a concise assistant. Output only the final answer, "
        "in a few words, as short as possible. No explanations. "
        "Do not output anything else."
    )

    def _greedy(question: str, context: str, new_loradict, *, include_context: bool):
        if include_context:
            prompt = (
                f"Reference:\n{context}\n\n"
                f"Based on the reference, answer this question:\n{question}"
            )
        else:
            prompt = question
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        ids = tok.apply_chat_template(
            messages,
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
                logits = model.llm.lm_head(out.last_hidden_state[:, -1, :])
                if banned_token_ids:
                    logits[:, list(banned_token_ids)] = -torch.inf
                nxt = int(logits.argmax(-1).item())
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
    plain_total_f1 = 0.0
    plain_total_em = 0.0
    context_total_f1 = 0.0
    context_total_em = 0.0

    iterator = enumerate(records)
    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            iterator,
            total=len(records),
            desc="SQuAD eval",
            dynamic_ncols=True,
        )
        iterator = pbar

    for idx, ex in iterator:
        context = str(ex["context"]).strip()
        question = str(ex["question"]).strip()
        golds = _answers_text(ex)

        row = {
            "id": ex.get("id", str(idx)),
            "context": context,
            "question": question,
            "answers": golds,
        }

        if run_shine:
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
            pred = _greedy(question, context, new_loradict, include_context=shine_include_context)
            f1 = max(_f1_score(pred, gold) for gold in golds)
            em = max(_exact_match(pred, gold) for gold in golds)
            total_f1 += f1
            total_em += em
            row.update({
                "prediction": pred,
                "f1": f1,
                "em": em,
            })

        if run_plain:
            plain_pred = _greedy(question, context, None, include_context=False)
            plain_f1 = max(_f1_score(plain_pred, gold) for gold in golds)
            plain_em = max(_exact_match(plain_pred, gold) for gold in golds)
            plain_total_f1 += plain_f1
            plain_total_em += plain_em
            row.update({
                "plain_prediction": plain_pred,
                "plain_f1": plain_f1,
                "plain_em": plain_em,
            })

        if run_context:
            context_pred = _greedy(question, context, None, include_context=True)
            context_f1 = max(_f1_score(context_pred, gold) for gold in golds)
            context_em = max(_exact_match(context_pred, gold) for gold in golds)
            context_total_f1 += context_f1
            context_total_em += context_em
            row.update({
                "context_prediction": context_pred,
                "context_f1": context_f1,
                "context_em": context_em,
            })

        outputs.append(row)

        if pbar is not None:
            postfix = {}
            if run_shine:
                postfix["f1"] = f"{total_f1 / (idx + 1):.4f}"
                postfix["em"] = f"{total_em / (idx + 1):.4f}"
            if run_plain:
                postfix["plain_f1"] = f"{plain_total_f1 / (idx + 1):.4f}"
                postfix["plain_em"] = f"{plain_total_em / (idx + 1):.4f}"
            if run_context:
                postfix["ctx_f1"] = f"{context_total_f1 / (idx + 1):.4f}"
                postfix["ctx_em"] = f"{context_total_em / (idx + 1):.4f}"
            pbar.set_postfix(postfix)
        if (idx + 1) % 50 == 0 or idx + 1 == len(records):
            msg = f"[squad_qa_gen] {idx + 1}/{len(records)}"
            if run_shine:
                msg += f" | SHINE F1={total_f1 / (idx + 1):.4f} EM={total_em / (idx + 1):.4f}"
            if run_plain:
                msg += (
                    f" | plain_noctx F1={plain_total_f1 / (idx + 1):.4f} "
                    f"EM={plain_total_em / (idx + 1):.4f}"
                )
            if run_context:
                msg += (
                    f" | plain_ctx F1={context_total_f1 / (idx + 1):.4f} "
                    f"EM={context_total_em / (idx + 1):.4f}"
                )
            print(msg, flush=True)

    summary = {
        "num_examples": len(outputs),
        "split": split,
        "eval_mode": eval_mode,
        "query_include_context": shine_include_context,
        "plain_baseline": run_plain,
        "context_baseline": run_context,
    }
    if run_shine:
        summary.update({
            "avg_f1": total_f1 / max(1, len(outputs)),
            "avg_em": total_em / max(1, len(outputs)),
        })
    if run_plain:
        summary.update({
            "plain_noctx_avg_f1": plain_total_f1 / max(1, len(outputs)),
            "plain_noctx_avg_em": plain_total_em / max(1, len(outputs)),
        })
    if run_context:
        summary.update({
            "plain_ctx_avg_f1": context_total_f1 / max(1, len(outputs)),
            "plain_ctx_avg_em": context_total_em / max(1, len(outputs)),
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "examples": outputs}, f, ensure_ascii=False, indent=2)

    print(f"\n[squad_qa_gen] summary: {summary}", flush=True)
    print(f"[squad_qa_gen] saved -> {out_path}", flush=True)
    barrier()
