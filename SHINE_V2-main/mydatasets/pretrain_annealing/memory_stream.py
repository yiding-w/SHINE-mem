#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streaming long-history memory QA dataset for SHINE_V2 (Phase 2).

Turns each synthesized long history (8k–128k tokens) into a SEQUENCE of samples
that share one ``repo`` id and are fed in order (shuffle MUST be false). The
detach_state machinery accumulates the history into W across these samples and
resets when ``repo`` changes.

Per history h with turn-groups g_1..g_K (each <= context_seq_length tokens):
    step i=1..K-1 : context_ids = g_i,  conversation_ids = g_{i+1}   (write W + next-segment recon loss)
    step K (QA)   : context_ids = g_K,  conversation_ids = <multi-QA chat>  (answer-only loss;
                    W already holds g_1..g_{K-1}, g_K is the fresh LoRA -> full history in memory)

Input JSONL = the intermediate format from datagen/generate_memory_data.py:
    {"id", "history_turns":[{"role","content"}], "qa":[{"question","answer",...}], ...}

Label masking keeps ONLY assistant-content tokens (+ trailing <|im_end|>); everything
else is -100. History (context_ids) is never a loss target — it only writes memory.

Required module interface (see utils/mydata.py): create_dataset_and_collator, preprocess, debug.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

import sys as _sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from mydatasets.base import BaseDataset, BaseCollator
from utils.mytokenizer import create_tokenizer, NOTHINKING_CHAT_TEMPLATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _build_label_mask_chat(tokenizer, conv_ids: List[int]) -> List[int]:
    """Keep only assistant-content tokens (+ trailing <|im_end|>); else -100.

    Scans for the ``<|im_start|> assistant`` pattern, unmasks from the newline
    after 'assistant' through the next <|im_end|> (inclusive). Mirrors the
    masking used by mydatasets/sft/msmarco_mqa.py.
    """
    imstart_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    imend_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")
    nl_id = tokenizer.convert_tokens_to_ids("\n")

    ids = conv_ids
    n = len(ids)
    labels = [-100] * n
    i = 0
    while i < n:
        if (ids[i] == imstart_id and i + 1 < n and ids[i + 1] == assistant_id):
            # find newline after 'assistant'
            j = i + 2
            while j < n and ids[j] != nl_id:
                j += 1
            content_start = j + 1  # token after the newline
            k = content_start
            while k < n and ids[k] != imend_id:
                k += 1
            content_end = k  # index of <|im_end|>
            unmask_end = content_end + 1 if content_end < n else content_end
            for t in range(content_start, unmask_end):
                labels[t] = ids[t]
            i = unmask_end
        else:
            i += 1
    return labels


def _encode_turns(tokenizer, turns: List[Dict[str, str]]) -> List[int]:
    """Chat-template a list of {role,content} turns into token ids (no padding)."""
    return tokenizer.apply_chat_template(
        turns, tokenize=True, add_generation_prompt=False,
        return_tensors=None, add_special_tokens=False,
    )


def _group_turns_by_token_budget(tokenizer, turns: List[Dict[str, str]], budget: int
                                 ) -> List[List[Dict[str, str]]]:
    """Greedily pack consecutive turns into groups each <= budget tokens (chat-templated)."""
    groups: List[List[Dict[str, str]]] = []
    cur: List[Dict[str, str]] = []
    cur_len = 0
    for t in turns:
        tlen = len(tokenizer.encode(t["content"], add_special_tokens=False)) + 8  # +template overhead
        if cur and cur_len + tlen > budget:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(t)
        cur_len += tlen
    if cur:
        groups.append(cur)
    return groups


def _qa_to_messages(qa: List[Dict[str, str]]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    for q in qa:
        msgs.append({"role": "user", "content": q["question"]})
        msgs.append({"role": "assistant", "content": str(q["answer"])})
    return msgs


# ---------------------------------------------------------------------------
# Dataset: builds the ordered, repo-contiguous sample list (on-the-fly tokenize)
# ---------------------------------------------------------------------------

class MemoryStreamDataset(BaseDataset):
    def __init__(self, model_path: str, records: List[Dict], context_seq_length: int,
                 tokenizer_cfg=None):
        super().__init__(model_path, tokenizer_cfg=tokenizer_cfg)
        self.tokenizer = create_tokenizer(model_path, tokenizer_cfg=tokenizer_cfg,
                                           chat_template=NOTHINKING_CHAT_TEMPLATE)
        self.budget = int(context_seq_length)
        self.samples: List[Dict[str, Any]] = []
        self._build(records)

    def _build(self, records: List[Dict]) -> None:
        for rec in records:
            repo = str(rec.get("id"))
            turns = rec["history_turns"]
            groups = _group_turns_by_token_budget(self.tokenizer, turns, self.budget)
            if not groups:
                continue
            group_ids = [_encode_turns(self.tokenizer, g) for g in groups]
            qa_ids = _encode_turns(self.tokenizer, _qa_to_messages(rec["qa"]))

            K = len(group_ids)
            # reconstruction steps (write W + predict next segment)
            for i in range(K - 1):
                self.samples.append({
                    "context_token_ids": group_ids[i],
                    "conversation_token_ids": group_ids[i + 1],
                    "repo": repo,
                })
            # final QA step: context = last group, conversation = all QA
            self.samples.append({
                "context_token_ids": group_ids[K - 1],
                "conversation_token_ids": qa_ids,
                "repo": repo,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "context_token_ids": torch.tensor(s["context_token_ids"], dtype=torch.long),
            "conversation_token_ids": torch.tensor(s["conversation_token_ids"], dtype=torch.long),
            "repo": s["repo"],
        }


class MemoryStreamCollator(BaseCollator):
    def __init__(self, model_path: str, context_seq_length: int, conv_seq_length: int,
                 pad_token_id: int = 0, num_mem_token: int = 0, tokenizer_cfg=None):
        super().__init__(model_path, tokenizer_cfg=tokenizer_cfg)
        self.tokenizer = create_tokenizer(model_path, tokenizer_cfg=tokenizer_cfg,
                                           chat_template=NOTHINKING_CHAT_TEMPLATE)
        self.context_seq_length = int(context_seq_length)
        self.conv_seq_length = int(conv_seq_length)
        self.pad_token_id = pad_token_id
        self.num_mem_token = num_mem_token

    def __call__(self, samples: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        B = len(samples)
        ctx_len = self.context_seq_length
        conv_len = self.conv_seq_length
        context_total_len = ctx_len + self.num_mem_token

        context_ids = torch.full((B, context_total_len), self.pad_token_id, dtype=torch.long)
        conversation_ids = torch.full((B, conv_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, conv_len), -100, dtype=torch.long)
        context_lengths = torch.zeros(B, dtype=torch.long)
        extra_info_list = []

        for i, s in enumerate(samples):
            c = s["context_token_ids"][:ctx_len]
            context_ids[i, :c.size(0)] = c
            context_lengths[i] = c.size(0)

            v = s["conversation_token_ids"][:conv_len]
            conversation_ids[i, :v.size(0)] = v
            lbl = _build_label_mask_chat(self.tokenizer, v.tolist())
            labels[i, :v.size(0)] = torch.tensor(lbl, dtype=torch.long)

            extra_info_list.append({"repo": s["repo"]})

        return [{
            "context_ids": context_ids,
            "conversation_ids": conversation_ids,
            "labels": labels,
            "context_lengths": context_lengths,
            "extra_info": extra_info_list,
        }]


# ---------------------------------------------------------------------------
# Module interface
# ---------------------------------------------------------------------------

def _data_files(data_cfg) -> Tuple[str, Optional[str]]:
    data_path = data_cfg.get("data_path")
    train = os.path.join(data_path, data_cfg.get("train_file", "train.jsonl"))
    val_name = data_cfg.get("val_file")
    val = os.path.join(data_path, val_name) if val_name else None
    return train, val


def preprocess(cfg, model_path: str):
    """No offline preprocessing: histories are tokenized on the fly at load time."""
    logger.info("[memory_stream] preprocess: on-the-fly tokenization, nothing to cache.")
    return None


def create_dataset_and_collator(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    data_cfg = cfg.data
    train_path, _ = _data_files(data_cfg)
    records = _load_jsonl(train_path)
    ctx_len = int(data_cfg.context_seq_length)
    conv_len = int(data_cfg.conv_seq_length)
    tok_cfg = cfg.get("tokenizer", None)

    dataset = MemoryStreamDataset(model_path, records, ctx_len, tokenizer_cfg=tok_cfg)
    collator = MemoryStreamCollator(model_path, ctx_len, conv_len, pad_token_id=pad_token_id,
                                    num_mem_token=num_mem_token, tokenizer_cfg=tok_cfg)
    logger.info(f"[memory_stream] {len(records)} histories -> {len(dataset)} stream samples "
                f"(ctx_len={ctx_len}, conv_len={conv_len})")
    return dataset, collator


def create_val_dataset(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """Return a single val Dataset (or None). Harness reuses the train collator."""
    data_cfg = cfg.data
    _, val_path = _data_files(data_cfg)
    if not val_path or not os.path.isfile(val_path):
        return None
    records = _load_jsonl(val_path)
    ctx_len = int(data_cfg.context_seq_length)
    tok_cfg = cfg.get("tokenizer", None)
    return MemoryStreamDataset(model_path, records, ctx_len, tokenizer_cfg=tok_cfg)


def debug(cfg, model_path: str):
    ds, col = create_dataset_and_collator(cfg, model_path, pad_token_id=0, num_mem_token=0)
    print(f"[memory_stream:debug] {len(ds)} samples")
    batch = col([ds[i] for i in range(min(3, len(ds)))])[0]
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {v}")
    n_loss = int((batch["labels"] != -100).sum())
    print(f"  loss tokens in first {batch['labels'].size(0)} samples: {n_loss}")
