#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streaming long-history memory QA dataset for SHINE_V2 (Phase 2).

Turns each synthesized long history (8k–128k tokens) into a SEQUENCE of samples
that share one ``repo`` id and are fed in order (shuffle MUST be false). The
detach_state machinery accumulates the history into W across these samples and
resets when ``repo`` changes.

Per history (segments s_1..s_K, one stream step each, all sharing repo=id):
    step i : context_ids = s_i.history  (writes s_i into W),
             conversation_ids = s_i per-segment QA  (answer-only loss)
    last step additionally appends final_qa (cross-segment: conflict/multi-hop/
             temporal/TTL) only when detach_state is active -> exercises reading
             the accumulated W.

WHY per-segment QA (not next-segment reconstruction): detach_state stores W with
requires_grad=False, so the final QA loss cannot backprop into earlier steps. Each
step needs its OWN loss requiring THAT segment's facts, so the hypernetwork actually
learns to WRITE each segment into W.

Input JSONL = the SEGMENTED format from datagen/generate_memory_seg.py:
    {"id", "segments":[{"date","history_turns":[{role,content}],"qa":[{question,answer,...}]}],
     "final_qa":[{question,answer,...}], ...}

Label masking keeps ONLY assistant-content tokens (+ trailing <|im_end|>); everything
else is -100. History (context_ids) is never a loss target — it only writes memory.

Required module interface (see utils/mydata.py): create_dataset_and_collator, preprocess, debug.
"""

from __future__ import annotations

import json
import logging
import os
import random
import zlib
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
    # "\n" is NOT a standalone vocab token in Qwen — use encode(...)[0], and degrade
    # gracefully if absent (matches mydatasets/sft/msmarco_mqa.py). Using
    # convert_tokens_to_ids("\n") here returns the wrong id -> all labels become -100.
    _nl = tokenizer.encode("\n", add_special_tokens=False)
    nl_id = _nl[0] if _nl else None

    ids = list(conv_ids)
    n = len(ids)
    labels = [-100] * n
    i = 0
    while i < n:
        if ids[i] == imstart_id and i + 1 < n and ids[i + 1] == assistant_id:
            header_end = i + 2
            if nl_id is not None and header_end < n and ids[header_end] == nl_id:
                header_end += 1  # skip the newline after 'assistant' if present
            content_start = header_end
            content_end = content_start
            while content_end < n and ids[content_end] != imend_id:
                content_end += 1
            unmask_end = content_end + 1 if content_end < n else content_end
            for t in range(content_start, unmask_end):
                labels[t] = ids[t]
            i = content_end + 1
        else:
            i += 1
    return labels


def _encode_turns(tokenizer, turns: List[Dict[str, str]]) -> List[int]:
    """Chat-template a list of {role,content} turns into token ids (no padding).

    Match the args used by mydatasets/sft/msmarco_mqa* under transformers 5.5.4.
    Passing add_special_tokens/return_tensors here makes 5.5.4's apply_chat_template
    skip tokenization and return strings (-> 'str' object cannot be interpreted as int).
    """
    return tokenizer.apply_chat_template(
        turns,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )


def _qa_to_messages(qa: List[Dict[str, str]]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    for q in qa:
        msgs.append({"role": "user", "content": q["question"]})
        msgs.append({"role": "assistant", "content": str(q["answer"])})
    return msgs


def _include_final_qa(cfg) -> bool:
    """Cross-segment final_qa is only learnable when detach_state accumulates W.
    Override with MEM_FINAL_QA=1 (force on, e.g. to measure final_qa on the
    no-W baseline) or MEM_FINAL_QA=0 (force off); unset -> auto by detach_state."""
    env = os.environ.get("MEM_FINAL_QA", "")
    if env == "1":
        return True
    if env == "0":
        return False
    ds_cfg = cfg.get("detach_state", None)
    if ds_cfg is None:
        return False
    return str(ds_cfg.get("type", "empty")) != "empty"


# ---------------------------------------------------------------------------
# Dataset: builds the ordered, repo-contiguous sample list (on-the-fly tokenize)
# ---------------------------------------------------------------------------

class MemoryStreamDataset(BaseDataset):
    def __init__(self, model_path: str, records: List[Dict], context_seq_length: int,
                 tokenizer_cfg=None, include_final_qa: bool = False,
                 apply_deferred: bool = False):
        super().__init__(model_path, tokenizer_cfg=tokenizer_cfg)
        self.tokenizer = create_tokenizer(model_path, tokenizer_cfg=tokenizer_cfg,
                                           chat_template=NOTHINKING_CHAT_TEMPLATE)
        self.budget = int(context_seq_length)
        self.include_final_qa = bool(include_final_qa)
        # Deferred cross-segment QA augmentation (MEM_DEFERRED_QA) is applied to
        # TRAIN only; val stays untouched so val_ppl / save_best stays meaningful.
        self.apply_deferred = apply_deferred
        self.samples: List[Dict[str, Any]] = []
        self._build(records)

    def _build(self, records: List[Dict]) -> None:
        """One stream sample per segment: context=segment history (writes W),
        conversation=per-segment QA (answer-only loss). The LAST segment also
        carries the cross-segment final_qa (W then holds all prior segments).

        MEM_DEFERRED_QA=N (env, default 0): for each segment si>0, also append N
        questions drawn from EARLIER segments (j<si). At this stream position W
        holds segments 0..si-1, so the answer lives ONLY in accumulated W -> this
        forces the hypernetwork to actually learn cross-segment retrieval (raises
        the cross-segment QA ratio WITHOUT regenerating data). Selection is
        crc32-seeded per (repo, si) so it is identical across DP/TP ranks."""
        n_deferred = int(os.environ.get("MEM_DEFERRED_QA", "0")) if self.apply_deferred else 0
        for rec in records:
            repo = str(rec.get("id"))
            segments = rec["segments"]
            final_qa = rec.get("final_qa", [])
            K = len(segments)
            for si, seg in enumerate(segments):
                ctx_ids = _encode_turns(self.tokenizer, seg["history_turns"])
                qa = list(seg.get("qa", []))
                if self.include_final_qa and si == K - 1 and final_qa:
                    qa = qa + final_qa  # cross-segment QA exercises reading accumulated W
                if n_deferred > 0 and si > 0:
                    pool = []
                    for pj in range(si):  # only EARLIER segments (answer is in W)
                        for q in segments[pj].get("qa", []):
                            q2 = dict(q)
                            q2["type"] = str(q.get("type", "segment_retrieval")) + "_deferred"
                            pool.append(q2)
                    if pool:
                        rng = random.Random(zlib.crc32(f"{repo}:{si}".encode()))
                        qa = qa + rng.sample(pool, min(n_deferred, len(pool)))
                conv_ids = _encode_turns(self.tokenizer, _qa_to_messages(qa))
                self.samples.append({
                    "context_token_ids": ctx_ids,
                    "conversation_token_ids": conv_ids,
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


def filter_by_segments(records: List[Dict]) -> List[Dict]:
    """Keep only histories whose n_segments is in MEM_SEGMENTS (e.g. "8,16").
    Lets you train/eval on just the short (easy) histories without regenerating
    data. Unset/empty -> keep everything. Falls back to len(segments) if a
    record has no n_segments field."""
    raw = os.environ.get("MEM_SEGMENTS", "").strip()
    if not raw:
        return records
    keep = {int(x) for x in raw.split(",") if x.strip()}
    out = [r for r in records
           if int(r.get("n_segments", len(r.get("segments", [])))) in keep]
    logger.info(f"[memory_stream] MEM_SEGMENTS={sorted(keep)} -> kept {len(out)}/{len(records)} histories")
    return out


def create_dataset_and_collator(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    data_cfg = cfg.data
    train_path, _ = _data_files(data_cfg)
    records = filter_by_segments(_load_jsonl(train_path))
    ctx_len = int(data_cfg.context_seq_length)
    conv_len = int(data_cfg.conv_seq_length)
    tok_cfg = cfg.get("tokenizer", None)
    include_final = _include_final_qa(cfg)

    dataset = MemoryStreamDataset(model_path, records, ctx_len, tokenizer_cfg=tok_cfg,
                                  include_final_qa=include_final,
                                  apply_deferred=True)
    collator = MemoryStreamCollator(model_path, ctx_len, conv_len, pad_token_id=pad_token_id,
                                    num_mem_token=num_mem_token, tokenizer_cfg=tok_cfg)
    logger.info(f"[memory_stream] {len(records)} histories -> {len(dataset)} stream samples "
                f"(ctx_len={ctx_len}, conv_len={conv_len}, include_final_qa={include_final})")
    return dataset, collator


def create_val_dataset(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """Return a single val Dataset (or None). Harness reuses the train collator."""
    data_cfg = cfg.data
    _, val_path = _data_files(data_cfg)
    if not val_path or not os.path.isfile(val_path):
        return None
    records = filter_by_segments(_load_jsonl(val_path))
    ctx_len = int(data_cfg.context_seq_length)
    tok_cfg = cfg.get("tokenizer", None)
    include_final = _include_final_qa(cfg)
    logger.info(f"[memory_stream] val include_final_qa={include_final}")
    return MemoryStreamDataset(model_path, records, ctx_len, tokenizer_cfg=tok_cfg,
                               include_final_qa=include_final)


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
