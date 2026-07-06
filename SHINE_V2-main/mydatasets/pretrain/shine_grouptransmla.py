#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHINE-style grouped transmla pretraining dataset for SHINE_V2.

This ports the original SHINE pretraining objective to the V2 batch format:

    context_ids      = a group of raw texts, shuffled and joined with EOT
    conversation_ids = chat turns asking <RECON> or <COMP>, with assistant text
    labels           = assistant-content tokens only; user/pad tokens are -100

The backbone LLM stays frozen in the training loop.  The loss trains the
hypernetwork/metalora to encode context text into generated LoRA parameters.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

import sys as _sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk

from mydatasets.base import BaseCollator, BaseDataset
from utils.mytokenizer import NOTHINKING_CHAT_TEMPLATE, create_tokenizer

logger = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _is_present(value: Any) -> bool:
    return value is not None and str(value).lower() not in ("null", "none", "")


def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_project_root, path)


def _load_text_dataset(data_cfg, split: str = "train"):
    data_path = _resolve_path(data_cfg.get("data_path", "data/transmla_pretrain_6B_tokens"))
    data_format = data_cfg.get("data_format", "hf_dataset")

    if data_format == "hf_dataset":
        return load_dataset(data_path, split=split)
    if data_format == "hf_disk":
        ds = load_from_disk(data_path)
        return ds[split] if hasattr(ds, "keys") and split in ds else ds
    if data_format == "jsonl":
        texts = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                texts.append(str(rec.get("text", "")))
        return HFDataset.from_dict({"text": texts})
    raise ValueError(f"Unsupported data_format={data_format!r}")


def _split_train_val(dataset, validation_split_num, seed: int):
    n = len(dataset)
    if validation_split_num is None or str(validation_split_num).lower() in ("null", "none", "-1"):
        return dataset, None

    if isinstance(validation_split_num, float) or (
        isinstance(validation_split_num, str) and "." in validation_split_num
    ):
        raw = float(validation_split_num)
        val_n = int(n * raw) if 0 < raw < 1 else int(raw)
    else:
        val_n = int(validation_split_num)

    if val_n <= 0:
        return dataset, None
    if val_n >= n:
        raise ValueError(f"validation_split_num ({val_n}) must be smaller than dataset size ({n})")

    split = dataset.train_test_split(test_size=val_n, seed=seed, shuffle=True)
    return split["train"], split["test"]


def _token_lengths(tokenizer, hf_dataset, *, num_proc: int, batch_size: int,
                   max_length: int, cache_file_name: Optional[str] = None) -> List[int]:
    def compute_len(batch):
        enc = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return {"tok_len": [len(ids) for ids in enc["input_ids"]]}

    map_kwargs = {}
    if cache_file_name:
        map_kwargs["cache_file_name"] = cache_file_name
    len_dataset = hf_dataset.map(
        compute_len,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=list(hf_dataset.column_names),
        desc="[shine_grouptransmla] Computing token lengths",
        **map_kwargs,
    )
    return [int(x) for x in len_dataset["tok_len"]]


def _build_groups(token_lens: Sequence[int], *, max_len: int, chat_len: int,
                  num_bins: int = 100) -> List[List[int]]:
    """Greedy bin-packing copied from the original SHINE grouped pretrain."""
    max_body_len = int(max_len)
    groups: List[List[int]] = []
    cache_groups: List[List[int]] = [[] for _ in range(num_bins)]
    cache_left = [max_body_len for _ in range(num_bins)]

    iterator = enumerate(token_lens)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=len(token_lens),
            desc="[shine_grouptransmla] Packing token lengths into groups",
            dynamic_ncols=True,
        )

    for i, tok_len in iterator:
        l = int(tok_len) + int(chat_len)
        if l > max_body_len:
            groups.append([i])
            continue

        placed = False
        for j, left in enumerate(cache_left):
            if l <= left:
                cache_groups[j].append(i)
                cache_left[j] -= l
                placed = True
                break

        if not placed:
            target = int(np.argmin(cache_left))
            if cache_groups[target]:
                groups.append(cache_groups[target])
            cache_groups[target] = [i]
            cache_left[target] = max_body_len - l

    for group in cache_groups:
        if group:
            groups.append(group)
    return groups


def _build_label_mask_chat(tokenizer, conv_ids: List[int]) -> List[int]:
    imstart_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    imend_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")
    nl = tokenizer.encode("\n", add_special_tokens=False)
    nl_id = nl[0] if nl else None

    labels = [-100] * len(conv_ids)
    i = 0
    n = len(conv_ids)
    while i < n:
        if conv_ids[i] == imstart_id and i + 1 < n and conv_ids[i + 1] == assistant_id:
            start = i + 2
            if nl_id is not None and start < n and conv_ids[start] == nl_id:
                start += 1
            end = start
            while end < n and conv_ids[end] != imend_id:
                end += 1
            unmask_end = end + 1 if end < n else end
            for pos in range(start, unmask_end):
                labels[pos] = conv_ids[pos]
            i = unmask_end
        else:
            i += 1
    return labels


class ShineGroupTransMLADataset(BaseDataset):
    def __init__(self, model_path: str, hf_dataset, data_cfg, *, tokenizer_cfg=None,
                 split_name: str = "train", seed: int = 42):
        super().__init__(model_path, tokenizer_cfg=tokenizer_cfg)
        self.tokenizer = create_tokenizer(
            model_path,
            tokenizer_cfg=tokenizer_cfg,
            chat_template=NOTHINKING_CHAT_TEMPLATE,
        )
        self.hf_dataset = hf_dataset
        self.context_seq_length = int(data_cfg.context_seq_length)
        self.conv_seq_length = int(data_cfg.conv_seq_length)
        self.seed = int(seed)
        self.split_name = split_name

        cache_dir = _resolve_path(data_cfg.get("group_cache_dir", "cache/shine_grouptransmla"))
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = data_cfg.get("group_cache_name", "transmla")
        self.cache_path = os.path.join(
            cache_dir,
            f"{cache_name}_{split_name}_groups_ctx{self.context_seq_length}_conv{self.conv_seq_length}.json",
        )

        overwrite = bool(data_cfg.get("overwrite_group_cache", False))
        if os.path.exists(self.cache_path) and not overwrite:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.groups = json.load(f)
            logger.info(
                "[shine_grouptransmla] Loaded %s groups from %s",
                len(self.groups), self.cache_path,
            )
        else:
            num_proc = int(data_cfg.get("map_num_proc", 16))
            batch_size = int(data_cfg.get("map_batch_size", 2048))
            chat_len = int(data_cfg.get("group_chat_len", 11))
            num_bins = int(data_cfg.get("group_num_bins", 100))
            length_cache_path = os.path.join(
                cache_dir,
                f"{cache_name}_{split_name}_toklen_ctx{self.context_seq_length}_conv{self.conv_seq_length}.arrow",
            )
            lengths = _token_lengths(
                self.tokenizer,
                self.hf_dataset,
                num_proc=num_proc,
                batch_size=batch_size,
                max_length=max(self.context_seq_length, self.conv_seq_length),
                cache_file_name=length_cache_path,
            )
            self.groups = _build_groups(
                lengths,
                max_len=self.conv_seq_length,
                chat_len=chat_len,
                num_bins=num_bins,
            )
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.groups, f)
            logger.info(
                "[shine_grouptransmla] Saved %s groups to %s",
                len(self.groups), self.cache_path,
            )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "textlist": [str(self.hf_dataset[int(i)]["text"]) for i in self.groups[idx]],
            "repo": f"{self.split_name}_{idx}",
        }


class ShineGroupTransMLACollator(BaseCollator):
    def __init__(self, model_path: str, data_cfg, *, pad_token_id: int,
                 num_mem_token: int = 0, tokenizer_cfg=None):
        super().__init__(model_path, tokenizer_cfg=tokenizer_cfg)
        self.tokenizer = create_tokenizer(
            model_path,
            tokenizer_cfg=tokenizer_cfg,
            chat_template=NOTHINKING_CHAT_TEMPLATE,
        )
        self.context_seq_length = int(data_cfg.context_seq_length)
        self.conv_seq_length = int(data_cfg.conv_seq_length)
        self.pad_token_id = int(pad_token_id)
        self.num_mem_token = int(num_mem_token)
        self.completion_freq = float(data_cfg.get("completion_freq", 0.5))
        self.min_completion_ratio = float(data_cfg.get("min_completion_ratio", 0.1))
        self.max_completion_ratio = float(data_cfg.get("max_completion_ratio", 0.3))
        self.eot = str(data_cfg.get("group_separator", self.tokenizer.eos_token or "<|endoftext|>"))

    def _split_text(self, text: str):
        words = text.split()
        if len(words) < 2:
            return text, text
        ratio = 1.0 - random.uniform(self.min_completion_ratio, self.max_completion_ratio)
        split_index = round(len(words) * ratio)
        left = words[:split_index]
        right = words[split_index:]
        if not right:
            left, right = words[:-1], words[-1:]
        elif not left:
            left, right = words[:1], words[1:]
        return " ".join(left), " ".join(left + right)

    def _encode_chat(self, messages: List[Dict[str, str]]) -> List[int]:
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
            enable_thinking=False,
        )

    def __call__(self, samples: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        B = len(samples)
        ctx_len = self.context_seq_length
        conv_len = self.conv_seq_length
        context_total_len = ctx_len + self.num_mem_token

        context_ids = torch.full((B, context_total_len), self.pad_token_id, dtype=torch.long)
        conversation_ids = torch.full((B, conv_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, conv_len), -100, dtype=torch.long)
        context_lengths = torch.zeros(B, dtype=torch.long)
        extra_info = []

        for bi, sample in enumerate(samples):
            texts = [str(x) for x in sample["textlist"]]
            evidence_texts = []
            answer_texts = []
            user_texts = []
            for text in texts:
                if random.random() < self.completion_freq:
                    evidence, answer = self._split_text(text)
                    user = "<COMP>"
                else:
                    evidence, answer = text, text
                    user = "<RECON>"
                evidence_texts.append(evidence)
                answer_texts.append(answer)
                user_texts.append(user)

            shuffled_evidence = random.sample(evidence_texts, len(evidence_texts))
            context_text = self.eot.join(shuffled_evidence)
            ctx_ids = self.tokenizer(
                context_text,
                max_length=ctx_len,
                truncation=True,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            indices = list(range(len(texts)))
            random.shuffle(indices)
            messages: List[Dict[str, str]] = []
            for idx in indices:
                messages.append({"role": "user", "content": user_texts[idx]})
                messages.append({"role": "assistant", "content": answer_texts[idx]})
            conv_ids = self._encode_chat(messages)[:conv_len]

            context_ids[bi, :len(ctx_ids)] = torch.tensor(ctx_ids, dtype=torch.long)
            context_lengths[bi] = len(ctx_ids)
            conversation_ids[bi, :len(conv_ids)] = torch.tensor(conv_ids, dtype=torch.long)
            conv_labels = _build_label_mask_chat(self.tokenizer, conv_ids)
            labels[bi, :len(conv_ids)] = torch.tensor(conv_labels, dtype=torch.long)
            extra_info.append({"repo": sample.get("repo", f"sample_{bi}")})

        return [{
            "context_ids": context_ids,
            "conversation_ids": conversation_ids,
            "labels": labels,
            "context_lengths": context_lengths,
            "extra_info": extra_info,
        }]


def create_dataset_and_collator(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    data_cfg = cfg.data
    seed = int(cfg.seed.dataset)
    raw_dataset = _load_text_dataset(data_cfg, split=data_cfg.get("train_split", "train"))
    train_dataset, _ = _split_train_val(
        raw_dataset,
        data_cfg.get("validation_split_num", -1),
        seed,
    )
    dataset = ShineGroupTransMLADataset(
        model_path,
        train_dataset,
        data_cfg,
        tokenizer_cfg=cfg.tokenizer,
        split_name="train",
        seed=seed,
    )
    collator = ShineGroupTransMLACollator(
        model_path,
        data_cfg,
        pad_token_id=pad_token_id,
        num_mem_token=num_mem_token,
        tokenizer_cfg=cfg.tokenizer,
    )
    return dataset, collator


def create_val_dataset(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    data_cfg = cfg.data
    seed = int(cfg.seed.dataset)

    val_file = data_cfg.get("val_file", None)
    if _is_present(val_file):
        val_cfg = dict(data_cfg)
        val_cfg["data_path"] = val_file
        val_cfg["data_format"] = data_cfg.get("val_data_format", "jsonl")
        raw_val = _load_text_dataset(val_cfg, split=data_cfg.get("val_split", "train"))
    else:
        raw_dataset = _load_text_dataset(data_cfg, split=data_cfg.get("train_split", "train"))
        _, raw_val = _split_train_val(
            raw_dataset,
            data_cfg.get("validation_split_num", -1),
            seed,
        )
        if raw_val is None:
            return None

    return ShineGroupTransMLADataset(
        model_path,
        raw_val,
        data_cfg,
        tokenizer_cfg=cfg.tokenizer,
        split_name="val",
        seed=seed,
    )


def preprocess(cfg, model_path: str):
    from utils.mydata import resolve_pad_token_id
    pad_token_id = resolve_pad_token_id(model_path, tokenizer_cfg=cfg.tokenizer)
    dataset, _ = create_dataset_and_collator(cfg, model_path, pad_token_id, num_mem_token=0)
    val_dataset = create_val_dataset(cfg, model_path, pad_token_id, num_mem_token=0)
    logger.info(
        "[shine_grouptransmla] preprocess complete: train_groups=%s val_groups=%s",
        len(dataset),
        len(val_dataset) if val_dataset is not None else 0,
    )


def debug(cfg, model_path: str):
    from utils.mydata import debug_dataset, resolve_pad_token_id
    pad_token_id = resolve_pad_token_id(model_path, tokenizer_cfg=cfg.tokenizer)
    dataset, collator = create_dataset_and_collator(cfg, model_path, pad_token_id, num_mem_token=0)
    debug_dataset(
        dataset=dataset,
        collator=collator,
        tokenizer=create_tokenizer(model_path, tokenizer_cfg=cfg.tokenizer),
        dataset_name=cfg.data.get("name", "shine_grouptransmla"),
        metadata={
            "context_seq_length": cfg.data.context_seq_length,
            "conv_seq_length": cfg.data.conv_seq_length,
            "groups": len(dataset),
            "completion_freq": cfg.data.get("completion_freq", 0.5),
        },
        num_samples=3,
        num_mem_token=0,
        pad_token_id=pad_token_id,
    )
