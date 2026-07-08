from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def qa_to_messages(qa: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for item in qa:
        messages.append({"role": "user", "content": str(item["question"])})
        messages.append({"role": "assistant", "content": str(item["answer"])})
    return messages


def turns_to_text(turns: List[Dict[str, str]]) -> str:
    parts = []
    for turn in turns:
        role = str(turn.get("role", "user"))
        content = str(turn.get("content", ""))
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


class MemoryStreamV1Dataset(Dataset):
    """
    SHINE v1-compatible view of SHINE v2 segmented memory-stream JSONL.

    Each history record is expanded into ordered stream samples:
      segment history -> per-segment QA loss
      last segment also gets final_qa when include_final_qa=True.

    Samples keep a repo id so the training loop can reset detach_state when
    moving between histories. Keep shuffle disabled for this dataset.
    """

    def __init__(self, records: List[Dict[str, Any]], *, include_final_qa: bool = False):
        self.include_final_qa = bool(include_final_qa)
        self.samples: List[Dict[str, Any]] = []
        self._build(records)

    def _build(self, records: List[Dict[str, Any]]) -> None:
        for rec in records:
            repo = str(rec.get("id", len(self.samples)))
            segments = rec["segments"]
            final_qa = rec.get("final_qa", [])
            last_idx = len(segments) - 1
            for seg_idx, seg in enumerate(segments):
                qa = list(seg.get("qa", []))
                if self.include_final_qa and seg_idx == last_idx and final_qa:
                    qa = qa + list(final_qa)
                self.samples.append({
                    "repo": repo,
                    "segment_idx": seg_idx,
                    "evidence": turns_to_text(seg.get("history_turns", [])),
                    "conversations": qa_to_messages(qa),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


@dataclass
class MemoryStreamV1Collator:
    tokenizer: Any
    context_max_length: int = 2048
    conversation_max_length: int = 1024
    cfg: Any = None

    def __post_init__(self):
        self.assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        self.imstart_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.imend_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def mask_label(self, labels):
        masks = torch.zeros_like(labels)
        for i, row in enumerate(labels):
            last_imend = labels.shape[1]
            for j in range(len(row) - 1, 0, -1):
                if row[j].item() == self.imend_token_id:
                    last_imend = j
                elif row[j].item() == self.assistant_token_id and row[j - 1].item() == self.imstart_token_id:
                    masks[i, j + 2:last_imend + 2] = 1
        return labels.masked_fill(masks == 0, -100)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        evidence_texts = [item["evidence"] for item in batch]
        messages = [item["conversations"] for item in batch]

        evidence_enc = self.tokenizer(
            evidence_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )

        input_enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            max_length=self.conversation_max_length,
            truncation=True,
            return_dict=True,
            padding="max_length",
            enable_thinking=False,
        )
        input_ids = input_enc["input_ids"]
        labels = self.mask_label(input_ids.clone())

        return {
            "evidence": evidence_texts,
            "evidence_ids": evidence_enc["input_ids"],
            "evidence_attention_mask": evidence_enc["attention_mask"],
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_enc["attention_mask"],
            "repo": [item["repo"] for item in batch],
            "segment_idx": [item["segment_idx"] for item in batch],
        }


def load_memory_stream_v1(
    data_path: str,
    file_name: str,
    *,
    include_final_qa: bool,
) -> MemoryStreamV1Dataset:
    path = file_name if os.path.isabs(file_name) else os.path.join(data_path, file_name)
    return MemoryStreamV1Dataset(load_jsonl(path), include_final_qa=include_final_qa)
