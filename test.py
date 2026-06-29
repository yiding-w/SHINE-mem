#!/usr/bin/env python
# -*- coding: utf-8 -*-

from csv import writer
import os
import math
import time
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Union
from functools import partial
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    AutoTokenizer,
    Qwen3ForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from datasets import Dataset as HFDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import importlib
from omegaconf import DictConfig, OmegaConf
import hydra
from datasets import load_dataset
import logging
from torch.utils.tensorboard import SummaryWriter
from metanetwork_family import Metanetwork

from utils.mydataset import HotpotqaDataset, SquadDataset, SquadCollator, GroupedSquadDataset, MsmarcoDataset, MusiqueDataset
from utils.myseed import set_seed
from utils.mylogging import get_logger
from utils.mysaveload import (
    save_checkpoint,
    load_checkpoint,
    save_training_state,
    load_training_state,
    get_latest_checkpoint,
)
from utils.myfreeze import freeze
from utils.myoptmize import init_optimize
from utils.myddp import (
    should_use_ddp,
    ddp_is_active,
    get_world_size,
    get_rank,
    get_local_rank,
    is_main_process,
    ddp_init_if_needed,
    ddp_cleanup_if_needed,
    distributed_mean,
    barrier,
)
from utils.myinit import _resolve_device, _import_class
from utils.myloradict import merge_loradicts
from collections import OrderedDict
import re

# NEW: compute_f1 import (your SQuAD-style scorer)
from calculate_f1 import compute_f1
from evaluation.hotpotqa import f1_score as hotpotqa_compute_f1

logger = get_logger("test")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True


def extract_think_and_answer(text: str) -> Tuple[str, str]:
    """
    Splits model output into (think_part, answer_part).
    If no valid <think>...</think> block exists, think = "".
    """

    lower = text.lower()
    start_tag = "<think>"
    end_tag = "</think>"

    think = ""
    answer = text.strip()

    # ---- Case 1: Proper <think>...</think> block exists ----
    start = lower.find(start_tag)
    end = lower.find(end_tag)
    if start != -1 and end != -1 and end > start:
        think = text[start + len(start_tag) : end].strip()
        answer = text[end + len(end_tag) :].strip()
    else:
        # ---- Case 2: No valid think block → think = "" ----
        answer = re.sub(
            r"<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL
        ).strip()
        think = ""

    # ---- Clean common prefixes like "Answer:" or "Final answer:" ----
    answer = re.sub(r"^(final answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE).strip()

    # ---- Take only the first non-empty line as final answer ----
    if "\n" in answer:
        for line in answer.splitlines():
            if line.strip():
                answer = line.strip()
                break

    return think, answer


def _to_answer_list(ground_truth: Any) -> List[str]:
    """
    Normalize possible ground_truth formats into a list[str].
    Handles common SQuAD-ish cases:
      - str
      - list[str]
      - dict with keys like "text"/"answers"
      - other -> stringified fallback
    """
    if ground_truth is None:
        return [""]

    if isinstance(ground_truth, str):
        return [ground_truth]

    if isinstance(ground_truth, (list, tuple)):
        out: List[str] = []
        for x in ground_truth:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict):
                if "text" in x and isinstance(x["text"], str):
                    out.append(x["text"])
                elif "answer" in x and isinstance(x["answer"], str):
                    out.append(x["answer"])
                else:
                    out.append(str(x))
            else:
                out.append(str(x))
        return out if len(out) > 0 else [""]

    if isinstance(ground_truth, dict):
        # common: {"text": [...]} or {"text": "..."}, or {"answers": {"text":[...]}}
        if "text" in ground_truth:
            if isinstance(ground_truth["text"], str):
                return [ground_truth["text"]]
            if isinstance(ground_truth["text"], (list, tuple)):
                return [str(t) for t in ground_truth["text"]] if len(ground_truth["text"]) else [""]
        if "answers" in ground_truth and isinstance(ground_truth["answers"], dict):
            ans = ground_truth["answers"]
            if "text" in ans:
                if isinstance(ans["text"], str):
                    return [ans["text"]]
                if isinstance(ans["text"], (list, tuple)):
                    return [str(t) for t in ans["text"]] if len(ans["text"]) else [""]
        # fallback
        return [str(ground_truth)]

    return [str(ground_truth)]


def compute_sample_f1(ground_truth: Any, pred: str, f1_metric) -> float:
    """
    SQuAD-style: if multiple gold answers exist, take max F1 over golds.
    """
    golds = _to_answer_list(ground_truth)
    if pred is None:
        pred = ""
    # max over golds
    best = 0.0
    for g in golds:
        try:
            best = max(best, float(f1_metric(g, pred)))
        except Exception:
            # very defensive fallback
            best = max(best, 0.0)
    return best


@torch.no_grad()
def test_and_save(
    cfg,
    metanetwork_ddp_or_module,
    tokenizer,
    testloader,
    split_name: str,
    f1_metric,
    use_metanet: bool = True,
    metalora: Any = None,
    device: torch.device = "cuda",
    output_suffix: str = ".json",
):
    """
    Run inference on `testloader`, stream results to disk (per-rank JSONL),
    support resuming from partial output, and finally gather & save a merged
    JSON file on rank 0.

    New:
      - Compute F1 for every sample, store as record["f1"].
      - After gathering, compute avg_f1 and save it to:
          {out_dir}/{split_name}_results.json

    Resumability:
      - Per rank we keep an intermediate file:
          {cfg.test.save_path}/{cfg.test.source}/{split_name}.rank{rank}.jsonl
      - Every written record has a monotonically increasing `sample_idx`.
      - On resume, we read this file, find the max existing `sample_idx`,
        and skip earlier samples in the DataLoader.
      - Final merged file {split_name}{output_suffix} does NOT contain
        `sample_idx`; it’s only used for resuming and ordering.
    """

    if use_metanet:
        assert metalora is not None, "metalora cannot be None when use_metanet is True"

    rank = get_rank()
    world_size = get_world_size()

    # Handle both wrapped and unwrapped metanetwork
    metanet = (
        metanetwork_ddp_or_module.module
        if isinstance(metanetwork_ddp_or_module, DDP)
        else metanetwork_ddp_or_module
    )
    metanet.eval()

    # ---------- Paths ----------
    out_dir = os.path.join(cfg.test.save_path, cfg.test.source)
    final_out_path = os.path.join(out_dir, f"{split_name}{output_suffix}")
    rank_tmp_path = os.path.join(out_dir, f"{split_name}.rank{rank}.jsonl")
    # NEW: summary results file
    results_out_path = os.path.join(out_dir, f"{split_name}_results.json")

    # Make sure directory exists on all ranks
    if is_main_process():
        os.makedirs(out_dir, exist_ok=True)
    if ddp_is_active():
        dist.barrier()

    # ---------- Figure out where to resume ----------
    start_sample_idx = 0
    if os.path.exists(rank_tmp_path):
        with open(rank_tmp_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "sample_idx" in rec:
                    start_sample_idx = max(start_sample_idx, rec["sample_idx"] + 1)
        if is_main_process():
            logger.info(
                f"[Rank {rank}] Resuming from sample_idx={start_sample_idx} for split '{split_name}'"
            )

    # Open rank tmp file for appending
    tmp_f = open(rank_tmp_path, "a", encoding="utf-8")

    sample_idx = 0  # global (per-rank) index of samples seen by this rank

    for batch_idx, batch in enumerate(testloader):
        batch_size = len(batch["questions"])

        # If this entire batch is already processed, skip without running the model
        if sample_idx + batch_size <= start_sample_idx:
            sample_idx += batch_size
            continue

        print(f"[Rank {rank}] Processing batch {batch_idx + 1}/{len(testloader)}...")

        evidences = batch["evidence"]
        evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
        evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        input_attention_mask = batch["input_attention_mask"].to(device, non_blocking=True)
        ground_truths = batch["full_answers"]
        questions = batch["questions"]
        labels = None if batch["labels"] is None else batch["labels"].to(device, non_blocking=True)

        loradict = None
        if use_metanet:
            loradict = metanet.generate_lora_dict(
                evidence_ids=evidence_ids,
                evidence_attention_mask=evidence_attention_mask,
                metalora=metalora,
            )

        gen_out = metanet.metamodel.generate(
            input_ids=input_ids,
            attention_mask=input_attention_mask,
            loradict=loradict,
            ignore_mem_token=True,
            max_new_tokens=cfg.test.max_new_tokens,
            do_sample=False,
        )

        input_lens = input_attention_mask.sum(dim=1).tolist()

        gen_out = gen_out.to("cpu")
        input_ids_cpu = input_ids.to("cpu")

        for i in range(gen_out.size(0)):
            # If this particular sample was already written in previous run, skip it
            if sample_idx < start_sample_idx:
                sample_idx += 1
                continue

            full_text = tokenizer.decode(gen_out[i], skip_special_tokens=True)
            input_text = tokenizer.decode(
                input_ids_cpu[i][-input_lens[i] :], skip_special_tokens=True
            )

            if full_text.startswith(input_text):
                answer_text = full_text[len(input_text) :]
            else:
                answer_text = full_text

            think, answer = extract_think_and_answer(answer_text)

            # NEW: compute per-sample f1 (max over golds if multiple)
            gt = ground_truths[i]
            f1_val = compute_sample_f1(gt, answer, f1_metric)

            record = {
                "sample_idx": sample_idx,  # used for resuming and sorting
                "evidence": evidences[i],
                "input": input_text,
                "question": questions[i],
                "think": think,
                "answer": answer,
                "ground_truth": gt,
                "f1": f1_val,  # NEW
            }

            tmp_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            tmp_f.flush()

            sample_idx += 1

    tmp_f.close()
    metanet.train()

    # ---------- Final gather & merged save ----------
    local_results = []
    if os.path.exists(rank_tmp_path):
        with open(rank_tmp_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                local_results.append(rec)

    if ddp_is_active():
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_results)

        if is_main_process():
            merged = []
            for part in gathered:
                if part:
                    merged.extend(part)
    else:
        merged = local_results

    if is_main_process():
        merged.sort(key=lambda x: x.get("sample_idx", 0))

        # NEW: compute avg f1 on merged
        f1_vals = []
        for rec in merged:
            if "f1" in rec and isinstance(rec["f1"], (int, float)):
                f1_vals.append(float(rec["f1"]))
        avg_f1 = float(sum(f1_vals) / max(1, len(f1_vals)))

        # remove sample_idx in final predictions file
        for rec in merged:
            rec.pop("sample_idx", None)

        # Save predictions (with f1 per sample)
        with open(final_out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        # Save summary results
        summary = {
            "dataset": split_name,
            "num_samples": len(merged),
            "avg_f1": avg_f1,
        }
        with open(results_out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(merged)} predictions to {final_out_path}")
        logger.info(f"Saved summary results to {results_out_path} (avg_f1={avg_f1:.6f})")


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    # ========= DDP init (safe for single-process) =========
    ddp_init_if_needed()

    if is_main_process():
        logger.info("Resolved config:")
        logger.info(f"\n\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Seed & device
    set_seed(int(cfg.run.seed) + get_rank())
    device = _resolve_device(cfg.run.device)
    torch.backends.cudnn.benchmark = True

    # Load model/tokenizer (supports your local LoRA-wrapped Qwen class)
    if is_main_process():
        logger.info("Loading model & tokenizer...")
    MetaModelCls = _import_class(cfg.model.metamodel_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)
    config = ConfigCls.from_pretrained(cfg.model.model_from)
    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers

    if cfg.metanetwork.type == "transformer":
        tmp_model = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
        assert tmp_model.lora_params_numel(cfg.model.lora_r) % (
            cfg.hidden_size * cfg.num_layers
        ) == 0, (
            "For transformer metanetwork, num_mem_token must be set to "
            "model.lora_params_numel(lora_r) / (hidden_size * num_layers)"
        )
        config.num_mem_token = (
            tmp_model.lora_params_numel(cfg.model.lora_r)
            // (cfg.hidden_size * cfg.num_layers)
        )
        cfg.num_mem_token = config.num_mem_token
        del tmp_model
        if is_main_process():
            logger.info(
                f"Using transformer metanetwork, set num_mem_token to {config.num_mem_token}"
            )
    else:
        config.num_mem_token = cfg.num_mem_token

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_from, padding_side="left", use_fast=True
    )
    metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.train()
    metanetwork.to(device)
    freeze(metamodel)

    # Training loop scaffolding
    hydra_run_dir = os.getcwd()
    ckpt_root = os.path.join("checkpoints", f"{cfg.name}", "train")

    test_checkpoint_dir = cfg.test.get("checkpoint_dir", None)
    if test_checkpoint_dir:
        resume_dir = str(test_checkpoint_dir)
        if not os.path.isabs(resume_dir):
            resume_dir = os.path.abspath(resume_dir)
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    elif cfg.test_global_step == "latest":
        resume_dir = get_latest_checkpoint(ckpt_root)
    elif cfg.test_global_step == "final":
        resume_dir = os.path.join(ckpt_root, "final")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    elif isinstance(cfg.test_global_step, int) and cfg.test_global_step > 0:
        resume_dir = os.path.join(ckpt_root, f"checkpoint-{cfg.test_global_step}")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    elif isinstance(cfg.test_global_step, str) and cfg.test_global_step.startswith("epoch-"):
        resume_dir = os.path.join(ckpt_root, f"checkpoint-{cfg.test_global_step}")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    else:
        raise ValueError(f"Invalid test_global_step: {cfg.test_global_step}")

    # Load model
    USE_ADDITIONAL_METALORA = bool(cfg.model.ift_additional_metalora_r >= 0)
    if is_main_process():
        logger.info(f"Resume mode, loading from {resume_dir}...")
    metanetwork, metalora, ift_additional_metalora = load_checkpoint(
        metanetwork,
        resume_dir,
        device,
        load_ift_additional_metalora=USE_ADDITIONAL_METALORA,
        zero_ift_additional_metalora=(cfg.model.ift_additional_metalora_r == 0),
    )
    if USE_ADDITIONAL_METALORA:
        metalora = merge_loradicts(metalora, ift_additional_metalora)

    # Data
    if is_main_process():
        logger.info("Preparing data...")
    if cfg.test.source == "squad":
        f1_metric=compute_f1
        names = [f"squad_{cfg.test.context_avg_len}"]
        datasets = []
        for testset in names:
            data = load_dataset(
                os.path.join("data", "squad"),
                split="validation",
            )
            if cfg.test.get("shuffle", True):
                data = data.shuffle(seed=42)
            N = int(cfg.test.get("num_samples", 1000))
            subset = data.select(range(N))
            datasets.append(GroupedSquadDataset(subset, tokenizer, cfg.test.context_avg_len))
            if is_main_process():
                logger.info(
                    f"Loaded {cfg.test.source}/{testset} with {len(subset)} samples "
                    f"(from {len(data)}, shuffle={cfg.test.get('shuffle', True)})"
                )
        collator = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length,
            cfg=cfg,
        )
        collator_no_metanet = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length + cfg.test.context_max_length,
            cfg=cfg,
            use_reference=True,
        )
        collator_only_question = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length,
            cfg=cfg,
            only_question=True,
        )
    elif cfg.test.source == "hotpotqa":
        f1_metric=hotpotqa_compute_f1
        names = [f"hotpotqa"]
        datasets = []
        for testset in names:
            data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
            data = data.shuffle(seed=42)
            N = 1000
            subset = data.select(range(N))
            datasets.append(HotpotqaDataset(subset))
            if is_main_process():
                logger.info(f"Loaded {cfg.test.source}/{testset} with {len(data)} samples")
        collator = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length,
            cfg=cfg,
        )
        collator_no_metanet = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length + cfg.test.context_max_length,
            cfg=cfg,
            use_reference=True,
        )
        collator_only_question = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length,
            cfg=cfg,
            only_question=True,
        )
    elif cfg.test.source == "musique":
        f1_metric=compute_f1
        names = [f"musique"]
        datasets = []
        for testset in names:
            data = load_dataset("dgslibisey/MuSiQue", split="validation")
            data = data.shuffle(seed=42)
            N = 1000
            subset = data.select(range(N))
            datasets.append(MusiqueDataset(subset))
            if is_main_process():
                logger.info(f"Loaded {cfg.test.source}/{testset} with {len(data)} samples")
        collator = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length,
            cfg=cfg,
        )
        collator_no_metanet = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length + cfg.test.context_max_length,
            cfg=cfg,
            use_reference=True,
        )
        collator_only_question = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length,
            cfg=cfg,
            only_question=True,
        )
    elif cfg.test.source == "2wikimultihopqa":
        f1_metric=hotpotqa_compute_f1
        names = [f"2wikimultihopqa"]
        datasets = []
        for testset in names:
            data = load_dataset("framolfese/2WikiMultihopQA", split="validation")
            data = data.shuffle(seed=42)
            N = 1000
            subset = data.select(range(N))
            datasets.append(HotpotqaDataset(subset))
            if is_main_process():
                logger.info(f"Loaded {cfg.test.source}/{testset} with {len(data)} samples")
        collator = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length,
            cfg=cfg,
        )
        collator_no_metanet = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length + cfg.test.context_max_length,
            cfg=cfg,
            use_reference=True,
        )
        collator_only_question = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length,
            cfg=cfg,
            only_question=True,
        )
    elif cfg.test.source == "msmarco_v1":
        f1_metric=compute_f1
        names = [f"msmarco_v1"]
        datasets = []
        for testset in names:
            data = load_dataset('microsoft/ms_marco', 'v1.1', split='test')
            data = data.shuffle(seed=42)
            N = 1000
            subset = data.select(range(N))
            datasets.append(MsmarcoDataset(subset))
            if is_main_process():
                logger.info(f"Loaded {cfg.test.source}/{testset} with {len(data)} samples")
        collator = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length,
            cfg=cfg,
        )
        collator_no_metanet = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length + cfg.test.context_max_length,
            cfg=cfg,
            use_reference=True,
        )
        collator_only_question = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length,
            cfg=cfg,
            only_question=True,
        )
    elif cfg.test.source == "msmarco_v2":
        f1_metric=compute_f1
        names = [f"msmarco_v2"]
        datasets = []
        for testset in names:
            data = load_dataset('microsoft/ms_marco', 'v2.1', split='validation')
            data = data.shuffle(seed=42)
            N = 1000
            subset = data.select(range(N))
            datasets.append(MsmarcoDataset(subset))
            if is_main_process():
                logger.info(f"Loaded {cfg.test.source}/{testset} with {len(data)} samples")
        collator = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length,
            cfg=cfg,
        )
        collator_no_metanet = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length + cfg.test.context_max_length,
            cfg=cfg,
            use_reference=True,
        )
        collator_only_question = SquadCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.test.context_max_length,
            conversation_max_length=cfg.test.conversation_max_length,
            cfg=cfg,
            only_question=True,
        )
        
    else:
        raise ValueError(f"Unknown data source: {cfg.test.source}")

    pin = device.type == "cuda"
    for i, ds in enumerate(datasets):
        test_sampler = (
            DistributedSampler(ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
            if get_world_size() > 1
            else None
        )
        num_workers_default = 2 if device.type == "cuda" else 0

        test_loader = DataLoader(
            ds,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            sampler=test_sampler,
            collate_fn=collator,
            pin_memory=pin,
            num_workers=getattr(cfg.test, "num_workers", num_workers_default),
            persistent_workers=pin and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        )
        test_loader_no_metanet = DataLoader(
            ds,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            sampler=test_sampler,
            collate_fn=collator_no_metanet,
            pin_memory=pin,
            num_workers=getattr(cfg.test, "num_workers", num_workers_default),
            persistent_workers=pin and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        )
        test_loader_only_question = DataLoader(
            ds,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            sampler=test_sampler,
            collate_fn=collator_only_question,
            pin_memory=pin,
            num_workers=getattr(cfg.test, "num_workers", num_workers_default),
            persistent_workers=pin and getattr(cfg.test, "num_workers", num_workers_default) > 0,
        )

        ckpt_root = os.path.join(hydra_run_dir, "checkpoints")

        if ddp_is_active():
            dist.barrier()

        eval_mode = str(cfg.test.get("eval_mode", "shine")).lower()
        valid_modes = {"shine", "plain", "context", "lora_context", "all"}
        if eval_mode not in valid_modes:
            raise ValueError(f"Unknown test.eval_mode={eval_mode}. Expected one of {sorted(valid_modes)}")

        jobs = []
        if eval_mode in {"shine", "all"}:
            jobs.append((test_loader, names[i], True, metalora))
        if eval_mode in {"plain", "all"}:
            jobs.append((test_loader_only_question, f"{names[i]}_only_question", False, None))
        if eval_mode in {"context", "all"}:
            jobs.append((test_loader_no_metanet, f"{names[i]}_no_metanet", False, None))
        if eval_mode in {"lora_context", "all"}:
            jobs.append((test_loader_no_metanet, f"{names[i]}_lora_context", True, metalora))

        for loader, split_name, use_metanet, job_metalora in jobs:
            test_and_save(
                cfg=cfg,
                metanetwork_ddp_or_module=metanetwork,
                tokenizer=tokenizer,
                testloader=loader,
                split_name=split_name,
                f1_metric=f1_metric,
                use_metanet=use_metanet,
                metalora=job_metalora,
                device=device,
                output_suffix=".json",
            )

    ddp_cleanup_if_needed()


if __name__ == "__main__":
    main()

