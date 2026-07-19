#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import string
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from MemoryTest.prepare_data.prompt_templates import format_answer, question_prompt, reconstruction_prompt
from MemoryTest.evaluation.metrics import make_eval_row, summarize_examples
from MemoryTest.training.lora_sft_utils import load_runtime_args
from MemoryTest.training.recurrent_data import (
    accumulated_qa,
    load_recurrent_dataset,
    sample_training_stream,
    sample_turn_qa,
)
from MemoryTest.training.shine_train_utils import (
    append_jsonl,
    cast_floating_tensors,
    clamp_lora_tensors,
    compute_combined_lora_loss,
    lora_tensor_stats,
    recurrent_memory_norm_stats,
    resolve_path,
    save_posttrain_checkpoint,
    set_posttrain_requires_grad,
    trainable_generate_context_lora,
    trainable_update_recurrent_memory,
)


LOGGER = logging.getLogger("posttrain_shine_memory")

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(args: argparse.Namespace) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(enabled=False, rank=0, local_rank=0, world_size=1)
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun distributed training requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Non-main ranks wait while rank 0 runs generation-based validation and checkpoint I/O.
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=2))
    args.device = "cuda"
    args.gpu_id = local_rank
    return DistributedContext(
        enabled=True,
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
    )


def distributed_barrier(context: DistributedContext) -> None:
    if context.enabled:
        dist.barrier(device_ids=[context.local_rank])


def distributed_any(value: bool, device: torch.device, context: DistributedContext) -> bool:
    if not context.enabled:
        return value
    flag = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def distributed_mean(value: float, device: torch.device, context: DistributedContext) -> float:
    if not context.enabled:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor / context.world_size).item())


def distributed_mean_optional(value: float | None, device: torch.device, context: DistributedContext) -> float | None:
    if not context.enabled:
        return value
    pair = torch.tensor(
        [0.0 if value is None else value, 0.0 if value is None else 1.0],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    return float((pair[0] / pair[1]).item()) if pair[1].item() > 0 else None


def trainable_parameters(trainable: dict[str, list[torch.Tensor]]) -> list[torch.Tensor]:
    return [parameter for group in trainable.values() for parameter in group]


def synchronize_gradients(
    parameters: list[torch.Tensor],
    device: torch.device,
    context: DistributedContext,
    bucket_size_mb: int,
) -> None:
    """Average registered hypernetwork and external MetaLoRA gradients across ranks."""
    if not context.enabled:
        return
    presence = torch.tensor(
        [1 if parameter.grad is not None else 0 for parameter in parameters],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(presence, op=dist.ReduceOp.SUM)
    active_parameters = []
    for index, parameter in enumerate(parameters):
        active_ranks = int(presence[index].item())
        if active_ranks == 0:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        active_parameters.append(parameter)

    max_bucket_bytes = max(1, bucket_size_mb) * 1024 * 1024
    bucket: list[torch.Tensor] = []
    bucket_bytes = 0

    def flush() -> None:
        nonlocal bucket, bucket_bytes
        if not bucket:
            return
        gradients = [parameter.grad for parameter in bucket]
        flat = torch._utils._flatten_dense_tensors(gradients)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(context.world_size)
        for gradient, averaged in zip(gradients, torch._utils._unflatten_dense_tensors(flat, gradients)):
            gradient.copy_(averaged)
        bucket = []
        bucket_bytes = 0

    for parameter in active_parameters:
        gradient = parameter.grad
        gradient_bytes = gradient.numel() * gradient.element_size()
        if bucket and (gradient.dtype != bucket[0].grad.dtype or bucket_bytes + gradient_bytes > max_bucket_bytes):
            flush()
        bucket.append(parameter)
        bucket_bytes += gradient_bytes
    flush()


def init_wandb(args: argparse.Namespace, context: DistributedContext):
    if not context.is_main or not args.wandb_project or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb or omit --wandb-project") from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        tags=args.wandb_tags or None,
        mode=args.wandb_mode,
        dir=str(resolve_path(args.output_dir)),
        config=vars(args),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-train SHINE on MemoryTest semantic facts.")
    parser.add_argument("--runtime-config", "--config", dest="runtime_config", type=str, default="MemoryTest/config/case_test.yaml")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument(
        "--checkpoint-profile",
        choices=["auto", "pretrain", "ift"],
        default="auto",
        help=(
            "Construction profile for the input SHINE checkpoint. pretrain exactly follows "
            "test_pretrain.py; ift follows inference.ipynb; auto infers from the path."
        ),
    )
    parser.add_argument("--train-file", type=str, default="")
    parser.add_argument("--val-file", type=str, default="MemoryTest/json_data/splits/semantic_val.json")
    parser.add_argument("--test-file", type=str, default="MemoryTest/json_data/splits/semantic_test.json")
    parser.add_argument("--output-dir", type=str, default="MemoryTest/checkpoints/shine_memory_posttrain")
    parser.add_argument("--fact-counts", type=int, nargs="+", default=[1, 2, 4, 8, 12, 20])
    parser.add_argument("--qa-per-context", type=int, default=4)
    parser.add_argument("--context-format", choices=["natural", "structured", "mixed"], default="mixed")
    parser.add_argument("--recurrent-steps", type=int, default=4, help="Number of context chunks in one memory stream.")
    parser.add_argument(
        "--ordered-stream-probability",
        "--ordered-stream-prob",
        dest="ordered_stream_probability",
        type=float,
        default=0.5,
        help="When a file contains both streams and facts, probability of sampling an ordered stream.",
    )
    parser.add_argument(
        "--turn-supervision",
        choices=["final", "every"],
        default="final",
        help="Apply readout loss only after the final chunk or after every recurrent chunk.",
    )
    parser.add_argument(
        "--turn-objective",
        choices=["qa", "reconstruction", "both", "mixed"],
        default="qa",
        help="Per-turn readout objective. mixed samples QA or reconstruction independently each turn.",
    )
    parser.add_argument(
        "--qa-turn-prob",
        type=float,
        default=0.5,
        help="Probability of choosing QA when --turn-objective mixed.",
    )
    parser.add_argument(
        "--reconstruction-scope",
        choices=["current", "cumulative"],
        default="current",
        help="Whether <RECON> repeats the current chunk or all chunks observed so far.",
    )
    parser.add_argument(
        "--qa-scope",
        choices=["current", "cumulative"],
        default="cumulative",
        help="Sample QA attached only to the current turn or to all turns observed so far.",
    )
    parser.add_argument(
        "--detach-recurrent-memory-every",
        type=int,
        default=0,
        help="Detach recurrent K/V every N chunks; 0 keeps full truncated-BPTT across the stream.",
    )
    parser.add_argument(
        "--memory-position-offset",
        type=int,
        default=None,
        help="Fixed RoPE position of memory slot 0; defaults to context_max_length.",
    )
    parser.add_argument(
        "--stream-window-policy",
        choices=["contiguous", "prefix", "full"],
        default="contiguous",
        help="How to select turns when an ordered stream is longer than recurrent_steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Streams accumulated per optimizer step on each rank; global batch equals this times world size.",
    )
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--metalora-learning-rate", type=float, default=None)
    parser.add_argument("--mem-token-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=500, help="Validation interval; 0 disables evaluation.")
    parser.add_argument(
        "--eval-at-start",
        action="store_true",
        help="Evaluate the untouched input checkpoint at step 0 before any optimizer update.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate the untouched input checkpoint at step 0 and exit without training or saving weights.",
    )
    parser.add_argument("--eval-trials", type=int, default=20)
    parser.add_argument("--eval-fact-counts", type=int, nargs="+", default=None)
    parser.add_argument("--eval-example-max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-example-max-chars", type=int, default=2000)
    parser.add_argument(
        "--print-eval-example",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print deterministic reconstruction reference/prediction pairs for every supervised turn in one stream.",
    )
    parser.add_argument(
        "--eval-objective",
        choices=["auto", "qa", "reconstruction", "both"],
        default="auto",
        help=(
            "Validation readout objective. auto mirrors --turn-objective; mixed validates both. "
            "Reconstruction validation uses teacher-forced token loss rather than generation."
        ),
    )
    parser.add_argument(
        "--best-metric",
        choices=["auto", "qa_accuracy", "reconstruction_loss"],
        default="auto",
        help="Metric used to select output-dir/best. auto uses reconstruction loss for reconstruction-only training.",
    )
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answer-max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context-max-length", type=int, default=None)
    parser.add_argument("--conversation-max-length", type=int, default=None)
    parser.add_argument("--torch-dtype", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], default="bf16")
    parser.add_argument("--use-gradient-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-answer-gradient-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--empty-cache-every", type=int, default=0)
    parser.add_argument("--use-contrastive", action="store_true")
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--use-reconstruction", action="store_true")
    parser.add_argument("--recon-weight", type=float, default=0.2)
    parser.add_argument("--generated-lora-clamp", type=float, default=10.0)
    parser.add_argument("--freeze-metalora", action="store_true")
    parser.add_argument(
        "--train-mem-tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also train the zero memory-token embeddings; disabled by default.",
    )
    parser.add_argument("--freeze-mem-tokens", action="store_false", dest="train_mem_tokens", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--distributed-bucket-mb", type=int, default=64)
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    return parser.parse_args()


def resolve_turn_objective(args: argparse.Namespace, rng: random.Random) -> str:
    objective = args.turn_objective
    if objective == "mixed":
        objective = "qa" if rng.random() < args.qa_turn_prob else "reconstruction"
    # Backward compatibility with the old flag, which meant QA + reconstruction.
    if args.use_reconstruction and objective == "qa":
        objective = "both"
    return objective


def resolve_eval_objective(args: argparse.Namespace) -> str:
    if args.eval_objective != "auto":
        return args.eval_objective
    objective = args.turn_objective
    if args.use_reconstruction and objective == "qa":
        objective = "both"
    return "both" if objective == "mixed" else objective


def resolve_best_metric(args: argparse.Namespace, eval_objective: str) -> tuple[str, bool]:
    metric = args.best_metric
    if metric == "auto":
        metric = "reconstruction_loss" if eval_objective == "reconstruction" else "qa_accuracy"
    if metric == "qa_accuracy" and eval_objective not in {"qa", "both"}:
        raise ValueError("--best-metric qa_accuracy requires QA validation")
    if metric == "reconstruction_loss" and eval_objective not in {"reconstruction", "both"}:
        raise ValueError("--best-metric reconstruction_loss requires reconstruction validation")
    return metric, metric == "qa_accuracy"


def build_training_records(
    qa_candidates: list[dict],
    qa_rows: list[dict],
    use_contrastive: bool,
    objective: str,
    reconstruction_text: str,
) -> list[dict]:
    records = []
    include_qa = objective in {"qa", "both"}
    include_reconstruction = objective in {"reconstruction", "both"}
    if include_qa:
        for row in qa_rows:
            records.append(
                {
                    "category": "answer",
                    "prompt": question_prompt(row["question"]),
                    "answer": " " + format_answer(row["answer"]),
                }
            )
    if include_qa and use_contrastive:
        letters = list(string.ascii_uppercase)
        for qa in qa_rows:
            candidates = []
            seen = set()
            for row in [qa] + qa_candidates:
                answer = str(row["answer"])
                if answer.casefold() not in seen:
                    candidates.append(answer)
                    seen.add(answer.casefold())
            candidates = candidates[: min(len(candidates), len(letters))]
            gold_idx = candidates.index(str(qa["answer"]))
            option_lines = [f"{letters[idx]}. {answer}" for idx, answer in enumerate(candidates)]
            prompt = (
                "Choose the option that answers the question using the current memory.\n\n"
                f"Question: {qa['question']}\n"
                + "\n".join(option_lines)
                + "\nAnswer with the option letter only."
            )
            records.append(
                {
                    "category": "contrastive",
                    "prompt": prompt,
                    "answer": " " + letters[gold_idx],
                }
            )
    if include_reconstruction:
        records.append(
            {
                "category": "reconstruction",
                "prompt": reconstruction_prompt(),
                # Official SHINE pretraining reconstructs evidence text after <RECON>.
                "answer": reconstruction_text,
            }
        )
    return records


def default_train_file() -> str:
    augmented = resolve_path("MemoryTest/json_data/splits/semantic_train_augmented.json")
    if augmented.exists():
        return str(augmented)
    return "MemoryTest/json_data/splits/semantic_train.json"


def load_shine_for_training(args: argparse.Namespace):
    from MemoryTest.case_test import (
        build_cfg,
        load_runtime,
        resolve_checkpoint_profile,
        resolve_device,
        resolve_torch_dtype,
    )

    runtime_args = load_runtime_args(args.runtime_config)
    checkpoint_profile = resolve_checkpoint_profile(
        args.checkpoint_dir,
        getattr(args, "checkpoint_profile", "auto"),
    )
    checkpoint_dir = args.checkpoint_dir
    latest_dir = resolve_path(args.output_dir) / "latest"
    if args.resume and latest_dir.exists():
        checkpoint_dir = str(latest_dir)
    runtime_args.checkpoint_dir = checkpoint_dir
    if args.device is not None:
        runtime_args.device = args.device
    if args.gpu_id is not None:
        runtime_args.gpu_id = args.gpu_id
    if args.context_max_length is not None:
        runtime_args.context_max_length = args.context_max_length
    if args.conversation_max_length is not None:
        runtime_args.conversation_max_length = args.conversation_max_length
    device = resolve_device(runtime_args.device, runtime_args.gpu_id)
    cfg = build_cfg(runtime_args)
    cfg.model.torch_dtype = args.torch_dtype
    metanetwork, metalora, tokenizer = load_runtime(
        cfg,
        runtime_args.checkpoint_dir,
        device,
        checkpoint_profile=checkpoint_profile,
    )
    dtype = resolve_torch_dtype(args.torch_dtype)
    if isinstance(dtype, torch.dtype):
        metanetwork.to(device=device, dtype=dtype)
        metalora = cast_floating_tensors(metalora, dtype)
    if hasattr(metanetwork.metamodel, "config"):
        metanetwork.metamodel.config.use_cache = False
    if hasattr(metanetwork.metamodel, "gradient_checkpointing_enable") and args.use_answer_gradient_checkpoint:
        metanetwork.metamodel.gradient_checkpointing_enable()
    metanetwork.train()
    trainable = set_posttrain_requires_grad(
        metanetwork,
        metalora,
        train_metalora=not args.freeze_metalora,
        train_mem_tokens=args.train_mem_tokens,
    )
    return runtime_args, cfg, device, metanetwork, metalora, tokenizer, trainable


def build_optimizer(trainable: dict[str, list[torch.Tensor]], args: argparse.Namespace):
    param_groups = []
    if trainable["metanetwork"]:
        param_groups.append(
            {
                "params": trainable["metanetwork"],
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "name": "metanetwork",
            }
        )
    if trainable["mem_tokens"]:
        param_groups.append(
            {
                "params": trainable["mem_tokens"],
                "lr": args.mem_token_learning_rate if args.mem_token_learning_rate is not None else args.learning_rate,
                "weight_decay": args.weight_decay,
                "name": "mem_tokens",
            }
        )
    if trainable["metalora"]:
        param_groups.append(
            {
                "params": trainable["metalora"],
                "lr": args.metalora_learning_rate if args.metalora_learning_rate is not None else args.learning_rate,
                "weight_decay": args.weight_decay,
                "name": "metalora",
            }
        )
    if not param_groups:
        raise ValueError("No trainable parameters selected.")
    LOGGER.info(
        "Optimizer groups: %s",
        ", ".join(f"{group['name']} n={sum(param.numel() for param in group['params'])} lr={group['lr']}" for group in param_groups),
    )
    return torch.optim.AdamW(param_groups)


def encode_choice_batch(tokenizer, context_rows: list[dict], qa_rows: list[dict], max_length: int, device):
    import string

    letters = list(string.ascii_uppercase)
    input_rows = []
    label_rows = []
    for qa in qa_rows:
        candidates = []
        seen = set()
        for row in [qa] + context_rows:
            answer = str(row["answer"])
            if answer.casefold() not in seen:
                candidates.append(answer)
                seen.add(answer.casefold())
        candidates = candidates[: min(len(candidates), len(letters))]
        gold_idx = candidates.index(str(qa["answer"]))
        option_lines = [f"{letters[idx]}. {answer}" for idx, answer in enumerate(candidates)]
        prompt = (
            "Choose the option that answers the question using the current memory.\n\n"
            f"Question: {qa['question']}\n"
            + "\n".join(option_lines)
            + "\nAnswer with the option letter only."
        )
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        answer_ids = tokenizer(" " + letters[gold_idx] + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + answer_ids)[-max_length:]
        labels = ([-100] * len(prompt_ids) + answer_ids)[-max_length:]
        input_rows.append(input_ids)
        label_rows.append(labels)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(row) for row in input_rows)
    return {
        "input_ids": torch.tensor([[pad_id] * (max_len - len(row)) + row for row in input_rows], dtype=torch.long, device=device),
        "labels": torch.tensor([[-100] * (max_len - len(row)) + labels for row, labels in zip(input_rows, label_rows)], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([[0] * (max_len - len(row)) + [1] * len(row) for row in input_rows], dtype=torch.long, device=device),
    }


def compute_contrastive_loss(context_rows, qa_rows, lora_dict, metanetwork, tokenizer, device, max_length, use_gradient_checkpoint: bool = False):
    batch = encode_choice_batch(tokenizer, context_rows, qa_rows, max_length=max_length, device=device)
    outputs = metanetwork.metamodel(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        loradict=lora_dict,
        ignore_mem_token=True,
        use_gradient_checkpoint=use_gradient_checkpoint,
    )
    return outputs.loss


def generate_answer(metanetwork, tokenizer, lora_dict, question: str, device, max_new_tokens: int, max_length: int):
    from MemoryTest.case_test import extract_think_and_answer
    from MemoryTest.prepare_data.prompt_templates import question_prompt

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": question_prompt(question)}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        return_dict=True,
        padding="max_length",
        enable_thinking=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = metanetwork.metamodel.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            ignore_mem_token=True,
            loradict=lora_dict,
        )
    raw = tokenizer.decode(outputs[0, input_ids.shape[1] :], skip_special_tokens=True)
    _, answer = extract_think_and_answer(raw)
    return answer, raw


def generate_reconstruction(metanetwork, tokenizer, lora_dict, device, max_new_tokens: int) -> str:
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": reconstruction_prompt()}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = metanetwork.metamodel.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            use_cache=True,
            ignore_mem_token=True,
            loradict=lora_dict,
        )
    return tokenizer.decode(outputs[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


def evaluate_current(metanetwork, metalora, tokenizer, cfg, data, args, runtime_args, device, rng) -> dict:
    metanetwork.eval()
    eval_rows = []
    reconstruction_loss_sum = 0.0
    reconstruction_tokens = 0
    reconstruction_readouts = 0
    reconstruction_examples = []
    eval_objective = resolve_eval_objective(args)
    evaluate_qa = eval_objective in {"qa", "both"}
    evaluate_reconstruction = eval_objective in {"reconstruction", "both"}
    eval_fact_counts = args.eval_fact_counts if args.eval_fact_counts is not None else args.fact_counts
    progress = tqdm(
        range(args.eval_trials),
        desc="posttrain val",
        dynamic_ncols=True,
        leave=False,
    ) if tqdm is not None else range(args.eval_trials)
    memory_position_offset = args.memory_position_offset or cfg.test.context_max_length
    with torch.no_grad():
        for trial in progress:
            stream = sample_training_stream(
                data,
                recurrent_steps=args.recurrent_steps,
                fact_counts=eval_fact_counts,
                context_format=args.context_format,
                ordered_stream_probability=args.ordered_stream_probability,
                window_policy=args.stream_window_policy,
                rng=rng,
            )
            recurrent_memory = None
            observed_turns = []
            for turn_index, turn in enumerate(stream.turns):
                observed_turns.append(turn)
                is_final_turn = turn_index + 1 == len(stream.turns)
                supervise_turn = args.turn_supervision == "every" or is_final_turn
                if supervise_turn:
                    lora_dict, recurrent_memory = trainable_generate_context_lora(
                        turn.text,
                        metanetwork,
                        tokenizer,
                        metalora,
                        cfg,
                        device,
                        recurrent_memory=recurrent_memory,
                        return_recurrent_state=True,
                        memory_position_offset=memory_position_offset,
                    )
                    lora_dict = clamp_lora_tensors(lora_dict, args.generated_lora_clamp)
                    if evaluate_reconstruction:
                        reconstruction_text = (
                            turn.text
                            if args.reconstruction_scope == "current"
                            else "\n".join(item.text for item in observed_turns)
                        )
                        reconstruction_records = build_training_records(
                            qa_candidates=[],
                            qa_rows=[],
                            use_contrastive=False,
                            objective="reconstruction",
                            reconstruction_text=reconstruction_text,
                        )
                        reconstruction_losses, token_counts = compute_combined_lora_loss(
                            reconstruction_records,
                            lora_dict,
                            metanetwork,
                            tokenizer,
                            device,
                            args.answer_max_length,
                            use_gradient_checkpoint=False,
                            return_token_counts=True,
                        )
                        token_count = token_counts["reconstruction"]
                        readout_loss = float(reconstruction_losses["reconstruction"].detach().cpu())
                        reconstruction_loss_sum += readout_loss * token_count
                        reconstruction_tokens += token_count
                        reconstruction_readouts += 1
                        if (
                            trial == 0
                            and args.eval_example_max_new_tokens > 0
                        ):
                            prediction = generate_reconstruction(
                                metanetwork,
                                tokenizer,
                                lora_dict,
                                device,
                                args.eval_example_max_new_tokens,
                            )
                            reconstruction_examples.append({
                                "stream_id": stream.stream_id,
                                "turn": turn_index + 1,
                                "turn_id": turn.turn_id,
                                "loss": readout_loss,
                                "perplexity": math.exp(min(readout_loss, 20.0)),
                                "target_tokens": token_count,
                                "reference": reconstruction_text[: args.eval_example_max_chars],
                                "prediction": prediction[: args.eval_example_max_chars],
                                "reference_truncated": len(reconstruction_text) > args.eval_example_max_chars,
                                "prediction_truncated": len(prediction) > args.eval_example_max_chars,
                            })
                else:
                    recurrent_memory = trainable_update_recurrent_memory(
                        turn.text,
                        metanetwork,
                        tokenizer,
                        metalora,
                        cfg,
                        device,
                        recurrent_memory=recurrent_memory,
                        memory_position_offset=memory_position_offset,
                    )
                if supervise_turn and evaluate_qa:
                    qa_source_turns = [turn] if args.qa_scope == "current" else observed_turns
                    qa_rows = sample_turn_qa(qa_source_turns, args.qa_per_context, rng)
                    for fact in qa_rows:
                        answer, raw = generate_answer(
                            metanetwork,
                            tokenizer,
                            lora_dict,
                            fact["question"],
                            device,
                            args.max_new_tokens or runtime_args.max_new_tokens,
                            runtime_args.conversation_max_length,
                        )
                        eval_rows.append(make_eval_row(len(eval_rows) + 1, fact, answer, raw=raw))
            if tqdm is not None:
                postfix = {}
                if evaluate_qa:
                    qa_summary = summarize_examples(eval_rows)
                    postfix.update({"acc": f"{qa_summary['accuracy']:.4f}", "qa_n": qa_summary["total"]})
                if evaluate_reconstruction and reconstruction_tokens:
                    postfix.update({"recon": f"{reconstruction_loss_sum / reconstruction_tokens:.4f}"})
                progress.set_postfix(postfix)
    metanetwork.train()
    summary = {"objective": eval_objective}
    if evaluate_qa:
        summary.update(summarize_examples(eval_rows))
    if evaluate_reconstruction:
        reconstruction_loss = reconstruction_loss_sum / reconstruction_tokens if reconstruction_tokens else float("nan")
        summary.update(
            {
                "reconstruction_loss": reconstruction_loss,
                "reconstruction_perplexity": math.exp(min(reconstruction_loss, 20.0)),
                "reconstruction_tokens": reconstruction_tokens,
                "reconstruction_readouts": reconstruction_readouts,
            }
        )
        if reconstruction_examples:
            summary["reconstruction_examples"] = reconstruction_examples
            # Compatibility with older log consumers that expected one final-turn example.
            summary["reconstruction_example"] = reconstruction_examples[-1]
    return summary


def validation_metric(val_summary: dict, best_metric_name: str) -> float:
    value = val_summary["accuracy"] if best_metric_name == "qa_accuracy" else val_summary["reconstruction_loss"]
    if not math.isfinite(value):
        raise FloatingPointError(f"Validation metric {best_metric_name} is non-finite: {value}")
    return value


def report_validation(step: int, val_summary: dict, best_metric_name: str, args: argparse.Namespace) -> float:
    current_metric = validation_metric(val_summary, best_metric_name)
    LOGGER.info(
        "val step=%s objective=%s %s=%.6f reconstruction_ppl=%s",
        step,
        val_summary["objective"],
        best_metric_name,
        current_metric,
        (
            f"{val_summary['reconstruction_perplexity']:.6f}"
            if "reconstruction_perplexity" in val_summary
            else "n/a"
        ),
    )
    if args.print_eval_example:
        examples = val_summary.get("reconstruction_examples", [])
        if not examples and val_summary.get("reconstruction_example") is not None:
            examples = [val_summary["reconstruction_example"]]
        for example in examples:
            LOGGER.info(
                "val reconstruction example step=%s stream=%s turn=%s turn_id=%s "
                "loss=%.6f ppl=%.6f target_tokens=%s\n"
                "----- reference%s -----\n%s\n"
                "----- prediction%s -----\n%s\n"
                "----- end reconstruction example -----",
                step,
                example["stream_id"],
                example.get("turn", "n/a"),
                example["turn_id"],
                example.get("loss", float("nan")),
                example.get("perplexity", float("nan")),
                example.get("target_tokens", "n/a"),
                " (truncated for log)" if example["reference_truncated"] else "",
                example["reference"],
                " (truncated for log)" if example["prediction_truncated"] else "",
                example["prediction"],
            )
    return current_metric


def validation_wandb_payload(val_summary: dict, best_metric_name: str, current_metric: float) -> dict:
    payload = {f"val/{best_metric_name}": current_metric}
    for key in (
        "accuracy",
        "reconstruction_loss",
        "reconstruction_perplexity",
        "reconstruction_tokens",
        "reconstruction_readouts",
    ):
        if key in val_summary:
            payload[f"val/{key}"] = val_summary[key]
    for example in val_summary.get("reconstruction_examples", []):
        turn = example.get("turn", len(payload))
        prefix = f"val/reconstruction_turn_{turn}"
        payload[f"{prefix}_loss"] = example["loss"]
        payload[f"{prefix}_perplexity"] = example["perplexity"]
        payload[f"{prefix}_reference"] = example["reference"]
        payload[f"{prefix}_prediction"] = example["prediction"]
    return payload


def compute_stream_training_result(
    stream,
    args,
    metanetwork,
    metalora,
    tokenizer,
    cfg,
    device,
    rng,
    memory_position_offset: int,
) -> dict:
    recurrent_memory = None
    observed_turns = []
    supervised_turns = []
    memory_norms_by_turn = []
    for turn_index, current_turn in enumerate(stream.turns):
        observed_turns.append(current_turn)
        is_final_turn = turn_index + 1 == len(stream.turns)
        supervise_turn = args.turn_supervision == "every" or is_final_turn
        if supervise_turn:
            lora_dict, recurrent_memory = trainable_generate_context_lora(
                current_turn.text,
                metanetwork,
                tokenizer,
                metalora,
                cfg,
                device,
                use_gradient_checkpoint=args.use_gradient_checkpoint,
                recurrent_memory=recurrent_memory,
                return_recurrent_state=True,
                memory_position_offset=memory_position_offset,
            )
            lora_dict = clamp_lora_tensors(lora_dict, args.generated_lora_clamp)
            objective = resolve_turn_objective(args, rng)
            qa_source_turns = [current_turn] if args.qa_scope == "current" else observed_turns
            qa_rows = sample_turn_qa(qa_source_turns, args.qa_per_context, rng)
            if objective in {"qa", "both"} and not qa_rows:
                if args.turn_objective == "mixed":
                    objective = "reconstruction"
                else:
                    raise ValueError(
                        f"Stream {stream.stream_id!r} turn {current_turn.turn_id!r} selected "
                        f"{objective} supervision but no QA targets are available."
                    )
            reconstruction_text = (
                current_turn.text
                if args.reconstruction_scope == "current"
                else "\n".join(turn.text for turn in observed_turns)
            )
            training_records = build_training_records(
                accumulated_qa(qa_source_turns),
                qa_rows,
                args.use_contrastive,
                objective,
                reconstruction_text=reconstruction_text,
            )
            category_losses = compute_combined_lora_loss(
                training_records,
                lora_dict,
                metanetwork,
                tokenizer,
                device,
                args.answer_max_length,
                use_gradient_checkpoint=args.use_answer_gradient_checkpoint,
            )
            first_category_loss = next(iter(category_losses.values()))
            turn_loss = first_category_loss.new_zeros(())
            if "answer" in category_losses:
                turn_loss = turn_loss + category_losses["answer"]
            if "contrastive" in category_losses:
                turn_loss = turn_loss + args.contrastive_weight * category_losses["contrastive"]
            if "reconstruction" in category_losses:
                reconstruction_weight = 1.0 if objective == "reconstruction" else args.recon_weight
                turn_loss = turn_loss + reconstruction_weight * category_losses["reconstruction"]
            supervised_turns.append(
                {
                    "turn": turn_index + 1,
                    "turn_id": current_turn.turn_id,
                    "objective": objective,
                    "qa_rows": qa_rows if objective in {"qa", "both"} else [],
                    "category_losses": category_losses,
                    "loss": turn_loss,
                }
            )
        else:
            recurrent_memory = trainable_update_recurrent_memory(
                current_turn.text,
                metanetwork,
                tokenizer,
                metalora,
                cfg,
                device,
                use_gradient_checkpoint=args.use_gradient_checkpoint,
                recurrent_memory=recurrent_memory,
                memory_position_offset=memory_position_offset,
            )
        memory_norms_by_turn.append(
            {
                "turn": turn_index + 1,
                "turn_id": current_turn.turn_id,
                "norms": recurrent_memory_norm_stats(recurrent_memory),
            }
        )
        detach_every = args.detach_recurrent_memory_every
        if detach_every > 0 and (turn_index + 1) % detach_every == 0 and turn_index + 1 < len(stream.turns):
            recurrent_memory = recurrent_memory.detach()

    loss = torch.stack([turn["loss"] for turn in supervised_turns]).mean()

    def mean_category_loss(category: str):
        values = [turn["category_losses"][category] for turn in supervised_turns if category in turn["category_losses"]]
        return torch.stack(values).mean() if values else None

    answer_loss = mean_category_loss("answer")
    contrastive_loss = mean_category_loss("contrastive")
    recon_loss = mean_category_loss("reconstruction")
    lora_stats = lora_tensor_stats(lora_dict)
    memory_norms = memory_norms_by_turn[-1]["norms"]
    turn_loss_log = [
        {
            "turn": turn["turn"],
            "turn_id": turn["turn_id"],
            "objective": turn["objective"],
            "loss": float(turn["loss"].detach().cpu()),
            "answer_loss": float(turn["category_losses"]["answer"].detach().cpu()) if "answer" in turn["category_losses"] else None,
            "contrastive_loss": float(turn["category_losses"]["contrastive"].detach().cpu()) if "contrastive" in turn["category_losses"] else None,
            "reconstruction_loss": float(turn["category_losses"]["reconstruction"].detach().cpu()) if "reconstruction" in turn["category_losses"] else None,
            "qa_fact_ids": [row["id"] for row in turn["qa_rows"]],
        }
        for turn in supervised_turns
    ]
    log = {
        "loss": float(loss.detach().cpu()),
        "answer_loss": float(answer_loss.detach().cpu()) if answer_loss is not None else None,
        "contrastive_loss": float(contrastive_loss.detach().cpu()) if contrastive_loss is not None else None,
        "reconstruction_loss": float(recon_loss.detach().cpu()) if recon_loss is not None else None,
        "stream_id": stream.stream_id,
        "source_kind": stream.source_kind,
        "turn_ids": [turn.turn_id for turn in stream.turns],
        "fact_ids_by_turn": [list(turn.fact_ids) for turn in stream.turns],
        "turn_losses": turn_loss_log,
        "lora_stats": lora_stats,
        "memory_norms": memory_norms,
        "memory_norms_by_turn": memory_norms_by_turn,
    }
    return {
        "loss": loss,
        "finite": bool(torch.isfinite(loss).detach().cpu()) and lora_stats["finite"] >= 1.0,
        "log": log,
    }


def aggregate_batch_logs(stream_logs: list[dict]) -> dict:
    def mean_optional(key: str):
        values = [row[key] for row in stream_logs if row[key] is not None]
        return sum(values) / len(values) if values else None

    memory_keys = sorted({key for row in stream_logs for key in row["memory_norms"]})
    memory_norms = {
        key: sum(row["memory_norms"][key] for row in stream_logs if key in row["memory_norms"])
        / sum(1 for row in stream_logs if key in row["memory_norms"])
        for key in memory_keys
    }
    return {
        "loss": sum(row["loss"] for row in stream_logs) / len(stream_logs),
        "answer_loss": mean_optional("answer_loss"),
        "contrastive_loss": mean_optional("contrastive_loss"),
        "reconstruction_loss": mean_optional("reconstruction_loss"),
        "stream_ids": [row["stream_id"] for row in stream_logs],
        "source_kinds": [row["source_kind"] for row in stream_logs],
        "streams": stream_logs,
        "lora_stats": {
            "max_abs": max(row["lora_stats"]["max_abs"] for row in stream_logs),
            "mean_abs": sum(row["lora_stats"]["mean_abs"] for row in stream_logs) / len(stream_logs),
            "finite": min(row["lora_stats"]["finite"] for row in stream_logs),
        },
        "memory_norms": memory_norms,
    }


def run_training(args: argparse.Namespace, distributed: DistributedContext) -> None:
    if not 0.0 <= args.qa_turn_prob <= 1.0:
        raise ValueError("--qa-turn-prob must be between 0 and 1")
    if not 0.0 <= args.ordered_stream_probability <= 1.0:
        raise ValueError("--ordered-stream-probability must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be non-negative")
    evaluation_enabled = args.eval_every > 0 or args.eval_at_start or args.eval_only
    if args.eval_trials < 1 and evaluation_enabled:
        raise ValueError("--eval-trials must be at least 1 when evaluation is enabled")
    if args.eval_example_max_new_tokens < 0:
        raise ValueError("--eval-example-max-new-tokens must be non-negative")
    if args.eval_example_max_chars < 1:
        raise ValueError("--eval-example-max-chars must be at least 1")
    if args.distributed_bucket_mb < 1:
        raise ValueError("--distributed-bucket-mb must be at least 1")
    eval_objective = resolve_eval_objective(args)
    best_metric_name, maximize_best_metric = resolve_best_metric(args, eval_objective)
    if not args.train_file:
        args.train_file = default_train_file()
    train_data = load_recurrent_dataset(resolve_path(args.train_file))
    val_data = load_recurrent_dataset(resolve_path(args.val_file)) if evaluation_enabled else None
    output_dir = resolve_path(args.output_dir)
    log_path = output_dir / "shine_posttrain_train_log.jsonl"
    if distributed.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(distributed)
    rng = random.Random(args.seed + distributed.rank * 100_003)

    # All ranks must start from identical newly-created tensors (notably mem_tokens).
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    runtime_args, cfg, device, metanetwork, metalora, tokenizer, trainable = load_shine_for_training(args)
    # Use rank-specific dropout/random streams after identical initialization and checkpoint loading.
    torch.manual_seed(args.seed + distributed.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + distributed.rank)
    model_max_positions = getattr(getattr(metanetwork.metamodel, "config", None), "max_position_embeddings", None)
    if model_max_positions is not None and args.answer_max_length > int(model_max_positions):
        raise ValueError(
            f"--answer-max-length ({args.answer_max_length}) exceeds the backbone limit "
            f"max_position_embeddings={model_max_positions}. Use a shorter recurrent window or "
            "--reconstruction-scope current."
        )
    memory_position_offset = args.memory_position_offset or cfg.test.context_max_length
    if memory_position_offset < cfg.test.context_max_length:
        raise ValueError(
            f"memory_position_offset ({memory_position_offset}) must be >= padded context length "
            f"({cfg.test.context_max_length})"
        )
    if distributed.is_main:
        LOGGER.info(
            "Recurrent SHINE: steps=%s fixed_memory_position_offset=%s detach_every=%s",
            args.recurrent_steps,
            memory_position_offset,
            args.detach_recurrent_memory_every,
        )
        LOGGER.info(
            "Synchronous data parallel: world_size=%s batch_per_rank=%s global_batch=%s",
            distributed.world_size,
            args.batch_size,
            args.batch_size * distributed.world_size,
        )
        LOGGER.info(
            "Training hypernetwork=%s Metalora=%s",
            bool(trainable["metanetwork"]),
            bool(trainable["metalora"]),
        )
    optimizer = build_optimizer(trainable, args)
    parameters = trainable_parameters(trainable)
    best_val_metric = (-float("inf") if maximize_best_metric else float("inf")) if evaluation_enabled else None
    wandb_run = init_wandb(args, distributed)

    initial_val_summary = None
    if args.eval_at_start or args.eval_only:
        if distributed.is_main:
            eval_rng = random.Random(args.seed + 1_000_003)
            initial_val_summary = evaluate_current(
                metanetwork,
                metalora,
                tokenizer,
                cfg,
                val_data,
                args,
                runtime_args,
                device,
                eval_rng,
            )
            initial_metric = report_validation(0, initial_val_summary, best_metric_name, args)
            best_val_metric = initial_metric
            append_jsonl(log_path, {"step": 0, "val": initial_val_summary, "config": vars(args)})
            if args.eval_at_start and not args.eval_only:
                save_posttrain_checkpoint(
                    output_dir / "best",
                    metanetwork,
                    metalora,
                    extra_state={"step": 0, "val": initial_val_summary, "config": vars(args)},
                )
            if wandb_run is not None:
                initial_payload = validation_wandb_payload(
                    initial_val_summary,
                    best_metric_name,
                    initial_metric,
                )
                initial_payload["val/best_metric"] = best_val_metric
                wandb_run.log(initial_payload, step=0)
        distributed_barrier(distributed)

    training_steps = 0 if args.eval_only else args.max_steps
    train_progress = tqdm(
        range(1, training_steps + 1),
        desc="SHINE posttrain",
        dynamic_ncols=True,
        leave=True,
    ) if tqdm is not None and distributed.is_main else range(1, args.max_steps + 1)

    for step in train_progress:
        optimizer.zero_grad(set_to_none=True)
        stream_logs = []
        for micro_batch in range(args.batch_size):
            stream = sample_training_stream(
                train_data,
                recurrent_steps=args.recurrent_steps,
                fact_counts=args.fact_counts,
                context_format=args.context_format,
                ordered_stream_probability=args.ordered_stream_probability,
                window_policy=args.stream_window_policy,
                rng=rng,
            )
            local_error = None
            stream_result = None
            try:
                stream_result = compute_stream_training_result(
                    stream,
                    args,
                    metanetwork,
                    metalora,
                    tokenizer,
                    cfg,
                    device,
                    rng,
                    memory_position_offset,
                )
            except Exception as exc:  # Keep collective order aligned before propagating rank-local failures.
                local_error = exc
            if distributed_any(local_error is not None, device, distributed):
                if local_error is not None:
                    raise RuntimeError(
                        f"rank={distributed.rank} step={step} micro_batch={micro_batch} failed"
                    ) from local_error
                raise RuntimeError(
                    f"Another rank failed at step={step} micro_batch={micro_batch}; see the torchrun rank traceback."
                )

            local_nonfinite = not stream_result["finite"]
            if distributed_any(local_nonfinite, device, distributed):
                if distributed.is_main:
                    append_jsonl(
                        log_path,
                        {
                            "step": step,
                            "micro_batch": micro_batch,
                            "rank": distributed.rank,
                            "nonfinite": True,
                            "stream": stream_result["log"],
                        },
                    )
                raise FloatingPointError(
                    f"Non-finite loss or generated LoRA on at least one rank at step={step}, "
                    f"micro_batch={micro_batch}."
                )

            local_backward_error = None
            try:
                (stream_result["loss"] / args.batch_size).backward()
            except Exception as exc:  # Let healthy ranks leave the step instead of waiting in gradient all-reduce.
                local_backward_error = exc
            if distributed_any(local_backward_error is not None, device, distributed):
                if local_backward_error is not None:
                    raise RuntimeError(
                        f"rank={distributed.rank} backward failed at step={step}, micro_batch={micro_batch}"
                    ) from local_backward_error
                raise RuntimeError(
                    f"Another rank failed during backward at step={step}, micro_batch={micro_batch}."
                )
            stream_logs.append(stream_result["log"])
            del stream_result

        synchronize_gradients(
            parameters,
            device,
            distributed,
            bucket_size_mb=args.distributed_bucket_mb,
        )
        grad_norm = None
        if args.grad_clip_norm > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm).detach().cpu())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        batch_log = aggregate_batch_logs(stream_logs)
        log_record = {
            "step": step,
            "loss": distributed_mean(batch_log["loss"], device, distributed),
            "answer_loss": distributed_mean_optional(batch_log["answer_loss"], device, distributed),
            "contrastive_loss": distributed_mean_optional(batch_log["contrastive_loss"], device, distributed),
            "reconstruction_loss": distributed_mean_optional(batch_log["reconstruction_loss"], device, distributed),
            "grad_norm": grad_norm,
            "batch_size_per_rank": args.batch_size,
            "world_size": distributed.world_size,
            "global_batch_size": args.batch_size * distributed.world_size,
            "rank0_batch": batch_log,
        }
        if tqdm is not None and distributed.is_main:
            postfix = {
                "loss": f"{log_record['loss']:.4f}",
                "lora_max": f"{batch_log['lora_stats']['max_abs']:.2f}",
                "mem_v": f"{batch_log['memory_norms'].get('value_rms_mean', 0.0):.3f}",
            }
            if log_record["answer_loss"] is not None:
                postfix["answer"] = f"{log_record['answer_loss']:.4f}"
            if log_record["contrastive_loss"] is not None:
                postfix["choice"] = f"{log_record['contrastive_loss']:.4f}"
            if log_record["reconstruction_loss"] is not None:
                postfix["recon"] = f"{log_record['reconstruction_loss']:.4f}"
            train_progress.set_postfix(postfix)
        wandb_step_payload = {}
        if distributed.is_main and (step % args.log_every == 0 or step == 1):
            append_jsonl(log_path, log_record)
            LOGGER.info(
                "step=%s loss=%.6f answer=%s reconstruction=%s grad_norm=%s global_batch=%s",
                step,
                log_record["loss"],
                f"{log_record['answer_loss']:.6f}" if log_record["answer_loss"] is not None else "n/a",
                f"{log_record['reconstruction_loss']:.6f}" if log_record["reconstruction_loss"] is not None else "n/a",
                f"{grad_norm:.6f}" if grad_norm is not None else "n/a",
                log_record["global_batch_size"],
            )
            if wandb_run is not None:
                wandb_payload = {
                    "train/loss": log_record["loss"],
                    "train/lora_max_abs_rank0": batch_log["lora_stats"]["max_abs"],
                    "train/lora_mean_abs_rank0": batch_log["lora_stats"]["mean_abs"],
                    "system/world_size": distributed.world_size,
                    "system/batch_size_per_rank": args.batch_size,
                    "system/global_batch_size": log_record["global_batch_size"],
                }
                if grad_norm is not None:
                    wandb_payload["train/grad_norm"] = grad_norm
                for category in ("answer", "contrastive", "reconstruction"):
                    value = log_record[f"{category}_loss"]
                    if value is not None:
                        wandb_payload[f"train/{category}_loss"] = value
                for name, value in batch_log["memory_norms"].items():
                    wandb_payload[f"memory_rank0/{name}"] = value
                wandb_step_payload.update(wandb_payload)

        eval_due = args.eval_every > 0 and (step % args.eval_every == 0 or step == training_steps)
        if eval_due:
            if distributed.is_main:
                # Use the same samples at every checkpoint so the qualitative example is comparable.
                eval_rng = random.Random(args.seed + 1_000_003)
                val_summary = evaluate_current(
                    metanetwork,
                    metalora,
                    tokenizer,
                    cfg,
                    val_data,
                    args,
                    runtime_args,
                    device,
                    eval_rng,
                )
                log_record["val"] = val_summary
                append_jsonl(log_path, log_record)
                current_val_metric = report_validation(step, val_summary, best_metric_name, args)
                if tqdm is not None:
                    displayed_best = (
                        max(best_val_metric, current_val_metric)
                        if maximize_best_metric
                        else min(best_val_metric, current_val_metric)
                    )
                    train_progress.set_postfix(
                        {
                            "loss": f"{log_record['loss']:.4f}",
                            "val": f"{current_val_metric:.4f}",
                            "best": f"{displayed_best:.4f}",
                        }
                    )
                save_posttrain_checkpoint(
                    output_dir / "latest",
                    metanetwork,
                    metalora,
                    extra_state={"step": step, "val": val_summary, "config": vars(args)},
                )
                is_better = (
                    current_val_metric > best_val_metric
                    if maximize_best_metric
                    else current_val_metric < best_val_metric
                )
                if is_better:
                    best_val_metric = current_val_metric
                    save_posttrain_checkpoint(
                        output_dir / "best",
                        metanetwork,
                        metalora,
                        extra_state={"step": step, "val": val_summary, "config": vars(args)},
                    )
                if wandb_run is not None:
                    val_payload = validation_wandb_payload(
                        val_summary,
                        best_metric_name,
                        current_val_metric,
                    )
                    val_payload["val/best_metric"] = best_val_metric
                    wandb_step_payload.update(val_payload)
            distributed_barrier(distributed)
        else:
            save_due = step == training_steps or (args.save_every > 0 and step % args.save_every == 0)
            if save_due:
                if distributed.is_main:
                    save_posttrain_checkpoint(
                        output_dir / "latest",
                        metanetwork,
                        metalora,
                        extra_state={"step": step, "config": vars(args)},
                    )
                distributed_barrier(distributed)
        if distributed.is_main and wandb_run is not None and wandb_step_payload:
            wandb_run.log(wandb_step_payload, step=step)
        if args.empty_cache_every > 0 and step % args.empty_cache_every == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if distributed.is_main:
        final_summary = {
            "best_val_metric": best_val_metric,
            "best_metric_name": best_metric_name if evaluation_enabled else None,
            "initial_val": initial_val_summary,
            "max_steps": training_steps,
            "world_size": distributed.world_size,
            "batch_size_per_rank": args.batch_size,
            "global_batch_size": args.batch_size * distributed.world_size,
            "config": vars(args),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(final_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "best_val_metric": best_val_metric,
                    "best_metric_name": best_metric_name if evaluation_enabled else None,
                }
            )
            wandb_run.finish()
    distributed_barrier(distributed)
    del metanetwork, metalora, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    distributed = initialize_distributed(args)
    logging.basicConfig(
        level=logging.INFO if distributed.is_main else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        run_training(args, distributed)
    finally:
        if distributed.enabled and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
