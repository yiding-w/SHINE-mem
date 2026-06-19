"""Held-out eval launcher for SHINE-hypernet.

(Distinct from the legacy ``scripts/eval.py`` — which is SQuAD-only,
no vLLM, no mode comparison. This launcher is the new harness.)

Data-parallel: launched under ``torchrun --nproc_per_node=8``, each
rank owns its own GPU + co-located vLLM + SHINE; items are sharded
across ranks via ``items[rank::world_size]``; rank 0 gathers all
per-item records and writes the unified JSONL + summary.

Launched as plain ``python scripts/run_eval.py``, it runs single-rank
on the lone visible GPU.

Examples:

    # Default: 8 GPUs, full v1 suite
    torchrun --nproc_per_node=8 --standalone scripts/run_eval.py \\
        --datasets squad,boolq,bbh,arc_challenge,gsm8k,humaneval \\
        --shots 1,4,16

    # Single-GPU quick run
    python scripts/run_eval.py --datasets squad --limit 100

    # K-shot sweep on ARC-Challenge over 8 GPUs
    torchrun --nproc_per_node=8 --standalone scripts/run_eval.py \\
        --datasets arc_challenge --shots 1,4,16 --modes shine,icl
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _bootstrap_cuda_env() -> None:
    """Pin this rank to its local GPU before importing torch / vllm.

    Mirrors ``scripts/train.py``: torchrun sets ``LOCAL_RANK`` and we
    narrow ``CUDA_VISIBLE_DEVICES`` so each rank sees exactly one GPU
    as ``cuda:0``. Single-process launch (no torchrun) → LOCAL_RANK=0.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", required=True,
                   help="Comma-separated dataset names (see meta_past/eval/datasets/)")
    p.add_argument("--modes", default="",
                   help="Comma-separated modes. Empty = each dataset's defaults.")
    p.add_argument("--shots", default="8",
                   help="Comma-separated K-shot list for bucket-B datasets. "
                        "Ignored for bucket A/C.")
    p.add_argument("--limit", type=int, default=None,
                   help="Truncate each dataset to this many items (for smoke runs).")
    p.add_argument("--out_dir", default="runs/eval",
                   help="Per-mode JSONL + summary will be written here.")
    p.add_argument("--ckpt_dir",
                   default=os.path.expanduser(
                       "~/huggingfacemodels/SHINE-ift_mqa_1qa"),
                   help="SHINE hypernet checkpoint.")
    p.add_argument("--backbone",
                   default=os.path.expanduser(
                       "~/huggingfacemodels/Qwen3-8B"),
                   help="HF model path for the base LM + vLLM engine.")
    p.add_argument("--max_loras", type=int, default=8,
                   help="LoRA ring-buffer size for shine mode. Default 8 "
                        "matches training; pushing higher OOMs the hypernet "
                        "forward on 80GB.")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.40,
                   help="vLLM HBM share. Training uses 0.30; eval has no "
                        "AdamW state so we can run a touch higher.")
    p.add_argument("--enable_thinking", default="true",
                   help="true (default) / false / null. With stock Qwen3 "
                        "template: true/null = no stub, model decides; "
                        "false = inject empty <think></think> stub. Must "
                        "match the value used at train time.")
    p.add_argument("--force_think", default="false",
                   help="true / false. If true, append '<think>\\n' to every "
                        "prompt so completions start mid-thinking and must "
                        "close </think> before the answer. Must match train.")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--metalora_r", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _bootstrap_cuda_env()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = (rank == 0)

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"%(asctime)s rank{rank} %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("meta_past.scripts.run_eval")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import torch.distributed as dist

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # Init NCCL only when torchrun-launched (world_size > 1). For per-item
    # score gather we use ``all_gather_object`` which works over NCCL but
    # transfers via gloo; we install gloo as the all-gather backend by
    # using the default backend (NCCL) — ``all_gather_object`` will
    # internally pickle + use the CPU gloo channel if needed.
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        log.info("torch.distributed initialized: rank=%d/%d", rank, world_size)

    from meta_past.eval.harness import run_one_dataset, write_summary
    from meta_past.eval.runner import EvalRunner, EvalRunnerConfig
    from meta_past.rl.vllm_engine import VLLMEngine, VLLMEngineConfig
    from meta_past.shine_adapter import ShineHypernet

    et = args.enable_thinking.strip().lower()
    enable_thinking: bool | None = None if et in ("null", "none", "") \
        else (et in ("true", "1", "yes"))
    ft = args.force_think.strip().lower()
    force_think: bool = ft in ("true", "1", "yes")

    log.info("booting vLLM engine on backbone=%s (rank %d/%d)",
             args.backbone, rank, world_size)
    # max_model_len = 8192 — enough headroom for K=16 GSM8K CoT demos.
    # K=16 packed prompts hit 4099 tokens with the longest demos
    # available, just over 4096. Going to 8192 doubles the KV-cache
    # budget but gives 2x worst-case margin so we don't crack again on
    # outlier-length items. Training uses 2048 because its prompts are
    # short single questions; eval packs many demos.
    engine = VLLMEngine(VLLMEngineConfig(
        model_path=args.backbone,
        max_loras=args.max_loras,
        max_lora_rank=args.lora_r,
        max_model_len=8192,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        seed=rank,  # differ per rank to avoid identical sampling RNG
    ))
    engine.boot()
    log.info("vLLM ready (asleep).")

    log.info("constructing SHINE hypernet ckpt=%s", args.ckpt_dir)
    net = ShineHypernet(
        ckpt_dir=args.ckpt_dir,
        device="cuda:0",
        backbone=args.backbone,
        lora_r=args.lora_r,
        metalora_r=args.metalora_r,
    )

    runner = EvalRunner(
        hypernet=net,
        engine=engine,
        cfg=EvalRunnerConfig(
            max_loras=args.max_loras,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            enable_thinking=enable_thinking,
            force_think=force_think,
        ),
    )

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    modes = [s.strip() for s in args.modes.split(",") if s.strip()] or None
    shots_list = [int(s) for s in args.shots.split(",") if s.strip()]

    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    try:
        for ds in datasets:
            log.info("=== %s ===", ds)
            all_results.extend(run_one_dataset(
                name=ds,
                runner=runner,
                out_dir=out_dir,
                modes=modes,
                shots_list=shots_list,
                limit=args.limit,
            ))
        if is_main:
            write_summary(all_results, out_dir / "summary.jsonl")
            log.info("\n=== SUMMARY ===")
            for r in all_results:
                shots_tag = f" k={r.shots}" if r.shots is not None else ""
                log.info("%-20s %s%s  n=%4d  score=%.4f  (%.1fs local)",
                         r.dataset, r.mode, shots_tag,
                         r.n_items, r.score_mean, r.elapsed_s)
    finally:
        engine.shutdown()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
