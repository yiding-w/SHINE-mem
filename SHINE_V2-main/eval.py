#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone SHINE_V2 benchmark evaluation.

This entry point intentionally does not import or call the training loop.  It
builds the TP model, loads model weights, then dispatches to a benchmark
runner such as eval_squad_gen.run_squad_qa_gen.

Example:
    torchrun --nproc_per_node=1 --master_port=29541 eval.py \
      --config-name=main_pretrain_annealing \
      model=Qwen3_5-4B \
      m2p_transformer=full_prenorm_gatedlastnorm_4b \
      data=pretrain_annealing/memory_stream \
      detach_state=origin \
      parallel.mode=tp parallel.tensor_parallel_size=1 parallel.total_gpus=1 \
      +checkpoint=checkpoint/mem_qwen35_4b/pretrain_annealing/memstream_v1/final \
      +squad.path=data/squad +squad.split=validation +squad.num=200 \
      +squad.out=outputs/squad/memstream_v1_squad200.json
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict

from eval_squad_gen import run_squad_qa_gen
from hypernetwork.tp_model_hypernetwork import TPModelHypernetwork
from utils.mygpu import all_gpu_stats
from utils.myparallel import (
    barrier,
    cleanup_distributed,
    init_distributed,
    is_main_process_per_node,
    setup_tensor_parallel,
)


logger = logging.getLogger(__name__)


def _cfg_select(cfg: DictConfig, key: str, default: Any = None) -> Any:
    value = OmegaConf.select(cfg, key)
    return default if value is None else value


def _resolve_path(path: str | None) -> str | None:
    if path is None or path == "":
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(get_original_cwd(), path)


def _load_model_only(model: TPModelHypernetwork, checkpoint_dir: str) -> dict:
    model_dir = os.path.join(checkpoint_dir, "model")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Checkpoint model dir not found: {model_dir}. "
            "Pass a step_N or final checkpoint directory."
        )

    model.load_model(model_dir)

    metadata_path = os.path.join(checkpoint_dir, "training_state", "metadata.pt")
    if os.path.exists(metadata_path):
        return torch.load(metadata_path, map_location="cpu")
    return {}


def _set_env_from_cfg(cfg: DictConfig) -> None:
    benchmark = str(_cfg_select(cfg, "benchmark", os.environ.get("EVAL_BENCHMARK", "squad"))).lower()
    if benchmark != "squad":
        raise ValueError(f"Unsupported benchmark '{benchmark}'. Currently only 'squad' is implemented.")

    mapping = {
        "squad.path": "SQUAD_DATA_PATH",
        "squad.split": "SQUAD_SPLIT",
        "squad.num": "SQUAD_NUM",
        "squad.max_new": "SQUAD_MAX_NEW",
        "squad.out": "SQUAD_OUT",
    }
    for cfg_key, env_key in mapping.items():
        value = _cfg_select(cfg, cfg_key)
        if value is not None:
            if cfg_key in ("squad.path", "squad.out"):
                value = _resolve_path(str(value))
            os.environ[env_key] = str(value)

    qic = _cfg_select(cfg, "squad.query_include_context")
    if qic is not None:
        os.environ["SQUAD_QUERY_INCLUDE_CONTEXT"] = "1" if bool(qic) else "0"

    eval_mode = _cfg_select(cfg, "squad.eval_mode")
    if eval_mode is not None:
        os.environ["SQUAD_EVAL_MODE"] = str(eval_mode)

    plain = _cfg_select(cfg, "squad.plain_baseline")
    if plain is not None:
        os.environ["SQUAD_PLAIN_BASELINE"] = "1" if bool(plain) else "0"

    context = _cfg_select(cfg, "squad.context_baseline")
    if context is not None:
        os.environ["SQUAD_CONTEXT_BASELINE"] = "1" if bool(context) else "0"


@hydra.main(version_base=None, config_path="configs", config_name="main_pretrain_annealing")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    init_distributed()

    try:
        if is_main_process_per_node():
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
                force=True,
            )
            logger.info("Starting standalone SHINE_V2 eval")

        tensor_parallel_size = int(cfg.parallel.get("tensor_parallel_size", cfg.parallel.total_gpus))
        if tensor_parallel_size != 1:
            raise ValueError(
                "eval.py currently supports tensor_parallel_size=1 for free-form generation. "
                "Use parallel.tensor_parallel_size=1 parallel.total_gpus=1."
            )

        tp_cfg = setup_tensor_parallel(
            total_gpus=int(cfg.parallel.total_gpus),
            tensor_parallel_size=tensor_parallel_size,
        )
        my_device = tp_cfg["device"]

        if cfg.get("detach_state"):
            with open_dict(cfg.model):
                cfg.model.detach_state = cfg.detach_state

        model = TPModelHypernetwork(
            model_cfg=cfg.model,
            m2p_transformer_cfg=cfg.m2p_transformer,
            tp_rank=tp_cfg["tp_rank"],
            tp_world=tp_cfg["tensor_parallel_size"],
            tp_process_group=tp_cfg.get("tp_process_group"),
            dtype=torch.bfloat16,
            activation_checkpointing=cfg.training.get("tp_knobs", {}).get("activation_checkpointing", True),
            ckpt_skip_stride=cfg.training.get("tp_knobs", {}).get("ckpt_skip_stride", 0),
            compile_hypernetwork=cfg.training.get("tp_knobs", {}).get("compile_hypernetwork", True),
        )
        all_gpu_stats("After model load")

        model.init_detach_state(
            local_batch_size=int(cfg.training.tp_batchsize.batch_size),
            micro_batch_size=int(cfg.training.tp_batchsize.batch_size),
            tp_rank=tp_cfg["tp_rank"],
            tp_world=tp_cfg["tensor_parallel_size"],
            tp_process_group=tp_cfg.get("tp_process_group"),
            data_parallel_size=tp_cfg["data_parallel_size"],
            grad_accum_steps=1,
        )

        checkpoint = (
            _cfg_select(cfg, "checkpoint")
            or _cfg_select(cfg, "eval.checkpoint")
            or os.environ.get("EVAL_CHECKPOINT")
            or os.environ.get("CHECKPOINT")
        )
        checkpoint = _resolve_path(str(checkpoint)) if checkpoint else None
        if not checkpoint:
            raise ValueError(
                "No checkpoint specified. Use +checkpoint=/path/to/step_or_final "
                "or set EVAL_CHECKPOINT."
            )

        metadata = _load_model_only(model, checkpoint)
        barrier()
        if is_main_process_per_node():
            step = metadata.get("global_step", "?")
            logger.info("Loaded checkpoint: %s (step=%s)", checkpoint, step)

        _set_env_from_cfg(cfg)
        run_squad_qa_gen(model, cfg, tp_cfg, my_device)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
