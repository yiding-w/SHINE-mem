"""Independent loader for the original SHINE pretrain checkpoint format."""

from __future__ import annotations

import ast
import gc
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from LoraQwen import LoraQwen3ForCausalLM, Qwen3Config
from metanetwork_family import Metanetwork
from utils.myfreeze import freeze
from utils.mysaveload import load_checkpoint
from utils.myseed import set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]


def _official_chat_template() -> str:
    source = REPO_ROOT / "test_pretrain.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "chat_template":
                if isinstance(node.value.value, str):
                    return node.value.value
    raise RuntimeError("Could not locate the official tokenizer.chat_template assignment")


def _build_cfg(raw):
    model_path = str(raw.model_path)
    return OmegaConf.create(
        {
            "run": {"seed": int(raw.seed), "device": "cuda", "use_gradient_checkpoint": False},
            "paths": {"model_path": model_path},
            "data": {
                "context_max_length": int(raw.context_max_length),
                "conversation_max_length": int(raw.conversation_max_length),
            },
            "model": {
                "lora_r": int(raw.lora_r),
                "metalora_r": int(raw.metalora_r),
                "ift_additional_metalora_r": -1,
                "num_mem_token": 4,
                "metamodel_class_path": "LoraQwen.LoraQwen3ForCausalLM",
                "config_class_path": "LoraQwen.Qwen3Config",
                "tokenizer_from": model_path,
                "model_from": model_path,
            },
            "metanetwork": {
                "type": "transformer",
                "method": "rl",
                "transformer_cfg": {
                    "encoder_cfg": {
                        "d_model": 4096,
                        "nhead": 32,
                        "dim_feedforward": 8192,
                        "dropout": 0,
                        "activation": "gelu",
                        "layer_norm_eps": 0.00001,
                        "batch_first": True,
                        "norm_first": False,
                        "bias": True,
                    },
                    "couple_encoder_cfg": {
                        "d_model": 4096,
                        "nhead": 32,
                        "dim_feedforward": 8192,
                        "dropout": 0,
                        "activation": "gelu",
                        "layer_norm_eps": 0.00001,
                        "batch_first": True,
                        "norm_first": False,
                        "bias": True,
                    },
                    "layer_transformer_first": True,
                    "mean_pool_size": 1,
                    "num_layers": int(raw.metanetwork_layers),
                    "couple_num_layers": 0,
                    "scale": 0.001,
                },
            },
            "test": {
                "context_max_length": int(raw.context_max_length),
                "conversation_max_length": int(raw.conversation_max_length),
            },
            "hidden_size": -1,
            "num_layers": -1,
            "num_mem_token": 4,
        }
    )


def _dtype(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _cast_tree(value, dtype):
    if torch.is_tensor(value):
        return value.to(dtype=dtype) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _cast_tree(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_cast_tree(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_cast_tree(item, dtype) for item in value)
    return value


def load_pretrain_runtime(config_path: str | Path, checkpoint_dir: str | Path, device: torch.device):
    raw = OmegaConf.load(config_path)
    cfg = _build_cfg(raw)
    set_seed(int(raw.seed))

    config = Qwen3Config.from_pretrained(cfg.model.model_from)
    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers

    # Preserve the official pretrain construction/RNG order.
    temporary = LoraQwen3ForCausalLM.from_pretrained(cfg.model.model_from, config=config)
    lora_params = temporary.lora_params_numel(cfg.model.lora_r)
    base_params = cfg.hidden_size * cfg.num_layers
    if lora_params % base_params:
        raise ValueError("LoRA parameter count is not divisible by hidden_size * num_layers")
    config.num_mem_token = lora_params // base_params
    cfg.num_mem_token = config.num_mem_token
    del temporary
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_from, padding_side="left", use_fast=True
    )
    tokenizer.add_tokens(["<RECON>", "<COMP>"])
    tokenizer.chat_template = _official_chat_template()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    metamodel = LoraQwen3ForCausalLM.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))
    metanetwork = Metanetwork(
        metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r)
    )
    metanetwork.to(device)
    freeze(metamodel)
    metanetwork, metalora, _ = load_checkpoint(
        metanetwork, str(checkpoint_dir), device
    )

    dtype = _dtype(str(raw.torch_dtype))
    metanetwork.to(device=device, dtype=dtype)
    metalora = _cast_tree(metalora, dtype)
    metanetwork.eval()
    return cfg, metanetwork, metalora, tokenizer

