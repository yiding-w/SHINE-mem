#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Utilities for Pipeline Parallel Training

This module provides utility functions for loading and partitioning models
in pipeline parallel training setups. It handles model distribution across
multiple GPUs and nodes with proper device mapping.

Key Features:
- Automatic model partitioning for pipeline parallelism
- Device-aware model loading
- Memory-optimized model distribution
- Support for various model architectures

Usage Examples:
-------------
1. Load model with pipeline parallelism:
   >>> from utils.mymodel import load_model_for_pipeline
   >>> model = load_model_for_pipeline(model_path, pipeline_config)

2. Partition model across devices:
   >>> from utils.mymodel import partition_model_for_pipeline
   >>> partitioned_model = partition_model_for_pipeline(model, device_mapping)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import logging
from transformers import AutoConfig, AutoModelForCausalLM
import os
from hydra.utils import get_original_cwd
from utils.myparallel import is_main_process, is_main_process_per_node, get_rank, get_pipeline_config
import transformers.utils.logging as hf_logging

logger = logging.getLogger(__name__)


def _format_single_model_lines(
    model_name: str,
    device_map: Dict[str, int],
    compact_map: Optional[Dict] = None,
) -> Dict[int, List[str]]:
    """
    Build per-GPU content lines for a single model.

    Returns:
        Dict[int, List[str]]: gpu_id → list of display strings (without box chars).
    """
    from collections import defaultdict
    import re

    layer_prefix: Optional[str] = None
    if compact_map is not None:
        layer_prefix = compact_map.get("layer_prefix", None)

    gpu_components: Dict[int, List[str]] = defaultdict(list)
    for comp, gpu in sorted(device_map.items(), key=lambda x: (x[1], x[0])):
        gpu_components[int(gpu)].append(comp)

    layer_pattern = re.compile(
        rf"^{re.escape(layer_prefix)}\.\d+$"
    ) if layer_prefix else None

    result: Dict[int, List[str]] = {}
    for gpu_id in sorted(gpu_components.keys()):
        comps = gpu_components[gpu_id]
        layer_indices: List[int] = []
        extras: List[str] = []
        for c in comps:
            if layer_pattern and layer_pattern.match(c):
                idx = int(c.rsplit(".", 1)[-1])
                layer_indices.append(idx)
            else:
                extras.append(c)
        layer_indices.sort()

        lines: List[str] = []
        # Layer ranges
        if layer_indices:
            ranges: List[str] = []
            start = layer_indices[0]
            prev = start
            for idx in layer_indices[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
                    start = idx
                    prev = idx
            ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
            prefix_short = layer_prefix if layer_prefix else "layer"
            lines.append(f"      [layers]  {prefix_short}.{{ {', '.join(ranges)} }}")
        # Extra components
        if extras:
            chunk_size = 3
            for i in range(0, len(extras), chunk_size):
                chunk = extras[i:i + chunk_size]
                if i == 0:
                    lines.append(f"      [extra ]  {', '.join(chunk)}")
                else:
                    lines.append(f"                {', '.join(chunk)}")
        if not lines:
            lines.append("      (no components)")
        result[gpu_id] = lines
    return result


def log_combined_device_map(
    models: List[tuple],
) -> None:
    """
    Log a combined, human-readable visualization of which layers of each
    model reside on each GPU.  The box width adapts to the longest line.

    Call this **after** all models have been loaded so that a single unified
    table is printed.

    Args:
        models: List of ``(model_label, device_map, compact_map)`` tuples.
            - model_label (str): e.g. ``"Qwen3_6-35B-A3B (LLM)"``
            - device_map (Dict[str, int]): full component→gpu mapping.
            - compact_map (Optional[Dict]): compact map with ``layer_prefix``.

    Example output::

        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║  Pipeline Device Map                                                        ║
        ╠══════════════════════════════════════════════════════════════════════════════╣
        ║  GPU 0                                                                      ║
        ║    ● Qwen3_6-35B-A3B (LLM)                                                 ║
        ║      [layers]  model.layers.{ 0-4 }                                         ║
        ║      [extra ]  model.embed_tokens, model.mem_tokens, model.rotary_emb       ║
        ║    ● m2p_transformer                                                        ║
        ║      (no components)                                                        ║
        ║──────────────────────────────────────────────────────────────────────────────║
        ║  GPU 1                                                                      ║
        ║    ● Qwen3_6-35B-A3B (LLM)                                                 ║
        ║      [layers]  model.layers.{ 5-10 }                                        ║
        ║    ● m2p_transformer                                                        ║
        ║      (no components)                                                        ║
        ║──────────────────────────────────────────────────────────────────────────────║
        ║  ...                                                                        ║
        ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    # Collect per-model per-GPU lines
    model_gpu_lines: List[tuple] = []  # [(label, {gpu_id: [lines]}), ...]
    for label, dmap, cmap in models:
        model_gpu_lines.append((label, _format_single_model_lines(label, dmap, cmap)))

    # Determine all GPU ids across all models
    all_gpus = sorted(set(
        gpu for _, gpu_lines in model_gpu_lines for gpu in gpu_lines
    ))

    # Build all content lines first, then compute width
    content_lines: List[str] = []  # raw text (no box chars yet)
    title = "  Pipeline Device Map"
    content_lines.append(title)
    content_lines.append(None)  # placeholder for ╠═══╣ separator

    for i, gpu_id in enumerate(all_gpus):
        content_lines.append(f"  GPU {gpu_id}")
        for label, gpu_lines in model_gpu_lines:
            content_lines.append(f"    ● {label}")
            lines_for_gpu = gpu_lines.get(gpu_id, ["      (no components)"])
            content_lines.extend(lines_for_gpu)
        # Add thin separator between GPUs (except after the last one)
        if i < len(all_gpus) - 1:
            content_lines.append(None)  # placeholder for ║───║ separator

    # Compute dynamic width: max content length + 2 (padding)
    max_len = max(
        len(line) for line in content_lines if line is not None
    )
    width = max(max_len + 2, 40)  # at least 40

    # Build the final box
    box_lines: List[str] = []
    box_lines.append("╔" + "═" * width + "╗")

    for line in content_lines:
        if line is None:
            # Check if this is the title separator or a GPU separator
            if len(box_lines) == 2:  # right after title
                box_lines.append("╠" + "═" * width + "╣")
            else:
                box_lines.append("║" + "─" * width + "║")
        else:
            box_lines.append("║" + line.ljust(width) + "║")

    box_lines.append("╚" + "═" * width + "╝")

    logger.info("\n" + "\n".join(box_lines))


def expand_device_map(compact_map: Dict) -> Dict[str, int]:
    """
    Expand a compact device map specification into a full layer-to-GPU mapping.
    
    The compact format uses layer ranges instead of listing every layer individually,
    making it much easier to configure in YAML.
    
    Compact format:
        {
            'layer_prefix': 'model.layers',       # prefix for decoder layers
            'gpu_map': [
                [0, 0, 11],    # [gpu_id, start_layer, end_layer] inclusive
                [1, 12, 23],
                [2, 24, 35],
                [3, 36, 47],
            ],
            'extra': {                             # non-layer components
                'model.embed_tokens': 0,
                'model.norm': 3,
                'lm_head': 3,
            }
        }
    
    Args:
        compact_map (Dict): Compact device map specification with keys:
            - layer_prefix (str): Prefix for numbered decoder layers
            - gpu_map (List[List[int]]): List of [gpu_id, start, end] triples
            - extra (Dict[str, int], optional): Mapping of non-layer components to GPUs
    
    Returns:
        Dict[str, int]: Full device map suitable for HuggingFace from_pretrained(device_map=...)
    
    Raises:
        ValueError: If required keys are missing or gpu_map entries are invalid
    """
    if 'layer_prefix' not in compact_map or 'gpu_map' not in compact_map:
        raise ValueError(
            "Compact device map must contain 'layer_prefix' and 'gpu_map' keys. "
            "Got keys: " + str(list(compact_map.keys()))
        )
    
    prefix = compact_map['layer_prefix']
    gpu_map = compact_map['gpu_map']
    extra = compact_map.get('extra', {})
    
    device_map = {}
    
    # Expand layer ranges
    for entry in gpu_map:
        if len(entry) != 3:
            raise ValueError(
                f"Each gpu_map entry must be [gpu_id, start_layer, end_layer], got {entry}"
            )
        gpu_id, start, end = int(entry[0]), int(entry[1]), int(entry[2])
        if start > end:
            raise ValueError(
                f"start_layer ({start}) must be <= end_layer ({end}) in gpu_map entry {entry}"
            )
        for layer_idx in range(start, end + 1):
            device_map[f"{prefix}.{layer_idx}"] = gpu_id
    
    # Add extra (non-layer) components
    for component_name, gpu_id in extra.items():
        device_map[component_name] = int(gpu_id)
    
    return device_map


def filter_device_map_for_stage(full_device_map: Dict[str, int], target_gpu: int, local_rank: int) -> Dict[str, int]:
    """
    Build a device map where layers belonging to this rank's GPU are placed on
    the local CUDA device, and all other layers are placed on the "meta" device.
    
    HuggingFace's from_pretrained requires a device_map that covers ALL model
    components. If a component is missing from the map, it falls back to the
    default device (usually the first GPU), which causes OOM when multiple
    ranks load simultaneously.
    
    Non-target layers are mapped to "meta" instead of "cpu" so that they only
    retain shape/dtype metadata without allocating any real memory. This avoids
    the massive CPU memory overhead that would occur if every rank kept ~86% of
    the model weights in host RAM.
    
    Args:
        full_device_map (Dict[str, int]): Complete device map mapping layer names to GPU IDs.
        target_gpu (int): The GPU ID in the full map that this rank is responsible for.
        local_rank (int): The local rank (GPU index) of the current process.
    
    Returns:
        Dict[str, int]: Device map covering all layers — target layers on local_rank,
                        others on "meta" (no memory allocation).
    """
    stage_map = {}
    for layer_name, gpu_id in full_device_map.items():
        if int(gpu_id) == target_gpu:
            stage_map[layer_name] = local_rank
        else:
            stage_map[layer_name] = "meta"
    return stage_map


def _import_class(dotted_path: str):
    """
    Dynamically import a class from a dotted module path.

    Example:
        >>> cls = _import_class("src_transformers_lora.LoraQwen3_5Moe.LoraQwen3_5MoeForCausalLM")
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_model_for_pipeline(model_path: str, pipeline_config: Dict, dtype: torch.dtype = torch.float32,
                           custom_device_map: Optional[Dict] = None,
                           model_class=None,
                           compile_mode: Optional[str] = None,
                           **model_kwargs) -> nn.Module:
    """
    Load model with pipeline parallel awareness.
    
    When a custom_device_map is provided, each rank only loads the layers
    assigned to its own GPU (pipeline stage). This avoids all ranks loading
    the full model onto all GPUs simultaneously, which would cause OOM.
    
    Args:
        model_path (str): Path to pre-trained model or model identifier
        pipeline_config (Dict): Pipeline configuration from setup_pipeline_parallel
        dtype (torch.dtype): Data type for model weights (default: torch.float32)
        custom_device_map (Optional[Dict]): Custom device map for layer-to-GPU placement.
        model_class: Model class to use for from_pretrained(). If None, falls back
            to ``AutoModelForCausalLM``.
            Accepts two formats:
            
            1. Compact format (recommended) — uses layer ranges:
                {
                    'layer_prefix': 'model.layers',
                    'gpu_map': [
                        [0, 0, 11],    # GPU 0: layers 0-11
                        [1, 12, 23],   # GPU 1: layers 12-23
                    ],
                    'extra': {'model.embed_tokens': 0, 'model.norm': 1, 'lm_head': 1}
                }
            
            2. Full format — explicit mapping of every layer name to GPU index:
                {'model.embed_tokens': 0, 'model.layers.0': 0, ...}
            
            If None, uses 'auto' with max_memory constraints.
        **model_kwargs: Additional keyword arguments for model loading
        
    Returns:
        nn.Module: Loaded model ready for pipeline parallel training
    """
    from utils.myparallel import get_local_rank
    
    # Get device for current pipeline stage
    device = pipeline_config['device']
    local_rank = get_local_rank()
    stage = pipeline_config['stage']
    
    # Configure model loading for pipeline parallelism
    if custom_device_map is not None:
        # Per-stage loading: each rank only loads layers assigned to its own GPU.
        # The full device_map maps layer names to GPU IDs (0..N-1).
        # We filter to keep only layers for this rank's GPU, then remap to local_rank.
        stage_device_map = filter_device_map_for_stage(
            full_device_map=custom_device_map,
            target_gpu=local_rank,  # local_rank == the GPU ID in the full map
            local_rank=local_rank,
        )
        
        gpu_layers = sum(1 for v in stage_device_map.values() if v != "meta")
        meta_layers = sum(1 for v in stage_device_map.values() if v == "meta")
        
        load_kwargs = {
            'device_map': stage_device_map,
            'dtype': dtype,
            'low_cpu_mem_usage': True,     # Load weights shard-by-shard to minimize peak memory
            'trust_remote_code': True,     # Allow custom model architectures like Qwen3 MoE
            **model_kwargs
        }
    else:
        # Use automatic device mapping with memory constraints — only for current GPU
        load_kwargs = {
            'device_map': {"":  device},
            'dtype': dtype,
            'low_cpu_mem_usage': True,     # Load weights shard-by-shard to minimize peak memory
            'trust_remote_code': True,     # Allow custom model architectures like Qwen3 MoE
            **model_kwargs
        }
    
    # Resolve relative path to absolute using Hydra's original working directory,
    # since Hydra changes CWD to the output directory at runtime.
    if not os.path.isabs(model_path):
        model_path = os.path.join(get_original_cwd(), model_path)
    
    try:
        # Disable tqdm progress bar on non-local-rank-0 processes to keep logs clean
        if not is_main_process_per_node():
            hf_logging.disable_progress_bar()
        
        # Liger-Kernel patches: must be called before from_pretrained so newly
        # constructed layers pick up the patched classes/forwards.
        # 1. RMSNorm: fuse 4 PyTorch ops into 1 Triton kernel (129 calls/step)
        # Set SHINE_LIGER=0 to disable for A/B comparison.
        if os.environ.get("SHINE_LIGER", "1") not in ("0", "", "false"):
            from utils.liger_patch import apply_liger_rmsnorm_patch
            apply_liger_rmsnorm_patch()

        # Load model with pipeline-aware configuration
        cls = model_class if model_class is not None else AutoModelForCausalLM
        if is_main_process_per_node():
            logger.info(f"[load_model_for_pipeline] Using model class: {cls.__name__}")

        # Suppress the harmless accelerate warning about model.mem_tokens not
        # matching any submodule.  mem_tokens is an nn.Parameter (not a
        # submodule), so accelerate's check_device_map() flags it, but the
        # parameter is still placed correctly.
        import warnings
        warnings.filterwarnings(
            "ignore",
            message=r".*device_map keys do not match any submodules.*",
            category=UserWarning,
            module=r"accelerate\.utils\.modeling",
        )
        model = cls.from_pretrained(model_path, **load_kwargs)
        warnings.filterwarnings(
            "default",
            message=r".*device_map keys do not match any submodules.*",
            category=UserWarning,
            module=r"accelerate\.utils\.modeling",
        )

        # Remove accelerate hooks installed by from_pretrained(device_map=...).
        # In our pipeline setup, tensor device placement is handled manually
        # via pipeline_send/pipeline_recv, so the hooks are redundant.
        # Removing them also eliminates torch.compile recompilation issues
        # caused by per-module type_id guards in the hook wrapper.
        from accelerate.hooks import remove_hook_from_module
        for module in model.modules():
            remove_hook_from_module(module)

        # Apply torch.compile to individual decoder layers on this stage.
        # This avoids the recompilation issues of compiling the entire
        # pipeline_forward_train (which has dynamic batch_id, layer_idx, etc.)
        # while still getting kernel fusion benefits within each layer.
        #
        # Increase dynamo cache/recompile limits to accommodate grad_mode
        # changes between training (torch.enable_grad) and evaluation
        # (torch.no_grad). Without this, switching between train/eval
        # triggers FailOnRecompileLimitHit for fullgraph=True layers.
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 64
        # torch 2.5 doesn't have recompile_limit; torch 2.6+ does. Older
        # versions used accumulated_cache_size_limit for the same purpose.
        if hasattr(torch._dynamo.config, "recompile_limit"):
            torch._dynamo.config.recompile_limit = 64
        if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
            torch._dynamo.config.accumulated_cache_size_limit = 1024

        if custom_device_map is not None:
            layer_prefix = None
            # Detect the layer container (e.g. model.model.layers or model.layers)
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layer_container = model.model.layers
                layer_prefix = "model.model.layers"
            elif hasattr(model, "layers"):
                layer_container = model.layers
                layer_prefix = "model.layers"
            else:
                layer_container = None

            if layer_container is not None:
                compiled_count = 0
                for idx, layer in enumerate(layer_container):
                    # Only compile layers that are actually on this device
                    # (non-meta layers have real parameters)
                    params = list(layer.parameters())
                    if params and params[0].device.type != "meta":
                        compile_kwargs = {"dynamic": False}
                        if compile_mode is not None:
                            compile_kwargs["mode"] = compile_mode
                        # Note: we do NOT use fullgraph=True even for full_attention
                        # layers, because Liger Triton kernels (RMSNorm) are
                        # marked @torch.compiler.disable and require graph breaks.
                        layer_container[idx] = torch.compile(
                            layer, **compile_kwargs
                        )
                        compiled_count += 1
                if is_main_process_per_node() and compiled_count > 0:
                    logger.info(
                        f"[load_model_for_pipeline] torch.compile applied to "
                        f"{compiled_count} decoder layers (dynamic=False, mode={compile_mode!r})"
                    )

        # Re-enable progress bar on non-local-rank-0 processes (in case other code needs it)
        if not is_main_process_per_node():
            hf_logging.enable_progress_bar()
        
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def partition_model_for_pipeline(model: nn.Module, device_mapping: List[torch.device]) -> nn.Module:
    """
    Partition model across multiple devices for pipeline parallelism.
    
    This function manually distributes model layers across different devices
    based on the provided device mapping. Useful for custom pipeline setups.
    
    Args:
        model (nn.Module): The model to partition
        device_mapping (List[torch.device]): List of devices for each pipeline stage
        
    Returns:
        nn.Module: Partitioned model
    """
    
    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'h'):
        if is_main_process_per_node():
            logger.warning("Model doesn't have standard transformer architecture, skipping manual partitioning")
        return model
    
    # Distribute transformer layers across devices
    layers = model.transformer.h
    layers_per_device = len(layers) // len(device_mapping)
    
    for i, layer in enumerate(layers):
        device_idx = i // layers_per_device
        if device_idx < len(device_mapping):
            layer.to(device_mapping[device_idx])
    
    # Move other components appropriately
    if hasattr(model.transformer, 'wte'):
        model.transformer.wte.to(device_mapping[0])  # Embedding on first device
    
    if hasattr(model.transformer, 'ln_f'):
        model.transformer.ln_f.to(device_mapping[-1])  # Final norm on last device
    
    if hasattr(model, 'lm_head'):
        model.lm_head.to(device_mapping[-1])  # Head on last device
    
    if is_main_process_per_node():
        logger.info(f"Model partitioned across {len(device_mapping)} devices")
    return model


def get_model_device_map(model: nn.Module) -> Dict:
    """
    Get current device mapping of model parameters.
    
    Useful for debugging and verifying model distribution across devices.
    
    Args:
        model (nn.Module): The model to analyze
        
    Returns:
        Dict: Device mapping information
    """
    device_map = {}
    
    for name, param in model.named_parameters():
        device_idx = param.device.index if param.device.type == 'cuda' else 'cpu'
        if device_idx not in device_map:
            device_map[device_idx] = []
        device_map[device_idx].append(name)
    
    # Count parameters per device
    device_stats = {}
    for device, params in device_map.items():
        device_stats[device] = len(params)
    
    return {
        'device_map': device_map,
        'device_stats': device_stats,
        'total_parameters': sum(len(params) for params in device_map.values())
    }


def build_layer_stage_mapping(device_map_cfg) -> Dict[int, int]:
    """
    Build a mapping from decoder layer index to pipeline stage (GPU id).

    This is derived from the compact device_map config (the same one used for
    model loading).  It tells the pipeline-parallel forward which layers
    belong to which stage so that send/recv can be inserted at stage
    boundaries.

    Args:
        device_map_cfg: Compact device map config (dict / DictConfig) with
            ``layer_prefix``, ``gpu_map``, and ``extra`` keys.

    Returns:
        Dict[int, int]: Mapping of layer_index → stage (gpu_id).
            e.g. {0: 0, 1: 0, ..., 5: 1, 6: 1, ...}
    """
    from omegaconf import OmegaConf

    if hasattr(device_map_cfg, "_metadata"):
        raw = OmegaConf.to_container(device_map_cfg, resolve=True)
    elif isinstance(device_map_cfg, dict):
        raw = device_map_cfg
    else:
        raw = dict(device_map_cfg)

    layer_to_stage: Dict[int, int] = {}
    for entry in raw["gpu_map"]:
        gpu_id, start, end = int(entry[0]), int(entry[1]), int(entry[2])
        for idx in range(start, end + 1):
            layer_to_stage[idx] = gpu_id
    return layer_to_stage


def get_extra_component_stages(device_map_cfg) -> Dict[str, int]:
    """
    Return the stage assignment for non-layer components (embed_tokens, norm,
    lm_head) from the compact device map config.

    Returns:
        Dict[str, int]: e.g. {"model.embed_tokens": 0, "model.norm": 7, "lm_head": 7}
    """
    from omegaconf import OmegaConf

    if hasattr(device_map_cfg, "_metadata"):
        raw = OmegaConf.to_container(device_map_cfg, resolve=True)
    elif isinstance(device_map_cfg, dict):
        raw = device_map_cfg
    else:
        raw = dict(device_map_cfg)

    return {k: int(v) for k, v in raw.get("extra", {}).items()}


def load_pretrained_llm_for_pipeline(
    model_cfg,
    dtype: torch.dtype = torch.float32,
    freeze: bool = True,
    use_lora_class: bool = True,
    compile_mode: Optional[str] = None,
    **model_kwargs,
) -> nn.Module:
    """
    High-level convenience function: load a pretrained LLM with pipeline parallelism.

    This encapsulates the full flow:
      1. Retrieve the pipeline config (must be set up beforehand via setup_pipeline_parallel).
      2. Extract and expand the compact device map from ``model_cfg.device_map``
         (if present).
      3. Load the model via load_model_for_pipeline (per-stage loading).
      4. Optionally freeze all parameters (default: True for a pretrained backbone).

    The model config is expected to contain an optional ``device_map`` key::

        name: Qwen3_6-35B-A3B
        path: "./models/Qwen3_6-35B-A3B"
        base_class: ...
        lora_class: ...
        device_map:
            layer_prefix: "model.layers"
            gpu_map:
                - [0, 0, 4]
                - [1, 5, 9]
                ...
            extra:
                model.embed_tokens: 0
                model.norm: 7
                lm_head: 7

    Prerequisites:
        ``init_distributed()`` and ``setup_pipeline_parallel()`` must have been
        called before invoking this function.  If not, a RuntimeError is raised.

    Args:
        model_cfg: Hydra DictConfig (or dict) with at least a ``path`` key pointing
            to the pretrained model directory.  May also contain ``name``,
            ``base_class``, ``lora_class`` (dotted import paths), and
            ``device_map`` (compact device map).
        dtype (torch.dtype): Weight dtype (default: torch.float32).
        freeze (bool): If True, set ``requires_grad=False`` on all parameters.
        use_lora_class (bool): If True, use ``model_cfg.lora_class`` instead of
            ``model_cfg.base_class`` (or AutoModelForCausalLM).  Default True.
        **model_kwargs: Extra kwargs forwarded to the model class's ``from_pretrained``.

    Returns:
        nn.Module: The loaded (and optionally frozen) model, placed on the correct
        pipeline-stage device.

    Raises:
        RuntimeError: If pipeline parallelism has not been initialised.
    """
    from omegaconf import OmegaConf

    # ------------------------------------------------------------------
    # 1. Pipeline config — must already be initialised
    # ------------------------------------------------------------------
    pipeline_config = get_pipeline_config()

    if pipeline_config["total_stages"] <= 1:
        raise RuntimeError(
            "Pipeline parallelism has not been initialised. "
            "Call init_distributed() and setup_pipeline_parallel() before "
            "load_pretrained_llm_for_pipeline()."
        )

    # ------------------------------------------------------------------
    # 2. Extract and expand compact device map from model_cfg
    # ------------------------------------------------------------------
    # Get device_map from model_cfg (supports both OmegaConf and plain dict)
    if hasattr(model_cfg, "get"):
        device_map_raw = model_cfg.get("device_map", None)
    elif isinstance(model_cfg, dict):
        device_map_raw = model_cfg.get("device_map", None)
    else:
        device_map_raw = getattr(model_cfg, "device_map", None)

    custom_device_map = None
    if device_map_raw is not None:
        # Convert OmegaConf → plain dict if necessary
        if hasattr(device_map_raw, "_metadata"):  # OmegaConf object
            raw_map = OmegaConf.to_container(device_map_raw, resolve=True)
        elif isinstance(device_map_raw, dict):
            raw_map = device_map_raw
        else:
            raw_map = dict(device_map_raw)

        if "layer_prefix" in raw_map and "gpu_map" in raw_map:
            custom_device_map = expand_device_map(raw_map)
        else:
            # Already a full device map
            custom_device_map = raw_map

        # If the model config has mem_tokens (num_mem_token > 0), ensure
        # "model.mem_tokens" is in the device map on the same GPU as
        # "model.embed_tokens".  Although mem_tokens is an nn.Parameter
        # (not a submodule) — which causes a harmless accelerate warning
        # about unmatched submodule keys — accelerate still requires it
        # in the device_map to know where to place the parameter.  Without
        # this entry, from_pretrained() will fail with:
        #   "does not give any device for the following parameters: model.mem_tokens"
        model_config = model_kwargs.get("config", None)
        num_mem = getattr(model_config, "num_mem_token", -1) if model_config is not None else -1
        if num_mem > 0 and "model.mem_tokens" not in custom_device_map:
            embed_gpu = custom_device_map.get("model.embed_tokens", 0)
            custom_device_map["model.mem_tokens"] = embed_gpu
            if is_main_process_per_node():
                logger.info(
                    f"[load_pretrained_llm] Added model.mem_tokens to device map "
                    f"on GPU {embed_gpu} (same as model.embed_tokens)"
                )

        # Ensure model.rotary_emb is explicitly in the device map.
        # rotary_emb is a tiny buffer-only module (inv_freq) that every
        # pipeline stage needs for computing position embeddings.  If it's
        # not in the device map, HF may place it on an arbitrary device.
        # We put it on the same GPU as embed_tokens (stage 0) by default;
        # the hypernetwork forward will handle it from there.
        if "model.rotary_emb" not in custom_device_map:
            embed_gpu = custom_device_map.get("model.embed_tokens", 0)
            custom_device_map["model.rotary_emb"] = embed_gpu
            if is_main_process_per_node():
                logger.info(
                    f"[load_pretrained_llm] Added model.rotary_emb to device map "
                    f"on GPU {embed_gpu} (same as model.embed_tokens)"
                )

    # ------------------------------------------------------------------
    # 3. Resolve model path and model class from config
    # ------------------------------------------------------------------
    model_path = str(model_cfg.path)

    # Determine which model class to use for from_pretrained().
    # Priority: lora_class (if use_lora_class) > base_class > AutoModelForCausalLM
    model_class = None
    class_attr = "lora_class" if use_lora_class else "base_class"
    class_path = getattr(model_cfg, class_attr, None)
    if class_path is None and use_lora_class:
        # Fallback: try base_class if lora_class is not specified
        class_path = getattr(model_cfg, "base_class", None)
    if class_path is not None:
        model_class = _import_class(str(class_path))
        if is_main_process_per_node():
            logger.info(
                f"[load_pretrained_llm] Resolved model class from "
                f"config.{class_attr} = {class_path}"
            )
    else:
        if is_main_process_per_node():
            logger.info(
                "[load_pretrained_llm] No model class specified in config, "
                "falling back to AutoModelForCausalLM"
            )

    # ------------------------------------------------------------------
    # 4. Load model with pipeline-aware device placement
    # ------------------------------------------------------------------
    model = load_model_for_pipeline(
        model_path=model_path,
        pipeline_config=pipeline_config,
        dtype=dtype,
        custom_device_map=custom_device_map,
        model_class=model_class,
        compile_mode=compile_mode,
        **model_kwargs,
    )

    # ------------------------------------------------------------------
    # 5. Freeze parameters if requested
    # ------------------------------------------------------------------
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        if is_main_process_per_node():
            logger.info("[load_pretrained_llm] All LLM parameters frozen")

    return model


def load_m2p_transformer_for_pipeline(
    m2p_transformer_cfg,
    dtype: torch.dtype = torch.float32,
    freeze: bool = False,
    compile_mode: Optional[str] = None,
) -> nn.Module:
    """
    High-level convenience function: build a TransformerModel (m2p_transformer)
    and distribute it across pipeline stages, mirroring the approach used by
    ``load_pretrained_llm_for_pipeline`` for pretrained LLMs.

    Memory-efficient strategy:
      1. Construct the full model skeleton on the **meta** device — this
         allocates zero real memory (only records shapes and dtypes).
      2. Parse the compact device map to determine which decoder layers belong
         to this pipeline stage.
      3. **Re-create** owned layers directly on the target CUDA device with
         proper weight initialisation.  Non-owned layers are replaced with
         empty ``nn.Module`` placeholders (no memory).
      4. Similarly materialise shared components (``norm``, ``rotary_emb``)
         only on the stage that owns them.

    This avoids the previous approach of building the entire model on CPU
    first, which required peak CPU memory equal to the full model size.

    The config is expected to have two top-level keys::

        init:                              # TransformerModel constructor kwargs
            hidden_size: 2048
            num_hidden_layers: 8
            ...

        device_map:                        # compact device map
            layer_prefix: 'layers'
            gpu_map:
                - [0, 0, 0]
                - [1, 1, 1]
                ...
            extra:
                norm: 7
                rotary_emb: 0

    Prerequisites:
        ``init_distributed()`` and ``setup_pipeline_parallel()`` must have been
        called before invoking this function.

    Args:
        m2p_transformer_cfg: Hydra DictConfig (or dict) with ``init`` and
            optional ``device_map`` keys.  ``init`` contains TransformerModel
            constructor kwargs; ``device_map`` contains the compact device map.
        dtype (torch.dtype): Weight dtype (default: torch.float32).
        freeze (bool): If True, set ``requires_grad=False`` on all parameters.

    Returns:
        nn.Module: The TransformerModel with layers distributed across pipeline
        stages.

    Raises:
        RuntimeError: If pipeline parallelism has not been initialised.
    """
    from omegaconf import OmegaConf
    from utils.myparallel import get_local_rank
    from hypernetwork.m2p_transformer import (
        TransformerModel, DecoderLayer, RotaryEmbedding, _make_norm,
        RMSNormGated, Experts, TopKRouter,
    )

    # ------------------------------------------------------------------
    # 1. Pipeline config — must already be initialised
    # ------------------------------------------------------------------
    pipeline_config = get_pipeline_config()

    if pipeline_config["total_stages"] <= 1:
        raise RuntimeError(
            "Pipeline parallelism has not been initialised. "
            "Call init_distributed() and setup_pipeline_parallel() before "
            "load_m2p_transformer_for_pipeline()."
        )

    local_rank = get_local_rank()
    stage = pipeline_config["stage"]
    device = pipeline_config["device"]

    # ------------------------------------------------------------------
    # 2. Convert config to plain dict and extract init / device_map
    # ------------------------------------------------------------------
    if hasattr(m2p_transformer_cfg, "_metadata"):
        full_cfg = OmegaConf.to_container(m2p_transformer_cfg, resolve=True)
    elif isinstance(m2p_transformer_cfg, dict):
        full_cfg = dict(m2p_transformer_cfg)
    else:
        full_cfg = dict(m2p_transformer_cfg)

    model_kwargs = full_cfg.get("init", full_cfg)
    device_map_cfg = full_cfg.get("device_map", None)

    # ------------------------------------------------------------------
    # 3. Build the full model skeleton on meta device (zero memory)
    # ------------------------------------------------------------------
    with torch.device("meta"):
        model = TransformerModel(**model_kwargs)

    # ------------------------------------------------------------------
    # 4. Distribute layers across pipeline stages
    # ------------------------------------------------------------------
    # Extract common layer-construction kwargs from the model config
    num_hidden_layers = model_kwargs.get("num_hidden_layers", 40)
    hidden_size = model_kwargs.get("hidden_size", 2048)
    num_attention_heads = model_kwargs.get("num_attention_heads", 16)
    num_key_value_heads = model_kwargs.get("num_key_value_heads", 2)
    head_dim = model_kwargs.get("head_dim", 256)
    attention_bias = model_kwargs.get("attention_bias", False)
    attention_dropout = model_kwargs.get("attention_dropout", 0.0)
    rms_norm_eps = model_kwargs.get("rms_norm_eps", 1e-6)
    use_gated_attention = model_kwargs.get("use_gated_attention", True)
    norm_zero_init = model_kwargs.get("norm_zero_init", True)
    prenorm = model_kwargs.get("prenorm", True)
    hidden_act = model_kwargs.get("hidden_act", "silu")
    intermediate_size = model_kwargs.get("intermediate_size", 11008)
    moe_intermediate_size = model_kwargs.get("moe_intermediate_size", 512)
    num_experts = model_kwargs.get("num_experts", 256)
    num_experts_per_tok = model_kwargs.get("num_experts_per_tok", 8)
    norm_topk_prob = model_kwargs.get("norm_topk_prob", True)
    initializer_range = model_kwargs.get("initializer_range", 0.02)
    last_norm = model_kwargs.get("last_norm", "normal")
    rope_theta = model_kwargs.get("rope_theta", 10000000.0)
    max_position_embeddings = model_kwargs.get("max_position_embeddings", 262144)

    # Per-layer MoE flags — parse the same way as TransformerModel.__init__
    layer_is_moe = model_kwargs.get("layer_is_moe", None)
    if layer_is_moe is None:
        layer_is_moe = [True] * num_hidden_layers
    elif isinstance(layer_is_moe, str):
        if layer_is_moe == "moe":
            layer_is_moe = [True] * num_hidden_layers
        elif layer_is_moe == "full":
            layer_is_moe = [False] * num_hidden_layers
        else:
            raise ValueError(f"layer_is_moe must be 'moe', 'full', or a list, got '{layer_is_moe}'")
    elif isinstance(layer_is_moe, list):
        parsed = []
        for i, v in enumerate(layer_is_moe):
            if v == "moe" or v is True:
                parsed.append(True)
            elif v == "full" or v is False:
                parsed.append(False)
            else:
                raise ValueError(f"layer_is_moe[{i}] must be 'moe' or 'full', got '{v}'")
        layer_is_moe = parsed

    effective_head_dim = head_dim or (hidden_size // num_attention_heads)

    def _init_weights(module: nn.Module) -> None:
        """Apply weight initialisation matching TransformerModel._init_weights."""
        std = initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Experts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)
        elif isinstance(module, TopKRouter):
            module.weight.data.normal_(mean=0.0, std=std)

    def _create_layer_on_device(layer_idx: int, target_device: torch.device) -> DecoderLayer:
        """Create a single DecoderLayer directly on the target device with proper init."""
        with torch.device(target_device):
            layer = DecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                layer_idx=layer_idx,
                head_dim=head_dim,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                rms_norm_eps=rms_norm_eps,
                use_gated_attention=use_gated_attention,
                norm_zero_init=norm_zero_init,
                prenorm=prenorm,
                use_moe=layer_is_moe[layer_idx],
                intermediate_size=intermediate_size,
                moe_intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                norm_topk_prob=norm_topk_prob,
                hidden_act=hidden_act,
            )
        layer = layer.to(dtype)
        # Apply weight initialisation (same logic as TransformerModel._init_weights)
        layer.apply(_init_weights)
        return layer

    if device_map_cfg is not None:
        # Convert OmegaConf → plain dict if necessary
        if hasattr(device_map_cfg, "_metadata"):
            raw_map = OmegaConf.to_container(device_map_cfg, resolve=True)
        elif isinstance(device_map_cfg, dict):
            raw_map = device_map_cfg
        else:
            raw_map = dict(device_map_cfg)

        # Build layer_index → gpu_id mapping
        layer_to_gpu: Dict[int, int] = {}
        for entry in raw_map["gpu_map"]:
            gpu_id, start, end = int(entry[0]), int(entry[1]), int(entry[2])
            for idx in range(start, end + 1):
                layer_to_gpu[idx] = gpu_id

        extra = {k: int(v) for k, v in raw_map.get("extra", {}).items()}

        # Determine which layers this stage owns
        my_layer_indices = sorted(
            [idx for idx, gpu in layer_to_gpu.items() if gpu == local_rank]
        )

        # Materialise owned layers on GPU; replace others with empty placeholders
        for i in range(num_hidden_layers):
            if i in my_layer_indices:
                model.layers[i] = _create_layer_on_device(i, device)
            else:
                # Empty placeholder — no parameters, no memory
                model.layers[i] = nn.Module()

        # Materialise extra components only on the owning stage
        norm_gpu = extra.get("norm", max(layer_to_gpu.values()))
        if norm_gpu == local_rank:
            if last_norm == "gated":
                with torch.device(device):
                    model.norm = RMSNormGated(hidden_size, rms_norm_eps)
                    model.norm_gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                model.norm = model.norm.to(dtype)
                model.norm_gate_proj = model.norm_gate_proj.to(dtype)
                model.norm_gate_proj.apply(_init_weights)
            elif last_norm == "normal":
                with torch.device(device):
                    model.norm = _make_norm(hidden_size, rms_norm_eps, norm_zero_init)
                model.norm = model.norm.to(dtype)
            else:  # "none"
                model.norm = nn.Identity()
        else:
            model.norm = nn.Module()  # placeholder
            if last_norm == "gated":
                model.norm_gate_proj = nn.Module()  # placeholder

        rotary_gpu = extra.get("rotary_emb", min(layer_to_gpu.values()))
        if rotary_gpu == local_rank:
            with torch.device(device):
                model.rotary_emb = RotaryEmbedding(
                    head_dim=effective_head_dim,
                    max_position_embeddings=max_position_embeddings,
                    rope_theta=rope_theta,
                )
        else:
            model.rotary_emb = nn.Module()  # placeholder

        gpu_layers = len(my_layer_indices)
        total_layers = len(layer_to_gpu)
    else:
        # No device map — build the entire model directly on this stage's device
        with torch.device(device):
            model = TransformerModel(**model_kwargs)
        model = model.to(dtype)
        if is_main_process_per_node():
            logger.info(
                f"[load_m2p_transformer] No device map; entire model on {device}"
            )

    # ------------------------------------------------------------------
    # 5. Freeze parameters if requested
    # ------------------------------------------------------------------
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        if is_main_process_per_node():
            logger.info("[load_m2p_transformer] All parameters frozen")

    # ------------------------------------------------------------------
    # Apply torch.compile to owned m2p decoder layers
    # ------------------------------------------------------------------
    compiled_count = 0
    compile_kwargs = {"dynamic": False, "fullgraph": True}
    if compile_mode is not None:
        compile_kwargs["mode"] = compile_mode
    for i in range(len(model.layers)):
        params = list(model.layers[i].parameters())
        if params and params[0].device.type != "meta":
            model.layers[i] = torch.compile(model.layers[i], **compile_kwargs)
            compiled_count += 1
    if is_main_process_per_node() and compiled_count > 0:
        logger.info(
            f"[load_m2p_transformer] torch.compile applied to "
            f"{compiled_count} m2p decoder layers (dynamic=False, fullgraph=True, mode={compile_mode!r})"
        )

    return model


if __name__ == "__main__":
    # Test the module
    print("Model Utilities for Pipeline Parallel Training")
    print("Available functions:")
    print("- load_model_for_pipeline(model_path, pipeline_config, **kwargs)")
    print("- partition_model_for_pipeline(model, device_mapping)")
    print("- get_model_device_map(model)")
    print("\nSee docstrings for detailed usage information.")