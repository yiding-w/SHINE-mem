from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.distributed as dist


def tp_detach_enabled(cfg) -> bool:
    parallel_cfg = cfg.get("parallel", None)
    if parallel_cfg is None:
        return False
    mode = parallel_cfg.get("mode", "none") if hasattr(parallel_cfg, "get") else getattr(parallel_cfg, "mode", "none")
    return str(mode).lower() == "tp_detach"


def get_tp_detach_config(cfg) -> Dict:
    if not tp_detach_enabled(cfg):
        return {"enabled": False, "tp_rank": 0, "tp_world": 1, "tp_group": None}
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("parallel.mode=tp_detach requires torchrun/distributed initialization")

    parallel_cfg = cfg.parallel
    requested_tp = int(parallel_cfg.get("tensor_parallel_size", dist.get_world_size()))
    world_size = dist.get_world_size()
    if requested_tp != world_size:
        raise ValueError(
            "The first v1 tp_detach path only supports one TP group: "
            f"tensor_parallel_size={requested_tp}, WORLD_SIZE={world_size}. "
            "Use torchrun --nproc_per_node equal to tensor_parallel_size."
        )
    return {
        "enabled": True,
        "tp_rank": dist.get_rank(),
        "tp_world": world_size,
        "tp_group": dist.group.WORLD,
    }


def configure_model_for_tp_detach(model, *, tp_rank: int, tp_world: int, tp_group=None) -> None:
    """Mark v1 LoraLinear modules with the detach_state shard axis they consume."""
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            attn.q_proj.set_detach_state_tp("col", tp_rank, tp_world, tp_group)
            attn.k_proj.set_detach_state_tp("col", tp_rank, tp_world, tp_group)
            attn.v_proj.set_detach_state_tp("col", tp_rank, tp_world, tp_group)
            attn.o_proj.set_detach_state_tp("row", tp_rank, tp_world, tp_group)
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            mlp.gate_proj.set_detach_state_tp("col", tp_rank, tp_world, tp_group)
            mlp.up_proj.set_detach_state_tp("col", tp_rank, tp_world, tp_group)
            mlp.down_proj.set_detach_state_tp("row", tp_rank, tp_world, tp_group)


def broadcast_loradict_from_rank0(loradict: Optional[Dict], group=None) -> None:
    if loradict is None or not (dist.is_available() and dist.is_initialized()):
        return
    src = 0 if group is None else dist.get_global_rank(group, 0)

    def _walk(node):
        if torch.is_tensor(node):
            dist.broadcast(node.data, src=src, group=group)
            return
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)

    _walk(loradict)
