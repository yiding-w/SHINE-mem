"""
Checkpoint save / load utilities for ModelHypernetwork training.

Provides:
    - Checkpoint directory management (list, rotate, resolve latest)
    - save_checkpoint / load_checkpoint for full training resume
    - resolve_forever_save_steps for parsing config values

Checkpoint layout::

    checkpoint/{name}/pretrain/step_{N}/                          — pretrain checkpoints
    checkpoint/{name}/pretrain/final/                              — pretrain final checkpoint
    checkpoint/{name}/pretrain_annealing/{annealing_name}/step_{N}/ — pretrain_annealing checkpoints
    checkpoint/{name}/pretrain_annealing/{annealing_name}/final/    — pretrain_annealing final checkpoint
    checkpoint/{name}/sft/{annealing_name}/{sft_name}/step_{N}/     — SFT checkpoints
    checkpoint/{name}/sft/{annealing_name}/{sft_name}/final/        — SFT final checkpoint
        model/                             — pipeline-stage-independent
            model_stage{S}.safetensors     — all model tensors per stage (fast mmap)
        training_state/                    — all files are pipeline-stage-independent
            optimizer_tensors_stage{S}.safetensors — optimizer tensor state (fast mmap)
            optimizer_meta_stage{S}.pt     — optimizer non-tensor metadata
            scheduler.pt                   — LR scheduler state
            metadata.pt                    — training metadata (step, epoch, ...)
"""

import os
import logging
import shutil
import time
import torch
from typing import Optional, List, Dict, Any

from utils.myparallel import is_node0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def get_checkpoint_dir(run_name: str) -> str:
    """Get the checkpoint base directory for a given run name.

    The run_name encodes the full path structure:
      - pretrain: "{name}/pretrain"
      - pretrain_annealing: "{name}/pretrain_annealing/{annealing_name}"
      - sft: "{name}/sft/{annealing_name}/{sft_name}"
    """
    from hydra.utils import get_original_cwd
    return os.path.join(get_original_cwd(), "checkpoint", run_name)


def build_checkpoint_run_name(exp_name: str, mode: str, annealing_name: str = "", sft_name: str = "") -> str:
    """Build the run_name path component for checkpoint directory.

    Args:
        exp_name: Main experiment name (pretrain name).
        mode: Training mode ('pretrain', 'pretrain_annealing', or 'sft').
        annealing_name: Annealing sub-experiment name (required for pretrain_annealing and sft modes).
        sft_name: SFT sub-experiment name (required for sft mode).

    Returns:
        Path component like:
          - 'name/pretrain'
          - 'name/pretrain_annealing/annealing_name'
          - 'name/sft/annealing_name/sft_name'
    """
    if mode == "sft":
        if not annealing_name:
            raise ValueError("annealing_name is required for SFT mode checkpoint path.")
        if not sft_name:
            raise ValueError("sft_name is required for SFT mode checkpoint path.")
        return os.path.join(exp_name, "sft", annealing_name, sft_name)
    elif mode == "pretrain_annealing":
        if not annealing_name:
            raise ValueError("annealing_name is required for pretrain_annealing mode checkpoint path.")
        return os.path.join(exp_name, "pretrain_annealing", annealing_name)
    else:
        return os.path.join(exp_name, "pretrain")


def get_pretrain_final_checkpoint(exp_name: str) -> Optional[str]:
    """Get the path to the pretrain final checkpoint for a given experiment name.

    Returns the path if it exists, None otherwise.
    """
    from hydra.utils import get_original_cwd
    final_dir = os.path.join(get_original_cwd(), "checkpoint", exp_name, "pretrain", "final")
    if os.path.exists(final_dir):
        return final_dir
    return None


def get_pretrain_annealing_final_checkpoint(exp_name: str, annealing_name: str) -> Optional[str]:
    """Get the path to the pretrain_annealing final checkpoint for a given experiment and annealing name.

    Returns the path if it exists, None otherwise.
    """
    from hydra.utils import get_original_cwd
    final_dir = os.path.join(get_original_cwd(), "checkpoint", exp_name, "pretrain_annealing", annealing_name, "final")
    if os.path.exists(final_dir):
        return final_dir
    return None


def get_step_checkpoint_dir(run_name: str, step: int) -> str:
    """Get the checkpoint directory for a specific step."""
    return os.path.join(get_checkpoint_dir(run_name), f"step_{step}")


def list_checkpoints(run_name: str) -> List[int]:
    """List all checkpoint steps for a run, sorted ascending."""
    ckpt_base = get_checkpoint_dir(run_name)
    if not os.path.exists(ckpt_base):
        return []
    steps = []
    for d in os.listdir(ckpt_base):
        if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_base, d)):
            try:
                steps.append(int(d[5:]))
            except ValueError:
                continue
    return sorted(steps)


def get_latest_checkpoint(run_name: str) -> Optional[str]:
    """Get the path to the latest checkpoint, or None if no checkpoints exist.

    Note: This only considers step_N checkpoints, not the 'final' checkpoint.
    """
    steps = list_checkpoints(run_name)
    if not steps:
        return None
    return get_step_checkpoint_dir(run_name, steps[-1])


# ---------------------------------------------------------------------------
# Config parsing helpers
# ---------------------------------------------------------------------------

def resolve_forever_save_steps(raw_value) -> set:
    """Parse forever_save_steps config value into a set of step numbers.

    Supports the same formats as DebugSchedule:
      - ``-1``          → empty set (disabled)
      - positive int N  → fires every N steps (returns special sentinel)
      - list of ints    → set of those specific steps

    For interval mode, we return a special object that supports ``in``
    checks via modular arithmetic.
    """
    from omegaconf import ListConfig
    if raw_value is None:
        return set()
    if isinstance(raw_value, ListConfig):
        raw_value = list(raw_value)
    if isinstance(raw_value, list):
        return set(int(v) for v in raw_value)
    if isinstance(raw_value, int):
        if raw_value == -1:
            return set()
        if raw_value > 0:
            # Interval mode: return a special set-like object
            return _IntervalSet(raw_value)
        return set()
    return set()


class _IntervalSet:
    """A set-like object that contains every multiple of ``interval``.

    Supports the ``in`` operator so that ``step in _IntervalSet(N)``
    returns True when ``step > 0 and step % N == 0``.
    """

    def __init__(self, interval: int):
        self._interval = interval

    def __contains__(self, step) -> bool:
        if not isinstance(step, int) or step <= 0:
            return False
        return step % self._interval == 0

    def __repr__(self) -> str:
        return f"_IntervalSet(every {self._interval} steps)"


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,  # ModelHypernetwork
    optimizer,
    lr_scheduler,
    global_step: int,
    epoch: int,
    micro_step: int,
    run_name: str,
    forever_save_steps,
    save_total_limit: int,
    running_loss: float = 0.0,
    epoch_loss_sum: float = 0.0,
    epoch_steps: int = 0,
    ema_time_per_step: float = 0.0,
    total_context_tokens: int = 0,
    total_conv_total_tokens: int = 0,
    total_conv_valid_tokens: int = 0,
    wandb_run_id: Optional[str] = None,
    t_start: float = 0.0,
    max_steps: int = 0,
    train_loader=None,
    config_selections: Optional[dict] = None,
    launch_cmd: Optional[str] = None,
) -> float:
    """
    Save a checkpoint at the given step.

    **Only node 0** writes checkpoint files.  Since DP replicas have
    identical trainable parameters (synced via allreduce), saving from
    a single node is sufficient.  All nodes can read the checkpoint on
    resume because it lives on shared NFS.

    Creates two subdirectories:
        step_dir/model/          — model trainable parameters (stage-independent)
        step_dir/training_state/ — optimizer, scheduler, training state for resume

    Node 0's pipeline stages each save their own portion (per-layer files).

    Args:
        model: The ModelHypernetwork model.
        optimizer: The optimizer.
        lr_scheduler: The learning rate scheduler.
        global_step: Current optimizer step.
        epoch: Current epoch.
        micro_step: Current micro-step within grad accumulation window.
        run_name: Experiment name (used as checkpoint subdirectory).
        forever_save_steps: Set (or _IntervalSet) of steps that should never be deleted.
        save_total_limit: Max non-forever checkpoints to keep.
        running_loss: Current running loss accumulator.
        epoch_loss_sum: Epoch loss sum accumulator.
        epoch_steps: Epoch steps counter.
        ema_time_per_step: EMA time per step.
        total_context_tokens: Total context tokens processed.
        total_conv_total_tokens: Total conversation tokens processed.
        total_conv_valid_tokens: Total valid conversation tokens processed.
        wandb_run_id: Wandb run ID for resume.
        t_start: Training start time (for elapsed calculation in log).
        max_steps: Total training steps (for ETA calculation in log).
        train_loader: PipelineDataLoader whose state (generator) is saved
            for perfect resume reproducibility.  Optional.

    Returns:
        Wall-clock seconds spent saving (0.0 on non-node-0 processes).
    """
    # Only node 0 saves — DP replicas have identical parameters
    if not is_node0():
        return 0.0

    save_t0 = time.time()

    step_dir = get_step_checkpoint_dir(run_name, global_step)
    model_dir = os.path.join(step_dir, "model")
    training_state_dir = os.path.join(step_dir, "training_state")

    # Save model (per-layer files — pipeline-stage-independent)
    model.save_model(model_dir)

    # Save training state (optimizer, scheduler, metadata)
    os.makedirs(training_state_dir, exist_ok=True)
    stage = model._my_stage

    # Save optimizer state keyed by globally-unique param names
    # (pipeline-stage-independent)
    _save_optimizer_state(model, optimizer, training_state_dir)

    # Save scheduler state (stage-independent; only stage 0 writes to
    # avoid duplicate writes, but the content is identical across stages)
    if stage == 0:
        torch.save(lr_scheduler.state_dict(),
                   os.path.join(training_state_dir, "scheduler.pt"))

    # Save training metadata (only from stage 0 to avoid duplicates)
    if stage == 0:
        # Compute elapsed training time up to this checkpoint
        elapsed_time = time.time() - t_start if t_start > 0 else 0.0
        metadata = {
            "global_step": global_step,
            "epoch": epoch,
            "micro_step": micro_step,
            "running_loss": running_loss,
            "epoch_loss_sum": epoch_loss_sum,
            "epoch_steps": epoch_steps,
            "ema_time_per_step": ema_time_per_step,
            "total_context_tokens": total_context_tokens,
            "total_conv_total_tokens": total_conv_total_tokens,
            "total_conv_valid_tokens": total_conv_valid_tokens,
            "wandb_run_id": wandb_run_id,
            "elapsed_time": elapsed_time,
            "config_selections": config_selections,
            "launch_cmd": launch_cmd,
            "prev_repo_per_mb": getattr(model, '_prev_repo_per_mb', None),
        }
        # Save dataloader state for perfect resume reproducibility
        if train_loader is not None and hasattr(train_loader, "state_dict"):
            metadata["dataloader_state"] = train_loader.state_dict()
        torch.save(metadata, os.path.join(training_state_dir, "metadata.pt"))

    # Save detach_state (per-node, per-stage — wdict differs across DP replicas)
    # This runs on node 0 only (same as the rest of save_checkpoint), but
    # for multi-node setups, save_detach_state_all_nodes() should be called
    # separately from the training loop on ALL nodes.
    _save_detach_state(model, step_dir)

    save_duration = time.time() - save_t0

    # Manage checkpoint rotation (delete oldest non-forever checkpoints)
    # Only stage 0 on node 0 manages deletion — single writer avoids races
    if stage == 0 and global_step not in forever_save_steps:
        _rotate_checkpoints(run_name, forever_save_steps, save_total_limit)

    return save_duration


def _rotate_checkpoints(run_name: str, forever_save_steps, save_total_limit: int):
    """
    Delete oldest non-forever checkpoints if we exceed save_total_limit.

    Only called by node 0, stage 0 — a single process, so no race conditions.

    Args:
        run_name: Experiment name.
        forever_save_steps: Set (or _IntervalSet) of steps that should never be deleted.
        save_total_limit: Max non-forever checkpoints to keep.
    """
    all_steps = list_checkpoints(run_name)
    # Filter to non-forever checkpoints
    non_forever_steps = [s for s in all_steps if s not in forever_save_steps]

    # Delete oldest if over limit
    while len(non_forever_steps) > save_total_limit:
        oldest_step = non_forever_steps.pop(0)
        oldest_dir = get_step_checkpoint_dir(run_name, oldest_step)
        try:
            shutil.rmtree(oldest_dir)
            logger.info(f"  [Checkpoint] Deleted old checkpoint: step_{oldest_step}")
        except FileNotFoundError:
            # Directory may have been removed externally (e.g. manual cleanup)
            pass
        except OSError as e:
            logger.warning(f"  [Checkpoint] Failed to delete step_{oldest_step}: {e}")


# ---------------------------------------------------------------------------
# DetachState save / load helpers (per-node, per-stage)
#
# Unlike model parameters (which are identical across DP replicas and only
# saved on node 0), the detach_state wdict differs across DP replicas because
# each replica processes different data. Therefore detach_state must be saved
# by EVERY node independently.
#
# For single-node training (the common case), this is handled inside
# save_checkpoint (which only runs on node 0). For multi-node training,
# the training loop should call save_detach_state_all_nodes() on ALL nodes.
# ---------------------------------------------------------------------------

def _save_detach_state(model, step_dir: str) -> None:
    """Save detach_state for the current node's stages.

    Creates step_dir/detach_state/dp{dp_rank}_stage{stage}.pt for each
    stage on this node that has a non-empty detach_state.

    Args:
        model: The ModelHypernetwork model.
        step_dir: Path to the step checkpoint directory.
    """
    if model.detach_state is None:
        return
    ds_state = model.detach_state.state_dict()
    if not ds_state:
        return  # EmptyDetachState returns {}

    from utils.myparallel import get_pipeline_config
    pp_cfg = get_pipeline_config()
    dp_rank = pp_cfg["data_parallel_rank"]
    stage = pp_cfg["stage"]

    detach_state_dir = os.path.join(step_dir, "detach_state")
    os.makedirs(detach_state_dir, exist_ok=True)
    save_path = os.path.join(detach_state_dir, f"dp{dp_rank}_stage{stage}.pt")
    torch.save(ds_state, save_path)


def save_detach_state_all_nodes(model, run_name: str, global_step: int) -> None:
    """Save detach_state on ALL nodes (called from training loop on every rank).

    For multi-node DP training, each node has a different wdict and must
    save its own copy. This function should be called on ALL ranks after
    save_checkpoint() (which only saves model/optimizer on node 0).

    Args:
        model: The ModelHypernetwork model.
        run_name: Experiment name (used to locate checkpoint directory).
        global_step: Current optimizer step.
    """
    if model.detach_state is None:
        return
    ds_state = model.detach_state.state_dict()
    if not ds_state:
        return

    from utils.myparallel import get_pipeline_config
    pp_cfg = get_pipeline_config()
    dp_rank = pp_cfg["data_parallel_rank"]
    stage = pp_cfg["stage"]

    step_dir = get_step_checkpoint_dir(run_name, global_step)
    detach_state_dir = os.path.join(step_dir, "detach_state")
    os.makedirs(detach_state_dir, exist_ok=True)
    save_path = os.path.join(detach_state_dir, f"dp{dp_rank}_stage{stage}.pt")
    torch.save(ds_state, save_path)


def _load_detach_state(model, checkpoint_dir: str, my_device: torch.device) -> None:
    """Load detach_state for the current node's stage.

    Looks for checkpoint_dir/detach_state/dp{dp_rank}_stage{stage}.pt.
    If found, loads and validates the state. If not found, logs a warning
    and leaves detach_state in its initial (zero/None) state.

    Args:
        model: The ModelHypernetwork model.
        checkpoint_dir: Path to the step checkpoint directory.
        my_device: Current GPU device for tensor mapping.
    """
    if model.detach_state is None:
        return

    from utils.myparallel import get_pipeline_config
    pp_cfg = get_pipeline_config()
    dp_rank = pp_cfg["data_parallel_rank"]
    stage = pp_cfg["stage"]

    ds_path = os.path.join(
        checkpoint_dir, "detach_state", f"dp{dp_rank}_stage{stage}.pt"
    )
    if os.path.exists(ds_path):
        ds_state = torch.load(ds_path, map_location=my_device)
        model.detach_state.load_state_dict(ds_state)
        logger.info(f"  [DetachState] Loaded from {ds_path}")
    else:
        # No detach_state checkpoint — this is normal for fresh starts
        # or when resuming from a checkpoint that didn't use detach_state.
        logger.info(
            f"  [DetachState] No checkpoint found at {ds_path}, starting fresh."
        )


# ---------------------------------------------------------------------------
# Optimizer state helpers (stage-independent)
# ---------------------------------------------------------------------------

def _build_param_key_map(model, optimizer) -> Dict[int, str]:
    """Build a mapping from optimizer flat-param-index → globally unique key.

    The key is the parameter's globally unique name:
      - Hypernetwork params: their ``named_parameters()`` name
        (e.g. ``m2p_transformer.layers.0.self_attn.q_proj.weight``)
      - Metalora tensors: ``metalora_layer{L}_tensor{T}``
        where L is the LLM layer index and T is the tensor's position
        within that layer's loradict.
      - Mem tokens: ``mem_tokens``

    This mapping is deterministic given the model structure and is
    independent of the pipeline stage assignment.
    """
    from utils.myloradict import collect_loradict_tensors

    my_device = model._my_device

    # Collect all param ids in the optimizer (flat list, preserving order)
    flat_params = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            flat_params.append(p)

    # Build id → flat_index
    id_to_flat_idx: Dict[int, int] = {}
    for idx, p in enumerate(flat_params):
        id_to_flat_idx[id(p)] = idx

    # Map flat_index → key
    idx_to_key: Dict[int, str] = {}

    # Hypernetwork named parameters
    for name, param in model.hypernetwork.named_parameters():
        pid = id(param)
        if pid in id_to_flat_idx:
            idx_to_key[id_to_flat_idx[pid]] = f"hypernet.{name}"

    # Metalora tensors
    if hasattr(model, 'metalora') and model.metalora is not None:
        for layer_idx, layer_lora in model.metalora.items():
            tensors = collect_loradict_tensors(layer_lora)
            for t_idx, t in enumerate(tensors):
                pid = id(t)
                if pid in id_to_flat_idx:
                    idx_to_key[id_to_flat_idx[pid]] = (
                        f"metalora_layer{layer_idx}_tensor{t_idx}"
                    )

    # Mem tokens
    if hasattr(model, '_llm_model') and hasattr(model._llm_model, 'mem_tokens'):
        mem = model._llm_model.mem_tokens
        if mem is not None:
            pid = id(mem)
            if pid in id_to_flat_idx:
                idx_to_key[id_to_flat_idx[pid]] = "mem_tokens"

    return idx_to_key


def _save_optimizer_state(model, optimizer, training_state_dir: str):
    """Save optimizer state in a pipeline-stage-independent format.

    Uses safetensors for tensor data (fast mmap loading) and a separate
    .pt file for non-tensor metadata (step counts, etc.).

    Each stage saves its own portion of the optimizer state, keyed by
    globally unique parameter names.

    Files:
        ``optimizer_tensors_stage{S}.safetensors`` — exp_avg, exp_avg_sq tensors
        ``optimizer_meta_stage{S}.pt``             — step counts and param_groups
    """
    from safetensors.torch import save_file

    idx_to_key = _build_param_key_map(model, optimizer)
    opt_sd = optimizer.state_dict()

    # Separate tensors from non-tensor state
    tensors_dict: Dict[str, torch.Tensor] = {}
    meta_dict: Dict[str, Any] = {}

    for str_idx, state in opt_sd["state"].items():
        idx = int(str_idx) if isinstance(str_idx, str) else str_idx
        key = idx_to_key.get(idx)
        if key is not None:
            param_meta = {}
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    # Flatten key: "hypernet.layers.0.weight" + "exp_avg"
                    # → "hypernet.layers.0.weight__exp_avg"
                    tensors_dict[f"{key}__{k}"] = v.cpu()
                else:
                    param_meta[k] = v
            if param_meta:
                meta_dict[key] = param_meta

    # Save param_groups metadata (lr, betas, etc.) without param indices.
    groups_meta = []
    for g in opt_sd["param_groups"]:
        meta = {k: v for k, v in g.items() if k != "params"}
        groups_meta.append(meta)

    stage = model._my_stage

    # Save tensors as safetensors (single large file, fast mmap load)
    if tensors_dict:
        save_file(tensors_dict, os.path.join(
            training_state_dir, f"optimizer_tensors_stage{stage}.safetensors"))

    # Save non-tensor metadata (step counts, param_groups) as .pt
    payload = {
        "non_tensor_state": meta_dict,
        "param_groups_meta": groups_meta,
    }
    torch.save(payload, os.path.join(
        training_state_dir, f"optimizer_meta_stage{stage}.pt"))


def _load_optimizer_state(
    model, optimizer, training_state_dir: str, my_device: torch.device
):
    """Load optimizer state from stage-independent safetensors files.

    Only loads tensors belonging to the current stage's parameters —
    other stages' optimizer state is skipped entirely via ``safe_open``
    selective reading.

    Format: ``optimizer_tensors_stage*.safetensors`` + ``optimizer_meta_stage*.pt``

    Merges all stage files, then maps the saved named states back to the
    current optimizer's flat param indices.
    """
    import glob
    from safetensors import safe_open

    st_files = sorted(glob.glob(
        os.path.join(training_state_dir, "optimizer_tensors_stage*.safetensors")))
    meta_files = sorted(glob.glob(
        os.path.join(training_state_dir, "optimizer_meta_stage*.pt")))

    if not st_files:
        raise FileNotFoundError(
            f"[_load_optimizer_state] STRICT LOAD FAILED — "
            f"No optimizer_tensors_stage*.safetensors files found in {training_state_dir}. "
            f"Cannot resume training without optimizer state."
        )

    # Build the set of param keys this stage owns
    idx_to_key = _build_param_key_map(model, optimizer)
    key_to_idx: Dict[str, int] = {v: k for k, v in idx_to_key.items()}
    my_param_keys: set = set(key_to_idx.keys())

    # Build the set of tensor keys we need (param_key__state_name)
    # We need all keys whose prefix (before last "__") is in my_param_keys
    device_str = str(my_device)
    needed_tensors: Dict[str, torch.Tensor] = {}
    for fi, f in enumerate(st_files):
        matched_count = 0
        with safe_open(f, framework="pt", device=device_str) as sf:
            for flat_key in sf.keys():
                sep_idx = flat_key.rfind("__")
                if sep_idx == -1:
                    continue
                param_key = flat_key[:sep_idx]
                if param_key in my_param_keys:
                    needed_tensors[flat_key] = sf.get_tensor(flat_key)
                    matched_count += 1
    # Load non-tensor metadata (small files, load all and filter)
    non_tensor_state: Dict[str, Dict[str, Any]] = {}
    groups_meta = None
    for f in meta_files:
        payload = torch.load(f, map_location="cpu")
        for k, v in payload.get("non_tensor_state", {}).items():
            if k in my_param_keys:
                non_tensor_state[k] = v
        if groups_meta is None:
            groups_meta = payload.get("param_groups_meta")

    # Reconstruct named_state from tensors + metadata
    merged_named_state: Dict[str, Dict[str, Any]] = {}
    for flat_key, tensor in needed_tensors.items():
        sep_idx = flat_key.rfind("__")
        param_key = flat_key[:sep_idx]
        state_name = flat_key[sep_idx + 2:]
        if param_key not in merged_named_state:
            merged_named_state[param_key] = {}
        merged_named_state[param_key][state_name] = tensor

    # Merge non-tensor state (e.g. step counts)
    for param_key, meta in non_tensor_state.items():
        if param_key not in merged_named_state:
            merged_named_state[param_key] = {}
        merged_named_state[param_key].update(meta)

    # Reconstruct optimizer state_dict in the standard format
    current_sd = optimizer.state_dict()
    new_state: Dict[int, Any] = {}
    for key, state in merged_named_state.items():
        idx = key_to_idx.get(key)
        if idx is not None:
            new_state[idx] = state

    # Build the full state_dict to load
    load_sd = {
        "state": new_state,
        "param_groups": current_sd["param_groups"],  # keep current structure
    }

    # Restore param_groups hyperparameters (lr, betas, etc.) from saved
    if groups_meta is not None and len(groups_meta) == len(load_sd["param_groups"]):
        for saved_meta, group in zip(groups_meta, load_sd["param_groups"]):
            for k, v in saved_meta.items():
                if k in group:
                    group[k] = v

    # --- Strict check: fail if any parameter is missing optimizer state ---
    missing_keys = my_param_keys - set(merged_named_state.keys())
    if missing_keys:
        raise RuntimeError(
            f"[_load_optimizer_state] STRICT LOAD FAILED — "
            f"{len(missing_keys)}/{len(my_param_keys)} parameters have NO optimizer state "
            f"in checkpoint at {training_state_dir}.\n"
            f"  First 5 missing: {sorted(missing_keys)[:5]}\n"
            f"  Scanned files: {[os.path.basename(f) for f in st_files]}\n"
            f"  This means the checkpoint is incomplete or incompatible. "
            f"All parameters must have saved optimizer state for resume."
        )

    optimizer.load_state_dict(load_sd)


def _load_scheduler_state(lr_scheduler, training_state_dir: str):
    """Load scheduler state (stage-independent)."""
    sched_path = os.path.join(training_state_dir, "scheduler.pt")
    if os.path.exists(sched_path):
        sched_state = torch.load(sched_path, map_location="cpu")
        lr_scheduler.load_state_dict(sched_state)
    else:
        logger.warning("No scheduler checkpoint found at %s", sched_path)


# ---------------------------------------------------------------------------
# Checkpoint load
# ---------------------------------------------------------------------------

def load_checkpoint(
    model,  # ModelHypernetwork
    optimizer,
    lr_scheduler,
    checkpoint_dir: str,
    my_device: torch.device,
) -> Dict[str, Any]:
    """
    Load a checkpoint for resume training.

    Restores model parameters, optimizer state, scheduler state, and
    training metadata.  All components are **pipeline-stage-independent**:
    a checkpoint saved with one pipeline configuration can be loaded with
    a different one.

    Args:
        model: The ModelHypernetwork model.
        optimizer: The optimizer (state will be loaded).
        lr_scheduler: The learning rate scheduler (state will be loaded).
        checkpoint_dir: Path to the step checkpoint directory.
        my_device: Current GPU device.

    Returns:
        Dictionary with training metadata (global_step, epoch, etc.)
    """
    model_dir = os.path.join(checkpoint_dir, "model")
    training_state_dir = os.path.join(checkpoint_dir, "training_state")

    # Load model parameters (stage-independent)
    model.load_model(model_dir)

    # Load optimizer state (stage-independent)
    _load_optimizer_state(model, optimizer, training_state_dir, my_device)

    # Load scheduler state (stage-independent)
    _load_scheduler_state(lr_scheduler, training_state_dir)

    # Load training metadata
    metadata_path = os.path.join(training_state_dir, "metadata.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, map_location="cpu")
    else:
        metadata = {}

    # Load detach_state (per-node, per-stage)
    _load_detach_state(model, checkpoint_dir, my_device)

    return metadata


# ---------------------------------------------------------------------------
# Final checkpoint (model-only, for SFT loading)
# ---------------------------------------------------------------------------

def save_final_checkpoint(
    model,  # ModelHypernetwork
    run_name: str,
    global_step: int,
    epoch: int,
) -> float:
    """
    Save a final checkpoint containing only model weights (no optimizer/scheduler).

    This is saved at the end of training and is used as the starting point
    for SFT training.  The checkpoint is stored at:
        checkpoint/{run_name}/final/

    **Only node 0** writes checkpoint files.

    Args:
        model: The ModelHypernetwork model.
        run_name: Checkpoint run name (e.g. 'name/pretrain').
        global_step: Final global step.
        epoch: Final epoch.

    Returns:
        Wall-clock seconds spent saving (0.0 on non-node-0 processes).
    """
    if not is_node0():
        return 0.0

    save_t0 = time.time()

    final_dir = os.path.join(get_checkpoint_dir(run_name), "final")
    model_dir = os.path.join(final_dir, "model")

    # Save model (per-layer files — pipeline-stage-independent)
    model.save_model(model_dir)

    # Save minimal metadata (only from stage 0)
    stage = model._my_stage
    if stage == 0:
        training_state_dir = os.path.join(final_dir, "training_state")
        os.makedirs(training_state_dir, exist_ok=True)
        metadata = {
            "global_step": global_step,
            "epoch": epoch,
            "is_final": True,
        }
        torch.save(metadata, os.path.join(training_state_dir, "metadata.pt"))

    save_duration = time.time() - save_t0
    return save_duration


def load_model_only(
    model,  # ModelHypernetwork
    checkpoint_dir: str,
) -> Dict[str, Any]:
    """
    Load only model weights from a checkpoint (no optimizer/scheduler).

    Used by SFT to load pretrain final checkpoint as starting point.

    Args:
        model: The ModelHypernetwork model.
        checkpoint_dir: Path to the checkpoint directory (e.g. .../final/).

    Returns:
        Dictionary with training metadata (if available).
    """
    model_dir = os.path.join(checkpoint_dir, "model")

    # Load model parameters (stage-independent)
    model.load_model(model_dir)

    # Load training metadata (if available)
    metadata_path = os.path.join(checkpoint_dir, "training_state", "metadata.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, map_location="cpu")
    else:
        metadata = {}

    return metadata
