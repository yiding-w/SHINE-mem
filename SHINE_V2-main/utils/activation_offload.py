"""
Activation offload with cross-layer backward prefetch optimization.

All checkpoint-wrapped layers share a single global context so that
backward prefetch can look ahead across layer boundaries — e.g. while
layer N is recomputing, layer N-1's hidden_states are already being
transferred from CPU to GPU.

Usage:
    from utils.activation_offload import PrefetchOffloadContext

    # Create once, reuse across forward passes:
    ctx = PrefetchOffloadContext(prefetch_ahead=1)

    # Wrap the entire LLM forward (all layers share this context):
    with ctx:
        output = llm_model(...)

    # Inside each layer's checkpoint wrapper, do NOT create a new context.
    # Just use plain checkpoint — the outer context handles offload.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Tuple

import torch
from torch import Tensor


class PrefetchOffloadContext(torch.autograd.graph.saved_tensors_hooks):
    """Global activation offload context with cross-layer backward prefetch.

    Forward behavior (pack): synchronously copy GPU tensors to CPU memory
    (pageable, not pinned, to avoid host RAM pressure with 8 procs/node).

    Backward behavior (unpack): when a tensor is requested, prefetch the
    next ``prefetch_ahead`` tensors (in backward order) on a dedicated CUDA
    stream. Since all layers share one storage, this naturally prefetches
    the next layer's hidden_states while the current layer is recomputing.

    Args:
        prefetch_ahead: Number of tensors to prefetch ahead during backward.
            Default 1 means: when unpacking tensor at index i, also start
            loading tensor at index i-1 (the next one needed in backward).
            For 64-layer models with ~100ms per layer backward, even 1 is
            enough to hide the ~5-10ms H2D transfer of a 160MB tensor.
    """

    def __init__(self, prefetch_ahead: int = 1):
        self._prefetch_ahead = prefetch_ahead

        # Storage for offloaded tensors: list of (gpu_device, cpu_tensor)
        self._storage: list[Tuple[torch.device, Optional[Tensor]]] = []
        self._pack_counter = 0

        # Prefetch state (backward only)
        self._prefetched: dict[int, Tuple[Tensor, torch.cuda.Event]] = {}
        self._prefetch_requested: set = set()
        self._reload_stream: Optional[torch.cuda.Stream] = None
        self._lock = threading.Lock()

        # Track the last unpacked index to determine backward direction
        self._last_unpack_idx: int = -1

        # Nesting depth: multiple `with ctx:` blocks may be used in one step
        self._depth = 0

        super().__init__(self._pack, self._unpack)

    def __enter__(self):
        """Activate hooks. Storage is NOT reset here — call
        reset_for_new_step() explicitly between training steps."""
        self._depth += 1
        return super().__enter__()

    def __exit__(self, *args):
        self._depth -= 1
        return super().__exit__(*args)

    def reset_for_new_step(self):
        """Call this at the beginning of each training step to clear state
        from the previous step. Must NOT be called while backward is active."""
        self._reset()

    def _reset(self):
        """Clear all state for a new forward pass."""
        # Free any lingering prefetched GPU tensors
        self._prefetched.clear()
        self._prefetch_requested.clear()
        self._storage.clear()
        self._pack_counter = 0
        self._last_unpack_idx = -1

    def _get_reload_stream(self, device: torch.device) -> torch.cuda.Stream:
        if self._reload_stream is None:
            self._reload_stream = torch.cuda.Stream(device=device)
        return self._reload_stream

    def _pack(self, tensor: Tensor) -> Any:
        """Forward: copy tensor to CPU (same as save_on_cpu)."""
        if not tensor.is_cuda:
            # Non-CUDA tensor: store as-is
            with self._lock:
                idx = self._pack_counter
                self._pack_counter += 1
                self._storage.append((tensor.device, tensor))
            return idx

        device = tensor.device
        # Synchronous D2H copy to pageable memory.
        # We intentionally do NOT use pin_memory to avoid host RAM pressure
        # when 8 processes per node are all doing offload simultaneously.
        cpu_tensor = tensor.to("cpu")

        with self._lock:
            idx = self._pack_counter
            self._pack_counter += 1
            self._storage.append((device, cpu_tensor))

        return idx

    def _unpack(self, idx: int) -> Tensor:
        """Backward: retrieve tensor from CPU, with prefetch optimization."""
        # Trigger prefetch for upcoming tensors (in backward direction)
        self._trigger_prefetch(idx)

        # Check if this tensor was already prefetched
        with self._lock:
            if idx in self._prefetched:
                gpu_tensor, reload_event = self._prefetched.pop(idx)
                self._prefetch_requested.discard(idx)
                # Release CPU tensor reference to free host memory
                self._storage[idx] = (self._storage[idx][0], None)
                # Update direction tracking
                self._last_unpack_idx = idx
                # Wait for prefetch to complete on current compute stream
                torch.cuda.current_stream(gpu_tensor.device).wait_event(reload_event)
                return gpu_tensor

        # Not prefetched — synchronous fallback
        if idx >= len(self._storage):
            raise IndexError(
                f"activation_offload: _unpack called with idx={idx} but "
                f"_storage has only {len(self._storage)} entries. This likely "
                f"means reset_for_new_step() was called while a previous "
                f"backward pass still needed these tensors."
            )
        device, cpu_tensor = self._storage[idx]
        # Release CPU tensor reference to free host memory
        self._storage[idx] = (device, None)
        # Update direction tracking
        self._last_unpack_idx = idx
        if device.type == "cuda":
            return cpu_tensor.to(device)
        return cpu_tensor

    def _trigger_prefetch(self, current_idx: int) -> None:
        """Prefetch ALL tensors of the next layer in backward direction.

        Each layer stores ~3 tensors in forward order:
            [hidden_states (160MB), cos (2-4MB), sin (2-4MB)]

        Strategy: find the next large tensor (hidden_states) by scanning
        backwards, then prefetch it AND all tensors between it and current_idx
        (which are the cos/sin of that same layer). This way, when a layer's
        backward starts, ALL its tensors are already on GPU.

        Backward unpacks in reverse order: sin(idx=N+2) → cos(idx=N+1) → hs(idx=N).
        When we unpack sin of layer[i], we prefetch everything for layer[i-1]:
        i.e. all tensors from the previous large tensor down to (and including)
        the small tensors that follow it.
        """
        SIZE_THRESHOLD = 10_000_000  # 10MB — identifies hidden_states

        with self._lock:
            # Limit: at most one layer's worth of prefetched tensors at a time
            # to control GPU memory (~168MB per layer = hs + cos + sin)
            if len(self._prefetched) >= 6:
                return

            storage_len = len(self._storage)

            # Step 1: Find the next large tensor (hidden_states) scanning backwards
            large_idx = -1
            scan_idx = current_idx - 1
            while scan_idx >= 0:
                if scan_idx >= storage_len:
                    scan_idx -= 1
                    continue
                if (scan_idx not in self._prefetched
                        and scan_idx not in self._prefetch_requested):
                    device, cpu_tensor = self._storage[scan_idx]
                    if device.type == "cuda" and cpu_tensor is not None:
                        tensor_bytes = cpu_tensor.numel() * cpu_tensor.element_size()
                        if tensor_bytes >= SIZE_THRESHOLD:
                            large_idx = scan_idx
                            break
                scan_idx -= 1

            if large_idx < 0:
                return

            # Step 2: Collect the large tensor AND all small tensors that follow it
            # (up to current_idx, exclusive). These are the cos/sin of the same layer.
            candidates = []
            for idx in range(large_idx, current_idx):
                if idx >= storage_len:
                    break
                if (idx not in self._prefetched
                        and idx not in self._prefetch_requested):
                    device, cpu_tensor = self._storage[idx]
                    if device.type == "cuda" and cpu_tensor is not None:
                        candidates.append((idx, cpu_tensor, device))
                        self._prefetch_requested.add(idx)

            if not candidates:
                return

            # Issue all H2D copies on reload stream (entire layer at once)
            device = candidates[0][2]
            reload_stream = self._get_reload_stream(device)

            for candidate_idx, cpu_tensor, device in candidates:
                with torch.cuda.stream(reload_stream):
                    gpu_tensor = cpu_tensor.to(device, non_blocking=True)
                reload_event = reload_stream.record_event()
                self._prefetched[candidate_idx] = (gpu_tensor, reload_event)