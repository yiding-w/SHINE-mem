from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


class V1FullDetachState:
    """
    Lightweight SHINE v1-compatible detach_state.

    It stores detached full delta weights instead of concatenating historical
    LoRA ranks:
        W_state <- W_state + A @ B
        C_state <- C_state + C
    The tree layout mirrors the v1 loradict, but each leaf is {"W", "C"}.
    """

    def __init__(self, cfg=None, *, local_batch_size: int = 1):
        self._cfg = cfg or {}
        self._local_batch_size = int(local_batch_size)
        self._wdict: Optional[Dict] = None
        self._last_sq_norms: Optional[List[float]] = None
        self._update_steps: List[int] = [0] * self._local_batch_size
        self._state_device = self._get_cfg_value("state_device", "cpu")
        self._state_dtype = self._parse_dtype(self._get_cfg_value("state_dtype", "bfloat16"))

    def read(self) -> Optional[Dict]:
        return self._wdict

    def write(self, loradict: Optional[Dict]) -> List[float]:
        """Accumulate one generated loradict into the detached state."""
        if loradict is None:
            return self.per_sample_sq_norms()

        with torch.no_grad():
            new_wdict = self._loradict_to_wdict(loradict)
            if self._wdict is None:
                self._wdict = new_wdict
            elif self._infer_batch_size(self._wdict) != self._infer_batch_size(new_wdict):
                self._wdict = new_wdict
                self._last_sq_norms = None
                self._update_steps = [0] * self._infer_batch_size(new_wdict)
            else:
                self._accumulate_wdict(self._wdict, new_wdict)
            self._ensure_update_steps_len(self._infer_batch_size(self._wdict))
            return self.per_sample_sq_norms()

    def reset(self) -> None:
        if self._wdict is not None:
            self._zero_wdict(self._wdict)
        self.init_steps()
        self._last_sq_norms = None

    def reset_slice(self, sample_idx: int) -> None:
        if self._wdict is not None:
            self._zero_wdict_slice(self._wdict, sample_idx)
        self._ensure_update_steps_len(sample_idx + 1)
        self._update_steps[sample_idx] = 0

    def init_steps(self) -> None:
        self._update_steps = [0] * self._local_batch_size

    def update_steps(self, sample_idx: int) -> None:
        self._ensure_update_steps_len(sample_idx + 1)
        self._update_steps[sample_idx] += 1

    def update_all_steps(self) -> None:
        batch_size = self._infer_batch_size(self._wdict) if self._wdict is not None else self._local_batch_size
        self._ensure_update_steps_len(batch_size)
        for sample_idx in range(batch_size):
            self._update_steps[sample_idx] += 1

    def set_last_sq_norms(self, sq_norms: List[float]) -> None:
        self._last_sq_norms = list(sq_norms)
        self._ensure_update_steps_len(len(self._last_sq_norms))

    def maybe_reset_slice(self, sample_idx: int) -> bool:
        threshold = self._get_cfg_value("reset_threshold", None)
        if threshold is None:
            return False

        should_reset = False
        if threshold <= 0:
            should_reset = True
        elif self._last_sq_norms is not None and sample_idx < len(self._last_sq_norms):
            should_reset = self._last_sq_norms[sample_idx] > threshold

        if should_reset:
            self.reset_slice(sample_idx)
        return should_reset

    def maybe_reset_all(self) -> int:
        num_samples = len(self._last_sq_norms) if self._last_sq_norms is not None else self._local_batch_size
        return sum(1 for sample_idx in range(num_samples) if self.maybe_reset_slice(sample_idx))

    def get_reset_stats(self) -> Tuple[float, float]:
        threshold = self._get_cfg_value("reset_threshold", None)
        num_samples = len(self._last_sq_norms) if self._last_sq_norms is not None else len(self._update_steps)
        if num_samples == 0:
            return 0.0, 0.0

        if threshold is None:
            reset_ratio = 0.0
        elif threshold <= 0:
            reset_ratio = 1.0
        elif self._last_sq_norms is None:
            reset_ratio = 0.0
        else:
            reset_ratio = sum(1 for sq in self._last_sq_norms[:num_samples] if sq > threshold) / num_samples

        steps = self._update_steps[:num_samples]
        mean_update_step = sum(steps) / max(len(steps), 1)
        return reset_ratio, mean_update_step

    def per_sample_sq_norms(self) -> List[float]:
        if self._wdict is None:
            return []
        batch_size = self._infer_batch_size(self._wdict)
        norms = [0.0] * batch_size

        def _walk(node):
            if isinstance(node, dict) and "W" in node:
                W = node["W"]
                for idx in range(W.shape[0]):
                    norms[idx] += float(W[idx].float().pow(2).sum().item())
                C = node.get("C", None)
                if C is not None:
                    for idx in range(C.shape[0]):
                        norms[idx] += float(C[idx].float().pow(2).sum().item())
                return
            if isinstance(node, dict):
                for value in node.values():
                    _walk(value)

        _walk(self._wdict)
        return norms

    def state_dict(self) -> Dict:
        return {
            "type": "v1_full",
            "local_batch_size": self._local_batch_size,
            "wdict": self._wdict,
            "last_sq_norms": self._last_sq_norms,
            "update_steps": list(self._update_steps),
        }

    def load_state_dict(self, state: Dict) -> None:
        if state is None:
            return
        self._wdict = state.get("wdict", None)
        self._last_sq_norms = state.get("last_sq_norms", None)
        self._update_steps = list(state.get("update_steps", [0] * self._local_batch_size))
        if self._wdict is not None:
            self._ensure_no_grad(self._wdict)
            self._ensure_update_steps_len(self._infer_batch_size(self._wdict))

    def _loradict_to_wdict(self, loradict: Dict) -> Dict:
        def _convert(node):
            if isinstance(node, dict) and "A" in node and "B" in node:
                A = self._to_state_storage(node["A"].detach())
                B = self._to_state_storage(node["B"].detach())
                W = torch.bmm(A, B).detach()
                C = node.get("C", None)
                C = None if C is None else self._to_state_storage(C.detach())
                return {"W": W, "C": C}
            if isinstance(node, dict):
                return {key: _convert(value) for key, value in node.items()}
            raise TypeError(f"Unsupported loradict node type: {type(node)}")

        return _convert(loradict)

    def _accumulate_wdict(self, dst: Dict, src: Dict) -> None:
        if "W" in dst:
            if dst["W"].shape != src["W"].shape:
                raise ValueError(f"detach_state W shape mismatch: {dst['W'].shape} vs {src['W'].shape}")
            dst["W"].add_(src["W"])
            if (dst.get("C", None) is None) != (src.get("C", None) is None):
                raise ValueError("detach_state C mismatch: one leaf has C and the other does not")
            if dst.get("C", None) is not None:
                dst["C"].add_(src["C"])
            return
        if dst.keys() != src.keys():
            raise ValueError("detach_state tree key mismatch")
        for key in dst.keys():
            self._accumulate_wdict(dst[key], src[key])

    def _zero_wdict(self, node: Dict) -> None:
        if "W" in node:
            node["W"].zero_()
            if node.get("C", None) is not None:
                node["C"].zero_()
            return
        for value in node.values():
            self._zero_wdict(value)

    def _zero_wdict_slice(self, node: Dict, sample_idx: int) -> None:
        if "W" in node:
            if sample_idx < node["W"].shape[0]:
                node["W"][sample_idx].zero_()
            if node.get("C", None) is not None and sample_idx < node["C"].shape[0]:
                node["C"][sample_idx].zero_()
            return
        for value in node.values():
            self._zero_wdict_slice(value, sample_idx)

    def _infer_batch_size(self, node: Dict) -> int:
        if "W" in node:
            return int(node["W"].shape[0])
        for value in node.values():
            return self._infer_batch_size(value)
        return self._local_batch_size

    def _ensure_update_steps_len(self, size: int) -> None:
        if size <= len(self._update_steps):
            return
        self._update_steps.extend([0] * (size - len(self._update_steps)))

    def _ensure_no_grad(self, node) -> None:
        if torch.is_tensor(node):
            node.requires_grad_(False)
            return
        if isinstance(node, dict):
            for value in node.values():
                self._ensure_no_grad(value)

    def _get_cfg_value(self, key: str, default=None):
        if hasattr(self._cfg, "get"):
            return self._cfg.get(key, default)
        return getattr(self._cfg, key, default)

    def _parse_dtype(self, value):
        if value is None or value == "none":
            return None
        if isinstance(value, torch.dtype):
            return value
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        key = str(value).lower()
        if key not in mapping:
            raise ValueError(f"Unsupported detach_state.state_dtype: {value}")
        return mapping[key]

    def _to_state_storage(self, tensor: torch.Tensor) -> torch.Tensor:
        device = torch.device(self._state_device)
        return tensor.to(device=device, dtype=self._state_dtype).detach()
