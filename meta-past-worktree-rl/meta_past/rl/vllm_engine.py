"""Per-rank co-located vLLM engine with sleep/wake mode.

Architecture (verl HybridEngine pattern):
- Each training rank owns one ``vllm.LLM(tensor_parallel_size=1,
  enable_sleep_mode=True)`` on its own GPU.
- During rollout: ``wake_up`` → push LoRA → generate → ``sleep``.
- During training: vLLM weights are offloaded to CPU; the GPU is free for
  the rescore forward + backward.
- LoRA push uses ``collective_rpc(callable)`` directly on the in-process
  LLM — no IPC, no msgspec serialization, no zmq. Tensors pass by Python
  reference between the trainer and the LoRA-loading worker callable.

Replaces the multi-process ``VLLMPool`` (which was an artifact of having
training and vLLM in separate processes / envs).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import torch  # noqa: F401  — annotations only


logger = logging.getLogger("meta_past.rl.vllm_engine")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class SampledSequence:
    text: str
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None


@dataclass
class VLLMEngineConfig:
    model_path: str
    max_loras: int = 16
    max_lora_rank: int = 8
    max_model_len: int = 2048
    dtype: str = "bfloat16"
    # During rollout vLLM has the GPU; during train SHINE+Adam own it. Set
    # this conservatively low so vLLM doesn't grab too much when awake — the
    # sleeping budget is what matters most.
    gpu_memory_utilization: float = 0.55
    enforce_eager: bool = False
    seed: int = 0


# ---------------------------------------------------------------------------
# Worker-side helpers
# ---------------------------------------------------------------------------


def _register_lora_in_worker(
    worker,
    lora_id: int,
    tensors: dict,
    peft_meta: dict,
) -> bool:
    """Build ``LoRAModel`` from in-memory tensors and register with this worker.

    Sent via ``llm.collective_rpc(callable, ...)``. Same as the multi-process
    ``vllm_pool`` version, but now runs in the same Python process — no
    serialization across zmq/msgspec.
    """
    from vllm.lora.models import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper

    mgr = worker.model_runner.lora_manager
    device = mgr.device
    dtype = worker.model_runner.lora_config.lora_dtype
    if dtype is None or (isinstance(dtype, str) and dtype == "auto"):
        dtype = worker.model_runner.model_config.dtype

    on_device_tensors = {
        k: v.to(device=device, dtype=dtype, non_blocking=False)
        for k, v in tensors.items()
    }
    peft_helper = PEFTHelper.from_dict(peft_meta)
    lora = LoRAModel.from_lora_tensors(
        lora_model_id=lora_id,
        tensors=on_device_tensors,
        peft_helper=peft_helper,
        device=device,
        dtype=dtype,
        embedding_modules=mgr.embedding_modules,
        embedding_padding_modules=mgr.embedding_padding_modules,
    )
    if len(mgr._adapter_manager) + 1 > mgr._adapter_manager.capacity:
        mgr._adapter_manager.remove_oldest_adapter()
    mgr._adapter_manager.add_adapter(lora)
    return True


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class VLLMEngine:
    """In-process vLLM wrapper with sleep/wake + in-memory LoRA push."""

    def __init__(self, cfg: VLLMEngineConfig):
        self.cfg = cfg
        self._llm = None
        self._asleep = False

    # -- lifecycle -------------------------------------------------------------

    def boot(self) -> None:
        """Construct the LLM. Must be called *after* CUDA_VISIBLE_DEVICES /
        torch.cuda.set_device has been pinned to the rank's GPU.

        ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` keeps the engine in this process
        so ``collective_rpc`` args (torch tensors, dicts) pass by Python ref
        and aren't mangled by zmq+msgspec.
        """
        if self._llm is not None:
            return
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from vllm import LLM

        logger.info("constructing LLM(model=%s, enable_sleep_mode=True)",
                    self.cfg.model_path)
        self._llm = LLM(
            model=self.cfg.model_path,
            tensor_parallel_size=1,
            enable_lora=True,
            max_loras=self.cfg.max_loras,
            max_lora_rank=self.cfg.max_lora_rank,
            max_cpu_loras=self.cfg.max_loras,
            max_model_len=self.cfg.max_model_len,
            dtype=self.cfg.dtype,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            enforce_eager=self.cfg.enforce_eager,
            seed=self.cfg.seed,
            disable_log_stats=True,
            enable_sleep_mode=True,
        )
        # vLLM constructs awake; immediately sleep so the GPU is free for
        # SHINE construction + AdamW init.
        self.sleep()

    def sleep(self, level: int = 1) -> None:
        """Move vLLM weights to CPU (level=1) or also free KV cache (level=2).

        Idempotent: a no-op if already asleep.
        """
        if self._llm is None or self._asleep:
            return
        try:
            self._llm.sleep(level=level)
            self._asleep = True
            logger.debug("vLLM asleep (level=%d).", level)
        except Exception:
            logger.exception("LLM.sleep failed; continuing.")

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Move vLLM weights back to GPU. Idempotent.

        Raises on failure so the caller (rollout) doesn't silently proceed
        to a push/generate against a half-woken engine. The most common
        failure is HBM exhaustion when the training residual + AdamW state
        leaves no room for vLLM weights — lower
        ``gpu_memory_utilization`` if you hit it.
        """
        if self._llm is None or not self._asleep:
            return
        kwargs: dict[str, Any] = {}
        if tags is not None:
            kwargs["tags"] = tags
        self._llm.wake_up(**kwargs)
        self._asleep = False
        logger.debug("vLLM awake.")

    def shutdown(self) -> None:
        if self._llm is not None:
            try:
                # Best effort; vLLM doesn't always expose a clean shutdown.
                del self._llm
            finally:
                self._llm = None

    # -- LoRA push -------------------------------------------------------------

    def push_lora_batch(
        self,
        per_b_tensors: Sequence[dict],
        lora_ids: Sequence[int],
        peft_meta: dict,
    ) -> None:
        """Register a batch of LoRAs on this rank's LLM.

        Each ``per_b_tensors[b]`` is a PEFT-name → tensor dict for one
        context. ``lora_ids[b]`` is the integer id this LoRA is registered
        under (must be ≥ 1; 0 is reserved by vLLM).
        """
        if self._llm is None:
            raise RuntimeError("VLLMEngine.boot() must be called first.")
        if self._asleep:
            raise RuntimeError("Cannot push LoRA while vLLM is asleep; wake_up first.")
        if len(per_b_tensors) != len(lora_ids):
            raise ValueError(
                f"per_b_tensors ({len(per_b_tensors)}) and lora_ids "
                f"({len(lora_ids)}) length mismatch."
            )
        for tensors, lid in zip(per_b_tensors, lora_ids):
            self._llm.collective_rpc(
                _register_lora_in_worker,
                args=(int(lid), tensors, peft_meta),
            )

    # -- sampling --------------------------------------------------------------

    def complete(
        self,
        prompts: Sequence[str],
        lora_ids: Sequence[int] | None,
        n: int,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> list[list[SampledSequence]]:
        """Sample ``n`` completions per prompt.

        ``lora_ids=None`` (or all-zero) → no LoRA: run the base model
        directly. Each ``lora_ids[i] > 0`` → use the corresponding
        pre-registered in-memory LoRA for prompt ``i``.

        Returns ``list[len(prompts)]`` of ``list[n]`` ``SampledSequence``\\s.
        """
        if self._llm is None:
            raise RuntimeError("VLLMEngine.boot() must be called first.")
        if self._asleep:
            raise RuntimeError("Cannot generate while vLLM is asleep; wake_up first.")
        if not prompts:
            return []

        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sp_kwargs: dict[str, Any] = {
            "n": int(n),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
        }
        if seed is not None:
            sp_kwargs["seed"] = int(seed)
        if stop is not None:
            sp_kwargs["stop"] = list(stop)
        sp = SamplingParams(**sp_kwargs)

        # No-LoRA path: lora_ids=None or every id is 0 (the reserved no-LoRA id).
        # vLLM's LoRARequest with id=0 + path='in-memory' triggers a path-load
        # and fails — so we must pass ``lora_request=None`` to skip the LoRA
        # path entirely.
        use_lora = lora_ids is not None and any(int(lid) > 0 for lid in lora_ids)

        if use_lora:
            if len(prompts) != len(lora_ids):  # type: ignore[arg-type]
                raise ValueError(
                    f"prompts ({len(prompts)}) and lora_ids "
                    f"({len(lora_ids)}) length mismatch."  # type: ignore[arg-type]
                )
            lora_requests = [
                LoRARequest(
                    lora_name=f"in-mem-{lid}",
                    lora_int_id=int(lid),
                    lora_path="in-memory",  # dummy; pre-registered LoRA short-circuits path-load
                )
                for lid in lora_ids  # type: ignore[union-attr]
            ]
        else:
            lora_requests = None  # type: ignore[assignment]

        outputs = self._llm.generate(
            prompts=list(prompts),
            sampling_params=sp,
            lora_request=lora_requests,
            use_tqdm=False,
        )
        result: list[list[SampledSequence]] = []
        for ro in outputs:
            samples = []
            for co in ro.outputs:
                samples.append(SampledSequence(
                    text=co.text,
                    token_ids=list(co.token_ids),
                    finish_reason=getattr(co, "finish_reason", None),
                ))
            result.append(samples)
        return result
