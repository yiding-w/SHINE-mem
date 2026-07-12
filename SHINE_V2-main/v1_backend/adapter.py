from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from hypernetwork.detach_state import create_detach_state
from utils.myloradict import concat_loradict, collect_loradict_tensors
from utils.myparallel import is_main_process_per_node
from v1_backend.tp_loader_v1 import load_v1_qwen3_for_tp


logger = logging.getLogger(__name__)


def _ensure_repo_root_on_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    v2_root = os.path.dirname(here)
    repo_root = os.path.dirname(v2_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def _move_loradict_leaf(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device).detach().requires_grad_()
    if isinstance(obj, dict):
        return {k: _move_loradict_leaf(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_loradict_leaf(v, device) for v in obj]
    return obj


def _wrap_loradict_with_wdict(loradict, wdict):
    if loradict is None:
        return None
    if wdict is None:
        return loradict

    def _wrap(ld_node, wd_node):
        if ld_node is None:
            return None
        if isinstance(ld_node, dict) and "A" in ld_node and "B" in ld_node:
            return {"grad": ld_node, "state": wd_node}
        return {k: _wrap(v, None if wd_node is None else wd_node.get(k)) for k, v in ld_node.items()}

    return _wrap(loradict, wdict)


def _state_only_loradict(wdict):
    if wdict is None:
        return None

    def _wrap(node):
        if isinstance(node, dict) and "W" in node:
            return {"grad": None, "state": node}
        if isinstance(node, dict):
            return {k: _wrap(v) for k, v in node.items()}
        return None

    return _wrap(wdict)


def _lengths_to_mask(input_ids: torch.Tensor, lengths: Optional[torch.Tensor]):
    if lengths is None:
        return None
    seq_len = input_ids.shape[1]
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).to(dtype=torch.long)


def _trim_context_to_lengths(input_ids: torch.Tensor, lengths: Optional[torch.Tensor]):
    """Drop right padding before feeding contexts to the SHINE-v1 metanetwork.

    V2 collators allocate ``context_seq_length + num_mem_token`` slots because
    native V2 models use explicit memory-token placeholders. SHINE-v1 appends
    its own mem tokens inside LoraQwen, so the adapter should pass only the real
    context prefix and mask that prefix as valid.
    """
    if lengths is None:
        return input_ids, lengths
    max_len = int(lengths.max().item())
    if max_len <= 0:
        max_len = 1
    return input_ids[:, :max_len], lengths


class V1TPModelAdapter(nn.Module):
    """Run a SHINE-v1 Qwen3/metanetwork checkpoint inside the v2 TP trainer."""

    def __init__(
        self,
        cfg,
        tp_rank: int,
        tp_world: int,
        tp_process_group,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        _ensure_repo_root_on_path()
        from metanetwork_family import Metanetwork

        self.cfg = cfg
        self.v1_cfg = cfg.v1_backend
        self._tp_rank = tp_rank
        self._tp_world = tp_world
        self._tp_group = tp_process_group
        self._dtype = dtype
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")

        model_path = str(self.v1_cfg.get("model_path", cfg.model.path))
        lora_r = int(self.v1_cfg.lora_r)
        mean_pool_size = int(self.v1_cfg.metanetwork.transformer_cfg.get("mean_pool_size", 1))
        self.llm = load_v1_qwen3_for_tp(
            model_path,
            lora_r=lora_r,
            mean_pool_size=mean_pool_size,
            tp_rank=tp_rank,
            tp_world=tp_world,
            tp_group=tp_process_group,
            dtype=dtype,
            freeze=True,
            device=self._device,
        )

        self._num_llm_layers = int(self.llm.config.num_hidden_layers)
        self._num_mem_token = int(self.llm.model.num_mem_token)
        self._vocab_size = int(self.llm.config.vocab_size)

        mn_cfg = self._build_v1_metanetwork_cfg()
        self.v1_metanetwork = Metanetwork(
            self.llm,
            mn_cfg,
            self.llm.lora_params_numel(lora_r),
        ).to(self._device)
        self.hypernetwork = self.v1_metanetwork.metanetwork
        self.metalora = self.llm.init_lora_dict(
            int(self.v1_cfg.metalora_r),
            scale=float(self.v1_cfg.metanetwork.transformer_cfg.scale),
            device=self._device,
        )
        self.detach_state = None
        self.w_transform_context = None
        self.w_transform_conversation = None
        self._cached_loradict_for_write = None
        self._cached_precomputed_wdict = None

        warm_start_dir = self.v1_cfg.get("warm_start_dir", None)
        if warm_start_dir:
            self.load_v1_checkpoint(str(warm_start_dir))

    @property
    def is_v1_backend(self) -> bool:
        return True

    def _build_v1_metanetwork_cfg(self):
        return OmegaConf.create(
            {
                "hidden_size": int(self.llm.config.hidden_size),
                "num_layers": int(self.llm.config.num_hidden_layers),
                "num_mem_token": int(self.llm.model.num_mem_token),
                "model": {"lora_r": int(self.v1_cfg.lora_r)},
                "optim": {"adapter_reg": float(self.v1_cfg.get("adapter_reg", 0.0))},
                "metanetwork": OmegaConf.to_container(self.v1_cfg.metanetwork, resolve=True),
            }
        )

    def load_v1_checkpoint(self, checkpoint_dir: str) -> None:
        if is_main_process_per_node():
            logger.info(f"[v1_backend] Loading v1 checkpoint from {checkpoint_dir}")
        mem_path = os.path.join(checkpoint_dir, "mem_tokens.pt")
        meta_path = os.path.join(checkpoint_dir, "metanetwork.pth")
        metalora_path = os.path.join(checkpoint_dir, "metalora.pth")
        if os.path.isfile(mem_path):
            mem = torch.load(mem_path, map_location=self._device, weights_only=False)
            if tuple(mem.shape) != tuple(self.llm.model.mem_tokens.shape):
                raise ValueError(
                    f"mem_tokens shape mismatch: checkpoint={tuple(mem.shape)} "
                    f"current={tuple(self.llm.model.mem_tokens.shape)}"
                )
            self.llm.model.mem_tokens.data.copy_(mem.to(self._device, dtype=self.llm.model.mem_tokens.dtype))
        self.hypernetwork.load_state_dict(torch.load(meta_path, map_location=self._device, weights_only=False))
        self.metalora = _move_loradict_leaf(
            torch.load(metalora_path, map_location=self._device, weights_only=False),
            self._device,
        )

    def init_detach_state(
        self,
        *,
        local_batch_size: int,
        micro_batch_size: int,
        tp_rank: int,
        tp_world: int,
        tp_process_group,
        data_parallel_size: int = 1,
        grad_accum_steps: int = 1,
    ) -> None:
        ds_cfg = self.cfg.get("detach_state", None)
        if ds_cfg is None:
            self.detach_state = None
            return
        self.detach_state = create_detach_state(
            cfg=ds_cfg,
            mode="tp",
            local_batch_size=local_batch_size,
            micro_batch_size=micro_batch_size,
            parallel_mode="tp",
            tp_rank=tp_rank,
            tp_world=tp_world,
            tp_process_group=tp_process_group,
            num_llm_layers=self._num_llm_layers,
            data_parallel_size=data_parallel_size,
        )

    def forward(
        self,
        context_ids,
        context_lengths,
        conversation_ids,
        labels,
        context_attention_mask=None,
        conv_attention_mask=None,
        return_per_token_loss: bool = False,
        grad_accum_steps: int = 1,
        **kwargs,
    ):
        _, ds_wdict = self.detach_state.read() if self.detach_state is not None else (None, None)
        context_ids, context_lengths = _trim_context_to_lengths(context_ids, context_lengths)
        if context_attention_mask is None:
            context_attention_mask = _lengths_to_mask(context_ids, context_lengths)

        raw_loradict, _plain = self.v1_metanetwork.generate_lora_dict(
            context_ids,
            context_attention_mask,
            self.metalora,
            use_gradient_checkpoint=bool(self.v1_cfg.get("use_gradient_checkpoint", True)),
            return_plain=True,
        )
        forward_loradict = _wrap_loradict_with_wdict(raw_loradict, ds_wdict)
        outputs = self.llm(
            input_ids=conversation_ids,
            attention_mask=conv_attention_mask,
            labels=labels,
            loradict=forward_loradict,
            ignore_mem_token=True,
            use_gradient_checkpoint=bool(self.v1_cfg.get("use_gradient_checkpoint", True)),
        )

        regu_sq_norm = 0.0
        regu_loss = None
        precomputed = None
        if self.detach_state is not None:
            regu_sq_norm, regu_loss, precomputed = self.detach_state.compute_regu_loss(
                raw_loradict,
                mb_idx=0,
                num_mb=1,
                grad_accum_steps=grad_accum_steps,
            )
            regu_sq_norm = 0.0 if regu_sq_norm is None else regu_sq_norm
        self._cached_loradict_for_write = raw_loradict
        self._cached_precomputed_wdict = precomputed

        if return_per_token_loss:
            if getattr(outputs, "logits", None) is None:
                raise RuntimeError("V1 backend cannot compute per-token loss because model output has no logits.")
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(shift_labels).float()
            return (outputs.loss, per_token_loss), regu_sq_norm, regu_loss

        return outputs.loss, regu_sq_norm, regu_loss

    def _read_detach_state(self, mb_idx=None):
        if self.detach_state is None:
            return None, None
        return self.detach_state.read(mb_idx)

    def _write_detach_state(self, loradict: Optional[Dict], mb_idx=None, precomputed_wdict=None):
        if self.detach_state is None or loradict is None:
            return
        self.detach_state.write(loradict, mb_idx=mb_idx, precomputed_wdict=precomputed_wdict)

    def compute_memory_states(
        self,
        context_ids,
        context_attention_mask=None,
        context_lengths=None,
        nograd_loradict=None,
        nograd_wdict=None,
        **kwargs,
    ):
        """Compatibility hook for eval_memory_gen.

        SHINE-v1's metanetwork API directly maps context tokens to a loradict,
        while the native v2 path separates memory-state computation from the
        loradict projection. Return the generated loradict here and let
        generate_loradict pass it through.
        """
        context_ids, context_lengths = _trim_context_to_lengths(context_ids, context_lengths)
        if context_attention_mask is None:
            context_attention_mask = _lengths_to_mask(context_ids, context_lengths)
        raw_loradict, _plain = self.v1_metanetwork.generate_lora_dict(
            context_ids,
            context_attention_mask,
            self.metalora,
            use_gradient_checkpoint=False,
            return_plain=True,
        )
        return raw_loradict

    def generate_loradict(self, memory_states):
        return memory_states

    def conversation_loradict_from_generated(self, loradict):
        return loradict

    def prepare_generation_loradict(self, loradict, ds_loradict=None, ds_wdict=None):
        wrapped = _wrap_loradict_with_wdict(loradict, ds_wdict)
        if wrapped is None:
            wrapped = _state_only_loradict(ds_wdict)
        return wrapped

    def post_backward_detach_state(self, grad_accum_steps: int = 1):
        if self.detach_state is not None and self._cached_loradict_for_write is not None:
            self.detach_state.write(
                self._cached_loradict_for_write,
                mb_idx=0,
                precomputed_wdict=self._cached_precomputed_wdict,
            )
        self._cached_loradict_for_write = None
        self._cached_precomputed_wdict = None
        return 0.0

    def save_model(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.llm.model.mem_tokens.detach().cpu(), os.path.join(save_dir, "mem_tokens.pt"))
        torch.save(self.hypernetwork.state_dict(), os.path.join(save_dir, "metanetwork.pth"))
        cpu_metalora = self._loradict_to_cpu(self.metalora)
        torch.save(cpu_metalora, os.path.join(save_dir, "metalora.pth"))

    def load_model(self, load_dir: str):
        self.load_v1_checkpoint(load_dir)

    def _loradict_to_cpu(self, obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: self._loradict_to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._loradict_to_cpu(v) for v in obj]
        return obj
