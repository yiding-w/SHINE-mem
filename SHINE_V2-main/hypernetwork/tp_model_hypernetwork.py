"""
Tensor-parallel ``ModelHypernetwork`` replacement.

The PP version (``model_hypernetwork.py``, 2674 lines) glues together
the LLM, the hypernetwork, and a pipeline-parallel comm dance
(mem_gather, lora_scatter, three-phase forward, reverse-order
backward). Under TP every collective lives inside the per-linear
forward / backward; the outer module becomes a thin wrapper.

``TPModelHypernetwork`` exposes:

  * ``self.llm`` — the TP-loaded ``LoraQwen3_5ForCausalLM`` (full_attention
    layers TP-sharded, linear_attention layers replicated).
  * ``self.hypernetwork`` — the ``TPHypernetwork`` (replicated
    m2p_transformer + 2D pos embeddings).
  * ``compute_memory_states(input_ids, attention_mask, context_lengths)``
    — Step-1 forward of the (frozen) LLM with ``use_mem_token=True``.
    Returns ``(B, L_fa, M, H)`` per-layer hidden states at the mem
    token positions, already filtered to full_attention layers when
    memory_method is ``only_full_1for1``.
  * ``generate_loradict(memory_states)`` — feeds memory_states through
    the hypernetwork and the LLM's full-dim ``generate_lora_dict`` to
    produce the full unsharded loradict dict.
  * ``compute_loss(input_ids, labels, loradict, attention_mask)`` —
    Step-4 forward + lm_head + cross-entropy. Returns scalar loss.
  * ``forward(...)`` — composes all three for one training step.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Dict

import torch
import torch._dynamo
import torch.nn as nn

from utils.myparallel import is_main_process_per_node
from utils.mytp.tp_load_model import load_pretrained_llm_for_tp
from hypernetwork.tp_hypernetwork import TPHypernetwork
from utils.myloradict import concat_loradict, collect_loradict_tensors


logger = logging.getLogger(__name__)


__all__ = ["TPModelHypernetwork"]


class TPModelHypernetwork(nn.Module):
    """Composes a TP-sharded LLM with a replicated hypernetwork."""

    def __init__(
        self,
        model_cfg,
        m2p_transformer_cfg,
        tp_rank: int,
        tp_world: int,
        tp_process_group,
        dtype: torch.dtype = torch.bfloat16,
        activation_checkpointing: bool = True,
        ckpt_skip_stride: int = 0,
        compile_hypernetwork: bool = True,
    ):
        super().__init__()
        self._tp_rank = tp_rank
        self._tp_world = tp_world
        self._tp_group = tp_process_group
        self._dtype = dtype
        self._activation_checkpointing = activation_checkpointing
        self._ckpt_skip_stride = int(ckpt_skip_stride)

        # ----------------------------------------------------------------
        # 0. Compute num_mem_token from lora_ranks and the LLM config so
        #    that the LLM allocates mem_tokens of the right size when we
        #    load it. Mirrors ModelHypernetwork's compute step. M*H must
        #    equal the per-full_attention-layer LoRA param count.
        # ----------------------------------------------------------------
        from omegaconf import OmegaConf
        from transformers import AutoConfig
        from hydra.utils import get_original_cwd
        from src_transformers_lora.LoraHelper import compute_layer_lora_params_numel
        if hasattr(model_cfg, "lora_ranks"):
            lr_raw = model_cfg.lora_ranks
            if hasattr(lr_raw, "_metadata"):
                lora_ranks_dict = OmegaConf.to_container(lr_raw, resolve=True)
            else:
                lora_ranks_dict = dict(lr_raw)
        else:
            raise ValueError("model_cfg.lora_ranks is required")

        model_path = str(model_cfg.path)
        if not os.path.isabs(model_path):
            model_path = os.path.join(get_original_cwd(), model_path)
        _peek_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        _peek_text_cfg = _peek_cfg.get_text_config() if hasattr(_peek_cfg, "get_text_config") else _peek_cfg
        _hidden_size = int(_peek_text_cfg.hidden_size)
        _layer_types = list(_peek_text_cfg.layer_types)
        _first_fa = next(i for i, t in enumerate(_layer_types) if t == "full_attention")
        layer_lora_numel, breakdown = compute_layer_lora_params_numel(
            _peek_text_cfg, lora_ranks_dict, layer_idx=_first_fa, verbose=True,
        )
        # memory_method only_full_1for1 uses raw layer_lora_numel (no /4).
        if layer_lora_numel % _hidden_size != 0:
            if is_main_process_per_node():
                logger.error(
                    f"layer_lora_numel ({layer_lora_numel}) is not divisible by "
                    f"hidden_size ({_hidden_size}). Cannot compute num_mem_token.\n"
                    f"Remainder: {layer_lora_numel} % {_hidden_size} = {layer_lora_numel % _hidden_size}\n"
                    f"LoRA param breakdown for layer {_first_fa} (lora_ranks={lora_ranks_dict}):\n"
                    f"{breakdown}\n"
                    f"To fix: adjust lora_ranks so that the total is divisible by hidden_size, "
                    f"or pad the LoRA parameter count."
                )
            raise ValueError(
                f"layer_lora_numel ({layer_lora_numel}) not divisible by hidden_size ({_hidden_size}); "
                f"can't compute num_mem_token. Adjust lora_ranks."
            )
        num_mem_token_for_load = layer_lora_numel // _hidden_size
        if is_main_process_per_node():
            logger.info(
                f"[TPModelHypernetwork] Computed num_mem_token = {num_mem_token_for_load} "
                f"(lora_params_numel={layer_lora_numel}, hidden_size={_hidden_size}, "
                f"lora_ranks={lora_ranks_dict}, "
                f"Only add LoRA for full_attention layers (attention + experts). "
                f"linear_attention layers have no LoRA.)\n"
                f"LoRA param breakdown for layer {_first_fa}:\n"
                f"{breakdown}"
            )

        # ----------------------------------------------------------------
        # 1. Load the TP LLM (full_attention layers TP-sharded,
        #    linear_attention layers replicated per the design doc).
        # ----------------------------------------------------------------
        self.llm = load_pretrained_llm_for_tp(
            model_cfg=model_cfg,
            tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_process_group,
            dtype=dtype,
            freeze=True,
            num_mem_token=num_mem_token_for_load,
        )

        # ----------------------------------------------------------------
        # 1.5 Make mem_tokens requires_grad=True (same as PP).
        #     We don't train it now but keep it saveable/loadable for
        #     future use. PP does the same: requires_grad=True but not
        #     included in optimizer param groups.
        # ----------------------------------------------------------------
        if hasattr(self.llm.model, "mem_tokens") and self.llm.model.mem_tokens is not None:
            self.llm.model.mem_tokens.requires_grad_(True)

        # ----------------------------------------------------------------
        # 2. Wire lora_method onto every LoraLinear / LoraHelper /
        #    Colwise / Rowwise so generate_lora_dict knows how to chop
        #    plain_tensor into A / B / C. The choice (rl/rr/lr/ll) comes
        #    from m2p_transformer_cfg.
        # ----------------------------------------------------------------
        from omegaconf import OmegaConf
        if hasattr(m2p_transformer_cfg, "_metadata"):
            m2p_full_cfg = OmegaConf.to_container(m2p_transformer_cfg, resolve=True)
        elif isinstance(m2p_transformer_cfg, dict):
            m2p_full_cfg = dict(m2p_transformer_cfg)
        else:
            m2p_full_cfg = dict(m2p_transformer_cfg)
        lora_method = m2p_full_cfg.get("lora_method", "rl")
        self._lora_method = lora_method
        self._generate_lora_scale = float(m2p_full_cfg["generate_lora_scale"])
        self._init_lora_var = float(m2p_full_cfg["init_lora_var"])
        self._memory_method = m2p_full_cfg["memory_method"]

        if is_main_process_per_node():
            logger.info(
                f"[TPModelHypernetwork] LoRA config: lora_method={lora_method}, "
                f"generate_lora_scale={self._generate_lora_scale}, "
                f"init_lora_var={self._init_lora_var}, "
                f"memory_method={self._memory_method}"
            )

        # Set generate_func on every full_attention layer's LoRA-bearing
        # modules. self.llm.model.set_generate_func recurses appropriately.
        self.llm.model.set_generate_func(lora_method)

        # ----------------------------------------------------------------
        # 3. Per-layer metadata for memory_states filtering + loradict
        #    construction.
        # ----------------------------------------------------------------
        text_model = self.llm.model
        text_config = text_model.config
        layer_types = list(text_config.layer_types)
        self._layer_types = layer_types
        self._num_llm_layers = len(layer_types)
        self._full_attn_layer_indices = [
            i for i, t in enumerate(layer_types) if t == "full_attention"
        ]
        self._num_full_attn_layers = len(self._full_attn_layer_indices)
        self._hidden_size = text_config.hidden_size
        self._num_mem_token = int(getattr(text_config, "num_mem_token", 0) or 0)
        self._vocab_size = int(text_config.vocab_size)
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")

        if self._num_mem_token <= 0:
            raise ValueError(
                "TPModelHypernetwork requires num_mem_token > 0 in the LLM config "
                "(so that Step 1 produces memory_states for the hypernetwork)."
            )

        # ----------------------------------------------------------------
        # 4. Build the replicated hypernetwork.
        # ----------------------------------------------------------------
        self.hypernetwork = TPHypernetwork(
            m2p_transformer_cfg=m2p_transformer_cfg,
            num_llm_layers=self._num_llm_layers,
            num_full_attn_layers=self._num_full_attn_layers,
            num_mem_token=self._num_mem_token,
            memory_method=self._memory_method,
            dtype=dtype,
            device=self._device,
        )

        # Optional: torch.compile the replicated hypernetwork. The TP
        # collectives in the LLM (`_CopyToTPRegion` and the RowwiseLora
        # all-reduce) used to be a hard compile boundary, but they're now
        # marked ``@torch.compiler.disable``, so compile traces the regions
        # *between* them — see ``SHINE_COMPILE_LLM`` below.
        # Gated by env so it can be A/B'd without a config change.
        # env overrides config: SHINE_COMPILE_HN=0 force-off, =1 force-on.
        _env = os.environ.get("SHINE_COMPILE_HN")
        _do_compile = compile_hypernetwork if _env is None else (_env not in ("0", "", "false"))
        if _do_compile:
            self.hypernetwork = torch.compile(self.hypernetwork)
            if is_main_process_per_node():
                logger.info(
                    "[TPModelHypernetwork] torch.compile enabled on hypernetwork "
                    "(+5% step, -8 GB peak vs eager; ~3 min extra warmup)"
                )

        # torch.compile all LLM decoder layers (same as PP). With SDPA
        # attention, both full_attention and linear_attention layers are
        # dynamo-friendly. The TP all-reduce entries (copy_to_tp_region,
        # reduce_from_tp_region) and LigerRMSNorm/SwiGLU are marked
        # @torch.compiler.disable so dynamo treats them as opaque hops and
        # compiles the GEMM/LoRA-delta/attention regions in between.
        # Disable with SHINE_COMPILE_LLM=0 for A/B comparison.
        torch._dynamo.config.cache_size_limit = 64
        if hasattr(torch._dynamo.config, "recompile_limit"):
            torch._dynamo.config.recompile_limit = 64
        if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
            torch._dynamo.config.accumulated_cache_size_limit = 1024

        if os.environ.get("SHINE_COMPILE_LLM", "1") not in ("0", "", "false"):
            n_compiled = 0
            for layer in self.llm.model.layers:
                layer.forward = torch.compile(layer.forward, dynamic=False)
                n_compiled += 1
            if is_main_process_per_node():
                logger.info(
                    f"[TPModelHypernetwork] torch.compile enabled on "
                    f"{n_compiled} decoder layers (dynamic=False; "
                    f"TP collectives + Liger kernels stay opaque via "
                    f"@torch.compiler.disable)"
                )

        # ----------------------------------------------------------------
        # 5. Collect LoRA ranks from model_cfg.
        # ----------------------------------------------------------------
        lora_ranks = getattr(model_cfg, "lora_ranks", None)
        if lora_ranks is None:
            raise ValueError("model_cfg.lora_ranks is required")
        if hasattr(lora_ranks, "_metadata"):
            self._lora_ranks = dict(OmegaConf.to_container(lora_ranks, resolve=True))
        elif isinstance(lora_ranks, dict):
            self._lora_ranks = dict(lora_ranks)
        else:
            self._lora_ranks = dict(lora_ranks)

        # ----------------------------------------------------------------
        # 5.5 Initialize metalora (same as PP).
        #     metalora is a persistent set of trainable LoRA parameters
        #     that is concatenated with the hypernetwork-generated loradict.
        #     Under TP all layers are on the same device, so we init for all
        #     full_attention layers.
        # ----------------------------------------------------------------
        metalora_ranks_cfg = model_cfg.get("metalora_ranks", None)
        if metalora_ranks_cfg is None:
            raise ValueError(
                "model_cfg must contain 'metalora_ranks' dict mapping component "
                "names to their LoRA ranks for the metalora initialization."
            )
        metalora_ranks: dict = OmegaConf.to_container(metalora_ranks_cfg, resolve=True)
        self._metalora_ranks = metalora_ranks
        self.metalora: Dict[int, dict] = {}
        for layer_idx in self._full_attn_layer_indices:
            self.metalora[layer_idx] = self.llm.model.layers[layer_idx].init_lora_dict(
                metalora_ranks, self._init_lora_var, self._device, torch.bfloat16
            )
        # Fill non-full-attention layers with None (same as PP) so that
        # direct indexing metalora[layer_idx] never raises KeyError.
        for layer_idx in range(self._num_llm_layers):
            if layer_idx not in self.metalora:
                self.metalora[layer_idx] = None
        if is_main_process_per_node():
            # Check metalora A matrix norms to verify init_lora_var is correct
            first_fa = self._full_attn_layer_indices[0]
            first_metalora = self.metalora[first_fa]
            from utils.myloradict import collect_loradict_tensors
            first_tensors = collect_loradict_tensors(first_metalora)
            a_norms = [t.norm().item() for t in first_tensors[::2]]  # A matrices are at even indices
            b_norms = [t.norm().item() for t in first_tensors[1::2]]  # B matrices are at odd indices
            logger.info(
                f"[TPModelHypernetwork] Initialized metalora with "
                f"metalora_ranks={metalora_ranks}, dtype=bf16, "
                f"device={self._device}, "
                f"layers={len(self._full_attn_layer_indices)} full_attn layers, "
                f"init_lora_var={self._init_lora_var}, "
                f"layer{first_fa} A_norms(first 4)={a_norms[:4]}, "
                f"B_norms(first 4)={b_norms[:4]}"
            )

        # ----------------------------------------------------------------
        # 5.6 Initialize DetachState (bound 1:1 with this ModelHypernetwork).
        #     DetachState maintains a persistent no-grad state that accumulates
        #     the effect of generated LoRA dicts over time.
        #     For FullTPDetachState, full initialization is deferred to
        #     init_detach_state() which is called from tp_main after
        #     batch_size is known.
        # ----------------------------------------------------------------
        self._detach_state_cfg = model_cfg.get("detach_state", None)
        self.detach_state = None  # Deferred init via init_detach_state()

        # ----------------------------------------------------------------
        # 5.7 W-Transform modules (CompressMLP for bridging objective mismatch).
        #     Initialized here but only active when detach_state is full and
        #     w_transform_context/conversation.method != "identity".
        #     The _active_w_transform instance variable is set during forward
        #     to control which transform is applied in the checkpoint wrapper.
        # ----------------------------------------------------------------
        self._model_cfg = model_cfg  # Keep reference for deferred w_transform init
        self.w_transform_context = None
        self.w_transform_conversation = None
        self._active_w_transform = None  # Set during forward to control per-phase transform

        if is_main_process_per_node():
            logger.info(
                f"[TPModelHypernetwork] tp_rank={tp_rank}/{tp_world} "
                f"hidden={self._hidden_size} num_layers={self._num_llm_layers} "
                f"num_full_attn={self._num_full_attn_layers} "
                f"num_mem={self._num_mem_token} vocab={self._vocab_size} "
                f"lora_method={lora_method} memory_method={self._memory_method} "
                f"activation_checkpointing={activation_checkpointing}"
            )

        # ----------------------------------------------------------------
        # 6. Activation checkpointing on every decoder layer's forward
        #    pass — drops the layer's activations after forward and
        #    recomputes them in backward. Trades ~1 extra forward per
        #    layer per backward for ~½ activation memory, which lets us
        #    push micro-batch / sequence length higher.
        #    Also wraps layers with w_transform injection (even if
        #    checkpointing is disabled, w_transform still needs wrapping).
        # ----------------------------------------------------------------
        self._wrap_layers_with_checkpoint(use_checkpoint=activation_checkpointing)

        # ----------------------------------------------------------------
        # 7. Initialize W-Transform modules if configured.
        # ----------------------------------------------------------------
        self._init_w_transforms()


    def _init_w_transforms(self) -> None:
        """Initialize W-Transform modules based on detach_state config.

        Creates WTransformModule instances for context and/or conversation
        forward passes if their method is not "identity".
        """
        from omegaconf import OmegaConf
        ds_cfg = self._detach_state_cfg
        if ds_cfg is None:
            return

        # Convert OmegaConf to plain dict if needed
        if hasattr(ds_cfg, '_metadata'):
            ds_cfg_dict = OmegaConf.to_container(ds_cfg, resolve=True)
        elif not isinstance(ds_cfg, dict):
            ds_cfg_dict = dict(ds_cfg)
        else:
            ds_cfg_dict = ds_cfg

        # Only full-type detach_state supports w_transform
        ds_type = ds_cfg_dict.get("type", "empty")
        if ds_type == "empty":
            return

        ctx_cfg = ds_cfg_dict.get("w_transform_context", {"method": "identity"})
        conv_cfg = ds_cfg_dict.get("w_transform_conversation", {"method": "identity"})

        ctx_method = ctx_cfg.get("method", "identity") if isinstance(ctx_cfg, dict) else "identity"
        conv_method = conv_cfg.get("method", "identity") if isinstance(conv_cfg, dict) else "identity"

        if ctx_method == "identity" and conv_method == "identity":
            return

        from utils.mytransform import create_transform

        if ctx_method != "identity":
            self.w_transform_context = create_transform(
                cfg=ctx_cfg,
                model_cfg=self._model_cfg,
                num_layers=self._num_llm_layers,
                tp_mode=True,
                tp_rank=self._tp_rank,
                tp_world=self._tp_world,
                tp_group=self._tp_group,
                device=self._device,
                dtype=self._dtype,
                llm_model=self.llm,
            )
            if is_main_process_per_node():
                logger.info(
                    f"[TPModelHypernetwork] w_transform_context initialized: "
                    f"method={ctx_method}"
                )

        if conv_method != "identity":
            self.w_transform_conversation = create_transform(
                cfg=conv_cfg,
                model_cfg=self._model_cfg,
                num_layers=self._num_llm_layers,
                tp_mode=True,
                tp_rank=self._tp_rank,
                tp_world=self._tp_world,
                tp_group=self._tp_group,
                device=self._device,
                dtype=self._dtype,
                llm_model=self.llm,
            )
            if is_main_process_per_node():
                logger.info(
                    f"[TPModelHypernetwork] w_transform_conversation initialized: "
                    f"method={conv_method}"
                )

        # Optional: torch.compile w_transform modules for kernel fusion.
        # The TP all-reduce inside CompressMLP is marked @torch.compiler.disable,
        # so dynamo compiles the regions between all-reduces (compress einsum,
        # MLP, FiLM, cross-attn, decompress einsum) independently.
        # Gated by env: SHINE_COMPILE_WT=0 force-off, =1 force-on (default: on).
        _env_wt = os.environ.get("SHINE_COMPILE_WT")
        _do_compile_wt = True if _env_wt is None else (_env_wt not in ("0", "", "false"))
        if _do_compile_wt:
            if self.w_transform_context is not None:
                self.w_transform_context = torch.compile(self.w_transform_context)
                if is_main_process_per_node():
                    logger.info(
                        "[TPModelHypernetwork] torch.compile enabled on "
                        "w_transform_context (TP all-reduce stays opaque via "
                        "@torch.compiler.disable)"
                    )
            if self.w_transform_conversation is not None:
                self.w_transform_conversation = torch.compile(self.w_transform_conversation)
                if is_main_process_per_node():
                    logger.info(
                        "[TPModelHypernetwork] torch.compile enabled on "
                        "w_transform_conversation (TP all-reduce stays opaque via "
                        "@torch.compiler.disable)"
                    )

    def _wrap_layers_with_checkpoint(self, use_checkpoint: bool = True) -> None:
        """Wrap decoder layers' forward in ``torch.utils.checkpoint`` so
        activations are recomputed during backward instead of held in memory.
        The LLM is frozen — no parameter gradients are lost; only the LoRA's
        input activations matter.

        ``ckpt_skip_stride`` (s) selectively *un*-checkpoints 1 layer in
        every s, evenly spread (layer i is kept un-checkpointed when
        ``i % s == 0``). Recompute is the dominant backward cost (full
        checkpointing ≈ one extra 64-layer forward in backward); each
        un-checkpointed layer trades a slice of that recompute for its
        retained activation memory. s=0 (default) → checkpoint all 64.
        s=4 → 16 layers kept (≈ 25% recompute saved), spread so the peak
        activation memory stays bounded.

        Also injects w_transform: if self._active_w_transform is set,
        the wrapper applies it to nograd_wdict before calling the layer.

        Args:
            use_checkpoint: If True, wrap layers with gradient checkpointing.
                If False, only wrap for w_transform injection (no recompute).
        """
        from torch.utils.checkpoint import checkpoint
        s = self._ckpt_skip_stride if use_checkpoint else 0
        n_skipped = 0
        n_checkpointed = 0
        model_ref = self  # Capture reference for closure
        for idx, layer in enumerate(self.llm.model.layers):
            orig_forward = layer.forward
            if use_checkpoint and s > 0 and idx % s == 0:
                n_skipped += 1
                # Un-checkpointed layer: still needs w_transform injection
                def make_no_ckpt(fn, layer_idx):
                    def wrapped(*args, **kwargs):
                        w_transform = model_ref._active_w_transform
                        if w_transform is not None and 'nograd_wdict' in kwargs and kwargs['nograd_wdict'] is not None:
                            kwargs = dict(kwargs)
                            kwargs['nograd_wdict'] = w_transform(
                                kwargs['nograd_wdict'], layer_idx
                            )
                        return fn(*args, **kwargs)
                    return wrapped
                layer.forward = make_no_ckpt(orig_forward, idx)
                continue
            if use_checkpoint:
                n_checkpointed += 1
                def make(fn, layer_idx):
                    def wrapped(*args, **kwargs):
                        # Capture w_transform reference BEFORE entering checkpoint
                        # so that recomputation during backward uses the same
                        # transform that was active during the original forward.
                        w_transform = model_ref._active_w_transform
                        def fwd(*a):
                            if w_transform is not None and 'nograd_wdict' in kwargs and kwargs['nograd_wdict'] is not None:
                                transformed_kwargs = dict(kwargs)
                                transformed_kwargs['nograd_wdict'] = w_transform(
                                    kwargs['nograd_wdict'], layer_idx
                                )
                                return fn(*a, **transformed_kwargs)
                            return fn(*a, **kwargs)
                        return checkpoint(fwd, *args, use_reentrant=False)
                    return wrapped
                layer.forward = make(orig_forward, idx)
            else:
                # No checkpoint, just w_transform injection
                def make_no_ckpt2(fn, layer_idx):
                    def wrapped(*args, **kwargs):
                        w_transform = model_ref._active_w_transform
                        if w_transform is not None and 'nograd_wdict' in kwargs and kwargs['nograd_wdict'] is not None:
                            kwargs = dict(kwargs)
                            kwargs['nograd_wdict'] = w_transform(
                                kwargs['nograd_wdict'], layer_idx
                            )
                        return fn(*args, **kwargs)
                    return wrapped
                layer.forward = make_no_ckpt2(orig_forward, idx)
        if is_main_process_per_node():
            total = len(self.llm.model.layers)
            if use_checkpoint:
                logger.info(
                    f"[TPModelHypernetwork] activation checkpointing: "
                    f"{n_checkpointed}/{total} layers checkpointed "
                    f"(ckpt_skip_stride={s}, {n_skipped} kept for less recompute)"
                )
            else:
                logger.info(
                    f"[TPModelHypernetwork] activation checkpointing disabled; "
                    f"all {total} layers wrapped for w_transform injection only"
                )

    # ------------------------------------------------------------------
    # Step 1: compute memory_states (context_ids + metalora → LLM)
    # ------------------------------------------------------------------

    def compute_memory_states(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        context_lengths: torch.LongTensor,
        nograd_loradict: Optional[Dict] = None,
        nograd_wdict: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Step 1 — forward the LLM with ``use_mem_token=True`` and metalora,
        return per-layer mem-token hidden states for full_attention layers.

        Same as PP: context_ids + metalora → LLM → memory_states.
        Gradients flow through metalora (it's trainable).

        For only_full_1for1: returns ``(B, L_fa, M, H)`` already filtered.
        For only_full_4for1: returns ``(B, L, M, H)`` unfiltered (the
            hypernetwork's ``_initial_reshape`` will reduce L by 4).

        Args:
            nograd_loradict: Optional detached loradict from detach_state.
                Passed as nograd_loradict to the LLM forward (no gradient).
            nograd_wdict: Optional detached wdict from detach_state.
                Passed as nograd_wdict to the LLM forward (no gradient).
        """
        # PP does not pass attention_mask (uses None → flash attention with
        # is_causal=True). To match PP behaviour, we also pass None here.
        # Padding tokens are handled at loss computation time via ignore_index=-100.
        out = self.llm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loradict=self.metalora,
            nograd_loradict=nograd_loradict,
            nograd_wdict=nograd_wdict,
            use_mem_token=True,
            context_lengths=context_lengths,
            use_cache=False,
        )
        mem = out.memory_states  # (B, L, M, H)
        if mem is None:
            raise RuntimeError(
                "compute_memory_states: text_model did not return memory_states. "
                "Check that the model config has num_mem_token > 0 and that "
                "context_lengths is correct."
            )
        if self._memory_method == "only_full_1for1":
            fa_idx = torch.tensor(self._full_attn_layer_indices, device=mem.device)
            mem = mem[:, fa_idx, :, :]
        # LoraQwen3_5TextModel.forward builds memory_states via torch.zeros
        # (defaults to fp32) then writes per-layer slices. Cast back to the
        # hypernetwork's dtype so the m2p_transformer (bf16) doesn't barf
        # on dtype-mismatched inputs.
        if mem.dtype != self._dtype:
            mem = mem.to(self._dtype)
        return mem

    # ------------------------------------------------------------------
    # Step 2: hypernetwork forward → loradict
    # ------------------------------------------------------------------

    def generate_loradict(self, memory_states: torch.Tensor) -> Dict[int, dict]:
        """Run the hypernetwork on memory_states, then produce one
        ``layer_loradict`` per LLM layer (with ``None`` for linear_attention
        layers, matching ``LoraQwen3_5DecoderLayer.generate_lora_dict`` for
        non-full_attention layers).

        memory_states comes from ``compute_memory_states``. The output
        loradict is in the format expected by ``self.llm.model.forward(..., loradict=...)``.
        """
        m2p_out = self.hypernetwork(memory_states)  # (B, effective_L, effective_M, H)
        B = m2p_out.shape[0]
        effective_L = m2p_out.shape[1]

        # m2p_out's L-axis maps to full_attention layers in order. Each
        # full_attention layer i gets the slice m2p_out[:, fa_counter, :, :]
        # flattened to plain_tensor [B, effective_M * H].
        loradict: Dict[int, dict] = {}
        for fa_counter, layer_idx in enumerate(self._full_attn_layer_indices):
            plain_tensor = m2p_out[:, fa_counter, :, :].reshape(B, -1)
            loradict[layer_idx] = self.llm.model.layers[layer_idx].generate_lora_dict(
                self._lora_ranks, self._generate_lora_scale, plain_tensor,
            )

        # Fill linear_attention layers with None (same as PP) so that
        # direct indexing loradict[layer_idx] never raises KeyError.
        for layer_idx in range(self._num_llm_layers):
            if layer_idx not in loradict:
                loradict[layer_idx] = None

        return loradict

    # ------------------------------------------------------------------
    # Step 3: Step-4 forward (with loradict) + loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        loradict: Dict[int, dict],
        attention_mask: Optional[torch.Tensor] = None,
        return_per_token_loss: bool = False,
        nograd_loradict: Optional[Dict] = None,
        nograd_wdict: Optional[Dict] = None,
        return_acc: bool = False,
    ):
        """Step 4 — run the LLM with the generated loradict, then compute
        causal-LM CE loss against shifted labels.

        Args:
            return_per_token_loss: If False (default), returns a scalar mean
                loss — identical to the previous behaviour, zero extra cost.
                If True, returns a tuple (scalar_loss, per_token_loss) where
                per_token_loss is a float32 tensor of shape (B, S-1) with the
                per-position CE loss (0.0 at ignored positions). This path
                skips the Liger FLCE kernel and materialises the full logits,
                so it is slower and should only be used for debugging.
            nograd_loradict: Optional detached loradict from detach_state.
                Passed as nograd_loradict to the LLM forward (no gradient).
            nograd_wdict: Optional detached wdict from detach_state.
                Passed as nograd_wdict to the LLM forward (no gradient).
        """
        out = self.llm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loradict=loradict,
            nograd_loradict=nograd_loradict,
            nograd_wdict=nograd_wdict,
            use_cache=False,
        )
        last_hs = out.last_hidden_state
        shift_hs = last_hs[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        B, S_minus_1, H = shift_hs.shape
        flat_hs = shift_hs.reshape(B * S_minus_1, H)
        flat_labels = shift_labels.reshape(B * S_minus_1)

        # ------------------------------------------------------------------
        # Optional teacher-forced answer-token accuracy (eval only). Runs
        # lm_head ONLY on answer positions (labels != -100), so it is cheap
        # and low-memory. Stored on self as a side channel -> the compute_loss
        # return signature is unchanged (zero cost when return_acc=False).
        # NOTE: lm_head produces FULL-vocab logits on every (TP) rank (the
        # CE paths below rely on this too), so argmax is globally correct.
        # ------------------------------------------------------------------
        if return_acc:
            # no_grad: argmax accuracy must not extend the autograd graph (this
            # path also runs during training when train-acc logging is on).
            with torch.no_grad():
                keep = flat_labels != -100                      # (N,) answer-token mask
                keep_idx = keep.nonzero(as_tuple=True)[0]
                corr = torch.zeros_like(flat_labels, dtype=torch.bool)
                n_corr = 0
                for i in range(0, keep_idx.numel(), 1024):
                    idx = keep_idx[i:i + 1024]
                    pred = self.llm.lm_head(flat_hs[idx]).argmax(dim=-1)
                    c = pred == flat_labels[idx]
                    corr[idx] = c
                    n_corr += int(c.sum().item())
                # (1) token-level accuracy (lenient: per answer-token, incl. scaffolding)
                self._last_eval_acc_correct = n_corr
                self._last_eval_acc_total = int(keep_idx.numel())
                # (2) answer-level exact match (strict: a contiguous run of answer
                #     tokens counts only if EVERY token is argmax-correct -> getting
                #     just the easy scaffolding right scores 0). The -100 question
                #     tokens split the sequence into one run per answer.
                prev = torch.zeros_like(keep)
                prev[1:] = keep[:-1]
                run_id = torch.cumsum((keep & ~prev).long(), dim=0)   # 1..R at keep positions
                num_runs = int(run_id[keep].amax().item()) if keep.any() else 0
                wrong = keep & ~corr
                num_bad = int(torch.unique(run_id[wrong]).numel()) if wrong.any() else 0
                self._last_eval_ans_correct = num_runs - num_bad
                self._last_eval_ans_total = num_runs

        # ------------------------------------------------------------------
        # per-token loss path (debug only): materialise full logits, use
        # reduction='none' to get per-position loss, then compute mean.
        # ------------------------------------------------------------------
        if return_per_token_loss:
            chunk = max(1, 1024 // max(1, B))
            per_token_flat = torch.zeros(B * S_minus_1, device=flat_hs.device, dtype=torch.float32)
            total_loss = torch.zeros((), device=flat_hs.device, dtype=torch.float32)
            total_count = torch.zeros((), device=flat_hs.device, dtype=torch.long)
            for i in range(0, flat_hs.shape[0], chunk):
                j = min(i + chunk, flat_hs.shape[0])
                chunk_logits = self.llm.lm_head(flat_hs[i:j])
                chunk_loss_none = nn.functional.cross_entropy(
                    chunk_logits, flat_labels[i:j], ignore_index=-100, reduction="none",
                ).float()
                per_token_flat[i:j] = chunk_loss_none
                valid = (flat_labels[i:j] != -100)
                total_loss = total_loss + chunk_loss_none[valid].sum()
                total_count = total_count + valid.sum()
            scalar_loss = (total_loss / total_count.clamp(min=1)).to(last_hs.dtype)
            per_token_loss = per_token_flat.view(B, S_minus_1)  # (B, S-1)
            return scalar_loss, per_token_loss

        # ------------------------------------------------------------------
        # Normal training path (return_per_token_loss=False): unchanged.
        # ------------------------------------------------------------------
        # Liger fused linear+CE: single Triton kernel does
        # ``lm_head(hs) → log_softmax → NLL → backward`` without ever
        # materialising the [B*T, V=248k] logits tensor. Numerically
        # equivalent to the prior chunked CE at bf16 noise (verified;
        # mean rel diff ~1e-5 on loss, ~1e-6 on grad_x). Set SHINE_FLCE=0
        # to fall back to the chunked Python loop for A/B comparison.
        if os.environ.get("SHINE_FLCE", "1") not in ("0", "", "false"):
            from utils.liger_patch import fused_lm_head_loss
            loss = fused_lm_head_loss(
                flat_hs, self.llm.lm_head.weight, flat_labels,
                ignore_index=-100, reduction="mean",
            )
            return loss.to(last_hs.dtype)
        # Chunked-CE fallback (unchanged from pre-Liger path).
        chunk = max(1, 1024 // max(1, B))
        total_loss = torch.zeros((), device=flat_hs.device, dtype=torch.float32)
        total_count = torch.zeros((), device=flat_hs.device, dtype=torch.long)
        for i in range(0, flat_hs.shape[0], chunk):
            j = min(i + chunk, flat_hs.shape[0])
            chunk_hs = flat_hs[i:j]
            chunk_labels = flat_labels[i:j]
            chunk_logits = self.llm.lm_head(chunk_hs)
            chunk_loss = nn.functional.cross_entropy(
                chunk_logits, chunk_labels, ignore_index=-100, reduction="sum",
            )
            total_loss = total_loss + chunk_loss.float()
            total_count = total_count + (chunk_labels != -100).sum()
        loss = total_loss / total_count.clamp(min=1).to(total_loss.dtype)
        return loss.to(last_hs.dtype)

    # ------------------------------------------------------------------
    # Distillation helpers
    # ------------------------------------------------------------------

    def teacher_forward(
        self,
        input_ids: torch.LongTensor,
        mode: str = "logits",
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher forward: base LLM without lora, no_grad.

        Used for distillation Phase A'. The teacher is the frozen base LLM
        (no lora applied), producing reference outputs that the student
        (lora-augmented) should match.

        Args:
            input_ids: (B, S) distillation conversation token ids.
            mode: "logits" — return lm_head(hidden_states) as (B, S, V).
                  "hidden_states" — return raw hidden states as (B, S, H).
            attention_mask: Optional attention mask.

        Returns:
            Detached teacher output tensor (no grad).
        """
        with torch.no_grad():
            out = self.llm.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loradict=None,
                use_cache=False,
            )
            hidden = out.last_hidden_state
            if mode == "logits":
                return self.llm.lm_head(hidden).detach()
            else:
                return hidden.detach()

    def student_forward(
        self,
        input_ids: torch.LongTensor,
        loradict: Dict[int, dict],
        mode: str = "logits",
        attention_mask: Optional[torch.Tensor] = None,
        nograd_loradict: Optional[Dict] = None,
        nograd_wdict: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Student forward: LLM with loradict, with grad.

        Used for distillation Phase C'. The student uses the same loradict
        generated by the hypernetwork, so gradients flow back through
        loradict → hypernetwork → metalora.

        Args:
            input_ids: (B, S) distillation conversation token ids.
            loradict: The generated loradict (same as used in compute_loss).
            mode: "logits" — return lm_head(hidden_states) as (B, S, V).
                  "hidden_states" — return raw hidden states as (B, S, H).
            attention_mask: Optional attention mask.
            nograd_loradict: Optional detached loradict from detach_state.
            nograd_wdict: Optional detached wdict from detach_state.

        Returns:
            Student output tensor (with grad attached).
        """
        out = self.llm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loradict=loradict,
            nograd_loradict=nograd_loradict,
            nograd_wdict=nograd_wdict,
            use_cache=False,
        )
        hidden = out.last_hidden_state
        if mode == "logits":
            return self.llm.lm_head(hidden)
        else:
            return hidden

    # ------------------------------------------------------------------
    # Deferred DetachState initialization
    # ------------------------------------------------------------------

    def init_detach_state(self, *, local_batch_size: int, micro_batch_size: int,
                          tp_rank: int, tp_world: int, tp_process_group,
                          data_parallel_size: int = 1,
                          grad_accum_steps: int = 1) -> None:
        """Initialize DetachState with runtime parameters known only after
        model creation (batch_size, DP size, grad_accum_steps).

        Called from tp_main() after the model is built and batch config is
        resolved. If detach_state_cfg is None or type='empty', this creates
        an EmptyDetachState or None (zero overhead).

        Args:
            local_batch_size: Per-rank batch size (== micro_batch_size in TP).
            micro_batch_size: Micro-batch size (== local_batch_size in TP).
            tp_rank: This rank's position in the TP group.
            tp_world: TP group size.
            tp_process_group: The TP process group handle.
            data_parallel_size: Number of DP replicas.
            grad_accum_steps: Gradient accumulation steps (stored for
                compute_regu_loss scaling).
        """
        self._grad_accum_steps = grad_accum_steps
        if self._detach_state_cfg is None:
            self.detach_state = None
            return
        from hypernetwork.detach_state import create_detach_state
        self.detach_state = create_detach_state(
            cfg=self._detach_state_cfg,
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
        if is_main_process_per_node():
            logger.info(
                f"[TPModelHypernetwork] DetachState initialized: {self.detach_state}"
            )

    # ------------------------------------------------------------------
    # End-to-end forward (same steps as PP)
    # ------------------------------------------------------------------

    def forward(
        self,
        context_ids: torch.LongTensor,
        context_lengths: torch.LongTensor,
        conversation_ids: torch.LongTensor,
        labels: torch.LongTensor,
        context_attention_mask: Optional[torch.Tensor] = None,
        conv_attention_mask: Optional[torch.Tensor] = None,
        return_per_token_loss: bool = False,
        distill_loss_fn=None,
        distill_conversation_ids: Optional[torch.LongTensor] = None,
        distill_labels: Optional[torch.LongTensor] = None,
        grad_accum_steps: int = 1,
        return_acc: bool = False,
    ):
        """One forward of the multi-step pipeline.

        Step 0: read(0) → nograd_wdict (reset is done at end of previous step).
        Step 1 (with grad): context_ids + metalora + nograd → LLM → memory_states.
        Step 2 (with grad): memory_states → hypernetwork → loradict.
        Step 3: concat loradict + metalora → new_loradict.
        Step 4 (with grad): conversation_ids + new_loradict + nograd → LLM → loss.
        Step 4' (optional, distillation).
        Step 5: compute_regu_loss → (sq_norm, regu_loss_tensor, precomputed).
        Step 6: write(loradict, precomputed_wdict=precomputed).

        Returns:
            A tuple: (result, regu_sq_norm, regu_loss_out) where:
            - result: the loss result (same structure as before, depends on
              return_per_token_loss and distill_loss_fn). Does NOT include regu_loss.
            - regu_sq_norm: float, local (this TP rank's) ||W_old + A@B||².
              0.0 if detach_state is None/Empty or regu_c=0.
              Caller should all_reduce SUM across TP group for full norm.
            - regu_loss_out: differentiable regu_loss tensor or None.
              Caller should add to loss before backward (but NOT include in loss_val for logging).
        """
        # Step 0: read detach_state (reset is now done at end of previous step)
        ds_nograd_loradict, ds_nograd_wdict = self._read_detach_state()

        # Step 1: context_ids + metalora + nograd → LLM → memory_states
        # w_transform_context decides what happens to wdict (identity/zero/compressed_mlp)
        torch.cuda.nvtx.range_push("Step1_Context_Forward")
        self._active_w_transform = self.w_transform_context  # Phase A transform
        memory_states = self.compute_memory_states(
            context_ids, context_attention_mask, context_lengths,
            nograd_loradict=ds_nograd_loradict,
            nograd_wdict=ds_nograd_wdict,
        )
        self._active_w_transform = None
        torch.cuda.nvtx.range_pop()  # Step1_Context_Forward
        # Step 2: memory_states → hypernetwork → loradict
        torch.cuda.nvtx.range_push("Step2_Hypernetwork")
        loradict = self.generate_loradict(memory_states)
        torch.cuda.nvtx.range_pop()  # Step2_Hypernetwork
        # Step 3: concat loradict + metalora → new_loradict
        new_loradict = {}
        for layer_idx in range(self._num_llm_layers):
            layer_lora = loradict[layer_idx]
            layer_meta = self.metalora[layer_idx]
            new_loradict[layer_idx] = concat_loradict([layer_lora, layer_meta])
        # Step 4: conversation_ids + new_loradict + nograd → LLM → CE loss
        torch.cuda.nvtx.range_push("Step4_Conversation_Forward")
        self._active_w_transform = self.w_transform_conversation  # Phase C transform
        result = self.compute_loss(
            conversation_ids, labels, new_loradict, conv_attention_mask,
            return_per_token_loss=return_per_token_loss,
            nograd_loradict=ds_nograd_loradict,
            nograd_wdict=ds_nograd_wdict,
            return_acc=return_acc,
        )
        self._active_w_transform = None
        torch.cuda.nvtx.range_pop()  # Step4_Conversation_Forward
        # Step 4' (optional): distillation
        _distill_loss_val = None
        if distill_loss_fn is not None:
            torch.cuda.nvtx.range_push("Step4_Distillation")
            ce_loss = result[0] if return_per_token_loss else result
            teacher_output = self.teacher_forward(
                distill_conversation_ids, mode=distill_loss_fn.mode,
            )
            self._active_w_transform = self.w_transform_conversation  # Same as Phase C
            student_output = self.student_forward(
                distill_conversation_ids, new_loradict,
                mode=distill_loss_fn.mode,
                nograd_loradict=ds_nograd_loradict,
                nograd_wdict=ds_nograd_wdict,
            )
            self._active_w_transform = None
            distill_loss = distill_loss_fn(
                teacher_output, student_output, distill_labels,
            )
            _distill_loss_val = distill_loss.detach()
            total_loss = ce_loss + distill_loss
            if return_per_token_loss:
                result = (total_loss, result[1], _distill_loss_val)
            else:
                result = (total_loss, _distill_loss_val)
            torch.cuda.nvtx.range_pop()  # Step4_Distillation
        # Cache generated loradict for monitoring (gen_lora_norm)
        self._last_generated_loradict = loradict

        # Step 5: compute regu_loss (returned separately, NOT added to result)
        # Step 6: write is deferred to post_backward_detach_state (after backward)
        # NOTE: Use raw loradict (without metalora) for regu_loss and write,
        # matching PP mode which uses all_raw_loradicts[mb_idx].
        regu_sq_norm = 0.0
        regu_loss_out = None  # Returned separately; caller adds for backward only
        if self.detach_state is not None:
            _ga = grad_accum_steps if grad_accum_steps > 0 else getattr(self, '_grad_accum_steps', 1)
            unscaled_sq_norm, regu_loss_tensor, precomputed = self.detach_state.compute_regu_loss(
                loradict, mb_idx=0, num_mb=1, grad_accum_steps=_ga,
            )
            if unscaled_sq_norm is not None:
                regu_sq_norm = unscaled_sq_norm
            # Do NOT add regu_loss to result — return it separately so that
            # loss_val for logging is pure CE (+ distill), matching PP mode.
            regu_loss_out = regu_loss_tensor
            # Cache for post_backward write
            self._cached_precomputed_wdict = precomputed
            self._cached_loradict_for_write = loradict
        else:
            self._cached_precomputed_wdict = None
            self._cached_loradict_for_write = loradict

        return result, regu_sq_norm, regu_loss_out

    def post_backward_detach_state(self, grad_accum_steps: int = 1):
        """Called AFTER backward() to write loradict into detach_state.

        This must be called after backward because _write_detach_state does
        inplace copy_ on self._wdict["W"], which would conflict with the
        autograd graph if done before backward.

        Returns:
            regu_sq_norm: float (already computed in forward, returned here
                for API compatibility — always 0.0 since the real value was
                returned from forward).
        """
        loradict = getattr(self, '_cached_loradict_for_write', None)
        precomputed = getattr(self, '_cached_precomputed_wdict', None)
        self._cached_loradict_for_write = None
        self._cached_precomputed_wdict = None

        if self.detach_state is not None and loradict is not None:
            self._write_detach_state(loradict, mb_idx=0, precomputed_wdict=precomputed)

        return 0.0

    # ------------------------------------------------------------------
    # DetachState helpers
    # ------------------------------------------------------------------

    def _read_detach_state(self, mb_idx=None):
        """Read from detach_state. Returns (nograd_loradict, nograd_wdict).
        Both are None if detach_state is None or EmptyDetachState.
        """
        if self.detach_state is None:
            return None, None
        return self.detach_state.read(mb_idx=mb_idx)

    def _write_detach_state(self, loradict: Optional[Dict], mb_idx=None,
                            precomputed_wdict=None) -> None:
        """Write the generated loradict to detach_state for future use."""
        if self.detach_state is None:
            return
        self.detach_state.write(loradict, mb_idx=mb_idx,
                                precomputed_wdict=precomputed_wdict)

    # ------------------------------------------------------------------
    # Save / Load — PP-compatible format
    # ------------------------------------------------------------------

    def save_model(self, save_dir: str):
        """
        Save all trainable parameters (hypernetwork + metalora + mem_tokens)
        to the given directory in a format **compatible with the PP
        ModelHypernetwork checkpoint**.

        Uses safetensors format with the same key naming convention as PP:
            - ``hypernet.<name>`` for hypernetwork parameters
            - ``metalora.layer<N>.tensor<M>`` for metalora tensors
            - ``mem_tokens`` for the LLM's mem_tokens parameter

        The file is saved as ``model_stage0.safetensors`` (TP has no
        pipeline stages, so we always use stage 0).

        Args:
            save_dir: Directory path to save the model checkpoint.
        """
        from safetensors.torch import save_file

        os.makedirs(save_dir, exist_ok=True)

        # Collect all tensors with PP-compatible keys
        tensors_dict: dict = {}

        # --- Hypernetwork parameters ---
        # Access the underlying module if torch.compile wrapped it
        hypernet = self.hypernetwork
        if hasattr(hypernet, "_orig_mod"):
            hypernet = hypernet._orig_mod

        for name, param in hypernet.named_parameters():
            tensors_dict[f"hypernet.{name}"] = param.data.cpu()

        # --- Metalora tensors ---
        for layer_idx, layer_lora in self.metalora.items():
            tensors = collect_loradict_tensors(layer_lora)
            for t_idx, t in enumerate(tensors):
                tensors_dict[f"metalora.layer{layer_idx}.tensor{t_idx}"] = t.data.cpu()

        # --- Mem tokens ---
        if hasattr(self.llm.model, "mem_tokens") and self.llm.model.mem_tokens is not None:
            tensors_dict["mem_tokens"] = self.llm.model.mem_tokens.data.cpu()

        # --- W-Transform parameters ---
        for wt_name in ['w_transform_context', 'w_transform_conversation']:
            wt_module = getattr(self, wt_name, None)
            if wt_module is not None:
                for pname, param in wt_module.named_parameters():
                    tensors_dict[f"w_transform.{wt_name}.{pname}"] = param.data.cpu()

        # Save as model_stage0.safetensors (TP uses a single "stage")
        if tensors_dict:
            save_file(tensors_dict, os.path.join(save_dir, "model_stage0.safetensors"))

    def load_model(self, load_dir: str):
        """
        Load trainable parameters (hypernetwork + metalora + mem_tokens)
        from a PP-compatible model checkpoint.

        This method reads all ``model_stage*.safetensors`` files and loads:
          - ``hypernet.*`` keys → hypernetwork parameters
          - ``metalora.layer<N>.tensor<M>`` keys → metalora tensors
          - ``mem_tokens`` key → LLM mem_tokens parameter

        This ensures full compatibility between TP and PP checkpoints.

        Args:
            load_dir: Directory path containing the model checkpoint.
        """
        import re
        import glob
        from safetensors import safe_open

        st_files = sorted(glob.glob(os.path.join(load_dir, "model_stage*.safetensors")))
        if not st_files:
            raise FileNotFoundError(
                f"No model_stage*.safetensors files found in {load_dir}."
            )

        # Access the underlying module if torch.compile wrapped it
        hypernet = self.hypernetwork
        if hasattr(hypernet, "_orig_mod"):
            hypernet = hypernet._orig_mod

        # Build the set of keys this model needs
        param_dict = dict(hypernet.named_parameters())
        needed_keys = {f"hypernet.{name}" for name in param_dict.keys()}

        # Also need metalora and mem_tokens keys
        for layer_idx, layer_lora in self.metalora.items():
            tensors = collect_loradict_tensors(layer_lora)
            for t_idx in range(len(tensors)):
                needed_keys.add(f"metalora.layer{layer_idx}.tensor{t_idx}")
        if hasattr(self.llm.model, "mem_tokens") and self.llm.model.mem_tokens is not None:
            needed_keys.add("mem_tokens")

        # Also need w_transform keys (if modules exist)
        for wt_name in ['w_transform_context', 'w_transform_conversation']:
            wt_module = getattr(self, wt_name, None)
            if wt_module is not None:
                for pname in dict(wt_module.named_parameters()).keys():
                    needed_keys.add(f"w_transform.{wt_name}.{pname}")

        # Load from all stage files (PP may have saved across multiple stages)
        # PP checkpoints may contain `_orig_mod` in key names when layers
        # were wrapped by torch.compile. We normalise those keys so they
        # match the TP model's expected names (which never have _orig_mod).
        device_str = str(self._device)
        loaded_tensors: dict = {}
        all_checkpoint_keys: set = set()  # All normalised keys in checkpoint
        for f in st_files:
            with safe_open(f, framework="pt", device=device_str) as sf:
                for raw_key in sf.keys():
                    # Normalise: strip all `._orig_mod.` segments
                    normalised_key = raw_key.replace("._orig_mod.", ".")
                    all_checkpoint_keys.add(normalised_key)
                    if normalised_key in needed_keys:
                        loaded_tensors[normalised_key] = sf.get_tensor(raw_key)

        # Restore hypernetwork parameters
        loaded_hypernet = 0
        for key, tensor in loaded_tensors.items():
            if key.startswith("hypernet."):
                name = key[len("hypernet."):]
                if name in param_dict:
                    param_dict[name].data.copy_(tensor)
                    loaded_hypernet += 1

        # Restore metalora tensors
        loaded_metalora = 0
        metalora_by_layer: dict = {}
        for key, tensor in loaded_tensors.items():
            m = re.match(r"metalora\.layer(\d+)\.tensor(\d+)", key)
            if m:
                layer_idx = int(m.group(1))
                t_idx = int(m.group(2))
                if layer_idx not in metalora_by_layer:
                    metalora_by_layer[layer_idx] = {}
                metalora_by_layer[layer_idx][t_idx] = tensor

        for layer_idx, tensor_map in metalora_by_layer.items():
            if layer_idx in self.metalora:
                current_tensors = collect_loradict_tensors(self.metalora[layer_idx])
                for t_idx, ct in enumerate(current_tensors):
                    if t_idx in tensor_map:
                        ct.data.copy_(tensor_map[t_idx])
                        loaded_metalora += 1

        # Restore mem_tokens
        loaded_mem = 0
        if "mem_tokens" in loaded_tensors:
            if hasattr(self.llm.model, "mem_tokens") and self.llm.model.mem_tokens is not None:
                self.llm.model.mem_tokens.data.copy_(loaded_tensors["mem_tokens"])
                loaded_mem = 1

        # Restore W-Transform parameters (non-strict: skip if not in checkpoint)
        loaded_wt = 0
        expected_wt = 0
        for wt_name in ['w_transform_context', 'w_transform_conversation']:
            wt_module = getattr(self, wt_name, None)
            if wt_module is not None:
                prefix = f"w_transform.{wt_name}."
                wt_param_dict = dict(wt_module.named_parameters())
                expected_wt += len(wt_param_dict)
                for pname, param in wt_param_dict.items():
                    full_key = prefix + pname
                    # Also try normalised key (strip _orig_mod)
                    normalised_key = full_key.replace("._orig_mod.", ".")
                    if full_key in loaded_tensors:
                        param.data.copy_(loaded_tensors[full_key])
                        loaded_wt += 1
                    elif normalised_key in loaded_tensors:
                        param.data.copy_(loaded_tensors[normalised_key])
                        loaded_wt += 1
        # W-transform loading is strict: fail if checkpoint is missing w_transform params.
        missing_wt = expected_wt - loaded_wt

        # --- Strict check: fail loudly if any expected parameters are missing ---
        missing_hypernet = len(param_dict) - loaded_hypernet
        expected_metalora = sum(
            len(collect_loradict_tensors(self.metalora[li]))
            for li in self.metalora
        )
        missing_metalora = expected_metalora - loaded_metalora
        expected_mem = 1 if (hasattr(self.llm.model, "mem_tokens") and self.llm.model.mem_tokens is not None) else 0
        missing_mem = expected_mem - loaded_mem

        # Check for unexpected keys in checkpoint (keys not needed by model)
        unexpected_keys = all_checkpoint_keys - needed_keys
        # Filter out keys that are just _orig_mod variants of needed keys
        # (already handled by normalisation above)
        unexpected_keys = sorted(unexpected_keys)

        if missing_hypernet > 0 or missing_metalora > 0 or missing_mem > 0 or missing_wt > 0:
            # Collect details for the error message
            missing_details = []
            if missing_hypernet > 0:
                loaded_names = set()
                for key in loaded_tensors:
                    if key.startswith("hypernet."):
                        loaded_names.add(key[len("hypernet."):])
                unloaded = sorted(set(param_dict.keys()) - loaded_names)
                missing_details.append(
                    f"hypernet: {missing_hypernet}/{len(param_dict)} params NOT loaded. "
                    f"First 5 missing: {unloaded[:5]}"
                )
            if missing_metalora > 0:
                missing_details.append(
                    f"metalora: {missing_metalora}/{expected_metalora} tensors NOT loaded"
                )
            if missing_mem > 0:
                missing_details.append("mem_tokens: NOT loaded")
            if missing_wt > 0:
                missing_details.append(
                    f"w_transform: {missing_wt}/{expected_wt} params NOT loaded"
                )
            raise RuntimeError(
                f"[TPModelHypernetwork.load_model] STRICT LOAD FAILED — "
                f"some parameters were not found in checkpoint at {load_dir}.\n"
                + "\n".join(f"  • {d}" for d in missing_details)
                + "\n  Scanned files: " + ", ".join(os.path.basename(f) for f in st_files)
                + "\n  Hint: if the checkpoint was saved with torch.compile, ensure "
                "_orig_mod keys are normalised (this should be handled automatically)."
            )

        if unexpected_keys:
            raise RuntimeError(
                f"[TPModelHypernetwork.load_model] STRICT LOAD FAILED — "
                f"checkpoint contains {len(unexpected_keys)} unexpected key(s) "
                f"not needed by the current model.\n"
                f"  First 10 unexpected: {unexpected_keys[:10]}\n"
                f"  Scanned files: " + ", ".join(os.path.basename(f) for f in st_files)
                + "\n  Hint: the checkpoint may have been saved with a different model "
                "configuration (e.g. extra w_transform modules, different metalora layers)."
            )

        if is_main_process_per_node():
            logger.info(
                f"[TPModelHypernetwork.load_model] Loaded from {load_dir}: "
                f"hypernet={loaded_hypernet}/{len(param_dict)}, "
                f"metalora={loaded_metalora} tensors, "
                f"mem_tokens={'yes' if loaded_mem else 'no'}, "
                f"w_transform={loaded_wt}/{expected_wt} "
                f"(scanned {len(st_files)} stage file(s))"
            )
