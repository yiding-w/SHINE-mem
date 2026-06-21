import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import logging
from typing import Optional, Dict, List
from collections import defaultdict

from transformers import AutoConfig
from utils.mymodel import load_pretrained_llm_for_pipeline, load_m2p_transformer_for_pipeline, build_layer_stage_mapping, get_extra_component_stages, log_combined_device_map
from utils.myparallel import (
    get_pipeline_config,
    pipeline_recv,
    pipeline_send,
    PipelineRecv,
    pipeline_send_with_grad,
    is_main_process_per_node,
)
logger = logging.getLogger(__name__)

from src_transformers_lora.LoraHelper import SUPPORTED_MODELS, compute_layer_lora_params_numel  # noqa: E402

from hypernetwork.hypernetwork import Hypernetwork  # noqa: E402


class ModelHypernetwork(nn.Module):
    """
    Hypernetwork wrapper that holds a pretrained LLM backbone.

    The LLM is loaded with pipeline parallelism (8 GPUs intra-node) and
    data parallelism (inter-node).  By default the LLM weights are frozen;
    only the hypernetwork parameters (to be added later) are trainable.

    Pipeline-parallel forward:
        Each process only holds a subset of the LLM layers (determined by
        the device_map config).  During forward, stage 0 runs embedding +
        its layers, sends hidden_states to stage 1, stage 1 runs its layers
        and sends to stage 2, etc.  The last stage runs the final norm and
        lm_head, then broadcasts the result back.

    Prerequisites:
        Before constructing this module, the training script must have called:
          1. ``init_distributed()``
          2. ``setup_pipeline_parallel(total_gpus=8, pipeline_parallel_size=8)``
        so that the global pipeline config is available.  If not, a
        RuntimeError will be raised during LLM loading.

    Supported models (register new ones in ``SUPPORTED_MODELS``):
        - ``Qwen3_6-35B-A3B``  → qwen3_5moe family
        - ``Qwen3_6-27B``  → qwen3_5 family
        - ``Qwen3-30B-A3B-Instruct-2507`` → qwen3moe family

    Args:
        model_cfg: Hydra DictConfig with at least ``name`` (str) and ``path``
            (str) pointing to the pretrained model directory.  Contains
            ``device_map`` for pipeline-parallel layer placement.
            Must also contain ``lora_ranks`` dict mapping component names
            to their LoRA ranks.
        m2p_transformer_cfg: Hydra DictConfig for the m2p_transformer, with
            ``init`` (TransformerModel kwargs) and ``device_map`` keys.
    """

    def __init__(
        self,
        model_cfg,
        m2p_transformer_cfg,
        training_cfg=None,
        debug_anchor: bool = True,
        compile_mode: Optional[str] = None,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self._debug_anchor = debug_anchor
        self._compile_mode = compile_mode
        self._activation_checkpointing = activation_checkpointing
        self._training_cfg = training_cfg

        # ---- Step 0: Validate model name and determine model family ----
        model_name = str(model_cfg.name)
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model name '{model_name}'. "
                f"Supported models: {list(SUPPORTED_MODELS.keys())}. "
                f"If you are adding a new model, please register it in "
                f"SUPPORTED_MODELS in model_hypernetwork.py and implement "
                f"the corresponding _prepare_inputs_* and _run_layer_* methods."
            )
        self._model_name = model_name
        self._model_family = SUPPORTED_MODELS[model_name]

        if is_main_process_per_node():
            logger.info(
                f"[ModelHypernetwork] Model name='{model_name}', "
                f"family='{self._model_family}', "
                f"activation_checkpointing={self._activation_checkpointing}"
            )

        # ---- Step 1: Load HF config and compute lora_params_numel arithmetically ----
        from omegaconf import OmegaConf
        import os
        from hydra.utils import get_original_cwd

        model_path = str(model_cfg.path)
        if not os.path.isabs(model_path):
            model_path = os.path.join(get_original_cwd(), model_path)

        # Load the HuggingFace config from the pretrained model directory
        llm_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # For nested configs (e.g. Qwen3_5MoeConfig which has text_config / vision_config),
        # the ForCausalLM class expects the text sub-config, not the top-level config.
        if hasattr(llm_config, "text_config") and llm_config.text_config is not None:
            text_config = llm_config.text_config
        else:
            text_config = llm_config

        # ---- Read per-component LoRA ranks from model config ----
        from omegaconf import OmegaConf as _OC
        lora_ranks_cfg = model_cfg.get("lora_ranks", None)
        if lora_ranks_cfg is None:
            raise ValueError(
                "model_cfg must contain 'lora_ranks' dict mapping component "
                "names to their LoRA ranks, e.g. {q_query: 16, k_proj: 8, ...}"
            )
        # Convert OmegaConf DictConfig to plain dict[str, int]
        lora_ranks: dict = _OC.to_container(lora_ranks_cfg, resolve=True)
        if is_main_process_per_node():
            logger.info(f"[ModelHypernetwork] lora_ranks = {lora_ranks}")

        # Compute per-layer lora_params_numel directly from config values (no model needed).
        # This is much faster than instantiating a full model on meta device.
        # We must find a layer that actually carries LoRA (e.g. full_attention),
        # since linear_attention layers have 0 LoRA params.
        num_layers = getattr(text_config, "num_hidden_layers", 40)
        layer_lora_numel = 0
        lora_layer_idx = -1
        for _li in range(num_layers):
            layer_lora_numel = compute_layer_lora_params_numel(text_config, lora_ranks, layer_idx=_li)
            if layer_lora_numel > 0:
                lora_layer_idx = _li
                break
        hidden_size = text_config.hidden_size

        # Read memory_method from m2p_transformer config
        from omegaconf import OmegaConf as _OC_m2p
        if hasattr(m2p_transformer_cfg, "_metadata"):
            _m2p_full_cfg = _OC_m2p.to_container(m2p_transformer_cfg, resolve=True)
        elif isinstance(m2p_transformer_cfg, dict):
            _m2p_full_cfg = dict(m2p_transformer_cfg)
        else:
            _m2p_full_cfg = dict(m2p_transformer_cfg)
        self._memory_method = _m2p_full_cfg.get("memory_method", "only_full_4for1")
        if self._memory_method not in ("only_full_4for1", "only_full_1for1"):
            raise ValueError(
                f"Unsupported memory_method '{self._memory_method}'. "
                f"Supported: 'only_full_4for1', 'only_full_1for1'."
            )
        if is_main_process_per_node():
            logger.info(f"[ModelHypernetwork] memory_method = '{self._memory_method}'")

        if self._memory_method == "only_full_4for1":
            gen_layer_lora_numel = layer_lora_numel // 4
        elif self._memory_method == "only_full_1for1":
            gen_layer_lora_numel = layer_lora_numel
        # num_mem_token must be exactly divisible
        if gen_layer_lora_numel % hidden_size != 0:
            if is_main_process_per_node():
                # Only the main GPU per node prints the detailed breakdown
                _, breakdown = compute_layer_lora_params_numel(
                    text_config, lora_ranks, layer_idx=max(lora_layer_idx, 0), verbose=True,
                )
                logger.error(
                    f"lora_params_numel ({gen_layer_lora_numel}) is not divisible by "
                    f"hidden_size ({hidden_size}). Cannot compute num_mem_token.\n"
                    f"Remainder: {gen_layer_lora_numel} % {hidden_size} = {gen_layer_lora_numel % hidden_size}\n"
                    f"LoRA param breakdown for layer {lora_layer_idx} (lora_ranks={lora_ranks}):\n"
                    f"{breakdown}\n"
                    f"To fix: adjust lora_ranks so that the total is divisible by hidden_size, "
                    f"or pad the LoRA parameter count."
                )
            # All ranks raise the same short error so they all exit consistently
            raise ValueError(
                f"lora_params_numel ({gen_layer_lora_numel}) is not divisible by "
                f"hidden_size ({hidden_size}). Cannot compute num_mem_token."
            )
        num_mem_token = gen_layer_lora_numel // hidden_size

        if is_main_process_per_node():
            # Always show the detailed breakdown for transparency
            _, breakdown = compute_layer_lora_params_numel(
                text_config, lora_ranks, layer_idx=max(lora_layer_idx, 0), verbose=True,
            )
            logger.info(
                f"[ModelHypernetwork] Computed num_mem_token = {num_mem_token} "
                f"(lora_params_numel={gen_layer_lora_numel}, hidden_size={hidden_size}, "
                f"lora_ranks={lora_ranks}, "
                f"Only add LoRA for full_attention layers (attention + experts). "
                f"linear_attention layers have no LoRA.)\n"
                f"LoRA param breakdown for layer {lora_layer_idx}:\n"
                f"{breakdown}"
            )

        # ---- Step 2: Update the config with num_mem_token ----
        text_config.num_mem_token = num_mem_token

        # ---- Step 3: Load the real LLM with the updated config ----
        # For nested configs (e.g. Qwen3_5MoeConfig), the ForCausalLM class
        # expects the text sub-config (which has hidden_size, vocab_size, etc.),
        # not the top-level multimodal config.
        self.llm = load_pretrained_llm_for_pipeline(
            model_cfg=model_cfg,
            dtype=torch.bfloat16,
            freeze=True,
            use_lora_class=True,
            compile_mode=self._compile_mode,
            config=text_config,
        )

        # ---- Step 3b: Initialize LoRA generate_func on all layers ----
        # Each LoRA linear layer needs a generate_func to convert plain
        # tensors into LoRA A/B/C matrices.
        lora_method = getattr(m2p_transformer_cfg, "lora_method", "rl")
        self.llm.set_generate_func(lora_method)

        # ---- Visualize combined device map for LLM + m2p_transformer ----
        if is_main_process_per_node():
            from omegaconf import OmegaConf

            viz_models = []

            # LLM device map
            llm_dm_raw = getattr(model_cfg, "device_map", None)
            if llm_dm_raw is not None:
                llm_raw = (
                    OmegaConf.to_container(llm_dm_raw, resolve=True)
                    if hasattr(llm_dm_raw, "_metadata")
                    else dict(llm_dm_raw) if not isinstance(llm_dm_raw, dict) else llm_dm_raw
                )
                from utils.mymodel import expand_device_map
                llm_full = expand_device_map(llm_raw) if "layer_prefix" in llm_raw else llm_raw
                # Re-add dynamic entries that load_pretrained_llm adds at runtime
                if "model.rotary_emb" not in llm_full:
                    llm_full["model.rotary_emb"] = llm_full.get("model.embed_tokens", 0)
                if hasattr(text_config, "num_mem_token") and text_config.num_mem_token > 0:
                    if "model.mem_tokens" not in llm_full:
                        llm_full["model.mem_tokens"] = llm_full.get("model.embed_tokens", 0)
                llm_compact = llm_raw if "layer_prefix" in llm_raw else None
                llm_label = f"{getattr(model_cfg, 'name', 'LLM')} (LLM)"
                viz_models.append((llm_label, llm_full, llm_compact))

            # m2p_transformer device map
            m2p_dm_raw = getattr(m2p_transformer_cfg, "device_map", None)
            if m2p_dm_raw is None and hasattr(m2p_transformer_cfg, "get"):
                m2p_dm_raw = m2p_transformer_cfg.get("device_map", None)
            if m2p_dm_raw is not None:
                m2p_raw = (
                    OmegaConf.to_container(m2p_dm_raw, resolve=True)
                    if hasattr(m2p_dm_raw, "_metadata")
                    else dict(m2p_dm_raw) if not isinstance(m2p_dm_raw, dict) else m2p_dm_raw
                )
                m2p_full = {}
                for entry in m2p_raw.get("gpu_map", []):
                    gpu_id, start, end = int(entry[0]), int(entry[1]), int(entry[2])
                    prefix = m2p_raw.get("layer_prefix", "layers")
                    for idx in range(start, end + 1):
                        m2p_full[f"{prefix}.{idx}"] = gpu_id
                for k, v in m2p_raw.get("extra", {}).items():
                    m2p_full[k] = int(v)
                m2p_compact = m2p_raw if "layer_prefix" in m2p_raw else None
                viz_models.append(("m2p_transformer", m2p_full, m2p_compact))

            if viz_models:
                log_combined_device_map(viz_models)

        # ---- Step 4: Precompute pipeline metadata for forward functions ----
        self._build_pipeline_metadata(model_cfg, m2p_transformer_cfg)

        # ---- Step 5: Cache text config and dtype for forward use ----
        cfg = self.llm.config
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            self._text_config = cfg.text_config
        else:
            self._text_config = cfg

        # Cache the model dtype so we don't call next(parameters()) every forward
        self._dtype = next(self.llm.parameters()).dtype

        # ---- Step 5b: Ensure mem_tokens dtype matches LLM dtype ----
        # mem_tokens is created in __init__ with default float32 dtype and is
        # not in the pretrained checkpoint, so from_pretrained (with device_map)
        # does not convert it.  Explicitly cast it to the LLM dtype here.
        if hasattr(self._llm_model, "mem_tokens") and self._llm_model.mem_tokens is not None:
            # transformers 5.x's from_pretrained(device_map=...) sometimes
            # leaves missing parameters on `meta` with a *wrong* shape — for
            # mem_tokens specifically we have observed shape
            # (head_dim*n_heads*2, hidden) instead of (num_mem_token, hidden)
            # on the meta device. Detect this and re-init at the expected
            # shape on this stage's real device.
            expected_shape = (self._llm_model.num_mem_token, self._text_config.hidden_size)
            mt = self._llm_model.mem_tokens
            if mt.device.type == "meta" or tuple(mt.shape) != expected_shape:
                if is_main_process_per_node():
                    logger.warning(
                        f"[mem_tokens] from_pretrained produced shape={tuple(mt.shape)} "
                        f"device={mt.device}; re-materializing to {expected_shape} on {self._my_device}"
                    )
                self._llm_model.mem_tokens = nn.Parameter(
                    torch.zeros(
                        expected_shape,
                        dtype=self._dtype,
                        device=self._my_device,
                    ),
                    requires_grad=True,
                )
            else:
                self._llm_model.mem_tokens.data = self._llm_model.mem_tokens.data.to(self._dtype)

        # ---- Step 6: Ensure LLM rotary_emb is available on this stage ----
        # rotary_emb is placed on the embed stage (GPU 0) by the device map.
        # Other stages get a meta-device placeholder.  Since rotary_emb is
        # tiny (just an inv_freq buffer), we re-create it on every stage
        # that doesn't own it so position embeddings can be computed locally
        # without cross-GPU communication.
        rotary = self._llm_rotary_emb
        inv_freq = getattr(rotary, "inv_freq", None)
        if inv_freq is not None and inv_freq.device.type == "meta":
            # Re-create the full rotary_emb module on the local device.
            # This is safe because rotary_emb is deterministic (computed
            # from config) and has no learned parameters.
            if self._model_family == "qwen3_5moe":
                from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                    Qwen3_5MoeTextRotaryEmbedding,
                )
                self._llm_model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(
                    config=self._text_config, device=self._my_device,
                )
            elif self._model_family == "qwen3_5":
                from transformers.models.qwen3_5.modeling_qwen3_5 import (
                    Qwen3_5TextRotaryEmbedding,
                )
                self._llm_model.rotary_emb = Qwen3_5TextRotaryEmbedding(
                    config=self._text_config, device=self._my_device,
                )
            elif self._model_family == "qwen3moe":
                from transformers.models.qwen3_moe.modeling_qwen3_moe import (
                    Qwen3MoeRotaryEmbedding,
                )
                self._llm_model.rotary_emb = Qwen3MoeRotaryEmbedding(
                    config=self._text_config, device=self._my_device,
                )
            if is_main_process_per_node():
                logger.info(
                    f"[ModelHypernetwork] Re-created rotary_emb on stage "
                    f"{self._my_stage} (device={self._my_device})"
                )

        # Bind model-family-specific methods at init time so there is
        # zero branching overhead during forward.
        if self._model_family == "qwen3_5moe":
            self._prepare_inputs = self._prepare_inputs_qwen3_5moe
            self._run_layer = self._run_layer_qwen3_5moe
        elif self._model_family == "qwen3_5":
            # Qwen3_5 (non-MoE) shares the same interface as Qwen3_5Moe
            # (layer_types, position_embeddings, linear_attn_mask, etc.)
            self._prepare_inputs = self._prepare_inputs_qwen3_5moe
            self._run_layer = self._run_layer_qwen3_5moe
        elif self._model_family == "qwen3moe":
            self._prepare_inputs = self._prepare_inputs_qwen3moe
            self._run_layer = self._run_layer_qwen3moe
        # (no else — already validated above)

        # ---- Step 7: Create Hypernetwork (self-contained) ----
        # Hypernetwork loads m2p_transformer, computes pipeline metadata,
        # parses layer_types, and builds per-layer forward functions internally.
        # Only external LLM-derived values are passed in.
        # Compute num_full_attn_layers from text_config for only_full_1for1 method.
        _llm_layer_types_pre = getattr(text_config, 'layer_types', None) or []
        self._num_full_attn_layers = sum(1 for lt in _llm_layer_types_pre if lt == "full_attention")
        self.hypernetwork = Hypernetwork(
            m2p_transformer_cfg=m2p_transformer_cfg,
            num_llm_layers=self._num_layers,
            num_mem_token=num_mem_token,
            dtype=self._dtype,
            compile_mode=self._compile_mode,
            memory_method=self._memory_method,
            num_full_attn_layers=self._num_full_attn_layers if self._memory_method == "only_full_1for1" else None,
            activation_checkpointing=self._activation_checkpointing,
        )

        # ---- LoRA generation metadata ----
        self._lora_ranks = lora_ranks
        # init_lora_var: variance for metalora random initialization
        #   A = randn(...) * sqrt(init_lora_var)
        self._init_lora_var = float(m2p_transformer_cfg.get("init_lora_var", 1.0))
        # generate_lora_scale: scale applied to hypernetwork-generated LoRA
        #   A = plain_tensor[...] * sqrt(generate_lora_scale)
        self._generate_lora_scale = float(m2p_transformer_cfg.get("generate_lora_scale", 1.0))

        # Precompute full_attention layer indices and per-stage distribution.
        # memory_states from hypernetwork has shape:
        #   only_full_4for1: (B, L/4, M*4, H) where L/4 = number of full_attention layers.
        #   only_full_1for1: (B, L_fa, M, H) where L_fa = number of full_attention layers.
        # We need to know which full_attention layers live on which pipeline
        # stage so we can scatter the right slices to the right GPUs.
        llm_layer_types = getattr(text_config, 'layer_types', None)
        self._llm_layer_types = llm_layer_types or []
        self._full_attn_layer_indices: List[int] = []
        if llm_layer_types is not None:
            for i, lt in enumerate(llm_layer_types):
                if lt == "full_attention":
                    self._full_attn_layer_indices.append(i)

        # Build per-stage mapping: stage → list of (full_attn_counter, layer_idx)
        # full_attn_counter is the index into the L/4 dimension of memory_states.
        self._stage_full_attn_info: Dict[int, List[tuple]] = defaultdict(list)
        for fa_counter, layer_idx in enumerate(self._full_attn_layer_indices):
            stage = self._layer_to_stage[layer_idx]
            self._stage_full_attn_info[stage].append((fa_counter, layer_idx))

        if is_main_process_per_node():
            logger.info(
                f"[ModelHypernetwork] LoRA scatter info: "
                f"lora_ranks={lora_ranks}, init_lora_var={self._init_lora_var}, "
                f"generate_lora_scale={self._generate_lora_scale}, "
                f"full_attn_layers={self._full_attn_layer_indices}, "
                f"stage_full_attn_info={dict(self._stage_full_attn_info)}"
            )
        self._m2p_hidden_size = self.hypernetwork._hidden_size
        self._m2p_num_mem_token = num_mem_token
        if is_main_process_per_node():
            _eff_L = self.hypernetwork._effective_L
            _eff_M = self.hypernetwork._effective_M
            logger.info(
                f"[ModelHypernetwork] Created Hypernetwork module with "
                f"m2p_transformer (memory_method='{self._memory_method}'), "
                f"layer_pos_emb=({_eff_L}, 1, {self._m2p_hidden_size}), "
                f"token_pos_emb=(1, {_eff_M}, {self._m2p_hidden_size}), "
                f"layer_types={self.hypernetwork._layer_types}"
            )

        # ---- Step 8: Initialize metalora (trainable LoRA via init_lora_dict) ----
        # Each GPU only initializes metalora for the layers it owns, matching
        # the pipeline-parallel layer placement of the LLM.
        metalora_ranks_cfg = model_cfg.get("metalora_ranks", None)
        if metalora_ranks_cfg is None:
            raise ValueError(
                "model_cfg must contain 'metalora_ranks' dict mapping component "
                "names to their LoRA ranks for the metalora initialization."
            )
        metalora_ranks: dict = _OC.to_container(metalora_ranks_cfg, resolve=True)
        self._metalora_ranks = metalora_ranks
        self.metalora: Dict[int, dict] = {}
        for layer_idx in self._my_layer_indices:
            self.metalora[layer_idx] = self._llm_layers[layer_idx].init_lora_dict(
                metalora_ranks, self._init_lora_var, self._my_device, torch.bfloat16
            )
        if is_main_process_per_node():
            logger.info(
                f"[ModelHypernetwork] Initialized metalora with "
                f"metalora_ranks={metalora_ranks}, dtype=bf16, "
                f"device={self._my_device}, "
                f"layers_on_this_stage={self._my_layer_indices}"
            )

        # ----------------------------------------------------------------
        # Initialize DetachState (bound 1:1 with this ModelHypernetwork).
        # DetachState maintains a persistent no-grad state that accumulates
        # the effect of generated LoRA dicts over time.
        # Each PP stage only needs state for the layers it owns.
        # ----------------------------------------------------------------
        from hypernetwork.detach_state import create_detach_state
        from utils.myparallel import get_pipeline_config
        detach_state_cfg = model_cfg.get("detach_state", None)
        if detach_state_cfg is not None:
            _pp_cfg = get_pipeline_config()
            _pp_batch_cfg = (self._training_cfg or {}).get("pp_batchsize", {})
            self.detach_state = create_detach_state(
                cfg=detach_state_cfg,
                mode="pp",
                local_batch_size=int(_pp_batch_cfg.get("local_batch_size", 1)),
                micro_batch_size=int(_pp_batch_cfg.get("local_micro_batch_size", 1)),
                parallel_mode="pp",
                total_stages=_pp_cfg["total_stages"],
                data_parallel_size=_pp_cfg["data_parallel_size"],
                total_gpus_per_node=_pp_cfg.get("total_gpus", 8),
                my_stage=_pp_cfg["stage"],
                my_layer_indices=self._my_layer_indices,
                num_llm_layers=self._num_layers,
            )
        else:
            self.detach_state = None

        # DetachState helper methods (defined here for proximity to init)
        # Actual method bodies are defined later in the class.

        # Initialize backward anchors list (also reset in invalidate_cache)
        self._backward_anchors = []
        # Training-time temporary state for pipeline backward communication.
        # These are set during forward and consumed during backward.
        self._lora_scatter_memory_states = None
        self._lora_scatter_dst_info = None
        self._mem_gather_local_memory = None
        self._mem_gather_src_info = None
        self._mem_gather_assembled = None
        # Input caches (also reset in invalidate_input_cache)
        self._cached_inputs_qwen3_5moe = None
        self._cached_inputs_qwen3moe = None

        # Cache pipeline total stages for debug gather
        parallel_cfg = get_pipeline_config()
        self._pp_total_stages = parallel_cfg["total_stages"]

    # ------------------------------------------------------------------
    # Input cache management
    # ------------------------------------------------------------------

    def invalidate_input_cache(self):
        """
        Clear the cached causal_mask / position_embeddings / position_ids.

        The cache is keyed on (batch_id, use_mem_token) where batch_id is a
        unique identifier for each batch of data.  It is automatically
        invalidated when a new batch_id is seen.  Call this explicitly if
        you want to force recomputation.
        """
        self._cached_inputs_qwen3_5moe = None
        self._cached_inputs_qwen3moe = None
        self._backward_anchors = []  # saved by with_grad functions for pipeline_backward()
        self._lora_scatter_memory_states = None  # saved by _scatter_and_generate_lora_dict_with_grad
        self._lora_scatter_dst_info = None
        # Saved by _gather_memory_states_with_grad for explicit backward
        self._mem_gather_local_memory = None   # memory_states tensor on each LLM stage
        self._mem_gather_src_info = None        # [(src_stage, src_layers), ...] on target stage
        self._mem_gather_assembled = None       # assembled memory_states on target stage (with retain_grad)

    # ------------------------------------------------------------------
    # Debug: gather anchor info from all stages to main GPU
    # ------------------------------------------------------------------

    def _gather_and_log_anchors(
        self, caller: str, extra_info: str = "",
    ):
        """
        Collect backward-anchor debug info from **every** pipeline stage
        **within this node** and print them on the node-local main process
        (local_rank 0) in a single, consolidated block.

        All stages on this node must call this collectively (it contains a
        ``dist.gather_object`` on the intra-node process group).

        Args:
            caller: Name of the calling function (for the log header).
            extra_info: Optional extra string appended to the header line.
        """
        # --- 1. Build local anchor summary (every stage) ---
        lines: List[str] = []
        header = (
            f"stage={self._my_stage}, "
            f"num_anchors={len(self._backward_anchors)}"
        )
        if extra_info:
            header += f", {extra_info}"
        lines.append(header)

        for i, anchor in enumerate(self._backward_anchors):
            src = getattr(anchor, '_anchor_source', 'UNKNOWN')
            has_grad = anchor.requires_grad and anchor.grad_fn is not None
            tag = "" if has_grad else "  ⚠ NO GRAD"
            lines.append(
                f"  anchor[{i}]: source={src}, "
                f"shape={list(anchor.shape)}, dtype={anchor.dtype}, "
                f"device={anchor.device}, "
                f"requires_grad={anchor.requires_grad}, "
                f"grad_fn={type(anchor.grad_fn).__name__ if anchor.grad_fn else None}"
                f"{tag}"
            )

        local_info = {
            "stage": self._my_stage,
            "lines": lines,
        }

        # --- 2. Gather within this node only ---
        parallel_cfg = get_pipeline_config()
        node_group = parallel_cfg.get("node_process_group")
        total_gpus = parallel_cfg.get("total_gpus", self._pp_total_stages)

        if dist.is_initialized() and node_group is not None:
            # local_rank 0 is rank 0 inside the node group
            # dst must be a global rank; convert group-rank 0 → global rank
            dst_global = dist.get_global_rank(node_group, 0)
            if is_main_process_per_node():
                gathered: List[Optional[dict]] = [None] * total_gpus
            else:
                gathered = None
            dist.gather_object(local_info, gathered, dst=dst_global, group=node_group)
        else:
            # Single-GPU fallback
            gathered = [local_info]

        # --- 3. Print on main process ---
        if is_main_process_per_node() and gathered is not None:
            logger.info(f"  [{caller}] ===== Anchor Debug (all stages) =====")
            # Sort by stage index for deterministic output
            gathered_sorted = sorted(
                [g for g in gathered if g is not None],
                key=lambda g: g["stage"],
            )
            for g in gathered_sorted:
                for line in g["lines"]:
                    logger.info(f"    [GPU {g['stage']}] {line}")
            logger.info(f"  [{caller}] ===== End Anchor Debug =====")

    # ------------------------------------------------------------------
    # Pipeline metadata
    # ------------------------------------------------------------------

    def _build_pipeline_metadata(self, model_cfg, m2p_transformer_cfg=None):
        """
        Precompute layer-to-stage mapping and stage transition info so that
        forward functions can route hidden_states across arbitrary GPU
        distributions without recomputing every call.

        Populates:
            self._layer_to_stage   : Dict[int, int]  — layer_idx → pipeline stage
            self._extra_stages     : Dict[str, int]   — component name → stage
            self._my_stage         : int
            self._my_device        : torch.device
            self._num_layers       : int
            self._hidden_size      : int
            self._my_layer_indices : List[int]  — sorted layer indices on this stage
            self._stage_transitions: List[Tuple[int, int, int]]
                Each entry is (layer_idx, from_stage, to_stage) indicating that
                after running layer_idx on from_stage, hidden_states must be sent
                to to_stage for the next layer.
            self._mem_gather_target_stage : int  — the stage that owns the first
                layer of the hypernetwork (m2p_transformer), where memory_states
                are gathered to.
        """
        device_map_cfg = getattr(model_cfg, "device_map", None)
        if device_map_cfg is None:
            raise RuntimeError(
                "model_cfg must contain a 'device_map' key for pipeline-parallel forward."
            )

        parallel_cfg = get_pipeline_config()
        self._my_stage = parallel_cfg["stage"]
        self._my_device = parallel_cfg["device"]

        self._layer_to_stage = build_layer_stage_mapping(device_map_cfg)
        self._extra_stages = get_extra_component_stages(device_map_cfg)
        self._num_layers = max(self._layer_to_stage.keys()) + 1
        self._my_layer_indices = sorted(
            [idx for idx, s in self._layer_to_stage.items() if s == self._my_stage]
        )

        # Get hidden_size from the loaded LLM config
        llm_config = self.llm.config
        if hasattr(llm_config, "text_config") and llm_config.text_config is not None:
            self._hidden_size = llm_config.text_config.hidden_size
        else:
            self._hidden_size = llm_config.hidden_size

        # Precompute stage transitions: after layer i, if layer i+1 is on a
        # different stage, we need a send/recv.
        self._stage_transitions = []
        for i in range(self._num_layers - 1):
            src = self._layer_to_stage[i]
            dst = self._layer_to_stage[i + 1]
            if src != dst:
                self._stage_transitions.append((i, src, dst))

        # Determine which stage owns embed_tokens, norm, lm_head
        self._embed_stage = self._extra_stages.get(
            "model.embed_tokens", self._layer_to_stage[0]
        )
        self._norm_stage = self._extra_stages.get(
            "model.norm", self._layer_to_stage[self._num_layers - 1]
        )
        self._lm_head_stage = self._extra_stages.get(
            "lm_head", self._layer_to_stage[self._num_layers - 1]
        )

        # Build reverse mapping: stage → sorted list of layer indices on that stage.
        # Used by _gather_memory_states to know which stages hold memory data.
        stage_to_layers: Dict[int, List[int]] = defaultdict(list)
        for idx, s in self._layer_to_stage.items():
            stage_to_layers[s].append(idx)
        self._stage_to_layers = {s: sorted(v) for s, v in stage_to_layers.items()}

        # Determine the target stage for memory_states gathering:
        # This is the stage that owns the first layer of the hypernetwork
        # (m2p_transformer), so memory_states are ready for the hypernetwork
        # forward pass without additional cross-GPU transfers.
        if m2p_transformer_cfg is not None:
            from omegaconf import OmegaConf
            if hasattr(m2p_transformer_cfg, "_metadata"):
                m2p_full_cfg = OmegaConf.to_container(m2p_transformer_cfg, resolve=True)
            elif isinstance(m2p_transformer_cfg, dict):
                m2p_full_cfg = dict(m2p_transformer_cfg)
            else:
                m2p_full_cfg = dict(m2p_transformer_cfg)
            m2p_device_map = m2p_full_cfg.get("device_map", None)
            if m2p_device_map is not None:
                m2p_layer_to_stage = build_layer_stage_mapping(m2p_device_map)
                # First layer of the hypernetwork
                self._mem_gather_target_stage = m2p_layer_to_stage[min(m2p_layer_to_stage.keys())]
            else:
                # Fallback: if no device_map for m2p_transformer, use norm stage
                self._mem_gather_target_stage = self._norm_stage
        else:
            self._mem_gather_target_stage = self._norm_stage

        # Stages that own LLM layers but are NOT the mem_gather_target_stage
        # — these need to send their memory_states slices to the target stage.
        self._mem_gather_stages = sorted(
            s for s in self._stage_to_layers if s != self._mem_gather_target_stage
        )

        if is_main_process_per_node():
            logger.info(
                f"[ModelHypernetwork] Pipeline metadata: "
                f"stage={self._my_stage}, layers={self._my_layer_indices}, "
                f"embed@{self._embed_stage}, norm@{self._norm_stage}, "
                f"lm_head@{self._lm_head_stage}, "
                f"mem_gather_target@{self._mem_gather_target_stage}, "
                f"transitions={self._stage_transitions}, "
                f"mem_gather_stages={self._mem_gather_stages}"
            )

    # ------------------------------------------------------------------
    # Internal: access LLM sub-modules
    # ------------------------------------------------------------------

    @property
    def _llm_model(self):
        """Return the inner model (e.g. LoraQwen3_5MoeTextModel) inside the ForCausalLM wrapper."""
        return self.llm.model

    @property
    def _llm_embed_tokens(self):
        return self._llm_model.embed_tokens

    @property
    def _llm_layers(self):
        return self._llm_model.layers

    @property
    def _llm_norm(self):
        return self._llm_model.norm

    @property
    def _llm_rotary_emb(self):
        return self._llm_model.rotary_emb

    @property
    def _llm_lm_head(self):
        return self.llm.lm_head

    # ------------------------------------------------------------------
    # Internal: pipeline communication helpers
    # ------------------------------------------------------------------

    def _sync_recv_hidden(
        self,
        shape: tuple,
        src_stage: int,
        dtype: torch.dtype,
        tag: int = 0,
    ) -> torch.Tensor:
        """
        Synchronously receive hidden_states from src_stage.
        Blocks until data is fully received.
        """
        buf = torch.empty(shape, dtype=dtype, device=self._my_device)
        pipeline_recv(buf, src_stage, tag)
        return buf

    # ------------------------------------------------------------------
    # Internal: gather memory_states across pipeline stages
    # ------------------------------------------------------------------

    def _gather_memory_states_no_grad(
        self,
        memory_states: Optional[torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Gather per-layer memory_states from all pipeline stages onto the
        mem_gather_target stage (the stage that owns the first layer of the
        hypernetwork / m2p_transformer) so that the returned tensor contains
        ALL layers' memory token hidden states.

        Each non-target stage sends its local memory_states slice to the
        target stage.  The target stage receives and writes the remote
        slices into its own buffer.  Communication is no-grad (used by
        llm_forward_no_grad).

        Args:
            memory_states: (B, num_layers, num_mem, H) buffer — each stage
                only has non-zero entries for its own layers.
            batch_size: batch size B.
            dtype: tensor dtype.

        Returns:
            On the mem_gather_target stage: the fully-populated memory_states tensor.
            On other stages: None (the data has been sent away).
        """
        if memory_states is None:
            return None

        num_mem = memory_states.shape[2]
        mem_tag_base = 1000  # avoid collision with hidden_states tags

        for src_stage in self._mem_gather_stages:
            src_layers = self._stage_to_layers[src_stage]
            # Shape of the slice this stage will send: (B, n_local_layers, num_mem, H)
            slice_shape = (batch_size, len(src_layers), num_mem, self._hidden_size)

            if self._my_stage == src_stage:
                # Pack local layers into a contiguous slice and send
                local_slice = torch.stack(
                    [memory_states[:, li, :, :] for li in src_layers], dim=1
                )  # (B, n_local_layers, num_mem, H)
                # Use blocking send to avoid NCCL WorkNCCL objects
                # accumulating in ProcessGroupNCCL's internal tracking
                # lists (memory leak).
                pipeline_send(
                    local_slice.contiguous(), self._mem_gather_target_stage, tag=mem_tag_base + src_stage
                )

            if self._my_stage == self._mem_gather_target_stage:
                buf = torch.empty(slice_shape, dtype=dtype, device=self._my_device)
                pipeline_recv(buf, src_stage, tag=mem_tag_base + src_stage)
                for i, li in enumerate(src_layers):
                    memory_states[:, li, :, :] = buf[:, i, :, :]

        if self._my_stage == self._mem_gather_target_stage:
            return memory_states
        return None

    def _gather_memory_states_with_grad(
        self,
        memory_states: Optional[torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Same as _gather_memory_states_no_grad but uses autograd-aware
        communication (PipelineAsyncSend / PipelineRecv) so that gradients
        can flow back through the memory_states during backward.

        Uses list-based collection + torch.stack instead of in-place
        tensor mutation so that the autograd graph is preserved.

        Args:
            memory_states: (B, num_layers, num_mem, H) tensor built via
                torch.stack (no in-place writes).
            batch_size: batch size B.
            dtype: tensor dtype.

        Returns:
            On the mem_gather_target stage: the fully-populated (B, L, M, H) memory_states.
            On other stages: None.
        """
        if memory_states is None:
            return None

        num_layers = memory_states.shape[1]
        num_mem = memory_states.shape[2]
        mem_tag_base = 1000

        # On the target stage, start with per-layer slices from the local
        # memory_states (which already has correct values for target-stage
        # layers and zeros for remote layers).
        if self._my_stage == self._mem_gather_target_stage:
            layer_slices = [memory_states[:, li, :, :] for li in range(num_layers)]

        # Communicate: non-target stages send, target stage receives.
        # We use **no-grad** isend here on purpose.  Using with-grad send
        # would create one backward anchor per source stage, and each
        # anchor's backward would re-traverse the step-1 LLM graph
        # (shared with the llm_layer_send anchor), causing duplicate
        # blocking sends to the previous pipeline stage — deadlock.
        #
        # Instead, each LLM stage saves its local memory_states tensor.
        # In pipeline_backward(), the mem_gather_target stage extracts
        # per-source-stage gradient slices from the assembled
        # memory_states.grad and sends them back.  Each LLM stage
        # receives its grad slice and calls
        # local_memory_states.backward(grad) exactly once.
        mem_gather_src_info = []  # [(src_stage, src_layers), ...]
        for src_stage in self._mem_gather_stages:
            src_layers = self._stage_to_layers[src_stage]
            slice_shape = (batch_size, len(src_layers), num_mem, self._hidden_size)

            if self._my_stage == src_stage:
                local_slice = torch.stack(
                    [memory_states[:, li, :, :] for li in src_layers], dim=1
                )
                # Use blocking send to avoid NCCL WorkNCCL objects accumulating
                # in ProcessGroupNCCL's internal tracking lists (memory leak).
                # This is safe because the target stage receives in the same
                # src_stage order, so sends complete sequentially.
                pipeline_send(
                    local_slice.contiguous(), self._mem_gather_target_stage,
                    tag=mem_tag_base + src_stage,
                )

            if self._my_stage == self._mem_gather_target_stage:
                buf = torch.empty(slice_shape, dtype=dtype, device=self._my_device)
                pipeline_recv(buf, src_stage, tag=mem_tag_base + src_stage)
                # Replace zero slices with received data (no in-place mutation)
                for i, li in enumerate(src_layers):
                    layer_slices[li] = buf[:, i, :, :]
                mem_gather_src_info.append((src_stage, src_layers))

        # No pending sends to wait for — all sends are blocking now

        # Save metadata for explicit backward in pipeline_backward()
        # Each LLM stage saves its local memory_states for backward.
        if self._my_stage in [s for s in self._mem_gather_stages]:
            self._mem_gather_local_memory = memory_states
        if self._my_stage == self._mem_gather_target_stage:
            self._mem_gather_src_info = mem_gather_src_info

        # Reassemble on the target stage via torch.stack (autograd-safe
        # for local layers; remote layers are plain tensors from recv).
        if self._my_stage == self._mem_gather_target_stage:
            assembled = torch.stack(layer_slices, dim=1)  # (B, L, M, H)
            # When the mem_gather_target stage has no local LLM layers
            # (e.g. it is a pure hypernetwork stage), all layer_slices
            # come from pipeline_recv (plain tensors) or zero placeholders,
            # so assembled.requires_grad is False.  We must explicitly
            # enable requires_grad so that the hypernetwork backward can
            # produce assembled.grad, which is then scattered back to the
            # LLM stages for their step-1 backward.
            if not assembled.requires_grad:
                assembled.requires_grad_(True)
            # retain_grad so we can read .grad after hypernetwork backward
            # and send per-source-stage gradient slices back to LLM stages.
            assembled.retain_grad()
            self._mem_gather_assembled = assembled
            return assembled
        return None

    # ------------------------------------------------------------------
    # Internal: model-family-specific input preparation
    # ------------------------------------------------------------------

    def _embed_and_mem_tokens(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        use_mem_token: bool,
        context_lengths: Optional[torch.LongTensor] = None,
    ):
        """
        Shared logic: run embedding + optional memory token placement on
        the embed stage.  Non-embed stages derive shape from input_ids or
        attention_mask.

        Scheme B: mem_token placeholders are already in input_ids at positions
        [context_lengths[i], context_lengths[i] + num_mem_token).  This method
        overwrites those placeholder embeddings with the learned mem_tokens.
        No attention_mask manipulation is needed (caller passes None).

        Returns:
            inputs_embeds: (B, S, H) on the embed stage, None on others
            attention_mask: unchanged (should be None for Scheme B context)
            seq_len: total sequence length
            batch_size: batch size
            context_lengths: passed through for downstream use
        """
        if self._my_stage == self._embed_stage:
            inputs_embeds = self._llm_embed_tokens(input_ids)
            if use_mem_token and getattr(self._llm_model, "has_mem_token", False):
                if context_lengths is None:
                    raise ValueError(
                        "context_lengths must be provided when use_mem_token=True (Scheme B)"
                    )
                num_mem = self._llm_model.num_mem_token
                mem = self._llm_model.mem_tokens.unsqueeze(0).expand(
                    inputs_embeds.shape[0], -1, -1
                )
                # Overwrite placeholder positions with mem_token embeddings
                for i in range(inputs_embeds.shape[0]):
                    start = context_lengths[i].item()
                    inputs_embeds[i, start:start + num_mem] = mem[i]
            batch_size = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]
        else:
            inputs_embeds = None
            # Derive shape from input_ids (preferred) or attention_mask
            if input_ids is not None:
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
            elif attention_mask is not None:
                batch_size = attention_mask.shape[0]
                seq_len = attention_mask.shape[1]
            else:
                raise ValueError(
                    "Non-embed stages need input_ids or attention_mask to infer shape"
                )
        return inputs_embeds, attention_mask, seq_len, batch_size

    def _prepare_inputs_qwen3_5moe(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        use_mem_token: bool,
        batch_id: Optional[str] = None,
        context_lengths: Optional[torch.LongTensor] = None,
    ):
        """
        Qwen3_5Moe-specific input preparation.

        Qwen3_5Moe uses:
          - 4D position_ids (4, B, S) for text + temporal/height/width
          - M-RoPE with 3D rotary position_ids (3, B, S)
          - Per-layer mask: causal_mask for full_attention, linear_attn_mask
            for linear_attention layers

        Caches causal_mask, position_embeddings, position_ids, and
        linear_attn_mask across calls with the same (batch_id, use_mem_token)
        to avoid redundant recomputation.  Each batch must have a unique
        batch_id; when batch_id is None, caching is disabled.

        Returns:
            (inputs_embeds, causal_mask, position_embeddings,
             seq_len, batch_size, text_position_ids, linear_attn_mask)
        """
        inputs_embeds, attention_mask, seq_len, batch_size = self._embed_and_mem_tokens(
            input_ids, attention_mask, use_mem_token, context_lengths=context_lengths,
        )

        cache_key = (batch_id, use_mem_token) if batch_id is not None else None
        cached = getattr(self, "_cached_inputs_qwen3_5moe", None)
        if cache_key is not None and cached is not None and cached["key"] == cache_key:
            # Reuse previously computed masks and embeddings
            return (
                inputs_embeds,
                cached["causal_mask"],
                cached["position_embeddings"],
                seq_len, batch_size,
                cached["position_ids"],
                cached["linear_attn_mask"],
            )

        device = self._my_device
        # A tiny tensor for shape/device/dtype inference — only .shape, .device,
        # .dtype are read by create_causal_mask and rotary_emb.
        dummy_embeds = torch.empty(
            batch_size, seq_len, 1, device=device, dtype=self._dtype,
        )

        # 4D position_ids: (4, B, S) — text, temporal, height, width
        position_ids = torch.arange(seq_len, device=device)
        position_ids = position_ids.view(1, 1, -1).expand(4, batch_size, -1)
        text_position_ids = position_ids[0]       # (B, S)
        rotary_position_ids = position_ids[1:]    # (3, B, S)

        # Causal mask for full-attention layers
        try:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import create_causal_mask
        except ImportError:
            from transformers.masking_utils import create_causal_mask
        # transformers 5.x added a required cache_position arg.
        _seq_len = dummy_embeds.shape[1]
        causal_mask = create_causal_mask(
            config=self._text_config,
            inputs_embeds=dummy_embeds,
            attention_mask=attention_mask,
            cache_position=torch.arange(_seq_len, device=dummy_embeds.device),
            past_key_values=None,
            position_ids=text_position_ids,
        )

        # Linear attention mask for linear_attention layers
        linear_attn_mask = attention_mask
        if attention_mask is not None and torch.all(attention_mask == 1):
            linear_attn_mask = None

        # Rotary position embeddings (cos, sin)
        position_embeddings = self._llm_rotary_emb(
            dummy_embeds, position_ids=rotary_position_ids,
        )

        # Cache for reuse (only when batch_id is provided)
        if cache_key is not None:
            self._cached_inputs_qwen3_5moe = {
                "key": cache_key,
                "causal_mask": causal_mask,
                "position_embeddings": position_embeddings,
                "position_ids": text_position_ids,
                "linear_attn_mask": linear_attn_mask,
            }

        return (
            inputs_embeds, causal_mask, position_embeddings,
            seq_len, batch_size, text_position_ids, linear_attn_mask,
        )

    def _prepare_inputs_qwen3moe(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        use_mem_token: bool,
        batch_id: Optional[str] = None,
        context_lengths: Optional[torch.LongTensor] = None,
    ):
        """
        Qwen3Moe-specific input preparation.

        Qwen3Moe uses:
          - 2D position_ids (1, S)
          - Standard 1-D RoPE
          - Same causal_mask for all layers (or sliding window)
          - No linear_attn_mask

        Caches causal_mask, position_embeddings, and position_ids across
        calls with the same (batch_id, use_mem_token) to avoid redundant
        recomputation.  Each batch must have a unique batch_id; when
        batch_id is None, caching is disabled.

        Returns:
            (inputs_embeds, causal_mask, position_embeddings,
             seq_len, batch_size, position_ids, None)
        """
        inputs_embeds, attention_mask, seq_len, batch_size = self._embed_and_mem_tokens(
            input_ids, attention_mask, use_mem_token, context_lengths=context_lengths,
        )

        cache_key = (batch_id, use_mem_token) if batch_id is not None else None
        cached = getattr(self, "_cached_inputs_qwen3moe", None)
        if cache_key is not None and cached is not None and cached["key"] == cache_key:
            # Reuse previously computed masks and embeddings
            return (
                inputs_embeds,
                cached["causal_mask"],
                cached["position_embeddings"],
                seq_len, batch_size,
                cached["position_ids"],
                None,
            )

        device = self._my_device
        # A tiny tensor for shape/device/dtype inference — only .shape, .device,
        # .dtype are read by create_causal_mask and rotary_emb.
        dummy_embeds = torch.empty(
            batch_size, seq_len, 1, device=device, dtype=self._dtype,
        )

        # 2D position_ids: (1, S)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            create_causal_mask, create_sliding_window_causal_mask,
        )
        sliding_window = getattr(self._text_config, "sliding_window", None)
        mask_fn = create_causal_mask if sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_fn(
            config=self._text_config,
            inputs_embeds=dummy_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )

        # Rotary position embeddings (cos, sin)
        position_embeddings = self._llm_rotary_emb(
            dummy_embeds, position_ids=position_ids,
        )

        # Cache for reuse (only when batch_id is provided)
        if cache_key is not None:
            self._cached_inputs_qwen3moe = {
                "key": cache_key,
                "causal_mask": causal_mask,
                "position_embeddings": position_embeddings,
                "position_ids": position_ids,
            }

        return (
            inputs_embeds, causal_mask, position_embeddings,
            seq_len, batch_size, position_ids, None,
        )

    # ------------------------------------------------------------------
    # Internal: model-family-specific per-layer call
    # ------------------------------------------------------------------

    def _run_layer_qwen3_5moe(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
        position_embeddings: tuple,
        position_ids: Optional[torch.Tensor],
        linear_attn_mask: Optional[torch.Tensor],
        loradict: Optional[Dict],
        nograd_loradict: Optional[Dict] = None,
        nograd_wdict: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Run a single Qwen3_5Moe decoder layer.

        Qwen3_5Moe has mixed layer types: full_attention layers use causal_mask,
        linear_attention layers use linear_attn_mask.
        Note: position_embeddings is a required positional arg (2nd param).
        """
        layer_types = self._text_config.layer_types
        if layer_types[layer_idx] == "linear_attention":
            layer_mask = linear_attn_mask
        else:
            layer_mask = causal_mask

        return self._llm_layers[layer_idx](
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=position_ids,
            loradict=loradict[layer_idx] if loradict is not None else None,
            nograd_loradict=nograd_loradict[layer_idx] if nograd_loradict is not None else None,
            nograd_wdict=nograd_wdict[layer_idx] if nograd_wdict is not None else None,
        )

    def _run_layer_qwen3moe(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
        position_embeddings: tuple,
        position_ids: Optional[torch.Tensor],
        linear_attn_mask: Optional[torch.Tensor],
        loradict: Optional[Dict],
        nograd_loradict: Optional[Dict] = None,
        nograd_wdict: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Run a single Qwen3Moe decoder layer.

        All layers use the same causal_mask. linear_attn_mask is unused (always
None).
        """
        return self._llm_layers[layer_idx](
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            loradict=loradict[layer_idx] if loradict is not None else None,
            nograd_loradict=nograd_loradict[layer_idx] if nograd_loradict is not None else None,
            nograd_wdict=nograd_wdict[layer_idx] if nograd_wdict is not None else None,
        )
    # ------------------------------------------------------------------
    # Internal: pipeline layer execution (shared skeleton)
    # ------------------------------------------------------------------

    def _run_layers_no_grad(
        self,
        hidden_states: Optional[torch.Tensor],
        causal_mask: Optional[torch.Tensor],
        position_embeddings: tuple,
        position_ids: Optional[torch.Tensor],
        linear_attn_mask: Optional[torch.Tensor],
        seq_len: int,
        batch_size: int,
        dtype: torch.dtype,
        use_mem_token: bool,
        loradict: Optional[Dict] = None,
        context_lengths: Optional[torch.LongTensor] = None,
        nograd_loradict: Optional[Dict] = None,
        nograd_wdict: Optional[Dict] = None,
    ) -> tuple:
        """
        Run decoder layers across pipeline stages WITHOUT gradient.
        Uses async send + sync recv.

        When use_mem_token is True, collects per-layer memory token hidden
        states into a (B, num_layers, num_mem, H) tensor.  After all layers
        are processed, memory_states from all stages are gathered onto the
        mem_gather_target stage (hypernetwork first layer's GPU) so the
        returned tensor contains ALL layers' data.

        Returns:
            (hidden_states, memory_states) — memory_states is fully populated
            on the mem_gather_target stage; None on other stages.
        """
        has_mem = use_mem_token and getattr(self._llm_model, "has_mem_token", False)
        num_mem = self._llm_model.num_mem_token if has_mem else 0

        # Pre-allocate per-layer memory_states buffer on this stage
        if has_mem:
            memory_states = torch.zeros(
                batch_size, self._num_layers, num_mem, self._hidden_size,
                device=self._my_device, dtype=dtype,
            )
        else:
            memory_states = None

        for layer_idx in range(self._num_layers):
            layer_stage = self._layer_to_stage[layer_idx]

            # --- Receive from previous stage if needed ---
            if layer_idx > 0:
                prev_stage = self._layer_to_stage[layer_idx - 1]
                if prev_stage != layer_stage and layer_stage == self._my_stage:
                    hidden_states = self._sync_recv_hidden(
                        shape=(batch_size, seq_len, self._hidden_size),
                        src_stage=prev_stage,
                        dtype=dtype,
                    )

            # --- Run layer if it's on this stage ---
            if layer_stage == self._my_stage:
                hidden_states = self._run_layer(
                    layer_idx, hidden_states,
                    causal_mask, position_embeddings,
                    position_ids, linear_attn_mask, loradict,
                    nograd_loradict, nograd_wdict,
                )
                # Collect per-layer memory token hidden states (Scheme B)
                if has_mem:
                    for b in range(batch_size):
                        start = context_lengths[b].item()
                        memory_states[b, layer_idx, :, :] = hidden_states[b, start:start + num_mem, :]

            # --- Send to next stage if needed (blocking) ---
            if layer_idx < self._num_layers - 1:
                next_stage = self._layer_to_stage[layer_idx + 1]
                if layer_stage != next_stage and layer_stage == self._my_stage:
                    # Use blocking send to avoid NCCL WorkNCCL objects
                    # accumulating (memory leak).
                    pipeline_send(
                        hidden_states.contiguous(), next_stage
                    )

        # --- Transfer to norm stage if last layer stage != norm stage ---
        last_layer_stage = self._layer_to_stage[self._num_layers - 1]
        if last_layer_stage != self._norm_stage:
            if last_layer_stage == self._my_stage:
                pipeline_send(hidden_states.contiguous(), self._norm_stage)
            if self._norm_stage == self._my_stage:
                hidden_states = self._sync_recv_hidden(
                    shape=(batch_size, seq_len, self._hidden_size),
                    src_stage=last_layer_stage,
                    dtype=dtype,
                )

        # --- Gather memory_states from all stages onto the norm stage ---
        memory_states = self._gather_memory_states_no_grad(
            memory_states, batch_size, dtype,
        )

        return hidden_states, memory_states

    def _run_layers_with_grad(
        self,
        hidden_states: Optional[torch.Tensor],
        causal_mask: Optional[torch.Tensor],
        position_embeddings: tuple,
        position_ids: Optional[torch.Tensor],
        linear_attn_mask: Optional[torch.Tensor],
        seq_len: int,
        batch_size: int,
        dtype: torch.dtype,
        use_mem_token: bool,
        loradict: Optional[Dict] = None,
        context_lengths: Optional[torch.LongTensor] = None,
        nograd_loradict: Optional[Dict] = None,
        nograd_wdict: Optional[Dict] = None,
    ) -> tuple:
        """
        Run decoder layers across pipeline stages WITH gradient.
        Uses PipelineAsyncSend (async, with grad) for sends and
        PipelineRecv (blocking, with grad) for receives.

        When use_mem_token is True, collects per-layer memory token hidden
        states into a (B, num_layers, num_mem, H) tensor.  After all layers
        are processed, memory_states from all stages are gathered onto the
        mem_gather_target stage (hypernetwork first layer's GPU) so the
        returned tensor contains ALL layers' data.

        Returns:
            (hidden_states, memory_states) — memory_states is fully populated
            on the mem_gather_target stage; None on other stages.
        """
        pending_send_tensor = None
        has_mem = use_mem_token and getattr(self._llm_model, "has_mem_token", False)
        num_mem = self._llm_model.num_mem_token if has_mem else 0

        # Collect per-layer memory slices in a list to avoid in-place ops
        # that would break the autograd graph.  Each entry is either a
        # (B, num_mem, H) tensor (for layers on this stage) or None.
        mem_slices: list = [] if has_mem else None

        for layer_idx in range(self._num_layers):
            layer_stage = self._layer_to_stage[layer_idx]

            # --- Receive from previous stage if needed ---
            if layer_idx > 0:
                prev_stage = self._layer_to_stage[layer_idx - 1]
                if prev_stage != layer_stage and layer_stage == self._my_stage:
                    placeholder = torch.empty(0, device=self._my_device, requires_grad=True)
                    hidden_states = PipelineRecv.apply(
                        placeholder, prev_stage,
                        (batch_size, seq_len, self._hidden_size),
                        dtype, self._my_device, 0,
                    )

            # --- Run layer if it's on this stage ---
            if layer_stage == self._my_stage:
                if self._activation_checkpointing:
                    hidden_states = torch_checkpoint(
                        self._run_layer,
                        layer_idx, hidden_states,
                        causal_mask, position_embeddings,
                        position_ids, linear_attn_mask, loradict,
                        nograd_loradict, nograd_wdict,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = self._run_layer(
                        layer_idx, hidden_states,
                        causal_mask, position_embeddings,
                        position_ids, linear_attn_mask, loradict,
                        nograd_loradict, nograd_wdict,
                    )
                # Collect per-layer memory token hidden states (Scheme B, no in-place)
                if has_mem:
                    # Extract mem tokens from per-sample positions
                    slices = []
                    for b in range(batch_size):
                        start = context_lengths[b].item()
                        slices.append(hidden_states[b, start:start + num_mem, :])
                    mem_slices.append(torch.stack(slices, dim=0))
            else:
                # Placeholder zero slice for layers not on this stage
                if has_mem:
                    mem_slices.append(
                        torch.zeros(
                            batch_size, num_mem, self._hidden_size,
                            device=self._my_device, dtype=dtype,
                        )
                    )

            # --- Send to next stage if needed ---
            if layer_idx < self._num_layers - 1:
                next_stage = self._layer_to_stage[layer_idx + 1]
                if layer_stage != next_stage and layer_stage == self._my_stage:
                    pending_send_tensor = pipeline_send_with_grad(
                        hidden_states, next_stage
                    )

        if pending_send_tensor is not None:
            # Save anchor only if the tensor has gradient.  When the LLM
            # is fully frozen (no LoRA), the send is a plain send and
            # no backward recv is needed.
            if pending_send_tensor.requires_grad and pending_send_tensor.grad_fn is not None:
                pending_send_tensor._anchor_source = f"llm_layer_send(stage{self._my_stage}→next)"
                self._backward_anchors.append(pending_send_tensor)
        last_layer_stage = self._layer_to_stage[self._num_layers - 1]
        if last_layer_stage != self._norm_stage:
            if last_layer_stage == self._my_stage:
                out = pipeline_send_with_grad(
                    hidden_states, self._norm_stage
                )
                # Save anchor only if the tensor has gradient.
                if out.requires_grad and out.grad_fn is not None:
                    out._anchor_source = f"llm_last_layer→norm_send(stage{last_layer_stage}→norm{self._norm_stage})"
                    self._backward_anchors.append(out)
            if self._norm_stage == self._my_stage:
                placeholder = torch.empty(0, device=self._my_device, requires_grad=True)
                hidden_states = PipelineRecv.apply(
                    placeholder, last_layer_stage,
                    (batch_size, seq_len, self._hidden_size),
                    dtype, self._my_device, 0,
                )

        # --- Stack memory slices into (B, L, M, H) tensor ---
        if has_mem:
            # torch.stack preserves the autograd graph (no in-place mutation)
            memory_states = torch.stack(mem_slices, dim=1)  # (B, L, M, H)
        else:
            memory_states = None

        # --- Gather memory_states from all stages onto the norm stage ---
        memory_states = self._gather_memory_states_with_grad(
            memory_states, batch_size, dtype,
        )

        return hidden_states, memory_states

    def _apply_norm_and_strip_mem(
        self,
        hidden_states: Optional[torch.Tensor],
        use_mem_token: bool,
        context_lengths: Optional[torch.LongTensor] = None,
    ):
        """
        Apply final norm on the norm stage.
        If use_mem_token, strip memory tokens and padding from hidden_states
        before norm (Scheme B: keep only first max(context_lengths) tokens).

        Note: per-layer memory_states are now collected inside
        _run_layers_no_grad / _run_layers_with_grad, so this method only
        strips the trailing mem tokens + padding and applies norm.

        Returns:
            hidden_states: (B, S_orig, H) after norm — only valid on norm stage
        """
        if self._my_stage == self._norm_stage:
            if use_mem_token and getattr(self._llm_model, "has_mem_token", False):
                # Scheme B: strip everything after valid tokens
                max_valid = context_lengths.max().item()
                hidden_states = hidden_states[:, :max_valid, :]
            hidden_states = self._llm_norm(hidden_states)
        return hidden_states

    # ==================================================================
    # Public forward functions — LLM only
    # ==================================================================

    @torch.no_grad()
    def llm_forward_no_grad(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        loradict: Optional[Dict] = None,
        use_mem_token: bool = False,
        batch_id: Optional[str] = None,
        context_lengths: Optional[torch.LongTensor] = None,
        nograd_loradict: Optional[Dict] = None,
        nograd_wdict: Optional[Dict] = None,
    ) -> tuple:
        """
        Forward through the LLM **without gradient**.

        Pipeline-parallel: each stage runs its own layers; hidden_states are
        transferred between stages via async send / sync recv.

        When ``use_mem_token=True``, memory token embeddings are written into
        the placeholder positions indicated by ``context_lengths`` (Scheme B).
        After all layers, the memory token hidden states are extracted
        (before norm) and returned separately.

        Args:
            input_ids:      (B, S) token ids — required on the embed stage,
                            ignored on other stages.
            attention_mask:  (B, S) or None.  Under Scheme B, pass None for
                            context so SDPA can use is_causal=True (Flash).
            loradict:       Optional per-layer LoRA dict (keys are layer indices).
            use_mem_token:  If True, write mem_token embeddings into placeholder
                            positions and return their hidden states separately.
            batch_id:       A unique identifier for this batch of data.
            context_lengths: (B,) number of valid tokens per sample.  Required
                            when use_mem_token=True (Scheme B).
            batch_id:       A unique identifier for this batch of data.  When
                            provided, causal_mask / position_embeddings /
                            position_ids are cached and reused across multiple
                            forward calls on the same batch (e.g. no_grad then
                            with_grad).  When None, caching is disabled.

        Returns:
            On the norm stage:
                hidden_states: (B, S, H) — final hidden states after norm
            On the mem_gather_target stage (hypernetwork first layer's GPU):
                memory_states: (B, num_layers, num_mem_token, H) if use_mem_token else None
                    Per-layer memory token hidden states (before norm),
                    gathered from ALL pipeline stages.
            On other stages:
                (None, None)
        """
        (
            hidden_states, causal_mask, position_embeddings,
            seq_len, batch_size, position_ids, linear_attn_mask,
        ) = self._prepare_inputs(
            input_ids, attention_mask, use_mem_token=use_mem_token,
            batch_id=batch_id, context_lengths=context_lengths,
        )

        # Embed stage → first layer stage transfer (if different)
        first_layer_stage = self._layer_to_stage[0]
        if self._embed_stage != first_layer_stage:
            if self._my_stage == self._embed_stage:
                pipeline_send(hidden_states.contiguous(), first_layer_stage)
                hidden_states = None
            if self._my_stage == first_layer_stage:
                hidden_states = self._sync_recv_hidden(
                    shape=(batch_size, seq_len, self._hidden_size),
                    src_stage=self._embed_stage,
                    dtype=self._dtype,
                )

        hidden_states, memory_states = self._run_layers_no_grad(
            hidden_states, causal_mask, position_embeddings,
            position_ids, linear_attn_mask,
            seq_len, batch_size,
            self._dtype, use_mem_token=use_mem_token, loradict=loradict,
            context_lengths=context_lengths,
            nograd_loradict=nograd_loradict,
            nograd_wdict=nograd_wdict,
        )

        hidden_states = self._apply_norm_and_strip_mem(
            hidden_states, use_mem_token=use_mem_token,
            context_lengths=context_lengths,
        )

        # hidden_states is valid on the norm stage;
        # memory_states is valid on the mem_gather_target stage.
        # Return whichever is valid on this stage, None otherwise.
        ret_hidden = hidden_states if self._my_stage == self._norm_stage else None
        ret_memory = memory_states if self._my_stage == self._mem_gather_target_stage else None
        if ret_hidden is not None or ret_memory is not None:
            return ret_hidden, ret_memory
        return None, None

    def llm_forward_with_grad(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        loradict: Optional[Dict] = None,
        use_mem_token: bool = False,
        batch_id: Optional[str] = None,
        context_lengths: Optional[torch.LongTensor] = None,
        nograd_loradict: Optional[Dict] = None,
        nograd_wdict: Optional[Dict] = None,
    ) -> tuple:
        """
        Forward through the LLM **with gradient**.

        Gradients flow back through the pipeline via PipelineAsyncSend /
        PipelineRecv autograd functions.  All stages must participate in
        backward() for the autograd send/recv to work.

        When ``use_mem_token=True``, memory tokens are appended to embeddings.
        After all layers, memory token hidden states are extracted (before norm)
        and returned alongside the normed hidden states.

        Args:
            input_ids:      (B, S) token ids — required on the embed stage.
            attention_mask:  (B, S) — **must be on ALL stages**.
            loradict:       Optional per-layer LoRA dict.
            use_mem_token:  If True, append memory tokens to embeddings and
                            return their hidden states separately.  Default False.
            batch_id:       A unique identifier for this batch of data.  When
                            provided, causal_mask / position_embeddings /
                            position_ids are cached and reused across multiple
                            forward calls on the same batch (e.g. no_grad then
                            with_grad).  When None, caching is disabled.

        Returns:
            On the norm stage:
                hidden_states: (B, S, H) — final hidden states after norm
            On the mem_gather_target stage (hypernetwork first layer's GPU):
                memory_states: (B, num_layers, num_mem_token, H) if use_mem_token else None
                    Per-layer memory token hidden states (before norm),
                    gathered from ALL pipeline stages.
            On other stages:
                (None, None)
        """
        (
            hidden_states, causal_mask, position_embeddings,
            seq_len, batch_size, position_ids, linear_attn_mask,
        ) = self._prepare_inputs(
            input_ids, attention_mask, use_mem_token=use_mem_token,
            batch_id=batch_id, context_lengths=context_lengths,
        )

        # Embed stage → first layer stage transfer (with grad)
        embed_anchor = None  # backward anchor for embed stage
        first_layer_stage = self._layer_to_stage[0]
        if self._embed_stage != first_layer_stage:
            if self._my_stage == self._embed_stage:
                out = pipeline_send_with_grad(
                    hidden_states, first_layer_stage
                )
                embed_anchor = out  # keep reference for backward
                hidden_states = None
            if self._my_stage == first_layer_stage:
                placeholder = torch.empty(0, device=self._my_device, requires_grad=True)
                hidden_states = PipelineRecv.apply(
                    placeholder, self._embed_stage,
                    (batch_size, seq_len, self._hidden_size),
                    self._dtype, self._my_device, 0,
                )

        hidden_states, memory_states = self._run_layers_with_grad(
            hidden_states, causal_mask, position_embeddings,
            position_ids, linear_attn_mask,
            seq_len, batch_size,
            self._dtype, use_mem_token=use_mem_token, loradict=loradict,
            context_lengths=context_lengths,
            nograd_loradict=nograd_loradict,
            nograd_wdict=nograd_wdict,
        )

        hidden_states = self._apply_norm_and_strip_mem(
            hidden_states, use_mem_token=use_mem_token,
            context_lengths=context_lengths,
        )

        # --- Save backward anchor for pipeline_backward() ---
        # On the norm stage, the caller will use the returned hidden_states
        # to compute loss and call pipeline_backward(loss).  On other stages
        # that have LLM layers, the necessary anchors (llm_layer_send,
        # llm_last_layer→norm_send) are already saved inside
        # _run_layers_with_grad.  The only case left is the embed-only stage
        # (embed on a different GPU than the first layer): it needs the
        # embed→first_layer send anchor to participate in backward.
        if self._my_stage != self._norm_stage:
            if embed_anchor is not None and embed_anchor.requires_grad and embed_anchor.grad_fn is not None:
                embed_anchor._anchor_source = f"embed_send(embed{self._embed_stage}→layer{first_layer_stage})"
                self._backward_anchors.append(embed_anchor)  # embed-only stage

        # --- Debug: gather and log anchors from ALL stages ---
        if self._debug_anchor:
            self._gather_and_log_anchors("llm_forward_with_grad")

        # hidden_states is valid on the norm stage;
        # memory_states is valid on the mem_gather_target stage.
        # Return whichever is valid on this stage, None otherwise.
        ret_hidden = hidden_states if self._my_stage == self._norm_stage else None
        ret_memory = memory_states if self._my_stage == self._mem_gather_target_stage else None
        if ret_hidden is not None or ret_memory is not None:
            return ret_hidden, ret_memory
        return None, None

    # ==================================================================
    # Pipeline-aware backward
    # ==================================================================

    def pipeline_backward(self, loss: Optional[torch.Tensor] = None, retain_graph: bool = False,
                          lora_scatter_recv_count: int = 1):
        """
        Trigger backward across **all** pipeline stages.

        In pipeline parallelism each stage has its own independent autograd
        engine.  The with-grad communication primitives (PipelineAsyncSend /
        PipelineRecv) embed blocking send/recv in their backward methods, so
        **every stage must run backward** for the gradient chain to complete
        without deadlock.

        Usage (after ``llm_forward_with_grad`` with ``loradict``)::

            hidden_states, _ = model.llm_forward_with_grad(...)
            if hidden_states is not None:          # norm stage
                loss = criterion(hidden_states, ...)
            else:
                loss = None
            model.pipeline_backward(loss)          # ALL stages call this

        What happens internally:

        * **Norm stage** (``loss is not None``): calls ``loss.backward()``.
          This triggers ``PipelineRecv.backward`` which sends gradients to
          the previous stage.
        * **Other stages**: call ``anchor.backward(torch.zeros_like(anchor))``
          on the backward anchors saved during the most recent
          ``llm_forward_with_grad`` and related with-grad functions.  This
          kicks off the local autograd engine which executes
          ``PipelineAsyncSend.backward`` (recv grad from the next stage)
          and ``PipelineRecv.backward`` (send grad to the previous stage),
          propagating gradients all the way back.

        The anchors include:
        * LLM forward anchors — ``PipelineAsyncSend`` outputs from the
          layer-to-layer pipeline communication.
        * Memory gather anchors — ``PipelineAsyncSend`` outputs from
          ``_gather_memory_states_with_grad`` on non-target stages.

        Additionally, the m2p_norm stage performs **explicit** lora-scatter
        backward: it receives lora gradients from every LLM stage (in
        reverse order to match the natural backward propagation) and calls
        ``memory_states.backward(grad)`` once to trigger the hypernetwork
        backward.  This avoids the deadlock that would occur if each
        lora-scatter anchor independently re-traversed the hypernetwork
        graph, issuing duplicate blocking
        sends to stage 6 with no matching recv.

        Args:
            loss: Scalar loss tensor on the norm stage; ``None`` on all
                other stages.
            retain_graph: If True, the computation graph is retained after
                backward so that additional backward passes can be performed.
                Default False.
        """
        # --- Debug: gather and log anchors from ALL stages ---
        if self._debug_anchor:
            self._gather_and_log_anchors(
                "pipeline_backward",
                extra_info=f"loss={'present' if loss is not None else 'None'}",
            )

        if loss is not None:
            # Norm stage — start the backward chain from the loss
            loss.backward(retain_graph=retain_graph)

        # ALL stages (including norm) backward through saved anchors.
        # These anchors are PipelineAsyncSend outputs whose backward methods
        # recv gradients from destination stages.  Without this, the
        # destination stages' PipelineRecv.backward (which sends gradients)
        # would have no matching recv, causing deadlock.
        #
        # Passing zeros as grad_output is safe because PipelineAsyncSend.backward
        # ignores grad_output and recvs the real gradient from the peer stage.
        # The recv'd gradient then flows back through the local autograd graph.
        #
        # Process in REVERSE order: NCCL does not support message tags — P2P
        # operations between a pair of ranks are matched purely by FIFO order.
        # When multiple forward passes create anchors (e.g. step-1 LLM then
        # step-4 LLM), the norm stage's loss.backward() sends gradients in
        # reverse topological order (step-4 first, step-1 second).  Non-norm
        # stages must recv in the same order, so we process the last-appended
        # anchors (step-4) first by reversing the list.
        for i, anchor in enumerate(reversed(self._backward_anchors)):
            src = getattr(anchor, '_anchor_source', 'UNKNOWN')
            if not anchor.requires_grad or anchor.grad_fn is None:
                logger.debug(
                    f"  [pipeline_backward] Skipping no-grad anchor on "
                    f"stage={self._my_stage}, anchor[{i}]: source={src}"
                )
                continue
            try:
                anchor.backward(torch.zeros_like(anchor), retain_graph=retain_graph)
            except RuntimeError as e:
                logger.error(
                    f"  [pipeline_backward] FAILED backward on stage={self._my_stage}, "
                    f"anchor[{i}]: source={src}, "
                    f"shape={anchor.shape}, dtype={anchor.dtype}, "
                    f"device={anchor.device}, "
                    f"requires_grad={anchor.requires_grad}, "
                    f"grad_fn={anchor.grad_fn}\n"
                    f"  Error: {e}"
                )
                raise

        # Clean up anchor attributes to release references.
        for anchor in self._backward_anchors:
            if hasattr(anchor, '_anchor_source'):
                del anchor._anchor_source

        # --- Explicit mem_gather backward ---
        #
        # This is handled separately by pipeline_backward_step1() which
        # combines the mem_gather gradient with the step-1 llm_layer_send
        # gradient into a single torch.autograd.backward call.
        # See pipeline_backward_step1() for details.

        # --- Explicit lora-scatter backward on the m2p_norm stage ---
        #
        # The lora scatter uses no-grad isend in forward (to avoid creating
        # anchors whose backward would re-traverse the hypernetwork graph
        # multiple times, causing duplicate blocking sends and deadlock).
        #
        # Here we manually receive the lora gradients from every LLM stage
        # in **reverse** order (highest stage first, matching the natural
        # backward propagation order: stage N finishes first, stage 0 last),
        # scatter-add them into a grad tensor for memory_states, and call
        # memory_states.backward(grad) exactly once to trigger the
        # hypernetwork backward.
        #
        # When distillation is enabled (lora_scatter_recv_count > 1), each
        # LLM stage sends lora grads multiple times (once per backward pass
        # that traverses the loradict computation graph — Phase C' backward
        # and Phase C backward). We recv all of them and accumulate.
        m2p_norm_stage = self.hypernetwork._m2p_norm_stage
        lora_mem = getattr(self, '_lora_scatter_memory_states', None)
        if self._my_stage == m2p_norm_stage and lora_mem is not None:
            dst_info = self._lora_scatter_dst_info      # [(dst_stage, indices), ...]
            tag_base = self._lora_scatter_tag_base
            M4 = self._lora_scatter_M4
            H = self._lora_scatter_H
            B = self._lora_scatter_B

            # Accumulate grad into a tensor shaped like memory_states
            mem_grad = torch.zeros_like(lora_mem)

            # Receive lora_scatter_recv_count times from each stage.
            # Each round corresponds to one backward pass that traversed
            # the loradict graph (e.g., Phase C' backward then Phase C backward).
            # Within each round, receive in reverse stage order (highest first)
            # to match the order in which LLM stages finish backward.
            for _recv_round in range(lora_scatter_recv_count):
                for dst_stage, indices in reversed(dst_info):
                    n_slices = len(indices)
                    recv_shape = (B, n_slices, M4, H)
                    grad_buf = torch.empty(recv_shape, dtype=lora_mem.dtype,
                                           device=self._my_device)
                    pipeline_recv(grad_buf, src_stage=dst_stage,
                                  tag=tag_base + dst_stage)
                    for i, idx in enumerate(indices):
                        mem_grad[:, idx, :, :] += grad_buf[:, i, :, :]

            # Single backward through the hypernetwork
            if lora_mem.requires_grad:
                lora_mem.backward(mem_grad, retain_graph=retain_graph)

            # Clean up
            self._lora_scatter_memory_states = None
            self._lora_scatter_dst_info = None

        # Clear the anchors to free memory (only if not retaining graph)
        if not retain_graph:
            self._backward_anchors = []

    def pipeline_backward_step1(self):
        """
        Phase-3 backward for the step-1 LLM forward pass.

        This method handles the backward through the step-1 LLM computation
        graph, which has TWO outputs that need gradients:

        1. ``llm_layer_send`` — hidden_states sent to the next pipeline stage
        2. ``mem_gather_send`` — memory_states sent to the mem_gather_target

        Both outputs share the same computation graph (the step-1 LLM layers).
        Calling backward on each separately would traverse the graph twice,
        causing duplicate NCCL sends via ``PipelineRecv.backward`` — deadlock.

        Instead, this method:

        1. On the ``mem_gather_target_stage``: extracts per-source-stage
           gradient slices from ``_mem_gather_assembled.grad`` (populated by
           the hypernetwork backward in phase 1+2) and sends them to each
           LLM stage.
        2. On each LLM stage: receives the mem_gather gradient, manually
           receives the ``llm_layer_send`` gradient from the next stage,
           and calls ``torch.autograd.backward`` with BOTH tensors and
           their gradients — a single backward through the shared graph.

        The ``_backward_anchors`` list must contain only step-1 anchors
        when this method is called.
        """
        mem_tag_base = 1000

        # --- Target stage: send mem_gather grad slices to LLM stages ---
        assembled = getattr(self, '_mem_gather_assembled', None)
        src_info = getattr(self, '_mem_gather_src_info', None)

        if self._my_stage == self._mem_gather_target_stage and assembled is not None and src_info is not None:
            assembled_grad = assembled.grad
            if assembled_grad is not None:
                num_mem = assembled_grad.shape[2]
                for src_stage, src_layers in src_info:
                    # Extract the gradient slice for this source stage
                    grad_slice = torch.stack(
                        [assembled_grad[:, li, :, :] for li in src_layers], dim=1
                    )
                    # Use blocking send to avoid NCCL WorkNCCL objects
                    # accumulating in ProcessGroupNCCL's internal tracking
                    # lists (memory leak).
                    pipeline_send(
                        grad_slice.contiguous(), src_stage,
                        tag=mem_tag_base + src_stage,
                    )
            else:
                # No gradient on assembled memory_states — send zeros
                for src_stage, src_layers in src_info:
                    num_mem = self._hidden_size  # placeholder
                    # We need the actual shape; get it from _mem_gather_local_memory
                    # on the target stage.  Since target stage may also be an LLM
                    # stage, it has local memory.
                    local_mem = getattr(self, '_mem_gather_local_memory', None)
                    if local_mem is not None:
                        num_mem = local_mem.shape[2]
                    B = assembled.shape[0]
                    H = self._hidden_size
                    grad_slice = torch.zeros(
                        B, len(src_layers), num_mem, H,
                        dtype=assembled.dtype, device=self._my_device,
                    )
                    # Use blocking send to avoid NCCL WorkNCCL objects
                    # accumulating in ProcessGroupNCCL's internal tracking
                    # lists (memory leak).
                    pipeline_send(
                        grad_slice.contiguous(), src_stage,
                        tag=mem_tag_base + src_stage,
                    )

        # --- LLM stages: recv mem_gather grad ---
        mem_grad = None
        local_memory = getattr(self, '_mem_gather_local_memory', None)
        if (self._my_stage in self._mem_gather_stages
                and local_memory is not None
                and local_memory.requires_grad):
            src_layers = self._stage_to_layers[self._my_stage]
            B = local_memory.shape[0]
            num_mem = local_memory.shape[2]
            H = self._hidden_size
            recv_shape = (B, len(src_layers), num_mem, H)
            mem_grad_buf = torch.empty(
                recv_shape, dtype=local_memory.dtype, device=self._my_device,
            )
            pipeline_recv(
                mem_grad_buf, src_stage=self._mem_gather_target_stage,
                tag=mem_tag_base + self._my_stage,
            )
            # Scatter the received gradient into the full memory_states shape
            # local_memory has shape (B, num_layers, num_mem, H) with zeros
            # for layers not on this stage.  We need grad for the local layers.
            mem_grad = torch.zeros_like(local_memory)
            for i, li in enumerate(src_layers):
                mem_grad[:, li, :, :] = mem_grad_buf[:, i, :, :]

        # --- Combined backward: step-1 anchors + mem_gather grad ---
        #
        # Each step-1 anchor is a PipelineAsyncSend output.  Its backward
        # (PipelineAsyncSendBackward) does a pipeline_recv internally.
        # We include anchors and local_memory in a single
        # torch.autograd.backward call so the shared step-1 LLM graph
        # is traversed exactly once, avoiding duplicate NCCL sends.
        anchors = list(reversed(self._backward_anchors))
        backward_tensors = []
        backward_grads = []

        for anchor in anchors:
            if not anchor.requires_grad or anchor.grad_fn is None:
                continue
            backward_tensors.append(anchor)
            backward_grads.append(torch.zeros_like(anchor))

        # Add local_memory_states if it has gradient
        if mem_grad is not None and local_memory is not None and local_memory.requires_grad:
            backward_tensors.append(local_memory)
            backward_grads.append(mem_grad)

        # Single backward through the step-1 graph
        if backward_tensors:
            torch.autograd.backward(backward_tensors, backward_grads)

        # Clean up — release all references to computation graph tensors.
        for anchor in self._backward_anchors:
            if hasattr(anchor, '_anchor_source'):
                del anchor._anchor_source
        self._backward_anchors = []
        self._mem_gather_local_memory = None
        self._mem_gather_src_info = None
        self._mem_gather_assembled = None

    # ==================================================================
    # Public forward functions — Hypernetwork (m2p_transformer)
    # ==================================================================

    @torch.no_grad()
    def hypernetwork_forward_no_grad(
        self,
        memory_states: torch.Tensor,
        batch_size: int,
    ) -> Dict[int, dict]:
        """
        Forward through the hypernetwork (m2p_transformer) **without gradient**,
        then scatter memory_states to pipeline stages and generate loradict.

        Args:
            memory_states: (B, L, M, H) tensor — per-layer memory token hidden
                states from the LLM.  Must be on the mem_gather_target stage.
            batch_size: batch size B — must be provided by the caller so that
                all pipeline stages know the tensor dimensions for recv buffers.

        Returns:
            loradict on ALL stages: ``{layer_idx: layer_lora_dict}`` containing
                only the layers owned by this stage (plus ``{"attention": None}``
                for non-full-attention layers on this stage).
        """
        # For only_full_1for1: filter memory_states to keep only full_attention layers
        if self._memory_method == "only_full_1for1" and memory_states is not None and memory_states.numel() > 0:
            if self._my_stage == self.hypernetwork._mem_gather_target_stage:
                fa_indices = torch.tensor(self._full_attn_layer_indices, device=memory_states.device)
                memory_states = memory_states[:, fa_indices, :, :]  # (B, L_fa, M, H)
        memory_states = self.hypernetwork.forward_no_grad(memory_states, batch_size)
        return self._scatter_and_generate_lora_dict_no_grad(memory_states, batch_size)

    def hypernetwork_forward_with_grad(
        self,
        memory_states: torch.Tensor,
        batch_size: int,
    ) -> Dict[int, dict]:
        """
        Forward through the hypernetwork (m2p_transformer) **with gradient**,
        then scatter memory_states to pipeline stages and generate loradict.

        Args:
            memory_states: (B, L, M, H) tensor — per-layer memory token hidden
                states from the LLM.  Must be on the mem_gather_target stage.
            batch_size: batch size B — must be provided by the caller so that
                all pipeline stages know the tensor dimensions for recv buffers.

        Returns:
            loradict on ALL stages: ``{layer_idx: layer_lora_dict}`` containing
                only the layers owned by this stage (plus ``{"attention": None}``
                for non-full-attention layers on this stage).
        """
        # For only_full_1for1: filter memory_states to keep only full_attention layers
        if self._memory_method == "only_full_1for1" and memory_states is not None and memory_states.numel() > 0:
            if self._my_stage == self.hypernetwork._mem_gather_target_stage:
                fa_indices = torch.tensor(self._full_attn_layer_indices, device=memory_states.device)
                memory_states = memory_states[:, fa_indices, :, :]  # (B, L_fa, M, H)
        memory_states = self.hypernetwork.forward_with_grad(memory_states, batch_size)

        # Collect backward anchors saved by the hypernetwork's pipeline comm.
        if self.hypernetwork._backward_anchors:
            for a in self.hypernetwork._backward_anchors:
                if not hasattr(a, '_anchor_source'):
                    a._anchor_source = f"hypernet_internal(stage{self._my_stage})"
            self._backward_anchors.extend(self.hypernetwork._backward_anchors)
            self.hypernetwork._backward_anchors = []

        loradict = self._scatter_and_generate_lora_dict_with_grad(memory_states, batch_size)

        # --- Debug: gather and log anchors from ALL stages ---
        if self._debug_anchor:
            self._gather_and_log_anchors("hypernetwork_forward_with_grad")

        return loradict

    # ------------------------------------------------------------------
    # Internal: scatter memory_states and generate loradict
    # ------------------------------------------------------------------

    def _scatter_and_generate_lora_dict_no_grad(
        self,
        memory_states: Optional[torch.Tensor],
        batch_size: int,
    ) -> Dict[int, dict]:
        """
        Scatter processed memory_states from the m2p_norm stage to all
        pipeline stages, then each stage calls ``generate_lora_dict`` on
        its local LLM layers to produce the loradict.

        Communication is no-grad (async isend + sync recv).

        The m2p_norm stage holds the full ``(B, L/4, M*4, H)`` tensor.
        For each *other* stage that owns full_attention layers, the norm
        stage sends the relevant slices.  Each stage then calls
        ``layer.generate_lora_dict(r, scale, plain_tensor)`` for its
        local full_attention layers and fills ``{"attention": None}`` for
        linear_attention layers.

        Args:
            memory_states: (B, L/4, M*4, H) on the m2p_norm stage; None elsewhere.
            batch_size: batch size B (known on all stages).

        Returns:
            loradict: ``{layer_idx: layer_lora_dict}`` for layers on this stage.
        """
        m2p_norm_stage = self.hypernetwork._m2p_norm_stage
        scatter_tag_base = 2000  # avoid collision with other tags
        M4 = self._m2p_num_mem_token * 4 if self._memory_method == "only_full_4for1" else self._m2p_num_mem_token
        H = self._hidden_size
        B = batch_size

        my_fa_info = self._stage_full_attn_info.get(self._my_stage, [])

        # --- Norm stage sends slices to other stages ---
        for dst_stage, fa_info_list in self._stage_full_attn_info.items():
            if dst_stage == m2p_norm_stage:
                continue  # norm stage keeps its own slices locally
            if not fa_info_list:
                continue

            if self._my_stage == m2p_norm_stage:
                # Pack the slices for this destination stage
                # fa_info_list: [(fa_counter, layer_idx), ...]
                indices = [fa_counter for fa_counter, _ in fa_info_list]
                # (B, n_local_fa, M*4, H)
                local_slice = torch.stack(
                    [memory_states[:, idx, :, :] for idx in indices], dim=1
                )
                # Use blocking send to avoid NCCL WorkNCCL objects
                # accumulating in ProcessGroupNCCL's internal tracking
                # lists (memory leak).
                pipeline_send(
                    local_slice.contiguous(), dst_stage,
                    tag=scatter_tag_base + dst_stage,
                )

        # --- Each non-norm stage receives its slices ---
        local_memory = None
        if self._my_stage != m2p_norm_stage and my_fa_info:
            n_local_fa = len(my_fa_info)
            recv_shape = (B, n_local_fa, M4, H)
            local_memory = torch.empty(
                recv_shape, dtype=self._dtype, device=self._my_device,
            )
            pipeline_recv(local_memory, m2p_norm_stage, tag=scatter_tag_base + self._my_stage)

        # No pending sends to wait for — all sends are blocking now

        # --- Generate loradict on each stage ---
        loradict: Dict[int, dict] = {}

        if self._my_stage == m2p_norm_stage and my_fa_info:
            # Norm stage: use local memory_states directly
            for fa_counter, layer_idx in my_fa_info:
                plain_tensor = memory_states[:, fa_counter, :, :].reshape(B, -1)
                loradict[layer_idx] = self._llm_layers[layer_idx].generate_lora_dict(
                    self._lora_ranks, self._generate_lora_scale, plain_tensor,
                )
        elif local_memory is not None:
            # Non-norm stage: use received slices
            for local_idx, (fa_counter, layer_idx) in enumerate(my_fa_info):
                plain_tensor = local_memory[:, local_idx, :, :].reshape(B, -1)
                loradict[layer_idx] = self._llm_layers[layer_idx].generate_lora_dict(
                    self._lora_ranks, self._generate_lora_scale, plain_tensor,
                )

        # Fill in non-full-attention layers on this stage (no LoRA)
        for layer_idx in self._my_layer_indices:
            if layer_idx not in loradict:
                loradict[layer_idx] = None

        return loradict

    def _scatter_and_generate_lora_dict_with_grad(
        self,
        memory_states: Optional[torch.Tensor],
        batch_size: int,
    ) -> Dict[int, dict]:
        """
        Same as ``_scatter_and_generate_lora_dict_no_grad`` but uses
        autograd-aware communication (PipelineAsyncSend / PipelineRecv)
        so that gradients flow back through the scattered memory_states.

        Args:
            memory_states: (B, L/4, M*4, H) on the m2p_norm stage; None elsewhere.
            batch_size: batch size B (known on all stages).

        Returns:
            loradict: ``{layer_idx: layer_lora_dict}`` for layers on this stage.
        """
        m2p_norm_stage = self.hypernetwork._m2p_norm_stage
        scatter_tag_base = 2000
        M4 = self._m2p_num_mem_token * 4 if self._memory_method == "only_full_4for1" else self._m2p_num_mem_token
        H = self._hidden_size
        B = batch_size

        my_fa_info = self._stage_full_attn_info.get(self._my_stage, [])

        # --- Norm stage sends slices to other stages ---
        # We use **no-grad** isend here on purpose.  Using with-grad send
        # would create one backward anchor per destination stage, and each
        # anchor's backward would re-traverse the entire hypernetwork graph
        # issuing duplicate blocking sends
        # to stage 6 that have no matching recv — causing deadlock.
        #
        # Instead, the m2p_norm stage saves `memory_states` and the
        # per-destination slice metadata.  In pipeline_backward() it
        # explicitly receives lora gradients from every LLM stage (in
        # reverse order to match the natural backward propagation), sums
        # them into a single grad tensor, and calls
        # memory_states.backward(grad) exactly once.
        lora_scatter_dst_info = []  # [(dst_stage, indices), ...]
        for dst_stage, fa_info_list in self._stage_full_attn_info.items():
            if dst_stage == m2p_norm_stage:
                continue
            if not fa_info_list:
                continue

            if self._my_stage == m2p_norm_stage:
                indices = [fa_counter for fa_counter, _ in fa_info_list]
                local_slice = torch.stack(
                    [memory_states[:, idx, :, :] for idx in indices], dim=1
                )
                # Use blocking send to avoid NCCL WorkNCCL objects
                # accumulating in ProcessGroupNCCL's internal tracking
                # lists (memory leak).
                pipeline_send(
                    local_slice.contiguous(), dst_stage,
                    tag=scatter_tag_base + dst_stage,
                )
                lora_scatter_dst_info.append((dst_stage, indices))

        # --- Each non-norm stage receives its slices (with grad) ---
        local_memory = None
        if self._my_stage != m2p_norm_stage and my_fa_info:
            n_local_fa = len(my_fa_info)
            recv_shape = (B, n_local_fa, M4, H)
            placeholder = torch.empty(0, device=self._my_device, requires_grad=True)
            local_memory = PipelineRecv.apply(
                placeholder, m2p_norm_stage, recv_shape,
                self._dtype, self._my_device,
                scatter_tag_base + self._my_stage,
            )

        # No pending sends to wait for — all sends are blocking now

        # Save metadata for explicit backward in pipeline_backward()
        if self._my_stage == m2p_norm_stage:
            self._lora_scatter_memory_states = memory_states
            self._lora_scatter_dst_info = lora_scatter_dst_info
            self._lora_scatter_tag_base = scatter_tag_base
            self._lora_scatter_M4 = M4
            self._lora_scatter_H = H
            self._lora_scatter_B = B

        # --- Generate loradict on each stage ---
        loradict: Dict[int, dict] = {}

        if self._my_stage == m2p_norm_stage and my_fa_info:
            for fa_counter, layer_idx in my_fa_info:
                plain_tensor = memory_states[:, fa_counter, :, :].reshape(B, -1)
                loradict[layer_idx] = self._llm_layers[layer_idx].generate_lora_dict(
                    self._lora_ranks, self._generate_lora_scale, plain_tensor,
                )
        elif local_memory is not None:
            for local_idx, (fa_counter, layer_idx) in enumerate(my_fa_info):
                plain_tensor = local_memory[:, local_idx, :, :].reshape(B, -1)
                loradict[layer_idx] = self._llm_layers[layer_idx].generate_lora_dict(
                    self._lora_ranks, self._generate_lora_scale, plain_tensor,
                )

        # Fill in non-full-attention layers on this stage (no LoRA)
        for layer_idx in self._my_layer_indices:
            if layer_idx not in loradict:
                loradict[layer_idx] = None

        # Save generated loradict for monitoring (param_norm monitor reads this)
        self._last_generated_loradict = loradict

        return loradict

    # ==================================================================
    # DetachState helpers
    # ==================================================================

    def _read_detach_state(self, mb_idx=None):
        """Read from detach_state. Returns (nograd_loradict, nograd_wdict).
        Both are None if detach_state is None or EmptyDetachState.

        Args:
            mb_idx: Optional micro-batch index. If provided, returns only
                    the slice for that micro-batch.
        """
        if self.detach_state is None:
            return None, None
        return self.detach_state.read(mb_idx=mb_idx)

    def _write_detach_state(self, loradict, mb_idx=None, precomputed_wdict=None) -> None:
        """Write the generated loradict to detach_state for future use.

        Args:
            loradict: The loradict to write (will be detached internally).
            mb_idx: Optional micro-batch index for batch-indexed writes.
            precomputed_wdict: Optional precomputed new wdict slice (from
                    compute_regu_loss). If provided, skips A@B recomputation.
        """
        if self.detach_state is None:
            return
        self.detach_state.write(loradict, mb_idx=mb_idx, precomputed_wdict=precomputed_wdict)

    # ==================================================================
    # DEPRECATED: pipeline_forward_train is no longer used.
    # Training now exclusively uses pipeline_forward_train_multi_mb.
    # Kept commented out for reference only.
    # ==================================================================
    # def pipeline_forward_train(
    #     self,
    #     context_ids: torch.Tensor,
    #     context_lengths: torch.Tensor,
    #     conversation_ids: torch.Tensor,
    #     batch_size: int,
    #     batch_id: str,
    # ) -> tuple:
    #     """..."""
    #     pass

    # ==================================================================
    # Multi micro-batch pipeline forward for training
    # ==================================================================

    def pipeline_forward_train_multi_mb(
        self,
        context_ids_list: list,
        context_lengths_list: list,
        conversation_ids_list: list,
        micro_batch_size: int,
        batch_id: str,
        distill_conversation_ids_list: list = None,
        distill_micro_batch_size: int = None,
        distill_mode: str = None,
        grad_accum_steps: int = 1,
    ) -> tuple:
        """
        Multi micro-batch pipeline forward that reduces bubble time by
        interleaving micro-batches at stage boundaries.

        The forward is split into five phases (A, A', B, C, C'):
          Phase A:  All micro-batches do Step1 (context → memory_states)
          Phase A': All distill micro-batches do teacher forward (no_grad, no lora)
          Phase B:  All micro-batches do Hypernetwork + lora_scatter
          Phase C:  All micro-batches do Step4 (conversation → hidden_states)
          Phase C': All distill micro-batches do student forward (with grad on loradict)

        Phase A' runs between A and B on LLM GPUs. Since Phase B runs on the
        Hypernetwork GPU, Phase A' and Phase B could overlap in principle, but
        NCCL FIFO on the LLM→Hyper channel requires Phase A to complete first.
        Phase A' uses the same LLM pipeline (tag=0) so it must be strictly
        sequential after Phase A and before Phase C.

        NCCL FIFO constraint: For each pair of ranks (GPU_i, GPU_j), all
        send/recv calls are matched in strict FIFO order. Within each phase,
        micro-batches are processed in order mb0, mb1, ..., mbN-1. This
        guarantees FIFO matching without tags.

        Args:
            context_ids_list:      List of (mb_size, S_ctx) tensors, one per micro-batch.
            context_lengths_list:  List of (mb_size,) tensors, one per micro-batch.
            conversation_ids_list: List of (mb_size, S_conv) tensors, one per micro-batch.
            micro_batch_size:      Size of each micro-batch.
            batch_id:              Unique identifier for this batch.
            distill_conversation_ids_list: List of (distill_mb_size, S_conv) tensors for
                                           distillation. None if distillation is disabled.
            distill_micro_batch_size: Size of each distill micro-batch.
            distill_mode: "logits" or "hidden_states" — determines what teacher
                          output to capture.

        Returns:
            (hidden_states_list, step1_anchor_counts, step4_anchor_counts,
             memory_states_list, distill_teacher_outputs, distill_student_outputs):
                hidden_states_list: List of (mb_size, S_conv, H) on norm stage; [None,...] elsewhere.
                step1_anchor_counts: List of int — per-mb step1 anchor counts.
                step4_anchor_counts: List of int — per-mb step4 anchor counts (including hypernetwork).
                memory_states_list: List of mem_gather state dicts from Phase A.
                distill_teacher_outputs: List of teacher outputs (logits or hidden_states) on norm stage.
                distill_student_outputs: List of student hidden_states on norm stage.
                    Both are None lists if distillation is disabled.
        """
        from utils.myloradict import concat_loradict

        num_mb = len(context_ids_list)
        do_distill = (distill_conversation_ids_list is not None and len(distill_conversation_ids_list) > 0)
        num_distill_mb = len(distill_conversation_ids_list) if do_distill else 0

        # Per-micro-batch state storage
        all_memory_states = []
        all_step1_anchors = []
        all_step4_anchors = []
        all_hidden_states = []

        # Distillation state storage
        distill_teacher_outputs = []
        distill_student_outputs = []

        # =============================================================
        # Phase A: Step1 for all micro-batches
        #   Each micro-batch: context_ids + metalora + nograd → LLM → memory_states
        #   FIFO order on each GPU pair: mb0, mb1, ..., mbN-1
        # =============================================================
        for mb_idx in range(num_mb):
            # Clear anchors before each micro-batch's step1
            self._backward_anchors = []

            # Reset is now done at end of previous step (after set_last_sq_norms)
            ds_nograd_loradict, ds_nograd_wdict = self._read_detach_state(mb_idx=mb_idx)

            # When w_transform_context method is "zero", do not inject detach_state
            # into context forward so that the accumulated W does not affect
            # the LLM + metalora context encoding (hypernetwork generation).
            _ctx_nograd_loradict = ds_nograd_loradict
            _ctx_nograd_wdict = ds_nograd_wdict
            if self.detach_state is not None:
                _ctx_cfg = self.detach_state._cfg
                _ctx_method = "identity"
                if isinstance(_ctx_cfg.get("w_transform_context"), dict):
                    _ctx_method = _ctx_cfg["w_transform_context"].get("method", "identity")
                if _ctx_method == "zero":
                    _ctx_nograd_loradict = None
                    _ctx_nograd_wdict = None

            _, memory_states = self.llm_forward_with_grad(
                input_ids=context_ids_list[mb_idx],
                attention_mask=None,
                loradict=self.metalora,
                context_lengths=context_lengths_list[mb_idx],
                use_mem_token=True,
                batch_id=f"{batch_id}_ctx_mb{mb_idx}",
                nograd_loradict=_ctx_nograd_loradict,
                nograd_wdict=_ctx_nograd_wdict,
            )

            # Save step1 anchors and state for this micro-batch
            all_step1_anchors.append(list(self._backward_anchors))
            self._backward_anchors = []

            # Save mem_gather state per micro-batch
            all_memory_states.append({
                'memory_states': memory_states,
                'mem_gather_local_memory': getattr(self, '_mem_gather_local_memory', None),
                'mem_gather_src_info': getattr(self, '_mem_gather_src_info', None),
                'mem_gather_assembled': getattr(self, '_mem_gather_assembled', None),
            })
            # Reset so next micro-batch gets fresh state
            self._mem_gather_local_memory = None
            self._mem_gather_src_info = None
            self._mem_gather_assembled = None

        # =============================================================
        # Phase A': Teacher forward for distillation (no_grad, no lora)
        #   Each distill micro-batch: distill_conv_ids → LLM (no lora) → teacher output
        #   FIFO order on each GPU pair: mb0, mb1, ..., mbN-1
        #   This runs on LLM GPUs while Hyper GPU is idle (before Phase B).
        # =============================================================
        if do_distill:
            with torch.no_grad():
                for mb_idx in range(num_distill_mb):
                    teacher_hidden, _ = self.llm_forward_no_grad(
                        input_ids=distill_conversation_ids_list[mb_idx],
                        attention_mask=None,
                        loradict=None,
                        use_mem_token=False,
                        batch_id=f"{batch_id}_distill_teacher_mb{mb_idx}",
                    )

                    # On norm stage: capture teacher output based on mode
                    if teacher_hidden is not None:
                        if distill_mode == "logits":
                            # Compute logits from hidden states
                            teacher_logits = self.llm.lm_head(teacher_hidden)
                            distill_teacher_outputs.append(teacher_logits.detach())
                        else:
                            # hidden_states mode: keep hidden states directly
                            distill_teacher_outputs.append(teacher_hidden.detach())
                    else:
                        distill_teacher_outputs.append(None)

        # =============================================================
        # Phase B: Hypernetwork + lora_scatter for all micro-batches
        #   Each micro-batch: memory_states → Hypernetwork → loradict
        #   FIFO order on GPU6→GPU7: mb0, mb1, ..., mbN-1
        #   FIFO order on GPU7→each GPU_i (lora_scatter): mb0, mb1, ..., mbN-1
        # =============================================================
        all_loradicts = []
        all_raw_loradicts = []  # Raw loradicts without metalora (for detach_state)
        for mb_idx in range(num_mb):
            self._backward_anchors = []

            mem_state = all_memory_states[mb_idx]['memory_states']
            loradict = self.hypernetwork_forward_with_grad(
                memory_states=(
                    mem_state if mem_state is not None
                    else torch.empty(0, device=self._my_device)
                ),
                batch_size=micro_batch_size,
            )

            # Save raw loradict (without metalora) for detach_state write/regu
            all_raw_loradicts.append(loradict)

            # Concat loradict and metalora → new_loradict
            if loradict is not None:
                new_loradict = {}
                for layer_idx in self._my_layer_indices:
                    layer_lora = loradict.get(layer_idx, None)
                    layer_meta = self.metalora.get(layer_idx, None)
                    new_loradict[layer_idx] = concat_loradict([layer_lora, layer_meta])
            else:
                new_loradict = self.metalora

            all_loradicts.append(new_loradict)

            # Save hypernetwork + lora_scatter anchors
            # Also save lora_scatter state per micro-batch
            all_step4_anchors.append({
                'anchors': list(self._backward_anchors),
                'lora_scatter_memory_states': getattr(self, '_lora_scatter_memory_states', None),
                'lora_scatter_dst_info': getattr(self, '_lora_scatter_dst_info', None),
                'lora_scatter_tag_base': getattr(self, '_lora_scatter_tag_base', None),
                'lora_scatter_M4': getattr(self, '_lora_scatter_M4', None),
                'lora_scatter_H': getattr(self, '_lora_scatter_H', None),
                'lora_scatter_B': getattr(self, '_lora_scatter_B', None),
            })
            self._backward_anchors = []
            self._lora_scatter_memory_states = None
            self._lora_scatter_dst_info = None

        # =============================================================
        # Phase C: Step4 for all micro-batches
        #   Each micro-batch: conversation_ids + new_loradict + nograd → LLM → hidden_states
        #   FIFO order on each GPU pair: mb0, mb1, ..., mbN-1
        # =============================================================
        for mb_idx in range(num_mb):
            self._backward_anchors = []

            ds_nograd_loradict, ds_nograd_wdict = self._read_detach_state(mb_idx=mb_idx)

            hidden_states, _ = self.llm_forward_with_grad(
                input_ids=conversation_ids_list[mb_idx],
                attention_mask=None,
                loradict=all_loradicts[mb_idx],
                use_mem_token=False,
                batch_id=f"{batch_id}_conv_mb{mb_idx}",
                nograd_loradict=ds_nograd_loradict,
                nograd_wdict=ds_nograd_wdict,
            )

            all_hidden_states.append(hidden_states)

            # Append step4 LLM anchors to the step4 anchor dict
            all_step4_anchors[mb_idx]['step4_llm_anchors'] = list(self._backward_anchors)
            self._backward_anchors = []

        # =============================================================
        # Phase C': Student forward for distillation (with grad on loradict)
        #   Each distill micro-batch: distill_conv_ids + new_loradict → LLM → student output
        #   FIFO order on each GPU pair: mb0, mb1, ..., mbN-1
        #   Gradients flow through loradict → hypernetwork → metalora.
        #
        #   We reuse the same loradicts from Phase B. If num_distill_mb differs
        #   from num_mb, we cycle through loradicts (typically they are equal).
        # =============================================================
        if do_distill:
            for mb_idx in range(num_distill_mb):
                self._backward_anchors = []

                # Map distill micro-batch to the corresponding loradict
                # (distill_local_batch_size == local_batch_size is enforced,
                #  so num_distill_mb should equal num_mb when micro_batch_sizes match)
                loradict_idx = mb_idx % num_mb

                ds_nograd_loradict, ds_nograd_wdict = self._read_detach_state(mb_idx=loradict_idx)

                student_hidden, _ = self.llm_forward_with_grad(
                    input_ids=distill_conversation_ids_list[mb_idx],
                    attention_mask=None,
                    loradict=all_loradicts[loradict_idx],
                    use_mem_token=False,
                    batch_id=f"{batch_id}_distill_student_mb{mb_idx}",
                    nograd_loradict=ds_nograd_loradict,
                    nograd_wdict=ds_nograd_wdict,
                )

                distill_student_outputs.append(student_hidden)

                # Save distill student anchors into step4_anchors for backward
                # We append them to the corresponding loradict's anchor dict
                all_step4_anchors[loradict_idx].setdefault('distill_student_anchors', []).append(
                    list(self._backward_anchors)
                )
                self._backward_anchors = []

            # Deferred write: write all detach_state slices after Phase C'
            # completes, so that Phase C' reads the same (unmodified) wdict
            # as Phase C did.
            # (Moved below to unified regu_loss + write section)

        # =============================================================
        # Regularization loss + Deferred write (unified for both distill/non-distill)
        #
        # After all forward phases complete (Phase C and optionally C'),
        # compute regu gradients and register hooks on loradict's A/B/C,
        # then write the new wdict.
        #
        # compute_regu_loss uses torch.autograd.grad to compute regu gradients
        # w.r.t. loradict tensors, then registers one-shot hooks so that
        # regu gradients are injected during the subsequent CE/distill backward.
        # No separate backward() call is needed (avoids PP deadlocks).
        #
        # compute_regu_loss also returns precomputed_wdict = detach(W_old + A@B),
        # reused by write to avoid redundant A@B computation.
        #
        # Returns per-mb sq_norms (local to this stage) for node-level sync.
        # =============================================================
        per_mb_sq_norms = []  # Per-mb unscaled ||W_old + A@B||² (this stage only)
        if self.detach_state is not None:
            for mb_idx in range(num_mb):
                # Use raw loradict (without metalora) for detach_state
                unscaled_sq_norm, _regu_loss_tensor, precomputed = self.detach_state.compute_regu_loss(
                    all_raw_loradicts[mb_idx], mb_idx, num_mb, grad_accum_steps
                )
                per_mb_sq_norms.append(unscaled_sq_norm if unscaled_sq_norm is not None else 0.0)

                # Write with precomputed wdict (skips redundant A@B)
                self._write_detach_state(all_raw_loradicts[mb_idx], mb_idx=mb_idx,
                                         precomputed_wdict=precomputed)
        else:
            # No detach_state — nothing to write or regularize
            per_mb_sq_norms = [0.0] * num_mb

        return (all_hidden_states, all_step1_anchors, all_step4_anchors,
                all_memory_states, distill_teacher_outputs, distill_student_outputs,
                per_mb_sq_norms)

    def pipeline_backward_multi_mb(
        self,
        losses: list,
        all_step1_anchors: list,
        all_step4_anchors: list,
        all_memory_states: list,
        distill_losses: list = None,
    ):
        """
        Multi micro-batch backward that processes micro-batches in reverse
        order to match the FIFO constraint.

        Backward is split into phases (reverse of forward):
          Phase C' backward: Distill student LLM backward (if distillation enabled)
          Phase C+B backward: Step4 LLM backward + lora_scatter + hypernetwork backward
          Phase A backward: Step1 LLM backward + mem_gather backward

        Phase C' must run BEFORE Phase C+B because:
          1. Phase C' was sent AFTER Phase C in forward, so its backward
             recv must happen first (FIFO order).
          2. On LLM stages, Phase C and C' share the loradict computation
             graph. Phase C' backward uses retain_graph=True to preserve
             the shared intermediate values for Phase C backward.

        On the norm stage, Phase C' backward calls distill_loss.backward(retain_graph=True)
        to trigger gradient sending to the previous stage and to loradict/hypernetwork.
        Phase C+B backward then calls CE_loss.backward() which propagates CE gradients.
        Both passes trigger lora_scatter backward (recv lora grads from LLM stages),
        so the hypernetwork receives gradients from both CE and distill losses.

        Within each phase, micro-batches are processed in REVERSE order
        (mbN-1, ..., mb1, mb0) because backward sends gradients in the
        opposite direction of forward, and FIFO requires matching order.

        Args:
            losses:            List of CE loss tensors (one per micro-batch),
                               on norm stage. None on other stages.
            all_step1_anchors: List of anchor lists from Phase A.
            all_step4_anchors: List of anchor dicts from Phase B+C+C'.
            all_memory_states: List of mem_gather state dicts from Phase A.
            distill_losses:    List of distill loss tensors (one per micro-batch),
                               on norm stage. None on other stages or when distill is disabled.
                               Used for Phase C' backward on norm stage.
        """
        num_mb = len(losses)
        m2p_norm_stage = self.hypernetwork._m2p_norm_stage

        # =============================================================
        # Phase C' backward: Distill student LLM backward
        #   Process in REVERSE micro-batch order for FIFO matching.
        #   Only runs if distill_student_anchors exist in step4_anchors.
        #
        #   On LLM stages, Phase C and Phase C' share the loradict
        #   computation graph (both use the same loradict tensors received
        #   via lora_scatter). Therefore Phase C' backward must use
        #   retain_graph=True to preserve the shared intermediate values
        #   for the subsequent Phase C backward.
        #
        #   On the norm stage, distill_losses[mb_idx].backward(retain_graph=True)
        #   is called to trigger gradient sending to the previous stage.
        #   retain_graph=True is needed because the combined loss (CE + distill)
        #   will be backward'd again in Phase C+B.
        # =============================================================
        has_distill = any(
            'distill_student_anchors' in step4_info
            for step4_info in all_step4_anchors
        )
        if has_distill:
            # Process in reverse mb order; within each mb, process distill
            # anchors in reverse order (last distill mb first)
            for mb_idx in reversed(range(num_mb)):
                step4_info = all_step4_anchors[mb_idx]
                distill_anchors_list = step4_info.get('distill_student_anchors', [])

                # On norm stage: backward the distill loss to trigger gradient sending
                d_loss = distill_losses[mb_idx] if distill_losses is not None else None

                # Set up lora_scatter state so m2p_norm_stage can recv lora grads
                # that are sent by LLM stages' PipelineRecv.backward (triggered
                # when autograd propagates through loradict in Phase C' backward).
                self._lora_scatter_memory_states = step4_info['lora_scatter_memory_states']
                self._lora_scatter_dst_info = step4_info['lora_scatter_dst_info']
                self._lora_scatter_tag_base = step4_info.get('lora_scatter_tag_base')
                self._lora_scatter_M4 = step4_info.get('lora_scatter_M4')
                self._lora_scatter_H = step4_info.get('lora_scatter_H')
                self._lora_scatter_B = step4_info.get('lora_scatter_B')

                if distill_anchors_list:
                    # Non-norm stages (and intermediate stages): process anchors
                    for distill_anchors in reversed(distill_anchors_list):
                        self._backward_anchors = distill_anchors
                        # retain_graph=True because the loradict computation graph
                        # is shared with Phase C on LLM stages, AND on norm stage
                        # the combined loss will be backward'd again in Phase C+B.
                        self.pipeline_backward(d_loss, retain_graph=True)
                        # Only pass d_loss on the first anchor iteration for this mb;
                        # subsequent anchors in the same mb don't need loss again
                        d_loss = None
                elif d_loss is not None:
                    # Norm stage: no distill anchors (it's the last stage, no send),
                    # but must backward d_loss to trigger PipelineRecv.backward
                    # which sends gradients to the previous stage.
                    self._backward_anchors = []
                    self.pipeline_backward(d_loss, retain_graph=True)

        # =============================================================
        # Phase C+B backward: Step4 LLM + lora_scatter + hypernetwork
        #   Process in REVERSE micro-batch order for FIFO matching.
        #   For each mb: loss.backward → step4 anchors → lora_scatter backward
        #
        #   On the norm stage, losses[mb_idx] contains ONLY the CE loss.
        #   CE_loss.backward() propagates CE gradients through Phase C's
        #   computation graph, triggering lora_scatter send from each LLM stage.
        #   The m2p_norm_stage recvs these lora grads and calls
        #   lora_mem.backward(retain_graph=False) to finalize hypernetwork backward.
        # =============================================================
        for mb_idx in reversed(range(num_mb)):
            step4_info = all_step4_anchors[mb_idx]

            # Combine hypernetwork anchors and step4 LLM anchors
            combined_anchors = step4_info['anchors'] + step4_info.get('step4_llm_anchors', [])

            # Set up model state for this micro-batch's lora_scatter backward
            self._lora_scatter_memory_states = step4_info['lora_scatter_memory_states']
            self._lora_scatter_dst_info = step4_info['lora_scatter_dst_info']
            self._lora_scatter_tag_base = step4_info.get('lora_scatter_tag_base')
            self._lora_scatter_M4 = step4_info.get('lora_scatter_M4')
            self._lora_scatter_H = step4_info.get('lora_scatter_H')
            self._lora_scatter_B = step4_info.get('lora_scatter_B')

            # Set anchors and run pipeline_backward
            self._backward_anchors = combined_anchors
            self.pipeline_backward(losses[mb_idx], retain_graph=False)

        # =============================================================
        # Phase A backward: Step1 LLM + mem_gather
        #   Process in REVERSE micro-batch order for FIFO matching.
        #   For each mb: mem_gather grad send/recv + step1 anchors backward
        # =============================================================
        for mb_idx in reversed(range(num_mb)):
            # Restore mem_gather state for this micro-batch
            mem_state = all_memory_states[mb_idx]
            self._mem_gather_local_memory = mem_state['mem_gather_local_memory']
            self._mem_gather_src_info = mem_state['mem_gather_src_info']
            self._mem_gather_assembled = mem_state['mem_gather_assembled']

            # Set step1 anchors and run pipeline_backward_step1
            self._backward_anchors = all_step1_anchors[mb_idx]
            self.pipeline_backward_step1()

    # ==================================================================
    # Checkpoint save / load
    # ==================================================================

    def save_model(self, save_dir: str):
        """
        Save all trainable parameters (hypernetwork + metalora + mem_tokens)
        to the given directory in a **pipeline-stage-independent** format.

        Uses safetensors format for fast mmap-based loading.
        All tensors from this pipeline stage are saved into a single file
        ``model_stage{S}.safetensors`` to minimize NFS IOPS.

        Args:
            save_dir: Directory path to save the model checkpoint.
        """
        import os
        from safetensors.torch import save_file

        os.makedirs(save_dir, exist_ok=True)

        # Collect all tensors into a flat dict with unique keys
        tensors_dict: dict = {}

        # --- Hypernetwork parameters ---
        for name, param in self.hypernetwork.named_parameters():
            if param.device == self._my_device:
                # Strip _orig_mod segments introduced by torch.compile wrappers
                clean_name = name.replace("._orig_mod.", ".")
                if clean_name.startswith("_orig_mod."):
                    clean_name = clean_name[len("_orig_mod."):]
                tensors_dict[f"hypernet.{clean_name}"] = param.data.cpu()

        # --- Metalora tensors ---
        from utils.myloradict import collect_loradict_tensors
        for layer_idx, layer_lora in self.metalora.items():
            tensors = collect_loradict_tensors(layer_lora)
            for t_idx, t in enumerate(tensors):
                tensors_dict[f"metalora.layer{layer_idx}.tensor{t_idx}"] = t.data.cpu()

        # --- Mem tokens (only on embed stage) ---
        if self._my_stage == self._embed_stage and hasattr(self._llm_model, "mem_tokens"):
            mem_tokens = self._llm_model.mem_tokens
            if mem_tokens is not None:
                tensors_dict["mem_tokens"] = mem_tokens.data.cpu()

        # Save all tensors in a single safetensors file (per stage)
        if tensors_dict:
            stage = self._my_stage
            save_file(tensors_dict, os.path.join(save_dir, f"model_stage{stage}.safetensors"))

    def load_model(self, load_dir: str):
        """
        Load trainable parameters from a model checkpoint directory.

        This method is **pipeline-stage-independent**: it loads parameters
        by content (param name / layer index), not by pipeline stage.
        Each stage only loads the parameters that belong to it.

        Only tensors belonging to the current stage are loaded into GPU
        memory — other stages' tensors are skipped entirely via
        ``safe_open`` selective reading.

        Format: ``model_stage*.safetensors`` (fast mmap loading)

        Args:
            load_dir: Directory path containing the model checkpoint.
        """
        import os
        import glob
        from safetensors import safe_open
        from utils.myloradict import collect_loradict_tensors
        import re

        st_files = sorted(glob.glob(os.path.join(load_dir, "model_stage*.safetensors")))
        if not st_files:
            raise FileNotFoundError(
                f"No model_stage*.safetensors files found in {load_dir}. "
                f"Run scripts/convert_checkpoints_to_safetensors.py to convert legacy checkpoints."
            )

        # --- Build the set of keys this stage needs ---
        needed_keys: set = set()

        # Hypernetwork params on this device
        # Strip _orig_mod from parameter names (torch.compile wrapper artifact)
        param_dict = dict(self.hypernetwork.named_parameters())
        # Build a mapping: clean_name -> original_name (for restoring later)
        _clean_to_orig: dict = {}
        for name, param in param_dict.items():
            if param.device == self._my_device:
                clean_name = name.replace("._orig_mod.", ".")
                if clean_name.startswith("_orig_mod."):
                    clean_name = clean_name[len("_orig_mod."):]
                needed_keys.add(f"hypernet.{clean_name}")
                _clean_to_orig[clean_name] = name

        # Metalora tensors on this stage
        for layer_idx, layer_lora in self.metalora.items():
            tensors = collect_loradict_tensors(layer_lora)
            for t_idx in range(len(tensors)):
                needed_keys.add(f"metalora.layer{layer_idx}.tensor{t_idx}")

        # Mem tokens (only on embed stage)
        if self._my_stage == self._embed_stage and hasattr(self._llm_model, "mem_tokens"):
            if self._llm_model.mem_tokens is not None:
                needed_keys.add("mem_tokens")

        # --- Selectively load only needed tensors from all stage files ---
        # Checkpoint keys may or may not contain _orig_mod; normalise them.
        device_str = str(self._my_device)
        loaded_tensors: dict = {}
        all_checkpoint_keys: set = set()  # All normalised keys in checkpoint
        for fi, f in enumerate(st_files):
            with safe_open(f, framework="pt", device=device_str) as sf:
                for raw_key in sf.keys():
                    normalised_key = raw_key.replace("._orig_mod.", ".")
                    if normalised_key.startswith("hypernet._orig_mod."):
                        normalised_key = "hypernet." + normalised_key[len("hypernet._orig_mod."):]
                    all_checkpoint_keys.add(normalised_key)
                    if normalised_key in needed_keys:
                        loaded_tensors[normalised_key] = sf.get_tensor(raw_key)

        # --- Restore hypernetwork parameters ---
        loaded_hypernet = 0
        for key, tensor in loaded_tensors.items():
            if key.startswith("hypernet."):
                clean_name = key[len("hypernet."):]
                orig_name = _clean_to_orig.get(clean_name)
                if orig_name and orig_name in param_dict:
                    param_dict[orig_name].data.copy_(tensor)
                    loaded_hypernet += 1

        # --- Restore metalora tensors ---
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

        # --- Restore mem_tokens ---
        loaded_mem = 0
        if "mem_tokens" in loaded_tensors:
            self._llm_model.mem_tokens.data.copy_(loaded_tensors["mem_tokens"])
            loaded_mem = 1

        # --- Strict check: fail loudly if any expected parameters are missing ---
        # Count loaded metalora tensors
        loaded_metalora = 0
        for layer_idx, tensor_map in metalora_by_layer.items():
            if layer_idx in self.metalora:
                current_tensors = collect_loradict_tensors(self.metalora[layer_idx])
                for t_idx in range(len(current_tensors)):
                    if t_idx in tensor_map:
                        loaded_metalora += 1

        expected_hypernet = len(_clean_to_orig)
        expected_metalora = sum(
            len(collect_loradict_tensors(self.metalora[li]))
            for li in self.metalora
        )
        expected_mem = 1 if (
            self._my_stage == self._embed_stage
            and hasattr(self._llm_model, "mem_tokens")
            and self._llm_model.mem_tokens is not None
        ) else 0

        missing_hypernet = expected_hypernet - loaded_hypernet
        missing_metalora = expected_metalora - loaded_metalora
        missing_mem = expected_mem - loaded_mem

        if missing_hypernet > 0 or missing_metalora > 0 or missing_mem > 0:
            missing_details = []
            if missing_hypernet > 0:
                loaded_names = set()
                for key in loaded_tensors:
                    if key.startswith("hypernet."):
                        loaded_names.add(key[len("hypernet."):])
                unloaded = sorted(set(_clean_to_orig.keys()) - loaded_names)
                missing_details.append(
                    f"hypernet: {missing_hypernet}/{expected_hypernet} params NOT loaded. "
                    f"First 5 missing: {unloaded[:5]}"
                )
            if missing_metalora > 0:
                missing_details.append(
                    f"metalora: {missing_metalora}/{expected_metalora} tensors NOT loaded"
                )
            if missing_mem > 0:
                missing_details.append("mem_tokens: NOT loaded")
            raise RuntimeError(
                f"[ModelHypernetwork.load_model] STRICT LOAD FAILED on stage {self._my_stage} — "
                f"some parameters were not found in checkpoint.\n"
                + "\n".join(f"  • {d}" for d in missing_details)
                + f"\n  Scanned files: {[os.path.basename(f) for f in st_files]}"
                + "\n  Hint: checkpoint may have incompatible key names or missing stages."
            )

        # Check for unexpected keys: keys in checkpoint that don't belong to
        # any known namespace. In PP mode, other stages' keys are expected in
        # the checkpoint, so we only flag keys with unrecognised prefixes.
        _KNOWN_PREFIXES = ("hypernet.", "metalora.", "mem_tokens", "w_transform.")
        unexpected_keys = sorted(
            k for k in all_checkpoint_keys
            if not any(k.startswith(p) or k == p for p in _KNOWN_PREFIXES)
        )
        if unexpected_keys:
            raise RuntimeError(
                f"[ModelHypernetwork.load_model] STRICT LOAD FAILED on stage {self._my_stage} — "
                f"checkpoint contains {len(unexpected_keys)} key(s) with unrecognised "
                f"namespace (not hypernet/metalora/mem_tokens/w_transform).\n"
                f"  First 10 unexpected: {unexpected_keys[:10]}\n"
                f"  Scanned files: {[os.path.basename(f) for f in st_files]}\n"
                f"  Hint: the checkpoint may have been saved with a different model "
                f"configuration or contains corrupted keys."
            )

        if is_main_process_per_node():
            logger.info(
                f"[ModelHypernetwork.load_model] Loaded on stage {self._my_stage}: "
                f"hypernet={loaded_hypernet}/{expected_hypernet}, "
                f"metalora={loaded_metalora}/{expected_metalora} tensors, "
                f"mem_tokens={'yes' if loaded_mem else 'n/a'} "
                f"(scanned {len(st_files)} stage file(s))"
            )
