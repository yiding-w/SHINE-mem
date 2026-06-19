"""Thin wrapper around SHINE's Metanetwork for ES training.

Mirrors the construction flow in third_party/SHINE/inference.ipynb exactly.
Exposes a flat list of perturbable parameters for ES and simple generate_lora /
answer calls. Nothing here trains; training lives in meta_past.es.trainer.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer


def _ensure_shine_on_path() -> None:
    """Make SHINE's modules importable.

    SHINE's source tree uses ``utils/`` and other directories as PEP 420
    namespace packages (no ``__init__.py``). Python's namespace finder
    can fail to resolve ``utils.<sub>`` reliably under some launcher /
    multi-package conditions — and worse, even when our own
    ``utils.<sub>`` import works, transitive imports inside SHINE
    (``myinit`` → ``utils.myddp`` → ...) re-hit the finder and fail.

    Fix: plant a fully-formed ``utils`` package object in ``sys.modules``
    pointing at ``SHINE/utils``. Any subsequent ``import utils.<sub>``
    (ours or SHINE's internal) finds the cached package and resolves
    submodules under it via the regular import machinery.
    """
    import types

    shine_root = Path(__file__).resolve().parent.parent / "third_party" / "SHINE"
    if not shine_root.is_dir():
        raise FileNotFoundError(
            f"SHINE submodule not found at {shine_root}. "
            "Did you run `git submodule update --init --recursive`?"
        )
    utils_dir = shine_root / "utils"
    if not (utils_dir / "myinit.py").is_file():
        raise FileNotFoundError(
            f"SHINE/utils/myinit.py not found under {shine_root}. "
            "The submodule appears incomplete; try "
            "`git submodule update --init --recursive`."
        )
    p = str(shine_root)
    if p not in sys.path:
        sys.path.insert(0, p)

    # Build a synthetic ``utils`` package whose ``__path__`` points at
    # SHINE's utils dir. Replace any prior ``utils`` (likely an empty
    # namespace package) and drop its cached submodules so re-imports
    # fall through to the new package. ``__path__`` makes
    # ``import utils.X`` look for ``SHINE/utils/X.py``.
    existing_paths = []
    if "utils" in sys.modules:
        existing_paths = list(getattr(sys.modules["utils"], "__path__", []) or [])
        for stale in [k for k in list(sys.modules)
                      if k == "utils" or k.startswith("utils.")]:
            del sys.modules[stale]
    pkg = types.ModuleType("utils")
    paths = [str(utils_dir)]
    for ep in existing_paths:
        if str(ep) not in paths and Path(ep).exists():
            paths.append(str(ep))
    pkg.__path__ = paths  # type: ignore[attr-defined]
    pkg.__file__ = str(utils_dir / "__init__.py")  # nominal; file may not exist
    sys.modules["utils"] = pkg

    import importlib
    importlib.invalidate_caches()


def _import_shine_utils_module(name: str):
    """Load ``utils.<name>`` from SHINE via the standard import machinery.

    Relies on ``_ensure_shine_on_path`` having planted a real ``utils``
    package object pointing at ``SHINE/utils``.
    """
    import importlib

    full_name = f"utils.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    return importlib.import_module(full_name)


def _default_cfg(
    backbone: str,
    lora_r: int,
    metalora_r: int,
    metanet_num_layers: int = 4,
    scale: float = 0.001,
    method: str = "rl",
) -> Any:
    """Build the OmegaConf dict that SHINE's Metanetwork expects.

    Only the fields actually read by Metanetwork + MetanetworkTransformer are
    populated. Defaults copied from configs/Qwen3-8B.yaml.
    """
    return OmegaConf.create(
        {
            "model": {
                "lora_r": lora_r,
                "metalora_r": metalora_r,
                "metamodel_class_path": "LoraQwen.LoraQwen3ForCausalLM",
                "config_class_path": "LoraQwen.Qwen3Config",
                "tokenizer_from": backbone,
                "model_from": backbone,
            },
            "metanetwork": {
                "type": "transformer",
                "method": method,
                "transformer_cfg": {
                    "encoder_cfg": {
                        "d_model": 4096,
                        "nhead": 32,
                        "dim_feedforward": 8192,
                        "dropout": 0,
                        "activation": "gelu",
                        "layer_norm_eps": 1e-5,
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
                        "layer_norm_eps": 1e-5,
                        "batch_first": True,
                        "norm_first": False,
                        "bias": True,
                    },
                    "layer_transformer_first": True,
                    "mean_pool_size": 1,
                    "num_layers": metanet_num_layers,
                    "couple_num_layers": 0,
                    "scale": scale,
                },
            },
            "hidden_size": -1,
            "num_layers": -1,
            "num_mem_token": -1,
        }
    )


# ---------------------------------------------------------------------------
# Helpers for SHINE loradict microbatching (concat / slice along Lb dim).
# ---------------------------------------------------------------------------


def _slice_loradict_lb(loradict: dict, start: int, end: int) -> dict:
    """Return a new loradict where each leaf tensor is sliced ``[start:end]`` on dim 0.

    SHINE loradict layout:
        {layer_i: {"attention": {"q"|"k"|"v"|"o": {"A": [Lb, ...], "B": [Lb, ...], "C": ...}},
                   "mlp":       {"gate"|"up"|"down": {"A": [...], "B": [...], "C": ...}}}}
    """
    out: dict = {}
    for layer_i, layer_dict in loradict.items():
        out_layer = {}
        for sub_key in ("attention", "mlp"):
            out_sub = {}
            for proj_key, proj_entry in layer_dict[sub_key].items():
                A = proj_entry["A"][start:end]
                B = proj_entry["B"][start:end]
                C = proj_entry.get("C", None)
                if C is not None:
                    C = C[start:end]
                out_sub[proj_key] = {"A": A, "B": B, "C": C}
            out_layer[sub_key] = out_sub
        out[layer_i] = out_layer
    return out


def _concat_loradicts_lb(loradicts: list[dict]) -> dict:
    """Concat a list of loradict chunks along the Lb dim. Layout as above."""
    if not loradicts:
        raise ValueError("_concat_loradicts_lb: empty input.")
    if len(loradicts) == 1:
        return loradicts[0]
    template = loradicts[0]
    out: dict = {}
    for layer_i, layer_dict in template.items():
        out_layer = {}
        for sub_key in ("attention", "mlp"):
            out_sub = {}
            for proj_key in layer_dict[sub_key].keys():
                A = torch.cat([d[layer_i][sub_key][proj_key]["A"] for d in loradicts], dim=0)
                B = torch.cat([d[layer_i][sub_key][proj_key]["B"] for d in loradicts], dim=0)
                Cs = [d[layer_i][sub_key][proj_key].get("C", None) for d in loradicts]
                C = None if all(c is None for c in Cs) else torch.cat(Cs, dim=0)
                out_sub[proj_key] = {"A": A, "B": B, "C": C}
            out_layer[sub_key] = out_sub
        out[layer_i] = out_layer
    return out


class ShineHypernet:
    """Holds the SHINE hypernetwork stack and exposes ES-friendly handles.

    The constructor mirrors ``inference.ipynb`` verbatim:

      1. Import ``LoraQwen3ForCausalLM`` / ``Qwen3Config`` via SHINE's dynamic
         import path.
      2. Load the config, set ``num_mem_token = -1`` temporarily.
      3. Use a ``torch.device("meta")`` temp model to compute
         ``num_mem_token = lora_params_numel(lora_r) * mean_pool_size /
         (hidden_size * num_layers)``. For Qwen3-8B with ``lora_r=8`` this is 148.
      4. Load tokenizer, append ``<RECON> <COMP> <NOTHING>``, install SHINE's
         custom chat template (needed so the added special tokens serialize
         correctly).
      5. Load the full backbone, ``reset_mem_tokens()``,
         ``resize_token_embeddings(len(tokenizer))``.
      6. Wrap in ``Metanetwork(metamodel, cfg, lora_params_numel)``.
      7. ``load_checkpoint`` — returns (metanetwork, metalora, _). The loader
         also re-calls ``freeze(metamodel)`` after moving tensors to device.
    """

    def __init__(
        self,
        ckpt_dir: str,
        device: str = "cuda",
        backbone: str = "Qwen/Qwen3-8B",
        lora_r: int = 8,
        metalora_r: int = 128,
        metanet_num_layers: int = 4,
        scale: float = 0.001,
        method: str = "rl",
    ):
        _ensure_shine_on_path()
        # Load SHINE's namespace-package ``utils`` submodules via direct
        # file-spec import — robust to any finder-cache weirdness across
        # the launcher's load order.
        _import_class = _import_shine_utils_module("myinit")._import_class
        load_checkpoint = _import_shine_utils_module("mysaveload").load_checkpoint
        freeze = _import_shine_utils_module("myfreeze").freeze
        # ``metanetwork_family`` is at SHINE root and IS importable normally
        # once SHINE is on sys.path.
        from metanetwork_family import Metanetwork  # type: ignore

        self._load_checkpoint = load_checkpoint
        self._freeze = freeze

        self.device = torch.device(device)
        cfg = _default_cfg(
            backbone=backbone,
            lora_r=lora_r,
            metalora_r=metalora_r,
            metanet_num_layers=metanet_num_layers,
            scale=scale,
            method=method,
        )

        MetaModelCls = _import_class(cfg.model.metamodel_class_path)
        ConfigCls = _import_class(cfg.model.config_class_path)

        config = ConfigCls.from_pretrained(cfg.model.model_from)
        config.num_mem_token = -1
        cfg.hidden_size = config.hidden_size
        cfg.num_layers = config.num_hidden_layers

        with torch.device("meta"):
            tmp_model = MetaModelCls(config)
        lora_numel = tmp_model.lora_params_numel(cfg.model.lora_r)
        base = cfg.hidden_size * cfg.num_layers
        mean_pool = cfg.metanetwork.transformer_cfg.mean_pool_size
        assert lora_numel % base == 0, (
            f"lora_params_numel ({lora_numel}) must divide hidden*layers ({base})"
        )
        config.num_mem_token = lora_numel * mean_pool // base
        cfg.num_mem_token = config.num_mem_token
        del tmp_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.tokenizer_from, padding_side="left", use_fast=True
        )
        tokenizer.add_tokens(["<RECON>", "<COMP>", "<NOTHING>"])
        # Note: we deliberately do NOT override tokenizer.chat_template here.
        # The stock Qwen3 template's enable_thinking flag (False → inject empty
        # <think></think> stub; True/None → no stub, model decides) covers
        # everything SHINE's original wholesale override was trying to do.

        metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
        metamodel.reset_mem_tokens()
        metamodel.resize_token_embeddings(len(tokenizer))

        metanetwork = Metanetwork(
            metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r)
        )
        metanetwork.to(self.device)
        freeze(metamodel)

        metanetwork, metalora, _ = load_checkpoint(metanetwork, ckpt_dir, device)

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.metanetwork = metanetwork
        self.metalora = metalora
        self.num_mem_token = int(config.num_mem_token)
        self.lora_r = int(cfg.model.lora_r)

    # -- perturbation surface --------------------------------------------------

    def all_perturbable_params(
        self,
        include_mem_tokens: bool = False,
        include_metalora: bool = True,
        min_rms: float = 0.0,
        exclude_bias: bool = True,
    ) -> list[tuple[str, torch.Tensor]]:
        """Return (name, tensor) pairs for every ES-perturbable tensor.

        Default scope (``include_mem_tokens=False``):

        * M2P transformer:   ``metanetwork.metanetwork.parameters()``
        * metalora leaves:   ``iter_learnable_tensors(metalora)``

        **mem_tokens is excluded by default.** The SHINE-ift_mqa_1qa checkpoint
        ships with ``mem_tokens = 0`` (its optimizer in DDP training filtered
        out ``module.metamodel.*`` and so never updated them). The downstream
        M2P was trained to read memory-states derived from attention-on-zero
        queries — any perturbation of mem_tokens breaks that assumption and
        sends F1 to ~0 even at σ=1e-4. See ``scripts/debug_subgroup_sweep.py``
        for the measurement; the proposal's "perturb all of m_h" advice
        (§3.2) does not survive this singularity.

        Frozen backbone tensors (everything else under
        ``metanetwork.metamodel``) remain excluded. Names are stable so the
        ES seed-shift maps each call to the same tensor.

        ``include_metalora=False`` is useful for smoke runs: the 504 metalora
        leaves have RMS ~0.01–0.03 each and under ES contribute mostly noise
        (SNR ∝ RMS_k under per-tensor signal), causing random-walk drift
        that dominates the m2p gradient signal. See run #7 (heldout 0.658 →
        0.575 in 10 steps with include_metalora=True at α=0.01).

        ``min_rms`` filters out tensors whose RMS falls below the threshold.
        A cheaper way to get the same SNR improvement than dropping metalora
        wholesale.

        ``exclude_bias=True`` drops every tensor whose name contains ".bias".
        Run #8's displacement check showed the M2P bias terms ate the entire
        ES update budget (avg 25%, max 62% of their RMS) but had negligible
        effect on the answer because biases on activations of magnitude ~1
        only contribute ~1% perturbation. Excluding them frees the budget
        for the weight matrices that actually carry the model's behaviour.
        """
        iter_learnable_tensors = _import_shine_utils_module(
            "myloradict"
        ).iter_learnable_tensors

        def _rms(t: torch.Tensor) -> float:
            return float(t.detach().float().pow(2).mean().sqrt().item()) if t.numel() else 0.0

        def _is_bias(name: str) -> bool:
            # ".bias" catches Linear/LayerNorm `weight`/`bias` pairs
            # ("...linear1.bias"). PyTorch's MultiheadAttention uses a
            # *single-token* `in_proj_bias`, which the dotted check misses
            # — caught explicitly here.
            return name.endswith(".bias") or name.endswith("in_proj_bias")

        def _keep(name: str, t: torch.Tensor) -> bool:
            if exclude_bias and _is_bias(name):
                return False
            return min_rms <= 0.0 or _rms(t) >= min_rms

        out: list[tuple[str, torch.Tensor]] = []
        for name, p in self.metanetwork.metanetwork.named_parameters():
            full = f"m2p.{name}"
            if _keep(full, p):
                out.append((full, p))
        if include_mem_tokens:
            mem = self.metanetwork.metamodel.model.mem_tokens
            if _keep("mem_tokens", mem):
                out.append(("mem_tokens", mem))
        if include_metalora:
            # metalora leaves are A/B/C of LoRA — no bias-style asymmetry,
            # exclude_bias has no effect on them. Keep them under the RMS
            # filter only.
            for i, t in enumerate(iter_learnable_tensors(self.metalora)):
                name = f"metalora.leaf_{i}"
                if _keep(name, t):
                    out.append((name, t))
        return out

    def assert_only_hypernet_trainable(self) -> None:
        """Sanity check that SHINE's freeze() did its job.

        After construction, the only trainable tensors under metamodel must be
        ``mem_tokens``. Anything else means a backbone leak.
        """
        leaked: list[str] = []
        for name, p in self.metanetwork.metamodel.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("mem_tokens") or name == "model.mem_tokens":
                continue
            leaked.append(name)
        if leaked:
            raise AssertionError(
                f"Backbone leak: {len(leaked)} trainable params under metamodel "
                f"besides mem_tokens. First few: {leaked[:5]}"
            )

    # -- forward paths ---------------------------------------------------------

    def generate_lora(
        self,
        evidence_ids: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> dict:
        """Deterministic context -> LoRA dict. Runs under ``no_grad``."""
        with torch.no_grad():
            return self.metanetwork.generate_lora_dict(
                evidence_ids, evidence_mask, self.metalora
            )

    def _generate_lora_grad_chunk(
        self,
        evidence_ids: torch.Tensor,
        evidence_mask: torch.Tensor,
        use_gradient_checkpoint: bool,
    ) -> dict:
        """Single hypernet forward for a chunk of contexts; see ``generate_lora_grad``."""
        embed_rows = self.metanetwork.metamodel.get_input_embeddings().num_embeddings
        if evidence_ids.numel() > 0:
            mn = int(evidence_ids.min().item())
            mx = int(evidence_ids.max().item())
            if mn < 0 or mx >= embed_rows:
                raise ValueError(
                    f"generate_lora_grad: evidence_ids out of vocab range "
                    f"[0, {embed_rows}); got min={mn}, max={mx}, "
                    f"shape={tuple(evidence_ids.shape)}"
                )
        outputs = self.metanetwork.metamodel(
            input_ids=evidence_ids,
            attention_mask=evidence_mask,
            loradict=self.metalora,
            use_gradient_checkpoint=use_gradient_checkpoint,
            use_cache=False,
        )
        memory_states = outputs.memory_states
        if use_gradient_checkpoint:
            from torch.utils.checkpoint import checkpoint
            plain_output = checkpoint(
                self.metanetwork.metanetwork,
                memory_states,
                use_reentrant=False,
            )
        else:
            plain_output = self.metanetwork.metanetwork(memory_states)
        return self.metanetwork.metamodel.generate_lora_dict(
            self.lora_r,
            scale=self.metanetwork.scale,
            plain_tensor=plain_output,
        )

    def generate_lora_grad(
        self,
        evidence_ids: torch.Tensor,
        evidence_mask: torch.Tensor,
        use_gradient_checkpoint: bool = True,
        microbatch_size: int = 0,
    ) -> dict:
        """Grad-enabled context -> LoRA dict (RL path), with optional microbatching.

        Grad-enabled equivalent of :meth:`generate_lora`. Replicates SHINE's
        ``Metanetwork.generate_lora_dict`` inline rather than calling it,
        because we need three flags SHINE's wrapper doesn't forward:

        * ``use_gradient_checkpoint=True`` on the evidence forward. The
          ~1200-token (evidence + 148 mem_tokens) pass through 36 Qwen3
          layers under autograd otherwise stores >10 GB of activations, which
          blows the budget when colocated with a vLLM worker.
        * ``use_cache=False``. With grad checkpointing, the default
          ``DynamicCache`` mutates during the forward pass and the recompute
          then sees a doubled ``k_len`` — same shape mismatch that bit
          :meth:`score_answer_logprobs`.
        * Gradient checkpointing on the **M2P transformer** itself when
          ``use_gradient_checkpoint=True``. With ``B=8`` contexts, the M2P
          FFN intermediate is ``[B*num_layers, num_mem_token,
          dim_feedforward] = [288, 148, 8192]`` ≈ 1.4 GB / layer × 4 layers
          plus autograd. Recomputing M2P during backward saves >5 GB.

        ``microbatch_size``: 0 (default) processes all B contexts in one
        forward. Otherwise, splits B into chunks of ``microbatch_size``,
        runs the hypernet per chunk, and concatenates the resulting
        loradicts along the ``Lb`` dim. Trades wall-clock for peak working
        memory; backward graphs of all chunks are kept until ``backward()``.
        Useful when M2P FFN intermediate at full B exceeds HBM headroom.
        """
        B = evidence_ids.shape[0]
        if microbatch_size <= 0 or microbatch_size >= B:
            return self._generate_lora_grad_chunk(
                evidence_ids, evidence_mask, use_gradient_checkpoint,
            )
        chunks: list[dict] = []
        for s in range(0, B, microbatch_size):
            e = min(s + microbatch_size, B)
            chunks.append(self._generate_lora_grad_chunk(
                evidence_ids[s:e], evidence_mask[s:e], use_gradient_checkpoint,
            ))
        return _concat_loradicts_lb(chunks)

    def _score_answer_logprobs_chunk(
        self,
        loradict: dict,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_mask: torch.Tensor,
        use_gradient_checkpoint: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single rescore forward; see ``score_answer_logprobs``."""
        # Defensive bounds check: token-id OOB on the embedding lookup is the
        # most common deferred-CUDA assertion in this path. Sync here so the
        # error message is local instead of surfacing at the next .all().
        embed_rows = self.metanetwork.metamodel.get_input_embeddings().num_embeddings
        if input_ids.numel() > 0:
            mn = int(input_ids.min().item())
            mx = int(input_ids.max().item())
            if mn < 0 or mx >= embed_rows:
                raise ValueError(
                    f"score_answer_logprobs: input_ids out of vocab range "
                    f"[0, {embed_rows}); got min={mn}, max={mx}, "
                    f"shape={tuple(input_ids.shape)}"
                )
        out = self.metanetwork.metamodel(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loradict=loradict,
            ignore_mem_token=True,
            use_gradient_checkpoint=use_gradient_checkpoint,
            # Disable KV cache: with gradient checkpointing, the cache is
            # mutated during forward and contaminates recomputation, causing
            # a shape mismatch (k_len doubles) in the backward pass.
            use_cache=False,
        )
        logits = out.logits  # [N, T, V]
        # Shift: logits at t predict token at t+1.
        shift_logits = logits[:, :-1, :]  # [N, T-1, V]
        shift_labels = input_ids[:, 1:]    # [N, T-1]
        shift_mask = answer_mask[:, 1:]    # [N, T-1] — target positions

        # Memory-efficient log-prob via fused cross_entropy. The naive
        # ``logsumexp + gather`` allocates a full fp32 [N, T-1, V] buffer
        # for the log-softmax (~7 GB at 128 seqs × 100 tokens × 151,672
        # vocab), which OOMs the training GPU. ``F.cross_entropy`` with
        # ``reduction='none'`` uses the fused log_softmax+nll kernel that
        # never materializes the full log-softmax tensor; only the per-row
        # output [N*(T-1)] is allocated.
        N, Tm1, V = shift_logits.shape
        neg_logp = torch.nn.functional.cross_entropy(
            shift_logits.reshape(N * Tm1, V),
            shift_labels.reshape(N * Tm1),
            reduction="none",
        )
        logprobs = -neg_logp.reshape(N, Tm1)
        return logprobs, shift_mask

    def score_answer_logprobs(
        self,
        loradict: dict,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_mask: torch.Tensor,
        use_gradient_checkpoint: bool = True,
        microbatch_contexts: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Grad-enabled per-token log π(y_t | y_<t, c, LoRA_c) on the answer span.

        ``input_ids`` is ``[N, T]`` concatenated ``[prompt || y]``, **ordered
        context-major**: rows ``[b*num_beams : (b+1)*num_beams]`` use LoRA
        slot ``b`` (SHINE's ``LoraLinear.num_beams = N // Lb`` indexing).
        ``answer_mask`` is ``[N, T]`` with 1 on target positions.

        Returns ``(logprobs[N, T-1], mask[N, T-1])``. The trailing length is
        ``T-1`` because of the causal shift; apply ``mask`` when reducing.

        ``microbatch_contexts``: 0 (default) runs one forward over all ``N``
        rows. Otherwise splits the work into chunks of ``microbatch_contexts``
        contexts (each chunk has ``microbatch_contexts * num_beams`` rows
        and a corresponding ``Lb=microbatch_contexts`` slice of the loradict).
        Useful when ``B*Q*K`` re-score samples don't fit in HBM at once.
        Backward graphs of all chunks are retained until ``backward()``;
        peak savings come from working-memory in the per-layer forward, not
        from stored activations (which are already small under grad ckpt).
        """
        # Discover Lb from any leaf of loradict so we can split correctly.
        if not loradict:
            raise ValueError("Empty loradict.")
        first_layer = next(iter(loradict.values()))
        first_proj = next(iter(first_layer["attention"].values()))
        Lb = int(first_proj["A"].shape[0])
        N = input_ids.shape[0]
        if N % Lb != 0:
            raise ValueError(
                f"score_answer_logprobs: N={N} must be a multiple of Lb={Lb} "
                f"(SHINE's LoraLinear.num_beams = N // Lb)."
            )
        num_beams = N // Lb

        if microbatch_contexts <= 0 or microbatch_contexts >= Lb:
            return self._score_answer_logprobs_chunk(
                loradict, input_ids, attention_mask, answer_mask,
                use_gradient_checkpoint,
            )

        logprob_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        for s in range(0, Lb, microbatch_contexts):
            e = min(s + microbatch_contexts, Lb)
            rs, re = s * num_beams, e * num_beams
            sub_lora = _slice_loradict_lb(loradict, s, e)
            lp, m = self._score_answer_logprobs_chunk(
                sub_lora,
                input_ids[rs:re],
                attention_mask[rs:re],
                answer_mask[rs:re],
                use_gradient_checkpoint,
            )
            logprob_chunks.append(lp)
            mask_chunks.append(m)
        return torch.cat(logprob_chunks, dim=0), torch.cat(mask_chunks, dim=0)

    def trainable_hypernet_params(self) -> list[tuple[str, torch.Tensor]]:
        """RL optimizer params: M2P + metalora, no mem_tokens, no bias filter.

        Share selection with ``all_perturbable_params`` but set the flags
        appropriate for gradient-based training:

        * ``include_metalora=True``  — we want grads through the LoRA leaves.
        * ``include_mem_tokens=False`` — the ``ift_mqa_1qa`` ckpt ships with
          mem_tokens=0 and the M2P reads from attention-on-zero; touching them
          under ES collapses F1. The same singularity applies here.
        * ``exclude_bias=False`` — biases are fine under Adam (budget-eating
          concern was ES-specific).
        """
        return self.all_perturbable_params(
            include_mem_tokens=False,
            include_metalora=True,
            min_rms=0.0,
            exclude_bias=False,
        )

    @torch.no_grad()
    def answer(
        self,
        loradict: dict,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Run the LoRA-conditioned base model under greedy decoding.

        Uses ``torch.no_grad`` rather than ``torch.inference_mode`` because the
        ``@torch.compile`` on ``Metanetwork.forward`` can conflict with
        inference mode. ``metamodel.generate`` itself is not compiled, so this
        is the safe path.
        """
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        return self.metanetwork.metamodel.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            ignore_mem_token=True,
            loradict=loradict,
            **gen_kwargs,
        )

    # -- persistence -----------------------------------------------------------

    def save(self, out_dir: str) -> None:
        """Delegates to SHINE's save_checkpoint."""
        _ensure_shine_on_path()
        save_checkpoint = _import_shine_utils_module(
            "mysaveload"
        ).save_checkpoint

        os.makedirs(out_dir, exist_ok=True)
        save_checkpoint(self.metanetwork, out_dir, self.metalora)
