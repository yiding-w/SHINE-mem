"""Attention backend helpers for doc-to-lora MAB eval."""

from __future__ import annotations

import os
from typing import Any

_D2L_ATTN_PATCH_APPLIED = False
_ORIG_CHECK_FLASH = None
_ORIG_AUTOSET = None


def flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_d2l_attn(agent_config: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Return (use_flash_attn, base_attn_implementation).

    doc-to-lora official path: flash_attention_2 when flash_attn is installed.
    Without flash: Qwen base uses sdpa; Idefics2Perceiver stays eager (no sdpa support).
    """
    agent_config = agent_config or {}

    if os.environ.get("D2L_FORCE_SDPA", "").strip() in ("1", "true", "yes"):
        return False, "sdpa"

    if os.environ.get("D2L_FORCE_EAGER", "").strip() in ("1", "true", "yes"):
        return False, "eager"

    use_flash = agent_config.get("use_flash_attn")
    if use_flash is None:
        env_flash = os.environ.get("D2L_USE_FLASH_ATTN", "").strip().lower()
        if env_flash in ("0", "false", "no"):
            use_flash = False
        elif env_flash in ("1", "true", "yes"):
            use_flash = True
        else:
            use_flash = flash_attn_available()

    attn_impl = (
        agent_config.get("attn_implementation")
        or os.environ.get("D2L_ATTN_IMPLEMENTATION")
        or os.environ.get("ATTN_IMPLEMENTATION")
    )

    if use_flash and flash_attn_available():
        return True, "flash_attention_2"

    if attn_impl in ("flash_attention_2", "flash_attn"):
        if flash_attn_available():
            return True, "flash_attention_2"
        print("[d2l_attn] flash_attention_2 requested but flash_attn missing; fallback eager", flush=True)
        return False, "eager"

    if attn_impl in ("sdpa", "eager"):
        return False, str(attn_impl)

    # Default without flash: sdpa for Qwen base (Idefics2 flash requests handled by patch -> eager).
    return False, "sdpa"


def apply_d2l_attn_patch(fallback_impl: str = "eager") -> None:
    """When flash_attn is missing, replace flash_attention_2 with fallback (eager/sdpa).

    No-op when using real flash_attention_2. Does NOT globally force sdpa on all submodules
    (Idefics2Perceiver does not support sdpa).
    """
    global _D2L_ATTN_PATCH_APPLIED, _ORIG_CHECK_FLASH, _ORIG_AUTOSET
    if fallback_impl in ("flash_attention_2", "flash_attn"):
        return
    if _D2L_ATTN_PATCH_APPLIED:
        return
    _D2L_ATTN_PATCH_APPLIED = True

    import transformers.modeling_utils as modeling_utils

    _ORIG_CHECK_FLASH = modeling_utils.PreTrainedModel._check_and_enable_flash_attn_2
    _ORIG_AUTOSET = modeling_utils.PreTrainedModel._autoset_attn_implementation

    def _set_flash_to_fallback(config, impl: str) -> None:
        for attr in ("_attn_implementation", "attn_implementation"):
            if hasattr(config, attr) and getattr(config, attr) in (None, "flash_attention_2"):
                setattr(config, attr, impl)

    @classmethod
    def _check_and_enable_flash_attn_2(cls, config, *args, **kwargs):
        try:
            return _ORIG_CHECK_FLASH(config, *args, **kwargs)
        except ImportError:
            _set_flash_to_fallback(config, fallback_impl)
            return config

    @classmethod
    def _autoset_attn_implementation(cls, config, *args, **kwargs):
        try:
            return _ORIG_AUTOSET(config, *args, **kwargs)
        except (ImportError, ValueError) as exc:
            msg = str(exc).lower()
            if "flash" in msg or "scaled_dot_product_attention" in msg or "sdpa" in msg:
                _set_flash_to_fallback(config, fallback_impl)
                return config
            raise

    modeling_utils.PreTrainedModel._check_and_enable_flash_attn_2 = _check_and_enable_flash_attn_2
    modeling_utils.PreTrainedModel._autoset_attn_implementation = _autoset_attn_implementation


def patch_d2l_state_dict_attn(state_dict: dict, attn_impl: str = "eager") -> None:
    if attn_impl in ("flash_attention_2", "flash_attn"):
        return
    hypernet_config = state_dict.get("hypernet_config")
    if hypernet_config is None:
        return
    agg_config = getattr(hypernet_config, "aggregator_config", None)
    if agg_config is not None:
        for attr in ("_attn_implementation", "attn_implementation"):
            if hasattr(agg_config, attr):
                setattr(agg_config, attr, attn_impl)
