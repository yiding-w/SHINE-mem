"""
Unified tokenizer factory for the SHINE_V2 project.

All code that needs a tokenizer MUST use the functions in this module
instead of calling AutoTokenizer.from_pretrained() directly.

This ensures:
  1. Extra special tokens are added consistently across all processes.
  2. New tokens use reserved embedding slots (no resize needed).
  3. Insufficient reserved slots are detected early with a clear error.
"""

import os
import logging
from typing import List, Optional, Dict, Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_tokenizer_cache: Dict[str, PreTrainedTokenizerBase] = {}

# Project root (two levels up from utils/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_tokenizer_config(
    config_path_or_cfg: Union[str, DictConfig],
) -> DictConfig:
    """Load the tokenizer YAML configuration.

    Args:
        config_path_or_cfg: One of:
            - str: path to a YAML file to load.
            - DictConfig: an already-loaded Hydra config object (returned as-is).

    Raises:
        TypeError: If config_path_or_cfg is None (must be explicitly provided).
    """
    if config_path_or_cfg is None:
        raise TypeError(
            "tokenizer_cfg must be explicitly provided. "
            "Pass either a DictConfig (from Hydra cfg.tokenizer) or a path to a YAML file."
        )
    if isinstance(config_path_or_cfg, DictConfig):
        return config_path_or_cfg
    path = config_path_or_cfg
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tokenizer config not found at '{path}'."
        )
    return OmegaConf.load(path)


def _validate_reserved_slots(
    tokenizer: PreTrainedTokenizerBase,
    extra_tokens: List[str],
    cfg: DictConfig,
) -> None:
    """
    Validate that there are enough reserved embedding slots for the extra tokens.

    Raises RuntimeError if the reserved slots are insufficient.
    """
    model_vocab_size = cfg.reserved_slots.model_vocab_size
    tokenizer_size_after = len(tokenizer)  # after add_tokens

    if tokenizer_size_after > model_vocab_size:
        num_extra = len(extra_tokens)
        total_reserved = cfg.reserved_slots.total_reserved
        raise RuntimeError(
            f"Insufficient reserved embedding slots!\n"
            f"  Model vocab_size (embedding rows): {model_vocab_size}\n"
            f"  Tokenizer size after adding {num_extra} extra tokens: {tokenizer_size_after}\n"
            f"  Total reserved slots available: {total_reserved}\n"
            f"  Overflow by: {tokenizer_size_after - model_vocab_size} tokens\n"
            f"\n"
            f"  You need to either:\n"
            f"    1. Reduce the number of extra_special_tokens in configs/tokenizer/origin.yaml\n"
            f"    2. Expand the model's vocab_size (requires model.resize_token_embeddings())\n"
        )


def create_tokenizer(
    model_path: str,
    *,
    tokenizer_cfg: Union[str, DictConfig],
    chat_template: Optional[str] = None,
) -> PreTrainedTokenizerBase:
    """
    Create a tokenizer with unified configuration.

    This is the PRIMARY factory function. All code that needs a tokenizer
    should call this function.

    Extra special tokens defined in the tokenizer config are ALWAYS added
    to ensure consistency across all processes.

    Args:
        model_path: Path to the pretrained model / tokenizer directory.
        tokenizer_cfg: Required. Either a DictConfig object (from Hydra
            cfg.tokenizer) or a path string to a tokenizer YAML config file.
        chat_template: Optional chat template string to override the
            tokenizer's default. If None, the tokenizer's original
            chat_template is preserved.

    Returns:
        A configured PreTrainedTokenizerBase instance.

    Raises:
        RuntimeError: If reserved embedding slots are insufficient.
        FileNotFoundError: If model_path or config_path doesn't exist.
        TypeError: If tokenizer_cfg is not provided.
    """
    # Load tokenizer config
    cfg = _load_tokenizer_config(tokenizer_cfg)

    # Create base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Always add extra special tokens to ensure consistency
    extra_tokens: List[str] = list(cfg.extra_special_tokens)
    if extra_tokens:
        num_added = tokenizer.add_tokens(extra_tokens)
        if num_added > 0:
            logger.debug(
                f"Added {num_added} extra tokens to tokenizer: {extra_tokens}"
            )

        # Validate reserved slots
        _validate_reserved_slots(tokenizer, extra_tokens, cfg)

    # Override chat template if provided
    if chat_template is not None:
        tokenizer.chat_template = chat_template

    return tokenizer


def get_cached_tokenizer(
    model_path: str,
    *,
    tokenizer_cfg: Union[str, DictConfig],
    chat_template: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> PreTrainedTokenizerBase:
    """
    Get or create a cached tokenizer instance.

    Useful for avoiding repeated tokenizer loading in the same process
    (e.g., in the main training loop). The cache is keyed by
    (model_path, cache_key).

    Extra special tokens are ALWAYS added to ensure consistency.

    Args:
        model_path: Path to the pretrained model / tokenizer directory.
        tokenizer_cfg: Required. Either a DictConfig object (from Hydra
            cfg.tokenizer) or a path string to a tokenizer YAML config file.
        chat_template: Optional chat template override.
        cache_key: Optional extra key for cache differentiation.
            Use this when the same model_path needs different configurations
            (e.g., different chat templates).

    Returns:
        A configured PreTrainedTokenizerBase instance (possibly cached).
    """
    abs_path = os.path.abspath(model_path)
    key = f"{abs_path}::{cache_key or 'default'}"

    if key not in _tokenizer_cache:
        _tokenizer_cache[key] = create_tokenizer(
            model_path,
            tokenizer_cfg=tokenizer_cfg,
            chat_template=chat_template,
        )

    return _tokenizer_cache[key]


def get_extra_token_ids(
    *,
    tokenizer_cfg: Union[str, DictConfig],
) -> Dict[str, int]:
    """
    Get a mapping of extra special token strings to their assigned IDs.

    This is useful for code that needs to know the IDs of custom tokens
    (e.g., for building label masks or special position markers).

    Args:
        tokenizer_cfg: Required. Either a DictConfig object (from Hydra
            cfg.tokenizer) or a path string to a tokenizer YAML config file.

    Returns:
        Dict mapping token string to token ID.
        Example: {"<RECON>": 248077, "<COMP>": 248078, "<NOTHING>": 248079}
    """
    cfg = _load_tokenizer_config(tokenizer_cfg)
    extra_tokens: List[str] = list(cfg.extra_special_tokens)

    # The extra tokens are assigned IDs starting from first_reserved_id
    first_id = cfg.reserved_slots.first_reserved_id
    return {token: first_id + i for i, token in enumerate(extra_tokens)}


def get_reserved_slots_info(
    *,
    tokenizer_cfg: Union[str, DictConfig],
) -> Dict[str, int]:
    """
    Get reserved slot metadata from the tokenizer config.

    Args:
        tokenizer_cfg: Required. Either a DictConfig object (from Hydra
            cfg.tokenizer) or a path string to a tokenizer YAML config file.

    Returns:
        Dict with keys: tokenizer_original_size, model_vocab_size,
        first_reserved_id, total_reserved.
    """
    cfg = _load_tokenizer_config(tokenizer_cfg)
    return {
        "tokenizer_original_size": cfg.reserved_slots.tokenizer_original_size,
        "model_vocab_size": cfg.reserved_slots.model_vocab_size,
        "first_reserved_id": cfg.reserved_slots.first_reserved_id,
        "total_reserved": cfg.reserved_slots.total_reserved,
    }


# ---------------------------------------------------------------------------
# Chat template presets
# ---------------------------------------------------------------------------

# No-thinking chat template (Qwen-style, with thinking disabled).
# Used by: pretrain_annealing (msmarco_mqa_conv), SFT (msmarco_mqa)
# The default mode is "preserve_thinking" — i.e., the tokenizer's original
# chat_template (which supports thinking) is kept unchanged. Only use
# NOTHINKING_CHAT_TEMPLATE when you explicitly need to disable thinking.
NOTHINKING_CHAT_TEMPLATE = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if (loop.last or (not loop.last and reasoning_content)) and (enable_thinking is not defined or enable_thinking != false) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is not defined or enable_thinking != false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
