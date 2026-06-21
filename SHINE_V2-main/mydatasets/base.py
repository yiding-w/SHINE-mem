#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Dataset and Collator for SHINE_V2

All dataset modules under ``datasets/`` should inherit from :class:`BaseDataset`
and all collators should inherit from :class:`BaseCollator`.  Both base classes
call :func:`utils.mydata.resolve_special_token_id` to obtain a dict of every
special token defined in the tokenizer's ``added_tokens_decoder`` and store it
as ``self.special_tokens``.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Abstract base dataset for SHINE_V2.

    Subclasses **must** call ``super().__init__(model_path, tokenizer_cfg=tokenizer_cfg)``
    so that ``self.special_tokens`` is populated.

    Attributes:
        special_tokens (dict): Mapping ``{token_string: token_id}`` for every
            entry in the tokenizer's ``added_tokens_decoder``.
    """

    def __init__(self, model_path: str, *, tokenizer_cfg=None):
        """
        Args:
            model_path: Path to the pretrained model / tokenizer directory.
            tokenizer_cfg: Optional. Either a DictConfig object (from Hydra
                cfg.tokenizer) or a path string to a tokenizer YAML config file.
                If None, uses cached result (resolve_pad_token_id must have been
                called first to populate the cache).
        """
        from utils.mydata import resolve_special_token_id
        self.special_tokens: Dict[str, int] = resolve_special_token_id(model_path, tokenizer_cfg=tokenizer_cfg)

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError


class BaseCollator:
    """
    Abstract base collator for SHINE_V2.

    Subclasses **must** call ``super().__init__(model_path, tokenizer_cfg=tokenizer_cfg)``
    so that ``self.special_tokens`` is populated.

    Attributes:
        special_tokens (dict): Mapping ``{token_string: token_id}`` for every
            entry in the tokenizer's ``added_tokens_decoder``.
    """

    def __init__(self, model_path: str, *, tokenizer_cfg=None):
        """
        Args:
            model_path: Path to the pretrained model / tokenizer directory.
            tokenizer_cfg: Optional. Either a DictConfig object (from Hydra
                cfg.tokenizer) or a path string to a tokenizer YAML config file.
                If None, uses cached result (resolve_pad_token_id must have been
                called first to populate the cache).
        """
        from utils.mydata import resolve_special_token_id
        self.special_tokens: Dict[str, int] = resolve_special_token_id(model_path, tokenizer_cfg=tokenizer_cfg)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
