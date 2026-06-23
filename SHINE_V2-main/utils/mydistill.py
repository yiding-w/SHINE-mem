#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Distillation Loss Module for SHINE_V2

Provides distillation loss functions that compare teacher and student outputs.
Supports two modes:
  - "logits": Compare teacher and student logits via KL divergence or similar.
  - "hidden_states": Compare teacher and student hidden states via MSE or cosine.

The factory function `create_distill_loss_fn` returns a callable that can be
directly invoked during training without any if/else branching.

Usage:
    distill_loss_fn = create_distill_loss_fn(cfg.training.distill)
    loss = distill_loss_fn(teacher_output, student_output, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Logits-mode loss functions
# ---------------------------------------------------------------------------

class KLDivLogitsLoss(nn.Module):
    """
    KL divergence loss between teacher and student logits.

    Computes: KL(softmax(teacher/T) || softmax(student/T)) * T^2

    The T^2 scaling ensures gradients have the same magnitude regardless of
    temperature, following Hinton et al. (2015).

    Supports ignore_index to mask padding positions.
    """

    def __init__(self, temperature: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            teacher_output: (B, S, V) teacher logits (detached, no grad).
            student_output: (B, S, V) student logits (has grad).
            labels: (B, S) token labels for masking. Positions with
                    ignore_index are excluded from loss.

        Returns:
            Scalar loss tensor.
        """
        # Shift to align with next-token prediction
        teacher_logits = teacher_output[:, :-1, :].contiguous()
        student_logits = student_output[:, :-1, :].contiguous()

        T = self.temperature

        # Compute soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)

        # KL divergence per token: sum over vocab dimension
        kl_per_token = F.kl_div(
            student_log_probs, teacher_probs, reduction='none'
        ).sum(dim=-1)  # (B, S-1)

        # Apply mask if labels provided
        if labels is not None:
            shift_labels = labels[:, 1:].contiguous()
            mask = (shift_labels != self.ignore_index).float()  # (B, S-1)
            kl_per_token = kl_per_token * mask
            num_valid = mask.sum().clamp(min=1.0)
            loss = kl_per_token.sum() / num_valid
        else:
            loss = kl_per_token.mean()

        # Scale by T^2 (Hinton et al.)
        loss = loss * (T * T)
        return loss


class ReverseKLLogitsLoss(nn.Module):
    """
    Reverse KL divergence: KL(student || teacher).

    This encourages mode-seeking behavior (student focuses on high-probability
    regions of teacher distribution).
    """

    def __init__(self, temperature: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        teacher_logits = teacher_output[:, :-1, :].contiguous()
        student_logits = student_output[:, :-1, :].contiguous()

        T = self.temperature

        teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
        student_probs = F.softmax(student_logits / T, dim=-1)

        # Reverse KL: KL(student || teacher) = sum(student * (log_student - log_teacher))
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        kl_per_token = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

        if labels is not None:
            shift_labels = labels[:, 1:].contiguous()
            mask = (shift_labels != self.ignore_index).float()
            kl_per_token = kl_per_token * mask
            num_valid = mask.sum().clamp(min=1.0)
            loss = kl_per_token.sum() / num_valid
        else:
            loss = kl_per_token.mean()

        loss = loss * (T * T)
        return loss


class JSDLogitsLoss(nn.Module):
    """
    Jensen-Shannon Divergence between teacher and student logits.

    JSD = 0.5 * KL(teacher || M) + 0.5 * KL(student || M)
    where M = 0.5 * (teacher + student)

    JSD is symmetric and bounded, providing more stable gradients.
    """

    def __init__(self, temperature: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        teacher_logits = teacher_output[:, :-1, :].contiguous()
        student_logits = student_output[:, :-1, :].contiguous()

        T = self.temperature

        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        student_probs = F.softmax(student_logits / T, dim=-1)

        # Mixture distribution
        M = 0.5 * (teacher_probs + student_probs)
        M_log = M.log()

        # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        kl_teacher_m = (teacher_probs * (teacher_probs.log() - M_log)).sum(dim=-1)
        kl_student_m = (student_probs * (student_probs.log() - M_log)).sum(dim=-1)
        jsd_per_token = 0.5 * (kl_teacher_m + kl_student_m)

        if labels is not None:
            shift_labels = labels[:, 1:].contiguous()
            mask = (shift_labels != self.ignore_index).float()
            jsd_per_token = jsd_per_token * mask
            num_valid = mask.sum().clamp(min=1.0)
            loss = jsd_per_token.sum() / num_valid
        else:
            loss = jsd_per_token.mean()

        loss = loss * (T * T)
        return loss


# ---------------------------------------------------------------------------
# Hidden-states-mode loss functions
# ---------------------------------------------------------------------------

class MSEHiddenLoss(nn.Module):
    """
    MSE loss between teacher and student hidden states.

    Optionally applies a linear projection to align dimensions if teacher
    and student have different hidden sizes (not typical in self-distillation).
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            teacher_output: (B, S, H) teacher hidden states (detached).
            student_output: (B, S, H) student hidden states (has grad).
            labels: (B, S) for masking padding positions.

        Returns:
            Scalar MSE loss.
        """
        # Shift to align with next-token prediction positions
        teacher_hidden = teacher_output[:, :-1, :].contiguous()
        student_hidden = student_output[:, :-1, :].contiguous()

        # Per-token MSE: mean over hidden dim
        mse_per_token = ((student_hidden - teacher_hidden) ** 2).mean(dim=-1)  # (B, S-1)

        if labels is not None:
            shift_labels = labels[:, 1:].contiguous()
            mask = (shift_labels != self.ignore_index).float()
            mse_per_token = mse_per_token * mask
            num_valid = mask.sum().clamp(min=1.0)
            loss = mse_per_token.sum() / num_valid
        else:
            loss = mse_per_token.mean()

        return loss


class CosineHiddenLoss(nn.Module):
    """
    Cosine similarity loss between teacher and student hidden states.

    Loss = 1 - cosine_similarity(teacher, student), averaged over valid tokens.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        teacher_hidden = teacher_output[:, :-1, :].contiguous()
        student_hidden = student_output[:, :-1, :].contiguous()

        # Cosine similarity per token
        cos_sim = F.cosine_similarity(student_hidden, teacher_hidden, dim=-1)  # (B, S-1)
        loss_per_token = 1.0 - cos_sim

        if labels is not None:
            shift_labels = labels[:, 1:].contiguous()
            mask = (shift_labels != self.ignore_index).float()
            loss_per_token = loss_per_token * mask
            num_valid = mask.sum().clamp(min=1.0)
            loss = loss_per_token.sum() / num_valid
        else:
            loss = loss_per_token.mean()

        return loss


class SmoothL1HiddenLoss(nn.Module):
    """
    Smooth L1 (Huber) loss between teacher and student hidden states.

    More robust to outliers than MSE.
    """

    def __init__(self, ignore_index: int = -100, beta: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta

    def forward(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        teacher_hidden = teacher_output[:, :-1, :].contiguous()
        student_hidden = student_output[:, :-1, :].contiguous()

        # Per-token smooth L1: mean over hidden dim
        sl1_per_token = F.smooth_l1_loss(
            student_hidden, teacher_hidden, reduction='none', beta=self.beta
        ).mean(dim=-1)  # (B, S-1)

        if labels is not None:
            shift_labels = labels[:, 1:].contiguous()
            mask = (shift_labels != self.ignore_index).float()
            sl1_per_token = sl1_per_token * mask
            num_valid = mask.sum().clamp(min=1.0)
            loss = sl1_per_token.sum() / num_valid
        else:
            loss = sl1_per_token.mean()

        return loss


# ---------------------------------------------------------------------------
# Wrapper that handles mode selection and coefficient
# ---------------------------------------------------------------------------

class DistillLossWrapper(nn.Module):
    """
    Wrapper that combines mode (logits/hidden_states) and loss function
    into a single callable. Also applies the distillation coefficient.

    At training time, simply call:
        distill_loss = wrapper(teacher_output, student_output, labels)

    No if/else branching needed — the mode and loss type are baked in
    at initialization time.
    """

    def __init__(self, loss_fn: nn.Module, coefficient: float, mode: str):
        super().__init__()
        self.loss_fn = loss_fn
        self.coefficient = coefficient
        self.mode = mode  # "logits" or "hidden_states"

    def forward(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute distillation loss with coefficient applied.

        Args:
            teacher_output: Teacher logits (B, S, V) or hidden states (B, S, H).
                            Must be detached (no grad).
            student_output: Student logits (B, S, V) or hidden states (B, S, H).
                            Has grad flowing through loradict.
            labels: (B, S) for masking padding positions.

        Returns:
            Scalar loss tensor (already multiplied by coefficient).
        """
        raw_loss = self.loss_fn(teacher_output, student_output, labels)
        return self.coefficient * raw_loss


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

# Registry of available loss functions per mode
_LOGITS_LOSSES = {
    "kl_div": KLDivLogitsLoss,
    "reverse_kl": ReverseKLLogitsLoss,
    "jsd": JSDLogitsLoss,
}

_HIDDEN_LOSSES = {
    "mse": MSEHiddenLoss,
    "cosine": CosineHiddenLoss,
    "smooth_l1": SmoothL1HiddenLoss,
}


def create_distill_loss_fn(distill_cfg) -> Optional[DistillLossWrapper]:
    """
    Factory function that creates a distillation loss callable from config.

    Args:
        distill_cfg: Config dict/DictConfig with keys:
            - enabled (bool): Whether distillation is enabled.
            - mode (str): "logits" or "hidden_states".
            - loss_type (str): Loss function name (e.g. "kl_div", "mse").
            - coefficient (float): Scaling factor for distill loss.
            - temperature (float, optional): Temperature for logits mode.
                                             Default 2.0.

    Returns:
        DistillLossWrapper instance, or None if distillation is disabled.
    """
    if distill_cfg is None:
        return None

    enabled = distill_cfg.get("enabled", False)
    if not enabled:
        return None

    mode = distill_cfg.get("mode", "logits")
    loss_type = distill_cfg.get("loss_type", "kl_div")
    coefficient = float(distill_cfg.get("coefficient", 1.0))
    temperature = float(distill_cfg.get("temperature", 1.0))
    ignore_index = int(distill_cfg.get("ignore_index", -100))

    if mode == "logits":
        if loss_type not in _LOGITS_LOSSES:
            raise ValueError(
                f"Unknown logits distill loss_type: '{loss_type}'. "
                f"Available: {list(_LOGITS_LOSSES.keys())}"
            )
        loss_cls = _LOGITS_LOSSES[loss_type]
        loss_fn = loss_cls(temperature=temperature, ignore_index=ignore_index)
    elif mode == "hidden_states":
        if loss_type not in _HIDDEN_LOSSES:
            raise ValueError(
                f"Unknown hidden_states distill loss_type: '{loss_type}'. "
                f"Available: {list(_HIDDEN_LOSSES.keys())}"
            )
        loss_cls = _HIDDEN_LOSSES[loss_type]
        if loss_type == "smooth_l1":
            beta = float(distill_cfg.get("smooth_l1_beta", 1.0))
            loss_fn = loss_cls(ignore_index=ignore_index, beta=beta)
        else:
            loss_fn = loss_cls(ignore_index=ignore_index)
    else:
        raise ValueError(
            f"Unknown distill mode: '{mode}'. Must be 'logits' or 'hidden_states'."
        )

    return DistillLossWrapper(loss_fn=loss_fn, coefficient=coefficient, mode=mode)
