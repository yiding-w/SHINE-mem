"""
Tensor-parallel training step.

The PP path in ``meta_train.py`` glues together:

  * pipeline_forward_train_multi_mb       (Step 1 + Step 4 over micro-batches)
  * mem_gather                            (cross-stage hidden-state collection)
  * Hypernetwork forward (forward_with_grad)
  * lora_scatter                          (cross-stage LoRA-dict broadcast)
  * pipeline_backward_multi_mb            (reverse-order backward including
                                          step-4 LLM, lora_scatter, hypernetwork,
                                          step-1 LLM, mem_gather)

Under TP all of that scaffolding collapses to a straightforward
forward → loss → backward → step. Both the trainable hypernetwork and
the frozen LLM are replicated across DP and TP-sharded internally; the
o_proj / down_proj all-reduces handle the cross-rank summation inline.

This module provides ``tp_train_step`` that runs one such step. Use it
as the inner per-batch call in the TP training loop. The caller is
responsible for:

  * computing ``memory_states`` (the Step-1 pass output) — typically by
    a no-grad forward of the frozen LLM with ``use_mem_token=True``,
  * supplying a hypernetwork module that maps memory_states to a flat
    ``plain_tensor`` of shape ``[Lb, model.lora_params_numel(lora_ranks)]``,
  * supplying ``input_ids`` / ``labels`` for the Step-4 forward,
  * DP gradient synchronisation between this step and ``optimizer.step()``
    (use ``utils.parallel.sync_gradients_across_dp``).
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


__all__ = ["tp_train_step", "tp_forward_loss"]


def tp_forward_loss(
    text_model: nn.Module,
    hypernetwork: nn.Module,
    memory_states: torch.Tensor,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    lora_ranks: dict,
    scale: float,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    attention_mask: Optional[torch.Tensor] = None,
    context_lengths: Optional[torch.LongTensor] = None,
    use_mem_token: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the trainable forward path: hypernetwork → loradict → text_model
    → loss. Returns ``(loss, last_hidden_state)``.

    Tensors flow:
      ``memory_states`` → ``hypernetwork`` → ``plain_tensor`` →
      ``text_model.generate_lora_dict`` → ``loradict`` → ``text_model(...)`` →
      ``last_hidden_state`` → ``loss_fn(hs, labels)``.

    The TP collectives (o_proj / down_proj all-reduce) happen inside
    text_model's forward; no explicit communication is needed here.
    """
    plain_tensor = hypernetwork(memory_states)
    loradict = text_model.generate_lora_dict(lora_ranks, scale, plain_tensor)

    out = text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loradict=loradict,
        use_cache=False,
        use_mem_token=use_mem_token,
        context_lengths=context_lengths,
    )

    last_hs = out.last_hidden_state
    loss = loss_fn(last_hs, labels)
    return loss, last_hs


def tp_train_step(
    text_model: nn.Module,
    hypernetwork: nn.Module,
    memory_states: torch.Tensor,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    optimizer: torch.optim.Optimizer,
    lora_ranks: dict,
    scale: float,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    attention_mask: Optional[torch.Tensor] = None,
    context_lengths: Optional[torch.LongTensor] = None,
    use_mem_token: bool = False,
    dp_sync_fn: Optional[Callable[[nn.Module], None]] = None,
    grad_clip_norm: Optional[float] = None,
) -> float:
    """One full forward → loss → backward → step.

    Returns the scalar loss value. Caller decides when to zero grads
    (we zero them at the end of the step, mirroring most LoRA training
    loops where each step is independent). ``dp_sync_fn`` is called
    after backward and before optimizer.step() to all-reduce grads
    across DP replicas — pass ``functools.partial(sync_gradients_across_dp,
    device=device)`` from ``utils.parallel``.
    """
    loss, _ = tp_forward_loss(
        text_model=text_model,
        hypernetwork=hypernetwork,
        memory_states=memory_states,
        input_ids=input_ids,
        labels=labels,
        lora_ranks=lora_ranks,
        scale=scale,
        loss_fn=loss_fn,
        attention_mask=attention_mask,
        context_lengths=context_lengths,
        use_mem_token=use_mem_token,
    )

    loss.backward()

    if dp_sync_fn is not None:
        dp_sync_fn(hypernetwork)
        dp_sync_fn(text_model)

    if grad_clip_norm is not None:
        trainable = [p for p in list(hypernetwork.parameters()) + list(text_model.parameters())
                     if p.requires_grad and p.grad is not None]
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return loss.detach().float().item()
