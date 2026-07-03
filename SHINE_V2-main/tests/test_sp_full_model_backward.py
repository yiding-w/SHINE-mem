"""
Debug test: full 64-layer LLM forward + backward with TP=4, SP=2.

This test loads the full LLM (all 64 layers), wraps each layer with
activation checkpointing + torch.compile (exactly like the real training),
then runs a forward + backward pass to see if it deadlocks.

If this test passes, the problem is in the hypernetwork/memory_states/loss
computation, not in the LLM layers themselves.

Run with:
  torchrun --nproc_per_node=8 tests/test_sp_full_model_backward.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def log(rank, msg):
    if rank == 0:
        print(f"[rank {rank}] {msg}", flush=True)


def log_all(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def create_tp_sp_groups(world_size=8, tp_size=4, sp_size=2):
    """Create TP and SP process groups.
    Layout: 8 GPUs = TP_size * SP_size = 4 * 2
      - TP group for SP_rank=0: [0,1,2,3]
      - TP group for SP_rank=1: [4,5,6,7]
      - SP group for TP_rank=0: [0,4]
      - SP group for TP_rank=1: [1,5]
      - SP group for TP_rank=2: [2,6]
      - SP group for TP_rank=3: [3,7]
    """
    tp_groups = []
    for sp_rank in range(sp_size):
        ranks = list(range(sp_rank * tp_size, (sp_rank + 1) * tp_size))
        tp_groups.append(ranks)

    sp_groups = []
    for tp_rank in range(tp_size):
        ranks = [sp_rank * tp_size + tp_rank for sp_rank in range(sp_size)]
        sp_groups.append(ranks)

    my_rank = dist.get_rank()
    my_tp_group = None
    my_sp_group = None
    my_tp_rank = my_rank % tp_size
    my_sp_rank = my_rank // tp_size

    for ranks in tp_groups:
        pg = dist.new_group(ranks)
        if my_rank in ranks:
            my_tp_group = pg

    for ranks in sp_groups:
        pg = dist.new_group(ranks)
        if my_rank in ranks:
            my_sp_group = pg

    return my_tp_group, my_sp_group, my_tp_rank, my_sp_rank, tp_size, sp_size


def main():
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    assert world_size == 8, f"This test requires exactly 8 GPUs, got {world_size}"

    log(rank, f"=== Full Model SP Backward Debug Test ===")
    log(rank, f"World size: {world_size}, Local rank: {local_rank}")

    # Create process groups
    tp_group, sp_group, tp_rank, sp_rank, tp_size, sp_size = create_tp_sp_groups()
    log_all(rank, f"TP rank={tp_rank}, SP rank={sp_rank}")

    # Load model config
    from transformers import AutoConfig
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Qwen3.6-27B")
    base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = base_config.text_config if hasattr(base_config, "text_config") else base_config
    text_config._attn_implementation = "sdpa"
    text_config._attn_implementation_internal = "sdpa"

    # Apply Liger patch
    from utils.liger_patch import apply_liger_rmsnorm_patch
    apply_liger_rmsnorm_patch()

    # Dynamo config
    import torch._dynamo as _dynamo
    _dynamo.config.cache_size_limit = 64
    if hasattr(_dynamo.config, "recompile_limit"):
        _dynamo.config.recompile_limit = 64

    # Load the full TP model
    log(rank, "Loading TP model...")
    from utils.mytp.tp_load_model import load_pretrained_llm_for_tp
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.create({
        "path": model_path,
        "lora_class": "src_transformers_lora.LoraQwen3_5.LoraQwen3_5ForCausalLM",
    })

    llm = load_pretrained_llm_for_tp(
        model_cfg,
        tp_rank=tp_rank,
        tp_world=tp_size,
        tp_process_group=tp_group,
        dtype=torch.bfloat16,
        freeze=True,
        num_mem_token=444,
        sp_group=sp_group,
        sp_world=sp_size,
    )
    log(rank, "Model loaded.")

    # Wrap layers with activation checkpointing + torch.compile (like real training)
    log(rank, "Wrapping layers with checkpoint + compile...")
    text_model = llm.model
    for idx, layer in enumerate(text_model.layers):
        # torch.compile each layer
        layer.forward = torch.compile(layer.forward, dynamic=False)

    # Now wrap with checkpoint
    for idx, layer in enumerate(text_model.layers):
        orig_forward = layer.forward
        def make_ckpt(fn, layer_idx):
            def wrapped(*args, **kwargs):
                def fwd(*a):
                    return fn(*a, **kwargs)
                return checkpoint(fwd, *args, use_reentrant=False)
            return wrapped
        layer.forward = make_ckpt(orig_forward, idx)

    log(rank, "All 64 layers wrapped with checkpoint + compile.")

    # Create a dummy trainable parameter (simulates hypernetwork grad flow)
    trainable_proj = nn.Linear(text_config.hidden_size, text_config.hidden_size, bias=False).to(device=device, dtype=torch.bfloat16)
    trainable_proj.weight.requires_grad_(True)

    # Prepare input
    seq_len_local = 512  # Each SP rank gets 512 tokens
    input_ids = torch.randint(0, text_config.vocab_size, (1, seq_len_local), device=device)

    log(rank, f"Running forward (seq_len_local={seq_len_local})...")
    dist.barrier()
    t0 = time.time()

    # Forward pass through the full model
    with torch.no_grad():
        # Get embeddings
        inputs_embeds = text_model.embed_tokens(input_ids)

    # Make it require grad through trainable_proj
    inputs_embeds = inputs_embeds.detach().requires_grad_(True)
    hidden_states = inputs_embeds + trainable_proj(inputs_embeds) * 0.01

    # Position embeddings
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    rotary = Qwen3_5TextRotaryEmbedding(config=text_config).to(device=device)
    position_ids = torch.arange(seq_len_local, device=device).unsqueeze(0)
    sp_rank_val = dist.get_rank(sp_group)
    position_ids_shifted = position_ids + sp_rank_val * seq_len_local
    position_embeddings = rotary(hidden_states, position_ids_shifted)

    # Run through all 64 layers
    for idx, layer in enumerate(text_model.layers):
        hidden_states = layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
            position_ids=position_ids_shifted,
        )
        if idx == 0 and rank == 0:
            log(rank, f"  Layer 0 done, shape={hidden_states.shape}")

    # Final norm + loss
    hidden_states = text_model.norm(hidden_states)
    loss = hidden_states.sum()

    dist.barrier()
    t1 = time.time()
    log_all(rank, f"Forward done in {t1 - t0:.1f}s, loss={loss.item():.4f}")

    # Backward
    log_all(rank, f"Running backward...")
    dist.barrier()
    t2 = time.time()

    loss.backward()

    dist.barrier()
    t3 = time.time()
    log_all(rank, f"Backward done in {t3 - t2:.1f}s")

    # Check gradient
    if trainable_proj.weight.grad is not None:
        grad_norm = trainable_proj.weight.grad.norm().item()
        log_all(rank, f"trainable_proj grad_norm={grad_norm:.6e}")
    else:
        log_all(rank, f"WARNING: trainable_proj.weight.grad is None!")

    log(rank, "")
    log(rank, "=" * 60)
    log(rank, "FULL MODEL TEST PASSED ✓✓✓")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
