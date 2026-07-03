"""
Debug test: isolate whether SP backward deadlock is in the network layers.

Tests one full_attention layer and one linear_attention layer with:
  - TP=4, SP=2 (8 GPUs total on one node)
  - activation checkpointing (use_reentrant=False)
  - torch.compile

Run with:
  torchrun --nproc_per_node=8 tests/test_sp_layer_backward.py
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def log(rank, msg):
    """Only print from rank 0."""
    if rank == 0:
        print(f"[rank {rank}] {msg}", flush=True)


def log_all(rank, msg):
    """Print from all ranks."""
    print(f"[rank {rank}] {msg}", flush=True)


def create_tp_sp_groups(world_size=8, tp_size=4, sp_size=2):
    """Create TP and SP process groups.
    
    With 8 GPUs, TP=4, SP=2:
      - TP groups: [0,1,2,3], [4,5,6,7]  (but we use all 8 as one TP*SP group)
      - Actually: TP=4 means 4 ranks share model shards
      - SP=2 means 2 TP groups process different sequence chunks
      
    Layout: 8 GPUs = TP_size * SP_size * DP_size = 4 * 2 * 1
    Ranks [0..7]:
      - TP group for SP_rank=0: [0,1,2,3]
      - TP group for SP_rank=1: [4,5,6,7]
      - SP group for TP_rank=0: [0,4]
      - SP group for TP_rank=1: [1,5]
      - SP group for TP_rank=2: [2,6]
      - SP group for TP_rank=3: [3,7]
    """
    # TP groups: ranks that share the same SP position
    tp_groups = []
    for sp_rank in range(sp_size):
        ranks = list(range(sp_rank * tp_size, (sp_rank + 1) * tp_size))
        tp_groups.append(ranks)

    # SP groups: ranks that share the same TP position
    sp_groups = []
    for tp_rank in range(tp_size):
        ranks = [sp_rank * tp_size + tp_rank for sp_rank in range(sp_size)]
        sp_groups.append(ranks)

    # Create process groups
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


def test_layer(layer_type, layer_idx, config, tp_rank, tp_world, tp_group, sp_group, sp_world, device, rank):
    """Test a single layer with forward + backward, checkpoint + compile."""
    from utils.mytp.tp_decoder_layer import TPLoraQwen3_5DecoderLayer, load_decoder_layer_weights_from_full
    from src_transformers_lora.LoraQwen3_5 import LoraQwen3_5DecoderLayer

    log(rank, f"--- Testing {layer_type} layer (idx={layer_idx}) ---")

    # 1. Create full layer on CPU, then TP layer on GPU
    log(rank, f"  Creating full layer...")
    full_layer = LoraQwen3_5DecoderLayer(config, layer_idx)
    full_layer = full_layer.to(dtype=torch.bfloat16, device="cpu")

    log(rank, f"  Creating TP layer...")
    tp_layer = TPLoraQwen3_5DecoderLayer(
        config, layer_idx,
        tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_group,
        sp_group=sp_group, sp_world=sp_world,
    ).to(device=device, dtype=torch.bfloat16)

    log(rank, f"  Loading weights...")
    load_decoder_layer_weights_from_full(tp_layer, full_layer)
    del full_layer

    # Enable SP on full_attention layers
    if layer_type == "full_attention":
        tp_layer.self_attn.enable_sp(sp_group, sp_world, sp_mode="alltoall_zigzag")

    # Freeze all params (like in real training)
    for p in tp_layer.parameters():
        p.requires_grad_(False)

    # 2. Create a dummy trainable parameter (simulates LoRA/hypernetwork grad flow)
    # This ensures backward actually runs through the layer
    trainable_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False).to(device=device, dtype=torch.bfloat16)
    trainable_proj.weight.requires_grad_(True)

    # 3. torch.compile the layer forward
    log(rank, f"  Compiling layer forward...")
    compiled_forward = torch.compile(tp_layer.forward, dynamic=False)

    # 4. Prepare input
    seq_len_local = 512  # Each SP rank gets 512 tokens
    hidden_states = torch.randn(1, seq_len_local, config.hidden_size, device=device, dtype=torch.bfloat16)
    # Make hidden_states require grad (simulates residual stream gradient flow)
    hidden_states = hidden_states.detach().requires_grad_(True)

    # Position embeddings for full_attention
    position_ids = torch.arange(seq_len_local, device=device).unsqueeze(0)
    # Compute rotary embeddings
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    rotary = Qwen3_5TextRotaryEmbedding(config=config).to(device=device)
    # Adjust position_ids for SP rank
    sp_rank_val = dist.get_rank(sp_group)
    position_ids_shifted = position_ids + sp_rank_val * seq_len_local
    position_embeddings = rotary(hidden_states, position_ids_shifted)

    # 5. Forward with activation checkpointing
    log(rank, f"  Running forward with checkpoint...")
    dist.barrier()
    t0 = time.time()

    def fwd_fn(hs):
        # Apply trainable projection first (to create grad path)
        hs_proj = hs + trainable_proj(hs) * 0.01
        # Run through the compiled layer
        out = compiled_forward(
            hs_proj,
            position_embeddings=position_embeddings,
            attention_mask=None,
            position_ids=position_ids_shifted,
        )
        return out

    output = checkpoint(fwd_fn, hidden_states, use_reentrant=False)
    loss = output.sum()

    dist.barrier()
    t1 = time.time()
    log_all(rank, f"  Forward done in {t1 - t0:.3f}s, loss={loss.item():.6f}")

    # 6. Backward
    log_all(rank, f"  Running backward...")
    dist.barrier()
    t2 = time.time()

    loss.backward()

    dist.barrier()
    t3 = time.time()
    log_all(rank, f"  Backward done in {t3 - t2:.3f}s")

    # Check gradient exists
    if trainable_proj.weight.grad is not None:
        grad_norm = trainable_proj.weight.grad.norm().item()
        log_all(rank, f"  trainable_proj grad_norm={grad_norm:.6e}")
    else:
        log_all(rank, f"  WARNING: trainable_proj.weight.grad is None!")

    if hidden_states.grad is not None:
        hs_grad_norm = hidden_states.grad.norm().item()
        log_all(rank, f"  hidden_states grad_norm={hs_grad_norm:.6e}")
    else:
        log_all(rank, f"  WARNING: hidden_states.grad is None!")

    log(rank, f"  {layer_type} layer test PASSED ✓")
    dist.barrier()

    # Cleanup
    del tp_layer, trainable_proj, output, loss, hidden_states
    torch.cuda.empty_cache()


def main():
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    assert world_size == 8, f"This test requires exactly 8 GPUs, got {world_size}"

    log(rank, f"=== SP Layer Backward Debug Test ===")
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

    # Find one full_attention and one linear_attention layer index
    layer_types = list(text_config.layer_types)
    fa_idx = next(i for i, t in enumerate(layer_types) if t == "full_attention")
    la_idx = next(i for i, t in enumerate(layer_types) if t == "linear_attention")
    log(rank, f"Testing full_attention layer idx={fa_idx}, linear_attention layer idx={la_idx}")

    # Suppress dynamo warnings
    import torch._dynamo as _dynamo
    _dynamo.config.cache_size_limit = 64
    if hasattr(_dynamo.config, "recompile_limit"):
        _dynamo.config.recompile_limit = 64

    # Test linear_attention layer first (this is the one that deadlocks)
    log(rank, "")
    log(rank, "=" * 60)
    test_layer("linear_attention", la_idx, text_config, tp_rank, tp_size, tp_group, sp_group, sp_size, device, rank)

    log(rank, "")
    log(rank, "=" * 60)

    # Test full_attention layer
    test_layer("full_attention", fa_idx, text_config, tp_rank, tp_size, tp_group, sp_group, sp_size, device, rank)

    log(rank, "")
    log(rank, "=" * 60)
    log(rank, "ALL TESTS PASSED ✓✓✓")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
