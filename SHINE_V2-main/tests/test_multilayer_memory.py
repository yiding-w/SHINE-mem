"""
Multi-layer memory & time benchmark for Scheme A vs Scheme B.

Aligned with Qwen3.6-27B model config:
  - hidden_size: 5120
  - num_attention_heads: 24
  - num_key_value_heads: 4
  - head_dim: 256
  - intermediate_size: 17408

Measures peak GPU memory and execution time with:
  - Sequence length: 64k (total)
  - TP=2, SP=4 (8 GPUs total)
  - Layers: 1, 2, 4, 8, 16, 32
  - Activation checkpointing enabled
  - Only full_attention layers (SP applies to these)

Run with:
    torchrun --nproc_per_node=8 tests/test_multilayer_memory.py
"""

import os
os.environ["PRECISION_ENHENCEMENT_FA2"] = "0"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gc
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from utils.mytp.ring_attention import (
    sp_flash_attention_contiguous,
    sp_flash_attention_alltoall_zigzag,
)


def thorough_cleanup(device):
    """Aggressively clean up GPU memory to avoid residual effects."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


class AttentionLayer(nn.Module):
    """Full attention layer aligned with Qwen3.6-27B config.
    
    Includes: RMSNorm -> QKV proj -> SP Attention -> O proj -> residual
              RMSNorm -> Gate+Up proj -> SiLU -> Down proj -> residual
    """

    def __init__(self, hidden_size, intermediate_size, num_q_heads, num_kv_heads,
                 head_dim, sp_fn, sp_group):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sp_fn = sp_fn
        self.sp_group = sp_group

        # Attention sub-layer
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)

        # MLP sub-layer
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        """x: [B, S_local, hidden_size]"""
        B, S, _ = x.shape

        # Attention sub-layer with residual
        residual = x
        h = self.input_layernorm(x)

        q = self.q_proj(h).view(B, S, self.num_q_heads, self.head_dim)
        k = self.k_proj(h).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(h).view(B, S, self.num_kv_heads, self.head_dim)

        attn_out = self.sp_fn(q, k, v, sp_group=self.sp_group, causal=True)
        attn_out = attn_out.reshape(B, S, self.num_q_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)

        x = residual + attn_out

        # MLP sub-layer with residual
        residual = x
        h = self.post_attention_layernorm(x)
        gate = torch.nn.functional.silu(self.gate_proj(h))
        up = self.up_proj(h)
        mlp_out = self.down_proj(gate * up)

        x = residual + mlp_out
        return x


class MultiLayerModel(nn.Module):
    """Stack of full attention layers with gradient checkpointing."""

    def __init__(self, num_layers, hidden_size, intermediate_size, num_q_heads,
                 num_kv_heads, head_dim, sp_fn, sp_group, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            AttentionLayer(hidden_size, intermediate_size, num_q_heads,
                          num_kv_heads, head_dim, sp_fn, sp_group)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint:
                x = activation_checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x


def measure(model, x, device, warmup_iters=3, measure_iters=5):
    """
    Measure peak memory and time during forward + backward.
    Returns: (peak_mem_bytes, avg_time_ms)
    """
    # Warmup phase
    for _ in range(warmup_iters):
        x_in = x.clone().requires_grad_(True)
        out = model(x_in)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize(device)
        del out, loss, x_in

    thorough_cleanup(device)
    dist.barrier()

    # Measurement phase
    peak_memories = []
    times_ms = []

    for _ in range(measure_iters):
        thorough_cleanup(device)
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        x_in = x.clone().requires_grad_(True)
        torch.cuda.synchronize(device)
        start_event.record()

        out = model(x_in)
        loss = out.sum()
        loss.backward()

        end_event.record()
        torch.cuda.synchronize(device)

        elapsed_ms = start_event.elapsed_time(end_event)
        peak_mem = torch.cuda.max_memory_allocated(device) - mem_before

        peak_memories.append(peak_mem)
        times_ms.append(elapsed_ms)

        del out, loss, x_in

    thorough_cleanup(device)

    # Median time, min memory
    times_ms.sort()
    median_time = times_ms[len(times_ms) // 2]
    min_peak_mem = min(peak_memories)

    return min_peak_mem, median_time


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    assert world_size == 8, f"This test requires exactly 8 GPUs, got {world_size}"

    # ===== Configuration aligned with Qwen3.6-27B =====
    tp_size = 2
    sp_size = 4
    S_full = 65536  # 64k total sequence length
    S_local = S_full // sp_size  # 16384 per SP rank

    # Qwen3.6-27B text_config
    hidden_size = 5120
    intermediate_size = 17408
    Hq_full = 24       # num_attention_heads
    Hkv_full = 4       # num_key_value_heads
    D = 256            # head_dim

    # TP-local heads
    Hq_local = Hq_full // tp_size   # 12
    Hkv_local = Hkv_full // tp_size  # 2

    # TP-local intermediate size
    intermediate_local = intermediate_size // tp_size  # 8704

    # Create process groups: rank = sp_rank * tp_size + tp_rank
    sp_rank = rank // tp_size
    tp_rank = rank % tp_size

    # SP groups: ranks with same tp_rank across different sp_ranks
    sp_groups = []
    for tp_r in range(tp_size):
        sp_ranks = [sp_r * tp_size + tp_r for sp_r in range(sp_size)]
        sp_groups.append(dist.new_group(sp_ranks))
    my_sp_group = sp_groups[tp_rank]

    if rank == 0:
        print(f"{'#'*80}")
        print(f"# Multi-layer Memory & Time Benchmark (Qwen3.6-27B aligned)")
        print(f"# Model: hidden={hidden_size}, Hq={Hq_full}, Hkv={Hkv_full}, D={D}")
        print(f"#        intermediate={intermediate_size}")
        print(f"# TP-local: Hq={Hq_local}, Hkv={Hkv_local}, intermediate={intermediate_local}")
        print(f"# Parallelism: TP={tp_size}, SP={sp_size}, S_full={S_full}, S_local={S_local}")
        print(f"# Device: {torch.cuda.get_device_name(local_rank)}")
        print(f"# Gradient Checkpointing: ENABLED")
        print(f"# Warmup: 2 iters, Measure: 3 iters (median time, min memory)")
        print(f"{'#'*80}")
        print(flush=True)

    num_layers_list = [1, 2, 4, 8, 16, 32]

    results = {}

    for num_layers in num_layers_list:
        for scheme_name, sp_fn in [
            ("A (contiguous)", sp_flash_attention_contiguous),
            ("B (alltoall+zigzag)", sp_flash_attention_alltoall_zigzag),
        ]:
            # Thorough cleanup before each test
            thorough_cleanup(device)
            dist.barrier()

            # Create model
            model = MultiLayerModel(
                num_layers=num_layers,
                hidden_size=hidden_size,
                intermediate_size=intermediate_local,
                num_q_heads=Hq_local,
                num_kv_heads=Hkv_local,
                head_dim=D,
                sp_fn=sp_fn,
                sp_group=my_sp_group,
                use_checkpoint=True,  # Always use gradient checkpointing
            ).to(device=device, dtype=torch.bfloat16)

            # Model parameter memory
            param_mem = sum(p.numel() * p.element_size() for p in model.parameters())

            # Input tensor
            x = torch.randn(1, S_local, hidden_size, dtype=torch.bfloat16, device=device)

            # Measure (fewer iters for large models to avoid timeout)
            wi = 1 if num_layers >= 16 else 2
            mi = 2 if num_layers >= 16 else 3
            try:
                peak_mem, time_ms = measure(model, x, device, warmup_iters=wi, measure_iters=mi)
                results[(num_layers, scheme_name)] = {
                    "peak_mem": peak_mem,
                    "param_mem": param_mem,
                    "time_ms": time_ms,
                }
                if rank == 0:
                    act_mem = peak_mem - param_mem
                    print(f"  [Done] layers={num_layers:>2}, {scheme_name}: "
                          f"peak={peak_mem/1024**2:.1f}MB, "
                          f"params={param_mem/1024**2:.1f}MB, "
                          f"act={act_mem/1024**2:.1f}MB, "
                          f"time={time_ms:.1f}ms")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results[(num_layers, scheme_name)] = {
                        "peak_mem": -1, "param_mem": param_mem, "time_ms": -1
                    }
                    if rank == 0:
                        print(f"  [OOM] layers={num_layers:>2}, {scheme_name}")
                    thorough_cleanup(device)
                else:
                    raise

            # Cleanup
            del model, x
            thorough_cleanup(device)
            dist.barrier()

    # ========== Print Results ==========
    if rank == 0:
        print(f"\n{'='*100}")
        print(f" RESULTS: Qwen3.6-27B Full Attention Layers (with Gradient Checkpointing)")
        print(f" Config: S_full=64k, TP=2, SP=4, hidden=5120, Hq=24, Hkv=4, D=256, FFN=17408")
        print(f"{'='*100}")

        # Full table
        header = (f"| {'Layers':<6} | {'Scheme':<22} | {'Param (MB)':<11} | "
                  f"{'Peak (MB)':<10} | {'Act (MB)':<10} | {'Time (ms)':<10} |")
        sep = f"|{'-'*8}|{'-'*24}|{'-'*13}|{'-'*12}|{'-'*12}|{'-'*12}|"

        print(f"\n{header}")
        print(sep)

        for num_layers in num_layers_list:
            for scheme_name in ["A (contiguous)", "B (alltoall+zigzag)"]:
                key = (num_layers, scheme_name)
                if key in results:
                    r = results[key]
                    if r["peak_mem"] == -1:
                        print(f"| {num_layers:<6} | {scheme_name:<22} | "
                              f"{r['param_mem']/1024**2:<11.1f} | "
                              f"{'OOM':<10} | {'OOM':<10} | {'OOM':<10} |")
                    else:
                        act_mem = r["peak_mem"] - r["param_mem"]
                        print(f"| {num_layers:<6} | {scheme_name:<22} | "
                              f"{r['param_mem']/1024**2:<11.1f} | "
                              f"{r['peak_mem']/1024**2:<10.1f} | "
                              f"{act_mem/1024**2:<10.1f} | "
                              f"{r['time_ms']:<10.1f} |")
            print(sep)

        # Comparison table
        print(f"\n{'='*100}")
        print(f" A vs B COMPARISON (Gradient Checkpointing ON)")
        print(f"{'='*100}")

        comp_header = (f"| {'Layers':<6} | "
                       f"{'A Peak (MB)':<12} | {'B Peak (MB)':<12} | {'B/A Mem':<8} | "
                       f"{'A Time (ms)':<12} | {'B Time (ms)':<12} | {'B/A Time':<9} |")
        comp_sep = f"|{'-'*8}|{'-'*14}|{'-'*14}|{'-'*10}|{'-'*14}|{'-'*14}|{'-'*11}|"

        print(f"\n{comp_header}")
        print(comp_sep)

        for num_layers in num_layers_list:
            key_a = (num_layers, "A (contiguous)")
            key_b = (num_layers, "B (alltoall+zigzag)")

            if key_a in results and key_b in results:
                ra = results[key_a]
                rb = results[key_b]

                if ra["peak_mem"] > 0 and rb["peak_mem"] > 0:
                    mem_ratio = rb["peak_mem"] / ra["peak_mem"]
                    time_ratio = rb["time_ms"] / ra["time_ms"]
                    print(f"| {num_layers:<6} | "
                          f"{ra['peak_mem']/1024**2:<12.1f} | "
                          f"{rb['peak_mem']/1024**2:<12.1f} | "
                          f"{mem_ratio:<8.3f} | "
                          f"{ra['time_ms']:<12.1f} | "
                          f"{rb['time_ms']:<12.1f} | "
                          f"{time_ratio:<9.3f} |")
                else:
                    a_str = "OOM" if ra["peak_mem"] < 0 else f"{ra['peak_mem']/1024**2:.1f}"
                    b_str = "OOM" if rb["peak_mem"] < 0 else f"{rb['peak_mem']/1024**2:.1f}"
                    print(f"| {num_layers:<6} | "
                          f"{a_str:<12} | {b_str:<12} | {'N/A':<8} | "
                          f"{'N/A':<12} | {'N/A':<12} | {'N/A':<9} |")
        print(comp_sep)

        # Per-layer cost analysis
        print(f"\n{'='*100}")
        print(f" PER-LAYER INCREMENTAL COST")
        print(f"{'='*100}")
        print(f"\n| {'From->To':<10} | {'A Mem/layer':<12} | {'B Mem/layer':<12} | "
              f"{'A Time/layer':<13} | {'B Time/layer':<13} |")
        print(f"|{'-'*12}|{'-'*14}|{'-'*14}|{'-'*15}|{'-'*15}|")

        prev_layers = None
        for num_layers in num_layers_list:
            if prev_layers is not None:
                key_a_cur = (num_layers, "A (contiguous)")
                key_b_cur = (num_layers, "B (alltoall+zigzag)")
                key_a_prev = (prev_layers, "A (contiguous)")
                key_b_prev = (prev_layers, "B (alltoall+zigzag)")

                if all(k in results for k in [key_a_cur, key_b_cur, key_a_prev, key_b_prev]):
                    ra_c = results[key_a_cur]
                    rb_c = results[key_b_cur]
                    ra_p = results[key_a_prev]
                    rb_p = results[key_b_prev]

                    if all(r["peak_mem"] > 0 for r in [ra_c, rb_c, ra_p, rb_p]):
                        delta_layers = num_layers - prev_layers
                        a_mem_per = (ra_c["peak_mem"] - ra_p["peak_mem"]) / delta_layers / 1024**2
                        b_mem_per = (rb_c["peak_mem"] - rb_p["peak_mem"]) / delta_layers / 1024**2
                        a_time_per = (ra_c["time_ms"] - ra_p["time_ms"]) / delta_layers
                        b_time_per = (rb_c["time_ms"] - rb_p["time_ms"]) / delta_layers
                        print(f"| {prev_layers}->{num_layers:<5} | "
                              f"{a_mem_per:<12.1f} | {b_mem_per:<12.1f} | "
                              f"{a_time_per:<13.1f} | {b_time_per:<13.1f} |")
            prev_layers = num_layers

        print(f"\nNotes:")
        print(f"  - All measurements use gradient checkpointing (recompute fwd in bwd)")
        print(f"  - Model config aligned with Qwen3.6-27B: hidden=5120, Hq=24, Hkv=4, D=256, FFN=17408")
        print(f"  - TP=2 splits: Hq_local=12, Hkv_local=2, FFN_local=8704")
        print(f"  - S_full=64k, SP=4, so S_local=16384 per SP rank")
        print(f"  - B/A Mem < 1 means B uses less memory; B/A Time < 1 means B is faster")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
