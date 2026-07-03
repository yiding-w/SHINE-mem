# W-Transform Module

W-Transform 是一个可插拔的变换模块，用于在将 detach_state 的 wdict 注入 LLM forward pass 之前，对其进行变换。

## 目录结构

```
utils/mytransform/
├── __init__.py           # Package entry, exports create_transform
├── create_transform.py   # Factory function (single entry point)
├── identity.py           # Identity transform (no-op)
├── zero.py               # Zero transform (disables injection)
├── compressed_mlp.py     # CompressedMLP transform (learned)
└── README.md             # This file
```

## 统一接口

所有 transform 模块遵循统一的 `nn.Module` 接口：

```python
def forward(self, layer_wdict: dict, layer_idx: int) -> Optional[dict]:
    """
    Args:
        layer_wdict: 单层的 wdict（已按 layer_idx 索引）
        layer_idx: 层索引

    Returns:
        变换后的 wdict（与输入结构相同），或 None（表示跳过注入）
    """
```

## 创建方式

通过统一的工厂函数 `create_transform()` 创建：

```python
from utils.mytransform import create_transform

transform = create_transform(
    cfg={"method": "compressed_mlp", "k": 16, "mlp_ratio": 4, ...},
    model_cfg=model_cfg,
    num_layers=64,
    tp_mode=True,
    tp_rank=rank,
    tp_world=world_size,
    tp_group=tp_group,
    device=device,
    dtype=torch.bfloat16,
    llm_model=llm_model,
)
```

---

## Transform 方法一览

| Method | 类名 | 可学习参数 | 用途 |
|--------|------|-----------|------|
| `identity` | `IdentityTransform` | 0 | 直接透传 wdict，不做任何变换 |
| `zero` | `ZeroTransform` | 0 | 返回 None，完全禁用 wdict 注入 |
| `compressed_mlp` | `CompressedMLPTransform` | ~360M (视模型而定) | 学习一个 Compress-MLP-Decompress 变换 |

---

## 1. Identity Transform

**配置：**
```yaml
w_transform_context:
  method: identity
```

**行为：** 直接返回输入的 `layer_wdict`，不做任何修改。零计算开销，零参数。

**适用场景：**
- Phase C（conversation forward）：wdict 本身就是为 conversation loss 训练的，无需额外变换。
- 调试/对照实验：作为 baseline 对比 learned transform 的效果。

---

## 2. Zero Transform

**配置：**
```yaml
w_transform_context:
  method: zero
```

**行为：** 始终返回 `None`。当 layer wrapper 收到 `None` 时，会跳过 wdict 注入。

**适用场景：**
- 等价于旧版 `dynamic_hypernetwork: false` 的行为。
- 在 context forward 中完全不使用累积的 wdict，hypernetwork 生成不受 accumulated state 影响。

---

## 3. Compressed MLP Transform

**配置：**
```yaml
w_transform_context:
  method: compressed_mlp
  k: 16
  mlp_ratio: 4
  activation: gelu
```

### 设计动机

detach_state 的 W 通过累积 LoRA 的 A@B 得到，用于在 context forward 时注入"记忆"。但 W 的训练目标（conversation loss）和 context forward 的目标之间存在 **objective mismatch**。CompressedMLP 学习一个桥接变换，让 W 在 context forward 中更有效。

### 架构

```
Input W [B, d_in, d_out]
    │
    ▼
┌─────────────────────────────────┐
│  Compress: z = Lᵀ @ W @ R      │  → [B, k, k]
│  (L: [d_in, k], R: [d_out, k]) │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  MLP + Residual:                │
│  z̃ = z + MLP(flatten(z))       │  → [B, k, k]
│  MLP: k² → k²×mlp_ratio → k²  │
│  (last layer zero-init)         │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Decompress: ΔW = L @ z̃ @ Rᵀ   │  → [B, d_in, d_out]
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Residual:                      │
│  W̃ = W + ΔW                    │
└─────────────────────────────────┘
    │
    ▼
Output W̃ [B, d_in, d_out]
```

### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `k` | int | 16 | 压缩维度。W 被压缩到 [B, k, k] 的低维空间。k 越大信息保留越多，但参数量增加。 |
| `mlp_ratio` | int | 4 | MLP 隐藏层扩展倍数。MLP 结构为 Linear(k², k²×mlp_ratio) → Act → Linear(k²×mlp_ratio, k²)。 |
| `activation` | str | "gelu" | MLP 激活函数。支持: `gelu`, `relu`, `silu`, `tanh`。 |

### 关键设计

1. **MLP 最后一层 zero-init**：
   - 训练初期 MLP 输出为 0，z̃ = z（纯残差）
   - ΔW = L @ z @ R^T = (L@L^T) @ W @ (R@R^T)，是 W 在 rank-k 子空间上的投影
   - 当 k << d 时，初始扰动量级约为 (k/d)² ≈ 极小值，对训练几乎无影响

2. **无 gate 设计**：
   - 直接 W̃ = W + ΔW，无额外门控
   - 避免 tanh 饱和问题，不限制 ΔW 的贡献幅度
   - 少一个超参数，代码更简洁
   - MLP zero-init 已保证初始扰动极小，无需 gate 保证 identity

3. **Per-layer, per-projection 独立实例**：
   - 每层每个投影（q_query, k, v, o, gate, up, down 等）有独立的 CompressMLP
   - 不同层有不同的权重分布和功能角色，独立学习更优

### TP 模式

在 TP 模式下，W 是被切分的：

| 切分方式 | 投影类型 | 压缩步骤 | 解压步骤 |
|---------|---------|---------|---------|
| Colwise | q_query, k, v, gate, up | 用 R_local 计算 partial z，all_reduce SUM 得完整 z | 用 R_local 只生成本地 shard 的 ΔW |
| Rowwise | o, down | 用 L_local 计算 partial z，all_reduce SUM 得完整 z | 用 L_local 只生成本地 shard 的 ΔW |

**关键点：**
- L 和 R 参数始终以 **full dimensions** 存储（不切分）
- 压缩时使用 local slice + all_reduce 得到完整 z
- MLP 在完整 z 上运行（每个 rank 独立运行，输入一致所以输出一致）
- 解压时只生成本地 shard 的 ΔW（无需通信）

### 参数规模估算

以 Qwen3.6-27B（64 层，8 个投影，k=16, mlp_ratio=4）为例：

每个 CompressMLP 实例：
- L: d_in × k（如 5120 × 16 = 81,920）
- R: d_out × k（如 6144 × 16 = 98,304）
- MLP: k²→4k²→k²（256→1024→256 ≈ 525K）
- gate: 1

总计 ≈ 64 layers × 8 projections × ~700K ≈ **~360M 参数**（~690 MB at bf16）

### 自动维度检测

`create_transform()` 通过调用 LLM 模型的 `init_lora_dict()` 方法自动探测各投影的 (d_in, d_out) 维度，无需硬编码任何模型特定的维度计算。这使得 CompressedMLP 对任何实现了 `init_lora_dict()` 的模型都是通用的。

### 可选增强

CompressedMLP 支持以下可选增强模块，可通过配置独立开关：

#### Enhancement B: Asymmetric Bases

**动机：** 当前压缩和解压共享同一组 L, R 基底。这意味着 ΔW 只能在 L, R 张成的子空间内修正。Asymmetric 模式使用独立的 L_dec, R_dec 进行解压，让压缩方向（"最有信息量"）和解压方向（"最需要修改"）可以独立优化。

**数学公式：**

```
Compress:   z = L_enc^T @ W @ R_enc     → [B, k, k]
MLP:        z̃ = z + MLP(z)
Decompress: ΔW = L_dec @ z̃ @ R_dec^T    → [B, d_in, d_out]
W̃ = W + ΔW
```

当 `asymmetric: false` 时，退化为 L_enc = L_dec = L, R_enc = R_dec = R（当前行为）。

**配置参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `asymmetric` | bool | `false` | 是否使用独立的解压基底 L_dec, R_dec |

**参数增量：** 每个 CompressMLP 实例增加 d_in×k + d_out×k 个参数（约 +30%）。

**TP 模式：** L_dec 和 R_dec 同样以 full dimensions 存储，forward 时自动切片 local 部分。

---

#### Enhancement E: FiLM Conditioning

**动机：** 压缩后的 z = L^T W R 只保留了 W 在 k 维子空间中的信息，丢失了全局统计特征（如整体大小、分布宽度）。FiLM conditioning 让 MLP 能感知 W 的全局状态，做出更有针对性的变换。

**数学公式：**

```
stats = [mean(W), std(W), ||W||_F, ...] ∈ ℝ^{d_stats}
(γ, β) = FiLM_Net(stats) ∈ ℝ^{k²} × ℝ^{k²}
z_modulated = γ ⊙ flatten(z) + β
z̃ = z + MLP(z_modulated)
```

**两种模式：**

| 模式 | `conditioning` | 行为 |
|------|---------------|------|
| FiLM | `"film"` | stats → FiLM_Net → (γ, β)，对 z_flat 做仿射调制后送入 MLP |
| Concat | `"concat"` | 直接将 stats 拼接到 z_flat 后面作为 MLP 输入 |
| 关闭 | `"none"` | 不使用 conditioning（默认） |

**配置参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `conditioning` | str | `"none"` | 模式选择：`"none"`, `"film"`, `"concat"` |
| `cond_stats` | list | `[]` | 要计算的统计量列表。支持: `mean`, `std`, `norm`, `max`, `min` |

**初始化保证：**
- FiLM_Net 初始化为 γ=1, β=0（恒等调制）
- 训练初期行为与无 conditioning 完全一致

**TP 模式注意：** stats 是在本地 shard 上计算的（不做 all-reduce），因为 FiLM 只是提供辅助信息，局部统计量已足够。

**数值稳定性：** stats 在送入 FiLM_Net 之前会经过 log1p 归一化：
```
stats_normalized = sign(stats) * log1p(|stats|)
```
这将大值（如 norm ≈ 100）压缩到 ≈ 4.6，同时保持小值和符号信息不变，防止 FiLM_Net 接收到量级差异巨大的输入导致数值不稳定。

---

#### Enhancement H: Cross-Projection Attention

**动机：** 当前每个投影（q_query, k, v, o, gate, up, down）的 CompressMLP 完全独立。但同一层内的不同投影之间有强相关性：
- q 和 k 必须协调变化（它们的内积决定 attention score）
- gate 和 up 必须协调变化（它们的 element-wise 乘积决定 FFN 激活）
- v 和 o 必须协调变化（v 的输出经过 o 投影）

Cross-Projection Attention 让各投影的 z_tilde 在解压前互相交流信息。

**架构（Two-Pass 设计）：**

```
同一层内的各投影分别压缩 + MLP：
z̃_q     = compress_and_mlp(W_q)       → [B, k²]
z̃_k     = compress_and_mlp(W_k)       → [B, k²]
z̃_v     = compress_and_mlp(W_v)       → [B, k²]
...（共 num_projs 个）
         │
         ▼ stack
Z = [z̃_q; z̃_k; z̃_v; ...]             → [B, num_projs, k²]
         │
         ▼ Multi-Head Attention (residual)
Z' = Z + MHA(Q=Z, K=Z, V=Z)          → [B, num_projs, k²]
         │
         ▼ unstack + decompress
W̃_q = decompress(W_q, Z'[:, 0, :])
W̃_k = decompress(W_k, Z'[:, 1, :])
...
```

**关键设计：**

1. **Two-Pass 不重构递归逻辑**：
   - `CompressMLP` 拆分为 `compress_and_mlp()` 和 `decompress()` 两个方法
   - `_SingleLayerTransform.forward()` 在 cross-attn 模式下先收集所有叶节点，统一处理
   - 不启用时退化为原来的单 pass 递归（完全向后兼容）

2. **Output projection zero-init**：
   - CrossProjectionAttn 的 out_proj 初始化为全零
   - 训练初期 cross-attn 输出为 0，Z' = Z（纯残差），行为等价于无 cross-attn

3. **极低计算开销**：
   - 序列长度 = num_projs ≈ 7（极短！）
   - 特征维度 = k² = 256
   - 一次 attention 的计算量 ≈ 7² × 256 ≈ 12K FLOPs（可忽略）

4. **不增加 TP 通信**：
   - 所有投影的 z 在 all_reduce 后已经是完整的（每个 rank 上一样）
   - Cross-attn 在完整的 z 上运行，每个 rank 独立运行，输入一致输出一致

**配置参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `blocks` | list | `[mlp]` | 处理块列表，可包含 "mlp" 和 "attn" |
| `attn_num_heads` | int | `4` | attn 块的 head 数量（需整除 k²） |

**参数增量：** 每层每个 attn 块增加一个 CrossProjectionAttn 模块：4 个 Linear(k², k²) ≈ 4 × 256² = 262K 参数/层。64 层共约 16.8M 参数（相比总 360M 约 +4.7%）。每个额外的 mlp 块增加约 0.5M 参数/投影。

---

### 完整公式（blocks: [mlp, attn] 时）

```
--- Per projection (Phase 1: compress + pre-attn MLP blocks) ---
Compress:  z = L_enc^T W R_enc                       [B, k, k]
Stats:     s = sign(stats(W)) * log1p(|stats(W)|)    [B, d_stats]
FiLM:      (γ, β) = FiLM_Net(s)                     [B, k²] × [B, k²]
MLP_0:     z̃ = z + MLP_0(γ ⊙ flatten(z) + β)        [B, k, k]

--- Cross-phase (attn and post-attn blocks) ---
Z = stack(z̃_q, z̃_k, z̃_v, ...)                      [B, num_projs, k²]
Z' = Z + MHA(Z, Z, Z)                               [B, num_projs, k²]
z̃'_q, z̃'_k, ... = unstack(Z')                      [B, k, k] each

--- Per projection (Phase 2: decompress) ---
Decompress:ΔW = L_dec @ z̃' @ R_dec^T                 [B, d_in, d_out]
Output:    W̃ = W + ΔW
```

对于更复杂的 blocks 配置（如 `[mlp, attn, mlp, attn]`）：
```
Compress → FiLM → MLP_0 → Attn_0 → MLP_1 → Attn_1 → Decompress
           ↑ Phase 1 ↑   ↑────── Cross-phase ──────↑   ↑ Phase 2 ↑
```

---

## 典型配置示例

### Phase A (Context Forward) 使用 CompressedMLP（基础版）

```yaml
# configs/detach_state/full_compressedmlp.yaml
w_transform_context:
  method: compressed_mlp
  k: 16
  mlp_ratio: 4
  activation: gelu

w_transform_conversation:
  method: identity
```

### Phase A 使用 CompressedMLP + Asymmetric + FiLM + Cross-Attn（全增强版）

```yaml
w_transform_context:
  method: compressed_mlp
  k: 16
  mlp_ratio: 4
  activation: gelu
  asymmetric: true
  conditioning: film
  cond_stats: [mean, std, norm, max, min]
  blocks: [mlp, attn]
  attn_num_heads: 4

w_transform_conversation:
  method: identity
```

### Phase A 使用更深的网络（MLP-Attn-MLP）

```yaml
w_transform_context:
  method: compressed_mlp
  k: 16
  mlp_ratio: 4
  activation: gelu
  asymmetric: true
  conditioning: film
  cond_stats: [mean, std, norm, max, min]
  blocks: [mlp, attn, mlp]
  attn_num_heads: 4

w_transform_conversation:
  method: identity
```

### Phase A 使用 CompressedMLP + Concat Conditioning

```yaml
w_transform_context:
  method: compressed_mlp
  k: 16
  mlp_ratio: 4
  activation: gelu
  conditioning: concat
  cond_stats: [mean, std]

w_transform_conversation:
  method: identity
```

### 不使用任何 Transform

```yaml
w_transform_context:
  method: identity

w_transform_conversation:
  method: identity
```

### 禁用 Context Forward 的 wdict 注入

```yaml
w_transform_context:
  method: zero

w_transform_conversation:
  method: identity
```
