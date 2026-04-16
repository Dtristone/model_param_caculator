# Model Parameter & Computation Calculator

Calculate the model's **parameter count**, **FLOPs**, and **HBM memory-access** for each component of a transformer model, directly from a HuggingFace `config.json`.

## Features

- **Parse** any HuggingFace `config.json` (Qwen2, GLM-4, GLM-5, LLaMA, Mistral, Mixtral, DeepSeek-V2, …)
- **Visualise** the model architecture as an ASCII diagram
- **Compute** per-component statistics:
  - QKV linear projections (supports MHA / GQA / MQA)
  - **Multi-Latent Attention (MLA)** — low-rank KV compression with absorb optimisation (DeepSeek-V2/V3, GLM-5)
  - **Differential Sparse Attention (DSA)** — indexer + sparse top-k attention (GLM-5)
  - Standard (pure) attention vs. Flash Attention (HBM traffic comparison)
  - FFN / MLP (SwiGLU, GeGLU, standard GELU/ReLU)
  - Mixture-of-Experts (MoE) with router and active-expert FLOPs
  - Layer normalisation (RMSNorm / LayerNorm)
  - Embedding + LM Head
- **KV Cache Analysis**:
  - Per-token KV cache size for each attention type (MHA/GQA/MQA/MLA/DSA+MLA)
  - Compression ratio vs standard MHA
  - KV cache hit ratio modeling (shared prefixes, decode step)
  - FLOPs and HBM write savings from cache reuse
- **Report** rich tables with params, FLOPs, weight memory, activation memory,
  HBM bandwidth, and arithmetic intensity (roofline metric)

## Installation

```bash
pip install rich          # only runtime dependency
```

No PyTorch or HuggingFace Transformers installation required.

## Quick Start

```bash
# Analyse Qwen2-7B
python main.py --config configs/qwen2_7b.json --seq-len 2048

# Analyse GLM-4-9B with Flash Attention
python main.py --config configs/glm4_9b.json --seq-len 4096 --flash-attn

# Analyse Mixtral-8x7B MoE
python main.py --config configs/mixtral_8x7b.json --seq-len 1024 --flash-attn

# Analyse DeepSeek-V2-Lite (MLA attention)
python main.py --config configs/deepseek_v2_lite.json --seq-len 2048

# Analyse GLM-5-9B (DSA + MLA attention)
python main.py --config configs/glm5_9b.json --seq-len 4096

# KV cache analysis with 50% hit ratio
python main.py --config configs/qwen2_7b.json --seq-len 2048 --kv-hit-ratio 0.5

# Use your own HuggingFace config
python main.py --config /path/to/config.json --seq-len 2048 --dtype bf16
```

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--config` / `-c` | required | Path to HuggingFace `config.json` |
| `--seq-len` / `-s` | `2048` | Sequence length |
| `--batch-size` / `-b` | `1` | Batch size |
| `--dtype` / `-d` | `bf16` | `fp32 / fp16 / bf16 / int8 / int4` |
| `--flash-attn` | off | Use Flash Attention HBM model |
| `--kv-hit-ratio` | `0.0` | KV cache hit ratio (0.0-1.0) for cache reuse analysis |
| `--no-diagram` | off | Skip ASCII architecture diagram |
| `--plain` | off | Plain text output (no rich) |

## Project Structure

```
model_param_caculator/
├── plan.md                  # Design document with all math formulas
├── main.py                  # CLI entry point
├── src/
│   ├── config_parser.py     # HuggingFace config -> ModelConfig
│   ├── calculators/
│   │   ├── base.py          # ComputeStats dataclass
│   │   ├── linear.py        # QKV / linear layer stats
│   │   ├── attention.py     # Standard & Flash attention
│   │   ├── mla.py           # Multi-Latent Attention (DeepSeek-V2, GLM-5)
│   │   ├── dsa.py           # Differential Sparse Attention (GLM-5)
│   │   ├── kv_cache.py      # KV cache size & hit ratio analysis
│   │   ├── ffn.py           # FFN / MLP (SwiGLU, standard)
│   │   ├── moe.py           # Mixture-of-Experts
│   │   └── model.py         # Full model aggregation
│   ├── visualizer.py        # ASCII diagram
│   └── report.py            # Rich table report
├── configs/
│   ├── qwen2_7b.json        # Qwen2-7B config
│   ├── glm4_9b.json         # GLM-4-9B config
│   ├── glm5_9b.json         # GLM-5-9B (DSA+MLA) config
│   ├── deepseek_v2_lite.json # DeepSeek-V2-Lite (MLA) config
│   └── mixtral_8x7b.json    # Mixtral-8x7B MoE config
├── examples/
│   ├── example_qwen.py
│   └── example_glm.py
└── tests/
    └── test_calculators.py  # 98 unit tests
```

## Mathematical Foundations

See [plan.md](plan.md) for the full derivation of all formulas including:
- FLOPs counting conventions (1 multiply-add = 2 ops)
- HBM access models for standard vs. flash attention
- Arithmetic intensity and roofline analysis
- MoE active vs. total parameter accounting
- **MLA (Multi-Latent Attention)** low-rank compression and absorb optimisation
- **DSA (Differential Sparse Attention)** indexer + sparse top-k attention
- **KV cache** per-token sizing for each attention type with hit ratio analysis

## Reviewed Computation Formulas

This section summarizes the formulas implemented in code **after reviewing them
against official model sources**.

### Official sources used for the review

- **DeepSeek-V2 official README**: <https://github.com/deepseek-ai/DeepSeek-V2>
  - states that DeepSeek-V2 uses **MLA (Multi-head Latent Attention)**
  - states that MLA reduces KV cache by **93.3%**
- **GLM-5 official implementation**: <https://github.com/THUDM/slime>
  - reviewed file: `slime_plugins/models/glm5/glm5.py`
  - shows the actual **absorbed MLA** path used by GLM-5
  - shows the **DSA indexer** path (`wq_b`, `wk`, `weights_proj`, `lighting_indexer`, `SparseMLA.apply`)

### Core notation

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `s` | query sequence length |
| `h` | hidden size |
| `a` | number of query heads |
| `k` | number of KV heads |
| `d` | standard head dimension |
| `c_q` | MLA query low-rank dimension (`q_lora_rank`) |
| `c_kv` | MLA KV low-rank dimension (`kv_lora_rank`) |
| `d_n` | MLA non-RoPE Q/K head dim (`qk_nope_head_dim`) |
| `d_r` | MLA RoPE Q/K head dim (`qk_rope_head_dim`) |
| `v` | MLA value head dim (`v_head_dim`) |
| `k_sel` | DSA selected top-k tokens (`index_topk`) |
| `N` | number of transformer layers |

### Counting convention

- **1 multiply-add = 2 FLOPs**
- Memory access is measured as **HBM read bytes + HBM write bytes**
- Reported activation memory is a **peak/working-set estimate**, not autograd training memory

### 1. Standard linear layer

For `Y = X W^T` with `X ∈ R^(B·s × in)` and `W ∈ R^(out × in)`:

- **Parameters**: `in · out`
- **FLOPs**: `2 · B · s · in · out`
- **HBM read**: `B · s · in + in · out (+ out if bias)`
- **HBM write**: `B · s · out`

### 2. Standard MHA / GQA / MQA attention

#### Projections

- `Q`: `2 · B · s · h · h`
- `K`: `2 · B · s · h · (k · d)`
- `V`: `2 · B · s · h · (k · d)`
- `O`: `2 · B · s · h · h`

#### Attention kernel

- `QK^T`: `2 · B · a · s² · d = 2 · B · s² · h`
- `softmax`: `5 · B · a · s²`
- `AV`: `2 · B · a · s² · d = 2 · B · s² · h`

Total:

- **Attention FLOPs**: `2 · B · s² · h + 5 · B · a · s² + 2 · B · s² · h`
- **Standard-attention HBM**:
  - read `Q, K, V`
  - write/read attention matrix `A`
  - write output
- **Flash-attention HBM**:
  - read `Q, K, V`
  - write output
  - no materialized `s × s` matrix
  - working-set statistics are two per-row buffers (`m`, `l`)

### 3. MLA (absorbed inference path)

The reviewed code now models the **absorbed MLA runtime path**, which matches
the GLM-5 implementation and optimized MLA inference.

#### MLA cache shape per token

Instead of caching full `K` and `V`, MLA caches:

- compressed KV latent: `c_kv`
- RoPE key part: `d_r`

So:

- **MLA KV cache / token / layer** = `c_kv + d_r`

This is why MLA can be much smaller than standard MHA cache.

#### MLA projection path

Runtime operations:

- `kv_a_proj`: `h → (c_kv + d_r)`
- `q_a_proj`: `h → c_q` (optional)
- `q_b_proj`: `c_q → a · (d_n + d_r)` (or direct `h → a · (d_n + d_r)`)
- `q_absorb`: `q_nope × W_kc`
- `v_expand`: `latent_out × W_vc`
- `o_proj`: `a · v → h`

FLOPs:

- `kv_a_proj`: `2 · B · s · h · (c_kv + d_r)`
- `q_a_proj`: `2 · B · s · h · c_q`
- `q_b_proj`: `2 · B · s · c_q · a · (d_n + d_r)`
- `q_absorb`: `2 · B · s · a · d_n · c_kv`
- `v_expand`: `2 · B · s · a · c_kv · v`
- `o_proj`: `2 · B · s · a · v · h`

Important note:

- the `kv_b_proj` **weights still exist as parameters**
- but in the absorbed path there is **no explicit** `kv_b_proj(hidden)` activation matmul
- instead, its weight is consumed through `W_kc` and `W_vc`

#### MLA attention kernel

Attention runs in compressed space:

- `QK^T` dimension = `c_kv + d_r`
- `AV` latent dimension = `c_kv`

FLOPs:

- `QK^T`: `2 · B · a · s² · (c_kv + d_r)`
- `softmax`: `5 · B · a · s²`
- `AV`: `2 · B · a · s² · c_kv`

HBM model:

- read query tensor: `B · s · a · (c_kv + d_r)`
- read compressed KV cache once: `B · s · (c_kv + d_r)`
- write latent attention output: `B · s · a · c_kv`

### 4. DSA (GLM-5)

GLM-5 adds an **indexer branch** before sparse MLA.

#### 4.1 DSA indexer branch

Official code path:

- `wq_b`: query projection for the indexer
- `wk`: key projection for the indexer
- `weights_proj`: per-head gating weights
- `lighting_indexer`: blockwise top-k token selection

The key correction in the reviewed math is:

- **compute remains O(s²)** because all query-key scores are examined
- but **HBM/activation storage is O(s · k_sel)** because only top-k scores and indices are retained
- we do **not** model the indexer as storing a full `s × s` score matrix

Indexer FLOPs:

- `wq_b`: `2 · B · s · c_q · (idx_heads · idx_dim)`
- `wk`: `2 · B · s · h · idx_dim`
- `weights_proj`: `2 · B · s · h · idx_heads`
- `index score dot-products`: `2 · B · idx_heads · s² · idx_dim`
- `top-k softmax`: `5 · B · s · k_sel`

#### 4.2 DSA sparse MLA main branch

After indexing, each query attends to only `k_sel` tokens:

- `QK^T`: `2 · B · a · s · k_sel · (c_kv + d_r)`
- `softmax`: `5 · B · a · s · k_sel`
- `AV`: `2 · B · a · s · k_sel · c_kv`

Key correction:

- DSA sparse HBM read for KV must scale with **gathered top-k KV access**
- so runtime KV reads are modeled with `O(B · s · k_sel · (c_kv + d_r))`
- not `O(B · s · (c_kv + d_r))`

### 5. KV cache formulas

#### Standard attention

- `K cache / token / layer = k · d`
- `V cache / token / layer = k · d`
- **total** = `2 · k · d`

#### MLA

- **total** = `c_kv + d_r`

For MLA compression ratio, the reference dense MHA cache uses the model's
standard head width:

- `d_std = hidden_size / num_attention_heads`
- reference MHA cache / token / layer = `2 · a · d_std`

#### DSA + MLA

- MLA cache: `c_kv + d_r`
- extra DSA indexer K cache: `index_head_dim`
- **total** = `c_kv + d_r + index_head_dim`

### 6. KV hit ratio assumption

`--kv-hit-ratio` models **cache-producing path reuse** only.

If `hit_ratio = r`, cached-token count is:

- `cached_tokens = floor(r · s)`

Current savings model:

- for standard attention: skip K/V projection work for cached tokens
- for MLA: skip `kv_a_proj` work for cached tokens
- for DSA+MLA: skip `kv_a_proj` and indexer `wk` work for cached tokens

So saved FLOPs are approximately:

- **standard**: `B · cached_tokens · 4 · h · k · d · N`
- **MLA**: `B · cached_tokens · 2 · h · (c_kv + d_r) · N`
- **DSA+MLA**: `B · cached_tokens · [2 · h · (c_kv + d_r) + 2 · h · index_head_dim] · N`

And saved cache writes are:

- `B · cached_tokens · cache_per_token · bytes_per_elem · N`

### 7. Important modeling assumptions

To keep the tool practical, the implementation intentionally uses these assumptions:

1. **Weights are read from HBM for each module invocation**  
   This is a roofline-style bandwidth model, not a detailed SRAM/L2 reuse simulator.

2. **MLA is modeled in absorbed inference form**  
   This matches the GLM-5 runtime path and optimized MLA deployment, not the naive eager-path decomposition some frameworks may use.

3. **DSA indexer storage is modeled as top-k, not dense `s × s`**  
   This matches the official blockwise top-k selection design.

4. **Peak activation memory is a peak, not a sum across layers**  
   Transformer layers are applied sequentially, so peak memory should use `max(...)`, not `sum(...)`.

5. **Embedding lookup reads only accessed rows**  
   It does **not** read the whole embedding table on each forward pass.

## Possible Improvements

The following enhancements could be added in the future:

1. **KV Cache Generation Tracking** — Track how KV cache grows during 
   auto-regressive generation across steps, and model the impact on memory
   as context length increases.

2. **Backward Pass / Training FLOPs** — Add a training mode that estimates backward-pass
   FLOPs (~2× forward) and optimizer state memory (Adam stores fp32 weights + momentum +
   variance ≈ 12 bytes per parameter for mixed-precision training).

3. **Multi-GPU Parallelism Modeling** — Model tensor parallelism (TP), pipeline
   parallelism (PP), and data parallelism (DP) to estimate per-GPU memory and
   communication overhead (all-reduce, point-to-point).

4. **Roofline Performance Prediction** — Accept GPU hardware specs (peak TFLOPS, HBM
   bandwidth in GB/s) and predict actual kernel wall time using the roofline model
   (`time = max(FLOPs/peak_flops, HBM_bytes/bandwidth)`).

5. **More Architecture Support** — Add configs and parser handling for Falcon, MPT, Phi-3,
   Gemma, Command-R, DeepSeek-V3-specific variants, and other emerging architectures.

6. **HuggingFace Hub Integration** — Auto-download `config.json` from the HuggingFace Hub
   by model ID (e.g. `python main.py --model Qwen/Qwen2-7B-Instruct`).

7. **JSON/CSV Export** — Add `--output-format json` and `--output-format csv` options to
   export the computed statistics for programmatic consumption and comparison.

8. **Model Comparison Mode** — Side-by-side comparison of two or more models to visualise
   trade-offs in parameters, FLOPs, and memory.

9. **Activation Checkpointing Analysis** — Estimate memory savings from gradient/activation
   checkpointing during training (trade compute for memory).

10. **Quantization-Aware Modeling** — More accurate FLOPs estimates for quantized models
    (INT8/INT4) which often use dequantize→matmul→requantize patterns with different
    arithmetic intensity characteristics.
