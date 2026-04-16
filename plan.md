# Model Parameter & Computation Calculator — Design Plan

## 1. Goal

Build a tool that, given a HuggingFace `config.json`, will:

1. **Visualise** the model architecture as an ASCII diagram.
2. **Compute** per-component computation (FLOPs) and memory-access (HBM bytes) statistics for:
   - QKV linear projections and output projection
   - Standard (pure) attention
   - Flash attention
   - Feed-forward network / MLP (SwiGLU, GeGLU, standard GELU/ReLU)
   - Mixture-of-Experts (MoE)
3. **Report** a summary table with parameter counts, FLOPs, weight bytes, activation bytes, and HBM bandwidth for each part and the whole model.

Primary example models: **Qwen2-7B** and **GLM-4-9B** (closest public configs to the requested Qwen-8B / GLM-5).

---

## 2. Reference Resources

- [Transformer FLOPs math (Karpathy)](https://github.com/karpathy/nanoGPT/blob/master/model.py)
- [Chinchilla paper: Compute-optimal training](https://arxiv.org/abs/2203.15556)
- [Flash Attention paper](https://arxiv.org/abs/2205.14135)
- [roofline model analysis](https://arxiv.org/abs/2011.01408)
- [llm-analysis (MosaicML)](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/benchmarking/README.md)

---

## 3. Mathematical Foundations

### Notation

| Symbol | Meaning |
|--------|---------|
| B | batch size |
| s | sequence length |
| h | hidden size |
| a | number of Q attention heads |
| k | number of KV heads (GQA/MQA; k=a for MHA) |
| d | head dimension = h / a |
| ffn_h | FFN intermediate size |
| N | number of transformer layers |
| V | vocabulary size |
| E | number of experts (MoE) |
| K | top-K experts per token |

### 3.1 Linear Layer

For Y = X W^T + b, where X ∈ ℝ^(B·s×in), W ∈ ℝ^(out×in):

- **Parameters**: in × out (+ out if bias)
- **FLOPs**: 2 × B × s × in × out  (multiply-add = 2 ops)
- **Weight bytes**: in × out × elem_size
- **HBM reads**: X (B·s·in) + W (in·out) elements
- **HBM writes**: Y (B·s·out) elements

### 3.2 QKV Projections + Output Projection

Per transformer layer:

| Projection | in | out | FLOPs |
|---|---|---|---|
| Q | h | h | 2·B·s·h² |
| K | h | k·d | 2·B·s·h·(k·d) |
| V | h | k·d | 2·B·s·h·(k·d) |
| O | h | h | 2·B·s·h² |

Total QKVO FLOPs = 4·B·s·h·(h + k·d)

Total QKVO params = h·(2h + 2k·d)

### 3.3 Standard (Pure) Attention

For all heads combined:

| Operation | FLOPs | Notes |
|---|---|---|
| QK^T | 2·B·a·s²·d = 2·B·s²·h | per head: 2·s²·d |
| Softmax | 5·B·a·s² | exp, max, sub, sum, div per row |
| A·V | 2·B·a·s²·d = 2·B·s²·h | |

Total attention FLOPs ≈ 4·B·s²·h (+ softmax overhead)

**HBM access (standard attention)**:
- Read Q, K, V: 3·B·s·h elements
- Write/read attention matrix: 2·B·a·s² elements (materialised in HBM)
- Write O: B·s·h elements
- Total: 4·B·s·h + 2·B·a·s²

**Activation memory**:
- Attention matrix A ∈ ℝ^(B×a×s×s) = B·a·s²·elem_size bytes

### 3.4 Flash Attention

Same FLOPs as standard attention, but the attention matrix is **never** materialised in HBM:

**HBM access (flash attention)**:
- Read Q, K, V tiles (each passes through once): 3·B·s·h elements
- Write O: B·s·h elements
- Total: 4·B·s·h  (O(s) not O(s²))

**Activation memory**: O(B·a·s) — only online statistics (max, sum) per row.

### 3.5 FFN / MLP

**SwiGLU** (used by Qwen, GLM, LLaMA-style):
- gate_proj: h → ffn_h  (FLOPs: 2·B·s·h·ffn_h)
- up_proj:   h → ffn_h  (FLOPs: 2·B·s·h·ffn_h)
- SiLU(gate) ⊙ up: element-wise  (FLOPs: 2·B·s·ffn_h)
- down_proj: ffn_h → h  (FLOPs: 2·B·s·ffn_h·h)
- Total FLOPs ≈ 6·B·s·h·ffn_h
- Params: 3·h·ffn_h

**Standard FFN** (GELU/ReLU):
- up: h → ffn_h  (FLOPs: 2·B·s·h·ffn_h)
- down: ffn_h → h  (FLOPs: 2·B·s·ffn_h·h)
- Total FLOPs = 4·B·s·h·ffn_h
- Params: 2·h·ffn_h

### 3.6 Mixture-of-Experts (MoE)

Per token, only K out of E experts are activated:

| Component | FLOPs | Params |
|---|---|---|
| Router (linear) | 2·B·s·h·E | h·E |
| K active experts | K × FFN_FLOPs | E × FFN_params |

**Note**: Parameter count uses all E experts; FLOPs only count K active ones.

### 3.7 Normalisation

**RMSNorm** (used by Qwen, LLaMA):
- FLOPs: ~4·B·s·h (mean-square, rsqrt, normalise, scale)
- Params: h (scale vector)

**LayerNorm**:
- FLOPs: ~8·B·s·h
- Params: 2·h (scale + bias)

### 3.8 Total Model

- Embedding: V·h params, 0 FLOPs (look-up)
- N × transformer layers (QKV + Attn + FFN/MoE + norm)
- Final norm + LM head: V·h params, 2·B·s·V·h FLOPs

---

## 4. Arithmetic Intensity & Roofline

Arithmetic Intensity (AI) = FLOPs / HBM_bytes

- For large-weight linear (decode, s=1): AI ≈ 1 FLOP/byte → **memory-bandwidth bound**
- For attention (long-context prefill, s≫1): AI ≈ s/2 → **compute bound**
- Flash attention improves AI further by reducing HBM traffic

---

## 5. Project Structure

```
model_param_caculator/
├── plan.md                     # This file
├── README.md
├── requirements.txt
├── main.py                     # CLI entry point
├── src/
│   ├── __init__.py
│   ├── config_parser.py        # HuggingFace config → ModelConfig
│   ├── calculators/
│   │   ├── __init__.py
│   │   ├── base.py             # ComputeStats dataclass + helpers
│   │   ├── linear.py           # Linear / QKV projections
│   │   ├── attention.py        # Standard & flash attention
│   │   ├── ffn.py              # FFN / MLP
│   │   ├── moe.py              # MoE
│   │   └── model.py            # Full model aggregation
│   ├── visualizer.py           # ASCII model diagram
│   └── report.py               # Rich-table report
├── configs/
│   ├── qwen2_7b.json
│   └── glm4_9b.json
├── examples/
│   ├── example_qwen.py
│   └── example_glm.py
└── tests/
    ├── __init__.py
    └── test_calculators.py
```

---

## 6. Supported Model Config Fields

### Standard HuggingFace fields
| Field | Description |
|---|---|
| `model_type` | Architecture identifier |
| `hidden_size` | Hidden dimension h |
| `num_hidden_layers` | Transformer layers N |
| `num_attention_heads` | Q heads a |
| `num_key_value_heads` | KV heads k (GQA; default = a) |
| `head_dim` | d (default = h/a) |
| `intermediate_size` | FFN intermediate dimension ffn_h |
| `vocab_size` | Vocabulary size V |
| `tie_word_embeddings` | Whether LM head shares embedding weights |

### GLM-specific aliases
| GLM field | Standard field |
|---|---|
| `num_layers` | `num_hidden_layers` |
| `ffn_hidden_size` | `intermediate_size` |
| `multi_query_group_num` | `num_key_value_heads` |

### MoE fields
| Field | Description |
|---|---|
| `num_experts` | Total number of experts E |
| `num_experts_per_tok` | Top-K activated experts K |
| `moe_intermediate_size` | Expert FFN intermediate size |

---

## 7. CLI Usage

```bash
# Analyse a local config file
python main.py --config configs/qwen2_7b.json --seq-len 2048 --batch-size 1

# Analyse with flash attention
python main.py --config configs/glm4_9b.json --seq-len 4096 --flash-attn

# Show only parameter summary
python main.py --config configs/qwen2_7b.json --mode params

# Show computation breakdown
python main.py --config configs/qwen2_7b.json --seq-len 1024 --mode compute
```
