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
    └── test_calculators.py  # 97 unit tests
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
   DeepSeek-V2 (MLA attention), Gemma, Command-R, and other emerging architectures.

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
