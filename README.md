# Model Parameter & Computation Calculator

Calculate the model's **parameter count**, **FLOPs**, and **HBM memory-access** for each component of a transformer model, directly from a HuggingFace `config.json`.

## Features

- **Parse** any HuggingFace `config.json` (Qwen2, GLM-4, LLaMA, Mistral, Mixtral, …)
- **Visualise** the model architecture as an ASCII diagram
- **Compute** per-component statistics:
  - QKV linear projections (supports MHA / GQA / MQA)
  - Standard (pure) attention vs. Flash Attention (HBM traffic comparison)
  - FFN / MLP (SwiGLU, GeGLU, standard GELU/ReLU)
  - Mixture-of-Experts (MoE) with router and active-expert FLOPs
  - Layer normalisation (RMSNorm / LayerNorm)
  - Embedding + LM Head
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
│   │   ├── ffn.py           # FFN / MLP (SwiGLU, standard)
│   │   ├── moe.py           # Mixture-of-Experts
│   │   └── model.py         # Full model aggregation
│   ├── visualizer.py        # ASCII diagram
│   └── report.py            # Rich table report
├── configs/
│   ├── qwen2_7b.json        # Qwen2-7B config
│   ├── glm4_9b.json         # GLM-4-9B config
│   └── mixtral_8x7b.json    # Mixtral-8x7B MoE config
├── examples/
│   ├── example_qwen.py
│   └── example_glm.py
└── tests/
    └── test_calculators.py  # 59 unit tests
```

## Mathematical Foundations

See [plan.md](plan.md) for the full derivation of all formulas including:
- FLOPs counting conventions (1 multiply-add = 2 ops)
- HBM access models for standard vs. flash attention
- Arithmetic intensity and roofline analysis
- MoE active vs. total parameter accounting
