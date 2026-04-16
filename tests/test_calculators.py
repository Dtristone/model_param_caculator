"""
test_calculators.py
===================
Unit tests for the model computation calculators.

Tests verify:
  1. Linear layer FLOPs and memory formulae
  2. QKV projection stats (MHA and GQA)
  3. Standard vs Flash attention HBM traffic
  4. FFN variants (SwiGLU, standard)
  5. MoE layer routing and expert counting
  6. Full model parameter counts for known architectures
  7. Config parser (Qwen2, GLM-4)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.calculators.base import ComputeStats, DType, fmt_num, fmt_bytes, dtype_bytes
from src.calculators.linear import LinearStats, qkv_proj_stats, output_proj_stats
from src.calculators.attention import attention_stats
from src.calculators.ffn import ffn_stats
from src.calculators.moe import moe_stats
from src.calculators.model import model_stats
from src.config_parser import load_config, from_dict, ModelConfig
from src.visualizer import visualize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def _qwen2_cfg() -> ModelConfig:
    return load_config(_CONFIGS_DIR / "qwen2_7b.json", name="Qwen2-7B")


def _glm4_cfg() -> ModelConfig:
    return load_config(_CONFIGS_DIR / "glm4_9b.json", name="GLM-4-9B")


# ---------------------------------------------------------------------------
# 1. Linear layer
# ---------------------------------------------------------------------------

class TestLinearStats:
    def test_flops_formula(self):
        """FLOPs = 2 * B * s * in * out."""
        s = LinearStats("test", in_features=512, out_features=1024, seq_len=16, batch_size=2)
        stats = s.compute()
        expected_flops = 2 * 2 * 16 * 512 * 1024
        assert stats.flops == expected_flops

    def test_params_no_bias(self):
        s = LinearStats("test", in_features=256, out_features=512, has_bias=False)
        assert s.compute().num_params == 256 * 512

    def test_params_with_bias(self):
        s = LinearStats("test", in_features=256, out_features=512, has_bias=True)
        assert s.compute().num_params == 256 * 512 + 512

    def test_weight_bytes_bf16(self):
        s = LinearStats("test", in_features=64, out_features=128, dtype=DType.BF16)
        stats = s.compute()
        assert stats.weight_bytes == 64 * 128 * 2   # 2 bytes per element

    def test_weight_bytes_fp32(self):
        s = LinearStats("test", in_features=64, out_features=128, dtype=DType.FP32)
        stats = s.compute()
        assert stats.weight_bytes == 64 * 128 * 4   # 4 bytes per element

    def test_hbm_read_includes_input_and_weight(self):
        in_f, out_f, s, B = 128, 256, 4, 1
        stat = LinearStats("t", in_f, out_f, seq_len=s, batch_size=B, dtype=DType.BF16).compute()
        expected_reads = int((B * s * in_f + in_f * out_f) * 2)
        assert stat.hbm_read_bytes == expected_reads

    def test_hbm_write_is_output(self):
        in_f, out_f, s, B = 128, 256, 4, 1
        stat = LinearStats("t", in_f, out_f, seq_len=s, batch_size=B, dtype=DType.BF16).compute()
        expected_writes = int(B * s * out_f * 2)
        assert stat.hbm_write_bytes == expected_writes

    def test_arithmetic_intensity(self):
        # For large weights and small batch, AI should be close to 1.0 FLOP/byte
        # (memory-bandwidth bound)
        s = LinearStats("t", 4096, 4096, seq_len=1, batch_size=1, dtype=DType.BF16)
        stats = s.compute()
        # AI = 2*in*out / (in*out*2 + 1*in*2 + 1*out*2) ≈ 1.0 for large in/out
        assert 0.5 <= stats.arithmetic_intensity <= 2.0


# ---------------------------------------------------------------------------
# 2. QKV projections
# ---------------------------------------------------------------------------

class TestQKVProjStats:
    def test_mha_params(self):
        """MHA: Q+K+V all use h → h projections."""
        h, a, d = 512, 8, 64  # a * d = h = 512
        stats = qkv_proj_stats(h, a, a, d, seq_len=1)
        # Q: 512*512, K: 512*512, V: 512*512 → 3 * 512² = 786432
        expected = 3 * h * h
        assert stats.num_params == expected

    def test_gqa_kv_param_reduction(self):
        """GQA: K and V heads are fewer than Q heads → smaller params."""
        h, a, k, d = 512, 8, 2, 64  # 8 Q heads, 2 KV heads
        mha = qkv_proj_stats(h, a, a, d, seq_len=1)
        gqa = qkv_proj_stats(h, a, k, d, seq_len=1)
        assert gqa.num_params < mha.num_params
        # GQA KV reduction: (a - k) * h * d fewer params per K and V
        expected_kv = h * (k * d)  # each of K, V
        expected_q  = h * h
        expected_total = expected_q + 2 * expected_kv
        assert gqa.num_params == expected_total

    def test_qkv_flops(self):
        """FLOPs = 2 * s * h * (q_out + k_out + v_out)."""
        h, a, k, d, s = 512, 8, 2, 64, 32
        stats = qkv_proj_stats(h, a, k, d, seq_len=s)
        q_out  = a * d   # 512
        kv_out = k * d   # 128
        expected = 2 * s * h * (q_out + kv_out + kv_out)
        assert stats.flops == expected

    def test_children_count(self):
        stats = qkv_proj_stats(512, 8, 8, 64, seq_len=1)
        assert len(stats.children) == 3  # Q, K, V


# ---------------------------------------------------------------------------
# 3. Attention
# ---------------------------------------------------------------------------

class TestAttentionStats:
    def test_flops_formula(self):
        """Attn FLOPs = 4*B*s²*h + softmax overhead."""
        h, a, k, d, s, B = 512, 8, 8, 64, 64, 1
        stats = attention_stats(h, a, k, d, s, B, use_flash_attn=False)
        # 2*B*a*s²*d (QK^T) + 5*B*a*s² (softmax) + 2*B*a*s²*d (AV)
        expected = 2 * B * a * s * s * d + 5 * B * a * s * s + 2 * B * a * s * s * d
        assert stats.flops == expected

    def test_flash_same_flops(self):
        """Flash attention has the same FLOPs as standard."""
        h, a, k, d, s = 512, 8, 8, 64, 128
        std   = attention_stats(h, a, k, d, s, use_flash_attn=False)
        flash = attention_stats(h, a, k, d, s, use_flash_attn=True)
        assert std.flops == flash.flops

    def test_flash_less_hbm(self):
        """Flash attention should have strictly less HBM traffic for s > 1."""
        h, a, k, d, s = 512, 8, 8, 64, 128
        std   = attention_stats(h, a, k, d, s, use_flash_attn=False)
        flash = attention_stats(h, a, k, d, s, use_flash_attn=True)
        assert flash.hbm_total_bytes < std.hbm_total_bytes

    def test_flash_less_activation_memory(self):
        """Flash attention should not materialise the full s×s attention matrix."""
        h, a, k, d, s = 512, 8, 8, 64, 256
        std   = attention_stats(h, a, k, d, s, use_flash_attn=False)
        flash = attention_stats(h, a, k, d, s, use_flash_attn=True)
        assert flash.act_bytes < std.act_bytes

    def test_standard_attn_has_attn_matrix_in_hbm(self):
        """Standard attention must read/write the attention matrix (B*a*s²)."""
        h, a, k, d, s, B = 512, 8, 8, 64, 64, 1
        eb = 2   # bf16
        std = attention_stats(h, a, k, d, s, B, use_flash_attn=False, dtype=DType.BF16)
        attn_matrix_bytes = B * a * s * s * eb
        # The attention matrix appears in both hbm_read and hbm_write
        assert std.hbm_write_bytes >= attn_matrix_bytes

    def test_no_learnable_params(self):
        """The attention kernel has no learnable parameters."""
        stats = attention_stats(512, 8, 8, 64, 64)
        assert stats.num_params == 0

    def test_gqa_reduces_kv_bandwidth(self):
        """GQA should use less KV read bandwidth than MHA."""
        h, a, d, s = 512, 8, 64, 64
        mha = attention_stats(h, a, a, d, s, use_flash_attn=True)   # k=a=8
        gqa = attention_stats(h, a, 2, d, s, use_flash_attn=True)   # k=2
        # With flash attention, HBM = Q+K+V+O; K+V in GQA is smaller
        assert gqa.hbm_total_bytes < mha.hbm_total_bytes


# ---------------------------------------------------------------------------
# 4. FFN
# ---------------------------------------------------------------------------

class TestFFNStats:
    def test_swiglu_params(self):
        """SwiGLU has 3 weight matrices: gate, up, down."""
        h, ffn_h = 512, 2048
        stats = ffn_stats(h, ffn_h, ffn_type="swiglu")
        # gate: h*ffn_h + up: h*ffn_h + down: ffn_h*h = 3*h*ffn_h
        assert stats.num_params == 3 * h * ffn_h

    def test_standard_params(self):
        """Standard FFN has 2 weight matrices."""
        h, ffn_h = 512, 2048
        stats = ffn_stats(h, ffn_h, ffn_type="standard")
        assert stats.num_params == 2 * h * ffn_h

    def test_swiglu_flops(self):
        """SwiGLU FLOPs ≈ 6*B*s*h*ffn_h (matmul part)."""
        h, ffn_h, s, B = 512, 2048, 32, 1
        stats = ffn_stats(h, ffn_h, ffn_type="swiglu", seq_len=s, batch_size=B)
        # gate + up + down = 2sh×ffn_h + 2sh×ffn_h + 2s×ffn_h×h = 6sh×ffn_h
        matmul_flops = 6 * B * s * h * ffn_h
        # total_flops also includes small element-wise term
        assert stats.flops >= matmul_flops

    def test_standard_flops(self):
        """Standard FFN FLOPs ≈ 4*B*s*h*ffn_h."""
        h, ffn_h, s, B = 512, 2048, 32, 1
        stats = ffn_stats(h, ffn_h, ffn_type="standard", seq_len=s, batch_size=B)
        matmul_flops = 4 * B * s * h * ffn_h
        assert stats.flops >= matmul_flops

    def test_swiglu_has_3_children(self):
        """gate, up, activation+mul, down = 4 children."""
        stats = ffn_stats(512, 2048, ffn_type="swiglu")
        assert len(stats.children) == 4

    def test_standard_has_3_children(self):
        """up, activation, down = 3 children."""
        stats = ffn_stats(512, 2048, ffn_type="standard")
        assert len(stats.children) == 3

    def test_swiglu_more_params_than_standard(self):
        h, ffn_h = 512, 2048
        swi = ffn_stats(h, ffn_h, "swiglu")
        std = ffn_stats(h, ffn_h, "standard")
        assert swi.num_params > std.num_params

    def test_weight_bytes_dtype(self):
        h, ffn_h = 128, 512
        bf16 = ffn_stats(h, ffn_h, "swiglu", dtype=DType.BF16)
        fp32 = ffn_stats(h, ffn_h, "swiglu", dtype=DType.FP32)
        assert fp32.weight_bytes == 2 * bf16.weight_bytes


# ---------------------------------------------------------------------------
# 5. MoE
# ---------------------------------------------------------------------------

class TestMoEStats:
    def test_active_flops_vs_dense(self):
        """MoE with K=2 out of E=8 experts should compute K×FFN_FLOPs."""
        h, ffn_h, s, B = 512, 2048, 32, 1
        E, K = 8, 2
        moe = moe_stats(h, E, K, ffn_h, ffn_type="swiglu", seq_len=s, batch_size=B)

        # Single expert FFN FLOPs
        one_expert = ffn_stats(h, ffn_h, "swiglu", seq_len=s, batch_size=B)
        # Router FLOPs
        from src.calculators.linear import LinearStats
        router = LinearStats("router", h, E, seq_len=s, batch_size=B).compute()

        # Total active FLOPs = router + K * expert_flops
        expected = router.flops + K * one_expert.flops
        assert moe.flops == expected

    def test_all_expert_params_in_memory(self):
        """All E experts' parameters should be counted (they all sit in HBM)."""
        h, ffn_h = 512, 2048
        E, K = 8, 2
        moe = moe_stats(h, E, K, ffn_h, ffn_type="swiglu")
        one_expert = ffn_stats(h, ffn_h, "swiglu")
        from src.calculators.linear import LinearStats
        router_params = LinearStats("r", h, E).compute().num_params
        expected_params = router_params + E * one_expert.num_params
        assert moe.num_params == expected_params

    def test_router_child_present(self):
        moe = moe_stats(512, 8, 2, 2048)
        assert any("Router" in c.name for c in moe.children)

    def test_expert_param_scaling(self):
        """Doubling E doubles expert parameters."""
        h, ffn_h = 512, 1024
        moe4 = moe_stats(h, 4, 2, ffn_h)
        moe8 = moe_stats(h, 8, 2, ffn_h)
        # router params scale with E; expert params scale with E
        assert moe8.num_params > moe4.num_params


# ---------------------------------------------------------------------------
# 6. Full model parameter counts
# ---------------------------------------------------------------------------

class TestModelStats:
    def test_qwen2_7b_total_params(self):
        """Qwen2-7B should have ~7.6B parameters."""
        cfg = _qwen2_cfg()
        ms = model_stats(cfg, seq_len=1, batch_size=1)
        # Expected: ~7.6B
        assert 7.0e9 < ms.total.num_params < 8.2e9, \
            f"Expected ~7.6B, got {fmt_num(ms.total.num_params)}"

    def test_glm4_9b_total_params(self):
        """GLM-4-9B should have ~9B parameters."""
        cfg = _glm4_cfg()
        ms = model_stats(cfg, seq_len=1, batch_size=1)
        assert 8.5e9 < ms.total.num_params < 10.0e9, \
            f"Expected ~9B, got {fmt_num(ms.total.num_params)}"

    def test_flops_scale_with_seq_len_squared_in_attn(self):
        """Attention FLOPs should scale as s² when seq_len increases."""
        cfg = _qwen2_cfg()
        ms1 = model_stats(cfg, seq_len=512,  batch_size=1)
        ms2 = model_stats(cfg, seq_len=1024, batch_size=1)
        # Total FLOPs will scale roughly between s and s²; with attention scaling
        # more, we just verify that doubling s increases FLOPs by more than 2×
        assert ms2.total.flops > 2 * ms1.total.flops

    def test_flash_attn_less_hbm_than_standard(self):
        """Model with flash attention should have less total HBM traffic."""
        cfg = _qwen2_cfg()
        std   = model_stats(cfg, seq_len=2048, batch_size=1, use_flash_attn=False)
        flash = model_stats(cfg, seq_len=2048, batch_size=1, use_flash_attn=True)
        assert flash.total.hbm_total_bytes < std.total.hbm_total_bytes

    def test_per_layer_params_reasonable(self):
        """Per-layer parameters should be a reasonable fraction of total.

        For Qwen2-7B (N=28 layers), one layer is ~3% of total params since
        embedding + LM head also contribute significantly.  We verify that
        all_layers.num_params == N * per_layer.num_params (exact scaling).
        """
        cfg = _qwen2_cfg()
        ms = model_stats(cfg, seq_len=1)
        N = cfg.num_hidden_layers
        # All-layers params should equal exactly N * per_layer params
        assert ms.all_layers.num_params == ms.per_layer.num_params * N
        # And per-layer should be a non-trivial fraction of total (>0.5%)
        layer_frac = ms.per_layer.num_params / ms.total.num_params
        assert layer_frac > 0.005

    def test_embedding_params(self):
        """Embedding params = vocab_size * hidden_size."""
        cfg = _qwen2_cfg()
        ms = model_stats(cfg, seq_len=1)
        expected = cfg.vocab_size * cfg.hidden_size
        assert ms.embedding.num_params == expected

    def test_batch_scales_flops(self):
        """Doubling batch size should double FLOPs."""
        cfg = _qwen2_cfg()
        ms1 = model_stats(cfg, seq_len=64, batch_size=1)
        ms2 = model_stats(cfg, seq_len=64, batch_size=2)
        assert ms2.total.flops == 2 * ms1.total.flops

    def test_dtype_affects_weight_bytes(self):
        """FP32 model should use 2× the weight bytes of BF16."""
        cfg = _qwen2_cfg()
        bf16 = model_stats(cfg, seq_len=1, dtype=DType.BF16)
        fp32 = model_stats(cfg, seq_len=1, dtype=DType.FP32)
        assert fp32.total.weight_bytes == 2 * bf16.total.weight_bytes


# ---------------------------------------------------------------------------
# 7. Config parser
# ---------------------------------------------------------------------------

class TestConfigParser:
    def test_qwen2_model_type(self):
        cfg = _qwen2_cfg()
        assert cfg.model_type == "qwen2"

    def test_qwen2_dimensions(self):
        cfg = _qwen2_cfg()
        assert cfg.hidden_size == 3584
        assert cfg.num_hidden_layers == 28
        assert cfg.num_attention_heads == 28
        assert cfg.num_key_value_heads == 4
        assert cfg.intermediate_size == 18944
        assert cfg.vocab_size == 152064

    def test_qwen2_gqa_detection(self):
        cfg = _qwen2_cfg()
        assert cfg.attention_type == "GQA"

    def test_qwen2_ffn_type(self):
        cfg = _qwen2_cfg()
        assert cfg.ffn_type == "swiglu"

    def test_glm4_field_aliases(self):
        """GLM config uses num_layers and ffn_hidden_size instead of standard names."""
        cfg = _glm4_cfg()
        assert cfg.num_hidden_layers == 40
        assert cfg.intermediate_size == 13696
        assert cfg.num_key_value_heads == 2

    def test_glm4_gqa(self):
        cfg = _glm4_cfg()
        assert cfg.attention_type == "GQA"

    def test_from_dict(self):
        """from_dict should produce a valid ModelConfig."""
        d = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "hidden_act": "silu",
        }
        cfg = from_dict(d, name="LLaMA-7B")
        assert cfg.hidden_size == 4096
        assert cfg.ffn_type == "swiglu"

    def test_moe_detection(self):
        """Mixtral-style config should set is_moe=True."""
        d = {
            "model_type": "mixtral",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        }
        cfg = from_dict(d, name="Mixtral-8x7B")
        assert cfg.is_moe is True
        assert cfg.num_experts == 8
        assert cfg.num_experts_per_tok == 2

    def test_head_dim_inference(self):
        """head_dim should be inferred from hidden_size / num_attention_heads."""
        d = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        }
        cfg = from_dict(d)
        assert cfg.head_dim == 128  # 4096 / 32


# ---------------------------------------------------------------------------
# 8. Visualizer (smoke test)
# ---------------------------------------------------------------------------

class TestVisualizer:
    def test_returns_string(self):
        cfg = _qwen2_cfg()
        diagram = visualize(cfg)
        assert isinstance(diagram, str)
        assert len(diagram) > 0

    def test_contains_model_name(self):
        cfg = _qwen2_cfg()
        diagram = visualize(cfg)
        assert "Qwen2-7B" in diagram

    def test_contains_dimensions(self):
        cfg = _qwen2_cfg()
        diagram = visualize(cfg)
        assert str(cfg.hidden_size) in diagram
        assert str(cfg.num_hidden_layers) in diagram

    def test_flash_attn_label(self):
        cfg = _qwen2_cfg()
        assert "flash attn" in visualize(cfg, use_flash_attn=True)
        assert "std attn"   in visualize(cfg, use_flash_attn=False)

    def test_moe_diagram(self):
        """MoE config should show expert routing in diagram."""
        d = {
            "model_type": "mixtral",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        }
        cfg = from_dict(d, name="Mixtral-8x7B")
        diagram = visualize(cfg)
        assert "Router" in diagram


# ---------------------------------------------------------------------------
# 9. Base helpers
# ---------------------------------------------------------------------------

class TestBaseHelpers:
    def test_fmt_num_billions(self):
        assert fmt_num(7_600_000_000) == "7.60B"

    def test_fmt_num_millions(self):
        assert fmt_num(544_000_000) == "544.00M"

    def test_fmt_bytes_gib(self):
        result = fmt_bytes(2 * 2**30)
        assert "GiB" in result

    def test_dtype_bytes(self):
        assert dtype_bytes(DType.FP32) == 4
        assert dtype_bytes(DType.BF16) == 2
        assert dtype_bytes(DType.INT8) == 1
        assert dtype_bytes(DType.INT4) == 0.5

    def test_compute_stats_add(self):
        a = ComputeStats("a", num_params=100, flops=200, weight_bytes=300,
                         hbm_read_bytes=400, hbm_write_bytes=50)
        b = ComputeStats("b", num_params=10,  flops=20,  weight_bytes=30,
                         hbm_read_bytes=40,  hbm_write_bytes=5)
        c = a + b
        assert c.num_params == 110
        assert c.flops == 220
        assert c.hbm_total_bytes == 495

    def test_compute_stats_aggregate_children(self):
        parent = ComputeStats("parent")
        child1 = ComputeStats("c1", num_params=100, flops=200)
        child2 = ComputeStats("c2", num_params=50,  flops=100)
        parent.children = [child1, child2]
        parent.aggregate_children()
        assert parent.num_params == 150
        assert parent.flops == 300
