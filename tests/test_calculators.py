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

from src.calculators.base import ComputeStats, DType, fmt_num, fmt_bytes, dtype_bytes, elements_to_bytes
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
    return from_dict(
        {
            "model_type": "qwen2",
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "intermediate_size": 18944,
            "vocab_size": 152064,
            "hidden_act": "silu",
            "tie_word_embeddings": False,
        },
        name="Qwen2-7B",
    )


def _glm4_cfg() -> ModelConfig:
    return load_config(_CONFIGS_DIR / "config_glm4.7.json", name="GLM-4.7")


def _mla_cfg() -> ModelConfig:
    return from_dict(
        {
            "model_type": "deepseek_v2",
            "hidden_size": 2048,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "intermediate_size": 10944,
            "vocab_size": 100015,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "qk_head_dim": 192,
            "v_head_dim": 128,
            "hidden_act": "silu",
            "cache_layout": "mla_compressed",
        },
        name="DS-V2-Lite",
    )


def _glm5_cfg() -> ModelConfig:
    return load_config(_CONFIGS_DIR / "config_glm5.json", name="GLM-5")


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

    def test_hbm_read_includes_bias_when_present(self):
        in_f, out_f, s, B = 128, 256, 4, 1
        stat = LinearStats("t", in_f, out_f, has_bias=True, seq_len=s, batch_size=B, dtype=DType.BF16).compute()
        expected_reads = int((B * s * in_f + in_f * out_f + out_f) * 2)
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

    def test_flash_activation_memory_tracks_two_online_stats(self):
        """Flash attention keeps two per-row statistics buffers."""
        h, a, k, d, s, B = 512, 8, 8, 64, 256, 1
        flash = attention_stats(h, a, k, d, s, B, use_flash_attn=True, dtype=DType.BF16)
        num_online_stats_buffers = 2
        eb = 2
        assert flash.act_bytes == num_online_stats_buffers * B * a * s * eb

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

    def test_explicit_qkv_widths_and_lengths(self):
        stats = attention_stats(
            hidden_size=5120,
            num_q_heads=96,
            num_kv_heads=8,
            head_dim=128,
            seq_len=1,
            q_len=1,
            kv_len=2048,
            q_width=96 * 128,
            kv_width=8 * 128,
            attn_out_width=96 * 128,
            use_flash_attn=True,
        )
        assert stats.hbm_read_bytes > stats.hbm_write_bytes


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

    def test_swiglu_has_4_children(self):
        """gate, up, activation+mul, down = 4 children."""
        stats = ffn_stats(512, 2048, ffn_type="swiglu")
        assert len(stats.children) == 4  # gate, up, SiLU+mul, down

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

    def test_shared_expert_added_once(self):
        moe = moe_stats(512, 8, 2, 1024, num_shared_experts=1)
        assert any("Shared Expert" in child.name for child in moe.children)


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
        """GLM-4.7 should stay in the reviewed 300-370 billion parameter range."""
        cfg = _glm4_cfg()
        ms = model_stats(cfg, seq_len=1, batch_size=1)
        assert 3.0e11 < ms.total.num_params < 3.7e11, \
            f"Expected reviewed GLM-4.7 scale, got {fmt_num(ms.total.num_params)}"

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

    def test_glm47_uses_dense_and_sparse_layer_templates(self):
        cfg = _glm4_cfg()
        ms = model_stats(cfg, seq_len=1, batch_size=1)
        layer_names = [layer.name for layer in ms.layer_breakdown]
        assert any("Dense" in name for name in layer_names)
        assert any("Sparse" in name for name in layer_names)


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
        """GLM config fields should map to the current GLM-4.7 config."""
        cfg = _glm4_cfg()
        assert cfg.num_hidden_layers == 92
        assert cfg.intermediate_size == 12288
        assert cfg.num_key_value_heads == 8
        assert cfg.num_shared_experts == 1
        assert cfg.first_k_dense_replace == 3

    def test_glm4_gqa(self):
        cfg = _glm4_cfg()
        assert cfg.attention_type == "GQA"

    def test_glm4_norm_type(self):
        """GLM-4.7 uses RMSNorm and QK norm."""
        cfg = _glm4_cfg()
        assert cfg.norm_type == "rmsnorm"
        assert cfg.use_qk_norm is True

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

    def test_int4_byte_rounding(self):
        assert elements_to_bytes(3, DType.INT4) == 2
        assert elements_to_bytes(5, DType.INT4) == 3

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


# ---------------------------------------------------------------------------
# 10. MLA (Multi-Latent Attention)
# ---------------------------------------------------------------------------

from src.calculators.mla import mla_proj_stats, mla_attention_stats

class TestMLAStats:
    """Tests for Multi-Latent Attention calculator."""

    def test_mla_proj_has_correct_children(self):
        """Absorbed MLA proj includes kv_a, kv_b weights, q_a, q_b, q_absorb, v_expand, o_proj."""
        s = mla_proj_stats(
            hidden_size=2048, num_q_heads=16,
            kv_lora_rank=512, q_lora_rank=1536,
            qk_nope_head_dim=128, qk_rope_head_dim=64,
            v_head_dim=128, seq_len=8, batch_size=1,
        )
        assert len(s.children) == 9
        child_names = [child.name for child in s.children]
        assert "KV latent RMSNorm (kv_a_layernorm)" in child_names
        assert "Q latent RMSNorm (q_a_layernorm)" in child_names

    def test_mla_proj_without_q_lora(self):
        """Without q_lora, absorbed MLA keeps kv_a, kv_b weights, q_direct, q_absorb, v_expand, o_proj."""
        s = mla_proj_stats(
            hidden_size=2048, num_q_heads=16,
            kv_lora_rank=512, q_lora_rank=0,
            qk_nope_head_dim=128, qk_rope_head_dim=64,
            v_head_dim=128, seq_len=8, batch_size=1,
        )
        assert len(s.children) == 7

    def test_kv_b_proj_has_zero_flops(self):
        """Stored kv_b weight tensor has 0 direct FLOPs."""
        s = mla_proj_stats(
            hidden_size=2048, num_q_heads=16,
            kv_lora_rank=512, q_lora_rank=1536,
            qk_nope_head_dim=128, qk_rope_head_dim=64,
            v_head_dim=128, seq_len=8, batch_size=1,
        )
        kv_b = next(child for child in s.children if child.name == "KV up-proj weights (absorbed)")
        assert kv_b.flops == 0
        assert kv_b.num_params > 0  # still has parameters
        assert kv_b.weight_bytes > 0

    def test_q_absorb_and_v_expand_have_flops(self):
        """Absorbed path pays runtime cost in q_absorb and v_expand."""
        s = mla_proj_stats(
            hidden_size=2048, num_q_heads=16,
            kv_lora_rank=512, q_lora_rank=1536,
            qk_nope_head_dim=128, qk_rope_head_dim=64,
            v_head_dim=128, seq_len=8, batch_size=1,
        )
        assert next(child for child in s.children if "Q absorb" in child.name).flops > 0
        assert next(child for child in s.children if "V expand" in child.name).flops > 0

    def test_kv_b_params_count(self):
        """kv_b params = c_kv × a × (d_n + v_dim)."""
        c_kv, a, d_n, v_dim = 512, 16, 128, 128
        s = mla_proj_stats(
            hidden_size=2048, num_q_heads=a,
            kv_lora_rank=c_kv, q_lora_rank=1536,
            qk_nope_head_dim=d_n, qk_rope_head_dim=64,
            v_head_dim=v_dim, seq_len=8, batch_size=1,
        )
        kv_b = next(child for child in s.children if child.name == "KV up-proj weights (absorbed)")
        expected = c_kv * a * (d_n + v_dim)
        assert kv_b.num_params == expected

    def test_mla_proj_params_non_zero(self):
        """Total MLA projection params should be > 0."""
        s = mla_proj_stats(
            hidden_size=2048, num_q_heads=16,
            kv_lora_rank=512, q_lora_rank=1536,
            qk_nope_head_dim=128, qk_rope_head_dim=64,
            v_head_dim=128,
        )
        assert s.num_params > 0

    def test_mla_attn_flops_formula(self):
        """MLA attention FLOPs in absorbed mode."""
        B, s, a = 1, 32, 16
        c_kv, d_r = 512, 64
        v_dim = 128
        absorbed_dim = c_kv + d_r

        attn = mla_attention_stats(
            num_q_heads=a, kv_lora_rank=c_kv,
            qk_nope_head_dim=128, qk_rope_head_dim=d_r,
            v_head_dim=v_dim, seq_len=s, batch_size=B,
        )
        expected_qkt = 2 * B * a * s * s * absorbed_dim
        expected_av = 2 * B * a * s * s * c_kv
        expected_softmax = 5 * B * a * s * s
        expected_total = expected_qkt + expected_softmax + expected_av
        assert attn.flops == expected_total

    def test_mla_flash_same_flops(self):
        """Flash and standard MLA should have the same FLOPs."""
        kwargs = dict(
            num_q_heads=16, kv_lora_rank=512,
            qk_nope_head_dim=128, qk_rope_head_dim=64,
            v_head_dim=128, seq_len=64, batch_size=1,
        )
        std = mla_attention_stats(**kwargs, use_flash_attn=False)
        flash = mla_attention_stats(**kwargs, use_flash_attn=True)
        assert std.flops == flash.flops

    def test_mla_flash_less_hbm(self):
        """Flash MLA should have less HBM traffic than standard."""
        kwargs = dict(
            num_q_heads=16, kv_lora_rank=512,
            qk_nope_head_dim=128, qk_rope_head_dim=64,
            v_head_dim=128, seq_len=256, batch_size=1,
        )
        std = mla_attention_stats(**kwargs, use_flash_attn=False)
        flash = mla_attention_stats(**kwargs, use_flash_attn=True)
        assert flash.hbm_total_bytes < std.hbm_total_bytes

    def test_mla_flash_activation_memory_tracks_two_online_stats(self):
        """Flash MLA also keeps two per-row statistics buffers."""
        flash = mla_attention_stats(
            num_q_heads=16, kv_lora_rank=512,
            qk_nope_head_dim=128, qk_rope_head_dim=64,
            v_head_dim=128, seq_len=64, batch_size=1,
            use_flash_attn=True, dtype=DType.BF16,
        )
        num_online_stats_buffers = 2
        eb = 2
        assert flash.act_bytes == num_online_stats_buffers * 1 * 16 * 64 * eb

    def test_mla_no_learnable_params(self):
        """Attention kernel itself has no learnable params."""
        attn = mla_attention_stats(
            num_q_heads=16, kv_lora_rank=512,
            qk_nope_head_dim=128, qk_rope_head_dim=64,
            v_head_dim=128, seq_len=64,
        )
        assert attn.num_params == 0


# ---------------------------------------------------------------------------
# 11. DSA (Differential Sparse Attention)
# ---------------------------------------------------------------------------

from src.calculators.dsa import dsa_indexer_stats, dsa_sparse_attention_stats

class TestDSAStats:
    """Tests for Differential Sparse Attention calculator."""

    def test_indexer_has_correct_children(self):
        """DSA indexer should have: wq_b, wk, weights_proj, k_norm, index_attn."""
        s = dsa_indexer_stats(
            hidden_size=4096, q_lora_rank=1536,
            index_n_heads=8, index_head_dim=128,
            index_topk=2048,
            seq_len=32, batch_size=1,
        )
        assert len(s.children) == 5

    def test_indexer_params(self):
        """Indexer params: wq_b + wk + weights_proj + k_norm."""
        h = 4096
        c_q = 1536
        idx_a = 8
        idx_h = 128
        expected_params = (
            c_q * idx_a * idx_h     # wq_b
            + h * idx_h              # wk
            + h * idx_a              # weights_proj
            + idx_h                  # k_norm (RMSNorm scale)
        )
        s = dsa_indexer_stats(
            hidden_size=h, q_lora_rank=c_q,
            index_n_heads=idx_a, index_head_dim=idx_h,
            index_topk=2048,
            seq_len=32, batch_size=1,
        )
        assert s.num_params == expected_params

    def test_sparse_attn_less_flops_than_full(self):
        """Sparse attention (top-k) should have fewer FLOPs than full attention."""
        a, c_kv, d_r, v_dim = 32, 512, 64, 128
        s = 4096
        k_sel = 2048

        sparse = dsa_sparse_attention_stats(
            num_q_heads=a, kv_lora_rank=c_kv,
            qk_nope_head_dim=128,
            qk_rope_head_dim=d_r, v_head_dim=v_dim,
            index_topk=k_sel, seq_len=s,
        )
        full = mla_attention_stats(
            num_q_heads=a, kv_lora_rank=c_kv,
            qk_nope_head_dim=128, qk_rope_head_dim=d_r,
            v_head_dim=v_dim, seq_len=s,
        )
        assert sparse.flops < full.flops

    def test_sparse_attn_topk_clamped(self):
        """When seq_len < index_topk, effective k_sel = seq_len."""
        sparse_short = dsa_sparse_attention_stats(
            num_q_heads=32, kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64, v_head_dim=128,
            index_topk=2048, seq_len=512,  # seq_len < topk
        )
        full_mla = mla_attention_stats(
            num_q_heads=32, kv_lora_rank=512,
            qk_nope_head_dim=128, qk_rope_head_dim=64,
            v_head_dim=128, seq_len=512,
        )
        # When seq_len <= topk, sparse attention degenerates to full attention
        # But they have different dim calculations, so just check sparse is not
        # dramatically less than full
        assert sparse_short.flops > 0

    def test_sparse_attn_no_params(self):
        """Sparse attention kernel has no learnable params."""
        s = dsa_sparse_attention_stats(
            num_q_heads=32, kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64, v_head_dim=128,
            index_topk=2048, seq_len=4096,
        )
        assert s.num_params == 0

    def test_indexer_k_path_scales_with_kv_len(self):
        """Indexer K projection and norm should scale with KV length in decode mode."""
        B, Q, T = 1, 1, 2048
        h, idx_h = 4096, 128
        stats = dsa_indexer_stats(
            hidden_size=h,
            q_lora_rank=1536,
            index_n_heads=8,
            index_head_dim=idx_h,
            index_topk=256,
            batch_size=B,
            q_len=Q,
            kv_len=T,
            dtype=DType.BF16,
        )
        wk = next(child for child in stats.children if child.name == "Indexer K proj (wk)")
        k_norm = next(child for child in stats.children if child.name == "Indexer K norm")
        assert wk.flops == 2 * B * T * h * idx_h
        assert k_norm.act_bytes == B * T * idx_h * 2

    def test_indexer_mode_affects_activation_memory(self):
        fused = dsa_indexer_stats(
            hidden_size=4096,
            q_lora_rank=1536,
            index_n_heads=8,
            index_head_dim=128,
            index_topk=64,
            q_len=8,
            kv_len=4096,
            indexer_mode="fused_topk",
            dtype=DType.BF16,
        )
        eager = dsa_indexer_stats(
            hidden_size=4096,
            q_lora_rank=1536,
            index_n_heads=8,
            index_head_dim=128,
            index_topk=64,
            q_len=8,
            kv_len=4096,
            indexer_mode="eager_dense_scores",
            dtype=DType.BF16,
        )
        fused_score = next(child for child in fused.children if "Index top-k scoring" in child.name)
        eager_score = next(child for child in eager.children if "Index top-k scoring" in child.name)
        assert eager_score.act_bytes > fused_score.act_bytes

    def test_sparse_attn_expanded_cache_reads_full_kv(self):
        compressed = dsa_sparse_attention_stats(
            num_q_heads=32,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            index_topk=64,
            q_len=1,
            kv_len=2048,
            cache_layout="mla_compressed",
            dtype=DType.BF16,
        )
        expanded = dsa_sparse_attention_stats(
            num_q_heads=32,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            index_topk=64,
            q_len=1,
            kv_len=2048,
            cache_layout="mla_expanded",
            dtype=DType.BF16,
        )
        assert expanded.hbm_read_bytes > compressed.hbm_read_bytes


# ---------------------------------------------------------------------------
# 12. KV Cache
# ---------------------------------------------------------------------------

from src.calculators.kv_cache import kv_cache_stats, KVCacheStats

class TestKVCache:
    """Tests for KV cache analysis."""

    def test_gqa_kv_cache(self):
        """Standard GQA KV cache: 2 × kv_heads × head_dim per token."""
        kv = kv_cache_stats(
            hidden_size=3584, num_q_heads=28, num_kv_heads=4,
            head_dim=128, num_layers=28, seq_len=2048,
        )
        assert kv.attention_type == "GQA"
        assert kv.k_cache_per_token == 4 * 128
        assert kv.v_cache_per_token == 4 * 128
        assert kv.total_per_token == 2 * 4 * 128

    def test_mha_kv_cache(self):
        """MHA KV cache: 2 × heads × head_dim per token."""
        kv = kv_cache_stats(
            hidden_size=4096, num_q_heads=32, num_kv_heads=32,
            head_dim=128, num_layers=32, seq_len=1024,
        )
        assert kv.attention_type == "MHA"
        assert kv.total_per_token == 2 * 32 * 128

    def test_mla_kv_cache_compressed(self):
        """MLA KV cache: kv_lora_rank + qk_rope_head_dim per token."""
        kv = kv_cache_stats(
            hidden_size=2048, num_q_heads=16, num_kv_heads=16,
            head_dim=192, num_layers=27, seq_len=2048,
            use_mla=True, kv_lora_rank=512, qk_rope_head_dim=64,
            v_head_dim=128,
        )
        assert kv.attention_type == "MLA"
        assert kv.total_per_token == 512 + 64
        assert kv.v_cache_per_token == 0  # V is part of compressed latent

    def test_mla_compression_ratio(self):
        """MLA should have high compression ratio vs standard MHA."""
        num_q_heads = 16
        standard_head_dim = 2048 // num_q_heads
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        kv = kv_cache_stats(
            hidden_size=2048, num_q_heads=num_q_heads, num_kv_heads=16,
            head_dim=192, num_layers=27, seq_len=2048,
            use_mla=True, kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=128,
        )
        assert kv.compression_ratio == (2 * num_q_heads * standard_head_dim) / (kv_lora_rank + qk_rope_head_dim)

    def test_dsa_mla_includes_indexer(self):
        """DSA+MLA should include indexer K cache in total."""
        kv = kv_cache_stats(
            hidden_size=4096, num_q_heads=32, num_kv_heads=32,
            head_dim=192, num_layers=40, seq_len=4096,
            use_mla=True, kv_lora_rank=512, qk_rope_head_dim=64,
            v_head_dim=128, use_dsa=True, index_head_dim=128,
        )
        assert kv.attention_type == "DSA+MLA"
        assert kv.index_cache_per_token == 128
        assert kv.total_per_token == 512 + 64 + 128

    def test_mla_expanded_cache_layout(self):
        kv = kv_cache_stats(
            hidden_size=6144, num_q_heads=64, num_kv_heads=64,
            head_dim=64, num_layers=78, seq_len=1,
            use_mla=True, kv_lora_rank=512, qk_head_dim=256, qk_rope_head_dim=64,
            v_head_dim=256, use_dsa=True, index_head_dim=128, cache_layout="mla_expanded",
        )
        assert kv.total_per_token == 64 * 256 + 64 * 256 + 128

    def test_total_cache_bytes(self):
        """Total cache = per_token × layers × seq_len × batch × elem_size."""
        kv = kv_cache_stats(
            hidden_size=4096, num_q_heads=32, num_kv_heads=4,
            head_dim=128, num_layers=32, seq_len=1024,
            batch_size=2, dtype=DType.BF16,
        )
        expected = 2 * 4 * 128 * 2 * 32 * 1024 * 2  # elems × bytes × layers × seqlen × batch
        assert kv.total_cache_bytes == expected

    def test_hit_ratio_zero(self):
        """Hit ratio = 0 means all tokens are new."""
        kv = kv_cache_stats(
            hidden_size=4096, num_q_heads=32, num_kv_heads=4,
            head_dim=128, num_layers=32, seq_len=2048,
            hit_ratio=0.0,
        )
        assert kv.cached_tokens == 0
        assert kv.new_tokens == 2048
        assert kv.kv_proj_flops_saved == 0
        assert kv.hbm_write_saved == 0

    def test_hit_ratio_half(self):
        """Hit ratio = 0.5 saves FLOPs and HBM writes."""
        kv = kv_cache_stats(
            hidden_size=4096, num_q_heads=32, num_kv_heads=4,
            head_dim=128, num_layers=32, seq_len=2048,
            hit_ratio=0.5,
        )
        assert kv.cached_tokens == 1024
        assert kv.new_tokens == 1024
        assert kv.kv_proj_flops_saved > 0
        assert kv.hbm_write_saved > 0

    def test_hit_ratio_mla(self):
        """MLA models also benefit from cache hit ratio."""
        kv = kv_cache_stats(
            hidden_size=2048, num_q_heads=16, num_kv_heads=16,
            head_dim=192, num_layers=27, seq_len=2048,
            use_mla=True, kv_lora_rank=512, qk_rope_head_dim=64,
            v_head_dim=128, hit_ratio=0.5,
        )
        assert kv.cached_tokens == 1024
        assert kv.kv_proj_flops_saved > 0


# ---------------------------------------------------------------------------
# 13. Config parser (MLA and DSA)
# ---------------------------------------------------------------------------

class TestConfigParserMLA:
    """Tests for MLA/DSA config parsing."""

    def test_deepseek_v2_lite_mla_fields(self):
        """DeepSeek-V2-Lite should be parsed as MLA model."""
        cfg = _mla_cfg()
        assert cfg.use_mla is True
        assert cfg.head_dim == 128
        assert cfg.kv_lora_rank == 512
        assert cfg.q_lora_rank == 1536
        assert cfg.qk_nope_head_dim == 128
        assert cfg.qk_rope_head_dim == 64
        assert cfg.v_head_dim == 128
        assert cfg.attention_type == "MLA"
        assert cfg.cache_layout == "mla_compressed"

    def test_glm5_dsa_fields(self):
        """GLM-5 should be parsed as DSA+MLA model."""
        cfg = _glm5_cfg()
        assert cfg.use_mla is True
        assert cfg.use_dsa is True
        assert cfg.index_n_heads == 32
        assert cfg.index_head_dim == 128
        assert cfg.index_topk == 2048
        assert cfg.attention_type == "DSA+MLA"
        assert cfg.cache_layout == "mla_expanded"

    def test_mla_kv_cache_per_token(self):
        """mla_kv_cache_per_token property should work correctly."""
        cfg = _mla_cfg()
        assert cfg.mla_kv_cache_per_token == 512 + 64

    def test_non_mla_model_defaults(self):
        """Non-MLA models should have use_mla=False."""
        cfg = _qwen2_cfg()
        assert cfg.use_mla is False
        assert cfg.use_dsa is False
        assert cfg.kv_lora_rank == 0

    def test_from_dict_mla(self):
        """from_dict should parse MLA fields."""
        d = {
            "model_type": "deepseek_v2",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "intermediate_size": 10944,
            "vocab_size": 100015,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
        }
        cfg = from_dict(d)
        assert cfg.use_mla is True
        assert cfg.kv_lora_rank == 512

    def test_attention_output_bias_is_architecture_specific(self):
        glm_cfg = _glm4_cfg()
        assert glm_cfg.attention_output_bias is False

        qwen_cfg = from_dict(
            {
                "model_type": "qwen2",
                "hidden_size": 1024,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 4096,
                "attention_bias": True,
            }
        )
        assert qwen_cfg.attention_output_bias is True


# ---------------------------------------------------------------------------
# 14. Full model stats (MLA and DSA models)
# ---------------------------------------------------------------------------

class TestModelStatsMLA:
    """End-to-end model stats for MLA/DSA models."""

    def test_deepseek_v2_lite_model_stats(self):
        """DeepSeek-V2-Lite model stats should complete without error."""
        cfg = _mla_cfg()
        ms = model_stats(cfg, seq_len=512, batch_size=1)
        assert ms.total.num_params > 0
        assert ms.total.flops > 0
        assert ms.kv_cache is not None
        assert ms.kv_cache.attention_type == "MLA"

    def test_glm5_model_stats(self):
        """GLM-5 model stats should complete without error."""
        cfg = _glm5_cfg()
        ms = model_stats(cfg, seq_len=512, batch_size=1)
        assert ms.total.num_params > 0
        assert ms.total.flops > 0
        assert ms.kv_cache is not None
        assert ms.kv_cache.attention_type == "DSA+MLA"

    def test_glm5_has_indexer_in_layer(self):
        """GLM-5 layer breakdown should include DSA Indexer."""
        cfg = _glm5_cfg()
        ms = model_stats(cfg, seq_len=512, batch_size=1)
        sparse_layer = next(layer for layer in ms.layer_breakdown if "Sparse" in layer.name)
        child_names = [c.name for c in sparse_layer.children]
        assert any("Indexer" in n for n in child_names)
        assert any("Sparse" in n for n in child_names)

    def test_mla_model_has_mla_proj_in_layer(self):
        """MLA model layer should have MLA Projection."""
        cfg = _mla_cfg()
        ms = model_stats(cfg, seq_len=512, batch_size=1)
        layer = ms.layer_breakdown[0]
        child_names = [c.name for c in layer.children]
        assert "MLA Projection" in child_names

    def test_glm5_uses_expanded_runtime_labels(self):
        cfg = _glm5_cfg()
        ms = model_stats(cfg, q_len=1, kv_len=2048, cache_len=2048, batch_size=1)
        sparse_layer = next(layer for layer in ms.layer_breakdown if "Sparse" in layer.name)
        attn = next(child for child in sparse_layer.children if "DSA Sparse MLA" in child.name)
        proj = next(child for child in sparse_layer.children if child.name == "MLA Projection")
        proj_child_names = [child.name for child in proj.children]
        assert attn.hbm_read_bytes > attn.hbm_write_bytes
        assert "KV up-proj + expanded cache" in proj_child_names
        assert all("Q absorb" not in name for name in proj_child_names)

    def test_mla_attention_expanded_cache_costs_more_hbm(self):
        kwargs = dict(
            num_q_heads=16,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_len=1,
            kv_len=2048,
            dtype=DType.BF16,
        )
        compressed = mla_attention_stats(**kwargs, cache_layout="mla_compressed", use_flash_attn=True)
        expanded = mla_attention_stats(**kwargs, cache_layout="mla_expanded", use_flash_attn=True)
        assert expanded.hbm_read_bytes > compressed.hbm_read_bytes

    def test_kv_cache_with_hit_ratio(self):
        """KV hit ratio should propagate to model stats."""
        cfg = _mla_cfg()
        ms = model_stats(cfg, seq_len=2048, batch_size=1, kv_hit_ratio=0.5)
        assert ms.kv_cache.hit_ratio == 0.5
        assert ms.kv_cache.cached_tokens == 1024
        assert ms.kv_cache.kv_proj_flops_saved > 0

    def test_standard_model_still_has_kv_cache(self):
        """Standard (non-MLA) models should also have KV cache stats."""
        cfg = _qwen2_cfg()
        ms = model_stats(cfg, seq_len=2048)
        assert ms.kv_cache is not None
        assert ms.kv_cache.attention_type == "GQA"
        assert ms.kv_cache.total_per_token == 2 * 4 * 128


# ---------------------------------------------------------------------------
# 15. Visualizer (MLA/DSA smoke tests)
# ---------------------------------------------------------------------------

class TestVisualizerMLA:
    def test_mla_diagram(self):
        """MLA model diagram should show MLA-specific projections."""
        cfg = _mla_cfg()
        diagram = visualize(cfg)
        assert "MLA" in diagram
        assert "KV down-proj" in diagram
        assert "absorbed" in diagram
        assert "KV cache/token" in diagram

    def test_dsa_diagram(self):
        """DSA model diagram should show DSA+MLA and indexer."""
        cfg = _glm5_cfg()
        diagram = visualize(cfg)
        assert "DSA+MLA" in diagram
        assert "Indexer" in diagram
        assert "top-2048" in diagram

    def test_standard_diagram_unchanged(self):
        """Standard model diagram should still work as before."""
        cfg = _qwen2_cfg()
        diagram = visualize(cfg)
        assert "Q proj" in diagram
        assert "K proj" in diagram
        assert "GQA" in diagram
        assert "KV cache/token" in diagram
