"""
mla.py
======
FLOPs, parameters, and memory-access statistics for Multi-Latent Attention (MLA).

MLA was introduced in DeepSeek-V2 (arXiv 2405.04434) and is also used in GLM-5.
It replaces standard multi-head K/V projections with a low-rank compression,
dramatically reducing KV cache size while maintaining model quality.

Architecture
------------
                       hidden_states (h)
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                   │
     q_a_proj (h→c_q)       │            kv_a_proj (h→c_kv+d_r)
          │                  │                   │
      q_norm                 │               kv_norm   ← cached: c_kv + d_r
          │                  │                   │
     q_b_proj             (skip if             kv_b_proj
  (c_q → a*(d_n+d_r))    q_lora=0)        (c_kv → a*(d_n+v_dim))
          │                  │                   │
     split Q:             Q proj              split KV:
   q_nope, q_rope       (h→h)             k_nope, v  and  k_rope
          │                  │                   │
          └─────────────  Attention  ────────────┘
                             │
                          O proj (a*v_dim → h)

Notation
--------
  h       = hidden_size
  a       = num_attention_heads
  c_kv    = kv_lora_rank          (KV compression dimension)
  c_q     = q_lora_rank           (Q compression dimension, 0 = skip)
  d_n     = qk_nope_head_dim      (non-RoPE part of Q/K head dim)
  d_r     = qk_rope_head_dim      (RoPE part of Q/K head dim)
  v_dim   = v_head_dim             (value head dimension)

KV Cache per token = c_kv + d_r   (instead of 2 * a * d for standard MHA)

FLOPs (absorb-optimised inference path)
---------------------------------------
In inference with "weight absorption", the kv_b_proj weight is absorbed into
the Q projection and O projection, so attention operates directly on the
compressed KV latent.  The FLOPs breakdown:

  Projections:
    kv_a_proj:  2 * B * s * h * (c_kv + d_r)           — compress KV
    q_a_proj:   2 * B * s * h * c_q                     — compress Q (if used)
    q_b_proj:   2 * B * s * c_q * a * (d_n + d_r)       — expand Q
    o_proj:     2 * B * s * a * v_dim * h               — output projection

  Attention kernel (per head):
    QK^T:  2 * B * a * s * s * (d_n + d_r)             — using absorbed Q
    AV:    2 * B * a * s * s * v_dim
    softmax: 5 * B * a * s²

  Note: kv_b_proj itself is NOT computed at inference (absorbed), but its
  parameters still exist and consume weight memory.
"""

from __future__ import annotations

from .base import ComputeStats, DType, dtype_bytes
from .linear import LinearStats


def mla_proj_stats(
    hidden_size: int,
    num_q_heads: int,
    kv_lora_rank: int,
    q_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    seq_len: int = 1,
    batch_size: int = 1,
    has_bias: bool = False,
    dtype: DType = DType.BF16,
) -> ComputeStats:
    """Compute FLOPs/params for MLA projection layers.

    Returns stats for the full set of MLA projections:
    kv_a_proj, kv_b_proj, q_a_proj (optional), q_b_proj, and o_proj.

    The kv_b_proj FLOPs are set to 0 (absorbed at inference) but its
    parameters and weight bytes are counted since they reside in memory.
    """
    a = num_q_heads
    c_kv = kv_lora_rank
    c_q = q_lora_rank
    d_n = qk_nope_head_dim
    d_r = qk_rope_head_dim
    v_dim = v_head_dim
    q_head_dim = d_n + d_r

    stats = ComputeStats(name="MLA Projection")

    # --- kv_a_proj: h → c_kv + d_r  (KV down-projection + RoPE key) ------
    kv_a = LinearStats(
        name="KV down-proj (kv_a)",
        in_features=hidden_size,
        out_features=c_kv + d_r,
        has_bias=has_bias,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
    ).compute()
    stats.children.append(kv_a)

    # --- kv_b_proj: c_kv → a * (d_n + v_dim) ---
    # This weight is absorbed into Q/O at inference → 0 FLOPs at inference,
    # but weight params and bytes still count.
    kv_b_params = c_kv * a * (d_n + v_dim)
    eb = dtype_bytes(dtype)
    kv_b = ComputeStats(
        name="KV up-proj (kv_b, absorbed)",
        num_params=kv_b_params,
        flops=0,          # absorbed at inference
        weight_bytes=int(kv_b_params * eb),
        act_bytes=0,
        hbm_read_bytes=0,
        hbm_write_bytes=0,
    )
    stats.children.append(kv_b)

    # --- q_a_proj: h → c_q  (Q down-projection, optional) ----------------
    if c_q > 0:
        q_a = LinearStats(
            name="Q down-proj (q_a)",
            in_features=hidden_size,
            out_features=c_q,
            has_bias=has_bias,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=dtype,
        ).compute()
        stats.children.append(q_a)

        # --- q_b_proj: c_q → a * (d_n + d_r) ---
        q_b = LinearStats(
            name="Q up-proj (q_b)",
            in_features=c_q,
            out_features=a * q_head_dim,
            has_bias=has_bias,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=dtype,
        ).compute()
        stats.children.append(q_b)
    else:
        # No Q compression — direct projection h → a * q_head_dim
        q_direct = LinearStats(
            name="Q proj (direct)",
            in_features=hidden_size,
            out_features=a * q_head_dim,
            has_bias=has_bias,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=dtype,
        ).compute()
        stats.children.append(q_direct)

    # --- o_proj: a * v_dim → h ---
    o = LinearStats(
        name="O Projection",
        in_features=a * v_dim,
        out_features=hidden_size,
        has_bias=has_bias,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
    ).compute()
    stats.children.append(o)

    stats.aggregate_children()
    return stats


def mla_attention_stats(
    num_q_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    seq_len: int = 1,
    batch_size: int = 1,
    use_flash_attn: bool = False,
    dtype: DType = DType.BF16,
) -> ComputeStats:
    """Compute stats for the MLA attention kernel (absorb-optimised).

    In the absorb path, Q has been multiplied by W_kc, so Q and K both
    have dimension (c_kv + d_r) instead of (d_n + d_r).  Attention output
    is then projected through W_vc to get per-head output of dim v_dim.

    For FLOPs accounting we use the effective attention dimensions:
      QK^T dim = c_kv + d_r   (absorbed: Q in compressed space)
      AV dim   = c_kv         (attention over compressed KV, then W_vc applied)
    """
    B, s = batch_size, seq_len
    a = num_q_heads
    c_kv = kv_lora_rank
    d_r = qk_rope_head_dim
    v_dim = v_head_dim
    eb = dtype_bytes(dtype)

    # In absorb mode: Q has dim (c_kv + d_r), K has dim (c_kv + d_r)
    # V has dim c_kv (the compressed latent), then projected by W_vc
    absorbed_dim = c_kv + d_r

    # FLOPs
    flops_qkt = 2 * B * a * s * s * absorbed_dim
    flops_av = 2 * B * a * s * s * c_kv  # AV in compressed space
    flops_softmax = 5 * B * a * s * s

    total_flops = flops_qkt + flops_softmax + flops_av

    # HBM access
    # Q shape: [B, s, a, absorbed_dim]
    # K shape: [B, s, 1, absorbed_dim]  (shared across heads from latent)
    # V shape: [B, s, 1, c_kv]          (shared across heads)
    q_elems = B * s * a * absorbed_dim
    kv_elems = B * s * (absorbed_dim + c_kv)  # K + V (single-head compressed)
    o_elems = B * s * a * c_kv  # output before W_vc projection

    if use_flash_attn:
        hbm_read = int((q_elems + kv_elems) * eb)
        hbm_write = int(o_elems * eb)
        act_bytes = int(B * a * s * eb)  # online statistics only
        label = "MLA Flash Attention (absorbed)"
    else:
        attn_matrix = int(B * a * s * s * eb)
        hbm_read = int((q_elems + kv_elems) * eb) + attn_matrix
        hbm_write = attn_matrix + int(o_elems * eb)
        act_bytes = attn_matrix
        label = "MLA Standard Attention (absorbed)"

    return ComputeStats(
        name=label,
        num_params=0,
        flops=total_flops,
        weight_bytes=0,
        act_bytes=act_bytes,
        hbm_read_bytes=hbm_read,
        hbm_write_bytes=hbm_write,
    )
