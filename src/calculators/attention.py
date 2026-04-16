"""
attention.py
============
FLOPs and memory-access statistics for the attention mechanism.

Two modes are supported:

  Standard attention  — full attention matrix is materialised in HBM.
  Flash attention     — tiled computation; attention matrix never written to HBM.

Mathematical overview
---------------------
Given Q ∈ ℝ^(B×a×s×d), K ∈ ℝ^(B×k×s×d), V ∈ ℝ^(B×k×s×d)

Step 1: S = Q K^T / sqrt(d)
  FLOPs  = 2 * B * a * s * s * d = 2 * B * s² * h   (h = a * d)

Step 2: P = softmax(S)   [per row]
  FLOPs  ≈ 5 * B * a * s²  (max, subtract, exp, sum, divide)

Step 3: O = P V
  FLOPs  = 2 * B * a * s * s * d = 2 * B * s² * h

Total matmul FLOPs ≈ 4 * B * s² * h  (softmax is usually negligible)

HBM access — standard attention
  Read  : Q (B·a·s·d) + K (B·k·s·d) + V (B·k·s·d)
  Write : P (B·a·s·s) — materialised attention matrix
  Read  : P (B·a·s·s)
  Write : O (B·a·s·d)
  Total ≈ 4*B*s*h + 2*B*a*s²

HBM access — flash attention (FA2)
  Read  : Q, K, V  (each accessed once from HBM in tiled fashion)
  Write : O
  Total = 4 * B * s * h  (no full s×s matrix in HBM)

Activation memory
  Standard  : B * a * s² * elem_size  (attention matrix P)
  Flash      : B * a * s * elem_size  (only row statistics m, l)
"""

from __future__ import annotations

from .base import ComputeStats, DType, dtype_bytes


def attention_stats(
    hidden_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int = 1,
    batch_size: int = 1,
    use_flash_attn: bool = False,
    dtype: DType = DType.BF16,
) -> ComputeStats:
    """Compute stats for the attention kernel (excluding projections).

    Parameters
    ----------
    hidden_size : int
        h = num_q_heads * head_dim
    num_q_heads : int
        Number of Q heads (a).
    num_kv_heads : int
        Number of KV heads (k).
    head_dim : int
        Dimension per attention head (d).
    seq_len : int
        Sequence length (s).
    batch_size : int
        Batch size (B).
    use_flash_attn : bool
        If True, compute HBM access for Flash Attention.
    """
    B, s, h = batch_size, seq_len, hidden_size
    a, k, d = num_q_heads, num_kv_heads, head_dim
    eb = dtype_bytes(dtype)

    # FLOPs —————————————————————————————————————————————————————————————
    # QK^T (note: with GQA each K head is shared by a/k Q heads,
    # but all a Q heads still compute s² attention → same total FLOPs)
    flops_qkt = 2 * B * a * s * s * d       # = 2 * B * s² * h
    flops_av  = 2 * B * a * s * s * d       # = 2 * B * s² * h
    flops_softmax = 5 * B * a * s * s       # minor term

    total_flops = flops_qkt + flops_softmax + flops_av

    # HBM access ————————————————————————————————————————————————————————
    kv_dim = k * d  # = num_kv_heads * head_dim

    # Read Q, K, V
    hbm_read_qkv = int((B * s * h + B * s * kv_dim + B * s * kv_dim) * eb)
    # Write O
    hbm_write_o = int(B * s * h * eb)

    if use_flash_attn:
        # Flash Attention: attention matrix never written to HBM
        hbm_read = hbm_read_qkv
        hbm_write = hbm_write_o
        # Activation: only per-row statistics (m, l) for each head
        act_bytes = int(B * a * s * eb)
        label = "Flash Attention"
    else:
        # Standard Attention: write then read back attention matrix
        attn_matrix = int(B * a * s * s * eb)
        hbm_read  = hbm_read_qkv + attn_matrix          # also read P for AV
        hbm_write = attn_matrix + hbm_write_o            # write P + write O
        # Activation: full attention matrix B×a×s×s
        act_bytes = attn_matrix
        label = "Standard Attention"

    return ComputeStats(
        name=label,
        num_params=0,          # attention kernel has no learnable weights
        flops=total_flops,
        weight_bytes=0,
        act_bytes=act_bytes,
        hbm_read_bytes=hbm_read,
        hbm_write_bytes=hbm_write,
    )
