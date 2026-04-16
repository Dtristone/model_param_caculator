"""
dsa.py
======
FLOPs, parameters, and memory-access statistics for Differential Sparse
Attention (DSA) as used in GLM-5.

DSA adds a lightweight "indexer" branch that selects the top-k most
relevant tokens before running the main MLA attention.  This dramatically
reduces attention FLOPs for long sequences (from O(s²) to O(s·k)).

Architecture (per layer)
------------------------
  hidden_states (h)
       │
       ├── Main MLA branch (operates on selected top-k tokens)
       │     Q absorbed: [B, s, a, c_kv+d_r]
       │     K compressed: [B, k, 1, c_kv+d_r]   ← only top-k tokens
       │     V compressed: [B, k, 1, c_kv]        ← only top-k tokens
       │     → SparseMLA kernel
       │     → W_vc projection
       │     → O projection
       │
       └── Indexer branch (lightweight, runs on ALL tokens)
             wq_b:  q_compressed (c_q) → index_heads × index_head_dim
             wk:    hidden (h) → index_head_dim                (shared key)
             k_norm: LayerNorm(index_head_dim)
             weights_proj: h → index_heads                     (per-head gating)
             → compute index scores: Q_idx @ K_idx^T
             → select top-k token indices

Notation
--------
  k_sel   = index_topk            (number of tokens selected, e.g. 2048)
  idx_h   = index_head_dim
  idx_a   = index_n_heads
  c_q     = q_lora_rank

FLOPs breakdown
---------------
Indexer projections:
  wq_b:         2 * B * s * c_q * (idx_a * idx_h)
  wk:           2 * B * s * h * idx_h
  weights_proj: 2 * B * s * h * idx_a
  index QK^T:   2 * B * idx_a * s * s * idx_h   (full attention for indexing)
  top-k select: negligible

Main attention (sparse — only k_sel tokens attend):
  QK^T:  2 * B * a * s * k_sel * (c_kv + d_r)
  AV:    2 * B * a * s * k_sel * c_kv
  softmax: 5 * B * a * s * k_sel

The DSA indexer's K cache per token = index_head_dim elements.
"""

from __future__ import annotations

from .base import ComputeStats, DType, dtype_bytes
from .linear import LinearStats


def dsa_indexer_stats(
    hidden_size: int,
    q_lora_rank: int,
    index_n_heads: int,
    index_head_dim: int,
    seq_len: int = 1,
    batch_size: int = 1,
    dtype: DType = DType.BF16,
) -> ComputeStats:
    """Compute stats for the DSA indexer branch.

    The indexer runs on all tokens to determine which top-k tokens each
    query should attend to in the main branch.
    """
    B, s = batch_size, seq_len
    idx_a = index_n_heads
    idx_h = index_head_dim
    c_q = q_lora_rank
    eb = dtype_bytes(dtype)

    stats = ComputeStats(name="DSA Indexer")

    # --- wq_b: c_q → idx_a * idx_h ---
    wq_b = LinearStats(
        name="Indexer Q proj (wq_b)",
        in_features=c_q,
        out_features=idx_a * idx_h,
        has_bias=False,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
    ).compute()
    stats.children.append(wq_b)

    # --- wk: h → idx_h (shared single-head key) ---
    wk = LinearStats(
        name="Indexer K proj (wk)",
        in_features=hidden_size,
        out_features=idx_h,
        has_bias=False,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
    ).compute()
    stats.children.append(wk)

    # --- weights_proj: h → idx_a (per-head gating weights) ---
    wp = LinearStats(
        name="Indexer head weights",
        in_features=hidden_size,
        out_features=idx_a,
        has_bias=False,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
    ).compute()
    stats.children.append(wp)

    # --- k_norm: LayerNorm on idx_h (negligible params, small FLOPs) ---
    k_norm_params = 2 * idx_h  # LayerNorm scale + bias
    k_norm_flops = 7 * B * s * idx_h
    k_norm = ComputeStats(
        name="Indexer K norm",
        num_params=k_norm_params,
        flops=k_norm_flops,
        weight_bytes=int(k_norm_params * eb),
        hbm_read_bytes=int(B * s * idx_h * eb + k_norm_params * eb),
        hbm_write_bytes=int(B * s * idx_h * eb),
    )
    stats.children.append(k_norm)

    # --- Index attention: Q_idx @ K_idx^T to score all tokens ---
    # Q_idx: [B, s, idx_a, idx_h], K_idx: [B, s, 1, idx_h]
    # Attention over full s×s to find top-k
    index_qkt_flops = 2 * B * idx_a * s * s * idx_h
    index_attn = ComputeStats(
        name="Index QK^T scoring",
        num_params=0,
        flops=index_qkt_flops,
        hbm_read_bytes=int((B * s * idx_a * idx_h + B * s * idx_h) * eb),
        hbm_write_bytes=int(B * idx_a * s * s * eb),  # score matrix
    )
    stats.children.append(index_attn)

    stats.aggregate_children()
    return stats


def dsa_sparse_attention_stats(
    num_q_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    index_topk: int,
    seq_len: int = 1,
    batch_size: int = 1,
    dtype: DType = DType.BF16,
) -> ComputeStats:
    """Compute stats for the DSA sparse main-branch attention.

    Unlike standard attention (O(s²)), DSA attention only computes over
    the selected top-k tokens (O(s·k_sel)), where k_sel << s.
    """
    B, s = batch_size, seq_len
    a = num_q_heads
    c_kv = kv_lora_rank
    d_r = qk_rope_head_dim
    eb = dtype_bytes(dtype)

    # Effective number of KV tokens per query
    k_sel = min(index_topk, seq_len)

    absorbed_dim = c_kv + d_r

    # FLOPs — sparse attention: each query attends to k_sel keys instead of s
    flops_qkt = 2 * B * a * s * k_sel * absorbed_dim
    flops_av = 2 * B * a * s * k_sel * c_kv
    flops_softmax = 5 * B * a * s * k_sel

    total_flops = flops_qkt + flops_softmax + flops_av

    # HBM access
    q_elems = B * s * a * absorbed_dim
    # K,V are gathered for only k_sel tokens per query, but in practice
    # read once from HBM and used for all queries in a block
    kv_elems = B * s * (absorbed_dim + c_kv)
    o_elems = B * s * a * c_kv

    # Indices: B * s * k_sel (int32 = 4 bytes each)
    index_bytes = B * s * k_sel * 4

    hbm_read = int((q_elems + kv_elems) * eb) + index_bytes
    hbm_write = int(o_elems * eb)
    act_bytes = int(B * a * s * k_sel * eb)  # sparse attention scores

    return ComputeStats(
        name=f"DSA Sparse MLA (top-{k_sel})",
        num_params=0,
        flops=total_flops,
        weight_bytes=0,
        act_bytes=act_bytes,
        hbm_read_bytes=hbm_read,
        hbm_write_bytes=hbm_write,
    )
