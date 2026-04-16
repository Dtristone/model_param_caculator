"""
linear.py
=========
FLOPs and memory-access statistics for linear (fully-connected) layers,
including the QKV projections and the output projection of the attention block.

All formulas use: 1 multiply-add = 2 FLOPs.

HBM access model
----------------
For Y = X @ W^T  (X: [B·s, in], W: [out, in])

  HBM reads  = X (B·s·in) + W (in·out) elements
  HBM writes = Y (B·s·out) elements
  Total elements = B·s·in + in·out + B·s·out

When batch×seq_len is small compared to in×out (e.g. decode with s=1),
the weight transfer dominates — this is the *memory-bandwidth-bound* regime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .base import ComputeStats, DType, dtype_bytes


# ---------------------------------------------------------------------------
# Single linear layer
# ---------------------------------------------------------------------------

@dataclass
class LinearStats:
    """Statistics for a single linear layer Y = X W^T (+b)."""

    name: str
    in_features: int
    out_features: int
    has_bias: bool = False
    seq_len: int = 1
    batch_size: int = 1
    dtype: DType = DType.BF16

    def compute(self) -> ComputeStats:
        B, s = self.batch_size, self.seq_len
        in_f, out_f = self.in_features, self.out_features
        eb = dtype_bytes(self.dtype)

        # Parameters
        n_params = in_f * out_f + (out_f if self.has_bias else 0)

        # FLOPs: 2 * B * s * in * out  (bias add is negligible)
        flops = 2 * B * s * in_f * out_f

        # Weight bytes (stored in model)
        w_bytes = int(n_params * eb)

        # HBM traffic
        tokens = B * s
        param_elems = n_params  # weights plus bias when present
        hbm_read = int((tokens * in_f + param_elems) * eb)
        hbm_write = int(tokens * out_f * eb)

        # Activation memory (input + output tensors)
        act = int((tokens * in_f + tokens * out_f) * eb)

        return ComputeStats(
            name=self.name,
            num_params=n_params,
            flops=flops,
            weight_bytes=w_bytes,
            act_bytes=act,
            hbm_read_bytes=hbm_read,
            hbm_write_bytes=hbm_write,
        )


# ---------------------------------------------------------------------------
# QKV projections
# ---------------------------------------------------------------------------

def qkv_proj_stats(
    hidden_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int = 1,
    batch_size: int = 1,
    has_bias: bool = False,
    dtype: DType = DType.BF16,
) -> ComputeStats:
    """Return combined ComputeStats for Q, K, and V projection matrices.

    Parameters
    ----------
    hidden_size : int
        Input hidden dimension h.
    num_q_heads : int
        Number of Q heads (a).
    num_kv_heads : int
        Number of KV heads (k).  k == a → MHA, k < a → GQA, k == 1 → MQA.
    head_dim : int
        Dimension per head (d).
    """
    q_out = num_q_heads * head_dim   # = hidden_size for MHA
    kv_out = num_kv_heads * head_dim

    stats = ComputeStats(name="QKV Projection")

    for proj_name, out_dim in [("Q proj", q_out), ("K proj", kv_out), ("V proj", kv_out)]:
        s = LinearStats(
            name=proj_name,
            in_features=hidden_size,
            out_features=out_dim,
            has_bias=has_bias,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=dtype,
        ).compute()
        stats.children.append(s)

    stats.aggregate_children()
    return stats


# ---------------------------------------------------------------------------
# Output projection
# ---------------------------------------------------------------------------

def output_proj_stats(
    hidden_size: int,
    seq_len: int = 1,
    batch_size: int = 1,
    has_bias: bool = False,
    dtype: DType = DType.BF16,
) -> ComputeStats:
    """Return ComputeStats for the attention output projection (O = AV W_o)."""
    return LinearStats(
        name="O Projection",
        in_features=hidden_size,
        out_features=hidden_size,
        has_bias=has_bias,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
    ).compute()
