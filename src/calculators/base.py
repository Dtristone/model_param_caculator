"""
base.py
=======
Core data types shared by all calculators.

Key design choices
------------------
* FLOPs are counted as multiply-adds, where one multiply-add = 2 operations
  (following the standard used by most hardware vendors and ML papers).
* Memory access is measured in *bytes* (HBM reads + writes).
* All formulas assume a generic forward pass (no autograd).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


# ---------------------------------------------------------------------------
# Data type helpers
# ---------------------------------------------------------------------------

class DType(str, Enum):
    """Supported element data types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


_DTYPE_BYTES = {
    DType.FP32: 4,
    DType.FP16: 2,
    DType.BF16: 2,
    DType.INT8: 1,
    DType.INT4: 0.5,
}


def dtype_bytes(dtype: DType | str) -> float:
    """Return the number of bytes for one element of the given dtype."""
    if isinstance(dtype, str):
        dtype = DType(dtype.lower())
    return _DTYPE_BYTES[dtype]


# ---------------------------------------------------------------------------
# ComputeStats
# ---------------------------------------------------------------------------

@dataclass
class ComputeStats:
    """Statistics for a single model component.

    Attributes
    ----------
    name : str
        Human-readable component name.
    num_params : int
        Number of learnable parameters (weights + biases).
    flops : int
        Total floating-point operations (1 multiply-add = 2 ops).
    weight_bytes : int
        Bytes required to store this component's parameters.
    act_bytes : int
        Peak activation memory (bytes) needed during the forward pass.
    hbm_read_bytes : int
        Bytes read from HBM (weights + inputs).
    hbm_write_bytes : int
        Bytes written to HBM (outputs).
    children : list[ComputeStats]
        Sub-components (populated by aggregating calculators).
    """

    name: str
    num_params: int = 0
    flops: int = 0
    weight_bytes: int = 0
    act_bytes: int = 0
    hbm_read_bytes: int = 0
    hbm_write_bytes: int = 0
    children: List["ComputeStats"] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def hbm_total_bytes(self) -> int:
        """Total HBM traffic (reads + writes)."""
        return self.hbm_read_bytes + self.hbm_write_bytes

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte of HBM traffic (roofline metric)."""
        if self.hbm_total_bytes == 0:
            return float("inf")
        return self.flops / self.hbm_total_bytes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def zero(cls, name: str) -> "ComputeStats":
        return cls(name=name)

    def __add__(self, other: "ComputeStats") -> "ComputeStats":
        """Merge two ComputeStats (used when summing layers)."""
        return ComputeStats(
            name=self.name,
            num_params=self.num_params + other.num_params,
            flops=self.flops + other.flops,
            weight_bytes=self.weight_bytes + other.weight_bytes,
            act_bytes=max(self.act_bytes, other.act_bytes),   # peak
            hbm_read_bytes=self.hbm_read_bytes + other.hbm_read_bytes,
            hbm_write_bytes=self.hbm_write_bytes + other.hbm_write_bytes,
        )

    def __iadd__(self, other: "ComputeStats") -> "ComputeStats":
        self.num_params += other.num_params
        self.flops += other.flops
        self.weight_bytes += other.weight_bytes
        self.act_bytes = max(self.act_bytes, other.act_bytes)
        self.hbm_read_bytes += other.hbm_read_bytes
        self.hbm_write_bytes += other.hbm_write_bytes
        return self

    def aggregate_children(self) -> None:
        """Sum children into self (used by aggregating calculators)."""
        for child in self.children:
            self.num_params += child.num_params
            self.flops += child.flops
            self.weight_bytes += child.weight_bytes
            self.hbm_read_bytes += child.hbm_read_bytes
            self.hbm_write_bytes += child.hbm_write_bytes
        # peak activation: max of individual peaks
        if self.children:
            self.act_bytes = max(c.act_bytes for c in self.children)


# ---------------------------------------------------------------------------
# Number formatting helpers
# ---------------------------------------------------------------------------

def fmt_num(n: int | float, precision: int = 2) -> str:
    """Format a large integer/float with SI suffix (K, M, B, T)."""
    for suffix, threshold in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= threshold:
            return f"{n / threshold:.{precision}f}{suffix}"
    return str(int(n))


def fmt_bytes(n: int | float, precision: int = 2) -> str:
    """Format byte count with appropriate unit."""
    for suffix, threshold in [("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)]:
        if abs(n) >= threshold:
            return f"{n / threshold:.{precision}f} {suffix}"
    return f"{int(n)} B"
