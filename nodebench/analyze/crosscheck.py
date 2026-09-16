"""Cross-check the PyTorch backend against the native C++ backend.

This is the feature that separates this tool from a script that prints numbers.

Any single measurement pipeline can be silently wrong: a unit slip, a missing
synchronise, a warmup left in, a buffer that fits in L2. The defence is not to
be more careful -- it is to measure the same physical quantity two independent
ways and require them to agree.

  torch.matmul          vs  cuBLAS / CUTLASS profiler       -> FLOPS
  tensor copy kernels   vs  BabelStream                     -> memory bandwidth
  pinned cudaMemcpy     vs  nvbandwidth host_to_device       -> PCIe
  torch.distributed     vs  nccl-tests all_reduce_perf       -> collectives

Agreement does not prove either is right, but disagreement proves one is wrong,
and that is worth far more than a third decimal place. Divergence is reported
whether or not it is flattering; a run where the two backends disagree by 30%
is a finding, not a failure.

Thresholds
----------
5% is the agreement bar. It comes from the observation that two honest
implementations of the same measurement on the same hardware land within a few
percent; beyond that, something structural differs (different shape, different
dtype, different buffer size, one of them measuring warmup).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

AGREE_PCT = 5.0
CONCERN_PCT = 15.0

# What each pair actually compares, stated so a reader can judge whether the
# comparison is fair rather than trusting the label.
PAIRS = {
    "flops": (
        "torch.matmul on a square GEMM",
        "cuBLAS/CUTLASS profiler best kernel",
        "Both are achieved throughput on one shape. CUTLASS may win by picking a "
        "better tile for that shape; a gap in its favour is normal, the reverse is not.",
    ),
    "membw": (
        "PyTorch tensor kernels (copy/mul/add/triad)",
        "BabelStream",
        "Same four kernels by construction. These should agree closely; if they do "
        "not, one of them is not saturating the array size.",
    ),
    "pcie_h2d": (
        "pinned host buffer + cudaMemcpyAsync via torch",
        "nvbandwidth host_to_device_memcpy_ce",
        "nvbandwidth uses the copy engine explicitly. A large gap usually means the "
        "torch-side buffer was pageable, not pinned.",
    ),
    "allreduce_busbw": (
        "torch.distributed all_reduce, CUDA-event timed",
        "nccl-tests all_reduce_perf",
        "Same NCCL underneath. Divergence here points at the harness -- different "
        "message size, missing barrier, or timing that includes launch overhead.",
    ),
}


def _one(name: str, torch_val: Optional[float], native_val: Optional[float]) -> Dict[str, Any]:
    a, b, note = PAIRS.get(name, ("torch", "native", ""))
    row: Dict[str, Any] = {
        "quantity": name,
        "torch_method": a,
        "native_method": b,
        "torch_value": torch_val,
        "native_value": native_val,
        "note": note,
    }
    if torch_val is None or native_val is None:
        row["status"] = "unavailable"
        row["reason"] = "only one backend produced this quantity"
        return row
    if native_val == 0:
        row["status"] = "unavailable"
        row["reason"] = "native value is zero"
        return row

    div = abs(torch_val - native_val) / max(abs(torch_val), abs(native_val)) * 100
    row["divergence_pct"] = round(div, 2)
    row["ratio_torch_over_native"] = round(torch_val / native_val, 4)
    if div <= AGREE_PCT:
        row["status"] = "agree"
    elif div <= CONCERN_PCT:
        row["status"] = "differ"
        row["reason"] = (
            f"{div:.1f}% apart. Tolerable for shape-sensitive quantities, but state it "
            "rather than quoting whichever number you prefer."
        )
    else:
        row["status"] = "conflict"
        row["reason"] = (
            f"{div:.1f}% apart. At least one of these is measuring something other than "
            "what its label says. Do not publish either until you know which."
        )
    return row


def cross_check(torch_results: Dict[str, Any], native_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare the two backends' readings of the same physical quantities.

    Both inputs are flat dicts of ``quantity -> value``. Extraction from raw
    bench results lives in the caller, so this stays pure and testable.
    """
    rows: List[Dict[str, Any]] = []
    for name in PAIRS:
        if name in torch_results or name in native_results:
            rows.append(_one(name, torch_results.get(name), native_results.get(name)))

    counts = {"agree": 0, "differ": 0, "conflict": 0, "unavailable": 0}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    if counts["conflict"]:
        headline = (
            f"{counts['conflict']} quantity(ies) disagree by more than {CONCERN_PCT:.0f}% "
            "between backends. Treat those rows as unresolved."
        )
    elif counts["differ"]:
        headline = (
            f"All quantities are within {CONCERN_PCT:.0f}%, "
            f"{counts['differ']} of them outside the {AGREE_PCT:.0f}% agreement bar."
        )
    elif counts["agree"]:
        headline = (
            f"All {counts['agree']} cross-checked quantities agree within {AGREE_PCT:.0f}% "
            "across two independent implementations."
        )
    else:
        headline = "No quantity was produced by both backends -- nothing to cross-check."

    return {
        "rows": rows,
        "counts": counts,
        "headline": headline,
        "agreement_threshold_pct": AGREE_PCT,
        "concern_threshold_pct": CONCERN_PCT,
    }
