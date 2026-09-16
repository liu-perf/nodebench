"""Shared plumbing for bench modules."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchResult:
    module: str
    backend: str  # "torch" | "native"
    ok: bool = True
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    skipped: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@contextmanager
def cuda_timer(device: int = 0):
    """Time a GPU region with CUDA events.

    Wall-clock timing around async CUDA work measures launch overhead, not the
    work. Events are recorded in the stream, so they measure the work. The
    result is exposed as ``t.ms`` after the block exits.
    """
    import torch

    class _T:
        ms = 0.0

    t = _T()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    try:
        yield t
    finally:
        end.record()
        torch.cuda.synchronize(device)
        t.ms = start.elapsed_time(end)


def robust_stats(samples: List[float]) -> Dict[str, float]:
    """Summary that does not let one scheduler hiccup define the result.

    ``best`` is what a peak-performance claim should quote; ``median`` is what a
    sustained-throughput claim should quote. Reporting both, plus the spread,
    makes it impossible to quietly cherry-pick.
    """
    if not samples:
        return {}
    s = sorted(samples)
    n = len(s)
    mid = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n
    return {
        "best": s[-1],
        "worst": s[0],
        "median": mid,
        "mean": mean,
        "stdev": var ** 0.5,
        "cv_pct": (var ** 0.5 / mean * 100) if mean else 0.0,
        "n": n,
    }


def dtype_map() -> Dict[str, Any]:
    import torch

    return {
        "fp32": torch.float32,
        "tf32": torch.float32,  # same storage, TF32 math path toggled separately
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp64": torch.float64,
    }
