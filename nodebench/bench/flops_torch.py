"""Dense GEMM throughput, pure PyTorch. No compilation, no CUTLASS build.

Accuracy note, stated up front so nobody over-claims from this number: what you
get here is achieved cuBLAS throughput on one square GEMM shape, not the silicon
peak. It typically lands within a few percent of a tuned CUTLASS profiler sweep
for the large shapes used here, and further below for small or awkward shapes.
Run ``--backend both`` if you need the two put side by side.

TF32 deserves its own row rather than being folded into FP32. On Ampere and
later, ``torch.matmul`` on fp32 tensors may or may not use the TF32 tensor-core
path depending on a global flag, and the two differ by roughly an order of
magnitude. A benchmark that does not say which one it measured is not reporting
anything.
"""

from __future__ import annotations

import gc
from typing import Any, Dict, List

from .base import BenchResult, robust_stats


def _time_gemm(torch, a, b, iters: int, warmup: int) -> List[float]:
    for _ in range(warmup):
        _ = a @ b
    torch.cuda.synchronize()
    samples: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        _ = a @ b
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0)  # seconds
    return samples


def _fp8_gemm(torch, n: int, iters: int, warmup: int) -> List[float]:
    """FP8 via torch._scaled_mm. Present from torch 2.2+ on sm_89 / sm_90 / sm_120."""
    e4m3 = getattr(torch, "float8_e4m3fn", None)
    scaled_mm = getattr(torch, "_scaled_mm", None)
    if e4m3 is None or scaled_mm is None:
        raise RuntimeError("this torch build has no float8_e4m3fn / _scaled_mm")
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16).to(e4m3)
    # _scaled_mm wants the second operand column-major.
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16).to(e4m3).t().contiguous().t()
    sa = torch.tensor(1.0, device="cuda")
    sb = torch.tensor(1.0, device="cuda")

    def once():
        return scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

    for _ in range(warmup):
        once()
    torch.cuda.synchronize()
    samples: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        once()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0)
    del a, b
    return samples


def run(cfg, device: int = 0) -> BenchResult:
    try:
        import torch
    except ImportError:
        return BenchResult("flops", "torch", ok=False, error="torch not installed")
    if not torch.cuda.is_available():
        return BenchResult("flops", "torch", ok=False, error="CUDA unavailable")

    n = int(cfg.get("bench.flops.size", 8192))
    iters = int(cfg.get("bench.flops.iters", 30))
    warmup = int(cfg.get("bench.flops.warmup", 10))
    want = list(cfg.get("bench.flops.dtypes", ["fp32", "tf32", "fp16", "bf16", "fp8"]))

    torch.cuda.set_device(device)
    flop = 2.0 * n * n * n  # multiply-add counted as two operations, the standard
    metrics: Dict[str, Any] = {}
    notes: List[str] = []

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32

    try:
        for name in want:
            try:
                if name == "fp8":
                    samples = _fp8_gemm(torch, n, iters, warmup)
                else:
                    if name == "fp32":
                        torch.backends.cuda.matmul.allow_tf32 = False
                        torch.backends.cudnn.allow_tf32 = False
                        dt = torch.float32
                    elif name == "tf32":
                        if torch.cuda.get_device_capability(device)[0] < 8:
                            notes.append("tf32 skipped: needs compute capability 8.0+")
                            continue
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        dt = torch.float32
                    elif name == "fp16":
                        dt = torch.float16
                    elif name == "bf16":
                        dt = torch.bfloat16
                    elif name == "fp64":
                        dt = torch.float64
                    else:
                        notes.append(f"unknown dtype '{name}', skipped")
                        continue
                    a = torch.randn(n, n, device="cuda", dtype=dt)
                    b = torch.randn(n, n, device="cuda", dtype=dt)
                    samples = _time_gemm(torch, a, b, iters, warmup)
                    del a, b

                tf = [flop / s / 1e12 for s in samples]
                st = robust_stats(tf)
                metrics[name] = {
                    "tflops_best": round(st["best"], 2),
                    "tflops_median": round(st["median"], 2),
                    "cv_pct": round(st["cv_pct"], 2),
                    "n": st["n"],
                }
            except Exception as e:  # one dtype failing must not kill the module
                metrics[name] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    return BenchResult(
        module="flops",
        backend="torch",
        metrics=metrics,
        meta={
            "shape": f"{n}x{n}x{n}",
            "flop_per_gemm": flop,
            "iters": iters,
            "warmup": warmup,
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
            "notes": notes,
            "caveat": (
                "Achieved cuBLAS throughput on one square shape. This is not the silicon "
                "peak and should not be quoted as one."
            ),
        },
    )
