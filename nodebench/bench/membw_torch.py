"""On-device memory bandwidth, BabelStream kernels reimplemented in PyTorch.

Same four kernels BabelStream uses, so the numbers are comparable to the
published ones, but with no cmake and no compiler:

    copy   c = a          2 arrays touched
    mul    b = k * c      2
    add    c = a + b      3
    triad  a = b + k * c  3

Bandwidth = (arrays touched) x (elements) x (bytes per element) / seconds.

This doubles as a sanity check on the whole measurement chain. If ``triad``
comes back at 95% of the datasheet figure, your timing, your device selection
and your clock state are all fine, and any *other* number that looks wrong is
genuinely wrong rather than an artefact.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BenchResult, robust_stats


def run(cfg, device: int = 0) -> BenchResult:
    try:
        import torch
    except ImportError:
        return BenchResult("membw", "torch", ok=False, error="torch not installed")
    if not torch.cuda.is_available():
        return BenchResult("membw", "torch", ok=False, error="CUDA unavailable")

    nelem = int(cfg.get("bench.membw.elements", 268_435_456))
    iters = int(cfg.get("bench.membw.iters", 30))
    warmup = int(cfg.get("bench.membw.warmup", 10))

    torch.cuda.set_device(device)
    itemsize = 4  # float32

    free, total = torch.cuda.mem_get_info(device)
    need = 3 * nelem * itemsize
    if need > free * 0.8:
        nelem = int(free * 0.8 / (3 * itemsize))
        nelem -= nelem % 1024
    bytes_per = nelem * itemsize

    a = torch.ones(nelem, device="cuda", dtype=torch.float32)
    b = torch.full((nelem,), 2.0, device="cuda", dtype=torch.float32)
    c = torch.zeros(nelem, device="cuda", dtype=torch.float32)
    k = 3.0

    kernels = {
        "copy":  (lambda: c.copy_(a),                    2),
        "mul":   (lambda: torch.mul(c, k, out=b),        2),
        "add":   (lambda: torch.add(a, b, out=c),        3),
        "triad": (lambda: torch.add(b, c, alpha=k, out=a), 3),
    }

    metrics: Dict[str, Any] = {}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for name, (fn, arrays) in kernels.items():
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            samples: List[float] = []
            for _ in range(iters):
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end) / 1000.0)
            moved = arrays * bytes_per
            gbs = [moved / s / 1e9 for s in samples]
            st = robust_stats(gbs)
            metrics[name] = {
                "gbs_best": round(st["best"], 1),
                "gbs_median": round(st["median"], 1),
                "cv_pct": round(st["cv_pct"], 2),
                "arrays_touched": arrays,
            }
        except Exception as e:
            metrics[name] = {"error": f"{type(e).__name__}: {e}"}

    del a, b, c
    torch.cuda.empty_cache()

    return BenchResult(
        module="membw",
        backend="torch",
        metrics=metrics,
        meta={
            "elements": nelem,
            "bytes_per_array": bytes_per,
            "dtype": "float32",
            "iters": iters,
            "warmup": warmup,
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
            "caveat": (
                "GB/s here is decimal (1e9), matching BabelStream and vendor datasheets. "
                "Do not compare against a GiB/s figure without converting."
            ),
        },
    )
