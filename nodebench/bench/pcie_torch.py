"""Host-device transfer bandwidth, PyTorch equivalent of nvbandwidth's CE tests.

Two things decide whether this number is real:

1. **The host buffer must be pinned.** A pageable buffer forces the driver to
   stage through an internal pinned bounce buffer, and you end up measuring that
   staging rather than the link. It is the single most common reason a Gen5 x16
   node "measures" 6 GB/s.
2. **Timing must use CUDA events, not wall clock**, because with
   ``non_blocking=True`` the copy is asynchronous and a wall-clock timer around
   it measures the launch, not the transfer.

The all-GPU concurrent test at the end is the one worth reading. Per-card
bandwidth in isolation says how fast one link is. Running every card at once
says whether the root complex and the host memory subsystem can actually feed
them all -- and that is where nodes with under-populated memory channels fall
apart, quietly, in a way no single-card test will ever reveal.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BenchResult, robust_stats


def _time_copy(torch, dst, src, iters: int, warmup: int, dev: int) -> List[float]:
    for _ in range(warmup):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize(dev)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    out: List[float] = []
    for _ in range(iters):
        start.record()
        dst.copy_(src, non_blocking=True)
        end.record()
        torch.cuda.synchronize(dev)
        out.append(start.elapsed_time(end) / 1000.0)
    return out


def run(cfg, devices: List[int] = None) -> BenchResult:
    try:
        import torch
    except ImportError:
        return BenchResult("pcie", "torch", ok=False, error="torch not installed")
    if not torch.cuda.is_available():
        return BenchResult("pcie", "torch", ok=False, error="CUDA unavailable")

    nbytes = int(cfg.get("bench.pcie.bytes", 1_073_741_824))
    iters = int(cfg.get("bench.pcie.iters", 20))
    warmup = int(cfg.get("bench.pcie.warmup", 5))
    if devices is None:
        devices = list(range(torch.cuda.device_count()))

    nelem = nbytes // 4
    per_gpu: Dict[str, Any] = {}

    for d in devices:
        try:
            torch.cuda.set_device(d)
            host = torch.empty(nelem, dtype=torch.float32, pin_memory=True)
            dev = torch.empty(nelem, dtype=torch.float32, device=f"cuda:{d}")

            h2d = _time_copy(torch, dev, host, iters, warmup, d)
            d2h = _time_copy(torch, host, dev, iters, warmup, d)

            h2d_gbs = robust_stats([nbytes / s / 1e9 for s in h2d])
            d2h_gbs = robust_stats([nbytes / s / 1e9 for s in d2h])
            per_gpu[str(d)] = {
                "h2d_gbs_best": round(h2d_gbs["best"], 2),
                "h2d_gbs_median": round(h2d_gbs["median"], 2),
                "d2h_gbs_best": round(d2h_gbs["best"], 2),
                "d2h_gbs_median": round(d2h_gbs["median"], 2),
                "cv_pct": round(max(h2d_gbs["cv_pct"], d2h_gbs["cv_pct"]), 2),
            }
            del host, dev
            torch.cuda.empty_cache()
        except Exception as e:
            per_gpu[str(d)] = {"error": f"{type(e).__name__}: {e}"}

    # --- concurrent: all cards at once ------------------------------------
    # An empty dict here would read as "measured, nothing to report". On a
    # single-card machine the contention test is not a failure and not an
    # omission -- there is no contention to create -- and saying so is the
    # difference between "measured and fine" and "never looked".
    concurrent: Dict[str, Any] = {
        "skipped": f"needs at least 2 GPUs, this run had {len(devices)}",
        "n_gpus": len(devices),
    }
    if len(devices) > 1:
        try:
            hosts, devs, streams = [], [], []
            for d in devices:
                torch.cuda.set_device(d)
                hosts.append(torch.empty(nelem, dtype=torch.float32, pin_memory=True))
                devs.append(torch.empty(nelem, dtype=torch.float32, device=f"cuda:{d}"))
                streams.append(torch.cuda.Stream(device=d))

            def sweep() -> float:
                import time as _t

                for d in devices:
                    torch.cuda.synchronize(d)
                t0 = _t.perf_counter()
                for i in range(len(devices)):
                    with torch.cuda.stream(streams[i]):
                        devs[i].copy_(hosts[i], non_blocking=True)
                for d in devices:
                    torch.cuda.synchronize(d)
                return _t.perf_counter() - t0

            for _ in range(max(2, warmup // 2)):
                sweep()
            times = [sweep() for _ in range(max(3, iters // 2))]
            total = nbytes * len(devices)
            agg = robust_stats([total / t / 1e9 for t in times])
            single_sum = sum(
                v.get("h2d_gbs_median", 0) for v in per_gpu.values() if "error" not in v
            )
            concurrent = {
                "h2d_aggregate_gbs_best": round(agg["best"], 1),
                "h2d_aggregate_gbs_median": round(agg["median"], 1),
                "sum_of_isolated_gbs": round(single_sum, 1),
                "retention_pct": round(agg["median"] / single_sum * 100, 1) if single_sum else None,
                "n_gpus": len(devices),
            }
            del hosts, devs
            torch.cuda.empty_cache()
        except Exception as e:
            concurrent = {"error": f"{type(e).__name__}: {e}"}

    return BenchResult(
        module="pcie",
        backend="torch",
        metrics={"per_gpu": per_gpu, "concurrent": concurrent},
        meta={
            "transfer_bytes": nbytes,
            "iters": iters,
            "warmup": warmup,
            "pinned_host_memory": True,
            "timing": "cudaEvent",
            # Pointing at retention_pct on a single-card run would send the
            # reader looking for a field that is not there, and a guide that
            # does not match the data is worse than no guide.
            "reading_guide": (
                "retention_pct is the number to read. Near 100% means the root complex and "
                "host memory keep up with every card at once. Well below it means the node "
                "cannot feed all its GPUs simultaneously, which no single-card test shows."
                if len(devices) > 1 else
                "Single card: only the isolated per-GPU figures exist. The contention "
                "question -- can the node feed every card at once -- is unanswerable here "
                "and is not being answered."
            ),
        },
    )
