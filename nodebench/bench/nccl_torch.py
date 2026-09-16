"""NCCL collective bandwidth, PyTorch equivalent of nccl-tests. No build step.

Algorithm bandwidth vs bus bandwidth
------------------------------------
``algbw = message_bytes / seconds`` is what a user perceives. It is *not*
comparable across collectives or across GPU counts, because different
collectives move a different amount of data over the wire for the same message.

``busbw = algbw x correction(N)`` normalises that away, and is the number that
should be compared against link speed. The corrections below are the ones
nccl-tests uses, so a nodebench figure and an nccl-tests figure mean the same
thing:

    all_reduce      2(N-1)/N     each byte is reduced then broadcast
    all_gather        (N-1)/N
    reduce_scatter    (N-1)/N
    all_to_all        (N-1)/N
    broadcast/reduce      1

Reporting algbw and calling it bandwidth is the most common way multi-GPU
numbers get quietly inflated. Both are emitted here, labelled.

Why spawn instead of torchrun
-----------------------------
Requiring ``torchrun`` would mean a stranger has to learn a second launcher
before getting a number. ``mp.spawn`` on a single node needs nothing.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base import BenchResult, robust_stats

BUSBW_FACTOR = {
    "all_reduce": lambda n: 2.0 * (n - 1) / n,
    "all_gather": lambda n: (n - 1) / n,
    "reduce_scatter": lambda n: (n - 1) / n,
    "all_to_all": lambda n: (n - 1) / n,
    "broadcast": lambda n: 1.0,
    "reduce": lambda n: 1.0,
}


def _worker(rank: int, world: int, gpu_ids, sizes, colls, iters, warmup, port, out_q):
    import torch
    import torch.distributed as dist

    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        torch.cuda.set_device(gpu_ids[rank])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)

        dev = torch.device(f"cuda:{gpu_ids[rank]}")
        results: Dict[str, Dict[str, Any]] = {}
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for coll in colls:
            results[coll] = {}
            for nbytes in sizes:
                nelem = nbytes // 4
                # Bound up front so the cleanup below can always drop them. Sizes
                # ascend, so a buffer left alive from the previous iteration is
                # still resident while the next, larger one is allocated -- that
                # is an avoidable OOM at the top of the sweep.
                buf = src = dst = None
                try:
                    if coll == "all_reduce":
                        buf = torch.ones(nelem, dtype=torch.float32, device=dev)
                        op = lambda: dist.all_reduce(buf)  # noqa: E731
                    elif coll == "all_gather":
                        # message size = output size, matching nccl-tests convention
                        shard = nelem // world
                        src = torch.ones(shard, dtype=torch.float32, device=dev)
                        dst = torch.empty(shard * world, dtype=torch.float32, device=dev)
                        op = lambda: dist.all_gather_into_tensor(dst, src)  # noqa: E731
                    elif coll == "reduce_scatter":
                        shard = nelem // world
                        src = torch.ones(shard * world, dtype=torch.float32, device=dev)
                        dst = torch.empty(shard, dtype=torch.float32, device=dev)
                        op = lambda: dist.reduce_scatter_tensor(dst, src)  # noqa: E731
                    elif coll == "all_to_all":
                        shard = (nelem // world) * world
                        src = torch.ones(shard, dtype=torch.float32, device=dev)
                        dst = torch.empty(shard, dtype=torch.float32, device=dev)
                        op = lambda: dist.all_to_all_single(dst, src)  # noqa: E731
                    elif coll == "broadcast":
                        buf = torch.ones(nelem, dtype=torch.float32, device=dev)
                        op = lambda: dist.broadcast(buf, src=0)  # noqa: E731
                    else:
                        continue

                    for _ in range(warmup):
                        op()
                    dist.barrier()
                    torch.cuda.synchronize(dev)

                    samples: List[float] = []
                    for _ in range(iters):
                        dist.barrier()
                        start.record()
                        op()
                        end.record()
                        torch.cuda.synchronize(dev)
                        samples.append(start.elapsed_time(end) / 1000.0)

                    if rank == 0:
                        algbw = [nbytes / s / 1e9 for s in samples]
                        st = robust_stats(algbw)
                        f = BUSBW_FACTOR.get(coll, lambda n: 1.0)(world)
                        results[coll][str(nbytes)] = {
                            "algbw_gbs_best": round(st["best"], 2),
                            "algbw_gbs_median": round(st["median"], 2),
                            "busbw_gbs_best": round(st["best"] * f, 2),
                            "busbw_gbs_median": round(st["median"] * f, 2),
                            "busbw_factor": round(f, 4),
                            "cv_pct": round(st["cv_pct"], 2),
                        }
                except Exception as e:
                    if rank == 0:
                        results[coll][str(nbytes)] = {"error": f"{type(e).__name__}: {e}"}
                finally:
                    # empty_cache() cannot release memory that is still
                    # referenced, so the names have to be dropped first.
                    op = None
                    buf = src = dst = None
                    del op
                    torch.cuda.empty_cache()

        if rank == 0:
            out_q.put({"ok": True, "results": results})
        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        if rank == 0:
            out_q.put({"ok": False, "error": f"{type(e).__name__}: {e}"})


def run_one_set(cfg, gpu_ids: List[int], name: str = "") -> Dict[str, Any]:
    """Run the collective sweep on one specific set of GPU ids."""
    import torch.multiprocessing as mp

    world = len(gpu_ids)
    if world < 2:
        return {"name": name, "gpus": gpu_ids, "skipped": "collectives need at least 2 GPUs"}

    sizes = list(cfg.get("bench.nccl.sizes", [16_777_216]))
    colls = list(cfg.get("bench.nccl.collectives", ["all_reduce"]))
    iters = int(cfg.get("bench.nccl.iters", 20))
    warmup = int(cfg.get("bench.nccl.warmup", 5))

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + (hash(tuple(gpu_ids)) % 2000)

    procs = []
    for r in range(world):
        p = ctx.Process(
            target=_worker,
            args=(r, world, gpu_ids, sizes, colls, iters, warmup, port, q),
        )
        p.start()
        procs.append(p)

    payload: Optional[dict] = None
    try:
        payload = q.get(timeout=1800)
    except Exception:
        payload = {"ok": False, "error": "timed out waiting for rank 0"}
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()

    out = {"name": name or f"{world}gpu", "gpus": gpu_ids, "world_size": world}
    if payload and payload.get("ok"):
        out["collectives"] = payload["results"]
    else:
        out["error"] = (payload or {}).get("error", "unknown failure")
    return out


def run(cfg, gpu_sets: List[Dict[str, Any]]) -> BenchResult:
    try:
        import torch
    except ImportError:
        return BenchResult("nccl", "torch", ok=False, error="torch not installed")
    if not torch.cuda.is_available():
        return BenchResult("nccl", "torch", ok=False, error="CUDA unavailable")
    if torch.cuda.device_count() < 2:
        return BenchResult("nccl", "torch", skipped="single GPU -- no collectives to measure")

    runs = []
    for s in gpu_sets:
        if s["n"] < 2:
            continue
        r = run_one_set(cfg, s["gpus"], s["name"])
        r["worst_link"] = s.get("worst_link")
        r["note"] = s.get("note", "")
        runs.append(r)

    return BenchResult(
        module="nccl",
        backend="torch",
        metrics={"sets": runs},
        meta={
            "sizes": list(cfg.get("bench.nccl.sizes", [])),
            "collectives": list(cfg.get("bench.nccl.collectives", [])),
            "iters": int(cfg.get("bench.nccl.iters", 20)),
            "warmup": int(cfg.get("bench.nccl.warmup", 5)),
            "busbw_factors": {k: "2(N-1)/N" if k == "all_reduce" else "(N-1)/N" for k in BUSBW_FACTOR},
            "launcher": "torch.multiprocessing.spawn (single node)",
            "reading_guide": (
                "Compare busbw, never algbw, across different GPU counts or collectives. "
                "algbw is included only so the conversion stays auditable."
            ),
        },
    )


def peak_allreduce_busbw(result: BenchResult) -> Optional[float]:
    """Best all_reduce busbw across every set and size. Feeds the P2P verdict."""
    best = None
    for s in (result.metrics or {}).get("sets", []):
        for _, entry in (s.get("collectives", {}).get("all_reduce", {}) or {}).items():
            v = entry.get("busbw_gbs_median")
            if v is not None and (best is None or v > best):
                best = v
    return best
