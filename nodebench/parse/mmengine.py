"""Parse mmengine training logs (mmdetection / mmsegmentation / mmpretrain).

This is the piece that turns "I ran a training job" into "I measured throughput
and scaling efficiency". Everything downstream -- speedup, efficiency, per-iter
inflation -- is derived from three numbers per log line:

    time        seconds per iteration (the whole step)
    data_time   seconds spent in the DataLoader inside that step
    memory      MB of GPU memory reported by mmengine

Steady state is the only part that counts
-----------------------------------------
The first iterations of any training job include CUDA context creation, cuDNN
autotuning, memory-pool growth and the DataLoader spinning up its workers. They
are several times slower than steady state and they are not what anyone means by
throughput. This module drops a warmup prefix, then reports the **median** of
what remains rather than the mean, so one page-cache miss cannot move the
result.

data_time is reported, not hidden
---------------------------------
If ``data_time / time`` is large, the GPUs spent the step waiting for JPEGs to
decode and you have measured your storage and CPU, not your accelerators. Any
scaling conclusion drawn from an input-bound run is meaningless, so the ratio is
surfaced as a first-class field with an explicit warning threshold.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# 2026/07/22 02:52:10 - mmengine - INFO - Epoch(train) [1][ 50/14786]  lr: 1.0e-04
#   eta: 5:12:33  time: 0.1403  data_time: 0.0089  memory: 9876  loss: 3.21
ITER_RE = re.compile(
    r"Epoch\(train\)\s*\[\s*(?P<epoch>\d+)\]\s*\[\s*(?P<it>\d+)\s*/\s*(?P<total>\d+)\s*\]"
)
# Iter-based schedules (mmseg): Iter(train) [ 50/80000]
ITER_RE2 = re.compile(r"Iter\(train\)\s*\[\s*(?P<it>\d+)\s*/\s*(?P<total>\d+)\s*\]")

FIELD_RE = {
    "time": re.compile(r"\btime:\s*([\d.]+)"),
    "data_time": re.compile(r"\bdata_time:\s*([\d.]+)"),
    "memory": re.compile(r"\bmemory:\s*([\d.]+)"),
    "loss": re.compile(r"\bloss:\s*([\d.eE+-]+)"),
    "lr": re.compile(r"\blr:\s*([\d.eE+-]+)"),
}

WORLD_RE = re.compile(r"(?:World size|world_size)\s*[:=]\s*(\d+)")
DIST_RE = re.compile(r"Distributed training:\s*(True|False)")


def parse_mmengine_log(text: str) -> Dict[str, Any]:
    """Pure function. Extract every training iteration record from a log."""
    iters: List[Dict[str, Any]] = []
    world_size: Optional[int] = None
    distributed: Optional[bool] = None
    errors: List[str] = []

    for ln in text.splitlines():
        if world_size is None:
            m = WORLD_RE.search(ln)
            if m:
                world_size = int(m.group(1))
        if distributed is None:
            m = DIST_RE.search(ln)
            if m:
                distributed = m.group(1) == "True"
        low = ln.lower()
        if any(k in low for k in ("out of memory", "cuda error", "nan or inf", "traceback")):
            errors.append(ln.strip()[:300])

        m = ITER_RE.search(ln)
        epoch = None
        if m:
            epoch = int(m.group("epoch"))
            it = int(m.group("it"))
            total = int(m.group("total"))
        else:
            m2 = ITER_RE2.search(ln)
            if not m2:
                continue
            it = int(m2.group("it"))
            total = int(m2.group("total"))

        rec: Dict[str, Any] = {"epoch": epoch, "iter": it, "iters_per_epoch": total}
        got_time = False
        for name, rx in FIELD_RE.items():
            mm = rx.search(ln)
            if mm:
                try:
                    rec[name] = float(mm.group(1))
                    if name == "time":
                        got_time = True
                except ValueError:
                    pass
        if got_time:
            iters.append(rec)

    return {
        "iters": iters,
        "n_iters": len(iters),
        "world_size": world_size,
        "distributed": distributed,
        "errors": errors[:20],
        "n_errors": len(errors),
    }


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def summarize_training(
    parsed: Dict[str, Any],
    gpus: int,
    batch_per_gpu: int,
    dataset_size: Optional[int] = None,
    warmup_iters: int = 50,
    warmup_frac: float = 0.3,
) -> Dict[str, Any]:
    """Turn raw iteration records into throughput facts.

    ``warmup_iters`` and ``warmup_frac`` both apply; the larger wins, so short
    runs still discard a sensible fraction and long runs do not throw away half
    the data.
    """
    recs = parsed.get("iters") or []
    if not recs:
        return {"ok": False, "error": "no training iterations found in log"}

    drop = max(min(warmup_iters, len(recs) - 1), int(len(recs) * warmup_frac))
    steady = recs[drop:] or recs[-1:]

    times = [r["time"] for r in steady if "time" in r]
    dts = [r["data_time"] for r in steady if "data_time" in r]
    mems = [r["memory"] for r in steady if "memory" in r]

    t_med = _median(times)
    t_mean = sum(times) / len(times)
    global_batch = batch_per_gpu * gpus
    samples_s = global_batch / t_med if t_med else 0.0

    dt_med = _median(dts) if dts else None
    dt_ratio = (dt_med / t_med) if (dt_med is not None and t_med) else None

    out: Dict[str, Any] = {
        "ok": True,
        "gpus": gpus,
        "batch_per_gpu": batch_per_gpu,
        "global_batch": global_batch,
        "iters_total": len(recs),
        "iters_warmup_dropped": drop,
        "iters_steady": len(steady),
        "iter_time_s_median": round(t_med, 4),
        "iter_time_s_mean": round(t_mean, 4),
        "iter_time_cv_pct": round(
            (sum((x - t_mean) ** 2 for x in times) / len(times)) ** 0.5 / t_mean * 100, 2
        ) if t_mean else None,
        "samples_per_s": round(samples_s, 2),
        "data_time_s_median": round(dt_med, 4) if dt_med is not None else None,
        "data_time_ratio": round(dt_ratio, 4) if dt_ratio is not None else None,
        "memory_mib_peak": round(max(mems), 0) if mems else None,
        "errors": parsed.get("errors", []),
    }

    if dataset_size and samples_s:
        out["dataset_size"] = dataset_size
        out["iters_per_epoch"] = round(dataset_size / global_batch, 1)
        out["epoch_time_s_extrapolated"] = round(dataset_size / samples_s, 1)
        out["epoch_time_caveat"] = (
            "Extrapolated from steady-state step time. It is a LOWER BOUND: it excludes "
            "per-epoch validation, checkpointing and dataloader re-initialisation."
        )

    warnings: List[str] = []
    if dt_ratio is not None and dt_ratio > 0.15:
        warnings.append(
            f"data_time is {dt_ratio:.0%} of step time. This run is input-bound -- you are "
            "measuring storage and CPU decode, not the GPUs. Scaling numbers derived from "
            "it are not valid. Raise num_workers or cache the decoded data."
        )
    if out["iter_time_cv_pct"] and out["iter_time_cv_pct"] > 10:
        warnings.append(
            f"Step time varies by {out['iter_time_cv_pct']:.0f}% (CV). Something else is "
            "using the node, or the job is thermally throttling. Check the monitor CSV."
        )
    if parsed.get("n_errors"):
        warnings.append(f"{parsed['n_errors']} error line(s) found in the log -- read them.")
    out["warnings"] = warnings
    return out
