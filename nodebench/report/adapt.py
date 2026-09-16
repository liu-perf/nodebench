"""Turn raw module results into the flat shape the HTML renderer consumes.

This layer exists so that neither side has to know about the other. A bench
module returns whatever is natural for the measurement it made; the renderer
consumes a stable, boring shape. When a module's output changes, only this file
moves.

It is also where "best vs median" is decided once. The bench modules report
both; the report shows the median as the headline (it is the honest number) and
keeps best in the table (it is the one everyone else quotes, so hiding it
invites the accusation of cherry-picking downward).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _m(res) -> Dict[str, Any]:
    """Metrics from a BenchResult, or {} if it did not run."""
    if res is None:
        return {}
    if isinstance(res, dict):
        return res.get("metrics") or {}
    return getattr(res, "metrics", None) or {}


def _meta(res) -> Dict[str, Any]:
    if res is None:
        return {}
    if isinstance(res, dict):
        return res.get("meta") or {}
    return getattr(res, "meta", None) or {}


def flops(res) -> Dict[str, Any]:
    metrics, meta = _m(res), _meta(res)
    n = (meta.get("shape") or "0x0x0").split("x")[0]
    rows: List[Dict[str, Any]] = []
    peak, peak_dtype = None, None
    for dt, v in metrics.items():
        if "error" in v:
            continue
        rows.append({
            "dtype": dt,
            "n": int(n) if str(n).isdigit() else None,
            "tflops": v.get("tflops_best"),
            "tflops_median": v.get("tflops_median"),
            "cv_pct": v.get("cv_pct"),
            "reps": v.get("n"),
        })
        if v.get("tflops_median") and (peak is None or v["tflops_median"] > peak):
            peak, peak_dtype = v["tflops_median"], dt
    rows.sort(key=lambda r: r.get("tflops") or 0, reverse=True)
    return {"rows": rows, "peak_tflops": peak, "peak_dtype": peak_dtype, "meta": meta}


def membw(res) -> Dict[str, Any]:
    metrics, meta = _m(res), _meta(res)
    bytes_per = meta.get("bytes_per_array") or 0
    rows = []
    triad = None
    for k in ("copy", "mul", "add", "triad"):
        v = metrics.get(k)
        if not v or "error" in v:
            continue
        rows.append({
            "kernel": k,
            "arrays": v.get("arrays_touched"),
            "gbs": v.get("gbs_median"),
            "bytes": (v.get("arrays_touched") or 0) * bytes_per,
            "reps": meta.get("iters"),
        })
        if k == "triad":
            triad = v.get("gbs_median")
    return {"rows": rows, "triad_gbs": triad, "meta": meta}


def pcie(res) -> Dict[str, Any]:
    metrics, meta = _m(res), _meta(res)
    buf_mib = round((meta.get("transfer_bytes") or 0) / (1024 ** 2), 1)
    per = []
    for gid, v in (metrics.get("per_gpu") or {}).items():
        if "error" in v:
            continue
        per.append({
            "gpu": int(gid),
            "h2d_gbs": v.get("h2d_gbs_median"),
            "d2h_gbs": v.get("d2h_gbs_median"),
            "buffer_mib": buf_mib,
        })
    per.sort(key=lambda r: r["gpu"])
    c = metrics.get("concurrent") or {}
    conc = {}
    if c and "error" not in c:
        conc = {
            "retention_pct": c.get("retention_pct"),
            "aggregate_gbs": c.get("h2d_aggregate_gbs_median"),
            "sum_isolated_gbs": c.get("sum_of_isolated_gbs"),
        }
    h2d = max((r["h2d_gbs"] or 0) for r in per) if per else None
    return {"per_gpu": per, "concurrent": conc, "h2d_gbs": h2d, "meta": meta}


def nccl(res) -> Dict[str, Any]:
    metrics, meta = _m(res), _meta(res)
    sets_out: List[Dict[str, Any]] = []
    peak_all, world = None, None
    for s in metrics.get("sets") or []:
        rows = []
        peak = None
        for coll, by_size in (s.get("collectives") or {}).items():
            for size_str, v in by_size.items():
                if "error" in v:
                    continue
                nbytes = int(size_str)
                alg = v.get("algbw_gbs_median")
                rows.append({
                    "collective": coll,
                    "size_mib": round(nbytes / (1024 ** 2), 1),
                    "algbw_gbs": alg,
                    "busbw_gbs": v.get("busbw_gbs_median"),
                    "time_ms": round(nbytes / alg / 1e6, 3) if alg else None,
                })
                if coll == "all_reduce" and v.get("busbw_gbs_median"):
                    peak = max(peak or 0, v["busbw_gbs_median"])
        rows.sort(key=lambda r: (r["collective"], r["size_mib"]))
        sets_out.append({
            "label": s.get("name"),
            "gpus": s.get("gpus"),
            "world_size": s.get("world_size"),
            "worst_link": s.get("worst_link"),
            "peak_busbw_gbs": peak,
            "rows": rows,
            "error": s.get("error"),
        })
        if peak and (peak_all is None or peak > peak_all):
            peak_all, world = peak, s.get("world_size")
    return {"sets": sets_out, "peak_busbw_gbs": peak_all, "world_size": world, "meta": meta}


def topology(topo: Dict[str, Any]) -> Dict[str, Any]:
    if not topo:
        return {}
    groups = topo.get("numa_groups") or []
    return {
        "devices": topo.get("gpus") or [],
        "matrix": topo.get("matrix") or {},
        "numa_groups": {f"group {i}": g for i, g in enumerate(groups)},
        "has_nvlink": topo.get("has_nvlink"),
        "is_multi_root": topo.get("is_multi_root"),
        "numa_source": topo.get("numa_source"),
    }


def crosscheck_inputs(torch_data: Dict[str, Any], native: Dict[str, Any]) -> tuple:
    """Pull the four comparable quantities out of each backend's results."""
    t: Dict[str, Optional[float]] = {}
    f = torch_data.get("flops") or {}
    if f.get("peak_tflops"):
        t["flops"] = f["peak_tflops"]
    mb = torch_data.get("membw") or {}
    if mb.get("triad_gbs"):
        t["membw"] = mb["triad_gbs"]
    pc = torch_data.get("pcie") or {}
    if pc.get("h2d_gbs"):
        t["pcie_h2d"] = pc["h2d_gbs"]
    nc = torch_data.get("nccl") or {}
    if nc.get("peak_busbw_gbs"):
        t["allreduce_busbw"] = nc["peak_busbw_gbs"]

    n: Dict[str, Optional[float]] = {}
    if (native.get("cutlass") or {}).get("peak_tflops"):
        n["flops"] = native["cutlass"]["peak_tflops"]
    bs = native.get("babelstream") or {}
    if (bs.get("triad") or {}).get("gbs"):
        n["membw"] = bs["triad"]["gbs"]
    nb = native.get("nvbandwidth") or {}
    if nb.get("h2d_mean_gbs"):
        n["pcie_h2d"] = nb["h2d_mean_gbs"]
    nt = native.get("nccl_tests") or {}
    if nt.get("peak_busbw_gbs"):
        n["allreduce_busbw"] = nt["peak_busbw_gbs"]
    return t, n
