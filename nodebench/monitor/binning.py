"""Split a monitor CSV into phases without anyone writing down time windows.

The problem
-----------
You run a scaling matrix -- 1, 2, 4, 8 GPUs -- while a monitor samples in the
background. Afterwards you need per-phase statistics. The usual approach is to
note the wall-clock start and end of each run and slice the CSV by hand.

That is fragile in a way that is hard to notice: the notes drift from reality by
a few seconds, and those seconds are exactly the process startup and the model
load, which are the highest-PCIe-traffic moments in the entire run. Include them
and your "training PCIe traffic" is really "the weights being uploaded once".
The error can be two orders of magnitude and it looks completely reasonable.

The fix
-------
Do not use time at all. Bin each row by *which set of GPUs was busy*. A row
where GPUs {0,1,2,3} are above the busy threshold belongs to the 4-GPU phase, by
construction, no matter when it happened or how long setup took.

This makes phase attribution a property of the data instead of a property of
someone's notes, and it means re-analysing an old CSV needs nothing but the CSV.
"""

from __future__ import annotations

import csv
import re
from typing import Any, Dict, List, Optional, Tuple


def load_csv(path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a nodebench monitor CSV, skipping the leading ``#`` provenance lines."""
    rows: List[Dict[str, str]] = []
    header: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for raw in reader:
            if not raw:
                continue
            if raw[0].startswith("#"):
                continue
            if not header:
                header = raw
                continue
            rows.append(dict(zip(header, raw)))
    return header, rows


def _gpu_ids(header: List[str]) -> List[int]:
    ids = set()
    for h in header:
        m = re.match(r"^gpu(\d+)_", h)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def bin_by_active_set(
    path, busy_threshold: float = 90.0, min_rows: int = 3
) -> Dict[str, Dict[str, Any]]:
    """Group rows by the set of GPUs whose ``busy_pct`` >= threshold.

    Returns {"0,1,2,3": {...stats...}, "idle": {...}, ...}
    """
    header, rows = load_csv(path)
    if not rows:
        return {}
    ids = _gpu_ids(header)
    bins: Dict[str, List[Dict[str, str]]] = {}

    for r in rows:
        active = [i for i in ids if _f(r, f"gpu{i}_busy_pct") >= busy_threshold]
        key = ",".join(str(i) for i in active) if active else "idle"
        bins.setdefault(key, []).append(r)

    out: Dict[str, Dict[str, Any]] = {}
    for key, group in bins.items():
        if len(group) < min_rows:
            continue
        active = [] if key == "idle" else [int(x) for x in key.split(",")]
        out[key] = _stats(group, ids, active, key)
    return out


def _stats(group: List[Dict[str, str]], all_ids: List[int], active: List[int], key: str
           ) -> Dict[str, Any]:
    n = len(group)

    def mean(field: str, over: List[int]) -> Optional[float]:
        if not over:
            return None
        vals = [_f(r, f"gpu{i}_{field}") for r in group for i in over]
        return sum(vals) / len(vals) if vals else None

    def peak(field: str, over: List[int]) -> Optional[float]:
        if not over:
            return None
        vals = [_f(r, f"gpu{i}_{field}") for r in group for i in over]
        return max(vals) if vals else None

    throttles: Dict[str, int] = {}
    for r in group:
        for i in (active or all_ids):
            for reason in (r.get(f"gpu{i}_throttle", "") or "").split("|"):
                if reason:
                    throttles[reason] = throttles.get(reason, 0) + 1

    scope = active or all_ids
    return {
        "label": "idle / control group" if key == "idle" else f"{len(active)} GPU(s) busy: {key}",
        "active_gpus": active,
        "n_active": len(active),
        "rows": n,
        # Averaged over the ACTIVE cards only. Averaging over all eight when
        # only one is working produces a number that looks like utilisation and
        # is not -- it is a phase tag with a percent sign.
        "busy_pct_mean_active": _round(mean("busy_pct", scope)),
        "tx_gbs_mean_active": _round(mean("tx_gbs", scope), 4),
        "rx_gbs_mean_active": _round(mean("rx_gbs", scope), 4),
        "tx_gbs_peak_active": _round(peak("tx_gbs", scope), 4),
        "rx_gbs_peak_active": _round(peak("rx_gbs", scope), 4),
        "power_w_mean_active": _round(mean("power_w", scope), 1),
        "power_w_total_mean": _round(
            sum(_f(r, f"gpu{i}_power_w") for r in group for i in all_ids) / n, 1
        ),
        "temp_c_max": _round(peak("temp_c", scope), 0),
        "sm_clk_mhz_mean": _round(mean("sm_clk_mhz", scope), 0),
        "mem_used_mib_peak": _round(peak("mem_used_mib", scope), 0),
        "throttle_counts": throttles,
        "perf_limited_rows": sum(
            1 for r in group
            for i in scope
            if any(x in (r.get(f"gpu{i}_throttle", "") or "")
                   for x in ("power_cap", "thermal", "hw_slowdown", "power_brake"))
        ),
    }


def _round(v, nd: int = 2):
    return None if v is None else round(v, nd)


def summarize_bins(bins: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Turn bins into report-ready facts, including the noise floor."""
    idle = bins.get("idle")
    busy = {k: v for k, v in bins.items() if k != "idle"}
    ordered = sorted(busy.items(), key=lambda kv: (kv[1]["n_active"], kv[0]))

    out: Dict[str, Any] = {
        "phases": [dict(key=k, **v) for k, v in ordered],
        "noise_floor": None,
        "signal_to_noise": None,
    }
    if idle:
        out["noise_floor"] = {
            "rows": idle["rows"],
            "tx_gbs_mean": idle["tx_gbs_mean_active"],
            "rx_gbs_mean": idle["rx_gbs_mean_active"],
            "power_w_total_mean": idle["power_w_total_mean"],
            "note": (
                "Control group. Any PCIe figure below this is indistinguishable from "
                "an idle machine and must not be reported as a measurement."
            ),
        }
        floor = max(idle["tx_gbs_mean_active"] or 0, idle["rx_gbs_mean_active"] or 0)
        if floor > 0 and ordered:
            peak = max(
                max(p[1]["tx_gbs_mean_active"] or 0, p[1]["rx_gbs_mean_active"] or 0)
                for p in ordered
            )
            out["signal_to_noise"] = round(peak / floor, 1)
    return out


def per_gpu_stats(path) -> Dict[str, Dict[str, Any]]:
    """Per-card statistics over the whole file, one entry per GPU.

    Used for the control window, where the question is not "what happened
    during which phase" but "was each individual card actually idle". A node
    where seven cards are quiet and one is busy averages out to "mostly idle",
    which is exactly the conclusion that must not be drawn.
    """
    header, rows = load_csv(path)
    if not rows:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for i in _gpu_ids(header):
        entry: Dict[str, Any] = {}
        for field in ("tx_gbs", "rx_gbs", "busy_pct", "mem_io_pct", "power_w",
                      "temp_c", "sm_clk_mhz", "mem_used_mib"):
            vals = [_f(r, f"gpu{i}_{field}") for r in rows if f"gpu{i}_{field}" in r]
            if not vals:
                continue
            entry[field] = {
                "mean": round(sum(vals) / len(vals), 4),
                "max": round(max(vals), 4),
                "min": round(min(vals), 4),
                "n": len(vals),
            }
        out[str(i)] = entry
    return out
