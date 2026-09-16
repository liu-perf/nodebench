"""Multi-GPU scaling: speedup, efficiency, and where the loss went.

The number everyone quotes is speedup. The number that actually tells you
something is **per-step inflation**: how much slower one training step gets when
you add cards. Speedup hides it, because throughput also grows with the global
batch, so a job can show "3.2x on 4 GPUs" while every individual step got 25%
slower -- and that 25% is the communication cost you were trying to measure.

Efficiency baseline
-------------------
Efficiency is always relative to a stated baseline, and the baseline must be a
real measured run on this node, not a datasheet. Comparing a 4-GPU run to a
theoretical single-GPU number folds the node's own single-card behaviour
(clocks, thermals, PCIe generation) into the "scaling" figure, and then the
scaling figure is no longer about scaling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Below this, the extra cards are not paying for themselves.
EFFICIENCY_POOR = 0.70
EFFICIENCY_GOOD = 0.85


def efficiency(speedup: float, gpus: int, base_gpus: int = 1) -> float:
    ideal = gpus / base_gpus
    return speedup / ideal if ideal else 0.0


def _verdict(eff: float) -> str:
    if eff >= EFFICIENCY_GOOD:
        return "good"
    if eff >= EFFICIENCY_POOR:
        return "warning"
    return "serious"


def scaling_table(
    runs: List[Dict[str, Any]],
    metric: str = "samples_per_s",
    step_metric: str = "iter_time_s_median",
    baseline_gpus: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a scaling table from a list of per-configuration summaries.

    Each entry in ``runs`` needs at least ``gpus`` and ``metric``. ``label`` and
    ``step_metric`` are used when present. Entries are sorted by GPU count and
    the smallest is the baseline unless ``baseline_gpus`` says otherwise.
    """
    usable = [r for r in runs if r.get(metric) and r.get("gpus")]
    if not usable:
        return {"ok": False, "error": f"no runs carry both 'gpus' and '{metric}'"}

    usable = sorted(usable, key=lambda r: r["gpus"])
    base = next((r for r in usable if r["gpus"] == baseline_gpus), usable[0]) if baseline_gpus else usable[0]
    b_gpus = base["gpus"]
    b_val = base[metric]
    b_step = base.get(step_metric)

    rows: List[Dict[str, Any]] = []
    for r in usable:
        sp = r[metric] / b_val if b_val else 0.0
        eff = efficiency(sp, r["gpus"], b_gpus)
        row: Dict[str, Any] = {
            "label": r.get("label") or f"{r['gpus']}gpu",
            "gpus": r["gpus"],
            metric: round(r[metric], 3),
            "speedup": round(sp, 3),
            "efficiency": round(eff, 4),
            "efficiency_pct": round(eff * 100, 1),
            "verdict": _verdict(eff),
        }
        step = r.get(step_metric)
        if step and b_step:
            row[step_metric] = round(step, 4)
            row["step_inflation_pct"] = round((step / b_step - 1) * 100, 1)
        rows.append(row)

    notes: List[str] = []
    worst = min(rows, key=lambda x: x["efficiency"])
    if worst["efficiency"] < EFFICIENCY_POOR:
        notes.append(
            f"{worst['label']} runs at {worst['efficiency_pct']:.0f}% efficiency. Below "
            f"{EFFICIENCY_POOR:.0%} the added cards are mostly paying for communication. "
            "Check the P2P verdict and the per-step inflation before blaming the model."
        )
    inflated = [x for x in rows if x.get("step_inflation_pct", 0) > 20]
    if inflated:
        notes.append(
            "Per-step time grew by more than 20% for "
            + ", ".join(x["label"] for x in inflated)
            + ". Throughput still rose because the global batch grew; the step itself "
            "got slower. That gap is the communication cost."
        )

    # Two same-N configurations that differ only in placement isolate topology.
    by_n: Dict[int, List[Dict[str, Any]]] = {}
    for x in rows:
        by_n.setdefault(x["gpus"], []).append(x)
    topology_pairs = []
    for n, group in by_n.items():
        if len(group) < 2:
            continue
        best = max(group, key=lambda x: x["speedup"])
        wrst = min(group, key=lambda x: x["speedup"])
        delta = (best["speedup"] / wrst["speedup"] - 1) * 100 if wrst["speedup"] else 0
        topology_pairs.append(
            {
                "gpus": n,
                "best": best["label"],
                "worst": wrst["label"],
                "delta_pct": round(delta, 1),
            }
        )
        if delta > 5:
            notes.append(
                f"At {n} GPUs, '{best['label']}' beats '{wrst['label']}' by {delta:.0f}% with "
                "the same card count. The difference is placement, not compute -- pin the "
                "job to the better set."
            )

    return {
        "ok": True,
        "metric": metric,
        "baseline": {"label": base.get("label") or f"{b_gpus}gpu", "gpus": b_gpus, metric: b_val},
        "rows": rows,
        "topology_pairs": topology_pairs,
        "notes": notes,
        "caveat": (
            "Efficiency is relative to the measured baseline above, not to a datasheet. "
            "Global batch grows with GPU count, so speedup and per-step time move in "
            "opposite directions -- read both columns."
        ),
    }
