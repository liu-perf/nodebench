"""The control group.

Every workload measurement in this tool is preceded by a fixed idle sampling
window. That window is not a warmup and it is not optional -- it is the control
group, and without it none of the workload numbers can be defended.

What it catches, in practice:

  * another user's job already on the node (power floor well above idle)
  * a card stuck in a high clock state, or stuck throttled
  * background PCIe traffic from monitoring agents or a mounted filesystem
  * a GPU that is not actually idle because a zombie process holds context

Any of these silently contaminates the workload numbers. The failure mode is
not a crash; it is a plausible-looking result that cannot be reproduced next
week, and by then nobody remembers what else was running.

Signal-to-noise is the deliverable
----------------------------------
Reporting "42 GB/s during the workload" is weaker than reporting "42 GB/s
during the workload against a 0.3 GB/s floor". The second states, in the
artifact itself, that the measurement is 140x above the noise. When the ratio
is small the honest conclusion is that the workload was not measured at all.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Below this the "measurement" is mostly floor.
SNR_UNUSABLE = 3.0
SNR_WEAK = 10.0

# Idle power above this fraction of the card's limit means someone else is here.
IDLE_POWER_SUSPECT_FRAC = 0.25
IDLE_BUSY_SUSPECT_PCT = 5.0


def assess_idle(idle_summary: Dict[str, Any], power_limit_w: Optional[float] = None) -> Dict[str, Any]:
    """Judge whether the node was actually idle during the control window.

    ``idle_summary`` is a per-GPU stats dict as produced by monitor.summary():
    ``{gpu_index: {"power_w": {"mean": ...}, "busy_pct": {"mean": ...}, ...}}``.
    """
    findings: List[Dict[str, str]] = []
    per_gpu: Dict[str, Any] = {}
    clean = True

    for gid, stats in (idle_summary or {}).items():
        power = (stats.get("power_w") or {}).get("mean")
        busy = (stats.get("busy_pct") or {}).get("mean")
        mem = (stats.get("mem_used_mib") or {}).get("mean")
        tx = (stats.get("tx_gbs") or {}).get("mean")
        rx = (stats.get("rx_gbs") or {}).get("mean")

        entry = {
            "power_w": power,
            "busy_pct": busy,
            "mem_used_mib": mem,
            "pcie_floor_gbs": round((tx or 0) + (rx or 0), 4),
        }
        per_gpu[str(gid)] = entry

        if busy is not None and busy > IDLE_BUSY_SUSPECT_PCT:
            clean = False
            findings.append(
                {
                    "gpu": str(gid),
                    "level": "serious",
                    "text": (
                        f"GPU {gid} reports {busy:.1f}% busy while idle. A kernel is resident. "
                        "Something else is using this card -- the workload numbers below are "
                        "contaminated."
                    ),
                }
            )
        if power_limit_w and power and power > power_limit_w * IDLE_POWER_SUSPECT_FRAC:
            clean = False
            findings.append(
                {
                    "gpu": str(gid),
                    "level": "warning",
                    "text": (
                        f"GPU {gid} draws {power:.0f} W at idle, over "
                        f"{IDLE_POWER_SUSPECT_FRAC:.0%} of its {power_limit_w:.0f} W limit. "
                        "Either the card is not idle or it is held in a high clock state."
                    ),
                }
            )
        if mem and mem > 1024:
            clean = False
            findings.append(
                {
                    "gpu": str(gid),
                    "level": "warning",
                    "text": (
                        f"GPU {gid} already holds {mem:.0f} MiB before the run. A process has "
                        "context on it. Run `nvidia-smi` and identify it before trusting "
                        "anything measured here."
                    ),
                }
            )

    return {
        "clean": clean,
        "per_gpu": per_gpu,
        "findings": findings,
        "headline": (
            "Control window clean: all GPUs idle before the workload."
            if clean
            else "Control window NOT clean. The node was busy before the workload started."
        ),
    }


def signal_to_noise(workload_value: float, idle_value: float, quantity: str = "value") -> Dict[str, Any]:
    """Express a workload measurement relative to its own noise floor."""
    if idle_value is None or idle_value <= 0:
        return {
            "quantity": quantity,
            "workload": workload_value,
            "noise_floor": idle_value,
            "ratio": None,
            "level": "ok",
            "text": "Noise floor is zero or unmeasured; the workload value stands alone.",
        }
    ratio = workload_value / idle_value
    if ratio < SNR_UNUSABLE:
        level, text = "critical", (
            f"{quantity} is only {ratio:.1f}x the idle floor. This is not a measurement of the "
            "workload; it is mostly the floor. Do not report it as a workload number."
        )
    elif ratio < SNR_WEAK:
        level, text = "warning", (
            f"{quantity} is {ratio:.1f}x the idle floor. Usable, but subtract the floor before "
            "drawing conclusions and say that you did."
        )
    else:
        level, text = "good", f"{quantity} is {ratio:.0f}x the idle floor. The signal dominates."
    return {
        "quantity": quantity,
        "workload": round(workload_value, 4),
        "noise_floor": round(idle_value, 4),
        "net": round(workload_value - idle_value, 4),
        "ratio": round(ratio, 2),
        "level": level,
        "text": text,
    }
