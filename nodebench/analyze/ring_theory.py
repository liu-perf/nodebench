"""Theory-vs-measurement reconciliation for ring collectives.

Why bother
----------
A measured bandwidth on its own is unfalsifiable. 40 GB/s could be excellent or
terrible; without a predicted value there is no way to tell, and no way to
notice when the measurement itself is wrong.

Computing what the number *should* be and dividing gives a ratio that catches
real mistakes:

  * ratio far below 1  -- something is bottlenecking, or the measurement missed
    part of the transfer
  * ratio near 1       -- the model and the machine agree; both are probably right
  * ratio far above 1  -- the model is wrong for this hardware (an NVLink or
    NVSwitch path the ring model does not describe), or a counter is being
    read with the wrong units

The ring model
--------------
Ring all-reduce runs in two phases -- reduce-scatter then all-gather -- each of
N-1 steps, each step moving 1/N of the buffer. So each rank sends::

    2 x (N-1)/N x buffer_bytes

That factor is exactly the busbw correction nccl-tests applies, which is why
the two are defined together here rather than in two places that can drift.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

FACTORS = {
    "all_reduce": lambda n: 2.0 * (n - 1) / n,
    "all_gather": lambda n: (n - 1) / n,
    "reduce_scatter": lambda n: (n - 1) / n,
    "all_to_all": lambda n: (n - 1) / n,
    "broadcast": lambda n: 1.0,
    "reduce": lambda n: 1.0,
}

FORMULAS = {
    "all_reduce": "2(N-1)/N",
    "all_gather": "(N-1)/N",
    "reduce_scatter": "(N-1)/N",
    "all_to_all": "(N-1)/N",
    "broadcast": "1",
    "reduce": "1",
}


def busbw_factor(collective: str, world_size: int) -> float:
    if world_size < 2:
        return 0.0
    return FACTORS.get(collective, lambda n: 1.0)(world_size)


def ring_allreduce_bytes(buffer_bytes: int, world_size: int) -> float:
    """Bytes each rank pushes across the wire for one ring all-reduce."""
    if world_size < 2:
        return 0.0
    return 2.0 * (world_size - 1) / world_size * buffer_bytes


def gradient_traffic_per_step(
    trainable_params: int, world_size: int, bytes_per_param: int = 4
) -> Dict[str, Any]:
    """Per-rank all-reduce volume for one optimizer step of data-parallel training.

    The reason to compute this: it explains, quantitatively, why LoRA scales
    almost linearly on a PCIe node while full fine-tuning does not. Only
    trainable parameters are reduced, so the ratio of full-finetune traffic to
    LoRA traffic is just the ratio of their trainable counts -- often two
    orders of magnitude. A node can be perfectly adequate for one and hopeless
    for the other, and no microbenchmark will tell you which.
    """
    if world_size < 2:
        return {"bytes_per_step": 0.0, "gib_per_step": 0.0, "world_size": world_size}
    b = ring_allreduce_bytes(trainable_params * bytes_per_param, world_size)
    return {
        "trainable_params": trainable_params,
        "bytes_per_param": bytes_per_param,
        "world_size": world_size,
        "bytes_per_step": b,
        "gib_per_step": round(b / (1024 ** 3), 6),
        "formula": f"2*(N-1)/N * {trainable_params} * {bytes_per_param}B, N={world_size}",
    }


def reconcile(
    measured_gbs: float,
    collective: str,
    world_size: int,
    algbw_gbs: Optional[float] = None,
) -> Dict[str, Any]:
    """Check that a reported busbw is consistent with its own algbw.

    This catches the single most common multi-GPU reporting error: quoting
    algbw as if it were busbw, or applying the correction twice.
    """
    f = busbw_factor(collective, world_size)
    out: Dict[str, Any] = {
        "collective": collective,
        "world_size": world_size,
        "busbw_factor": round(f, 4),
        "formula": FORMULAS.get(collective, "1"),
        "measured_busbw_gbs": measured_gbs,
    }
    if algbw_gbs:
        expected = algbw_gbs * f
        out["algbw_gbs"] = algbw_gbs
        out["expected_busbw_gbs"] = round(expected, 3)
        out["ratio_measured_over_expected"] = round(measured_gbs / expected, 4) if expected else None
        ok = expected and abs(measured_gbs / expected - 1) < 0.02
        out["consistent"] = bool(ok)
        if not ok and expected:
            out["warning"] = (
                "busbw does not equal algbw x factor. Either the correction was not applied, "
                "or it was applied twice, or the two came from different runs."
            )
    return out


def reconcile_pcie_against_theory(
    measured_pcie_gbs: float,
    trainable_params: int,
    world_size: int,
    step_time_s: float,
    bytes_per_param: int = 4,
) -> Dict[str, Any]:
    """Compare monitored PCIe throughput to what the gradient exchange requires.

    Read the ratio, not the difference, and read it with the NVML caveat in
    mind: the PCIe counter is a ~20 ms windowed average, so a measured mean
    above theory is expected -- the counter samples the bursts, and the
    collective is bursty. A ratio in the 1.2-3x range is normal on a healthy
    node. A ratio far BELOW 1 means the traffic is not going where you think.
    """
    theory = gradient_traffic_per_step(trainable_params, world_size, bytes_per_param)
    if step_time_s <= 0:
        return {"error": "step_time_s must be positive", **theory}
    theory_gbs = theory["bytes_per_step"] / step_time_s / 1e9
    ratio = measured_pcie_gbs / theory_gbs if theory_gbs else None
    return {
        **theory,
        "step_time_s": step_time_s,
        "theoretical_gbs": round(theory_gbs, 4),
        "measured_gbs": round(measured_pcie_gbs, 4),
        "ratio_measured_over_theory": round(ratio, 3) if ratio else None,
        "interpretation": (
            "Measured above theory is expected: NVML's PCIe counter averages a ~20 ms window "
            "and collectives are bursty, so sampling is biased toward the bursts. Ratios of "
            "roughly 1.2-3x are normal. A ratio below 1 means the transfer is not happening "
            "over the path being monitored."
        ),
    }
