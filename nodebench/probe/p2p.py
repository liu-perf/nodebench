"""Hardware P2P vs software P2P -- the discriminator.

The mistake this module exists to prevent
-----------------------------------------
You run::

    nvidia-smi topo -p2p r

and every cell says ``CNS`` (Chipset Not Supported). You run
``p2pBandwidthLatencyTest`` and every pair says "CANNOT Access Peer". You run
``nvbandwidth`` and all 22 device-to-device tests come back ``Waived``.

Three independent tools agree, so you write "P2P is not enabled on this node"
in your report -- and you are wrong, because NCCL is quietly moving 40 GB/s.

Why all three tools are blind
-----------------------------
They probe exactly one thing: whether the *CUDA driver exposes a peer mapping*
between two devices, which on PCIe depends on the root complex allowing
peer-to-peer TLP routing (ACS off, same root complex, etc.). Call that
**hardware P2P**.

But that is not the only fast path. A driver can implement peer transfer at the
software layer -- staging through a driver-managed path, or a vendor-enabled
routing mode -- without ever exposing ``cudaDeviceCanAccessPeer == 1``. Call
that **software P2P**. It is invisible to every tool above, because every tool
above asks the *capability* question, not the *throughput* question.

The only way to see software P2P is to measure something that would be slow
without it:

  * NCCL bus bandwidth on a collective, and
  * real multi-GPU training scaling efficiency.

The discriminator below
-----------------------
We combine four pieces of evidence and emit a verdict with its reasoning
attached, so a reader can disagree with the conclusion but not with the data:

  1. ``cudaDeviceCanAccessPeer`` for every pair (via torch)
  2. ``nvidia-smi topo -p2p r`` -- the hardware/ACS view
  3. measured single-card H2D bandwidth (the yardstick)
  4. measured NCCL all-reduce bus bandwidth

The ratio ``busbw / h2d`` is the tell. When every byte has to bounce through
host memory, an all-reduce moves each byte across PCIe several times and the
achievable bus bandwidth collapses to roughly a fifth of the raw link rate.
When a peer path exists -- hardware or software -- it lands near or above half
of it. The thresholds below are deliberately wide, with an explicit
"inconclusive" band, because a benchmark that reports a confident wrong answer
is worse than one that says it does not know.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# busbw / h2d ratio. Calibrated against PCIe Gen4/Gen5 x16 nodes with and
# without a peer path. The gap between the two thresholds is the honest
# "cannot tell" region.
RATIO_PEER_PATH = 0.55  # at or above: a peer path is definitely being used
RATIO_HOST_BOUNCE = 0.30  # at or below: consistent with host staging only


@dataclass
class P2PVerdict:
    mode: str  # hardware | software | host_bounce | disabled | inconclusive | unknown
    confidence: str  # high | medium | low
    headline: str
    explanation: str
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sh(cmd: List[str], timeout: int = 120) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout or r.stderr
    except Exception:
        return None


# --------------------------------------------------------------------------
# Evidence collection
# --------------------------------------------------------------------------

def cuda_peer_matrix() -> Dict[str, Any]:
    """`cudaDeviceCanAccessPeer` for every ordered pair, via torch."""
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch.cuda unavailable"}
    n = torch.cuda.device_count()
    m: Dict[int, Dict[int, bool]] = {}
    for i in range(n):
        m[i] = {}
        for j in range(n):
            if i == j:
                continue
            try:
                m[i][j] = bool(torch.cuda.can_device_access_peer(i, j))
            except Exception:
                m[i][j] = False
    pairs = [(i, j) for i in m for j in m[i]]
    ok = [(i, j) for i, j in pairs if m[i][j]]
    return {
        "available": True,
        "matrix": m,
        "n_pairs": len(pairs),
        "n_peer_capable": len(ok),
        "all_capable": len(ok) == len(pairs) and len(pairs) > 0,
        "none_capable": len(ok) == 0,
    }


def parse_topo_p2p(text: str) -> Dict[str, Any]:
    """Pure function. Parse `nvidia-smi topo -p2p r` / `-p2p w`.

    Cell values: ``OK`` supported, ``CNS`` chipset not supported,
    ``NS`` not supported, ``X`` self, ``NA`` not applicable.
    """
    counts: Dict[str, int] = {}
    matrix: Dict[int, Dict[int, str]] = {}
    for ln in text.splitlines():
        m = re.match(r"^\s*GPU(\d+)\s+(.*)$", ln)
        if not m:
            continue
        gid = int(m.group(1))
        cells = m.group(2).split()
        row: Dict[int, str] = {}
        for j, c in enumerate(cells):
            if c == "X":
                continue
            row[j] = c
            counts[c] = counts.get(c, 0) + 1
        matrix[gid] = row
    total = sum(counts.values())
    return {
        "parsed": total > 0,
        "matrix": matrix,
        "counts": counts,
        "total_cells": total,
        "all_ok": total > 0 and counts.get("OK", 0) == total,
        "none_ok": total > 0 and counts.get("OK", 0) == 0,
        "dominant": max(counts, key=counts.get) if counts else None,
    }


def hardware_p2p_view() -> Dict[str, Any]:
    out = _sh(["nvidia-smi", "topo", "-p2p", "r"])
    if not out:
        return {"parsed": False, "reason": "nvidia-smi topo -p2p unavailable"}
    d = parse_topo_p2p(out)
    d["raw"] = out
    return d


# --------------------------------------------------------------------------
# The discriminator
# --------------------------------------------------------------------------

def classify_p2p(
    cuda_peer: Dict[str, Any],
    hw_view: Dict[str, Any],
    h2d_gbs: Optional[float],
    allreduce_busbw_gbs: Optional[float],
    nvlink: bool = False,
) -> P2PVerdict:
    """Pure function -- all four evidence sources in, one verdict out.

    Kept free of IO so it can be unit-tested against recorded evidence.
    """
    ev: Dict[str, Any] = {
        "cuda_can_access_peer": {
            "n_peer_capable": cuda_peer.get("n_peer_capable"),
            "n_pairs": cuda_peer.get("n_pairs"),
            "all_capable": cuda_peer.get("all_capable"),
        },
        "nvidia_smi_topo_p2p": {
            "counts": hw_view.get("counts"),
            "all_ok": hw_view.get("all_ok"),
            "none_ok": hw_view.get("none_ok"),
        },
        "h2d_gbs": h2d_gbs,
        "allreduce_busbw_gbs": allreduce_busbw_gbs,
        "busbw_over_h2d": None,
        "nvlink": nvlink,
        "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE"),
        "NCCL_P2P_LEVEL": os.environ.get("NCCL_P2P_LEVEL"),
        "thresholds": {"peer_path": RATIO_PEER_PATH, "host_bounce": RATIO_HOST_BOUNCE},
    }

    ratio = None
    if h2d_gbs and allreduce_busbw_gbs and h2d_gbs > 0:
        ratio = allreduce_busbw_gbs / h2d_gbs
        ev["busbw_over_h2d"] = round(ratio, 3)

    hw_ok = bool(hw_view.get("all_ok"))
    hw_none = bool(hw_view.get("none_ok"))
    cuda_ok = bool(cuda_peer.get("all_capable"))
    cuda_none = bool(cuda_peer.get("none_capable"))
    disabled = os.environ.get("NCCL_P2P_DISABLE") == "1"

    if nvlink:
        return P2PVerdict(
            mode="hardware",
            confidence="high",
            headline="NVLink present -- peer transfer is over NVLink, not PCIe",
            explanation=(
                "The topology matrix reports NV* links. PCIe P2P discussion does not apply: "
                "collectives ride NVLink and the busbw/H2D ratio used elsewhere in this "
                "module is not a meaningful test here."
            ),
            evidence=ev,
        )

    if disabled:
        return P2PVerdict(
            mode="disabled",
            confidence="high",
            headline="P2P is switched off by NCCL_P2P_DISABLE=1",
            explanation=(
                "Whatever the hardware supports, NCCL has been told not to use it. Any "
                "scaling number from this run is a lower bound. Unset the variable and "
                "re-measure before drawing a conclusion about the node."
            ),
            evidence=ev,
        )

    if ratio is None:
        mode = "hardware" if (hw_ok or cuda_ok) else "unknown"
        return P2PVerdict(
            mode=mode,
            confidence="low",
            headline="Capability probed, throughput not measured",
            explanation=(
                "Only the capability question was answered. Without an NCCL bus-bandwidth "
                "measurement it is impossible to tell software P2P apart from no P2P -- "
                "that is exactly the case this module exists for. Run the nccl module."
            ),
            evidence=ev,
        )

    fast = ratio >= RATIO_PEER_PATH
    slow = ratio <= RATIO_HOST_BOUNCE

    # --- the interesting case -------------------------------------------
    if (hw_none or cuda_none) and fast:
        return P2PVerdict(
            mode="software",
            confidence="high",
            headline="Software P2P is active and every hardware probe is blind to it",
            explanation=(
                f"All-reduce bus bandwidth is {allreduce_busbw_gbs:.2f} GB/s, which is "
                f"{ratio:.0%} of the {h2d_gbs:.2f} GB/s host-to-device rate. Pure host "
                "staging cannot reach that -- each byte would cross the link several times. "
                "So a peer path is in use.\n\n"
                "Meanwhile nvidia-smi topo -p2p and cudaDeviceCanAccessPeer both report no "
                "peer capability. Those tools only ask whether the driver exposes a peer "
                "*mapping*; they cannot see a driver- or platform-level routing mode. This "
                "combination -- hardware probes negative, measured throughput high -- is the "
                "signature of software P2P.\n\n"
                "Practical consequence: do not accept or reject a node on `topo -p2p`. "
                "Accept on NCCL bus bandwidth plus real multi-GPU scaling efficiency."
            ),
            evidence=ev,
        )

    if (hw_ok or cuda_ok) and fast:
        return P2PVerdict(
            mode="hardware",
            confidence="high",
            headline="Hardware P2P is available and being used",
            explanation=(
                f"Peer capability is reported by the hardware probes, and measured all-reduce "
                f"bus bandwidth is {allreduce_busbw_gbs:.2f} GB/s ({ratio:.0%} of H2D), which "
                "confirms the path is actually taken rather than merely advertised."
            ),
            evidence=ev,
        )

    if (hw_ok or cuda_ok) and slow:
        return P2PVerdict(
            mode="host_bounce",
            confidence="medium",
            headline="Peer capability is advertised but the traffic is not using it",
            explanation=(
                f"The probes say peer access is possible, yet all-reduce bus bandwidth is only "
                f"{allreduce_busbw_gbs:.2f} GB/s ({ratio:.0%} of H2D) -- the profile of host "
                "staging. Check NCCL_P2P_LEVEL, whether the process is pinned to the wrong "
                "NUMA node, and whether an IOMMU setting is forcing traffic through the host. "
                "Run with NCCL_DEBUG=INFO and read which transport NCCL selected."
            ),
            evidence=ev,
        )

    if slow:
        return P2PVerdict(
            mode="host_bounce",
            confidence="high",
            headline="No effective peer path -- collectives are staging through host memory",
            explanation=(
                f"All-reduce bus bandwidth is {allreduce_busbw_gbs:.2f} GB/s, only {ratio:.0%} "
                f"of the {h2d_gbs:.2f} GB/s host-to-device rate, and no probe reports peer "
                "capability. Every byte is making a round trip through system memory.\n\n"
                "Two things are worth checking before concluding the node is simply built this "
                "way, because both are cheap and both have been seen to cause exactly this:\n"
                "  1. Host memory channel population. A node with only some channels filled "
                "starves the staging path, and staging is the whole data path here.\n"
                "  2. Whether the platform offers a driver-level peer routing mode that is "
                "currently off."
            ),
            evidence=ev,
        )

    return P2PVerdict(
        mode="inconclusive",
        confidence="low",
        headline="Measurement falls between the two thresholds",
        explanation=(
            f"busbw/H2D = {ratio:.0%}, between the host-staging ceiling "
            f"({RATIO_HOST_BOUNCE:.0%}) and the peer-path floor ({RATIO_PEER_PATH:.0%}). "
            "nodebench will not guess. Re-run with larger messages, confirm no other job is "
            "on the node, and read NCCL_DEBUG=INFO for the selected transport."
        ),
        evidence=ev,
    )


def probe_p2p(
    h2d_gbs: Optional[float] = None,
    allreduce_busbw_gbs: Optional[float] = None,
    nvlink: bool = False,
) -> Dict[str, Any]:
    """Collect evidence and classify. Safe to call with no measurements."""
    cuda_peer = cuda_peer_matrix()
    hw = hardware_p2p_view()
    verdict = classify_p2p(cuda_peer, hw, h2d_gbs, allreduce_busbw_gbs, nvlink=nvlink)
    return {
        "cuda_peer": cuda_peer,
        "hardware_view": {k: v for k, v in hw.items() if k != "raw"},
        "hardware_view_raw": hw.get("raw", ""),
        "verdict": verdict.to_dict(),
    }
