"""Host-side facts: PCIe link, CPU, memory, NIC placement.

Why a GPU benchmark cares about the host
----------------------------------------
Every number here is a *denominator* for a number measured somewhere else.

A host-to-device reading of 25.5 GB/s means nothing on its own. Against a Gen5
x16 link (63.0 GB/s theoretical) it is 40% and something is broken. Against a
Gen5 x8 link (31.5 GB/s) it is 81% and the node is healthy. Same measurement,
opposite conclusions, and the only thing that separates them is the link -- which
is why this module exists.

The PCIe generation trap
------------------------
``nvidia-smi -q`` reports Max and Current for both generation and width, and you
want a *different one from each*:

* **Generation: take Max.** An idle link downtrains to Gen1 to save power. On the
  machine this module was written on, an idle Blackwell card reports
  ``PCIe Generation / Current : 1`` while being a perfectly healthy Gen5 part.
  Build the theoretical figure from that and a real 25.5 GB/s measurement scores
  1275% of theoretical -- wrong by a factor of sixteen, and invisible, because
  nobody files a bug against hardware that beat its spec.
* **Width: take Current.** Width does not downtrain with idleness. A card whose
  Max is 16x and Current is 8x sits in a slot that only wired eight lanes. That
  is permanent, it is real, and it is the most common reason one node transfers
  at half the speed of its supposedly identical twin.

Both readings are kept in the output regardless, because the *disagreement*
between them is itself a finding.

Platform
--------
``nvidia-smi -q`` works everywhere. ``lscpu``, ``/proc/meminfo`` and
``/sys/class/net`` are Linux-only, and on other platforms this module reports
what the standard library can see plus an explicit note about what is missing.
A blank field and an unavailable field are different facts.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

# GB/s per lane, per direction. Gen1-2 are 8b/10b, Gen3-5 are 128b/130b, Gen6 is
# PAM4 with FLIT encoding. These are the raw-rate-times-efficiency numbers the
# spec sheets quote, so a measurement can be read as a percentage of them.
PCIE_GBS_PER_LANE = {1: 0.250, 2: 0.500, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}


def pcie_theoretical_gbs(gen: Optional[int], width: Optional[int]) -> Optional[float]:
    """One-direction theoretical bandwidth, or None if either input is unknown.

    None rather than 0.0 on purpose: a zero would propagate into a ratio and
    produce either a crash or an infinity, and both read as a measurement.
    """
    if not gen or not width:
        return None
    per_lane = PCIE_GBS_PER_LANE.get(int(gen))
    if per_lane is None:
        return None
    return round(per_lane * int(width), 2)


def parse_pcie_query(text: str) -> Dict[int, Dict[str, Any]]:
    """Pure function. Pull per-GPU link info out of ``nvidia-smi -q``.

    The output is an indented tree, and the same field names (``Max``,
    ``Current``) appear under both ``PCIe Generation`` and ``Link Width``. So
    this tracks which subsection it is inside rather than matching on the leaf
    name, which would silently take whichever came last.
    """
    out: Dict[int, Dict[str, Any]] = {}
    idx = -1
    section: Optional[str] = None
    for ln in text.splitlines():
        s = ln.strip()
        m = re.match(r"^GPU\s+[0-9A-Fa-f]{8}:", s)
        if m:
            idx += 1
            out[idx] = {}
            section = None
            continue
        if idx < 0:
            continue
        if s.startswith("PCIe Generation"):
            section = "gen"
            continue
        if s.startswith("Link Width"):
            section = "width"
            continue
        # Any other subsection header ends the one we were in. Without this a
        # "Max" belonging to some later block would be filed as a link figure.
        if s and not s.startswith(("Max", "Current", "Device", "Host")) and ":" in s:
            if section and s.split(":")[0].strip() not in ("Max", "Current"):
                section = None
        m = re.match(r"^(Max|Current)\s*:\s*(\S+)", s)
        if not m or section is None:
            continue
        key, raw = m.group(1).lower(), m.group(2)
        val = _int_or_none(raw.rstrip("xX"))
        out[idx][f"{section}_{key}"] = val
    return out


def _int_or_none(s: str) -> Optional[int]:
    try:
        return int(s)
    except ValueError:
        return None


def summarize_pcie(links: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Turn raw Max/Current readings into the denominator plus its caveats."""
    per_gpu: Dict[str, Any] = {}
    notes: List[str] = []
    for i, d in sorted(links.items()):
        gen_max, gen_cur = d.get("gen_max"), d.get("gen_current")
        w_max, w_cur = d.get("width_max"), d.get("width_current")
        theo = pcie_theoretical_gbs(gen_max, w_cur)
        entry = {
            "gen_max": gen_max,
            "gen_current": gen_cur,
            "width_max": w_max,
            "width_current": w_cur,
            "theoretical_gbs": theo,
            # Spelled out so the report never has to re-derive which half of
            # which pair went into the number.
            "theoretical_basis": (
                f"Gen{gen_max} x{w_cur} = {theo} GB/s (max generation, negotiated width)"
                if theo else "unknown -- nvidia-smi did not report both fields"
            ),
        }
        if w_max and w_cur and w_cur < w_max:
            entry["width_downgraded"] = True
            notes.append(
                f"GPU {i} negotiated x{w_cur} on a card capable of x{w_max}. That halves the "
                f"host transfer ceiling and it is a property of the slot, not of the workload."
            )
        if gen_max and gen_cur and gen_cur < gen_max:
            entry["gen_downtrained_at_idle"] = True
        per_gpu[str(i)] = entry
    if any(v.get("gen_downtrained_at_idle") for v in per_gpu.values()):
        notes.append(
            "At least one link reports a current generation below its maximum. This is normal "
            "at idle -- PCIe downtrains to save power and retrains under load -- which is "
            "exactly why the theoretical figure above is built from the maximum generation. "
            "Using the idle reading would inflate the achieved percentage by up to 16x."
        )
    return {"per_gpu": per_gpu, "notes": notes}


def parse_lscpu(text: str) -> Dict[str, Any]:
    """Pure function. ``lscpu`` key: value pairs, reduced to what matters here."""
    kv: Dict[str, str] = {}
    for ln in text.splitlines():
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        kv[k.strip()] = v.strip()
    numa_nodes = _int_or_none(kv.get("NUMA node(s)", ""))
    out: Dict[str, Any] = {
        "model": kv.get("Model name"),
        "sockets": _int_or_none(kv.get("Socket(s)", "")),
        "cores_per_socket": _int_or_none(kv.get("Core(s) per socket", "")),
        "threads_per_core": _int_or_none(kv.get("Thread(s) per core", "")),
        "logical_cpus": _int_or_none(kv.get("CPU(s)", "")),
        "numa_nodes": numa_nodes,
    }
    out["numa_cpu_ranges"] = {
        # "NUMA node0 CPU(s)" -> "node0". Index 1, not 2: index 2 is the literal
        # "CPU(s)", which would file every node under one key and keep only the last.
        k.split()[1]: v for k, v in kv.items() if re.match(r"^NUMA node\d+ CPU", k)
    }
    return out


def parse_meminfo(text: str) -> Dict[str, Any]:
    """Pure function. ``/proc/meminfo`` -> total and available in GiB."""
    out: Dict[str, Any] = {}
    for ln in text.splitlines():
        m = re.match(r"^(MemTotal|MemAvailable):\s+(\d+)\s*kB", ln)
        if m:
            out[m.group(1)] = round(int(m.group(2)) / (1024 * 1024), 1)
    return {
        "total_gib": out.get("MemTotal"),
        "available_gib": out.get("MemAvailable"),
    }


def match_nics_to_gpus(
    nics: List[Dict[str, Any]], gpu_numa: Dict[Any, Optional[int]]
) -> Dict[str, Any]:
    """Pure function. Which GPUs sit on the same NUMA node as each NIC.

    This is the question that decides where a multi-node job's bottleneck is. A
    NIC on node 0 feeding GPUs on node 1 sends every byte across the CPU
    interconnect, and the resulting number looks like "the network is slow"
    rather than "the job was pinned wrong".
    """
    by_node: Dict[Optional[int], List[int]] = {}
    for g, node in gpu_numa.items():
        by_node.setdefault(node, []).append(int(g))
    for v in by_node.values():
        v.sort()

    rows: List[Dict[str, Any]] = []
    for n in nics:
        node = n.get("numa_node")
        local = by_node.get(node, []) if node is not None else []
        rows.append({**n, "local_gpus": local, "n_local_gpus": len(local)})

    known = [r for r in rows if r.get("numa_node") is not None]
    unmatched = [r["name"] for r in known if not r["local_gpus"]]
    if not rows:
        verdict = "no NICs found (or this platform does not expose /sys)"
    elif not known:
        verdict = "NICs found but none reported a NUMA node, so affinity is unknown"
    elif not any(v is not None for v in gpu_numa.values()):
        verdict = "GPU NUMA placement is unknown, so NIC affinity cannot be resolved"
    elif unmatched:
        verdict = (
            f"{len(unmatched)} NIC(s) have no GPU on their NUMA node ({', '.join(unmatched)}). "
            "Traffic through them crosses the CPU interconnect."
        )
    else:
        verdict = "every NIC has at least one GPU on its own NUMA node"
    return {"nics": rows, "verdict": verdict, "gpus_by_numa": {str(k): v for k, v in by_node.items()}}


# --------------------------------------------------------------------------
# IO layer. Everything above is pure and unit-tested; everything below just
# fetches text and hands it upward.
# --------------------------------------------------------------------------


def _sh(cmd: List[str], timeout: int = 30) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def _read(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def read_nics() -> List[Dict[str, Any]]:
    """Network interfaces with a real PCI device behind them, plus IB devices.

    Virtual interfaces (lo, docker0, veth*) have no ``device`` symlink, so they
    drop out here without needing a name blacklist that would rot.
    """
    out: List[Dict[str, Any]] = []
    for base, kind in (("/sys/class/net", "net"), ("/sys/class/infiniband", "ib")):
        for p in sorted(glob.glob(os.path.join(base, "*"))):
            dev = os.path.join(p, "device")
            if not os.path.exists(dev):
                continue
            real = os.path.realpath(dev)
            node = (_read(os.path.join(dev, "numa_node")) or "").strip()
            entry = {
                "name": os.path.basename(p),
                "kind": kind,
                "pci": os.path.basename(real),
                # The path under /sys/devices shows which root complex it hangs
                # off, which is the same question topo -m answers for GPUs.
                "sys_path": os.path.dirname(real).replace("/sys/devices/", ""),
                "numa_node": _int_or_none(node) if node not in ("", "-1") else None,
            }
            if kind == "net":
                speed = (_read(os.path.join(p, "speed")) or "").strip()
                entry["speed_mbps"] = _int_or_none(speed)
            out.append(entry)
    return out


def probe_host(gpu_numa: Optional[Dict[Any, Optional[int]]] = None) -> Dict[str, Any]:
    """Collect everything in this module. Never raises; missing is stated."""
    host: Dict[str, Any] = {"platform": platform.system(), "unavailable": {}}

    # --- PCIe link (works on every platform) ------------------------------
    q = _sh(["nvidia-smi", "-q"], timeout=60)
    if q:
        host["pcie"] = summarize_pcie(parse_pcie_query(q))
    else:
        host["unavailable"]["pcie"] = (
            "nvidia-smi -q returned nothing. Without it the theoretical link bandwidth is "
            "unknown, so host-transfer numbers below have no denominator and are reported "
            "as absolute GB/s only."
        )

    # --- CPU / memory ------------------------------------------------------
    ls = _sh(["lscpu"])
    if ls:
        host["cpu"] = parse_lscpu(ls)
    else:
        host["cpu"] = {
            "model": platform.processor() or None,
            "logical_cpus": os.cpu_count(),
            "sockets": None,
            "cores_per_socket": None,
            "numa_nodes": None,
        }
        host["unavailable"]["cpu_detail"] = (
            "lscpu is not available on this platform. Logical CPU count comes from the standard "
            "library; socket count, core count and NUMA layout are unknown -- which means a "
            "cross-NUMA result here cannot be distinguished from a single-socket one."
        )

    mi = _read("/proc/meminfo")
    if mi:
        host["memory"] = parse_meminfo(mi)
    else:
        host["unavailable"]["memory"] = "/proc/meminfo is not available on this platform."

    # --- NIC affinity ------------------------------------------------------
    if os.path.isdir("/sys/class/net"):
        host["nic"] = match_nics_to_gpus(read_nics(), gpu_numa or {})
    else:
        host["unavailable"]["nic"] = (
            "/sys/class/net is not available on this platform, so GPU-to-NIC NUMA affinity is "
            "not measured. On a multi-node job this is the difference between 'the network is "
            "slow' and 'the job was pinned to the wrong socket'."
        )
    return host


def pcie_achieved_pct(measured_gbs: float, theoretical_gbs: Optional[float]) -> Optional[float]:
    """Measured over theoretical, as a percentage. None when there is no basis.

    Deliberately does not clamp. A result above 100% means the denominator is
    wrong -- almost always the idle-generation trap in this module's header --
    and hiding that behind a clamp would remove the only clue.
    """
    if not theoretical_gbs:
        return None
    return round(measured_gbs / theoretical_gbs * 100, 1)
