"""Parse `nvidia-smi topo -m` into something a benchmark can plan against.

Why this matters: on a dual-socket box the GPUs are split across two PCIe root
complexes. Four cards inside one NUMA node and four cards spanning two NUMA
nodes are physically different experiments, and if you only ever test
"4 GPUs" you will never see the difference. This module derives both sets
automatically so the scaling matrix is a controlled experiment rather than
whatever GPU ids happened to be free.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

# Ranked best -> worst. Used to summarise "what is the worst hop in this set".
LINK_RANK = ["NV18", "NV12", "NV8", "NV6", "NV4", "NV2", "NV1", "PIX", "PXB", "PHB", "NODE", "SYS"]

LINK_MEANING = {
    "X": "self",
    "PIX": "same PCIe switch (single hop)",
    "PXB": "multiple PCIe switches, does not cross the root complex",
    "PHB": "traverses the PCIe host bridge",
    "NODE": "same NUMA node, different host bridge",
    "SYS": "crosses the CPU interconnect (QPI/UPI) -- worst case",
}


def _sh(cmd: List[str], timeout: int = 60) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        # A non-zero exit means the option was rejected or the query failed. Its
        # stdout is then either empty or partial, and parsing partial output is
        # how a tool ends up reporting a topology that does not exist.
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def parse_topo_matrix(text: str) -> Dict[str, Any]:
    """Pure function. Parse the `nvidia-smi topo -m` table.

    Returns {"gpus": [...], "matrix": {i: {j: link}}, "numa": {i: node},
             "cpu_affinity": {i: "0-31"}}
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*GPU0\b", ln) or re.search(r"\bGPU0\b.*\bGPU1\b", ln):
            header_idx = i
            break
    if header_idx is None:
        return {"gpus": [], "matrix": {}, "numa": {}, "cpu_affinity": {}}

    header = lines[header_idx].split()
    col_names = [c for c in header if re.match(r"^(GPU|NIC|mlx)", c)]
    has_numa = "NUMA" in lines[header_idx]
    # No NUMA column index: nvidia-smi pads the affinity column with ranges and
    # dashes, so the NUMA node is found by scanning the tail for the first bare
    # integer instead. That survives the column shifting, which it does when NIC
    # rows are present.
    aff_col = None
    for j, c in enumerate(header):
        if c.startswith("CPU"):
            aff_col = j

    matrix: Dict[int, Dict[int, str]] = {}
    numa: Dict[int, Optional[int]] = {}
    affinity: Dict[int, str] = {}
    gpus: List[int] = []

    for ln in lines[header_idx + 1 :]:
        m = re.match(r"^\s*GPU(\d+)\s+(.*)$", ln)
        if not m:
            if re.match(r"^\s*(Legend|NIC\d)", ln):
                break
            continue
        gid = int(m.group(1))
        cells = m.group(2).split()
        gpus.append(gid)
        row: Dict[int, str] = {}
        for j, name in enumerate(col_names):
            if not name.startswith("GPU") or j >= len(cells):
                continue
            try:
                peer = int(name[3:])
            except ValueError:
                continue
            row[peer] = cells[j]
        matrix[gid] = row
        # NUMA / affinity columns come after the GPU/NIC columns
        tail_start = len(col_names)
        tail = cells[tail_start:]
        if aff_col is not None and tail:
            affinity[gid] = tail[0]
        if has_numa and tail:
            for cell in tail:
                if re.fullmatch(r"\d+", cell):
                    numa[gid] = int(cell)
                    break
            else:
                numa[gid] = None
        else:
            numa[gid] = None

    return {"gpus": sorted(gpus), "matrix": matrix, "numa": numa, "cpu_affinity": affinity}


def _numa_groups_from_matrix(topo: Dict[str, Any]) -> List[List[int]]:
    """Group GPUs that are NOT separated by a SYS hop.

    Prefer the NUMA column when nvidia-smi reports it. Fall back to connectivity:
    two GPUs whose link is SYS are on different root complexes.
    """
    gpus = topo["gpus"]
    numa = topo.get("numa") or {}
    if gpus and all(numa.get(g) is not None for g in gpus):
        buckets: Dict[int, List[int]] = {}
        for g in gpus:
            buckets.setdefault(numa[g], []).append(g)
        return [sorted(v) for _, v in sorted(buckets.items())]

    matrix = topo["matrix"]
    groups: List[List[int]] = []
    for g in gpus:
        placed = False
        for grp in groups:
            if all(matrix.get(g, {}).get(o, "SYS") != "SYS" for o in grp):
                grp.append(g)
                placed = True
                break
        if not placed:
            groups.append([g])
    return [sorted(g) for g in groups]


def _gpu_indices() -> List[int]:
    """Device list from the query interface, which works everywhere."""
    out = _sh(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]) or ""
    idx = []
    for ln in out.splitlines():
        ln = ln.strip().rstrip(",")
        if ln.isdigit():
            idx.append(int(ln))
    return sorted(idx)


def probe_topology(numa_override: Optional[List[List[int]]] = None) -> Dict[str, Any]:
    raw = _sh(["nvidia-smi", "topo", "-m"]) or ""
    topo = parse_topo_matrix(raw)
    topo["raw"] = raw
    topo["matrix_available"] = bool(topo["gpus"])

    # `topo -m` is Linux-only -- on Windows the option does not exist, and inside
    # some containers it is filtered out. Without a fallback the whole run would
    # then benchmark zero GPUs, which is a far worse outcome than losing the link
    # detail. So fall back to the device list and say so, rather than reporting an
    # empty topology as if the machine had no cards.
    if not topo["gpus"]:
        found = _gpu_indices()
        if found:
            topo["gpus"] = found
            topo["numa"] = {g: None for g in found}
            topo["note"] = (
                "nvidia-smi topo -m is unavailable on this platform, so link types and NUMA "
                "placement are unknown. GPU count comes from the query interface. Treat every "
                "inter-GPU link as unspecified: no same-NUMA vs cross-NUMA comparison is "
                "possible here."
            )

    if numa_override:
        topo["numa_groups"] = [sorted(g) for g in numa_override]
        topo["numa_source"] = "config"
    else:
        topo["numa_groups"] = _numa_groups_from_matrix(topo)
        topo["numa_source"] = "auto"
    topo["is_multi_root"] = len(topo["numa_groups"]) > 1
    topo["has_nvlink"] = any(
        v.startswith("NV") for row in topo["matrix"].values() for v in row.values()
    )
    return topo


def worst_link(topo: Dict[str, Any], gpu_set: List[int]) -> str:
    """The worst hop inside a GPU set -- i.e. what will bound its collectives."""
    matrix = topo.get("matrix", {})
    worst = "X"
    worst_rank = -1
    for a in gpu_set:
        for b in gpu_set:
            if a == b:
                continue
            link = matrix.get(a, {}).get(b, "SYS")
            r = LINK_RANK.index(link) if link in LINK_RANK else len(LINK_RANK)
            if r > worst_rank:
                worst_rank, worst = r, link
    return worst


def suggest_gpu_sets(topo: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the scaling matrix: 1 / 2 / 4-same / 4-cross / all.

    The 4-same vs 4-cross pair is the whole point. Same card count, same
    workload, only the topology differs -- that is a controlled variable.
    """
    gpus = topo["gpus"]
    groups = topo["numa_groups"]
    n = len(gpus)
    sets: List[Dict[str, Any]] = []

    def add(name: str, ids: List[int], note: str = "") -> None:
        if not ids or len(ids) > n:
            return
        if any(s["gpus"] == ids for s in sets):
            return
        sets.append(
            {"name": name, "gpus": ids, "n": len(ids), "note": note, "worst_link": worst_link(topo, ids)}
        )

    if n >= 1:
        add("1gpu", gpus[:1], "single-card baseline")
    if n >= 2:
        add("2gpu", gpus[:2])

    if n >= 4:
        big = max(groups, key=len) if groups else gpus
        if len(big) >= 4:
            add("4gpu_same_numa", sorted(big[:4]), "all four inside one NUMA node")
        else:
            add("4gpu", gpus[:4])
        if len(groups) > 1:
            a, b = groups[0], groups[1]
            take_a = a[: min(2, len(a))]
            take_b = b[: min(4 - len(take_a), len(b))]
            cross = sorted(take_a + take_b)
            if len(cross) == 4:
                add("4gpu_cross_numa", cross, "split across two NUMA nodes -- controlled contrast")

    if n >= 2:
        add(f"{n}gpu_all", gpus, "full node")
    return sets


def summarize(topo: Dict[str, Any]) -> str:
    parts = [f"{len(topo['gpus'])} GPUs"]
    # Without the matrix, "no NVLink" and "single root" are assumptions, not
    # observations, and printing them as observations is exactly the kind of
    # quiet overstatement this tool exists to avoid.
    if not topo.get("matrix_available", True):
        parts.append("link types and NUMA placement unknown (nvidia-smi topo -m unavailable)")
        return ", ".join(parts)
    if topo.get("has_nvlink"):
        parts.append("NVLink present")
    else:
        parts.append("no NVLink (PCIe only)")
    ng = topo.get("numa_groups") or []
    if len(ng) > 1:
        parts.append("multi-root: " + " + ".join(str(len(g)) for g in ng))
    else:
        parts.append("single root complex")
    return ", ".join(parts)
