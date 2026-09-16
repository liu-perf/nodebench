"""Parsers for the C++ toolchain: nccl-tests, nvbandwidth, BabelStream, CUTLASS.

Only needed for ``--backend native`` / ``--backend both``. Kept pure so the
cross-check layer can compare a native reading against the PyTorch reading of
the same quantity without either side knowing about the other.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def parse_nccl_tests(text: str) -> Dict[str, Any]:
    """Parse ``all_reduce_perf`` & friends.

    Data rows look like::

        # size  count  type  redop  root  time  algbw  busbw  #wrong  time algbw busbw #wrong
        1073741824 268435456  float   sum   -1  26312   40.81   71.42      0 ...

    Columns 5-8 are the out-of-place group; 9-12 repeat them in-place. We take
    the out-of-place group, which is what nccl-tests reports as the headline.
    """
    rows: List[Dict[str, Any]] = []
    avg_busbw = None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            m = re.search(r"Avg bus bandwidth\s*:\s*([\d.]+)", s)
            if m:
                avg_busbw = float(m.group(1))
            continue
        parts = s.split()
        if len(parts) < 8:
            continue
        try:
            size = int(parts[0])
            count = int(parts[1])
        except ValueError:
            continue
        try:
            rows.append(
                {
                    "size_bytes": size,
                    "count": count,
                    "dtype": parts[2],
                    "redop": parts[3],
                    "time_us": float(parts[5]),
                    "algbw_gbs": float(parts[6]),
                    "busbw_gbs": float(parts[7]),
                }
            )
        except (ValueError, IndexError):
            continue
    return {
        "rows": rows,
        "avg_busbw_gbs": avg_busbw,
        "peak_busbw_gbs": max((r["busbw_gbs"] for r in rows), default=None),
        "n_rows": len(rows),
    }


def parse_nvbandwidth(text: str) -> Dict[str, Any]:
    """Parse nvbandwidth output into per-test SUM plus per-device values.

    Also records which tests were ``Waived``. That list is not noise: on a node
    with no hardware peer path, every device-to-device test waives, and knowing
    *that* is the whole point of running it.
    """
    tests: Dict[str, Any] = {}
    waived: List[str] = []
    current = None
    for ln in text.splitlines():
        s = ln.strip()
        # nvbandwidth writes "Running <test>." with a trailing period but "SUM
        # <test> <value>" without one. Keeping the period would file the same
        # test under two keys, so the per-device values and the SUM would end up
        # in different entries and neither would look wrong.
        m = re.match(r"^Running\s+(\S+)", s)
        if m:
            current = m.group(1).rstrip(".")
            tests.setdefault(current, {"values": {}, "sum": None, "waived": False})
            continue
        if "Waived" in s and current:
            tests[current]["waived"] = True
            waived.append(current)
            continue
        m = re.match(r"^SUM\s+(\S+)\s+([\d.]+)", s)
        if m:
            name = m.group(1).rstrip(".")
            tests.setdefault(name, {"values": {}, "sum": None, "waived": False})
            tests[name]["sum"] = float(m.group(2))
            continue
        if current and re.match(r"^\d+\s+[\d.]+", s):
            parts = s.split()
            try:
                tests[current]["values"][int(parts[0])] = float(parts[1])
            except (ValueError, IndexError):
                pass
    for d in tests.values():
        vals = list(d["values"].values())
        d["mean"] = round(sum(vals) / len(vals), 2) if vals else None
        d["n_devices"] = len(vals)
    return {
        "tests": tests,
        "waived": waived,
        "n_waived": len(waived),
        "h2d_sum_gbs": (tests.get("host_to_device_memcpy_ce") or {}).get("sum"),
        "h2d_mean_gbs": (tests.get("host_to_device_memcpy_ce") or {}).get("mean"),
        "d2h_sum_gbs": (tests.get("device_to_host_memcpy_ce") or {}).get("sum"),
        "device_local_read_gbs": (tests.get("device_local_read_sm") or {}).get("mean"),
    }


def parse_cutlass_profiler(text: str) -> Dict[str, Any]:
    """Parse ``cutlass_profiler --operation=Gemm`` output.

    The verification trap
    ---------------------
    ``cutlass_profiler`` verifies results by default, and verification runs
    inside the timed region. A run with ``Verification: ON`` reports a runtime
    that includes a reference GEMM plus a comparison pass, so the GFLOPs figure
    comes out low -- typically by a factor of two to three, occasionally worse.

    Nothing about that output looks wrong. The units are right, the shape is
    right, the number is stable across repeats, and it is simply too small. The
    usual way to get there is pasting a multi-line command into a terminal that
    breaks the line, dropping ``--verification-enabled=false`` off the end.

    So the verification state is parsed out and carried with every row, and
    ``verification_on`` is set at the top level. The caller is expected to refuse
    to report those numbers as throughput. Recording them silently would put a
    wrong FLOPS figure into a cross-check against torch, where a 2-3x gap would
    read as "the two implementations disagree" rather than "one of them measured
    the wrong thing".

    Output is CSV when ``--output=<file>`` is given, and a labelled block
    otherwise. Both shapes appear in practice, so both are handled.
    """
    rows: List[Dict[str, Any]] = []
    verification_states: List[str] = []

    # --- CSV form ---------------------------------------------------------
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header_i = next(
        (i for i, ln in enumerate(lines) if "GFLOPs" in ln and "," in ln), None
    )
    if header_i is not None:
        cols = [c.strip() for c in lines[header_i].split(",")]
        want = {c: i for i, c in enumerate(cols)}
        for ln in lines[header_i + 1:]:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) != len(cols):
                continue
            row = _cutlass_row(
                op=_at(parts, want.get("Operation")),
                gflops=_at(parts, want.get("GFLOPs")),
                runtime=_at(parts, want.get("Runtime")),
                status=_at(parts, want.get("Disposition")) or _at(parts, want.get("Status")),
                m=_at(parts, want.get("m")), n=_at(parts, want.get("n")), k=_at(parts, want.get("k")),
            )
            if row:
                rows.append(row)

    # --- labelled block form ---------------------------------------------
    if not rows:
        cur: Dict[str, Any] = {}
        for ln in lines:
            s = ln.strip()
            m = re.match(r"^(Operation|Disposition|Status|GFLOPs|Runtime|Bytes|Math)\s*:\s*(.+)$", s)
            if m:
                cur[m.group(1)] = m.group(2).strip()
                continue
            m = re.match(r"^Problem\s+\d+\s*:.*\[(\w+)\]", s)
            if m or s.startswith("====="):
                if cur.get("GFLOPs"):
                    row = _cutlass_row(
                        op=cur.get("Operation"), gflops=cur.get("GFLOPs"),
                        runtime=cur.get("Runtime"),
                        status=cur.get("Disposition") or cur.get("Status"),
                    )
                    if row:
                        rows.append(row)
                cur = {}
        if cur.get("GFLOPs"):
            row = _cutlass_row(
                op=cur.get("Operation"), gflops=cur.get("GFLOPs"),
                runtime=cur.get("Runtime"),
                status=cur.get("Disposition") or cur.get("Status"),
            )
            if row:
                rows.append(row)

    for ln in text.splitlines():
        m = re.search(r"Verification\s*:\s*(\w+)", ln)
        if m:
            verification_states.append(m.group(1).strip().upper())

    verification_on = any(v in ("ON", "TRUE", "ENABLED") for v in verification_states)
    peak = max((r["gflops"] for r in rows), default=None)
    out: Dict[str, Any] = {
        "rows": rows,
        "n_rows": len(rows),
        "verification_states": verification_states,
        "verification_on": verification_on,
        "peak_gflops": peak,
        "peak_tflops": round(peak / 1000.0, 2) if peak else None,
    }
    if verification_on:
        # Blank the headline rather than leaving a plausible number that a
        # downstream chart would happily plot. The rows stay so the evidence is
        # still on disk; what goes away is the field other modules read.
        out["peak_tflops"] = None
        out["peak_gflops"] = None
        out["rejected"] = (
            "cutlass_profiler ran with verification enabled, so the reported runtime includes "
            "the reference GEMM and the comparison pass. These are not throughput numbers and "
            "are not being reported as such. Re-run with --verification-enabled=false, and "
            "paste the command as a single line -- a wrapped paste dropping that flag is the "
            "usual cause."
        )
    return out


def _at(parts: List[str], i: Optional[int]) -> Optional[str]:
    return parts[i] if i is not None and 0 <= i < len(parts) else None


def _cutlass_row(op=None, gflops=None, runtime=None, status=None,
                 m=None, n=None, k=None) -> Optional[Dict[str, Any]]:
    try:
        g = float(gflops)
    except (TypeError, ValueError):
        return None
    row: Dict[str, Any] = {"operation": op, "gflops": g, "tflops": round(g / 1000.0, 2)}
    try:
        row["runtime_ms"] = float(runtime)
    except (TypeError, ValueError):
        pass
    if status:
        row["status"] = status
    for name, v in (("m", m), ("n", n), ("k", k)):
        try:
            row[name] = int(v)
        except (TypeError, ValueError):
            pass
    return row


def parse_babelstream(text: str) -> Dict[str, Any]:
    """Parse BabelStream. Values are MBytes/sec; converted to GB/s here."""
    out: Dict[str, Any] = {}
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) < 2:
            continue
        name = parts[0].strip().lower()
        if name not in ("copy", "mul", "add", "triad", "dot"):
            continue
        try:
            out[name] = {"gbs": round(float(parts[1]) / 1000.0, 1)}
        except ValueError:
            continue
    return out
