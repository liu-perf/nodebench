"""Render a run's results as one self-contained HTML file.

Design rules this file follows:

1. **Data in, template out.** Nothing here is hand-written prose about a
   specific machine. Every sentence in the output is either a fixed
   explanation of a method or a string generated from a measured value. That is
   what makes the report reproducible: run it on another node and the
   conclusions change because the numbers changed, not because someone rewrote
   the text.

2. **No chart library, no runtime.** Bar meters are CSS. The file opens from a
   USB stick on a machine with no network and renders identically with
   JavaScript disabled.

3. **Every chart has a table.** The meters exist to make magnitudes comparable
   at a glance; the tables carry the values. Colour never carries meaning
   alone -- status is always a dot plus a word.

4. **Limitations are a section, not a footnote.**
"""

from __future__ import annotations

import html as _html
import os
from typing import Any, Dict, Iterable, Optional, Sequence

_TEMPLATE = os.path.join(os.path.dirname(__file__), "templates", "report.html")

LEVEL_WORD = {
    "good": "OK",
    "ok": "OK",
    "warn": "Check",
    "warning": "Check",
    "serious": "Problem",
    "critical": "Blocker",
    "error": "Blocker",
    "muted": "n/a",
}


def esc(x: Any) -> str:
    return _html.escape("" if x is None else str(x))


def fmt(x: Any, digits: int = 2, dash: str = "--") -> str:
    if x is None or x == "":
        return dash
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:,.{digits}f}"
    return esc(x)


# --------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------

def tile(label: str, value: Any, unit: str = "", sub: str = "", digits: int = 1) -> str:
    v = fmt(value, digits) if not isinstance(value, str) else esc(value)
    u = f'<span class="unit">{esc(unit)}</span>' if unit else ""
    s = f'<div class="sub">{esc(sub)}</div>' if sub else ""
    return (
        f'<div class="tile"><div class="label">{esc(label)}</div>'
        f'<div class="value">{v}{u}</div>{s}</div>'
    )


def tiles(items: Sequence[str]) -> str:
    return '<div class="tiles">' + "".join(items) + "</div>"


def pill(level: str, text: str = "") -> str:
    lv = level if level in LEVEL_WORD else "muted"
    cls = {"ok": "good", "error": "critical", "warn": "warning"}.get(lv, lv)
    word = text or LEVEL_WORD.get(lv, lv)
    return f'<span class="pill {cls}"><span class="dot"></span>{esc(word)}</span>'


def meter(
    label: str,
    value: Optional[float],
    vmax: float,
    display: Optional[str] = None,
    level: str = "",
    ref_frac: Optional[float] = None,
) -> str:
    """One horizontal bar. ``ref_frac`` draws a reference line (e.g. ideal = 1.0)."""
    if value is None or vmax <= 0:
        pct = 0.0
    else:
        pct = max(0.0, min(100.0, value / vmax * 100.0))
    cls = f" {level}" if level in ("good", "warning", "serious", "critical", "s2") else ""
    ref = ""
    if ref_frac is not None:
        ref = f'<span class="mref" style="left:{max(0.0, min(100.0, ref_frac * 100)):.2f}%"></span>'
    shown = display if display is not None else fmt(value, 1)
    return (
        f'<div class="meter"><span class="mlabel">{esc(label)}</span>'
        f'<span class="mtrack"><span class="mfill{cls}" style="width:{pct:.2f}%"></span>{ref}</span>'
        f'<span class="mval">{esc(shown)}</span></div>'
    )


def meter_block(head: Sequence[str], rows: Iterable[str]) -> str:
    h = "".join(f"<span>{esc(x)}</span>" for x in head)
    return f'<div class="meterhead">{h}</div><div class="meters">' + "".join(rows) + "</div>"


def legend(entries: Sequence[tuple]) -> str:
    ks = "".join(
        f'<span class="k"><span class="sw" style="background:var(--{var})"></span>{esc(name)}</span>'
        for name, var in entries
    )
    return f'<div class="legend">{ks}</div>'


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], numeric: Sequence[int] = ()) -> str:
    num = set(numeric)
    th = "".join(
        f'<th class="num">{esc(h)}</th>' if i in num else f"<th>{esc(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            cls = ' class="num"' if i in num else ""
            cell = c if isinstance(c, str) and c.startswith("<span") else esc(fmt(c))
            tds.append(f"<td{cls}>{cell}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def callout(level: str, text: str, strong: str = "") -> str:
    lv = level if level in ("good", "warning", "serious", "critical") else ""
    s = f"<strong>{esc(strong)}</strong> " if strong else ""
    return f'<div class="callout {lv}">{s}{esc(text)}</div>'


def section(title: str, lede: str, *chunks: str) -> str:
    inner = "".join(c for c in chunks if c)
    lede_html = f'<p class="lede">{esc(lede)}</p>' if lede else ""
    return f"<section><h2>{esc(title)}</h2>{lede_html}{inner}</section>"


def bullets(items: Iterable[str]) -> str:
    li = "".join(f"<li>{esc(x)}</li>" for x in items if x)
    return f'<ul class="plain">{li}</ul>' if li else ""


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def _sec_headline(res: Dict[str, Any]) -> str:
    flops = res.get("flops") or {}
    membw = res.get("membw") or {}
    pcie = res.get("pcie") or {}
    nccl = res.get("nccl") or {}
    p2p = res.get("p2p") or {}

    items = []
    if flops.get("peak_tflops"):
        items.append(tile("Peak GEMM", flops["peak_tflops"], " TFLOP/s",
                          f"{flops.get('peak_dtype', 'best dtype')}, achieved not peak", 1))
    if membw.get("triad_gbs"):
        items.append(tile("Memory (triad)", membw["triad_gbs"], " GB/s",
                          f"{membw.get('pct_of_spec_str', 'streaming, large buffers')}", 0))
    if pcie.get("h2d_gbs"):
        items.append(tile("Host to device", pcie["h2d_gbs"], " GB/s", "pinned, per GPU", 1))
    if nccl.get("peak_busbw_gbs"):
        items.append(tile("All-reduce busbw", nccl["peak_busbw_gbs"], " GB/s",
                          f"{nccl.get('world_size', '?')} GPUs, bus-corrected", 1))
    if p2p.get("mode"):
        items.append(tile("GPU-to-GPU path", p2p["mode"].replace("_", " "), "",
                          f"confidence: {p2p.get('confidence', 'unknown')}"))
    if not items:
        return ""
    return section(
        "Headline",
        "Single numbers, stated with the condition that produced them. None of these is a "
        "datasheet figure; each is what this node did on this configuration.",
        tiles(items),
    )


def _sec_control(base: Dict[str, Any]) -> str:
    if not base:
        return ""
    lvl = "good" if base.get("clean") else "serious"
    rows = []
    for gid, s in (base.get("per_gpu") or {}).items():
        rows.append([f"GPU {gid}", s.get("power_w"), s.get("busy_pct"),
                     s.get("mem_used_mib"), s.get("pcie_floor_gbs")])
    body = table(
        ["Device", "Idle power (W)", "Busy (%)", "Memory held (MiB)", "PCIe floor (GB/s)"],
        rows, numeric=(1, 2, 3, 4),
    )
    finds = "".join(callout(f["level"], f["text"]) for f in base.get("findings", []))
    return section(
        "Control group",
        "Sampled before the workload with nothing running. This is the noise floor every "
        "number below is measured against -- without it, a busy node and a fast node look "
        "the same.",
        callout(lvl, base.get("headline", "")),
        body,
        finds,
        '<p class="note">"Busy" is NVML\'s utilization.gpu: the fraction of the sampling period '
        'with at least one kernel resident. It is not SM occupancy.</p>',
    )


def _sec_compat(compat: Dict[str, Any]) -> str:
    if not compat:
        return ""
    rows = []
    for f in compat.get("findings", []):
        rows.append([pill(f.get("level", "muted")), f.get("title", ""), f.get("detail", "")])
    env = compat.get("env") or {}
    gpus = env.get("gpus") or []
    gtable = table(
        ["Device", "Name", "Compute cap", "Driver", "VRAM (MiB)"],
        [[g.get("index"), g.get("name"), g.get("compute_cap"), g.get("driver"), g.get("memory_mib")]
         for g in gpus],
        numeric=(4,),
    )
    torch_info = env.get("torch") or {}
    tinfo = table(
        ["torch", "built for CUDA", "NCCL", "compiled architectures"],
        [[torch_info.get("version"), torch_info.get("cuda_built"), torch_info.get("nccl_version"),
          ", ".join(torch_info.get("arch_list") or []) or "--"]],
    )
    return section(
        "Environment and compatibility",
        "Three things must line up: the GPU's compute capability, the CUDA version, and "
        "whether the installed torch build actually contains code for that architecture. "
        "The third is the one people forget -- a CUDA 12.8 wheel can still ship without "
        "an sm_120 cubin, and the failure looks like a mysterious kernel error at runtime.",
        gtable, tinfo,
        table(["", "Subject", "Finding"], rows) if rows else callout("good", "No compatibility problems found."),
    )


def _sec_topology(topo: Dict[str, Any]) -> str:
    if not topo:
        return ""
    matrix = topo.get("matrix") or {}
    devs = topo.get("devices") or []
    if devs and matrix:
        head = [""] + [f"GPU{d}" for d in devs]
        rows = [[f"GPU{a}"] + [matrix.get(a, {}).get(b, "--") for b in devs] for a in devs]
        mt = table(head, rows)
    else:
        mt = ""
    groups = topo.get("numa_groups") or {}
    gt = table(
        ["NUMA group", "GPUs"],
        [[k, ", ".join(str(x) for x in v)] for k, v in groups.items()],
    ) if groups else ""
    return section(
        "Topology",
        "Which GPUs are close to each other. This is the variable behind every multi-GPU "
        "result: the same four cards chosen two different ways can differ by more than the "
        "difference between two GPU models.",
        mt, gt,
        '<p class="note">Link ranking, best to worst: NV# (NVLink) &gt; PIX &gt; PXB &gt; PHB '
        '&gt; NODE &gt; SYS. A SYS hop means the traffic crosses the CPU interconnect.</p>',
    )


def _sec_host(h: Dict[str, Any]) -> str:
    """CPU, memory, PCIe link and NIC placement.

    None of this is a measurement. It is the set of denominators the measurements
    are divided by, which is why it gets its own section instead of being folded
    into a footnote: a reader who disagrees with an achieved-percentage figure
    needs to be able to see what it was computed against.
    """
    if not h:
        return ""
    cpu = h.get("cpu") or {}
    mem = h.get("memory") or {}
    pcie = h.get("pcie") or {}
    nic = h.get("nic") or {}
    unavail = h.get("unavailable") or {}

    t = []
    if cpu.get("model"):
        t.append(tile("CPU", cpu["model"]))
    if cpu.get("logical_cpus"):
        socks, cps = cpu.get("sockets"), cpu.get("cores_per_socket")
        sub = f"{socks} socket x {cps} cores" if socks and cps else ""
        t.append(tile("Logical CPUs", cpu["logical_cpus"], sub=sub))
    if cpu.get("numa_nodes"):
        t.append(tile("NUMA nodes", cpu["numa_nodes"]))
    if mem.get("total_gib"):
        t.append(tile("Host memory", mem["total_gib"], " GiB",
                      sub=f"{fmt(mem.get('available_gib'), 1)} GiB available", digits=1))

    per = pcie.get("per_gpu") or {}
    link_rows = []
    for gid, d in sorted(per.items(), key=lambda kv: int(kv[0])):
        gen_note = ""
        if d.get("gen_downtrained_at_idle"):
            gen_note = " (idle)"
        link_rows.append([
            f"GPU {gid}",
            f"Gen{fmt(d.get('gen_max'))}",
            f"Gen{fmt(d.get('gen_current'))}{gen_note}",
            f"x{fmt(d.get('width_max'))}",
            f"x{fmt(d.get('width_current'))}",
            d.get("theoretical_gbs"),
        ])
    link_tbl = table(
        ["GPU", "Gen max", "Gen current", "Width max", "Width current",
         "Theoretical (GB/s)"],
        link_rows, numeric=(5,),
    ) if link_rows else ""

    nic_rows = [
        [n.get("name"), n.get("kind"), n.get("speed_mbps"), n.get("numa_node"),
         ", ".join(f"GPU{g}" for g in n.get("local_gpus") or []) or "none"]
        for n in nic.get("nics") or []
    ]
    nic_tbl = table(["Interface", "Kind", "Speed (Mbps)", "NUMA node", "GPUs on that node"],
                    nic_rows, numeric=(2, 3)) if nic_rows else ""
    nic_verdict = ""
    if nic.get("verdict"):
        lvl = "good" if "every NIC" in nic["verdict"] else "warning"
        nic_verdict = callout(lvl, nic["verdict"], "NIC affinity:")

    notes = bullets(pcie.get("notes") or [])
    missing = bullets(unavail.values()) if unavail else ""
    return section(
        "Host: CPU, memory, PCIe link, NIC placement",
        "Not measurements -- denominators. The link row below is what the host-transfer "
        "figures are divided by, and the NIC rows are what a multi-node result would be "
        "explained by.",
        tiles(t) if t else "",
        link_tbl,
        '<p class="note">The theoretical figure is built from the <em>maximum</em> generation '
        'and the <em>current</em> width, and the mismatch is deliberate. A link downtrains to '
        'Gen1 when idle and retrains under load, so the current generation understates it by up '
        'to 16x; width does not downtrain, so a current width below the maximum is a permanent '
        'property of the slot.</p>' if link_rows else "",
        notes,
        nic_tbl,
        nic_verdict,
        ('<p class="note">Not collected on this host:</p>' + missing) if missing else "",
    )


def _sec_p2p(v: Dict[str, Any]) -> str:
    if not v:
        return ""
    lvl = {"hardware": "good", "software": "good", "host_bounce": "serious",
           "disabled": "warning", "inconclusive": "warning"}.get(v.get("mode"), "muted")
    ev = v.get("evidence") or {}
    rows = [[k.replace("_", " "), ev[k]] for k in ev]
    return section(
        "GPU-to-GPU path",
        "Whether traffic between two GPUs takes a direct peer path or bounces through host "
        "memory. Every standard probe -- nvidia-smi topo -p2p, p2pBandwidthLatencyTest, "
        "nvbandwidth D2D, cudaDeviceCanAccessPeer -- answers only whether the HARDWARE "
        "supports it. A platform that routes peer traffic in the driver is invisible to all "
        "four, so the verdict below is inferred from achieved bandwidth instead.",
        callout(lvl, v.get("headline", ""), f"{str(v.get('mode', '')).replace('_', ' ')}:"),
        f'<p class="note">{esc(v.get("explanation", ""))}</p>',
        table(["Evidence", "Value"], rows, numeric=(1,)) if rows else "",
        '<p class="note">Discriminator: all-reduce busbw divided by host-to-device bandwidth. '
        'At or above 0.55 the traffic is taking a peer path; at or below 0.30 it is bouncing '
        'through the host. Between the two the answer is reported as inconclusive rather than '
        'guessed.</p>',
    )


def _sec_flops(f: Dict[str, Any]) -> str:
    rows = f.get("rows") or []
    if not rows:
        return ""
    vmax = max((r.get("tflops") or 0) for r in rows) or 1
    ms = [meter(f"{r.get('dtype')} n={r.get('n')}", r.get("tflops"), vmax,
                display=f"{fmt(r.get('tflops'), 1)} TF") for r in rows]
    return section(
        "Compute",
        "Square GEMM throughput, timed with CUDA events. This is achieved cuBLAS throughput "
        "on one shape -- it is not the silicon peak, and real models spend much of their time "
        "on shapes that do worse.",
        meter_block(["Configuration", "TFLOP/s", ""], ms),
        table(
            ["dtype", "n", "TFLOP/s (best)", "TFLOP/s (median)", "CV %", "reps"],
            [[r.get("dtype"), r.get("n"), r.get("tflops"), r.get("tflops_median"),
              r.get("cv_pct"), r.get("reps")] for r in rows],
            numeric=(1, 2, 3, 4, 5),
        ),
        '<p class="note">Best and median are both reported so a single lucky repetition cannot '
        'be quoted as the result. A high CV means the card was not in a steady state.</p>',
    )


def _sec_membw(m: Dict[str, Any]) -> str:
    rows = m.get("rows") or []
    if not rows:
        return ""
    vmax = max((r.get("gbs") or 0) for r in rows) or 1
    ms = [meter(r.get("kernel", ""), r.get("gbs"), vmax, display=f"{fmt(r.get('gbs'), 0)} GB/s")
          for r in rows]
    return section(
        "Memory bandwidth",
        "The four BabelStream kernels, implemented in PyTorch. Buffers are sized past L2 on "
        "purpose, so this measures HBM/GDDR streaming and not cache.",
        meter_block(["Kernel", "GB/s", ""], ms),
        table(["Kernel", "Arrays touched", "GB/s", "Bytes moved", "reps"],
              [[r.get("kernel"), r.get("arrays"), r.get("gbs"), r.get("bytes"), r.get("reps")]
               for r in rows], numeric=(1, 2, 3, 4)),
        '<p class="note">GB/s here is decimal (1e9 bytes/s), matching how vendors quote memory '
        'bandwidth. Using GiB/s would understate the number by 7% against the datasheet.</p>',
    )


def _sec_pcie(p: Dict[str, Any]) -> str:
    per = p.get("per_gpu") or []
    if not per:
        return ""
    measured_max = max(max(r.get("h2d_gbs") or 0, r.get("d2h_gbs") or 0) for r in per) or 1
    # Scale the bars to the link when one is known, so their length reads as a
    # fraction of what the slot can carry rather than as a fraction of the best
    # bar in the chart -- which always looks full regardless of how bad it is.
    vmax = max(measured_max, p.get("link_theoretical_max_gbs")
               or p.get("link_theoretical_gbs") or 0) or 1
    ms = []
    for r in per:
        ms.append(meter(f"GPU {r.get('gpu')} H2D", r.get("h2d_gbs"), vmax,
                        display=f"{fmt(r.get('h2d_gbs'), 1)} GB/s"))
        ms.append(meter(f"GPU {r.get('gpu')} D2H", r.get("d2h_gbs"), vmax,
                        display=f"{fmt(r.get('d2h_gbs'), 1)} GB/s", level="s2"))
    # The link ceiling, when the host probe found one. Drawn as a reference line
    # on every bar rather than as a separate table column, because "40% of the
    # link" is the reading that matters and a column of percentages next to a
    # column of GB/s invites comparing the wrong pair.
    link = p.get("link_theoretical_gbs")
    ceil_html = ""
    if link:
        best = max((r.get("h2d_gbs_pct_of_link") or 0) for r in per) or None
        lvl = "good" if (best or 0) >= 70 else "warning" if (best or 0) >= 45 else "serious"
        over = (best or 0) > 105
        lo, hi = link, p.get("link_theoretical_max_gbs") or link
        uniform = lo == hi
        # The basis string belongs to the slowest card. Printing it next to a
        # percentage that came from the fastest one would name the wrong link, so
        # it only appears when every card shares a link and there is no wrong one.
        span = (
            f"{fmt(lo, 1)} GB/s -- {p.get('link_basis', '')}".strip(" -")
            if uniform else f"link ceilings on this node: {fmt(lo, 1)}-{fmt(hi, 1)} GB/s"
        )
        ceil_html = callout(
            "warning" if over else lvl,
            (
                f"Best host-to-device reading is {fmt(best, 0)}% of what its own card's link can "
                f"carry ({span})."
                + (
                    " A figure above 100% means the denominator is wrong, not that the hardware "
                    "beat its specification -- check the generation and width the host probe read."
                    if over else ""
                )
            ).strip(),
            "Against the link:",
        )
        if p.get("link_note"):
            ceil_html += f'<p class="note">{esc(p["link_note"])}</p>'

    conc = p.get("concurrent") or {}
    ret = conc.get("retention_pct")
    ret_lvl = "good" if (ret or 0) >= 85 else "warning" if (ret or 0) >= 65 else "serious"
    conc_html = ""
    if ret is not None:
        conc_html = (
            legend([("Aggregate when all GPUs transfer at once", "series-1")])
            + meter_block(
                ["Concurrent", "% of isolated sum", ""],
                [meter("All GPUs at once", ret, 100, display=f"{fmt(ret, 0)}%", level=ret_lvl,
                       ref_frac=1.0)],
            )
            + callout(
                ret_lvl,
                f"Running every GPU at once retains {fmt(ret, 0)}% of the sum of their isolated "
                f"rates ({fmt(conc.get('aggregate_gbs'), 1)} of "
                f"{fmt(conc.get('sum_isolated_gbs'), 1)} GB/s). Loss here comes from host memory "
                "bandwidth and PCIe root-complex contention, not from the links.",
            )
        )
    return section(
        "PCIe / host transfer",
        "Pinned host buffers, timed with CUDA events. Both details matter: pageable memory "
        "forces a staging copy that can make a Gen5 link measure like Gen2, and wall-clock "
        "timing around an async copy measures the launch, not the transfer.",
        legend([("Host to device", "series-1"), ("Device to host", "series-2")]),
        meter_block(["Direction", "GB/s", ""], ms),
        table(
            ["GPU", "H2D (GB/s)", "D2H (GB/s)", "Buffer (MiB)"]
            + (["H2D % of link", "D2H % of link"] if link else []),
            [
                [r.get("gpu"), r.get("h2d_gbs"), r.get("d2h_gbs"), r.get("buffer_mib")]
                + ([r.get("h2d_gbs_pct_of_link"), r.get("d2h_gbs_pct_of_link")] if link else [])
                for r in per
            ],
            numeric=(1, 2, 3, 4, 5),
        ),
        ceil_html,
        conc_html,
        '<p class="note">The concurrent number is the one that reveals under-populated memory '
        'channels. No single-card test can show it, and it is what limits multi-GPU data '
        'loading in practice.</p>',
    )


def _sec_nccl(n: Dict[str, Any]) -> str:
    sets = n.get("sets") or []
    if not sets:
        return ""
    rows = []
    for s in sets:
        for r in s.get("rows", []):
            rows.append([s.get("label"), r.get("collective"), r.get("size_mib"),
                         r.get("algbw_gbs"), r.get("busbw_gbs"), r.get("time_ms")])
    peaks = [(s.get("label"), s.get("peak_busbw_gbs")) for s in sets if s.get("peak_busbw_gbs")]
    vmax = max((v for _, v in peaks), default=1) or 1
    ms = [meter(lbl, v, vmax, display=f"{fmt(v, 1)} GB/s") for lbl, v in peaks]
    return section(
        "Collectives",
        "torch.distributed over NCCL, one process per GPU, CUDA-event timed with a barrier "
        "between iterations. Bandwidth is reported bus-corrected (busbw), because the raw "
        "bytes/second figure is not comparable across collectives or GPU counts.",
        meter_block(["GPU set", "Peak all-reduce busbw", ""], ms),
        table(["GPU set", "Collective", "Size (MiB)", "algbw (GB/s)", "busbw (GB/s)", "time (ms)"],
              rows, numeric=(2, 3, 4, 5)),
        '<p class="note">busbw = algbw x factor, where the factor is 2(N-1)/N for all-reduce, '
        '(N-1)/N for all-gather, reduce-scatter and all-to-all, and 1 for broadcast and reduce. '
        'Quoting algbw across different GPU counts compares two different quantities.</p>',
    )


def _sec_scaling(s: Dict[str, Any]) -> str:
    if not s or not s.get("ok"):
        return ""
    rows = s.get("rows") or []
    ms = [
        meter(r["label"], r["efficiency_pct"], 100,
              display=f"{fmt(r['efficiency_pct'], 0)}%", level=r["verdict"], ref_frac=1.0)
        for r in rows
    ]
    metric = s.get("metric", "throughput")
    return section(
        "Scaling",
        f"Speedup and efficiency relative to the measured {s['baseline']['label']} baseline on "
        "this same node -- never against a datasheet, which would fold the node's own "
        "single-card behaviour into the scaling figure.",
        meter_block(["Configuration", "Efficiency vs ideal", ""], ms),
        table(
            ["Configuration", "GPUs", metric, "Speedup", "Efficiency", "Step time inflation", ""],
            [[r["label"], r["gpus"], r.get(metric), r["speedup"],
              f"{fmt(r['efficiency_pct'], 1)}%",
              (f"+{fmt(r['step_inflation_pct'], 1)}%" if r.get("step_inflation_pct") is not None else "--"),
              pill(r["verdict"])] for r in rows],
            numeric=(1, 2, 3, 4, 5),
        ),
        "".join(callout("warning", nte) for nte in s.get("notes", [])),
        f'<p class="note">{esc(s.get("caveat", ""))}</p>',
    )


def _sec_crosscheck(c: Dict[str, Any]) -> str:
    if not c or not c.get("rows"):
        return ""
    lvl = {"agree": "good", "differ": "warning", "conflict": "serious", "unavailable": "muted"}
    rows = [
        [r["quantity"], r.get("torch_value"), r.get("native_value"),
         (f"{fmt(r['divergence_pct'], 1)}%" if r.get("divergence_pct") is not None else "--"),
         pill(lvl.get(r["status"], "muted"), r["status"])]
        for r in c["rows"]
    ]
    return section(
        "Cross-check: two independent implementations",
        "The same physical quantity measured two different ways -- PyTorch on one side, the "
        "native C++ toolchain (nccl-tests, nvbandwidth, BabelStream, CUTLASS) on the other. "
        "Agreement does not prove either is right, but disagreement proves one is wrong, and "
        "that is worth more than another decimal place.",
        callout("good" if not c["counts"].get("conflict") else "serious", c["headline"]),
        table(["Quantity", "PyTorch", "Native", "Divergence", "Verdict"], rows, numeric=(1, 2, 3)),
        "".join(
            f'<p class="note"><strong>{esc(r["quantity"])}</strong>: {esc(r.get("note", ""))} '
            f'{esc(r.get("reason", ""))}</p>'
            for r in c["rows"] if r.get("note") or r.get("reason")
        ),
    )


def _sec_caveats(cv: Dict[str, Any]) -> str:
    if not cv:
        return ""
    chunks = ["".join(callout("warning", t) for t in cv.get("conditional", []))]
    cov = cv.get("covered_with_limits") or []
    if cov:
        chunks.append("<h3>What was measured, and what those measurements cannot tell you</h3>")
        chunks.append(table(
            ["Module", "Structural limits"],
            [[c["module"], " / ".join(c["limits"])] for c in cov],
        ))
    nr = cv.get("not_run") or []
    if nr:
        chunks.append("<h3>Not run in this session</h3>")
        chunks.append(table(["Module", "Why", "Consequence"],
                            [[x["module"], x["reason"], x["consequence"]] for x in nr]))
    oos = cv.get("out_of_scope") or []
    if oos:
        chunks.append("<h3>Outside this tool entirely</h3>")
        chunks.append(bullets(x["text"] for x in oos))
    return section(
        "What this run does NOT tell you",
        cv.get("principle", ""),
        *chunks,
    )


def _sec_manifest(m: Dict[str, Any]) -> str:
    if not m:
        return ""
    files = m.get("files") or []
    rows = [[f["path"], f.get("bytes"), f.get("lines"),
             f'<span class="mono">{esc(f.get("md5", "")[:16])}</span>' if f.get("md5") else "--",
             f.get("produced_by", "")] for f in files]
    return section(
        "Raw artifacts",
        f"{m.get('n_files', 0)} files, {fmt(round((m.get('total_bytes', 0)) / 1e6, 2))} MB. Every "
        "number in this report was derived from one of these, and every hash lets a reader "
        "confirm the file has not changed since.",
        table(["File", "Bytes", "Lines", "md5 (first 16)", "Produced by"], rows, numeric=(1, 2)),
        "".join(callout("warning", n) for n in m.get("notes", [])),
    )


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def render_report(data: Dict[str, Any], theme: str = "light") -> str:
    """Build the whole HTML document from a run's result dict."""
    node = data.get("node") or {}
    meta = data.get("meta") or {}
    title = f"Node benchmark: {node.get('label') or 'unnamed node'}"
    ngpu = len((data.get("compat", {}).get("env", {}) or {}).get("gpus") or [])
    subtitle = " / ".join(
        x for x in [
            meta.get("started_at", ""),
            f"{ngpu} GPU" + ("s" if ngpu != 1 else "") if ngpu else "",
            f"nodebench {meta.get('version', '')}".strip(),
            node.get("notes", ""),
        ] if x
    )

    body = "".join([
        _sec_headline(data),
        _sec_control(data.get("baseline") or {}),
        _sec_p2p(data.get("p2p") or {}),
        _sec_scaling(data.get("scaling") or {}),
        _sec_nccl(data.get("nccl") or {}),
        _sec_pcie(data.get("pcie") or {}),
        _sec_flops(data.get("flops") or {}),
        _sec_membw(data.get("membw") or {}),
        _sec_crosscheck(data.get("crosscheck") or {}),
        _sec_topology(data.get("topology") or {}),
        _sec_host(data.get("host") or {}),
        _sec_compat(data.get("compat") or {}),
        _sec_caveats(data.get("caveats") or {}),
        _sec_manifest(data.get("manifest") or {}),
    ])

    with open(_TEMPLATE, "r", encoding="utf-8") as f:
        tpl = f.read()
    footer = (
        "Generated by nodebench. Every figure above is a measurement of this node under the "
        "stated conditions; none is a vendor specification."
    )
    return (
        tpl.replace("__THEME__", theme)
        .replace("__TITLE__", esc(title))
        .replace("__SUBTITLE__", esc(subtitle))
        .replace("__BODY__", body)
        .replace("__FOOTER__", esc(footer))
    )


def write_report(data: Dict[str, Any], path: str, theme: str = "light") -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_report(data, theme))
    return path
