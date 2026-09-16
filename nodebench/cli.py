"""Command line entry point.

Six commands, in the order a new user meets them::

    nodebench doctor            # 10 s   -- can this machine run anything?
    nodebench run               # 5 min  -- measure, write raw + report
    nodebench monitor -- <cmd>  #        -- watch someone else's workload
    nodebench parse-mm <log>    #        -- turn a training log into throughput
    nodebench report <dir>      #        -- rebuild HTML from a finished run
    nodebench setup --native    # 30-90m -- optional C++ toolchain

The default path deliberately needs no compiler, no dataset, no root and no
network. That is the whole design: a stranger clones the repo and has a report
before deciding whether the project is worth their time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import __version__ as VERSION

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _step(msg: str) -> None:
    print(f"  -> {msg}", flush=True)


def _write_json(obj: Any, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    from .probe.compat import check_compat, probe_environment
    from .probe.topology import probe_topology, suggest_gpu_sets, summarize

    env = probe_environment()
    findings = check_compat(env)

    _say(f"nodebench {VERSION} doctor")
    _say("")
    gpus = env.get("gpus") or []
    if not gpus:
        _say("No GPU visible. Everything below is moot.")
    for g in gpus:
        _say(f"  GPU {g.get('index')}  {g.get('name')}  "
             f"cc {g.get('compute_cap')}  driver {g.get('driver')}  {g.get('memory_mib')} MiB")

    t = env.get("torch") or {}
    if t.get("error"):
        _say(f"\n  torch: {t['error']}")
    else:
        if t.get("nccl_version"):
            nccl = f"NCCL {t['nccl_version']}"
        elif t.get("nccl_available"):
            nccl = "NCCL present, version unreadable"
        else:
            nccl = "no NCCL in this build (collectives unavailable)"
        _say(f"\n  torch {t.get('version')} built for CUDA {t.get('cuda_built')}, {nccl}")
        _say(f"  compiled architectures: {', '.join(t.get('arch_list') or []) or 'none'}")

    _say("")
    worst = "ok"
    for f in findings:
        lvl = f.get("level", "ok")
        mark = {"ok": "ok  ", "warn": "WARN", "error": "FAIL"}.get(lvl, "    ")
        _say(f"  [{mark}] {f.get('title')}")
        if f.get("detail"):
            _say(f"         {f['detail']}")
        if lvl == "error":
            worst = "error"
        elif lvl == "warn" and worst != "error":
            worst = "warn"

    if gpus:
        topo = probe_topology()
        _say("")
        _say(f"  topology: {summarize(topo)}")
        for s in suggest_gpu_sets(topo):
            note = f"  ({s['note']})" if s.get("note") else ""
            _say(f"    {s['name']:<18} gpus {s['gpus']}  worst link {s['worst_link']}{note}")

    _say("")
    if worst == "error":
        _say("Result: this machine cannot produce trustworthy numbers yet. Fix the FAIL lines.")
        return 1
    _say("Result: ready. Run `nodebench run` next.")
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def _idle_baseline(cfg, outdir: str, seconds: int) -> Dict[str, Any]:
    """The control group. Sample the node doing nothing before touching it."""
    from .analyze.baseline import assess_idle
    from .monitor.binning import per_gpu_stats
    from .monitor.nvml import NvmlMonitor

    if seconds <= 0:
        return {"skipped": "idle_baseline_s is 0 -- no control group, so no noise floor"}

    _step(f"control group: sampling {seconds}s of idle before anything runs")
    mon = NvmlMonitor(outdir, interval_s=float(cfg.get("monitor.interval_s", 1.0)),
                      filename="idle_baseline.csv")
    mon.start()
    if mon.error:
        return {"error": mon.error}
    time.sleep(seconds)
    mon.stop()
    summ = mon.summary()
    if summ.get("error"):
        return {"error": summ["error"]}

    # Judged per card, not in aggregate. Seven quiet cards and one busy one
    # averages to "mostly idle", which is the conclusion that must not be drawn.
    out = assess_idle(per_gpu_stats(summ["csv"]))
    out["monitor"] = summ
    return out


def cmd_run(args: argparse.Namespace) -> int:
    from .analyze.crosscheck import cross_check
    from .config import load_config
    from .monitor.nvml import NvmlMonitor
    from .probe.compat import check_compat, probe_environment
    from .probe.p2p import probe_p2p
    from .probe.topology import probe_topology, suggest_gpu_sets
    from .report import adapt, build_caveats, build_manifest, write_report

    overrides: Dict[str, Any] = {}
    if args.modules:
        overrides.setdefault("bench", {})["modules"] = args.modules.split(",")
    if args.backend:
        overrides.setdefault("bench", {})["backend"] = args.backend
    if args.label:
        overrides.setdefault("node", {})["label"] = args.label
    if args.out:
        overrides.setdefault("paths", {})["workdir"] = args.out
    if args.quick:
        overrides.setdefault("bench", {}).setdefault("flops", {})["iters"] = 8
        overrides["bench"].setdefault("membw", {})["iters"] = 8
        overrides["bench"].setdefault("pcie", {})["iters"] = 6
        overrides["bench"].setdefault("nccl", {})["iters"] = 6
        overrides.setdefault("monitor", {})["idle_baseline_s"] = 5

    cfg = load_config(args.config, overrides)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rundir = os.path.join(str(cfg.workdir), f"run_{stamp}")
    rawdir = os.path.join(rundir, "raw")
    os.makedirs(rawdir, exist_ok=True)

    _say(f"nodebench {VERSION}")
    _say(f"node    : {cfg.get('node.label')}")
    _say(f"output  : {rundir}")
    _say("")

    modules: List[str] = list(cfg.get("bench.modules", []))
    backend = cfg.get("bench.backend", "torch")
    skipped: Dict[str, str] = {}
    results: Dict[str, Any] = {}

    # ---- probe -----------------------------------------------------------
    _step("probing environment")
    env = probe_environment()
    findings = check_compat(env)
    _write_json({"env": env, "findings": findings}, os.path.join(rawdir, "environment.json"))
    blockers = [f for f in findings if f.get("level") == "error"]
    if blockers and not args.force:
        _say("")
        for f in blockers:
            _say(f"  FAIL {f['title']}: {f.get('detail', '')}")
        _say("\nStopping. These would produce numbers that mean nothing. "
             "Pass --force to measure anyway.")
        return 1

    ngpu = len(env.get("gpus") or [])
    topo = probe_topology(cfg.get("topology.numa_groups"))
    _write_json(topo, os.path.join(rawdir, "topology.json"))

    # Host facts are denominators, not measurements. The PCIe link in
    # particular decides whether the host-transfer number further down is 81% of
    # what the slot can do or 40% of it, and those are opposite conclusions from
    # the same reading. Probed before any workload so the link state recorded is
    # the same idle state everything else is compared against.
    from .probe.host import probe_host
    host = probe_host(gpu_numa=topo.get("numa") or {})
    _write_json(host, os.path.join(rawdir, "host.json"))

    gpu_sets = cfg.get("bench.nccl.gpu_sets")
    if gpu_sets:
        sets = [{"name": s.get("name", f"{len(s['gpus'])}gpu"), "gpus": s["gpus"],
                 "n": len(s["gpus"]), "note": s.get("note", ""), "worst_link": ""} for s in gpu_sets]
    else:
        sets = suggest_gpu_sets(topo)

    # ---- control group ---------------------------------------------------
    baseline = _idle_baseline(cfg, rawdir, int(cfg.get("monitor.idle_baseline_s", 12)))
    _write_json(baseline, os.path.join(rawdir, "idle_baseline.json"))

    # ---- workload, with monitoring running underneath --------------------
    mon: Optional[Any] = None
    if cfg.get("monitor.enabled", True):
        mon = NvmlMonitor(rawdir, interval_s=float(cfg.get("monitor.interval_s", 1.0)),
                          filename="monitor.csv")
        mon.start()
        # start() swallows a missing pynvml and reports it on the object, so a
        # monitor that never sampled would otherwise be indistinguishable from
        # one that sampled and found nothing.
        if mon.error:
            skipped["monitor"] = mon.error
            mon = None
    else:
        skipped["monitor"] = "disabled in config (monitor.enabled: false)"

    try:
        if "flops" in modules:
            from .bench import flops_torch
            _step("compute (GEMM)")
            if mon:
                mon.mark("flops")
            r = flops_torch.run(cfg, device=0)
            _write_json(r.to_dict(), os.path.join(rawdir, "flops.json"))
            results["flops"] = adapt.flops(r)

        if "membw" in modules:
            from .bench import membw_torch
            _step("memory bandwidth")
            if mon:
                mon.mark("membw")
            r = membw_torch.run(cfg, device=0)
            _write_json(r.to_dict(), os.path.join(rawdir, "membw.json"))
            results["membw"] = adapt.membw(r)

        if "pcie" in modules:
            from .bench import pcie_torch
            _step("host transfer (isolated, then all cards at once)")
            if mon:
                mon.mark("pcie")
            r = pcie_torch.run(cfg)
            _write_json(r.to_dict(), os.path.join(rawdir, "pcie.json"))
            results["pcie"] = adapt.pcie(r)

        if "nccl" in modules and ngpu >= 2:
            from .bench import nccl_torch
            _step(f"collectives across {len(sets)} GPU set(s)")
            if mon:
                mon.mark("nccl")
            r = nccl_torch.run(cfg, sets)
            _write_json(r.to_dict(), os.path.join(rawdir, "nccl.json"))
            results["nccl"] = adapt.nccl(r)
        elif "nccl" in modules:
            skipped["nccl"] = "only one GPU visible -- there is nothing to collect across"
    finally:
        if mon:
            mon.stop()
            _write_json(mon.summary(), os.path.join(rawdir, "monitor_summary.json"))

    for m in ("flops", "membw", "pcie", "nccl"):
        if m not in modules:
            skipped[m] = "not in the configured module list"

    # ---- interpretation --------------------------------------------------
    _step("classifying the GPU-to-GPU path")
    h2d = (results.get("pcie") or {}).get("h2d_gbs")
    busbw = (results.get("nccl") or {}).get("peak_busbw_gbs")
    p2p = probe_p2p(h2d_gbs=h2d, allreduce_busbw_gbs=busbw, nvlink=bool(topo.get("has_nvlink")))
    _write_json(p2p, os.path.join(rawdir, "p2p.json"))

    crosscheck = {}
    if backend in ("native", "both"):
        native = _load_native(rawdir)
        if native:
            t, n = adapt.crosscheck_inputs(results, native)
            crosscheck = cross_check(t, n)
            _write_json(crosscheck, os.path.join(rawdir, "crosscheck.json"))
        else:
            skipped["native_backend"] = (
                "no native tool output found -- run `nodebench setup --native` first"
            )

    # p2p and monitor are not bench modules -- they are not in `modules` and
    # never will be -- but they do execute, and leaving them out made the report
    # print "Nothing in this report speaks to p2p" directly above the P2P
    # section. What ran is what produced output, not what was configured.
    ran = [m for m in modules if m in results]
    ran.append("p2p")
    ran.append("host")
    if mon is not None:
        ran.append("monitor")

    # Give the host-transfer figures their denominator. Done here rather than in
    # the bench module because the bench module measures and this interprets --
    # and because a link speed read from nvidia-smi is evidence of a different
    # kind from a timed copy.
    if results.get("pcie") and (host.get("pcie") or {}).get("per_gpu"):
        _attach_pcie_ceiling(results["pcie"], host["pcie"])

    caveats = build_caveats(
        modules_run=ran,
        modules_skipped=skipped,
        findings={
            "idle_not_clean": baseline.get("clean") is False,
            "p2p_inconclusive": (p2p.get("verdict") or {}).get("mode") == "inconclusive",
            "single_gpu": ngpu < 2,
        },
    )

    data: Dict[str, Any] = {
        "node": cfg.get("node"),
        "meta": {"version": VERSION, "started_at": stamp, "config_source": cfg.source,
                 "backend": backend},
        "compat": {"env": env, "findings": findings},
        "topology": adapt.topology(topo),
        "host": host,
        "baseline": baseline,
        "p2p": (p2p.get("verdict") or {}),
        "crosscheck": crosscheck,
        "caveats": caveats,
        **results,
    }

    if cfg.get("report.manifest", True):
        data["manifest"] = build_manifest(rundir)

    _write_json(data, os.path.join(rundir, "results.json"))
    out_html = os.path.join(rundir, "report.html")
    write_report(data, out_html)

    _say("")
    _say(f"report : {out_html}")
    _say(f"raw    : {rawdir}")
    v = data["p2p"]
    if v.get("headline"):
        _say("")
        _say(f"verdict: {v['headline']}")
    return 0


def _attach_pcie_ceiling(pcie: Dict[str, Any], host_pcie: Dict[str, Any]) -> None:
    """Add theoretical GB/s and achieved % to the adapted pcie block, in place.

    Achieved percentages are deliberately not clamped to 100. A figure above
    100% means the denominator is wrong -- on an idle link that is almost always
    the generation-downtrain trap documented in probe/host.py -- and clamping
    would erase the only visible symptom.
    """
    from .probe.host import pcie_achieved_pct

    per = host_pcie.get("per_gpu") or {}
    theo = [v.get("theoretical_gbs") for v in per.values() if v.get("theoretical_gbs")]
    if not theo:
        return
    # Cards in one node can sit in different slots. The lowest ceiling is the
    # honest one to quote against an aggregate figure.
    ceiling = min(theo)
    pcie["link_theoretical_gbs"] = ceiling
    pcie["link_theoretical_max_gbs"] = max(theo)
    pcie["link_basis"] = next(
        (v.get("theoretical_basis") for v in per.values()
         if v.get("theoretical_gbs") == ceiling), None
    )

    # Per-card percentages use that card's own link, never the node-wide minimum.
    # A card in a x16 slot should not be marked down because its neighbour is in a
    # x8 one; dividing a x16 card's reading by a x8 ceiling produces something
    # above 100%, which is the report's signal for a broken denominator and would
    # be raised here by the tool itself.
    best: Dict[str, Optional[float]] = {}
    for row in pcie.get("per_gpu") or []:
        own = (per.get(str(row.get("gpu"))) or {}).get("theoretical_gbs") or ceiling
        for key in ("h2d_gbs", "d2h_gbs"):
            val = row.get(key)
            if isinstance(val, (int, float)):
                pct = pcie_achieved_pct(float(val), own)
                row[f"{key}_pct_of_link"] = pct
                if pct is not None and pct > (best.get(key) or 0):
                    best[key] = pct
    # The headline figure is a peak, not an aggregate, so it is reported as the
    # best card measured against that card's own link rather than as the peak
    # divided by the worst link in the box.
    for key, pct in best.items():
        pcie[f"{key}_pct_of_link"] = pct

    if len(set(theo)) > 1:
        pcie["link_note"] = (
            f"Cards do not share a link: ceilings range {min(theo)}-{max(theo)} GB/s. Each "
            "percentage above is against that card's own link, so the columns are comparable "
            "to each other even though the GB/s columns are not."
        )


def _load_native(rawdir: str) -> Dict[str, Any]:
    """Read whatever native tool output happens to be sitting in raw/."""
    from .parse.native import (
        parse_babelstream,
        parse_cutlass_profiler,
        parse_nccl_tests,
        parse_nvbandwidth,
    )

    out: Dict[str, Any] = {}
    pairs = [
        ("nccl_tests", "nccl_tests.txt", parse_nccl_tests),
        ("nvbandwidth", "nvbandwidth.txt", parse_nvbandwidth),
        ("babelstream", "babelstream.txt", parse_babelstream),
        # `nodebench setup` spends 30-90 minutes building CUTLASS, and until
        # this line existed nothing ever read its output -- report/adapt.py
        # looked up a `cutlass.peak_tflops` key that no parser produced, so the
        # slowest tool in the toolchain contributed nothing to any report.
        ("cutlass", "cutlass.txt", parse_cutlass_profiler),
    ]
    for key, fn, parser in pairs:
        p = os.path.join(rawdir, fn)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                out[key] = parser(f.read())
    return out


# --------------------------------------------------------------------------
# monitor
# --------------------------------------------------------------------------

def cmd_monitor(args: argparse.Namespace) -> int:
    """Watch a workload that nodebench did not launch.

    This is the command that gets used most in practice: someone else's
    training job is already running and you need to know what the hardware is
    doing while it runs.
    """
    from .config import load_config
    from .monitor.binning import bin_by_active_set, summarize_bins
    from .monitor.nvml import NvmlMonitor

    cfg = load_config(args.config)
    outdir = args.out or str(cfg.raw_dir)

    mon = NvmlMonitor(outdir, interval_s=args.interval, filename=args.name)
    mon.start()
    if mon.error:
        _say(mon.error)
        return 2
    _say(f"sampling every {args.interval}s -> {mon.path}")

    # argparse.REMAINDER keeps the "--" separator, and passing it to the OS as
    # argv[0] is not a useful error message.
    cmd = list(args.command or [])
    while cmd and cmd[0] == "--":
        cmd.pop(0)

    rc = 0
    try:
        if cmd:
            _say(f"running: {' '.join(cmd)}")
            try:
                rc = subprocess.call(cmd)
            except OSError as e:
                # Still fall through to stop() and write the CSV: the samples
                # already collected are real data and the run is reproducible
                # once the command is fixed.
                _say(f"could not start {cmd[0]!r}: {e}")
                rc = 127
        else:
            _say("no command given; sampling until Ctrl-C")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        _say("\nstopped")
    finally:
        mon.stop()

    summ = mon.summary()
    _say("")
    _say(f"rows {summ['rows']}, duration {summ['duration_s']}s, "
         f"actual interval {summ.get('actual_interval_s')}s")
    if summ.get("interval_warning"):
        _say(f"  {summ['interval_warning']}")

    bins = bin_by_active_set(summ["csv"], busy_threshold=args.busy_threshold)
    s = summarize_bins(bins)
    _write_json({"monitor": summ, "bins": bins, "summary": s},
                os.path.join(outdir, "monitor_summary.json"))

    _say("")
    _say("phases, binned by which GPUs were actually busy:")
    for p in s.get("phases", []):
        _say(f"  gpus [{p['key']:<12}] rows {p['rows']:>5}  "
             f"tx {p.get('tx_gbs_mean_active')} GB/s  rx {p.get('rx_gbs_mean_active')} GB/s  "
             f"power {p.get('power_w_total_mean')} W")
    if s.get("noise_floor"):
        nf = s["noise_floor"]
        _say(f"  idle floor: tx {nf['tx_gbs_mean']} / rx {nf['rx_gbs_mean']} GB/s")
    if s.get("signal_to_noise"):
        _say(f"  signal-to-noise: {s['signal_to_noise']}x")
    return rc


# --------------------------------------------------------------------------
# parse-mm
# --------------------------------------------------------------------------

def cmd_parse_mm(args: argparse.Namespace) -> int:
    from .analyze.scaling import scaling_table
    from .parse.mmengine import parse_mmengine_log, summarize_training

    runs = []
    for spec in args.logs:
        # Accept "path" or "path:label:gpus:batch". Split from the right and only
        # for as many fields as are actually present, because a Windows path
        # starts with "D:" and a plain left split would take the drive letter as
        # the file and "\logs\train.log" as the label.
        parts = [spec]
        if not os.path.exists(spec):
            for n in (3, 2, 1):
                cand = spec.rsplit(":", n)
                if len(cand) == n + 1 and os.path.exists(cand[0]):
                    parts = cand
                    break
        path = parts[0]
        label = parts[1] if len(parts) > 1 and parts[1] else os.path.basename(path)
        gpus = int(parts[2]) if len(parts) > 2 and parts[2] else args.gpus
        batch = int(parts[3]) if len(parts) > 3 and parts[3] else args.batch_per_gpu

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            parsed = parse_mmengine_log(f.read())
        if parsed.get("world_size") and not (len(parts) > 2 and parts[2]):
            gpus = parsed["world_size"]
        s = summarize_training(parsed, gpus=gpus, batch_per_gpu=batch,
                               dataset_size=args.dataset_size)
        s["label"] = label
        runs.append(s)

        _say(f"{label}: {gpus} GPU x {batch}  "
             f"step {s.get('iter_time_s_median')}s  {s.get('samples_per_s')} samples/s  "
             f"data_time {s.get('data_time_ratio')}")
        for w in s.get("warnings", []):
            _say(f"    warning: {w}")

    out: Dict[str, Any] = {"runs": runs}
    if len(runs) > 1:
        tbl = scaling_table(runs)
        out["scaling"] = tbl
        _say("")
        _say("scaling:")
        for r in tbl.get("rows", []):
            inf = (f"  step +{r['step_inflation_pct']}%" if r.get("step_inflation_pct") is not None else "")
            _say(f"  {r['label']:<20} {r['gpus']:>2} GPU  speedup {r['speedup']:>5}  "
                 f"efficiency {r['efficiency_pct']:>5}%{inf}")
        for n in tbl.get("notes", []):
            _say(f"    {n}")

    if args.out:
        _write_json(out, args.out)
        _say(f"\nwrote {args.out}")
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_manifest, write_report

    src = args.rundir
    jf = src if src.endswith(".json") else os.path.join(src, "results.json")
    if not os.path.isfile(jf):
        _say(f"no results.json at {jf}")
        return 1
    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.refresh_manifest:
        data["manifest"] = build_manifest(os.path.dirname(os.path.abspath(jf)))
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(jf)), "report.html")
    write_report(data, out, theme=args.theme)
    _say(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------

def cmd_setup(args: argparse.Namespace) -> int:
    from .config import load_config
    from .setup.build import build_all

    cfg = load_config(args.config)
    plan = build_all(str(cfg.third_party), tools=args.tools.split(",") if args.tools else None,
                     mirror_prefix=args.mirror, dry_run=not args.execute)

    arch = plan["arch"]
    _say(f"architecture: sm_{arch.get('sm')}  ->  nvcc arch flag {arch.get('arch_flag')}")
    _say(f"  {arch.get('note')}")
    if arch.get("warning"):
        _say(f"  WARNING {arch['warning']}")
    _say("")
    for w in plan["warnings"]:
        _say(f"  ! {w}")
    if plan["skipped"]:
        _say("")
        for s in plan["skipped"]:
            _say(f"  skip {s['tool']}: {s['reason']}")
    _say("")
    if not plan["steps"]:
        _say("Nothing to build.")
        return 0
    _say(f"would build {len(plan['steps'])} tool(s), worst case "
         f"~{plan['estimated_total_minutes_worst_case']} minutes:")
    for s in plan["steps"]:
        _say(f"  {s['tool']:<14} ~{s['estimated_minutes']} min   arch {s['arch_flag']}")
        _say(f"      {s['configure_hint']}")
    # A build that finished is not a tool that runs. Printed with the plan rather
    # than after the build, because the person reading this is the same person who
    # will open a fresh terminal an hour from now and get "libnccl.so.2: cannot
    # open shared object file" from a binary that built cleanly.
    _say("")
    _say(f"after the build, write {plan['env_file']} and verify:")
    for v in plan.get("verify") or []:
        _say(f"  $ {v['command']}")
        _say(f"      {v['checks']}")
    _say("")
    _say(plan["reminder"])
    if not args.execute:
        _say("\nThis was a plan only. Re-run with --execute to actually build.")
    else:
        _say("\nExecution is intentionally not automated yet: the commands above are printed so "
             "you can run them and see which one fails. Automating a 90-minute build that dies "
             "silently helps nobody.")
    _write_json(plan, os.path.join(str(cfg.workdir), "setup_plan.json"))
    return 0


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nodebench",
        description="Benchmark one GPU node and say what the numbers mean.",
    )
    p.add_argument("--version", action="version", version=f"nodebench {VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check the machine can produce trustworthy numbers")
    d.set_defaults(func=cmd_doctor)

    r = sub.add_parser("run", help="measure and write a report")
    r.add_argument("-c", "--config", help="YAML config (see nodebench.example.yaml)")
    r.add_argument("-o", "--out", help="output directory")
    r.add_argument("--label", help="node label for the report")
    r.add_argument("--modules", help="comma list: flops,membw,pcie,nccl")
    r.add_argument("--backend", choices=["torch", "native", "both"], help="measurement backend")
    r.add_argument("--quick", action="store_true", help="fewer iterations; for a first look")
    r.add_argument("--force", action="store_true", help="measure even if compatibility fails")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("monitor", help="sample NVML while some other workload runs")
    m.add_argument("-c", "--config")
    m.add_argument("-o", "--out")
    m.add_argument("-i", "--interval", type=float, default=1.0)
    m.add_argument("-n", "--name", default=None, help="CSV filename")
    m.add_argument("--busy-threshold", type=float, default=90.0,
                   help="busy_pct at or above which a GPU counts as active for binning")
    m.add_argument("command", nargs=argparse.REMAINDER,
                   help="optional command to run; sampling stops when it exits")
    m.set_defaults(func=cmd_monitor)

    pm = sub.add_parser("parse-mm", help="mmengine training log -> throughput and scaling")
    pm.add_argument("logs", nargs="+", help="log path, or path:label:gpus:batch_per_gpu")
    pm.add_argument("--gpus", type=int, default=1)
    pm.add_argument("--batch-per-gpu", type=int, default=1)
    pm.add_argument("--dataset-size", type=int, default=None)
    pm.add_argument("-o", "--out")
    pm.set_defaults(func=cmd_parse_mm)

    rp = sub.add_parser("report", help="rebuild HTML from a finished run")
    rp.add_argument("rundir", help="run directory or results.json")
    rp.add_argument("-o", "--out")
    rp.add_argument("--theme", default="light", choices=["light", "dark"])
    rp.add_argument("--refresh-manifest", action="store_true")
    rp.set_defaults(func=cmd_report)

    st = sub.add_parser("setup", help="plan/build the optional native C++ toolchain")
    st.add_argument("-c", "--config")
    st.add_argument("--native", action="store_true", help="accepted for readability; implied")
    st.add_argument("--tools", help="comma list: nccl-tests,nvbandwidth,babelstream,cutlass")
    st.add_argument("--mirror", default="", help="prefix for a download mirror")
    st.add_argument("--execute", action="store_true")
    st.set_defaults(func=cmd_setup)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return args.func(args)
    except KeyboardInterrupt:
        _say("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
