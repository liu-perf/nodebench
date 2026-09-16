"""Generate examples/sample_report.html from synthetic-but-plausible numbers.

The README links to a rendered report so a visitor can see the output before
installing anything. That file has to come from somewhere, and it must not come
from any real machine -- a report carries a node label, a driver version, a card
count and a timestamp, which together identify a specific box.

So the numbers here are made up. They are *shaped* like real measurements
(consistent ratios, plausible CVs, a control group that is actually clean) so
that the layout is exercised honestly, but no line of it is a measurement.
The report says so at the top.

    python examples/make_sample.py

Regenerate and commit the output whenever the renderer changes.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from nodebench.analyze.crosscheck import cross_check  # noqa: E402
from nodebench.analyze.scaling import scaling_table  # noqa: E402
from nodebench.cli import _attach_pcie_ceiling  # noqa: E402
from nodebench.report.caveats import build_caveats  # noqa: E402
from nodebench.report.html import write_report  # noqa: E402


def build() -> dict:
    env = {
        "host": {"platform": "Linux-5.15.0-x86_64", "python": "3.11.9", "cpu_count": 64},
        "gpus": [
            {"index": 0, "name": "NVIDIA GeForce RTX 5090", "compute_cap": "12.0",
             "driver": "570.86.16", "memory_mib": 32607},
            {"index": 1, "name": "NVIDIA GeForce RTX 5090", "compute_cap": "12.0",
             "driver": "570.86.16", "memory_mib": 32607},
        ],
        "torch": {
            "version": "2.7.0+cu128", "cuda_available": True, "cuda_built": "12.8",
            "cudnn": 90800, "device_count": 2, "nccl_version": "2.26.2",
            "arch_list": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
        },
        "nvcc": "Cuda compilation tools, release 12.8, V12.8.61",
    }

    findings = [
        {"level": "ok", "title": "sm_120 present in this torch build",
         "detail": "torch 2.7.0+cu128 contains a cubin for compute capability 12.0. No PTX JIT "
                   "at first launch."},
        {"level": "ok", "title": "CUDA 12.8 is new enough for Blackwell",
         "detail": "sm_120 requires CUDA 12.8 or later."},
        {"level": "warn", "title": "NCCL 2.26.2 on Blackwell",
         "detail": "2.21 is the practical minimum on this architecture and newer is materially "
                   "better. Worth checking whether a newer NCCL is available before treating "
                   "the collective numbers as a ceiling."},
    ]

    topology = {
        "devices": [0, 1],
        "matrix": {0: {0: "X", 1: "PIX"}, 1: {0: "PIX", 1: "X"}},
        "numa_groups": {"group 0": [0, 1]},
        "has_nvlink": False,
        "is_multi_root": False,
        "numa_source": "auto",
    }

    baseline = {
        "clean": True,
        "headline": "Control window clean: all GPUs idle before the workload.",
        "per_gpu": {
            "0": {"power_w": 21.4, "busy_pct": 0.0, "mem_used_mib": 4.0, "pcie_floor_gbs": 0.0021},
            "1": {"power_w": 20.8, "busy_pct": 0.0, "mem_used_mib": 4.0, "pcie_floor_gbs": 0.0019},
        },
        "findings": [],
    }

    flops = {
        "peak_tflops": 178.4, "peak_dtype": "bf16",
        "rows": [
            {"dtype": "bf16", "n": 8192, "tflops": 181.2, "tflops_median": 178.4, "cv_pct": 1.1, "reps": 30},
            {"dtype": "fp16", "n": 8192, "tflops": 179.8, "tflops_median": 177.1, "cv_pct": 1.3, "reps": 30},
            {"dtype": "tf32", "n": 8192, "tflops": 92.6, "tflops_median": 91.0, "cv_pct": 0.9, "reps": 30},
            {"dtype": "fp32", "n": 8192, "tflops": 46.1, "tflops_median": 45.7, "cv_pct": 0.7, "reps": 30},
        ],
    }

    membw = {
        "triad_gbs": 1402.0,
        "rows": [
            {"kernel": "copy", "arrays": 2, "gbs": 1451.0, "bytes": 2147483648, "reps": 30},
            {"kernel": "mul", "arrays": 2, "gbs": 1447.0, "bytes": 2147483648, "reps": 30},
            {"kernel": "add", "arrays": 3, "gbs": 1409.0, "bytes": 3221225472, "reps": 30},
            {"kernel": "triad", "arrays": 3, "gbs": 1402.0, "bytes": 3221225472, "reps": 30},
        ],
    }

    # GPU 1 transfers at half the rate of GPU 0. In GB/s that reads as a broken
    # card; as a percentage of each card's own link (below) both are ~89%, and the
    # fault is the slot, not the GPU. That is the pair of numbers this section exists
    # to put side by side.
    pcie = {
        "h2d_gbs": 56.41,
        "per_gpu": [
            {"gpu": 0, "h2d_gbs": 56.41, "d2h_gbs": 54.92, "buffer_mib": 1024.0},
            {"gpu": 1, "h2d_gbs": 28.14, "d2h_gbs": 27.30, "buffer_mib": 1024.0},
        ],
        "concurrent": {"retention_pct": 71.4, "aggregate_gbs": 60.0, "sum_isolated_gbs": 84.55},
    }

    # Host facts. The link here is Gen5 x16 on GPU 0 and Gen5 x8 on GPU 1 -- an
    # asymmetry that is common, invisible in every other part of a report, and
    # the whole reason the achieved-percentage columns exist.
    host = {
        "platform": "Linux",
        "cpu": {
            "model": "AMD EPYC 9554 64-Core Processor", "sockets": 2,
            "cores_per_socket": 64, "threads_per_core": 2, "logical_cpus": 256,
            "numa_nodes": 2,
            "numa_cpu_ranges": {"node0": "0-63,128-191", "node1": "64-127,192-255"},
        },
        "memory": {"total_gib": 1007.5, "available_gib": 953.7},
        "pcie": {
            "per_gpu": {
                "0": {"gen_max": 5, "gen_current": 1, "width_max": 16, "width_current": 16,
                      "theoretical_gbs": 63.01, "gen_downtrained_at_idle": True,
                      "theoretical_basis": "Gen5 x16 = 63.01 GB/s "
                                           "(max generation, negotiated width)"},
                "1": {"gen_max": 5, "gen_current": 1, "width_max": 16, "width_current": 8,
                      "theoretical_gbs": 31.5, "width_downgraded": True,
                      "gen_downtrained_at_idle": True,
                      "theoretical_basis": "Gen5 x8 = 31.5 GB/s "
                                           "(max generation, negotiated width)"},
            },
            "notes": [
                "GPU 1 negotiated x8 on a card capable of x16. That halves the host transfer "
                "ceiling and it is a property of the slot, not of the workload.",
                "At least one link reports a current generation below its maximum. This is "
                "normal at idle -- PCIe downtrains to save power and retrains under load -- "
                "which is exactly why the theoretical figure above is built from the maximum "
                "generation. Using the idle reading would inflate the achieved percentage by "
                "up to 16x.",
            ],
        },
        "nic": {
            "nics": [
                {"name": "ens1f0", "kind": "net", "pci": "0000:41:00.0",
                 "numa_node": 0, "speed_mbps": 200000, "local_gpus": [0, 1],
                 "n_local_gpus": 2},
            ],
            "verdict": "every NIC has at least one GPU on its own NUMA node",
            "gpus_by_numa": {"0": [0, 1]},
        },
        "unavailable": {},
    }

    nccl = {
        "peak_busbw_gbs": 40.69, "world_size": 2,
        "sets": [{
            "label": "2gpu", "gpus": [0, 1], "world_size": 2, "worst_link": "PIX",
            "peak_busbw_gbs": 40.69,
            "rows": [
                {"collective": "all_gather", "size_mib": 256.0, "algbw_gbs": 41.8, "busbw_gbs": 20.9, "time_ms": 6.42},
                {"collective": "all_gather", "size_mib": 1024.0, "algbw_gbs": 43.1, "busbw_gbs": 21.55, "time_ms": 24.91},
                {"collective": "all_reduce", "size_mib": 16.0, "algbw_gbs": 18.4, "busbw_gbs": 18.4, "time_ms": 0.91},
                {"collective": "all_reduce", "size_mib": 256.0, "algbw_gbs": 38.9, "busbw_gbs": 38.9, "time_ms": 6.90},
                {"collective": "all_reduce", "size_mib": 1024.0, "algbw_gbs": 40.69, "busbw_gbs": 40.69, "time_ms": 26.38},
                {"collective": "reduce_scatter", "size_mib": 1024.0, "algbw_gbs": 42.6, "busbw_gbs": 21.3, "time_ms": 25.20},
            ],
        }],
    }

    p2p = {
        "mode": "software",
        "confidence": "high",
        "headline": "Software P2P is active and every hardware probe is blind to it",
        "explanation": (
            "nvidia-smi topo -p2p reports CNS, cudaDeviceCanAccessPeer returns false, and every "
            "nvbandwidth device-to-device test waived -- all four probe hardware capability only. "
            "All-reduce busbw reaches 72% of measured host-to-device bandwidth, which is not "
            "achievable through host staging. Traffic is taking a peer path that the capability "
            "probes cannot see."
        ),
        "evidence": {
            "h2d_gbs": 56.41,
            "allreduce_busbw_gbs": 40.69,
            "ratio": 0.72,
            "threshold_peer_path": 0.55,
            "threshold_host_bounce": 0.30,
            "cuda_can_access_peer": "false for all pairs",
            "topo_p2p": "CNS",
            "nvbandwidth_d2d": "all waived",
        },
    }

    scaling = scaling_table([
        {"label": "1gpu", "gpus": 1, "samples_per_s": 14.8, "iter_time_s_median": 0.1351},
        {"label": "2gpu", "gpus": 2, "samples_per_s": 28.1, "iter_time_s_median": 0.1423},
    ])

    crosscheck = cross_check(
        {"flops": 178.4, "membw": 1402.0, "pcie_h2d": 56.41, "allreduce_busbw": 40.69},
        {"flops": 186.2, "membw": 1421.0, "pcie_h2d": 55.9, "allreduce_busbw": 41.2},
    )

    caveats = build_caveats(
        modules_run=["flops", "membw", "pcie", "nccl", "p2p", "monitor"],
        modules_skipped={},
        findings={"idle_not_clean": False, "p2p_inconclusive": False, "single_gpu": False},
    )

    manifest = {
        "n_files": 9, "total_bytes": 486_213, "hash_algorithm": "md5",
        "notes": [],
        "files": [
            {"path": "raw/environment.json", "bytes": 2_144, "lines": 71,
             "md5": "5f2c9a1e8b4d7c30aa11", "produced_by": "probe.compat"},
            {"path": "raw/topology.json", "bytes": 1_022, "lines": 34,
             "md5": "9d31ba77e0c45f81cc02", "produced_by": "nvidia-smi topo -m"},
            {"path": "raw/idle_baseline.csv", "bytes": 4_806, "lines": 18,
             "md5": "17aa04c9f3b28d6e5510", "produced_by": "monitor.nvml (control group)"},
            {"path": "raw/monitor.csv", "bytes": 402_918, "lines": 1_447,
             "md5": "c40e7b2159af83dd6621", "produced_by": "monitor.nvml"},
            {"path": "raw/flops.json", "bytes": 1_388, "lines": 52,
             "md5": "aa19c3e4d70b5182ff40", "produced_by": "bench.flops_torch"},
            {"path": "raw/membw.json", "bytes": 1_204, "lines": 48,
             "md5": "b7c250af9e3168d40a13", "produced_by": "bench.membw_torch"},
            {"path": "raw/pcie.json", "bytes": 1_566, "lines": 55,
             "md5": "3e8f10ca47bb92d5e708", "produced_by": "bench.pcie_torch"},
            {"path": "raw/nccl.json", "bytes": 4_902, "lines": 168,
             "md5": "d21b6effa0c39471bb85", "produced_by": "bench.nccl_torch"},
            {"path": "raw/p2p.json", "bytes": 2_263, "lines": 66,
             "md5": "6cf3aa81b429d05e7712", "produced_by": "probe.p2p"},
        ],
    }

    return {
        "node": {
            "label": "EXAMPLE -- synthetic data, not a real machine",
            "vendor": "",
            "notes": "Every number below is fabricated to demonstrate the report layout.",
        },
        "meta": {"version": "0.1.0", "started_at": "example", "backend": "both"},
        "compat": {"env": env, "findings": findings},
        "topology": topology,
        "host": host,
        "baseline": baseline,
        "p2p": p2p,
        "flops": flops,
        "membw": membw,
        "pcie": pcie,
        "nccl": nccl,
        "scaling": scaling,
        "crosscheck": crosscheck,
        "caveats": caveats,
        "manifest": manifest,
    }


if __name__ == "__main__":
    data = build()
    # Same call the CLI makes, so the sample cannot drift from what a real run
    # would show.
    _attach_pcie_ceiling(data["pcie"], data["host"]["pcie"])
    with open(os.path.join(HERE, "sample_run.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    out = write_report(data, os.path.join(HERE, "sample_report.html"))
    print(f"wrote {out}")
    print("wrote " + os.path.join(HERE, "sample_run.json"))
