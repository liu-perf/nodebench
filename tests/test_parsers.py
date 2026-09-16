"""Tests for the pure layers.

Only functions with no IO and no GPU are tested here, which is the point of
keeping the parsers and the classifiers pure: the interesting logic can be
checked on a laptop with no CUDA, and an upstream tool changing its output
format is caught by a test instead of by a wrong number in a report.

    pip install -e ".[dev]" && pytest
"""

from __future__ import annotations

import pytest

from nodebench.analyze.crosscheck import cross_check
from nodebench.analyze.ring_theory import (
    busbw_factor,
    gradient_traffic_per_step,
    reconcile,
    ring_allreduce_bytes,
)
from nodebench.analyze.scaling import scaling_table
from nodebench.parse.mmengine import parse_mmengine_log, summarize_training
from nodebench.parse.native import (
    parse_babelstream,
    parse_cutlass_profiler,
    parse_nccl_tests,
    parse_nvbandwidth,
)
from nodebench.probe.host import (
    match_nics_to_gpus,
    parse_lscpu,
    parse_meminfo,
    parse_pcie_query,
    pcie_achieved_pct,
    pcie_theoretical_gbs,
    summarize_pcie,
)
from nodebench.probe.p2p import classify_p2p, parse_topo_p2p
from nodebench.probe.topology import parse_topo_matrix


@pytest.fixture(autouse=True)
def _clean_nccl_env(monkeypatch):
    """classify_p2p reads NCCL_P2P_* from the environment, and a developer who
    happens to have them exported would otherwise see unrelated failures."""
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("NCCL_P2P_LEVEL", raising=False)

# --------------------------------------------------------------------------
# fixtures: real output, trimmed
# --------------------------------------------------------------------------

NCCL_TESTS = """\
# nThread 1 nGpus 1 minBytes 1073741824 maxBytes 8589934592 step: 2(factor)
#
#       size         count      type   redop    root     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)
  1073741824     268435456     float     sum      -1    26312   40.81   71.42      0    26299   40.83   71.45      0
  2147483648     536870912     float     sum      -1    52410   40.97   71.70      0    52398   40.98   71.72      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 71.5725
"""

NVBANDWIDTH = """\
Running host_to_device_memcpy_ce.
memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)
           0         1
0      26.05     25.88
SUM host_to_device_memcpy_ce 51.93

Running device_to_device_memcpy_read_ce.
Waived.
"""

BABELSTREAM = """\
Function    MBytes/sec  Min (sec)   Max         Average
Copy        1451000.0   0.00037     0.00041     0.00038
Mul         1447000.0   0.00037     0.00040     0.00038
Add         1409000.0   0.00057     0.00061     0.00058
Triad       1402000.0   0.00057     0.00062     0.00058
"""

TOPO = """\
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PIX     0-31            0
GPU1    PIX      X      0-31            0

Legend:
  X    = Self
"""

CUTLASS_CSV = """\
Problem,Provider,OperationKind,Operation,Disposition,Status,m,n,k,Bytes,Flops,Runtime,GB/s,GFLOPs
1,CUTLASS,gemm,cutlass_tensorop_h16816gemm_256x128,Passed,Success,8192,8192,8192,402653184,1099511627776,3.421,117.7,321400.5
2,CUTLASS,gemm,cutlass_tensorop_h16816gemm_128x128,Passed,Success,4096,4096,4096,100663296,137438953472,0.512,196.7,268435.4
"""

# Same machine, same kernel, verification left on. The GFLOPs figure is a third
# of the one above and nothing in the output says so.
CUTLASS_VERIFY_ON = """\
=============================
  Operation: cutlass_tensorop_h16816gemm_256x128
  Verification: ON
  Disposition: Passed
  Status: Success
  GFLOPs: 108422.1
  Runtime: 10.140
"""

# Trimmed from a real ``nvidia-smi -q``. Both halves of the trap are present: the
# generation reads Gen1 because the card is idle, and the width reads x8 because
# the slot only wired eight lanes. "Device Current" and "Host Max" are in here on
# purpose -- they are the lines a looser parser picks up by mistake.
NVSMI_Q = """\
GPU 00000000:02:00.0
    Product Name                                       : NVIDIA GeForce RTX 5060 Ti
    GPU Link Info
        PCIe Generation
            Max                                        : 5
            Current                                    : 1
            Device Current                             : 1
            Device Max                                 : 5
            Host Max                                   : 5
        Link Width
            Max                                        : 16x
            Current                                    : 8x
    Bridge Chip
        Type                                           : N/A
    Clocks
        Max                                            : 2617 MHz
"""

LSCPU = """\
Architecture:          x86_64
CPU(s):                256
Thread(s) per core:    2
Core(s) per socket:    64
Socket(s):             2
Model name:            AMD EPYC 9554 64-Core Processor
NUMA node(s):          2
NUMA node0 CPU(s):     0-63,128-191
NUMA node1 CPU(s):     64-127,192-255
"""

MEMINFO = """\
MemTotal:       1056485376 kB
MemFree:         900000000 kB
MemAvailable:   1000000000 kB
"""

MMENGINE = "\n".join(
    ["2026/07/22 02:50:01 - mmengine - INFO - Distributed training: True",
     "2026/07/22 02:50:01 - mmengine - INFO - World size: 4"]
    + [f"2026/07/22 02:5{i // 60}:{i % 60:02d} - mmengine - INFO - Epoch(train) "
       f"[1][{i:4d}/14786]  lr: 1.0000e-04  eta: 5:12:33  time: {0.500 if i < 30 else 0.140:.4f}  "
       f"data_time: 0.0089  memory: 9876  loss: 3.21"
       for i in range(1, 201)]
)


# --------------------------------------------------------------------------
# native parsers
# --------------------------------------------------------------------------

def test_nccl_tests_takes_out_of_place_columns():
    r = parse_nccl_tests(NCCL_TESTS)
    assert r["n_rows"] == 2
    assert r["rows"][0]["busbw_gbs"] == 71.42        # not 71.45, the in-place column
    assert r["avg_busbw_gbs"] == 71.5725
    assert r["peak_busbw_gbs"] == 71.70


def test_nvbandwidth_records_waived_tests():
    r = parse_nvbandwidth(NVBANDWIDTH)
    # All-D2D-waived is itself the finding, so it must survive parsing.
    assert "device_to_device_memcpy_read_ce" in r["waived"]
    assert r["h2d_sum_gbs"] == 51.93


def test_babelstream_converts_mb_to_gb():
    r = parse_babelstream(BABELSTREAM)
    assert r["triad"]["gbs"] == 1402.0


def test_cutlass_csv_reads_gflops_and_shape():
    r = parse_cutlass_profiler(CUTLASS_CSV)
    assert r["n_rows"] == 2
    assert r["peak_gflops"] == 321400.5
    assert r["peak_tflops"] == 321.4
    assert r["rows"][0]["k"] == 8192


def test_cutlass_refuses_to_report_a_verified_run():
    # The number parses fine. Reporting it is the bug, because verification runs
    # inside the timed region and the result is 2-3x low for no visible reason.
    r = parse_cutlass_profiler(CUTLASS_VERIFY_ON)
    assert r["verification_on"] is True
    assert r["peak_tflops"] is None
    assert r["rows"][0]["gflops"] == 108422.1     # evidence stays on disk
    assert "--verification-enabled=false" in r["rejected"]


def test_cutlass_csv_without_verification_line_is_reported():
    r = parse_cutlass_profiler(CUTLASS_CSV)
    assert r["verification_on"] is False
    assert "rejected" not in r


# --------------------------------------------------------------------------
# host: PCIe link, CPU, NIC placement
# --------------------------------------------------------------------------

def test_pcie_query_keeps_generation_and_width_apart():
    links = parse_pcie_query(NVSMI_Q)
    assert links[0] == {"gen_max": 5, "gen_current": 1,
                        "width_max": 16, "width_current": 8}
    # "Clocks / Max : 2617 MHz" comes after Link Width and must not be read as one.
    assert links[0]["width_max"] == 16


def test_pcie_denominator_uses_max_generation_and_current_width():
    s = summarize_pcie(parse_pcie_query(NVSMI_Q))["per_gpu"]["0"]
    # Gen5 x8, not Gen1 x8 (0.25 GB/s * 8 = 2.0) and not Gen5 x16 (63.0).
    assert s["theoretical_gbs"] == 31.5
    assert s["width_downgraded"] is True
    assert s["gen_downtrained_at_idle"] is True


def test_pcie_idle_downtrain_is_called_out_not_silently_used():
    notes = summarize_pcie(parse_pcie_query(NVSMI_Q))["notes"]
    assert any("16x" in n for n in notes)


def test_pcie_theoretical_is_none_not_zero_when_unknown():
    # A zero here would divide into a ratio and come back as infinity, which
    # renders as a number and reads as a measurement.
    assert pcie_theoretical_gbs(None, 16) is None
    assert pcie_theoretical_gbs(5, None) is None
    assert pcie_theoretical_gbs(5, 16) == 63.01


def test_pcie_achievement_is_not_clamped_above_100():
    # 25.48 GB/s against a Gen1 x8 denominator. The absurd answer is the signal.
    assert pcie_achieved_pct(25.48, 2.0) == 1274.0
    assert pcie_achieved_pct(25.48, 31.5) == 80.9
    assert pcie_achieved_pct(25.48, None) is None


def test_lscpu_reads_socket_and_numa_layout():
    c = parse_lscpu(LSCPU)
    assert c["sockets"] == 2
    assert c["cores_per_socket"] == 64
    assert c["logical_cpus"] == 256
    assert c["numa_nodes"] == 2
    assert c["numa_cpu_ranges"]["node1"] == "64-127,192-255"


def test_meminfo_in_gib():
    m = parse_meminfo(MEMINFO)
    assert m["total_gib"] == 1007.5
    assert m["available_gib"] == 953.7


def test_nic_on_a_node_with_no_gpu_is_flagged():
    v = match_nics_to_gpus(
        [{"name": "ens1f0", "numa_node": 0}, {"name": "ens2f0", "numa_node": 1}],
        {0: 0, 1: 0},
    )
    assert v["nics"][0]["local_gpus"] == [0, 1]
    assert v["nics"][1]["local_gpus"] == []
    assert "ens2f0" in v["verdict"]


def test_nic_affinity_unknown_is_said_not_guessed():
    v = match_nics_to_gpus([{"name": "ens1f0", "numa_node": None}], {0: 0})
    assert "unknown" in v["verdict"]


def test_topo_matrix_and_numa():
    t = parse_topo_matrix(TOPO)
    assert t["gpus"] == [0, 1]
    assert t["matrix"][0][1] == "PIX"


# --------------------------------------------------------------------------
# ring theory
# --------------------------------------------------------------------------

def test_busbw_factors_match_nccl_tests_convention():
    assert busbw_factor("all_reduce", 2) == 1.0
    assert busbw_factor("all_reduce", 4) == 1.5
    assert busbw_factor("all_gather", 4) == 0.75
    assert busbw_factor("broadcast", 8) == 1.0


def test_ring_bytes():
    # 1 GiB buffer, 4 ranks -> 2 * 3/4 * 1 GiB
    assert ring_allreduce_bytes(1 << 30, 4) == 1.5 * (1 << 30)


def test_reconcile_flags_missing_correction():
    # busbw quoted equal to algbw on 4 ranks: the correction was never applied.
    r = reconcile(measured_gbs=40.0, collective="all_reduce", world_size=4, algbw_gbs=40.0)
    assert r["consistent"] is False
    assert "warning" in r


def test_lora_vs_full_traffic_ratio():
    full = gradient_traffic_per_step(7_000_000_000, 4)
    lora = gradient_traffic_per_step(20_000_000, 4)
    assert full["bytes_per_step"] / lora["bytes_per_step"] == 350.0


# --------------------------------------------------------------------------
# p2p classifier -- the calibration cases
# --------------------------------------------------------------------------

NO_CUDA_PEER = {"available": True, "n_pairs": 2, "n_peer_capable": 0,
                "all_capable": False, "none_capable": True}
NO_HW_PEER = {"parsed": True, "counts": {"CNS": 2}, "all_ok": False, "none_ok": True}


def test_p2p_topo_parser_reads_cns():
    d = parse_topo_p2p("        GPU0    GPU1\nGPU0     X      CNS\nGPU1    CNS      X\n")
    assert d["none_ok"] is True
    assert d["dominant"] == "CNS"


def test_software_p2p_detected_when_all_hardware_probes_say_no():
    # The calibration case: hardware probes unanimous "no", ratio 0.72.
    v = classify_p2p(NO_CUDA_PEER, NO_HW_PEER, h2d_gbs=56.41,
                     allreduce_busbw_gbs=40.69, nvlink=False)
    assert v.mode == "software"
    assert v.confidence == "high"


def test_host_bounce_detected():
    # Same node before the fix: ratio 0.22.
    v = classify_p2p(NO_CUDA_PEER, NO_HW_PEER, h2d_gbs=37.0,
                     allreduce_busbw_gbs=8.11, nvlink=False)
    assert v.mode == "host_bounce"


def test_middle_band_is_inconclusive_not_guessed():
    v = classify_p2p(NO_CUDA_PEER, NO_HW_PEER, h2d_gbs=50.0,
                     allreduce_busbw_gbs=20.0, nvlink=False)   # ratio 0.40
    assert v.mode == "inconclusive"


def test_no_throughput_measurement_means_no_verdict():
    v = classify_p2p(NO_CUDA_PEER, NO_HW_PEER, h2d_gbs=None,
                     allreduce_busbw_gbs=None, nvlink=False)
    assert v.mode == "unknown"
    assert v.confidence == "low"


# --------------------------------------------------------------------------
# mmengine + scaling
# --------------------------------------------------------------------------

def test_mmengine_drops_warmup_and_uses_median():
    parsed = parse_mmengine_log(MMENGINE)
    assert parsed["n_iters"] == 200
    assert parsed["world_size"] == 4
    s = summarize_training(parsed, gpus=4, batch_per_gpu=2)
    # The 0.500s warmup steps must not drag the median up.
    assert abs(s["iter_time_s_median"] - 0.140) < 1e-6
    assert s["global_batch"] == 8


def test_data_time_ratio_warns_when_input_bound():
    log = "\n".join(
        f"Epoch(train) [1][{i:4d}/1000]  time: 0.2000  data_time: 0.0800  memory: 100  loss: 1.0"
        for i in range(1, 101)
    )
    s = summarize_training(parse_mmengine_log(log), gpus=1, batch_per_gpu=1)
    assert s["data_time_ratio"] == 0.4
    assert any("input-bound" in w for w in s["warnings"])


def test_scaling_reports_step_inflation_alongside_speedup():
    t = scaling_table([
        {"label": "1gpu", "gpus": 1, "samples_per_s": 10.0, "iter_time_s_median": 0.100},
        {"label": "4gpu", "gpus": 4, "samples_per_s": 32.0, "iter_time_s_median": 0.125},
    ])
    row = t["rows"][1]
    assert row["speedup"] == 3.2
    assert row["efficiency_pct"] == 80.0
    assert row["step_inflation_pct"] == 25.0   # hidden by speedup alone
    assert any("communication cost" in n for n in t["notes"])


def test_scaling_flags_topology_pair():
    t = scaling_table([
        {"label": "1gpu", "gpus": 1, "samples_per_s": 10.0},
        {"label": "4gpu_same_numa", "gpus": 4, "samples_per_s": 36.0},
        {"label": "4gpu_cross_numa", "gpus": 4, "samples_per_s": 28.0},
    ])
    assert t["topology_pairs"][0]["best"] == "4gpu_same_numa"
    assert any("placement, not compute" in n for n in t["notes"])


# --------------------------------------------------------------------------
# cross-check
# --------------------------------------------------------------------------

def test_crosscheck_marks_large_divergence_as_conflict():
    c = cross_check({"membw": 1400.0}, {"membw": 900.0})
    assert c["rows"][0]["status"] == "conflict"
    assert c["counts"]["conflict"] == 1


def test_crosscheck_agrees_within_five_percent():
    c = cross_check({"pcie_h2d": 56.4}, {"pcie_h2d": 55.9})
    assert c["rows"][0]["status"] == "agree"
