"""Automatic "what this run does NOT tell you" section.

The most common way a benchmark report misleads is not a wrong number. It is a
correct number read as if it answered a broader question than it does. A single
square GEMM becomes "the card's FP16 performance"; one message size becomes
"the interconnect"; a 60-second run becomes "sustained".

Rather than trusting the author to remember, the framework derives the
limitations from the configuration that actually ran. Every module declares
what it covers and what it structurally cannot cover; anything not in the
executed module list produces an explicit "not measured" line.

The rule this encodes: **not covered is recorded, never omitted.** A reader
must be able to tell the difference between "we measured it and it was fine"
and "we never looked".
"""

from __future__ import annotations

from typing import Any, Dict, List

# What each module structurally cannot tell you, no matter how well it runs.
STRUCTURAL = {
    "flops": [
        "Achieved throughput on square GEMMs at the configured sizes. Not the silicon peak, "
        "and not representative of the skinny/irregular shapes real models spend time in.",
        "Single-GPU only. Says nothing about how compute overlaps with communication.",
    ],
    "membw": [
        "Streaming access on large contiguous buffers -- the best case for the memory system. "
        "Random or strided access patterns will be far slower and are not measured.",
        "Cache behaviour is deliberately excluded by sizing buffers past L2.",
    ],
    "pcie": [
        "Bulk transfer of large pinned buffers. Small-transfer latency and the per-call "
        "overhead that dominates inference serving are not measured.",
        "Host-to-device and device-to-host only. Does not establish whether device-to-device "
        "traffic takes a peer path -- see the P2P section for that.",
    ],
    "nccl": [
        "Collectives in isolation on an otherwise idle node. Real training overlaps them with "
        "compute, so these are an upper bound on what a training job will see.",
        "Only the configured message sizes and collectives. Small-message latency behaviour "
        "differs qualitatively from large-message bandwidth behaviour.",
    ],
    "p2p": [
        "The verdict is inferred from a bandwidth ratio, not read from a register. It is "
        "stated with a confidence level for that reason.",
    ],
    "host": [
        "Nothing here is measured. These are declarations read from nvidia-smi, lscpu and "
        "/sys, used as denominators for numbers measured elsewhere.",
        "The link figure is the theoretical rate of the negotiated width at the maximum "
        "generation. It is a ceiling, not a prediction: encoding and protocol overhead mean "
        "no real transfer reaches it.",
        "NIC affinity is read from NUMA node numbers, which says where the devices sit. "
        "Whether a job was actually pinned there is a property of the job, not of the node.",
    ],
    "monitor": [
        "NVML's PCIe counter is a ~20 ms windowed average, so it cannot be integrated to a "
        "transfer volume and its peaks are 20 ms peaks, not link capacity.",
        "'busy_pct' is the fraction of time at least one kernel was resident. It is not SM "
        "occupancy; a card can read 100% busy while using a few percent of its SMs.",
    ],
}

# Things nobody measured because no module covers them at all.
NEVER_COVERED = [
    ("thermal_sustained", "Sustained multi-hour thermal behaviour. Runs here are minutes long; "
     "a datacentre job is hours, and clocks drop over that horizon."),
    ("multi_node", "Anything across nodes. This tool measures one node. InfiniBand/RoCE fabric, "
     "NCCL tuning across hosts, and rail topology are entirely out of scope."),
    ("numerics", "Numerical accuracy. Throughput is measured, correctness of the results is not. "
     "A fast kernel producing wrong values would pass every test here."),
    ("storage", "Storage and dataset pipeline throughput, except indirectly via data_time in "
     "training-log parsing."),
    ("power_efficiency", "Performance per watt as a design conclusion. Power is sampled, but no "
     "controlled efficiency comparison is made."),
]


def build_caveats(
    modules_run: List[str],
    modules_skipped: Dict[str, str],
    findings: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compose the limitations section from what actually executed.

    ``modules_skipped`` maps module name -> reason. Reasons matter: "no second
    GPU" and "user disabled it" are different facts about the run.
    """
    findings = findings or {}
    covered: List[Dict[str, Any]] = []
    for m in modules_run:
        covered.append({"module": m, "limits": STRUCTURAL.get(m, [])})

    not_run: List[Dict[str, str]] = []
    for m, reason in (modules_skipped or {}).items():
        not_run.append(
            {
                "module": m,
                "reason": reason or "not requested",
                "consequence": f"Nothing in this report speaks to {m}.",
            }
        )
    for m in STRUCTURAL:
        if m not in modules_run and m not in (modules_skipped or {}):
            not_run.append(
                {"module": m, "reason": "not in the configured module list",
                 "consequence": f"Nothing in this report speaks to {m}."}
            )

    out_of_scope = [{"topic": k, "text": v} for k, v in NEVER_COVERED]

    conditional: List[str] = []
    if findings.get("idle_not_clean"):
        conditional.append(
            "The control window showed the node was not idle. Every number below inherits "
            "that contamination; treat them as a lower bound at best."
        )
    if findings.get("p2p_inconclusive"):
        conditional.append(
            "The P2P path could not be classified from the evidence available. The "
            "multi-GPU numbers are still valid measurements, but the reason behind them "
            "is not established."
        )
    if findings.get("single_gpu"):
        conditional.append(
            "Only one GPU was visible, so every multi-GPU section is absent by necessity, "
            "not by choice."
        )

    return {
        "covered_with_limits": covered,
        "not_run": not_run,
        "out_of_scope": out_of_scope,
        "conditional": conditional,
        "principle": (
            "Anything not covered is recorded here rather than omitted, so a reader can "
            "tell 'measured and fine' apart from 'never looked'."
        ),
    }
