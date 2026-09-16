"""Environment probe + the compatibility rule that costs people the most time.

The failure this exists to prevent
----------------------------------
You put a brand-new GPU in a machine, run your usual container, and get::

    RuntimeError: CUDA error: no kernel image is available for execution on the device

Nothing is broken. The GPU is simply newer than the CUDA toolkit that compiled
the binary, so the binary contains no cubin for that architecture. The check is
mechanical, takes 50 ms, and almost nobody runs it before burning an afternoon.

`check_compat` runs it. It compares three things that must agree:

  1. the *device* compute capability     (nvidia-smi / torch)
  2. the CUDA toolkit torch was built with
  3. the architecture list torch was actually compiled for (`get_arch_list`)

Item 3 is the one people forget: a torch built with CUDA 12.8 can still lack
`sm_120` if whoever built it did not list that arch.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

# Minimum CUDA toolkit and torch version per compute capability.
# Sources: CUDA release notes + PyTorch release notes. Keep this table honest --
# it is the single most useful thing in the package for a new-hardware bring-up.
ARCH_REQUIREMENTS: Dict[Tuple[int, int], Dict[str, Any]] = {
    (12, 0): {"name": "Blackwell (RTX 50xx / RTX PRO 6000)", "cuda": "12.8", "torch": "2.7"},
    (10, 3): {"name": "Blackwell Ultra (GB300)",             "cuda": "12.9", "torch": "2.7"},
    (10, 0): {"name": "Blackwell datacenter (B100/B200)",    "cuda": "12.8", "torch": "2.7"},
    (9, 0):  {"name": "Hopper (H100/H200)",                  "cuda": "11.8", "torch": "2.0"},
    (8, 9):  {"name": "Ada Lovelace (RTX 40xx / L40S)",      "cuda": "11.8", "torch": "2.0"},
    (8, 6):  {"name": "Ampere consumer (RTX 30xx / A10)",    "cuda": "11.1", "torch": "1.8"},
    (8, 0):  {"name": "Ampere datacenter (A100)",            "cuda": "11.0", "torch": "1.7"},
    (7, 5):  {"name": "Turing (RTX 20xx / T4)",              "cuda": "10.0", "torch": "1.2"},
    (7, 0):  {"name": "Volta (V100)",                        "cuda": "9.0",  "torch": "1.0"},
}


def _ver_tuple(s: str) -> Tuple[int, ...]:
    out: List[int] = []
    for part in str(s).split("."):
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def _ver_lt(a: str, b: str) -> bool:
    """True if version a < version b, comparing component-wise."""
    ta, tb = _ver_tuple(a), _ver_tuple(b)
    n = max(len(ta), len(tb))
    ta = ta + (0,) * (n - len(ta))
    tb = tb + (0,) * (n - len(tb))
    return ta < tb


def _sh(cmd: List[str], timeout: int = 30) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else (r.stdout or r.stderr)
    except Exception:
        return None


def _nvidia_smi_gpus() -> List[Dict[str, Any]]:
    q = "index,name,compute_cap,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max"
    out = _sh(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    gpus: List[Dict[str, Any]] = []
    if not out:
        return gpus
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            cc = parts[2]
            major, minor = (int(x) for x in cc.split("."))
        except Exception:
            major = minor = 0
        gpus.append(
            {
                "index": int(parts[0]) if parts[0].isdigit() else len(gpus),
                "name": parts[1],
                "compute_cap": parts[2],
                "sm": f"sm_{major}{minor}",
                "cc_major": major,
                "cc_minor": minor,
                "driver": parts[3],
                "memory_mib": int(float(parts[4])) if parts[4] not in ("", "[N/A]") else None,
                "pcie_gen_max": parts[5] if len(parts) > 5 else None,
                "pcie_width_max": parts[6] if len(parts) > 6 else None,
            }
        )
    return gpus


def probe_environment() -> Dict[str, Any]:
    """Collect everything about the software+hardware stack. No workload is run."""
    env: Dict[str, Any] = {
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": None,
        },
        "gpus": _nvidia_smi_gpus(),
        "torch": None,
        "nvcc": None,
    }
    try:
        import os

        env["host"]["cpu_count"] = os.cpu_count()
    except Exception:
        pass

    nvcc = _sh(["nvcc", "--version"])
    if nvcc:
        for line in nvcc.splitlines():
            if "release" in line:
                env["nvcc"] = line.strip()
                break

    try:
        import torch

        env["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_built": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "nccl_version": None,
            "nccl_available": False,
        }
        # "No NCCL" and "could not determine the NCCL version" are different
        # facts and lead to different actions, so they get different fields.
        # Windows torch builds ship without NCCL entirely -- there, collectives
        # are not slow, they are absent, and a report that says `NCCL None` is
        # inviting the reader to guess which of the two it meant.
        try:
            if torch.distributed.is_nccl_available():
                env["torch"]["nccl_available"] = True
                v = torch.cuda.nccl.version()
                env["torch"]["nccl_version"] = ".".join(str(x) for x in v)
        except Exception:
            pass
    except ImportError:
        env["torch"] = {"error": "torch not installed"}
    return env


def check_compat(env: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return a list of findings. Each has level in {ok, warn, error}.

    An `error` means: stop, this run will produce garbage or crash.
    A `warn` means: it will run, but a number may be misleading.
    """
    findings: List[Dict[str, str]] = []
    gpus = env.get("gpus") or []
    torch_info = env.get("torch") or {}

    if not gpus:
        findings.append(
            {
                "level": "error",
                "title": "No GPU visible",
                "detail": "nvidia-smi returned nothing. Driver not loaded, or the container "
                "was started without --gpus.",
            }
        )
        return findings

    if torch_info.get("error"):
        findings.append(
            {"level": "error", "title": "PyTorch not installed", "detail": torch_info["error"]}
        )
        return findings

    if not torch_info.get("cuda_available"):
        findings.append(
            {
                "level": "error",
                "title": "torch.cuda.is_available() is False",
                "detail": "The GPU is visible to nvidia-smi but not to torch. Usually a "
                "CPU-only torch wheel, or a container missing the device nodes "
                "(after a host driver upgrade, `docker restart` the container).",
            }
        )
        return findings

    cuda_built = torch_info.get("cuda_built") or "0.0"
    arch_list = torch_info.get("arch_list") or []
    torch_ver = torch_info.get("version") or "0.0"

    # Only the distinct architectures matter, not each card.
    seen: set = set()
    for g in gpus:
        key = (g["cc_major"], g["cc_minor"])
        if key in seen:
            continue
        seen.add(key)
        req = ARCH_REQUIREMENTS.get(key)
        sm = g["sm"]
        arch_tag = f"sm_{g['cc_major']}{g['cc_minor']}"

        if req is None:
            findings.append(
                {
                    "level": "warn",
                    "title": f"{sm} is not in the compatibility table",
                    "detail": f"{g['name']} reports compute capability {g['compute_cap']}. "
                    "nodebench has no rule for it; verify the CUDA requirement yourself.",
                }
            )
            continue

        if _ver_lt(cuda_built, req["cuda"]):
            findings.append(
                {
                    "level": "error",
                    "title": f"{sm} needs CUDA >= {req['cuda']}, torch was built with {cuda_built}",
                    "detail": f"{g['name']} is {req['name']}. This combination produces "
                    "'no kernel image is available for execution on the device'. "
                    f"Install a torch built against CUDA {req['cuda']} or newer.",
                }
            )
        if _ver_lt(torch_ver, req["torch"]):
            findings.append(
                {
                    "level": "error",
                    "title": f"{sm} needs torch >= {req['torch']}, found {torch_ver}",
                    "detail": f"{g['name']} is {req['name']}.",
                }
            )

        # The check people forget: right CUDA, wrong arch list.
        if arch_list and not any(arch_tag in a for a in arch_list):
            findings.append(
                {
                    "level": "error",
                    "title": f"torch has no cubin for {arch_tag}",
                    "detail": f"torch.cuda.get_arch_list() = {arch_list}. Even though the CUDA "
                    "version is high enough, this build was not compiled for your "
                    "device. It will fall back to PTX JIT if the PTX arch is "
                    "compatible, or fail outright.",
                }
            )

    # Blackwell + NCCL: an old NCCL against a new driver is a classic symbol error.
    if any(g["cc_major"] >= 10 for g in gpus):
        nccl = torch_info.get("nccl_version")
        if nccl and _ver_lt(nccl, "2.21"):
            findings.append(
                {
                    "level": "warn",
                    "title": f"NCCL {nccl} is old for this architecture",
                    "detail": "Symptoms look like `undefined symbol: ncclCommQueryProperties` or "
                    "silently poor collective bandwidth. NCCL >= 2.21 is recommended.",
                }
            )

    names = {g["name"] for g in gpus}
    if len(names) > 1:
        findings.append(
            {
                "level": "warn",
                "title": "Mixed GPU models in one node",
                "detail": f"Found {sorted(names)}. Collectives run at the speed of the slowest "
                "member and scaling numbers are not comparable to a homogeneous node.",
            }
        )

    if not findings:
        findings.append(
            {
                "level": "ok",
                "title": "Software stack matches the hardware",
                "detail": f"{len(gpus)}x {gpus[0]['name']} ({gpus[0]['sm']}), driver "
                f"{gpus[0]['driver']}, torch {torch_ver} built with CUDA {cuda_built}.",
            }
        )
    return findings
