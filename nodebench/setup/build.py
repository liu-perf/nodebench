"""Build the native C++ benchmark tools.

This is a port of a shell script that took several days of failed builds to get
right. Each guard below exists because something actually broke; they are
documented individually so nobody deletes one for being ugly.

The nine traps
---------------

1. **Architecture flag.** ``nvcc -arch`` wants ``sm_90`` where nvidia-smi
   reports ``9.0``. Getting this wrong produces a binary that builds cleanly
   and then dies at runtime with "no kernel image is available".

2. **The 'a' suffix.** From Hopper onward the architecture-specific features
   (wgmma, the TMA path, and everything CUTLASS 3.x builds its fast kernels on)
   live behind ``sm_90a`` / ``sm_120a``, not ``sm_90`` / ``sm_120``. Build
   without the suffix and CUTLASS silently falls back to slow kernels. This is
   the single most common reason a Hopper or Blackwell card benchmarks like a
   previous-generation part.

3. **CMake 4.x breaks CUTLASS 3.9.x.** CMake 4 removed compatibility with
   ``cmake_minimum_required(VERSION 3.x)`` below 3.5, which CUTLASS still
   declares. Pin ``cmake<4`` in a user-local install rather than fighting it.

4. **git clone dies on slow or filtered links.** These repos are large and a
   clone that fails at 90% leaves nothing to resume. Download release tarballs
   with ``curl -C -`` / ``wget -c`` instead, so a retry continues rather than
   restarting. A mirror prefix is configurable for networks where the direct
   host is unreachable.

5. **Poisoned CMake cache.** A build directory whose ``CMakeCache.txt`` was
   written by a different compiler, CUDA version or CMake version fails in ways
   that look like source errors. Detect and delete rather than trusting it.

6. **No root is the normal case.** Benchmark nodes are usually shared and
   unprivileged. Everything installs under the work directory; anything that
   genuinely needs root is skipped with a stated reason instead of failing the
   run.

7. **Reruns must be cheap.** A full inventory pass runs first, and any tool
   already present is skipped. Rebuilding CUTLASS because a later step failed
   is how a 20-minute job becomes a 2-hour one.

8. **Library headers are checked before the build, not during it.**
   nvbandwidth needs Boost.program_options. Without it the build fails partway
   through, after the download and the configure step have already run, with a
   compiler error that reads like a source problem. It is also the one
   dependency here that normally needs root to install, so discovering it late
   means discovering it in the wrong terminal. Checked up front and reported as
   a prerequisite instead.

9. **A successful build is not a working tool.** The binaries link against
   libraries that live in the build tree, so a fresh terminal that has not
   sourced the environment file gets ``libnccl.so.2: cannot open shared object
   file`` -- or worse, resolves against an older system NCCL and dies with
   ``undefined symbol: ncclCommQueryProperties``. Both look like build failures
   and are not. That is why this module writes an environment file and why
   ``verify_commands()`` exists: ``ldd $(command -v all_reduce_perf)`` is the
   only thing that actually answers which NCCL got loaded.

Traps 1-7 are build-time. Traps 8 and 9 are the two that bite either side of
the build, which is why they took longest to recognise.
"""

from __future__ import annotations

import os
import posixpath
import shutil
import subprocess
from typing import Any, Dict, List, Optional

TOOLS = {
    "nccl-tests": {
        "binaries": ["all_reduce_perf", "all_gather_perf", "reduce_scatter_perf", "alltoall_perf"],
        "url": "https://github.com/NVIDIA/nccl-tests/archive/refs/tags/v2.13.10.tar.gz",
        "dirname": "nccl-tests-2.13.10",
        "minutes": "2-5",
    },
    "nvbandwidth": {
        "binaries": ["nvbandwidth"],
        "url": "https://github.com/NVIDIA/nvbandwidth/archive/refs/tags/v0.7.tar.gz",
        "dirname": "nvbandwidth-0.7",
        "minutes": "3-8",
    },
    "babelstream": {
        "binaries": ["cuda-stream"],
        "url": "https://github.com/UoB-HPC/BabelStream/archive/refs/tags/v5.0.tar.gz",
        "dirname": "BabelStream-5.0",
        "minutes": "1-3",
    },
    "cutlass": {
        "binaries": ["cutlass_profiler"],
        "url": "https://github.com/NVIDIA/cutlass/archive/refs/tags/v3.9.2.tar.gz",
        "dirname": "cutlass-3.9.2",
        "minutes": "30-90",
    },
}


# --------------------------------------------------------------------------
# trap 1 + 2: architecture detection
# --------------------------------------------------------------------------

def detect_arch() -> Dict[str, Any]:
    """Turn nvidia-smi's ``12.0`` into nvcc's ``120a``.

    Returns both the plain and suffixed forms. Callers should use ``arch_flag``;
    ``sm`` is kept for tools that reject the suffix.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"cannot query compute capability: {e}"}

    caps = sorted({ln.strip() for ln in out.splitlines() if ln.strip()})
    if not caps:
        return {"ok": False, "error": "nvidia-smi returned no compute capability"}

    sms = [c.replace(".", "") for c in caps]
    sm = sms[0]
    try:
        needs_a = int(sm) >= 90
    except ValueError:
        needs_a = False

    return {
        "ok": True,
        "compute_caps": caps,
        "sm": sm,
        "arch_flag": f"{sm}a" if needs_a else sm,
        "mixed": len(sms) > 1,
        "all_sms": sms,
        "note": (
            "Architecture-specific features (wgmma, TMA) require the 'a' suffix from Hopper "
            "onward. Without it CUTLASS builds, runs, and quietly uses slow kernels."
            if needs_a else "No 'a' suffix needed below sm_90."
        ),
        "warning": (
            "GPUs report different compute capabilities. Binaries will be built for the "
            f"lowest ({sm}); cards above it will not use their newest instructions."
            if len(sms) > 1 else None
        ),
    }


# --------------------------------------------------------------------------
# trap 6 + 7: environment inventory
# --------------------------------------------------------------------------

def find_bin(name: str, extra_dirs: Optional[List[str]] = None) -> Optional[str]:
    """Look in PATH, then in the work tree. Skipping an existing tool is the point."""
    p = shutil.which(name)
    if p:
        return p
    for d in extra_dirs or []:
        for root, _dirs, files in os.walk(d):
            if name in files:
                cand = os.path.join(root, name)
                if os.access(cand, os.X_OK):
                    return cand
    return None


def inventory(third_party: str) -> Dict[str, Any]:
    """What is already built. Run this before deciding to build anything."""
    have: Dict[str, Any] = {}
    for tool, spec in TOOLS.items():
        found = {b: find_bin(b, [third_party]) for b in spec["binaries"]}
        have[tool] = {
            "binaries": found,
            "complete": all(found.values()),
            "partial": any(found.values()) and not all(found.values()),
        }
    return have


def probe_toolchain() -> Dict[str, Any]:
    """Compilers and privileges. No root is the normal case, not an error."""
    def ver(cmd: List[str]) -> Optional[str]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return (r.stdout or r.stderr).strip().splitlines()[0] if r.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None

    is_root = hasattr(os, "geteuid") and os.geteuid() == 0
    return {
        "nvcc": ver(["nvcc", "--version"]),
        "gcc": ver(["gcc", "--version"]),
        "cmake": ver(["cmake", "--version"]),
        "make": ver(["make", "--version"]),
        "mpi": ver(["mpirun", "--version"]),
        "boost_program_options": find_boost(),
        "is_root": is_root,
        "root_note": (
            "Running without root. Everything installs under the work directory; steps that "
            "genuinely require privileges (persistence mode, clock locking) are skipped and "
            "recorded, not failed."
            if not is_root else "Running as root."
        ),
    }


# --------------------------------------------------------------------------
# trap 8: check the one library that needs root, before the build starts
# --------------------------------------------------------------------------

BOOST_HEADER = "boost/program_options.hpp"
BOOST_SEARCH = ["/usr/include", "/usr/local/include", "/opt/homebrew/include"]


def find_boost(extra_dirs: Optional[List[str]] = None) -> Optional[str]:
    """Path to the Boost.program_options header, or None.

    Only nvbandwidth needs it. Checking here means a missing Boost costs a line
    of output instead of a download, a configure step and a compiler error that
    names a template rather than a package.
    """
    for d in list(extra_dirs or []) + BOOST_SEARCH:
        p = os.path.join(d, BOOST_HEADER)
        if os.path.isfile(p):
            return p
    return None


def boost_prerequisite() -> Dict[str, Any]:
    """Report Boost as a prerequisite, never as a failure.

    A missing Boost costs exactly one tool, and the fallback for that tool --
    the PyTorch host-transfer measurement -- already ran. So this downgrades
    nvbandwidth rather than stopping the toolchain.
    """
    found = find_boost()
    if found:
        return {"ok": True, "header": found}
    return {
        "ok": False,
        "affects": ["nvbandwidth"],
        "install": {
            "debian": ["sudo", "apt", "install", "-y", "libboost-program-options-dev"],
            "rhel": ["sudo", "dnf", "install", "-y", "boost-devel"],
        },
        "reason": (
            "Boost.program_options headers were not found, so nvbandwidth cannot be built. "
            "Installing them needs root, which benchmark nodes usually do not grant, so this "
            "is reported now rather than after the download and configure steps have run. "
            "Everything else in the toolchain builds without it, and the PyTorch host-transfer "
            "measurement is unaffected -- what is lost is the second, independent reading that "
            "the cross-check compares against."
        ),
    }


# --------------------------------------------------------------------------
# trap 9: a built tool is not yet a runnable tool
# --------------------------------------------------------------------------

def _posix(path: str) -> str:
    """Backslashes to forward slashes. The commands below run on the node."""
    return path.replace("\\", "/")


def env_file_contents(work_dir: str, nccl_lib: str = "") -> str:
    """The shell file every new terminal has to source.

    Without it the nccl-tests binaries either fail to find libnccl.so.2 or --
    the nastier case -- resolve against an older system copy and abort with
    ``undefined symbol: ncclCommQueryProperties``. Both read as build failures
    and neither is one.
    """
    # posixpath, not os.path: this file is sourced by a shell on the benchmark
    # node, which is Linux even when the plan was printed on a Windows laptop.
    root = _posix(work_dir)
    lib = nccl_lib or posixpath.join(root, "third_party", "nccl", "build", "lib")
    bins = posixpath.join(root, "bin")
    return (
        "# Written by `nodebench setup`. Source this in every new terminal.\n"
        "#\n"
        "# LD_LIBRARY_PATH goes first on purpose: a system libnccl earlier on the\n"
        "# path will be picked instead, and the resulting `undefined symbol:\n"
        "# ncclCommQueryProperties` looks like a broken build rather than a\n"
        "# resolution order problem. Verify with:\n"
        "#     ldd $(command -v all_reduce_perf) | grep nccl\n"
        f'export NODEBENCH_HOME="{root}"\n'
        f'export LD_LIBRARY_PATH="{lib}:$LD_LIBRARY_PATH"\n'
        f'export PATH="{bins}:$PATH"\n'
    )


def verify_commands(work_dir: str) -> List[Dict[str, str]]:
    """Post-build checks that answer "will this actually run", not "did it build"."""
    return [
        {
            "command": f"source {posixpath.join(_posix(work_dir), 'bench_env.sh')}",
            "checks": "every new terminal needs this; nothing below works without it",
        },
        {
            "command": "ldd $(command -v all_reduce_perf) | grep nccl",
            "checks": (
                "which libnccl actually gets loaded. It must point inside the build tree. "
                "A path under /usr/lib means the system copy won and the collectives "
                "numbers will either be wrong or the binary will not start at all."
            ),
        },
        {
            "command": "nvbandwidth -l",
            "checks": "the tool starts and lists its testcases (skip if Boost was missing)",
        },
    ]


# --------------------------------------------------------------------------
# trap 3: cmake pin
# --------------------------------------------------------------------------

def ensure_cmake(min_ok: str = "3.20", max_exclusive: str = "4.0") -> Dict[str, Any]:
    """CUTLASS 3.9.x does not build under CMake 4.

    CMake 4 dropped compatibility with ``cmake_minimum_required`` below 3.5,
    which CUTLASS still declares. The fix is a user-local pinned install, not a
    patch to CUTLASS.
    """
    raw = probe_toolchain().get("cmake") or ""
    cur = None
    for tok in raw.split():
        if tok[:1].isdigit():
            cur = tok
            break
    if cur:
        major = int(cur.split(".")[0])
        if major >= int(max_exclusive.split(".")[0]):
            return {
                "action": "pin",
                "current": cur,
                "command": ["python3", "-m", "pip", "install", "--user", "cmake<4"],
                "reason": (
                    f"CMake {cur} removed support for the minimum-version declaration CUTLASS "
                    "still uses. Install a pinned cmake<4 into the user site directory and put "
                    "~/.local/bin ahead of it on PATH."
                ),
            }
        return {"action": "ok", "current": cur}
    return {
        "action": "install",
        "current": None,
        "command": ["python3", "-m", "pip", "install", "--user", "cmake<4"],
        "reason": "No cmake found.",
    }


# --------------------------------------------------------------------------
# trap 4: resumable download instead of git clone
# --------------------------------------------------------------------------

def fetch_commands(url: str, dest: str, mirror_prefix: str = "") -> List[List[str]]:
    """Resumable download. ``git clone`` on these repos fails at 90% and leaves nothing."""
    real = f"{mirror_prefix.rstrip('/')}/{url}" if mirror_prefix else url
    return [
        ["curl", "-fL", "-C", "-", "-o", dest, real],
        ["wget", "-c", "-O", dest, real],  # fallback if curl is absent
    ]


# --------------------------------------------------------------------------
# trap 5: poisoned build directory
# --------------------------------------------------------------------------

def check_build_dir(build_dir: str) -> Dict[str, Any]:
    """A cache written by a different toolchain fails as if the source were broken."""
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.isdir(build_dir):
        return {"action": "create"}
    if not os.path.isfile(cache):
        return {
            "action": "wipe",
            "reason": "build directory exists without a CMakeCache -- a previous configure died partway",
        }
    try:
        with open(cache, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return {"action": "wipe", "reason": "CMakeCache is unreadable"}

    home = ""
    for ln in text.splitlines():
        if ln.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
            home = ln.split("=", 1)[1].strip()
            break
    if home and not os.path.isdir(home):
        return {
            "action": "wipe",
            "reason": f"cache points at a source directory that no longer exists ({home})",
        }
    return {"action": "reuse"}


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------

def build_all(third_party: str, tools: Optional[List[str]] = None,
              mirror_prefix: str = "", dry_run: bool = True) -> Dict[str, Any]:
    """Produce (and optionally execute) a build plan.

    Defaults to ``dry_run=True``. A tool that is about to spend 90 minutes and
    several GB should say what it is going to do before it does it.
    """
    # The env file and the binaries live beside third_party/, not inside it.
    work = os.path.normpath(os.path.join(third_party, os.pardir))
    arch = detect_arch()
    tc = probe_toolchain()
    have = inventory(third_party)
    cmake = ensure_cmake()
    boost = boost_prerequisite()
    wanted = tools or list(TOOLS)

    steps: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for name in wanted:
        spec = TOOLS.get(name)
        if not spec:
            skipped.append({"tool": name, "reason": "unknown tool"})
            continue
        if have.get(name, {}).get("complete"):
            skipped.append({"tool": name, "reason": "already built -- rerun is free"})
            continue
        # Checked before the fetch, not after it. The failure mode this avoids
        # is a download plus a configure step plus a template error naming no
        # package, forty minutes into a build that was never going to finish.
        if not boost["ok"] and name in boost.get("affects", []):
            skipped.append({"tool": name, "reason": boost["reason"]})
            continue
        src = os.path.join(third_party, spec["dirname"])
        tgz = os.path.join(third_party, f"{name}.tar.gz")
        build_dir = os.path.join(src, "build")
        steps.append(
            {
                "tool": name,
                "estimated_minutes": spec["minutes"],
                "fetch": fetch_commands(spec["url"], tgz, mirror_prefix),
                "extract": ["tar", "-xzf", tgz, "-C", third_party],
                "build_dir_action": check_build_dir(build_dir),
                "arch_flag": arch.get("arch_flag"),
                "configure_hint": (
                    f"cmake -S {src} -B {build_dir} -DCMAKE_CUDA_ARCHITECTURES={arch.get('arch_flag')}"
                    if name != "nccl-tests"
                    # The 'a' suffix belongs on both halves of the gencode pair.
                    # Dropping it here would reintroduce trap #2 from this
                    # module's docstring in the one command that spells the
                    # architecture out by hand.
                    else f"make -C {src} -j CUDA_HOME=$CUDA_HOME NVCC_GENCODE="
                         f"'-gencode=arch=compute_{arch.get('arch_flag')},"
                         f"code=sm_{arch.get('arch_flag')}'"
                ),
            }
        )

    warnings = [w for w in [arch.get("warning"), tc.get("root_note") if not tc["is_root"] else None] if w]
    if cmake["action"] in ("pin", "install"):
        warnings.append(cmake["reason"])
    if not boost["ok"]:
        warnings.append(boost["reason"])
    if not tc.get("nvcc"):
        warnings.append(
            "nvcc is not on PATH. None of these tools can be built. This is not fatal for "
            "nodebench -- the default PyTorch backend needs no compiler."
        )

    total = sum(int(str(TOOLS[s["tool"]]["minutes"]).split("-")[-1]) for s in steps)
    plan = {
        "arch": arch,
        "toolchain": tc,
        "inventory": have,
        "cmake": cmake,
        "boost": boost,
        "steps": steps,
        "skipped": skipped,
        "warnings": warnings,
        "estimated_total_minutes_worst_case": total,
        "dry_run": dry_run,
        # Building the tools and being able to run them are separate problems,
        # and the second one is invisible until a fresh terminal fails.
        "env_file": os.path.normpath(os.path.join(third_party, "..", "bench_env.sh")),
        "env_file_contents": env_file_contents(work),
        "verify": verify_commands(work),
        "reminder": (
            "None of this is required. It exists only so `--backend native` and `--backend both` "
            "have something to cross-check against. The default PyTorch path needs no compiler, "
            "no root and no downloads."
        ),
    }
    return plan
