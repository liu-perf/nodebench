"""NVML sampler with the counter semantics written down instead of assumed.

Two traps are baked into NVML, and both silently produce a plausible-looking
number that means something other than what you think.

Trap 1 -- ``nvmlDeviceGetPcieThroughput`` is a windowed average
--------------------------------------------------------------
It does not return "bytes since last call". It returns the average rate over an
internal window of roughly 20 ms. So if you sample once a second, each sample
describes about 2% of that second and says nothing about the other 98%.

Consequences, all of which have been shipped in real reports:

  * You cannot multiply a sample by the interval to get transferred volume.
    There is no such thing as total volume from this counter.
  * A peak sample is the peak of a 20 ms window, not a link capacity.
  * Comparing means across phases of equal length is fair. Comparing sums,
    peaks, or integrals is not.

Trap 2 -- ``utilizationRates().gpu`` is not SM occupancy
--------------------------------------------------------
NVIDIA documents it as the fraction of the sample period during which *at least
one* kernel was resident. One kernel using a single SM out of 170 for the whole
period reports 100%. It is an "is the GPU busy" flag with a percent sign, and it
is genuinely useful for phase segmentation -- which is what
``monitor/binning.py`` uses it for -- but it is not utilisation. The column is
named ``busy_pct`` here rather than ``sm_pct`` so nobody can misread it later.

Also recorded, because they change what a benchmark result means:
temperature, SM/memory clocks, and the throttle-reason bitmask. A node that
looks 12% slow is often a node that spent 12% of the run clocked down, and
without the throttle column you will go looking for a software cause.

Every run writes an actual measured sampling interval, not the requested one.
Python sleep plus NVML query latency drifts, and a 1.0 s request routinely
becomes 1.3 s. Any per-second reasoning downstream must use the real number.
"""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# nvmlDeviceGetCurrentClocksThrottleReasons bitmask
THROTTLE_REASONS = [
    (0x0000000000000001, "gpu_idle"),
    (0x0000000000000002, "applications_clocks_setting"),
    (0x0000000000000004, "sw_power_cap"),
    (0x0000000000000008, "hw_slowdown"),
    (0x0000000000000010, "sync_boost"),
    (0x0000000000000020, "sw_thermal_slowdown"),
    (0x0000000000000040, "hw_thermal_slowdown"),
    (0x0000000000000080, "hw_power_brake_slowdown"),
    (0x0000000000000100, "display_clock_setting"),
]

# Reasons that actually cost you performance (gpu_idle and the clock-setting
# ones do not).
PERF_LIMITING = {
    "sw_power_cap",
    "hw_slowdown",
    "sw_thermal_slowdown",
    "hw_thermal_slowdown",
    "hw_power_brake_slowdown",
}

FIELDS = ["tx_gbs", "rx_gbs", "busy_pct", "mem_io_pct", "power_w", "temp_c",
          "sm_clk_mhz", "mem_clk_mhz", "mem_used_mib", "throttle"]


def decode_throttle(mask: int) -> List[str]:
    return [name for bit, name in THROTTLE_REASONS if mask & bit]


class NvmlMonitor:
    """Background sampler. Use as a context manager or start()/stop().

    >>> with NvmlMonitor(Path("out"), interval_s=1.0) as mon:
    ...     mon.mark("idle_baseline")
    ...     time.sleep(12)
    ...     mon.mark("flops")
    ...     run_workload()
    >>> mon.summary()
    """

    def __init__(
        self,
        outdir,
        interval_s: float = 1.0,
        devices: Optional[List[int]] = None,
        filename: Optional[str] = None,
    ):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.interval = float(interval_s)
        self.devices = devices
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.outdir / (filename or f"monitor_{stamp}.csv")
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._marks: List[Dict[str, Any]] = []
        self._rows = 0
        self._t0: Optional[float] = None
        self._t1: Optional[float] = None
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "NvmlMonitor":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> "NvmlMonitor":
        try:
            import pynvml  # noqa: F401
        except ImportError:
            self.error = "pynvml not installed (pip install nvidia-ml-py); monitoring disabled"
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 4 + 5)

    def mark(self, label: str) -> None:
        """Timestamp a phase boundary. Purely advisory -- binning does not need it."""
        with self._lock:
            self._marks.append({"label": label, "t": time.time(), "iso": datetime.now().isoformat()})

    # -- sampling ----------------------------------------------------------
    def _loop(self) -> None:
        import pynvml

        try:
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            ids = self.devices if self.devices is not None else list(range(n))
            handles = [(i, pynvml.nvmlDeviceGetHandleByIndex(i)) for i in ids]

            header = ["timestamp", "elapsed_s"]
            for i, _ in handles:
                header += [f"gpu{i}_{f}" for f in FIELDS]

            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                # Provenance ahead of the header: what the counters mean travels
                # with the data, not in a separate doc. Written raw rather than
                # through the csv writer, which would quote any comment that
                # happens to contain a comma and leave the rest bare.
                for line in (
                    f"# nodebench monitor  requested_interval_s={self.interval}",
                    "# pcie tx/rx: nvmlDeviceGetPcieThroughput, ~20ms windowed AVERAGE "
                    "-- means only, never sums, peaks or integrals",
                    "# busy_pct: nvmlDeviceGetUtilizationRates().gpu -- fraction of the "
                    "period with >=1 kernel resident. NOT SM occupancy.",
                ):
                    fh.write(line + "\n")
                w.writerow(header)

                # Prime the PCIe counters; the first read initialises them.
                for _, h in handles:
                    try:
                        pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                        pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                    except Exception:
                        pass
                time.sleep(self.interval)

                self._t0 = time.time()
                while not self._stop.is_set():
                    now = time.time()
                    row: List[Any] = [datetime.now().isoformat(), round(now - self._t0, 3)]
                    for _, h in handles:
                        row += self._sample(pynvml, h)
                    w.writerow(row)
                    fh.flush()
                    self._rows += 1
                    self._t1 = now
                    # Subtract query cost so drift does not accumulate.
                    slack = self.interval - (time.time() - now)
                    self._stop.wait(max(0.0, slack))
            pynvml.nvmlShutdown()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    @staticmethod
    def _sample(pynvml, h) -> List[Any]:
        def safe(fn, default=None):
            try:
                return fn()
            except Exception:
                return default

        tx = safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES), 0)
        rx = safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES), 0)
        util = safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(h))
        pw = safe(lambda: pynvml.nvmlDeviceGetPowerUsage(h), 0)
        temp = safe(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU), 0)
        smclk = safe(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM), 0)
        mclk = safe(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM), 0)
        mem = safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(h))
        thr = safe(lambda: pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h), 0)

        # NVML reports PCIe throughput in KB/s.
        return [
            round((tx or 0) / 1e6, 4),
            round((rx or 0) / 1e6, 4),
            util.gpu if util else 0,
            util.memory if util else 0,
            round((pw or 0) / 1000.0, 1),
            temp or 0,
            smclk or 0,
            mclk or 0,
            int((mem.used if mem else 0) / 1024 / 1024),
            "|".join(decode_throttle(thr or 0)),
        ]

    # -- summary -----------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """What actually happened, including the real sampling interval."""
        dur = (self._t1 - self._t0) if (self._t0 and self._t1) else 0.0
        actual = (dur / (self._rows - 1)) if self._rows > 1 else None
        s: Dict[str, Any] = {
            "csv": str(self.path),
            "rows": self._rows,
            "duration_s": round(dur, 2),
            "requested_interval_s": self.interval,
            "actual_interval_s": round(actual, 4) if actual else None,
            "marks": self._marks,
            "error": self.error,
        }
        if actual and abs(actual - self.interval) / self.interval > 0.05:
            s["interval_warning"] = (
                f"Requested {self.interval:.2f} s but sampled every {actual:.3f} s "
                f"({(actual / self.interval - 1) * 100:+.0f}%). Any per-second reasoning "
                "downstream must use the measured interval."
            )
        return s
