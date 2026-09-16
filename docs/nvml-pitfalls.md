# Four NVML counters that do not mean what their names say

Every GPU monitoring script ever written is thirty lines of pynvml in a `while True` loop. Mine was too. Then I tried to reconcile what it printed against what the hardware was physically capable of, and the numbers did not close.

Four counters were responsible. All four are documented correctly by NVIDIA. All four are almost universally misread, because the obvious reading is the wrong one.

---

## 1. `nvmlDeviceGetPcieThroughput` is a windowed average, not a byte counter

```python
tx = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES)
```

The name says `TX_BYTES`. The natural reading is "bytes since I last asked", which would let you sum the samples and get a transfer volume.

It is not that. It is **the average throughput over an internal ~20 ms window**, in KB/s, and it has no relationship to when you last called it. Three consequences, each of which has produced a wrong number in a published report:

**You cannot integrate it.** Multiplying the reading by elapsed time to get GB transferred is measuring a rate against the wrong interval. The internal window is 20 ms; your sampling interval is 1000 ms. You are extrapolating 2% of the timeline across 100% of it.

**A peak is a 20 ms peak.** If you sample at 1 Hz and see 45 GB/s, that is the average over one 20 ms window that happened to be in flight when you asked. It is not the link's sustained capacity, and it is not the transfer's average rate. Quoting it as "peak PCIe bandwidth" overstates by however bursty the traffic was.

**Only means across equal-length phases are comparable.** Two phases of the same duration, sampled at the same interval, produce means you can compare honestly. Anything else — different durations, different intervals, mean-vs-peak — compares two different quantities.

What you can do with it: compare phases, detect that traffic is happening at all, and establish a noise floor. What you cannot do: state how many gigabytes moved.

---

## 2. `utilization.gpu` is not SM occupancy

This is the single most consequential misreading in GPU monitoring, and it has its own row in every dashboard ever built.

```python
util = pynvml.nvmlDeviceGetUtilizationRates(h)
util.gpu    # NOT "percent of the GPU in use"
```

The actual definition: **the fraction of the sampling period during which at least one kernel was resident on the device.**

One kernel. Using one SM out of 170. For the whole period. That reads **100%**.

So `utilization.gpu = 100%` means "the GPU was never idle". It does not mean the GPU was busy in any sense a person cares about. A memory-bound kernel at 6% of peak FLOPS and a perfectly tuned GEMM both read 100%, and the counter has no way to tell you which one you have.

nodebench names the field `busy_pct` and repeats the definition in the CSV header, the report, and the docs, because the name `utilization` is actively misleading and there is no way to fix it other than not using it.

**What to use instead:** occupancy needs Nsight Compute or CUPTI. If you only have NVML, `busy_pct` is a *phase tag* — useful for answering "which GPUs were participating in this workload", which is exactly what nodebench uses it for when binning samples.

---

## 3. Your sampling interval is not your sampling interval

```python
while True:
    sample()
    time.sleep(1.0)
```

This does not sample every second. It samples every `1.0 + query_time + scheduler_jitter` seconds, and the query time is not small: reading eight fields from eight GPUs is dozens of NVML round trips.

Measured on a real 8-GPU node, nominal interval 1.000 s:

| Run | Actual mean interval |
|---|---|
| A | **1.376 s** |
| B | **1.342 s** |

A 34–38% error. Every per-second calculation downstream inherits it, and it inherits it *silently*, because nothing in the CSV records what the interval actually was.

Two fixes, both in `nodebench/monitor/nvml.py`:

```python
t0 = time.time()
row = sample_all()
elapsed = time.time() - t0
time.sleep(max(0.0, self.interval - elapsed))   # subtract the query cost
```

and then, more importantly, **measure the result and report it**:

```python
actual = duration / (rows - 1)
if abs(actual - requested) / requested > 0.05:
    warn(f"Requested {requested:.2f}s but sampled every {actual:.3f}s")
```

Compensating is nice. Reporting the residual is what makes the data defensible — the number is in the CSV header and in the report, so nobody downstream has to assume.

---

## 4. Throttle reasons are a bitmask, and most of the bits are boring

```python
mask = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)
```

`GpuIdle` is a throttle reason. So is `ApplicationsClocksSetting`. Neither means anything is wrong; an idle GPU throttling is an idle GPU.

The bits that mean the hardware is limiting your performance:

| Bit | Meaning |
|---|---|
| `SwPowerCap` | Driver hit the power limit |
| `HwSlowdown` | Hardware emergency slowdown |
| `HwThermalSlowdown` | Too hot |
| `HwPowerBrakeSlowdown` | External power brake asserted |
| `SyncBoost` | Clocks pinned to match other GPUs in a sync-boost group |

nodebench decodes the mask to names, keeps the full list in the CSV, and counts only the performance-limiting subset in the summary. "It got slower over the run" then becomes a checkable claim with a bit behind it, instead of an impression.

---

## The pattern

All four traps share a shape: **the counter is correct and the name suggests something else.** `TX_BYTES` is not bytes. `utilization` is not utilisation. `sleep(1.0)` is not one second. `throttleReasons != 0` is not throttling.

The general defence is cheap: write the counter's semantics into the artifact itself. nodebench emits provenance lines into the CSV before the header, so the definition travels with the data:

```
# nodebench monitor v0.1.0
# tx_gbs,rx_gbs: NVML PCIe throughput. ~20ms windowed AVERAGE, not bytes-since-last-call.
#   Cannot be integrated to a volume. Peaks are 20ms peaks, not link capacity.
# busy_pct: NVML utilization.gpu. Fraction of period with >=1 kernel resident.
#   NOT SM occupancy. A single kernel on one SM reads 100%.
# requested_interval_s: 1.0  (actual measured interval is in monitor_summary.json)
timestamp,gpu0_tx_gbs,gpu0_rx_gbs,gpu0_busy_pct,...
```

Six months later, when someone opens that CSV without the surrounding context, the caveats are still attached to the numbers. That is the whole trick.

---

*Implementation: [`nodebench/monitor/nvml.py`](../nodebench/monitor/nvml.py) and [`nodebench/monitor/binning.py`](../nodebench/monitor/binning.py).*
