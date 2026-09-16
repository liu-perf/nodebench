# Methodology

What makes a benchmark number defensible rather than merely printed.

Nothing here is exotic. It is the set of habits that survive contact with someone asking "how do you know?" — and each one exists because a number produced without it turned out to be wrong.

---

## 1. Every run has a control group

Twelve seconds of idle sampling before any workload starts. Recorded to its own CSV, hashed into the manifest, reported alongside the results.

**Why:** a measurement without a noise floor is unfalsifiable. 0.4 GB/s of PCIe traffic during a workload is either signal or the monitoring agent — there is no way to tell from the workload sample alone.

**What it catches, in order of how often:**

1. Another user's job already on the node. The power floor gives it away immediately
2. A zombie process holding CUDA context and VRAM
3. A card stuck in a high clock state, or stuck throttled from a previous job
4. Background PCIe traffic from monitoring agents or a mounted network filesystem

Any of these contaminates every number that follows. The failure mode is not a crash — it is a plausible result that will not reproduce next week, and by then nobody remembers what else was running.

**How it is reported:** not as a footnote but as a ratio.

> "42 GB/s during the workload" → weak
> "42 GB/s during the workload, against a 0.3 GB/s idle floor — 140× signal-to-noise" → defensible

Below 3× signal-to-noise, nodebench refuses to report the value as a workload measurement, because it isn't one.

---

## 2. Two independent implementations, and the divergence is published

The same physical quantity, measured two ways that share no code:

| Quantity | PyTorch backend | Native backend |
|---|---|---|
| FLOPS | `torch.matmul` on a square GEMM | cuBLAS / CUTLASS profiler |
| Memory bandwidth | tensor copy/mul/add/triad kernels | BabelStream |
| PCIe | pinned buffer + `cudaMemcpyAsync` | nvbandwidth `host_to_device_memcpy_ce` |
| Collectives | `torch.distributed`, CUDA-event timed | nccl-tests `all_reduce_perf` |

**Why:** any single pipeline can be silently wrong — a unit slip, a missing synchronise, warmup left in the timed region, a buffer small enough to fit in L2. The defence is not to be more careful. It is to require two implementations to agree.

Agreement does not prove either is right. **Disagreement proves one is wrong**, and that is worth far more than a third decimal place.

The bar is 5%. Beyond that, something structural differs between the two — different shape, different dtype, different buffer size, one of them timing something the other isn't. Beyond 15%, nodebench marks the row `conflict` and says in the report that neither number should be published until you know which one is broken.

Two full independent re-runs of the same configuration on the same node landed within **0.2%** — that is the repeatability floor this method achieves in practice.

---

## 3. Best and median, never just one

Every timed measurement reports both, plus the coefficient of variation and the repetition count.

**Why:** reporting only the best invites cherry-picking, and everyone knows it. Reporting only the median hides the fact that the hardware *can* go faster under good conditions, which is real information. Reporting both, with the CV, makes the spread visible and makes cherry-picking impossible in either direction.

A high CV is itself the finding: it means the card was not in a steady state — thermal drift, clock instability, or something else on the node.

For inference latency, the convention is 3 repetitions, take-min: the minimum is the closest available estimate of the workload with no interference, and interference is exactly what you are trying to exclude.

---

## 4. Theory reconciliation

Every collective measurement is checked against what the ring algorithm predicts.

Ring all-reduce runs in two phases — reduce-scatter then all-gather — each of N−1 steps, each moving 1/N of the buffer. Per rank:

```
bytes_on_the_wire = 2 × (N−1)/N × buffer_bytes
```

Divide measured by predicted and read the ratio:

- **far below 1** — something is bottlenecking, or the measurement missed part of the transfer
- **near 1** — model and machine agree; both are probably right
- **far above 1** — the model is wrong for this hardware (an NVLink/NVSwitch path the ring model doesn't describe), or a counter is being read with the wrong units

A measured bandwidth on its own is unfalsifiable — 40 GB/s could be excellent or terrible. A predicted value makes it checkable.

The same factor is the busbw correction, which is why nodebench defines them in one place rather than two that can drift apart.

---

## 5. `busbw`, never `algbw`

`algbw = bytes / seconds` is not comparable across collectives or across GPU counts. The bus-bandwidth correction makes it comparable:

| Collective | Factor |
|---|---|
| all-reduce | `2(N−1)/N` |
| all-gather, reduce-scatter, all-to-all | `(N−1)/N` |
| broadcast, reduce | `1` |

nodebench reports both, so the conversion stays auditable, and states in the report which one to compare. The most common multi-GPU reporting error is quoting algbw as if it were busbw — or applying the correction twice. The reconciliation check in `analyze/ring_theory.py` catches both.

---

## 6. Bin by active GPU set, not by time window

When attributing monitored samples to workload phases, nodebench groups rows by **which set of GPUs was above the busy threshold**, not by wall-clock slice.

**Why:** a training run's timeline is `[model load → warmup → steady state]`, and model load is a large H2D burst. Slicing by time inevitably includes part of it in the "training" window. The resulting PCIe average is wrong by two orders of magnitude — and it looks completely reasonable, which is what makes it dangerous.

Binning by active set separates them automatically: model load has a different active-GPU signature than distributed training, so they land in different bins without anyone having to guess where the boundary was.

One further detail: statistics within a bin are averaged over the **active cards only**. Averaging over all eight when one is working produces a number that looks like utilisation and is not — it is a phase tag with a percent sign on it.

---

## 7. The controlled variable in multi-GPU scaling

nodebench auto-generates a `4gpu_same_numa` and a `4gpu_cross_numa` set whenever the topology permits.

Same card count. Same workload. Same everything except placement. Whatever the difference is, it is attributable to topology, because nothing else changed. On a dual-root node the gap is routinely larger than the gap between two GPU models — and it is invisible to any benchmark that just uses `CUDA_VISIBLE_DEVICES=0,1,2,3`.

Efficiency is always computed against a **measured baseline on the same node**, never a datasheet. Comparing a 4-GPU run to a theoretical single-GPU number folds the node's own single-card behaviour into the "scaling" figure, and the figure is then no longer about scaling.

**Read speedup and per-step inflation together.** Throughput rises with the global batch, so a job can show 3.2× on 4 GPUs while every individual step got 25% slower. That 25% is the communication cost, and speedup hides it.

---

## 8. Provenance for every artifact

Path, byte count, line count, md5, and the command that produced it. Written into the report.

md5 because this is tamper-evidence for honest work, not security. It answers "is this the same file the report was generated from", cheaply, which is the only question being asked.

The habit is more valuable than the hashes: **if a number in the report has no artifact behind it, writing the manifest is where you find out.**

Empty artifacts are recorded as empty rather than skipped. A tool that ran and produced nothing is a result.

---

## 9. What was not covered is a section, not an omission

Every report ends with an auto-generated section derived from the configuration that actually ran:

- what each executed module structurally **cannot** tell you
- which modules did **not** run, and why — "no second GPU" and "user disabled it" are different facts
- what is outside the tool entirely (multi-node fabric, sustained thermals, numerical correctness)

**Why:** the most common way a benchmark misleads is not a wrong number. It is a correct number read as if it answered a broader question than it does. One square GEMM becomes "the card's FP16 performance". One message size becomes "the interconnect". Sixty seconds becomes "sustained".

A reader must always be able to distinguish *"measured and it was fine"* from *"we never looked"*. Generating the section from the config rather than trusting the author to remember is the only way that stays true on the twentieth run.

---

## 10. Zero datasets, on purpose

Every workload in `recipes/` uses synthetic input — random token IDs at the configured sequence length.

**Why:** a real dataset makes the benchmark unrunnable by a stranger (tens of GB, preprocessing, a specific directory layout) and introduces storage and CPU decode as uncontrolled variables. For measuring the GPUs it adds nothing: the model does not care whether the token IDs mean anything. The compute, the memory traffic and the gradient all-reduce are identical.

What synthetic input does **not** measure is the input pipeline. That is stated in the output rather than glossed over — and when a real training log is available, `parse-mm` surfaces `data_time / time` precisely so the input pipeline can be judged separately.

---

*Every point above is implemented and can be read: [`analyze/`](../nodebench/analyze/) for reconciliation and scaling, [`report/manifest.py`](../nodebench/report/manifest.py) for provenance, [`report/caveats.py`](../nodebench/report/caveats.py) for the limitations generator, [`monitor/binning.py`](../nodebench/monitor/binning.py) for phase binning.*
