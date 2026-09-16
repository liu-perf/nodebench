# nodebench

[![ci](https://github.com/liu-perf/nodebench/actions/workflows/ci.yml/badge.svg)](https://github.com/liu-perf/nodebench/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Benchmark a whole multi-GPU node in 5 minutes, and get a report that says what the numbers mean.**

No compilation. No datasets. No root. No CUDA toolkit. If you have PyTorch and NVIDIA drivers, you have everything.

```bash
pip install -e .
nodebench doctor      # 10 seconds: can this machine produce trustworthy numbers?
nodebench run         # 5 minutes: measure, then write report.html
```

📊 **[See an example report](examples/sample_report.html)** — self-contained HTML, opens with no network and no JavaScript. Look at it before installing anything.

<!-- Add a screenshot here once you have one:
<p align="center"><img src="docs/img/report.png" alt="Example report" width="760"></p>
-->

---

## Why another GPU benchmark

Most GPU benchmarks answer *"how fast is this card"*. Almost nobody buys one card. What breaks in practice is the **node**: the interconnect, the NUMA placement, the host memory bandwidth, whether GPU-to-GPU traffic is quietly bouncing through system RAM.

nodebench measures the node, and it does three things that microbenchmarks generally do not:

**1. It samples a control group before every run.** Twelve seconds of idle, recorded, hashed, and reported alongside the workload. Without a noise floor a measurement is unfalsifiable — and a node that already has someone else's job on it looks exactly like a slow node.

**2. It measures the same quantity two independent ways and reports the divergence.** PyTorch on one side, the native C++ toolchain (nccl-tests, nvbandwidth, BabelStream, CUTLASS) on the other. Agreement does not prove either is right, but disagreement proves one is wrong — which is worth more than another decimal place. See [`docs/methodology.md`](docs/methodology.md).

**3. It states what it did not measure.** Every report ends with an auto-generated section listing which modules did not run, why, and what each measurement structurally cannot tell you. A reader can always distinguish *"measured and fine"* from *"never looked"*.

---

## What it finds

| Question | How | Why it matters |
|---|---|---|
| Is GPU-to-GPU traffic taking a peer path or bouncing through host RAM? | Ratio of all-reduce busbw to H2D bandwidth | Every standard probe answers only whether the *hardware* supports P2P. Software P2P is invisible to all of them. [Explained here](docs/p2p-explained.md) |
| Does the node feed all its GPUs at once? | Isolated H2D per card, then all cards concurrently | The retention % exposes under-populated memory channels. No single-card test can show it |
| Same 4 GPUs, two different placements — how much does it cost? | Auto-generated `4gpu_same_numa` vs `4gpu_cross_numa` | Same card count, same workload, only topology differs. Often larger than the gap between two GPU models |
| Will this torch build actually run on this card? | `compute_cap` × CUDA version × `torch.cuda.get_arch_list()` | The third check is the one people forget. A CUDA 12.8 wheel can ship without an `sm_120` cubin |
| Is the training job input-bound? | `data_time / time` from the mmengine log | Above 15% you are benchmarking your storage, and every scaling number derived from it is void |
| Was the card throttling? | NVML throttle bitmask, decoded per sample | "It got slower over time" is a measurement, not a vibe |

---

## Install

```bash
git clone https://github.com/liu-perf/nodebench
cd nodebench
pip install -e .
```

Requires Python 3.9+, an NVIDIA driver, and a working PyTorch with CUDA. `torch` is deliberately **not** a declared dependency — you already have a specific CUDA build, and pip pulling a different one would break your machine.

---

## Use

### Check the machine first

```bash
nodebench doctor
```

```
nodebench 0.1.0 doctor

  GPU 0  NVIDIA GeForce RTX 5090  cc 12.0  driver 570.86.16  32607 MiB
  GPU 1  NVIDIA GeForce RTX 5090  cc 12.0  driver 570.86.16  32607 MiB

  torch 2.7.0+cu128 built for CUDA 12.8, NCCL 2.26.2
  compiled architectures: sm_75, sm_80, sm_86, sm_90, sm_100, sm_120

  [ok  ] sm_120 supported by this torch build
  [WARN] NCCL 2.26.2 on Blackwell: 2.21 is the practical minimum, newer is better

  topology: 2 GPUs, no NVLink, 1 NUMA group
    1gpu               gpus [0]     worst link X    (single-card baseline)
    2gpu               gpus [0, 1]  worst link PIX
    2gpu_all           gpus [0, 1]  worst link PIX  (full node)

Result: ready. Run `nodebench run` next.
```

### Measure

```bash
nodebench run --label "my-node"           # everything, ~5 min
nodebench run --quick                     # fewer iterations, first look
nodebench run --modules flops,membw       # just compute and memory
nodebench run -c my-node.yaml             # reproducible config
```

Output lands in `nodebench_out/run_<timestamp>/`:

```
report.html          self-contained, no network, no JS libraries
results.json         everything the report was built from
raw/
  environment.json   driver, torch build, arch list, compat findings
  topology.json      nvidia-smi topo -m, parsed
  idle_baseline.csv  the control group
  monitor.csv        NVML samples with provenance headers
  flops.json  membw.json  pcie.json  nccl.json  p2p.json
```

Every file in that tree is md5-hashed into the report's manifest, so a reader can confirm the raw output matches the numbers.

### Watch someone else's workload

The command that gets used most in practice — a training job is already running and you need to know what the hardware is doing:

```bash
nodebench monitor -i 1.0 -- python train.py
nodebench monitor -i 0.5                    # attach to whatever is already running, Ctrl-C to stop
```

Samples are binned by **which set of GPUs was actually busy**, not by wall-clock window. That distinction matters: slicing by time silently folds the model-load H2D burst into the training phase, an error of two orders of magnitude that produces a perfectly reasonable-looking number.

### Turn a training log into throughput

```bash
nodebench parse-mm train.log --gpus 4 --batch-per-gpu 2 --dataset-size 118287

# several runs at once -> a scaling table
nodebench parse-mm \
  logs/1gpu.log:1gpu:1:2 \
  logs/2gpu.log:2gpu:2:2 \
  logs/4gpu_same.log:4gpu_same_numa:4:2 \
  logs/4gpu_cross.log:4gpu_cross_numa:4:2 \
  -o scaling.json
```

Warmup is discarded, the **median** step time is used, and `data_time` is reported as a first-class field rather than hidden.

### Optional: the native toolchain

```bash
nodebench setup --native        # prints a plan; nothing is downloaded
nodebench run --backend both    # measure both ways, report the divergence
```

This takes 30–90 minutes and is never required. It exists so `--backend both` has something to cross-check against. The builder encodes nine specific traps (the `sm_90a` suffix, the CMake 4 incompatibility, poisoned CMake caches, resumable downloads, …) — each is documented in [`nodebench/setup/build.py`](nodebench/setup/build.py) with the failure it prevents.

---

## Configuration

Everything machine-specific lives in one YAML file. Nothing in the code hardcodes a path, a GPU id, or a size.

```bash
cp nodebench.example.yaml my-node.yaml
nodebench run -c my-node.yaml
```

```yaml
node:
  label: "2x5090-workstation"

bench:
  modules: [flops, membw, pcie, nccl]
  nccl:
    sizes: [16777216, 67108864, 268435456, 1073741824]
    gpu_sets: null        # null = derive from topology

monitor:
  idle_baseline_s: 12     # the control group. Do not set this to 0.
```

---

## Reading the numbers

Three things that trip people up, all documented in full under [`docs/`](docs/):

**`busbw`, never `algbw`.** `bytes / seconds` is not comparable across collectives or GPU counts. All-reduce carries a `2(N-1)/N` correction; all-gather, reduce-scatter and all-to-all carry `(N-1)/N`. Quoting algbw across different GPU counts compares two different quantities.

**NVML's PCIe counter is a ~20 ms windowed average.** It cannot be multiplied by elapsed time to get a transfer volume, and its peaks are 20 ms peaks, not link capacity. Only means across equal-length phases are comparable. → [`docs/nvml-pitfalls.md`](docs/nvml-pitfalls.md)

**`utilization.gpu` is not SM occupancy.** It is the fraction of the sampling period with at least one kernel resident. A card can read 100% while using a few percent of its SMs. nodebench calls the field `busy_pct` for exactly this reason.

---

## Project layout

```
nodebench/
  probe/      compat.py  topology.py  p2p.py      what is this machine?
  bench/      flops  membw  pcie  nccl            measurement, returns dicts only
  monitor/    nvml.py  binning.py                 NVML sampling + phase binning
  parse/      native.py  mmengine.py              pure str -> dict parsers
  analyze/    ring_theory  scaling  crosscheck  baseline    interpretation
  report/     manifest  caveats  html             data + template, never handwritten
  setup/      build.py                            optional native toolchain
recipes/      llm_lora/                           real workloads, synthetic data
docs/         methodology  p2p-explained  nvml-pitfalls  compat-matrix
```

Four rules the code obeys:

1. Bench modules return plain dicts. They never print and never format.
2. Parsers are pure functions — `str` in, `dict` out, no IO. Each is testable against a saved fixture of real tool output.
3. No path, GPU id, or size is hardcoded anywhere outside the config.
4. Anything not covered is recorded, not omitted.

---

## Limitations

Stated here as well as in every report:

- **One node.** No InfiniBand/RoCE fabric, no cross-node NCCL tuning, no rail topology.
- **Minutes, not hours.** Sustained thermal behaviour over a real training run is out of scope.
- **Throughput, not correctness.** A fast kernel producing wrong numbers passes every test here.
- **NVIDIA only.** ROCm and Intel GPUs are not supported.
- **The P2P verdict is inferred, not read from a register.** It is reported with a confidence level for that reason, and reports `inconclusive` rather than guessing.

---

## License

MIT. See [LICENSE](LICENSE).
