# Four tools told me P2P was disabled. All four were wrong.

*Or: why "can these GPUs talk to each other" and "are these GPUs talking to each other" are different questions.*

---

## The setup

Two GPUs in one box. A data-parallel training job scaling badly. The obvious suspect is peer-to-peer: if the cards cannot DMA into each other's memory, every gradient exchange stages through host RAM, and the interconnect cost roughly doubles.

So you check. There are four standard ways, and you run all four.

**1. `nvidia-smi topo -p2p r`**

```
        GPU0    GPU1
 GPU0   X       CNS
 GPU1   CNS     X
```

`CNS` = "Chipset Not Supported". No peer access.

**2. `p2pBandwidthLatencyTest`** (from cuda-samples)

```
Device=0 CANNOT Access Peer Device=1
Device=1 CANNOT Access Peer Device=0
```

No peer access.

**3. `nvbandwidth`**

```
Running device_to_device_memcpy_read_ce.
Waived.
```

Every device-to-device test waived. No peer access.

**4. `cudaDeviceCanAccessPeer`**

```python
torch.cuda.can_device_access_peer(0, 1)   # False
```

No peer access.

Four independent tools, unanimous. Case closed — except the scaling numbers don't agree.

---

## The part that doesn't fit

Measure the actual bandwidths:

| Measurement | Value |
|---|---|
| Host-to-device (pinned, per card) | 56.4 GB/s |
| All-reduce busbw, 2 GPUs | 40.7 GB/s |

If peer traffic were bouncing through host memory, every byte would cross PCIe twice — once up, once down — and share the host link with the other card doing the same thing. The achievable all-reduce busbw would land somewhere well under half the H2D figure. Instead it is **72% of it**.

That number is not reachable through a host bounce. Something is moving data between those cards on a path that none of the four probes can see.

---

## The two questions

Here is the distinction that took me too long to see:

> **Capability:** *can* these two GPUs address each other's memory directly, through a hardware path the driver exposes as CUDA peer access?
>
> **Behaviour:** *is* traffic between these two GPUs actually taking a fast path in practice?

All four probes answer the first question. Only the first question. They inspect a capability flag, and the flag is genuinely false.

But CUDA peer access is not the only way data gets from one GPU to another quickly. NCCL, the driver, and platform-specific routing can move data over paths that never set that flag:

- **PCIe peer routing through the root complex or a PLX switch**, arranged by the driver rather than exposed as CUDA P2P
- **NCCL's own transport selection** — it picks a transport at init time and does not ask `cudaDeviceCanAccessPeer` for permission
- **Copy-engine paths** that stage through a small pinned buffer while still avoiding a full round trip through system memory
- **Virtualisation and platform layers** that reroute transparently

Every one of these makes multi-GPU traffic fast. None of them flips the capability flag. So the flag being false tells you nothing about whether your training job is going to scale.

**The capability probes are not lying. They are answering a question you did not ask.**

---

## Measuring behaviour instead

If the flag can't tell you, measure the outcome. The discriminator nodebench uses:

```
ratio = allreduce_busbw / h2d_bandwidth
```

Both are measured on the same node in the same run, so machine-specific factors cancel.

| Ratio | Reading |
|---|---|
| **≥ 0.55** | A peer path is being used. This is not reachable through host staging |
| **≤ 0.30** | Consistent with a host bounce and nothing better |
| **between** | **Inconclusive.** Reported as inconclusive, not guessed |

Why those thresholds and not others: a host bounce costs each byte two PCIe crossings on a link shared with the other cards doing the same, so its ceiling sits well below half of H2D. A direct peer path shares the link too, but crosses it once. The gap between the two regimes is wide, and the band in the middle is where a measurement genuinely cannot distinguish them — so the honest output there is "I don't know", printed in those words.

Calibrated against a real node before and after fixing its NCCL configuration:

| | H2D | All-reduce busbw | Ratio | Verdict |
|---|---|---|---|---|
| Before | 37.0 GB/s | 8.1 GB/s | **0.22** | host bounce |
| After | 56.4 GB/s | 40.7 GB/s | **0.72** | peer path |

Same hardware. Same four capability probes returning the same "no P2P" both times. The framework's verdict changed because the behaviour changed, which is the thing anyone actually cares about.

---

## What the classifier reports

`nodebench` combines the capability evidence with the measured ratio and returns one of six modes:

| Mode | Meaning |
|---|---|
| `hardware` | NVLink or CUDA peer access is available and the bandwidth confirms it is being used |
| `software` | Every capability probe says no, and the bandwidth says yes anyway. **This is the interesting one** |
| `host_bounce` | Capability absent, bandwidth consistent with host staging. Scaling will suffer and you now know why |
| `disabled` | Capability present but bandwidth says it is not being used. Usually a config problem — worth fixing, because the hardware is already there |
| `inconclusive` | The ratio landed in the middle band. Stated, not guessed |
| `unknown` | Not enough measurements to say anything |

The `disabled` case is worth dwelling on: it means the node *has* the fast path and is not using it. That is the highest-value finding the classifier can produce, and no capability probe can produce it, because from a capability probe's point of view everything is fine.

---

## The general lesson

A capability probe answers *what the hardware permits*. A benchmark answers *what actually happened*. When they disagree, the benchmark is describing your workload and the probe is describing a datasheet.

This generalises past P2P:

- `nvidia-smi` reports a **power limit**; what matters is the throttle bitmask during the run
- A card is advertised at **1008 GB/s** memory bandwidth; what matters is achieved streaming throughput
- PCIe reports **Gen5 x16**; what matters is whether pageable memory is forcing a staging copy that makes it measure like Gen2
- `utilization.gpu` reports **100%**; what matters is that the field means "a kernel was resident", not "the SMs were busy"

In every case the configuration-space answer is available in a second and the behaviour-space answer takes a real measurement — which is exactly why so many reports quote the first one.

---

*Implementation: [`nodebench/probe/p2p.py`](../nodebench/probe/p2p.py). The classifier is a pure function — pass it two numbers and it returns a verdict with its confidence and its evidence, so you can disagree with the thresholds and re-derive.*
