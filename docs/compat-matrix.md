# sm_XX × CUDA × torch: the three checks, one of which everyone forgets

A new GPU generation arrives and the same week is lost by thousands of people to the same three-way version problem. It is entirely mechanical. Here is the whole thing.

---

## The three checks

**Check 1 — does your CUDA version know this architecture at all?**

`nvcc` needs to have shipped with support for the compute capability. A CUDA 12.4 toolkit has never heard of `sm_120`; it cannot generate code for it, and the error message will not say so clearly.

**Check 2 — does your framework build target a CUDA that new?**

`torch.version.cuda` tells you what CUDA the wheel was built against. A torch built for CUDA 12.4 cannot contain `sm_120` code regardless of what toolkit is installed on the machine.

**Check 3 — does this specific wheel actually contain a cubin for your architecture?**

```python
torch.cuda.get_arch_list()
```

**This is the one people forget.** Checks 1 and 2 can both pass and this can still fail. A wheel built with CUDA 12.8 will only contain code for the architectures whoever built it chose to compile — and that list is a build-time decision, not a property of the CUDA version.

The symptom when check 3 fails on its own is the confusing one, because everything *looks* right:

```
CUDA error: no kernel image is available for execution on the device
```

or worse, it runs — via PTX JIT from a forward-compatible cubin — and is inexplicably slow, with a long stall on the first kernel launch while the driver compiles.

---

## The table

| Compute cap | Architecture | Example cards | Min CUDA | Min torch |
|---|---|---|---|---|
| 12.0 (`sm_120`) | Blackwell (consumer) | RTX 5090, 5080, 5070 | **12.8** | **2.7** |
| 10.0 (`sm_100`) | Blackwell (datacenter) | B200, GB200 | **12.8** | 2.7 |
| 9.0 (`sm_90`) | Hopper | H100, H200 | 11.8 | 2.0 |
| 8.9 (`sm_89`) | Ada Lovelace | RTX 4090, L40S | 11.8 | 2.0 |
| 8.6 (`sm_86`) | Ampere (consumer) | RTX 3090, A10 | 11.1 | 1.8 |
| 8.0 (`sm_80`) | Ampere (datacenter) | A100 | 11.0 | 1.7 |
| 7.5 (`sm_75`) | Turing | T4, RTX 2080 Ti | 10.0 | 1.3 |

---

## Run the three checks in one line

```bash
nodebench doctor
```

Or by hand:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list())"
```

You want your card's `sm_XX` to appear in that arch list. If it does not, the wheel is wrong for the card no matter how correct the CUDA version looks.

---

## The `a` suffix, if you compile anything

From Hopper onward, the architecture-specific instructions — `wgmma`, the TMA path, everything CUTLASS 3.x builds its fast kernels on — live behind an `a`-suffixed target:

```bash
-gencode arch=compute_90a,code=sm_90a     # Hopper
-gencode arch=compute_120a,code=sm_120a   # Blackwell consumer
```

Build without the suffix and it compiles, links, runs, and quietly uses slow kernels. This is the most common reason a brand-new Hopper or Blackwell card benchmarks like a previous-generation part, and nothing in the output tells you.

The conversion from `nvidia-smi` to `nvcc` also trips people: nvidia-smi reports `12.0`, nvcc wants `120`. Get it wrong and you get a binary that builds cleanly and dies at runtime.

```bash
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
ARCH="$SM"; [ "$SM" -ge 90 ] && ARCH="${SM}a"
```

`nodebench setup` does this for you and prints what it derived.

---

## NCCL, separately

NCCL versions independently of CUDA and torch, and on a new architecture the version matters:

```python
torch.cuda.nccl.version()
```

On Blackwell, **2.21 is the practical minimum** and newer is materially better — early NCCL versions on a new architecture fall back to conservative transports and leave a large amount of bandwidth unclaimed. `nodebench doctor` warns on this specifically, because it is a silent performance problem rather than an error.

---

## Driver, and forward compatibility

The driver must be new enough for the CUDA runtime your wheel was built against. CUDA 12.8 wants driver ≥ 570 on Linux.

CUDA minor-version compatibility means a 12.x runtime generally works on a driver shipped for any 12.y — but this does not extend backwards across a *major* version, and it does not conjure a cubin for an architecture the driver's compiler predates.

---

## When something is wrong, in order

1. `nvidia-smi` — is the card visible at all? If you are in a container, was it started with `--gpus`?
2. `nvidia-smi --query-gpu=compute_cap` — what architecture is it, really?
3. `torch.version.cuda` — what CUDA is this wheel built for? Compare against the table.
4. `torch.cuda.get_arch_list()` — **is your `sm_XX` actually in there?**
5. `torch.cuda.nccl.version()` — new enough for this architecture?
6. Only then start suspecting your code.

Steps 1–5 take thirty seconds and rule out the overwhelming majority of "it doesn't work on the new cards" reports.

---

*Implementation: [`nodebench/probe/compat.py`](../nodebench/probe/compat.py). The `ARCH_REQUIREMENTS` table is keyed by compute capability, so adding a new architecture is one row.*
