"""LoRA fine-tuning throughput on synthetic data.

Why synthetic
-------------
Every real dataset makes a benchmark unrunnable by a stranger: it must be
downloaded (tens of GB), it must be preprocessed, and it introduces storage and
CPU decode as uncontrolled variables. For measuring *the GPUs*, a real dataset
adds nothing -- the model does not care whether the token IDs mean anything.

So this recipe generates random token IDs at the configured sequence length. The
compute, the memory traffic and the gradient all-reduce are identical to a real
run. What it does NOT measure is the input pipeline, and that is stated in the
output rather than glossed over.

Why LoRA specifically
---------------------
LoRA is the case where a PCIe node looks good, and understanding why is the
point. Only the adapter weights are reduced each step -- often a few tens of
millions of parameters against several billion. Communication per step drops by
two orders of magnitude, so scaling stays near-linear on hardware that would
choke on full fine-tuning. Running both and comparing is how you demonstrate
that a node's interconnect is adequate *for a workload class*, which is a much
more useful statement than a bandwidth number.

Usage
-----
    torchrun --nproc_per_node=4 bench_lora.py --model <hf-id> --seq-len 1024
    python bench_lora.py --model <hf-id> --steps 30        # single GPU
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B",
                   help="HF model id or local path. Any causal LM works.")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-per-gpu", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=8,
                   help="Steps discarded before timing. The first steps include CUDA context "
                        "creation, autotuning and memory-pool growth.")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--full-finetune", action="store_true",
                   help="Train all parameters instead. Use this to show the communication "
                        "difference; expect scaling to fall apart on a PCIe node.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        print("This recipe needs `transformers`. Install with: pip install 'nodebench[recipes]'",
              file=sys.stderr)
        return 2

    # transformers 5.0 renamed `torch_dtype` to `dtype`. The old name still works
    # but prints a deprecation warning on every single run, and it will be
    # removed. Passing the new name to a 4.x install is worse -- it lands in
    # **kwargs and the model quietly loads in fp32, which would make every
    # throughput number here wrong by roughly a factor of two with nothing on
    # screen to say so. So pick by version rather than by try/except.
    _dtype_kw = ("dtype" if int(transformers.__version__.split(".")[0]) >= 5
                 else "torch_dtype")

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world > 1

    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = getattr(torch, args.dtype)

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    log(f"loading {args.model} ...")
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, **{_dtype_kw: dtype}
    ).to(device)

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    adapter = "full"
    if not args.full_finetune:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            print("LoRA mode needs `peft`. Install with: pip install 'nodebench[recipes]', "
                  "or pass --full-finetune.", file=sys.stderr)
            return 2
        lcfg = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lcfg)
        adapter = f"lora_r{args.lora_rank}"

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable {trainable:,} / {total_params:,} ({trainable / total_params:.3%})")

    if distributed:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # Synthetic batch: fixed shape, random token ids, generated once and reused.
    # Regenerating each step would measure the RNG, not the model.
    vocab = getattr(cfg, "vocab_size", 32000)
    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    ids = torch.randint(0, vocab, (args.batch_per_gpu, args.seq_len), generator=g).to(device)
    attn = torch.ones_like(ids)

    times: List[float] = []
    peak_mem = 0
    # A diverged run still produces step times, and step times are all this
    # script reports. So the loss has to be carried out with the throughput --
    # otherwise "fp16 without a grad scaler overflowed on step 3" is a fact
    # that exists only in the scrollback.
    nonfinite_steps = 0
    last_loss = float("nan")
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(args.steps):
        if distributed:
            dist.barrier()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        out = model(input_ids=ids, attention_mask=attn, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        if step >= args.warmup:
            times.append(dt)
        last_loss = float(out.loss.item())
        if last_loss != last_loss or last_loss in (float("inf"), float("-inf")):
            nonfinite_steps += 1
        if rank == 0 and (step % 5 == 0 or step == args.steps - 1):
            log(f"  step {step:3d}  {dt * 1000:8.1f} ms  loss {last_loss:.4f}")

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    med = statistics.median(times) if times else 0.0
    mean = statistics.fmean(times) if times else 0.0
    cv = (statistics.pstdev(times) / mean * 100) if len(times) > 1 and mean else 0.0
    global_batch = args.batch_per_gpu * world
    tok_s = global_batch * args.seq_len / med if med else 0.0

    # Gradient traffic per step. This is why LoRA scales on PCIe and full does not.
    bytes_per_param = 4 if args.dtype == "float32" else 2
    comm_bytes = (2.0 * (world - 1) / world * trainable * bytes_per_param) if world > 1 else 0.0

    result: Dict[str, Any] = {
        "model": args.model,
        "mode": adapter,
        "world_size": world,
        "seq_len": args.seq_len,
        "batch_per_gpu": args.batch_per_gpu,
        "global_batch": global_batch,
        "dtype": args.dtype,
        "grad_checkpointing": args.grad_ckpt,
        "total_params": total_params,
        "trainable_params": trainable,
        "trainable_frac": round(trainable / total_params, 6),
        "steps_timed": len(times),
        "steps_warmup": args.warmup,
        "step_time_s_median": round(med, 5),
        "step_time_s_mean": round(mean, 5),
        "step_time_cv_pct": round(cv, 2),
        "tokens_per_s": round(tok_s, 1),
        "samples_per_s": round(global_batch / med, 3) if med else 0.0,
        "peak_memory_mib": round(peak_mem, 1),
        "final_loss": round(last_loss, 4) if last_loss == last_loss else None,
        "nonfinite_loss_steps": nonfinite_steps,
        "allreduce_bytes_per_step": comm_bytes,
        "allreduce_gib_per_step": round(comm_bytes / (1024 ** 3), 6),
        "allreduce_gbs_required": round(comm_bytes / med / 1e9, 3) if med else 0.0,
        "caveats": [
            "Input is synthetic random token IDs. Storage and CPU decode are deliberately "
            "excluded; this measures the GPUs, not the data pipeline.",
            "Step time is the median of timed steps after warmup. Reported with its CV so a "
            "noisy run is visible rather than averaged away.",
            "allreduce_gbs_required is what the interconnect must sustain to keep up. Compare "
            "it against the measured all-reduce busbw: if it is close, communication is the "
            "ceiling and adding GPUs will stop helping.",
        ],
    }

    if nonfinite_steps:
        result["caveats"].insert(0, (
            f"Loss was non-finite on {nonfinite_steps} of {args.steps} steps. The timings "
            "below are still real work on real tensors, but this run did not train. "
            "float16 full fine-tuning without a gradient scaler is the usual cause -- "
            "use --dtype bfloat16."
        ))
        if rank == 0:
            print(f"WARNING: loss went non-finite on {nonfinite_steps}/{args.steps} steps; "
                  "throughput is still valid, convergence is not.", file=sys.stderr)

    if rank == 0:
        print(json.dumps(result, indent=2))
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            log(f"wrote {args.out}")

    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
