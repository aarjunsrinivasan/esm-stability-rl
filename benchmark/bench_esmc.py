"""
bench_esmc.py
─────────────────────────────────────────────────────────────────────────────
Hardware feasibility benchmark: can *this* machine load ESM-C 300M / 600M /
6B and run a forward pass and a forward+backward pass, and at what batch
size?

For each (model, seq_len) this doubles the batch size (1, 2, 4, 8, ...) until
a step fails, then binary-searches between the last success and first
failure to tighten the bound. "Failure" is either a CUDA OOM (expected,
handled: free the cache and move on to the next config) or an unexpected
error (logged with a traceback; by default the script also moves on, but
--halt-on-error makes it fatal). A model that fails to *load* at all is
skipped entirely and every combo under it is recorded as load_failed.

The backward pass uses LoRA (r=8, alpha=16, target_modules=["out_proj"]) by
default, matching this project's actual training config in
align/train_dpo.py — that's the memory profile this project cares about, not
full fine-tuning. Pass --full-finetune to instead make every parameter
trainable (a heavier, worst-case check, mainly useful for the 300M/600M
models — 6B will almost certainly OOM even at batch size 1).

A JSON dump and a Markdown report are always written to benchmark/results/,
even on a halt or Ctrl-C, so a partial run still produces a report.

Usage (from rl_esm/):
    pixi run python benchmark/bench_esmc.py
    pixi run python benchmark/bench_esmc.py --models biohub/ESMC-300M biohub/ESMC-600M
    pixi run python benchmark/bench_esmc.py --seq-lens 128 512 --dtype bf16
    pixi run python benchmark/bench_esmc.py --full-finetune
    pixi run python benchmark/bench_esmc.py --skip-backward
    pixi run python benchmark/bench_esmc.py --halt-on-error
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # rl_esm/
RESULTS_DIR = HERE / "results"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_MODELS = ["biohub/ESMC-300M", "biohub/ESMC-600M", "biohub/ESMC-6B"]
DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class Halt(Exception):
    """Raised to unwind the whole run when --halt-on-error trips."""


# ─────────────────────────────────────────────────────────────────────────────
# RESULT RECORD
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfigResult:
    model: str
    mode: str                 # "forward" | "backward"
    seq_len: int
    dtype: str
    max_batch_size: int | None = None   # largest batch size that ran cleanly
    peak_mem_gb: float | None = None    # at max_batch_size
    latency_s: float | None = None      # at max_batch_size
    throughput_seq_s: float | None = None
    status: str = "not_run"     # ok | oom_at_min | load_failed | error | halted
    detail: str = ""
    batch_sizes_tried: list[int] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA
# ─────────────────────────────────────────────────────────────────────────────

def random_sequences(n: int, length: int, rng: random.Random) -> list[str]:
    return ["".join(rng.choices(AMINO_ACIDS, k=length)) for _ in range(n)]


def make_mlm_batch(tokenizer, sequences, device, mlm_prob, rng):
    enc = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}

    special_ids = {tokenizer.cls_token_id, tokenizer.eos_token_id,
                   tokenizer.pad_token_id, tokenizer.mask_token_id} - {None}
    input_ids = enc["input_ids"]
    is_special = torch.isin(input_ids, torch.tensor(sorted(special_ids), device=device))
    maskable = (~is_special) & enc["attention_mask"].bool()

    probs = torch.full(input_ids.shape, mlm_prob, device=device)
    probs[~maskable] = 0.0
    do_mask = torch.bernoulli(probs).bool()
    # guarantee at least one masked token per non-empty sequence so loss is defined
    for i in range(input_ids.shape[0]):
        if not do_mask[i].any() and maskable[i].any():
            idx = maskable[i].nonzero()[0]
            do_mask[i, idx] = True

    labels = input_ids.clone()
    labels[~do_mask] = -100
    masked_input_ids = input_ids.clone()
    masked_input_ids[do_mask] = tokenizer.mask_token_id
    enc["input_ids"] = masked_input_ids
    return enc, labels


# ─────────────────────────────────────────────────────────────────────────────
# ONE STEP
# ─────────────────────────────────────────────────────────────────────────────

def run_step(model, tokenizer, device, mode, batch_size, seq_len, mlm_prob, rng):
    sequences = random_sequences(batch_size, seq_len, rng)
    enc, labels = make_mlm_batch(tokenizer, sequences, device, mlm_prob, rng)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    if mode == "forward":
        with torch.inference_mode():
            model(**enc)
    else:
        model.zero_grad(set_to_none=True)
        out = model(**enc, labels=labels)
        out.loss.backward()

    torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    return latency, peak_mem_gb


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def reclaim():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# BATCH-SIZE SEARCH (double up, then binary-search the boundary)
# ─────────────────────────────────────────────────────────────────────────────

def find_max_batch_size(model, tokenizer, device, mode, seq_len, dtype_name,
                         batch_min, batch_max, mlm_prob, rng, halt_on_error,
                         log) -> ConfigResult:
    result = ConfigResult(model="", mode=mode, seq_len=seq_len, dtype=dtype_name)

    last_ok: tuple[int, float, float] | None = None   # (batch, latency, peak_mem)
    first_fail_bs: int | None = None
    bs = batch_min

    def attempt(b):
        nonlocal last_ok, first_fail_bs
        result.batch_sizes_tried.append(b)
        try:
            latency, mem = run_step(model, tokenizer, device, mode, b, seq_len, mlm_prob, rng)
            last_ok = (b, latency, mem)
            log(f"    bs={b:<5} OK   peak_mem={mem:6.2f} GB  latency={latency:6.3f}s")
            return True
        except Exception as exc:  # noqa: BLE001 - deliberately broad, classified below
            if is_oom(exc):
                log(f"    bs={b:<5} OOM  ({str(exc).splitlines()[0][:100]})")
                reclaim()
                first_fail_bs = b
                return False
            log(f"    bs={b:<5} ERROR (not OOM): {exc!r}")
            traceback.print_exc()
            reclaim()
            first_fail_bs = b
            if halt_on_error:
                raise Halt(f"{model} mode={mode} seq_len={seq_len} bs={b}: {exc}") from exc
            result.status = "error"
            result.detail = repr(exc)
            return False

    # doubling phase
    while bs <= batch_max:
        ok = attempt(bs)
        if not ok:
            break
        bs *= 2
    else:
        # reached batch_max without failing
        first_fail_bs = None

    # binary-search refinement between last_ok and first_fail_bs (bounded steps)
    if last_ok is not None and first_fail_bs is not None:
        lo, hi = last_ok[0], first_fail_bs
        for _ in range(3):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            if attempt(mid):
                lo = mid
            else:
                hi = mid
        # last_ok now reflects the tightest known-good batch size

    if last_ok is None:
        result.status = "oom_at_min" if result.status == "not_run" else result.status
        result.detail = result.detail or f"failed at the minimum batch size ({batch_min})"
    else:
        b, latency, mem = last_ok
        result.max_batch_size = b
        result.peak_mem_gb = round(mem, 3)
        result.latency_s = round(latency, 4)
        result.throughput_seq_s = round(b / latency, 2) if latency > 0 else None
        if result.status == "not_run":
            result.status = "ok"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_id, dtype, device, backward_mode, lora_r, lora_alpha, lora_dropout):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForMaskedLM.from_pretrained(model_id, dtype=dtype).to(device)

    if backward_mode == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          target_modules=["out_proj"])
        model = get_peft_model(base, cfg)
    else:
        model = base

    return model, tokenizer


def n_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    p.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512])
    p.add_argument("--batch-min", type=int, default=1)
    p.add_argument("--batch-max", type=int, default=128)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--skip-forward", action="store_true")
    p.add_argument("--skip-backward", action="store_true")
    p.add_argument("--full-finetune", action="store_true",
                    help="backward pass trains ALL params instead of LoRA (heavier, worst-case)")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--halt-on-error", action="store_true",
                    help="stop the whole script on a non-OOM error instead of skipping the config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device visible; this benchmark is GPU-only.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    dtype = DTYPE_MAP[args.dtype]
    backward_mode = "full" if args.full_finetune else "lora"
    modes = [m for m, skip in (("forward", args.skip_forward), ("backward", args.skip_backward)) if not skip]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    gpu_name = torch.cuda.get_device_name(device)
    gpu_total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"GPU: {gpu_name}  ({gpu_total_gb:.1f} GB total)")
    print(f"dtype={args.dtype}  backward_mode={backward_mode}  modes={modes}  "
          f"seq_lens={args.seq_lens}  batch range=[{args.batch_min}, {args.batch_max}]")

    results: list[ConfigResult] = []
    halted = False
    halt_reason = ""

    def log(msg):
        print(msg, flush=True)

    try:
        for model_id in args.models:
            print(f"\n=== {model_id} ===")
            rng = random.Random(args.seed)
            model = tokenizer = None
            try:
                t0 = time.perf_counter()
                model, tokenizer = load_model(
                    model_id, dtype, device, backward_mode if "backward" in modes else "frozen",
                    args.lora_r, args.lora_alpha, args.lora_dropout,
                )
                model.eval()
                load_s = time.perf_counter() - t0
                n_params = sum(p.numel() for p in model.parameters())
                n_train = n_trainable_params(model)
                print(f"  loaded in {load_s:.1f}s  |  {n_params/1e6:.0f}M params total, "
                      f"{n_train/1e6:.2f}M trainable ({backward_mode})")
            except Exception as exc:  # noqa: BLE001
                print(f"  LOAD FAILED: {exc!r}")
                traceback.print_exc()
                reclaim()
                for seq_len in args.seq_lens:
                    for mode in modes:
                        results.append(ConfigResult(
                            model=model_id, mode=mode, seq_len=seq_len, dtype=args.dtype,
                            status="load_failed", detail=repr(exc),
                        ))
                if args.halt_on_error:
                    raise Halt(f"load failed for {model_id}: {exc}") from exc
                continue

            try:
                for seq_len in args.seq_lens:
                    for mode in modes:
                        print(f"  -- mode={mode} seq_len={seq_len} --")
                        model.train(mode == "backward")
                        r = find_max_batch_size(
                            model, tokenizer, device, mode, seq_len, args.dtype,
                            args.batch_min, args.batch_max, args.mlm_prob, rng,
                            args.halt_on_error, log,
                        )
                        r.model = model_id
                        results.append(r)
            finally:
                del model, tokenizer
                reclaim()

    except Halt as h:
        halted = True
        halt_reason = str(h)
        print(f"\nHALTED: {halt_reason}", file=sys.stderr)
    except KeyboardInterrupt:
        halted = True
        halt_reason = "KeyboardInterrupt"
        print("\nInterrupted by user.", file=sys.stderr)
    finally:
        write_report(results, args, gpu_name, gpu_total_gb, backward_mode, ts, halted, halt_reason)

    sys.exit(1 if halted else 0)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(results, args, gpu_name, gpu_total_gb, backward_mode, ts, halted, halt_reason):
    json_path = args.output_dir / f"bench_{ts}.json"
    md_path = args.output_dir / f"report_{ts}.md"

    payload = {
        "timestamp": ts,
        "gpu": gpu_name,
        "gpu_total_gb": round(gpu_total_gb, 1),
        "dtype": args.dtype,
        "backward_mode": backward_mode,
        "halted": halted,
        "halt_reason": halt_reason,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2))

    lines = []
    lines.append(f"# ESM-C benchmark report — {ts}")
    lines.append("")
    lines.append(f"- GPU: **{gpu_name}** ({gpu_total_gb:.1f} GB)")
    lines.append(f"- dtype: `{args.dtype}`  |  backward mode: `{backward_mode}`"
                 + (f" (r={args.lora_r}, alpha={args.lora_alpha})" if backward_mode == "lora" else ""))
    lines.append(f"- batch search range: [{args.batch_min}, {args.batch_max}] (doubling + binary-search refine)")
    if halted:
        lines.append(f"- **Run halted early:** {halt_reason}")
    lines.append("")
    lines.append("| model | mode | seq_len | status | max batch size | peak mem (GB) | latency (s) | throughput (seq/s) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.model} | {r.mode} | {r.seq_len} | {r.status} "
            f"| {r.max_batch_size if r.max_batch_size is not None else '—'} "
            f"| {r.peak_mem_gb if r.peak_mem_gb is not None else '—'} "
            f"| {r.latency_s if r.latency_s is not None else '—'} "
            f"| {r.throughput_seq_s if r.throughput_seq_s is not None else '—'} |"
        )

    failures = [r for r in results if r.status not in ("ok",)]
    if failures:
        lines.append("")
        lines.append("## Notes / failures")
        for r in failures:
            lines.append(f"- **{r.model}** / {r.mode} / seq_len={r.seq_len}: `{r.status}` — {r.detail}")

    lines.append("")
    lines.append(f"Raw data: `{json_path.relative_to(ROOT)}`")
    md_path.write_text("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nWrote {json_path.relative_to(ROOT)} and {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
