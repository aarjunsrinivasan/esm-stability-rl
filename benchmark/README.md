# benchmark/

Hardware feasibility check: can this machine load ESM-C 300M / 600M / 6B
(`biohub/ESMC-300M`, `biohub/ESMC-600M`, `biohub/ESMC-6B` — the same repo IDs
used elsewhere in this project, see `align/configs/*.yaml`) and run a forward
pass and a forward+backward pass, and at what batch size?

## Usage (from `rl_esm/`)

```bash
# full sweep: all 3 models, seq_lens [128, 512], bf16, LoRA backward
pixi run python benchmark/bench_esmc.py

# just the models that matter right now
pixi run python benchmark/bench_esmc.py --models biohub/ESMC-300M biohub/ESMC-600M

# different sequence lengths / dtype
pixi run python benchmark/bench_esmc.py --seq-lens 128 512 --dtype bf16

# worst-case backward: train every parameter instead of LoRA
pixi run python benchmark/bench_esmc.py --full-finetune

# forward-only (e.g. just checking reward-probe-style frozen inference)
pixi run python benchmark/bench_esmc.py --skip-backward

# stop the whole run on the first non-OOM error instead of skipping the config
pixi run python benchmark/bench_esmc.py --halt-on-error
```

## What it does

For each `(model, mode, seq_len)` combo, batch size is doubled (1, 2, 4, ...)
until a step fails, then binary-searched between the last success and first
failure (a few extra steps) to tighten the bound. Two failure kinds are
handled differently:

- **CUDA OOM** — expected outcome, not a bug. Caught, GPU cache freed, and
  the search for that combo stops there; the run moves on to the next combo.
- **Any other exception** — logged with a full traceback. By default the
  script still skips to the next combo (marked `status: error` in the
  report); pass `--halt-on-error` to make it fatal instead.

A model that fails to *load* (bad repo id, corrupt cache, disk error, etc.)
is skipped entirely — every combo under it is recorded as `load_failed`
rather than crashing the whole sweep.

The backward pass defaults to **LoRA** (`r=8, alpha=16, target_modules=
["out_proj"]`), matching this project's actual training config in
`align/train_dpo.py` — that's the memory profile this project cares about.
`--full-finetune` instead makes every parameter trainable, mainly useful as
a worst-case check on the smaller models (6B will very likely OOM even at
batch size 1 with all params trainable).

A JSON dump and a Markdown report are always written to `benchmark/results/`
(gitignored — machine-specific, regenerate locally), even on `--halt-on-error`
or Ctrl-C, so a partial run still leaves a report behind.
