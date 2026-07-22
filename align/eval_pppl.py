"""
eval_pppl.py
─────────────────────────────────────────────────────────────────────────────
Catastrophic-forgetting check: is the stability-aligned ESM-C policy still a
general protein language model, or did aligning on ≤75-residue domains damage it
on real, full-length natural proteins?

Prepare the data first:

    pixi run python data/download_swissprot.py    # → data/prepared/swissprot_eval.csv

WHAT THIS MEASURES
──────────────────
Masked **pseudo-perplexity** per protein, base vs aligned, bucketed by length:

    log_pppl = −(1/L) Σ_i log p(x_i | x_\\i)        pppl = exp(log_pppl)

i.e. mask each residue once, predict it from the rest, average over residues.
This is the standard MLM forgetting metric, and it is the quantity the README
has always claimed as a held-out oracle but nothing computed.

WHY THIS AND NOT THE KL TERM
────────────────────────────
train_dpo.py's KL measures how far the policy drifted *on the distribution it
trained on* — 31–75 residue Tsuboyama domains. It cannot see damage at 400
residues, and neither can eval_fireprot.py (mutant *ranking*, ≤448 residues) or
eval_base.py (same Tsuboyama domains). Swiss-Prot enzymes are natural,
median ~300 residues, and appear in no part of the DPO reward.

The **(0,75] bucket is the in-distribution control** — the length range DPO
actually trained on. The signal to read is the *trend across buckets*: a delta
that grows with length means alignment on short domains degraded long-protein
modelling. A flat, near-zero delta is the expected and desired outcome, which is
why it's reported with a bootstrap CI — "no forgetting" has to be a bounded
claim, not two similar-looking numbers.

LENGTH NORMALIZATION IS NOT OPTIONAL HERE
─────────────────────────────────────────
Every other eval config in this repo has a `length_norm` knob that must *match
the training run*, because raw summed pseudo-LL is what DPO optimized. This
script has no such knob: perplexity is per-residue by definition, and comparing
summed log-prob across a 37→512 residue range would just be measuring length.
score_masked is called with length_norm=True, always.

BASE vs ALIGNED IN ONE PASS
───────────────────────────
Pass --adapter and both the aligned policy (adapter on) and the base model (peft
`disable_adapter()` — identical weights minus LoRA) are scored on the exact same
sequences, so the per-protein deltas are paired. Omit --adapter to characterize
just the base model.

COST
────
Masked pseudo-LL needs Σ L_i forward passes and attention is O(L²). At the
default 80 proteins/bucket (≈400 proteins, median ~300 residues) that's ~120k
masked variants **per model** — tens of minutes for the pair. --n-per-bucket in
download_swissprot.py and --max-seqs here are the cost knobs.

Usage (from rl_esm/):
    pixi run python align/eval_pppl.py --config align/configs/eval_pppl.yaml
    pixi run python align/eval_pppl.py --adapter <run>/best --max-seqs 20 --batch-size 4  # smoke

Writes align/dpo_out/pppl_eval/<timestamp>/ : per_sequence.csv (one row per
protein × model), summary.csv (per bucket × model + the paired delta), summary.md.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from scoring import (
    DEVICE,
    PREP,
    adapter_ctx,
    apply_config,
    coerce_paths,
    git_sha,
    load_scoring_model,
    score_masked,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SWISSPROT_CSV = PREP / "swissprot_eval.csv"
OUT_DIR = ROOT / "align" / "dpo_out" / "pppl_eval"


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_all(model, tokenizer, special_set, df, models, args):
    """Masked pseudo-perplexity for every protein × model → tidy DataFrame.

    Scored bucket by bucket rather than in one call: batches stay length-homogeneous
    (masked_seq_logp pads each chunk to its longest member, so mixing a 40-residue and
    a 500-residue protein wastes most of the batch), and a multi-minute run reports
    progress instead of going silent.
    """
    out = []
    for name, aligned in models:
        for bucket, g in tqdm(list(df.groupby("length_bucket", sort=False)),
                              desc=f"{name}", unit="bucket"):
            with adapter_ctx(model, aligned):
                # length_norm=True is mandatory — see module docstring.
                mean_logp = score_masked(model, tokenizer, g.aa_seq.tolist(),
                                         special_set, args.batch_size, True)
            out.append(g.assign(model=name, log_pppl=-mean_logp,
                                pppl=np.exp(-mean_logp)))
    return pd.concat(out, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

# A percentile bootstrap resamples the observed points, so below a handful of them it
# reports a tight CI around whatever those few points happened to be — on a --max-seqs
# smoke run (n=2/bucket) that reads as a confident "forgetting detected". Refuse to
# quote a CI there instead of emitting a false alarm.
MIN_BOOT_N = 5


def paired_bootstrap(delta, n_boot, seed):
    """Percentile CI on the mean of `delta` (per-protein aligned − base), resampling
    proteins with replacement. Paired, so the base/aligned correlation across proteins
    is preserved and the CI is on the delta itself, not the difference of two
    independent CIs. Returns (mean, lo, hi); CI is nan below MIN_BOOT_N proteins."""
    delta = np.asarray(delta, float)
    delta = delta[np.isfinite(delta)]
    if len(delta) < MIN_BOOT_N:
        return (float(delta.mean()) if len(delta) else float("nan"),
                float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    return (float(delta.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def aggregate(per_seq, n_boot, seed):
    """Per bucket (plus an `all` row): n, base/aligned pseudo-perplexity, and the
    paired mean Δlog_pppl with its 95% CI.

    Everything is averaged in **log space** — log_pppl is the mean per-residue NLL,
    which is the additive quantity. Means of raw perplexities are dominated by the
    worst-modelled proteins; we exponentiate only for display.
    """
    wide = per_seq.pivot_table(index=["accession", "length_bucket"],
                              columns="model", values="log_pppl").reset_index()
    has_aligned = "aligned" in wide.columns

    # pivot_table sorts its index by accession, so groupby order would be arbitrary.
    # Buckets must be reported shortest→longest: the trend across them IS the result.
    bucket_order = per_seq.sort_values("length").length_bucket.drop_duplicates().tolist()

    rows = []
    groups = [(b, wide[wide.length_bucket == b]) for b in bucket_order]
    groups.append(("all", wide))
    for bucket, g in groups:
        row = {"length_bucket": bucket, "n": len(g),
               "base_log_pppl": g["base"].mean(),
               "base_pppl": np.exp(g["base"].mean())}
        if has_aligned:
            row.update(aligned_log_pppl=g["aligned"].mean(),
                       aligned_pppl=np.exp(g["aligned"].mean()))
            mean, lo, hi = paired_bootstrap(g["aligned"] - g["base"], n_boot, seed)
            row.update(delta_log_pppl=mean, ci_lo=lo, ci_hi=hi,
                       # CI excluding 0 = the drift is distinguishable from noise at
                       # this n. Positive delta = aligned is WORSE (higher perplexity).
                       significant=bool(np.isfinite(lo) and (lo > 0 or hi < 0)))
        rows.append(row)
    return pd.DataFrame(rows)


def sanity_note(summ):
    """The base model's own numbers must be plausible before any delta means anything
    — the analog of eval_fireprot.py's base-sign check. ESM-C's vocab has 20 standard
    residues, so pppl ≈ 20 is a model that has learned nothing; a real PLM sits well
    below that."""
    row = summ[summ.length_bucket == "all"]
    if not len(row):
        return ""
    p = float(row.base_pppl.iloc[0])
    if 1.0 < p < 20.0:
        return (f"\nbase pppl (all lengths) = {p:.3f} — plausible ✓ "
                f"(1 < pppl < 20 = better than uniform over the 20 standard residues)")
    return (f"\nbase pppl (all lengths) = {p:.3f} — ⚠ IMPLAUSIBLE. ≥20 means the base "
            f"model is at or worse than uniform over 20 residues; check tokenization/"
            f"length_norm before trusting any delta.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_args(argv=None):
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("--config", type=Path, default=None)
    cfg_args, remaining = cfg_parser.parse_known_args(argv)

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                parents=[cfg_parser])
    p.add_argument("--model", default="biohub/ESMC-300M")
    p.add_argument("--adapter", type=Path, default=None,
                   help="LoRA adapter dir (…/best or …/last). Given → score aligned AND "
                        "base (adapters disabled) on the same seqs. Omitted → base only.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="masked variants per forward pass. Lower than other evals on "
                        "purpose: attention is O(L²) and these proteins are ~4x longer "
                        "than the Tsuboyama domains the other defaults were tuned for")
    p.add_argument("--max-seqs", type=int, default=0,
                   help="cap total proteins scored (0=all; for smoke runs)")
    p.add_argument("--n-boot", type=int, default=5000,
                   help="paired bootstrap resamples for the delta CI (the ESM3 paper's nboot)")
    p.add_argument("--seed", type=int, default=0)

    if cfg_args.config:
        apply_config(p, cfg_args.config)

    args = p.parse_args(remaining)
    args.config = cfg_args.config
    return coerce_paths(args, "adapter")


def main(argv=None):
    args = build_args(argv)
    assert SWISSPROT_CSV.exists(), \
        f"missing {SWISSPROT_CSV} — run: pixi run python data/download_swissprot.py"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv(SWISSPROT_CSV)
    # Sort by length so each bucket's batches are near-homogeneous; ties broken by
    # accession so the --max-seqs subsample is deterministic.
    df = df.sort_values(["length", "accession"]).reset_index(drop=True)
    if args.max_seqs:
        df = df.groupby("length_bucket", sort=False, group_keys=False).head(
            max(1, args.max_seqs // df.length_bucket.nunique()))
    print(f"Scoring {len(df)} proteins ({df.length.min()}–{df.length.max()} residues) "
          f"across {df.length_bucket.nunique()} length buckets")

    model, tokenizer, special_ids, special_set, models = load_scoring_model(
        args.model, args.adapter)

    per_seq = score_all(model, tokenizer, special_set, df, models, args)
    summ = aggregate(per_seq, args.n_boot, args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / ts
    out.mkdir(parents=True, exist_ok=True)
    per_seq.to_csv(out / "per_sequence.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)

    config_dump = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config_dump.update(git_sha=git_sha(), device=DEVICE, timestamp=ts,
                       swissprot_csv=str(SWISSPROT_CSV))
    meta_path = SWISSPROT_CSV.with_suffix(".meta.json")
    if meta_path.exists():
        config_dump["swissprot_meta"] = json.loads(meta_path.read_text())
    (out / "config.json").write_text(json.dumps(config_dump, indent=2, default=str))

    lines = [f"# SwissProt pseudo-perplexity — {args.model}"
             + (f"  + LoRA {args.adapter}" if args.adapter is not None else " (base only)"),
             f"git_sha {git_sha()}  |  {ts}  |  n_boot={args.n_boot}",
             "", "## Masked pseudo-perplexity by length "
             "(delta = aligned − base in log space; positive = aligned is worse)", "",
             summ.round(4).to_string(index=False), sanity_note(summ)]
    if args.adapter is not None:
        buckets = summ[summ.length_bucket != "all"]
        testable = buckets[buckets.n >= MIN_BOOT_N]
        sig = testable[testable.significant]
        verdict = [f"{len(sig)}/{len(testable)} testable length buckets show a drift "
                   f"whose 95% CI excludes zero."]
        if len(testable) < len(buckets):
            verdict.append(f"{len(buckets) - len(testable)} bucket(s) had <{MIN_BOOT_N} "
                           f"proteins and are reported without a CI — too few to bootstrap "
                           f"(are you on a --max-seqs smoke run?).")
        elif not len(sig):
            verdict.append("No detectable forgetting at this n — the alignment left "
                           "general protein modelling intact.")
        lines += ["", "## Verdict", "", " ".join(verdict)]
    summary = "\n".join(lines)
    (out / "summary.md").write_text(summary)
    print("\n" + summary)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
