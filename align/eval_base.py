"""
eval_base.py
─────────────────────────────────────────────────────────────────────────────
Characterize the **base ESM-C** (biohub/ESMC-300M, exactly as the authors ship
it — no LoRA, no alignment) on the Tsuboyama stability data, split-wise, so you
have a before/after reference row for the DPO run in train_dpo.py.

This is deliberately NOT train_dpo.py with a config: every train/val metric in
that script is DPO-relative (policy − reference), and before training the policy
IS the reference, so reward_acc/margin/kl_drift all collapse to 0 and tell you
nothing about the base model. Here we report the model's *intrinsic* ranking
ability instead, with the same scoring functions the training loop uses (imported
below — no duplication), so the numbers are directly comparable.

Splits follow train_dpo.py exactly (NOT the foldseek split column):
  test  = the entire de novo set          (reward_table origin==de_novo, 148 domains)
  train = natural domains in dpo_pairs_train.csv   (267 domains, group-disjoint)
  val   = natural domains in dpo_pairs_val.csv     (54 domains, disjoint from train)
  (natural domains in neither pair file land in an "unused" bucket so nothing is dropped)
The base model trained on none of this, so every split is out-of-sample for it —
evaluating on all of them is exactly the reference row you want. (The train/test
distinction only starts to matter once you compare against the *aligned* model,
for which train is in-sample.)

Two views, both for every split:

  A. Ranking — Spearman(pseudo-LL, ΔG) and Spearman(pseudo-LL, ΔΔG) over the
     per-sequence rows of reward_table.csv, grouped by split. This is the same
     quantity train_dpo.py's heldout eval reports as denovo_spearman_base (for test),
     generalized to every split. Higher |ρ| = the raw pseudo-LL already tracks
     folding stability.

  B. Pair accuracy — fraction of preference pairs with logp(chosen) > logp(rejected)
     on dpo_pairs_train.csv / dpo_pairs_val.csv. No reference needed; this is the
     "does the base model already prefer the more stable variant" baseline that
     DPO tries to push toward 1.0.

Both are computed with BOTH scorings:
  single  one forward pass/seq   (the cheap proxy the DPO loop trains on)
  masked  mask each residue once (L fwd passes/seq — no self-leakage, rigorous)
so you can also read off the self-leakage gap on the base model.

Scoring the full 772k-row table (× L for masked) is intractable, so each group is
subsampled: --n-per-group rows for single-pass, a further --mask-n subsample for
masked. Spearman is reported with the n it was actually computed on.

Usage (from rl_esm/):
    python align/eval_base.py                      # train/val/test, default caps
    python align/eval_base.py --n-per-group 5000 --mask-n 1000
    python align/eval_base.py --config align/configs/eval_base.yaml           # yaml defaults
    python align/eval_base.py --config align/configs/eval_base.yaml --mask-n 0  # yaml + CLI override
    python align/eval_base.py --splits train val test unused   # include unused natural

Writes a tidy table to align/dpo_out/base_eval/<timestamp>/ (metrics.csv,
pair_metrics.csv, summary.md) and prints it.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# reuse the exact scoring path the training loop uses — same proxy, comparable numbers
from scoring import (
    DEVICE,
    PREP,
    apply_config,
    coerce_paths,
    git_sha,
    load_scoring_model,
    rho as _rho,
    score_masked,
    score_single,
)
from train_dpo import REWARD_CSV

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAIN_PAIRS = PREP / "dpo_pairs_train.csv"
VAL_PAIRS = PREP / "dpo_pairs_val.csv"
OUT_DIR = ROOT / "align" / "dpo_out" / "base_eval"


# ─────────────────────────────────────────────────────────────────────────────
# A. PER-SEQUENCE RANKING  (Spearman vs ΔG / ΔΔG, per split × origin)
# ─────────────────────────────────────────────────────────────────────────────

def assign_split(df):
    """Label each reward_table row train/val/test/unused, matching train_dpo.py:
    test = all de novo; train/val = natural domains in the two pair files; any other
    natural domain → 'unused' (kept so the whole dataset is accounted for)."""
    train_wts = set(pd.read_csv(TRAIN_PAIRS, usecols=["WT_name"]).WT_name)
    val_wts = set(pd.read_csv(VAL_PAIRS, usecols=["WT_name"]).WT_name)
    split = np.where(
        df.origin == "de_novo", "test",
        np.where(df.WT_name.isin(train_wts), "train",
                 np.where(df.WT_name.isin(val_wts), "val", "unused")))
    return pd.Series(split, index=df.index)


def ranking_eval(model, tokenizer, special_ids, special_set, args):
    df = pd.read_csv(REWARD_CSV, usecols=["WT_name", "origin", "aa_seq", "dG", "ddG"])
    df = df[df.dG.notna()]
    df["split"] = assign_split(df)
    if args.splits:
        df = df[df.split.isin(args.splits)]

    rng = np.random.RandomState(args.seed)
    rows = []
    for split, g in df.groupby("split"):
        n = len(g)
        take = n if not args.n_per_group else min(args.n_per_group, n)
        idx = rng.choice(n, size=take, replace=False)
        sub = g.iloc[idx]
        seqs, dG, ddG = sub.aa_seq.tolist(), sub.dG.to_numpy(), sub.ddG.to_numpy()
        origin = "de_novo" if split == "test" else "natural"
        print(f"[{split}] scoring {take}/{n} seqs (single) …")
        single = score_single(model, tokenizer, seqs, special_ids,
                               args.batch_size, args.length_norm)

        rec = {
            "split": split, "origin": origin, "group_total": n,
            "n_single": take,
            "spearman_single_dG": _rho(single, dG),
            "spearman_single_ddG": _rho(single, ddG),
        }

        if args.mask_n:
            m = min(args.mask_n, take)
            print(f"[{split}] scoring {m} seqs (masked, {m} × L fwd) …")
            masked = score_masked(model, tokenizer, seqs[:m], special_set,
                                  args.batch_size, args.length_norm)
            rec.update({
                "n_masked": m,
                "spearman_masked_dG": _rho(masked, dG[:m]),
                "spearman_masked_ddG": _rho(masked, ddG[:m]),
            })
        rows.append(rec)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# B. PAIR ACCURACY  (base already prefers the more-stable variant?)
# ─────────────────────────────────────────────────────────────────────────────

def pair_eval(model, tokenizer, special_ids, special_set, args):
    rows = []
    for split, path in [("train", TRAIN_PAIRS), ("val", VAL_PAIRS)]:
        if not path.exists():
            print(f"[pairs/{split}] {path} missing — skipping")
            continue
        df = pd.read_csv(path)
        n = len(df)
        take = n if not args.n_per_group else min(args.n_per_group, n)
        df = df.sample(take, random_state=args.seed)
        chosen, rejected = df.chosen.tolist(), df.rejected.tolist()

        print(f"[pairs/{split}] scoring {take}/{n} pairs (single) …")
        c = score_single(model, tokenizer, chosen, special_ids, args.batch_size, args.length_norm)
        r = score_single(model, tokenizer, rejected, special_ids, args.batch_size, args.length_norm)
        rec = {"split": split, "n_pairs": take,
               "pair_acc_single": float((c > r).mean())}

        if args.mask_n:
            m = min(args.mask_n, take)
            print(f"[pairs/{split}] scoring {m} pairs (masked) …")
            cm = score_masked(model, tokenizer, chosen[:m], special_set, args.batch_size, args.length_norm)
            rm = score_masked(model, tokenizer, rejected[:m], special_set, args.batch_size, args.length_norm)
            rec.update({"n_pairs_masked": m, "pair_acc_masked": float((cm > rm).mean())})
        rows.append(rec)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_args(argv=None):
    """Two-pass parse mirroring train_dpo.py: --config (if given) sets YAML defaults,
    explicit CLI flags win. YAML keys must match the underscore dest names below."""
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("--config", type=Path, default=None,
                            help="YAML file of defaults (e.g. align/configs/eval_base.yaml); "
                                 "keys must match the underscore dest names — explicit CLI "
                                 "flags still override whatever it sets")
    cfg_args, remaining = cfg_parser.parse_known_args(argv)

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                parents=[cfg_parser])
    p.add_argument("--model", default="biohub/ESMC-300M")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-per-group", type=int, default=3000,
                   help="cap sequences/pairs scored per split group for single-pass (0=all)")
    p.add_argument("--mask-n", type=int, default=500,
                   help="cap sequences/pairs scored with masked pseudo-LL per group (0=skip masked)")
    p.add_argument("--splits", nargs="*", default=["train", "val", "test"],
                   help="which splits to score: train/val/test (train_dpo.py scheme) and/or "
                        "'unused' (natural domains in neither pair file). Default: train val test")
    p.add_argument("--length-norm", action="store_true",
                   help="length-normalize pseudo-LL (match train_dpo --length-norm)")
    p.add_argument("--no-pairs", action="store_true", help="skip the pair-accuracy view")
    p.add_argument("--adapter", type=Path, default=None,
                   help="path to a saved LoRA adapter dir (e.g. align/dpo_out/runs/<exp>/<run>/best) "
                        "to evaluate the ALIGNED policy instead of the base model — same metrics, "
                        "apples-to-apples with a base run. Omit to evaluate the plain base model")
    p.add_argument("--seed", type=int, default=0)

    if cfg_args.config:
        apply_config(p, cfg_args.config)

    args = p.parse_args(remaining)
    args.config = cfg_args.config
    # set_defaults() bypasses type=Path for YAML-supplied values — coerce explicitly
    return coerce_paths(args, "adapter")


def main(argv=None):
    args = build_args(argv)
    assert REWARD_CSV.exists(), f"missing {REWARD_CSV}"
    for pth in (TRAIN_PAIRS, VAL_PAIRS):
        assert pth.exists(), f"missing {pth} — run data/build_dpo_pairs.py"

    # The subsample (np.random / df.sample below) is what determines *which* rows are
    # scored, and it's seeded via args.seed. Scoring itself has no stochastic ops
    # (eval mode, LoRA dropout 0, no sampling), so these torch seeds don't change any
    # value — they're hygiene/future-proofing. Bit-exact CUDA determinism would also
    # need use_deterministic_algorithms(True), not worth the slowdown for Spearman.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # this script characterizes ONE model per run (base, or aligned via --adapter), so
    # the base/aligned pair `load_scoring_model` offers is discarded — the adapter, if
    # given, just stays on for every call.
    model, tokenizer, special_ids, special_set, _ = load_scoring_model(
        args.model, args.adapter)

    rank_df = ranking_eval(model, tokenizer, special_ids, special_set, args)
    pair_df = None if args.no_pairs else pair_eval(model, tokenizer, special_ids, special_set, args)

    which = "aligned" if args.adapter is not None else "base"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"{ts}_{which}"
    out.mkdir(parents=True, exist_ok=True)
    rank_df.to_csv(out / "metrics.csv", index=False)
    if pair_df is not None:
        pair_df.to_csv(out / "pair_metrics.csv", index=False)

    # full run config alongside the results, so any metrics.csv is reproducible
    # (seed fixes the random subsample) — mirrors train_dpo.py's config.json
    config_dump = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config_dump.update(git_sha=git_sha(), device=DEVICE, timestamp=ts,
                       reward_csv=str(REWARD_CSV), started=datetime.now().isoformat())
    (out / "config.json").write_text(json.dumps(config_dump, indent=2, default=str))

    lines = [f"# {which.capitalize()}-model eval — {args.model}"
             + (f"  + LoRA {args.adapter}" if args.adapter is not None else ""),
             f"git_sha {git_sha()}  |  {ts}  |  length_norm={args.length_norm}",
             f"n_per_group={args.n_per_group}  mask_n={args.mask_n}",
             "", "## Ranking — Spearman(pseudo-LL, stability)", "",
             rank_df.round(4).to_string(index=False)]
    if pair_df is not None:
        lines += ["", "## Pair accuracy — P(logp(chosen) > logp(rejected))", "",
                  pair_df.round(4).to_string(index=False)]
    summary = "\n".join(lines)
    (out / "summary.md").write_text(summary)
    print("\n" + summary)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
