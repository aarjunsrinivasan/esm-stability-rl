"""Leakage-free DPO train/val preference pairs from a WT-level structural split.

data/prepare.py's dpo_pairs.csv is built from ALL natural domains; align/train_dpo.py
then carves off a random val_frac of WT domains at *runtime* (load_pairs). That's
group-disjoint (no WT_name in both splits) but doesn't know about structural
redundancy — two near-identical natural domains (see data/foldseek_split.py) could
still land on opposite sides of that random carve, letting the policy implicitly
see val-domain structure during training.

This script instead builds separate dpo_pairs_train.csv / dpo_pairs_val.csv straight
from a WT-level split file — by default the denovo-safe Foldseek split (structural
non-redundancy between train/val, with the natural-train / de-novo-test pretraining-
leakage guarantee preserved). Same pairing algorithm as prepare.py (ΔG margin,
pairs/domain cap, via prepare.sample_preference_pairs), just scoped to each split's
WT pool. Test is untouched: reward_table.csv's de novo rows, scored directly by
align/train_dpo.py --heldout-eval (all 148 domains, regardless of which split file
you pick here — that guarantee only matters for train/val).

Usage:
  python data/foldseek_split.py --stratify-pppl                                  # prerequisite
  python data/build_dpo_pairs.py                                                  # denovo-safe, full variant (recommended)
  python data/build_dpo_pairs.py --split-file wt_split_foldseek_denovo_safe.csv   # paper-exact, smaller train

Then in align/train_dpo.py:
  python align/train_dpo.py --train-pairs data/prepared/dpo_pairs_train.csv \\
                             --val-pairs   data/prepared/dpo_pairs_val.csv  --heldout-eval
"""
from __future__ import annotations

import argparse
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "prepared"
REWARD_CSV = OUT / "reward_table.csv"

MARGIN = 1.0
MAX_PAIRS_PER_WT = 200


def main() -> None:
    import pandas as pd
    from prepare import sample_preference_pairs

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-file", default="wt_split_foldseek_full_denovo_safe.csv",
                     help="WT-level split file in data/prepared/ (needs WT_name + split columns, "
                          "with train/val/test labels — i.e. one of the data/foldseek_split.py outputs)")
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--max-pairs-per-wt", type=int, default=MAX_PAIRS_PER_WT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert REWARD_CSV.exists(), f"missing {REWARD_CSV} — run: python data/prepare.py --dataset tsuboyama"
    split_path = OUT / args.split_file
    assert split_path.exists(), f"missing {split_path} — run: python data/foldseek_split.py --stratify-pppl"

    reward = pd.read_csv(REWARD_CSV, usecols=["WT_name", "origin", "aa_seq", "dG"])
    split = pd.read_csv(split_path, usecols=["WT_name", "split"])

    train_wts = set(split.loc[split.split == "train", "WT_name"])
    val_wts   = set(split.loc[split.split == "val", "WT_name"])
    test_wts  = set(split.loc[split.split == "test", "WT_name"])

    denovo_wts = set(reward.loc[reward.origin == "de_novo", "WT_name"])
    if test_wts and test_wts != denovo_wts:
        print(f"  NOTE: {args.split_file}'s test set ({len(test_wts)} domains) != all de novo "
              f"domains ({len(denovo_wts)}) — this split file is origin-agnostic and does NOT "
              f"preserve the pretraining-leakage guarantee (see README '2b'). heldout-eval in "
              f"align/train_dpo.py always scores all de novo domains regardless of this split.")

    for name, wts in [("train", train_wts), ("val", val_wts)]:
        assert wts, f"no WT domains assigned to '{name}' in {args.split_file} — nothing to pair"
        pool = reward[reward.WT_name.isin(wts) & reward.dG.notna()]
        pairs = sample_preference_pairs(pool, args.margin, args.max_pairs_per_wt, args.seed)
        out_path = OUT / f"dpo_pairs_{name}.csv"
        pairs.to_csv(out_path, index=False)
        print(f"dpo_pairs_{name}: {len(pairs):,} pairs across {pairs.WT_name.nunique()} domains  →  {out_path}")


if __name__ == "__main__":
    main()
