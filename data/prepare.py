"""Prepare training inputs from a downloaded dataset.

Reads raw files from data/<dataset>/, writes processed files to data/prepared/.

Usage:
  python data/prepare.py --dataset tsuboyama
"""
from __future__ import annotations

import argparse
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


# ── shared helpers ────────────────────────────────────────────────────────────

def parse_dG(x):
    """Parse Tsuboyama censored/normal stability values."""
    import numpy as np

    if x == "<-1":
        return -1.0
    if x == ">5":
        return 5.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def sample_preference_pairs(df, margin: float, max_pairs_per_wt: int, seed: int):
    """Within each WT_name, sample (chosen, rejected) pairs with |ΔG gap| >= margin.

    `df` must have columns WT_name, aa_seq, dG. Reused by data/build_dpo_pairs.py
    to build pairs scoped to an arbitrary WT-level split (e.g. the Foldseek split).
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    pairs = []
    for wt, g in df.groupby("WT_name"):
        seqs, dGs = g.aa_seq.to_numpy(), g.dG.to_numpy()
        n = len(g)
        if n < 2:
            continue
        cand = rng.integers(0, n, size=(min(max_pairs_per_wt * 6, n * n), 2))
        cand = cand[cand[:, 0] != cand[:, 1]]
        kept = 0
        for i, j in cand:
            if abs(dGs[i] - dGs[j]) < margin:
                continue
            hi, lo = (i, j) if dGs[i] > dGs[j] else (j, i)
            pairs.append((wt, seqs[hi], seqs[lo], float(dGs[hi]), float(dGs[lo])))
            kept += 1
            if kept >= max_pairs_per_wt:
                break
    return pd.DataFrame(pairs, columns=["WT_name", "chosen", "rejected", "dG_chosen", "dG_rejected"])


# ── tsuboyama ────────────────────────────────────────────────────────────────

def prepare_tsuboyama() -> None:
    """Tsuboyama 2023 mega-scale stability → reward table, DPO pairs."""
    import warnings
    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")

    RAW = DATA_DIR / "tsuboyama"
    OUT = DATA_DIR / "prepared"
    CSV = RAW / "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"

    assert CSV.exists(), f"missing {CSV} — run: python data/download.py --dataset tsuboyama --match Processed_K50_dG"

    USECOLS = ["name", "aa_seq", "mut_type", "WT_name", "WT_cluster",
               "dG_ML", "ddG_ML", "Stabilizing_mut"]
    MARGIN           = 1.0   # kcal/mol: min ΔG gap for a confident DPO pair
    MAX_PAIRS_PER_WT = 200

    print(f"loading {CSV.name} …")
    df = pd.read_csv(CSV, usecols=USECOLS, low_memory=False)
    df["dG"]         = df["dG_ML"].map(parse_dG)
    df["ddG"]        = df["ddG_ML"].map(parse_dG)
    df["WT_cluster"] = df["WT_cluster"].astype(str)
    df["is_wt"]      = df["mut_type"].eq("wt")
    df["origin"]     = np.where(df.WT_cluster.str.fullmatch(r"\d+"), "natural", "de_novo")
    print(f"  {len(df):,} rows  |  dG missing: {df.dG.isna().sum():,}")

    OUT.mkdir(parents=True, exist_ok=True)

    # reward table
    reward = (
        df[df.dG.notna()]
        .loc[:, ["name", "WT_name", "origin", "WT_cluster", "mut_type", "is_wt", "aa_seq", "dG", "ddG"]]
        .reset_index(drop=True)
    )
    out = OUT / "reward_table.csv"
    reward.to_csv(out, index=False)
    print(f"reward_table:  {len(reward):,} rows  →  {out}")

    # DPO preference pairs — natural domains only (de novo held out for eval).
    # NOTE: this is the simple default (all natural domains, random val carve done
    # later at train_dpo.py runtime). For a split that also guards against
    # structural redundancy between train/val, run foldseek_split.py then
    # build_dpo_pairs.py instead (see README "2b").
    natural = df[(df.origin == "natural") & df.dG.notna()]
    dpo = sample_preference_pairs(natural, MARGIN, MAX_PAIRS_PER_WT, seed=0)
    out = OUT / "dpo_pairs.csv"
    dpo.to_csv(out, index=False)
    print(f"dpo_pairs:     {len(dpo):,} pairs across {dpo.WT_name.nunique()} natural domains  →  {out}")


# ── registry ─────────────────────────────────────────────────────────────────

DATASETS: dict[str, tuple] = {
    "tsuboyama": (prepare_tsuboyama, "Tsuboyama 2023 mega-scale folding stability"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS), metavar="NAME",
                    help=f"dataset to prepare ({', '.join(DATASETS)})")
    args = ap.parse_args()

    fn, desc = DATASETS[args.dataset]
    print(f"preparing: {desc}")
    fn()
    print("done.")


if __name__ == "__main__":
    main()
