"""
download_fireprot.py
─────────────────────────────────────────────────────────────────────────────
Download + prepare the **homolog-free FireProt** ΔΔG dataset for held-out
evaluation of the stability-aligned ESM-C policy.

WHY THIS FILE / PROVENANCE
──────────────────────────
The ESM3 paper (docs/esm3.txt, App. A.1.4.4 "Learned stability directions
generalize to new datasets") validates its Megascale-trained stability signal on
FireProt — *specifically the homolog-free version created in ThermoMPNN* (Dieckhaus
et al.), which was filtered to remove any FireProt sequence with >25% identity to
the Megascale (Tsuboyama) training data. That filtered file is
`data_all/testing/fireprot_HF.csv` in the ThermoMPNN repo (HF = Homolog-Free).

We pull that CSV directly from a **pinned commit** of the ThermoMPNN GitHub repo
(no auth, ~a few MB). We deliberately do NOT use the raw Zenodo FireProtDB dump
(zenodo 8169289, `fireprot_upload.zip`) — that is ThermoMPNN's *input*, before the
homology filtering the paper relies on. Using the raw dump would silently
reintroduce train/eval leakage against Megascale.

    source repo   : https://github.com/Kuhlman-Lab/ThermoMPNN
    source file   : data_all/testing/fireprot_HF.csv
    pinned commit : 2b04fd370e399911b1fa5848112cc9013f084110  (2026-03-26)

To re-pin later, bump COMMIT below to a newer SHA and re-run.

WHAT IT PRODUCES
────────────────
`data/prepared/fireprot_eval.csv`, one row per single-point mutant, columns chosen
to mirror `reward_table.csv` so the existing scoring code in align/eval_base.py /
align/train_dpo.py can consume it unchanged:

    WT_name    protein id (ThermoMPNN's `pdb_id_corrected`) — the per-protein group key
    mut_type   e.g. "A123V"  (1-indexed, like Tsuboyama's mut_type)
    wt_seq     wildtype sequence (ThermoMPNN's `pdb_sequence`)
    aa_seq     mutant sequence  (wt_seq with the single substitution applied)
    ddG        experimental ΔΔG, ThermoMPNN convention: **lower ddG = MORE stable**
    position   1-indexed residue position
    wild_type  WT amino acid
    mutation   mutant amino acid
    pH         assay pH (kept for optional stratification)

SIGN CONVENTION (read this before interpreting Spearman)
────────────────────────────────────────────────────────
In `fireprot_HF.csv` (and in the output here) **lower ddG = MORE stable** — the
standard ΔΔG_folding convention where a negative ΔΔG is stabilizing. Evidence in
ThermoMPNN itself: their ProteinMPNN-as-predictor baseline
(analysis/thermompnn_benchmarking.py) sets model `ddG = -log_likelihood` (higher
likelihood = more stable → lower ddG) to make it correlate *positively* with the
ground-truth column, and their datasets.py negates Megascale's ddG_ML to bring it
into this same "negative = stabilizing" convention.

Consequence: a stability-tracking score (ESM-C pseudo-log-likelihood, or the
aligned policy's pseudo-LL) correlates **negatively** with raw `ddG`. eval_fireprot.py
therefore reports a *stability-oriented* Spearman (score vs −ddG) so that positive =
tracking stability correctly, and it confirms the base model comes out positive
before trusting the aligned-model delta.

Because ΔΔG is only meaningful *within* a protein (scores across different
wildtypes/lengths aren't comparable), evaluation must be **per-protein Spearman**,
then averaged over proteins — never one global Spearman over all rows.

USAGE (from rl_esm/)
────────────────────
    pixi run python data/download_fireprot.py            # download + prepare
    pixi run python data/download_fireprot.py --list     # just show the raw schema
    pixi run python data/download_fireprot.py --keep-raw # also save the raw CSV
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent
PREP_DIR = DATA_DIR / "prepared"
RAW_DEST = DATA_DIR / "fireprot"

# Pinned ThermoMPNN source — bump COMMIT to re-pin (see module docstring).
COMMIT = "2b04fd370e399911b1fa5848112cc9013f084110"
REPO = "Kuhlman-Lab/ThermoMPNN"
CSV_PATH = "data_all/testing/fireprot_HF.csv"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{CSV_PATH}"

AA = set("ACDEFGHIKLMNPQRSTVWY")


def fetch_raw() -> pd.DataFrame:
    """Download fireprot_HF.csv from the pinned commit into a DataFrame."""
    print(f"[get ] {RAW_URL}")
    r = requests.get(RAW_URL, timeout=120)
    r.raise_for_status()
    RAW_DEST.mkdir(parents=True, exist_ok=True)
    raw_file = RAW_DEST / "fireprot_HF.csv"
    raw_file.write_bytes(r.content)
    print(f"[ok  ] saved raw → {raw_file}  ({len(r.content)/1e6:.2f} MB)")
    return pd.read_csv(raw_file)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Turn ThermoMPNN's fireprot_HF schema into a reward_table-style eval table.

    Mirrors the parsing/asserts ThermoMPNN's FireProtDataset uses (datasets.py):
    group by `pdb_id_corrected`; the wildtype sequence is `pdb_sequence`; each
    mutation is (`wild_type`, `mutation`) at 0-indexed `pdb_position`, and must
    agree with the wildtype sequence at that position. We build the mutant
    sequence by applying that single substitution.
    """
    need = ["pdb_id_corrected", "pdb_sequence", "wild_type", "mutation",
            "pdb_position", "ddG", "position", "pH"]
    missing = [c for c in need if c not in df.columns]
    assert not missing, f"fireprot_HF.csv is missing expected columns: {missing}"

    df = df.dropna(subset=["ddG", "pdb_sequence", "pdb_position"]).copy()
    df["pdb_position"] = df["pdb_position"].astype(int)

    rows = []
    dropped = 0
    for _, r in df.iterrows():
        wt_seq = str(r.pdb_sequence)
        idx = int(r.pdb_position)                      # 0-indexed into pdb_sequence
        wt_aa, mut_aa = str(r.wild_type), str(r.mutation)

        # Skip anything that isn't a clean single substitution of standard AAs.
        if (not (0 <= idx < len(wt_seq)) or wt_aa not in AA or mut_aa not in AA
                or wt_aa == mut_aa or wt_seq[idx] != wt_aa):
            dropped += 1
            continue

        mut_seq = wt_seq[:idx] + mut_aa + wt_seq[idx + 1:]
        rows.append({
            "WT_name": r.pdb_id_corrected,
            "mut_type": f"{wt_aa}{idx + 1}{mut_aa}",    # 1-indexed, Tsuboyama style
            "wt_seq": wt_seq,
            "aa_seq": mut_seq,
            "ddG": float(r.ddG),
            "position": idx + 1,
            "wild_type": wt_aa,
            "mutation": mut_aa,
            "pH": r.pH,
        })

    out = pd.DataFrame(rows).drop_duplicates(
        subset=["WT_name", "mut_type", "ddG"]).reset_index(drop=True)
    print(f"[prep] kept {len(out)} single-mutant rows "
          f"across {out.WT_name.nunique()} proteins "
          f"(dropped {dropped} non-single/invalid rows)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="download and print the raw schema/head, then exit")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep the raw fireprot_HF.csv under data/fireprot/ (deleted by "
                         "default — it's a 1.9 MB re-fetchable download, and the prepared "
                         "data/prepared/fireprot_eval.csv is what the evals read)")
    ap.add_argument("--out", type=Path, default=PREP_DIR / "fireprot_eval.csv")
    args = ap.parse_args()

    raw = fetch_raw()
    if args.list:
        print("\ncolumns:", list(raw.columns))
        print("\nhead:\n", raw.head().to_string())
        return

    prepared = prepare(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.out, index=False)
    print(f"[done] wrote {args.out}")
    print("\nper-protein mutation counts (top 10):")
    print(prepared.WT_name.value_counts().head(10).to_string())

    if not args.keep_raw:
        (RAW_DEST / "fireprot_HF.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
