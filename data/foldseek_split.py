"""ESM3-paper-style leakage-free split for the Tsuboyama stability dataset.

Replicates the stability-prediction split in docs/esm3.txt (App. A.1.4.4 /
Table S5): domains are clustered *structurally* with Foldseek so that no
structural cluster spans train/val/test, then (optionally) stratified by
ESM-C pseudoperplexity to remove perplexity confounding across splits.

Differences from the paper, and why:
  - pppl is scored with whatever --model you pass (default ESMC-300M, matching
    the reward-probe gate in README); pass --model biohub/ESMC-6B --dtype bf16
    to match the paper's choice exactly (fp32, the "auto" default, doesn't fit
    ESMC-6B alongside activations on a 24GB GPU). Bin edges (low 0-2 / medium
    2-8 / high 8-20) are unchanged regardless of model.
  - the paper additionally subsamples the non-singleton (train) pool down to
    47 families to hit an exact pppl-balanced count; that subsampling
    algorithm isn't specified. We keep the full train pool and just report
    its pppl-bin distribution — trim it yourself if you need exact balance.

Stage 1 (structural, always run):
  1. extract data/AlphaFold_model_PDBs.zip -> data/tsuboyama/alphafold_pdbs/
  2. map each WT_name to its structure file (handles the "|" -> ":" filename
     substitution and the ~78 point-mutant "pseudo-WT" names that reuse their
     parent domain's structure, e.g. "1A0N.pdb_L7S" -> "1A0N.pdb")
  3. `foldseek easy-cluster -c 0.5` over the structures
  4. singleton clusters (1 WT_name) -> eval pool (val/test candidates).
     non-singleton clusters (>1 WT_name structurally matching) -> train, in
     TWO variants written side by side (same eval pool, same val/test split —
     they only differ in how the train side of non-singleton clusters is used):
       - "representative" (paper-exact): keep ONE representative per cluster,
         drop the rest. Non-redundant but throws away most of the data — e.g.
         on this dataset a single 50-domain cluster (many solved structures of
         the same well-studied fold, or one de novo design campaign exploring
         one topology) contributes 1 domain to train and 49 to "excluded".
       - "full": every domain in a non-singleton cluster goes to train. Still
         zero train/eval leakage (redundancy is *within* train, not between
         train and eval) — just doesn't discard the redundant members.

  IMPORTANT: this origin-agnostic clustering does NOT preserve a natural-train
  / de-novo-test boundary (the guarantee that the held-out set is free of ESM
  pretraining leakage, since de novo sequences can't be in ESM's pretraining
  corpus — see data/prepare.py's origin split). Foldseek clusters purely by
  structure — on this dataset, 93% of de novo domains land in non-singleton
  (mostly all-de-novo) clusters and end up in TRAIN, and test becomes a
  natural+de-novo mix. Don't use wt_split_foldseek*.csv if you need that
  pretraining-leakage guarantee — use the denovo-safe variant below instead.

Stage 1b (denovo-safe variant, always run alongside stage 1):
  Preserves the natural-train / de-novo-test guarantee while still deduping
  structural redundancy: every de novo domain -> test (fixed, all 148 of
  them). A natural domain whose structural cluster also contains a de novo
  domain is excluded (it would be a near-duplicate of a test example —
  training on it leaks test-set structure). The remaining "pure natural"
  clusters get the
  same representative/full treatment as stage 1: singleton -> val,
  non-singleton -> train (1 representative, or all members).

Stage 2 (--stratify-pppl, optional):
  5. score each domain's wildtype sequence with masked pseudo-LL (reusing
     align/train_dpo.py's masked_seq_logp) -> pppl = exp(-mean log p/residue)
  6. bin into low/medium/high; used to balance val/test in the origin-agnostic
     (stage 1) variants only — the denovo-safe variants' val/test are already
     fully determined by origin + clustering, so pppl is reported there but
     doesn't change the assignment.

Usage:
  python data/foldseek_split.py                                   # stage 1 only
  python data/foldseek_split.py --stratify-pppl                    # + stage 2
  python data/foldseek_split.py --stratify-pppl --model biohub/ESMC-600M
  python data/foldseek_split.py --stratify-pppl --model biohub/ESMC-6B --dtype bf16   # paper-exact pppl model

Writes four files:
  data/prepared/wt_split_foldseek.csv                    # representative, origin-agnostic (paper-exact)
  data/prepared/wt_split_foldseek_full.csv               # full, origin-agnostic
  data/prepared/wt_split_foldseek_denovo_safe.csv        # representative, de-novo test preserved
  data/prepared/wt_split_foldseek_full_denovo_safe.csv   # full, de-novo test preserved
"""
from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
RAW      = DATA_DIR / "tsuboyama"
OUT      = DATA_DIR / "prepared"
ZIP      = DATA_DIR / "AlphaFold_model_PDBs.zip"
PDB_DIR  = RAW / "alphafold_pdbs"
CSV      = RAW / "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"

COV_THRESHOLD = 0.5   # foldseek -c, per the paper
PPPL_BINS     = [(0.0, 2.0, "low"), (2.0, 8.0, "medium"), (8.0, 20.0, "high")]


# ── stage 1: structures + clustering ─────────────────────────────────────────

def extract_pdbs() -> None:
    if PDB_DIR.exists() and any(PDB_DIR.iterdir()):
        return
    assert ZIP.exists(), f"missing {ZIP} — run: python data/download.py --dataset tsuboyama --match AlphaFold_model_PDBs"
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"extracting {ZIP.name} -> {PDB_DIR} …")
    with zipfile.ZipFile(ZIP) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if not name.endswith(".pdb") or name.startswith("._") or "__MACOSX" in info.filename:
                continue
            with zf.open(info) as src, open(PDB_DIR / name, "wb") as dst:
                dst.write(src.read())
    print(f"  {len(list(PDB_DIR.iterdir())):,} structure files")


def resolve_structure_stems(wt_names: list[str]) -> tuple[dict[str, str], list[str]]:
    """WT_name -> foldseek identifier (filename stem, no .pdb). Handles the
    "|"->":" substitution and point-mutant pseudo-WT names."""
    files = {p.name for p in PDB_DIR.iterdir()}
    mapping: dict[str, str] = {}
    unresolved: list[str] = []
    for wt in wt_names:
        fixed = wt.replace("|", ":")
        if fixed in files:
            mapping[wt] = fixed[:-len(".pdb")]
            continue
        base = fixed.split(".pdb_")[0] + ".pdb"
        if base in files:
            mapping[wt] = base[:-len(".pdb")]
            continue
        unresolved.append(wt)
    return mapping, unresolved


def run_foldseek(work_dir: Path, cov: float) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    prefix = work_dir / "res"
    tmp    = work_dir / "tmp"
    cmd = ["foldseek", "easy-cluster", str(PDB_DIR), str(prefix), str(tmp), "-c", str(cov)]
    print(f"running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return Path(f"{prefix}_cluster.tsv")


def parse_cluster_tsv(tsv_path: Path) -> dict[str, str]:
    """foldseek identifier -> cluster representative identifier."""
    stem_to_rep: dict[str, str] = {}
    with open(tsv_path) as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")
            stem_to_rep[member] = rep
    return stem_to_rep


def structural_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    wt_names = df.WT_name.tolist()
    stem_of, unresolved = resolve_structure_stems(wt_names)
    if unresolved:
        print(f"  WARNING: {len(unresolved)} WT_name(s) have no matching structure, excluding: {unresolved[:10]}{' …' if len(unresolved) > 10 else ''}")

    tsv = run_foldseek(RAW / "foldseek_work", COV_THRESHOLD)
    rep_of_stem = parse_cluster_tsv(tsv)

    df = df[df.WT_name.isin(stem_of)].copy()
    df["stem"] = df.WT_name.map(stem_of)
    df["foldseek_cluster"] = df.stem.map(rep_of_stem)
    n_clusters = df.foldseek_cluster.nunique()
    print(f"  {len(df):,} domains -> {n_clusters:,} structural clusters (coverage -c {COV_THRESHOLD})")

    cluster_size = df.groupby("foldseek_cluster").WT_name.transform("nunique")
    df["cluster_size"] = cluster_size

    rng = np.random.default_rng(seed)
    split_representative = pd.Series("excluded", index=df.index)
    split_full            = pd.Series("train", index=df.index)
    for cluster_rep, g in df.groupby("foldseek_cluster"):
        if g.WT_name.nunique() == 1:
            split_representative.loc[g.index] = "eval_pool"   # singleton -> val/test candidate
            split_full.loc[g.index]           = "eval_pool"
        else:
            rep_wt = rng.choice(g.WT_name.unique())            # non-singleton -> 1 representative to train
            split_representative.loc[g[g.WT_name == rep_wt].index] = "train"
            # split_full: every member of a non-singleton cluster stays "train" (its default)
    df["split_representative"] = split_representative
    df["split_full"]           = split_full
    return df


def denovo_safe_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Alternative to structural_split's assignment that preserves the
    natural-train / de-novo-test pretraining-leakage guarantee. All de novo
    domains -> test. A natural domain sharing a structural cluster with any de
    novo domain is excluded (near-duplicate of a test example). Remaining
    "pure natural" clusters get the same representative/full treatment:
    singleton -> val, non-singleton -> train (1 representative, or all)."""
    df = df.copy()
    cluster_has_denovo = df.groupby("foldseek_cluster").origin.transform(lambda s: (s == "de_novo").any())

    split_representative = pd.Series("excluded", index=df.index)
    split_full            = pd.Series("excluded", index=df.index)
    split_representative.loc[df.origin == "de_novo"] = "test"
    split_full.loc[df.origin == "de_novo"] = "test"

    rng = np.random.default_rng(seed)
    pure_natural = df[(df.origin == "natural") & ~cluster_has_denovo]
    for cluster_rep, g in pure_natural.groupby("foldseek_cluster"):
        if g.WT_name.nunique() == 1:
            split_representative.loc[g.index] = "val"
            split_full.loc[g.index]           = "val"
        else:
            split_full.loc[g.index] = "train"
            rep_wt = rng.choice(g.WT_name.unique())
            split_representative.loc[g[g.WT_name == rep_wt].index] = "train"
    df["split_representative_denovo_safe"] = split_representative
    df["split_full_denovo_safe"]           = split_full
    return df


# ── stage 2: pseudoperplexity stratification ─────────────────────────────────

def pppl_bin(pppl: float) -> str:
    for lo, hi, name in PPPL_BINS:
        if lo <= pppl < hi:
            return name
    return "out_of_range"


def compute_pppl(sequences: list[str], model_id: str, batch_size: int, dtype: str = "auto") -> np.ndarray:
    import sys
    sys.path.insert(0, str(DATA_DIR.parent / "align"))
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from train_dpo import masked_seq_logp
    from scoring import DTYPE_MAP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id, dtype=DTYPE_MAP[dtype]).to(device).eval()
    special_ids = {i for i in (tok.cls_token_id, tok.eos_token_id, tok.pad_token_id) if i is not None}

    mean_logp = masked_seq_logp(model, tok, sequences, special_ids, batch_size, length_norm=True)
    return np.exp(-mean_logp)


def stratify_pppl(df: pd.DataFrame, model_id: str, batch_size: int, val_frac: float, seed: int, dtype: str = "auto") -> pd.DataFrame:
    """Bins by pppl and assigns val/test on the (shared) eval pool. Both variant
    columns get identical val/test labels — they only differ on the train side."""
    print(f"scoring pseudoperplexity with {model_id} (dtype={dtype}) …")
    pppl = compute_pppl(df.aa_seq.tolist(), model_id, batch_size, dtype)
    df = df.copy()
    df["pppl"] = pppl
    df["pppl_bin"] = df.pppl.map(pppl_bin)

    print("  representative-variant train pool pppl-bin distribution:")
    print(df[df.split_representative == "train"].pppl_bin.value_counts().to_string())

    rng = np.random.default_rng(seed)
    eval_mask = df.split_representative == "eval_pool"   # identical index set to split_full == "eval_pool"
    for _, g in df[eval_mask].groupby("pppl_bin"):
        idx = rng.permutation(g.index.to_numpy())
        n_val = round(len(idx) * val_frac)
        for col in ("split_representative", "split_full"):
            df.loc[idx[:n_val], col] = "val"
            df.loc[idx[n_val:], col] = "test"
    return df


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stratify-pppl", action="store_true", help="run stage 2 (pppl-balanced val/test split)")
    ap.add_argument("--model", default="biohub/ESMC-300M", help="ESM-C model for pppl scoring")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                    help="model load dtype; ESMC-6B needs bf16 (fp32 alone is ~23GB)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of the eval pool assigned to val (rest -> test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert CSV.exists(), f"missing {CSV} — run: python data/download.py --dataset tsuboyama --match Processed_K50_dG"

    print("loading domain table …")
    raw = pd.read_csv(CSV, usecols=["WT_name", "WT_cluster", "mut_type", "aa_seq"], low_memory=False)
    raw["WT_cluster"] = raw["WT_cluster"].astype(str)
    # one row per domain: prefer the mut_type=="wt" row (a handful of domains
    # lack one, e.g. entries only present as mutants -> fall back to any row)
    wt = (
        raw.sort_values("mut_type", key=lambda s: s.ne("wt"))
        .drop_duplicates("WT_name")
        .reset_index(drop=True)
    )
    wt["origin"] = np.where(wt.WT_cluster.str.fullmatch(r"\d+"), "natural", "de_novo")
    print(f"  {len(wt):,} unique WT domains ({wt.origin.value_counts().to_dict()})")

    extract_pdbs()
    df = structural_split(wt, args.seed)
    df = denovo_safe_split(df, args.seed)

    if args.stratify_pppl:
        df = stratify_pppl(df, args.model, args.batch_size, args.val_frac, args.seed, args.dtype)
    else:
        for col in ("split_representative", "split_full"):
            df.loc[df[col] == "eval_pool", col] = "test"   # no stratification: eval pool -> test wholesale

    OUT.mkdir(parents=True, exist_ok=True)
    base_cols = ["WT_name", "origin", "WT_cluster", "foldseek_cluster", "cluster_size"]
    if "pppl" in df.columns:
        base_cols += ["pppl", "pppl_bin"]

    for variant, col, fname in [
        ("representative",             "split_representative",             "wt_split_foldseek.csv"),
        ("full",                       "split_full",                       "wt_split_foldseek_full.csv"),
        ("representative, denovo-safe", "split_representative_denovo_safe", "wt_split_foldseek_denovo_safe.csv"),
        ("full, denovo-safe",           "split_full_denovo_safe",           "wt_split_foldseek_full_denovo_safe.csv"),
    ]:
        out_path = OUT / fname
        out_df = df[base_cols].copy()
        out_df["split"] = df[col]
        out_df.to_csv(out_path, index=False)
        print(f"\nwt_split_foldseek ({variant}): {out_df.split.value_counts().to_dict()}  ->  {out_path}")


if __name__ == "__main__":
    main()
