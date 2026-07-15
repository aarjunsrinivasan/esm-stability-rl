"""
eval_fireprot.py
─────────────────────────────────────────────────────────────────────────────
Held-out generalization check: does the stability-aligned ESM-C policy rank
FireProt ΔΔG mutants better than the base model?

FireProt is the paper's held-out oracle (docs/esm3.txt App. A.1.4.4) — the
homolog-free version (>25% identity to Megascale removed) that appears in NO part
of the DPO reward. Prepare it first with:

    pixi run python data/download_fireprot.py     # → data/prepared/fireprot_eval.csv

WHAT THIS MEASURES
──────────────────
For each protein, a stability-oriented Spearman over its mutants, then averaged
across proteins (mean and median). ΔΔG is only comparable *within* a wildtype, so
this is strictly per-protein — never a global Spearman.

SIGN: in fireprot_eval.csv **lower ddG = more stable** (standard ΔΔG_folding
convention; see download_fireprot.py). So we report ρ = Spearman(pseudo-LL, −ddG),
i.e. a *stability-oriented* correlation where **positive = the score correctly
tracks stability** (higher pseudo-LL ↔ more stable). The script prints the
base-model value so you can confirm it's positive before trusting the aligned delta
— matching eval_base.py, where positive ρ on dG also means "tracks stability".

This is the FireProt analog of the de novo `test`-split Spearman in eval_base.py —
same scoring path (seq_logp / masked_seq_logp imported from train_dpo.py, no
duplication), so numbers are directly comparable.

BASE vs ALIGNED IN ONE PASS
───────────────────────────
Pass --adapter and the script scores BOTH the aligned policy (adapter on) and the
base model (peft `disable_adapter()`, i.e. identical weights minus LoRA) on the
exact same sequences, so every per-protein delta is apples-to-apples. Omit
--adapter to characterize just the base model.

Two scorings, both per protein:
  single  one forward pass/seq          (the cheap proxy DPO trains on; all mutants)
  masked  mask each residue once        (L fwd passes/seq, no self-leakage; subsampled)

Usage (from rl_esm/):
    pixi run python align/eval_fireprot.py \
        --adapter align/dpo_out/runs/base_full_20260710/20260710_082958_b0.1_lr1e-04_bs16_r8/best
    pixi run python align/eval_fireprot.py --config align/configs/eval_fireprot.yaml
    pixi run python align/eval_fireprot.py --adapter <dir> --max-proteins 3 --mask-n-proteins 0  # quick smoke

Writes align/dpo_out/fireprot_eval/<timestamp>/ : per_protein.csv (one row per
protein × model × scoring), summary.csv (aggregate ρ per model × scoring), summary.md.
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from train_dpo import DEVICE, PREP, git_sha, masked_seq_logp, seq_logp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIREPROT_CSV = PREP / "fireprot_eval.csv"
OUT_DIR = ROOT / "align" / "dpo_out" / "fireprot_eval"


# ─────────────────────────────────────────────────────────────────────────────
# SCORING  (identical path to eval_base.py)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def score_single(model, tokenizer, seqs, special_ids, batch_size, length_norm):
    out = []
    for s in tqdm(range(0, len(seqs), batch_size), desc="  single", leave=False):
        enc = tokenizer(seqs[s:s + batch_size], return_tensors="pt",
                        padding=True).to(DEVICE)
        lp = seq_logp(model, enc["input_ids"], enc["attention_mask"],
                      special_ids, length_norm)
        out.append(lp.cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def _stab_rho(score, ddG):
    """Stability-oriented Spearman: corr(score, −ddG), so positive = score tracks
    stability (lower ddG = more stable). Over finite rows; nan if <3 usable rows or
    either side constant."""
    x, y = np.asarray(score, float), -np.asarray(ddG, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _adapter_ctx(model, aligned):
    """aligned=True → use the LoRA policy as-is; False → disable adapters (base)."""
    if aligned or not hasattr(model, "disable_adapter"):
        return nullcontext()
    return model.disable_adapter()


# ─────────────────────────────────────────────────────────────────────────────
# PER-PROTEIN EVAL
# ─────────────────────────────────────────────────────────────────────────────

def per_protein_eval(model, tokenizer, special_ids, special_set, args, models):
    """models: list of ('base'|'aligned', aligned_bool) to score. Returns a tidy
    per-(protein × model × scoring) DataFrame."""
    df = pd.read_csv(FIREPROT_CSV)
    df = df.dropna(subset=["ddG", "aa_seq"])
    counts = df.WT_name.value_counts()
    keep = counts[counts >= args.min_muts].index
    df = df[df.WT_name.isin(keep)].copy()

    proteins = list(df.WT_name.drop_duplicates())
    if args.max_proteins:
        proteins = proteins[:args.max_proteins]
    # proteins to also score masked: the largest by mutation count (most informative ρ)
    mask_set = set(counts.reindex(proteins).sort_values(ascending=False)
                   .head(args.mask_n_proteins).index) if args.mask_n_proteins else set()

    rng = np.random.RandomState(args.seed)
    rows = []
    for wt in tqdm(proteins, desc="proteins"):
        g = df[df.WT_name == wt]
        seqs, ddG = g.aa_seq.tolist(), g.ddG.to_numpy()

        for name, aligned in models:
            with _adapter_ctx(model, aligned):
                single = score_single(model, tokenizer, seqs, special_ids,
                                      args.batch_size, args.length_norm)
            rows.append({"WT_name": wt, "model": name, "scoring": "single",
                         "n": len(seqs), "spearman": _stab_rho(single, ddG)})

            if wt in mask_set:
                m = min(args.mask_per_protein, len(seqs)) if args.mask_per_protein else len(seqs)
                idx = rng.choice(len(seqs), size=m, replace=False)
                sub_seqs = [seqs[i] for i in idx]
                with _adapter_ctx(model, aligned):
                    masked = masked_seq_logp(model, tokenizer, sub_seqs, special_set,
                                             args.batch_size, args.length_norm)
                rows.append({"WT_name": wt, "model": name, "scoring": "masked",
                             "n": m, "spearman": _stab_rho(masked, ddG[idx])})
    return pd.DataFrame(rows)


def aggregate(per_protein):
    """Mean/median per-protein Spearman for each model × scoring, over proteins
    where ρ is defined."""
    rows = []
    for (model, scoring), g in per_protein.groupby(["model", "scoring"]):
        rho = g.spearman.dropna()
        rows.append({"model": model, "scoring": scoring, "n_proteins": len(rho),
                     "mean_spearman": rho.mean(), "median_spearman": rho.median(),
                     "frac_positive": float((rho > 0).mean())})
    return pd.DataFrame(rows).sort_values(["scoring", "model"]).reset_index(drop=True)


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
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-muts", type=int, default=10,
                   help="skip proteins with fewer mutants than this (per-protein ρ needs enough points)")
    p.add_argument("--max-proteins", type=int, default=0, help="cap #proteins scored (0=all; for smoke)")
    p.add_argument("--mask-n-proteins", type=int, default=10,
                   help="also score masked pseudo-LL for the N largest proteins (0=skip masked)")
    p.add_argument("--mask-per-protein", type=int, default=40,
                   help="cap mutants scored masked per protein (0=all mutants of that protein)")
    p.add_argument("--length-norm", action="store_true",
                   help="length-normalize pseudo-LL (match the train_dpo.py run)")
    p.add_argument("--seed", type=int, default=0)

    if cfg_args.config:
        assert cfg_args.config.exists(), f"missing config {cfg_args.config}"
        cfg = yaml.safe_load(cfg_args.config.read_text()) or {}
        known = {a.dest for a in p._actions}
        unknown = set(cfg) - known
        assert not unknown, f"unknown key(s) in {cfg_args.config}: {unknown}"
        p.set_defaults(**cfg)

    args = p.parse_args(remaining)
    args.config = cfg_args.config
    if args.adapter is not None:
        args.adapter = Path(args.adapter)
    return args


def main(argv=None):
    args = build_args(argv)
    assert FIREPROT_CSV.exists(), \
        f"missing {FIREPROT_CSV} — run: pixi run python data/download_fireprot.py"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading base {args.model} on {DEVICE} …")
    model = AutoModelForMaskedLM.from_pretrained(args.model, dtype="auto").to(DEVICE)
    models = [("base", False)]
    if args.adapter is not None:
        from peft import PeftModel
        assert args.adapter.exists(), f"missing adapter dir {args.adapter}"
        print(f"Wrapping with LoRA adapter (aligned policy) → {args.adapter}")
        model = PeftModel.from_pretrained(model, str(args.adapter)).to(DEVICE)
        models = [("aligned", True), ("base", False)]   # base = adapters disabled, same weights
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    special_ids = torch.tensor(
        [i for i in (tokenizer.cls_token_id, tokenizer.eos_token_id,
                     tokenizer.pad_token_id) if i is not None], device=DEVICE)
    special_set = set(special_ids.tolist())

    per_protein = per_protein_eval(model, tokenizer, special_ids, special_set, args, models)
    summ = aggregate(per_protein)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / ts
    out.mkdir(parents=True, exist_ok=True)
    per_protein.to_csv(out / "per_protein.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)

    config_dump = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config_dump.update(git_sha=git_sha(), device=DEVICE, timestamp=ts,
                       fireprot_csv=str(FIREPROT_CSV))
    (out / "config.json").write_text(json.dumps(config_dump, indent=2, default=str))

    base_single = summ[(summ.model == "base") & (summ.scoring == "single")]
    sign_note = ""
    if len(base_single):
        ms = base_single.mean_spearman.iloc[0]
        sign_note = (f"\nbase single-pass mean ρ (stability-oriented, score vs −ddG) = {ms:+.4f} "
                     f"({'positive ✓ expected — base pseudo-LL already tracks stability' if ms > 0 else 'NEGATIVE ⚠ unexpected — base pseudo-LL should track stability; check data/sign'})")

    lines = [f"# FireProt held-out eval — {args.model}"
             + (f"  + LoRA {args.adapter}" if args.adapter is not None else " (base only)"),
             f"git_sha {git_sha()}  |  {ts}  |  length_norm={args.length_norm}",
             f"min_muts={args.min_muts}  mask_n_proteins={args.mask_n_proteins}  "
             f"mask_per_protein={args.mask_per_protein}",
             "", "## Aggregate per-protein Spearman(pseudo-LL, ddG)", "",
             summ.round(4).to_string(index=False), sign_note]
    if args.adapter is not None:
        # explicit base→aligned delta on the headline metric
        piv = summ.pivot_table(index="scoring", columns="model", values="mean_spearman")
        if {"base", "aligned"}.issubset(piv.columns):
            piv["delta_aligned_minus_base"] = piv["aligned"] - piv["base"]
            lines += ["", "## Base → aligned delta (mean per-protein ρ)", "",
                      piv.round(4).to_string()]
    summary = "\n".join(lines)
    (out / "summary.md").write_text(summary)
    print("\n" + summary)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
