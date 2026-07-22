"""
fit_probe.py
─────────────────────────────────────────────────────────────────────────────
v1 reward oracle for the RL-ESM project: a *linear ridge probe* on top of
**frozen** ESM-C penultimate-layer embeddings that predicts folding stability
ΔG from sequence alone.

Why linear-on-frozen: the ESM3 paper (App. A.1.4.4) shows Megascale is not ESM
pretraining data, and a ridge probe on frozen ESM-C reps already predicts ΔG at
Spearman ~0.68–0.8 — on par with FoldX/Rosetta. So the reward is a cheap, solved
component; the project's contribution is the DPO-vs-GRPO alignment comparison,
not the predictor. This script is the **gate**: if held-out Spearman < ~0.4,
fix the oracle before touching any alignment.

Two ways to split train/val/test:
  default (no --split-file)  natural domains -> train, de novo -> held-out test,
                              no val (ridge penalty picked by RidgeCV's internal
                              k-fold CV on train). Simple, leakage-free by origin.
  --split-file <csv>         consume a WT_name -> split (train/val/test) column
                              from data/foldseek_split.py's paper-style structural
                              + pseudoperplexity split (docs/esm3.txt App. A.1.4.4,
                              Table S5) instead. Ridge penalty is then swept and
                              picked by **val Spearman** (matching the paper's
                              selection criterion), not internal CV.
  Use data/prepared/wt_split_foldseek.csv (the origin-agnostic "representative"
  variant) to match the paper — its split is purely structural, not natural-vs-
  de-novo, so it deliberately mixes origins across train/val/test. Don't
  substitute the *_denovo_safe.csv variant here: that one exists for this
  repo's own DPO pretraining-leakage guarantee, not for reproducing the
  paper's numbers.

Embeddings: local ESM-C via 🤗 transformers (AutoModelForMaskedLM), penultimate
transformer layer, mean-pooled over residue positions (BOS/EOS/pad excluded).
Cached to .npy keyed by (model, layer, sequence set) so re-runs are instant.

Usage (run from the rl_esm/ directory):
    python reward/fit_probe.py                       # defaults: 300M, 20k train / all de novo
    python reward/fit_probe.py --model biohub/ESMC-600M --layer -2
    python reward/fit_probe.py --model biohub/ESMC-6B --dtype bf16   # fp32 alone OOMs a 24GB GPU
    python reward/fit_probe.py --split-file data/prepared/wt_split_foldseek.csv --n-train 0 --n-eval 0
    python reward/fit_probe.py --config reward/configs/fit_probe_300M.yaml
    python reward/fit_probe.py --no-cache            # force re-embed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # rl_esm/
REWARD_CSV = ROOT / "data" / "prepared" / "reward_table.csv"
CACHE_DIR = ROOT / "data" / "prepared" / "embeddings"
OUT_DIR = ROOT / "reward" / "probe_out"

sys.path.insert(0, str(ROOT / "align"))
from scoring import DTYPE_MAP, apply_config, coerce_paths, git_sha  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# paper (docs/esm3.txt App. A.1.4.4): "ridge penalty was tuned on the validation
# set, searching between 10^-3 and 10^12" — one point per decade over that range.
ALPHAS = np.logspace(-3, 12, 16)


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def embed_sequences(
    sequences: list[str],
    model_id: str,
    layer: int,
    batch_size: int,
    dtype: str,
) -> np.ndarray:
    """Mean-pooled ESM-C hidden state at `layer` for each sequence.

    Returns (N, D) float32. hidden_states is length (n_layers+1); index 0 is the
    embedding layer and -1 the final layer, so layer=-2 is the penultimate
    transformer block. Special tokens (BOS/EOS/pad) are excluded from the mean.
    """
    print(f"Loading {model_id} (dtype={dtype}) on {DEVICE} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = (
        AutoModelForMaskedLM.from_pretrained(model_id, dtype=DTYPE_MAP[dtype]).to(DEVICE).eval()
    )
    special_ids = torch.tensor(
        [i for i in (tokenizer.cls_token_id, tokenizer.eos_token_id,
                     tokenizer.pad_token_id) if i is not None],
        device=DEVICE,
    )

    embs: list[np.ndarray] = []
    for start in tqdm(range(0, len(sequences), batch_size),
                      desc="  embedding", unit="batch"):
        batch = sequences[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True).to(DEVICE)
        with torch.inference_mode():
            hs = model(**enc, output_hidden_states=True).hidden_states[layer]  # (B,L,D)

        # residue mask: attended positions that are not special tokens
        mask = enc["attention_mask"].bool() & ~torch.isin(enc["input_ids"], special_ids)
        mask = mask.unsqueeze(-1)                       # (B,L,1)
        summed = (hs * mask).sum(dim=1)                 # (B,D)
        counts = mask.sum(dim=1).clamp(min=1)           # (B,1)
        pooled = (summed / counts).float().cpu().numpy()
        embs.append(pooled)

    return np.concatenate(embs, axis=0)


def get_embeddings(
    sequences: list[str],
    model_id: str,
    layer: int,
    batch_size: int,
    dtype: str,
    use_cache: bool,
) -> np.ndarray:
    """Embed unique sequences (with an on-disk cache) and expand back to input order."""
    if not sequences:
        return np.empty((0, 0), dtype=np.float32)
    uniq = sorted(set(sequences))
    key = hashlib.sha1(
        (model_id + f"|L{layer}|" + "\n".join(uniq)).encode()
    ).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{model_id.split('/')[-1]}_L{layer}_{key}.npz"

    if use_cache and cache.exists():
        print(f"Cache hit: {cache.name}")
        z = np.load(cache, allow_pickle=True)
        emb_by_seq = dict(zip(z["seqs"], z["emb"]))
    else:
        print(f"Embedding {len(uniq)} unique sequences "
              f"(of {len(sequences)} total) …")
        emb = embed_sequences(uniq, model_id, layer, batch_size, dtype)
        np.savez(cache, seqs=np.array(uniq, dtype=object), emb=emb)
        print(f"Cached → {cache.name}")
        emb_by_seq = dict(zip(uniq, emb))

    return np.stack([emb_by_seq[s] for s in sequences])


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def _subsample(part: pd.DataFrame, n: int | None, rng: np.random.RandomState) -> pd.DataFrame:
    if n and len(part) > n:
        part = part.iloc[rng.permutation(len(part))[:n]]
    return part.reset_index(drop=True)


def load_split(split_file: Path | None, n_train: int | None, n_eval: int | None, seed: int):
    """Returns (train, val, test). Without --split-file: natural -> train,
    de novo -> test, val is empty (legacy behavior). With --split-file: consumes
    its WT_name -> split (train/val/test) column instead of natural/de-novo
    origin — see module docstring for which split-file variant to use."""
    df = pd.read_csv(REWARD_CSV, usecols=["WT_name", "origin", "aa_seq", "dG"])
    df = df.dropna(subset=["dG", "aa_seq"])

    if split_file is None:
        train = df[df.origin == "natural"]
        val = df.iloc[0:0]
        test = df[df.origin == "de_novo"]
    else:
        split_df = pd.read_csv(split_file, usecols=["WT_name", "split"])
        df = df.merge(split_df, on="WT_name", how="inner")
        train = df[df.split == "train"]
        val = df[df.split == "val"]
        test = df[df.split == "test"]

    rng = np.random.RandomState(seed)
    train = _subsample(train, n_train, rng)
    val = val.reset_index(drop=True)
    test = _subsample(test, n_eval, rng)

    for name, part in (("train", train), ("val", val), ("test", test)):
        if len(part):
            print(f"{name:5s}: {len(part)} rows, {part.WT_name.nunique()} domains")
    return train, val, test


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rho = spearmanr(y_true, y_pred).statistic
    r = pearsonr(y_true, y_pred).statistic
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"  {name:18s}  Spearman {rho:.3f}   Pearson {r:.3f}   RMSE {rmse:.3f}")
    return {"split": name, "spearman": rho, "pearson": r, "rmse": rmse, "n": len(y_true)}


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
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="model load dtype; ESMC-6B needs bf16 (fp32 alone is ~23GB)")
    p.add_argument("--layer", type=int, default=-2,
                   help="hidden_states index (-2 = penultimate transformer layer)")
    p.add_argument("--split-file", type=Path, default=None,
                   help="WT_name -> split (train/val/test) csv from data/foldseek_split.py; "
                        "omit to keep the natural-vs-de-novo split (no val)")
    p.add_argument("--n-train", type=int, default=20000,
                   help="subsample train rows (0 = all)")
    p.add_argument("--n-eval", type=int, default=0,
                   help="subsample test/held-out rows (0 = all)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true")

    if cfg_args.config:
        apply_config(p, cfg_args.config)

    args = p.parse_args(remaining)
    args.config = cfg_args.config
    return coerce_paths(args, "split_file")


def main(argv=None) -> None:
    args = build_args(argv)

    assert REWARD_CSV.exists(), f"missing {REWARD_CSV} — run the dataset notebook first"
    if args.split_file is not None:
        assert args.split_file.exists(), f"missing split file {args.split_file}"

    train, val, test = load_split(
        args.split_file, args.n_train or None, args.n_eval or None, args.seed
    )

    X_tr = get_embeddings(train.aa_seq.tolist(), args.model, args.layer,
                          args.batch_size, args.dtype, not args.no_cache)
    X_ho = get_embeddings(test.aa_seq.tolist(), args.model, args.layer,
                          args.batch_size, args.dtype, not args.no_cache)
    y_tr = train.dG.to_numpy()
    y_ho = test.dG.to_numpy()

    has_val = len(val) > 0
    if has_val:
        X_val = get_embeddings(val.aa_seq.tolist(), args.model, args.layer,
                               args.batch_size, args.dtype, not args.no_cache)
        y_val = val.dG.to_numpy()

    # standardise features (fit on train only)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_ho = (X_ho - mu) / sd
    if has_val:
        X_val = (X_val - mu) / sd

    print("\nFitting ridge probe …")
    if has_val:
        # paper-style: sweep alpha, select by val Spearman (not internal k-fold CV)
        best_alpha, best_rho = None, -np.inf
        for alpha in ALPHAS:
            m = Ridge(alpha=alpha).fit(X_tr, y_tr)
            r = spearmanr(y_val, m.predict(X_val)).statistic
            if r > best_rho:
                best_alpha, best_rho = alpha, r
        print(f"  best alpha: {best_alpha:.3g}   (val Spearman {best_rho:.3f}, D={X_tr.shape[1]})")
        probe = Ridge(alpha=best_alpha).fit(X_tr, y_tr)
    else:
        probe = RidgeCV(alphas=ALPHAS)
        probe.fit(X_tr, y_tr)
        best_alpha = probe.alpha_
        print(f"  best alpha: {best_alpha:.3g}   (D={X_tr.shape[1]}, internal CV)")

    print("\n=== ΔG probe results ===")
    metrics = [report("train", y_tr, probe.predict(X_tr))]
    if has_val:
        metrics.append(report("val", y_val, probe.predict(X_val)))
    heldout_name = "test" if args.split_file is not None else "HELD-OUT de novo"
    metrics.append(report(heldout_name, y_ho, probe.predict(X_ho)))

    gate = metrics[-1]["spearman"]
    verdict = ("PASS — proceed to alignment" if gate >= 0.4
               else "FAIL — fix the oracle before RL")
    print(f"\nGate (held-out Spearman ≥ 0.40): {gate:.3f} → {verdict}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.split('/')[-1]}_L{args.layer}"
    if args.split_file is not None:
        tag += f"_{args.split_file.stem}"

    pd.DataFrame(metrics).to_csv(OUT_DIR / f"metrics_{tag}.csv", index=False)
    pd.DataFrame({
        "WT_name": test.WT_name, "aa_seq": test.aa_seq,
        "dG_true": y_ho, "dG_pred": probe.predict(X_ho),
    }).to_csv(OUT_DIR / f"heldout_preds_{tag}.csv", index=False)

    config_dump = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config_dump.update(
        git_sha=git_sha(), device=DEVICE, ridge_alpha=float(best_alpha),
        n_train=len(train), n_val=len(val), n_test=len(test),
    )
    (OUT_DIR / f"config_{tag}.json").write_text(json.dumps(config_dump, indent=2, default=str))

    print(f"Wrote metrics + held-out predictions + config → {OUT_DIR}/*_{tag}.*")


if __name__ == "__main__":
    main()
