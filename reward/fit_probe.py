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

Split (leakage-free): fit on **natural** domains, report held-out Spearman/RMSE
on the **de novo** domains (topology-code clusters like EEHH/HHH — not in ESM /
ESMFold pretraining). See ../rl_esm_project.md.

Embeddings: local ESM-C via 🤗 transformers (AutoModelForMaskedLM), penultimate
transformer layer, mean-pooled over residue positions (BOS/EOS/pad excluded).
Cached to .npy keyed by (model, layer, sequence set) so re-runs are instant.

Usage (run from the rl_esm/ directory):
    python reward/fit_probe.py                       # defaults: 300M, 20k train / all de novo
    python reward/fit_probe.py --model biohub/ESMC-600M --layer -2
    python reward/fit_probe.py --n-train 40000 --n-eval 20000
    python reward/fit_probe.py --no-cache            # force re-embed
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # rl_esm/
REWARD_CSV = ROOT / "data" / "prepared" / "reward_table.csv"
CACHE_DIR = ROOT / "data" / "prepared" / "embeddings"
OUT_DIR = ROOT / "reward" / "probe_out"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def embed_sequences(
    sequences: list[str],
    model_id: str,
    layer: int,
    batch_size: int,
) -> np.ndarray:
    """Mean-pooled ESM-C hidden state at `layer` for each sequence.

    Returns (N, D) float32. hidden_states is length (n_layers+1); index 0 is the
    embedding layer and -1 the final layer, so layer=-2 is the penultimate
    transformer block. Special tokens (BOS/EOS/pad) are excluded from the mean.
    """
    print(f"Loading {model_id} on {DEVICE} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = (
        AutoModelForMaskedLM.from_pretrained(model_id, dtype="auto").to(DEVICE).eval()
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
    use_cache: bool,
) -> np.ndarray:
    """Embed unique sequences (with an on-disk cache) and expand back to input order."""
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
        emb = embed_sequences(uniq, model_id, layer, batch_size)
        np.savez(cache, seqs=np.array(uniq, dtype=object), emb=emb)
        print(f"Cached → {cache.name}")
        emb_by_seq = dict(zip(uniq, emb))

    return np.stack([emb_by_seq[s] for s in sequences])


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_split(n_train: int | None, n_eval: int | None, seed: int):
    """Natural rows → train, de novo rows → held-out eval. Optional subsample."""
    df = pd.read_csv(REWARD_CSV, usecols=["WT_name", "origin", "aa_seq", "dG"])
    df = df.dropna(subset=["dG", "aa_seq"])
    train = df[df.origin == "natural"]
    heldout = df[df.origin == "de_novo"]

    rng = np.random.RandomState(seed)
    if n_train is not None and len(train) > n_train:
        train = train.iloc[rng.permutation(len(train))[:n_train]]
    if n_eval is not None and len(heldout) > n_eval:
        heldout = heldout.iloc[rng.permutation(len(heldout))[:n_eval]]

    print(f"train (natural): {len(train)} rows, {train.WT_name.nunique()} domains")
    print(f"held-out (de novo): {len(heldout)} rows, {heldout.WT_name.nunique()} domains")
    return train.reset_index(drop=True), heldout.reset_index(drop=True)


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rho = spearmanr(y_true, y_pred).statistic
    r = pearsonr(y_true, y_pred).statistic
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"  {name:18s}  Spearman {rho:.3f}   Pearson {r:.3f}   RMSE {rmse:.3f}")
    return {"split": name, "spearman": rho, "pearson": r, "rmse": rmse, "n": len(y_true)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="biohub/ESMC-300M")
    p.add_argument("--layer", type=int, default=-2,
                   help="hidden_states index (-2 = penultimate transformer layer)")
    p.add_argument("--n-train", type=int, default=20000,
                   help="subsample natural rows (0 = all)")
    p.add_argument("--n-eval", type=int, default=0,
                   help="subsample de novo rows (0 = all)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    assert REWARD_CSV.exists(), f"missing {REWARD_CSV} — run the dataset notebook first"

    train, heldout = load_split(
        args.n_train or None, args.n_eval or None, args.seed
    )

    X_tr = get_embeddings(train.aa_seq.tolist(), args.model, args.layer,
                          args.batch_size, not args.no_cache)
    X_ho = get_embeddings(heldout.aa_seq.tolist(), args.model, args.layer,
                          args.batch_size, not args.no_cache)
    y_tr = train.dG.to_numpy()
    y_ho = heldout.dG.to_numpy()

    # standardise features; ridge with built-in alpha search
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_ho = (X_ho - mu) / sd

    print("\nFitting ridge probe …")
    probe = RidgeCV(alphas=np.logspace(-1, 4, 12))
    probe.fit(X_tr, y_tr)
    print(f"  best alpha: {probe.alpha_:.3g}   (D={X_tr.shape[1]})")

    print("\n=== ΔG probe results ===")
    m_tr = report("train (natural)", y_tr, probe.predict(X_tr))
    m_ho = report("HELD-OUT de novo", y_ho, probe.predict(X_ho))

    gate = m_ho["spearman"]
    verdict = ("PASS — proceed to alignment" if gate >= 0.4
               else "FAIL — fix the oracle before RL")
    print(f"\nGate (held-out Spearman ≥ 0.40): {gate:.3f} → {verdict}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.split('/')[-1]}_L{args.layer}"
    pd.DataFrame([m_tr, m_ho]).to_csv(OUT_DIR / f"metrics_{tag}.csv", index=False)
    pd.DataFrame({
        "WT_name": heldout.WT_name, "aa_seq": heldout.aa_seq,
        "dG_true": y_ho, "dG_pred": probe.predict(X_ho),
    }).to_csv(OUT_DIR / f"heldout_preds_{tag}.csv", index=False)
    print(f"Wrote metrics + held-out predictions → {OUT_DIR}/*_{tag}.csv")


if __name__ == "__main__":
    main()
