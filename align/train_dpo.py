"""
train_dpo.py
─────────────────────────────────────────────────────────────────────────────
Offline arm of the RL-ESM comparison: align ESM-C to folding stability with
**Direct Preference Optimization** on the Tsuboyama preference pairs.

Why a custom loop (not TRL DPOTrainer): ESM-C is a **masked / bidirectional**
LM, not an autoregressive one, so `log π(seq) = Σ log p(x_i | x_<i)` doesn't
apply and TRL's causal-LM machinery doesn't fit. We instead define the sequence
log-probability as the **single-pass pseudo-log-likelihood**

    log π(seq) = Σ_i  log softmax( model(seq).logits )[i, x_i]           (1 fwd pass)

summed over residue positions (BOS/EOS/pad excluded). This is the standard cheap
differentiable MLM fitness proxy. (The more rigorous masked pseudo-LL masks each
position — L× more forward passes — used only for final eval, not the loop.)

DPO loss (Rafailov et al. 2023), with the pseudo-LL as the log-prob:

    Δ_chosen   = log π(chosen)   − log π_ref(chosen)
    Δ_rejected = log π(rejected) − log π_ref(rejected)
    L = − log σ( β · (Δ_chosen − Δ_rejected) )

Policy = ESM-C + LoRA (peft). Reference = the same base weights with adapters
disabled; ref log-probs are **precomputed once** for both splits (train and val
alike — the pairs in each are fixed) so the training loop only ever forwards the
policy, gathering the matching precomputed reference values by row index instead
of re-running the frozen base model every step.

Splits (all group-disjoint by WT_name — no domain appears in two splits):
  • train pairs   — ~90% of the 331 natural WT domains  (dpo_pairs.csv)
  • val   pairs   — the other ~10% of natural domains    → in-training metrics
  • TEST          — the 148 de novo domains (reward_table.csv), leakage-free

By default train/val is a random per-run carve of dpo_pairs.csv (see load_pairs),
which is group-disjoint but blind to structural redundancy — two near-identical
natural domains can still land on opposite sides of the carve. For a split that
also guards against that, run data/foldseek_split.py then data/build_dpo_pairs.py,
and pass --train-pairs/--val-pairs below.

In-training validation (logged every --eval-steps):
  reward_acc   fraction of val pairs with correct implicit-reward ordering
  reward_margin mean β·(Δ_chosen − Δ_rejected)
  val_loss     DPO loss on the val pairs
  kl_drift     mean(log π − log π_ref) on chosen  (sanity: policy not blowing up)

The real test — whether the aligned policy ranks *unseen de novo folds* by true
ΔG better than the base model — is a Spearman(Δ-pseudo-LL, ΔG) reported on the
de novo set (--heldout-eval), reusing the same held-out domains as fit_probe.py.

Usage (from rl_esm/):
    python align/train_dpo.py --smoke                      # tiny sanity run
    python align/train_dpo.py --epochs 1 --beta 0.1
    python align/train_dpo.py --heldout-eval --heldout-n 2000
    python align/train_dpo.py --config align/configs/base.yaml --lr 3e-4  # yaml + CLI override
    python align/sweep_dpo.py --config align/configs/sweep.yaml --n-trials 30  # optuna search

Each run writes to align/dpo_out/runs/<exp_name>/<run_id>/ (config.json, history.json,
metrics.csv, tensorboard/, best/, last/) and appends a summary row to
align/dpo_out/runs_index.csv. <exp_name> groups related runs (--exp-name, else the
--config file stem, else 'default'). See README "4. DPO alignment" for details.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREP = ROOT / "data" / "prepared"
PAIRS_CSV = PREP / "dpo_pairs.csv"
REWARD_CSV = PREP / "reward_table.csv"
OUT_DIR = ROOT / "align" / "dpo_out"
RUNS_INDEX = OUT_DIR / "runs_index.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.chosen = df.chosen.tolist()
        self.rejected = df.rejected.tolist()

    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, i):
        return self.chosen[i], self.rejected[i], i


def make_collate(tokenizer):
    def collate(batch):
        chosen, rejected, idx = zip(*batch)
        # tokenize chosen+rejected together so they share one padded tensor
        enc = tokenizer(list(chosen) + list(rejected), return_tensors="pt",
                        padding=True)
        return enc["input_ids"], enc["attention_mask"], len(chosen), torch.tensor(idx)
    return collate


def load_pairs(val_frac: float, seed: int, max_pairs: int | None,
               train_path: Path | None = None, val_path: Path | None = None):
    """Group-disjoint train/val split of the natural-domain preference pairs.

    If train_path/val_path are given (from data/build_dpo_pairs.py), load those
    directly instead of carving val_frac out of PAIRS_CSV at runtime — use this
    to train on a split that also guards against structural redundancy between
    train/val (see README "2b" / data/foldseek_split.py)."""
    if train_path is not None:
        train, val = pd.read_csv(train_path), pd.read_csv(val_path)
        if max_pairs:
            # cap both sides — val is re-scored every --eval-steps, so an uncapped
            # val (it can be much bigger than a val_frac carve of dpo_pairs.csv)
            # makes eval, not training, the bottleneck.
            train = train.sample(min(max_pairs, len(train)), random_state=seed)
            val = val.sample(min(max_pairs, len(val)), random_state=seed)
        print(f"train pairs: {len(train)} ({train.WT_name.nunique()} domains)  [{train_path.name}]")
        print(f"val   pairs: {len(val)} ({val.WT_name.nunique()} domains, disjoint)  [{val_path.name}]")
        return train.reset_index(drop=True), val.reset_index(drop=True)

    df = pd.read_csv(PAIRS_CSV)
    if max_pairs:
        df = df.sample(min(max_pairs, len(df)), random_state=seed)
    wts = df.WT_name.unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(wts)
    n_val = max(1, int(len(wts) * val_frac))
    val_wts = set(wts[:n_val])
    val = df[df.WT_name.isin(val_wts)].reset_index(drop=True)
    train = df[~df.WT_name.isin(val_wts)].reset_index(drop=True)
    print(f"train pairs: {len(train)} ({train.WT_name.nunique()} natural domains)")
    print(f"val   pairs: {len(val)} ({val.WT_name.nunique()} natural domains, disjoint)")
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# PSEUDO-LOG-LIKELIHOOD  (the MLM "log π(seq)")
# ─────────────────────────────────────────────────────────────────────────────

def seq_logp(model, input_ids, attention_mask, special_ids, length_norm=False):
    """Single-pass Σ log p(x_i | context) over residue positions. Returns (B,)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)      # (B,L)
    resid = attention_mask.bool() & ~torch.isin(input_ids, special_ids)  # (B,L)
    summed = (tok_logp * resid).sum(dim=1)
    if length_norm:
        summed = summed / resid.sum(dim=1).clamp(min=1)
    return summed


def split_pair(vec, n):
    """A batch tensor holding [chosen ; rejected] → (chosen, rejected)."""
    return vec[:n], vec[n:]


@torch.inference_mode()
def masked_seq_logp(model, tokenizer, sequences, special_id_set, batch_size,
                    length_norm):
    """Rigorous masked pseudo-LL: mask each residue once and predict it from the
    rest (L forward passes/seq → no self-leakage). Slow, so eval-only. Returns (N,)."""
    mask_id, pad_id = tokenizer.mask_token_id, tokenizer.pad_token_id
    flat, pos_of, true_of, owner = [], [], [], []
    for si, s in enumerate(sequences):
        ids = tokenizer(s)["input_ids"]
        for pos, tid in enumerate(ids):
            if tid in special_id_set:
                continue
            masked = list(ids); masked[pos] = mask_id
            flat.append(masked); pos_of.append(pos); true_of.append(tid); owner.append(si)

    totals = np.zeros(len(sequences)); counts = np.zeros(len(sequences))
    for start in range(0, len(flat), batch_size):
        chunk = flat[start:start + batch_size]
        maxL = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), maxL), pad_id, dtype=torch.long)
        amsk = torch.zeros((len(chunk), maxL), dtype=torch.long)
        for r, x in enumerate(chunk):
            ids[r, :len(x)] = torch.tensor(x); amsk[r, :len(x)] = 1
        logits = model(input_ids=ids.to(DEVICE), attention_mask=amsk.to(DEVICE)).logits
        lp = F.log_softmax(logits.float(), dim=-1)
        for r in range(len(chunk)):
            gi = start + r
            totals[owner[gi]] += lp[r, pos_of[gi], true_of[gi]].item()
            counts[owner[gi]] += 1
    return totals / np.clip(counts, 1, None) if length_norm else totals


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE LOG-PROBS  (precomputed once — the pairs are fixed)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def precompute_ref(model, loader, special_ids, length_norm):
    """Log-probs under the frozen reference (adapters disabled). Returns (chosen, rejected)."""
    model.eval()
    ref_c, ref_r = [], []
    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else _null()
    with ctx:
        for ids, amsk, n, _ in tqdm(loader, desc="  ref log-probs", leave=False):
            lp = seq_logp(model, ids.to(DEVICE), amsk.to(DEVICE), special_ids, length_norm)
            c, r = split_pair(lp, n)
            ref_c.append(c.cpu()); ref_r.append(r.cpu())
    return torch.cat(ref_c), torch.cat(ref_r)


@torch.inference_mode()
def sanity_check_ref_precompute(model, train_df, ref_tr_c, ref_tr_r, tokenizer,
                                special_ids, length_norm, n_check=5, seed=0):
    """--smoke only: spot-check a few precomputed train-ref rows against a fresh
    disable_adapter() forward, to catch any row-index misalignment regression."""
    rng = np.random.RandomState(seed)
    rows = rng.choice(len(train_df), size=min(n_check, len(train_df)), replace=False)
    seqs = train_df.chosen.iloc[rows].tolist() + train_df.rejected.iloc[rows].tolist()
    enc = tokenizer(seqs, return_tensors="pt", padding=True).to(DEVICE)
    with model.disable_adapter():
        lp = seq_logp(model, enc["input_ids"], enc["attention_mask"], special_ids, length_norm)
    fresh_c, fresh_r = split_pair(lp.cpu(), len(rows))
    assert torch.allclose(fresh_c, ref_tr_c[rows], atol=1e-3), "train ref-precompute mismatch (chosen)"
    assert torch.allclose(fresh_r, ref_tr_r[rows], atol=1e-3), "train ref-precompute mismatch (rejected)"
    print(f"  ref-precompute sanity check passed ({len(rows)} rows)")


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ─────────────────────────────────────────────────────────────────────────────
# DPO LOSS + METRICS
# ─────────────────────────────────────────────────────────────────────────────

def dpo_terms(pol_c, pol_r, ref_c, ref_r, beta):
    d_chosen = pol_c - ref_c
    d_rejected = pol_r - ref_r
    margin = beta * (d_chosen - d_rejected)
    loss = -F.logsigmoid(margin).mean()
    return loss, margin, d_chosen


@torch.inference_mode()
def evaluate(model, loader, ref_c_all, ref_r_all, special_ids, beta, length_norm):
    model.eval()
    losses, margins, drifts, correct, n = [], [], [], 0, 0
    for bi, (ids, amsk, nb, _) in enumerate(loader):
        pol = seq_logp(model, ids.to(DEVICE), amsk.to(DEVICE), special_ids, length_norm)
        pol_c, pol_r = split_pair(pol, nb)
        rc = ref_c_all[n:n + nb].to(DEVICE); rr = ref_r_all[n:n + nb].to(DEVICE)
        loss, margin, d_chosen = dpo_terms(pol_c, pol_r, rc, rr, beta)
        losses.append(loss.item()); margins.append(margin.mean().item())
        drifts.append(d_chosen.mean().item())
        correct += (margin > 0).sum().item(); n += nb
    return {
        "val_loss": float(np.mean(losses)),
        "reward_acc": correct / max(n, 1),
        "reward_margin": float(np.mean(margins)),
        "kl_drift": float(np.mean(drifts)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELD-OUT DE NOVO EVAL  (the real test: does pseudo-LL rank ΔG on unseen folds?)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def heldout_denovo_eval(model, tokenizer, special_ids, n_seqs, batch_size,
                        length_norm, seed, mask_scoring=False):
    df = pd.read_csv(REWARD_CSV, usecols=["origin", "aa_seq", "dG"])
    df = df[(df.origin == "de_novo") & df.dG.notna()]
    if n_seqs and len(df) > n_seqs:
        df = df.sample(n_seqs, random_state=seed)
    seqs, dG = df.aa_seq.tolist(), df.dG.to_numpy()
    special_set = set(special_ids.tolist())

    def score(use_policy: bool):
        ctx = _null() if use_policy else model.disable_adapter()
        with ctx:
            if mask_scoring:  # rigorous masked pseudo-LL (no self-leakage), slow
                return masked_seq_logp(model, tokenizer, seqs, special_set,
                                       batch_size, length_norm)
            out = []
            for s in range(0, len(seqs), batch_size):
                enc = tokenizer(seqs[s:s + batch_size], return_tensors="pt",
                                padding=True).to(DEVICE)
                lp = seq_logp(model, enc["input_ids"], enc["attention_mask"],
                              special_ids, length_norm)
                out.append(lp.cpu().numpy())
            return np.concatenate(out)

    model.eval()
    pol = score(True)
    ref = score(False)
    delta = pol - ref
    # at baseline policy==ref, so delta is constant and its Spearman is undefined
    delta_rho = float(spearmanr(delta, dG).statistic) if delta.std() > 0 else float("nan")
    return {
        "denovo_spearman_policy": float(spearmanr(pol, dG).statistic),
        "denovo_spearman_base": float(spearmanr(ref, dG).statistic),
        "denovo_spearman_delta_vs_dG": delta_rho,
        "denovo_n": len(seqs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_id, rank, alpha, dropout):
    from peft import LoraConfig, get_peft_model
    base = AutoModelForMaskedLM.from_pretrained(model_id, dtype="auto")
    # ESM-C uses fused nn.Parameter tensors for QKV/FFN (not nn.Linear) — those go
    # under target_parameters; out_proj is a normal Linear (see finetune_esmc_lora.py).
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["out_proj"],
        target_parameters=["layernorm_qkv.weight", "ffn.fc1_weight", "ffn.fc2_weight"],
    )
    model = get_peft_model(base, cfg).to(DEVICE)
    model.print_trainable_parameters()
    return model


def build_args(argv=None) -> argparse.Namespace:
    """Two-pass parse: --config (if given) sets YAML defaults, explicit CLI flags win."""
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("--config", type=Path, default=None,
                            help="YAML file of defaults (e.g. align/configs/base.yaml); "
                                 "keys must match the underscore dest names below — explicit "
                                 "CLI flags still override whatever it sets")
    cfg_args, remaining = cfg_parser.parse_known_args(argv)

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                parents=[cfg_parser])
    p.add_argument("--model", default="biohub/ESMC-300M")
    p.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8, help="pairs per step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="ignored if --train-pairs/--val-pairs are given")
    p.add_argument("--train-pairs", type=Path, default=None,
                   help="pre-built train pairs CSV (from data/build_dpo_pairs.py) — "
                        "bypasses the runtime val_frac carve of dpo_pairs.csv; use this for "
                        "a split that also guards against structural redundancy (README '2b')")
    p.add_argument("--val-pairs", type=Path, default=None,
                   help="pre-built val pairs CSV, must be given together with --train-pairs")
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--length-norm", action="store_true",
                   help="length-normalize pseudo-LL (helps if pairs differ in length)")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0,
                   help="must be 0 with fused-param LoRA targets (ParamWrapper)")
    p.add_argument("--max-pairs", type=int, default=0, help="cap pairs (0=all)")
    p.add_argument("--heldout-eval", action="store_true",
                   help="also run the de novo ΔG-ranking test")
    p.add_argument("--heldout-n", type=int, default=2000)
    p.add_argument("--mask-scoring", action="store_true",
                   help="use rigorous masked pseudo-LL for the de novo eval (slow, no "
                        "self-leakage); the train/val DPO loss always uses single-pass "
                        "scoring to stay consistent with training")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp-name", default=None,
                   help="experiment this run belongs to; runs group under "
                        "align/dpo_out/runs/<exp_name>/<run_id>/. Default: the --config "
                        "file stem if given, else 'default'")
    p.add_argument("--run-name", default=None,
                   help="run id tag; default auto-derived from beta/lr/batch_size/lora_rank")
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end run: 200 pairs, 6 steps, eval every 3")

    if cfg_args.config:
        assert cfg_args.config.exists(), f"missing config {cfg_args.config}"
        cfg = yaml.safe_load(cfg_args.config.read_text()) or {}
        known = {a.dest for a in p._actions}
        unknown = set(cfg) - known
        assert not unknown, f"unknown key(s) in {cfg_args.config}: {unknown} (expected one of {sorted(known)})"
        p.set_defaults(**cfg)

    args = p.parse_args(remaining)
    args.config = cfg_args.config
    # set_defaults() bypasses type=Path conversion for values that came from the YAML
    # (only argparse-supplied CLI strings go through `type=`) — coerce explicitly.
    if args.train_pairs is not None:
        args.train_pairs = Path(args.train_pairs)
    if args.val_pairs is not None:
        args.val_pairs = Path(args.val_pairs)
    return args


def make_run_id(args: argparse.Namespace) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.run_name or f"b{args.beta:g}_lr{args.lr:.0e}_bs{args.batch_size}_r{args.lora_rank}"
    return f"{ts}_{tag}"


def exp_name_of(args: argparse.Namespace) -> str:
    """Experiment folder for this run. Explicit --exp-name wins; else fall back to the
    --config file stem (so a config = an experiment); else 'default'."""
    if args.exp_name:
        return args.exp_name
    if args.config is not None:
        return Path(args.config).stem
    return "default"


def append_runs_index(row: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNS_INDEX.exists()
    with open(RUNS_INDEX, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def run(args: argparse.Namespace, trial: optuna.Trial | None = None) -> dict:
    """Runs one full train+eval job under align/dpo_out/runs/<run_id>/. If `trial` is
    given (from align/sweep_dpo.py), reports reward_acc for Optuna pruning and may
    raise optuna.TrialPruned() — callers driving a study should let that propagate."""
    if args.smoke:
        args.max_pairs, args.batch_size, args.eval_steps = 200, 4, 3
    torch.manual_seed(args.seed)

    if args.train_pairs or args.val_pairs:
        assert args.train_pairs and args.val_pairs, "--train-pairs and --val-pairs must be given together"
        assert args.train_pairs.exists(), f"missing {args.train_pairs}"
        assert args.val_pairs.exists(), f"missing {args.val_pairs}"
    else:
        assert PAIRS_CSV.exists(), f"missing {PAIRS_CSV} — run the dataset notebook first"
    train_df, val_df = load_pairs(args.val_frac, args.seed, args.max_pairs or None,
                                  args.train_pairs, args.val_pairs)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    special_ids = torch.tensor(
        [i for i in (tokenizer.cls_token_id, tokenizer.eos_token_id,
                     tokenizer.pad_token_id) if i is not None], device=DEVICE)
    collate = make_collate(tokenizer)
    train_loader = DataLoader(PairDataset(train_df), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    # unshuffled, index-aligned twin of train_loader — used once to precompute ref
    # log-probs (row i here == row i of train_df, same guarantee val already relies on)
    train_ref_loader = DataLoader(PairDataset(train_df), batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate)
    val_loader = DataLoader(PairDataset(val_df), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate)

    print(f"\nLoading {args.model} + LoRA on {DEVICE} …")
    model = build_model(args.model, args.lora_rank, args.lora_alpha, args.lora_dropout)

    # Reference log-probs, precomputed once for BOTH splits (frozen base, pairs are
    # fixed) — the training loop below only ever forwards the policy and gathers the
    # matching precomputed ref values by row index, instead of re-running the frozen
    # base model every step.
    print("\nPrecomputing val reference log-probs (frozen base) …")
    ref_va_c, ref_va_r = precompute_ref(model, val_loader, special_ids, args.length_norm)
    print("Precomputing train reference log-probs (frozen base) …")
    ref_tr_c, ref_tr_r = precompute_ref(model, train_ref_loader, special_ids, args.length_norm)
    if args.smoke:
        sanity_check_ref_precompute(model, train_df, ref_tr_c, ref_tr_r, tokenizer,
                                    special_ids, args.length_norm, seed=args.seed)

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                            lr=args.lr, weight_decay=0.01)

    run_id = make_run_id(args)
    exp = exp_name_of(args)
    run_dir = OUT_DIR / "runs" / exp / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    config_dump = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config_dump.update(exp_name=exp, run_id=run_id, git_sha=sha, started=datetime.now().isoformat())
    (run_dir / "config.json").write_text(json.dumps(config_dump, indent=2, default=str))
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    print(f"\nrun: {run_id}  →  {run_dir}")

    history, step, best_acc = [], 0, -1.0

    def run_eval():
        m = evaluate(model, val_loader, ref_va_c, ref_va_r, special_ids,
                     args.beta, args.length_norm)
        if args.heldout_eval:
            m.update(heldout_denovo_eval(model, tokenizer, special_ids,
                                         args.heldout_n, args.batch_size,
                                         args.length_norm, args.seed,
                                         args.mask_scoring))
        m["step"] = step
        history.append(m)
        for k, v in m.items():
            if k != "step" and isinstance(v, float) and not np.isnan(v):
                writer.add_scalar(k, v, step)
        msg = (f"[step {step:4d}] val_loss {m['val_loss']:.4f}  "
               f"reward_acc {m['reward_acc']:.3f}  margin {m['reward_margin']:+.3f}  "
               f"kl_drift {m['kl_drift']:+.3f}")
        if args.heldout_eval:
            msg += (f"  | de novo Spearman base {m['denovo_spearman_base']:.3f}"
                    f" → policy {m['denovo_spearman_policy']:.3f}")
        print(msg)
        if trial is not None:
            trial.report(m["reward_acc"], step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return m["reward_acc"]

    try:
        print("\n=== baseline (untrained policy = reference) ===")
        run_eval()

        print("\n=== training ===")
        for epoch in range(args.epochs):
            model.train()
            for ids, amsk, nb, idx in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
                pol = seq_logp(model, ids.to(DEVICE), amsk.to(DEVICE), special_ids,
                               args.length_norm)
                pol_c, pol_r = split_pair(pol, nb)
                # frozen-base ref for this batch's rows — gathered, not recomputed
                rc, rr = ref_tr_c[idx].to(DEVICE), ref_tr_r[idx].to(DEVICE)
                loss, _, _ = dpo_terms(pol_c, pol_r, rc, rr, args.beta)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [q for q in model.parameters() if q.requires_grad], args.grad_clip)
                opt.step()
                step += 1

                if step % args.eval_steps == 0:
                    acc = run_eval()
                    if acc > best_acc:
                        best_acc = acc
                        model.save_pretrained(run_dir / "best")
                    model.train()

        print("\n=== final ===")
        run_eval()
        model.save_pretrained(run_dir / "last")
    finally:
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)
        writer.close()
        last = history[-1] if history else {}
        append_runs_index({
            "run_id": run_id, "exp_name": exp, "git_sha": sha,
            "beta": args.beta, "lr": args.lr, "batch_size": args.batch_size,
            "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha, "epochs": args.epochs,
            "length_norm": args.length_norm,
            "train_pairs": str(args.train_pairs) if args.train_pairs else PAIRS_CSV.name,
            "val_pairs": str(args.val_pairs) if args.val_pairs else "",
            "best_reward_acc": best_acc,
            "final_val_loss": last.get("val_loss"), "final_kl_drift": last.get("kl_drift"),
            "denovo_spearman_policy": last.get("denovo_spearman_policy"),
            "denovo_spearman_delta_vs_dG": last.get("denovo_spearman_delta_vs_dG"),
            "run_name": args.run_name or "",
        })

    print(f"\nSaved adapters → {run_dir}/(best|last), history → {run_dir}/history.json")
    print(f"best val reward_acc: {best_acc:.3f}")
    return {"best_reward_acc": best_acc, **(history[-1] if history else {})}


if __name__ == "__main__":
    run(build_args())
