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
disabled; ref log-probs are **precomputed once** (the pairs are fixed) so the
training loop only forwards the policy.

Splits (all group-disjoint by WT_name — no domain appears in two splits):
  • train pairs   — ~90% of the 331 natural WT domains  (dpo_pairs.csv)
  • val   pairs   — the other ~10% of natural domains    → in-training metrics
  • TEST          — the 148 de novo domains (reward_table.csv), leakage-free

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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREP = ROOT / "data" / "prepared"
PAIRS_CSV = PREP / "dpo_pairs.csv"
REWARD_CSV = PREP / "reward_table.csv"
OUT_DIR = ROOT / "align" / "dpo_out"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        return self.chosen[i], self.rejected[i]


def make_collate(tokenizer):
    def collate(batch):
        chosen, rejected = zip(*batch)
        # tokenize chosen+rejected together so they share one padded tensor
        enc = tokenizer(list(chosen) + list(rejected), return_tensors="pt",
                        padding=True)
        return enc["input_ids"], enc["attention_mask"], len(chosen)
    return collate


def load_pairs(val_frac: float, seed: int, max_pairs: int | None):
    """Group-disjoint train/val split of the natural-domain preference pairs."""
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
        for ids, amsk, n in tqdm(loader, desc="  ref log-probs", leave=False):
            lp = seq_logp(model, ids.to(DEVICE), amsk.to(DEVICE), special_ids, length_norm)
            c, r = split_pair(lp, n)
            ref_c.append(c.cpu()); ref_r.append(r.cpu())
    return torch.cat(ref_c), torch.cat(ref_r)


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
    for bi, (ids, amsk, nb) in enumerate(loader):
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="biohub/ESMC-300M")
    p.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8, help="pairs per step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.1)
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
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end run: 200 pairs, 6 steps, eval every 3")
    args = p.parse_args()

    if args.smoke:
        args.max_pairs, args.batch_size, args.eval_steps = 200, 4, 3
    torch.manual_seed(args.seed)

    assert PAIRS_CSV.exists(), f"missing {PAIRS_CSV} — run the dataset notebook first"
    train_df, val_df = load_pairs(args.val_frac, args.seed, args.max_pairs or None)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    special_ids = torch.tensor(
        [i for i in (tokenizer.cls_token_id, tokenizer.eos_token_id,
                     tokenizer.pad_token_id) if i is not None], device=DEVICE)
    collate = make_collate(tokenizer)
    train_loader = DataLoader(PairDataset(train_df), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(PairDataset(val_df), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate)

    print(f"\nLoading {args.model} + LoRA on {DEVICE} …")
    model = build_model(args.model, args.lora_rank, args.lora_alpha, args.lora_dropout)

    # Val reference log-probs, precomputed once (val_loader is unshuffled, so the
    # order lines up with evaluate()'s running index). Train-side reference is
    # recomputed per step because the train loader is shuffled.
    print("\nPrecomputing val reference log-probs (frozen base) …")
    ref_va_c, ref_va_r = precompute_ref(model, val_loader, special_ids, args.length_norm)

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                            lr=args.lr, weight_decay=0.01)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
        msg = (f"[step {step:4d}] val_loss {m['val_loss']:.4f}  "
               f"reward_acc {m['reward_acc']:.3f}  margin {m['reward_margin']:+.3f}  "
               f"kl_drift {m['kl_drift']:+.3f}")
        if args.heldout_eval:
            msg += (f"  | de novo Spearman base {m['denovo_spearman_base']:.3f}"
                    f" → policy {m['denovo_spearman_policy']:.3f}")
        print(msg)
        return m["reward_acc"]

    print("\n=== baseline (untrained policy = reference) ===")
    run_eval()

    print("\n=== training ===")
    for epoch in range(args.epochs):
        model.train()
        for ids, amsk, nb in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            pol = seq_logp(model, ids.to(DEVICE), amsk.to(DEVICE), special_ids,
                           args.length_norm)
            pol_c, pol_r = split_pair(pol, nb)
            # ref for THIS batch: adapters off, no grad (frozen base is the reference)
            with torch.inference_mode(), model.disable_adapter():
                rp = seq_logp(model, ids.to(DEVICE), amsk.to(DEVICE), special_ids,
                              args.length_norm)
            rc, rr = split_pair(rp, nb)
            loss, _, _ = dpo_terms(pol_c, pol_r, rc.detach(), rr.detach(), args.beta)

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
                    model.save_pretrained(OUT_DIR / "best")
                model.train()

    print("\n=== final ===")
    run_eval()
    model.save_pretrained(OUT_DIR / "last")
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nSaved adapters → {OUT_DIR}/(best|last), history → {OUT_DIR}/history.json")
    print(f"best val reward_acc: {best_acc:.3f}")


if __name__ == "__main__":
    main()
