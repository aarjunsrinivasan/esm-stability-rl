"""
scoring.py
─────────────────────────────────────────────────────────────────────────────
Shared plumbing for the eval scripts (eval_base.py, eval_fireprot.py,
eval_pppl.py). Nothing here is new logic — it is the code those scripts had
independently grown identical copies of, lifted to one place so a fourth eval
doesn't grow a fourth copy.

The actual pseudo-LL math still lives in train_dpo.py (seq_logp /
masked_seq_logp) and is imported, never reimplemented: every eval must score
sequences through the exact path the training loop uses, or its numbers aren't
comparable to the training metrics. This module only adds the batching,
adapter-switching, and config-parsing shells around it.

DEVICE / PREP / git_sha are re-exported so an eval script needs one import line.

WHAT'S HERE
───────────
  score_single        one forward pass/seq        → (N,) numpy pseudo-LL
  score_masked        mask each residue once      → (N,) numpy pseudo-LL (no self-leakage)
  rho                 nan-safe Spearman
  adapter_ctx         LoRA on (policy) vs disable_adapter() (base), same weights
  load_scoring_model  base model (+ optional LoRA) → model, tokenizer, ids, models list
  apply_config        --config YAML → argparse defaults, with unknown-key validation
  coerce_paths        work around set_defaults() bypassing type=Path
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from train_dpo import DEVICE, PREP, git_sha, masked_seq_logp, seq_logp

__all__ = [
    "DEVICE", "PREP", "git_sha",
    "score_single", "score_masked", "rho", "adapter_ctx", "load_scoring_model",
    "apply_config", "coerce_paths", "DTYPE_MAP",
]

# str -> torch.dtype for a --dtype CLI flag; "auto" defers to the checkpoint's own
# dtype (transformers' default), which for ESMC-6B resolves to fp32 and doesn't fit
# alongside anything else on a 24GB GPU — pass bf16 explicitly for that model, same
# convention as benchmark/bench_esmc.py's DTYPE_MAP.
DTYPE_MAP = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def score_single(model, tokenizer, seqs, special_ids, batch_size, length_norm):
    """Single-pass pseudo-LL for a list of sequences → (N,) numpy."""
    out = []
    for s in tqdm(range(0, len(seqs), batch_size), desc="  single", leave=False):
        enc = tokenizer(seqs[s:s + batch_size], return_tensors="pt",
                        padding=True).to(DEVICE)
        lp = seq_logp(model, enc["input_ids"], enc["attention_mask"],
                      special_ids, length_norm)
        out.append(lp.cpu().numpy())
    return np.concatenate(out) if out else np.array([])


@torch.inference_mode()
def score_masked(model, tokenizer, seqs, special_id_set, batch_size, length_norm):
    """Rigorous masked pseudo-LL (L fwd passes/seq) → (N,) numpy."""
    return masked_seq_logp(model, tokenizer, seqs, special_id_set, batch_size,
                           length_norm)


def rho(x, y):
    """Spearman over rows where both sides are finite (ΔΔG has NaNs for some rows);
    undefined (nan) if <3 usable rows or either side is constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL / ADAPTER
# ─────────────────────────────────────────────────────────────────────────────

def adapter_ctx(model, aligned):
    """aligned=True → use the LoRA policy as-is; False → disable adapters (base)."""
    if aligned or not hasattr(model, "disable_adapter"):
        return nullcontext()
    return model.disable_adapter()


def load_scoring_model(model_id, adapter=None, dtype="auto"):
    """Load the base model, optionally wrap it in a LoRA adapter, and derive the
    token-id collections the two scoring functions need.

    dtype: key into DTYPE_MAP ("auto" | "bf16" | "fp16" | "fp32"). ESMC-6B in its
    default ("auto") dtype is fp32 and alone occupies ~23GB, leaving no room for
    activations — pass "bf16" for that model.

    Returns (model, tokenizer, special_ids, special_set, models) where:
      special_ids  device tensor  — what seq_logp wants (torch.isin)
      special_set  set of ints    — what masked_seq_logp wants (`tid in ...`)
      models       list of (name, aligned_bool) to iterate over. With an adapter this
                   is [("aligned", True), ("base", False)] so both are scored on the
                   same sequences from the same weights (base = adapters disabled),
                   making every delta apples-to-apples. Without, just [("base", False)].
    """
    print(f"Loading base {model_id} (dtype={dtype}) on {DEVICE} …")
    model = AutoModelForMaskedLM.from_pretrained(model_id, dtype=DTYPE_MAP[dtype]).to(DEVICE)
    models = [("base", False)]
    if adapter is not None:
        from peft import PeftModel
        adapter = Path(adapter)
        assert adapter.exists(), f"missing adapter dir {adapter}"
        print(f"Wrapping with LoRA adapter (aligned policy) → {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter)).to(DEVICE)
        models = [("aligned", True), ("base", False)]
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    special_ids = torch.tensor(
        [i for i in (tokenizer.cls_token_id, tokenizer.eos_token_id,
                     tokenizer.pad_token_id) if i is not None], device=DEVICE)
    return model, tokenizer, special_ids, set(special_ids.tolist()), models


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (--config YAML + CLI override, shared by every eval script)
# ─────────────────────────────────────────────────────────────────────────────

def apply_config(parser, config_path):
    """Fold a --config YAML into `parser` as defaults, so explicit CLI flags still win.

    Keys must be argparse *dest* names; an unknown key is a hard error rather than a
    silently ignored typo.
    """
    assert config_path.exists(), f"missing config {config_path}"
    cfg = yaml.safe_load(config_path.read_text()) or {}
    known = {a.dest for a in parser._actions}
    unknown = set(cfg) - known
    assert not unknown, \
        f"unknown key(s) in {config_path}: {unknown} (expected one of {sorted(known)})"
    parser.set_defaults(**cfg)


def coerce_paths(args, *names):
    """set_defaults() bypasses argparse's type=Path, so a path supplied via YAML
    arrives as a str. Coerce the named args back to Path (None passes through)."""
    for n in names:
        v = getattr(args, n, None)
        if v is not None:
            setattr(args, n, Path(v))
    return args
