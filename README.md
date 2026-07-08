# esm-stability-rl — RL-aligning a protein language model on folding stability

Align **ESM-C** to a *measurable* fitness objective (protein folding stability, ΔG) two
ways — **offline DPO vs. online GRPO** — and run an honest head-to-head: which optimizes
better, which stays in-distribution, and which **reward-hacks**. Validated against real
experimental data (Tsuboyama 2023 mega-scale ΔG, held-out de novo domains + ProteinGym).

> The headline deliverable is **not** "reward went up" — it's the DPO-vs-GRPO comparison
> plus a reward-hacking analysis, with held-out oracles and ground-truth validation.
> Full design & rationale: [`docs/project_outline_dev.md`](docs/project_outline_dev.md).

```
  OFFLINE:  Megascale ΔG ──▶ preference pairs (A≻B) ──▶ DPO  ──▶ aligned policy
  ONLINE:   policy ──samples──▶ ridge-probe reward ──▶ GRPO (reward − β·KL) ──▶ policy
  held-out checks (NOT in either reward):  ESMFold pLDDT · base-model perplexity · ProteinGym
```

---

## Status

| Step | What | State |
|---|---|---|
| **0. Data prep** | Tsuboyama 2023 → reward table, DPO pairs, leakage-free split | ✅ done |
| **1. Reward oracle (gate)** | ridge probe on frozen ESM-C → ΔG; held-out Spearman | ✅ done — **passes** |
| **2. DPO arm** | custom pseudo-LL DPO on preference pairs (offline), LoRA | ✅ built |
| 3. GRPO arm | GRPO on the probe reward (online) | ⬜ next |
| 4. Comparison + hacking analysis | every held-out metric, KL sweep, ProteinGym | ⬜ |

### Step 1 result — the reward oracle passes the gate

The ESM3 paper (App. A.1.4.4) shows Megascale is *not* ESM pretraining data, and a linear
probe on frozen ESM-C reps already predicts ΔG at Spearman ~0.68–0.8. So the reward is a
cheap, solved component — the gate is: **held-out Spearman ≥ 0.40 before touching RL.**

Fit on **natural** domains, evaluated on **de novo** domains (topology-code clusters like
`EEHH`/`HHH` — not in ESM/ESMFold pretraining, so leakage-free):

| split | Spearman | Pearson | RMSE | n |
|---|---|---|---|---|
| train (natural) | 0.855 | 0.845 | 0.93 | 30,000 |
| **held-out de novo** | **0.518** | 0.532 | 1.98 | 40,000 |

→ **PASS.** `biohub/ESMC-300M`, penultimate layer, mean-pooled. Below the paper's 0.68–0.8
because that's the 6B model — model *size* is the lever, not the probe. Run the 600M model
(`pixi run probe-600m`) for a boost.

### Step 2 — DPO (offline arm)

ESM-C is a **masked / bidirectional** LM, so `log π(seq) = Σ log p(x_i | x_<i)` doesn't
apply and TRL's causal-LM `DPOTrainer` doesn't fit. [align/train_dpo.py](align/train_dpo.py)
is a **custom DPO loop** that defines the sequence log-prob as the single-pass
**pseudo-log-likelihood** (`Σ_i log softmax(logits)[i, x_i]` over residues, one forward
pass) — the standard differentiable MLM fitness proxy. Policy = ESM-C + **LoRA** (peft);
reference = the same base weights with adapters disabled.

**Splits — group-disjoint by `WT_name`, no domain in two splits:**

| split | source | role |
|---|---|---|
| train pairs | ~90% of the 331 **natural** domains (`dpo_pairs.csv`) | DPO optimization |
| val pairs | the other ~10% of natural domains | in-training metrics |
| **test** | the 148 **de novo** domains (`reward_table.csv`) | leakage-free eval |

**In-training validation** (logged every `--eval-steps`): `reward_acc` (fraction of val
pairs ordered correctly, → 1.0), `reward_margin` (mean `β·(Δchosen − Δrejected)`, ↑),
`val_loss`, and `kl_drift` (`mean(log π − log π_ref)`, watch it doesn't blow up).
**The real test** (`--heldout-eval`): Spearman of the policy's Δ-pseudo-LL vs *true ΔG* on
the de novo folds — does alignment improve stability ranking on domains never trained on,
vs the base model? Same held-out set as the reward probe.

**On masking.** The train *and* val DPO loss use the **single-pass** pseudo-LL — no
position is masked; each `log p(x_i | context)` is read from one unmasked forward pass. This
has a self-leakage bias (the true token is visible), but it's identical for `log π` and
`log π_ref`, so it **cancels in the DPO margin** `(log π − log π_ref)`, and keeping val
scoring identical to train is what makes `val_loss`/`reward_acc` comparable. For the *held-
out de novo* test only, `--mask-scoring` switches to the rigorous **masked** pseudo-LL (mask
each residue once, L forward passes/seq, no leakage) — accurate but slow, so pair it with a
large `--eval-steps`.

Smoke run confirms the mechanics (val_loss ↓, reward_margin ↑, reward_acc ~0.78, kl_drift
climbing). A full run is the next thing to sweep (β, LR, epochs).

---

## Setup

Self-contained [pixi](https://pixi.sh) environment — no dependency on the parent ESM repo.
On `linux-64` you get the CUDA torch wheel; on `osx-arm64` the MPS wheel.

```bash
curl -fsSL https://pixi.sh/install.sh | bash        # if you don't have pixi
cd rl_esm
pixi install                                         # resolves the env from pixi.toml
```

ESM-C weights are pulled from the 🤗 Hub on first use (`biohub/ESMC-300M`, ~1 GB) — no
manual download. Optional speed/precision kernels (`xformers`, `flash-attn`,
`transformer_engine`) are **not** required; the code falls back to torch SDPA. They make no
difference to the probe (see note at the bottom).

<details><summary>pip alternative (no pixi)</summary>

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "torch>=2.4" "transformers>=4.57" accelerate "peft>=0.15" \
            scikit-learn scipy "numpy<2" pandas tqdm matplotlib requests jupyter
```
</details>

---

## Reproduce

### 1. Download the Tsuboyama data (~1 GB)

All commands run inside the pixi env — prefix with `pixi run` (or `pixi shell` once, then
drop the prefix).

```bash
pixi run python data/download.py --dataset tsuboyama --match Processed_K50_dG_datasets   # Zenodo 7992926, resumable + md5-verified
unzip -o data/tsuboyama/Processed_K50_dG_datasets.zip -d data/tsuboyama/
mv data/tsuboyama/Processed_K50_dG_datasets/* data/tsuboyama/   # zip nests one level — flatten
rm -rf data/tsuboyama/Processed_K50_dG_datasets
find data/tsuboyama -type d -name '__MACOSX' -exec rm -rf {} + 2>/dev/null || true
```

Source of truth: `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` (776k rows, the ML table
with `aa_seq` + ΔG). `Dataset1` is DNA-only — skipped.

### 2. Build the training inputs

```bash
pixi run python data/prepare.py --dataset tsuboyama
```

Produces `data/prepared/`:

| file | rows | contents |
|---|---|---|
| `reward_table.csv` | 771,761 | `aa_seq → dG` (+ `ddG`, origin, WT) — reward-probe training set |
| `dpo_pairs.csv` | 66,012 | `(chosen, rejected)` per natural WT, ΔG-margin ≥ 1 kcal/mol |
| `wt_split.csv` | 479 | per-WT `train_natural` / `heldout_denovo` assignment |

Key data facts: 479 WT domains = **331 natural + 148 de novo** (split on `WT_cluster`:
numeric = natural, topology code = de novo). ΔG label is **censored** at `<-1` / `>5` →
parsed to bounds (naive `float()` silently drops ~100k rows). Sequences 31–75 aa (cheap for
ESMFold later). De novo domains are held out; reward + pairs built from natural only.

### 3. Fit the reward probe (the gate)

```bash
pixi run python reward/fit_probe.py                              # 20k natural train / all de novo held-out
# or tune it:
pixi run python reward/fit_probe.py --model biohub/ESMC-600M --layer -2
pixi run python reward/fit_probe.py --n-train 40000 --n-eval 20000
pixi run python reward/fit_probe.py --no-cache                   # force re-embed
```

Embeddings are mean-pooled ESM-C hidden states at the chosen layer (penultimate by
default), cached to `data/prepared/embeddings/*.npz` keyed by (model, layer, sequence set),
so re-runs are instant. Outputs land in `reward/probe_out/` (metrics + held-out predictions).

### 4. DPO alignment (offline arm)

```bash
# quick sanity check (200 pairs, ~6 steps)
pixi run python align/train_dpo.py --smoke --heldout-eval --heldout-n 300

# ── full run: all 66k pairs, 1 epoch, de novo ΔG eval every 200 steps ──
pixi run python align/train_dpo.py \
    --epochs 1 --batch-size 16 --beta 0.1 --lr 1e-4 \
    --eval-steps 200 --heldout-eval --heldout-n 3000

# same, but score the de novo eval with rigorous masked pseudo-LL (slower — keep eval-steps high)
pixi run python align/train_dpo.py \
    --epochs 1 --batch-size 16 --beta 0.1 --eval-steps 500 \
    --heldout-eval --heldout-n 1500 --mask-scoring
```

`--beta` is the main knob (KL strength: lower = more drift, higher reward-hacking risk).

Saves LoRA adapters to `align/dpo_out/{best,last}` and per-eval metrics to
`align/dpo_out/history.json`. See the "Step 2 — DPO" section above for the split design
and what each metric means.

**Analysis:** [notebooks/analyze_dpo.ipynb](notebooks/analyze_dpo.ipynb) reads `history.json` and
plots before-vs-after — training curves, the de novo test Spearman (base vs policy), the
likelihood-displacement decomposition, and a reloaded-model pseudo-LL-vs-ΔG scatter.

```bash
pixi run jupyter lab        # open notebooks/analyze_dpo.ipynb
```

---

## Repo layout

```
esm-stability-rl/
├── README.md
├── pixi.toml                     # self-contained env
├── docs/
│   └── project_outline_dev.md    # full design: DPO-vs-GRPO, reward hacking, eval strategy
├── data/
│   ├── download.py               # Zenodo downloader — python data/download.py --dataset <name>
│   └── prepare.py                # data prep → data/prepared/ — python data/prepare.py --dataset <name>
├── reward/
│   └── fit_probe.py              # ✅ ridge probe on frozen ESM-C → ΔG (the gate)
├── align/
│   ├── train_dpo.py              # ✅ custom pseudo-LL DPO on preference pairs (offline)
│                                 # ⬜ train_grpo.py (online arm)
├── notebooks/
│   ├── tsuboyama_eda.ipynb       # ✅ dataset schema exploration + EDA
│   └── analyze_dpo.ipynb         # ✅ DPO before/after + metric analysis
└── analysis/                     # ⬜ compare.py, hacking_report.py, kl_sweep.py, validate_proteingym.py
```

`data/`, `reward/probe_out/`, and `.pixi/` are git-ignored — regenerate them locally.

---

## Notes

**On the "install xformers / flash-attn" warnings from ESM-C.** These are optional fused
kernels ESM-C *prefers*; it falls back to torch's `F.scaled_dot_product_attention` (which
carries a bundled FlashAttention-2 backend on Ampere+ **in fp16/bf16**). The default load is
fp32, so no fused attention runs — but for 40–75 aa sequences embedded once and cached, the
speed difference is seconds and the numeric drift is on the residual stream the ridge probe
refits over anyway. Not worth installing. Load in bf16 (or `pip install xformers`) only if
you later scale up embedding volume.
