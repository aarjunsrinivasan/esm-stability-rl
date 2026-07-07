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
| 2. DPO arm | TRL DPO on preference pairs (offline) | ⬜ next |
| 3. GRPO arm | TRL GRPO on the probe reward (online) | ⬜ |
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
pip install "torch>=2.4" "transformers>=4.57" accelerate scikit-learn scipy \
            "numpy<2" pandas tqdm matplotlib requests jupyter
```
</details>

---

## Reproduce

### 1. Download the Tsuboyama data (~1 GB)

```bash
pixi run download                                    # Zenodo record 7992926, resumable + md5-verified
unzip -o data/Processed_K50_dG_datasets.zip -d data/tsuboyama/
mv data/tsuboyama/Processed_K50_dG_datasets/* data/tsuboyama/   # zip nests one level — flatten
rm -rf data/tsuboyama/Processed_K50_dG_datasets data/tsuboyama/__MACOSX
```

Source of truth: `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` (776k rows, the ML table
with `aa_seq` + ΔG). `Dataset1` is DNA-only — skipped.

### 2. Build the training inputs

```bash
pixi run lab        # open tsuboyama_dataset.ipynb, run all
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
pixi run probe                                       # 20k natural train / all de novo held-out
# or tune it:
python reward/fit_probe.py --model biohub/ESMC-600M --layer -2
python reward/fit_probe.py --n-train 40000 --n-eval 20000
python reward/fit_probe.py --no-cache                # force re-embed
```

Embeddings are mean-pooled ESM-C hidden states at the chosen layer (penultimate by
default), cached to `data/prepared/embeddings/*.npz` keyed by (model, layer, sequence set),
so re-runs are instant. Outputs land in `reward/probe_out/` (metrics + held-out predictions).

---

## Repo layout

```
esm-stability-rl/
├── README.md
├── pixi.toml                     # self-contained env
├── docs/
│   └── project_outline_dev.md    # full design: DPO-vs-GRPO, reward hacking, eval strategy
├── download_tsuboyama.py         # Zenodo downloader (resumable, md5-verified)
├── tsuboyama_dataset.ipynb       # schema + data prep → data/prepared/
├── reward/
│   └── fit_probe.py              # ✅ ridge probe on frozen ESM-C → ΔG (the gate)
├── align/                        # ⬜ train_dpo.py, train_grpo.py, policy.py
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
