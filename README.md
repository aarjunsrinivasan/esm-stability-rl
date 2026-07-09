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
| 0. Data prep | Tsuboyama 2023 → reward table, DPO pairs, leakage-free split | ✅ done |
| 1. Reward oracle (gate) | ridge probe on frozen ESM-C → ΔG; held-out Spearman ≥ 0.40 | ✅ **passes** (0.518) |
| 2. DPO arm | custom pseudo-LL DPO on preference pairs (offline), LoRA | ✅ built, smoke-tested |
| 3. GRPO arm | GRPO on the probe reward (online) | ⬜ next |
| 4. Comparison + hacking analysis | every held-out metric, KL sweep, ProteinGym | ⬜ |

**Reward probe** (`biohub/ESMC-300M`, penultimate layer, mean-pooled; fit on natural
domains, evaluated on de novo — leakage-free since ESM/ESMFold never trained on them):

| split | Spearman | Pearson | RMSE | n |
|---|---|---|---|---|
| train (natural) | 0.855 | 0.845 | 0.93 | 30,000 |
| held-out de novo | **0.518** | 0.532 | 1.98 | 40,000 |

Below the ESM3 paper's reported 0.68–0.8 because that's their 6B model — size is the lever,
not the probe. Try `--model biohub/ESMC-600M` for a boost.

**DPO arm** — ESM-C is masked/bidirectional, so TRL's causal-LM `DPOTrainer` doesn't apply.
[`align/train_dpo.py`](align/train_dpo.py) is a custom loop scoring sequences by single-pass
pseudo-log-likelihood, policy = ESM-C + LoRA, reference = same weights with adapters
disabled. Full method, split design, and metric definitions are in the script's docstring.
Smoke run confirms the mechanics (val_loss ↓, reward_acc → 0.78). Full sweep (β, LR, epochs)
is next.

---

## Setup

Self-contained [pixi](https://pixi.sh) environment.

```bash
curl -fsSL https://pixi.sh/install.sh | bash        # if you don't have pixi
cd rl_esm && pixi install
```

ESM-C weights pull from the 🤗 Hub on first use (`biohub/ESMC-300M`, ~1 GB). Transformers is
pinned to a Biohub fork (`pixi.toml`) since ESM-C support isn't in stock PyPI transformers
yet. Optional fused kernels (`xformers`, `flash-attn`, `transformer_engine`) are **not
required** — the code falls back to torch SDPA, and for this sequence length/volume the
speed difference is seconds. Only bother if you scale up embedding volume a lot.

<details><summary>pip alternative (no pixi)</summary>

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "torch>=2.6" "transformers @ git+https://github.com/Biohub/transformers.git@main" \
            accelerate "peft>=0.19" scikit-learn scipy "numpy<2" pandas tqdm matplotlib requests jupyter
```
</details>

---

## Reproduce

All commands run inside the pixi env — prefix with `pixi run` (or `pixi shell` once, then
drop the prefix).

**1. Download Tsuboyama data (~1 GB)**

```bash
pixi run python data/download.py --dataset tsuboyama --match Processed_K50_dG_datasets   # Zenodo 7992926
unzip -o data/tsuboyama/Processed_K50_dG_datasets.zip -d data/tsuboyama/
mv data/tsuboyama/Processed_K50_dG_datasets/* data/tsuboyama/ && rm -rf data/tsuboyama/Processed_K50_dG_datasets
```

**2. Build training inputs**

```bash
pixi run python data/prepare.py --dataset tsuboyama
```

Produces `data/prepared/`: `reward_table.csv` (771,761 rows, `aa_seq → dG`), `dpo_pairs.csv`
(66,012 preference pairs, ΔG-margin ≥ 1 kcal/mol), `wt_split.csv` (479 WT domains: 331
natural + 148 de novo, held out). Details (censored ΔG parsing, split logic) in the script.

**2b. Structural (Foldseek) split — stricter, optional**

`wt_split.csv` splits on natural-vs-de-novo origin; `data/foldseek_split.py` instead
replicates the ESM3 paper's method (App. A.1.4.4, [`docs/esm3.txt`](docs/esm3.txt)) —
clusters domains structurally with Foldseek so no near-identical structure spans
train/eval, optionally stratified by pseudoperplexity. Rationale and exact deviations from
the paper are in the script's docstring.

```bash
pixi run python data/download.py --dataset tsuboyama --match AlphaFold_model_PDBs   # structures
pixi run python data/foldseek_split.py                      # stage 1: structural split
pixi run python data/foldseek_split.py --stratify-pppl       # + stage 2: pppl-balanced val/test
```

Output: `data/prepared/wt_split_foldseek.csv` — 479 domains → 118 structural clusters → 54
train / 12 val / 52 test / 361 excluded (same order of magnitude as the paper's 47/13/50).
Swap this in wherever leakage-free eval matters, at the cost of a much smaller train pool.
`foldseek` installs automatically via `pixi install` (bioconda channel).

**3. Fit the reward probe (the gate)**

```bash
pixi run python reward/fit_probe.py                              # 20k natural train / all de novo held-out
pixi run python reward/fit_probe.py --model biohub/ESMC-600M --layer -2
pixi run python reward/fit_probe.py --no-cache                    # force re-embed
```

Embeddings cache to `data/prepared/embeddings/*.npz`, so re-runs are instant. Outputs land
in `reward/probe_out/`.

**4. DPO alignment (offline arm)**

```bash
pixi run python align/train_dpo.py --smoke --heldout-eval --heldout-n 300   # sanity check

pixi run python align/train_dpo.py \
    --epochs 1 --batch-size 16 --beta 0.1 --lr 1e-4 \
    --eval-steps 200 --heldout-eval --heldout-n 3000                        # full run, 66k pairs
```

`--beta` is the main knob (KL strength: lower = more drift, higher reward-hacking risk).
Saves LoRA adapters to `align/dpo_out/{best,last}` and metrics to
`align/dpo_out/history.json`.

**Analysis:** [`notebooks/analyze_dpo.ipynb`](notebooks/analyze_dpo.ipynb) reads
`history.json` and plots training curves, de novo Spearman (base vs. policy), and the
likelihood-displacement decomposition.

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
│   ├── download.py               # Zenodo downloader
│   ├── prepare.py                # → data/prepared/{reward_table,dpo_pairs,wt_split}.csv
│   └── foldseek_split.py         # ESM3-paper-style structural split
├── reward/
│   └── fit_probe.py              # ✅ ridge probe on frozen ESM-C → ΔG (the gate)
├── align/
│   ├── train_dpo.py              # ✅ custom pseudo-LL DPO on preference pairs (offline)
│                                  # ⬜ train_grpo.py (online arm)
├── notebooks/
│   ├── tsuboyama_eda.ipynb       # dataset schema exploration + EDA
│   └── analyze_dpo.ipynb         # DPO before/after + metric analysis
└── analysis/                     # ⬜ compare.py, hacking_report.py, kl_sweep.py, validate_proteingym.py
```

`data/`, `reward/probe_out/`, and `.pixi/` are git-ignored — regenerate them locally.
