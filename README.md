# esm-stability-rl — aligning a protein language model on folding stability

Align **ESM-C** to a measurable physical objective — folding stability (ΔG) — with offline
**DPO**, then check the result against experimental oracles that appear in no reward signal.
ESM-C is masked, not causal, so off-the-shelf DPO trainers (TRL) don't apply; the training
loop, the leakage-safe splits, and the reward probe are all built here.

```
  Megascale ΔG ──▶ preference pairs (A≻B) ──▶ DPO ──▶ aligned policy
  held-out checks (not in the reward):  FireProt ΔΔG · Swiss-Prot pseudo-perplexity
```

Two questions, two checks. *Did alignment work?* → FireProt ΔΔG. *Did it break anything else?*
→ Swiss-Prot pseudo-perplexity. Training's KL term can answer neither: it only measures drift
on the distribution DPO trained on.

---

## Results

Base ESM-C vs the `base_full` run (β=0.1, lr=1e-4, r=8) — one specific run; a different config
will not reproduce these, by design. Both evals score the aligned policy and the base model in
one pass (LoRA on vs `disable_adapter()`), so every delta is same-weights and paired.

**Generalization** — held-out FireProt ΔΔG, mean per-protein Spearman(pseudo-LL, −ddG),
oriented so positive = the score tracks stability:

| scoring | base | aligned | Δ | aligned wins |
|---|---|---|---|---|
| single (42 proteins) | +0.188 | **+0.445** | **+0.257** | 33/42 (79%) |
| **masked** (10 largest) | +0.425 | **+0.517** | **+0.092** | 7/10 (70%) |

The alignment generalizes: **+0.092 masked Spearman on a zero-leakage oracle**, with the base
already strong. Masked is the number to quote — single-pass partly reflects DPO tightening the
very quantity it optimized, masked pseudo-LL has no such self-leakage.

**Cost** — held-out Swiss-Prot masked pseudo-perplexity (lower = better), Δ in log space
(positive = aligned is worse), 95% CI from a paired bootstrap over proteins:

| length bucket | n | base | aligned | Δ log-pppl [95% CI] | worse |
|---|---|---|---|---|---|
| **(0, 75]** ← DPO's own range | 80 | 7.79 | 13.23 | **+0.530** [+0.401, +0.669] | 76/80 |
| (75, 150] | 80 | 2.94 | 3.44 | +0.156 [+0.113, +0.204] | 76/80 |
| (150, 250] | 80 | 3.03 | 3.46 | +0.132 [+0.105, +0.164] | 78/80 |
| (250, 350] | 80 | 3.29 | 3.93 | +0.179 [+0.135, +0.232] | 79/80 |
| (350, 512] | 80 | 3.04 | 3.60 | +0.171 [+0.133, +0.219] | 80/80 |
| **all** | 400 | **3.70** | **4.67** | **+0.234** [+0.199, +0.271] | **389/400** |

The alignment is not free: pseudo-perplexity rises ×1.26 overall, every bucket's CI excludes
zero. The damage is **worst in the (0,75] band DPO trained on** (×1.70), not at long lengths —
the opposite of what this eval was built to test. Training's `kl_drift` fell monotonically to
−313 nats and never flagged it; summed nats on training pairs can't say what drift costs on
real proteins.

**Bottom line: +0.092 masked FireProt Spearman, bought with ×1.26 pseudo-perplexity.**
Whether that trade is worth it is a `--beta` question — and this second eval is what makes a
sweep meaningful, since reward accuracy and FireProt alone will always prefer more drift.

---

## Setup

Self-contained [pixi](https://pixi.sh) environment — `transformers` is a Biohub fork with
ESM-C support, `foldseek` comes from bioconda.

```bash
curl -fsSL https://pixi.sh/install.sh | bash        # if you don't have pixi
cd rl_esm && pixi install
```

Every command below runs inside it: prefix with `pixi run`, or run `pixi shell` once and drop
the prefix. `pixi run test` / `pixi run lint` for the test suite and linter.

---

## Pipeline

Each script's module docstring is its design doc — provenance, sign conventions, and the
rationale for every non-obvious choice live there, not here.

**1. Data** — [`download.py`](data/download.py) · [`prepare.py`](data/prepare.py) ·
[`foldseek_split.py`](data/foldseek_split.py) · [`build_dpo_pairs.py`](data/build_dpo_pairs.py)

```bash
# Tsuboyama megascale ΔG (~1 GB, Zenodo 7992926) + AlphaFold structures
pixi run python data/download.py --dataset tsuboyama --match Processed_K50_dG_datasets
unzip -o data/tsuboyama/Processed_K50_dG_datasets.zip -d data/tsuboyama/
mv data/tsuboyama/Processed_K50_dG_datasets/* data/tsuboyama/ && rm -rf data/tsuboyama/Processed_K50_dG_datasets
pixi run python data/download.py --dataset tsuboyama --match AlphaFold_model_PDBs

pixi run python data/prepare.py --dataset tsuboyama    # → reward_table.csv + dpo_pairs.csv
pixi run python data/foldseek_split.py --stratify-pppl # → structural WT-level splits
pixi run python data/build_dpo_pairs.py                # → dpo_pairs_train/val.csv
```

`reward_table.csv` is 771,761 rows (`aa_seq → dG`) over 479 WT domains, 331 natural + 148 de
novo; `dpo_pairs.csv` is 66,012 preference pairs from the natural domains at ΔG-margin ≥ 1
kcal/mol. `build_dpo_pairs.py` re-derives train/val pairs from a WT-level split file
(267/54 domains, 53,183/10,800 pairs) so structurally redundant domains never span train/val
— `train_dpo.py`'s runtime carve is group-disjoint but blind to redundancy. Test is untouched.

`foldseek_split.py` replicates the ESM3 paper's structural clustering, writing four split
files over the same 118 clusters:

| file | train | val | test | |
|---|---|---|---|---|
| `wt_split_foldseek.csv` | 54 | 12 | 52 | paper-exact: 1 domain per redundant cluster |
| `wt_split_foldseek_full.csv` | 415 | 12 | 52 | keeps every redundant-cluster domain in train |
| `wt_split_foldseek_denovo_safe.csv` | 36 | 54 | 148 | as above, de novo domains forced to test |
| `wt_split_foldseek_full_denovo_safe.csv` | 267 | 54 | 148 | **recommended** |

⚠ **Use a denovo-safe variant** unless you specifically don't need the pretraining-leakage
guarantee. Foldseek clusters purely by structure, so the origin-agnostic files pull 93% of de
novo domains into *train* — silently breaking the reason de novo domains are held out at all
(they can't be in ESM-C's pretraining corpus).

**2. Reward probe** — [`reward/fit_probe.py`](reward/fit_probe.py)

Ridge probe on frozen ESM-C embeddings → ΔG. The gate to clear before alignment; embeddings
cache to `data/prepared/embeddings/`, so re-runs are instant.

```bash
pixi run python reward/fit_probe.py                          # 20k natural train / all de novo held out
pixi run python reward/fit_probe.py --model biohub/ESMC-600M --layer -2
```

**3. DPO alignment** — [`align/train_dpo.py`](align/train_dpo.py)

Custom loop scoring sequences by single-pass pseudo-log-likelihood; policy is ESM-C + LoRA,
reference is the same weights with adapters disabled.

```bash
pixi run python align/train_dpo.py --smoke --heldout-eval --heldout-n 300   # sanity check
pixi run python align/train_dpo.py --config align/configs/base.yaml         # recommended
```

`--beta` is the main knob (KL strength: lower = more drift, higher reward-hacking risk). Every
run writes `align/dpo_out/runs/<exp_name>/<run_id>/` — `config.json` (resolved args + git sha),
`history.json`/`metrics.csv`, `tensorboard/`, `best/` and `last/` adapters — plus a summary row
in `runs_index.csv`.

Hyperparameter search over β × lr × lora_rank × batch_size, reusing the training loop
in-process so bad trials prune early on `reward_acc`:

```bash
pixi run python align/sweep_dpo.py --config align/configs/sweep.yaml --n-trials 30
```

Resumable via `align/dpo_out/optuna_study.db`; writes the winner to
`align/configs/best_sweep_config.yaml`.

**4. Held-out evals** — the two [Results](#results) tables above

```bash
pixi run python data/download_fireprot.py      # 2,560 mutants / 89 proteins, pinned commit
pixi run python align/eval_fireprot.py --config align/configs/eval_fireprot.yaml

pixi run python data/download_swissprot.py     # 400 enzymes, 80/bucket (~15 min, caches raw TSV)
pixi run python align/eval_pppl.py --config align/configs/eval_pppl.yaml
```

**FireProt** ([`eval_fireprot.py`](align/eval_fireprot.py)) is the **homolog-free** version
from a pinned ThermoMPNN commit, filtered to drop anything >25% identical to Megascale — it
appears in no part of the reward. Scored **single** (1 forward pass/seq, all mutants) and
**masked** (L passes/seq, no self-leakage, 10 largest proteins), always as per-protein Spearman
averaged over proteins, since ΔΔG is only comparable within a wildtype.

**Swiss-Prot** ([`eval_pppl.py`](align/eval_pppl.py)) scores masked pseudo-perplexity on
natural enzymes bucketed by length. The `(0,75]` bucket is the in-distribution control; the
signal is the trend across buckets.

---

## Repository layout

| Path | What it is |
|---|---|
| [`align/train_dpo.py`](align/train_dpo.py) | Custom offline DPO loop — ESM-C + LoRA vs frozen reference, pseudo-LL loss |
| [`align/scoring.py`](align/scoring.py) | Shared scoring / adapter-switching / config plumbing for every eval |
| [`align/eval_fireprot.py`](align/eval_fireprot.py) · [`eval_pppl.py`](align/eval_pppl.py) · [`eval_base.py`](align/eval_base.py) | The two held-out evals + base-model characterization |
| [`align/sweep_dpo.py`](align/sweep_dpo.py) | Optuna hyperparameter search |
| `align/configs/*.yaml` | Per-run configs; any CLI flag overrides the YAML |
| [`data/prepare.py`](data/prepare.py) | Tsuboyama raw → `reward_table.csv` + `dpo_pairs.csv` |
| [`data/foldseek_split.py`](data/foldseek_split.py) | Structural split, incl. the denovo-safe variants |
| [`data/build_dpo_pairs.py`](data/build_dpo_pairs.py) · [`download*.py`](data/) | Split-aware pair builder; Zenodo / FireProt / Swiss-Prot downloaders |
| [`reward/fit_probe.py`](reward/fit_probe.py) | Ridge probe on frozen embeddings → ΔG (the gate) |
| [`benchmark/`](benchmark/README.md) | Hardware feasibility sweep across ESM-C sizes |
| [`notebooks/`](notebooks/) | EDA and DPO/eval result analysis |
| [`tests/`](tests/) | `pixi run test` — data-prep invariants, split leakage guarantees, DPO loss, eval sign conventions |
| [`AGENTS.md`](AGENTS.md) | Conventions and invariants for anyone (human or agent) changing code here |

Generated outputs (`data/prepared/`, `align/dpo_out/`, `reward/probe_out/`,
`benchmark/results/`) are gitignored — regenerate them locally.

---

## Conventions

See [AGENTS.md](AGENTS.md) before changing code: the leakage guarantees, the per-dataset sign
conventions, and why the pinned data sources are pinned.
