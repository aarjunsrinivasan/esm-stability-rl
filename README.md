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

Produces `data/prepared/`: `reward_table.csv` (771,761 rows, `aa_seq → dG`, all 479 WT
domains: 331 natural + 148 de novo) and `dpo_pairs.csv` (66,012 preference pairs from the
331 natural domains, ΔG-margin ≥ 1 kcal/mol). Details (censored ΔG parsing, pairing logic)
in the script.

**2b. Structural (Foldseek) split — stricter, optional**

The natural-vs-de-novo `origin` column above is a simple split; `data/foldseek_split.py`
instead replicates the ESM3 paper's method (App. A.1.4.4, [`docs/esm3.txt`](docs/esm3.txt))
— clusters domains structurally with Foldseek so no near-identical structure spans
train/eval, optionally stratified by pseudoperplexity. Rationale and exact deviations from
the paper are in the script's docstring.

```bash
pixi run python data/download.py --dataset tsuboyama --match AlphaFold_model_PDBs   # structures
pixi run python data/foldseek_split.py                      # stage 1: structural split
pixi run python data/foldseek_split.py --stratify-pppl       # + stage 2: pppl-balanced val/test
```

Writes two files, sharing the same 118 structural clusters and identical val/test (both
come from the same singleton-cluster eval pool) — they differ only in how non-singleton
(structurally redundant) clusters feed train:

| file | train | val | test | notes |
|---|---|---|---|---|
| `wt_split_foldseek.csv` | 54 | 12 | 52 | **representative**: 1 domain per non-singleton cluster, rest **excluded** (361) — paper-exact (Table S5: 47/13/50) |
| `wt_split_foldseek_full.csv` | 415 | 12 | 52 | **full**: every non-singleton-cluster domain kept in train, nothing excluded — still zero train/eval leakage, just doesn't throw away redundant data |

The 361-domain "excluded" bucket is mostly structural redundancy, not noise: a few clusters
are huge (e.g. one 50-domain cluster of homologs/re-solved structures of the same natural
fold, several 30-domain de novo design-campaign batches) — the representative variant keeps
1 domain per cluster and drops the other 49; the full variant keeps all of them in train.

**⚠️ Both files above are origin-agnostic and break the pretraining-leakage guarantee.**
Foldseek clusters purely by structure, with no notion of natural vs. de novo — so 93% of de
novo domains land in non-singleton (mostly all-de-novo) clusters and get pulled into
**train**, and `test` becomes a natural+de-novo mix. That's fine for the paper's original
purpose (avoiding near-duplicate structural memorization) but it silently breaks the *other*
guarantee this project relies on: de novo domains are held out specifically because they
can't be in ESM-C's pretraining corpus, so evaluating on them tests generalization beyond
pretraining memorization. Do **not** use `wt_split_foldseek.csv` / `_full.csv` if that
guarantee matters to you (e.g. for the reward-probe gate or DPO's held-out eval).

For that, use the **denovo-safe** variants instead — same structural dedup, but every de
novo domain is forced to `test` (fixed at the original 148), and any natural domain that
structurally clusters with a de novo domain is excluded rather than trained on (it would be
a near-duplicate of a test example):

| file | train | val | test | notes |
|---|---|---|---|---|
| `wt_split_foldseek_denovo_safe.csv` | 36 | 54 | 148 | representative: 1 domain per pure-natural non-singleton cluster |
| `wt_split_foldseek_full_denovo_safe.csv` | 267 | 54 | 148 | full: all pure-natural non-singleton-cluster domains kept |

`test` is exactly the 148 de novo domains in both (verified — same set as the `origin`
column); `train`/`val` are natural-only. 10 natural domains structurally match a de novo
domain and are excluded in both variants (training on them would leak test-set structure).

Bottom line: use a denovo-safe variant (or just the plain `origin` split from step 2) for
anything that needs the pretraining-leakage guarantee; only reach for the origin-agnostic
variants if that guarantee doesn't matter for what you're testing. `foldseek` installs
automatically via `pixi install` (bioconda channel).

**2c. Leakage-free DPO train/val pairs (recommended before step 4)**

`dpo_pairs.csv` (step 2) is built from all 331 natural domains; `align/train_dpo.py`'s
`load_pairs()` then carves a random 10% of WT domains into val *at runtime*. That's
group-disjoint but blind to structural redundancy — two near-identical natural domains can
still land on opposite sides of that random carve. `data/build_dpo_pairs.py` instead builds
train/val pairs straight from a WT-level split file, so redundant domains never span
train/val either:

```bash
pixi run python data/foldseek_split.py --stratify-pppl        # prerequisite (see 2b)
pixi run python data/build_dpo_pairs.py                        # denovo-safe, full variant (recommended)
```

Writes `data/prepared/dpo_pairs_train.csv` / `dpo_pairs_val.csv` (267 / 54 domains,
53,183 / 10,800 pairs by default). Test is untouched — `align/train_dpo.py --heldout-eval`
always scores all 148 de novo domains from `reward_table.csv` directly, no pairs file
involved. Pass `--split-file wt_split_foldseek_denovo_safe.csv` for the paper-exact
(smaller, non-redundant) train pool instead.

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

# recommended: config-driven, structurally leakage-free train/val (step 2c) — see
# align/configs/base.yaml for every default (paths, beta, lr, batch size, LoRA rank...).
# Any flag also passed on the CLI overrides the YAML.
pixi run python align/train_dpo.py --config align/configs/base.yaml

# or plain CLI flags, no config file — random val carve of all 66k natural-domain pairs
pixi run python align/train_dpo.py \
    --epochs 1 --batch-size 16 --beta 0.1 --lr 1e-4 \
    --eval-steps 200 --heldout-eval --heldout-n 3000
```

`--beta` is the main knob (KL strength: lower = more drift, higher reward-hacking risk).
Every run gets its own `align/dpo_out/runs/<exp_name>/<timestamp>_<tag>/` — `config.json`
(resolved args + git sha), `history.json`/`metrics.csv`, a `tensorboard/` dir
(`pixi run tensorboard --logdir align/dpo_out/runs`), and `best/`/`last/` LoRA adapters —
plus a summary row appended to `align/dpo_out/runs_index.csv` for comparing runs at a glance.
`<exp_name>` groups related runs: pass `--exp-name`, else it falls back to the `--config`
file stem, else `default` (a sweep uses its `--study-name`).

**Hyperparameter search** — `align/sweep_dpo.py` runs an Optuna study over
β × lr × lora_rank × batch_size (space defined in `align/configs/sweep.yaml`), reusing
`train_dpo.py`'s training loop in-process so bad trials can be pruned early from
intermediate `reward_acc`:

```bash
pixi run python align/sweep_dpo.py --config align/configs/sweep.yaml --n-trials 30
```

Persists to a local `align/dpo_out/optuna_study.db` (resumable — rerun with the same
`--study-name` to continue) and writes the winning hyperparams to
`align/configs/best_sweep_config.yaml`, ready for a full-data confirmatory run:
`train_dpo.py --config align/configs/best_sweep_config.yaml --max-pairs 0 --heldout-eval`.


---
