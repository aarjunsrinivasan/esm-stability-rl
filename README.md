# esm-stability-rl — aligning a protein language model on folding stability

Align **ESM-C** to a measurable fitness objective — protein folding stability (ΔG) — with
offline **DPO**, and validate against real experimental data (Tsuboyama 2023 mega-scale ΔG,
held-out de novo domains, held-out **FireProt ΔΔG**, ProteinGym). Held-out oracles that
appear in no reward signal (FireProt ΔΔG, ESMFold pLDDT, base-model perplexity, ProteinGym)
provide ground-truth checks.

```
  Megascale ΔG ──▶ preference pairs (A≻B) ──▶ DPO ──▶ aligned policy
  held-out checks (not in the reward):  FireProt ΔΔG · Swiss-Prot pseudo-perplexity · ESMFold pLDDT · ProteinGym
```

Two questions, two kinds of check. *Did alignment work?* → FireProt ΔΔG (§5), a fully
independent stability oracle. *Did alignment break anything else?* → Swiss-Prot
pseudo-perplexity (§6), which asks whether a model tuned on 75-residue domains is still a
general protein language model at 500 residues. The KL term in training only measures drift
on the distribution DPO trained on, so it cannot answer the second question.

---

## Contents

- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Pipeline](#pipeline)
  1. [Download Tsuboyama data](#1-download-tsuboyama-data-1-gb)
  2. [Build training inputs](#2-build-training-inputs)
  3. [Fit the reward probe](#3-fit-the-reward-probe-the-gate)
  4. [DPO alignment](#4-dpo-alignment)
  5. [Held-out FireProt ΔΔG evaluation](#5-held-out-fireprot-ddg-evaluation-the-cleanest-generalization-check)
  6. [Held-out forgetting check](#6-held-out-forgetting-check-swiss-prot-pseudo-perplexity)
- [Coding-agent conventions](#coding-agent-conventions)

---

## Repository layout

Every script is self-documenting — full rationale, provenance, and usage examples live in
each file's module docstring. This table is just a map to find the right one.

| Path | What it is |
|---|---|
| **`align/`** | DPO training, evaluation, and shared scoring code |
| [`align/train_dpo.py`](align/train_dpo.py) | Custom offline DPO loop — ESM-C + LoRA policy vs. frozen reference, pseudo-log-likelihood loss (§4) |
| [`align/sweep_dpo.py`](align/sweep_dpo.py) | Optuna hyperparameter search (β × lr × lora_rank × batch_size) reusing `train_dpo.py` in-process |
| [`align/scoring.py`](align/scoring.py) | Shared scoring/adapter-switching helpers used by every eval script — wraps `train_dpo.py`'s pseudo-LL functions, never reimplements them |
| [`align/eval_base.py`](align/eval_base.py) | Base-model-only characterization — the before/after reference row for a DPO run |
| [`align/eval_fireprot.py`](align/eval_fireprot.py) | Held-out FireProt ΔΔG generalization eval (§5) |
| [`align/eval_pppl.py`](align/eval_pppl.py) | Held-out Swiss-Prot forgetting check (§6) |
| `align/configs/*.yaml` | Per-run configs (train/sweep/eval); any CLI flag overrides the YAML |
| `align/dpo_out/` | Run artifacts — checkpoints, `history.json`/`metrics.csv`, tensorboard, eval outputs (gitignored, regenerate locally) |
| **`data/`** | Dataset download + preparation scripts |
| [`data/download.py`](data/download.py) | Generic Zenodo dataset downloader (checksum-verified, resumable) |
| [`data/prepare.py`](data/prepare.py) | Tsuboyama raw data → `reward_table.csv` (ΔG per sequence) + `dpo_pairs.csv` (preference pairs) |
| [`data/foldseek_split.py`](data/foldseek_split.py) | ESM3-paper-style structural (Foldseek) train/val/test split, incl. denovo-safe variant |
| [`data/build_dpo_pairs.py`](data/build_dpo_pairs.py) | Leakage-free DPO train/val pairs built from a WT-level split file |
| [`data/download_fireprot.py`](data/download_fireprot.py) | Pulls the homolog-free FireProt ΔΔG set from a pinned ThermoMPNN commit (§5) |
| [`data/download_swissprot.py`](data/download_swissprot.py) | Builds the length-stratified Swiss-Prot eval set from UniProt REST (§6) |
| `data/prepared/` | Generated tables consumed by `align/` and `reward/` (gitignored, regenerate locally) |
| **`reward/`** | Reward oracle |
| [`reward/fit_probe.py`](reward/fit_probe.py) | Ridge probe on frozen ESM-C embeddings → ΔG; the gate to clear before alignment |
| `reward/probe_out/` | Probe metrics/predictions (gitignored) |
| **`benchmark/`** | Hardware feasibility sweep (load + fwd/bwd pass at increasing batch size) — see [`benchmark/README.md`](benchmark/README.md) |
| **`notebooks/`** | EDA (`tsuboyama_eda.ipynb`) and DPO/eval result analysis (`dpo_results_analysis.ipynb`, `analyze_dpo.ipynb`) |
| **`tests/`** | pytest unit tests for data-prep invariants — `pixi run pytest` (no GPU/model tests; those aren't unit-testable) |
| **`docs/`** | Personal scratch notes + a pinned copy of the ESM3 paper text referenced above (gitignored, local-only — not shipped with the repo) |
| `pixi.toml` / `pixi.lock` | Self-contained environment (see Setup) |
| [`AGENTS.md`](AGENTS.md) | Conventions, invariants, and gotchas for anyone (or any agent) working in this repo |

---

## Setup

Self-contained [pixi](https://pixi.sh) environment.

```bash
curl -fsSL https://pixi.sh/install.sh | bash        # if you don't have pixi
cd rl_esm && pixi install
```

All commands below run inside the pixi env — prefix with `pixi run` (or run `pixi shell`
once, then drop the prefix).

---

## Pipeline

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

**2c. Leakage-free DPO train/val pairs (recommended before training)**

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

**4. DPO alignment**

ESM-C is masked/bidirectional, so TRL's causal-LM `DPOTrainer` doesn't apply.
[`align/train_dpo.py`](align/train_dpo.py) is a custom loop that scores sequences by
single-pass pseudo-log-likelihood; the policy is ESM-C + LoRA and the reference is the same
weights with adapters disabled. Full method, split design, and metric definitions are in the
script's docstring.

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

**5. Held-out FireProt ΔΔG evaluation (the cleanest generalization check)**

The de novo `test` split above is held out from the *reward* but is still Tsuboyama data.
FireProt is a fully independent oracle: natural proteins, aggregated across many assays
(alanine-scan-heavy), lengths 53–448 (vs Megascale's ≤75), and — critically — the
**homolog-free** version created in ThermoMPNN, filtered to drop any FireProt sequence with
>25% identity to Megascale. It appears in no part of the DPO reward, so it tests whether the
alignment *generalizes* rather than sharpens the training distribution. This is the ESM3
paper's held-out FireProt check (App. A.1.4.4, [`docs/esm3.txt`](docs/esm3.txt)).

*Data & provenance.* [`data/download_fireprot.py`](data/download_fireprot.py) pulls
`data_all/testing/fireprot_HF.csv` directly from a **pinned commit** of the
[ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN) repo (`2b04fd3`, ~1.9 MB, no auth)
and reshapes it into a `reward_table`-style eval table. We deliberately do **not** use the
raw Zenodo FireProtDB dump (zenodo 8169289) — that is ThermoMPNN's *input*, before the
homology filtering, and would silently reintroduce leakage against Megascale.

```bash
pixi run python data/download_fireprot.py          # → data/prepared/fireprot_eval.csv
pixi run python data/download_fireprot.py --list   # just inspect the raw ThermoMPNN schema
```

Produces `data/prepared/fireprot_eval.csv`: 2,560 single-point mutants across 89 proteins
(42 with ≥10 mutants), one row per mutant with `WT_name`, `mut_type`, `wt_seq`, `aa_seq`
(mutant sequence), `ddG`, `position`, `pH`. **Sign convention: lower `ddG` = more stable**
(standard ΔΔG_folding; evidenced by ThermoMPNN's own ProteinMPNN baseline negating its
likelihood to match this column) — see the script docstring.

*Run the eval.* [`align/eval_fireprot.py`](align/eval_fireprot.py) reuses `train_dpo.py`'s
exact scoring path and scores the **aligned policy and the base model in one pass** (LoRA on
vs peft `disable_adapter()` — same weights, so every per-protein delta is apples-to-apples).
Because ΔΔG is only comparable within a wildtype, the metric is strictly **per-protein
Spearman, averaged over proteins** (never one global Spearman), oriented so **positive =
the score correctly tracks stability** (i.e. Spearman(pseudo-LL, −ddG)).

```bash
# point --adapter at the run you trained (best/ or last/); base is scored automatically
pixi run python align/eval_fireprot.py --config align/configs/eval_fireprot.yaml
# or ad-hoc, e.g. a quick smoke on 3 proteins:
pixi run python align/eval_fireprot.py --adapter align/dpo_out/runs/<exp>/<run>/best \
    --max-proteins 3 --mask-n-proteins 2
```

Two scorings: **single** (one forward pass/seq — the cheap proxy DPO trains on, all mutants)
and **masked** (mask each residue once, L passes/seq — no self-leakage, the rigorous number;
run on the 10 largest proteins, 40 mutants each, by default). Writes
`align/dpo_out/fireprot_eval/<timestamp>/` (`per_protein.csv`, `summary.csv`, `summary.md`).

*Result* — base ESM-C vs the `base_full` DPO run (`β=0.1, lr=1e-4, r=8`), stability-oriented
mean per-protein Spearman:

| scoring | base | aligned | Δ | aligned wins |
|---|---|---|---|---|
| **single** (42 proteins) | +0.188 | **+0.445** | **+0.257** | 33/42 (79%) |
| **masked** (10 largest)  | +0.425 | **+0.517** | **+0.092** | 7/10 (70%) |

**The alignment generalizes: +0.092 masked Spearman on a zero-leakage oracle, base already
strong.** Not an outlier artifact (median per-protein Δ +0.240 single / +0.083 masked).

<details>
<summary>Why <strong>masked</strong> is the number to quote, and how it compares to the paper</summary>

The **masked** row is the honest number to quote: single-pass partly reflects DPO tightening
the very quantity it optimized (mild self-leakage), whereas masked pseudo-LL has none — and it
still improves +0.092 with base already strong (+0.425). For comparison, the ESM3 paper
reports ~0.5 Spearman via its embedding *probe*; our masked aligned pseudo-LL is +0.517 in the
same ballpark (different method, so not a strict apples-to-apples). The base model coming out
positive (+0.19 / +0.43, as expected) is the built-in sign check that the metric orientation is
correct.

</details>

**6. Held-out forgetting check (Swiss-Prot pseudo-perplexity)**

§5 answers *did alignment work*. This answers *did alignment break anything else*. Every
other eval here — including the KL term in training — scores 31–75 residue Tsuboyama domains
or ≤448-residue FireProt mutants, so none of them can see damage to general protein modelling
at 500 residues. `--beta` controls drift, but KL only measures it **on the distribution DPO
trained on**; a model can sit at a comfortable KL and still have forgotten how to model a
full-length enzyme.

So: masked **pseudo-perplexity** (mask each residue once, predict from the rest, average over
residues — `pppl = exp(−mean log p)`) on natural Swiss-Prot enzymes, bucketed by length, base
vs aligned. The `(0,75]` bucket is the **in-distribution control** — the length range DPO
actually trained on. The signal is the *trend across buckets*: a delta that grows with length
means aligning on short domains degraded long-protein modelling.

*Data & provenance.* [`data/download_swissprot.py`](data/download_swissprot.py) pulls
UniProtKB reviewed enzymes ≤512 residues (`reviewed:true AND ec:* AND length:[1 TO 512]`,
222,178 hits at release **2026_02**). UniProt has no commit to pin the way
[`download_fireprot.py`](data/download_fireprot.py) pins a ThermoMPNN SHA, and no way to
request an old release over REST — so the script instead **detects drift**: it warns loudly if
the live `X-UniProt-Release` header or result count has moved off the pinned constants, and
records what it actually fetched in a `swissprot_eval.meta.json` sidecar so every eval run can
state which snapshot it scored. Rows with more than one EC number, a partial EC (`3.5.-.-`),
or a non-standard residue are dropped and counted — the residue filter matters here in a way
it never did for Tsuboyama/FireProt, since `seq_logp` only treats cls/eos/pad as special and
would score a selenocysteine as an ordinary residue.

```bash
pixi run python data/download_swissprot.py          # → data/prepared/swissprot_eval.csv (~15 min)
pixi run python data/download_swissprot.py --list   # check the pins (one page, seconds)
```

Produces `data/prepared/swissprot_eval.csv`: **400 proteins, 80 per length bucket**, sampled
from the 146,369 that survive filtering (out of 222,178 raw: 12,928 multi-EC, 35,737
partial-EC, 794 non-standard-residue rows dropped), spanning 4,192 EC numbers — one row per
protein with `accession`, `ec`, `aa_seq`, `length`, `length_bucket`. Alongside it,
`swissprot_eval.meta.json` records the release and counts actually fetched.

*Cost & why.* The first run pages through `/search` at 500 rows/request (~450 requests,
~15 min) and caches **~49 MB** of gzipped TSV under `data/swissprot/`; the eval table it
distills is ~400 proteins. Two deliberate choices behind that:

- **Not the `/stream` endpoint**, which is the obvious pick and the wrong one — measured, it
  serves this query at ~4 KB/s (≈3 h) while paginated `/search` returns 500 rows *with
  sequences* in ~2 s. Identical data.
- **Download all 222k to keep 400.** UniProt has no random-sampling API, and taking the first
  N by cursor is accession-ordered — which correlates with how well-studied a protein is, and
  so with how heavily it appears in ESM-C's pretraining. Reading the full population is what
  makes the length-stratified sample unbiased. The raw TSV is kept (unlike FireProt's 2 MB
  re-fetchable CSV) so re-subsampling with a different `--seed`/`--n-per-bucket` is instant.

*Run the eval.* [`align/eval_pppl.py`](align/eval_pppl.py) reuses the same scoring path as
every other eval ([`align/scoring.py`](align/scoring.py)) and scores the aligned policy and
the base model **in one pass** (LoRA on vs `disable_adapter()`), so the per-protein deltas are
**paired** — which is what makes the bootstrap CI on the delta meaningful.

```bash
pixi run python align/eval_pppl.py --config align/configs/eval_pppl.yaml
pixi run python align/eval_pppl.py --adapter <run>/best --max-seqs 20 --batch-size 4  # smoke
```

Unlike every other eval config, `eval_pppl.yaml` has **no `length_norm` knob**: perplexity is
per-residue by definition, so it's hardcoded on. Summed log-prob across a 37→512 residue range
would just be measuring length. Cost is the design constraint — masked pppl needs `Σ Lᵢ`
forward passes and attention is O(L²), so `batch_size` defaults to 8 (vs 16 elsewhere) and the
download defaults to 80 proteins/bucket (~35 min for base+aligned). Writes
`align/dpo_out/pppl_eval/<timestamp>/` (`per_sequence.csv`, `summary.csv`, `summary.md`).

*Result* — base ESM-C vs the same `base_full` DPO run as §5 (`β=0.1, lr=1e-4, r=8`), masked
pseudo-perplexity (lower = better), Δ in log space (positive = **aligned is worse**):

| length bucket | n | base pppl | aligned pppl | Δ log-pppl [95% CI] | proteins worse |
|---|---|---|---|---|---|
| **(0, 75]** ← DPO's own range | 80 | 7.79 | 13.23 | **+0.530** [+0.401, +0.669] | 76/80 (95%) |
| (75, 150] | 80 | 2.94 | 3.44 | +0.156 [+0.113, +0.204] | 76/80 (95%) |
| (150, 250] | 80 | 3.03 | 3.46 | +0.132 [+0.105, +0.164] | 78/80 (98%) |
| (250, 350] | 80 | 3.29 | 3.93 | +0.179 [+0.135, +0.232] | 79/80 (99%) |
| (350, 512] | 80 | 3.04 | 3.60 | +0.171 [+0.133, +0.219] | 80/80 (100%) |
| **all** | 400 | **3.70** | **4.67** | **+0.234** [+0.199, +0.271] | **389/400 (97%)** |

**The alignment is not free: overall pseudo-perplexity rises 3.70 → 4.67 (×1.26), worst in the
band DPO trained on, not at long lengths — and the KL term never saw it coming.** Every
bucket's CI excludes zero; 97% of individual proteins get worse.

<details>
<summary>Full analysis: where the damage lands, why KL didn't catch it, and whether it's fatal</summary>

**The damage is worst where DPO trained, not at long lengths.** This is the opposite of the
hypothesis the eval was built to test. Drift does *not* grow with length — Spearman(length, Δ)
= **−0.16**, slightly negative — while the `(0,75]` control bucket, the band the training
domains actually live in, degrades **×1.70** (7.79 → 13.23), 3× harder than any long bucket.
Long-protein modelling is dinged (~×1.19) but not disproportionately.

**Why, and why the KL term didn't save us.** DPO optimizes the *margin* between chosen and
rejected and is indifferent to absolute likelihood — the classic consequence is that both
sides get less likely. The training run's own `kl_drift` (= `mean(log π − log π_ref)` on the
**chosen** sequences, [`train_dpo.py:45`](align/train_dpo.py#L45)) shows exactly that, and it
never once goes positive: it falls monotonically from 0 to −313 nats over the run. The
adapter scored above is `best/` — step 1000, `kl_drift = −92.3` — i.e. *already* the
mildest-drift checkpoint among the high-accuracy ones, and it still costs ×1.26 perplexity.
(The `last/` checkpoint sits at −253.9 and would be worse.) So the diagnostic *did* see the
displacement; what it can't say, being summed nats on training pairs, is whether the drift
stays in-distribution or what it costs on real proteins. This eval answers both: it doesn't,
and ×1.26.

**But it isn't catastrophic.** 4.67 is still far below 20 — the pseudo-perplexity of a model
that has learned nothing and guesses uniformly over the 20 standard residues. ESM-C is still a
protein language model, just a measurably worse one. Read together with §5, the honest summary of
this run is: **+0.092 masked FireProt Spearman, bought with ×1.26 pseudo-perplexity on natural
proteins.** Whether that trade is worth it is a `--beta` question, and this eval is the
counterweight that makes the sweep meaningful — reward_acc and FireProt alone will always
prefer more drift.

</details>

---

## Coding-agent conventions

See [AGENTS.md](AGENTS.md) for the invariants (leakage guarantees, sign conventions, pinned
data sources) and repo conventions that anyone — human or agent — should know before changing
code here.
