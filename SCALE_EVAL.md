# Megascale + FireProt stability eval across ESMC 300M / 600M / 6B

Reproduces, at all three ESMC scales, the two stability evals from the ESM3 paper
(`docs/esm3.txt` App. A.1.4.4): a ridge probe on frozen embeddings against Megascale ΔG
(structural + pseudoperplexity split), and a held-out generalization check against FireProt
ΔΔG. All runs are **frozen base model only** — no new DPO training. See
`/home/arjun/.claude/plans/docs-esm3-txt-2163-what-is-the-prancy-wave.md` for the design
rationale.

**Provenance.** Produced 2026-07-21/22 on an RTX 3090 (24GB), against the working tree at
commit `ec5e66e` **plus local uncommitted changes** (see `git diff ec5e66e` for the exact
code state — mainly `reward/fit_probe.py`, `data/foldseek_split.py`, `align/scoring.py`,
`align/eval_fireprot.py`, and the new config files below). Commit before rerunning if you
need a clean, citable git sha.

## 0. What changed to make this possible

- **`--dtype` flag** added to `align/scoring.py` (`load_scoring_model`), `align/eval_fireprot.py`,
  `data/foldseek_split.py`, and `reward/fit_probe.py`. ESMC-6B in its default dtype (`"auto"`,
  which resolves to fp32) alone occupies ~23GB and OOMs a 24GB GPU before any activations —
  discovered when the first 6B run crashed on load. `--dtype bf16` fixes it (same fix
  `benchmark/bench_esmc.py` already used, just not previously threaded into the eval scripts).
- **`reward/fit_probe.py`**: added `--config`/`--split-file` support so it can consume
  `data/foldseek_split.py`'s paper-style structural+pppl split instead of the simpler
  natural-vs-de-novo default, and — when a val set is present — selects the ridge penalty by
  **val Spearman** (matching the paper's stated method) instead of `RidgeCV`'s internal
  k-fold CV. The alpha grid was also widened from `logspace(-1, 4, 12)` to `logspace(-3, 12,
  16)` to match the paper's stated search range (10⁻³–10¹²) — the narrower grid was clipping
  the selected alpha to its upper edge on the first real run.
- New config directories: `reward/configs/` (mirrors `align/configs/`).

## 1. Paper-exact split

```bash
pixi run python data/download.py --dataset tsuboyama --match AlphaFold_model_PDBs   # already present
pixi run python data/foldseek_split.py --stratify-pppl --model biohub/ESMC-6B --dtype bf16 --batch-size 32
```

Writes `data/prepared/wt_split_foldseek.csv` (the origin-agnostic **representative** variant
— matches the paper's split, which is structural, not natural-vs-de-novo). pppl stratified
with ESMC-6B, matching the paper's choice of model for that step.

**Result:** 54 train / 13 val / 51 test domains (paper: 47/13/50 — see
`data/foldseek_split.py`'s docstring for the one known deviation: this repo doesn't replicate
the paper's unspecified train-pool subsampling algorithm, so train is slightly larger).
Expanded to rows via `reward_table.csv` (all mutants per domain): **96,187 train / 24,183 val
/ 100,535 test** rows.

## 2. Megascale ΔG ridge probe (`reward/fit_probe.py`)

```bash
pixi run python reward/fit_probe.py --config reward/configs/fit_probe_300M.yaml
pixi run python reward/fit_probe.py --config reward/configs/fit_probe_600M.yaml
pixi run python reward/fit_probe.py --config reward/configs/fit_probe_6B.yaml
```

| model | ridge α (val-selected) | train Spearman | val Spearman | **test Spearman** |
|---|---|---|---|---|
| ESMC-300M | 1e4 | 0.869 | 0.762 | **0.484** |
| ESMC-600M | 1e3 | 0.904 | 0.565 | **0.528** |
| ESMC-6B | 1e5 | 0.875 | 0.752 | **0.613** |

**Monotonic improvement with scale on held-out test** (0.484 → 0.528 → 0.613) — reproduces the
paper's qualitative finding that global-ΔG generalization improves with model size (paper:
ESMC-6B reaches 0.68 on its own, larger, 47-family split; magnitudes aren't expected to match
exactly given the smaller/differently-composed train pool here). Full per-split
metrics/predictions/config: `reward/probe_out/{metrics,heldout_preds,config}_ESMC-<size>_L-2_wt_split_foldseek.*`.

## 3. FireProt ΔΔG held-out generalization (`align/eval_fireprot.py`)

```bash
pixi run python data/download_fireprot.py   # already present
pixi run python align/eval_fireprot.py --config align/configs/eval_fireprot_300M.yaml
pixi run python align/eval_fireprot.py --config align/configs/eval_fireprot_600M.yaml
pixi run python align/eval_fireprot.py --config align/configs/eval_fireprot_6B.yaml
```

Base model only (no `adapter` key in any of the three configs), 42 proteins single-pass, 10
largest proteins also masked-scored:

| model | masked mean ρ (n=10) | masked median ρ | single mean ρ (n=42) | single median ρ |
|---|---|---|---|---|
| ESMC-300M | **0.443** | 0.444 | 0.188 | 0.275 |
| ESMC-600M | **0.440** | 0.462 | 0.165 | 0.252 |
| ESMC-6B | **0.363** | 0.338 | 0.145 | 0.226 |

**Not monotonic — scale does not help here, and 6B is the worst of the three on both
scorings.** This is a genuine result, not a run-to-run fluke of one score (single and masked
agree on the direction), and it's worth flagging as a real divergence from the paper's own
FireProt finding (paper reports its probe-based FireProt Spearman improving with model scale,
same as its Megascale result). The likely explanation is methodological, not a bug: the
paper's FireProt number comes from the *same linear stability direction* fit on Megascale
(the Section 2 probe above) applied to FireProt embeddings, whereas `eval_fireprot.py` scores
FireProt directly via raw pseudo-log-likelihood — a different, less-tuned signal that was
never fit to predict stability at all, so there's no reason to expect it to inherit
Megascale's clean scale trend. (300M's masked ρ = 0.443 here is consistent with the value
already documented in README §5, +0.425, for the same base model — small difference is the
10-largest-protein random subsample.) Confirming this would take applying the Megascale
probe's learned direction to FireProt embeddings per size, which is out of scope for this
run — flag if you want that added.

Full output per run: `align/dpo_out/fireprot_eval/<timestamp>/{summary.csv,summary.md,per_protein.csv,config.json}`:
- 300M → `20260722_070953`
- 600M → `20260722_071953`
- 6B → `20260722_074937`

## Caveats

- Split composition (54/13/51) is this repo's own reproduction, not the paper's exact
  47/13/50 — see §1.
- FireProt masked Spearman is over only 10 proteins per model (compute-capped); treat the
  6B-vs-300M/600M gap as suggestive, not statistically airtight, given that n.
- Results are pinned to `ec5e66e` **plus uncommitted local changes** — commit the working
  tree first if you need a result set reproducible by git sha alone.
