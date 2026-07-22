# AGENTS.md

Conventions for anyone — human or agent — working in this repo. See [README.md](README.md)
for what the project does; this file is about *how* to work in it without breaking something
that isn't obvious from a diff.

## What this repo is

Aligning ESM-C to folding stability (ΔG) with offline DPO, validated against held-out
experimental oracles (FireProt ΔΔG, Swiss-Prot pseudo-perplexity) that appear in no reward
signal. The [Repository layout](README.md#repository-layout) table in the README maps every
script to its purpose.

## Environment

Everything runs inside the [pixi](https://pixi.sh) env defined by `pixi.toml`/`pixi.lock` —
`transformers` is a git dependency (a Biohub fork with ESM-C support, not stock PyPI
transformers), and `foldseek` comes from bioconda. Never assume a bare `python`/`pip` on PATH
has the right deps.

```bash
pixi run python <script>.py     # every script, always
pixi run pytest                 # tests/ (pytest.ini: testpaths = tests)
pixi shell                      # or drop the `pixi run` prefix for a whole session
```

Tests cover data-prep invariants only (`parse_dG` censoring, `sample_preference_pairs`,
Swiss-Prot filter logic) — no GPU/model tests. Don't add tests that mock away the actual
computation just to pad coverage; if a change needs GPU verification, say so and run it
manually instead.

## Conventions this codebase already follows — match them

- **Docstring-as-design-doc.** Every script's module docstring explains *why* (provenance,
  rationale for non-obvious choices, sign conventions), not just what — see any script under
  `data/` or `align/` for the pattern. New scripts should do the same; inline comments are for
  the genuinely non-obvious only.
- **Config + CLI override.** `align/configs/*.yaml` sets defaults; any CLI flag overrides the
  YAML (`apply_config` in `align/scoring.py`). Add new tunables this way rather than
  hardcoding constants.
- **Shared logic has one home.** `align/scoring.py` holds what all three eval scripts need in
  common; `data/prepare.py`'s `sample_preference_pairs` is reused by `data/build_dpo_pairs.py`
  rather than copied. Look for an existing shared home before adding a new copy — a fourth eval
  script should not grow a fourth copy of the scoring loop.
- **Every eval scores through the training path.** `align/scoring.py` imports `seq_logp` /
  `masked_seq_logp` from `train_dpo.py`; it never reimplements pseudo-LL math. If a number
  isn't computed through that shared path, it isn't comparable to the training metrics.
- **Run artifacts have one layout.** Every DPO run writes to
  `align/dpo_out/runs/<exp_name>/<run_id>/` and appends a row to `runs_index.csv`. Don't
  hand-roll a different output convention for a new run type.

## Non-negotiable invariants

These exist because the project's whole value proposition is trustworthy held-out evals — a
change that "simplifies" one of these can silently reintroduce the leakage the eval was built
to rule out.

- **Split logic is load-bearing.** Natural-vs-de-novo (`data/prepare.py`), structural
  (Foldseek) clustering (`data/foldseek_split.py`), and the denovo-safe variants each protect a
  specific guarantee. Re-read the relevant docstring before touching split code — it explains
  exactly what leaks if you change it.
- **Sign conventions differ by dataset.** Reward-table `dG`: higher = more stable. FireProt
  `ddG`: **lower = more stable** (see `data/download_fireprot.py`). Evals report
  `Spearman(pseudo-LL, −ddG)` for FireProt specifically to normalize this — check which
  convention is in play before "fixing" a sign.
- **Pinned data sources are pinned on purpose.** `data/download_fireprot.py` pins a ThermoMPNN
  git commit instead of the raw Zenodo FireProtDB dump because the raw dump reintroduces
  leakage against Megascale. `data/download_swissprot.py` can't pin a UniProt release, so it
  detects and warns on drift instead (release + row count vs. the constants in the script) —
  don't silence that warning without checking the `.meta.json` sidecar it writes.

## Gotchas

- `docs/` is gitignored (personal scratch notes + a pinned copy of paper text for reference)
  — don't assume anything there is tracked, and don't put anything there that needs to survive
  a fresh clone.
- `data/**` is gitignored except `*.py` — raw downloads and `data/prepared/*.csv` are
  regenerated locally, never committed. Same idea for `align/dpo_out/**`,
  `reward/probe_out/**`, `benchmark/results/**` (run artifacts, machine-specific).
- Result tables quoted in the README (FireProt Spearman deltas, Swiss-Prot pseudo-perplexity
  deltas) came from one specific run (`base_full`, β=0.1, lr=1e-4, r=8) — regenerating them
  with a different config will not reproduce the same numbers, by design.

## Commit style

Scoped, roughly conventional-commits (see `git log`): `feat(align): ...`, `feat(data): ...`,
`docs: ...`. Keep the scope matching the top-level directory the commit mostly touches.
