# AGENTS.md

How to work in this repo without breaking something that isn't obvious from a diff.
See [README.md](README.md) for what the project does.

## Environment

Everything runs inside the [pixi](https://pixi.sh) env — `transformers` is a Biohub git fork
with ESM-C support (not stock PyPI), `foldseek` comes from bioconda. Never assume a bare
`python`/`pip` on PATH has the right deps.

```bash
pixi run python <script>.py     # every script, always
pixi run test                   # pytest (testpaths = tests)
pixi run lint                   # ruff check
pixi shell                      # or drop the `pixi run` prefix for a session
```

Tests are CPU-only and mock nothing but the `foldseek` shell-out. Don't add tests that mock
away the actual computation to pad coverage; if a change needs GPU verification, say so and
run it manually.

## Conventions — match them

- **Docstring-as-design-doc.** Module docstrings explain *why* (provenance, sign conventions,
  rationale for non-obvious choices), not just what. New scripts should do the same; inline
  comments are for the genuinely non-obvious only.
- **Config + CLI override.** `align/configs/*.yaml` sets defaults; any CLI flag wins
  (`apply_config` in `align/scoring.py`). Add tunables this way, not as constants.
- **Shared logic has one home.** `align/scoring.py` holds what the eval scripts share;
  `sample_preference_pairs` lives in `data/prepare.py` and is imported by
  `data/build_dpo_pairs.py`. A fourth eval script should not grow a fourth scoring loop.
- **Every eval scores through the training path.** `align/scoring.py` imports `seq_logp` /
  `masked_seq_logp` from `train_dpo.py` and never reimplements pseudo-LL math. A number not
  computed through that path isn't comparable to the training metrics.
- **Run artifacts have one layout.** Every DPO run writes to
  `align/dpo_out/runs/<exp_name>/<run_id>/` and appends to `runs_index.csv`.

## Non-negotiable invariants

The project's value is trustworthy held-out evals — "simplifying" one of these can silently
reintroduce the leakage the eval exists to rule out. Each is locked by a test; break one and
the named file fails.

- **Split logic is load-bearing.** Natural-vs-de-novo (`data/prepare.py`), structural
  clustering (`data/foldseek_split.py`), and the denovo-safe variants each protect a specific
  guarantee — re-read the docstring before touching them.
  → `tests/test_foldseek_split.py`, `tests/test_load_pairs.py`, `tests/test_prepare.py`
- **Sign conventions differ by dataset.** Reward-table `dG`: higher = more stable. FireProt
  `ddG`: **lower = more stable**. Evals report `Spearman(pseudo-LL, −ddG)` to normalize this —
  check which convention is in play before "fixing" a sign.
  → `tests/test_eval_metrics.py`
- **Pinned data sources are pinned on purpose.** `data/download_fireprot.py` pins a ThermoMPNN
  commit because the raw Zenodo dump reintroduces leakage against Megascale.
  `data/download_swissprot.py` can't pin a UniProt release, so it warns on drift instead —
  don't silence that without checking the `.meta.json` sidecar it writes.

## Gotchas

- `docs/` is gitignored (scratch notes + a pinned copy of the ESM3 paper text). Nothing there
  survives a fresh clone, so don't link to it from tracked files.
- `data/**` is gitignored except `*.py`; same for `align/dpo_out/**`, `reward/probe_out/**`,
  `benchmark/results/**`. Run artifacts are regenerated locally, never committed.
- README result tables come from one run (`base_full`, β=0.1, lr=1e-4, r=8) — a different
  config won't reproduce them.

## Commit style

Conventional-commits, scope = the top-level directory the commit mostly touches:
`feat(align): ...`, `feat(data): ...`, `test: ...`, `docs: ...`.
