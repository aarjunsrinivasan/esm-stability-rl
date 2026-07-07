# rl_esm — RL alignment of ESM-C on protein stability

Working code for [`../rl_esm_project.md`](../rl_esm_project.md): align ESM-C to folding
stability two ways (**DPO vs GRPO**) and analyse reward hacking.

# ENV SETUP

Runs through the ESM repo's pixi env (torch/MPS, ESM-C, transformers). Prefix commands with
the manifest path and the OpenMP workaround:

```
M=/Users/arjun/Desktop/esm/pyproject.toml
KMP_DUPLICATE_LIB_OK=TRUE pixi run --manifest-path $M python <script>
```

# data download

```
cd rl_esm
# list files
pixi run --manifest-path /Users/arjun/Desktop/esm/pyproject.toml python download_tsuboyama.py --list

# the one you want for the reward/DPO project (1 GB)
... python download_tsuboyama.py --match Processed_K50_dG_datasets

# structures for the structure variant (already fetched ✓)
... python download_tsuboyama.py --match AlphaFold_model_PDBs
```

Then unzip into `data/tsuboyama/` and open the notebook:

```
cd rl_esm
unzip -o data/Processed_K50_dG_datasets.zip -d data/tsuboyama/
# the zip nests everything under Processed_K50_dG_datasets/ — flatten it
mv data/tsuboyama/Processed_K50_dG_datasets/* data/tsuboyama/
rm -rf data/tsuboyama/Processed_K50_dG_datasets data/tsuboyama/__MACOSX
```

The downloader is resumable and md5-verified (auto-discovers files from the Zenodo API).

# schema & data prep

`tsuboyama_dataset.ipynb` inspects the schema and builds the training inputs.

**Source:** `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` (the ML table with `aa_seq` + ΔG;
`Dataset1` is DNA-only, skip it).

- **776,298 rows**, **479 WT domains = 331 natural + 148 de novo**
  (split on `WT_cluster`: numeric = natural, topology code like `EEHH`/`HHH` = de novo).
- Label `dG_ML` = folding ΔG (kcal/mol), **censored** at `<-1` / `>5` → parsed to bounds
  (naive `float()` would silently drop ~100k rows).
- Sequences 31–75 aa (median 58) — cheap for ESMFold.
- **De novo domains held out** (leakage-free eval); reward + DPO pairs built from natural only.

**Outputs → `data/prepared/`:**
| file | rows | contents |
|---|---|---|
| `reward_table.csv` | 771,761 | `aa_seq → dG` (+ `ddG`, origin, WT) — reward-probe training set |
| `dpo_pairs.csv` | 66,012 | `(chosen, rejected)` seqs per natural WT, ΔG-margin ≥ 1 kcal/mol |
| `wt_split.csv` | 479 | per-WT `train_natural` / `heldout_denovo` assignment |

# next

`reward/fit_probe.py` (ridge probe on frozen ESM-C → ΔG, report held-out Spearman on de novo)
→ `align/train_dpo.py` → `align/train_grpo.py` → reward-hacking analysis. See the project spec.
