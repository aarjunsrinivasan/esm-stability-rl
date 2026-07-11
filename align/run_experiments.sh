#!/usr/bin/env bash
# Run the full DPO base experiment, then an Optuna sweep — each under its own
# experiment folder (align/dpo_out/runs/<exp_name>/...).
#
#   bash align/run_experiments.sh                     # default names + 30 trials
#   BASE_EXP=base_v2 SWEEP_EXP=beta_scan N_TRIALS=50 bash align/run_experiments.sh
#
# Runs from the repo root regardless of where it's invoked. Stops on first error.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# reduce allocator fragmentation on the 24GB card (harmless elsewhere)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── knobs (override via env) ────────────────────────────────────────────────
STAMP="$(date +%Y%m%d)"
BASE_EXP="${BASE_EXP:-base_full_${STAMP}}"      # base experiment folder / label
SWEEP_EXP="${SWEEP_EXP:-sweep_${STAMP}}"        # sweep study name == exp folder
N_TRIALS="${N_TRIALS:-30}"
BASE_CFG="${BASE_CFG:-align/configs/base.yaml}"
SWEEP_CFG="${SWEEP_CFG:-align/configs/sweep.yaml}"

LOG_DIR="align/dpo_out/logs"
mkdir -p "$LOG_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo " base experiment : $BASE_EXP   (config: $BASE_CFG)"
echo " sweep experiment: $SWEEP_EXP  (config: $SWEEP_CFG, trials: $N_TRIALS)"
echo " logs            : $LOG_DIR/"
echo "════════════════════════════════════════════════════════════════════"

# ── 1. full base run (all pairs, held-out de novo eval; both from base.yaml) ──
echo -e "\n[1/2] full base experiment → runs/$BASE_EXP/ …"
pixi run python align/train_dpo.py \
  --config "$BASE_CFG" \
  --exp-name "$BASE_EXP" \
  2>&1 | tee "$LOG_DIR/${BASE_EXP}.log"

# ── 2. Optuna sweep (trials grouped under runs/$SWEEP_EXP/) ───────────────────
echo -e "\n[2/2] sweep → runs/$SWEEP_EXP/ …"
pixi run python align/sweep_dpo.py \
  --config "$SWEEP_CFG" \
  --study-name "$SWEEP_EXP" \
  --n-trials "$N_TRIALS" \
  2>&1 | tee "$LOG_DIR/${SWEEP_EXP}.log"

echo -e "\n════════════════════════════════════════════════════════════════════"
echo " done."
echo "   base runs : align/dpo_out/runs/$BASE_EXP/"
echo "   sweep runs: align/dpo_out/runs/$SWEEP_EXP/"
echo "   winner cfg: align/configs/best_sweep_config.yaml"
echo "   compare   : align/dpo_out/runs_index.csv"
echo "   next      : confirm the sweep winner on full data —"
echo "     pixi run python align/train_dpo.py --config align/configs/best_sweep_config.yaml \\"
echo "       --max-pairs 0 --heldout-eval --exp-name ${SWEEP_EXP}_winner"
echo "════════════════════════════════════════════════════════════════════"
