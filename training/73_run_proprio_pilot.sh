#!/bin/bash
# Launch under srun; this script submits no jobs and updates no existing run.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/${USER}-mpl
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo 'Usage: bash training/73_run_proprio_pilot.sh OUTPUT_DIRECTORY [RESUME_CHECKPOINT]' >&2
  exit 2
fi
run_dir="$1"
resume_args=()
if [ "$#" -eq 2 ]; then resume_args=(--resume "$2"); fi
${PY_TRACKON} -B -u training/52_train_policy.py \
  --mode train --proprioception --steps 758 --batch-size 16 --eval-every 190 \
  --seed 2026 --lr 3e-4 --warmup 50 --workers 4 --checkpoint \
  --validation-plan $PTP_DATA/validation_plan_v1.json --out "$run_dir" "${resume_args[@]}"
# Epoch-end fitting comparison, even if validation selected an earlier best model.
if [ ! -f "$run_dir/end_fit/summary.md" ]; then
  ${PY_TRACKON} -B -u evaluation/64_compare_train_val.py \
    --checkpoint "$run_dir/last.pt" --out "$run_dir/end_fit"
fi
${PY_TRACKON} -B -u evaluation/57_evaluate_policy.py \
  --checkpoint "$run_dir/best.pt" --ablate --out "$run_dir/evaluation"
${PY_TRACKON} -B -u evaluation/72_report_proprio.py --run "$run_dir"
