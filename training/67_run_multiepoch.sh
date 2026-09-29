#!/bin/bash
# Run under srun or an existing GPU compute-node shell. This script submits no jobs.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/${USER}-mpl
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo 'Usage: bash training/67_run_multiepoch.sh OUTPUT_DIRECTORY [RESUME_CHECKPOINT]' >&2
  exit 2
fi
run_dir="$1"
resume_args=()
if [ "$#" -eq 2 ]; then
  resume_args=(--resume "$2")
fi
${PY_TRACKON} -B -u training/52_train_policy.py \
  --mode train --steps 3790 --batch-size 16 --eval-every 758 --warmup 50 \
  --seed 2026 --lr 3e-4 --workers 4 --checkpoint \
  --validation-plan $PTP_DATA/validation_plan_v1.json \
  --train-eval-plan $PTP_DATA/training_eval_plan_v1.json \
  --out "$run_dir" "${resume_args[@]}"
# Write the learning comparison before the separate ablation, so it survives an
# interrupted post-training evaluation.
${PY_TRACKON} -B -u evaluation/68_report_multiepoch.py --run "$run_dir"
${PY_TRACKON} -B -u evaluation/57_evaluate_policy.py \
  --checkpoint "$run_dir/best.pt" --ablate --out "$run_dir/evaluation"
${PY_TRACKON} -B -u evaluation/68_report_multiepoch.py --run "$run_dir"
