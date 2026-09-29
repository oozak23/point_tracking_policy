#!/bin/bash
#SBATCH --job-name=qp_pilot1epoch
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pilot-%j.out
#SBATCH --error=logs/pilot-%j.err
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
run_dir="$PTP_OUT/policy_pilot_${SLURM_JOB_ID}"
${PY_TRACKON} -B -u training/52_train_policy.py \
  --mode train --steps 758 --batch-size 16 --eval-every 190 --warmup 50 \
  --workers 4 --checkpoint --validation-plan $PTP_DATA/validation_plan_v1.json --out "$run_dir"
${PY_TRACKON} -B -u evaluation/57_evaluate_policy.py \
  --checkpoint "$run_dir/best.pt" --ablate --out "$run_dir/evaluation"
