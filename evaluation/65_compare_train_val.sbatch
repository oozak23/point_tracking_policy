#!/bin/bash
#SBATCH --job-name=qp_seen_vs_val
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/fit-diagnostic-%j.out
#SBATCH --error=logs/fit-diagnostic-%j.err
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
${PY_TRACKON} -B -u evaluation/64_compare_train_val.py \
  --checkpoint $PTP_DATA/policy_pilot_1657753/best.pt \
  --plan $PTP_DATA/validation_plan_v1.json \
  --batch-size 16 --workers 4 \
  --out "$PTP_OUT/policy_fit_diagnostic_${SLURM_JOB_ID}"
