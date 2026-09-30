#!/bin/bash
#SBATCH --job-name=ptp_runB
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/point_run_B-%j.out
#SBATCH --error=logs/point_run_B-%j.err
# Run B of docs/POINT_POLICY_PLAN.md: run A's recipe (5 epochs, 3,790 updates, one cosine
# schedule) plus each gripper point's tracked coordinates and visibility. No future-point task.
# B vs A: do coordinates alone help? B vs D: did the future-point task hurt D's actions?
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/${USER}-mpl
${PY_TRACKON} -B -u training/52_train_policy.py \
  --mode train --steps 3790 --batch-size 16 --eval-every 758 --warmup 50 \
  --seed 2026 --lr 3e-4 --workers 4 --checkpoint \
  --validation-plan $PTP_DATA/validation_plan_v1.json \
  --coords \
  --out "$PTP_OUT/policy_point_B_${SLURM_JOB_ID}"
date; echo done
