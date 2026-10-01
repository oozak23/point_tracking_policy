#!/bin/bash
#SBATCH --job-name=ptp_runC
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/chunk_run_C-%j.out
#SBATCH --error=logs/chunk_run_C-%j.err
# Run C of docs/POINT_POLICY_PLAN.md: run A's recipe (5 epochs, 3,790 updates, one cosine
# schedule) but predicting the next 16 actions per frame instead of one. Does a long horizon,
# as actions, make the scene matter? First-step metrics stay comparable with A, B and D.
# The overfit gate passed at update 500 (job 1724287).
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
  --action-chunk 16 \
  --out "$PTP_OUT/policy_chunk_C_${SLURM_JOB_ID}"
date; echo done
