#!/bin/bash
#SBATCH --job-name=ptp_pilot
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/point_pilot-%j.out
#SBATCH --error=logs/point_pilot-%j.err
# 1-epoch pilot of run D (docs/POINT_POLICY_PLAN.md): 758 updates, split like the full 1 + 4 run --
# 152 updates pre-training the future-point head alone, then 606 of the full model. Validation
# every 190 updates. It checks the model WORKS, not whether it is better.
# Stop conditions (fixed in the plan): losses do not fall or become NaN, or the future-point head is
# no better than "points don't move" at any horizon. Otherwise continue to the 5-epoch run.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/${USER}-mpl
${PY_TRACKON} -B -u training/52_train_policy.py \
  --mode train --steps 758 --batch-size 16 --eval-every 190 --warmup 50 \
  --seed 2026 --lr 3e-4 --workers 4 --checkpoint \
  --validation-plan $PTP_DATA/validation_plan_v1.json \
  --coords --future-points --pretrain-steps 152 \
  --out "$PTP_OUT/policy_point_pilot_${SLURM_JOB_ID}"
date; echo done
