#!/bin/bash
#SBATCH --job-name=ptp_overfit
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/point_overfit-%j.out
#SBATCH --error=logs/point_overfit-%j.err
# Overfit gate for the point-prediction policy (docs/POINT_POLICY_PLAN.md): memorise the same 10
# fixed windows as the original gate (53_overfit_policy.sbatch), now with current coordinates, the
# future-point head and the plan. Passes when every action component < 0.02, total < 0.05, gripper
# accuracy >= 99.5%, all transitions correct, AND the future-point loss < 0.02. The first 30 updates
# pre-train part 1 alone, exercising the two-stage switch the real runs use.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
${PY_TRACKON} -B -u training/52_train_policy.py \
  --mode overfit --clips 10 --batch-size 10 --steps 1000 --eval-every 50 \
  --lr 0.0003 --warmup 20 --no-checkpoint \
  --coords --future-points --pretrain-steps 30 \
  --out "$PTP_OUT/policy_point_overfit_${SLURM_JOB_ID}"
date; echo done
