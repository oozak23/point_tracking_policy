#!/bin/bash
#SBATCH --job-name=qp_overfit10
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/overfit-%j.out
#SBATCH --error=logs/overfit-%j.err
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
${PY_TRACKON} -B -u training/52_train_policy.py \
  --mode overfit --clips 10 --batch-size 10 --steps 1000 --eval-every 50 \
  --lr 0.0003 --warmup 20 --no-checkpoint \
  --out "$PTP_OUT/policy_overfit_${SLURM_JOB_ID}"
