#!/bin/bash
#SBATCH --job-name=ptp_c_overfit
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/chunk_overfit-%j.out
#SBATCH --error=logs/chunk_overfit-%j.err
# Overfit gate for action chunking (run C, docs/POINT_POLICY_PLAN.md): memorise the same 10 fixed
# windows as every earlier gate, now predicting 16 actions per frame. Passes when the first step
# meets the original criteria (each part < 0.02, total < 0.05, gripper accuracy >= 99.5%, all
# transitions correct) AND the loss over all 16 steps does (total < 0.05, each part < 0.02).
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
${PY_TRACKON} -B -u training/52_train_policy.py \
  --mode overfit --clips 10 --batch-size 10 --steps 3000 --eval-every 50 \
  --lr 0.0003 --warmup 20 --no-checkpoint \
  --action-chunk 16 \
  --out "$PTP_OUT/policy_chunk_overfit_${SLURM_JOB_ID}"
date; echo done
