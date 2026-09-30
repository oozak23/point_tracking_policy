#!/bin/bash
#SBATCH --job-name=ptp_step0
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/point_step0-%j.out
#SBATCH --error=logs/point_step0-%j.err
# Step 0 of docs/POINT_POLICY_PLAN.md: per-horizon loss scales (training clips only), then the
# point-prediction baselines on the validation windows. CPU only, no training.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname
${PY_POINTWORLD} -B -u pipeline/85_point_loss_scales.py
${PY_POINTWORLD} -B -u evaluation/86_point_baselines.py --out "$PTP_OUT/point_baselines_${SLURM_JOB_ID}"
date; echo done
