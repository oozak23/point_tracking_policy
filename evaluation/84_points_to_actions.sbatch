#!/bin/bash
#SBATCH --job-name=ptp_p2a
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/points_to_actions-%j.out
#SBATCH --error=logs/points_to_actions-%j.err
# Points -> actions, tests 1-4 on all 1,326 validation windows. CPU only: FK and PnP.
# Test 0 (the converter) is checks/82_verify_pose_fit.py and must pass first.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # parallel over clips, not inside PnP
${PY_POINTWORLD} -B checks/82_verify_pose_fit.py
${PY_POINTWORLD} -B -u evaluation/83_points_to_actions.py --out "$PTP_OUT/points_to_actions_${SLURM_JOB_ID}"
date; echo done
