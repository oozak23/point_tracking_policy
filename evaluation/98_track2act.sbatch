#!/bin/bash
#SBATCH --job-name=ptp_t2a
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/track2act-%j.out
#SBATCH --error=logs/track2act-%j.err
# Track2Act stages 1-2 on a run's saved point predictions: tracks -> PnP pose plan -> actions.
# Usage: sbatch evaluation/98_track2act.sbatch <run directory>
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname
run="$1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
${PY_POINTWORLD} -B -u evaluation/97_track2act_convert.py --run "$run" --out "$PTP_OUT/track2act_$(basename "$run")"
date; echo done
