#!/bin/bash
#SBATCH --job-name=qp_score
#SBATCH --partition=ai
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Phase 1: score the EXISTING approach-1 point sets with the rigid-motion test.
# Validates the scorer on data we already have, before the 40-clip re-run lands.
# CPU only.

# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail

readonly REPO="${PTP_REPO}"
readonly PY=${PY_POINTWORLD}

cd "${REPO}"
date; hostname

echo
echo "########## approach 1 + motion window (the 11/12 result) ##########"
"${PY}" -u analysis/05_score_points.py \
  --points-dir $PTP_DATA/mw/query_points \
  --label approach1_motionwindow

echo
echo "########## approach 1, whole video (the ~6/12 control) ##########"
"${PY}" -u analysis/05_score_points.py \
  --points-dir $PTP_DATA/query_points \
  --label approach1_wholevideo

date
echo "done"
