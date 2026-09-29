#!/bin/bash
#SBATCH --job-name=qp_score
#SBATCH --partition=ai
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/slurm-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/slurm-%j.err

# Phase 1: score the EXISTING approach-1 point sets with the rigid-motion test.
# Validates the scorer on data we already have, before the 40-clip re-run lands.
# CPU only.

set -euo pipefail

readonly REPO=/scratch/oozak23/droid_qp
readonly PY=/home/oozak23/.conda/envs/pointworld-env/bin/python

cd "${REPO}"
date; hostname

echo
echo "########## approach 1 + motion window (the 11/12 result) ##########"
"${PY}" -u scripts/05_score_points.py \
  --points-dir out/mw/query_points \
  --label approach1_motionwindow

echo
echo "########## approach 1, whole video (the ~6/12 control) ##########"
"${PY}" -u scripts/05_score_points.py \
  --points-dir out/query_points \
  --label approach1_wholevideo

date
echo "done"
