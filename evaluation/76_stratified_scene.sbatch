#!/bin/bash
#SBATCH --job-name=qp_stratscene
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/stratscene-%j.out
#SBATCH --error=logs/stratscene-%j.err
# Evaluation only -- no training. Three forward passes per validation window (clean,
# scene-shuffled, proprio-shuffled) on the proprioception checkpoint, then stratified.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname
${PY_TRACKON} -B -u evaluation/75_stratified_scene_shuffle.py \
  --checkpoint $PTP_OUT/policy_proprio_1658371/best.pt
date; echo done
