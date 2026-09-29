#!/bin/bash
#SBATCH --job-name=qp_a1
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Phase 2: re-run approach 1 on the fixed 40-clip evaluation set.
#
# Changes vs the 12-clip run:
#   * longest motion range, not motion_ranges[0]. 11.4% of clips carry >1 range and
#     range[0] is the earliest burst -- usually the arm reaching, not the object moving.
#   * 40 clips over 16 categories, all inside the overlap so every one is scorable.
#
# Both variants run so the motion-window effect stays measurable.

# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail

readonly REPO="${PTP_REPO}"
readonly PY=${PY_MOLMO}

cd "${REPO}"
date; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "########## motion window (longest range) ##########"
"${PY}" -u pipeline/03_object_points.py \
  --text-free \
  --ground-in-motion-window \
  --mask-choice best_iou \
  --kmeans-k 100 \
  --video-ids-from $PTP_DATA/eval_clips.json \
  --n 40 \
  --out $PTP_DATA/a1_mw40

echo
echo "########## whole video (control) ##########"
"${PY}" -u pipeline/03_object_points.py \
  --text-free \
  --mask-choice best_iou \
  --kmeans-k 100 \
  --video-ids-from $PTP_DATA/eval_clips.json \
  --n 40 \
  --out $PTP_DATA/a1_full40

date
echo "done"
