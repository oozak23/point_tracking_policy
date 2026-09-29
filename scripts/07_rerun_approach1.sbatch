#!/bin/bash
#SBATCH --job-name=qp_a1
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/slurm-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/slurm-%j.err

# Phase 2: re-run approach 1 on the fixed 40-clip evaluation set.
#
# Changes vs the 12-clip run:
#   * longest motion range, not motion_ranges[0]. 11.4% of clips carry >1 range and
#     range[0] is the earliest burst -- usually the arm reaching, not the object moving.
#   * 40 clips over 16 categories, all inside the overlap so every one is scorable.
#
# Both variants run so the motion-window effect stays measurable.

set -euo pipefail

readonly REPO=/scratch/oozak23/droid_qp
readonly PY=/scratch/oozak23/envs/molmo-motion/bin/python

cd "${REPO}"
date; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "########## motion window (longest range) ##########"
"${PY}" -u scripts/03_object_points.py \
  --text-free \
  --ground-in-motion-window \
  --mask-choice best_iou \
  --kmeans-k 100 \
  --video-ids-from out/eval_clips.json \
  --n 40 \
  --out out/a1_mw40

echo
echo "########## whole video (control) ##########"
"${PY}" -u scripts/03_object_points.py \
  --text-free \
  --mask-choice best_iou \
  --kmeans-k 100 \
  --video-ids-from out/eval_clips.json \
  --n 40 \
  --out out/a1_full40

date
echo "done"
