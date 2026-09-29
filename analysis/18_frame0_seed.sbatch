#!/bin/bash
#SBATCH --job-name=qp_f0
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Option 1: seed at FRAME 0 from MolmoMotion's bidirectional tracks, so the scene memory
# is populated from the first frame instead of 7-40% of the clip being blind.
#
# Scored by tracking forward and evaluating where the points END UP at the original
# detection frame -- a frame-0 seed cannot be scored in place (the gripper holds nothing
# at frame 0, so the rigid-motion test is undefined).
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
readonly REPO="${PTP_REPO}"
readonly PY_GPU=${PY_MOLMO}
readonly PY_TO=${PY_TRACKON}
readonly PY_PW=${PY_POINTWORLD}
cd "${REPO}"; date; hostname

echo "########## 1. MolmoMotion points at FRAME 0 ##########"
"${PY_PW}" -u pipeline/14_molmomotion_as_queries.py \
  --like $PTP_DATA/mm_sam --frame 0 --out $PTP_DATA/mm_queries_f0

echo; echo "########## 2. SAM + k-means at frame 0 ##########"
"${PY_GPU}" -u pipeline/03_object_points.py --text-free \
  --seed-from-npz $PTP_DATA/mm_queries_f0/query_points \
  --mask-choice best_iou --kmeans-k 100 \
  --video-ids-from $PTP_DATA/eval_clips_natural.json --n 60 --out $PTP_DATA/mm_sam_f0

echo; echo "########## 3. Track-On from frame 0 (no blind window) ##########"
"${PY_TO}" -u pipeline/16_run_trackon.py \
  --points-dir $PTP_DATA/mm_sam_f0/query_points --n 60 --out $PTP_DATA/trackon_mm_sam_f0

echo; echo "########## 4. where did the points END UP at the detection frame ##########"
"${PY_PW}" -u analysis/17_tracked_to_queries.py \
  --tracks $PTP_DATA/trackon_mm_sam_f0/tracks --ref $PTP_DATA/mm_sam/query_points \
  --out $PTP_DATA/tracked_f0_at_det

echo; echo "########## 5. score -- compare against mm_sam at the SAME frame ##########"
"${PY_PW}" -u analysis/05_score_points.py \
  --points-dir $PTP_DATA/tracked_f0_at_det/query_points --label tracked_f0
date; echo done
