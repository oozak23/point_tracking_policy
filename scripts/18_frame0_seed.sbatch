#!/bin/bash
#SBATCH --job-name=qp_f0
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/slurm-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/slurm-%j.err

# Option 1: seed at FRAME 0 from MolmoMotion's bidirectional tracks, so the scene memory
# is populated from the first frame instead of 7-40% of the clip being blind.
#
# Scored by tracking forward and evaluating where the points END UP at the original
# detection frame -- a frame-0 seed cannot be scored in place (the gripper holds nothing
# at frame 0, so the rigid-motion test is undefined).
set -euo pipefail
readonly REPO=/scratch/oozak23/droid_qp
readonly PY_GPU=/scratch/oozak23/envs/molmo-motion/bin/python
readonly PY_TO=/scratch/oozak23/envs/track_on_r/bin/python
readonly PY_PW=/home/oozak23/.conda/envs/pointworld-env/bin/python
cd "${REPO}"; date; hostname

echo "########## 1. MolmoMotion points at FRAME 0 ##########"
"${PY_PW}" -u scripts/14_molmomotion_as_queries.py \
  --like out/mm_sam --frame 0 --out out/mm_queries_f0

echo; echo "########## 2. SAM + k-means at frame 0 ##########"
"${PY_GPU}" -u scripts/03_object_points.py --text-free \
  --seed-from-npz out/mm_queries_f0/query_points \
  --mask-choice best_iou --kmeans-k 100 \
  --video-ids-from out/eval_clips_natural.json --n 60 --out out/mm_sam_f0

echo; echo "########## 3. Track-On from frame 0 (no blind window) ##########"
"${PY_TO}" -u scripts/16_run_trackon.py \
  --points-dir out/mm_sam_f0/query_points --n 60 --out out/trackon_mm_sam_f0

echo; echo "########## 4. where did the points END UP at the detection frame ##########"
"${PY_PW}" -u scripts/17_tracked_to_queries.py \
  --tracks out/trackon_mm_sam_f0/tracks --ref out/mm_sam/query_points \
  --out out/tracked_f0_at_det

echo; echo "########## 5. score -- compare against mm_sam at the SAME frame ##########"
"${PY_PW}" -u scripts/05_score_points.py \
  --points-dir out/tracked_f0_at_det/query_points --label tracked_f0
date; echo done
