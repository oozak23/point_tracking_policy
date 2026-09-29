#!/bin/bash
#SBATCH --job-name=qp_mmseed
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/slurm-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/slurm-%j.err

# Seed SAM from MolmoMotion's RELEASED points instead of MolmoPoint.
# Their grounding measured better than our text-free MolmoPoint on the same 60 clips
# (50.3% vs 38.4% attached recall), and this keeps our validated SAM + k-means so N is
# a fixed 100 rather than their variable 31-94. No 4B model in the loop.
set -euo pipefail
readonly REPO=/scratch/oozak23/droid_qp
readonly PY_GPU=/scratch/oozak23/envs/molmo-motion/bin/python
readonly PY_PW=/home/oozak23/.conda/envs/pointworld-env/bin/python
cd "${REPO}"
date; hostname

echo "########## C: MM-seeded SAM + k-means ##########"
"${PY_GPU}" -u scripts/03_object_points.py --text-free \
  --seed-from-npz out/mm_queries/query_points \
  --mask-choice best_iou --kmeans-k 100 \
  --video-ids-from out/eval_clips_natural.json --n 60 --out out/mm_sam

echo; echo "########## D: MM-seeded + geometry correction ##########"
"${PY_GPU}" -u scripts/03_object_points.py --text-free \
  --seed-from-npz out/mm_queries/query_points \
  --mask-choice best_iou --kmeans-k 100 \
  --geometry-correct --correct-below 0.3 \
  --video-ids-from out/eval_clips_natural.json --n 60 --out out/mm_sam_geo

echo; echo "########## scoring ##########"
"${PY_PW}" -u scripts/05_score_points.py --points-dir out/mm_sam/query_points --label mm_sam
echo
"${PY_PW}" -u scripts/05_score_points.py --points-dir out/mm_sam_geo/query_points --label mm_sam_geo
date; echo done
