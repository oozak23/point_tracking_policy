#!/bin/bash
#SBATCH --job-name=qp_nat
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/slurm-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/slurm-%j.err

# 60 clips sampled by NATURAL category frequency (19 "Move object", vs 3 in the
# stratified set). The stratified set answers "which categories fail"; this one answers
# "what would we get on the real 14,083", and the two differ a lot -- 43.4% unweighted
# there vs a ~71.6% frequency-weighted estimate.
#
# Two variants on identical clips:
#   A  baseline            MolmoPoint -> SAM -> k-means
#   B  geometry-corrected  same, but re-seed SAM from the attached cluster when the mask
#                          misses what actually moves with the gripper (<30% recall)
#
# B only differs where PointWorld supplies a cluster (~45% of clips); elsewhere it is
# identical to A by construction.

set -euo pipefail

readonly REPO=/scratch/oozak23/droid_qp
readonly PY_GPU=/scratch/oozak23/envs/molmo-motion/bin/python
readonly PY_PW=/home/oozak23/.conda/envs/pointworld-env/bin/python

cd "${REPO}"
date; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "########## A: baseline (natural sample) ##########"
"${PY_GPU}" -u scripts/03_object_points.py \
  --text-free --ground-in-motion-window \
  --mask-choice best_iou --kmeans-k 100 \
  --video-ids-from out/eval_clips_natural.json --n 60 \
  --out out/nat_base

echo
echo "########## B: geometry-corrected ##########"
"${PY_GPU}" -u scripts/03_object_points.py \
  --text-free --ground-in-motion-window \
  --mask-choice best_iou --kmeans-k 100 \
  --geometry-correct --correct-below 0.3 \
  --video-ids-from out/eval_clips_natural.json --n 60 \
  --out out/nat_geo

echo
echo "########## scoring both ##########"
"${PY_PW}" -u scripts/05_score_points.py \
  --points-dir out/nat_base/query_points --label nat_base
echo
"${PY_PW}" -u scripts/05_score_points.py \
  --points-dir out/nat_geo/query_points --label nat_geo

echo
echo "########## worst cases, for inspection ##########"
"${PY_PW}" -u scripts/12_viz_attached.py \
  --points-dir out/nat_base --score out/score_nat_base.json \
  --worst 6 --best 3 --out-dir out/attached_viz_nat || true

date
echo "done"
