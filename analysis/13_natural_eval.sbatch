#!/bin/bash
#SBATCH --job-name=qp_nat
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

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

# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail

readonly REPO="${PTP_REPO}"
readonly PY_GPU=${PY_MOLMO}
readonly PY_PW=${PY_POINTWORLD}

cd "${REPO}"
date; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "########## A: baseline (natural sample) ##########"
"${PY_GPU}" -u pipeline/03_object_points.py \
  --text-free --ground-in-motion-window \
  --mask-choice best_iou --kmeans-k 100 \
  --video-ids-from $PTP_DATA/eval_clips_natural.json --n 60 \
  --out $PTP_DATA/nat_base

echo
echo "########## B: geometry-corrected ##########"
"${PY_GPU}" -u pipeline/03_object_points.py \
  --text-free --ground-in-motion-window \
  --mask-choice best_iou --kmeans-k 100 \
  --geometry-correct --correct-below 0.3 \
  --video-ids-from $PTP_DATA/eval_clips_natural.json --n 60 \
  --out $PTP_DATA/nat_geo

echo
echo "########## scoring both ##########"
"${PY_PW}" -u analysis/05_score_points.py \
  --points-dir $PTP_DATA/nat_base/query_points --label nat_base
echo
"${PY_PW}" -u analysis/05_score_points.py \
  --points-dir $PTP_DATA/nat_geo/query_points --label nat_geo

echo
echo "########## worst cases, for inspection ##########"
"${PY_PW}" -u analysis/12_viz_attached.py \
  --points-dir $PTP_DATA/nat_base --score $PTP_DATA/score_nat_base.json \
  --worst 6 --best 3 --out-dir $PTP_DATA/attached_viz_nat || true

date
echo "done"
