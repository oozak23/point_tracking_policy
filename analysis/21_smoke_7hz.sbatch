#!/bin/bash
#SBATCH --job-name=qp_smoke7
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Gate for the 3.6 Hz -> 7.2 Hz migration, on the 8 clips that already have robot points
# at the old rate so results are comparable.
#
# Everything writes to out/smoke7_* -- FRESH directories, never the existing out/mm_sam_f0
# or out/robot_points_grip. That is not tidiness: video_io.reconstruct returns a cached
# file whenever the destination exists, so reusing a directory would silently hand back
# 3.6 Hz videos and renumber every frame index downstream. cache_is_stale() is the second
# line of defence; separate directories are the first.
#
# Seeds are REUSED from out/mm_queries_f0 without regeneration: those query points sit at
# frame 0 with pixel coordinates in 854x480, and neither frame 0 nor the pixel space moves
# when the rate changes. That is a property of frame-0 seeding specifically -- any other
# MolmoMotion frame index would need rates.mm_frame_to_release.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
readonly REPO="${PTP_REPO}"
readonly PY_GPU=${PY_MOLMO}
readonly PY_TO=${PY_TRACKON}
readonly PY_PW=${PY_POINTWORLD}
cd "${REPO}"; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo; echo "########## 1. object points @ 7.2 Hz (reconstructs the videos) ##########"
# --seed-from-npz reuses MolmoMotion's frame-0 medoid seeds; SAM + k-means re-run because
# the video is new. Videos land in out/smoke7_obj/release_videos.
"${PY_GPU}" -u pipeline/03_object_points.py --text-free \
  --seed-from-npz $PTP_DATA/mm_queries_f0/query_points \
  --mask-choice best_iou --kmeans-k 100 \
  --video-ids-from $PTP_DATA/smoke7_clips.json --n 8 --out $PTP_DATA/smoke7_obj

echo; echo "########## 2. robot points via FK (birth-frame fix) ##########"
"${PY_PW}" -u pipeline/19_robot_points.py \
  --like $PTP_DATA/smoke7_obj --n 8 --num-points 128 --gripper-only \
  --out $PTP_DATA/smoke7_robot

echo; echo "########## 3. robot points filtered by SAM mask ##########"
"${PY_GPU}" -u pipeline/20_robot_points_sam_filter.py \
  --robot-dir $PTP_DATA/smoke7_robot \
  --videos-dir $PTP_DATA/smoke7_obj/release_videos \
  --out $PTP_DATA/smoke7_robot_sam

echo; echo "########## 4. frozen Track-On -> token stream ##########"
"${PY_TO}" -u pipeline/16_run_trackon.py \
  --points-dir $PTP_DATA/smoke7_obj/query_points \
  --videos-dir $PTP_DATA/smoke7_obj/release_videos \
  --n 8 --out $PTP_DATA/smoke7_trackon

echo; echo "########## 5. gates ##########"
"${PY_PW}" -u checks/21_verify_7hz.py \
  --obj $PTP_DATA/smoke7_obj --robot $PTP_DATA/smoke7_robot \
  --robot-sam $PTP_DATA/smoke7_robot_sam --trackon $PTP_DATA/smoke7_trackon

date; echo done
