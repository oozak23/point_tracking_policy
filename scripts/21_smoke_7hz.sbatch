#!/bin/bash
#SBATCH --job-name=qp_smoke7
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/slurm-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/slurm-%j.err

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
set -euo pipefail
readonly REPO=/scratch/oozak23/droid_qp
readonly PY_GPU=/scratch/oozak23/envs/molmo-motion/bin/python
readonly PY_TO=/scratch/oozak23/envs/track_on_r/bin/python
readonly PY_PW=/home/oozak23/.conda/envs/pointworld-env/bin/python
cd "${REPO}"; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo; echo "########## 1. object points @ 7.2 Hz (reconstructs the videos) ##########"
# --seed-from-npz reuses MolmoMotion's frame-0 medoid seeds; SAM + k-means re-run because
# the video is new. Videos land in out/smoke7_obj/release_videos.
"${PY_GPU}" -u scripts/03_object_points.py --text-free \
  --seed-from-npz out/mm_queries_f0/query_points \
  --mask-choice best_iou --kmeans-k 100 \
  --video-ids-from out/smoke7_clips.json --n 8 --out out/smoke7_obj

echo; echo "########## 2. robot points via FK (birth-frame fix) ##########"
"${PY_PW}" -u scripts/19_robot_points.py \
  --like out/smoke7_obj --n 8 --num-points 128 --gripper-only \
  --out out/smoke7_robot

echo; echo "########## 3. robot points filtered by SAM mask ##########"
"${PY_GPU}" -u scripts/20_robot_points_sam_filter.py \
  --robot-dir out/smoke7_robot \
  --videos-dir out/smoke7_obj/release_videos \
  --out out/smoke7_robot_sam

echo; echo "########## 4. frozen Track-On -> token stream ##########"
"${PY_TO}" -u scripts/16_run_trackon.py \
  --points-dir out/smoke7_obj/query_points \
  --videos-dir out/smoke7_obj/release_videos \
  --n 8 --out out/smoke7_trackon

echo; echo "########## 5. gates ##########"
"${PY_PW}" -u scripts/21_verify_7hz.py \
  --obj out/smoke7_obj --robot out/smoke7_robot \
  --robot-sam out/smoke7_robot_sam --trackon out/smoke7_trackon

date; echo done
