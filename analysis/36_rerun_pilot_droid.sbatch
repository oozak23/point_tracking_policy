#!/bin/bash
#SBATCH --job-name=qp_redo
#SBATCH --partition=avg
# `avg` is AllowAccounts=avg, so the account must be named explicitly -- without it the
# job sits in PENDING (PartitionConfig) forever even with idle nodes.
#SBATCH --account=avg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/redo-%j.out
#SBATCH --error=logs/redo-%j.err

# Re-run the pilot's SAM filter -> merge -> Track-On on the CORRECTED robot points, then
# measure tracker drift against ground truth that is actually trustworthy.
#
# Why the whole chain and not a patch: the existing trackon/ tracks came from the OLD
# query tensor (~98 robot points born at frame 5-8). The new set is 128 points born at
# frame 0. Row j refers to a different physical spot in each, so drift computed across
# them would be comparing unrelated points.
#
# Two bugs are fixed in the input to this run:
#   * zero-fill -- PointWorld's uncovered steps became joints=0, which FK reads as the arm
#     standing upright rather than as missing. 1.085% of those passed the visibility test
#     and entered the ground truth as fake positions.
#   * gripper scale -- DROID stores a NORMALISED 0-1 width; the URDF wants the finger
#     joint ANGLE. The factor is exactly 0.725 (Robotiq 2F-85 limit, 41.5 deg), measured
#     identical in every episode. Passing the raw value opened the fingers 1.38x too far,
#     moving every query point since all of them sit on the gripper.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
readonly REPO="${PTP_REPO}"
readonly PY_GPU=${PY_MOLMO}
readonly PY_TO=${PY_TRACKON}
readonly PY_PW=${PY_POINTWORLD}
cd "${REPO}"; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for S in 0000 0001; do
  D="$PTP_DATA/gen/${S}"
  echo; echo "########## shard ${S}: 1/3 SAM filter on the corrected FK points ##########"
  "${PY_GPU}" -u pipeline/20_robot_points_sam_filter.py \
    --robot-dir "${D}/robot_droid" --videos-dir "${D}/obj/release_videos" \
    --out "${D}/robot_droid_sam"

  echo; echo "########## shard ${S}: 2/3 merge object + robot ##########"
  "${PY_PW}" -u pipeline/28_merge_queries.py \
    --obj "${D}/obj" --robot "${D}/robot_droid_sam" --out "${D}/merged_droid"

  echo; echo "########## shard ${S}: 3/3 frozen Track-On ##########"
  "${PY_TO}" -u pipeline/16_run_trackon.py \
    --points-dir "${D}/merged_droid/query_points" \
    --videos-dir "${D}/obj/release_videos" \
    --n 0 --out "${D}/trackon_droid"
done

echo; echo "########## drift vs CORRECTED FK ground truth ##########"
"${PY_GPU}" -u pipeline/34_gt_trajectories.py \
  --trackon-sub trackon_droid --merged-sub merged_droid --per-lab 3 \
  --out $PTP_DATA/gt_trajectories_droid

date; echo done
