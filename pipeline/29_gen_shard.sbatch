#!/bin/bash
#SBATCH --job-name=qp_gen
#SBATCH --partition=avg
# `avg` is the lab's own partition (rtx_a6000 / lovelace_l40s, 48 GB) and is
# AllowAccounts=avg -- without naming the account the job sits in PENDING
# (PartitionConfig) forever, even with idle nodes.
#SBATCH --account=avg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/shard-%a-%A.out
#SBATCH --error=logs/shard-%a-%A.err

# One array task = one shard of ~100 clips, carried end to end through all five stages.
#
# Why one task rather than five dependent array jobs: the three conda envs are mutually
# incompatible (pointworld-env has pytorch_kinematics but no transformers; molmo-motion
# has transformers but no kinematics), so the work must cross interpreters regardless.
# Doing it inside one task keeps every shard independent -- no cross-job dependencies, and
# a task that dies takes 100 clips with it instead of 14,083.
#
# Stage 0 (MolmoMotion seeds) is NOT here. It streams a single 2.3 GB tar sequentially and
# must run ONCE, before this array; 141 tasks each re-streaming it off /frozen would be
# absurd. See 30_gen_seeds.sbatch.
#
#   sbatch --array=0-140%16 pipeline/29_gen_shard.sbatch
# %16 caps concurrency: the stages read /frozen heavily and the partition is shared.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
readonly REPO="${PTP_REPO}"
readonly PY_GPU=${PY_MOLMO}
readonly PY_TO=${PY_TRACKON}
readonly PY_PW=${PY_POINTWORLD}

readonly SHARD=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
readonly DIR="${PTP_DATA}/gen/${SHARD}"
readonly SEEDS="${PTP_DATA}/mm_seeds_f0/query_points"

cd "${REPO}"
mkdir -p "${PTP_DATA}/gen/logs"
date; hostname; echo "shard ${SHARD}"

# Resume at shard granularity. A completed shard is skipped outright, so re-submitting the
# whole array after a partial run costs only the failed shards. Granularity is the shard
# and not the clip because stages 2-5 mirror stage 1's output directory with a glob, so a
# partially-populated directory would silently change what the later stages see.
if [[ -f "${DIR}/DONE" ]]; then
  echo "shard ${SHARD} already complete -- skipping"; exit 0
fi
if [[ ! -d "${SEEDS}" ]]; then
  echo "FATAL: ${SEEDS} missing -- run 30_gen_seeds.sbatch first" >&2; exit 1
fi
# A partial directory from a killed task is not resumable for the reason above.
rm -rf "${DIR}"; mkdir -p "${DIR}"

echo; echo "===== 1/5 object points (reconstructs video @7.2Hz, SAM, k-means) ====="
"${PY_GPU}" -u pipeline/03_object_points.py --text-free \
  --seed-from-npz "${SEEDS}" --mask-choice best_iou --kmeans-k 100 \
  --video-ids-from "$PTP_DATA/shards/${SHARD}.json" --n 0 --out "${DIR}/obj"

echo; echo "===== 2/5 robot points via FK (DROID joints) ====="
# --source droid for the same reason as the action labels, plus one specific to FK:
# `episode_joint_series` ZERO-FILLS steps PointWorld does not cover, and zeros are a valid
# pose to FK -- the arm standing upright. That usually projects out of frame so `fk_ok`
# rejects it, but that is luck depending on camera placement: measured over 80 clips,
# 1.085% of uncovered point-steps pass anyway and enter the ground truth as fake
# positions (one TRI clip: 2,304 of 2,304). Reading DROID removes the failure mode
# instead of masking it, and takes FK coverage 75.9% -> 96.8% with points born at frame 0.
"${PY_PW}" -u pipeline/19_robot_points.py \
  --like "${DIR}/obj" --n 0 --num-points 128 --gripper-only --min-visible 64 \
  --source droid --out "${DIR}/robot"

echo; echo "===== 3/5 robot points filtered by SAM mask (3-frame vote) ====="
# --mask-frame vote: the filter asks which of the 128 FK indices sit on the VISIBLE
# gripper, and `presample(n, seed)` makes index n the same physical spot at every frame --
# so the question is frame-independent and the mask can be built wherever the gripper is
# clearest. Evaluating it at the BIRTH frame (now frame 0, the arm's start position, small
# and often half out of view) was costing points for no reason.
#
# A/B/C on shard 0000 (job 1638594), 106 clips:
#     birth  mean 91.8%  min 10.9%  median 121   5 clips below 50%
#     best   mean 92.9%  min 48.8%  median 120   1 clip  below 50%
#     vote   mean 96.6%  min 57.8%  median 123   0 clips below 50%
# `vote` recovers the 96.6% we had before births moved to frame 0, so it pays back that
# change in full while keeping its coverage benefit. Costs 3 SAM calls instead of 1 on
# this stage only.
"${PY_GPU}" -u pipeline/20_robot_points_sam_filter.py \
  --robot-dir "${DIR}/robot" --videos-dir "${DIR}/obj/release_videos" \
  --mask-frame vote --mask-votes 3 --out "${DIR}/robot_sam"

echo; echo "===== 4/5 action labels (from DROID, not PointWorld) ====="
# --source droid: the fields are bit-identical to PointWorld's copy (verified 0.000e+00 on
# joint_positions, joint_velocities, cartesian_position, gripper_positions) but DROID has
# EVERY step while PointWorld's windows cover a median 75% of an episode. Measured on the
# 194-clip pilot: valid frames 71.7% -> 99.6%, clips under 50% valid 26 -> 0, worst clip
# 6.9% -> 97.9%. The gaps were also not uniform -- the first tenth of each episode was
# 1.7x over-represented among missing frames, i.e. we were disproportionately blind to the
# approach phase, which is the hardest decision the policy makes.
"${PY_PW}" -u pipeline/24_action_labels.py \
  --like "${DIR}/obj" --n 0 --source droid --out "${DIR}/actions"

echo; echo "===== 5/5 merge queries, then frozen Track-On -> token stream ====="
"${PY_PW}" -u pipeline/28_merge_queries.py \
  --obj "${DIR}/obj" --robot "${DIR}/robot_sam" --out "${DIR}/merged"
"${PY_TO}" -u pipeline/16_run_trackon.py \
  --points-dir "${DIR}/merged/query_points" \
  --videos-dir "${DIR}/obj/release_videos" \
  --n 0 --out "${DIR}/trackon"

# Written only after every stage exits 0 (set -e), so its presence means the shard is
# genuinely complete rather than merely attempted.
date > "${DIR}/DONE"
echo "shard ${SHARD} complete"
