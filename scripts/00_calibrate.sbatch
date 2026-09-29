#!/bin/bash
#SBATCH --job-name=qp_calib
#SBATCH --partition=ai
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/slurm-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/slurm-%j.err

# Phase 0: calibrate gripper_pose (which link) + quaternion order.
# CPU only -- URDF forward kinematics and NumPy, no GPU requested.

set -euo pipefail

readonly REPO=/scratch/oozak23/droid_qp
readonly PY=/home/oozak23/.conda/envs/pointworld-env/bin/python

cd "${REPO}"
mkdir -p out

date
hostname
echo "python: ${PY}"

"${PY}" -u scripts/00_calibrate_gripper.py \
  --episodes 40 \
  --windows-per-episode 2 \
  --out out/gripper_calibration.json

date
echo "done"
