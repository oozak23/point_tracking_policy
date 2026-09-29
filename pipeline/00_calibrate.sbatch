#!/bin/bash
#SBATCH --job-name=qp_calib
#SBATCH --partition=ai
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Phase 0: calibrate gripper_pose (which link) + quaternion order.
# CPU only -- URDF forward kinematics and NumPy, no GPU requested.

# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail

readonly REPO="${PTP_REPO}"
readonly PY=${PY_POINTWORLD}

cd "${REPO}"
mkdir -p out

date
hostname
echo "python: ${PY}"

"${PY}" -u pipeline/00_calibrate_gripper.py \
  --episodes 40 \
  --windows-per-episode 2 \
  --out $PTP_DATA/gripper_calibration.json

date
echo "done"
