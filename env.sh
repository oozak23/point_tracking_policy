# Paths for the job scripts; droid_qp/paths.py reads the same variables on the Python side.
# Put your own values in env.local.sh (git-ignored) instead of editing this file, e.g.
#     export PY_TRACKON=/path/to/your/envs/track_on_r/bin/python
#     export PTP_TRACK_ON=/path/to/your/track_on
# Every job script sources this; submit jobs from the repository root.
PTP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$PTP_REPO/env.local.sh" ]; then source "$PTP_REPO/env.local.sh"; fi

# One interpreter per conda env; they are not interchangeable (see README).
: "${PY_TRACKON:=/scratch/oozak23/envs/track_on_r/bin/python}"             # torch: model, training, evaluation
: "${PY_MOLMO:=/scratch/oozak23/envs/molmo-motion/bin/python}"             # SAM, h5py: object points, GT export
: "${PY_POINTWORLD:=/home/oozak23/.conda/envs/pointworld-env/bin/python}"  # kinematics: robot points, labels, rollout

# Dataset (read) and outputs (written). Same rule as droid_qp/paths.py.
if [ -z "${PTP_DATA:-}" ]; then
  if [ -d "$PTP_REPO/out/gen" ]; then PTP_DATA="$PTP_REPO/out"
  else PTP_DATA=/frozen/avg/ozan/point_tracking_policy_data; fi
fi
: "${PTP_OUT:=$PTP_REPO/out}"

export PTP_REPO PTP_DATA PTP_OUT PY_TRACKON PY_MOLMO PY_POINTWORLD
mkdir -p "$PTP_REPO/logs" "$PTP_OUT"
