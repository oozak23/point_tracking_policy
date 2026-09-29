#!/bin/bash
#SBATCH --job-name=qp_gt
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gt-%j.out
#SBATCH --error=logs/gt-%j.err

# Post-pass over every finished shard: pair FK ground truth with what Track-On produced,
# and persist both together.
#
# Run AFTER generation, not inside it. Every input already exists per shard --
# `merged/` holds fk_uv / fk_ok / point_index and `trackon/` holds tracks / visibility /
# role -- so this writes a convenience copy, not new information. Doing it as a separate
# pass means the 141 shards stay uniform and nothing already finished needs re-running.
#
# No GPU: numpy distances plus one overlay per clip.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname
readonly PY=${PY_MOLMO}

# --per-lab is a sampling cap; a large value means "every clip".
"${PY}" -u pipeline/34_gt_trajectories.py \
  --trackon-sub trackon --merged-sub merged \
  --per-lab 100000 --save-npz --out $PTP_DATA/gt_all

echo; echo "########## occlusion -> return: does the tracker come back to the RIGHT point? ##########"
"${PY}" -u pipeline/40_occlusion_recovery.py --gt $PTP_DATA/gt_all
date; echo done
