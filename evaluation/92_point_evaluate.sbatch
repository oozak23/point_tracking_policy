#!/bin/bash
#SBATCH --job-name=ptp_eval
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/point_evaluate-%j.out
#SBATCH --error=logs/point_evaluate-%j.err
# Full evaluation of a point-policy run (docs/POINT_POLICY_PLAN.md), on its best checkpoint:
# memory-shuffle tests with point metrics and the perfect-plan bound (57), then the scene shuffle
# split by difficulty (75). Usage: sbatch evaluation/92_point_evaluate.sbatch <run directory>
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname
run="$1"
[ -f "$run/best.pt" ] || { echo "no checkpoint at $run/best.pt" >&2; exit 2; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/${USER}-mpl
${PY_TRACKON} -B -u evaluation/57_evaluate_policy.py --checkpoint "$run/best.pt" --ablate --out "$run/evaluation"
${PY_TRACKON} -B -u evaluation/75_stratified_scene_shuffle.py --checkpoint "$run/best.pt" --out "$run/stratified_scene"
date; echo done
