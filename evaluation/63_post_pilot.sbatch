#!/bin/bash
#SBATCH --job-name=qp_rollout_report
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rollout-%j.out
#SBATCH --error=logs/rollout-%j.err
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/${USER}-mpl
run_dir="$1"
${PY_POINTWORLD} -B -u evaluation/60_evaluate_rollout.py \
  --evaluation "$run_dir/evaluation" --out "$run_dir/evaluation/rollout.json"
${PY_POINTWORLD} -B -u evaluation/62_report_pilot.py --run "$run_dir"
