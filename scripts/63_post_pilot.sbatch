#!/bin/bash
#SBATCH --job-name=qp_rollout_report
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/gen/logs/rollout-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/gen/logs/rollout-%j.err
set -euo pipefail
cd /scratch/oozak23/droid_qp
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/oozak23-mpl
run_dir="$1"
/home/oozak23/.conda/envs/pointworld-env/bin/python -B -u scripts/60_evaluate_rollout.py \
  --evaluation "$run_dir/evaluation" --out "$run_dir/evaluation/rollout.json"
/home/oozak23/.conda/envs/pointworld-env/bin/python -B -u scripts/62_report_pilot.py --run "$run_dir"
