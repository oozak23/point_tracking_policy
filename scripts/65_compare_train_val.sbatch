#!/bin/bash
#SBATCH --job-name=qp_seen_vs_val
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/gen/logs/fit-diagnostic-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/gen/logs/fit-diagnostic-%j.err
set -euo pipefail
cd /scratch/oozak23/droid_qp
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
/scratch/oozak23/envs/track_on_r/bin/python -B -u scripts/64_compare_train_val.py \
  --checkpoint out/policy_pilot_1657753/best.pt \
  --plan out/validation_plan_v1.json \
  --batch-size 16 --workers 4 \
  --out "out/policy_fit_diagnostic_${SLURM_JOB_ID}"
