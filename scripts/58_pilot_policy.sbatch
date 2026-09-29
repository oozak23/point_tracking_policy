#!/bin/bash
#SBATCH --job-name=qp_pilot1epoch
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/gen/logs/pilot-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/gen/logs/pilot-%j.err
set -euo pipefail
cd /scratch/oozak23/droid_qp
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
run_dir="/scratch/oozak23/droid_qp/out/policy_pilot_${SLURM_JOB_ID}"
/scratch/oozak23/envs/track_on_r/bin/python -B -u scripts/52_train_policy.py \
  --mode train --steps 758 --batch-size 16 --eval-every 190 --warmup 50 \
  --workers 4 --checkpoint --validation-plan out/validation_plan_v1.json --out "$run_dir"
/scratch/oozak23/envs/track_on_r/bin/python -B -u scripts/57_evaluate_policy.py \
  --checkpoint "$run_dir/best.pt" --ablate --out "$run_dir/evaluation"
