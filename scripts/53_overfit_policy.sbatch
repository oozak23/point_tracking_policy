#!/bin/bash
#SBATCH --job-name=qp_overfit10
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/gen/logs/overfit-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/gen/logs/overfit-%j.err
set -euo pipefail
cd /scratch/oozak23/droid_qp
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
/scratch/oozak23/envs/track_on_r/bin/python -B -u scripts/52_train_policy.py \
  --mode overfit --clips 10 --batch-size 10 --steps 1000 --eval-every 50 \
  --lr 0.0003 --warmup 20 --no-checkpoint \
  --out "/scratch/oozak23/droid_qp/out/policy_overfit_${SLURM_JOB_ID}"
