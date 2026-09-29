#!/bin/bash
#SBATCH --job-name=qp_proprio5ep
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/gen/logs/proprio5ep-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/gen/logs/proprio5ep-%j.err
# The new baseline: proprioception AND five epochs. Each helped on its own (proprio halved
# the loss; five epochs doubled scene use), never tested together. Pairs with the visual-only
# five-epoch run out/policy_multiepoch_1658338. Batch job, so no session can kill it.
set -euo pipefail
cd /scratch/oozak23/droid_qp; date; hostname
nvidia-smi --query-gpu=name --format=csv,noheader
bash scripts/78_run_proprio_multiepoch.sh "out/policy_proprio_5ep_${SLURM_JOB_ID}"
date; echo done
