#!/bin/bash
#SBATCH --job-name=qp_proprio5ep
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/proprio5ep-%j.out
#SBATCH --error=logs/proprio5ep-%j.err
# The new baseline: proprioception AND five epochs. Each helped on its own (proprio halved
# the loss; five epochs doubled scene use), never tested together. Pairs with the visual-only
# five-epoch run out/policy_multiepoch_1658338. Batch job, so no session can kill it.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname
nvidia-smi --query-gpu=name --format=csv,noheader
bash training/78_run_proprio_multiepoch.sh "$PTP_OUT/policy_proprio_5ep_${SLURM_JOB_ID}"
date; echo done
