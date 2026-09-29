#!/bin/bash
#SBATCH --job-name=qp_5ep_resume
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/5ep-resume-%j.out
#SBATCH --error=logs/5ep-resume-%j.err
# Finish the five-epoch visual-only run that died with interactive session 1658338.
#
# That session ended at 14:29:29 during the step-3790 evaluation, so training had completed
# but the epoch-5 checkpoint was never written; the newest on disk is epoch 4 (step 3032).
# Resume restores model, optimizer, scheduler, step, data cursor and both RNG states, so
# steps 3033-3790 repeat the same trajectory. metrics.jsonl was trimmed back to step 3032
# first (it is append-only, and would otherwise log those steps twice). The epoch-4 state
# is preserved in epoch4_backup/.
#
# A batch job this time, so a closed terminal cannot kill it.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname
nvidia-smi --query-gpu=name --format=csv,noheader
bash training/67_run_multiepoch.sh $PTP_OUT/policy_multiepoch_1658338 \
  $PTP_OUT/policy_multiepoch_1658338/epoch4_backup/last.pt
date; echo done
