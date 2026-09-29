#!/bin/bash
#SBATCH --job-name=qp_proprio
#SBATCH --partition=avg
#SBATCH --account=avg
# Pin the card: avg also contains Tesla T4s, and an unpinned request that lands on one is
# ~10x slower -- which would make this A/B incomparable with the L40S pilot it measures
# against.
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/proprio-%j.out
#SBATCH --error=logs/proprio-%j.err

# One-epoch proprioception A/B against the visual-only pilot (job 1657753).
#
# A batch job rather than `nohup` inside the interactive session: this cluster disables
# nohup, which is why the first launch attempt died at 11:58 with a one-line log. It is
# also independent of interactive job 1658338, so a dropped connection cannot kill it.
#
# One variable changes: --proprioception (joint positions, joint velocities, gripper
# width at frame t -- never t+1). steps / batch / lr / warmup / seed / eval cadence /
# workers / checkpointing all match the pilot, verified before launch, and
# 70_verify_proprio.py passed 6/6 including the future-state leak test.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash training/73_run_proprio_pilot.sh "$PTP_OUT/policy_proprio_${SLURM_JOB_ID}"
date; echo done
