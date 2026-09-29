#!/bin/bash
#SBATCH --job-name=qp_stratscene
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/gen/logs/stratscene-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/gen/logs/stratscene-%j.err
# Evaluation only -- no training. Three forward passes per validation window (clean,
# scene-shuffled, proprio-shuffled) on the proprioception checkpoint, then stratified.
set -euo pipefail
cd /scratch/oozak23/droid_qp; date; hostname
/scratch/oozak23/envs/track_on_r/bin/python -B -u scripts/75_stratified_scene_shuffle.py \
  --checkpoint out/policy_proprio_1658371/best.pt
date; echo done
