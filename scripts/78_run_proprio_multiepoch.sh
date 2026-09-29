#!/bin/bash
# Five-epoch run WITH proprioception: identical to 67_run_multiepoch.sh (steps 3790,
# eval every 758, same seed / lr / warmup / plans) plus --proprioception. The two runs
# are therefore a clean pair: same budget, one variable.
# Run under srun or an existing GPU compute-node shell. This script submits no jobs.
set -euo pipefail
cd /scratch/oozak23/droid_qp
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/oozak23-mpl
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo 'Usage: bash scripts/67_run_multiepoch.sh OUTPUT_DIRECTORY [RESUME_CHECKPOINT]' >&2
  exit 2
fi
run_dir="$1"
resume_args=()
if [ "$#" -eq 2 ]; then
  resume_args=(--resume "$2")
fi
/scratch/oozak23/envs/track_on_r/bin/python -B -u scripts/52_train_policy.py --proprioception \
  --mode train --steps 3790 --batch-size 16 --eval-every 758 --warmup 50 \
  --seed 2026 --lr 3e-4 --workers 4 --checkpoint \
  --validation-plan out/validation_plan_v1.json \
  --train-eval-plan out/training_eval_plan_v1.json \
  --out "$run_dir" "${resume_args[@]}"
# Write the learning comparison before the separate ablation, so it survives an
# interrupted post-training evaluation.
/scratch/oozak23/envs/track_on_r/bin/python -B -u scripts/68_report_multiepoch.py --run "$run_dir"
/scratch/oozak23/envs/track_on_r/bin/python -B -u scripts/57_evaluate_policy.py \
  --checkpoint "$run_dir/best.pt" --ablate --out "$run_dir/evaluation"
/scratch/oozak23/envs/track_on_r/bin/python -B -u scripts/68_report_multiepoch.py --run "$run_dir"
