#!/bin/bash
#SBATCH --job-name=qp_copy_frozen
#SBATCH --account=ai
#SBATCH --partition=mid
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/oozak23/droid_qp/logs/copy_frozen-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/logs/copy_frozen-%j.err
# Copy (not move) the shareable dataset to the group-readable frozen folder so other
# `avg` members can read it. The scratch copy is kept until this one is verified.
# Directory layout mirrors droid_qp/out/, so code only has to change its data root.
set -euo pipefail
SRC=/scratch/oozak23/droid_qp/out
DST=/frozen/avg/ozan/point_tracking_policy_data
ITEMS=(gen gt_all shards mm_seeds_f0
       dataset_manifest.json shard_index.json manifest.json gripper_calibration.json
       validation_plan_v1.json training_eval_plan_v1.json policy_loss_scales.json)
RUNS=$(cd "$SRC" && ls -d policy_*/)

mkdir -p "$DST"
cd "$SRC"
echo "start $(date)"
# gen/logs holds Slurm logs of generation jobs, not data.
rsync -a --stats --exclude='gen/logs/' "${ITEMS[@]}" $RUNS "$DST"/

echo; echo "===== verify: a second pass must find nothing to transfer ====="
left=$(rsync -a --dry-run --itemize-changes --exclude='gen/logs/' "${ITEMS[@]}" $RUNS "$DST"/ | wc -l)
echo "items still differing: $left"
n_src=$(find "${ITEMS[@]}" $RUNS -path 'gen/logs' -prune -o -type f -print | wc -l)
n_dst=$(cd "$DST" && find . -type f | wc -l)
echo "files: source $n_src  destination $n_dst"

# group avg may read, nobody may write; others see nothing
chmod -R g+rX,g-w,o-rwx "$DST"
du -sh "$DST"
[ "$left" -eq 0 ] && [ "$n_src" -eq "$n_dst" ] && echo "VERIFIED" || echo "MISMATCH"
echo "done $(date)"
