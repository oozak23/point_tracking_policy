#!/bin/bash
#SBATCH --job-name=qp_copy_frozen
#SBATCH --account=ai
#SBATCH --partition=ai
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
# NO trailing slash: to rsync, "policy_x/" means the folder's CONTENTS. The first run of
# this script had one, which poured all nine runs into the top level of $DST.
RUNS=$(cd "$SRC" && ls -d policy_*/ | sed 's#/$##')

mkdir -p "$DST"
cd "$SRC"
echo "start $(date)"
# gen/logs holds Slurm logs of generation jobs, not data.
rsync -a --stats --exclude='gen/logs/' "${ITEMS[@]}" $RUNS "$DST"/

echo; echo "===== verify ====="
# 1. a second rsync pass must find nothing to transfer (before permissions change below)
left=$(rsync -a --dry-run --itemize-changes --exclude='gen/logs/' "${ITEMS[@]}" $RUNS "$DST"/ | wc -l)
# 2. the exact same file LIST on both sides. A self-consistent rsync can still put files in
#    the wrong place (see RUNS above); comparing the lists is what catches that.
tmp=$(mktemp -d)
find "${ITEMS[@]}" $RUNS -path 'gen/logs' -prune -o -type f -print | sort > "$tmp/src"
(cd "$DST" && find . -type f | sed 's#^\./##' | sort) > "$tmp/dst"
missing=$(comm -23 "$tmp/src" "$tmp/dst" | wc -l); extra=$(comm -13 "$tmp/src" "$tmp/dst" | wc -l)
echo "rsync items differing: $left   files: $(wc -l < "$tmp/src") source, missing $missing, extra $extra"

# rsync -a keeps the source group (hpcusers = every cluster user); the dataset belongs to avg.
# avg may read, nobody may write, others see nothing; setgid so new files inherit avg.
chgrp -R avg "$DST"
chmod -R g+rX,g-w,o-rwx "$DST"
find "$DST" -type d -exec chmod g+s {} +
du -sh "$DST"
[ "$left" -eq 0 ] && [ "$missing" -eq 0 ] && [ "$extra" -eq 0 ] && echo "VERIFIED" || echo "MISMATCH"
echo "done $(date)"
