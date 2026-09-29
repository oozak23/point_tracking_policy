#!/bin/bash
# Submit all 141 shards in batches, because the QOS caps SUBMITTED jobs per user (~50)
# and `--array=0-140` counts as 141 of them. Waits for each batch to drain before
# submitting the next, so the cap is never exceeded.
#
# Not an sbatch script -- it runs on the login node and only calls sbatch/squeue.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -uo pipefail
cd "$PTP_REPO"
BATCH=${BATCH:-35}
CONC=${CONC:-16}
LAST=${LAST:-140}
LOG=$PTP_DATA/gen/logs/chain.log
: > "$LOG"

# Wait for any generation array already in flight before touching anything. The skip
# logic below is DONE-marker based, and a shard that is RUNNING has no marker yet -- so
# without this the chainer would re-submit it, and 29_gen_shard.sbatch begins with
# `rm -rf "${DIR}"`, wiping a directory another task is writing.
while squeue -u "$USER" -h -n qp_gen 2>/dev/null | grep -q .; do
  echo "$(date +%H:%M:%S) waiting for an in-flight qp_gen array to drain" >>"$LOG"
  sleep 60
done

start=0
while [ "$start" -le "$LAST" ]; do
  end=$(( start + BATCH - 1 ))
  [ "$end" -gt "$LAST" ] && end=$LAST
  # skip shards already complete, so a re-run of this script costs nothing
  todo=""
  for i in $(seq "$start" "$end"); do
    [ -f "$(printf "$PTP_DATA/gen/%04d/DONE" "$i")" ] || todo="${todo}${todo:+,}${i}"
  done
  if [ -z "$todo" ]; then
    echo "$(date +%H:%M:%S) shards ${start}-${end} already complete, skipping" >>"$LOG"
    start=$(( end + 1 )); continue
  fi
  job=$(sbatch --parsable --array="${todo}%${CONC}" pipeline/29_gen_shard.sbatch 2>&1)
  if ! [[ "$job" =~ ^[0-9]+$ ]]; then
    echo "$(date +%H:%M:%S) SUBMIT FAILED for ${start}-${end}: $job" >>"$LOG"
    sleep 120; continue          # transient QOS pressure -- retry the same batch
  fi
  echo "$(date +%H:%M:%S) submitted job ${job} for shards ${start}-${end}" >>"$LOG"
  while squeue -h -j "$job" 2>/dev/null | grep -q .; do sleep 60; done
  ndone=$(ls $PTP_DATA/gen/*/DONE 2>/dev/null | wc -l)
  echo "$(date +%H:%M:%S) job ${job} drained -- ${ndone}/141 shards complete" >>"$LOG"
  start=$(( end + 1 ))
done
echo "$(date +%H:%M:%S) ALL BATCHES SUBMITTED AND DRAINED" >>"$LOG"
