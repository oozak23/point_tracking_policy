#!/bin/bash
#SBATCH --job-name=qp_chain
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/gen/logs/chainjob-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/gen/logs/chainjob-%j.err

# The batch chainer, running AS A SLURM JOB rather than a login-node background process.
#
# Why: the QOS caps submitted jobs at ~50, so `--array=0-140` is rejected and the 141
# shards must be submitted in batches as earlier ones drain. That needs a live process to
# do the waiting -- and a login-node process dies when the user's session ends (`nohup` is
# disabled here, so it cannot be detached). Running the waiter inside SLURM means the whole
# generation survives a closed laptop.
#
# Verified sbatch is available on the compute nodes (job 1638851), so a job can submit
# jobs. This one holds 1 CPU and 2 GB for the duration, which is negligible.
#
# Idempotent: 38_chain_shards.sh skips any shard with a DONE marker, so this can be
# re-submitted at any point and will only do what is left.
set -uo pipefail
cd /scratch/oozak23/droid_qp
date; hostname; echo "chainer starting"
exec bash scripts/38_chain_shards.sh
