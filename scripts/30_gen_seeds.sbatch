#!/bin/bash
#SBATCH --job-name=qp_seeds
#SBATCH --partition=ai
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/gen/logs/seeds-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/gen/logs/seeds-%j.err

# Stage 0, run ONCE before the shard array.
#
# Pulls the frame-0 object seed for all 14,083 overlap clips out of MolmoMotion's released
# tracks. The tar is 2.3 GB and 14_molmomotion_as_queries.py streams it in a single pass
# filtered by the wanted set -- so this is one sequential read, and sharding it would turn
# that into 141 reads of the same 2.3 GB off shared read-only storage.
#
# Frame 0 specifically: their AllTracker is BIDIRECTIONAL so object positions exist at
# frame 0, whereas Track-On is causal. Seeding at frame 0 leaves no blind window, and it
# is also the one MolmoMotion frame index that needs no rate conversion (frame 0 is frame
# 0 at any subsample; every other index needs rates.mm_frame_to_release).
#
# No GPU: no MolmoPoint, no SAM. Just tar streaming and numpy.
set -euo pipefail
cd /scratch/oozak23/droid_qp
mkdir -p out/gen/logs
date; hostname

/home/oozak23/.conda/envs/pointworld-env/bin/python -u \
  scripts/14_molmomotion_as_queries.py \
  --video-ids-from out/overlap_manifest.json --frame 0 \
  --out out/mm_seeds_f0

echo "seeds: $(ls out/mm_seeds_f0/query_points | wc -l) of 14083"
date; echo done
