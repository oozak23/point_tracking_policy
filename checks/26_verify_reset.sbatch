#!/bin/bash
#SBATCH --job-name=qp_reset
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Removing the per-clip Predictor reload is only safe if reset() clears ALL per-clip
# state. Asserting that from the source is not the same as demonstrating it: re-run the
# same 8 clips with one shared Predictor and require BIT-IDENTICAL output against the
# smoke run, which used a fresh Predictor per clip.
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname

${PY_TRACKON} -u pipeline/16_run_trackon.py \
  --points-dir $PTP_DATA/smoke7_obj/query_points \
  --videos-dir $PTP_DATA/smoke7_obj/release_videos \
  --n 8 --out $PTP_DATA/reset_check

${PY_MOLMO} -u - <<'PYEOF'
import numpy as np, glob, os, sys
bad = 0
files = sorted(glob.glob("$PTP_DATA/reset_check/tracks/*.npz"))
print(f"\ncomparing {len(files)} clips against $PTP_DATA/smoke7_trackon (fresh-Predictor run)\n")
for f in files:
    ref = f.replace('reset_check', 'smoke7_trackon')
    if not os.path.exists(ref):
        print(f"  MISSING REF {os.path.basename(f)}"); bad += 1; continue
    a, b = np.load(f), np.load(ref)
    diffs = []
    for k in ['tracks', 'visibility', 'tokens', 'q_init', 'queries']:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            diffs.append(f"{k}: shape {x.shape} vs {y.shape}"); continue
        if x.dtype == bool:
            n = int((x != y).sum())
            if n: diffs.append(f"{k}: {n} bools differ")
        else:
            d = np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))
            if d > 0: diffs.append(f"{k}: max|diff|={d:.3e}")
    print(f"  {'IDENTICAL' if not diffs else 'DIFFERS  '} {os.path.basename(f)[:46]}"
          + ("" if not diffs else "  " + "; ".join(diffs)))
    bad += bool(diffs)
print(f"\n===== {len(files)-bad}/{len(files)} bit-identical =====")
sys.exit(1 if bad else 0)
PYEOF
date; echo done
