#!/bin/bash
#SBATCH --job-name=qp_mask
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/oozak23/droid_qp/out/gen/logs/maskab-%j.out
#SBATCH --error=/scratch/oozak23/droid_qp/out/gen/logs/maskab-%j.err

# A/B/C the mask-frame choice on shard 0000 before it is baked into 13,705 clips.
#
#   birth  what we ran: mask built at the birth frame, now frame 0 -- the arm's START
#          position, small and often half out of view. Mean keep fell to ~91% with a long
#          tail (one TRI clip kept 14 of 128; one clip lost its robot half entirely).
#   best   mask built where the gripper is largest and fullest. One SAM call, same cost.
#   vote   three such frames, keep points a majority agree on. 3x the SAM cost for this
#          stage, robust to a single bad mask.
#
# The filter is frame-independent by construction: `presample(n, seed)` makes FK index n
# the same physical spot at every frame, so a decision made anywhere applies to points
# born at f0.
set -euo pipefail
cd /scratch/oozak23/droid_qp; date; hostname
readonly PY=/scratch/oozak23/envs/molmo-motion/bin/python
readonly D=out/gen/0000

for MODE in birth best vote; do
  echo; echo "########## mask-frame = ${MODE} ##########"
  "${PY}" -u scripts/20_robot_points_sam_filter.py \
    --robot-dir "${D}/robot_droid" --videos-dir "${D}/obj/release_videos" \
    --mask-frame "${MODE}" --mask-votes 3 --out "out/maskab_${MODE}"
done

echo; echo "########## comparison ##########"
"${PY}" -u - <<'PYEOF'
import json, numpy as np
rows = {}
for m in ("birth", "best", "vote"):
    r = json.load(open(f"out/maskab_{m}/report.json"))
    rows[m] = {x["video_id"]: x for x in r}
common = set.intersection(*(set(v) for v in rows.values()))
print(f"{len(common)} clips completed in all three modes\n")
print(f"{'mode':>6s} {'clips':>6s} {'mean keep':>10s} {'median kept':>12s} "
      f"{'min keep':>9s} {'clips <50%':>11s}")
for m, d in rows.items():
    k = np.array([d[v]["keep_frac"] for v in common])
    n = np.array([d[v]["n_kept"] for v in common])
    print(f"{m:>6s} {len(d):6d} {k.mean()*100:9.1f}% {np.median(n):12.0f} "
          f"{k.min()*100:8.1f}% {int((k<0.5).sum()):11d}")

print("\nthe clips 'birth' handled worst:")
worst = sorted(common, key=lambda v: rows["birth"][v]["keep_frac"])[:8]
print(f"  {'clip':44s} {'birth':>7s} {'best':>7s} {'vote':>7s}")
for v in worst:
    print(f"  {v[:44]:44s} " + " ".join(
        f"{rows[m][v]['keep_frac']*100:6.1f}%" for m in ("birth", "best", "vote")))
PYEOF
date; echo done
