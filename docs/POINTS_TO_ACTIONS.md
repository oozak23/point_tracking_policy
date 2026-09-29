# Points → actions — can the action be read back from the gripper's points?

2026-09-29 · job 1718573 · `evaluation/83_points_to_actions.py` · all 1,326 validation windows

**Question (Fatma hoca):** if we know where the gripper's points are, can we recover the action?
If perfect points cannot give the action back, a policy that predicts points is pointless.

**Method.** Only the pose change is recovered (translation, rotation; not the gripper opening),
from the 68 palm points that are rigid with the wrist flange. The other 60 points, on the fingers,
move up to 58 mm as the gripper opens. 3D points → pose by a rigid fit (Kabsch); 2D pixels →
pose by PnP with the camera intrinsics. Two poses → the body-frame action, exactly as the labels
define it. Horizons of 1, 4, 8 and 16 steps (0.14 s to 2.2 s).

## Sanity checks (all passed)

- Test 0 (`checks/82_verify_pose_fit.py`, 9 checks): the converter recovers known poses and
  motions exactly, reproduces the stored action labels on a real clip, and returns "unsolvable"
  (not a crash) for degenerate points.
- The camera model reproduces the stored FK pixels to 0.00003 px.
- FK of the recorded joints matches the recorded flange pose to 0.0001 mm and 0.004°.
- Momentum at k = 1 is 4.55 mm, exactly the copy-last baseline used everywhere else.

## Results — mean translation error (mm), rotation (°), coverage

| test | k = 1 | k = 4 | k = 8 | k = 16 |
|---|---|---|---|---|
| zero motion | 13.0 · 2.0° | 49.6 · 7.5° | 90.7 · 13.6° | 149.8 · 22.8° |
| momentum (repeat last action) | **4.55** · 0.8° | **36.8** · 5.8° | 97.0 · 14.7° | 225.5 · 33.6° |
| 1 exact 3D points | 0.00 · 100% | 0.00 · 100% | 0.00 · 100% | 0.00 · 100% |
| 2 exact 2D pixels | 0.00 · 95% | 0.00 · 95% | 0.00 · 94% | 0.00 · 93% |
| 2v exact 2D, only points Track-On sees | 0.00 · 85% | 0.00 · 83% | 0.00 · 82% | 0.00 · 80% |
| **4 Track-On's tracked pixels** | **30.9** · 5.4° · 85% | **71.2** · 12.0° | **109.9** · 18.3° | **161.9** · 26.9° |
| 4r same, robust PnP (RANSAC) | 97.6 · 8.8° | 139.3 · 14.4° | 199.3 · 19.9° | 248.8 · 27.7° |

On the same frame pairs, test 4 against the trivial baselines:

| k | test 4 mean / median | zero motion mean / median | momentum mean / median |
|---|---|---|---|
| 1 | 30.9 / 12.4 | 12.9 / 9.9 | **4.5 / 3.3** |
| 4 | 71.2 / 37.3 | 48.7 / 38.2 | **36.3 / 28.2** |
| 8 | 109.9 / **64.0** | **88.5** / 73.7 | 95.3 / 79.4 |
| 16 | 161.9 / **101.0** | **145.2** / 130.2 | 221.6 / 194.5 |

Controlled noise on exact pixels (mean mm, k = 1 / k = 16):

| noise | 1 px | 2 px | 4 px | 8 px | 16 px |
|---|---|---|---|---|---|
| independent per point | 5.2 / 5.8 | 10.3 / 11.5 | 20.9 / 23.3 | 41.6 / 46.6 | 85.9 / 97.0 |
| whole cluster drifts together | 3.5 / 3.7 | 7.1 / 7.3 | 14.2 / 14.7 | 28.5 / 29.5 | 57.3 / 59.1 |
| half the points dropped | 7.3 / 8.2 | 14.6 / 16.4 | 29.7 / 33.5 | 59.2 / 66.9 | 119.1 / 134.8 |
| one side of the palm hidden | 9.4 / 10.5 | 19.6 / 21.8 | 38.4 / 42.8 | 75.8 / 85.4 | 170.4 / 192.3 |

Track-On's palm points are a median 10.2 px from their FK positions. It collapsed all visible palm
points onto one spot in 147 of 83,227 frames (0.2%), so collapse is not the main problem.

## What it means

1. **In principle, yes.** Perfect points give the action exactly, even from 2D pixels alone,
   in 95% of frames (85% when limited to the points Track-On sees). The gripper's pose is fully
   determined by where its points are in the image. The conversion is not the bottleneck.
2. **With Track-On's real tracks, no.** At one step the error is 31 mm, 7× copy-last. Averaged
   over frames, it is worse than simply predicting no motion at every horizon.
3. **About the decision rule.** The rule fixed beforehand was "viable at a horizon if test 4 beats
   momentum". At k = 16 it does (162 vs 222 mm). But by then momentum has become worse than
   predicting no motion (222 vs 145 mm), and test 4 loses to no-motion. **I don't count this as a
   pass.** The rule's baseline was too weak at long horizons.
4. **The required accuracy.** To match copy-last at one step, points need to be right to about
   1 px. Track-On's are about 10 px off.
5. **Track-On's errors are structured, not random.** 16 px of random noise still gives 97 mm at
   k = 16, better than no motion. Track-On's 10 px gives 162 mm. Its error also grows with the
   horizon, while random noise stays flat: tracker drift accumulates over time, and a cluster
   that slides along the gripper looks like real motion.
6. **The typical frame is better than the average.** From k = 8 on, test 4's *median* beats both
   baselines (101 vs 130 and 195 mm at k = 16). A tail of failed fits ruins the mean: at k = 1,
   5% of pairs are over 100 mm. Robust PnP made it worse, not better.

## Implications for a points-based policy

- **Predict FK gripper points, not tracker points.** The FK positions are exact labels, and with
  exact points the conversion is exact (tests 1–2).
- **Predict chunks, not single steps.** The conversion's error per pose does not grow with the
  horizon (noise rows are flat from k = 1 to 16), while the motion does. So the conversion's
  relative error falls as the horizon grows. At one step, ~1 px accuracy is out of reach.
- **Reject bad fits.** The tail comes from a minority of frames. A fit-quality filter (PnP
  reprojection residual) is the obvious next test.
