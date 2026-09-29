# analysis/ — exploration and one-off diagnostics

**Kept for the record, not maintained.** These scripts answered specific questions while the
project was being designed. Many read folders from those experiments that aren't in the shared
dataset (e.g. `mm_sam`, `smoke7_*`, the 40-clip evaluation set), so most won't run as-is. The
conclusions are written up in `docs/`. Read these for *how* a decision was reached, not to
reuse the code.

## Choosing the object query points

How to put query points on the manipulated object, and how to tell whether they're good.

| script | question |
|---|---|
| `02_extract_nouns.py` | caption → object noun with Qwen3 (abandoned: DROID captions are too vague) |
| `03a_sam_check.py` | does SAM 2.1 + k-means give good points, independent of grounding? |
| `05_score_points.py`, `05_score.sbatch` | an objective score: do the points actually move with the gripper? |
| `06_select_eval_clips.py` | the fixed clip set every approach was compared on |
| `07_rerun_approach1.sbatch`, `13_natural_eval.sbatch`, `15_mm_seeded.sbatch`, `18_frame0_seed.sbatch` | the grounding variants compared (own MolmoPoint vs MolmoMotion's released seeds, frame 0 vs detection frame) |
| `08_calibrate_threshold.py` | calibrate the "moves with the gripper" threshold from data |
| `09_diag_gripper_at_detection.py`, `10_probe_gripper_region.py`, `11_diag_camera_effect.py` | why some clips scored near zero |
| `12_viz_attached.py`, `17_tracked_to_queries.py` | visualise and re-score tracked points |

→ `docs/PLAN.md`, `docs/RESULTS.md`

## Settling dataset choices

| script | question |
|---|---|
| `21_smoke_7hz.sbatch` | the smoke test for moving from 3.6 Hz to 7.2 Hz |
| `22_probe_position.py` | can a point's (x, y) be read back out of its Track-On token? (decision #1) |
| `23_sweep_min_visible.py` | how many visible FK points to require at the robot's birth frame |
| `36_rerun_pilot_droid.sbatch` | re-run the early 194-clip dataset test on corrected robot points (joints read from DROID, not PointWorld) |
| `37_maskframe_ab.sbatch` | which frame(s) to build the gripper SAM mask on (chose a 3-frame vote) |

→ `docs/DECISIONS.md`, `docs/CHANGES_7HZ.md`

## Looking at the data

| script | draws |
|---|---|
| `25_viz_actions.py` | the action labels over time |
| `32_viz_tracked_roles.py` | tracked videos, object and robot points in different colours |
| `33_points_vs_action.py` | which points produced a given motion |
| `35_viz_fk_only.py`, `42_viz_gt_readable.py` | FK ground-truth trajectories (42 is the legible version) |

`dbg_rp.py` and `inspect_gripper.py` are small debugging helpers.
