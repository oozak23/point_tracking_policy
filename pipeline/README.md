# pipeline/ — builds the dataset

**You probably don't need this folder.** The dataset is already built and shared at
`/frozen/avg/ozan/point_tracking_policy_data`. Use it only to rebuild the data or change how it
is made. In that case, point `PTP_DATA` at a directory you can write to: the shared copy is
read-only.

**The numbers are not the run order.** They are the order the scripts were written, which is
how the build log refers to them. The real order has three phases.

## A. Once, before generating anything

| step | run | env | produces |
|---|---|---|---|
| 1 | `01_build_manifest.py` | any | `manifest.json`: which DROID clips have a source video and a `trajectory.h5` |
| 2 | `00_calibrate.sbatch` → `00_calibrate_gripper.py` | `$PY_POINTWORLD` | `gripper_calibration.json` |
| 3 | `04_overlap_manifest.py` | `$PY_MOLMO` | `overlap_manifest.json`: the 14,083 clips in both MolmoMotion and PointWorld |
| 4 | `30_gen_seeds.sbatch` → `14_molmomotion_as_queries.py` | `$PY_POINTWORLD` | `mm_seeds_f0/`: one seed point on the manipulated object per clip |
| 5 | `27_build_shards.py` | any | `shards/`: ~100 clips per shard, about an hour of work each |

## B. Per shard — one job does all of it

`29_gen_shard.sbatch` is an array job, one task per shard. Each task runs:

| stage | script | env | writes into `gen/<shard>/` |
|---|---|---|---|
| 1 | `03_object_points.py` | `$PY_MOLMO` | `obj/`: seed → SAM 2.1 mask → k-means → 100 object points, plus the 7.2 Hz video |
| 2 | `19_robot_points.py` | `$PY_POINTWORLD` | `robot/`: 128 points on the gripper mesh, FK-projected every frame (`fk_uv`, `fk_ok`) |
| 3 | `20_robot_points_sam_filter.py` | `$PY_MOLMO` | `robot_sam/`: only the gripper points SAM says are visible |
| 4 | `24_action_labels.py` | `$PY_POINTWORLD` | `actions/`: body-frame action labels and proprioception |
| 5 | `28_merge_queries.py`, then `16_run_trackon.py` | `$PY_POINTWORLD`, `$PY_TRACKON` | `merged/`, then `trackon/`: tracks and the 256-number tokens |

The cluster caps array submissions at about 50 tasks, so `38_chain_shards.sh` (submitted by
`41_chain_job.sbatch`) feeds the 142 shards through in batches. When they finish,
`31_gen_report.py` lists failed shards and prints the exact `--array` expression to resubmit.
A shard is done when `gen/<shard>/DONE` exists.

## C. Once, after all shards finish

| step | run | env | produces |
|---|---|---|---|
| 1 | `39_export_gt.sbatch` → `34_gt_trajectories.py`, `40_occlusion_recovery.py` | `$PY_MOLMO` | `gt_all/`: FK ground truth and each clip's tracking error |
| 2 | `43_quality_histogram.py` | `$PY_MOLMO` | the tracking-quality histogram |
| 3 | `44_build_dataset_manifest.py` | any | `dataset_manifest.json`: 50 px cut, clip weights, train/val split by episode |
| 4 | `50_compute_loss_scales.py` | `$PY_TRACKON` | `policy_loss_scales.json`: s_p and s_r, from training labels only |
| 5 | `56_prepare_validation.py` | `$PY_TRACKON` | `validation_plan_v1.json`: frozen validation windows and shuffle donors |
| 6 | `66_prepare_training_eval.py` | any | `training_eval_plan_v1.json`: frozen training windows for train-vs-val comparison |
| 7 | `81_copy_to_frozen.sbatch` | — | the shared copy |

## Two cautions

- **Steps C3–C6 define the evaluation.** Re-running them changes the split, the loss scales or
  the evaluation windows, and new results would no longer be comparable with anything already
  reported. `66` refuses to overwrite an existing plan for exactly this reason.
- **C6 depends on training.** It recovers the windows from a finished diagnostic run
  (`policy_fit_diagnostic_1658338`), so it can only run after that exists.

Why each stage is built the way it is — the SAM vote, the birth-frame rule, reading joints
from DROID rather than PointWorld — is in the comments at the top of each script and in
`docs/CHANGES_7HZ.md`.
