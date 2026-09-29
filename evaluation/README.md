# evaluation/ — score a trained policy

All evaluation uses the **frozen validation plan** (`$PTP_DATA/validation_plan_v1.json`):
63,598 targets in 1,326 windows, from episodes never seen in training. Every number in the
reports is on exactly these frames, so results are directly comparable.

These are **one-step, teacher-forced** predictions: at each frame the model sees the real
recorded history and predicts the next action. Nothing here measures closed-loop success.

## Evaluate a checkpoint

Needs a GPU. Run it in a batch job or under `srun`, not on the login node:

```bash
source env.sh
$PY_TRACKON -B evaluation/57_evaluate_policy.py \
  --checkpoint $PTP_OUT/<run>/best.pt --ablate --out $PTP_OUT/<run>/evaluation
```

Without `--ablate` it computes only the clean metrics. With it, it also runs the memory-shuffle
tests below. The output is `report.json`, with totals and a per-lab breakdown.

## What the numbers mean

| metric | meaning |
|---|---|
| `position_mm` | mean distance between predicted and true gripper translation, per frame |
| `rotation_deg` | mean angle between predicted and true rotation |
| `copy_position_mm` | **the bar**: error of simply repeating the previous action (4.55 mm) |
| `zero_mm` | error of predicting no motion (13.0 mm); easy to beat, proves little |
| `events` | gripper open/close events matched within ±2 frames: matched, missed, extra, F1 |
| `timing_cost_frames` | (Σ timing error + 3 × (missed + extra)) / true events. Lower is better. Repeating the current gripper state scores 1.06 |

**The gates**, fixed in `docs/PILOT_PLAN.md` before the first run:

1. position error below 4.35 mm **and** below copy-last;
2. shuffling the scene must worsen position error by **more than 10%**;
3. gripper timing cost below 1.06 frames.

Only the five-epoch proprioception run passes any of them (gate 3).

## The memory-shuffle tests (`--ablate`)

Each test swaps one input for the same input from another validation episode and measures
how much worse the predictions get. The donor episodes are fixed in the validation plan.

| report key | what is swapped | scored on |
|---|---|---|
| `scene_shuffle` | the whole object-point memory | all frames |
| `robot_history_anchor` | the robot's past tokens; its current frame and scene are kept | the last frame of each window only |
| `clean_anchor`, `scene_anchor` | nothing / the scene, on those same last frames | the baseline for the row above |
| `proprio_shuffle` | joint state and gripper width (proprioception runs only) | all frames |

The robot-history test scores only the last frame, because that is the only frame whose present
is real while its whole memory comes from someone else. A large effect means the model depends
on that input; a small one does not prove the input is unused.

## Other scripts

| script | what it does | env |
|---|---|---|
| `75_stratified_scene_shuffle.py` | splits the scene-shuffle effect by frame difficulty. It first checks that it reproduces the run's reported effect; `--reuse` re-analyses saved results without a GPU | `$PY_TRACKON` |
| `60_evaluate_rollout.py` | chains predicted actions over 20 and 40 steps and projects the gripper into the image. Reads the predictions `57` saved | `$PY_POINTWORLD` |
| `64_compare_train_val.py` | the same checkpoint on windows it trained on vs validation windows, to tell underfitting from overfitting | `$PY_TRACKON` |
| `83_points_to_actions.py` (job: `84_...sbatch`, CPU, ~2 min) | can the action be recovered from the gripper's points? Tests 1–4 of the points → actions experiment; results in `docs/POINTS_TO_ACTIONS.md` | `$PY_POINTWORLD` |
| `62_report_pilot.py`, `68_report_multiepoch.py`, `72_report_proprio.py` | turn saved results into the Markdown reports and plots in each run folder | `$PY_TRACKON` |

The `.sbatch` files are the exact jobs behind the reported results. `76` and `80` run `75` on
the one-epoch proprioception and five-epoch visual runs, and `65` runs `64` on the pilot. They
read the reference runs from `$PTP_DATA` and write to `$PTP_OUT`. `63` takes a run folder as its
argument and writes the rollout and report into it, so point it at one of your own runs.
