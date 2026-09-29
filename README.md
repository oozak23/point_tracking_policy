# point_tracking_policy

A robot policy whose state is a **point memory**. A frozen Track-On-R tracker follows points
on the manipulated object (the scene) and points on the gripper (the robot); a trainable
replica of Track-On's decoder layer reads both memories and predicts the next action.
Data comes from DROID. Advisor: Fatma Güney, KUIS AI, Koç University.

Start with `docs/DECODER_REPORT.html` (how the decoder was built, the loss, every run and test)
and `docs/DECODER_LOG.md` (the step-by-step build log).

## Layout

| folder | what it holds |
|---|---|
| `droid_qp/` | the Python package: data loading, geometry, model, loss, training, evaluation |
| `droid_qp/paths.py` | **every location the code reads or writes**, overridable by environment variables |
| `pipeline/` | builds the dataset, in order: object points → robot points → SAM filter → merge → Track-On → action labels → ground truth → manifest → evaluation plans |
| `training/` | training entry point (`52_train_policy.py`) and the job scripts for each run |
| `evaluation/` | evaluation, memory-shuffle tests, stratified analysis, reports |
| `checks/` | the verification suite: run it after any change |
| `analysis/` | exploration and one-off diagnostics; kept for the record, not maintained |
| `docs/` | plans, decisions, build log, reports |
| `environment/` | conda env specs |
| `env.sh` | paths for the job scripts (sourced by every one of them) |

Scripts keep their original numbers. The number is the order they were written, and it's how
the build log and reports refer to them.

## The dataset

The shared copy is at **`/frozen/avg/ozan/point_tracking_policy_data/`**: readable by the `avg`
group, writable by nobody else. 13,449 clips (12,123 train / 1,326 validation, split by episode).

| path | what it is |
|---|---|
| `gen/0000 … 0141/` | the dataset, 142 shards (below) |
| `gt_all/` | FK ground-truth trajectories for the robot points, and the tracking-quality report |
| `dataset_manifest.json` | every clip: split, episode, tracking-quality weight |
| `shard_index.json` | which shard each clip is in |
| `validation_plan_v1.json`, `training_eval_plan_v1.json` | the frozen evaluation windows and donors |
| `policy_loss_scales.json` | the loss scales s_p and s_r, measured on the training split |
| `policy_*_<jobid>/` | the reference runs: checkpoints, metrics, evaluation reports |
| `manifest.json`, `shards/`, `mm_seeds_f0/`, `gripper_calibration.json` | pipeline inputs |

Inside each shard:

| folder | contents |
|---|---|
| `obj/` | object query points (100 per clip) and the reconstructed 7.2 Hz videos |
| `robot/`, `robot_sam/` | FK gripper points (`fk_uv`, `fk_ok`), before and after the SAM visibility filter |
| `merged/` | the combined object + robot Track-On queries |
| `trackon/` | tracks, visibility and the 256-number token per point per frame |
| `actions/` | action labels (body-frame pose deltas, gripper state) and proprioception |

**Never modify the evaluation files** (the manifest, both plans, the loss scales). Every result
is comparable to every other only because they have not changed.

## Setup

```bash
module load git                       # git is not on PATH by default

# 1. the code
git clone git@github.com:<owner>/point_tracking_policy.git && cd point_tracking_policy

# 2. the upstream repositories this code imports from
git clone https://github.com/gorkaydemir/track_on       # used at 7e838e8; checkpoint in checkpoints/track_on_r.pt
git clone https://github.com/NVlabs/PointWorld           # used at 0548482; pipeline only
git clone https://github.com/allenai/molmo-motion        # used at 61f5b21; pipeline only

# 3. the environments: one per upstream stack, not interchangeable
conda env create -f environment/track_on_r.yml       # torch: model, training, evaluation
conda env create -f environment/molmo-motion.yml     # SAM, h5py: object points, GT export
conda env create -f environment/pointworld-env.yml   # kinematics: robot points, labels, rollout
```

Then create `env.local.sh` at the repository root (git-ignored) with your own paths:

```bash
export PY_TRACKON=/path/to/envs/track_on_r/bin/python
export PY_MOLMO=/path/to/envs/molmo-motion/bin/python
export PY_POINTWORLD=/path/to/envs/pointworld-env/bin/python
export PTP_TRACK_ON=/path/to/track_on
export PTP_POINTWORLD=/path/to/PointWorld
export PTP_MOLMO_MOTION=/path/to/molmo-motion
# export PTP_OUT=/scratch/<you>/ptp_runs    # default: out/ inside the repository
```

Python reads the same variables through `droid_qp/paths.py`, so for interactive work run
`source env.sh` first. If `PTP_DATA` is unset, a fresh clone reads the shared dataset
automatically.

## Running

**Checks** (CPU, a few minutes). Run after any change; all must pass:

```bash
source env.sh
for s in 45_verify_dataloader 46_verify_replica 48_verify_readout 49_verify_head \
         51_verify_loss 54_verify_training 59_verify_validation 70_verify_proprio; do
  $PY_TRACKON -B checks/$s.py || echo "FAILED: $s"
done
$PY_POINTWORLD -B checks/61_verify_rollout.py      # needs the kinematics env
```

**Training and evaluation** run as batch jobs. Submit from the repository root, since the job
scripts find `env.sh` through the submission directory:

```bash
sbatch training/79_proprio_5ep.sbatch        # proprioception, 5 epochs, with evaluation at the end
```

Logs go to `logs/`; runs go to `$PTP_OUT`.

## Cluster notes

- Use batch jobs for anything long. Interactive sessions kill their processes when they close,
  and `nohup` is disabled.
- GPU jobs use `--account=avg --partition=avg --gres=gpu:lovelace_l40s:1`. Always pin the L40S:
  unpinned requests can land on a T4, about 10× slower.
- `/frozen` is read-only shared storage. Never extract archives or write there.
