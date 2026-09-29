# training/ — train the policy

Everything goes through one entry point, `52_train_policy.py`. The `.sbatch` and `.sh` files
are the exact commands behind each reference run, kept so every result can be reproduced.

## Start a run

Submit from the **repository root** (job scripts find `env.sh` through the submission directory):

```bash
sbatch training/79_proprio_5ep.sbatch      # the current best setup: proprioception, 5 epochs
```

To run something new, copy the closest job script, then change its `--job-name` and output
folder. Always write to a new output folder.

- **Proprioception, 5 epochs:** `79_proprio_5ep.sbatch` → `78_run_proprio_multiepoch.sh`.
- **Visual-only, 5 epochs:** no fresh job script exists. Copy `79`, and make it call
  `67_run_multiepoch.sh` instead of `78_...`. (`77_resume_multiepoch.sbatch` only resumes one
  specific interrupted run.)

GPU jobs use `--account=avg --partition=avg --gres=gpu:lovelace_l40s:1`. Pin the L40S: the
`avg` partition also has T4s, about 10× slower. One epoch takes about 42 minutes at batch 16;
five epochs plus the final evaluation took 3 hours.

## `52_train_policy.py`

| flag | meaning |
|---|---|
| `--mode overfit` / `train` | `overfit` memorises a few fixed windows (the wiring test); `train` uses the full split |
| `--out DIR` | output folder (required) |
| `--steps`, `--batch-size` | updates and windows per update. One epoch = 758 updates at batch 16 |
| `--eval-every` | validate every N updates |
| `--validation-plan` | the frozen validation windows, `$PTP_DATA/validation_plan_v1.json` |
| `--train-eval-plan` | optional frozen training windows, evaluated alongside validation |
| `--proprioception` | add joint angles, joint velocities and gripper width at the current frame |
| `--lr`, `--warmup`, `--seed` | defaults 3e-4, 20, 2026; the full-split runs below used warmup 50 |
| `--resume last.pt` | continue a run (below) |
| `--layers` | replica layers, default 3 |
| `--checkpoint` / `--no-checkpoint` | gradient checkpointing, on by default |

The model settings (D=256, memory 24 slots, scene 24 frames, window 48 + 23 warm-up frames,
k=1) are fixed in the package; `docs/DECODER_BUILD.md` explains each choice.

## What a run folder contains

| file | contents |
|---|---|
| `config.json` | every setting of the run |
| `metrics.jsonl` | one line per logged training step and per evaluation |
| `best.pt` / `last.pt` | best checkpoint by validation loss / latest checkpoint |
| `summary.json` | final status and metrics |
| `evaluation/report.json` | held-out metrics and memory-shuffle tests (written by `evaluation/57_...`) |
| `experiment_summary.md`, `*.png` | human-readable report and learning curves |

## Resuming

`--resume last.pt` restores the model, optimizer, schedule, step and random state, so the
continuation matches an uninterrupted run. **`metrics.jsonl` is appended to**: if you resume
from a checkpoint older than the log's last line, first delete the lines after the checkpoint's
step, or the log will contain both versions.

## Reference runs

They ship with the dataset (`$PTP_DATA/policy_*`). All numbers are on the same 63,598 held-out
targets. The number in each folder name is the Slurm job ID. Several runs share `1658338`
because they ran inside one interactive allocation.

| run | job script | what | position error | scene effect |
|---|---|---|---:|---:|
| `policy_overfit_1657724` | `53_overfit_policy.sbatch` | wiring test: memorise 10 windows | passed at update 150 | — |
| `policy_pilot_1657753` | `58_pilot_policy.sbatch` | visual only, 1 epoch | 9.99 mm | +4.1% |
| `policy_proprio_1658371` | `74_proprio_pilot.sbatch` | + proprioception, 1 epoch | 7.65 mm | +2.1% |
| `policy_proprio_1658338` | same command, run interactively | an accidental replicate of the row above | 7.65 mm | +2.1% |
| `policy_multiepoch_1658338` | `67_...`, resumed by `77_...` | visual only, 5 epochs | 8.49 mm | +8.7% |
| `policy_proprio_5ep_1658584` | `79_proprio_5ep.sbatch` | + proprioception, 5 epochs | **6.49 mm** | +4.2% |

The bar is **copying the previous action: 4.55 mm**. No run beats it yet. The two
one-epoch proprioception runs agree to 0.002 mm, so run-to-run noise is far smaller than any
difference in this table.

`policy_cpu_smoke_*` are 2- and 10-update CPU tests of the training loop, not experiments.

`69_watch_wandb.py` optionally mirrors a run's `metrics.jsonl` to Weights & Biases without
touching the training process. It needs `wandb` installed and logged in.
