# Bounded five-epoch fitting experiment

The completed one-epoch checkpoint scores 10.030 mm on the 12,123 windows it saw and
9.991 mm on 1,326 validation windows. Copy-previous baselines are 4.612 and 4.551 mm.
Similar train/validation errors motivate testing a larger optimization budget before
attributing the problem to generalization.

- Fresh initialization with the same seed (2026), architecture, loss, physical-head LR
  multipliers, batch size 16, dropout 0.1, weight decay 1e-4 and BF16 as the pilot.
- Five epochs: 3,790 updates, 758 per epoch. One independently sampled window per clip
  each epoch, using the existing deterministic epoch/index sampler. T=48 and M=24.
- Base LR 3e-4, 50-step warmup, cosine decay across all 3,790 updates. This differs from
  the completed pilot's one-epoch cosine schedule; it is a new optimization-budget
  experiment, not an exact resume or a pure duration-only comparison.
- At update 0 and after every epoch, evaluate all 12,123 fixed epoch-0 training windows
  recovered by diagnostic 1658338 and all 1,326 frozen validation windows. Dropout is
  disabled for both. At update 0 the fixed training windows have not yet been seen.
- Report the same losses, physical metrics, state-based copy/persistence baselines,
  transition counts/costs and per-lab breakdowns on both splits. Compare within labs
  and against each split's baseline, since motion distributions can differ.
- Select best checkpoint only by weighted validation loss. Save last checkpoint with
  optimizer, schedule, data cursor and RNG for recovery within this same experiment.
- After training, run both frozen scene and robot-history interventions on the best
  checkpoint. The original pilot gates and limitations remain unchanged.
- Stop after five epochs. No automatic long-run extension. If both splits improve,
  further training remains a candidate; if both stall, investigate optimization/inputs;
  if training improves while validation stalls/worsens, investigate generalization.
  These patterns guide investigation rather than proving a unique cause.

Files: `training/67_run_multiepoch.sh`, `evaluation/68_report_multiepoch.py`, and immutable
`out/training_eval_plan_v1.json`. The user launches the GPU experiment; no job is submitted
by the preparation step. Approximate runtime is 3–4 hours on an L40S, including fixed
training-set evaluation, extrapolated from the pilot; other GPUs and I/O may differ.
