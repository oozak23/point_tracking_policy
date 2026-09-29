# One-epoch current-proprioception pilot

Prepare and run alongside the existing visual-only five-epoch experiment. The comparator
is the original one-epoch pilot 1657753, not epoch 1 of the five-epoch run: the latter has
a different cosine schedule. The user launches the GPU experiment.

## Single variant

At prediction frame t, input the seven recorded joint positions q[t], seven recorded
joint velocities qdot[t], and current gripper position. Targets remain observed body-frame
EE displacement/rotation t->t+1 and next observed binary gripper state. No future states,
target deltas, previous-action inputs, finite differences, or recorded commands are added.

The label pipeline directly subsamples recorded DROID joint velocities; the variant does
not estimate velocity from future poses. The current-state array is aligned using the
same supervised start and real-frame mask as the visual inputs. Missing/nonfinite active
states fail explicitly. Padded state features are masked.

Use fixed physical scales: q/pi, qdot/(1 rad/s), and gripper_position/0.725. The last value
recovers recorded normalized **closure**, with 0=open and 1=closed; it is not a metric finger
width. Values are not clipped or standardized using held-out statistics.

A 15->256->256 GELU MLP encodes state and adds its output to the visual action-readout
feature immediately before the unchanged action heads. ActionMemory and visual attention
remain structurally unchanged. The state encoder adds 69,888 parameters. Its last layer
starts at zero, preserving the initial visual-only predictions. Its construction preserves
the shared initialization and global CPU RNG state; it adds no dropout operations.

## Matched protocol

- Fresh seed 2026; D=256, three layers, M=24, T=48, batch 16, BF16, gradient checkpointing.
- One epoch, 758 updates, LR 3e-4, warmup 50, cosine decay over 758 updates. Same AdamW
  physical-head LR multipliers, loss/scales, dropout 0.1 and weight decay 1e-4.
- Identical epoch-0 clip/window sampling and validation plan. Validate at 0/190/380/570/758.
- Save best by validation loss. Evaluate the completed last checkpoint on all exact seen
  training and validation windows, so the epoch-end fitting comparison is explicit.
- Evaluate the best checkpoint with the original paired memory interventions and gates.
  Keep proprioception unchanged when shuffling scene or robot visual history. Add a
  separate donor-state shuffle as a sensitivity diagnostic. Shuffles break joint input
  compatibility and do not uniquely establish causal feature use.
- Report epoch-end train/validation comparison, best-checkpoint comparison, all baselines,
  gripper event costs, per-lab results and paired anchor errors. No automatic longer run.

## Interpretation and limits

An improvement would support adding state for this recipe/budget. A failure would not
rule out other fusion designs or longer training. There is no matched-capacity control,
and this is one seed. Current proprioception can improve motion prediction while further
reducing scene dependence, so inspect both prediction errors and interventions.

All evaluations still condition on recorded trajectories; this is not robot execution.
The existing visual-only process is left running with its already-loaded implementation.
New script options default to visual-only for old checkpoints and future evaluations.

## Files and checks

- `droid_qp/policy_proprio.py`: current-state loader and late-fusion policy.
- `scripts/70_verify_proprio.py`: alignment, future-state isolation, padding, shared
  initialization/RNG, encoder gradients and checkpoint roundtrip, real short-clip checks.
- `out/proprio_data_audit.json`: all 13,449 clips have complete finite sensor fields.
- `scripts/73_run_proprio_pilot.sh`: one epoch, epoch-end fit comparison, ablations, report.
- `scripts/72_report_proprio.py`: verifies matching recipes and exact evaluated windows
  before comparing against the visual-only pilot and diagnostic 1658338.

Results will be in the chosen output directory as `proprio_comparison.md` and
`proprio_comparison.png`. W&B monitor 69 also supports this run and records the variant flag.
