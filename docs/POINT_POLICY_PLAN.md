# Point-prediction policy — protocol fixed before training

Written 2026-09-30, before any code or run for this experiment exists, so the comparisons and
criteria cannot be chosen after seeing results. Decided with Ozan after the points → actions
experiment (`docs/POINTS_TO_ACTIONS.md`) and the ATM / Track2Act reading.

## The idea

Today's model maps the point memory straight to one action. The new model (ATM-style) first
predicts where each gripper point will move over the next 16 frames, then gives that prediction
to the action head as a hint. Predicting 16 frames ahead should require knowing where the object
is, which may make the scene memory matter. Because the action head is learned, the predicted
points do not need the ~1 px accuracy that geometric conversion needs.

## Model

| part | decision |
|---|---|
| Track-On, scene tokens, robot tokens | unchanged: frozen, cached |
| Scene memory (24 frames) | unchanged |
| Replica block (3 layers, action memory, scene cross-attention) | unchanged |
| Current coordinates | **new**: each robot point's Track-On **tracked** position (normalised) plus its **visibility flag**, given alongside its token. Tokens hold position only to ~25 px (`out/probe_position.json`) |
| Part 1: future-point head | **new**: per robot point, its **displacement** for each of the next 16 frames |
| Plan embedding | **new**: each point's predicted displacements → a 256-number vector, added to its token |
| Part 2: readout + action head | unchanged, except the readout sees the plan-enriched tokens |
| Proprioception | not used |

## Targets and losses

- **Targets:** FK displacement of each point: FK position at t+j minus FK position at t, for
  j = 1…16. Exact, and unaffected by tracking error. Point identities come from the FK index.
- **Masked** (left out of the loss): targets whose start or end position is outside the image (6.4% of point-frames in the
  validation windows), behind the camera, or past the end of the stored clip. A target frame must
  exist in the file; the validity flag alone is not enough.
- **Point loss:** mean over valid points and horizons. Each horizon j is divided by its own typical
  displacement, measured on training data only, so predicting no motion costs ≈ 1 at every horizon.
- **Action loss:** unchanged.
- **Total:** action loss + λ × point loss, with **λ = 1**. Any tuning uses validation data only.
- **The action head always receives part 1's prediction**, never the recorded future.

## Runs

All runs: 5 epochs in total, evaluated every epoch, same data, split, frozen validation plan,
batch size 16, pinned L40S. No proprioception.

| run | what | answers |
|---|---|---|
| A | today's visual-only model, 5 epochs | done: 8.49 mm, scene effect +8.74% |
| B | A + current tracked coordinates and visibility | do coordinates alone help? |
| C | A + action chunking: the next 16 actions per frame | does a long horizon, as actions, make the scene matter? |
| D | new model: 1 epoch pre-training part 1 alone, then 4 epochs of the full model with **part 1 still training** | the main run: does predicting future points help? |
| E | as D, but part 1 **frozen** after the pre-training epoch | keep training part 1, or freeze it? |

D and E share the same pre-training epoch.

| comparison | isolates |
|---|---|
| B vs A | knowing where the points are |
| C vs A | a long horizon, predicted as actions |
| D vs B | the future-point idea itself (D is B plus parts 1 and 2) |
| D vs C | long horizon as points vs as actions |
| D vs E | keep training vs freezing part 1 |

## Evaluation

Every run, on the same 63,598 validation targets:

- position error vs copy-last (4.55 mm) and run A (8.49 mm);
- rotation error; gripper timing cost;
- scene-shuffle effect, including the difficulty split, vs A (+8.74%) and the 10% gate;
- robot-history shuffle;
- learning curves per epoch.

Run C also scores each chunk step against momentum and no-motion **at that horizon**: momentum
becomes worse than no-motion beyond ~8 steps (`docs/POINTS_TO_ACTIONS.md`), so both are needed.

Run D additionally:

- **part 1's future-point error** at 1, 4, 8, 16 frames vs "points don't move" and "points keep
  their last motion";
- **perfect-plan upper bound**: D fed the recorded future at evaluation only;
- **geometry vs learned**: part 1's predictions converted with PnP (`droid_qp/pose_fit.py`) vs
  D's action head.

## Order of work

**One job at a time: nothing runs in parallel.** Code lives on the branch `point-policy-ozan` and is
merged into `main` only after its checks pass.

1. **No GPU:** point-prediction baselines; per-horizon loss scales (training data only).
2. **Build** current coordinates as an input, the future-point head, the plan embedding and
   two-stage training, each with checks in `checks/`.
3. **Overfit gate** for the new model: memorise 10 fixed windows, points and actions both, the
   same gate every model has passed before a real run.
4. **Pilot of D: 1 epoch in total**, split like the full run (about 150 updates pre-training
   part 1, then about 600 updates of the full model). It checks that the model *works*: losses
   fall, no NaNs, part 1 beats "points don't move", speed and memory are acceptable. It does
   **not** decide whether the model is better: one-epoch results have misled before (the scene
   effect was +4.1% after one epoch and +8.7% after five).
   **Stop conditions:** losses do not fall or become NaN, or part 1 is no better than "points don't
   move" at any horizon. Otherwise continue.
5. **Run D, 5 epochs** (1 + 4).
6. **Run E**: 4 epochs from D's pre-training checkpoint, part 1 frozen.
7. **Run B**: coordinates only, 5 epochs.
8. **Build** action chunking (future action targets, per-step evaluation), with checks and the
   overfit gate → **run C**, 5 epochs.

## Follow-up, decided now

If the best run is still improving at epoch 5, rerun it **and run A** for 10 epochs, so a gain
cannot be explained by extra training alone.
