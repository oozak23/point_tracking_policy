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
- **Masked** (left out of the loss): targets outside the image (6.4% of point-frames in the
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

1. No GPU: point-prediction baselines; per-horizon loss scales (training data only).
2. Current coordinates as an input (needed by B, D, E).
3. Run B.
4. Future-point head, plan embedding, two-stage training → runs D and E.
5. Action chunking: future action targets and per-step evaluation → run C.

Every new component gets checks in `checks/` and must pass the overfit gate before a full run.

## Follow-up, decided now

If the best run is still improving at epoch 5, rerun it **and run A** for 10 epochs, so a gain
cannot be explained by extra training alone.
