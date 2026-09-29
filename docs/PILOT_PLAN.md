# Validation pilot — protocol fixed before training

Date: 2026-09-10. The 10-window overfit gate passed; this experiment tests generalization.

1. Freeze one uniformly sampled 48-frame window per validation clip (seed 202610),
   retaining every lab. Measure baselines on these exact windows before the pilot.
2. Train a freshly initialized policy for one complete pass through all 12,123 training
   clips: batch 16, 758 updates, T=48, M=24, 3 layers, D=256. Use BF16, dropout 0.1,
   AdamW weight decay 1e-4, base LR 3e-4, 50-step warm-up and cosine decay. Keep the
   physical-unit projection LR multipliers established in step 6. Save best by weighted
   validation loss only, including the initial checkpoint as a candidate.
3. Evaluate the chosen checkpoint and perform scene and robot-history interventions.
4. Report every aggregate with counts and per-lab breakdowns. Do not extend training
   automatically if the following criteria fail.
5. Integrate predicted deltas at fixed horizons and report the FK/projection proxy with
   its additional assumptions separately from one-step prediction.

## Decision rule

All three conditions must pass before considering a long run:

- Mean validation position error on the copy-previous-action comparable mask must be
  below BOTH 4.35 mm (the proposed external reference) and our measured copy-previous
  baseline on exactly the same mask. Missing previous actions are excluded from both.
- Cross-window scene shuffling must increase that validation position error by >10%.
  This is evidence of sensitivity to the intervention, not proof of semantic object use.
- Gripper transition timing cost must be strictly lower than current-state persistence.
  Match same-direction events one-to-one within +/-2 frames, maximizing match count then
  minimizing absolute timing error. Cost is `(sum(abs(error)) + 3*(misses + extras)) /
  number_of_true_events`. Report precision/recall/F1, timing MAE of matches, counts and
  opening/closing statistics as well. This prevents good timing on a few cherry-picked
  matched events from hiding missed or spurious transitions.

The machine-readable validation plan records baseline values, exact windows, donor
assignments, thresholds and manifest hash before training. These thresholds will not be
relaxed after seeing results. A failed pilot triggers investigation, not a long run.

## Baselines and interventions

Zero translation, identity rotation, repeat the preceding recorded body-frame action,
and repeat the current observed gripper state. Repeating body-frame action components is
the literal copy-command baseline; it is not transport of a world-frame velocity.

Scene shuffle: replace the entire object token stream with another validation episode's
stream, keeping robot tokens and labels intact. Donors have a complete history and are
matched within lab when possible; cross-lab donor counts are reported. Evaluate all
supervised frames and separately the final frame of each window.

Robot-history shuffle: use a donor's prior robot tokens mapped by stable FK point ID,
while preserving the recipient's current (last) frame and its scene. Score ONLY that
last frame. Donors have all 128 robot point IDs, avoiding dropped-point fabrication.
Compare clean, scene-shuffled and robot-history-shuffled predictions on the same anchors.
The frozen current-frame tokens already encode prior tracking context; this intervention
removes only explicit decoder input history, not all historical information.

Shuffled components retain real marginal token values but can violate joint scene/robot
compatibility. An unchanged score does not prove information is unused; a changed score
does not uniquely identify copying. Interpret the paired outcomes as diagnostic evidence.

## Limits

Every prediction uses observations of the demonstrator's recorded trajectory. The
policy's action memory is model-generated, but its visual inputs are not caused by its
own actions. One-step validation cannot measure policy-induced observation drift or
closed-loop compounding. Integrated-action errors can accumulate under recorded inputs,
but that remains an offline proxy, not a counterfactual robot execution.

EE pose alone does not determine a redundant arm's joint configuration. For projected
gripper-point rollout, use the verified gripper-only FK geometry and explicitly state
how finger opening is supplied. Never silently call oracle finger geometry a learned
gripper rollout. No simulator, robot interaction or synthesized future Track-On frames
are part of this pilot.

## Rollout reporting clarification (gates unchanged)

In addition to zero motion and holding the initial previous action fixed, report a
recorded-previous baseline that copies the preceding demonstrated action at each step.
It uses successive recorded observations, as the offline policy evaluation does. The
fixed-initial-action baseline has less information and is an extrapolation comparison.
This adds an offline diagnostic; it does not change any frozen pilot gate or window.
