# Building the action decoder — implementation steps

Everything upstream is finished and measured. This is the order to build in, and what to
check at each step before moving on.

## Status — updated 2026-09-10

    1 dataloader     DONE   droid_qp/policy_data.py       verified 45_verify_dataloader.py
    2 replica block  DONE   droid_qp/action_decoder.py    verified 46_verify_replica.py
    3 readout        DONE   droid_qp/action_decoder.py    verified 48_verify_readout.py
    4 head           DONE   droid_qp/action_decoder.py    verified 49_verify_head.py
    5 loss           DONE   droid_qp/policy_loss.py       verified 51_verify_loss.py
    6 training       DONE   52_train_policy.py; overfit gate passed at step 150
    7 evaluation     next   held-out/generalization checks; full-split training not run

Results and what each step actually cost are in `DECODER_LOG.md`.

**Settled by measurement, not preference:**

| | value | why |
|---|---|---|
| `T_win` / `M` | 48 / 24 | Track-On's own recipe (`config/train.yaml`) |
| `scene_slots` | **24** (full) | 24 slots cost only 0.72 s/batch more than 1 — nearly free |
| batch | **16** | 3.30 s/batch, 7.5 GB of 46 → ~42 min/epoch |
| GPU | **pin `lovelace_l40s`** | `avg` mixes in T4s; unpinned lands there ~half the time and is 10x slower |
| storage | keep `.npz` | 41 ms/sample cold is not the bottleneck; no `.npy` conversion needed |

**Two corrections these steps forced**, both listed in their step below:
the action memory now warms up over the history frames (step 2), so the warm-up loss
masking step 5 anticipated is **no longer needed on the action side**; and the readout's
input shape is fixed by what `ActionMemory` returns (step 3).

The bias toward verification is earned: today alone, three bugs shipped clean output and
were caught only by comparing against something independent — actions of 70 cm/frame that
only physics flagged, a gripper threshold only a cross-source diff exposed, and a
`birth` control that revealed my own refactor had changed the baseline. Assume the same
here. A decoder that trains without crashing tells you almost nothing.

## What already exists

    out/gen/<shard>/trackon/tracks/<vid>.npz
        tokens      (T, N, 256) fp16    the scene+action memory content
        role        (N,) int8           0 = object, 1 = robot
        tracks      (T, N, 2)           where each point went
        visibility  (T, N) bool
        q_init      (N, 256) fp16       birth anchor, kept OUTSIDE the memory

    out/gen/<shard>/actions/labels/<vid>.npz
        delta_pos_body (T,3)  delta_rot6d_body (T,6)
        gripper_open_next (T,1)         next observed binary state, a command proxy
        valid (T,)                      99.6% true

    out/gen/<shard>/merged/query_points/<vid>_f0.npz
        fk_uv (T,128,2)  fk_ok (T,128)  point_index (n_rob,)
        exact gripper positions -- the evaluation ground truth

    out/dataset_manifest.json
        13,449 clips, per-clip split + weight, 0 episode overlap

---

## Step 1 — dataloader

The one place where a silent mistake is unrecoverable, so build it first and test it
hardest.

Per sample: pick a clip, pick a timestep `t`, return

    scene_mem  (M, 100, 256)     tokens[t-M+1 : t+1, role==0]
    robot_mem  (M, 128, 256)     tokens[t-M+1 : t+1, role==1], padded
    robot_pad  (128,) bool       which robot slots are real
    target     delta_pos(3) + delta_rot6d(6) + gripper(1)  at t
    valid      bool              from the action labels
    weight     float             from the manifest

`M = 24` during training, matching Track-On's own recipe (`config/train.yaml`), extended to
72 at inference via `memory_extension`.

Details that will bite:

- **Object N is exactly 100** (k-means k=100). **Robot N varies** clip to clip, so pad to
  128 and carry a mask. Never let padding reach attention unmasked.
- **Early frames have a short memory.** At `t < M-1` there are fewer than M slots. Pad at
  the FRONT and mask, so slot ordering stays oldest→newest as Track-On writes it.
- **Slice `[:t+1]` and nothing beyond.** This is what makes training causal. A dataloader
  that accidentally reads `t+1` will train a model that looks like it works and cannot be
  deployed.

**Verify before continuing:**
1. sample 1,000 windows and assert no index above `t` is ever touched
2. assert `scene_mem[-1]` equals `tokens[t, role==0]` exactly
3. assert padded robot slots are all zero and the mask marks them
4. assert every returned target has `valid == True`
5. take one clip, reconstruct the full token stream from consecutive windows, compare to
   the npz — they must be identical

## Step 2 — the replica block

Fatma hoca's design: a trainable copy of Track-On's transformer block, operating on robot
tokens, with one attention added.

    q = robot tokens at frame t                    (B, 128, 256)
      -> self-attention across robot points        the gripper as one body
      -> memory attention over the ACTION fifo     its own recent history, M=24
      -> cross-attention into the SCENE memory     what the objects are doing   [NEW]
      -> write the output into the action fifo

Ordering: memory attention BEFORE the scene cross-attention. Cross-attention is a
query-formation step — a token that already knows its own trajectory asks the scene a
better question than one that only knows the current frame. The consequential part is that
what gets WRITTEN to the FIFO is then scene-aware, so next frame's memory attention sees
history that already contains object information. Ablate the other order; it is two lines.

**This makes training recurrent.** The block writes its own output into its own FIFO, so
the window must be rolled forward frame by frame and cannot be parallelised over `t`.
Track-On has the same property and handles it with T=48 windows — use that.

**Verify:** with the scene cross-attention zeroed, the block must reduce to something
stable; and a single-frame forward must not depend on any future frame (perturb `t+1`
inputs, assert the output at `t` is bit-identical).

## Step 3 — readout: ~120 tokens to one action  — DONE

`ActionMemory` returns `(B, T_win, N_rob, D)` — one token per robot point per supervised
frame. A policy emits **one** action per frame, so something must reduce N to 1.

Use `k` learned action queries that cross-attend over the robot tokens (DETR / ACT style):
`self.action_queries = nn.Embedding(k, D)`. Start at `k=1`. Larger k provides output slots
for action chunking, but future-action semantics also require aligned future targets,
validity masks, and the output head; extra queries alone do not implement chunking.

Concrete interface:

    in   robot_out (B, T_win, N_rob, D)   from ActionMemory
         robot_pad (B, N_rob) bool        True = padded, must be masked as a key
    out  (B, T_win, k, D)                 k action tokens per frame

Fold `T_win` into the batch — the readout is per-frame and carries no state of its own, so
unlike step 2 it needs no loop and is cheap.

`ActionReadout(D=256, heads=4, k=1)` lives beside `ActionMemory` in
`droid_qp/action_decoder.py` and reuses Track-On's `MHA_Block`. It returns features, not
position/rotation/gripper predictions; that is step 4. At least one real robot point is
required per example. Padded values are zeroed before projection and masked as keys.

**Mask boundary:** `PolicyWindows` currently returns `robot_pad=True` for REAL points,
despite its name. Invert that mask once before passing it to both modules:

```python
point_mask = ~batch["robot_pad"]  # boolean tensor; now True = padding
robot_out = memory(robot, scene, hist_valid, point_mask, T_win)
action_tokens = readout(robot_out, point_mask)
```

**Verify:** permuting the robot points must leave the output *unchanged* (not permuted —
this is the one place that must be permutation-INVARIANT rather than equivariant, since
point order is an artifact of how the query tensor was assembled). Padded slots must change
nothing. With `k>1`, verify distinct outputs at initialization and gradients to each query;
this is not a guarantee against collapse during future training.

## Step 4 — head  — DONE

    delta position   (3)   metres, body frame
    delta rotation   (6)   6D continuous, NOT quaternion (double cover)
    gripper          (1)   logit for the next observed open/closed state

Position is metres, rotation is unitless, gripper is binary. Use shared group scales
computed on the train split only. Do not independently standardize the six rotation
coordinates: the near-constant identity entries have tiny standard deviations. Decode
the 6D output to a valid rotation matrix for the initial rotation loss.

`ActionHead(D=256)` in `droid_qp/action_decoder.py` applies three linear projections to
each readout token. It accepts `(B, T_win, k, D)` and preserves all three leading axes,
including the singleton k axis when `k=1`. It returns a dictionary:

| key | shape | meaning |
|---|---|---|
| `delta_pos_body` | `(B, T_win, k, 3)` | body-frame displacement in metres |
| `delta_rot6d_body` | `(B, T_win, k, 6)` | raw first two rotation columns, no standardization |
| `delta_rotmat_body` | `(B, T_win, k, 3, 3)` | differentiably decoded rotation for loss/evaluation |
| `gripper_logit` | `(B, T_win, k, 1)` | logit for next observed open state |

```python
predictions = head(action_tokens)
# For k=1, align the loader targets (B, T_win, 10) with a singleton query axis.
# For k>1, step 5 needs future targets; do not broadcast the same target over queries.
```

**Initialization:** position/rotation projection weights have standard deviation
`1e-3 / sqrt(D)`; gripper weights use `1e-2 / sqrt(D)`. Biases are zero except rotation,
which starts at `[1,0,0,0,1,0]`. This gives near-zero motion and near-identity rotations
without blocking initial gradients to the readout. These are initialization scales, not
output bounds or dataset normalization constants. Shared training-data scales belong to
the step-5 loss; the head always returns translation in physical units.

`rotation_6d_to_matrix` follows the column order in `actions.rot_to_6d`. It uses
Gram-Schmidt with re-orthogonalization, promoting half/bfloat16 inputs to float32 during
decoding. Zero first vectors use +x, and collinear second vectors use a perpendicular
coordinate-axis fallback. Those exceptional fallbacks are not smooth at their boundaries;
the identity initialization avoids them in the normal initial forward pass.

Gripper logits stay unrestricted for BCE-with-logits. Apply sigmoid only for probabilities.
The target is a thresholded observed width at t+1, not a directly recorded control command.

## Step 5 — loss  — DONE

Initial baseline agreed with Ozan on 2026-09-10: first establish whether the model learns.

    per_frame = scaled translation squared error
              + scaled rotation-matrix squared distance
              + binary cross-entropy with logits for gripper_open_next
    total = sum(per_frame * valid * clip_weight) / sum(valid * clip_weight)

`PolicyLoss` in `droid_qp/policy_loss.py` uses the following concrete per-target terms:

    position = ||predicted_dp - target_dp||² / position_scale_m²
    rotation = 0.5 * ||predicted_R - target_R||_F² / rotation_scale²
    gripper  = BCEWithLogits(predicted_gripper_logit, target_open_next)

All three coefficients start at 1. The target rotation is decoded from the saved 6D
columns using the same convention as the head. There is no loss on frozen Track-On or
separate supervision target for action memory.

**Scales:** `scripts/50_compute_loss_scales.py` streams only training action labels,
using valid frames and manifest clip weights. It computes

    position_scale_m² = weighted_mean(||target_dp||²)
    rotation_scale²   = weighted_mean(0.5 * ||target_R - I||_F²)

Thus the zero-translation and identity-rotation baselines each have mean component loss
1 on the weighted training-frame population. This is group RMS scaling, not per-dimension
standardization or mean subtraction. The clip/window sampler has a different distribution,
so individual minibatches need not start at 1. Both scales must be finite and positive;
a corpus with zero motion energy requires an explicit alternate scale.

Saved artifact: `out/policy_loss_scales.json`. Measured over **12,123 training clips,
9,441 episodes, 1,924,243 valid frames**, with summed frame weight 1,529,894.1839:

| scale | value |
|---|---|
| translation | **0.0166820011 m** (1.6682 cm displacement norm) |
| rotation | **0.0456758645**, RMS chordal distance / sqrt(2), approximately 2.617° for small rotations |

The artifact records the manifest hash, the hash of used continuous labels/validity/weights,
paths, counts, formulas, and creation time. Scale loading can verify the current manifest
hash. The recorded label hash is provenance; loading does not rescan every label file.
Scales are module buffers, so include the loss state in training checkpoints.

**Interface and reduction:** the loss accepts the `ActionHead` output dictionary, loader
targets `(B,T,10)`, validity `(B,T)` and weights `(B,)` for k=1. For k>1, require explicit
future-aligned targets `(B,T,k,10)` and validity `(B,T,k)`; never broadcast one target over
all queries. It returns `loss`, `position`, `rotation`, `gripper`, `weight_sum`, `n_valid`.
The last count means valid targets with positive clip weight.

Invalid and zero-weight entries are selected out before geometry, squared error, or BCE.
Active nonfinite values, nonbinary gripper labels, negative/nonfinite weights, and shape
mismatches raise errors. The reduction runs in float32 under mixed precision.

```python
criterion = PolicyLoss.from_json(
    OUT / "policy_loss_scales.json", manifest=OUT / "dataset_manifest.json"
).to(device)
terms = criterion(predictions, batch["target"], batch["target_valid"], batch["weight"])
if terms["weight_sum"].item() > 0:
    terms["loss"].backward()
    # step 6 will own optimizer stepping, gradient clipping and logging
```

An empty/zero-weight batch returns differentiable zero losses. **Skip its optimizer step**
as well: AdamW weight decay could otherwise update parameters with no supervision.

- **mask by `valid`** — supervise only label pairs with a valid successor; the labels now
  come directly from DROID, rather than PointWorld's window coverage
- **weight by the clip's manifest weight** — a noisy clip teaches less
- **No extra warm-up loss mask.** Step 2 processes available history before collecting
  supervised outputs. Near clip starts, memories may still be short; `hist_valid` excludes
  front-padding inside the memory, while real early actions remain supervised.

**Later comparison:** regression can average incompatible actions if multiple modes remain
after conditioning on the memories. That is a hypothesis to evaluate here, not a measured
failure of this decoder. Whole-action classification plus residuals or diffusion are later
options requiring head/training changes. MSE/Huber or rotation-loss changes can preserve
the head. Report fixed physical-unit metrics when comparing different training losses.

## Step 6 — training  — OVERFIT CHECK PASSED

AdamW, cosine schedule, gradient clipping. Track-On stays FROZEN — the cached token stream
is only valid because of that, and fine-tuning it invalidates 315 GB instantly.

**Verify before a long run:** overfit 10 clips to near-zero loss. If it cannot memorise ten
examples, the architecture or the wiring is broken and no amount of data will fix it.

**Implemented:** `droid_qp/policy_train.py` composes memory/readout/head, wraps the cached
windows, performs AdamW steps, and reports loss components plus physical action metrics.
`scripts/52_train_policy.py` provides overfit and full-split modes, cosine scheduling,
gradient clipping at 1, BF16 autocast, checkpoints/resume, and JSONL metrics. Empty batches
skip the optimizer and scheduler. No frozen tracker is instantiated or updated.

The initial shared optimizer rate overshot the physical-unit translation projection in a
CPU smoke run. The optimizer now uses the base learning rate for memory/readout/gripper,
base rate times `position_scale_m` for the position projection, and base rate times
`rotation_scale` for the rotation projection. Output units and loss definitions are unchanged.

**Actual gate run:** job **1657724**, L40S, 10 fixed 48-frame windows from 10 distinct
training episodes (480 targets, 14 gripper-change frames). Selected windows include motion
and gripper changes, with clip weights >=0.75. This is not memorization of all frames in
those ten videos. Dropout and weight decay were disabled for this diagnostic; the full
training mode uses dropout 0.1 and weight decay 1e-4.

At D=256, 3 replica layers, M=24, scene_slots=24, B=10, BF16 and checkpointing off,
the run stopped automatically at **150 updates** after passing the predefined gate:
position/rotation/gripper component losses each <0.02, total <0.05, overall gripper
accuracy >=99.5%, and transition-frame accuracy 100%. Measured training/evaluation time
was **132.66 seconds**, with **24.53 GB** peak allocated GPU memory.

| metric on the fixed windows | initial | step 150 |
|---|---|---|
| total loss | 2.26694 | **0.018774** |
| mean position error | 1.13056 cm | **0.103071 cm** |
| mean rotation error | 1.66864 degrees | **0.195020 degrees** |
| gripper accuracy | 52.29% | **100%** |
| correct next state on gripper-change frames | 21.43% | **100% (14/14)** |

The zero-motion baselines are 1.03908 cm and 1.65254 degrees; copying the current observed
gripper state gives 97.08% overall accuracy and 0% at change frames. This establishes that
the assembled model can fit these examples, not generalization or closed-loop success.

Artifacts: `out/policy_overfit_1657724/{config.json,metrics.jsonl,summary.json,best.pt,last.pt,
learning_curve.png}`. Checkpoints include model, loss scales, optimizer, scheduler, next
batch position and torch CPU/CUDA RNG states. `scripts/54_verify_training.py` passes six
CPU checks including identical next updates after checkpoint restoration.

**Why 48:** this follows Track-On's training recipe (`track_on/config/train.yaml`), not a
measured optimum for this policy. Each sample processes 23 preceding warm-up frames plus
48 supervised frames (71 total, front-padded near episode starts). Fixed lengths bound
recurrent backpropagation cost and batch padding. Full-split mode samples a new window per
clip/epoch instead of keeping the fixed overfit windows. This can cover video content over
training, but windows restart action memory and do not train full-episode dependencies.
Longer unrolls or carrying/detaching memory between consecutive chunks are later options.

Full-split pilot **1657753** and rollout/report job **1657754** completed. The
predeclared pilot gates failed; see step 7 and the completed report. This is one sampled window per training clip, not a pass
through every frame of every video.

## Step 7 — held-out evaluation and pilot — COMPLETE; PILOT GATES FAILED

The immutable `out/validation_plan_v1.json` freezes 1,326 validation windows from held-out
episodes, exact starts, donor assignments, baselines, manifest hash and decision thresholds.
There are 63,598 valid targets and 63,556 targets with a valid previous action. Validation
includes all 12 labs, including short clips with padded invalid supervision.

**Baselines measured before training:** whole-validation copy-previous position error is
4.33767 mm versus zero motion 12.29696 mm. On the fixed pilot windows, the same-mask
numbers are 4.55066 mm and 13.02935 mm. Literal body-frame action copying is reported for
translation and rotation; observed-current-state persistence is the gripper baseline.

**Pilot:** fresh initialization, one sampled window per each of 12,123 train clips, 758
updates at B=16, D=256, three layers, M=24, T=48, dropout 0.1, AdamW decay 1e-4, BF16,
gradient checkpointing, LR 3e-4 with the step-6 physical-head multipliers, 50-step warmup
and cosine decay. Best checkpoint is selected only by weighted validation loss, including
initialization as a candidate. Validation runs at updates 0, 190, 380, 570 and 758.

**Frozen gates, all required:** same-mask position error <4.35 mm; scene shuffling increases
that error by >10%; gripper event timing cost beats persistence. Match same-direction
events within +/-2 frames, maximize matches then minimize timing error, and penalize each
missed/extra event by 3 frames. Report event counts, precision/recall/F1, matched timing and
the penalized cost. Accuracy is retained as a secondary diagnostic, never a success gate.

**Paired interventions:** scene tokens come from another validation episode. Robot history
comes from a donor aligned by stable FK point IDs, preserving the current final frame;
score only that anchor and compare clean/scene/robot interventions on the same anchors.
Donors have all 128 robot points and full history; 80 recipients need cross-lab donors.
Shuffles can violate joint scene/robot compatibility. The current frozen robot tokens also
retain tracking context, so these tests do not erase all historical information or uniquely
prove object use/copying. Every metric and baseline has a per-lab breakdown.

**Offline trajectory proxy:** compose predicted body-frame EE deltas from measured initial
poses over 1, 5, 10, 20 and 40 frames; 48-frame prediction windows cannot support horizon
50. Compare flange pose and projected gripper FK point error against the recorded targets.
One projection uses oracle future finger articulation to isolate pose error. A second snaps
predicted binary gripper state to open/closed URDF endpoints, ignoring actuator dynamics.
Baseline rollouts copy the preceding recorded action at each step (matching the offline
observation conditioning), hold the action known just before each origin fixed, or hold zero
motion. Holding the initial action has less information than the recorded-observation policy.
All methods use the same origins with a known preceding action. Predicted off-screen points
remain in pixel errors; behind-camera failures are counted separately. FK reconstruction
and ground-truth delta composition are checked before reporting each clip.

These are **recorded-observation predictions**, not closed-loop robot behavior. Even the
integrated trajectory still receives predictions conditioned on the demonstrator's real
frames; it cannot measure policy-induced observation drift. EE pose cannot determine a
redundant arm's joint configuration, so the FK proxy uses only gripper geometry and makes
no claim of full-arm IK or task success. No simulator or robot execution is available.

Implementation: `policy_eval.py`, `policy_rollout.py`, scripts `56_prepare_validation.py`,
`57_evaluate_policy.py`, `58_pilot_policy.sbatch`, `60_evaluate_rollout.py` and
`62_report_pilot.py`. Checks: `59_verify_validation.py` (4 passed),
`61_verify_rollout.py` (2 passed, including a real-clip oracle reconstruction).


### Completed results (reviewed 2026-09-11)

All jobs finished; the selected checkpoint is update 758. The 1,326 held-out windows
represent 1,048 episodes. Weighted validation loss improved from 2.84579 to 1.73949,
but **all three predeclared gates failed**:

| Gate | Result | Required |
|---|---:|---:|
| position error | 9.991 mm | <4.350 mm |
| scene-shuffle position increase | 4.13% | >10% |
| gripper event timing cost | 5.773 frames | <1.056 frames |

Copy-previous error is 4.551 mm on identical targets and beats the policy in all 12 labs.
On 1,304 matched last-frame anchors, clean position error is 10.606 mm, scene-shuffled
10.728 mm, and robot-history-shuffled 23.679 mm. This shows much stronger sensitivity to
explicit robot history than to scene shuffling; it does not uniquely prove copying.
There are 568 missed and 1,889 extra gripper transitions (F1 0.413).

The offline integrated policy trajectory reaches 114.284 mm error at 20 frames and
166.553 mm at 40. Copying each preceding recorded action gives 21.919 and 22.972 mm.
Projected gripper errors with oracle finger width are 50.640 and 73.182 px respectively.
These remain recorded-observation proxies, not closed-loop execution. Geometry QA passed
all 1,326 clips: max saved-FK disagreement 0.0000341 px, max GT composition error
0.000524 mm. Rollouts exclude 22 additional targets lacking a stored successor pose.

No extended run was launched. Before another pilot, compare performance on fixed training
and held-out windows, inspect missed/spurious gripper events, and diagnose why scene
intervention has little effect. A single epoch is insufficient to establish the eventual
capacity of the architecture; the failed gate is a decision against scaling this recipe
without investigation, not proof that longer training can never help.

Report and plots: `out/policy_pilot_1657753/PILOT_REPORT.md`; detailed per-lab metrics in
`evaluation/report.json` and `evaluation/rollout.json` beneath that run directory.

---

## Order, and what blocks what

    1 dataloader      <- start here, verify hardest
    2 replica block   <- needs 1
    3 readout         <- needs 2
    4 head            <- needs 3
    5 loss            <- needs 4; initial regression baseline agreed
    6 training        <- overfit 10 clips before any long run
    7 evaluation      <- can be written in parallel with 2-5

The initial loss choice is settled for the learning baseline; later comparisons remain open.


## Next bounded fitting experiment — prepared

The completed fixed-window comparison found training/validation errors of 10.030/9.991 mm,
with similar deficits to their own copy baselines. A fresh five-epoch optimization-budget
experiment is prepared in `MULTIEPOCH_PLAN.md` and `scripts/67_run_multiepoch.sh`. It adds
fixed training-set evaluation to the existing trainer, preserves validation-only checkpoint
selection, and repeats both memory ablations after training. The user launches the run;
no results from that experiment are available yet.


## One-epoch proprioception variant — implemented, awaiting user launch

See `PROPRIO_PILOT_PLAN.md`. The optional current-state encoder adds q, recorded qdot and
gripper closure at readout, with the original one-epoch pilot as the comparator. Script
`73_run_proprio_pilot.sh` runs training, full fitting comparison and ablations, writing
`proprio_comparison.md`/`.png`. New and regression checks passed; no GPU results exist yet.
