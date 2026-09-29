# Action decoder — build log

One entry per step of `DECODER_BUILD.md`: what was built, what was checked, what the
numbers were, and anything decided along the way. Written as the work happens so the
reasoning survives, not reconstructed afterwards.

---

## Step 1 — dataloader ✅ 2026-09-09

**Built:** `droid_qp/policy_data.py` (`PolicyWindows`), verified by
`scripts/45_verify_dataloader.py`.

### What a sample is

Not one timestep. The action memory is recurrent — the replica block writes its own output
into its own FIFO — so training has to roll forward over a window, as Track-On does
(`config/train.yaml`: T=48, M=24). A sample is a contiguous slice of the token **stream**,
and the model maintains its own memory across it:

    scene (71, 100, 256)   object points, frozen, read-only
    robot (71, 128, 256)   robot points, input to the trainable replica
    target (48, 10)        delta_pos(3) + delta_rot6d(6) + gripper(1)

71 = (M−1) + T_win. Returning the stream rather than a stack of per-frame memories matters:
the memory at frame *t* is `stream[t−M+1 : t+1]`, and consecutive frames overlap by 23, so
materialising them separately would multiply the data **24×** for no extra information.

### Facts that shaped the design

Measured over 200 clips before writing anything:

| | |
|---|---|
| `N_obj` | **exactly 100** in every clip (k-means k=100) |
| `N_rob` | **42–128**, varies per clip — the SAM filter rejects a different number each time |
| `T` | min 48, median 125, max 1057 |
| clips shorter than 71 frames | 25 of 200 (12.5%) |

Two consequences. Robot points are **padded to 128 with a mask**, since their count is not
fixed. And because 12.5% of clips are shorter than (M−1)+T_win, history is **front-padded
and masked** — front, not back, so slot order stays oldest→newest, which is the order
Track-On writes the FIFO and the order `t_embedding` expects. The minimum T is 48, exactly
`T_win`, so no clip has to be dropped.

### Verification — all five passed

| check | result |
|---|---|
| window equals the raw stream slice, **re-read from disk** | 0/300 mismatched |
| no frame at or beyond `s+T_win` appears in the window | 0/300 leaked |
| `scene[M−1] == tokens[s]` | 0/200 |
| padded robot slots zero and masked | 0/300 |
| front-padded history zero and masked | 0 bad of 122 early-start windows |
| `target[j] == labels[s+j]` | 0/200 |
| translation magnitudes physically possible | max **6.93 cm/frame** (~0.5 m/s at 7.2 Hz) |
| train/val episodes disjoint | 0 shared |

The first check is the one that matters, and it is deliberately verified **against the file
rather than against the loader's own arithmetic** — a wrong offset computed the same way
twice would verify itself. A causality bug here produces a model that trains beautifully
and cannot be deployed, with nothing crashing to say so.

The physical-plausibility check is there because that exact class of bug already bit once
in this project: action labels computed across PointWorld coverage gaps read as 70 cm/frame,
over 5 m/s at the end effector, and no summary statistic looked wrong. Only the physics
caught it. 6.93 cm/frame is a fast but real teleoperated motion.

### Throughput — no optimisation needed

    41 ms/sample cold, 4.2 ms warm, 8.3 MB per sample

A clip's npz is 58 MB and numpy cannot slice one without reading the whole member, so the
worry was that IO would dominate. It does not: with 8 dataloader workers this is ~195
samples/s, and a full pass over 12,123 clips costs ~8 minutes of IO. The GPU step will be
slower than that.

**Decision: do not convert the 234 GB to memory-mappable `.npy`.** It would make cold reads
cheap but costs a full rewrite of the dataset, and the measurement says the bottleneck is
elsewhere. Revisit only if profiling during training says otherwise — a small LRU cache
(8 clips) already absorbs most of it.

### Interfaces the next steps rely on

    scene       (71, 100, 256) fp16     memory at frame t = scene[t-23 : t+1]
    robot       (71, 128, 256) fp16     same indexing
    hist_valid  (71,) bool              False where front-padded
    robot_pad   (128,) bool             True where a real point — never let padding
                                        reach attention unmasked
    target      (48, 10) fp32
    target_valid(48,) bool              PointWorld coverage; 99.6% true
    weight      scalar                  per-clip, from the manifest

`PolicyWindows` has **no torch dependency**, so it can be verified in any of the three
conda envs. Wrap it in a `Dataset` where torch is available.

**Next:** step 2, the replica block.

---

## Step 2 — replica block ✅ 2026-09-09

**Built:** `droid_qp/action_decoder.py` (`ReplicaLayer`, `ActionMemory`), verified by
`scripts/46_verify_replica.py`, benchmarked by `scripts/47_bench_replica.sbatch`.

### The block

    q = robot tokens at frame t              (B, 128, 256)
      -> self-attention across robot points
      -> memory attention over the ACTION fifo, per point
      -> cross-attention into the SCENE memory                  [NEW]
      -> written into the action fifo

`MHA_Block` is **imported from Track-On**, not reimplemented, so attention, MLP ratio and
norm placement are identical by construction rather than by inspection.

Track-On's block opens with `feature_attention` over the whole frame feature map because it
does not know where the point is and must search. We do know — FK gives exact pixel
positions — so the replica has no feature attention and needs no frame features at training
time. That is what makes the cached token stream sufficient and the 5.3 PB of feature maps
unnecessary.

### A design flaw the verification caught

The check *"changing the past must move a later frame"* **failed** — max delta exactly
0.000e+00. The memory was not live.

Not a test bug. The block was starting with an **empty FIFO at the first supervised frame**,
so the robot history the dataloader already loads was never read on the action side; it was
feeding only the scene. Fixed by rolling over the whole stream and collecting output only
for the supervised part. Every supervised frame now has a **full** memory, and the warm-up
loss masking `DECODER_BUILD.md` step 5 anticipated is no longer needed on the action side.

This is the second time in two days that running a control caught a defect that reading the
code did not — the first was `mask-frame birth`, where reproducing the baseline revealed a
refactor had shifted it.

### Verification — nine checks, all by perturbation

| check | result |
|---|---|
| scrambling every frame after *t* leaves output at *t* bit-identical | 0/6 frames changed |
| changing the past moves a later frame (memory is live) | delta 0.755 |
| scrambling padded robot slots changes real points | **exactly 0** |
| scrambling masked history changes anything | **exactly 0** |
| changing the scene changes the output | delta 0.423 |
| permuting points permutes the output identically | delta 1.4e-06 |
| deterministic in eval | yes |
| every parameter receives non-zero gradient | 0 of 0 missing |

Perturbation rather than inspection: change an input that should not matter and require
bit-identity. Reading code and agreeing with it is not evidence.

### Performance — one equivalence, one hardware lesson

First benchmark: **17.55 s/batch at B=8**, which is ~7 hours per epoch. Two causes.

**The memory attention was doing 25x redundant work.** It passed all M+1 = 25 slots through
`MHA_Block` as queries but only ever reads the last one — and everything after the attention
in that block (residual, both LayerNorms, the 4x MLP) is **position-wise**. So computing one
query position is bit-identical to computing 25 and discarding 24. Confirmed empirically:
the "changing the past" delta is 0.7546 before and after, unchanged to seven digits.

**The job landed on a Tesla T4.** `avg` contains T4 nodes as well as L40S/A6000, and an
unpinned request takes whatever is free. 17.6 s/batch reads as a broken model rather than a
slow card. The sbatch now pins `--gres=gpu:lovelace_l40s:1`.

L40S, after the fix:

| scene_slots | B=8 | B=16 | peak GB (B=16) |
|---|---|---|---|
| **24** | 2.09 s | **3.30 s** | 7.5 |
| 8 | 1.55 s | 2.58 s | 5.9 |
| 1 | 1.61 s | 2.42 s | 5.2 |

**Chosen: `scene_slots=24`, B=16 -> 757 batches/epoch, ~42 min/epoch, 6.9 h for 10 epochs.**

The scene cross-attention is nearly free: the full 24-slot memory costs 0.72 s/batch more
than a single slot, and slots=1 is not even faster than slots=8. So there is no reason to
truncate the scene context. The cost lives in the 71-step sequential rollout, which is
inherent to a recurrent memory.

7.5 GB of 46 means large headroom. Gradient checkpointing is currently ON, costing roughly
30% compute for memory we now demonstrably do not need — worth testing OFF at B=16 as a
free speedup, but not blocking.

**Next:** step 3, the readout — turning ~120 robot tokens into one action.

---

## Step 3 — learned-query readout ✅ 2026-09-10

**Built:** `ActionReadout` in `droid_qp/action_decoder.py`, verified by
`scripts/48_verify_readout.py`. Default `D=256, heads=4, k=1`.

    ActionMemory output (B, T_win, N_rob, D)
      -> k learned queries cross-attend over robot points, with padding masked
      -> action tokens (B, T_win, k, D)

Uses Track-On's existing `MHA_Block`, including its residuals, norms and FFN. Time is
folded into the batch; the readout has no recurrent state or point-index embeddings.
Larger k produces distinct query slots at initialization, but predicting future actions
still needs corresponding targets and the step-4 head. No policy head or loss was built.

**Mask convention:** the loader's `robot_pad=True` means REAL, whereas both attention
modules take True = PADDED. Invert the loader mask once for both modules. The readout
rejects examples with no real robot points, and zeros padded values before projection
as well as masking attention keys, so even NaNs in unused slots cannot affect its output.

### Integration defect caught and fixed

The new backward check initially found **NaN gradients in `ActionMemory.t_embedding`**
despite finite forward outputs. During front-padding, all scene keys were masked.
Track-On's `MHA_Block` zeros the attention result for such rows, but on the installed
torch 2.4.1 CPU path that did not prevent NaNs in backward. Earlier replica checks tested
front-padding only in forward, and their gradient check used fully valid history.

`ActionMemory` now skips padded observations per example, runs its layers only on live
rows, and writes masked zeros for inactive rows. Every processed example has a valid
current scene frame. No upstream Track-On files were changed.

### Verification

All **8 readout tests pass on CPU**, including:

- k=1/k=3 shapes and distinct multi-query outputs at initialization;
- independent point permutations per example, moving padding with the points;
- padded-value perturbations (including NaN) and comparison with padding removed;
- per-frame independence, batch/frame reshaping, and eval determinism;
- finite nonzero parameter/query gradients and zero gradient at padded robot inputs;
- explicit failures for malformed masks/shapes and all-padded inputs;
- memory-to-readout backward through entirely and partially front-padded batches,
  including training with gradient checkpointing enabled, plus end-to-end causality;
- one actual cached window at D=256: 115 real robot points,
  output **(1, 3, 1, 256)**, all finite. This CPU integration check uses a short window
  and one replica layer; it is not a full production GPU benchmark.

The existing `46_verify_replica.py` also passes after the padding fix, including its
unchanged 0.7546 past-perturbation result and zero future-frame influence.

Reproduce from `/scratch/oozak23/droid_qp`:

```bash
/scratch/oozak23/envs/track_on_r/bin/python -B scripts/48_verify_readout.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /scratch/oozak23/envs/track_on_r/bin/python -B scripts/46_verify_replica.py
```

`DECODER_BUILD.md` also records the loss baseline agreed in this conversation: scaled
translation MSE, squared rotation-matrix distance, and gripper BCE. Gripper labels are
next observed states derived from measured width, not directly recorded commands.

**Next:** step 4, the position/rotation/gripper head. Training has not started.

---

## Step 4 — action head ✅ 2026-09-10

**Built:** `ActionHead(D=256)` and `rotation_6d_to_matrix` in
`droid_qp/action_decoder.py`; verified by `scripts/49_verify_head.py`.

Three independent linear projections map each readout token to body-frame translation
(3 values, metres), raw 6D rotation (6 values), and next observed gripper-state logit
(1 value). The head has **2,570 trainable parameters** at D=256. All outputs preserve
`(B, T_win, k, ...)`, including k=1, and the output dictionary additionally contains the
decoded 3x3 rotation matrix for the agreed rotation loss.

Position and rotation weights start small but nonzero (`std=1e-3/sqrt(D)`), with the
rotation bias set to `[1,0,0,0,1,0]`. Gripper uses `std=1e-2/sqrt(D)` and zero bias.
This initializes near zero motion/identity rotation with gripper probability near 0.5,
while allowing gradients to reach the memory on the first backward pass. There is no
output clipping, per-coordinate rotation standardization, or sigmoid in the head.
Training-set group normalization remains part of step 5, not an implicit output transform.

The tensor rotation decoder uses the label pipeline's first-two-COLUMNS convention.
It applies Gram-Schmidt with re-orthogonalization to handle numerical cancellation near
collinear vectors. Zero/collinear vectors have deterministic axis fallbacks; these
exceptional paths are not smooth at their boundaries. Half/bfloat16 rotation inputs are
promoted to float32 for decoding, while float64 is preserved for numerical verification.
It uses only PyTorch and does not import the kinematics-dependent action-label module.

### Verification

All **10 head tests pass on CPU**:

- k=1/k=4 shapes, finite outputs, and near-zero/identity initialization;
- known +90-degree z rotation to detect a row/column or direction reversal;
- 128 random quaternion-derived rotations from the existing `geom.quat_to_R` formula
  round-trip through the 6D decoder at float64 tolerance 1e-12;
- proper rotations (orthogonal with determinant +1) for zero, collinear, nearly
  collinear, tiny, half/bfloat16 inputs, and CPU bfloat16 autocast;
- finite-difference gradient checking away from degenerate inputs;
- physical translation units and unrestricted gripper logits;
- gradients from each output branch to its input features;
- independence of individual frame/query slots and invalid-shape rejection;
- actual cached data through `ActionMemory -> ActionReadout -> ActionHead`, using
  D=256, T_win=3, M=4 and one replica layer: finite, nonzero gradients in every
  parameter with checkpointing enabled, using actual labels in a diagnostic objective.

The existing **8 readout tests also pass**. These are CPU correctness checks, not a
production GPU benchmark or evidence that a trained policy succeeds at a task.

Reproduce from `/scratch/oozak23/droid_qp`:

```bash
/scratch/oozak23/envs/track_on_r/bin/python -B scripts/49_verify_head.py
/scratch/oozak23/envs/track_on_r/bin/python -B scripts/48_verify_readout.py
```

**Next:** step 5, the agreed weighted/masked loss and training-set group scales.
No production loss, optimizer loop, training run, or checkpoint was added in step 4.

---

## Step 5 — weighted behavior-cloning loss ✅ 2026-09-10

**Built:** `PolicyLoss` and `compute_loss_scales` in `droid_qp/policy_loss.py`, scale
generation in `scripts/50_compute_loss_scales.py`, verification in
`scripts/51_verify_loss.py`. Artifact: `out/policy_loss_scales.json`.

The loss combines translation squared error, half the squared Frobenius rotation-matrix
error, and BCE-with-logits against the next observed gripper state. Each continuous term
is divided by its own group scale squared; all three coefficients are 1. Reduction is
`sum(weight * valid * per_target_loss) / sum(weight * valid)`, with float32 arithmetic
under autocast. The loss returns the total and each component separately for logging.

### Training-only scale measurement

Read all **12,123 train clips / 9,441 episodes / 1,924,243 valid frames**. No validation
labels or cached visual tokens are read by scale computation. Manifest quality weights
sum to **1,529,894.1839** over valid training frames.

| group | formula for scale squared | measured scale |
|---|---|---|
| position | weighted mean `||target_dp||²` | **0.016682001054574205 m** |
| rotation | weighted mean `0.5 * ||target_R - I||_F²` | **0.04567586452607171** |

The rotation scale is chordal, approximately 2.617 degrees for small rotations. This
calibrates zero motion and identity rotation to component loss 1 over the full weighted
training-frame population. Individual minibatches, sampled by clip/window, can differ.
There is no centering or per-coordinate 6D normalization.

The JSON includes manifest/used-label hashes, source paths, formulas and counts. Loading
with `manifest=...` checks that manifest's hash. It does not reread all labels to verify
the stored label hash. The scales are buffers in the loss module, so save its state with
the training checkpoint. Recompute the artifact if labels, split or weights change.

### Edge cases and interface

- Loader targets `(B,T,10)` are accepted directly for k=1. For k>1, explicit aligned
  targets `(B,T,k,10)` and masks `(B,T,k)` are required; silent broadcasting is rejected.
- Invalid and zero-weight entries are excluded before computing geometry or BCE, so
  invalid-label NaNs do not contaminate the loss. Active nonfinite values raise errors.
- Negative/nonfinite weights, nonbinary active gripper labels, and shape mismatches fail.
- All-invalid/all-zero-weight batches return differentiable zeros with `weight_sum=0`.
  Step 6 must skip their optimizer step too, including AdamW weight decay.
- `n_valid` reports active targets with positive clip weights; `weight_sum` supports
  aggregating component numerators and denominators across evaluation batches.

### Verification

All **10 loss tests pass on CPU**, covering analytical weighted values and gradients,
known 90-degree rotation error, physical translation scaling, perfect predictions,
masked NaNs/zero gradients, empty batches, global weight-rescaling invariance, explicit
chunk alignment, checkpointed scales, invalid-input rejection, and train-only statistics.
The scale fixture deliberately has no validation label file or index entry; touching
validation labels would fail. Changing validation weight leaves both scales unchanged,
and an overlapping train/validation episode is rejected.

The actual loss also runs through a real cached D=256 window with one replica layer,
M=4, T_win=3, CPU bfloat16 autocast and gradient checkpointing. Every head/readout/memory
parameter received a finite nonzero gradient. Observed initial component values for this
one window: position **0.06389**, rotation **0.02336**, gripper **0.68755**, total **0.77479**.
These are wiring-check results, not trained-model performance. The **10 head tests** pass
as well. No optimizer step or training run was performed.

Reproduce from `/scratch/oozak23/droid_qp`:

```bash
/scratch/oozak23/envs/track_on_r/bin/python -B -u scripts/50_compute_loss_scales.py
/scratch/oozak23/envs/track_on_r/bin/python -B scripts/51_verify_loss.py
/scratch/oozak23/envs/track_on_r/bin/python -B scripts/49_verify_head.py
```

**Next:** step 6, training. Start with the planned small-clip overfit check before a long run.

---

## Step 6 — optimizer loop and overfit gate ✅ 2026-09-10

Implemented `droid_qp/policy_train.py`, `scripts/52_train_policy.py`,
`scripts/53_overfit_policy.sbatch`, and `scripts/54_verify_training.py`.
The six CPU tests pass: actual parameter updates in all modules, no AdamW update on an
empty batch, physical-unit head learning rates, warm-up/cosine scheduling, identical next
updates after checkpoint/optimizer restoration, and identity/180-degree rotation metrics.

A CPU smoke run exposed an important optimizer scaling issue. With one shared AdamW rate
of 3e-4, the first update moved translation error from 0.844 cm to 9.645 cm. Adam's moment
normalization does not adjust parameter steps for physical output units. With position
and rotation projection rates multiplied by their respective loss scales, ten updates
reduced loss from 1.05325 to 0.77178 and position error from 0.844 to 0.406 cm. Memory,
readout and gripper retain the base rate; head outputs and loss definitions are unchanged.
Both smoke runs are retained under `out/policy_cpu_smoke*_step6` for provenance.

### GPU result: passed

Slurm job **1657724**, pinned NVIDIA L40S, ran the production D=256/3-layer decoder with
M=24, scene_slots=24, T_win=48, B=10, BF16 autocast, and gradient checkpointing disabled.
The overfit diagnostic disables dropout and weight decay. Base LR is 3e-4 with 20 warm-up
steps and a cosine schedule planned for up to 1,000 updates.

One fixed 48-frame window was selected from each of 10 distinct training episodes, with
tracking weight >=0.75, motion, and at least one gripper-state change. There are **480
valid supervised targets and 14 gripper-change frames**. The run fits these selected
windows; it does not use every frame of ten complete videos.

It passed the predeclared gate at **step 150** and stopped automatically: total loss
<0.05, each component <0.02, gripper accuracy >=99.5%, all transition frames correct.
Recorded training/evaluation time: **132.66 seconds**. Peak allocated GPU memory:
**24.53 GB**. At this configuration updates took approximately 0.86 seconds after startup.

| metric | initial | final |
|---|---|---|
| total loss | 2.266938 | **0.018774** |
| position loss | 0.740753 | **0.005470** |
| rotation loss | 0.833565 | **0.007381** |
| gripper BCE | 0.692621 | **0.005922** |
| mean translation error | 1.13056 cm | **0.103071 cm (1.031 mm)** |
| mean rotation error | 1.66864 degrees | **0.195020 degrees** |
| gripper accuracy | 52.29% | **100%** |
| next-state accuracy on change frames | 3/14 | **14/14** |

Zero translation/identity rotation score **1.03908 cm / 1.65254 degrees** on these windows.
The current-observed-state persistence baseline scores **97.08%** gripper accuracy but
**0/14** change frames. These are offline action predictions; no robot was controlled.

Artifacts in `out/policy_overfit_1657724`:

- `summary.json`: `status=overfit_passed`, initial/final metrics, full configuration;
- `metrics.jsonl`: per-10-step training losses, evaluation every 50 updates;
- `config.json`: exact selected clip IDs/starts, hyperparameters and data/scale hashes;
- `best.pt`, `last.pt`: model, loss buffers, optimizer, scheduler, step/epoch/cursor and RNG;
- `learning_curve.png`: loss, translation, rotation and gripper curves on the fixed windows.

**Interpretation:** the decoder and optimizer can memorize these examples, including the
rare gripper-change frames. This is not yet evidence of held-out generalization, causal
use of scene memory, or successful robot control. Full-split training remains unrun.

The 48-frame length follows Track-On's recipe, bounding recurrent activation memory and
batch padding. Samples include another 23 preceding warm-up frames; only the final 48
receive supervision. Full-split mode varies window starts across epochs, while this test
keeps them fixed to make memorization measurable. Entire videos or state carried between
consecutive chunks remain possible future tests for longer dependencies.


## Step 7 — held-out evaluator and predeclared pilot — 2026-09-10

Added fixed validation windows, copy-previous translation/rotation baselines, gripper event
matching with missed/extra penalties, per-lab metrics, and paired scene/robot-history
interventions. `PILOT_PLAN.md` and immutable `out/validation_plan_v1.json` were written
before submitting job 1657753. Whole-validation copy previous is 4.33767 mm; fixed-window
copy previous is 4.55066 mm. The strict position gate remains 4.35 mm.

The pilot starts fresh and samples one window per train clip, with dropout restored to
0.1 and checkpoints selected by weighted validation loss. It cannot automatically start
a long run. All three gates must pass; see the frozen plan for exact event definitions
and ablation donor restrictions.

Added offline EE integration and projected gripper-only FK evaluation, with separate
oracle-width and binary-endpoint-width outputs and explicit recorded-observation limits.
The analytic body-rotation test and real-clip oracle reconstruction passed. On that clip,
maximum saved-FK disagreement was 0.0000212 px; GT delta composition error was
0.000232 mm. Validation tests passed four checks, including a short held-out clip.

Pilot and rollout jobs completed; results follow.


### Completed pilot and rollout results — reviewed 2026-09-11

Jobs 1657753 and 1657754 completed all training, clean evaluation, paired interventions,
1,326-clip FK rollout and report generation. Best checkpoint by validation loss is update
758. Validation loss 2.84579 -> 1.73949, position 13.80085 -> 9.99133 mm. Every lab still
loses to copying the preceding action (aggregate 4.55066 mm). All frozen gates failed:
position >=4.35 mm, scene-shuffle increase 4.1267% <=10%, event cost 5.77251 >=1.05583.

Paired anchors: clean 10.60646 mm, scene shuffle 10.72818 mm, robot-history shuffle
23.67941 mm. This supports strong history dependence, with the protocol's causal
interpretation and donor caveats retained. Gripper events: 865 matched, 568 missed,
1,889 extra, F1 0.41318. No longer training run was started.

Offline rollout policy/copy-each-recorded-previous errors: 20 steps 114.284/21.919 mm;
40 steps 166.553/22.972 mm. Oracle-width projected policy errors 50.640/73.182 px.
All geometry checks passed. Twenty-two one-step action targets lack a stored successor
pose, so all rollout methods exclude them; one-step rollout count is 63,534 rather than
63,556. Exact per-lab results, all horizons and plots are in
`out/policy_pilot_1657753/PILOT_REPORT.md` and its linked JSON artifacts.


## Fixed training comparison and next bounded experiment — 2026-09-11

Diagnostic 1658338 evaluated the step-758 checkpoint without weight updates. On all
12,123 seen windows: loss 1.695, position 10.030 mm versus copy 4.612 mm. On validation:
loss 1.739, position 9.991 mm versus copy 4.551 mm. There is little aggregate fitting gap;
this does not identify whether duration, optimization or input information limits fitting.

Prepared a fresh five-epoch experiment (3,790 updates) with the same model/loss and seed,
but a five-epoch cosine schedule. All recovered epoch-0 training windows are frozen in
`out/training_eval_plan_v1.json`; they and the original validation windows are evaluated
at initialization and every epoch. Best remains selected only by validation loss. Final
paired memory ablations retain the original pilot gates. No budget extension is automatic.

The user will launch `scripts/67_run_multiepoch.sh` in their GPU allocation. No GPU job
was submitted during preparation. Protocol: `MULTIEPOCH_PLAN.md`.


## Current-proprioception one-epoch variant — prepared 2026-09-11

Added an opt-in current-state loader and late-fusion policy in `policy_proprio.py`. Inputs
are q[t] (7), recorded qdot[t] (7), and current scaled gripper closure (1). Fixed scales
are pi, 1 rad/s, and 0.725 respectively; no future differences or target fields are used.
A 15->256->256 GELU encoder adds state features immediately before the unchanged heads.
Its final projection is zero-initialized and its initialization preserves the shared model
weights and RNG. It adds 69,888 parameters.

All 13,449 sensor files passed finite-value/coverage audit. Six new checks passed, covering
real and short clips, frame alignment, future-state isolation, padding, identical shared
initialization/predictions/RNG, encoder gradients and checkpoint restoration. Six existing
training checks also passed. A CPU end-to-end smoke run completed an optimizer update,
validation, checkpoint reload and scene/robot-history/proprioception interventions.

Protocol is in `PROPRIO_PILOT_PLAN.md`. `73_run_proprio_pilot.sh` uses the original
one-epoch schedule (758 updates), then runs the complete epoch-end train/validation
diagnostic, best-checkpoint ablations and comparative report. It does not use the five-
epoch schedule and does not alter the already-running visual model. No GPU run submitted;
the user launches it in a separate available GPU step.
