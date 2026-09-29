# Open decisions — action decoder / policy

Things that need a call before or during the policy build. Each says what it blocks, so
work that does not depend on it can proceed meanwhile.

Measured evidence behind these is in `RESULTS.md`. Nothing here is a blocker for
**robot points** or **action-label extraction**, which is why those go first.

---

## 1. What does the action decoder read as state?  ← the main one

**The question.** Fatma hoca's design says the action decoder reads Track-On's
`point_memory` as the state memory. But that memory holds *appearance*, not *location*:

| | contents | shape |
|---|---|---|
| `point_memory` | 256-d feature per point, last 72 frames | `(100, 72, 256)` |
| tracked positions | where each point is, per frame | `(100, 2)` |
| visibility | is the point visible right now | `(100,)` |

`q_new` — the token written to memory each frame — is sampled from the backbone feature
map *at the point's location* (`trackon_predictor.py:95-96`). It encodes what the point
looks like, not where it is. Track-On itself recovers position via its prediction head
using the **current frame** plus memory, not from memory alone.

**Why it matters.** A policy that must reach toward an object needs to know where that
object is on screen. If position is not recoverable from the memory tokens, a
memory-only decoder is missing the one thing it most needs.

**Options.**
- (a) `point_memory` only — as specified.
- (b) `point_memory` + tracked `(x, y)` + visibility. *(Ozan's preference.)*
- (c) (b) plus a positional encoding of `(x, y)` rather than raw coordinates.

**Cost:** zero either way — `16_run_trackon.py` already saves tracks and visibility
alongside the memory.

**Correction (2026-08-14).** The table above understates what is in the memory. The token
written each frame is NOT the raw feature-map sample -- `track_frame` returns
`q_t.squeeze(0)` (`trackon.py:538`), the fully refined decoder output, after frame
cross-attention, cross-point self-attention, memory attention, re-ranking and both
projections. The raw `grid_sample` token is only `q_init`, the birth anchor, kept outside
the memory. So **visibility is in the memory by construction** --
`v_logit = prediction_head(q_t, ...)` takes it as principal input -- and Fatma hoca is
right on that point. Position is the open half. See ACTION_DECODER.md section 0.

**First measurement (`22_probe_position.py`, 8 smoke clips, held out BY CLIP):**

| probe | median error | R2 |
|---|---|---|
| constant baseline | 57.6 px | -- |
| ridge (linear) | **24.6 px** | **0.808** |
| MLP (256,128) | 43.4 px | 0.367 |

Substantial signal -- notable given there is no spatial positional encoding in the decoder
at all (the only one, `t_embedding`, is temporal over the 72 memory slots). Most likely
DINOv3's own patch embeddings leaking through the cross-attention pooling.

**Not yet decisive, for three reasons:** the MLP hit `ConvergenceWarning` at 60 iterations
and scored *below* ridge, so the nonlinear ceiling is unmeasured; all 8 clips are AUTOLab
on 2 camera serials, so near-identical workspace geometry lets a probe exploit scene
priors rather than token content; and 24.6 px is roughly the size of the manipulated
object (~1.1% of frame area). Re-run on the 200-clip pilot across labs, with a converged
MLP, before this settles anything.

**Recommendation:** store tracks + visibility regardless -- 1.4 GB of ~182 GB, i.e. 0.8% --
so the option stays open at no cost, and Fatma hoca's call governs whether the decoder
*uses* them.

**Blocks:** the decoder's input layer.

---

## 2. Action space

`gripper_pose (11,7)`, `joint_positions (11,7)`, `joint_velocities (11,7)`,
`gripper_positions (11,)`, `gripper_open (11,1)` are all in PointWorld per window, so any
of these is free — but they need different heads and normalisation.

- (a) **EE delta pose + gripper** — 6-DoF delta from `gripper_pose` + binary gripper.
  DROID's own convention. `gripper_pose` is `panda_link8`, the flange (calibrated, §7).
- (b) **Joint velocities (7) + gripper** — directly available, avoids rotation
  parameterisation entirely.
- (c) Absolute EE pose — simplest to supervise, worst to deploy.

**Recommendation:** (a), with rotation as 6D continuous rather than quaternion or Euler.
Note the quaternion in `gripper_pose` is **XYZW** (measured: 0.00° vs 166.31° for WXYZ).

**Blocks:** the output head, the loss, normalisation statistics.

---

## 3. Single-step or chunked prediction?

- (a) one action per frame
- (b) chunk of *k* future actions (ACT / Diffusion-Policy style)

Chunking usually beats single-step on this kind of demonstration data and reduces
compounding error, at the cost of a larger output and a temporal-ensembling scheme.

**Blocks:** output shape, loss, evaluation protocol.

---

## 4. `support_grid_size` — ~~0 or 20?~~ SETTLED 2026-08-14: grid points do not enter the memory

`Predictor` can append its own uniform 20×20 grid at t=0 (400 points, `trackon_predictor.py:211-217`).
Their trajectories are discarded, but their tokens would land in `point_memory` too.

**Fatma hoca's call: the grid does not go into the memory.** Rationale: the background /
static-coverage argument for keeping it was a *world-model* argument (it would have tested
the sampler-bias diagnosis behind PointWorld's "always predicts motion" failure), and we
are not building a world model. The policy's state is object points + robot points.

Still open as a pure *tracking* question, which is a measurement and not a decision:
whether grid=20 improves localisation of our own points. The two are separable — track
with the grid on, then slice the grid rows off before saving, so it costs nothing in
storage. When slicing use `tracking_to_original` (`trackon_predictor.py:231-249`); row
order follows the order queries are *born*, not the order they were passed in.

**Blocks:** nothing further.

---

## 5. Query-point composition

Currently **object points only** (100 per clip). But `point_memory` is the policy's
*entire* state, and an object-only set contains no robot and no background.

- (a) object only — current
- (b) object + robot (FK)
- (c) object + robot + background (grid or motion-stratified)

**SUPERSEDED 2026-08-14 -- see ACTION_DECODER.md section 1.** An earlier note here said
robot points go in the SCENE memory, on the reading that the action memory holds k action
tokens' own history. That was my proposal, not Fatma hoca's spec, and her call is the
opposite: **robot points go in the ACTION memory.**

Her version is the coherent one. The action memory is a trainable *replica of Track-On's
transformer block*, so it is shaped `(N_rob, 72, 256)` -- the same shape family as the
scene memory, which is what "the two memories can attend to each other" requires. A memory
of k action queries would not have that shape. It also fixes a hole: a robot slice of the
FROZEN Track-On memory would itself be frozen, leaving the replica nothing to train.

  scene memory  = object points, frozen Track-On, read-only
  action memory = robot points, trainable replica, cross-attends INTO the scene memory

Consequence: **no role array is stored.** Two separate arrays on disk (`scene_tokens`,
`robot_tokens`) make the role implicit in which file you read; a role vector exists only
as an intermediate inside the single Track-On pass, to do the split.

**Recommendation:** (b) at minimum — the architecture explicitly routes robot points to
the action decoder. (c) is worth an ablation; it is also the composition that would test
the original sampler-bias diagnosis, if the world model is ever revisited.

**Blocks:** nothing yet — robot points are being built regardless.

---

## 6. Frame rate — SETTLED 2026-08-14: **7.2 Hz**

The release metadata's claim of 15 Hz is **wrong** (§1). Three rates are available:

| rate | source | control step |
|---|---|---|
| 14.3 Hz | every source frame = every trajectory step | 70 ms |
| **7.2 Hz** | PointWorld's own subsampling — **chosen** | **139 ms** |
| 3.6 Hz | MolmoMotion release — what we used until now | 278 ms |

**Fatma hoca's call: generate the dataset at 7.2 Hz.** Three reasons, strongest first:

1. **Action labels become 1:1.** `joint_positions`, `gripper_pose` and
   `gripper_positions` live in PointWorld's h5 windows indexed at *its* step rate, 7.2 Hz.
   At 3.6 Hz we consumed every other step — half the labels discarded — and carried a
   `rel = step // 2` mapping through `19_robot_points.py` and `20_robot_points_sam_filter.py`.
   At 7.2 Hz that mapping is the identity and can be deleted.
   *Note the alignment is with PointWorld's **proprioception and calibration**, which are
   clean (FK reproduces `gripper_pose` to 0.0000 m; extrinsics were fitted by aligning
   robot points to the image). NOT with its 3D tracks, which are the contaminated artifact
   — AllTracker lifted by ViPE monocular depth, 13–20% teleporting.*
2. **Tracking gets easier** — halving the timestep halves inter-frame displacement.
3. 139 ms per action instead of 278 ms.

**Not 14.3 Hz**, tempting as it is: the action labels only exist at PointWorld's 7.2 Hz
window rate, so going faster means returning to DROID's raw `trajectory.h5` and
re-deriving the labels and re-checking calibration.

**The cost, worth stating plainly.** Track-On's memory is a fixed **72 slots**, so the
horizon is wall-clock and halves:

| | horizon | vs median episode |
|---|---|---|
| 3.6 Hz | 20.0 s | whole episode |
| **7.2 Hz** | **10.0 s** | **~65%** |

Measured over 58 clips: median episode **15.4 s**, p90 **32.3 s**. So the policy forgets
the start of a median episode. Fine for reactive pick-and-place; it would only bite on
tasks with "remember what I already did" structure. `extend_memory` (`trackon.py:69-76`)
could raise M by interpolating `t_embedding`, but that perturbs frozen-model behaviour in
an untested way — leave it unless the horizon demonstrably hurts.

**Consequences to implement:** `video_io.reconstruct` changes `not(mod(n,4))` →
`not(mod(n,2))`; T roughly doubles (mean 70 → 140); storage at N=180 fp16 is ~13 MB/clip,
**182 GB** for 14,083 clips.

---

## 7. Which clips, and how many

- **14,083** overlap clips have MolmoMotion seeds (the winning `mm_sam` source, 59.5%).
- **42,935** PointWorld episodes exist; the other ~28,850 have no MolmoMotion entry and
  would need MolmoPoint (38.4%), a grid, or geometry alone.

**Blocks:** dataset size, and whether the object-grounding problem has to be solved again
for the non-overlap majority.

---

## 8b. Coverage filter -- RESOLVED 2026-09-01, no longer needed

An earlier concern: PointWorld covers a median 75% of an episode, and 13.4% of pilot clips
fell under 50% valid frames, the worst at 6.9% -- nearly useless while still costing a full
17 MB of token stream. That argued for dropping low-coverage clips.

**It dissolved rather than being decided.** The action labels now come from DROID's
`trajectory.h5` directly (`24_action_labels.py --source droid`), which has every step;
PointWorld was only ever a windowed copy of the same numbers, verified identical to
0.000e+00 on every field. Measured over the 194-clip pilot:

| | PointWorld | DROID |
|---|---|---|
| valid frames | 71.7% | **99.6%** |
| clips under 50% valid | 26 | **0** |
| worst clip | 6.9% | **97.9%** |

Note what this recovered, which matters more than the raw 35%: PointWorld's gaps were NOT
uniform. The first tenth of each episode held 17.1% of the missing frames against 10% if
random, and the middle only 6.7%. So we were disproportionately blind to the APPROACH
phase -- the static scene where the policy has to decide what to do, and precisely the
regime PointWorld's own "always predicts motion" failure lived in.

PointWorld is still required for camera calibration. Only the labels moved.

## 8. Point-survival filter

Median **78 of 100** points survive from frame 0 to the detection frame, but the range is
**6 to 100**. A clip retaining 6 points gives the policy a nearly empty state.

**Recommendation:** drop clips below some survival threshold, or make survival a sample
weight. Needs a number; the distribution is in `out/trackon_mm_sam_f0/report.json`.

**Blocks:** the training set definition.

---

## 9. How does the replicated block aggregate across points?

**Track-On's MEMORY is per-point** — `trackon.py:216` does `q_t.view(N_t, 1, D)`, so
each point attends only to its own 72-slot history. NOTE (corrected): the decoder DOES mix
across points each frame, via `query_attention[i](q_t, q_t, q_t)` at `trackon.py:212`. So
replicating Track-On's block already gives cross-point attention; only the memory is
independent per point.

A policy must emit **one action** from **many points**, so aggregation has to happen
somewhere. That is presumably what the extra cross-attention is for. Copying the block
verbatim would inherit per-point independence and leave the decoder unable to relate the
gripper to the object.

**Blocks:** the decoder architecture — this is the core design work.

---

## Already settled (context, not open)

- **Query points:** MolmoMotion seed → our SAM + k-means, fixed N=100, **seeded at
  frame 0** so the scene memory is populated from the first frame (§10–11).
- **`gripper_pose` is `panda_link8`** (flange, 0.0000 m error); fingertips are the
  `*_inner_finger` links, 0.125 m further out.
- **Quaternion order is XYZW.**
- **Projection is exact** — 100% of visible, depth-valid points land in frame.
- **Scene memory shape:** `(100, 72, 256)`, filled one token per frame, with a
  `temporal_mask` marking unfilled slots.

---

# Action decoder — proposed design (for discussion)

**1. A learned action token.** Track-On's decoder mixes across points
(`query_attention`, trackon.py:212) but only ever outputs one token PER POINT -- there is
no mechanism that reduces N points to a single decision, and its memory is per-point
(trackon.py:216). A policy must emit ONE action from N points. So add a learned action query that
cross-attends over all points' memories — that token *is* the aggregation, and its own
FIFO becomes the action memory, matching "same shape so the two memories can attend".
Precedent: DETR object queries (Carion et al., ECCV 2020); CLS token (BERT / ViT);
**ACT** (Zhao et al., RSS 2023) is the closest — a transformer policy using DETR-style
queries to emit a chunk of actions. Read ACT first.

**2. Role embeddings** — robot vs object points play different roles; a learned per-role
embedding lets the decoder tell "my gripper" from "the thing I am reaching for".

**3. Position + visibility encoding** (see decision #1 above). The memory stores
appearance, not location: `q_new` is sampled from the backbone feature map AT the point's
location, so it encodes what the point looks like. Track-On recovers position via its
prediction head using the current frame, not from memory alone.

**4. Gripper-distance attention bias** — bias the action token's attention by distance
between gripper points and object points (T5/Swin-style relative bias). Encodes *contact*
directly rather than hoping it is inferred.

**5. Discretised actions, not L2 regression.** The strongest idea, because it reuses the
argument that started this project: **the regression-to-the-mean failure we diagnosed in
PointWorld applies equally to behaviour cloning.** Demonstrations are multimodal — several
valid ways to do a task — so an L2 head emits their average, an action no demonstrator
took. Discretise into bins with cross-entropy (RT-1, Brohan et al.) or use diffusion
(Diffusion Policy, Chi et al.). Same fix as Track-On's classification + bounded-offset head.

**6. Action chunking** — predict k future actions, reducing compounding error (ACT).

**7. Mask the loss during warm-up** — `point_memory` starts as zeros and fills one token
per frame, so early actions would be supervised from a nearly empty state.

## Robot points: FK + SAM filter (implemented)

FK geometry is exact but "in frame" != "visible": the Robotiq 2F-85 is an open skeletal
frame, so points on the far side of a finger project onto background seen through the
gaps. Filtering FK points by a SAM mask (seeded from the FK gripper position — exact, no
grounding model) keeps **mean 79%, median 81 points**, all on the visible robot, while
preserving both the stable FK index (point *n* = same physical spot every clip) and the
free per-frame FK ground truth. A SAM-sampled point set would lose both.


---

## 10. Tracking-quality filter on the training set  (NEW 2026-09-02)

Measured over all 13,643 clips (job 1657144), Track-On's error against FK ground truth is
**median 7.72 px with only 59.8% of point-frames within 10 px**, and per-clip medians run
from ~1.6 px to **376 px**. A clip at 300 px median is not tracking at all -- its scene
memory holds tokens sampled from wherever the tracker wandered.

Since `point_memory` IS the policy's entire input, this is a ceiling on everything
downstream. Unlike the object-grounding problem it is **measurable per clip**, because the
robot half carries exact ground truth.

**Options.**
- (a) train on everything and let the model cope with the noise
- (b) drop clips above a per-clip median-error threshold
- (c) use tracking quality as a per-clip sample WEIGHT rather than a hard cut

**What the distribution supports.** Clips above ~50 px median are unambiguously broken
rather than merely noisy, so the extreme is easy to justify. Where to draw the line between
10 and 50 px is a judgement about how much input noise the policy tolerates -- itself an
empirical question.

**Recommendation:** (c), or (b) at a deliberately loose threshold. A tight cut would remove
much of AUTOLab and IPRL, and lab diversity has already proved to matter more than assumed:
every quality number produced before 2026-09-01 came from AUTOLab alone, and none survived
contact with the other eleven labs.

**Blocks:** the training-set definition. Per-clip medians are in `out/gt_all/report.json`,
so this can be settled with a histogram rather than an argument.

**Note:** replaces the coverage filter (§8b), which dissolved when the action labels moved
to DROID.
