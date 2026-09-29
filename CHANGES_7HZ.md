# Change list — 3.6 Hz → 7.2 Hz, plus the token stream

Batch 1 of the run-up to dataset generation. Scope: make the pipeline produce 7.2 Hz data
and save the artifact the policy actually consumes. **Nothing here writes to `/frozen`**
and nothing here runs at scale — the 14,083-clip harness and the action labels are batch 2.

Verified against the current files; line numbers are as of 2026-08-14.

---

## 1. NEW — `droid_qp/rates.py`

The subsample factor is currently hard-coded in four places (`video_io.py:21`,
`geom.py:167`, `19_robot_points.py:94`, plus two docstrings). Flipping them one at a time
is how a pipeline ends up half at each rate, with no error anywhere to say so.

```python
SUBSAMPLE    = 2     # source frames per release frame. 2 -> 7.2 Hz, 4 -> 3.6 Hz
PW_SUBSAMPLE = 2     # source frames per PointWorld step. Fixed by PointWorld, never ours.
SOURCE_HZ    = 14.3  # measured from canonical_timestamps, NOT the 60 fps container header
RELEASE_HZ   = SOURCE_HZ / SUBSAMPLE

_R = SUBSAMPLE // PW_SUBSAMPLE          # 1 at 7.2 Hz, 2 at 3.6 Hz

def release_to_pw_step(f):  return f * _R      # vectorised: works on int or ndarray
def pw_step_to_release(s):  return s // _R
```

Both helpers stay vectorised so `19_robot_points.py` can call them on an `arange`.

## 2. `droid_qp/video_io.py`

- line 21 — `SUBSAMPLE = 4` deleted; import from `rates` instead
- line 44 — the `-vf` string already interpolates `{SUBSAMPLE}`, so it follows automatically
- line 7 — docstring says `mod(n,4) -> 854x480 @ 15 fps`; **both halves are wrong** (the
  15 fps claim is the frame-rate error corrected in `RESULTS.md` §1). Rewrite.
- `reconstruct()` gains `subsample: int = rates.SUBSAMPLE` so a caller can be explicit.

**⚠ The stale-video trap — the one thing here that can silently poison the dataset.**
`reconstruct()` line 37 is `if dst_mp4.exists() and not overwrite: return dst_mp4`. Every
MP4 under `out/*/release_videos/` is **3.6 Hz**. Regenerating into those paths returns the
old videos, and every frame index downstream is then wrong with nothing raising. Fix both
ends:

- videos go to a rate-tagged directory (`release_videos_7hz/`)
- `reconstruct()` asserts the decoded frame count is consistent with `subsample` before
  accepting a cached file

## 3. `droid_qp/geom.py`

- lines 161-167 — `release_to_pw_step` becomes a one-line delegate to `rates`. **Keep the
  name and signature**: `05_score_points.py`, `09_diag_gripper_at_detection.py`,
  `12_viz_attached.py` and `attached.py:48` all import it and need no edit.
- its docstring hard-codes "release frames are trajectory/4" — update.

## 4. `scripts/19_robot_points.py` — the rate change **and** the bug

- lines 10-11 — docstring states the `//2` mapping; update.
- line 94 — `rel = np.arange(len(uv)) // 2` → `rel = rates.pw_step_to_release(np.arange(len(uv)))`
  (identity at 7.2 Hz).
- line 95 — **the bug.** Currently:

  ```python
  keep_step = (rel < T) & filled          # covered by a window, lands inside the video
  s0 = int(np.argmax(keep_step))
  vis0 = ok[s0]                           # never checked to contain any True
  ```

  `ok` is the projection test. At episode start the arm sits in its retracted home pose,
  frequently outside an exterior camera's view, so `vis0` can be all-False. The npz is
  then written with an empty `(0,3)` query set and the failure only surfaces one stage
  later, in `20_…py:83`, as *"no FK step at frame N"*. Measured **1 of 8 clips**:
  `AUTOLab+5d05c5aa+2023-11-17-22h-46m-01s__24400334` has coverage from step ~4 but no
  visible FK point until step 8.

  ```python
  MIN_VIS = 5      # matches the threshold 20_…py:89 already enforces
  keep_step = (rel < T) & filled & (ok.sum(axis=1) >= MIN_VIS)
  ```

  Picking 5 makes the two scripts agree on what counts as usable instead of disagreeing.

- guard: `continue` rather than writing an npz when the query set is empty, so a failure
  reports in the stage that caused it.

**Consequence to carry forward:** on those clips the robot points are born later, so the
action memory is empty for the first few frames. That is physical — there is nothing to
sample when the arm is not in the picture — so it gets handled by masking the loss over
the warm-up, not by moving the birth frame.

## 5. `scripts/20_robot_points_sam_filter.py` — **no change**

It reads `fk_release_frame` from stage A's npz rather than recomputing the mapping
(line 82), and takes the segmentation frame from the filename (lines 41, 97). Both follow
the new rate automatically. `segment.py` is untouched too — the SAM frame is data-driven
in both pipelines (`03_object_points.py:240-241`, `20_…py:97`).

## 5b. `droid_qp/index.py` + `scripts/03_object_points.py` — **found during implementation**

Not in the original list. `index.py:24` carried its own `SUBSAMPLE = 4`, which looked at
first like a fifth copy of the same constant. It is not — it is a **different constant
with a colliding name**, and getting that wrong in either direction breaks something.

**There are three frame spaces, not two:**

| space | factor | who owns it |
|---|---|---|
| trajectory step | 1 | DROID, ~14.3 Hz |
| PointWorld step | `PW_SUBSAMPLE = 2` | PointWorld — action labels, calibration |
| MolmoMotion release | `MM_SUBSAMPLE = 4` | the published dataset — **fixed forever** |
| our release | `SUBSAMPLE = 2` | us |

`MM_SUBSAMPLE` must NOT follow `SUBSAMPLE`: `droid_clips.json`'s `num_frames` and
`motion_ranges`, and every index in the released `*_2d.npz` tracks, are in MolmoMotion's
3.6 Hz space regardless of what we generate at. So `index.py` keeps using 4, now
re-exported from `rates` with the reasoning attached rather than written out locally.

**What does change** is that a MolmoMotion index no longer addresses one of our frames.
New helpers `rates.mm_frame_to_release` / `release_to_mm_frame`. Frame 0 maps to frame 0,
which is why the frame-0 seeding path (`mm_sam_f0`, the winning variant) is rate-invariant
— that is luck of the design, not a general property.

**The live bug this exposed:** `03_object_points.py:184-198`, the
`--ground-in-motion-window` path. `s, e` are read from `rec.motion_ranges` and clipped
against `rec.num_frames` — both MolmoMotion space — then handed to `video_io.trim` on
**our** reconstruction. At `SUBSAMPLE=2` that trims half the intended window at half the
intended position, with nothing raising. Fixed by converting through
`mm_frame_to_release`, extending the inclusive end by `span - 1` since one MolmoMotion
frame now spans two of ours.

Not hit by the current default path (frame-0 seeding does not trim), but it is exactly
the class of error that survives review and shows up as unexplained quality loss.

## 6. `scripts/16_run_trackon.py` — save the token stream

Today it writes `tracks / visibility / queries` (line 186) and never saves the memory,
which is the actual product.

**No Track-On edit is needed.** `trackon_predictor.py:178` does
`self.point_memory[:N, -1] = q_new`, so the newest memory slot *is* the per-frame token.
`run_clip` already holds `pm = predictor.point_memory[:predictor.N]` at line 107 for its
fill statistics.

- in `run_clip`, allocate `tokens = np.zeros((T, N, 256), np.float16)` and after each
  `forward_frame`, scatter `pm[:n_active, -1]` through the same `idx = order[:n_active]`
  used for `tracks` at line 102. **The rows are in birth order, not query order** — using
  `idx` is what keeps token *i* aligned with query *i*.
- also capture `q_init = predictor.q_init[:predictor.N]` once — `(N, 256)`, ~46 KB. It is
  the birth anchor, kept outside the memory and re-fed as the decoder's input query every
  frame (`trackon_predictor.py:95-96`), so the replica needs its equivalent.
- save `tokens` fp16, `q_init` fp16, alongside the existing arrays.

### The one design call inside this change

Object and robot points go through Track-On in **one pass**, and the result is split into
two arrays at save time:

```
scene_tokens (T, N_obj, 256)      # object points -- read-only, what the action memory attends INTO
robot_tokens (T, N_rob, 256)      # robot points  -- input to the trainable replica
```

Two arrays, so **no role array is stored** — the role is implicit in which file you read.
A role vector exists only as an intermediate inside the pass, to do the split.

Known imperfection, deliberately deferred: under Fatma hoca's design the replica's input
should strictly be the backbone sampled at the **true FK pixel**, whereas Track-On's token
sits at its **estimated** position. The two differ by tracker drift. Saving `fk_uv`
alongside lets us *measure* that drift and re-extract only if it matters, rather than
settling it by argument first.

---

## Smoke test — the gate before batch 2

Same 8 robot clips and a handful of object clips, end to end at 7.2 Hz. Four things to
check, each of which has a right answer known in advance:

1. reconstructed videos have **~2× the frame count** of the 3.6 Hz ones (not equal — that
   would mean the stale-cache trap fired)
2. all 8 robot clips produce a non-empty query set; `AUTOLab+5d05c5aa+…` specifically now
   works, born at its first genuinely visible step
3. `fk_release_frame` equals `arange(S)` exactly — the mapping is the identity
4. `tokens` is `(T, N, 256)` with `T` matching the video, and rows align with `tracks`
   (a token whose `visibility` is False should still be written — Track-On writes memory
   every frame regardless)

---

## Result — job 1500276, 2026-08-14, **9/9 gates passed**

`scripts/21_smoke_7hz.sbatch` on the 8 clips that had robot points at the old rate.
Artifacts in `out/smoke7_{obj,robot,robot_sam,trackon}/`.

| stage | 3.6 Hz | 7.2 Hz |
|---|---|---|
| object points | 8/8, 100 pts, 0% off-mask | **identical** — frame-0 seeding is rate-invariant |
| video length | = MolmoMotion `num_frames` | **exactly 2x**, all 8 (1.98-2.00) |
| robot FK | **7/8**, one empty query set | **8/8**; `5d05c5aa` born@f8, N=32 |
| robot SAM filter | mean 98% over 7 | mean 96.6% min 87.5% over **8** |
| Track-On | memory `[100, 72, 256]` | unchanged; **+ `tokens (T,100,256)` fp16** |

The SAM-filter mean drops only because the 8th clip is now included at 87.5%; the other
seven are unchanged. `S == T` on clip 1 (451 == 451) confirms the identity mapping.

Storage measured: 8.6 MB/clip of tokens at N=100 on this sample. Extrapolating to N=180
over 14,083 gives **~218 GB** here versus the ~182 GB predicted from the 58-clip
distribution — this 8-clip set skews long (mean T=168 vs 140). Both sit comfortably in the
206 TB available; take 180-220 GB as the range.

### Follow-up found by the run, not a blocker

`5d05c5aa` yields only **32 of 128** robot points because `--min-visible 5` births at the
*first* step where the gripper is entering frame. Track-On births are once-only, so the
other 96 points are lost for the entire clip.

**Swept (`23_sweep_min_visible.py`, `out/min_visible_sweep.json`) -> use 64.** The
aggregate table is useless: median birth frame is 5.0 at EVERY threshold, because only 3
of 8 clips move at all. The per-clip breakdown is where the answer is:

| clip | th=5 | th=64 | th=96 |
|---|---|---|---|
| `5d05c5aa` | f8 / **32** | f9 / **98** | f9 / 98 |
| `15h-11` | f5 / 58 | f6 / 65 | f8 / **103** |
| `16h-58` | f0 / 77 | f0 / 77 | f7 / 110 |
| other 5 | unchanged | unchanged | unchanged |

At **64** the pathological clip goes 32 -> 98 points for ONE extra blind frame: the 96 dead
slots were an artifact of birthing the instant the gripper's edge appeared. Pushing to 96
gains a little on `15h-11` but costs `16h-58` seven blind frames out of 52. 64 is the knee.

Note this never threatened the fixed-dimension property -- `presample(n, seed)` is
deterministic, so the policy input is 128 fixed slots plus a validity mask regardless. The
threshold decides how many slots are ever populated, not whether the layout is stable.

## Batch 2 — done since

### Action labels (`droid_qp/actions.py`, `scripts/24_action_labels.py`)

The supervision target. All six proprioception fields are extracted, not just the chosen
action space, because they share the same h5 windows -- so DECISIONS.md #2 (EE delta vs
joint velocities vs absolute pose) can be settled AFTER the data exists rather than
blocking it. Per clip, aligned to video frames:

    delta_pos_{body,world} (T,3)      delta_rot6d_{body,world} (T,6)
    position (T,3)  rot_6d (T,6)      joint_{positions,velocities,torques} (T,7)
    gripper_open / gripper_open_next  gripper_positions        valid (T,)

Both delta conventions are stored because both are free and they suit different heads:
world frame matches DROID's cartesian command space; body frame is translation-invariant,
so the same motion carries the same label wherever in the workspace it happens.

Rotation is **6D continuous** (Zhou et al., CVPR 2019), not quaternion. The raw
`gripper_pose` quaternion flips sign mid-episode -- q and -q are the same rotation, but an
L2 loss reads the flip as a near-maximal jump and would train the model to predict a
violent flick that never happened.

**Bug found and fixed by the physics, not by the tests.** A delta spans steps t and t+1,
so it is meaningful only if BOTH are covered by a PointWorld window. Checking only
`filled[t]` recorded the arm's motion across an entire uncovered gap as a single action:
7 "actions" of 51-75 cm per frame, i.e. >5 m/s at the end effector. All 7 sat exactly on
coverage boundaries; none had both ends covered. The summary statistics looked unremarkable
throughout -- the giveaway was that no Franka moves that fast.

The consequence was not just bad rows. Normalisation std over these clips went

    before  [4.46, 1.72, 1.88] cm/frame        after  [1.09, 1.17, 1.28]

so x was inflated 4x by seven frames. Training on that would have silently down-weighted x
against y and z with nothing looking wrong. Invalid deltas are now zeroed as well as
masked, so a forgotten mask downstream degrades to a no-op rather than a 70 cm lunge.

Verified: rotations orthonormal to 4e-9 with det exactly +1, 6D round-trip exact to 1.6e-7,
body and world delta magnitudes agree to 7e-9. Plot: `scripts/25_viz_actions.py` ->
`out/action_labels/actions.png`. The gripper panel is the useful check -- the binary
command steps first and the physical width ramps after, which is the causal order that
proves the label is the command and not a copy of the state.

### Per-clip Predictor reload removed (`16_run_trackon.py`)

`main` rebuilt the whole Predictor for every clip -- re-reading the 94 MB checkpoint and
reconstructing the network -- to obtain state that `run_clip` already cleared with
`reset()` on entry. Measured: **3.20 s** per rebuild against **97 us** for `reset()`, i.e.
12.5 hours of pure reloading across 14,083 clips.

`reset()` (`trackon_predictor.py:51-62`) clears `t`, `point_memory`, `temporal_mask`,
`q_init`, `N`, `capacity`, `device` -- everything the class mutates except `self.model`,
which is exactly what should survive. Verified rather than argued: job 1618149 re-ran the
same 8 clips with one shared Predictor and all five arrays came back **bit-identical**
against the fresh-Predictor smoke run. Checkpoint loads went 9 -> 1.

### Generation harness (`27`, `28`, `29`, `30`, `31`)

    30_gen_seeds.sbatch    stage 0, ONCE   MolmoMotion frame-0 seeds, all 14,083
    27_build_shards.py     14,083 -> 141 shards of 74-123 clips
    29_gen_shard.sbatch    the array       one task = one shard, all five stages
    28_merge_queries.py    object + robot -> one query tensor + role
    31_gen_report.py       what failed, and the --array= string to rerun it

**One task per shard, not five dependent array jobs.** The three conda envs are mutually
incompatible, so the work crosses interpreters regardless; doing it inside one task keeps
every shard independent, with no cross-job dependencies, and a dead task costs 100 clips
rather than 14,083. Resume is at shard granularity via a `DONE` marker written only after
every stage exits 0 -- not per clip, because stages 2-5 mirror stage 1's output directory
with a glob, so a half-populated directory would silently change what they see.

Shard assignment is a **stable hash** of `video_id`, not position, so adding or reordering
clips later does not reshuffle every shard and invalidate finished work.

**Stage 0 is separate and must not be sharded.** `14_molmomotion_as_queries.py` streams a
2.3 GB tar in ONE sequential pass filtered by the wanted set. As 141 array tasks that
becomes 141 reads of the same 2.3 GB off shared read-only storage. It also gained
`--video-ids-from`, because it previously required an existing run dir to mirror and at
scale there is none.

**A silent footgun removed.** `--n` defaulted to 12 / 8 / 12 in `03`, `19`, `16`, and
`--n 0` meant EMPTY rather than ALL. A forgotten flag would have processed 12 clips per
shard and still reported success: ~1,700 clips instead of 14,083, every stage green. `0`
now means all everywhere, matching `24_action_labels.py`.

**Lab distribution -- worth knowing before trusting any number we have.**

| lab | share | | lab | share |
|---|---|---|---|---|
| TRI | 31.8% | | REAL | 7.1% |
| AUTOLab | **14.8%** | | PennPAL | 6.1% |
| IRIS | 10.4% | | ILIAD | 4.6% |
| IPRL | 10.1% | | WEIRD | 2.8% |
| CLVR | 8.4% | | +3 more | 3.9% |

Every clip measured so far -- the 59.5% attached recall, the position probe -- is
**AUTOLab**, under a sixth of the data, and the largest lab is entirely unseen. Shards are
hash-assigned and therefore lab-mixed, so the pilot is the first look at the rest.

## Batch 3 -- reading proprioception from DROID instead of PointWorld

Prompted by a question that looked like bookkeeping ("are the FK tracks coming from
PointWorld's dataset?") and turned into two bugs, one of which would have corrupted every
number we measure against.

### Why the source changed

PointWorld's proprioception is a **verbatim copy** of DROID's `trajectory.h5` -- verified
0.000e+00 on joint_positions, joint_velocities, cartesian_position and gripper_positions,
with the mapping PW step s <-> DROID raw step 2s exact at every sampled step. But it is
WINDOWED, covering a median ~75% of an episode. Reading DROID directly loses nothing and
gains the gaps back.

    action labels   valid frames   71.7% -> 99.6%   (clips under 50% valid: 26 -> 0)
    FK robot points frames with data  75.9% -> 96.8%
    query points    median 98 of 128 -> median 128 of 128, born at frame 0

DROID stores EE orientation as EULER, not quaternion; the convention is scipy extrinsic
`xyz`, measured against PointWorld's quaternion at 0.0000 deg (next-best 9.03, worst
179.82). Unambiguous, but silent if wrong.

**PointWorld is still required for camera calibration** -- their extrinsics were fitted by
aligning FK robot points to the image, so they are tuned for exactly the projection we do.
Only the proprioception moved.

### Bug 1 -- zero-fill is not "missing" to FK

`episode_joint_series` zero-fills steps no PointWorld window covers. Zeros are not absent
data to forward kinematics; they are a valid pose -- the arm standing upright -- and FK
projects it happily. Usually that lands outside the image so `fk_ok` rejects it, which is
LUCK depending on camera placement, not a guarantee. Measured over 80 pilot clips:
**1.085% of uncovered point-steps pass the visibility test**, and where it happens it is
near-total (one TRI clip: 2,304 of 2,304). Those frames entered the ground truth as
plausible pixel coordinates for a pose the arm was never in.

Fixed at the source by reading DROID (no gaps to fill), plus `ok &= filled[:, None]` as a
second line of defence for anyone using `--source pointworld`.

### Bug 2 -- the gripper is a JOINT ANGLE, not a normalised width

Found only by diffing the new FK against the old, which disagreed on **32% of steps, up to
16 px**. Joints matched exactly, so it was the gripper:

    DROID   0.0881  0.1410  0.3040  0.4626      normalised width, 0-1
    PW      0.0639  0.1022  0.2204  0.3354      URDF finger joint angle, radians
    ratio   0.725   0.725   0.725   0.725

Exactly **0.725000** in every episode -- min = max = median over 11 random ones, no spread.
That is the Robotiq 2F-85 finger limit (41.5 deg). `build_robotiq_joint_dict` expects the
angle; passing the raw 0-1 value opens the fingers 1/0.725 = 1.38x too far, and since
EVERY query point sits on the gripper it moves all of them.

After the fix: **99.885% of point-steps identical** to the PointWorld-sourced geometry
(from 68.354%).

The same constant invalidated `GRIPPER_OPEN_THR`, calibrated against PointWorld's scale.
Both fixes together took the gripper labels from 9.40% disagreement -> 0.44% (threshold)
-> **0.01%** (scale), i.e. 2 frames of 20,965.

### What this cost, and the lesson

Neither bug was caught by a test or by output looking wrong. Bug 1 needed asking where the
data came from; bug 2 needed diffing a new implementation against an old one that was
already correct. The FK ground truth is what every policy number will be measured against,
so shipping the DROID reader without that diff would have made the entire evaluation
quietly wrong while every plot still looked fine.

Drift figures quoted before 2026-09-01 (median 5.83 px, ILIAD 24.74 px, TRI p90 94 px)
were computed against the contaminated ground truth and are **withdrawn**. Job 1638036
re-measured them; see below.

### SUPERSEDED -- see the full-dataset measurement at the end of this file

The 34-clip pilot figures below OVERSTATED tracking quality. Kept for the record;
use the 13,643-clip numbers instead.

### Track-On vs FK ground truth -- the measured baseline (job 1638036, 34 clips, 12 labs)

    median error 4.55 px   |   78.0% of point-frames within 10 px
    per-clip:  p10 2.77   p50 4.55   p90 16.74   max 27.00

**The error PLATEAUS rather than compounding** -- the single most useful thing this
measurement produced:

    0-10 frames since birth     0.64 px     essentially sub-pixel
    10-30                       4.01
    30-60                       6.08
    60-120                      6.99
    120+                        7.17        flat

That follows from Track-On's design: every frame it takes an argmax over the feature map
(`trackon.py:242`) rather than integrating a displacement, so there is no mechanism for
error to accumulate. It matters for evaluation: a policy rolled out open-loop WILL
compound because it integrates its own predictions, whereas the tracker will not. So ~7 px
is a STABLE reference usable at any horizon, instead of a baseline that degrades alongside
whatever it is measuring.

**The tail is systematic, and it inverts an assumption.** AUTOLab is the WORST lab for the
tracker -- all three sampled clips at 11-13 px median, ~40% within 10 px -- while CLVR and
GuptaLab sit at 2-3 px with >95% inside 10 px. Worst individual cases: an ILIAD clip at
25.94 px (20.6% within 10) and an IPRL clip with p90 = 102 px.

Every quality number we produced before 2026-09-01 was measured on AUTOLab alone, on the
assumption it was representative. For tracking specifically it is the hardest lab, not a
typical one. Since Track-On's output IS the scene memory, this spread is a measured
ceiling on the policy's input quality.

### Left open by this run

`20_robot_points_sam_filter.py` builds its gripper mask **at the birth frame**, which is
now frame 0 -- where the arm is at its start position, small in the image and sometimes
half out of view. Keep fraction fell to mean ~91% (from 96.6%) with a long tail: one TRI
clip kept 14 of 128, and one clip of 194 lost its robot half entirely.

The fix does not require trading coverage against mask quality. The filter answers a
question that is not frame-specific -- *which of the 128 FK indices sit on the visible
gripper* -- and `presample(n, seed)` makes index n the same physical spot at every frame.
So the mask can be built where the gripper is clearest (most FK points in frame, largest
mask), or at several frames with a majority vote, and applied to points born at frame 0.

## Explicitly NOT in this batch

- the full 141-task array (~182 GB, days of GPU) -- gated on the pilot's yield.
- anything under `/frozen/avg/ozan/`.


---

# Full-dataset measurement (job 1657144, 13,643 clips)

The 34-clip pilot was not representative. Both corrections below were quoted earlier in
this document as if settled.

| | pilot (34 clips) | **full (13,643)** |
|---|---|---|
| median error vs FK | 4.55 px | **7.72 px** |
| within 10 px | 78.0% | **59.8%** |
| worst clip | 27 px | **376.54 px** |

So **40% of point-frames sit more than 10 px from truth**, not 22%.

### The error does NOT fully plateau

| frames since birth | pilot | full |
|---|---|---|
| 0-10 | 0.64 | **0.74 px** |
| 10-30 | 4.01 | **5.48 px** |
| 30-60 | 6.08 | **8.50 px** |
| 60-120 | 6.99 | **9.36 px** |
| 120+ | 7.17 | **10.30 px** |

Increments are +4.74, +3.02, +0.86, **+0.94**. Growth decelerates about 5x after the first
~30 frames but keeps creeping. The pilot's final step was +0.18, which genuinely looked
flat, and "plateaus at ~7 px / a stable reference at any horizon" was written on that
basis. **That claim is withdrawn.** Accurate version: the error is not compounding, but it
is not stationary either, so any comparison must state its horizon. Reference values: ~5 px
at 20 frames, ~10 px beyond 120.

### Occlusion recovery -- 1,799,558 events over 13,589 clips

Fatma hoca's fourth question ("it leaves and comes back -- will the tracker recognise
it?"), answerable rather than merely illustrable because robot points carry exact FK
ground truth at the return frame.

    within  5 px : 19.3%
    beyond 20 px : 32.5%      <- came back to the WRONG place

**One in three re-acquisitions is wrong.** The finding is the second column:

    gap  3-5 frames   median 11.70 px   30.2% bad
    gap  5-10                12.24      31.7%
    gap 10-20                12.86      33.3%
    gap 20-50                12.73      33.1%
    gap 50+                  12.84      34.1%

**Occlusion LENGTH barely matters** -- 30% at three frames against 34% at fifty-plus. Under
memory decay a longer gap would be far worse; the flatness says the mis-identification
happens AT THE MOMENT OF LOSS, not through forgetting. Once the tracker lets go, more time
does not make it worse.

Return frames are also worse than ordinary ones: ~12 px median against 7.72 px overall.

Figure: `out/gt_all/occlusion.png`. Data: `out/gt_all/occlusion_recovery.json`.

### Consequence for the policy

The scene memory is noisier than assumed: 40% of point-frames beyond 10 px, and a third of
post-occlusion returns landing somewhere else entirely. That is the quality ceiling on the
policy's ENTIRE input, since Track-On's output IS the scene memory. See DECISIONS.md #10.
