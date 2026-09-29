# Object query points on DROID — results

Reimplementation of MolmoMotion's stage-1 query-point selection, so we control the point
distribution and can later add robot points. Everything is 2D. See `PLAN.md` for the design
and the surrounding architecture (frozen Track-On, `point_memory` as state memory).

Code: `droid_qp/` (package) + `scripts/` (01–03a). Env:
`$PY_MOLMO` (+ `scikit-learn`, `imageio-ffmpeg`,
`accelerate` installed this session).

---

## 1. Data access — settled

`pipeline/01_build_manifest.py` → `out/manifest.json`

**All 23,532 clips resolve**, across all 13 labs, to both a source MP4 *and* a
`trajectory.h5` under `/frozen/avg/bora_data/.../droid_raw/1.0.1/`. The plan had only
`AUTOLab` spot-checked and flagged coverage as an open risk; it is closed at 100%.

**Frame INDEX mapping is confirmed**: release frame i = source frame 4i = index i on the
tracks-NPZ T axis, 1280x720 -> 854x480. Ratio measured 3.93-4.00 over 8 random clips.

**Frame RATE: `droid_videos_index.json` is wrong, and so was my first conclusion here.**
It claims source 60 fps -> release 15 fps. The raw MP4s do carry a 60 fps container
header, but each source frame corresponds 1:1 to a `trajectory.h5` step. Measured over 10
random episodes using PointWorld's `canonical_timestamps` (absolute wall-clock ms):

    source_frames / trajectory_length = 0.99 - 1.00
    trajectory rate                   = 14.1 - 14.5 Hz

So the **source is ~15 Hz and the MolmoMotion release is ~3.6 Hz**. I earlier declared
this risk closed on the strength of the 4.00 frame-count ratio — that test cannot
discriminate the two hypotheses, since the ratio is 4 either way. Only absolute
timestamps settle it, and neither DROID's h5 nor the release ships them; PointWorld does.

Consequence: every release video generated so far is ~3.6 Hz, so inter-frame motion is
**4x larger** than assumed. Rates available: 14.3 Hz (every source frame), 7.2 Hz
(PointWorld subsamples the trajectory by 2), 3.6 Hz (MolmoMotion release). With
PointWorld as the dataset and their tracks not serving as labels, there is no longer a
reason to keep the /4.

**Released tracks are in release pixel space** (854x480). Measured over 200 clips /
938,864 point-observations: 99.68% of *visible* observations fall inside 854x480.
`ds_dim` is `(H, W)`. Invisible points are extrapolated and can sit far outside the
frame — always mask by `visibility` first.

## 2. Segmentation + sampling — validated

`analysis/03a_sam_check.py` → `out/sam_check_sheet.png`, `out/sam_check_png/`

Tests the SAM 2.1 + k-means half in isolation by substituting a known-good prompt point
(the median of MolmoMotion's released track points) for MolmoPoint. Metric is
**reference-point recall**: the share of *MolmoMotion's own* object points that land
inside the mask **we** produced — an external check, not self-consistency.

**24/24 clips produced a mask. Median recall 100%, min 89%.**

### Mask-choice rule: MolmoMotion's does not transfer

Their rule is "smallest mask with any pixel within 20 px of the point"
(`querypoints_from_video.py:329-353`). That is correct for SAM 3 **text** prompts, where
each returned mask is a whole object instance. It is wrong for SAM 2 **point** prompts,
where the three candidates are deliberately *subpart / part / whole* — so "smallest"
systematically returns a subpart.

Measured on the same 24 clips:

| rule | median recall | mean recall | >70% | median mask area |
|---|---|---|---|---|
| `smallest` (MolmoMotion) | 100.0% | 90.8% | 21/24 | 0.81% |
| `largest` | 100.0% | 99.4% | 24/24 | 4.25% |
| **`best_iou`** (default) | 100.0% | **99.0%** | **24/24** | **0.86%** |

`largest` ties on recall but produces **5x larger** masks — table and background bleeding
in, which would put query points on things that are not the object. `best_iou` (SAM 2's
own predicted mask quality) gets the same recall with tight masks. Now the default in
`segment.choose_mask()`.

### Known failure mode

One clip segmented the **whole table** rather than an object (14.5% mask area). Notably
MolmoMotion's reference points there are also spread over that table — so their
annotation is the table too. A dataset property, not a bug in our port, but it is the
kind of clip that would teach a world model that a table moves.

## 3. Grounding — the caption path is largely a dead end

**`clips_by_object` carries no object name.** Every one of the 23,532 entries is keyed by
the literal placeholder string `"object"` — 1 unique key in the whole release. So the
release ships motion ranges only, and the noun phrase has to be regenerated.

**Qwen3-0.6B hallucinates on DROID's templated captions**, copying its own few-shot
examples out of the prompt:

    'Hang or unhang object'                      -> 'container | strawberry toy | moose toy'
    'Move object to a new position...'           -> 'red plush toy'
    'Use object to pick up something'            -> 'red plush toy'
    'Use cup to pour something granular'         -> 'granular two'

This is more dangerous than plain vagueness: `is_vague_object("red plush toy")` returns
**False**, so such a clip skips the re-caption branch and gets segmented against an object
that is not in the video. `grounding.is_hallucinated()` guards it by enforcing the
extraction prompt's own rule — *"Use ONLY words from the task description"*.

On genuine free-text captions Qwen3 is fine:
`'stack cup into bowl' -> 'cup | bowl'`, `'put brick in drawer shelf and close drawer' -> 'brick'`.

## 4. Text-free grounding — removes the whole chain

`molmo2_pointing.py:35` ships a prompt that needs no object name:

    DEFAULT_PROMPT = "Point to all objects in contact and moved by robot arm"

If MolmoPoint works with this on DROID, the entire grounding chain disappears — **no
Qwen3, no Molmo2-8B re-caption, no SLURM, no vague captions, no hallucination guard** —
and it applies to all 23,532 clips instead of the small non-vague subset. It is also
simpler than the plan's original text-free route, which needed the FK gripper point and
camera calibration; this needs neither.

Confirmed working (`pipeline/03_object_points.py --text-free`): MolmoPoint returns points
with no object name, and the point lands on the same object the reference tracks cover
(clip 2: text-free (595, 270) vs reference-derived (593, 230), 100% recall).

## 5. Grounding is the weak stage — and where to ground matters most

Two runs over the **same 12 clips**, differing only in which frames MolmoPoint sees:

| | whole video | motion window (`--ground-in-motion-window`) |
|---|---|---|
| clips producing points | 12/12 | 12/12 |
| prompt point inside mask | 12/12 | 12/12 |
| median mask area | 1.1% | 1.0% |
| **agreement with MolmoMotion** | median **100%** | median **45%** |
| **points on the right object (by eye)** | ~6/12 | **11/12** |

Grounding on the whole clip asks "what is the arm touching?" over frames where it is
touching nothing, and MolmoPoint drifts to large flat textured surfaces — a yellow
placemat instead of the bag on it, a tan mat instead of the drawer. Restricting
grounding to the motion window fixed the placemat case and most others.

**Agreement with the reference went DOWN while quality went UP.** This is the sharpest
evidence so far that the released tracks are not a ground truth: grounding at frame 0
reproduces their answer because their annotation is also anchored near frame 0 and has
drifted, while grounding in the motion window finds what the gripper actually holds and
therefore disagrees with them.

**Retire `ref_point_recall` as a quality score.** It is a valid check on *segmentation*
— does SAM grow a sensible mask around a given point — and an invalid one on
*grounding*. On a clip where MolmoMotion is wrong, high recall is a failure signal.

### Worked example: `WEIRD+f61abce9+2023-12-13-21h-09m-22s__23804457`

Caption "Move object into or out of container", motion window [35, 58] of 59 frames.
The object is a **blue marker** the arm lifts from a red cup (confirmed by eye at frame
46, and by frame differencing over the motion window).

| | picks |
|---|---|
| MolmoMotion's released tracks | the **yellow storage shelves** in the background |
| ours, whole-video grounding | (inherited — validation used their point) |
| ours, motion-window grounding | the **wicker basket** |

Neither finds the marker. Their tracks additionally wander between the shelves and the
gripper across the clip rather than tracking one thing. Two distinct grounding failures
worth keeping as a test case: background-surface capture, and object/container ambiguity.

Also note from the frame-difference heatmap: the largest motion in that clip is **the
robot arm itself**, not the object. "Where is the motion" is not "where is the object".

### Not the clip filter

`--ground-in-motion-window` uses `clips_by_object` motion ranges to choose *where to look
for the object*. MolmoMotion uses the same ranges to *delete static clips from the
dataset*, which is the sampler bias we are undoing. Static frames stay in our data: the
query points found in the motion window are tracked across the whole clip.

Artifacts: `out/contact_sheet.png` (whole video), `out/mw/contact_sheet.png` (motion
window), `out/03_textfree.log`, `out/03_mw.log`.

---

## Deviations from MolmoMotion, and why

| | |
|---|---|
| **Taken** | stage-1 selection: point at object → segment → k-means (`sampling.kmeans_sample` is a verbatim port) |
| **Changed** | SAM 2.1 instead of gated SAM 3; point prompt instead of text; `best_iou` instead of `smallest`; text-free instead of caption-driven |
| **Rejected** | their model; their tracks as labels (ViPE-lifted, 13–20% teleport); 3D lifting; **`clip_segments.py`'s motion ≥ 0.02 m/frame filter** |

That last rejection is the point of the whole exercise. Their filter keeps only moving
windows and drops any video where nothing survives, and their points cover a single object
mask — no background, no robot, no distractors. A model trained on that **has never seen a
static point**, which is the sampler bias behind PointWorld's static-scene failure. We
reimplement stage 1 specifically so we can keep static windows and add robot points.

## 6. PointWorld overlap — and a proposed change of approach

Fatma hoca directed the work onto the PointWorld dataset. Inventory of
`/frozen/avg/bora_data/droid_datasets/pointworld_droid_restored/droid/flows-fs-optimized`
(**42,935 unpacked `<episode>_flows.h5`**, no extraction needed):

Per episode: root attrs (`uuid`, `current_task`, `trajectory_length`,
`canonical_timestamps`, and serial + 6-DoF extrinsics + mp4 path for ext1/ext2/wrist),
then ~15-19 clip windows keyed `"start:end"` (11 frames, stride 5). Per window per
exterior camera: `extrinsic (4,4)` world->cam, `intrinsic (3,3)` for 320x180,
`initial_rgb` JPEG, `initial_depth (180,320)`, and `scene_flows (11, N~26600, 3)` float16
plus colors / normals / visibility / depth-valid masks. Per window robot state:
`joint_positions (11,7)`, `gripper_positions (11,)`, `gripper_pose (11,7)`,
`gripper_open (11,1)`, torques, velocities. Episode level: `ee_pos_motion_magnitudes`,
`ee_rot_motion_magnitudes`, `has_gripper_movement` as per-window *attributes*.

**There are no 2D points — everything is 3D.** But 2D is exact, not fitted: projecting
`scene_flows` through `extrinsic` then `intrinsic` puts **100.0%** of visible,
depth-valid points inside the 320x180 frame at every timestep. This removes the plan's
Step 2 entirely (fitting DROID extrinsics, ~60 px error, per-serial blacklisting).

**Overlap:** 10,904 episodes = **14,086 MolmoMotion videos** (episode+camera; PointWorld
files are per episode and hold both exterior cameras). The camera MolmoMotion chose is
present in PointWorld in 200/200 sampled. ~203,000 11-frame windows. PointWorld covers a
median 77% of each episode timeline (min 14%).

**Captions do not improve with the overlap.** PointWorld's `current_task` is the same
DROID caption plus a fixed taxonomy parenthetical ("(ex: drawer, clothes hamper, ...)"),
identical for every clip in a category. 104/120 sampled differ only by that suffix. And
MolmoMotion never pointed with the caption -- it used Qwen3/Molmo2-8B output that is not
shipped. Irrelevant to us anyway: our grounding is text-free.

### Proposal: find the object geometrically, not semantically

Both observed grounding failures picked something that never moves (a marker on a
whiteboard tray; a wicker basket instead of the pen in it). PointWorld makes the
geometric definition available: **the object is what the gripper holds and moves with.**

    1. grasp frame        = the 1->0 transition in `gripper_open`   (model-free, exact)
    2. carrying phase     = steps where `gripper_open == 0`
    3. gripper motion     = rigid transform from `gripper_pose` over that phase
    4. score each of ~26,600 dense points: does its 3D track match that motion?
    5. object = points near the gripper at grasp AND moving with it -> project to 2D
    6. optionally SAM from their centroid for a clean mask, then k-means to exactly 100

Measured on 300 random episodes: **94% have a clean gripper-close transition**, 94% have
a carrying phase, median 62% of the episode is spent holding. So this is the main path,
not an edge case; the VLM becomes the ~6% fallback.

It also handles cases where "the object" is semantically awkward -- drawers, lids,
curtains, doors (1,777 Open/Close + 1,278 Cover/Lid clips): the manipulated thing moves
with the gripper whether or not it is a portable object.

Caveats: a close transition does not prove the gripper closed on the *intended* thing;
low-motion tasks (403 Press/Turn/Twist) will have weak rigid-motion signal. Both are
measurable, which is the main gain -- "does this point move with the gripper" is a
number, unlike "is this the right object", which is currently only my eyeball judgement.

Cheap fix worth making regardless: **ground at the grasp frame, not the motion-window
start.** `image_num` was 0.0 in all 12 clips, so we always land on the window's first
frame, often before the object is held.

Consequence for ordering: robot points now come *before* finishing object points,
reversing the plan. Justified because the plan budgeted for fitting camera extrinsics,
and PointWorld ships them exactly.

## 7. Phase 0 — the two gripper calibrations (settled)

`pipeline/00_calibrate_gripper.py`, 79 windows over 40 episodes →
`out/gripper_calibration.json`. Both were guesses before this; both are now measured.

**`gripper_pose[:, :3]` is `panda_link8` — the Franka FLANGE, not the fingertips.**
Position error **0.0000 m**, offset in the link frame ~1e-8, unanimous 79/79. The
Robotiq 2F-85 extends past it: `outer_knuckle` 0.070 m, `outer_finger` 0.088 m,
**`inner_finger` 0.125 m**. So a grasped object sits ~0.13–0.17 m from `gripper_pose`,
which is exactly the gap measured to the nearest dense scene point. **Any "near the
gripper" test must use the `inner_finger` links from FK, never `gripper_pose`.**

**`gripper_pose[:, 3:7]` is XYZW**, not WXYZ: 0.00° vs 166.31° median geodesic error
against FK. Reading it as WXYZ still yields plausible-looking residuals, so this would
have corrupted every score silently.

Bonus: FK reproduces `gripper_pose` to 0.0000 m, which verifies the URDF chain, the
`panda_joint1..7` mapping and the Robotiq mimic joints end to end. Robot points via FK
are therefore already validated — the plan had budgeted a whole step for that.

## 8. The scorer, and two corrections to my own claims

`analysis/05_score_points.py` + `droid_qp/geom.py`.

**First formulation was wrong.** It asked, per query point, "is your nearest dense point
attached to the gripper?" Calibration (`analysis/08_calibrate_threshold.py`) shows only
~52–64 of ~26,600 dense points are attached in a carrying window — ~2% of the scene — so
a query point correctly on the object usually still finds a STATIC nearest neighbour.
The metric under-reports. Replaced by **attached recall**: of the points provably moving
with the gripper, what fraction lands inside our mask. Same direction as the old
reference-recall, but against a reference we trust.

**Threshold calibration also had a bug**: it filtered windows by end-effector motion but
not by whether the gripper was CLOSED, so it included windows where the arm swings
holding nothing. Fixed (`--carrying-only`):

| residual cut | all high-motion windows | carrying only |
|---|---|---|
| 0.010 m | 4 attached | **28** |
| 0.020 m | 14 | **52** |
| 0.030 m | 20 | **64** |
| windows with ≥20 attached | 51% | **64%** |

Operating point **r_max = 0.03 m**. The residual histogram is unimodal — the static
population peaks at 0.04–0.10 m, i.e. at the gripper displacement, exactly as the rigid
model predicts — so this is a chosen cut, not a discovered valley.

### Geometric coverage: the number moved three times, so here is the audit trail

| claim | basis | verdict |
|---|---|---|
| 94% | episodes with a `gripper_open` 1→0 transition | measures grasp DETECTION, not object reconstruction — wrong quantity |
| 43% | attached points at approach 1's detection-frame window | partly an artifact: residual averaged across the grasp |
| 87% | `analysis/10_probe_gripper_region.py`, windows with the gripper closed in ALL 11 frames and ≥0.08 m motion | real, but that population is selected for clean carrying |
| **~40–50%** | the eval set, after the carrying-span fix | **the honest operating number** |

Three further diagnostics narrowed the cause of the ~50% that fail:

* **Not timing.** Failing clips have *full* 11-frame carrying spans and *more* gripper
  motion than passing ones (0.163 m vs 0.097 m).
* **Not camera choice.** Scoring both exterior views recovers only 2/12 failures;
  with the best camera the failing group still has median 8 attached points.
* **Not missing geometry.** Failing clips have *hundreds* of dense points near the FK
  fingertips (near_tip 148, 483, 890) while attached counts stay at 4–19.

So the object region **is** reconstructed and its 3D tracks are simply too noisy to move
rigidly with the gripper — ViPE monocular depth, the same contamination that disqualified
these tracks as labels. Relaxing `r_max` does not help: yield peaks at 0.03 and falls
away at 0.05 because the points then fail the movement condition instead.

### Consequence

Geometry is a **verifier on ~40–50% of clips**, not a replacement for the VLM. The
scorer inherits that ceiling — it is an unbiased sample on the adjudicable half, which is
far better than eyeball counts, but it cannot judge everything.

### (superseded) the first framing of this correction

I previously reported 94% coverage. That figure is for **grasp detection** — episodes
with a `gripper_open` 1→0 transition. It says nothing about whether the held object
produces dense points that move rigidly with the gripper.

Measured (`analysis/09_diag_gripper_at_detection.py`) on the 40-clip eval set: of the 21
clips with too few attached points, **20 had the gripper CLOSED at the detection frame**.
The robot was definitely holding something and the geometric signal was still absent.

    episodes with a grasp transition                     94%
    clips where a held object actually reconstructs      15/35 = 43%

So the geometric method **cannot be the primary path** — it is absent on more than half
of clips where the robot is provably holding something. This is the thin-object failure
mode anticipated in §6's caveats: a 320x180 depth map often does not resolve a grasped
object, and ViPE depth on small objects is too noisy to move rigidly.

Consequences: geometry is a **verifier/filter on ~43% of clips**, not a replacement for
the VLM. The scorer inherits the same ceiling — it can adjudicate ~40% of clips, which
is still an unbiased sample and far better than eyeball counts.

## 9. Approach 1, measured (40 clips, `out/a1_mw40`)

Re-run with the longest-motion-range fix, `best_iou`, k=100, on `out/eval_clips.json`
(40 clips over 16 categories, all inside the overlap). 40/40 produced 100 points with
the prompt point inside the mask; median mask area 0.9%.

Scored: **mean attached recall 40.9%, median 42.4%** over the 15 adjudicable clips;
5/15 above 70%, 8/15 above 30%, 5/15 at ~0%. Median centroid distance to the attached
cluster 63 px. This is the first objective number for approach 1 — previous figures were
my visual judgement.

## 10. Query-point source: settled by measurement (60 natural-frequency clips)

All variants scored with attached recall on the same clips. `*` marks a circular
number -- the geometry correction optimises the metric it is scored on, so those are not
comparable with the rest.

| variant | grounding | mean | median | ~0% |
|---|---|---|---|---|
| `nat_base` | our MolmoPoint (text-free) | 38.4% | 12.2% | 15/33 |
| `mm_queries` | MolmoMotion's released points, raw | 50.3% | 53.7% | 4/33 |
| **`mm_sam`** | **MolmoMotion seed + our SAM + k-means** | **59.5%** | **69.6%** | **2/33** |
| `nat_geo` | MolmoPoint + geometry correction | 78.0%* | 87.5%* | 0/33 |
| `mm_sam_geo` | MolmoMotion + SAM + geometry | 74.6%* | 77.9%* | 0/33 |

**Our own grounding lost.** MolmoMotion's released points beat our text-free MolmoPoint
(50.3% vs 38.4%) -- the Molmo2-8B re-caption branch we skipped was doing real work.
**Our segmentation won**: seeding SAM from their points lifts them 50.3% -> 59.5% and
gives a fixed N=100 instead of their variable 31-94, with no 4B model in the loop.

Decision: **`mm_sam`** for the 14,083 overlap clips. The other 28,852 PointWorld episodes
have no MolmoMotion entry and still need MolmoPoint, a grid, or geometry alone.

## 11. Frozen Track-On produces the scene memory (`out/trackon_*`)

Frames + query points -> `Predictor` -> `point_memory` of shape **(100, 72, 256)**, i.e.
(N points, M=72 memory slots, D=256). Nothing had to be built: the memory is a live
attribute allocated as zeros and filled one token per frame
(`trackon_predictor.py:177-178`). Run online via `forward_frame`, which is also how the
policy will consume it. `support_grid_size=0` so the memory holds only our points --
turning it on adds 400 support tokens the policy would also see.

**Birth frames left 7-40% of every clip with an EMPTY memory.** Track-On is causal, and
the detection frame comes from the motion window, so the opening of each episode -- the
approach phase, where a policy most needs state -- had nothing in it.

Fixed by seeding at frame 0 from MolmoMotion's *bidirectional* tracks (ours cannot look
backwards). Scored by tracking forward and evaluating where the points END UP at the
original detection frame, since a frame-0 seed cannot be scored in place (the gripper
holds nothing at frame 0, so the rigid-motion test is undefined):

| | seed at detection frame | seed at frame 0 |
|---|---|---|
| attached recall (same frame) | 59.5% | 55.3% |
| visibility over the whole clip | 64.1% post-birth | **73.0%** |
| blind frames | **7-40% of clip** | **0** |

**Seed at frame 0.** Four points of recall buys a fully populated memory from the first
frame. Track-On carries their frame-0 positions forward reliably -- median 78 of 100
points survive to the detection frame, though the range is 6 to 100, so a survival
filter is advisable before training. 2/60 clips had no visible MolmoMotion point at
frame 0 and were dropped.

## Not done

- Robot points (URDF FK via PointWorld's `RobotSampler`) — needs camera extrinsics
- Track-On-R wiring; drift vs FK ground truth
- Whether the causal limitation forces a reversed pass for `t < birth_frame`
- Distributional comparison against the released tracks at scale
