# Query points for DROID — object + robot → frozen Track-On

Status as of 2026-08-03. Resume here tomorrow.

---

## 0. Where the project stands

### Architecture (decided with Fatma hoca, 2026-08-03)

Changed from the earlier design. **No point decoder is being built.**

- Frames + query points → **frozen Track-On-R**
- Track-On's own internal `point_memory` **is** the state/scene memory
- The **action decoder** reads that same state memory
- The **action memory** is a *replica of Track-On's transformer block*, so both memories have the
  same shape and can attend to each other
- Robot points go to the action decoder; object points are the scene points

**2D only — no 3D lifting.**

### First task (this document)

Work out how to choose query points, implement it, and test on DROID videos with points on the
object *and* the robot. Everything downstream (world model, action decoder, static-scene fix) is
out of scope for now.

---

## 1. Findings — bring these to the next meeting

### MolmoMotion-1M has no static clips, and tracks only object points

Verified by reading `/scratch/oozak23/molmo-motion/data_generation`:

1. `pipeline/clip_segments.py` keeps only frame ranges where motion ≥ `clip_threshold = 0.02`
   m/frame, and **drops any video where nothing survives**. All 23,532 DROID entries have ≥1
   motion range.
2. Points are k-means centroids of a **single object mask**. No background, no static objects, no
   distractors, no robot.
3. Therefore a model trained on these tracks **has never seen a static point.**

This is the sampler-bias hypothesis, confirmed and stronger than expected: it is not that points
were selected *because* they moved — it is that **nothing else was ever tracked.** Independent
justification for building our own point selection.

Nuance worth stating precisely: *within* a kept clip, points are **not** motion-selected (k-means
over the mask is uniform). The bias is at the clip level and in what gets masked at all.

### Two blockers

| Blocker | Evidence | Decision |
|---|---|---|
| `facebook/sam3` gated | `gated=manual`, download returns 403 GatedRepoError | Use **SAM 2.1** (`facebook/sam2.1-hiera-large`, ungated). Request SAM3 access anyway. |
| DROID captions are useless as text prompts | **454 unique captions across 23,532 clips; 84.3% contain a vague noun.** Top caption is "Move object into or out of container" ×6,022 | **Go text-free** — prompt SAM 2.1 with the FK-projected gripper point. No captioner, no VLM, no language. |

Reproduce the caption stat with `annotations/droid_clips.json` + MolmoMotion's own `_VAGUE_NOUNS`
set (`third_party/sam3/molmo2_pointing.py:536-543`).

### What MolmoMotion's stage 1 actually does

```
caption → Qwen3-0.6B noun phrase → (Molmo2-8B recaption if vague)
        → MolmoPoint-Vid-4B point → SAM3 text prompt → masks
        → smallest mask within 20px of the point → KMeans(k=100)
```

The sampling itself is ~25 lines — `kmeans_sample()` at
`third_party/sam3/querypoints_from_video.py:66-90`. KMeans over the `(x,y)` pixel coords of the
mask, centers rounded to int, output `(N,3)` int32 rows `[frame_idx, x, y]`.
Stages 2–6 (ViPE depth, AllTracker, 3D lift, filter/smooth, clipping) are **not needed** — we're 2D.

---

## 2. Decisions taken

| Question | Decision |
|---|---|
| Robot points | **URDF forward kinematics**, as PointWorld does |
| Object points | **Text-free**: SAM 2.1 prompted with the FK gripper point |
| Frame rate | **RELEASE mode** — every 4th frame, comparable to MolmoMotion reference tracks |

### Why URDF FK for robot points, not segment + k-means

`RobotSampler.presample()` is **seeded and deterministic**, so point index *n* is the same physical
location on the robot in every clip. A k-means point set over a robot mask has arbitrary per-clip
index semantics — close to useless as a fixed-dimension input to an action decoder. FK also gives
free per-frame ground truth to measure Track-On drift against, and spends no point budget on the
static base.

---

## 3. What exists on disk

| Thing | Path | State |
|---|---|---|
| Track-On-R checkpoint | `/scratch/oozak23/track_on/checkpoints/track_on_r.pt` | ✅ 94 MB, ready |
| DINOv3 backbone | HF cache | ✅ downloaded |
| Franka URDF | `/scratch/oozak23/PointWorld/assets/franka_description/franka_panda_robotiq_2f85.urdf` | ✅ |
| Robot point sampler | `/scratch/oozak23/PointWorld/robot_sampler.py` | ✅ 587 lines, reusable |
| DROID annotations | `/scratch/oozak23/droid/annotations/` | ✅ 23,532 clips |
| DROID camera calib | `/scratch/oozak23/droid/camera/camera-0000.tar` | ✅ 18,092 scenes, not extracted |
| DROID videos | `/scratch/oozak23/droid/` | ❌ not shipped — rebuild from raw |
| DROID raw MP4s | `/frozen/avg/bora_data/droid_datasets/droid_raw_full_v2/droid_raw/1.0.1/` | ✅ full mirror |
| MolmoMotion reference tracks | `/frozen/avg/ozan/molmo-motion-1m/droid/tracks/tracks-0000.tar` | ✅ all 23,532 clips |
| SAM 2.1 weights | — | ⬜ **download tomorrow** |

### Environments

| Stage | Env | Has |
|---|---|---|
| A robot points | `/home/oozak23/.conda/envs/pointworld-env` | pytorch_kinematics, urdfpy, trimesh, h5py, sklearn, cv2 |
| B segmentation | `/scratch/oozak23/envs/molmo-motion` | transformers 4.57.6 (`Sam2Model`), cv2, h5py — **needs `pip install scikit-learn`** |
| C tracking | `/scratch/oozak23/envs/track_on_r` | torch, transformers, cv2 |

Each stage writes `.npz`, so the env split is fine.

---

## 4. Design

```
                    DROID trajectory.h5                    raw MP4
                  (joint_positions,                   (every 4th frame)
                   gripper_positions)                        │
                           │                                 │
              ┌────────────▼─────────────┐                   │
              │  A. ROBOT POINTS (URDF)  │                   │
              │  RobotSampler.presample  │                   │
              │  + compute_points        │                   │
              │  → project via calib     │                   │
              └────────────┬─────────────┘                   │
                           │  gripper pixel (t*, x, y)       │
              ┌────────────▼─────────────┐                   │
              │  B. OBJECT POINTS        │                   │
              │  SAM2.1 point-prompt     │                   │
              │  − robot mask → KMeans   │                   │
              └────────────┬─────────────┘                   │
                           │                                 │
              ┌────────────▼─────────────────────────────────▼──┐
              │  C. frozen Track-On-R (Predictor)               │
              │  queries (1,N,3) = (t,x,y) in ORIGINAL pixels   │
              └─────────────────────────────────────────────────┘
```

### Reuse — do not rewrite

- `/scratch/oozak23/PointWorld/robot_sampler.py` — `RobotSampler`, `convert_joints_to_dict`,
  `build_robotiq_joint_dict`
  - `.presample(num_points, gripper_filter='both', seed=...)` → deterministic surface sampling
  - `.compute_points(joint_dict)` → `(T, N, 3)` in robot base frame
- `/scratch/oozak23/PointWorld/dataset_components/robot.py:393` (`_get_robot_flows_droid`) — the
  exact DROID joint-dict pattern: `panda_joint1..7` + `build_robotiq_joint_dict(gripper_positions)`
- `/scratch/oozak23/track_on/model/trackon_predictor.py::Predictor` — construct with
  `support_grid_size=20` for sparse queries (as `demo.py:501` does)
- `/scratch/oozak23/track_on/utils/train_utils.py::load_args_from_yaml`
- `/scratch/oozak23/track_on/utils/vis_utils.py::save_video`
- `querypoints_from_video.py:66-90` — port `kmeans_sample` verbatim

**Do not reuse** `vis_utils.py::plot_tracks_wo_tail` — hard-codes color from birth-y, one matplotlib
figure per frame. Write ~40 lines of `cv2.circle` instead.

### Camera calibration

Extrinsics measured over 40 clips (EE reprojection error vs reference object-track centroid):

| Source | Error |
|---|---|
| h5 `observation/camera_extrinsics/<serial>_left`, inverted | **60.6 px** ← best |
| `optimized_extrinsics` as world→cam | 76.2 px |
| `vggt_extrinsics` as world→cam | 79.1 px |
| either, inverted | 228–349 px |

So `optimized_`/`vggt_extrinsics` are **world→camera** and "world" ≈ **robot base**. Default to the
h5 key, fall back to `optimized_extrinsics`. Per-serial quality varies a lot (`22008760`: 39–65 px;
`24400334`: 85–108 px) — blacklist bad serials. `measured_intrinsics` is stored at ~640×360 and
must be rescaled to `ds_dim = [480, 854]`.

The 60 px includes the object-centroid offset, so true calibration error is lower. Part is likely a
constant flange↔TCP offset — fit one global offset in Step 2.

### Package layout — `/scratch/oozak23/droid_qp/`

```
droid_qp/
  index.py          # droid_clips.json + droid_videos_index.json + /frozen mirror → ClipRecord
                    #   owns the release_frame ↔ h5_index mapping
  video_io.py       # decode raw MP4, take every 4th frame (RELEASE mode)
  calib.py          # extrinsics load, pose6→T, K rescale, project()
  proprio.py        # h5 reader + grasp-frame detection
  robot_points.py   # RobotSampler → (T,N,3) base → (T,N,2) pixels
  segment.py        # SAM 2.1 point-prompt → object mask, robot-mask subtraction
  sampling.py       # kmeans_sample port (+ optional reproject_to_mask)
  trackon.py        # Predictor wrapper, assemble (1,N,3) queries
  viz.py            # cv2 overlay — cyan = object, magenta = robot
scripts/
  01_build_manifest.py   02_robot_points.py   03_overlay_check.py
  04_object_points.py    05_run_trackon.py    06_compare_molmomotion.py
```

No edits to `/scratch/oozak23/molmo-motion` or `/scratch/oozak23/track_on`.

---

## 5. Tomorrow — order of work

**Step 0 (do first, in parallel)**
- [ ] Submit `facebook/sam3` gated-access request on HF
- [ ] Download `facebook/sam2.1-hiera-large`
- [ ] `pip install scikit-learn` into `/scratch/oozak23/envs/molmo-motion`

**Step 1 — manifest.** Join `droid_clips.json` + `droid_videos_index.json` + the raw mirror +
`camera-0000.tar`. Report how many of the 23,532 clips have a resolvable MP4 **and**
`trajectory.h5` **and** a camera entry — only `AUTOLab` was spot-checked. Assert
`h5_T ≈ 4 × num_frames` and log violators. **Also resolve the frame-rate question (see Risks).**

**Step 2 — robot points.** No GPU, no models. URDF FK → project → overlay. Fit the global
flange→TCP offset on ~50 clips. Lowest risk, highest information; de-risks everything after it.

**Step 3 — Track-On-R on robot points only.** Validates tracker wiring against FK ground truth
before any segmentation exists. Produces a real drift curve. Decide here whether Track-On's causal
limitation forces a reversed second pass.

**Step 4 — object points.** SAM 2.1 point-prompt from the calibrated gripper point, offset ~3–5 cm
along the EE −z approach axis; `multimask_output`; reject masks with >30% IoU vs the robot mask;
k-means k=100.

**Step 5 — combined run + comparison** against the MolmoMotion reference tracks. Compare
**distributionally** (mask IoU at the detection frame, point-set Chamfer, endpoint displacement),
**not** point-for-point — the reference went through stages 5–6, so its N is 31–94, never 100.

---

## 6. Verification

Per sampled clip:
1. `{vid}_seed.png` — grasp frame, object mask, object points (cyan), robot points (magenta), SAM2
   prompt point marked
2. `{vid}_tracked.mp4` — Track-On-R output, two colors, hollow circle when invisible
3. `{vid}_fkgt.mp4` — **tracked** robot points vs **FK-projected** robot points → quantitative drift
   curve, not eyeballing

Aggregate gates:
- Fraction of clips where the FK gripper point lands inside the SAM2 object mask, **reported per
  camera serial** (automatic calibration QA)
- Track-On drift on robot points vs FK ground truth, as a function of frames since query
- Object-mask IoU vs MolmoMotion reference

---

## 7. Risks

| Risk | Handling |
|---|---|
| **Effective frame rate may be ~3.5 Hz, not 15.** Raw `22008760.mp4` decodes 308 frames with a 60 fps header (5.13 s), but the matching h5 has 309 steps over **21.78 s** at ~14.1 Hz. If the header is wrong, every-4th-frame gives very large inter-frame motion. **UNRESOLVED — verify in Step 1.** | Step 3's FK ground truth quantifies whether Track-On copes. FULL 15 Hz mode is the fallback. |
| SAM 2.1 mask snaps to the gripper, not the object | Offset prompt along approach axis; `multimask_output`; reject high-IoU-with-robot masks |
| Flange vs TCP ambiguity in `cartesian_position` | Fit one global offset in Step 2 before trusting the 60 px figure |
| Thin/deformable objects (1,614 "Fold/spread/clump" clips) | Accept lower yield; Grounding-DINO text route is the fallback |
| **Track-On-R is causal** — `Predictor.forward` leaves `pred_trajectory` zero for `t < birth_frame` (`trackon_predictor.py:193-275`), unlike MolmoMotion's bidirectional AllTracker | Seed at frame 0, or add a reversed pass. Decide in Step 3. |
| `np.rint(centroid)` can land outside a non-convex mask (cloth, handles) — MolmoMotion has no reprojection | Add opt-in `reproject_to_mask`, off by default so it can be A/B'd |
| Only `AUTOLab` clip resolution was spot-checked | Step 1 gates everything on the real coverage number |

---

## 8. Loose ends

- **Not read: the whiteboard photo** (`IMG_3308.jpeg`) and **the MolmoMotion PDF** — both are on the
  Mac, not the cluster. Copy them over:
  ```
  scp ~/Desktop/IMG_3308.jpeg ~/Desktop/"indep study"/molmoMotion.pdf ai12.kuvalar.ku.edu.tr:/scratch/oozak23/
  ```
  Everything above is derived from the code and from the verbal description of the meeting.
- Open design question flagged but not settled: **does the action decoder want *tracked* gripper
  points (which drift) or the FK projections directly?** Worth settling before over-investing in
  tracking the robot set.
