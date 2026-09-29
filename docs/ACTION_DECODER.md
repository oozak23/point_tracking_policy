# Action decoder — design note

Written after Fatma hoca's 2026-08-14 call. Three things were decided by her, one is open,
and one turns out to be forced by storage arithmetic rather than by taste.

Companion docs: `RESULTS.md` (what the query-point pipeline measures), `DECISIONS.md`
(the older open-questions list — §1 and §5 there are now superseded by this file).

---

## 0. Correction to `DECISIONS.md` §1 — the memory is NOT raw appearance

I previously wrote that the token written into `point_memory` each frame is sampled
straight from the backbone feature map, and therefore "encodes what the point looks like,
not where it is". **That is wrong**, and the difference matters for everything below.

`trackon.py:538`:

```python
return p, v_logit, q_t.squeeze(0)          # <- this q_t goes into point_memory
```

`q_t` at that line has already been through **all** of it: cross-attention into the frame
features, self-attention across points, memory attention, `projection1`, the re-ranking
head, `projection2`. The raw `grid_sample` token (`trackon_predictor.py:95-96`) is only
`q_init` — the birth-frame anchor, kept separately and re-fed as the decoder's input query
every frame. It is never what lands in memory.

The offline path does the same thing at `trackon.py:263-265`.

Consequences:

| quantity | in the memory token? |
|---|---|
| **visibility** | **yes, by construction.** `v_logit` is `prediction_head(q_t, …)` — `q_t` is its principal input. A separate visibility flag is redundant. Fatma hoca is right. |
| **position** | **conditionally.** `p = p_patch + o[-1]`, where `p_patch = argmax(correlate(q_t, f_fused_t))`. The token localizes the point *by correlation against the current frame*. There is **no spatial positional encoding anywhere in the decoder** — the only positional embedding is `t_embedding`, over the 72 memory slots (temporal, not spatial). So absolute `(x, y)` is recoverable from `q_t` **plus** the frame feature map, and only weakly from `q_t` alone (via whatever positional signal DINOv3's own embeddings leak through the cross-attention pooling). |

So the real question is not "should we append `(x,y)` to the memory" — it is **does the
action decoder get to look at the frame feature map?** If yes, position is available
exactly the way Track-On itself gets it, and Fatma hoca's call is straightforwardly
correct. §4 shows why the answer is effectively no, which puts the question back.

---

## 1. Where robot points live — action memory, not scene memory

I told Ozan the opposite last session. Fatma hoca's version is coherent and is the one to
build; my "action memory holds the *k* action tokens' own history" was my own DETR-style
proposal (`DECISIONS.md`, proposed design item 1), not her spec, and I conflated the two.

Her spec, taken literally, is consistent:

```
                   ┌──────────────────────────────────────┐
   object points → │  FROZEN Track-On-R                   │ → SCENE MEMORY
   (SAM + k-means) │  point_memory (N_obj, 72, 256)       │   read-only
                   └──────────────────────────────────────┘
                                                              ▲
                                                              │ cross-attn
                   ┌──────────────────────────────────────┐   │
   robot points  → │  REPLICA block (TRAINABLE)           │ ──┘
   (FK-projected)  │  action_memory (N_rob, 72, 256)      │ → ACTION MEMORY
                   └──────────────────────────────────────┘
                                    │
                                    ▼  aggregate over N_rob
                              action head
```

Both memories are `(N, 72, 256)` — that is what "same shape, so the two can attend to each
other" means, and it only works if the action memory holds *points*. A memory of *k*
action queries would not have that shape.

It also fixes a problem with my version: if the action memory were just a slice of the
frozen Track-On memory, it would be frozen too, and the "replica of the transformer
module" would have nothing to do. Under Fatma hoca's reading the replica is a separate
trainable copy of the block, and the robot half is where gradients live.

**Consequence for the pipeline.** Robot points do *not* need to go through frozen
Track-On. Their positions are known exactly at every frame from FK (§`robot_points.py`),
and proprioception is available at deployment too, so this is not cheating. The replica's
input token per robot point per frame is `grid_sample(f_fused_t, fk_uv[t])` — the same
operation Track-On uses for `q_init`.

**And this is what makes the whole thing precomputable.** §4 shows the frame feature map
cannot be cached (5.3 PB). But the robot half never needs the *map* — only the map
*sampled at the FK pixels*, and those pixels are known offline because FK contains no
learned component. So `robot_tokens (T, N_rob, 256)` bakes ahead of time; the 5.3 PB never
has to exist.

It also simplifies the replica. Track-On opens each layer with `feature_attention`,
cross-attending over the whole map, because it does **not know where the point is** and
must search for it. We do know. Direct sampling replaces the search:

```
q_rob[t] = precomputed backbone sample at fk_uv[t]   # replaces feature_attention
  -> query_attention(q_rob, q_rob, q_rob)            # the gripper as a coherent body
  -> memory_attention(action_mem, q_rob)             # its own trajectory
  -> scene_cross_attention(q_rob, scene_mem)         # what the object is doing
  -> write to the action FIFO
```

Trainable, fully precomputable, no frame features at training time.

That said, running frozen Track-On over the robot points as well costs almost nothing
(one combined pass, N≈180) and buys the **drift measurement against FK ground truth**,
which is the only quantitative check we have that the tracker survives 3.6 Hz. Worth doing
once on a few hundred clips as a diagnostic, not as the production path.

**Supersedes** `DECISIONS.md` §5 and the note added to it.

---

## 2. Where the memory attention goes — recommend BEFORE the scene cross-attention

Track-On's own layer order (`trackon.py:490-511`, `track_frame`):

```
feature_attention(q, f_fused)      # cross-attn into the frame
query_attention(q, q, q)           # self-attn across points
memory_attention([mem; q])         # per-point, over its own 72-slot history
```

Memory attention is **last** — it is the step that resolves the token for its downstream
use (the prediction head). Our replica has a fourth attention to place, and by that same
logic it should go after memory attention:

```
feature_attention(q_rob, f_fused)      # where am I, appearance-wise
query_attention(q_rob, q_rob, q_rob)   # the gripper as a coherent body
memory_attention([action_mem; q_rob])  # my own trajectory: velocity, approach, closing
scene_cross_attention(q_rob, scene_mem)# NEW: what is the object doing
```

**Argument for `before`.** Cross-attention is a query-formation step. A token that has
already absorbed its own history knows *"I am descending and to the left, fingers
closing"* and can ask the scene a targeted question. A token that has only seen the
current frame knows *"I am a fingertip"* and can only ask a generic one. Ordering the
history first makes the scene lookup conditioned on motion, which is what contact
prediction needs.

**The consequential choice is not really the ordering** — it is **what gets written to the
action FIFO**. With memory attention before the scene cross-attention, the written token
is post-cross-attention, so the action memory accumulates *scene-aware* history, and the
next frame's memory attention sees robot history that already contains object information.
That compounds over 72 frames and is, I think, the real payoff. Putting the cross-attention
first would instead keep the action memory robot-pure.

Both are a two-line ablation once the block exists. Recommend `before`, measure `after`.

---

## 3. Aggregating N robot tokens → one action

Still genuinely open, and the core design work (`DECISIONS.md` §9). The replica block
emits one token *per robot point*; a policy emits one action. Three ways:

- (a) **mean/max pool** over robot tokens → MLP head. Simplest, loses which point matters.
- (b) **learned action query** cross-attending over the robot tokens — DETR / ACT style.
  This is where my earlier proposal belongs: not as the action *memory*, but as the
  *readout* on top of it. Naturally extends to a chunk of *k* future actions.
- (c) **designated tokens** — the fingertip points already have stable FK indices
  (`robot_points.py`: `presample(n, seed)` makes point *n* the same physical spot in every
  clip), so a fixed subset could be read out directly.

Recommend (b), because it is also how action chunking arrives for free.

---

## 4. The 12k precompute — yes, with one arithmetic constraint that decides §0

Ozan's proposal: run the object + robot pipeline over the overlap set, write the result to
`/frozen/avg/ozan/` as training data. Agreed in principle. Details:

**Storage is writable.** `/frozen/avg/ozan` is `drwxr-s--- oozak23 avg`, 206 TB free.
`CLAUDE.md`'s blanket "never write to `/frozen`" needs narrowing — the prohibition is
about the *shared read-only mirrors* (`/frozen/avg/bora_data/…`, and never extracting the
MolmoMotion tars in place). Ozan's own directory is fine to write.

**The count is 14,083, not 12k** (`out/overlap_manifest.json`) — clips with both a
MolmoMotion seed and PointWorld coverage.

**What we can and cannot precompute.** Frozen Track-On is deterministic, so caching its
output is a pure training-speed win. But:

| artifact | per clip (T≈60, N≈180, D=256, fp16) | × 14,083 |
|---|---|---|
| token stream `(T, N, D)` | 5.5 MB | **78 GB** ✅ |
| tracked positions `(T, N, 2)` fp32 | 86 KB | **1.2 GB** ✅ |
| visibility `(T, N)` bool | 11 KB | 155 MB ✅ |
| **frame features `f_fused` `(T, 12288, 256)`** | **377 MB** | **5.3 PB** ❌ |

(`input_size: [384, 512]`, stride 4 → 96×128 = 12,288 patches.)

So the frame feature map **cannot be cached**, and re-running the DINOv3 backbone online
during policy training would throw away the entire benefit of freezing Track-On.

**Therefore the action decoder reads memory only — and §0's position question comes back.**
It is not a design preference; it falls out of the storage budget.

**The resolution costs nothing.** Positions and visibility are 1.4 GB out of 78 GB — under
2%. So: **store them regardless**, and let Fatma hoca's call govern whether the decoder
*uses* them. No argument needed, nothing foreclosed, and the ablation is a flag.

Worth running first, because it settles the question with a number rather than an opinion:
**a linear probe from `q_t` to `(x, y)`** on a few hundred clips. If `(x,y)` is linearly
decodable from the memory token, Fatma hoca is right in the strong sense and we drop the
extra inputs for good. If it is not, we have evidence to bring back to her. One day of
work; the tracks are already saved in `out/trackon_mm_sam_f0/tracks/`.

---

## 4b. Storing the whole trajectory does not leak the future

A reasonable objection: if we cache `(T, N, 256)` for the whole episode, how is the policy
causal? Answer: **what is stored is the stream; the memory is a window over it.**

```
stream            [t0][t1][t2] ... [tT]     (T, N, 256)    <- on disk
memory at time t   stream[t-71 : t+1]       (<=72, N, 256) <- what the model sees
```

The dataloader picks a timestep and slices `[:t+1]`. Nothing after `t` is ever visible.
The reason to store the stream rather than the memory is that the memory is a *sliding
window* — saving it per frame would write each token 72 times, for identical information.

This is also exactly how the policy runs at deployment: online, one frame at a time,
memory filling as it goes. The stream is the offline transcript of that online process,
and replaying it is bit-identical to running it live **only because Track-On is frozen and
deterministic**. If Track-On is ever fine-tuned, every cached stream is stale that instant
and we are forced back to running the backbone online — which §4 shows we cannot afford.
So "Track-On stays frozen" is not just an architecture choice; the whole 182 GB depends on
it.

**What this does constrain: the action side is recurrent.** The scene stream is
precomputed, but the action memory is not — the replica writes its own output into its own
FIFO, so it has to be rolled forward through time and cannot be parallelised across `t`.

Track-On hits the same problem and gives us the recipe (`config/train.yaml`): train on
**T=48 frame windows with M=24 memory slots**, then extend to **M=72** at inference via
`memory_extension` (`trackon.py:69-76`, which interpolates `t_embedding` to the new
length). At 7.2 Hz a 48-frame window is 6.7 s, comfortably inside the 15.4 s median
episode. Use the upstream path rather than inventing one.

**Settled 2026-08-14 (Fatma hoca): train the replica at M=24, as Track-On does.**
Two distinct `M`s are involved and they do not interact — worth keeping straight:

| | value | when | whose |
|---|---|---|---|
| Track-On's internal memory | `M_i = 72` | data *generation* | the frozen tracker's own, `config/test.yaml` |
| the replica's action memory | `M = 24` | policy *training* | ours, a window over the stored stream |

Track-On stays at 72 while generating, because that is its shipped inference
configuration (the checkpoint was trained at M=24 and extended); changing it would change
the tokens themselves. Our 24 is applied afterwards, by the dataloader, to the stream.

**Because we store the stream and not the memory, `M` is free at training time.** Train at
24, extend to 72 at inference, ablate anything in between — none of it requires
regenerating a clip. That is a third argument for the stream format, alongside the 72x
storage saving and causal slicing.

**On `q_init`.** Track-On keeps `q_init` `(N, D)` outside the memory
(`trackon_predictor.py:95-96`): the birth-frame sample, re-fed as the decoder's input
query on *every* frame — the point's identity anchor, not part of its history. The replica
needs an equivalent, so it is saved (~46 KB).

For robot points there is a better option worth ablating. `presample(n, seed)` is
deterministic, so **FK point *n* is the same physical spot on the robot in every clip and
every episode**. The anchor can therefore be a *learned embedding indexed by FK point
index* — the model learns what "left fingertip" means once, across the whole dataset. An
object point cannot do this; its k-means index is arbitrary per clip.

---

## 5. What to fix before generating 14k clips

Generating the full set locks in choices, so these come first:

1. **Frame rate** (`DECISIONS.md` §6). We are at **3.6 Hz** — inherited from MolmoMotion,
   whose tracks we no longer use as labels. A control action spanning ~280 ms is coarse.
   **7.2 Hz** aligns with PointWorld's own windows and doubles both storage and yield.
   This is the single most expensive thing to get wrong: it invalidates the whole 78 GB.
2. **`16_run_trackon.py` does not save the memory tokens.** It writes
   `tracks / visibility / queries` only. The token stream is the main artifact and needs
   adding — the memory is a live attribute on `Predictor`, so this is a few lines.
3. **`19_robot_points.py`: ~1 clip in 8 fails** with "no FK step at frame N" — the birth
   frame does not land on a PointWorld-covered step. Indexing bug, not a method problem.
4. **`support_grid_size`** (`DECISIONS.md` §4). Currently 0. **This is two questions, and
   only one of them is Fatma hoca's.**

   The flag drops a uniform 20×20 grid (400 points) on frame 0 and tracks it alongside
   ours (`trackon_predictor.py:211-217`). The grid's own trajectories are discarded —
   `pred_trajectory` covers only the original N — so it exists purely to help *our* points
   track, via `query_attention[i](q_t, q_t, q_t)`, the self-attention across points. Note
   that **Track-On's own eval uses grid=20** (`evaluation/eval.py:70`); their published
   numbers assume it. We run 0.

   - *(a) Does the grid improve tracking of our points?* **A measurement, not a decision.**
     Grid 0 vs 20 on the same clips, scored by drift against FK ground truth on the robot
     points. Just run it.
   - *(b) Should the policy **see** the grid points as state?* **The real question.** The
     two come apart: we can track with grid=20 and slice the grid rows off before saving,
     so (a) costs nothing in storage. Keeping them costs 5× (78 GB → 390 GB) and hands the
     decoder 400 points we did not choose — but they are precisely the motion-unbiased
     background coverage the object-only set lacks, i.e. the free fix for the sampler bias
     we blamed for PointWorld's "always predicts motion" failure. Worth putting to her in
     those terms.

   When slicing, use `tracking_to_original` (`trackon_predictor.py:231-249`) rather than
   assuming rows `0..N_orig` are ours — row order follows the order queries are *born*.
5. **Object-point quality is 59.5%** attached recall (`RESULTS.md`). Roughly 40% of clips
   will have object points on something other than the manipulated object. For a *world
   model* that was fatal; for a *policy* it may be survivable, since the action signal
   rides on the robot points and the object points are context. But it should be stated
   honestly in whatever we report, not quietly averaged away.

**Recommended order:** ~~settle (1) and (4)~~ (both settled 2026-08-14) → fix (2) and (3)
→ generate query points for all 14,083 → run the linear probe → then generate the token
stream.

### Settled 2026-08-14

- **Frame rate: 7.2 Hz.** `DECISIONS.md` §6 carries the reasoning and the costs. Storage
  becomes ~13 MB/clip, **182 GB** for 14,083. `video_io.reconstruct` goes
  `not(mod(n,4))` → `not(mod(n,2))`; the `rel = step // 2` mapping in
  `19_robot_points.py` / `20_robot_points_sam_filter.py` becomes the identity and should
  be deleted rather than left as a no-op.
- **Support grid: not in the memory.** The keep-it argument was a world-model argument
  (static coverage, sampler bias) and we are not building a world model. Still worth
  measuring whether grid=20 helps *tracking* — that is separable, since the grid rows can
  be sliced off before saving.

Item (3), the `19_robot_points.py` failure, is **not** fixed by the rate change — it is
independent. `keep_step = (rel < T) & filled` (line 95) never requires that any FK point
is actually visible at the chosen step, so `s0 = argmax(keep_step)` can land on a step
where `ok[s0]` is all-False; script 19 then writes an empty query set and script 20
reports "no FK step at frame N". The fix is `keep_step = (rel < T) & filled &
ok.any(axis=1)`.

---

## 6. Layout proposed for `/frozen/avg/ozan/action_decoder_data/`

```
action_decoder_data/
  README.md                      # provenance: commit-less, so record script + args + date
  manifest.json                  # 14,083 entries: video_id, T, N_obj, N_rob, birth frames
  query_points/<vid>.npz         # object (SAM+kmeans) + robot (FK), roles, FK ground truth
  tokens/<vid>.npz               # scene_tokens (T,N_obj,256) fp16, robot_uv (T,N_rob,2),
                                 #   tracks (T,N_obj,2), visibility (T,N_obj)
  actions/<vid>.npz              # gripper_pose (T,7) XYZW, joint_velocities (T,7),
                                 #   gripper_open (T,), valid mask
  splits.json                    # train/val by SCENE, not by clip — DROID has many clips
                                 #   per scene and a clip-level split leaks
```

Two things easy to get wrong and expensive to discover late: the **XYZW** quaternion order
in `gripper_pose` (measured: 0.00° vs 166.31° for WXYZ), and splitting by **scene** rather
than by clip.
