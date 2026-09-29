# droid_qp/ — the Python package

Scripts import from here. The name is historical ("DROID query points", the project's first
phase); the package now also holds the whole policy.

## Configuration

| module | what it holds |
|---|---|
| `paths.py` | every file location, each overridable by an environment variable. **Never hard-code a path; add it here** |
| `rates.py` | frame rates and the mapping between frame spaces. The dataset is 7.2 Hz, every second DROID step |

## The policy — what you'll mostly touch

| module | what it does |
|---|---|
| `policy_data.py` | the dataloader: one 48-frame window plus 23 warm-up frames of scene tokens, robot tokens and action targets |
| `action_decoder.py` | the model: `ActionMemory` (the replica layers and their FIFO), `ActionReadout` (k learned queries → action tokens), `ActionHead`, and the 6D-rotation decoder |
| `policy_proprio.py` | optional late fusion of joint angles, velocities and gripper width |
| `policy_loss.py` | the scaled, quality-weighted behaviour-cloning loss, and the scale computation |
| `policy_train.py` | `ActionPolicy` (the full model), optimizer, schedule, checkpoints, overfit-window selection |
| `policy_eval.py` | validation metrics, the copy-last baseline, gripper-event matching |
| `policy_rollout.py` | chains predicted actions over time for the rollout evaluation |

## Building the dataset (used by `pipeline/`)

| module | what it does |
|---|---|
| `index.py` | joins DROID annotations to the raw videos and trajectories |
| `video_io.py` | reconstructs the video at our frame rate from the raw MP4 |
| `grounding.py`, `segment.py`, `sampling.py` | finding the object, SAM 2.1 masks, k-means query points |
| `robot_points.py` | gripper points via forward kinematics, projected to pixels |
| `actions.py` | action labels and proprioception from the recorded trajectory |
| `geom.py` | camera projection and the rigid-motion test |
| `attached.py`, `reference.py`, `viz.py` | point-scoring helpers, MolmoMotion's released tracks, overlays |

## Conventions that bite if you get them wrong

- **Masks are inverted between the loader and the model.** The loader's `robot_pad` is
  `True` for a *real* point; both attention modules take `True` for *padding*. It is inverted
  exactly once, in `ActionPolicy.forward` (`policy_train.py`).
- **The action target is 10 numbers per frame:** `[0:3]` translation in metres, `[3:9]` rotation
  as the first two **columns** of the rotation matrix, `[9]` whether the gripper is open at the
  next frame (0/1). Translation and rotation are in the body frame of the wrist flange
  (`panda_link8`), not the fingertips.
- **Gripper open** means DROID's recorded gripper position is below 0.10 (0 = fully open, larger =
  more closed). This matches PointWorld's own flag on 100% of frames.
- **Tokens** are Track-On's 256-number state per point per frame, stored as fp16: 100 object
  points, and up to 128 gripper points padded to 128.
- **Memory sizes:** the scene memory is frames t−23 … t; the action FIFO holds the last 24
  outputs. Both are the same in training and inference. The 48-frame window is only the
  training unroll length.
- **Everything is causal.** No module may read frame t+1 to predict frame t; `checks/` tests this.
