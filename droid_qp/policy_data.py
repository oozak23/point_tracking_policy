"""Dataloader for the action decoder. Step 1 of DECODER_BUILD.md.

This is the component where a silent mistake is unrecoverable: if a training sample ever
contains information from after the frame being predicted, the model trains beautifully
and cannot be deployed, and nothing crashes to tell you. Everything below is arranged so
that reading the future is structurally impossible rather than merely avoided, and
`scripts/45_verify_dataloader.py` asserts it empirically.

WHAT A SAMPLE IS

Not a single timestep. The action memory is recurrent -- the replica block writes its own
output into its own FIFO -- so training rolls forward over a window, as Track-On does
(`config/train.yaml`: T=48, M=24). A sample is therefore a contiguous slice of the token
STREAM, and the model maintains its own memory across it:

    stream of (M-1 + T_win) frames  ->  supervise the last T_win of them

Returning the stream rather than a stack of per-frame memories matters: the memory at
frame t is `stream[t-M+1 : t+1]`, and consecutive frames overlap by M-1, so materialising
them separately would multiply the data by 24x for no information.

    scene_stream (M-1+T_win, 100, 256)   object points -- frozen, read-only
    robot_stream (M-1+T_win, 128, 256)   robot points  -- input to the trainable replica

SHAPES ARE NOT ALL FIXED

`N_obj` is exactly 100 in every clip (k-means with k=100). `N_rob` varies 42-128, because
the SAM filter rejects a different number of gripper points per clip. So robot points are
padded to 128 and carry a mask; padding must never reach attention unmasked.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

N_OBJ = 100        # k-means k, identical in every clip
N_ROB_MAX = 128    # RobotSampler.presample(128); survivors vary, slots do not
D = 256

TARGET_KEYS = ("delta_pos_body", "delta_rot6d_body", "gripper_open_next")
TARGET_DIM = 3 + 6 + 1


def build_shard_index(gen: Path = OUT / "gen", cache: Path = OUT / "shard_index.json"):
    """video_id -> shard, so a clip's files can be found without globbing 141 dirs."""
    if cache.exists():
        return json.load(open(cache))
    idx = {}
    for p in sorted(gen.glob("*/trackon/tracks/*.npz")):
        idx[p.stem] = p.parents[2].name
    json.dump(idx, open(cache, "w"))
    return idx


class PolicyWindows:
    """Windows over the token stream, paired with action targets.

    Deliberately not a torch Dataset: it has no torch dependency, so it can be verified in
    any env. Wrap it in one where torch is available.
    """

    def __init__(self, split: str = "train", *, T_win: int = 48, M: int = 24,
                 manifest: Path = OUT / "dataset_manifest.json",
                 gen: Path = OUT / "gen", cache_clips: int = 8, seed: int = 0):
        man = json.load(open(manifest))
        self.clips = [c for c in man["clips"] if c["split"] == split]
        self.T_win, self.M, self.gen = T_win, M, gen
        self.shard = build_shard_index(gen)
        self.rng = np.random.default_rng(seed)
        self._cache: dict[str, tuple] = {}
        self._cache_order: list[str] = []
        self._cache_max = cache_clips
        missing = [c["video_id"] for c in self.clips if c["video_id"] not in self.shard]
        if missing:
            raise FileNotFoundError(f"{len(missing)} manifest clips absent from {gen}, "
                                    f"e.g. {missing[:3]}")

    def __len__(self):
        return len(self.clips)

    # ---- clip loading, with a small LRU so consecutive windows reuse the decode --------
    def _load(self, vid: str):
        if vid in self._cache:
            return self._cache[vid]
        sh = self.shard[vid]
        tk = np.load(self.gen / sh / "trackon" / "tracks" / f"{vid}.npz")
        ac = np.load(self.gen / sh / "actions" / "labels" / f"{vid}.npz")
        role = tk["role"]
        tokens = tk["tokens"]
        obj = tokens[:, role == 0]
        rob = tokens[:, role == 1]
        tgt = np.concatenate([ac[k] for k in TARGET_KEYS], axis=1).astype(np.float32)
        item = (obj, rob, tgt, ac["valid"].astype(bool))
        self._cache[vid] = item
        self._cache_order.append(vid)
        while len(self._cache_order) > self._cache_max:
            self._cache.pop(self._cache_order.pop(0), None)
        return item

    def sample(self, i: int, start: int | None = None) -> dict:
        """One window from clip `i`. `start` fixes the first SUPERVISED frame (for tests)."""
        c = self.clips[i]
        obj, rob, tgt, valid = self._load(c["video_id"])
        T = min(len(obj), len(tgt))
        Tw, M = self.T_win, self.M

        # The supervised span is [s, s+Tw). Clips are never shorter than Tw (measured
        # minimum T is 48), so this always has room.
        hi = max(T - Tw, 0)
        s = int(self.rng.integers(0, hi + 1)) if start is None else int(start)
        s = max(0, min(s, hi))

        # History is the M-1 frames before s. Near the clip start there are fewer, so pad
        # at the FRONT and mask -- keeping slot order oldest->newest, the order Track-On
        # writes and the order t_embedding expects.
        lo = s - (M - 1)
        pad = max(0, -lo)
        lo = max(0, lo)
        end = min(s + Tw, T)

        o = obj[lo:end]
        r = rob[lo:end]
        n_rob = r.shape[1]

        L = (M - 1) + Tw
        scene = np.zeros((L, N_OBJ, D), np.float16)
        robot = np.zeros((L, N_ROB_MAX, D), np.float16)
        scene[pad:pad + len(o)] = o
        robot[pad:pad + len(r), :n_rob] = r

        hist_valid = np.zeros(L, bool)
        hist_valid[pad:pad + len(o)] = True

        robot_pad = np.zeros(N_ROB_MAX, bool)
        robot_pad[:n_rob] = True

        y = np.zeros((Tw, TARGET_DIM), np.float32)
        m = np.zeros(Tw, bool)
        n = end - s
        y[:n] = tgt[s:end]
        m[:n] = valid[s:end]

        return {
            "video_id": c["video_id"], "start": s,
            "scene": scene,            # (L, 100, 256)
            "robot": robot,            # (L, 128, 256)
            "hist_valid": hist_valid,  # (L,)   False where front-padded
            "robot_pad": robot_pad,    # (128,) True where a real point
            "target": y,               # (Tw, 10)
            "target_valid": m,         # (Tw,)
            "weight": np.float32(c["weight"]),
            "n_rob": n_rob,
        }
