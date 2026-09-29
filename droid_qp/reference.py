"""Read MolmoMotion's released 2D tracks, for scoring our query points against theirs.

These are a *reference*, never a label: they came from AllTracker lifted by ViPE
monocular depth, and 13-20% of trajectories teleport. We use them only to check that
our segmentation found the same object, via `ref_point_recall`.

Coordinates are release pixel space (854x854 wide, 480 high) -- measured over 200
clips, 99.7% of *visible* observations fall inside those bounds. Always mask by
`visibility` before using them; invisible points are extrapolated and can sit far
outside the frame.

/frozen is read-only shared storage: stream tar members, never extract.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np

TRACKS_TAR = Path("/frozen/avg/ozan/molmo-motion-1m/droid/tracks/tracks-0000.tar")


def load_reference_tracks(video_ids: set[str], tar_path: Path = TRACKS_TAR) -> dict:
    """-> {video_id: (tracks_2d (T,N,2) float32, visibility (T,N) bool)}."""
    out = {}
    if not Path(tar_path).exists():
        return out
    with tarfile.open(tar_path) as t:
        for m in t:
            if not m.name.endswith("_2d.npz"):
                continue
            vid = Path(m.name).name[: -len("_2d.npz")]
            if vid not in video_ids:
                continue
            d = np.load(io.BytesIO(t.extractfile(m).read()))
            out[vid] = (d["tracks_2d"], d["visibility"])
            if len(out) == len(video_ids):
                break
    return out


def ref_point_recall(mask_bool: np.ndarray, tracks: np.ndarray, vis: np.ndarray,
                     frame_idx: int) -> float | None:
    """Share of MolmoMotion's visible object points at `frame_idx` inside our mask.

    High recall means we segmented the object they segmented. It is deliberately
    one-directional: our mask may legitimately cover more than their point set, since
    theirs was thinned by their stages 5-6 down to 31-94 points.
    """
    if frame_idx >= len(tracks):
        return None
    pts = tracks[frame_idx][vis[frame_idx]]
    if len(pts) == 0:
        return None
    h, w = mask_bool.shape
    xs = np.clip(pts[:, 0].round().astype(int), 0, w - 1)
    ys = np.clip(pts[:, 1].round().astype(int), 0, h - 1)
    return float(mask_bool[ys, xs].mean())
