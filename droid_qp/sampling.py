"""K-means query-point sampling over an object mask.

`kmeans_sample` is a verbatim port of MolmoMotion's
`data_generation/third_party/sam3/querypoints_from_video.py:66-90`, so our points are
drawn the same way theirs were. `reproject_to_mask` is our addition and is OFF by
default: rounding a cluster centre can land outside a non-convex mask (cloth, handles,
mug rims). MolmoMotion does no such reprojection, so keeping it opt-in lets us A/B it.
"""
from __future__ import annotations

import numpy as np


def kmeans_sample(
    mask_bool: np.ndarray,
    k: int,
    frame_idx: int = 0,
    seed: int = 0,
    reproject_to_mask: bool = False,
) -> np.ndarray:
    """Sample up to `k` query points on a boolean mask.

    Returns (N, 3) int32 rows [frame_idx, x, y]. N < k when the mask has fewer
    than k pixels.
    """
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    coords = np.stack([xs, ys], axis=1)
    n_clusters = min(int(k), len(coords))

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(coords)
    centers = np.rint(km.cluster_centers_).astype(np.int32)

    if reproject_to_mask:
        centers = _snap_to_mask(centers, coords)

    frame_col = np.full((len(centers),), int(frame_idx), dtype=np.int32)
    return np.stack([frame_col, centers[:, 0], centers[:, 1]], axis=1)


def _snap_to_mask(centers: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Move each centre to the nearest mask pixel if it is not already on the mask."""
    on_mask = set(map(tuple, coords))
    out = centers.copy()
    for i, c in enumerate(centers):
        if tuple(c) in on_mask:
            continue
        d = ((coords - c) ** 2).sum(axis=1)
        out[i] = coords[int(np.argmin(d))]
    return out


def fraction_off_mask(points_xyt: np.ndarray, mask_bool: np.ndarray) -> float:
    """Diagnostic for the reproject_to_mask A/B: share of points landing off the mask."""
    if len(points_xyt) == 0:
        return 0.0
    h, w = mask_bool.shape
    x = np.clip(points_xyt[:, 1], 0, w - 1)
    y = np.clip(points_xyt[:, 2], 0, h - 1)
    return float((~mask_bool[y, x]).mean())
