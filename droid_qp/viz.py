"""Overlay rendering for query-point inspection.

Deliberately not reusing track_on/utils/vis_utils.py::plot_tracks_wo_tail -- it colours
points by birth-y and renders a matplotlib figure per frame, neither of which we want.
"""
from __future__ import annotations

import cv2
import numpy as np

OBJECT_BGR = (255, 255, 0)  # cyan
ROBOT_BGR = (255, 0, 255)  # magenta
PROMPT_BGR = (0, 0, 255)  # red
MASK_BGR = (0, 200, 255)  # amber


def overlay_mask(frame_rgb: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.35,
                 color=MASK_BGR) -> np.ndarray:
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
    if mask_bool is not None and mask_bool.any():
        tint = np.zeros_like(img)
        tint[:] = color
        img[mask_bool] = cv2.addWeighted(img, 1 - alpha, tint, alpha, 0)[mask_bool]
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, contours, -1, color, 1)
    return img


def draw_points(img_bgr: np.ndarray, pts_xy: np.ndarray, color, radius: int = 2,
                filled: bool = True) -> np.ndarray:
    for x, y in np.asarray(pts_xy).reshape(-1, 2):
        cv2.circle(img_bgr, (int(x), int(y)), radius, color, -1 if filled else 1,
                   lineType=cv2.LINE_AA)
    return img_bgr


def seed_frame_png(
    frame_rgb: np.ndarray,
    mask_bool: np.ndarray | None,
    object_points_xy: np.ndarray | None,
    prompt_xy: tuple[float, float] | None,
    caption: str = "",
    obj_phrase: str = "",
    out_path: str | None = None,
) -> np.ndarray:
    """Render the detection frame: mask tint, k-means points (cyan), Molmo point (red)."""
    img = overlay_mask(frame_rgb, mask_bool)

    if object_points_xy is not None and len(object_points_xy):
        draw_points(img, object_points_xy, OBJECT_BGR, radius=2)

    if prompt_xy is not None:
        px, py = int(prompt_xy[0]), int(prompt_xy[1])
        cv2.drawMarker(img, (px, py), PROMPT_BGR, cv2.MARKER_CROSS, 14, 2)
        cv2.circle(img, (px, py), 20, PROMPT_BGR, 1, lineType=cv2.LINE_AA)

    _banner(img, caption, obj_phrase,
            0 if object_points_xy is None else len(object_points_xy))

    if out_path:
        cv2.imwrite(str(out_path), img)
    return img


def _banner(img, caption, obj_phrase, n_points):
    h, w = img.shape[:2]
    lines = []
    if caption:
        lines.append(f'caption: "{caption}"')
    if obj_phrase:
        lines.append(f'object:  "{obj_phrase}"   N={n_points}')
    if not lines:
        return
    pad, lh = 6, 16
    box_h = pad * 2 + lh * len(lines)
    cv2.rectangle(img, (0, h - box_h), (w, h), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t[:110], (pad, h - box_h + pad + lh * (i + 1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def contact_sheet(images_bgr: list[np.ndarray], cols: int = 3,
                  cell_w: int = 560, out_path: str | None = None) -> np.ndarray:
    """Tile per-clip seed frames into one PNG so many examples can be scanned at once."""
    if not images_bgr:
        raise ValueError("no images")
    cells = []
    for im in images_bgr:
        s = cell_w / im.shape[1]
        cells.append(cv2.resize(im, (cell_w, int(round(im.shape[0] * s)))))
    ch = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, ch - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
             for c in cells]
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    if out_path:
        cv2.imwrite(str(out_path), sheet)
    return sheet
