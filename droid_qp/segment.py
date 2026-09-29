"""Object segmentation from a point prompt, using SAM 2.1.

MolmoMotion used SAM 3 with a *text* prompt, then kept the smallest returned mask with
any pixel within 20 px of the MolmoPoint point. `facebook/sam3` is gated (gated=manual,
downloads 403), so we use `facebook/sam2.1-hiera-large` (ungated) and prompt it with the
MolmoPoint point directly. That drops the text hop entirely; the mask-choice rule is kept
in spirit -- SAM 2 returns 3 candidates for one point and we take the smallest one above
an area floor, which is MolmoMotion's "most specific mask" preference.

Swapping in SAM 3 if access is granted
--------------------------------------
Only the *weights* are gated; the code is already vendored at
`molmo-motion/data_generation/third_party/sam3/sam3/model_builder.py`
(`build_sam3_image_model`, `build_sam3_video_predictor`). transformers 4.57.6 has no
`Sam3VideoModel`, so the vendored builder is the route, not `from_pretrained`.

What would change, and nothing else:
  - the two lines that produce `masks` (a (K, H, W) bool stack) from a prompt;
  - the prompt itself becomes text, which needs the noun phrase -- and for DROID that
    means the Molmo2-8B re-caption branch for 84.3% of clips.
`choose_mask()` below is written against a plain (K, H, W) stack precisely so it is
shared by both backends; with a text prompt, gate candidates on `point_hits_mask()`
first to reproduce MolmoMotion's exact rule.

Worth noting: text prompting is not obviously the better path here. A point prompt needs
no caption at all, so it sidesteps the vague-caption problem entirely. Treat SAM 3 as an
A/B to run, not an upgrade to wait for.
"""
from __future__ import annotations

import numpy as np
import torch

SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"

# MolmoMotion's tolerance when matching a point to a mask (querypoints_from_video.py:329)
POINT_RADIUS = 20


def load_sam2(model_id: str = SAM2_MODEL_ID, device: str = "cuda", dtype=torch.float32):
    from transformers import Sam2Model, Sam2Processor

    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return model, processor


@torch.inference_mode()
def segment_from_point(
    model,
    processor,
    frame_rgb: np.ndarray,
    point_xy: tuple[float, float],
    min_area_frac: float = 0.002,
    max_area_frac: float = 0.40,
    mask_choice: str = "best_iou",
) -> tuple[np.ndarray | None, dict]:
    """Segment the object under `point_xy`.

    Returns (mask_bool (H,W) or None, info). The area window rejects the two common
    failure modes of a bare point prompt: a few-pixel speckle, and the whole
    table/background swallowing the object.
    """
    h, w = frame_rgb.shape[:2]
    device = next(model.parameters()).device

    inputs = processor(
        images=frame_rgb,
        input_points=[[[[float(point_xy[0]), float(point_xy[1])]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs, multimask_output=True)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0]
    masks = masks.squeeze(0).numpy().astype(bool)  # (3, H, W)
    scores = outputs.iou_scores.squeeze().float().cpu().numpy().reshape(-1)

    return choose_mask(masks, scores, min_area_frac, max_area_frac, mask_choice)


MASK_CHOICES = ("smallest", "largest", "best_iou")


def choose_mask(
    masks: np.ndarray,
    scores: np.ndarray | None = None,
    min_area_frac: float = 0.002,
    max_area_frac: float = 0.40,
    mask_choice: str = "best_iou",
) -> tuple[np.ndarray | None, dict]:
    """Pick one mask from a (K, H, W) bool stack. Backend-agnostic (SAM 2 or SAM 3).

    `mask_choice` matters more than it looks:

      "smallest"  MolmoMotion's rule. Correct for SAM 3 *text* prompts, where each
                  returned mask is a whole object instance and the smallest is the
                  most specific instance. WRONG for SAM 2 *point* prompts: SAM 2's
                  three candidates are deliberately subpart / part / whole, so
                  "smallest" systematically returns a subpart. Measured on 9 clips,
                  this loses reference points on elongated or multi-part objects.
      "largest"   whole-object end of the same axis; risks swallowing the table.
      "best_iou"  SAM 2's own predicted mask quality. Default.

    The area window is applied first in every case.
    """
    if len(masks) == 0:
        return None, {"candidates": [], "n_valid": 0, "reason": "no candidate masks"}

    h, w = masks.shape[-2:]
    total = float(h * w)
    cands = []
    for i, m in enumerate(masks):
        area = int(m.sum())
        cands.append({
            "idx": i, "area": area, "area_frac": area / total,
            "iou": float(scores[i]) if scores is not None else None,
        })

    ok = [c for c in cands if min_area_frac <= c["area_frac"] <= max_area_frac]
    info = {"candidates": cands, "n_valid": len(ok)}
    if not ok:
        info["reason"] = "all candidate masks outside the area window"
        return None, info

    if mask_choice == "smallest":
        best = min(ok, key=lambda c: c["area"])
    elif mask_choice == "largest":
        best = max(ok, key=lambda c: c["area"])
    elif mask_choice == "best_iou":
        best = max(ok, key=lambda c: (c["iou"] if c["iou"] is not None else -c["area"]))
    else:
        raise ValueError(f"mask_choice must be one of {MASK_CHOICES}, got {mask_choice!r}")

    info["chosen"] = best
    info["mask_choice"] = mask_choice
    return masks[best["idx"]], info


@torch.inference_mode()
def segment_from_points(
    model,
    processor,
    frame_rgb: np.ndarray,
    points_xy: np.ndarray,
    min_area_frac: float = 0.002,
    max_area_frac: float = 0.40,
    mask_choice: str = "largest",
) -> tuple[np.ndarray | None, dict]:
    """Segment from SEVERAL positive prompts at once.

    A single point is ambiguous -- SAM answers with subpart / part / whole and we have to
    guess which was meant. Multiple points spread over the target remove the ambiguity:
    every returned mask must contain all of them, so SAM cannot collapse to a sub-part.

    This is why robot points can be segmented more reliably than object points. For an
    object we have one guessed location; for the robot, FK gives us ~128 exact locations
    on the gripper, so we can hand SAM a dozen of them. Measured failure without this: a
    0.6%-of-frame sliver on one strip of the gripper, rejecting 90 points that were
    correctly on the robot.

    `mask_choice` defaults to "largest" here rather than "best_iou": with the extent
    already pinned down by the prompts, the remaining choice is how much of the
    articulated gripper to include, and more is better.
    """
    h, w = frame_rgb.shape[:2]
    device = next(model.parameters()).device
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)

    inputs = processor(
        images=frame_rgb,
        input_points=[[[[float(x), float(y)] for x, y in pts]]],
        input_labels=[[[1] * len(pts)]],
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs, multimask_output=True)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0]
    masks = masks.squeeze(0).numpy().astype(bool)
    scores = outputs.iou_scores.squeeze().float().cpu().numpy().reshape(-1)
    if masks.ndim == 2:
        masks = masks[None]
    return choose_mask(masks, scores, min_area_frac, max_area_frac, mask_choice)


def spread_prompts(points_xy: np.ndarray, k: int = 12) -> np.ndarray:
    """Pick k well-separated points, so the prompts span the target rather than clumping.

    Farthest-point sampling: start from the medoid, then repeatedly add the point
    furthest from everything chosen so far.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) <= k:
        return pts
    c = pts.mean(axis=0)
    idx = [int(np.argmin(((pts - c) ** 2).sum(axis=1)))]
    d = ((pts - pts[idx[0]]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, ((pts - pts[j]) ** 2).sum(axis=1))
    return pts[idx]


def point_hits_mask(mask_bool: np.ndarray, point_xy, radius: int = POINT_RADIUS) -> bool:
    """MolmoMotion's relaxed containment test (querypoints_from_video.py:336-345)."""
    h, w = mask_bool.shape
    x = int(np.clip(round(point_xy[0]), 0, w - 1))
    y = int(np.clip(round(point_xy[1]), 0, h - 1))
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    return bool(mask_bool[y0:y1, x0:x1].any())
