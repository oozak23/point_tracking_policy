#!/usr/bin/env python3
"""Keep only the FK robot points that land on the VISIBLE robot, using a SAM mask.

FK geometry is exact, but "inside the frame and in front of the camera" is not the same
as "visible". The Robotiq 2F-85 is an open skeletal frame, so points sampled on the far
side of a finger, or inside the mechanism, project onto background seen through the
gaps. Measured: many points sit off the robot even with gripper_only sampling.

Fix without giving up what FK buys us. SAM is seeded from the FK gripper position (exact,
no grounding model needed) to get the visible gripper mask, then FK points are FILTERED
by it. Because this is a filter rather than a replacement:

  * surviving points keep their FK index, so point *n* is still the same physical spot
    on the robot in every clip -- the property a fixed-dimension policy input needs, and
    the thing a SAM-sampled point set cannot provide
  * surviving points keep their FK trajectory, so the free per-frame ground truth for
    the drift measurement is preserved

Reads stage A's npz (fk_uv, fk_ok) so no pytorch_kinematics is needed here -- this runs
in the molmo-motion env, which has SAM but not the kinematics stack.

    python scripts/20_robot_points_sam_filter.py --robot-dir out/robot_points_grip
Run with /scratch/oozak23/envs/molmo-motion/bin/python
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp import segment, video_io

OUT = Path(__file__).resolve().parent.parent / "out"
NAME_RE = re.compile(r"^(?P<vid>.+)_f(?P<frame>\d+)\.npz$")


def choose_mask_frames(fk_uv, fk_ok, steps, mode: str, n_votes: int):
    """Frames to build the gripper mask on, best first.

    Score = how many FK points are in frame, times how spread out they are. Both matter:
    a count alone is satisfied by a gripper that is fully visible but tiny and far away,
    where SAM's mask is a few pixels and rejects almost everything. The spread term
    prefers frames where the gripper is CLOSE, which is where its silhouette is
    unambiguous.
    """
    if len(steps) == 0:
        return []
    n_in = fk_ok[steps].sum(axis=1).astype(np.float64)
    uv = fk_uv[steps]
    ok = fk_ok[steps]
    spread = np.zeros(len(steps))
    for k in range(len(steps)):
        p = uv[k][ok[k]]
        if len(p) > 2:
            spread[k] = np.sqrt(((p - p.mean(0)) ** 2).sum(1)).mean()
    score = n_in * (spread + 1.0)
    order = steps[np.argsort(-score)]
    if mode == "birth":
        return [int(steps[0])]
    if mode == "best":
        return [int(order[0])]
    # vote: take the top frames, but spaced out so they are not consecutive duplicates
    picked = []
    for s in order:
        if all(abs(int(s) - q) >= 5 for q in picked):
            picked.append(int(s))
        if len(picked) >= max(1, n_votes):
            break
    return picked


def medoid(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    return pts[int(np.argmin(((pts - c) ** 2).sum(axis=1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-dir", type=Path, default=OUT / "robot_points_grip")
    ap.add_argument("--videos-dir", type=Path, default=OUT / "mm_sam_f0" / "release_videos")
    ap.add_argument("--mask-choice", default="largest", choices=list(segment.MASK_CHOICES))
    ap.add_argument("--min-area-frac", type=float, default=0.003,
                    help="raised from 0.001: a 0.6%% sliver on one strip of the gripper "
                         "was being accepted, rejecting 90 points correctly on the robot")
    ap.add_argument("--n-prompts", type=int, default=12,
                    help="how many FK points to hand SAM as positive prompts. One point "
                         "is ambiguous (subpart/part/whole); several spread over the "
                         "gripper force a mask containing all of them.")
    ap.add_argument("--mask-frame", default="vote", choices=["birth", "best", "vote"],
                    help="which frame(s) to build the gripper mask on. 'birth' is the old "
                         "behaviour and couples mask quality to the birth frame; 'best' "
                         "picks the frame where the gripper is largest and fullest; "
                         "'vote' (default) evaluates several and keeps points a majority "
                         "agree on, which survives one bad mask.")
    ap.add_argument("--mask-votes", type=int, default=3,
                    help="frames to evaluate when --mask-frame vote. Each costs one SAM "
                         "call per clip, so this multiplies the GPU cost of this stage.")
    ap.add_argument("--out", type=Path, default=OUT / "robot_points_sam")
    args = ap.parse_args()

    files = sorted((args.robot_dir / "query_points").glob("*.npz"))
    print(f"[rsam] {len(files)} clips from {args.robot_dir}")

    print("[rsam] loading SAM 2.1 ...")
    sam, sam_proc = segment.load_sam2()

    qdir, vdir = args.out / "query_points", args.out / "overlay"
    qdir.mkdir(parents=True, exist_ok=True)
    vdir.mkdir(parents=True, exist_ok=True)

    report = []
    for i, p in enumerate(files, 1):
        m = NAME_RE.match(p.name)
        vid, f0 = m["vid"], int(m["frame"])
        try:
            d = np.load(p)
            fk_uv, fk_ok = d["fk_uv"], d["fk_ok"]
            rel = d["fk_release_frame"]

            steps = np.where((rel == f0) & fk_ok.any(axis=1))[0]
            if len(steps) == 0:
                print(f"  [{i}] {vid[:44]} -- no FK step at frame {f0}")
                continue
            s0 = int(steps[0])
            vis = fk_ok[s0]
            if vis.sum() < 5:
                print(f"  [{i}] {vid[:44]} -- too few FK points in frame")
                continue

            mp4 = args.videos_dir / f"{vid}.mp4"
            if not mp4.exists():
                print(f"  [{i}] {vid[:44]} -- no video")
                continue

            # WHICH FRAME TO BUILD THE MASK ON.
            #
            # The filter answers a question that is NOT frame-specific -- which of the 128
            # FK indices sit on the VISIBLE gripper rather than on background seen through
            # the Robotiq's open frame. `presample(n, seed)` is deterministic, so index n
            # is the same physical spot at every frame, which means the mask may be built
            # wherever the gripper is clearest and the answer still applies to points born
            # at f0.
            #
            # Evaluating at the birth frame couples the two, and since robot points are
            # now born at frame 0 that is the arm's START position -- far, small, often
            # half out of view. Measured on the pilot: keep fraction fell to mean ~91%
            # (from 96.6%) with a long tail; one TRI clip kept 14 of 128 and one clip lost
            # its robot half entirely.
            # Bound candidates by the VIDEO length. Reading joints from DROID gives FK
            # every trajectory step, which can exceed the reconstructed video by a frame
            # or two (measured: S=356 against T=355). 19_robot_points.py already bounds
            # the BIRTH frame this way; the mask frame needs the same bound or
            # `read_frame` fails on a frame that does not exist.
            T_vid = video_io.frame_count(mp4)
            if T_vid <= 0:
                print(f"  [{i}] {vid[:44]} -- could not read frame count")
                continue
            steps_ok = np.where((fk_ok.sum(axis=1) >= 5) & (rel < T_vid))[0]
            cand = ([s0] if args.mask_frame == "birth"
                    else choose_mask_frames(fk_uv, fk_ok, steps_ok, args.mask_frame,
                                            args.mask_votes) or [s0])
            H = W = None
            votes = None
            seen = None
            info = None
            ov = None
            for s_m in cand:
                fr = video_io.read_frame(mp4, int(rel[s_m]))
                H, W = fr.shape[:2]
                vm = fk_ok[s_m]
                if vm.sum() < 5:
                    continue
                pr = segment.spread_prompts(fk_uv[s_m][vm].astype(np.float64),
                                            k=args.n_prompts)
                mk, inf = segment.segment_from_points(
                    sam, sam_proc, fr, pr, min_area_frac=args.min_area_frac,
                    mask_choice=args.mask_choice)
                if mk is None:
                    continue
                um = fk_uv[s_m]
                xm = np.clip(np.rint(um[:, 0]).astype(int), 0, W - 1)
                ym = np.clip(np.rint(um[:, 1]).astype(int), 0, H - 1)
                v = (vm & mk.astype(bool)[ym, xm]).astype(np.int32)
                votes = v if votes is None else votes + v
                seen = vm.astype(np.int32) if seen is None else seen + vm
                info = inf
                if ov is None:               # first (best-scoring) frame, for the overlay
                    ov = (mk.astype(bool), int(rel[s_m]), fr, pr, vm)
            if votes is None:
                print(f"  [{i}] {vid[:44]} -- SAM produced no usable mask")
                continue

            # A point counts as on-robot if the majority of evaluated frames said so.
            # With one frame this reduces to the old behaviour exactly.
            # Majority over the frames where the point was actually IN FRAME. A point
            # never in frame at any mask frame has no evidence against it -- FK says it is
            # on the robot and the mask exists only to remove background seen through the
            # gripper's open skeleton, so the benefit of the doubt is to keep it.
            with np.errstate(invalid="ignore"):
                maj = np.where(seen > 0, votes * 2 >= seen, True)
            on_robot = vis & maj
            # Query points are still emitted at the BIRTH frame; only the decision about
            # which FK indices are on the visible gripper came from elsewhere.
            uv = fk_uv[s0]
            mask, mask_t, frame, prompts, vis_m = ov
            H, W = frame.shape[:2]
            xi = np.clip(np.rint(uv[:, 0]).astype(int), 0, W - 1)
            yi = np.clip(np.rint(uv[:, 1]).astype(int), 0, H - 1)

            kept = np.where(on_robot)[0]
            if len(kept) < 5:
                print(f"  [{i}] {vid[:44]} -- only {len(kept)} points survived the mask")
                continue

            qp = np.stack([np.full(len(kept), f0), np.rint(uv[kept, 0]),
                           np.rint(uv[kept, 1])], axis=1).astype(np.int32)
            np.savez(qdir / f"{vid}_f{f0}.npz", query_points=qp,
                     dim=np.array([H, W], dtype=np.int32), mask=mask,
                     point_index=kept.astype(np.int32),     # FK index -> stable identity
                     fk_uv=fk_uv, fk_ok=fk_ok, fk_release_frame=rel)

            # Drawn on the MASK frame, not the birth frame -- that is where the accept /
            # reject decision was actually made, so it is what a reviewer needs to see.
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
            tint = np.zeros_like(img); tint[:] = (0, 200, 255)
            img[mask] = cv2.addWeighted(img, 0.65, tint, 0.35, 0)[mask]
            s_show = int(np.argmin(np.abs(rel - mask_t)))
            uvm = fk_uv[s_show]
            xm = np.clip(np.rint(uvm[:, 0]).astype(int), 0, W - 1)
            ym = np.clip(np.rint(uvm[:, 1]).astype(int), 0, H - 1)
            for k in np.where(vis_m & ~on_robot)[0]:                # rejected
                cv2.circle(img, (xm[k], ym[k]), 3, (0, 0, 255), 1, cv2.LINE_AA)
            for k in np.where(vis_m & on_robot)[0]:                 # kept
                cv2.circle(img, (xm[k], ym[k]), 2, (0, 255, 0), -1, cv2.LINE_AA)
            for (sx, sy) in prompts:
                cv2.drawMarker(img, (int(sx), int(sy)), (255, 0, 255),
                               cv2.MARKER_CROSS, 12, 2)
            cv2.rectangle(img, (0, H - 24), (W, H), (0, 0, 0), -1)
            cv2.putText(img, f"born f{f0}, mask from f{mask_t} ({len(cand)} frame"
                             f"{'s' if len(cand) > 1 else ''})  kept={len(kept)}"
                             f"  rejected={int((vis & ~on_robot).sum())}"
                             f"  mask={info['chosen']['area_frac']*100:.1f}%",
                        (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imwrite(str(vdir / f"{vid}.png"), img)

            keep_frac = float(len(kept) / max(int(vis.sum()), 1))
            print(f"  [{i:2d}] {vid[:44]:44s} f{f0:<3d} in-frame={int(vis.sum()):3d} "
                  f"kept={len(kept):3d} ({keep_frac*100:4.1f}%) "
                  f"mask={info['chosen']['area_frac']*100:4.1f}%")
            report.append({"video_id": vid, "frame": f0, "n_in_frame": int(vis.sum()),
                           "n_kept": int(len(kept)), "keep_frac": keep_frac,
                           "mask_area_frac": info["chosen"]["area_frac"]})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [{i}] {vid[:44]} -- {type(e).__name__}: {e}")

    json.dump(report, open(args.out / "report.json", "w"), indent=1)
    print("\n===== summary =====")
    print(f"clips        : {len(report)}")
    if report:
        kf = np.array([r["keep_frac"] for r in report])
        nk = np.array([r["n_kept"] for r in report])
        print(f"kept fraction: mean {kf.mean()*100:.1f}%  min {kf.min()*100:.1f}%")
        print(f"points kept  : median {np.median(nk):.0f}")
    print(f"[rsam] -> {args.out}")


if __name__ == "__main__":
    main()
