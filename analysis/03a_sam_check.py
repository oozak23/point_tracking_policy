#!/usr/bin/env python3
"""Step 3a -- validate the SAM 2.1 + k-means half, decoupled from grounding.

The full chain is  MolmoPoint -> SAM 2.1 -> k-means. This script swaps MolmoPoint for
a known-good point taken from MolmoMotion's *released* tracks, so segmentation and
sampling can be tested and looked at before the grounding models are ready.

Released tracks are in release pixel space (854x480) -- measured over 200 clips,
99.7% of visible observations fall inside those bounds -- so they overlay directly on
the reconstructed release video.

The metric that matters: if SAM 2.1 segmented the right object, MolmoMotion's own
object points should land INSIDE our mask. That is a genuine external check, not a
self-consistency one.

    python analysis/03a_sam_check.py --n 9
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp import segment, video_io, viz
from droid_qp.index import load_manifest
from droid_qp.sampling import kmeans_sample

TRACKS_TAR = Path("/frozen/avg/ozan/molmo-motion-1m/droid/tracks/tracks-0000.tar")
from droid_qp.paths import DATA  # noqa: E402
OUT = DATA  # the dataset root: droid_qp/paths.py
def iter_reference_tracks(tar_path: Path, wanted: set[str], limit: int):
    """Stream _2d.npz members out of the frozen tar. /frozen is read-only: never extract."""
    out = []
    with tarfile.open(tar_path) as t:
        for m in t:
            if not m.name.endswith("_2d.npz"):
                continue
            vid = Path(m.name).name[: -len("_2d.npz")]
            if vid not in wanted:
                continue
            d = np.load(io.BytesIO(t.extractfile(m).read()))
            out.append((vid, d["tracks_2d"], d["visibility"]))
            if len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--kmeans-k", type=int, default=100)
    ap.add_argument("--mask-choice", default="best_iou", choices=list(segment.MASK_CHOICES),
                    help="rule for picking among SAM 2's 3 candidate masks")
    ap.add_argument("--ab-mask-choice", action="store_true",
                    help="score all three rules on the same clips and print a comparison")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    recs = {r.video_id: r for r in load_manifest()}
    print(f"[sam-check] {len(recs)} resolvable clips; streaming reference tracks ...")
    refs = iter_reference_tracks(TRACKS_TAR, set(recs), args.n)
    print(f"[sam-check] {len(refs)} clips with released 2D tracks")

    vdir, pdir = args.out / "release_videos", args.out / "sam_check_png"
    vdir.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)

    print("[sam-check] loading SAM 2.1 ...")
    sam, sam_proc = segment.load_sam2()

    tiles, report = [], []
    for i, (vid, tracks, vis) in enumerate(refs, 1):
        rec = recs[vid]
        print(f"\n[{i}/{len(refs)}] {vid}")
        print(f"  caption: {rec.caption!r}   T={tracks.shape[0]} N={tracks.shape[1]}")
        row = {"video_id": vid, "caption": rec.caption, "category": rec.category,
               "T": int(tracks.shape[0]), "N_ref": int(tracks.shape[1])}
        try:
            mp4 = video_io.reconstruct(rec.source_mp4, vdir / f"{vid}.mp4")

            # Use the frame where the most reference points are visible.
            fidx = int(np.argmax(vis.sum(axis=1)))
            pts = tracks[fidx][vis[fidx]]
            if len(pts) == 0:
                row["status"] = "no_visible_ref_points"
                report.append(row)
                continue
            # Median is robust to the odd escaped track; mean is not.
            prompt = (float(np.median(pts[:, 0])), float(np.median(pts[:, 1])))
            row.update({"frame": fidx, "prompt": list(prompt), "n_ref_visible": int(len(pts))})
            print(f"  frame {fidx}, {len(pts)} ref points visible, "
                  f"prompt=({prompt[0]:.0f}, {prompt[1]:.0f})")

            frame = video_io.read_frame(mp4, fidx)

            if args.ab_mask_choice:
                h, w = frame.shape[:2]
                xs = np.clip(pts[:, 0].round().astype(int), 0, w - 1)
                ys = np.clip(pts[:, 1].round().astype(int), 0, h - 1)
                row["ab"] = {}
                for rule in segment.MASK_CHOICES:
                    m, inf = segment.segment_from_point(
                        sam, sam_proc, frame, prompt, mask_choice=rule)
                    row["ab"][rule] = (
                        None if m is None
                        else {"recall": float(m[ys, xs].mean()),
                              "area_frac": inf["chosen"]["area_frac"]}
                    )
                for rule, v in row["ab"].items():
                    txt = "no mask" if v is None else \
                        f"recall {v['recall']*100:5.1f}%  area {v['area_frac']*100:5.2f}%"
                    print(f"    {rule:10s} {txt}")

            mask, info = segment.segment_from_point(
                sam, sam_proc, frame, prompt, mask_choice=args.mask_choice)
            if mask is None:
                row["status"] = "no_mask"
                row["reason"] = info.get("reason")
                row["candidate_area_fracs"] = [c["area_frac"] for c in info["candidates"]]
                areas = ", ".join("%.1f%%" % (c["area_frac"] * 100) for c in info["candidates"])
                print(f"  -> no mask: {info.get('reason')} (candidate areas: {areas})")
                tiles.append(viz.seed_frame_png(frame, None, None, prompt,
                                                rec.caption, "NO MASK"))
                report.append(row)
                continue

            qp = kmeans_sample(mask, args.kmeans_k, frame_idx=fidx)

            # External check: do MolmoMotion's own object points land in our mask?
            h, w = mask.shape
            xs = np.clip(pts[:, 0].round().astype(int), 0, w - 1)
            ys = np.clip(pts[:, 1].round().astype(int), 0, h - 1)
            recall = float(mask[ys, xs].mean())

            row.update({"status": "ok", "mask_area_frac": info["chosen"]["area_frac"],
                        "n_query_points": int(len(qp)), "ref_points_in_mask": recall})
            print(f"  mask {info['chosen']['area_frac']*100:.1f}% of frame | "
                  f"{len(qp)} points | {recall*100:.0f}% of reference points inside mask")

            img = viz.overlay_mask(frame, mask)
            viz.draw_points(img, pts, (0, 255, 0), radius=3, filled=False)  # reference: green
            viz.draw_points(img, qp[:, 1:], viz.OBJECT_BGR, radius=2)      # ours: cyan
            import cv2

            cv2.drawMarker(img, (int(prompt[0]), int(prompt[1])), viz.PROMPT_BGR,
                           cv2.MARKER_CROSS, 14, 2)
            viz._banner(img, rec.caption,
                        f"ours={len(qp)} cyan / ref={len(pts)} green / recall={recall*100:.0f}%",
                        len(qp))
            cv2.imwrite(str(pdir / f"{vid}_f{fidx}.png"), img)
            tiles.append(img)
            report.append(row)

        except Exception as e:
            row["status"] = f"error: {type(e).__name__}: {e}"
            report.append(row)
            traceback.print_exc()

    if tiles:
        viz.contact_sheet(tiles, cols=3, out_path=args.out / "sam_check_sheet.png")
        print(f"\n[sam-check] contact sheet -> {args.out / 'sam_check_sheet.png'}")
    json.dump(report, open(args.out / "sam_check_report.json", "w"), indent=1)

    ok = [r for r in report if r.get("status") == "ok"]
    print("\n===== summary =====")
    print(f"clips attempted : {len(report)}")
    print(f"ok              : {len(ok)}")
    if ok:
        rc = np.array([r["ref_points_in_mask"] for r in ok])
        ar = np.array([r["mask_area_frac"] for r in ok])
        print(f"reference-point recall: median {np.median(rc)*100:.0f}%  "
              f"min {rc.min()*100:.0f}%  max {rc.max()*100:.0f}%")
        print(f"mask area (% of frame): median {np.median(ar)*100:.1f}%  "
              f"min {ar.min()*100:.1f}%  max {ar.max()*100:.1f}%")
        print(f"clips with recall >70%: {int((rc > 0.7).sum())}/{len(ok)}")

    if args.ab_mask_choice:
        print("\n----- mask-choice A/B (reference-point recall) -----")
        print(f"{'rule':12s} {'median':>8s} {'mean':>8s} {'>70%':>6s} {'no mask':>8s} "
              f"{'med area':>9s}")
        for rule in segment.MASK_CHOICES:
            vals = [r["ab"][rule] for r in report if "ab" in r]
            got = [v for v in vals if v is not None]
            if not got:
                print(f"{rule:12s} {'-':>8s}")
                continue
            rec = np.array([v["recall"] for v in got])
            ar = np.array([v["area_frac"] for v in got])
            print(f"{rule:12s} {np.median(rec)*100:7.1f}% {rec.mean()*100:7.1f}% "
                  f"{int((rec > 0.7).sum()):3d}/{len(rec):<2d} "
                  f"{len(vals)-len(got):8d} {np.median(ar)*100:8.2f}%")


if __name__ == "__main__":
    main()
