#!/usr/bin/env python3
"""Step 3 -- put query points on the manipulated object, MolmoMotion-style.

Per clip:
    reconstruct release video (854x480 @ 15 fps, every 4th source frame)
    caption -> Qwen3 noun phrase                       [cached by scripts/02]
    MolmoPoint-Vid-4B "point to <obj> gripped and picked up by the robot gripper"
        -> (detection_frame, x, y)
    SAM 2.1 point prompt at (x, y)                     -> object mask
    k-means over mask pixels (k=100)                   -> query points (N,3) [t, x, y]

Writes per clip:
    out/query_points/<video_id>_f<frame>.npz   query_points (N,3) int32, dim (2,) [H,W]
    out/seed_png/<video_id>_f<frame>.png       mask + points + Molmo point overlay
and one out/contact_sheet.png tiling every clip processed.

Only clips whose caption yields a NON-vague noun are handled here -- those need no
Molmo2-8B, so the whole thing fits on the 16 GB login A4000.

    python scripts/03_object_points.py --n 12
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp import grounding, rates, segment, video_io, viz
from droid_qp.index import load_manifest
from droid_qp.sampling import fraction_off_mask, kmeans_sample

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12,
                    help="number of clips; 0 = ALL. At scale each shard holds 74-123 clips,\n"
                         "so a forgotten --n would silently process only the default and\n"
                         "still report success -- hence 0 means all, matching 24_action_labels.py.")
    ap.add_argument("--kmeans-k", type=int, default=100)
    ap.add_argument("--kmeans-seed", type=int, default=0)
    ap.add_argument("--reproject-to-mask", action="store_true",
                    help="snap k-means centres onto the mask (MolmoMotion does not)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--category", default=None, help="restrict to one clips.json category")
    ap.add_argument("--mask-choice", default="best_iou", choices=list(segment.MASK_CHOICES))
    ap.add_argument("--text-free", action="store_true",
                    help="prompt MolmoPoint with TEXT_FREE_PROMPT -- no caption, no noun, "
                         "no Qwen3, no Molmo2-8B. Usable on all 23,532 clips.")
    ap.add_argument("--compare-ref", action="store_true",
                    help="score against MolmoMotion's released tracks where available")
    ap.add_argument("--ground-in-motion-window", action="store_true",
                    help="run MolmoPoint on the motion window only, not the whole clip. "
                         "Fixes text-free grounding at frame 0, where nothing is yet in "
                         "contact. Chooses where to LOOK, does not filter training data.")
    ap.add_argument("--seed-from-npz", type=Path, default=None,
                    help="Take the SAM seed from an existing query-point dir (e.g. "
                         "out/mm_queries/query_points) instead of running MolmoPoint. "
                         "Used to seed from MolmoMotion's released points, whose "
                         "grounding measured better than our text-free MolmoPoint "
                         "(50.3%% vs 38.4%% attached recall on the same 60 clips). "
                         "Keeps SAM + k-means, so N is a fixed 100 rather than their "
                         "variable 31-94, and drops the 4B model from the loop.")
    ap.add_argument("--geometry-correct", action="store_true",
                    help="re-seed SAM from the attached cluster when the mask badly "
                         "misses what actually moves with the gripper. Only fires where "
                         "PointWorld supplies a cluster (~45% of clips); otherwise "
                         "MolmoPoint's answer is kept unchanged.")
    ap.add_argument("--correct-below", type=float, default=0.3,
                    help="re-seed only when attached-recall is below this. Deliberately "
                         "not 'whenever they disagree': for 'pick the pen from the cup' "
                         "the cup may move too, so the cluster can legitimately be "
                         "larger than the target object.")
    ap.add_argument("--video-ids", default=None,
                    help="comma-separated video_ids to run (overrides sampling)")
    ap.add_argument("--video-ids-from", type=Path, default=None,
                    help="JSON report to take video_ids from (e.g. out/sam_check_report.json). "
                         "Reuses already-reconstructed videos and clips with reference tracks.")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    nouns = {}
    if not args.text_free:
        nouns_path = OUT / "caption_nouns.json"
        if not nouns_path.exists():
            sys.exit(f"missing {nouns_path} -- run scripts/02_extract_nouns.py first, "
                     f"or pass --text-free")
        nouns = json.load(open(nouns_path))

    recs = load_manifest()
    if args.category:
        recs = [r for r in recs if r.category == args.category]
    if args.text_free:
        usable = recs
        print(f"[obj] text-free mode: all {len(usable)} clips usable (no caption needed)")
    else:
        # Recomputed from the cached noun rather than trusting a cached flag: the
        # hallucination guard was added after the first cache was built.
        def direct_ok(r):
            e = nouns.get(r.caption)
            return e is not None and not grounding.needs_recaption(e["noun"], r.caption)

        usable = [r for r in recs if direct_ok(r)]
        n_vague = sum(1 for r in recs if r.caption in nouns
                      and grounding.is_vague(nouns[r.caption]["noun"]))
        n_hall = sum(1 for r in recs if r.caption in nouns
                     and grounding.is_hallucinated(nouns[r.caption]["noun"], r.caption))
        print(f"[obj] {len(recs)} resolvable clips | usable={len(usable)} "
              f"vague={n_vague} hallucinated={n_hall}")
        if not usable:
            sys.exit("no clips with a caption-grounded object noun -- all need Molmo2-8B")

    random.seed(args.seed)
    if args.video_ids:
        by_id = {r.video_id: r for r in usable}
        chosen = [by_id[v] for v in args.video_ids.split(",") if v in by_id]
        print(f"[obj] selected {len(chosen)} clips by explicit id")
    elif args.video_ids_from:
        wanted = [r["video_id"] for r in json.load(open(args.video_ids_from))]
        by_id = {r.video_id: r for r in usable}
        chosen = [by_id[v] for v in wanted if v in by_id][: args.n or None]
        print(f"[obj] selected {len(chosen)} clips from {args.video_ids_from}")
    else:
        # One clip per distinct caption where possible, so the examples are varied
        # rather than 12 copies of the same task.
        by_caption: dict[str, list] = {}
        for r in usable:
            by_caption.setdefault(r.caption, []).append(r)
        caps = sorted(by_caption)
        random.shuffle(caps)
        chosen = [random.choice(by_caption[c]) for c in caps[: args.n or None]]
        print(f"[obj] selected {len(chosen)} clips across {len(chosen)} distinct captions")

    vdir = args.out / "release_videos"
    qdir = args.out / "query_points"
    pdir = args.out / "seed_png"
    for d in (vdir, qdir, pdir):
        d.mkdir(parents=True, exist_ok=True)

    seed_npz = {}
    if args.seed_from_npz:
        import re as _re
        pat = _re.compile(r"^(?P<vid>.+)_f(?P<frame>\d+)\.npz$")
        for p in sorted(Path(args.seed_from_npz).glob("*.npz")):
            m = pat.match(p.name)
            if m:
                seed_npz[m["vid"]] = (int(m["frame"]), p)
        print(f"[obj] seeding from {args.seed_from_npz}: {len(seed_npz)} clips "
              f"(MolmoPoint will not be loaded)")

    molmo = molmo_proc = None
    if not args.seed_from_npz:
        print("[obj] loading MolmoPoint-Vid-4B ...")
        molmo, molmo_proc = grounding.load_pointer()
    print("[obj] loading SAM 2.1 ...")
    sam, sam_proc = segment.load_sam2()

    ref_tracks = {}
    if args.compare_ref:
        from droid_qp.reference import load_reference_tracks

        ref_tracks = load_reference_tracks({r.video_id for r in chosen})
        print(f"[obj] {len(ref_tracks)}/{len(chosen)} clips have released reference tracks")

    tiles, report = [], []
    for i, rec in enumerate(chosen, 1):
        obj = None if args.text_free else nouns[rec.caption]["noun"]
        print(f"\n[{i}/{len(chosen)}] {rec.video_id}")
        print(f"  caption: {rec.caption!r}  ->  object: {obj!r}"
              if obj else f"  caption: {rec.caption!r}  ->  [text-free]")
        row = {"video_id": rec.video_id, "caption": rec.caption, "noun": obj,
               "category": rec.category, "num_frames": rec.num_frames}
        try:
            mp4 = video_io.reconstruct(rec.source_mp4, vdir / f"{rec.video_id}.mp4")

            frame_offset = 0
            ground_mp4 = mp4
            if args.ground_in_motion_window and rec.motion_ranges:
                # Longest range, not the first. 11.4% of clips carry >1 range and the
                # ranges are in temporal order, so range[0] is the earliest burst of
                # motion -- for a pick-and-place that is the arm reaching, not the
                # object moving. On AUTOLab+0d4edc83+2023-10-21-19h-03m-59s the ranges
                # are [[26,36],[49,76]] while the grasp (from PointWorld's gripper_open)
                # is at release frames ~46-54, i.e. inside the SECOND range.
                # Interim fix: the principled window is the grasp interval itself.
                s, e = max(rec.motion_ranges, key=lambda r: r[1] - r[0])
                s = int(np.clip(s, 0, rec.num_frames - 1))
                e = int(np.clip(e, s, rec.num_frames - 1))
                # `motion_ranges` and `num_frames` are in MolmoMotion's 3.6 Hz frame
                # space; `mp4` is OUR reconstruction at rates.SUBSAMPLE. Trimming with
                # unconverted indices cuts the wrong window -- at SUBSAMPLE=2, half the
                # intended length at half the intended position, silently. One MM frame
                # spans `span` of ours, so the inclusive end extends by span-1.
                span = rates.MM_SUBSAMPLE // rates.SUBSAMPLE
                s = int(rates.mm_frame_to_release(s))
                e = int(rates.mm_frame_to_release(e)) + span - 1
                ground_mp4 = video_io.trim(
                    mp4, vdir / f"{rec.video_id}_m{s}-{e}.mp4", s, e)
                frame_offset = s
                row["motion_window"] = [s, e]
                print(f"  grounding on motion window [{s}, {e}] of {rec.num_frames}")

            if args.seed_from_npz:
                if rec.video_id not in seed_npz:
                    row["status"] = "no_seed_npz"
                    report.append(row)
                    print("  -> no seed npz for this clip")
                    continue
                fidx, spath = seed_npz[rec.video_id]
                spts = np.load(spath)["query_points"][:, 1:].astype(np.float64)
                if len(spts) == 0:
                    row["status"] = "empty_seed_npz"
                    report.append(row)
                    continue
                # Medoid, not mean: for an L-shaped or split point set the mean can
                # land off the object entirely, the same way a k-means centre can fall
                # outside a non-convex mask.
                c = spts.mean(axis=0)
                px, py = spts[int(np.argmin(((spts - c) ** 2).sum(axis=1)))]
                row.update({"detection_frame": fidx, "seed_point": [float(px), float(py)],
                            "n_seed_points": int(len(spts)), "seed_source": str(spath.name)})
                print(f"  seed frame {fidx}, medoid of {len(spts)} released points "
                      f"({px:.0f}, {py:.0f})")
            else:
                points = grounding.point_at_object(molmo, molmo_proc, str(ground_mp4), obj)
                if not points:
                    row["status"] = "no_molmo_point"
                    report.append(row)
                    print("  -> MolmoPoint returned nothing")
                    continue

                # One detection frame per clip: the first point MolmoPoint emitted.
                # Its index is relative to whatever video we grounded on, so shift it
                # back into full-clip frame space.
                fidx, px, py = points[0]
                fidx = int(np.clip(fidx + frame_offset, 0, rec.num_frames - 1))
                row.update({"detection_frame": fidx,
                            "molmo_point": [float(px), float(py)],
                            "n_molmo_points": len(points)})
                print(f"  detection frame {fidx}, point ({px:.0f}, {py:.0f})")

            frame = video_io.read_frame(mp4, fidx)
            mask, info = segment.segment_from_point(sam, sam_proc, frame, (px, py),
                                                    mask_choice=args.mask_choice)
            if mask is None:
                row["status"] = "no_mask"
                row["sam_info"] = info.get("reason")
                report.append(row)
                tiles.append(viz.seed_frame_png(frame, None, None, (px, py),
                                                rec.caption, f"{obj} [NO MASK]"))
                print(f"  -> SAM 2.1: {info.get('reason')}")
                continue

            if args.geometry_correct:
                from droid_qp.attached import attached_cluster_2d, mask_recall

                ac = attached_cluster_2d(rec.video_id, fidx,
                                         out_size=(frame.shape[1], frame.shape[0]))
                row["geo_n_attached"] = ac.get("n_attached", 0)
                row["geo_reason"] = ac.get("reason", "")
                if ac["ok"]:
                    rec0 = mask_recall(mask, ac["uv"])
                    row["geo_recall_before"] = rec0
                    print(f"  geometry: {ac['n_attached']} attached points, "
                          f"mask covers {rec0*100:.0f}%")
                    if rec0 < args.correct_below:
                        mx, my = float(ac["medoid"][0]), float(ac["medoid"][1])
                        m2, i2 = segment.segment_from_point(
                            sam, sam_proc, frame, (mx, my), mask_choice=args.mask_choice)
                        if m2 is not None:
                            rec1 = mask_recall(m2, ac["uv"])
                            row["geo_recall_after"] = rec1
                            if rec1 > rec0:
                                print(f"  -> re-seeded at ({mx:.0f},{my:.0f}); "
                                      f"recall {rec0*100:.0f}% -> {rec1*100:.0f}%")
                                mask, info, px, py = m2, i2, mx, my
                                row["geo_corrected"] = True
                            else:
                                print(f"  -> re-seed did not improve "
                                      f"({rec0*100:.0f}% -> {rec1*100:.0f}%); kept original")

            area_frac = info["chosen"]["area_frac"]
            hit = segment.point_hits_mask(mask, (px, py))
            qp = kmeans_sample(mask, args.kmeans_k, frame_idx=fidx, seed=args.kmeans_seed,
                               reproject_to_mask=args.reproject_to_mask)
            off = fraction_off_mask(qp, mask)
            row.update({"status": "ok", "mask_area_frac": area_frac,
                        "point_in_mask": hit, "n_query_points": int(len(qp)),
                        "frac_points_off_mask": off})
            print(f"  mask {area_frac*100:.1f}% of frame | point_in_mask={hit} "
                  f"| {len(qp)} query points | {off*100:.0f}% off-mask")

            if rec.video_id in ref_tracks:
                from droid_qp.reference import ref_point_recall

                tr, vs = ref_tracks[rec.video_id]
                recall = ref_point_recall(mask, tr, vs, fidx)
                row["ref_points_in_mask"] = recall
                if recall is not None:
                    print(f"  vs MolmoMotion reference: {recall*100:.0f}% of their "
                          f"points inside our mask")

            np.savez(qdir / f"{rec.video_id}_f{fidx}.npz",
                     query_points=qp.astype(np.int32),
                     dim=np.array(frame.shape[:2], dtype=np.int32),
                     mask=mask)
            img = viz.seed_frame_png(
                frame, mask, qp[:, 1:], (px, py), rec.caption, obj,
                out_path=pdir / f"{rec.video_id}_f{fidx}.png")
            tiles.append(img)
            report.append(row)

        except Exception as e:
            row["status"] = f"error: {type(e).__name__}: {e}"
            report.append(row)
            traceback.print_exc()

    if tiles:
        viz.contact_sheet(tiles, cols=3, out_path=args.out / "contact_sheet.png")
        print(f"\n[obj] contact sheet -> {args.out / 'contact_sheet.png'}")

    json.dump(report, open(args.out / "object_points_report.json", "w"), indent=1)

    ok = [r for r in report if r.get("status") == "ok"]
    print(f"\n===== summary =====")
    print(f"clips attempted : {len(report)}")
    print(f"ok              : {len(ok)}")
    for st in ("no_molmo_point", "no_mask"):
        n = sum(r.get("status") == st for r in report)
        if n:
            print(f"{st:16s}: {n}")
    if ok:
        print(f"point-in-mask   : {sum(r['point_in_mask'] for r in ok)}/{len(ok)}")
        print(f"median mask area: {np.median([r['mask_area_frac'] for r in ok])*100:.1f}% of frame")
        print(f"median N points : {np.median([r['n_query_points'] for r in ok]):.0f}")
        print(f"mean off-mask   : {np.mean([r['frac_points_off_mask'] for r in ok])*100:.1f}%")
        rc = [r["ref_points_in_mask"] for r in ok
              if r.get("ref_points_in_mask") is not None]
        if rc:
            rc = np.array(rc)
            print(f"vs reference    : median {np.median(rc)*100:.0f}% recall over "
                  f"{len(rc)} clips, {int((rc > 0.7).sum())} above 70%")


if __name__ == "__main__":
    main()
