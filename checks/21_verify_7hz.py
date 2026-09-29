#!/usr/bin/env python3
"""Gate the 7.2 Hz migration. Every check has an answer known BEFORE the run.

A smoke test that only reports numbers is useless here -- the whole risk of this change
is that it produces plausible-looking output at the wrong frame rate. So each check below
states what the value must be and fails loudly otherwise.

    python checks/21_verify_7hz.py --obj out/smoke7_obj --robot out/smoke7_robot \
        --robot-sam out/smoke7_robot_sam --trackon out/smoke7_trackon
Run with any env that has numpy + cv2 (molmo-motion or pointworld-env).
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

from droid_qp import rates
from droid_qp.index import load_manifest

from droid_qp.paths import DATA  # noqa: E402
OUT = DATA  # the dataset root: droid_qp/paths.py
NAME_RE = re.compile(r"^(?P<vid>.+)_f(?P<frame>\d+)\.npz$")

# The clip whose robot points were EMPTY before the birth-frame fix: PointWorld coverage
# starts around step 4 but no FK point projects into frame until step 8.
REGRESSION_CLIP = "AUTOLab+5d05c5aa+2023-11-17-22h-46m-01s__24400334"


class Gate:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, expected, got):
        self.rows.append((bool(ok), name, expected, got))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n"
              f"         expected: {expected}\n"
              f"         got     : {got}")

    def summary(self) -> int:
        bad = [r for r in self.rows if not r[0]]
        print(f"\n===== {len(self.rows) - len(bad)}/{len(self.rows)} gates passed =====")
        for _, name, exp, got in bad:
            print(f"  FAILED: {name} -- expected {exp}, got {got}")
        return 1 if bad else 0


def frames_of(mp4: Path) -> int:
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", type=Path, default=OUT / "smoke7_obj")
    ap.add_argument("--robot", type=Path, default=OUT / "smoke7_robot")
    ap.add_argument("--robot-sam", type=Path, default=OUT / "smoke7_robot_sam")
    ap.add_argument("--trackon", type=Path, default=OUT / "smoke7_trackon")
    args = ap.parse_args()

    print(f"[verify] {rates.describe()}\n")
    g = Gate()

    # ---- gate 1: the videos are actually at the new rate -------------------------------
    # The failure this catches is `video_io.reconstruct` returning a cached 3.6 Hz MP4,
    # which renumbers every frame index downstream with nothing raising. Compared against
    # MolmoMotion's own num_frames rather than against the old files, so it is an absolute
    # check and not a relative one.
    print("--- gate 1: reconstructed videos are at SUBSAMPLE=%d ---" % rates.SUBSAMPLE)
    span = rates.MM_SUBSAMPLE // rates.SUBSAMPLE
    by_id = {r.video_id: r for r in load_manifest()}
    mp4s = sorted((args.obj / "release_videos").glob("*.mp4"))
    worst, detail = None, []
    for p in mp4s:
        rec = by_id.get(p.stem)
        if rec is None:
            continue
        n, want = frames_of(p), rec.num_frames * span
        err = abs(n - want) / max(want, 1)
        detail.append(f"{p.stem[:34]} n={n} want~{want}")
        if worst is None or err > worst:
            worst = err
    g.check(f"video frame count ~= {span}x MolmoMotion's num_frames",
            worst is not None and worst < 0.05,
            f"within 5% of {span}x for all {len(mp4s)} clips",
            f"worst deviation {worst:.1%}" if worst is not None else "no videos found")
    for d in detail[:4]:
        print(f"           {d}")

    # ---- gate 2: the birth-frame bug is gone -------------------------------------------
    print("\n--- gate 2: every robot clip has a non-empty query set ---")
    rfiles = sorted((args.robot / "query_points").glob("*.npz"))
    empties = [p.name for p in rfiles if len(np.load(p)["query_points"]) == 0]
    g.check("no empty robot query sets", not empties,
            "0 empty", f"{len(empties)} empty: {empties[:3]}")

    hit = [p for p in rfiles if p.name.startswith(REGRESSION_CLIP)]
    if hit:
        d = np.load(hit[0])
        f0 = int(NAME_RE.match(hit[0].name)["frame"])
        nvis = d["fk_ok"].sum(axis=1)
        g.check(f"regression clip {REGRESSION_CLIP[:28]}... born on a VISIBLE step",
                len(d["query_points"]) >= 5,
                ">=5 query points, born where FK points are in frame",
                f"{len(d['query_points'])} points at f{f0} "
                f"(visible at that step: {int(nvis[f0]) if f0 < len(nvis) else 'n/a'})")
    else:
        g.check("regression clip present", False, REGRESSION_CLIP, "not in the run")

    # ---- gate 3: the FK <-> release mapping is the identity -----------------------------
    print("\n--- gate 3: fk_release_frame == arange(S) ---")
    bad = []
    for p in rfiles:
        rel = np.load(p)["fk_release_frame"]
        if not np.array_equal(rel, np.arange(len(rel))):
            bad.append(p.name)
    g.check("PointWorld step == release frame at 7.2 Hz",
            rates.RATIO != 1 or not bad,
            "identity mapping for all clips" if rates.RATIO == 1 else f"ratio {rates.RATIO}",
            f"{len(bad)} clips deviate: {bad[:3]}" if bad else "identity, all clips")

    # ---- gate 4: the token stream exists and lines up -----------------------------------
    print("\n--- gate 4: token stream shape and alignment ---")
    tfiles = sorted((args.trackon / "tracks").glob("*.npz"))
    if not tfiles:
        g.check("token stream written", False, "npz with a `tokens` key", "no track files")
    else:
        d = np.load(tfiles[0])
        has = "tokens" in d.files and "q_init" in d.files
        g.check("tokens + q_init saved", has, "both keys present", sorted(d.files))
        if has:
            tk, tr = d["tokens"], d["tracks"]
            g.check("tokens (T, N, 256) matches tracks (T, N, 2)",
                    tk.shape[:2] == tr.shape[:2] and tk.shape[2] == 256,
                    f"({tr.shape[0]}, {tr.shape[1]}, 256)", str(tk.shape))
            g.check("tokens are fp16", tk.dtype == np.float16, "float16", str(tk.dtype))
            # Track-On writes memory EVERY frame, visible or not, so a live point must
            # have a non-zero token even where visibility is False. All-zero rows for a
            # born point would mean the birth-order scatter is wrong.
            born = ~np.isnan(tr).any(axis=2)
            nz = (np.abs(tk).sum(axis=2) > 0)
            leak = int((born & ~nz).sum())
            g.check("every tracked point has a token on every frame it is alive",
                    leak == 0, "0 alive-but-zero tokens", f"{leak} of {int(born.sum())}")
            g.check("subsample recorded in the npz",
                    int(d["subsample"]) == rates.SUBSAMPLE if "subsample" in d.files else False,
                    f"{rates.SUBSAMPLE}",
                    str(d["subsample"]) if "subsample" in d.files else "key absent")

    # ---- context: SAM filter yield -------------------------------------------------------
    rep = args.robot_sam / "report.json"
    if rep.exists():
        r = json.load(open(rep))
        if r:
            kf = np.array([x["keep_frac"] for x in r])
            print(f"\n--- context: robot SAM filter kept mean {kf.mean()*100:.1f}% "
                  f"min {kf.min()*100:.1f}% over {len(r)} clips "
                  f"(was mean 98% at 3.6 Hz) ---")

    sys.exit(g.summary())


if __name__ == "__main__":
    main()
