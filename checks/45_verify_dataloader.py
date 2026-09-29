#!/usr/bin/env python3
"""Prove the dataloader is causal and correctly aligned. Step 1 of DECODER_BUILD.md.

Five checks, each of which has failed in some project somewhere and produced a model that
trained fine and could not be deployed. The one that matters most is #1: if a sample
contains anything from after the supervised frame, everything downstream is invalid and
nothing crashes to say so.

    python checks/45_verify_dataloader.py
Run with any env with numpy.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp.policy_data import D, N_OBJ, N_ROB_MAX, TARGET_DIM, PolicyWindows

from droid_qp.paths import DATA  # noqa: E402
OUT = DATA  # the dataset root: droid_qp/paths.py
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main():
    ds = PolicyWindows("train", seed=0)
    va = PolicyWindows("val", seed=0)
    print(f"train {len(ds):,} clips | val {len(va):,} clips\n")

    # ---- 1. CAUSALITY -----------------------------------------------------------------
    # Reload the raw stream and confirm the window is exactly the slice it claims, with
    # nothing from beyond the last supervised frame. Checked against the file, not against
    # the loader's own arithmetic -- otherwise a wrong offset would verify itself.
    print("1. causality: no sample contains data from after its supervised span")
    bad = 0
    rng = np.random.default_rng(0)
    for i in rng.integers(0, len(ds), 300):
        b = ds.sample(int(i))
        obj, rob, tgt, valid = ds._load(b["video_id"])
        s, M, Tw = b["start"], ds.M, ds.T_win
        lo, end = max(0, s - (M - 1)), min(s + Tw, min(len(obj), len(tgt)))
        pad = max(0, (M - 1) - s)
        ref = np.zeros_like(b["scene"])
        ref[pad:pad + (end - lo)] = obj[lo:end]
        if not np.array_equal(ref, b["scene"]):
            bad += 1
    check("window equals the raw stream slice, re-read from disk", bad == 0,
          f"{bad}/300 mismatched")

    # the supervised frames end at s+Tw-1; nothing at s+Tw or beyond may appear
    leaked = 0
    for i in rng.integers(0, len(ds), 300):
        b = ds.sample(int(i))
        obj, _, tgt, _ = ds._load(b["video_id"])
        T = min(len(obj), len(tgt))
        nxt = b["start"] + ds.T_win
        if nxt < T:
            future = obj[nxt]
            if np.any(np.all(b["scene"] == future[None], axis=(1, 2))):
                leaked += 1
    check("no frame at or beyond s+T_win appears in the window", leaked == 0,
          f"{leaked}/300 leaked")

    # ---- 2. ALIGNMENT -----------------------------------------------------------------
    print("\n2. alignment: the newest memory slot is the frame being predicted")
    bad = 0
    for i in rng.integers(0, len(ds), 200):
        b = ds.sample(int(i))
        obj, _, _, _ = ds._load(b["video_id"])
        # slot (M-1) of the stream is the first SUPERVISED frame s
        if not np.array_equal(b["scene"][ds.M - 1], obj[b["start"]]):
            bad += 1
    check("scene[M-1] == tokens[s] for the object points", bad == 0, f"{bad}/200")

    # ---- 3. PADDING -------------------------------------------------------------------
    print("\n3. padding: masks mark every padded slot, and padding is zero")
    bad_rob = bad_hist = 0
    n_early = 0
    for i in rng.integers(0, len(ds), 300):
        b = ds.sample(int(i))
        nr = b["n_rob"]
        if nr < N_ROB_MAX and np.abs(b["robot"][:, nr:]).max() != 0:
            bad_rob += 1
        if b["robot_pad"].sum() != nr:
            bad_rob += 1
        if not b["hist_valid"].all():
            n_early += 1
            if np.abs(b["scene"][~b["hist_valid"]]).max() != 0:
                bad_hist += 1
    check("padded robot slots are zero and masked", bad_rob == 0, f"{bad_rob}/300")
    check("front-padded history is zero and masked", bad_hist == 0,
          f"{bad_hist} bad of {n_early} early-start windows")

    # ---- 4. TARGETS -------------------------------------------------------------------
    print("\n4. targets: line up with the supervised frames, and are physically sane")
    bad = 0
    dp = []
    for i in rng.integers(0, len(ds), 200):
        b = ds.sample(int(i))
        _, _, tgt, valid = ds._load(b["video_id"])
        s = b["start"]
        n = int(b["target_valid"].sum())
        if n and not np.allclose(b["target"][:n], tgt[s:s + n], atol=0, rtol=0):
            bad += 1
        dp.append(np.linalg.norm(b["target"][b["target_valid"], :3], axis=1))
    check("target[j] == labels[s+j]", bad == 0, f"{bad}/200")
    dp = np.concatenate([d for d in dp if len(d)])
    check("translation magnitudes are physically possible", dp.max() < 0.25,
          f"max {dp.max()*100:.2f} cm/frame (0.25 m would be >1.8 m/s at 7.2 Hz)")

    # ---- 5. SPLIT INTEGRITY -----------------------------------------------------------
    print("\n5. split: no episode appears on both sides")
    tr_ep = {c["episode"] for c in ds.clips}
    va_ep = {c["episode"] for c in va.clips}
    check("train and val episodes are disjoint", len(tr_ep & va_ep) == 0,
          f"{len(tr_ep & va_ep)} shared")

    # ---- shapes + throughput ----------------------------------------------------------
    b = ds.sample(0)
    L = ds.M - 1 + ds.T_win
    check("shapes", b["scene"].shape == (L, N_OBJ, D)
          and b["robot"].shape == (L, N_ROB_MAX, D)
          and b["target"].shape == (ds.T_win, TARGET_DIM),
          f"scene {b['scene'].shape} robot {b['robot'].shape} target {b['target'].shape}")

    t0 = time.time()
    for i in rng.integers(0, len(ds), 40):
        ds.sample(int(i))
    cold = (time.time() - t0) / 40
    t0 = time.time()
    for _ in range(40):
        ds.sample(0)
    warm = (time.time() - t0) / 40
    mb = (b["scene"].nbytes + b["robot"].nbytes) / 1e6
    print(f"\n  throughput: {cold*1000:.0f} ms/sample cold, {warm*1000:.1f} ms warm "
          f"| {mb:.1f} MB per sample")

    print(f"\n===== {'ALL CHECKS PASSED' if not FAIL else str(len(FAIL)) + ' FAILED: ' + ', '.join(FAIL)} =====")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
