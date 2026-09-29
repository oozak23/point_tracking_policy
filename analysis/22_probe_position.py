#!/usr/bin/env python3
"""Is (x, y) recoverable from a Track-On memory token? -- decision #1, settled by measurement.

Why this matters (DECISIONS.md §1, ACTION_DECODER.md §0 and §4). Fatma hoca's call is that
positions and visibility need not be stored, because Track-On already puts them in its
memory. For VISIBILITY that is true by construction: `v_logit = prediction_head(q_t, ...)`
takes the memory token as its principal input (trackon.py:527).

Position is different. `p = p_patch + o[-1]` where
`p_patch = argmax(correlate(q_t, f_fused_t))` -- the token localises the point BY
CORRELATION AGAINST THE CURRENT FRAME, and there is no spatial positional encoding
anywhere in the decoder (the only one, `t_embedding`, is over the 72 memory slots and is
temporal). So position is available from token + frame features, and the open question is
how much survives in the token ALONE.

That question is forced by storage, not preference: the frame feature map is
(T, 12288, 256) ~ 377 MB/clip, i.e. 5.3 PB over 14,083 clips, so the policy reads memory
only (ACTION_DECODER.md §4).

Reading the result
------------------
  probe BEATS the constant baseline by a wide margin -> position IS in the token, Fatma
      hoca is right in the strong sense, and we can drop the extra inputs for good.
  probe barely beats it -> the token is appearance, position must be fed explicitly.
      It costs 1.4 GB of 182 GB, so this is cheap to act on.

One asymmetry to respect: a linear probe SUCCEEDING proves the information is present; a
linear probe FAILING does not prove it is absent, only that it is not linearly available.
So an MLP probe runs as well, and only agreement between the two is conclusive.

Split is BY CLIP, never by frame: tracks are temporally smooth, so a random frame split
puts near-duplicate rows on both sides and would report a strong result from leakage alone.

    python analysis/22_probe_position.py --tracks out/smoke7_trackon/tracks
Run with $PY_MOLMO  (needs sklearn)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp.paths import DATA  # noqa: E402
OUT = DATA  # the dataset root: droid_qp/paths.py
def load(tracks_dir: Path, max_clips: int):
    clips = []
    for p in sorted(tracks_dir.glob("*.npz"))[:max_clips]:
        d = np.load(p)
        if "tokens" not in d.files:
            continue
        tk, tr, vi = d["tokens"], d["tracks"], d["visibility"]
        # Only rows where the point is alive AND visible: an invisible point has no
        # well-defined position to regress onto, so including it would measure noise.
        alive = ~np.isnan(tr).any(axis=2) & vi
        if alive.sum() < 100:
            continue
        clips.append({
            "name": p.stem,
            "X": tk[alive].astype(np.float32),          # (n, 256)
            "Y": tr[alive].astype(np.float32),          # (n, 2) pixels
        })
    return clips


def evaluate(name, pred, Yte, base_px):
    err = np.linalg.norm(pred - Yte, axis=1)
    med, p90 = float(np.median(err)), float(np.percentile(err, 90))
    ss_res = ((pred - Yte) ** 2).sum()
    ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot)
    print(f"  {name:22s} median {med:6.1f} px   p90 {p90:6.1f} px   R2 {r2:6.3f}   "
          f"{base_px / max(med, 1e-6):5.2f}x better than constant")
    return {"median_px": med, "p90_px": p90, "r2": r2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=Path, default=OUT / "smoke7_trackon" / "tracks")
    ap.add_argument("--max-clips", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=2, help="clips held out")
    ap.add_argument("--out", type=Path, default=OUT / "probe_position.json")
    args = ap.parse_args()

    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor

    clips = load(args.tracks, args.max_clips)
    if len(clips) < args.n_test + 2:
        sys.exit(f"need >= {args.n_test + 2} clips with tokens, found {len(clips)}")

    te, tr_ = clips[: args.n_test], clips[args.n_test:]
    Xtr = np.concatenate([c["X"] for c in tr_]); Ytr = np.concatenate([c["Y"] for c in tr_])
    Xte = np.concatenate([c["X"] for c in te]);  Yte = np.concatenate([c["Y"] for c in te])

    print(f"[probe] train {len(tr_)} clips / {len(Xtr):,} point-frames")
    print(f"[probe] test  {len(te)} clips / {len(Xte):,} point-frames")
    print(f"[probe] held out: {[c['name'][:38] for c in te]}\n")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n, Xte_n = (Xtr - mu) / sd, (Xte - mu) / sd

    # Baseline: the best you can do knowing NOTHING about the token -- predict the mean
    # training position for every point. Any probe must beat this to have learned anything.
    const = np.repeat(Ytr.mean(axis=0, keepdims=True), len(Yte), axis=0)
    base_px = float(np.median(np.linalg.norm(const - Yte, axis=1)))
    print("results (test = held-out CLIPS, so no temporal leakage)")
    res = {"baseline": evaluate("constant (baseline)", const, Yte, base_px)}

    res["ridge"] = evaluate("linear (ridge)",
                            Ridge(alpha=1.0).fit(Xtr_n, Ytr).predict(Xte_n), Yte, base_px)

    mlp = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=60, early_stopping=True,
                       random_state=0)
    res["mlp"] = evaluate("mlp (256,128)", mlp.fit(Xtr_n, Ytr).predict(Xte_n), Yte, base_px)

    lin, mlp_r = res["ridge"]["median_px"], res["mlp"]["median_px"]
    best = min(lin, mlp_r)
    print()
    if best < 0.4 * base_px:
        verdict = ("position IS substantially recoverable from the token alone -- "
                   "decision #1 option (a) is defensible, drop the extra inputs")
    elif best < 0.75 * base_px:
        verdict = ("position is PARTIALLY recoverable -- some signal, not enough to rely "
                   "on. Store tracks + visibility (1.4 GB of 182) and ablate")
    else:
        verdict = ("position is NOT recoverable from the token alone -- the memory is "
                   "appearance, and a memory-only decoder starts blind. Feed positions")
    print(f"VERDICT: {verdict}")
    if mlp_r < 0.8 * lin:
        print("NOTE: the MLP clearly beats the linear probe, so whatever positional signal "
              "exists is non-linear. Judge by the MLP, not the ridge.")

    res["verdict"] = verdict
    res["baseline_px"] = base_px
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\n[probe] -> {args.out}")


if __name__ == "__main__":
    main()
