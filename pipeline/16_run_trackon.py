#!/usr/bin/env python3
"""Run the frozen Track-On-R on our query points, look at the tracks, inspect the memory.

This is the "point decoder" step of the architecture: frames + query points go into a
frozen tracker, and the tracker's own `point_memory` IS the scene memory the action
decoder will read. Nothing needs to be built to produce that memory -- it is a live
attribute on `Predictor`, allocated as zeros and filled one token per frame
(trackon_predictor.py:177-178). No robot points are required for this.

Uses `forward_frame` rather than `forward` so we advance the video one frame at a time
and can snapshot the memory as it fills -- which is also how the policy will run it
(causal, online).

Outputs per clip:
    tracks/<vid>.npz     tracks (T,N,2), visibility (T,N), queries (N,3), memory stats
    videos/<vid>.mp4     the tracks drawn over the video, to actually look at

    python pipeline/16_run_trackon.py --points-dir out/mm_sam/query_points --n 12
Run with $PY_TRACKON
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droid_qp.paths import DATA, TRACK_ON  # noqa: E402
sys.path.insert(0, str(TRACK_ON))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp import rates  # noqa: E402  (after sys.path setup)

OUT = DATA  # the dataset root: droid_qp/paths.py
NAME_RE = re.compile(r"^(?P<vid>.+)_f(?P<frame>\d+)\.npz$")


def load_predictor(config: Path, ckpt: Path, support_grid: int, device: str):
    from model.trackon_predictor import Predictor
    from utils.train_utils import load_args_from_yaml

    model_args = load_args_from_yaml(str(config))
    p = Predictor(model_args, checkpoint_path=str(ckpt),
                  support_grid_size=support_grid).to(device).eval()
    return p


def read_video(mp4: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"decoded 0 frames from {mp4}")
    return np.stack(frames)


@torch.no_grad()
def run_clip(predictor, video: np.ndarray, queries: np.ndarray, device: str):
    """queries (N,3) rows (t,x,y) in the SAME pixel space as `video`.

    Track-On is causal: a point does not exist before its birth frame, so we hand each
    query to `forward_frame` on the frame its `t` names. `forward_frame` takes (x, y)
    only -- the birth time is implied by when you pass it.
    """
    predictor.reset() if hasattr(predictor, "reset") else None
    T, H, W = video.shape[0], video.shape[1], video.shape[2]
    births = queries[:, 0].astype(int)
    order = np.argsort(births, kind="stable")   # feed in birth order
    q_sorted = queries[order]
    b_sorted = births[order]

    tracks = np.full((T, len(queries), 2), np.nan, dtype=np.float32)
    vis = np.zeros((T, len(queries)), dtype=bool)
    # The per-frame token stream -- the artifact the policy actually consumes. Stored as
    # a stream rather than as the memory itself: the memory is a sliding 72-slot window,
    # so saving it per frame would write every token 72 times for identical information.
    # memory at time t == tokens[max(0, t-71) : t+1]; slicing [:t+1] is what keeps
    # training causal. fp16 halves 182 GB across the 14,083-clip set at no measurable cost.
    D = predictor.model.D
    tokens = np.zeros((T, len(queries), D), dtype=np.float16)
    mem_fill, mem_norm = [], []
    cursor = 0
    n_active = 0

    for t in range(T):
        frame = torch.from_numpy(video[t]).permute(2, 0, 1)[None].float().to(device)

        new_q = None
        start = cursor
        while cursor < len(q_sorted) and b_sorted[cursor] == t:
            cursor += 1
        if cursor > start:
            new_q = torch.from_numpy(
                q_sorted[start:cursor, 1:].astype(np.float32)).to(device)

        p, v = predictor.forward_frame(frame, new_queries=new_q)
        if p.numel() == 0:
            continue
        n_active = p.shape[0]
        # Points were appended in birth order, so the first n_active rows of `order`
        # are exactly the queries currently alive.
        idx = order[:n_active]
        tracks[t, idx] = p.detach().cpu().numpy()[:n_active]
        vis[t, idx] = v.detach().cpu().numpy()[:n_active]

        pm = predictor.point_memory[:predictor.N]      # (N, M, D) -- the SCENE MEMORY
        # `forward_frame` has just written this frame's token into the newest slot
        # (trackon_predictor.py:177-178), so pm[:, -1] IS q_new for frame t. Nothing in
        # Track-On needs modifying to expose it.
        #
        # NOTE the scatter through `idx`, not a plain slice: rows of point_memory are in
        # BIRTH order, while `tracks`/`tokens` columns are in query order. Writing
        # tokens[t, :n_active] instead would silently pair token i with a different
        # point than query i whenever births are not already sorted.
        tokens[t, idx] = pm[:n_active, -1].float().cpu().numpy().astype(np.float16)
        mem_fill.append(float((pm.abs().sum(-1) > 0).float().mean()))
        mem_norm.append(float(pm.norm(dim=-1).mean()))

    # q_init is the birth-frame anchor, held OUTSIDE the memory and re-fed as the
    # decoder's input query on every frame (trackon_predictor.py:95-96). It is the
    # point's identity, not part of its history, so the action-decoder replica needs its
    # own equivalent. ~46 KB.
    q_init = np.zeros((len(queries), D), dtype=np.float16)
    if n_active:
        q_init[order[:n_active]] = (
            predictor.q_init[:n_active].float().cpu().numpy().astype(np.float16))

    return tracks, vis, tokens, q_init, {
        "n_active": int(n_active),
        "memory_shape": list(predictor.point_memory[:predictor.N].shape),
        "token_shape": list(tokens.shape),
        "memory_fill_per_frame": mem_fill,
        "memory_norm_per_frame": mem_norm,
    }


def draw(video, tracks, vis, out_mp4: Path, tail=8):
    T, H, W = video.shape[0], video.shape[1], video.shape[2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    for t in range(T):
        img = cv2.cvtColor(video[t], cv2.COLOR_RGB2BGR).copy()
        for k in range(tracks.shape[1]):
            for dt in range(1, min(tail, t) + 1):
                a, b = tracks[t - dt], tracks[t - dt + 1]
                if np.isnan(a[k]).any() or np.isnan(b[k]).any():
                    continue
                cv2.line(img, tuple(a[k].astype(int)), tuple(b[k].astype(int)),
                         (120, 120, 0), 1, cv2.LINE_AA)
            p = tracks[t, k]
            if np.isnan(p).any():
                continue
            c = (255, 255, 0) if vis[t, k] else (90, 90, 90)
            cv2.circle(img, tuple(p.astype(int)), 2, c, -1 if vis[t, k] else 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, H - 22), (W, H), (0, 0, 0), -1)
        cv2.putText(img, f"t={t}/{T-1}  visible={int(vis[t].sum())}/{tracks.shape[1]}",
                    (6, H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        vw.write(img)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points-dir", type=Path, required=True)
    ap.add_argument("--videos-dir", type=Path, default=None,
                    help="defaults to <points-dir>/../release_videos")
    ap.add_argument("--config", type=Path, default=TRACK_ON / "config/test.yaml")
    ap.add_argument("--ckpt", type=Path, default=TRACK_ON / "checkpoints/track_on_r.pt")
    ap.add_argument("--support-grid", type=int, default=0,
                    help="Predictor can append its own uniform grid at t=0. Kept at 0 so "
                         "the memory contains only OUR points -- otherwise the policy "
                         "would also see 400 support tokens we did not choose.")
    ap.add_argument("--n", type=int, default=12,
                    help="number of clips; 0 = ALL. At scale each shard holds 74-123 clips,\n"
                         "so a forgotten --n would silently process only the default and\n"
                         "still report success -- hence 0 means all, matching 24_action_labels.py.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vdir = args.videos_dir or (args.points_dir.parent / "release_videos")
    out = args.out or (OUT / f"trackon_{args.points_dir.parent.name}")
    (out / "tracks").mkdir(parents=True, exist_ok=True)

    files = sorted(args.points_dir.glob("*.npz"))[: args.n or None]
    print(f"[trackon] {len(files)} clips | device={device} | support_grid={args.support_grid}")
    print(f"[trackon] ckpt {args.ckpt}")

    predictor = load_predictor(args.config, args.ckpt, args.support_grid, device)
    print("[trackon] predictor ready")

    report = []
    for i, p in enumerate(files, 1):
        m = NAME_RE.match(p.name)
        vid = m["vid"]
        mp4 = vdir / f"{vid}.mp4"
        if not mp4.exists():
            print(f"  [{i}] {vid[:44]} -- no video at {mp4}")
            continue
        try:
            _npz = np.load(p)
            q = _npz["query_points"].astype(np.float32)
            # Carried through untouched so the token stream can be split into
            # scene_tokens / robot_tokens downstream (28_merge_queries.py).
            role = _npz["role"] if "role" in _npz.files else None
            video = read_video(mp4)
            # Predictor DOES keep queries and memory across calls, so per-clip state must
            # be cleared -- but `reset()` (trackon_predictor.py:51-62) already clears all
            # of it: t, point_memory, temporal_mask, q_init, N, capacity, device. Only
            # `self.model` survives, which is exactly what we want to keep. `run_clip`
            # calls reset() on entry, so rebuilding the Predictor here was redundant:
            # measured 3.20 s to re-read the 94 MB checkpoint against 97 us for reset(),
            # i.e. 12.5 hours of pure reloading across 14,083 clips.
            tracks, vis, tokens, q_init, meta = run_clip(predictor, video, q, device)

            np.savez(out / "tracks" / f"{vid}.npz", tracks=tracks, visibility=vis,
                     queries=q, tokens=tokens, q_init=q_init,
                     subsample=np.int32(rates.SUBSAMPLE),
                     **({"role": role} if role is not None else {}))
            draw(video, tracks, vis, out / "videos" / f"{vid}.mp4")

            frac_vis = float(vis.mean())
            surv = float(vis[-1].mean())
            print(f"  [{i:2d}] {vid[:44]:44s} T={video.shape[0]:3d} N={q.shape[0]:3d} "
                  f"mem={meta['memory_shape']} visible={frac_vis*100:4.1f}% "
                  f"last-frame={surv*100:4.1f}%")
            report.append({"video_id": vid, "T": int(video.shape[0]),
                           "N": int(q.shape[0]), "memory_shape": meta["memory_shape"],
                           "frac_visible": frac_vis, "last_frame_visible": surv,
                           "memory_fill_final": meta["memory_fill_per_frame"][-1]
                           if meta["memory_fill_per_frame"] else 0.0})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [{i}] {vid[:44]} -- {type(e).__name__}: {e}")

    json.dump(report, open(out / "report.json", "w"), indent=1)
    print(f"\n===== summary =====")
    print(f"clips tracked : {len(report)}")
    if report:
        fv = np.array([r["frac_visible"] for r in report])
        lf = np.array([r["last_frame_visible"] for r in report])
        print(f"mean visible over all frames : {fv.mean()*100:.1f}%")
        print(f"visible at the LAST frame    : {lf.mean()*100:.1f}%  "
              f"(how many points survive the whole clip)")
        print(f"scene memory shape           : {report[0]['memory_shape']}  (N, M, D)")
    print(f"\n[trackon] wrote {out}")


if __name__ == "__main__":
    main()
