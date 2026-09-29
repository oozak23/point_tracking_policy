"""Reconstruct the RELEASE video from the frozen upstream MP4, and read its frames.

The release ships no RGB. `droid/reconstruct_videos.py` rebuilds all 23,532 at once;
this does the same transform for an arbitrary subset, which is what we want while
iterating on a handful of clips.

    ffmpeg select='not(mod(n,SUBSAMPLE))', scale=854:480   ->   854x480

SUBSAMPLE lives in `rates.py` and is currently 2, i.e. ~7.2 Hz. It was 4 (~3.6 Hz) while
we were matching the MolmoMotion release. Note the output is NOT 15 fps -- that figure
comes from `droid_videos_index.json` and is wrong; see `rates.py` and RESULTS.md §1.

The written container header is meaningless either way: ffmpeg stamps a nominal rate that
has nothing to do with the true capture rate, which is why `read_frames` counts frames
rather than trusting metadata.

/frozen is read-only shared storage: sources are only ever read, never written.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .rates import SUBSAMPLE

RELEASE_W, RELEASE_H = 854, 480


def ffmpeg_exe() -> str:
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def frame_count(mp4: Path) -> int:
    """Container frame count, or -1 if unavailable. Metadata read, no decode."""
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n if n > 0 else -1


def cache_is_stale(src_mp4: Path, dst_mp4: Path, subsample: int) -> bool:
    """Is an existing reconstruction at a DIFFERENT subsample than the one asked for?

    This guard exists because the failure it prevents is silent. `reconstruct` returns a
    cached file whenever the path exists, and every MP4 produced before 2026-08-14 was
    built at SUBSAMPLE=4. Reading one of those as if it were SUBSAMPLE=2 renumbers every
    frame index in the pipeline -- query birth frames, FK->release mapping, tracked
    positions -- with nothing raising anywhere. Writing to a rate-tagged directory is the
    first defence; this is the second.

    Unknown counts are treated as stale: re-encoding costs seconds, a silently wrong
    dataset costs a rerun of 14,083 clips.
    """
    n_src, n_dst = frame_count(src_mp4), frame_count(dst_mp4)
    if n_src < 0 or n_dst < 0:
        return True
    expected = -(-n_src // subsample)        # ceil: frame 0 is always kept
    return abs(n_dst - expected) > 1         # 1 frame of slack for encoder edge effects


def reconstruct(src_mp4: Path, dst_mp4: Path, overwrite: bool = False,
                subsample: int = SUBSAMPLE) -> Path:
    """Re-encode one source MP4 into its release form. Matches reconstruct_videos.py."""
    dst_mp4 = Path(dst_mp4)
    if dst_mp4.exists() and not overwrite:
        if not cache_is_stale(src_mp4, dst_mp4, subsample):
            return dst_mp4
        print(f"[video_io] {dst_mp4.name}: cached copy is not at subsample={subsample}, "
              f"re-encoding")
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_mp4.with_suffix(".tmp.mp4")
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src_mp4),
        "-vf", f"select='not(mod(n\\,{subsample}))',scale={RELEASE_W}:{RELEASE_H}",
        "-vsync", "0", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an",
        str(tmp),
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(res.stderr.decode(errors="replace").strip() or "ffmpeg failed")
    os.replace(tmp, dst_mp4)
    return dst_mp4


def trim(src_mp4: Path, dst_mp4: Path, start: int, end: int,
         overwrite: bool = False) -> Path:
    """Cut release frames [start, end] inclusive into a new MP4.

    Used to restrict *grounding* to the motion window. MolmoPoint's text-free prompt
    asks for "objects in contact and moved by robot arm", and at frame 0 nothing is in
    contact yet -- so on a full video it guesses, and was measured pointing at a cloth
    on the counter and a jacket on a chair instead of the manipulated object.

    Note this is NOT MolmoMotion's clip filter. They use motion ranges to *discard*
    static windows from the dataset, which is the sampler bias we are trying to undo.
    Here the ranges only choose where to LOOK for the object; static windows stay in
    the data, and the query points found this way get tracked across the whole clip.
    """
    dst_mp4 = Path(dst_mp4)
    if dst_mp4.exists() and not overwrite:
        return dst_mp4
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_mp4.with_suffix(".tmp.mp4")
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src_mp4),
        "-vf", f"select='between(n\\,{int(start)}\\,{int(end)})'",
        "-vsync", "0", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an",
        str(tmp),
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(res.stderr.decode(errors="replace").strip() or "ffmpeg trim failed")
    os.replace(tmp, dst_mp4)
    return dst_mp4


def read_frames(mp4: Path, max_frames: int | None = None) -> np.ndarray:
    """-> (T, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"decoded 0 frames from {mp4}")
    return np.stack(frames)


def read_frame(mp4: Path, index: int) -> np.ndarray:
    """Read a single frame by index. Falls back to sequential read if seeking fails."""
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, f = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(int(index) + 1):
            ok, f = cap.read()
            if not ok:
                break
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {index} of {mp4}")
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def frame_count(mp4: Path) -> int:
    cap = cv2.VideoCapture(str(mp4))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n
