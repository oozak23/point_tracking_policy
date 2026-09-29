"""Join the DROID release annotations to the frozen upstream raw mirror.

The molmo-motion DROID release ships annotations + tracks but not the RGB videos.
`droid_videos_index.json` maps each release `video_id` to a path under DROID v1.0.1,
which we resolve against the read-only mirror at RAW_ROOT.

Frame conventions, quoted from droid_videos_index.json and verified on 8 random clips
(source/release frame-count ratio measured 3.93-4.00):

    source MP4 : 1280x720 @ 60 fps
    release    : 854x480  @ 15 fps, every 4th source frame starting at 0
    release frame i  <->  source frame 4*i  <->  index i on the T axis of tracks NPZs
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

RAW_ROOT = Path("/frozen/avg/bora_data/droid_datasets/droid_raw_full_v2/droid_raw/1.0.1")
ANN_DIR = Path("/scratch/oozak23/droid/annotations")

# MolmoMotion's own subsample factor: source frame -> THEIR release frame. This is a
# property of the published dataset, not of our pipeline, so it is fixed at 4 and must
# NOT track `rates.SUBSAMPLE`. Every field this module reads out of `droid_clips.json`
# (`num_frames`, `motion_ranges`) is in MolmoMotion's 3.6 Hz frame space. To address one
# of OUR frames from a MolmoMotion index, go through `rates.mm_frame_to_release`.
from .rates import MM_SUBSAMPLE as SUBSAMPLE  # noqa: F401  (re-exported; 01_ imports it)


@dataclass(frozen=True)
class ClipRecord:
    video_id: str  # "<lab>+<uuid>+<timestamp>__<cam>"
    cam: str  # ZED serial
    lab: str
    num_frames: int  # frames in the RELEASE video
    fps: int
    ds_dim: tuple[int, int]  # (H, W) = (480, 854)
    caption: str
    category: str
    motion_ranges: list[tuple[int, int]]  # release-frame [start, end] windows with motion
    source_rel: str  # path under RAW_ROOT

    @property
    def source_mp4(self) -> Path:
        return RAW_ROOT / self.source_rel

    @property
    def scene_dir(self) -> Path:
        """<lab>/success/<date>/<scene>/ — holds recordings/ and trajectory.h5."""
        return (RAW_ROOT / self.source_rel).parent.parent.parent

    @property
    def trajectory_h5(self) -> Path:
        return self.scene_dir / "trajectory.h5"

    def release_to_source_frame(self, i: int) -> int:
        return i * SUBSAMPLE


def load_clips(ann_dir: Path = ANN_DIR) -> list[ClipRecord]:
    clips = json.load(open(ann_dir / "droid_clips.json"))
    src_map = json.load(open(ann_dir / "droid_videos_index.json"))["video_id_to_source_path"]

    out = []
    for e in clips:
        vid = e["file"]
        rel = src_map.get(vid)
        if rel is None:
            continue
        # clips_by_object is keyed by noun phrase, but every DROID entry uses the
        # literal placeholder "object" (verified: 1 unique key across all 23,532),
        # so it carries motion ranges only -- no usable object name.
        ranges = [tuple(r) for rs in e["clips_by_object"].values() for r in rs]
        out.append(
            ClipRecord(
                video_id=vid,
                cam=e["cam"],
                lab=vid.split("+", 1)[0],
                num_frames=e["num_frames"],
                fps=e["fps"],
                ds_dim=tuple(e["ds_dim"]),
                caption=e["caption"],
                category=e["category"],
                motion_ranges=ranges,
                source_rel=rel,
            )
        )
    return out


def load_manifest(path: Path | None = None) -> list[ClipRecord]:
    """Load the clips already known to resolve, from out/manifest.json.

    Prefer this over `load_clips()` + `.exists()`: stat-ing 23,532 paths on the beegfs
    mount takes ~25 minutes, and scripts/01 has already done it once and cached the
    answer (all 23,532 resolve, so nothing is lost by trusting it).
    """
    path = path or (Path(__file__).resolve().parent.parent / "out" / "manifest.json")
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} -- run scripts/01_build_manifest.py first")
    rows = json.load(open(path))
    src_map = json.load(open(ANN_DIR / "droid_videos_index.json"))["video_id_to_source_path"]
    return [
        ClipRecord(
            video_id=r["video_id"], cam=r["cam"], lab=r["lab"],
            num_frames=r["num_frames"], fps=15, ds_dim=(480, 854),
            caption=r["caption"], category=r["category"],
            motion_ranges=[tuple(x) for x in r["motion_ranges"]],
            source_rel=src_map[r["video_id"]],
        )
        for r in rows
    ]


def resolvable(rec: ClipRecord) -> dict[str, bool]:
    return {
        "mp4": rec.source_mp4.exists(),
        "h5": rec.trajectory_h5.exists(),
    }


def release_video_path(video_id: str, videos_dir: Path) -> Path:
    return Path(videos_dir) / f"{video_id}.mp4"
