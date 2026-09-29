"""Frame rates and the mappings between the three frame spaces. One source of truth.

Three timelines exist and they are all different:

    trajectory step   DROID's `trajectory.h5`, one per source video frame. ~14.3 Hz.
    PointWorld step   trajectory / PW_SUBSAMPLE. ~7.2 Hz. Fixed by PointWorld -- their
                      windows, joint states, calibration and action labels are indexed
                      here, so it is not ours to change.
    release frame     trajectory / SUBSAMPLE. What our reconstructed MP4s contain, and
                      the space Track-On query points live in.

`SUBSAMPLE` is the only knob. At 2 the release space coincides with PointWorld's, which
is what we generate the policy dataset at; at 4 it reproduces the MolmoMotion release.

Why 14.3 and not 60 or 15
-------------------------
The raw DROID MP4s carry a 60 fps container header and `droid_videos_index.json` claims
source 60 Hz -> release 15 Hz. Both are wrong. Each source frame corresponds 1:1 to a
`trajectory.h5` step (measured over 10 random episodes: source_frames / trajectory_length
= 1.00), and the trajectory rate from PointWorld's `canonical_timestamps` -- absolute
wall-clock ms -- is 14.1-14.5 Hz. So the source is ~14.3 Hz and the MolmoMotion release
is ~3.6 Hz, not 15. Do not use the ratio source/release = 4 as evidence for either
reading; it holds under both. See RESULTS.md §1.

Changing SUBSAMPLE invalidates every artifact generated under the old value, because
release frame indices renumber. Reconstructed videos must go to a rate-tagged directory
-- `video_io.reconstruct` returns a cached file if the path exists, and a stale 3.6 Hz
MP4 read as 7.2 Hz produces wrong frame indices everywhere with nothing raising.
"""
from __future__ import annotations

SUBSAMPLE = 2          # source frames per release frame. 2 -> 7.2 Hz, 4 -> 3.6 Hz.
PW_SUBSAMPLE = 2       # source frames per PointWorld step. Fixed by PointWorld.
MM_SUBSAMPLE = 4       # source frames per MolmoMotion release frame. FIXED AT 4 FOREVER.

SOURCE_HZ = 14.3       # measured, not the container header
RELEASE_HZ = SOURCE_HZ / SUBSAMPLE
PW_HZ = SOURCE_HZ / PW_SUBSAMPLE

if SUBSAMPLE % PW_SUBSAMPLE:
    raise ValueError(
        f"SUBSAMPLE={SUBSAMPLE} is not a multiple of PW_SUBSAMPLE={PW_SUBSAMPLE}; "
        "release frames would not land on PointWorld steps and the mappings below "
        "would be lossy in both directions."
    )

#: release frames per PointWorld step. 1 at 7.2 Hz, 2 at 3.6 Hz.
RATIO = SUBSAMPLE // PW_SUBSAMPLE

#: memory horizon in seconds, for the M=72 inference-time memory
MEMORY_SECONDS = 72 / RELEASE_HZ


def release_to_pw_step(release_frame):
    """Release frame -> PointWorld step. Exact; vectorised over ndarray or int."""
    return release_frame * RATIO


def pw_step_to_release(pw_step):
    """PointWorld step -> release frame. Vectorised over ndarray or int.

    LOSSY when RATIO > 1: at 3.6 Hz two PointWorld steps share one release frame, so this
    is a floor and the inverse of `release_to_pw_step` only up to that collapse. That
    aliasing is what made the `19_robot_points.py` birth-frame bug hard to read -- one of
    the two steps could have the arm out of frame and the other in it. At 7.2 Hz
    (RATIO == 1) both directions are the identity and the ambiguity does not exist.
    """
    return pw_step // RATIO


def mm_frame_to_release(mm_frame):
    """MolmoMotion release frame -> OUR release frame. Vectorised.

    `MM_SUBSAMPLE` is 4 permanently, because it is a property of the *published dataset*
    and not of our pipeline: `droid_clips.json`'s `num_frames` and `motion_ranges`, and
    every index in the released `*_2d.npz` tracks, are in MolmoMotion's 3.6 Hz space. It
    does not follow SUBSAMPLE, and `index.py` correctly keeps using it to reach source
    frames.

    What DOES change: at SUBSAMPLE=2 our video has twice as many frames as theirs, so any
    MolmoMotion index must be scaled before it addresses one of our frames. Frame 0 maps
    to frame 0, which is why the frame-0 seeding path (`mm_sam_f0`, the winning variant)
    is rate-invariant and needs no conversion. Everything else does -- in particular
    `motion_ranges` as used by `--ground-in-motion-window` and `video_io.trim`, and
    `14_molmomotion_as_queries.py --frame N` for N > 0.
    """
    return mm_frame * (MM_SUBSAMPLE // SUBSAMPLE)


def release_to_mm_frame(release_frame):
    """OUR release frame -> MolmoMotion release frame. Lossy when SUBSAMPLE < 4."""
    return release_frame // (MM_SUBSAMPLE // SUBSAMPLE)


def describe() -> str:
    """One line for logs and dataset provenance, so an artifact records its own rate."""
    return (f"SUBSAMPLE={SUBSAMPLE} release={RELEASE_HZ:.1f}Hz "
            f"pw={PW_HZ:.1f}Hz ratio={RATIO} memory={MEMORY_SECONDS:.1f}s")
