"""Every location this project reads or writes, in one place.

Each can be overridden with an environment variable, so the same code runs from another
account without edits. The defaults reproduce the original setup on the KUIS cluster.

    PTP_DATA     the dataset: gen/, gt_all/, manifests, evaluation plans, loss scales and
                 the reference runs. Read by training and evaluation; written only by
                 pipeline/, which is what creates it.
    PTP_OUT      where new runs and reports are written.
    PTP_TRACK_ON, PTP_POINTWORLD, PTP_MOLMO_MOTION
                 the upstream repositories this code imports from.
    PTP_DROID_ANNOTATIONS
                 MolmoMotion's DROID annotations; only the pipeline needs them.

env.sh sets the same variables for the job scripts.

The shared copy of the dataset, /frozen/avg/ozan/point_tracking_policy_data, is readable
by the `avg` group and writable by nobody else, so PTP_OUT must never point inside it.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SHARED_DATA = Path("/frozen/avg/ozan/point_tracking_policy_data")


def _env(name: str, default) -> Path:
    return Path(os.environ.get(name) or default).expanduser()


# A checkout that already holds the dataset in out/ -- the original one -- keeps using it;
# a fresh clone reads the shared copy.
_LOCAL = REPO / "out"
DATA = _env("PTP_DATA", _LOCAL if (_LOCAL / "gen").is_dir() else SHARED_DATA)
OUT = _env("PTP_OUT", _LOCAL)

TRACK_ON = _env("PTP_TRACK_ON", "/scratch/oozak23/track_on")
POINTWORLD = _env("PTP_POINTWORLD", "/scratch/oozak23/PointWorld")
MOLMO_MOTION = _env("PTP_MOLMO_MOTION", "/scratch/oozak23/molmo-motion")
DROID_ANNOTATIONS = _env("PTP_DROID_ANNOTATIONS", "/scratch/oozak23/droid/annotations")

# Shared read-only inputs on /frozen; the same for every account.
DROID_RAW = Path("/frozen/avg/bora_data/droid_datasets/droid_raw_full_v2/droid_raw/1.0.1")
PW_FLOWS = Path("/frozen/avg/bora_data/droid_datasets/pointworld_droid_restored"
                "/droid/flows-fs-optimized")
MM_TRACKS_TAR = Path("/frozen/avg/ozan/molmo-motion-1m/droid/tracks/tracks-0000.tar")
