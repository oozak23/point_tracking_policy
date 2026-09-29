"""Stage-1 grounding: caption -> object noun phrase -> a 2D point on the object.

Thin wrapper over MolmoMotion's own code (imported, never copied, per the workspace
convention on vendored repos):

    Qwen3-0.6B       noun-phrase extraction from the DROID caption
    Molmo2-8B        visual re-caption when that phrase is vague
    MolmoPoint-Vid   point at the object in the video, returning (frame_idx, x, y)

DROID captions are mostly unusable on their own: 454 unique captions across 23,532
clips and 84.3% contain a vague noun, so the Molmo2-8B re-caption branch is the main
path, not an edge case. Molmo2-8B is ~16 GB in bf16 and does NOT fit the 16 GB login
A4000 -- run those clips on SLURM. `plan_clips` splits a clip list on that boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

MOLMO_MOTION = Path("/scratch/oozak23/molmo-motion")
_SAM3_DIR = MOLMO_MOTION / "data_generation" / "third_party" / "sam3"
if str(_SAM3_DIR) not in sys.path:
    sys.path.insert(0, str(_SAM3_DIR))

# Prompt templates, verbatim from data_generation/configs/robot.yaml
AGENT = "robot gripper"
POINT_PROMPT_TEMPLATE = "point to {obj} gripped and picked up by the {agent}"


def load_noun_extractor():
    from molmo2_pointing import load_qwen3

    return load_qwen3()


def extract_noun(qwen_model, qwen_tokenizer, caption: str) -> str:
    from molmo2_pointing import extract_object_with_qwen3

    return extract_object_with_qwen3(qwen_model, qwen_tokenizer, caption, agent=AGENT)


def is_vague(obj: str) -> bool:
    from molmo2_pointing import is_vague_object

    return is_vague_object(obj)


# Words allowed in an extraction without appearing in the caption.
_FREE_WORDS = frozenset({"a", "an", "the", "of", "and", "or", "with", "in", "on", "|"})


def is_hallucinated(obj: str, caption: str) -> bool:
    """True if the extraction invents words that are not in the caption.

    Qwen3-0.6B leaks its own few-shot examples on DROID's templated captions --
    observed: "Hang or unhang object" -> "container | strawberry toy | moose toy",
    "Move object to a new position and orientation" -> "red plush toy". Both are
    verbatim from the example list in extract_object_with_qwen3().

    This matters because is_vague_object() does NOT catch them: "red plush toy" is a
    perfectly concrete phrase, so a hallucinated clip would otherwise skip the
    re-caption branch and get segmented against an object that is not in the video.
    The extraction prompt already demands "Use ONLY words from the task description",
    so this just enforces what was asked for.
    """
    if not obj:
        return False
    import re

    cap_words = set(re.findall(r"[a-z0-9]+", caption.lower()))
    obj_words = set(re.findall(r"[a-z0-9]+", obj.lower())) - _FREE_WORDS
    return bool(obj_words - cap_words)


def needs_recaption(obj: str, caption: str) -> bool:
    """A clip needs Molmo2-8B if the caption yields nothing usable by itself."""
    return (not obj) or is_vague(obj) or is_hallucinated(obj, caption)


def load_pointer():
    from molmo2_pointing import load_molmo2

    return load_molmo2()


# MolmoPoint's own no-object-name prompt (molmo2_pointing.py:35). If this works on
# DROID it removes the entire grounding chain -- no Qwen3, no Molmo2-8B re-caption,
# and none of DROID's caption problems (84.3% vague, plus Qwen3 leaking its few-shot
# examples on the templated ones).
TEXT_FREE_PROMPT = "Point to all objects in contact and moved by robot arm"


def point_prompt(obj_phrase: str | None) -> str:
    if not obj_phrase:
        return TEXT_FREE_PROMPT
    return POINT_PROMPT_TEMPLATE.format(obj=obj_phrase, agent=AGENT)


def point_at_object(molmo_model, molmo_processor, video_path: str, obj_phrase: str):
    """-> list of (frame_idx, x, y) in the given video's frame/pixel space.

    Pass the RELEASE video (854x480 @ 15 fps): frame_idx then indexes release frames
    directly, matching the T axis of the tracks NPZs.
    """
    from molmo2_pointing import detect_manipulated_objects

    return detect_manipulated_objects(
        molmo_model, molmo_processor, video_path=video_path, prompt=point_prompt(obj_phrase)
    )


def split_objects(phrase: str) -> list[str]:
    """Qwen3 returns multiple grasped objects separated by ' | '."""
    return [o.strip() for o in (phrase or "").split("|") if o.strip()]


def plan_clips(records, qwen_model, qwen_tokenizer, verbose: bool = True):
    """Extract the noun for each clip and split by whether re-captioning is needed.

    Returns (direct, needs_recaption) lists of (record, noun_phrase). Only `direct`
    can run end to end on the 16 GB login GPU.
    """
    direct, needs = [], []
    for i, rec in enumerate(records):
        obj = extract_noun(qwen_model, qwen_tokenizer, rec.caption)
        (needs if (not obj or is_vague(obj)) else direct).append((rec, obj))
        if verbose and (i + 1) % 25 == 0:
            print(f"  [plan] {i+1}/{len(records)}  direct={len(direct)} recaption={len(needs)}")
    return direct, needs
