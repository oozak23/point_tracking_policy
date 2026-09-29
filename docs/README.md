# docs/ — what was decided, built and measured

## Read these first

| document | what it is |
|---|---|
| `DECODER_REPORT.html` | **start here**: how the decoder was built, the loss and why, when and why proprioception was added, every training run and test. Open in a browser |
| `PROGRESS_REPORT.html` | the meeting summary: results against the baseline, charts, open decisions |
| `DECODER_LOG.md` | the step-by-step build log: what each step built, how it was verified, what went wrong |
| `DECODER_BUILD.md` | the build plan and the settled configuration, with the reason for each setting |

## Design decisions

| document | what it is |
|---|---|
| `DECISIONS.md` | the list of design questions (state, action space, chunking, frame rate, …), each marked open or settled |
| `ACTION_DECODER.md` | the decoder design note from the 2026-08-14 meeting: where robot points live, how many tokens become one action. Supersedes parts of `DECISIONS.md` |
| `CHANGES_7HZ.md` | the move from 3.6 Hz to 7.2 Hz data, file by file, including a bug found on the way |

## Experiment protocols, written before each run

Each fixes the pass/fail criteria before the results exist, so thresholds can't move afterwards.

| document | run |
|---|---|
| `PILOT_PLAN.md` | the first one-epoch run, and **the three gates every run is judged by** |
| `PROPRIO_PILOT_PLAN.md` | the one-epoch run with proprioception |
| `MULTIEPOCH_PLAN.md` | the five-epoch runs |

## Results of individual experiments

| document | what it is |
|---|---|
| `POINTS_TO_ACTIONS.md` | can the action be recovered from the gripper's points? Yes with exact points, no with Track-On's |

## The earlier phase: choosing query points

| document | what it is |
|---|---|
| `PLAN.md` | the original plan: points on the object and the robot → frozen Track-On → its memory as policy state |
| `RESULTS.md` | measurements from building the object-point selection: data access, segmentation, grounding |

These two predate the reorganisation, so they refer to the old `scripts/` folder. The script
numbers are unchanged; see the folder READMEs for where each script now lives.
