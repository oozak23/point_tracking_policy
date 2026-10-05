# Point-prediction policy — results so far

Protocol: `docs/POINT_POLICY_PLAN.md`. All numbers on the frozen 63,598 validation targets.
Branch `point-policy-ozan`.

## Runs

| | A: visual only | B: + coordinates | D: + coordinates + future points |
|---|---:|---:|---:|
| job | 1658338 | 1722184 | 1721709 |
| action error | 8.49 mm | **8.48 mm** | 9.93 mm |
| rotation | 1.64° | 1.64° | 1.89° |
| gripper F1 / timing cost | 0.60 / 3.64 | 0.59 / 3.78 | 0.32 / 6.31 |
| scene-shuffle effect (actions) | +8.74% | +8.20% | +6.10% |
| robot-history effect | +239% | +242% | +192% |
| gates passed | 0 / 3 | 0 / 3 | 0 / 3 |

Copy-last: 4.55 mm. Scene effect by difficulty, easiest → hardest: A +9.1 … +8.4%, B +8.2 … +7.8%,
D +7.1 … +5.1%; flat in every run.

Per epoch (action error, mm): A 10.21, 9.90, 9.27, 8.68, 8.49 · B 10.36, 9.99, 9.26, 8.66, 8.48 ·
D 13.69 (pre-training), 10.43, 10.11, 9.98, 9.93.

## Run C: action chunking (job 1724301)

16 actions per frame instead of one, run A's recipe otherwise. Overfit gate passed at update 500.

| | first step | 4 steps | 8 steps | 16 steps |
|---|---:|---:|---:|---:|
| C, endpoint error after chaining h actions | 10.6 mm | 41.4 | 76.1 | **128.9** |
| no motion | 13.0 | 49.6 | 90.8 | 149.6 |
| repeat the last action | **4.6** | **36.9** | 97.1 | 225.6 |
| **scene-shuffle effect on C's prediction** | +7.8% | +8.6% | **+10.6%** | **+11.3%** |

First-step action error 10.56 mm against A's 8.49; gripper F1 0.50, timing cost 4.62. Gates: 0 / 3.
Scene effect by difficulty (first step): +7.7 … +6.8%, flat as in every other run.

**The scene effect grows with the horizon** (+7.8% at one step to +11.3% at sixteen) and crosses the
10% gate at 8 and 16 steps. The same pattern as run D's point plan (+11.1% at 16 frames): the longer
the horizon, the more the object points matter. The cost is the near term: chunking is 2 mm worse
than A on the first step and loses to "repeat the last action" out to 4 steps.

## Run D's future points

| frames ahead | 1 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| D | **3.93 px** | **18.0** | **35.0** | **58.9** |
| points don't move | 6.84 | 25.5 | 45.5 | 72.9 |
| repeat last tracked motion | 4.44 | 22.8 | 50.8 | 104.9 |
| scene-shuffle effect on D's plan | +10.6% | +8.8% | +10.3% | +11.1% |

The point loss plateaued at ~0.54 from epoch 3 on, on training and validation windows alike.

**Perfect-plan bound:** fed the recorded future instead of its own plan, D's action error goes from
9.93 to 9.84 mm (−1%). The action head barely uses the plan.

## What it means

1. **Coordinates alone change nothing** (B = A on every measure).
2. **The future-point task makes the model use the scene** (+11% on the 16-frame plan, the first
   measure over 10%), **but a one-step action does not use that knowledge**: a perfect plan helps
   the actions by 1%.
3. **D's actions are worse than A's because of the future-point task**, not the coordinates.
   Likely causes: competition for the shared layers (λ = 1), and a schedule confound: the one
   cosine schedule gave D's high-learning-rate first epoch to pre-training.
4. The long-horizon information has to reach an output that needs it: action chunking (run C),
   and later D combined with chunked actions.
