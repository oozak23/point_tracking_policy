# checks/ — the verification suite

**Run these after any change to the code, and before trusting a new result.** They take about
five minutes on a CPU, and all of them must pass.

```bash
source env.sh
for s in 45_verify_dataloader 46_verify_replica 48_verify_readout 49_verify_head \
         51_verify_loss 54_verify_training 59_verify_validation 70_verify_proprio; do
  $PY_TRACKON -B checks/$s.py || echo "FAILED: $s"
done
$PY_POINTWORLD -B checks/61_verify_rollout.py      # needs the kinematics env
$PY_POINTWORLD -B checks/82_verify_pose_fit.py
```

Most checks work **by perturbation**: change an input that must not matter and require the
output to stay bit-identical; change one that must matter and require it to move. Reading code
and agreeing with it is not evidence. Checks like these caught two real bugs that code review
missed: a memory that was never actually read, and NaN gradients from fully masked attention.

| script | what it proves | time |
|---|---|---:|
| `45_verify_dataloader.py` | windows are causal and correctly aligned; padding and masks behave | ~1 min |
| `46_verify_replica.py` | the replica block is causal; the memory is live (changing the past changes later frames); point order and padding don't leak | ~30 s |
| `48_verify_readout.py` | the readout ignores padded points (even NaNs there) and point order | ~20 s |
| `49_verify_head.py` | action-head conventions; 6D rotations round-trip exactly; gradients reach every layer | ~20 s |
| `51_verify_loss.py` | the loss matches hand-computed values; scales come from training labels only, never validation | ~20 s |
| `54_verify_training.py` | optimizer steps really update weights; resuming from a checkpoint gives the identical next update | ~20 s |
| `59_verify_validation.py` | baselines, gripper-event matching and shuffle-donor alignment | ~25 s |
| `61_verify_rollout.py` | chaining actions over time is correct on a real clip (`$PY_POINTWORLD`) | ~10 s |
| `70_verify_proprio.py` | proprioception reads only the current frame, never the next | ~20 s |
| `82_verify_pose_fit.py` | the points → actions converter: exact recovery of known poses, agreement with the action labels (`$PY_POINTWORLD`) | ~5 s |

Several checks load real clips, so they need the dataset (`PTP_DATA`).

## Not part of the routine suite

| script | why it's separate |
|---|---|
| `21_verify_7hz.py` | the one-off gate for moving the dataset from 3.6 Hz to 7.2 Hz; needs smoke-test outputs that aren't in the shared data |
| `26_verify_reset.sbatch` | a GPU job confirming Track-On's per-clip reset clears all state; only relevant when regenerating tokens |
| `71_audit_proprio_data.py` | a one-time audit that every clip's sensor data is complete and finite (all 13,449 passed); writes a JSON report to `$PTP_OUT` |
