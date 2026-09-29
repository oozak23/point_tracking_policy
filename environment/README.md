# environment/ — the three conda environments

The project spans three upstream code stacks with conflicting dependencies, so it uses three
environments. **They are not interchangeable**: each script says which one it needs, and
`env.sh` gives each an interpreter variable.

| env | Python | torch | variable | used for | notably lacks |
|---|---|---|---|---|---|
| `track_on_r` | 3.12 | 2.4.1 | `$PY_TRACKON` | the model, training, evaluation, most checks, Track-On tokens | h5py, kinematics |
| `molmo-motion` | 3.11 | 2.9.1 (CUDA 12.8) | `$PY_MOLMO` | SAM 2.1, object points, ground-truth export | kinematics |
| `pointworld-env` | 3.10 | 2.5.1 (CUDA 12.4) | `$PY_POINTWORLD` | forward kinematics: robot points, action labels, rollout evaluation | transformers |

**For training and evaluation you only need `track_on_r`.** The other two are for rebuilding
the dataset, plus `pointworld-env` for the rollout check and evaluation.

## Create them

```bash
conda env create -f environment/track_on_r.yml
conda env create -f environment/molmo-motion.yml
conda env create -f environment/pointworld-env.yml
```

Then put the interpreter paths in `env.local.sh` at the repository root (see the main README).

## Known gaps

- **`molmo-motion` also needs the MolmoMotion package itself**, installed from its repository:
  `pip install -e /path/to/molmo-motion[viz]`. It is left out of the spec because it is a local
  install, not something pip can download.
- **Track-On and PointWorld are not installed as packages.** The code imports them from their
  checkouts, found through `PTP_TRACK_ON` and `PTP_POINTWORLD`.
- The specs were exported on 2026-09-29 from the working environments, without build strings.
  If conda can't solve one exactly, the torch version is what matters most; relax the rest.
