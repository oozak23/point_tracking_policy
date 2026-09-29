"""Offline composition of EE body deltas; no counterfactual observations or IK."""
import numpy as np


def integrate(p, R, dp, dR):
    """Compose K body-frame actions from a measured initial flange pose."""
    p, R = np.asarray(p, dtype=np.float64).copy(), np.asarray(R, dtype=np.float64).copy()
    for translation, rotation in zip(dp, dR):
        p += R @ translation
        R = R @ rotation
    return p, R


def pose_error(p, R, target_p, target_R):
    angle = np.arccos(np.clip((np.trace(R.T @ target_R)-1)/2, -1, 1))
    return float(np.linalg.norm(p-target_p)*1000), float(np.degrees(angle))


def place_local_points(local, p, R):
    return np.asarray(local) @ R.T + p


class RolloutMetrics:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def add(self, name, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError(f'nonfinite rollout metric: {name}')
        self.sums[name] = self.sums.get(name, 0.) + float(values.sum())
        self.counts[name] = self.counts.get(name, 0) + len(values)

    def result(self):
        return {name: {'mean': self.sums[name]/n if n else None, 'count': n}
                for name, n in sorted(self.counts.items())}
