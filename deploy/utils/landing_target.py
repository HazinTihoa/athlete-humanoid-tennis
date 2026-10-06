"""Deployment-only Gaussian landing sampler; no training-runtime dependency."""

import numpy as np

from utils.tppo_student import quaternion_to_rotation_matrix_wxyz


class LandingTargetSampler:
    def __init__(self, mean_xy, std_xy, startup_pose, min_world_x, seed=0):
        self.mean = np.asarray(mean_xy, dtype=float)
        self.std = np.asarray(std_xy, dtype=float)
        pose = np.asarray(startup_pose, dtype=float)
        if (self.mean.shape != (2,) or self.std.shape != (2,)
                or pose.shape != (7,) or not np.isfinite(pose).all()
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or np.any(self.std < 0) or not np.isfinite(min_world_x)):
            raise ValueError("Invalid landing target sampling configuration")
        rotation = quaternion_to_rotation_matrix_wxyz(pose[3:])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        self.rotation = np.array([[c, -s], [s, c]])
        self.origin = pose[:2].copy()
        self.min_world_x = float(min_world_x)
        self.rng = np.random.default_rng(seed)

    def sample(self):
        xy = self.origin + self.rotation @ self.rng.normal(self.mean, self.std)
        # Match training's clamp, not rejection sampling.
        xy[0] = max(xy[0], self.min_world_x)
        return np.array([xy[0], xy[1], 0.0])
