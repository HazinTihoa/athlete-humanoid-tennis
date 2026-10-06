#!/usr/bin/env python3
"""Replay G1 state plus raw, Bridge-filtered, and Kalman ball streams.

Offline reader only: no ROS nodes, policy inference, motor commands or mj_step.
All streams use bag receive timestamps. Stamped messages also retain their source
timestamps, which are used to recover the bridge's piecewise frame calibration.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
import sys
import threading
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

DEPLOY_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = DEPLOY_DIR.parent
sys.path.insert(0, str(DEPLOY_DIR))

from utils.tppo_simulation_physics import build_training_aligned_model

RAW_COLOR = np.array([1.0, 0.48, 0.05, 0.95], dtype=np.float32)
FILTERED_COLOR = np.array([0.08, 0.55, 1.0, 0.80], dtype=np.float32)
KALMAN_COLOR = np.array([1.0, 0.85, 0.05, 0.95], dtype=np.float32)
POLICY_BALL_COLOR = np.array([1.0, 0.05, 0.05, 1.0], dtype=np.float32)
OBSERVED_SWEET_COLOR = np.array([1.0, 0.12, 0.72, 0.95], dtype=np.float32)
FK_SWEET_COLOR = np.array([0.2, 1.0, 0.4, 1.0], dtype=np.float32)
SWEET_ERROR_COLOR = np.array([1.0, 1.0, 1.0, 0.7], dtype=np.float32)
MOCAP_FRAME_COLORS = (
    np.array([1.0, 0.1, 0.1, 0.70], dtype=np.float32),
    np.array([0.1, 1.0, 0.1, 0.70], dtype=np.float32),
    np.array([0.1, 0.3, 1.0, 0.70], dtype=np.float32),
)
IDENTITY = np.eye(3).ravel()
INTENT_OBSERVATION_DIMS = frozenset((628, 631, 638))
SWEET_SPOT_POSITION_SLICE = slice(128, 131)
POLICY_BALL_POSITION_SLICE = slice(104, 107)


def resolve_path(value: str | Path, *bases: Path) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [path, *(b / path for b in bases)]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(str(path))


def load_config(value: str | Path) -> tuple[dict, Path]:
    path = resolve_path(value, DEPLOY_DIR / "configs", REPO_ROOT)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return config, path


@dataclass
class Samples:
    time: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    source_time: np.ndarray

    @classmethod
    def from_rows(cls, rows: list, width: int) -> Samples:
        if not rows:
            return cls(np.empty(0), np.empty((0, width)), np.empty(0, bool), np.empty(0))
        t, v, valid, source = zip(*rows)
        order = np.argsort(t, kind="stable")
        return cls(np.asarray(t)[order], np.asarray(v, dtype=float)[order],
                   np.asarray(valid, dtype=bool)[order], np.asarray(source)[order])

    def index(self, timestamp: float) -> int:
        return int(np.searchsorted(self.time, timestamp, side="right")) - 1

    def current(self, timestamp: float, max_age: float, *, hold_valid=False) -> tuple[np.ndarray | None, float]:
        index = self.index(timestamp)
        if hold_valid:
            while index >= 0 and not self.valid[index]:
                index -= 1
        if index < 0:
            return None, float("inf")
        age = timestamp - self.time[index]
        if not self.valid[index] or age > max_age:
            return None, age
        return self.values[index], age


@dataclass
class Calibration:
    start: float
    rotation: np.ndarray
    translation: np.ndarray
    pairs: int
    position_p95: float
    orientation_p95_deg: float


def recover_calibrations(raw: Samples, published: Samples, requests: np.ndarray) -> list[Calibration]:
    """Recover yaw/XY changes from paired Pelvis poses, even while stationary.

    Bridge publishes the most recently received valid raw pose. Pair against its
    publication stamp, not the raw camera stamp (which precedes transport).
    Rejected/repeated calibration requests simply produce identical transforms.
    """
    keep = raw.valid
    if not np.any(keep) or not len(published.time):
        raise ValueError("Frame alignment needs valid raw AND published Pelvis poses.")
    raw_t, raw_pose = raw.time[keep], raw.values[keep]
    cuts = np.unique(np.concatenate(([-np.inf], requests, [np.inf])))
    result = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        chosen = np.flatnonzero(published.valid & (published.time >= start) & (published.time < end))
        if not len(chosen):
            continue
        pub = published.values[chosen]
        query = published.source_time[chosen]
        ids = np.searchsorted(raw_t, query, side="right") - 1
        clipped = np.clip(ids, 0, len(raw_t) - 1)
        usable = (ids >= 0) & (query - raw_t[clipped] <= 0.05)
        if np.count_nonzero(usable) < 3:
            raise ValueError(f"Not enough synchronized Pelvis pairs after t={start:.3f}s.")
        a, b = raw_pose[clipped[usable]], pub[usable]
        relative = Rotation.from_quat(b[:, 3:]) * Rotation.from_quat(a[:, 3:]).inv()
        matrices = relative.as_matrix()
        yaws = np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0])
        center = np.arctan2(np.sin(yaws).mean(), np.cos(yaws).mean())
        yaw = center + np.median(np.angle(np.exp(1j * (yaws - center))))
        rotation = Rotation.from_euler("z", yaw)
        translations = b[:, :3] - rotation.apply(a[:, :3])
        translation = np.median(translations, axis=0)
        # The deployment calibration never shifts the floor height.
        translation[2] = 0.0
        errors = np.linalg.norm(rotation.apply(a[:, :3]) + translation - b[:, :3], axis=1)
        angles = np.rad2deg((rotation.inv() * relative).magnitude())
        p95, a95 = float(np.quantile(errors, .95)), float(np.quantile(angles, .95))
        if p95 > .02 or a95 > 1.0:
            raise ValueError(
                f"Frame calibration is inconsistent after t={start:.3f}s: "
                f"Pelvis P95={p95:.4f}m/{a95:.2f}deg. "
                "Check missing calibration events, body names or changed bridge extrinsics."
            )
        result.append(Calibration(float(start), rotation.as_matrix(), translation,
                                  len(errors), p95, a95))
    if not result:
        raise ValueError("No usable frame calibration found.")
    return result


def calibration_indices(times: np.ndarray, calibrations: list[Calibration]) -> np.ndarray:
    return np.searchsorted([c.start for c in calibrations], times, side="right") - 1


def transform_raw_positions(samples: Samples, calibrations: list[Calibration]) -> Samples:
    values = samples.values.copy()
    ids = calibration_indices(samples.time, calibrations)
    for index, calibration in enumerate(calibrations):
        keep = ids == index
        values[keep, :3] = values[keep, :3] @ calibration.rotation.T + calibration.translation
    # Before the first calibration the bridge publishes in the original world.
    return Samples(samples.time, values, samples.valid, samples.source_time)


def local_positions_to_world(samples: Samples, pelvis: Samples, max_root_age=.25) -> Samples:
    """Transform recorded Pelvis-frame points with the root available at each sample."""
    values = np.full_like(samples.values, np.nan)
    valid = samples.valid.copy()
    for index, (timestamp, position_b) in enumerate(zip(samples.time, samples.values)):
        root, _ = pelvis.current(timestamp, max_root_age, hold_valid=True)
        if not valid[index] or root is None:
            valid[index] = False
            continue
        values[index] = root[:3] + Rotation.from_quat(root[3:]).apply(position_b)
    return Samples(samples.time, values, valid, samples.source_time)


@dataclass(frozen=True)
class OfflineKalmanConfig:
    """Physics-aware EKF settings used only by this replay tool."""

    radius_m: float
    mass_kg: float
    air_density_kg_m3: float
    drag_coefficient: float
    court_restitution: float
    tangent_speed_retention: float
    ground_height_m: float
    integration_dt_s: float
    measurement_noise_m: float = .005
    process_acceleration_noise_mps2: float = 8.0
    initial_velocity_std_mps: float = 5.0
    innovation_gate_chi2: float = 16.27
    maximum_speed_mps: float = 40.0
    maximum_prediction_s: float = 2.0
    reassociation_timeout_s: float = .50
    static_speed_threshold_mps: float = .15
    static_hold_s: float = .15
    initialization_samples: int = 5
    reacquisition_samples: int = 3

    @classmethod
    def from_deploy_config(cls, config: dict) -> OfflineKalmanConfig:
        bridge = config.get("mocap_bridge", {})
        physics = config.get("tennis_ball_physics", {})
        return cls(
            radius_m=float(physics.get("radius_m", .0335)),
            mass_kg=float(physics.get("mass_kg", .0577)),
            air_density_kg_m3=float(physics.get("air_density_kg_m3", 1.225)),
            drag_coefficient=float(physics.get("drag_coefficient", .55)),
            court_restitution=float(bridge.get("court_restitution", .745)),
            tangent_speed_retention=float(
                bridge.get("ground_tangent_speed_retention", .825)
            ),
            ground_height_m=float(bridge.get("ground_height_m", 0.0)),
            integration_dt_s=float(bridge.get("ball_integration_dt_s", .0025)),
            measurement_noise_m=float(bridge.get("replay_kalman_measurement_noise_m", .005)),
            process_acceleration_noise_mps2=float(
                bridge.get("replay_kalman_process_acceleration_noise_mps2", 8.0)
            ),
            initial_velocity_std_mps=float(
                bridge.get("replay_kalman_initial_velocity_std_mps", 5.0)
            ),
            innovation_gate_chi2=float(
                bridge.get("replay_kalman_innovation_gate_chi2", 16.27)
            ),
            maximum_speed_mps=float(bridge.get("ball_maximum_speed_mps", 40.0)),
            maximum_prediction_s=float(
                bridge.get("ball_maximum_prediction_s", 2.0)
            ),
            reassociation_timeout_s=float(
                bridge.get("ball_reassociation_timeout_s", .50)
            ),
            static_speed_threshold_mps=float(
                bridge.get("ball_static_speed_threshold_mps", .15)
            ),
            static_hold_s=float(bridge.get("ball_static_hold_s", .15)),
        )

    def validate(self) -> None:
        positive = (
            self.radius_m,
            self.mass_kg,
            self.air_density_kg_m3,
            self.integration_dt_s,
            self.measurement_noise_m,
            self.process_acceleration_noise_mps2,
            self.initial_velocity_std_mps,
            self.innovation_gate_chi2,
            self.maximum_speed_mps,
            self.maximum_prediction_s,
            self.reassociation_timeout_s,
        )
        if not all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError("Offline Kalman positive parameters must be finite and positive.")
        if not 0 <= self.court_restitution <= 1 or not 0 <= self.tangent_speed_retention <= 1:
            raise ValueError("Offline Kalman restitution values must be in [0, 1].")
        if self.drag_coefficient < 0 or self.static_speed_threshold_mps < 0:
            raise ValueError("Offline Kalman drag/static thresholds must be nonnegative.")
        if self.initialization_samples < 2 or self.reacquisition_samples < 2:
            raise ValueError("Offline Kalman sample counts must be at least two.")


class PhysicsBallKalman:
    """Causal position/velocity EKF with drag, gravity, bounce and outlier gating."""

    def __init__(self, config: OfflineKalmanConfig):
        config.validate()
        self.config = config
        self.state = np.zeros(6, dtype=np.float64)
        self.covariance = np.eye(6, dtype=np.float64)
        self.state_time_s: float | None = None
        self.last_measurement_time_s: float | None = None
        self.is_static = True
        self.static_candidate_time_s: float | None = None
        self.measurements: deque[tuple[float, np.ndarray]] = deque(maxlen=7)
        self.reacquisition: deque[tuple[float, np.ndarray]] = deque(
            maxlen=config.reacquisition_samples
        )
        self.accepted_measurements = 0
        self.rejected_measurements = 0
        self.reinitializations = 0

    @property
    def initialized(self) -> bool:
        return self.state_time_s is not None

    def reset(self) -> None:
        self.state.fill(0.0)
        self.covariance[:] = np.eye(6)
        self.state_time_s = None
        self.last_measurement_time_s = None
        self.is_static = True
        self.static_candidate_time_s = None
        self.measurements.clear()
        self.reacquisition.clear()

    def _initialize(self, position: np.ndarray, timestamp_s: float) -> None:
        self.state[:3] = position
        self.state[2] = max(
            self.state[2], self.config.ground_height_m + self.config.radius_m
        )
        self.state[3:] = 0.0
        position_variance = self.config.measurement_noise_m**2
        velocity_variance = self.config.initial_velocity_std_mps**2
        self.covariance[:] = np.diag(
            [position_variance] * 3 + [velocity_variance] * 3
        )
        self.state_time_s = timestamp_s
        self.last_measurement_time_s = timestamp_s
        self.is_static = True
        self.static_candidate_time_s = timestamp_s
        self.measurements.clear()
        self.measurements.append((timestamp_s, position.copy()))
        self.reacquisition.clear()

    def _velocity_fit(
        self, values: deque[tuple[float, np.ndarray]], minimum_samples: int
    ) -> np.ndarray | None:
        if len(values) < minimum_samples:
            return None
        times = np.asarray([sample[0] for sample in values], dtype=np.float64)
        positions = np.asarray([sample[1] for sample in values], dtype=np.float64)
        times -= times[-1]
        if times[-1] - times[0] <= 1e-6:
            return None
        design = np.column_stack((np.ones_like(times), times))
        coefficients, _, _, _ = np.linalg.lstsq(design, positions, rcond=None)
        velocity = coefficients[1]
        if not np.isfinite(velocity).all():
            return None
        if np.linalg.norm(velocity) > self.config.maximum_speed_mps:
            return None
        return velocity

    def _predict_state(
        self, state: np.ndarray, covariance: np.ndarray, duration_s: float
    ) -> tuple[np.ndarray, np.ndarray, int]:
        state = state.copy()
        covariance = covariance.copy()
        if duration_s <= 0 or self.is_static:
            return state, covariance, 0
        area = np.pi * self.config.radius_m**2
        drag_factor = (
            .5
            * self.config.air_density_kg_m3
            * self.config.drag_coefficient
            * area
            / self.config.mass_kg
        )
        ground_z = self.config.ground_height_m + self.config.radius_m
        acceleration_noise = self.config.process_acceleration_noise_mps2**2
        remaining = duration_s
        bounce_count = 0
        eye = np.eye(3)
        while remaining > 1e-12:
            dt = min(self.config.integration_dt_s, remaining)
            position, velocity = state[:3], state[3:]
            speed = float(np.linalg.norm(velocity))
            acceleration = -drag_factor * speed * velocity
            acceleration[2] -= 9.81
            if speed > 1e-9:
                acceleration_jacobian = -drag_factor * (
                    speed * eye + np.outer(velocity, velocity) / speed
                )
            else:
                acceleration_jacobian = np.zeros((3, 3))
            transition = np.eye(6)
            transition[:3, 3:] = dt * eye + .5 * dt**2 * acceleration_jacobian
            transition[3:, 3:] = eye + dt * acceleration_jacobian
            noise_map = np.vstack((.5 * dt**2 * eye, dt * eye))
            covariance = (
                transition @ covariance @ transition.T
                + acceleration_noise * noise_map @ noise_map.T
            )
            next_velocity = velocity + acceleration * dt
            next_position = position + .5 * (velocity + next_velocity) * dt
            if next_position[2] < ground_z and next_velocity[2] < 0:
                next_position[2] = ground_z
                next_velocity[:2] *= self.config.tangent_speed_retention
                next_velocity[2] *= -self.config.court_restitution
                impact = np.eye(6)
                impact[2, 2] = 0.0
                impact[3, 3] = self.config.tangent_speed_retention
                impact[4, 4] = self.config.tangent_speed_retention
                impact[5, 5] = -self.config.court_restitution
                covariance = impact @ covariance @ impact.T
                covariance[2, 2] += self.config.measurement_noise_m**2
                covariance[3:, 3:] += np.eye(3) * .25
                bounce_count += 1
            state[:3], state[3:] = next_position, next_velocity
            remaining -= dt
        covariance = .5 * (covariance + covariance.T)
        return state, covariance, bounce_count

    def _confirmed_reacquisition(self) -> np.ndarray | None:
        velocity = self._velocity_fit(
            self.reacquisition, self.config.reacquisition_samples
        )
        if velocity is None:
            return None
        times = np.asarray([sample[0] for sample in self.reacquisition])
        if np.max(np.diff(times)) > .05:
            return None
        positions = np.asarray([sample[1] for sample in self.reacquisition])
        centered_time = times - times[-1]
        intercept = positions[-1]
        residual = positions - (intercept + centered_time[:, None] * velocity)
        if np.max(np.linalg.norm(residual, axis=1)) > max(
            .02, 4 * self.config.measurement_noise_m
        ):
            return None
        return velocity

    def _enforce_ground(self) -> None:
        ground_z = self.config.ground_height_m + self.config.radius_m
        if self.state[2] >= ground_z:
            return
        self.state[2] = ground_z
        if self.state[5] < 0:
            self.state[3:5] *= self.config.tangent_speed_retention
            self.state[5] *= -self.config.court_restitution
        self.covariance[2, 2] += self.config.measurement_noise_m**2
        self.covariance[3:, 3:] += np.eye(3) * .25

    def observe(self, position_w: np.ndarray, timestamp_s: float) -> None:
        position = np.asarray(position_w, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            return
        if not np.isfinite(timestamp_s):
            return
        if not self.initialized:
            self._initialize(position, timestamp_s)
            self.accepted_measurements += 1
            return
        assert self.state_time_s is not None
        assert self.last_measurement_time_s is not None
        if timestamp_s <= self.last_measurement_time_s:
            return
        gap_s = timestamp_s - self.last_measurement_time_s
        if gap_s > self.config.reassociation_timeout_s:
            self._initialize(position, timestamp_s)
            self.reinitializations += 1
            self.accepted_measurements += 1
            return

        if self.is_static:
            self.measurements.append((timestamp_s, position.copy()))
            velocity = self._velocity_fit(
                self.measurements, self.config.initialization_samples
            )
            displacement = float(np.linalg.norm(position - self.state[:3]))
            if velocity is not None and np.linalg.norm(velocity) > self.config.static_speed_threshold_mps:
                self.state[:3], self.state[3:] = position, velocity
                self.covariance[:] = np.diag(
                    [self.config.measurement_noise_m**2] * 3 + [.5**2] * 3
                )
                self.is_static = False
                self.static_candidate_time_s = None
            elif displacement <= max(.10, 8 * self.config.measurement_noise_m):
                variance = self.config.measurement_noise_m**2
                gain = self.covariance[:3, :3] @ np.linalg.inv(
                    self.covariance[:3, :3] + np.eye(3) * variance
                )
                self.state[:3] += gain @ (position - self.state[:3])
                self.covariance[:3, :3] = (
                    np.eye(3) - gain
                ) @ self.covariance[:3, :3]
                self.reacquisition.clear()
            else:
                self.reacquisition.append((timestamp_s, position.copy()))
                confirmed = self._confirmed_reacquisition()
                if confirmed is not None:
                    self._initialize(position, timestamp_s)
                    self.state[3:] = confirmed
                    self.is_static = False
                    self.reinitializations += 1
                    self.accepted_measurements += 1
                else:
                    self.rejected_measurements += 1
                return
            self.state_time_s = timestamp_s
            self.last_measurement_time_s = timestamp_s
            self._enforce_ground()
            self.accepted_measurements += 1
            return

        predicted, predicted_covariance, _ = self._predict_state(
            self.state, self.covariance, timestamp_s - self.state_time_s
        )
        measurement_covariance = np.eye(3) * self.config.measurement_noise_m**2
        innovation = position - predicted[:3]
        innovation_covariance = predicted_covariance[:3, :3] + measurement_covariance
        innovation_chi2 = float(
            innovation @ np.linalg.solve(innovation_covariance, innovation)
        )
        if innovation_chi2 > self.config.innovation_gate_chi2:
            self.reacquisition.append((timestamp_s, position.copy()))
            confirmed = self._confirmed_reacquisition()
            if confirmed is not None:
                self._initialize(position, timestamp_s)
                self.state[3:] = confirmed
                self.is_static = False
                self.reinitializations += 1
                self.accepted_measurements += 1
            else:
                self.rejected_measurements += 1
            return

        kalman_gain = predicted_covariance[:, :3] @ np.linalg.inv(
            innovation_covariance
        )
        self.state[:] = predicted + kalman_gain @ innovation
        identity_minus_kh = np.eye(6)
        identity_minus_kh[:, :3] -= kalman_gain
        self.covariance[:] = (
            identity_minus_kh @ predicted_covariance @ identity_minus_kh.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )
        self._enforce_ground()
        self.state_time_s = timestamp_s
        self.last_measurement_time_s = timestamp_s
        self.measurements.append((timestamp_s, position.copy()))
        self.reacquisition.clear()
        self.accepted_measurements += 1
        speed = float(np.linalg.norm(self.state[3:]))
        if speed <= self.config.static_speed_threshold_mps:
            if self.static_candidate_time_s is None:
                self.static_candidate_time_s = timestamp_s
            elif timestamp_s - self.static_candidate_time_s >= self.config.static_hold_s:
                self.is_static = True
                self.state[3:] = 0.0
        else:
            self.static_candidate_time_s = None

    def estimate(self, timestamp_s: float) -> tuple[np.ndarray, np.ndarray, float] | None:
        if not self.initialized:
            return None
        assert self.state_time_s is not None
        assert self.last_measurement_time_s is not None
        age_s = max(0.0, timestamp_s - self.last_measurement_time_s)
        if age_s > self.config.maximum_prediction_s:
            return None
        state, _, _ = self._predict_state(
            self.state, self.covariance, max(0.0, timestamp_s - self.state_time_s)
        )
        return state[:3], state[3:], age_s


def build_kalman_ball_stream(
    raw_ball: Samples, output_clock: Samples, calibrations: list[Calibration], config: dict
) -> tuple[Samples, PhysicsBallKalman]:
    """Replay raw measurements causally and sample EKF state at Bridge output times."""
    kalman = PhysicsBallKalman(OfflineKalmanConfig.from_deploy_config(config))
    rows = []
    raw_index = 0
    calibration_ids = calibration_indices(raw_ball.time, calibrations)
    active_calibration = calibration_ids[0] if len(calibration_ids) else -1
    for output_time, source_time in zip(output_clock.time, output_clock.source_time):
        while raw_index < len(raw_ball.time) and raw_ball.time[raw_index] <= output_time:
            calibration_id = calibration_ids[raw_index]
            if calibration_id != active_calibration:
                kalman.reset()
                active_calibration = calibration_id
            if raw_ball.valid[raw_index]:
                kalman.observe(
                    raw_ball.values[raw_index, :3], raw_ball.source_time[raw_index]
                )
            raw_index += 1
        estimate = kalman.estimate(source_time)
        if estimate is None:
            rows.append((output_time, np.full(6, np.nan), False, source_time))
        else:
            position, velocity, _ = estimate
            values = np.concatenate((position, velocity))
            rows.append((output_time, values, bool(np.isfinite(values).all()), source_time))
    return Samples.from_rows(rows, 6), kalman


def pose_values(pose) -> np.ndarray:
    p, q = pose.position, pose.orientation
    values = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(values[3:])
    if norm > 1e-8:
        values[3:] /= norm
    return values


def vector_values(vector) -> np.ndarray:
    return np.array([vector.x, vector.y, vector.z], dtype=float)


def finite_pose(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).all() and np.linalg.norm(values[3:]) > .99)


@dataclass
class Recording:
    path: Path
    pelvis: Samples
    joints: Samples
    raw_ball: Samples
    filtered_ball: Samples
    filtered_ball_velocity: Samples
    kalman_ball: Samples
    policy_ball: Samples
    observed_sweet: Samples
    fsm: Samples
    calibrations: list[Calibration]
    kalman_config: OfflineKalmanConfig
    kalman_stats: dict[str, int]
    start: float
    end: float
    first_control: float


def read_recording(path: Path, config: dict) -> Recording:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    # MCAP carries its schemas; this fallback also supports local SQLite bags.
    typestore = get_typestore(Stores.ROS2_JAZZY)
    msg_dir = Path.home() / "optitrack_companion_ws/src/mocap_msgs/msg"
    for name in ("Marker", "RigidBody", "RigidBodies"):
        source = msg_dir / f"{name}.msg"
        if source.exists():
            typestore.register(get_types_from_msg(source.read_text(), f"mocap_msgs/msg/{name}"))
    bridge = config.get("mocap_bridge", {})
    raw_topic = bridge.get("input_topic", "/rigid_bodies")
    names = (bridge.get("pelvis_name", "G1_pelvis"), bridge.get("ball_name", "Ball"))
    topics = {raw_topic, "/g1_pelvis/pose", "/ball/pose", "/ball/velocity",
              "/deploy_robot/joint_state",
              "/deploy_robot/student_observation",
              "/deploy_robot/fsm_state", "/deploy_robot/root_calibration_request"}
    raw_pelvis, raw_ball, pelvis, filtered, filtered_velocity, joints, policy_ball_b, observed_sweet_b, fsm, requests = (
        [], [], [], [], [], [], [], [], [], []
    )
    with AnyReader([path], default_typestore=typestore) as reader:
        required = topics - {"/deploy_robot/fsm_state", "/deploy_robot/root_calibration_request"}
        missing = required - {c.topic for c in reader.connections}
        if missing:
            raise ValueError(f"Bag is missing required topics: {sorted(missing)}")
        origin = reader.start_time
        connections = [c for c in reader.connections if c.topic in topics]
        for connection, timestamp, data in reader.messages(connections=connections):
            message = reader.deserialize(data, connection.msgtype)
            t = (timestamp - origin) * 1e-9
            source_t = t
            if hasattr(message, "header"):
                stamp = message.header.stamp
                source_t = (stamp.sec * 10**9 + stamp.nanosec - origin) * 1e-9
            if connection.topic == raw_topic:
                bodies = {b.rigid_body_name: b for b in message.rigidbodies}
                for name, rows in zip(names, (raw_pelvis, raw_ball)):
                    body = bodies.get(name)
                    values = pose_values(body.pose) if body is not None else np.full(7, np.nan)
                    valid = body is not None and body.tracking_valid and finite_pose(values)
                    if name == names[0] and body is not None:
                        valid = valid and np.isfinite(body.mean_error) and body.mean_error <= bridge.get("pelvis_max_mean_error_m", .005)
                    elif name == names[1] and body is not None:
                        valid = valid and np.isfinite(body.mean_error) and body.mean_error <= bridge.get("ball_max_mean_error_m", .010)
                    rows.append((t, values, valid, source_t))
            elif connection.topic in ("/g1_pelvis/pose", "/ball/pose"):
                values = pose_values(message.pose)
                rows = pelvis if connection.topic == "/g1_pelvis/pose" else filtered
                rows.append((t, values, finite_pose(values), source_t))
            elif connection.topic == "/ball/velocity":
                values = vector_values(message.vector)
                filtered_velocity.append(
                    (t, values, bool(np.isfinite(values).all()), source_t)
                )
            elif connection.topic == "/deploy_robot/joint_state":
                values = np.asarray(message.data, dtype=float)
                if len(values) < 58:
                    raise ValueError(f"joint_state needs q[29]+dq[29], got {len(values)}")
                joints.append((t, values[:29], bool(np.isfinite(values[:29]).all()), t))
            elif connection.topic == "/deploy_robot/student_observation":
                values = np.asarray(message.data, dtype=float)
                if len(values) not in INTENT_OBSERVATION_DIMS:
                    raise ValueError(
                        "student_observation needs a 628/631/638-D Intent observation "
                        f"with sweet_spot_position[128:131], got {len(values)}"
                    )
                point_b = values[SWEET_SPOT_POSITION_SLICE]
                observed_sweet_b.append((t, point_b, bool(np.isfinite(point_b).all()), t))
                ball_b = values[POLICY_BALL_POSITION_SLICE]
                policy_ball_b.append(
                    (t, ball_b, bool(np.isfinite(ball_b).all()), t)
                )
            elif connection.topic == "/deploy_robot/fsm_state":
                fsm.append((t, [dict(damp=0, home=1, control=2).get(message.data, -1)], True, t))
            elif message.data == "calibrate":
                requests.append(t)
    raw_root = Samples.from_rows(raw_pelvis, 7)
    root = Samples.from_rows(pelvis, 7)
    joint = Samples.from_rows(joints, 29)
    raw = Samples.from_rows(raw_ball, 7)
    filt = Samples.from_rows(filtered, 7)
    filt_velocity = Samples.from_rows(filtered_velocity, 3)
    policy_ball = Samples.from_rows(policy_ball_b, 3)
    observed_sweet = Samples.from_rows(observed_sweet_b, 3)
    state = Samples.from_rows(fsm, 1)
    for label, stream in (("Pelvis", root), ("joints", joint), ("raw Ball", raw),
                          ("filtered Ball", filt), ("filtered Ball velocity", filt_velocity)):
        if np.count_nonzero(stream.valid) < 2:
            raise ValueError(f"Not enough valid {label} samples in {path}")
    calibrations = recover_calibrations(raw_root, root, np.asarray(requests))
    raw = transform_raw_positions(raw, calibrations)
    kalman_ball, kalman_filter = build_kalman_ball_stream(
        raw, filt, calibrations, config
    )
    kalman_stats = {
        "accepted": kalman_filter.accepted_measurements,
        "rejected": kalman_filter.rejected_measurements,
        "reinitialized": kalman_filter.reinitializations,
    }
    observed_sweet = local_positions_to_world(
        observed_sweet, root, float(config.get("state_timeout_s", .25))
    )
    policy_ball = local_positions_to_world(
        policy_ball, root, float(config.get("state_timeout_s", .25))
    )
    start = max(root.time[root.valid][0], joint.time[joint.valid][0])
    end = min(root.time[root.valid][-1], joint.time[joint.valid][-1])
    if end <= start:
        raise ValueError("Pelvis and joint streams have no overlapping time range.")
    active = state.time[(state.values[:, 0] == 2) & (state.time >= start) & (state.time <= end)]
    return Recording(
        path,
        root,
        joint,
        raw,
        filt,
        filt_velocity,
        kalman_ball,
        policy_ball,
        observed_sweet,
        state,
        calibrations,
        kalman_filter.config,
        kalman_stats,
        float(start),
        float(end),
        float(active[0]) if len(active) else float(start),
    )


def build_model(config: dict, config_path: Path) -> mujoco.MjModel:
    simulation = config.get("simulation", {})
    court = simulation.get("court", {})
    incoming = simulation["incoming_ball"]
    model, _ = build_training_aligned_model(
        robot_xml_path=resolve_path(config["simulation_xml_path"], REPO_ROOT, DEPLOY_DIR, config_path.parent),
        court_xml_path=resolve_path(config.get("tennis_court_xml_path", "deploy/simulation/assets/tennis_court.xml"), REPO_ROOT, config_path.parent),
        joint_names=tuple(n.strip() for n in config["joint_names"].split(",")),
        physics_dt=float(simulation.get("physics_dt", .0025)),
        nominal_ball_physics=config["tennis_ball_physics"],
        randomization_config=simulation.get("physics_domain_randomization", {}),
        court_net_x_m=float(court.get("net_x_m", 5.6)),
        court_net_half_width_m=float(court.get("net_half_width_m", 5.485)),
        launch_position_min_w=incoming["position_min_w"],
        launch_position_max_w=incoming["position_max_w"],
    )
    return model


def add_sphere(scene, position, radius, color):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                       np.array([radius, radius, radius]), position, IDENTITY, color)
    scene.ngeom += 1


def add_line(scene, start, end, color, width=.005, arrow=False):
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(end - start) < 1e-7:
        return
    geom = scene.geoms[scene.ngeom]
    kind = mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE
    mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), IDENTITY, color)
    mujoco.mjv_connector(geom, kind, width, start, end)
    scene.ngeom += 1


def trail_segments(samples: Samples, timestamp: float, duration: float,
                   calibrations: list[Calibration], max_gap=.05) -> tuple[np.ndarray, np.ndarray]:
    left = np.searchsorted(samples.time, timestamp - duration)
    right = np.searchsorted(samples.time, timestamp, side="right")
    times, values, valid = samples.time[left:right], samples.values[left:right, :3], samples.valid[left:right]
    epochs = calibration_indices(times, calibrations)
    good = valid[:-1] & valid[1:] & (np.diff(times) <= max_gap) & (epochs[:-1] == epochs[1:])
    # Reassociation jumps must not draw a line across the court.
    good &= np.linalg.norm(np.diff(values, axis=0), axis=1) <= 40.0 * np.diff(times) + .05
    ids = np.flatnonzero(good)
    if len(ids) > 400:
        ids = ids[np.linspace(0, len(ids) - 1, 400, dtype=int)]
    return values[ids], values[ids + 1]


class Replay:
    def __init__(self, recording: Recording, config: dict, config_path: Path):
        self.recording, self.config = recording, config
        self.model = build_model(config, config_path)
        self.data = mujoco.MjData(self.model)
        self.root_id = self.model.body("pelvis").id
        root_joint = int(self.model.body_jntadr[self.root_id])
        if self.model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Pelvis must have a floating joint.")
        self.root_qpos = int(self.model.jnt_qposadr[root_joint])
        names = [n.strip() for n in config["joint_names"].split(",")]
        if len(names) != 29 or len(set(names)) != 29:
            raise ValueError("Replay requires 29 unique joint names in recorded hardware order.")
        self.joint_qpos = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in names])
        self.site_id = self.model.site(config.get("sweet_spot_site", "racket_sweet_spot")).id
        self.sp_offset = np.asarray(config.get("sweet_spot_position_offset_b", [0, 0, 0]), dtype=float)
        self.radius = float(config["tennis_ball_physics"]["radius_m"])
        self.model.geom_rgba[self.model.geom("tennis_ball_geom").id, 3] = 0.0
        ball_qpos = int(self.model.jnt_qposadr[self.model.joint("tennis_ball_freejoint").id])
        self.data.qpos[ball_qpos:ball_qpos + 3] = (0, 0, -10)
        self.data.qpos[ball_qpos + 3:ball_qpos + 7] = (1, 0, 0, 0)
        self.show_raw = self.show_filtered = self.show_kalman = True
        self.show_policy_ball = True
        self.show_observed_sweet = True
        self.show_trails = self.show_mocap_frame = True
        self.velocity_arrow_scale_s = float(
            config.get("mocap_bridge", {}).get("velocity_arrow_scale_s", .10)
        )

    def set_time(self, timestamp: float) -> dict:
        root, root_age = self.recording.pelvis.current(timestamp, float("inf"), hold_valid=True)
        joints, joint_age = self.recording.joints.current(timestamp, float("inf"), hold_valid=True)
        if root is None or joints is None:
            raise ValueError("No robot state at this time; select the recorded overlap range.")
        self.data.qpos[self.root_qpos:self.root_qpos + 3] = root[:3]
        self.data.qpos[self.root_qpos + 3:self.root_qpos + 7] = root[[6, 3, 4, 5]]
        self.data.qpos[self.joint_qpos] = joints
        self.data.time = timestamp
        mujoco.mj_forward(self.model, self.data)
        raw, raw_age = self.recording.raw_ball.current(timestamp, .05)
        filt, filtered_age = self.recording.filtered_ball.current(timestamp, .10)
        filtered_velocity, filtered_velocity_age = (
            self.recording.filtered_ball_velocity.current(timestamp, .10)
        )
        kalman, kalman_age = self.recording.kalman_ball.current(timestamp, .10)
        policy_ball, policy_ball_age = self.recording.policy_ball.current(
            timestamp, .10, hold_valid=True
        )
        observed_sweet, observed_sweet_age = self.recording.observed_sweet.current(
            timestamp, .10, hold_valid=True
        )
        sp = self.data.site_xpos[self.site_id] + self.data.xmat[self.root_id].reshape(3, 3) @ self.sp_offset
        mocap_rotation = Rotation.from_quat(root[3:]).as_matrix()
        state, _ = self.recording.fsm.current(timestamp, float("inf"))
        return {"raw": raw, "filtered": filt, "filtered_velocity": filtered_velocity,
                "kalman": kalman, "sweet": sp.copy(),
                "policy_ball": policy_ball,
                "observed_sweet": observed_sweet,
                "mocap_root": root.copy(), "mocap_rotation": mocap_rotation,
                "root_age": root_age, "joint_age": joint_age,
                "raw_age": raw_age, "filtered_age": filtered_age,
                "filtered_velocity_age": filtered_velocity_age,
                "kalman_age": kalman_age,
                "policy_ball_age": policy_ball_age,
                "observed_sweet_age": observed_sweet_age,
                "fsm": {-1: "unknown", 0: "damp", 1: "home", 2: "control"}.get(int(state[0]), "unknown") if state is not None else "unknown"}

    def draw(self, scene, timestamp, values, trail_s):
        for show, stream, key, color in (
            (self.show_raw, self.recording.raw_ball, "raw", RAW_COLOR),
            (self.show_filtered, self.recording.filtered_ball, "filtered", FILTERED_COLOR),
            (self.show_kalman, self.recording.kalman_ball, "kalman", KALMAN_COLOR),
        ):
            if not show:
                continue
            if self.show_trails:
                a, b = trail_segments(stream, timestamp, trail_s, self.recording.calibrations)
                for start, end in zip(a, b):
                    add_line(scene, start, end, color)
            point = values[key]
            if point is not None:
                add_sphere(scene, point[:3], self.radius, color)
        if (
            self.show_filtered
            and values["filtered"] is not None
            and values["filtered_velocity"] is not None
        ):
            start = values["filtered"][:3]
            end = start + self.velocity_arrow_scale_s * values["filtered_velocity"]
            add_line(scene, start, end, FILTERED_COLOR, .008, True)
        if self.show_kalman and values["kalman"] is not None:
            start = values["kalman"][:3]
            end = start + self.velocity_arrow_scale_s * values["kalman"][3:]
            add_line(scene, start, end, KALMAN_COLOR, .008, True)
        if self.show_policy_ball and values["policy_ball"] is not None:
            add_sphere(
                scene,
                values["policy_ball"],
                1.05 * self.radius,
                POLICY_BALL_COLOR,
            )
        add_sphere(scene, values["sweet"], .012, FK_SWEET_COLOR)
        observed_sweet = values["observed_sweet"]
        if self.show_observed_sweet and observed_sweet is not None:
            if self.show_trails:
                a, b = trail_segments(
                    self.recording.observed_sweet,
                    timestamp,
                    trail_s,
                    self.recording.calibrations,
                )
                for start, end in zip(a, b):
                    add_line(scene, start, end, OBSERVED_SWEET_COLOR)
            add_sphere(scene, observed_sweet, .018, OBSERVED_SWEET_COLOR)
            add_line(scene, values["sweet"], observed_sweet, SWEET_ERROR_COLOR, .003)
        if self.show_mocap_frame:
            root, rotation = values["mocap_root"][:3], values["mocap_rotation"]
            for axis, color in enumerate(MOCAP_FRAME_COLORS):
                add_line(scene, root, root + .45 * rotation[:, axis], color, .006, True)

    def camera(self):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.data.xpos[self.root_id] + [0, 0, .1]
        camera.distance, camera.azimuth, camera.elevation = 4.5, 135., -20.
        return camera

    def snapshot(self, path: Path, timestamp: float, trail_s: float):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("--snapshot needs Pillow: uv pip install pillow") from exc
        values = self.set_time(timestamp)
        width = min(1280, self.model.vis.global_.offwidth)
        height = min(720, self.model.vis.global_.offheight)
        with mujoco.Renderer(self.model, width=width, height=height) as renderer:
            renderer.update_scene(self.data, camera=self.camera())
            self.draw(renderer.scene, timestamp, values, trail_s)
            pixels = renderer.render().copy()
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path)
        print(f"Snapshot: {path}")


def report(recording: Recording):
    print(f"Bag: {recording.path}")
    print(f"Clock: bag receive time; robot overlap {recording.start:.3f}..{recording.end:.3f}s; "
          f"first control {recording.first_control:.3f}s")
    for name, samples in (("Pelvis", recording.pelvis), ("Joint q[29]", recording.joints),
                          ("Raw Ball", recording.raw_ball), ("Filtered Ball", recording.filtered_ball),
                          ("Filtered Ball velocity", recording.filtered_ball_velocity),
                          ("Kalman Ball [position, velocity]", recording.kalman_ball),
                          ("Policy-input Ball", recording.policy_ball),
                          ("Observed sweet point", recording.observed_sweet)):
        if not len(samples.time):
            print(f"{name}: not recorded")
            continue
        print(f"{name}: {len(samples.time)} samples, valid={samples.valid.sum()}, "
              f"t={samples.time[0]:.3f}..{samples.time[-1]:.3f}s")
    for calibration in recording.calibrations:
        yaw = np.rad2deg(np.arctan2(calibration.rotation[1, 0], calibration.rotation[0, 0]))
        print(f"Calibration from {calibration.start:.3f}s: yaw={yaw:+.5f}deg "
              f"translation={calibration.translation.round(6).tolist()}, "
              f"{calibration.pairs} pairs; Pelvis P95={calibration.position_p95:.6f}m/"
              f"{calibration.orientation_p95_deg:.6f}deg")
    cfg = recording.kalman_config
    stats = recording.kalman_stats
    print(
        "Replay Kalman: "
        f"measurement_sigma={cfg.measurement_noise_m * 1000:.1f}mm, "
        f"process_accel_sigma={cfg.process_acceleration_noise_mps2:.1f}m/s^2, "
        f"NIS_gate={cfg.innovation_gate_chi2:.2f}; "
        f"accepted={stats['accepted']}, rejected={stats['rejected']}, "
        f"reinitialized={stats['reinitialized']}"
    )
    print("Orange: raw Ball transformed into calibrated world; blue: recorded /ball/pose; "
          "yellow: replay-only physics Kalman; red: recorded policy-input Ball; "
          "green: replay FK sweet point; "
          "magenta: recorded sweet-point position observation.")


def run_viewer(replay: Replay, args, start: float):
    import mujoco.viewer
    commands = SimpleQueue()
    state = {"t": start, "paused": args.paused, "speed": args.speed, "quit": False, "dragging": False}
    root = None
    ui_status = None
    slider = None
    if not args.no_ui:
        import tkinter as tk
        from tkinter import ttk
        root = tk.Tk()
        root.title("G1 recorded deployment replay")
        root.option_add("*Font", "TkDefaultFont 13")
        root.geometry("1240x230")
        def finish_seek(event):
            state.update(t=float(slider.get()), dragging=False)
        slider = tk.Scale(root, from_=replay.recording.start, to=replay.recording.end,
                          resolution=.001, orient=tk.HORIZONTAL, takefocus=False)
        slider.bind("<ButtonPress-1>", lambda e: state.update(dragging=True, paused=True))
        slider.bind("<ButtonRelease-1>", finish_seek)
        slider.set(start)
        slider.pack(fill=tk.X, padx=14, pady=8)
        row = ttk.Frame(root)
        row.pack()
        for label, command in (("-1s", "back"), ("-20ms", "prev"), ("Play / Pause", "pause"),
                               ("+20ms", "next"), ("+1s", "forward"), ("First control", "control")):
            ttk.Button(row, text=label, command=lambda c=command: commands.put(c)).pack(side=tk.LEFT, padx=3)
        speed = tk.StringVar(value=str(args.speed))
        box = ttk.Combobox(row, textvariable=speed, values=(.1, .25, .5, 1., 2.), width=5, state="readonly")
        box.pack(side=tk.LEFT, padx=6)
        box.bind("<<ComboboxSelected>>", lambda e: state.update(speed=float(speed.get())))
        toggles = ttk.Frame(root)
        toggles.pack(pady=8)
        for label, attr in (("Raw (orange)", "show_raw"), ("Bridge (blue)", "show_filtered"),
                            ("Kalman (yellow)", "show_kalman"),
                            ("Policy input (red)", "show_policy_ball"),
                            ("Observed sweet (magenta)", "show_observed_sweet"),
                            ("Trails", "show_trails"),
                            ("Mocap pelvis frame (RGB)", "show_mocap_frame")):
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(toggles, text=label, variable=var,
                            command=lambda v=var, a=attr: setattr(replay, a, v.get())).pack(side=tk.LEFT, padx=6)
        ui_status = tk.StringVar()
        ttk.Label(root, textvariable=ui_status).pack()
        root.protocol("WM_DELETE_WINDOW", lambda: state.update(quit=True))
    keys = {32: "pause", 263: "prev", 262: "next", ord("R"): "control", ord("Q"): "quit"}
    def callback(key):
        if key in keys:
            commands.put(keys[key])
    replay.set_time(start)
    existing_threads = set(threading.enumerate())
    viewer_threads = []
    try:
        with mujoco.viewer.launch_passive(replay.model, replay.data, key_callback=callback,
                                          show_left_ui=False, show_right_ui=False) as viewer:
            viewer_threads = [thread for thread in threading.enumerate() if thread not in existing_threads]
            cam = replay.camera()
            viewer.cam.lookat[:] = cam.lookat
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam.distance, cam.azimuth, cam.elevation
            last = time.monotonic()
            while viewer.is_running() and not state["quit"]:
                now = time.monotonic()
                if not state["paused"]:
                    state["t"] += (now - last) * state["speed"]
                last = now
                if root is not None:
                    root.update()
                    if state["dragging"]:
                        state["t"] = float(slider.get())
                while True:
                    try:
                        command = commands.get_nowait()
                    except Empty:
                        break
                    if command == "pause":
                        state["paused"] = not state["paused"]
                    elif command == "quit":
                        state["quit"] = True
                    elif command == "control":
                        state["t"] = replay.recording.first_control
                    else:
                        state["paused"] = True
                        state["t"] += dict(back=-1., prev=-.02, next=.02, forward=1.)[command]
                if state["t"] > replay.recording.end:
                    state["t"] = replay.recording.start if args.loop else replay.recording.end
                    if not args.loop:
                        state["paused"] = True
                state["t"] = max(state["t"], replay.recording.start)
                t = state["t"]
                with viewer.lock():
                    values = replay.set_time(t)
                    viewer.user_scn.ngeom = 0
                    replay.draw(viewer.user_scn, t, values, args.trail_seconds)
                status = f"t={t:.3f}s / {replay.recording.end:.3f}s  {state['speed']:g}x  {values['fsm']}"
                if state["paused"]:
                    status += "  PAUSED"
                ages = f"Pelvis age={values['root_age']*1000:.0f}ms   joints age={values['joint_age']*1000:.0f}ms"
                sweet_error = None
                if values["observed_sweet"] is not None:
                    sweet_error = np.linalg.norm(values["observed_sweet"] - values["sweet"])
                    ages += f"   sweet obs-FK={sweet_error*1000:.1f}mm"
                if values["root_age"] > .04 or values["joint_age"] > .05:
                    ages += "  STALE: holding recorded robot pose"
                ball_status = []
                if values["filtered_velocity"] is not None:
                    ball_status.append(
                        f"Bridge |v|={np.linalg.norm(values['filtered_velocity']):.2f}m/s"
                    )
                if values["kalman"] is not None:
                    ball_status.append(
                        f"Kalman |v|={np.linalg.norm(values['kalman'][3:]):.2f}m/s"
                    )
                if values["filtered"] is not None and values["kalman"] is not None:
                    ball_status.append(
                        "K-B position="
                        f"{np.linalg.norm(values['kalman'][:3] - values['filtered'][:3]) * 1000:.1f}mm"
                    )
                if values["filtered_velocity"] is not None and values["kalman"] is not None:
                    ball_status.append(
                        "velocity="
                        f"{np.linalg.norm(values['kalman'][3:] - values['filtered_velocity']):.2f}m/s"
                    )
                if values["raw"] is not None:
                    raw_errors = []
                    if values["filtered"] is not None:
                        raw_errors.append(
                            f"Bridge={np.linalg.norm(values['filtered'][:3] - values['raw'][:3]) * 1000:.1f}mm"
                        )
                    if values["kalman"] is not None:
                        raw_errors.append(
                            f"Kalman={np.linalg.norm(values['kalman'][:3] - values['raw'][:3]) * 1000:.1f}mm"
                        )
                    if raw_errors:
                        ball_status.append("position vs raw " + "/".join(raw_errors))
                ball_line = "   ".join(ball_status) if ball_status else "Ball state unavailable"
                policy_line = "Policy ball unavailable"
                if values["policy_ball"] is not None:
                    policy_line = (
                        "Policy ball W="
                        f"{np.array2string(values['policy_ball'], precision=3)}"
                    )
                    if values["filtered"] is not None:
                        policy_line += (
                            "   Policy-Bridge="
                            f"{np.linalg.norm(values['policy_ball'] - values['filtered'][:3]) * 1000:.1f}mm"
                        )
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  status + "\n" + ages + "\n" + ball_line
                                  + "\n" + policy_line
                                  + "\nSpace pause | Arrows step | R first control | Q quit",
                                  "Orange: RAW ball\nBlue: recorded Bridge ball + velocity\n"
                                  "Yellow: replay Kalman ball + velocity\n"
                                  "Red: recorded policy-input ball\n"
                                  "Green: replay FK sweet\nMagenta: RECORDED sweet obs\n"
                                  "Mocap pelvis: RGB 0.45m"))
                if root is not None:
                    if not state["dragging"]:
                        slider.set(t)
                    ui_status.set(status)
                viewer.sync()
                time.sleep(max(0., 1 / 60 - (time.monotonic() - now)))
    finally:
        # Handle.close() only requests exit. Finish renderer teardown before
        # interpreter shutdown calls glfw.terminate on its daemon thread.
        for thread in viewer_threads:
            thread.join(timeout=5.)
        if root is not None:
            root.destroy()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", nargs="?", type=Path, help="Bag directory or .mcap; default newest named recording.")
    parser.add_argument("--config", default="g1_tppo_student_m14_9.yaml")
    parser.add_argument("--start", type=float, help="Seconds since bag start; default first control.")
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--trail-seconds", type=float, default=1.)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-ui", action="store_true", help="Use only the native viewer and keyboard.")
    parser.add_argument("--inspect", action="store_true", help="Read-only frame/data checks without opening a viewer.")
    parser.add_argument("--snapshot", type=Path, help="Render one frame to PNG without opening a viewer.")
    args = parser.parse_args(argv)
    if not np.isfinite(args.speed) or args.speed <= 0 or not np.isfinite(args.trail_seconds) or args.trail_seconds < 0:
        parser.error("speed must be positive and trail-seconds nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    config, config_path = load_config(args.config)
    if args.bag is None:
        paths = sorted(p.parent for p in (DEPLOY_DIR / "recordings").glob("*/metadata.yaml"))
        if not paths:
            raise ValueError("No recordings found; pass a bag path.")
        path = paths[-1]
    else:
        path = resolve_path(args.bag, DEPLOY_DIR, REPO_ROOT)
    recording = read_recording(path, config)
    report(recording)
    if args.inspect:
        return
    start = recording.first_control if args.start is None else args.start
    if not np.isfinite(start) or not recording.start <= start <= recording.end:
        raise ValueError(f"--start must be in [{recording.start:.3f}, {recording.end:.3f}].")
    replay = Replay(recording, config, config_path)
    if args.snapshot:
        replay.snapshot(args.snapshot.expanduser().resolve(), start, args.trail_seconds)
    else:
        run_viewer(replay, args, start)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (ValueError, FileNotFoundError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc
