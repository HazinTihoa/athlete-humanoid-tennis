"""State estimation primitives for the NatNet-to-policy ROS bridge."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math

import numpy as np


GRAVITY_MPS2 = 9.81


def _vec3(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return result.copy()


@dataclass(frozen=True)
class BallEstimatorConfig:
    radius_m: float = 0.0335
    mass_kg: float = 0.0577
    air_density_kg_m3: float = 1.225
    drag_coefficient: float = 0.55
    court_restitution: float = 0.745
    tangent_speed_retention: float = 0.825
    ground_height_m: float = 0.0
    velocity_window_size: int = 7
    velocity_min_samples: int = 5
    short_dropout_s: float = 0.10
    maximum_prediction_s: float = 2.0
    reassociation_timeout_s: float = 0.50
    static_speed_threshold_mps: float = 0.15
    static_hold_s: float = 0.15
    maximum_speed_mps: float = 40.0
    position_correction_gain: float = 0.80
    velocity_correction_gain: float = 0.55
    integration_dt_s: float = 0.0025

    def validate(self) -> None:
        if self.radius_m <= 0.0 or self.mass_kg <= 0.0:
            raise ValueError("Ball radius and mass must be positive")
        if self.air_density_kg_m3 <= 0.0 or self.drag_coefficient < 0.0:
            raise ValueError("Invalid aerodynamic parameters")
        if not 0.0 <= self.court_restitution <= 1.0:
            raise ValueError("court_restitution must be in [0, 1]")
        if not 0.0 <= self.tangent_speed_retention <= 1.0:
            raise ValueError("tangent_speed_retention must be in [0, 1]")
        if self.velocity_min_samples < 2:
            raise ValueError("velocity_min_samples must be at least 2")
        if self.velocity_window_size < self.velocity_min_samples:
            raise ValueError("velocity_window_size is too small")
        if self.short_dropout_s <= 0.0 or self.maximum_prediction_s <= 0.0:
            raise ValueError("Dropout and prediction windows must be positive")
        if self.maximum_prediction_s < self.short_dropout_s:
            raise ValueError("maximum_prediction_s must cover short_dropout_s")
        if self.reassociation_timeout_s <= 0.0:
            raise ValueError("reassociation_timeout_s must be positive")
        if self.static_speed_threshold_mps < 0.0 or self.static_hold_s < 0.0:
            raise ValueError("Invalid static-ball thresholds")
        if self.maximum_speed_mps <= 0.0:
            raise ValueError("maximum_speed_mps must be positive")
        if not 0.0 <= self.position_correction_gain <= 1.0:
            raise ValueError("position_correction_gain must be in [0, 1]")
        if not 0.0 <= self.velocity_correction_gain <= 1.0:
            raise ValueError("velocity_correction_gain must be in [0, 1]")
        if self.integration_dt_s <= 0.0:
            raise ValueError("integration_dt_s must be positive")


def propagate_ball_state(
    position_w: np.ndarray,
    velocity_w: np.ndarray,
    duration_s: float,
    config: BallEstimatorConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Propagate a tennis ball through drag, gravity, and hard-court bounces."""
    position = _vec3(position_w, "position_w")
    velocity = _vec3(velocity_w, "velocity_w")
    if not math.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError("duration_s must be finite and non-negative")

    ground_center_z = config.ground_height_m + config.radius_m
    position[2] = max(position[2], ground_center_z)
    area = math.pi * config.radius_m**2
    drag_factor = (
        0.5
        * config.air_density_kg_m3
        * config.drag_coefficient
        * area
        / config.mass_kg
    )
    remaining = duration_s
    bounce_count = 0

    while remaining > 1.0e-12:
        dt = min(config.integration_dt_s, remaining)
        speed = float(np.linalg.norm(velocity))
        acceleration = -drag_factor * speed * velocity
        acceleration[2] -= GRAVITY_MPS2

        next_velocity = velocity + acceleration * dt
        next_position = position + 0.5 * (velocity + next_velocity) * dt

        if next_position[2] < ground_center_z and next_velocity[2] < 0.0:
            next_position[2] = ground_center_z
            next_velocity[:2] *= config.tangent_speed_retention
            next_velocity[2] = -config.court_restitution * next_velocity[2]
            bounce_count += 1

        position = next_position
        velocity = next_velocity
        remaining -= dt

    return position, velocity, bounce_count


class BallStateEstimator:
    """Track one ball and bridge short rigid-body dropouts with physics."""

    def __init__(self, config: BallEstimatorConfig) -> None:
        config.validate()
        self.config = config
        self.samples: deque[tuple[float, np.ndarray]] = deque(
            maxlen=config.velocity_window_size
        )
        self.position_w = np.zeros(3, dtype=np.float64)
        self.velocity_w = np.zeros(3, dtype=np.float64)
        self.state_time_s: float | None = None
        self.last_valid_time_s: float | None = None
        self.track_epoch = 0
        self.is_static = True
        self._static_candidate_since_s: float | None = None

    @property
    def initialized(self) -> bool:
        return self.state_time_s is not None

    def _reset_track(self, position_w: np.ndarray, timestamp_s: float) -> None:
        self.position_w[:] = position_w
        self.velocity_w.fill(0.0)
        self.state_time_s = timestamp_s
        self.last_valid_time_s = timestamp_s
        self.samples.clear()
        self.samples.append((timestamp_s, position_w.copy()))
        self.track_epoch += 1
        self.is_static = True
        self._static_candidate_since_s = timestamp_s

    def _velocity_fit(self) -> np.ndarray | None:
        if len(self.samples) < self.config.velocity_min_samples:
            return None
        times = np.asarray([sample[0] for sample in self.samples])
        positions = np.asarray([sample[1] for sample in self.samples])
        times = times - times[-1]
        if float(times[-1] - times[0]) <= 1.0e-6:
            return None
        design = np.column_stack((np.ones_like(times), times, times**2))
        coefficients, _, _, _ = np.linalg.lstsq(design, positions, rcond=None)
        velocity = coefficients[1]
        if not np.isfinite(velocity).all():
            return None
        if float(np.linalg.norm(velocity)) > self.config.maximum_speed_mps:
            return None
        return velocity

    def _update_static_state(self, timestamp_s: float) -> None:
        speed = float(np.linalg.norm(self.velocity_w))
        if speed > self.config.static_speed_threshold_mps:
            self.is_static = False
            self._static_candidate_since_s = None
            return
        if self._static_candidate_since_s is None:
            self._static_candidate_since_s = timestamp_s
        if timestamp_s - self._static_candidate_since_s >= self.config.static_hold_s:
            self.is_static = True
            self.velocity_w.fill(0.0)

    def observe(self, position_w: np.ndarray, timestamp_s: float) -> bool:
        """Update from one valid measurement; return True for a new track epoch."""
        position = _vec3(position_w, "position_w")
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if not self.initialized:
            self._reset_track(position, timestamp_s)
            return True
        assert self.state_time_s is not None
        assert self.last_valid_time_s is not None
        if timestamp_s <= self.last_valid_time_s:
            return False

        gap_s = timestamp_s - self.last_valid_time_s
        # A long loss starts a new track; do not integrate the entire gap first.
        if gap_s > self.config.reassociation_timeout_s:
            self._reset_track(position, timestamp_s)
            return True
        if self.is_static:
            predicted_position = self.position_w.copy()
            predicted_velocity = np.zeros(3, dtype=np.float64)
            bounce_count = 0
        else:
            predicted_position, predicted_velocity, bounce_count = propagate_ball_state(
                self.position_w,
                self.velocity_w,
                timestamp_s - self.state_time_s,
                self.config,
            )
        innovation = position - predicted_position
        # A bounce often invalidates the rigid body for more than the short-dropout
        # display window. Keep the same physical track when the recovered point is
        # still consistent with the bounded physics prediction.
        # NatNet keeps the same rigid-body identity. Spatial innovation must not
        # create a new epoch: tethered tests, impacts, and model mismatch can all
        # deviate substantially from the free-flight predictor. Only a prolonged
        # loss starts a new physical track.
        if gap_s > self.config.short_dropout_s or bounce_count > 0:
            # Do not fit velocity across an occlusion or an impact discontinuity.
            self.samples.clear()
        self.samples.append((timestamp_s, position.copy()))
        self.position_w[:] = (
            predicted_position
            + self.config.position_correction_gain * innovation
        )
        velocity_fit = self._velocity_fit()
        if velocity_fit is None and len(self.samples) >= 2:
            previous_time, previous_position = self.samples[-2]
            dt = timestamp_s - previous_time
            if dt > 1.0e-6:
                rough_velocity = (position - previous_position) / dt
                if (
                    np.isfinite(rough_velocity).all()
                    and float(np.linalg.norm(rough_velocity))
                    <= self.config.maximum_speed_mps
                ):
                    velocity_fit = rough_velocity
        if velocity_fit is not None:
            self.velocity_w[:] = (
                (1.0 - self.config.velocity_correction_gain)
                * predicted_velocity
                + self.config.velocity_correction_gain * velocity_fit
            )
        else:
            self.velocity_w[:] = predicted_velocity

        self.state_time_s = timestamp_s
        self.last_valid_time_s = timestamp_s
        self._update_static_state(timestamp_s)
        return False

    def estimate(self, timestamp_s: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Return a state at timestamp without mutating the measurement filter."""
        if not self.initialized:
            return None
        assert self.state_time_s is not None
        assert self.last_valid_time_s is not None
        if self.is_static:
            return self.position_w.copy(), np.zeros(3, dtype=np.float64)

        requested_dt = max(0.0, timestamp_s - self.state_time_s)
        prediction_dt = min(requested_dt, self.config.maximum_prediction_s)
        position, velocity, _ = propagate_ball_state(
            self.position_w, self.velocity_w, prediction_dt, self.config
        )
        if timestamp_s - self.last_valid_time_s > self.config.maximum_prediction_s:
            velocity.fill(0.0)
        return position, velocity

    def mode(self, timestamp_s: float) -> str:
        if not self.initialized:
            return "uninitialized"
        assert self.last_valid_time_s is not None
        age_s = max(0.0, timestamp_s - self.last_valid_time_s)
        if self.is_static:
            return "static" if age_s <= self.config.short_dropout_s else "static_hold"
        if age_s <= 1.0 / 120.0 * 1.5:
            return "tracked_moving"
        if age_s <= self.config.short_dropout_s:
            return "predicting_short"
        if age_s <= self.config.maximum_prediction_s:
            return "predicting"
        return "absent_hold"


@dataclass(frozen=True)
class BallOutputSample:
    position_w: np.ndarray
    velocity_w: np.ndarray
    timestamp_s: float
    track_epoch: int
    source: str


class MeasurementFirstBallStream:
    """Select immediate measurements or timer fills, independent of ROS.

    Each fill tick consumes the flag set by accepted measurements since the
    previous tick. No new measurement means predict now; there is no additional
    12.5/20 ms dropout gate. Synthetic samples never feed observe().
    """

    def __init__(self, config: BallEstimatorConfig) -> None:
        self.estimator = BallStateEstimator(replace(config, position_correction_gain=1.0))
        self.last_sample: BallOutputSample | None = None
        self._updated_since_tick = False
        self._last_tick_s: float | None = None
        self._held_estimate: tuple[np.ndarray, np.ndarray] | None = None

    def observe(self, position_w: np.ndarray, timestamp_s: float) -> BallOutputSample | None:
        position = np.asarray(position_w, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all() or not math.isfinite(timestamp_s):
            return None
        previous = self.estimator.last_valid_time_s
        if previous is not None and timestamp_s <= previous:
            return None
        self.estimator.observe(position, timestamp_s)
        self._updated_since_tick = True
        self._held_estimate = None
        self.last_sample = BallOutputSample(
            position.copy(), self.estimator.velocity_w.copy(), timestamp_s,
            self.estimator.track_epoch, "measured",
        )
        return self.last_sample

    def prediction_tick(self, timestamp_s: float) -> BallOutputSample | None:
        if not math.isfinite(timestamp_s):
            return None
        if self._last_tick_s is not None and timestamp_s <= self._last_tick_s:
            return None
        self._last_tick_s = timestamp_s
        if self._updated_since_tick:
            self._updated_since_tick = False
            return None
        if not self.estimator.initialized:
            return None
        assert self.estimator.last_valid_time_s is not None
        if timestamp_s <= self.estimator.last_valid_time_s:
            return None
        expired = timestamp_s - self.estimator.last_valid_time_s > self.estimator.config.maximum_prediction_s
        if expired and self._held_estimate is not None:
            position, velocity = self._held_estimate
        else:
            estimate = self.estimator.estimate(timestamp_s)
            assert estimate is not None
            position, velocity = estimate
            if expired:
                self._held_estimate = (position.copy(), velocity.copy())
        self.last_sample = BallOutputSample(
            position.copy(), velocity.copy(), timestamp_s, self.estimator.track_epoch,
            "held" if expired or self.estimator.is_static else "predicted",
        )
        return self.last_sample
