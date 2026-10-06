"""Target-aware smooth trajectory generator for motion-matching queries.

This is intentionally standalone. It does not drive MuJoCo or the tennis
motion-matching loop directly.

The generator samples boundary conditions and creates a smooth SE(2)
trajectory:
  - start position defaults to the origin
  - start facing defaults to world +X (yaw=0)
  - start velocity is sampled
  - target distance is sampled in [0.5, 10.0] meters
  - target yaw is sampled
  - target speed is sampled, with explicit probability of exactly zero speed
  - the optimized model uses body-frame forward/lateral/yaw-rate controls, so
    forward running, backward steps and lateral shuffles are limited separately

For MM use, `sample_query_features` returns future local positions and heading
vectors at the requested frame offsets, matching the usual trajectory feature
shape: positions [n_offsets, 2], headings [n_offsets, 2].
"""

from __future__ import annotations

import argparse
import ctypes
import os
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import yaml
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


DEFAULT_OFFSETS = (20, 40, 60)
DEFAULT_CONFIG = Path(__file__).with_name("target_trajectory_config.yaml")
CONSOLE = Console(highlight=False)


def _log(message: str, style: str | None = None) -> None:
    CONSOLE.print(message, style=style, markup=False, soft_wrap=True)


def wrap_angle(x: float | np.ndarray) -> float | np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_to_near(angle: float, reference: float) -> float:
    return float(reference + wrap_angle(angle - reference))


def smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def quintic_coefficients(
    p0: np.ndarray,
    v0: np.ndarray,
    a0: np.ndarray,
    p1: np.ndarray,
    v1: np.ndarray,
    a1: np.ndarray,
    duration: float,
) -> np.ndarray:
    """Return coefficients c[0:6] for p(t)=c0+c1*t+...+c5*t^5.

    Vectorized over the trailing shape of p0. For 2D positions, the result is
    shaped [6, 2]. For scalar yaw, it is shaped [6].
    """
    T = float(duration)
    if T <= 1e-6:
        raise ValueError("duration must be positive")

    p0 = np.asarray(p0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    a0 = np.asarray(a0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)

    c0 = p0
    c1 = v0
    c2 = 0.5 * a0
    dp = p1 - (c0 + c1 * T + c2 * T * T)
    dv = v1 - (c1 + 2.0 * c2 * T)
    da = a1 - (2.0 * c2)

    A = np.array(
        [
            [T**3, T**4, T**5],
            [3.0 * T**2, 4.0 * T**3, 5.0 * T**4],
            [6.0 * T, 12.0 * T**2, 20.0 * T**3],
        ],
        dtype=np.float64,
    )
    rhs = np.stack([dp, dv, da], axis=0)
    c3_c5 = np.linalg.solve(A, rhs.reshape(3, -1)).reshape((3,) + p0.shape)
    return np.concatenate([np.stack([c0, c1, c2], axis=0), c3_c5], axis=0)


def eval_quintic(coeff: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate position, velocity, acceleration at times t."""
    coeff = np.asarray(coeff, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    powers = np.stack([t**i for i in range(6)], axis=-1)
    pos = np.tensordot(powers, coeff, axes=([-1], [0]))

    dcoeff = np.stack([coeff[i] * i for i in range(1, 6)], axis=0)
    dpowers = np.stack([t**i for i in range(5)], axis=-1)
    vel = np.tensordot(dpowers, dcoeff, axes=([-1], [0]))

    ddcoeff = np.stack([coeff[i] * i * (i - 1) for i in range(2, 6)], axis=0)
    ddpowers = np.stack([t**i for i in range(4)], axis=-1)
    acc = np.tensordot(ddpowers, ddcoeff, axes=([-1], [0]))
    return pos, vel, acc


@dataclass(frozen=True)
class BoundaryCondition:
    p0: np.ndarray
    yaw0: float
    v0: np.ndarray
    yaw_rate0: float
    p1: np.ndarray
    yaw1: float
    v1: np.ndarray
    yaw_rate1: float
    duration: float
    lateral_offset: float = 0.0


@dataclass(frozen=True)
class Trajectory:
    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    yaw: np.ndarray
    yaw_rate: np.ndarray
    yaw_acc: np.ndarray
    boundary: BoundaryCondition

    @property
    def heading(self) -> np.ndarray:
        return np.stack([np.cos(self.yaw), np.sin(self.yaw)], axis=-1)


@dataclass(frozen=True)
class SamplingConfig:
    fps: float = 50.0
    trajectory_model: str = "optimized_omni"
    distance_range: tuple[float, float] = (0.5, 10.0)
    target_bearing_range: tuple[float, float] = (-np.pi, np.pi)
    duration_range: tuple[float, float] = (0.8, 4.0)
    start_speed_range: tuple[float, float] = (0.0, 0.0)
    start_vel_yaw_range: tuple[float, float] = (-np.pi / 3.0, np.pi / 3.0)
    target_yaw_range: tuple[float, float] = (0.0, 0.0)
    target_speed_range: tuple[float, float] = (0.0, 1.5)
    target_zero_speed_prob: float = 0.35
    max_speed: float = 5.8
    max_accel: float = 9.0
    max_yaw_rate: float = 5.0
    lateral_offset_range: tuple[float, float] = (0.0, 0.0)
    max_lateral_ratio: float = 0.10
    max_tries: int = 32
    opt_knots: int = 12
    opt_max_iter: int = 80
    opt_ftol: float = 1e-5
    opt_terminal_tol: float = 0.08
    opt_minimize_duration: bool = True
    opt_duration_search_steps: int = 3
    opt_forward_speed: float = 5.5
    opt_backward_speed: float = 2.3
    opt_lateral_speed: float = 2.6
    opt_forward_accel: float = 6.5
    opt_forward_brake: float = 8.0
    opt_backward_accel: float = 3.2
    opt_backward_brake: float = 5.0
    opt_lateral_accel: float = 4.0
    opt_yaw_accel: float = 9.0
    opt_w_speed: float = 0.02
    opt_w_accel: float = 0.35
    opt_w_jerk: float = 0.04
    opt_w_lateral: float = 0.0
    opt_w_backward: float = 0.0
    opt_w_yaw_rate: float = 5.0
    opt_w_yaw_deviation: float = 8.0
    opt_w_path: float = 2.0
    opt_w_reverse_progress: float = 0.25


def _sample_uniform(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def sample_boundary(rng: np.random.Generator, cfg: SamplingConfig) -> BoundaryCondition:
    """Sample one boundary condition.

    Start pose is fixed at origin facing +X. Start velocity is sampled.
    Endpoint position is sampled by distance and bearing. Endpoint yaw and speed
    are sampled independently; endpoint velocity direction follows endpoint yaw.
    """
    p0 = np.zeros(2, dtype=np.float64)
    yaw0 = 0.0

    start_speed = _sample_uniform(rng, cfg.start_speed_range)
    start_vel_yaw = _sample_uniform(rng, cfg.start_vel_yaw_range)
    v0 = start_speed * np.array([np.cos(start_vel_yaw), np.sin(start_vel_yaw)], dtype=np.float64)

    dist = _sample_uniform(rng, cfg.distance_range)
    bearing = _sample_uniform(rng, cfg.target_bearing_range)
    p1 = dist * np.array([np.cos(bearing), np.sin(bearing)], dtype=np.float64)
    lat_bound = abs(float(cfg.max_lateral_ratio)) * dist
    lat_lo, lat_hi = cfg.lateral_offset_range
    lat_lo = max(float(lat_lo), -lat_bound)
    lat_hi = min(float(lat_hi), lat_bound)
    lateral_offset = _sample_uniform(rng, (lat_lo, lat_hi)) if lat_lo <= lat_hi else 0.0

    yaw1 = _sample_uniform(rng, cfg.target_yaw_range)
    if rng.random() < cfg.target_zero_speed_prob:
        target_speed = 0.0
    else:
        target_speed = _sample_uniform(rng, cfg.target_speed_range)
    v1 = target_speed * np.array([np.cos(yaw1), np.sin(yaw1)], dtype=np.float64)

    duration = _sample_uniform(rng, cfg.duration_range)
    return BoundaryCondition(
        p0=p0,
        yaw0=yaw0,
        v0=v0,
        yaw_rate0=0.0,
        p1=p1,
        yaw1=yaw1,
        v1=v1,
        yaw_rate1=0.0,
        duration=duration,
        lateral_offset=lateral_offset,
    )


def _generate_free_quintic_trajectory(boundary: BoundaryCondition, fps: float) -> Trajectory:
    """Generate a C2 continuous minimum-jerk trajectory."""
    duration = float(boundary.duration)
    n = max(2, int(np.ceil(duration * fps)) + 1)
    t = np.linspace(0.0, duration, n, dtype=np.float64)

    pos_coeff = quintic_coefficients(
        boundary.p0,
        boundary.v0,
        np.zeros(2, dtype=np.float64),
        boundary.p1,
        boundary.v1,
        np.zeros(2, dtype=np.float64),
        duration,
    )
    pos, vel, acc = eval_quintic(pos_coeff, t)

    yaw1 = unwrap_to_near(boundary.yaw1, boundary.yaw0)
    yaw_coeff = quintic_coefficients(
        np.array(boundary.yaw0),
        np.array(boundary.yaw_rate0),
        np.array(0.0),
        np.array(yaw1),
        np.array(boundary.yaw_rate1),
        np.array(0.0),
        duration,
    )
    yaw, yaw_rate, yaw_acc = eval_quintic(yaw_coeff, t)
    yaw = wrap_angle(yaw)
    return Trajectory(t=t, pos=pos, vel=vel, acc=acc, yaw=yaw, yaw_rate=yaw_rate,
                      yaw_acc=yaw_acc, boundary=boundary)


def _generate_omni_progress_trajectory(boundary: BoundaryCondition, fps: float) -> Trajectory:
    """Generate an omnidirectional trajectory with near-straight XY progress.

    Translation is defined by scalar progress from p0 to p1 plus an optional
    bounded lateral bump. Body yaw is independent from translation direction, so
    the character can move forward, backward, or sideways relative to its yaw.
    """
    duration = float(boundary.duration)
    n = max(2, int(np.ceil(duration * fps)) + 1)
    t = np.linspace(0.0, duration, n, dtype=np.float64)

    delta = np.asarray(boundary.p1 - boundary.p0, dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        direction = np.array([1.0, 0.0], dtype=np.float64)
        dist = 1e-6
    else:
        direction = delta / dist
    left = np.array([-direction[1], direction[0]], dtype=np.float64)

    # Endpoint speeds affect progress along the path, not body yaw. Negative
    # start projection is clamped so a "move to target" query does not begin by
    # stepping away from the target.
    s_dot0 = max(0.0, float(np.dot(boundary.v0, direction)) / dist)
    s_dot1 = max(0.0, float(np.linalg.norm(boundary.v1)) / dist)
    s_coeff = quintic_coefficients(
        np.array(0.0), np.array(s_dot0), np.array(0.0),
        np.array(1.0), np.array(s_dot1), np.array(0.0),
        duration,
    )
    s, s_dot, s_ddot = eval_quintic(s_coeff, t)

    lateral_offset = float(boundary.lateral_offset)
    lateral = 0.5 * lateral_offset * (1.0 - np.cos(2.0 * np.pi * s))
    lateral_dot = lateral_offset * np.pi * np.sin(2.0 * np.pi * s) * s_dot
    lateral_ddot = lateral_offset * np.pi * (
        2.0 * np.pi * np.cos(2.0 * np.pi * s) * s_dot * s_dot
        + np.sin(2.0 * np.pi * s) * s_ddot
    )

    pos = boundary.p0 + np.outer(s * dist, direction) + np.outer(lateral, left)
    vel = np.outer(s_dot * dist, direction) + np.outer(lateral_dot, left)
    acc = np.outer(s_ddot * dist, direction) + np.outer(lateral_ddot, left)

    yaw1 = unwrap_to_near(boundary.yaw1, boundary.yaw0)
    yaw_coeff = quintic_coefficients(
        np.array(boundary.yaw0),
        np.array(boundary.yaw_rate0),
        np.array(0.0),
        np.array(yaw1),
        np.array(boundary.yaw_rate1),
        np.array(0.0),
        duration,
    )
    yaw, yaw_rate, yaw_acc = eval_quintic(yaw_coeff, t)
    yaw = wrap_angle(yaw)
    return Trajectory(t=t, pos=pos, vel=vel, acc=acc, yaw=yaw, yaw_rate=yaw_rate,
                      yaw_acc=yaw_acc, boundary=boundary)


def _generate_straight_omni_duration(boundary: BoundaryCondition, fps: float) -> Trajectory:
    """Generate a straight XY path with independent yaw over a fixed duration."""
    straight_boundary = replace(boundary, lateral_offset=0.0)
    return _generate_omni_progress_trajectory(straight_boundary, fps=fps)


def _generate_straight_omni_min_time_trajectory(
    boundary: BoundaryCondition,
    fps: float,
    cfg: SamplingConfig,
) -> Trajectory:
    """Fast shortest-feasible straight-path omnidirectional trajectory.

    There is no path optimization here: XY is constrained to the line from start
    to target. Duration is the only search variable, checked against body-frame
    forward/lateral/backward speed and acceleration limits.
    """
    def make(duration: float) -> Trajectory:
        return _generate_straight_omni_duration(replace(boundary, duration=float(duration)), fps=fps)

    if not cfg.opt_minimize_duration:
        return make(boundary.duration)

    lo, hi = cfg.duration_range
    lo = max(1e-3, float(lo))
    hi = max(lo, float(hi))
    high = make(hi)
    if not trajectory_is_feasible(high, cfg):
        return high

    low = make(lo)
    if trajectory_is_feasible(low, cfg):
        return low

    best = high
    bad, good = lo, hi
    for _ in range(max(0, int(cfg.opt_duration_search_steps))):
        mid = 0.5 * (bad + good)
        candidate = make(mid)
        if trajectory_is_feasible(candidate, cfg):
            best = candidate
            good = mid
        else:
            bad = mid
    return best


def _world_vel_to_body(world_vel: np.ndarray, yaw: float | np.ndarray) -> np.ndarray:
    world_vel = np.asarray(world_vel, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy = world_vel[..., 0], world_vel[..., 1]
    return np.stack([c * vx + s * vy, -s * vx + c * vy], axis=-1)


def _body_vel_to_world(body_vel: np.ndarray, yaw: float | np.ndarray) -> np.ndarray:
    body_vel = np.asarray(body_vel, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    vf, vl = body_vel[..., 0], body_vel[..., 1]
    return np.stack([c * vf - s * vl, s * vf + c * vl], axis=-1)


def _control_limits(cfg: SamplingConfig) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([-cfg.opt_backward_speed, -cfg.opt_lateral_speed, -cfg.max_yaw_rate], dtype=np.float64)
    hi = np.array([cfg.opt_forward_speed, cfg.opt_lateral_speed, cfg.max_yaw_rate], dtype=np.float64)
    return lo, hi


def _control_in_limits(control: np.ndarray, cfg: SamplingConfig, eps: float = 1e-6) -> bool:
    lo, hi = _control_limits(cfg)
    control = np.asarray(control, dtype=np.float64)
    return bool(np.all(control >= lo - eps) and np.all(control <= hi + eps))


def _simulate_control_knots(
    boundary: BoundaryCondition,
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate body-frame velocity/yaw-rate control knots with midpoint dynamics."""
    controls = np.asarray(controls, dtype=np.float64)
    k_count = int(controls.shape[0])
    if k_count < 2:
        raise ValueError("optimized trajectory needs at least two control knots")

    duration = float(boundary.duration)
    knot_t = np.linspace(0.0, duration, k_count, dtype=np.float64)
    pos = np.zeros((k_count, 2), dtype=np.float64)
    yaw = np.zeros(k_count, dtype=np.float64)
    pos[0] = boundary.p0
    yaw[0] = float(boundary.yaw0)

    for k in range(k_count - 1):
        dt = float(knot_t[k + 1] - knot_t[k])
        u_mid = 0.5 * (controls[k] + controls[k + 1])
        yaw_mid = yaw[k] + 0.5 * u_mid[2] * dt
        pos[k + 1] = pos[k] + dt * _body_vel_to_world(u_mid[:2], yaw_mid)
        yaw[k + 1] = yaw[k] + dt * u_mid[2]
    return knot_t, pos, yaw


def _optimized_initial_controls(boundary: BoundaryCondition, cfg: SamplingConfig, k_count: int) -> np.ndarray:
    base = _generate_omni_progress_trajectory(boundary, fps=max(cfg.fps, k_count / max(boundary.duration, 1e-6)))
    knot_t = np.linspace(0.0, boundary.duration, k_count, dtype=np.float64)
    _, world_vel, yaw = _interp_traj(base, knot_t)
    body_vel = _world_vel_to_body(world_vel, yaw)
    yaw_rate = np.interp(knot_t, base.t, base.yaw_rate)
    controls = np.column_stack([body_vel, yaw_rate])

    yaw1 = unwrap_to_near(boundary.yaw1, boundary.yaw0)
    controls[0] = np.array([
        *_world_vel_to_body(boundary.v0, boundary.yaw0),
        boundary.yaw_rate0,
    ], dtype=np.float64)
    controls[-1] = np.array([
        *_world_vel_to_body(boundary.v1, yaw1),
        boundary.yaw_rate1,
    ], dtype=np.float64)

    lo, hi = _control_limits(cfg)
    controls = np.clip(controls, lo, hi)
    return controls


def _trajectory_from_control_knots(
    boundary: BoundaryCondition,
    knot_t: np.ndarray,
    knot_pos: np.ndarray,
    knot_yaw: np.ndarray,
    controls: np.ndarray,
    fps: float,
) -> Trajectory:
    """Cubic-Hermite resample optimized knots into the regular trajectory grid."""
    duration = float(boundary.duration)
    n = max(2, int(np.ceil(duration * fps)) + 1)
    t = np.linspace(0.0, duration, n, dtype=np.float64)
    seg = np.searchsorted(knot_t, t, side="right") - 1
    seg = np.clip(seg, 0, len(knot_t) - 2)

    t0 = knot_t[seg]
    t1 = knot_t[seg + 1]
    dt = np.maximum(t1 - t0, 1e-9)
    s = (t - t0) / dt
    s2, s3 = s * s, s * s * s

    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    dh00 = 6.0 * s2 - 6.0 * s
    dh10 = 3.0 * s2 - 4.0 * s + 1.0
    dh01 = -6.0 * s2 + 6.0 * s
    dh11 = 3.0 * s2 - 2.0 * s
    ddh00 = 12.0 * s - 6.0
    ddh10 = 6.0 * s - 4.0
    ddh01 = -12.0 * s + 6.0
    ddh11 = 6.0 * s - 2.0

    p0 = knot_pos[seg]
    p1 = knot_pos[seg + 1]
    y0 = knot_yaw[seg]
    y1 = knot_yaw[seg + 1]
    v0 = _body_vel_to_world(controls[seg, :2], y0)
    v1 = _body_vel_to_world(controls[seg + 1, :2], y1)
    yr0 = controls[seg, 2]
    yr1 = controls[seg + 1, 2]

    dt_col = dt[:, None]
    pos = (
        h00[:, None] * p0
        + h10[:, None] * dt_col * v0
        + h01[:, None] * p1
        + h11[:, None] * dt_col * v1
    )
    vel = (
        dh00[:, None] * p0
        + dh10[:, None] * dt_col * v0
        + dh01[:, None] * p1
        + dh11[:, None] * dt_col * v1
    ) / dt_col
    acc = (
        ddh00[:, None] * p0
        + ddh10[:, None] * dt_col * v0
        + ddh01[:, None] * p1
        + ddh11[:, None] * dt_col * v1
    ) / (dt_col * dt_col)

    yaw = h00 * y0 + h10 * dt * yr0 + h01 * y1 + h11 * dt * yr1
    yaw_rate = (dh00 * y0 + dh10 * dt * yr0 + dh01 * y1 + dh11 * dt * yr1) / dt
    yaw_acc = (ddh00 * y0 + ddh10 * dt * yr0 + ddh01 * y1 + ddh11 * dt * yr1) / (dt * dt)
    return Trajectory(t=t, pos=pos, vel=vel, acc=acc, yaw=wrap_angle(yaw), yaw_rate=yaw_rate,
                      yaw_acc=yaw_acc, boundary=boundary)


def _optimize_omni_fixed_duration(
    boundary: BoundaryCondition,
    fps: float,
    cfg: SamplingConfig,
) -> Trajectory | None:
    """Optimize body-frame omnidirectional controls for one fixed duration."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return None

    k_count = max(6, int(cfg.opt_knots))
    duration = float(boundary.duration)
    if duration <= 1e-6:
        return None

    yaw1 = unwrap_to_near(boundary.yaw1, boundary.yaw0)
    start_control = np.array([
        *_world_vel_to_body(boundary.v0, boundary.yaw0),
        boundary.yaw_rate0,
    ], dtype=np.float64)
    end_control = np.array([
        *_world_vel_to_body(boundary.v1, yaw1),
        boundary.yaw_rate1,
    ], dtype=np.float64)
    if not _control_in_limits(start_control, cfg) or not _control_in_limits(end_control, cfg):
        return None

    u0 = _optimized_initial_controls(boundary, cfg, k_count)
    u0[0] = start_control
    u0[-1] = end_control
    lo, hi = _control_limits(cfg)
    bounds = [(float(lo[j]), float(hi[j])) for _ in range(k_count) for j in range(3)]
    knot_dt = duration / float(k_count - 1)
    accel_scale = np.array([
        max(cfg.opt_forward_accel, cfg.opt_forward_brake, cfg.opt_backward_accel, cfg.opt_backward_brake, 1e-6),
        max(cfg.opt_lateral_accel, 1e-6),
        max(cfg.opt_yaw_accel, 1e-6),
    ], dtype=np.float64)

    delta = np.asarray(boundary.p1 - boundary.p0, dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    if dist > 1e-6:
        target_dir = delta / dist
        target_left = np.array([-target_dir[1], target_dir[0]], dtype=np.float64)
    else:
        target_dir = np.array([1.0, 0.0], dtype=np.float64)
        target_left = np.array([0.0, 1.0], dtype=np.float64)

    def unpack(z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64).reshape(k_count, 3)

    def equality(z: np.ndarray) -> np.ndarray:
        controls = unpack(z)
        _, pos, yaw = _simulate_control_knots(boundary, controls)
        return np.concatenate([
            pos[-1] - boundary.p1,
            np.array([yaw[-1] - yaw1], dtype=np.float64),
            controls[0] - start_control,
            controls[-1] - end_control,
        ])

    def inequality(z: np.ndarray) -> np.ndarray:
        controls = unpack(z)
        du = np.diff(controls, axis=0) / knot_dt
        mid_forward = 0.5 * (controls[:-1, 0] + controls[1:, 0])
        pos_forward_limit = np.where(mid_forward >= 0.0, cfg.opt_forward_accel, cfg.opt_backward_brake)
        neg_forward_limit = np.where(mid_forward >= 0.0, cfg.opt_forward_brake, cfg.opt_backward_accel)
        speed_norm = np.linalg.norm(controls[:, :2], axis=-1)
        accel_norm = np.linalg.norm(du[:, :2], axis=-1)
        return np.concatenate([
            pos_forward_limit - du[:, 0],
            neg_forward_limit + du[:, 0],
            cfg.opt_lateral_accel - du[:, 1],
            cfg.opt_lateral_accel + du[:, 1],
            cfg.opt_yaw_accel - du[:, 2],
            cfg.opt_yaw_accel + du[:, 2],
            cfg.max_speed - speed_norm,
            cfg.max_accel - accel_norm,
        ])

    def objective(z: np.ndarray) -> float:
        controls = unpack(z)
        _, pos, yaw = _simulate_control_knots(boundary, controls)
        du = np.diff(controls, axis=0) / knot_dt
        cost = 0.0
        cost += cfg.opt_w_speed * float(np.mean(
            (controls[:, 0] / max(cfg.opt_forward_speed, 1e-6)) ** 2
            + (controls[:, 1] / max(cfg.opt_lateral_speed, 1e-6)) ** 2
            + (controls[:, 2] / max(cfg.max_yaw_rate, 1e-6)) ** 2
        ))
        cost += cfg.opt_w_accel * float(np.mean((du / accel_scale) ** 2))
        if len(controls) > 2:
            ddu = np.diff(du, axis=0) / knot_dt
            cost += cfg.opt_w_jerk * float(np.mean((ddu / accel_scale) ** 2))
        if cfg.opt_w_backward > 0.0:
            back = np.minimum(controls[:, 0], 0.0)
            cost += cfg.opt_w_backward * float(np.mean((back / max(cfg.opt_backward_speed, 1e-6)) ** 2))
        if cfg.opt_w_lateral > 0.0:
            cost += cfg.opt_w_lateral * float(np.mean((controls[:, 1] / max(cfg.opt_lateral_speed, 1e-6)) ** 2))
        cost += cfg.opt_w_yaw_rate * float(np.mean((controls[:, 2] / max(cfg.max_yaw_rate, 1e-6)) ** 2))
        cost += cfg.opt_w_yaw_deviation * float(np.mean((wrap_angle(yaw - boundary.yaw0) / np.pi) ** 2))
        if dist > 1e-6:
            rel = pos - boundary.p0
            cross_track = rel @ target_left
            progress = rel @ target_dir
            cost += cfg.opt_w_path * float(np.mean((cross_track / max(dist, 1.0)) ** 2))
            progress_step = np.diff(progress)
            cost += cfg.opt_w_reverse_progress * float(np.mean(np.minimum(progress_step, 0.0) ** 2))
        return float(cost)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Values in x were outside bounds*", category=RuntimeWarning)
        result = minimize(
            objective,
            u0.reshape(-1),
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "eq", "fun": equality},
                {"type": "ineq", "fun": inequality},
            ],
            options={"maxiter": int(cfg.opt_max_iter), "ftol": float(cfg.opt_ftol), "disp": False},
        )
    controls = unpack(result.x if np.isfinite(result.x).all() else u0.reshape(-1))
    knot_t, knot_pos, knot_yaw = _simulate_control_knots(boundary, controls)
    term_pos_err = float(np.linalg.norm(knot_pos[-1] - boundary.p1))
    term_yaw_err = float(abs(knot_yaw[-1] - yaw1))
    min_margin = float(np.min(inequality(controls.reshape(-1))))
    if term_pos_err > cfg.opt_terminal_tol or term_yaw_err > 0.15 or min_margin < -1e-3:
        return None
    return _trajectory_from_control_knots(boundary, knot_t, knot_pos, knot_yaw, controls, fps=fps)


def _generate_optimized_omni_trajectory(
    boundary: BoundaryCondition,
    fps: float,
    cfg: SamplingConfig,
) -> Trajectory:
    """Optimize the shortest feasible body-frame omnidirectional trajectory."""
    def try_duration(duration: float) -> Trajectory | None:
        candidate_bc = replace(boundary, duration=float(duration))
        candidate = _optimize_omni_fixed_duration(candidate_bc, fps=fps, cfg=cfg)
        if candidate is None:
            return None
        return candidate if trajectory_is_feasible(candidate, cfg) else None

    if not cfg.opt_minimize_duration:
        candidate = try_duration(boundary.duration)
        if candidate is not None:
            return candidate
        return _generate_omni_progress_trajectory(boundary, fps=fps)

    lo, hi = cfg.duration_range
    lo = max(1e-3, float(lo))
    hi = max(lo, float(hi))

    best = try_duration(hi)
    if best is None:
        return _generate_omni_progress_trajectory(replace(boundary, duration=hi), fps=fps)

    low_candidate = try_duration(lo)
    if low_candidate is not None:
        return low_candidate

    bad, good = lo, hi
    for _ in range(max(0, int(cfg.opt_duration_search_steps))):
        mid = 0.5 * (bad + good)
        candidate = try_duration(mid)
        if candidate is None:
            bad = mid
        else:
            best = candidate
            good = mid
    return best


def generate_trajectory(boundary: BoundaryCondition, fps: float = 50.0,
                        model: str = "optimized_omni",
                        cfg: SamplingConfig | None = None) -> Trajectory:
    model = str(model)
    if model == "free_quintic":
        return _generate_free_quintic_trajectory(boundary, fps=fps)
    if model == "omni_progress":
        return _generate_omni_progress_trajectory(boundary, fps=fps)
    if model == "straight_omni_min_time":
        if cfg is None:
            cfg = SamplingConfig(fps=fps, trajectory_model=model)
        return _generate_straight_omni_min_time_trajectory(boundary, fps=fps, cfg=cfg)
    if model == "optimized_omni":
        if cfg is None:
            cfg = SamplingConfig(fps=fps, trajectory_model=model)
        return _generate_optimized_omni_trajectory(boundary, fps=fps, cfg=cfg)
    raise ValueError(f"unknown trajectory model: {model}")


def _body_limit_feasible(traj: Trajectory, cfg: SamplingConfig, slack: float = 1.12) -> bool:
    body_vel = _world_vel_to_body(traj.vel, traj.yaw)
    if len(traj.t) > 1:
        body_acc = np.gradient(body_vel, traj.t, axis=0, edge_order=1)
    else:
        body_acc = np.zeros_like(body_vel)

    forward_ok = bool(np.all(body_vel[:, 0] <= cfg.opt_forward_speed * slack))
    backward_ok = bool(np.all(body_vel[:, 0] >= -cfg.opt_backward_speed * slack))
    lateral_ok = bool(np.all(np.abs(body_vel[:, 1]) <= cfg.opt_lateral_speed * slack))

    mid_forward = body_vel[:, 0]
    forward_accel_limit = np.where(mid_forward >= 0.0, cfg.opt_forward_accel, cfg.opt_backward_brake)
    forward_brake_limit = np.where(mid_forward >= 0.0, cfg.opt_forward_brake, cfg.opt_backward_accel)
    accel_ok = bool(np.all(body_acc[:, 0] <= forward_accel_limit * slack))
    brake_ok = bool(np.all(body_acc[:, 0] >= -forward_brake_limit * slack))
    lateral_accel_ok = bool(np.all(np.abs(body_acc[:, 1]) <= cfg.opt_lateral_accel * slack))
    yaw_accel_ok = bool(np.all(np.abs(traj.yaw_acc) <= cfg.opt_yaw_accel * slack))
    return forward_ok and backward_ok and lateral_ok and accel_ok and brake_ok and lateral_accel_ok and yaw_accel_ok


def trajectory_is_feasible(traj: Trajectory, cfg: SamplingConfig) -> bool:
    speed = np.linalg.norm(traj.vel, axis=-1)
    accel = np.linalg.norm(traj.acc, axis=-1)
    yaw_rate = np.abs(traj.yaw_rate)
    finite_ok = all(np.isfinite(x).all() for x in (traj.pos, traj.vel, traj.acc, traj.yaw, traj.yaw_rate))
    bounds_ok = (
        bool(np.all(speed <= cfg.max_speed))
        and bool(np.all(accel <= cfg.max_accel))
        and bool(np.all(yaw_rate <= cfg.max_yaw_rate))
    )
    if cfg.trajectory_model in ("optimized_omni", "straight_omni_min_time"):
        body_slack = 1.03 if cfg.trajectory_model == "straight_omni_min_time" else 1.12
        if not _body_limit_feasible(traj, cfg, slack=body_slack):
            return False
    if cfg.trajectory_model not in ("omni_progress", "optimized_omni", "straight_omni_min_time"):
        return finite_ok and bounds_ok

    delta = traj.boundary.p1 - traj.boundary.p0
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return finite_ok and bounds_ok
    direction = delta / dist
    progress = (traj.pos - traj.boundary.p0) @ direction / dist
    monotonic_ok = bool(np.min(np.diff(progress)) >= -0.025)
    range_ok = bool(np.min(progress) >= -0.03 and np.max(progress) <= 1.03)
    return finite_ok and bounds_ok and monotonic_ok and range_ok


def sample_feasible_trajectory(rng: np.random.Generator, cfg: SamplingConfig) -> Trajectory:
    last = None
    for _ in range(max(1, cfg.max_tries)):
        bc = sample_boundary(rng, cfg)
        traj = generate_trajectory(bc, fps=cfg.fps, model=cfg.trajectory_model, cfg=cfg)
        last = traj
        if trajectory_is_feasible(traj, cfg):
            return traj
    assert last is not None
    return last


def sample_query_features(
    traj: Trajectory,
    offsets: Iterable[int] = DEFAULT_OFFSETS,
    fps: float = 50.0,
) -> dict[str, np.ndarray]:
    """Sample trajectory features at MM frame offsets.

    Times beyond the trajectory duration hold the terminal pose and heading.
    Returned arrays:
      pos: [n_offsets, 2]
      heading: [n_offsets, 2]
      vel: [n_offsets, 2]
      yaw: [n_offsets]
    """
    offsets = tuple(int(o) for o in offsets)
    sample_t = np.asarray(offsets, dtype=np.float64) / float(fps)
    sample_t_clamped = np.minimum(sample_t, traj.t[-1])

    pos = np.stack([np.interp(sample_t_clamped, traj.t, traj.pos[:, i]) for i in range(2)], axis=-1)
    vel = np.stack([np.interp(sample_t_clamped, traj.t, traj.vel[:, i]) for i in range(2)], axis=-1)
    yaw_unwrapped = np.unwrap(traj.yaw)
    yaw = wrap_angle(np.interp(sample_t_clamped, traj.t, yaw_unwrapped))
    heading = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)

    held = sample_t > traj.t[-1]
    if np.any(held):
        pos[held] = traj.pos[-1]
        vel[held] = 0.0
        yaw[held] = traj.yaw[-1]
        heading[held] = traj.heading[-1]
    return {"pos": pos, "heading": heading, "vel": vel, "yaw": yaw, "offsets": np.asarray(offsets)}


def _interp_traj(traj: Trajectory, sample_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_t = np.asarray(sample_t, dtype=np.float64)
    sample_t_clamped = np.minimum(np.maximum(sample_t, 0.0), traj.t[-1])
    pos = np.stack([np.interp(sample_t_clamped, traj.t, traj.pos[:, i]) for i in range(2)], axis=-1)
    vel = np.stack([np.interp(sample_t_clamped, traj.t, traj.vel[:, i]) for i in range(2)], axis=-1)
    yaw = wrap_angle(np.interp(sample_t_clamped, traj.t, np.unwrap(traj.yaw)))
    held = sample_t > traj.t[-1]
    if np.any(held):
        pos[held] = traj.pos[-1]
        vel[held] = 0.0
        yaw[held] = traj.yaw[-1]
    return pos, vel, yaw


def _rot2(yaw: float) -> np.ndarray:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def sample_future_query_at(
    traj: Trajectory,
    current_time: float,
    offsets: Iterable[int] = DEFAULT_OFFSETS,
    fps: float = 50.0,
) -> dict[str, np.ndarray]:
    """Sample future trajectory features relative to the pose at current_time.

    Returned positions are in the current root-facing frame. This is the form
    expected by the MM trajectory feature: local future XY points and local
    future heading vectors.
    """
    offsets = tuple(int(o) for o in offsets)
    now_pos, _, now_yaw_arr = _interp_traj(traj, np.asarray([current_time], dtype=np.float64))
    now_yaw = float(now_yaw_arr[0])
    sample_t = float(current_time) + np.asarray(offsets, dtype=np.float64) / float(fps)
    fut_pos, fut_vel, fut_yaw = _interp_traj(traj, sample_t)

    R_inv = _rot2(-now_yaw)
    local_pos = (R_inv @ (fut_pos - now_pos[0]).T).T
    local_vel = (R_inv @ fut_vel.T).T
    rel_yaw = wrap_angle(fut_yaw - now_yaw)
    local_heading = np.stack([np.cos(rel_yaw), np.sin(rel_yaw)], axis=-1)
    return {
        "pos": local_pos,
        "vel": local_vel,
        "heading": local_heading,
        "yaw": rel_yaw,
        "world_pos": fut_pos,
        "world_yaw": fut_yaw,
        "offsets": np.asarray(offsets),
    }


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _add_sphere(viewer, pos: np.ndarray, rgba: Iterable[float], radius: float = 0.05) -> None:
    import mujoco

    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


def _add_capsule(viewer, p0: np.ndarray, p1: np.ndarray, rgba: Iterable[float], width: float = 0.015) -> None:
    import mujoco

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    if np.linalg.norm(p1 - p0) < 1e-6 or viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.zeros(9, dtype=np.float64),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, int(mujoco.mjtGeom.mjGEOM_CAPSULE), width, p0, p1)
    viewer.user_scn.ngeom += 1


def _draw_arrow(viewer, p0: np.ndarray, vec: np.ndarray, rgba: Iterable[float], width: float = 0.018) -> None:
    _add_capsule(viewer, p0, p0 + vec, rgba, width=width)


def _draw_circle(viewer, center_xy: np.ndarray, radius: float, rgba: Iterable[float],
                 z: float, width: float, segments: int = 96) -> None:
    center_xy = np.asarray(center_xy, dtype=np.float64)
    radius = float(radius)
    if radius <= 1e-6:
        return
    theta = np.linspace(0.0, 2.0 * np.pi, max(12, int(segments)) + 1, dtype=np.float64)
    pts = np.column_stack([
        center_xy[0] + radius * np.cos(theta),
        center_xy[1] + radius * np.sin(theta),
        np.full_like(theta, z),
    ])
    for a, b in zip(pts[:-1], pts[1:]):
        _add_capsule(viewer, a, b, rgba, width=width)


def _court_line(viewer, x0: float, y0: float, x1: float, y1: float,
                rgba: Iterable[float], z: float, width: float) -> None:
    _add_capsule(
        viewer,
        np.array([x0, y0, z], dtype=np.float64),
        np.array([x1, y1, z], dtype=np.float64),
        rgba,
        width=width,
    )


def _draw_tennis_court_overlay(viewer) -> None:
    """Draw a regulation tennis court on the ground.

    Coordinate convention: court length is world X, court width is world Y,
    and the net center is at world origin.
    """
    z = 0.012
    line = [0.95, 0.95, 0.90, 0.85]
    singles = [0.80, 0.95, 1.0, 0.65]
    service = [0.95, 0.95, 0.90, 0.72]
    net = [0.05, 0.05, 0.05, 0.90]
    width = 0.018

    half_len = 23.77 * 0.5
    half_doubles = 10.97 * 0.5
    half_singles = 8.23 * 0.5
    service_x = 6.40

    # Doubles court outer rectangle.
    _court_line(viewer, -half_len, -half_doubles, half_len, -half_doubles, line, z, width)
    _court_line(viewer, -half_len, half_doubles, half_len, half_doubles, line, z, width)
    _court_line(viewer, -half_len, -half_doubles, -half_len, half_doubles, line, z, width)
    _court_line(viewer, half_len, -half_doubles, half_len, half_doubles, line, z, width)

    # Singles sidelines.
    _court_line(viewer, -half_len, -half_singles, half_len, -half_singles, singles, z, width)
    _court_line(viewer, -half_len, half_singles, half_len, half_singles, singles, z, width)

    # Service boxes and center service line.
    _court_line(viewer, -service_x, -half_singles, -service_x, half_singles, service, z, width)
    _court_line(viewer, service_x, -half_singles, service_x, half_singles, service, z, width)
    _court_line(viewer, -service_x, 0.0, service_x, 0.0, service, z, width)

    # Small baseline center marks.
    _court_line(viewer, -half_len, -0.12, -half_len, 0.12, service, z, width)
    _court_line(viewer, half_len, -0.12, half_len, 0.12, service, z, width)

    # Net line, lifted slightly so it remains visible above the court lines.
    _court_line(viewer, 0.0, -half_doubles, 0.0, half_doubles, net, z + 0.07, 0.024)


def _draw_trajectory_overlay(
    viewer,
    traj: Trajectory,
    current_time: float,
    offsets: tuple[int, ...],
    fps: float,
    draft_target: tuple[np.ndarray, float] | None = None,
    root_radius: float | None = 2.0,
) -> None:
    z = 0.035
    viewer.user_scn.ngeom = 0
    _draw_tennis_court_overlay(viewer)

    # Full trajectory path.
    stride = max(1, len(traj.pos) // 120)
    path = traj.pos[::stride]
    for a, b in zip(path[:-1], path[1:]):
        _add_capsule(viewer, np.array([a[0], a[1], z]), np.array([b[0], b[1], z]), [0.20, 0.45, 1.0, 0.45], 0.012)

    start = np.array([traj.pos[0, 0], traj.pos[0, 1], z + 0.02])
    target = np.array([traj.pos[-1, 0], traj.pos[-1, 1], z + 0.02])
    _add_sphere(viewer, start, [0.0, 1.0, 0.3, 1.0], 0.08)
    _add_sphere(viewer, target, [1.0, 0.15, 0.1, 1.0], 0.10)
    target_heading = np.array([np.cos(traj.yaw[-1]), np.sin(traj.yaw[-1]), 0.0])
    _draw_arrow(viewer, target, target_heading * 0.75, [1.0, 0.0, 1.0, 1.0], 0.024)
    if draft_target is not None:
        draft_p, draft_yaw = draft_target
        draft = np.array([draft_p[0], draft_p[1], z + 0.14])
        draft_heading = np.array([np.cos(draft_yaw), np.sin(draft_yaw), 0.0])
        _add_sphere(viewer, draft, [1.0, 0.85, 0.0, 1.0], 0.12)
        _draw_arrow(viewer, draft, draft_heading * 0.90, [1.0, 0.85, 0.0, 1.0], 0.028)

    pos, vel, yaw = _interp_traj(traj, np.asarray([current_time], dtype=np.float64))
    cur = np.array([pos[0, 0], pos[0, 1], z + 0.05])
    if root_radius is not None and root_radius > 0.0:
        _draw_circle(viewer, pos[0], float(root_radius),
                     [1.0, 0.82, 0.05, 0.70], z + 0.01, 0.012)
    heading = np.array([np.cos(yaw[0]), np.sin(yaw[0]), 0.0])
    _add_sphere(viewer, cur, [1.0, 1.0, 1.0, 1.0], 0.07)
    _draw_arrow(viewer, cur, heading * 0.55, [1.0, 0.0, 0.0, 1.0], 0.022)
    _draw_arrow(viewer, cur, np.array([vel[0, 0], vel[0, 1], 0.0]) * 0.25, [1.0, 0.55, 0.0, 1.0], 0.018)

    # Current MM query future samples.
    query = sample_future_query_at(traj, current_time, offsets=offsets, fps=fps)
    prev = cur.copy()
    for wp, wyaw, off in zip(query["world_pos"], query["world_yaw"], query["offsets"]):
        p = np.array([wp[0], wp[1], z + 0.09])
        _add_sphere(viewer, p, [1.0, 1.0, 1.0, 1.0], 0.055)
        _add_capsule(viewer, prev, p, [1.0, 1.0, 1.0, 0.70], 0.010)
        h = np.array([np.cos(wyaw), np.sin(wyaw), 0.0])
        _draw_arrow(viewer, p, h * 0.35, [0.0, 1.0, 1.0, 1.0], 0.014)
        prev = p

    # World +X/+Y axes near the origin.
    _draw_arrow(viewer, np.array([0.0, 0.0, z]), np.array([0.7, 0.0, 0.0]), [1.0, 0.0, 0.0, 1.0], 0.012)
    _draw_arrow(viewer, np.array([0.0, 0.0, z]), np.array([0.0, 0.7, 0.0]), [0.0, 1.0, 0.0, 1.0], 0.012)


def _cursor_to_ground_point(
    window,
    cursor_x: float,
    cursor_y: float,
    viewer,
    model,
    data,
    opt,
    scene,
    ground_z: float = 0.0,
) -> np.ndarray | None:
    """Project a GLFW cursor position onto the horizontal ground plane."""
    import glfw
    import mujoco

    win_w, win_h = glfw.get_window_size(window)
    fb_w, fb_h = glfw.get_framebuffer_size(window)
    viewport = viewer.viewport
    if win_w <= 0 or win_h <= 0 or fb_w <= 0 or fb_h <= 0 or viewport is None:
        return None
    if viewport.width <= 0 or viewport.height <= 0:
        return None

    x_fb = float(cursor_x) * float(fb_w) / float(win_w)
    y_fb_top = float(cursor_y) * float(fb_h) / float(win_h)
    y_fb = float(fb_h) - y_fb_top
    relx = (x_fb - float(viewport.left)) / float(viewport.width)
    rely = (y_fb - float(viewport.bottom)) / float(viewport.height)
    if relx < 0.0 or relx > 1.0 or rely < 0.0 or rely > 1.0:
        return None

    mujoco.mjv_updateScene(model, data, opt, None, viewer.cam,
                           mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    camera = scene.camera[0]
    pos = np.asarray(camera.pos, dtype=np.float64)
    forward = np.asarray(camera.forward, dtype=np.float64)
    up = np.asarray(camera.up, dtype=np.float64)
    forward /= max(1e-12, float(np.linalg.norm(forward)))
    up /= max(1e-12, float(np.linalg.norm(up)))
    right = np.cross(forward, up)
    right /= max(1e-12, float(np.linalg.norm(right)))

    near = max(1e-6, float(camera.frustum_near))
    frustum_height = float(camera.frustum_top - camera.frustum_bottom)
    aspect = float(viewport.width) / float(viewport.height)
    frustum_width = float(camera.frustum_width)
    if abs(frustum_width) < 1e-9:
        frustum_width = frustum_height * aspect
    x_near = float(camera.frustum_center) + (relx - 0.5) * frustum_width
    y_near = float(camera.frustum_bottom) + rely * frustum_height

    if int(camera.orthographic):
        ray_origin = pos + right * x_near + up * y_near
        ray_dir = forward
    else:
        ray_origin = pos
        ray_dir = forward * near + right * x_near + up * y_near
        ray_dir /= max(1e-12, float(np.linalg.norm(ray_dir)))

    denom = float(ray_dir[2])
    if abs(denom) < 1e-9:
        return None
    alpha = (float(ground_z) - float(ray_origin[2])) / denom
    if alpha <= 0.0:
        return None
    point = ray_origin + alpha * ray_dir
    if not np.isfinite(point).all():
        return None
    return point[:2].astype(np.float64)


def run_mujoco_viewer(
    traj: Trajectory,
    offsets: tuple[int, ...],
    fps: float,
    playback_fps: float,
    root_height: float,
    loop: bool,
    max_frames: int | None = None,
    rng: np.random.Generator | None = None,
    sampling_cfg: SamplingConfig | None = None,
    manual_planner: Callable[[BoundaryCondition], Trajectory] | None = None,
    resample_planner: Callable[[], Trajectory] | None = None,
    root_radius: float | None = 2.0,
) -> None:
    import mujoco
    import mujoco.viewer

    repo = Path(__file__).resolve().parents[1]
    xml = repo / "robots/replay_unitree_description/mjcf/g1.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    if model.nq < 36:
        raise RuntimeError(f"expected G1-like model with nq>=36, got {model.nq}")
    pick_opt = mujoco.MjvOption()
    pick_scn = mujoco.MjvScene(model, maxgeom=max(model.ngeom + 16, 128))

    state = {
        "traj": traj,
        "t0": time.perf_counter(),
        "paused": False,
        "reset_camera": True,
        "manual_edit": False,
        "dragging_target": False,
        "draft_p1": None,
        "draft_yaw": None,
        "pending_manual": None,
        "optimizing": False,
        "current_root_pos": np.asarray(traj.pos[0], dtype=np.float64).copy(),
        "current_root_vel": np.asarray(traj.vel[0], dtype=np.float64).copy(),
        "current_root_yaw": float(traj.yaw[0]),
        "current_root_yaw_rate": float(traj.yaw_rate[0]) if len(traj.yaw_rate) else 0.0,
        "current_root_time": 0.0,
        "next_root_status_update": 0.0,
        "mouse_window": None,
        "mouse_window_addr": None,
        "glfw_raw": None,                     # independent ctypes handle to the GLFW lib
        "prev_mouse_button_callback": None,   # RAW C fn pointers (viewer's camera handlers)
        "prev_cursor_pos_callback": None,
        "mouse_button_callback_c": None,
        "cursor_pos_callback_c": None,
    }
    _MB_SIG = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    _CP_SIG = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_double, ctypes.c_double)

    def _print_target(prefix: str, current: Trajectory) -> None:
        bc = current.boundary
        _log(
            f"{prefix} target=({bc.p1[0]:.2f},{bc.p1[1]:.2f}) "
            f"yaw={np.rad2deg(bc.yaw1):.1f}deg v1={np.linalg.norm(bc.v1):.2f} "
            f"T={bc.duration:.2f}s",
            style="cyan",
        )

    # MuJoCo's viewer installs its camera handlers as C++ GLFW callbacks — raw C function
    # pointers, not Python callables. pyGLFW cannot hold, call, or re-install them: passing the
    # pointer (an int) back through glfw.set_*_callback wraps it as "callable", and the next mouse
    # event raises TypeError inside pyGLFW's deferred-exception path, killing the render thread.
    # So capture / forward / restore the originals at the raw ctypes level.
    def _glfw_raw():
        import glfw
        if state.get("glfw_raw") is None:
            raw = ctypes.CDLL(glfw._glfw._name)
            for fn in (raw.glfwSetMouseButtonCallback, raw.glfwSetCursorPosCallback):
                fn.restype = ctypes.c_void_p                    # returns the PREVIOUS C callback
                fn.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
            state["glfw_raw"] = raw
        return state["glfw_raw"]

    def _win_addr(window):
        if isinstance(window, int):
            return window
        if isinstance(window, ctypes.c_void_p):
            return window.value
        return ctypes.cast(window, ctypes.c_void_p).value

    def _window_ptr(window):
        import glfw

        addr = _win_addr(window)
        return ctypes.cast(ctypes.c_void_p(addr), ctypes.POINTER(glfw._GLFWwindow))

    def _call_previous_callback(prev_ptr, sig, window, *args) -> None:
        """Forward an event to the viewer's ORIGINAL C callback (camera control)."""
        if not prev_ptr:
            return
        try:
            sig(prev_ptr)(_win_addr(window), *args)
        except Exception as exc:
            _log(f"[viewer] forward to camera callback failed: {exc}", style="yellow")

    def _manual_boundary(p1: np.ndarray, yaw: float) -> BoundaryCondition:
        cfg = sampling_cfg or SamplingConfig(fps=fps)
        current_bc = state["traj"].boundary
        speed1 = float(np.linalg.norm(current_bc.v1))
        v1 = speed1 * np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
        p0 = np.asarray(state["current_root_pos"], dtype=np.float64)
        v0 = np.asarray(state["current_root_vel"], dtype=np.float64)
        yaw0 = float(state["current_root_yaw"])
        yaw_rate0 = float(state["current_root_yaw_rate"])
        return BoundaryCondition(
            p0=p0.copy(),
            yaw0=yaw0,
            v0=v0.copy(),
            yaw_rate0=yaw_rate0,
            p1=np.asarray(p1, dtype=np.float64),
            yaw1=float(wrap_angle(yaw)),
            v1=v1,
            yaw_rate1=0.0,
            duration=float(cfg.duration_range[1]),
            lateral_offset=0.0,
        )

    def _queue_manual_target(p1: np.ndarray, yaw: float) -> None:
        state["pending_manual"] = (np.asarray(p1, dtype=np.float64).copy(), float(wrap_angle(yaw)))
        _log(
            f"[viewer manual target] queued target=({p1[0]:.2f},{p1[1]:.2f}) "
            f"yaw={np.rad2deg(wrap_angle(yaw)):.1f}deg",
            style="magenta",
        )

    def _root_status_panel(
        status: str,
        cur_t: float,
        duration: float,
        pos_xy: np.ndarray,
        vel_xy: np.ndarray,
        yaw_value: float,
        yaw_rate: float,
        speed: float,
    ) -> Panel:
        status_style = {
            "RUN": "green",
            "EDIT": "yellow",
            "OPT": "magenta",
            "DONE": "cyan",
        }.get(status, "white")
        table = Table.grid(padding=(0, 2), expand=True)
        table.add_column(style="dim", justify="right", no_wrap=True)
        table.add_column()
        table.add_row("mode", Text(status, style=f"bold {status_style}"))
        table.add_row("time", f"{cur_t:5.2f}/{duration:5.2f}s")
        table.add_row("pos", f"({pos_xy[0]: .3f}, {pos_xy[1]: .3f}, {root_height: .3f})")
        table.add_row("yaw", f"{np.rad2deg(yaw_value): .1f} deg")
        table.add_row("vel", f"({vel_xy[0]: .3f}, {vel_xy[1]: .3f})")
        table.add_row("speed", f"{speed: .3f} m/s")
        table.add_row("yaw rate", f"{yaw_rate: .3f} rad/s")
        table.add_row("keys", "M mouse edit | R resample | Space pause")
        return Panel(
            table,
            title="Root Status",
            border_style=status_style,
        )

    def _mouse_ground(window, cursor_x: float, cursor_y: float) -> np.ndarray | None:
        return _cursor_to_ground_point(window, cursor_x, cursor_y, viewer, model, data, pick_opt, pick_scn)

    def _mouse_button_callback(window, button: int, action: int, mods: int) -> None:
        # NEVER let an exception escape: pyGLFW re-raises it on the next glfw call inside the
        # render loop, killing the render thread.
        try:
            import glfw

            window_ptr = _window_ptr(window)
            if state["manual_edit"] and button == glfw.MOUSE_BUTTON_LEFT:
                if action == glfw.PRESS:
                    cursor_x, cursor_y = glfw.get_cursor_pos(window_ptr)
                    p = _mouse_ground(window_ptr, cursor_x, cursor_y)
                    if p is not None:
                        state["dragging_target"] = True
                        state["draft_p1"] = p
                        state["draft_yaw"] = float(state["traj"].boundary.yaw1)
                        return
                elif action == glfw.RELEASE:
                    if state["dragging_target"] and state["draft_p1"] is not None:
                        yaw = float(state["draft_yaw"] if state["draft_yaw"] is not None else state["traj"].boundary.yaw1)
                        _queue_manual_target(np.asarray(state["draft_p1"], dtype=np.float64), yaw)
                    state["dragging_target"] = False
                    return
            _call_previous_callback(state["prev_mouse_button_callback"], _MB_SIG,
                                    window_ptr, button, action, mods)
        except Exception as exc:
            _log(f"[viewer] mouse button callback error: {exc}", style="red")

    def _cursor_pos_callback(window, cursor_x: float, cursor_y: float) -> None:
        try:
            window_ptr = _window_ptr(window)
            if state["manual_edit"] and state["dragging_target"] and state["draft_p1"] is not None:
                p = _mouse_ground(window_ptr, cursor_x, cursor_y)
                if p is not None:
                    delta = p - np.asarray(state["draft_p1"], dtype=np.float64)
                    if float(np.linalg.norm(delta)) >= 0.15:
                        state["draft_yaw"] = float(np.arctan2(delta[1], delta[0]))
                    return
            _call_previous_callback(state["prev_cursor_pos_callback"], _CP_SIG,
                                    window_ptr, cursor_x, cursor_y)
        except Exception as exc:
            _log(f"[viewer] cursor callback error: {exc}", style="red")

    def _install_mouse_callbacks() -> bool:
        import glfw

        window = glfw.get_current_context()
        if not window:
            _log("[viewer] mouse edit unavailable: GLFW context not found", style="yellow")
            return False
        addr = _win_addr(window)
        if state["mouse_window_addr"] == addr:
            return True
        raw = _glfw_raw()
        state["mouse_window"] = window
        state["mouse_window_addr"] = addr
        # Capture the viewer's original C callbacks as RAW pointers (clear-and-read), then
        # install ours raw too. pyGLFW's None/int callback path can leave an int callable
        # registered and crash the render thread on the next GLFW event.
        state["prev_mouse_button_callback"] = raw.glfwSetMouseButtonCallback(addr, None)
        state["prev_cursor_pos_callback"] = raw.glfwSetCursorPosCallback(addr, None)
        state["mouse_button_callback_c"] = _MB_SIG(_mouse_button_callback)
        state["cursor_pos_callback_c"] = _CP_SIG(_cursor_pos_callback)
        raw.glfwSetMouseButtonCallback(
            addr, ctypes.cast(state["mouse_button_callback_c"], ctypes.c_void_p)
        )
        raw.glfwSetCursorPosCallback(
            addr, ctypes.cast(state["cursor_pos_callback_c"], ctypes.c_void_p)
        )
        return True

    def _restore_mouse_callbacks() -> None:
        window = state.get("mouse_window")
        if not window:
            return
        raw = _glfw_raw()
        addr = state.get("mouse_window_addr") or _win_addr(window)
        raw.glfwSetMouseButtonCallback(addr, state.get("prev_mouse_button_callback") or None)
        raw.glfwSetCursorPosCallback(addr, state.get("prev_cursor_pos_callback") or None)
        state["mouse_window"] = None
        state["mouse_window_addr"] = None
        state["prev_mouse_button_callback"] = None
        state["prev_cursor_pos_callback"] = None
        state["mouse_button_callback_c"] = None
        state["cursor_pos_callback_c"] = None
        state["dragging_target"] = False

    def _key_callback(key: int) -> None:
        ch = chr(key).lower() if 0 <= key < 256 else ""
        if ch == "r":
            if resample_planner is not None:
                state["traj"] = resample_planner()
            elif rng is not None and sampling_cfg is not None:
                state["traj"] = sample_feasible_trajectory(rng, sampling_cfg)
            else:
                _log("[viewer] resample unavailable: rng/config not provided", style="yellow")
                return
            state["t0"] = time.perf_counter()
            _print_target("[viewer resample]", state["traj"])
        elif key == 32:
            state["paused"] = not state["paused"]
            if not state["paused"]:
                state["t0"] = time.perf_counter()
            _log(f"[viewer] paused={state['paused']}", style="yellow")
        elif ch == "m":
            if state["manual_edit"]:
                state["manual_edit"] = False
                _restore_mouse_callbacks()
                _log("[viewer] mouse target edit OFF", style="yellow")
            elif _install_mouse_callbacks():
                state["manual_edit"] = True
                state["dragging_target"] = False
                state["draft_p1"] = None
                state["draft_yaw"] = None
                _log(
                    "[viewer] mouse target edit ON: left-click target, drag direction, release to optimize",
                    style="yellow",
                )

    dt = 1.0 / float(playback_fps if playback_fps > 0.0 else fps)
    _print_target("[viewer init]", state["traj"])

    with mujoco.viewer.launch_passive(model, data, key_callback=_key_callback) as viewer:
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
        frame = 0
        current = state["traj"]
        live = Live(
            _root_status_panel(
                "INIT",
                0.0,
                float(current.t[-1]),
                current.pos[0],
                np.zeros(2, dtype=np.float64),
                float(current.yaw[0]),
                0.0,
                0.0,
            ),
            console=CONSOLE,
            refresh_per_second=4,
            transient=False,
        )
        live.start()
        try:
            while viewer.is_running():
                pending = state.get("pending_manual")
                if pending is not None:
                    state["pending_manual"] = None
                    p1, target_yaw = pending
                    cfg = sampling_cfg or SamplingConfig(fps=fps)
                    state["optimizing"] = True
                    try:
                        manual_bc = _manual_boundary(p1, target_yaw)
                        _log(
                            f"[viewer manual target] optimizing from current root="
                            f"({manual_bc.p0[0]:.2f},{manual_bc.p0[1]:.2f}) "
                            f"yaw={np.rad2deg(manual_bc.yaw0):.1f}deg",
                            style="magenta",
                        )
                        if manual_planner is None:
                            state["traj"] = generate_trajectory(
                                manual_bc,
                                fps=cfg.fps,
                                model=cfg.trajectory_model,
                                cfg=cfg,
                            )
                        else:
                            state["traj"] = manual_planner(manual_bc)
                        state["t0"] = time.perf_counter()
                        state["paused"] = False
                        state["draft_p1"] = None
                        state["draft_yaw"] = None
                        _print_target("[viewer manual target]", state["traj"])
                    finally:
                        state["optimizing"] = False
                current = state["traj"]
                if state["reset_camera"]:
                    viewer.cam.distance = max(12.0, float(np.linalg.norm(current.boundary.p1)) * 0.75)
                    state["reset_camera"] = False
                elapsed = 0.0 if state["paused"] else time.perf_counter() - state["t0"]
                if loop and current.t[-1] > 1e-6:
                    cur_t = elapsed % current.t[-1]
                    playback_done = False
                else:
                    cur_t = min(elapsed, current.t[-1])
                    playback_done = elapsed >= current.t[-1]

                pos, vel, yaw = _interp_traj(current, np.asarray([cur_t], dtype=np.float64))
                yaw_rate = float(np.interp(cur_t, current.t, current.yaw_rate))
                if playback_done:
                    vel[:] = 0.0
                    yaw_rate = 0.0
                speed = float(np.linalg.norm(vel[0]))
                state["current_root_pos"] = pos[0].copy()
                state["current_root_vel"] = vel[0].copy()
                state["current_root_yaw"] = float(yaw[0])
                state["current_root_yaw_rate"] = float(yaw_rate)
                state["current_root_time"] = float(cur_t)
                data.qpos[:3] = [pos[0, 0], pos[0, 1], root_height]
                data.qpos[3:7] = _quat_from_yaw(float(yaw[0]))
                data.qpos[7:36] = 0.0
                data.qvel[:] = 0.0
                data.qvel[:2] = vel[0]
                if data.qvel.shape[0] >= 6:
                    data.qvel[5] = yaw_rate
                mujoco.mj_forward(model, data)

                draft_target = None
                if state["manual_edit"] and state["draft_p1"] is not None:
                    draft_yaw = float(state["draft_yaw"] if state["draft_yaw"] is not None else current.boundary.yaw1)
                    draft_target = (np.asarray(state["draft_p1"], dtype=np.float64), draft_yaw)
                _draw_trajectory_overlay(viewer, current, cur_t, offsets=offsets, fps=fps,
                                         draft_target=draft_target, root_radius=root_radius)
                status = "OPT" if state["optimizing"] else ("EDIT" if state["manual_edit"] else ("DONE" if playback_done else "RUN"))
                now = time.perf_counter()
                if now >= float(state["next_root_status_update"]):
                    live.update(
                        _root_status_panel(status, cur_t, float(current.t[-1]), pos[0], vel[0], float(yaw[0]), yaw_rate, speed),
                        refresh=True,
                    )
                    state["next_root_status_update"] = now + 0.5
                viewer.sync()
                if state["paused"]:
                    time.sleep(0.05)
                else:
                    time.sleep(max(0.0, dt - 0.001))
                frame += 1
                if max_frames is not None and frame >= max_frames:
                    live.stop()
                    CONSOLE.line()
                    _log("[viewer smoke exit]", style="green")
                    os._exit(0)
        finally:
            live.stop()
        CONSOLE.line()
        _log("[viewer exit]", style="green")
        os._exit(0)


def validate_mujoco_trajectory(traj: Trajectory, offsets: tuple[int, ...], fps: float,
                               root_height: float, samples: int = 16) -> dict[str, float]:
    """Validate trajectory/query generation and G1 FK without opening a viewer."""
    import mujoco

    repo = Path(__file__).resolve().parents[1]
    xml = repo / "robots/replay_unitree_description/mjcf/g1.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    times = np.linspace(0.0, traj.t[-1], max(2, int(samples)), dtype=np.float64)
    max_query_abs = 0.0
    for cur_t in times:
        pos, vel, yaw = _interp_traj(traj, np.asarray([cur_t], dtype=np.float64))
        data.qpos[:3] = [pos[0, 0], pos[0, 1], root_height]
        data.qpos[3:7] = _quat_from_yaw(float(yaw[0]))
        data.qpos[7:36] = 0.0
        data.qvel[:] = 0.0
        data.qvel[:2] = vel[0]
        if data.qvel.shape[0] >= 6:
            data.qvel[5] = float(np.interp(cur_t, traj.t, traj.yaw_rate))
        mujoco.mj_forward(model, data)
        query = sample_future_query_at(traj, cur_t, offsets=offsets, fps=fps)
        arrays = [data.qpos, data.qvel, data.xpos, query["pos"], query["heading"], query["vel"]]
        if not all(np.isfinite(a).all() for a in arrays):
            raise RuntimeError(f"non-finite value while validating trajectory at t={cur_t:.3f}")
        max_query_abs = max(max_query_abs, float(np.max(np.abs(query["pos"]))))
    speed = np.linalg.norm(traj.vel, axis=-1)
    accel = np.linalg.norm(traj.acc, axis=-1)
    return {
        "duration": float(traj.t[-1]),
        "distance": float(np.linalg.norm(traj.boundary.p1 - traj.boundary.p0)),
        "max_speed": float(np.max(speed)),
        "max_accel": float(np.max(accel)),
        "max_yaw_rate": float(np.max(np.abs(traj.yaw_rate))),
        "max_query_abs": max_query_abs,
    }


def save_npz(path: Path, trajectories: list[Trajectory], offsets: tuple[int, ...], fps: float) -> None:
    query = [sample_query_features(traj, offsets=offsets, fps=fps) for traj in trajectories]
    np.savez_compressed(
        path,
        t=np.array([traj.t for traj in trajectories], dtype=object),
        pos=np.array([traj.pos for traj in trajectories], dtype=object),
        vel=np.array([traj.vel for traj in trajectories], dtype=object),
        acc=np.array([traj.acc for traj in trajectories], dtype=object),
        yaw=np.array([traj.yaw for traj in trajectories], dtype=object),
        query_pos=np.stack([q["pos"] for q in query], axis=0),
        query_heading=np.stack([q["heading"] for q in query], axis=0),
        query_vel=np.stack([q["vel"] for q in query], axis=0),
        query_yaw=np.stack([q["yaw"] for q in query], axis=0),
        query_offsets=np.asarray(offsets, dtype=np.int64),
        fps=np.asarray(fps, dtype=np.float64),
    )


def _range_deg(values: list[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("range expects two values")
    return float(np.deg2rad(values[0])), float(np.deg2rad(values[1]))


def _load_yaml_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"trajectory config not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"trajectory config must be a mapping: {path}")
    return data


def _tuple2(value, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if len(value) != 2:
        raise ValueError(f"expected two values, got {value}")
    return float(value[0]), float(value[1])


def _int_tuple(value, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    return tuple(int(v) for v in value)


def _sampling_config_from_yaml(sampling: dict) -> SamplingConfig:
    opt = sampling.get("optimization", {}) or {}
    return SamplingConfig(
        fps=float(sampling.get("fps", 50.0)),
        trajectory_model=str(sampling.get("trajectory_model", "optimized_omni")),
        distance_range=_tuple2(sampling.get("distance_range"), (0.5, 10.0)),
        target_bearing_range=_range_deg(list(sampling.get("target_bearing_deg", [-180.0, 180.0]))),
        duration_range=_tuple2(sampling.get("duration_range"), (0.8, 4.0)),
        start_speed_range=_tuple2(sampling.get("start_speed_range"), (0.0, 0.0)),
        start_vel_yaw_range=_range_deg(list(sampling.get("start_vel_yaw_deg", [-60.0, 60.0]))),
        target_yaw_range=_range_deg(list(sampling.get("target_yaw_deg", [0.0, 0.0]))),
        target_speed_range=_tuple2(sampling.get("target_speed_range"), (0.0, 1.5)),
        target_zero_speed_prob=float(sampling.get("target_zero_speed_prob", 0.35)),
        max_speed=float(sampling.get("max_speed", 5.8)),
        max_accel=float(sampling.get("max_accel", 9.0)),
        max_yaw_rate=float(sampling.get("max_yaw_rate", 5.0)),
        lateral_offset_range=_tuple2(sampling.get("lateral_offset_range"), (0.0, 0.0)),
        max_lateral_ratio=float(sampling.get("max_lateral_ratio", 0.10)),
        max_tries=int(sampling.get("max_tries", 32)),
        opt_knots=int(opt.get("knots", 12)),
        opt_max_iter=int(opt.get("max_iter", 80)),
        opt_ftol=float(opt.get("ftol", 1e-5)),
        opt_terminal_tol=float(opt.get("terminal_tol", 0.08)),
        opt_minimize_duration=bool(opt.get("minimize_duration", True)),
        opt_duration_search_steps=int(opt.get("duration_search_steps", 3)),
        opt_forward_speed=float(opt.get("forward_speed", 5.5)),
        opt_backward_speed=float(opt.get("backward_speed", 2.3)),
        opt_lateral_speed=float(opt.get("lateral_speed", 2.6)),
        # accel/brake merged: if *_brake is omitted it defaults to *_accel, so a single
        # accel limit governs BOTH speeding up and braking (symmetric acceleration).
        opt_forward_accel=(_fa := float(opt.get("forward_accel", 6.5))),
        opt_forward_brake=float(opt.get("forward_brake", _fa)),
        opt_backward_accel=(_ba := float(opt.get("backward_accel", 3.2))),
        opt_backward_brake=float(opt.get("backward_brake", _ba)),
        opt_lateral_accel=float(opt.get("lateral_accel", 4.0)),
        opt_yaw_accel=float(opt.get("yaw_accel", 9.0)),
        opt_w_speed=float(opt.get("w_speed", 0.02)),
        opt_w_accel=float(opt.get("w_accel", 0.35)),
        opt_w_jerk=float(opt.get("w_jerk", 0.04)),
        opt_w_lateral=float(opt.get("w_lateral", 0.0)),
        opt_w_backward=float(opt.get("w_backward", 0.0)),
        opt_w_yaw_rate=float(opt.get("w_yaw_rate", 5.0)),
        opt_w_yaw_deviation=float(opt.get("w_yaw_deviation", 8.0)),
        opt_w_path=float(opt.get("w_path", 2.0)),
        opt_w_reverse_progress=float(opt.get("w_reverse_progress", 0.25)),
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample smooth target trajectories for MM query generation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="YAML file controlling sampling, query, viewer and validation parameters")
    parser.add_argument("--num", type=int, default=None, help="Override sampling.num from YAML")
    parser.add_argument("--seed", type=int, default=None, help="Override sampling.seed from YAML")
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer for one sampled trajectory")
    parser.add_argument("--validate-mujoco", action="store_true",
                        help="Validate generated trajectories through MuJoCo FK without opening a viewer")
    parser.add_argument("--save", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    raw_cfg = _load_yaml_config(args.config)
    sampling = raw_cfg.get("sampling", {})
    query_cfg = raw_cfg.get("query", {})
    viewer_cfg = raw_cfg.get("viewer", {})
    validation_cfg = raw_cfg.get("validation", {})
    output_cfg = raw_cfg.get("output", {})

    cfg = _sampling_config_from_yaml(sampling)
    num = int(args.num if args.num is not None else sampling.get("num", 8))
    seed = int(args.seed if args.seed is not None else sampling.get("seed", 0))
    offsets = _int_tuple(query_cfg.get("offsets"), DEFAULT_OFFSETS)
    rng = np.random.default_rng(seed)
    trajectories = [sample_feasible_trajectory(rng, cfg) for _ in range(max(1, num))]

    for i, traj in enumerate(trajectories):
        bc = traj.boundary
        q = sample_query_features(traj, offsets=offsets, fps=cfg.fps)
        speed0 = float(np.linalg.norm(bc.v0))
        speed1 = float(np.linalg.norm(bc.v1))
        dist = float(np.linalg.norm(bc.p1 - bc.p0))
        _log(
            f"[{i:03d}] model={cfg.trajectory_model} dist={dist:5.2f}m T={bc.duration:4.2f}s "
            f"v0={speed0:4.2f} v1={speed1:4.2f} "
            f"yaw1={np.rad2deg(bc.yaw1):7.2f}deg "
            f"lat={bc.lateral_offset:5.2f} "
            f"query_pos={np.array2string(q['pos'], precision=2, suppress_small=True)}"
        )

    do_validate = bool(args.validate_mujoco or validation_cfg.get("enabled", False))
    root_height = float(viewer_cfg.get("root_height", 0.78))
    if do_validate:
        samples = int(validation_cfg.get("samples", 16))
        for i, traj in enumerate(trajectories):
            stats = validate_mujoco_trajectory(traj, offsets=offsets, fps=cfg.fps,
                                               root_height=root_height, samples=samples)
            _log(
                f"[validate {i:03d}] duration={stats['duration']:.3f}s "
                f"distance={stats['distance']:.3f}m max_speed={stats['max_speed']:.3f} "
                f"max_accel={stats['max_accel']:.3f} max_yaw_rate={stats['max_yaw_rate']:.3f} "
                f"max_query_abs={stats['max_query_abs']:.3f}"
            )

    save_path = args.save
    if save_path is None and output_cfg.get("save"):
        save_path = Path(output_cfg["save"])
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(save_path, trajectories, offsets, cfg.fps)
        _log(f"saved {len(trajectories)} trajectories to {save_path}", style="green")

    if bool(args.viewer or viewer_cfg.get("enabled", False)):
        idx = int(viewer_cfg.get("trajectory_index", 0))
        idx = max(0, min(idx, len(trajectories) - 1))
        run_mujoco_viewer(
            trajectories[idx],
            offsets=offsets,
            fps=cfg.fps,
            playback_fps=float(viewer_cfg.get("playback_fps", cfg.fps)),
            root_height=root_height,
            loop=bool(viewer_cfg.get("loop", True)),
            max_frames=viewer_cfg.get("max_frames"),
            rng=rng,
            sampling_cfg=cfg,
            root_radius=2.0,
        )


if __name__ == "__main__":
    main()
