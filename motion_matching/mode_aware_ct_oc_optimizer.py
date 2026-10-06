"""Two-mode continuous-time optimal-control planner for strike-entry movement.

This module keeps the output contract of target_trajectory_optimizer.py:
it returns a Trajectory that can be sampled into motion-matching query features
and viewed with the same MuJoCo trajectory/query overlay.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from rich.table import Table

from target_trajectory_optimizer import (
    DEFAULT_CONFIG,
    DEFAULT_OFFSETS,
    BoundaryCondition,
    CONSOLE,
    SamplingConfig,
    Trajectory,
    _body_limit_feasible,
    _body_vel_to_world,
    _generate_omni_progress_trajectory,
    _interp_traj,
    _load_yaml_config,
    _log,
    _sampling_config_from_yaml,
    _world_vel_to_body,
    generate_trajectory,
    run_mujoco_viewer,
    sample_boundary,
    sample_query_features,
    save_npz,
    trajectory_is_feasible,
    unwrap_to_near,
    validate_mujoco_trajectory,
    wrap_angle,
)


ModeName = Literal["translation", "turn_run"]
COURT_SERVICE_LINE_X = -6.40
COURT_CENTER_Y = 0.0
COURT_FORWARD_YAW = 0.0


@dataclass(frozen=True)
class CTOCConfig:
    mode_switch_distance: float = 2.0
    knots: int = 12
    max_iter: int = 180
    ftol: float = 1e-5
    time_weight: float = 1.0
    terminal_weight: float = 25.0
    terminal_pos_tolerance: float = 0.10
    terminal_yaw_tolerance: float = 0.10
    terminal_vel_tolerance: float = 0.25
    terminal_yaw_rate_tolerance: float = 0.25
    dynamics_tolerance: float = 1e-2
    smooth_weight: float = 0.08
    jerk_weight: float = 0.01
    path_weight: float = 0.8
    reverse_progress_weight: float = 0.25
    translation_yaw_weight: float = 3.0
    translation_yaw_rate_weight: float = 0.8
    translation_sprint_weight: float = 0.5
    turn_run_yaw_weight: float = 4.0
    turn_run_forward_weight: float = 0.9
    turn_run_lateral_weight: float = 1.2
    turn_run_recover_weight: float = 3.0
    turn_phase: float = 0.25
    recover_phase: float = 0.25
    turn_run_turn_speed: float = 0.8
    turn_run_turn_progress: float = 0.35
    turn_run_corridor_width: float = 0.55
    turn_run_corridor_ratio: float = 0.12
    turn_run_recover_distance: float = 0.75
    turn_run_yaw_tolerance_deg: float = 35.0


@dataclass(frozen=True)
class PlanCandidate:
    mode: str
    trajectory: Trajectory | None
    success: bool
    duration: float
    score: float
    terminal_error: float
    max_speed: float
    max_accel: float
    max_yaw_rate: float
    message: str
    fallback_used: bool = False


@dataclass(frozen=True)
class ModeAwarePlan:
    trajectory: Trajectory
    selected_mode: str
    candidates: tuple[PlanCandidate, ...]
    fallback_used: bool


def _ctoc_config_from_yaml(raw_cfg: dict) -> CTOCConfig:
    raw = raw_cfg.get("ct_oc", {}) or {}
    return CTOCConfig(
        mode_switch_distance=float(raw.get("mode_switch_distance", 2.0)),
        knots=int(raw.get("knots", 12)),
        max_iter=int(raw.get("max_iter", 180)),
        ftol=float(raw.get("ftol", 1e-5)),
        time_weight=float(raw.get("time_weight", 1.0)),
        terminal_weight=float(raw.get("terminal_weight", 25.0)),
        terminal_pos_tolerance=float(raw.get("terminal_pos_tolerance", 0.10)),
        terminal_yaw_tolerance=float(raw.get("terminal_yaw_tolerance", 0.10)),
        terminal_vel_tolerance=float(raw.get("terminal_vel_tolerance", 0.25)),
        terminal_yaw_rate_tolerance=float(raw.get("terminal_yaw_rate_tolerance", 0.25)),
        dynamics_tolerance=float(raw.get("dynamics_tolerance", 1e-2)),
        smooth_weight=float(raw.get("smooth_weight", 0.08)),
        jerk_weight=float(raw.get("jerk_weight", 0.01)),
        path_weight=float(raw.get("path_weight", 0.8)),
        reverse_progress_weight=float(raw.get("reverse_progress_weight", 0.25)),
        translation_yaw_weight=float(raw.get("translation_yaw_weight", 3.0)),
        translation_yaw_rate_weight=float(raw.get("translation_yaw_rate_weight", 0.8)),
        translation_sprint_weight=float(raw.get("translation_sprint_weight", 0.5)),
        turn_run_yaw_weight=float(raw.get("turn_run_yaw_weight", 4.0)),
        turn_run_forward_weight=float(raw.get("turn_run_forward_weight", 0.9)),
        turn_run_lateral_weight=float(raw.get("turn_run_lateral_weight", 1.2)),
        turn_run_recover_weight=float(raw.get("turn_run_recover_weight", 3.0)),
        turn_phase=float(raw.get("turn_phase", 0.25)),
        recover_phase=float(raw.get("recover_phase", 0.25)),
        turn_run_turn_speed=float(raw.get("turn_run_turn_speed", 0.8)),
        turn_run_turn_progress=float(raw.get("turn_run_turn_progress", 0.35)),
        turn_run_corridor_width=float(raw.get("turn_run_corridor_width", 0.55)),
        turn_run_corridor_ratio=float(raw.get("turn_run_corridor_ratio", 0.12)),
        turn_run_recover_distance=float(raw.get("turn_run_recover_distance", 0.75)),
        turn_run_yaw_tolerance_deg=float(raw.get("turn_run_yaw_tolerance_deg", 35.0)),
    )


def _boundary_states(boundary: BoundaryCondition) -> tuple[np.ndarray, np.ndarray, float]:
    yaw0 = float(boundary.yaw0)
    yaw1 = unwrap_to_near(boundary.yaw1, yaw0)
    start_body_vel = _world_vel_to_body(boundary.v0, yaw0)
    end_body_vel = _world_vel_to_body(boundary.v1, yaw1)
    start = np.array(
        [boundary.p0[0], boundary.p0[1], yaw0, start_body_vel[0], start_body_vel[1], boundary.yaw_rate0],
        dtype=np.float64,
    )
    target = np.array(
        [boundary.p1[0], boundary.p1[1], yaw1, end_body_vel[0], end_body_vel[1], boundary.yaw_rate1],
        dtype=np.float64,
    )
    return start, target, yaw1


def _state_dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    world_vel = _body_vel_to_world(x[..., 3:5], x[..., 2])
    return np.concatenate(
        [
            world_vel,
            x[..., 5:6],
            u,
        ],
        axis=-1,
    )


def _run_direction(boundary: BoundaryCondition) -> float:
    delta = np.asarray(boundary.p1 - boundary.p0, dtype=np.float64)
    if float(np.linalg.norm(delta)) < 1e-6:
        return float(boundary.yaw0)
    return float(np.arctan2(delta[1], delta[0]))


def _target_axes(boundary: BoundaryCondition) -> tuple[float, np.ndarray, np.ndarray, float]:
    delta = np.asarray(boundary.p1 - boundary.p0, dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        direction = np.array([np.cos(boundary.yaw0), np.sin(boundary.yaw0)], dtype=np.float64)
        dist = 1e-6
    else:
        direction = delta / dist
    left = np.array([-direction[1], direction[0]], dtype=np.float64)
    return dist, direction, left, max(dist, 1.0)


def _duration_guess(boundary: BoundaryCondition, cfg: SamplingConfig, mode: ModeName, deadline: float | None) -> float:
    dist = float(np.linalg.norm(boundary.p1 - boundary.p0))
    lo, hi = cfg.duration_range
    if deadline is not None:
        hi = min(float(hi), float(deadline))
    hi = max(float(lo), float(hi))
    yaw1 = unwrap_to_near(boundary.yaw1, boundary.yaw0)
    if mode == "translation":
        speed = max(0.75 * max(cfg.opt_lateral_speed, cfg.opt_backward_speed, 0.5 * cfg.opt_forward_speed), 0.5)
        yaw_time = abs(float(wrap_angle(yaw1 - boundary.yaw0))) / max(0.45 * cfg.max_yaw_rate, 1e-6)
        guess = dist / speed + yaw_time + 0.25
    else:
        run_yaw = unwrap_to_near(_run_direction(boundary), boundary.yaw0)
        target_from_run = unwrap_to_near(boundary.yaw1, run_yaw)
        turn_time = abs(run_yaw - boundary.yaw0) / max(0.75 * cfg.max_yaw_rate, 1e-6)
        recover_time = abs(target_from_run - run_yaw) / max(0.75 * cfg.max_yaw_rate, 1e-6)
        run_time = dist / max(0.82 * cfg.opt_forward_speed, 0.5)
        guess = turn_time + run_time + recover_time + 0.4
    return float(np.clip(max(float(lo), guess), float(lo), hi))


def _initial_guess(
    boundary: BoundaryCondition,
    cfg: SamplingConfig,
    oc_cfg: CTOCConfig,
    mode: ModeName,
    duration: float,
    k_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    start, target, yaw1 = _boundary_states(boundary)
    guess_bc = replace(boundary, duration=float(duration))
    base = _generate_omni_progress_trajectory(guess_bc, fps=max(cfg.fps, k_count / max(duration, 1e-6)))
    knot_t = np.linspace(0.0, duration, k_count, dtype=np.float64)
    pos, world_vel, base_yaw = _interp_traj(base, knot_t)
    tau = knot_t / max(duration, 1e-9)

    if mode == "translation":
        s = np.clip((tau - 0.65) / 0.35, 0.0, 1.0)
        s = s * s * (3.0 - 2.0 * s)
        yaw = boundary.yaw0 + s * (yaw1 - boundary.yaw0)
    else:
        run_yaw = unwrap_to_near(_run_direction(boundary), boundary.yaw0)
        yaw1_from_run = unwrap_to_near(boundary.yaw1, run_yaw)
        turn_end = float(np.clip(oc_cfg.turn_phase, 0.1, 0.45))
        recover_start = float(np.clip(1.0 - oc_cfg.recover_phase, turn_end + 0.1, 0.95))
        yaw = np.empty_like(tau)
        for i, x in enumerate(tau):
            if x <= turn_end:
                s = x / turn_end
                s = s * s * (3.0 - 2.0 * s)
                yaw[i] = boundary.yaw0 + s * (run_yaw - boundary.yaw0)
            elif x < recover_start:
                yaw[i] = run_yaw
            else:
                s = (x - recover_start) / max(1e-9, 1.0 - recover_start)
                s = s * s * (3.0 - 2.0 * s)
                yaw[i] = run_yaw + s * (yaw1_from_run - run_yaw)
        yaw1 = yaw1_from_run

    body_vel = _world_vel_to_body(world_vel, yaw)
    body_vel[:, 0] = np.clip(body_vel[:, 0], -cfg.opt_backward_speed, cfg.opt_forward_speed)
    body_vel[:, 1] = np.clip(body_vel[:, 1], -cfg.opt_lateral_speed, cfg.opt_lateral_speed)
    yaw_rate = np.gradient(yaw, knot_t, edge_order=1)
    yaw_rate = np.clip(yaw_rate, -cfg.max_yaw_rate, cfg.max_yaw_rate)
    states = np.column_stack([pos, yaw, body_vel, yaw_rate])
    states[0] = start
    states[-1] = target
    states[-1, 2] = yaw1

    controls = np.gradient(states[:, 3:6], knot_t, axis=0, edge_order=1)
    af_lo = -max(cfg.opt_forward_brake, cfg.opt_backward_accel)
    af_hi = max(cfg.opt_forward_accel, cfg.opt_backward_brake)
    controls[:, 0] = np.clip(controls[:, 0], af_lo, af_hi)
    controls[:, 1] = np.clip(controls[:, 1], -cfg.opt_lateral_accel, cfg.opt_lateral_accel)
    controls[:, 2] = np.clip(controls[:, 2], -cfg.opt_yaw_accel, cfg.opt_yaw_accel)
    return states, controls


def _trajectory_stats(traj: Trajectory) -> tuple[float, float, float]:
    speed = float(np.max(np.linalg.norm(traj.vel, axis=-1))) if len(traj.t) else 0.0
    accel = float(np.max(np.linalg.norm(traj.acc, axis=-1))) if len(traj.t) else 0.0
    yaw_rate = float(np.max(np.abs(traj.yaw_rate))) if len(traj.t) else 0.0
    return speed, accel, yaw_rate


def _trajectory_from_ctoc_knots(
    boundary: BoundaryCondition,
    knot_t: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    fps: float,
) -> Trajectory:
    duration = float(knot_t[-1])
    n = max(2, int(np.ceil(duration * fps)) + 1)
    t = np.linspace(0.0, duration, n, dtype=np.float64)
    pos = np.stack([np.interp(t, knot_t, states[:, i]) for i in range(2)], axis=-1)
    yaw = np.interp(t, knot_t, states[:, 2])
    body_vel = np.stack([np.interp(t, knot_t, states[:, 3 + i]) for i in range(2)], axis=-1)
    yaw_rate = np.interp(t, knot_t, states[:, 5])
    body_acc = np.stack([np.interp(t, knot_t, controls[:, i]) for i in range(2)], axis=-1)
    yaw_acc = np.interp(t, knot_t, controls[:, 2])

    vel = _body_vel_to_world(body_vel, yaw)
    # d(R(yaw) v_body)/dt = R(yaw) a_body + yaw_rate * R(yaw) J v_body.
    coriolis_body = np.stack([-body_vel[:, 1], body_vel[:, 0]], axis=-1) * yaw_rate[:, None]
    acc = _body_vel_to_world(body_acc + coriolis_body, yaw)
    return Trajectory(
        t=t,
        pos=pos,
        vel=vel,
        acc=acc,
        yaw=wrap_angle(yaw),
        yaw_rate=yaw_rate,
        yaw_acc=yaw_acc,
        boundary=boundary,
    )


def _ctoc_trajectory_feasible(traj: Trajectory, cfg: SamplingConfig, slack: float = 1.03) -> bool:
    speed, accel, yaw_rate = _trajectory_stats(traj)
    finite_ok = all(np.isfinite(x).all() for x in (traj.pos, traj.vel, traj.acc, traj.yaw, traj.yaw_rate, traj.yaw_acc))
    bounds_ok = (
        speed <= cfg.max_speed * slack
        and accel <= cfg.max_accel * slack
        and yaw_rate <= cfg.max_yaw_rate * slack
    )
    return bool(finite_ok and bounds_ok and _body_limit_feasible(traj, cfg, slack=1.12))


def _phase_summary(mode: str, duration: float, oc_cfg: CTOCConfig) -> str:
    if mode == "translation":
        translate_t = 0.65 * duration
        align_t = duration - translate_t
        return f"translate={translate_t:.2f}s align={align_t:.2f}s"
    if mode == "turn_run":
        turn_t = oc_cfg.turn_phase * duration
        recover_t = oc_cfg.recover_phase * duration
        run_t = max(0.0, duration - turn_t - recover_t)
        return f"turn={turn_t:.2f}s run={run_t:.2f}s recover={recover_t:.2f}s"
    return "-"


def _terminal_errors(states: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    pos_err = float(np.linalg.norm(states[-1, :2] - target[:2]))
    yaw_err = float(abs(wrap_angle(states[-1, 2] - target[2])))
    vel_err = float(np.linalg.norm(states[-1, 3:5] - target[3:5]))
    yaw_rate_err = float(abs(states[-1, 5] - target[5]))
    return pos_err, yaw_err, vel_err, yaw_rate_err


def _terminal_error_ratios(states: np.ndarray, target: np.ndarray, oc_cfg: CTOCConfig) -> tuple[float, float, float, float]:
    pos_err, yaw_err, vel_err, yaw_rate_err = _terminal_errors(states, target)
    return (
        pos_err / max(float(oc_cfg.terminal_pos_tolerance), 1e-6),
        yaw_err / max(float(oc_cfg.terminal_yaw_tolerance), 1e-6),
        vel_err / max(float(oc_cfg.terminal_vel_tolerance), 1e-6),
        yaw_rate_err / max(float(oc_cfg.terminal_yaw_rate_tolerance), 1e-6),
    )


def _terminal_cost(states: np.ndarray, target: np.ndarray, oc_cfg: CTOCConfig) -> float:
    ratios = np.asarray(_terminal_error_ratios(states, target, oc_cfg), dtype=np.float64)
    return float(np.dot(ratios, ratios))


def _terminal_success(states: np.ndarray, target: np.ndarray, oc_cfg: CTOCConfig) -> bool:
    return bool(max(_terminal_error_ratios(states, target, oc_cfg)) <= 1.0)


def solve_mode_ct_oc(
    boundary: BoundaryCondition,
    cfg: SamplingConfig,
    oc_cfg: CTOCConfig,
    mode: ModeName,
    deadline: float | None = None,
) -> PlanCandidate:
    try:
        from scipy.optimize import minimize
    except Exception as exc:
        return PlanCandidate(mode, None, False, 0.0, np.inf, np.inf, 0.0, 0.0, 0.0, f"scipy unavailable: {exc}")

    start, target, yaw1 = _boundary_states(boundary)
    k_count = max(8, int(oc_cfg.knots))
    lo_t, hi_t = cfg.duration_range
    if deadline is not None:
        hi_t = min(float(hi_t), float(deadline))
    lo_t = max(1e-3, float(lo_t))
    hi_t = max(lo_t, float(hi_t))
    duration0 = _duration_guess(boundary, cfg, mode, deadline)
    x0, u0 = _initial_guess(boundary, cfg, oc_cfg, mode, duration0, k_count)

    af_lo = -max(cfg.opt_forward_brake, cfg.opt_backward_accel)
    af_hi = max(cfg.opt_forward_accel, cfg.opt_backward_brake)
    max_extent = max(15.0, 2.0 * float(np.linalg.norm(boundary.p1 - boundary.p0)) + 5.0)
    yaw_pad = 2.0 * np.pi
    yaw_min = min(boundary.yaw0, yaw1, _run_direction(boundary)) - yaw_pad
    yaw_max = max(boundary.yaw0, yaw1, _run_direction(boundary)) + yaw_pad
    bounds: list[tuple[float, float]] = [(lo_t, hi_t)]
    for _ in range(k_count):
        bounds.extend(
            [
                (-max_extent, max_extent),
                (-max_extent, max_extent),
                (yaw_min, yaw_max),
                (-cfg.opt_backward_speed, cfg.opt_forward_speed),
                (-cfg.opt_lateral_speed, cfg.opt_lateral_speed),
                (-cfg.max_yaw_rate, cfg.max_yaw_rate),
            ]
        )
    for _ in range(k_count):
        bounds.extend(
            [
                (af_lo, af_hi),
                (-cfg.opt_lateral_accel, cfg.opt_lateral_accel),
                (-cfg.opt_yaw_accel, cfg.opt_yaw_accel),
            ]
        )

    def pack(duration: float, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        return np.concatenate([[duration], states.reshape(-1), controls.reshape(-1)])

    def unpack(z: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        duration = float(z[0])
        off = 1
        states = np.asarray(z[off: off + k_count * 6], dtype=np.float64).reshape(k_count, 6)
        off += k_count * 6
        controls = np.asarray(z[off: off + k_count * 3], dtype=np.float64).reshape(k_count, 3)
        return duration, states, controls

    dist, target_dir, target_left, dist_scale = _target_axes(boundary)
    run_yaw = unwrap_to_near(_run_direction(boundary), boundary.yaw0)
    target_yaw_from_run = unwrap_to_near(boundary.yaw1, run_yaw)
    tau = np.linspace(0.0, 1.0, k_count, dtype=np.float64)
    turn_end = float(np.clip(oc_cfg.turn_phase, 0.1, 0.45))
    recover_start = float(np.clip(1.0 - oc_cfg.recover_phase, turn_end + 0.1, 0.95))
    accel_scale = np.array(
        [
            max(cfg.opt_forward_accel, cfg.opt_forward_brake, cfg.opt_backward_accel, cfg.opt_backward_brake, 1e-6),
            max(cfg.opt_lateral_accel, 1e-6),
            max(cfg.opt_yaw_accel, 1e-6),
        ],
        dtype=np.float64,
    )

    def equality(z: np.ndarray) -> np.ndarray:
        duration, states, controls = unpack(z)
        dt = duration / float(k_count - 1)
        f0 = _state_dynamics(states[:-1], controls[:-1])
        f1 = _state_dynamics(states[1:], controls[1:])
        collocation = states[1:] - states[:-1] - 0.5 * dt * (f0 + f1)
        return np.concatenate([states[0] - start, collocation.reshape(-1)])

    def inequality(z: np.ndarray) -> np.ndarray:
        _, states, controls = unpack(z)
        vf = states[:, 3]
        body_speed = np.linalg.norm(states[:, 3:5], axis=-1)
        body_accel = np.linalg.norm(controls[:, :2], axis=-1)
        pos_forward_limit = np.where(vf >= 0.0, cfg.opt_forward_accel, cfg.opt_backward_brake)
        values = [
            cfg.max_speed - body_speed,
            cfg.max_accel - body_accel,
            pos_forward_limit - controls[:, 0],
        ]
        # A zero-velocity endpoint after forward motion is still braking, not backward launch.
        forward_context = vf > 1e-4
        if vf.size > 1:
            forward_context[:-1] |= vf[1:] > 1e-4
            forward_context[1:] |= vf[:-1] > 1e-4
        neg_forward_limit = np.where(forward_context, cfg.opt_forward_brake, cfg.opt_backward_accel)
        values.append(neg_forward_limit + controls[:, 0])
        if mode == "translation":
            translation_speed_limit = max(cfg.opt_lateral_speed, cfg.opt_backward_speed, 0.60 * cfg.opt_forward_speed)
            values.append(translation_speed_limit - body_speed)
        else:
            rel = states[:, :2] - boundary.p0
            progress = rel @ target_dir
            cross_track = rel @ target_left
            early = tau <= turn_end
            middle = (tau > turn_end) & (tau < recover_start)
            late = tau >= recover_start
            corridor = max(0.15, min(float(oc_cfg.turn_run_corridor_width), float(oc_cfg.turn_run_corridor_ratio) * dist))
            recover_distance = max(0.15, float(oc_cfg.turn_run_recover_distance))
            yaw_tol = np.deg2rad(max(1.0, float(oc_cfg.turn_run_yaw_tolerance_deg)))
            if np.any(early):
                values.append(float(oc_cfg.turn_run_turn_speed) - body_speed[early])
                values.append(float(oc_cfg.turn_run_turn_progress) - progress[early])
            if np.any(middle):
                values.append(corridor - np.abs(cross_track[middle]))
                values.append(yaw_tol - np.abs(wrap_angle(states[middle, 2] - run_yaw)))
                values.append(0.35 * cfg.opt_lateral_speed - np.abs(states[middle, 4]))
            if np.any(late):
                values.append(progress[late] - (dist - recover_distance))
                values.append(corridor - np.abs(cross_track[late]))
        return np.concatenate(values)

    def path_cost(states: np.ndarray) -> float:
        if dist <= 1e-6:
            return 0.0
        rel = states[:, :2] - boundary.p0
        cross_track = rel @ target_left
        progress = rel @ target_dir
        reverse = np.minimum(np.diff(progress), 0.0)
        return (
            oc_cfg.path_weight * float(np.mean((cross_track / dist_scale) ** 2))
            + oc_cfg.reverse_progress_weight * float(np.mean(reverse ** 2))
        )

    def translation_mode_cost(states: np.ndarray) -> float:
        s = np.clip((tau - 0.65) / 0.35, 0.0, 1.0)
        s = s * s * (3.0 - 2.0 * s)
        yaw_ref = boundary.yaw0 + s * (target[2] - boundary.yaw0)
        high_forward = np.maximum(np.abs(states[:, 3]) - 0.65 * cfg.opt_forward_speed, 0.0)
        return (
            oc_cfg.translation_yaw_weight * float(np.mean((wrap_angle(states[:, 2] - yaw_ref) / np.pi) ** 2))
            + oc_cfg.translation_yaw_rate_weight * float(np.mean((states[:, 5] / max(cfg.max_yaw_rate, 1e-6)) ** 2))
            + oc_cfg.translation_sprint_weight * float(np.mean((high_forward / max(cfg.opt_forward_speed, 1e-6)) ** 2))
            + path_cost(states)
        )

    def turn_run_mode_cost(states: np.ndarray) -> float:
        early = tau <= turn_end
        middle = (tau > turn_end) & (tau < recover_start)
        late = tau >= recover_start
        cost = path_cost(states)
        if np.any(early):
            s = tau[early] / max(turn_end, 1e-9)
            yaw_ref = boundary.yaw0 + s * (run_yaw - boundary.yaw0)
            cost += oc_cfg.turn_run_yaw_weight * float(np.mean((wrap_angle(states[early, 2] - yaw_ref) / np.pi) ** 2))
        if np.any(middle):
            cost += oc_cfg.turn_run_yaw_weight * float(np.mean((wrap_angle(states[middle, 2] - run_yaw) / np.pi) ** 2))
            cost += oc_cfg.turn_run_forward_weight * float(
                np.mean(((states[middle, 3] - 0.92 * cfg.opt_forward_speed) / max(cfg.opt_forward_speed, 1e-6)) ** 2)
            )
            cost += oc_cfg.turn_run_lateral_weight * float(
                np.mean((states[middle, 4] / max(cfg.opt_lateral_speed, 1e-6)) ** 2)
            )
            cost += 0.25 * float(np.mean((states[middle, 5] / max(cfg.max_yaw_rate, 1e-6)) ** 2))
        if np.any(late):
            s = (tau[late] - recover_start) / max(1.0 - recover_start, 1e-9)
            yaw_ref = run_yaw + s * (target_yaw_from_run - run_yaw)
            vel_ref = target[3:5]
            cost += oc_cfg.turn_run_recover_weight * float(np.mean((wrap_angle(states[late, 2] - yaw_ref) / np.pi) ** 2))
            cost += 0.5 * oc_cfg.turn_run_recover_weight * float(
                np.mean(((states[late, 3:5] - vel_ref) / max(cfg.opt_forward_speed, 1e-6)) ** 2)
            )
        return cost

    def objective(z: np.ndarray) -> float:
        duration, states, controls = unpack(z)
        dt = duration / float(k_count - 1)
        terminal = _terminal_cost(states, target, oc_cfg)
        smooth = float(np.mean((controls / accel_scale) ** 2))
        jerk = 0.0
        if k_count > 2:
            d_controls = np.diff(controls, axis=0) / max(dt, 1e-9)
            jerk = float(np.mean((d_controls / accel_scale) ** 2))
        mode_cost = translation_mode_cost(states) if mode == "translation" else turn_run_mode_cost(states)
        return float(
            oc_cfg.time_weight * duration
            + oc_cfg.terminal_weight * terminal
            + oc_cfg.smooth_weight * smooth
            + oc_cfg.jerk_weight * jerk
            + mode_cost
        )

    z0 = pack(duration0, x0, u0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Values in x were outside bounds*", category=RuntimeWarning)
        result = minimize(
            objective,
            z0,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "eq", "fun": equality},
                {"type": "ineq", "fun": inequality},
            ],
            options={"maxiter": int(oc_cfg.max_iter), "ftol": float(oc_cfg.ftol), "disp": False},
        )

    z = result.x if np.isfinite(result.x).all() else z0
    duration, states, controls = unpack(z)
    knot_t = np.linspace(0.0, duration, k_count, dtype=np.float64)
    solved_bc = replace(boundary, duration=float(duration))
    traj = _trajectory_from_ctoc_knots(solved_bc, knot_t, states, controls, fps=cfg.fps)
    eq_values = equality(z)
    dynamics_residual = float(np.max(np.abs(eq_values))) if eq_values.size else 0.0
    min_margin = float(np.min(inequality(z)))
    terminal_ratios = _terminal_error_ratios(states, target, oc_cfg)
    term = float(max(terminal_ratios))
    pos_err, yaw_err, vel_err, yaw_rate_err = _terminal_errors(states, target)
    terminal_ok = _terminal_success(states, target, oc_cfg)
    dynamics_ok = dynamics_residual <= float(oc_cfg.dynamics_tolerance)
    feasible = (
        terminal_ok
        and dynamics_ok
        and min_margin >= -1e-3
        and _ctoc_trajectory_feasible(traj, cfg)
    )
    max_speed, max_accel, max_yaw_rate = _trajectory_stats(traj)
    return PlanCandidate(
        mode=mode,
        trajectory=traj,
        success=feasible,
        duration=float(duration),
        score=float(objective(z)),
        terminal_error=float(term),
        max_speed=max_speed,
        max_accel=max_accel,
        max_yaw_rate=max_yaw_rate,
        message=(
            f"{result.message}; solver_success={bool(result.success)} "
            f"dyn_res={dynamics_residual:.2e} min_margin={min_margin:.2e}; "
            f"terminal pos={pos_err:.3f}m yaw={yaw_err:.3f}rad "
            f"vel={vel_err:.3f}m/s yaw_rate={yaw_rate_err:.3f}rad/s "
            f"ratio={term:.2f}"
        ),
    )


def plan_mode_aware_ct_oc(
    boundary: BoundaryCondition,
    cfg: SamplingConfig,
    oc_cfg: CTOCConfig,
    deadline: float | None = None,
) -> ModeAwarePlan:
    dist = float(np.linalg.norm(boundary.p1 - boundary.p0))
    threshold = float(oc_cfg.mode_switch_distance)
    requested_mode: ModeName = "translation" if dist <= threshold else "turn_run"
    candidates = (solve_mode_ct_oc(boundary, cfg, oc_cfg, requested_mode, deadline=deadline),)
    feasible = [c for c in candidates if c.success and c.trajectory is not None]
    if deadline is not None:
        feasible_in_time = [c for c in feasible if c.duration <= deadline + 1e-6]
        if feasible_in_time:
            feasible = feasible_in_time
    if feasible:
        selected = min(feasible, key=lambda c: (c.duration, c.score))
        assert selected.trajectory is not None
        return ModeAwarePlan(selected.trajectory, selected.mode, candidates, fallback_used=False)

    fallback_cfg = replace(cfg, trajectory_model="optimized_omni")
    fallback = generate_trajectory(boundary, fps=cfg.fps, model="optimized_omni", cfg=fallback_cfg)
    max_speed, max_accel, max_yaw_rate = _trajectory_stats(fallback)
    fallback_candidate = PlanCandidate(
        mode="fallback_optimized_omni",
        trajectory=fallback,
        success=trajectory_is_feasible(fallback, fallback_cfg),
        duration=float(fallback.t[-1]),
        score=np.inf,
        terminal_error=0.0,
        max_speed=max_speed,
        max_accel=max_accel,
        max_yaw_rate=max_yaw_rate,
        message=(
            f"{requested_mode} CT-OC failed for dist={dist:.2f}m "
            f"(threshold={threshold:.2f}m); used current optimized_omni fallback"
        ),
        fallback_used=True,
    )
    return ModeAwarePlan(fallback, fallback_candidate.mode, candidates + (fallback_candidate,), fallback_used=True)


def _print_plan_summary(plan: ModeAwarePlan, offsets: tuple[int, ...], fps: float, oc_cfg: CTOCConfig) -> None:
    table = Table(title="Mode-Aware CT-OC Candidates")
    table.add_column("mode", no_wrap=True)
    table.add_column("ok", justify="center", no_wrap=True)
    table.add_column("T", justify="right", no_wrap=True)
    table.add_column("J", justify="right", no_wrap=True)
    table.add_column("terminal", justify="right", no_wrap=True)
    table.add_column("vmax", justify="right", no_wrap=True)
    table.add_column("amax", justify="right", no_wrap=True)
    table.add_column("yawrate", justify="right", no_wrap=True)
    for c in plan.candidates:
        table.add_row(
            c.mode,
            "yes" if c.success else "no",
            f"{c.duration:.3f}",
            "inf" if not np.isfinite(c.score) else f"{c.score:.3f}",
            f"{c.terminal_error:.3g}",
            f"{c.max_speed:.3f}",
            f"{c.max_accel:.3f}",
            f"{c.max_yaw_rate:.3f}",
        )
    CONSOLE.print(table)
    for c in plan.candidates:
        if not c.success:
            _log(f"[candidate {c.mode}] {c.message}", style="yellow")
    q = sample_query_features(plan.trajectory, offsets=offsets, fps=fps)
    _log(
        f"[selected] mode={plan.selected_mode} fallback={plan.fallback_used} "
        f"T={plan.trajectory.t[-1]:.3f}s phase=({_phase_summary(plan.selected_mode, float(plan.trajectory.t[-1]), oc_cfg)}) "
        f"query_pos={np.array2string(q['pos'], precision=2, suppress_small=True)}",
        style="green" if not plan.fallback_used else "yellow",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-mode continuous-time OC strike-entry trajectory planner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deadline", type=float, default=None, help="Optional ball deadline / available time in seconds")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--validate-mujoco", action="store_true")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-yaw-deg", type=float, default=None)
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument("--start-x", type=float, default=COURT_SERVICE_LINE_X,
                        help="World X of current root state; default is own service-line center")
    parser.add_argument("--start-y", type=float, default=COURT_CENTER_Y,
                        help="World Y of current root state; default is court center line")
    parser.add_argument("--start-yaw-deg", type=float, default=np.rad2deg(COURT_FORWARD_YAW),
                        help="Current root yaw in world frame; 0 deg faces opponent court along +X")
    return parser


def _start_pose_from_args(args: argparse.Namespace) -> tuple[np.ndarray, float]:
    return (
        np.array([float(args.start_x), float(args.start_y)], dtype=np.float64),
        float(np.deg2rad(args.start_yaw_deg)),
    )


def _transform_sampled_boundary(boundary: BoundaryCondition, start_pos: np.ndarray, start_yaw: float) -> BoundaryCondition:
    c, s = np.cos(start_yaw), np.sin(start_yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    rel_p1 = np.asarray(boundary.p1 - boundary.p0, dtype=np.float64)
    return BoundaryCondition(
        p0=np.asarray(start_pos, dtype=np.float64),
        yaw0=float(start_yaw),
        v0=rot @ np.asarray(boundary.v0, dtype=np.float64),
        yaw_rate0=float(boundary.yaw_rate0),
        p1=np.asarray(start_pos, dtype=np.float64) + rot @ rel_p1,
        yaw1=float(wrap_angle(start_yaw + boundary.yaw1)),
        v1=rot @ np.asarray(boundary.v1, dtype=np.float64),
        yaw_rate1=float(boundary.yaw_rate1),
        duration=float(boundary.duration),
        lateral_offset=float(boundary.lateral_offset),
    )


def _manual_boundary_from_args(
    args: argparse.Namespace,
    cfg: SamplingConfig,
    start_pos: np.ndarray,
    start_yaw: float,
) -> BoundaryCondition | None:
    if args.target_x is None and args.target_y is None and args.target_yaw_deg is None and args.target_speed is None:
        return None
    p1 = np.array([float(args.target_x or 0.0), float(args.target_y or 0.0)], dtype=np.float64)
    yaw1 = np.deg2rad(float(args.target_yaw_deg or 0.0))
    speed = float(args.target_speed if args.target_speed is not None else 0.0)
    v1 = speed * np.array([np.cos(yaw1), np.sin(yaw1)], dtype=np.float64)
    return BoundaryCondition(
        p0=np.asarray(start_pos, dtype=np.float64),
        yaw0=float(start_yaw),
        v0=np.zeros(2, dtype=np.float64),
        yaw_rate0=0.0,
        p1=p1,
        yaw1=float(wrap_angle(yaw1)),
        v1=v1,
        yaw_rate1=0.0,
        duration=float(cfg.duration_range[1]),
        lateral_offset=0.0,
    )


def main() -> None:
    args = build_argparser().parse_args()
    raw_cfg = _load_yaml_config(args.config)
    sampling = raw_cfg.get("sampling", {})
    query_cfg = raw_cfg.get("query", {})
    viewer_cfg = raw_cfg.get("viewer", {})
    validation_cfg = raw_cfg.get("validation", {})
    output_cfg = raw_cfg.get("output", {})

    cfg = _sampling_config_from_yaml(sampling)
    oc_cfg = _ctoc_config_from_yaml(raw_cfg)
    num = int(args.num if args.num is not None else sampling.get("num", 8))
    seed = int(args.seed if args.seed is not None else sampling.get("seed", 0))
    offsets = tuple(int(v) for v in query_cfg.get("offsets", DEFAULT_OFFSETS))
    rng = np.random.default_rng(seed)
    start_pos, start_yaw = _start_pose_from_args(args)
    _log(
        f"[start] service-line center root=({start_pos[0]:.2f},{start_pos[1]:.2f}) "
        f"yaw={np.rad2deg(start_yaw):.1f}deg; +X points to opponent court",
        style="cyan",
    )

    manual_bc = _manual_boundary_from_args(args, cfg, start_pos=start_pos, start_yaw=start_yaw)
    boundaries = (
        [manual_bc]
        if manual_bc is not None
        else [_transform_sampled_boundary(sample_boundary(rng, cfg), start_pos, start_yaw) for _ in range(max(1, num))]
    )
    plans = [plan_mode_aware_ct_oc(bc, cfg, oc_cfg, deadline=args.deadline) for bc in boundaries]

    for i, plan in enumerate(plans):
        bc = plan.trajectory.boundary
        dist = float(np.linalg.norm(bc.p1 - bc.p0))
        threshold = float(oc_cfg.mode_switch_distance)
        requested_mode = "translation" if dist <= threshold else "turn_run"
        _log(
            f"[{i:03d}] start=({bc.p0[0]:.2f},{bc.p0[1]:.2f}) "
            f"target=({bc.p1[0]:.2f},{bc.p1[1]:.2f}) "
            f"start_target_dist={dist:.2f}m threshold={threshold:.2f}m "
            f"requested_mode={requested_mode} yaw={np.rad2deg(bc.yaw1):.1f}deg"
        )
        _print_plan_summary(plan, offsets=offsets, fps=cfg.fps, oc_cfg=oc_cfg)

    do_validate = bool(args.validate_mujoco or validation_cfg.get("enabled", False))
    root_height = float(viewer_cfg.get("root_height", 0.78))
    if do_validate:
        samples = int(validation_cfg.get("samples", 16))
        for i, plan in enumerate(plans):
            stats = validate_mujoco_trajectory(plan.trajectory, offsets=offsets, fps=cfg.fps,
                                               root_height=root_height, samples=samples)
            _log(
                f"[validate {i:03d}] duration={stats['duration']:.3f}s distance={stats['distance']:.3f}m "
                f"max_speed={stats['max_speed']:.3f} max_accel={stats['max_accel']:.3f} "
                f"max_yaw_rate={stats['max_yaw_rate']:.3f} max_query_abs={stats['max_query_abs']:.3f}"
            )

    save_path = args.save
    if save_path is None and output_cfg.get("save"):
        save_path = Path(output_cfg["save"])
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(save_path, [p.trajectory for p in plans], offsets, cfg.fps)
        _log(f"saved {len(plans)} trajectories to {save_path}", style="green")

    if bool(args.viewer or viewer_cfg.get("enabled", False)):
        idx = int(viewer_cfg.get("trajectory_index", 0))
        idx = max(0, min(idx, len(plans) - 1))

        def manual_planner(boundary: BoundaryCondition) -> Trajectory:
            dist = float(np.linalg.norm(boundary.p1 - boundary.p0))
            threshold = float(oc_cfg.mode_switch_distance)
            requested_mode = "translation" if dist <= threshold else "turn_run"
            _log(
                f"[manual target] start_target_dist={dist:.2f}m threshold={threshold:.2f}m "
                f"requested_mode={requested_mode}",
                style="cyan",
            )
            plan = plan_mode_aware_ct_oc(boundary, cfg, oc_cfg, deadline=args.deadline)
            _print_plan_summary(plan, offsets=offsets, fps=cfg.fps, oc_cfg=oc_cfg)
            return plan.trajectory

        def resample_planner() -> Trajectory:
            bc = _transform_sampled_boundary(sample_boundary(rng, cfg), start_pos, start_yaw)
            dist = float(np.linalg.norm(bc.p1 - bc.p0))
            threshold = float(oc_cfg.mode_switch_distance)
            requested_mode = "translation" if dist <= threshold else "turn_run"
            _log(
                f"[resample target] start_target_dist={dist:.2f}m threshold={threshold:.2f}m "
                f"requested_mode={requested_mode}",
                style="cyan",
            )
            plan = plan_mode_aware_ct_oc(bc, cfg, oc_cfg, deadline=args.deadline)
            _print_plan_summary(plan, offsets=offsets, fps=cfg.fps, oc_cfg=oc_cfg)
            return plan.trajectory

        run_mujoco_viewer(
            plans[idx].trajectory,
            offsets=offsets,
            fps=cfg.fps,
            playback_fps=float(viewer_cfg.get("playback_fps", cfg.fps)),
            root_height=root_height,
            loop=bool(viewer_cfg.get("loop", True)),
            max_frames=viewer_cfg.get("max_frames"),
            rng=None,
            sampling_cfg=None,
            manual_planner=manual_planner,
            resample_planner=resample_planner,
            root_radius=float(oc_cfg.mode_switch_distance),
        )


if __name__ == "__main__":
    main()
