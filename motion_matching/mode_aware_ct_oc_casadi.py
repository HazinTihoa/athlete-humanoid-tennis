"""CasADi + IPOPT version of the two-mode CT-OC strike-entry planner.

Parallel implementation to mode_aware_ct_oc_optimizer.py (the SLSQP version, left
untouched). Same CLI, same output contract (Trajectory / PlanCandidate / ModeAwarePlan),
same MuJoCo viewer.

Differences vs the SLSQP version:

1. FORMULATION FIX (the "always fallback" root cause): the SLSQP knot constraint bounded
   only the BODY acceleration |u|, but the post-check `_ctoc_trajectory_feasible` bounds
   the WORLD acceleration, which includes the coriolis term R(yaw)(u + w x v_body).
   Turning at speed blew the cap in the post-check even when the solver "succeeded".
   Here the knot constraint bounds |u[:2] + w x v_body| directly.

2. SOLVER: IPOPT with exact AD derivatives instead of finite-difference SLSQP.

3. MULTI-PHASE TRANSCRIPTION with FREE phase durations: each mode's phases
   (translation: translate->align; turn_run: turn->run->recover) get their own duration
   decision variable T_p. The optimizer chooses how to split time between phases —
   `turn_phase` / `recover_phase` / `translation_align_phase` in the config are only the
   INITIAL-GUESS fractions, not fixed splits.

4. Explicit DISTANCE (path length) minimization: distance_weight * path_len / dist_scale.

5. Dense output via cubic Hermite splines (C1-smooth path, no polyline corners), with
   adaptive knot spacing and small constraint margins so the dense trajectory passes the
   post-check's slack.

6. Mode escalation: if the requested mode is infeasible the OTHER mode is tried before
   falling back to optimized_omni.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from mode_aware_ct_oc_optimizer import (
    COURT_FORWARD_YAW,
    CTOCConfig,
    ModeAwarePlan,
    ModeName,
    PlanCandidate,
    _boundary_states,
    _ctoc_config_from_yaml,
    _ctoc_trajectory_feasible,
    _duration_guess,
    _initial_guess,
    _manual_boundary_from_args,
    _print_plan_summary,
    _run_direction,
    _start_pose_from_args,
    _state_dynamics,
    _target_axes,
    _terminal_error_ratios,
    _terminal_errors,
    _terminal_success,
    _trajectory_stats,
    _transform_sampled_boundary,
    build_argparser,
)
from target_trajectory_optimizer import (
    DEFAULT_OFFSETS,
    BoundaryCondition,
    SamplingConfig,
    Trajectory,
    _body_vel_to_world,
    _load_yaml_config,
    _log,
    _sampling_config_from_yaml,
    generate_trajectory,
    run_mujoco_viewer,
    sample_boundary,
    save_npz,
    trajectory_is_feasible,
    unwrap_to_near,
    validate_mujoco_trajectory,
    wrap_angle,
)

_SMOOTH_EPS = 1e-8
# Smooth forward/backward accel-limit switch: width + a CONSERVATIVE shift. The post-check
# (_body_limit_feasible) uses a sharp where(vf>=0); shifting the tanh by +2*width makes the
# smooth limit sit at/below the sharp limit for every vf (it reaches the restrictive branch
# already at vf~0), so a solution feasible at the knots is feasible for the post-check too.
_VF_SWITCH_EPS = 0.15    # m/s — switch width
_VF_SWITCH_SHIFT = 0.30  # m/s — conservative shift (2 * width)
_KNOT_SPACING_GUESS = 0.20  # s — initial knot spacing per phase
_KNOT_SPACING_MAX = 0.28    # s — hard cap via T_p upper bound (limits between-knot drift)


@dataclass(frozen=True)
class CTOCConfigX(CTOCConfig):
    """CTOCConfig + casadi-planner extras (old SLSQP planner ignores these)."""
    distance_weight: float = 0.35        # explicit path-length cost: w * path_len / dist_scale
    phase_min_time: float = 0.10         # lower bound of each free phase duration (s)
    translation_align_phase: float = 0.35  # INITIAL-GUESS fraction of the align phase


def _ctoc_config_from_yaml_x(raw_cfg: dict) -> CTOCConfigX:
    base = _ctoc_config_from_yaml(raw_cfg)
    raw = raw_cfg.get("ct_oc", {}) or {}
    return CTOCConfigX(
        **asdict(base),
        distance_weight=float(raw.get("distance_weight", 0.35)),
        phase_min_time=float(raw.get("phase_min_time", 0.10)),
        translation_align_phase=float(raw.get("translation_align_phase", 0.35)),
    )


def _soft_neg(x):
    """Smooth min(x, 0)."""
    import casadi as ca

    return 0.5 * (x - ca.sqrt(x * x + _SMOOTH_EPS))


def _soft_pos(x):
    """Smooth max(x, 0)."""
    import casadi as ca

    return 0.5 * (x + ca.sqrt(x * x + _SMOOTH_EPS))


def _smoothstep(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def _trajectory_from_ctoc_knots_smooth(
    boundary: BoundaryCondition,
    knot_t: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    fps: float,
) -> Trajectory:
    """Dense output via cubic Hermite splines (knot states + knot derivatives f(x,u)).

    Linear interpolation of the knot states (the SLSQP version) draws a polyline with a
    visible corner at every knot. The cubic Hermite spline through (x_k, xdot_k) is the
    natural C1-smooth dense output of trapezoidal collocation. Handles non-uniform knot_t.
    """
    from scipy.interpolate import CubicHermiteSpline

    duration = float(knot_t[-1])
    n = max(2, int(np.ceil(duration * fps)) + 1)
    t = np.linspace(0.0, duration, n, dtype=np.float64)
    xdot = _state_dynamics(states, controls)                     # (k, 6) knot derivatives
    dense = np.empty((n, 6), dtype=np.float64)
    dense_dvel = np.empty((n, 3), dtype=np.float64)              # d/dt of (vf, vl, w) = body accel
    for i in range(6):
        sp = CubicHermiteSpline(knot_t, states[:, i], xdot[:, i])
        dense[:, i] = sp(t)
        if i >= 3:
            dense_dvel[:, i - 3] = sp.derivative()(t)

    pos = dense[:, :2]
    yaw = dense[:, 2]
    body_vel = dense[:, 3:5]
    yaw_rate = dense[:, 5]
    body_acc = dense_dvel[:, :2]
    yaw_acc = dense_dvel[:, 2]
    vel = _body_vel_to_world(body_vel, yaw)
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


def solve_mode_ct_oc_casadi(
    boundary: BoundaryCondition,
    cfg: SamplingConfig,
    oc_cfg: CTOCConfig,
    mode: ModeName,
    deadline: float | None = None,
) -> PlanCandidate:
    try:
        import casadi as ca
    except Exception as exc:  # pragma: no cover
        return PlanCandidate(mode, None, False, 0.0, np.inf, np.inf, 0.0, 0.0, 0.0, f"casadi unavailable: {exc}")

    start, target, yaw1 = _boundary_states(boundary)
    lo_t, hi_t = cfg.duration_range
    if deadline is not None:
        hi_t = min(float(hi_t), float(deadline))
    lo_t = max(1e-3, float(lo_t))
    hi_t = max(lo_t, float(hi_t))
    duration0 = _duration_guess(boundary, cfg, mode, deadline)
    phase_min = max(0.02, float(getattr(oc_cfg, "phase_min_time", 0.10)))
    distance_weight = float(getattr(oc_cfg, "distance_weight", 0.35))
    align_frac = float(np.clip(getattr(oc_cfg, "translation_align_phase", 0.35), 0.1, 0.6))

    # ---- multi-phase layout: FREE phase durations, fixed knots per phase --------------------
    # config fractions are only the INITIAL GUESS of the time split.
    if mode == "translation":
        phase_names = ("translate", "align")
        fracs = np.array([1.0 - align_frac, align_frac], dtype=np.float64)
    else:
        tf = float(np.clip(oc_cfg.turn_phase, 0.1, 0.45))
        rf = float(np.clip(oc_cfg.recover_phase, 0.1, 0.45))
        phase_names = ("turn", "run", "recover")
        fracs = np.array([tf, max(0.1, 1.0 - tf - rf), rf], dtype=np.float64)
    fracs = fracs / fracs.sum()
    P = len(fracs)
    Tp0 = np.maximum(fracs * duration0, phase_min + 0.02)
    kp = [int(np.clip(np.ceil(t / _KNOT_SPACING_GUESS) + 1, 5, 18)) for t in Tp0]
    K = sum(kp) - (P - 1)                                   # phases share their boundary knot
    starts = [0]
    for p in range(P):
        starts.append(starts[-1] + kp[p] - 1)               # starts[p] .. starts[p+1] inclusive
    phase_knots = [list(range(starts[p], starts[p + 1] + 1)) for p in range(P)]
    seg_phase = np.empty(K - 1, dtype=int)                  # segment i sits between knots i, i+1
    for p in range(P):
        seg_phase[starts[p]: starts[p + 1]] = p

    x0_np, u0_np = _initial_guess(boundary, cfg, oc_cfg, mode, duration0, K)

    af_lo = -max(cfg.opt_forward_brake, cfg.opt_backward_accel)
    af_hi = max(cfg.opt_forward_accel, cfg.opt_backward_brake)
    max_extent = max(15.0, 2.0 * float(np.linalg.norm(boundary.p1 - boundary.p0)) + 5.0)
    yaw_pad = 2.0 * np.pi
    yaw_min = min(boundary.yaw0, yaw1, _run_direction(boundary)) - yaw_pad
    yaw_max = max(boundary.yaw0, yaw1, _run_direction(boundary)) + yaw_pad

    dist, target_dir, target_left, dist_scale = _target_axes(boundary)
    run_yaw = unwrap_to_near(_run_direction(boundary), boundary.yaw0)
    target_yaw_from_run = unwrap_to_near(boundary.yaw1, run_yaw)
    accel_scale = np.array(
        [
            max(cfg.opt_forward_accel, cfg.opt_forward_brake, cfg.opt_backward_accel, cfg.opt_backward_brake, 1e-6),
            max(cfg.opt_lateral_accel, 1e-6),
            max(cfg.opt_yaw_accel, 1e-6),
        ],
        dtype=np.float64,
    )

    # ---- decision variables: z = [Tp (P), vec(X 6xK), vec(U 3xK)] ----------------------------
    Tp = ca.MX.sym("Tp", P)
    X = ca.MX.sym("X", 6, K)   # per knot: x, y, yaw, vf, vl, w
    U = ca.MX.sym("U", 3, K)   # per knot: af, al, aw
    z = ca.vertcat(Tp, ca.vec(X), ca.vec(U))
    T_total = ca.sum1(Tp)
    seg_dt = [Tp[int(seg_phase[i])] / float(kp[int(seg_phase[i])] - 1) for i in range(K - 1)]

    def dyn(xi, ui):
        yaw = xi[2]
        vf_, vl_, w_ = xi[3], xi[4], xi[5]
        c, s = ca.cos(yaw), ca.sin(yaw)
        return ca.vertcat(c * vf_ - s * vl_, s * vf_ + c * vl_, w_, ui[0], ui[1], ui[2])

    g: list = []
    lbg: list = []
    ubg: list = []

    def add(expr, lo, hi):
        g.append(expr)
        n = expr.numel()
        lbg.extend([lo] * n if np.isscalar(lo) else list(lo))
        ubg.extend([hi] * n if np.isscalar(hi) else list(hi))

    # total duration window (each Tp additionally has its own bounds below)
    add(T_total, lo_t, hi_t)

    # initial state (equality)
    add(X[:, 0] - ca.DM(start), 0.0, 0.0)

    # trapezoidal collocation with the segment's phase dt (equality)
    for i in range(K - 1):
        f0 = dyn(X[:, i], U[:, i])
        f1 = dyn(X[:, i + 1], U[:, i + 1])
        add(X[:, i + 1] - X[:, i] - 0.5 * seg_dt[i] * (f0 + f1), 0.0, 0.0)

    # terminal state within configured tolerances (hard; squared forms stay smooth)
    add((X[0, -1] - target[0]) ** 2 + (X[1, -1] - target[1]) ** 2,
        -np.inf, float(oc_cfg.terminal_pos_tolerance) ** 2)
    add((X[2, -1] - target[2]) ** 2, -np.inf, float(oc_cfg.terminal_yaw_tolerance) ** 2)
    add((X[3, -1] - target[3]) ** 2 + (X[4, -1] - target[4]) ** 2,
        -np.inf, float(oc_cfg.terminal_vel_tolerance) ** 2)
    add((X[5, -1] - target[5]) ** 2, -np.inf, float(oc_cfg.terminal_yaw_rate_tolerance) ** 2)

    vf, vl, w = X[3, :], X[4, :], X[5, :]
    # Small margins so the cubic-Hermite dense output (which can genuinely drift a little
    # BETWEEN knots) still passes the post-check's 1.03x/1.12x slack.
    body_speed_sq = vf ** 2 + vl ** 2
    add(body_speed_sq.T, -np.inf, (0.98 * float(cfg.max_speed)) ** 2)

    # WORLD acceleration cap incl. the coriolis term w x v_body — this is what
    # _ctoc_trajectory_feasible checks; omitting it was the SLSQP version's fallback bug.
    world_acc_sq = (U[0, :] - w * vl) ** 2 + (U[1, :] + w * vf) ** 2
    add(world_acc_sq.T, -np.inf, (0.97 * float(cfg.max_accel)) ** 2)

    # asymmetric forward accel/brake limits, smoothly + conservatively switched on sign(vf)
    sigma = 0.5 * (1.0 - ca.tanh((vf - _VF_SWITCH_SHIFT) / _VF_SWITCH_EPS))  # ~0 forward, ~1 at vf<=0
    pos_lim = cfg.opt_forward_accel + (cfg.opt_backward_brake - cfg.opt_forward_accel) * sigma
    neg_lim = cfg.opt_forward_brake + (cfg.opt_backward_accel - cfg.opt_forward_brake) * sigma
    add((pos_lim - U[0, :]).T, 0.0, np.inf)
    add((neg_lim + U[0, :]).T, 0.0, np.inf)

    rel_x = X[0, :] - float(boundary.p0[0])
    rel_y = X[1, :] - float(boundary.p0[1])
    progress = rel_x * float(target_dir[0]) + rel_y * float(target_dir[1])
    cross = rel_x * float(target_left[0]) + rel_y * float(target_left[1])

    if mode == "translation":
        translation_speed_limit = max(cfg.opt_lateral_speed, cfg.opt_backward_speed, 0.60 * cfg.opt_forward_speed)
        add(body_speed_sq.T, -np.inf, float(translation_speed_limit) ** 2)
    else:
        corridor = max(0.15, min(float(oc_cfg.turn_run_corridor_width), float(oc_cfg.turn_run_corridor_ratio) * dist))
        recover_distance = max(0.15, float(oc_cfg.turn_run_recover_distance))
        yaw_tol = np.deg2rad(max(1.0, float(oc_cfg.turn_run_yaw_tolerance_deg)))
        for i in phase_knots[0]:                              # TURN
            add(body_speed_sq[i], -np.inf, float(oc_cfg.turn_run_turn_speed) ** 2)
            add(progress[i], -np.inf, float(oc_cfg.turn_run_turn_progress))
        for i in phase_knots[1]:                              # RUN
            add(cross[i], -corridor, corridor)
            add(X[2, i] - run_yaw, -yaw_tol, yaw_tol)
            add(X[4, i], -0.35 * cfg.opt_lateral_speed, 0.35 * cfg.opt_lateral_speed)
        for i in phase_knots[2]:                              # RECOVER
            add(progress[i], dist - recover_distance, np.inf)
            add(cross[i], -corridor, corridor)

    # ---- objective ---------------------------------------------------------------------------
    terminal_ratios_sq = (
        ((X[0, -1] - target[0]) ** 2 + (X[1, -1] - target[1]) ** 2) / max(float(oc_cfg.terminal_pos_tolerance), 1e-6) ** 2
        + (X[2, -1] - target[2]) ** 2 / max(float(oc_cfg.terminal_yaw_tolerance), 1e-6) ** 2
        + ((X[3, -1] - target[3]) ** 2 + (X[4, -1] - target[4]) ** 2) / max(float(oc_cfg.terminal_vel_tolerance), 1e-6) ** 2
        + (X[5, -1] - target[5]) ** 2 / max(float(oc_cfg.terminal_yaw_rate_tolerance), 1e-6) ** 2
    )

    smooth = (
        ca.sumsqr(U[0, :] / accel_scale[0]) + ca.sumsqr(U[1, :] / accel_scale[1]) + ca.sumsqr(U[2, :] / accel_scale[2])
    ) / float(K)
    jerk = 0
    if K > 2:
        jerk_terms = []
        for i in range(K - 1):
            du = (U[:, i + 1] - U[:, i]) / seg_dt[i]
            jerk_terms.append(ca.sumsqr(du / ca.DM(accel_scale)))
        jerk = sum(jerk_terms) / float(K - 1)

    # explicit path-length minimization (smooth |dp|) + straightness/no-backtracking
    dx = X[0, 1:] - X[0, :-1]
    dy = X[1, 1:] - X[1, :-1]
    path_len = ca.sum2(ca.sqrt(dx ** 2 + dy ** 2 + _SMOOTH_EPS))
    distance_cost = distance_weight * path_len / dist_scale
    path_cost = 0
    if dist > 1e-6:
        path_cost = oc_cfg.path_weight * ca.sumsqr(cross / dist_scale) / float(K)
        dprog = progress[1:] - progress[:-1]
        path_cost = path_cost + oc_cfg.reverse_progress_weight * ca.sumsqr(_soft_neg(dprog)) / float(K - 1)

    def _local_s(p: int) -> np.ndarray:
        n = len(phase_knots[p])
        return np.linspace(0.0, 1.0, n, dtype=np.float64)

    if mode == "translation":
        idx_tr, idx_al = phase_knots[0], phase_knots[1]
        abs_vf = ca.sqrt(vf ** 2 + _SMOOTH_EPS)
        high_forward = _soft_pos(abs_vf - 0.65 * cfg.opt_forward_speed)
        mode_cost = (
            oc_cfg.translation_yaw_rate_weight * ca.sumsqr(w / max(cfg.max_yaw_rate, 1e-6)) / float(K)
            + oc_cfg.translation_sprint_weight * ca.sumsqr(high_forward / max(cfg.opt_forward_speed, 1e-6)) / float(K)
            + path_cost
        )
        # translate phase holds the current heading; align phase blends to the target yaw.
        mode_cost = mode_cost + oc_cfg.translation_yaw_weight * ca.sumsqr(
            (X[2, idx_tr] - float(boundary.yaw0)) / np.pi) / float(len(idx_tr))
        yaw_ref_al = boundary.yaw0 + _smoothstep(_local_s(1)) * (target[2] - boundary.yaw0)
        mode_cost = mode_cost + oc_cfg.translation_yaw_weight * ca.sumsqr(
            (X[2, idx_al] - ca.DM(yaw_ref_al).T) / np.pi) / float(len(idx_al))
    else:
        idx_t, idx_r, idx_c = phase_knots[0], phase_knots[1], phase_knots[2]
        mode_cost = path_cost
        yaw_ref_t = boundary.yaw0 + _smoothstep(_local_s(0)) * (run_yaw - boundary.yaw0)
        mode_cost = mode_cost + oc_cfg.turn_run_yaw_weight * ca.sumsqr(
            (X[2, idx_t] - ca.DM(yaw_ref_t).T) / np.pi) / float(len(idx_t))
        mode_cost = mode_cost + oc_cfg.turn_run_yaw_weight * ca.sumsqr(
            (X[2, idx_r] - run_yaw) / np.pi) / float(len(idx_r))
        mode_cost = mode_cost + oc_cfg.turn_run_forward_weight * ca.sumsqr(
            (X[3, idx_r] - 0.92 * cfg.opt_forward_speed) / max(cfg.opt_forward_speed, 1e-6)) / float(len(idx_r))
        mode_cost = mode_cost + oc_cfg.turn_run_lateral_weight * ca.sumsqr(
            X[4, idx_r] / max(cfg.opt_lateral_speed, 1e-6)) / float(len(idx_r))
        mode_cost = mode_cost + 0.25 * ca.sumsqr(
            X[5, idx_r] / max(cfg.max_yaw_rate, 1e-6)) / float(len(idx_r))
        yaw_ref_c = run_yaw + _smoothstep(_local_s(2)) * (target_yaw_from_run - run_yaw)
        mode_cost = mode_cost + oc_cfg.turn_run_recover_weight * ca.sumsqr(
            (X[2, idx_c] - ca.DM(yaw_ref_c).T) / np.pi) / float(len(idx_c))
        mode_cost = mode_cost + 0.5 * oc_cfg.turn_run_recover_weight * (
            ca.sumsqr((X[3, idx_c] - target[3]) / max(cfg.opt_forward_speed, 1e-6))
            + ca.sumsqr((X[4, idx_c] - target[4]) / max(cfg.opt_forward_speed, 1e-6))
        ) / float(len(idx_c))

    objective = (
        oc_cfg.time_weight * T_total
        + 0.1 * oc_cfg.terminal_weight * terminal_ratios_sq   # mild centering; tolerances are hard constraints
        + oc_cfg.smooth_weight * smooth
        + oc_cfg.jerk_weight * jerk
        + distance_cost
        + mode_cost
    )

    # ---- bounds & initial guess ----------------------------------------------------------------
    state_lb = [-max_extent, -max_extent, yaw_min,
                -0.97 * cfg.opt_backward_speed, -0.97 * cfg.opt_lateral_speed, -0.98 * cfg.max_yaw_rate]
    state_ub = [max_extent, max_extent, yaw_max,
                0.97 * cfg.opt_forward_speed, 0.97 * cfg.opt_lateral_speed, 0.98 * cfg.max_yaw_rate]
    ctrl_lb = [af_lo, -cfg.opt_lateral_accel, -cfg.opt_yaw_accel]
    ctrl_ub = [af_hi, cfg.opt_lateral_accel, cfg.opt_yaw_accel]
    # T_p bounds: min phase time, and a per-phase cap keeping the knot spacing <= _KNOT_SPACING_MAX
    # (sparser knots let the state genuinely drift |u|*dt/4 beyond its bound between knots).
    tp_lb = [phase_min] * P
    tp_ub = [_KNOT_SPACING_MAX * (kp[p] - 1) for p in range(P)]
    lbx = np.concatenate([tp_lb, np.tile(state_lb, K), np.tile(ctrl_lb, K)])
    ubx = np.concatenate([tp_ub, np.tile(state_ub, K), np.tile(ctrl_ub, K)])
    z0 = np.concatenate([Tp0, x0_np.reshape(-1), u0_np.reshape(-1)])
    z0 = np.clip(z0, lbx, ubx)

    opts = {
        "ipopt.print_level": 0,
        "print_time": False,
        "ipopt.sb": "yes",
        "ipopt.max_iter": max(300, int(oc_cfg.max_iter)),
        "ipopt.tol": 1e-6,
        "ipopt.acceptable_tol": 1e-4,
        "ipopt.acceptable_iter": 8,
        "ipopt.mu_strategy": "adaptive",
    }
    try:
        solver = ca.nlpsol("ctoc", "ipopt", {"x": z, "f": objective, "g": ca.vertcat(*g)}, opts)
        sol = solver(x0=z0, lbx=lbx, ubx=ubx, lbg=np.asarray(lbg, dtype=np.float64), ubg=np.asarray(ubg, dtype=np.float64))
        stats = solver.stats()
    except Exception as exc:
        return PlanCandidate(mode, None, False, 0.0, np.inf, np.inf, 0.0, 0.0, 0.0, f"ipopt raised: {exc}")

    z_opt = np.asarray(sol["x"]).reshape(-1)
    if not np.isfinite(z_opt).all():
        z_opt = z0
    tp_opt = np.asarray(z_opt[:P], dtype=np.float64)
    duration = float(tp_opt.sum())
    states = z_opt[P: P + 6 * K].reshape(K, 6)
    controls = z_opt[P + 6 * K:].reshape(K, 3)

    # non-uniform knot time grid from the solved phase durations
    knot_t = np.zeros(K, dtype=np.float64)
    for i in range(K - 1):
        p = int(seg_phase[i])
        knot_t[i + 1] = knot_t[i] + tp_opt[p] / float(kp[p] - 1)
    solved_bc = replace(boundary, duration=duration)
    traj = _trajectory_from_ctoc_knots_smooth(solved_bc, knot_t, states, controls, fps=cfg.fps)

    # post checks (same criteria as the SLSQP version)
    seg_dt_np = np.diff(knot_t)
    f_all = _state_dynamics(states, controls)
    colloc = states[1:] - states[:-1] - 0.5 * seg_dt_np[:, None] * (f_all[:-1] + f_all[1:])
    dynamics_residual = float(np.max(np.abs(np.concatenate([states[0] - start, colloc.reshape(-1)]))))
    solver_ok = bool(stats.get("success", False))
    status = str(stats.get("return_status", "?"))
    terminal_ok = _terminal_success(states, target, oc_cfg)
    dynamics_ok = dynamics_residual <= float(oc_cfg.dynamics_tolerance)
    feasible = solver_ok and terminal_ok and dynamics_ok and _ctoc_trajectory_feasible(traj, cfg)
    pos_err, yaw_err, vel_err, yaw_rate_err = _terminal_errors(states, target)
    term = float(max(_terminal_error_ratios(states, target, oc_cfg)))
    max_speed, max_accel, max_yaw_rate = _trajectory_stats(traj)
    path_len_np = float(np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=-1)))
    phases_str = " ".join(f"{n}={t:.2f}s" for n, t in zip(phase_names, tp_opt))
    return PlanCandidate(
        mode=mode,
        trajectory=traj,
        success=feasible,
        duration=duration,
        score=float(sol["f"]),
        terminal_error=term,
        max_speed=max_speed,
        max_accel=max_accel,
        max_yaw_rate=max_yaw_rate,
        message=(
            f"ipopt {status} iters={stats.get('iter_count', '?')} "
            f"dyn_res={dynamics_residual:.2e}; phases[{phases_str}] "
            f"path_len={path_len_np:.2f}m (straight={dist:.2f}m); "
            f"terminal pos={pos_err:.3f}m yaw={yaw_err:.3f}rad "
            f"vel={vel_err:.3f}m/s yaw_rate={yaw_rate_err:.3f}rad/s ratio={term:.2f}"
        ),
    )


def plan_mode_aware_ct_oc_casadi(
    boundary: BoundaryCondition,
    cfg: SamplingConfig,
    oc_cfg: CTOCConfig,
    deadline: float | None = None,
) -> ModeAwarePlan:
    dist = float(np.linalg.norm(boundary.p1 - boundary.p0))
    threshold = float(oc_cfg.mode_switch_distance)
    requested_mode: ModeName = "translation" if dist <= threshold else "turn_run"
    candidates: tuple[PlanCandidate, ...] = (
        solve_mode_ct_oc_casadi(boundary, cfg, oc_cfg, requested_mode, deadline=deadline),
    )
    # Mode escalation: try the other mode before giving up (e.g. a tight side-step target
    # that translation can't reach in time may still be feasible as turn_run and vice versa).
    if not candidates[0].success:
        other: ModeName = "turn_run" if requested_mode == "translation" else "translation"
        candidates = candidates + (solve_mode_ct_oc_casadi(boundary, cfg, oc_cfg, other, deadline=deadline),)

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
            f"both CT-OC modes failed for dist={dist:.2f}m "
            f"(threshold={threshold:.2f}m); used optimized_omni fallback"
        ),
        fallback_used=True,
    )
    return ModeAwarePlan(fallback, fallback_candidate.mode, candidates + (fallback_candidate,), fallback_used=True)


def _log_selected(plan: ModeAwarePlan) -> None:
    """Log the selected candidate's solver message (contains the ACTUAL phase durations —
    the table's phase column shows only the nominal config fractions)."""
    for c in plan.candidates:
        if c.mode == plan.selected_mode and c.success:
            _log(f"[selected detail] {c.message}", style="green")
            return


def main() -> None:
    parser = build_argparser()
    parser.description = "Two-mode CT-OC strike-entry planner (CasADi + IPOPT, free phase times)"
    args = parser.parse_args()
    raw_cfg = _load_yaml_config(args.config)
    sampling = raw_cfg.get("sampling", {})
    query_cfg = raw_cfg.get("query", {})
    viewer_cfg = raw_cfg.get("viewer", {})
    validation_cfg = raw_cfg.get("validation", {})
    output_cfg = raw_cfg.get("output", {})

    cfg = _sampling_config_from_yaml(sampling)
    oc_cfg = _ctoc_config_from_yaml_x(raw_cfg)
    num = int(args.num if args.num is not None else sampling.get("num", 8))
    seed = int(args.seed if args.seed is not None else sampling.get("seed", 0))
    offsets = tuple(int(v) for v in query_cfg.get("offsets", DEFAULT_OFFSETS))
    rng = np.random.default_rng(seed)
    start_pos, start_yaw = _start_pose_from_args(args)
    _log(
        f"[start] service-line center root=({start_pos[0]:.2f},{start_pos[1]:.2f}) "
        f"yaw={np.rad2deg(start_yaw):.1f}deg; +X points to opponent court  [solver=casadi+ipopt]",
        style="cyan",
    )

    manual_bc = _manual_boundary_from_args(args, cfg, start_pos=start_pos, start_yaw=start_yaw)
    boundaries = (
        [manual_bc]
        if manual_bc is not None
        else [_transform_sampled_boundary(sample_boundary(rng, cfg), start_pos, start_yaw) for _ in range(max(1, num))]
    )
    plans = [plan_mode_aware_ct_oc_casadi(bc, cfg, oc_cfg, deadline=args.deadline) for bc in boundaries]

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
        _log_selected(plan)

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
            plan = plan_mode_aware_ct_oc_casadi(boundary, cfg, oc_cfg, deadline=args.deadline)
            _print_plan_summary(plan, offsets=offsets, fps=cfg.fps, oc_cfg=oc_cfg)
            _log_selected(plan)
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
            plan = plan_mode_aware_ct_oc_casadi(bc, cfg, oc_cfg, deadline=args.deadline)
            _print_plan_summary(plan, offsets=offsets, fps=cfg.fps, oc_cfg=oc_cfg)
            _log_selected(plan)
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
