"""Tennis Motion Matching — OFFICIAL move-MM core on the tennis_mm_mj skeleton.

Parallel implementation to tennis_mm_mj.py (left untouched). Same skeleton
(CasADi CT-OC target trajectory + closed-loop phase, strike-entry planner,
tennis court, ghost, mouse/keyboard interaction), but the MOVE-stage motion
matching internals and the database are ported from the official
Motion-Matching implementation (Motion-Matching-current-git, the Holden
controller.cpp Python/MuJoCo port):

  - 33-dim extended official feature layout: Lfoot/Rfoot pos (6) + Lfoot/Rfoot vel (6)
    + hip(root) vel (3) + future root pos xz @20/40/60 (6) + future dir xz (6),
    + right racket-hand position (3) + root-relative hand velocity (3),
    in the y-up internal facing frame, with the official per-group
    normalization (mean + group-mean-std / build-weight baked into scale).
  - Official query: foot/root and racket-hand features are COPIED from the
    current matched frame's denormalized features; only the trajectory block
    comes from the (CasADi) target trajectory.
  - Official brute-force search: additive transition_cost on candidates, the
    current frame competes at no extra cost, ignore_range_end /
    ignore_surrounding masks, runtime feature weights = (requested/built)^2.
  - Database = the official binary resources built by
    build_official_mm_db.py (features.bin + g1_state.npz + metadata.json +
    strike_metadata.npz; database.bin / bone-local skeleton is NOT loaded —
    display and FK go through the per-frame G1 state + MuJoCo).

Coordinate note (verified numerically to 1e-8): the internal y-up features'
horizontal components (x_int, z_int) equal the z-up facing-frame (-y_local,
x_local); make_query converts the z-up trajectory samples accordingly.

Usage:
    python motion_matching/tennis_official_mm_mj.py \
        --db_file motion_db/official_tennis_runwalk_hand33 \
        --config motion_matching/tennis_official_mm_config.yaml \
        --interactive --ghost
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace
import json
import math
import os
import select
import struct
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import mujoco
from rich.console import Console
import mujoco.viewer
import numpy as np
import torch
import yaml

from target_trajectory_optimizer import (
    BoundaryCondition as TargetBoundaryCondition,
    DEFAULT_CONFIG as TARGET_TRAJ_DEFAULT_CONFIG,
    _cursor_to_ground_point as target_cursor_to_ground_point,
    _draw_tennis_court_overlay,
    _interp_traj as interp_target_traj,
    _load_yaml_config as load_target_yaml_config,
    _sampling_config_from_yaml as target_sampling_config_from_yaml,
    generate_trajectory as generate_target_trajectory,
    wrap_angle as target_wrap_angle,
)
from mode_aware_ct_oc_casadi import _ctoc_config_from_yaml_x, plan_mode_aware_ct_oc_casadi

TWO_PI = 2.0 * math.pi


def build_mujoco_model(court_cfg: dict) -> tuple[mujoco.MjModel, bool]:
    """Build the G1 viewer model, optionally with the shared training court.

    The motion-matching viewer normally renders a line-only regulation court
    centered at the world origin. When ``scene_xml`` is supplied, attach the
    exact court XML used by training and deployment at the configured
    net position instead. The ball body in that XML is removed because this
    viewer visualizes a target ball with markers rather than simulating it.
    """
    scene_xml = court_cfg.get("scene_xml")
    if not scene_xml:
        return mujoco.MjModel.from_xml_path(str(_G1_XML)), False

    court_xml = Path(scene_xml).expanduser()
    if not court_xml.is_absolute():
        court_xml = (Path(__file__).resolve().parents[1] / court_xml).resolve()
    if not court_xml.is_file():
        raise FileNotFoundError(f"Configured tennis court XML was not found: {court_xml}")

    net_x = float(court_cfg.get("net_x", 5.6))
    if net_x <= 0.0:
        raise ValueError(f"court.net_x must be positive, got {net_x}")

    scene_spec = mujoco.MjSpec.from_file(str(_G1_XML))
    # G1's original MJCF has a plane named ``floor`` at z=0.  The shared court
    # has a court surface whose top is also z=0; rendering both causes z-fighting.
    # Keep the floor object for its named contact pairs, but make it invisible.
    floor = scene_spec.geom("floor")
    if floor is not None:
        floor.rgba = [0.0, 0.0, 0.0, 0.0]
    court_spec = mujoco.MjSpec.from_file(str(court_xml))
    ball_body = court_spec.body("tennis_ball")
    if ball_body is not None:
        court_spec.delete(ball_body)
    frame = scene_spec.worldbody.add_frame(pos=[net_x, 0.0, 0.0])
    scene_spec.attach(court_spec, prefix="tennis_court/", frame=frame)
    render_cfg = court_cfg.get("render", {}) or {}
    scene_spec.visual.quality.shadowsize = int(render_cfg.get("shadowsize", 0))
    scene_spec.visual.quality.offsamples = int(render_cfg.get("offsamples", 0))
    return scene_spec.compile(), True


# =============================================================================
# Pure-torch quaternion helpers (wxyz convention, matches Isaac Lab & MuJoCo)
# =============================================================================

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quat_inv(q):
    # conjugate (unit quaternion inverse)
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def quat_apply(q, v):
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w.unsqueeze(-1) * t + torch.cross(q_vec, t, dim=-1)


def quat_apply_inverse(q, v):
    return quat_apply(quat_inv(q), v)


def _facing_quat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    zeros = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


# =============================================================================
# Joint-order remap (Isaac Lab BFS  ->  MuJoCo DFS).  From replay_npz_mujoco.py
# =============================================================================

MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ISAACLAB_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
# qpos[7:36][i] = joint_pos_il[ISAACLAB_TO_MUJOCO[i]]
ISAACLAB_TO_MUJOCO = np.array([ISAACLAB_JOINT_NAMES.index(n) for n in MUJOCO_JOINT_NAMES])

_REPO = Path(__file__).resolve().parents[1]
_G1_XML = _REPO / "robots/replay_unitree_description/mjcf/g1.xml"


# =============================================================================
# Keyboard (interactive) — reads TERMINAL stdin (engine-independent)
# =============================================================================

class KeyboardController:
    def __init__(self):
        self._last_key = ""
        self._lock = threading.Lock()
        self._old_settings = termios.tcgetattr(sys.stdin)
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        tty.setraw(sys.stdin.fileno())
        try:
            while self._running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    with self._lock:
                        self._last_key = ch.lower()
                    if ch == "\x03":
                        self._running = False
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def pop_key(self):
        with self._lock:
            k = self._last_key
            self._last_key = ""
            return k

    def stop(self):
        self._running = False
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
        except Exception:
            pass


# =============================================================================
# Inertialization blending  (verbatim from tennis_mm.py — engine-agnostic)
# =============================================================================

class InertializationBlender:
    def __init__(self, blend_frames, fps, num_envs, device):
        self.blend_duration = blend_frames / fps
        self.num_envs = num_envs
        self.device = device
        self.timer = torch.full((num_envs,), self.blend_duration, device=device)
        self.offset_root_pos = torch.zeros(num_envs, 3, device=device)
        self.offset_root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, -1).clone()
        self.offset_root_linvel = torch.zeros(num_envs, 3, device=device)
        self.offset_root_angvel = torch.zeros(num_envs, 3, device=device)
        self.offset_joint_pos = None
        self.offset_joint_vel = None

    @staticmethod
    def _quintic_vec(t, duration):
        x = (t / max(duration, 1e-6)).clamp(max=1.0)
        return 1.0 - (1.0 - x) ** 5

    def trigger(self, mask, cur_root_pos, new_root_pos, cur_root_quat, new_root_quat,
                cur_root_linvel, new_root_linvel, cur_root_angvel, new_root_angvel,
                cur_joint_pos, new_joint_pos, cur_joint_vel, new_joint_vel):
        if not mask.any():
            return
        self.timer[mask] = 0.0
        self.offset_root_pos[mask] = cur_root_pos[mask] - new_root_pos[mask]
        self.offset_root_quat[mask] = quat_mul(cur_root_quat[mask], quat_inv(new_root_quat[mask]))
        self.offset_root_linvel[mask] = cur_root_linvel[mask] - new_root_linvel[mask]
        self.offset_root_angvel[mask] = cur_root_angvel[mask] - new_root_angvel[mask]
        if self.offset_joint_pos is None:
            self.offset_joint_pos = torch.zeros_like(cur_joint_pos)
            self.offset_joint_vel = torch.zeros_like(cur_joint_vel)
        self.offset_joint_pos[mask] = cur_joint_pos[mask] - new_joint_pos[mask]
        self.offset_joint_vel[mask] = cur_joint_vel[mask] - new_joint_vel[mask]

    def apply(self, root_pos, root_quat, root_linvel, root_angvel, joint_pos, joint_vel):
        blending = self.timer < self.blend_duration
        if not blending.any():
            return root_pos, root_quat, root_linvel, root_angvel, joint_pos, joint_vel

        self.timer = self.timer + 1.0 / 50.0
        alpha = self._quintic_vec(self.timer, self.blend_duration)
        inv_alpha = (1.0 - alpha).unsqueeze(-1)
        blend_mask = blending.unsqueeze(-1)

        root_pos = root_pos + self.offset_root_pos * inv_alpha * blend_mask
        root_linvel = root_linvel + self.offset_root_linvel * inv_alpha * blend_mask
        root_angvel = root_angvel + self.offset_root_angvel * inv_alpha * blend_mask

        if self.offset_joint_pos is not None:
            joint_pos = joint_pos + self.offset_joint_pos * inv_alpha * blend_mask
            joint_vel = joint_vel + self.offset_joint_vel * inv_alpha * blend_mask

        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=root_quat.device).expand_as(root_quat)
        inv_alpha_scalar = 1.0 - alpha
        blend_q = self._quat_slerp_batched(identity, self.offset_root_quat, inv_alpha_scalar)
        blended_quat = quat_mul(blend_q, root_quat)
        root_quat = torch.where(blending.unsqueeze(-1), blended_quat, root_quat)
        return root_pos, root_quat, root_linvel, root_angvel, joint_pos, joint_vel

    @staticmethod
    def _quat_slerp_batched(q0, q1, t):
        t = t.unsqueeze(-1)
        dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        q1 = torch.where(dot < 0, -q1, q1)
        dot = dot.abs()
        theta = torch.acos(dot.clamp(max=0.9999))
        sin_theta = torch.sin(theta).clamp(min=1e-6)
        w0 = torch.sin((1.0 - t) * theta) / sin_theta
        w1 = torch.sin(t * theta) / sin_theta
        near_zero = (theta.abs() < 1e-5).squeeze(-1)
        result = w0 * q0 + w1 * q1
        if near_zero.any():
            result[near_zero] = (1.0 - t[near_zero]) * q0[near_zero] + t[near_zero] * q1[near_zero]
        return torch.nn.functional.normalize(result, dim=-1)


# =============================================================================
# Database  (verbatim MM logic from tennis_mm.py — engine-agnostic torch)
# =============================================================================

class SimulationSpring:
    """Critically-damped spring that turns a desired velocity into a smooth velocity + a rolled-out
    future trajectory (positions + headings). Ported from moving_motion_matching.py — used so the
    move-mode query carries a SMOOTH desired path toward the ball (steers via the traj feature)."""

    def __init__(self, halflife: float = 0.15, device="cpu"):
        self.halflife = halflife
        self.vel = None
        self.acc = None
        self.device = device

    def reset(self):
        self.vel = None
        self.acc = None

    def _step(self, vel, acc, desired_vel, dt):
        y = 0.6931472 / max(self.halflife, 1e-6)
        eydt = math.exp(-y * dt)
        j0 = vel - desired_vel
        j1 = acc + j0 * y
        delta_pos = (eydt * ((-j1 / (y * y)) + ((-j0 - j1 * dt) / y))
                     + j1 / (y * y) + j0 / y + desired_vel * dt)
        new_vel = eydt * (j0 + j1 * dt) + desired_vel
        new_acc = eydt * (acc - j1 * y * dt)
        return new_vel, new_acc, delta_pos

    def update(self, desired_vel, dt):
        if self.vel is None:
            self.vel = desired_vel.clone()
            self.acc = torch.zeros_like(desired_vel)
            return desired_vel.clone()
        self.vel, self.acc, _ = self._step(self.vel, self.acc, desired_vel, dt)
        return self.vel.clone()

    def predict(self, desired_vel, offsets, frame_dt):
        """Returns positions [N, n_off, 3] and headings [N, n_off, 2] in the local (facing) frame."""
        N = desired_vel.shape[0]
        n_offsets = len(offsets)
        if self.vel is None:
            positions = torch.stack([desired_vel * (o * frame_dt) for o in offsets], dim=1)
            headings = torch.zeros(N, n_offsets, 2, device=desired_vel.device)
            vn = desired_vel[:, :2].norm(dim=-1)
            hv = vn > 1e-3
            if hv.any():
                headings[hv] = torch.nn.functional.normalize(
                    desired_vel[hv, :2], dim=-1).unsqueeze(1).expand(-1, n_offsets, -1)
            return positions, headings
        vel = self.vel.clone()
        acc = self.acc.clone()
        pos = torch.zeros_like(vel)
        max_offset = max(offsets)
        offsets_set = set(offsets)
        sampled_pos, sampled_heading = {}, {}
        last_valid_heading = torch.nn.functional.normalize(desired_vel[:, :2].clamp(min=1e-6), dim=-1)
        for frame in range(1, max_offset + 1):
            vel, acc, dp = self._step(vel, acc, desired_vel, frame_dt)
            pos = pos + dp
            if frame in offsets_set:
                sampled_pos[frame] = pos.clone()
                v_xy = vel[:, :2]
                vlen = v_xy.norm(dim=-1, keepdim=True)
                has_speed = (vlen > 1e-3).squeeze(-1)
                heading = last_valid_heading.clone()
                if has_speed.any():
                    heading[has_speed] = v_xy[has_speed] / vlen[has_speed]
                    last_valid_heading[has_speed] = heading[has_speed]
                sampled_heading[frame] = heading
        positions = torch.stack([sampled_pos[o] for o in offsets], dim=1)
        headings = torch.stack([sampled_heading[o] for o in offsets], dim=1)
        return positions, headings


def _read_array1d(f, dtype):
    (n,) = struct.unpack("i", f.read(4))
    return np.frombuffer(f.read(n * np.dtype(dtype).itemsize), dtype=dtype).copy()


def _read_array2d(f, dtype):
    rows, cols = struct.unpack("ii", f.read(8))
    a = np.frombuffer(f.read(rows * cols * np.dtype(dtype).itemsize), dtype=dtype).copy()
    return a.reshape(rows, cols)


class OfficialMMDatabase:
    """Motion database backed by the OFFICIAL Motion-Matching binary resources.

    Loads:
      features.bin        27-dim legacy or 33-dim hand-extended features (already
                          normalized, build weights baked into scale) plus offset/scale
      g1_state.npz        per-frame G1 root pos/quat/vel + joint pos/vel (z-up, wxyz,
                          IsaacLab BFS joint order) -> drives display/FK directly
      metadata.json       fps, clip ranges (type/strike_frame/side/file), built weights
      strike_metadata.npz per-frame racket sweet-spot (y-up internal) -> converted z-up

    database.bin (bone-local Y-up skeleton) is intentionally NOT loaded: the tennis
    skeleton displays via G1 qpos and FKs via MuJoCo.

    Feature layout (33, y-up internal facing frame; legacy databases stop at 27):
      [0:3] Lfoot pos  [3:6] Rfoot pos  [6:9] Lfoot vel  [9:12] Rfoot vel
      [12:15] hip(root) vel  [15:21] future root pos xz @20/40/60  [21:27] future dir xz
      [27:30] right-hand pos relative to root  [30:33] right-hand vel relative to root
    Internal horizontal (x_int, z_int) == z-up facing (-y_local, x_local)  [verified 1e-8].

    Compatibility: exposes body_pos_w/body_quat_w/body_linvel/body_angvel as [T, 1, *]
    (root only, root_idx=0) plus joint_pos/joint_vel/advance_frames/clip_start/clip_end/
    motion_boundaries/strike_frames/... so the tennis_mm_mj skeleton code runs unchanged.
    """

    TRAJ_OFFSETS = (20, 40, 60)
    FEATURE_WEIGHT_SLICES = {
        "foot_position": (0, 6),
        "foot_velocity": (6, 12),
        "hip_velocity": (12, 15),
        "trajectory_position": (15, 21),
        "trajectory_direction": (21, 27),
        "hand_position": (27, 30),
        "hand_velocity": (30, 33),
    }
    FEATURE_WEIGHT_DEFAULTS = {
        "foot_position": 0.75,
        "foot_velocity": 1.0,
        "hip_velocity": 1.0,
        "trajectory_position": 1.0,
        "trajectory_direction": 1.5,
        "hand_position": 0.5,
        "hand_velocity": 0.25,
    }

    def __init__(
        self,
        resource_dir,
        device,
        weights_cfg=None,
        min_neck_height=0.0,
        move_hand_hold_cfg=None,
    ):
        resource_dir = Path(resource_dir)
        self.device = device

        with open(resource_dir / "features.bin", "rb") as f:
            self.features_np = _read_array2d(f, np.float32)
            self.features_offset = _read_array1d(f, np.float32)
            self.features_scale = _read_array1d(f, np.float32)
        self.feature_dim = int(self.features_np.shape[1])
        if self.feature_dim not in (27, 33):
            raise ValueError(f"expected legacy 27-dim or hand-extended 33-dim features, got {self.feature_dim}")
        if self.features_offset.shape != (self.feature_dim,) or self.features_scale.shape != (self.feature_dim,):
            raise ValueError("features.bin offset/scale dimensions do not match the feature matrix")

        with open(resource_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.fps = float(meta.get("fps", 50.0))
        self.T = int(self.features_np.shape[0])
        self.built_feature_weights = dict(meta.get("feature_weights", self.FEATURE_WEIGHT_DEFAULTS))

        g1 = np.load(resource_dir / "g1_state.npz")
        # Root-only "body" arrays shaped [T, 1, *] so skeleton code using
        # db.body_pos_w[f, db.root_idx] works unchanged with root_idx=0.
        self.body_pos_w = torch.tensor(g1["root_pos_w"], dtype=torch.float32, device=device).unsqueeze(1)
        self.body_quat_w = torch.tensor(g1["root_quat_w"], dtype=torch.float32, device=device).unsqueeze(1)
        self.body_linvel = torch.tensor(g1["root_lin_vel_w"], dtype=torch.float32, device=device).unsqueeze(1)
        self.body_angvel = torch.tensor(g1["root_ang_vel_w"], dtype=torch.float32, device=device).unsqueeze(1)
        self.joint_pos = torch.tensor(g1["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(g1["joint_vel"], dtype=torch.float32, device=device)
        self.root_idx = 0
        assert self.body_pos_w.shape[0] == self.T, "g1_state frames != features frames"

        # Per-frame racket sweet spot: internal y-up -> z-up (x, y_z, z_z) = (x_i, z_i, y_i).
        sm = np.load(resource_dir / "strike_metadata.npz")
        rk = sm["racket_pos_internal"].astype(np.float32)
        self.racket_pos_w = torch.tensor(
            np.stack([rk[:, 0], rk[:, 2], rk[:, 1]], axis=-1), dtype=torch.float32, device=device)
        self.racket_offset = torch.tensor(
            meta.get("racket_offset_z_up", [0.38, 0.01, 0.27]), dtype=torch.float32, device=device)

        # Clip ranges / masks (torch for skeleton indexing, numpy for search masks).
        self.traj_offsets = list(self.TRAJ_OFFSETS)
        self.motion_boundaries = []
        self.strike_frames = []
        self.range_starts = []
        self.range_stops = []
        self.clip_start = torch.zeros(self.T, dtype=torch.long, device=device)
        self.clip_end = torch.full((self.T,), self.T - 1, dtype=torch.long, device=device)
        self.is_fh_frame = torch.zeros(self.T, dtype=torch.bool, device=device)
        self.is_bh_frame = torch.zeros(self.T, dtype=torch.bool, device=device)
        self.is_lafan_locomotion = torch.zeros(self.T, dtype=torch.bool, device=device)
        for r in meta.get("ranges", []):
            start, stop = int(r["start"]), int(r["stop"])
            name = str(r.get("file", f"clip_{len(self.motion_boundaries)}"))
            self.range_starts.append(start)
            self.range_stops.append(stop)
            self.motion_boundaries.append((start, stop, name))
            self.clip_start[start:stop] = start
            self.clip_end[start:stop] = stop - 1
            local_strike = int(r.get("strike_frame", -1))
            self.strike_frames.append(start + local_strike if local_strike >= 0 else -1)
            side = str(r.get("side", ""))
            if side == "fh":
                self.is_fh_frame[start:stop] = True
            elif side == "bh":
                self.is_bh_frame[start:stop] = True
            if str(r.get("type", "locomotion")) == "locomotion":
                self.is_lafan_locomotion[start:stop] = True
        self.range_starts = np.asarray(self.range_starts, dtype=np.int64)
        self.range_stops = np.asarray(self.range_stops, dtype=np.int64)
        self.frame_is_locomotion_np = self.is_lafan_locomotion.cpu().numpy()
        self._has_lafan_loco = bool(self.is_lafan_locomotion.any())
        self.phase_label = None  # official db has clip types instead of per-frame phases

        # Move-search mask: locomotion frames, optionally excluding low postures
        # (neck = shoulder mid-point below min_neck_height: crouches, dives, get-ups).
        self.neck_height = self._load_or_compute_neck_height(resource_dir)
        self.move_allowed_np = self.frame_is_locomotion_np.copy()
        if min_neck_height > 0.0:
            low = self.neck_height < float(min_neck_height)
            n_cut = int((self.move_allowed_np & low).sum())
            self.move_allowed_np &= ~low
            print(f"[DB] move filter: neck<{min_neck_height:.2f}m cut {n_cut} loco frames "
                  f"({int(self.move_allowed_np.sum())} searchable left)", end="\r\n")
            if not self.move_allowed_np.any():
                raise RuntimeError(f"min_neck_height={min_neck_height} filtered out ALL loco frames")

        self.feature_cost_weights = self.make_feature_cost_weights(weights_cfg or {})
        self._refresh_weighted_cache()
        self.move_hand_hold_cost = self._build_move_hand_hold_cost(
            resource_dir, move_hand_hold_cfg or {}
        )

        n_strike = int(sum(1 for s in self.strike_frames if s >= 0))
        print(f"\r\n[DB] Official resources {resource_dir}  T={self.T}  fps={self.fps:.0f}  "
              f"feat_dim={self.feature_dim}  clips={len(self.motion_boundaries)} (strike={n_strike})", end="\r\n")
        print(f"[DB] built feature weights: {self.built_feature_weights}", end="\r\n\r\n")

    def _load_or_compute_neck_height(self, resource_dir):
        """Per-frame neck (shoulder mid-point) height via MuJoCo kinematics, cached to
        neck_height_cache.npy next to the resources (first build ~15s, then instant)."""
        cache = resource_dir / "neck_height_cache.npy"
        if cache.exists():
            nh = np.load(cache)
            if nh.shape[0] == self.T:
                return nh.astype(np.float32)
        print(f"[DB] computing neck heights for {self.T} frames (one-time, cached)...", end="\r\n")
        model = mujoco.MjModel.from_xml_path(str(_G1_XML))
        data = mujoco.MjData(model)
        lsh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link")
        rsh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link")
        root_pos = self.body_pos_w[:, 0].cpu().numpy().astype(np.float64)
        root_quat = self.body_quat_w[:, 0].cpu().numpy().astype(np.float64)
        jp_mj = self.joint_pos.cpu().numpy()[:, ISAACLAB_TO_MUJOCO].astype(np.float64)
        nh = np.empty(self.T, dtype=np.float32)
        for f in range(self.T):
            data.qpos[:3] = root_pos[f]
            data.qpos[3:7] = root_quat[f]
            data.qpos[7:36] = jp_mj[f]
            mujoco.mj_kinematics(model, data)
            nh[f] = 0.5 * (data.xpos[lsh, 2] + data.xpos[rsh, 2])
        try:
            np.save(cache, nh)
            print(f"[DB] neck heights cached -> {cache}", end="\r\n")
        except OSError as exc:
            print(f"[DB] could not cache neck heights: {exc}", end="\r\n")
        return nh

    def _load_or_compute_wrist_root_positions(self, resource_dir):
        """Both wrists in each database frame's root-facing coordinate frame."""
        cache = resource_dir / "wrist_root_facing_cache.npy"
        if cache.exists():
            wrist_pos = np.load(cache)
            if wrist_pos.shape == (self.T, 2, 3):
                return wrist_pos.astype(np.float32)

        print(
            f"[DB] computing root-frame wrist positions for {self.T} frames "
            "(one-time, cached)...",
            end="\r\n",
        )
        model = mujoco.MjModel.from_xml_path(str(_G1_XML))
        data = mujoco.MjData(model)
        wrist_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"),
        ]
        assert all(wrist_id >= 0 for wrist_id in wrist_ids), "g1.xml missing wrist bodies"
        root_pos = self.body_pos_w[:, 0].cpu().numpy().astype(np.float64)
        root_quat = self.body_quat_w[:, 0].cpu().numpy().astype(np.float64)
        jp_mj = self.joint_pos.cpu().numpy()[:, ISAACLAB_TO_MUJOCO].astype(np.float64)
        wrist_world = np.empty((self.T, 2, 3), dtype=np.float32)
        for f in range(self.T):
            data.qpos[:3] = root_pos[f]
            data.qpos[3:7] = root_quat[f]
            data.qpos[7:36] = jp_mj[f]
            mujoco.mj_kinematics(model, data)
            wrist_world[f] = data.xpos[wrist_ids]

        root_pos_t = torch.tensor(root_pos, dtype=torch.float32, device=self.device)
        root_quat_t = torch.tensor(root_quat, dtype=torch.float32, device=self.device)
        wrist_world_t = torch.tensor(wrist_world, dtype=torch.float32, device=self.device)
        facing_quat = _facing_quat(root_quat_t)[:, None, :].expand(-1, 2, -1)
        wrist_local = quat_apply_inverse(
            facing_quat.reshape(-1, 4),
            (wrist_world_t - root_pos_t[:, None, :]).reshape(-1, 3),
        ).reshape(self.T, 2, 3).cpu().numpy()
        try:
            np.save(cache, wrist_local)
            print(f"[DB] root-frame wrist positions cached -> {cache}", end="\r\n")
        except OSError as exc:
            print(f"[DB] could not cache wrist positions: {exc}", end="\r\n")
        return wrist_local

    @staticmethod
    def _reference_wrist_hold_position(reference_motion):
        """Final reference-frame wrist positions in the root-facing frame."""
        motion = np.load(reference_motion)
        model = mujoco.MjModel.from_xml_path(str(_G1_XML))
        data = mujoco.MjData(model)
        wrist_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"),
        ]
        assert all(wrist_id >= 0 for wrist_id in wrist_ids), "g1.xml missing wrist bodies"
        root_pos = motion["body_pos_w"][-1, 0].astype(np.float64)
        root_quat = motion["body_quat_w"][-1, 0].astype(np.float64)
        joint_pos = motion["joint_pos"][-1, ISAACLAB_TO_MUJOCO].astype(np.float64)
        data.qpos[:3] = root_pos
        data.qpos[3:7] = root_quat
        data.qpos[7:36] = joint_pos
        mujoco.mj_kinematics(model, data)
        wrist_world = torch.tensor(data.xpos[wrist_ids], dtype=torch.float32)
        root_pos_t = torch.tensor(root_pos, dtype=torch.float32).expand(2, -1)
        root_quat_t = torch.tensor(root_quat, dtype=torch.float32).expand(2, -1)
        return quat_apply_inverse(
            _facing_quat(root_quat_t), wrist_world - root_pos_t
        ).numpy()

    def _build_move_hand_hold_cost(self, resource_dir, cfg):
        """Move-only two-wrist dead-zone cost; strike search never uses it."""
        if not bool(cfg.get("enabled", False)):
            return None
        reference_motion = Path(cfg.get("reference_motion", ""))
        if not reference_motion.is_absolute():
            reference_motion = (_REPO / reference_motion).resolve()
        if not reference_motion.exists():
            raise FileNotFoundError(
                f"move_hand_hold.reference_motion not found: {reference_motion}"
            )
        tolerance = float(cfg.get("tolerance_radius", 0.08))
        weight = float(cfg.get("weight", 4.0))
        if tolerance < 0.0 or weight < 0.0:
            raise ValueError(
                "move_hand_hold tolerance_radius and weight must be non-negative"
            )

        desired = self._reference_wrist_hold_position(reference_motion)
        candidate = self._load_or_compute_wrist_root_positions(resource_dir)
        distance = np.linalg.norm(candidate - desired[None, :, :], axis=-1)
        excess = np.maximum(distance - tolerance, 0.0) / max(tolerance, 1e-6)
        cost = (weight * np.square(excess).sum(axis=-1)).astype(np.float32)
        print(
            "[DB] move hand hold: "
            f"ref={reference_motion.name} final frame, radius={tolerance:.3f}m, "
            f"weight={weight:.2f}, target left={desired[0]}, right={desired[1]}",
            end="\r\n",
        )
        return cost

    # ---- official runtime feature weights: (requested / built)^2 per group -------------
    def make_feature_cost_weights(self, requested):
        weights = np.ones(self.feature_dim, dtype=np.float32)
        built = dict(self.FEATURE_WEIGHT_DEFAULTS)
        built.update({k: float(v) for k, v in (self.built_feature_weights or {}).items()})
        for name, (s, e) in self.FEATURE_WEIGHT_SLICES.items():
            if e > self.feature_dim:  # hand groups are absent in legacy 27-dim resources
                continue
            b = max(float(built.get(name, 1.0)), 1e-8)
            req = float(requested.get(name, b))
            ratio = req / b
            weights[s:e] = ratio * ratio
        return weights

    # ---- official query: pose + optional hand blocks copied from the current frame;
    #      traj block from target trajectory (z-up facing -> internal (-y, x)) ----------
    def make_query(self, frame_index, traj_pos_local, traj_dir_local):
        f = int(frame_index)
        # Copy every denormalized feature first. This preserves the official pose
        # continuation query and automatically includes hand[27:33] for the new DB.
        q = (self.features_np[f] * self.features_scale + self.features_offset).astype(np.float32, copy=True)
        for j in range(3):
            dx, dy = float(traj_pos_local[j][0]), float(traj_pos_local[j][1])
            hx, hy = float(traj_dir_local[j][0]), float(traj_dir_local[j][1])
            q[15 + 2 * j] = -dy
            q[16 + 2 * j] = dx
            q[21 + 2 * j] = -hy
            q[22 + 2 * j] = hx
        return q

    def _refresh_weighted_cache(self):
        """GEMV decomposition of the weighted L2 search:
        cost_i = sum_d w_d (f_id - q_d)^2 = sqf_i - 2 f_i . (w*q) + sum_d w_d q_d^2.
        The naive full-matrix diff is much slower; the matvec form keeps both
        27-dim legacy and 33-dim hand-extended search inexpensive."""
        w = self.feature_cost_weights
        self._sqf = np.einsum("td,d,td->t", self.features_np, w, self.features_np).astype(np.float32)
        # static validity prefix per clip is rebuilt in search (depends on ignore_range_end)
        self._valid_cache: dict[int, np.ndarray] = {}

    # ---- official brute-force search (additive transition cost; current frame competes
    #      at no extra cost; range-end and surrounding frames masked out) ---------------
    def search(self, curr_index, query, transition_cost=0.0,
               ignore_range_end=20, ignore_surrounding=20, allowed_mask=None,
               extra_candidate_cost=None):
        qn = ((query - self.features_offset) / self.features_scale).astype(np.float32)
        w = self.feature_cost_weights

        if curr_index >= 0 and (allowed_mask is None or bool(allowed_mask[curr_index])):
            cd = (qn - self.features_np[curr_index]) ** 2
            best_index = int(curr_index)
            best_cost = float(np.sum(cd * w))
            if extra_candidate_cost is not None:
                best_cost += float(extra_candidate_cost[curr_index])
        else:
            best_index = -1
            best_cost = float("inf")

        base_valid = self._valid_cache.get(int(ignore_range_end))
        if base_valid is None:
            base_valid = np.zeros(self.T, dtype=bool)
            for start, stop in zip(self.range_starts, self.range_stops):
                base_valid[start: max(start, stop - ignore_range_end)] = True
            self._valid_cache[int(ignore_range_end)] = base_valid
        valid = base_valid.copy()
        if allowed_mask is not None:
            valid &= allowed_mask
        if curr_index >= 0:
            lo = max(0, curr_index - ignore_surrounding + 1)
            hi = min(self.T, curr_index + ignore_surrounding)
            valid[lo:hi] = False

        wq = (w * qn).astype(np.float32)
        costs = self._sqf - 2.0 * (self.features_np @ wq) + float(np.dot(wq, qn)) + transition_cost
        if extra_candidate_cost is not None:
            if extra_candidate_cost.shape != (self.T,):
                raise ValueError("extra_candidate_cost must have one value per database frame")
            costs = costs + extra_candidate_cost
        costs[~valid] = np.inf
        cand = int(np.argmin(costs))
        if float(costs[cand]) < best_cost:
            best_index, best_cost = cand, float(costs[cand])
        return best_index, best_cost

    def denormalized_features(self, frame_index):
        f = int(frame_index)
        return self.features_np[f] * self.features_scale + self.features_offset

    # ---- skeleton-compat helpers --------------------------------------------------------
    def compute_racket_pos(self, wrist_pos, wrist_quat):
        offset = self.racket_offset.expand_as(wrist_pos)
        return wrist_pos + quat_apply(wrist_quat, offset)

    def advance_frames(self, current_frames):
        next_frames = current_frames + 1
        clip_ends = self.clip_end[current_frames]
        overflow = next_frames > clip_ends
        if overflow.any():
            next_frames[overflow] = self.clip_start[current_frames[overflow]]
        return next_frames, overflow

    def get_clip_for_frame(self, frame_idx):
        f = frame_idx if isinstance(frame_idx, int) else frame_idx.item()
        for start, end, name in self.motion_boundaries:
            if start <= f < end:
                return start, end, name
        return 0, self.T, "unknown"


# =============================================================================
# Config (yaml) + ball sampling
# =============================================================================

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def random_ball_pos(root_pos_w, root_quat_w, ball_sample):
    device = root_pos_w.device

    def _sample(rng, lo, hi):
        a, b = (rng if rng else [lo, hi])
        return a + torch.rand(1, device=device).item() * (b - a)

    radius = ball_sample.get("radius")
    if radius is not None:
        # Disk mode: uniform over a radius-R disk centered on the CURRENT g1 position, clipped to
        # the court rectangle (rejection sampling; the ball must be both within reach AND in-court).
        # ABSOLUTE contact-height range (set from the tennis dataset coverage).
        cx, cy = root_pos_w[0].item(), root_pos_w[1].item()
        court_x = ball_sample.get("court_x", [-11.885, 0.0])   # own half: baseline .. net
        court_y = ball_sample.get("court_y", [-4.115, 4.115])  # singles width
        R = float(radius)
        bx, by = cx, cy
        for _ in range(200):
            r = R * math.sqrt(torch.rand(1, device=device).item())
            th = TWO_PI * torch.rand(1, device=device).item()
            bx, by = cx + r * math.cos(th), cy + r * math.sin(th)
            if court_x[0] <= bx <= court_x[1] and court_y[0] <= by <= court_y[1]:
                break
        else:
            # disk barely overlaps the court: clamp the center-projected point into bounds
            bx = min(max(bx, court_x[0]), court_x[1])
            by = min(max(by, court_y[0]), court_y[1])
        z = _sample(ball_sample.get("z_abs"), 0.55, 1.60)
        return (bx, by, z)

    local_x = _sample(ball_sample.get("x"), 0.3, 1.1)
    local_y = _sample(ball_sample.get("y"), -0.8, 0.2)
    local_z = _sample(ball_sample.get("z"), 0.0, 0.6)
    fq = _facing_quat(root_quat_w.unsqueeze(0))
    world_xy = quat_apply(fq, torch.tensor([[local_x, local_y, 0.0]], device=device))[0]
    base_z = root_pos_w[2].item()
    return (root_pos_w[0] + world_xy[0]).item(), (root_pos_w[1] + world_xy[1]).item(), base_z + local_z


def random_tti(ball_sample, device):
    lo, hi = ball_sample.get("tti", [0.6, 1.8])
    return lo + torch.rand(1, device=device).item() * (hi - lo)


# =============================================================================
# MuJoCo viewer markers (ball + wrist)
# =============================================================================

def add_sphere(viewer, pos, rgba, r=0.05):
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]),
                        np.asarray(pos, dtype=np.float64), np.eye(3).flatten(),
                        np.asarray(rgba, dtype=np.float32))
    viewer.user_scn.ngeom += 1


def add_flat_disc(viewer, pos, rgba, r=0.10, thickness=0.006):
    """Draw a thin horizontal disc for an XY target marker."""
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([r, thickness, 0.0]),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


def add_arrow(viewer, p0, p1, rgba, width=0.012):
    """Draw a fat line (capsule) from p0 to p1 — used for velocity / trajectory vectors."""
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), width,
                         np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64))
    viewer.user_scn.ngeom += 1


def add_direction_arrow(viewer, p0, p1, rgba, width=0.025):
    """Draw a MuJoCo arrow with a visible arrowhead."""
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                        np.zeros(3), np.zeros(9), np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_ARROW), width,
                         np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64))
    viewer.user_scn.ngeom += 1


def draw_dashed_path_with_arrow(
    viewer,
    points,
    rgba,
    *,
    dash_length=0.10,
    gap_length=0.07,
    width=0.009,
    arrow_length=0.22,
    arrow_spacing=0.50,
    arrow_rgba=None,
):
    """Draw a dashed 3-D path with sparse arrowheads indicating travel direction."""
    path = np.asarray(points, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 3:
        return

    drawing = True
    phase_remaining = float(dash_length)
    for start, end in zip(path[:-1], path[1:]):
        delta = end - start
        segment_length = float(np.linalg.norm(delta))
        if segment_length <= 1.0e-8:
            continue
        direction = delta / segment_length
        cursor = start.copy()
        remaining = segment_length
        while remaining > 1.0e-8:
            step = min(remaining, phase_remaining)
            next_cursor = cursor + direction * step
            if drawing:
                add_arrow(viewer, cursor, next_cursor, rgba, width=width)
            cursor = next_cursor
            remaining -= step
            phase_remaining -= step
            if phase_remaining <= 1.0e-8:
                drawing = not drawing
                phase_remaining = float(dash_length if drawing else gap_length)

    segments = path[1:] - path[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 1.0e-4:
        return

    spacing = max(float(arrow_spacing), float(arrow_length) * 1.5)
    arrow_distances = list(np.arange(spacing, total_length, spacing))
    # Always mark the current endpoint, but avoid placing two arrowheads very close together.
    if not arrow_distances or total_length - arrow_distances[-1] >= 0.5 * spacing:
        arrow_distances.append(total_length)
    else:
        arrow_distances[-1] = total_length

    for distance in arrow_distances:
        segment_idx = min(
            int(np.searchsorted(cumulative, distance, side="right") - 1),
            len(segments) - 1,
        )
        while segment_idx >= 0 and lengths[segment_idx] <= 1.0e-8:
            segment_idx -= 1
        if segment_idx < 0:
            continue
        direction = segments[segment_idx] / lengths[segment_idx]
        tip = path[segment_idx] + direction * (distance - cumulative[segment_idx])
        add_direction_arrow(
            viewer,
            tip - direction * float(arrow_length),
            tip,
            rgba if arrow_rgba is None else arrow_rgba,
            width=max(width * 2.2, 0.018),
        )


def add_axes(viewer, origin=(0.0, 0.0, 0.0), length=0.5, width=0.012):
    """Draw a world (or any) coordinate frame: X=red, Y=green, Z=blue, from `origin`."""
    o = np.asarray(origin, dtype=np.float64)
    for vec, rgba in (([length, 0, 0], [1, 0, 0, 1]),
                      ([0, length, 0], [0, 1, 0, 1]),
                      ([0, 0, length], [0, 0, 1, 1])):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.zeros(9), np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), width,
                             o, o + np.asarray(vec, dtype=np.float64))
        viewer.user_scn.ngeom += 1


# =============================================================================
# Episode collection (npz format compatible with motion/djvkovic_npz_data_v5)
# =============================================================================
# One episode = full move + strike run (serve -> approach -> cut-in -> swing end).
# Saved fields match v5 EXACTLY, including the 34-body layout (fps/joint_pos/joint_vel/
# body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w + <name>.json {"strike_frame": k}).
# The recorder FKs the 30 g1.xml bodies each frame; at save time augment_30_to_34 adds
# the 4 fixed frames (pelvis_contour/logo/head/racket_link) using constant transforms
# derived from v5 itself — so episodes feed the training MotionCommand loader
# (G1_CYLINDER, 34 bodies) directly.

EPISODE_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]


_AUGMENT_34_PATH = _REPO / "motion_db/g1_racket_34body_augment.npz"
_augment_34_cache = None


def _np_qmul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def _np_qrot(q, v):
    t = 2.0 * np.cross(q[..., 1:], v)
    return v + q[..., 0:1] * t + np.cross(q[..., 1:], t)


def augment_30_to_34(body_pos30, body_quat30):
    """Lift 30-body FK arrays to the v5 34-body layout (G1_CYLINDER / g1_racket_29).

    The 4 extra bodies are FIXED frames (pelvis_contour, logo, head, racket_link) whose
    parent + constant transforms were derived numerically from v5 itself
    (motion_db/g1_racket_34body_augment.npz; round-trip error 0.00 mm on held-out clips).
    """
    global _augment_34_cache
    if _augment_34_cache is None:
        a = np.load(_AUGMENT_34_PATH)
        _augment_34_cache = (a["base_place"].astype(int), a["extra_idx"].astype(int),
                             a["extra_parent"].astype(int),
                             a["extra_p_off"].astype(np.float64), a["extra_q_off"].astype(np.float64))
    base_place, extra_idx, extra_parent, extra_p, extra_q = _augment_34_cache
    T = body_pos30.shape[0]
    P = np.zeros((T, 34, 3), dtype=np.float64)
    Q = np.zeros((T, 34, 4), dtype=np.float64)
    for j30, i34 in base_place:
        P[:, i34] = body_pos30[:, j30]
        Q[:, i34] = body_quat30[:, j30]
    for k, i in enumerate(extra_idx):
        pj = int(extra_parent[k])
        P[:, i] = body_pos30[:, pj] + _np_qrot(body_quat30[:, pj], extra_p[k][None, :])
        Q[:, i] = _np_qmul(body_quat30[:, pj], np.broadcast_to(extra_q[k], (T, 4)))
    return P, Q


def augment_30_to_34_angvel(body_angvel30):
    """Lift 30-body world angular velocities to the G1_CYLINDER 34-body layout.

    The four augmented bodies are rigidly fixed to their parents, hence they
    have exactly the same world angular velocity as their parent body.
    """
    global _augment_34_cache
    if _augment_34_cache is None:
        a = np.load(_AUGMENT_34_PATH)
        _augment_34_cache = (a["base_place"].astype(int), a["extra_idx"].astype(int),
                             a["extra_parent"].astype(int),
                             a["extra_p_off"].astype(np.float64), a["extra_q_off"].astype(np.float64))
    base_place, extra_idx, extra_parent, _, _ = _augment_34_cache
    T = body_angvel30.shape[0]
    A = np.zeros((T, 34, 3), dtype=np.float64)
    for j30, i34 in base_place:
        A[:, i34] = body_angvel30[:, j30]
    for k, i34 in enumerate(extra_idx):
        A[:, i34] = body_angvel30[:, int(extra_parent[k])]
    return A


def save_episode_npz(path, fps, joint_pos, joint_vel, body_pos, body_quat,
                     body_angvel, root_linvel, root_angvel, strike_frame, meta=None):
    """Save one collected episode in the djvkovic_npz_data_v5 layout (34 bodies).

    The recorder captures the 30 FK-able bodies; augment_30_to_34 lifts them to the
    full 34-body training layout. Linear velocities use finite differences;
    angular velocities are sampled from MuJoCo's world-frame body cvel after
    qvel has been written, with fixed augmented bodies inheriting their parent.
    """
    path = Path(path)
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    joint_vel = np.asarray(joint_vel, dtype=np.float32)
    body_pos30 = np.asarray(body_pos, dtype=np.float64)
    body_quat30 = np.asarray(body_quat, dtype=np.float64)
    body_pos, body_quat = augment_30_to_34(body_pos30, body_quat30)
    body_angvel = augment_30_to_34_angvel(np.asarray(body_angvel, dtype=np.float64))
    body_pos = body_pos.astype(np.float32)
    body_quat = body_quat.astype(np.float32)
    T = body_pos.shape[0]
    if T > 1:
        body_lin_vel = (np.gradient(body_pos.astype(np.float64), axis=0) * float(fps)).astype(np.float32)
    else:
        body_lin_vel = np.zeros_like(body_pos)
    body_ang_vel = body_angvel.astype(np.float32)
    body_lin_vel[:, 0] = np.asarray(root_linvel, dtype=np.float32)
    body_ang_vel[:, 0] = np.asarray(root_angvel, dtype=np.float32)
    np.savez(
        path,
        fps=np.asarray([int(fps)], dtype=np.int64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
    )
    info = {"strike_frame": int(strike_frame)}
    if meta:
        info.update(meta)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    return path


# =============================================================================
# Gamepad control mode (official simulation-object springs, z-up)
# =============================================================================
# Ported from the official Motion-Matching runtime: a spring-damped "simulation
# object" follows the desired stick velocity/heading, and the query trajectory is
# the ANALYTIC spring prediction at +20/40/60 frames (the official
# simulation_positions_update / simple_spring_damper formulas evaluated at t_i —
# exact for any t, so no per-step integration is needed).

def _halflife_to_damping(halflife, eps=1e-5):
    return (4.0 * 0.69314718056) / (halflife + eps)


def _spring_vel_update(vel, acc, desired_vel, halflife, t):
    """Official simulation_positions_update, z-up 2D, returning (dpos, vel, acc) at time t."""
    y = _halflife_to_damping(halflife) / 2.0
    j0 = vel - desired_vel
    j1 = acc + j0 * y
    eydt = math.exp(-y * t)
    dpos = (eydt * (((-j1) / (y * y)) + ((-j0 - j1 * t) / y))
            + (j1 / (y * y)) + j0 / y + desired_vel * t)
    new_vel = eydt * (j0 + j1 * t) + desired_vel
    new_acc = eydt * (acc - j1 * y * t)
    return dpos, new_vel, new_acc


def _spring_scalar_update(x, v, goal, halflife, t):
    """Official simple_spring_damper_exact_scalar at time t."""
    y = _halflife_to_damping(halflife) / 2.0
    j0 = x - goal
    j1 = v + j0 * y
    eydt = math.exp(-y * t)
    return eydt * (j0 + j1 * t) + goal, eydt * (v - j1 * y * t)


class GamepadInput:
    """GLFW joystick polling (trimmed port of the official GamepadController).

    Left stick = move, right stick X = turn (direct yaw), A = walk toggle-while-held,
    LT = strafe (heading locked while translating). Returns None when no pad is found —
    keyboard commands stay active.
    """

    def __init__(self, index=0, deadzone=0.2):
        self.index = index
        self.deadzone = deadzone
        self.jid = None
        self.next_scan = 0.0
        try:
            import glfw
            self.glfw = glfw
            if not glfw.init():
                self.glfw = None
        except Exception:
            self.glfw = None
        if self.glfw is not None:
            self._scan(True)

    def _scan(self, verbose=False):
        g = self.glfw
        for jid in range(getattr(g, "JOYSTICK_1", 0), getattr(g, "JOYSTICK_LAST", 15) + 1):
            if g.joystick_present(jid):
                self.jid = jid
                name = g.get_joystick_name(jid)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                print(f"[GAMEPAD] using joystick {jid}: {name}", end="\r\n")
                return
        self.jid = None
        self.next_scan = time.perf_counter() + 1.0
        if verbose:
            print("[GAMEPAD] no joystick found; keyboard WASD stays active", end="\r\n")

    def _dz(self, v):
        return 0.0 if abs(v) < self.deadzone else float(v)

    @staticmethod
    def _deref(ret):
        """pyGLFW joystick getters return a (C-pointer, count) tuple — deref to a list."""
        if ret is None:
            return []
        try:
            ptr, count = ret
            return [ptr[i] for i in range(int(count))]
        except (TypeError, ValueError):
            return list(ret)   # some builds return a plain sequence already

    def poll(self):
        """Returns (stick_left[2], turn_axis, walk, strafe) or None if no pad."""
        if self.glfw is None:
            return None
        if self.jid is None or not self.glfw.joystick_present(self.jid):
            if time.perf_counter() >= self.next_scan:
                self._scan()
            if self.jid is None:
                return None
        try:
            axes = self._deref(self.glfw.get_joystick_axes(self.jid))
            buttons = self._deref(self.glfw.get_joystick_buttons(self.jid))
        except Exception:
            self.jid = None
            return None
        if len(axes) < 2:
            return None
        # GLFW axes: [0]=LX (right +), [1]=LY (down +). pad_stick semantics: (forward, left).
        #   forward = -LY (push up), left = -LX (push left)
        fwd = -self._dz(float(axes[1]))
        left = -self._dz(float(axes[0]))
        # raw Linux Xbox mapping: LX, LY, LT, RX, RY, RT (official port's fallback indices)
        rx = self._dz(float(axes[3])) if len(axes) > 3 else 0.0
        lt = float(axes[2]) if len(axes) > 2 else -1.0
        walk = bool(buttons[0]) if len(buttons) > 0 else False
        n = math.hypot(fwd, left)
        if n > 1.0:
            fwd, left = fwd / n, left / n
        return np.array([fwd, left], dtype=np.float64), float(rx), walk, bool(lt > 0.25)


# =============================================================================
# Main loop
# =============================================================================

def run(args, cfg, keyboard=None):
    device = torch.device("cpu")
    N = 1  # single MuJoCo model
    clean_view = bool(getattr(args, "clean_view", False))
    if clean_view:
        # Keep only the live G1 model, tennis court, and strike target.  This
        # deliberately overrides visualization flags supplied by the YAML or
        # command line.
        args.ghost = False
        args.speed_plot = False

    # Official search cadence/cost parameters (mm_config.yaml semantics).
    rematch_interval = cfg.get("rematch_interval", 5)               # frames between searches (5 @ 50fps = official search_time 0.1s)
    transition_cost = float(cfg.get("transition_cost", 0.0))        # ADDITIVE on candidates (official; current frame competes at no cost)
    ignore_range_end = int(cfg.get("ignore_range_end", 20))         # official: last N frames of each clip unmatchable
    ignore_surrounding = int(cfg.get("ignore_surrounding", 20))     # official: +-N frames around the current frame unmatchable
    move_xy_tolerance = cfg.get("move_xy_tolerance", 0.20)          # fallback stop tolerance when no strike entry exists
    blend_frames = cfg.get("blend_frames", 20)
    weights_cfg = cfg.get("weights", {})                            # official groups + optional hand_position/hand_velocity
    ball_sample = cfg.get("ball_sample", {})
    ball_default = cfg.get("ball_default", {})
    reset_pose = cfg.get("reset_pose", {})
    court_cfg = cfg.get("court", {}) or {}
    target_traj_cfg = cfg.get("target_trajectory", {}) or {}
    incoming_ball_cfg = cfg.get("incoming_ball_visualization", {}) or {}
    strike_entry_cfg = cfg.get("strike_entry", {}) or {}
    strike_entry_enabled = bool(strike_entry_cfg.get("enabled", True))
    entry_min_lead_frames = max(1, int(strike_entry_cfg.get("min_lead_frames", 10)))
    entry_max_search_frames = max(1, int(strike_entry_cfg.get("max_search_frames", 45)))
    entry_predict_move_frames = max(0, int(strike_entry_cfg.get("predict_move_frames", 20)))
    entry_w_key_pose = float(strike_entry_cfg.get("w_key_pose", 2.0))
    entry_w_joint_pose = float(strike_entry_cfg.get("w_joint_pose", 0.25))
    entry_w_yaw = float(strike_entry_cfg.get("w_yaw", 0.75))
    entry_w_clip_start = float(strike_entry_cfg.get("w_clip_start", 0.03))
    entry_w_contact_z = float(strike_entry_cfg.get("w_contact_z", 8.0))
    entry_w_contact_xy = float(strike_entry_cfg.get("w_contact_xy", 0.1))
    entry_switch_xy_threshold = float(strike_entry_cfg.get("switch_xy_threshold", move_xy_tolerance))
    entry_switch_yaw_threshold = math.radians(float(strike_entry_cfg.get("switch_yaw_threshold_deg", 35.0)))
    entry_switch_timeout_s = max(0.0, float(strike_entry_cfg.get("switch_timeout_s", 1.0)))
    entry_blend_frames = max(1, int(strike_entry_cfg.get("entry_blend_frames", 25)))  # longer blend just for the strike cut-in
    strike_postcontact_frames = max(0, int(strike_entry_cfg.get("postcontact_frames", 0)))
    incoming_ball_viz = bool(incoming_ball_cfg.get("enabled", True))
    incoming_ball_start_distance = max(
        0.1,
        float(incoming_ball_cfg.get("start_distance", 4.0)),
    )
    incoming_ball_start_height = float(incoming_ball_cfg.get("start_height_offset", 0.6))
    incoming_ball_arc_height = max(0.0, float(incoming_ball_cfg.get("arc_height", 0.8)))
    incoming_ball_samples = max(8, int(incoming_ball_cfg.get("samples", 49)))

    min_neck_height = float(cfg.get("min_neck_height", 0.0))       # move filter: drop low-posture frames (0 = off)
    move_hand_hold_cfg = cfg.get("move_hand_hold", {}) or {}
    db = OfficialMMDatabase(args.db_file, device, weights_cfg=weights_cfg,
                            min_neck_height=min_neck_height,
                            move_hand_hold_cfg=move_hand_hold_cfg)
    if not db._has_lafan_loco:
        raise RuntimeError("move stage requires locomotion clips in the official resources")
    blender = InertializationBlender(blend_frames, db.fps, N, device)
    target_cfg_path = Path(target_traj_cfg.get("config", TARGET_TRAJ_DEFAULT_CONFIG))
    if not target_cfg_path.is_absolute():
        target_cfg_path = (Path(__file__).resolve().parents[1] / target_cfg_path).resolve()
    target_raw_cfg = load_target_yaml_config(target_cfg_path)
    target_sampling_cfg = target_sampling_config_from_yaml(target_raw_cfg.get("sampling", {}))
    target_sampling_cfg = replace(target_sampling_cfg, fps=float(db.fps))
    target_ctoc_cfg = _ctoc_config_from_yaml_x(target_raw_cfg)
    # "ct_oc_casadi" (two-mode CT-OC via IPOPT, default) or "optimized_omni" (old SLSQP model).
    target_planner = str(target_traj_cfg.get("planner", "ct_oc_casadi"))
    target_yaw_world = float(target_traj_cfg.get("default_yaw", 0.0))
    mouse_select_yaw = bool(target_traj_cfg.get("mouse_select_yaw", False))  # False: click picks XY only, yaw=default_yaw, no yaw arrow
    target_traj = None
    target_traj_origin_xy = np.zeros(2, dtype=np.float64)
    target_traj_origin_yaw = 0.0
    target_traj_start_step = 0
    target_traj_phase = 0.0                          # playback phase along the planned trajectory
    traj_lag_tolerance = float(target_traj_cfg.get("lag_tolerance", 0.5))
    traj_closed_loop = bool(target_traj_cfg.get("closed_loop", False))  # False = open-loop (time-driven phase)
    query_pred_traj = torch.zeros(N, len(db.traj_offsets), 3, device=device)
    query_pred_headings = torch.zeros(N, len(db.traj_offsets), 2, device=device)
    query_desired_vel = torch.zeros(N, 3, device=device)
    print(f"[TRAJ] using target trajectory config: {target_cfg_path}", end="\r\n")

    # MuJoCo model
    model, uses_shared_court = build_mujoco_model(court_cfg)
    mj_data = mujoco.MjData(model)
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    assert wrist_id >= 0, "right_wrist_yaw_link not found in g1.xml"
    # Height reference = mid-point of the two shoulder_pitch links ("neck"). They are the
    # roots of the arm chains, so the point is unaffected by arm/racket motion, and being
    # the highest rigid frame it reflects crouch/lean height changes more directly than torso.
    _lsh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link")
    _rsh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link")
    assert _lsh_id >= 0 and _rsh_id >= 0, "shoulder_pitch links not found in g1.xml"

    def neck_pos():
        return 0.5 * (mj_data.xpos[_lsh_id] + mj_data.xpos[_rsh_id])

    # logging switches (yaml `logging:` section)
    log_cfg = cfg.get("logging", {}) or {}
    log_matches = bool(log_cfg.get("match_log", False))          # per-search MM line (off by default)
    viz_mode_switch_radius = bool(
        log_cfg.get("mode_switch_radius_viz", log_cfg.get("ball_range_viz", True))
    )  # near/far planning boundary around the current robot root
    viz_root_heading = bool(log_cfg.get("root_heading_viz", False))
    viz_motion_matching = bool(log_cfg.get("motion_matching_viz", False))
    viz_root_lag = bool(log_cfg.get("root_lag_viz", True))        # query root (traj phase) vs robot root
    log_height = bool(log_cfg.get("height_log", True))           # periodic neck world position
    log_height_interval = max(1, int(log_cfg.get("height_log_interval", 25)))  # frames (25 = 0.5s @ 50fps)
    viz_height = bool(log_cfg.get("height_viz", True))           # neck marker in the viewer
    if clean_view:
        viz_mode_switch_radius = False
        viz_root_heading = False
        viz_motion_matching = False
        viz_root_lag = False
        viz_height = False
    assert model.nq >= 36, f"expected >=36 qpos (7 free + 29 joints), got {model.nq}"

    ball_x = ball_default.get("x", 0.5)
    ball_y = ball_default.get("y", -0.3)
    ball_z = ball_default.get("z", 0.9)
    time_remaining = ball_default.get("time_remaining", 1.0)
    move_target_xy_world = np.array([ball_x, ball_y], dtype=np.float64)
    move_target_yaw_world = float(target_yaw_world)
    move_target_vel_world = np.zeros(2, dtype=np.float64)
    move_target_yaw_rate = 0.0
    selected_entry = None
    strike_active = False
    strike_completed = False

    current_frame = torch.zeros(N, dtype=torch.long, device=device)
    root_pos_offset = torch.zeros(N, 3, device=device)
    root_yaw_offset = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(N, -1).clone()
    locked_side = torch.zeros(N, dtype=torch.int8, device=device)

    # "currently applied" state (Isaac-Lab joint order) — replaces robot.data.* reads.
    cur = {}

    def apply_frame(frame_indices):
        """Replaces set_robot_states: db frame + offsets + blend -> mj qpos (remapped) -> mj_forward.
        Also stores the applied state into `cur` (IL order) for the next blend trigger."""
        root_pos = db.body_pos_w[frame_indices, db.root_idx].clone()
        root_quat = db.body_quat_w[frame_indices, db.root_idx].clone()
        root_linvel = db.body_linvel[frame_indices, db.root_idx].clone()
        root_angvel = db.body_angvel[frame_indices, db.root_idx].clone()
        joint_pos = db.joint_pos[frame_indices].clone()
        joint_vel = db.joint_vel[frame_indices].clone()

        root_pos = quat_apply(root_yaw_offset, root_pos)
        root_quat = quat_mul(root_yaw_offset, root_quat)
        root_linvel = quat_apply(root_yaw_offset, root_linvel)
        root_angvel = quat_apply(root_yaw_offset, root_angvel)
        root_pos = root_pos + root_pos_offset

        root_pos, root_quat, root_linvel, root_angvel, joint_pos, joint_vel = blender.apply(
            root_pos, root_quat, root_linvel, root_angvel, joint_pos, joint_vel)

        # write to MuJoCo (env 0). joint order IL(BFS) -> MuJoCo(DFS).
        # Guard: a non-finite pose (NaN from a degenerate blend/match) fed to mj_forward or
        # the renderer hard-segfaults MuJoCo. Skip the write instead of crashing.
        if not (torch.isfinite(root_pos[0]).all() and torch.isfinite(root_quat[0]).all()
                and torch.isfinite(joint_pos[0]).all()):
            # Keep the last good `cur`; do NOT propagate NaN (it would poison the next blend,
            # the ball sampling and the markers -> renderer segfault).
            print(f"[ERROR] non-finite pose at frame {int(frame_indices[0])} "
                  f"(blend/match produced NaN) — kept last good pose", end="\r\n")
            return

        jp = joint_pos[0].cpu().numpy()
        mj_data.qpos[:3] = root_pos[0].cpu().numpy()
        mj_data.qpos[3:7] = root_quat[0].cpu().numpy()
        mj_data.qpos[7:36] = jp[ISAACLAB_TO_MUJOCO]
        # MuJoCo derives per-body cvel from qvel.  Populate the full free-joint
        # and articulated velocity state so collection can save body angular
        # velocities instead of only the root angular velocity.
        jv = joint_vel[0].cpu().numpy()
        mj_data.qvel[:3] = root_linvel[0].cpu().numpy()
        mj_data.qvel[3:6] = root_angvel[0].cpu().numpy()
        mj_data.qvel[6:35] = jv[ISAACLAB_TO_MUJOCO]
        mujoco.mj_forward(model, mj_data)

        cur.update(root_pos=root_pos, root_quat=root_quat, root_linvel=root_linvel,
                   root_angvel=root_angvel, joint_pos=joint_pos, joint_vel=joint_vel)

    def wrist_world():
        """Right-wrist (body 29) world pos/quat from MuJoCo FK (last mj_forward)."""
        wp = torch.tensor(mj_data.xpos[wrist_id], dtype=torch.float32, device=device).unsqueeze(0)
        wq = torch.tensor(mj_data.xquat[wrist_id], dtype=torch.float32, device=device).unsqueeze(0)
        return wp, wq

    def compute_ball_and_racket_local():
        ball_world = torch.tensor([[ball_x, ball_y, ball_z]], device=device).expand(N, -1)
        root_pos_w = cur["root_pos"]
        fq = _facing_quat(cur["root_quat"])
        ball_local = quat_apply_inverse(fq, ball_world - root_pos_w)
        wp, wq = wrist_world()
        racket_pos_w = db.compute_racket_pos(wp, wq)
        racket_local = quat_apply_inverse(fq, racket_pos_w - root_pos_w)
        return ball_local, racket_local, ball_local.clone()

    def compute_move_target_local():
        target_world = torch.tensor(
            [[move_target_xy_world[0], move_target_xy_world[1], cur["root_pos"][0, 2].item()]],
            dtype=torch.float32, device=device,
        ).expand(N, -1)
        root_pos_w = cur["root_pos"]
        fq = _facing_quat(cur["root_quat"])
        return quat_apply_inverse(fq, target_world - root_pos_w)

    def reset_to_fixed_pose():
        nonlocal ball_x, ball_y, ball_z, time_remaining
        rx = reset_pose.get("x", 0.0)
        ry = reset_pose.get("y", 0.0)
        ryaw = reset_pose.get("yaw", 0.0)
        rframe = max(0, min(int(reset_pose.get("frame", 0)), db.T - 1))
        current_frame.fill_(rframe)
        half = 0.5 * ryaw
        target_quat = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]], device=device).expand(N, -1).clone()
        target_pos = torch.tensor([[rx, ry, 0.0]], device=device).expand(N, -1).clone()
        new_root_pos_db = db.body_pos_w[current_frame, db.root_idx].clone()
        new_root_quat_db = db.body_quat_w[current_frame, db.root_idx].clone()
        new_fq = _facing_quat(new_root_quat_db)
        root_yaw_offset[:] = quat_mul(target_quat, quat_inv(new_fq))
        rotated = quat_apply(root_yaw_offset, new_root_pos_db)
        off = target_pos - rotated
        off[:, 2] = 0.0
        root_pos_offset[:] = off
        blender.timer[:] = blender.blend_duration  # cancel pending blend
        apply_frame(current_frame)

    def lock_side():
        ball_local_now, _, _ = compute_ball_and_racket_local()
        locked_side.copy_(torch.sign(ball_local_now[:, 1]).to(torch.int8))

    def align_current_yaw_to_world(yaw_world=0.0):
        """Face the currently played frame toward a world yaw while keeping root XY fixed."""
        half = 0.5 * float(yaw_world)
        target_quat = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]],
                                   device=device).expand(N, -1).clone()
        root_pos_db = db.body_pos_w[current_frame, db.root_idx].clone()
        root_quat_db = db.body_quat_w[current_frame, db.root_idx].clone()
        root_yaw_offset[:] = quat_mul(target_quat, quat_inv(_facing_quat(root_quat_db)))
        rotated = quat_apply(root_yaw_offset, root_pos_db)
        off = cur["root_pos"] - rotated
        off[:, 2] = 0.0
        root_pos_offset[:] = off

    def yaw_from_quat_tensor(q):
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def rot2_np(yaw):
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def sample_ball_z_world():
        rng = ball_sample.get("z")
        if rng:
            lo, hi = float(rng[0]), float(rng[1])
            local_z = lo + torch.rand(1, device=device).item() * (hi - lo)
        else:
            local_z = float(ball_default.get("z", 0.9)) - float(cur["root_pos"][0, 2].item())
        return float(cur["root_pos"][0, 2].item() + local_z)

    def rebuild_target_trajectory(reset_timer=True, reason="target"):
        nonlocal target_traj, target_traj_origin_xy, target_traj_origin_yaw, target_traj_start_step, time_remaining
        nonlocal target_traj_phase
        root_xy = cur["root_pos"][0, :2].detach().cpu().numpy().astype(np.float64)
        root_yaw = float(yaw_from_quat_tensor(_facing_quat(cur["root_quat"]))[0].item())
        R_inv = rot2_np(-root_yaw)
        p1_local = R_inv @ (np.asarray(move_target_xy_world, dtype=np.float64) - root_xy)
        yaw1_local = float(target_wrap_angle(move_target_yaw_world - root_yaw))
        v1_local = R_inv @ np.asarray(move_target_vel_world, dtype=np.float64)
        boundary = TargetBoundaryCondition(
            p0=np.zeros(2, dtype=np.float64),
            yaw0=0.0,
            v0=np.zeros(2, dtype=np.float64),
            yaw_rate0=0.0,
            p1=p1_local,
            yaw1=yaw1_local,
            v1=v1_local,
            yaw_rate1=float(move_target_yaw_rate),
            duration=float(target_sampling_cfg.duration_range[1]),
            lateral_offset=0.0,
        )
        t0 = time.perf_counter()
        if target_planner == "ct_oc_casadi":
            # Two-mode CT-OC (translation / turn_run) solved with CasADi+IPOPT, minimizing duration
            # within the free duration_range (shortest feasible move). Boundary is in the current
            # root-facing frame (p0=0, yaw0=0) — the planner is frame-agnostic.
            plan = plan_mode_aware_ct_oc_casadi(boundary, target_sampling_cfg, target_ctoc_cfg, deadline=None)
            target_traj = plan.trajectory
            plan_note = f"planner=ct_oc_casadi mode={plan.selected_mode} fallback={plan.fallback_used}"
        else:
            target_traj = generate_target_trajectory(
                boundary,
                fps=float(db.fps),
                model=target_sampling_cfg.trajectory_model,
                cfg=target_sampling_cfg,
            )
            plan_note = f"model={target_sampling_cfg.trajectory_model}"
        target_traj_origin_xy = root_xy
        target_traj_origin_yaw = root_yaw
        if reset_timer:
            target_traj_start_step = sim_step
            target_traj_phase = 0.0
            time_remaining = float(target_traj.t[-1])
        print(
            f"[TRAJ {reason}] {plan_note} "
            f"move_target=({move_target_xy_world[0]:.2f},{move_target_xy_world[1]:.2f}) "
            f"ball=({ball_x:.2f},{ball_y:.2f},{ball_z:.2f}) "
            f"yaw={math.degrees(move_target_yaw_world):.1f}deg "
            f"v1=({move_target_vel_world[0]:.2f},{move_target_vel_world[1]:.2f}) "
            f"T={target_traj.t[-1]:.2f}s "
            f"opt={(time.perf_counter() - t0):.2f}s",
            end="\r\n",
        )

    def sample_target_query(elapsed):
        nonlocal query_pred_traj, query_pred_headings, query_desired_vel
        if target_traj is None:
            return query_pred_traj, query_pred_headings, query_desired_vel

        cur_t = min(max(float(elapsed), 0.0), float(target_traj.t[-1]))
        now_pos, now_vel, _ = interp_target_traj(target_traj, np.asarray([cur_t], dtype=np.float64))
        R_origin = rot2_np(target_traj_origin_yaw)
        vel_world_xy = R_origin @ now_vel[0]
        vel_world = torch.tensor([[vel_world_xy[0], vel_world_xy[1], 0.0]],
                                 dtype=torch.float32, device=device)
        cur_fq = _facing_quat(cur["root_quat"])
        query_desired_vel = quat_apply_inverse(cur_fq, vel_world)

        offsets = np.asarray(db.traj_offsets, dtype=np.float64)
        sample_t = cur_t + offsets / float(db.fps)
        fut_pos, _, fut_yaw = interp_target_traj(target_traj, sample_t)
        fut_world_xy = target_traj_origin_xy[None, :] + (R_origin @ fut_pos.T).T
        root_pos = cur["root_pos"][0].detach().cpu().numpy().astype(np.float64)
        fut_world = torch.tensor(
            np.column_stack([fut_world_xy, np.full(len(fut_world_xy), root_pos[2], dtype=np.float64)]),
            dtype=torch.float32, device=device,
        ).unsqueeze(0)
        root_pos_t = cur["root_pos"][:, None, :]
        query_pred_traj = quat_apply_inverse(cur_fq[:, None, :], fut_world - root_pos_t)

        world_yaw = target_traj_origin_yaw + fut_yaw
        headings_world = torch.tensor(
            np.stack([np.cos(world_yaw), np.sin(world_yaw), np.zeros_like(world_yaw)], axis=-1),
            dtype=torch.float32, device=device,
        ).unsqueeze(0)
        query_pred_headings = quat_apply_inverse(cur_fq[:, None, :], headings_world)[..., :2]
        return query_pred_traj, query_pred_headings, query_desired_vel

    def target_traj_elapsed():
        if target_traj is None:
            return 0.0
        return min(max((sim_step - target_traj_start_step) * frame_dt, 0.0), float(target_traj.t[-1]))

    def advance_traj_phase():
        """Trajectory playback phase.
        Open-loop (default): advance by wall-clock every frame, so the actual move time equals
        the planned trajectory duration (needed for total-time control).
        Closed-loop (opt-in, target_trajectory.closed_loop=true): advance only while the robot
        stays within lag_tolerance of the current path point — keeps the query's future-sample
        deltas in-distribution when the matched motion can't keep up, at the cost of stretching
        the move time."""
        nonlocal target_traj_phase
        if target_traj is None:
            return 0.0
        t_end = float(target_traj.t[-1])
        if traj_closed_loop:
            pos_local, _, _ = interp_target_traj(target_traj, np.asarray([target_traj_phase], dtype=np.float64))
            path_xy = target_traj_origin_xy + rot2_np(target_traj_origin_yaw) @ pos_local[0]
            root_xy = cur["root_pos"][0, :2].detach().cpu().numpy().astype(np.float64)
            if float(np.linalg.norm(path_xy - root_xy)) > traj_lag_tolerance:
                return target_traj_phase          # robot lagging: hold phase until it catches up
        target_traj_phase = min(target_traj_phase + frame_dt, t_end)
        return target_traj_phase

    ghost_query_root_pos = torch.zeros(N, 3, device=device)
    ghost_query_root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(N, -1).clone()
    ghost_snapshot_interval = max(1, int(log_cfg.get("ghost_snapshot_interval", 20)))
    ghost_strike_snapshot_interval = max(1, int(log_cfg.get("ghost_strike_snapshot_interval", 40)))
    ghost_strike_precontact_frames = max(0, int(log_cfg.get("ghost_strike_precontact_frames", 2)))
    ghost_loco_min_distance = max(0.0, float(log_cfg.get("ghost_loco_min_distance", 0.3)))
    executed_ghost_qpos: list[np.ndarray] = []
    executed_ghost_is_strike: list[bool] = []
    racket_center_path_viz = (
        bool(log_cfg.get("racket_center_path_viz", True)) and not clean_view
    )
    racket_center_path: list[np.ndarray] = []
    ghost_last_snapshot_step = -ghost_snapshot_interval
    mouse_state = {
        "manual_edit": False,
        "dragging_target": False,
        "draft_xy": None,
        "draft_yaw": None,
        "pending": None,
        "window": None,
        "window_addr": None,
        "raw": None,                          # independent ctypes handle to the GLFW lib
        "prev_mouse_button_callback": None,   # RAW C fn pointers (viewer's camera handlers)
        "prev_cursor_pos_callback": None,
    }
    pick_opt = mujoco.MjvOption()
    pick_scn = mujoco.MjvScene(model, maxgeom=max(model.ngeom + 16, 128))

    def capture_ghost_query_frame():
        """Capture the task frame and reset executed-motion snapshots for a new target."""
        nonlocal ghost_last_snapshot_step
        ghost_query_root_pos.copy_(cur["root_pos"])
        ghost_query_root_quat.copy_(cur["root_quat"])
        executed_ghost_qpos.clear()
        executed_ghost_is_strike.clear()
        racket_center_path.clear()
        ghost_last_snapshot_step = sim_step - ghost_snapshot_interval

    # ---- Strike clip index + entry planner ---------------------------------
    # A strike clip is aligned in two steps:
    #   1. rotate the racket-head velocity at contact so the outgoing swing is world +X;
    #   2. translate XY so the racket sweet spot at contact lands on the sampled ball XY.
    # Z is deliberately not translated; ball height selects the clip.
    strike_data = mujoco.MjData(model)
    strike_wrist = wrist_id
    strike_off = db.racket_offset.cpu().numpy()

    def yaw_quat_torch(yaw):
        return torch.tensor([[math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]],
                            dtype=torch.float32, device=device)

    def _strike_pose_qpos(g):
        strike_data.qpos[:3] = db.body_pos_w[g, 0].cpu().numpy()
        strike_data.qpos[3:7] = db.body_quat_w[g, 0].cpu().numpy()
        strike_data.qpos[7:36] = db.joint_pos[g].cpu().numpy()[ISAACLAB_TO_MUJOCO]
        mujoco.mj_forward(model, strike_data)

    def strike_racket_world(g):
        # Per-frame racket sweet spot is precomputed in the official resources
        # (strike_metadata.npz, converted to z-up at load) — no FK needed.
        return db.racket_pos_w[int(g)].cpu().numpy()

    def selected_strike_point_world():
        """Return the aligned racket-center position at the selected strike frame."""
        if selected_entry is None:
            return np.array([ball_x, ball_y, ball_z], dtype=np.float64)
        strike_frame = int(selected_entry["clip"]["strike"])
        racket_ref = torch.tensor(
            strike_racket_world(strike_frame),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        strike_point = (
            quat_apply(selected_entry["q_off"], racket_ref)
            + selected_entry["translation"]
        )[0]
        return strike_point.detach().cpu().numpy().astype(np.float64)

    def incoming_ball_path():
        """Construct a purely visual incoming arc that terminates at the strike point."""
        strike_point = selected_strike_point_world()
        incoming_direction = np.array(
            [math.cos(target_yaw_world), math.sin(target_yaw_world), 0.0],
            dtype=np.float64,
        )
        start = strike_point.copy()
        start += incoming_ball_start_distance * incoming_direction
        start[2] += incoming_ball_start_height
        phase = np.linspace(0.0, 1.0, incoming_ball_samples, dtype=np.float64)
        path = (1.0 - phase[:, None]) * start + phase[:, None] * strike_point
        path[:, 2] += 4.0 * incoming_ball_arc_height * phase * (1.0 - phase)
        path[-1] = strike_point
        return path

    def incoming_ball_progress():
        """Synchronize visual ball arrival with the selected motion's strike frame."""
        if strike_completed:
            return 1.0
        if not strike_active or selected_entry is None:
            return 0.0
        entry_frame = int(selected_entry["entry_frame"])
        strike_frame = int(selected_entry["clip"]["strike"])
        duration_frames = max(1, strike_frame - entry_frame)
        return float(
            np.clip(
                (int(current_frame[0].item()) - entry_frame) / duration_frames,
                0.0,
                1.0,
            )
        )

    strike_clips = []
    for i, (o, e, name) in enumerate(db.motion_boundaries):
        base = os.path.basename(str(name)).lower()
        side = "fh" if base.startswith("fh_") else "bh" if base.startswith("bh_") else None
        if side is None:
            continue
        sg = int(db.strike_frames[i]) if i < len(db.strike_frames) else (o + e) // 2
        sg = max(o, min(e - 1, sg))
        strike_w = db.racket_pos_w[sg:sg + 1]
        root_p = db.body_pos_w[o:o + 1, db.root_idx]
        root_fq = _facing_quat(db.body_quat_w[o:o + 1, db.root_idx])
        strike_local = quat_apply_inverse(root_fq, strike_w - root_p)[0].cpu().numpy()
        strike_clips.append({
            "start": int(o),
            "end": int(e),
            "strike": int(sg),
            "side": side,
            "strike_local": strike_local,
            "name": str(name),
        })
    print(f"[STRIKE] indexed {len(strike_clips)} tennis clips for entry planning", end="\r\n")

    # Key-body pose for the entry-pose cost. The official resources carry only the G1
    # root+joints (no full-body positions), so key-body world positions come from MuJoCo
    # FK on demand, cached per frame (entry search touches only strike-clip head windows
    # + one predicted move frame per plan).
    KEY_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link",
                      "left_wrist_yaw_link", "right_wrist_yaw_link")
    key_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in KEY_BODY_NAMES]
    assert all(b >= 0 for b in key_body_ids), f"missing key bodies: {KEY_BODY_NAMES}"
    _key_pose_cache: dict[int, torch.Tensor] = {}

    def frame_key_pose_local(frame_idx):
        f = int(frame_idx)
        hit = _key_pose_cache.get(f)
        if hit is not None:
            return hit
        _strike_pose_qpos(f)
        key_w = torch.tensor(strike_data.xpos[key_body_ids], dtype=torch.float32, device=device)
        root_p = db.body_pos_w[f, db.root_idx]
        root_fq = _facing_quat(db.body_quat_w[f:f + 1, db.root_idx])
        local = quat_apply_inverse(root_fq.expand(len(key_body_ids), -1), key_w - root_p)
        _key_pose_cache[f] = local
        return local

    def advance_frame_steps(frame_tensor, steps):
        f = frame_tensor.clone()
        for _ in range(max(0, int(steps))):
            f, _ = db.advance_frames(f)
        return f

    def aligned_strike_transform(clip):
        o, e, sg = clip["start"], clip["end"], clip["strike"]
        ka, kb = max(o, sg - 2), min(e - 1, sg + 2)
        v = (strike_racket_world(kb) - strike_racket_world(ka))[:2]
        if float(np.linalg.norm(v)) < 1e-6:
            theta = 0.0
        else:
            theta = -float(np.arctan2(v[1], v[0]))  # contact swing direction -> world +X
        q_off = yaw_quat_torch(theta)
        racket0 = torch.tensor(strike_racket_world(sg), dtype=torch.float32, device=device).unsqueeze(0)
        rotated_racket0 = quat_apply(q_off, racket0)
        translation = torch.zeros(1, 3, dtype=torch.float32, device=device)
        translation[:, :2] = torch.tensor([[ball_x, ball_y]], dtype=torch.float32, device=device) - rotated_racket0[:, :2]
        return q_off, translation

    def aligned_root_state(frame_idx, q_off, translation):
        f = torch.tensor([int(frame_idx)], dtype=torch.long, device=device)
        root_pos = quat_apply(q_off, db.body_pos_w[f, db.root_idx]) + translation
        root_quat = quat_mul(q_off, db.body_quat_w[f, db.root_idx])
        root_linvel = quat_apply(q_off, db.body_linvel[f, db.root_idx])
        root_angvel = quat_apply(q_off, db.body_angvel[f, db.root_idx])
        return root_pos, root_quat, root_linvel, root_angvel

    def entry_pose_cost(pred_frame, entry_frame, entry_root_quat, clip_start):
        pred_frame_i = int(pred_frame)
        entry_frame_i = int(entry_frame)
        key_cost = float((frame_key_pose_local(pred_frame_i) - frame_key_pose_local(entry_frame_i)).pow(2).mean().item())
        joint_cost = float((db.joint_pos[pred_frame_i] - db.joint_pos[entry_frame_i]).pow(2).mean().item())
        pred_root_quat = quat_mul(root_yaw_offset, db.body_quat_w[torch.tensor([pred_frame_i], device=device), db.root_idx])
        pred_yaw = float(yaw_from_quat_tensor(_facing_quat(pred_root_quat))[0].item())
        entry_yaw = float(yaw_from_quat_tensor(_facing_quat(entry_root_quat))[0].item())
        yaw_cost = float(target_wrap_angle(pred_yaw - entry_yaw) ** 2)
        start_cost = float(max(0, entry_frame_i - int(clip_start))) / float(max(1, entry_max_search_frames))
        return (
            entry_w_key_pose * key_cost
            + entry_w_joint_pose * joint_cost
            + entry_w_yaw * yaw_cost
            + entry_w_clip_start * start_cost
        ), key_cost, joint_cost, yaw_cost

    def plan_strike_entry(reason="target"):
        nonlocal move_target_xy_world, move_target_yaw_world, move_target_vel_world, move_target_yaw_rate
        nonlocal selected_entry, strike_active, strike_completed
        strike_active = False
        strike_completed = False
        selected_entry = None
        if not strike_entry_enabled or not strike_clips:
            move_target_xy_world = np.array([ball_x, ball_y], dtype=np.float64)
            move_target_yaw_world = float(target_yaw_world)
            move_target_vel_world = np.zeros(2, dtype=np.float64)
            move_target_yaw_rate = 0.0
            rebuild_target_trajectory(reset_timer=True, reason=reason)
            return

        want = "fh" if int(locked_side[0].item()) < 0 else "bh"
        ball_world = torch.tensor([[ball_x, ball_y, ball_z]], dtype=torch.float32, device=device)
        query_fq = _facing_quat(ghost_query_root_quat)
        ball_local = quat_apply_inverse(query_fq, ball_world - ghost_query_root_pos)[0].cpu().numpy()
        predicted_move_frame = advance_frame_steps(current_frame, entry_predict_move_frames)[0].item()
        candidates = []
        for clip in strike_clips:
            if clip["side"] != want:
                continue
            o, e, sg = clip["start"], clip["end"], clip["strike"]
            first_entry = o
            last_entry = min(sg - entry_min_lead_frames, o + entry_max_search_frames, e - 1)
            if last_entry < first_entry:
                continue
            contact_local = clip["strike_local"]
            contact_cost = (
                entry_w_contact_z * abs(float(contact_local[2] - ball_local[2]))
                + entry_w_contact_xy * float(np.linalg.norm(contact_local[:2] - ball_local[:2]))
            )
            q_off, translation = aligned_strike_transform(clip)
            for entry_frame in range(first_entry, last_entry + 1):
                entry_root_pos, entry_root_quat, entry_root_linvel, entry_root_angvel = aligned_root_state(
                    entry_frame, q_off, translation)
                pose_cost, key_cost, joint_cost, yaw_cost = entry_pose_cost(
                    predicted_move_frame, entry_frame, entry_root_quat, o)
                total_cost = float(contact_cost + pose_cost)
                candidates.append((
                    total_cost,
                    {
                        "clip": clip,
                        "entry_frame": int(entry_frame),
                        "q_off": q_off.clone(),
                        "translation": translation.clone(),
                        "entry_root_pos": entry_root_pos.clone(),
                        "entry_root_quat": entry_root_quat.clone(),
                        "entry_root_linvel": entry_root_linvel.clone(),
                        "entry_root_angvel": entry_root_angvel.clone(),
                        "contact_cost": float(contact_cost),
                        "pose_cost": float(pose_cost),
                        "key_cost": float(key_cost),
                        "joint_cost": float(joint_cost),
                        "yaw_cost": float(yaw_cost),
                    },
                ))

        if not candidates:
            print(f"[ENTRY {reason}] no {want} candidate; falling back to ball XY", end="\r\n")
            move_target_xy_world = np.array([ball_x, ball_y], dtype=np.float64)
            move_target_yaw_world = float(target_yaw_world)
            move_target_vel_world = np.zeros(2, dtype=np.float64)
            move_target_yaw_rate = 0.0
            rebuild_target_trajectory(reset_timer=True, reason=reason)
            return

        candidates.sort(key=lambda item: item[0])
        total_cost, selected_entry = candidates[0]
        entry_root_pos = selected_entry["entry_root_pos"]
        entry_root_quat = selected_entry["entry_root_quat"]
        entry_root_linvel = selected_entry["entry_root_linvel"]
        entry_root_angvel = selected_entry["entry_root_angvel"]
        move_target_xy_world = entry_root_pos[0, :2].detach().cpu().numpy().astype(np.float64)
        move_target_yaw_world = float(yaw_from_quat_tensor(_facing_quat(entry_root_quat))[0].item())
        move_target_vel_world = entry_root_linvel[0, :2].detach().cpu().numpy().astype(np.float64)
        move_target_yaw_rate = float(entry_root_angvel[0, 2].item())
        rebuild_target_trajectory(reset_timer=True, reason=reason)
        clip = selected_entry["clip"]
        print(
            f"[ENTRY {reason}] side={want} clip={os.path.basename(clip['name'])} "
            f"entry=f{selected_entry['entry_frame']} strike=f{clip['strike']} "
            f"root=({move_target_xy_world[0]:.2f},{move_target_xy_world[1]:.2f}) "
            f"yaw={math.degrees(move_target_yaw_world):.1f}deg "
            f"v=({move_target_vel_world[0]:.2f},{move_target_vel_world[1]:.2f}) "
            f"cost={total_cost:.3f} contact={selected_entry['contact_cost']:.3f} "
            f"pose={selected_entry['pose_cost']:.3f}",
            end="\r\n",
        )

    def repick_entry_frame_with_actual_pose():
        """Re-pick the entry FRAME (clip stays) using the pose the robot ACTUALLY arrived
        with. plan_strike_entry chose the entry at serve time against a predicted move-end
        pose ('current clip advanced 20 frames'); by arrival the matcher has switched clips
        many times, so that prediction is stale — the single biggest source of the
        move->strike snap. Costs the entry window of ONE clip (key poses are FK-cached)."""
        nonlocal selected_entry
        clip = selected_entry["clip"]
        o, e, sg = clip["start"], clip["end"], clip["strike"]
        first_entry = o
        last_entry = min(sg - entry_min_lead_frames, o + entry_max_search_frames, e - 1)
        if last_entry < first_entry:
            return
        actual_frame = int(current_frame[0].item())
        q_off = selected_entry["q_off"]
        translation = selected_entry["translation"]
        best = None
        for entry_frame in range(first_entry, last_entry + 1):
            _, entry_root_quat, _, _ = aligned_root_state(entry_frame, q_off, translation)
            pose_cost, _, _, _ = entry_pose_cost(actual_frame, entry_frame, entry_root_quat, o)
            if best is None or pose_cost < best[0]:
                best = (pose_cost, entry_frame)
        old_frame = int(selected_entry["entry_frame"])
        if best is not None and best[1] != old_frame:
            selected_entry = dict(selected_entry)
            selected_entry["entry_frame"] = int(best[1])
            print(f"[STRIKE ENTER] re-picked entry f{old_frame} -> f{best[1]} "
                  f"(actual-pose cost {best[0]:.3f})", end="\r\n")

    def enter_selected_strike():
        nonlocal strike_active, strike_completed
        if selected_entry is None:
            return False
        repick_entry_frame_with_actual_pose()
        old_root_pos = cur["root_pos"]
        old_root_quat = cur["root_quat"]
        old_root_linvel = cur["root_linvel"]
        old_root_angvel = cur["root_angvel"]
        old_joint_pos = cur["joint_pos"]
        old_joint_vel = cur["joint_vel"]

        entry_frame = int(selected_entry["entry_frame"])
        entry_tensor = torch.tensor([entry_frame], dtype=torch.long, device=device)
        root_yaw_offset[:] = selected_entry["q_off"]
        root_pos_offset[:] = selected_entry["translation"]

        new_root_pos_db = db.body_pos_w[entry_tensor, db.root_idx].clone()
        new_root_quat_db = db.body_quat_w[entry_tensor, db.root_idx].clone()
        new_pos_t = quat_apply(root_yaw_offset, new_root_pos_db) + root_pos_offset
        new_quat_t = quat_mul(root_yaw_offset, new_root_quat_db)
        new_lin_t = quat_apply(root_yaw_offset, db.body_linvel[entry_tensor, db.root_idx])
        new_ang_t = quat_apply(root_yaw_offset, db.body_angvel[entry_tensor, db.root_idx])
        mask = torch.ones(N, dtype=torch.bool, device=device)
        # The move->strike pose gap (run stance -> ready stance, up to switch_yaw_threshold of
        # heading) is much larger than a move-mode frame switch — give it its own, longer blend.
        blender.blend_duration = entry_blend_frames / db.fps
        blender.trigger(mask, old_root_pos, new_pos_t, old_root_quat, new_quat_t,
                        old_root_linvel, new_lin_t, old_root_angvel, new_ang_t,
                        old_joint_pos, db.joint_pos[entry_tensor],
                        old_joint_vel, db.joint_vel[entry_tensor])
        current_frame[:] = entry_tensor
        strike_active = True
        strike_completed = False
        racket_center_path.clear()
        clip = selected_entry["clip"]
        strike_end_frame = min(int(clip["end"]) - 1,
                               int(clip["strike"]) + strike_postcontact_frames)
        print(
            f"[STRIKE ENTER] clip={os.path.basename(clip['name'])} "
            f"entry=f{entry_frame} strike=f{clip['strike']} end=f{strike_end_frame}",
            end="\r\n",
        )
        return True

    def queue_manual_target(xy, yaw):
        mouse_state["pending"] = (np.asarray(xy, dtype=np.float64).copy(), float(target_wrap_angle(yaw)))
        print(f"[MOUSE TARGET] queued xy=({xy[0]:.2f},{xy[1]:.2f}) "
              f"yaw={math.degrees(float(target_wrap_angle(yaw))):.1f}deg", end="\r\n")

    def mouse_ground(window, cursor_x, cursor_y, viewer):
        return target_cursor_to_ground_point(window, cursor_x, cursor_y, viewer, model, mj_data, pick_opt, pick_scn)

    # MuJoCo's viewer installs its camera handlers as C++ GLFW callbacks. pyGLFW cannot hold,
    # call, or re-install those (they are raw C function pointers, not Python callables): passing
    # the pointer back through glfw.set_*_callback wraps the INT as a "callable" and the next
    # mouse event raises TypeError inside pyGLFW's deferred-exception machinery, killing the
    # render thread. So capture/forward/restore the original callbacks at the raw ctypes level.
    _MB_SIG = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    _CP_SIG = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_double, ctypes.c_double)

    def _glfw_raw():
        """Independent ctypes handle to the loaded GLFW lib (own argtypes; leaves pyGLFW's alone)."""
        import glfw
        if mouse_state.get("raw") is None:
            raw = ctypes.CDLL(glfw._glfw._name)
            for fn in (raw.glfwSetMouseButtonCallback, raw.glfwSetCursorPosCallback):
                fn.restype = ctypes.c_void_p                    # returns the PREVIOUS C callback
                fn.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
            mouse_state["raw"] = raw
        return mouse_state["raw"]

    def _win_addr(window):
        return ctypes.cast(window, ctypes.c_void_p).value

    def call_previous_callback(prev_ptr, sig, window, *args):
        """Forward an event to the viewer's ORIGINAL C callback (camera control)."""
        if not prev_ptr:
            return
        try:
            sig(prev_ptr)(_win_addr(window), *args)
        except Exception as exc:
            print(f"[MOUSE] forward to viewer callback failed: {exc}", end="\r\n")

    def make_mouse_button_callback(viewer):
        def _mouse_button_callback(window, button, action, mods):
            # NEVER let an exception escape: pyGLFW re-raises it on the next glfw call inside
            # the viewer's render loop, killing the render thread.
            try:
                import glfw

                if mouse_state["manual_edit"] and button == glfw.MOUSE_BUTTON_LEFT:
                    if action == glfw.PRESS:
                        cursor_x, cursor_y = glfw.get_cursor_pos(window)
                        p = mouse_ground(window, cursor_x, cursor_y, viewer)
                        if p is not None:
                            mouse_state["dragging_target"] = True
                            mouse_state["draft_xy"] = p
                            mouse_state["draft_yaw"] = target_yaw_world
                            return
                    elif action == glfw.RELEASE:
                        if mouse_state["dragging_target"] and mouse_state["draft_xy"] is not None:
                            if mouse_select_yaw:
                                yaw = float(mouse_state["draft_yaw"] if mouse_state["draft_yaw"] is not None else target_yaw_world)
                            else:
                                yaw = float(target_traj_cfg.get("default_yaw", 0.0))   # yaw selection off: fixed default
                            queue_manual_target(mouse_state["draft_xy"], yaw)
                        mouse_state["dragging_target"] = False
                        return
                call_previous_callback(mouse_state["prev_mouse_button_callback"], _MB_SIG,
                                       window, button, action, mods)
            except Exception as exc:
                print(f"[MOUSE] button callback error: {exc}", end="\r\n")
        return _mouse_button_callback

    def make_cursor_pos_callback(viewer):
        def _cursor_pos_callback(window, cursor_x, cursor_y):
            try:
                if (mouse_select_yaw and mouse_state["manual_edit"] and mouse_state["dragging_target"]
                        and mouse_state["draft_xy"] is not None):
                    p = mouse_ground(window, cursor_x, cursor_y, viewer)
                    if p is not None:
                        delta = p - np.asarray(mouse_state["draft_xy"], dtype=np.float64)
                        if float(np.linalg.norm(delta)) >= 0.15:
                            mouse_state["draft_yaw"] = float(np.arctan2(delta[1], delta[0]))
                        return
                call_previous_callback(mouse_state["prev_cursor_pos_callback"], _CP_SIG,
                                       window, cursor_x, cursor_y)
            except Exception as exc:
                print(f"[MOUSE] cursor callback error: {exc}", end="\r\n")
        return _cursor_pos_callback

    def install_mouse_callbacks(viewer):
        import glfw

        window = glfw.get_current_context()
        if not window:
            print("[MOUSE] edit unavailable: GLFW context not found", end="\r\n")
            return False
        if mouse_state.get("window_addr") == _win_addr(window):
            return True
        raw = _glfw_raw()
        addr = _win_addr(window)
        mouse_state["window"] = window
        mouse_state["window_addr"] = addr
        # Capture the viewer's original C callbacks as RAW pointers (clear-and-read), then
        # install ours through pyGLFW. The raw pointers are what restore puts back.
        mouse_state["prev_mouse_button_callback"] = raw.glfwSetMouseButtonCallback(addr, None)
        mouse_state["prev_cursor_pos_callback"] = raw.glfwSetCursorPosCallback(addr, None)
        glfw.set_mouse_button_callback(window, make_mouse_button_callback(viewer))
        glfw.set_cursor_pos_callback(window, make_cursor_pos_callback(viewer))
        return True

    def restore_mouse_callbacks():
        import glfw

        window = mouse_state.get("window")
        if not window:
            return
        raw = _glfw_raw()
        addr = mouse_state.get("window_addr") or _win_addr(window)
        try:
            # Drop our pyGLFW-managed callbacks first (cleans pyGLFW's registry)...
            glfw.set_mouse_button_callback(window, None)
            glfw.set_cursor_pos_callback(window, None)
        except Exception as exc:
            print(f"[MOUSE] could not clear callbacks: {exc}", end="\r\n")
        # ...then put the viewer's ORIGINAL C camera handlers back, at the raw level.
        raw.glfwSetMouseButtonCallback(addr, mouse_state.get("prev_mouse_button_callback") or None)
        raw.glfwSetCursorPosCallback(addr, mouse_state.get("prev_cursor_pos_callback") or None)
        mouse_state["window"] = None
        mouse_state["window_addr"] = None
        mouse_state["prev_mouse_button_callback"] = None
        mouse_state["prev_cursor_pos_callback"] = None
        mouse_state["dragging_target"] = False

    def viewer_key_callback(key):
        ch = chr(key).lower() if 0 <= key < 256 else ""
        if gamepad_mode:
            # gamepad mode: viewer window drives locomotion too (W/A/S/D/X/C); no mouse target
            handle_pad_command(ch)
            return
        if ch == "m":
            if mouse_state["manual_edit"]:
                mouse_state["manual_edit"] = False
                restore_mouse_callbacks()
                print("[MOUSE] target edit OFF", end="\r\n")
            else:
                if install_mouse_callbacks(viewer):
                    mouse_state["manual_edit"] = True
                    mouse_state["dragging_target"] = False
                    mouse_state["draft_xy"] = None
                    mouse_state["draft_yaw"] = None
                    print("[MOUSE] target edit ON: left-click XY, drag yaw, release", end="\r\n")

    # ---- Ghost visualization: persistent snapshots of the executed G1 sequence ----
    ghost_tracks = []        # [{poses: [(qpos[36], is_strike), ...], cursor, rgba, strike_rgba}, ...]
    if args.ghost:
        gdata = strike_data
        g_wrist = strike_wrist
        g_off = strike_off
        # Add ghost geoms directly to the viewer scene. Mesh data IDs are scene-local,
        # so copying them from a separate MjvScene corrupts meshes after the court is attached.
        ghost_opt = mujoco.MjvOption()
        ghost_opt.geomgroup[:] = 0
        ghost_opt.geomgroup[2] = 1
        ghost_pert = mujoco.MjvPerturb()
        print(
            f"[GHOST] executed-motion snapshots every {ghost_snapshot_interval} move frames / "
            f"{ghost_strike_snapshot_interval} strike frames, anchored at "
            f"strike-{ghost_strike_precontact_frames}; "
            f"loco spacing >= {ghost_loco_min_distance:.2f} m; "
            "reference-motion playback disabled",
            end="\r\n",
        )

        ghost_count = max(1, int(args.ghost_count))
        ghost_palette = [
            (np.array([0.50, 0.70, 1.00, 0.34], dtype=np.float32),
             np.array([1.00, 0.50, 0.00, 0.62], dtype=np.float32)),
            (np.array([0.25, 1.00, 0.60, 0.30], dtype=np.float32),
             np.array([1.00, 0.95, 0.20, 0.58], dtype=np.float32)),
            (np.array([1.00, 0.35, 0.85, 0.28], dtype=np.float32),
             np.array([1.00, 0.25, 0.25, 0.58], dtype=np.float32)),
            (np.array([0.95, 0.85, 0.35, 0.26], dtype=np.float32),
             np.array([0.20, 1.00, 1.00, 0.56], dtype=np.float32)),
            (np.array([0.70, 0.50, 1.00, 0.26], dtype=np.float32),
             np.array([1.00, 0.60, 0.20, 0.56], dtype=np.float32)),
        ]

        def compute_ghost_tracks(bx, by, bz, side_flag):
            want = "fh" if side_flag < 0 else "bh"            # right ball -> forehand
            cands = [c for c in strike_clips if c["side"] == want]
            if not cands:
                return []

            ball_world = torch.tensor([[bx, by, bz]], dtype=torch.float32, device=device)
            root_fq = _facing_quat(ghost_query_root_quat)
            ball_local = quat_apply_inverse(root_fq, ball_world - ghost_query_root_pos)[0].cpu().numpy()

            def _geo_dist(c):
                return float(abs(c["strike_local"][2] - ball_local[2])
                             + 0.1 * np.linalg.norm(c["strike_local"][:2] - ball_local[:2]))

            # Ghost MIRRORS the strike-entry planner: the PRIMARY ghost is the exact clip chosen
            # for execution, with the SAME alignment (q_off/translation) and entry frame, played
            # from entry->end. So the ghost is the stroke that will actually run, and its root
            # path starts at the entry root — i.e. exactly at the CT-OC trajectory endpoint.
            # Extra ghosts (ghost_count>1) are the geometric-nearest OTHER clips, shown whole.
            plan = []   # (clip, q_off, translation)
            if selected_entry is not None and selected_entry["clip"]["side"] == want:
                se = selected_entry
                plan.append((se["clip"], se["q_off"], se["translation"]))
            for c in sorted(cands, key=_geo_dist):
                if len(plan) >= ghost_count:
                    break
                if plan and c is plan[0][0]:
                    continue
                q_off_c, translation_c = aligned_strike_transform(c)
                plan.append((c, q_off_c, translation_c))

            tracks = []
            for rank, (clip, q_off, translation) in enumerate(plan):
                o, e, sg = clip["start"], clip["end"], clip["strike"]
                # Ghost ANIMATES the full stroke (clip start -> clip end, incl. follow-through),
                # but the drawn root PATH stops at the strike frame (approach + swing to contact).
                # The CT-OC trajectory endpoint (entry root) lands on the entry_frame point
                # partway along that path.
                poses = []
                for k in range(o, e):
                    new_rp = (quat_apply(q_off, db.body_pos_w[k, 0:1]) + translation)[0].cpu().numpy()
                    new_rq = quat_mul(q_off, db.body_quat_w[k, 0:1])[0].cpu().numpy()
                    qp = np.zeros(36, dtype=np.float64)
                    qp[:3] = new_rp; qp[3:7] = new_rq
                    qp[7:36] = db.joint_pos[k].cpu().numpy()[ISAACLAB_TO_MUJOCO]
                    poses.append((qp, abs(k - sg) <= 2))               # flash around contact

                rgba, strike_rgba = ghost_palette[rank % len(ghost_palette)]
                sg_local = int(np.clip(sg - o, 0, len(poses) - 1))
                # Root path drawn only up to the strike frame; strike_root = root at contact.
                root_path = [np.array([qp[0], qp[1], 0.02]) for qp, _ in poses[:sg_local + 1:3]]
                strike_root = np.array([poses[sg_local][0][0], poses[sg_local][0][1], 0.02])
                tracks.append({
                    "poses": poses,
                    "cursor": 0,
                    "rgba": rgba,
                    "strike_rgba": strike_rgba,
                    "dist": _geo_dist(clip),
                    "strike_frame": sg,
                    "clip_start": o,
                    "clip_end": e,
                    "root_path": root_path,
                    "strike_root": strike_root,
                })
            return tracks

        def _draw_reference_ghost_disabled(viewer):
            # Dynamic ghost: advance the cursor and append the current G1 visual frame directly
            # into user_scn, then recolor the appended geoms. Direct insertion preserves the
            # destination scene's mesh IDs and avoids copying uninitialized label metadata.
            if not ghost_tracks:
                return
            for track in ghost_tracks:
                poses = track["poses"]
                if not poses:
                    continue
                # Ghost root XY trajectory (its own palette color, opaque line) + contact-root dot.
                _r, _g, _b = track["rgba"][:3]
                _line = [_r, _g, _b, 0.85]
                for _a, _bp in zip(track["root_path"][:-1], track["root_path"][1:]):
                    add_arrow(viewer, _a, _bp, _line, width=0.008)
                add_sphere(viewer, track["strike_root"], [_r, _g, _b, 1.0], r=0.045)
                track["cursor"] = (track["cursor"] + 1) % len(poses)
                qp, is_strike = poses[track["cursor"]]
                gdata.qpos[:36] = qp
                mujoco.mj_forward(model, gdata)
                ghost_geom_start = viewer.user_scn.ngeom
                mujoco.mjv_addGeoms(
                    model,
                    gdata,
                    ghost_opt,
                    ghost_pert,
                    mujoco.mjtCatBit.mjCAT_DYNAMIC.value,
                    viewer.user_scn,
                )
                rgba = track["strike_rgba"] if is_strike else track["rgba"]
                for i in range(ghost_geom_start, viewer.user_scn.ngeom):
                    geom = viewer.user_scn.geoms[i]
                    geom.objtype = int(mujoco.mjtObj.mjOBJ_UNKNOWN)
                    geom.objid = -1
                    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
                    geom.label = ""
                    geom.matid = -1
                    geom.rgba[:] = rgba
                    geom.transparent = 1

        def record_ghost_snapshot():
            """Save the actually displayed G1 pose at a fixed frame interval."""
            nonlocal ghost_last_snapshot_step
            if strike_completed:
                return
            current_is_strike = bool(strike_active)
            if current_is_strike:
                if selected_entry is None:
                    return
                strike_frame = int(selected_entry["clip"]["strike"])
                anchor_frame = strike_frame - ghost_strike_precontact_frames
                displayed_frame = int(current_frame[0].item())
                if (displayed_frame - anchor_frame) % ghost_strike_snapshot_interval != 0:
                    return
            else:
                if sim_step - ghost_last_snapshot_step < ghost_snapshot_interval:
                    return
            if not current_is_strike and ghost_loco_min_distance > 0.0:
                current_root_xy = mj_data.qpos[:2]
                for qpos, is_strike in zip(executed_ghost_qpos, executed_ghost_is_strike):
                    if not is_strike and np.linalg.norm(current_root_xy - qpos[:2]) < ghost_loco_min_distance:
                        return
            executed_ghost_qpos.append(mj_data.qpos[:36].copy())
            executed_ghost_is_strike.append(current_is_strike)
            ghost_last_snapshot_step = sim_step

        def draw_ghost(viewer):
            """Draw accumulated executed-motion poses as persistent static ghosts."""
            move_ghost_rgba = np.array([0.45, 0.75, 1.0, 0.22], dtype=np.float32)
            strike_ghost_rgba = np.array([0.55, 1.0, 0.60, 0.28], dtype=np.float32)
            for qpos, is_strike in zip(executed_ghost_qpos, executed_ghost_is_strike):
                gdata.qpos[:36] = qpos
                mujoco.mj_forward(model, gdata)
                ghost_geom_start = viewer.user_scn.ngeom
                mujoco.mjv_addGeoms(
                    model,
                    gdata,
                    ghost_opt,
                    ghost_pert,
                    mujoco.mjtCatBit.mjCAT_DYNAMIC.value,
                    viewer.user_scn,
                )
                for i in range(ghost_geom_start, viewer.user_scn.ngeom):
                    geom = viewer.user_scn.geoms[i]
                    geom.objtype = int(mujoco.mjtObj.mjOBJ_UNKNOWN)
                    geom.objid = -1
                    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
                    geom.label = ""
                    geom.matid = -1
                    geom.rgba[:] = strike_ghost_rgba if is_strike else move_ghost_rgba
                    geom.transparent = 1
    else:
        def compute_ghost_tracks(*a):
            return []

        def record_ghost_snapshot():
            pass

        def draw_ghost(viewer):
            pass

    sim_step = 0
    match_count = 0
    frame_dt = 1.0 / db.fps

    # ---- gamepad control mode (no ball, no strike; pure locomotion following) ----
    _control = getattr(args, "control", None) or (cfg.get("control", {}) or {}).get("mode", "ball")
    gamepad_mode = str(_control).lower() == "gamepad"
    pad_cfg = cfg.get("gamepad", {}) or {}
    pad_vel_halflife = float(pad_cfg.get("velocity_halflife", 0.27))
    pad_rot_halflife = float(pad_cfg.get("rotation_halflife", 0.27))
    pad_run_speeds = (float(pad_cfg.get("run", {}).get("forward_speed", 3.0)),
                      float(pad_cfg.get("run", {}).get("side_speed", 1.5)),
                      float(pad_cfg.get("run", {}).get("back_speed", 1.5)))
    pad_walk_speeds = (float(pad_cfg.get("walk", {}).get("forward_speed", 1.75)),
                       float(pad_cfg.get("walk", {}).get("side_speed", 1.0)),
                       float(pad_cfg.get("walk", {}).get("back_speed", 1.0)))
    pad_yaw_speed = float(pad_cfg.get("yaw_speed", 2.0))
    pad_force_vel_thr = float(pad_cfg.get("force_search_vel_change", 3.0))    # m/s^2 of desired-vel change
    pad_stick_frame = str(pad_cfg.get("stick_frame", "body")).lower()         # body (official direct-yaw) | camera
    pad = GamepadInput() if gamepad_mode else None
    # simulation-object state (z-up): only velocity/acc/yaw carry state; the sim position is
    # re-anchored to the character root every frame (prediction starts from where we ARE).
    pad_stick = np.zeros(2, dtype=np.float64)       # keyboard command (camera-relative fwd/left)
    pad_turn_axis = 0.0
    pad_walk = False
    pad_strafe = False
    pad_sim_vel = np.zeros(2, dtype=np.float64)
    pad_sim_acc = np.zeros(2, dtype=np.float64)
    pad_sim_yaw = 0.0
    pad_sim_yaw_vel = 0.0
    pad_desired_vel = np.zeros(2, dtype=np.float64)
    pad_desired_yaw = 0.0
    pad_prev_desired_vel = np.zeros(2, dtype=np.float64)
    pad_prev_turn = 0.0
    pad_force_cooldown = 0.0
    pad_kb_override = False        # True while a keyboard command is driving (pad idle)
    pad_pred_pts = np.zeros((3, 2), dtype=np.float64)   # world future points (viz)
    pad_pred_yaws = np.zeros(3, dtype=np.float64)

    def handle_pad_command(ch) -> bool:
        """Keyboard command-style control (official set_command): W/A/S/D pick a translation
        direction (body frame; does NOT turn), J/L step the heading +-30deg, X stops,
        C toggles walk/run. Returns True if consumed."""
        nonlocal pad_walk, pad_kb_override, pad_desired_yaw, pad_turn_axis
        if ch == "w":
            pad_stick[:] = (1.0, 0.0)
        elif ch == "s":
            pad_stick[:] = (-1.0, 0.0)
        elif ch == "a":
            pad_stick[:] = (0.0, 1.0)
        elif ch == "d":
            pad_stick[:] = (0.0, -1.0)
        elif ch == "x" or ch == " ":
            pad_stick[:] = (0.0, 0.0)
        elif ch == "j":
            pad_desired_yaw += math.radians(30.0)
            print(f"[GAMEPAD] heading -> {math.degrees(pad_desired_yaw):.0f}deg", end="\r\n")
        elif ch == "l":
            pad_desired_yaw -= math.radians(30.0)
            print(f"[GAMEPAD] heading -> {math.degrees(pad_desired_yaw):.0f}deg", end="\r\n")
        elif ch == "c":
            pad_walk = not pad_walk
            print(f"[GAMEPAD] gait -> {'walk' if pad_walk else 'run'}", end="\r\n")
        else:
            return False
        pad_turn_axis = 0.0
        pad_kb_override = True    # keyboard drives until the pad produces input again
        return True

    # ---- episode collection (--collect N): auto-run move+strike, save each episode ----
    collect_n = int(getattr(args, "collect", 0) or 0)
    col_cfg = cfg.get("collect", {}) or {}
    collecting = collect_n > 0 and not gamepad_mode
    collect_done = False
    collect_out = Path(col_cfg.get("out_dir", "data/generated/collected_episodes"))
    if not collect_out.is_absolute():
        collect_out = (_REPO / collect_out).resolve()
    collect_prefix = str(col_cfg.get("prefix", "ep"))
    collect_timeout = float(col_cfg.get("episode_timeout", 12.0))       # s from serve; discard + resample
    collect_min_frames = int(col_cfg.get("min_frames", 60))
    ep_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in EPISODE_BODY_NAMES]
    assert all(b >= 0 for b in ep_body_ids), "g1.xml missing episode bodies"
    ep_jp: list = []; ep_jv: list = []; ep_bp: list = []; ep_bq: list = []; ep_bav: list = []
    ep_rlv: list = []; ep_rav: list = []
    ep_strike_frame = -1
    ep_start_step = 0
    ep_count = 0                 # episodes saved THIS run (progress toward collect_n)
    ep_file_idx = 0              # next file number (auto-continues from existing files)
    ep_ball = (0.0, 0.0, 0.0)
    ep_ball_local = (0.0, 0.0, 0.0)   # served ball in robot frame at serve time: [fwd, left, height]
    ball_local_cur = [0.0, 0.0, 0.0]  # latest serve's robot-frame ball (spawn writes, ep_reset reads)
    ball_local_stats: list = []       # all serves' [fwd, left, height] for the distribution figure
    # Collection normally skips markers for speed. Draw everything if collect.visualize=true OR
    # the user explicitly passed --ghost (they want to watch the collection).
    collect_viz = (
        bool(col_cfg.get("visualize", False)) or bool(getattr(args, "ghost", False))
    ) and not clean_view
    collect_sync_interval = max(1, int(col_cfg.get("sync_interval", 10)))  # viewer.sync every N frames
    if collecting:
        collect_out.mkdir(parents=True, exist_ok=True)
        # Resume numbering: never overwrite — continue after the highest existing index.
        for p in collect_out.glob(f"{collect_prefix}_*.npz"):
            try:
                ep_file_idx = max(ep_file_idx, int(p.stem.rsplit("_", 1)[-1]) + 1)
            except ValueError:
                pass
        print(f"[COLLECT] target={collect_n} new episodes -> {collect_out} "
              f"(resuming at {collect_prefix}_{ep_file_idx:04d}, viz={'on' if collect_viz else 'off'})",
              end="\r\n")

    # ---- init ----
    reset_to_fixed_pose()
    if gamepad_mode:
        strike_active = False
        strike_completed = False
        selected_entry = None
        pad_sim_yaw = float(yaw_from_quat_tensor(_facing_quat(cur["root_quat"]))[0].item())
        pad_desired_yaw = pad_sim_yaw
    # (ball-mode init planning happens below, after the serve/ball closures are defined)

    print("=" * 64, end="\r\n")
    print(f"  MuJoCo Tennis MM   db={args.db_file}  T={db.T}  fps={db.fps:.0f}"
          f"  mode={'GAMEPAD' if gamepad_mode else 'ball'}", end="\r\n")
    if keyboard and gamepad_mode:
        print("  [GAMEPAD — terminal keys] W/A/S/D=translate(no turn)  J/L=turn +-30deg  X=stop  C=walk/run  Q=quit"
              "  (joystick: Lstick translate, Rstick turn, A walk, LT strafe-to-camera)", end="\r\n")
    elif keyboard:
        print("  [Controls — press in TERMINAL] B=new ball  R=restart  Q=quit", end="\r\n")
    if not gamepad_mode:
        print("  [Viewer] M=mouse target edit (left-click XY, drag yaw, release; Z sampled)", end="\r\n")
    print("=" * 64, end="\r\n\r\n")

    console = Console()

    # ---- live root-speed plot (--speed_plot): matplotlib window, non-blocking updates ----
    headless = bool(getattr(args, "headless", False))
    speed_plot = bool(getattr(args, "speed_plot", False)) and not headless   # no live plot without a display
    _sp = None
    if speed_plot:
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as _plt
            from collections import deque as _deque
            _plt.ion()
            _sp_win = 300     # ~6 s at 50 fps
            _sp_fig, _sp_ax = _plt.subplots(figsize=(5.5, 2.6))
            _sp_t = list(range(-_sp_win, 0))
            _sp = {
                "plt": _plt, "fig": _sp_fig, "ax": _sp_ax,
                "mag": _deque([0.0] * _sp_win, maxlen=_sp_win),
                "fwd": _deque([0.0] * _sp_win, maxlen=_sp_win),
                "lat": _deque([0.0] * _sp_win, maxlen=_sp_win),
            }
            (_sp["l_mag"],) = _sp_ax.plot(_sp_t, list(_sp["mag"]), color="#111", lw=1.8, label="|v| horiz")
            (_sp["l_fwd"],) = _sp_ax.plot(_sp_t, list(_sp["fwd"]), color="#d33", lw=1.2, label="forward")
            (_sp["l_lat"],) = _sp_ax.plot(_sp_t, list(_sp["lat"]), color="#0a0", lw=1.2, label="lateral")
            _sp_ax.set_ylim(-3.0, 4.0); _sp_ax.set_xlim(-_sp_win, 0)
            _sp_ax.axhline(0, color="#999", lw=0.6)
            _sp_ax.set_ylabel("root speed (m/s)"); _sp_ax.set_xlabel("frame (now=0)")
            _sp_ax.legend(loc="upper left", fontsize=8); _sp_ax.grid(alpha=0.3)
            _sp_fig.tight_layout()
            print("[speed_plot] live root-speed window opened", end="\r\n")
        except Exception as _exc:
            print(f"[speed_plot] disabled ({_exc})", end="\r\n")
            _sp = None

    ball_box_edges: list = []    # 12 wireframe edges (world) of the ball-sampling box

    def update_ball_box():
        """Wireframe of the ball sampling range, anchored at the CURRENT root pose.
        Disk mode (ball_sample.radius): cylinder wireframe around g1, absolute heights.
        Box mode: facing-frame box (local x=fwd, y=left, z=height offset)."""
        ball_box_edges.clear()
        base = cur["root_pos"][0].cpu().numpy()
        radius = ball_sample.get("radius")
        if radius is not None:
            R = float(radius)
            z_lo, z_hi = [float(v) for v in ball_sample.get("z_abs", [0.55, 1.60])]
            segs = 28
            ring = [np.array([base[0] + R * math.cos(TWO_PI * k / segs),
                              base[1] + R * math.sin(TWO_PI * k / segs)]) for k in range(segs + 1)]
            for z in (z_lo, z_hi):                      # top + bottom rings
                for a, b in zip(ring[:-1], ring[1:]):
                    ball_box_edges.append((np.array([a[0], a[1], z]), np.array([b[0], b[1], z])))
            for k in range(0, segs, segs // 4):         # 4 vertical struts
                p = ring[k]
                ball_box_edges.append((np.array([p[0], p[1], z_lo]), np.array([p[0], p[1], z_hi])))
            return
        xr = ball_sample.get("x", [0.3, 1.1])
        yr = ball_sample.get("y", [-0.8, 0.2])
        zr = ball_sample.get("z", [0.0, 0.6])
        fq = _facing_quat(cur["root_quat"])
        def corner(cx, cy, cz):
            w = quat_apply(fq, torch.tensor([[float(cx), float(cy), 0.0]], device=device))[0].cpu().numpy()
            return np.array([base[0] + w[0], base[1] + w[1], base[2] + float(cz)])
        c = {(i, j, k): corner(xr[i], yr[j], zr[k]) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
        for j in (0, 1):
            for k in (0, 1):
                ball_box_edges.append((c[0, j, k], c[1, j, k]))   # x edges
        for i in (0, 1):
            for k in (0, 1):
                ball_box_edges.append((c[i, 0, k], c[i, 1, k]))   # y edges
        for i in (0, 1):
            for j in (0, 1):
                ball_box_edges.append((c[i, j, 0], c[i, j, 1]))   # z edges

    def spawn_new_ball(reason="random", do_reset=False):
        # Continuous rally: by default DON'T reset the robot — the next ball is sampled around
        # wherever it finished the last strike. do_reset=True only for the very first ball / R.
        nonlocal ball_x, ball_y, ball_z, target_yaw_world
        if do_reset:
            reset_to_fixed_pose()
        ball_x, ball_y, ball_z = random_ball_pos(cur["root_pos"][0], cur["root_quat"][0], ball_sample)
        update_ball_box()
        if collecting:
            # ball in the true PELVIS (root) link frame at serve time: origin = pelvis world
            # position, rotation = full pelvis quat (not yaw-only). So z is relative to the
            # pelvis, not the ground.
            _bw = torch.tensor([[ball_x, ball_y, ball_z]], dtype=torch.float32, device=device)
            _bl = quat_apply_inverse(cur["root_quat"], _bw - cur["root_pos"])[0]
            ball_local_cur[:] = [float(_bl[0]), float(_bl[1]), float(_bl[2])]
            ball_local_stats.append(list(ball_local_cur))
        target_yaw_world = float(target_traj_cfg.get("default_yaw", 0.0))
        lock_side()
        capture_ghost_query_frame()
        plan_strike_entry(reason=reason)
        ghost_tracks.clear()
        console.print(f"[yellow][NEW BALL {reason}][/] ({ball_x:.2f},{ball_y:.2f},{ball_z:.2f})  "
                      f"side={locked_side[0].item():+d}", end="\r\n")

    def ep_reset_buffers():
        nonlocal ep_strike_frame, ep_start_step, ep_ball, ep_ball_local
        ep_jp.clear(); ep_jv.clear(); ep_bp.clear(); ep_bq.clear(); ep_bav.clear()
        ep_rlv.clear(); ep_rav.clear()
        ep_strike_frame = -1
        ep_start_step = sim_step
        ep_ball = (float(ball_x), float(ball_y), float(ball_z))
        ep_ball_local = tuple(ball_local_cur)   # serve-time ball in robot frame [fwd, left, height]

    def ep_record_frame():
        nonlocal ep_strike_frame
        ep_jp.append(cur["joint_pos"][0].cpu().numpy().copy())
        ep_jv.append(cur["joint_vel"][0].cpu().numpy().copy())
        ep_bp.append(mj_data.xpos[ep_body_ids].copy())
        ep_bq.append(mj_data.xquat[ep_body_ids].copy())
        # cvel = [angular, linear], expressed in the world frame. qvel is
        # written in apply_frame immediately before mj_forward.
        ep_bav.append(mj_data.cvel[ep_body_ids, :3].copy())
        ep_rlv.append(cur["root_linvel"][0].cpu().numpy().copy())
        ep_rav.append(cur["root_angvel"][0].cpu().numpy().copy())
        if (strike_active and selected_entry is not None
                and int(current_frame[0].item()) == int(selected_entry["clip"]["strike"])):
            ep_strike_frame = len(ep_jp) - 1

    def save_ball_distribution_figure():
        """Distribution of served ball targets in the ROBOT facing frame (x=forward, y=left,
        z=height): top-down scatter + fwd/left/height histograms. Saved as a PNG next to the
        episodes; also shown if a display is available."""
        if not ball_local_stats:
            return
        try:
            import matplotlib
            if headless:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[dist] matplotlib unavailable: {exc}", end="\r\n")
            return
        a = np.asarray(ball_local_stats, dtype=np.float64)
        fwd, left, hz = a[:, 0], a[:, 1], a[:, 2]
        fig, ax = plt.subplots(2, 2, figsize=(9, 8))
        # top-down scatter (robot at origin, facing +x). Note: plot left(y) on horizontal, fwd(x)
        # on vertical, and invert x so +left is to the LEFT — a bird's-eye view from behind g1.
        sc = ax[0, 0].scatter(left, fwd, c=hz, cmap="viridis", s=18, alpha=0.8)
        ax[0, 0].scatter([0], [0], c="red", marker="^", s=120, label="robot")
        _R = float(ball_sample.get("radius", 3.0))
        ax[0, 0].add_patch(plt.Circle((0, 0), _R, fill=False, ls="--", color="gray"))
        ax[0, 0].set_xlim(-_R * 1.1, _R * 1.1); ax[0, 0].set_ylim(-_R * 1.1, _R * 1.1)
        ax[0, 0].invert_xaxis(); ax[0, 0].set_aspect("equal")
        ax[0, 0].set_xlabel("left (m)"); ax[0, 0].set_ylabel("forward (m)")
        ax[0, 0].set_title(f"ball in robot frame (top-down, N={len(a)})")
        ax[0, 0].legend(loc="upper right", fontsize=8); ax[0, 0].grid(alpha=0.3)
        fig.colorbar(sc, ax=ax[0, 0], label="z rel pelvis (m)", fraction=0.046)
        for _ax, _d, _lab in ((ax[0, 1], fwd, "forward (m)"), (ax[1, 0], left, "left (m)"),
                              (ax[1, 1], hz, "z rel pelvis (m)")):
            _ax.hist(_d, bins=24, color="#3a7", alpha=0.85, edgecolor="white")
            _ax.set_xlabel(_lab); _ax.set_ylabel("count"); _ax.grid(alpha=0.3)
            _ax.set_title(f"{_lab}: mean={_d.mean():.2f} std={_d.std():.2f}")
        fig.tight_layout()
        out_png = collect_out / "ball_distribution.png"
        fig.savefig(out_png, dpi=110)
        print(f"[dist] ball-target distribution saved -> {out_png}", end="\r\n")
        if not headless:
            plt.show(block=False); plt.pause(0.1)

    def ep_finalize():
        nonlocal ep_count, ep_file_idx, collect_done
        T = len(ep_jp)
        if ep_strike_frame >= 0 and T >= collect_min_frames:
            name = f"{collect_prefix}_{ep_file_idx:04d}"
            # ball_local in the episode's FIRST-FRAME pelvis(root) link frame, so it round-trips
            # exactly with body_pos_w[0]/body_quat_w[0]: world = root_pos0 + R(root_quat0) @ ball_local.
            _rp0 = torch.tensor(ep_bp[0][0:1], dtype=torch.float32, device=device)
            _rq0 = torch.tensor(ep_bq[0][0:1], dtype=torch.float32, device=device)
            _bw = torch.tensor([list(ep_ball)], dtype=torch.float32, device=device)
            _bl0 = quat_apply_inverse(_rq0, _bw - _rp0)[0]
            ball_local_json = [float(_bl0[0]), float(_bl0[1]), float(_bl0[2])]
            save_episode_npz(
                collect_out / f"{name}.npz", db.fps,
                np.stack(ep_jp), np.stack(ep_jv), np.stack(ep_bp), np.stack(ep_bq),
                np.stack(ep_bav), np.stack(ep_rlv), np.stack(ep_rav),
                strike_frame=ep_strike_frame,
                meta={"ball": list(ep_ball),                       # world frame [x, y, z]
                      "ball_local": ball_local_json,               # pelvis(root) link frame of frame 0
                      "frames": T,
                      "clip": os.path.basename(selected_entry["clip"]["name"]) if selected_entry else ""},
            )
            ep_file_idx += 1
            ep_count += 1
            console.print(f"[green][EP {ep_count}/{collect_n}][/] saved {name}.npz  "
                          f"T={T}  strike_frame={ep_strike_frame}", end="\r\n")
        else:
            console.print(f"[red][EP][/] discarded (strike_frame={ep_strike_frame}, T={T})", end="\r\n")
        if ep_count >= collect_n:
            collect_done = True
            console.print(f"[green][COLLECT DONE][/] {ep_count} episodes -> {collect_out}", end="\r\n")
            save_ball_distribution_figure()
        else:
            spawn_new_ball("collect")            # continuous: next ball around the finish position
            ep_reset_buffers()

    if collecting:                      # start the first auto episode (reset to the serve pose once)
        spawn_new_ball("collect", do_reset=True)
        ep_reset_buffers()
    elif not gamepad_mode:              # ball-mode init planning (needs the closures above)
        lock_side()
        capture_ghost_query_frame()
        plan_strike_entry(reason="init")
        ghost_tracks.clear()
        update_ball_box()               # show the sampling range from the start

    # Headless (--headless): no viewer window; run the loop directly (collection only). All
    # drawing / sync is guarded by `viewer is not None`, so nothing renders.
    import contextlib
    if headless:
        _viewer_ctx = contextlib.nullcontext(None)
    else:
        _viewer_ctx = mujoco.viewer.launch_passive(model, mj_data, key_callback=viewer_key_callback,
                                                   show_left_ui=False, show_right_ui=False)
    with _viewer_ctx as viewer:
        while (viewer.is_running() if viewer is not None else not collect_done):
            if collect_done:
                break
            if collecting and (sim_step - ep_start_step) > collect_timeout * db.fps:
                console.print("[red][EP][/] timeout — resampling ball", end="\r\n")
                spawn_new_ball("collect", do_reset=True)   # stuck: reset to serve pose and retry
                ep_reset_buffers()

            pending = mouse_state.get("pending")
            if gamepad_mode or collecting:
                pending = None
                mouse_state["pending"] = None
            if pending is not None:
                mouse_state["pending"] = None
                p_xy, p_yaw = pending
                ball_x, ball_y = float(p_xy[0]), float(p_xy[1])
                ball_z = sample_ball_z_world()
                target_yaw_world = float(p_yaw)
                lock_side()
                capture_ghost_query_frame()
                plan_strike_entry(reason="mouse")
                ghost_tracks.clear()
                console.print(f"[yellow][MOUSE BALL][/] ({ball_x:.2f},{ball_y:.2f},{ball_z:.2f})  "
                              f"yaw={math.degrees(move_target_yaw_world):.1f}deg  "
                              f"T={time_remaining:.2f}s  side={locked_side[0].item():+d}", end="\r\n")

            if keyboard:
                key = keyboard.pop_key()
                if gamepad_mode and key is not None and handle_pad_command(key):
                    pass
                elif key == "b" and not gamepad_mode:
                    spawn_new_ball("random")
                    if collecting:
                        ep_reset_buffers()
                elif key == "r" and not gamepad_mode:
                    reset_to_fixed_pose()
                    lock_side()
                    capture_ghost_query_frame()
                    plan_strike_entry(reason="restart")
                    ghost_tracks.clear()
                    console.print("[cyan][RESTART][/]", end="\r\n")
                elif key == "q" or key == "\x03":
                    break

            time_remaining -= frame_dt
            pad_force = False

            if gamepad_mode:
                # ---- OFFICIAL gamepad control (direct-yaw semantics) ----------------------
                # Left stick = pure horizontal translation (does NOT turn the body).
                # Right stick X = the ONLY heading source (integrates desired yaw).
                pol = pad.poll() if pad is not None else None
                if pol is not None:
                    p_stick, p_turn, p_walk, p_strafe = pol
                    pad_input_active = (float(np.linalg.norm(p_stick)) > 0.0 or abs(p_turn) > 0.0
                                        or p_walk or p_strafe)
                    if pad_input_active:
                        pad_kb_override = False
                    if not pad_kb_override:      # keyboard command wins until the pad moves again
                        pad_stick[:] = p_stick
                        pad_turn_axis = p_turn
                        pad_strafe = p_strafe
                        if p_walk:
                            pad_walk = True
                        elif pad_input_active:
                            pad_walk = False
                else:
                    pad_turn_axis = 0.0

                cam_yaw = math.radians(float(viewer.cam.azimuth))
                fwd, side, back = pad_walk_speeds if pad_walk else pad_run_speeds
                # stick (fwd,left) reference frame: body (official direct-yaw) or camera
                ref_yaw = cam_yaw if pad_stick_frame == "camera" else pad_sim_yaw
                c, s = math.cos(ref_yaw), math.sin(ref_yaw)
                stick_world = np.array([c * pad_stick[0] - s * pad_stick[1],
                                        s * pad_stick[0] + c * pad_stick[1]])
                # official speed scaling: decompose in the SIM heading frame, scale fwd/side/back
                cy, sy = math.cos(pad_sim_yaw), math.sin(pad_sim_yaw)
                local = np.array([cy * stick_world[0] + sy * stick_world[1],
                                  -sy * stick_world[0] + cy * stick_world[1]])
                scale = np.array([fwd if local[0] > 0.0 else back, side])
                loc_scaled = scale * local
                pad_desired_vel = np.array([cy * loc_scaled[0] - sy * loc_scaled[1],
                                            sy * loc_scaled[0] + cy * loc_scaled[1]])
                # heading: right stick only (left stick NEVER turns the body). Optional
                # LT-strafe snaps the heading to the camera. Keyboard J/L steps +-30deg.
                if abs(pad_turn_axis) > 0.01:
                    pad_desired_yaw = pad_desired_yaw - pad_yaw_speed * frame_dt * pad_turn_axis
                elif pad_strafe:
                    pad_desired_yaw = cam_yaw

                # force search on sharp desired-velocity change OR turn-axis change
                # (official edge triggers, simplified)
                dv_rate = float(np.linalg.norm(pad_desired_vel - pad_prev_desired_vel)) / frame_dt
                pad_prev_desired_vel = pad_desired_vel.copy()
                turn_changed = (abs(pad_turn_axis) > 0.01) != (abs(pad_prev_turn) > 0.01) or (
                    pad_turn_axis * pad_prev_turn < -1e-4)
                pad_prev_turn = pad_turn_axis
                if pad_force_cooldown > 0.0:
                    pad_force_cooldown -= frame_dt
                elif dv_rate >= pad_force_vel_thr or turn_changed:
                    pad_force = True
                    pad_force_cooldown = rematch_interval * frame_dt

                # sim springs step (dt) + re-anchor sim position to the character root
                _, pad_sim_vel, pad_sim_acc = _spring_vel_update(
                    pad_sim_vel, pad_sim_acc, pad_desired_vel, pad_vel_halflife, frame_dt)
                goal_yaw = pad_sim_yaw + float(target_wrap_angle(pad_desired_yaw - pad_sim_yaw))
                pad_sim_yaw, pad_sim_yaw_vel = _spring_scalar_update(
                    pad_sim_yaw, pad_sim_yaw_vel, goal_yaw, pad_rot_halflife, frame_dt)

                # ---- OFFICIAL stepwise prediction at +20/40/60 frames ----------------------
                # Key to turn responsiveness (this is what the official
                # trajectory_*_predict chain does): the FUTURE desired yaw keeps integrating
                # the held turn axis, and the future desired velocity ROTATES with the
                # predicted heading — so a held turn asks for a real arc/large turn instead
                # of springing to today's heading only.
                root_xy = cur["root_pos"][0, :2].cpu().numpy().astype(np.float64)
                cur_yaw_r = float(yaw_from_quat_tensor(_facing_quat(cur["root_quat"]))[0].item())
                cr, sr = math.cos(-cur_yaw_r), math.sin(-cur_yaw_r)
                _pt = torch.zeros(N, 3, 3, device=device)
                _ph = torch.zeros(N, 3, 2, device=device)
                smooth_vel = torch.zeros(N, 3, device=device)
                v_loc = np.array([cr * pad_sim_vel[0] - sr * pad_sim_vel[1],
                                  sr * pad_sim_vel[0] + cr * pad_sim_vel[1]])
                smooth_vel[0, 0], smooth_vel[0, 1] = float(v_loc[0]), float(v_loc[1])
                dt_step = float(db.traj_offsets[0]) / db.fps          # 20 frames = 0.4 s
                yaw_p, yawv_p = pad_sim_yaw, pad_sim_yaw_vel
                des_yaw_p = pad_sim_yaw + float(target_wrap_angle(pad_desired_yaw - pad_sim_yaw))
                vel_p, acc_p = pad_sim_vel.copy(), pad_sim_acc.copy()
                pos_p = np.zeros(2, dtype=np.float64)
                stick_mag = float(np.linalg.norm(pad_stick))
                for j in range(3):
                    if abs(pad_turn_axis) > 0.01:                     # extrapolate held turn
                        des_yaw_p = des_yaw_p - pad_yaw_speed * dt_step * pad_turn_axis
                    yaw_p, yawv_p = _spring_scalar_update(yaw_p, yawv_p, des_yaw_p,
                                                          pad_rot_halflife, dt_step)
                    # future desired velocity rotates with the predicted heading (body frame)
                    if stick_mag > 1e-6:
                        ref_p = cam_yaw if pad_stick_frame == "camera" else yaw_p
                        cp, sp = math.cos(ref_p), math.sin(ref_p)
                        sw = np.array([cp * pad_stick[0] - sp * pad_stick[1],
                                       sp * pad_stick[0] + cp * pad_stick[1]])
                        cyp, syp = math.cos(yaw_p), math.sin(yaw_p)
                        lo = np.array([cyp * sw[0] + syp * sw[1], -syp * sw[0] + cyp * sw[1]])
                        sc = np.array([fwd if lo[0] > 0.0 else back, side])
                        des_v_p = np.array([cyp * (sc[0] * lo[0]) - syp * (sc[1] * lo[1]),
                                            syp * (sc[0] * lo[0]) + cyp * (sc[1] * lo[1])])
                    else:
                        des_v_p = np.zeros(2, dtype=np.float64)
                    dpos, vel_p, acc_p = _spring_vel_update(vel_p, acc_p, des_v_p,
                                                            pad_vel_halflife, dt_step)
                    pos_p = pos_p + dpos
                    pad_pred_pts[j] = root_xy + pos_p
                    pad_pred_yaws[j] = yaw_p
                    d_loc = np.array([cr * pos_p[0] - sr * pos_p[1], sr * pos_p[0] + cr * pos_p[1]])
                    h_world = np.array([math.cos(yaw_p), math.sin(yaw_p)])
                    h_loc = np.array([cr * h_world[0] - sr * h_world[1], sr * h_world[0] + cr * h_world[1]])
                    _pt[0, j, 0], _pt[0, j, 1] = float(d_loc[0]), float(d_loc[1])
                    _ph[0, j, 0], _ph[0, j, 1] = float(h_loc[0]), float(h_loc[1])
                d2target = float("inf")
                target_reached = False
                far = True
                traj_elapsed = 0.0
            else:
                # Desired motion comes from the target trajectory module. It is sampled in world space
                # and converted to the current G1 facing frame for the MM trajectory/root-velocity query.
                ball_local, racket_local, root_to_ball = compute_ball_and_racket_local()
                root_to_move_target = compute_move_target_local()
                d2target = float(root_to_move_target[0, :2].norm())
                target_reached = d2target <= entry_switch_xy_threshold
                far = not target_reached
                traj_elapsed = target_traj_elapsed()      # wall-clock; keeps the traj_done escape below
                traj_phase = advance_traj_phase()         # closed-loop phase; the query samples THIS
                _pt, _ph, smooth_vel = sample_target_query(traj_phase)

                if selected_entry is not None and not strike_active and not strike_completed:
                    cur_yaw = float(yaw_from_quat_tensor(_facing_quat(cur["root_quat"]))[0].item())
                    yaw_err = abs(float(target_wrap_angle(cur_yaw - move_target_yaw_world)))
                    traj_done = target_traj is not None and traj_elapsed >= float(target_traj.t[-1]) - 1e-4
                    pose_ready = target_reached and yaw_err <= entry_switch_yaw_threshold
                    timed_out = (
                        target_traj is not None
                        and (sim_step - target_traj_start_step) * frame_dt
                        >= float(target_traj.t[-1]) + entry_switch_timeout_s
                    )
                    if pose_ready or (target_reached and traj_done) or timed_out:
                        if timed_out and not target_reached:
                            print(
                                f"[STRIKE ENTER timeout] trajectory ended with "
                                f"xy_err={d2target:.3f}m yaw_err={math.degrees(yaw_err):.1f}deg; "
                                f"forcing transition",
                                end="\r\n",
                            )
                        enter_selected_strike()
                        target_reached = True
                        far = False

                if target_reached and selected_entry is None and not strike_active and not strike_completed:
                    smooth_vel.zero_()
                    _pt.zero_()
                    align_current_yaw_to_world(move_target_yaw_world)

            # OFFICIAL search cadence: steady timer (rematch_interval frames = search_time)
            # + end-of-anim (clip about to wrap) + leaving locomotion (event-driven)
            # + gamepad force-search on sharp input change.
            near_wrap = (db.clip_end[current_frame] - current_frame) <= 2
            can_rematch = not strike_active and not strike_completed
            should_rematch = can_rematch and ((sim_step % rematch_interval == 0 and sim_step > 0) or pad_force)
            # Event-driven rematch: rematch the instant the current clip would leave locomotion.
            leaving_move = False
            if can_rematch:
                nf, _ = db.advance_frames(current_frame)
                leaving_move = bool((~db.is_lafan_locomotion[nf]).any())

            # Move stage (OFFICIAL core): the query's pose half is copied from the current
            # matched frame; the trajectory half comes from the CasADi target trajectory.
            # Search is the official brute force: additive transition_cost on candidates,
            # the current frame competes at no extra cost, locomotion-only allowed mask.
            if can_rematch and (should_rematch or near_wrap.any() or leaving_move):
                cf = current_frame[0].item()
                end_of_anim = bool(near_wrap.any()) or leaving_move
                traj_pos_local = _pt[0, :, :2].detach().cpu().numpy()   # z-up facing (dx, dy) x3
                traj_dir_local = _ph[0].detach().cpu().numpy()          # z-up facing (hx, hy) x3
                query = db.make_query(cf, traj_pos_local, traj_dir_local)
                best_index, best_cost = db.search(
                    -1 if end_of_anim else cf,                          # official: at clip end, drop the current frame
                    query,
                    transition_cost=transition_cost,
                    ignore_range_end=ignore_range_end,
                    ignore_surrounding=ignore_surrounding,
                    allowed_mask=db.move_allowed_np,                    # locomotion minus low-posture frames
                    extra_candidate_cost=db.move_hand_hold_cost,
                )
                should_switch = torch.tensor([best_index != cf], dtype=torch.bool, device=device)
                best_frame = torch.tensor([best_index], dtype=torch.long, device=device)

                match_count += 1
                if log_matches:
                    delta = best_index - cf
                    sw = "SWITCH" if should_switch[0].item() else "KEEP"
                    lbl = "move" if bool(db.is_lafan_locomotion[cf]) else "strike"
                    lst = "green" if lbl == "move" else "bold red"
                    console.print(
                        f"Match {match_count:4d}  f{cf:5d}  [{lst}]{lbl:9s}[/]  "
                        f"base→entry {d2target:5.2f}m  "
                        f"search=[green]move  [/]  {sw:6s} D{delta:+5d}  best={best_cost:7.2f}",
                        highlight=False, end="\r\n")

                if should_switch.any():
                    sw = should_switch
                    blender.blend_duration = blend_frames / db.fps   # move switches use the short blend
                    old_root_pos = cur["root_pos"]
                    old_root_quat = cur["root_quat"]
                    old_root_linvel = cur["root_linvel"]
                    old_root_angvel = cur["root_angvel"]
                    old_joint_pos = cur["joint_pos"]
                    old_joint_vel = cur["joint_vel"]

                    new_root_pos_db = db.body_pos_w[best_frame, db.root_idx].clone()
                    new_root_quat_db = db.body_quat_w[best_frame, db.root_idx].clone()
                    old_fq = _facing_quat(old_root_quat)
                    new_fq = _facing_quat(new_root_quat_db)
                    new_yaw_offset = quat_mul(old_fq, quat_inv(new_fq))
                    root_yaw_offset[sw] = new_yaw_offset[sw]
                    rotated = quat_apply(root_yaw_offset, new_root_pos_db)
                    off = old_root_pos - rotated
                    off[:, 2] = 0.0
                    root_pos_offset[sw] = off[sw]

                    new_pos_t = rotated + root_pos_offset
                    new_quat_t = quat_mul(root_yaw_offset, new_root_quat_db)
                    new_lin_t = quat_apply(root_yaw_offset, db.body_linvel[best_frame, db.root_idx])
                    new_ang_t = quat_apply(root_yaw_offset, db.body_angvel[best_frame, db.root_idx])
                    blender.trigger(sw, old_root_pos, new_pos_t, old_root_quat, new_quat_t,
                                    old_root_linvel, new_lin_t, old_root_angvel, new_ang_t,
                                    old_joint_pos, db.joint_pos[best_frame],
                                    old_joint_vel, db.joint_vel[best_frame])
                    current_frame[sw] = best_frame[sw]

            # -- teleport (write qpos) --
            apply_frame(current_frame)
            if collecting:
                ep_record_frame()
            if _sp is not None:
                # body-frame decomposition: forward / lateral (via facing quat)
                _lv = cur["root_linvel"][0]
                _bfq = _facing_quat(cur["root_quat"])
                _lvb = quat_apply_inverse(_bfq, _lv.unsqueeze(0))[0]
                _sp["mag"].append(float(_lv[:2].norm()))
                _sp["fwd"].append(float(_lvb[0]))
                _sp["lat"].append(float(_lvb[1]))
                if sim_step % 4 == 0:
                    _sp["l_mag"].set_ydata(list(_sp["mag"]))
                    _sp["l_fwd"].set_ydata(list(_sp["fwd"]))
                    _sp["l_lat"].set_ydata(list(_sp["lat"]))
                    _sp["fig"].canvas.draw_idle()
                    _sp["fig"].canvas.flush_events()
            if log_height and sim_step % log_height_interval == 0:
                _tp = neck_pos()
                _tv = cur["root_linvel"][0]
                console.print(
                    f"[dim]neck[/] x={_tp[0]:+7.3f}  y={_tp[1]:+7.3f}  z={_tp[2]:6.3f}  "
                    f"|v|={float(_tv[:2].norm()):4.2f} m/s",
                    highlight=False, end="\r\n")
            if target_reached and selected_entry is None and not strike_active and not strike_completed:
                mj_data.qvel[:] = 0.0
                cur["root_linvel"].zero_()
                cur["root_angvel"].zero_()
                cur["joint_vel"].zero_()

            # -- markers (world axes, ball=yellow, racket center=green) --
            # Collection speed mode: skip all marker drawing and throttle viewer.sync
            # (markers ~40 capsules/frame + GUI sync were a large share of per-frame cost).
            skip_draw = viewer is None or (collecting and not collect_viz)
            if viewer is not None:
                viewer.user_scn.ngeom = 0
            if not skip_draw:
                if not uses_shared_court:
                    _draw_tennis_court_overlay(viewer)  # fallback: regulation court centered at world origin
                # Compact cyan root-heading marker; hide it after the reference ends.
                if viz_root_heading and not strike_completed:
                    _root_heading_yaw = float(
                        yaw_from_quat_tensor(_facing_quat(cur["root_quat"]))[0].item()
                    )
                    _root_heading_base = np.array(
                        [cur["root_pos"][0, 0].item(), cur["root_pos"][0, 1].item(), 0.07]
                    )
                    _root_heading_tip = _root_heading_base + np.array(
                        [math.cos(_root_heading_yaw), math.sin(_root_heading_yaw), 0.0]
                    ) * 0.35
                    add_direction_arrow(
                        viewer,
                        _root_heading_base,
                        _root_heading_tip,
                        [0.1, 0.9, 0.95, 0.95],
                        width=0.025,
                    )
                if not gamepad_mode:
                    if incoming_ball_viz:
                        _ball_path = incoming_ball_path()
                        # Pale dashed arc: these are user-scene decorations only and therefore
                        # never participate in MuJoCo contact or dynamics.
                        for _path_i, (_a, _b) in enumerate(
                            zip(_ball_path[:-1], _ball_path[1:])
                        ):
                            if (_path_i // 2) % 2 == 0:
                                add_arrow(
                                    viewer,
                                    _a,
                                    _b,
                                    [1.0, 0.82, 0.20, 0.72],
                                    width=0.009,
                                )
                        _ball_phase = incoming_ball_progress()
                        _ball_sample = _ball_phase * (len(_ball_path) - 1)
                        _ball_i0 = min(int(math.floor(_ball_sample)), len(_ball_path) - 1)
                        _ball_i1 = min(_ball_i0 + 1, len(_ball_path) - 1)
                        _ball_alpha = _ball_sample - _ball_i0
                        _ball_pos = (
                            (1.0 - _ball_alpha) * _ball_path[_ball_i0]
                            + _ball_alpha * _ball_path[_ball_i1]
                        )
                        add_sphere(
                            viewer,
                            _ball_path[-1],
                            [1.0, 0.30, 0.05, 0.72],
                            r=0.042,
                        )
                        add_sphere(viewer, _ball_pos, [1, 1, 0, 1], r=0.06)
                    else:
                        add_sphere(viewer, [ball_x, ball_y, ball_z], [1, 1, 0, 1], r=0.06)
                    if viz_mode_switch_radius:
                        # Near/far planning boundary centered on the current robot root.
                        _R = float(target_ctoc_cfg.mode_switch_distance)
                        _cx = cur["root_pos"][0, 0].item(); _cy = cur["root_pos"][0, 1].item()
                        _segs = 48
                        _ring = [np.array([_cx + _R * math.cos(TWO_PI * _k / _segs),
                                           _cy + _R * math.sin(TWO_PI * _k / _segs), 0.02])
                                 for _k in range(_segs + 1)]
                        for _a, _b in zip(_ring[:-1], _ring[1:]):
                            add_arrow(viewer, _a, _b, [1.0, 0.85, 0.05, 0.85], width=0.010)
                if viz_height:
                    # neck (shoulder mid-point) marker (magenta) + vertical drop line to ground
                    _tp = neck_pos().copy()
                    add_sphere(viewer, _tp, [1.0, 0.15, 0.75, 0.9], r=0.05)
                    add_arrow(viewer, [_tp[0], _tp[1], 0.005], _tp, [1.0, 0.15, 0.75, 0.45], width=0.006)
                    add_sphere(viewer, [_tp[0], _tp[1], 0.01], [1.0, 0.15, 0.75, 0.9], r=0.03)
                if target_traj is not None and not clean_view:
                    R_origin = rot2_np(target_traj_origin_yaw)
                    path_xy = target_traj_origin_xy[None, :] + (R_origin @ target_traj.pos.T).T
                    stride = max(1, len(path_xy) // 100)
                    path = path_xy[::stride]
                    z_path = 0.035
                    for a, b in zip(path[:-1], path[1:]):
                        add_arrow(viewer, [a[0], a[1], z_path], [b[0], b[1], z_path],
                                  [0.15, 0.45, 1.0, 0.50], width=0.015)
                    # Optimized-trajectory heading: short green arrow at each sampled point
                    # (world yaw = origin_yaw + local traj yaw).
                    # Show roughly six heading samples over the full trajectory.
                    hstride = max(1, math.ceil(len(path_xy) / 6))
                    for _i in range(0, len(path_xy), hstride):
                        _wy = target_traj_origin_yaw + float(target_traj.yaw[_i])
                        _b = np.array([path_xy[_i, 0], path_xy[_i, 1], z_path + 0.02])
                        add_direction_arrow(
                            viewer,
                            _b,
                            _b + np.array([math.cos(_wy), math.sin(_wy), 0.0]) * 0.45,
                            [0.1, 0.9, 0.3, 0.9],
                            width=0.018,
                        )
                    add_sphere(viewer, [path_xy[-1, 0], path_xy[-1, 1], z_path + 0.035],
                               [1.0, 0.15, 0.1, 1.0], r=0.055)
                if (
                    not clean_view
                    and mouse_state["manual_edit"]
                    and mouse_state["draft_xy"] is not None
                ):
                    draft_xy = np.asarray(mouse_state["draft_xy"], dtype=np.float64)
                    draft_base = np.array([draft_xy[0], draft_xy[1], 0.025])
                    add_flat_disc(
                        viewer,
                        draft_base,
                        [1.0, 0.05, 0.05, 0.92],
                        r=0.11,
                        thickness=0.006,
                    )  # selected target XY on the court plane
                    if mouse_select_yaw:                                            # heading draft arrow
                        draft_yaw = float(mouse_state["draft_yaw"] if mouse_state["draft_yaw"] is not None else target_yaw_world)
                        draft_vec = np.array([math.cos(draft_yaw), math.sin(draft_yaw), 0.0])
                        add_direction_arrow(
                            viewer,
                            draft_base,
                            draft_base + draft_vec * 0.90,
                            [1.0, 0.85, 0.0, 1.0],
                            width=0.03,
                        )
                if not clean_view:
                    _wp, _wq = wrist_world()
                    _racket_c = db.compute_racket_pos(_wp, _wq)[0].cpu().numpy()
                    if racket_center_path_viz and (strike_active or strike_completed):
                        if (
                            strike_active
                            and (
                                not racket_center_path
                                or np.linalg.norm(_racket_c - racket_center_path[-1]) >= 0.01
                            )
                        ):
                            racket_center_path.append(_racket_c.copy())
                        draw_dashed_path_with_arrow(
                            viewer,
                            racket_center_path,
                            [0.55, 1.0, 0.60, 0.92],
                            dash_length=0.10,
                            gap_length=0.07,
                            width=0.009,
                            arrow_length=0.18,
                            arrow_spacing=0.50,
                            arrow_rgba=[0.02, 0.38, 0.10, 1.0],
                        )
                    add_sphere(
                        viewer,
                        _racket_c,
                        [0, 1, 0, 1],
                        r=0.05,
                    )  # racket center (= wrist + racket_offset)

                # -- viz: query desired vel (ORANGE) vs matched db-frame hip vel (CYAN)
                #    + matched db traj (MAGENTA future root path).
                # Official features are y-up internal; horizontal (x_int, z_int)
                # == z-up facing (-y_local, x_local)  =>  (x_local, y_local) = (z_int, -x_int).
                _cfv = current_frame[0].item()
                _bg = np.array([cur["root_pos"][0, 0].item(), cur["root_pos"][0, 1].item(), 0.12])
                _fq = _facing_quat(cur["root_quat"])
                _df = db.denormalized_features(_cfv)


                def _rot(vx, vy):  # facing-frame XY -> world 3-vec
                    return quat_apply(_fq, torch.tensor([[float(vx), float(vy), 0.0]],
                                                        device=device))[0].cpu().numpy()

                def _vbase_tip(vx, vy, scale=0.25, maxlen=1.5):
                    w = _rot(vx, vy) * scale
                    n = float(np.linalg.norm(w))
                    return _bg + (w * (maxlen / n) if n > maxlen else w)

                if viz_motion_matching:
                    _bl, _, _ = compute_ball_and_racket_local()
                    if target_reached and selected_entry is None and not strike_active and not strike_completed:
                        _qv = np.zeros(2, dtype=np.float32)
                    else:
                        _qv = smooth_vel[0, :2].detach().cpu().numpy()             # query root velocity
                    add_arrow(viewer, _bg, _vbase_tip(_qv[0], _qv[1]),
                              [1, 0.5, 0, 1], width=0.022)                         # orange
                    # Matched-frame hip velocity: features[12:15] internal -> z-up facing.
                    add_arrow(viewer, _bg, _vbase_tip(float(_df[14]), float(-_df[12])),
                              [0, 1, 1, 1], width=0.022)
                    # Motion-matching trajectory diagnostics are useful during locomotion, but
                    # become distracting after the controller enters the strike clip.
                    if not strike_active:
                        _prev = _bg
                        for _j in range(3):                                      # matched db traj (magenta)
                            _p = _bg + _rot(float(_df[16 + 2 * _j]), float(-_df[15 + 2 * _j]))
                            add_sphere(viewer, _p, [1, 0, 1, 1], r=0.04)
                            add_arrow(viewer, _prev, _p, [1, 0, 1, 0.6], width=0.01)
                            _prev = _p

                        # QUERY traj (WHITE) = target samples used by the move-mode MM query.
                        if far:
                            _prev = _bg
                            for _qi in range(len(db.traj_offsets)):
                                _qp = _bg + _rot(float(_pt[0, _qi, 0]), float(_pt[0, _qi, 1]))
                                add_sphere(viewer, _qp, [1, 1, 1, 1], r=0.045)
                                add_arrow(viewer, _prev, _qp, [1, 1, 1, 0.75], width=0.012)
                                _prev = _qp

                # ROOT lag: query root (BLUE = trajectory phase point, where the query expects the
                # robot to be now) vs robot root (GREEN = actual). Their gap = how far the robot
                # trails the (now open-loop) trajectory phase. Green line connects them.
                if viz_root_lag and target_traj is not None:
                    _robot_root = np.array([cur["root_pos"][0, 0].item(),
                                            cur["root_pos"][0, 1].item(), 0.03])
                    _pl, _, _ = interp_target_traj(target_traj, np.asarray([target_traj_phase], dtype=np.float64))
                    _qxy = target_traj_origin_xy + rot2_np(target_traj_origin_yaw) @ _pl[0]
                    _query_root = np.array([_qxy[0], _qxy[1], 0.03])
                    add_sphere(viewer, _query_root, [0.2, 0.4, 1.0, 1.0], r=0.06)   # blue = query root
                    add_sphere(viewer, _robot_root, [0.1, 1.0, 0.2, 1.0], r=0.06)   # green = robot root
                    add_arrow(viewer, _robot_root, _query_root, [0.1, 1.0, 0.2, 0.6], width=0.010)

                if not gamepad_mode:
                    record_ghost_snapshot()
                    draw_ghost(viewer)   # persistent snapshots of the executed move-to-strike sequence
            if viewer is not None and (not skip_draw or sim_step % collect_sync_interval == 0):
                viewer.sync()

            strike_end_frame = None
            if strike_active and selected_entry is not None:
                strike_clip = selected_entry["clip"]
                strike_end_frame = min(
                    int(strike_clip["end"]) - 1,
                    int(strike_clip["strike"]) + strike_postcontact_frames,
                )
            if (strike_active and strike_end_frame is not None
                    and current_frame[0].item() >= strike_end_frame):
                strike_active = False
                strike_completed = True
                current_frame.fill_(strike_end_frame)
                cur["root_linvel"].zero_()
                cur["root_angvel"].zero_()
                cur["joint_vel"].zero_()
                if collecting:
                    ep_finalize()        # save episode + auto-serve the next ball (or finish)
                else:
                    print("[STRIKE DONE] holding final pose; press B/R or mouse-pick a new target", end="\r\n")
            elif not strike_completed:
                current_frame, overflow = db.advance_frames(current_frame)
                if overflow.any():
                    ov = overflow
                    old_root_pos = cur["root_pos"]
                    old_root_quat = cur["root_quat"]
                    new_root_pos_db = db.body_pos_w[current_frame, db.root_idx].clone()
                    new_root_quat_db = db.body_quat_w[current_frame, db.root_idx].clone()
                    old_fq = _facing_quat(old_root_quat)
                    new_fq = _facing_quat(new_root_quat_db)
                    root_yaw_offset[ov] = quat_mul(old_fq, quat_inv(new_fq))[ov]
                    rotated = quat_apply(root_yaw_offset, new_root_pos_db)
                    off = old_root_pos - rotated
                    off[:, 2] = 0.0
                    root_pos_offset[ov] = off[ov]

            sim_step += 1
            if args.max_frames is not None and sim_step >= args.max_frames:
                print("[viewer smoke exit]", end="\r\n", flush=True)
                if keyboard is not None:
                    keyboard.stop()
                os._exit(0)
            if args.playback_fps > 0:
                time.sleep(max(0, 1.0 / args.playback_fps - 0.001))
            else:
                time.sleep(frame_dt)

        # MuJoCo's passive-viewer GL-context teardown can throw GLXBadContext and HANG on quit
        # on some X/driver setups. Restore the terminal and hard-exit instead of unwinding the
        # viewer (whose __exit__ destroys the GL context) — that's what was hanging on Ctrl+C.
        if keyboard is not None:
            keyboard.stop()
        print("\r\n[exit]", end="\r\n", flush=True)
        os._exit(0)


PHASE_LABEL_NAMES = {0: "move", 1: "strike", 2: "recovery", -1: "?"}
PHASE_LABEL_STYLE = {0: "green", 1: "bold red", 2: "blue", -1: "dim"}


def main():
    parser = argparse.ArgumentParser(description="Tennis Motion Matching (OFFICIAL move-MM core) — MuJoCo viewer")
    parser.add_argument("--db_file", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "motion_db/official_tennis_runwalk_hand33"),
                        help="Official MM resource DIRECTORY (features.bin/g1_state.npz/metadata.json/strike_metadata.npz)")
    parser.add_argument("--interactive", action="store_true", help="B=new ball R=restart Q=quit (terminal keys)")
    parser.add_argument("--control", type=str, choices=("ball", "gamepad"), default=None,
                        help="ball = strike-entry pipeline (default); gamepad = pure locomotion "
                             "following W/A/S/D or a GLFW joystick — no ball, no strike")
    parser.add_argument("--speed_plot", action="store_true",
                        help="Open a live matplotlib window plotting the root speed (|v|, forward, lateral).")
    parser.add_argument("--headless", action="store_true",
                        help="No viewer window (collection only); runs the loop directly, nothing renders.")
    parser.add_argument("--collect", type=int, default=0,
                        help="Auto-collect N move+strike episodes (random balls), saved as "
                             "djvkovic_npz_data_v5-compatible npz+json (see yaml `collect:` section)")
    parser.add_argument("--ghost", action="store_true",
                        help="Executed-motion trail: retain one static G1 pose every configured N frames "
                             "until the next target update.")
    parser.add_argument(
        "--clean_view",
        "--clean-view",
        action="store_true",
        help="Render only the live G1, tennis court, incoming visual ball trajectory, and strike target; overrides all other visualization options, including --ghost.",
    )
    parser.add_argument("--ghost_count", "--ghost-count", type=int, default=1,
                        help="Deprecated compatibility option; executed-motion ghost count is interval-driven.")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent / "tennis_official_mm_config.yaml"))
    parser.add_argument("--playback_fps", type=float, default=0)
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Run only this many viewer frames, for automated smoke tests")
    parser.add_argument("--check_joint_order", action="store_true",
                        help="Print the IL<->MuJoCo joint-name mapping and exit")
    args = parser.parse_args()

    if args.check_joint_order:
        print(f"{'mj_idx':>6} {'MuJoCo (DFS, qpos[7:])':<28} {'<- IL idx':>9}  {'Isaac Lab name':<28}")
        for mj_idx, name in enumerate(MUJOCO_JOINT_NAMES):
            il_idx = ISAACLAB_TO_MUJOCO[mj_idx]
            ok = "OK" if ISAACLAB_JOINT_NAMES[il_idx] == name else "!!MISMATCH!!"
            print(f"{mj_idx:>6} {name:<28} {il_idx:>9}  {ISAACLAB_JOINT_NAMES[il_idx]:<28} {ok}")
        return

    if not args.db_file:
        parser.error("--db_file is required (unless --check_joint_order)")
    cfg = load_cfg(args.config)
    keyboard = KeyboardController() if args.interactive else None
    try:
        run(args, cfg, keyboard)
    finally:
        if keyboard:
            keyboard.stop()


if __name__ == "__main__":
    main()
