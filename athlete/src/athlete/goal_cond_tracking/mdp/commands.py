from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from .analytic_strike import (
  align_quaternion_axis_to_vector,
  inverse_racket_velocity_and_normal,
  solve_low_arc_tennis_velocity,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))
MUJOCO_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
ISAACLAB_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
)

# Exact articulation order enforced by tennis/scripts/rsl_rl/collect_policy_rollouts.py.
# The rollout NPZ files omit body names, so use this canonical layout instead of
# guessing between bodies with coincident origins (notably waist_roll_link and
# torso_link).
ISAACLAB_G1_34_BODY_NAMES = (
  "pelvis",
  "left_hip_pitch_link",
  "pelvis_contour_link",
  "right_hip_pitch_link",
  "waist_yaw_link",
  "left_hip_roll_link",
  "right_hip_roll_link",
  "waist_roll_link",
  "left_hip_yaw_link",
  "right_hip_yaw_link",
  "torso_link",
  "left_knee_link",
  "right_knee_link",
  "head_link",
  "left_shoulder_pitch_link",
  "logo_link",
  "right_shoulder_pitch_link",
  "left_ankle_pitch_link",
  "right_ankle_pitch_link",
  "left_shoulder_roll_link",
  "right_shoulder_roll_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
  "left_wrist_roll_link",
  "right_wrist_roll_link",
  "left_wrist_pitch_link",
  "right_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "racket_link",
)
ISAACLAB_TO_MUJOCO = np.array(
  [ISAACLAB_JOINT_NAMES.index(name) for name in MUJOCO_JOINT_NAMES],
  dtype=np.int64,
)


def transform_reference_frame0_target(
  target_pos_frame0: torch.Tensor,
  reference_anchor_pos_0_w: torch.Tensor,
  reference_anchor_quat_0_w: torch.Tensor,
) -> torch.Tensor:
  """Convert a frame-0 anchor-local target directly to world coordinates."""
  return reference_anchor_pos_0_w + quat_apply(
    reference_anchor_quat_0_w, target_pos_frame0
  )


def transform_startup_frame_ground_target(
  target_pos_startup: torch.Tensor,
  startup_anchor_pos_w: torch.Tensor,
  startup_anchor_yaw_w: torch.Tensor,
  env_origin_w: torch.Tensor,
) -> torch.Tensor:
  """Convert a startup-anchor-local ground target to world coordinates."""
  target_pos_w = startup_anchor_pos_w + quat_apply(
    startup_anchor_yaw_w, target_pos_startup
  )
  target_pos_w[:, 2] = env_origin_w[:, 2] + target_pos_startup[:, 2]
  return target_pos_w


def reference_frame0_alignment(
  reference_anchor_quat_0_w: torch.Tensor,
  robot_anchor_pos_w: torch.Tensor,
  robot_anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return position and yaw transform aligning reference frame 0 to the robot."""
  align_yaw_w = yaw_quat(
    quat_mul(robot_anchor_quat_w, quat_inv(reference_anchor_quat_0_w))
  )
  return robot_anchor_pos_w, align_yaw_w


def detect_joint_order(model, joint_pos: np.ndarray) -> str:
  """Detect whether joint_pos is in MuJoCo (DFS) or Isaac Lab (BFS) order."""
  n_joints = min(joint_pos.shape[1], model.njnt - 1)  # skip freejoint
  jnt_lo = np.zeros(n_joints)
  jnt_hi = np.zeros(n_joints)
  for i in range(n_joints):
    jid = i + 1
    if model.jnt_limited[jid]:
      jnt_lo[i] = model.jnt_range[jid, 0]
      jnt_hi[i] = model.jnt_range[jid, 1]
    else:
      jnt_lo[i] = -1e6
      jnt_hi[i] = 1e6

  vals_dfs = joint_pos[:, :n_joints]
  violations_dfs = np.sum((vals_dfs < jnt_lo - 0.02) | (vals_dfs > jnt_hi + 0.02))

  vals_bfs = joint_pos[:, ISAACLAB_TO_MUJOCO[:n_joints]]
  violations_bfs = np.sum((vals_bfs < jnt_lo - 0.02) | (vals_bfs > jnt_hi + 0.02))
  return "mujoco" if violations_dfs <= violations_bfs else "isaaclab"


def infer_body_columns(
  model: mujoco.MjModel,
  joint_pos_mj: np.ndarray,
  body_pos_w: np.ndarray,
  body_quat_w: np.ndarray,
  body_names: tuple[str, ...],
  *,
  max_frames: int = 24,
) -> np.ndarray:
  """Infer NPZ body columns for the requested MuJoCo robot body indexes.

  Older motion NPZs do not store body names.  Their body arrays can be in the
  IsaacLab articulation order while MJLab indexes bodies in MuJoCo tree order.
  Since root pose + joint positions fully determine body poses, use MuJoCo FK to
  match each requested body to the closest NPZ body column.
  """
  if body_pos_w.shape[1] == 0:
    raise ValueError("Motion file contains no body positions.")

  n_frames = body_pos_w.shape[0]
  frame_ids = np.linspace(0, n_frames - 1, min(max_frames, n_frames), dtype=np.int64)
  data = mujoco.MjData(model)
  fk_pos = []
  fk_quat = []
  for frame in frame_ids:
    data.qpos[:3] = body_pos_w[frame, 0]
    data.qpos[3:7] = body_quat_w[frame, 0]
    data.qpos[7 : 7 + joint_pos_mj.shape[1]] = joint_pos_mj[frame]
    mujoco.mj_forward(model, data)
    fk_pos.append(data.xpos.copy())
    fk_quat.append(data.xquat.copy())
  fk_pos = np.stack(fk_pos, axis=0)
  fk_quat = np.stack(fk_quat, axis=0)

  model_body_ids = []
  for body_name in body_names:
    candidates = (body_name, f"robot/{body_name}")
    body_id = -1
    for candidate in candidates:
      body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
      if body_id >= 0:
        break
    if body_id < 0:
      suffix_matches = [
        i
        for i in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith(
          f"/{body_name}"
        )
      ]
      if len(suffix_matches) == 1:
        body_id = suffix_matches[0]
    if body_id < 0:
      raise ValueError(f"Could not find body {body_name!r} in MuJoCo model.")
    model_body_ids.append(body_id)

  model_body_ids_np = np.array(model_body_ids, dtype=np.int64)

  # Collected G1 motions have a declared 34-body IsaacLab layout. Validate the
  # declared mapping against FK before using it so unrelated 34-body datasets
  # still fall back to inference.
  if body_pos_w.shape[1] == len(ISAACLAB_G1_34_BODY_NAMES) and all(
    name in ISAACLAB_G1_34_BODY_NAMES for name in body_names
  ):
    known_columns = np.array(
      [ISAACLAB_G1_34_BODY_NAMES.index(name) for name in body_names],
      dtype=np.int64,
    )
    known_pos_error = np.linalg.norm(
      body_pos_w[frame_ids][:, known_columns] - fk_pos[:, model_body_ids_np],
      axis=-1,
    )
    known_quat_dot = np.abs(
      np.sum(
        body_quat_w[frame_ids][:, known_columns] * fk_quat[:, model_body_ids_np],
        axis=-1,
      )
    )
    known_ori_error = 2.0 * np.arccos(np.clip(known_quat_dot, -1.0, 1.0))
    max_pos_error = float(known_pos_error.max())
    max_ori_error = float(known_ori_error.max())
    if max_pos_error <= 1.0e-3 and max_ori_error <= 1.0e-2:
      print(
        "[INFO] Using deterministic IsaacLab G1 34-body mapping: "
        f"{dict(zip(body_names, known_columns.tolist(), strict=True))}"
      )
      return known_columns
    print(
      "[WARN] Declared IsaacLab G1 34-body mapping failed FK validation "
      f"(position={max_pos_error:.4g} m, orientation={max_ori_error:.4g} rad); "
      "falling back to position inference."
    )

  distances = np.linalg.norm(
    body_pos_w[frame_ids, :, None, :] - fk_pos[:, None, model_body_ids_np, :],
    axis=-1,
  ).mean(axis=0)
  inferred = distances.argmin(axis=0).astype(np.int64)

  direct_columns = np.arange(len(body_names), dtype=np.int64)
  direct_ok = body_pos_w.shape[1] >= len(body_names) and np.allclose(
    inferred, direct_columns
  )
  if not direct_ok:
    parts = []
    for body_name, body_id, npz_col, err in zip(
      body_names,
      model_body_ids_np.tolist(),
      inferred.tolist(),
      distances[inferred, np.arange(len(inferred))].tolist(),
      strict=True,
    ):
      model_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
      parts.append(f"{body_name}->{model_body_name}:npz[{npz_col}] err={err:.4g}")
    print("[INFO] Remapping motion body columns by FK match: " + ", ".join(parts))

  return inferred


def staged_landing_target_std(
  global_step: int,
  stage_steps: tuple[int, ...],
  stage_stds: tuple[tuple[float, float, float], ...],
) -> tuple[int, tuple[float, float, float]]:
  """Resolve the active landing-target standard deviation curriculum stage."""
  if global_step < 0:
    raise ValueError("Curriculum global step must be non-negative.")
  if not stage_steps or len(stage_steps) != len(stage_stds):
    raise ValueError("Landing curriculum steps and stds must be non-empty and aligned.")
  if stage_steps[0] != 0 or any(
    current <= previous for previous, current in pairwise(stage_steps)
  ):
    raise ValueError("Landing curriculum steps must start at zero and increase.")
  if any(value < 0.0 for std in stage_stds for value in std):
    raise ValueError("Landing curriculum std values must be non-negative.")

  stage = 0
  for index, start_step in enumerate(stage_steps):
    if global_step < start_step:
      break
    stage = index
  return stage, stage_stds[stage]


def linear_landing_target_std(
  global_step: int,
  duration_steps: int,
  initial_std: tuple[float, float, float],
  final_std: tuple[float, float, float],
) -> tuple[float, tuple[float, float, float]]:
  """Interpolate sampling stds using control steps, independent of env count."""
  if global_step < 0 or duration_steps <= 0:
    raise ValueError("Landing curriculum requires non-negative steps and a positive duration.")
  if any(not math.isfinite(v) or v < 0.0 for v in (*initial_std, *final_std)):
    raise ValueError("Landing curriculum stds must be finite and non-negative.")
  progress = min(global_step / duration_steps, 1.0)
  std = tuple(a + progress * (b - a) for a, b in zip(initial_std, final_std, strict=True))
  return progress, std


def truncate_landing_targets_at_net(
  samples_w: torch.Tensor,
  mean_w: torch.Tensor,
  variance_x: torch.Tensor,
  covariance_xy: torch.Tensor,
  minimum_x_w: torch.Tensor,
) -> torch.Tensor:
  """Condition a planar Gaussian on world X >= minimum, retaining XY covariance."""
  std_x = variance_x.sqrt()
  alpha = (minimum_x_w - mean_w[:, 0]) / std_x.clamp_min(1e-12)
  cdf_lower = 0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))
  probability = cdf_lower + (1.0 - cdf_lower) * torch.rand_like(std_x)
  eps = torch.finfo(samples_w.dtype).eps
  quantile = math.sqrt(2.0) * torch.erfinv(2.0 * probability.clamp(eps, 1.0 - eps) - 1.0)
  x = mean_w[:, 0] + std_x * quantile
  x = torch.maximum(x, minimum_x_w)
  result = samples_w.clone()
  # Change X while keeping the independent conditional-Y residual unchanged.
  result[:, 1] += covariance_xy / variance_x.clamp_min(1e-12) * (x - samples_w[:, 0])
  result[:, 0] = x
  return result


def staged_target_pos_std_scale(
  global_step: int,
  stage_steps: tuple[int, ...],
  stage_scales: tuple[float, ...],
) -> tuple[int, float]:
  """Resolve the active sampled-target position std scale."""
  if global_step < 0:
    raise ValueError("Curriculum global step must be non-negative.")
  if not stage_steps or len(stage_steps) != len(stage_scales):
    raise ValueError(
      "Target curriculum steps and scales must be non-empty and aligned."
    )
  if stage_steps[0] != 0 or any(
    current <= previous for previous, current in pairwise(stage_steps)
  ):
    raise ValueError("Target curriculum steps must start at zero and increase.")
  if any(scale < 0.0 for scale in stage_scales):
    raise ValueError("Target curriculum scales must be non-negative.")

  stage = 0
  for index, start_step in enumerate(stage_steps):
    if global_step < start_step:
      break
    stage = index
  return stage, stage_scales[stage]


def rigid_point_linear_velocity(
  parent_linear_velocity_w: torch.Tensor,
  parent_angular_velocity_w: torch.Tensor,
  parent_quat_w: torch.Tensor,
  point_offset_parent: torch.Tensor,
) -> torch.Tensor:
  """Compute a rigidly attached point's world-frame linear velocity."""
  point_offset_w = quat_apply(parent_quat_w, point_offset_parent)
  return parent_linear_velocity_w + torch.cross(
    parent_angular_velocity_w,
    point_offset_w,
    dim=-1,
  )


def tennis_target_pos_std_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str,
  stage_steps: tuple[int, ...],
  stage_scales: tuple[float, ...],
) -> dict[str, float]:
  """Scale sampled strike-target position noise at fixed training stages."""
  del env_ids
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiTargetMotionCommand):
    raise TypeError(f"Command {command_name!r} must be a multi-target motion command.")
  stage, scale = staged_target_pos_std_scale(
    env.common_step_counter,
    stage_steps,
    stage_scales,
  )
  command.target_pos_std_scale = scale
  return {"stage": float(stage), "scale": scale}


def tennis_landing_target_std_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str,
  stage_steps: tuple[int, ...],
  stage_stds: tuple[tuple[float, float, float], ...],
) -> dict[str, float]:
  """Update landing-target randomization globally when environments reset."""
  del env_ids
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiTargetMotionCommand):
    raise TypeError(f"Command {command_name!r} must be a multi-target motion command.")
  stage, std = staged_landing_target_std(
    env.common_step_counter,
    stage_steps,
    stage_stds,
  )
  command.set_landing_target_std(std)
  return {
    "stage": float(stage),
    "std_x": std[0],
    "std_y": std[1],
  }


def tennis_landing_target_linear_std_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str,
) -> dict[str, float]:
  """Report the linear schedule; sampling also updates it between episode resets."""
  del env_ids
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiTargetMotionCommand):
    raise TypeError(f"Command {command_name!r} must be a multi-target motion command.")
  return command.update_landing_target_std_curriculum()


# motion loader takes in the motion file, body indexes, and constructs an object which has positions velocities and angles for the motion. We index into this during motion tracking.
class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    body_indexes: torch.Tensor,
    body_names: tuple[str, ...],
    model: mujoco.MjModel | None = None,
    device: str = "cpu",
  ) -> None:
    self._body_indexes = body_indexes
    self._body_names = body_names
    with np.load(motion_file) as data:
      joint_pos = data["joint_pos"].astype(np.float32)
      joint_vel = data["joint_vel"].astype(np.float32)
      body_pos_w = data["body_pos_w"].astype(np.float32)
      body_quat_w = data["body_quat_w"].astype(np.float32)
      body_lin_vel_w = data["body_lin_vel_w"].astype(np.float32)
      body_ang_vel_w = data["body_ang_vel_w"].astype(np.float32)
      declared_body_names = (
        tuple(str(name) for name in data["body_names"].tolist())
        if "body_names" in data.files
        else None
      )

    if model is not None:
      joint_order = detect_joint_order(model, joint_pos)
      if joint_order == "isaaclab":
        print(f"[INFO] Remapping motion joints from IsaacLab order for {motion_file}")
        joint_pos = joint_pos[:, ISAACLAB_TO_MUJOCO]
        joint_vel = joint_vel[:, ISAACLAB_TO_MUJOCO]

    if declared_body_names is not None:
      if len(declared_body_names) != body_pos_w.shape[1]:
        raise ValueError(
          f"body_names has {len(declared_body_names)} entries but body arrays "
          f"have {body_pos_w.shape[1]} columns in {motion_file}"
        )
      if len(set(declared_body_names)) != len(declared_body_names):
        raise ValueError(f"body_names contains duplicates in {motion_file}")
      missing_names = [
        name for name in self._body_names if name not in declared_body_names
      ]
      if missing_names:
        raise ValueError(
          f"Requested bodies are absent from body_names in {motion_file}: "
          f"{missing_names}"
        )
      body_columns = np.array(
        [declared_body_names.index(name) for name in self._body_names],
        dtype=np.int64,
      )
      print(
        "[INFO] Using declared NPZ body_names mapping: "
        f"{dict(zip(self._body_names, body_columns.tolist(), strict=True))}"
      )
    elif model is not None:
      body_columns = infer_body_columns(
        model, joint_pos, body_pos_w, body_quat_w, self._body_names
      )
    else:
      body_columns = self._body_indexes.detach().cpu().numpy().astype(np.int64)

    self.joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(body_pos_w, dtype=torch.float32, device=device)
    self._body_quat_w = torch.tensor(body_quat_w, dtype=torch.float32, device=device)
    self._body_lin_vel_w = torch.tensor(
      body_lin_vel_w, dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      body_ang_vel_w, dtype=torch.float32, device=device
    )
    self._body_columns = torch.tensor(body_columns, dtype=torch.long, device=device)
    self.body_pos_w = self._body_pos_w[:, self._body_columns]
    self.body_quat_w = self._body_quat_w[:, self._body_columns]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_columns]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_columns]
    self.time_step_total = self.joint_pos.shape[0]


@dataclass  # this means that each of the field: type will be populated in an implicit __init__ method.
class MotionGoalCfg:  # this indicates the conditioned goal(s) in each motion
  """Configuration for a single target within a motion."""

  goal_type: Literal["position", "orientation", "velocity"] = "position"
  goal_weight: float = 1.0  # this is the weight of the position, orientation, or velocity reward for this target depending on the obs type
  source_link: str | None = None
  source_type: Literal["body", "site"] = "body"
  target_link: str | None = None
  target_type: Literal["body", "site"] = "body"
  target_pos_mean: dict[str, float] = field(
    default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
  )
  target_pos_std: dict[str, float] = field(
    default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
  )
  target_pos_frame: Literal["current_anchor", "reference_anchor_frame0"] = (
    "current_anchor"
  )
  target_vel_mean: dict[str, float] = field(
    default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
  )
  target_vel_std: dict[str, float] = field(
    default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
  )
  target_orientation_mean: dict[str, float] = field(
    default_factory=lambda: {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
  )
  target_orientation_std: dict[str, float] = field(
    default_factory=lambda: {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
  )
  orientation_axis: Literal["x", "y", "z"] = "y"

  target_phase_start: float = 0.0  # phase start and end
  target_phase_end: float = 1.0


@dataclass
class MotionCfg:
  """Per-motion target configuration — a list of sub-targets, each active during its own phase window."""

  name: str = ""
  sub_targets: list[MotionGoalCfg] = field(default_factory=list)
  sampling_weight: float = 1.0  # how often we sample this motion
  probe_points: list[tuple[str, float]] = field(
    default_factory=list
  )  # (site_or_body_name, phase)


@dataclass(frozen=True)
class MotionResamplePlan:
  """One atomic motion, strike-target, deadline, and ball-launch reset plan."""

  motion_ids: torch.Tensor
  target_indices: torch.Tensor
  target_positions_w: torch.Tensor
  contact_times: torch.Tensor
  launch_positions_w: torch.Tensor
  launch_linear_velocities_w: torch.Tensor
  launch_angular_velocities_w: torch.Tensor
  match_distances: torch.Tensor
  valid: torch.Tensor
  attempts: torch.Tensor


MotionPlanProvider = Callable[[torch.Tensor], MotionResamplePlan]


def signed_parallel_direction_from_template(
  template_axis: torch.Tensor,
  direction: torch.Tensor,
) -> torch.Tensor:
  """Choose the direction sign that agrees with a motion-template axis."""
  alignment = torch.sum(template_axis * direction, dim=-1, keepdim=True)
  sign = torch.where(
    alignment >= 0.0,
    torch.ones_like(alignment),
    -torch.ones_like(alignment),
  )
  return sign * direction


def incoming_ball_racket_orientation_active(
  strike_time_error_s: torch.Tensor,
  ball_speed: torch.Tensor,
  target_valid: torch.Tensor,
  *,
  is_paused: torch.Tensor,
  window_before_s: float,
  window_after_s: float,
  minimum_ball_speed: float,
) -> torch.Tensor:
  """Keep the signed racket-normal target active around the strike deadline."""
  if window_before_s < 0.0 or window_after_s < 0.0:
    raise ValueError("Racket-orientation windows must be non-negative.")
  return (
    ~is_paused
    & (strike_time_error_s >= -window_before_s)
    & (strike_time_error_s <= window_after_s)
    & ((ball_speed >= minimum_ball_speed) | target_valid)
  )


def override_single_motion_strike_position(
  motion_configs: list[MotionCfg],
  position_frame0: tuple[float, float, float],
) -> list[MotionCfg]:
  """Override a single motion's position target in its frame-0 anchor frame."""
  if len(motion_configs) != 1:
    raise ValueError(
      "Incoming-ball strike-position override requires exactly one motion, "
      f"got {len(motion_configs)}."
    )
  overridden = copy.deepcopy(motion_configs)
  target = dict(zip(("x", "y", "z"), position_frame0, strict=True))
  position_targets = [
    sub_target
    for sub_target in overridden[0].sub_targets
    if sub_target.goal_type == "position"
  ]
  if not position_targets:
    raise ValueError("Incoming-ball motion has no position target to override.")
  for sub_target in position_targets:
    sub_target.target_pos_mean = target.copy()
    sub_target.target_pos_frame = "reference_anchor_frame0"
  return overridden


class MultiTargetMotionCommand(CommandTerm):
  """Motion command supporting multiple motions with per-motion target points.

  Each motion has a source link that should reach a target position during a
  specific phase window of the motion.  Targets can be static (sampled from
  a Gaussian) or dynamic (tracking another body link).
  """

  cfg: MultiTargetMotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MultiTargetMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    # Build per-motion loaders; configs come directly from cfg.
    self.motion_loaders: list[MotionLoader] = []
    for motion_file in self.cfg.motion_files:
      self.motion_loaders.append(
        MotionLoader(
          motion_file,
          self.body_indexes,
          self.cfg.body_names,
          self._env.sim.mj_model,
          device=self.device,
        )
      )
    self.motion_configs: list[MotionCfg] = list(self.cfg.motion_target_cfgs)
    if self.cfg.incoming_ball_strike_target_position_frame0 is not None:
      self.motion_configs = override_single_motion_strike_position(
        self.motion_configs,
        self.cfg.incoming_ball_strike_target_position_frame0,
      )
      self.cfg.motion_target_cfgs = copy.deepcopy(self.motion_configs)
      print(
        "[INFO] Incoming-ball strike target override (reference frame-0): "
        f"{self.cfg.incoming_ball_strike_target_position_frame0}"
      )

    # Source/target indices per motion per sub-target (body or site depending on type).
    self.source_body_indices: list[list[int]] = []
    self.source_is_site: list[list[bool]] = []
    self.target_body_indices: list[list[int | None]] = []
    self.target_is_site: list[list[bool]] = []
    for mc in self.motion_configs:
      m_src, m_src_site, m_tgt, m_tgt_site = [], [], [], []
      for st in mc.sub_targets:
        if st.source_type == "site":
          m_src.append(self.robot.find_sites(st.source_link, preserve_order=True)[0][0])
          m_src_site.append(True)
        else:
          m_src.append(self.robot.body_names.index(st.source_link))
          m_src_site.append(False)
        if st.target_link is not None:
          if st.target_type == "site":
            m_tgt.append(
              self.robot.find_sites(st.target_link, preserve_order=True)[0][0]
            )
            m_tgt_site.append(True)
          else:
            m_tgt.append(self.robot.body_names.index(st.target_link))
            m_tgt_site.append(False)
        else:
          m_tgt.append(None)
          m_tgt_site.append(False)
      self.source_body_indices.append(m_src)
      self.source_is_site.append(m_src_site)
      self.target_body_indices.append(m_tgt)
      self.target_is_site.append(m_tgt_site)

    # Pre-stack motion data for vectorized access.
    num_motions = len(self.motion_loaders)
    max_t = max(m.time_step_total for m in self.motion_loaders)

    def _pad_stack(tensors: list[torch.Tensor], max_len: int) -> torch.Tensor:
      padded = []
      for t in tensors:
        if t.shape[0] < max_len:
          pad = t[-1:].expand((max_len - t.shape[0],) + t.shape[1:])
          t = torch.cat([t, pad], dim=0)
        padded.append(t)
      return torch.stack(padded)

    self._stacked_joint_pos = _pad_stack(
      [m.joint_pos for m in self.motion_loaders], max_t
    )
    self._stacked_joint_vel = _pad_stack(
      [m.joint_vel for m in self.motion_loaders], max_t
    )
    self._stacked_body_pos_w = _pad_stack(
      [m.body_pos_w for m in self.motion_loaders], max_t
    )
    self._stacked_body_quat_w = _pad_stack(
      [m.body_quat_w for m in self.motion_loaders], max_t
    )
    self._stacked_body_lin_vel_w = _pad_stack(
      [m.body_lin_vel_w for m in self.motion_loaders], max_t
    )
    self._stacked_body_ang_vel_w = _pad_stack(
      [m.body_ang_vel_w for m in self.motion_loaders], max_t
    )
    self._time_step_totals = torch.tensor(
      [m.time_step_total for m in self.motion_loaders], device=self.device
    )

    # Pre-compute per-motion, per-subtarget indices and parameters as tensors.
    # Shape convention: (num_motions, max_subtargets, ...), padded with zeros/False.
    max_subtargets = max(len(mc.sub_targets) for mc in self.motion_configs)
    self.max_subtargets = max_subtargets

    # Per-subtarget observation type — taken from the first motion config and
    # padded with "position" for any slots that don't exist in that motion.
    first_sub_targets = self.motion_configs[0].sub_targets
    self._subtarget_goal_types: list[str] = [
      first_sub_targets[s].goal_type if s < len(first_sub_targets) else "position"
      for s in range(max_subtargets)
    ]

    def _pad_to(vals: list, length: int, pad):
      return vals + [pad] * (length - len(vals))

    self._source_body_indices_t = torch.tensor(
      [_pad_to(row, max_subtargets, 0) for row in self.source_body_indices],
      device=self.device,
      dtype=torch.long,
    )  # (num_motions, max_subtargets)
    self._target_body_indices_t = torch.tensor(
      [
        _pad_to([idx if idx is not None else 0 for idx in row], max_subtargets, 0)
        for row in self.target_body_indices
      ],
      device=self.device,
      dtype=torch.long,
    )  # (num_motions, max_subtargets)
    self._has_target_link = torch.tensor(
      [
        _pad_to([idx is not None for idx in row], max_subtargets, False)
        for row in self.target_body_indices
      ],
      device=self.device,
      dtype=torch.bool,
    )  # (num_motions, max_subtargets)
    self._source_is_site_t = torch.tensor(
      [_pad_to(row, max_subtargets, False) for row in self.source_is_site],
      device=self.device,
      dtype=torch.bool,
    )  # (num_motions, max_subtargets)
    self._target_is_site_t = torch.tensor(
      [_pad_to(row, max_subtargets, False) for row in self.target_is_site],
      device=self.device,
      dtype=torch.bool,
    )  # (num_motions, max_subtargets)

    # Pre-compute target sampling parameters as tensors.
    # (num_motions, max_subtargets, 3)
    self._target_pos_means_t = torch.stack(
      [
        torch.stack(
          [
            torch.tensor(
              [st.target_pos_mean.get(k, 0.0) for k in ["x", "y", "z"]],
              device=self.device,
            )
            for st in _pad_to(mc.sub_targets, max_subtargets, mc.sub_targets[0])
          ]
        )
        for mc in self.motion_configs
      ]
    )
    self._target_pos_stds_t = torch.stack(
      [
        torch.stack(
          [
            torch.tensor(
              [st.target_pos_std.get(k, 0.0) for k in ["x", "y", "z"]],
              device=self.device,
            )
            for st in _pad_to(mc.sub_targets, max_subtargets, mc.sub_targets[0])
          ]
        )
        for mc in self.motion_configs
      ]
    )
    self._target_pos_frame0_t = torch.tensor(
      [
        _pad_to(
          [st.target_pos_frame == "reference_anchor_frame0" for st in mc.sub_targets],
          max_subtargets,
          False,
        )
        for mc in self.motion_configs
      ],
      device=self.device,
      dtype=torch.bool,
    )
    self._target_ori_means_t = torch.stack(
      [
        torch.stack(
          [
            torch.tensor(
              [
                st.target_orientation_mean.get(k, 0.0) for k in ["roll", "pitch", "yaw"]
              ],
              device=self.device,
            )
            for st in _pad_to(mc.sub_targets, max_subtargets, mc.sub_targets[0])
          ]
        )
        for mc in self.motion_configs
      ]
    )  # (num_motions, max_subtargets, 3)
    self._target_ori_stds_t = torch.stack(
      [
        torch.stack(
          [
            torch.tensor(
              [st.target_orientation_std.get(k, 0.0) for k in ["roll", "pitch", "yaw"]],
              device=self.device,
            )
            for st in _pad_to(mc.sub_targets, max_subtargets, mc.sub_targets[0])
          ]
        )
        for mc in self.motion_configs
      ]
    )  # (num_motions, max_subtargets, 3)
    self._target_phase_starts_t = torch.tensor(
      [
        _pad_to(
          [st.target_phase_start for st in mc.sub_targets],
          max_subtargets,
          0.0,
        )
        for mc in self.motion_configs
      ],
      device=self.device,
    )  # (num_motions, max_subtargets)
    self._target_phase_ends_t = torch.tensor(
      [
        _pad_to([st.target_phase_end for st in mc.sub_targets], max_subtargets, 0.0)
        for mc in self.motion_configs
      ],
      device=self.device,
    )  # (num_motions, max_subtargets)

    def _goal_weight(st: MotionGoalCfg, kind: str) -> float:
      return st.goal_weight if st.goal_type == kind else 0.0

    self._target_pos_reward_weights_t = torch.tensor(
      [
        _pad_to(
          [_goal_weight(st, "position") for st in mc.sub_targets],
          max_subtargets,
          0.0,
        )
        for mc in self.motion_configs
      ],
      device=self.device,
    )  # (num_motions, max_subtargets)
    self._target_ori_reward_weights_t = torch.tensor(
      [
        _pad_to(
          [_goal_weight(st, "orientation") for st in mc.sub_targets],
          max_subtargets,
          0.0,
        )
        for mc in self.motion_configs
      ],
      device=self.device,
    )  # (num_motions, max_subtargets)
    _axis_vec_map = {
      "x": [1.0, 0.0, 0.0],
      "y": [0.0, 1.0, 0.0],
      "z": [0.0, 0.0, 1.0],
    }
    self._target_ori_axes_t = torch.tensor(
      [
        _pad_to(
          [_axis_vec_map[st.orientation_axis] for st in mc.sub_targets],
          max_subtargets,
          [0.0, 0.0, 0.0],
        )
        for mc in self.motion_configs
      ],
      device=self.device,
    )  # (num_motions, max_subtargets, 3)
    self._target_vel_means_t = torch.stack(
      [
        torch.stack(
          [
            torch.tensor(
              [st.target_vel_mean.get(k, 0.0) for k in ["x", "y", "z"]],
              device=self.device,
            )
            for st in _pad_to(mc.sub_targets, max_subtargets, mc.sub_targets[0])
          ]
        )
        for mc in self.motion_configs
      ]
    )  # (num_motions, max_subtargets, 3)
    self._target_vel_stds_t = torch.stack(
      [
        torch.stack(
          [
            torch.tensor(
              [st.target_vel_std.get(k, 0.0) for k in ["x", "y", "z"]],
              device=self.device,
            )
            for st in _pad_to(mc.sub_targets, max_subtargets, mc.sub_targets[0])
          ]
        )
        for mc in self.motion_configs
      ]
    )  # (num_motions, max_subtargets, 3)
    self._target_vel_reward_weights_t = torch.tensor(
      [
        _pad_to(
          [_goal_weight(st, "velocity") for st in mc.sub_targets],
          max_subtargets,
          0.0,
        )
        for mc in self.motion_configs
      ],
      device=self.device,
    )  # (num_motions, max_subtargets)
    self._reference_source_strike_lin_vel_t = (
      self._build_reference_source_strike_lin_velocities()
    )

    # Motion sampling weights — normalised to sum to 1.
    num_motions = len(self.motion_loaders)
    raw_weights = self.cfg.motion_sampling_weights
    w = torch.tensor(raw_weights, dtype=torch.float32, device=self.device)
    self._motion_weights_t = w / w.sum()

    # Per-env state.
    self.which_motion = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    if cfg.target_pos_std_scale < 0.0:
      raise ValueError("Target position std scale must be non-negative.")
    self.target_pos_std_scale = cfg.target_pos_std_scale

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._reference_align_root_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self._reference_align_yaw_w = torch.zeros(self.num_envs, 4, device=self.device)
    self._reference_align_yaw_w[:, 0] = 1.0
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0
    self._ghost_align_root_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self._ghost_align_yaw_w = torch.zeros(self.num_envs, 4, device=self.device)
    self._ghost_align_yaw_w[:, 0] = 1.0

    # Target tracking (world frame) — (num_envs, max_subtargets, 3/4).
    # Per-episode sampled offset for moving targets, stored in anchor-aligned world frame
    # so _update_command can apply it every step without re-randomizing.
    self._moving_target_offset_w = torch.zeros(
      self.num_envs, max_subtargets, 3, device=self.device
    )
    self.target_position_w = torch.zeros(
      self.num_envs, max_subtargets, 3, device=self.device
    )
    self.target_orientation_w = torch.zeros(
      self.num_envs, max_subtargets, 4, device=self.device
    )
    self.target_orientation_w[:, :, 0] = 1.0
    self.target_velocity_w = torch.zeros(
      self.num_envs, max_subtargets, 3, device=self.device
    )
    self.incoming_ball_racket_orientation_target_w = torch.zeros(
      self.num_envs, max_subtargets, 3, device=self.device
    )
    self.incoming_ball_racket_orientation_target_valid = torch.zeros(
      self.num_envs, max_subtargets, dtype=torch.bool, device=self.device
    )

    if cfg.analytic_strike_planner_enabled:
      required_goal_types = {"position", "velocity", "orientation"}
      missing_goal_types = required_goal_types - set(self._subtarget_goal_types)
      if missing_goal_types:
        raise ValueError(
          "Analytic strike planner requires position, velocity, and orientation "
          f"sub-targets; missing {sorted(missing_goal_types)}."
        )
      if not cfg.landing_target_enabled:
        raise ValueError("Analytic strike planner requires landing targets.")
    self.target_ball_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_ball_out_speed = torch.full(
      (self.num_envs,), cfg.analytic_ball_preferred_speed, device=self.device
    )
    self.target_racket_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_racket_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_racket_speed = torch.zeros(self.num_envs, device=self.device)
    self.analytic_strike_feasible = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.target_ball_net_height = torch.zeros(self.num_envs, device=self.device)

    # Optional tennis landing target and physical/predicted landing state.
    if cfg.landing_target_frame not in ("environment", "startup"):
      raise ValueError(
        "Landing target frame must be 'environment' or 'startup', got "
        f"{cfg.landing_target_frame!r}."
      )
    self.startup_anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.startup_anchor_yaw_w = torch.zeros(self.num_envs, 4, device=self.device)
    self.startup_anchor_yaw_w[:, 0] = 1.0
    self.target_landing_position_startup = torch.zeros(
      self.num_envs, 3, device=self.device
    )
    self.target_landing_position_w = torch.zeros(self.num_envs, 3, device=self.device)
    self._landing_target_mean_t = torch.tensor(
      cfg.landing_target_mean, device=self.device
    )
    self._landing_target_std_t = torch.tensor(
      cfg.landing_target_std, device=self.device
    )
    self.landing_target_curriculum_step_offset = 0
    if cfg.landing_target_std_final is not None:
      linear_landing_target_std(
        0, cfg.landing_target_std_ramp_steps,
        cfg.landing_target_std, cfg.landing_target_std_final,
      )
    if cfg.landing_target_net_margin_m is not None and (
      not math.isfinite(cfg.landing_target_net_margin_m)
      or cfg.landing_target_net_margin_m <= 0.0
    ):
      raise ValueError("Landing target net margin must be finite and positive.")
    self.ball_has_been_struck = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.ball_landing_recorded = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.ball_landing_position_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.ball_previous_vertical_velocity = torch.zeros(
      self.num_envs, device=self.device
    )
    self.ball_previous_linear_velocity_w = torch.zeros(
      self.num_envs, 3, device=self.device
    )
    self.ball_previous_racket_offset_w = torch.zeros(
      self.num_envs, 3, device=self.device
    )
    self.ball_strike_state_initialized = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.ball_post_strike_steps = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.ball_post_strike_elapsed_s = torch.zeros(self.num_envs, device=self.device)
    self.ball_hit_reward = torch.zeros(self.num_envs, device=self.device)
    self.ball_direction_reward = torch.zeros(self.num_envs, device=self.device)
    self.ball_net_clearance_reward = torch.zeros(self.num_envs, device=self.device)
    self.ball_out_speed_reward = torch.zeros(self.num_envs, device=self.device)
    self.racket_ball_min_distance = torch.full(
      (self.num_envs,), torch.inf, device=self.device
    )
    self.racket_ball_score_max = torch.zeros(self.num_envs, device=self.device)
    self.racket_ball_min_abs_time_error = torch.full(
      (self.num_envs,), torch.inf, device=self.device
    )
    self.racket_ball_distance_at_deadline = torch.zeros(
      self.num_envs, device=self.device
    )

    # Source-state cache — refreshed once per step in _update_command so that
    # get_source_pos_w / get_source_quat_w / get_source_lin_vel_w are free reads.
    self._source_pos_w_cache = torch.zeros(
      self.num_envs, max_subtargets, 3, device=self.device
    )
    self._source_quat_w_cache = torch.zeros(
      self.num_envs, max_subtargets, 4, device=self.device
    )
    self._source_quat_w_cache[:, :, 0] = 1.0
    self._source_lin_vel_w_cache = torch.zeros(
      self.num_envs, max_subtargets, 3, device=self.device
    )

    # Adaptive sampling per motion.
    self.bin_counts = [
      int(self.motion_loaders[i].time_step_total // (1 / self._env.step_dt)) + 1
      for i in range(num_motions)
    ]
    self.bin_failed_counts = [
      torch.zeros(self.bin_counts[i], dtype=torch.float, device=self.device)
      for i in range(num_motions)
    ]
    self._current_bin_failed = [
      torch.zeros(self.bin_counts[i], dtype=torch.float, device=self.device)
      for i in range(num_motions)
    ]
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    # Between-motion pause.
    self.between_motion_pause_range = self.cfg.between_motion_pause_range
    self.between_motion_pause_time = torch.zeros(self.num_envs, device=self.device)
    self._sampled_pause_lengths = torch.zeros(self.num_envs, device=self.device)
    self.is_paused = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.auto_chain_motion = torch.full(
      (self.num_envs,),
      self.cfg.auto_chain_motion,
      dtype=torch.bool,
      device=self.device,
    )
    self._motion_plan_batch_interval_steps = int(
      self.cfg.incoming_ball_torch_match_chain_batch_interval_steps
    )
    if self._motion_plan_batch_interval_steps <= 0:
      raise ValueError(
        "incoming_ball_torch_match_chain_batch_interval_steps must be positive."
      )
    self._motion_plan_batch_step = 0
    self._motion_plan_provider: MotionPlanProvider | None = None
    self.planned_contact_time = torch.full(
      (self.num_envs,), torch.nan, device=self.device
    )
    self.planned_target_index = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.planned_launch_position_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.planned_launch_linear_velocity_w = torch.zeros(
      self.num_envs, 3, device=self.device
    )
    self.planned_launch_angular_velocity_w = torch.zeros(
      self.num_envs, 3, device=self.device
    )
    self.motion_chain_count = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    num_bodies = len(self.cfg.body_names)
    num_joints = self.robot.data.joint_pos.shape[-1]
    self._paused_body_pos_w = torch.zeros(
      self.num_envs, num_bodies, 3, device=self.device
    )
    self._paused_body_quat_w = torch.zeros(
      self.num_envs, num_bodies, 4, device=self.device
    )
    self._paused_body_quat_w[..., 0] = 1.0
    self._paused_joint_pos = torch.zeros(self.num_envs, num_joints, device=self.device)
    self._paused_joint_vel = torch.zeros(self.num_envs, num_joints, device=self.device)

    # Metrics.
    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_target_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_target_pos_at_target"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_ball_landing"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["ball_landing_prediction_valid"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["ball_net_crossing_height"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["ball_out_speed"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["ball_target_projected_speed"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["ball_direction_error_degrees"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["post_strike_heading_error_degrees"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["post_strike_heading_active"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["target_ball_out_speed"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["target_racket_speed"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["target_ball_net_height"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["analytic_strike_feasible"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["racket_ball_min_distance"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["racket_ball_score_max"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["racket_ball_min_abs_time_error"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["racket_ball_distance_at_deadline"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    self.metrics["trajectory_match_distance"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["trajectory_match_valid"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.trajectory_match_valid = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.metrics["failure_trajectory"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["trajectory_match_attempt"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["trajectory_prediction_error"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["motion_chain_count"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["reference_sweet_spot_speed"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["racket_sweet_spot_projected_speed"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["racket_sweet_spot_speed_reward_active"] = torch.zeros(
      self.num_envs, device=self.device
    )

    # Ghost model for visualization (created lazily).
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  def _build_reference_source_strike_lin_velocities(
    self, relative_to_pelvis: bool = False,
  ) -> torch.Tensor:
    """Precompute source-point velocities at each motion target's phase center."""
    result = torch.zeros(
      len(self.motion_configs),
      self.max_subtargets,
      3,
      device=self.device,
    )
    model = self._env.sim.mj_model
    for motion_id, (motion_cfg, loader) in enumerate(
      zip(self.motion_configs, self.motion_loaders, strict=True)
    ):
      last_frame = loader.time_step_total - 1
      for subtarget_id, subtarget in enumerate(motion_cfg.sub_targets):
        if subtarget.source_link is None:
          raise ValueError(
            f"Motion {motion_cfg.name!r} sub-target {subtarget_id} has no source link."
          )
        phase_center = 0.5 * (subtarget.target_phase_start + subtarget.target_phase_end)
        frame = min(max(round(phase_center * last_frame), 0), last_frame)

        if subtarget.source_type == "site":
          site_ids = [
            site_id
            for site_id in range(model.nsite)
            if (
              site_name := mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                site_id,
              )
            )
            and site_name.rsplit("/", 1)[-1] == subtarget.source_link
          ]
          if len(site_ids) != 1:
            raise ValueError(
              f"Reference source site {subtarget.source_link!r} matched "
              f"{len(site_ids)} compiled sites."
            )
          site_id = site_ids[0]
          parent_model_id = int(model.site_bodyid[site_id])
          parent_full_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            parent_model_id,
          )
          parent_name = parent_full_name.rsplit("/", 1)[-1]
          if parent_name not in self.cfg.body_names:
            raise ValueError(
              f"Parent body {parent_name!r} for site {subtarget.source_link!r} "
              "is absent from the reference body list."
            )
          body_id = self.cfg.body_names.index(parent_name)
          point_offset_parent = torch.as_tensor(
            model.site_pos[site_id],
            dtype=torch.float32,
            device=self.device,
          )
        else:
          if subtarget.source_link not in self.cfg.body_names:
            raise ValueError(
              f"Reference source body not found: {subtarget.source_link}"
            )
          body_id = self.cfg.body_names.index(subtarget.source_link)
          point_offset_parent = torch.zeros(3, device=self.device)

        result[motion_id, subtarget_id] = rigid_point_linear_velocity(
          self._stacked_body_lin_vel_w[motion_id, frame, body_id],
          self._stacked_body_ang_vel_w[motion_id, frame, body_id],
          self._stacked_body_quat_w[motion_id, frame, body_id],
          point_offset_parent,
        )
        if relative_to_pelvis:
          from .relative_velocity import point_velocity_relative_to_frame

          anchor = self.motion_anchor_body_index
          point_position_w = (
            self._stacked_body_pos_w[motion_id, frame, body_id]
            + quat_apply(
              self._stacked_body_quat_w[motion_id, frame, body_id],
              point_offset_parent,
            )
          )
          result[motion_id, subtarget_id] = point_velocity_relative_to_frame(
            point_position_w,
            result[motion_id, subtarget_id],
            self._stacked_body_pos_w[motion_id, frame, anchor],
            self._stacked_body_quat_w[motion_id, frame, anchor],
            self._stacked_body_lin_vel_w[motion_id, frame, anchor],
            self._stacked_body_ang_vel_w[motion_id, frame, anchor],
          )
    return result

  # ------------------------------------------------------------------
  # Time-step helpers
  # ------------------------------------------------------------------

  def _clamped_time_steps(self) -> torch.Tensor:
    totals = self._time_step_totals[self.which_motion]
    return torch.where(
      self.time_steps >= totals,
      torch.zeros_like(self.time_steps),
      self.time_steps,
    )

  @property
  def normalized_phase(self) -> torch.Tensor:
    """Reward phase for the current reference sample.

    The integer-frame command intentionally keeps its historical ``frame / N``
    convention. Continuous phase commands override this property with their
    native phase state.
    """
    totals = self._time_step_totals[self.which_motion].float()
    return self.time_steps.float() / totals

  def _uses_reference_frame0(self, motion_ids: torch.Tensor) -> torch.Tensor:
    return torch.any(self._target_pos_frame0_t[motion_ids], dim=1)

  def _align_reference_frame0_to_robot(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    motion_ids = self.which_motion[env_ids]
    active = self._uses_reference_frame0(motion_ids)
    if not torch.any(active):
      return
    active_env_ids = env_ids[active]
    active_motion_ids = motion_ids[active]
    ref_quat_0 = self._stacked_body_quat_w[
      active_motion_ids, 0, self.motion_anchor_body_index
    ]
    aligned_pos, align_yaw = reference_frame0_alignment(
      ref_quat_0,
      self.robot.data.body_link_pos_w[active_env_ids, self.robot_anchor_body_index],
      self.robot.data.body_link_quat_w[active_env_ids, self.robot_anchor_body_index],
    )
    self._reference_align_root_pos_w[active_env_ids] = aligned_pos
    self._reference_align_yaw_w[active_env_ids] = align_yaw

  def _align_reference_body_pos_w(
    self, raw_pos_w: torch.Tensor, env_ids: torch.Tensor
  ) -> torch.Tensor:
    motion_ids = self.which_motion[env_ids]
    active = self._uses_reference_frame0(motion_ids)
    if not torch.any(active):
      return raw_pos_w
    result = raw_pos_w.clone()
    active_env_ids = env_ids[active]
    active_motion_ids = motion_ids[active]
    ref_pos_0 = (
      self._stacked_body_pos_w[active_motion_ids, 0, self.motion_anchor_body_index]
      + self._env.scene.env_origins[active_env_ids]
    )
    offsets = raw_pos_w[active] - ref_pos_0[:, None, :]
    align_yaw = self._reference_align_yaw_w[active_env_ids]
    expanded_yaw = align_yaw[:, None, :].expand(-1, offsets.shape[1], -1)
    result[active] = self._reference_align_root_pos_w[
      active_env_ids, None, :
    ] + quat_apply(
      expanded_yaw.reshape(-1, 4),
      offsets.reshape(-1, 3),
    ).reshape_as(offsets)
    return result

  def _align_reference_quat_w(
    self, raw_quat_w: torch.Tensor, env_ids: torch.Tensor
  ) -> torch.Tensor:
    motion_ids = self.which_motion[env_ids]
    active = self._uses_reference_frame0(motion_ids)
    if not torch.any(active):
      return raw_quat_w
    result = raw_quat_w.clone()
    align_yaw = self._reference_align_yaw_w[env_ids[active]]
    result[active] = quat_mul(
      align_yaw[:, None, :].expand(-1, raw_quat_w.shape[1], -1),
      raw_quat_w[active],
    )
    return result

  def _align_reference_vector_w(
    self, raw_vector_w: torch.Tensor, env_ids: torch.Tensor
  ) -> torch.Tensor:
    motion_ids = self.which_motion[env_ids]
    active = self._uses_reference_frame0(motion_ids)
    if not torch.any(active):
      return raw_vector_w
    result = raw_vector_w.clone()
    vectors = raw_vector_w[active]
    align_yaw = self._reference_align_yaw_w[env_ids[active]]
    expanded_yaw = align_yaw[:, None, :].expand(-1, vectors.shape[1], -1)
    result[active] = quat_apply(
      expanded_yaw.reshape(-1, 4),
      vectors.reshape(-1, 3),
    ).reshape_as(vectors)
    return result

  # ------------------------------------------------------------------
  # Properties: command observation
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """Joint pos/vel + per-subtarget observation (position, orientation, or velocity).

    Each subtarget emits only the quantity selected by its ``goal_type``:
      "position"    → 3 values:
                        static target  → anchor-frame position (changes as robot moves)
                        dynamic target → target-body-frame position (the originally sampled offset)
      "orientation" → 4 values (anchor-frame quaternion)
      "velocity"    → 3 values (world-frame linear velocity)

    For dynamic (moving) targets we emit the originally sampled offset in the
    target body's local frame — the same value that's rotated into world frame
    each step to produce the reward target.
    """
    robot_quat_inv = quat_inv(self.robot_anchor_quat_w)  # (E, 4)
    quat_inv_exp = robot_quat_inv[:, None, :].expand(-1, self.max_subtargets, -1)
    flat_quat_inv = quat_inv_exp.reshape(-1, 4)

    # Pelvis-frame position — used for static targets.
    target_pos_b = quat_apply(
      flat_quat_inv,
      (self.target_position_w - self.robot_anchor_pos_w[:, None, :]).reshape(-1, 3),
    ).reshape(self.num_envs, self.max_subtargets, 3)
    target_ori_b = quat_mul(
      flat_quat_inv,
      self.target_orientation_w.reshape(-1, 4),
    ).reshape(self.num_envs, self.max_subtargets, 4)
    target_vel_b = self.target_velocity_w.reshape(
      self.num_envs, self.max_subtargets, 3
    )  # velocity is already in world frame, no need to transform

    # Per-env, per-subtarget dynamic flag: True when a target_link is set.  (E, S)
    is_dynamic = self._has_target_link[self.which_motion]

    parts: list[torch.Tensor] = [
      self.joint_pos,
      self.joint_vel,
    ]  # always use joint pos and vel

    for s, goal_type in enumerate(
      self._subtarget_goal_types
    ):  # for each of the subtarget, add their goal type
      if goal_type == "position":
        # Static → pelvis-relative (tells the policy where the goal is w.r.t. the robot).
        # Dynamic → target-body-local (the originally sampled offset, constant per episode).
        dynamic_mask = is_dynamic[:, s].unsqueeze(-1)  # (E, 1)
        pos_obs = torch.where(
          dynamic_mask,
          self._moving_target_offset_w[:, s],  # target-body local frame
          target_pos_b[:, s],  # pelvis frame — for static targets
        )
        # print out the moving target offset and the target pos in pelvis frame for debugging
        # print("Moving target offset (body frame):", self._moving_target_offset_w[:, s])
        parts.append(pos_obs)
      elif goal_type == "orientation":
        parts.append(target_ori_b[:, s])
      else:  # velocity
        parts.append(target_vel_b[:, s])

    return torch.cat(parts, dim=1)

  # ------------------------------------------------------------------
  # Properties: motion data
  # ------------------------------------------------------------------
  @property
  def joint_pos(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    val = self._stacked_joint_pos[self.which_motion, t]
    if torch.any(self.is_paused):
      val[self.is_paused] = self._paused_joint_pos[self.is_paused]
    return val

  @property
  def joint_vel(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    val = self._stacked_joint_vel[self.which_motion, t]
    if torch.any(self.is_paused):
      val[self.is_paused] = self._paused_joint_vel[self.is_paused]
    return val

  @property
  def body_pos_w(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    env_ids = torch.arange(self.num_envs, device=self.device)
    raw_pos = (
      self._stacked_body_pos_w[self.which_motion, t]
      + self._env.scene.env_origins[:, None, :]
    )
    body_pos = self._align_reference_body_pos_w(raw_pos, env_ids)
    if torch.any(self.is_paused):
      body_pos[self.is_paused] = self._paused_body_pos_w[self.is_paused]
    return body_pos

  @property
  def body_quat_w(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    env_ids = torch.arange(self.num_envs, device=self.device)
    raw_quat = self._stacked_body_quat_w[self.which_motion, t]
    val = self._align_reference_quat_w(raw_quat, env_ids)
    if torch.any(self.is_paused):
      val[self.is_paused] = self._paused_body_quat_w[self.is_paused]
    return val

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    env_ids = torch.arange(self.num_envs, device=self.device)
    raw_vel = self._stacked_body_lin_vel_w[self.which_motion, t]
    vel = self._align_reference_vector_w(raw_vel, env_ids)
    vel[self.is_paused] = 0.0
    return vel

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    env_ids = torch.arange(self.num_envs, device=self.device)
    raw_vel = self._stacked_body_ang_vel_w[self.which_motion, t]
    vel = self._align_reference_vector_w(raw_vel, env_ids)
    vel[self.is_paused] = 0.0
    return vel

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, self.motion_anchor_body_index]

  # ------------------------------------------------------------------
  # Properties: robot state
  # ------------------------------------------------------------------

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  # ------------------------------------------------------------------
  # Link-state helpers (body or site)
  # ------------------------------------------------------------------

  def _fetch_link_state(
    self,
    env_ids: torch.Tensor,
    indices_t: torch.Tensor,
    is_site_t: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pos, quat)`` in world frame for each env in *env_ids*.

    *indices_t* and *is_site_t* are per-env body or site indices and flags
    (already indexed for this batch).
    """
    pos = torch.zeros(len(env_ids), 3, device=self.device)
    quat = torch.zeros(len(env_ids), 4, device=self.device)
    quat[:, 0] = 1.0
    body_mask = ~is_site_t
    if torch.any(body_mask):
      bi = env_ids[body_mask]
      pos[body_mask] = self.robot.data.body_link_pos_w[bi, indices_t[body_mask]]
      quat[body_mask] = self.robot.data.body_link_quat_w[bi, indices_t[body_mask]]
    if torch.any(is_site_t):
      si = env_ids[is_site_t]
      pos[is_site_t] = self.robot.data.site_pos_w[si, indices_t[is_site_t]]
      quat[is_site_t] = self.robot.data.site_quat_w[si, indices_t[is_site_t]]
    return pos, quat

  def _fetch_link_state_batched(
    self,
    env_ids: torch.Tensor,
    indices_t: torch.Tensor,
    is_site_t: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pos, quat)`` of shape ``(E, S, 3/4)`` for *env_ids* × sub-targets.

    *indices_t* and *is_site_t* are ``(E, S)`` body/site indices and flags.
    Fetches all sub-targets in one vectorized gather, blending body and site
    data via ``torch.where``.
    """
    E, S = indices_t.shape
    env_ids_exp = env_ids[:, None].expand(E, S)  # (E, S)
    # Clamp to valid range for each array so out-of-domain indices don't OOB.
    # Values at clamped-invalid positions are masked out by torch.where below.
    body_idx = indices_t.clamp(max=self.robot.data.body_link_pos_w.shape[1] - 1)
    site_idx = indices_t.clamp(max=self.robot.data.site_pos_w.shape[1] - 1)
    body_pos = self.robot.data.body_link_pos_w[env_ids_exp, body_idx]  # (E, S, 3)
    body_quat = self.robot.data.body_link_quat_w[env_ids_exp, body_idx]  # (E, S, 4)
    site_pos = self.robot.data.site_pos_w[env_ids_exp, site_idx]  # (E, S, 3)
    site_quat = self.robot.data.site_quat_w[env_ids_exp, site_idx]  # (E, S, 4)
    is_site = is_site_t.unsqueeze(-1)  # (E, S, 1)
    pos = torch.where(is_site, site_pos, body_pos)
    quat = torch.where(is_site.expand(-1, -1, 4), site_quat, body_quat)
    return pos, quat

  def _fetch_target_pos_quat(
    self, env_ids: torch.Tensor, motion_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-frame ``(pos, quat)`` of shape ``(E, S, 3/4)`` for each target link."""
    return self._fetch_link_state_batched(
      env_ids,
      self._target_body_indices_t[motion_ids],
      self._target_is_site_t[motion_ids],
    )

  # ------------------------------------------------------------------
  # Target sampling
  # ------------------------------------------------------------------

  def _sample_targets(self, env_ids: torch.Tensor) -> None:
    """Sample target positions/orientations for *env_ids* across all sub-targets."""
    motion_ids = self.which_motion[env_ids]

    anchor_pos_w = self.robot.data.body_link_pos_w[
      env_ids, self.robot_anchor_body_index
    ]
    anchor_quat_w = self.robot.data.body_link_quat_w[
      env_ids, self.robot_anchor_body_index
    ]

    for s in range(self.max_subtargets):
      has_moving = self._has_target_link[motion_ids, s]  # (E,)

      # Moving targets — fetch only when needed, using the non-batched helper.
      if torch.any(has_moving):
        mv_eids = env_ids[has_moving]
        mv_mids = motion_ids[has_moving]
        mv_idx = torch.where(has_moving)[0]
        tgt_pos, tgt_quat = self._fetch_link_state(
          mv_eids,
          self._target_body_indices_t[mv_mids, s],
          self._target_is_site_t[mv_mids, s],
        )
        pos_offset = self._target_pos_means_t[
          mv_mids, s
        ]  # (E_mv, 3), target body local frame
        pos_std = self._target_pos_stds_t[mv_mids, s] * self.target_pos_std_scale
        rand_offset = (
          pos_offset + torch.randn(len(mv_eids), 3, device=self.device) * pos_std
        )

        # Persist per-episode random offset (target-body local frame) so _update_command
        # reapplies it every step without re-randomizing.
        self._moving_target_offset_w[mv_eids, s] = rand_offset

        self.target_position_w[mv_eids, s] = tgt_pos + quat_apply(tgt_quat, rand_offset)
        self.target_orientation_w[mv_eids, s] = tgt_quat
        vel_mean_mv = self._target_vel_means_t[mv_mids, s]
        vel_std_mv = self._target_vel_stds_t[mv_mids, s]
        rand_vel_mv = (
          vel_mean_mv + torch.randn(len(mv_eids), 3, device=self.device) * vel_std_mv
        )
        self.target_velocity_w[mv_eids, s] = quat_apply(
          anchor_quat_w[mv_idx], rand_vel_mv
        )

      # Static targets — sampled from Gaussian in anchor frame.
      has_static = ~has_moving
      if torch.any(has_static):
        st_idx = torch.where(has_static)[0]
        st_eids = env_ids[has_static]
        st_mids = motion_ids[has_static]
        ns = int(has_static.sum().item())

        pos_mean = self._target_pos_means_t[st_mids, s]
        pos_std = self._target_pos_stds_t[st_mids, s] * self.target_pos_std_scale
        rand_pos = pos_mean + torch.randn(ns, 3, device=self.device) * pos_std

        ori_mean = self._target_ori_means_t[st_mids, s]
        ori_std = self._target_ori_stds_t[st_mids, s]
        rand_euler = ori_mean + torch.randn(ns, 3, device=self.device) * ori_std
        rand_quat = quat_from_euler_xyz(
          rand_euler[:, 0], rand_euler[:, 1], rand_euler[:, 2]
        )

        st_anchor_pos = anchor_pos_w[st_idx]
        st_anchor_quat = anchor_quat_w[st_idx]
        sampled_target_pos_w = quat_apply(st_anchor_quat, rand_pos) + st_anchor_pos
        frame0_mask = self._target_pos_frame0_t[st_mids, s]
        if torch.any(frame0_mask):
          frame0_eids = st_eids[frame0_mask]
          frame0_mids = st_mids[frame0_mask]
          reference_anchor_pos_0_w = self._reference_align_root_pos_w[frame0_eids]
          raw_reference_quat_0_w = self._stacked_body_quat_w[
            frame0_mids, 0, self.motion_anchor_body_index
          ]
          reference_anchor_quat_0_w = quat_mul(
            self._reference_align_yaw_w[frame0_eids],
            raw_reference_quat_0_w,
          )
          sampled_target_pos_w[frame0_mask] = transform_reference_frame0_target(
            rand_pos[frame0_mask],
            reference_anchor_pos_0_w,
            reference_anchor_quat_0_w,
          )
        self.target_position_w[st_eids, s] = sampled_target_pos_w
        self.target_orientation_w[st_eids, s] = quat_mul(st_anchor_quat, rand_quat)
        vel_mean_st = self._target_vel_means_t[st_mids, s]
        vel_std_st = self._target_vel_stds_t[st_mids, s]
        rand_vel_st = vel_mean_st + torch.randn(ns, 3, device=self.device) * vel_std_st
        self.target_velocity_w[st_eids, s] = quat_apply(st_anchor_quat, rand_vel_st)

    self._sample_landing_targets(env_ids)
    self.incoming_ball_racket_orientation_target_w[env_ids] = 0.0
    self.incoming_ball_racket_orientation_target_valid[env_ids] = False

  def _sample_landing_targets(self, env_ids: torch.Tensor) -> None:
    """Sample optional tennis landing targets and cache their world position."""
    if not self.cfg.landing_target_enabled or len(env_ids) == 0:
      return
    if self.cfg.landing_target_std_final is not None:
      self.update_landing_target_std_curriculum()
    samples = self._landing_target_mean_t + (
      torch.randn(len(env_ids), 3, device=self.device) * self._landing_target_std_t
    )
    env_origins = self._env.scene.env_origins[env_ids]
    if self.cfg.landing_target_frame == "startup":
      self.target_landing_position_startup[env_ids] = samples
      self.target_landing_position_w[env_ids] = transform_startup_frame_ground_target(
        samples,
        self.startup_anchor_pos_w[env_ids],
        self.startup_anchor_yaw_w[env_ids],
        env_origins,
      )
    else:
      target_pos_w = samples + env_origins
      self.target_landing_position_w[env_ids] = target_pos_w
      self.target_landing_position_startup[env_ids] = quat_apply(
        quat_inv(self.startup_anchor_yaw_w[env_ids]),
        target_pos_w - self.startup_anchor_pos_w[env_ids],
      )
    if self.cfg.landing_target_net_margin_m is not None:
      mean = self._landing_target_mean_t.expand(len(env_ids), -1)
      if self.cfg.landing_target_frame == "startup":
        mean_w = transform_startup_frame_ground_target(
          mean, self.startup_anchor_pos_w[env_ids],
          self.startup_anchor_yaw_w[env_ids], env_origins,
        )
        rotation = matrix_from_quat(self.startup_anchor_yaw_w[env_ids])[:, :2, :2]
        variance = self._landing_target_std_t[:2].square()
        variance_x = (rotation[:, 0].square() * variance).sum(dim=-1)
        covariance_xy = (rotation[:, 0] * rotation[:, 1] * variance).sum(dim=-1)
      else:
        mean_w = mean + env_origins
        variance_x = self._landing_target_std_t[0].square().expand(len(env_ids))
        covariance_xy = torch.zeros_like(variance_x)
      self.target_landing_position_w[env_ids] = truncate_landing_targets_at_net(
        self.target_landing_position_w[env_ids], mean_w, variance_x, covariance_xy,
        env_origins[:, 0] + self.cfg.tennis_court_net_x_m + self.cfg.landing_target_net_margin_m,
      )
      bounded_startup = quat_apply(
        quat_inv(self.startup_anchor_yaw_w[env_ids]),
        self.target_landing_position_w[env_ids] - self.startup_anchor_pos_w[env_ids],
      )
      bounded_startup[:, 2] = samples[:, 2]
      self.target_landing_position_startup[env_ids] = bounded_startup
    self._apply_analytic_strike_targets(env_ids)
    self.ball_has_been_struck[env_ids] = False
    self.ball_landing_recorded[env_ids] = False
    self.ball_landing_position_w[env_ids] = 0.0
    self.ball_previous_vertical_velocity[env_ids] = 0.0
    self.ball_previous_linear_velocity_w[env_ids] = 0.0
    self.ball_previous_racket_offset_w[env_ids] = 0.0
    self.ball_strike_state_initialized[env_ids] = False
    self.ball_post_strike_steps[env_ids] = 0
    self.ball_post_strike_elapsed_s[env_ids] = 0.0
    self.ball_hit_reward[env_ids] = 0.0
    self.ball_direction_reward[env_ids] = 0.0
    self.ball_net_clearance_reward[env_ids] = 0.0
    self.ball_out_speed_reward[env_ids] = 0.0
    self.racket_ball_min_distance[env_ids] = torch.inf
    self.racket_ball_score_max[env_ids] = 0.0
    self.racket_ball_min_abs_time_error[env_ids] = torch.inf
    self.racket_ball_distance_at_deadline[env_ids] = 0.0
    self.metrics["error_ball_landing"][env_ids] = 0.0
    self.metrics["ball_landing_prediction_valid"][env_ids] = 0.0
    self.metrics["ball_net_crossing_height"][env_ids] = 0.0
    self.metrics["ball_out_speed"][env_ids] = 0.0
    self.metrics["ball_target_projected_speed"][env_ids] = 0.0
    self.metrics["ball_direction_error_degrees"][env_ids] = 0.0
    self.metrics["post_strike_heading_error_degrees"][env_ids] = 0.0
    self.metrics["post_strike_heading_active"][env_ids] = 0.0
    self.metrics["racket_ball_min_distance"][env_ids] = 0.0
    self.metrics["racket_ball_score_max"][env_ids] = 0.0
    self.metrics["racket_ball_min_abs_time_error"][env_ids] = 0.0
    self.metrics["racket_ball_distance_at_deadline"][env_ids] = 0.0

  def _apply_analytic_strike_targets(self, env_ids: torch.Tensor) -> None:
    """Derive ball launch, racket velocity, and racket normal from sampled targets."""
    if not self.cfg.analytic_strike_planner_enabled or len(env_ids) == 0:
      return
    position_index = self._subtarget_goal_types.index("position")
    velocity_index = self._subtarget_goal_types.index("velocity")
    orientation_index = self._subtarget_goal_types.index("orientation")

    strike_position_w = self.target_position_w[env_ids, position_index]
    landing_position_w = self.target_landing_position_w[env_ids].clone()
    landing_position_w[:, 2] += self.cfg.analytic_ball_radius
    env_origins = self._env.scene.env_origins[env_ids]
    ball_velocity_w, feasible, net_height = solve_low_arc_tennis_velocity(
      strike_position_w,
      landing_position_w,
      preferred_speed=self.cfg.analytic_ball_preferred_speed,
      maximum_speed=self.cfg.analytic_ball_maximum_speed,
      net_x_w=env_origins[:, 0] + self.cfg.analytic_net_x,
      net_y_w=env_origins[:, 1],
      net_half_width=self.cfg.analytic_net_half_width,
      minimum_net_height=(
        self.cfg.analytic_net_height
        + self.cfg.analytic_ball_radius
        + self.cfg.analytic_net_clearance
      ),
      maximum_net_height=self.cfg.analytic_maximum_net_height,
      gravity_magnitude=self.cfg.analytic_gravity_magnitude,
    )
    incoming_velocity_w = torch.tensor(
      self.cfg.analytic_incoming_ball_velocity,
      device=self.device,
      dtype=ball_velocity_w.dtype,
    ).expand_as(ball_velocity_w)
    racket_velocity_w, racket_normal_w = inverse_racket_velocity_and_normal(
      incoming_velocity_w,
      ball_velocity_w,
      effective_restitution=self.cfg.analytic_racket_effective_restitution,
    )

    motion_ids = self.which_motion[env_ids]
    orientation_axis = self._target_ori_axes_t[motion_ids, orientation_index]
    self.target_velocity_w[env_ids, velocity_index] = racket_velocity_w
    self.target_orientation_w[env_ids, orientation_index] = (
      align_quaternion_axis_to_vector(
        self.target_orientation_w[env_ids, orientation_index],
        orientation_axis,
        racket_normal_w,
      )
    )
    self.target_ball_velocity_w[env_ids] = ball_velocity_w
    self.target_ball_out_speed[env_ids] = torch.linalg.vector_norm(
      ball_velocity_w, dim=-1
    )
    self.target_racket_velocity_w[env_ids] = racket_velocity_w
    self.target_racket_normal_w[env_ids] = racket_normal_w
    self.target_racket_speed[env_ids] = torch.linalg.vector_norm(
      racket_velocity_w, dim=-1
    )
    self.analytic_strike_feasible[env_ids] = feasible
    self.target_ball_net_height[env_ids] = net_height
    self.metrics["target_ball_out_speed"][env_ids] = self.target_ball_out_speed[env_ids]
    self.metrics["target_racket_speed"][env_ids] = self.target_racket_speed[env_ids]
    self.metrics["target_ball_net_height"][env_ids] = net_height
    self.metrics["analytic_strike_feasible"][env_ids] = feasible.to(
      self.metrics["analytic_strike_feasible"].dtype
    )

  def set_landing_target_std(self, std: tuple[float, float, float]) -> None:
    """Set the standard deviation used by future landing-target samples."""
    if any(value < 0.0 for value in std):
      raise ValueError("Landing target std values must be non-negative.")
    self._landing_target_std_t.copy_(
      torch.tensor(std, device=self.device, dtype=self._landing_target_std_t.dtype)
    )

  def update_landing_target_std_curriculum(self) -> dict[str, float]:
    """Update future samples without moving the target of an in-flight ball."""
    if self.cfg.landing_target_std_final is None:
      return {}
    progress, std = linear_landing_target_std(
      self._env.common_step_counter + self.landing_target_curriculum_step_offset,
      self.cfg.landing_target_std_ramp_steps,
      self.cfg.landing_target_std,
      self.cfg.landing_target_std_final,
    )
    self.set_landing_target_std(std)
    return {"progress": progress, "std_x": std[0], "std_y": std[1]}

  # ------------------------------------------------------------------
  # Source position / orientation helpers (body or site)
  # ------------------------------------------------------------------

  def get_source_pos_w(self) -> torch.Tensor:
    """Return ``(num_envs, max_subtargets, 3)`` world-frame source positions."""
    return self._source_pos_w_cache

  def get_source_quat_w(self) -> torch.Tensor:
    """Return ``(num_envs, max_subtargets, 4)`` world-frame source quaternions."""
    return self._source_quat_w_cache

  def get_source_lin_vel_w(self) -> torch.Tensor:
    """Return ``(num_envs, max_subtargets, 3)`` world-frame source linear velocities."""
    return self._source_lin_vel_w_cache

  def get_reference_source_strike_lin_vel_w(self) -> torch.Tensor:
    """Return each selected motion's strike-frame source-point velocity."""
    raw_velocity_w = self._reference_source_strike_lin_vel_t[self.which_motion]
    env_ids = torch.arange(self.num_envs, device=self.device)
    return self._align_reference_vector_w(raw_velocity_w, env_ids)

  def get_reference_source_strike_relative_lin_vel_b(self) -> torch.Tensor:
    """Strike-point velocity relative to the reference strike-frame Pelvis.

    Local vectors do not need world/yaw trajectory alignment.
    """
    if not hasattr(self, "_reference_source_strike_relative_lin_vel_t"):
      self._reference_source_strike_relative_lin_vel_t = (
        self._build_reference_source_strike_lin_velocities(relative_to_pelvis=True)
      )
    return self._reference_source_strike_relative_lin_vel_t[self.which_motion]

  def get_incoming_ball_racket_orientation_target_w(
    self,
    ball_velocity_w: torch.Tensor,
    *,
    minimum_ball_speed: float,
  ) -> torch.Tensor:
    """Return a signed target normal, frozen at the last pre-strike sample."""
    axis_vecs = self._target_ori_axes_t[self.which_motion]
    template_axis_w = quat_apply(
      self.target_orientation_w.reshape(-1, 4), axis_vecs.reshape(-1, 3)
    ).reshape(self.num_envs, self.max_subtargets, 3)
    expanded_ball_velocity_w = ball_velocity_w[:, None, :].expand_as(template_axis_w)
    live_target_w = signed_parallel_direction_from_template(
      template_axis_w, expanded_ball_velocity_w
    )

    strike_time_error_s = getattr(self, "strike_time_error_s", None)
    if strike_time_error_s is None:
      return live_target_w
    window_before_s = getattr(
      self.cfg,
      "racket_orientation_reward_window_before_s",
      self.cfg.contact_reward_window_s,
    )
    ball_speed = torch.linalg.vector_norm(ball_velocity_w, dim=-1)
    refresh_env = (
      (strike_time_error_s >= -window_before_s)
      & (strike_time_error_s <= 0.0)
      & (ball_speed >= minimum_ball_speed)
      & ~self.ball_has_been_struck
      & ~self.is_paused
    )
    orientation_target = self._target_ori_reward_weights_t[self.which_motion] > 0.0
    refresh = refresh_env[:, None] & orientation_target
    self.incoming_ball_racket_orientation_target_w[:] = torch.where(
      refresh[:, :, None],
      live_target_w,
      self.incoming_ball_racket_orientation_target_w,
    )
    self.incoming_ball_racket_orientation_target_valid |= refresh
    return torch.where(
      self.incoming_ball_racket_orientation_target_valid[:, :, None],
      self.incoming_ball_racket_orientation_target_w,
      live_target_w,
    )

  # ------------------------------------------------------------------
  # Metrics
  # ------------------------------------------------------------------

  def _update_metrics(self) -> None:
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

    target_pos_errors = torch.norm(
      self.get_source_pos_w() - self.target_position_w, dim=-1
    )
    position_mask = self._target_pos_reward_weights_t[self.which_motion] > 0.0
    self.metrics["error_target_pos"] = (target_pos_errors * position_mask.float()).sum(
      dim=-1
    ) / position_mask.sum(dim=-1).clamp(min=1)
    phase = self.normalized_phase
    active = (phase[:, None] >= self._target_phase_starts_t[self.which_motion]) & (
      phase[:, None] <= self._target_phase_ends_t[self.which_motion]
    )
    active_position_mask = position_mask & active
    active_count = active_position_mask.sum(dim=-1)
    in_target_window = active_count > 0
    if torch.any(in_target_window):
      self.metrics["error_target_pos_at_target"][in_target_window] = (
        target_pos_errors * active_position_mask.float()
      ).sum(dim=-1)[in_target_window] / active_count[in_target_window]

  # ------------------------------------------------------------------
  # Adaptive sampling
  # ------------------------------------------------------------------

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    episode_failed = self._env.termination_manager.terminated[env_ids]

    if torch.any(episode_failed):
      for i in range(len(self.motion_loaders)):
        motion_mask = self.which_motion[env_ids] == i
        failed_mask = episode_failed & motion_mask
        if torch.any(failed_mask):
          current_bin_index = torch.clamp(
            (self.time_steps[env_ids[failed_mask]] * self.bin_counts[i])
            // max(self.motion_loaders[i].time_step_total, 1),
            0,
            self.bin_counts[i] - 1,
          )
          self._current_bin_failed[i] = torch.bincount(
            current_bin_index, minlength=self.bin_counts[i]
          )

    sampling_probs_per_motion: list[torch.Tensor] = []
    for i in range(len(self.motion_loaders)):
      probs = self.bin_failed_counts[i] + self.cfg.adaptive_uniform_ratio / float(
        self.bin_counts[i]
      )
      probs = torch.nn.functional.pad(
        probs.unsqueeze(0).unsqueeze(0),
        (0, self.cfg.adaptive_kernel_size - 1),
        mode="replicate",
      )
      probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).view(-1)
      probs = probs / probs.sum()
      sampling_probs_per_motion.append(probs)

    sampled_bins = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
    for i in range(len(self.motion_loaders)):
      motion_mask = self.which_motion[env_ids] == i
      if torch.any(motion_mask):
        num_samples = int(motion_mask.sum().item())
        sampled = torch.multinomial(
          sampling_probs_per_motion[i], num_samples, replacement=True
        )
        sampled_bins[motion_mask] = sampled

    motion_indices = self.which_motion[env_ids]
    bin_counts_for_envs = torch.tensor(
      [self.bin_counts[m] for m in motion_indices.tolist()],
      device=self.device,
    )
    timestep_totals = torch.tensor(
      [self.motion_loaders[m].time_step_total for m in motion_indices.tolist()],
      device=self.device,
    )

    random_offsets = torch.rand(len(env_ids), device=self.device)
    new_timesteps = (
      (sampled_bins.float() + random_offsets)
      / bin_counts_for_envs.float()
      * (timestep_totals.float() - 1)
    ).long()
    self.time_steps[env_ids] = new_timesteps

    for i in range(len(self.motion_loaders)):
      motion_mask = self.which_motion[env_ids] == i
      if torch.any(motion_mask):
        probs = sampling_probs_per_motion[i]
        h = -(probs * (probs + 1e-12).log()).sum()
        h_norm = h / math.log(self.bin_counts[i]) if self.bin_counts[i] > 1 else 1.0
        pmax, imax = probs.max(dim=0)
        env_indices = torch.where(motion_mask)[0]
        actual_env_ids = env_ids[env_indices]
        self.metrics["sampling_entropy"][actual_env_ids] = h_norm
        self.metrics["sampling_top1_prob"][actual_env_ids] = pmax
        self.metrics["sampling_top1_bin"][actual_env_ids] = (
          imax.float() / self.bin_counts[i]
        )

  # ------------------------------------------------------------------
  # Resampling
  # ------------------------------------------------------------------

  def _sample_motion_ids(self, n: int) -> torch.Tensor:
    """Sample *n* motion indices according to ``motion_sampling_weights``.

    Weights are honoured proportionally: each motion gets exactly
    ``round(weight * n)`` slots, with any remainder filled by multinomial
    sampling.  The result is shuffled before being returned.
    """
    num_motions = len(self.motion_loaders)
    counts = (self._motion_weights_t * n).floor().long()
    remainder = n - counts.sum().item()
    if remainder > 0:
      extra = torch.multinomial(
        self._motion_weights_t, int(remainder), replacement=True
      )
      for idx in extra.tolist():
        counts[idx] += 1
    ids = torch.repeat_interleave(torch.arange(num_motions, device=self.device), counts)
    return ids[torch.randperm(n, device=self.device)]

  def set_motion_plan_provider(self, provider: MotionPlanProvider | None) -> None:
    """Install an optional reset-time provider used by trajectory-matched tasks."""
    self._motion_plan_provider = provider

  def _validate_motion_plan(
    self, plan: MotionResamplePlan, env_ids: torch.Tensor
  ) -> None:
    count = len(env_ids)
    expected_shapes = {
      "motion_ids": (count,),
      "target_indices": (count,),
      "target_positions_w": (count, 3),
      "contact_times": (count,),
      "launch_positions_w": (count, 3),
      "launch_linear_velocities_w": (count, 3),
      "launch_angular_velocities_w": (count, 3),
      "match_distances": (count,),
      "valid": (count,),
      "attempts": (count,),
    }
    for name, expected in expected_shapes.items():
      value = getattr(plan, name)
      if value.shape != expected:
        raise ValueError(
          f"Motion plan {name} has shape {tuple(value.shape)}, expected {expected}."
        )
      if value.device != env_ids.device:
        raise ValueError(
          f"Motion plan {name} is on {value.device}, expected {env_ids.device}."
        )
    if torch.any((plan.motion_ids < 0) | (plan.motion_ids >= len(self.motion_loaders))):
      raise ValueError("Motion plan contains an out-of-range motion id.")
    if torch.any(
      (plan.target_indices < 0) | (plan.target_indices >= self.max_subtargets)
    ):
      raise ValueError("Motion plan contains an out-of-range target index.")
    floating = (
      plan.target_positions_w,
      plan.contact_times,
      plan.launch_positions_w,
      plan.launch_linear_velocities_w,
      plan.launch_angular_velocities_w,
      plan.match_distances,
    )
    if any(not torch.all(torch.isfinite(value)) for value in floating):
      raise ValueError("Motion plan contains a non-finite value.")

  def _motion_plan_failure_mask(self, plan: MotionResamplePlan) -> torch.Tensor:
    # Without FailureTraj, an over-distance nearest match still starts its motion.
    # Its motion, target, and deadline must be applied together.
    if self.cfg.incoming_ball_failure_trajectory_probability > 0.0:
      return ~plan.valid
    return torch.zeros_like(plan.valid)

  def _apply_motion_plan(self, plan: MotionResamplePlan, env_ids: torch.Tensor) -> None:
    """Apply one atomic trajectory, motion, target, and deadline plan."""
    failure = self._motion_plan_failure_mask(plan)
    valid_env_ids = env_ids[~failure]
    valid_target_indices = plan.target_indices[~failure]
    if len(valid_env_ids) > 0:
      self.target_position_w[valid_env_ids, valid_target_indices] = (
        plan.target_positions_w[~failure]
      )
      self._apply_analytic_strike_targets(valid_env_ids)
    self.planned_contact_time[env_ids] = plan.contact_times
    self.planned_target_index[env_ids] = plan.target_indices
    self.planned_launch_position_w[env_ids] = plan.launch_positions_w
    self.planned_launch_linear_velocity_w[env_ids] = plan.launch_linear_velocities_w
    self.planned_launch_angular_velocity_w[env_ids] = plan.launch_angular_velocities_w
    self.metrics["trajectory_match_distance"][env_ids] = plan.match_distances
    self.trajectory_match_valid[env_ids] = plan.valid
    self.metrics["trajectory_match_valid"][env_ids] = plan.valid.float()
    self.metrics["failure_trajectory"][env_ids] = failure.float()
    self.metrics["trajectory_match_attempt"][env_ids] = plan.attempts.float()
    self.metrics["trajectory_prediction_error"][env_ids] = 0.0

  def _record_stroke_outcomes(
    self, env_ids: torch.Tensor, *, force: bool = False
  ) -> None:
    """Hook for deadline-aware commands that track physical strike outcomes."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return

    self._record_stroke_outcomes(env_ids, force=True)

    self.motion_chain_count[env_ids] = 0
    self.metrics["motion_chain_count"][env_ids] = 0.0

    plan = (
      None
      if self._motion_plan_provider is None
      else self._motion_plan_provider(env_ids)
    )
    if plan is None:
      self.which_motion[env_ids] = self._sample_motion_ids(len(env_ids))
    else:
      self._validate_motion_plan(plan, env_ids)
      self.which_motion[env_ids] = plan.motion_ids

    if plan is not None or self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      motion_indices = self.which_motion[env_ids]
      timestep_totals = torch.tensor(
        [self.motion_loaders[m].time_step_total for m in motion_indices.tolist()],
        device=self.device,
      )
      self.time_steps[env_ids] = (
        torch.rand(len(env_ids), device=self.device) * (timestep_totals.float() - 1)
      ).long()
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    self._reset_robot_from_selected_motion(env_ids)
    if plan is not None:
      self._apply_motion_plan(plan, env_ids)
    self.motion_chain_count[env_ids] += 1
    self.metrics["motion_chain_count"][env_ids] = self.motion_chain_count[
      env_ids
    ].float()

  def _reset_robot_from_selected_motion(self, env_ids: torch.Tensor) -> None:
    """Apply the shared physical-state initialization for selected motions."""
    motion_ids = self.which_motion[env_ids]
    frame0_local_mask = self._uses_reference_frame0(motion_ids)
    frame0_env_ids = env_ids[frame0_local_mask]
    # Frame-0-local tennis targets require a well-defined initial alignment.
    # These motions always begin at frame 0; native motions keep their configured
    # start/uniform/adaptive sampling behavior.
    self.time_steps[frame0_env_ids] = 0

    t = self._clamped_time_steps()
    root_pos = (
      self._stacked_body_pos_w[self.which_motion, t, 0] + self._env.scene.env_origins
    ).clone()
    root_ori = self._stacked_body_quat_w[self.which_motion, t, 0].clone()
    root_lin_vel = self._stacked_body_lin_vel_w[self.which_motion, t, 0].clone()
    root_ang_vel = self._stacked_body_ang_vel_w[self.which_motion, t, 0].clone()

    if len(frame0_env_ids) > 0:
      # Put the G1 planar root at each environment's local world origin while
      # preserving the reference pelvis height and roll/pitch.
      zero_yaw_delta = quat_inv(yaw_quat(root_ori[frame0_env_ids]))
      root_pos[frame0_env_ids, :2] = self._env.scene.env_origins[frame0_env_ids, :2]
      root_ori[frame0_env_ids] = quat_mul(zero_yaw_delta, root_ori[frame0_env_ids])
      root_lin_vel[frame0_env_ids] = quat_apply(
        zero_yaw_delta, root_lin_vel[frame0_env_ids]
      )
      root_ang_vel[frame0_env_ids] = quat_apply(
        zero_yaw_delta, root_ang_vel[frame0_env_ids]
      )

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    # Keep frame-0-local tennis motions exactly at local x/y/yaw zero.
    rand_samples[frame0_local_mask] = 0.0
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self._stacked_joint_pos[self.which_motion, t].clone()
    joint_vel = self._stacked_joint_vel[self.which_motion, t].clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore[arg-type]
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids],
      soft_joint_pos_limits[:, :, 0],
      soft_joint_pos_limits[:, :, 1],
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

    # _sample_targets reads body/site world transforms; ensure they reflect
    # the newly written reset state before sampling.
    self._env.sim.forward()
    self.startup_anchor_pos_w[env_ids] = self.robot_anchor_pos_w[env_ids]
    self.startup_anchor_yaw_w[env_ids] = yaw_quat(self.robot_anchor_quat_w[env_ids])
    self._align_reference_frame0_to_robot(env_ids)
    self._update_ghost_alignment(env_ids)
    self._sample_targets(env_ids)
    self._update_source_cache()

  def _update_ghost_alignment(self, env_ids: torch.Tensor) -> None:
    """Cache a fixed frame-0 reference-to-robot transform for ghost playback."""
    if self.cfg.viz.ghost_root_mode != "initial_aligned" or len(env_ids) == 0:
      return
    motion_ids = self.which_motion[env_ids]
    ref_root_quat_0 = self._stacked_body_quat_w[motion_ids, 0, 0]
    robot_root_quat = self.robot.data.body_link_quat_w[env_ids, 0]
    self._ghost_align_root_pos_w[env_ids] = self.robot.data.body_link_pos_w[env_ids, 0]
    self._ghost_align_yaw_w[env_ids] = yaw_quat(
      quat_mul(robot_root_quat, quat_inv(ref_root_quat_0))
    )

  def _sample_next_motion(self, env_ids: torch.Tensor) -> None:
    """Pick and align the next motion without resetting the physical robot."""
    if len(env_ids) == 0:
      return
    self._record_stroke_outcomes(env_ids, force=True)
    plan = None
    reference_env_ids = env_ids
    if self._motion_plan_provider is not None:
      chain_provider = getattr(
        self._motion_plan_provider, "plan_for_motion_chain", None
      )
      if chain_provider is None:
        raise TypeError(
          "Motion-plan provider must implement plan_for_motion_chain() when "
          "auto_chain_motion is enabled."
        )
      plan = chain_provider(env_ids)
      self._validate_motion_plan(plan, env_ids)
      failure = self._motion_plan_failure_mask(plan)
      reference_env_ids = env_ids[~failure]
      self.which_motion[reference_env_ids] = plan.motion_ids[~failure]
    else:
      self.which_motion[env_ids] = self._sample_motion_ids(len(env_ids))
    self.time_steps[reference_env_ids] = 0
    if plan is not None:
      failed_env_ids = env_ids[failure]
      if len(failed_env_ids) > 0:
        failed_lengths = self._time_step_totals[self.which_motion[failed_env_ids]]
        self.time_steps[failed_env_ids] = failed_lengths - 1
    # Re-anchor the selected reference frame 0 and its ball target to the
    # physical G1 root without resetting the robot.
    self._align_reference_frame0_to_robot(reference_env_ids)
    self._update_ghost_alignment(reference_env_ids)
    self._sample_targets(reference_env_ids)
    if plan is not None:
      self._apply_motion_plan(plan, env_ids)
    self.motion_chain_count[env_ids] += 1
    self.metrics["motion_chain_count"][env_ids] = self.motion_chain_count[
      env_ids
    ].float()

  def set_motion_chaining(self, env_ids: torch.Tensor, *, enabled: bool) -> None:
    """Enable or disable automatic motion/target sampling after completion."""
    self.auto_chain_motion[env_ids] = enabled
    if not enabled:
      self.is_paused[env_ids] = False
      self.between_motion_pause_time[env_ids] = 0.0

  # ------------------------------------------------------------------
  # Step update
  # ------------------------------------------------------------------

  def _update_command(self) -> None:
    self.time_steps += 1
    self.ball_post_strike_elapsed_s += (
      self.ball_has_been_struck.to(self.ball_post_strike_elapsed_s.dtype)
      * self._env.step_dt
    )

    # Update moving targets each step — fully vectorized over (E, S).
    has_moving = self._has_target_link[self.which_motion]  # (E, S)
    if torch.any(has_moving):
      E = self.num_envs
      S = self.max_subtargets
      motion_ids = self.which_motion  # (E,)
      all_env_ids = torch.arange(E, device=self.device)
      tgt_pos_all, tgt_quat_all = self._fetch_link_state_batched(
        all_env_ids,
        self._target_body_indices_t[motion_ids],  # (E, S)
        self._target_is_site_t[motion_ids],  # (E, S)
      )  # (E, S, 3/4)
      # Use the per-episode random offset sampled at reset (target-body local frame),
      # rotated by the live target-body orientation each step.
      rand_offset_all = self._moving_target_offset_w[
        all_env_ids
      ]  # (E, S, 3), body local
      world_offset_all = quat_apply(
        tgt_quat_all.reshape(-1, 4),
        rand_offset_all.reshape(-1, 3),
      ).reshape(E, S, 3)
      self.target_position_w[has_moving] = (tgt_pos_all + world_offset_all)[has_moving]
      self.target_orientation_w[has_moving] = tgt_quat_all[has_moving]

    # Handle motion completion with between-motion pause.
    timestep_limits = self._time_step_totals[self.which_motion]
    env_ids_at_end = torch.where(self.time_steps >= timestep_limits)[0]
    held_at_end = env_ids_at_end[~self.auto_chain_motion[env_ids_at_end]]
    if len(held_at_end) > 0:
      self.time_steps[held_at_end] = timestep_limits[held_at_end] - 1
      self.between_motion_pause_time[held_at_end] = 0.0
    env_ids_at_end = env_ids_at_end[self.auto_chain_motion[env_ids_at_end]]

    newly_paused = env_ids_at_end[~self.is_paused[env_ids_at_end]]
    if len(newly_paused) > 0:
      lo, hi = self.between_motion_pause_range
      self._sampled_pause_lengths[newly_paused] = (
        torch.rand(len(newly_paused), device=self.device) * (hi - lo) + lo
      )
      motion_ids = self.which_motion[newly_paused]
      raw_pos_0 = (
        self._stacked_body_pos_w[motion_ids, 0]
        + self._env.scene.env_origins[newly_paused, None, :]
      )
      raw_quat_0 = self._stacked_body_quat_w[motion_ids, 0]
      self._paused_body_pos_w[newly_paused] = self._align_reference_body_pos_w(
        raw_pos_0, newly_paused
      )
      self._paused_body_quat_w[newly_paused] = self._align_reference_quat_w(
        raw_quat_0, newly_paused
      )
      self._paused_joint_pos[newly_paused] = self._stacked_joint_pos[motion_ids, 0]
      self._paused_joint_vel[newly_paused] = self._stacked_joint_vel[motion_ids, 0]

    self.is_paused[:] = False
    self.is_paused[env_ids_at_end] = True
    self.between_motion_pause_time[env_ids_at_end] += self._env.step_dt
    env_ids_to_continue = env_ids_at_end[
      self.between_motion_pause_time[env_ids_at_end]
      >= self._sampled_pause_lengths[env_ids_at_end]
    ]

    self._motion_plan_batch_step = (
      self._motion_plan_batch_step + 1
    ) % self._motion_plan_batch_interval_steps
    planning_batch_due = self._motion_plan_batch_step == 0
    if len(env_ids_to_continue) > 0 and planning_batch_due:
      self.between_motion_pause_time[env_ids_to_continue] = 0.0
      self._sample_next_motion(env_ids_to_continue)

    # Update relative body poses.
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

    if self.cfg.sampling_mode == "adaptive":
      for i in range(len(self.motion_loaders)):
        self.bin_failed_counts[i] = (
          self.cfg.adaptive_alpha * self._current_bin_failed[i]
          + (1 - self.cfg.adaptive_alpha) * self.bin_failed_counts[i]
        )
        self._current_bin_failed[i].zero_()

    self._update_source_cache()

  # ------------------------------------------------------------------
  # Source-state cache
  # ------------------------------------------------------------------

  def _update_source_cache(self) -> None:
    """Fetch source link pos/quat/vel once and store in cache tensors.

    Called at the end of every _update_command and after _sample_targets so
    that get_source_pos_w / get_source_quat_w / get_source_lin_vel_w are
    simple tensor reads with no redundant data fetches.
    """
    all_env_ids = torch.arange(self.num_envs, device=self.device)
    source_indices = self._source_body_indices_t[self.which_motion]  # (E, S)
    source_is_site = self._source_is_site_t[self.which_motion]  # (E, S)

    pos, quat = self._fetch_link_state_batched(
      all_env_ids, source_indices, source_is_site
    )
    self._source_pos_w_cache = pos
    self._source_quat_w_cache = quat

    E, S = source_indices.shape
    env_ids_exp = all_env_ids[:, None].expand(E, S)
    body_idx = source_indices.clamp(
      max=self.robot.data.body_link_lin_vel_w.shape[1] - 1
    )
    site_idx = source_indices.clamp(max=self.robot.data.site_lin_vel_w.shape[1] - 1)
    body_vel = self.robot.data.body_link_lin_vel_w[env_ids_exp, body_idx]
    site_vel = self.robot.data.site_lin_vel_w[env_ids_exp, site_idx]
    self._source_lin_vel_w_cache = torch.where(
      source_is_site.unsqueeze(-1), site_vel, body_vel
    )

  # ------------------------------------------------------------------
  # Visualization
  # ------------------------------------------------------------------

  _VIZ_LINK_COLORS = ((1.0, 0.3, 0.3), (0.3, 1.0, 0.3), (0.3, 0.3, 1.0))

  def _add_incoming_ball_launch_region_vis(
    self, visualizer: DebugVisualizer, batch: int
  ) -> None:
    """Draw fixed launch-position sampling bounds in the environment frame."""
    if (
      not self.cfg.incoming_ball_torch_match_enabled
      or not self.cfg.viz.show_incoming_ball_launch_region
    ):
      return

    lower = torch.as_tensor(
      self.cfg.incoming_ball_torch_match_position_min,
      device=self.device,
      dtype=self._env.scene.env_origins.dtype,
    )
    upper = torch.as_tensor(
      self.cfg.incoming_ball_torch_match_position_max,
      device=self.device,
      dtype=self._env.scene.env_origins.dtype,
    )

    local_center = 0.5 * (lower + upper)
    half_extents = 0.5 * (upper - lower)
    center_w = self._env.scene.env_origins[batch] + local_center

    visualizer.add_box(
      center=center_w.cpu().numpy(),
      size=half_extents.cpu().numpy(),
      mat=np.eye(3),
      color=(1.0, 0.85, 0.0, 0.20),
      label=f"incoming_ball_launch_region_{batch}",
    )

  def _add_target_vis(self, visualizer: DebugVisualizer, batch: int) -> None:
    """Draw position, orientation-axis, and velocity targets for one env."""
    self._add_incoming_ball_launch_region_vis(visualizer, batch)
    source_pos_all = self.get_source_pos_w()[batch].cpu().numpy()  # (S, 3)
    source_quat_all = self.get_source_quat_w()[batch]  # (S, 4)
    motion_cfg = self.motion_configs[self.which_motion[batch].item()]
    num_active = len(motion_cfg.sub_targets)
    for s in range(num_active):
      sub_target = motion_cfg.sub_targets[s]
      target_pos = self.target_position_w[batch, s].cpu().numpy()
      if sub_target.goal_type != "position":
        matching_position = next(
          (
            i
            for i, candidate in enumerate(motion_cfg.sub_targets)
            if candidate.goal_type == "position"
            and candidate.source_link == sub_target.source_link
          ),
          None,
        )
        if matching_position is not None:
          target_pos = self.target_position_w[batch, matching_position].cpu().numpy()
      if sub_target.goal_type == "position":
        visualizer.add_sphere(
          center=target_pos,
          radius=0.025,
          color=(0.0, 1.0, 0.0, 0.7),
          label=f"target_{batch}_{s}",
        )
      source_rotm = matrix_from_quat(source_quat_all[s].unsqueeze(0))[0].cpu().numpy()
      visualizer.add_frame(
        position=source_pos_all[s],
        rotation_matrix=source_rotm,
        scale=0.12,
        label=f"source_{batch}_{s}",
        axis_colors=self._VIZ_LINK_COLORS,
      )
      if sub_target.goal_type == "orientation":
        orientation_axis = self._target_ori_axes_t[
          self.which_motion[batch], s
        ].unsqueeze(0)
        source_axis_t = quat_apply(source_quat_all[s].unsqueeze(0), orientation_axis)[0]
        template_axis_t = quat_apply(
          self.target_orientation_w[batch, s].unsqueeze(0), orientation_axis
        )[0]
        target_axis_t = template_axis_t
        target_color = (0.1, 0.5, 1.0, 0.9)
        if self.cfg.viz.show_incoming_ball_racket_orientation_target:
          ball_velocity_w = self._env.scene[
            self.cfg.viz.incoming_ball_entity_name
          ].data.root_link_lin_vel_w[batch]
          target_axes_w = self.get_incoming_ball_racket_orientation_target_w(
            self._env.scene[
              self.cfg.viz.incoming_ball_entity_name
            ].data.root_link_lin_vel_w,
            minimum_ball_speed=self.cfg.viz.incoming_ball_minimum_speed,
          )
          orientation_active = incoming_ball_racket_orientation_active(
            self.strike_time_error_s[batch],
            torch.linalg.vector_norm(ball_velocity_w),
            self.incoming_ball_racket_orientation_target_valid[batch, s],
            is_paused=self.is_paused[batch],
            window_before_s=self.cfg.racket_orientation_reward_window_before_s,
            window_after_s=self.cfg.racket_orientation_reward_window_after_s,
            minimum_ball_speed=self.cfg.viz.incoming_ball_minimum_speed,
          )
          if not bool(orientation_active):
            continue
          target_axis_t = target_axes_w[batch, s]
          target_color = (1.0, 0.85, 0.05, 0.95)
        arrow_length = self.cfg.viz.racket_orientation_arrow_length
        source_axis = source_axis_t.cpu().numpy()
        source_axis /= max(np.linalg.norm(source_axis), 1.0e-6)
        visualizer.add_arrow(
          start=source_pos_all[s],
          end=source_pos_all[s] + source_axis * arrow_length,
          color=(0.1, 0.5, 1.0, 0.9),
          label=f"source_ori_{batch}_{s}",
        )
        target_axis = target_axis_t.cpu().numpy()
        target_axis /= max(np.linalg.norm(target_axis), 1.0e-6)
        visualizer.add_arrow(
          start=target_pos,
          end=target_pos + target_axis * arrow_length,
          color=target_color,
          label=f"target_ori_{batch}_{s}",
        )
      if sub_target.goal_type == "velocity":
        target_vel = self.target_velocity_w[batch, s].cpu().numpy()
        if np.linalg.norm(target_vel) <= 1e-4:
          continue
        visualizer.add_arrow(
          start=target_pos,
          # 0.50 metres of arrow per 1 m/s of target speed.
          end=target_pos + target_vel * 0.50,
          color=(1.0, 0.5, 0.0, 0.9),
          label=f"target_vel_{batch}_{s}",
        )

    if self.cfg.landing_target_enabled:
      center = self.target_landing_position_w[batch].cpu().numpy().copy()
      center[2] = 0.025
      radius = self.cfg.landing_target_radius
      angles = np.linspace(0.0, 2.0 * np.pi, 25)
      points = np.column_stack(
        (
          center[0] + radius * np.cos(angles),
          center[1] + radius * np.sin(angles),
          np.full_like(angles, center[2]),
        )
      )
      for segment in range(len(points) - 1):
        visualizer.add_cylinder(
          start=points[segment],
          end=points[segment + 1],
          radius=0.018,
          color=(1.0, 0.85, 0.05, 0.9),
          label=f"landing_target_{batch}_{segment}",
        )

  def _add_root_frame_vis(self, visualizer: DebugVisualizer, batch: int) -> None:
    if not self.cfg.viz.show_root_frame:
      return
    visualizer.add_frame(
      position=self.robot.data.root_link_pos_w[batch].cpu().numpy(),
      rotation_matrix=matrix_from_quat(
        self.robot.data.root_link_quat_w[batch]
      ).cpu().numpy(),
      scale=0.3,
      label=f"robot_root_{batch}",
    )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    for batch in env_indices:
      self._add_root_frame_vis(visualizer, batch)

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        motion_id = self.which_motion[batch]
        uses_frame0_alignment = bool(
          self._uses_reference_frame0(motion_id.reshape(1))[0]
        )
        if uses_frame0_alignment:
          ghost_root_pos = self.body_pos_w[batch, 0]
          ghost_root_quat = self.body_quat_w[batch, 0]
        elif self.cfg.viz.ghost_root_mode == "initial_aligned":
          ref_root_pos_0 = (
            self._stacked_body_pos_w[motion_id, 0, 0]
            + self._env.scene.env_origins[batch]
          )
          ref_displacement = self.body_pos_w[batch, 0] - ref_root_pos_0
          ghost_root_pos = (
            self._ghost_align_root_pos_w[batch]
            + quat_apply(
              self._ghost_align_yaw_w[batch].unsqueeze(0),
              ref_displacement.unsqueeze(0),
            )[0]
          )
          ghost_root_quat = quat_mul(
            self._ghost_align_yaw_w[batch], self.body_quat_w[batch, 0]
          )
        else:
          ghost_root_pos = self.body_pos_relative_w[batch, 0]
          ghost_root_quat = self.body_quat_relative_w[batch, 0]
        qpos[free_joint_q_adr[0:3]] = ghost_root_pos.cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = ghost_root_quat.cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()
        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")
        self._add_target_vis(visualizer, batch)

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_rotm = matrix_from_quat(self.body_quat_w[batch]).cpu().numpy()
        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_rotm = (
          matrix_from_quat(self.robot_body_quat_w[batch]).cpu().numpy()
        )

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )
        self._add_target_vis(visualizer, batch)


@dataclass(kw_only=True)
class MultiTargetMotionCommandCfg(CommandTermCfg):
  """Configuration for the multi-target motion command."""

  entity_name: str
  motion_files: list[str] = field(default_factory=list)
  anchor_body_name: str = ""
  body_names: tuple[str, ...] = ()

  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)

  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

  motion_sampling_weights: list[float] = field(default_factory=list)
  """Per-motion sampling proportions. Defaults to uniform when empty."""

  motion_target_cfgs: list[MotionCfg] = field(default_factory=list)
  """Per-motion target configurations, each containing one or more sub-targets."""

  target_pos_std_scale: float = 1.0
  """Global scale applied to every sampled position target standard deviation."""

  between_motion_pause_range: tuple[float, float] = (0.0, 1.0)
  """(min, max) seconds for the between-motion pause, sampled uniformly per env."""

  auto_chain_motion: bool = True
  """Automatically sample another motion after the current one completes."""

  landing_target_enabled: bool = False
  """Sample and expose a tennis first-bounce target when enabled."""
  landing_target_frame: Literal["environment", "startup"] = "environment"
  """Frame for landing-target mean/std; startup is the episode-initial root SE(2)."""
  landing_target_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Mean first-bounce target expressed in ``landing_target_frame``."""
  landing_target_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Per-axis Gaussian standard deviation for first-bounce target sampling."""
  landing_target_std_final: tuple[float, float, float] | None = None
  """Optional endpoint of a linear std curriculum, before net truncation."""
  landing_target_std_ramp_steps: int = 0
  """Control steps to reach final std, independent of the parallel env count."""
  landing_target_net_margin_m: float | None = None
  """Optional minimum target X beyond the net in the fixed court frame."""
  landing_target_radius: float = 0.5
  """Radius that receives full first-bounce reward."""

  incoming_ball_launch_enabled: bool = False
  """Launch a configured physical-ball trajectory toward the strike target."""
  incoming_ball_initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Ball launch position in environment-local world coordinates."""
  incoming_ball_initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Ball launch linear velocity in world coordinates."""
  incoming_ball_initial_angular_velocity: tuple[float, float, float] = (
    0.0,
    0.0,
    0.0,
  )
  """Ball launch angular velocity in world coordinates."""
  incoming_ball_flight_time_s: float = 0.0
  """Nominal time from launch to the configured strike point."""
  incoming_ball_trajectory_drives_deadline: bool = False
  """Use the trajectory flight time as the strike deadline and launch immediately."""
  incoming_ball_strike_target_position_frame0: tuple[float, float, float] | None = None
  """Optional fixed strike position in the reference frame-0 anchor coordinates."""
  incoming_ball_position_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Per-axis launch-position noise, sampled once per episode."""
  incoming_ball_velocity_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Per-axis launch-velocity noise, sampled once per episode."""
  incoming_ball_max_strike_deviation: float = 0.0
  """Maximum trajectory translation at strike; zero disables the bound."""

  incoming_ball_torch_match_enabled: bool = False
  """Select a motion by fixed-step Torch simulation of each sampled launch."""
  incoming_ball_torch_match_expected_motions: int = 0
  """Required motion count; zero disables the dataset-size assertion."""
  incoming_ball_torch_match_dt: float = 0.01
  incoming_ball_torch_match_horizon_s: float = 4.0
  incoming_ball_torch_match_max_distance: float = 0.15
  incoming_ball_torch_match_attempts: int = 4
  incoming_ball_torch_match_retry_rounds: int = 1
  incoming_ball_torch_match_adaptive_attempt_batch_size: int = 0
  """Candidates per adaptive wave; zero keeps fixed-size best-of-N matching."""
  incoming_ball_torch_match_motion_chunk_size: int = 8
  incoming_ball_torch_match_hierarchical_top_k: int = 0
  incoming_ball_torch_match_hierarchical_coarse_neighbor_radius: int = 1
  incoming_ball_torch_match_chain_batch_interval_steps: int = 1
  """Accumulate ready motion-chain envs for this many control steps."""
  incoming_ball_torch_match_cache_size: int = 0
  """Plans cached per environment; zero recomputes a plan at every reset."""
  incoming_ball_failure_trajectory_probability: float = 0.0
  """Probability of retaining a physically invalid launch as an idle sample."""
  incoming_ball_failure_trajectory_xy_distance_threshold_m: float = 3.0
  """Over-net strike segments staying beyond this planar root distance fail."""
  incoming_ball_failure_no_net_fraction: float = 0.5
  """Preferred fraction of failure samples that do not clear the net."""
  incoming_ball_failure_rollout_horizon_s: float | None = None
  """Maximum abnormal-ball rollout while waiting for a physical stop condition."""
  incoming_ball_failure_observation_radius_m: float | None = None
  """End abnormal trajectories beyond this current-root observation radius."""
  incoming_ball_failure_stop_speed_m_s: float | None = None
  """Treat speeds below this numerical tolerance as stopped."""
  incoming_ball_failure_random_direction_fraction: float = 0.0
  """Candidate fraction with uniformly sampled 360-degree launch headings."""
  incoming_ball_failure_random_position_radius_range_m: tuple[float, float] = (
    2.0,
    5.0,
  )
  """Root-relative source radius for full-azimuth abnormal trajectories."""
  incoming_ball_failure_random_position_height_range_m: tuple[float, float] = (
    0.8,
    2.2,
  )
  """Root-relative source height range for full-azimuth abnormal trajectories."""
  incoming_ball_failure_random_vertical_speed_range_m_s: tuple[float, float] = (
    -2.0,
    4.0,
  )
  """Vertical-speed range for full-azimuth abnormal launch candidates."""
  incoming_ball_failure_overhead_fraction: float = 0.0
  """Fraction of deliberately failed balls that fly above the current root."""
  incoming_ball_failure_overhead_speed_range: tuple[float, float] = (8.0, 12.0)
  incoming_ball_failure_overhead_height_range: tuple[float, float] = (2.2, 2.8)
  incoming_ball_trajectory_rollout_backend: Literal["torch", "warp_fused"] = "torch"
  """Backend used only for candidate-ball fixed-step trajectory rollout."""
  incoming_ball_torch_match_root_directed_sampling: bool = False
  """Sample a fixed-court launch aimed at the current G1 root and solve vz."""
  incoming_ball_torch_match_horizontal_speed_range_m_s: tuple[float, float] = (
    3.5,
    5.25,
  )
  incoming_ball_torch_match_horizontal_angle_half_width_deg: float = 15.0
  tennis_court_net_x_m: float = 5.6
  """Net X coordinate shared by the court, trajectory planner, and rewards."""
  tennis_court_net_half_width_m: float = 5.485
  """Net half-width shared by the court, trajectory planner, and rewards."""
  incoming_ball_torch_match_net_crossing_height_range_m: tuple[float, float] = (
    1.5,
    3.5,
  )
  incoming_ball_torch_match_maximum_initial_speed_m_s: float = 7.0
  incoming_ball_torch_match_deadline_offset_s: float = 0.0
  """Command deadline minus the matched physical-trajectory sample time."""
  incoming_ball_torch_match_position_min: tuple[float, float, float] = (
    10.5,
    -3.0,
    1.0,
  )
  incoming_ball_torch_match_position_max: tuple[float, float, float] = (
    13.0,
    3.0,
    2.0,
  )
  incoming_ball_torch_match_velocity_min: tuple[float, float, float] = (
    -10.5,
    -2.5,
    2.5,
  )
  incoming_ball_torch_match_velocity_max: tuple[float, float, float] = (
    -7.0,
    2.5,
    6.0,
  )

  analytic_strike_planner_enabled: bool = False
  """Derive ball and racket goals analytically from strike and landing positions."""
  analytic_ball_preferred_speed: float = 10.0
  analytic_ball_maximum_speed: float = 20.0
  analytic_incoming_ball_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
  analytic_racket_effective_restitution: float = 0.7
  analytic_gravity_magnitude: float = 9.81
  analytic_ball_radius: float = 0.0335
  analytic_net_x: float = 5.6
  analytic_net_height: float = 0.914
  analytic_net_half_width: float = 5.485
  analytic_net_clearance: float = 0.0
  analytic_maximum_net_height: float = 5.0

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    show_root_frame: bool = False
    """Draw the actual robot root XYZ axes (0.3 m), including roll and pitch."""
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)
    ghost_root_mode: Literal["robot_relative", "initial_aligned"] = "robot_relative"
    show_incoming_ball_launch_region: bool = True
    """Draw the Torch/Warp launch-position sampling region in debug viewers."""
    show_incoming_ball_racket_orientation_target: bool = False
    """Draw the signed incoming-velocity racket-normal target when enabled."""
    incoming_ball_entity_name: str = "tennis_ball"
    incoming_ball_minimum_speed: float = 0.5
    racket_orientation_arrow_length: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MultiTargetMotionCommand:
    return MultiTargetMotionCommand(self, env)
