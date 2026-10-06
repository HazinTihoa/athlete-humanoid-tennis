from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MultiTargetMotionCommand
from .phase_commands import PhaseAwareMultiTargetMotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_root_tilt(
  env: ManagerBasedRlEnv, command_name: str, max_tilt_degrees: float
) -> torch.Tensor:
  """Terminate when the robot root anchor tilts beyond the configured angle."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, command.robot.data.gravity_vec_w
  )
  upright_cosine = -gravity_b[:, 2] / torch.linalg.vector_norm(
    gravity_b, dim=-1
  ).clamp_min(1.0e-6)
  return upright_cosine < math.cos(math.radians(max_tilt_degrees))


def bad_pelvis_and_torso_tilt(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_tilt_degrees: float,
  pelvis_body_name: str = "pelvis",
  torso_body_name: str = "torso_link",
) -> torch.Tensor:
  """Terminate only when both pelvis and torso exceed the world-up tilt limit."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  robot = command.robot
  try:
    pelvis_index = robot.body_names.index(pelvis_body_name)
    torso_index = robot.body_names.index(torso_body_name)
  except ValueError as error:
    raise ValueError(
      "Pelvis/torso tilt termination requires bodies "
      f"{pelvis_body_name!r} and {torso_body_name!r}; available bodies: "
      f"{robot.body_names}"
    ) from error

  body_quat_w = robot.data.body_link_quat_w[:, (pelvis_index, torso_index)]
  num_envs, num_bodies = body_quat_w.shape[:2]
  gravity_w = robot.data.gravity_vec_w[:, None, :].expand(-1, num_bodies, -1)
  # mjlab's quaternion helper operates on a flat batch, not [env, body, ...].
  gravity_b = quat_apply_inverse(
    body_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3)
  ).reshape(num_envs, num_bodies, 3)
  upright_cosine = -gravity_b[..., 2] / torch.linalg.vector_norm(
    gravity_b, dim=-1
  ).clamp_min(1.0e-6)
  max_tilt_cosine = math.cos(math.radians(max_tilt_degrees))
  # Keep the threshold strictly outside the allowed angle despite float roundoff.
  return torch.all(upright_cosine < max_tilt_cosine - 1.0e-6, dim=-1)


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)


def phase_motion_complete(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """End an episode after the phase-aware reference finishes its follow-through."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, PhaseAwareMultiTargetMotionCommand):
    raise TypeError(
      f"Command {command_name!r} must be phase-aware, got {type(command).__name__}."
    )
  return command.phase >= 1.0 - 1.0e-6
