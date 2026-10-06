from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_error_magnitude, yaw_quat

from .commands import (
  MultiTargetMotionCommand,
  incoming_ball_racket_orientation_active,
  signed_parallel_direction_from_template,
)
from .phase_commands import PhaseAwareMultiTargetMotionCommand
from .relative_velocity import point_velocity_relative_to_frame

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MultiTargetMotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def torso_upright_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_name: str = "torso_link",
  std_degrees: float = 30.0,
) -> torch.Tensor:
  """Reward torso alignment with gravity's opposite direction, independent of yaw.

  Uses the actual torso pose, including waist compensation for a tilted pelvis.
  The score is one when upright and decays smoothly from zero tilt; it applies
  throughout the rally, including failed-ball and waiting phases.
  """
  if not math.isfinite(std_degrees) or std_degrees <= 0.0:
    raise ValueError("Torso upright reward std_degrees must be finite and positive.")
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  robot = command.robot
  body_index = robot.body_names.index(body_name)
  gravity_b = quat_apply_inverse(
    robot.data.body_link_quat_w[:, body_index], robot.data.gravity_vec_w
  )
  upright_cosine = (
    -gravity_b[:, 2] / torch.linalg.vector_norm(gravity_b, dim=-1).clamp_min(1.0e-6)
  ).clamp(-1.0, 1.0)
  tilt = torch.acos(upright_cosine)
  return torch.exp(-torch.square(tilt / math.radians(std_degrees)))


def _racket_reward_source_state(
  command: MultiTargetMotionCommand, *, velocity: bool = False
) -> torch.Tensor:
  """Use the compiled collider center for racket rewards, not policy inputs.

  Other source bodies/sites and robots without a racket collider retain their
  original semantics. Reference targets remain the dataset's target values.
  """
  original_pos = command.get_source_pos_w()
  original = command.get_source_lin_vel_w() if velocity else original_pos
  robot = getattr(command, "robot", None)
  if robot is None:
    return original
  if not hasattr(command, "_racket_reward_source_indices"):
    if (
      "racket_ball_collision" not in robot.geom_names
      or "racket_sweet_spot" not in robot.site_names
    ):
      command._racket_reward_source_indices = None
    else:
      command._racket_reward_source_indices = (
        robot.geom_names.index("racket_ball_collision"),
        robot.site_names.index("racket_sweet_spot"),
      )
  indices = command._racket_reward_source_indices
  if indices is None:
    return original
  geom_index, site_index = indices
  geom_id = robot.data.indexing.geom_ids[geom_index]
  center_w = robot.data.data.geom_xpos[:, geom_id]
  is_racket = (
    command._source_is_site_t[command.which_motion]
    & (command._source_body_indices_t[command.which_motion] == site_index)
  )
  if velocity:
    angular_velocity_w = robot.data.site_ang_vel_w[:, site_index, None, :]
    offset_w = center_w[:, None, :] - original_pos
    replacement = original + torch.linalg.cross(
      angular_velocity_w.expand_as(offset_w), offset_w, dim=-1
    )
  else:
    replacement = center_w[:, None, :].expand_as(original)
  return torch.where(is_racket[..., None], replacement, original)


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def action_term_rate_l2(
  env: ManagerBasedRlEnv,
  action_name: str,
) -> torch.Tensor:
  """Penalize changes in one action term without including Teacher-only actions."""
  start = 0
  for name in env.action_manager.active_terms:
    term = env.action_manager.get_term(name)
    stop = start + term.action_dim
    if name == action_name:
      delta = (
        env.action_manager.action[:, start:stop]
        - env.action_manager.prev_action[:, start:stop]
      )
      return torch.sum(torch.square(delta), dim=-1)
    start = stop
  raise KeyError(f"Unknown action term: {action_name!r}")


def _phase_and_subtarget_weights(
  command: MultiTargetMotionCommand,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return ``(phase, active, pos_weights, ori_weights, vel_weights)`` for the current step.

  *phase*       – (num_envs,) normalised motion phase in [0, 1]
  *active*      – (num_envs, max_subtargets) bool, True when sub-target's
                  phase window contains the current phase
  *pos_weights* – (num_envs, max_subtargets) per-subtarget position reward weights
  *ori_weights* – (num_envs, max_subtargets) per-subtarget orientation reward weights
  *vel_weights* – (num_envs, max_subtargets) per-subtarget velocity reward weights
  """
  which_motion = command.which_motion
  phase = command.normalized_phase  # (E,)
  phase_starts = command._target_phase_starts_t[which_motion]  # (E, S)
  phase_ends = command._target_phase_ends_t[which_motion]  # (E, S)
  active = (phase[:, None] >= phase_starts) & (phase[:, None] <= phase_ends)
  if isinstance(command, PhaseAwareMultiTargetMotionCommand):
    contact_active = command.time_remaining <= command.cfg.contact_reward_window_s
    active = active & contact_active[:, None]
  pos_weights = command._target_pos_reward_weights_t[which_motion]  # (E, S)
  ori_weights = command._target_ori_reward_weights_t[which_motion]  # (E, S)
  vel_weights = command._target_vel_reward_weights_t[which_motion]  # (E, S)
  return phase, active, pos_weights, ori_weights, vel_weights


def all_motions_target_position_error_exp(
  env: ManagerBasedRlEnv,
  target_command_name: str,
  std: float,
) -> torch.Tensor:
  """Phase-gated exponential reward for target position tracking.

  Each active sub-target contributes independently; rewards are summed over
  sub-targets weighted by per-subtarget ``pos_reward_weight``.
  """
  command = cast(
    MultiTargetMotionCommand,
    env.command_manager.get_term(target_command_name),
  )
  source_pos = _racket_reward_source_state(command)  # (E, S, 3)
  error = torch.sum(
    torch.square(command.target_position_w - source_pos), dim=-1
  )  # (E, S)
  reward = torch.exp(-error / std**2)  # (E, S)
  _, active, pos_weights, _, _ = _phase_and_subtarget_weights(command)

  return (reward * active.float() * pos_weights).sum(dim=-1)


def all_motions_target_orientation_axis_alignment_error_exp(
  env: ManagerBasedRlEnv,
  target_command_name: str,
  std: float,
  tolerance_degrees: float = 0.0,
) -> torch.Tensor:
  """Phase-gated exponential reward for per-subtarget axis orientation alignment.

  Each active sub-target contributes independently using its configured
  ``ori_axis``; rewards are summed over sub-targets weighted by ``ori_reward_weight``.
  """
  command = cast(
    MultiTargetMotionCommand,
    env.command_manager.get_term(target_command_name),
  )
  # axis_vecs: (E, S, 3) — per-subtarget axis vectors for this env's motion
  axis_vecs = command._target_ori_axes_t[command.which_motion]  # (E, S, 3)
  axis_vecs_flat = axis_vecs.reshape(-1, 3)  # (E*S, 3)
  # target_orientation_w / source_quat: (E, S, 4) → reshape to (E*S, 4)
  target_axis_w = quat_apply(
    command.target_orientation_w.reshape(-1, 4), axis_vecs_flat
  ).reshape(command.num_envs, command.max_subtargets, 3)
  source_axis_w = quat_apply(
    command.get_source_quat_w().reshape(-1, 4), axis_vecs_flat
  ).reshape(command.num_envs, command.max_subtargets, 3)

  reward = direction_alignment_reward_with_tolerance(
    source_axis_w,
    target_axis_w,
    std=std,
    tolerance_degrees=tolerance_degrees,
  )
  _, active, _, ori_weights, _ = _phase_and_subtarget_weights(command)
  return (reward * active.float() * ori_weights).sum(dim=-1)


def direction_alignment_reward_with_tolerance(
  source_vectors: torch.Tensor,
  target_vectors: torch.Tensor,
  *,
  std: float,
  tolerance_degrees: float,
) -> torch.Tensor:
  """Return full reward inside an angular cone and exponential decay outside it."""
  if std <= 0.0:
    raise ValueError("Direction reward std must be positive.")
  if not 0.0 <= tolerance_degrees < 180.0:
    raise ValueError("Direction tolerance must be in [0, 180) degrees.")

  source_norm = torch.linalg.vector_norm(source_vectors, dim=-1)
  target_norm = torch.linalg.vector_norm(target_vectors, dim=-1)
  valid = (source_norm > 1.0e-6) & (target_norm > 1.0e-6)
  denominator = (source_norm * target_norm).clamp(min=1.0e-6)
  dot = torch.sum(source_vectors * target_vectors, dim=-1) / denominator
  dot = dot.clamp(-1.0, 1.0)
  tolerance_error = 1.0 - math.cos(math.radians(tolerance_degrees))
  excess_error = (1.0 - dot - tolerance_error).clamp(min=0.0)
  reward = torch.exp(-torch.square(excess_error) / std**2)
  return torch.where(valid, reward, torch.zeros_like(reward))


def tennis_racket_incoming_velocity_alignment_score(
  source_axis_w: torch.Tensor,
  template_axis_w: torch.Tensor,
  incoming_ball_velocity_w: torch.Tensor,
  *,
  std: float,
  tolerance_degrees: float,
) -> torch.Tensor:
  """Align a racket normal with the signed incoming-ball velocity direction.

  The motion template selects which of the two parallel directions is valid. This
  preserves the forehand/backhand face-normal convention without classifying
  motions from filenames inside the reward.
  """
  desired_axis_w = signed_parallel_direction_from_template(
    template_axis_w, incoming_ball_velocity_w
  )
  return direction_alignment_reward_with_tolerance(
    source_axis_w,
    desired_axis_w,
    std=std,
    tolerance_degrees=tolerance_degrees,
  )


def tennis_racket_incoming_velocity_orientation_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  tolerance_degrees: float = 60.0,
  minimum_ball_speed: float = 0.5,
  ball_entity_name: str = "tennis_ball",
) -> torch.Tensor:
  """Reward the directed racket-face normal near the incoming-ball deadline."""
  if minimum_ball_speed < 0.0:
    raise ValueError("Minimum incoming-ball speed must be non-negative.")
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  axis_vecs = command._target_ori_axes_t[command.which_motion]
  flat_axes = axis_vecs.reshape(-1, 3)
  source_axis_w = quat_apply(
    command.get_source_quat_w().reshape(-1, 4), flat_axes
  ).reshape(command.num_envs, command.max_subtargets, 3)
  ball_velocity_w = env.scene[ball_entity_name].data.root_link_lin_vel_w
  target_axis_w = command.get_incoming_ball_racket_orientation_target_w(
    ball_velocity_w,
    minimum_ball_speed=minimum_ball_speed,
  )
  score = direction_alignment_reward_with_tolerance(
    source_axis_w,
    target_axis_w,
    std=std,
    tolerance_degrees=tolerance_degrees,
  )

  orientation_weights = command._target_ori_reward_weights_t[command.which_motion]
  orientation_active = orientation_weights > 0.0
  ball_speed = torch.linalg.vector_norm(ball_velocity_w, dim=-1)
  reward_active = incoming_ball_racket_orientation_active(
    command.strike_time_error_s[:, None],
    ball_speed[:, None],
    command.incoming_ball_racket_orientation_target_valid,
    is_paused=command.is_paused[:, None],
    window_before_s=command.cfg.racket_orientation_reward_window_before_s,
    window_after_s=command.cfg.racket_orientation_reward_window_after_s,
    minimum_ball_speed=minimum_ball_speed,
  )
  active = reward_active & orientation_active
  active_count = active.sum(dim=-1).clamp_min(1)
  return (score * active.to(score.dtype)).sum(dim=-1) / active_count


def tennis_reference_sweet_spot_velocity_score(
  source_velocity_w: torch.Tensor,
  reference_velocity_w: torch.Tensor,
  *,
  target_speed_scale: float = 0.75,
  minimum_reference_speed: float = 0.5,
) -> torch.Tensor:
  """Reward signed speed along each motion's reference sweet-point velocity."""
  if target_speed_scale <= 0.0:
    raise ValueError("Reference sweet-point target-speed scale must be positive.")
  if minimum_reference_speed < 0.0:
    raise ValueError("Minimum reference sweet-point speed must be non-negative.")

  reference_speed = torch.linalg.vector_norm(reference_velocity_w, dim=-1)
  reference_direction = (
    reference_velocity_w / reference_speed.clamp_min(1.0e-6)[..., None]
  )
  projected_speed = torch.sum(source_velocity_w * reference_direction, dim=-1)
  target_speed = target_speed_scale * reference_speed
  score = torch.clamp(projected_speed / target_speed.clamp_min(1.0e-6), 0.0, 1.0)
  return torch.where(
    reference_speed >= minimum_reference_speed,
    score,
    torch.zeros_like(score),
  )


def tennis_reference_sweet_spot_velocity_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  target_speed_scale: float = 0.75,
  contact_window_s: float = 0.1,
  proximity_std: float = 0.2,
  minimum_reference_speed: float = 0.5,
  minimum_ball_speed: float = 0.5,
  ball_entity_name: str = "tennis_ball",
  relative_to_pelvis: bool = False,
) -> torch.Tensor:
  """Reward a timely racket swing in the selected forehand/backhand direction."""
  if contact_window_s <= 0.0:
    raise ValueError("Sweet-point velocity contact window must be positive.")
  if proximity_std <= 0.0:
    raise ValueError("Sweet-point velocity proximity std must be positive.")
  if minimum_ball_speed < 0.0:
    raise ValueError("Minimum incoming-ball speed must be non-negative.")

  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  source_position_w = _racket_reward_source_state(command)
  source_velocity_w = _racket_reward_source_state(command, velocity=True)
  if relative_to_pelvis:
    source_velocity_w = point_velocity_relative_to_frame(
      source_position_w, source_velocity_w,
      command.robot_anchor_pos_w[:, None],
      command.robot_anchor_quat_w[:, None],
      command.robot_anchor_lin_vel_w[:, None],
      command.robot_anchor_ang_vel_w[:, None],
    )
    reference_velocity_w = command.get_reference_source_strike_relative_lin_vel_b()
  else:
    reference_velocity_w = command.get_reference_source_strike_lin_vel_w()
  velocity_score = tennis_reference_sweet_spot_velocity_score(
    source_velocity_w,
    reference_velocity_w,
    target_speed_scale=target_speed_scale,
    minimum_reference_speed=minimum_reference_speed,
  )

  ball = env.scene[ball_entity_name]
  ball_position_w = ball.data.root_link_pos_w
  ball_velocity_w = ball.data.root_link_lin_vel_w
  distance = torch.linalg.vector_norm(
    source_position_w - ball_position_w[:, None, :],
    dim=-1,
  )
  proximity_score = torch.exp(-torch.square(distance / proximity_std))
  position_mask = command._target_pos_reward_weights_t[command.which_motion] > 0.0
  time_active = tennis_strike_time_window_active(
    command.time_remaining,
    contact_window_s,
  )
  ball_speed = torch.linalg.vector_norm(ball_velocity_w, dim=-1)
  env_active = (
    time_active & (ball_speed >= minimum_ball_speed) & ~command.ball_has_been_struck
  )
  active = position_mask & env_active[:, None]
  active_count = active.sum(dim=-1).clamp_min(1)

  reference_speed = torch.linalg.vector_norm(reference_velocity_w, dim=-1)
  reference_direction = (
    reference_velocity_w / reference_speed.clamp_min(1.0e-6)[..., None]
  )
  projected_speed = torch.sum(source_velocity_w * reference_direction, dim=-1)
  command.metrics["reference_sweet_spot_speed"] = (reference_speed * position_mask).sum(
    dim=-1
  ) / position_mask.sum(dim=-1).clamp_min(1)
  command.metrics["racket_sweet_spot_projected_speed"] = (projected_speed * active).sum(
    dim=-1
  ) / active_count
  command.metrics["racket_sweet_spot_speed_reward_active"] = env_active.float()

  return (velocity_score * proximity_score * active.to(velocity_score.dtype)).sum(
    dim=-1
  ) / active_count


def all_motions_target_velocity_error_exp(
  env: ManagerBasedRlEnv,
  target_command_name: str,
  std: float,
  tolerance_degrees: float = 0.0,
) -> torch.Tensor:
  """Phase-gated exponential reward for target world-frame linear velocity.

  Each active sub-target contributes independently; rewards are summed over
  sub-targets weighted by per-subtarget ``vel_reward_weight``.
  """
  command = cast(
    MultiTargetMotionCommand,
    env.command_manager.get_term(target_command_name),
  )
  source_vel = _racket_reward_source_state(command, velocity=True)  # (E, S, 3)
  target_vel = command.target_velocity_w  # (E, S, 3) — sampled per episode
  reward = direction_alignment_reward_with_tolerance(
    source_vel,
    target_vel,
    std=std,
    tolerance_degrees=tolerance_degrees,
  )
  _, active, _, _, vel_weights = _phase_and_subtarget_weights(command)
  return (reward * active.float() * vel_weights).sum(dim=-1)


def all_motions_target_velocity_magnitude_error_exp(
  env: ManagerBasedRlEnv,
  target_command_name: str,
  std: float,
) -> torch.Tensor:
  """Phase-gated reward for matching target linear-speed magnitude."""
  if std <= 0.0:
    raise ValueError("Target velocity magnitude reward std must be positive.")
  command = cast(
    MultiTargetMotionCommand,
    env.command_manager.get_term(target_command_name),
  )
  source_speed = torch.linalg.vector_norm(
    _racket_reward_source_state(command, velocity=True), dim=-1
  )
  target_speed = torch.linalg.vector_norm(command.target_velocity_w, dim=-1)
  reward = torch.exp(-torch.square(source_speed - target_speed) / std**2)
  _, active, _, _, vel_weights = _phase_and_subtarget_weights(command)
  return (reward * active.float() * vel_weights).sum(dim=-1)


def tennis_racket_ball_distance_score(
  distance: torch.Tensor,
  strike_time_error_s: torch.Tensor,
  *,
  distance_std: float,
  time_std_s: float,
  window_half_width_s: float | None = None,
) -> torch.Tensor:
  """Score racket-ball proximity inside an optional hard strike-time window."""
  if distance_std <= 0.0:
    raise ValueError("Racket-ball distance reward std must be positive.")
  if time_std_s <= 0.0:
    raise ValueError("Racket-ball time reward std must be positive.")
  if window_half_width_s is not None and window_half_width_s <= 0.0:
    raise ValueError("Racket-ball window half-width must be positive.")
  distance_score = torch.exp(-torch.square(distance / distance_std))
  time_score = torch.exp(-torch.square(strike_time_error_s / time_std_s))
  score = distance_score * time_score
  if window_half_width_s is not None:
    score *= tennis_strike_time_window_active(strike_time_error_s, window_half_width_s)
  return score


def tennis_racket_ball_distance_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  distance_std: float,
  time_std_s: float,
  window_half_width_s: float | None = None,
  ball_entity_name: str = "tennis_ball",
  source_index: int = 0,
) -> torch.Tensor:
  """Densely bring the racket sweet point to the live ball near strike time."""

  command = env.command_manager.get_term(command_name)
  if not isinstance(command, PhaseAwareMultiTargetMotionCommand):
    raise TypeError(
      f"Command {command_name!r} must be phase-aware, got {type(command).__name__}."
    )
  sweet_spot_position_w = _racket_reward_source_state(command)[:, source_index]
  ball_position_w = env.scene[ball_entity_name].data.root_link_pos_w
  distance = torch.linalg.vector_norm(sweet_spot_position_w - ball_position_w, dim=-1)

  score = tennis_racket_ball_distance_score(
    distance,
    command.strike_time_error_s,
    distance_std=distance_std,
    time_std_s=time_std_s,
    window_half_width_s=window_half_width_s,
  )
  command.racket_ball_min_distance.copy_(
    torch.minimum(command.racket_ball_min_distance, distance)
  )
  command.racket_ball_score_max.copy_(
    torch.maximum(command.racket_ball_score_max, score)
  )
  abs_time_error = command.strike_time_error_s.abs()
  closer_to_deadline = abs_time_error < command.racket_ball_min_abs_time_error
  command.racket_ball_min_abs_time_error.copy_(
    torch.minimum(command.racket_ball_min_abs_time_error, abs_time_error)
  )
  command.racket_ball_distance_at_deadline.copy_(
    torch.where(
      closer_to_deadline,
      distance,
      command.racket_ball_distance_at_deadline,
    )
  )
  command.metrics["racket_ball_min_distance"].copy_(command.racket_ball_min_distance)
  command.metrics["racket_ball_score_max"].copy_(command.racket_ball_score_max)
  command.metrics["racket_ball_min_abs_time_error"].copy_(
    command.racket_ball_min_abs_time_error
  )
  command.metrics["racket_ball_distance_at_deadline"].copy_(
    command.racket_ball_distance_at_deadline
  )
  return score


def tennis_ball_first_bounce_target_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  ball_entity_name: str = "tennis_ball",
  ball_radius: float = 0.0335,
  target_radius: float = 0.5,
  strike_speed_threshold: float = 0.5,
  bounce_height_tolerance: float = 0.15,
) -> torch.Tensor:
  """Reward the physical ball's first post-strike bounce near its sampled target."""
  if std <= 0.0:
    raise ValueError("Tennis landing reward std must be positive.")
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  if not command.cfg.landing_target_enabled:
    raise RuntimeError("Tennis landing reward requires landing targets.")

  ball = env.scene[ball_entity_name]
  ball_position = ball.data.root_link_pos_w
  ball_velocity = ball.data.root_link_lin_vel_w
  horizontal_speed = torch.linalg.vector_norm(ball_velocity[:, :2], dim=-1)
  command.ball_has_been_struck |= horizontal_speed >= strike_speed_threshold

  vertical_velocity = ball_velocity[:, 2]
  bounced = (
    command.ball_has_been_struck
    & ~command.ball_landing_recorded
    & (command.ball_previous_vertical_velocity < -0.05)
    & (vertical_velocity >= 0.0)
    & (ball_position[:, 2] <= ball_radius + bounce_height_tolerance)
  )
  command.ball_previous_vertical_velocity.copy_(vertical_velocity)

  distance = torch.linalg.vector_norm(
    ball_position[:, :2] - command.target_landing_position_w[:, :2], dim=-1
  )
  outside_distance = (distance - target_radius).clamp(min=0.0)
  score = torch.exp(-torch.square(outside_distance) / std**2)

  command.ball_landing_recorded |= bounced
  command.ball_landing_position_w[:] = torch.where(
    bounced[:, None], ball_position, command.ball_landing_position_w
  )
  command.metrics["error_ball_landing"][:] = torch.where(
    bounced, distance, command.metrics["error_ball_landing"]
  )
  return torch.where(bounced, score, torch.zeros_like(score))


def ballistic_first_landing_position(
  position_w: torch.Tensor,
  velocity_w: torch.Tensor,
  *,
  gravity_magnitude: float,
  ground_contact_height: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Predict first ground contact for drag-free ballistic flight."""
  if gravity_magnitude <= 0.0:
    raise ValueError("Ballistic gravity magnitude must be positive.")
  if ground_contact_height < 0.0:
    raise ValueError("Ballistic ground contact height must be non-negative.")

  height = (position_w[:, 2] - ground_contact_height).clamp(min=0.0)
  vertical_velocity = velocity_w[:, 2]
  discriminant = torch.square(vertical_velocity) + 2.0 * gravity_magnitude * height
  flight_time = (
    vertical_velocity + torch.sqrt(discriminant.clamp(min=0.0))
  ) / gravity_magnitude
  flight_time = flight_time.clamp(min=0.0)

  landing_position_w = position_w + velocity_w * flight_time[:, None]
  landing_position_w[:, 2] = ground_contact_height
  return landing_position_w, flight_time


def tennis_net_clearance_score(
  height_at_net: torch.Tensor,
  valid_crossing: torch.Tensor,
  *,
  maximum_full_reward_height: float,
  excess_height_std: float,
) -> torch.Tensor:
  """Give full credit below a height cap, then decay excessive trajectories."""
  if maximum_full_reward_height <= 0.0:
    raise ValueError("Maximum full-reward net height must be positive.")
  if excess_height_std <= 0.0:
    raise ValueError("Net excess-height std must be positive.")
  excess_height = (height_at_net - maximum_full_reward_height).clamp(min=0.0)
  score = torch.exp(-torch.square(excess_height) / excess_height_std**2)
  return torch.where(valid_crossing, score, torch.zeros_like(score))


def tennis_ball_net_clearance_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Consume the one-shot net-clearance score produced by landing prediction."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  reward = command.ball_net_clearance_reward.clone()
  command.ball_net_clearance_reward.zero_()
  return reward


def tennis_ball_hit_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Consume the one-shot reward emitted by the physical strike detector."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  reward = command.ball_hit_reward.clone()
  command.ball_hit_reward.zero_()
  return reward


def tennis_ball_xy_direction_score(
  ball_velocity_xy: torch.Tensor,
  target_offset_xy: torch.Tensor,
  *,
  std: float,
) -> torch.Tensor:
  """Score horizontal ball velocity alignment toward the landing target."""
  return direction_alignment_reward_with_tolerance(
    ball_velocity_xy,
    target_offset_xy,
    std=std,
    tolerance_degrees=0.0,
  )


def tennis_ball_direction_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Consume the one-shot post-strike horizontal direction score."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  reward = command.ball_direction_reward.clone()
  command.ball_direction_reward.zero_()
  return reward


def tennis_ball_out_speed_score(
  speed: torch.Tensor,
  *,
  target_speed: float | torch.Tensor,
  std: float,
) -> torch.Tensor:
  """Reward reaching a target speed without penalizing faster outgoing balls."""
  if not isinstance(target_speed, torch.Tensor) and target_speed <= 0.0:
    raise ValueError("Target ball-out speed must be positive.")
  if std <= 0.0:
    raise ValueError("Ball-out speed reward std must be positive.")
  speed_deficit = (target_speed - speed).clamp(min=0.0)
  return torch.exp(-torch.square(speed_deficit) / std**2)


def tennis_ball_target_projected_speed(
  ball_velocity_xy: torch.Tensor,
  target_offset_xy: torch.Tensor,
) -> torch.Tensor:
  """Return positive horizontal ball speed toward the landing target."""
  target_distance = torch.linalg.vector_norm(target_offset_xy, dim=-1)
  target_direction = target_offset_xy / target_distance.clamp(min=1.0e-6)[:, None]
  projected_speed = torch.sum(ball_velocity_xy * target_direction, dim=-1)
  return torch.where(
    target_distance > 1.0e-6,
    projected_speed.clamp(min=0.0),
    torch.zeros_like(projected_speed),
  )


def tennis_ball_out_speed_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Consume the one-shot post-strike ball-speed score."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  reward = command.ball_out_speed_reward.clone()
  command.ball_out_speed_reward.zero_()
  return reward


def tennis_post_strike_heading_to_landing_target_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  recovery_delay_s: float = 0.3,
  tolerance_degrees: float = 15.0,
  std_degrees: float = 30.0,
) -> torch.Tensor:
  """Face the sampled landing target after a valid strike and follow-through."""
  if recovery_delay_s < 0.0:
    raise ValueError("Post-strike heading recovery delay must be non-negative.")
  if not 0.0 <= tolerance_degrees < 180.0:
    raise ValueError("Post-strike heading tolerance must lie in [0, 180).")
  if std_degrees <= 0.0:
    raise ValueError("Post-strike heading reward std must be positive.")

  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  if not command.cfg.landing_target_enabled:
    raise RuntimeError("Post-strike heading reward requires a landing target.")

  target_offset_xy = (
    command.target_landing_position_w[:, :2] - command.robot_anchor_pos_w[:, :2]
  )
  target_distance = torch.linalg.vector_norm(target_offset_xy, dim=-1)
  target_direction_xy = target_offset_xy / target_distance.clamp_min(1.0e-6)[:, None]

  local_forward = torch.zeros_like(command.robot_anchor_pos_w)
  local_forward[:, 0] = 1.0
  root_forward_xy = quat_apply(command.robot_anchor_quat_w, local_forward)[:, :2]
  root_forward_xy /= torch.linalg.vector_norm(root_forward_xy, dim=-1).clamp_min(
    1.0e-6
  )[:, None]

  cosine = torch.sum(root_forward_xy * target_direction_xy, dim=-1).clamp(-1.0, 1.0)
  angle_error = torch.acos(cosine)
  tolerance = math.radians(tolerance_degrees)
  std = math.radians(std_degrees)
  outside_tolerance = (angle_error - tolerance).clamp_min(0.0)
  score = torch.exp(-torch.square(outside_tolerance) / std**2)
  active = (
    command.ball_has_been_struck
    & (command.ball_post_strike_elapsed_s >= recovery_delay_s)
    & (target_distance > 1.0e-6)
  )
  command.metrics["post_strike_heading_error_degrees"][:] = torch.where(
    active,
    torch.rad2deg(angle_error),
    torch.zeros_like(angle_error),
  )
  command.metrics["post_strike_heading_active"][:] = active.to(score.dtype)
  return torch.where(active, score, torch.zeros_like(score))


def tennis_post_strike_heading_to_startup_x_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  recovery_delay_s: float = 0.3,
  tolerance_degrees: float = 15.0,
  std_degrees: float = 30.0,
) -> torch.Tensor:
  """Face the episode-start root frame's +X axis after follow-through."""
  if recovery_delay_s < 0.0:
    raise ValueError("Post-strike heading recovery delay must be non-negative.")
  if not 0.0 <= tolerance_degrees < 180.0:
    raise ValueError("Post-strike heading tolerance must lie in [0, 180).")
  if std_degrees <= 0.0:
    raise ValueError("Post-strike heading reward std must be positive.")

  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  local_forward = torch.zeros(
    command.robot_anchor_quat_w.shape[0],
    3,
    dtype=command.robot_anchor_quat_w.dtype,
    device=command.robot_anchor_quat_w.device,
  )
  local_forward[:, 0] = 1.0
  startup_forward_xy = quat_apply(command.startup_anchor_yaw_w, local_forward)[:, :2]
  root_forward_xy = quat_apply(yaw_quat(command.robot_anchor_quat_w), local_forward)[
    :, :2
  ]
  startup_forward_xy /= torch.linalg.vector_norm(startup_forward_xy, dim=-1).clamp_min(
    1.0e-6
  )[:, None]
  root_forward_xy /= torch.linalg.vector_norm(root_forward_xy, dim=-1).clamp_min(
    1.0e-6
  )[:, None]

  cosine = torch.sum(root_forward_xy * startup_forward_xy, dim=-1).clamp(-1.0, 1.0)
  angle_error = torch.acos(cosine)
  tolerance = math.radians(tolerance_degrees)
  std = math.radians(std_degrees)
  outside_tolerance = (angle_error - tolerance).clamp_min(0.0)
  score = torch.exp(-torch.square(outside_tolerance) / std**2)
  active = command.ball_has_been_struck & (
    command.ball_post_strike_elapsed_s >= recovery_delay_s
  )
  command.metrics["post_strike_heading_error_degrees"][:] = torch.where(
    active,
    torch.rad2deg(angle_error),
    torch.zeros_like(angle_error),
  )
  command.metrics["post_strike_heading_active"][:] = active.to(score.dtype)
  return torch.where(active, score, torch.zeros_like(score))


def tennis_ball_strike_event(
  ball_velocity_w: torch.Tensor,
  previous_ball_velocity_w: torch.Tensor,
  racket_offset_w: torch.Tensor,
  previous_racket_offset_w: torch.Tensor,
  state_initialized: torch.Tensor,
  *,
  speed_change_threshold: float,
  proximity_threshold: float,
) -> torch.Tensor:
  """Detect a racket-near horizontal velocity impulse over one control step."""
  if speed_change_threshold <= 0.0:
    raise ValueError("Strike speed-change threshold must be positive.")
  if proximity_threshold <= 0.0:
    raise ValueError("Strike proximity threshold must be positive.")

  horizontal_delta = torch.linalg.vector_norm(
    ball_velocity_w[:, :2] - previous_ball_velocity_w[:, :2], dim=-1
  )
  horizontal_speed = torch.linalg.vector_norm(ball_velocity_w[:, :2], dim=-1)
  offset_delta = racket_offset_w - previous_racket_offset_w
  segment_length_sq = torch.sum(torch.square(offset_delta), dim=-1)
  closest_fraction = (
    -torch.sum(previous_racket_offset_w * offset_delta, dim=-1)
    / segment_length_sq.clamp(min=1.0e-12)
  ).clamp(0.0, 1.0)
  closest_offset = previous_racket_offset_w + closest_fraction[:, None] * offset_delta
  closest_distance = torch.linalg.vector_norm(closest_offset, dim=-1)
  return (
    state_initialized
    & (horizontal_delta >= speed_change_threshold)
    & (horizontal_speed >= speed_change_threshold)
    & (closest_distance <= proximity_threshold)
  )


def tennis_strike_time_window_active(
  time_remaining: torch.Tensor,
  contact_window_s: float,
) -> torch.Tensor:
  """Return which environments are within the accepted strike-time window."""
  if contact_window_s < 0.0:
    raise ValueError("Contact window must be non-negative.")
  return torch.abs(time_remaining) <= contact_window_s


def tennis_ball_predicted_landing_target_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  ball_entity_name: str = "tennis_ball",
  ball_radius: float = 0.0335,
  target_radius: float = 0.5,
  strike_speed_change_threshold: float = 0.05,
  strike_proximity_threshold: float = 0.25,
  source_index: int = 0,
  prediction_delay_steps: int = 2,
  gravity_magnitude: float = 9.81,
  net_x: float = 5.6,
  net_height: float = 0.914,
  net_half_width: float = 5.485,
  maximum_full_reward_net_height: float = 2.0,
  excess_net_height_std: float = 0.25,
  target_out_speed: float = 10.0,
  out_speed_std: float = 10.0,
  ball_direction_std: float = 0.5,
  use_analytic_target_out_speed: bool = False,
  align_out_speed_to_landing_target: bool = False,
) -> torch.Tensor:
  """Reward a post-strike ballistic landing prediction without waiting for bounce."""
  if std <= 0.0:
    raise ValueError("Tennis landing reward std must be positive.")
  if prediction_delay_steps < 0:
    raise ValueError("Prediction delay steps must be non-negative.")
  if strike_speed_change_threshold <= 0.0:
    raise ValueError("Strike speed-change threshold must be positive.")
  if strike_proximity_threshold <= 0.0:
    raise ValueError("Strike proximity threshold must be positive.")
  if target_out_speed <= 0.0:
    raise ValueError("Target ball-out speed must be positive.")
  if out_speed_std <= 0.0:
    raise ValueError("Ball-out speed reward std must be positive.")
  if ball_direction_std <= 0.0:
    raise ValueError("Ball direction reward std must be positive.")
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  if not command.cfg.landing_target_enabled:
    raise RuntimeError("Tennis landing reward requires landing targets.")

  ball = env.scene[ball_entity_name]
  ball_position = ball.data.root_link_pos_w
  ball_velocity = ball.data.root_link_lin_vel_w
  sweet_spot_position_w = _racket_reward_source_state(command)[:, source_index]
  racket_offset_w = ball_position - sweet_spot_position_w
  strike_time_active = tennis_strike_time_window_active(
    command.time_remaining,
    command.cfg.contact_reward_window_s,
  )
  newly_struck = (
    ~command.ball_has_been_struck
    & strike_time_active
    & tennis_ball_strike_event(
      ball_velocity,
      command.ball_previous_linear_velocity_w,
      racket_offset_w,
      command.ball_previous_racket_offset_w,
      command.ball_strike_state_initialized,
      speed_change_threshold=strike_speed_change_threshold,
      proximity_threshold=strike_proximity_threshold,
    )
  )
  command.ball_previous_linear_velocity_w.copy_(ball_velocity)
  command.ball_previous_racket_offset_w.copy_(racket_offset_w)
  command.ball_strike_state_initialized.fill_(True)
  command.ball_has_been_struck |= newly_struck
  command.ball_hit_reward[:] = torch.where(
    newly_struck,
    torch.ones_like(command.ball_hit_reward),
    command.ball_hit_reward,
  )
  command.ball_post_strike_steps[:] = torch.where(
    newly_struck,
    torch.zeros_like(command.ball_post_strike_steps),
    command.ball_post_strike_steps,
  )

  pending = command.ball_has_been_struck & ~command.ball_landing_recorded
  ready = pending & (command.ball_post_strike_steps >= prediction_delay_steps)
  command.ball_post_strike_steps += (pending & ~ready).to(
    command.ball_post_strike_steps.dtype
  )

  predicted_landing_w, flight_time = ballistic_first_landing_position(
    ball_position,
    ball_velocity,
    gravity_magnitude=gravity_magnitude,
    ground_contact_height=ball_radius,
  )
  distance = torch.linalg.vector_norm(
    predicted_landing_w[:, :2] - command.target_landing_position_w[:, :2], dim=-1
  )
  outside_distance = (distance - target_radius).clamp(min=0.0)
  score = torch.exp(-torch.square(outside_distance) / std**2)

  env_origins = env.scene.env_origins
  net_x_w = env_origins[:, 0] + net_x
  net_y_w = env_origins[:, 1]
  horizontal_velocity_x = ball_velocity[:, 0]
  safe_velocity_x = torch.where(
    horizontal_velocity_x.abs() > 1.0e-6,
    horizontal_velocity_x,
    torch.ones_like(horizontal_velocity_x),
  )
  time_to_net = (net_x_w - ball_position[:, 0]) / safe_velocity_x
  crosses_net = (
    (horizontal_velocity_x.abs() > 1.0e-6)
    & (time_to_net >= 0.0)
    & (time_to_net <= flight_time)
  )
  height_at_net = (
    ball_position[:, 2]
    + ball_velocity[:, 2] * time_to_net
    - 0.5 * gravity_magnitude * torch.square(time_to_net)
  )
  lateral_position_at_net = ball_position[:, 1] + ball_velocity[:, 1] * time_to_net
  intersects_net_width = (
    lateral_position_at_net - net_y_w
  ).abs() <= net_half_width + ball_radius
  valid_net_crossing = (
    crosses_net & intersects_net_width & (height_at_net > net_height + ball_radius)
  )
  valid_prediction = ready & valid_net_crossing
  net_clearance_score = tennis_net_clearance_score(
    height_at_net,
    valid_net_crossing,
    maximum_full_reward_height=maximum_full_reward_net_height,
    excess_height_std=excess_net_height_std,
  )
  out_speed = torch.linalg.vector_norm(ball_velocity, dim=-1)
  target_offset_xy = command.target_landing_position_w[:, :2] - ball_position[:, :2]
  ball_direction_score = tennis_ball_xy_direction_score(
    ball_velocity[:, :2],
    target_offset_xy,
    std=ball_direction_std,
  )
  target_projected_out_speed = tennis_ball_target_projected_speed(
    ball_velocity[:, :2], target_offset_xy
  )
  rewarded_out_speed = (
    target_projected_out_speed if align_out_speed_to_landing_target else out_speed
  )
  speed_target = (
    command.target_ball_out_speed if use_analytic_target_out_speed else target_out_speed
  )
  out_speed_score = tennis_ball_out_speed_score(
    rewarded_out_speed,
    target_speed=speed_target,
    std=out_speed_std,
  )
  if align_out_speed_to_landing_target:
    out_speed_score *= ball_direction_score
  horizontal_speed = torch.linalg.vector_norm(ball_velocity[:, :2], dim=-1)
  horizontal_target_distance = torch.linalg.vector_norm(target_offset_xy, dim=-1)
  valid_direction = (horizontal_speed > 1.0e-6) & (horizontal_target_distance > 1.0e-6)
  direction_cosine = (
    torch.sum(ball_velocity[:, :2] * target_offset_xy, dim=-1)
    / (horizontal_speed * horizontal_target_distance).clamp(min=1.0e-6)
  ).clamp(-1.0, 1.0)
  direction_error_degrees = torch.rad2deg(torch.acos(direction_cosine))
  direction_error_degrees = torch.where(
    valid_direction,
    direction_error_degrees,
    torch.full_like(direction_error_degrees, 180.0),
  )

  command.ball_landing_recorded |= ready
  command.ball_landing_position_w[:] = torch.where(
    ready[:, None], predicted_landing_w, command.ball_landing_position_w
  )
  command.metrics["error_ball_landing"][:] = torch.where(
    ready, distance, command.metrics["error_ball_landing"]
  )
  command.metrics["ball_landing_prediction_valid"][:] = torch.where(
    ready,
    valid_net_crossing.to(command.metrics["ball_landing_prediction_valid"].dtype),
    command.metrics["ball_landing_prediction_valid"],
  )
  command.metrics["ball_net_crossing_height"][:] = torch.where(
    ready, height_at_net, command.metrics["ball_net_crossing_height"]
  )
  command.metrics["ball_out_speed"][:] = torch.where(
    ready, out_speed, command.metrics["ball_out_speed"]
  )
  command.metrics["ball_target_projected_speed"][:] = torch.where(
    ready,
    target_projected_out_speed,
    command.metrics["ball_target_projected_speed"],
  )
  command.metrics["ball_direction_error_degrees"][:] = torch.where(
    ready,
    direction_error_degrees,
    command.metrics["ball_direction_error_degrees"],
  )
  command.ball_direction_reward[:] = torch.where(
    ready, ball_direction_score, command.ball_direction_reward
  )
  command.ball_net_clearance_reward[:] = torch.where(
    ready, net_clearance_score, command.ball_net_clearance_reward
  )
  command.ball_out_speed_reward[:] = torch.where(
    ready, out_speed_score, command.ball_out_speed_reward
  )
  return torch.where(valid_prediction, score, torch.zeros_like(score))


def phase_timing_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Keep continuous phase on schedule to reach strike at contact time."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, PhaseAwareMultiTargetMotionCommand):
    raise TypeError(
      f"Command {command_name!r} must be phase-aware, got {type(command).__name__}."
    )
  active = (command.time_remaining > 0.0) | (command.phase < command.strike_phase)
  error = torch.square(command.phase - command.desired_strike_phase)
  reward = torch.exp(-error / std**2)
  return torch.where(active, reward, torch.ones_like(reward))


def phase_residual_l2(
  env: ManagerBasedRlEnv, action_name: str = "phase_rate"
) -> torch.Tensor:
  """Penalize unnecessary deviation from the analytic phase schedule."""
  action = env.action_manager.get_term(action_name).raw_action
  return torch.sum(torch.square(action), dim=-1)
