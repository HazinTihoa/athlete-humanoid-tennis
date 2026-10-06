from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from mjlab.utils.buffers import CircularBuffer
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_inv,
  subtract_frame_transforms,
  yaw_quat,
)

from .commands import MultiTargetMotionCommand
from .ball_delay import shared_gaussian_ball_delay
from .relative_velocity import point_velocity_relative_to_frame
from .phase_commands import (
  PhaseAccelerationMultiTargetMotionCommand,
  PhaseAwareMultiTargetMotionCommand,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def encode_tennis_ball_position_radial(
  position_b: torch.Tensor,
  *,
  linear_radius_m: float = 10.0,
  limit_radius_m: float = 20.0,
) -> torch.Tensor:
  """Keep ball coordinates linear through 10 m, then smoothly bound radius."""
  if not (0.0 < linear_radius_m < limit_radius_m):
    raise ValueError("Ball encoding requires 0 < linear radius < limit radius.")
  radius = torch.linalg.vector_norm(position_b, dim=-1, keepdim=True)
  width = limit_radius_m - linear_radius_m
  encoded_radius = linear_radius_m + width * torch.tanh(
    (radius - linear_radius_m).clamp_min(0.0) / width
  )
  scale = torch.where(
    radius <= linear_radius_m,
    torch.ones_like(radius),
    encoded_radius / radius.clamp_min(torch.finfo(position_b.dtype).eps),
  )
  return position_b * scale


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def motion_phase_state(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Current phase, remaining strike phase, and realized phase rate."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, PhaseAwareMultiTargetMotionCommand):
    raise TypeError(
      f"Command {command_name!r} must be phase-aware, got {type(command).__name__}."
    )
  return torch.stack(
    [command.phase, command.remaining_phase_to_hit, command.phase_rate], dim=-1
  )


def motion_phase_acceleration_state(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Phase state plus the deadline-required rate for learned scheduling."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, PhaseAccelerationMultiTargetMotionCommand):
    raise TypeError(
      f"Command {command_name!r} must use phase acceleration, "
      f"got {type(command).__name__}."
    )
  return torch.stack(
    [
      command.phase,
      command.remaining_phase_to_hit,
      command.phase_rate,
      command.required_phase_rate,
    ],
    dim=-1,
  )


def motion_task_goal(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Deployable target goal plus the episode's fixed sampled contact time.

  The base motion command starts with reference joint position and velocity.
  A phase-aware command appends the dynamic time remaining.  Neither belongs
  in the fixed Student goal, so retain only the target fields in between and
  append the sampled contact time explicitly.
  """
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, PhaseAwareMultiTargetMotionCommand):
    raise TypeError(
      f"Command {command_name!r} must be phase-aware, got {type(command).__name__}."
    )

  full_command = command.command
  reference_dim = 2 * command.joint_pos.shape[-1]
  target_goal = full_command[:, reference_dim:-1]
  return torch.cat([target_goal, command.contact_time[:, None]], dim=-1)


def motion_strike_target_position_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  target_index: int = 0,
) -> torch.Tensor:
  """Sampled strike position expressed in the current robot anchor frame."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiTargetMotionCommand):
    raise TypeError(
      f"Command {command_name!r} must be multi-target, got {type(command).__name__}."
    )
  if not 0 <= target_index < command.target_position_w.shape[1]:
    raise IndexError(
      f"Strike target index {target_index} is outside "
      f"[0, {command.target_position_w.shape[1]})."
    )
  return quat_apply(
    quat_inv(command.robot_anchor_quat_w),
    command.target_position_w[:, target_index] - command.robot_anchor_pos_w,
  )


def motion_reference_state(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Privileged reference joint position and velocity at the current phase."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  return torch.cat([command.joint_pos, command.joint_vel], dim=-1)


def motion_time_remaining(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Wall-clock seconds remaining until the sampled strike deadline."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, PhaseAwareMultiTargetMotionCommand):
    raise TypeError(
      f"Command {command_name!r} must be phase-aware, got {type(command).__name__}."
    )
  return command.time_remaining[:, None]


def robot_global_root_pos_w(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Current robot anchor position in world coordinates."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  return command.robot_anchor_pos_w


def robot_projected_gravity_b(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """World gravity direction expressed in the current robot anchor frame."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  return quat_apply(
    quat_inv(command.robot_anchor_quat_w),
    command.robot.data.gravity_vec_w,
  )


def tennis_startup_heading_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Episode-start +X heading expressed in the current robot yaw frame."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  local_forward = torch.zeros(
    command.robot_anchor_quat_w.shape[0],
    3,
    dtype=command.robot_anchor_quat_w.dtype,
    device=command.robot_anchor_quat_w.device,
  )
  local_forward[:, 0] = 1.0
  startup_forward_w = quat_apply(command.startup_anchor_yaw_w, local_forward)
  current_yaw_w = yaw_quat(command.robot_anchor_quat_w)
  startup_forward_b = quat_apply(quat_inv(current_yaw_w), startup_forward_w)
  return startup_forward_b[:, :2]


def tennis_landing_target_b(
  env: ManagerBasedRlEnv, command_name: str, current_root_frame: bool = False,
) -> torch.Tensor:
  """Sampled first-bounce target XY in its configured observation frame."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  if not command.cfg.landing_target_enabled:
    raise RuntimeError("Tennis landing target observation is not enabled.")
  if command.cfg.landing_target_frame == "startup" and not current_root_frame:
    return command.target_landing_position_startup[:, :2]
  offset_w = command.target_landing_position_w - command.robot_anchor_pos_w
  offset_b = quat_apply(quat_inv(command.robot_anchor_quat_w), offset_w)
  return offset_b[:, :2]


def tennis_ball_position_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  ball_entity_name: str = "tennis_ball",
) -> torch.Tensor:
  """Physical ball position relative to the current robot anchor frame."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  ball_position_w = env.scene[ball_entity_name].data.root_link_pos_w
  offset_w = ball_position_w - command.robot_anchor_pos_w
  return quat_apply(quat_inv(command.robot_anchor_quat_w), offset_w)


def tennis_ball_linear_velocity_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  ball_entity_name: str = "tennis_ball",
) -> torch.Tensor:
  """Physical ball world velocity expressed in the robot anchor orientation."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  ball_velocity_w = env.scene[ball_entity_name].data.root_link_lin_vel_w
  return quat_apply(quat_inv(command.robot_anchor_quat_w), ball_velocity_w)


def tennis_sweet_spot_position_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  source_index: int = 0,
) -> torch.Tensor:
  """Racket collider-center site position in the current robot anchor frame."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  source_position_w = command.get_source_pos_w()[:, source_index]
  offset_w = source_position_w - command.robot_anchor_pos_w
  return quat_apply(quat_inv(command.robot_anchor_quat_w), offset_w)


def gather_strided_history(
  buffer: CircularBuffer, sample_lags: tuple[int, ...]
) -> torch.Tensor:
  """Gather chronological strided samples and flatten their feature dimensions."""
  return torch.cat([buffer[lag] for lag in sample_lags], dim=-1)


class _TennisBallPerceptionState:
  """Shared delayed/held ball measurements for every Student observation term."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    *,
    ball_entity_name: str,
    latency_min_steps: int,
    latency_max_steps: int,
    dropout_start_probability: float,
    dropout_duration_min_steps: int,
    dropout_duration_max_steps: int,
    dropout_duration_modes: tuple[tuple[float, int, int], ...],
    position_noise_std: float,
    velocity_noise_std: float,
    control_dt_s: float,
  ) -> None:
    if latency_min_steps < 0 or latency_max_steps < latency_min_steps:
      raise ValueError("Ball latency steps must form a non-negative ordered range.")
    if not 0.0 <= dropout_start_probability <= 1.0:
      raise ValueError("Ball dropout start probability must be in [0, 1].")
    if (
      dropout_duration_min_steps < 1
      or dropout_duration_max_steps < dropout_duration_min_steps
    ):
      raise ValueError("Ball dropout duration must be a positive ordered range.")
    if dropout_duration_modes:
      if abs(sum(mode[0] for mode in dropout_duration_modes) - 1.0) > 1e-6:
        raise ValueError("Ball dropout duration mode probabilities must sum to 1.")
      for probability, minimum, maximum in dropout_duration_modes:
        if probability <= 0.0 or minimum < 1 or maximum < minimum:
          raise ValueError("Ball dropout duration modes must have valid weights and ranges.")
    if position_noise_std < 0.0 or velocity_noise_std < 0.0:
      raise ValueError(
        "Ball perception noise standard deviations must be non-negative."
      )
    if control_dt_s <= 0.0:
      raise ValueError("Ball perception control_dt_s must be positive.")

    self.signature = (
      ball_entity_name,
      latency_min_steps,
      latency_max_steps,
      dropout_start_probability,
      dropout_duration_min_steps,
      dropout_duration_max_steps,
      dropout_duration_modes,
      position_noise_std,
      velocity_noise_std,
      control_dt_s,
    )
    self.ball_entity_name = ball_entity_name
    self.latency_min_steps = latency_min_steps
    self.latency_max_steps = latency_max_steps
    self.dropout_start_probability = dropout_start_probability
    self.dropout_duration_min_steps = dropout_duration_min_steps
    self.dropout_duration_max_steps = dropout_duration_max_steps
    self.dropout_duration_modes = dropout_duration_modes
    self.position_noise_std = position_noise_std
    self.velocity_noise_std = velocity_noise_std
    self.control_dt_s = control_dt_s
    raw_buffer_length = latency_max_steps + 1
    self.raw_position_w = CircularBuffer(
      max_len=raw_buffer_length,
      batch_size=env.num_envs,
      device=env.device,
    )
    self.raw_velocity_w = CircularBuffer(
      max_len=raw_buffer_length,
      batch_size=env.num_envs,
      device=env.device,
    )
    self.perceived_position_w = torch.zeros(env.num_envs, 3, device=env.device)
    self.perceived_velocity_w = torch.zeros(env.num_envs, 3, device=env.device)
    self.valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    self.age_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.dropout_remaining = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )
    self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._last_step: int | None = None
    self._needs_update_after_reset = True

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    indices = slice(None) if env_ids is None else env_ids
    self.raw_position_w.reset(batch_ids=env_ids)
    self.raw_velocity_w.reset(batch_ids=env_ids)
    self.valid[indices] = True
    self.age_steps[indices] = 0
    self.dropout_remaining[indices] = 0
    self.initialized[indices] = False
    self._needs_update_after_reset = True

  def update(
    self, env: ManagerBasedRlEnv
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_step = int(env.common_step_counter)
    if self._last_step == current_step and not self._needs_update_after_reset:
      return (
        self.perceived_position_w,
        self.perceived_velocity_w,
        self.valid,
        self.age_steps.float() * self.control_dt_s,
      )

    ball = env.scene[self.ball_entity_name]
    self.raw_position_w.append(ball.data.root_link_pos_w)
    self.raw_velocity_w.append(ball.data.root_link_lin_vel_w)
    delay_steps = torch.randint(
      self.latency_min_steps,
      self.latency_max_steps + 1,
      (env.num_envs,),
      device=env.device,
    )
    measured_position_w = self.raw_position_w[delay_steps]
    measured_velocity_w = self.raw_velocity_w[delay_steps]
    if self.position_noise_std > 0.0:
      measured_position_w = (
        measured_position_w
        + torch.randn_like(measured_position_w) * self.position_noise_std
      )
    if self.velocity_noise_std > 0.0:
      measured_velocity_w = (
        measured_velocity_w
        + torch.randn_like(measured_velocity_w) * self.velocity_noise_std
      )

    can_start_dropout = self.dropout_remaining == 0
    if self.dropout_start_probability > 0.0:
      starts_dropout = can_start_dropout & (
        torch.rand(env.num_envs, device=env.device) < self.dropout_start_probability
      )
      if self.dropout_duration_modes:
        mode_draw = torch.rand(env.num_envs, device=env.device)
        durations = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        lower_probability = 0.0
        for probability, minimum, maximum in self.dropout_duration_modes:
          selected_mode = (mode_draw >= lower_probability) & (
            mode_draw < lower_probability + probability
          )
          sampled_duration = torch.randint(
            minimum,
            maximum + 1,
            (env.num_envs,),
            device=env.device,
          )
          durations = torch.where(selected_mode, sampled_duration, durations)
          lower_probability += probability
      else:
        durations = torch.randint(
          self.dropout_duration_min_steps,
          self.dropout_duration_max_steps + 1,
          (env.num_envs,),
          device=env.device,
        )
      self.dropout_remaining = torch.where(
        starts_dropout, durations, self.dropout_remaining
      )

    dropped = self.dropout_remaining > 0
    refresh = ~dropped | ~self.initialized
    self.perceived_position_w = torch.where(
      refresh[:, None], measured_position_w, self.perceived_position_w
    )
    self.perceived_velocity_w = torch.where(
      refresh[:, None], measured_velocity_w, self.perceived_velocity_w
    )
    self.age_steps = torch.where(
      dropped & self.initialized,
      self.age_steps + 1,
      delay_steps,
    )
    self.valid = ~dropped
    self.initialized[:] = True
    self.dropout_remaining = torch.clamp(self.dropout_remaining - 1, min=0)
    self._last_step = current_step
    self._needs_update_after_reset = False
    return (
      self.perceived_position_w,
      self.perceived_velocity_w,
      self.valid,
      self.age_steps.float() * self.control_dt_s,
    )


def _shared_tennis_ball_perception(
  env: ManagerBasedRlEnv,
  *,
  ball_entity_name: str,
  perception_latency_min_steps: int,
  perception_latency_max_steps: int,
  perception_dropout_start_probability: float,
  perception_dropout_duration_min_steps: int,
  perception_dropout_duration_max_steps: int,
  perception_dropout_duration_modes: tuple[tuple[float, int, int], ...],
  perception_position_noise_std: float,
  perception_velocity_noise_std: float,
  perception_control_dt_s: float,
) -> _TennisBallPerceptionState:
  signature = (
    ball_entity_name,
    perception_latency_min_steps,
    perception_latency_max_steps,
    perception_dropout_start_probability,
    perception_dropout_duration_min_steps,
    perception_dropout_duration_max_steps,
    perception_dropout_duration_modes,
    perception_position_noise_std,
    perception_velocity_noise_std,
    perception_control_dt_s,
  )
  state = getattr(env, "_athlete_tennis_ball_perception", None)
  if state is None:
    state = _TennisBallPerceptionState(
      env,
      ball_entity_name=ball_entity_name,
      latency_min_steps=perception_latency_min_steps,
      latency_max_steps=perception_latency_max_steps,
      dropout_start_probability=perception_dropout_start_probability,
      dropout_duration_min_steps=perception_dropout_duration_min_steps,
      dropout_duration_max_steps=perception_dropout_duration_max_steps,
      dropout_duration_modes=perception_dropout_duration_modes,
      position_noise_std=perception_position_noise_std,
      velocity_noise_std=perception_velocity_noise_std,
      control_dt_s=perception_control_dt_s,
    )
    setattr(env, "_athlete_tennis_ball_perception", state)
  elif state.signature != signature:
    raise ValueError("All robust ball observation terms must share one configuration.")
  return cast(_TennisBallPerceptionState, state)


class TennisBallStridedHistory(ManagerTermBase):
  """Keep a dense ball-state buffer and expose selected historical frames."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    sample_lags = tuple(int(lag) for lag in cfg.params["sample_lags"])
    buffer_length = int(cfg.params["buffer_length"])
    quantity = str(cfg.params["quantity"])
    if not sample_lags or any(lag < 0 for lag in sample_lags):
      raise ValueError("sample_lags must contain non-negative integers.")
    if any(first <= second for first, second in pairwise(sample_lags)):
      raise ValueError("sample_lags must be strictly descending (oldest to newest).")
    if buffer_length <= max(sample_lags):
      raise ValueError("buffer_length must be larger than every sample lag.")
    if quantity not in {"position", "linear_velocity"}:
      raise ValueError("quantity must be 'position' or 'linear_velocity'.")

    self.sample_lags = sample_lags
    self.quantity = quantity
    self.perception = shared_gaussian_ball_delay(cfg.params, env)
    if bool(cfg.params.get("use_shared_perception", False)):
      self.perception = _shared_tennis_ball_perception(
        env,
        ball_entity_name=str(cfg.params["ball_entity_name"]),
        perception_latency_min_steps=int(cfg.params["perception_latency_min_steps"]),
        perception_latency_max_steps=int(cfg.params["perception_latency_max_steps"]),
        perception_dropout_start_probability=float(
          cfg.params["perception_dropout_start_probability"]
        ),
        perception_dropout_duration_min_steps=int(
          cfg.params["perception_dropout_duration_min_steps"]
        ),
        perception_dropout_duration_max_steps=int(
          cfg.params["perception_dropout_duration_max_steps"]
        ),
        perception_dropout_duration_modes=tuple(
          tuple(mode)
          for mode in cfg.params.get("perception_dropout_duration_modes", ())
        ),
        perception_position_noise_std=float(
          cfg.params["perception_position_noise_std"]
        ),
        perception_velocity_noise_std=float(
          cfg.params["perception_velocity_noise_std"]
        ),
        perception_control_dt_s=float(cfg.params["perception_control_dt_s"]),
      )
    self.buffer = CircularBuffer(
      max_len=buffer_length,
      batch_size=env.num_envs,
      device=env.device,
    )
    self._last_step: int | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.buffer.reset(batch_ids=env_ids)
    if self.perception is not None:
      self.perception.reset(env_ids)
    self._last_step = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    ball_entity_name: str,
    quantity: str,
    sample_lags: tuple[int, ...],
    buffer_length: int,
    use_shared_perception: bool = False,
    perception_latency_min_steps: int = 0,
    perception_latency_max_steps: int = 0,
    perception_dropout_start_probability: float = 0.0,
    perception_dropout_duration_min_steps: int = 1,
    perception_dropout_duration_max_steps: int = 1,
    perception_dropout_duration_modes: tuple[tuple[float, int, int], ...] = (),
    perception_position_noise_std: float = 0.0,
    perception_velocity_noise_std: float = 0.0,
    perception_control_dt_s: float = 0.02,
    delay_mean_s: float | None = None,
    delay_std_s: float = 0.01,
    delay_max_s: float = 0.1,
    delay_min_s: float = 0.0,
    measurement_perception: dict | None = None,
    position_encoding: dict | None = None,
  ) -> torch.Tensor:
    del (
      quantity,
      sample_lags,
      buffer_length,
      use_shared_perception,
      perception_latency_min_steps,
      perception_latency_max_steps,
      perception_dropout_start_probability,
      perception_dropout_duration_min_steps,
      perception_dropout_duration_max_steps,
      perception_dropout_duration_modes,
      perception_position_noise_std,
      perception_velocity_noise_std,
      perception_control_dt_s,
      delay_mean_s, delay_std_s, delay_max_s, delay_min_s, measurement_perception,
    )
    current_step = int(env.common_step_counter)
    if self._last_step != current_step or not self.buffer.is_initialized:
      if self.perception is not None:
        position_w, velocity_w, _, _ = self.perception.update(env)
        command = cast(
          MultiTargetMotionCommand, env.command_manager.get_term(command_name)
        )
        value_w = position_w if self.quantity == "position" else velocity_w
        if self.quantity == "position":
          value_w = value_w - command.robot_anchor_pos_w
        value = quat_apply(quat_inv(command.robot_anchor_quat_w), value_w)
      elif self.quantity == "position":
        value = tennis_ball_position_b(env, command_name, ball_entity_name)
      else:
        value = tennis_ball_linear_velocity_b(env, command_name, ball_entity_name)
      if self.quantity == "position" and position_encoding is not None:
        value = encode_tennis_ball_position_radial(
          value,
          linear_radius_m=float(position_encoding["linear_radius_m"]),
          limit_radius_m=float(position_encoding["limit_radius_m"]),
        )
      self.buffer.append(value)
      self._last_step = current_step
    return gather_strided_history(self.buffer, self.sample_lags)


class TennisBallCurrentAnchorStridedHistory(ManagerTermBase):
  """Expose ball history in the *current* robot anchor frame.

  ``TennisBallStridedHistory`` stores each sample after converting it to the
  anchor frame at that sample time. That is suitable for the existing MLP
  observations, but a temporal model would otherwise receive tokens expressed
  in different frames after the robot turns. This term stores world values and
  converts every selected sample with the current anchor on readout.
  """

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    sample_lags = tuple(int(lag) for lag in cfg.params["sample_lags"])
    buffer_length = int(cfg.params["buffer_length"])
    quantity = str(cfg.params["quantity"])
    if not sample_lags or any(lag < 0 for lag in sample_lags):
      raise ValueError("sample_lags must contain non-negative integers.")
    if any(first <= second for first, second in pairwise(sample_lags)):
      raise ValueError("sample_lags must be strictly descending (oldest to newest).")
    if buffer_length <= max(sample_lags):
      raise ValueError("buffer_length must be larger than every sample lag.")
    if quantity not in {"position", "linear_velocity"}:
      raise ValueError("quantity must be 'position' or 'linear_velocity'.")

    self.sample_lags = sample_lags
    self.quantity = quantity
    self.perception = shared_gaussian_ball_delay(cfg.params, env)
    if bool(cfg.params.get("use_shared_perception", False)):
      self.perception = _shared_tennis_ball_perception(
        env,
        ball_entity_name=str(cfg.params["ball_entity_name"]),
        perception_latency_min_steps=int(cfg.params["perception_latency_min_steps"]),
        perception_latency_max_steps=int(cfg.params["perception_latency_max_steps"]),
        perception_dropout_start_probability=float(
          cfg.params["perception_dropout_start_probability"]
        ),
        perception_dropout_duration_min_steps=int(
          cfg.params["perception_dropout_duration_min_steps"]
        ),
        perception_dropout_duration_max_steps=int(
          cfg.params["perception_dropout_duration_max_steps"]
        ),
        perception_dropout_duration_modes=tuple(
          tuple(mode)
          for mode in cfg.params.get("perception_dropout_duration_modes", ())
        ),
        perception_position_noise_std=float(
          cfg.params["perception_position_noise_std"]
        ),
        perception_velocity_noise_std=float(
          cfg.params["perception_velocity_noise_std"]
        ),
        perception_control_dt_s=float(cfg.params["perception_control_dt_s"]),
      )
    self.buffer = CircularBuffer(
      max_len=buffer_length,
      batch_size=env.num_envs,
      device=env.device,
    )
    self._last_step: int | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.buffer.reset(batch_ids=env_ids)
    if self.perception is not None:
      self.perception.reset(env_ids)
    self._last_step = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    ball_entity_name: str,
    quantity: str,
    sample_lags: tuple[int, ...],
    buffer_length: int,
    use_shared_perception: bool = False,
    perception_latency_min_steps: int = 0,
    perception_latency_max_steps: int = 0,
    perception_dropout_start_probability: float = 0.0,
    perception_dropout_duration_min_steps: int = 1,
    perception_dropout_duration_max_steps: int = 1,
    perception_dropout_duration_modes: tuple[tuple[float, int, int], ...] = (),
    perception_position_noise_std: float = 0.0,
    perception_velocity_noise_std: float = 0.0,
    perception_control_dt_s: float = 0.02,
    delay_mean_s: float | None = None,
    delay_std_s: float = 0.01,
    delay_max_s: float = 0.1,
    delay_min_s: float = 0.0,
    measurement_perception: dict | None = None,
    position_encoding: dict | None = None,
  ) -> torch.Tensor:
    del (
      quantity,
      sample_lags,
      buffer_length,
      use_shared_perception,
      perception_latency_min_steps,
      perception_latency_max_steps,
      perception_dropout_start_probability,
      perception_dropout_duration_min_steps,
      perception_dropout_duration_max_steps,
      perception_dropout_duration_modes,
      perception_position_noise_std,
      perception_velocity_noise_std,
      perception_control_dt_s,
      delay_mean_s, delay_std_s, delay_max_s, delay_min_s, measurement_perception,
    )
    current_step = int(env.common_step_counter)
    if self._last_step != current_step or not self.buffer.is_initialized:
      if self.perception is not None:
        position_w, velocity_w, _, _ = self.perception.update(env)
        value = position_w if self.quantity == "position" else velocity_w
      else:
        ball = env.scene[ball_entity_name]
        value = (
          ball.data.root_link_pos_w
          if self.quantity == "position"
          else ball.data.root_link_lin_vel_w
        )
      self.buffer.append(value)
      self._last_step = current_step

    command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
    values_w = torch.stack([self.buffer[lag] for lag in self.sample_lags], dim=1)
    num_envs, history_length, _ = values_w.shape
    anchor_quat_w = command.robot_anchor_quat_w[:, None, :].expand(
      -1, history_length, -1
    )
    if self.quantity == "position":
      values_w = values_w - command.robot_anchor_pos_w[:, None, :]
    values_b = quat_apply(
      quat_inv(anchor_quat_w.reshape(-1, 4)), values_w.reshape(-1, 3)
    )
    if self.quantity == "position" and position_encoding is not None:
      values_b = encode_tennis_ball_position_radial(
        values_b,
        linear_radius_m=float(position_encoding["linear_radius_m"]),
        limit_radius_m=float(position_encoding["limit_radius_m"]),
      )
    return values_b.reshape(num_envs, -1)


class TennisBallObservationStatusHistory(ManagerTermBase):
  """Expose per-token ball visibility or observation age for robust z_S."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    self.sample_lags = tuple(int(lag) for lag in cfg.params["sample_lags"])
    buffer_length = int(cfg.params["buffer_length"])
    self.quantity = str(cfg.params["quantity"])
    if self.quantity not in {"valid", "age"}:
      raise ValueError("Ball observation status quantity must be 'valid' or 'age'.")
    if buffer_length <= max(self.sample_lags):
      raise ValueError("buffer_length must be larger than every sample lag.")
    self.perception = _shared_tennis_ball_perception(
      env,
      ball_entity_name=str(cfg.params["ball_entity_name"]),
      perception_latency_min_steps=int(cfg.params["perception_latency_min_steps"]),
      perception_latency_max_steps=int(cfg.params["perception_latency_max_steps"]),
      perception_dropout_start_probability=float(
        cfg.params["perception_dropout_start_probability"]
      ),
      perception_dropout_duration_min_steps=int(
        cfg.params["perception_dropout_duration_min_steps"]
      ),
      perception_dropout_duration_max_steps=int(
        cfg.params["perception_dropout_duration_max_steps"]
      ),
      perception_dropout_duration_modes=tuple(
        tuple(mode)
        for mode in cfg.params.get("perception_dropout_duration_modes", ())
      ),
      perception_position_noise_std=float(cfg.params["perception_position_noise_std"]),
      perception_velocity_noise_std=float(cfg.params["perception_velocity_noise_std"]),
      perception_control_dt_s=float(cfg.params["perception_control_dt_s"]),
    )
    self.buffer = CircularBuffer(
      max_len=buffer_length,
      batch_size=env.num_envs,
      device=env.device,
    )
    self._last_step: int | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.buffer.reset(batch_ids=env_ids)
    self.perception.reset(env_ids)
    self._last_step = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    ball_entity_name: str,
    quantity: str,
    sample_lags: tuple[int, ...],
    buffer_length: int,
    use_shared_perception: bool,
    perception_latency_min_steps: int,
    perception_latency_max_steps: int,
    perception_dropout_start_probability: float,
    perception_dropout_duration_min_steps: int,
    perception_dropout_duration_max_steps: int,
    perception_dropout_duration_modes: tuple[tuple[float, int, int], ...],
    perception_position_noise_std: float,
    perception_velocity_noise_std: float,
    perception_control_dt_s: float,
  ) -> torch.Tensor:
    del (
      ball_entity_name,
      quantity,
      sample_lags,
      buffer_length,
      use_shared_perception,
      perception_latency_min_steps,
      perception_latency_max_steps,
      perception_dropout_start_probability,
      perception_dropout_duration_min_steps,
      perception_dropout_duration_max_steps,
      perception_dropout_duration_modes,
      perception_position_noise_std,
      perception_velocity_noise_std,
      perception_control_dt_s,
    )
    current_step = int(env.common_step_counter)
    if self._last_step != current_step or not self.buffer.is_initialized:
      _, _, valid, age_s = self.perception.update(env)
      value = valid.float() if self.quantity == "valid" else age_s
      self.buffer.append(value[:, None])
      self._last_step = current_step
    return gather_strided_history(self.buffer, self.sample_lags)


class TennisStudentStateStridedHistory(ManagerTermBase):
  """Keep noisy deployable proprioception as chronological state tokens."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    sample_lags = tuple(int(lag) for lag in cfg.params["sample_lags"])
    buffer_length = int(cfg.params["buffer_length"])
    if not sample_lags or any(lag < 0 for lag in sample_lags):
      raise ValueError("sample_lags must contain non-negative integers.")
    if any(first <= second for first, second in pairwise(sample_lags)):
      raise ValueError("sample_lags must be strictly descending (oldest to newest).")
    if buffer_length <= max(sample_lags):
      raise ValueError("buffer_length must be larger than every sample lag.")

    self.sample_lags = sample_lags
    self.apply_observation_noise = bool(
      cfg.params.get("apply_observation_noise", False)
    )
    self.base_ang_vel_noise = tuple(
      float(value) for value in cfg.params.get("base_ang_vel_noise", (0.0, 0.0))
    )
    self.projected_gravity_noise = tuple(
      float(value) for value in cfg.params.get("projected_gravity_noise", (0.0, 0.0))
    )
    self.joint_pos_noise = tuple(
      float(value) for value in cfg.params.get("joint_pos_noise", (0.0, 0.0))
    )
    self.joint_vel_noise = tuple(
      float(value) for value in cfg.params.get("joint_vel_noise", (0.0, 0.0))
    )
    for name, noise_range in (
      ("base_ang_vel_noise", self.base_ang_vel_noise),
      ("projected_gravity_noise", self.projected_gravity_noise),
      ("joint_pos_noise", self.joint_pos_noise),
      ("joint_vel_noise", self.joint_vel_noise),
    ):
      if len(noise_range) != 2 or noise_range[0] > noise_range[1]:
        raise ValueError(f"{name} must be an ordered (min, max) pair.")
    self.buffer = CircularBuffer(
      max_len=buffer_length,
      batch_size=env.num_envs,
      device=env.device,
    )
    self._last_step: int | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.buffer.reset(batch_ids=env_ids)
    self._last_step = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    action_name: str,
    sensor_name: str,
    sample_lags: tuple[int, ...],
    buffer_length: int,
    apply_observation_noise: bool = False,
    base_ang_vel_noise: tuple[float, float] = (0.0, 0.0),
    projected_gravity_noise: tuple[float, float] = (0.0, 0.0),
    joint_pos_noise: tuple[float, float] = (0.0, 0.0),
    joint_vel_noise: tuple[float, float] = (0.0, 0.0),
  ) -> torch.Tensor:
    del (
      sample_lags,
      buffer_length,
      apply_observation_noise,
      base_ang_vel_noise,
      projected_gravity_noise,
      joint_pos_noise,
      joint_vel_noise,
    )
    current_step = int(env.common_step_counter)
    if self._last_step != current_step or not self.buffer.is_initialized:
      command = cast(
        MultiTargetMotionCommand, env.command_manager.get_term(command_name)
      )
      robot = command.robot
      joint_pos = robot.data.joint_pos_biased - robot.data.default_joint_pos
      joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
      value = torch.cat(
        [
          env.scene[sensor_name].data,
          robot_projected_gravity_b(env, command_name),
          joint_pos,
          joint_vel,
          env.action_manager.get_term(action_name).raw_action,
        ],
        dim=-1,
      )
      self.buffer.append(value)
      self._last_step = current_step
    history = gather_strided_history(self.buffer, self.sample_lags)
    if not self.apply_observation_noise:
      return history

    tokens = history.reshape(-1, len(self.sample_lags), 93)
    # Match the Student observation corruption without storing noisy values in
    # the temporal buffer. Each read therefore represents a fresh sensor sample.
    for start, end, noise_range in (
      (0, 3, self.base_ang_vel_noise),
      (3, 6, self.projected_gravity_noise),
      (6, 35, self.joint_pos_noise),
      (35, 64, self.joint_vel_noise),
    ):
      low, high = noise_range
      if low != 0.0 or high != 0.0:
        tokens[..., start:end].add_(
          torch.empty_like(tokens[..., start:end]).uniform_(low, high)
        )
    return tokens.reshape(history.shape[0], -1)


def motion_future_reference_window(
  env: ManagerBasedRlEnv,
  command_name: str,
  future_offsets: tuple[int, ...],
) -> torch.Tensor:
  """Flatten future reference joint position and velocity tokens for Teacher intent."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  offsets = torch.as_tensor(future_offsets, dtype=torch.long, device=command.device)
  if offsets.ndim != 1 or offsets.numel() == 0 or torch.any(offsets <= 0):
    raise ValueError("future_offsets must be a non-empty tuple of positive steps.")
  if torch.any(offsets[1:] <= offsets[:-1]):
    raise ValueError("future_offsets must be strictly ascending.")

  current_steps = command._clamped_time_steps()[:, None]
  max_steps = command._time_step_totals[command.which_motion, None] - 1
  future_steps = torch.minimum(current_steps + offsets[None, :], max_steps)
  motion_ids = command.which_motion[:, None]
  joint_pos = command._stacked_joint_pos[motion_ids, future_steps]
  joint_vel = command._stacked_joint_vel[motion_ids, future_steps]
  return torch.cat([joint_pos, joint_vel], dim=-1).reshape(env.num_envs, -1)


def tennis_sweet_spot_linear_velocity_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  source_index: int = 0,
) -> torch.Tensor:
  """Racket collider-center site velocity in the robot anchor orientation."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  source_velocity_w = command.get_source_lin_vel_w()[:, source_index]
  return quat_apply(quat_inv(command.robot_anchor_quat_w), source_velocity_w)


def tennis_sweet_spot_relative_linear_velocity_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  source_index: int = 0,
) -> torch.Tensor:
  """Simulator point velocity relative to Pelvis, without differencing or EMA."""
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  return point_velocity_relative_to_frame(
    command.get_source_pos_w()[:, source_index],
    command.get_source_lin_vel_w()[:, source_index],
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.robot_anchor_lin_vel_w,
    command.robot_anchor_ang_vel_w,
  )
