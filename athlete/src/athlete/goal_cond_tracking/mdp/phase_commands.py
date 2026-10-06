from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from .commands import MultiTargetMotionCommand, MultiTargetMotionCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def phase_segment_indices(
  phase: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Map normalized phase to a valid source segment and local blend value."""
  if torch.any(lengths < 2):
    raise ValueError("Phase-parameterized motions require at least two frames.")
  frame = phase.clamp(0.0, 1.0) * (lengths - 1).float()
  frame0 = torch.minimum(frame.floor().long(), lengths - 2)
  frame1 = frame0 + 1
  alpha = frame - frame0.float()
  return frame0, frame1, alpha


def phase_parent_time_steps(phase: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
  """Map phase to the counter expected before the integer command increments it."""
  projected_frame = torch.floor(phase.clamp(0.0, 1.0) * (lengths - 1).float()).long()
  time_steps = projected_frame - 1
  completed = phase >= 1.0 - 1.0e-6
  return torch.where(completed, lengths - 1, time_steps)


def cubic_hermite_phase_sample(
  positions: torch.Tensor,
  velocities: torch.Tensor,
  motion_ids: torch.Tensor,
  phase: torch.Tensor,
  lengths: torch.Tensor,
  source_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Sample a trajectory and its exact derivative with respect to phase."""
  value, derivative_per_phase, _ = cubic_hermite_phase_sample_with_derivatives(
    positions, velocities, motion_ids, phase, lengths, source_dt
  )
  return value, derivative_per_phase


def cubic_hermite_phase_sample_with_derivatives(
  positions: torch.Tensor,
  velocities: torch.Tensor,
  motion_ids: torch.Tensor,
  phase: torch.Tensor,
  lengths: torch.Tensor,
  source_dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Sample a trajectory and its first two derivatives with respect to phase."""
  frame0, frame1, alpha = phase_segment_indices(phase, lengths)
  p0 = positions[motion_ids, frame0]
  p1 = positions[motion_ids, frame1]
  v0 = velocities[motion_ids, frame0]
  v1 = velocities[motion_ids, frame1]

  expand = (len(alpha),) + (1,) * (p0.ndim - 1)
  s = alpha.reshape(expand)
  s2 = s * s
  s3 = s2 * s

  h00 = 2.0 * s3 - 3.0 * s2 + 1.0
  h10 = s3 - 2.0 * s2 + s
  h01 = -2.0 * s3 + 3.0 * s2
  h11 = s3 - s2
  value = h00 * p0 + h10 * source_dt * v0 + h01 * p1 + h11 * source_dt * v1

  dh00 = 6.0 * s2 - 6.0 * s
  dh10 = 3.0 * s2 - 4.0 * s + 1.0
  dh01 = -6.0 * s2 + 6.0 * s
  dh11 = 3.0 * s2 - 2.0 * s
  derivative_per_segment = (
    dh00 * p0 + dh10 * source_dt * v0 + dh01 * p1 + dh11 * source_dt * v1
  )
  segment_count = (lengths - 1).reshape(expand)
  derivative_per_phase = derivative_per_segment * segment_count

  d2h00 = 12.0 * s - 6.0
  d2h10 = 6.0 * s - 4.0
  d2h01 = -12.0 * s + 6.0
  d2h11 = 6.0 * s - 2.0
  second_derivative_per_segment = (
    d2h00 * p0 + d2h10 * source_dt * v0 + d2h01 * p1 + d2h11 * source_dt * v1
  )
  second_derivative_per_phase = second_derivative_per_segment * segment_count.square()
  return value, derivative_per_phase, second_derivative_per_phase


def _resolve_joint_limits(
  joint_names: tuple[str, ...] | list[str],
  limits: dict[str, float],
  *,
  label: str,
  device: str,
) -> torch.Tensor:
  """Resolve regex keyed joint limits in robot joint order."""
  resolved = []
  for joint_name in joint_names:
    matches = [
      value for pattern, value in limits.items() if re.fullmatch(pattern, joint_name)
    ]
    if len(matches) != 1:
      raise ValueError(
        f"Expected exactly one {label} limit for {joint_name!r}, got {len(matches)}."
      )
    resolved.append(matches[0])
  result = torch.tensor(resolved, dtype=torch.float32, device=device)
  if torch.any(result <= 0.0):
    raise ValueError(f"All {label} limits must be positive.")
  return result


def joint_safe_phase_rate(
  derivative_per_phase: torch.Tensor,
  second_derivative_per_phase: torch.Tensor,
  joint_velocity_limits: torch.Tensor,
  joint_acceleration_limits: torch.Tensor,
) -> torch.Tensor:
  """Conservative phase-rate limit from joint velocity and path curvature."""
  eps = torch.finfo(derivative_per_phase.dtype).eps
  velocity_rate = joint_velocity_limits / derivative_per_phase.abs().clamp(min=eps)
  curvature_rate = torch.sqrt(
    joint_acceleration_limits / second_derivative_per_phase.abs().clamp(min=eps)
  )
  return torch.minimum(velocity_rate, curvature_rate).amin(dim=-1)


def joint_safe_phase_acceleration_bounds(
  derivative_per_phase: torch.Tensor,
  second_derivative_per_phase: torch.Tensor,
  phase_rate: torch.Tensor,
  joint_acceleration_limits: torch.Tensor,
  phase_acceleration_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return phase-acceleration bounds satisfying every joint acceleration limit."""
  curvature = second_derivative_per_phase * phase_rate[:, None].square()
  derivative = derivative_per_phase
  active = derivative.abs() > 1.0e-6
  safe_derivative = torch.where(active, derivative, torch.ones_like(derivative))
  bound0 = (-joint_acceleration_limits - curvature) / safe_derivative
  bound1 = (joint_acceleration_limits - curvature) / safe_derivative
  lower_per_joint = torch.minimum(bound0, bound1)
  upper_per_joint = torch.maximum(bound0, bound1)
  lower_per_joint = torch.where(
    active, lower_per_joint, torch.full_like(lower_per_joint, -torch.inf)
  )
  upper_per_joint = torch.where(
    active, upper_per_joint, torch.full_like(upper_per_joint, torch.inf)
  )
  lower = lower_per_joint.amax(dim=-1).clamp(min=-phase_acceleration_limit)
  upper = upper_per_joint.amin(dim=-1).clamp(max=phase_acceleration_limit)
  return lower, upper


def _max_phase_distance(
  phase_rate: torch.Tensor,
  duration: torch.Tensor,
  phase_acceleration: torch.Tensor,
  max_phase_rate: torch.Tensor,
) -> torch.Tensor:
  """Maximum phase distance reachable under constant acceleration/rate bounds."""
  duration = duration.clamp(min=0.0)
  acceleration = phase_acceleration.clamp(min=0.0)
  time_to_limit = torch.where(
    acceleration > 1.0e-8,
    (max_phase_rate - phase_rate).clamp(min=0.0) / acceleration.clamp(min=1.0e-8),
    torch.full_like(duration, torch.inf),
  )
  ramp_time = torch.minimum(duration, time_to_limit)
  return (
    phase_rate * ramp_time
    + 0.5 * acceleration * ramp_time.square()
    + max_phase_rate * (duration - ramp_time)
  )


def _min_phase_distance(
  phase_rate: torch.Tensor,
  duration: torch.Tensor,
  phase_deceleration: torch.Tensor,
  min_phase_rate: torch.Tensor,
) -> torch.Tensor:
  """Minimum phase distance reachable under constant deceleration/rate bounds."""
  duration = duration.clamp(min=0.0)
  deceleration = phase_deceleration.clamp(min=0.0)
  time_to_limit = torch.where(
    deceleration > 1.0e-8,
    (phase_rate - min_phase_rate).clamp(min=0.0) / deceleration.clamp(min=1.0e-8),
    torch.full_like(duration, torch.inf),
  )
  ramp_time = torch.minimum(duration, time_to_limit)
  return (
    phase_rate * ramp_time
    - 0.5 * deceleration * ramp_time.square()
    + min_phase_rate * (duration - ramp_time)
  )


def _linear_phase_sample(
  values: torch.Tensor,
  motion_ids: torch.Tensor,
  phase: torch.Tensor,
  lengths: torch.Tensor,
) -> torch.Tensor:
  frame0, frame1, alpha = phase_segment_indices(phase, lengths)
  value0 = values[motion_ids, frame0]
  value1 = values[motion_ids, frame1]
  blend = alpha.reshape((len(alpha),) + (1,) * (value0.ndim - 1))
  return value0 + (value1 - value0) * blend


def _quat_slerp(
  q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
  """Vectorized shortest-path quaternion interpolation in wxyz convention."""
  original_shape = q0.shape
  q0_flat = q0.reshape(-1, 4)
  q1_flat = q1.reshape(-1, 4)
  blend = alpha.reshape(-1, *([1] * (q0.ndim - 2))).expand(q0.shape[:-1])
  blend_flat = blend.reshape(-1, 1)

  dot = torch.sum(q0_flat * q1_flat, dim=-1, keepdim=True)
  q1_flat = torch.where(dot < 0.0, -q1_flat, q1_flat)
  dot = dot.abs().clamp(max=1.0)

  theta = torch.acos(dot)
  sin_theta = torch.sin(theta)
  safe = sin_theta.abs() > 1.0e-6
  denominator = torch.where(safe, sin_theta, torch.ones_like(sin_theta))
  weight0 = torch.sin((1.0 - blend_flat) * theta) / denominator
  weight1 = torch.sin(blend_flat * theta) / denominator
  slerped = weight0 * q0_flat + weight1 * q1_flat
  linear = q0_flat + blend_flat * (q1_flat - q0_flat)
  result = torch.where(safe, slerped, linear)
  return torch.nn.functional.normalize(result, dim=-1).reshape(original_shape)


class PhaseAwareMultiTargetMotionCommand(MultiTargetMotionCommand):
  """Continuous-phase motion command with a sampled strike deadline."""

  cfg: PhaseAwareMultiTargetMotionCommandCfg

  def __init__(
    self, cfg: PhaseAwareMultiTargetMotionCommandCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)

    self._source_dt = 1.0 / cfg.source_fps
    durations = (self._time_step_totals.float() - 1.0) * self._source_dt
    self._nominal_phase_rates = durations.reciprocal()
    self._strike_phases = torch.tensor(
      [self._strike_phase_from_cfg(motion_cfg) for motion_cfg in self.motion_configs],
      dtype=torch.float32,
      device=self.device,
    )
    self._nominal_contact_times = self._strike_phases * durations

    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.phase_rate = torch.zeros(self.num_envs, device=self.device)
    self.phase_rate_schedule = torch.zeros(self.num_envs, device=self.device)
    self.phase_residual = torch.zeros(self.num_envs, device=self.device)
    self.contact_time = torch.zeros(self.num_envs, device=self.device)
    self.time_remaining = torch.zeros(self.num_envs, device=self.device)
    self.strike_time_error_s = torch.zeros(self.num_envs, device=self.device)

    self.metrics["phase"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["phase_rate"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["contact_time"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["time_remaining"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["phase_timing_error"] = torch.zeros(self.num_envs, device=self.device)
    self._motion_is_forehand_t = torch.tensor(
      ["forehand" in motion.name.lower() for motion in self.motion_configs],
      dtype=torch.bool,
      device=self.device,
    )
    self._motion_is_backhand_t = torch.tensor(
      ["backhand" in motion.name.lower() for motion in self.motion_configs],
      dtype=torch.bool,
      device=self.device,
    )
    self._stroke_attempt_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._stroke_outcome_recorded = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    for side in ("forehand", "backhand"):
      self.metrics[f"{side}_strike_attempts"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics[f"{side}_hits"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics[f"{side}_hit_success_rate"] = torch.zeros(
        self.num_envs, device=self.device
      )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    self._record_stroke_outcomes(env_ids, force=True)
    aggregate_rates = {}
    for side in ("forehand", "backhand"):
      hits = self.metrics[f"{side}_hits"][env_ids].sum()
      attempts = self.metrics[f"{side}_strike_attempts"][env_ids].sum()
      aggregate_rates[f"{side}_hit_success_rate"] = (
        (hits / attempts).item() if attempts.item() > 0.0 else 0.0
      )
    extras = super().reset(env_ids)
    extras.update(aggregate_rates)
    return extras

  def _sample_targets(self, env_ids: torch.Tensor) -> None:
    super()._sample_targets(env_ids)
    self._stroke_attempt_active[env_ids] = self.cfg.incoming_ball_launch_enabled
    self._stroke_outcome_recorded[env_ids] = False

  def _record_stroke_outcomes(
    self, env_ids: torch.Tensor, *, force: bool = False
  ) -> None:
    if len(env_ids) == 0:
      return
    pending = (
      self._stroke_attempt_active[env_ids]
      & ~self._stroke_outcome_recorded[env_ids]
      & self.trajectory_match_valid[env_ids]
    )
    if not force:
      pending &= (
        self.strike_time_error_s[env_ids]
        >= self.cfg.racket_orientation_reward_window_after_s
      )
    pending_env_ids = env_ids[pending]
    if len(pending_env_ids) == 0:
      return

    motion_ids = self.which_motion[pending_env_ids]
    hit = self.ball_has_been_struck[pending_env_ids]
    for side, side_lookup in (
      ("forehand", self._motion_is_forehand_t),
      ("backhand", self._motion_is_backhand_t),
    ):
      side_mask = side_lookup[motion_ids]
      self.metrics[f"{side}_strike_attempts"][pending_env_ids] += side_mask.float()
      self.metrics[f"{side}_hits"][pending_env_ids] += (side_mask & hit).float()
      attempts = self.metrics[f"{side}_strike_attempts"]
      hits = self.metrics[f"{side}_hits"]
      self.metrics[f"{side}_hit_success_rate"][:] = torch.where(
        attempts > 0.0,
        hits / attempts.clamp_min(1.0),
        torch.zeros_like(attempts),
      )
    self._stroke_outcome_recorded[pending_env_ids] = True

  @staticmethod
  def _strike_phase_from_cfg(motion_cfg) -> float:
    position_targets = [
      target for target in motion_cfg.sub_targets if target.goal_type == "position"
    ]
    if position_targets:
      target = position_targets[0]
      return 0.5 * (target.target_phase_start + target.target_phase_end)
    if motion_cfg.probe_points:
      return float(motion_cfg.probe_points[0][1])
    raise ValueError(f"Motion {motion_cfg.name!r} has no strike phase annotation.")

  @property
  def normalized_phase(self) -> torch.Tensor:
    return self.phase

  @property
  def strike_phase(self) -> torch.Tensor:
    return self._strike_phases[self.which_motion]

  @property
  def remaining_phase_to_hit(self) -> torch.Tensor:
    return (self.strike_phase - self.phase).clamp(min=0.0)

  @property
  def desired_strike_phase(self) -> torch.Tensor:
    progress = 1.0 - self.time_remaining / self.contact_time.clamp(min=1.0e-6)
    return self.strike_phase * progress.clamp(0.0, 1.0)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([super().command, self.time_remaining[:, None]], dim=1)

  def set_phase_residual(self, residual: torch.Tensor) -> None:
    self.phase_residual[:] = residual

  def _motion_lengths(self) -> torch.Tensor:
    return self._time_step_totals[self.which_motion]

  def _sample_contact_times(self, env_ids: torch.Tensor) -> None:
    motion_ids = self.which_motion[env_ids]
    maximum = self._nominal_contact_times[motion_ids]
    minimum = maximum / self.cfg.max_contact_speedup
    if (
      getattr(self.cfg, "incoming_ball_torch_match_enabled", False)
      and self._motion_plan_provider is not None
    ):
      sampled = self.planned_contact_time[env_ids]
      if torch.any(~torch.isfinite(sampled)):
        raise RuntimeError("Torch-matched contact time was not planned before reset.")
      invalid = (sampled < minimum - 1.0e-6) | (sampled > maximum + 1.0e-6)
      if torch.any(invalid):
        local_id = int(torch.where(invalid)[0][0].item())
        raise ValueError(
          "Torch-matched contact time is outside the motion deadline interval: "
          f"motion={motion_ids[local_id].item()}, "
          f"contact_time={sampled[local_id].item():.6f}, "
          f"minimum={minimum[local_id].item():.6f}, "
          f"nominal={maximum[local_id].item():.6f}"
        )
    elif self.cfg.incoming_ball_trajectory_drives_deadline:
      if not self.cfg.incoming_ball_launch_enabled:
        raise ValueError(
          "incoming_ball_trajectory_drives_deadline requires incoming-ball launch."
        )
      sampled = torch.full_like(maximum, self.cfg.incoming_ball_flight_time_s)
      invalid = (sampled < minimum - 1.0e-6) | (sampled > maximum + 1.0e-6)
      if torch.any(invalid):
        local_id = int(torch.where(invalid)[0][0].item())
        raise ValueError(
          "Incoming-ball flight time is outside the motion deadline interval: "
          f"motion={motion_ids[local_id].item()}, "
          f"flight_time={sampled[local_id].item():.6f}, "
          f"minimum={minimum[local_id].item():.6f}, "
          f"nominal={maximum[local_id].item():.6f}"
        )
    else:
      counts = torch.ceil((maximum - minimum) / self.cfg.contact_time_step_s).long() + 1
      sample_index = torch.floor(
        torch.rand(len(env_ids), device=self.device) * counts.float()
      )
      sampled = minimum + sample_index * self.cfg.contact_time_step_s
      sampled = torch.minimum(sampled, maximum)
    self.contact_time[env_ids] = sampled
    self.time_remaining[env_ids] = self.contact_time[env_ids]
    self.strike_time_error_s[env_ids] = -self.contact_time[env_ids]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self.phase[env_ids] = 0.0
    self.phase_residual[env_ids] = 0.0
    self._sample_contact_times(env_ids)
    schedule = self.strike_phase[env_ids] / self.contact_time[env_ids].clamp(min=1.0e-6)
    self.phase_rate_schedule[env_ids] = schedule
    self.phase_rate[env_ids] = schedule.clamp(
      self.cfg.phase_rate_min, self.cfg.phase_rate_max
    )
    self.time_steps[env_ids] = 0

    # The reset pose comes from frame zero. Match its velocity to the sampled
    # execution speed so the first policy observation and simulator state agree.
    self.robot.write_joint_velocity_to_sim(self.joint_vel[env_ids], env_ids=env_ids)
    root_velocity = torch.cat(
      [self.body_lin_vel_w[env_ids, 0], self.body_ang_vel_w[env_ids, 0]], dim=-1
    )
    self.robot.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)
    self._env.sim.forward()
    self._update_source_cache()

  def _sample_next_motion(self, env_ids: torch.Tensor) -> None:
    """Chain into a newly sampled motion without resetting the robot."""
    super()._sample_next_motion(env_ids)
    if len(env_ids) == 0:
      return
    self.phase[env_ids] = 0.0
    self.phase_residual[env_ids] = 0.0
    self._sample_contact_times(env_ids)
    schedule = self.strike_phase[env_ids] / self.contact_time[env_ids].clamp(min=1.0e-6)
    self.phase_rate_schedule[env_ids] = schedule
    self.phase_rate[env_ids] = schedule.clamp(
      self.cfg.phase_rate_min, self.cfg.phase_rate_max
    )
    self.time_steps[env_ids] = 0
    self.is_paused[env_ids] = False
    self._update_source_cache()

  def _advance_phase(self) -> None:
    dt = self._env.step_dt
    strike_phase = self.strike_phase
    previous_phase = self.phase.clone()
    time_before_step = self.time_remaining.clone()
    before_strike = previous_phase < strike_phase - 1.0e-6
    before_deadline = time_before_step > 1.0e-6
    waiting_at_strike = (~before_strike) & before_deadline
    overdue = before_strike & (~before_deadline)
    after_strike = (~before_strike) & (~before_deadline)

    schedule = torch.zeros_like(self.phase)
    scheduled = before_strike & before_deadline
    schedule[scheduled] = (
      strike_phase[scheduled] - previous_phase[scheduled]
    ) / time_before_step[scheduled].clamp(min=1.0e-6)
    schedule[overdue] = self.cfg.phase_rate_max
    schedule[after_strike] = self._nominal_phase_rates[self.which_motion[after_strike]]
    self.phase_rate_schedule[:] = schedule

    residual_multiplier = 1.0 + self.cfg.phase_residual_scale * torch.tanh(
      self.phase_residual
    )
    requested_rate = schedule * residual_multiplier
    requested_rate[waiting_at_strike] = 0.0
    requested_rate[overdue] = self.cfg.phase_rate_max
    requested_rate[after_strike] = schedule[after_strike]
    requested_rate.clamp_(self.cfg.phase_rate_min, self.cfg.phase_rate_max)

    next_phase = previous_phase + requested_rate * dt
    next_phase[before_strike] = torch.minimum(
      next_phase[before_strike], strike_phase[before_strike]
    )
    self.phase[:] = next_phase.clamp(0.0, 1.0)
    self.phase_rate[:] = (self.phase - previous_phase) / dt
    self.strike_time_error_s += dt
    self.time_remaining[:] = (-self.strike_time_error_s).clamp(min=0.0)

  def _update_command(self) -> None:
    self._advance_phase()

    # Reuse the mature target/body-alignment update in the integer command.
    # Setting frame-1 here compensates for its initial ``time_steps += 1``;
    # all reference properties below still query the continuous phase.
    lengths = self._motion_lengths()
    self.time_steps[:] = phase_parent_time_steps(self.phase, lengths)
    super()._update_command()

  def _update_metrics(self) -> None:
    previous_target_window_error = self.metrics["error_target_pos_at_target"].clone()
    super()._update_metrics()
    self.metrics["error_target_pos_at_target"][:] = previous_target_window_error
    position_mask = self._target_pos_reward_weights_t[self.which_motion] > 0.0
    phase_active = (
      self.phase[:, None] >= self._target_phase_starts_t[self.which_motion]
    ) & (self.phase[:, None] <= self._target_phase_ends_t[self.which_motion])
    contact_active = self.time_remaining <= self.cfg.contact_reward_window_s
    active_position_mask = position_mask & phase_active & contact_active[:, None]
    active_count = active_position_mask.sum(dim=-1)
    in_target_window = active_count > 0
    if torch.any(in_target_window):
      target_pos_errors = torch.norm(
        self.get_source_pos_w() - self.target_position_w, dim=-1
      )
      weighted_error = (target_pos_errors * active_position_mask.float()).sum(dim=-1)
      self.metrics["error_target_pos_at_target"][in_target_window] = (
        weighted_error[in_target_window] / active_count[in_target_window]
      )
    self.metrics["phase"][:] = self.phase
    self.metrics["phase_rate"][:] = self.phase_rate
    self.metrics["contact_time"][:] = self.contact_time
    self.metrics["time_remaining"][:] = self.time_remaining
    active = (self.time_remaining > 0.0) | (self.phase < self.strike_phase)
    timing_error = torch.where(
      active, self.phase - self.desired_strike_phase, torch.zeros_like(self.phase)
    )
    self.metrics["phase_timing_error"][:] = timing_error.abs()
    env_ids = torch.arange(self.num_envs, device=self.device)
    self._record_stroke_outcomes(env_ids)

  def _joint_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
    return cubic_hermite_phase_sample(
      self._stacked_joint_pos,
      self._stacked_joint_vel,
      self.which_motion,
      self.phase,
      self._motion_lengths(),
      self._source_dt,
    )

  @property
  def joint_pos(self) -> torch.Tensor:
    value, _ = self._joint_sample()
    return value

  @property
  def joint_vel(self) -> torch.Tensor:
    _, derivative_per_phase = self._joint_sample()
    scale = self.phase_rate.reshape(-1, 1)
    return derivative_per_phase * scale

  def _body_position_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
    return cubic_hermite_phase_sample(
      self._stacked_body_pos_w,
      self._stacked_body_lin_vel_w,
      self.which_motion,
      self.phase,
      self._motion_lengths(),
      self._source_dt,
    )

  @property
  def body_pos_w(self) -> torch.Tensor:
    value, _ = self._body_position_sample()
    raw = value + self._env.scene.env_origins[:, None, :]
    env_ids = torch.arange(self.num_envs, device=self.device)
    return self._align_reference_body_pos_w(raw, env_ids)

  @property
  def body_quat_w(self) -> torch.Tensor:
    lengths = self._motion_lengths()
    frame0, frame1, alpha = phase_segment_indices(self.phase, lengths)
    quat0 = self._stacked_body_quat_w[self.which_motion, frame0]
    quat1 = self._stacked_body_quat_w[self.which_motion, frame1]
    raw = _quat_slerp(quat0, quat1, alpha)
    env_ids = torch.arange(self.num_envs, device=self.device)
    return self._align_reference_quat_w(raw, env_ids)

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    _, derivative_per_phase = self._body_position_sample()
    raw = derivative_per_phase * self.phase_rate.reshape(-1, 1, 1)
    env_ids = torch.arange(self.num_envs, device=self.device)
    return self._align_reference_vector_w(raw, env_ids)

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    raw = _linear_phase_sample(
      self._stacked_body_ang_vel_w,
      self.which_motion,
      self.phase,
      self._motion_lengths(),
    )
    nominal_rate = self._nominal_phase_rates[self.which_motion]
    raw = raw * (self.phase_rate / nominal_rate).reshape(-1, 1, 1)
    env_ids = torch.arange(self.num_envs, device=self.device)
    return self._align_reference_vector_w(raw, env_ids)


class PhaseAccelerationMultiTargetMotionCommand(PhaseAwareMultiTargetMotionCommand):
  """Learn a smooth phase schedule through joint-safe phase acceleration."""

  cfg: PhaseAccelerationMultiTargetMotionCommandCfg

  def __init__(
    self, cfg: PhaseAccelerationMultiTargetMotionCommandCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)
    self._joint_velocity_limits = _resolve_joint_limits(
      self.robot.joint_names,
      cfg.joint_velocity_limits,
      label="velocity",
      device=self.device,
    )
    self._joint_acceleration_limits = _resolve_joint_limits(
      self.robot.joint_names,
      cfg.joint_acceleration_limits,
      label="acceleration",
      device=self.device,
    )
    self._safe_rate_envelopes, self._safe_rate_grid_lengths = (
      self._build_safe_rate_envelopes()
    )
    self._future_safe_rate_caps = self._build_future_safe_rate_caps()
    self.phase_acceleration_action = torch.zeros(self.num_envs, device=self.device)
    self.phase_acceleration = torch.zeros(self.num_envs, device=self.device)
    self.phase_hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.failure_trajectory_time_remaining = torch.zeros(
      self.num_envs, device=self.device
    )
    self.required_phase_rate = torch.zeros(self.num_envs, device=self.device)
    self.safe_max_phase_rate = torch.zeros(self.num_envs, device=self.device)
    self.deadline_projection_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.deadline_feasible = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._projection_count = torch.zeros(self.num_envs, device=self.device)
    self._control_step_count = torch.zeros(self.num_envs, device=self.device)
    self._episode_deadline_feasible = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.next_rate_safety_projection = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.nominal_rate_recovery = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.phase_acceleration_safety_override = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.rate_envelope_recovery = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._hard_phase_acceleration_lower = torch.zeros(self.num_envs, device=self.device)
    self._hard_phase_acceleration_upper = torch.zeros(self.num_envs, device=self.device)
    self._next_rate_safety_projection_count = torch.zeros(
      self.num_envs, device=self.device
    )
    self._nominal_rate_recovery_count = torch.zeros(self.num_envs, device=self.device)
    self._phase_acceleration_safety_override_count = torch.zeros(
      self.num_envs, device=self.device
    )
    self._rate_envelope_recovery_count = torch.zeros(self.num_envs, device=self.device)

    for name in (
      "phase_acceleration",
      "required_phase_rate",
      "safe_max_phase_rate",
      "deadline_projection_fraction",
      "deadline_feasible",
      "next_rate_safety_projection_fraction",
      "nominal_rate_recovery_fraction",
      "phase_acceleration_safety_override_fraction",
      "rate_envelope_recovery_fraction",
    ):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  def set_phase_acceleration_action(self, action: torch.Tensor) -> None:
    self.phase_acceleration_action[:] = action

  def _build_safe_rate_envelopes(self) -> tuple[torch.Tensor, torch.Tensor]:
    subdivisions = self.cfg.safe_rate_samples_per_frame
    motion_count = len(self._time_step_totals)
    max_frames = int(self._time_step_totals.max().item())
    max_grid_length = (max_frames - 1) * subdivisions + 1
    envelopes = torch.full(
      (motion_count, max_grid_length),
      torch.inf,
      dtype=torch.float32,
      device=self.device,
    )
    grid_lengths = torch.zeros(motion_count, dtype=torch.long, device=self.device)

    for motion_id in range(motion_count):
      frame_count = int(self._time_step_totals[motion_id].item())
      grid_length = (frame_count - 1) * subdivisions + 1
      phase = torch.linspace(0.0, 1.0, grid_length, device=self.device)
      motion_ids = torch.full(
        (grid_length,), motion_id, dtype=torch.long, device=self.device
      )
      lengths = torch.full(
        (grid_length,), frame_count, dtype=torch.long, device=self.device
      )
      _, derivative, second_derivative = cubic_hermite_phase_sample_with_derivatives(
        self._stacked_joint_pos,
        self._stacked_joint_vel,
        motion_ids,
        phase,
        lengths,
        self._source_dt,
      )
      local_limit = joint_safe_phase_rate(
        derivative,
        second_derivative,
        self._joint_velocity_limits,
        self._joint_acceleration_limits,
      )
      phase_left = torch.nextafter(phase, torch.full_like(phase, -torch.inf)).clamp(
        min=0.0
      )
      _, derivative_left, second_derivative_left = (
        cubic_hermite_phase_sample_with_derivatives(
          self._stacked_joint_pos,
          self._stacked_joint_vel,
          motion_ids,
          phase_left,
          lengths,
          self._source_dt,
        )
      )
      left_limit = joint_safe_phase_rate(
        derivative_left,
        second_derivative_left,
        self._joint_velocity_limits,
        self._joint_acceleration_limits,
      )
      local_limit = torch.minimum(local_limit, left_limit)
      local_limit *= self.cfg.joint_limit_safety_factor

      delta_phase = 1.0 / (grid_length - 1)
      index = torch.arange(grid_length, device=self.device, dtype=torch.float32)
      envelope_acceleration = (
        self.cfg.phase_acceleration_limit * self.cfg.joint_limit_safety_factor
      )
      slope = 2.0 * envelope_acceleration * delta_phase
      squared = local_limit.square()

      backward_shifted = squared + slope * index
      backward = (
        torch.flip(
          torch.cummin(torch.flip(backward_shifted, dims=(0,)), dim=0).values,
          dims=(0,),
        )
        - slope * index
      )
      squared = torch.minimum(squared, backward.clamp(min=0.0))

      forward_shifted = squared - slope * index
      forward = torch.cummin(forward_shifted, dim=0).values + slope * index
      envelope = torch.sqrt(torch.minimum(squared, forward.clamp(min=0.0)))

      nominal_rate = self._nominal_phase_rates[motion_id]
      minimum_rate, minimum_index = envelope.min(dim=0)
      if minimum_rate < nominal_rate - 1.0e-5:
        minimum_phase = minimum_index.float() / (grid_length - 1)
        raise RuntimeError(
          "The configured joint limits make nominal motion infeasible: "
          f"motion={motion_id}, phase={minimum_phase.item():.6f}, "
          f"nominal_rate={nominal_rate.item():.6f}, "
          f"safe_max_rate={minimum_rate.item():.6f}."
        )
      envelopes[motion_id, :grid_length] = envelope
      grid_lengths[motion_id] = grid_length

    return envelopes, grid_lengths

  def _build_future_safe_rate_caps(self) -> torch.Tensor:
    caps = self._safe_rate_envelopes.clone()
    for motion_id in range(len(self._time_step_totals)):
      grid_length = int(self._safe_rate_grid_lengths[motion_id].item())
      strike_index = min(
        math.ceil(self._strike_phases[motion_id].item() * (grid_length - 1)),
        grid_length - 1,
      )
      values = self._safe_rate_envelopes[motion_id, : strike_index + 1]
      if strike_index > 0:
        delta_phase = 1.0 / (grid_length - 1)
        segment_rate = torch.minimum(values[:-1], values[1:])
        segment_time = delta_phase / segment_rate.clamp(min=1.0e-6)
        remaining_time = torch.flip(
          torch.cumsum(torch.flip(segment_time, dims=(0,)), dim=0),
          dims=(0,),
        )
        remaining_phase = delta_phase * torch.arange(
          strike_index, 0, -1, device=self.device, dtype=torch.float32
        )
        effective_rate = remaining_phase / remaining_time
        nominal_rate = self._nominal_phase_rates[motion_id]
        caps[motion_id, :strike_index] = torch.maximum(
          effective_rate * self.cfg.joint_limit_safety_factor,
          nominal_rate,
        )
      caps[motion_id, strike_index] = values[-1]
    return caps

  def _future_safe_rate_cap(self) -> torch.Tensor:
    grid_lengths = self._safe_rate_grid_lengths[self.which_motion]
    grid_position = self.phase.clamp(0.0, 1.0) * (grid_lengths - 1).float()
    index = torch.minimum(grid_position.floor().long(), grid_lengths - 1)
    return self._future_safe_rate_caps[self.which_motion, index]

  def _safe_rate_envelope(self, phase: torch.Tensor) -> torch.Tensor:
    """Interpolate the reachable squared-rate envelope at a phase."""
    grid_lengths = self._safe_rate_grid_lengths[self.which_motion]
    grid_position = phase.clamp(0.0, 1.0) * (grid_lengths - 1).float()
    index0 = torch.minimum(grid_position.floor().long(), grid_lengths - 1)
    index1 = torch.minimum(index0 + 1, grid_lengths - 1)
    alpha = grid_position - index0.float()
    envelope0 = self._safe_rate_envelopes[self.which_motion, index0]
    envelope1 = self._safe_rate_envelopes[self.which_motion, index1]
    squared = torch.lerp(envelope0.square(), envelope1.square(), alpha)
    return torch.sqrt(squared.clamp(min=0.0))

  def _safe_next_phase_rate(self) -> torch.Tensor:
    """Solve rate <= envelope(phase + rate * dt) for the next state."""
    nominal_rate = self._nominal_phase_rates[self.which_motion]
    low = nominal_rate.clone()
    high = torch.full_like(low, self.cfg.phase_rate_max)
    for _ in range(self.cfg.deadline_projection_iterations):
      midpoint = 0.5 * (low + high)
      next_phase = self.phase + midpoint * self._env.step_dt
      _, derivative, second_derivative = self._joint_phase_sample(next_phase)
      local_limit = joint_safe_phase_rate(
        derivative,
        second_derivative,
        self._joint_velocity_limits,
        self._joint_acceleration_limits,
      )
      local_limit *= self.cfg.joint_limit_safety_factor
      safe_limit = torch.minimum(
        local_limit, self._safe_rate_envelope(next_phase)
      ).clamp(max=self.cfg.phase_rate_max)
      feasible = midpoint <= safe_limit
      low = torch.where(feasible, midpoint, low)
      high = torch.where(feasible, high, midpoint)
    return low

  def _joint_phase_sample(
    self, phase: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if phase is None:
      phase = self.phase
    return cubic_hermite_phase_sample_with_derivatives(
      self._stacked_joint_pos,
      self._stacked_joint_vel,
      self.which_motion,
      phase,
      self._motion_lengths(),
      self._source_dt,
    )

  def _joint_safe_rate_at_phase(
    self, phase: torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> torch.Tensor:
    if env_ids is None:
      motion_ids = self.which_motion
    else:
      motion_ids = self.which_motion[env_ids]
    lengths = self._time_step_totals[motion_ids]
    _, derivative, second_derivative = cubic_hermite_phase_sample_with_derivatives(
      self._stacked_joint_pos,
      self._stacked_joint_vel,
      motion_ids,
      phase.clamp(0.0, 1.0),
      lengths,
      self._source_dt,
    )
    safe_rate = joint_safe_phase_rate(
      derivative,
      second_derivative,
      self._joint_velocity_limits,
      self._joint_acceleration_limits,
    )
    return safe_rate * self.cfg.joint_limit_safety_factor

  def _project_joint_safe_next_rate(self, acceleration: torch.Tensor) -> torch.Tensor:
    dt = self._env.step_dt
    candidate_rate = (self.phase_rate + acceleration * dt).clamp(
      min=self.cfg.phase_rate_min, max=self.cfg.phase_rate_max
    )
    candidate_phase = self.phase + candidate_rate * dt
    candidate_limit = self._joint_safe_rate_at_phase(candidate_phase)
    active = self.phase < 1.0 - 1.0e-6
    needs_projection = active & (candidate_rate > candidate_limit + 1.0e-6)
    self.next_rate_safety_projection.zero_()
    if not torch.any(needs_projection):
      return acceleration

    env_ids = torch.where(needs_projection)[0]
    current_rate = self.phase_rate[env_ids]
    low_rate = (current_rate + self._hard_phase_acceleration_lower[env_ids] * dt).clamp(
      min=self.cfg.phase_rate_min, max=self.cfg.phase_rate_max
    )
    high_rate = (
      current_rate + self._hard_phase_acceleration_upper[env_ids] * dt
    ).clamp(min=self.cfg.phase_rate_min, max=self.cfg.phase_rate_max)
    requested_rate = candidate_rate[env_ids]
    best_rate = requested_rate.clone()
    best_distance = torch.full_like(requested_rate, torch.inf)
    best_margin = torch.full_like(requested_rate, -torch.inf)
    found = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

    for sample_index in range(self.cfg.next_rate_projection_samples):
      alpha = sample_index / (self.cfg.next_rate_projection_samples - 1)
      sample_rate = torch.lerp(low_rate, high_rate, alpha)
      sample_phase = self.phase[env_ids] + sample_rate * dt
      sample_limit = self._joint_safe_rate_at_phase(sample_phase, env_ids)
      margin = sample_limit - sample_rate
      feasible = margin >= -1.0e-6
      distance = (sample_rate - requested_rate).abs()
      better = feasible & (distance < best_distance)
      best_rate = torch.where(better, sample_rate, best_rate)
      best_distance = torch.where(better, distance, best_distance)
      best_margin = torch.maximum(best_margin, margin)
      found |= feasible

    if not torch.all(found):
      local_id = int(torch.where(~found)[0][0].item())
      env_id = int(env_ids[local_id].item())
      raise RuntimeError(
        "No joint-safe next phase rate in the one-step reachable interval: "
        f"env={env_id}, motion={self.which_motion[env_id].item()}, "
        f"phase={self.phase[env_id].item():.6f}, "
        f"phase_rate={self.phase_rate[env_id].item():.6f}, "
        f"requested_rate={candidate_rate[env_id].item():.6f}, "
        f"low_rate={low_rate[local_id].item():.6f}, "
        f"high_rate={high_rate[local_id].item():.6f}, "
        f"best_margin={best_margin[local_id].item():.6f}."
      )

    projected = acceleration.clone()
    projected[env_ids] = (best_rate - current_rate) / dt
    self.next_rate_safety_projection[env_ids] = True
    return projected

  def _phase_control_limits(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, derivative, second_derivative = self._joint_phase_sample()
    safe_max_rate = self._safe_next_phase_rate()
    nominal_rate = self._nominal_phase_rates[self.which_motion]
    completed = (self.phase >= 1.0 - 1.0e-6) | self.phase_hold
    infeasible = (~completed) & (safe_max_rate < nominal_rate - 1.0e-5)
    if torch.any(infeasible):
      env_id = int(torch.where(infeasible)[0][0].item())
      raise RuntimeError(
        "The configured joint limits make nominal motion infeasible: "
        f"env={env_id}, phase={self.phase[env_id].item():.6f}, "
        f"nominal_rate={nominal_rate[env_id].item():.6f}, "
        f"safe_max_rate={safe_max_rate[env_id].item():.6f}."
      )

    raw_joint_lower, raw_joint_upper = joint_safe_phase_acceleration_bounds(
      derivative,
      second_derivative,
      self.phase_rate,
      self._joint_acceleration_limits,
      math.inf,
    )
    dt = self._env.step_dt
    absolute_rate_lower = (self.cfg.phase_rate_min - self.phase_rate) / dt
    absolute_rate_upper = (self.cfg.phase_rate_max - self.phase_rate) / dt
    hard_lower = torch.maximum(raw_joint_lower, absolute_rate_lower)
    hard_upper = torch.minimum(raw_joint_upper, absolute_rate_upper)
    self._hard_phase_acceleration_lower[:] = hard_lower
    self._hard_phase_acceleration_upper[:] = hard_upper
    hard_feasible = hard_lower <= hard_upper + 1.0e-6

    nominal_delta = (nominal_rate - self.phase_rate) / dt
    nominal_lower = torch.maximum(hard_lower, nominal_delta)
    nominal_feasible = hard_feasible & (nominal_lower <= hard_upper + 1.0e-6)
    control_lower = torch.where(nominal_feasible, nominal_lower, hard_lower)
    control_upper = hard_upper

    rate_upper = (safe_max_rate - self.phase_rate) / dt
    envelope_upper = torch.minimum(control_upper, rate_upper)
    envelope_feasible = hard_feasible & (control_lower <= envelope_upper + 1.0e-6)
    smooth_lower = torch.maximum(
      control_lower,
      torch.full_like(control_lower, -self.cfg.phase_acceleration_limit),
    )
    smooth_upper = torch.minimum(
      envelope_upper,
      torch.full_like(envelope_upper, self.cfg.phase_acceleration_limit),
    )
    smooth_feasible = hard_feasible & (smooth_lower <= smooth_upper + 1.0e-6)

    zero = torch.zeros_like(control_lower)
    envelope_acceleration = torch.minimum(
      torch.maximum(zero, control_lower), envelope_upper
    )
    hard_acceleration = torch.minimum(torch.maximum(zero, control_lower), control_upper)
    normal_override = torch.where(
      envelope_feasible, envelope_acceleration, hard_acceleration
    )
    nominal_recovery_acceleration = torch.minimum(
      torch.maximum(nominal_delta, hard_lower), hard_upper
    )
    nominal_rate_recovery = hard_feasible & ~nominal_feasible
    override_acceleration = torch.where(
      nominal_rate_recovery, nominal_recovery_acceleration, normal_override
    )
    safety_override = hard_feasible & (nominal_rate_recovery | ~smooth_feasible)
    rate_envelope_recovery = safety_override & (
      override_acceleration > rate_upper + 1.0e-6
    )

    recovery_next_rate = self.phase_rate + override_acceleration * dt
    safe_max_rate = torch.where(
      rate_envelope_recovery, recovery_next_rate, safe_max_rate
    )
    acceleration_lower = torch.where(
      safety_override, override_acceleration, smooth_lower
    )
    acceleration_upper = torch.where(
      safety_override, override_acceleration, smooth_upper
    )

    acceleration_lower[completed] = 0.0
    acceleration_upper[completed] = 0.0
    safe_max_rate[completed] = nominal_rate[completed]
    nominal_rate_recovery[completed] = False
    safety_override[completed] = False
    rate_envelope_recovery[completed] = False
    self.nominal_rate_recovery[:] = nominal_rate_recovery
    self.phase_acceleration_safety_override[:] = safety_override
    self.rate_envelope_recovery[:] = rate_envelope_recovery

    unresolved = (~completed) & ~hard_feasible
    if torch.any(unresolved):
      env_id = int(torch.where(unresolved)[0][0].item())
      raise RuntimeError(
        "No joint-safe phase acceleration exists within absolute rate limits: "
        f"env={env_id}, motion={self.which_motion[env_id].item()}, "
        f"phase={self.phase[env_id].item():.6f}, "
        f"phase_rate={self.phase_rate[env_id].item():.6f}, "
        f"nominal_rate={nominal_rate[env_id].item():.6f}, "
        f"safe_max_rate={safe_max_rate[env_id].item():.6f}, "
        f"raw_joint_lower={raw_joint_lower[env_id].item():.6f}, "
        f"raw_joint_upper={raw_joint_upper[env_id].item():.6f}, "
        f"absolute_lower={absolute_rate_lower[env_id].item():.6f}, "
        f"absolute_upper={absolute_rate_upper[env_id].item():.6f}."
      )
    return safe_max_rate, acceleration_lower, acceleration_upper

  def _deadline_project(
    self,
    requested_acceleration: torch.Tensor,
    acceleration_lower: torch.Tensor,
    acceleration_upper: torch.Tensor,
    min_phase_rate: torch.Tensor,
    max_phase_rate: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Project acceleration onto the one-step deadline-reachable set."""
    dt = self._env.step_dt
    active = (self.time_remaining > 1.0e-6) & (self.phase < self.strike_phase - 1.0e-6)
    if not torch.any(active):
      return requested_acceleration, torch.ones_like(active)

    future_time = (self.time_remaining - dt).clamp(min=0.0)
    future_max_phase_rate = self._future_safe_rate_cap()

    def state_after(acceleration: torch.Tensor):
      next_rate = (self.phase_rate + acceleration * dt).clamp(
        min=min_phase_rate, max=max_phase_rate
      )
      next_phase = self.phase + next_rate * dt
      remaining = self.strike_phase - next_phase
      return next_rate, remaining

    def can_still_reach(acceleration: torch.Tensor) -> torch.Tensor:
      next_rate, remaining = state_after(acceleration)
      reachable = _max_phase_distance(
        next_rate,
        future_time,
        torch.full_like(next_rate, self.cfg.phase_acceleration_limit),
        future_max_phase_rate,
      )
      return remaining <= reachable + 1.0e-6

    def can_avoid_early(acceleration: torch.Tensor) -> torch.Tensor:
      next_rate, remaining = state_after(acceleration)
      minimum = _min_phase_distance(
        next_rate,
        future_time,
        torch.full_like(next_rate, self.cfg.phase_acceleration_limit),
        min_phase_rate,
      )
      return remaining + 1.0e-6 >= minimum

    projected = requested_acceleration
    upper_reachable = can_still_reach(acceleration_upper)
    needs_more = active & ~can_still_reach(projected)
    low = projected
    high = acceleration_upper
    for _ in range(self.cfg.deadline_projection_iterations):
      midpoint = 0.5 * (low + high)
      midpoint_reachable = can_still_reach(midpoint)
      high = torch.where(midpoint_reachable, midpoint, high)
      low = torch.where(midpoint_reachable, low, midpoint)
    projected = torch.where(needs_more & upper_reachable, high, projected)
    projected = torch.where(
      needs_more & ~upper_reachable, acceleration_upper, projected
    )

    lower_avoids_early = can_avoid_early(acceleration_lower)
    needs_less = active & ~can_avoid_early(projected)
    low = acceleration_lower
    high = projected
    for _ in range(self.cfg.deadline_projection_iterations):
      midpoint = 0.5 * (low + high)
      midpoint_avoids_early = can_avoid_early(midpoint)
      low = torch.where(midpoint_avoids_early, midpoint, low)
      high = torch.where(midpoint_avoids_early, high, midpoint)
    projected = torch.where(needs_less & lower_avoids_early, low, projected)
    projected = torch.where(
      needs_less & ~lower_avoids_early, acceleration_lower, projected
    )

    feasible = (~active) | (upper_reachable & lower_avoids_early)
    return projected, feasible

  def _reset_phase_diagnostics(self, env_ids: torch.Tensor) -> None:
    self.deadline_projection_active[env_ids] = False
    self.deadline_feasible[env_ids] = True
    self.next_rate_safety_projection[env_ids] = False
    self.nominal_rate_recovery[env_ids] = False
    self.phase_acceleration_safety_override[env_ids] = False
    self.rate_envelope_recovery[env_ids] = False
    self._projection_count[env_ids] = 0.0
    self._next_rate_safety_projection_count[env_ids] = 0.0
    self._nominal_rate_recovery_count[env_ids] = 0.0
    self._phase_acceleration_safety_override_count[env_ids] = 0.0
    self._rate_envelope_recovery_count[env_ids] = 0.0
    self._control_step_count[env_ids] = 0.0
    self._episode_deadline_feasible[env_ids] = True

  def _set_external_motion_ids(
    self, env_ids: torch.Tensor, motion_ids: torch.Tensor
  ) -> None:
    if env_ids.ndim != 1 or motion_ids.shape != env_ids.shape:
      raise ValueError("env_ids and motion_ids must be matching 1D tensors")
    motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
    if torch.any((motion_ids < 0) | (motion_ids >= len(self.motion_configs))):
      raise ValueError("motion_ids contains an out-of-range motion index")
    self.which_motion[env_ids] = motion_ids

  def hold_at_phase_zero(
    self, env_ids: torch.Tensor, motion_ids: torch.Tensor | None = None
  ) -> None:
    """Hold selected references at frame zero without changing robot state."""
    if motion_ids is not None:
      self._set_external_motion_ids(env_ids, motion_ids)
    self.phase_hold[env_ids] = True
    self.phase[env_ids] = 0.0
    self.phase_rate[env_ids] = 0.0
    self.phase_acceleration_action[env_ids] = 0.0
    self.phase_acceleration[env_ids] = 0.0
    self.contact_time[env_ids] = 0.0
    self.time_remaining[env_ids] = 0.0
    self.strike_time_error_s[env_ids] = 0.0
    self.required_phase_rate[env_ids] = 0.0
    self.phase_rate_schedule[env_ids] = 0.0
    self.safe_max_phase_rate[env_ids] = 0.0
    self.time_steps[env_ids] = 0
    self.is_paused[env_ids] = False
    self.between_motion_pause_time[env_ids] = 0.0
    self._reset_phase_diagnostics(env_ids)
    self._update_source_cache()

  def _configure_failure_pause_reference(
    self, env_ids: torch.Tensor, duration: torch.Tensor
  ) -> None:
    """Extend the native between-motion pause while an unmatched ball passes."""
    if len(env_ids) == 0:
      return

    self.phase_hold[env_ids] = False
    self.phase[env_ids] = 1.0
    self.phase_rate[env_ids] = 0.0
    self.phase_acceleration_action[env_ids] = 0.0
    self.phase_acceleration[env_ids] = 0.0
    self.contact_time[env_ids] = 0.0
    self.time_remaining[env_ids] = 0.0
    self.strike_time_error_s[env_ids] = 0.0
    self.required_phase_rate[env_ids] = 0.0
    self.phase_rate_schedule[env_ids] = 0.0
    self.safe_max_phase_rate[env_ids] = 0.0
    self.time_steps[env_ids] = self._motion_lengths()[env_ids] - 1
    self.is_paused[env_ids] = True
    self.between_motion_pause_time[env_ids] = 0.0
    self._sampled_pause_lengths[env_ids] = torch.inf
    self.failure_trajectory_time_remaining[env_ids] = duration
    self._reset_phase_diagnostics(env_ids)
    self._update_source_cache()

  def _configure_planned_motion_or_failure_hold(self, env_ids: torch.Tensor) -> None:
    """Start matched references or extend the prior native pause."""
    failure_enabled = (
      self.cfg.incoming_ball_failure_trajectory_probability > 0.0
      and self._motion_plan_provider is not None
    )
    if failure_enabled:
      failure = ~self.trajectory_match_valid[env_ids]
    else:
      failure = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
    active_env_ids = env_ids[~failure]
    failure_env_ids = env_ids[failure]

    self.phase_hold[env_ids] = False
    self.phase[env_ids] = 0.0
    self.phase_acceleration_action[env_ids] = 0.0
    self.phase_acceleration[env_ids] = 0.0
    self.failure_trajectory_time_remaining[env_ids] = 0.0

    if len(active_env_ids) > 0:
      self._sample_contact_times(active_env_ids)
      nominal_rate = self._nominal_phase_rates[self.which_motion[active_env_ids]]
      self.phase_rate[active_env_ids] = nominal_rate
      required_rate = self.strike_phase[active_env_ids] / self.contact_time[
        active_env_ids
      ].clamp(min=1.0e-6)
      self.required_phase_rate[active_env_ids] = required_rate
      self.phase_rate_schedule[active_env_ids] = required_rate
      self.safe_max_phase_rate[active_env_ids] = nominal_rate

    if len(failure_env_ids) > 0:
      failure_duration = self.planned_contact_time[failure_env_ids].clamp_min(
        self._env.step_dt
      )
      self._configure_failure_pause_reference(failure_env_ids, failure_duration)
      # An unmatched launch is a negative sample, not a failed stroke attempt.
      self._stroke_attempt_active[failure_env_ids] = False
      self._stroke_outcome_recorded[failure_env_ids] = True

    self.time_steps[active_env_ids] = 0
    self.is_paused[active_env_ids] = False
    self.between_motion_pause_time[active_env_ids] = 0.0
    self._reset_phase_diagnostics(env_ids)
    self._update_source_cache()

  def activate_motion_deadline(
    self,
    env_ids: torch.Tensor,
    motion_ids: torch.Tensor,
    contact_times: torch.Tensor,
  ) -> None:
    """Start selected motions with explicit, fixed strike deadlines."""
    motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
    self._set_external_motion_ids(env_ids, motion_ids)
    contact_times = contact_times.to(device=self.device, dtype=self.phase.dtype)
    if contact_times.shape != env_ids.shape:
      raise ValueError("contact_times must match env_ids")
    nominal = self._nominal_contact_times[motion_ids]
    minimum = nominal / self.cfg.max_contact_speedup
    invalid = (contact_times < minimum - 1.0e-6) | (contact_times > nominal + 1.0e-6)
    if torch.any(invalid):
      local_id = int(torch.where(invalid)[0][0].item())
      raise ValueError(
        "contact time is outside the trained motion interval: "
        f"motion={motion_ids[local_id].item()}, "
        f"contact={contact_times[local_id].item():.6f}, "
        f"minimum={minimum[local_id].item():.6f}, "
        f"nominal={nominal[local_id].item():.6f}"
      )

    self.phase_hold[env_ids] = False
    self.phase[env_ids] = 0.0
    self.phase_acceleration_action[env_ids] = 0.0
    self.phase_acceleration[env_ids] = 0.0
    nominal_rate = self._nominal_phase_rates[motion_ids]
    self.phase_rate[env_ids] = nominal_rate
    self.contact_time[env_ids] = contact_times
    self.time_remaining[env_ids] = contact_times
    self.strike_time_error_s[env_ids] = -contact_times
    required_rate = self.strike_phase[env_ids] / contact_times.clamp(min=1.0e-6)
    self.required_phase_rate[env_ids] = required_rate
    self.phase_rate_schedule[env_ids] = required_rate
    self.safe_max_phase_rate[env_ids] = nominal_rate
    self.time_steps[env_ids] = 0
    self.is_paused[env_ids] = False
    self.between_motion_pause_time[env_ids] = 0.0
    self._reset_phase_diagnostics(env_ids)
    self._update_source_cache()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    MultiTargetMotionCommand._resample_command(self, env_ids)
    self._configure_planned_motion_or_failure_hold(env_ids)

    self.robot.write_joint_velocity_to_sim(self.joint_vel[env_ids], env_ids=env_ids)
    root_velocity = torch.cat(
      [self.body_lin_vel_w[env_ids, 0], self.body_ang_vel_w[env_ids, 0]], dim=-1
    )
    self.robot.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)
    self._env.sim.forward()
    self._update_source_cache()

  def _sample_next_motion(self, env_ids: torch.Tensor) -> None:
    """Chain into a newly sampled phase-acceleration motion in-place."""
    MultiTargetMotionCommand._sample_next_motion(self, env_ids)
    if len(env_ids) == 0:
      return
    self._configure_planned_motion_or_failure_hold(env_ids)

  def _update_command(self) -> None:
    failure_active = (~self.trajectory_match_valid) & (
      self.failure_trajectory_time_remaining > 0.0
    )
    failure_env_ids = torch.where(failure_active)[0]
    saved_auto_chain = self.auto_chain_motion[failure_env_ids].clone()
    self.auto_chain_motion[failure_env_ids] = True
    self._sampled_pause_lengths[failure_env_ids] = torch.inf
    super()._update_command()
    self.auto_chain_motion[failure_env_ids] = saved_auto_chain

    self.failure_trajectory_time_remaining[failure_active] -= self._env.step_dt
    self.failure_trajectory_time_remaining.clamp_(min=0.0)
    self.time_remaining[failure_active] = 0.0
    self.strike_time_error_s[failure_active] = 0.0
    continue_ids = torch.where(
      failure_active & (self.failure_trajectory_time_remaining <= 0.0)
    )[0]
    if len(continue_ids) > 0:
      self._sample_next_motion(continue_ids)

  def _advance_phase(self) -> None:
    dt = self._env.step_dt
    previous_phase = self.phase.clone()
    previous_rate = self.phase_rate.clone()
    time_before_step = self.time_remaining.clone()
    held = self.phase_hold
    completed = previous_phase >= 1.0 - 1.0e-6
    nominal_min_phase_rate = self._nominal_phase_rates[self.which_motion]
    max_phase_rate, acceleration_lower, acceleration_upper = (
      self._phase_control_limits()
    )
    min_phase_rate = torch.where(
      self.nominal_rate_recovery,
      torch.full_like(nominal_min_phase_rate, self.cfg.phase_rate_min),
      nominal_min_phase_rate,
    )
    requested_acceleration = self.cfg.phase_acceleration_limit * torch.tanh(
      self.phase_acceleration_action
    )
    requested_acceleration = torch.maximum(
      torch.minimum(requested_acceleration, acceleration_upper),
      acceleration_lower,
    )

    projected_acceleration = requested_acceleration
    deadline_feasible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    if self.cfg.deadline_projection:
      projected_acceleration, deadline_feasible = self._deadline_project(
        requested_acceleration,
        acceleration_lower,
        acceleration_upper,
        min_phase_rate,
        max_phase_rate,
      )

    projected_acceleration = self._project_joint_safe_next_rate(projected_acceleration)
    next_rate_min = torch.where(
      self.next_rate_safety_projection,
      torch.full_like(min_phase_rate, self.cfg.phase_rate_min),
      min_phase_rate,
    )
    next_rate_max = torch.where(
      self.next_rate_safety_projection,
      torch.full_like(max_phase_rate, self.cfg.phase_rate_max),
      max_phase_rate,
    )
    next_rate = previous_rate + projected_acceleration * dt
    next_rate = torch.maximum(torch.minimum(next_rate, next_rate_max), next_rate_min)
    applied_acceleration = (next_rate - previous_rate) / dt
    next_phase = previous_phase + next_rate * dt
    next_rate[completed] = 0.0
    applied_acceleration[completed] = 0.0
    next_phase[completed] = 1.0
    next_rate[held] = 0.0
    applied_acceleration[held] = 0.0
    next_phase[held] = 0.0

    self.phase[:] = next_phase.clamp(0.0, 1.0)
    self.phase_rate[:] = (self.phase - previous_phase) / dt
    self.phase_acceleration[:] = applied_acceleration
    self.safe_max_phase_rate[:] = max_phase_rate
    remaining_phase = (self.strike_phase - self.phase).clamp(min=0.0)
    next_time_remaining = (self.time_remaining - dt).clamp(min=0.0)
    next_time_remaining[next_time_remaining < 1.0e-5] = 0.0
    next_time_remaining[held] = time_before_step[held]
    self.time_remaining[:] = next_time_remaining
    self.strike_time_error_s[:] = torch.where(
      held,
      self.strike_time_error_s,
      self.strike_time_error_s + dt,
    )
    self.required_phase_rate[:] = torch.where(
      self.time_remaining > 1.0e-6,
      remaining_phase / self.time_remaining.clamp(min=1.0e-6),
      torch.where(remaining_phase > 1.0e-6, max_phase_rate, min_phase_rate),
    )
    self.required_phase_rate[held] = 0.0
    self.phase_rate_schedule[:] = self.required_phase_rate
    self.deadline_projection_active[:] = (
      projected_acceleration - requested_acceleration
    ).abs() > 1.0e-5
    self.deadline_feasible[:] = deadline_feasible
    self.deadline_projection_active[held] = False
    self.deadline_feasible[held] = True
    self.next_rate_safety_projection[held] = False
    self.nominal_rate_recovery[held] = False
    self.phase_acceleration_safety_override[held] = False
    self.rate_envelope_recovery[held] = False
    active_step = (~held).float()
    self._control_step_count += active_step
    self._projection_count += self.deadline_projection_active.float()
    self._next_rate_safety_projection_count += self.next_rate_safety_projection.float()
    self._nominal_rate_recovery_count += self.nominal_rate_recovery.float()
    self._phase_acceleration_safety_override_count += (
      self.phase_acceleration_safety_override.float()
    )
    self._rate_envelope_recovery_count += self.rate_envelope_recovery.float()
    deadline_reached = (
      (~held) & (time_before_step > 1.0e-6) & (next_time_remaining <= 0.0)
    )
    deadline_met = (
      self.phase - self.strike_phase
    ).abs() <= self.cfg.deadline_phase_tolerance
    self._episode_deadline_feasible[deadline_reached] = deadline_met[deadline_reached]

  def _update_metrics(self) -> None:
    super()._update_metrics()
    self.metrics["phase_acceleration"][:] = self.phase_acceleration
    self.metrics["required_phase_rate"][:] = self.required_phase_rate
    self.metrics["safe_max_phase_rate"][:] = self.safe_max_phase_rate
    self.metrics["deadline_projection_fraction"][:] = self._projection_count / (
      self._control_step_count.clamp(min=1.0)
    )
    self.metrics["deadline_feasible"][:] = self._episode_deadline_feasible.float()
    self.metrics["next_rate_safety_projection_fraction"][:] = (
      self._next_rate_safety_projection_count / self._control_step_count.clamp(min=1.0)
    )
    self.metrics["nominal_rate_recovery_fraction"][:] = (
      self._nominal_rate_recovery_count / self._control_step_count.clamp(min=1.0)
    )
    self.metrics["phase_acceleration_safety_override_fraction"][:] = (
      self._phase_acceleration_safety_override_count
      / self._control_step_count.clamp(min=1.0)
    )
    self.metrics["rate_envelope_recovery_fraction"][:] = (
      self._rate_envelope_recovery_count / self._control_step_count.clamp(min=1.0)
    )


@dataclass(kw_only=True)
class PhaseAwareMultiTargetMotionCommandCfg(MultiTargetMotionCommandCfg):
  source_fps: float = 50.0
  contact_time_step_s: float = 0.1
  contact_reward_window_s: float = 0.05
  racket_orientation_reward_window_before_s: float = 0.05
  racket_orientation_reward_window_after_s: float = 0.05
  max_contact_speedup: float = 2.0
  phase_rate_min: float = 0.0
  phase_rate_max: float = 4.0
  phase_residual_scale: float = 1.0

  def __post_init__(self) -> None:
    if self.source_fps <= 0.0:
      raise ValueError("source_fps must be positive.")
    if self.contact_time_step_s <= 0.0:
      raise ValueError("contact_time_step_s must be positive.")
    if self.contact_reward_window_s < 0.0:
      raise ValueError("contact_reward_window_s must be non-negative.")
    if self.racket_orientation_reward_window_before_s < 0.0:
      raise ValueError("Racket-orientation pre-strike window must be non-negative.")
    if self.racket_orientation_reward_window_after_s < 0.0:
      raise ValueError("Racket-orientation post-strike window must be non-negative.")
    if self.max_contact_speedup < 1.0:
      raise ValueError("max_contact_speedup must be at least 1.0.")
    if self.phase_rate_min < 0.0 or self.phase_rate_max <= self.phase_rate_min:
      raise ValueError("Invalid phase-rate limits.")

  def build(self, env: ManagerBasedRlEnv) -> PhaseAwareMultiTargetMotionCommand:
    return PhaseAwareMultiTargetMotionCommand(self, env)


@dataclass(kw_only=True)
class PhaseAccelerationMultiTargetMotionCommandCfg(
  PhaseAwareMultiTargetMotionCommandCfg
):
  joint_velocity_limits: dict[str, float] = field(default_factory=dict)
  joint_acceleration_limits: dict[str, float] = field(default_factory=dict)
  phase_acceleration_limit: float = 1.0
  joint_limit_safety_factor: float = 0.98
  safe_rate_samples_per_frame: int = 16
  next_rate_projection_samples: int = 65
  deadline_projection: bool = False
  deadline_projection_iterations: int = 12
  deadline_phase_tolerance: float = 0.005

  def __post_init__(self) -> None:
    super().__post_init__()
    if not self.joint_velocity_limits:
      raise ValueError("joint_velocity_limits must not be empty.")
    if not self.joint_acceleration_limits:
      raise ValueError("joint_acceleration_limits must not be empty.")
    if self.phase_acceleration_limit <= 0.0:
      raise ValueError("phase_acceleration_limit must be positive.")
    if not 0.0 < self.joint_limit_safety_factor <= 1.0:
      raise ValueError("joint_limit_safety_factor must be in (0, 1].")
    if self.safe_rate_samples_per_frame <= 0:
      raise ValueError("safe_rate_samples_per_frame must be positive.")
    if self.next_rate_projection_samples < 3:
      raise ValueError("next_rate_projection_samples must be at least 3.")
    if self.deadline_projection_iterations <= 0:
      raise ValueError("deadline_projection_iterations must be positive.")
    if self.deadline_phase_tolerance < 0.0:
      raise ValueError("deadline_phase_tolerance must be non-negative.")

  def build(self, env: ManagerBasedRlEnv) -> PhaseAccelerationMultiTargetMotionCommand:
    return PhaseAccelerationMultiTargetMotionCommand(self, env)
