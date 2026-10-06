from __future__ import annotations

from types import SimpleNamespace

import torch
from athlete.goal_cond_tracking.mdp.observations import (
  motion_future_reference_window,
)
from athlete.goal_cond_tracking.mdp.phase_commands import (
  PhaseAccelerationMultiTargetMotionCommand,
  _max_phase_distance,
  _min_phase_distance,
  cubic_hermite_phase_sample,
  cubic_hermite_phase_sample_with_derivatives,
  joint_safe_phase_acceleration_bounds,
  joint_safe_phase_rate,
  phase_parent_time_steps,
  phase_segment_indices,
)


def test_phase_segment_indices_include_both_motion_endpoints() -> None:
  phase = torch.tensor([0.0, 0.5, 1.0])
  lengths = torch.tensor([6, 6, 6])

  frame0, frame1, alpha = phase_segment_indices(phase, lengths)

  torch.testing.assert_close(frame0, torch.tensor([0, 2, 4]))
  torch.testing.assert_close(frame1, torch.tensor([1, 3, 5]))
  torch.testing.assert_close(alpha, torch.tensor([0.0, 0.5, 1.0]))


def test_completed_phase_signals_one_past_motion_end_to_parent() -> None:
  phase = torch.tensor([0.0, 0.5, 1.0 - 2.0e-6, 1.0])
  lengths = torch.tensor([6, 6, 6, 6])

  before_parent_increment = phase_parent_time_steps(phase, lengths)
  after_parent_increment = before_parent_increment + 1

  torch.testing.assert_close(after_parent_increment[:3], torch.tensor([0, 2, 4]))
  torch.testing.assert_close(after_parent_increment[3:], torch.tensor([6]))
  assert torch.all(after_parent_increment[:3] < lengths[:3])
  assert torch.all(after_parent_increment[3:] >= lengths[3:])


def test_hermite_phase_derivative_obeys_chain_rule() -> None:
  source_dt = 0.02
  frames = 6
  time = torch.arange(frames, dtype=torch.float32) * source_dt
  positions = time.reshape(1, frames, 1)
  velocities = torch.ones_like(positions)
  motion_ids = torch.zeros(3, dtype=torch.long)
  phase = torch.tensor([0.0, 0.37, 1.0])
  lengths = torch.full((3,), frames, dtype=torch.long)

  value, derivative_per_phase = cubic_hermite_phase_sample(
    positions,
    velocities,
    motion_ids,
    phase,
    lengths,
    source_dt,
  )

  duration = (frames - 1) * source_dt
  torch.testing.assert_close(value[:, 0], phase * duration, atol=1.0e-6, rtol=0.0)
  torch.testing.assert_close(
    derivative_per_phase[:, 0],
    torch.full((3,), duration),
    atol=1.0e-6,
    rtol=0.0,
  )
  nominal_rate = 1.0 / duration
  torch.testing.assert_close(
    derivative_per_phase[:, 0] * (2.0 * nominal_rate),
    torch.full((3,), 2.0),
    atol=1.0e-5,
    rtol=0.0,
  )


def test_hermite_second_phase_derivative_for_quadratic_path() -> None:
  source_dt = 0.02
  frames = 6
  time = torch.arange(frames, dtype=torch.float32) * source_dt
  positions = time.square().reshape(1, frames, 1)
  velocities = (2.0 * time).reshape(1, frames, 1)
  motion_ids = torch.zeros(3, dtype=torch.long)
  phase = torch.tensor([0.0, 0.37, 1.0])
  lengths = torch.full((3,), frames, dtype=torch.long)

  value, derivative, second_derivative = cubic_hermite_phase_sample_with_derivatives(
    positions,
    velocities,
    motion_ids,
    phase,
    lengths,
    source_dt,
  )

  duration = (frames - 1) * source_dt
  torch.testing.assert_close(value[:, 0], (phase * duration).square())
  torch.testing.assert_close(
    derivative[:, 0], 2.0 * phase * duration**2, atol=1.0e-6, rtol=0.0
  )
  torch.testing.assert_close(
    second_derivative[:, 0],
    torch.full((3,), 2.0 * duration**2),
    atol=1.0e-5,
    rtol=0.0,
  )


def test_joint_safe_phase_rate_combines_velocity_and_curvature_limits() -> None:
  derivative = torch.tensor([[2.0, 1.0]])
  second_derivative = torch.tensor([[0.0, 4.0]])
  velocity_limits = torch.tensor([1.0, 10.0])
  acceleration_limits = torch.tensor([100.0, 1.0])

  safe_rate = joint_safe_phase_rate(
    derivative, second_derivative, velocity_limits, acceleration_limits
  )

  torch.testing.assert_close(safe_rate, torch.tensor([0.5]))


def test_joint_safe_phase_acceleration_bounds() -> None:
  lower, upper = joint_safe_phase_acceleration_bounds(
    derivative_per_phase=torch.tensor([[2.0]]),
    second_derivative_per_phase=torch.tensor([[0.0]]),
    phase_rate=torch.tensor([0.5]),
    joint_acceleration_limits=torch.tensor([4.0]),
    phase_acceleration_limit=3.0,
  )

  torch.testing.assert_close(lower, torch.tensor([-2.0]))
  torch.testing.assert_close(upper, torch.tensor([2.0]))


def test_phase_distance_respects_rate_and_acceleration_limits() -> None:
  maximum = _max_phase_distance(
    phase_rate=torch.tensor([0.2]),
    duration=torch.tensor([1.0]),
    phase_acceleration=torch.tensor([0.2]),
    max_phase_rate=torch.tensor([0.3]),
  )
  minimum = _min_phase_distance(
    phase_rate=torch.tensor([0.3]),
    duration=torch.tensor([1.0]),
    phase_deceleration=torch.tensor([0.2]),
    min_phase_rate=torch.tensor([0.2]),
  )

  torch.testing.assert_close(maximum, torch.tensor([0.275]))
  torch.testing.assert_close(minimum, torch.tensor([0.225]))


def test_failure_future_reference_uses_native_terminal_pause_tokens() -> None:
  command = object.__new__(PhaseAccelerationMultiTargetMotionCommand)
  command._env = SimpleNamespace(device=torch.device("cpu"))
  command.which_motion = torch.tensor([0, 1])
  command.time_steps = torch.tensor([3, 0])
  command._time_step_totals = torch.tensor([4, 4])
  command.trajectory_match_valid = torch.tensor([False, True])
  command.failure_trajectory_time_remaining = torch.tensor([1.0, 0.0])
  command._stacked_joint_pos = torch.tensor(
    [
      [[0.0], [1.0], [2.0], [3.0]],
      [[10.0], [11.0], [12.0], [13.0]],
    ]
  )
  command._stacked_joint_vel = torch.tensor(
    [
      [[0.0], [0.1], [0.2], [0.3]],
      [[1.0], [1.1], [1.2], [1.3]],
    ]
  )
  env = SimpleNamespace(
    num_envs=2,
    command_manager=SimpleNamespace(get_term=lambda _: command),
  )

  future = motion_future_reference_window(env, "motion", (1, 2)).reshape(2, 2, 2)

  torch.testing.assert_close(future[0], torch.tensor([[3.0, 0.3], [3.0, 0.3]]))
  torch.testing.assert_close(future[1], torch.tensor([[11.0, 1.1], [12.0, 1.2]]))


def test_failure_reference_extends_native_pause_without_replacing_targets() -> None:
  command = object.__new__(PhaseAccelerationMultiTargetMotionCommand)
  command._env = SimpleNamespace(step_dt=0.02)
  command.phase_hold = torch.ones(2, dtype=torch.bool)
  command.phase = torch.zeros(2)
  command.phase_rate = torch.ones(2)
  command.phase_acceleration_action = torch.ones(2)
  command.phase_acceleration = torch.ones(2)
  command.contact_time = torch.ones(2)
  command.time_remaining = torch.ones(2)
  command.strike_time_error_s = -torch.ones(2)
  command.required_phase_rate = torch.ones(2)
  command.phase_rate_schedule = torch.ones(2)
  command.safe_max_phase_rate = torch.ones(2)
  command.time_steps = torch.zeros(2, dtype=torch.long)
  command.is_paused = torch.zeros(2, dtype=torch.bool)
  command.between_motion_pause_time = torch.ones(2)
  command._sampled_pause_lengths = torch.zeros(2)
  command.failure_trajectory_time_remaining = torch.zeros(2)
  command.target_position_w = torch.randn(2, 1, 3)
  command.target_orientation_w = torch.randn(2, 1, 4)
  command.target_velocity_w = torch.randn(2, 1, 3)
  command._motion_lengths = lambda: torch.tensor([4, 5])
  command._reset_phase_diagnostics = lambda env_ids: None
  command._update_source_cache = lambda: None
  targets_before = (
    command.target_position_w.clone(),
    command.target_orientation_w.clone(),
    command.target_velocity_w.clone(),
  )

  command._configure_failure_pause_reference(
    torch.tensor([1]), torch.tensor([1.25])
  )

  assert command.phase[1].item() == 1.0
  assert command.phase_rate[1].item() == 0.0
  assert command.time_steps[1].item() == 4
  assert command.is_paused[1].item()
  assert torch.isinf(command._sampled_pause_lengths[1])
  assert command.failure_trajectory_time_remaining[1].item() == 1.25
  torch.testing.assert_close(command.target_position_w, targets_before[0])
  torch.testing.assert_close(command.target_orientation_w, targets_before[1])
  torch.testing.assert_close(command.target_velocity_w, targets_before[2])
