"""Parity tests for the fused Warp incoming-ball trajectory rollout."""

from __future__ import annotations

import torch
from athlete.goal_cond_tracking.torch_tennis_planner import (
  simulate_tennis_trajectories_torch,
)
from athlete.goal_cond_tracking.warp_tennis_planner import (
  simulate_tennis_trajectories_warp_fused,
)
from athlete.scripts.tennis_physics import STANDARD_TENNIS_PHYSICS


def test_warp_fused_rollout_matches_torch_on_cpu() -> None:
  physics = STANDARD_TENNIS_PHYSICS
  initial_positions = torch.tensor(
    [[9.2, -0.5, 1.4], [8.4, 0.7, 1.8]], dtype=torch.float32
  )
  initial_velocities = torch.tensor(
    [[-4.8, 0.4, 3.8], [-3.9, -0.6, 4.5]], dtype=torch.float32
  )
  mass = torch.tensor([0.055, 0.059], dtype=torch.float32)
  restitution = torch.tensor([0.72, 0.78], dtype=torch.float32)
  tangent_retention = torch.tensor([0.80, 0.86], dtype=torch.float32)
  drag = torch.tensor(
    [physics.ball.drag_coefficient * 0.9, physics.ball.drag_coefficient * 1.1],
    dtype=torch.float32,
  )
  kwargs = {
    "ball_mass_kg": mass,
    "court_restitution": restitution,
    "tangent_speed_retention": tangent_retention,
    "drag_coefficient": drag,
    "dt": 0.01,
    "horizon_s": 4.0,
  }

  torch_rollout = simulate_tennis_trajectories_torch(
    initial_positions, initial_velocities, **kwargs
  )
  warp_rollout = simulate_tennis_trajectories_warp_fused(
    initial_positions, initial_velocities, **kwargs
  )

  torch.testing.assert_close(warp_rollout.times, torch_rollout.times)
  torch.testing.assert_close(
    warp_rollout.positions, torch_rollout.positions, atol=1e-5, rtol=1e-5
  )
  torch.testing.assert_close(
    warp_rollout.velocities, torch_rollout.velocities, atol=1e-5, rtol=1e-5
  )
  assert torch.equal(warp_rollout.bounce_counts, torch_rollout.bounce_counts)
  assert torch.equal(warp_rollout.net_cleared, torch_rollout.net_cleared)


def test_warp_fused_rollout_accepts_expanded_fallback_inputs() -> None:
  count = 3
  positions = torch.tensor([[9.0, 0.0, 1.5]], dtype=torch.float32).expand(
    count, -1
  )
  velocities = torch.tensor([[-4.0, 0.0, 4.0]], dtype=torch.float32).expand(
    count, -1
  )
  parameters = torch.full((count,), 0.0577, dtype=torch.float32)

  rollout = simulate_tennis_trajectories_warp_fused(
    positions,
    velocities,
    ball_mass_kg=parameters,
    court_restitution=torch.full((count,), 0.745),
    tangent_speed_retention=torch.full((count,), 0.825),
    drag_coefficient=torch.full((count,), 0.55),
    dt=0.01,
    horizon_s=0.1,
  )

  assert rollout.positions.shape == (count, 11, 3)
  assert torch.all(torch.isfinite(rollout.positions))
