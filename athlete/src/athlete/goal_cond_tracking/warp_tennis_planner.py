"""Single-kernel Warp rollout for batched incoming tennis trajectories."""

from __future__ import annotations

import math

import torch
import warp as wp

from athlete.goal_cond_tracking.torch_tennis_planner import (
  TorchTennisTrajectoryBatch,
)
from athlete.scripts.tennis_physics import STANDARD_TENNIS_PHYSICS


@wp.kernel
def _simulate_tennis_trajectories_kernel(
  initial_positions: wp.array(dtype=wp.vec3),
  initial_velocities: wp.array(dtype=wp.vec3),
  ball_mass_kg: wp.array(dtype=wp.float32),
  court_restitution: wp.array(dtype=wp.float32),
  tangent_speed_retention: wp.array(dtype=wp.float32),
  drag_coefficient: wp.array(dtype=wp.float32),
  steps: int,
  dt: float,
  ground_z: float,
  net_x: float,
  net_height_with_ball: float,
  net_half_width: float,
  air_density: float,
  ball_cross_section: float,
  stokes_drag_scale: float,
  positions: wp.array(dtype=wp.vec3, ndim=2),
  velocities: wp.array(dtype=wp.vec3, ndim=2),
  bounce_counts: wp.array(dtype=wp.int8, ndim=2),
  net_cleared: wp.array(dtype=wp.int8),
):
  trajectory_id = wp.tid()
  position = initial_positions[trajectory_id]
  velocity = initial_velocities[trajectory_id]
  bounce_count = wp.int32(0)
  cleared = wp.int8(1)
  positions[trajectory_id, 0] = position
  velocities[trajectory_id, 0] = velocity
  bounce_counts[trajectory_id, 0] = wp.int8(0)

  mass = ball_mass_kg[trajectory_id]
  restitution = court_restitution[trajectory_id]
  tangent_retention = tangent_speed_retention[trajectory_id]
  drag = drag_coefficient[trajectory_id]
  gravity = wp.vec3(0.0, 0.0, -9.81)
  eps = 1.0e-7

  for step in range(1, steps + 1):
    speed = wp.length(velocity)
    drag_scale = (
      0.5 * air_density * drag * ball_cross_section * speed + stokes_drag_scale
    )
    acceleration = gravity - velocity * (drag_scale / mass)
    next_velocity = velocity + acceleration * dt
    next_position = position + next_velocity * dt

    if position[0] > net_x and next_position[0] <= net_x:
      denominator = wp.max(position[0] - next_position[0], eps)
      alpha = wp.clamp((position[0] - net_x) / denominator, 0.0, 1.0)
      net_position = position + (next_position - position) * alpha
      if (
        wp.abs(net_position[1]) <= net_half_width
        and net_position[2] <= net_height_with_ball
      ):
        cleared = wp.int8(0)

    if (
      bounce_count < 2
      and next_position[2] < ground_z
      and next_velocity[2] < 0.0
    ):
      denominator = wp.max(position[2] - next_position[2], eps)
      fraction = wp.clamp((position[2] - ground_z) / denominator, 0.0, 1.0)
      impact_position = position + (next_position - position) * fraction
      next_velocity = wp.vec3(
        next_velocity[0] * tangent_retention,
        next_velocity[1] * tangent_retention,
        -next_velocity[2] * restitution,
      )
      remaining_dt = (1.0 - fraction) * dt
      next_position = impact_position + next_velocity * remaining_dt
      next_position = wp.vec3(
        next_position[0], next_position[1], wp.max(next_position[2], ground_z)
      )
      bounce_count += 1

    position = next_position
    velocity = next_velocity
    positions[trajectory_id, step] = position
    velocities[trajectory_id, step] = velocity
    bounce_counts[trajectory_id, step] = wp.int8(bounce_count)

  net_cleared[trajectory_id] = cleared


def _validate_inputs(
  initial_positions: torch.Tensor,
  initial_velocities: torch.Tensor,
  ball_mass_kg: torch.Tensor,
  court_restitution: torch.Tensor,
  tangent_speed_retention: torch.Tensor,
  drag_coefficient: torch.Tensor,
  dt: float,
  horizon_s: float,
) -> None:
  if initial_positions.ndim != 2 or initial_positions.shape[-1] != 3:
    raise ValueError("initial_positions must have shape (N, 3).")
  if initial_velocities.shape != initial_positions.shape:
    raise ValueError("initial_velocities must match initial_positions.")
  if initial_positions.dtype != torch.float32:
    raise TypeError("Warp trajectory rollout requires float32 tensors.")
  if not initial_positions.is_contiguous() or not initial_velocities.is_contiguous():
    raise ValueError("Warp trajectory inputs must be contiguous.")
  if initial_velocities.device != initial_positions.device:
    raise ValueError("All Warp trajectory tensors must use the same device.")
  if dt <= 0.0 or horizon_s <= 0.0:
    raise ValueError("dt and horizon_s must be positive.")
  count = initial_positions.shape[0]
  for name, value in (
    ("ball_mass_kg", ball_mass_kg),
    ("court_restitution", court_restitution),
    ("tangent_speed_retention", tangent_speed_retention),
    ("drag_coefficient", drag_coefficient),
  ):
    if value.shape != (count,):
      raise ValueError(f"{name} must have shape ({count},).")
    if value.dtype != torch.float32 or value.device != initial_positions.device:
      raise TypeError(f"{name} must be float32 on {initial_positions.device}.")
    if not value.is_contiguous():
      raise ValueError(f"{name} must be contiguous.")
  if torch.any(ball_mass_kg <= 0.0):
    raise ValueError("ball_mass_kg must be positive.")
  if torch.any((court_restitution < 0.0) | (court_restitution > 1.0)):
    raise ValueError("court_restitution must lie in [0, 1].")
  if torch.any(
    (tangent_speed_retention < 0.0) | (tangent_speed_retention > 1.0)
  ):
    raise ValueError("tangent_speed_retention must lie in [0, 1].")


def simulate_tennis_trajectories_warp_fused(
  initial_positions: torch.Tensor,
  initial_velocities: torch.Tensor,
  *,
  ball_mass_kg: torch.Tensor,
  court_restitution: torch.Tensor,
  tangent_speed_retention: torch.Tensor,
  drag_coefficient: torch.Tensor,
  dt: float,
  horizon_s: float,
  ground_z: float | None = None,
  net_x: float | None = None,
  net_height: float | None = None,
  net_half_width: float | None = None,
) -> TorchTennisTrajectoryBatch:
  """Roll trajectories in one Warp launch; contact-boundary results are not bit exact."""
  initial_positions = initial_positions.contiguous()
  initial_velocities = initial_velocities.contiguous()
  ball_mass_kg = ball_mass_kg.contiguous()
  court_restitution = court_restitution.contiguous()
  tangent_speed_retention = tangent_speed_retention.contiguous()
  drag_coefficient = drag_coefficient.contiguous()
  _validate_inputs(
    initial_positions,
    initial_velocities,
    ball_mass_kg,
    court_restitution,
    tangent_speed_retention,
    drag_coefficient,
    dt,
    horizon_s,
  )
  wp.init()
  physics = STANDARD_TENNIS_PHYSICS
  ground = physics.ball.radius_m if ground_z is None else float(ground_z)
  court_net_x = physics.court.net_x_m if net_x is None else float(net_x)
  court_net_height = (
    physics.court.net_height_m if net_height is None else float(net_height)
  )
  court_net_half_width = (
    physics.court.net_half_width_m
    if net_half_width is None
    else float(net_half_width)
  )
  steps = max(1, round(horizon_s / dt))
  count = initial_positions.shape[0]
  device = initial_positions.device
  positions = torch.empty((count, steps + 1, 3), dtype=torch.float32, device=device)
  velocities = torch.empty_like(positions)
  bounce_counts = torch.empty((count, steps + 1), dtype=torch.int8, device=device)
  net_cleared = torch.empty(count, dtype=torch.int8, device=device)

  stream = None
  if device.type == "cuda":
    stream = wp.stream_from_torch(torch.cuda.current_stream(device))
  wp.launch(
    _simulate_tennis_trajectories_kernel,
    dim=count,
    inputs=[
      wp.from_torch(initial_positions, dtype=wp.vec3),
      wp.from_torch(initial_velocities, dtype=wp.vec3),
      wp.from_torch(ball_mass_kg),
      wp.from_torch(court_restitution),
      wp.from_torch(tangent_speed_retention),
      wp.from_torch(drag_coefficient),
      steps,
      dt,
      ground,
      court_net_x,
      court_net_height + physics.ball.radius_m,
      court_net_half_width,
      physics.air.density_kg_m3,
      physics.ball.cross_section_m2,
      3.0
      * math.pi
      * physics.ball.diameter_m
      * physics.air.dynamic_viscosity_pa_s,
    ],
    outputs=[
      wp.from_torch(positions, dtype=wp.vec3),
      wp.from_torch(velocities, dtype=wp.vec3),
      wp.from_torch(bounce_counts),
      wp.from_torch(net_cleared),
    ],
    device=str(device),
    stream=stream,
  )
  times = torch.arange(steps + 1, dtype=torch.float32, device=device) * dt
  return TorchTennisTrajectoryBatch(
    times=times,
    positions=positions,
    velocities=velocities,
    bounce_counts=bounce_counts,
    net_cleared=net_cleared.bool(),
  )
