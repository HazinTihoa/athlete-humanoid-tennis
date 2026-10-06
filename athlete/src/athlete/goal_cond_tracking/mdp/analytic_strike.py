"""Analytic tennis strike goals derived from a sampled landing target."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul


def solve_low_arc_tennis_velocity(
  strike_position_w: torch.Tensor,
  landing_position_w: torch.Tensor,
  *,
  preferred_speed: float,
  maximum_speed: float,
  net_x_w: torch.Tensor,
  net_y_w: torch.Tensor,
  net_half_width: float,
  minimum_net_height: float,
  maximum_net_height: float,
  gravity_magnitude: float = 9.81,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Solve the closest-to-preferred feasible low-arc ballistic launch.

  The returned velocity lands at ``landing_position_w``. A result is feasible
  only when its trajectory crosses the finite-width net inside the requested
  height interval without exceeding ``maximum_speed``.
  """
  if preferred_speed <= 0.0:
    raise ValueError("Preferred ball speed must be positive.")
  if maximum_speed < preferred_speed:
    raise ValueError("Maximum ball speed must not be below preferred speed.")
  if gravity_magnitude <= 0.0:
    raise ValueError("Gravity magnitude must be positive.")
  if not 0.0 < minimum_net_height < maximum_net_height:
    raise ValueError("Net height interval must be positive and ordered.")
  if net_half_width <= 0.0:
    raise ValueError("Net half-width must be positive.")
  displacement = landing_position_w - strike_position_w
  horizontal_offset = displacement[:, :2]
  horizontal_distance = torch.linalg.vector_norm(displacement[:, :2], dim=-1)
  vertical_offset = displacement[:, 2]
  squared_displacement = torch.square(horizontal_distance) + torch.square(
    vertical_offset
  )
  minimum_speed_time = torch.sqrt(
    2.0 * torch.sqrt(squared_displacement) / gravity_magnitude
  )

  def low_arc_time_for_speed(speed: float) -> tuple[torch.Tensor, torch.Tensor]:
    speed_sq = torch.full_like(horizontal_distance, speed**2)
    root_center = speed_sq - gravity_magnitude * vertical_offset
    discriminant = torch.square(root_center) - (
      gravity_magnitude**2 * squared_displacement
    )
    # This stable form is the short-flight-time (low-arc) quadratic root.
    time_sq = 2.0 * squared_displacement / (
      root_center + torch.sqrt(discriminant.clamp(min=0.0))
    ).clamp(min=1.0e-6)
    return torch.sqrt(time_sq.clamp(min=0.0)), discriminant >= 0.0

  preferred_time, preferred_reachable = low_arc_time_for_speed(preferred_speed)
  preferred_time = torch.where(
    preferred_reachable, preferred_time, minimum_speed_time
  )
  maximum_speed_time, maximum_speed_reachable = low_arc_time_for_speed(maximum_speed)

  x_displacement = displacement[:, 0]
  net_fraction = (net_x_w - strike_position_w[:, 0]) / torch.where(
    x_displacement.abs() > 1.0e-6,
    x_displacement,
    torch.ones_like(x_displacement),
  )
  linear_net_height = strike_position_w[:, 2] + net_fraction * vertical_offset
  arc_height_coefficient = (
    0.5 * gravity_magnitude * net_fraction * (1.0 - net_fraction)
  )

  minimum_height_time_sq = (
    (minimum_net_height - linear_net_height) / arc_height_coefficient.clamp(min=1.0e-6)
  ).clamp(min=0.0)
  maximum_height_time_sq = (
    (maximum_net_height - linear_net_height) / arc_height_coefficient.clamp(min=1.0e-6)
  ).clamp(min=0.0)
  minimum_allowed_time = torch.maximum(
    maximum_speed_time, torch.sqrt(minimum_height_time_sq)
  )
  maximum_allowed_time = torch.minimum(
    minimum_speed_time, torch.sqrt(maximum_height_time_sq)
  )
  selected_time = torch.minimum(
    torch.maximum(preferred_time, minimum_allowed_time), maximum_allowed_time
  )

  horizontal_velocity = horizontal_offset / selected_time.clamp(min=1.0e-6)[:, None]
  vertical_velocity = vertical_offset / selected_time.clamp(
    min=1.0e-6
  ) + 0.5 * gravity_magnitude * selected_time
  velocity_w = torch.cat((horizontal_velocity, vertical_velocity[:, None]), dim=-1)
  net_height = linear_net_height + arc_height_coefficient * torch.square(selected_time)
  lateral_at_net = (
    strike_position_w[:, 1] + net_fraction * displacement[:, 1]
  )
  selected_speed = torch.linalg.vector_norm(velocity_w, dim=-1)
  feasible = (
    (horizontal_distance > 1.0e-6)
    & (x_displacement.abs() > 1.0e-6)
    & (net_fraction > 0.0)
    & (net_fraction < 1.0)
    & ((lateral_at_net - net_y_w).abs() <= net_half_width)
    & maximum_speed_reachable
    & (minimum_allowed_time <= maximum_allowed_time)
    & (maximum_net_height >= linear_net_height)
    & (selected_speed <= maximum_speed + 1.0e-4)
    & (net_height >= minimum_net_height - 1.0e-5)
    & (net_height <= maximum_net_height + 1.0e-5)
  )
  return velocity_w, feasible, net_height


def inverse_racket_velocity_and_normal(
  incoming_ball_velocity_w: torch.Tensor,
  outgoing_ball_velocity_w: torch.Tensor,
  *,
  effective_restitution: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Invert the normal restitution equation for racket velocity and face normal."""
  if not 0.0 <= effective_restitution <= 1.0:
    raise ValueError("Effective restitution must be in [0, 1].")
  impulse_velocity = outgoing_ball_velocity_w - incoming_ball_velocity_w
  impulse_norm = torch.linalg.vector_norm(impulse_velocity, dim=-1)
  normal_w = impulse_velocity / impulse_norm.clamp(min=1.0e-6)[:, None]
  outgoing_normal_speed = torch.sum(outgoing_ball_velocity_w * normal_w, dim=-1)
  incoming_normal_speed = torch.sum(incoming_ball_velocity_w * normal_w, dim=-1)
  racket_normal_speed = (
    outgoing_normal_speed + effective_restitution * incoming_normal_speed
  ) / (1.0 + effective_restitution)
  racket_velocity_w = racket_normal_speed[:, None] * normal_w
  return racket_velocity_w, normal_w


def _shortest_arc_quaternion(
  source_vector: torch.Tensor, target_vector: torch.Tensor
) -> torch.Tensor:
  source = source_vector / torch.linalg.vector_norm(source_vector, dim=-1).clamp(
    min=1.0e-6
  )[:, None]
  target = target_vector / torch.linalg.vector_norm(target_vector, dim=-1).clamp(
    min=1.0e-6
  )[:, None]
  dot = torch.sum(source * target, dim=-1).clamp(-1.0, 1.0)
  cross = torch.linalg.cross(source, target, dim=-1)
  quaternion = torch.cat(((1.0 + dot)[:, None], cross), dim=-1)

  opposite = dot < -0.999999
  if torch.any(opposite):
    basis_x = torch.tensor([1.0, 0.0, 0.0], device=source.device, dtype=source.dtype)
    basis_y = torch.tensor([0.0, 1.0, 0.0], device=source.device, dtype=source.dtype)
    use_y = source[opposite, 0].abs() > 0.9
    basis = torch.where(use_y[:, None], basis_y, basis_x)
    axis = torch.linalg.cross(source[opposite], basis, dim=-1)
    axis = axis / torch.linalg.vector_norm(axis, dim=-1).clamp(min=1.0e-6)[:, None]
    quaternion[opposite, 0] = 0.0
    quaternion[opposite, 1:] = axis

  return quaternion / torch.linalg.vector_norm(quaternion, dim=-1).clamp(min=1.0e-6)[
    :, None
  ]


def align_quaternion_axis_to_vector(
  quaternion_w: torch.Tensor,
  local_axis: torch.Tensor,
  target_vector_w: torch.Tensor,
) -> torch.Tensor:
  """Preserve twist while rotating a quaternion's local axis onto a world vector."""
  current_axis_w = quat_apply(quaternion_w, local_axis)
  alignment = _shortest_arc_quaternion(current_axis_w, target_vector_w)
  result = quat_mul(alignment, quaternion_w)
  return result / torch.linalg.vector_norm(result, dim=-1).clamp(min=1.0e-6)[:, None]
