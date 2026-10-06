"""Batched Torch trajectory rollout and motion matching for incoming tennis balls."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from athlete.scripts.tennis_physics import (
  STANDARD_TENNIS_PHYSICS,
  tennis_ball_aerodynamic_wrench_torch,
)


@dataclass(frozen=True)
class TorchTennisTrajectoryBatch:
  """Fixed-step incoming trajectories and their contact-state masks."""

  times: torch.Tensor
  positions: torch.Tensor
  velocities: torch.Tensor
  bounce_counts: torch.Tensor
  net_cleared: torch.Tensor
  valid_mask: torch.Tensor | None = None

  @property
  def post_first_bounce_mask(self) -> torch.Tensor:
    mask = (self.bounce_counts == 1) & self.net_cleared[:, None]
    return mask if self.valid_mask is None else mask & self.valid_mask


@dataclass(frozen=True)
class TorchMotionMatchBatch:
  """Best motion and discrete trajectory intercept for each environment."""

  motion_ids: torch.Tensor
  trajectory_indices: torch.Tensor
  target_positions: torch.Tensor
  contact_times: torch.Tensor
  distances: torch.Tensor
  valid: torch.Tensor


@dataclass(frozen=True)
class TennisFailureTrajectoryBatch:
  """Physical failure labels and planar closest-approach diagnostics."""

  failure: torch.Tensor
  did_not_clear_net: torch.Tensor
  trajectory_too_far: torch.Tensor
  closest_trajectory_positions: torch.Tensor
  closest_trajectory_distances: torch.Tensor


def classify_tennis_failure_trajectories(
  trajectories: TorchTennisTrajectoryBatch,
  root_positions_xy: torch.Tensor,
  *,
  trajectory_xy_distance_threshold_m: float,
  net_x: float | None = None,
  net_height: float | None = None,
  net_half_width: float | None = None,
) -> TennisFailureTrajectoryBatch:
  """Classify no-net and over-net trajectories that never approach the robot."""
  if root_positions_xy.shape != (len(trajectories.positions), 2):
    raise ValueError("root_positions_xy must have shape (N, 2).")
  if trajectory_xy_distance_threshold_m <= 0.0:
    raise ValueError("Failure trajectory XY-distance threshold must be positive.")

  physics = STANDARD_TENNIS_PHYSICS
  court_net_x = physics.court.net_x_m if net_x is None else float(net_x)
  court_net_height = (
    physics.court.net_height_m if net_height is None else float(net_height)
  )
  court_net_half_width = (
    physics.court.net_half_width_m
    if net_half_width is None
    else float(net_half_width)
  )
  positions = trajectories.positions
  before = positions[:, :-1]
  after = positions[:, 1:]
  crosses_net_plane = (before[..., 0] > court_net_x) & (
    after[..., 0] <= court_net_x
  )
  if trajectories.valid_mask is not None:
    crosses_net_plane &= trajectories.valid_mask[:, :-1] & trajectories.valid_mask[:, 1:]
  denominator = (before[..., 0] - after[..., 0]).clamp_min(
    torch.finfo(positions.dtype).eps
  )
  alpha = ((before[..., 0] - court_net_x) / denominator).clamp(0.0, 1.0)
  crossing_position = before + alpha[..., None] * (after - before)
  before_first_bounce = trajectories.bounce_counts[:, :-1] == 0
  clears_net_step = (
    crosses_net_plane
    & before_first_bounce
    & (crossing_position[..., 1].abs() <= court_net_half_width)
    & (crossing_position[..., 2] > court_net_height + physics.ball.radius_m)
  )
  cleared_net = clears_net_step.any(dim=1) & trajectories.net_cleared

  strike_segment = (trajectories.bounce_counts == 1) & cleared_net[:, None]
  if trajectories.valid_mask is not None:
    strike_segment &= trajectories.valid_mask
  has_strike_segment = strike_segment.any(dim=1)
  trajectory_xy_distances = torch.linalg.vector_norm(
    positions[..., :2] - root_positions_xy[:, None, :],
    dim=-1,
  )
  masked_distances = trajectory_xy_distances.masked_fill(~strike_segment, torch.inf)
  closest_trajectory_distances, closest_indices = masked_distances.min(dim=1)
  batch_indices = torch.arange(len(positions), device=positions.device)
  closest_trajectory_positions = positions[batch_indices, closest_indices].clone()
  closest_trajectory_positions = torch.where(
    has_strike_segment[:, None],
    closest_trajectory_positions,
    torch.full_like(closest_trajectory_positions, torch.nan),
  )

  did_not_clear_net = ~cleared_net
  trajectory_too_far = (
    cleared_net
    & has_strike_segment
    & (closest_trajectory_distances > trajectory_xy_distance_threshold_m)
  )
  return TennisFailureTrajectoryBatch(
    failure=did_not_clear_net | trajectory_too_far,
    did_not_clear_net=did_not_clear_net,
    trajectory_too_far=trajectory_too_far,
    closest_trajectory_positions=closest_trajectory_positions,
    closest_trajectory_distances=closest_trajectory_distances,
  )


def sample_root_directed_tennis_launches_torch(
  root_positions_xy: torch.Tensor,
  *,
  launch_position_min: torch.Tensor,
  launch_position_max: torch.Tensor,
  horizontal_speed_range_m_s: tuple[float, float],
  horizontal_angle_half_width_deg: float,
  net_crossing_height_range_m: tuple[float, float],
  maximum_initial_speed_m_s: float,
  net_x_m: float | None = None,
  maximum_resample_rounds: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Sample fixed-court launches aimed at each root with a ballistic vz bound."""
  if root_positions_xy.ndim != 2 or root_positions_xy.shape[1] != 2:
    raise ValueError("root_positions_xy must have shape (N, 2).")
  if launch_position_min.shape != (3,) or launch_position_max.shape != (3,):
    raise ValueError("Launch position bounds must have shape (3,).")
  if launch_position_min.device != root_positions_xy.device or (
    launch_position_max.device != root_positions_xy.device
  ):
    raise ValueError("Launch bounds and root positions must use the same device.")
  if launch_position_min.dtype != root_positions_xy.dtype or (
    launch_position_max.dtype != root_positions_xy.dtype
  ):
    raise ValueError("Launch bounds and root positions must use the same dtype.")
  speed_min, speed_max = horizontal_speed_range_m_s
  height_min, height_max = net_crossing_height_range_m
  if not 0.0 < speed_min <= speed_max:
    raise ValueError("Horizontal speed bounds must be positive and ordered.")
  if not 0.0 < height_min <= height_max:
    raise ValueError("Net crossing height bounds must be positive and ordered.")
  if maximum_initial_speed_m_s <= speed_max:
    raise ValueError("Maximum initial speed must exceed horizontal speed maximum.")
  if not 0.0 <= horizontal_angle_half_width_deg < 90.0:
    raise ValueError("Horizontal angle half width must lie in [0, 90).")
  if maximum_resample_rounds <= 0:
    raise ValueError("maximum_resample_rounds must be positive.")

  count = len(root_positions_xy)
  positions = torch.empty(
    (count, 3), device=root_positions_xy.device, dtype=root_positions_xy.dtype
  )
  velocities = torch.empty_like(positions)
  pending = torch.arange(count, device=root_positions_xy.device)
  net_x = (
    STANDARD_TENNIS_PHYSICS.court.net_x_m
    if net_x_m is None
    else float(net_x_m)
  )
  angle_half_width = math.radians(horizontal_angle_half_width_deg)

  for _ in range(maximum_resample_rounds):
    if pending.numel() == 0:
      break
    pending_count = pending.numel()
    position = launch_position_min + torch.rand(
      (pending_count, 3),
      device=root_positions_xy.device,
      dtype=root_positions_xy.dtype,
    ) * (launch_position_max - launch_position_min)
    direction = root_positions_xy[pending] - position[:, :2]
    direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(
      1.0e-6
    )
    angle = (
      torch.rand(
        pending_count,
        device=root_positions_xy.device,
        dtype=root_positions_xy.dtype,
      )
      * 2.0
      - 1.0
    ) * angle_half_width
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    direction = torch.stack(
      (
        cos_angle * direction[:, 0] - sin_angle * direction[:, 1],
        sin_angle * direction[:, 0] + cos_angle * direction[:, 1],
      ),
      dim=1,
    )
    horizontal_speed = speed_min + torch.rand(
      pending_count,
      device=root_positions_xy.device,
      dtype=root_positions_xy.dtype,
    ) * (speed_max - speed_min)
    horizontal_velocity = direction * horizontal_speed[:, None]
    vx = horizontal_velocity[:, 0]
    time_to_net = (net_x - position[:, 0]) / vx.clamp(max=-1.0e-6)
    crosses_net_forward = (vx < 0.0) & (time_to_net > 0.0)

    vz_min = (
      height_min - position[:, 2] + 0.5 * 9.81 * time_to_net.square()
    ) / time_to_net.clamp_min(1.0e-6)
    vz_max = (
      height_max - position[:, 2] + 0.5 * 9.81 * time_to_net.square()
    ) / time_to_net.clamp_min(1.0e-6)
    vertical_speed_cap = torch.sqrt(
      (maximum_initial_speed_m_s**2 - horizontal_speed.square()).clamp_min(0.0)
    )
    lower = torch.maximum(vz_min, -vertical_speed_cap)
    upper = torch.minimum(vz_max, vertical_speed_cap)
    feasible = crosses_net_forward & (lower <= upper)
    accepted = pending[feasible]
    sampled_vz = lower[feasible] + torch.rand(
      accepted.numel(),
      device=root_positions_xy.device,
      dtype=root_positions_xy.dtype,
    ) * (upper[feasible] - lower[feasible])
    positions[accepted] = position[feasible]
    velocities[accepted, :2] = horizontal_velocity[feasible]
    velocities[accepted, 2] = sampled_vz
    pending = pending[~feasible]

  if pending.numel() > 0:
    # A candidate-level rejection must not terminate distributed training. The
    # downstream trajectory matcher already handles invalid candidates and env
    # retries, so provide a finite speed-bounded fallback for the rare samples
    # that exhaust the analytic rejection budget.
    fallback_position = ((launch_position_min + launch_position_max) * 0.5).expand(
      pending.numel(), -1
    )
    fallback_direction = root_positions_xy[pending] - fallback_position[:, :2]
    fallback_norm = torch.linalg.vector_norm(
      fallback_direction, dim=1, keepdim=True
    )
    default_direction = torch.tensor(
      [-1.0, 0.0],
      device=root_positions_xy.device,
      dtype=root_positions_xy.dtype,
    ).expand_as(fallback_direction)
    fallback_direction = torch.where(
      fallback_norm > 1.0e-6,
      fallback_direction / fallback_norm.clamp_min(1.0e-6),
      default_direction,
    )
    fallback_horizontal_velocity = fallback_direction * speed_min
    fallback_vx = fallback_horizontal_velocity[:, 0]
    fallback_crosses_net = fallback_vx < -1.0e-6
    fallback_time_to_net = (net_x - fallback_position[:, 0]) / torch.where(
      fallback_crosses_net,
      fallback_vx,
      torch.full_like(fallback_vx, -1.0e-6),
    )
    target_height = 0.5 * (height_min + height_max)
    required_vz = (
      target_height
      - fallback_position[:, 2]
      + 0.5 * 9.81 * fallback_time_to_net.square()
    ) / fallback_time_to_net.clamp_min(1.0e-6)
    vertical_speed_cap = math.sqrt(
      max(maximum_initial_speed_m_s**2 - speed_min**2, 0.0)
    )
    fallback_vz = torch.where(
      fallback_crosses_net,
      required_vz.clamp(-vertical_speed_cap, vertical_speed_cap),
      torch.zeros_like(required_vz),
    )
    positions[pending] = fallback_position
    velocities[pending, :2] = fallback_horizontal_velocity
    velocities[pending, 2] = fallback_vz
  return positions, velocities


def retarget_tennis_launches_to_miss_roots_torch(
  initial_positions: torch.Tensor,
  initial_velocities: torch.Tensor,
  root_positions_xy: torch.Tensor,
  *,
  minimum_xy_distance_m: float,
  net_crossing_height_range_m: tuple[float, float],
  maximum_initial_speed_m_s: float,
  net_x_m: float,
) -> torch.Tensor:
  """Aim launches along paths whose XY lines miss each root by a margin."""
  if initial_positions.ndim != 2 or initial_positions.shape[1] != 3:
    raise ValueError("initial_positions must have shape (N, 3).")
  if initial_velocities.shape != initial_positions.shape:
    raise ValueError("initial_velocities must match initial_positions.")
  if root_positions_xy.shape != (len(initial_positions), 2):
    raise ValueError("root_positions_xy must have shape (N, 2).")
  if minimum_xy_distance_m <= 0.0:
    raise ValueError("Minimum trajectory XY distance must be positive.")

  velocity = initial_velocities.clone()
  launch_from_root = initial_positions[:, :2] - root_positions_xy
  launch_distance = torch.linalg.vector_norm(launch_from_root, dim=1)
  direct_to_root = -launch_from_root / launch_distance[:, None].clamp_min(1.0e-6)
  minimum_miss = torch.full_like(launch_distance, minimum_xy_distance_m + 0.25)
  maximum_miss = torch.minimum(
    launch_distance * 0.8,
    torch.full_like(launch_distance, minimum_xy_distance_m + 1.0),
  )
  miss_distance = minimum_miss + torch.rand_like(launch_distance) * (
    maximum_miss - minimum_miss
  ).clamp_min(0.0)
  miss_distance = torch.minimum(miss_distance, launch_distance * 0.8)
  angle = torch.asin((miss_distance / launch_distance.clamp_min(1.0e-6)).clamp(0.0, 0.8))
  side = torch.where(
    torch.rand_like(angle) < 0.5,
    -torch.ones_like(angle),
    torch.ones_like(angle),
  )
  sin_angle = torch.sin(side * angle)
  cos_angle = torch.cos(angle)
  miss_direction = torch.stack(
    (
      cos_angle * direct_to_root[:, 0] - sin_angle * direct_to_root[:, 1],
      sin_angle * direct_to_root[:, 0] + cos_angle * direct_to_root[:, 1],
    ),
    dim=1,
  )
  horizontal_speed = torch.linalg.vector_norm(initial_velocities[:, :2], dim=1)
  velocity[:, :2] = miss_direction * horizontal_speed[:, None]

  time_to_net = (
    (net_x_m - initial_positions[:, 0]) / velocity[:, 0].clamp(max=-1.0e-6)
  ).clamp_min(1.0e-6)
  height_min, height_max = net_crossing_height_range_m
  crossing_height = height_min + torch.rand_like(time_to_net) * (
    height_max - height_min
  )
  required_vz = (
    crossing_height
    - initial_positions[:, 2]
    + 0.5 * 9.81 * time_to_net.square()
  ) / time_to_net
  vertical_speed_cap = torch.sqrt(
    (
      maximum_initial_speed_m_s**2 - torch.square(horizontal_speed)
    ).clamp_min(0.0)
  )
  velocity[:, 2] = required_vz.clamp(-vertical_speed_cap, vertical_speed_cap)
  return velocity


def randomize_tennis_launch_directions_torch(
  initial_positions: torch.Tensor,
  *,
  horizontal_speed_range_m_s: tuple[float, float],
  vertical_speed_range_m_s: tuple[float, float],
  maximum_initial_speed_m_s: float,
) -> torch.Tensor:
  """Give valid launch points full-azimuth, speed-bounded flight directions."""
  if initial_positions.ndim != 2 or initial_positions.shape[1] != 3:
    raise ValueError("initial_positions must have shape (N, 3).")
  horizontal_min, horizontal_max = horizontal_speed_range_m_s
  vertical_min, vertical_max = vertical_speed_range_m_s
  if not 0.0 < horizontal_min <= horizontal_max:
    raise ValueError("Horizontal launch-speed bounds must be positive and ordered.")
  if vertical_min > vertical_max:
    raise ValueError("Vertical launch-speed bounds must be ordered.")
  if maximum_initial_speed_m_s <= horizontal_max:
    raise ValueError("Maximum speed must exceed the horizontal speed maximum.")

  count = len(initial_positions)
  dtype, device = initial_positions.dtype, initial_positions.device
  horizontal_speed = horizontal_min + torch.rand(
    count, dtype=dtype, device=device
  ) * (horizontal_max - horizontal_min)
  azimuth = torch.rand(count, dtype=dtype, device=device) * (2.0 * math.pi)
  vertical_speed_cap = torch.sqrt(
    (maximum_initial_speed_m_s**2 - horizontal_speed.square()).clamp_min(0.0)
  )
  lower = torch.maximum(
    torch.full_like(vertical_speed_cap, vertical_min), -vertical_speed_cap
  )
  upper = torch.minimum(
    torch.full_like(vertical_speed_cap, vertical_max), vertical_speed_cap
  )
  if torch.any(lower > upper):
    raise ValueError("Vertical speed bounds exceed the configured total speed cap.")
  vertical_speed = lower + torch.rand_like(lower) * (upper - lower)
  velocity = torch.empty_like(initial_positions)
  velocity[:, 0] = horizontal_speed * torch.cos(azimuth)
  velocity[:, 1] = horizontal_speed * torch.sin(azimuth)
  velocity[:, 2] = vertical_speed
  return velocity


def simulate_tennis_trajectories_torch(
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
  stop_origin_w: torch.Tensor | None = None,
  stop_distance_m: float | None = None,
  stop_speed_m_s: float | None = None,
) -> TorchTennisTrajectoryBatch:
  """Roll independent tennis balls with fixed-step semi-implicit Euler updates."""
  if initial_positions.ndim != 2 or initial_positions.shape[-1] != 3:
    raise ValueError("initial_positions must have shape (N, 3).")
  if initial_velocities.shape != initial_positions.shape:
    raise ValueError("initial_velocities must match initial_positions.")
  if dt <= 0.0 or horizon_s <= 0.0:
    raise ValueError("dt and horizon_s must be positive.")
  if stop_origin_w is not None and stop_origin_w.shape != initial_positions.shape:
    raise ValueError("stop_origin_w must match initial_positions.")
  if stop_distance_m is not None and stop_distance_m <= 0.0:
    raise ValueError("stop_distance_m must be positive.")
  if stop_speed_m_s is not None and stop_speed_m_s < 0.0:
    raise ValueError("stop_speed_m_s must be non-negative.")
  count = initial_positions.shape[0]
  for name, value in (
    ("ball_mass_kg", ball_mass_kg),
    ("court_restitution", court_restitution),
    ("tangent_speed_retention", tangent_speed_retention),
    ("drag_coefficient", drag_coefficient),
  ):
    if value.shape != (count,):
      raise ValueError(f"{name} must have shape ({count},).")
  if torch.any(ball_mass_kg <= 0.0):
    raise ValueError("ball_mass_kg must be positive.")
  if torch.any((court_restitution < 0.0) | (court_restitution > 1.0)):
    raise ValueError("court_restitution must lie in [0, 1].")
  if torch.any(
    (tangent_speed_retention < 0.0) | (tangent_speed_retention > 1.0)
  ):
    raise ValueError("tangent_speed_retention must lie in [0, 1].")

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
  dtype = initial_positions.dtype
  device = initial_positions.device
  times = torch.arange(steps + 1, dtype=dtype, device=device) * dt
  positions = torch.empty((count, steps + 1, 3), dtype=dtype, device=device)
  velocities = torch.empty_like(positions)
  bounce_counts = torch.empty(
    (count, steps + 1), dtype=torch.int8, device=device
  )
  valid_mask = torch.ones((count, steps + 1), dtype=torch.bool, device=device)

  position = initial_positions.clone()
  velocity = initial_velocities.clone()
  bounce_count = torch.zeros(count, dtype=torch.int8, device=device)
  net_cleared = torch.ones(count, dtype=torch.bool, device=device)
  active = torch.ones(count, dtype=torch.bool, device=device)
  angular_velocity = torch.zeros_like(velocity)
  positions[:, 0] = position
  velocities[:, 0] = velocity
  bounce_counts[:, 0] = bounce_count
  initial_out_of_range = torch.zeros_like(active)
  if stop_origin_w is not None and stop_distance_m is not None:
    initial_out_of_range = torch.linalg.vector_norm(
      position - stop_origin_w, dim=-1
    ) >= stop_distance_m
  initial_stopped = torch.zeros_like(active)
  if stop_speed_m_s is not None:
    initial_stopped = torch.linalg.vector_norm(velocity, dim=-1) <= stop_speed_m_s
  active &= ~(initial_out_of_range | initial_stopped)

  gravity = torch.tensor((0.0, 0.0, -9.81), dtype=dtype, device=device)
  eps = torch.finfo(dtype).eps
  for step in range(1, steps + 1):
    force, _ = tennis_ball_aerodynamic_wrench_torch(
      velocity,
      angular_velocity,
      drag_coefficient=drag_coefficient,
      magnus_coefficient=0.0,
    )
    acceleration = gravity + force / ball_mass_kg[:, None]
    next_velocity = velocity + acceleration * dt
    next_position = position + next_velocity * dt

    crosses_net = (position[:, 0] > court_net_x) & (
      next_position[:, 0] <= court_net_x
    )
    net_alpha = (position[:, 0] - court_net_x) / (
      position[:, 0] - next_position[:, 0]
    ).clamp_min(eps)
    net_alpha.clamp_(0.0, 1.0)
    net_position = position + net_alpha[:, None] * (next_position - position)
    hits_net = (
      crosses_net
      & (net_position[:, 1].abs() <= court_net_half_width)
      & (net_position[:, 2] <= court_net_height + physics.ball.radius_m)
    )
    hits_net &= active
    net_cleared &= ~hits_net

    bounced = active & (bounce_count < 2) & (next_position[:, 2] < ground) & (
      next_velocity[:, 2] < 0.0
    )
    crossing_fraction = (position[:, 2] - ground) / (
      position[:, 2] - next_position[:, 2]
    ).clamp_min(eps)
    crossing_fraction.clamp_(0.0, 1.0)
    impact_position = position + crossing_fraction[:, None] * (
      next_position - position
    )
    reflected_velocity = next_velocity.clone()
    reflected_velocity[:, :2] *= tangent_speed_retention[:, None]
    reflected_velocity[:, 2] *= -court_restitution
    remaining_dt = (1.0 - crossing_fraction) * dt
    rebound_position = impact_position + reflected_velocity * remaining_dt[:, None]
    rebound_position[:, 2].clamp_(min=ground)
    next_position = torch.where(bounced[:, None], rebound_position, next_position)
    next_velocity = torch.where(
      bounced[:, None], reflected_velocity, next_velocity
    )
    bounce_count = bounce_count + bounced.to(torch.int8)

    next_position = torch.where(active[:, None], next_position, position)
    next_velocity = torch.where(active[:, None], next_velocity, velocity)
    position = next_position
    velocity = next_velocity
    positions[:, step] = position
    velocities[:, step] = velocity
    bounce_counts[:, step] = bounce_count
    # Keep the threshold-crossing sample, but exclude all later held samples.
    valid_mask[:, step] = active
    if stop_origin_w is not None and stop_distance_m is not None:
      out_of_observation_range = torch.linalg.vector_norm(
        position - stop_origin_w, dim=-1
      ) >= stop_distance_m
    else:
      out_of_observation_range = torch.zeros_like(active)
    if stop_speed_m_s is not None:
      stopped = torch.linalg.vector_norm(velocity, dim=-1) <= stop_speed_m_s
    else:
      stopped = torch.zeros_like(active)
    active &= ~(out_of_observation_range | stopped)

  return TorchTennisTrajectoryBatch(
    times=times,
    positions=positions,
    velocities=velocities,
    bounce_counts=bounce_counts,
    net_cleared=net_cleared,
    valid_mask=valid_mask,
  )


def match_tennis_trajectories_to_motions_torch(
  trajectories: TorchTennisTrajectoryBatch,
  motion_target_positions: torch.Tensor,
  minimum_contact_times: torch.Tensor,
  nominal_contact_times: torch.Tensor,
  *,
  maximum_distance: float,
  motion_chunk_size: int = 25,
  contact_time_offset_s: float = 0.0,
  hierarchical_top_k: int = 0,
  hierarchical_coarse_neighbor_radius: int = 1,
  include_pre_bounce: bool = False,
) -> TorchMotionMatchBatch:
  """Match post-bounce trajectory samples to spatially and temporally valid motions.

  When ``hierarchical_top_k`` is positive and smaller than the motion library,
  a strided trajectory first ranks every motion.  The original full-resolution
  search then runs only over the selected motions for each trajectory.
  """
  if maximum_distance < 0.0:
    raise ValueError("maximum_distance must be non-negative.")
  if motion_chunk_size <= 0:
    raise ValueError("motion_chunk_size must be positive.")
  if hierarchical_top_k < 0:
    raise ValueError("hierarchical_top_k must be non-negative.")
  if hierarchical_coarse_neighbor_radius < 0:
    raise ValueError("hierarchical_coarse_neighbor_radius must be non-negative.")
  if not math.isfinite(contact_time_offset_s) or contact_time_offset_s < 0.0:
    raise ValueError("contact_time_offset_s must be finite and non-negative.")
  if motion_target_positions.ndim not in (2, 3) or (
    motion_target_positions.shape[-1] != 3
  ):
    raise ValueError(
      "motion_target_positions must have shape (M, 3) or (N, M, 3)."
    )
  per_trajectory_targets = motion_target_positions.ndim == 3
  if per_trajectory_targets and motion_target_positions.shape[0] != len(trajectories.positions):
    raise ValueError(
      "Per-trajectory motion targets must match the trajectory batch size."
    )
  motion_count = motion_target_positions.shape[-2]
  if minimum_contact_times.shape != (motion_count,) or nominal_contact_times.shape != (
    motion_count,
  ):
    raise ValueError("Motion contact-time bounds must have shape (M,).")
  if torch.any(minimum_contact_times > nominal_contact_times):
    raise ValueError("Minimum contact times cannot exceed nominal contact times.")

  positions = trajectories.positions
  count, _, _ = positions.shape
  device = positions.device
  dtype = positions.dtype
  best_distance_sq = torch.full((count,), torch.inf, dtype=dtype, device=device)
  best_motion = torch.zeros(count, dtype=torch.long, device=device)
  best_time_index = torch.zeros(count, dtype=torch.long, device=device)
  path_mask = trajectories.post_first_bounce_mask
  if include_pre_bounce:
    pre_bounce = trajectories.bounce_counts == 0
    if trajectories.valid_mask is not None:
      pre_bounce &= trajectories.valid_mask
    already_on_robot_side = (
      positions[:, 0, 0] <= STANDARD_TENNIS_PHYSICS.court.net_x_m
    )
    no_net_pre_bounce = pre_bounce & (
      (positions[..., 0] >= STANDARD_TENNIS_PHYSICS.court.net_x_m)
      | already_on_robot_side[:, None]
    )
    pre_bounce = torch.where(
      trajectories.net_cleared[:, None], pre_bounce, no_net_pre_bounce
    )
    path_mask = path_mask | pre_bounce
  command_times = trajectories.times + contact_time_offset_s

  use_hierarchical = 0 < hierarchical_top_k < motion_count
  if use_hierarchical:
    if per_trajectory_targets:
      coarse_targets = motion_target_positions
    else:
      coarse_targets = motion_target_positions[None].expand(count, -1, -1)

    # Incoming trajectories keep their horizontal x direction through the
    # ground rebound.  Search the monotonic x coordinate to get a cheap
    # spatial intercept estimate for every motion, then inspect nearby samples.
    x_direction = torch.sign(positions[:, -1, 0] - positions[:, 0, 0])
    x_direction = torch.where(
      x_direction == 0.0, torch.ones_like(x_direction), x_direction
    )
    monotonic_x = (positions[:, :, 0] * x_direction[:, None]).contiguous()
    target_x = (coarse_targets[:, :, 0] * x_direction[:, None]).contiguous()
    intercept_indices = torch.searchsorted(monotonic_x, target_x).clamp(
      max=positions.shape[1] - 1
    )

    minimum_time_indices = torch.searchsorted(
      command_times, minimum_contact_times
    )
    maximum_time_indices = (
      torch.searchsorted(command_times, nominal_contact_times, right=True) - 1
    ).clamp(min=0, max=positions.shape[1] - 1)
    path_exists = path_mask.any(dim=1)
    first_path_index = path_mask.to(torch.int8).argmax(dim=1)
    last_path_index = positions.shape[1] - 1 - torch.flip(
      path_mask, dims=(1,)
    ).to(torch.int8).argmax(dim=1)
    lower_indices = torch.maximum(
      first_path_index[:, None], minimum_time_indices[None]
    )
    upper_indices = torch.minimum(
      last_path_index[:, None], maximum_time_indices[None]
    )
    valid_window = path_exists[:, None] & (lower_indices <= upper_indices)
    intercept_indices = torch.maximum(
      torch.minimum(intercept_indices, upper_indices), lower_indices
    )

    neighbor_offsets = torch.arange(
      -hierarchical_coarse_neighbor_radius,
      hierarchical_coarse_neighbor_radius + 1,
      device=device,
    )
    neighbor_indices = intercept_indices[:, :, None] + neighbor_offsets
    neighbor_indices = torch.maximum(
      torch.minimum(neighbor_indices, upper_indices[:, :, None]),
      lower_indices[:, :, None],
    ).clamp(min=0, max=positions.shape[1] - 1)
    flat_neighbor_indices = neighbor_indices.reshape(count, -1)
    sampled_positions = positions.gather(
      1, flat_neighbor_indices[:, :, None].expand(-1, -1, 3)
    ).reshape(count, motion_count, len(neighbor_offsets), 3)
    coarse_distance_sq = torch.sum(
      torch.square(sampled_positions - coarse_targets[:, :, None, :]), dim=-1
    )
    coarse_distance_sq.masked_fill_(~valid_window[:, :, None], torch.inf)
    coarse_motion_distances = coarse_distance_sq.min(dim=-1).values

    selected_motion_ids = torch.topk(
      coarse_motion_distances,
      k=min(hierarchical_top_k, motion_count),
      dim=1,
      largest=False,
      sorted=False,
    ).indices
    if per_trajectory_targets:
      selected_targets = motion_target_positions.gather(
        1, selected_motion_ids[:, :, None].expand(-1, -1, 3)
      )
    else:
      selected_targets = motion_target_positions[selected_motion_ids]
    selected_minimum_times = minimum_contact_times[selected_motion_ids]
    selected_nominal_times = nominal_contact_times[selected_motion_ids]
    position_sq = torch.sum(positions * positions, dim=-1)[:, None, :]
    target_sq = torch.sum(selected_targets * selected_targets, dim=-1)[:, :, None]
    cross = torch.einsum("ntd,nkd->nkt", positions, selected_targets)
    distance_sq = (position_sq + target_sq - 2.0 * cross).clamp_min_(0.0)
    timing_mask = (
      command_times[None, None, :] >= selected_minimum_times[:, :, None]
    ) & (command_times[None, None, :] <= selected_nominal_times[:, :, None])
    distance_sq.masked_fill_(~(path_mask[:, None, :] & timing_mask), torch.inf)
    motion_time_distance, motion_time_index = distance_sq.min(dim=-1)
    best_distance_sq, selected_offset = motion_time_distance.min(dim=-1)
    best_motion = selected_motion_ids.gather(
      1, selected_offset[:, None]
    ).squeeze(1)
    best_time_index = motion_time_index.gather(
      1, selected_offset[:, None]
    ).squeeze(1)
  else:
    position_sq = torch.sum(positions * positions, dim=-1)[:, None, :]

    for start in range(0, motion_count, motion_chunk_size):
      end = min(start + motion_chunk_size, motion_count)
      if per_trajectory_targets:
        targets = motion_target_positions[:, start:end]
        target_sq = torch.sum(targets * targets, dim=-1)[:, :, None]
        cross = torch.einsum("ntd,nmd->nmt", positions, targets)
      else:
        targets = motion_target_positions[start:end]
        target_sq = torch.sum(targets * targets, dim=-1)[None, :, None]
        cross = torch.einsum("ntd,md->nmt", positions, targets)
      distance_sq = (position_sq + target_sq - 2.0 * cross).clamp_min_(0.0)
      timing_mask = (
        command_times[None, None, :]
        >= minimum_contact_times[None, start:end, None]
      ) & (
        command_times[None, None, :]
        <= nominal_contact_times[None, start:end, None]
      )
      valid_samples = path_mask[:, None, :] & timing_mask
      distance_sq.masked_fill_(~valid_samples, torch.inf)
      chunk_time_distance, chunk_time_index = distance_sq.min(dim=-1)
      chunk_distance, chunk_motion_offset = chunk_time_distance.min(dim=-1)
      candidate_time_index = chunk_time_index.gather(
        1, chunk_motion_offset[:, None]
      ).squeeze(1)
      better = chunk_distance < best_distance_sq
      best_distance_sq = torch.where(better, chunk_distance, best_distance_sq)
      best_motion = torch.where(better, chunk_motion_offset + start, best_motion)
      best_time_index = torch.where(better, candidate_time_index, best_time_index)

  env_ids = torch.arange(count, device=device)
  distances = torch.sqrt(best_distance_sq)
  finite = torch.isfinite(distances)
  return TorchMotionMatchBatch(
    motion_ids=best_motion,
    trajectory_indices=best_time_index,
    target_positions=positions[env_ids, best_time_index],
    contact_times=command_times[best_time_index],
    distances=distances,
    valid=finite & (distances <= maximum_distance),
  )
