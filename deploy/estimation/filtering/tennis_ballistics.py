"""ROS-free tennis-ball trajectory prediction shared by deploy and simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81
COEFF_OF_RESTITUTION = 0.85


@dataclass(frozen=True)
class TrajectoryIntersection:
  """A point interpolated along a sampled trajectory segment."""

  time: float
  position: np.ndarray
  segment_index: int


def _vec3(value: np.ndarray, name: str) -> np.ndarray:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (3,):
    raise ValueError(f"{name} must have shape (3,), got {result.shape}")
  if not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must contain only finite values")
  return result.copy()


def predict_ball_trajectory(
  position: np.ndarray,
  velocity: np.ndarray,
  *,
  duration: float = 5.0,
  dt: float = 0.02,
  gravity: float = GRAVITY,
  restitution: float = COEFF_OF_RESTITUTION,
  ground_z: float = 0.0,
  max_bounces: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
  """Predict ballistic flight with instantaneous vertical ground bounces.

  Horizontal velocity is conserved. At each impact, vertical velocity is
  reflected and multiplied by ``restitution``. Positions describe the ball
  center, so ``ground_z`` should be the ball radius for a physical floor.
  """
  if duration < 0.0:
    raise ValueError("duration must be non-negative")
  if dt <= 0.0:
    raise ValueError("dt must be positive")
  if gravity <= 0.0:
    raise ValueError("gravity must be positive")
  if not 0.0 <= restitution <= 1.0:
    raise ValueError("restitution must be in [0, 1]")
  if max_bounces < 0:
    raise ValueError("max_bounces must be non-negative")

  pos = _vec3(position, "position")
  vel = _vec3(velocity, "velocity")
  pos[2] = max(pos[2], ground_z)

  times = [0.0]
  positions = [pos.copy()]
  elapsed = 0.0
  bounce_count = 0
  eps = 1e-9

  while elapsed < duration - eps:
    height = max(pos[2] - ground_z, 0.0)

    # Resolve a state that is already moving into the floor before finding the
    # next positive root. This also keeps noisy measured states well-defined.
    if height <= eps and vel[2] < -eps:
      if bounce_count >= max_bounces:
        break
      vel[2] = -restitution * vel[2]
      bounce_count += 1
      continue

    discriminant = vel[2] ** 2 + 2.0 * gravity * height
    impact_dt = (vel[2] + np.sqrt(max(discriminant, 0.0))) / gravity
    if impact_dt <= eps:
      break

    remaining = duration - elapsed
    arc_duration = min(impact_dt, remaining)
    sample_offsets = np.arange(dt, arc_duration + eps, dt, dtype=np.float64)
    if sample_offsets.size == 0 or sample_offsets[-1] < arc_duration - eps:
      sample_offsets = np.append(sample_offsets, arc_duration)
    elif sample_offsets[-1] > arc_duration + eps:
      sample_offsets[-1] = arc_duration

    arc_positions = pos[None, :] + sample_offsets[:, None] * vel[None, :]
    arc_positions[:, 2] -= 0.5 * gravity * sample_offsets**2
    if impact_dt <= remaining + eps:
      arc_positions[-1, 2] = ground_z

    times.extend((elapsed + sample_offsets).tolist())
    positions.extend(arc_positions)

    if impact_dt >= remaining - eps:
      break

    pos = pos + impact_dt * vel
    pos[2] = ground_z
    impact_vz = vel[2] - gravity * impact_dt
    vel[2] = -restitution * impact_vz
    elapsed += impact_dt
    bounce_count += 1
    if bounce_count > max_bounces:
      break

  return np.asarray(times, dtype=np.float64), np.asarray(positions, dtype=np.float64)


def first_plane_intersection(
  times: np.ndarray,
  positions: np.ndarray,
  plane_point: np.ndarray,
  plane_normal: np.ndarray,
) -> TrajectoryIntersection | None:
  """Return the first sampled segment crossing an infinite plane."""
  times = np.asarray(times, dtype=np.float64)
  positions = np.asarray(positions, dtype=np.float64)
  point = _vec3(plane_point, "plane_point")
  normal = _vec3(plane_normal, "plane_normal")
  normal_norm = np.linalg.norm(normal)
  if normal_norm <= 1e-12:
    raise ValueError("plane_normal must be non-zero")
  normal /= normal_norm
  if positions.ndim != 2 or positions.shape[1] != 3:
    raise ValueError("positions must have shape (N, 3)")
  if times.shape != (positions.shape[0],):
    raise ValueError("times and positions must contain the same number of samples")
  if len(times) < 2:
    return None

  distances = (positions - point) @ normal
  for i in range(len(distances) - 1):
    d0 = distances[i]
    d1 = distances[i + 1]
    if d0 == 0.0:
      return TrajectoryIntersection(float(times[i]), positions[i].copy(), i)
    if d0 * d1 > 0.0:
      continue
    denominator = d1 - d0
    if abs(denominator) <= 1e-12:
      continue
    alpha = float(np.clip(-d0 / denominator, 0.0, 1.0))
    position = positions[i] + alpha * (positions[i + 1] - positions[i])
    time = float(times[i] + alpha * (times[i + 1] - times[i]))
    return TrajectoryIntersection(time, position, i)
  return None


def first_ground_contact(
  times: np.ndarray,
  positions: np.ndarray,
  *,
  ground_z: float = 0.0,
  tolerance: float = 1e-6,
) -> TrajectoryIntersection | None:
  """Return the first transition from above the ground to ground contact."""
  times = np.asarray(times, dtype=np.float64)
  positions = np.asarray(positions, dtype=np.float64)
  if positions.ndim != 2 or positions.shape[1] != 3:
    raise ValueError("positions must have shape (N, 3)")
  if times.shape != (positions.shape[0],):
    raise ValueError("times and positions must contain the same number of samples")

  for i in range(len(positions) - 1):
    z0 = positions[i, 2]
    z1 = positions[i + 1, 2]
    if z0 <= ground_z + tolerance or z1 > ground_z + tolerance:
      continue
    denominator = z1 - z0
    alpha = 1.0 if abs(denominator) <= 1e-12 else (ground_z - z0) / denominator
    alpha = float(np.clip(alpha, 0.0, 1.0))
    position = positions[i] + alpha * (positions[i + 1] - positions[i])
    position[2] = ground_z
    time = float(times[i] + alpha * (times[i + 1] - times[i]))
    return TrajectoryIntersection(time, position, i)
  return None
