"""World-frame trajectory matching for tennis motion selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class StrokeSide(str, Enum):
  FOREHAND = "forehand"
  BACKHAND = "backhand"


@dataclass(frozen=True)
class PlaneCrossing:
  time: float
  position_world: np.ndarray
  lateral_position: float
  side: StrokeSide | None
  segment_index: int


@dataclass(frozen=True)
class MotionReachTarget:
  motion_index: int
  side: StrokeSide
  mean_world: np.ndarray
  nominal_strike_time: float
  minimum_strike_time: float


@dataclass(frozen=True)
class MotionTrajectoryMatch:
  motion_index: int
  side: StrokeSide
  distance: float
  target_world: np.ndarray
  arrival_time: float
  nominal_strike_time: float
  minimum_strike_time: float
  contact_time: float
  wait_time: float
  time_slack: float
  segment_index: int


def _vec3(value: np.ndarray, name: str) -> np.ndarray:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (3,):
    raise ValueError(f"{name} must have shape (3,), got {result.shape}")
  if not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must contain finite values")
  return result.copy()


def _trajectory(
  times: np.ndarray, positions_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  times = np.asarray(times, dtype=np.float64)
  positions = np.asarray(positions_world, dtype=np.float64)
  if positions.ndim != 2 or positions.shape[1] != 3:
    raise ValueError("positions_world must have shape (N, 3)")
  if times.shape != (positions.shape[0],):
    raise ValueError("times and positions_world must contain the same samples")
  if len(times) == 0:
    raise ValueError("trajectory must contain at least one sample")
  if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
    raise ValueError("trajectory must contain only finite values")
  if np.any(np.diff(times) < 0.0):
    raise ValueError("trajectory times must be non-decreasing")
  return times, positions


def first_forward_plane_crossing(
  times: np.ndarray,
  positions_world: np.ndarray,
  plane_origin_world: np.ndarray,
  forward_axis_world: np.ndarray,
  lateral_axis_world: np.ndarray,
  *,
  side_deadband: float = 0.1,
) -> PlaneCrossing | None:
  """Find the first trajectory crossing from local +X through local X=0."""
  times, positions = _trajectory(times, positions_world)
  origin = _vec3(plane_origin_world, "plane_origin_world")
  forward = _vec3(forward_axis_world, "forward_axis_world")
  lateral = _vec3(lateral_axis_world, "lateral_axis_world")
  if side_deadband < 0.0:
    raise ValueError("side_deadband must be non-negative")

  forward_norm = np.linalg.norm(forward)
  lateral_norm = np.linalg.norm(lateral)
  if forward_norm <= 1e-12 or lateral_norm <= 1e-12:
    raise ValueError("planning-frame axes must be non-zero")
  forward /= forward_norm
  lateral /= lateral_norm

  local_x = (positions - origin) @ forward
  for i in range(len(local_x) - 1):
    x0 = float(local_x[i])
    x1 = float(local_x[i + 1])
    if not (x0 > 0.0 and x1 <= 0.0):
      continue
    denominator = x0 - x1
    if denominator <= 1e-12:
      continue
    alpha = float(np.clip(x0 / denominator, 0.0, 1.0))
    point = positions[i] + alpha * (positions[i + 1] - positions[i])
    time = float(times[i] + alpha * (times[i + 1] - times[i]))
    lateral_position = float((point - origin) @ lateral)
    side = None
    if lateral_position < -side_deadband:
      side = StrokeSide.FOREHAND
    elif lateral_position > side_deadband:
      side = StrokeSide.BACKHAND
    return PlaneCrossing(
      time=time,
      position_world=point,
      lateral_position=lateral_position,
      side=side,
      segment_index=i,
    )
  return None


def closest_trajectory_point(
  times: np.ndarray,
  positions_world: np.ndarray,
  point_world: np.ndarray,
) -> tuple[float, np.ndarray, float, int]:
  """Return continuous point-to-polyline distance, projection, time and segment."""
  times, positions = _trajectory(times, positions_world)
  point = _vec3(point_world, "point_world")
  if len(positions) == 1:
    return (
      float(np.linalg.norm(point - positions[0])),
      positions[0].copy(),
      float(times[0]),
      0,
    )

  starts = positions[:-1]
  segments = positions[1:] - starts
  denominators = np.einsum("ij,ij->i", segments, segments)
  numerators = np.einsum("ij,ij->i", point[None, :] - starts, segments)
  alphas = np.divide(
    numerators,
    denominators,
    out=np.zeros_like(numerators),
    where=denominators > 1e-18,
  )
  alphas = np.clip(alphas, 0.0, 1.0)
  projections = starts + alphas[:, None] * segments
  squared_distances = np.einsum(
    "ij,ij->i", projections - point[None, :], projections - point[None, :]
  )
  segment_index = int(np.argmin(squared_distances))
  alpha = float(alphas[segment_index])
  arrival_time = float(
    times[segment_index]
    + alpha * (times[segment_index + 1] - times[segment_index])
  )
  return (
    float(np.sqrt(squared_distances[segment_index])),
    projections[segment_index].copy(),
    arrival_time,
    segment_index,
  )


def rank_motion_matches(
  times: np.ndarray,
  positions_world: np.ndarray,
  motions: list[MotionReachTarget] | tuple[MotionReachTarget, ...],
  *,
  side: StrokeSide | None,
  max_distance: float = 0.2,
  reaction_margin: float = 0.0,
) -> list[MotionTrajectoryMatch]:
  """Rank side-compatible motions that are spatially and temporally feasible."""
  evaluated = evaluate_motion_matches(times, positions_world, motions, side=side)
  return filter_motion_matches(
    evaluated,
    max_distance=max_distance,
    reaction_margin=reaction_margin,
  )


def evaluate_motion_matches(
  times: np.ndarray,
  positions_world: np.ndarray,
  motions: list[MotionReachTarget] | tuple[MotionReachTarget, ...],
  *,
  side: StrokeSide | None,
) -> list[MotionTrajectoryMatch]:
  """Evaluate all side-compatible motions without distance or timing filters."""
  evaluated: list[MotionTrajectoryMatch] = []
  for motion in motions:
    if side is not None and motion.side is not side:
      continue
    if motion.minimum_strike_time < 0.0:
      raise ValueError("motion minimum_strike_time must be non-negative")
    if motion.nominal_strike_time < motion.minimum_strike_time:
      raise ValueError(
        "motion nominal_strike_time must be at least minimum_strike_time"
      )
    distance, target, arrival_time, segment_index = closest_trajectory_point(
      times, positions_world, motion.mean_world
    )
    contact_time = min(arrival_time, motion.nominal_strike_time)
    evaluated.append(
      MotionTrajectoryMatch(
        motion_index=motion.motion_index,
        side=motion.side,
        distance=distance,
        target_world=target,
        arrival_time=arrival_time,
        nominal_strike_time=motion.nominal_strike_time,
        minimum_strike_time=motion.minimum_strike_time,
        contact_time=contact_time,
        wait_time=max(0.0, arrival_time - motion.nominal_strike_time),
        time_slack=arrival_time - motion.minimum_strike_time,
        segment_index=segment_index,
      )
    )

  evaluated.sort(key=lambda item: (item.distance, item.motion_index))
  return evaluated



def filter_motion_matches(
  evaluated: list[MotionTrajectoryMatch] | tuple[MotionTrajectoryMatch, ...],
  *,
  max_distance: float = 0.2,
  reaction_margin: float = 0.0,
) -> list[MotionTrajectoryMatch]:
  """Apply the spatial and execution-time gates to evaluated matches."""
  if max_distance < 0.0:
    raise ValueError("max_distance must be non-negative")
  if reaction_margin < 0.0:
    raise ValueError("reaction_margin must be non-negative")
  ranked = [
    match
    for match in evaluated
    if match.distance <= max_distance and match.time_slack >= reaction_margin
  ]
  ranked.sort(key=lambda item: (item.distance, -item.time_slack, item.motion_index))
  return ranked


def retain_motion_match(
  evaluated: list[MotionTrajectoryMatch] | tuple[MotionTrajectoryMatch, ...],
  motion_index: int | None,
  *,
  max_distance: float = 0.2,
) -> MotionTrajectoryMatch | None:
  """Keep a prepared motion while its updated intercept remains executable."""
  if motion_index is None:
    return None
  if max_distance < 0.0:
    raise ValueError("max_distance must be non-negative")
  return next(
    (
      match
      for match in evaluated
      if match.motion_index == motion_index
      and match.distance <= max_distance
      and match.time_slack >= 0.0
    ),
    None,
  )
