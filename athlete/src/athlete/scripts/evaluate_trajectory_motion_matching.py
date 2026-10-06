"""Evaluate incoming-ball matching coverage for a local motion set."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat
from athlete.goal_cond_tracking.torch_tennis_planner import (
  TorchTennisTrajectoryBatch,
  match_tennis_trajectories_to_motions_torch,
  sample_root_directed_tennis_launches_torch,
  simulate_tennis_trajectories_torch,
)
from athlete.motion_sets.motion_set import MotionSet
from athlete.scripts.tennis_physics import (
  STANDARD_TENNIS_DOMAIN_RANDOMIZATION,
  STANDARD_TENNIS_PHYSICS,
)


@dataclass(frozen=True)
class TrajectoryMotionMatchingEvalCfg:
  motion_config: Path
  num_trajectories: int = 98_304
  batch_size: int = 4096
  first_bounce_radius_m: float = 3.0
  filter_first_bounce_radius: bool = True
  maximum_match_distance_m: float = 0.15
  dt: float = 0.01
  horizon_s: float = 4.0
  launch_position_min: tuple[float, float, float] = (8.0, -2.0, 0.5)
  launch_position_max: tuple[float, float, float] = (10.0, 2.0, 1.3)
  launch_velocity_min: tuple[float, float, float] = (-5.25, -1.25, 3.0)
  launch_velocity_max: tuple[float, float, float] = (-3.5, 1.25, 5.0)
  root_directed_vz_sampling: bool = False
  horizontal_speed_range_m_s: tuple[float, float] = (3.5, 5.25)
  horizontal_angle_half_width_deg: float = 15.0
  robot_root_xy: tuple[float, float] = (0.0, 0.0)
  net_crossing_height_range_m: tuple[float, float] = (1.5, 3.5)
  maximum_initial_speed_m_s: float = 7.0
  max_contact_speedup: float = 2.0
  deadline_offset_s: float = 0.01
  motion_chunk_size: int = 4
  excluded_motion_indices: tuple[int, ...] = ()
  physics_group_size: int = 1
  seed: int = 42
  device: str = "cuda:0"
  output_file: Path | None = None


def _uniform(
  minimum: tuple[float, ...],
  maximum: tuple[float, ...],
  count: int,
  *,
  device: torch.device,
) -> torch.Tensor:
  lower = torch.tensor(minimum, dtype=torch.float32, device=device)
  upper = torch.tensor(maximum, dtype=torch.float32, device=device)
  return lower + torch.rand((count, len(minimum)), device=device) * (upper - lower)


def _uniform_scalar(
  bounds: tuple[float, float], count: int, *, device: torch.device
) -> torch.Tensor:
  return bounds[0] + torch.rand(count, device=device) * (bounds[1] - bounds[0])


def _sample_root_directed_launches(
  cfg: TrajectoryMotionMatchingEvalCfg,
  count: int,
  *,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Sample fixed-court positions and root-directed velocities with solved vz."""
  root_xy = torch.tensor(
    cfg.robot_root_xy, dtype=torch.float32, device=device
  ).expand(count, -1)
  return sample_root_directed_tennis_launches_torch(
    root_xy,
    launch_position_min=torch.tensor(
      cfg.launch_position_min, dtype=torch.float32, device=device
    ),
    launch_position_max=torch.tensor(
      cfg.launch_position_max, dtype=torch.float32, device=device
    ),
    horizontal_speed_range_m_s=cfg.horizontal_speed_range_m_s,
    horizontal_angle_half_width_deg=cfg.horizontal_angle_half_width_deg,
    net_crossing_height_range_m=cfg.net_crossing_height_range_m,
    maximum_initial_speed_m_s=cfg.maximum_initial_speed_m_s,
  )


def _actual_net_crossing(
  trajectories: TorchTennisTrajectoryBatch,
  *,
  height_range_m: tuple[float, float],
) -> tuple[
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
]:
  """Return actual crossing, physical clearance, planned height, and details."""
  physics = STANDARD_TENNIS_PHYSICS
  positions = trajectories.positions
  before = positions[:, :-1]
  after = positions[:, 1:]
  crossings = (before[:, :, 0] > physics.court.net_x_m) & (
    after[:, :, 0] <= physics.court.net_x_m
  )
  crossed = crossings.any(dim=1)
  crossing_index = crossings.to(torch.int64).argmax(dim=1)
  batch_ids = torch.arange(len(positions), device=positions.device)
  p0 = before[batch_ids, crossing_index]
  p1 = after[batch_ids, crossing_index]
  alpha = (p0[:, 0] - physics.court.net_x_m) / (
    p0[:, 0] - p1[:, 0]
  ).clamp_min(torch.finfo(positions.dtype).eps)
  crossing_position = p0 + alpha[:, None].clamp(0.0, 1.0) * (p1 - p0)
  crossing_bounce_count = trajectories.bounce_counts[:, :-1][
    batch_ids, crossing_index
  ]
  lower, upper = height_range_m
  physical_clearance = (
    crossed
    & (crossing_bounce_count == 0)
    & (crossing_position[:, 1].abs() <= physics.court.net_half_width_m)
    & trajectories.net_cleared
  )
  within_planned_height = (
    physical_clearance
    & (crossing_position[:, 2] >= lower)
    & (crossing_position[:, 2] <= upper)
  )
  crossing_height = torch.where(
    crossed,
    crossing_position[:, 2],
    torch.full_like(crossing_position[:, 2], torch.nan),
  )
  return (
    crossed,
    physical_clearance,
    within_planned_height,
    crossing_height,
    crossing_bounce_count,
  )


def _load_motion_match_data(
  motion_config: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
  motion_set = MotionSet.from_toml(motion_config)
  motion_files = motion_set.local_motion_files
  motion_cfgs = motion_set.local_motion_cfgs()
  if motion_files is None or motion_cfgs is None:
    raise ValueError("Matching evaluation requires a local motion dataset.")
  if len(motion_files) != len(motion_cfgs):
    raise RuntimeError("Motion files and configs have different lengths.")

  target_means: list[list[float]] = []
  root_positions: list[np.ndarray] = []
  root_quaternions: list[np.ndarray] = []
  nominal_contact_times: list[float] = []
  motion_names: list[str] = []
  for motion_file, motion_cfg in zip(motion_files, motion_cfgs, strict=True):
    position_targets = [
      target for target in motion_cfg.sub_targets if target.goal_type == "position"
    ]
    if len(position_targets) != 1:
      raise ValueError(
        f"Expected one position target for {motion_file}, got {len(position_targets)}."
      )
    target = position_targets[0]
    with np.load(motion_file) as data:
      frame_count = int(data["joint_pos"].shape[0])
      root_positions.append(data["body_pos_w"][0, 0].astype(np.float32))
      root_quaternions.append(data["body_quat_w"][0, 0].astype(np.float32))
    strike_phase = 0.5 * (target.target_phase_start + target.target_phase_end)
    duration_s = (frame_count - 1) / 50.0
    nominal_contact_times.append(strike_phase * duration_s)
    target_means.append(
      [
        float(target.target_pos_mean[axis])
        for axis in ("x", "y", "z")
      ]
    )
    motion_names.append(Path(motion_file).stem)

  roots = torch.as_tensor(np.stack(root_positions), device=device)
  root_quats = torch.as_tensor(np.stack(root_quaternions), device=device)
  means = torch.tensor(target_means, dtype=torch.float32, device=device)
  yaw_removed_quats = quat_mul(quat_inv(yaw_quat(root_quats)), root_quats)
  aligned_roots = torch.zeros_like(roots)
  aligned_roots[:, 2] = roots[:, 2]
  targets = aligned_roots + quat_apply(yaw_removed_quats, means)
  nominal = torch.tensor(
    nominal_contact_times, dtype=torch.float32, device=device
  )
  return targets, nominal / 2.0, nominal, motion_names


def _quantile(values: torch.Tensor, q: float) -> float:
  if values.numel() == 0:
    return float("nan")
  return float(torch.quantile(values.float(), q).item())


def _mean_over_valid(values: torch.Tensor, valid: torch.Tensor) -> float:
  selected = values[valid]
  return float(selected.mean().item()) if selected.numel() else float("nan")


@torch.no_grad()
def run_evaluation(cfg: TrajectoryMotionMatchingEvalCfg) -> dict[str, object]:
  if cfg.num_trajectories <= 0 or cfg.batch_size <= 0:
    raise ValueError("num_trajectories and batch_size must be positive.")
  if cfg.filter_first_bounce_radius and cfg.first_bounce_radius_m <= 0.0:
    raise ValueError("first_bounce_radius_m must be positive.")
  if cfg.max_contact_speedup < 1.0:
    raise ValueError("max_contact_speedup must be at least one.")
  if cfg.physics_group_size <= 0:
    raise ValueError("physics_group_size must be positive.")

  device = torch.device(cfg.device)
  torch.manual_seed(cfg.seed)
  targets, minimum_times, nominal_times, motion_names = _load_motion_match_data(
    cfg.motion_config, device
  )
  if cfg.excluded_motion_indices:
    excluded = set(cfg.excluded_motion_indices)
    invalid = sorted(
      index for index in excluded if not 0 <= index < len(motion_names)
    )
    if invalid:
      raise ValueError(f"Excluded motion indices are out of range: {invalid}")
    keep = torch.tensor(
      [index not in excluded for index in range(len(motion_names))],
      dtype=torch.bool,
      device=device,
    )
    targets = targets[keep]
    minimum_times = minimum_times[keep]
    nominal_times = nominal_times[keep]
    motion_names = [
      name for index, name in enumerate(motion_names) if index not in excluded
    ]
  randomization = STANDARD_TENNIS_DOMAIN_RANDOMIZATION

  within_radius_all: list[torch.Tensor] = []
  has_first_bounce_all: list[torch.Tensor] = []
  net_cleared_all: list[torch.Tensor] = []
  net_planned_height_all: list[torch.Tensor] = []
  net_crossed_all: list[torch.Tensor] = []
  net_crossing_height_all: list[torch.Tensor] = []
  net_crossing_bounce_count_all: list[torch.Tensor] = []
  initial_speed_all: list[torch.Tensor] = []
  matched_all: list[torch.Tensor] = []
  bounce_radius_all: list[torch.Tensor] = []
  match_distance_all: list[torch.Tensor] = []
  matched_motion_ids: list[torch.Tensor] = []
  start_time = time.perf_counter()

  completed = 0
  while completed < cfg.num_trajectories:
    count = min(cfg.batch_size, cfg.num_trajectories - completed)
    if cfg.root_directed_vz_sampling:
      initial_positions, initial_velocities = _sample_root_directed_launches(
        cfg, count, device=device
      )
    else:
      initial_positions = _uniform(
        cfg.launch_position_min, cfg.launch_position_max, count, device=device
      )
      initial_velocities = _uniform(
        cfg.launch_velocity_min, cfg.launch_velocity_max, count, device=device
      )
    physics_count = math.ceil(count / cfg.physics_group_size)

    def grouped_physics(
      bounds: tuple[float, float],
      physics_sample_count: int = physics_count,
      trajectory_count: int = count,
    ) -> torch.Tensor:
      return _uniform_scalar(
        bounds, physics_sample_count, device=device
      ).repeat_interleave(
        cfg.physics_group_size
      )[:trajectory_count]

    mass = grouped_physics(randomization.ball_mass_kg)
    restitution = _uniform_scalar(
      randomization.court_restitution, physics_count, device=device
    ).repeat_interleave(cfg.physics_group_size)[:count]
    tangent_retention = grouped_physics(
      randomization.ground_tangent_speed_retention
    )
    drag = grouped_physics(randomization.drag_coefficient)
    trajectories = simulate_tennis_trajectories_torch(
      initial_positions,
      initial_velocities,
      ball_mass_kg=mass,
      court_restitution=restitution,
      tangent_speed_retention=tangent_retention,
      drag_coefficient=drag,
      dt=cfg.dt,
      horizon_s=cfg.horizon_s,
    )
    (
      net_crossed,
      verified_net_clear,
      within_planned_net_height,
      net_crossing_height,
      net_crossing_bounce_count,
    ) = _actual_net_crossing(
      trajectories, height_range_m=cfg.net_crossing_height_range_m
    )

    first_bounce_mask = trajectories.bounce_counts >= 1
    has_first_bounce = first_bounce_mask.any(dim=1)
    first_bounce_index = first_bounce_mask.to(torch.int64).argmax(dim=1)
    batch_ids = torch.arange(count, device=device)
    first_bounce_position = trajectories.positions[batch_ids, first_bounce_index]
    bounce_radius = torch.linalg.vector_norm(first_bounce_position[:, :2], dim=1)
    within_radius = has_first_bounce & (bounce_radius <= cfg.first_bounce_radius_m)
    eligible = (
      within_radius.clone()
      if cfg.filter_first_bounce_radius
      else has_first_bounce.clone()
    )
    if cfg.root_directed_vz_sampling:
      eligible &= verified_net_clear

    matched = torch.zeros(count, dtype=torch.bool, device=device)
    match_distance = torch.full((count,), torch.nan, device=device)
    motion_ids = torch.full((count,), -1, dtype=torch.long, device=device)
    selected = torch.where(eligible)[0]
    if len(selected) > 0:
      selected_trajectories = TorchTennisTrajectoryBatch(
        times=trajectories.times,
        positions=trajectories.positions[selected],
        velocities=trajectories.velocities[selected],
        bounce_counts=trajectories.bounce_counts[selected],
        net_cleared=(
          verified_net_clear[selected]
          if cfg.root_directed_vz_sampling
          else trajectories.net_cleared[selected]
        ),
      )
      match = match_tennis_trajectories_to_motions_torch(
        selected_trajectories,
        targets,
        minimum_times,
        nominal_times,
        maximum_distance=cfg.maximum_match_distance_m,
        motion_chunk_size=cfg.motion_chunk_size,
        contact_time_offset_s=cfg.deadline_offset_s,
      )
      matched[selected] = match.valid
      match_distance[selected] = match.distances
      motion_ids[selected] = match.motion_ids

    within_radius_all.append(within_radius.cpu())
    has_first_bounce_all.append(has_first_bounce.cpu())
    net_cleared_all.append(verified_net_clear.cpu())
    net_planned_height_all.append(within_planned_net_height.cpu())
    net_crossed_all.append(net_crossed.cpu())
    net_crossing_height_all.append(net_crossing_height.cpu())
    net_crossing_bounce_count_all.append(net_crossing_bounce_count.cpu())
    initial_speed_all.append(
      torch.linalg.vector_norm(initial_velocities, dim=1).cpu()
    )
    matched_all.append(matched.cpu())
    bounce_radius_all.append(bounce_radius.cpu())
    match_distance_all.append(match_distance.cpu())
    matched_motion_ids.append(motion_ids.cpu())
    completed += count
    print(f"[MATCH EVAL] {completed}/{cfg.num_trajectories}", flush=True)

  if device.type == "cuda":
    torch.cuda.synchronize(device)
  elapsed_s = time.perf_counter() - start_time
  within_radius = torch.cat(within_radius_all)
  has_first_bounce = torch.cat(has_first_bounce_all)
  net_cleared = torch.cat(net_cleared_all)
  net_planned_height = torch.cat(net_planned_height_all)
  net_crossed = torch.cat(net_crossed_all)
  net_crossing_height = torch.cat(net_crossing_height_all)
  net_crossing_bounce_count = torch.cat(net_crossing_bounce_count_all)
  initial_speed = torch.cat(initial_speed_all)
  matched = torch.cat(matched_all)
  bounce_radius = torch.cat(bounce_radius_all)
  match_distance = torch.cat(match_distance_all)
  motion_ids = torch.cat(matched_motion_ids)
  net_and_within = within_radius & net_cleared
  eligible = (
    within_radius.clone()
    if cfg.filter_first_bounce_radius
    else has_first_bounce.clone()
  )
  if cfg.root_directed_vz_sampling:
    eligible &= net_cleared
  net_and_eligible = eligible & net_cleared

  def rate(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    count = int(denominator.sum().item())
    return float((numerator & denominator).sum().item() / count) if count else math.nan

  def grouped_rate(group_size: int) -> tuple[int, float]:
    group_count = cfg.num_trajectories // group_size
    grouped = matched[: group_count * group_size].reshape(group_count, group_size)
    return group_count, float(grouped.any(dim=1).float().mean().item())

  groups8, attempts8_rate = grouped_rate(8)
  groups24, attempts24_rate = grouped_rate(24)
  groups96, attempts96_rate = grouped_rate(96)
  successful_distances = match_distance[matched]
  successful_motion_ids = motion_ids[matched]
  unique_motion_count = int(torch.unique(successful_motion_ids).numel())
  results: dict[str, object] = {
    "motion_config": str(cfg.motion_config.resolve()),
    "motion_count": len(motion_names),
    "excluded_motion_indices": list(cfg.excluded_motion_indices),
    "trajectory_count": cfg.num_trajectories,
    "physics_group_size": cfg.physics_group_size,
    "seed": cfg.seed,
    "root_directed_vz_sampling": cfg.root_directed_vz_sampling,
    "horizontal_speed_range_m_s": list(cfg.horizontal_speed_range_m_s),
    "horizontal_angle_half_width_deg": cfg.horizontal_angle_half_width_deg,
    "robot_root_xy": list(cfg.robot_root_xy),
    "net_crossing_height_range_m": list(cfg.net_crossing_height_range_m),
    "maximum_initial_speed_m_s": cfg.maximum_initial_speed_m_s,
    "filter_first_bounce_radius": cfg.filter_first_bounce_radius,
    "first_bounce_radius_m": cfg.first_bounce_radius_m,
    "maximum_match_distance_m": cfg.maximum_match_distance_m,
    "has_first_bounce_count": int(has_first_bounce.sum().item()),
    "has_first_bounce_rate": float(has_first_bounce.float().mean().item()),
    "net_crossed_count": int(net_crossed.sum().item()),
    "net_crossed_rate": float(net_crossed.float().mean().item()),
    "net_cleared_count": int(net_cleared.sum().item()),
    "net_cleared_rate": float(net_cleared.float().mean().item()),
    "net_within_planned_height_count": int(net_planned_height.sum().item()),
    "net_within_planned_height_rate": float(
      net_planned_height.float().mean().item()
    ),
    "net_crossed_before_first_bounce_rate": float(
      (net_crossed & (net_crossing_bounce_count == 0)).float().mean().item()
    ),
    "net_crossing_height_mean_m": _mean_over_valid(
      net_crossing_height, torch.isfinite(net_crossing_height)
    ),
    "net_crossing_height_p05_m": _quantile(
      net_crossing_height[torch.isfinite(net_crossing_height)], 0.05
    ),
    "net_crossing_height_p95_m": _quantile(
      net_crossing_height[torch.isfinite(net_crossing_height)], 0.95
    ),
    "initial_speed_mean_m_s": float(initial_speed.mean().item()),
    "initial_speed_p95_m_s": _quantile(initial_speed, 0.95),
    "initial_speed_max_m_s": float(initial_speed.max().item()),
    "first_bounce_within_radius_count": int(within_radius.sum().item()),
    "first_bounce_within_radius_rate": float(within_radius.float().mean().item()),
    "net_cleared_within_radius_count": int(net_and_within.sum().item()),
    "net_clear_rate_within_radius": rate(net_cleared, within_radius),
    "single_trajectory_match_count": int(matched.sum().item()),
    "single_trajectory_match_rate_all_samples": float(matched.float().mean().item()),
    "single_trajectory_match_rate_eligible": rate(matched, eligible),
    "single_trajectory_match_rate_net_cleared_eligible": rate(
      matched, net_and_eligible
    ),
    "single_trajectory_match_rate_within_radius": rate(matched, within_radius),
    "single_trajectory_match_rate_net_cleared_within_radius": rate(
      matched, net_and_within
    ),
    "eight_attempt_group_count": groups8,
    "eight_attempt_match_rate_all_samples": attempts8_rate,
    "twenty_four_attempt_group_count": groups24,
    "twenty_four_attempt_match_rate_all_samples": attempts24_rate,
    "ninety_six_attempt_group_count": groups96,
    "ninety_six_attempt_match_rate_all_samples": attempts96_rate,
    "successful_match_distance_mean_m": (
      float(successful_distances.mean().item())
      if successful_distances.numel()
      else math.nan
    ),
    "successful_match_distance_p95_m": _quantile(successful_distances, 0.95),
    "first_bounce_radius_p10_m": _quantile(bounce_radius, 0.10),
    "first_bounce_radius_median_m": _quantile(bounce_radius, 0.50),
    "first_bounce_radius_p90_m": _quantile(bounce_radius, 0.90),
    "matched_unique_motion_count": unique_motion_count,
    "matched_motion_coverage_rate": unique_motion_count / len(motion_names),
    "elapsed_s": elapsed_s,
  }
  print(json.dumps(results, indent=2, ensure_ascii=False))
  if cfg.output_file is not None:
    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_file.write_text(
      json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
  return results


def main() -> None:
  run_evaluation(tyro.cli(TrajectoryMotionMatchingEvalCfg))


if __name__ == "__main__":
  main()
