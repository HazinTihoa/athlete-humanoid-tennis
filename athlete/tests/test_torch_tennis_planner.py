"""Tests for fixed-step Torch incoming-ball simulation and motion matching."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from mjlab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg
from athlete.goal_cond_tracking.config.g1.env_cfgs import (
  unitree_g1_tennis_launch_distill_env_cfg,
  unitree_g1_tennis_launch_distill_torch_match_200_env_cfg,
  unitree_g1_tennis_launch_distill_warp_match_200_env_cfg,
  unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_env_cfg,
  unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_env_cfg,
  unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_legacy_env_cfg,
  unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_env_cfg,
  unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg,
  unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_env_cfg,
  unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_env_cfg,
  unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_radial20_env_cfg,
  unitree_g1_tennis_small_court_intent_transformer_sweet_speed50_racket_ground10_env_cfg,
)
from athlete.goal_cond_tracking.mdp.commands import (
  MotionResamplePlan,
  MultiTargetMotionCommand,
)
from athlete.goal_cond_tracking.mdp.rewards import (
  tennis_post_strike_heading_to_landing_target_reward,
  tennis_post_strike_heading_to_startup_x_reward,
  tennis_racket_incoming_velocity_orientation_reward,
)
from athlete.goal_cond_tracking.torch_tennis_planner import (
  TorchTennisTrajectoryBatch,
  classify_tennis_failure_trajectories,
  match_tennis_trajectories_to_motions_torch,
  randomize_tennis_launch_directions_torch,
  retarget_tennis_launches_to_miss_roots_torch,
  sample_root_directed_tennis_launches_torch,
  simulate_tennis_trajectories_torch,
)
from athlete.goal_cond_tracking.mdp.observations import (
  encode_tennis_ball_position_radial,
)
from athlete.scripts.tennis_physics import STANDARD_TENNIS_PHYSICS
from athlete.scripts.tennis_scene import TorchIncomingBallMotionPlanner


def test_fixed_step_trajectory_marks_only_first_to_second_bounce_segment() -> None:
  physics = STANDARD_TENNIS_PHYSICS
  trajectories = simulate_tennis_trajectories_torch(
    torch.tensor([[11.93, -0.098, 1.2]]),
    torch.tensor([[-8.5, 0.7594239, 4.1717736]]),
    ball_mass_kg=torch.tensor([physics.ball.mass_kg]),
    court_restitution=torch.tensor([physics.court.restitution]),
    tangent_speed_retention=torch.tensor([0.825]),
    drag_coefficient=torch.tensor([physics.ball.drag_coefficient]),
    dt=0.01,
    horizon_s=4.0,
  )

  assert trajectories.net_cleared.tolist() == [True]
  assert trajectories.bounce_counts.max().item() == 2
  assert torch.any(trajectories.post_first_bounce_mask)
  assert torch.all(trajectories.bounce_counts[trajectories.post_first_bounce_mask] == 1)
  assert torch.all(torch.isfinite(trajectories.positions))
  assert torch.all(torch.isfinite(trajectories.velocities))


def test_failure_rollout_freezes_after_ball_leaves_observation_radius() -> None:
  positions = torch.tensor([[0.0, 0.0, 10.0]])
  velocities = torch.tensor([[10.0, 0.0, 20.0]])
  trajectory = simulate_tennis_trajectories_torch(
    positions,
    velocities,
    ball_mass_kg=torch.tensor([STANDARD_TENNIS_PHYSICS.ball.mass_kg]),
    court_restitution=torch.tensor([STANDARD_TENNIS_PHYSICS.court.restitution]),
    tangent_speed_retention=torch.tensor([0.8]),
    drag_coefficient=torch.tensor([STANDARD_TENNIS_PHYSICS.ball.drag_coefficient]),
    dt=0.01,
    horizon_s=5.0,
    stop_origin_w=torch.zeros_like(positions),
    stop_distance_m=21.0,
    stop_speed_m_s=0.05,
  )
  radius = torch.linalg.vector_norm(trajectory.positions, dim=-1)[0]
  crossing = int(torch.where(radius >= 21.0)[0][0])
  assert crossing > 0
  assert torch.all(radius[crossing:] == radius[crossing])
  assert trajectory.valid_mask[0, crossing].item()
  assert not trajectory.valid_mask[0, crossing + 1 :].any().item()
  torch.testing.assert_close(
    trajectory.positions[0, crossing:],
    trajectory.positions[0, crossing].expand_as(trajectory.positions[0, crossing:]),
  )


def test_radial_encoding_is_linear_inside_ten_metres_and_bounded() -> None:
  positions = torch.tensor([[3.0, 4.0, 0.0], [6.0, 8.0, 0.0], [9.0, 12.0, 0.0]])
  encoded = encode_tennis_ball_position_radial(positions)
  torch.testing.assert_close(encoded[:2], positions[:2])
  radii = torch.linalg.vector_norm(encoded, dim=-1)
  assert radii[2] > 10.0
  assert radii[2] < 20.0
  assert torch.allclose(encoded[2] / encoded[2].norm(), positions[2] / positions[2].norm())


def test_random_failure_launches_cover_all_headings_with_speed_cap() -> None:
  torch.manual_seed(1234)
  count = 128
  source_angle = torch.rand(count) * (2.0 * math.pi)
  source_radius = 2.0 + 3.0 * torch.rand(count)
  positions = torch.stack(
    (
      source_radius * torch.cos(source_angle),
      source_radius * torch.sin(source_angle),
      0.8 + 1.4 * torch.rand(count),
    ),
    dim=-1,
  )
  velocities = randomize_tennis_launch_directions_torch(
    positions,
    horizontal_speed_range_m_s=(3.5, 5.25),
    vertical_speed_range_m_s=(-2.0, 4.0),
    maximum_initial_speed_m_s=7.0,
  )
  azimuth = torch.atan2(velocities[:, 1], velocities[:, 0])
  assert torch.any((azimuth >= 0.0) & (azimuth < math.pi / 2))
  assert torch.any((azimuth >= math.pi / 2) & (azimuth < math.pi))
  assert torch.any((azimuth < -math.pi / 2))
  assert torch.any((azimuth >= -math.pi / 2) & (azimuth < 0.0))
  assert torch.all(torch.linalg.vector_norm(velocities, dim=-1) <= 7.0)
  assert torch.all((velocities[:, 2] >= -2.0) & (velocities[:, 2] <= 4.0))
  physics = STANDARD_TENNIS_PHYSICS
  trajectory = simulate_tennis_trajectories_torch(
    positions,
    velocities,
    ball_mass_kg=torch.full((count,), physics.ball.mass_kg),
    court_restitution=torch.full((count,), physics.court.restitution),
    tangent_speed_retention=torch.full((count,), 0.8),
    drag_coefficient=torch.full((count,), physics.ball.drag_coefficient),
    dt=0.01,
    horizon_s=12.0,
    stop_origin_w=torch.zeros_like(positions),
    stop_distance_m=20.0,
    stop_speed_m_s=0.05,
  )
  assert torch.all(trajectory.valid_mask.sum(dim=1) < trajectory.valid_mask.shape[1])


def test_pre_bounce_reachability_keeps_catchable_no_net_ball_on_normal_plan() -> None:
  trajectory = TorchTennisTrajectoryBatch(
    times=torch.tensor([0.0, 0.5, 1.0]),
    positions=torch.tensor([[[8.0, 0.0, 1.0], [6.0, 0.0, 1.0], [4.0, 0.0, 0.5]]]),
    velocities=torch.zeros(1, 3, 3),
    bounce_counts=torch.zeros(1, 3, dtype=torch.int8),
    net_cleared=torch.tensor([False]),
  )
  kwargs = dict(
    motion_target_positions=torch.tensor([[6.0, 0.0, 1.0]]),
    minimum_contact_times=torch.tensor([0.25]),
    nominal_contact_times=torch.tensor([0.75]),
    maximum_distance=0.15,
  )
  legacy = match_tennis_trajectories_to_motions_torch(trajectory, **kwargs)
  reachable = match_tennis_trajectories_to_motions_torch(
    trajectory, **kwargs, include_pre_bounce=True
  )
  assert legacy.valid.tolist() == [False]
  assert reachable.valid.tolist() == [True]


def test_frozen_trajectory_tail_is_not_used_for_motion_matching() -> None:
  trajectory = TorchTennisTrajectoryBatch(
    times=torch.tensor([0.0, 0.5, 1.0]),
    positions=torch.tensor([[[8.0, 0.0, 1.0], [6.0, 0.0, 1.0], [4.0, 0.0, 1.0]]]),
    velocities=torch.zeros(1, 3, 3),
    bounce_counts=torch.zeros(1, 3, dtype=torch.int8),
    net_cleared=torch.tensor([False]),
    valid_mask=torch.tensor([[True, False, False]]),
  )
  matched = match_tennis_trajectories_to_motions_torch(
    trajectory,
    torch.tensor([[6.0, 0.0, 1.0]]),
    torch.tensor([0.25]),
    torch.tensor([0.75]),
    maximum_distance=0.15,
    include_pre_bounce=True,
  )
  assert matched.valid.tolist() == [False]


def test_reachable_no_net_ball_already_on_robot_side_keeps_match_target() -> None:
  trajectory = TorchTennisTrajectoryBatch(
    times=torch.tensor([0.0, 0.5, 1.0]),
    positions=torch.tensor([[[1.0, 0.0, 1.0], [1.5, 0.0, 1.0], [2.0, 0.0, 1.0]]]),
    velocities=torch.tensor([[[1.0, 0.0, 0.0]] * 3]),
    bounce_counts=torch.zeros(1, 3, dtype=torch.int8),
    net_cleared=torch.tensor([False]),
  )
  matched = match_tennis_trajectories_to_motions_torch(
    trajectory,
    torch.tensor([[1.5, 0.0, 1.0]]),
    torch.tensor([0.25]),
    torch.tensor([0.75]),
    maximum_distance=0.15,
    include_pre_bounce=True,
  )
  assert matched.valid.tolist() == [True]
  torch.testing.assert_close(matched.target_positions, torch.tensor([[1.5, 0.0, 1.0]]))


def test_failure_trajectory_classification_uses_net_and_closest_xy_distance() -> None:
  trajectories = TorchTennisTrajectoryBatch(
    times=torch.tensor([0.0, 0.5, 1.0, 1.5]),
    positions=torch.tensor(
      [
        [[8.0, 0.0, 1.0], [5.0, 0.0, 2.0], [4.0, 0.0, 0.04], [2.5, 0.0, 0.5]],
        [[8.0, 0.0, 1.0], [5.0, 0.0, 2.0], [4.0, 0.0, 0.04], [3.2, 0.0, 0.5]],
        [[8.0, 0.0, 1.0], [5.0, 0.0, 0.5], [2.0, 0.0, 0.04], [1.0, 0.0, 0.5]],
        [[8.0, 0.0, 1.0], [7.0, 0.0, 1.2], [6.0, 0.0, 0.04], [5.8, 0.0, 0.5]],
      ]
    ),
    velocities=torch.zeros(4, 4, 3),
    bounce_counts=torch.tensor(
      [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
      dtype=torch.int8,
    ),
    net_cleared=torch.tensor([True, True, False, True]),
  )

  failure = classify_tennis_failure_trajectories(
    trajectories,
    root_positions_xy=torch.zeros(4, 2),
    trajectory_xy_distance_threshold_m=3.0,
    net_x=5.6,
    net_half_width=5.485,
  )

  assert failure.did_not_clear_net.tolist() == [False, False, True, True]
  assert failure.trajectory_too_far.tolist() == [False, True, False, False]
  assert failure.failure.tolist() == [False, True, True, True]
  torch.testing.assert_close(
    failure.closest_trajectory_distances,
    torch.tensor([2.5, 3.2, torch.inf, torch.inf]),
  )


def test_far_failure_launches_miss_root_and_still_target_net_height() -> None:
  count = 1024
  positions = torch.tensor([[9.0, 0.0, 1.0]]).expand(count, -1).clone()
  velocities = torch.tensor([[-4.0, 0.0, 4.0]]).expand(count, -1).clone()
  roots_xy = torch.zeros(count, 2)

  redirected = retarget_tennis_launches_to_miss_roots_torch(
    positions,
    velocities,
    roots_xy,
    minimum_xy_distance_m=3.0,
    net_crossing_height_range_m=(1.5, 3.5),
    maximum_initial_speed_m_s=20.0,
    net_x_m=5.6,
  )

  horizontal_speed = torch.linalg.vector_norm(redirected[:, :2], dim=1)
  line_distance = torch.abs(
    positions[:, 0] * redirected[:, 1] - positions[:, 1] * redirected[:, 0]
  ) / horizontal_speed
  time_to_net = (5.6 - positions[:, 0]) / redirected[:, 0]
  net_height = (
    positions[:, 2]
    + redirected[:, 2] * time_to_net
    - 0.5 * 9.81 * time_to_net.square()
  )

  assert torch.all(line_distance > 3.0)
  assert torch.all(redirected[:, 0] < 0.0)
  assert torch.all((net_height >= 1.5) & (net_height <= 3.5))
  torch.testing.assert_close(horizontal_speed, torch.full((count,), 4.0))


def test_motion_match_combines_spatial_and_contact_time_constraints() -> None:
  trajectories = TorchTennisTrajectoryBatch(
    times=torch.tensor([0.0, 1.0, 2.0, 3.0]),
    positions=torch.tensor(
      [[[4.0, 0.0, 1.0], [0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 0.1]]]
    ),
    velocities=torch.zeros(1, 4, 3),
    bounce_counts=torch.tensor([[0, 1, 1, 2]], dtype=torch.int8),
    net_cleared=torch.tensor([True]),
  )
  match = match_tennis_trajectories_to_motions_torch(
    trajectories,
    motion_target_positions=torch.tensor([[2.0, 0.0, 1.0], [2.1, 0.0, 1.0]]),
    minimum_contact_times=torch.tensor([0.5, 1.5]),
    nominal_contact_times=torch.tensor([1.1, 2.5]),
    maximum_distance=0.2,
    motion_chunk_size=1,
  )

  assert match.motion_ids.tolist() == [1]
  assert match.trajectory_indices.tolist() == [2]
  torch.testing.assert_close(match.contact_times, torch.tensor([2.0]))
  torch.testing.assert_close(match.distances, torch.tensor([0.1]))
  assert match.valid.tolist() == [True]


def test_motion_match_rejects_a_trajectory_that_does_not_clear_the_net() -> None:
  trajectories = TorchTennisTrajectoryBatch(
    times=torch.tensor([0.0, 1.0, 2.0]),
    positions=torch.zeros(1, 3, 3),
    velocities=torch.zeros(1, 3, 3),
    bounce_counts=torch.tensor([[0, 1, 1]], dtype=torch.int8),
    net_cleared=torch.tensor([False]),
  )
  match = match_tennis_trajectories_to_motions_torch(
    trajectories,
    motion_target_positions=torch.zeros(1, 3),
    minimum_contact_times=torch.tensor([0.5]),
    nominal_contact_times=torch.tensor([2.0]),
    maximum_distance=0.2,
  )

  assert torch.isinf(match.distances).all()
  assert match.valid.tolist() == [False]


def test_motion_match_offsets_command_deadline_from_trajectory_time() -> None:
  trajectories = TorchTennisTrajectoryBatch(
    times=torch.tensor([0.0, 1.0, 2.0]),
    positions=torch.tensor([[[3.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]]),
    velocities=torch.zeros(1, 3, 3),
    bounce_counts=torch.tensor([[0, 1, 1]], dtype=torch.int8),
    net_cleared=torch.tensor([True]),
  )
  match = match_tennis_trajectories_to_motions_torch(
    trajectories,
    motion_target_positions=torch.tensor([[1.0, 0.0, 1.0]]),
    minimum_contact_times=torch.tensor([1.01]),
    nominal_contact_times=torch.tensor([1.01]),
    maximum_distance=0.01,
    contact_time_offset_s=0.01,
  )

  assert match.trajectory_indices.tolist() == [1]
  torch.testing.assert_close(match.contact_times, torch.tensor([1.01]))
  assert match.valid.tolist() == [True]


def test_motion_match_accepts_per_trajectory_motion_targets() -> None:
  trajectories = TorchTennisTrajectoryBatch(
    times=torch.tensor([0.0, 1.0, 2.0]),
    positions=torch.tensor(
      [
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
      ]
    ),
    velocities=torch.zeros(2, 3, 3),
    bounce_counts=torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.int8),
    net_cleared=torch.tensor([True, True]),
  )
  match = match_tennis_trajectories_to_motions_torch(
    trajectories,
    motion_target_positions=torch.tensor(
      [
        [[1.0, 0.0, 1.0], [10.0, 0.0, 1.0]],
        [[10.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
      ]
    ),
    minimum_contact_times=torch.tensor([0.0, 0.0]),
    nominal_contact_times=torch.tensor([2.0, 2.0]),
    maximum_distance=0.01,
  )

  assert match.motion_ids.tolist() == [0, 1]
  assert match.trajectory_indices.tolist() == [1, 2]
  assert match.valid.tolist() == [True, True]


def test_motion_match_chunk_200_matches_single_motion_chunks() -> None:
  trajectories = TorchTennisTrajectoryBatch(
    times=torch.tensor([0.0, 1.0, 2.0]),
    positions=torch.tensor(
      [
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 2.0, 1.0]],
      ]
    ),
    velocities=torch.zeros(2, 3, 3),
    bounce_counts=torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.int8),
    net_cleared=torch.tensor([True, True]),
  )
  targets = torch.full((200, 3), 20.0)
  targets[17] = torch.tensor([1.0, 0.0, 1.0])
  targets[143] = torch.tensor([0.0, 2.0, 1.0])
  minimum_times = torch.zeros(200)
  nominal_times = torch.full((200,), 2.0)

  chunk_1 = match_tennis_trajectories_to_motions_torch(
    trajectories,
    targets,
    minimum_times,
    nominal_times,
    maximum_distance=0.2,
    motion_chunk_size=1,
  )
  chunk_200 = match_tennis_trajectories_to_motions_torch(
    trajectories,
    targets,
    minimum_times,
    nominal_times,
    maximum_distance=0.2,
    motion_chunk_size=200,
  )
  hierarchical = match_tennis_trajectories_to_motions_torch(
    trajectories,
    targets,
    minimum_times,
    nominal_times,
    maximum_distance=0.2,
    motion_chunk_size=200,
    hierarchical_top_k=32,
    hierarchical_coarse_neighbor_radius=1,
  )

  torch.testing.assert_close(chunk_200.distances, chunk_1.distances)
  torch.testing.assert_close(chunk_200.target_positions, chunk_1.target_positions)
  assert torch.equal(chunk_200.motion_ids, chunk_1.motion_ids)
  assert torch.equal(chunk_200.trajectory_indices, chunk_1.trajectory_indices)
  assert torch.equal(chunk_200.valid, chunk_1.valid)
  torch.testing.assert_close(hierarchical.distances, chunk_1.distances)
  torch.testing.assert_close(hierarchical.target_positions, chunk_1.target_positions)
  assert torch.equal(hierarchical.motion_ids, chunk_1.motion_ids)
  assert torch.equal(hierarchical.trajectory_indices, chunk_1.trajectory_indices)
  assert torch.equal(hierarchical.valid, chunk_1.valid)


def test_root_directed_sampler_respects_ballistic_height_and_speed_bounds() -> None:
  torch.manual_seed(7)
  root_xy = torch.tensor([[0.0, 0.0], [1.0, -0.5]]).repeat_interleave(64, dim=0)
  positions, velocities = sample_root_directed_tennis_launches_torch(
    root_xy,
    launch_position_min=torch.tensor([8.0, -2.0, 0.5]),
    launch_position_max=torch.tensor([10.0, 2.0, 1.3]),
    horizontal_speed_range_m_s=(3.5, 5.25),
    horizontal_angle_half_width_deg=15.0,
    net_crossing_height_range_m=(1.5, 3.5),
    maximum_initial_speed_m_s=7.0,
  )

  time_to_net = (STANDARD_TENNIS_PHYSICS.court.net_x_m - positions[:, 0]) / velocities[
    :, 0
  ]
  ballistic_height = (
    positions[:, 2] + velocities[:, 2] * time_to_net - 0.5 * 9.81 * time_to_net.square()
  )
  horizontal_direction = velocities[:, :2] / torch.linalg.vector_norm(
    velocities[:, :2], dim=1, keepdim=True
  )
  root_direction = root_xy - positions[:, :2]
  root_direction /= torch.linalg.vector_norm(root_direction, dim=1, keepdim=True)
  direction_cosine = torch.sum(horizontal_direction * root_direction, dim=1)

  assert torch.all(positions >= torch.tensor([8.0, -2.0, 0.5]))
  assert torch.all(positions <= torch.tensor([10.0, 2.0, 1.3]))
  assert torch.all(ballistic_height >= 1.5 - 1.0e-5)
  assert torch.all(ballistic_height <= 3.5 + 1.0e-5)
  assert torch.all(torch.linalg.vector_norm(velocities, dim=1) <= 7.0 + 1.0e-5)
  assert torch.all(direction_cosine >= math.cos(math.radians(15.0)) - 1.0e-5)


def test_root_directed_sampler_falls_back_without_crashing() -> None:
  root_xy = torch.tensor([[20.0, 100.0]])
  positions, velocities = sample_root_directed_tennis_launches_torch(
    root_xy,
    launch_position_min=torch.tensor([8.0, -2.0, 0.5]),
    launch_position_max=torch.tensor([10.0, 2.0, 1.3]),
    horizontal_speed_range_m_s=(3.5, 5.25),
    horizontal_angle_half_width_deg=15.0,
    net_crossing_height_range_m=(1.5, 3.5),
    maximum_initial_speed_m_s=7.0,
    maximum_resample_rounds=1,
  )

  assert torch.isfinite(positions).all()
  assert torch.isfinite(velocities).all()
  assert torch.linalg.vector_norm(velocities, dim=1).item() <= 7.0


def test_torch_match_task_is_isolated_from_base_launch_distill() -> None:
  base_cfg = unitree_g1_tennis_launch_distill_env_cfg()
  match_cfg = unitree_g1_tennis_launch_distill_torch_match_200_env_cfg()
  warp_cfg = unitree_g1_tennis_launch_distill_warp_match_200_env_cfg()
  root_directed_cfg = (
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_env_cfg()
  )
  no_tracking_cfg = (
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_env_cfg()
  )
  no_tracking_racket0_cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_env_cfg()
  floor03_cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg()
  legacy_cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_legacy_env_cfg()
  base_motion = base_cfg.commands["motion"]
  match_motion = match_cfg.commands["motion"]
  warp_motion = warp_cfg.commands["motion"]
  root_directed_motion = root_directed_cfg.commands["motion"]

  assert not base_motion.incoming_ball_torch_match_enabled
  assert base_motion.incoming_ball_trajectory_rollout_backend == "torch"
  assert base_motion.incoming_ball_strike_target_position_frame0 is not None
  assert match_motion.incoming_ball_torch_match_enabled
  assert match_motion.incoming_ball_trajectory_rollout_backend == "torch"
  assert warp_motion.incoming_ball_torch_match_enabled
  assert warp_motion.incoming_ball_trajectory_rollout_backend == "warp_fused"
  assert warp_motion.incoming_ball_torch_match_expected_motions == 200
  assert root_directed_motion.incoming_ball_trajectory_rollout_backend == "warp_fused"
  assert root_directed_cfg.sim.mujoco.timestep == 0.0025
  assert root_directed_cfg.decimation == 8
  assert no_tracking_cfg.sim.mujoco.timestep == 0.0025
  assert no_tracking_cfg.decimation == 8
  assert no_tracking_racket0_cfg.sim.mujoco.timestep == 0.0025
  assert no_tracking_racket0_cfg.decimation == 8
  assert legacy_cfg.sim.mujoco.timestep == 0.005
  assert legacy_cfg.decimation == 4
  assert root_directed_motion.incoming_ball_torch_match_cache_size == 0
  assert root_directed_motion.incoming_ball_torch_match_max_distance == 0.20
  assert root_directed_motion.incoming_ball_torch_match_adaptive_attempt_batch_size == 2
  assert root_directed_motion.incoming_ball_torch_match_motion_chunk_size == 200
  assert root_directed_motion.incoming_ball_torch_match_hierarchical_top_k == 32
  assert (
    root_directed_motion.incoming_ball_torch_match_hierarchical_coarse_neighbor_radius
    == 1
  )
  assert root_directed_motion.incoming_ball_torch_match_chain_batch_interval_steps == 8
  assert root_directed_motion.incoming_ball_torch_match_root_directed_sampling
  assert root_directed_motion.incoming_ball_torch_match_net_crossing_height_range_m == (
    1.5,
    3.5,
  )
  assert root_directed_motion.incoming_ball_torch_match_maximum_initial_speed_m_s == 7.0
  assert root_directed_motion.landing_target_mean == (7.0, 0.0, 0.0)
  assert root_directed_motion.landing_target_std == (0.0, 0.0, 0.0)
  assert root_directed_motion.landing_target_frame == "startup"
  assert "global_root_pos" not in root_directed_cfg.observations["student"].terms
  root_directed_heading = root_directed_cfg.observations["student"].terms[
    "startup_heading"
  ]
  assert isinstance(root_directed_heading.noise, UniformNoiseCfg)
  assert root_directed_heading.noise.n_min == -0.05
  assert root_directed_heading.noise.n_max == 0.05
  assert root_directed_cfg.observations["critic"].terms["startup_heading"].noise is None
  root_directed_ball_velocity = root_directed_cfg.observations["student"].terms[
    "ball_linear_velocity"
  ]
  assert isinstance(root_directed_ball_velocity.noise, GaussianNoiseCfg)
  assert root_directed_ball_velocity.noise.std == 0.1
  match_ball_velocity = match_cfg.observations["student"].terms["ball_linear_velocity"]
  warp_ball_velocity = warp_cfg.observations["student"].terms["ball_linear_velocity"]
  assert match_ball_velocity.noise.std == 0.2
  assert warp_ball_velocity.noise.std == 0.2
  root_directed_gravity = root_directed_cfg.observations["student"].terms[
    "projected_gravity"
  ]
  assert isinstance(root_directed_gravity.noise, UniformNoiseCfg)
  assert root_directed_gravity.noise.n_min == -0.05
  assert root_directed_gravity.noise.n_max == 0.05
  assert match_cfg.observations["student"].terms["projected_gravity"].noise is None
  assert warp_cfg.observations["student"].terms["projected_gravity"].noise is None
  assert (
    root_directed_cfg.observations["critic"].terms["projected_gravity"].noise is None
  )
  assert match_motion.landing_target_mean == (6.5, 0.0, 0.0)
  assert match_motion.landing_target_std == (0.5, 0.5, 0.0)
  assert warp_motion.landing_target_mean == (6.5, 0.0, 0.0)
  assert warp_motion.landing_target_std == (0.5, 0.5, 0.0)
  assert root_directed_cfg.rewards is not None
  assert match_cfg.rewards is not None
  assert warp_cfg.rewards is not None
  assert root_directed_cfg.rewards["racket_to_live_ball_reward"].weight == 50.0
  assert match_cfg.rewards["racket_to_live_ball_reward"].weight == 10.0
  assert warp_cfg.rewards["racket_to_live_ball_reward"].weight == 10.0
  heading_reward = root_directed_cfg.rewards["post_strike_heading_reward"]
  assert heading_reward.func is tennis_post_strike_heading_to_startup_x_reward
  assert heading_reward.weight == 2.0
  assert heading_reward.params == {
    "command_name": "motion",
    "recovery_delay_s": 0.3,
    "tolerance_degrees": 15.0,
    "std_degrees": 30.0,
  }
  assert "post_strike_heading_reward" not in match_cfg.rewards
  assert "post_strike_heading_reward" not in warp_cfg.rewards
  assert "target_position_reward" in root_directed_cfg.rewards
  assert "target_orientation_reward" not in root_directed_cfg.rewards
  assert "target_velocity_reward" not in root_directed_cfg.rewards
  assert "target_orientation_reward" in match_cfg.rewards
  assert "target_velocity_reward" in match_cfg.rewards
  assert "target_orientation_reward" in warp_cfg.rewards
  assert "target_velocity_reward" in warp_cfg.rewards
  assert match_motion.incoming_ball_torch_match_expected_motions == 200
  assert match_motion.incoming_ball_torch_match_attempts == 8
  assert match_motion.incoming_ball_torch_match_retry_rounds == 3
  assert match_motion.incoming_ball_torch_match_adaptive_attempt_batch_size == 0
  assert match_motion.incoming_ball_torch_match_motion_chunk_size == 4
  assert match_motion.incoming_ball_torch_match_hierarchical_top_k == 0
  assert match_motion.incoming_ball_torch_match_hierarchical_coarse_neighbor_radius == 1
  assert match_motion.incoming_ball_torch_match_chain_batch_interval_steps == 1
  assert match_motion.incoming_ball_torch_match_cache_size == 4
  assert base_motion.incoming_ball_torch_match_cache_size == 0
  assert match_motion.incoming_ball_torch_match_deadline_offset_s == 0.01
  assert match_motion.incoming_ball_torch_match_position_min == (8.0, -2.0, 0.5)
  assert match_motion.incoming_ball_torch_match_position_max == (10.0, 2.0, 1.3)
  assert match_motion.viz.show_incoming_ball_launch_region
  assert match_motion.incoming_ball_torch_match_velocity_min == (
    -5.25,
    -1.25,
    3.0,
  )
  assert match_motion.incoming_ball_torch_match_velocity_max == (-3.5, 1.25, 5.0)
  assert match_motion.landing_target_mean == (6.5, 0.0, 0.0)
  assert match_motion.landing_target_std == (0.5, 0.5, 0.0)
  assert match_cfg.rewards["ball_landing_reward"].params["target_out_speed"] == 3.5
  assert match_cfg.rewards["ball_landing_reward"].params["out_speed_std"] == 2.5
  assert base_cfg.rewards["ball_landing_reward"].params["target_out_speed"] == 7.0
  assert match_motion.incoming_ball_strike_target_position_frame0 is None
  assert match_motion.auto_chain_motion
  assert "motion_complete" not in (match_cfg.terminations or {})
  reference_observations = {
    "reference_motion_state",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "base_lin_vel",
  }
  assert reference_observations <= set(root_directed_cfg.observations["student"].terms)
  assert reference_observations.isdisjoint(match_cfg.observations["student"].terms)
  tracking_reward_weights = {
    "motion_global_root_pos": 0.5,
    "motion_global_root_ori": 0.5,
    "motion_body_pos": 1.0,
    "motion_body_ori": 1.0,
    "motion_body_lin_vel": 1.0,
    "motion_body_ang_vel": 1.0,
  }
  for reward_name, expected_weight in tracking_reward_weights.items():
    assert root_directed_cfg.rewards[reward_name].weight == expected_weight
    assert reward_name not in match_cfg.rewards
    assert reward_name not in no_tracking_cfg.rewards
  assert reference_observations.isdisjoint(
    no_tracking_cfg.observations["student"].terms
  )
  assert "startup_heading" in no_tracking_cfg.observations["student"].terms
  assert "global_root_pos" not in no_tracking_cfg.observations["student"].terms
  no_tracking_motion = no_tracking_cfg.commands["motion"]
  assert no_tracking_motion.landing_target_frame == "startup"
  assert no_tracking_motion.incoming_ball_torch_match_hierarchical_top_k == 32
  assert no_tracking_cfg.rewards["racket_to_live_ball_reward"].weight == 50.0
  assert no_tracking_racket0_cfg.rewards["racket_to_live_ball_reward"].weight == 0.0
  assert (
    "racket_incoming_velocity_orientation_reward" not in no_tracking_racket0_cfg.rewards
  )
  racket_orientation_reward = floor03_cfg.rewards[
    "racket_incoming_velocity_orientation_reward"
  ]
  assert (
    racket_orientation_reward.func is tennis_racket_incoming_velocity_orientation_reward
  )
  assert racket_orientation_reward.weight == 50.0
  assert racket_orientation_reward.params == {
    "command_name": "motion",
    "std": 0.5,
    "tolerance_degrees": 30.0,
    "minimum_ball_speed": 0.5,
    "ball_entity_name": "tennis_ball",
  }
  floor03_motion = floor03_cfg.commands["motion"]
  assert floor03_motion.viz.show_incoming_ball_racket_orientation_target
  assert floor03_motion.viz.incoming_ball_entity_name == "tennis_ball"
  assert floor03_motion.viz.incoming_ball_minimum_speed == 0.5
  assert floor03_motion.viz.racket_orientation_arrow_length == 0.7
  assert not no_tracking_racket0_cfg.commands[
    "motion"
  ].viz.show_incoming_ball_racket_orientation_target
  assert tuple(no_tracking_racket0_cfg.observations) == tuple(
    no_tracking_cfg.observations
  )
  assert tuple(no_tracking_racket0_cfg.observations["student"].terms) == tuple(
    no_tracking_cfg.observations["student"].terms
  )
  assert legacy_cfg.commands["motion"].landing_target_frame == "environment"
  assert "startup_heading" not in legacy_cfg.observations["student"].terms
  assert "startup_heading" not in legacy_cfg.observations["critic"].terms
  assert (
    legacy_cfg.rewards["post_strike_heading_reward"].func
    is tennis_post_strike_heading_to_landing_target_reward
  )
  base_physics_cfg = base_cfg.events["tennis_physics_domain_randomization"].params[
    "cfg"
  ]
  match_physics_cfg = match_cfg.events["tennis_physics_domain_randomization"].params[
    "cfg"
  ]
  assert not base_physics_cfg.explicit_ground_rebound
  assert match_physics_cfg.explicit_ground_rebound


def test_small_court_task_overrides_only_court_and_ball_sampling_geometry() -> None:
  baseline = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_env_cfg()
  small = unitree_g1_tennis_small_court_intent_transformer_sweet_speed50_racket_ground10_env_cfg()
  baseline_motion = baseline.commands["motion"]
  small_motion = small.commands["motion"]

  assert small_motion.tennis_court_net_x_m == 3.5
  assert small_motion.tennis_court_net_half_width_m == 2.5
  assert small_motion.incoming_ball_torch_match_position_min == (5.5, -1.5, 0.6)
  assert small_motion.incoming_ball_torch_match_position_max == (6.5, 1.5, 1.2)
  assert small_motion.incoming_ball_torch_match_horizontal_speed_range_m_s == (
    2.8,
    4.2,
  )
  assert small_motion.incoming_ball_torch_match_horizontal_angle_half_width_deg == 15.0
  assert small_motion.landing_target_mean == (6.0, 0.0, 0.0)
  assert small_motion.landing_target_std == (0.0, 0.0, 0.0)
  assert small.rewards is not None
  assert small.rewards["ball_landing_reward"].params["net_x"] == 3.5
  assert small.rewards["ball_landing_reward"].params["net_half_width"] == 2.5
  assert set(small.rewards) == set(baseline.rewards)
  for reward_name in small.rewards:
    assert small.rewards[reward_name].weight == baseline.rewards[reward_name].weight
  assert small.rewards["reference_sweet_spot_velocity_reward"].weight == 50.0
  assert tuple(small.observations) == tuple(baseline.observations)
  assert tuple(small.terminations or ()) == tuple(baseline.terminations or ())
  assert small_motion.incoming_ball_torch_match_cache_size == (
    baseline_motion.incoming_ball_torch_match_cache_size
  )
  assert small_motion.incoming_ball_torch_match_expected_motions == (
    baseline_motion.incoming_ball_torch_match_expected_motions
  )
  assert small.sim.mujoco.timestep == baseline.sim.mujoco.timestep
  assert small.decimation == baseline.decimation


def test_launch_region_visualization_is_fixed_in_environment_frame() -> None:
  command = object.__new__(MultiTargetMotionCommand)
  command.robot_anchor_body_index = 0
  half_sqrt = math.sqrt(0.5)
  command.robot = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor([[[110.0, 220.0, 0.8]]]),
      body_link_quat_w=torch.tensor([[[half_sqrt, 0.0, 0.0, half_sqrt]]]),
    )
  )
  command._env = SimpleNamespace(
    device=torch.device("cpu"),
    scene=SimpleNamespace(env_origins=torch.tensor([[100.0, 200.0, 0.0]])),
  )
  command.cfg = SimpleNamespace(
    incoming_ball_torch_match_enabled=True,
    incoming_ball_torch_match_position_min=(8.0, -2.0, 0.5),
    incoming_ball_torch_match_position_max=(10.0, 2.0, 1.3),
    viz=SimpleNamespace(show_incoming_ball_launch_region=True),
  )

  class BoxRecorder:
    box: dict | None = None

    def add_box(self, **kwargs) -> None:
      self.box = kwargs

  visualizer = BoxRecorder()
  command._add_incoming_ball_launch_region_vis(visualizer, 0)

  assert visualizer.box is not None
  torch.testing.assert_close(
    torch.from_numpy(visualizer.box["center"]),
    torch.tensor([109.0, 200.0, 0.9]),
  )
  torch.testing.assert_close(
    torch.from_numpy(visualizer.box["size"]),
    torch.tensor([1.0, 2.0, 0.4]),
  )
  torch.testing.assert_close(
    torch.from_numpy(visualizer.box["mat"]), torch.eye(3, dtype=torch.float64)
  )
  assert visualizer.box["color"] == (1.0, 0.85, 0.0, 0.20)


def test_cached_plan_is_aligned_to_current_robot_pose_for_motion_chain() -> None:
  planner = object.__new__(TorchIncomingBallMotionPlanner)
  planner.cache_size = 1
  planner.root_directed_sampling = False
  planner.env = SimpleNamespace(
    device=torch.device("cpu"),
    scene=SimpleNamespace(env_origins=torch.tensor([[100.0, 200.0, 0.0]])),
  )
  half_sqrt = math.sqrt(0.5)
  planner.motion = SimpleNamespace(
    robot_anchor_pos_w=torch.tensor([[110.0, 220.0, 0.8]]),
    robot_anchor_quat_w=torch.tensor([[half_sqrt, 0.0, 0.0, half_sqrt]]),
  )
  planner._plan_cache_ready = torch.tensor([True])
  planner._plan_cache = {
    "motion_ids": torch.tensor([[3]]),
    "target_indices": torch.tensor([[0]]),
    "target_positions_w": torch.tensor([[[101.0, 200.0, 1.0]]]),
    "contact_times": torch.tensor([[2.0]]),
    "launch_positions_w": torch.tensor([[[102.0, 200.0, 2.0]]]),
    "launch_linear_velocities_w": torch.tensor([[[1.0, 0.0, 0.0]]]),
    "launch_angular_velocities_w": torch.tensor([[[0.0, 1.0, 0.0]]]),
    "match_distances": torch.tensor([[0.05]]),
    "valid": torch.tensor([[True]]),
    "attempts": torch.tensor([[2]]),
  }

  plan = planner.plan_for_motion_chain(torch.tensor([0]))

  torch.testing.assert_close(
    plan.target_positions_w, torch.tensor([[110.0, 221.0, 1.0]])
  )
  torch.testing.assert_close(
    plan.launch_positions_w, torch.tensor([[110.0, 222.0, 2.0]])
  )
  torch.testing.assert_close(
    plan.launch_linear_velocities_w,
    torch.tensor([[0.0, 1.0, 0.0]]),
    atol=1e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    plan.launch_angular_velocities_w,
    torch.tensor([[-1.0, 0.0, 0.0]]),
    atol=1e-6,
    rtol=0.0,
  )


def test_root_directed_planning_targets_use_fixed_court_frame() -> None:
  planner = object.__new__(TorchIncomingBallMotionPlanner)
  planner.env = SimpleNamespace(
    device=torch.device("cpu"),
    scene=SimpleNamespace(env_origins=torch.tensor([[100.0, 200.0, 0.0]])),
  )
  half_sqrt = math.sqrt(0.5)
  planner.motion = SimpleNamespace(
    robot_anchor_pos_w=torch.tensor([[110.0, 220.0, 0.8]]),
    robot_anchor_quat_w=torch.tensor([[half_sqrt, 0.0, 0.0, half_sqrt]]),
  )
  planner.motion_target_positions = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 1.2]])

  root_xy, targets = planner._planning_targets(torch.tensor([0]), align_to_robot=True)

  torch.testing.assert_close(root_xy, torch.tensor([[10.0, 20.0]]))
  torch.testing.assert_close(
    targets,
    torch.tensor([[[10.0, 21.0, 1.0], [8.0, 20.0, 1.2]]]),
    atol=1.0e-6,
    rtol=0.0,
  )


def test_adaptive_attempts_stop_after_all_environments_match() -> None:
  planner = object.__new__(TorchIncomingBallMotionPlanner)
  planner.attempt_count = 8
  planner.retry_rounds = 3
  planner.adaptive_attempt_batch_size = 2
  planner.verbose_samples = False
  planner.env = SimpleNamespace(
    device=torch.device("cpu"),
    scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
  )
  planner.target_indices = torch.tensor([0])
  planner.initial_angular_velocity = torch.zeros(3)
  planner.fallback_position = torch.zeros(3)
  planner.fallback_velocity = torch.zeros(3)
  planner._physics_parameters = lambda env_ids: tuple(
    torch.ones(len(env_ids)) for _ in range(4)
  )
  planner._planning_targets = lambda env_ids, align_to_robot: (
    torch.zeros(len(env_ids), 2),
    torch.zeros(len(env_ids), 1, 3),
  )
  sample_calls: list[tuple[int, int]] = []

  def sample_wave(
    mass,
    restitution,
    tangent_retention,
    drag,
    root_positions_xy,
    motion_target_positions,
    *,
    attempt_count,
  ) -> MotionResamplePlan:
    del restitution, tangent_retention, drag, root_positions_xy
    del motion_target_positions
    count = len(mass)
    sample_calls.append((count, attempt_count))
    if len(sample_calls) == 1:
      distances = torch.tensor([0.1, 0.3])
      valid = torch.tensor([True, False])
    else:
      distances = torch.tensor([0.05])
      valid = torch.tensor([True])
    return MotionResamplePlan(
      motion_ids=torch.zeros(count, dtype=torch.long),
      target_indices=torch.zeros(count, dtype=torch.long),
      target_positions_w=torch.zeros(count, 3),
      contact_times=torch.ones(count),
      launch_positions_w=torch.zeros(count, 3),
      launch_linear_velocities_w=torch.zeros(count, 3),
      launch_angular_velocities_w=torch.zeros(count, 3),
      match_distances=distances,
      valid=valid,
      attempts=torch.ones(count, dtype=torch.long),
    )

  planner._sample_candidate_round = sample_wave
  planner._simulate_trajectories = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError("fallback simulation should not run for finite matches")
  )

  plan = planner._plan_uncached(torch.tensor([0, 1]), align_to_robot=True)

  assert sample_calls == [(2, 2), (1, 2)]
  assert plan.valid.tolist() == [True, True]
  assert plan.attempts.tolist() == [1, 3]


def test_failure_probability_replaces_requested_rows_with_failure_plans() -> None:
  planner = object.__new__(TorchIncomingBallMotionPlanner)
  planner.attempt_count = 2
  planner.retry_rounds = 1
  planner.adaptive_attempt_batch_size = 0
  planner.failure_trajectory_probability = 1.0
  planner.verbose_samples = False
  planner.env = SimpleNamespace(
    device=torch.device("cpu"),
    scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
  )
  planner.target_indices = torch.tensor([0])
  planner.initial_angular_velocity = torch.zeros(3)
  planner.fallback_position = torch.zeros(3)
  planner.fallback_velocity = torch.zeros(3)
  planner._physics_parameters = lambda env_ids: tuple(
    torch.ones(len(env_ids)) for _ in range(4)
  )
  planner._planning_targets = lambda env_ids, align_to_robot: (
    torch.zeros(len(env_ids), 2),
    torch.zeros(len(env_ids), 1, 3),
  )

  def successful_plan(*args, **kwargs) -> MotionResamplePlan:
    count = len(args[0])
    return MotionResamplePlan(
      motion_ids=torch.zeros(count, dtype=torch.long),
      target_indices=torch.zeros(count, dtype=torch.long),
      target_positions_w=torch.ones(count, 3),
      contact_times=torch.ones(count),
      launch_positions_w=torch.ones(count, 3),
      launch_linear_velocities_w=torch.ones(count, 3),
      launch_angular_velocities_w=torch.zeros(count, 3),
      match_distances=torch.full((count,), 0.05),
      valid=torch.ones(count, dtype=torch.bool),
      attempts=torch.ones(count, dtype=torch.long),
    )

  def failure_plan(*args, **kwargs) -> MotionResamplePlan:
    count = len(args[0])
    return MotionResamplePlan(
      motion_ids=torch.zeros(count, dtype=torch.long),
      target_indices=torch.zeros(count, dtype=torch.long),
      target_positions_w=torch.full((count, 3), 2.0),
      contact_times=torch.full((count,), 3.0),
      launch_positions_w=torch.full((count, 3), 3.0),
      launch_linear_velocities_w=torch.full((count, 3), 4.0),
      launch_angular_velocities_w=torch.zeros(count, 3),
      match_distances=torch.full((count,), 0.5),
      valid=torch.zeros(count, dtype=torch.bool),
      attempts=torch.full((count,), 2, dtype=torch.long),
    )

  planner._sample_candidate_round = successful_plan
  planner._sample_failure_round = failure_plan
  planner._simulate_trajectories = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError("fallback simulation should not run")
  )

  plan = planner._plan_uncached(torch.tensor([0, 1]), align_to_robot=True)

  assert plan.valid.tolist() == [False, False]
  torch.testing.assert_close(plan.contact_times, torch.full((2,), 3.0))
  torch.testing.assert_close(plan.launch_positions_w, torch.full((2, 3), 3.0))


def test_initial_plan_disables_failure_injection() -> None:
  planner = object.__new__(TorchIncomingBallMotionPlanner)
  planner.cache_size = 0
  planner.retry_rounds = 1
  calls: list[bool] = []

  def plan_uncached(
    env_ids: torch.Tensor,
    *,
    align_to_robot: bool = False,
    allow_failure_trajectories: bool = True,
  ) -> MotionResamplePlan:
    del align_to_robot
    calls.append(allow_failure_trajectories)
    count = len(env_ids)
    return MotionResamplePlan(
      motion_ids=torch.zeros(count, dtype=torch.long),
      target_indices=torch.zeros(count, dtype=torch.long),
      target_positions_w=torch.zeros(count, 3),
      contact_times=torch.ones(count),
      launch_positions_w=torch.zeros(count, 3),
      launch_linear_velocities_w=torch.zeros(count, 3),
      launch_angular_velocities_w=torch.zeros(count, 3),
      match_distances=torch.zeros(count),
      valid=torch.ones(count, dtype=torch.bool),
      attempts=torch.ones(count, dtype=torch.long),
    )

  planner._plan_uncached = plan_uncached

  plan = planner(torch.tensor([0, 1]))

  assert plan.valid.tolist() == [True, True]
  assert calls == [False]


def test_failure_radial20_task_matches_student_and_intent_position_encoding() -> None:
  cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_radial20_env_cfg()
  motion = cfg.commands["motion"]
  assert motion.incoming_ball_failure_trajectory_probability == 0.05
  assert motion.incoming_ball_failure_rollout_horizon_s == 12.0
  assert motion.incoming_ball_failure_observation_radius_m == 20.0
  assert motion.incoming_ball_failure_stop_speed_m_s == 0.05
  assert motion.incoming_ball_failure_random_direction_fraction == 0.2
  assert motion.incoming_ball_failure_random_position_radius_range_m == (2.0, 5.0)
  assert motion.incoming_ball_failure_random_position_height_range_m == (0.8, 2.2)
  assert motion.incoming_ball_failure_overhead_fraction == 0.2
  student_encoding = cfg.observations["student"].terms["ball_position"].params[
    "position_encoding"
  ]
  intent_encoding = cfg.observations["intent_ball_history"].terms["position"].params[
    "position_encoding"
  ]
  assert student_encoding == intent_encoding == {
    "linear_radius_m": 10.0,
    "limit_radius_m": 20.0,
  }


def test_existing_failure_trajectory_task_keeps_its_previous_settings() -> None:
  cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_env_cfg()
  motion = cfg.commands["motion"]
  assert motion.incoming_ball_failure_rollout_horizon_s is None
  assert motion.incoming_ball_failure_observation_radius_m is None
  assert motion.incoming_ball_failure_stop_speed_m_s is None
  assert "position_encoding" not in cfg.observations["student"].terms[
    "ball_position"
  ].params
