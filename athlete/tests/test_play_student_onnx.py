import math
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import mujoco
import numpy as np
import pytest
from athlete.scripts.play_student_onnx import (
  INCOMING_MATCH_DEADLINE_OFFSET_S,
  INTENT_BALL_HISTORY_STEPS,
  INTENT_BALL_TOKEN_DIM,
  INTENT_CURRENT_OBSERVATION_DIM,
  INTENT_HISTORY_LAGS,
  INTENT_OBSERVATION_DIM,
  INTENT_ROBUST_BALL_TOKEN_DIM,
  INTENT_ROBUST_OBSERVATION_DIM,
  INTENT_STATE_HISTORY_STEPS,
  INTENT_STATE_TOKEN_DIM,
  STANDALONE_RACKET_CONTACT_DAMPING,
  STANDALONE_RACKET_RESTITUTION,
  StudentOnnxPlayConfig,
  _GoalDistribution,
  _RootDirectedIncomingBallPlanner,
  _is_robot_half_singles_point,
  _quat_from_rpy,
  _resolve_court_profile,
  _sample_manual_bounce_time,
  _StandaloneStudentSimulation,
  _ViewerStatusOverlay,
  _draw_sweet_point,
)
from athlete.scripts.tennis_physics import (
  STANDARD_TENNIS_DOMAIN_RANDOMIZATION,
  contact_damping_for_restitution,
)


def test_sweet_point_marker_tracks_site_and_can_be_hidden() -> None:
  scene = mujoco.MjvScene(mujoco.MjModel.from_xml_string("<mujoco/>"), maxgeom=1)
  for position in (np.array([0.38, 0.01, 0.27]), np.array([1.0, 2.0, 3.0])):
    _draw_sweet_point(scene, position, visible=True)
    assert scene.ngeom == 1
    np.testing.assert_allclose(scene.geoms[0].pos, position)
    np.testing.assert_allclose(scene.geoms[0].size, [0.008] * 3)
    assert scene.geoms[0].label == "sweet point"
  _draw_sweet_point(scene, position, visible=False)
  assert scene.ngeom == 0


def test_student_play_defaults_match_floor005_training_task() -> None:
  cfg = StudentOnnxPlayConfig(onnx_policy_file=Path("student.onnx"))

  assert cfg.physics_dt == 0.0025
  assert cfg.control_dt == 0.02
  assert cfg.court_preset == "standard"
  assert cfg.landing_target_x is None
  assert cfg.landing_target_y is None
  assert cfg.landing_target_std_x == 0.0
  assert cfg.landing_target_std_y == 0.0
  assert cfg.landing_target_min_x is None
  assert cfg.show_landing_target
  assert not cfg.timing_plot
  assert not cfg.manual_mode
  assert not cfg.manual_mouse_select
  assert cfg.manual_launch_x == 9.0
  assert cfg.manual_bounce_x == 3.5
  assert cfg.manual_bounce_time_s == 1.2
  assert not cfg.manual_random_bounce_time
  assert cfg.manual_bounce_time_min_s == 0.9
  assert cfg.manual_bounce_time_max_s == 1.5
  assert cfg.ball_observation_latency_min_steps == 0
  assert cfg.ball_observation_latency_max_steps == 3
  assert cfg.ball_observation_dropout_start_probability == 0.03
  assert cfg.ball_observation_dropout_duration_min_steps == 2
  assert cfg.ball_observation_dropout_duration_max_steps == 5
  expected_restitution = (
    sum(STANDARD_TENNIS_DOMAIN_RANDOMIZATION.racket_restitution) / 2.0
  )
  assert np.isclose(STANDALONE_RACKET_RESTITUTION, expected_restitution)
  assert np.isclose(
    STANDALONE_RACKET_CONTACT_DAMPING,
    contact_damping_for_restitution(expected_restitution),
  )
  assert INCOMING_MATCH_DEADLINE_OFFSET_S == 0.01


def test_small_court_preset_matches_training_geometry_and_sampling() -> None:
  court = _resolve_court_profile("small")

  assert court.net_x_m == 3.5
  assert court.net_half_width_m == 2.5
  assert court.launch_position_min == (5.5, -1.5, 0.6)
  assert court.launch_position_max == (6.5, 1.5, 1.2)
  assert court.horizontal_speed_range_m_s == (2.8, 4.2)
  assert court.landing_target_xy == (6.0, 0.0)
  assert court.viewer_lookat == (3.9, 0.0, 0.55)
  assert court.viewer_distance == 14.0
  assert _is_robot_half_singles_point(3.5, 2.75, court)
  assert not _is_robot_half_singles_point(3.51, 0.0, court)
  assert not _is_robot_half_singles_point(0.0, 2.76, court)


@pytest.fixture
def landing_simulation() -> _StandaloneStudentSimulation:
  simulation = object.__new__(_StandaloneStudentSimulation)
  simulation.cfg = StudentOnnxPlayConfig(
    onnx_policy_file=Path("student.onnx"),
    court_preset="small",
    landing_target_x=5.0,
    landing_target_std_x=1.5,
    landing_target_std_y=1.5,
    landing_target_min_x=3.75,
  )
  simulation.court = _resolve_court_profile("small")
  simulation.rng = np.random.default_rng(20260906)
  simulation._set_startup_anchor(np.array([0.0, 0.0, 0.8]), np.eye(3))
  return simulation


def test_landing_target_min_x_matches_training_truncated_gaussian(
  landing_simulation: _StandaloneStudentSimulation,
) -> None:
  simulation = landing_simulation
  samples = np.stack([simulation._sample_landing_target() for _ in range(10000)])
  assert np.all(samples[:, 0] >= 3.75)
  assert not np.any(samples[:, 0] == 3.75)
  alpha = (3.75 - 5.0) / 1.5
  tail = 0.5 * math.erfc(alpha / math.sqrt(2.0))
  shift = math.exp(-alpha**2 / 2.0) / math.sqrt(2.0 * math.pi) / tail
  assert np.mean(samples[:, 0]) == pytest.approx(5.0 + 1.5 * shift, abs=0.03)
  assert np.std(samples[:, 1]) == pytest.approx(1.5, abs=0.03)
  assert np.all(samples[:, 2] == 0.0)


def test_landing_ground_transform_uses_only_startup_yaw(
  landing_simulation: _StandaloneStudentSimulation,
) -> None:
  yaw = 0.6
  rotation = np.empty(9)
  mujoco.mju_quat2Mat(rotation, _quat_from_rpy(np.array([0.2, 0.3, yaw])))
  landing_simulation._set_startup_anchor(
    np.array([0.3, -0.4, 0.85]), rotation.reshape(3, 3)
  )
  target = np.array([5.0, 1.0, 0.0])
  expected = [
    0.3 + 5.0 * math.cos(yaw) - math.sin(yaw),
    -0.4 + 5.0 * math.sin(yaw) + math.cos(yaw),
    0.0,
  ]
  np.testing.assert_allclose(landing_simulation._landing_target_to_world(target), expected)


def test_rotated_landing_gaussian_keeps_world_bound_and_xy_covariance(
  landing_simulation: _StandaloneStudentSimulation,
) -> None:
  simulation = landing_simulation
  simulation.cfg = replace(simulation.cfg, landing_target_std_y=0.7)
  yaw = 0.65
  rotation = np.empty(9)
  mujoco.mju_quat2Mat(rotation, _quat_from_rpy(np.array([0.2, -0.1, yaw])))
  simulation._set_startup_anchor(np.array([0.3, -0.4, 0.82]), rotation.reshape(3, 3))
  samples = np.stack([simulation._sample_landing_target() for _ in range(10000)])
  world = np.stack([simulation._landing_target_to_world(sample) for sample in samples])
  planar_rotation = np.array([
    [math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]
  ])
  mean_w = np.array([0.3, -0.4]) + planar_rotation @ [5.0, 0.0]
  covariance = planar_rotation @ np.diag([1.5**2, 0.7**2]) @ planar_rotation.T
  std_x = math.sqrt(covariance[0, 0])
  alpha = (3.75 - mean_w[0]) / std_x
  tail = 0.5 * math.erfc(alpha / math.sqrt(2.0))
  shift = math.exp(-alpha**2 / 2.0) / math.sqrt(2.0 * math.pi) / tail
  expected_mean = mean_w + covariance[:, 0] / std_x * shift
  expected_covariance = covariance + np.outer(covariance[:, 0], covariance[0]) / (
    std_x**2
  ) * (alpha * shift - shift**2)
  assert np.all(world[:, 0] >= 3.75 - 1.0e-12)
  assert not np.any(np.isclose(world[:, 0], 3.75, atol=1.0e-10, rtol=0.0))
  assert np.all(world[:, 2] == 0.0)
  np.testing.assert_allclose(world[:, :2].mean(axis=0), expected_mean, atol=0.03)
  np.testing.assert_allclose(np.cov(world[:, :2], rowvar=False), expected_covariance, atol=0.04)


@pytest.mark.parametrize("minimum, expected_x", [(None, 5.0), (3.75, 5.0), (5.5, 5.5)])
def test_zero_std_landing_target_stays_finite(
  landing_simulation: _StandaloneStudentSimulation,
  minimum: float | None,
  expected_x: float,
) -> None:
  simulation = landing_simulation
  simulation.cfg = replace(
    simulation.cfg, landing_target_std_x=0.0, landing_target_std_y=0.0,
    landing_target_min_x=minimum,
  )
  np.testing.assert_allclose(simulation._sample_landing_target(), [expected_x, 0.0, 0.0])


def test_viewer_sync_picks_up_landing_and_manual_marker_model_changes() -> None:
  simulation = object.__new__(_StandaloneStudentSimulation)
  simulation.cfg = StudentOnnxPlayConfig(onnx_policy_file=Path("student.onnx"))
  simulation.policy = SimpleNamespace(observation_dim=INTENT_OBSERVATION_DIM)
  simulation.model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody>'
    '<geom type="cylinder" size="0.5 0.008"/>'
    '<geom type="cylinder" size="0.06 0.012"/>'
    '<geom type="cylinder" size="0.28 0.01"/>'
    '</worldbody></mujoco>'
  )
  simulation.landing_target_marker_geom_id = 0
  simulation.landing_target_center_geom_id = 1
  simulation.manual_bounce_marker_geom_id = 2
  simulation._viewer_model_dirty = False
  simulation.sweet_spot_site_id = 0
  simulation.data = SimpleNamespace(site_xpos=np.zeros((1, 3)))
  viewer = MagicMock()
  viewer.user_scn = mujoco.MjvScene(simulation.model, maxgeom=1)

  for target in ([5.8134, 1.3177, 0.0], [4.9919, -0.7684, 0.0]):
    simulation.landing_target_position_w = np.array(target)
    simulation._update_landing_target_marker()
    simulation.sync_viewer(viewer)
    viewer.sync.assert_called_with(state_only=False)
    np.testing.assert_allclose(simulation.model.geom_pos[0, :2], target[:2])
    simulation.sync_viewer(viewer)
    viewer.sync.assert_called_with(state_only=True)

  simulation.manual_bounce_position_xy_w = np.array([2.0, 1.0])
  simulation._update_manual_bounce_marker()
  simulation.sync_viewer(viewer)
  viewer.sync.assert_called_with(state_only=False)
  simulation.sync_viewer(viewer)
  viewer.sync.assert_called_with(state_only=True)


def test_failed_viewer_sync_keeps_pending_model_changes() -> None:
  simulation = object.__new__(_StandaloneStudentSimulation)
  simulation._viewer_model_dirty = True
  simulation.cfg = StudentOnnxPlayConfig(onnx_policy_file=Path("student.onnx"))
  simulation.sweet_spot_site_id = 0
  simulation.data = SimpleNamespace(site_xpos=np.zeros((1, 3)))
  viewer = MagicMock()
  viewer.user_scn = mujoco.MjvScene(
    mujoco.MjModel.from_xml_string("<mujoco/>"), maxgeom=1
  )
  viewer.sync.side_effect = RuntimeError("sync failed")
  with pytest.raises(RuntimeError, match="sync failed"):
    simulation.sync_viewer(viewer)
  assert simulation._viewer_model_dirty


def test_manual_mouse_selection_accepts_only_robot_half_singles_court() -> None:
  assert _is_robot_half_singles_point(0.0, 0.0)
  assert _is_robot_half_singles_point(5.6, 4.115)
  assert not _is_robot_half_singles_point(5.61, 0.0)
  assert not _is_robot_half_singles_point(0.0, 4.116)


def test_manual_bounce_time_can_be_uniformly_sampled() -> None:
  cfg = StudentOnnxPlayConfig(
    onnx_policy_file=Path("student.onnx"),
    manual_random_bounce_time=True,
    manual_bounce_time_min_s=0.9,
    manual_bounce_time_max_s=1.5,
  )
  rng = np.random.default_rng(42)
  samples = np.array([_sample_manual_bounce_time(cfg, rng) for _ in range(100)])

  assert np.all(samples >= 0.9)
  assert np.all(samples <= 1.5)
  assert np.ptp(samples) > 0.4


def test_viewer_overlay_displays_realtime_ball_speed() -> None:
  class _Viewer:
    texts = None

    def set_texts(self, texts: object) -> None:
      self.texts = texts

  viewer = _Viewer()
  overlay = _ViewerStatusOverlay(control_dt=0.02, realtime=True)

  overlay.update(viewer, ball_speed_mps=10.0)  # type: ignore[arg-type]

  assert viewer.texts is not None
  assert "Ball speed" in viewer.texts[2]
  assert "10.00 m/s (36.0 km/h)" in viewer.texts[3]


def test_intent_onnx_observation_contract_is_628_dims() -> None:
  assert INTENT_CURRENT_OBSERVATION_DIM == 133
  assert INTENT_STATE_HISTORY_STEPS * INTENT_STATE_TOKEN_DIM == 465
  assert INTENT_BALL_HISTORY_STEPS * INTENT_BALL_TOKEN_DIM == 30
  assert INTENT_OBSERVATION_DIM == 628
  assert INTENT_BALL_HISTORY_STEPS * INTENT_ROBUST_BALL_TOKEN_DIM == 40
  assert INTENT_ROBUST_OBSERVATION_DIM == 638


def test_strided_history_backfills_and_uses_oldest_to_newest_lags() -> None:
  history: deque[np.ndarray] = deque(maxlen=25)
  history.append(np.array([1.0, 10.0]))
  first = _StandaloneStudentSimulation._gather_strided_history(
    history, INTENT_HISTORY_LAGS
  )
  np.testing.assert_array_equal(first, np.tile([1.0, 10.0], 5))

  for value in range(2, 26):
    history.append(np.array([float(value), float(value * 10)]))
  selected = _StandaloneStudentSimulation._gather_strided_history(
    history, INTENT_HISTORY_LAGS
  ).reshape(5, 2)
  np.testing.assert_array_equal(selected[:, 0], [5.0, 10.0, 15.0, 20.0, 25.0])


def test_startup_heading_uses_only_current_yaw() -> None:
  yaw = np.deg2rad(90.0)
  rotation = np.array(
    [
      [np.cos(yaw), -np.sin(yaw), 0.0],
      [np.sin(yaw), np.cos(yaw), 0.0],
      [0.0, 0.0, 1.0],
    ]
  )
  heading = _StandaloneStudentSimulation._startup_heading_b(rotation)
  np.testing.assert_allclose(heading, [0.0, -1.0], atol=1.0e-12)


def test_continuous_motion_targets_follow_root_xy_and_yaw_not_height() -> None:
  distribution = _GoalDistribution(
    source_file=Path("motion.npz"),
    clip="fh_test",
    frames=100,
    strike_frame=50,
    fps=50.0,
    position_mean=np.array([1.0, 0.0, 0.5]),
    aligned_position_mean=np.array([1.0, 0.0, 1.25]),
    position_std=np.zeros(3),
    velocity_mean=np.zeros(3),
    velocity_std=np.zeros(3),
    orientation_rpy_mean=np.zeros(3),
    orientation_rpy_std=np.zeros(3),
    sampling_weight=1.0,
    initial_anchor_position_w=np.array([0.0, 0.0, 0.75]),
    initial_anchor_quaternion_w=np.array([1.0, 0.0, 0.0, 0.0]),
    initial_anchor_linear_velocity_w=np.zeros(3),
    initial_anchor_angular_velocity_w=np.zeros(3),
    initial_joint_position=np.zeros(29),
    initial_joint_velocity=np.zeros(29),
  )
  pelvis_position = np.array([2.0, 3.0, 8.0])
  pitch = np.deg2rad(20.0)
  pelvis_rotation = np.array(
    [
      [0.0, -np.cos(pitch), np.sin(pitch)],
      [1.0, 0.0, 0.0],
      [0.0, np.sin(pitch), np.cos(pitch)],
    ]
  )

  targets = _RootDirectedIncomingBallPlanner._aligned_motion_targets_w(
    [distribution], pelvis_position, pelvis_rotation
  )

  np.testing.assert_allclose(targets[0], [2.0, 4.0, 1.25], atol=1.0e-12)


def test_manual_launch_shooting_hits_requested_first_bounce() -> None:
  distribution = _GoalDistribution(
    source_file=Path("motion.npz"),
    clip="fh_test",
    frames=200,
    strike_frame=150,
    fps=50.0,
    position_mean=np.array([2.0, 0.0, 0.7]),
    # Deliberately impossible as a motion target: manual mode must ignore it.
    aligned_position_mean=np.array([100.0, 100.0, 100.0]),
    position_std=np.zeros(3),
    velocity_mean=np.zeros(3),
    velocity_std=np.zeros(3),
    orientation_rpy_mean=np.zeros(3),
    orientation_rpy_std=np.zeros(3),
    sampling_weight=1.0,
    initial_anchor_position_w=np.array([0.0, 0.0, 0.76]),
    initial_anchor_quaternion_w=np.array([1.0, 0.0, 0.0, 0.0]),
    initial_anchor_linear_velocity_w=np.zeros(3),
    initial_anchor_angular_velocity_w=np.zeros(3),
    initial_joint_position=np.zeros(29),
    initial_joint_velocity=np.zeros(29),
  )
  planner = _RootDirectedIncomingBallPlanner([distribution], seed=42)

  plan = planner.sample_manual(
    np.array([0.0, 0.0, 0.76]),
    np.eye(3),
    launch_position_w=np.array([9.0, 0.5, 1.0]),
    bounce_position_xy_w=np.array([3.5, -0.2]),
    bounce_time_s=1.2,
    solver_iterations=5,
    maximum_initial_speed_m_s=15.0,
  )

  assert plan.first_bounce_position_w is not None
  assert plan.first_bounce_time_s is not None
  assert not plan.motion_matched
  assert math.isnan(plan.match_distance_m)
  np.testing.assert_allclose(plan.first_bounce_position_w[:2], [3.5, -0.2], atol=0.03)
  assert abs(plan.first_bounce_time_s - 1.2) <= 0.02
  assert np.linalg.norm(plan.launch_velocity_w) <= 15.0


def test_manual_launch_rejects_a_trajectory_that_does_not_clear_net() -> None:
  distribution = _GoalDistribution(
    source_file=Path("motion.npz"),
    clip="fh_test",
    frames=200,
    strike_frame=150,
    fps=50.0,
    position_mean=np.zeros(3),
    aligned_position_mean=np.zeros(3),
    position_std=np.zeros(3),
    velocity_mean=np.zeros(3),
    velocity_std=np.zeros(3),
    orientation_rpy_mean=np.zeros(3),
    orientation_rpy_std=np.zeros(3),
    sampling_weight=1.0,
    initial_anchor_position_w=np.array([0.0, 0.0, 0.76]),
    initial_anchor_quaternion_w=np.array([1.0, 0.0, 0.0, 0.0]),
    initial_anchor_linear_velocity_w=np.zeros(3),
    initial_anchor_angular_velocity_w=np.zeros(3),
    initial_joint_position=np.zeros(29),
    initial_joint_velocity=np.zeros(29),
  )
  planner = _RootDirectedIncomingBallPlanner([distribution], seed=42)

  with pytest.raises(RuntimeError, match="does not clear the physical net"):
    planner.sample_manual(
      np.array([0.0, 0.0, 0.76]),
      np.eye(3),
      launch_position_w=np.array([9.0, 0.0, 1.0]),
      bounce_position_xy_w=np.array([3.5, 0.0]),
      bounce_time_s=0.5,
      solver_iterations=5,
      maximum_initial_speed_m_s=20.0,
    )
