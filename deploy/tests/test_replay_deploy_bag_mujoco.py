"""Offline replay contracts, with no ROS nodes or hardware interaction."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimation.mocap.replay_deploy_bag_mujoco import (
    Calibration, OfflineKalmanConfig, PhysicsBallKalman, Recording, Replay,
    Samples, build_kalman_ball_stream, load_config, local_positions_to_world,
    recover_calibrations, trail_segments, transform_raw_positions,
)
from utils.natnet_bridge_state import BallEstimatorConfig, propagate_ball_state


def samples(times, values, valid=None, source=None):
    times = np.asarray(times, dtype=float)
    return Samples(times, np.asarray(values, dtype=float),
                   np.ones(len(times), dtype=bool) if valid is None else np.asarray(valid, bool),
                   times.copy() if source is None else np.asarray(source, dtype=float))


def paired_poses(raw_times, yaw, translation, *, moving=False):
    n = len(raw_times)
    positions = np.tile([1.2, -.7, .8], (n, 1))
    orientations = Rotation.from_euler("xyz", np.tile([.04, -.1, .7], (n, 1)))
    if moving:
        positions[:, 0] += np.arange(n) * .1
    transform = Rotation.from_euler("z", yaw)
    q = orientations.as_quat()
    q[::2] *= -1  # Quaternion sign is not a physical orientation change.
    raw = samples(raw_times, np.column_stack((positions, q)))
    published = samples(np.asarray(raw_times) + .015,
                        np.column_stack((transform.apply(positions) + translation,
                                         (transform * orientations).as_quat())),
                        source=np.asarray(raw_times) + .001)
    return raw, published


def kalman_config(**overrides):
    values = dict(
        radius_m=.0335,
        mass_kg=.0577,
        air_density_kg_m3=1.225,
        drag_coefficient=.55,
        court_restitution=.745,
        tangent_speed_retention=.825,
        ground_height_m=0.,
        integration_dt_s=.0025,
    )
    values.update(overrides)
    return OfflineKalmanConfig(**values)


class CalibrationTests(unittest.TestCase):
    def test_stationary_pelvis_recovers_yaw_not_just_translation(self):
        raw, published = paired_poses(np.arange(1, 1.10, .01), -.7, [-.5, .9, 0.])
        transforms = recover_calibrations(raw, published, np.array([.99]))
        self.assertEqual(len(transforms), 1)
        transform = transforms[0]
        np.testing.assert_allclose(transform.rotation, Rotation.from_euler("z", -.7).as_matrix(), atol=1e-12)
        np.testing.assert_allclose(transform.translation, [-.5, .9, 0.], atol=1e-12)
        self.assertLess(transform.position_p95, 1e-12)
        self.assertLess(transform.orientation_p95_deg, 1e-10)

    def test_pair_using_publication_stamp_not_future_receive_time(self):
        raw, published = paired_poses(np.arange(1, 1.1, .01), .3, [1., -2., 0.], moving=True)
        transform = recover_calibrations(raw, published, np.array([]))[0]
        np.testing.assert_allclose(transform.translation, [1., -2., 0.], atol=1e-12)

    def test_multiple_calibrations_transform_ball_in_correct_segment(self):
        a, b = paired_poses([1., 1.01, 1.02, 1.03], .2, [1., 2., 0.])
        c, d = paired_poses([3., 3.01, 3.02, 3.03], -.4, [-2., 1., 0.])
        raw = samples(np.r_[a.time, c.time], np.r_[a.values, c.values])
        pub = samples(np.r_[b.time, d.time], np.r_[b.values, d.values], source=np.r_[b.source_time, d.source_time])
        transforms = recover_calibrations(raw, pub, np.array([.9, 2.9]))
        self.assertEqual(len(transforms), 2)
        balls = samples([.5, 1.02, 3.02], [[.3, .4, .5]] * 3)
        result = transform_raw_positions(balls, transforms)
        np.testing.assert_allclose(result.values[0], balls.values[0])
        for index, transform in enumerate(transforms, start=1):
            np.testing.assert_allclose(result.values[index], transform.rotation @ balls.values[index] + transform.translation)
        np.testing.assert_allclose(result.values[:, 2], .5)

    def test_rejects_z_offset_instead_of_silently_changing_floor(self):
        raw, published = paired_poses([1., 1.01, 1.02, 1.03], 0., [0., 0., .15])
        with self.assertRaisesRegex(ValueError, "Frame calibration is inconsistent"):
            recover_calibrations(raw, published, np.array([]))

    def test_no_synchronized_pairs_is_an_explicit_error(self):
        raw, published = paired_poses([1., 1.01, 1.02, 1.03], 0., [0., 0., 0.])
        published.source_time[:] = 2.
        with self.assertRaisesRegex(ValueError, "Not enough synchronized"):
            recover_calibrations(raw, published, np.array([]))


class TimelineTests(unittest.TestCase):
    def test_hold_latest_does_not_use_future_or_interpolate_invalid_ball(self):
        stream = samples([1., 1.01, 1.02], [[1.], [2.], [3.]], [True, False, True])
        self.assertIsNone(stream.current(.99, 1.)[0])
        np.testing.assert_allclose(stream.current(1.005, .05)[0], [1.])
        self.assertIsNone(stream.current(1.015, .05)[0])
        self.assertIsNone(stream.current(1.1, .05)[0])
        value, age = stream.current(1.015, float("inf"), hold_valid=True)
        np.testing.assert_allclose(value, [1.])
        self.assertAlmostEqual(age, .015)

    def test_trails_break_at_invalid_samples_gaps_and_calibration(self):
        times = [0., .01, .02, .03, .20, .21, .22, .23, .24]
        positions = np.column_stack((times, np.zeros((len(times), 2))))
        stream = samples(times, positions, [True, True, False, True, True, True, True, True, True])
        transform = Calibration(.215, np.eye(3), np.zeros(3), 3, 0., 0.)
        begin, end = trail_segments(stream, .235, 1., [transform])
        np.testing.assert_allclose(begin[:, 0], [0., .20, .22])
        np.testing.assert_allclose(end[:, 0], [.01, .21, .23])
        self.assertTrue(np.all(end[:, 0] <= .235))

    def test_empty_trail_and_reassociation_jump(self):
        stream = samples([1., 1.01], [[0., 0., 1.], [4., 0., 1.]])
        self.assertEqual(len(trail_segments(stream, .5, 1., [])[0]), 0)
        self.assertEqual(len(trail_segments(stream, 1.02, 1., [])[0]), 0)

    def test_sweet_observation_is_transformed_from_pelvis_to_world(self):
        root_q = Rotation.from_euler("z", np.pi / 2).as_quat()
        pelvis = samples([1., 2.], [[10., 20., .8, *root_q]] * 2)
        local = samples([1.01, 1.02], [[1., 2., 3.], [-1., 0., .5]])
        world = local_positions_to_world(local, pelvis)
        np.testing.assert_allclose(world.values[0], [8., 21., 3.8], atol=1e-12)
        np.testing.assert_allclose(world.values[1], [10., 19., 1.3], atol=1e-12)

    def test_stale_pelvis_invalidates_sweet_observation(self):
        pelvis = samples([1.], [[0., 0., 0., 0., 0., 0., 1.]])
        local = samples([1.3], [[1., 2., 3.]])
        world = local_positions_to_world(local, pelvis, max_root_age=.25)
        self.assertFalse(world.valid[0])


class KalmanTests(unittest.TestCase):
    def test_static_noisy_ball_stays_static(self):
        estimator = PhysicsBallKalman(kalman_config())
        center = np.array([.4, -.2, .9])
        for index in range(120):
            noise = .001 * np.array([
                np.sin(index * .7), np.cos(index * .4), np.sin(index * .2)
            ])
            estimator.observe(center + noise, index / 120.)
        position, velocity, age = estimator.estimate(1.0)
        np.testing.assert_allclose(position, center, atol=.003)
        np.testing.assert_allclose(velocity, 0., atol=1e-12)
        self.assertAlmostEqual(age, 1 / 120.)

    def test_ballistic_motion_survives_short_measurement_dropout(self):
        config = kalman_config()
        estimator = PhysicsBallKalman(config)
        initial_position = np.array([.2, -.4, 1.2])
        initial_velocity = np.array([5., 1., 3.])
        for timestamp in np.arange(-.10, 0., 1 / 120.):
            estimator.observe(initial_position, timestamp)
        physics = BallEstimatorConfig(
            radius_m=config.radius_m,
            mass_kg=config.mass_kg,
            air_density_kg_m3=config.air_density_kg_m3,
            drag_coefficient=config.drag_coefficient,
            court_restitution=config.court_restitution,
            tangent_speed_retention=config.tangent_speed_retention,
            integration_dt_s=config.integration_dt_s,
        )
        final_position = final_velocity = None
        for timestamp in np.arange(0., .6 + 1e-9, 1 / 120.):
            final_position, final_velocity, _ = propagate_ball_state(
                initial_position, initial_velocity, timestamp, physics
            )
            if not .22 <= timestamp <= .30:
                estimator.observe(final_position, timestamp)
        position, velocity, _ = estimator.estimate(.6)
        np.testing.assert_allclose(position, final_position, atol=.04)
        np.testing.assert_allclose(velocity, final_velocity, atol=.5)

    def test_single_outlier_is_rejected_without_state_jump(self):
        estimator = PhysicsBallKalman(kalman_config())
        for index in range(20):
            estimator.observe(np.array([index * .01, 0., 1.]), index * .01)
        before = estimator.estimate(.20)[0]
        estimator.observe(np.array([20., -10., 8.]), .20)
        after = estimator.estimate(.20)[0]
        self.assertEqual(estimator.rejected_measurements, 1)
        self.assertLess(np.linalg.norm(after - before), .2)

    def test_three_consistent_reacquisition_samples_start_new_track(self):
        estimator = PhysicsBallKalman(kalman_config())
        for index in range(20):
            estimator.observe(np.array([index * .01, 0., 1.]), index * .01)
        for timestamp, x in ((.20, 10.), (.21, 10.02), (.22, 10.04)):
            estimator.observe(np.array([x, 0., 1.]), timestamp)
        position, velocity, _ = estimator.estimate(.22)
        self.assertEqual(estimator.reinitializations, 1)
        np.testing.assert_allclose(position, [10.04, 0., 1.], atol=1e-12)
        self.assertGreater(velocity[0], 1.)

    def test_replay_stream_does_not_use_measurement_received_in_future(self):
        raw = samples(
            [.10, .30],
            [[1., 2., 1.], [4., 5., 1.]],
            source=[.10, .15],
        )
        clock = samples([.20, .25], [[0.], [0.]], source=[.20, .25])
        calibration = Calibration(-np.inf, np.eye(3), np.zeros(3), 1, 0., 0.)
        stream, _ = build_kalman_ball_stream(
            raw, clock, [calibration], {"tennis_ball_physics": {}}
        )
        np.testing.assert_allclose(stream.values[:, :3], [[1., 2., 1.]] * 2)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config, path = load_config("g1_tppo_student_m14_9.yaml")
        cls.root_pos = np.array([.4, -.3, .81])
        cls.root_quat = Rotation.from_euler("xyz", [.12, -.2, .6]).as_quat()
        root = samples([1., 2.], [np.r_[cls.root_pos, cls.root_quat]] * 2)
        cls.joints = np.linspace(-.12, .12, 29)
        joint = samples([1., 2.], [cls.joints, cls.joints + .01])
        ball = samples([1., 2.], [[.5, -.3, 1.2, 0, 0, 0, 1.]] * 2)
        observed_sweet = samples([1., 2.], [[.2, -.1, 1.1], [.2, -.1, 1.1]])
        velocity = samples([1., 2.], [[1., 0., 0.]] * 2)
        kalman = samples(
            [1., 2.],
            [np.r_[ball.values[0, :3], [1., 0., 0.]]] * 2,
        )
        policy_ball = samples([1., 2.], [ball.values[0, :3]] * 2)
        recording = Recording(
            path=Path("unused-bag"),
            pelvis=root,
            joints=joint,
            raw_ball=ball,
            filtered_ball=ball,
            filtered_ball_velocity=velocity,
            kalman_ball=kalman,
            policy_ball=policy_ball,
            observed_sweet=observed_sweet,
            fsm=samples([1., 2.], [[2], [0]]),
            calibrations=[],
            kalman_config=kalman_config(),
            kalman_stats={"accepted": 2, "rejected": 0, "reinitialized": 0},
            start=1.,
            end=2.,
            first_control=1.,
        )
        cls.replay = Replay(recording, config, path)

    def test_recorded_pose_and_joint_order_without_physics_steps(self):
        replay = self.replay
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("Replay must not step physics")):
            replay.set_time(1.2)
            np.testing.assert_allclose(replay.data.xpos[replay.root_id], self.root_pos, atol=1e-12)
            np.testing.assert_allclose(replay.data.xmat[replay.root_id].reshape(3, 3),
                                       Rotation.from_quat(self.root_quat).as_matrix(), atol=1e-12)
            for i, name in enumerate(replay.config["joint_names"].split(",")):
                self.assertAlmostEqual(replay.data.joint(name.strip()).qpos[0], self.joints[i])
            before = replay.data.qpos.copy()
            replay.set_time(2.)
            replay.set_time(1.2)
            np.testing.assert_allclose(replay.data.qpos, before)

    def test_trails_frames_and_balls_are_drawn_in_model_world(self):
        replay = self.replay
        state = replay.set_time(1.02)
        scene = mujoco.MjvScene(replay.model, maxgeom=100)
        replay.draw(scene, 1.02, state, 1.)
        # Three state balls, two arrows, policy ball, two sweet points, error line, and RGB axes.
        self.assertEqual(scene.ngeom, 12)
        np.testing.assert_allclose(scene.geoms[0].pos, state["raw"][:3], atol=1e-6)
        np.testing.assert_allclose(scene.geoms[1].pos, state["filtered"][:3], atol=1e-6)
        np.testing.assert_allclose(scene.geoms[2].pos, state["kalman"][:3], atol=1e-6)
        np.testing.assert_allclose(scene.geoms[5].pos, state["policy_ball"], atol=1e-6)
        self.assertGreater(replay.model.nlight, 0)
        visible_planes = ((replay.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
                          & (replay.model.geom_rgba[:, 3] > 0))
        self.assertEqual(np.count_nonzero(visible_planes), 1)


if __name__ == "__main__":
    unittest.main()
