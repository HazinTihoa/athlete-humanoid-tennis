"""Tests for NatNet ball tracking and dropout prediction."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np


DEPLOY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_DIR))

from utils.natnet_bridge_state import (
    BallEstimatorConfig,
    BallStateEstimator,
    MeasurementFirstBallStream,
    propagate_ball_state,
)
from utils.mocap_frame_calibration import (  # noqa: E402
    planar_calibration_from_root_pose,
    quaternion_xyzw_multiply,
    quaternion_xyzw_to_matrix,
)


class NatNetBallEstimatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BallEstimatorConfig(
            drag_coefficient=0.0,
            integration_dt_s=0.001,
            short_dropout_s=0.10,
            maximum_prediction_s=1.0,
        )

    def test_static_ball_holds_position_and_zero_velocity(self) -> None:
        estimator = BallStateEstimator(self.config)
        position = np.array([1.0, 2.0, 0.5])
        for index in range(20):
            estimator.observe(position, index / 120.0)

        estimate = estimator.estimate(1.0)
        self.assertIsNotNone(estimate)
        estimated_position, estimated_velocity = estimate
        np.testing.assert_allclose(estimated_position, position, atol=1.0e-6)
        np.testing.assert_allclose(estimated_velocity, 0.0, atol=1.0e-9)
        self.assertEqual(estimator.mode(1.0), "static_hold")

    def test_prediction_resolves_ground_bounce(self) -> None:
        position, velocity, bounces = propagate_ball_state(
            np.array([0.0, 0.0, 0.05]),
            np.array([1.0, 0.0, -2.0]),
            0.05,
            self.config,
        )
        self.assertGreaterEqual(bounces, 1)
        self.assertGreaterEqual(position[2], self.config.radius_m)
        self.assertGreater(velocity[2], 0.0)

    def test_short_bounce_dropout_keeps_track_epoch(self) -> None:
        estimator = BallStateEstimator(self.config)
        estimator.observe(np.array([0.0, 0.0, 0.05]), 0.0)
        estimator.is_static = False
        estimator.velocity_w[:] = (1.0, 0.0, -2.0)
        predicted = estimator.estimate(0.05)
        self.assertIsNotNone(predicted)
        epoch = estimator.track_epoch

        new_epoch = estimator.observe(predicted[0], 0.05)
        self.assertFalse(new_epoch)
        self.assertEqual(estimator.track_epoch, epoch)

    def test_bounce_dropout_beyond_short_window_keeps_track_epoch(self) -> None:
        estimator = BallStateEstimator(self.config)
        estimator.observe(np.array([0.0, 0.0, 0.10]), 0.0)
        estimator.is_static = False
        estimator.velocity_w[:] = (2.0, 0.0, -2.0)
        predicted = estimator.estimate(0.15)
        self.assertIsNotNone(predicted)
        epoch = estimator.track_epoch

        new_epoch = estimator.observe(predicted[0], 0.15)

        self.assertFalse(new_epoch)
        self.assertEqual(estimator.track_epoch, epoch)
        self.assertEqual(len(estimator.samples), 1)

    def test_predictable_dropout_keeps_track_epoch(self) -> None:
        estimator = BallStateEstimator(self.config)
        estimator.observe(np.array([0.0, 0.0, 1.0]), 0.0)
        epoch = estimator.track_epoch

        new_epoch = estimator.observe(np.array([0.2, 0.0, 1.0]), 0.20)

        self.assertFalse(new_epoch)
        self.assertEqual(estimator.track_epoch, epoch)

    def test_reassociation_at_timeout_boundary_keeps_track_epoch(self) -> None:
        estimator = BallStateEstimator(self.config)
        estimator.observe(np.array([0.0, 0.0, 1.0]), 0.0)
        epoch = estimator.track_epoch

        new_epoch = estimator.observe(np.array([0.2, 0.0, 1.0]), 0.50)

        self.assertFalse(new_epoch)
        self.assertEqual(estimator.track_epoch, epoch)

    def test_reassociation_timeout_starts_new_track_epoch(self) -> None:
        estimator = BallStateEstimator(self.config)
        estimator.observe(np.array([0.0, 0.0, 1.0]), 0.0)
        epoch = estimator.track_epoch

        new_epoch = estimator.observe(np.array([0.2, 0.0, 1.0]), 0.51)

        self.assertTrue(new_epoch)
        self.assertEqual(estimator.track_epoch, epoch + 1)
        np.testing.assert_allclose(estimator.velocity_w, 0.0)

    def test_large_motion_within_timeout_keeps_track_epoch(self) -> None:
        estimator = BallStateEstimator(self.config)
        estimator.observe(np.array([0.0, 0.0, 1.0]), 0.0)
        epoch = estimator.track_epoch

        new_epoch = estimator.observe(np.array([1.0, 0.0, 1.0]), 0.20)

        self.assertFalse(new_epoch)
        self.assertEqual(estimator.track_epoch, epoch)

    def test_planar_root_calibration_maps_home_pose_to_policy_origin(self) -> None:
        raw_position = np.array([-1.395, -0.441, 0.779])
        raw_yaw = np.radians(-72.0)
        raw_quaternion = np.array(
            [0.0, 0.0, np.sin(raw_yaw / 2.0), np.cos(raw_yaw / 2.0)]
        )

        rotation, translation, yaw_quaternion = planar_calibration_from_root_pose(
            raw_position,
            raw_quaternion,
            np.zeros(2),
            0.0,
        )
        calibrated_position = rotation @ raw_position + translation
        calibrated_quaternion = quaternion_xyzw_multiply(
            yaw_quaternion, raw_quaternion
        )
        calibrated_rotation = quaternion_xyzw_to_matrix(calibrated_quaternion)
        calibrated_yaw = np.arctan2(
            calibrated_rotation[1, 0], calibrated_rotation[0, 0]
        )

        np.testing.assert_allclose(calibrated_position, [0.0, 0.0, 0.779])
        self.assertAlmostEqual(float(calibrated_yaw), 0.0)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)


class MeasurementFirstStreamTest(unittest.TestCase):
    def setUp(self):
        self.stream = MeasurementFirstBallStream(BallEstimatorConfig())

    def test_normal_120hz_frames_have_no_extra_timer_output(self):
        for i in range(30):
            stamp = i / 120.0
            position = np.array([stamp, 0., .5])
            sample = self.stream.observe(position, stamp)
            np.testing.assert_array_equal(sample.position_w, position)
            self.assertEqual(sample.source, "measured")
            self.assertIsNone(self.stream.prediction_tick(stamp + .002))

    def test_missing_frame_fills_next_tick_without_waiting_20ms(self):
        self.stream.observe(np.array([0., 0., 1.]), 0.)
        self.assertIsNone(self.stream.prediction_tick(.001))
        original_samples = list(self.stream.estimator.samples)
        self.stream.estimator.is_static = False
        self.stream.estimator.velocity_w[:] = [1., 0., 1.]
        for i in range(1, 6):
            stamp = .001 + i / 120.0
            sample = self.stream.prediction_tick(stamp)
            self.assertIsNotNone(sample)
            self.assertEqual(sample.timestamp_s, stamp)
            self.assertEqual(sample.source, "predicted")
            expected = self.stream.estimator.estimate(stamp)
            np.testing.assert_allclose(sample.position_w, expected[0])
            np.testing.assert_allclose(sample.velocity_w, expected[1])
        self.assertEqual(len(self.stream.estimator.samples), len(original_samples))
        self.assertEqual(self.stream.estimator.last_valid_time_s, 0.)

    def test_recovery_immediately_overrides_prediction_and_keeps_epoch(self):
        self.stream.observe(np.array([0., 0., 1.]), 0.)
        self.stream.prediction_tick(.001)
        self.stream.prediction_tick(.010)
        sample = self.stream.observe(np.array([.2, .3, .4]), .011)
        np.testing.assert_array_equal(sample.position_w, [.2, .3, .4])
        self.assertEqual(sample.track_epoch, 1)
        self.assertIsNone(self.stream.prediction_tick(.018))

    def test_duplicate_invalid_samples_do_not_suppress_timer_fills(self):
        self.stream.observe(np.array([0., 0., 1.]), 1.)
        self.stream.prediction_tick(1.001)
        self.assertIsNone(self.stream.observe(np.zeros(3), 1.))
        self.assertIsNone(self.stream.observe(np.zeros(3), .9))
        self.assertIsNone(self.stream.observe(np.array([np.nan, 0, 0]), 1.01))
        self.assertIsNotNone(self.stream.prediction_tick(1.01))
        self.assertIsNone(self.stream.prediction_tick(1.01))

    def test_no_prediction_before_initialization(self):
        self.assertIsNone(self.stream.prediction_tick(0.))

    def test_long_dropout_hold_is_cached_and_recovery_does_not_integrate_gap(self):
        self.stream.observe(np.array([0., 0., 1.]), 0.)
        self.stream.estimator.is_static = False
        self.stream.estimator.velocity_w[:] = [1., 0., 1.]
        self.stream.prediction_tick(.001)
        first = self.stream.prediction_tick(2.1)
        with patch.object(self.stream.estimator, "estimate", side_effect=AssertionError("hold must be cached")):
            held = self.stream.prediction_tick(2.2)
        np.testing.assert_array_equal(held.position_w, first.position_w)
        np.testing.assert_array_equal(held.velocity_w, 0.)
        with patch("utils.natnet_bridge_state.propagate_ball_state", side_effect=AssertionError("do not integrate a minute-long gap")):
            recovered = self.stream.observe(np.ones(3), 60.)
        self.assertEqual(recovered.track_epoch, 2)
        self.assertIsNone(self.stream.prediction_tick(60.001))


if __name__ == "__main__":
    unittest.main()
