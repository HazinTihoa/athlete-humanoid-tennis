"""Tests for mocap-and-Bridge-only bag replay helpers."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimation.mocap.replay_bag_ball import (
    BRIDGE_COLOR,
    DIRECT_COLOR,
    RAW_COLOR,
    TRAIL_POINT_RADIUS_M,
    TrajectoryReplay,
    _direct_estimator_config,
    _match_by_time,
    align_raw_positions_to_bridge_frame,
    audit_measurement_first,
    fit_planar_transform,
    parse_args,
    simulate_measurement_first,
    trail_points,
)
from utils.natnet_bridge_state import BallStateEstimator


class FrameAlignmentTests(unittest.TestCase):
    def test_stationary_pelvis_pose_recovers_yaw_and_xy_translation(self):
        count = 20
        source_position = np.tile([1.2, -0.7, 0.82], (count, 1))
        source_rotation = Rotation.from_euler("xyz", [0.04, -0.08, 0.65])
        source_quaternion = np.tile(source_rotation.as_quat(), (count, 1))
        source = np.column_stack((source_position, source_quaternion))

        yaw = -0.73
        rotation = Rotation.from_euler("z", yaw)
        translation = np.array([-0.4, 0.9, 0.0])
        target_position = rotation.apply(source_position) + translation
        target_quaternion = np.tile(
            (rotation * source_rotation).as_quat(), (count, 1)
        )
        target = np.column_stack((target_position, target_quaternion))

        fitted_rotation, fitted_translation = fit_planar_transform(source, target)
        np.testing.assert_allclose(fitted_rotation, rotation.as_matrix(), atol=1e-12)
        np.testing.assert_allclose(fitted_translation, translation, atol=1e-12)

    def test_transform_preserves_world_z_offset_policy(self):
        source = np.tile([1.0, 2.0, 0.8, 0.0, 0.0, 0.0, 1.0], (4, 1))
        target = source.copy()
        target[:, :3] += [0.2, -0.3, 0.15]
        _, translation = fit_planar_transform(source, target)
        np.testing.assert_allclose(translation, [0.2, -0.3, 0.0], atol=1e-12)

    def test_time_matching_uses_nearest_receive_sample(self):
        query = np.array([1.009, 1.021])
        reference = np.array([1.0, 1.01, 1.02, 1.03])
        query_ids, reference_ids = _match_by_time(query, reference, 0.02)
        np.testing.assert_array_equal(query_ids, [0, 1])
        np.testing.assert_array_equal(reference_ids, [1, 2])

    def test_calibration_is_applied_only_to_its_time_segment(self):
        raw_t = np.arange(0.0, 2.0, 0.01)
        raw_position = np.tile([0.3, -0.2, 1.0], (raw_t.size, 1))
        raw_pose = np.column_stack(
            (
                np.tile([1.2, -0.7, 0.82], (raw_t.size, 1)),
                np.tile(Rotation.from_euler("z", 0.4).as_quat(), (raw_t.size, 1)),
            )
        )
        target_pose = raw_pose.copy()
        after = raw_t >= 1.0
        frame_rotation = Rotation.from_euler("z", -0.6)
        translation = np.array([0.8, 0.3, 0.0])
        target_pose[after, :3] = frame_rotation.apply(raw_pose[after, :3]) + translation
        target_pose[after, 3:] = (
            frame_rotation * Rotation.from_quat(raw_pose[after, 3:])
        ).as_quat()

        aligned, segments = align_raw_positions_to_bridge_frame(
            raw_t,
            raw_position,
            raw_t,
            raw_pose,
            raw_t,
            target_pose,
            np.array([1.0]),
        )

        np.testing.assert_allclose(aligned[~after], raw_position[~after], atol=1e-12)
        np.testing.assert_allclose(
            aligned[after],
            frame_rotation.apply(raw_position[after]) + translation,
            atol=1e-12,
        )
        self.assertEqual(len(segments), 2)


class TrailTests(unittest.TestCase):
    def test_scatter_retains_samples_without_interpolating_over_gaps(self):
        times = np.array([0.00, 0.01, 0.02, 0.10, 0.11, 0.12, 0.13])
        values = np.column_stack((times, np.zeros((times.size, 2))))
        values[-1, 0] = 10.0
        valid = np.array([True, True, False, True, True, True, True])
        points = trail_points(
            times,
            values,
            valid,
            timestamp=0.13,
            duration_s=1.0,
        )
        np.testing.assert_allclose(points[:, 0], [0.00, 0.01, 0.10, 0.11, 0.12, 10.0])

    def test_two_second_history_excludes_older_and_future_points(self):
        times = np.arange(0.0, 4.0, 0.5)
        positions = np.column_stack((times, np.zeros((len(times), 2))))
        points = trail_points(times, positions, np.ones(len(times), dtype=bool), 3.0, 2.0)
        np.testing.assert_array_equal(points[:, 0], [1.0, 1.5, 2.0, 2.5, 3.0])

    def test_default_history_is_two_seconds(self):
        with patch.object(sys, "argv", ["replay_bag_ball.py"]):
            self.assertEqual(parse_args().trail_seconds, 2.0)


class MeasurementFirstTests(unittest.TestCase):
    @staticmethod
    def dataset(count=61):
        times = np.arange(count) / 120.0
        position = np.column_stack((2 * times, times, np.ones(count)))
        return {
            "raw_t_s": times,
            "raw_source_t_s": times - 0.002,
            "raw_pos_w": position,
            "raw_tracking_valid": np.ones(count, dtype=bool),
            "filtered_t_s": np.array([]),
        }

    def test_normal_frames_exactly_forwarded_without_timer_duplicates(self):
        data = self.dataset()
        output = simulate_measurement_first(data, {})
        np.testing.assert_array_equal(output["direct_t_s"], data["raw_t_s"])
        np.testing.assert_array_equal(output["direct_pos_w"], data["raw_pos_w"])
        np.testing.assert_array_equal(output["direct_raw_index"], np.arange(61))
        np.testing.assert_allclose(output["direct_measurement_age_s"], 0.002)
        self.assertTrue(np.all(output["direct_mode"] == "measured"))

    def test_gap_predicts_at_timer_time_then_snaps_to_reacquired_measurement(self):
        data = self.dataset()
        data["raw_tracking_valid"][20:29] = False
        output = simulate_measurement_first(data, {})
        predicted = output["direct_raw_index"] < 0
        ticks = output["direct_t_s"][predicted]
        self.assertEqual(len(ticks), 9)
        np.testing.assert_allclose(np.diff(ticks), 1 / 120.0)
        self.assertAlmostEqual(ticks[0] - data["raw_t_s"][19], 1 / 120.0)
        estimator = BallStateEstimator(_direct_estimator_config({}))
        for i in range(20):
            estimator.observe(data["raw_pos_w"][i], data["raw_source_t_s"][i])
        expected_p, expected_v = estimator.estimate(ticks[0])
        np.testing.assert_allclose(output["direct_pos_w"][predicted][0], expected_p)
        np.testing.assert_allclose(output["direct_velocity_w"][predicted][0], expected_v)
        row = np.flatnonzero(output["direct_raw_index"] == 29)[0]
        np.testing.assert_array_equal(output["direct_pos_w"][row], data["raw_pos_w"][29])
        self.assertEqual(output["direct_epoch"][row], output["direct_epoch"][0])

    def test_prediction_does_not_use_future_measurements(self):
        data = self.dataset()
        data["raw_tracking_valid"][20:29] = False
        original = simulate_measurement_first(data, {})
        data["raw_pos_w"][29:] += [3, 2, 1]
        changed = simulate_measurement_first(data, {})
        before = original["direct_t_s"] < data["raw_t_s"][29]
        np.testing.assert_array_equal(original["direct_pos_w"][before], changed["direct_pos_w"][before])
        np.testing.assert_array_equal(original["direct_velocity_w"][before], changed["direct_velocity_w"][before])

    def test_invalid_positions_and_duplicate_source_times_not_forwarded(self):
        data = self.dataset()
        data["raw_tracking_valid"][10:12] = False
        data["raw_source_t_s"][12] = data["raw_source_t_s"][9]
        data["raw_pos_w"][13] = np.nan
        output = simulate_measurement_first(data, {})
        for i in range(10, 14):
            self.assertNotIn(i, output["direct_raw_index"])
        self.assertTrue(np.isfinite(output["direct_pos_w"]).all())

    def test_tracked_position_survives_rigid_body_fit_error_and_invalid_orientation(self):
        data = self.dataset()
        data["raw_mean_error"] = np.full(61, 0.03)
        data["raw_pose_valid"] = np.zeros(61, dtype=bool)
        data["raw_mean_error"][15] = np.nan
        output = simulate_measurement_first(data, {"mocap_bridge": {"ball_max_mean_error_m": .01}})
        np.testing.assert_array_equal(output["direct_raw_index"], np.arange(61))
        np.testing.assert_array_equal(output["direct_pos_w"], data["raw_pos_w"])
        self.assertTrue(np.all(output["direct_mode"] == "measured"))

    def test_tracking_positions_at_bounce_are_not_replaced_with_predicted_ground_height(self):
        data = self.dataset()
        data["raw_mean_error"] = np.full(61, 0.03)
        data["raw_pos_w"][:, 2] = np.abs(data["raw_t_s"] - .25) - .007
        output = simulate_measurement_first(data, {})
        np.testing.assert_array_equal(output["direct_pos_w"], data["raw_pos_w"])
        self.assertFalse(np.any(output["direct_raw_index"] < 0))

    def test_audit_detects_valid_frames_silently_rejected_by_old_quality_gate(self):
        data = self.dataset()
        data["raw_mean_error"] = np.zeros(61)
        data["raw_mean_error"][20:29] = .03
        rejected = dict(data)
        rejected["raw_tracking_valid"] = data["raw_tracking_valid"].copy()
        rejected["raw_tracking_valid"][20:29] = False
        data.update(simulate_measurement_first(rejected, {}))
        old = audit_measurement_first(data)
        self.assertEqual(old["unforwarded_tracked_frames"], 9)
        self.assertEqual(old["mismatched_tracked_frames"], 9)
        self.assertGreater(old["fresh_timeline_mismatches"], 0)
        data.update(simulate_measurement_first(data, {}))
        corrected = audit_measurement_first(data)
        self.assertEqual(corrected["unforwarded_tracked_frames"], 0)
        self.assertEqual(corrected["mismatched_tracked_frames"], 0)
        self.assertEqual(corrected["fresh_timeline_mismatches"], 0)

    def test_recorded_ball_bridge_only_bag_all_tracked_positions_agree(self):
        path = Path(__file__).resolve().parents[1] / "recordings/2026-09-09_20-20-17_ball_bridge_only/ball_trajectory.npz"
        if not path.is_file():
            self.skipTest("Local regression recording is not available")
        with np.load(path) as data:
            dataset = dict(data)
        dataset.update(simulate_measurement_first(dataset, {}))
        audit = audit_measurement_first(dataset)
        self.assertGreater(audit["tracked_position_frames"], 6000)
        self.assertEqual(audit["unforwarded_tracked_frames"], 0)
        self.assertEqual(audit["mismatched_tracked_frames"], 0)
        self.assertEqual(audit["fresh_timeline_mismatches"], 0)

    def test_empty_valid_stream_never_generates_predictions(self):
        data = self.dataset()
        data["raw_tracking_valid"][:] = False
        output = simulate_measurement_first(data, {})
        self.assertEqual(output["direct_pos_w"].shape, (0, 3))

    def test_calibration_rotates_predicted_position_and_velocity(self):
        data = self.dataset()
        data["raw_tracking_valid"][20:29] = False
        output = simulate_measurement_first(data, {})
        rotation = Rotation.from_euler("z", 0.6)
        translation = np.array([2.0, -1.0, 0.0])
        data["raw_pos_mocap_w"] = data["raw_pos_w"].copy()
        data["raw_pos_w"] = rotation.apply(data["raw_pos_w"]) + translation
        data["calibration_segment_starts_s"] = [-np.inf]
        data["calibration_segment_yaws_rad"] = [0.6]
        data["calibration_segment_translations_w"] = [translation]
        transformed = simulate_measurement_first(data, {})
        np.testing.assert_allclose(transformed["direct_pos_w"], rotation.apply(output["direct_pos_w"]) + translation)
        np.testing.assert_allclose(transformed["direct_velocity_w"], rotation.apply(output["direct_velocity_w"]), atol=1e-12)

    def test_jittered_final_arrival_not_dropped_or_extended(self):
        data = self.dataset()
        data["raw_t_s"][-1] += 0.003
        output = simulate_measurement_first(data, {})
        measured = output["direct_raw_index"] >= 0
        np.testing.assert_array_equal(output["direct_t_s"][measured], data["raw_t_s"])
        np.testing.assert_allclose(output["direct_t_s"][~measured], [.5])
        self.assertEqual(output["direct_t_s"][-1], data["raw_t_s"][-1])

    def test_long_loss_stops_prediction_and_new_track_resets_epoch(self):
        data = self.dataset(361)
        data["raw_tracking_valid"][20:-1] = False
        output = simulate_measurement_first(data, {})
        held = output["direct_mode"] == "absent_hold"
        self.assertTrue(held.any())
        np.testing.assert_array_equal(output["direct_velocity_w"][held], 0)
        np.testing.assert_allclose(np.diff(output["direct_pos_w"][held], axis=0), 0)
        self.assertEqual(output["direct_epoch"][-1], 2)

    def test_config_uses_existing_velocity_settings_but_no_position_blend(self):
        cfg = _direct_estimator_config({"mocap_bridge": {
            "velocity_window_size": 9, "ball_velocity_correction_gain": 0.3,
            "ball_position_correction_gain": 0.8, "ball_reassociation_timeout_s": 0.6,
        }})
        self.assertEqual(cfg.velocity_window_size, 9)
        self.assertEqual(cfg.velocity_correction_gain, 0.3)
        self.assertEqual(cfg.position_correction_gain, 1.0)
        self.assertEqual(cfg.reassociation_timeout_s, 0.6)


class ComparisonRenderingTests(unittest.TestCase):
    def setUp(self):
        self.data = MeasurementFirstTests.dataset()
        self.data["filtered_t_s"] = self.data["raw_t_s"].copy()
        self.data["filtered_pos_w"] = self.data["raw_pos_w"].copy()
        self.data.update(simulate_measurement_first(self.data, {}))
        self.model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
            <body name="tennis_ball"><freejoint name="tennis_ball_freejoint"/>
            <geom type="sphere" size="0.0335" mass="0.0577"/></body>
            </worldbody></mujoco>''')
        self.replay = TrajectoryReplay(
            self.model, mujoco.MjData(self.model), self.data,
            ball_radius_m=.0335, speed=1, trail_seconds=1, show_filtered=True,
        )

    def test_current_position_never_uses_future_sample(self):
        self.assertIsNone(self.replay.sample_direct(-.001))
        np.testing.assert_array_equal(self.replay.sample_direct(.004), self.data["raw_pos_w"][0])
        self.assertIsNone(self.replay.sample_direct(1.0))

    def test_independent_stream_visibility_and_scatter_geoms(self):
        scene = mujoco.MjvScene(self.model, maxgeom=2000)
        for ball, trail, color in (
            ("show_raw", "show_raw_trail", RAW_COLOR),
            ("show_filtered", "show_bridge_trail", BRIDGE_COLOR),
            ("show_direct", "show_direct_trail", DIRECT_COLOR),
        ):
            self.replay.draw(scene, .4)
            matching = [scene.geoms[i] for i in range(scene.ngeom) if np.allclose(scene.geoms[i].rgba, color)]
            self.assertTrue(all(g.type == mujoco.mjtGeom.mjGEOM_SPHERE for g in matching))
            self.assertTrue(any(np.isclose(g.size[0], TRAIL_POINT_RADIUS_M) for g in matching))
            self.assertTrue(any(np.isclose(g.size[0], self.replay.ball_radius_m) for g in matching))
            setattr(self.replay, ball, False)
            setattr(self.replay, trail, False)
            self.replay.draw(scene, .4)
            self.assertFalse(any(np.allclose(scene.geoms[i].rgba, color) for i in range(scene.ngeom)))


if __name__ == "__main__":
    unittest.main()
