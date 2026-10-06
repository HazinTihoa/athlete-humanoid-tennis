"""Exercise actual Bridge callbacks in an isolated ROS domain, never hardware."""

import copy
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

DEPLOY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_DIR))
try:
    import rclpy
    from rclpy.time import Time
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, Vector3Stamped
    from mocap_msgs.msg import RigidBodies, RigidBody
    from std_msgs.msg import String
    from estimation.mocap import natnet_policy_bridge as bridge_module
except ImportError as exc:
    raise unittest.SkipTest("Source deploy/setup_deploy.zsh to run ROS callback tests") from exc


class Recorder:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(copy.deepcopy(message))


class BridgeCallbacksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Even accidental publisher use in a test cannot reach deployment DDS.
        rclpy.init(domain_id=231)

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.now = 10.0
        config = {"mocap_bridge": {
            "visualization_enabled": False,
            "root_calibration_enabled": False,
            "pelvis_timeout_s": .04,
            "pelvis_recovery_s": .02,
            "ball_max_mean_error_m": .01,
            "ball_position_correction_gain": .8,
        }}
        with patch.object(bridge_module, "load_config", return_value=(config, Path("test.yaml"))):
            self.node = bridge_module.NatNetPolicyBridge("test.yaml")
        self.monotonic_patch = patch.object(bridge_module.time, "monotonic", side_effect=lambda: self.now)
        self.monotonic_patch.start()
        self.clock_patch = patch.object(self.node, "get_clock")
        self.clock = self.clock_patch.start().return_value
        self.clock.now.side_effect = lambda: Time(nanoseconds=round(self.now * 1e9))
        self.node._start_monotonic = self.now
        for name in ("pelvis_pose_pub", "ball_pose_pub", "ball_velocity_pub", "ball_epoch_pub", "fsm_request_pub", "status_pub", "marker_pub"):
            setattr(self.node, name, Recorder())

    def tearDown(self):
        self.clock_patch.stop()
        self.monotonic_patch.stop()
        self.node.destroy_node()

    @staticmethod
    def body(name, position, *, valid=True, error=.0, quat=(0., 0., 0., 1.)):
        body = RigidBody()
        body.rigid_body_name = name
        body.tracking_valid = bool(valid)
        body.mean_error = float(error)
        body.pose.position.x, body.pose.position.y, body.pose.position.z = map(float, position)
        (body.pose.orientation.x, body.pose.orientation.y,
         body.pose.orientation.z, body.pose.orientation.w) = map(float, quat)
        return body

    def message(self, source_time=None, *, ball=True, pelvis=True, position=(1., 2., .5), valid=True, error=.03):
        msg = RigidBodies()
        msg.header.stamp = Time(nanoseconds=round((self.now if source_time is None else source_time) * 1e9)).to_msg()
        if pelvis:
            msg.rigidbodies.append(self.body("G1_pelvis", (0., 0., .8)))
        if ball:
            # Deliberately no ball orientation and a rigid-fit error above 10mm.
            msg.rigidbodies.append(self.body("Ball", position, valid=valid, error=error, quat=(0., 0., 0., 0.)))
        return msg

    @staticmethod
    def position(msg):
        p = msg.pose.position
        return np.array([p.x, p.y, p.z])

    def test_measured_position_is_immediate_even_with_large_fit_error(self):
        msg = self.message()
        self.node._mocap_callback(msg)
        self.assertEqual(len(self.node.ball_pose_pub.messages), 1)
        pose = self.node.ball_pose_pub.messages[-1]
        velocity = self.node.ball_velocity_pub.messages[-1]
        np.testing.assert_array_equal(self.position(pose), [1., 2., .5])
        self.assertEqual(pose.header.stamp, msg.header.stamp)
        self.assertEqual(velocity.header.stamp, msg.header.stamp)
        self.assertEqual(self.node.ball_stream.last_sample.source, "measured")

    def test_optional_50hz_packet_cadence_with_120hz_source(self):
        self.node._ball_measurement_min_period_s = .02
        for frame in range(120):
            self.now = 10. + frame / 120.
            self.node._mocap_callback(self.message(position=(1. + frame * .002, 0., 1.)))
        self.assertEqual(self.node._ball_measured_count, 50)

    def test_only_empty_120hz_slots_publish_fills_and_50hz_does_not(self):
        self.node._mocap_callback(self.message())
        self.now += .001
        self.node._ball_prediction_callback()
        self.node._output_callback()
        self.assertEqual(len(self.node.ball_pose_pub.messages), 1)
        self.node.ball_estimator.is_static = False
        self.node.ball_estimator.velocity_w[:] = [1., 0., 1.]
        self.now += 1 / 120
        self.node._ball_prediction_callback()
        self.assertEqual(len(self.node.ball_pose_pub.messages), 2)
        self.assertEqual(self.node.ball_stream.last_sample.source, "predicted")
        self.assertEqual(self.node.ball_pose_pub.messages[-1].header.stamp, self.node.ball_velocity_pub.messages[-1].header.stamp)
        self.assertEqual(self.node._ball_fill_count, 1)
        self.now += .001
        self.node._mocap_callback(self.message(position=(2., 3., .4)))
        np.testing.assert_array_equal(self.position(self.node.ball_pose_pub.messages[-1]), [2., 3., .4])
        self.now += .007
        self.node._ball_prediction_callback()
        self.assertEqual(len(self.node.ball_pose_pub.messages), 3)

    def test_no_ball_is_published_before_first_valid_measurement(self):
        self.node._ball_prediction_callback()
        self.node._mocap_callback(self.message(valid=False))
        self.now += .02
        self.node._ball_prediction_callback()
        self.assertEqual(len(self.node.ball_pose_pub.messages), 0)

    def test_invalid_position_and_old_stamp_do_not_refresh_valid_arrival(self):
        self.node._mocap_callback(self.message())
        last = self.node._ball_last_valid_monotonic
        self.now += .005
        self.node._mocap_callback(self.message(source_time=10.))
        self.now += .005
        self.node._mocap_callback(self.message(position=(np.nan, 0., 1.)))
        self.now += .005
        self.node._mocap_callback(self.message(valid=False))
        self.assertEqual(len(self.node.ball_pose_pub.messages), 1)
        self.assertEqual(self.node._ball_last_valid_monotonic, last)

    def test_pelvis_loss_forces_damp_even_while_ball_stream_is_healthy(self):
        self.node._mocap_callback(self.message())
        self.node._output_callback()
        self.assertEqual(len(self.node.pelvis_pose_pub.messages), 1)
        self.now += .05
        self.node._mocap_callback(self.message(pelvis=False))
        self.node._output_callback()
        self.assertTrue(self.node._pelvis_fault_active)
        self.assertEqual(self.node.fsm_request_pub.messages[-1].data, "damp")
        self.assertEqual(len(self.node.pelvis_pose_pub.messages), 1)
        self.now += .01
        self.node._mocap_callback(self.message())
        self.node._output_callback()
        self.assertTrue(self.node._pelvis_fault_active)
        self.now += .025
        self.node._mocap_callback(self.message())
        self.node._output_callback()
        self.assertFalse(self.node._pelvis_fault_active)
        self.assertFalse(any(m.data == "control" for m in self.node.fsm_request_pub.messages))

    def test_calibration_rotates_both_measured_and_predicted_ball_state(self):
        self.node.root_calibration_enabled = True
        self.node._root_calibrated = False
        self.node._fsm_state = "home"
        msg = self.message()
        msg.rigidbodies[0] = self.body("G1_pelvis", (2., 3., .8), quat=(0., 0., np.sin(.3), np.cos(.3)))
        self.node._mocap_callback(msg)
        request = String(); request.data = "calibrate"
        self.node._root_calibration_request_callback(request)
        self.assertTrue(self.node._root_calibrated)
        self.now += .008
        self.node._mocap_callback(self.message(pelvis=False, position=(3., 4., 1.)))
        expected = self.node._transform_position_w(np.array([3., 4., 1.]))
        np.testing.assert_allclose(self.position(self.node.ball_pose_pub.messages[-1]), expected)
        self.node._ball_prediction_callback()
        self.now += 1 / 120
        self.node._ball_prediction_callback()
        sample = self.node.ball_stream.last_sample
        np.testing.assert_allclose(self.position(self.node.ball_pose_pub.messages[-1]), self.node._transform_position_w(sample.position_w))
        v = self.node.ball_velocity_pub.messages[-1].vector
        np.testing.assert_allclose([v.x, v.y, v.z], self.node._transform_velocity_w(sample.velocity_w))

    def test_watchdog_and_prediction_timers_are_independent(self):
        periods = sorted(t.timer_period_ns for t in self.node.timers)
        self.assertEqual(periods, [8333333, 20000000])

    def test_isolated_ros_topics_deliver_measurements_and_dropout_fills(self):
        self.clock_patch.stop()
        self.monotonic_patch.stop()
        self.node._start_monotonic = time.monotonic()
        for attribute, topic in (("ball_pose_pub", "/ball/pose"), ("ball_velocity_pub", "/ball/velocity")):
            setattr(self.node, attribute, next(p for p in self.node.publishers if p.topic_name == topic))
        probe = Node("measurement_first_wire_probe")
        executor = SingleThreadedExecutor()
        executor.add_node(self.node)
        executor.add_node(probe)
        positions = []
        velocities = []
        probe.create_subscription(PoseStamped, "/ball/pose", positions.append, 100)
        probe.create_subscription(Vector3Stamped, "/ball/velocity", velocities.append, 100)
        publisher = probe.create_publisher(RigidBodies, "/rigid_bodies", 10)
        sent = {}

        def stamp_key(stamp):
            return stamp.sec * 1_000_000_000 + stamp.nanosec

        try:
            deadline = time.monotonic() + 2.0
            while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=.01)
            self.assertGreater(publisher.get_subscription_count(), 0)
            start = time.monotonic()

            def send():
                elapsed = time.monotonic() - start
                self.now = probe.get_clock().now().nanoseconds * 1e-9
                position = np.array([elapsed, .5, 1.])
                valid = not .4 < elapsed < .65
                message = self.message(position=position, valid=valid)
                if valid:
                    sent[stamp_key(message.header.stamp)] = position
                publisher.publish(message)

            timer = probe.create_timer(1 / 120, send)
            while time.monotonic() - start < 1.1:
                executor.spin_once(timeout_sec=.005)
            timer.cancel()
            deadline = time.monotonic() + .05
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=.005)
            measured = 0
            fills = 0
            for message in positions:
                key = stamp_key(message.header.stamp)
                if key in sent:
                    measured += 1
                    np.testing.assert_array_equal(self.position(message), sent[key])
                else:
                    fills += 1
            self.assertGreater(measured, 30)
            self.assertGreater(fills, 5)
            position_stamps = {stamp_key(p.header.stamp) for p in positions}
            velocity_stamps = {stamp_key(v.header.stamp) for v in velocities}
            self.assertGreater(len(position_stamps & velocity_stamps), 30)
            print(f"\nIsolated ROS wire: {measured} exact measurements, {fills} prediction/hold samples")
        finally:
            executor.remove_node(probe)
            executor.remove_node(self.node)
            executor.shutdown()
            probe.destroy_node()

    def test_all_recorded_valid_positions_pass_through_actual_ros_callbacks(self):
        path = DEPLOY_DIR / "recordings/2026-09-09_20-20-17_ball_bridge_only/ball_trajectory.npz"
        if not path.exists():
            self.skipTest("Regression bag is not available")
        with np.load(path) as archive:
            data = dict(archive)
        times = data["raw_t_s"]
        index = 0
        checked = 0
        for tick in np.arange(0., times[-1] + 1 / 120, 1 / 120):
            while index < len(times) and times[index] <= tick:
                self.now = 100. + times[index]
                position = data["raw_pos_mocap_w"][index]
                valid = bool(data["raw_tracking_valid"][index])
                before = len(self.node.ball_pose_pub.messages)
                self.node._mocap_callback(self.message(
                    source_time=100. + data["raw_source_t_s"][index], pelvis=False,
                    position=position, valid=valid, error=data["raw_mean_error"][index],
                ))
                if valid and np.isfinite(position).all():
                    self.assertEqual(len(self.node.ball_pose_pub.messages), before + 1)
                    np.testing.assert_array_equal(self.position(self.node.ball_pose_pub.messages[-1]), position)
                    checked += 1
                else:
                    self.assertEqual(len(self.node.ball_pose_pub.messages), before)
                index += 1
            self.now = 100. + tick
            self.node._ball_prediction_callback()
        self.assertEqual(checked, 6586)
        self.assertGreater(self.node._ball_fill_count, 0)
        self.assertEqual(len(self.node.ball_pose_pub.messages), len(self.node.ball_velocity_pub.messages))
        for p, v in zip(self.node.ball_pose_pub.messages, self.node.ball_velocity_pub.messages):
            self.assertEqual(p.header.stamp, v.header.stamp)


if __name__ == "__main__":
    unittest.main()
