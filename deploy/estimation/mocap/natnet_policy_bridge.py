"""Forward NatNet ball measurements immediately and fill gaps at 120 Hz."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, Vector3Stamped
from mocap_msgs.msg import RigidBodies
from rclpy.node import Node
from std_msgs.msg import String, UInt64
from visualization_msgs.msg import Marker, MarkerArray
import yaml


DEPLOY_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = DEPLOY_DIR.parent
sys.path.insert(0, str(DEPLOY_DIR))

from utils.natnet_bridge_state import (
    BallEstimatorConfig, BallOutputSample, MeasurementFirstBallStream,
)
from utils.mocap_frame_calibration import (
    planar_calibration_from_root_pose,
    quaternion_xyzw_multiply,
    quaternion_xyzw_to_matrix,
)


def load_config(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        config_candidate = DEPLOY_DIR / "configs" / path
        path = config_candidate if config_candidate.exists() else REPO_ROOT / path
    path = path.resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Deployment config must be a mapping: {path}")
    return config, path


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def finite_pose(body) -> tuple[np.ndarray, np.ndarray] | None:
    position = np.array(
        [body.pose.position.x, body.pose.position.y, body.pose.position.z],
        dtype=np.float64,
    )
    quaternion_xyzw = np.array(
        [
            body.pose.orientation.x,
            body.pose.orientation.y,
            body.pose.orientation.z,
            body.pose.orientation.w,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(position).all() or not np.isfinite(quaternion_xyzw).all():
        return None
    norm = float(np.linalg.norm(quaternion_xyzw))
    if norm < 1.0e-8:
        return None
    return position, quaternion_xyzw / norm


def point_message(position: np.ndarray) -> Point:
    point = Point()
    point.x = float(position[0])
    point.y = float(position[1])
    point.z = float(position[2])
    return point


class NatNetPolicyBridge(Node):
    """Validate mocap state, estimate ball velocity, and guard pelvis loss."""

    def __init__(self, config_path: str | Path) -> None:
        super().__init__("natnet_policy_bridge")
        config, resolved_path = load_config(config_path)
        bridge = config.get("mocap_bridge", {})
        if not isinstance(bridge, dict):
            raise TypeError("mocap_bridge must be a mapping")
        rate = float(bridge.get('measurement_rate_hz', 0.0))
        if not math.isfinite(rate) or rate < 0:
            raise ValueError('measurement_rate_hz must be finite and non-negative')
        self._ball_measurement_min_period_s = 0.0 if rate == 0 else 1.0 / rate
        self._ball_measurement_next_s = None
        ball_physics = config.get("tennis_ball_physics", {})
        if not isinstance(ball_physics, dict):
            raise TypeError("tennis_ball_physics must be a mapping")

        self.input_topic = str(bridge.get("input_topic", "/rigid_bodies"))
        self.world_frame = str(bridge.get("world_frame", "world"))
        self.pelvis_name = str(bridge.get("pelvis_name", "G1_pelvis"))
        self.ball_name = str(bridge.get("ball_name", "Ball"))
        self.output_rate_hz = float(bridge.get("output_rate_hz", 50.0))
        self.ball_prediction_rate_hz = float(bridge.get("ball_prediction_rate_hz", 120.0))
        self.pelvis_startup_grace_s = float(
            bridge.get("pelvis_startup_grace_s", 10.0)
        )
        self.pelvis_timeout_s = float(bridge.get("pelvis_timeout_s", 0.040))
        self.pelvis_recovery_s = float(bridge.get("pelvis_recovery_s", 0.20))
        self.pelvis_max_mean_error_m = float(
            bridge.get("pelvis_max_mean_error_m", 0.005)
        )
        self.damp_republish_s = float(bridge.get("damp_republish_s", 0.10))
        self.status_publish_s = float(bridge.get("status_publish_s", 0.20))
        self.visualization_enabled = bool(bridge.get("visualization_enabled", True))
        self.visualization_topic = str(
            bridge.get("visualization_topic", "/mocap_policy_bridge/markers")
        )
        self.trajectory_duration_s = float(
            bridge.get("trajectory_duration_s", 5.0)
        )
        self.pelvis_axis_length_m = float(
            bridge.get("pelvis_axis_length_m", 0.30)
        )
        self.velocity_arrow_scale_s = float(
            bridge.get("velocity_arrow_scale_s", 0.10)
        )
        self.root_calibration_enabled = bool(
            bridge.get("root_calibration_enabled", False)
        )
        self.root_calibration_target_xy_w = np.asarray(
            bridge.get("root_calibration_target_xy_w", [0.0, 0.0]),
            dtype=np.float64,
        )
        self.root_calibration_target_yaw_rad = math.radians(
            float(bridge.get("root_calibration_target_yaw_deg", 0.0))
        )
        if not math.isfinite(self.output_rate_hz) or self.output_rate_hz <= 0.0:
            raise ValueError("mocap_bridge.output_rate_hz must be finite and positive")
        if not math.isfinite(self.ball_prediction_rate_hz) or self.ball_prediction_rate_hz <= 0.0:
            raise ValueError("mocap_bridge.ball_prediction_rate_hz must be finite and positive")
        if (
            self.pelvis_startup_grace_s < 0.0
            or self.pelvis_timeout_s <= 0.0
            or self.pelvis_recovery_s < 0.0
        ):
            raise ValueError("Invalid pelvis watchdog timing")
        if (
            self.root_calibration_target_xy_w.shape != (2,)
            or not np.isfinite(self.root_calibration_target_xy_w).all()
            or not math.isfinite(self.root_calibration_target_yaw_rad)
        ):
            raise ValueError("Invalid Mocap root calibration target")

        estimator_config = BallEstimatorConfig(
            radius_m=float(ball_physics.get("radius_m", 0.0335)),
            mass_kg=float(ball_physics.get("mass_kg", 0.0577)),
            air_density_kg_m3=float(
                ball_physics.get("air_density_kg_m3", 1.225)
            ),
            drag_coefficient=float(ball_physics.get("drag_coefficient", 0.55)),
            court_restitution=float(bridge.get("court_restitution", 0.745)),
            tangent_speed_retention=float(
                bridge.get("ground_tangent_speed_retention", 0.825)
            ),
            ground_height_m=float(bridge.get("ground_height_m", 0.0)),
            velocity_window_size=int(bridge.get("velocity_window_size", 7)),
            velocity_min_samples=int(bridge.get("velocity_min_samples", 5)),
            short_dropout_s=float(bridge.get("ball_short_dropout_s", 0.10)),
            maximum_prediction_s=float(
                bridge.get("ball_maximum_prediction_s", 2.0)
            ),
            reassociation_timeout_s=float(
                bridge.get("ball_reassociation_timeout_s", 0.50)
            ),
            static_speed_threshold_mps=float(
                bridge.get("ball_static_speed_threshold_mps", 0.15)
            ),
            static_hold_s=float(bridge.get("ball_static_hold_s", 0.15)),
            maximum_speed_mps=float(bridge.get("ball_maximum_speed_mps", 40.0)),
            position_correction_gain=1.0,
            velocity_correction_gain=float(
                bridge.get("ball_velocity_correction_gain", 0.55)
            ),
            integration_dt_s=float(bridge.get("ball_integration_dt_s", 0.0025)),
        )
        self.ball_stream = MeasurementFirstBallStream(estimator_config)
        self.ball_estimator = self.ball_stream.estimator

        self.pelvis_pose_pub = self.create_publisher(
            PoseStamped, "/g1_pelvis/pose", 10
        )
        self.ball_pose_pub = self.create_publisher(PoseStamped, "/ball/pose", 1)
        self.ball_velocity_pub = self.create_publisher(
            Vector3Stamped, "/ball/velocity", 1
        )
        self.ball_epoch_pub = self.create_publisher(UInt64, "/ball/track_epoch", 1)
        self.fsm_request_pub = self.create_publisher(
            String, "/deploy_robot/fsm_request", 10
        )
        self.status_pub = self.create_publisher(
            String, "/mocap_policy_bridge/status", 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, self.visualization_topic, 10
        )
        self.create_subscription(RigidBodies, self.input_topic, self._mocap_callback, 1)
        self.create_subscription(
            String,
            "/deploy_robot/fsm_state",
            self._fsm_state_callback,
            10,
        )
        self.create_subscription(
            String,
            "/deploy_robot/root_calibration_request",
            self._root_calibration_request_callback,
            10,
        )

        self._start_monotonic = time.monotonic()
        self._pelvis_last_valid_monotonic: float | None = None
        self._pelvis_last_source_s: float | None = None
        self._pelvis_position_w = np.zeros(3, dtype=np.float64)
        self._pelvis_quaternion_xyzw = np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float64
        )
        self._pelvis_raw_valid = False
        self._fsm_state = "damp"
        self._root_calibrated = not self.root_calibration_enabled
        self._root_calibration_rotation = np.eye(3, dtype=np.float64)
        self._root_calibration_translation = np.zeros(3, dtype=np.float64)
        self._root_calibration_quaternion_xyzw = np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float64
        )
        self._pelvis_fault_active = False
        self._pelvis_recovery_since: float | None = None
        self._last_damp_publish = -math.inf
        self._last_status_publish = -math.inf
        self._last_ball_mode = "uninitialized"
        self._ball_last_valid_monotonic: float | None = None
        self._ball_measured_count = 0
        self._ball_fill_count = 0
        self._visualized_ball_epoch = 0
        self._ball_trajectory_w: list[np.ndarray] = []
        self._trajectory_max_samples = max(
            2, int(math.ceil(self.trajectory_duration_s * self.output_rate_hz))
        )

        self.create_timer(1.0 / self.output_rate_hz, self._output_callback)
        self.create_timer(1.0 / self.ball_prediction_rate_hz, self._ball_prediction_callback)
        self.get_logger().info(
            "NatNet bridge ready: "
            f"input={self.input_topic} ball=measurement-first "
            f"fill_timer={self.ball_prediction_rate_hz:.1f}Hz "
            f"pelvis_visualization={self.output_rate_hz:.1f}Hz "
            f"startup_grace={self.pelvis_startup_grace_s:.2f}s "
            f"pelvis_timeout={self.pelvis_timeout_s * 1000.0:.1f}ms "
            f"root_calibration={'required' if self.root_calibration_enabled else 'disabled'} "
            f"config={resolved_path}"
        )
        if "ball_max_mean_error_m" in bridge or "ball_position_correction_gain" in bridge:
            self.get_logger().info(
                "Measurement-first ignores legacy ball_max_mean_error_m and "
                "ball_position_correction_gain; tracked finite ball positions pass through"
            )

    def _body_is_valid(self, body, max_mean_error_m: float) -> bool:
        return (
            bool(body.tracking_valid)
            and math.isfinite(float(body.mean_error))
            and float(body.mean_error) <= max_mean_error_m
            and finite_pose(body) is not None
        )

    def _transform_position_w(self, position_w: np.ndarray) -> np.ndarray:
        return self._root_calibration_rotation @ position_w + self._root_calibration_translation

    def _transform_velocity_w(self, velocity_w: np.ndarray) -> np.ndarray:
        return self._root_calibration_rotation @ velocity_w

    def _transform_quaternion_xyzw(self, quaternion_xyzw: np.ndarray) -> np.ndarray:
        return quaternion_xyzw_multiply(
            self._root_calibration_quaternion_xyzw, quaternion_xyzw
        )

    def _fsm_state_callback(self, message: String) -> None:
        previous = self._fsm_state
        self._fsm_state = message.data
        if (
            self.root_calibration_enabled
            and not self._root_calibrated
            and self._fsm_state == "control"
            and previous != "control"
        ):
            self.get_logger().error(
                "Root is not calibrated; press Y while FSM is in home"
            )
            self._request_damp(time.monotonic())

    def _root_calibration_request_callback(self, message: String) -> None:
        if message.data != "calibrate":
            self.get_logger().warn(f"Unknown root calibration request: {message.data!r}")
            return
        if not self.root_calibration_enabled:
            self.get_logger().warn("Root calibration is disabled in this config")
            return
        if self._fsm_state != "home":
            self.get_logger().warn("Root calibration rejected: FSM must be in home")
            return
        if not self._pelvis_raw_valid or self._pelvis_last_valid_monotonic is None:
            self.get_logger().error("Root calibration rejected: Pelvis tracking is invalid")
            return
        pelvis_age_s = time.monotonic() - self._pelvis_last_valid_monotonic
        if pelvis_age_s > self.pelvis_timeout_s:
            self.get_logger().error(
                f"Root calibration rejected: Pelvis data is {pelvis_age_s * 1000.0:.1f}ms old"
            )
            return

        raw_position = self._pelvis_position_w.copy()
        raw_quaternion = self._pelvis_quaternion_xyzw.copy()
        (
            self._root_calibration_rotation,
            self._root_calibration_translation,
            self._root_calibration_quaternion_xyzw,
        ) = planar_calibration_from_root_pose(
            raw_position,
            raw_quaternion,
            self.root_calibration_target_xy_w,
            self.root_calibration_target_yaw_rad,
        )
        self._root_calibrated = True
        self._ball_trajectory_w.clear()

        calibrated_position = self._transform_position_w(raw_position)
        calibrated_quaternion = self._transform_quaternion_xyzw(raw_quaternion)
        raw_yaw = math.degrees(
            math.atan2(
                quaternion_xyzw_to_matrix(raw_quaternion)[1, 0],
                quaternion_xyzw_to_matrix(raw_quaternion)[0, 0],
            )
        )
        calibrated_yaw = math.degrees(
            math.atan2(
                quaternion_xyzw_to_matrix(calibrated_quaternion)[1, 0],
                quaternion_xyzw_to_matrix(calibrated_quaternion)[0, 0],
            )
        )
        self.get_logger().info(
            "Mocap root calibrated: "
            f"raw_pos=[{raw_position[0]:+.3f}, {raw_position[1]:+.3f}, {raw_position[2]:+.3f}]m "
            f"raw_yaw={raw_yaw:+.2f}deg -> "
            f"policy_pos=[{calibrated_position[0]:+.3f}, "
            f"{calibrated_position[1]:+.3f}, {calibrated_position[2]:+.3f}]m "
            f"policy_yaw={calibrated_yaw:+.2f}deg"
        )

    def _mocap_callback(self, message: RigidBodies) -> None:
        arrival = time.monotonic()
        source_time = stamp_to_seconds(message.header.stamp)
        pelvis = None
        ball = None
        for body in message.rigidbodies:
            if body.rigid_body_name == self.pelvis_name:
                pelvis = body
            elif body.rigid_body_name == self.ball_name:
                ball = body

        source_valid = math.isfinite(source_time)
        pelvis_fresh = source_valid and (
            self._pelvis_last_source_s is None or source_time > self._pelvis_last_source_s
        )
        pelvis_valid = pelvis_fresh and pelvis is not None and self._body_is_valid(
            pelvis, self.pelvis_max_mean_error_m
        )
        if pelvis_valid:
            pose = finite_pose(pelvis)
            assert pose is not None
            first_valid_pelvis = self._pelvis_last_valid_monotonic is None
            self._pelvis_position_w[:], self._pelvis_quaternion_xyzw[:] = pose
            if not self._pelvis_raw_valid:
                self._pelvis_recovery_since = arrival
            self._pelvis_raw_valid = True
            self._pelvis_last_valid_monotonic = arrival
            self._pelvis_last_source_s = source_time
            if first_valid_pelvis:
                startup_delay_s = arrival - self._start_monotonic
                self.get_logger().info(
                    f"First valid Pelvis frame received after "
                    f"{startup_delay_s * 1000.0:.1f}ms; watchdog armed"
                )
        else:
            self._pelvis_raw_valid = False
            self._pelvis_recovery_since = None

        # Optional packet-rate alignment for the new measurement-training tasks.
        # Existing deployments retain every accepted NatNet frame by default.
        period = getattr(self, '_ball_measurement_min_period_s', 0.0)
        next_time = getattr(self, '_ball_measurement_next_s', None)
        if period and next_time is not None and source_time < next_time - 1e-6:
            return
        # Ball position does not require a valid rigid-body orientation or a
        # low rigid-fit error. In particular, deformation at a bounce is not loss.
        if ball is not None and bool(ball.tracking_valid) and source_valid:
            point = ball.pose.position
            position = np.array([point.x, point.y, point.z], dtype=np.float64)
            previous_epoch = self.ball_estimator.track_epoch
            sample = self.ball_stream.observe(position, source_time)
            if sample is None:
                return
            if period:
                base = source_time if next_time is None else next_time
                # Keep cadence phase: 120Hz input should average 50Hz, not
                # accept every third frame and silently become 40Hz.
                steps = max(1, math.floor((source_time - base) / period + 1e-6) + 1)
                self._ball_measurement_next_s = base + steps * period
            self._ball_last_valid_monotonic = arrival
            self._ball_measured_count += 1
            self._publish_ball(message.header.stamp, sample)
            if sample.track_epoch != previous_epoch:
                self.get_logger().info(
                    f"Ball track epoch {self.ball_estimator.track_epoch} started"
                )

    def _ball_prediction_callback(self) -> None:
        now = self.get_clock().now()
        sample = self.ball_stream.prediction_tick(now.nanoseconds * 1.0e-9)
        if sample is not None:
            self._ball_fill_count += 1
            self._publish_ball(now.to_msg(), sample)

    def _request_damp(self, now_monotonic: float) -> None:
        if now_monotonic - self._last_damp_publish < self.damp_republish_s:
            return
        message = String()
        message.data = "damp"
        self.fsm_request_pub.publish(message)
        self._last_damp_publish = now_monotonic

    def _pelvis_age_s(self, now_monotonic: float) -> float:
        reference = (
            self._start_monotonic
            if self._pelvis_last_valid_monotonic is None
            else self._pelvis_last_valid_monotonic
        )
        return max(0.0, now_monotonic - reference)

    def _update_pelvis_watchdog(self, now_monotonic: float) -> bool:
        if self._pelvis_last_valid_monotonic is None:
            startup_age_s = now_monotonic - self._start_monotonic
            if startup_age_s <= self.pelvis_startup_grace_s:
                return False
            if not self._pelvis_fault_active:
                self.get_logger().error(
                    "No valid Pelvis frame received within "
                    f"{self.pelvis_startup_grace_s:.2f}s; requesting damp"
                )
            self._pelvis_fault_active = True
            self._request_damp(now_monotonic)
            return False

        age_s = self._pelvis_age_s(now_monotonic)
        timed_out = age_s > self.pelvis_timeout_s
        if timed_out:
            if not self._pelvis_fault_active:
                self.get_logger().error(
                    f"Pelvis tracking lost for {age_s * 1000.0:.1f}ms; requesting damp"
                )
            self._pelvis_fault_active = True
            self._pelvis_recovery_since = None
            self._request_damp(now_monotonic)
            return False

        if self._pelvis_fault_active:
            if not self._pelvis_raw_valid:
                self._pelvis_recovery_since = None
                self._request_damp(now_monotonic)
                return False
            if self._pelvis_recovery_since is None:
                self._pelvis_recovery_since = now_monotonic
            if now_monotonic - self._pelvis_recovery_since < self.pelvis_recovery_s:
                self._request_damp(now_monotonic)
                return False
            self._pelvis_fault_active = False
            self.get_logger().info(
                "Pelvis tracking recovered; FSM remains damp until operator control"
            )
        return True

    def _publish_pelvis(self, stamp) -> None:
        position_w = self._transform_position_w(self._pelvis_position_w)
        quaternion_xyzw = self._transform_quaternion_xyzw(
            self._pelvis_quaternion_xyzw
        )
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.world_frame
        message.pose.position.x = float(position_w[0])
        message.pose.position.y = float(position_w[1])
        message.pose.position.z = float(position_w[2])
        message.pose.orientation.x = float(quaternion_xyzw[0])
        message.pose.orientation.y = float(quaternion_xyzw[1])
        message.pose.orientation.z = float(quaternion_xyzw[2])
        message.pose.orientation.w = float(quaternion_xyzw[3])
        self.pelvis_pose_pub.publish(message)

    def _publish_ball(
        self, stamp, sample: BallOutputSample
    ) -> tuple[np.ndarray, np.ndarray]:
        position = self._transform_position_w(sample.position_w)
        velocity = self._transform_velocity_w(sample.velocity_w)
        pose_message = PoseStamped()
        pose_message.header.stamp = stamp
        pose_message.header.frame_id = self.world_frame
        pose_message.pose.position.x = float(position[0])
        pose_message.pose.position.y = float(position[1])
        pose_message.pose.position.z = float(position[2])
        pose_message.pose.orientation.w = 1.0
        self.ball_pose_pub.publish(pose_message)

        velocity_message = Vector3Stamped()
        velocity_message.header.stamp = stamp
        velocity_message.header.frame_id = self.world_frame
        velocity_message.vector.x = float(velocity[0])
        velocity_message.vector.y = float(velocity[1])
        velocity_message.vector.z = float(velocity[2])
        self.ball_velocity_pub.publish(velocity_message)

        epoch_message = UInt64()
        epoch_message.data = sample.track_epoch
        self.ball_epoch_pub.publish(epoch_message)
        return position, velocity

    def _new_marker(self, stamp, marker_id: int, marker_type: int) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.world_frame
        marker.ns = "policy_input"
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _axis_marker(
        self,
        stamp,
        marker_id: int,
        origin: np.ndarray,
        endpoint: np.ndarray,
        color: tuple[float, float, float],
    ) -> Marker:
        marker = self._new_marker(stamp, marker_id, Marker.ARROW)
        marker.points = [point_message(origin), point_message(endpoint)]
        marker.scale.x = 0.015
        marker.scale.y = 0.035
        marker.scale.z = 0.050
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        return marker

    def _publish_visualization(
        self,
        stamp,
        ball_estimate: tuple[np.ndarray, np.ndarray] | None,
        timestamp_s: float,
        pelvis_ready: bool,
    ) -> None:
        markers = MarkerArray()
        pelvis_initialized = self._pelvis_last_valid_monotonic is not None
        pelvis_position_w = self._transform_position_w(self._pelvis_position_w)
        pelvis_quaternion_xyzw = self._transform_quaternion_xyzw(
            self._pelvis_quaternion_xyzw
        )
        rotation_wb = quaternion_xyzw_to_matrix(pelvis_quaternion_xyzw)

        if pelvis_initialized:
            axis_colors = (
                (1.0, 0.1, 0.1),
                (0.1, 1.0, 0.1),
                (0.1, 0.4, 1.0),
            )
            for axis, color in enumerate(axis_colors):
                endpoint = (
                    pelvis_position_w
                    + rotation_wb[:, axis] * self.pelvis_axis_length_m
                )
                markers.markers.append(
                    self._axis_marker(
                        stamp,
                        axis,
                        pelvis_position_w,
                        endpoint,
                        color,
                    )
                )

            pelvis_label = self._new_marker(stamp, 3, Marker.TEXT_VIEW_FACING)
            pelvis_label.pose.position = point_message(
                pelvis_position_w + np.array([0.0, 0.0, 0.15])
            )
            pelvis_label.scale.z = 0.08
            pelvis_label.color.r = 1.0
            pelvis_label.color.g = 1.0 if pelvis_ready else 0.15
            pelvis_label.color.b = 1.0 if pelvis_ready else 0.15
            pelvis_label.color.a = 1.0
            pelvis_label.text = (
                "policy pelvis frame"
                if pelvis_ready
                else (
                    "ROOT NOT CALIBRATED - press Y in home"
                    if self.root_calibration_enabled and not self._root_calibrated
                    else "PELVIS INVALID - DAMP REQUESTED"
                )
            )
            markers.markers.append(pelvis_label)
        else:
            for marker_id in range(4):
                delete_marker = self._new_marker(stamp, marker_id, Marker.ARROW)
                delete_marker.action = Marker.DELETE
                markers.markers.append(delete_marker)

        if ball_estimate is None:
            self._ball_trajectory_w.clear()
            for marker_id in range(10, 15):
                delete_marker = self._new_marker(stamp, marker_id, Marker.SPHERE)
                delete_marker.action = Marker.DELETE
                markers.markers.append(delete_marker)

            warning = self._new_marker(stamp, 15, Marker.TEXT_VIEW_FACING)
            warning_position = (
                pelvis_position_w + np.array([0.0, 0.0, 0.35])
                if pelvis_initialized
                else np.array([0.0, 0.0, 0.5])
            )
            warning.pose.position = point_message(warning_position)
            warning.scale.z = 0.09
            warning.color.r = 1.0
            warning.color.g = 0.1
            warning.color.b = 0.1
            warning.color.a = 1.0
            warning.text = "BALL NOT TRACKED - no policy ball input"
            markers.markers.append(warning)
            self.marker_pub.publish(markers)
            return

        delete_warning = self._new_marker(stamp, 15, Marker.TEXT_VIEW_FACING)
        delete_warning.action = Marker.DELETE
        markers.markers.append(delete_warning)

        ball_position_w, ball_velocity_w = ball_estimate
        epoch = self.ball_estimator.track_epoch
        if epoch != self._visualized_ball_epoch:
            self._ball_trajectory_w.clear()
            self._visualized_ball_epoch = epoch
        self._ball_trajectory_w.append(ball_position_w.copy())
        if len(self._ball_trajectory_w) > self._trajectory_max_samples:
            del self._ball_trajectory_w[: -self._trajectory_max_samples]

        ball = self._new_marker(stamp, 10, Marker.SPHERE)
        ball.pose.position = point_message(ball_position_w)
        diameter = 2.0 * self.ball_estimator.config.radius_m
        ball.scale.x = diameter
        ball.scale.y = diameter
        ball.scale.z = diameter
        ball.color.r = 1.0
        ball.color.g = 0.85
        ball.color.b = 0.05
        ball.color.a = 0.95
        markers.markers.append(ball)

        speed = float(np.linalg.norm(ball_velocity_w))
        if speed > 1.0e-4:
            velocity_endpoint = (
                ball_position_w + ball_velocity_w * self.velocity_arrow_scale_s
            )
            markers.markers.append(
                self._axis_marker(
                    stamp,
                    11,
                    ball_position_w,
                    velocity_endpoint,
                    (0.0, 1.0, 1.0),
                )
            )
        else:
            delete_velocity = self._new_marker(stamp, 11, Marker.ARROW)
            delete_velocity.action = Marker.DELETE
            markers.markers.append(delete_velocity)

        if pelvis_initialized:
            relative_line = self._new_marker(stamp, 12, Marker.LINE_LIST)
            relative_line.points = [
                point_message(pelvis_position_w),
                point_message(ball_position_w),
            ]
            relative_line.scale.x = 0.008
            relative_line.color.r = 0.85
            relative_line.color.g = 0.85
            relative_line.color.b = 0.85
            relative_line.color.a = 0.55
            markers.markers.append(relative_line)
        else:
            delete_relative_line = self._new_marker(stamp, 12, Marker.LINE_LIST)
            delete_relative_line.action = Marker.DELETE
            markers.markers.append(delete_relative_line)

        trajectory = self._new_marker(stamp, 13, Marker.LINE_STRIP)
        trajectory.points = [point_message(point) for point in self._ball_trajectory_w]
        trajectory.scale.x = 0.018
        trajectory.color.r = 1.0
        trajectory.color.g = 0.35
        trajectory.color.b = 0.05
        trajectory.color.a = 0.90
        markers.markers.append(trajectory)

        ball_label = self._new_marker(stamp, 14, Marker.TEXT_VIEW_FACING)
        ball_label.pose.position = point_message(
            ball_position_w + np.array([0.0, 0.0, 0.12])
        )
        ball_label.scale.z = 0.065
        ball_label.color.r = 1.0
        ball_label.color.g = 1.0
        ball_label.color.b = 1.0
        ball_label.color.a = 1.0
        mode = self.ball_estimator.mode(timestamp_s)
        if self.ball_stream.last_sample is not None:
            mode = f"{self.ball_stream.last_sample.source}/{mode}"
        if pelvis_ready:
            ball_position_b = rotation_wb.T @ (
                ball_position_w - pelvis_position_w
            )
            ball_velocity_b = rotation_wb.T @ ball_velocity_w
            ball_label.text = (
                f"epoch={epoch} {mode}  speed={speed:.2f} m/s\n"
                f"ball_b=[{ball_position_b[0]:+.3f}, {ball_position_b[1]:+.3f}, "
                f"{ball_position_b[2]:+.3f}] m\n"
                f"vel_b=[{ball_velocity_b[0]:+.2f}, {ball_velocity_b[1]:+.2f}, "
                f"{ball_velocity_b[2]:+.2f}] m/s"
            )
        else:
            ball_label.color.r = 1.0
            ball_label.color.g = 0.2
            ball_label.color.b = 0.2
            ball_label.text = (
                f"epoch={epoch} {mode}  speed={speed:.2f} m/s\n"
                "Pelvis invalid: local policy input unavailable"
            )
        markers.markers.append(ball_label)

        self.marker_pub.publish(markers)

    def _publish_status(self, now_monotonic: float, timestamp_s: float) -> None:
        if now_monotonic - self._last_status_publish < self.status_publish_s:
            return
        ball_age_ms = None
        if self._ball_last_valid_monotonic is not None:
            ball_age_ms = 1000.0 * max(
                0.0, now_monotonic - self._ball_last_valid_monotonic
            )
        payload = {
            "pelvis_age_ms": round(1000.0 * self._pelvis_age_s(now_monotonic), 2),
            "pelvis_initialized": self._pelvis_last_valid_monotonic is not None,
            "pelvis_raw_valid": self._pelvis_raw_valid,
            "pelvis_fault": self._pelvis_fault_active,
            "root_calibration_enabled": self.root_calibration_enabled,
            "root_calibrated": self._root_calibrated,
            "ball_age_ms": None if ball_age_ms is None else round(ball_age_ms, 2),
            "ball_mode": self.ball_estimator.mode(timestamp_s),
            "ball_output_source": (
                self.ball_stream.last_sample.source if self.ball_stream.last_sample is not None else "uninitialized"
            ),
            "ball_measured_count": self._ball_measured_count,
            "ball_fill_count": self._ball_fill_count,
            "ball_prediction_rate_hz": self.ball_prediction_rate_hz,
            "ball_track_epoch": self.ball_estimator.track_epoch,
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"))
        self.status_pub.publish(message)
        self._last_status_publish = now_monotonic

    def _output_callback(self) -> None:
        now_monotonic = time.monotonic()
        now = self.get_clock().now()
        timestamp_s = now.nanoseconds * 1.0e-9
        stamp = now.to_msg()

        pelvis_tracking_ready = self._update_pelvis_watchdog(now_monotonic)
        pelvis_ready = pelvis_tracking_ready and self._root_calibrated
        if pelvis_ready:
            self._publish_pelvis(stamp)
        # RViz uses the last actually published state, never a separate estimate.
        sample = self.ball_stream.last_sample
        ball_estimate = None if sample is None else (
            self._transform_position_w(sample.position_w),
            self._transform_velocity_w(sample.velocity_w),
        )
        if self.visualization_enabled:
            self._publish_visualization(
                stamp,
                ball_estimate,
                timestamp_s,
                pelvis_ready,
            )

        ball_mode = self.ball_estimator.mode(timestamp_s)
        if ball_mode != self._last_ball_mode:
            self.get_logger().info(
                f"Ball mode: {self._last_ball_mode} -> {ball_mode}"
            )
            self._last_ball_mode = ball_mode
        self._publish_status(now_monotonic, timestamp_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Deployment YAML path or name")
    args = parser.parse_args()

    rclpy.init()
    node = NatNetPolicyBridge(args.config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
