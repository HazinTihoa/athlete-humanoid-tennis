"""Adapt the existing tennis estimator outputs to the TPPO Student goal API."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64, UInt64


DEPLOY_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEPLOY_DIR))

from utils.tppo_control_node import load_deploy_config
from utils.tppo_student import (
    TaskGoal,
    normalize_quaternion_wxyz,
    quaternion_multiply_wxyz,
    quaternion_to_rotation_matrix_wxyz,
)


def quaternion_from_rpy_wxyz(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return normalize_quaternion_wxyz(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        )
    )


class TppoGoalAdapter(Node):
    """Latch one estimator intercept as a fixed Student episode goal."""

    def __init__(self, config_path: str) -> None:
        super().__init__("tppo_goal_adapter")
        config, _ = load_deploy_config(config_path)
        goal_config = config.get("task_goal")
        if not isinstance(goal_config, dict):
            raise ValueError("Deployment config requires a task_goal mapping.")
        self.activation_time_s = float(goal_config.get("activation_time_s", 3.2))
        self.target_racket_velocity_b = np.asarray(
            goal_config["target_racket_velocity_b"], dtype=np.float64
        )
        self.target_orientation_b = quaternion_from_rpy_wxyz(
            np.asarray(goal_config["target_racket_orientation_rpy_b"], dtype=np.float64)
        )
        self.landing_target_b = np.asarray(
            goal_config["landing_target_position_b"], dtype=np.float64
        )
        if (
            self.target_racket_velocity_b.shape != (3,)
            or self.landing_target_b.shape != (3,)
            or self.activation_time_s < 0.0
        ):
            raise ValueError("Invalid task_goal vector dimensions or activation_time_s.")

        self.pelvis_position_w = np.zeros(3, dtype=np.float64)
        self.pelvis_orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self.target_position_b = np.zeros(3, dtype=np.float64)
        self.target_time_s = -1.0
        self._has_pelvis = False
        self._has_target = False
        self._active = False
        self._epoch = 0
        self._latched_goal: TaskGoal | None = None

        self.goal_pub = self.create_publisher(
            Float32MultiArray, "deploy_robot/task_goal", 10
        )
        self.epoch_pub = self.create_publisher(UInt64, "deploy_robot/task_epoch", 10)
        self.time_remaining_pub = self.create_publisher(
            Float64, "deploy_robot/time_remaining", 10
        )
        self.create_subscription(
            PoseStamped, "/g1_pelvis/filtered_pose", self._pelvis_callback, 10
        )
        self.create_subscription(
            PoseStamped, "/ball/target_pose", self._target_pose_callback, 10
        )
        self.create_subscription(
            Float64, "/ball/target_time", self._target_time_callback, 10
        )
        self.create_timer(float(config.get("control_dt", 0.02)), self._publish)

    def _pelvis_callback(self, msg: PoseStamped) -> None:
        position = msg.pose.position
        orientation = msg.pose.orientation
        self.pelvis_position_w[:] = (position.x, position.y, position.z)
        self.pelvis_orientation_wxyz[:] = (
            orientation.w,
            orientation.x,
            orientation.y,
            orientation.z,
        )
        self._has_pelvis = True

    def _target_pose_callback(self, msg: PoseStamped) -> None:
        # Existing estimator contract: /ball/target_pose position is pelvis-local.
        position = msg.pose.position
        self.target_position_b[:] = (position.x, position.y, position.z)
        self._has_target = True

    def _target_time_callback(self, msg: Float64) -> None:
        previous = self.target_time_s
        self.target_time_s = float(msg.data)
        valid = self.target_time_s >= 0.0 and (
            self.activation_time_s == 0.0
            or self.target_time_s <= self.activation_time_s
        )
        if valid and not self._active and self._has_pelvis and self._has_target:
            self._start_episode()
        elif self._active and self.target_time_s < 0.0 and previous >= 0.0:
            self._active = False
            self._latched_goal = None

    def _start_episode(self) -> None:
        rotation_wb = quaternion_to_rotation_matrix_wxyz(
            self.pelvis_orientation_wxyz
        )
        self._latched_goal = TaskGoal(
            strike_position_w=self.pelvis_position_w
            + rotation_wb @ self.target_position_b,
            target_racket_velocity_w=rotation_wb @ self.target_racket_velocity_b,
            target_racket_orientation_wxyz=quaternion_multiply_wxyz(
                self.pelvis_orientation_wxyz, self.target_orientation_b
            ),
            contact_time_s=self.target_time_s,
            landing_target_position_w=self.pelvis_position_w
            + rotation_wb @ self.landing_target_b,
        )
        self._active = True
        self._epoch += 1
        epoch = UInt64()
        epoch.data = self._epoch
        self.epoch_pub.publish(epoch)
        self.get_logger().info(
            f"Latched TPPO goal epoch={self._epoch} contact_time={self.target_time_s:.3f}s"
        )

    def _publish(self) -> None:
        if self._latched_goal is None:
            return
        message = Float32MultiArray()
        message.data = self._latched_goal.to_message().tolist()
        self.goal_pub.publish(message)
        epoch = UInt64()
        epoch.data = self._epoch
        self.epoch_pub.publish(epoch)
        time_remaining = Float64()
        time_remaining.data = max(0.0, self.target_time_s)
        self.time_remaining_pub.publish(time_remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description="TPPO Student goal adapter.")
    parser.add_argument("--config", required=True, help="Deployment YAML path or name.")
    args = parser.parse_args()
    rclpy.init()
    node = TppoGoalAdapter(args.config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
