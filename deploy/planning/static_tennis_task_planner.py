"""Shared static-ball task planner for TPPO simulation and hardware deployment."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64, String, UInt64


DEPLOY_DIR = Path(__file__).resolve().parents[1]
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


class StaticTennisTaskPlanner(Node):
    """Latch a stationary ball on each transition into the control FSM state."""

    def __init__(self, config_path: str) -> None:
        super().__init__("static_tennis_task_planner")
        config, _ = load_deploy_config(config_path)
        goal_config = config.get("task_goal")
        planner_config = config.get("static_task_planner", {})
        if not isinstance(goal_config, dict) or not isinstance(planner_config, dict):
            raise ValueError("Deployment config requires task_goal/planner mappings.")

        self.control_dt = float(config.get("control_dt", 0.02))
        self.state_timeout_s = float(
            planner_config.get("state_timeout_s", config.get("state_timeout_s", 0.25))
        )
        self.stationary_speed_threshold = float(
            planner_config.get("stationary_speed_threshold_mps", 0.15)
        )
        self.stationary_hold_s = float(
            planner_config.get("stationary_hold_s", 0.20)
        )
        self.contact_time_s = float(goal_config["contact_time_s"])
        self.target_racket_velocity_b = np.asarray(
            goal_config["target_racket_velocity_b"], dtype=np.float64
        )
        self.target_orientation_b = quaternion_from_rpy_wxyz(
            np.asarray(goal_config["target_racket_orientation_rpy_b"], dtype=np.float64)
        )
        self.landing_target_b = np.asarray(
            goal_config["landing_target_position_b"], dtype=np.float64
        )
        if min(
            self.control_dt,
            self.state_timeout_s,
            self.stationary_speed_threshold,
            self.stationary_hold_s,
            self.contact_time_s,
        ) <= 0.0:
            raise ValueError("Static task planner timing and thresholds must be positive.")
        if self.target_racket_velocity_b.shape != (3,) or self.landing_target_b.shape != (3,):
            raise ValueError("Static task goal vectors must contain 3 values.")

        self.pelvis_position_w = np.zeros(3, dtype=np.float64)
        self.pelvis_orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self.ball_position_w = np.zeros(3, dtype=np.float64)
        self.ball_velocity_w = np.zeros(3, dtype=np.float64)
        self.control_time_s = 0.0
        self.fsm_state = "init"
        self._last_fsm_state = "init"
        self._ready = {name: False for name in ("pelvis", "ball", "velocity", "clock", "fsm")}
        self._last_update = {name: 0.0 for name in self._ready}
        self._stationary_since_control_time: float | None = None
        self._started_this_control = False
        self._active = False
        self._epoch = 0
        self._start_control_time_s = 0.0
        self._goal: TaskGoal | None = None
        self._last_wait_log = 0.0

        self.goal_pub = self.create_publisher(
            Float32MultiArray, "deploy_robot/task_goal", 10
        )
        self.time_pub = self.create_publisher(
            Float64, "deploy_robot/time_remaining", 10
        )
        self.epoch_pub = self.create_publisher(UInt64, "deploy_robot/task_epoch", 10)
        self.create_subscription(PoseStamped, "/g1_pelvis/pose", self._pelvis_callback, 10)
        self.create_subscription(PoseStamped, "/ball/pose", self._ball_callback, 10)
        self.create_subscription(
            Vector3Stamped, "/ball/velocity", self._ball_velocity_callback, 10
        )
        self.create_subscription(
            Float64, "deploy_robot/control_time", self._control_time_callback, 10
        )
        self.create_subscription(String, "deploy_robot/fsm", self._fsm_callback, 10)
        self.create_timer(self.control_dt, self._update)

    def _mark_ready(self, name: str) -> None:
        self._ready[name] = True
        self._last_update[name] = time.monotonic()

    def _pelvis_callback(self, msg: PoseStamped) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        values = np.array([p.x, p.y, p.z, q.w, q.x, q.y, q.z], dtype=np.float64)
        if not np.isfinite(values).all():
            return
        try:
            orientation = normalize_quaternion_wxyz(values[3:7])
        except ValueError:
            return
        self.pelvis_position_w[:] = values[:3]
        self.pelvis_orientation_wxyz[:] = orientation
        self._mark_ready("pelvis")

    def _ball_callback(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        values = np.array([p.x, p.y, p.z], dtype=np.float64)
        if not np.isfinite(values).all():
            return
        self.ball_position_w[:] = values
        self._mark_ready("ball")

    def _ball_velocity_callback(self, msg: Vector3Stamped) -> None:
        v = msg.vector
        values = np.array([v.x, v.y, v.z], dtype=np.float64)
        if not np.isfinite(values).all():
            return
        self.ball_velocity_w[:] = values
        self._mark_ready("velocity")

    def _control_time_callback(self, msg: Float64) -> None:
        value = float(msg.data)
        if not math.isfinite(value):
            return
        if self._ready["clock"] and value < self.control_time_s - 1.0e-6:
            self._started_this_control = False
            self._deactivate()
        self.control_time_s = value
        self._mark_ready("clock")

    def _fsm_callback(self, msg: String) -> None:
        state = msg.data
        if state not in {"init", "damp", "home", "control"}:
            return
        self.fsm_state = state
        self._mark_ready("fsm")

    def _deactivate(self) -> None:
        self._active = False
        self._goal = None
        self._stationary_since_control_time = None

    def _missing_or_stale(self) -> list[str]:
        now = time.monotonic()
        missing = [name for name, ready in self._ready.items() if not ready]
        for name, ready in self._ready.items():
            if ready and now - self._last_update[name] > self.state_timeout_s:
                missing.append(f"{name}(stale)")
        return missing

    def _start_task(self) -> None:
        rotation_wb = quaternion_to_rotation_matrix_wxyz(
            self.pelvis_orientation_wxyz
        )
        self._goal = TaskGoal(
            strike_position_w=self.ball_position_w.copy(),
            target_racket_velocity_w=rotation_wb @ self.target_racket_velocity_b,
            target_racket_orientation_wxyz=quaternion_multiply_wxyz(
                self.pelvis_orientation_wxyz, self.target_orientation_b
            ),
            contact_time_s=self.contact_time_s,
            landing_target_position_w=(
                self.pelvis_position_w + rotation_wb @ self.landing_target_b
            ),
        )
        strike_position_b = rotation_wb.T @ (
            self.ball_position_w - self.pelvis_position_w
        )
        self._epoch += 1
        self._start_control_time_s = self.control_time_s
        self._started_this_control = True
        self._active = True
        self.get_logger().info(
            "Static task started: "
            f"epoch={self._epoch} contact_time={self.contact_time_s:.3f}s "
            f"strike_position_b={strike_position_b.round(4).tolist()}"
        )

    def _publish_task(self) -> None:
        if not self._active or self._goal is None:
            return
        goal = Float32MultiArray()
        goal.data = self._goal.to_message().tolist()
        self.goal_pub.publish(goal)
        remaining = Float64()
        remaining.data = max(
            0.0,
            self.contact_time_s
            - (self.control_time_s - self._start_control_time_s),
        )
        self.time_pub.publish(remaining)
        epoch = UInt64()
        epoch.data = self._epoch
        self.epoch_pub.publish(epoch)

    def _update(self) -> None:
        if self.fsm_state != self._last_fsm_state:
            if self.fsm_state != "control":
                self._started_this_control = False
                self._deactivate()
            self._last_fsm_state = self.fsm_state

        missing = self._missing_or_stale()
        if self.fsm_state != "control" or missing:
            if self.fsm_state == "control" and time.monotonic() - self._last_wait_log > 2.0:
                self.get_logger().warn(f"Static planner waiting for: {', '.join(missing)}")
                self._last_wait_log = time.monotonic()
            return

        if not self._started_this_control:
            if np.linalg.norm(self.ball_velocity_w) <= self.stationary_speed_threshold:
                if self._stationary_since_control_time is None:
                    self._stationary_since_control_time = self.control_time_s
                elif (
                    self.control_time_s - self._stationary_since_control_time
                    >= self.stationary_hold_s
                ):
                    self._start_task()
            else:
                self._stationary_since_control_time = None
        self._publish_task()


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared static tennis task planner.")
    parser.add_argument("--config", required=True, help="Deployment YAML path or name.")
    args = parser.parse_args()
    rclpy.init()
    node = StaticTennisTaskPlanner(args.config)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
