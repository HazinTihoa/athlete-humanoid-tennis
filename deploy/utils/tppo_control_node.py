"""Shared ROS controller node for TPPO Student simulation and hardware adapters."""

from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64, String, UInt64

from utils.tppo_intent_student import IntentObservationBuilder
from utils.tppo_student import (
    ACTION_DIM,
    BallState,
    RobotState,
    StudentPolicyController,
    SweetSpotVelocityEstimator,
    quaternion_to_rotation_matrix_wxyz,
)


DEPLOY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DEPLOY_DIR.parent


def resolve_deploy_path(path: str | Path, *, config_dir: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if config_dir is not None:
        relative_to_config = (config_dir / candidate).resolve()
        if relative_to_config.exists():
            return relative_to_config
    return (REPO_ROOT / candidate).resolve()


def load_deploy_config(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        candidate = DEPLOY_DIR / "configs" / path
        path = candidate if candidate.exists() else REPO_ROOT / path
    path = path.resolve()
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"Deployment config must be a mapping: {path}")
    return config, path


class TppoStudentControlNode(Node):
    """Build the Student observation and publish the original 145-D PD command."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        backend: str,
        publish_commands: bool,
    ) -> None:
        if backend not in ("simulation", "hardware"):
            raise ValueError(f"Unsupported backend: {backend}")
        super().__init__(f"tppo_student_{backend}_control")
        self.backend = backend
        self.publish_commands = publish_commands
        self.config, self.config_path = load_deploy_config(config_path)

        policy_path = resolve_deploy_path(
            self.config["policy_path"], config_dir=self.config_path.parent
        )
        self.controller = StudentPolicyController(policy_path)
        policy = self.controller.policy
        expected_observation_dim = self.config.get("expected_observation_dim")
        if (
            expected_observation_dim is not None
            and int(expected_observation_dim) != policy.observation_dim
        ):
            raise ValueError(
                "Configured expected_observation_dim does not match ONNX: "
                f"expected={expected_observation_dim}, actual={policy.observation_dim}"
            )
        self.control_dt = float(self.config.get("control_dt", 0.02))
        self.state_timeout_s = float(self.config.get("state_timeout_s", 0.25))
        self.ball_observation_valid_timeout_s = float(
            self.config.get(
                "ball_observation_valid_timeout_s",
                max(0.05, 1.5 * self.control_dt),
            )
        )
        self.ball_observation_hard_timeout_s = float(
            self.config.get("ball_observation_hard_timeout_s", 0.5)
        )
        self.ball_observation_nominal_latency_s = float(
            self.config.get("ball_observation_nominal_latency_s", 0.0)
        )
        if self.control_dt <= 0.0 or self.state_timeout_s <= 0.0:
            raise ValueError("control_dt and state_timeout_s must be positive.")
        if (
            self.ball_observation_valid_timeout_s <= 0.0
            or self.ball_observation_hard_timeout_s
            < self.ball_observation_valid_timeout_s
            or self.ball_observation_nominal_latency_s < 0.0
        ):
            raise ValueError(
                "Ball observation timeouts must be positive, hard >= valid, "
                "and nominal latency non-negative."
            )

        gain_source = self.config.get("gain_source", "onnx")
        if gain_source == "onnx":
            self.kp = policy.joint_stiffness.copy()
            self.kd = policy.joint_damping.copy()
        elif gain_source == "config":
            self.kp = np.asarray(self.config["Kp"], dtype=np.float64)
            self.kd = np.asarray(self.config["Kd"], dtype=np.float64)
        else:
            raise ValueError("gain_source must be 'onnx' or 'config'.")
        if self.kp.shape != (ACTION_DIM,) or self.kd.shape != (ACTION_DIM,):
            raise ValueError("Kp and Kd must contain 29 values.")

        kinematics_xml = resolve_deploy_path(
            self.config["kinematics_xml_path"], config_dir=self.config_path.parent
        )
        self.sweet_spot_estimator = SweetSpotVelocityEstimator(
            kinematics_xml,
            policy.joint_names,
            site_name=str(self.config.get("sweet_spot_site", "racket_sweet_spot")),
            smoothing=float(self.config.get("sweet_spot_velocity_smoothing", 0.35)),
            position_offset_b=np.asarray(
                self.config.get("sweet_spot_position_offset_b", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            ),
        )
        self.sweet_spot_velocity_mode = self.config.get(
            "sweet_spot_velocity_mode", "world_difference"
        )
        if self.sweet_spot_velocity_mode not in {"world_difference", "relative_joint"}:
            raise ValueError("Invalid sweet_spot_velocity_mode")

        if not policy.is_intent_policy:
            raise ValueError("Deployment requires a supported Intent policy.")
        expected_policy_type = f"intent_{policy.observation_dim}"
        if self.config.get("policy_type") != expected_policy_type:
            raise ValueError(f"Config requires policy_type: {expected_policy_type}")

        self.intent_builder = IntentObservationBuilder(
            default_joint_position=policy.default_joint_position,
            landing_target_startup=np.asarray(
                self.config["landing_target_startup"], dtype=np.float64
            ),
            buffer_length=int(self.config.get("history_buffer_length", 25)),
            history_lags=tuple(
                int(value)
                for value in self.config.get(
                    "history_lags", [20, 15, 10, 5, 0]
                )
            ),
            include_ball_observation_status=(
                policy.includes_ball_observation_status
            ),
            include_global_root_pos=policy.includes_global_root_pos,
            landing_target_observation_frame=self.config.get(
                "landing_target_observation_frame", "startup"
            ),
        )

        self._sim_landing_target_w = None
        self._use_sim_landing_target = backend == "simulation" and bool(
            self.config.get("sim_landing_target_sampling", {}).get("enabled", False)
        )
        if self._use_sim_landing_target:
            if self.intent_builder.landing_target_observation_frame != "current_root":
                raise ValueError("Sampled sim landing target requires current_root observations")
            self.create_subscription(
                Vector3Stamped, "deploy_robot/landing_target", self._landing_target_callback, 10
            )

        self.command_pub = self.create_publisher(
            Float32MultiArray, "deploy_robot/command", 10
        )
        self.observation_pub = self.create_publisher(
            Float32MultiArray, "deploy_robot/student_observation", 10
        )
        self.action_pub = self.create_publisher(
            Float32MultiArray, "deploy_robot/student_action", 10
        )
        self.create_subscription(
            Float32MultiArray,
            "deploy_robot/pelvis_imu_state",
            self._pelvis_imu_callback,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            "deploy_robot/joint_state",
            self._joint_state_callback,
            10,
        )
        self.create_subscription(
            PoseStamped, "/g1_pelvis/pose", self._root_pose_callback, 10
        )
        self.create_subscription(
            PoseStamped, "/ball/pose", self._ball_pose_callback, 10
        )
        self.create_subscription(
            Vector3Stamped, "/ball/velocity", self._ball_velocity_callback, 10
        )
        self.create_subscription(
            UInt64, "/ball/track_epoch", self._ball_track_epoch_callback, 10
        )
        self.fsm_request_pub = self.create_publisher(
            String, "deploy_robot/fsm_request", 10
        )
        self.create_subscription(
            String, "deploy_robot/fsm_state", self._fsm_state_callback, 10
        )
        if self.backend == "simulation":
            self.create_subscription(
                Float64,
                "deploy_robot/control_time",
                self._simulation_control_time_callback,
                10,
            )

        self.base_angular_velocity_b = np.zeros(3, dtype=np.float64)
        self.projected_gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.root_position_w = np.zeros(3, dtype=np.float64)
        self.root_orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self.joint_position = policy.default_joint_position.copy()
        self.joint_velocity = np.zeros(ACTION_DIM, dtype=np.float64)
        self.ball_position_w = np.zeros(3, dtype=np.float64)
        self.ball_velocity_w = np.zeros(3, dtype=np.float64)
        self.sweet_spot_position_w = np.zeros(3, dtype=np.float64)
        self.sweet_spot_velocity_w = np.zeros(3, dtype=np.float64)
        self.fsm_state = "damp"
        self.intent_needs_reset = True
        self._ball_track_epoch: int | None = None
        self._ready = {
            "imu": False,
            "joints": False,
            "root": False,
            "ball_position": False,
            "ball_velocity": False,
        }
        self._last_update_monotonic = {name: 0.0 for name in self._ready}
        if self._use_sim_landing_target:
            self._ready["landing_target"] = False
            self._last_update_monotonic["landing_target"] = 0.0
        self._last_wait_log = 0.0
        self._last_simulation_control_time_s: float | None = None
        self._control_timer = (
            None
            if self.backend == "simulation"
            else self.create_timer(self.control_dt, self._control_callback)
        )

        self.get_logger().info(
            f"Loaded Student {policy_path.name}: obs={policy.observation_dim} "
            f"global_root_pos={policy.includes_global_root_pos} action=29 backend={backend} "
            f"publish_commands={publish_commands} "
            f"intent={policy.is_intent_policy} sweet_spot_state=FK "
            f"clock={'simulation' if backend == 'simulation' else 'wall'}"
        )

    def _mark_ready(self, name: str) -> None:
        self._ready[name] = True
        self._last_update_monotonic[name] = time.monotonic()

    def _pelvis_imu_callback(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float64)
        if data.shape[0] < 13:
            self.get_logger().error(f"pelvis_imu_state needs 13 values, got {data.shape[0]}")
            return
        try:
            rotation_wb = quaternion_to_rotation_matrix_wxyz(data[3:7])
        except ValueError as exc:
            self.get_logger().error(f"Invalid pelvis IMU quaternion: {exc}")
            return
        self.base_angular_velocity_b[:] = data[7:10]
        self.projected_gravity_b[:] = rotation_wb.T @ np.array([0.0, 0.0, -1.0])
        self._mark_ready("imu")

    def _joint_state_callback(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float64)
        if data.shape[0] < 2 * ACTION_DIM:
            self.get_logger().error(f"joint_state needs at least 58 values, got {data.shape[0]}")
            return
        self.joint_position[:] = data[:ACTION_DIM]
        self.joint_velocity[:] = data[ACTION_DIM : 2 * ACTION_DIM]
        self._mark_ready("joints")

    def _root_pose_callback(self, msg: PoseStamped) -> None:
        position = msg.pose.position
        orientation = msg.pose.orientation
        self.root_position_w[:] = (position.x, position.y, position.z)
        self.root_orientation_wxyz[:] = (
            orientation.w,
            orientation.x,
            orientation.y,
            orientation.z,
        )
        self._mark_ready("root")

    def _ball_pose_callback(self, msg: PoseStamped) -> None:
        position = msg.pose.position
        self.ball_position_w[:] = (position.x, position.y, position.z)
        self._mark_ready("ball_position")

    def _landing_target_callback(self, msg: Vector3Stamped) -> None:
        target = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=np.float64)
        if not np.isfinite(target).all():
            self.get_logger().error("Rejected non-finite landing target")
            return
        self._sim_landing_target_w = target
        self._mark_ready("landing_target")

    def _ball_velocity_callback(self, msg: Vector3Stamped) -> None:
        velocity = msg.vector
        self.ball_velocity_w[:] = (velocity.x, velocity.y, velocity.z)
        self._mark_ready("ball_velocity")

    def _ball_track_epoch_callback(self, msg: UInt64) -> None:
        epoch = int(msg.data)
        if self._ball_track_epoch is None:
            self._ball_track_epoch = epoch
            return
        if epoch == self._ball_track_epoch:
            return
        self._ball_track_epoch = epoch
        self.intent_builder.reset_ball_history()
        self.get_logger().info(f"Ball track epoch {epoch}: history reset")

    def _fsm_state_callback(self, msg: String) -> None:
        previous = self.fsm_state
        self.fsm_state = msg.data
        if previous != "control" and self.fsm_state == "control":
            self.intent_needs_reset = True
        if previous == "control" and self.fsm_state != "control":
            self.intent_needs_reset = True
            self.controller.reset()
            self.sweet_spot_estimator.reset()

    def _simulation_control_time_callback(self, msg: Float64) -> None:
        simulation_time_s = float(msg.data)
        if not np.isfinite(simulation_time_s):
            self.get_logger().error("Rejected non-finite simulation control time")
            return
        if (
            self._last_simulation_control_time_s is not None
            and simulation_time_s <= self._last_simulation_control_time_s
        ):
            if simulation_time_s < self._last_simulation_control_time_s:
                self.intent_needs_reset = True
            else:
                return
        self._last_simulation_control_time_s = simulation_time_s
        self._control_callback(sample_time_s=simulation_time_s)

    def request_fsm(self, requested_state: str) -> None:
        if requested_state not in {"damp", "home", "control"}:
            raise ValueError(f"Unsupported FSM request: {requested_state}")
        msg = String()
        msg.data = requested_state
        self.fsm_request_pub.publish(msg)

    def request_damp(self) -> None:
        """Request backend damping before the foreground controller exits."""
        self.request_fsm("damp")
        self.get_logger().info("Shutdown requested: publishing FSM damp")

    def _missing_or_stale_state(self) -> list[str]:
        now = time.monotonic()
        missing = [name for name, ready in self._ready.items() if not ready]

        for name in self._ready:
            timeout_s = self.state_timeout_s
            if (
                self.controller.policy.includes_ball_observation_status
                and name in {"ball_position", "ball_velocity"}
            ):
                timeout_s = self.ball_observation_hard_timeout_s
            if (
                self._ready[name]
                and now - self._last_update_monotonic[name]
                > timeout_s
            ):
                missing.append(f"{name}(stale)")

        return missing

    def _ball_observation_status(self) -> tuple[bool, float]:
        now = time.monotonic()
        age_s = max(
            now - self._last_update_monotonic["ball_position"],
            now - self._last_update_monotonic["ball_velocity"],
        )
        valid = (
            self._ready["ball_position"]
            and self._ready["ball_velocity"]
            and age_s <= self.ball_observation_valid_timeout_s
        )
        return valid, max(0.0, age_s + self.ball_observation_nominal_latency_s)

    def _control_callback(self, sample_time_s: float | None = None) -> None:
        if self.fsm_state != "control":
            return

        policy = self.controller.policy
        waiting_reasons = self._missing_or_stale_state()

        if waiting_reasons:
            now = time.monotonic()
            if now - self._last_wait_log > 2.0:
                self.get_logger().warn(
                    f"Student waiting for: {', '.join(waiting_reasons)}"
                )
                self._last_wait_log = now
            return

        try:
            if self.intent_needs_reset:
                self.controller.reset()
                self.sweet_spot_estimator.reset()
                self.intent_builder.reset(self.root_orientation_wxyz, self.root_position_w)
                self.intent_needs_reset = False
                self.get_logger().info("Intent policy state and history reset")

            if self._use_sim_landing_target:
                # reset() installs the fixed fallback; reapply the latest world
                # target without resetting ball/state histories on each new ball.
                self.intent_builder.landing_target_w[:] = self._sim_landing_target_w

            self.sweet_spot_velocity_w[:] = (
                self.sweet_spot_estimator.update(
                    self.root_position_w,
                    self.root_orientation_wxyz,
                    self.joint_position,
                    time.monotonic() if sample_time_s is None else sample_time_s,
                )
            )
            self.sweet_spot_position_w[:] = (
                self.sweet_spot_estimator.position_w
            )
            relative_velocity_b = None
            if self.sweet_spot_velocity_mode == "relative_joint":
                rotation_wb = quaternion_to_rotation_matrix_wxyz(self.root_orientation_wxyz)
                relative_velocity_b = rotation_wb.T @ self.sweet_spot_estimator.relative_velocity_w(
                    self.joint_velocity
                )

            robot = RobotState(
                root_position_w=self.root_position_w,
                root_orientation_wxyz=self.root_orientation_wxyz,
                base_angular_velocity_b=self.base_angular_velocity_b,
                joint_position=self.joint_position,
                joint_velocity=self.joint_velocity,
                sweet_spot_velocity_w=self.sweet_spot_velocity_w,
            )
            ball = BallState(
                position_w=self.ball_position_w,
                velocity_w=self.ball_velocity_w,
            )
            ball_observation_valid, ball_observation_age_s = (
                self._ball_observation_status()
            )

            observation = self.intent_builder.build_observation(
                robot=robot,
                ball=ball,
                projected_gravity_b=self.projected_gravity_b,
                last_action=self.controller.last_action,
                sweet_spot_position_w=self.sweet_spot_position_w,
                ball_observation_valid=ball_observation_valid,
                ball_observation_age_s=ball_observation_age_s,
                sweet_spot_relative_velocity_b=relative_velocity_b,
            )
            action = policy.infer(observation)
            self.controller.last_action[:] = action
            joint_target = policy.default_joint_position + policy.action_scale * action

        except (RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Student inference rejected: {exc}")
            return

        observation_msg = Float32MultiArray()
        observation_msg.data = observation.tolist()
        self.observation_pub.publish(observation_msg)
        action_msg = Float32MultiArray()
        action_msg.data = action.astype(np.float32).tolist()
        self.action_pub.publish(action_msg)
        if not self.publish_commands:
            return

        command = Float32MultiArray()
        command.data = np.concatenate(
            [
                joint_target,
                np.zeros(ACTION_DIM),
                self.kp,
                self.kd,
                np.zeros(ACTION_DIM),
            ]
        ).astype(np.float32).tolist()
        self.command_pub.publish(command)


def run_student_control_node(
    *,
    backend: str,
    config_path: str,
    publish_commands: bool,
) -> None:
    rclpy.init()
    node = TppoStudentControlNode(
        config_path,
        backend=backend,
        publish_commands=publish_commands,
    )
    stdin_fd: int | None = None
    terminal_settings: list | None = None
    if sys.stdin.isatty():
        stdin_fd = sys.stdin.fileno()
        terminal_settings = termios.tcgetattr(stdin_fd)
        tty.setcbreak(stdin_fd)
        node.get_logger().info(
            "Keyboard FSM: b=damp, a=home, x=control, q=damp and quit"
        )
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if stdin_fd is None:
                continue
            readable, _, _ = select.select([stdin_fd], [], [], 0.0)
            if readable:
                key = os.read(stdin_fd, 1).lower()
                requested_state = {
                    b"b": "damp",
                    b"a": "home",
                    b"x": "control",
                }.get(key)
                if requested_state is not None:
                    node.request_fsm(requested_state)
                    node.get_logger().info(
                        f"Keyboard requested FSM {requested_state}"
                    )
                elif key == b"q":
                    node.request_damp()
                    node.get_logger().info("q pressed; shutting down")
                    break
    except KeyboardInterrupt:
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        if terminal_settings is not None and stdin_fd is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, terminal_settings)
        if rclpy.ok():
            node.request_damp()
            # Give DDS and the 500 Hz hardware loop time to apply damping before
            # the launcher terminates the background hardware process.
            deadline = time.monotonic() + 0.15
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.02)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
