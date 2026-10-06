"""MuJoCo ROS backend for the deployable TPPO tennis Student."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64, String

DEPLOY_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEPLOY_DIR))

from utils.deployment_fsm import DeploymentFSM
from utils.landing_target import LandingTargetSampler
from utils.tppo_control_node import load_deploy_config, resolve_deploy_path
from utils.tppo_simulation_physics import (
    ExplicitGroundRebound,
    build_training_aligned_model,
)

ACTION_DIM = 29
TENNIS_NET_HEIGHT_M = 0.914


def _rotate_xy(vector: np.ndarray, angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ),
        dtype=np.float64,
    )


class TppoStudentSimulationNode(Node):
    """Preserve the original deploy ROS API while simulating the new task state."""

    def __init__(
        self,
        config_path: str,
        *,
        headless: bool,
        realtime: bool,
        duration_s: float | None,
        interactive_perturbation: bool = False,
    ) -> None:
        super().__init__("tppo_student_simulation")
        self.config, config_file = load_deploy_config(config_path)
        simulation_config = self.config.get("simulation", {})
        self.headless = headless
        self.realtime = realtime
        self.duration_s = duration_s
        self.interactive_perturbation = interactive_perturbation
        if self.interactive_perturbation and self.headless:
            raise ValueError("interactive perturbation requires the native viewer")
        self.control_dt = float(self.config.get("control_dt", 0.02))
        self.sensor_publish_dt = float(simulation_config.get("sensor_publish_dt", 0.01))
        self.configured_physics_dt = float(simulation_config.get("physics_dt", 0.0025))
        court_config = simulation_config.get("court", {})
        self.court_net_x_m = float(court_config.get("net_x_m", 5.6))
        self.court_net_half_width_m = float(
            court_config.get("net_half_width_m", 5.485)
        )
        self.court_net_height_m = float(
            court_config.get("net_height_m", TENNIS_NET_HEIGHT_M)
        )
        incoming_ball_config = simulation_config["incoming_ball"]
        self.incoming_ball_rng = np.random.default_rng(
            int(incoming_ball_config.get("random_seed", 0))
        )
        self.launch_wait_range_s = np.asarray(
            incoming_ball_config["launch_wait_range_s"], dtype=np.float64
        )
        self.maximum_ball_flight_time_s = float(
            incoming_ball_config["maximum_flight_time_s"]
        )
        self.launch_position_min_w = np.asarray(
            incoming_ball_config["position_min_w"], dtype=np.float64
        )
        self.launch_position_max_w = np.asarray(
            incoming_ball_config["position_max_w"], dtype=np.float64
        )
        self.launch_linear_velocity_min_w = np.asarray(
            incoming_ball_config["linear_velocity_min_w"], dtype=np.float64
        )
        self.launch_linear_velocity_max_w = np.asarray(
            incoming_ball_config["linear_velocity_max_w"], dtype=np.float64
        )
        self.launch_angular_velocity_min_w = np.asarray(
            incoming_ball_config["angular_velocity_min_w"], dtype=np.float64
        )
        self.launch_angular_velocity_max_w = np.asarray(
            incoming_ball_config["angular_velocity_max_w"], dtype=np.float64
        )
        self.failure_trajectory_probability = float(
            incoming_ball_config.get("failure_trajectory_probability", 0.0)
        )
        self.failure_trajectory_no_net_fraction = float(
            incoming_ball_config.get("failure_trajectory_no_net_fraction", 0.5)
        )
        self.failure_trajectory_xy_distance_threshold_m = float(
            incoming_ball_config.get(
                "failure_trajectory_xy_distance_threshold_m", 3.0
            )
        )
        self.failure_trajectory_sampling_attempts = int(
            incoming_ball_config.get("failure_trajectory_sampling_attempts", 24)
        )
        self.failure_net_crossing_height_range_m = np.asarray(
            incoming_ball_config.get(
                "failure_net_crossing_height_range_m", [1.5, 3.5]
            ),
            dtype=np.float64,
        )
        self.failure_maximum_initial_speed_m_s = float(
            incoming_ball_config.get("failure_maximum_initial_speed_m_s", 7.0)
        )
        interactive_ball_config = simulation_config.get("interactive_ball", {})
        self.interactive_ball_position_w = np.asarray(
            interactive_ball_config.get(
                "position_w", [0.31931576, 0.93759470, 1.08916278]
            ),
            dtype=np.float64,
        )

        self.joint_names = tuple(
            name.strip() for name in self.config["joint_names"].split(",")
        )
        if len(self.joint_names) != ACTION_DIM:
            raise ValueError("joint_names must contain 29 names.")
        physics_config = self.config["tennis_ball_physics"]
        self.ball_radius_m = float(physics_config["radius_m"])
        self.air_density_kg_m3 = float(physics_config["air_density_kg_m3"])
        self.dynamic_viscosity_pa_s = float(physics_config["dynamic_viscosity_pa_s"])
        self.wind_w = np.asarray(physics_config["wind_w"], dtype=np.float64)
        self.magnus_coefficient = float(physics_config["magnus_coefficient"])
        self.angular_drag_coefficient = float(
            physics_config["angular_drag_coefficient"]
        )
        if self.wind_w.shape != (3,):
            raise ValueError("tennis_ball_physics.wind_w must contain 3 values.")

        robot_xml_path = resolve_deploy_path(
            self.config["simulation_xml_path"], config_dir=config_file.parent
        )
        court_xml_path = resolve_deploy_path(
            self.config.get(
                "tennis_court_xml_path",
                "deploy/simulation/assets/tennis_court.xml",
            ),
            config_dir=config_file.parent,
        )
        self.model, self.physics_sample = build_training_aligned_model(
            robot_xml_path=robot_xml_path,
            court_xml_path=court_xml_path,
            joint_names=self.joint_names,
            physics_dt=self.configured_physics_dt,
            nominal_ball_physics=physics_config,
            randomization_config=simulation_config.get(
                "physics_domain_randomization", {}
            ),
            court_net_x_m=self.court_net_x_m,
            court_net_half_width_m=self.court_net_half_width_m,
            launch_position_min_w=self.launch_position_min_w,
            launch_position_max_w=self.launch_position_max_w,
        )
        self.data = mujoco.MjData(self.model)
        self.sim_dt = float(self.model.opt.timestep)
        self.control_decimation = round(self.control_dt / self.sim_dt)
        self.sensor_decimation = round(self.sensor_publish_dt / self.sim_dt)
        if self.control_decimation < 1 or not math.isclose(
            self.control_decimation * self.sim_dt,
            self.control_dt,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("control_dt must be an integer multiple of physics_dt.")
        if self.sensor_decimation < 1 or not math.isclose(
            self.sensor_decimation * self.sim_dt,
            self.sensor_publish_dt,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "sensor_publish_dt must be an integer multiple of physics_dt."
            )
        self.drag_coefficient = self.physics_sample.drag_coefficient
        self.default_base = np.asarray(
            self.config["default_base_pos"], dtype=np.float64
        )
        self.default_joint_position = np.asarray(
            self.config["default_joint_pos"], dtype=np.float64
        )
        self.hold_kp = np.asarray(self.config["Kp"], dtype=np.float64)
        self.hold_kd = np.asarray(self.config["Kd"], dtype=np.float64)
        self.safety_max_tilt_rad = math.radians(
            float(self.config.get("safety_max_tilt_deg", 60.0))
        )
        if self.default_base.shape != (7,) or self.default_joint_position.shape != (
            ACTION_DIM,
        ):
            raise ValueError(
                "default_base_pos/default_joint_pos dimensions are invalid."
            )
        if self.hold_kp.shape != (ACTION_DIM,) or self.hold_kd.shape != (ACTION_DIM,):
            raise ValueError("Kp/Kd must contain 29 values.")
        if not math.isfinite(self.safety_max_tilt_rad) or self.safety_max_tilt_rad <= 0.0:
            raise ValueError("safety_max_tilt_deg must be finite and positive.")
        if (
            min(
                self.control_dt,
                self.sensor_publish_dt,
                self.sim_dt,
                self.maximum_ball_flight_time_s,
                self.court_net_x_m,
                self.court_net_half_width_m,
                self.court_net_height_m,
            )
            <= 0.0
        ):
            raise ValueError("Simulation and publishing periods must be positive.")
        range_vectors = (
            self.launch_position_min_w,
            self.launch_position_max_w,
            self.launch_linear_velocity_min_w,
            self.launch_linear_velocity_max_w,
            self.launch_angular_velocity_min_w,
            self.launch_angular_velocity_max_w,
            self.interactive_ball_position_w,
        )
        if any(value.shape != (3,) for value in range_vectors):
            raise ValueError("Ball position and range vectors must contain 3 values.")
        if self.launch_wait_range_s.shape != (2,):
            raise ValueError("launch_wait_range_s must contain 2 values.")
        if np.any(self.launch_position_min_w >= self.launch_position_max_w):
            raise ValueError("Incoming-ball position range is invalid.")
        if np.any(
            self.launch_linear_velocity_min_w > self.launch_linear_velocity_max_w
        ):
            raise ValueError("Incoming-ball linear velocity range is invalid.")
        if np.any(
            self.launch_angular_velocity_min_w > self.launch_angular_velocity_max_w
        ):
            raise ValueError("Incoming-ball angular velocity range is invalid.")
        if not 0.0 <= self.launch_wait_range_s[0] <= self.launch_wait_range_s[1]:
            raise ValueError("Incoming-ball launch wait range is invalid.")
        if not 0.0 <= self.failure_trajectory_probability <= 1.0:
            raise ValueError("failure_trajectory_probability must lie in [0, 1].")
        if not 0.0 <= self.failure_trajectory_no_net_fraction <= 1.0:
            raise ValueError(
                "failure_trajectory_no_net_fraction must lie in [0, 1]."
            )
        if self.failure_trajectory_xy_distance_threshold_m <= 0.0:
            raise ValueError(
                "failure_trajectory_xy_distance_threshold_m must be positive."
            )
        if self.failure_trajectory_sampling_attempts <= 0:
            raise ValueError("failure_trajectory_sampling_attempts must be positive.")
        if (
            self.failure_net_crossing_height_range_m.shape != (2,)
            or not np.all(np.isfinite(self.failure_net_crossing_height_range_m))
            or self.failure_net_crossing_height_range_m[0] <= 0.0
            or self.failure_net_crossing_height_range_m[0]
            > self.failure_net_crossing_height_range_m[1]
        ):
            raise ValueError("failure_net_crossing_height_range_m is invalid.")
        if self.failure_maximum_initial_speed_m_s <= 0.0:
            raise ValueError("failure_maximum_initial_speed_m_s must be positive.")
        if self.duration_s is not None and self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive when provided.")

        self.pelvis_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.ball_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "tennis_ball")
        self.ball_joint_id = self._required_id(
            mujoco.mjtObj.mjOBJ_JOINT, "tennis_ball_freejoint"
        )
        self.ball_geom_id = self._required_id(
            mujoco.mjtObj.mjOBJ_GEOM, "tennis_ball_geom"
        )
        self.racket_geom_id = self._required_id(
            mujoco.mjtObj.mjOBJ_GEOM, "racket_ball_collision"
        )
        self.sweet_spot_site_id = self._required_id(
            mujoco.mjtObj.mjOBJ_SITE, "racket_sweet_spot"
        )
        expected_ball_mass = self.physics_sample.ball_mass_kg
        actual_ball_mass = float(self.model.body_mass[self.ball_body_id])
        actual_ball_radius = float(self.model.geom_size[self.ball_geom_id, 0])
        if not math.isclose(
            actual_ball_mass, expected_ball_mass, rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise ValueError(
                f"Ball mass mismatch: XML={actual_ball_mass} config={expected_ball_mass}"
            )
        if not math.isclose(
            actual_ball_radius, self.ball_radius_m, rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise ValueError(
                f"Ball radius mismatch: XML={actual_ball_radius} config={self.ball_radius_m}"
            )
        self.ball_qpos_address = int(self.model.jnt_qposadr[self.ball_joint_id])
        self.ball_dof_address = int(self.model.jnt_dofadr[self.ball_joint_id])
        self.ground_rebound = ExplicitGroundRebound(
            self.model,
            ball_radius_m=self.ball_radius_m,
            court_restitution=self.physics_sample.court_restitution,
            tangent_speed_retention=(
                self.physics_sample.ground_tangent_speed_retention
            ),
        )
        self.joint_qpos_addresses = np.array(
            [
                self.model.jnt_qposadr[
                    self._required_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in self.joint_names
            ],
            dtype=np.int32,
        )
        self.joint_dof_addresses = np.array(
            [
                self.model.jnt_dofadr[
                    self._required_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in self.joint_names
            ],
            dtype=np.int32,
        )
        self.actuator_ids = np.array(
            [
                self._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.joint_names
            ],
            dtype=np.int32,
        )
        self.deployment_fsm = DeploymentFSM(
            default_joint_position=self.default_joint_position,
            home_kp=self.hold_kp,
            home_kd=self.hold_kd,
            home_duration_s=float(self.config["home_pos_duration"]),
            command_timeout_s=float(self.config["command_timeout_s"]),
            damp_kd=float(self.config.get("damp_kd", 3.0)),
        )
        self._last_logged_fsm_state = self.deployment_fsm.state
        self._last_logged_fault: str | None = None
        self._startup_support_active = True
        landing_cfg = self.config.get("sim_landing_target_sampling", {})
        self.landing_sampler = None
        self.landing_target_w = None
        if landing_cfg.get("enabled", False):
            self.landing_sampler = LandingTargetSampler(
                self.config["landing_target_startup"], landing_cfg["std_xy"],
                self.default_base, self.court_net_x_m + landing_cfg["net_margin_m"],
                landing_cfg.get("seed", 0),
            )
        self.landing_ring_radius = float(landing_cfg.get("ring_radius_m", .5))
        if not math.isfinite(self.landing_ring_radius) or self.landing_ring_radius <= 0:
            raise ValueError("Landing ring radius must be positive")
        self.landing_target_pub = self.create_publisher(
            Vector3Stamped, "deploy_robot/landing_target", 10
        )

        self.pelvis_imu_pub = self.create_publisher(
            Float32MultiArray, "deploy_robot/pelvis_imu_state", 10
        )
        self.joint_state_pub = self.create_publisher(
            Float32MultiArray, "deploy_robot/joint_state", 10
        )
        self.simulation_time_pub = self.create_publisher(
            Float64, "deploy_robot/simulation_time", 10
        )
        self.control_time_pub = self.create_publisher(
            Float64, "deploy_robot/control_time", 10
        )
        self.fsm_state_pub = self.create_publisher(
            String, "deploy_robot/fsm_state", 10
        )
        self.fsm_time_pub = self.create_publisher(
            Float64, "deploy_robot/fsm_time", 10
        )
        self.root_pose_pub = self.create_publisher(PoseStamped, "/g1_pelvis/pose", 10)
        self.ball_pose_pub = self.create_publisher(PoseStamped, "/ball/pose", 10)
        self.ball_velocity_pub = self.create_publisher(
            Vector3Stamped, "/ball/velocity", 10
        )
        self.create_subscription(
            Float32MultiArray, "deploy_robot/command", self._command_callback, 10
        )
        self.create_subscription(
            String,
            "deploy_robot/fsm_request",
            self._fsm_request_callback,
            10,
        )

        self.ball_cycle_index = 0
        self.ball_flying = False
        self.launch_wait_remaining_s = 0.0
        self.ball_flight_elapsed_s = 0.0
        self.pending_ball_position_w = np.zeros(3)
        self.pending_ball_linear_velocity_w = np.zeros(3)
        self.pending_ball_angular_velocity_w = np.zeros(3)
        self.minimum_sweet_spot_distance_m = math.inf
        self.minimum_distance_time_s = 0.0
        self.racket_ball_contact_time_s: float | None = None
        self.pending_ball_trajectory_type = "normal"
        self.scheduled_ball_count = 0
        self.scheduled_failure_count = 0
        self._interactive_ball_reset_requested = False
        self._reset_simulation()

        self.stop_requested = False
        self.viewer = None
        if not headless:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                key_callback=self._viewer_key_callback,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.cam.azimuth = 135.0
            self.viewer.cam.elevation = -30.0
            self.viewer.cam.distance = 21.5
            self.viewer.cam.lookat[:] = (6.0, 0.0, 0.55)
            if self.interactive_perturbation:
                self._configure_interactive_perturbation()

        self.start_wall_time = time.perf_counter()
        self.next_step_deadline = self.start_wall_time + self.sim_dt
        self.physics_step_count = 0
        self.actual_realtime = 0.0
        self.viewer_fps = 0.0
        self._runtime_window_start = self.start_wall_time
        self._runtime_window_simulated_s = 0.0
        self._runtime_window_frames = 0
        self._last_overlay_update = 0.0
        self._last_viewer_sync = 0.0
        self._viewer_sync_period_s = 1.0 / 60.0
        self.sensor_publish_accumulator = 0.0
        self.control_publish_accumulator = 0.0
        self.create_timer(1.0e-6, self._step)
        self.get_logger().info(
            f"Built training-aligned TPPO scene from {robot_xml_path}; "
            f"dt={self.sim_dt:.4f}s "
            f"control_decimation={self.control_decimation} "
            f"headless={headless} realtime={realtime} "
            f"interactive_perturbation={interactive_perturbation}"
        )
        self.get_logger().info(
            "Physics sample: "
            f"ball_mass={self.physics_sample.ball_mass_kg:.4f}kg "
            f"court_e={self.physics_sample.court_restitution:.3f} "
            "tangent_retention="
            f"{self.physics_sample.ground_tangent_speed_retention:.3f} "
            f"Cd={self.physics_sample.drag_coefficient:.3f} "
            f"racket_mass={self.physics_sample.racket_mass_kg:.3f}kg "
            f"racket_e={self.physics_sample.racket_restitution:.3f} "
            f"foot_friction={self.physics_sample.foot_sliding_friction:.3f}"
        )
        self.get_logger().info(
            "Deploy-sim failure trajectories: "
            f"probability={self.failure_trajectory_probability:.3f} "
            f"no_net_fraction={self.failure_trajectory_no_net_fraction:.3f} "
            "far_threshold="
            f"{self.failure_trajectory_xy_distance_threshold_m:.3f}m"
        )

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo model is missing {object_type.name}: {name}")
        return object_id

    def _viewer_key_callback(self, keycode: int) -> None:
        try:
            key = chr(keycode).lower()
        except (TypeError, ValueError):
            return
        requested_state = {"b": "damp", "a": "home", "x": "control"}.get(key)
        if requested_state is not None:
            self.deployment_fsm.request(requested_state)
            self.get_logger().info(f"Viewer requested FSM {requested_state}")
        elif key == "q":
            self.deployment_fsm.request("damp")
            self.stop_requested = True
            self.get_logger().info("Viewer requested damp and shutdown")
        elif key == "r" and self.interactive_perturbation:
            self._interactive_ball_reset_requested = True
            self.get_logger().info("Viewer requested a new incoming ball")

    def _configure_interactive_perturbation(self) -> None:
        if self.viewer is None:
            return
        with self.viewer.lock():
            self.viewer.perturb.select = self.ball_body_id
            self.viewer.perturb.localpos[:] = 0.0
            self.viewer.opt.flags[
                mujoco.mjtVisFlag.mjVIS_PERTFORCE
            ] = True
            self.viewer.opt.flags[
                mujoco.mjtVisFlag.mjVIS_PERTOBJ
            ] = True
        self.get_logger().info(
            "Interactive perturbation enabled; tennis ball is preselected"
        )

    def _apply_interactive_perturbation(self) -> None:
        if not self.interactive_perturbation or self.viewer is None:
            return
        with self.viewer.lock():
            mujoco.mjv_applyPerturbPose(
                self.model, self.data, self.viewer.perturb, 0
            )
            mujoco.mjv_applyPerturbForce(
                self.model, self.data, self.viewer.perturb
            )

    def _set_ball_state(
        self,
        position_w: np.ndarray,
        linear_velocity_w: np.ndarray | None = None,
        angular_velocity_w: np.ndarray | None = None,
    ) -> None:
        self.data.qpos[self.ball_qpos_address : self.ball_qpos_address + 3] = position_w
        self.data.qpos[self.ball_qpos_address + 3 : self.ball_qpos_address + 7] = (
            1.0,
            0.0,
            0.0,
            0.0,
        )
        velocity = self.data.qvel[self.ball_dof_address : self.ball_dof_address + 6]
        velocity[:3] = 0.0 if linear_velocity_w is None else linear_velocity_w
        velocity[3:6] = 0.0 if angular_velocity_w is None else angular_velocity_w

    def _sample_uniform_incoming_ball_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position_w = self.incoming_ball_rng.uniform(
            self.launch_position_min_w, self.launch_position_max_w
        )
        linear_velocity_w = self.incoming_ball_rng.uniform(
            self.launch_linear_velocity_min_w,
            self.launch_linear_velocity_max_w,
        )
        angular_velocity_w = self.incoming_ball_rng.uniform(
            self.launch_angular_velocity_min_w,
            self.launch_angular_velocity_max_w,
        )
        return position_w, linear_velocity_w, angular_velocity_w

    def _retarget_failure_velocity(
        self,
        position_w: np.ndarray,
        linear_velocity_w: np.ndarray,
        root_position_xy: np.ndarray,
        failure_type: str,
    ) -> np.ndarray:
        velocity_w = linear_velocity_w.copy()
        if velocity_w[0] >= -1.0e-6:
            velocity_w[0] = -max(abs(velocity_w[0]), 1.0e-3)
        time_to_net = max(
            (self.court_net_x_m - position_w[0]) / velocity_w[0], 1.0e-6
        )

        if failure_type == "no_net":
            crossing_height = 0.5 * self.court_net_height_m
        elif failure_type == "too_far":
            launch_from_root = position_w[:2] - root_position_xy
            launch_distance = float(np.linalg.norm(launch_from_root))
            if launch_distance <= 1.0e-6:
                raise RuntimeError("Cannot direct a failure trajectory from the root.")
            direct_to_root = -launch_from_root / launch_distance
            minimum_miss = min(
                self.failure_trajectory_xy_distance_threshold_m + 0.25,
                launch_distance * 0.8,
            )
            maximum_miss = min(
                self.failure_trajectory_xy_distance_threshold_m + 1.0,
                launch_distance * 0.8,
            )
            miss_distance = float(
                self.incoming_ball_rng.uniform(minimum_miss, maximum_miss)
            )
            miss_angle = math.asin(np.clip(miss_distance / launch_distance, 0.0, 0.8))
            miss_angle *= -1.0 if self.incoming_ball_rng.random() < 0.5 else 1.0
            horizontal_direction = _rotate_xy(direct_to_root, miss_angle)
            horizontal_speed = float(np.linalg.norm(velocity_w[:2]))
            velocity_w[:2] = horizontal_direction * horizontal_speed
            time_to_net = max(
                (self.court_net_x_m - position_w[0]) / velocity_w[0], 1.0e-6
            )
            crossing_height = float(
                self.incoming_ball_rng.uniform(
                    *self.failure_net_crossing_height_range_m
                )
            )
        else:
            raise ValueError(f"Unsupported failure trajectory type: {failure_type}")

        required_vz = (
            crossing_height
            - position_w[2]
            + 0.5 * 9.81 * time_to_net**2
        ) / time_to_net
        horizontal_speed_sq = float(np.dot(velocity_w[:2], velocity_w[:2]))
        vertical_speed_cap = math.sqrt(
            max(self.failure_maximum_initial_speed_m_s**2 - horizontal_speed_sq, 0.0)
        )
        velocity_w[2] = float(np.clip(required_vz, -vertical_speed_cap, vertical_speed_cap))
        return velocity_w

    def _predict_incoming_trajectory_type(
        self,
        position_w: np.ndarray,
        linear_velocity_w: np.ndarray,
        root_position_xy: np.ndarray,
    ) -> str:
        position = position_w.astype(np.float64, copy=True)
        velocity = linear_velocity_w.astype(np.float64, copy=True)
        net_cleared = False
        bounce_count = 0
        closest_post_bounce_xy = math.inf
        ground_z = self.ball_radius_m
        steps = max(1, round(self.maximum_ball_flight_time_s / self.sim_dt))
        diameter = 2.0 * self.ball_radius_m
        cross_section = math.pi * self.ball_radius_m**2

        for _ in range(steps):
            relative_velocity = velocity - self.wind_w
            speed = float(np.linalg.norm(relative_velocity))
            drag_scale = (
                0.5
                * self.air_density_kg_m3
                * self.drag_coefficient
                * cross_section
                * speed
                + 3.0
                * math.pi
                * diameter
                * self.dynamic_viscosity_pa_s
            )
            acceleration = np.array((0.0, 0.0, -9.81), dtype=np.float64)
            acceleration -= (
                drag_scale / self.physics_sample.ball_mass_kg
            ) * relative_velocity
            next_velocity = velocity + acceleration * self.sim_dt
            next_position = position + next_velocity * self.sim_dt

            if (
                bounce_count == 0
                and position[0] > self.court_net_x_m >= next_position[0]
            ):
                denominator = position[0] - next_position[0]
                alpha = (
                    (position[0] - self.court_net_x_m) / denominator
                    if denominator > 1.0e-12
                    else 0.0
                )
                crossing = position + np.clip(alpha, 0.0, 1.0) * (
                    next_position - position
                )
                net_cleared = bool(
                    abs(crossing[1]) <= self.court_net_half_width_m
                    and crossing[2]
                    > self.court_net_height_m + self.ball_radius_m
                )

            if next_position[2] < ground_z and next_velocity[2] < 0.0:
                denominator = position[2] - next_position[2]
                fraction = (
                    (position[2] - ground_z) / denominator
                    if denominator > 1.0e-12
                    else 0.0
                )
                fraction = float(np.clip(fraction, 0.0, 1.0))
                impact_position = position + fraction * (next_position - position)
                reflected_velocity = next_velocity.copy()
                reflected_velocity[:2] *= (
                    self.physics_sample.ground_tangent_speed_retention
                )
                reflected_velocity[2] *= -self.physics_sample.court_restitution
                remaining_dt = (1.0 - fraction) * self.sim_dt
                next_position = impact_position + reflected_velocity * remaining_dt
                next_position[2] = max(next_position[2], ground_z)
                next_velocity = reflected_velocity
                bounce_count += 1

            position = next_position
            velocity = next_velocity
            if bounce_count == 1 and net_cleared:
                closest_post_bounce_xy = min(
                    closest_post_bounce_xy,
                    float(np.linalg.norm(position[:2] - root_position_xy)),
                )
            if bounce_count >= 2:
                break

        if not net_cleared:
            return "no_net"
        if (
            math.isfinite(closest_post_bounce_xy)
            and closest_post_bounce_xy
            > self.failure_trajectory_xy_distance_threshold_m
        ):
            return "too_far"
        return "normal"

    def _sample_incoming_ball_state(
        self, root_position_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
        request_failure = bool(
            self.incoming_ball_rng.random() < self.failure_trajectory_probability
        )
        requested_type = "normal"
        if request_failure:
            requested_type = (
                "no_net"
                if self.incoming_ball_rng.random()
                < self.failure_trajectory_no_net_fraction
                else "too_far"
            )
        fallback_failure = None

        for attempt in range(1, self.failure_trajectory_sampling_attempts + 1):
            position_w, velocity_w, angular_velocity_w = (
                self._sample_uniform_incoming_ball_state()
            )
            if request_failure:
                velocity_w = self._retarget_failure_velocity(
                    position_w,
                    velocity_w,
                    root_position_xy,
                    requested_type,
                )
            actual_type = self._predict_incoming_trajectory_type(
                position_w, velocity_w, root_position_xy
            )
            sample = (
                position_w,
                velocity_w,
                angular_velocity_w,
                actual_type,
                attempt,
            )
            if actual_type == requested_type:
                return sample
            if request_failure and actual_type != "normal":
                fallback_failure = sample

        if fallback_failure is not None:
            return fallback_failure
        raise RuntimeError(
            "Deploy sim could not sample requested incoming trajectory type "
            f"{requested_type!r} after {self.failure_trajectory_sampling_attempts} attempts."
        )

    def _log_completed_ball(self) -> None:
        if self.ball_cycle_index <= 0:
            return
        contact = (
            "none"
            if self.racket_ball_contact_time_s is None
            else f"{self.racket_ball_contact_time_s:.3f}s"
        )
        self.get_logger().info(
            f"Ball {self.ball_cycle_index} diagnostics: "
            f"min_sweet_ball={self.minimum_sweet_spot_distance_m:.4f}m "
            f"at={self.minimum_distance_time_s:.3f}s contact={contact}"
        )

    def _schedule_next_ball(self) -> None:
        if self.ball_flying:
            self._log_completed_ball()
        self._sample_landing_target()
        (
            self.pending_ball_position_w,
            self.pending_ball_linear_velocity_w,
            self.pending_ball_angular_velocity_w,
            self.pending_ball_trajectory_type,
            sampling_attempts,
        ) = self._sample_incoming_ball_state(
            self.data.xpos[self.pelvis_body_id, :2].copy()
        )
        self.scheduled_ball_count += 1
        if self.pending_ball_trajectory_type != "normal":
            self.scheduled_failure_count += 1
        self.launch_wait_remaining_s = float(
            self.incoming_ball_rng.uniform(*self.launch_wait_range_s)
        )
        self.ball_flying = False
        self.ball_flight_elapsed_s = 0.0
        self.minimum_sweet_spot_distance_m = math.inf
        self.minimum_distance_time_s = 0.0
        self.racket_ball_contact_time_s = None
        self.ground_rebound.reset()
        self.data.xfrc_applied[self.ball_body_id] = 0.0
        self._set_ball_state(self.pending_ball_position_w)
        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info(
            "Next ball waiting: "
            f"delay={self.launch_wait_remaining_s:.3f}s "
            f"position={np.round(self.pending_ball_position_w, 3).tolist()} "
            f"velocity={np.round(self.pending_ball_linear_velocity_w, 3).tolist()} "
            f"trajectory={self.pending_ball_trajectory_type} "
            f"attempts={sampling_attempts}"
        )

    def _sample_landing_target(self) -> None:
        if self.landing_sampler is not None:
            self.landing_target_w = self.landing_sampler.sample()
            self.get_logger().info(
                f"Landing target sample: world={self.landing_target_w.round(4).tolist()} "
                f"std_xy={self.landing_sampler.std.tolist()}"
            )

    def _reset_interactive_ball(self) -> None:
        self._sample_landing_target()
        if self.ball_flying:
            self._log_completed_ball()
        self.pending_ball_position_w[:] = self.interactive_ball_position_w
        self.pending_ball_linear_velocity_w.fill(0.0)
        self.pending_ball_angular_velocity_w.fill(0.0)
        self.ball_flying = False
        self.ball_flight_elapsed_s = 0.0
        self.minimum_sweet_spot_distance_m = math.inf
        self.minimum_distance_time_s = 0.0
        self.racket_ball_contact_time_s = None
        self.ground_rebound.reset()
        self.data.xfrc_applied[self.ball_body_id] = 0.0
        self._set_ball_state(self.pending_ball_position_w)
        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info(
            "Interactive ball reset: "
            f"position={np.round(self.pending_ball_position_w, 3).tolist()}"
        )

    def _interactive_ball_perturbation_active(self) -> bool:
        if self.viewer is None:
            return False
        with self.viewer.lock():
            perturb = self.viewer.perturb
            return bool(
                perturb.select == self.ball_body_id and perturb.active != 0
            )

    def _release_interactive_ball(self) -> None:
        self.ball_cycle_index += 1
        self.ball_flying = True
        self.ball_flight_elapsed_s = 0.0
        self.ground_rebound.reset()
        self.get_logger().info(
            f"Interactive ball {self.ball_cycle_index} released by mouse perturbation"
        )

    def _launch_pending_ball(self) -> None:
        self.ball_cycle_index += 1
        self.ball_flying = True
        self.ball_flight_elapsed_s = 0.0
        self.ground_rebound.reset()
        self._set_ball_state(
            self.pending_ball_position_w,
            self.pending_ball_linear_velocity_w,
            self.pending_ball_angular_velocity_w,
        )
        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info(
            f"Launched ball {self.ball_cycle_index}: "
            f"position={np.round(self.pending_ball_position_w, 3).tolist()} "
            f"velocity={np.round(self.pending_ball_linear_velocity_w, 3).tolist()} "
            f"trajectory={self.pending_ball_trajectory_type}"
        )

    def _reset_simulation(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.default_base
        self.data.qpos[self.joint_qpos_addresses] = self.default_joint_position
        self.data.qvel[:] = 0.0
        self.ground_rebound.reset()
        mujoco.mj_forward(self.model, self.data)
        if self.interactive_perturbation:
            self._reset_interactive_ball()
        else:
            self._schedule_next_ball()

    def _command_callback(self, msg: Float32MultiArray) -> None:
        command = np.asarray(msg.data, dtype=np.float64)
        if command.shape != (5 * ACTION_DIM,):
            self.get_logger().error(
                f"deploy_robot/command needs {5 * ACTION_DIM} values, got {command.shape}"
            )
            return
        if not np.isfinite(command).all():
            self.get_logger().error("Rejected command containing NaN or Inf.")
            return
        self.deployment_fsm.update_policy_command(
            command, now_s=float(self.data.time)
        )

    def _fsm_request_callback(self, msg: String) -> None:
        try:
            self.deployment_fsm.request(msg.data)
        except ValueError:
            self.get_logger().error(f"Rejected FSM request: {msg.data!r}")

    def _log_fsm_changes(self) -> None:
        state = self.deployment_fsm.state
        fault = self.deployment_fsm.fault_reason
        if state != self._last_logged_fsm_state:
            self.get_logger().info(
                f"FSM: {self._last_logged_fsm_state} -> {state}"
            )
            self._last_logged_fsm_state = state
        if fault != self._last_logged_fault:
            if fault is not None:
                self.get_logger().error(f"FSM forced damp: {fault}")
            self._last_logged_fault = fault

    def _compute_torque(self) -> np.ndarray:
        joint_position = self.data.qpos[self.joint_qpos_addresses]
        joint_velocity = self.data.qvel[self.joint_dof_addresses]
        return (
            self.deployment_fsm.kp
            * (self.deployment_fsm.q_target - joint_position)
            + self.deployment_fsm.kd
            * (self.deployment_fsm.dq_target - joint_velocity)
            + self.deployment_fsm.tau_ff
        )

    def _apply_startup_pelvis_support(self) -> None:
        """Hold only the floating pelvis until the first control entry."""
        self.data.qpos[:7] = self.default_base
        self.data.qvel[:6] = 0.0

    def _object_linear_velocity_w(
        self, object_type: mujoco.mjtObj, object_id: int
    ) -> np.ndarray:
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, object_type, object_id, velocity, 0
        )
        return velocity[3:].copy()

    def _apply_ball_aerodynamics(self) -> None:
        velocity = self.data.qvel[self.ball_dof_address : self.ball_dof_address + 6]
        linear_velocity_w = velocity[:3]
        angular_velocity_w = velocity[3:6]
        relative_velocity = linear_velocity_w - self.wind_w
        speed = float(np.linalg.norm(relative_velocity))
        angular_speed = float(np.linalg.norm(angular_velocity_w))
        diameter = 2.0 * self.ball_radius_m
        cross_section = math.pi * self.ball_radius_m**2
        volume = (4.0 / 3.0) * math.pi * self.ball_radius_m**3
        fluid_angular_inertia = (8.0 / 15.0) * math.pi * self.ball_radius_m**5
        linear_drag = (
            -(
                0.5
                * self.air_density_kg_m3
                * self.drag_coefficient
                * cross_section
                * speed
                + 3.0 * math.pi * diameter * self.dynamic_viscosity_pa_s
            )
            * relative_velocity
        )
        magnus = (
            self.air_density_kg_m3
            * volume
            * self.magnus_coefficient
            * np.cross(angular_velocity_w, relative_velocity)
        )
        angular_drag = (
            -(
                self.air_density_kg_m3
                * self.angular_drag_coefficient
                * fluid_angular_inertia
                * angular_speed
                + math.pi * diameter**3 * self.dynamic_viscosity_pa_s
            )
            * angular_velocity_w
        )
        self.data.xfrc_applied[self.ball_body_id, :3] += linear_drag + magnus
        self.data.xfrc_applied[self.ball_body_id, 3:6] += angular_drag

    def _update_strike_diagnostics(self) -> None:
        if not self.ball_flying:
            return
        distance = float(
            np.linalg.norm(
                self.data.site_xpos[self.sweet_spot_site_id]
                - self.data.xpos[self.ball_body_id]
            )
        )
        if distance < self.minimum_sweet_spot_distance_m:
            self.minimum_sweet_spot_distance_m = distance
            self.minimum_distance_time_s = self.ball_flight_elapsed_s
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if {int(contact.geom1), int(contact.geom2)} == {
                self.ball_geom_id,
                self.racket_geom_id,
            }:
                if self.racket_ball_contact_time_s is None:
                    self.racket_ball_contact_time_s = self.ball_flight_elapsed_s
                    self.get_logger().info(
                        "Racket-ball contact: "
                        f"ball={self.ball_cycle_index} "
                        f"flight_time={self.ball_flight_elapsed_s:.3f}s"
                    )
                break

    def _publish_pose(
        self, publisher, position: np.ndarray, quaternion_wxyz: np.ndarray
    ) -> None:
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.pose.position.x, message.pose.position.y, message.pose.position.z = (
            position
        )
        message.pose.orientation.w = float(quaternion_wxyz[0])
        message.pose.orientation.x = float(quaternion_wxyz[1])
        message.pose.orientation.y = float(quaternion_wxyz[2])
        message.pose.orientation.z = float(quaternion_wxyz[3])
        publisher.publish(message)

    def _publish_vector(self, publisher, vector: np.ndarray) -> None:
        message = Vector3Stamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.vector.x, message.vector.y, message.vector.z = vector
        publisher.publish(message)

    def _publish_state(self) -> None:
        if self.landing_target_w is not None:
            self._publish_vector(self.landing_target_pub, self.landing_target_w)
        root_position = self.data.xpos[self.pelvis_body_id].copy()
        root_quaternion = self.data.xquat[self.pelvis_body_id].copy()
        root_angular_velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.pelvis_body_id,
            root_angular_velocity,
            1,
        )
        imu = Float32MultiArray()
        imu.data = (
            np.concatenate(
                [
                    np.zeros(3),
                    root_quaternion,
                    root_angular_velocity[:3],
                    np.zeros(3),
                ]
            )
            .astype(np.float32)
            .tolist()
        )
        self.pelvis_imu_pub.publish(imu)

        joint_state = Float32MultiArray()
        joint_state.data = (
            np.concatenate(
                [
                    self.data.qpos[self.joint_qpos_addresses],
                    self.data.qvel[self.joint_dof_addresses],
                    np.zeros(ACTION_DIM),
                    self.data.ctrl[self.actuator_ids],
                ]
            )
            .astype(np.float32)
            .tolist()
        )
        self.joint_state_pub.publish(joint_state)
        self._publish_pose(self.root_pose_pub, root_position, root_quaternion)
        self._publish_pose(
            self.ball_pose_pub,
            self.data.xpos[self.ball_body_id],
            self.data.xquat[self.ball_body_id],
        )
        self._publish_vector(
            self.ball_velocity_pub,
            self._object_linear_velocity_w(mujoco.mjtObj.mjOBJ_BODY, self.ball_body_id),
        )

        fsm_state = String()
        fsm_state.data = self.deployment_fsm.state
        self.fsm_state_pub.publish(fsm_state)
        fsm_time = Float64()
        fsm_time.data = max(
            0.0, float(self.data.time) - self.deployment_fsm.state_entry_s
        )
        self.fsm_time_pub.publish(fsm_time)

    def _publish_control_tick(self) -> None:
        control_time = Float64()
        control_time.data = float(self.data.time)
        self.control_time_pub.publish(control_time)

    def _draw_landing_target(self) -> None:
        if self.landing_target_w is None:
            return
        with self.viewer.lock():
            scene = self.viewer.user_scn
            scene.ngeom = 0
            angles = np.linspace(0, 2 * np.pi, 65)
            points = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(65)))
            points *= self.landing_ring_radius
            points += self.landing_target_w + np.array([0., 0., .025])
            for start, end in zip(points[:-1], points[1:]):
                if scene.ngeom >= scene.maxgeom:
                    break
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                   np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                   np.array([1., .75, .05, .85]))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .012, start, end)
                scene.ngeom += 1

    def _publish_due_state_and_control(self) -> None:
        self.sensor_publish_accumulator += self.sim_dt
        self.control_publish_accumulator += self.sim_dt
        state_published = False
        if self.sensor_publish_accumulator + 1.0e-12 >= self.sensor_publish_dt:
            self._publish_state()
            self.sensor_publish_accumulator -= self.sensor_publish_dt
            state_published = True
        if self.control_publish_accumulator + 1.0e-12 >= self.control_dt:
            if not state_published:
                self._publish_state()
            self._publish_control_tick()
            self.control_publish_accumulator -= self.control_dt

    def _update_viewer_overlay(self, simulated_dt: float) -> None:
        if self.viewer is None or not self.viewer.is_running():
            return
        now = time.perf_counter()
        self._runtime_window_simulated_s += simulated_dt
        self._runtime_window_frames += 1
        elapsed = now - self._runtime_window_start
        if elapsed >= 0.5:
            measured_realtime = self._runtime_window_simulated_s / elapsed
            measured_fps = self._runtime_window_frames / elapsed
            smoothing = 0.3 if self.actual_realtime > 0.0 else 1.0
            self.actual_realtime += smoothing * (
                measured_realtime - self.actual_realtime
            )
            self.viewer_fps += smoothing * (measured_fps - self.viewer_fps)
            self._runtime_window_start = now
            self._runtime_window_simulated_s = 0.0
            self._runtime_window_frames = 0

        if self._last_overlay_update and now - self._last_overlay_update < 0.25:
            return
        status = "FLYING" if self.ball_flying else "WAITING"
        if self.interactive_perturbation and not self.ball_flying:
            status = "HELD FOR MOUSE"
        if self._startup_support_active:
            status = f"{status} / STARTUP SUPPORT"
        ball_time = (
            self.ball_flight_elapsed_s
            if self.ball_flying
            else self.launch_wait_remaining_s
        )
        target_realtime = "1.00x" if self.realtime else "uncapped"
        left = (
            "FSM\nBall mode\nTrajectory\nFailure ratio\nPhysics step\nBall cycle\nBall time\nStatus\n"
            "Target RT\nActual RT"
        )
        failure_ratio = self.scheduled_failure_count / max(
            self.scheduled_ball_count, 1
        )
        right = (
            f"{self.deployment_fsm.state}\n"
            f"{'INTERACTIVE' if self.interactive_perturbation else 'AUTO'}\n"
            f"{self.pending_ball_trajectory_type}\n"
            f"{self.scheduled_failure_count}/{self.scheduled_ball_count} "
            f"({failure_ratio:.1%})\n"
            f"{self.physics_step_count}\n"
            f"{self.ball_cycle_index}\n"
            f"{ball_time:.3f} s\n"
            f"{status}\n"
            f"{target_realtime}\n"
            f"{self.actual_realtime:.2f}x ({self.viewer_fps:.0f} FPS)"
        )
        self.viewer.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_150.value,
                mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                left,
                right,
            )
        )
        self._last_overlay_update = now

    def _step(self) -> None:
        if self.interactive_perturbation and self._interactive_ball_reset_requested:
            self._interactive_ball_reset_requested = False
            self._reset_interactive_ball()

        if self.interactive_perturbation:
            if (
                not self.ball_flying
                and self._interactive_ball_perturbation_active()
            ):
                self._release_interactive_ball()
        elif not self.ball_flying:
            self.launch_wait_remaining_s -= self.sim_dt
            if self.launch_wait_remaining_s <= 0.0:
                self._launch_pending_ball()

        if self.interactive_perturbation:
            self.data.xfrc_applied.fill(0.0)
            self._apply_interactive_perturbation()
        else:
            self.data.xfrc_applied[self.ball_body_id] = 0.0

        if self.ball_flying:
            self._apply_ball_aerodynamics()
        else:
            self._set_ball_state(self.pending_ball_position_w)

        # Policy ticks and command freshness must share simulation time. A slow
        # viewer must not expire commands while physics is not advancing.
        now = float(self.data.time)
        if self._startup_support_active:
            self._apply_startup_pelvis_support()
            mujoco.mj_forward(self.model, self.data)
        else:
            pelvis_up_z = float(self.data.xmat[self.pelvis_body_id, 8])
            pelvis_tilt = math.acos(np.clip(pelvis_up_z, -1.0, 1.0))
            if pelvis_tilt > self.safety_max_tilt_rad:
                self.deployment_fsm.force_damp(
                    "tilt_limit", now_s=now, hard=True
                )
        self.deployment_fsm.step(
            self.data.qpos[self.joint_qpos_addresses], now_s=now
        )
        if self.deployment_fsm.state == "control":
            self._startup_support_active = False
        self._log_fsm_changes()
        self.data.ctrl[self.actuator_ids] = self._compute_torque()
        mujoco.mj_step(self.model, self.data)
        if self._startup_support_active:
            self._apply_startup_pelvis_support()
            mujoco.mj_forward(self.model, self.data)
        self.physics_step_count += 1
        simulated_dt = self.sim_dt

        if self.ball_flying:
            self.ground_rebound.after_step(self.model, self.data)
            self.ball_flight_elapsed_s += self.sim_dt
            self._update_strike_diagnostics()
            if (
                not self.interactive_perturbation
                and self.ball_flight_elapsed_s >= self.maximum_ball_flight_time_s
            ):
                self._schedule_next_ball()
        else:
            # Keep the sampled ball exactly stationary and visible before launch.
            self._set_ball_state(self.pending_ball_position_w)
            mujoco.mj_forward(self.model, self.data)

        simulation_time = Float64()
        simulation_time.data = float(self.data.time)
        self.simulation_time_pub.publish(simulation_time)
        self._publish_due_state_and_control()

        if self.viewer is not None and self.viewer.is_running():
            self._update_viewer_overlay(simulated_dt)
            viewer_now = time.perf_counter()
            if viewer_now - self._last_viewer_sync >= self._viewer_sync_period_s:
                self._draw_landing_target()
                self.viewer.sync()
                self._last_viewer_sync = viewer_now
        if self.realtime:
            remaining = self.next_step_deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            self.next_step_deadline += self.sim_dt
        if (
            self.duration_s is not None
            and time.perf_counter() - self.start_wall_time >= self.duration_s
        ):
            self.stop_requested = True

    def destroy_node(self) -> None:
        viewer = self.viewer
        self.viewer = None
        if viewer is not None:
            viewer.close()
        super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser(description="TPPO Student MuJoCo backend.")
    parser.add_argument("--config", required=True, help="Deployment YAML path or name.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument(
        "--interactive-perturbation",
        action="store_true",
        help="Enable native mouse perturbation for the tennis ball and G1.",
    )
    args = parser.parse_args()
    rclpy.init()
    node = TppoStudentSimulationNode(
        args.config,
        headless=args.headless,
        realtime=not args.no_realtime,
        duration_s=args.duration_s,
        interactive_perturbation=args.interactive_perturbation,
    )
    native_viewer_used = node.viewer is not None
    try:
        while (
            rclpy.ok()
            and not node.stop_requested
            and (node.viewer is None or node.viewer.is_running())
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if native_viewer_used:
        # Avoid a MuJoCo/GLFW teardown crash after the passive viewer is closed.
        os._exit(0)


if __name__ == "__main__":
    main()
