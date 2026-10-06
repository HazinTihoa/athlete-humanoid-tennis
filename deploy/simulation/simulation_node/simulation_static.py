##
#
# Simulation node using Mujoco to simulate the robot.
#
##

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.math_utils import (
    quat_conjugate,
    quat_multiply,
    quat_to_rotation_matrix,
    quat_to_rpy,
    rpy_to_quat
)

# Motion names for display — kept in sync with the config motion order.
_MOTION_NAMES = [
    "right_kick",
    "left_kick",
    "two_step_right_kick",
    "two_step_left_kick",
    "two_baby_step_right_kick",
]
_BALL_SPEED = 1.0     # m/s for joystick positioning
_BALL_X_LIMIT = 2.0   # world frame absolute X (depth) limit (m)
_BALL_Y_LIMIT = 2.5   # world frame absolute Y (lateral) limit (m)
_BALL_Z_MIN = 0.0     # world frame Z floor (m)
_BALL_Z_MAX = 2.0     # world frame Z ceiling (m)
# Only switch motion when the new nominal is this much closer than the current one.
# Prevents rapid oscillation when the ball sits near a boundary between two targets.
_MOTION_SWITCH_HYSTERESIS = 0.05  # metres


############################################################################
# SIMULATION NODE
############################################################################


class SimulationNode(Node):
    """
    Asynchronous simulation node that runs the Mujoco simulation.
    """

    def __init__(self, config_path: str, apply_noise: bool = False):
        super().__init__("simulation_node")

        self.config = self._load_config(config_path)
        self.apply_noise = apply_noise

        self._init_params()
        self._init_simulation()

        self.motion_idx = 0

        # Build nominal positions (pelvis frame) directly from config position goals,
        # indexed by motion_index, so they always match the configured motions.
        _pos_by_idx = {
            g["motion_index"]: np.array(g["vector"], dtype=np.float32)
            for g in self.config.get("goals", [])
            if g["type"] == "position"
        }
        n_motions = max(_pos_by_idx.keys()) + 1 if _pos_by_idx else 0
        self._nominal_positions = np.array(
            [_pos_by_idx[i] for i in range(n_motions)], dtype=np.float32
        )
        print(f"Motion nominal positions ({n_motions} motions): {self._nominal_positions[:, 1].tolist()}")

        # Hit-position ball in world frame — fixed in space, not attached to the robot.
        # Initialised at the first motion's nominal position in the robot's starting pose.
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        _pelvis_quat0 = self.mj_data.body("pelvis").xquat.astype(np.float32)
        _R0 = quat_to_rotation_matrix(_pelvis_quat0)
        _p0 = self.mj_data.body("pelvis").xpos.astype(np.float32)
        self._ball_pos_w = _R0 @ self._nominal_positions[0] + _p0
        # Snapshot the startup pelvis quat once.  _init_goals() uses this to
        # express orientation targets in world frame — so re-calling _init_goals()
        # on a motion switch never changes the stored world-frame target, avoiding
        # the orientation discontinuity that caused backward stumbling on return.
        self._init_pelvis_quat = _pelvis_quat0.copy()
        self._last_joystick_t: float | None = None
        self._init_goals()

        # publishers
        self.pelvis_imu_state_pub = self.create_publisher(
            Float32MultiArray, "deploy_robot/pelvis_imu_state", 10
        )
        self.joint_state_pub = self.create_publisher(
            Float32MultiArray, "deploy_robot/joint_state", 10
        )
        self.simulation_time_pub = self.create_publisher(
            Float64, "deploy_robot/simulation_time", 10
        )
        self.goal_pub = self.create_publisher(Float32MultiArray, "deploy_robot/goals", 10)
        self.which_motion_pub = self.create_publisher(
            Float64, "deploy_robot/which_motion", 10
        )

        # subscribers
        self.create_subscription(
            Float32MultiArray, "deploy_robot/command", self._command_callback, 10
        )
        self.create_subscription(
            Float64, "deploy_robot/motion_frame", self._motion_frame_callback, 10
        )
        self.create_subscription(
            Float32MultiArray, "deploy_robot/joystick", self._which_motion_callback, 10
        )

        self.command_received = False
        self.qpos_des = np.zeros(self.nu)
        self.qvel_des = np.zeros(self.nu)
        self.tau_ff = np.zeros(self.nu)
        self.Kp = np.zeros(self.nu)
        self.Kd = np.zeros(self.nu)
        self.motion_frame = 0
        self._motion_in_progress = False

        self.create_timer(0.0, self._step_simulation)
        self.create_timer(self.sim_dt, self._publish_pelvis_imu)
        self.create_timer(self.sim_dt, self._publish_joint_state)
        self.create_timer(self.sim_dt, self._publish_goals)

        print("Simulation node initialized.")
        print("    Press [Tab] to toggle the left UI.")
        print("    Press [Shift + Tab] to toggle the right UI.")

    #################################################################
    # INITIALIZATION
    #################################################################

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT.parent / path

    def _load_config(self, config_path: str) -> dict:
        path = Path(config_path)
        if not path.is_absolute():
            # try relative to deploy/configs/ first
            candidate = REPO_ROOT / "configs" / config_path
            if candidate.exists():
                path = candidate
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from [{path}].")
        return config

    def _init_params(self):
        self.default_base = np.array(self.config["default_base_pos"])
        self.default_joints = np.array(self.config["default_joint_pos"])

    def _init_simulation(self):
        xml_path = REPO_ROOT.parent / "robots" / self.config["xml_path"]

        self.mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.mj_data = mujoco.MjData(self.mj_model)

        self.nq = self.mj_model.nq
        self.nv = self.mj_model.nv
        self.nu = self.mj_model.nu
        self.sim_dt = self.mj_model.opt.timestep

        assert len(self.default_joints) == self.nu, (
            f"default_joint_pos must have {self.nu} entries, got {len(self.default_joints)}."
        )

        self.mj_data.qpos[:7] = self.default_base
        self.mj_data.qpos[7 : 7 + self.nu] = self.default_joints

        # build per-joint sensor name lists (matching actuator order)
        self.joint_pos_sensor_names = []
        self.joint_vel_sensor_names = []
        for i in range(self.nu):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self.joint_pos_sensor_names.append(f"{joint_name}_pos_sensor")
            self.joint_vel_sensor_names.append(f"{joint_name}_vel_sensor")

        print(f"Loaded Mujoco model from [{xml_path}].")
        print(f"    sim_dt={self.sim_dt}s  nq={self.nq}  nv={self.nv}  nu={self.nu}")

        self.viewer = mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
        )

        self._viewer_font_scale = getattr(
            mujoco.mjtFontScale,
            "mjFONTSCALE_250",
            getattr(
                mujoco.mjtFontScale,
                "mjFONTSCALE_200",
                mujoco.mjtFontScale.mjFONTSCALE_150,
            ),
        )

        self.viewer.cam.azimuth = 135
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 2.5
        self.viewer.cam.lookat[:] = list(self.default_base[:3])

        self.viewer_render_hz = 50.0
        self._last_viewer_sync = 0.0
        self._real_start_time = time.perf_counter()
        self._next_step_deadline = self._real_start_time + self.sim_dt

    def _init_goals(self):
        goals_cfg = self.config.get("goals", [])

        self._goal_types: list[str] = []
        self._goal_vel_w: list[np.ndarray] = []
        # _goal_quat_w stores orientation targets in world frame.  We use the
        # STARTUP pelvis quat (self._init_pelvis_quat, always identity here) rather
        # than the current pelvis quat, so re-calling this on a motion switch never
        # changes the stored world-frame target and causes no orientation discontinuity.
        self._goal_quat_w: list[np.ndarray] = []

        for goal in [g for g in goals_cfg if g["motion_index"] == self.motion_idx]:
            vec = np.array(goal["vector"], dtype=np.float32)
            goal_type = goal["type"]
            self._goal_types.append(goal_type)
            if goal_type == "position":
                self._goal_vel_w.append(np.zeros(3, dtype=np.float32))
                self._goal_quat_w.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            elif goal_type == "velocity":
                self._goal_vel_w.append(vec)
                self._goal_quat_w.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            elif goal_type == "orientation":
                self._goal_vel_w.append(np.zeros(3, dtype=np.float32))
                # Use the startup pelvis quat — stable regardless of when this is called.
                self._goal_quat_w.append(quat_multiply(self._init_pelvis_quat, rpy_to_quat(vec)))
            else:
                raise ValueError(f"Unsupported goal type: {goal_type!r}")

        names = [g["name"] for g in goals_cfg if g["motion_index"] == self.motion_idx]
        print(f"Goals initialized for motion {self.motion_idx}: {names}")

    #################################################################
    # CALLBACKS
    #################################################################

    def _command_callback(self, msg):
        data = np.array(msg.data)
        self.command_received = True
        self.qpos_des = data[0 * self.nu : 1 * self.nu]
        self.qvel_des = data[1 * self.nu : 2 * self.nu]
        self.Kp = data[2 * self.nu : 3 * self.nu]
        self.Kd = data[3 * self.nu : 4 * self.nu]
        self.tau_ff = data[4 * self.nu : 5 * self.nu]

    def _motion_frame_callback(self, msg):
        new_frame = int(msg.data)
        if self._motion_in_progress and self.motion_frame > 0 and new_frame == 0:
            self._motion_in_progress = False
        self.motion_frame = new_frame

    def _which_motion_callback(self, msg):
        # B button (data[4]) triggers a motion — lock until it completes.
        if len(msg.data) > 4 and float(msg.data[4]) > 0.5:
            self._motion_in_progress = True

        # Joystick layout: [LS_X, LS_Y, RS_X, RS_Y, B]
        # LS_Y → X (depth), RS_X → Y (lateral), RS_Y → Z (height)
        x_axis = float(msg.data[1]) if len(msg.data) > 1 else 0.0
        y_axis = float(msg.data[2]) if len(msg.data) > 2 else 0.0
        z_axis = float(msg.data[3]) if len(msg.data) > 3 else 0.0

        now = time.monotonic()
        dt = (now - self._last_joystick_t) if self._last_joystick_t is not None else 0.02
        self._last_joystick_t = now

        # Move ball in world frame — stays fixed when the robot moves.
        self._ball_pos_w[0] = float(np.clip(
            self._ball_pos_w[0] + x_axis * _BALL_SPEED * dt,
            -_BALL_X_LIMIT, _BALL_X_LIMIT,
        ))
        self._ball_pos_w[1] = float(np.clip(
            self._ball_pos_w[1] + y_axis * _BALL_SPEED * dt,
            -_BALL_Y_LIMIT, _BALL_Y_LIMIT,
        ))
        self._ball_pos_w[2] = float(np.clip(
            self._ball_pos_w[2] + z_axis * _BALL_SPEED * dt,
            _BALL_Z_MIN, _BALL_Z_MAX,
        ))

        if self._motion_in_progress:
            return

        # Convert to pelvis frame to compare against nominal positions.
        pelvis_pos = self.mj_data.body("pelvis").xpos.astype(np.float32)
        R = quat_to_rotation_matrix(self.mj_data.body("pelvis").xquat.astype(np.float32))
        ball_in_pelvis = R.T @ (self._ball_pos_w - pelvis_pos)
        dists = np.linalg.norm(self._nominal_positions - ball_in_pelvis, axis=1)
        new_idx = int(np.argmin(dists))
        # Hysteresis: only commit to a different motion when the candidate is
        # meaningfully closer than the current one.  Without this, the argmin
        # oscillates as the ball crosses a nominal boundary, calling _init_goals()
        # dozens of times per second and making the policy observation unstable.
        if (
            new_idx != self.motion_idx
            and dists[new_idx] + _MOTION_SWITCH_HYSTERESIS < dists[self.motion_idx]
        ):
            self.motion_idx = new_idx
            self._init_goals()

    #################################################################
    # PUBLISHING
    #################################################################

    def _publish_pelvis_imu(self):
        pelvis_quat = self.mj_data.sensor("pelvis_imu_quat_sensor").data.copy()
        pelvis_gyro = self.mj_data.sensor("pelvis_imu_gyro_sensor").data.copy()
        pelvis_acc = self.mj_data.sensor("pelvis_imu_acc_sensor").data.copy()
        pelvis_rpy = quat_to_rpy(pelvis_quat)

        msg = Float32MultiArray()
        msg.data = np.concatenate(
            [pelvis_rpy, pelvis_quat, pelvis_gyro, pelvis_acc]
        ).tolist()
        self.pelvis_imu_state_pub.publish(msg)

    def _publish_joint_state(self):
        qpos_joints = np.array(
            [self.mj_data.sensor(n).data[0] for n in self.joint_pos_sensor_names]
        )
        qvel_joints = np.array(
            [self.mj_data.sensor(n).data[0] for n in self.joint_vel_sensor_names]
        )
        ddq_joints = np.zeros(self.nu)
        tau_est_joints = self.mj_data.ctrl[: self.nu].copy()

        msg = Float32MultiArray()
        msg.data = np.concatenate(
            [qpos_joints, qvel_joints, ddq_joints, tau_est_joints]
        ).tolist()
        self.joint_state_pub.publish(msg)

    def _publish_goals(self):
        which_motion_msg = Float64()
        which_motion_msg.data = float(self.motion_idx)
        self.which_motion_pub.publish(which_motion_msg)

        if not self._goal_types:
            return

        pelvis_quat = self.mj_data.body("pelvis").xquat.astype(np.float32)
        pelvis_pos = self.mj_data.body("pelvis").xpos.astype(np.float32)
        R = quat_to_rotation_matrix(pelvis_quat)
        pelvis_quat_inv = quat_conjugate(pelvis_quat)

        goal_vecs = []
        for goal_type, vel_w, quat_w in zip(
            self._goal_types, self._goal_vel_w, self._goal_quat_w
        ):
            if goal_type == "position":
                # if not self.action_triggered and self.motion_idx < len(self._nominal_positions):
                #     # When no action has been triggered, keep the goal anchored to the nominal position
                #     # for the current motion, rather than the ball.  This lets the policy sit still
                #     # without drifting when the ball is moved around before triggering.
                #     goal_vecs.append(self._nominal_positions[self.motion_idx])
                # else:
                # Express the world-frame ball position in current pelvis frame.
                goal_vecs.append(R.T @ (self._ball_pos_w - pelvis_pos))
            elif goal_type == "velocity":
                goal_vecs.append(vel_w)
            elif goal_type == "orientation":
                # quat_w is anchored to the startup pelvis pose (identity), so this
                # always gives the same heading-relative orientation signal the policy
                # was trained on — no jump when _init_goals() is re-called mid-run.
                goal_vecs.append(quat_multiply(pelvis_quat_inv, quat_w))

        msg = Float32MultiArray()
        msg.data = np.concatenate(goal_vecs).tolist()
        self.goal_pub.publish(msg)

    #################################################################
    # VISUALIZATION
    #################################################################

    def _update_ball_viz(self) -> None:
        """Draw the hit-position ball as a green sphere in the viewer user scene."""
        ball_world = self._ball_pos_w.astype(np.float64)

        scn = self.viewer.user_scn
        scn.ngeom = 0
        mujoco.mjv_initGeom(
            scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.06, 0.0, 0.0], dtype=np.float64),
            ball_world,
            np.eye(3, dtype=np.float64).flatten(),
            np.array([0.0, 1.0, 0.0, 0.8], dtype=np.float64),
        )
        scn.ngeom = 1

    #################################################################
    # SIMULATION
    #################################################################

    def _apply_sensor_noise(self):
        for i in range(self.mj_model.nsensor):
            std = self.mj_model.sensor_noise[i]
            if std <= 0.0:
                continue
            adr = self.mj_model.sensor_adr[i]
            dim = self.mj_model.sensor_dim[i]
            self.mj_data.sensordata[adr : adr + dim] += np.random.normal(
                0.0, std, size=dim
            )

    def _compute_torque(self):
        qpos_joints = self.mj_data.qpos[7 : 7 + self.nu]
        qvel_joints = self.mj_data.qvel[6 : 6 + self.nu]
        return (
            self.Kp * (self.qpos_des - qpos_joints)
            + self.Kd * (self.qvel_des - qvel_joints)
            + self.tau_ff
        )

    def _step_simulation(self):
        if self.command_received:
            self.mj_data.ctrl[:] = self._compute_torque()
        else:
            self.mj_data.ctrl[:] = 0.0

        mujoco.mj_step(self.mj_model, self.mj_data)

        if self.apply_noise:
            self._apply_sensor_noise()

        time_msg = Float64()
        time_msg.data = self.mj_data.time
        self.simulation_time_pub.publish(time_msg)

        now = time.perf_counter()
        if (
            self.viewer.is_running()
            and (now - self._last_viewer_sync) >= 1.0 / self.viewer_render_hz
        ):
            self._update_ball_viz()
            self.viewer.sync()
            real_elapsed = now - self._real_start_time
            motion_name = (
                _MOTION_NAMES[self.motion_idx]
                if self.motion_idx < len(_MOTION_NAMES)
                else str(self.motion_idx)
            )
            self.viewer.set_texts(
                (
                    self._viewer_font_scale,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    (
                        f"Sim time:   {self.mj_data.time:.2f}s\n"
                        f"Real time: {real_elapsed:.2f}s\n"
                        f"Motion: {motion_name}\n"
                        f"Ball x={self._ball_pos_w[0]:.2f}  y={self._ball_pos_w[1]:.2f}  z={self._ball_pos_w[2]:.2f}"
                    ),
                    "",
                )
            )
            self._last_viewer_sync = now

        remaining = self._next_step_deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        self._next_step_deadline += self.sim_dt

    def destroy_node(self):
        if self.viewer.is_running():
            self.viewer.close()
        super().destroy_node()


############################################################################
# MAIN
############################################################################


def main(args=None):
    rclpy.init()

    parser = argparse.ArgumentParser(description="ATHLETE simulation node.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--noise", action="store_true", help="Enable sensor noise.")
    args = parser.parse_args()

    sim_node = SimulationNode(args.config, apply_noise=args.noise)

    try:
        while rclpy.ok() and sim_node.viewer.is_running():
            rclpy.spin_once(sim_node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        sim_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print("Simulation shutdown complete.")


if __name__ == "__main__":
    main()
