##
#
# Control node for 29DoF athlete policy.
#
##

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.policy import Policy
from utils.math_utils import (
    quat_conjugate,
    quat_multiply,
    quat_to_rot6d,
    yaw_quat,
    rpy_to_quat
)


############################################################################
# CONTROL NODE
############################################################################


class ControlNode(Node):
    """
    Asynchronous control node that runs the athlete policy and sends
    actions to the simulation.
    """

    def __init__(self, config_path: str):
        super().__init__("control_node")

        self.config = self._load_config(config_path)
        self._init_policy()

        # publishers
        self.command_pub = self.create_publisher(
            Float32MultiArray, "deploy_robot/command", 10
        )
        self.motion_frame_pub = self.create_publisher(
            Float64, "deploy_robot/motion_frame", 10
        )

        # subscribers
        self.create_subscription(
            Float32MultiArray,
            "deploy_robot/pelvis_imu_state",
            self._pelvis_imu_callback,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            "deploy_robot/joint_state",
            self._joint_sensor_callback,
            10,
        )
        self.create_subscription(
            Float64, "deploy_robot/simulation_time", self._time_callback, 10
        )
        self.create_subscription(
            Float32MultiArray, "deploy_robot/goals", self._goal_callback, 10
        )
        self.create_subscription(
            Float32MultiArray,
            "deploy_robot/joystick",
            self._motion_trigger_callback,
            10,
        )
        self.create_subscription(
            Float64, "deploy_robot/which_motion", self._which_motion_callback, 10
        )

        self.create_timer(self.ctrl_dt, self._control_callback)

        # sensor state
        self.anchor_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.pelvis_omega = np.zeros(3, dtype=np.float32)
        self.qpos_joints = self.qpos_joints_default.copy()
        self.qvel_joints = np.zeros_like(self.qpos_joints_default)
        self.sim_time = 0.0

        # policy state
        self.action = np.zeros(self.act_size, dtype=np.float32)
        self.action_triggered = False
        self.motion_idx = 0
        self.goal_targets = np.zeros(self._goal_dim, dtype=np.float32)

        self.init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.policy_start_time = None

        print("Control node initialized.")

    #################################################################
    # INITIALIZATION
    #################################################################

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT.parent / path

    def _load_config(self, config_path: str) -> dict:
        path = Path(config_path)
        if not path.is_absolute():
            candidate = REPO_ROOT / "configs" / config_path
            if candidate.exists():
                path = candidate
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from [{path}].")
        return config

    def _init_policy(self):
        self.qpos_joints_default = np.array(
            self.config["default_joint_pos"], dtype=np.float32
        )
        self.action_scale = np.array(self.config["action_scale"], dtype=np.float32)
        self.Kp = np.array(self.config["Kp"], dtype=np.float32)
        self.Kd = np.array(self.config["Kd"], dtype=np.float32)
        self.ctrl_dt = float(self.config["control_dt"])

        policy_path = self._resolve_path(self.config["policy_path"])
        self.policy = Policy(str(policy_path))
        self.obs_size = self.policy.input_size
        self.act_size = self.policy.output_size

        print(f"Loaded policy from [{policy_path}].")
        print(f"    type={self.policy._policy_type}  in={self.obs_size}  out={self.act_size}")
        print(f"    control_dt={self.ctrl_dt}s  ({1.0/self.ctrl_dt:.0f} Hz)")

        # load motion reference files
        self.motions = []
        for mp in self.config["motion_paths"]:
            path = self._resolve_path(mp)
            m = np.load(str(path))
            entry = {
                "fps": float(m["fps"].item()),
                "joint_pos": m["joint_pos"].astype(np.float32),
                "joint_vel": m["joint_vel"].astype(np.float32),
                "body_quat_w": m["body_quat_w"].astype(np.float32),
                "num_frames": int(m["joint_pos"].shape[0]),
            }
            self.motions.append(entry)
            print(
                f"Loaded motion from [{path}]  "
                f"frames={entry['num_frames']}  fps={entry['fps']}  "
                f"duration={entry['num_frames']/entry['fps']:.1f}s"
            )

        # find anchor body index in the MuJoCo model body list
        anchor_name = self.policy.metadata["anchor_body_name"]
        xml_path = REPO_ROOT.parent / "robots" / self.config["xml_path"]
        mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        motion_body_names = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(1, mj_model.nbody)  # skip world body (id 0)
        ]
        self.anchor_body_idx = motion_body_names.index(anchor_name)

        if "pelvis" in anchor_name.lower():
            self.anchor = "pelvis"
        elif "torso" in anchor_name.lower():
            self.anchor = "torso"
        else:
            raise ValueError(f"Unsupported anchor body name: {anchor_name!r}")

        print(f"    anchor_body={anchor_name} (motion idx {self.anchor_body_idx})")

        # goal vector dimensionality: sum per-type sizes for motion 0
        # position→3, velocity→3, orientation→4
        _type_dim = {"position": 3, "velocity": 3, "orientation": 4}
        goals_cfg = self.config.get("goals", [])
        self._goal_dim = sum(
            _type_dim[g["type"]] for g in goals_cfg if g["motion_index"] == 0
        ) or 3

        # nominal goal positions and orientations per motion (pelvis frame, from config).
        # Used to condition the policy when no motion is in progress so it
        # sees a stable, consistent target rather than a live ball position.
        _pos_by_idx = {
            g["motion_index"]: np.array(g["vector"], dtype=np.float32)
            for g in goals_cfg
            if g["type"] == "position"
        }
        _n = max(_pos_by_idx.keys()) + 1 if _pos_by_idx else 0
        self._nominal_positions = np.array(
            [_pos_by_idx[i] for i in range(_n)], dtype=np.float32
        )

        _ori_by_idx = {
            g["motion_index"]: rpy_to_quat(np.array(g["vector"], dtype=np.float32))
            for g in goals_cfg
            if g["type"] == "orientation"
        }
        _n_ori = max(_ori_by_idx.keys()) + 1 if _ori_by_idx else 0
        self._nominal_orientations = np.array(
            [_ori_by_idx[i] for i in range(_n_ori)], dtype=np.float32
        )

    #################################################################
    # CALLBACKS
    #################################################################

    def _time_callback(self, msg):
        self.sim_time = msg.data

    def _pelvis_imu_callback(self, msg):
        # layout: [rpy(3), quat(4), gyro(3), acc(3)]
        data = np.array(msg.data, dtype=np.float32)
        self.pelvis_omega = data[7:10]
        if self.anchor == "pelvis":
            self.anchor_quat = data[3:7]

    def _joint_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        n = len(self.qpos_joints_default)
        self.qpos_joints = data[:n]
        self.qvel_joints = data[n : 2 * n]

    def _goal_callback(self, msg):
        self.goal_targets = np.array(msg.data, dtype=np.float32)

    def _motion_trigger_callback(self, msg):
        # trigger on B button (data[4])
        if not self.action_triggered and len(msg.data) > 4 and msg.data[4] > 0.5:
            self.action_triggered = True
            self.policy_start_time = self.sim_time

    def _which_motion_callback(self, msg):
        if not self.action_triggered:
            self.motion_idx = int(msg.data)

    #################################################################
    # OBSERVATION
    #################################################################

    def _build_observation(self):
        motion = self.motions[self.motion_idx]

        if self.action_triggered:
            elapsed = self.sim_time - self.policy_start_time
            frame = int(elapsed / self.ctrl_dt)
            if frame >= motion["num_frames"] - 1:
                frame = motion["num_frames"] - 1
                self.action_triggered = False
        else:
            frame = 0

        frame_msg = Float64()
        frame_msg.data = float(frame)
        self.motion_frame_pub.publish(frame_msg)

        # command: joint_pos_ref + joint_vel_ref + goal_targets
        # When no motion is running, substitute the nominal target position for the
        # chosen motion so the policy is conditioned on a stable reference rather
        # than whatever live ball position was last received.
        if not self.action_triggered and self.motion_idx < len(self._nominal_positions):
            goal_targets = self.goal_targets.copy()
            goal_targets[:3] = self._nominal_positions[self.motion_idx]
        else:
            goal_targets = self.goal_targets
        command = np.concatenate(
            [motion["joint_pos"][frame], motion["joint_vel"][frame], goal_targets]
        )
        print(f"goal targets: {goal_targets}")

        # motion_anchor_ori_b: desired anchor orientation in base frame (6D rotation)
        motion_anchor_quat_w = motion["body_quat_w"][frame, self.anchor_body_idx]
        ref_quat_corrected = quat_multiply(self.init_quat, motion_anchor_quat_w)
        rel_quat = quat_multiply(quat_conjugate(self.anchor_quat), ref_quat_corrected)
        anchor_ori_b = quat_to_rot6d(rel_quat)

        # base_ang_vel: pelvis angular velocity from IMU
        base_ang_vel_b = self.pelvis_omega

        # joint_pos: relative to default
        qj = self.qpos_joints - self.qpos_joints_default

        # joint_vel
        dqj = self.qvel_joints

        obs = np.concatenate(
            [command, anchor_ori_b, base_ang_vel_b, qj, dqj, self.action]
        ).astype(np.float32)

        return obs, frame

    #################################################################
    # CONTROL
    #################################################################

    def _control_callback(self):
        # on the very first tick, capture the yaw alignment between current
        # robot heading and motion frame 0
        if self.policy_start_time is None:
            self.policy_start_time = self.sim_time
            motion_anchor_quat_0 = self.motions[self.motion_idx]["body_quat_w"][
                0, self.anchor_body_idx
            ]
            self.init_quat = quat_multiply(
                yaw_quat(self.anchor_quat),
                quat_conjugate(yaw_quat(motion_anchor_quat_0)),
            )

        obs, frame = self._build_observation()
        self.action = self.policy.inference(obs, time_step=frame)

        qpos_des = self.action * self.action_scale + self.qpos_joints_default
        qvel_des = np.zeros(self.act_size, dtype=np.float32)
        tau_ff = np.zeros(self.act_size, dtype=np.float32)

        cmd_msg = Float32MultiArray()
        cmd_msg.data = np.concatenate(
            [qpos_des, qvel_des, self.Kp, self.Kd, tau_ff]
        ).tolist()
        self.command_pub.publish(cmd_msg)


############################################################################
# MAIN
############################################################################


def main(args=None):
    rclpy.init()

    parser = argparse.ArgumentParser(description="ATHLETE control node.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    args = parser.parse_args()

    ctrl_node = ControlNode(args.config)

    try:
        rclpy.spin(ctrl_node)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
