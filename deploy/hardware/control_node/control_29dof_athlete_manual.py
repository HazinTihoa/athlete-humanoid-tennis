##
#
# Control node for 29DoF MjLab mimic tracking.
#
##


# standard imports
import argparse

# other imports
import mujoco
import numpy as np
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray, String

# directory imports
import sys
import os
from pathlib import Path

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

DEPLOY_DIR = str(Path(__file__).resolve().parents[1])
REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(DEPLOY_DIR)

# custom imports
from utils.policy import Policy
from utils.math_utils import (
  quat_conjugate,
  quat_multiply,
  quat_to_rot6d,
  yaw_quat,
)


############################################################################
# CONTROLLER NODE
############################################################################


class ControlNode(Node):
  """
  Asynchronous control node that runs the mimic policy and sends actions to the simulation.
  """

  def __init__(self, config_path: str):
    super().__init__("control_node")

    # load config file
    self.config = self.load_config(config_path)

    # load params
    self.init_policy()

    # ROS publishers
    self.command_pub = self.create_publisher(
      Float32MultiArray, "deploy_robot/command", 10
    )

    self.motion_frame_pub = self.create_publisher(
      Float64, "deploy_robot/motion_frame", 10
    )

    # ROS subscribers
    self.pelvis_imu_sub = self.create_subscription(
      Float32MultiArray, "deploy_robot/pelvis_imu_state", self.pelvis_imu_callback, 10
    )
    self.joint_sensor_sub = self.create_subscription(
      Float32MultiArray, "deploy_robot/joint_state", self.joint_sensor_callback, 10
    )
    self.fsm_sub = self.create_subscription(
      String, "deploy_robot/fsm", self.fsm_callback, 10
    )
    self.fsm_time_sub = self.create_subscription(
      Float64, "deploy_robot/fsm_time", self.time_callback, 10
    )

    # athlete subscribers
    self.goal_sub = self.create_subscription(
      Float32MultiArray, "deploy_robot/goals", self.goal_callback, 10
    )
    self.motion_trigger_sub = self.create_subscription(
      Float32MultiArray, "deploy_robot/joystick", self.motion_trigger_callback, 10
    )
    self.which_motion_sub = self.create_subscription(
      Float64, "deploy_robot/which_motion", self.which_motion_callback, 10
    )
    # control timer to run the policy at a fixed frequency
    self.control_timer = self.create_timer(self.ctrl_dt, self.control_callback)

    # sensor state
    self.anchor_quat = np.array(
      [1.0, 0.0, 0.0, 0.0], dtype=np.float32
    )  # (w, x, y, z) from anchor IMU
    self.pelvis_omega = np.zeros(
      3, dtype=np.float32
    )  # base_ang_vel, always from pelvis IMU
    self.qpos_joints = np.array(self.qpos_joints_default.copy())
    self.qvel_joints = np.zeros_like(self.qpos_joints_default)
    self.fsm_state = "init"
    self.fsm_time = 0.0

    # initialize the action
    self.action = np.zeros(self.act_size)
    self.action_triggered = False
    self.target_time = -1.0
    self.motion_idx = 0

    # initialize goal targets — orientation is published as a quaternion (4) not the raw 3-vec
    goal_dim = 3  # TODO this is hardcoded for now but should be set from config?
    self.goal_targets = np.zeros(goal_dim, dtype=np.float32)

    # yaw alignment between robot-at-policy-start and motion frame 0 (re-captured each time FSM enters "control")
    self.init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    self.init_quat_captured = False

    print("Control node initialized.")

  #################################################################
  # INITIALIZATION
  #################################################################

  # load the config file
  def load_config(self, config_path: str):
    # open the config file and load it
    config_path_full = DEPLOY_DIR + "/configs/" + config_path
    with open(config_path_full, "r") as f:
      config = yaml.safe_load(f)

    print(f"Loaded config from [{config_path_full}].")

    return config

  # initialize the policy
  def init_policy(self):
    # default joint positions
    self.qpos_joints_default = np.array(self.config["default_joint_pos"])

    # scaling params
    self.action_scale = np.array(self.config["action_scale"], dtype=np.float32)

    # PD gains
    self.Kp = np.array(self.config["Kp"], dtype=np.float32)
    self.Kd = np.array(self.config["Kd"], dtype=np.float32)

    # control frequency
    self.ctrl_dt = self.config["control_dt"]

    # import the policy
    policy_path = self.config["policy_path"]
    policy_path_full = policy_path if os.path.isabs(policy_path) else os.path.join(ROOT_DIR, "policy", policy_path)

    # load the policy
    self.policy = Policy(policy_path_full)

    # alias for convenience
    self.obs_size = self.policy.input_size
    self.act_size = self.policy.output_size

    print(f"Loading policy from [{policy_path_full}].")
    print(f"    Policy type: {self.policy._policy_type}")
    print(f"    Input size: {self.obs_size}")
    print(f"    Output size: {self.act_size}")
    print(f"    Control frequency: {1.0 / self.ctrl_dt} Hz")

    # load motion reference data
    contact_phase_cfg = self.config["contact_phase"]
    contact_phases = (
      contact_phase_cfg if isinstance(contact_phase_cfg, list) else [contact_phase_cfg]
    )
    self.motions = []
    self.time_to_contact = []
    for i, mp in enumerate(self.config["motion_paths"]):
      path = mp if os.path.isabs(mp) else os.path.join(REPO_ROOT, mp)
      m = np.load(path)
      num_frames = m["joint_pos"].shape[0]
      entry = {
        "fps": float(m["fps"]),
        "joint_pos": m["joint_pos"].astype(np.float32),
        "joint_vel": m["joint_vel"].astype(np.float32),
        "body_quat_w": m["body_quat_w"].astype(np.float32),
        "num_frames": num_frames,
      }
      self.motions.append(entry)
      ttc = contact_phases[i] * num_frames * self.ctrl_dt
      self.time_to_contact.append(ttc)
      print(f"Loaded motion from [{path}].")
      print(f"    FPS: {entry['fps']}")
      print(f"    Frames: {num_frames}")
      print(f"    Duration: {num_frames / entry['fps']:.1f}s")
      print(f"    Time to contact: {ttc:.3f}s")

    # find anchor body index against robot's full body list
    anchor_name = self.policy.metadata["anchor_body_name"]
    xml_path_cfg = self.config["xml_path"]
    xml_path = xml_path_cfg if os.path.isabs(xml_path_cfg) else os.path.join(REPO_ROOT, "robots", xml_path_cfg)
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    motion_body_names = [
      mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
      for i in range(1, mj_model.nbody)  # skip world (id 0)
    ]
    self.anchor_body_idx = motion_body_names.index(anchor_name)

    # select IMU based on anchor body
    if "pelvis" in anchor_name.lower():
      self.anchor = "pelvis"
    elif "torso" in anchor_name.lower():
      self.anchor = "torso"
    else:
      raise ValueError(f"Unsupported anchor body name: {anchor_name}")

    print(f"    Anchor body: {anchor_name} (index {self.anchor_body_idx})")

  #################################################################
  # CALLBACKS
  #################################################################

  # FSM state
  def fsm_callback(self, msg):
    self.fsm_state = msg.data

  # fsm time — only count time when in "control" state
  def time_callback(self, msg):
    if self.fsm_state == "control":
      self.fsm_time = msg.data
    else:
      self.fsm_time = 0.0

  # pelvis IMU: [rpy(3), quat(4), gyro(3), acc(3)] — base_ang_vel plus anchor_quat when anchor = pelvis
  def pelvis_imu_callback(self, msg):
    data = np.array(msg.data, dtype=np.float32)
    self.pelvis_omega = data[7:10]
    if self.anchor == "pelvis":
      self.anchor_quat = data[3:7]

  # joint data: [q(29), dq(29), ddq(29), tau_est(29)] — we only need q and dq
  def joint_sensor_callback(self, msg):
    data = np.array(msg.data, dtype=np.float32)
    n = len(self.qpos_joints_default)
    self.qpos_joints = data[:n]
    self.qvel_joints = data[n : 2 * n]

  # goal callback
  def goal_callback(self, msg):
    data = np.array(msg.data, dtype=np.float32)
    # The hardware node publishes an empty goals message until it has received
    # a pelvis pose from perception (_goals_initialized). Ignore goals whose
    # length does not match the expected goal dimension so we keep the sane
    # default and the observation stays the size the policy expects.
    if data.shape[0] != self.goal_targets.shape[0]:
      return
    self.goal_targets = data

  # motion trigger callback
  def motion_trigger_callback(self, msg):
    if not self.action_triggered and msg.data[1] > 0.5:
      self.action_triggered = True
      self.policy_start_time = self.fsm_time

  def which_motion_callback(self, msg: Float64):
    if not self.action_triggered:
      self.motion_idx = int(msg.data)

  #################################################################
  # OBSERVATION
  #################################################################

  # build the observation vector for the policy
  # ['command', 'motion_anchor_ori_b', 'base_ang_vel', 'joint_pos', 'joint_vel', 'actions']
  def build_observation(self):
    motion = self.motions[self.motion_idx]

    # motion frame: 1 frame per control_dt, matching training (time_steps += 1 per step_dt)
    # fsm_time is reset to 0 when the FSM is not in "control", so it's already relative to policy start
    if self.action_triggered:
      elapsed = self.fsm_time - self.policy_start_time
      frame = int(elapsed / self.ctrl_dt)
      if frame >= motion["num_frames"] - 2:
        self.action_triggered = False
    else:
      frame = 0

    # publish the motion frame
    frame_msg = Float64()
    frame_msg.data = float(frame)
    self.motion_frame_pub.publish(frame_msg)

    # --- command (58) : motion reference joint_pos + joint_vel ---
    command = np.concatenate(
      [
        motion["joint_pos"][frame],
        motion["joint_vel"][frame],
        self.goal_targets,
      ]
    )
    print(f"goal targets: {self.goal_targets}")

    # --- motion_anchor_ori_b (6) : desired anchor orientation in base frame (6D rotation) ---
    # apply the captured yaw offset so the motion is replayed in the robot's initial heading
    motion_anchor_quat_w = motion["body_quat_w"][frame, self.anchor_body_idx]
    ref_quat_corrected = quat_multiply(self.init_quat, motion_anchor_quat_w)
    rel_quat = quat_multiply(quat_conjugate(self.anchor_quat), ref_quat_corrected)
    anchor_ori_b = quat_to_rot6d(rel_quat)

    # --- base_ang_vel (3) : pelvis angular velocity (training uses imu_in_pelvis site) ---
    base_ang_vel_b = self.pelvis_omega

    # --- joint_pos (29) : relative to default ---
    qj = self.qpos_joints - self.qpos_joints_default

    # --- joint_vel (29) ---
    dqj = self.qvel_joints

    # --- actions (29) : previous action ---
    # concatenate: 58 + 6 + 3 + 29 + 29 + 29 = 154
    obs = np.concatenate(
      [command, anchor_ori_b, base_ang_vel_b, qj, dqj, self.action]
    ).astype(np.float32)

    return obs, frame

  #################################################################
  # CONTROL
  #################################################################

  # control published at the control frequency
  def control_callback(self):
    # only run policy when in "control" state
    if self.fsm_state != "control":
      self.action = np.zeros(self.act_size)
      self.init_quat_captured = False
      return

    # on the first control tick, align motion frame 0 with the robot's current yaw
    if not self.init_quat_captured:
      motion_anchor_quat_0 = self.motions[self.motion_idx]["body_quat_w"][
        0, self.anchor_body_idx
      ]
      self.init_quat = quat_multiply(
        yaw_quat(self.anchor_quat),
        quat_conjugate(yaw_quat(motion_anchor_quat_0)),
      )
      self.init_quat_captured = True

    # get the current observation and motion frame index
    obs, frame = self.build_observation()

    # target joint positions (PD control)
    self.action = self.policy.inference(obs, time_step=frame)

    # build the command: [qpos_des, qvel_des, Kp, Kd, tau_ff]
    qpos_des = self.action * self.action_scale + self.qpos_joints_default
    qvel_des = np.zeros(self.act_size, dtype=np.float32)
    tau_ff = np.zeros(self.act_size, dtype=np.float32)

    # publish the command
    cmd_msg = Float32MultiArray()
    cmd_msg.data = np.concatenate(
      [qpos_des, qvel_des, self.Kp, self.Kd, tau_ff]
    ).tolist()
    self.command_pub.publish(cmd_msg)


############################################################################
# MAIN FUNCTION
############################################################################


def main(args=None):
  # init ROS2
  rclpy.init()

  # parse arguments
  parser = argparse.ArgumentParser(
    description="Asynchronous Control Node for MjLab Mimic Policy."
  )
  # config path argument
  parser.add_argument(
    "--config",
    type=str,
    required=True,
    help='Path to the config yaml file. Example: "g1_29dof_mimic.yaml".',
  )
  args = parser.parse_args()

  # create the simulation node
  ctrl_node = ControlNode(args.config)

  # execute the policy
  try:
    # spin the node
    rclpy.spin(ctrl_node)

  except KeyboardInterrupt:
    pass

  finally:
    # close everything
    ctrl_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()
