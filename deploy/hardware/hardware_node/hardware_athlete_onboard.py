##
#
# Deployment code for Unitree G1 robot.
#
# Common G1 "hg" topics: https://support.unitree.com/home/en/G1_developer/dds_services_interface
#
##

# standard imports
import argparse
import numpy as np
import time
import threading

# other imports
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray, String
from geometry_msgs.msg import PoseStamped

# directory imports
import os
import sys
from pathlib import Path

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

DEPLOY_DIR = str(Path(__file__).resolve().parents[1])
REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(DEPLOY_DIR)

# custom imports
from utils.math_utils import (
  quat_to_rotation_matrix,
  quat_conjugate,
  quat_multiply,
  rpy_to_quat,
)

# Unitree SDK imports (C++ wrapper: submodules/unitree_sdk2_wrapper)
import unitree_interface
from utils.unitree_utils import RecurrentThread


########################################################################
# GLOBAL VARIABLES (DO NOT CHANGE)
########################################################################

G1_NUM_MOTOR = 29


class G1JointIndex:
  LeftHipPitch = 0
  LeftHipRoll = 1
  LeftHipYaw = 2
  LeftKnee = 3
  LeftAnklePitch = 4
  LeftAnkleB = 4
  LeftAnkleRoll = 5
  LeftAnkleA = 5
  RightHipPitch = 6
  RightHipRoll = 7
  RightHipYaw = 8
  RightKnee = 9
  RightAnklePitch = 10
  RightAnkleB = 10
  RightAnkleRoll = 11
  RightAnkleA = 11
  WaistYaw = 12
  WaistRoll = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
  WaistA = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
  WaistPitch = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked
  WaistB = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked
  LeftShoulderPitch = 15
  LeftShoulderRoll = 16
  LeftShoulderYaw = 17
  LeftElbow = 18
  LeftWristRoll = 19
  LeftWristPitch = 20  # NOTE: INVALID for g1 23dof
  LeftWristYaw = 21  # NOTE: INVALID for g1 23dof
  RightShoulderPitch = 22
  RightShoulderRoll = 23
  RightShoulderYaw = 24
  RightElbow = 25
  RightWristRoll = 26
  RightWristPitch = 27  # NOTE: INVALID for g1 23dof
  RightWristYaw = 28  # NOTE: INVALID for g1 23dof


class Mode:
  PR = 0  # Series Control for Pitch/Roll Joints
  AB = 1  # Parallel Control for A/B Joints


# low-level control frequency
LOW_LEVEL_CONTROL_DT = 0.002  # [sec]

# ROS2 sensor publishing frequency
ROS_SENSOR_PUBLISH_DT = 0.01  # [sec]

# safety: max allowable pelvis roll/pitch before forcing damp (when you fall)
SAFETY_MAX_TILT = np.radians(60.0)  # [rad]

# Only switch motion when the new nominal is this much closer than the current one.
# Prevents rapid oscillation when the ball sits near a boundary between two targets.
_MOTION_SWITCH_HYSTERESIS = 0.25  # metres


########################################################################
# CONTROL
########################################################################


class ControlNode(Node):
  def __init__(self, config_path: str):
    super().__init__("hardware_node")

    # import config
    self.config = self.load_config(config_path)

    # load parameters
    self.load_params()

    # IMU states
    self.pelvis_imu_rpy = None  # roll, pitch, yaw
    self.pelvis_imu_quaternion = None  # orientation
    self.pelvis_imu_gyroscope = None  # angular velocity
    self.pelvis_imu_accelerometer = None  # linear acceleration

    # Joint states
    self.q = np.zeros(G1_NUM_MOTOR)  # joint positions
    self.dq = np.zeros(G1_NUM_MOTOR)  # joint velocities
    self.ddq = np.zeros(G1_NUM_MOTOR)  # joint accelerations
    self.tau_est = np.zeros(G1_NUM_MOTOR)  # estimated joint torques

    # command arrays
    self.q_cmd = np.array(self.default_joint_pos, dtype=np.float64)
    self.dq_cmd = np.zeros(G1_NUM_MOTOR)
    self.Kp_cmd = np.array(self.Kp, dtype=np.float64)
    self.Kd_cmd = np.array(self.Kd, dtype=np.float64)
    self.tau_ff_cmd = np.zeros(G1_NUM_MOTOR)

    # locks for thread safety
    self.fsm_lock = threading.Lock()  # protects FSM state
    self.sensor_lock = threading.Lock()  # protects sensor state arrays
    self.cmd_lock = threading.Lock()  # protects command arrays

    # finite state machine state
    self.fsm_state = "init"
    self.prev_fsm_state = "init"
    self.fsm_start_time = 0.0
    self.fsm_start_q = np.zeros(G1_NUM_MOTOR)
    self.fsm_time = 0.0

    # safety flags
    self.safety_triggered = False

    # perception: pelvis and ball poses from external system
    self.pelvis_pose_position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    self.pelvis_pose_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    self.ball_pose_position = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # motion frame from control node
    self.motion_frame = 0
    self.motion_idx = 0

    # goal targets per motion index, initialized once on first pelvis pose
    # {motion_idx: {"types": [...], "pos_w": [...], "vel_w": [...], "quat_w": [...]}}
    self._goals: dict = {}
    self._goals_initialized = False

    # robot interface (created in Init) and network interface name
    self.robot = None
    self.network = None

    # hardware time and latest state flag
    self.time_ = 0.0
    self.state_ready = False

  #################################################################
  # INITIALIZATION
  #################################################################

  # load the config file
  def load_config(self, config_path: str):
    # open the config file and load it
    config_path_full = DEPLOY_DIR + "/configs/" + config_path
    with open(config_path_full, "r") as f:
      config = yaml.safe_load(f)

    print("Config file loaded successfully from: [{}].".format(config_path_full))

    return config

  # load params from config
  def load_params(self):
    # time to interpolate to initial
    self.home_pos_duration = self.config["home_pos_duration"]  # float

    # default joint positions
    self.default_joint_pos = self.config["default_joint_pos"]  # list

    # motion and contact params
    contact_phase_cfg = self.config["contact_phase"]
    self.contact_phases = (
      contact_phase_cfg if isinstance(contact_phase_cfg, list) else [contact_phase_cfg]
    )
    self.contact_duration = float(self.config["contact_duration"])
    self.motion_num_frames = [
      int(np.load(mp if os.path.isabs(mp) else os.path.join(REPO_ROOT, mp))["joint_pos"].shape[0])
      for mp in self.config["motion_paths"]
    ]

    # pelvis-frame nominal target position per motion, from config — used for motion selection
    self._nominal_positions_b = {
      g["motion_index"]: np.array(g["vector"], dtype=np.float32)
      for g in self.config.get("goals", [])
      if g["type"] == "position"
    }

    # human-readable motion names derived from motion_paths filenames
    self._motion_names = [
      Path(mp).stem for mp in self.config["motion_paths"]
    ]

    # PD gains
    self.Kp = self.config["Kp"]  # list
    self.Kd = self.config["Kd"]  # list

    # type checks
    assert type(self.home_pos_duration) in [float], "home_pos_duration must be a float."
    assert type(self.default_joint_pos) == list, "default_joint_pos must be a list."
    assert type(self.Kp) == list, "Kp must be a list."
    assert type(self.Kd) == list, "Kd must be a list."

    # length checks
    assert len(self.Kp) == G1_NUM_MOTOR, (
      f"Expected {G1_NUM_MOTOR} Kp values, got {len(self.Kp)}."
    )
    assert len(self.Kd) == G1_NUM_MOTOR, (
      f"Expected {G1_NUM_MOTOR} Kd values, got {len(self.Kd)}."
    )

    # value checks
    assert self.home_pos_duration >= 3.0, (
      "home_pos_duration must take at least 3 seconds."
    )
    assert len(self.default_joint_pos) == G1_NUM_MOTOR, (
      f"Expected {G1_NUM_MOTOR} default joint positions, "
      f"got {len(self.default_joint_pos)}"
    )
    for i in range(G1_NUM_MOTOR):
      assert self.Kp[i] >= 0.0, f"Kp for joint {i} must be non-negative."
      assert self.Kd[i] >= 0.0, f"Kd for joint {i} must be non-negative."

    print("Config parameters loaded successfully.")

  # anchor goal targets to the robot's initial pelvis pose in world frame
  def init_goals(self, pelvis_pos: np.ndarray, pelvis_quat: np.ndarray):
    goals_cfg = self.config.get("goals", [])
    R_init = quat_to_rotation_matrix(pelvis_quat)

    types, pos_w, vel_w, quat_w = [], [], [], []
    for goal in goals_cfg:
      if goal["motion_index"] != self.motion_idx:
        continue
      vec = np.array(goal["vector"], dtype=np.float32)
      goal_type = goal["type"]
      types.append(goal_type)
      if goal_type == "position":
        pos_w.append(R_init @ vec + pelvis_pos)
        vel_w.append(np.zeros(3, dtype=np.float32))
        quat_w.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
      elif goal_type == "velocity":
        pos_w.append(np.zeros(3, dtype=np.float32))
        vel_w.append(vec)
        quat_w.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
      elif goal_type == "orientation":
        pos_w.append(np.zeros(3, dtype=np.float32))
        vel_w.append(np.zeros(3, dtype=np.float32))
        quat_w.append(quat_multiply(pelvis_quat, rpy_to_quat(vec)))

    self._goals[self.motion_idx] = {
      "types": types,
      "pos_w": pos_w,
      "vel_w": vel_w,
      "quat_w": quat_w,
    }
    names = [g["name"] for g in goals_cfg if g["motion_index"] == self.motion_idx]
    print(f"Goals initialized for motion {self.motion_idx}: {names}")

  # initialize the robot interface, publishers, and subscribers
  def Init(self):
    # create the G1 interface (HG messages); this also initializes DDS on the
    # given network interface and starts the wrapper's internal 500Hz command
    # writer + state subscriber threads.
    self.robot = unitree_interface.UnitreeInterface.create_g1(self.network)

    # release any active high-level motion service (loco/sport) so our LowCmd
    # writes are not fought by it. Replaces the MotionSwitcherClient loop.
    self.robot.release_motion_control()

    # Series control for pitch/roll joints (matches previous Mode.PR usage).
    self.robot.set_control_mode(unitree_interface.ControlMode.PR)

    print("Unitree robot interface initialized successfully.")

    # ROS2 publishers
    self.pelvis_imu_state_pub = self.create_publisher(
      Float32MultiArray, "deploy_robot/pelvis_imu_state", 10
    )
    self.joint_state_pub = self.create_publisher(
      Float32MultiArray, "deploy_robot/joint_state", 10
    )
    self.hardware_time_pub = self.create_publisher(
      Float64, "deploy_robot/hardware_time", 10
    )
    self.goals_pub = self.create_publisher(Float32MultiArray, "deploy_robot/goals", 10)
    self.which_motion_pub = self.create_publisher(
      Float64, "deploy_robot/which_motion", 10
    )

    # Hardware specific publishers
    self.fsm_time_pub = self.create_publisher(Float64, "deploy_robot/fsm_time", 10)

    # ROS2 subscribers
    self.command_sub = self.create_subscription(
      Float32MultiArray, "deploy_robot/command", self.command_callback, 10
    )
    self.motion_frame_sub = self.create_subscription(
      Float64, "deploy_robot/motion_frame", self.motion_frame_callback, 10
    )

    # Hardware specific subscribers
    self.fsm_sub = self.create_subscription(
      String, "deploy_robot/fsm", self.fsm_callback, 10
    )

    # Perception subscribers
    self.pelvis_pose_sub = self.create_subscription(
      PoseStamped, "/g1_pelvis/pose", self.pelvis_pose_callback, 10
    )
    self.ball_pose_sub = self.create_subscription(
      PoseStamped, "/ball/target_pose", self.ball_pose_callback, 10
    )
    # sensor publish timer
    self.pub_timer = self.create_timer(ROS_SENSOR_PUBLISH_DT, self.publish_sensor_data)

    print("ROS2 publishers and subscribers initialized successfully.")

  # create a thread to run the low-level control loop
  def Start(self):
    # create a thread for low-level control loop, but do not start it yet
    self.lowCmdWriteThreadPtr = RecurrentThread(
      interval=LOW_LEVEL_CONTROL_DT, target=self.LowCmdWrite, name="control"
    )

    # wait until we receive the first valid low state from the robot
    while not self.read_robot_state():
      print("Waiting for first low state from robot...")
      time.sleep(1)
    self.state_ready = True

    # start the low-level control thread
    self.lowCmdWriteThreadPtr.Start()
    print("Low-level robot control thread started successfully.")

  #################################################################
  # ROS PUBLISHING AND CALLBACKS
  #################################################################

  # callback to receive FSM state from joystick
  def fsm_callback(self, msg: String):
    with self.fsm_lock:
      self.fsm_state = msg.data

  # callback to receive command messages from ROS2
  def command_callback(self, msg: Float32MultiArray):
    # expected layout: [q(29), dq(29), Kp(29), Kd(29), tau_ff(29)] = 145 floats
    data = np.array(msg.data, dtype=np.float64)

    # safety check on command length
    if len(data) != 5 * G1_NUM_MOTOR:
      self.get_logger().warn(
        f"Expected {5 * G1_NUM_MOTOR} values in command, got {len(data)}"
      )
      return

    # update command arrays under lock
    nu = G1_NUM_MOTOR
    with self.cmd_lock:
      self.q_cmd[:] = data[0 * nu : 1 * nu]
      self.dq_cmd[:] = data[1 * nu : 2 * nu]
      self.Kp_cmd[:] = data[2 * nu : 3 * nu]
      self.Kd_cmd[:] = data[3 * nu : 4 * nu]
      self.tau_ff_cmd[:] = data[4 * nu : 5 * nu]

  # callback to receive motion frame from control node
  def motion_frame_callback(self, msg: Float64):
    self.motion_frame = int(msg.data)

  # publish sensor data to ROS2 topics
  def publish_sensor_data(self):
    # read sensor data under lock
    with self.sensor_lock:
      # pelvis IMU state
      pelvis_imu_rpy = (
        np.array(self.pelvis_imu_rpy, dtype=np.float64)
        if self.pelvis_imu_rpy is not None
        else np.zeros(3)
      )
      pelvis_imu_quat = (
        np.array(self.pelvis_imu_quaternion, dtype=np.float64)
        if self.pelvis_imu_quaternion is not None
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
      )
      pelvis_imu_gyro = (
        np.array(self.pelvis_imu_gyroscope, dtype=np.float64)
        if self.pelvis_imu_gyroscope is not None
        else np.zeros(3)
      )
      pelvis_imu_accel = (
        np.array(self.pelvis_imu_accelerometer, dtype=np.float64)
        if self.pelvis_imu_accelerometer is not None
        else np.zeros(3)
      )

      # joint state
      q = self.q.copy()
      dq = self.dq.copy()
      ddq = self.ddq.copy()
      tau_est = self.tau_est.copy()

    # imu_state: [rpy(3), quaternion(4), gyroscope(3), accelerometer(3)] = 13 floats
    pelvis_imu_msg = Float32MultiArray()
    pelvis_imu_msg.data = np.concatenate(
      [pelvis_imu_rpy, pelvis_imu_quat, pelvis_imu_gyro, pelvis_imu_accel]
    ).tolist()

    # joint_state: [q(29), dq(29), ddq(29), tau_est(29)] = 116 floats
    joint_msg = Float32MultiArray()
    joint_msg.data = np.concatenate([q, dq, ddq, tau_est]).tolist()

    # hardware_time: single float
    time_msg = Float64()
    time_msg.data = self.time_

    # fsm_time: time since entering current state
    fsm_time_msg = Float64()
    fsm_time_msg.data = self.fsm_time
    # compute goals in anchor (pelvis) frame
    goals_msg = Float32MultiArray()
    with self.sensor_lock:
      pelvis_quat = (
        np.array(self.pelvis_imu_quaternion, dtype=np.float32)
        if self.pelvis_imu_quaternion is not None
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
      )
      ball_pos_b = np.array(self.ball_pose_position, dtype=np.float32)
    if self._goals_initialized:
      pelvis_quat_inv = quat_conjugate(pelvis_quat)
      g = self._goals[self.motion_idx]
      motion_frame = self.motion_frame
      # contact_end_frame = int(
      #   self.motion_num_frames[self.motion_idx]
      #   * (self.contact_phases[self.motion_idx] + self.contact_duration)
      # )
      goal_vecs = []
      for goal_type, _, vel_w, goal_quat_w in zip(
        g["types"], g["pos_w"], g["vel_w"], g["quat_w"]
      ):
        if goal_type == "position":
          if motion_frame == 0:
            goal_vecs.append(self._nominal_positions_b[self.motion_idx])
          else:
            print('motion triggered, sending ball position as goal')
            goal_vecs.append(ball_pos_b)
        elif goal_type == "velocity":
          goal_vecs.append(vel_w)
        elif goal_type == "orientation":
          goal_vecs.append(quat_multiply(pelvis_quat_inv, goal_quat_w))
      goals_msg.data = np.concatenate(goal_vecs).tolist()
    # publish to ROS2 topics
    which_motion_msg = Float64()
    which_motion_msg.data = float(self.motion_idx)

    self.pelvis_imu_state_pub.publish(pelvis_imu_msg)
    self.joint_state_pub.publish(joint_msg)
    self.hardware_time_pub.publish(time_msg)
    self.fsm_time_pub.publish(fsm_time_msg)
    self.goals_pub.publish(goals_msg)
    self.which_motion_pub.publish(which_motion_msg)

  #################################################################
  # SDK HARDWARE
  #################################################################

  # read the latest low state from the robot interface and update sensor arrays.
  # returns True once a valid state has been received from the robot.
  def read_robot_state(self):
    st = self.robot.read_low_state()
    q = np.asarray(st.motor.q, dtype=np.float64)
    quat = np.asarray(st.imu.quat, dtype=np.float64)

    # before any DDS state has arrived the wrapper buffers are zero-filled; a
    # valid IMU quaternion has unit norm, so use that to gate readiness.
    if q.shape[0] < G1_NUM_MOTOR or np.linalg.norm(quat) < 0.5:
      return False

    # update sensor states under lock
    with self.sensor_lock:
      # update IMU states (wrapper field names: rpy / quat / omega / accel)
      self.pelvis_imu_rpy = np.asarray(st.imu.rpy, dtype=np.float64)
      self.pelvis_imu_quaternion = quat
      self.pelvis_imu_gyroscope = np.asarray(st.imu.omega, dtype=np.float64)
      self.pelvis_imu_accelerometer = np.asarray(st.imu.accel, dtype=np.float64)

      # update joint states
      self.q[:] = q[:G1_NUM_MOTOR]
      self.dq[:] = np.asarray(st.motor.dq, dtype=np.float64)[:G1_NUM_MOTOR]
      self.tau_est[:] = np.asarray(st.motor.tau_est, dtype=np.float64)[:G1_NUM_MOTOR]
      # wrapper exposes no joint acceleration; downstream uses only q and dq.
      self.ddq[:] = 0.0
    return True

  # callback to receive pelvis pose messages from perception
  def pelvis_pose_callback(self, msg: PoseStamped):
    p = msg.pose.position
    q = msg.pose.orientation  # ROS: (x, y, z, w)
    with self.sensor_lock:
      self.pelvis_pose_position = np.array([p.x, p.y, p.z], dtype=np.float64)
      self.pelvis_pose_quaternion = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)

    if not self._goals_initialized:
      pelvis_pos = np.array(self.pelvis_pose_position, dtype=np.float32)
      with self.sensor_lock:
        pelvis_quat = (
          np.array(self.pelvis_imu_quaternion, dtype=np.float32)
          if self.pelvis_imu_quaternion is not None
          else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
      old_idx = self.motion_idx
      for idx in range(len(self.motion_num_frames)):
        self.motion_idx = idx
        self.init_goals(pelvis_pos, pelvis_quat)
      self.motion_idx = old_idx
      self._goals_initialized = True

  def ball_pose_callback(self, msg: PoseStamped):
    p = msg.pose.position
    # target positions are always sent in local frame!
    ball_pos_b = np.array([p.x, p.y, p.z], dtype=np.float32)

    with self.sensor_lock:
      self.ball_pose_position = ball_pos_b

    if not self._goals_initialized:
      return
     
    if self.motion_frame > 0:
      return

    # Select the closest motion nominal. Track both the global best and the
    # current motion's distance so we can apply hysteresis before switching.
    best_idx = self.motion_idx
    best_dist = float("inf")
    curr_dist = float("inf")
    for idx, nominal_b in self._nominal_positions_b.items():
      dist = float(np.linalg.norm(ball_pos_b - nominal_b))
      if idx == self.motion_idx:
        curr_dist = dist
      if dist < best_dist:
        best_dist = dist
        best_idx = idx

    # Hysteresis: only commit to a different motion when the candidate is
    # meaningfully closer than the current one.  Without this, the argmin
    # oscillates as the ball crosses a nominal boundary, which sends rapid
    # alternating motion references to the control node and destabilises it.
    if (
      best_idx != self.motion_idx
      and best_dist + _MOTION_SWITCH_HYSTERESIS < curr_dist
    ):
      nominal = self._nominal_positions_b[best_idx]
      name = self._motion_names[best_idx] if best_idx < len(self._motion_names) else str(best_idx)
      print(
        f"Motion switch -> {name} (idx {best_idx})  "
        f"nominal=({nominal[0]:.3f}, {nominal[1]:.3f}, {nominal[2]:.3f})  "
        f"ball=({ball_pos_b[0]:.3f}, {ball_pos_b[1]:.3f}, {ball_pos_b[2]:.3f})"
      )
      self.motion_idx = best_idx

  # main control loop to send low-level commands
  def LowCmdWrite(self):
    # refresh the latest robot state (IMU + joints) from the wrapper buffers
    self.read_robot_state()

    # update hardware time
    self.time_ += LOW_LEVEL_CONTROL_DT

    # read FSM state under lock
    with self.fsm_lock:
      fsm_state = self.fsm_state

    # detect state transition
    if fsm_state != self.prev_fsm_state:
      print(f"FSM: {self.prev_fsm_state} -> {fsm_state}")
      self.fsm_start_time = self.time_
      with self.sensor_lock:
        self.fsm_start_q = self.q.copy()
      self.prev_fsm_state = fsm_state

    # update fsm time
    self.fsm_time = self.time_ - self.fsm_start_time

    # safety: force damp if pelvis tilts beyond specified threshold
    if not self.safety_triggered:
      with self.sensor_lock:
        rpy = self.pelvis_imu_rpy
      if rpy is not None:
        roll, pitch = abs(rpy[0]), abs(rpy[1])
        if roll > SAFETY_MAX_TILT or pitch > SAFETY_MAX_TILT:
          print()
          print("*" * 70)
          print(
            f"SAFETY: roll={np.degrees(roll):.2f} pitch={np.degrees(pitch):.2f} -> FORCING DAMP. PLEASE RESTART!"
          )
          print("*" * 70)
          self.safety_triggered = True
    if self.safety_triggered:
      fsm_state = "damp"

    # Build the per-joint command arrays for the active FSM state. The wrapper's
    # internal 500Hz writer fills mode=1, mode_pr, mode_machine, and CRC; we only
    # provide q/dq/kp/kd/tau_ff.
    q_t = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
    dq_t = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
    kp = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
    kd = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
    tau = np.zeros(G1_NUM_MOTOR, dtype=np.float64)

    # [init]: zero out all commands (arrays already zero)
    if fsm_state == "init":
      pass

    # [damp]: Kd damping, no position tracking
    elif fsm_state == "damp":
      kd[:] = 3.0

    # [home]: interpolate to default joint positions and gains
    elif fsm_state == "home":
      ratio = np.clip(self.fsm_time / self.home_pos_duration, 0.0, 1.0)
      q_t[:] = (1.0 - ratio) * self.fsm_start_q + ratio * np.asarray(
        self.default_joint_pos, dtype=np.float64
      )
      kp[:] = ratio * np.asarray(self.Kp, dtype=np.float64)
      kd[:] = ratio * np.asarray(self.Kd, dtype=np.float64)

    # [control]: read from ROS2 command subscriber
    elif fsm_state == "control":
      with self.cmd_lock:
        q_t[:] = self.q_cmd
        dq_t[:] = self.dq_cmd
        kp[:] = self.Kp_cmd
        kd[:] = self.Kd_cmd
        tau[:] = self.tau_ff_cmd

    # send the command to the robot (whole-list assignment: pybind11 list
    # properties are copy-on-read, so element-wise writes would not persist).
    cmd = self.robot.create_zero_command()
    cmd.q_target = q_t.tolist()
    cmd.dq_target = dq_t.tolist()
    cmd.kp = kp.tolist()
    cmd.kd = kd.tolist()
    cmd.tau_ff = tau.tolist()
    self.robot.write_low_command(cmd)


############################################################################
# MAIN FUNCTION
############################################################################


def main(args=None):
  # init ROS2
  rclpy.init()

  # parse arguments
  parser = argparse.ArgumentParser(
    description="Hardware deployment node intended to run onboard the G1 robot."
  )
  # network interface name argument.
  # Onboard the robot, DDS reaches the motion-control board over the internal
  # wired link "eth0" (192.168.123.x), so this defaults to "eth0" and need not
  # be passed. Override only if running off-board over a different interface.
  parser.add_argument(
    "--network",
    type=str,
    default="eth0",
    help='Network interface name for robot communication. Onboard default: "eth0".',
  )
  # config path argument
  parser.add_argument(
    "--config",
    type=str,
    required=True,
    help='Path to the config yaml file for hardware. Example: "g1_29dof_hardware.yaml".',
  )
  args = parser.parse_args()

  print()
  while input("Press [Enter] to continue: ") != "":
    pass
  print()

  # instantiate the custom control class (DDS is initialized inside the wrapper
  # using the network interface stored below)
  ctrl_node = ControlNode(args.config)
  ctrl_node.network = args.network
  ctrl_node.Init()

  # spin ROS2 node in background thread
  ros_running = True

  def spin_ros():
    while ros_running and rclpy.ok():
      try:
        rclpy.spin_once(ctrl_node, timeout_sec=0.1)
      except Exception:
        break

  ros_thread = threading.Thread(target=spin_ros, daemon=True)
  ros_thread.start()

  # start the control loop
  ctrl_node.Start()

  # run normally
  try:
    while True:
      time.sleep(1)
  # ctrl + C
  except KeyboardInterrupt:
    print("\nExiting...")
  # graceful shutdown on any exception
  finally:
    ros_running = False
    ros_thread.join(timeout=1.0)
    try:
      ctrl_node.destroy_node()
    except Exception:
      pass
    try:
      if rclpy.ok():
        rclpy.shutdown()
    except Exception:
      pass

  print("Hardware shutdown complete.")


if __name__ == "__main__":
  main()
