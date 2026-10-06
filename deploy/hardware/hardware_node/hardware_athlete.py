##
#
# Deployment code for Unitree G1 robot.
#
# Common G1 "hg" topics: https://support.unitree.com/home/en/G1_developer/dds_services_interface
#
##

# standard imports
import argparse

# directory imports
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ROS2 imports
import rclpy

# other imports
import yaml
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64, String

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

DEPLOY_DIR = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, DEPLOY_DIR)

# custom imports
# Unitree SDK imports (C++ wrapper: submodules/unitree_sdk2_wrapper)
import unitree_interface
from utils.deployment_fsm import DeploymentFSM
from utils.safety import (
  UNITREE_REMOTE_A,
  UNITREE_REMOTE_B,
  UNITREE_REMOTE_X,
  UNITREE_REMOTE_Y,
)
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

########################################################################
# CONTROL
########################################################################


class ControlNode(Node):
  def __init__(
    self,
    config_path: str,
    *,
    enable_low_level_commands: bool = True,
    use_unitree_remote_fsm: bool = False,
  ):
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

    # locks for thread safety
    self.sensor_lock = threading.Lock()  # protects sensor state arrays

    self.use_unitree_remote_fsm = use_unitree_remote_fsm
    self.deployment_fsm = DeploymentFSM(
      default_joint_position=self.default_joint_pos,
      home_kp=self.Kp,
      home_kd=self.Kd,
      home_duration_s=self.home_pos_duration,
      command_timeout_s=self.command_timeout_s,
      damp_kd=self.damp_kd,
    )
    self._last_logged_fsm_state = self.deployment_fsm.state
    self._last_logged_fault = None

    # robot interface (created in Init) and network interface name
    self.robot = None
    self.network = None

    # hardware time and latest state flag
    self.time_ = 0.0
    self.control_time_origin_monotonic = time.monotonic()
    self.state_ready = False
    self.enable_low_level_commands = enable_low_level_commands
    self._wireless_keys = 0

  #################################################################
  # INITIALIZATION
  #################################################################

  # load the config file
  def load_config(self, config_path: str):
    # open the config file and load it
    config_candidate = Path(config_path).expanduser()
    if not config_candidate.is_absolute():
      config_candidate = Path(DEPLOY_DIR) / "configs" / config_candidate
    config_path_full = str(config_candidate.resolve())
    with open(config_path_full, "r") as f:
      config = yaml.safe_load(f)

    print(f"Config file loaded successfully from: [{config_path_full}].")

    return config

  # load params from config
  def load_params(self):
    # time to interpolate to initial
    self.home_pos_duration = self.config["home_pos_duration"]  # float

    # default joint positions
    self.default_joint_pos = self.config["default_joint_pos"]  # list

    # PD gains
    self.Kp = self.config["Kp"]  # list
    self.Kd = self.config["Kd"]  # list
    self.command_timeout_s = float(self.config.get("command_timeout_s", 0.1))
    self.damp_kd = float(self.config.get("damp_kd", 3.0))
    self.safety_max_tilt_rad = np.radians(
      float(self.config.get("safety_max_tilt_deg", 60.0))
    )

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
    assert np.isfinite(self.command_timeout_s) and self.command_timeout_s > 0.0, (
      "command_timeout_s must be finite and positive."
    )
    assert np.isfinite(self.damp_kd) and self.damp_kd >= 0.0, (
      "damp_kd must be finite and non-negative."
    )
    assert np.isfinite(self.safety_max_tilt_rad) and self.safety_max_tilt_rad > 0.0, (
      "safety_max_tilt_deg must be finite and positive."
    )
    assert len(self.default_joint_pos) == G1_NUM_MOTOR, (
      f"Expected {G1_NUM_MOTOR} default joint positions, "
      f"got {len(self.default_joint_pos)}"
    )
    for i in range(G1_NUM_MOTOR):
      assert self.Kp[i] >= 0.0, f"Kp for joint {i} must be non-negative."
      assert self.Kd[i] >= 0.0, f"Kd for joint {i} must be non-negative."

    print("Config parameters loaded successfully.")

  # initialize the robot interface, publishers, and subscribers
  def Init(self):
    # create the G1 interface (HG messages); this also initializes DDS on the
    # given network interface and starts the wrapper's internal 500Hz command
    # writer + state subscriber threads.
    self.robot = unitree_interface.UnitreeInterface.create_g1(self.network)

    if self.enable_low_level_commands:
      if self.use_unitree_remote_fsm:
        print(
          "Hold the G1 wireless remote B button to authorize low-level control..."
        )
        while True:
          keys = int(self.robot.read_wireless_controller().keys)
          if keys & UNITREE_REMOTE_B:
            self._wireless_keys = keys
            print("G1 wireless remote confirmed; startup remains in damp.")
            break
          time.sleep(0.05)
      # Release high-level motion only when this process is explicitly armed.
      self.robot.release_motion_control()
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
    self.control_time_pub = self.create_publisher(
      Float64, "deploy_robot/control_time", 10
    )
    # Hardware specific publishers
    self.fsm_time_pub = self.create_publisher(Float64, "deploy_robot/fsm_time", 10)
    self.fsm_state_pub = self.create_publisher(String, "deploy_robot/fsm_state", 10)
    self.fsm_request_pub = self.create_publisher(
      String, "deploy_robot/fsm_request", 10
    )
    self.root_calibration_request_pub = self.create_publisher(
      String, "deploy_robot/root_calibration_request", 10
    )

    # ROS2 subscribers
    self.command_sub = self.create_subscription(
      Float32MultiArray, "deploy_robot/command", self.command_callback, 10
    )
    # Hardware specific subscribers
    self.fsm_request_sub = self.create_subscription(
      String, "deploy_robot/fsm_request", self.fsm_request_callback, 10
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
    if self.enable_low_level_commands:
      self.lowCmdWriteThreadPtr.Start()
      print("Low-level robot control thread started successfully.")
    else:
      print("READ-ONLY: robot state is live; LowCmd output is disabled.")
    if self.use_unitree_remote_fsm:
      print(
        "Unitree wireless FSM enabled: B=damp/emergency, A=home, "
        "Y=calibrate root, X=control; startup state=damp."
      )

  #################################################################
  # ROS PUBLISHING AND CALLBACKS
  #################################################################

  def fsm_request_callback(self, msg: String):
    try:
      self.deployment_fsm.request(msg.data)
    except ValueError:
      self.get_logger().error(f"Rejected FSM request: {msg.data!r}")

  def _publish_fsm_request(self, requested_state: str):
    self.deployment_fsm.request(requested_state)
    message = String()
    message.data = requested_state
    self.fsm_request_pub.publish(message)

  def _update_unitree_remote_fsm(self):
    if not self.use_unitree_remote_fsm or self.robot is None:
      return
    controller = self.robot.read_wireless_controller()
    keys = int(controller.keys)
    previous_keys = self._wireless_keys
    current_state = self.deployment_fsm.state
    self._wireless_keys = keys

    pressed = (keys & ~previous_keys) & 0xFFFF
    if pressed:
      button_names = [
        name
        for mask, name in (
          (UNITREE_REMOTE_A, "A"),
          (UNITREE_REMOTE_B, "B"),
          (UNITREE_REMOTE_X, "X"),
          (UNITREE_REMOTE_Y, "Y"),
        )
        if pressed & mask
      ]
      label = "+".join(button_names) if button_names else f"unknown(0x{pressed:04x})"
      self.get_logger().info(
        f"Unitree remote pressed: {label}; FSM state={current_state}"
      )
    if pressed & UNITREE_REMOTE_Y:
      home_elapsed_s = time.monotonic() - self.deployment_fsm.state_entry_s
      if current_state == "home" and home_elapsed_s >= self.home_pos_duration:
        message = String()
        message.data = "calibrate"
        self.root_calibration_request_pub.publish(message)
        self.get_logger().info("Requested Mocap root calibration from current Home pose")
      elif current_state == "home":
        self.get_logger().warn(
          "Root calibration ignored: wait for the Home interpolation to finish"
        )
      else:
        self.get_logger().warn("Root calibration ignored: press Y while FSM is in home")
    if pressed & UNITREE_REMOTE_B:
      self._publish_fsm_request("damp")
    elif pressed & UNITREE_REMOTE_A:
      self._publish_fsm_request("home")
    elif pressed & UNITREE_REMOTE_X:
      self._publish_fsm_request("control")

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
    if not np.isfinite(data).all():
      self.get_logger().error("Rejected command containing NaN or Inf.")
      return

    self.deployment_fsm.update_policy_command(data, now_s=time.monotonic())

  # publish sensor data to ROS2 topics
  def publish_sensor_data(self):
    # Keep ROS state live in read-only and shadow modes, where the 500 Hz
    # LowCmd writer is intentionally disabled.
    if self.robot is not None:
      self.read_robot_state()
    self._update_unitree_remote_fsm()
    now = time.monotonic()

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

    if not self.enable_low_level_commands:
      self.deployment_fsm.step(q, now_s=now)
      self._log_fsm_changes()

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
    control_time_msg = Float64()
    control_time_msg.data = time.monotonic() - self.control_time_origin_monotonic

    # fsm_time: time since entering current state
    fsm_time_msg = Float64()
    fsm_time_msg.data = max(0.0, now - self.deployment_fsm.state_entry_s)
    fsm_state_msg = String()
    fsm_state_msg.data = self.deployment_fsm.state

    self.pelvis_imu_state_pub.publish(pelvis_imu_msg)
    self.joint_state_pub.publish(joint_msg)
    self.hardware_time_pub.publish(time_msg)
    self.control_time_pub.publish(control_time_msg)
    self.fsm_time_pub.publish(fsm_time_msg)
    self.fsm_state_pub.publish(fsm_state_msg)

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

  def _log_fsm_changes(self):
    state = self.deployment_fsm.state
    fault = self.deployment_fsm.fault_reason
    if state != self._last_logged_fsm_state:
      self.get_logger().info(f"FSM: {self._last_logged_fsm_state} -> {state}")
      self._last_logged_fsm_state = state
    if fault != self._last_logged_fault:
      if fault is not None:
        self.get_logger().error(f"FSM forced damp: {fault}")
      self._last_logged_fault = fault

  # main control loop to send low-level commands
  def LowCmdWrite(self):
    # refresh the latest robot state (IMU + joints) from the wrapper buffers
    if not self.read_robot_state():
      return

    # update hardware time
    self.time_ += LOW_LEVEL_CONTROL_DT

    now = time.monotonic()
    with self.sensor_lock:
      q = self.q.copy()
      rpy = None if self.pelvis_imu_rpy is None else self.pelvis_imu_rpy.copy()
    if rpy is not None:
      roll, pitch = abs(float(rpy[0])), abs(float(rpy[1]))
      if roll > self.safety_max_tilt_rad or pitch > self.safety_max_tilt_rad:
        if self.deployment_fsm.fault_reason != "tilt_limit":
          self.get_logger().error(
            "Pelvis tilt limit exceeded: "
            f"roll={np.degrees(roll):.1f}deg pitch={np.degrees(pitch):.1f}deg"
          )
        self.deployment_fsm.force_damp(
          "tilt_limit",
          now_s=now,
          hard=True,
        )

    self.deployment_fsm.step(q, now_s=now)
    self._log_fsm_changes()

    # send the command to the robot (whole-list assignment: pybind11 list
    # properties are copy-on-read, so element-wise writes would not persist).
    cmd = self.robot.create_zero_command()
    cmd.q_target = self.deployment_fsm.q_target.tolist()
    cmd.dq_target = self.deployment_fsm.dq_target.tolist()
    cmd.kp = self.deployment_fsm.kp.tolist()
    cmd.kd = self.deployment_fsm.kd.tolist()
    cmd.tau_ff = self.deployment_fsm.tau_ff.tolist()
    self.robot.write_low_command(cmd)


############################################################################
# MAIN FUNCTION
############################################################################


def main(args=None):
  # init ROS2
  rclpy.init()

  # parse arguments
  parser = argparse.ArgumentParser(
    description="Hardware deployment node using Unitree SDK2 for Python."
  )
  # network interface name argument
  parser.add_argument(
    "--network",
    type=str,
    required=True,
    help='Network interface name for robot communication. Example: "enp8s0".',
  )
  # config path argument
  parser.add_argument(
    "--config",
    type=str,
    required=True,
    help='Path to the config yaml file for hardware. Example: "g1_29dof_hardware.yaml".',
  )
  parser.add_argument(
    "--read-only",
    action="store_true",
    help="Read and publish Unitree state without releasing motion control or sending LowCmd.",
  )
  parser.add_argument(
    "--no-prompt",
    action="store_true",
    help="Skip the interactive Enter prompt (for supervised launch scripts).",
  )
  parser.add_argument(
    "--unitree-remote-fsm",
    action="store_true",
    help="Use the G1 wireless remote for damp/home/control transitions.",
  )
  args = parser.parse_args()

  if not args.no_prompt:
    print()
    while input("Press [Enter] to continue: ") != "":
      pass
    print()

  # instantiate the custom control class (DDS is initialized inside the wrapper
  # using the network interface stored below)
  ctrl_node = ControlNode(
    args.config,
    enable_low_level_commands=not args.read_only,
    use_unitree_remote_fsm=args.unitree_remote_fsm,
  )
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
    while rclpy.ok():
      time.sleep(0.1)
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
