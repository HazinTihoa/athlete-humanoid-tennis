##
#
# Ball pose estimator node.
#
# Subscribes to raw ball detections and publishes estimated pose to /ball/target_pose.
#
##

# standard imports
import argparse
import numpy as np
import threading
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64
from geometry_msgs.msg import PoseStamped

# directory imports
import os
import sys

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

DEPLOY_DIR = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(DEPLOY_DIR)

############################################################################
# SE(3) EKF — single integrator (constant-pose) process model
############################################################################


class SE3EKF:
  """
  EKF for a pose in SE(3) with single-integrator dynamics (ẋ = 0 + noise).

  State:    T = (p, q) — position ℝ³ + unit quaternion [x,y,z,w]
  Error state: (δp, δθ) ∈ ℝ⁶  (tangent-space / Lie-algebra representation)
  Process:  F = I₆,  P ← P + Q·dt   (random-walk in the tangent space)
  Measure:  full SE(3) pose;  H = I₆  (error-state Jacobian is identity)
  """

  def __init__(
    self,
    process_noise_pos: float = 0.05,
    process_noise_rot: float = 0.05,
    meas_noise_pos: float = 0.005,
    meas_noise_rot: float = 0.005,
  ):
    self.p = np.zeros(3)
    self.q = np.array([0.0, 0.0, 0.0, 1.0])  # [x,y,z,w]
    self.P = np.eye(6)                         # error-state covariance

    self.Q = np.diag([process_noise_pos**2] * 3 + [process_noise_rot**2] * 3)
    self.R = np.diag([meas_noise_pos**2] * 3 + [meas_noise_rot**2] * 3)
    self.initialized = False

  # ---- quaternion helpers ------------------------------------------------

  @staticmethod
  def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
      w1*x2 + x1*w2 + y1*z2 - z1*y2,
      w1*y2 - x1*z2 + y1*w2 + z1*x2,
      w1*z2 + x1*y2 - y1*x2 + z1*w2,
      w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])

  @staticmethod
  def _qinv(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])

  @staticmethod
  def _exp(v: np.ndarray) -> np.ndarray:
    """SO(3) exponential: rotation vector → quaternion [x,y,z,w]."""
    angle = np.linalg.norm(v)
    if angle < 1e-8:
      return np.array([0.0, 0.0, 0.0, 1.0])
    s = np.sin(0.5 * angle)
    return np.array([v[0]/angle*s, v[1]/angle*s, v[2]/angle*s, np.cos(0.5 * angle)])

  @staticmethod
  def _log(q: np.ndarray) -> np.ndarray:
    """SO(3) logarithm: quaternion [x,y,z,w] → rotation vector."""
    v, w = q[:3], q[3]
    nv = np.linalg.norm(v)
    if nv < 1e-8:
      return np.zeros(3)
    return (2.0 * np.arctan2(nv, w)) * v / nv

  # ---- EKF steps ---------------------------------------------------------

  def predict(self, dt: float) -> None:
    """Single-integrator predict: state unchanged, covariance grows by Q·dt."""
    self.P += self.Q * dt

  def update(self, p_meas: np.ndarray, q_meas: np.ndarray) -> None:
    """Correct state with a full SE(3) pose measurement."""
    if not self.initialized:
      self.p = p_meas.copy()
      self.q = q_meas.copy()
      self.initialized = True
      return

    # innovation in ℝ³ × so(3)
    dp = p_meas - self.p
    dtheta = self._log(self._qmul(q_meas, self._qinv(self.q)))
    inn = np.concatenate([dp, dtheta])

    # H = I₆  →  S = P + R
    S = self.P + self.R
    K = self.P @ np.linalg.inv(S)

    delta = K @ inn

    # apply correction via Lie group composition
    self.p += delta[:3]
    self.q = self._qmul(self._exp(delta[3:]), self.q)
    self.q /= np.linalg.norm(self.q)

    # Joseph-form covariance update (numerically stable)
    IKH = np.eye(6) - K
    self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T


############################################################################
# ESTIMATOR NODE
############################################################################


class BallEstimatorNode(Node):
  def __init__(self):
    super().__init__("ball_estimator_node")

    # state
    self.ball_pos = np.zeros(3, dtype=np.float64)
    self.ball_received = False
    self.pelvis_pos = np.zeros(3, dtype=np.float64)
    self.pelvis_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    self.target_pos = np.zeros(3, dtype=np.float64)
    self.target_time = -1.0

    config_path = os.path.join(
      os.path.dirname(__file__), "..", "configs", "g1_athlete.yaml"
    )
    with open(config_path) as f:
      config = yaml.safe_load(f)
    goals = {g["name"]: g for g in config["goals"]}
    self.nominal_target_pos_pelvis = np.array(
      goals["forehand_position"]["vector"], dtype=np.float64
    )
    self.nominal_target_pos = np.zeros(3, dtype=np.float64)

    self.pelvis_ekf = SE3EKF(
      process_noise_pos=0.05,
      process_noise_rot=0.05,
      meas_noise_pos=0.005,
      meas_noise_rot=0.005,
    )
    self._last_predict_time: float | None = None

    self.lock = threading.Lock()

    # subscribers
    self.ball_pos_sub = self.create_subscription(
      PoseStamped, "/ball/pose", self.ball_pose_callback, 10
    )
    self.pelvis_pos_sub = self.create_subscription(
      PoseStamped, "/g1_pelvis/pose", self.pelvis_pose_callback, 10
    )

    # publishers
    self.ball_pose_pub = self.create_publisher(PoseStamped, "/ball/target_pose", 10)
    self.pelvis_est_pub = self.create_publisher(PoseStamped, "/g1_pelvis/filtered_pose", 10)
    # self.ball_target_time = self.create_publisher(Float64, "/ball/target_time", 10)

    self.create_timer(0.02, self.timer_callback)

    print("Ball estimator node initialized.")

  def ball_pose_callback(self, msg: PoseStamped):
    p = msg.pose.position
    self.ball_pos = np.array([p.x, p.y, p.z], dtype=np.float64)
    self.ball_received = True

  # callback: pelvis pose in world frame — EKF update then recompute target
  def pelvis_pose_callback(self, msg: PoseStamped):
    p_raw = np.array(
      [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64
    )
    q_raw = np.array(
      [
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
      ],
      dtype=np.float64,
    )

    self.pelvis_ekf.update(p_raw, q_raw)

    self.pelvis_pos = self.pelvis_ekf.p
    self.pelvis_quat = self.pelvis_ekf.q

    q_vec = self.pelvis_quat[:3]
    qw = self.pelvis_quat[3]
    t = 2.0 * np.cross(q_vec, self.nominal_target_pos_pelvis)
    self.nominal_target_pos = (
      self.nominal_target_pos_pelvis + qw * t + np.cross(q_vec, t) + self.pelvis_pos
    )

  # given the estimated position and velocity, find the point closest to the robot target point in world frame along the ball's trajectory, this will be the target pos
  def estimate_target_point(self):
    v = self.ball_pos - self.pelvis_pos
    q_vec = self.pelvis_quat[:3]
    qw = self.pelvis_quat[3]
    t = 2.0 * np.cross(q_vec, v)
    self.target_pos = v - qw * t + np.cross(q_vec, t)

  def timer_callback(self):
    now = self.get_clock().now().nanoseconds * 1e-9
    if self._last_predict_time is not None:
      dt = now - self._last_predict_time
      if 0.0 < dt < 1.0:
        self.pelvis_ekf.predict(dt)
    self._last_predict_time = now

    if self.pelvis_ekf.initialized:
      self.publish_pelvis_estimate()
    if not self.ball_received:
      return
    self.estimate_target_point()
    self.publish_pose()

  def publish_pelvis_estimate(self):
    msg = PoseStamped()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = "world"
    msg.pose.position.x = self.pelvis_ekf.p[0]
    msg.pose.position.y = self.pelvis_ekf.p[1]
    msg.pose.position.z = self.pelvis_ekf.p[2]
    msg.pose.orientation.x = self.pelvis_ekf.q[0]
    msg.pose.orientation.y = self.pelvis_ekf.q[1]
    msg.pose.orientation.z = self.pelvis_ekf.q[2]
    msg.pose.orientation.w = self.pelvis_ekf.q[3]
    self.pelvis_est_pub.publish(msg)

  def publish_pose(self):
    msg = PoseStamped()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = "world"
    msg.pose.position.x = self.target_pos[0]  # in world frame (for all of these)
    msg.pose.position.y = self.target_pos[1]
    msg.pose.position.z = self.target_pos[2]
    msg.pose.orientation.w = 1.0
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    self.ball_pose_pub.publish(msg)


############################################################################
# MAIN FUNCTION
############################################################################


def main(args=None):
  rclpy.init()

  parser = argparse.ArgumentParser(description="Ball pose estimator node.")
  args = parser.parse_args()

  node = BallEstimatorNode()

  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()

  print("Ball estimator shutdown complete.")


if __name__ == "__main__":
  main()
