##
#
# Ball pose estimator node.
#
# Subscribes to raw ball detections and publishes estimated pose to /ball/pose.
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
from geometry_msgs.msg import PoseStamped, Vector3Stamped

# directory imports
import os
import sys

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

DEPLOY_DIR = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(DEPLOY_DIR)

try:
  from .tennis_ballistics import (
    first_plane_intersection,
    predict_ball_trajectory,
  )
except ImportError:
  from tennis_ballistics import (
    first_plane_intersection,
    predict_ball_trajectory,
  )


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
    angle = np.linalg.norm(v)
    if angle < 1e-8:
      return np.array([0.0, 0.0, 0.0, 1.0])
    s = np.sin(0.5 * angle)
    return np.array([v[0]/angle*s, v[1]/angle*s, v[2]/angle*s, np.cos(0.5 * angle)])

  @staticmethod
  def _log(q: np.ndarray) -> np.ndarray:
    v, w = q[:3], q[3]
    nv = np.linalg.norm(v)
    if nv < 1e-8:
      return np.zeros(3)
    return (2.0 * np.arctan2(nv, w)) * v / nv

  def predict(self, dt: float) -> None:
    self.P += self.Q * dt

  def update(self, p_meas: np.ndarray, q_meas: np.ndarray) -> None:
    if not self.initialized:
      self.p = p_meas.copy()
      self.q = q_meas.copy()
      self.initialized = True
      return

    dp = p_meas - self.p
    dtheta = self._log(self._qmul(q_meas, self._qinv(self.q)))
    inn = np.concatenate([dp, dtheta])

    S = self.P + self.R
    K = self.P @ np.linalg.inv(S)

    delta = K @ inn
    self.p += delta[:3]
    self.q = self._qmul(self._exp(delta[3:]), self.q)
    self.q /= np.linalg.norm(self.q)

    IKH = np.eye(6) - K
    self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T


############################################################################
# ESTIMATOR NODE
############################################################################

class BallEstimatorNode(Node):
  def __init__(self, config_path: str = "g1_athlete_tennis.yaml"):
    super().__init__("ball_estimator_node")

    # state
    self.ball_pos = np.zeros(3, dtype=np.float64)
    self.ball_vel = np.zeros(3, dtype=np.float64)
    self.pelvis_pos = np.zeros(3, dtype=np.float64)
    self.pelvis_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    self.target_pos = np.zeros(3, dtype=np.float64)
    self.target_time = -1.0
    self.ball_trajectory_positions = np.zeros((0, 3), dtype=np.float64)
    self.ball_trajectory_times = np.zeros(0, dtype=np.float64)
    self.ball_trajectory_time_length = 5.0  # this is the amount of time in the future that the trajectory prediction should cover
    self.intercept_plane_dist = 0.5  # metres in front of the pelvis where the intercept plane sits

    self.dt = 0.02
    self.ball_initialized = False  # set True once first /ball/pose message arrives

    if not os.path.isabs(config_path):
      config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", config_path
      )
    with open(config_path) as f:
      config = yaml.safe_load(f)
    position_goals = [g for g in config["goals"] if g["type"] == "position"]
    self.nominal_target_pos_pelvis = [
      np.array(g["vector"], dtype=np.float64) for g in position_goals
    ]
    self.nominal_target_pos = [np.zeros(3, dtype=np.float64) for _ in position_goals]
    self.nominal_motion_indices = [int(g["motion_index"]) for g in position_goals]
    self.target_motion_idx = self.nominal_motion_indices[0]
    self.motion_names = [
      os.path.basename(os.path.dirname(mp)) for mp in config["motion_paths"]
    ]

    self.pelvis_ekf = SE3EKF( # TODO TUNE THESE NOISE VALUES
      process_noise_pos=0.05,
      process_noise_rot=0.05,
      meas_noise_pos=0.1,
      meas_noise_rot=0.1,
    )
    self._last_predict_time: float | None = None

    self.lock = threading.Lock()

    # subscribers
    self.ball_pos_sub = self.create_subscription(
      PoseStamped, "/ball/pose", self.ball_state_callback, 10
    )
    self.ball_vel_sub = self.create_subscription(
      Vector3Stamped, "/ball/velocity", self.ball_velocity_callback, 10
    )
    self.pelvis_pos_sub = self.create_subscription(
      PoseStamped, "/g1_pelvis/pose", self.pelvis_pose_callback, 10
    )

    # publishers
    self.ball_pose_pub = self.create_publisher(PoseStamped, "/ball/target_pose", 10)
    self.pelvis_est_pub = self.create_publisher(PoseStamped, "/g1_pelvis/filtered_pose", 10)
    self.ball_target_time = self.create_publisher(Float64, "/ball/target_time", 10)
    self.ball_trajectory_pub = self.create_publisher(
      Float32MultiArray, "/ball/trajectory", 10
    )
    self.which_motion = self.create_publisher(Float64, "/ball/which_motion", 10)

    self.create_timer(self.dt, self.timer_callback)

    print("Ball estimator node initialized.")

  # callback: ball position [x, y, z] in world frame
  def ball_state_callback(self, msg: PoseStamped):
    z = np.array(
      [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64
    )
    self.ball_pos = z.copy()
    if not self.ball_initialized:
      self.ball_initialized = True

  # callback: ball velocity [vx, vy, vz] in world frame from KF in marker_bundle_to_ball_point
  def ball_velocity_callback(self, msg: Vector3Stamped):
    self.ball_vel = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=np.float64)


  # callback: pelvis pose in world frame — EKF update then recompute targets
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
    for i, body_pos in enumerate(self.nominal_target_pos_pelvis):
      t = 2.0 * np.cross(q_vec, body_pos)
      self.nominal_target_pos[i] = (
        body_pos + qw * t + np.cross(q_vec, t) + self.pelvis_pos
      )

  # Find where the ball's predicted trajectory crosses a plane 0.5 m in front of
  # the pelvis (defined in the pelvis forward direction), then choose the nominal
  # target point and motion whose pelvis-frame position is closest to that
  # intersection.
  def estimate_target_point(self):
    # --- pelvis forward axis in world frame (body +x rotated by pelvis quat) ---
    q_vec = self.pelvis_quat[:3]
    qw = self.pelvis_quat[3]
    forward_body = np.array([1.0, 0.0, 0.0])
    t_fwd = 2.0 * np.cross(q_vec, forward_body)
    forward_world = forward_body + qw * t_fwd + np.cross(q_vec, t_fwd)
    norm = np.linalg.norm(forward_world)
    if norm > 1e-8:
      forward_world /= norm

    # --- intercept plane: normal n, anchor p0 ---
    n = forward_world
    p0 = self.pelvis_pos + self.intercept_plane_dist * forward_world

    # --- find first trajectory segment that crosses the plane ---
    intersection_world = None
    intersection_time = -1.0

    intersection = first_plane_intersection(
      self.ball_trajectory_times, self.ball_trajectory_positions, p0, n
    )
    if intersection is not None:
      intersection_world = intersection.position
      intersection_time = intersection.time

    if intersection_world is not None:
      # convert intersection point from world → pelvis frame (inverse rotation)
      v = intersection_world - self.pelvis_pos
      t_inv = 2.0 * np.cross(q_vec, v)
      intersection_pelvis = v - qw * t_inv + np.cross(q_vec, t_inv)

      # pick the nominal target whose pelvis-frame position is closest
      dists = [
        np.sum((intersection_pelvis - wp) ** 2)
        for wp in self.nominal_target_pos_pelvis
      ]
      best_target_idx = int(np.argmin(dists))
      best_dist = float(np.sqrt(dists[best_target_idx]))

      self.target_motion_idx = self.nominal_motion_indices[best_target_idx]

      if best_dist > 1.0:  # if the intersection point is more than 1 meter from every nominal target, and it's more than 0.5 seconds in the future, then it's probably a bad estimate due to noise or an unexpected bounce, so ignore it and use the nominal target instead
        # intersection is too far from every nominal target — use the nominal
        # position directly so the robot doesn't reach for an unreachable point
        self.target_pos = self.nominal_target_pos_pelvis[best_target_idx].copy()
        self.target_time = -1.0
        print(
          f"Intercept point {best_dist:.2f} m from nearest target; "
          "using nominal target:", self.target_pos
        )
      else:
        self.target_pos = intersection_pelvis.copy()  # actual intercept point, not nominal
        self.target_time = intersection_time
        print(
          f"Using intercept point at {intersection_world} (pelvis frame: {intersection_pelvis}), time {intersection_time:.2f} s, distance to nearest nominal target {best_dist:.2f} m")
        print(f"motion for this target: {self.motion_names[self.target_motion_idx]}")

        # TODO OFFSETS HERE ARE HACKED, FIND OUT WHY AND FIX PROPERLY
        self.target_pos[2] += (
          0.00  # nudge target slightly above the estimated intercept point
        )
    else:
      # Trajectory does not cross the intercept plane (e.g. ball moving away).
      # Fall back to the nominal target closest to the current ball position.
      dists = [np.sum((self.ball_pos - wp) ** 2) for wp in self.nominal_target_pos]
      best_target_idx = int(np.argmin(dists))

      self.target_pos = self.nominal_target_pos_pelvis[best_target_idx].copy()
      self.target_motion_idx = self.nominal_motion_indices[best_target_idx]
      self.target_time = -1.0
      print(
        "Ball trajectory does not intersect intercept plane; "
        "using nominal target:", self.target_pos
      )

  def estimate_ball_trajectory(self):
    self.ball_trajectory_times, self.ball_trajectory_positions = (
      predict_ball_trajectory(
        self.ball_pos,
        self.ball_vel,
        duration=self.ball_trajectory_time_length,
        dt=self.dt,
      )
    )

  def timer_callback(self):
    now = self.get_clock().now().nanoseconds * 1e-9
    if self._last_predict_time is not None:
      dt = now - self._last_predict_time
      if 0.0 < dt < 1.0:
        self.pelvis_ekf.predict(dt)
    self._last_predict_time = now

    if self.pelvis_ekf.initialized:
      self.publish_pelvis_estimate()

    if not self.ball_initialized:
      return
    self.estimate_ball_trajectory()
    self.estimate_target_point()
    self.publish_pose()
    self.publish_target_time()
    self.publish_trajectory()
    self.publish_which_motion()

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
    print(f"Publishing target pose: {self.target_pos}, time {self.target_time:.2f} s, motion: {self.motion_names[self.target_motion_idx]}")
    msg = PoseStamped()
    msg.header.stamp = self.get_clock().now().to_msg()
    # The estimator API publishes the intercept directly in the pelvis frame.
    msg.header.frame_id = "pelvis"
    msg.pose.position.x = self.target_pos[0]  # in pelvis frame (for all of these)
    msg.pose.position.y = self.target_pos[1]
    msg.pose.position.z = self.target_pos[2]
    msg.pose.orientation.w = 1.0
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    self.ball_pose_pub.publish(msg)

  def publish_target_time(self):
    # publish the time until the ball reaches the target point
    msg = Float64()
    msg.data = self.target_time
    self.ball_target_time.publish(msg)

  def publish_trajectory(self):
    msg = Float32MultiArray()
    msg.data = self.ball_trajectory_positions.flatten().tolist()
    self.ball_trajectory_pub.publish(msg)

  def publish_which_motion(self):
    msg = Float64()
    msg.data = float(self.target_motion_idx)
    self.which_motion.publish(msg)


############################################################################
# MAIN FUNCTION
############################################################################


def main(args=None):
  rclpy.init()

  parser = argparse.ArgumentParser(description="Ball pose estimator node.")
  parser.add_argument(
    "--config",
    default="g1_athlete_tennis.yaml",
    help="Deployment YAML path or config filename.",
  )
  args = parser.parse_args()

  node = BallEstimatorNode(args.config)

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
