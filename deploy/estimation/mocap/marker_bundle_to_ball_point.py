import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped

GRAVITY = 9.81
ROBOT_MARKER_EXCL_RADIUS = 0.2  # metres
MARKER_OUTLIER_DISTANCE_THRESHOLD = 0.2  # metres; reject ball detections that are further than this from the KF prediction
NUM_ROBOT_MARKERS = 9
KF_POS_VAR_THRESHOLD = (
  0.01  # m^2; physics filter inactive until all position variances drop below this
)


class BallTracker(Node):
  def __init__(self):
    super().__init__("ball_tracker")

    # robot rigid-body marker positions in world frame (updated from topics)
    self.robot_marker_pos: list[np.ndarray | None] = [None] * NUM_ROBOT_MARKERS

    # Kalman filter state: [x, y, z, vx, vy, vz]
    self.kf_state = np.zeros(6, dtype=np.float64)
    self.kf_P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    self.kf_Q = np.diag([1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2])
    self.kf_R = np.eye(3, dtype=np.float64) * 0.01
    self.kf_initialized = False
    self.kf_last_time: float | None = None

    # observation matrix: position only
    self.H = np.zeros((3, 6), dtype=np.float64)
    self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

    # for outlier rejection, we can keep a short history of recent ball detections, and reject new detections that are too far from the moving average of the detections
    self.filter_history_length = 10
    self.filter_positions = np.zeros((self.filter_history_length, 3), dtype=np.float64)
    self.stdevs_to_reject = 3.0

    self.pending_points: list[np.ndarray] = []

    for i in range(NUM_ROBOT_MARKERS):
      self.create_subscription(
        PointStamped,
        f"/g1_pelvis/marker{i}/pose",
        lambda msg, idx=i: self._robot_marker_callback(msg, idx),
        10,
      )

    self.sub = self.create_subscription(
      PointCloud, "/pointcloud", self.pointcloud_callback, 10
    )
    self.pub = self.create_publisher(PoseStamped, "/ball/pose", 10)
    self.vel_pub = self.create_publisher(Vector3Stamped, "/ball/velocity", 10)
    self.create_timer(0.02, self.timer_callback)

  def _robot_marker_callback(self, msg: PointStamped, idx: int):
    # print(f"Received robot marker {idx} position")
    p = msg.point
    self.robot_marker_pos[idx] = np.array([p.x, p.y, p.z], dtype=np.float64)

  def _is_robot_marker(self, p) -> bool:
    for rmp in self.robot_marker_pos:
      if rmp is None:
        # print(f"Warning: robot marker position not received yet, cannot exclude robot markers from ball tracking")
        continue
      dx = p.x - rmp[0]
      dy = p.y - rmp[1]
      dz = p.z - rmp[2]
      #   print(f"Checking marker at ({p.x:.2f}, {p.y:.2f}, {p.z:.2f}) against robot marker at ({rmp[0]:.2f}, {rmp[1]:.2f}, {rmp[2]:.2f}), distance squared = {dx*dx + dy*dy + dz*dz:.4f}")
      if (dx * dx + dy * dy + dz * dz) < ROBOT_MARKER_EXCL_RADIUS**2:
        # print(f"Excluding robot marker at ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")
        return True
    # print(f"Not a robot marker: ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")
    return False

  def pointcloud_callback(self, msg: PointCloud):
    for p in msg.points:
      if self._is_robot_marker(p):
        continue
      self.pending_points.append(np.array([p.x, p.y, p.z], dtype=np.float64))

  def timer_callback(self):
    now = self.get_clock().now().nanoseconds * 1e-9
    candidates = self.pending_points
    self.pending_points = []

    if not self.kf_initialized:
      if candidates:
        self.kf_state[:3] = np.mean(candidates, axis=0)
        self.kf_initialized = True
        self.kf_last_time = now
        self._publish_estimate(self.get_clock().now().to_msg())
      return

    if candidates:
      # centroid of all visible non-robot markers
      z = np.mean(candidates, axis=0)
      self._kf_update(z, now)
    else:
      self._kf_predict(now)

    self._publish_estimate(self.get_clock().now().to_msg())

  def _kf_predict(self, now: float):
    if self.kf_last_time is None:
      self.kf_last_time = now
      return
    dt = now - self.kf_last_time
    if dt <= 0:
      return
    self.kf_last_time = now

    F = np.eye(6, dtype=np.float64)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    u = np.array([0.0, 0.0, -0.5 * GRAVITY * dt**2, 0.0, 0.0, -GRAVITY * dt])
    self.kf_state = F @ self.kf_state + u
    self.kf_P = F @ self.kf_P @ F.T + self.kf_Q

  def _kf_update(self, z: np.ndarray, now: float):
    if self.kf_last_time is None:
      self.kf_last_time = now
      return
    dt = now - self.kf_last_time
    if dt <= 0:
      return
    self.kf_last_time = now

    F = np.eye(6, dtype=np.float64)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    # u = np.array([0.0, 0.0, -0.5 * GRAVITY * dt**2, 0.0, 0.0, -GRAVITY * dt])
    u = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # no control input during update step
    x_pred = F @ self.kf_state + u
    P_pred = F @ self.kf_P @ F.T + self.kf_Q

    innov = z - self.H @ x_pred
    S = self.H @ P_pred @ self.H.T + self.kf_R

    K = P_pred @ self.H.T @ np.linalg.inv(S)
    self.kf_state = x_pred + K @ innov
    self.kf_P = (np.eye(6) - K @ self.H) @ P_pred

    # if float(innov @ np.linalg.inv(S) @ innov) <= MAHALANOBIS_GATE:
    #   K = P_pred @ self.H.T @ np.linalg.inv(S)
    #   self.kf_state = x_pred + K @ innov
    #   self.kf_P = (np.eye(6) - K @ self.H) @ P_pred
    # else:
    #   self.kf_state = x_pred
    #   self.kf_P = P_pred

  def _publish_estimate(self, stamp):
    out = PoseStamped()
    out.header.stamp = stamp
    out.header.frame_id = "world"
    out.pose.position.x = self.kf_state[0]
    out.pose.position.y = self.kf_state[1]
    out.pose.position.z = 0.0 #self.kf_state[2] #TODO should this be a value other than zero? 
    out.pose.orientation.w = 1.0
    self.pub.publish(out)

    vel_out = Vector3Stamped()
    vel_out.header.stamp = stamp
    vel_out.header.frame_id = "world"
    vel_out.vector.x = self.kf_state[3]
    vel_out.vector.y = self.kf_state[4]
    vel_out.vector.z = self.kf_state[5]
    self.vel_pub.publish(vel_out)


def main():
  rclpy.init()
  node = BallTracker()
  rclpy.spin(node)
  node.destroy_node()
  rclpy.shutdown()


if __name__ == "__main__":
  main()
