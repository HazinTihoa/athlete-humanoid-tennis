##
#
# Ball estimator visualization node.
#
# Live 3D plot: pelvis, ball, target point, ball history, and estimated trajectory.
#
##

import threading
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import os
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

DEPLOY_DIR = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(DEPLOY_DIR)

HISTORY_LEN = 100


class BallEstimatorPlottingNode(Node):
  def __init__(self):
    super().__init__("ball_estimator_plotting_node")

    self.lock = threading.Lock()
    self.ball_pos: np.ndarray | None = None
    self.pelvis_pos: np.ndarray | None = None
    self.pelvis_quat: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])
    self.target_pos_body: np.ndarray | None = None  # in pelvis frame
    self.trajectory: np.ndarray | None = None  # shape (N, 3)
    self.ball_history: deque[tuple[float, np.ndarray]] = deque(maxlen=HISTORY_LEN)

    self.create_subscription(PoseStamped, "/ball/pose", self.ball_callback, 10)
    self.create_subscription(PoseStamped, "/g1_pelvis/filtered_pose", self.pelvis_callback, 10)
    self.create_subscription(PoseStamped, "/ball/target_pose", self.target_callback, 10)
    self.create_subscription(
      Float32MultiArray, "/ball/trajectory", self.trajectory_callback, 10
    )

  def ball_callback(self, msg: PoseStamped):
    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    with self.lock:
      self.ball_pos = pos
      self.ball_history.append((t, pos.copy()))

  def pelvis_callback(self, msg: PoseStamped):
    with self.lock:
      self.pelvis_pos = np.array(
        [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
      )
      self.pelvis_quat = np.array(
        [
          msg.pose.orientation.x,
          msg.pose.orientation.y,
          msg.pose.orientation.z,
          msg.pose.orientation.w,
        ]
      )

  def target_callback(self, msg: PoseStamped):
    with self.lock:
      self.target_pos_body = np.array(
        [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
      )

  def trajectory_callback(self, msg: Float32MultiArray):
    data = np.array(msg.data, dtype=np.float64)
    with self.lock:
      self.trajectory = data.reshape(-1, 3) if len(data) >= 3 else None

  def get_state(self):
    with self.lock:
      ball = self.ball_pos.copy() if self.ball_pos is not None else None
      pelvis = self.pelvis_pos.copy() if self.pelvis_pos is not None else None
      quat = self.pelvis_quat.copy()
      target = None
      if self.target_pos_body is not None and self.pelvis_pos is not None:
        q_vec = self.pelvis_quat[:3]
        qw = self.pelvis_quat[3]
        t = 2.0 * np.cross(q_vec, self.target_pos_body)
        target = self.target_pos_body + qw * t + np.cross(q_vec, t) + self.pelvis_pos
      traj = self.trajectory.copy() if self.trajectory is not None else None
      history = list(self.ball_history)
    return ball, pelvis, quat, target, traj, history


def main():
  rclpy.init()
  node = BallEstimatorPlottingNode()

  ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
  ros_thread.start()

  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection="3d")

  def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
      [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
      [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
      [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])

  def update(_frame):
    ball, pelvis, quat, target, traj, history = node.get_state()
    ax.cla()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Ball Estimator")

    if history:
      hist = np.array([h[1] for h in history])
      ax.plot(
        hist[:, 0],
        hist[:, 1],
        hist[:, 2],
        "b.",
        markersize=3,
        alpha=0.5,
        label="Ball history",
      )

    if traj is not None and len(traj) > 1:
      ax.plot(
        traj[:, 0],
        traj[:, 1],
        traj[:, 2],
        "c--",
        linewidth=1.5,
        label="Estimated trajectory",
      )

    if ball is not None:
      ax.scatter(*ball, c="green", s=80, marker="o", label="Ball", zorder=5)

    if pelvis is not None:
      ax.scatter(*pelvis, c="orange", s=120, marker="s", label="Pelvis", zorder=5)
      R = _quat_to_rotmat(quat)
      L = 0.3  # axis arrow length in metres
      for col, axis in zip(("red", "green", "blue"), R.T):
        ax.quiver(
          pelvis[0], pelvis[1], pelvis[2],
          axis[0] * L, axis[1] * L, axis[2] * L,
          color=col, linewidth=1.5, arrow_length_ratio=0.2,
        )

      # --- intercept plane: 0.5 m in front of the pelvis (+x body axis) ---
      forward_world = R[:, 0]            # body +x in world frame
      p0 = pelvis + 0.5 * forward_world  # plane anchor
      # two orthonormal vectors spanning the plane
      ref = np.array([0.0, 0.0, 1.0]) if abs(forward_world[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
      u_vec = np.cross(forward_world, ref)
      u_vec /= np.linalg.norm(u_vec)
      v_vec = np.cross(forward_world, u_vec)
      HALF = 0.5  # half-size of the displayed plane square (metres)
      ss = np.linspace(-HALF, HALF, 6)
      tt = np.linspace(-HALF, HALF, 6)
      S, T = np.meshgrid(ss, tt)
      px = p0[0] + S * u_vec[0] + T * v_vec[0]
      py = p0[1] + S * u_vec[1] + T * v_vec[1]
      pz = p0[2] + S * u_vec[2] + T * v_vec[2]
      ax.plot_surface(px, py, pz, alpha=0.25, color="purple")
      ax.scatter(*p0, c="purple", s=40, marker="x", zorder=5)  # plane centre

    if target is not None:
      ax.scatter(*target, c="red", s=150, marker="*", label="Target", zorder=5)

    center = ball if ball is not None else pelvis
    if center is not None:
      r = 4.0
      ax.set_xlim(center[0] - r, center[0] + r)
      ax.set_ylim(center[1] - r, center[1] + r)
      ax.set_zlim(0, max(center[2] + r, 2.0))

    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(facecolor="purple", alpha=0.4, label="Intercept plane"))
    labels.append("Intercept plane")
    ax.legend(handles=handles, labels=labels, loc="upper left", fontsize=8)
    return []

  ani = animation.FuncAnimation(fig, update, interval=50, blit=False)  # noqa: F841
  plt.show()

  node.destroy_node()
  rclpy.shutdown()


if __name__ == "__main__":
  main()
