"""Interactively tune the MuJoCo tennis-racket collision ellipsoid."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import signal
import time

import mujoco
import mujoco.viewer
import numpy as np

from athlete.scripts.tennis_court_spec import (
  RACKET_COLLISION_HALF_SIZE_M,
  RACKET_COLLISION_POS_WRIST_M,
  RACKET_COLLISION_QUAT_WXYZ,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROBOT_XML = REPO_ROOT / "robots/replay_unitree_description/mjcf/g1.xml"
COLLIDER_NAME = "racket_ball_collision"
COLLIDER_RGBA = (1.0, 0.12, 0.02, 0.48)


def full_cm_to_half_m(full_dimensions_cm: np.ndarray) -> np.ndarray:
  """Convert full dimensions in centimeters to MuJoCo ellipsoid semi-axes."""
  dimensions = np.asarray(full_dimensions_cm, dtype=np.float64)
  if dimensions.shape != (3,) or not np.isfinite(dimensions).all():
    raise ValueError("Collider dimensions must contain three finite values.")
  if np.any(dimensions <= 0.0):
    raise ValueError("Collider dimensions must be positive.")
  return dimensions * 0.005


def half_m_to_full_cm(half_axes_m: np.ndarray) -> np.ndarray:
  """Convert MuJoCo ellipsoid semi-axes to full dimensions in centimeters."""
  half_axes = np.asarray(half_axes_m, dtype=np.float64)
  if half_axes.shape != (3,) or not np.isfinite(half_axes).all():
    raise ValueError("Collider half-axes must contain three finite values.")
  if np.any(half_axes <= 0.0):
    raise ValueError("Collider half-axes must be positive.")
  return half_axes * 200.0


def local_offset_to_wrist_position(
  base_pos: np.ndarray, quat_wxyz: np.ndarray, offset_cm: np.ndarray
) -> np.ndarray:
  """Translate along collider axes, then express the center in its parent body."""
  offset_cm = np.asarray(offset_cm, dtype=np.float64)
  if offset_cm.shape != (3,) or not np.isfinite(offset_cm).all():
    raise ValueError("Position offset must contain three finite values.")
  rotation = np.empty(9)
  mujoco.mju_quat2Mat(rotation, quat_wxyz)
  return np.asarray(base_pos) + rotation.reshape(3, 3) @ (offset_cm * 0.01)


def apply_collider_settings(model, data, geom_id, half_axes, position) -> None:
  model.geom_size[geom_id] = half_axes
  model.geom_pos[geom_id] = position
  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "racket_sweet_spot")
  if site_id >= 0:
    if model.site_bodyid[site_id] != model.geom_bodyid[geom_id]:
      raise ValueError("Sweet point and racket collider must share a parent body.")
    model.site_pos[site_id] = position
    model.site_quat[site_id] = model.geom_quat[geom_id]
  mujoco.mj_forward(model, data)


def draw_normal(scene, data, geom_id) -> None:
  scene.ngeom = 0
  center = data.geom_xpos[geom_id]
  normal = data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
  geom = scene.geoms[0]
  mujoco.mjv_initGeom(
    geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), center,
    np.eye(3).ravel(), np.array([0.05, 0.8, 1.0, 1.0], dtype=np.float32),
  )
  mujoco.mjv_connector(
    geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.003, center, center + 0.12 * normal
  )
  scene.ngeom = 1


def close_viewer(viewer) -> None:
  viewer.close()
  # Native close requests shutdown asynchronously. Wait for its weakref to
  # expire before Python's GLFW atexit cleanup can race the rendering thread.
  deadline = time.monotonic() + 2.0
  while viewer._sim() is not None and time.monotonic() < deadline:
    time.sleep(0.01)


def _build_model(robot_xml: Path) -> tuple[mujoco.MjModel, int]:
  spec = mujoco.MjSpec.from_file(str(robot_xml))
  wrist = spec.body("right_wrist_yaw_link")
  if wrist is None:
    raise ValueError(f"Robot XML has no right_wrist_yaw_link: {robot_xml}")
  existing = spec.geom(COLLIDER_NAME)
  if existing is None:
    wrist.add_geom(
      name=COLLIDER_NAME,
      type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
      pos=RACKET_COLLISION_POS_WRIST_M,
      quat=RACKET_COLLISION_QUAT_WXYZ,
      size=RACKET_COLLISION_HALF_SIZE_M,
      rgba=COLLIDER_RGBA,
      density=0.0,
      contype=1,
      conaffinity=1,
    )
  else:
    existing.rgba = COLLIDER_RGBA
  spec.worldbody.add_geom(
    name="racket_collider_tuner_floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(3.0, 3.0, 0.05),
    rgba=(0.16, 0.17, 0.19, 1.0),
    contype=0,
    conaffinity=0,
  )
  model = spec.compile()
  geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, COLLIDER_NAME)
  if geom_id < 0:
    raise RuntimeError(f"Compiled model has no geom {COLLIDER_NAME!r}.")
  if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_ELLIPSOID:
    raise ValueError("This tuner requires an ellipsoid racket collider.")
  return model, geom_id


def _set_display_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  data.qpos[:] = model.qpos0
  data.qvel[:] = 0.0
  pose = {
    "left_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_hip_pitch_joint": -0.312,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_elbow_joint": 0.6,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_elbow_joint": 0.6,
  }
  for name, value in pose.items():
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id >= 0:
      data.qpos[int(model.jnt_qposadr[joint_id])] = value
  mujoco.mj_forward(model, data)


class ColliderControlPanel:
  """Tk sliders exposing full collider dimensions while MuJoCo stays live."""

  def __init__(
    self,
    dimensions_cm: np.ndarray,
    on_change: Callable[[np.ndarray, np.ndarray], None],
    base_position: np.ndarray,
    quat_wxyz: np.ndarray,
    offset_cm: np.ndarray,
  ) -> None:
    import tkinter as tk
    from tkinter import ttk

    self._tk = tk
    self._on_change = on_change
    self._defaults = np.asarray(dimensions_cm, dtype=np.float64).copy()
    self._default_offset = np.asarray(offset_cm, dtype=np.float64).copy()
    self._base_position = np.asarray(base_position).copy()
    self._quat = np.asarray(quat_wxyz).copy()
    self._last_half_axes = full_cm_to_half_m(dimensions_cm)
    self._last_offset = self._default_offset.copy()
    self._last_position = local_offset_to_wrist_position(
      self._base_position, self._quat, offset_cm
    )
    self.closed = False

    self.root = tk.Tk()
    self.root.title("Racket collider tuner")
    self.root.geometry("720x600+24+48")
    self.root.minsize(640, 540)
    self.root.protocol("WM_DELETE_WINDOW", self.close)
    self.root.columnconfigure(0, weight=1)

    style = ttk.Style(self.root)
    style.configure("TunerTitle.TLabel", font=("Sans", 15, "bold"))
    style.configure("TunerValue.TLabel", font=("Monospace", 11))

    main = ttk.Frame(self.root, padding=24)
    main.grid(row=0, column=0, sticky="nsew")
    main.columnconfigure(1, weight=1)
    ttk.Label(main, text="Racket collision ellipsoid", style="TunerTitle.TLabel").grid(
      row=0, column=0, columnspan=4, sticky="w", pady=(0, 8)
    )
    ttk.Label(
      main,
      text="Controls show full dimensions. MuJoCo size uses half-axes.",
    ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 16))

    limits = (
      ("Short axis", 8.0, 35.0, 0.5),
      ("Long axis", 12.0, 50.0, 0.5),
      ("Thickness", 0.4, 6.0, 0.1),
    )
    self.variables = []
    for row, ((label, lower, upper, increment), value) in enumerate(
      zip(limits, dimensions_cm, strict=True), start=2
    ):
      variable = tk.DoubleVar(value=round(float(value), 4))
      self.variables.append(variable)
      ttk.Label(main, text=label, width=12).grid(row=row, column=0, sticky="w", pady=7)
      ttk.Scale(
        main,
        from_=lower,
        to=upper,
        variable=variable,
        command=self._changed,
      ).grid(row=row, column=1, sticky="ew", padx=(8, 12), pady=7)
      ttk.Spinbox(
        main,
        from_=lower,
        to=upper,
        increment=increment,
        textvariable=variable,
        width=8,
        command=self._changed,
      ).grid(row=row, column=2, sticky="e", pady=7)
      ttk.Label(main, text="cm").grid(row=row, column=3, sticky="w", padx=(6, 0))

    ttk.Separator(main).grid(row=5, column=0, columnspan=4, sticky="ew", pady=12)
    self.offset_variables = []
    for row, (label, value) in enumerate(zip(
      ("Offset X", "Offset Y", "Normal Z"), offset_cm, strict=True
    ), start=6):
      variable = tk.DoubleVar(value=float(value))
      self.offset_variables.append(variable)
      ttk.Label(main, text=label, width=12).grid(row=row, column=0, sticky="w", pady=7)
      ttk.Scale(main, from_=-5.0, to=5.0, variable=variable).grid(
        row=row, column=1, sticky="ew", padx=(8, 12), pady=7
      )
      ttk.Spinbox(
        main, from_=-5.0, to=5.0, increment=0.05,
        textvariable=variable, width=8,
      ).grid(row=row, column=2, sticky="e", pady=7)
      ttk.Label(main, text="cm").grid(row=row, column=3, sticky="w", padx=(6, 0))

    self.readout = tk.StringVar()
    ttk.Label(main, textvariable=self.readout, style="TunerValue.TLabel").grid(
      row=9, column=0, columnspan=4, sticky="w", pady=(18, 12)
    )
    buttons = ttk.Frame(main)
    buttons.grid(row=10, column=0, columnspan=4, sticky="ew")
    buttons.columnconfigure((0, 1), weight=1)
    ttk.Button(buttons, text="Print value", command=self.print_value).grid(
      row=0, column=0, sticky="ew", padx=(0, 5)
    )
    ttk.Button(buttons, text="Restore defaults", command=self.restore_defaults).grid(
      row=0, column=1, sticky="ew", padx=(5, 0)
    )
    for variable in self.variables + self.offset_variables:
      variable.trace_add("write", self._changed)
    self._changed()
    self.root.update_idletasks()
    width = min(self.root.winfo_reqwidth() + 24, self.root.winfo_screenwidth() - 64)
    height = min(self.root.winfo_reqheight() + 24, self.root.winfo_screenheight() - 96)
    self.root.geometry(f"{width}x{height}+24+48")
    self.root.minsize(width, height)

  def dimensions_cm(self) -> np.ndarray:
    return np.asarray([variable.get() for variable in self.variables])

  def _changed(self, *_args) -> None:
    try:
      half_axes = full_cm_to_half_m(self.dimensions_cm())
      offset = np.array([v.get() for v in self.offset_variables])
      position = local_offset_to_wrist_position(self._base_position, self._quat, offset)
      self.readout.set(
        "size = [" + ", ".join(f"{v:.5f}" for v in half_axes) + "] m\n"
        "pos  = [" + ", ".join(f"{v:.5f}" for v in position) + "] m (wrist)"
      )
      self._on_change(half_axes, position)
      self._last_half_axes = half_axes.copy()
      self._last_position = position.copy()
      self._last_offset = offset.copy()
    except (ValueError, self._tk.TclError):
      self.readout.set("Invalid dimensions or position; keeping last valid values")

  def print_value(self) -> None:
    # Retain numeric values even after Tk widgets have been destroyed on close.
    print(
      "[RACKET COLLIDER] size=["
      + ", ".join(f"{value:.6f}" for value in self._last_half_axes)
      + "]\npos=[" + ", ".join(f"{v:.6f}" for v in self._last_position)
      + "]  # meters in right_wrist_yaw_link\nquat=["
      + ", ".join(f"{v:.8f}" for v in self._quat)
      + "]  # wxyz\nlocal_offset_cm=["
      + ", ".join(f"{v:.4f}" for v in self._last_offset) + "]",
      flush=True,
    )

  def restore_defaults(self) -> None:
    for variable, value in zip(self.variables, self._defaults, strict=True):
      variable.set(round(float(value), 4))
    for variable, value in zip(self.offset_variables, self._default_offset, strict=True):
      variable.set(float(value))

  def poll(self) -> bool:
    if self.closed:
      return False
    try:
      self.root.update_idletasks()
      self.root.update()
    except self._tk.TclError:
      self.closed = True
    return not self.closed

  def close(self) -> None:
    if self.closed:
      return
    self.closed = True
    try:
      self.root.destroy()
    except self._tk.TclError:
      pass


def _parse_args() -> argparse.Namespace:
  defaults_cm = half_m_to_full_cm(np.asarray(RACKET_COLLISION_HALF_SIZE_M))
  parser = argparse.ArgumentParser(
    description="Visually tune the racket collision ellipsoid in native MuJoCo."
  )
  parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
  parser.add_argument("--short-axis-cm", type=float, default=float(defaults_cm[0]))
  parser.add_argument("--long-axis-cm", type=float, default=float(defaults_cm[1]))
  parser.add_argument("--thickness-cm", type=float, default=float(defaults_cm[2]))
  parser.add_argument("--offset-x-cm", type=float, default=0.0)
  parser.add_argument("--offset-y-cm", type=float, default=0.0)
  parser.add_argument("--normal-offset-cm", type=float, default=0.0)
  parser.add_argument("--headless", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  robot_xml = args.robot_xml.expanduser().resolve()
  if not robot_xml.is_file():
    raise FileNotFoundError(f"Robot XML not found: {robot_xml}")
  initial_dimensions_cm = np.asarray(
    (args.short_axis_cm, args.long_axis_cm, args.thickness_cm), dtype=np.float64
  )
  initial_half_axes = full_cm_to_half_m(initial_dimensions_cm)
  model, geom_id = _build_model(robot_xml)
  base_position = model.geom_pos[geom_id].copy()
  quat_wxyz = model.geom_quat[geom_id].copy()
  offset_cm = np.array([args.offset_x_cm, args.offset_y_cm, args.normal_offset_cm])
  initial_position = local_offset_to_wrist_position(base_position, quat_wxyz, offset_cm)
  data = mujoco.MjData(model)
  _set_display_pose(model, data)
  apply_collider_settings(model, data, geom_id, initial_half_axes, initial_position)
  print(
    "[RACKET COLLIDER] initial size=["
    + ", ".join(f"{value:.4f}" for value in initial_half_axes)
    + "] m; pos=[" + ", ".join(f"{v:.6f}" for v in initial_position) + "] m (wrist)",
    flush=True,
  )
  if args.headless:
    return

  requested_half_axes = initial_half_axes.copy()
  requested_position = initial_position.copy()

  def request_update(half_axes: np.ndarray, position: np.ndarray) -> None:
    requested_half_axes[:] = half_axes
    requested_position[:] = position

  panel = ColliderControlPanel(
    initial_dimensions_cm, request_update, base_position, quat_wxyz, offset_cm
  )
  stop_requested = False

  def request_stop(_signum: int, _frame: object) -> None:
    nonlocal stop_requested
    stop_requested = True

  previous_sigint_handler = signal.signal(signal.SIGINT, request_stop)
  viewer = None
  try:
    with mujoco.viewer.launch_passive(
      model,
      data,
      show_left_ui=False,
      show_right_ui=False,
    ) as viewer:
      wrist_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
      )
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      viewer.cam.trackbodyid = wrist_id
      viewer.cam.distance = 1.15
      viewer.cam.azimuth = 135.0
      viewer.cam.elevation = -12.0
      while viewer.is_running() and not stop_requested and panel.poll():
        with viewer.lock():
          apply_collider_settings(
            model, data, geom_id, requested_half_axes, requested_position
          )
          draw_normal(viewer.user_scn, data, geom_id)
        viewer.sync()
        time.sleep(1.0 / 60.0)
  finally:
    signal.signal(signal.SIGINT, previous_sigint_handler)
    if viewer is not None:
      close_viewer(viewer)
    panel.print_value()
    panel.close()


if __name__ == "__main__":
  main()
