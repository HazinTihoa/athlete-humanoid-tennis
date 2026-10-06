#!/usr/bin/env python3
"""Visualize a physical tennis ball and the deploy trajectory prediction in MuJoCo."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from itertools import pairwise
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from athlete.scripts.tennis_physics import (
  STANDARD_TENNIS_PHYSICS,
  contact_damping_for_restitution,
  tennis_ball_aerodynamic_wrench_numpy,
)

from deploy.estimation.filtering.tennis_ballistics import (
  COEFF_OF_RESTITUTION,
  first_ground_contact,
  first_plane_intersection,
  predict_ball_trajectory,
)
from deploy.simulation.tennis_launch_ui import (
  LaunchControlPanel,
  speed_angles_from_velocity,
)

PHYSICS = STANDARD_TENNIS_PHYSICS
BALL_RADIUS = PHYSICS.ball.radius_m
COURT_HALF_LENGTH = 11.885
COURT_HALF_WIDTH = 5.485
SERVICE_LINE_X = 6.40
SINGLES_HALF_WIDTH = 4.115
NET_HEIGHT = 0.914

PREDICTION_COLOR = np.array([0.05, 0.85, 1.0, 0.90], dtype=np.float32)
HISTORY_COLOR = np.array([1.0, 0.84, 0.10, 0.95], dtype=np.float32)
PREDICTED_BOUNCE_COLOR = np.array([1.0, 0.34, 0.04, 0.95], dtype=np.float32)
ACTUAL_BOUNCE_COLOR = np.array([0.95, 0.05, 0.06, 0.95], dtype=np.float32)
INTERCEPT_COLOR = np.array([0.95, 0.15, 0.85, 0.95], dtype=np.float32)
PLANE_COLOR = np.array([0.75, 0.18, 0.85, 0.42], dtype=np.float32)
LAUNCH_COLOR = np.array([0.20, 1.0, 0.35, 0.85], dtype=np.float32)
LAUNCH_DIRECTION_COLOR = np.array([0.10, 1.0, 0.25, 0.95], dtype=np.float32)
MEASUREMENT_COLOR = np.array([1.0, 1.0, 1.0, 0.95], dtype=np.float32)
IDENTITY = np.eye(3, dtype=np.float64).reshape(-1)


def _build_standard_tennis_model(xml_path: Path) -> mujoco.MjModel:
  """Compile the visual court with the shared ball/contact parameters."""
  spec = mujoco.MjSpec.from_file(str(xml_path))
  ball_geom = spec.geom("ball_geom")
  if ball_geom is None:
    raise RuntimeError("ball_geom is missing from the MuJoCo court model")
  ball_geom.size = [PHYSICS.ball.radius_m, 0.0, 0.0]
  ball_geom.mass = PHYSICS.ball.mass_kg
  ball_geom.friction = PHYSICS.ball.geom_friction
  ball_geom.solref = PHYSICS.ball.geom_solref
  ball_geom.solimp = PHYSICS.ball.geom_solimp
  ball_geom.rgba = PHYSICS.ball.rgba

  court_damping = contact_damping_for_restitution(PHYSICS.court.restitution)
  for name, other_geom, solref, solimp, friction in (
    (
      "standard_tennis_ball_court",
      "court_surface",
      (PHYSICS.court.court_solref_time_constant_s, court_damping),
      PHYSICS.court.court_solimp,
      PHYSICS.court.court_friction,
    ),
    (
      "standard_tennis_ball_surround",
      "surround",
      (PHYSICS.court.court_solref_time_constant_s, court_damping),
      PHYSICS.court.court_solimp,
      PHYSICS.court.surround_friction,
    ),
    (
      "standard_tennis_ball_net",
      "net_collision",
      PHYSICS.court.net_solref,
      PHYSICS.court.net_solimp,
      PHYSICS.court.net_friction,
    ),
  ):
    spec.add_pair(
      name=name,
      geomname1="ball_geom",
      geomname2=other_geom,
      condim=3,
      solref=solref,
      solimp=solimp,
      friction=friction,
    )
  return spec.compile()


class TennisTrajectoryDemo:
  """Own the physical simulation, deploy prediction, and viewer overlays."""

  def __init__(self, args: argparse.Namespace):
    self.args = args
    self.xml_path = Path(args.xml).expanduser().resolve()
    self.model = _build_standard_tennis_model(self.xml_path)
    self.data = mujoco.MjData(self.model)

    joint_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint"
    )
    if joint_id < 0:
      raise RuntimeError("ball_freejoint is missing from the MuJoCo model")
    self.ball_qpos_adr = int(self.model.jnt_qposadr[joint_id])
    self.ball_dof_adr = int(self.model.jnt_dofadr[joint_id])
    self.ball_geom_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"
    )
    self.ball_body_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_BODY, "tennis_ball"
    )
    if self.ball_body_id < 0:
      raise RuntimeError("tennis_ball body is missing from the MuJoCo model")
    self.court_geom_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_GEOM, "court_surface"
    )

    self.launch_position = np.asarray(args.launch_position, dtype=np.float64)
    self.launch_velocity = np.asarray(args.launch_velocity, dtype=np.float64)
    model_ball_mass = float(self.model.body_mass[self.ball_body_id])
    self.ball_mass_kg = (
      model_ball_mass if args.ball_mass_g is None else 0.001 * args.ball_mass_g
    )
    self.position_noise_std = float(args.position_noise_std)
    if not np.isfinite(self.position_noise_std) or self.position_noise_std < 0.0:
      raise ValueError("position noise standard deviation must be non-negative")
    self.noise_rng = np.random.default_rng(args.noise_seed)
    self.measured_ball_position = self.launch_position.copy()
    self.measurement_true_position = self.launch_position.copy()
    self.receiver_position = np.asarray(args.receiver_position, dtype=np.float64)
    receiver_forward = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    self.intercept_plane_normal = receiver_forward
    self.intercept_plane_point = (
      self.receiver_position + args.intercept_plane_distance * receiver_forward
    )

    history_capacity = max(2, int(args.history_seconds / args.visual_dt) + 1)
    self.actual_history: deque[np.ndarray] = deque(maxlen=history_capacity)
    self.prediction_times = np.zeros(0, dtype=np.float64)
    self.prediction_positions = np.zeros((0, 3), dtype=np.float64)
    self.predicted_bounce = None
    self.predicted_intercept = None
    self.initial_predicted_bounce = None
    self.actual_bounce_position: np.ndarray | None = None
    self.actual_bounce_time: float | None = None
    self.bounce_outgoing_vz: float | None = None
    self.net_crossing_z: float | None = None
    self._last_history_time = -np.inf
    self._last_prediction_time = -np.inf
    self._court_contact_active = False
    self._relaunch_requested = False
    self._pending_launch: (
      tuple[np.ndarray, np.ndarray, float, float] | None
    ) = None
    self.rally_start_time = 0.0
    self.reset()

  @property
  def ball_position(self) -> np.ndarray:
    return self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 3].copy()

  @property
  def ball_velocity(self) -> np.ndarray:
    return self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 3].copy()

  @property
  def rally_time(self) -> float:
    return float(self.data.time - self.rally_start_time)

  def _set_ball_mass(self, mass_kg: float) -> None:
    mass_kg = float(mass_kg)
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
      raise ValueError("ball mass must be a positive finite value")
    self.ball_mass_kg = mass_kg
    sphere_inertia = 0.4 * mass_kg * BALL_RADIUS**2
    self.model.body_mass[self.ball_body_id] = mass_kg
    self.model.body_inertia[self.ball_body_id] = sphere_inertia
    mujoco.mj_setConst(self.model, self.data)

  def reset(self) -> None:
    self._set_ball_mass(self.ball_mass_kg)
    mujoco.mj_resetData(self.model, self.data)
    qpos = self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 7]
    qpos[:3] = self.launch_position
    qpos[3:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    qvel = self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 6]
    qvel[:] = 0.0
    qvel[:3] = self.launch_velocity
    mujoco.mj_forward(self.model, self.data)

    self.rally_start_time = float(self.data.time)
    self.actual_history.clear()
    self.actual_history.append(self.launch_position.copy())
    self.actual_bounce_position = None
    self.actual_bounce_time = None
    self.bounce_outgoing_vz = None
    self.net_crossing_z = None
    self._last_history_time = self.data.time
    self._last_prediction_time = -np.inf
    self._court_contact_active = False
    self.update_prediction(force=True)
    self.initial_predicted_bounce = self.predicted_bounce

  def request_launch(
    self,
    position: np.ndarray,
    velocity: np.ndarray,
    ball_mass_kg: float | None = None,
    position_noise_std: float | None = None,
  ) -> None:
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,):
      raise ValueError("launch position and velocity must both have shape (3,)")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
      raise ValueError("launch position and velocity must contain finite values")
    if position[2] < BALL_RADIUS:
      raise ValueError(f"launch Z must be at least {BALL_RADIUS:.4f} m")

    mass_kg = self.ball_mass_kg if ball_mass_kg is None else float(ball_mass_kg)
    noise_std = (
      self.position_noise_std
      if position_noise_std is None
      else float(position_noise_std)
    )
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
      raise ValueError("ball mass must be a positive finite value")
    if not np.isfinite(noise_std) or noise_std < 0.0:
      raise ValueError("position noise standard deviation must be non-negative")
    self._pending_launch = (
      position.copy(),
      velocity.copy(),
      mass_kg,
      noise_std,
    )

  def _apply_launch_request(self) -> bool:
    if self._pending_launch is not None:
      (
        self.launch_position,
        self.launch_velocity,
        self.ball_mass_kg,
        self.position_noise_std,
      ) = self._pending_launch
      self._pending_launch = None
    elif not self._relaunch_requested:
      return False
    self._relaunch_requested = False
    self.reset()
    return True

  def update_prediction(self, *, force: bool = False) -> None:
    if (
      not force
      and self.data.time - self._last_prediction_time
      < self.args.prediction_update_dt
    ):
      return
    self._last_prediction_time = float(self.data.time)
    true_position = self.ball_position
    self.measurement_true_position = true_position
    if self.position_noise_std > 0.0:
      position_noise = self.noise_rng.normal(
        0.0, self.position_noise_std, size=3
      )
      self.measured_ball_position = true_position + position_noise
    else:
      self.measured_ball_position = true_position
    self.prediction_times, self.prediction_positions = predict_ball_trajectory(
      self.measured_ball_position,
      self.ball_velocity,
      duration=self.args.prediction_horizon,
      dt=self.args.prediction_dt,
      restitution=self.args.estimator_restitution,
      ground_z=BALL_RADIUS,
    )
    self.predicted_bounce = first_ground_contact(
      self.prediction_times,
      self.prediction_positions,
      ground_z=BALL_RADIUS,
    )
    self.predicted_intercept = first_plane_intersection(
      self.prediction_times,
      self.prediction_positions,
      self.intercept_plane_point,
      self.intercept_plane_normal,
    )

  def _ball_touches_court(self) -> bool:
    for contact in self.data.contact[: self.data.ncon]:
      geom_pair = {int(contact.geom1), int(contact.geom2)}
      if geom_pair == {self.ball_geom_id, self.court_geom_id}:
        return True
    return False

  def _apply_ball_aerodynamics(self) -> None:
    velocity = self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 6]
    force_w, torque_w = tennis_ball_aerodynamic_wrench_numpy(
      velocity[:3], velocity[3:6]
    )
    self.data.xfrc_applied[self.ball_body_id, :3] = force_w
    self.data.xfrc_applied[self.ball_body_id, 3:6] = torque_w

  def step(self) -> None:
    previous_position = self.ball_position
    previous_velocity = self.ball_velocity
    self._apply_ball_aerodynamics()
    mujoco.mj_step(self.model, self.data)
    position = self.ball_position
    velocity = self.ball_velocity

    if previous_position[0] < 0.0 <= position[0] and self.net_crossing_z is None:
      alpha = -previous_position[0] / (position[0] - previous_position[0])
      self.net_crossing_z = float(
        previous_position[2] + alpha * (position[2] - previous_position[2])
      )

    court_contact = self._ball_touches_court()
    vertical_rebound = (
      previous_velocity[2] < 0.0 <= velocity[2]
      and position[2] < BALL_RADIUS + 0.08
    )
    new_court_contact = court_contact and not self._court_contact_active
    if (
      self.actual_bounce_position is None
      and (new_court_contact or vertical_rebound)
    ):
      self.actual_bounce_position = position.copy()
      self.actual_bounce_position[2] = BALL_RADIUS
      self.actual_bounce_time = self.rally_time
      if vertical_rebound:
        self.bounce_outgoing_vz = float(velocity[2])
    if (
      self.actual_bounce_position is not None
      and self.bounce_outgoing_vz is None
      and self._court_contact_active
      and not court_contact
      and velocity[2] > 0.0
    ):
      self.bounce_outgoing_vz = float(velocity[2])
    self._court_contact_active = court_contact

    if self.data.time - self._last_history_time >= self.args.visual_dt:
      self.actual_history.append(position.copy())
      self._last_history_time = float(self.data.time)
    self.update_prediction()

  @staticmethod
  def _add_sphere(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    radius: float,
    color: np.ndarray,
  ) -> None:
    if scene.ngeom >= scene.maxgeom:
      return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
      geom,
      mujoco.mjtGeom.mjGEOM_SPHERE,
      np.array([radius, 0.0, 0.0], dtype=np.float64),
      np.asarray(position, dtype=np.float64),
      IDENTITY,
      color,
    )
    scene.ngeom += 1

  @staticmethod
  def _add_line(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    color: np.ndarray,
  ) -> None:
    if scene.ngeom >= scene.maxgeom:
      return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
      geom,
      mujoco.mjtGeom.mjGEOM_LINE,
      np.zeros(3, dtype=np.float64),
      np.zeros(3, dtype=np.float64),
      IDENTITY,
      color,
    )
    mujoco.mjv_connector(
      geom,
      mujoco.mjtGeom.mjGEOM_LINE,
      width,
      np.asarray(start, dtype=np.float64),
      np.asarray(end, dtype=np.float64),
    )
    geom.rgba[:] = color
    scene.ngeom += 1

  @staticmethod
  def _add_arrow(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    color: np.ndarray,
  ) -> None:
    if scene.ngeom >= scene.maxgeom:
      return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
      geom,
      mujoco.mjtGeom.mjGEOM_ARROW,
      np.zeros(3, dtype=np.float64),
      np.asarray(start, dtype=np.float64),
      IDENTITY,
      color,
    )
    mujoco.mjv_connector(
      geom,
      mujoco.mjtGeom.mjGEOM_ARROW,
      width,
      np.asarray(start, dtype=np.float64),
      np.asarray(end, dtype=np.float64),
    )
    geom.rgba[:] = color
    scene.ngeom += 1

  @classmethod
  def _add_polyline(
    cls,
    scene: mujoco.MjvScene,
    points: np.ndarray,
    width: float,
    color: np.ndarray,
    max_segments: int,
  ) -> None:
    if len(points) < 2:
      return
    stride = max(1, int(np.ceil((len(points) - 1) / max_segments)))
    sampled = points[::stride]
    if not np.array_equal(sampled[-1], points[-1]):
      sampled = np.vstack((sampled, points[-1]))
    for start, end in pairwise(sampled):
      cls._add_line(scene, start, end, width, color)

  def draw_prediction(self, scene: mujoco.MjvScene, *, clear: bool) -> None:
    if clear:
      scene.ngeom = 0

    self._add_polyline(
      scene,
      self.prediction_positions,
      width=3.0,
      color=PREDICTION_COLOR,
      max_segments=130,
    )
    if len(self.prediction_positions):
      marker_stride = max(1, len(self.prediction_positions) // 22)
      for point in self.prediction_positions[::marker_stride]:
        self._add_sphere(scene, point, 0.018, PREDICTION_COLOR)

    if self.position_noise_std > 0.0:
      self._add_line(
        scene,
        self.measurement_true_position,
        self.measured_ball_position,
        2.0,
        MEASUREMENT_COLOR,
      )
      self._add_sphere(
        scene, self.measured_ball_position, 0.045, MEASUREMENT_COLOR
      )

    history = np.asarray(self.actual_history, dtype=np.float64)
    self._add_polyline(
      scene,
      history,
      width=4.0,
      color=HISTORY_COLOR,
      max_segments=120,
    )
    self._add_sphere(scene, self.launch_position, 0.055, LAUNCH_COLOR)
    launch_speed = float(np.linalg.norm(self.launch_velocity))
    if launch_speed > 1e-8:
      launch_arrow_start = self.launch_position + np.array([0.0, 0.0, 0.18])
      launch_arrow_end = (
        launch_arrow_start + 1.8 * self.launch_velocity / launch_speed
      )
      self._add_line(
        scene,
        self.launch_position,
        launch_arrow_start,
        3.0,
        LAUNCH_DIRECTION_COLOR,
      )
      self._add_arrow(
        scene,
        launch_arrow_start,
        launch_arrow_end,
        0.05,
        LAUNCH_DIRECTION_COLOR,
      )

    if self.predicted_bounce is not None:
      self._add_sphere(
        scene,
        self.predicted_bounce.position,
        0.12,
        PREDICTED_BOUNCE_COLOR,
      )
    if self.actual_bounce_position is not None:
      self._add_sphere(
        scene,
        self.actual_bounce_position,
        0.10,
        ACTUAL_BOUNCE_COLOR,
      )
    if self.predicted_intercept is not None:
      self._add_sphere(
        scene,
        self.predicted_intercept.position,
        0.11,
        INTERCEPT_COLOR,
      )

    # Deploy's target estimator searches the first trajectory crossing of this
    # receiver-relative plane. The rectangle makes that otherwise invisible
    # estimation surface explicit in the demo.
    x = self.intercept_plane_point[0]
    corners = np.array(
      [
        [x, -2.2, 0.05],
        [x, 2.2, 0.05],
        [x, 2.2, 2.6],
        [x, -2.2, 2.6],
      ],
      dtype=np.float64,
    )
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
      self._add_line(scene, start, end, 2.0, PLANE_COLOR)

  def _viewer_text(self) -> list[tuple[int, int, str, str]]:
    position = self.ball_position
    velocity = self.ball_velocity
    launch_speed, launch_azimuth, launch_elevation = speed_angles_from_velocity(
      self.launch_velocity
    )
    measurement_error = float(
      np.linalg.norm(self.measured_ball_position - self.measurement_true_position)
    )
    predicted_bounce = (
      "none"
      if self.predicted_bounce is None
      else f"{self.predicted_bounce.position.round(3)} @ {self.predicted_bounce.time:.2f}s"
    )
    intercept = (
      "none"
      if self.predicted_intercept is None
      else f"{self.predicted_intercept.position.round(3)} @ {self.predicted_intercept.time:.2f}s"
    )
    actual_bounce = (
      "pending"
      if self.actual_bounce_position is None
      else f"{self.actual_bounce_position.round(3)} @ {self.actual_bounce_time:.2f}s"
    )
    left = (
      "Time\nBall xyz\nBall velocity\nLaunch xyz\nLaunch speed\n"
      "Launch azimuth / elevation\nBall mass\nPosition noise sigma\n"
      "Measurement error\nPredicted bounce\nEstimated intercept\n"
      "Actual first bounce"
    )
    right = (
      f"{self.rally_time:.2f} s\n"
      f"{position.round(3)}\n"
      f"{velocity.round(3)}\n"
      f"{self.launch_position.round(3)}\n"
      f"{launch_speed:.2f} m/s\n"
      f"{launch_azimuth:.1f} / {launch_elevation:.1f} deg\n"
      f"{1000.0 * self.ball_mass_kg:.1f} g\n"
      f"{self.position_noise_std:.3f} m\n"
      f"{measurement_error:.3f} m\n"
      f"{predicted_bounce}\n"
      f"{intercept}\n"
      f"{actual_bounce}"
    )
    legend = (
      "Cyan: deploy prediction   Yellow: physical history\n"
      "Orange: predicted bounce   Red: actual bounce\n"
      "Magenta: predicted intercept / receiver plane\n"
      "Green: active launch point / direction\n"
      "White: noisy position measurement\n"
      "Space or R: relaunch"
    )
    font = int(mujoco.mjtFontScale.mjFONTSCALE_150)
    return [
      (font, int(mujoco.mjtGridPos.mjGRID_TOPLEFT), left, right),
      (font, int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT), legend, ""),
    ]

  @staticmethod
  def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.4, 0.0, 0.55], dtype=np.float64)
    camera.distance = 21.5
    camera.azimuth = 135.0
    camera.elevation = -30.0
    return camera

  def render_snapshot(self, output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(self.model, height=720, width=1280)
    try:
      renderer.update_scene(self.data, camera=self._camera())
      self.draw_prediction(renderer.scene, clear=False)
      pixels = renderer.render().copy()
    finally:
      renderer.close()
    from PIL import Image

    Image.fromarray(pixels).save(output_path)
    print(f"Saved render: {output_path}")

  def summary(self) -> dict[str, object]:
    predicted = (
      None
      if self.initial_predicted_bounce is None
      else self.initial_predicted_bounce.position.round(6).tolist()
    )
    actual = (
      None
      if self.actual_bounce_position is None
      else self.actual_bounce_position.round(6).tolist()
    )
    error = None
    if self.initial_predicted_bounce is not None and self.actual_bounce_position is not None:
      error = float(
        np.linalg.norm(
          self.initial_predicted_bounce.position[:2]
          - self.actual_bounce_position[:2]
        )
      )
    in_opponent_court = bool(
      self.actual_bounce_position is not None
      and 0.0 < self.actual_bounce_position[0] < COURT_HALF_LENGTH
      and abs(self.actual_bounce_position[1]) < COURT_HALF_WIDTH
    )
    in_service_box = bool(
      self.actual_bounce_position is not None
      and 0.0 < self.actual_bounce_position[0] < SERVICE_LINE_X
      and abs(self.actual_bounce_position[1]) < SINGLES_HALF_WIDTH
    )
    launch_speed, launch_azimuth, launch_elevation = speed_angles_from_velocity(
      self.launch_velocity
    )
    return {
      "launch_position": self.launch_position.tolist(),
      "launch_velocity": self.launch_velocity.tolist(),
      "launch_speed": launch_speed,
      "launch_azimuth_deg": launch_azimuth,
      "launch_elevation_deg": launch_elevation,
      "ball_mass_g": 1000.0 * self.ball_mass_kg,
      "position_noise_std": self.position_noise_std,
      "latest_position_measurement": self.measured_ball_position.tolist(),
      "latest_position_measurement_error": float(
        np.linalg.norm(
          self.measured_ball_position - self.measurement_true_position
        )
      ),
      "net_crossing_z": self.net_crossing_z,
      "net_clearance": (
        None if self.net_crossing_z is None else self.net_crossing_z - NET_HEIGHT
      ),
      "predicted_first_bounce": predicted,
      "actual_first_bounce": actual,
      "bounce_xy_error": error,
      "bounce_outgoing_vz": self.bounce_outgoing_vz,
      "landed_in_opponent_court": in_opponent_court,
      "landed_in_opponent_service_box": in_service_box,
    }

  def run_headless(self) -> None:
    render_saved = False
    end_time = float(self.data.time + self.args.duration)
    while self.data.time < end_time:
      self.step()
      if (
        self.args.render_output is not None
        and not render_saved
        and self.rally_time >= self.args.render_time
      ):
        self.render_snapshot(Path(self.args.render_output))
        render_saved = True
    if self.args.render_output is not None and not render_saved:
      self.render_snapshot(Path(self.args.render_output))
    print(json.dumps(self.summary(), indent=2))

  def run_viewer(self) -> None:
    import mujoco.viewer

    def key_callback(keycode: int) -> None:
      if keycode in (ord(" "), ord("R"), ord("r")):
        self._relaunch_requested = True

    viewer = mujoco.viewer.launch_passive(
      self.model,
      self.data,
      key_callback=key_callback,
      show_left_ui=False,
      show_right_ui=False,
    )
    viewer.cam.lookat[:] = self._camera().lookat
    viewer.cam.distance = self._camera().distance
    viewer.cam.azimuth = self._camera().azimuth
    viewer.cam.elevation = self._camera().elevation

    control_panel = None
    if self.args.control_ui:
      try:
        control_panel = LaunchControlPanel(
          self.launch_position,
          self.launch_velocity,
          on_launch=self.request_launch,
          ball_radius=BALL_RADIUS,
          ball_mass_kg=self.ball_mass_kg,
          position_noise_std=self.position_noise_std,
        )
      except Exception as exc:
        viewer.close()
        raise RuntimeError(
          "Could not create launch control UI; use --no-control-ui to disable it"
        ) from exc

    accumulator = 0.0
    previous_wall_time = time.perf_counter()
    render_saved = False
    try:
      while viewer.is_running():
        if control_panel is not None and not control_panel.poll():
          break
        now = time.perf_counter()
        accumulator += min(now - previous_wall_time, 0.05)
        previous_wall_time = now

        if self._apply_launch_request():
          accumulator = 0.0
        while accumulator >= self.model.opt.timestep:
          self.step()
          accumulator -= self.model.opt.timestep
          if self.args.loop and self.rally_time >= self.args.rally_duration:
            self.reset()
            accumulator = 0.0
            break

        with viewer.lock():
          self.draw_prediction(viewer.user_scn, clear=True)
        viewer.set_texts(self._viewer_text())
        viewer.sync()

        if (
          self.args.render_output is not None
          and not render_saved
          and self.rally_time >= self.args.render_time
        ):
          self.render_snapshot(Path(self.args.render_output))
          render_saved = True
        time.sleep(0.001)
    except KeyboardInterrupt:
      pass
    finally:
      viewer.close()
      if control_panel is not None:
        control_panel.close()
      print(json.dumps(self.summary(), indent=2))


def parse_args() -> argparse.Namespace:
  default_xml = Path(__file__).resolve().parent / "assets" / "tennis_court.xml"
  parser = argparse.ArgumentParser(
    description=(
      "Launch a physical tennis ball in a MuJoCo court and visualize the "
      "shared deploy trajectory/target estimate."
    )
  )
  parser.add_argument("--xml", default=str(default_xml), help="MuJoCo court XML")
  parser.add_argument(
    "--launch-position",
    type=float,
    nargs=3,
    default=(-6.4, -1.5, 1.2),
    metavar=("X", "Y", "Z"),
  )
  parser.add_argument(
    "--launch-velocity",
    type=float,
    nargs=3,
    default=(8.4, 0.8, 4.3),
    metavar=("VX", "VY", "VZ"),
  )
  parser.add_argument(
    "--ball-mass-g",
    type=float,
    default=None,
    help="physical ball mass in grams (defaults to the XML mass)",
  )
  parser.add_argument(
    "--position-noise-std",
    type=float,
    default=0.0,
    help="isotropic Gaussian position measurement sigma in metres",
  )
  parser.add_argument(
    "--noise-seed",
    type=int,
    default=0,
    help="random seed for position measurement noise",
  )
  parser.add_argument(
    "--receiver-position",
    type=float,
    nargs=3,
    default=(5.6, 0.0, 0.9),
    metavar=("X", "Y", "Z"),
  )
  parser.add_argument("--intercept-plane-distance", type=float, default=0.5)
  parser.add_argument("--prediction-horizon", type=float, default=5.0)
  parser.add_argument("--prediction-dt", type=float, default=0.02)
  parser.add_argument("--prediction-update-dt", type=float, default=0.02)
  parser.add_argument(
    "--estimator-restitution", type=float, default=COEFF_OF_RESTITUTION
  )
  parser.add_argument("--visual-dt", type=float, default=0.02)
  parser.add_argument("--history-seconds", type=float, default=4.0)
  parser.add_argument("--rally-duration", type=float, default=4.0)
  parser.add_argument(
    "--loop",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="automatically relaunch after rally-duration in the interactive viewer",
  )
  parser.add_argument(
    "--control-ui",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="show the position/speed/direction launch control window",
  )
  parser.add_argument("--headless", action="store_true")
  parser.add_argument(
    "--duration", type=float, default=2.5, help="headless simulation duration"
  )
  parser.add_argument("--render-output", help="optionally save one PNG frame")
  parser.add_argument("--render-time", type=float, default=0.75)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  demo = TennisTrajectoryDemo(args)
  if args.headless:
    demo.run_headless()
  else:
    demo.run_viewer()


if __name__ == "__main__":
  main()
