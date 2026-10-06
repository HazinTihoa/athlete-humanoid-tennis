"""Lightweight shared MuJoCo tennis-court assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import mujoco
from athlete.scripts.tennis_physics import STANDARD_TENNIS_PHYSICS

REPO_ROOT = Path(__file__).resolve().parents[4]
COURT_XML_PATH = REPO_ROOT / "deploy/simulation/assets/tennis_court.xml"
PHYSICS = STANDARD_TENNIS_PHYSICS
COURT_NET_X = PHYSICS.court.net_x_m
COURT_NET_HALF_WIDTH = PHYSICS.court.net_half_width_m
ROBOT_TERRAIN_COLLISION_BIT = 1 << 1
RACKET_NOMINAL_MASS_KG = 0.3
RACKET_NOMINAL_COM_WRIST_M = (0.38, 0.01, 0.27)
RACKET_NOMINAL_INERTIA_KG_M2 = (0.006, 0.012, 0.008)
RACKET_COLLISION_POS_WRIST_M = (0.38, -0.003, 0.27)
RACKET_COLLISION_QUAT_WXYZ = (
  -0.2753522,
  0.2753522,
  0.6512919,
  0.6512919,
)
# MuJoCo ellipsoid size values are semi-axes, not full dimensions.
RACKET_COLLISION_HALF_SIZE_M = (0.12, 0.17, 0.012)


def _copy_court_visual_settings(
  scene_spec: mujoco.MjSpec, court_spec: mujoco.MjSpec
) -> None:
  scene_spec.visual.global_.offwidth = court_spec.visual.global_.offwidth
  scene_spec.visual.global_.offheight = court_spec.visual.global_.offheight
  scene_spec.visual.global_.azimuth = court_spec.visual.global_.azimuth
  scene_spec.visual.global_.elevation = court_spec.visual.global_.elevation
  scene_spec.visual.quality.shadowsize = court_spec.visual.quality.shadowsize
  scene_spec.visual.quality.offsamples = court_spec.visual.quality.offsamples
  scene_spec.visual.map.znear = court_spec.visual.map.znear
  scene_spec.visual.map.zfar = court_spec.visual.map.zfar
  scene_spec.visual.map.fogstart = court_spec.visual.map.fogstart
  scene_spec.visual.map.fogend = court_spec.visual.map.fogend
  scene_spec.visual.rgba.haze[:] = court_spec.visual.rgba.haze


def attach_tennis_court_spec(
  scene_spec: mujoco.MjSpec,
  *,
  mode: Literal["visual", "physical"],
  court_contact_damping: float,
  net_x_m: float = COURT_NET_X,
  net_half_width_m: float = COURT_NET_HALF_WIDTH,
  racket_contact_damping: float | None = None,
  ball_geom_name: str = "tennis_ball/tennis_ball_geom",
  racket_geom_name: str = "robot/racket_ball_collision",
  robot_geom_names: tuple[str, ...] | None = None,
) -> None:
  """Attach the exact court XML and contact pairs used by Train and Play."""
  if net_x_m <= 0.0:
    raise ValueError("Tennis net X must be positive.")
  if net_half_width_m <= 0.0:
    raise ValueError("Tennis net half-width must be positive.")
  court_spec = mujoco.MjSpec.from_file(str(COURT_XML_PATH))
  court_ball = court_spec.body("tennis_ball")
  if court_ball is None:
    raise ValueError(f"Court XML has no tennis_ball body: {COURT_XML_PATH}")
  court_spec.delete(court_ball)

  net_geom_names = (
    "net_collision",
    "net_horizontal_1",
    "net_horizontal_2",
    "net_horizontal_3",
  )
  for geom_name in net_geom_names:
    geom = court_spec.geom(geom_name)
    if geom is None:
      raise ValueError(f"Court XML has no {geom_name!r} geom")
    geom.size[1] = net_half_width_m
  net_tape = court_spec.geom("net_tape")
  if net_tape is None:
    raise ValueError("Court XML has no 'net_tape' geom")
  net_tape.size[1] = net_half_width_m + 0.035
  for geom_name, side in (("net_post_left", -1.0), ("net_post_right", 1.0)):
    geom = court_spec.geom(geom_name)
    if geom is None:
      raise ValueError(f"Court XML has no {geom_name!r} geom")
    geom.pos[1] = side * (net_half_width_m + 0.065)

  if mode == "visual":
    for geom in court_spec.geoms:
      geom.contype = 0
      geom.conaffinity = 0
  else:
    for geom_name in ("surround", "court_surface"):
      geom = court_spec.geom(geom_name)
      if geom is None:
        raise ValueError(f"Court XML has no {geom_name!r} geom")
      geom.contype = 0
      geom.conaffinity = 0

  _copy_court_visual_settings(scene_spec, court_spec)
  court_spec.option.timestep = scene_spec.option.timestep
  court_spec.option.integrator = scene_spec.option.integrator
  court_spec.option.iterations = scene_spec.option.iterations
  court_spec.option.cone = scene_spec.option.cone
  court_spec.nconmax = scene_spec.nconmax
  court_spec.njmax = scene_spec.njmax

  frame = scene_spec.worldbody.add_frame(pos=[net_x_m, 0.0, 0.0])
  scene_spec.attach(court_spec, prefix="tennis_court/", frame=frame)

  old_terrain = scene_spec.geom("terrain")
  if old_terrain is not None:
    old_terrain.rgba = [0.0, 0.0, 0.0, 0.0]
  if mode == "visual":
    return

  if old_terrain is not None:
    old_terrain.contype = ROBOT_TERRAIN_COLLISION_BIT
    old_terrain.conaffinity = 0
    for geom in scene_spec.geoms:
      is_robot_geom = (
        geom.name.startswith("robot/")
        if robot_geom_names is None
        else geom.name in robot_geom_names
      )
      if is_robot_geom and (geom.contype != 0 or geom.conaffinity != 0):
        geom.conaffinity |= ROBOT_TERRAIN_COLLISION_BIT

  court_solref = (
    PHYSICS.court.court_solref_time_constant_s,
    court_contact_damping,
  )
  racket_solref = PHYSICS.court.racket_solref
  if racket_contact_damping is not None:
    racket_solref = (racket_solref[0], racket_contact_damping)
  pair_specs = (
    (
      "court",
      "tennis_court/court_surface",
      court_solref,
      PHYSICS.court.court_solimp,
      PHYSICS.court.court_friction,
    ),
    (
      "surround",
      "tennis_court/surround",
      court_solref,
      PHYSICS.court.court_solimp,
      PHYSICS.court.surround_friction,
    ),
    (
      "net",
      "tennis_court/net_collision",
      PHYSICS.court.net_solref,
      PHYSICS.court.net_solimp,
      PHYSICS.court.net_friction,
    ),
    (
      "racket",
      racket_geom_name,
      racket_solref,
      PHYSICS.ball.geom_solimp,
      PHYSICS.court.racket_friction,
    ),
  )
  for pair_name, other_geom, solref, solimp, friction in pair_specs:
    scene_spec.add_pair(
      name=f"tennis_ball_{pair_name}",
      geomname1=ball_geom_name,
      geomname2=other_geom,
      condim=3,
      solref=solref,
      solimp=solimp,
      friction=friction,
    )
