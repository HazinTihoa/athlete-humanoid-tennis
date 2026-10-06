"""Play a tracking policy against an estimated incoming tennis-ball trajectory."""

from __future__ import annotations

import copy
import sys
import tempfile
import traceback
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import cast

import mujoco
import numpy as np
import torch
import tyro
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim import Simulation
from mjlab.tasks.registry import list_tasks
from mjlab.utils.lab_api.math import quat_apply, quat_mul
from mjlab.viewer import NativeMujocoViewer, ViewerConfig
from mjlab.viewer.base import VerbosityLevel
from athlete.goal_cond_tracking.mdp.commands import (
  reference_frame0_alignment,
)
from athlete.goal_cond_tracking.mdp.phase_commands import (
  PhaseAccelerationMultiTargetMotionCommand,
)
from athlete.goal_cond_tracking.tennis_planner import (
  MotionReachTarget,
  MotionTrajectoryMatch,
  PlaneCrossing,
  StrokeSide,
  evaluate_motion_matches,
  filter_motion_matches,
  first_forward_plane_crossing,
  retain_motion_match,
)
from athlete.scripts.play import PlayConfig, build_play_session
from athlete.scripts.tennis_physics import (
  STANDARD_TENNIS_PHYSICS,
  contact_damping_for_restitution,
  tennis_ball_aerodynamic_wrench_torch,
)

from deploy.estimation.filtering.tennis_ballistics import (
  COEFF_OF_RESTITUTION,
  TrajectoryIntersection,
  first_ground_contact,
  predict_ball_trajectory,
)
from deploy.simulation.tennis_launch_ui import (
  LaunchControlPanel,
  speed_angles_from_velocity,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
COURT_XML_PATH = REPO_ROOT / "deploy/simulation/assets/tennis_court.xml"
PHYSICS = STANDARD_TENNIS_PHYSICS
COURT_NET_X = PHYSICS.court.net_x_m
BALL_RADIUS = PHYSICS.ball.radius_m
BALL_GEOM_SOLREF = PHYSICS.ball.geom_solref
BALL_RACKET_SOLREF = PHYSICS.court.racket_solref
BALL_GEOM_SOLIMP = PHYSICS.ball.geom_solimp
# These are MuJoCo's equal-priority mixes of the standalone ball and court
# materials. Explicit pairs preserve that response while the court floor geoms
# remain visual-only for ordinary collision filtering.
BALL_COURT_SOLREF = (PHYSICS.court.court_solref_time_constant_s, 0.0)
BALL_COURT_SOLIMP = PHYSICS.court.court_solimp
BALL_COURT_FRICTION = PHYSICS.court.court_friction
BALL_SURROUND_FRICTION = PHYSICS.court.surround_friction
BALL_NET_SOLREF = PHYSICS.court.net_solref
BALL_NET_SOLIMP = PHYSICS.court.net_solimp
BALL_NET_FRICTION = PHYSICS.court.net_friction
BALL_RACKET_FRICTION = PHYSICS.court.racket_friction
# Keep the robot on the exact infinite plane used during training while the
# tennis ball uses explicit pairs against the rendered court surface.
ROBOT_TERRAIN_COLLISION_BIT = 1 << 1
PREDICTION_COLOR = np.array([0.05, 0.85, 1.0, 0.90], dtype=np.float32)
HISTORY_COLOR = np.array([1.0, 0.84, 0.10, 0.95], dtype=np.float32)
PREDICTED_BOUNCE_COLOR = np.array([1.0, 0.34, 0.04, 0.95], dtype=np.float32)
ACTUAL_BOUNCE_COLOR = np.array([0.95, 0.05, 0.06, 0.95], dtype=np.float32)
LAUNCH_COLOR = np.array([0.20, 1.0, 0.35, 0.85], dtype=np.float32)
LAUNCH_DIRECTION_COLOR = np.array([0.10, 1.0, 0.25, 0.95], dtype=np.float32)
PLANE_COLOR = np.array([0.72, 0.18, 0.88, 0.50], dtype=np.float32)
CROSSING_COLOR = np.array([0.35, 0.45, 1.0, 0.95], dtype=np.float32)
MEAN_COLOR = np.array([0.95, 0.15, 0.85, 0.95], dtype=np.float32)
TARGET_COLOR = np.array([0.10, 1.0, 0.28, 0.95], dtype=np.float32)
MEASUREMENT_COLOR = np.array([1.0, 1.0, 1.0, 0.95], dtype=np.float32)
IDENTITY = np.eye(3, dtype=np.float64).reshape(-1)


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


def _standalone_ball_spec(spec_fn):
  """Apply the standalone demo ball material to the play-only ball."""

  def build_spec() -> mujoco.MjSpec:
    spec = spec_fn()
    geom = spec.geom("tennis_ball_geom")
    if geom is None:
      raise ValueError("Play tennis ball geom is missing.")
    geom.friction = PHYSICS.ball.geom_friction
    geom.solref = BALL_GEOM_SOLREF
    geom.solimp = BALL_GEOM_SOLIMP
    return spec

  return build_spec


def _attach_tennis_court(
  scene_spec: mujoco.MjSpec, court_contact_damping: float
) -> None:
  """Attach the standalone court without its duplicate ball body."""
  court_spec = mujoco.MjSpec.from_file(str(COURT_XML_PATH))
  court_ball = court_spec.body("tennis_ball")
  if court_ball is None:
    raise ValueError(f"Court XML has no tennis_ball body: {COURT_XML_PATH}")
  court_spec.delete(court_ball)

  # Court floor geoms are visual-only for normal collision filtering. Explicit
  # pairs below retain their physical response for the tennis ball.
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

  frame = scene_spec.worldbody.add_frame(pos=[COURT_NET_X, 0.0, 0.0])
  scene_spec.attach(court_spec, prefix="tennis_court/", frame=frame)

  old_terrain = scene_spec.geom("terrain")
  if old_terrain is not None:
    old_terrain.contype = ROBOT_TERRAIN_COLLISION_BIT
    old_terrain.conaffinity = 0
    old_terrain.rgba = [0.0, 0.0, 0.0, 0.0]

    # Robot collision geoms retain their original bit-0 interactions and also
    # accept contacts from the training terrain's dedicated bit. The ball is
    # deliberately excluded and reaches the court only through explicit pairs.
    for geom in scene_spec.geoms:
      if geom.name.startswith("robot/") and (
        geom.contype != 0 or geom.conaffinity != 0
      ):
        geom.conaffinity |= ROBOT_TERRAIN_COLLISION_BIT

  ball_geom = "tennis_ball/tennis_ball_geom"
  court_solref = (BALL_COURT_SOLREF[0], court_contact_damping)
  pair_specs = (
    (
      "court",
      "tennis_court/court_surface",
      court_solref,
      BALL_COURT_SOLIMP,
      BALL_COURT_FRICTION,
    ),
    (
      "surround",
      "tennis_court/surround",
      court_solref,
      BALL_COURT_SOLIMP,
      BALL_SURROUND_FRICTION,
    ),
    (
      "net",
      "tennis_court/net_collision",
      BALL_NET_SOLREF,
      BALL_NET_SOLIMP,
      BALL_NET_FRICTION,
    ),
    (
      "racket",
      "robot/racket_ball_collision",
      BALL_RACKET_SOLREF,
      BALL_GEOM_SOLIMP,
      BALL_RACKET_FRICTION,
    ),
  )
  for pair_name, other_geom, solref, solimp, friction in pair_specs:
    scene_spec.add_pair(
      name=f"tennis_ball_{pair_name}",
      geomname1=ball_geom,
      geomname2=other_geom,
      condim=3,
      solref=solref,
      solimp=solimp,
      friction=friction,
    )


def _configure_tennis_play_env(
  env_cfg, physical_restitution: float = PHYSICS.court.restitution
) -> None:
  court_contact_damping = contact_damping_for_restitution(
    physical_restitution
  )
  ball_entity = env_cfg.scene.entities.get("tennis_ball")
  if ball_entity is None or ball_entity.spec_fn is None:
    raise ValueError("Tennis court play requires the play-only physical ball.")
  ball_entity.spec_fn = _standalone_ball_spec(ball_entity.spec_fn)

  previous_scene_hook = env_cfg.scene.spec_fn

  def configure_scene(scene_spec: mujoco.MjSpec) -> None:
    if previous_scene_hook is not None:
      previous_scene_hook(scene_spec)
    _attach_tennis_court(scene_spec, court_contact_damping)

  env_cfg.scene.spec_fn = configure_scene
  env_cfg.scene.extent = 20.0
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.textures = ()
    env_cfg.scene.terrain.materials = ()
    env_cfg.scene.terrain.lights = ()

  # Keep the policy's training-time dynamics. The court contributes geometry,
  # materials, and visual settings, but must not replace the robot simulator's
  # timestep, integrator, solver, or control decimation.
  env_cfg.sim.nconmax = max(env_cfg.sim.nconmax or 0, 200)
  env_cfg.sim.njmax = max(env_cfg.sim.njmax or 0, 500)

  env_cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
  env_cfg.viewer.lookat = (COURT_NET_X + 0.4, 0.0, 0.55)
  env_cfg.viewer.distance = 21.5
  env_cfg.viewer.fovy = 45.0
  env_cfg.viewer.azimuth = 135.0
  env_cfg.viewer.elevation = -30.0
  env_cfg.viewer.width = 1280
  env_cfg.viewer.height = 720
  print(
    "[INFO]: Standalone tennis court enabled: "
    f"net_x=5.600 m, policy physics_dt={env_cfg.sim.mujoco.timestep:.3f} s, "
    f"control_dt={env_cfg.sim.mujoco.timestep * env_cfg.decimation:.3f} s, "
    f"physical_restitution={physical_restitution:.3f}, "
    f"contact_damping={court_contact_damping:.6f}"
  )


@dataclass(frozen=True)
class TennisEstimatorPlayConfig(PlayConfig):
  """Play configuration plus incoming-ball trajectory-planning controls."""

  launch_position: tuple[float, float, float] = (6.72, -1.0, 1.2)
  launch_velocity: tuple[float, float, float] = (-2.0, 0.0, 3.0)
  ball_mass_g: float = PHYSICS.ball.mass_kg * 1000.0
  position_noise_std: float = 0.0
  noise_seed: int = 0
  prediction_horizon: float = 5.0
  prediction_dt: float = 0.02
  prediction_update_dt: float = 0.02
  trajectory_source: str = "estimator"
  """`estimator` uses deploy ballistics; `oracle-mujoco` rolls out shadow physics."""
  estimator_restitution: float = COEFF_OF_RESTITUTION
  physical_restitution: float = PHYSICS.court.restitution
  """Calibrated target for physical ball-ground restitution."""
  side_deadband: float = 0.10
  match_distance: float = 0.20
  reaction_margin: float = 0.30
  """Minimum timing margin required when selecting a new motion."""
  trigger_tolerance: float = 0.03
  """Start the motion this many seconds before its exact trigger time."""
  motion_switch_hysteresis: float = 0.0
  control_ui: bool = True
  auto_launch: bool = False
  loop: bool = False
  history_seconds: float = 4.0
  headless_steps: int | None = None


class RallyState(str, Enum):
  WAITING = "waiting"
  TRACKING = "tracking"
  EXECUTING = "executing"
  COMPLETE = "complete"


@dataclass(frozen=True)
class LaunchRequest:
  position: np.ndarray
  velocity: np.ndarray
  mass_kg: float
  position_noise_std: float


class TennisEstimatorController:
  """Own incoming-ball state, fixed-frame planning, matching, and triggering."""

  def __init__(self, vec_env, cfg: TennisEstimatorPlayConfig) -> None:
    if vec_env.num_envs != 1:
      raise ValueError("Tennis estimator play requires exactly one environment.")
    self.vec_env = vec_env
    self.env = vec_env.unwrapped
    self.cfg = cfg
    if cfg.trajectory_source not in {"estimator", "oracle-mujoco"}:
      raise ValueError(
        "trajectory_source must be 'estimator' or 'oracle-mujoco', got "
        f"{cfg.trajectory_source!r}"
      )
    if cfg.reaction_margin < 0.0:
      raise ValueError("reaction_margin must be non-negative")
    if cfg.trigger_tolerance < 0.0:
      raise ValueError("trigger_tolerance must be non-negative")
    self.ball = self.env.scene["tennis_ball"]
    self.command = cast(
      PhaseAccelerationMultiTargetMotionCommand,
      self.env.command_manager.get_term("motion"),
    )
    if not isinstance(self.command, PhaseAccelerationMultiTargetMotionCommand):
      raise TypeError(
        "Tennis estimator play requires a phase-acceleration motion command."
      )

    self.position_slots = []
    self.motion_sides = []
    self.nominal_strike_times = []
    self.minimum_strike_times = []
    for motion_index, motion_cfg in enumerate(self.command.motion_configs):
      slots = [
        index
        for index, target in enumerate(motion_cfg.sub_targets)
        if target.goal_type == "position"
      ]
      if len(slots) != 1:
        raise ValueError(
          f"Motion {motion_index} must have exactly one position target, got {slots}."
        )
      self.position_slots.append(slots[0])
      name = motion_cfg.name.lower()
      if "forehand" in name:
        self.motion_sides.append(StrokeSide.FOREHAND)
      elif "backhand" in name:
        self.motion_sides.append(StrokeSide.BACKHAND)
      else:
        raise ValueError(
          f"Cannot infer forehand/backhand for motion {motion_index}: {motion_cfg.name}"
        )
      nominal = float(self.command._nominal_contact_times[motion_index].item())
      self.nominal_strike_times.append(nominal)
      self.minimum_strike_times.append(
        nominal / self.command.cfg.max_contact_speedup
      )

    history_capacity = max(2, round(cfg.history_seconds / self.env.step_dt) + 1)
    self.actual_history: deque[np.ndarray] = deque(maxlen=history_capacity)
    self.noise_rng = np.random.default_rng(cfg.noise_seed)
    self.state = RallyState.WAITING
    self.pending_launch: LaunchRequest | None = None
    self.last_launch = LaunchRequest(
      position=np.asarray(cfg.launch_position, dtype=np.float64),
      velocity=np.asarray(cfg.launch_velocity, dtype=np.float64),
      mass_kg=0.001 * cfg.ball_mass_g,
      position_noise_std=cfg.position_noise_std,
    )
    self.motion_targets: tuple[MotionReachTarget, ...] = ()
    self.planning_position_world: np.ndarray | None = None
    self.planning_quaternion_world: np.ndarray | None = None
    self.planning_forward_world: np.ndarray | None = None
    self.planning_lateral_world: np.ndarray | None = None
    self.planning_up_world: np.ndarray | None = None
    self.prediction_times = np.zeros(0, dtype=np.float64)
    self.prediction_positions = np.zeros((0, 3), dtype=np.float64)
    self._oracle_times_from_launch = np.zeros(0, dtype=np.float64)
    self._oracle_positions_from_launch = np.zeros((0, 3), dtype=np.float64)
    self._oracle_shadow_sim: Simulation | None = None
    self._oracle_ball_body_id: int | None = None
    self._oracle_ball_qpos_adr: int | None = None
    self._oracle_ball_dof_adr: int | None = None
    self.measured_ball_position = self.last_launch.position.copy()
    self.true_ball_position = self.last_launch.position.copy()
    self.ball_velocity = self.last_launch.velocity.copy()
    self.predicted_bounce: TrajectoryIntersection | None = None
    self.actual_bounce_position: np.ndarray | None = None
    self.actual_bounce_time: float | None = None
    self.actual_bounce_restitution: float | None = None
    self.crossing: PlaneCrossing | None = None
    self.evaluated_matches: list[MotionTrajectoryMatch] = []
    self.ranked_matches: list[MotionTrajectoryMatch] = []
    self.selected_match: MotionTrajectoryMatch | None = None
    self.locked_target_world: np.ndarray | None = None
    self._last_prediction_step = -1
    self._launch_step = 0
    self._previous_vertical_velocity = float(self.last_launch.velocity[2])
    self._last_descending_velocity: float | None = None
    self._reset_requested = False
    self._relaunch_after_reset: bool | None = None
    self._prepared_motion_index: int | None = None
    self._initial_robot_root_state: torch.Tensor | None = None
    self._initial_robot_joint_pos: torch.Tensor | None = None
    self._initialized = False

  @property
  def current_step(self) -> int:
    return int(self.env.common_step_counter)

  @property
  def rally_time(self) -> float:
    return max(0, self.current_step - self._launch_step) * self.env.step_dt

  @property
  def policy_enabled(self) -> bool:
    return self.state in {RallyState.EXECUTING, RallyState.COMPLETE}

  def initialize(self) -> None:
    if self._initialized:
      return
    self._initialized = True
    env_ids = torch.zeros(1, dtype=torch.long, device=self.env.device)
    self.command.set_motion_chaining(env_ids, enabled=False)
    self._hold_reference_at_phase_zero()
    self._capture_initial_robot_state()
    self.enforce_robot_freeze()
    self._align_waiting_reference_to_robot()
    self._park_ball()
    if self.cfg.auto_launch:
      self.pending_launch = self.last_launch

  def request_launch(
    self,
    position: np.ndarray,
    velocity: np.ndarray,
    mass_kg: float | None = None,
    position_noise_std: float | None = None,
  ) -> None:
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,):
      raise ValueError("launch position and velocity must have shape (3,)")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
      raise ValueError("launch state must contain finite values")
    if position[2] < BALL_RADIUS:
      raise ValueError(f"launch Z must be at least {BALL_RADIUS:.4f} m")
    mass = self.last_launch.mass_kg if mass_kg is None else float(mass_kg)
    noise = (
      self.last_launch.position_noise_std
      if position_noise_std is None
      else float(position_noise_std)
    )
    if not np.isfinite(mass) or mass <= 0.0:
      raise ValueError("ball mass must be positive")
    if not np.isfinite(noise) or noise < 0.0:
      raise ValueError("position noise standard deviation must be non-negative")
    self.pending_launch = LaunchRequest(position.copy(), velocity.copy(), mass, noise)

  def request_relaunch(self) -> None:
    self.pending_launch = self.last_launch

  def preview_launch_position(self, position: np.ndarray) -> None:
    """Update the next launch point and move the parked ball immediately."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
      raise ValueError("launch position must contain three finite values")
    if position[2] < BALL_RADIUS:
      raise ValueError(f"launch Z must be at least {BALL_RADIUS:.4f} m")
    self.last_launch = replace(self.last_launch, position=position.copy())
    if self.state is RallyState.WAITING:
      self.true_ball_position = position.copy()
      self.measured_ball_position = position.copy()
      self._write_ball_state(position, np.zeros(3, dtype=np.float64))

  def request_robot_reset(self) -> None:
    """Queue a full G1 and rally reset without automatically relaunching."""
    self.pending_launch = None
    self._relaunch_after_reset = False
    self._reset_requested = True
    print("[TENNIS] G1 reset queued")

  def _write_ball_state(self, position: np.ndarray, velocity: np.ndarray) -> None:
    state = torch.zeros((1, 13), dtype=torch.float32, device=self.env.device)
    state[0, :3] = torch.as_tensor(position, device=self.env.device)
    state[0, 3] = 1.0
    state[0, 7:10] = torch.as_tensor(velocity, device=self.env.device)
    self.ball.write_root_state_to_sim(state)
    self.env.sim.forward()

  def _set_ball_mass(self, mass_kg: float) -> None:
    body_id = int(self.ball.data.indexing.root_body_id)
    sphere_inertia = 0.4 * mass_kg * BALL_RADIUS**2
    warp_model = self.env.sim.model
    body_mass = warp_model.body_mass
    body_inertia = warp_model.body_inertia
    if body_mass.ndim == 2:
      body_mass[:, body_id] = mass_kg
      body_inertia[:, body_id, :] = sphere_inertia
    else:
      body_mass[body_id] = mass_kg
      body_inertia[body_id, :] = sphere_inertia

    cpu_model = self.env.sim.mj_model
    cpu_data = self.env.sim.mj_data
    if cpu_model is not None and cpu_data is not None:
      cpu_model.body_mass[body_id] = mass_kg
      cpu_model.body_inertia[body_id, :] = sphere_inertia
      mujoco.mj_setConst(cpu_model, cpu_data)

  def _ensure_oracle_shadow_sim(self) -> None:
    """Create an isolated MJWarp world with the live solver/model settings."""
    if self._oracle_shadow_sim is not None:
      return
    live_model = self.env.sim.mj_model
    if live_model is None:
      raise RuntimeError("oracle-mujoco requires an available CPU MuJoCo model")

    # The bindings expose data copying but not model copying. Serializing to a
    # temporary MJB gives the shadow rollout an independent mutable model.
    with tempfile.NamedTemporaryFile(suffix=".mjb") as model_file:
      mujoco.mj_saveModel(live_model, model_file.name)
      shadow_model = mujoco.MjModel.from_binary_path(model_file.name)

    ball_body_id = int(self.ball.data.indexing.root_body_id)
    ball_geom_ids = np.flatnonzero(shadow_model.geom_bodyid == ball_body_id)
    if len(ball_geom_ids) == 0:
      raise RuntimeError(
        "oracle-mujoco could not find a geom for the tennis ball body "
        f"id={ball_body_id}"
      )

    # Keep ball/court/net collision identical to the live world. Every other
    # geom belongs to G1 (or disabled base terrain) and is removed only from
    # the shadow model, so an unmoved waiting robot cannot corrupt the oracle.
    for geom_id in range(shadow_model.ngeom):
      geom_name = mujoco.mj_id2name(
        shadow_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
      )
      is_ball_or_court = geom_id in ball_geom_ids or (
        geom_name is not None and geom_name.startswith("tennis_court/")
      )
      if not is_ball_or_court:
        shadow_model.geom_contype[geom_id] = 0
        shadow_model.geom_conaffinity[geom_id] = 0

    ball_joint_id = int(shadow_model.body_jntadr[ball_body_id])
    if ball_joint_id < 0:
      raise RuntimeError("oracle-mujoco tennis ball body has no free joint")
    self._oracle_shadow_sim = Simulation(
      num_envs=1,
      cfg=copy.deepcopy(self.env.sim.cfg),
      model=shadow_model,
      device=self.env.device,
    )
    self._oracle_ball_body_id = ball_body_id
    self._oracle_ball_qpos_adr = int(shadow_model.jnt_qposadr[ball_joint_id])
    self._oracle_ball_dof_adr = int(shadow_model.jnt_dofadr[ball_joint_id])

  def _prepare_oracle_trajectory(self) -> None:
    """Roll ball/court physics forward from the exact launch state once."""
    self._ensure_oracle_shadow_sim()
    shadow_sim = self._oracle_shadow_sim
    ball_body_id = self._oracle_ball_body_id
    ball_qpos_adr = self._oracle_ball_qpos_adr
    ball_dof_adr = self._oracle_ball_dof_adr
    if (
      shadow_sim is None
      or ball_body_id is None
      or ball_qpos_adr is None
      or ball_dof_adr is None
    ):
      raise RuntimeError("oracle-mujoco simulation state is unavailable")

    # Copy the authoritative GPU state. The host `mj_data` mirror is stale
    # while MJWarp runs, so CPU MuJoCo cannot serve as an exact shadow. The
    # shadow uses the same MJWarp solver, timestep and dynamic model instead.
    shadow_sim.data.qpos[:] = self.env.sim.data.qpos
    shadow_sim.data.qvel[:] = self.env.sim.data.qvel
    live_mass = self.env.sim.model.body_mass
    live_inertia = self.env.sim.model.body_inertia
    shadow_mass = shadow_sim.model.body_mass
    shadow_inertia = shadow_sim.model.body_inertia
    if shadow_mass.ndim == 2:
      shadow_mass[:, ball_body_id] = live_mass[:, ball_body_id]
      shadow_inertia[:, ball_body_id, :] = live_inertia[:, ball_body_id, :]
    else:
      shadow_mass[ball_body_id] = live_mass[ball_body_id]
      shadow_inertia[ball_body_id, :] = live_inertia[ball_body_id, :]
    shadow_sim.recompute_constants(level=RecomputeLevel.set_const)
    shadow_sim.forward()

    horizon = self.cfg.prediction_horizon
    sample_dt = self.cfg.prediction_dt
    times = [0.0]
    positions = [shadow_sim.data.qpos[0, ball_qpos_adr : ball_qpos_adr + 3].clone()]
    physics_dt = self.env.sim.cfg.mujoco.timestep
    sample_steps = max(1, round(sample_dt / physics_dt))
    total_steps = int(np.ceil(horizon / physics_dt))
    for step in range(1, total_steps + 1):
      force_w, torque_w = tennis_ball_aerodynamic_wrench_torch(
        shadow_sim.data.qvel[:, ball_dof_adr : ball_dof_adr + 3],
        shadow_sim.data.qvel[:, ball_dof_adr + 3 : ball_dof_adr + 6],
      )
      shadow_sim.data.xfrc_applied[:, ball_body_id, :3] = force_w
      shadow_sim.data.xfrc_applied[:, ball_body_id, 3:6] = torque_w
      shadow_sim.step()
      if step % sample_steps != 0 and step != total_steps:
        continue
      times.append(min(step * physics_dt, horizon))
      positions.append(
        shadow_sim.data.qpos[0, ball_qpos_adr : ball_qpos_adr + 3].clone()
      )

    self._oracle_times_from_launch = np.asarray(times, dtype=np.float64)
    self._oracle_positions_from_launch = torch.stack(positions).cpu().numpy()
    print(
      "[TENNIS] Oracle MuJoCo trajectory "
      f"samples={len(times)} horizon={horizon:.2f}s dt={sample_dt:.3f}s"
    )

  def _oracle_prediction_from_current_time(self) -> tuple[np.ndarray, np.ndarray]:
    """Express the cached oracle path in seconds remaining from the current step."""
    elapsed = self.rally_time
    times = self._oracle_times_from_launch
    positions = self._oracle_positions_from_launch
    if len(times) == 0 or elapsed > times[-1]:
      return np.zeros(0, dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    first_future = int(np.searchsorted(times, elapsed, side="left"))
    current_position = np.array(
      [np.interp(elapsed, times, positions[:, axis]) for axis in range(3)],
      dtype=np.float64,
    )
    future_times = times[first_future:] - elapsed
    future_positions = positions[first_future:].copy()
    if len(future_times) == 0 or future_times[0] > 1e-10:
      future_times = np.concatenate(([0.0], future_times))
      future_positions = np.vstack((current_position, future_positions))
    else:
      future_times[0] = 0.0
      future_positions[0] = current_position
    return future_times, future_positions

  def _park_ball(self) -> None:
    self._write_ball_state(self.last_launch.position, np.zeros(3, dtype=np.float64))

  def _capture_planning_frame(self) -> None:
    planning_pos = self.command.robot_anchor_pos_w[0].detach()
    planning_quat = self.command.robot_anchor_quat_w[0].detach()
    num_motions = len(self.command.motion_configs)
    reference_quats = self.command._stacked_body_quat_w[
      :, 0, self.command.motion_anchor_body_index
    ]
    robot_positions = planning_pos.unsqueeze(0).expand(num_motions, -1)
    robot_quats = planning_quat.unsqueeze(0).expand(num_motions, -1)
    aligned_positions, align_yaws = reference_frame0_alignment(
      reference_quats,
      robot_positions,
      robot_quats,
    )
    aligned_positions = aligned_positions.clone()
    # XY/yaw follow the frozen G1 planning frame. Preserve each motion's
    # original frame-0 pelvis height so the selected pose keeps valid foot
    # contact instead of being vertically shifted into or above the court.
    reference_positions = (
      self.command._stacked_body_pos_w[
        :, 0, self.command.motion_anchor_body_index
      ]
      + self.env.scene.env_origins[0]
    )
    aligned_positions[:, 2] = reference_positions[:, 2]
    aligned_quats = quat_mul(align_yaws, reference_quats)
    local_means = torch.stack(
      [
        self.command._target_pos_means_t[index, self.position_slots[index]]
        for index in range(num_motions)
      ]
    )
    means_world = aligned_positions + quat_apply(aligned_quats, local_means)

    basis = torch.eye(3, dtype=planning_quat.dtype, device=self.env.device)
    repeated_quat = planning_quat.unsqueeze(0).expand(3, -1)
    axes_world = quat_apply(repeated_quat, basis)
    self.planning_position_world = planning_pos.cpu().numpy().copy()
    self.planning_quaternion_world = planning_quat.cpu().numpy().copy()
    self.planning_forward_world = axes_world[0].cpu().numpy().copy()
    self.planning_lateral_world = axes_world[1].cpu().numpy().copy()
    self.planning_up_world = axes_world[2].cpu().numpy().copy()
    means_np = means_world.detach().cpu().numpy()
    self.motion_targets = tuple(
      MotionReachTarget(
        motion_index=index,
        side=self.motion_sides[index],
        mean_world=means_np[index].copy(),
        nominal_strike_time=self.nominal_strike_times[index],
        minimum_strike_time=self.minimum_strike_times[index],
      )
      for index in range(num_motions)
    )

  def _apply_launch(self, request: LaunchRequest) -> None:
    self.last_launch = request
    self._set_ball_mass(request.mass_kg)
    # Freeze the exact live Play state at launch while trajectory planning runs.
    self._capture_initial_robot_state()
    self._capture_planning_frame()
    self._write_ball_state(request.position, request.velocity)
    self.actual_history.clear()
    self.actual_history.append(request.position.copy())
    self.prediction_times = np.zeros(0, dtype=np.float64)
    self.prediction_positions = np.zeros((0, 3), dtype=np.float64)
    self._oracle_times_from_launch = np.zeros(0, dtype=np.float64)
    self._oracle_positions_from_launch = np.zeros((0, 3), dtype=np.float64)
    self.ball_velocity = request.velocity.copy()
    self.predicted_bounce = None
    self.actual_bounce_position = None
    self.actual_bounce_time = None
    self.actual_bounce_restitution = None
    self.crossing = None
    self.evaluated_matches = []
    self.ranked_matches = []
    self.selected_match = None
    self.locked_target_world = None
    self._prepared_motion_index = None
    self._last_prediction_step = -1
    self._launch_step = self.current_step
    self._previous_vertical_velocity = float(request.velocity[2])
    self._last_descending_velocity = None
    self.state = RallyState.TRACKING
    if self.cfg.trajectory_source == "oracle-mujoco":
      self._prepare_oracle_trajectory()
    self._update_prediction(force=True)
    print(
      "[TENNIS] Launch "
      f"position={request.position.round(3).tolist()} "
      f"velocity={request.velocity.round(3).tolist()} "
      f"physical_restitution_target={self.cfg.physical_restitution:.3f} "
      f"estimator_restitution={self.cfg.estimator_restitution:.3f} "
      f"trajectory_source={self.cfg.trajectory_source}"
    )
    if self.crossing is None:
      print("[TENNIS] Plan: predicted trajectory does not cross frozen local x=0")
    elif self.evaluated_matches:
      nearest = self.evaluated_matches[0]
      spatial_count = sum(
        match.distance <= self.cfg.match_distance
        for match in self.evaluated_matches
      )
      side = "center" if self.crossing.side is None else self.crossing.side.value
      print(
        f"[TENNIS] Plan: pelvis={self.planning_position_world.round(3).tolist()} "
        f"side={side} crossing={self.crossing.time:.3f}s "
        f"nearest_motion={nearest.motion_index} distance={nearest.distance:.3f}m "
        f"mean={self.motion_targets[nearest.motion_index].mean_world.round(3).tolist()} "
        f"closest={nearest.target_world.round(3).tolist()} "
        f"arrival={nearest.arrival_time:.3f}s "
        f"minimum={nearest.minimum_strike_time:.3f}s "
        f"nominal={nearest.nominal_strike_time:.3f}s "
        f"contact={nearest.contact_time:.3f}s slack={nearest.time_slack:.3f}s "
        f"within_{self.cfg.match_distance:.2f}m={spatial_count} "
        f"time_feasible={len(self.ranked_matches)}"
      )
      if self.selected_match is not None:
        selected = self.selected_match
        print(
          "[TENNIS] Selected "
          f"motion={selected.motion_index} distance={selected.distance:.3f}m "
          f"arrival={selected.arrival_time:.3f}s "
          f"minimum={selected.minimum_strike_time:.3f}s "
          f"nominal={selected.nominal_strike_time:.3f}s "
          f"contact={selected.contact_time:.3f}s "
          f"wait={selected.wait_time:.3f}s slack={selected.time_slack:.3f}s"
        )
    if self.predicted_bounce is not None:
      print(
        "[TENNIS] Planned first bounce "
        f"position={self.predicted_bounce.position.round(3).tolist()} "
        f"time={self.predicted_bounce.time:.3f}s"
      )

  def _choose_stable_match(
    self, ranked: list[MotionTrajectoryMatch]
  ) -> MotionTrajectoryMatch | None:
    if not ranked:
      return None
    best = ranked[0]
    previous = self.selected_match
    if previous is None or previous.motion_index == best.motion_index:
      return best
    current = next(
      (item for item in ranked if item.motion_index == previous.motion_index), None
    )
    if (
      current is not None
      and current.distance <= best.distance + self.cfg.motion_switch_hysteresis
    ):
      return current
    return best

  def _update_prediction(self, *, force: bool = False) -> None:
    if self.state not in {RallyState.TRACKING, RallyState.EXECUTING}:
      return
    update_steps = max(1, round(self.cfg.prediction_update_dt / self.env.step_dt))
    if not force and self.current_step - self._last_prediction_step < update_steps:
      return
    self._last_prediction_step = self.current_step

    pose = self.ball.data.root_link_pose_w[0]
    velocity = self.ball.data.root_link_vel_w[0]
    true_position = pose[:3].detach().cpu().numpy().astype(np.float64, copy=True)
    ball_velocity = velocity[:3].detach().cpu().numpy().astype(np.float64, copy=True)
    self.true_ball_position = true_position
    self.ball_velocity = ball_velocity

    vertical_velocity = float(ball_velocity[2])
    near_ground = true_position[2] < BALL_RADIUS + 0.12
    if near_ground and vertical_velocity < 0.0:
      self._last_descending_velocity = vertical_velocity
    if (
      self.actual_bounce_position is None
      and near_ground
      and self._previous_vertical_velocity < 0.0 <= vertical_velocity
    ):
      incoming = (
        self._last_descending_velocity
        if self._last_descending_velocity is not None
        else self._previous_vertical_velocity
      )
      self.actual_bounce_position = true_position.copy()
      self.actual_bounce_position[2] = BALL_RADIUS
      self.actual_bounce_time = self.rally_time
      self.actual_bounce_restitution = vertical_velocity / max(-incoming, 1e-6)
      print(
        "[TENNIS] Physical bounce "
        f"position={self.actual_bounce_position.round(3).tolist()} "
        f"incoming_vz={incoming:.3f}m/s outgoing_vz={vertical_velocity:.3f}m/s "
        f"restitution={self.actual_bounce_restitution:.3f}"
      )
    self._previous_vertical_velocity = vertical_velocity

    if self.last_launch.position_noise_std > 0.0:
      self.measured_ball_position = true_position + self.noise_rng.normal(
        0.0, self.last_launch.position_noise_std, size=3
      )
    else:
      self.measured_ball_position = true_position.copy()

    if self.cfg.trajectory_source == "oracle-mujoco":
      self.prediction_times, self.prediction_positions = (
        self._oracle_prediction_from_current_time()
      )
    else:
      self.prediction_times, self.prediction_positions = predict_ball_trajectory(
        self.measured_ball_position,
        ball_velocity,
        duration=self.cfg.prediction_horizon,
        dt=self.cfg.prediction_dt,
        restitution=self.cfg.estimator_restitution,
        ground_z=BALL_RADIUS,
      )
    self.predicted_bounce = first_ground_contact(
      self.prediction_times,
      self.prediction_positions,
      ground_z=BALL_RADIUS,
    )
    self.actual_history.append(true_position.copy())
    # A no-candidate rally can outlive the fixed oracle horizon. There is no
    # remaining path to rank in that case; keep G1 frozen and wait for reset or
    # relaunch instead of passing an empty trajectory to the planner.
    if len(self.prediction_times) == 0:
      self.crossing = None
      if self.state is not RallyState.EXECUTING:
        self.evaluated_matches = []
        self.ranked_matches = []
        self.selected_match = None
      return
    if (
      self.planning_position_world is None
      or self.planning_forward_world is None
      or self.planning_lateral_world is None
    ):
      return
    self.crossing = first_forward_plane_crossing(
      self.prediction_times,
      self.prediction_positions,
      self.planning_position_world,
      self.planning_forward_world,
      self.planning_lateral_world,
      side_deadband=self.cfg.side_deadband,
    )
    if self.state is RallyState.EXECUTING:
      return
    if self.crossing is None:
      self.evaluated_matches = []
      self.ranked_matches = []
      self.selected_match = None
      return
    self.evaluated_matches = evaluate_motion_matches(
      self.prediction_times,
      self.prediction_positions,
      self.motion_targets,
      side=self.crossing.side,
    )
    initial_candidates = filter_motion_matches(
      self.evaluated_matches,
      max_distance=self.cfg.match_distance,
      reaction_margin=self.cfg.reaction_margin,
    )
    retained = retain_motion_match(
      self.evaluated_matches,
      None if self.selected_match is None else self.selected_match.motion_index,
      max_distance=self.cfg.match_distance,
    )
    committed = (
      retained is not None and retained.wait_time < self.cfg.reaction_margin
    )
    if committed:
      self.selected_match = retained
      self.ranked_matches = [retained] + [
        match
        for match in initial_candidates
        if match.motion_index != retained.motion_index
      ]
    else:
      self.ranked_matches = initial_candidates
      self.selected_match = self._choose_stable_match(self.ranked_matches)
    if self.selected_match is not None:
      self._prepare_match(self.selected_match)
      if self.selected_match.wait_time <= self.cfg.trigger_tolerance:
        self._activate_match(self.selected_match)

  def _set_planning_alignment(
    self, motion_index: int, env_ids: torch.Tensor
  ) -> None:
    if self.planning_position_world is None or self.planning_quaternion_world is None:
      raise RuntimeError("Planning frame is not initialized.")
    reference_quat = self.command._stacked_body_quat_w[
      motion_index, 0, self.command.motion_anchor_body_index
    ].unsqueeze(0)
    planning_position = torch.as_tensor(
      self.planning_position_world,
      dtype=reference_quat.dtype,
      device=self.env.device,
    ).unsqueeze(0)
    planning_quaternion = torch.as_tensor(
      self.planning_quaternion_world,
      dtype=reference_quat.dtype,
      device=self.env.device,
    ).unsqueeze(0)
    aligned_position, align_yaw = reference_frame0_alignment(
      reference_quat,
      planning_position,
      planning_quaternion,
    )
    reference_position = (
      self.command._stacked_body_pos_w[
        motion_index, 0, self.command.motion_anchor_body_index
      ]
      + self.env.scene.env_origins[0]
    )
    aligned_position[:, 2] = reference_position[2]
    self.command._reference_align_root_pos_w[env_ids] = aligned_position
    self.command._reference_align_yaw_w[env_ids] = align_yaw

  def _hold_reference_at_phase_zero(
    self, motion_index: int | None = None
  ) -> None:
    env_ids = torch.zeros(1, dtype=torch.long, device=self.env.device)
    motion_ids = None
    if motion_index is not None:
      motion_ids = torch.full(
        (1,), motion_index, dtype=torch.long, device=self.env.device
      )
    self.command.hold_at_phase_zero(env_ids, motion_ids)

  def _capture_initial_robot_state(self) -> None:
    """Capture the live state produced by the normal Play reset path."""
    robot = self.command.robot
    self._initial_robot_root_state = torch.cat(
      [robot.data.root_link_pose_w, robot.data.root_link_vel_w], dim=-1
    ).clone()
    self._initial_robot_joint_pos = robot.data.joint_pos.clone()

  def _align_waiting_reference_to_robot(self) -> None:
    """Keep phase-zero reference aligned to the captured Play reset pose."""
    env_ids = torch.zeros(1, dtype=torch.long, device=self.env.device)
    self.command._align_reference_frame0_to_robot(env_ids)
    self.command._update_ghost_alignment(env_ids)
    self.command._update_source_cache()

  def _write_robot_state(
    self,
    root_state: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    env_ids = torch.zeros(1, dtype=torch.long, device=self.env.device)
    robot = self.command.robot
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    robot.reset(env_ids=env_ids)
    self.env.sim.forward()

  def enforce_robot_freeze(self) -> None:
    """Pin G1 at the captured Play pose while ball planning continues."""
    if self.policy_enabled:
      return
    if (
      self._initial_robot_root_state is None
      or self._initial_robot_joint_pos is None
    ):
      raise RuntimeError("Initial G1 state has not been captured.")
    root_state = self._initial_robot_root_state.clone()
    root_state[:, 7:] = 0.0
    joint_vel = torch.zeros_like(self._initial_robot_joint_pos)
    self._write_robot_state(
      root_state,
      self._initial_robot_joint_pos,
      joint_vel,
    )

  def _reset_policy_handoff_state(self, env_ids: torch.Tensor) -> None:
    """Give the actor the same clean history it receives after an env reset."""
    self.env.action_manager.reset(env_ids)
    self.env.observation_manager.reset(env_ids)
    self.env.episode_length_buf[env_ids] = 0

  def _prepare_match(self, match: MotionTrajectoryMatch) -> None:
    env_ids = torch.zeros(1, dtype=torch.long, device=self.env.device)
    motion_index = match.motion_index
    if self._prepared_motion_index != motion_index:
      self._hold_reference_at_phase_zero(motion_index)
      self._set_planning_alignment(motion_index, env_ids)
      self.command._update_ghost_alignment(env_ids)
      self.command._sample_targets(env_ids)
      self._prepared_motion_index = motion_index

    position_slot = self.position_slots[motion_index]
    self.command.target_position_w[0, position_slot] = torch.as_tensor(
      match.target_world,
      dtype=self.command.target_position_w.dtype,
      device=self.env.device,
    )
    self.command._update_source_cache()

  def _activate_match(self, match: MotionTrajectoryMatch) -> None:
    env_ids = torch.zeros(1, dtype=torch.long, device=self.env.device)
    motion_ids = torch.full(
      (1,), match.motion_index, dtype=torch.long, device=self.env.device
    )
    self._prepare_match(match)
    contact_times = torch.full(
      (1,),
      match.contact_time,
      dtype=self.command.phase.dtype,
      device=self.env.device,
    )
    self.command.activate_motion_deadline(env_ids, motion_ids, contact_times)
    self._reset_policy_handoff_state(env_ids)
    self.locked_target_world = match.target_world.copy()
    self.selected_match = match
    self.state = RallyState.EXECUTING
    print(
      "[TENNIS] Trigger "
      f"side={match.side.value} motion={match.motion_index} "
      f"distance={match.distance:.3f}m target={match.target_world.round(3).tolist()} "
      f"arrival={match.arrival_time:.3f}s "
      f"minimum={match.minimum_strike_time:.3f}s "
      f"nominal={match.nominal_strike_time:.3f}s "
      f"contact={match.contact_time:.3f}s wait={match.wait_time:.3f}s "
      f"slack={match.time_slack:.3f}s"
    )

  def _reset_rally(self, *, env_already_reset: bool = False) -> None:
    relaunch = (
      self.cfg.loop and self.state is not RallyState.WAITING
      if self._relaunch_after_reset is None
      else self._relaunch_after_reset
    )
    if not env_already_reset:
      self.vec_env.reset()
    self._hold_reference_at_phase_zero()
    self._capture_initial_robot_state()
    self._align_waiting_reference_to_robot()
    self.state = RallyState.WAITING
    self.motion_targets = ()
    self.planning_position_world = None
    self.planning_quaternion_world = None
    self.planning_forward_world = None
    self.planning_lateral_world = None
    self.planning_up_world = None
    self.crossing = None
    self.evaluated_matches = []
    self.ranked_matches = []
    self.selected_match = None
    self.locked_target_world = None
    self._prepared_motion_index = None
    self.prediction_times = np.zeros(0, dtype=np.float64)
    self.prediction_positions = np.zeros((0, 3), dtype=np.float64)
    self.predicted_bounce = None
    self.actual_bounce_position = None
    self.actual_bounce_time = None
    self.actual_bounce_restitution = None
    self.ball_velocity = np.zeros(3, dtype=np.float64)
    self._previous_vertical_velocity = 0.0
    self._last_descending_velocity = None
    self.actual_history.clear()
    self._reset_requested = False
    self._relaunch_after_reset = None
    self._park_ball()
    print("[TENNIS] G1 and rally reset through Play environment logic")
    if relaunch:
      self.pending_launch = self.last_launch

  def before_policy_step(self) -> None:
    self.initialize()
    if self._reset_requested:
      self._reset_rally()
    if self.pending_launch is not None:
      request = self.pending_launch
      self.pending_launch = None
      self._apply_launch(request)
    if self.state is RallyState.WAITING:
      self._hold_reference_at_phase_zero()
      self._park_ball()
    elif self.state is RallyState.TRACKING:
      self._hold_reference_at_phase_zero()
    if not self.policy_enabled:
      self.enforce_robot_freeze()

  def after_env_step(self) -> None:
    if bool(self.env.reset_buf[0].item()):
      print("[TENNIS] Environment termination/reset detected")
      self._reset_rally(env_already_reset=True)
      return
    self._update_prediction()
    if self.state is RallyState.EXECUTING and self.selected_match is not None:
      motion_index = self.selected_match.motion_index
      total = int(self.command._time_step_totals[motion_index].item())
      if int(self.command.time_steps[0].item()) >= total - 1:
        self.state = RallyState.COMPLETE
        print(f"[TENNIS] Motion {motion_index} complete")
    if not self.policy_enabled:
      self.enforce_robot_freeze()

  def status_lines(self) -> tuple[str, str]:
    crossing_side = (
      "none"
      if self.crossing is None
      else "center" if self.crossing.side is None else self.crossing.side.value
    )
    nearest = self.evaluated_matches[0] if self.evaluated_matches else None
    spatial_count = sum(
      match.distance <= self.cfg.match_distance for match in self.evaluated_matches
    )
    nearest_distance = "none" if nearest is None else f"{nearest.distance:.3f} m"
    nearest_wait = "none" if nearest is None else f"{nearest.wait_time:.3f} s"
    selected_timing = (
      "none"
      if self.selected_match is None
      else (
        f"{self.selected_match.arrival_time:.3f} / "
        f"{self.selected_match.contact_time:.3f} / "
        f"{self.selected_match.wait_time:.3f} / "
        f"{self.selected_match.time_slack:.3f} s"
      )
    )
    motion = "none" if self.selected_match is None else str(self.selected_match.motion_index)
    control_mode = "policy" if self.policy_enabled else "frozen"
    target = (
      "none"
      if self.selected_match is None
      else np.array2string(self.selected_match.target_world, precision=3)
    )
    left = (
      "Rally\nG1 control\nStroke side\nSide motions\nWithin distance\nTime feasible\n"
      "Nearest distance\nNearest wait\nSelected motion\n"
      "Arrival / contact / wait / slack\nTarget world"
    )
    right = (
      f"{self.state.value}\n{control_mode}\n{crossing_side}\n"
      f"{len(self.evaluated_matches)}\n"
      f"{spatial_count} / {self.cfg.match_distance:.2f} m\n"
      f"{len(self.ranked_matches)}\n{nearest_distance}\n{nearest_wait}\n"
      f"{motion}\n{selected_timing}\n{target}"
    )
    return left, right


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
    np.array([radius, 0.0, 0.0]),
    np.asarray(position, dtype=np.float64),
    IDENTITY,
    color,
  )
  geom.category = mujoco.mjtCatBit.mjCAT_DECOR
  scene.ngeom += 1


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
    np.zeros(3),
    np.zeros(3),
    np.zeros(9),
    color,
  )
  geom.category = mujoco.mjtCatBit.mjCAT_DECOR
  mujoco.mjv_connector(
    geom,
    mujoco.mjtGeom.mjGEOM_LINE,
    width,
    np.asarray(start, dtype=np.float64),
    np.asarray(end, dtype=np.float64),
  )
  scene.ngeom += 1


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
    np.zeros(3),
    np.asarray(start, dtype=np.float64),
    IDENTITY,
    color,
  )
  geom.category = mujoco.mjtCatBit.mjCAT_DECOR
  mujoco.mjv_connector(
    geom,
    mujoco.mjtGeom.mjGEOM_ARROW,
    width,
    np.asarray(start, dtype=np.float64),
    np.asarray(end, dtype=np.float64),
  )
  scene.ngeom += 1


def _add_polyline(
  scene: mujoco.MjvScene,
  points: np.ndarray,
  width: float,
  color: np.ndarray,
  max_segments: int = 120,
) -> None:
  if len(points) < 2:
    return
  stride = max(1, int(np.ceil((len(points) - 1) / max_segments)))
  sampled = points[::stride]
  if not np.array_equal(sampled[-1], points[-1]):
    sampled = np.vstack((sampled, points[-1]))
  for start, end in pairwise(sampled):
    _add_line(scene, start, end, width, color)


class TennisEstimatorViewer(NativeMujocoViewer):
  def __init__(self, env, policy, controller: TennisEstimatorController) -> None:
    self.controller = controller
    self.control_panel: LaunchControlPanel | None = None
    self._idle_actions = torch.zeros(
      (env.num_envs, env.num_actions), device=env.device
    )
    super().__init__(
      env,
      policy,
      frame_rate=60.0,
      key_callback=self._key_callback,
      verbosity=VerbosityLevel.SILENT,
    )

  def setup(self) -> None:
    super().setup()
    self.controller.initialize()
    if self.controller.cfg.control_ui:
      launch = self.controller.last_launch
      self.control_panel = LaunchControlPanel(
        launch.position,
        launch.velocity,
        on_launch=self.controller.request_launch,
        on_position_change=self.controller.preview_launch_position,
        on_reset_robot=self.controller.request_robot_reset,
        ball_radius=BALL_RADIUS,
        ball_mass_kg=launch.mass_kg,
        position_noise_std=launch.position_noise_std,
      )

  def _key_callback(self, keycode: int) -> None:
    if keycode in (ord(" "), ord("R"), ord("r")):
      self.controller.request_relaunch()

  def tick(self) -> bool:
    if self.control_panel is not None and not self.control_panel.poll():
      self.control_panel = None
    return super().tick()

  def _execute_step(self) -> bool:
    self.controller.before_policy_step()
    try:
      with torch.no_grad():
        if self.controller.policy_enabled:
          obs = self.env.get_observations()
          actions = self.policy(obs)
        else:
          actions = self._idle_actions
        self.env.step(actions)
        self.controller.after_env_step()
        self._step_count += 1
        self._stats_steps += 1
        return True
    except Exception:  # noqa: BLE001
      self._last_error = traceback.format_exc()
      self.log(
        f"[ERROR] Exception during step:\n{self._last_error}",
        VerbosityLevel.SILENT,
      )
      self.pause()
      return False

  def _set_status_overlay(self, viewer: mujoco.viewer.Handle) -> None:
    status = self.get_status()
    capped = " [CAPPED]" if status.capped else ""
    controller = self.controller
    launch_speed, launch_azimuth, launch_elevation = speed_angles_from_velocity(
      controller.last_launch.velocity
    )
    measurement_error = float(
      np.linalg.norm(
        controller.measured_ball_position - controller.true_ball_position
      )
    )
    predicted_bounce = (
      "none"
      if controller.predicted_bounce is None
      else (
        f"{controller.predicted_bounce.position.round(3)} "
        f"@ {controller.predicted_bounce.time:.2f}s"
      )
    )
    actual_bounce = (
      "pending"
      if controller.actual_bounce_position is None
      else (
        f"{controller.actual_bounce_position.round(3)} "
        f"@ {controller.actual_bounce_time:.2f}s, "
        f"e={controller.actual_bounce_restitution:.2f}"
      )
    )
    base_left = (
      "Env\nStep\nStatus\nSpeed\nTarget RT\nActual RT\nRally time\n"
      "Ball xyz\nBall velocity\nLaunch xyz\nLaunch speed\n"
      "Launch azimuth / elevation\nBall mass\nPosition noise sigma\n"
      "Physical restitution target\nEstimator restitution\nTrajectory source\n"
      "Measurement error\nPredicted bounce\nActual first bounce"
    )
    base_right = (
      f"1/1\n{status.step_count}\n"
      f"{'PAUSED' if status.paused else 'RUNNING'}{capped}\n"
      f"{status.speed_label}\n{status.target_realtime:.2f}x\n"
      f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)\n"
      f"{controller.rally_time:.2f} s\n"
      f"{controller.true_ball_position.round(3)}\n"
      f"{controller.ball_velocity.round(3)}\n"
      f"{controller.last_launch.position.round(3)}\n"
      f"{launch_speed:.2f} m/s\n"
      f"{launch_azimuth:.1f} / {launch_elevation:.1f} deg\n"
      f"{1000.0 * controller.last_launch.mass_kg:.1f} g\n"
      f"{controller.last_launch.position_noise_std:.3f} m\n"
      f"{controller.cfg.physical_restitution:.3f}\n"
      f"{controller.cfg.estimator_restitution:.3f}\n"
      f"{controller.cfg.trajectory_source}\n"
      f"{measurement_error:.3f} m\n"
      f"{predicted_bounce}\n{actual_bounce}"
    )
    planner_left, planner_right = controller.status_lines()
    font = mujoco.mjtFontScale.mjFONTSCALE_150.value
    viewer.set_texts(
      [
        (font, mujoco.mjtGridPos.mjGRID_TOPLEFT.value, base_left, base_right),
        (font, mujoco.mjtGridPos.mjGRID_TOPRIGHT.value, planner_left, planner_right),
        (
          font,
          mujoco.mjtGridPos.mjGRID_BOTTOMLEFT.value,
          (
            "Cyan: estimator/oracle path  Yellow: physical history  Purple: frozen x=0 plane\n"
            "Orange: predicted bounce  Red: actual bounce  Green: launch / direction\n"
            "Blue: side crossing  Magenta: selected motion mean  Lime: target\n"
            "White: noisy position measurement\n"
            "Space/R: relaunch"
          ),
          "",
        ),
      ]
    )

  def _update_debug_visualizers(self, viewer: mujoco.viewer.Handle) -> None:
    super()._update_debug_visualizers(viewer)
    controller = self.controller
    scene = viewer.user_scn
    _add_polyline(scene, controller.prediction_positions, 3.0, PREDICTION_COLOR)
    if len(controller.prediction_positions):
      marker_stride = max(1, len(controller.prediction_positions) // 22)
      for point in controller.prediction_positions[::marker_stride]:
        _add_sphere(scene, point, 0.018, PREDICTION_COLOR)

    history = np.asarray(controller.actual_history, dtype=np.float64)
    _add_polyline(scene, history, 4.0, HISTORY_COLOR)

    _add_sphere(scene, controller.last_launch.position, 0.055, LAUNCH_COLOR)
    launch_speed = float(np.linalg.norm(controller.last_launch.velocity))
    if launch_speed > 1e-8:
      arrow_start = controller.last_launch.position + np.array([0.0, 0.0, 0.18])
      arrow_end = (
        arrow_start + 1.8 * controller.last_launch.velocity / launch_speed
      )
      _add_line(
        scene,
        controller.last_launch.position,
        arrow_start,
        3.0,
        LAUNCH_DIRECTION_COLOR,
      )
      _add_arrow(scene, arrow_start, arrow_end, 0.05, LAUNCH_DIRECTION_COLOR)

    if controller.predicted_bounce is not None:
      _add_sphere(
        scene,
        controller.predicted_bounce.position,
        0.12,
        PREDICTED_BOUNCE_COLOR,
      )
    if controller.actual_bounce_position is not None:
      _add_sphere(
        scene,
        controller.actual_bounce_position,
        0.10,
        ACTUAL_BOUNCE_COLOR,
      )

    if controller.last_launch.position_noise_std > 0.0:
      _add_line(
        scene,
        controller.true_ball_position,
        controller.measured_ball_position,
        2.0,
        MEASUREMENT_COLOR,
      )
      _add_sphere(scene, controller.measured_ball_position, 0.045, MEASUREMENT_COLOR)
    if controller.crossing is not None:
      _add_sphere(scene, controller.crossing.position_world, 0.07, CROSSING_COLOR)
    if controller.selected_match is not None:
      target = controller.motion_targets[controller.selected_match.motion_index]
      _add_sphere(scene, target.mean_world, 0.055, MEAN_COLOR)
      _add_sphere(scene, controller.selected_match.target_world, 0.065, TARGET_COLOR)

    if (
      controller.planning_position_world is not None
      and controller.planning_lateral_world is not None
      and controller.planning_up_world is not None
    ):
      origin = controller.planning_position_world
      lateral = controller.planning_lateral_world
      up = controller.planning_up_world
      corners = np.array(
        [
          origin - 2.0 * lateral - 0.8 * up,
          origin + 2.0 * lateral - 0.8 * up,
          origin + 2.0 * lateral + 1.5 * up,
          origin - 2.0 * lateral + 1.5 * up,
        ]
      )
      for line_start, line_end in zip(corners, np.roll(corners, -1, axis=0)):
        _add_line(scene, line_start, line_end, 2.0, PLANE_COLOR)

  def close(self) -> None:
    if self.control_panel is not None:
      self.control_panel.close()
      self.control_panel = None
    super().close()


def run_tennis_estimator_play(
  task_id: str, cfg: TennisEstimatorPlayConfig
) -> None:
  if cfg.num_envs not in (None, 1):
    raise ValueError("Tennis estimator play supports only --num-envs 1.")
  if cfg.viewer != "native" and cfg.headless_steps is None:
    raise ValueError("Tennis estimator overlays currently require --viewer native.")
  if cfg.video:
    raise ValueError("VideoRecorder is not supported by tennis estimator play yet.")
  cfg = replace(cfg, num_envs=1, physical_ball=True)
  env, policy = build_play_session(
    task_id,
    cfg,
    install_physical_ball_controller=False,
    env_cfg_hook=lambda env_cfg: _configure_tennis_play_env(
      env_cfg, cfg.physical_restitution
    ),
  )
  controller = TennisEstimatorController(env, cfg)
  try:
    if cfg.headless_steps is not None:
      controller.initialize()
      if not cfg.auto_launch:
        controller.request_relaunch()
      idle_actions = torch.zeros(
        (env.num_envs, env.num_actions), device=env.device
      )
      for _ in range(cfg.headless_steps):
        controller.before_policy_step()
        with torch.no_grad():
          if controller.policy_enabled:
            obs = env.get_observations()
            actions = policy(obs)
          else:
            actions = idle_actions
          env.step(actions)
        controller.after_env_step()
      left, right = controller.status_lines()
      print("[TENNIS] Headless summary")
      print(left.replace("\n", " | "))
      print(right.replace("\n", " | "))
    else:
      TennisEstimatorViewer(env, policy, controller).run()
  finally:
    env.close()


def main() -> None:
  import mjlab.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    TennisEstimatorPlayConfig,
    args=remaining_args,
    default=TennisEstimatorPlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_tennis_estimator_play(chosen_task, args)


if __name__ == "__main__":
  main()
