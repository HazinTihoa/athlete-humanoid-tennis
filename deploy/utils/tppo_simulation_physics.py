"""Training-aligned MuJoCo model assembly for the TPPO simulation backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROBOT_TERRAIN_COLLISION_BIT = 1 << 1
RACKET_NOMINAL_MASS_KG = 0.3
RACKET_NOMINAL_COM_WRIST_M = np.array((0.38, 0.01, 0.27), dtype=np.float64)
RACKET_NOMINAL_INERTIA_KG_M2 = np.array((0.006, 0.012, 0.008), dtype=np.float64)

COURT_NET_X_M = 5.6
COURT_NET_HALF_WIDTH_M = 5.485
COURT_SOLREF_TIME_CONSTANT_S = 0.065
COURT_SOLIMP = (0.925, 0.985, 0.0015, 0.5, 2.0)
COURT_FRICTION = (0.62, 0.62, 0.02, 0.002, 0.002)
SURROUND_FRICTION = (0.8, 0.8, 0.02, 0.002, 0.002)
NET_SOLREF = (0.013, 0.2275)
NET_SOLIMP = COURT_SOLIMP
NET_FRICTION = (0.55, 0.55, 0.02, 0.002, 0.002)
RACKET_SOLREF_TIME_CONSTANT_S = 0.020
RACKET_FRICTION = (0.55, 0.55, 0.02, 0.002, 0.002)
BALL_GEOM_FRICTION = (0.55, 0.02, 0.002)
BALL_GEOM_SOLREF = (0.020, 0.300)
BALL_GEOM_SOLIMP = (0.93, 0.99, 0.001, 0.5, 2.0)

RESTITUTION_SAMPLES = np.array(
    (0.5373, 0.6120, 0.6368, 0.6500, 0.6850, 0.7593, 0.8389, 0.9240),
    dtype=np.float64,
)
CONTACT_DAMPING_SAMPLES = np.array(
    (0.120, 0.105, 0.100, 0.095, 0.085, 0.072, 0.060, 0.056),
    dtype=np.float64,
)


@dataclass(frozen=True)
class TennisPhysicsSample:
    """One deployment world sampled from the train-time physics distribution."""

    ball_mass_kg: float
    court_restitution: float
    ground_tangent_speed_retention: float
    drag_coefficient: float
    racket_mass_kg: float
    racket_restitution: float
    racket_com_offset_m: np.ndarray
    torso_com_offset_m: np.ndarray
    foot_sliding_friction: float
    torso_mass_scale: float = 1.0


def contact_damping_for_restitution(restitution: float) -> float:
    """Training map, including bounded uncalibrated low-end extrapolation."""
    if not 0.5 <= restitution <= RESTITUTION_SAMPLES[-1]:
        raise ValueError(
            "Restitution is outside the supported range "
            f"[0.5, {RESTITUTION_SAMPLES[-1]}]: "
            f"{restitution}"
        )
    if restitution < RESTITUTION_SAMPLES[0]:
        slope = (CONTACT_DAMPING_SAMPLES[1] - CONTACT_DAMPING_SAMPLES[0]) / (
            RESTITUTION_SAMPLES[1] - RESTITUTION_SAMPLES[0]
        )
        return float(CONTACT_DAMPING_SAMPLES[0] + (restitution - RESTITUTION_SAMPLES[0]) * slope)
    return float(np.interp(restitution, RESTITUTION_SAMPLES, CONTACT_DAMPING_SAMPLES))


def _range(
    config: Mapping[str, Any], name: str, *, default: tuple[float, float]
) -> tuple[float, float]:
    raw = np.asarray(config.get(name, default), dtype=np.float64)
    if raw.shape != (2,) or not np.all(np.isfinite(raw)) or raw[0] > raw[1]:
        raise ValueError(f"Invalid physics randomization range {name}: {raw}")
    return float(raw[0]), float(raw[1])


def _sample_range(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    name: str,
    *,
    default: tuple[float, float],
) -> float:
    lower, upper = _range(config, name, default=default)
    return float(rng.uniform(lower, upper))


def sample_tennis_physics(
    config: Mapping[str, Any],
    nominal_ball_physics: Mapping[str, Any],
) -> TennisPhysicsSample:
    """Sample once per process, matching one train-time vectorized environment."""
    enabled = bool(config.get("enabled", True))
    rng = np.random.default_rng(int(config.get("random_seed", 1)))

    if enabled:
        ball_mass = _sample_range(rng, config, "ball_mass_kg", default=(0.0560, 0.0594))
        court_restitution = _sample_range(
            rng, config, "court_restitution", default=(0.70, 0.80)
        )
        tangent_retention = _sample_range(
            rng,
            config,
            "ground_tangent_speed_retention",
            default=(0.75, 0.90),
        )
        drag = _sample_range(rng, config, "drag_coefficient", default=(0.50, 0.65))
        racket_mass = _sample_range(rng, config, "racket_mass_kg", default=(0.27, 0.33))
        racket_restitution = _sample_range(
            rng, config, "racket_restitution", default=(0.54, 0.68)
        )
        racket_com_offset = np.array(
            (
                _sample_range(
                    rng, config, "racket_com_offset_x_m", default=(-0.015, 0.015)
                ),
                _sample_range(
                    rng, config, "racket_com_offset_y_m", default=(-0.010, 0.010)
                ),
                _sample_range(
                    rng, config, "racket_com_offset_z_m", default=(-0.015, 0.015)
                ),
            ),
            dtype=np.float64,
        )
        torso_com_offset = np.array(
            (
                _sample_range(
                    rng, config, "torso_com_offset_x_m", default=(-0.025, 0.025)
                ),
                _sample_range(
                    rng, config, "torso_com_offset_y_m", default=(-0.050, 0.050)
                ),
                _sample_range(
                    rng, config, "torso_com_offset_z_m", default=(-0.050, 0.050)
                ),
            ),
            dtype=np.float64,
        )
        foot_friction = _sample_range(
            rng, config, "foot_sliding_friction", default=(0.3, 1.2)
        )
    else:
        ball_mass = float(nominal_ball_physics["mass_kg"])
        court_restitution = 0.745
        tangent_retention = 0.825
        drag = float(nominal_ball_physics["drag_coefficient"])
        racket_mass = RACKET_NOMINAL_MASS_KG
        racket_restitution = 0.61
        racket_com_offset = np.zeros(3, dtype=np.float64)
        torso_com_offset = np.zeros(3, dtype=np.float64)
        foot_friction = 1.0

    return TennisPhysicsSample(
        ball_mass_kg=ball_mass,
        court_restitution=court_restitution,
        ground_tangent_speed_retention=tangent_retention,
        drag_coefficient=drag,
        racket_mass_kg=racket_mass,
        racket_restitution=racket_restitution,
        racket_com_offset_m=racket_com_offset,
        torso_com_offset_m=torso_com_offset,
        foot_sliding_friction=foot_friction,
        torso_mass_scale=(
            _sample_range(rng, config, "torso_mass_scale", default=(1.0, 1.0))
            if enabled and "torso_mass_scale" in config else 1.0
        ),
    )


def _required_model_id(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Required MuJoCo object is missing: {name}")
    return object_id


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


def _add_torque_actuators(spec: mujoco.MjSpec, joint_names: Sequence[str]) -> None:
    if spec.actuators:
        raise ValueError("Training robot XML must not already contain actuators.")
    passive_model = spec.compile()
    for name in joint_names:
        joint_id = _required_model_id(passive_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        effort_limit = float(np.max(np.abs(passive_model.jnt_actfrcrange[joint_id])))
        actuator = spec.add_actuator(name=name, target=name)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.set_to_motor()
        if effort_limit > 0.0:
            actuator.forcelimited = True
            actuator.forcerange = (-effort_limit, effort_limit)


def _add_training_racket_and_ball(
    spec: mujoco.MjSpec,
    nominal_ball_physics: Mapping[str, Any],
    *,
    launch_position_min_w: Sequence[float] = (8.0, -2.0, 0.5),
    launch_position_max_w: Sequence[float] = (10.0, 2.0, 1.3),
) -> tuple[str, ...]:
    wrist = spec.body("right_wrist_yaw_link")
    if wrist is None:
        raise ValueError("Training robot XML has no right_wrist_yaw_link.")
    if spec.body("racket_inertia") is not None:
        raise ValueError("Training robot XML already contains racket_inertia.")

    inertia_body = wrist.add_body(name="racket_inertia")
    inertia_body.explicitinertial = True
    inertia_body.mass = RACKET_NOMINAL_MASS_KG
    inertia_body.ipos = RACKET_NOMINAL_COM_WRIST_M
    inertia_body.inertia = RACKET_NOMINAL_INERTIA_KG_M2
    wrist.add_geom(
        name="racket_ball_collision",
        type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        pos=(0.38, -0.003, 0.27),
        quat=(-0.2753522, 0.2753522, 0.6512919, 0.6512919),
        size=(0.12, 0.17, 0.012),
        rgba=(0.0, 0.0, 0.0, 0.0),
        density=0.0,
        contype=1,
        conaffinity=1,
    )

    robot_geom_names = tuple(
        geom.name for geom in spec.geoms if geom.contype != 0 or geom.conaffinity != 0
    )
    terrain = spec.worldbody.add_body(name="terrain")
    terrain.add_geom(
        name="terrain",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.0, 0.0, 0.01),
        rgba=(0.0, 0.0, 0.0, 0.0),
        contype=ROBOT_TERRAIN_COLLISION_BIT,
        conaffinity=0,
    )
    for geom in spec.geoms:
        if geom.name in robot_geom_names:
            geom.conaffinity |= ROBOT_TERRAIN_COLLISION_BIT

    ball = spec.worldbody.add_body(name="tennis_ball")
    ball.add_joint(name="tennis_ball_freejoint", type=mujoco.mjtJoint.mjJNT_FREE)
    ball.add_geom(
        name="tennis_ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(float(nominal_ball_physics["radius_m"]), 0.0, 0.0),
        mass=float(nominal_ball_physics["mass_kg"]),
        friction=BALL_GEOM_FRICTION,
        solref=BALL_GEOM_SOLREF,
        solimp=BALL_GEOM_SOLIMP,
        rgba=(0.75, 0.90, 0.08, 1.0),
        contype=1,
        conaffinity=1,
    )
    launch_min = np.asarray(launch_position_min_w, dtype=np.float64)
    launch_max = np.asarray(launch_position_max_w, dtype=np.float64)
    if (
        launch_min.shape != (3,)
        or launch_max.shape != (3,)
        or not np.all(np.isfinite(launch_min))
        or not np.all(np.isfinite(launch_max))
        or np.any(launch_min >= launch_max)
    ):
        raise ValueError("Incoming-ball launch position bounds are invalid.")
    spec.worldbody.add_geom(
        name="incoming_ball_launch_region",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=0.5 * (launch_min + launch_max),
        size=0.5 * (launch_max - launch_min),
        rgba=(1.0, 0.82, 0.0, 0.16),
        contype=0,
        conaffinity=0,
    )
    return robot_geom_names


def _attach_training_court(
    spec: mujoco.MjSpec,
    court_xml_path: Path,
    robot_geom_names: Sequence[str],
    *,
    net_x_m: float = COURT_NET_X_M,
    net_half_width_m: float = COURT_NET_HALF_WIDTH_M,
) -> None:
    if not np.isfinite(net_x_m) or net_x_m <= 0.0:
        raise ValueError("Tennis net X must be finite and positive.")
    if not np.isfinite(net_half_width_m) or net_half_width_m <= 0.0:
        raise ValueError("Tennis net half-width must be finite and positive.")
    court_spec = mujoco.MjSpec.from_file(str(court_xml_path))
    court_ball = court_spec.body("tennis_ball")
    if court_ball is None:
        raise ValueError(f"Court XML has no tennis_ball body: {court_xml_path}")
    court_spec.delete(court_ball)

    for geom_name in (
        "net_collision",
        "net_horizontal_1",
        "net_horizontal_2",
        "net_horizontal_3",
    ):
        geom = court_spec.geom(geom_name)
        if geom is None:
            raise ValueError(f"Court XML has no {geom_name!r} geom.")
        geom.size[1] = net_half_width_m
    net_tape = court_spec.geom("net_tape")
    if net_tape is None:
        raise ValueError("Court XML has no 'net_tape' geom.")
    net_tape.size[1] = net_half_width_m + 0.035
    for geom_name, side in (("net_post_left", -1.0), ("net_post_right", 1.0)):
        geom = court_spec.geom(geom_name)
        if geom is None:
            raise ValueError(f"Court XML has no {geom_name!r} geom.")
        geom.pos[1] = side * (net_half_width_m + 0.065)

    for geom_name in ("surround", "court_surface"):
        geom = court_spec.geom(geom_name)
        if geom is None:
            raise ValueError(f"Court XML has no {geom_name!r} geom.")
        geom.contype = 0
        geom.conaffinity = 0

    _copy_court_visual_settings(spec, court_spec)
    court_spec.option.timestep = spec.option.timestep
    court_spec.option.integrator = spec.option.integrator
    court_spec.option.iterations = spec.option.iterations
    court_spec.option.cone = spec.option.cone
    court_spec.nconmax = spec.nconmax
    court_spec.njmax = spec.njmax

    frame = spec.worldbody.add_frame(pos=(net_x_m, 0.0, 0.0))
    spec.attach(court_spec, prefix="tennis_court/", frame=frame)

    pair_specs = (
        (
            "court",
            "tennis_court/court_surface",
            (COURT_SOLREF_TIME_CONSTANT_S, 0.075),
            COURT_SOLIMP,
            COURT_FRICTION,
        ),
        (
            "surround",
            "tennis_court/surround",
            (COURT_SOLREF_TIME_CONSTANT_S, 0.075),
            COURT_SOLIMP,
            SURROUND_FRICTION,
        ),
        ("net", "tennis_court/net_collision", NET_SOLREF, NET_SOLIMP, NET_FRICTION),
        (
            "racket",
            "racket_ball_collision",
            (RACKET_SOLREF_TIME_CONSTANT_S, 0.105),
            BALL_GEOM_SOLIMP,
            RACKET_FRICTION,
        ),
    )
    for pair_name, other_geom, solref, solimp, friction in pair_specs:
        spec.add_pair(
            name=f"tennis_ball_{pair_name}",
            geomname1="tennis_ball_geom",
            geomname2=other_geom,
            condim=3,
            solref=solref,
            solimp=solimp,
            friction=friction,
        )

    # Keep the invisible terrain exclusive to G1. Court and surround are selected
    # for the ball only through the explicit contact pairs above.
    for geom in spec.geoms:
        if geom.name in robot_geom_names:
            geom.conaffinity |= ROBOT_TERRAIN_COLLISION_BIT


def _apply_physics_sample(model: mujoco.MjModel, sample: TennisPhysicsSample) -> None:
    ball_id = _required_model_id(model, mujoco.mjtObj.mjOBJ_BODY, "tennis_ball")
    racket_id = _required_model_id(model, mujoco.mjtObj.mjOBJ_BODY, "racket_inertia")
    torso_id = _required_model_id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    nominal_ball_mass = float(model.body_mass[ball_id])
    nominal_racket_mass = float(model.body_mass[racket_id])
    model.body_inertia[ball_id] *= sample.ball_mass_kg / nominal_ball_mass
    model.body_mass[ball_id] = sample.ball_mass_kg
    model.body_inertia[racket_id] *= sample.racket_mass_kg / nominal_racket_mass
    model.body_mass[racket_id] = sample.racket_mass_kg
    model.body_ipos[racket_id] = RACKET_NOMINAL_COM_WRIST_M + sample.racket_com_offset_m
    model.body_ipos[torso_id] += sample.torso_com_offset_m
    if not np.isfinite(sample.torso_mass_scale) or sample.torso_mass_scale <= 0:
        raise ValueError("Torso mass scale must be finite and positive.")
    model.body_mass[torso_id] *= sample.torso_mass_scale
    model.body_inertia[torso_id] *= sample.torso_mass_scale

    for side in ("left", "right"):
        for index in range(1, 8):
            geom_id = _required_model_id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{side}_foot{index}_collision",
            )
            model.geom_friction[geom_id, 0] = sample.foot_sliding_friction

    court_damping = contact_damping_for_restitution(sample.court_restitution)
    racket_damping = contact_damping_for_restitution(sample.racket_restitution)
    for pair_name in ("tennis_ball_court", "tennis_ball_surround"):
        pair_id = _required_model_id(model, mujoco.mjtObj.mjOBJ_PAIR, pair_name)
        model.pair_solref[pair_id] = (
            COURT_SOLREF_TIME_CONSTANT_S,
            court_damping,
        )
    racket_pair_id = _required_model_id(
        model, mujoco.mjtObj.mjOBJ_PAIR, "tennis_ball_racket"
    )
    model.pair_solref[racket_pair_id] = (
        RACKET_SOLREF_TIME_CONSTANT_S,
        racket_damping,
    )


def build_training_aligned_model(
    *,
    robot_xml_path: Path,
    court_xml_path: Path,
    joint_names: Sequence[str],
    physics_dt: float,
    nominal_ball_physics: Mapping[str, Any],
    randomization_config: Mapping[str, Any],
    court_net_x_m: float = COURT_NET_X_M,
    court_net_half_width_m: float = COURT_NET_HALF_WIDTH_M,
    launch_position_min_w: Sequence[float] = (8.0, -2.0, 0.5),
    launch_position_max_w: Sequence[float] = (10.0, 2.0, 1.3),
) -> tuple[mujoco.MjModel, TennisPhysicsSample]:
    """Build the deployment simulator from the exact training robot and court assets."""
    spec = mujoco.MjSpec.from_file(str(robot_xml_path))
    _add_torque_actuators(spec, joint_names)
    robot_geom_names = _add_training_racket_and_ball(
        spec,
        nominal_ball_physics,
        launch_position_min_w=launch_position_min_w,
        launch_position_max_w=launch_position_max_w,
    )

    spec.option.timestep = physics_dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.nconmax = max(spec.nconmax, 200)
    spec.njmax = max(spec.njmax, 500)
    _attach_training_court(
        spec,
        court_xml_path,
        robot_geom_names,
        net_x_m=court_net_x_m,
        net_half_width_m=court_net_half_width_m,
    )

    model = spec.compile()
    sample = sample_tennis_physics(randomization_config, nominal_ball_physics)
    _apply_physics_sample(model, sample)
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return model, sample


class ExplicitGroundRebound:
    """Apply the same deterministic post-impact velocity correction as training."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        ball_radius_m: float,
        court_restitution: float,
        tangent_speed_retention: float,
    ) -> None:
        ball_joint_id = _required_model_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "tennis_ball_freejoint"
        )
        self.qpos_address = int(model.jnt_qposadr[ball_joint_id])
        self.dof_address = int(model.jnt_dofadr[ball_joint_id])
        self.ball_radius_m = float(ball_radius_m)
        self.court_restitution = float(court_restitution)
        self.tangent_speed_retention = float(tangent_speed_retention)
        self.reset()

    def reset(self) -> None:
        self._pending = False
        self._age = 0
        self._impact_velocity_w = np.zeros(3, dtype=np.float64)

    def after_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        position_z = float(data.qpos[self.qpos_address + 2])
        velocity = data.qvel[self.dof_address : self.dof_address + 6]
        linear_velocity = velocity[:3]

        if self._pending:
            self._age += 1
        elif linear_velocity[2] < -0.05 and position_z <= self.ball_radius_m + 0.03:
            self._pending = True
            self._age = 0
            self._impact_velocity_w[:] = linear_velocity
            height_to_ground = max(position_z - self.ball_radius_m, 0.0)
            self._impact_velocity_w[2] = -np.sqrt(
                linear_velocity[2] ** 2 + 2.0 * 9.81 * height_to_ground
            )

        impact_due = self._pending and position_z <= self.ball_radius_m + 0.002
        if impact_due:
            data.qpos[self.qpos_address + 2] = self.ball_radius_m + 1.0e-4
            linear_velocity[:2] = (
                self._impact_velocity_w[:2] * self.tangent_speed_retention
            )
            linear_velocity[2] = -self._impact_velocity_w[2] * self.court_restitution
            self.reset()
            mujoco.mj_forward(model, data)
            return True

        if self._pending and self._age > 20:
            self.reset()
        return False
