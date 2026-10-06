"""Shared tennis-court and physical-ball setup for Train and Play."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import (
    EventTermCfg,
    RecomputeLevel,
    requires_model_fields,
)
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat
from mjlab.utils.noise import GaussianNoiseCfg
from mjlab.viewer import ViewerConfig
from athlete.goal_cond_tracking.mdp.commands import MotionResamplePlan
from athlete.goal_cond_tracking.mdp.rewards import (
    tennis_ball_strike_event,
    tennis_strike_time_window_active,
)
from athlete.goal_cond_tracking.torch_tennis_planner import (
    classify_tennis_failure_trajectories,
    match_tennis_trajectories_to_motions_torch,
    randomize_tennis_launch_directions_torch,
    retarget_tennis_launches_to_miss_roots_torch,
    sample_root_directed_tennis_launches_torch,
    simulate_tennis_trajectories_torch,
)
from athlete.scripts.tennis_court_spec import (
    RACKET_COLLISION_HALF_SIZE_M,
    RACKET_COLLISION_POS_WRIST_M,
    RACKET_COLLISION_QUAT_WXYZ,
    RACKET_NOMINAL_COM_WRIST_M,
    RACKET_NOMINAL_INERTIA_KG_M2,
    RACKET_NOMINAL_MASS_KG,
    attach_tennis_court_spec,
)
from athlete.scripts.tennis_physics import (
    MIN_SUPPORTED_RESTITUTION,
    STANDARD_TENNIS_DOMAIN_RANDOMIZATION,
    STANDARD_TENNIS_PHYSICS,
    TennisDomainRandomizationCfg,
    contact_damping_for_restitution,
    tennis_ball_aerodynamic_wrench_torch,
)

TennisCourtMode = Literal["none", "visual", "physical"]

REPO_ROOT = Path(__file__).resolve().parents[4]
COURT_XML_PATH = REPO_ROOT / "deploy/simulation/assets/tennis_court.xml"
PHYSICS = STANDARD_TENNIS_PHYSICS
COURT_NET_X = PHYSICS.court.net_x_m
COURT_NET_HEIGHT = PHYSICS.court.net_height_m
COURT_NET_HALF_WIDTH = PHYSICS.court.net_half_width_m
COURT_FAR_SERVICE_LINE_X = PHYSICS.court.far_service_line_x_m
BALL_RADIUS = PHYSICS.ball.radius_m
DEFAULT_BALL_RELEASE_LEAD_S = 0.02
DEFAULT_STRIKE_SPEED_CHANGE_THRESHOLD = 0.05
DEFAULT_STRIKE_PROXIMITY_THRESHOLD = 0.25
BALL_GEOM_SOLREF = PHYSICS.ball.geom_solref
BALL_RACKET_SOLREF = PHYSICS.court.racket_solref
BALL_GEOM_SOLIMP = PHYSICS.ball.geom_solimp
BALL_COURT_SOLREF = (PHYSICS.court.court_solref_time_constant_s, 0.0)
BALL_COURT_SOLIMP = PHYSICS.court.court_solimp
BALL_COURT_FRICTION = PHYSICS.court.court_friction
BALL_SURROUND_FRICTION = PHYSICS.court.surround_friction
BALL_NET_SOLREF = PHYSICS.court.net_solref
BALL_NET_SOLIMP = PHYSICS.court.net_solimp
BALL_NET_FRICTION = PHYSICS.court.net_friction
BALL_RACKET_FRICTION = PHYSICS.court.racket_friction
ROBOT_TERRAIN_COLLISION_BIT = 1 << 1
RACKET_INERTIA_BODY_NAME = "racket_inertia"


@dataclass
class TennisDomainRandomizationState:
    """Per-world physical samples consumed by simulation controllers."""

    ball_mass_kg: torch.Tensor
    court_restitution: torch.Tensor
    ground_tangent_speed_retention: torch.Tensor
    drag_coefficient: torch.Tensor
    racket_mass_kg: torch.Tensor
    racket_restitution: torch.Tensor
    racket_com_offset_m: torch.Tensor
    explicit_ground_rebound: bool = False


def tennis_ball_spec() -> mujoco.MjSpec:
    """Build the free tennis-ball entity shared by Train and Play."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="tennis_ball")
    body.add_joint(name="tennis_ball_freejoint", type=mujoco.mjtJoint.mjJNT_FREE)
    body.add_geom(
        name="tennis_ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BALL_RADIUS, 0.0, 0.0],
        mass=PHYSICS.ball.mass_kg,
        friction=PHYSICS.ball.geom_friction,
        solref=BALL_GEOM_SOLREF,
        solimp=BALL_GEOM_SOLIMP,
        rgba=PHYSICS.ball.rgba,
        contype=1,
        conaffinity=1,
    )
    return spec


def add_racket_ball_collision(spec_fn):
    """Wrap a robot spec factory with an invisible racket-head collider."""

    def build_spec() -> mujoco.MjSpec:
        spec = spec_fn()
        wrist = spec.body("right_wrist_yaw_link")
        if wrist is None:
            raise ValueError("Physical ball requires body right_wrist_yaw_link.")
        if spec.body(RACKET_INERTIA_BODY_NAME) is None:
            inertia_body = wrist.add_body(name=RACKET_INERTIA_BODY_NAME)
            inertia_body.explicitinertial = True
            inertia_body.mass = RACKET_NOMINAL_MASS_KG
            inertia_body.ipos = RACKET_NOMINAL_COM_WRIST_M
            inertia_body.inertia = RACKET_NOMINAL_INERTIA_KG_M2
        if spec.geom("racket_ball_collision") is not None:
            return spec
        wrist.add_geom(
            name="racket_ball_collision",
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            pos=RACKET_COLLISION_POS_WRIST_M,
            quat=RACKET_COLLISION_QUAT_WXYZ,
            size=RACKET_COLLISION_HALF_SIZE_M,
            rgba=[0.0, 0.0, 0.0, 0.0],
            density=0.0,
            contype=1,
            conaffinity=1,
        )
        return spec

    return build_spec


def _sample_uniform_range(
    bounds: tuple[float, float], count: int, device: str
) -> torch.Tensor:
    lower, upper = bounds
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError(f"Invalid tennis domain-randomization range: {bounds}")
    return torch.empty(count, device=device).uniform_(lower, upper)


def _required_model_id(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Required tennis physics object is missing: {name}")
    return object_id


def _select_default_model_value(
    env: ManagerBasedRlEnv,
    field_name: str,
    env_ids: torch.Tensor,
    object_id: int,
) -> torch.Tensor:
    default = env.sim.get_default_field(field_name)
    model_field = getattr(env.sim.model, field_name)
    if default.ndim == model_field.ndim:
        return default[env_ids, object_id]
    value = default[object_id]
    return value.expand((len(env_ids), *value.shape))


def _contact_damping_from_restitution_tensor(
    restitution: torch.Tensor,
) -> torch.Tensor:
    samples = torch.tensor(
        PHYSICS.court.restitution_samples,
        dtype=restitution.dtype,
        device=restitution.device,
    )
    damping = torch.tensor(
        PHYSICS.court.contact_damping_samples,
        dtype=restitution.dtype,
        device=restitution.device,
    )
    if (
        not torch.isfinite(restitution).all()
        or torch.any(restitution < MIN_SUPPORTED_RESTITUTION)
        or torch.any(restitution > samples[-1])
    ):
        raise ValueError(
            "Tennis restitution samples exceed the supported MuJoCo contact range."
        )
    # Negative interpolation ratios extend the first segment down to 0.5.
    upper = torch.searchsorted(samples, restitution, right=False).clamp(
        min=1, max=len(samples) - 1
    )
    lower = upper - 1
    ratio = (restitution - samples[lower]) / (samples[upper] - samples[lower])
    return damping[lower] + ratio * (damping[upper] - damping[lower])


@requires_model_fields(
    "body_mass",
    "body_inertia",
    "body_ipos",
    "pair_solref",
    recompute=RecomputeLevel.set_const,
)
def randomize_tennis_physics(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    cfg: TennisDomainRandomizationCfg = STANDARD_TENNIS_DOMAIN_RANDOMIZATION,
) -> None:
    """Sample one physical domain per environment at simulator startup."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    count = len(env_ids)
    model = env.sim.mj_model

    ball_body_id = _required_model_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "tennis_ball/tennis_ball"
    )
    racket_body_id = _required_model_id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{RACKET_INERTIA_BODY_NAME}"
    )
    court_pair_id = _required_model_id(
        model, mujoco.mjtObj.mjOBJ_PAIR, "tennis_ball_court"
    )
    surround_pair_id = _required_model_id(
        model, mujoco.mjtObj.mjOBJ_PAIR, "tennis_ball_surround"
    )
    racket_pair_id = _required_model_id(
        model, mujoco.mjtObj.mjOBJ_PAIR, "tennis_ball_racket"
    )

    ball_mass = _sample_uniform_range(cfg.ball_mass_kg, count, env.device)
    court_restitution = _sample_uniform_range(cfg.court_restitution, count, env.device)
    tangent_retention = _sample_uniform_range(
        cfg.ground_tangent_speed_retention, count, env.device
    )
    drag_coefficient = _sample_uniform_range(cfg.drag_coefficient, count, env.device)
    racket_mass = _sample_uniform_range(cfg.racket_mass_kg, count, env.device)
    racket_restitution = _sample_uniform_range(
        cfg.racket_restitution, count, env.device
    )
    racket_com_offset = torch.stack(
        (
            _sample_uniform_range(cfg.racket_com_offset_x_m, count, env.device),
            _sample_uniform_range(cfg.racket_com_offset_y_m, count, env.device),
            _sample_uniform_range(cfg.racket_com_offset_z_m, count, env.device),
        ),
        dim=-1,
    )

    default_ball_mass = _select_default_model_value(
        env, "body_mass", env_ids, ball_body_id
    )
    default_ball_inertia = _select_default_model_value(
        env, "body_inertia", env_ids, ball_body_id
    )
    default_racket_mass = _select_default_model_value(
        env, "body_mass", env_ids, racket_body_id
    )
    default_racket_inertia = _select_default_model_value(
        env, "body_inertia", env_ids, racket_body_id
    )
    default_racket_com = _select_default_model_value(
        env, "body_ipos", env_ids, racket_body_id
    )

    env.sim.model.body_mass[env_ids, ball_body_id] = ball_mass
    env.sim.model.body_inertia[env_ids, ball_body_id] = (
        default_ball_inertia * (ball_mass / default_ball_mass)[:, None]
    )
    env.sim.model.body_mass[env_ids, racket_body_id] = racket_mass
    env.sim.model.body_inertia[env_ids, racket_body_id] = (
        default_racket_inertia * (racket_mass / default_racket_mass)[:, None]
    )
    env.sim.model.body_ipos[env_ids, racket_body_id] = (
        default_racket_com + racket_com_offset
    )

    default_pair_solref = env.sim.get_default_field("pair_solref")
    pair_solref = env.sim.model.pair_solref
    court_damping = _contact_damping_from_restitution_tensor(court_restitution)
    racket_damping = _contact_damping_from_restitution_tensor(racket_restitution)
    for pair_id in (court_pair_id, surround_pair_id):
        pair_solref[env_ids, pair_id, 0] = default_pair_solref[pair_id, 0]
        pair_solref[env_ids, pair_id, 1] = court_damping
    pair_solref[env_ids, racket_pair_id, 0] = default_pair_solref[racket_pair_id, 0]
    pair_solref[env_ids, racket_pair_id, 1] = racket_damping

    state = TennisDomainRandomizationState(
        ball_mass_kg=ball_mass,
        court_restitution=court_restitution,
        ground_tangent_speed_retention=tangent_retention,
        drag_coefficient=drag_coefficient,
        racket_mass_kg=racket_mass,
        racket_restitution=racket_restitution,
        racket_com_offset_m=racket_com_offset,
        explicit_ground_rebound=cfg.explicit_ground_rebound,
    )
    env._tennis_domain_randomization = state  # type: ignore[attr-defined]
    print(
        "[INFO]: Tennis physics DR sampled per environment: "
        f"ball_mass=[{ball_mass.min().item():.4f}, {ball_mass.max().item():.4f}] kg, "
        f"court_e=[{court_restitution.min().item():.3f}, "
        f"{court_restitution.max().item():.3f}], "
        f"tangent_retention=[{tangent_retention.min().item():.3f}, "
        f"{tangent_retention.max().item():.3f}], "
        f"Cd=[{drag_coefficient.min().item():.3f}, "
        f"{drag_coefficient.max().item():.3f}], Magnus=0"
    )


def configure_tennis_physics_domain_randomization(
    env_cfg: ManagerBasedRlEnvCfg,
    cfg: TennisDomainRandomizationCfg = STANDARD_TENNIS_DOMAIN_RANDOMIZATION,
) -> None:
    """Enable the restricted train-time tennis physics distribution."""
    env_cfg.events["tennis_physics_domain_randomization"] = EventTermCfg(
        func=randomize_tennis_physics,
        mode="startup",
        params={"cfg": cfg},
    )


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
    def build_spec() -> mujoco.MjSpec:
        spec = spec_fn()
        geom = spec.geom("tennis_ball_geom")
        if geom is None:
            raise ValueError("Physical tennis ball geom is missing.")
        geom.friction = PHYSICS.ball.geom_friction
        geom.solref = BALL_GEOM_SOLREF
        geom.solimp = BALL_GEOM_SOLIMP
        return spec

    return build_spec


def tennis_ball_predicted_contact(
    ball_position_w: torch.Tensor,
    ball_velocity_w: torch.Tensor,
    sweet_spot_position_w: torch.Tensor,
    sweet_spot_velocity_w: torch.Tensor,
    *,
    prediction_horizon_s: float,
    proximity_threshold: float,
) -> torch.Tensor:
    """Predict whether relative linear motion enters the racket-near region."""
    if prediction_horizon_s < 0.0:
        raise ValueError("Contact prediction horizon must be non-negative.")
    if proximity_threshold <= 0.0:
        raise ValueError("Contact prediction proximity threshold must be positive.")

    offset_w = ball_position_w - sweet_spot_position_w
    relative_velocity_w = ball_velocity_w - sweet_spot_velocity_w
    relative_speed_sq = torch.sum(torch.square(relative_velocity_w), dim=-1)
    time_to_closest = -torch.sum(offset_w * relative_velocity_w, dim=-1) / (
        relative_speed_sq.clamp(min=1.0e-12)
    )
    within_horizon = (time_to_closest >= 0.0) & (
        time_to_closest <= prediction_horizon_s
    )
    closest_offset_w = (
        offset_w
        + time_to_closest.clamp(min=0.0, max=prediction_horizon_s)[:, None]
        * relative_velocity_w
    )
    closest_distance = torch.linalg.vector_norm(closest_offset_w, dim=-1)
    return (
        (relative_speed_sq > 1.0e-12)
        & within_horizon
        & (closest_distance <= proximity_threshold)
    )


def attach_tennis_court(
    scene_spec: mujoco.MjSpec,
    *,
    mode: Literal["visual", "physical"],
    court_contact_damping: float,
    net_x_m: float = COURT_NET_X,
    net_half_width_m: float = COURT_NET_HALF_WIDTH,
    ball_geom_name: str = "tennis_ball/tennis_ball_geom",
    racket_geom_name: str = "robot/racket_ball_collision",
    robot_geom_names: tuple[str, ...] | None = None,
) -> None:
    """Attach the shared court to either MJWarp or standalone MuJoCo models."""
    attach_tennis_court_spec(
        scene_spec,
        mode=mode,
        court_contact_damping=court_contact_damping,
        net_x_m=net_x_m,
        net_half_width_m=net_half_width_m,
        ball_geom_name=ball_geom_name,
        racket_geom_name=racket_geom_name,
        robot_geom_names=robot_geom_names,
    )


def configure_tennis_court_env(
    env_cfg: ManagerBasedRlEnvCfg,
    *,
    mode: Literal["visual", "physical"],
    physical_restitution: float = PHYSICS.court.restitution,
    add_physical_ball: bool = True,
    align_env_origins: bool = False,
) -> None:
    """Configure a court for all batched worlds before scene compilation."""
    court_contact_damping = contact_damping_for_restitution(physical_restitution)

    if mode == "physical":
        robot_entity = env_cfg.scene.entities.get("robot")
        if robot_entity is None or robot_entity.spec_fn is None:
            raise ValueError("Physical tennis requires a robot scene entity.")
        if add_physical_ball:
            robot_entity.spec_fn = add_racket_ball_collision(robot_entity.spec_fn)
            env_cfg.scene.entities["tennis_ball"] = EntityCfg(spec_fn=tennis_ball_spec)
        ball_entity = env_cfg.scene.entities.get("tennis_ball")
        if ball_entity is None or ball_entity.spec_fn is None:
            raise ValueError("Physical tennis requires a tennis_ball scene entity.")
        ball_entity.spec_fn = _standalone_ball_spec(ball_entity.spec_fn)

    previous_scene_hook = env_cfg.scene.spec_fn

    def configure_scene(scene_spec: mujoco.MjSpec) -> None:
        if previous_scene_hook is not None:
            previous_scene_hook(scene_spec)
        motion_cfg = env_cfg.commands.get("motion")
        net_x_m = float(getattr(motion_cfg, "tennis_court_net_x_m", COURT_NET_X))
        net_half_width_m = float(
            getattr(
                motion_cfg,
                "tennis_court_net_half_width_m",
                COURT_NET_HALF_WIDTH,
            )
        )
        attach_tennis_court(
            scene_spec,
            mode=mode,
            court_contact_damping=court_contact_damping,
            net_x_m=net_x_m,
            net_half_width_m=net_half_width_m,
        )

    env_cfg.scene.spec_fn = configure_scene
    env_cfg.scene.extent = 20.0
    if align_env_origins:
        env_cfg.scene.env_spacing = 0.0
    if env_cfg.scene.terrain is not None:
        env_cfg.scene.terrain.textures = ()
        env_cfg.scene.terrain.materials = ()
        env_cfg.scene.terrain.lights = ()

    if mode == "physical":
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
        f"[INFO]: Tennis court mode={mode}, net_x={COURT_NET_X:.3f} m, "
        f"env_spacing={env_cfg.scene.env_spacing:.3f} m, "
        f"ball_mass={PHYSICS.ball.mass_kg * 1000.0:.1f} g, "
        f"restitution={physical_restitution:.3f}"
    )


def configure_tennis_landing_task(
    env_cfg: ManagerBasedRlEnvCfg,
    *,
    target_std_x: float,
    target_std_y: float,
    target_radius: float,
    reward_std: float,
    reward_weight: float,
    prediction_delay_steps: int,
    ball_direction_std: float,
    net_clearance_reward_weight: float,
    maximum_full_reward_net_height: float,
    excess_net_height_std: float,
    out_speed_target: float,
    out_speed_std: float,
    out_speed_reward_weight: float,
    strike_speed_change_threshold: float,
    strike_proximity_threshold: float,
    direction_tolerance_degrees: float,
    target_std_curriculum: tuple[tuple[int, float], ...] | None,
    curriculum_steps_per_iteration: int,
) -> None:
    """Add a sampled opponent-service-line target and physical bounce reward."""
    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.managers.reward_manager import RewardTermCfg
    from athlete.goal_cond_tracking import mdp
    from athlete.goal_cond_tracking.mdp import MultiTargetMotionCommandCfg

    if target_std_x < 0.0 or target_std_y < 0.0:
        raise ValueError("Tennis landing target std values must be non-negative.")
    if target_radius <= 0.0:
        raise ValueError("Tennis landing target radius must be positive.")
    if reward_std <= 0.0:
        raise ValueError("Tennis landing reward std must be positive.")
    if reward_weight <= 0.0:
        raise ValueError("Tennis landing reward weight must be positive.")
    if prediction_delay_steps < 0:
        raise ValueError("Prediction delay steps must be non-negative.")
    if ball_direction_std <= 0.0:
        raise ValueError("Ball direction reward std must be positive.")
    if net_clearance_reward_weight <= 0.0:
        raise ValueError("Net-clearance reward weight must be positive.")
    if maximum_full_reward_net_height <= COURT_NET_HEIGHT + BALL_RADIUS:
        raise ValueError("Maximum full-reward height must clear the tennis net.")
    if excess_net_height_std <= 0.0:
        raise ValueError("Net excess-height std must be positive.")
    if out_speed_target <= 0.0:
        raise ValueError("Target ball-out speed must be positive.")
    if out_speed_std <= 0.0:
        raise ValueError("Ball-out speed reward std must be positive.")
    if out_speed_reward_weight <= 0.0:
        raise ValueError("Ball-out speed reward weight must be positive.")
    if strike_speed_change_threshold <= 0.0:
        raise ValueError("Strike speed-change threshold must be positive.")
    if strike_proximity_threshold <= 0.0:
        raise ValueError("Strike proximity threshold must be positive.")
    if not 0.0 <= direction_tolerance_degrees < 180.0:
        raise ValueError("Direction tolerance must be in [0, 180) degrees.")
    if curriculum_steps_per_iteration <= 0:
        raise ValueError("Curriculum steps per iteration must be positive.")

    stage_steps: tuple[int, ...] = ()
    stage_stds: tuple[tuple[float, float, float], ...] = ()
    if target_std_curriculum is not None:
        if not target_std_curriculum:
            raise ValueError("Landing target std curriculum must not be empty.")
        stage_iterations = tuple(stage[0] for stage in target_std_curriculum)
        stage_scales = tuple(stage[1] for stage in target_std_curriculum)
        if stage_iterations[0] != 0 or any(
            current <= previous for previous, current in pairwise(stage_iterations)
        ):
            raise ValueError(
                "Landing curriculum iterations must start at zero and increase."
            )
        if any(scale < 0.0 for scale in stage_scales):
            raise ValueError("Landing curriculum scales must be non-negative.")
        stage_steps = tuple(
            iteration * curriculum_steps_per_iteration for iteration in stage_iterations
        )
        stage_stds = tuple(
            (target_std_x * scale, target_std_y * scale, 0.0) for scale in stage_scales
        )

    motion_cfg = env_cfg.commands.get("motion")
    if not isinstance(motion_cfg, MultiTargetMotionCommandCfg):
        raise TypeError("Tennis landing reward requires a multi-target motion command.")
    motion_cfg.landing_target_enabled = True
    motion_cfg.landing_target_mean = (COURT_FAR_SERVICE_LINE_X, 0.0, 0.0)
    motion_cfg.landing_target_std = (
        stage_stds[0]
        if target_std_curriculum is not None
        else (target_std_x, target_std_y, 0.0)
    )
    motion_cfg.landing_target_radius = target_radius

    if target_std_curriculum is not None:
        env_cfg.curriculum["landing_target_std"] = CurriculumTermCfg(
            func=mdp.tennis_landing_target_std_curriculum,
            params={
                "command_name": "motion",
                "stage_steps": stage_steps,
                "stage_stds": stage_stds,
            },
        )

    critic_group = env_cfg.observations.get("critic")
    if isinstance(critic_group, ObservationGroupCfg):
        critic_group.terms = {
            "reference_motion_state": ObservationTermCfg(
                func=mdp.motion_reference_state,
                params={"command_name": "motion"},
            ),
            **{
                name: term
                for name, term in critic_group.terms.items()
                if name != "command"
            },
        }
        critic_group.terms["task_goal"] = ObservationTermCfg(
            func=mdp.motion_task_goal,
            params={"command_name": "motion"},
        )
        critic_group.terms["time_remaining"] = ObservationTermCfg(
            func=mdp.motion_time_remaining,
            params={"command_name": "motion"},
        )

    policy_group_names = (
        ("student",) if "student" in env_cfg.observations else ("actor",)
    )
    for group_name in (*policy_group_names, "critic"):
        group = env_cfg.observations.get(group_name)
        if not isinstance(group, ObservationGroupCfg):
            continue
        group.terms["landing_target"] = ObservationTermCfg(
            func=mdp.tennis_landing_target_b,
            params={"command_name": "motion"},
        )
        group.terms["ball_position"] = ObservationTermCfg(
            func=mdp.tennis_ball_position_b,
            params={"command_name": "motion", "ball_entity_name": "tennis_ball"},
        )
        group.terms["ball_linear_velocity"] = ObservationTermCfg(
            func=mdp.tennis_ball_linear_velocity_b,
            params={"command_name": "motion", "ball_entity_name": "tennis_ball"},
        )
        group.terms["sweet_spot_linear_velocity"] = ObservationTermCfg(
            func=mdp.tennis_sweet_spot_linear_velocity_b,
            params={"command_name": "motion", "source_index": 0},
        )

    student_group = env_cfg.observations.get("student")
    if isinstance(student_group, ObservationGroupCfg):
        student_group.terms["global_root_pos"] = ObservationTermCfg(
            func=mdp.robot_global_root_pos_w,
            params={"command_name": "motion"},
            noise=GaussianNoiseCfg(mean=0.0, std=0.1),
        )

    critic_group = env_cfg.observations.get("critic")
    if isinstance(critic_group, ObservationGroupCfg):
        critic_group.terms["global_root_pos"] = ObservationTermCfg(
            func=mdp.robot_global_root_pos_w,
            params={"command_name": "motion"},
        )

    for group_name in ("student", "critic"):
        group = env_cfg.observations.get(group_name)
        if not isinstance(group, ObservationGroupCfg):
            continue
        group.terms["projected_gravity"] = ObservationTermCfg(
            func=mdp.robot_projected_gravity_b,
            params={"command_name": "motion"},
        )

    if env_cfg.rewards is None:
        raise ValueError("Tennis landing reward requires an active reward manager.")
    for reward_name in ("target_orientation_reward", "target_velocity_reward"):
        if reward_name not in env_cfg.rewards:
            raise KeyError(f"Tennis landing task requires reward {reward_name!r}.")
        env_cfg.rewards[reward_name].params["tolerance_degrees"] = (
            direction_tolerance_degrees
        )
    env_cfg.rewards["ball_landing_reward"] = RewardTermCfg(
        func=mdp.tennis_ball_predicted_landing_target_reward,
        weight=reward_weight,
        params={
            "command_name": "motion",
            "std": reward_std,
            "ball_entity_name": "tennis_ball",
            "ball_radius": BALL_RADIUS,
            "target_radius": target_radius,
            "prediction_delay_steps": prediction_delay_steps,
            "net_x": COURT_NET_X,
            "net_height": COURT_NET_HEIGHT,
            "net_half_width": COURT_NET_HALF_WIDTH,
            "maximum_full_reward_net_height": maximum_full_reward_net_height,
            "excess_net_height_std": excess_net_height_std,
            "target_out_speed": out_speed_target,
            "out_speed_std": out_speed_std,
            "ball_direction_std": ball_direction_std,
            "strike_speed_change_threshold": strike_speed_change_threshold,
            "strike_proximity_threshold": strike_proximity_threshold,
            "source_index": 0,
        },
    )
    env_cfg.rewards["net_clearance_reward"] = RewardTermCfg(
        func=mdp.tennis_ball_net_clearance_reward,
        weight=net_clearance_reward_weight,
        params={"command_name": "motion"},
    )
    env_cfg.rewards["ball_out_speed_reward"] = RewardTermCfg(
        func=mdp.tennis_ball_out_speed_reward,
        weight=out_speed_reward_weight,
        params={"command_name": "motion"},
    )
    print(
        "[INFO]: Tennis landing target enabled: "
        f"mean=({COURT_FAR_SERVICE_LINE_X:.3f}, 0.000) m, "
        f"std=({target_std_x:.3f}, {target_std_y:.3f}) m, "
        f"full_reward_radius={target_radius:.3f} m, "
        f"landing_reward_weight={reward_weight:.1f}, "
        f"ball_direction_std={ball_direction_std:.2f}, "
        f"prediction_delay_steps={prediction_delay_steps}, "
        f"full_net_height<={maximum_full_reward_net_height:.3f} m, "
        f"target_out_speed={out_speed_target:.3f} m/s, "
        f"out_speed_reward_weight={out_speed_reward_weight:.1f}, "
        f"strike_speed_change_threshold={strike_speed_change_threshold:.3f} m/s, "
        f"goal_direction_tolerance={direction_tolerance_degrees:.1f} deg"
    )
    if target_std_curriculum is not None:
        print(
            "[INFO]: Landing-target std curriculum: "
            + ", ".join(
                f"iter {iteration}: ({target_std_x * scale:.3f}, {target_std_y * scale:.3f}) m"
                for iteration, scale in target_std_curriculum
            )
        )


class TennisBallTargetController:
    """Hold each ball until predicted contact, impact, or the strike deadline."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        release_lead_s: float,
        *,
        verbose_samples: bool = False,
        contact_prediction_horizon_s: float | None = None,
        strike_speed_change_threshold: float = DEFAULT_STRIKE_SPEED_CHANGE_THRESHOLD,
        strike_proximity_threshold: float = DEFAULT_STRIKE_PROXIMITY_THRESHOLD,
    ) -> None:
        if release_lead_s < 0.0:
            raise ValueError("release_lead_s must be non-negative.")
        self.env = env
        self.ball = env.scene["tennis_ball"]
        self.motion = env.command_manager.get_term("motion")
        self.verbose_samples = verbose_samples
        self.release_lead_s = release_lead_s
        self.contact_prediction_horizon_s = (
            env.step_dt
            if contact_prediction_horizon_s is None
            else contact_prediction_horizon_s
        )
        if self.contact_prediction_horizon_s < 0.0:
            raise ValueError("Contact prediction horizon must be non-negative.")
        if strike_speed_change_threshold <= 0.0:
            raise ValueError("Strike speed-change threshold must be positive.")
        if strike_proximity_threshold <= 0.0:
            raise ValueError("Strike proximity threshold must be positive.")
        self.strike_speed_change_threshold = strike_speed_change_threshold
        self.strike_proximity_threshold = strike_proximity_threshold
        if not hasattr(self.motion, "time_remaining") or not hasattr(
            self.motion, "contact_time"
        ):
            raise TypeError(
                "Physical tennis training requires a deadline-aware command."
            )
        self.contact_window_s = float(self.motion.cfg.contact_reward_window_s)
        if self.contact_window_s < 0.0:
            raise ValueError("Physical tennis contact window must be non-negative.")

        target_indices: list[int] = []
        strike_frames: list[int] = []
        for motion_id, cfg in enumerate(self.motion.motion_configs):
            try:
                target_index = next(
                    i
                    for i, target in enumerate(cfg.sub_targets)
                    if target.goal_type == "position"
                )
            except StopIteration as exc:
                raise ValueError(
                    f"Motion {motion_id} has no position target for the physical ball."
                ) from exc
            total = int(self.motion._time_step_totals[motion_id].item())
            target_cfg = cfg.sub_targets[target_index]
            strike_phase = 0.5 * (
                target_cfg.target_phase_start + target_cfg.target_phase_end
            )
            strike_frames.append(round(strike_phase * (total - 1)))
            target_indices.append(target_index)

        device = env.device
        self._target_indices = torch.tensor(
            target_indices, device=device, dtype=torch.long
        )
        self._strike_frames = torch.tensor(
            strike_frames, device=device, dtype=torch.long
        )
        self._previous_time_remaining = torch.full_like(
            self.motion.time_remaining, torch.inf
        )
        self._previous_motion_ids = torch.full_like(self.motion.which_motion, -1)
        self._released = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self._env_ids = torch.arange(env.num_envs, device=device)
        self._held_pose = torch.zeros((env.num_envs, 7), device=device)
        self._held_pose[:, 3] = 1.0
        self._zero_velocity = torch.zeros((env.num_envs, 6), device=device)
        self._previous_ball_velocity_w = torch.zeros((env.num_envs, 3), device=device)
        self._previous_racket_offset_w = torch.zeros((env.num_envs, 3), device=device)
        self._strike_state_initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=device
        )
        self._held_during_step = torch.zeros(
            env.num_envs, dtype=torch.bool, device=device
        )

    def before_step(self) -> None:
        time_remaining = self.motion.time_remaining
        motion_ids = self.motion.which_motion
        restarted = (
            time_remaining > self._previous_time_remaining + 0.5 * self.env.step_dt
        ) | (motion_ids != self._previous_motion_ids)
        self._released[restarted] = False
        self._strike_state_initialized[restarted] = False

        if self.verbose_samples and torch.any(restarted):
            for env_id in torch.where(restarted)[0].tolist():
                motion_id = int(motion_ids[env_id].item())
                motion_file = Path(self.motion.cfg.motion_files[motion_id]).name
                strike_frame = int(self._strike_frames[motion_id].item())
                total = int(self.motion._time_step_totals[motion_id].item())
                print(
                    f"[PLAY SAMPLE] env={env_id} motion_id={motion_id} "
                    f"file={motion_file} strike_frame={strike_frame} "
                    f"sampled_contact_time={self.motion.contact_time[env_id].item():.3f}s "
                    f"total_frames={total}",
                    flush=True,
                )

        target_indices = self._target_indices[motion_ids]
        target_position_w = self.motion.target_position_w[self._env_ids, target_indices]
        source_position_w = self.motion.get_source_pos_w()[
            self._env_ids, target_indices
        ]
        source_velocity_w = self.motion.get_source_lin_vel_w()[
            self._env_ids, target_indices
        ]
        predicted_contact = tennis_ball_predicted_contact(
            target_position_w,
            self._zero_velocity[:, :3],
            source_position_w,
            source_velocity_w,
            prediction_horizon_s=self.contact_prediction_horizon_s,
            proximity_threshold=self.strike_proximity_threshold,
        )
        strike_time_active = tennis_strike_time_window_active(
            time_remaining,
            self.contact_window_s,
        )
        was_held = ~self._released
        predictive_release = was_held & strike_time_active & predicted_contact
        timed_release = (
            was_held & strike_time_active & (time_remaining <= self.release_lead_s)
        )
        self._released |= predictive_release | timed_release
        held = ~self._released
        self._held_pose[:, :3] = self.motion.target_position_w[
            self._env_ids, target_indices
        ]

        qpos_adr = self.ball.data.indexing.free_joint_q_adr
        qvel_adr = self.ball.data.indexing.free_joint_v_adr
        current_pose = self.ball.data.data.qpos[:, qpos_adr]
        current_velocity = self.ball.data.data.qvel[:, qvel_adr]
        self.ball.data.data.qpos[:, qpos_adr] = torch.where(
            held[:, None], self._held_pose, current_pose
        )
        self.ball.data.data.qvel[:, qvel_adr] = torch.where(
            held[:, None], self._zero_velocity, current_velocity
        )

        ball_position_before_step = torch.where(
            held[:, None], target_position_w, self.ball.data.root_link_pos_w
        )
        ball_velocity_before_step = torch.where(
            held[:, None],
            self._zero_velocity[:, :3],
            self.ball.data.root_link_lin_vel_w,
        )
        self._previous_ball_velocity_w.copy_(ball_velocity_before_step)
        self._previous_racket_offset_w.copy_(
            ball_position_before_step - source_position_w
        )
        self._strike_state_initialized.fill_(True)
        self._held_during_step.copy_(held)

        if self.verbose_samples:
            for reason, release_mask in (
                ("predicted_contact", predictive_release),
                ("deadline", timed_release & ~predictive_release),
            ):
                for env_id in torch.where(release_mask)[0].tolist():
                    print(
                        f"[BALL RELEASE] env={env_id} reason={reason} "
                        f"time_remaining={time_remaining[env_id].item():.3f}s",
                        flush=True,
                    )

        self._previous_time_remaining.copy_(time_remaining)
        self._previous_motion_ids.copy_(motion_ids)

    def after_step(self, reset_mask: torch.Tensor | None = None) -> None:
        """Latch dynamics after a held ball receives a racket-near impulse."""
        motion_ids = self.motion.which_motion
        target_indices = self._target_indices[motion_ids]
        source_position_w = self.motion.get_source_pos_w()[
            self._env_ids, target_indices
        ]
        ball_position_w = self.ball.data.root_link_pos_w
        ball_velocity_w = self.ball.data.root_link_lin_vel_w
        strike_time_active = tennis_strike_time_window_active(
            self.motion.time_remaining,
            self.contact_window_s,
        )
        impact_release = (
            self._held_during_step
            & strike_time_active
            & tennis_ball_strike_event(
                ball_velocity_w,
                self._previous_ball_velocity_w,
                ball_position_w - source_position_w,
                self._previous_racket_offset_w,
                self._strike_state_initialized,
                speed_change_threshold=self.strike_speed_change_threshold,
                proximity_threshold=self.strike_proximity_threshold,
            )
        )
        if reset_mask is not None:
            impact_release &= ~reset_mask
        self._released |= impact_release
        if self.verbose_samples:
            for env_id in torch.where(impact_release)[0].tolist():
                print(
                    f"[BALL RELEASE] env={env_id} reason=impact "
                    f"time_remaining={self.motion.time_remaining[env_id].item():.3f}s",
                    flush=True,
                )


class TorchIncomingBallMotionPlanner:
    """Sample incoming launches and match them to frame-0-local motion targets."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        motion,
        *,
        verbose_samples: bool = False,
    ) -> None:
        self.env = env
        self.motion = motion
        self.cfg = motion.cfg
        self.verbose_samples = verbose_samples
        motion_count = len(motion.motion_loaders)
        expected = int(self.cfg.incoming_ball_torch_match_expected_motions)
        if expected > 0 and motion_count != expected:
            raise ValueError(
                "Torch motion matching loaded the wrong number of motions: "
                f"expected {expected}, got {motion_count}."
            )
        if motion_count == 0:
            raise ValueError("Torch motion matching requires at least one motion.")
        if not hasattr(motion, "_nominal_contact_times"):
            raise TypeError("Torch motion matching requires a phase-aware command.")

        self.motion_count = motion_count
        self.dt = float(self.cfg.incoming_ball_torch_match_dt)
        self.horizon_s = float(self.cfg.incoming_ball_torch_match_horizon_s)
        self.maximum_distance = float(self.cfg.incoming_ball_torch_match_max_distance)
        self.attempt_count = int(self.cfg.incoming_ball_torch_match_attempts)
        self.retry_rounds = int(self.cfg.incoming_ball_torch_match_retry_rounds)
        self.adaptive_attempt_batch_size = int(
            self.cfg.incoming_ball_torch_match_adaptive_attempt_batch_size
        )
        self.motion_chunk_size = int(
            self.cfg.incoming_ball_torch_match_motion_chunk_size
        )
        self.hierarchical_top_k = int(
            self.cfg.incoming_ball_torch_match_hierarchical_top_k
        )
        self.hierarchical_coarse_neighbor_radius = int(
            self.cfg.incoming_ball_torch_match_hierarchical_coarse_neighbor_radius
        )
        self.cache_size = int(self.cfg.incoming_ball_torch_match_cache_size)
        self.failure_trajectory_probability = float(
            self.cfg.incoming_ball_failure_trajectory_probability
        )
        self.failure_trajectory_xy_distance_threshold_m = float(
            self.cfg.incoming_ball_failure_trajectory_xy_distance_threshold_m
        )
        self.failure_no_net_fraction = float(
            self.cfg.incoming_ball_failure_no_net_fraction
        )
        self.failure_overhead_fraction = float(self.cfg.incoming_ball_failure_overhead_fraction)
        configured_failure_horizon_s = getattr(
            self.cfg, "incoming_ball_failure_rollout_horizon_s", None
        )
        self.failure_rollout_horizon_s = (
            self.horizon_s
            if configured_failure_horizon_s is None
            else float(configured_failure_horizon_s)
        )
        configured_observation_radius = getattr(
            self.cfg, "incoming_ball_failure_observation_radius_m", None
        )
        configured_stop_speed = getattr(
            self.cfg, "incoming_ball_failure_stop_speed_m_s", None
        )
        self.failure_observation_radius_m = (
            None
            if configured_observation_radius is None
            else float(configured_observation_radius)
        )
        self.failure_stop_speed_m_s = (
            None if configured_stop_speed is None else float(configured_stop_speed)
        )
        self.failure_random_direction_fraction = float(
            getattr(self.cfg, "incoming_ball_failure_random_direction_fraction", 0.0)
        )
        self.failure_random_position_radius_range_m = tuple(
            float(value)
            for value in getattr(
                self.cfg,
                "incoming_ball_failure_random_position_radius_range_m",
                (2.0, 5.0),
            )
        )
        self.failure_random_position_height_range_m = tuple(
            float(value)
            for value in getattr(
                self.cfg,
                "incoming_ball_failure_random_position_height_range_m",
                (0.8, 2.2),
            )
        )
        self.failure_random_vertical_speed_range_m_s = tuple(
            float(value)
            for value in getattr(
                self.cfg,
                "incoming_ball_failure_random_vertical_speed_range_m_s",
                (-2.0, 4.0),
            )
        )
        if self.failure_rollout_horizon_s < self.horizon_s:
            raise ValueError("Failure rollout horizon must cover the normal rollout horizon.")
        if (self.failure_observation_radius_m is None) != (
            self.failure_stop_speed_m_s is None
        ):
            raise ValueError(
                "Failure observation radius and stop speed must be configured together."
            )
        if self.failure_observation_radius_m is not None and (
            self.failure_observation_radius_m <= 0.0
            or self.failure_stop_speed_m_s < 0.0
        ):
            raise ValueError("Failure trajectory stop limits are invalid.")
        if not 0.0 <= self.failure_overhead_fraction <= 1.0:
            raise ValueError('Failure overhead fraction must lie in [0, 1].')
        if not 0.0 <= self.failure_random_direction_fraction <= 1.0:
            raise ValueError("Failure random-direction fraction must lie in [0, 1].")
        for name, bounds in (
            ("random source radius", self.failure_random_position_radius_range_m),
            ("random source height", self.failure_random_position_height_range_m),
            ("random vertical speed", self.failure_random_vertical_speed_range_m_s),
        ):
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"Failure {name} bounds must be ordered pairs.")
        if self.failure_random_position_radius_range_m[0] <= 0.0:
            raise ValueError("Failure random source radius must be positive.")
        if self.failure_random_position_height_range_m[0] <= BALL_RADIUS:
            raise ValueError("Failure random source height must clear the ground.")
        self.trajectory_rollout_backend = str(
            self.cfg.incoming_ball_trajectory_rollout_backend
        )
        self.root_directed_sampling = bool(
            self.cfg.incoming_ball_torch_match_root_directed_sampling
        )
        self.horizontal_speed_range_m_s = tuple(
            float(value)
            for value in self.cfg.incoming_ball_torch_match_horizontal_speed_range_m_s
        )
        self.horizontal_angle_half_width_deg = float(
            self.cfg.incoming_ball_torch_match_horizontal_angle_half_width_deg
        )
        self.net_x_m = float(self.cfg.tennis_court_net_x_m)
        self.net_half_width_m = float(self.cfg.tennis_court_net_half_width_m)
        self.net_crossing_height_range_m = tuple(
            float(value)
            for value in self.cfg.incoming_ball_torch_match_net_crossing_height_range_m
        )
        self.maximum_initial_speed_m_s = float(
            self.cfg.incoming_ball_torch_match_maximum_initial_speed_m_s
        )
        self.deadline_offset_s = float(
            self.cfg.incoming_ball_torch_match_deadline_offset_s
        )
        if self.dt <= 0.0 or self.horizon_s <= 0.0:
            raise ValueError("Torch trajectory dt and horizon must be positive.")
        if self.maximum_distance < 0.0:
            raise ValueError("Torch match maximum distance must be non-negative.")
        if self.net_x_m <= 0.0 or self.net_half_width_m <= 0.0:
            raise ValueError("Tennis net X and half-width must be positive.")
        if (
            self.attempt_count <= 0
            or self.retry_rounds <= 0
            or self.motion_chunk_size <= 0
        ):
            raise ValueError(
                "Torch match attempts, retry rounds, and chunk size must be positive."
            )
        if self.hierarchical_top_k < 0:
            raise ValueError("Hierarchical motion Top-K must be non-negative.")
        if self.hierarchical_coarse_neighbor_radius < 0:
            raise ValueError(
                "Hierarchical coarse neighbor radius must be non-negative."
            )
        if not 0 <= self.adaptive_attempt_batch_size <= self.attempt_count:
            raise ValueError(
                "Adaptive attempt batch size must be zero or lie in "
                "[1, incoming_ball_torch_match_attempts]."
            )
        if self.cache_size < 0:
            raise ValueError("Torch match cache size must be non-negative.")
        if not 0.0 <= self.failure_trajectory_probability <= 1.0:
            raise ValueError("Failure trajectory probability must lie in [0, 1].")
        if self.failure_trajectory_xy_distance_threshold_m <= 0.0:
            raise ValueError("Failure trajectory XY-distance threshold must be positive.")
        if not 0.0 <= self.failure_no_net_fraction <= 1.0:
            raise ValueError("Failure no-net fraction must lie in [0, 1].")
        if self.root_directed_sampling and self.cache_size != 0:
            raise ValueError(
                "Fixed-court root-directed sampling requires cache_size=0."
            )
        if self.trajectory_rollout_backend not in {"torch", "warp_fused"}:
            raise ValueError(
                "Trajectory rollout backend must be 'torch' or 'warp_fused'."
            )
        if not math.isfinite(self.deadline_offset_s) or self.deadline_offset_s < 0.0:
            raise ValueError("Torch deadline offset must be finite and non-negative.")

        dtype = motion.target_position_w.dtype
        device = env.device

        def vector(name: str) -> torch.Tensor:
            value = torch.as_tensor(getattr(self.cfg, name), device=device, dtype=dtype)
            if value.shape != (3,) or not torch.all(torch.isfinite(value)):
                raise ValueError(f"{name} must contain three finite values.")
            return value

        self.position_min = vector("incoming_ball_torch_match_position_min")
        self.position_max = vector("incoming_ball_torch_match_position_max")
        self.velocity_min = vector("incoming_ball_torch_match_velocity_min")
        self.velocity_max = vector("incoming_ball_torch_match_velocity_max")
        self.fallback_position = vector("incoming_ball_initial_position")
        self.fallback_velocity = vector("incoming_ball_initial_velocity")
        if torch.any(self.position_min >= self.position_max):
            raise ValueError("Torch launch position minima must be below maxima.")
        if torch.any(self.velocity_min >= self.velocity_max):
            raise ValueError("Torch launch velocity minima must be below maxima.")

        target_indices = []
        for motion_cfg in motion.motion_configs:
            position_indices = [
                index
                for index, target in enumerate(motion_cfg.sub_targets)
                if target.goal_type == "position"
            ]
            if len(position_indices) != 1:
                raise ValueError(
                    "Torch motion matching requires exactly one position target "
                    f"per motion, got {len(position_indices)} for {motion_cfg.name!r}."
                )
            target_indices.append(position_indices[0])
        self.target_indices = torch.tensor(
            target_indices, dtype=torch.long, device=device
        )
        motion_ids = torch.arange(motion_count, device=device)
        if not torch.all(motion._target_pos_frame0_t[motion_ids, self.target_indices]):
            raise ValueError(
                "Torch motion matching requires reference_anchor_frame0 targets."
            )

        reference_position_0 = motion._stacked_body_pos_w[
            :, 0, motion.motion_anchor_body_index
        ]
        reference_quat_0 = motion._stacked_body_quat_w[
            :, 0, motion.motion_anchor_body_index
        ]
        aligned_quat_0 = quat_mul(
            quat_inv(yaw_quat(reference_quat_0)), reference_quat_0
        )
        aligned_anchor_position = torch.zeros_like(reference_position_0)
        aligned_anchor_position[:, 2] = reference_position_0[:, 2]
        target_means = motion._target_pos_means_t[motion_ids, self.target_indices]
        self.motion_target_positions = aligned_anchor_position + quat_apply(
            aligned_quat_0, target_means
        )
        self.nominal_contact_times = motion._nominal_contact_times
        self.minimum_contact_times = (
            self.nominal_contact_times / motion.cfg.max_contact_speedup
        )
        self.initial_angular_velocity = torch.as_tensor(
            self.cfg.incoming_ball_initial_angular_velocity,
            dtype=dtype,
            device=device,
        )
        self._plan_cache: dict[str, torch.Tensor] = {}
        self._plan_cache_ready = torch.zeros(
            env.num_envs, dtype=torch.bool, device=device
        )
        print(
            "[INFO]: Incoming-ball motion matcher enabled: "
            f"rollout_backend={self.trajectory_rollout_backend}, "
            f"motions={motion_count}, dt={self.dt:.4f}s, "
            f"horizon={self.horizon_s:.2f}s, attempts={self.attempt_count}, "
            f"retry_rounds={self.retry_rounds}, "
            f"adaptive_attempt_batch_size={self.adaptive_attempt_batch_size}, "
            f"motion_chunk_size={self.motion_chunk_size}, "
            f"hierarchical_top_k={self.hierarchical_top_k}, "
            f"hierarchical_coarse_neighbor_radius="
            f"{self.hierarchical_coarse_neighbor_radius}, "
            f"cache_size={self.cache_size}, "
            f"root_directed_sampling={self.root_directed_sampling}, "
            f"failure_probability={self.failure_trajectory_probability:.3f}, "
            f"failure_trajectory_xy_distance="
            f"{self.failure_trajectory_xy_distance_threshold_m:.3f}m, "
            f"failure_no_net_fraction={self.failure_no_net_fraction:.3f}, "
            f"max_distance={self.maximum_distance:.3f}m, "
            f"deadline_offset={self.deadline_offset_s:.3f}s"
        )

    def _sample_uniform(
        self, minimum: torch.Tensor, maximum: torch.Tensor, count: int
    ) -> torch.Tensor:
        return minimum + torch.rand(
            (count, 3), device=self.env.device, dtype=minimum.dtype
        ) * (maximum - minimum)

    def _physics_parameters(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = getattr(self.env, "_tennis_domain_randomization", None)
        count = len(env_ids)
        dtype = self.motion.target_position_w.dtype
        if state is not None:
            return (
                state.ball_mass_kg[env_ids],
                state.court_restitution[env_ids],
                state.ground_tangent_speed_retention[env_ids],
                state.drag_coefficient[env_ids],
            )
        return (
            torch.full(
                (count,), PHYSICS.ball.mass_kg, device=self.env.device, dtype=dtype
            ),
            torch.full(
                (count,), PHYSICS.court.restitution, device=self.env.device, dtype=dtype
            ),
            torch.full(
                (count,),
                sum(STANDARD_TENNIS_DOMAIN_RANDOMIZATION.ground_tangent_speed_retention)
                / 2.0,
                device=self.env.device,
                dtype=dtype,
            ),
            torch.full(
                (count,),
                PHYSICS.ball.drag_coefficient,
                device=self.env.device,
                dtype=dtype,
            ),
        )

    def _simulate_trajectories(
        self,
        initial_position: torch.Tensor,
        initial_velocity: torch.Tensor,
        *,
        mass: torch.Tensor,
        restitution: torch.Tensor,
        tangent_retention: torch.Tensor,
        drag: torch.Tensor,
        stop_origin_w: torch.Tensor | None = None,
        failure_rollout: bool = False,
    ):
        if self.trajectory_rollout_backend == "warp_fused" and not failure_rollout:
            from athlete.goal_cond_tracking.warp_tennis_planner import (
                simulate_tennis_trajectories_warp_fused,
            )

            rollout_fn = simulate_tennis_trajectories_warp_fused
        else:
            rollout_fn = simulate_tennis_trajectories_torch
        horizon_s = (
            self.failure_rollout_horizon_s if failure_rollout else self.horizon_s
        )
        rollout_kwargs = dict(
            ball_mass_kg=mass,
            court_restitution=restitution,
            tangent_speed_retention=tangent_retention,
            drag_coefficient=drag,
            dt=self.dt,
            horizon_s=horizon_s,
            net_x=self.net_x_m,
            net_half_width=self.net_half_width_m,
        )
        if failure_rollout and self.failure_observation_radius_m is not None:
            rollout_kwargs.update(
                stop_origin_w=stop_origin_w,
                stop_distance_m=self.failure_observation_radius_m,
                stop_speed_m_s=self.failure_stop_speed_m_s,
            )
        return rollout_fn(initial_position, initial_velocity, **rollout_kwargs)

    def _failure_trajectory_end_times(
        self, trajectories, stop_origins_w: torch.Tensor
    ) -> torch.Tensor:
        """First time a failed-ball rollout leaves 20 m or slows to rest."""
        if self.failure_observation_radius_m is None:
            return torch.full(
                (len(trajectories.positions),),
                self.failure_rollout_horizon_s,
                dtype=trajectories.times.dtype,
                device=trajectories.times.device,
            )
        distance = torch.linalg.vector_norm(
            trajectories.positions - stop_origins_w[:, None, :], dim=-1
        )
        speed = torch.linalg.vector_norm(trajectories.velocities, dim=-1)
        stopped = (distance >= self.failure_observation_radius_m) | (
            speed <= self.failure_stop_speed_m_s
        )
        reached_stop = stopped.any(dim=1)
        stop_index = stopped.to(torch.int64).argmax(dim=1)
        end_time = trajectories.times[stop_index]
        fallback = torch.full_like(end_time, self.failure_rollout_horizon_s)
        return torch.where(reached_stop, end_time, fallback)

    def _match_failure_candidates(self, trajectories, motion_target_positions):
        candidate_targets = (
            motion_target_positions
            if motion_target_positions.ndim == 3
            else motion_target_positions
        )
        return match_tennis_trajectories_to_motions_torch(
            trajectories,
            candidate_targets,
            self.minimum_contact_times,
            self.nominal_contact_times,
            maximum_distance=self.maximum_distance,
            motion_chunk_size=self.motion_chunk_size,
            contact_time_offset_s=self.deadline_offset_s,
            hierarchical_top_k=self.hierarchical_top_k,
            hierarchical_coarse_neighbor_radius=(
                self.hierarchical_coarse_neighbor_radius
            ),
            include_pre_bounce=True,
        )

    def _sample_candidate_round(
        self,
        mass: torch.Tensor,
        restitution: torch.Tensor,
        tangent_retention: torch.Tensor,
        drag: torch.Tensor,
        root_positions_xy: torch.Tensor,
        motion_target_positions: torch.Tensor,
        *,
        attempt_count: int | None = None,
    ) -> MotionResamplePlan:
        count = len(mass)
        attempts = self.attempt_count if attempt_count is None else attempt_count
        if attempts <= 0:
            raise ValueError("Candidate attempt count must be positive.")
        candidate_count = count * attempts
        if self.root_directed_sampling:
            initial_position, initial_velocity = (
                sample_root_directed_tennis_launches_torch(
                    root_positions_xy.repeat_interleave(attempts, dim=0),
                    launch_position_min=self.position_min,
                    launch_position_max=self.position_max,
                    horizontal_speed_range_m_s=self.horizontal_speed_range_m_s,
                    horizontal_angle_half_width_deg=(
                        self.horizontal_angle_half_width_deg
                    ),
                    net_crossing_height_range_m=(self.net_crossing_height_range_m),
                    maximum_initial_speed_m_s=self.maximum_initial_speed_m_s,
                    net_x_m=self.net_x_m,
                )
            )
        else:
            initial_position = self._sample_uniform(
                self.position_min, self.position_max, candidate_count
            )
            initial_velocity = self._sample_uniform(
                self.velocity_min, self.velocity_max, candidate_count
            )
        trajectories = self._simulate_trajectories(
            initial_position,
            initial_velocity,
            mass=mass.repeat_interleave(attempts),
            restitution=restitution.repeat_interleave(attempts),
            tangent_retention=tangent_retention.repeat_interleave(attempts),
            drag=drag.repeat_interleave(attempts),
        )
        candidate_motion_targets = (
            motion_target_positions.repeat_interleave(attempts, dim=0)
            if motion_target_positions.ndim == 3
            else motion_target_positions
        )
        match = match_tennis_trajectories_to_motions_torch(
            trajectories,
            candidate_motion_targets,
            self.minimum_contact_times,
            self.nominal_contact_times,
            maximum_distance=self.maximum_distance,
            motion_chunk_size=self.motion_chunk_size,
            contact_time_offset_s=self.deadline_offset_s,
            hierarchical_top_k=self.hierarchical_top_k,
            hierarchical_coarse_neighbor_radius=(
                self.hierarchical_coarse_neighbor_radius
            ),
        )
        candidate_distances = match.distances.view(count, attempts)
        best_attempt_index = candidate_distances.argmin(dim=1)
        selected = (
            torch.arange(count, device=self.env.device) * attempts + best_attempt_index
        )
        motion_ids = match.motion_ids[selected]
        return MotionResamplePlan(
            motion_ids=motion_ids,
            target_indices=self.target_indices[motion_ids],
            target_positions_w=match.target_positions[selected],
            contact_times=match.contact_times[selected],
            launch_positions_w=initial_position[selected],
            launch_linear_velocities_w=initial_velocity[selected],
            launch_angular_velocities_w=self.initial_angular_velocity.expand(
                count, -1
            ).clone(),
            match_distances=match.distances[selected],
            valid=match.valid[selected],
            attempts=best_attempt_index + 1,
        )

    def _sample_overhead_round(
        self, mass, restitution, tangent_retention, drag,
        root_positions_xy, motion_target_positions, root_positions_w,
    ) -> MotionResamplePlan:
        """Physically verify fast, unbounced passes above the current G1 root."""
        from athlete.goal_cond_tracking.overhead_launch import sample_overhead_candidates, select_overhead_candidates

        count, attempts = len(mass), self.attempt_count
        roots = root_positions_xy.repeat_interleave(attempts, dim=0)
        p, v = sample_overhead_candidates(
            roots, self.position_min, self.position_max,
            self.cfg.incoming_ball_failure_overhead_speed_range,
            self.cfg.incoming_ball_failure_overhead_height_range,
        )
        trajectories = self._simulate_trajectories(
            p, v, mass=mass.repeat_interleave(attempts),
            restitution=restitution.repeat_interleave(attempts),
            tangent_retention=tangent_retention.repeat_interleave(attempts),
            drag=drag.repeat_interleave(attempts),
            stop_origin_w=root_positions_w.repeat_interleave(attempts, dim=0),
            failure_rollout=True,
        )
        eligible = select_overhead_candidates(trajectories, roots).view(count, attempts)
        match = self._match_failure_candidates(
            trajectories,
            motion_target_positions.repeat_interleave(attempts, dim=0)
            if motion_target_positions.ndim == 3
            else motion_target_positions,
        )
        reachable = match.valid.view(count, attempts) & eligible
        unreachable = eligible & ~match.valid.view(count, attempts)
        select_unreachable = unreachable.any(dim=1)
        selectable = torch.where(select_unreachable[:, None], unreachable, reachable)
        scores = torch.rand_like(selectable, dtype=p.dtype).masked_fill(~selectable, -1.0)
        choice = scores.argmax(dim=1)
        selected = torch.arange(count, device=p.device) * attempts + choice
        has_candidate = selectable.any(dim=1)
        selected_reachable = reachable.any(dim=1) & ~select_unreachable
        ids = torch.where(
            selected_reachable,
            match.motion_ids[selected],
            torch.zeros(count, device=p.device, dtype=torch.long),
        )
        target = torch.where(
            selected_reachable[:, None],
            match.target_positions[selected],
            motion_target_positions[:, 0]
            if motion_target_positions.ndim == 3
            else motion_target_positions[0].expand(count, -1),
        ).clone()
        end_times = self._failure_trajectory_end_times(
            trajectories,
            root_positions_w.repeat_interleave(attempts, dim=0),
        )[selected]
        return MotionResamplePlan(
            motion_ids=ids,
            target_indices=self.target_indices[ids],
            target_positions_w=target,
            contact_times=torch.where(
                selected_reachable, match.contact_times[selected], end_times
            ),
            launch_positions_w=p[selected], launch_linear_velocities_w=v[selected],
            launch_angular_velocities_w=self.initial_angular_velocity.expand(count, -1).clone(),
            match_distances=torch.where(
                selected_reachable,
                match.distances[selected],
                torch.full((count,), self.maximum_distance + 1, device=p.device, dtype=p.dtype),
            ),
            valid=~select_unreachable,
            attempts=torch.where(has_candidate, choice + 1, torch.zeros_like(choice)),
        )

    def _sample_failure_round(
        self,
        mass: torch.Tensor,
        restitution: torch.Tensor,
        tangent_retention: torch.Tensor,
        drag: torch.Tensor,
        root_positions_xy: torch.Tensor,
        motion_target_positions: torch.Tensor,
        root_positions_w: torch.Tensor,
    ) -> MotionResamplePlan:
        """Retain a no-net or over-net-but-far candidate for each requested row."""
        count = len(mass)
        attempts = self.attempt_count
        candidate_count = count * attempts
        if self.root_directed_sampling:
            initial_position, initial_velocity = (
                sample_root_directed_tennis_launches_torch(
                    root_positions_xy.repeat_interleave(attempts, dim=0),
                    launch_position_min=self.position_min,
                    launch_position_max=self.position_max,
                    horizontal_speed_range_m_s=self.horizontal_speed_range_m_s,
                    horizontal_angle_half_width_deg=(
                        self.horizontal_angle_half_width_deg
                    ),
                    net_crossing_height_range_m=(self.net_crossing_height_range_m),
                    maximum_initial_speed_m_s=self.maximum_initial_speed_m_s,
                    net_x_m=self.net_x_m,
                )
            )
            no_net_attempts = round(attempts * self.failure_no_net_fraction)
            if self.failure_no_net_fraction > 0.0:
                no_net_attempts = max(1, no_net_attempts)
            if self.failure_no_net_fraction < 1.0:
                no_net_attempts = min(attempts - 1, no_net_attempts)
            attempt_ids = torch.arange(candidate_count, device=self.env.device) % attempts
            no_net_candidates = attempt_ids < no_net_attempts
            far_candidates = ~no_net_candidates
            if torch.any(far_candidates):
                candidate_roots_xy = root_positions_xy.repeat_interleave(
                    attempts, dim=0
                )
                initial_velocity[far_candidates] = (
                    retarget_tennis_launches_to_miss_roots_torch(
                        initial_position[far_candidates],
                        initial_velocity[far_candidates],
                        candidate_roots_xy[far_candidates],
                        minimum_xy_distance_m=(
                            self.failure_trajectory_xy_distance_threshold_m
                        ),
                        net_crossing_height_range_m=(
                            self.net_crossing_height_range_m
                        ),
                        maximum_initial_speed_m_s=self.maximum_initial_speed_m_s,
                        net_x_m=self.net_x_m,
                    )
                )
            if torch.any(no_net_candidates):
                no_net_position = initial_position[no_net_candidates]
                no_net_velocity = initial_velocity[no_net_candidates]
                time_to_net = (
                    (self.net_x_m - no_net_position[:, 0])
                    / no_net_velocity[:, 0].clamp(max=-1.0e-6)
                ).clamp_min(1.0e-6)
                low_crossing_height = 0.5 * PHYSICS.court.net_height_m
                no_net_vz = (
                    low_crossing_height
                    - no_net_position[:, 2]
                    + 0.5 * 9.81 * time_to_net.square()
                ) / time_to_net
                horizontal_speed_sq = torch.sum(
                    torch.square(no_net_velocity[:, :2]), dim=1
                )
                vertical_speed_cap = torch.sqrt(
                    (
                        self.maximum_initial_speed_m_s**2 - horizontal_speed_sq
                    ).clamp_min(0.0)
                )
                initial_velocity[no_net_candidates, 2] = no_net_vz.clamp(
                    -vertical_speed_cap, vertical_speed_cap
                )
        else:
            initial_position = self._sample_uniform(
                self.position_min, self.position_max, candidate_count
            )
            initial_velocity = self._sample_uniform(
                self.velocity_min, self.velocity_max, candidate_count
            )

        random_direction_fraction = self.failure_random_direction_fraction
        if random_direction_fraction > 0.0:
            random_attempts = min(
                attempts,
                max(1, round(attempts * random_direction_fraction)),
            )
            attempt_ids = torch.arange(candidate_count, device=self.env.device) % attempts
            random_direction_mask = attempt_ids >= attempts - random_attempts
            candidate_roots_xy = root_positions_xy.repeat_interleave(
                attempts, dim=0
            )
            radius_min, radius_max = self.failure_random_position_radius_range_m
            source_radius = radius_min + torch.rand(
                count * random_attempts,
                device=self.env.device,
                dtype=initial_position.dtype,
            ) * (radius_max - radius_min)
            source_azimuth = torch.rand_like(source_radius) * (2.0 * math.pi)
            initial_position[random_direction_mask, 0] = (
                candidate_roots_xy[random_direction_mask, 0]
                + source_radius * torch.cos(source_azimuth)
            )
            initial_position[random_direction_mask, 1] = (
                candidate_roots_xy[random_direction_mask, 1]
                + source_radius * torch.sin(source_azimuth)
            )
            height_min, height_max = self.failure_random_position_height_range_m
            initial_position[random_direction_mask, 2] = height_min + torch.rand_like(
                source_radius
            ) * (height_max - height_min)
            initial_velocity[random_direction_mask] = (
                randomize_tennis_launch_directions_torch(
                    initial_position[random_direction_mask],
                    horizontal_speed_range_m_s=self.horizontal_speed_range_m_s,
                    vertical_speed_range_m_s=(
                        self.failure_random_vertical_speed_range_m_s
                    ),
                    maximum_initial_speed_m_s=self.maximum_initial_speed_m_s,
                )
            )

        trajectories = self._simulate_trajectories(
            initial_position,
            initial_velocity,
            mass=mass.repeat_interleave(attempts),
            restitution=restitution.repeat_interleave(attempts),
            tangent_retention=tangent_retention.repeat_interleave(attempts),
            drag=drag.repeat_interleave(attempts),
            stop_origin_w=root_positions_w.repeat_interleave(attempts, dim=0),
            failure_rollout=True,
        )
        failure = classify_tennis_failure_trajectories(
            trajectories,
            root_positions_xy.repeat_interleave(attempts, dim=0),
            trajectory_xy_distance_threshold_m=(
                self.failure_trajectory_xy_distance_threshold_m
            ),
            net_x=self.net_x_m,
            net_half_width=self.net_half_width_m,
        )
        no_net = failure.did_not_clear_net.view(count, attempts)
        far_trajectory = failure.trajectory_too_far.view(count, attempts)
        any_failure = failure.failure.view(count, attempts)
        match = self._match_failure_candidates(
            trajectories,
            motion_target_positions.repeat_interleave(attempts, dim=0)
            if motion_target_positions.ndim == 3
            else motion_target_positions,
        )
        reachable = match.valid.view(count, attempts)
        uncatchable = any_failure & ~reachable
        no_net_uncatchable = no_net & ~reachable
        far_uncatchable = far_trajectory & ~reachable
        prefer_no_net = torch.rand(count, device=self.env.device) < self.failure_no_net_fraction
        preferred = torch.where(
            prefer_no_net[:, None], no_net_uncatchable, far_uncatchable
        )
        preferred = torch.where(
            preferred.any(dim=1, keepdim=True), preferred, uncatchable
        )
        use_failure = preferred.any(dim=1)
        selectable = torch.where(use_failure[:, None], preferred, reachable)
        has_candidate = selectable.any(dim=1)
        selection_score = torch.rand(
            (count, attempts), device=self.env.device,
            dtype=trajectories.positions.dtype,
        )
        selection_score.masked_fill_(~selectable, -1.0)
        selected_attempt = selection_score.argmax(dim=1)
        selected = (
            torch.arange(count, device=self.env.device) * attempts + selected_attempt
        )

        selected_reachable = has_candidate & ~use_failure
        motion_ids = torch.where(
            selected_reachable,
            match.motion_ids[selected],
            torch.zeros(count, dtype=torch.long, device=self.env.device),
        )
        default_target = (
            motion_target_positions[:, 0]
            if motion_target_positions.ndim == 3
            else motion_target_positions[0].expand(count, -1)
        )
        target_positions = torch.where(
            selected_reachable[:, None], match.target_positions[selected], default_target
        ).clone()
        match_distances = torch.where(
            selected_reachable,
            match.distances[selected],
            torch.full_like(match.distances[selected], self.maximum_distance + 1.0),
        )
        failure_end_times = self._failure_trajectory_end_times(
            trajectories,
            root_positions_w.repeat_interleave(attempts, dim=0),
        )[selected]
        contact_times = torch.where(
            selected_reachable, match.contact_times[selected], failure_end_times
        )
        return MotionResamplePlan(
            motion_ids=motion_ids,
            target_indices=self.target_indices[motion_ids],
            target_positions_w=target_positions,
            contact_times=contact_times,
            launch_positions_w=initial_position[selected],
            launch_linear_velocities_w=initial_velocity[selected],
            launch_angular_velocities_w=self.initial_angular_velocity.expand(
                count, -1
            ).clone(),
            match_distances=match_distances,
            valid=~(has_candidate & use_failure),
            attempts=torch.where(
                has_candidate, selected_attempt + 1,
                torch.zeros_like(selected_attempt),
            ),
        )

    @torch.no_grad()
    def _planning_targets(
        self, env_ids: torch.Tensor, *, align_to_robot: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(env_ids)
        if not align_to_robot:
            return (
                torch.zeros(
                    (count, 2),
                    device=self.env.device,
                    dtype=self.motion_target_positions.dtype,
                ),
                self.motion_target_positions,
            )

        origins = self.env.scene.env_origins[env_ids]
        robot_position_local = self.motion.robot_anchor_pos_w[env_ids] - origins
        robot_yaw_w = yaw_quat(self.motion.robot_anchor_quat_w[env_ids])
        motion_count = len(self.motion_target_positions)
        aligned_targets = quat_apply(
            robot_yaw_w[:, None, :].expand(-1, motion_count, -1).reshape(-1, 4),
            self.motion_target_positions[None, :, :]
            .expand(count, -1, -1)
            .reshape(-1, 3),
        ).reshape(count, motion_count, 3)
        aligned_targets[:, :, :2] += robot_position_local[:, None, :2]
        return robot_position_local[:, :2], aligned_targets

    @torch.no_grad()
    def _plan_uncached(
        self,
        env_ids: torch.Tensor,
        *,
        align_to_robot: bool = False,
        allow_failure_trajectories: bool = True,
    ) -> MotionResamplePlan:
        count = len(env_ids)
        mass, restitution, tangent_retention, drag = self._physics_parameters(env_ids)
        root_positions_xy, motion_target_positions = self._planning_targets(
            env_ids, align_to_robot=align_to_robot
        )
        origins = self.env.scene.env_origins[env_ids]
        robot_anchor_w = getattr(
            getattr(self, "motion", None), "robot_anchor_pos_w", None
        )
        if robot_anchor_w is None:
            root_positions_w = torch.cat(
                (root_positions_xy, torch.zeros_like(root_positions_xy[:, :1])), dim=1
            )
        else:
            root_positions_w = robot_anchor_w[env_ids] - origins
            root_positions_w = root_positions_w.clone()
            root_positions_w[:, :2] = root_positions_xy
        total_attempt_budget = self.attempt_count * self.retry_rounds
        wave_size = self.adaptive_attempt_batch_size or self.attempt_count
        wave_sizes = []
        remaining_attempts = total_attempt_budget
        while remaining_attempts > 0:
            current_wave_size = min(wave_size, remaining_attempts)
            wave_sizes.append(current_wave_size)
            remaining_attempts -= current_wave_size

        initial = self._sample_candidate_round(
            mass,
            restitution,
            tangent_retention,
            drag,
            root_positions_xy,
            motion_target_positions,
            attempt_count=wave_sizes[0],
        )
        best_distance = initial.match_distances.clone()
        best_motion = initial.motion_ids.clone()
        best_target = initial.target_positions_w.clone()
        best_contact_time = initial.contact_times.clone()
        best_position = initial.launch_positions_w.clone()
        best_velocity = initial.launch_linear_velocities_w.clone()
        best_valid = initial.valid.clone()
        best_attempt = initial.attempts.clone()

        attempt_offset = wave_sizes[0]
        for current_wave_size in wave_sizes[1:]:
            retry_env_ids = (~best_valid).nonzero().flatten()
            if len(retry_env_ids) == 0:
                break
            retry = self._sample_candidate_round(
                mass[retry_env_ids],
                restitution[retry_env_ids],
                tangent_retention[retry_env_ids],
                drag[retry_env_ids],
                root_positions_xy[retry_env_ids],
                (
                    motion_target_positions[retry_env_ids]
                    if motion_target_positions.ndim == 3
                    else motion_target_positions
                ),
                attempt_count=current_wave_size,
            )
            better = retry.match_distances < best_distance[retry_env_ids]
            update_ids = retry_env_ids[better]
            retry_ids = better.nonzero().flatten()
            best_distance[update_ids] = retry.match_distances[retry_ids]
            best_motion[update_ids] = retry.motion_ids[retry_ids]
            best_target[update_ids] = retry.target_positions_w[retry_ids]
            best_contact_time[update_ids] = retry.contact_times[retry_ids]
            best_position[update_ids] = retry.launch_positions_w[retry_ids]
            best_velocity[update_ids] = retry.launch_linear_velocities_w[retry_ids]
            best_valid[update_ids] = retry.valid[retry_ids]
            best_attempt[update_ids] = retry.attempts[retry_ids] + attempt_offset
            attempt_offset += current_wave_size

        missing = ~torch.isfinite(best_distance)
        missing_ids = missing.nonzero().flatten()
        if len(missing_ids) > 0:
            fallback_position = self.fallback_position.expand(len(missing_ids), -1)
            fallback_velocity = self.fallback_velocity.expand(len(missing_ids), -1)
            fallback_trajectories = self._simulate_trajectories(
                fallback_position,
                fallback_velocity,
                mass=mass[missing_ids],
                restitution=restitution[missing_ids],
                tangent_retention=tangent_retention[missing_ids],
                drag=drag[missing_ids],
            )
            fallback_targets = (
                motion_target_positions[missing_ids]
                if motion_target_positions.ndim == 3
                else motion_target_positions
            )
            fallback_match = match_tennis_trajectories_to_motions_torch(
                fallback_trajectories,
                fallback_targets,
                self.minimum_contact_times,
                self.nominal_contact_times,
                maximum_distance=self.maximum_distance,
                motion_chunk_size=self.motion_chunk_size,
                contact_time_offset_s=self.deadline_offset_s,
                hierarchical_top_k=self.hierarchical_top_k,
                hierarchical_coarse_neighbor_radius=(
                    self.hierarchical_coarse_neighbor_radius
                ),
            )
            best_distance[missing_ids] = fallback_match.distances
            best_motion[missing_ids] = fallback_match.motion_ids
            best_target[missing_ids] = fallback_match.target_positions
            best_contact_time[missing_ids] = fallback_match.contact_times
            best_position[missing_ids] = fallback_position
            best_velocity[missing_ids] = fallback_velocity
            best_valid[missing_ids] = fallback_match.valid
            best_attempt[missing_ids] = total_attempt_budget + 1

        if torch.any(~torch.isfinite(best_distance)):
            failed = int((~torch.isfinite(best_distance)).sum().item())
            raise RuntimeError(
                "Torch trajectory matching produced no valid post-bounce segment "
                f"for {failed}/{count} environments after "
                f"{total_attempt_budget} attempts."
            )

        failure_probability = (
            getattr(self, "failure_trajectory_probability", 0.0)
            if allow_failure_trajectories
            else 0.0
        )
        failure_requested = (
            torch.rand(count, device=self.env.device) < failure_probability
        )
        overhead_requested = failure_requested & (
            torch.rand(count, device=self.env.device) < getattr(self, 'failure_overhead_fraction', 0.0)
        ) if getattr(self, 'failure_overhead_fraction', 0.0) > 0 else torch.zeros_like(failure_requested)
        pending_failure_ids = torch.where(failure_requested)[0]
        for _ in range(self.retry_rounds):
            if len(pending_failure_ids) == 0:
                break
            failure_plan = self._sample_failure_round(
                mass[pending_failure_ids],
                restitution[pending_failure_ids],
                tangent_retention[pending_failure_ids],
                drag[pending_failure_ids],
                root_positions_xy[pending_failure_ids],
                (
                motion_target_positions[pending_failure_ids]
                if motion_target_positions.ndim == 3
                else motion_target_positions
                ),
                root_positions_w[pending_failure_ids],
            )
            overhead = overhead_requested[pending_failure_ids]
            if torch.any(overhead):
                rows = pending_failure_ids[overhead]
                overhead_plan = self._sample_overhead_round(
                    mass[rows], restitution[rows], tangent_retention[rows], drag[rows],
                    root_positions_xy[rows],
                    motion_target_positions[rows] if motion_target_positions.ndim == 3 else motion_target_positions,
                    root_positions_w[rows],
                )
                for name in MotionResamplePlan.__dataclass_fields__:
                    getattr(failure_plan, name)[overhead] = getattr(overhead_plan, name)
            selected = failure_plan.attempts > 0
            selected_ids = pending_failure_ids[selected]
            if len(selected_ids) > 0:
                best_motion[selected_ids] = failure_plan.motion_ids[selected]
                best_target[selected_ids] = failure_plan.target_positions_w[selected]
                best_contact_time[selected_ids] = failure_plan.contact_times[selected]
                best_position[selected_ids] = failure_plan.launch_positions_w[selected]
                best_velocity[selected_ids] = failure_plan.launch_linear_velocities_w[selected]
                best_distance[selected_ids] = failure_plan.match_distances[selected]
                best_valid[selected_ids] = failure_plan.valid[selected]
                best_attempt[selected_ids] = failure_plan.attempts[selected]
            pending_failure_ids = pending_failure_ids[~selected]

        target_indices = self.target_indices[best_motion]
        origins = self.env.scene.env_origins[env_ids]
        launch_position_w = best_position + origins
        target_position_w = best_target + origins
        angular_velocity = self.initial_angular_velocity.expand(count, -1).clone()
        if getattr(self, 'failure_overhead_fraction', 0.0) > 0:
            self.env.extras.setdefault('log', {}).update({
                'Incoming/failed_requested_fraction': failure_requested.float().mean(),
                'Incoming/failed_actual_fraction': (~best_valid).float().mean(),
                'Incoming/overhead_actual_fraction': (overhead_requested & ~best_valid).float().mean(),
            })
        if self.verbose_samples:
            valid_count = int(best_valid.sum().item())
            print(
                "[TORCH MOTION MATCH] "
                f"envs={count} valid={valid_count}/{count} "
                f"failure={count - valid_count}/{count} "
                f"mean_error={best_distance.mean().item():.3f}m "
                f"max_error={best_distance.max().item():.3f}m",
                flush=True,
            )
        return MotionResamplePlan(
            motion_ids=best_motion,
            target_indices=target_indices,
            target_positions_w=target_position_w,
            contact_times=best_contact_time,
            launch_positions_w=launch_position_w,
            launch_linear_velocities_w=best_velocity,
            launch_angular_velocities_w=angular_velocity,
            match_distances=best_distance,
            valid=best_valid,
            attempts=best_attempt,
        )

    @torch.no_grad()
    def _populate_cache(self, env_ids: torch.Tensor) -> None:
        """Generate a valid initial-plan bank using each environment's physics."""
        if self.cache_size <= 0 or len(env_ids) == 0:
            return
        field_names = MotionResamplePlan.__dataclass_fields__.keys()
        valid_count = 0
        for slot in range(self.cache_size):
            plan = self._plan_uncached(env_ids, allow_failure_trajectories=False)
            if not self._plan_cache:
                for name in field_names:
                    value = getattr(plan, name)
                    shape = (self.env.num_envs, self.cache_size, *value.shape[1:])
                    self._plan_cache[name] = torch.empty(
                        shape, dtype=value.dtype, device=value.device
                    )
            for name in field_names:
                self._plan_cache[name][env_ids, slot] = getattr(plan, name)
            valid_count += int(plan.valid.sum().item())
        self._plan_cache_ready[env_ids] = True
        print(
            "[INFO]: Torch motion-plan cache populated: "
            f"envs={len(env_ids)}, slots={self.cache_size}, "
            f"valid={valid_count}/{len(env_ids) * self.cache_size}",
            flush=True,
        )

    @torch.no_grad()
    def _require_valid_initial_plan(
        self, env_ids: torch.Tensor, plan: MotionResamplePlan
    ) -> MotionResamplePlan:
        """Retry unmatched rows so an episode never starts with a failure sample."""
        if torch.all(plan.valid):
            return plan
        for _ in range(self.retry_rounds):
            pending = ~plan.valid
            if not torch.any(pending):
                return plan
            retry = self._plan_uncached(
                env_ids[pending], allow_failure_trajectories=False
            )
            for name in MotionResamplePlan.__dataclass_fields__:
                getattr(plan, name)[pending] = getattr(retry, name)

        failed = int((~plan.valid).sum().item())
        if failed > 0:
            raise RuntimeError(
                "Initial incoming-ball planning could not find a valid motion "
                f"match for {failed}/{len(env_ids)} environments."
            )
        return plan

    @torch.no_grad()
    def __call__(self, env_ids: torch.Tensor) -> MotionResamplePlan:
        """Plan the first ball without intentionally injecting a failed match."""
        if self.cache_size <= 0:
            plan = self._plan_uncached(env_ids, allow_failure_trajectories=False)
            return self._require_valid_initial_plan(env_ids, plan)

        missing = env_ids[~self._plan_cache_ready[env_ids]]
        self._populate_cache(missing)
        cached_valid = self._plan_cache["valid"][env_ids]
        random_scores = torch.rand(
            cached_valid.shape, device=self.env.device, dtype=torch.float32
        )
        random_scores.masked_fill_(~cached_valid, -1.0)
        slots = random_scores.argmax(dim=1)
        no_valid_slot = ~cached_valid.any(dim=1)
        if torch.any(no_valid_slot):
            cached_distances = self._plan_cache["match_distances"][env_ids]
            slots[no_valid_slot] = cached_distances[no_valid_slot].argmin(dim=1)

        plan = MotionResamplePlan(
            **{name: value[env_ids, slots] for name, value in self._plan_cache.items()}
        )
        return self._require_valid_initial_plan(env_ids, plan)

    @torch.no_grad()
    def plan_for_motion_chain(self, env_ids: torch.Tensor) -> MotionResamplePlan:
        """Select a cached plan and align it to the robot's current planar pose."""
        if self.root_directed_sampling:
            return self._plan_uncached(
                env_ids,
                align_to_robot=True,
                allow_failure_trajectories=True,
            )

        if getattr(self, "failure_trajectory_probability", 0.0) > 0.0:
            plan = self._plan_uncached(env_ids, allow_failure_trajectories=True)
        else:
            plan = self(env_ids)
        env_origins = self.env.scene.env_origins[env_ids]
        robot_position_w = self.motion.robot_anchor_pos_w[env_ids]
        robot_yaw_w = yaw_quat(self.motion.robot_anchor_quat_w[env_ids])

        aligned_origin_w = env_origins.clone()
        aligned_origin_w[:, :2] = robot_position_w[:, :2]

        def align_point(point_w: torch.Tensor) -> torch.Tensor:
            return aligned_origin_w + quat_apply(robot_yaw_w, point_w - env_origins)

        return MotionResamplePlan(
            motion_ids=plan.motion_ids,
            target_indices=plan.target_indices,
            target_positions_w=align_point(plan.target_positions_w),
            contact_times=plan.contact_times,
            launch_positions_w=align_point(plan.launch_positions_w),
            launch_linear_velocities_w=quat_apply(
                robot_yaw_w, plan.launch_linear_velocities_w
            ),
            launch_angular_velocities_w=quat_apply(
                robot_yaw_w, plan.launch_angular_velocities_w
            ),
            match_distances=plan.match_distances,
            valid=plan.valid,
            attempts=plan.attempts,
        )


class TennisIncomingBallController:
    """Launch one configured incoming trajectory to arrive at the strike deadline."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        *,
        verbose_samples: bool = False,
    ) -> None:
        self.env = env
        self.ball = env.scene["tennis_ball"]
        self.motion = env.command_manager.get_term("motion")
        self.verbose_samples = verbose_samples
        cfg = self.motion.cfg
        if not cfg.incoming_ball_launch_enabled:
            raise ValueError("Incoming-ball launch is not enabled for this command.")
        if not hasattr(self.motion, "time_remaining") or not hasattr(
            self.motion, "contact_time"
        ):
            raise TypeError("Incoming-ball launch requires a deadline-aware command.")
        if cfg.incoming_ball_flight_time_s <= 0.0:
            raise ValueError("incoming_ball_flight_time_s must be positive.")

        self._motion_planner = None
        torch_match_enabled = bool(
            getattr(cfg, "incoming_ball_torch_match_enabled", False)
        )
        if torch_match_enabled:
            self._motion_planner = TorchIncomingBallMotionPlanner(
                env,
                self.motion,
                verbose_samples=verbose_samples,
            )
            self.motion.set_motion_plan_provider(self._motion_planner)

        dtype = self.ball.data.data.qpos.dtype
        device = env.device

        def vector(name: str) -> torch.Tensor:
            value = torch.as_tensor(getattr(cfg, name), device=device, dtype=dtype)
            if value.shape != (3,):
                raise ValueError(f"{name} must contain exactly three values.")
            if not torch.all(torch.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values.")
            return value

        self.flight_time_s = float(cfg.incoming_ball_flight_time_s)
        self._trajectory_drives_deadline = bool(
            cfg.incoming_ball_trajectory_drives_deadline or torch_match_enabled
        )
        if self._trajectory_drives_deadline:
            print(
                "[INFO]: Incoming-ball trajectory drives strike deadline: "
                f"contact_time={self.flight_time_s:.6f}s, launch=first control step"
            )
        self._initial_position = vector("incoming_ball_initial_position")
        self._initial_velocity = vector("incoming_ball_initial_velocity")
        self._initial_angular_velocity = vector(
            "incoming_ball_initial_angular_velocity"
        )
        self._position_noise_std = vector("incoming_ball_position_noise_std")
        self._velocity_noise_std = vector("incoming_ball_velocity_noise_std")
        self._max_strike_deviation = float(cfg.incoming_ball_max_strike_deviation)
        if torch.any(self._position_noise_std < 0.0) or torch.any(
            self._velocity_noise_std < 0.0
        ):
            raise ValueError(
                "Incoming-ball noise standard deviations must be non-negative."
            )
        if (
            not math.isfinite(self._max_strike_deviation)
            or self._max_strike_deviation < 0.0
        ):
            raise ValueError("incoming_ball_max_strike_deviation must be non-negative.")
        if self._max_strike_deviation > 0.0 and (
            self._position_noise_std[2] != 0.0
            or torch.any(self._velocity_noise_std != 0.0)
        ):
            raise ValueError(
                "A bounded strike deviation requires horizontal position noise only; "
                "vertical position and velocity noise must be zero."
            )
        if self._max_strike_deviation > 0.0:
            print(
                "[INFO]: Incoming-ball trajectory randomization: "
                "position_std="
                f"{tuple(round(float(value), 3) for value in self._position_noise_std)}m "
                f"strike_deviation<={self._max_strike_deviation:.3f}m"
            )

        self._previous_time_remaining = torch.full_like(
            self.motion.time_remaining, torch.inf
        )
        self._previous_motion_ids = torch.full_like(self.motion.which_motion, -1)
        motion_chain_count = getattr(
            self.motion,
            "motion_chain_count",
            torch.zeros_like(self.motion.which_motion),
        )
        self._previous_motion_chain_count = motion_chain_count.clone()
        self._force_restart = torch.ones(env.num_envs, dtype=torch.bool, device=device)
        self._launched = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self._launch_pose = torch.zeros((env.num_envs, 7), device=device, dtype=dtype)
        self._launch_pose[:, 3] = 1.0
        self._launch_velocity = torch.zeros(
            (env.num_envs, 6), device=device, dtype=dtype
        )
        self._launch_translation = torch.zeros(
            (env.num_envs, 3), device=device, dtype=dtype
        )
        self._zero_velocity = torch.zeros_like(self._launch_velocity)

    def before_step(self) -> None:
        time_remaining = self.motion.time_remaining
        motion_ids = self.motion.which_motion
        motion_chain_count = getattr(
            self.motion, "motion_chain_count", self._previous_motion_chain_count
        )
        restarted = (
            self._force_restart
            | (time_remaining > self._previous_time_remaining + 0.5 * self.env.step_dt)
            | (motion_ids != self._previous_motion_ids)
            | (motion_chain_count != self._previous_motion_chain_count)
        )
        self._force_restart[restarted] = False
        self._launched[restarted] = False

        if self._motion_planner is not None:
            deadline_crossed = (
                self._launched
                & (self._previous_time_remaining > 0.0)
                & (time_remaining <= 0.0)
                & ~restarted
            )
            if torch.any(deadline_crossed):
                env_ids = torch.where(deadline_crossed)[0]
                target_indices = self.motion.planned_target_index[env_ids]
                target_position_w = self.motion.target_position_w[
                    env_ids, target_indices
                ]
                prediction_error = torch.linalg.vector_norm(
                    self.ball.data.root_link_pos_w[env_ids] - target_position_w,
                    dim=-1,
                )
                self.motion.metrics["trajectory_prediction_error"][env_ids] = (
                    prediction_error
                )

        if torch.any(restarted):
            count = int(restarted.sum().item())
            position_noise = (
                torch.randn(
                    (count, 3), device=self.env.device, dtype=self._launch_pose.dtype
                )
                * self._position_noise_std
            )
            if self._max_strike_deviation > 0.0:
                noise_norm = torch.linalg.vector_norm(
                    position_noise, dim=-1, keepdim=True
                )
                scale = torch.clamp(
                    self._max_strike_deviation
                    / noise_norm.clamp_min(torch.finfo(position_noise.dtype).eps),
                    max=1.0,
                )
                position_noise *= scale
            velocity_noise = (
                torch.randn(
                    (count, 3),
                    device=self.env.device,
                    dtype=self._launch_velocity.dtype,
                )
                * self._velocity_noise_std
            )
            self._launch_translation[restarted] = position_noise
            self._launch_pose[restarted, :3] = (
                self.env.scene.env_origins[restarted]
                + self._initial_position
                + position_noise
            )
            self._launch_velocity[restarted, :3] = (
                self._initial_velocity + velocity_noise
            )
            self._launch_velocity[restarted, 3:] = self._initial_angular_velocity

            if self._motion_planner is not None:
                planned_position = self.motion.planned_launch_position_w[restarted]
                planned_velocity = self.motion.planned_launch_linear_velocity_w[
                    restarted
                ]
                planned_angular_velocity = (
                    self.motion.planned_launch_angular_velocity_w[restarted]
                )
                if not torch.all(torch.isfinite(planned_position)) or not torch.all(
                    torch.isfinite(planned_velocity)
                ):
                    raise RuntimeError(
                        "Torch-matched ball launch was not populated before stepping."
                    )
                self._launch_translation[restarted] = 0.0
                self._launch_pose[restarted, :3] = planned_position
                self._launch_velocity[restarted, :3] = planned_velocity
                self._launch_velocity[restarted, 3:] = planned_angular_velocity

            if self.verbose_samples:
                for env_id in torch.where(restarted)[0].tolist():
                    motion_id = int(motion_ids[env_id].item())
                    motion_file = Path(self.motion.cfg.motion_files[motion_id]).name
                    flight_time_s = (
                        float(self.motion.contact_time[env_id].item())
                        if self._motion_planner is not None
                        else self.flight_time_s
                    )
                    delay_s = max(
                        0.0,
                        float(self.motion.contact_time[env_id].item()) - flight_time_s,
                    )
                    p0 = self._launch_pose[env_id, :3].tolist()
                    v0 = self._launch_velocity[env_id, :3].tolist()
                    strike_deviation = torch.linalg.vector_norm(
                        self._launch_translation[env_id]
                    ).item()
                    print(
                        f"[INCOMING BALL SAMPLE] env={env_id} file={motion_file} "
                        f"contact_time={self.motion.contact_time[env_id].item():.3f}s "
                        f"launch_delay={delay_s:.3f}s flight_time={flight_time_s:.3f}s "
                        f"p0=({p0[0]:.3f}, {p0[1]:.3f}, {p0[2]:.3f}) "
                        f"v0=({v0[0]:.3f}, {v0[1]:.3f}, {v0[2]:.3f}) "
                        f"strike_deviation={strike_deviation:.3f}m",
                        flush=True,
                    )

        launch_due = (
            torch.ones_like(restarted)
            if self._trajectory_drives_deadline
            else (time_remaining <= self.flight_time_s)
        )
        launch_now = (~self._launched) & launch_due
        self._launched |= launch_now
        held = ~self._launched

        qpos_adr = self.ball.data.indexing.free_joint_q_adr
        qvel_adr = self.ball.data.indexing.free_joint_v_adr
        current_pose = self.ball.data.data.qpos[:, qpos_adr]
        current_velocity = self.ball.data.data.qvel[:, qvel_adr]
        set_launch_pose = held | launch_now
        self.ball.data.data.qpos[:, qpos_adr] = torch.where(
            set_launch_pose[:, None], self._launch_pose, current_pose
        )
        next_velocity = torch.where(
            held[:, None], self._zero_velocity, current_velocity
        )
        self.ball.data.data.qvel[:, qvel_adr] = torch.where(
            launch_now[:, None], self._launch_velocity, next_velocity
        )

        if self.verbose_samples:
            for env_id in torch.where(launch_now)[0].tolist():
                print(
                    f"[INCOMING BALL LAUNCH] env={env_id} "
                    f"time_remaining={time_remaining[env_id].item():.3f}s",
                    flush=True,
                )

        self._previous_time_remaining.copy_(time_remaining)
        self._previous_motion_ids.copy_(motion_ids)
        self._previous_motion_chain_count.copy_(motion_chain_count)

    def after_step(self, reset_mask: torch.Tensor | None = None) -> None:
        """Schedule reset ball state to be restored before the next physics step."""
        if reset_mask is not None:
            self._force_restart |= reset_mask


class TennisBallAerodynamicsController:
    """Write the shared ball-only aerodynamic wrench before every sim substep."""

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        self.env = env
        self.ball = env.scene["tennis_ball"]
        self.domain_state: TennisDomainRandomizationState | None = getattr(
            env, "_tennis_domain_randomization", None
        )
        self._pre_step_linear_velocity_w: torch.Tensor | None = None
        self._ground_impact_pending = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self._ground_impact_age = torch.zeros(
            env.num_envs, dtype=torch.int32, device=env.device
        )
        self._ground_impact_velocity_w = torch.zeros(
            (env.num_envs, 3),
            dtype=self.ball.data.root_link_lin_vel_w.dtype,
            device=env.device,
        )

    def before_sim_step(self) -> None:
        linear_velocity_w = self.ball.data.root_link_lin_vel_w
        angular_velocity_w = self.ball.data.root_link_ang_vel_w
        domain_state = getattr(self, "domain_state", None)
        if domain_state is not None:
            self._pre_step_linear_velocity_w = linear_velocity_w.clone()
        force_w, torque_w = tennis_ball_aerodynamic_wrench_torch(
            linear_velocity_w,
            angular_velocity_w,
            drag_coefficient=(
                None if domain_state is None else domain_state.drag_coefficient
            ),
            magnus_coefficient=0.0,
        )
        self.ball.write_external_wrench_to_sim(
            force_w,
            torque_w,
            body_ids=(0,),
        )

    def after_sim_step(self) -> None:
        """Apply the sampled horizontal speed retention at a ground rebound."""
        domain_state = getattr(self, "domain_state", None)
        pre_step_velocity = getattr(self, "_pre_step_linear_velocity_w", None)
        if domain_state is None or pre_step_velocity is None:
            return
        current_linear = self.ball.data.root_link_lin_vel_w
        position = self.ball.data.root_link_pos_w
        court_height = self.env.scene.env_origins[:, 2] + BALL_RADIUS + 0.03
        bounced = (
            (pre_step_velocity[:, 2] < -0.05)
            & (current_linear[:, 2] > 0.05)
            & (position[:, 2] <= court_height)
        )
        if domain_state.explicit_ground_rebound:
            self._ground_impact_age[self._ground_impact_pending] += 1
            entering = (
                ~self._ground_impact_pending
                & (current_linear[:, 2] < -0.05)
                & (position[:, 2] <= court_height)
            )
            self._ground_impact_pending[entering] = True
            self._ground_impact_age[entering] = 0
            self._ground_impact_velocity_w[entering] = current_linear[entering]
            height_to_ground = (
                position[entering, 2]
                - self.env.scene.env_origins[entering, 2]
                - BALL_RADIUS
            ).clamp_min(0.0)
            vertical_speed = current_linear[entering, 2].abs()
            self._ground_impact_velocity_w[entering, 2] = -torch.sqrt(
                vertical_speed.square() + 2.0 * 9.81 * height_to_ground
            )

            ground_height = self.env.scene.env_origins[:, 2] + BALL_RADIUS
            impact_due = self._ground_impact_pending & (
                position[:, 2] <= ground_height + 0.002
            )
            env_ids = impact_due.nonzero().flatten()
            if len(env_ids) > 0:
                root_pose = self.ball.data.root_link_pose_w[env_ids].clone()
                root_pose[:, 2] = ground_height[env_ids] + 1.0e-4
                root_velocity = torch.cat(
                    (current_linear, self.ball.data.root_link_ang_vel_w), dim=-1
                )[env_ids].clone()
                root_velocity[:, :2] = (
                    self._ground_impact_velocity_w[env_ids, :2]
                    * domain_state.ground_tangent_speed_retention[env_ids, None]
                )
                root_velocity[:, 2] = (
                    -self._ground_impact_velocity_w[env_ids, 2]
                    * domain_state.court_restitution[env_ids]
                )
                self.ball.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
                self.ball.write_root_link_velocity_to_sim(
                    root_velocity, env_ids=env_ids
                )
                self._ground_impact_pending[env_ids] = False
                self._ground_impact_age[env_ids] = 0

            expired = self._ground_impact_pending & (self._ground_impact_age > 20)
            self._ground_impact_pending[expired] = False
            self._ground_impact_age[expired] = 0
            return

        env_ids = bounced.nonzero().flatten()
        if len(env_ids) == 0:
            return
        root_velocity = torch.cat(
            (current_linear, self.ball.data.root_link_ang_vel_w), dim=-1
        )[env_ids].clone()
        root_velocity[:, :2] = (
            pre_step_velocity[env_ids, :2]
            * domain_state.ground_tangent_speed_retention[env_ids, None]
        )
        self.ball.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)


def install_tennis_ball_aerodynamics(
    env: ManagerBasedRlEnv,
) -> TennisBallAerodynamicsController:
    """Install ball-only air forces without enabling global MuJoCo fluid drag."""
    existing = getattr(env, "_tennis_ball_aerodynamics", None)
    if existing is not None:
        return existing

    controller = TennisBallAerodynamicsController(env)
    base_sim_step = env.sim.step

    def step_with_tennis_ball_aerodynamics(_base_sim_step=base_sim_step) -> None:
        controller.before_sim_step()
        _base_sim_step()
        controller.after_sim_step()

    env.sim.step = step_with_tennis_ball_aerodynamics  # type: ignore[method-assign]
    env._tennis_ball_aerodynamics = controller  # type: ignore[attr-defined]
    print(
        "[INFO]: Tennis ball aero enabled: "
        f"Cd={'domain-randomized' if controller.domain_state is not None else f'{PHYSICS.ball.drag_coefficient:.3f}'}, "
        "Magnus=0.000, "
        "global_air_density=0 (G1 unaffected)"
    )
    return controller


def install_tennis_ball_controller(
    env: ManagerBasedRlEnv,
    release_lead_s: float,
    *,
    verbose_samples: bool = False,
) -> TennisBallTargetController | TennisIncomingBallController:
    """Install one non-recursive pre-step controller on a raw mjlab env."""
    motion = env.command_manager.get_term("motion")
    if motion.cfg.incoming_ball_launch_enabled:
        controller: TennisBallTargetController | TennisIncomingBallController = (
            TennisIncomingBallController(env, verbose_samples=verbose_samples)
        )
    else:
        controller = TennisBallTargetController(
            env,
            release_lead_s,
            verbose_samples=verbose_samples,
        )
    install_tennis_ball_aerodynamics(env)
    from athlete.goal_cond_tracking.mdp.full_flight import (
        FullFlightMotionCommandCfg, install_full_flight_tracker,
    )
    full_flight = None
    if isinstance(motion.cfg, FullFlightMotionCommandCfg):
        full_flight = install_full_flight_tracker(env, controller, verbose=verbose_samples)
    base_step = env.step

    def step_with_physical_ball(action: torch.Tensor, _base_step=base_step):
        controller.before_step()
        if full_flight is not None:
            full_flight.begin_control_step()
        result = _base_step(action)
        controller.after_step(result[2] | result[3])
        return result

    env.step = step_with_physical_ball  # type: ignore[method-assign]
    return controller
