"""Unitree G1 flat tracking environment configurations."""

import copy
import math
from dataclasses import fields, replace

from mjlab.asset_zoo.robots import (
    G1_ACTION_SCALE,
    get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg

from athlete.goal_cond_tracking import mdp
from athlete.goal_cond_tracking.mdp import (
    MultiTargetMotionCommandCfg,
    PhaseAccelerationActionCfg,
    PhaseAccelerationMultiTargetMotionCommandCfg,
    PhaseAwareMultiTargetMotionCommandCfg,
    PhaseResidualActionCfg,
)
from athlete.goal_cond_tracking.tracking_env_cfg import (
    make_multi_target_tracking_env_cfg,
)

G1_JOINT_VELOCITY_LIMITS = {
    r".*_hip_yaw_joint": 32.0,
    r".*_hip_roll_joint": 20.0,
    r".*_hip_pitch_joint": 32.0,
    r".*_knee_joint": 20.0,
    r".*_ankle_(pitch|roll)_joint": 37.0,
    r"waist_(roll|pitch)_joint": 37.0,
    r"waist_yaw_joint": 32.0,
    r".*_(shoulder_pitch|shoulder_roll|shoulder_yaw|elbow|wrist_roll)_joint": 37.0,
    r".*_wrist_(pitch|yaw)_joint": 22.0,
}

G1_JOINT_ACCELERATION_LIMITS = {r".*": 10000.0}

# Standard-physics baseline trajectory at the 1.64 s control-step strike deadline,
# expressed in the relabelled ep_0000 reference frame-0 pelvis coordinates.
LAUNCH_DISTILL_STRIKE_TARGET_FRAME0 = (
    0.31967799,
    0.94392872,
    0.10662800,
)
LAUNCH_DISTILL_FLIGHT_TIME_S = 1.641330776

REFERENCE_TRACKING_REWARD_NAMES = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
)

STUDENT_REFERENCE_TRACKING_OBSERVATION_NAMES = (
    "reference_motion_state",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "base_lin_vel",
)


def _add_student_reference_tracking(cfg: ManagerBasedRlEnvCfg) -> None:
    """Restore mimic rewards and expose their minimal reference state to Student."""
    student = cfg.observations["student"]
    teacher = cfg.observations["actor"]
    critic = cfg.observations["critic"]
    assert isinstance(student, ObservationGroupCfg)
    assert isinstance(teacher, ObservationGroupCfg)
    assert isinstance(critic, ObservationGroupCfg)
    student.terms["reference_motion_state"] = ObservationTermCfg(
        func=mdp.motion_reference_state,
        params={"command_name": "motion"},
    )
    student.terms["motion_anchor_pos_b"] = copy.deepcopy(
        teacher.terms["motion_anchor_pos_b"]
    )
    student.terms["motion_anchor_ori_b"] = copy.deepcopy(
        teacher.terms["motion_anchor_ori_b"]
    )
    student.terms["base_lin_vel"] = copy.deepcopy(critic.terms["base_lin_vel"])

    if cfg.rewards is None:
        raise ValueError("Reference tracking requires an active reward manager.")
    cfg.rewards.update(
        {
            "motion_global_root_pos": RewardTermCfg(
                func=mdp.motion_global_anchor_position_error_exp,
                weight=0.5,
                params={"command_name": "motion", "std": 0.3},
            ),
            "motion_global_root_ori": RewardTermCfg(
                func=mdp.motion_global_anchor_orientation_error_exp,
                weight=0.5,
                params={"command_name": "motion", "std": 0.4},
            ),
            "motion_body_pos": RewardTermCfg(
                func=mdp.motion_relative_body_position_error_exp,
                weight=1.0,
                params={"command_name": "motion", "std": 0.3},
            ),
            "motion_body_ori": RewardTermCfg(
                func=mdp.motion_relative_body_orientation_error_exp,
                weight=1.0,
                params={"command_name": "motion", "std": 0.4},
            ),
            "motion_body_lin_vel": RewardTermCfg(
                func=mdp.motion_global_body_linear_velocity_error_exp,
                weight=1.0,
                params={"command_name": "motion", "std": 1.0},
            ),
            "motion_body_ang_vel": RewardTermCfg(
                func=mdp.motion_global_body_angular_velocity_error_exp,
                weight=1.0,
                params={"command_name": "motion", "std": 3.14},
            ),
        }
    )


def unitree_g1_multi_target_tracking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 multi-target tracking configuration."""
    cfg = make_multi_target_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_g1_robot_cfg()}

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MultiTargetMotionCommandCfg)
    motion_cmd.anchor_body_name = "pelvis"  # TODO note that this can be changed to torso_link if we want torso imu instead of pelvis
    motion_cmd.body_names = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot[1-7]_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

    if cfg.terminations is not None and "ee_body_pos" in cfg.terminations:
        cfg.terminations["ee_body_pos"].params["body_names"] = (
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        )

    cfg.viewer.body_name = "torso_link"

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.sampling_mode = "start"

    return cfg


def unitree_g1_phase_aware_tracking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the continuous-phase Unitree G1 tennis tracking task."""
    cfg = unitree_g1_multi_target_tracking_env_cfg(play=play)

    base_motion_cmd = cfg.commands["motion"]
    assert isinstance(base_motion_cmd, MultiTargetMotionCommandCfg)
    inherited = {
        field.name: copy.deepcopy(getattr(base_motion_cmd, field.name))
        for field in fields(MultiTargetMotionCommandCfg)
    }
    motion_cmd = PhaseAwareMultiTargetMotionCommandCfg(
        **inherited,
        source_fps=50.0,
        contact_time_step_s=0.1,
        contact_reward_window_s=0.05,
        max_contact_speedup=2.0,
        phase_rate_min=0.0,
        phase_rate_max=4.0,
        phase_residual_scale=1.0,
    )
    motion_cmd.sampling_mode = "start"
    motion_cmd.between_motion_pause_range = (0.0, 1.0)
    cfg.commands["motion"] = motion_cmd

    cfg.actions["phase_rate"] = PhaseResidualActionCfg(
        entity_name="robot", command_name="motion"
    )

    for group_name in ("actor", "critic"):
        group = cfg.observations[group_name]
        assert isinstance(group, ObservationGroupCfg)
        group.terms["phase_state"] = ObservationTermCfg(
            func=mdp.motion_phase_state,
            params={"command_name": "motion"},
        )

    cfg.rewards["phase_timing"] = RewardTermCfg(
        func=mdp.phase_timing_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.1},
    )
    cfg.rewards["phase_residual"] = RewardTermCfg(
        func=mdp.phase_residual_l2,
        weight=-0.01,
        params={"action_name": "phase_rate"},
    )

    return cfg


def unitree_g1_phase_aware_root_pos_obs_tracking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Add reference-to-robot root position error to the phase-aware actor."""
    cfg = unitree_g1_phase_aware_tracking_env_cfg(play=play)
    actor = cfg.observations["actor"]
    assert isinstance(actor, ObservationGroupCfg)
    actor.terms["motion_anchor_pos_b"] = ObservationTermCfg(
        func=mdp.motion_anchor_pos_b,
        params={"command_name": "motion"},
    )
    return cfg


def _unitree_g1_phase_acceleration_root_pos_obs_tracking_env_cfg(
    *, play: bool, deadline_projection: bool
) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_multi_target_tracking_env_cfg(play=play)

    base_motion_cmd = cfg.commands["motion"]
    assert isinstance(base_motion_cmd, MultiTargetMotionCommandCfg)
    inherited = {
        field.name: copy.deepcopy(getattr(base_motion_cmd, field.name))
        for field in fields(MultiTargetMotionCommandCfg)
    }
    motion_cmd = PhaseAccelerationMultiTargetMotionCommandCfg(
        **inherited,
        source_fps=50.0,
        contact_time_step_s=0.1,
        contact_reward_window_s=0.05,
        max_contact_speedup=2.0,
        phase_rate_min=0.0,
        phase_rate_max=4.0,
        phase_residual_scale=1.0,
        joint_velocity_limits=copy.deepcopy(G1_JOINT_VELOCITY_LIMITS),
        joint_acceleration_limits=copy.deepcopy(G1_JOINT_ACCELERATION_LIMITS),
        phase_acceleration_limit=4.0,
        deadline_projection=deadline_projection,
    )
    motion_cmd.sampling_mode = "start"
    motion_cmd.between_motion_pause_range = (0.0, 1.0)
    cfg.commands["motion"] = motion_cmd

    cfg.actions["phase_acceleration"] = PhaseAccelerationActionCfg(
        entity_name="robot", command_name="motion"
    )

    for group_name in ("actor", "critic"):
        group = cfg.observations[group_name]
        assert isinstance(group, ObservationGroupCfg)
        group.terms["phase_state"] = ObservationTermCfg(
            func=mdp.motion_phase_acceleration_state,
            params={"command_name": "motion"},
        )

    actor = cfg.observations["actor"]
    assert isinstance(actor, ObservationGroupCfg)
    actor.terms["motion_anchor_pos_b"] = ObservationTermCfg(
        func=mdp.motion_anchor_pos_b,
        params={"command_name": "motion"},
    )

    cfg.rewards["phase_timing"] = RewardTermCfg(
        func=mdp.phase_timing_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.1},
    )

    return cfg


def unitree_g1_phase_acceleration_root_pos_obs_tracking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Learn a joint-safe phase-acceleration schedule without deadline projection."""
    return _unitree_g1_phase_acceleration_root_pos_obs_tracking_env_cfg(
        play=play, deadline_projection=False
    )


def unitree_g1_phase_acceleration_deadline_root_pos_obs_tracking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Learn a joint-safe phase schedule with deadline-feasibility projection."""
    return _unitree_g1_phase_acceleration_root_pos_obs_tracking_env_cfg(
        play=play, deadline_projection=True
    )


def unitree_g1_phase_acceleration_deadline_tppo_distillation_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create separate Student and Teacher observations for TPPO distillation.

    The Teacher keeps the original ``actor`` observation group so checkpoints
    trained with the deadline-projection task retain their exact input contract.
    The independent ``student`` copy is the only group that should be reduced or
    replaced when defining deployable observations.
    """
    cfg = unitree_g1_phase_acceleration_deadline_root_pos_obs_tracking_env_cfg(
        play=play
    )
    teacher_actor = cfg.observations["actor"]
    assert isinstance(teacher_actor, ObservationGroupCfg)
    cfg.observations["student"] = ObservationGroupCfg(
        terms={
            "task_goal": ObservationTermCfg(
                func=mdp.motion_task_goal,
                params={"command_name": "motion"},
            ),
            "time_remaining": ObservationTermCfg(
                func=mdp.motion_time_remaining,
                params={"command_name": "motion"},
            ),
            "base_ang_vel": copy.deepcopy(teacher_actor.terms["base_ang_vel"]),
            "joint_pos": copy.deepcopy(teacher_actor.terms["joint_pos"]),
            "joint_vel": copy.deepcopy(teacher_actor.terms["joint_vel"]),
            "actions": ObservationTermCfg(
                func=mdp.last_action,
                params={"action_name": "joint_pos"},
            ),
        },
        concatenate_terms=True,
        enable_corruption=teacher_actor.enable_corruption,
    )

    for reward_name in (*REFERENCE_TRACKING_REWARD_NAMES, "phase_timing"):
        cfg.rewards.pop(reward_name, None)
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.action_term_rate_l2,
        weight=-1.0e-1,
        params={"action_name": "joint_pos"},
    )
    return cfg


def unitree_g1_phase_acceleration_deadline_tppo_landing_distillation_env_cfg(
    play: bool = False,
    *,
    maximum_full_reward_net_height: float = 2.0,
    out_speed_target: float = 10.0,
    out_speed_std: float = 10.0,
) -> ManagerBasedRlEnvCfg:
    """Add the fixed physical-court landing task to TPPO distillation."""
    cfg = unitree_g1_phase_acceleration_deadline_tppo_distillation_env_cfg(play=play)
    from athlete.scripts.tennis_scene import (
        DEFAULT_STRIKE_PROXIMITY_THRESHOLD,
        DEFAULT_STRIKE_SPEED_CHANGE_THRESHOLD,
        configure_tennis_court_env,
        configure_tennis_landing_task,
    )

    configure_tennis_court_env(
        cfg,
        mode="physical",
        align_env_origins=True,
    )
    configure_tennis_landing_task(
        cfg,
        target_std_x=1.0,
        target_std_y=1.0,
        target_radius=0.5,
        reward_std=1.0,
        reward_weight=50.0,
        prediction_delay_steps=2,
        ball_direction_std=0.5,
        net_clearance_reward_weight=50.0,
        maximum_full_reward_net_height=maximum_full_reward_net_height,
        excess_net_height_std=0.25,
        out_speed_target=out_speed_target,
        out_speed_std=out_speed_std,
        out_speed_reward_weight=50.0,
        strike_speed_change_threshold=DEFAULT_STRIKE_SPEED_CHANGE_THRESHOLD,
        strike_proximity_threshold=DEFAULT_STRIKE_PROXIMITY_THRESHOLD,
        direction_tolerance_degrees=60.0,
        target_std_curriculum=None,
        curriculum_steps_per_iteration=24,
    )
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    if not play:
        motion_cfg.target_pos_std_scale = 0.0
        cfg.curriculum["sample_target_std"] = CurriculumTermCfg(
            func=mdp.tennis_target_pos_std_curriculum,
            params={
                "command_name": "motion",
                "stage_steps": (0, 12000, 24000, 36000, 48000),
                "stage_scales": (0.0, 0.25, 0.5, 0.75, 1.0),
            },
        )
    return cfg


def unitree_g1_phase_acceleration_deadline_tppo_distilllinear_env_cfg(
    play: bool = False,
    *,
    target_out_speed: float = 7.0,
    out_speed_std: float = 5.0,
) -> ManagerBasedRlEnvCfg:
    """Configure the non-analytic Landing task used by DistillLinear runs."""
    cfg = unitree_g1_phase_acceleration_deadline_tppo_landing_distillation_env_cfg(
        play=play,
        maximum_full_reward_net_height=5.0,
        out_speed_target=target_out_speed,
        out_speed_std=out_speed_std,
    )
    return cfg


def unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_env_cfg(
    play: bool = False,
    *,
    target_out_speed: float = 7.0,
    out_speed_std: float = 5.0,
) -> ManagerBasedRlEnvCfg:
    """Add a per-step survival reward to the non-analytic DistillLinear task."""
    cfg = unitree_g1_phase_acceleration_deadline_tppo_distilllinear_env_cfg(
        play=play,
        target_out_speed=target_out_speed,
        out_speed_std=out_speed_std,
    )
    cfg.rewards["alive_reward"] = RewardTermCfg(func=mdp.is_alive, weight=1.0)
    return cfg


def unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_student_no_global_root_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Remove world root position from only the deployable Student policy."""
    cfg = unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_env_cfg(
        play=play
    )
    student_group = cfg.observations["student"]
    assert isinstance(student_group, ObservationGroupCfg)
    student_group.terms.pop("global_root_pos")
    return cfg


def unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_ball_launch_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Launch the relabelled ep_0000 baseline trajectory at its fixed strike point."""
    cfg = unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_env_cfg(
        play=play
    )
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.target_pos_std_scale = 0.0
    motion_cfg.incoming_ball_launch_enabled = True
    motion_cfg.incoming_ball_initial_position = (11.93, -0.098, 1.2)
    motion_cfg.incoming_ball_initial_velocity = (
        -8.5,
        0.7594238934967334,
        4.171773616594168,
    )
    motion_cfg.incoming_ball_initial_angular_velocity = (0.0, 0.0, 0.0)
    motion_cfg.incoming_ball_flight_time_s = LAUNCH_DISTILL_FLIGHT_TIME_S
    motion_cfg.incoming_ball_position_noise_std = (0.0, 0.0, 0.0)
    motion_cfg.incoming_ball_velocity_noise_std = (0.0, 0.0, 0.0)
    cfg.curriculum.pop("sample_target_std", None)
    return cfg


def unitree_g1_tennis_launch_distill_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Launch an incoming ball while distilling into a ball-history Student."""
    cfg = unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_ball_launch_env_cfg(
        play=play
    )
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.incoming_ball_strike_target_position_frame0 = (
        LAUNCH_DISTILL_STRIKE_TARGET_FRAME0
    )
    motion_cfg.incoming_ball_trajectory_drives_deadline = True
    motion_cfg.incoming_ball_position_noise_std = (0.05, 0.05, 0.0)
    motion_cfg.incoming_ball_velocity_noise_std = (0.0, 0.0, 0.0)
    motion_cfg.incoming_ball_max_strike_deviation = 0.15
    student_group = cfg.observations["student"]
    assert isinstance(student_group, ObservationGroupCfg)
    student_group.terms.pop("task_goal")
    student_group.terms.pop("time_remaining")
    student_group.terms.pop("global_root_pos")
    ball_history_params = {
        "command_name": "motion",
        "ball_entity_name": "tennis_ball",
        "sample_lags": (20, 15, 10, 5, 0),
        "buffer_length": 25,
    }
    student_group.terms["ball_position"] = ObservationTermCfg(
        func=mdp.TennisBallStridedHistory,
        params={**ball_history_params, "quantity": "position"},
        noise=GaussianNoiseCfg(mean=0.0, std=0.05),
    )
    student_group.terms["ball_linear_velocity"] = ObservationTermCfg(
        func=mdp.TennisBallStridedHistory,
        params={**ball_history_params, "quantity": "linear_velocity"},
        noise=GaussianNoiseCfg(mean=0.0, std=0.2),
    )
    student_group.terms["sweet_spot_position"] = ObservationTermCfg(
        func=mdp.tennis_sweet_spot_position_b,
        params={"command_name": "motion", "source_index": 0},
        noise=GaussianNoiseCfg(mean=0.0, std=0.05),
    )
    student_group.terms["sweet_spot_linear_velocity"].noise = GaussianNoiseCfg(
        mean=0.0, std=0.2
    )

    critic_group = cfg.observations["critic"]
    assert isinstance(critic_group, ObservationGroupCfg)
    critic_group.terms["sweet_spot_position"] = ObservationTermCfg(
        func=mdp.tennis_sweet_spot_position_b,
        params={"command_name": "motion", "source_index": 0},
    )
    if cfg.rewards is None:
        raise ValueError("Launch Distill requires an active reward manager.")
    cfg.rewards["racket_to_live_ball_reward"] = RewardTermCfg(
        func=mdp.tennis_racket_ball_distance_reward,
        weight=10.0,
        params={
            "command_name": "motion",
            "distance_std": 0.20,
            "time_std_s": 0.05,
            "window_half_width_s": 0.05,
            "ball_entity_name": "tennis_ball",
            "source_index": 0,
        },
    )
    if not play:
        from athlete.scripts.tennis_scene import (
            configure_tennis_physics_domain_randomization,
        )

        configure_tennis_physics_domain_randomization(cfg)
    return cfg


def unitree_g1_tennis_launch_distill_torch_match_200_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Match slower incoming trajectories against 200 motion targets in Torch."""
    cfg = unitree_g1_tennis_launch_distill_env_cfg(play=play)
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.landing_target_mean = (6.5, 0.0, 0.0)
    motion_cfg.landing_target_std = (0.5, 0.5, 0.0)
    motion_cfg.incoming_ball_strike_target_position_frame0 = None
    motion_cfg.incoming_ball_position_noise_std = (0.0, 0.0, 0.0)
    motion_cfg.incoming_ball_velocity_noise_std = (0.0, 0.0, 0.0)
    motion_cfg.incoming_ball_max_strike_deviation = 0.0
    motion_cfg.incoming_ball_torch_match_enabled = True
    motion_cfg.incoming_ball_torch_match_expected_motions = 200
    motion_cfg.incoming_ball_torch_match_dt = 0.01
    motion_cfg.incoming_ball_torch_match_horizon_s = 4.0
    motion_cfg.incoming_ball_torch_match_max_distance = 0.15
    motion_cfg.incoming_ball_torch_match_attempts = 8
    motion_cfg.incoming_ball_torch_match_retry_rounds = 3
    motion_cfg.incoming_ball_torch_match_motion_chunk_size = 4
    motion_cfg.incoming_ball_torch_match_cache_size = 4
    motion_cfg.incoming_ball_torch_match_deadline_offset_s = 0.01
    motion_cfg.incoming_ball_torch_match_position_min = (8.0, -2.0, 0.5)
    motion_cfg.incoming_ball_torch_match_position_max = (10.0, 2.0, 1.3)
    motion_cfg.incoming_ball_torch_match_velocity_min = (-5.25, -1.25, 3.0)
    motion_cfg.incoming_ball_torch_match_velocity_max = (-3.5, 1.25, 5.0)
    motion_cfg.auto_chain_motion = True

    if cfg.rewards is None:
        raise ValueError("TorchMatch-200 requires an active reward manager.")
    landing_reward = cfg.rewards["ball_landing_reward"]
    landing_reward.params["target_out_speed"] = 3.5
    landing_reward.params["out_speed_std"] = 2.5
    print(
        "[INFO]: TorchMatch-200 slow-ball task: "
        "landing_mean=(6.500, 0.000)m, landing_std=(0.500, 0.500)m, "
        "target_out_speed=3.500m/s, out_speed_std=2.500m/s, "
        "launch_x=8.000..10.000m, incoming_vx=-5.250..-3.500m/s, "
        "incoming_vz=3.000..5.000m/s"
    )

    from athlete.scripts.tennis_physics import (
        STANDARD_TENNIS_DOMAIN_RANDOMIZATION,
        STANDARD_TENNIS_PHYSICS,
    )
    from athlete.scripts.tennis_scene import (
        configure_tennis_physics_domain_randomization,
    )

    physics_randomization = replace(
        STANDARD_TENNIS_DOMAIN_RANDOMIZATION,
        explicit_ground_rebound=True,
    )
    if play:
        nominal_tangent_retention = (
            sum(STANDARD_TENNIS_DOMAIN_RANDOMIZATION.ground_tangent_speed_retention)
            / 2.0
        )
        nominal_racket_restitution = (
            sum(STANDARD_TENNIS_DOMAIN_RANDOMIZATION.racket_restitution) / 2.0
        )
        physics_randomization = replace(
            physics_randomization,
            ball_mass_kg=(
                STANDARD_TENNIS_PHYSICS.ball.mass_kg,
                STANDARD_TENNIS_PHYSICS.ball.mass_kg,
            ),
            court_restitution=(
                STANDARD_TENNIS_PHYSICS.court.restitution,
                STANDARD_TENNIS_PHYSICS.court.restitution,
            ),
            ground_tangent_speed_retention=(
                nominal_tangent_retention,
                nominal_tangent_retention,
            ),
            drag_coefficient=(
                STANDARD_TENNIS_PHYSICS.ball.drag_coefficient,
                STANDARD_TENNIS_PHYSICS.ball.drag_coefficient,
            ),
            racket_mass_kg=(0.3, 0.3),
            racket_restitution=(
                nominal_racket_restitution,
                nominal_racket_restitution,
            ),
            racket_com_offset_x_m=(0.0, 0.0),
            racket_com_offset_y_m=(0.0, 0.0),
            racket_com_offset_z_m=(0.0, 0.0),
        )
    configure_tennis_physics_domain_randomization(cfg, physics_randomization)

    if cfg.terminations is not None:
        cfg.terminations.pop("motion_complete", None)
    return cfg


def unitree_g1_tennis_launch_distill_warp_match_200_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Use one fused Warp kernel for trajectory rollout before Torch matching."""
    cfg = unitree_g1_tennis_launch_distill_torch_match_200_env_cfg(play=play)
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.incoming_ball_trajectory_rollout_backend = "warp_fused"
    return cfg


def unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Generate fresh fixed-court, root-directed Warp trajectories per strike."""
    cfg = unitree_g1_tennis_launch_distill_warp_match_200_env_cfg(play=play)
    cfg.sim.mujoco.timestep = 0.0025
    cfg.decimation = 8
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.incoming_ball_torch_match_cache_size = 0
    motion_cfg.incoming_ball_torch_match_max_distance = 0.20
    motion_cfg.incoming_ball_torch_match_adaptive_attempt_batch_size = 2
    motion_cfg.incoming_ball_torch_match_motion_chunk_size = 200
    motion_cfg.incoming_ball_torch_match_hierarchical_top_k = 32
    motion_cfg.incoming_ball_torch_match_hierarchical_coarse_neighbor_radius = 1
    motion_cfg.incoming_ball_torch_match_chain_batch_interval_steps = 8
    motion_cfg.incoming_ball_torch_match_root_directed_sampling = True
    motion_cfg.incoming_ball_torch_match_horizontal_speed_range_m_s = (3.5, 5.25)
    motion_cfg.incoming_ball_torch_match_horizontal_angle_half_width_deg = 15.0
    motion_cfg.incoming_ball_torch_match_net_crossing_height_range_m = (1.5, 3.5)
    motion_cfg.incoming_ball_torch_match_maximum_initial_speed_m_s = 7.0
    motion_cfg.landing_target_frame = "startup"
    motion_cfg.landing_target_mean = (7.0, 0.0, 0.0)
    motion_cfg.landing_target_std = (0.0, 0.0, 0.0)
    student_group = cfg.observations["student"]
    assert isinstance(student_group, ObservationGroupCfg)
    student_group.terms.pop("global_root_pos", None)
    student_group.terms["startup_heading"] = ObservationTermCfg(
        func=mdp.tennis_startup_heading_b,
        params={"command_name": "motion"},
        noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    )
    student_group.terms["ball_linear_velocity"].noise = GaussianNoiseCfg(
        mean=0.0,
        std=0.1,
    )
    student_group.terms["projected_gravity"].noise = UniformNoiseCfg(
        n_min=-0.05,
        n_max=0.05,
    )
    critic_group = cfg.observations["critic"]
    assert isinstance(critic_group, ObservationGroupCfg)
    critic_group.terms["startup_heading"] = ObservationTermCfg(
        func=mdp.tennis_startup_heading_b,
        params={"command_name": "motion"},
    )
    if cfg.rewards is None:
        raise ValueError(
            "WarpRootDirected-Cache0-200 requires an active reward manager."
        )
    cfg.rewards["racket_to_live_ball_reward"].weight = 50.0
    cfg.rewards["post_strike_heading_reward"] = RewardTermCfg(
        func=mdp.tennis_post_strike_heading_to_startup_x_reward,
        weight=2.0,
        params={
            "command_name": "motion",
            "recovery_delay_s": 0.3,
            "tolerance_degrees": 15.0,
            "std_degrees": 30.0,
        },
    )
    cfg.rewards.pop("target_orientation_reward", None)
    cfg.rewards.pop("target_velocity_reward", None)
    _add_student_reference_tracking(cfg)
    return cfg


def unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Ablate Student reference observations and reference tracking rewards."""
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_env_cfg(
        play=play
    )
    student_group = cfg.observations["student"]
    assert isinstance(student_group, ObservationGroupCfg)
    for observation_name in STUDENT_REFERENCE_TRACKING_OBSERVATION_NAMES:
        student_group.terms.pop(observation_name, None)

    if cfg.rewards is None:
        raise ValueError(
            "WarpRootDirected NoTracking requires an active reward manager."
        )
    for reward_name in REFERENCE_TRACKING_REWARD_NAMES:
        cfg.rewards.pop(reward_name, None)
    return cfg


def unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Use the deployable NoTracking observations without racket proximity reward."""
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_env_cfg(
        play=play
    )
    if cfg.rewards is None:
        raise ValueError("WarpRootDirected NoTracking Racket0 requires rewards.")
    cfg.rewards["racket_to_live_ball_reward"].weight = 0.0
    return cfg


def unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Add signed incoming-ball racket orientation to the Floor0.3 task."""
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_env_cfg(
        play=play
    )
    if cfg.rewards is None:
        raise ValueError(
            "WarpRootDirected NoTracking Racket0 Floor0.3 requires rewards."
        )
    cfg.rewards["racket_incoming_velocity_orientation_reward"] = RewardTermCfg(
        func=mdp.tennis_racket_incoming_velocity_orientation_reward,
        weight=50.0,
        params={
            "command_name": "motion",
            "std": 0.5,
            "tolerance_degrees": 30.0,
            "minimum_ball_speed": 0.5,
            "ball_entity_name": "tennis_ball",
        },
    )
    cfg.rewards["ball_landing_reward"].params["align_out_speed_to_landing_target"] = (
        True
    )
    cfg.rewards["action_rate_l2"].weight = -0.04
    for reward_name in (
        "ball_landing_reward",
        "net_clearance_reward",
        "ball_out_speed_reward",
    ):
        cfg.rewards[reward_name].weight = 100.0
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.racket_orientation_reward_window_before_s = 0.1
    motion_cfg.racket_orientation_reward_window_after_s = 0.1
    motion_cfg.viz.show_incoming_ball_racket_orientation_target = True
    motion_cfg.viz.incoming_ball_entity_name = "tennis_ball"
    motion_cfg.viz.incoming_ball_minimum_speed = 0.5
    motion_cfg.viz.racket_orientation_arrow_length = 0.7
    return cfg


def unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Use root tilt only, rather than reference-motion termination checks."""
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg(
        play=play
    )
    if cfg.terminations is None:
        cfg.terminations = {}
    for termination_name in ("anchor_pos", "anchor_ori", "ee_body_pos"):
        cfg.terminations.pop(termination_name, None)
    cfg.terminations["root_tilt"] = TerminationTermCfg(
        func=mdp.bad_root_tilt,
        params={"command_name": "motion", "max_tilt_degrees": 45.0},
    )
    return cfg


def unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Add isolated temporal intent inputs to the RootTilt Launch Distill task.

    ``student`` remains the existing 133-D deployable current observation. The
    three extra groups are consumed only by ``IntentTransformerActor`` and do
    not alter any pre-existing task's observation contract.
    """
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_env_cfg(
        play=play
    )
    state_history_params = {
        "command_name": "motion",
        "action_name": "joint_pos",
        "sensor_name": "robot/imu_ang_vel",
        "sample_lags": (20, 15, 10, 5, 0),
        "buffer_length": 25,
        # Same deployed sensor corruption ranges as the current Student terms.
        # The term applies it on readout, so the history buffer stays physical.
        "apply_observation_noise": not play,
        "base_ang_vel_noise": (-0.2, 0.2),
        "projected_gravity_noise": (-0.05, 0.05),
        "joint_pos_noise": (-0.01, 0.01),
        "joint_vel_noise": (-0.5, 0.5),
    }
    ball_history_params = {
        "command_name": "motion",
        "ball_entity_name": "tennis_ball",
        "sample_lags": (20, 15, 10, 5, 0),
        "buffer_length": 25,
    }
    cfg.observations["intent_state_history"] = ObservationGroupCfg(
        terms={
            "state": ObservationTermCfg(
                func=mdp.TennisStudentStateStridedHistory,
                params=state_history_params,
            ),
        },
        concatenate_terms=True,
        enable_corruption=False,
    )
    cfg.observations["intent_ball_history"] = ObservationGroupCfg(
        terms={
            "position": ObservationTermCfg(
                func=mdp.TennisBallCurrentAnchorStridedHistory,
                params={**ball_history_params, "quantity": "position"},
                noise=GaussianNoiseCfg(mean=0.0, std=0.05),
            ),
            "linear_velocity": ObservationTermCfg(
                func=mdp.TennisBallCurrentAnchorStridedHistory,
                params={**ball_history_params, "quantity": "linear_velocity"},
                noise=GaussianNoiseCfg(mean=0.0, std=0.1),
            ),
        },
        concatenate_terms=True,
        enable_corruption=True,
    )
    cfg.observations["intent_reference_future"] = ObservationGroupCfg(
        terms={
            "joint_state": ObservationTermCfg(
                func=mdp.motion_future_reference_window,
                params={
                    "command_name": "motion",
                    "future_offsets": (1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56),
                },
            ),
        },
        concatenate_terms=True,
        enable_corruption=False,
    )
    return cfg


def unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_racket_ground10_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Penalize physical racket-head contacts with the robot terrain plane."""
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(
        play=play
    )
    racket_ground_sensor = ContactSensorCfg(
        name="racket_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern="racket_ball_collision",
            entity="robot",
        ),
        secondary=ContactMatch(mode="geom", pattern="terrain"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        secondary_policy="error",
        history_length=cfg.decimation,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, racket_ground_sensor)
    if cfg.rewards is None:
        raise ValueError("Intent racket-ground task requires active rewards.")
    cfg.rewards["racket_ground_collision"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-10.0,
        params={
            "sensor_name": "racket_ground_contact",
            "force_threshold": 10.0,
        },
    )
    return cfg


def unitree_g1_tennis_small_court_intent_transformer_sweet_speed50_racket_ground10_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Fine-tune the m11-3 Intent Student setup in the 10 m x 5.5 m area."""
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_env_cfg(
        play=play
    )
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.tennis_court_net_x_m = 3.5
    motion_cfg.tennis_court_net_half_width_m = 2.5
    motion_cfg.incoming_ball_torch_match_position_min = (5.5, -1.5, 0.6)
    motion_cfg.incoming_ball_torch_match_position_max = (6.5, 1.5, 1.2)
    motion_cfg.incoming_ball_torch_match_horizontal_speed_range_m_s = (2.8, 4.2)
    motion_cfg.incoming_ball_torch_match_horizontal_angle_half_width_deg = 15.0
    motion_cfg.landing_target_mean = (6.0, 0.0, 0.0)
    motion_cfg.landing_target_std = (0.0, 0.0, 0.0)
    motion_cfg.analytic_net_x = 3.5
    motion_cfg.analytic_net_half_width = 2.5

    if cfg.rewards is None:
        raise ValueError("Small-court Intent task requires active rewards.")
    landing_reward = cfg.rewards["ball_landing_reward"]
    landing_reward.params["net_x"] = 3.5
    landing_reward.params["net_half_width"] = 2.5

    cfg.viewer.lookat = (3.9, 0.0, 0.55)
    cfg.viewer.distance = 14.0
    print(
        "[INFO]: Small-court Intent task: net_x=3.500m, net_width=5.000m, "
        "launch=(5.500..6.500, -1.500..1.500, 0.600..1.200)m, "
        "horizontal_speed=2.800..4.200m/s, landing_mean=(6.000, 0.000)m, "
        "landing_std=(0.000, 0.000)m"
    )
    return cfg


def unitree_g1_tennis_small_court_intent_transformer_frozen_intent_epoch3_action_rate020_sweet_speed_scale03_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Run the frozen-Intent small-court ablation with smoother, slower swings."""
    cfg = unitree_g1_tennis_small_court_intent_transformer_sweet_speed50_racket_ground10_env_cfg(
        play=play
    )
    if cfg.rewards is None:
        raise ValueError("Frozen-Intent epoch-3 small-court task requires rewards.")
    cfg.rewards["action_rate_l2"].weight = -0.2
    cfg.rewards["reference_sweet_spot_velocity_reward"].params["target_speed_scale"] = (
        0.3
    )
    print(
        "[INFO]: Frozen-Intent epoch-3 small-court ablation: "
        "action_rate_l2=-0.200, sweet_spot_target_speed_scale=0.300"
    )
    return cfg


def unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Keep the m11-9 settings with the M9 distillation landing goal at 5 m."""
    cfg = unitree_g1_tennis_small_court_intent_transformer_frozen_intent_epoch3_action_rate020_sweet_speed_scale03_env_cfg(
        play=play
    )
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.landing_target_mean = (5.0, 0.0, 0.0)
    return cfg


def unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_pelvis_torso_tilt50_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Landing5 variant that waits for both pelvis and torso to be tilted."""
    cfg = unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_env_cfg(
        play=play
    )
    if cfg.terminations is None:
        raise ValueError("Pelvis/torso tilt task requires active terminations.")
    cfg.terminations.pop("root_tilt", None)
    cfg.terminations["pelvis_torso_tilt"] = TerminationTermCfg(
        func=mdp.bad_pelvis_and_torso_tilt,
        params={
            "command_name": "motion",
            "max_tilt_degrees": 50.0,
            "pelvis_body_name": "pelvis",
            "torso_body_name": "torso_link",
        },
    )
    return cfg


def unitree_g1_tennis_small_court_landing5_pelvis_torso_tilt50_ball_dr150_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Expand only the train-time tennis-ball physics ranges by 1.5x."""
    cfg = unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_pelvis_torso_tilt50_env_cfg(
        play=play
    )
    if play:
        return cfg
    return _apply_tennis_ball_dr150(cfg)


def _apply_tennis_ball_dr150(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Apply the m14-6 ball ranges without scaling twice or changing racket DR."""

    from athlete.scripts.tennis_physics import (
        STANDARD_TENNIS_DOMAIN_RANDOMIZATION,
        STANDARD_TENNIS_PHYSICS,
    )

    event = cfg.events["tennis_physics_domain_randomization"]
    physics_randomization = event.params["cfg"]
    scale = 1.5

    def expand(
        bounds: tuple[float, float], nominal: float
    ) -> tuple[float, float]:
        return (
            nominal + scale * (bounds[0] - nominal),
            nominal + scale * (bounds[1] - nominal),
        )

    tangent_nominal = sum(
        STANDARD_TENNIS_DOMAIN_RANDOMIZATION.ground_tangent_speed_retention
    ) / 2.0
    event.params["cfg"] = replace(
        physics_randomization,
        ball_mass_kg=expand(
            STANDARD_TENNIS_DOMAIN_RANDOMIZATION.ball_mass_kg,
            STANDARD_TENNIS_PHYSICS.ball.mass_kg,
        ),
        court_restitution=expand(
            STANDARD_TENNIS_DOMAIN_RANDOMIZATION.court_restitution,
            STANDARD_TENNIS_PHYSICS.court.restitution,
        ),
        ground_tangent_speed_retention=expand(
            STANDARD_TENNIS_DOMAIN_RANDOMIZATION.ground_tangent_speed_retention,
            tangent_nominal,
        ),
        drag_coefficient=expand(
            STANDARD_TENNIS_DOMAIN_RANDOMIZATION.drag_coefficient,
            STANDARD_TENNIS_PHYSICS.ball.drag_coefficient,
        ),
    )
    print(
        "[INFO]: Tennis-ball physics DR width=1.5x: "
        "mass=0.05515..0.06025kg, court_e=0.6775..0.8275, "
        "tangent_retention=0.7125..0.9375, Cd=0.4750..0.7000; "
        "racket DR unchanged"
    )
    return cfg


def unitree_g1_tennis_small_court_landing_std_curriculum150_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Ramp startup-frame landing XY std to 1.5 m, always beyond the court net."""
    cfg = unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_pelvis_torso_tilt50_env_cfg(
        play=play
    )
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.landing_target_net_margin_m = 0.25
    if play:
        motion_cfg.landing_target_std = (1.5, 1.5, 0.0)
    else:
        motion_cfg.landing_target_std_final = (1.5, 1.5, 0.0)
        motion_cfg.landing_target_std_ramp_steps = 30000 * 24
        cfg.curriculum["landing_target_std"] = CurriculumTermCfg(
            func=mdp.tennis_landing_target_linear_std_curriculum,
            params={"command_name": "motion"},
        )
    return cfg


def unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Add torso mass/inertia DR and m14-6 ball physics DR to the m14-5 task."""
    cfg = unitree_g1_tennis_small_court_landing_std_curriculum150_env_cfg(
        play=play
    )
    if play:
        return cfg

    _apply_tennis_ball_dr150(cfg)
    physics_event = cfg.events["tennis_physics_domain_randomization"]
    # Expand the original 0.54..0.68 range about 0.61, without altering m14-6.
    physics_event.params["cfg"] = replace(
        physics_event.params["cfg"], racket_restitution=(0.505, 0.715)
    )
    print("[INFO]: Racket-ball restitution DR width=1.5x: 0.505..0.715")

    # pseudo_inertia scales mass and inertia by exp(2 * alpha) while keeping
    # the COM unchanged. Existing torso COM randomization remains independent.
    torso_mass_inertia = EventTermCfg(
        mode="startup",
        func=dr.pseudo_inertia,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=("torso_link",),
            ),
            "alpha_range": (
                0.5 * math.log(0.9),
                0.5 * math.log(1.1),
            ),
        },
    )
    # pseudo_inertia writes body_ipos as part of its consistent inertial update.
    # Run it before base_com so the existing COM offset remains effective.
    ordered_events = {}
    inserted = False
    for name, event in cfg.events.items():
        if name == "base_com":
            ordered_events["torso_mass_inertia"] = torso_mass_inertia
            inserted = True
        ordered_events[name] = event
    if not inserted:
        ordered_events["torso_mass_inertia"] = torso_mass_inertia
    cfg.events = ordered_events
    print(
        "[INFO]: Torso physical DR: mass/inertia scale=0.900..1.100, "
        "nominal_mass=7.818kg, sampled_mass=7.0362..8.5998kg; COM DR unchanged"
    )
    return cfg


def unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_global_root_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Actor global root XYZ with 5 cm noise; Critic retains clean root XYZ."""
    cfg = unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_env_cfg(play=play)
    student = cfg.observations["student"]
    assert isinstance(student, ObservationGroupCfg)
    if "global_root_pos" in student.terms:
        raise ValueError("Global-root ablation requires a no-global baseline.")
    student.terms["global_root_pos"] = ObservationTermCfg(
        func=mdp.robot_global_root_pos_w,
        params={"command_name": "motion"},
        noise=None if play else GaussianNoiseCfg(mean=0.0, std=0.05),
    )
    critic = cfg.observations["critic"]
    assert isinstance(critic, ObservationGroupCfg)
    critic.terms["global_root_pos"] = ObservationTermCfg(
        func=mdp.robot_global_root_pos_w,
        params={"command_name": "motion"},
    )
    return cfg


def unitree_g1_tennis_small_court_global_root_sweet_fk_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """m14-8 ablation: joint FK position and deployment-style velocity estimate."""
    from athlete.goal_cond_tracking.mdp.racket_fk import TennisSweetSpotFK

    cfg = unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_global_root_env_cfg(play)
    for group_name in ("student", "critic"):
        group = cfg.observations[group_name]
        assert isinstance(group, ObservationGroupCfg)
        for name, quantity in (("sweet_spot_position", "position"),
                               ("sweet_spot_linear_velocity", "linear_velocity")):
            term = group.terms[name]
            term.func = TennisSweetSpotFK
            term.params = {
                "command_name": "motion", "quantity": quantity,
                "site_name": "racket_sweet_spot", "smoothing": 0.35,
            }
    return cfg


def unitree_g1_tennis_small_court_no_global_root_sweet_fk_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """m14-10: no Student XYZ, delayed ball and instantaneous relative swing speed."""
    cfg = unitree_g1_tennis_small_court_global_root_sweet_fk_env_cfg(play)
    student = cfg.observations["student"]
    assert isinstance(student, ObservationGroupCfg)
    del student.terms["global_root_pos"]
    for group_name, names in (
        ("student", ("ball_position", "ball_linear_velocity")),
        ("intent_ball_history", ("position", "linear_velocity")),
    ):
        group = cfg.observations[group_name]
        assert isinstance(group, ObservationGroupCfg)
        for name in names:
            group.terms[name].params.update(
                delay_mean_s=0.020, delay_std_s=0.010, delay_max_s=0.050
            )
    for group_name in ("student", "critic"):
        group = cfg.observations[group_name]
        assert isinstance(group, ObservationGroupCfg)
        term = group.terms["sweet_spot_linear_velocity"]
        term.func = mdp.tennis_sweet_spot_relative_linear_velocity_b
        term.params = {"command_name": "motion", "source_index": 0}
    cfg.rewards["reference_sweet_spot_velocity_reward"].params["relative_to_pelvis"] = True
    return cfg


def unitree_g1_tennis_small_court_relative_sweet_ball_dr225_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """m14-12: expand four current ball DR widths by 1.5 about their midpoints."""
    cfg = unitree_g1_tennis_small_court_no_global_root_sweet_fk_env_cfg(play)
    if not play:
        event = cfg.events["tennis_physics_domain_randomization"]
        physics = event.params["cfg"]
        ranges = {}
        for name in ("ball_mass_kg", "court_restitution",
                     "ground_tangent_speed_retention", "drag_coefficient"):
            low, high = getattr(physics, name)
            center, half_width = (low + high) / 2, (high - low) * 0.75
            ranges[name] = (center - half_width, center + half_width)
        event.params["cfg"] = replace(physics, **ranges)
    return cfg


def unitree_g1_tennis_small_court_relative_sweet_foot_force_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """m14-13: independent finite-duration force disturbances on both feet."""
    cfg = unitree_g1_tennis_small_court_no_global_root_sweet_fk_env_cfg(play)
    if not play:
        cfg.events["foot_force"] = EventTermCfg(
            func=mdp.apply_body_impulse,
            mode="step",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
                ),
                "force_range": (-20.0, 20.0),
                "torque_range": (0.0, 0.0),
                "duration_s": (0.1, 0.2),
                "cooldown_s": (2.0, 4.0),
            },
        )
    return cfg


def unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """m14-11: m14-10 plus Student XYZ and live-root landing-target observations."""
    cfg = unitree_g1_tennis_small_court_no_global_root_sweet_fk_env_cfg(play)
    student = cfg.observations["student"]
    assert isinstance(student, ObservationGroupCfg)
    student.terms["global_root_pos"] = ObservationTermCfg(
        func=mdp.robot_global_root_pos_w,
        params={"command_name": "motion"},
        noise=None if play else GaussianNoiseCfg(mean=0.0, std=0.05),
    )
    for name in ("student", "critic"):
        group = cfg.observations[name]
        assert isinstance(group, ObservationGroupCfg)
        group.terms["landing_target"].params["current_root_frame"] = True
    return cfg


def unitree_g1_tennis_small_court_global_root_actuator_robust_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """m14-11 derivative: delayed perception, randomized PD gains and commands."""
    cfg = unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg(play)
    for group_name, names in (
        ("student", ("ball_position", "ball_linear_velocity")),
        ("intent_ball_history", ("position", "linear_velocity")),
    ):
        for name in names:
            cfg.observations[group_name].terms[name].params.update(
                delay_mean_s=0.030, delay_std_s=0.015, delay_max_s=0.060
            )
    if not play:
        if not math.isclose(cfg.sim.mujoco.timestep, 0.0025):
            raise ValueError("Actuator lag 2..6 requires a 2.5ms physics timestep")
        # The robot factory shares its articulation as well as actuator configs.
        cfg.scene.entities["robot"] = copy.deepcopy(cfg.scene.entities["robot"])
        for actuator in cfg.scene.entities["robot"].articulation.actuators:
            actuator.delay_min_lag = 2
            actuator.delay_max_lag = 6
            actuator.delay_update_period = 400  # One second, not policy steps.
            actuator.delay_hold_prob = 0.0
            # Sample on the first compute after reset, rather than start at lag 0.
            actuator.delay_per_env_phase = False
        cfg.events["actuator_pd_gains"] = EventTermCfg(
            func=dr.pd_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
                "kp_range": (0.8, 1.2),
                "kd_range": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
    return cfg


def unitree_g1_tennis_small_court_global_root_actuator_robust_half_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """m14-14 control: halve only the new random widths, preserving their means."""
    cfg = unitree_g1_tennis_small_court_global_root_actuator_robust_env_cfg(play)
    for group_name, names in (
        ("student", ("ball_position", "ball_linear_velocity")),
        ("intent_ball_history", ("position", "linear_velocity")),
    ):
        for name in names:
            cfg.observations[group_name].terms[name].params.update(
                delay_std_s=0.0075, delay_min_s=0.015, delay_max_s=0.045
            )
    if not play:
        for actuator in cfg.scene.entities["robot"].articulation.actuators:
            actuator.delay_min_lag = 3
            actuator.delay_max_lag = 5
        cfg.events["actuator_pd_gains"].params.update(
            kp_range=(0.9, 1.1), kd_range=(0.9, 1.1)
        )
    return cfg


def unitree_g1_tennis_small_court_intent_transformer_ball_observation_robust_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Train the small-court Student with delayed and intermittently lost ball data."""
    cfg = unitree_g1_tennis_small_court_intent_transformer_sweet_speed50_racket_ground10_env_cfg(
        play=play
    )
    perception_params = {
        "use_shared_perception": True,
        "perception_latency_min_steps": 0,
        "perception_latency_max_steps": 0 if play else 3,
        "perception_dropout_start_probability": 0.0 if play else 0.03,
        "perception_dropout_duration_min_steps": 2,
        "perception_dropout_duration_max_steps": 5,
        "perception_position_noise_std": 0.0 if play else 0.05,
        "perception_velocity_noise_std": 0.0 if play else 0.05,
        "perception_control_dt_s": 0.02,
    }

    student_group = cfg.observations["student"]
    intent_ball_group = cfg.observations["intent_ball_history"]
    assert isinstance(student_group, ObservationGroupCfg)
    assert isinstance(intent_ball_group, ObservationGroupCfg)
    for group in (student_group, intent_ball_group):
        for term_name in ("ball_position", "position"):
            if term_name in group.terms:
                group.terms[term_name].params.update(perception_params)
                group.terms[term_name].noise = None
        for term_name in ("ball_linear_velocity", "linear_velocity"):
            if term_name in group.terms:
                group.terms[term_name].params.update(perception_params)
                group.terms[term_name].noise = None

    status_params = {
        "ball_entity_name": "tennis_ball",
        "sample_lags": (20, 15, 10, 5, 0),
        "buffer_length": 25,
        **perception_params,
    }
    intent_ball_group.terms["valid"] = ObservationTermCfg(
        func=mdp.TennisBallObservationStatusHistory,
        params={**status_params, "quantity": "valid"},
    )
    intent_ball_group.terms["age"] = ObservationTermCfg(
        func=mdp.TennisBallObservationStatusHistory,
        params={**status_params, "quantity": "age"},
    )
    intent_ball_group.enable_corruption = False
    print(
        "[INFO]: Robust ball perception: token=position3+velocity3+valid1+age1, "
        f"latency_steps=0..{perception_params['perception_latency_max_steps']}, "
        "dropout_start_probability="
        f"{perception_params['perception_dropout_start_probability']:.3f}, "
        "dropout_duration_steps=2..5, "
        f"position_noise={perception_params['perception_position_noise_std']:.2f}m, "
        f"velocity_noise={perception_params['perception_velocity_noise_std']:.2f}m/s"
    )
    return cfg


def unitree_g1_tennis_small_court_intent_transformer_ball_observation_robust_smooth_slow_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Make the robust small-court task smoother and easier to land."""
    cfg = unitree_g1_tennis_small_court_intent_transformer_ball_observation_robust_env_cfg(
        play=play
    )
    if cfg.rewards is None:
        raise ValueError("Robust smooth/slow small-court task requires active rewards.")
    cfg.rewards["action_rate_l2"].weight = -0.08
    cfg.rewards["ball_landing_reward"].params["target_out_speed"] = 2.5

    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.landing_target_mean = (5.0, 0.0, 0.0)
    motion_cfg.landing_target_std = (0.0, 0.0, 0.0)
    print(
        "[INFO]: Robust smooth/slow small-court task: "
        "action_rate_l2=-0.080, target_out_speed=2.500m/s, "
        "landing_mean=(5.000, 0.000)m"
    )
    return cfg


def unitree_g1_tennis_small_court_intent_transformer_failure_trajectory05_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Use 5% no-net or over-net-but-far launches without extra observations."""
    cfg = unitree_g1_tennis_small_court_intent_transformer_sweet_speed50_racket_ground10_env_cfg(
        play=play
    )
    if cfg.rewards is None:
        raise ValueError("Failure-trajectory small-court task requires active rewards.")
    cfg.rewards["action_rate_l2"].weight = -0.08
    cfg.rewards["ball_landing_reward"].params["target_out_speed"] = 2.5

    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.landing_target_mean = (5.0, 0.0, 0.0)
    motion_cfg.landing_target_std = (0.0, 0.0, 0.0)
    motion_cfg.incoming_ball_failure_trajectory_probability = 0.05
    motion_cfg.incoming_ball_failure_trajectory_xy_distance_threshold_m = 3.0
    motion_cfg.incoming_ball_failure_no_net_fraction = 0.5

    intent_ball_group = cfg.observations["intent_ball_history"]
    assert isinstance(intent_ball_group, ObservationGroupCfg)
    if "valid" in intent_ball_group.terms or "age" in intent_ball_group.terms:
        raise ValueError("Failure-trajectory task must keep 6-D ball-history tokens.")
    print(
        "[INFO]: Small-court failure-trajectory task: probability=0.050, "
        "types=no-net|trajectory-min-xy-distance>3.000m, "
        "ball_token=position3+velocity3, action_rate_l2=-0.080, "
        "target_out_speed=2.500m/s, landing_mean=(5.000, 0.000)m"
    )
    return cfg


def unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed20_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Add reference-directed sweet-point swing speed to Intent Floor0.05."""
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(
        play=play
    )
    if cfg.rewards is None:
        raise ValueError("Intent sweet-point speed task requires active rewards.")
    cfg.rewards["reference_sweet_spot_velocity_reward"] = RewardTermCfg(
        func=mdp.tennis_reference_sweet_spot_velocity_reward,
        weight=20.0,
        params={
            "command_name": "motion",
            "target_speed_scale": 0.75,
            "contact_window_s": 0.1,
            "proximity_std": 0.2,
            "minimum_reference_speed": 0.5,
            "minimum_ball_speed": 0.5,
            "ball_entity_name": "tennis_ball",
        },
    )
    return cfg


def unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Combine reference-directed swing speed with racket-ground protection."""
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed20_env_cfg(
        play=play
    )
    if cfg.rewards is None:
        raise ValueError(
            "Intent sweet-speed racket-ground task requires active rewards."
        )
    cfg.rewards["reference_sweet_spot_velocity_reward"].weight = 50.0
    racket_ground_sensor = ContactSensorCfg(
        name="racket_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern="racket_ball_collision",
            entity="robot",
        ),
        secondary=ContactMatch(mode="geom", pattern="terrain"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        secondary_policy="error",
        history_length=cfg.decimation,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, racket_ground_sensor)
    cfg.rewards["racket_ground_collision"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-10.0,
        params={
            "sensor_name": "racket_ground_contact",
            "force_threshold": 10.0,
        },
    )
    return cfg


def unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Add 5% no-net or over-net-but-far balls to the large-court m11-1 task."""
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_env_cfg(
        play=play
    )
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.incoming_ball_failure_trajectory_probability = 0.05
    motion_cfg.incoming_ball_failure_trajectory_xy_distance_threshold_m = 3.0
    motion_cfg.incoming_ball_failure_no_net_fraction = 0.5
    intent_ball_group = cfg.observations["intent_ball_history"]
    assert isinstance(intent_ball_group, ObservationGroupCfg)
    if "valid" in intent_ball_group.terms or "age" in intent_ball_group.terms:
        raise ValueError("Large-court FailureTraj task must keep 6-D ball tokens.")
    print(
        "[INFO]: Large-court m11-1 FailureTraj task: probability=0.050, "
        "types=no-net|trajectory-min-xy-distance>3.000m, type_mix=0.500/0.500, "
        "ball_token=position3+velocity3; all other m11-1 settings unchanged"
    )
    return cfg


def unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_radial20_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Failure coverage with a deployment-exported 10-to-20 m radial encoding."""
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_env_cfg(
        play=play
    )
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.incoming_ball_failure_random_direction_fraction = 0.2
    motion_cfg.incoming_ball_failure_overhead_fraction = 0.2
    motion_cfg.incoming_ball_failure_rollout_horizon_s = 12.0
    motion_cfg.incoming_ball_failure_observation_radius_m = 20.0
    motion_cfg.incoming_ball_failure_stop_speed_m_s = 0.05
    encoding = {"linear_radius_m": 10.0, "limit_radius_m": 20.0}
    student = cfg.observations["student"]
    intent_ball = cfg.observations["intent_ball_history"]
    assert isinstance(student, ObservationGroupCfg)
    assert isinstance(intent_ball, ObservationGroupCfg)
    student.terms["ball_position"].params["position_encoding"] = dict(encoding)
    intent_ball.terms["position"].params["position_encoding"] = dict(encoding)
    print(
        "[INFO]: Failure coverage mix: 5% abnormal trajectories; 20% of "
        "failure rows are high-overhead candidates and a small fraction of "
        "remaining candidates start 2-5m from the root with 360-degree "
        "headings. Every candidate is checked "
        "against physical reachability. Root-frame ball radius <=10m is "
        "linear and larger radii smoothly approach 20m; trajectories stop "
        "at 20m or 0.05m/s (12s safety cap)."
    )
    return cfg


def unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_radial20_ball_occlusion_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Add randomized stale ball measurements to the radial-20 comparison task."""
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_radial20_env_cfg(
        play=play
    )
    perception_params = {
        "use_shared_perception": True,
        "perception_latency_min_steps": 0,
        "perception_latency_max_steps": 0 if play else 3,
        "perception_dropout_start_probability": 0.0 if play else 0.01,
        "perception_dropout_duration_min_steps": 1,
        "perception_dropout_duration_max_steps": 50,
        # Isolated missed frames, brief body occlusions, medium tracking gaps,
        # and long occlusions all hold the last world-frame measurement.
        "perception_dropout_duration_modes": (
            ()
            if play
            else ((0.25, 1, 1), (0.50, 2, 5), (0.20, 8, 15), (0.05, 25, 50))
        ),
        "perception_position_noise_std": 0.0 if play else 0.05,
        "perception_velocity_noise_std": 0.0 if play else 0.1,
        "perception_control_dt_s": 0.02,
    }

    student_group = cfg.observations["student"]
    intent_ball_group = cfg.observations["intent_ball_history"]
    assert isinstance(student_group, ObservationGroupCfg)
    assert isinstance(intent_ball_group, ObservationGroupCfg)
    for group in (student_group, intent_ball_group):
        for term_name in ("ball_position", "position"):
            if term_name in group.terms:
                group.terms[term_name].params.update(perception_params)
                group.terms[term_name].noise = None
        for term_name in ("ball_linear_velocity", "linear_velocity"):
            if term_name in group.terms:
                group.terms[term_name].params.update(perception_params)
                group.terms[term_name].noise = None

    status_params = {
        "ball_entity_name": "tennis_ball",
        "sample_lags": (20, 15, 10, 5, 0),
        "buffer_length": 25,
        **perception_params,
    }
    intent_ball_group.terms["valid"] = ObservationTermCfg(
        func=mdp.TennisBallObservationStatusHistory,
        params={**status_params, "quantity": "valid"},
    )
    intent_ball_group.terms["age"] = ObservationTermCfg(
        func=mdp.TennisBallObservationStatusHistory,
        params={**status_params, "quantity": "age"},
    )
    intent_ball_group.enable_corruption = False
    print(
        "[INFO]: Radial20 occlusion comparison: latency=0.."
        f"{perception_params['perception_latency_max_steps']} frames; "
        "dropout_start_probability="
        f"{perception_params['perception_dropout_start_probability']:.3f}; "
        "hold-last-world-measurement duration mix: 1 frame (25%), "
        "2-5 frames (50%), 8-15 frames (20%), 25-50 frames (5%); "
        "position_noise=0.05m, velocity_noise=0.10m/s."
    )
    return cfg


def unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_legacy_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Reproduce the pre-startup-frame 131D Student for legacy checkpoints."""
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_env_cfg(
        play=play
    )
    cfg.sim.mujoco.timestep = 0.005
    cfg.decimation = 4
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.landing_target_frame = "environment"

    student_group = cfg.observations["student"]
    critic_group = cfg.observations["critic"]
    assert isinstance(student_group, ObservationGroupCfg)
    assert isinstance(critic_group, ObservationGroupCfg)
    student_group.terms.pop("startup_heading", None)
    critic_group.terms.pop("startup_heading", None)

    if cfg.rewards is None:
        raise ValueError("Legacy NoTracking requires an active reward manager.")
    cfg.rewards[
        "post_strike_heading_reward"
    ].func = mdp.tennis_post_strike_heading_to_landing_target_reward
    return cfg


def unitree_g1_tennis_launch_distill_goal_time_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Expose deployable goal fields and reset after one launched-ball strike."""
    cfg = unitree_g1_tennis_launch_distill_env_cfg(play=play)
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.auto_chain_motion = False

    if cfg.terminations is None:
        cfg.terminations = {}
    cfg.terminations["motion_complete"] = TerminationTermCfg(
        func=mdp.phase_motion_complete,
        params={"command_name": "motion"},
    )

    student_group = cfg.observations["student"]
    assert isinstance(student_group, ObservationGroupCfg)
    student_group.terms["strike_target_position"] = ObservationTermCfg(
        func=mdp.motion_strike_target_position_b,
        params={"command_name": "motion", "target_index": 0},
    )
    student_group.terms["time_remaining"] = ObservationTermCfg(
        func=mdp.motion_time_remaining,
        params={"command_name": "motion"},
    )
    return cfg


def unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_speed5_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Use a 5 m/s target and 3 m/s std for the Alive comparison task."""
    cfg = unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_env_cfg(
        play=play,
        target_out_speed=5.0,
        out_speed_std=3.0,
    )
    return cfg


def unitree_g1_phase_acceleration_deadline_tppo_analytic_strike_distillation_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Derive racket and outgoing-ball goals from strike and landing positions."""
    cfg = unitree_g1_phase_acceleration_deadline_tppo_landing_distillation_env_cfg(
        play=play,
        maximum_full_reward_net_height=5.0,
    )
    from athlete.scripts.tennis_scene import (
        BALL_RADIUS,
        COURT_NET_HALF_WIDTH,
        COURT_NET_HEIGHT,
        COURT_NET_X,
    )

    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiTargetMotionCommandCfg)
    motion_cfg.analytic_strike_planner_enabled = True
    motion_cfg.analytic_ball_preferred_speed = 10.0
    motion_cfg.analytic_ball_maximum_speed = 20.0
    motion_cfg.analytic_incoming_ball_velocity = (0.0, 0.0, 0.0)
    motion_cfg.analytic_racket_effective_restitution = 0.7
    motion_cfg.analytic_gravity_magnitude = 9.81
    motion_cfg.analytic_ball_radius = BALL_RADIUS
    motion_cfg.analytic_net_x = COURT_NET_X
    motion_cfg.analytic_net_height = COURT_NET_HEIGHT
    motion_cfg.analytic_net_half_width = COURT_NET_HALF_WIDTH
    motion_cfg.analytic_net_clearance = 0.0
    motion_cfg.analytic_maximum_net_height = 5.0

    cfg.rewards["target_velocity_magnitude_reward"] = RewardTermCfg(
        func=mdp.all_motions_target_velocity_magnitude_error_exp,
        weight=1.0,
        params={"target_command_name": "motion", "std": 3.0},
    )
    landing_reward = cfg.rewards["ball_landing_reward"]
    landing_reward.params["use_analytic_target_out_speed"] = True
    print(
        "[INFO]: Analytic strike planner enabled: "
        "preferred_ball_speed=10.000 m/s, maximum_ball_speed=20.000 m/s, "
        "racket_effective_restitution=0.700, maximum_net_height=5.000 m"
    )
    return cfg
