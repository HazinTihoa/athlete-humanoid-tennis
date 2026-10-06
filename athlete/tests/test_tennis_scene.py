from __future__ import annotations

import math
from types import SimpleNamespace

import mujoco
import torch
from mjlab.scene import Scene
from mjlab.sensor import ContactMatch
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.noise import GaussianNoiseCfg
from athlete.goal_cond_tracking import mdp
from athlete.goal_cond_tracking.config.g1.env_cfgs import (
    LAUNCH_DISTILL_FLIGHT_TIME_S,
    LAUNCH_DISTILL_STRIKE_TARGET_FRAME0,
    unitree_g1_multi_target_tracking_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_analytic_strike_distillation_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_ball_launch_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_speed5_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_student_no_global_root_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_distilllinear_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_landing_distillation_env_cfg,
    unitree_g1_tennis_launch_distill_env_cfg,
    unitree_g1_tennis_launch_distill_goal_time_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_racket_ground10_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed20_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_env_cfg,
    unitree_g1_tennis_small_court_intent_transformer_failure_trajectory05_env_cfg,
    unitree_g1_tennis_small_court_intent_transformer_sweet_speed50_racket_ground10_env_cfg,
)
from athlete.goal_cond_tracking.mdp.analytic_strike import (
    align_quaternion_axis_to_vector,
    inverse_racket_velocity_and_normal,
    solve_low_arc_tennis_velocity,
)
from athlete.goal_cond_tracking.mdp.commands import (
    MotionCfg,
    MotionGoalCfg,
    incoming_ball_racket_orientation_active,
    override_single_motion_strike_position,
    rigid_point_linear_velocity,
    staged_landing_target_std,
    staged_target_pos_std_scale,
    tennis_target_pos_std_curriculum,
    transform_startup_frame_ground_target,
)
from athlete.goal_cond_tracking.mdp.observations import (
    TennisBallStridedHistory,
    gather_strided_history,
    motion_reference_state,
    motion_strike_target_position_b,
    motion_task_goal,
    motion_time_remaining,
    robot_global_root_pos_w,
    robot_projected_gravity_b,
    tennis_ball_linear_velocity_b,
    tennis_ball_position_b,
    tennis_landing_target_b,
    tennis_startup_heading_b,
    tennis_sweet_spot_linear_velocity_b,
    tennis_sweet_spot_position_b,
)
from athlete.goal_cond_tracking.mdp.phase_commands import (
    PhaseAwareMultiTargetMotionCommand,
)
from athlete.goal_cond_tracking.mdp.rewards import (
    ballistic_first_landing_position,
    direction_alignment_reward_with_tolerance,
    tennis_ball_direction_reward,
    tennis_ball_first_bounce_target_reward,
    tennis_ball_hit_reward,
    tennis_ball_net_clearance_reward,
    tennis_ball_out_speed_reward,
    tennis_ball_out_speed_score,
    tennis_ball_predicted_landing_target_reward,
    tennis_ball_strike_event,
    tennis_ball_target_projected_speed,
    tennis_ball_xy_direction_score,
    tennis_net_clearance_score,
    tennis_post_strike_heading_to_landing_target_reward,
    tennis_post_strike_heading_to_startup_x_reward,
    tennis_racket_ball_distance_reward,
    tennis_racket_ball_distance_score,
    tennis_racket_incoming_velocity_alignment_score,
    tennis_reference_sweet_spot_velocity_reward,
    tennis_reference_sweet_spot_velocity_score,
    tennis_strike_time_window_active,
)
from athlete.scripts.tennis_physics import (
    STANDARD_TENNIS_DOMAIN_RANDOMIZATION,
    STANDARD_TENNIS_PHYSICS,
    tennis_ball_aerodynamic_wrench_torch,
)
from athlete.scripts.tennis_scene import (
    COURT_FAR_SERVICE_LINE_X,
    RACKET_INERTIA_BODY_NAME,
    RACKET_NOMINAL_COM_WRIST_M,
    RACKET_NOMINAL_MASS_KG,
    TennisBallAerodynamicsController,
    TennisBallTargetController,
    TennisDomainRandomizationState,
    TennisIncomingBallController,
    configure_tennis_court_env,
    randomize_tennis_physics,
    tennis_ball_predicted_contact,
)


def test_ball_aerodynamic_wrench_keeps_multi_env_batch_shape() -> None:
    calls: dict[str, object] = {}
    ball = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_w=torch.randn(8, 3),
            root_link_ang_vel_w=torch.randn(8, 3),
        ),
    )

    def write_external_wrench_to_sim(
        forces: torch.Tensor,
        torques: torch.Tensor,
        *,
        body_ids: tuple[int, ...],
    ) -> None:
        calls.update(forces=forces, torques=torques, body_ids=body_ids)

    ball.write_external_wrench_to_sim = write_external_wrench_to_sim
    controller = TennisBallAerodynamicsController.__new__(
        TennisBallAerodynamicsController
    )
    controller.ball = ball

    controller.before_sim_step()

    assert calls["forces"].shape == (8, 3)
    assert calls["torques"].shape == (8, 3)
    assert calls["body_ids"] == (0,)


def test_ball_aerodynamics_supports_per_env_drag_without_magnus() -> None:
    velocity = torch.tensor([[8.0, 0.0, 1.0], [8.0, 0.0, 1.0]])
    spin = torch.tensor([[0.0, 50.0, 0.0], [0.0, -50.0, 0.0]])
    drag = torch.tensor([0.50, 0.65])

    force, _ = tennis_ball_aerodynamic_wrench_torch(
        velocity,
        spin,
        drag_coefficient=drag,
        magnus_coefficient=0.0,
    )

    assert STANDARD_TENNIS_PHYSICS.ball.magnus_coefficient == 0.0
    assert torch.all(force[:, 0] < 0.0)
    assert torch.linalg.vector_norm(force[1]) > torch.linalg.vector_norm(force[0])


def test_ground_bounce_applies_per_env_tangent_speed_retention() -> None:
    written: dict[str, torch.Tensor] = {}
    ball = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.04], [0.0, 0.0, 1.0]]),
            root_link_lin_vel_w=torch.tensor([[3.0, 1.0, 4.0], [2.0, 0.0, 1.0]]),
            root_link_ang_vel_w=torch.zeros(2, 3),
        )
    )

    def write_root_link_velocity_to_sim(
        velocity: torch.Tensor, *, env_ids: torch.Tensor
    ) -> None:
        written["velocity"] = velocity
        written["env_ids"] = env_ids

    ball.write_root_link_velocity_to_sim = write_root_link_velocity_to_sim
    controller = TennisBallAerodynamicsController.__new__(
        TennisBallAerodynamicsController
    )
    controller.env = SimpleNamespace(
        scene=SimpleNamespace(env_origins=torch.zeros(2, 3))
    )
    controller.ball = ball
    controller.domain_state = TennisDomainRandomizationState(
        ball_mass_kg=torch.full((2,), 0.0577),
        court_restitution=torch.full((2,), 0.745),
        ground_tangent_speed_retention=torch.tensor([0.8, 0.9]),
        drag_coefficient=torch.full((2,), 0.55),
        racket_mass_kg=torch.full((2,), 0.3),
        racket_restitution=torch.full((2,), 0.6),
        racket_com_offset_m=torch.zeros(2, 3),
    )
    controller._pre_step_linear_velocity_w = torch.tensor(
        [[4.0, 2.0, -3.0], [5.0, 0.0, -2.0]]
    )

    controller.after_sim_step()

    assert written["env_ids"].tolist() == [0]
    torch.testing.assert_close(written["velocity"][0, :2], torch.tensor([3.2, 1.6]))
    assert written["velocity"][0, 2] == 4.0


def _geom_ids_with_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        ).startswith(prefix)
    ]


def test_visual_court_is_collision_free_and_shared_across_worlds() -> None:
    cfg = unitree_g1_multi_target_tracking_env_cfg()
    cfg.scene.num_envs = 8
    original_nconmax = cfg.sim.nconmax
    original_njmax = cfg.sim.njmax

    configure_tennis_court_env(
        cfg,
        mode="visual",
        align_env_origins=True,
    )

    assert cfg.scene.env_spacing == 0.0
    assert "tennis_ball" not in cfg.scene.entities
    assert cfg.sim.nconmax == original_nconmax
    assert cfg.sim.njmax == original_njmax

    model = Scene(cfg.scene, device="cpu").compile()
    court_geom_ids = _geom_ids_with_prefix(model, "tennis_court/")
    assert len(court_geom_ids) == 20
    assert not model.geom_contype[court_geom_ids].any()
    assert not model.geom_conaffinity[court_geom_ids].any()

    terrain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    assert terrain_id >= 0
    assert model.geom_rgba[terrain_id, 3] == 0.0


def test_small_court_moves_and_narrows_the_physical_net() -> None:
    cfg = unitree_g1_tennis_small_court_intent_transformer_sweet_speed50_racket_ground10_env_cfg(
        play=True
    )
    model = Scene(cfg.scene, device="cpu").compile()
    net_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "tennis_court/net_collision",
    )

    assert net_id >= 0
    assert math.isclose(model.geom_pos[net_id, 0], 3.5)
    assert math.isclose(model.geom_size[net_id, 1], 2.5)


def test_failure_trajectory_task_keeps_six_dimensional_ball_tokens() -> None:
    cfg = unitree_g1_tennis_small_court_intent_transformer_failure_trajectory05_env_cfg(
        play=True
    )
    motion_cfg = cfg.commands["motion"]
    ball_group = cfg.observations["intent_ball_history"]

    assert motion_cfg.incoming_ball_failure_trajectory_probability == 0.05
    assert motion_cfg.incoming_ball_failure_trajectory_xy_distance_threshold_m == 3.0
    assert motion_cfg.incoming_ball_failure_no_net_fraction == 0.5
    assert tuple(ball_group.terms) == ("position", "linear_velocity")
    assert "valid" not in ball_group.terms
    assert "age" not in ball_group.terms


def test_large_court_failure_trajectory_task_only_adds_failure_sampling() -> None:
    baseline = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_env_cfg(
        play=True
    )
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_env_cfg(
        play=True
    )
    baseline_motion = baseline.commands["motion"]
    motion_cfg = cfg.commands["motion"]
    ball_group = cfg.observations["intent_ball_history"]

    assert baseline_motion.incoming_ball_failure_trajectory_probability == 0.0
    assert motion_cfg.incoming_ball_failure_trajectory_probability == 0.05
    assert motion_cfg.incoming_ball_failure_trajectory_xy_distance_threshold_m == 3.0
    assert motion_cfg.incoming_ball_failure_no_net_fraction == 0.5
    assert motion_cfg.tennis_court_net_x_m == baseline_motion.tennis_court_net_x_m
    assert (
        motion_cfg.incoming_ball_torch_match_position_min
        == baseline_motion.incoming_ball_torch_match_position_min
    )
    assert (
        motion_cfg.incoming_ball_torch_match_position_max
        == baseline_motion.incoming_ball_torch_match_position_max
    )
    assert (
        motion_cfg.incoming_ball_torch_match_horizontal_speed_range_m_s
        == baseline_motion.incoming_ball_torch_match_horizontal_speed_range_m_s
    )
    assert motion_cfg.landing_target_mean == baseline_motion.landing_target_mean
    assert tuple(ball_group.terms) == ("position", "linear_velocity")
    assert "failure_hold" not in baseline.observations
    assert "failure_hold" not in cfg.observations


def test_physical_court_adds_ball_racket_and_explicit_pairs() -> None:
    cfg = unitree_g1_multi_target_tracking_env_cfg()
    configure_tennis_court_env(
        cfg,
        mode="physical",
        align_env_origins=True,
    )

    assert "tennis_ball" in cfg.scene.entities
    assert cfg.sim.nconmax >= 200
    assert cfg.sim.njmax >= 500

    model = Scene(cfg.scene, device="cpu").compile()
    assert (
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "tennis_ball/tennis_ball_geom"
        )
        >= 0
    )
    assert (
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "robot/racket_ball_collision"
        )
        >= 0
    )
    racket_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        f"robot/{RACKET_INERTIA_BODY_NAME}",
    )
    assert racket_body_id >= 0
    assert math.isclose(model.body_mass[racket_body_id], RACKET_NOMINAL_MASS_KG)
    torch.testing.assert_close(
        torch.as_tensor(model.body_ipos[racket_body_id]),
        torch.tensor(RACKET_NOMINAL_COM_WRIST_M, dtype=torch.float64),
    )
    for pair_name in (
        "tennis_ball_court",
        "tennis_ball_surround",
        "tennis_ball_net",
        "tennis_ball_racket",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, pair_name) >= 0


def test_tppo_task_includes_fixed_landing_configuration() -> None:
    cfg = unitree_g1_phase_acceleration_deadline_tppo_landing_distillation_env_cfg()
    teacher_terms = tuple(cfg.observations["actor"].terms)

    motion_cfg = cfg.commands["motion"]
    assert motion_cfg.landing_target_enabled
    assert motion_cfg.landing_target_mean == (COURT_FAR_SERVICE_LINE_X, 0.0, 0.0)
    assert motion_cfg.landing_target_std == (1.0, 1.0, 0.0)
    assert motion_cfg.landing_target_radius == 0.5
    assert motion_cfg.contact_reward_window_s == 0.05
    assert motion_cfg.target_pos_std_scale == 0.0
    assert tuple(cfg.observations["actor"].terms) == teacher_terms
    assert "landing_target" in cfg.observations["student"].terms
    assert "landing_target" in cfg.observations["critic"].terms
    for term_name in (
        "ball_position",
        "ball_linear_velocity",
        "sweet_spot_linear_velocity",
    ):
        assert term_name in cfg.observations["student"].terms
        assert term_name in cfg.observations["critic"].terms
        assert term_name not in cfg.observations["actor"].terms
    assert "command" not in cfg.observations["critic"].terms
    assert (
        cfg.observations["critic"].terms["reference_motion_state"].func
        is motion_reference_state
    )
    assert cfg.observations["critic"].terms["task_goal"].func is motion_task_goal
    assert (
        cfg.observations["critic"].terms["time_remaining"].func is motion_time_remaining
    )
    student_root_pos = cfg.observations["student"].terms["global_root_pos"]
    critic_root_pos = cfg.observations["critic"].terms["global_root_pos"]
    assert cfg.observations["student"].enable_corruption
    assert not cfg.observations["critic"].enable_corruption
    assert student_root_pos.func is robot_global_root_pos_w
    assert isinstance(student_root_pos.noise, GaussianNoiseCfg)
    assert student_root_pos.noise.mean == 0.0
    assert student_root_pos.noise.std == 0.1
    assert critic_root_pos.func is robot_global_root_pos_w
    assert critic_root_pos.noise is None
    assert "global_root_pos" not in cfg.observations["actor"].terms
    assert (
        cfg.observations["student"].terms["projected_gravity"].func
        is robot_projected_gravity_b
    )
    assert (
        cfg.observations["critic"].terms["projected_gravity"].func
        is robot_projected_gravity_b
    )
    assert "projected_gravity" not in cfg.observations["actor"].terms
    assert tuple(cfg.observations["critic"].terms)[-8:] == (
        "task_goal",
        "time_remaining",
        "landing_target",
        "ball_position",
        "ball_linear_velocity",
        "sweet_spot_linear_velocity",
        "global_root_pos",
        "projected_gravity",
    )
    assert "ball_landing_reward" in cfg.rewards
    landing_reward = cfg.rewards["ball_landing_reward"]
    assert landing_reward.func is tennis_ball_predicted_landing_target_reward
    assert landing_reward.weight == 50.0
    assert landing_reward.params["prediction_delay_steps"] == 2
    assert landing_reward.params["maximum_full_reward_net_height"] == 2.0
    assert landing_reward.params["excess_net_height_std"] == 0.25
    assert "hit_ball_reward" not in cfg.rewards
    assert "ball_direction_reward" not in cfg.rewards
    assert landing_reward.params["ball_direction_std"] == 0.5
    assert "net_clearance_reward" in cfg.rewards
    assert cfg.rewards["net_clearance_reward"].func is tennis_ball_net_clearance_reward
    assert cfg.rewards["net_clearance_reward"].weight == 50.0
    assert cfg.rewards["target_orientation_reward"].params["tolerance_degrees"] == 60.0
    assert cfg.rewards["target_velocity_reward"].params["tolerance_degrees"] == 60.0
    assert "ball_out_speed_reward" in cfg.rewards
    speed_reward = cfg.rewards["ball_out_speed_reward"]
    assert speed_reward.func is tennis_ball_out_speed_reward
    assert speed_reward.weight == 50.0
    assert landing_reward.params["target_out_speed"] == 10.0
    assert landing_reward.params["out_speed_std"] == 10.0
    assert landing_reward.params["strike_speed_change_threshold"] == 0.05
    assert landing_reward.params["strike_proximity_threshold"] == 0.25
    assert "racket_ball_distance_reward" not in cfg.rewards
    assert "tennis_ball" in cfg.scene.entities
    assert cfg.scene.env_spacing == 0.0
    assert "landing_target_std" not in cfg.curriculum
    curriculum = cfg.curriculum["sample_target_std"]
    assert curriculum.func is tennis_target_pos_std_curriculum
    assert curriculum.params["stage_steps"] == (0, 12000, 24000, 36000, 48000)
    assert curriculum.params["stage_scales"] == (0.0, 0.25, 0.5, 0.75, 1.0)

    play_cfg = unitree_g1_phase_acceleration_deadline_tppo_landing_distillation_env_cfg(
        play=True
    )
    assert play_cfg.commands["motion"].landing_target_std == (1.0, 1.0, 0.0)
    assert play_cfg.commands["motion"].target_pos_std_scale == 1.0
    assert "landing_target_std" not in play_cfg.curriculum
    assert "sample_target_std" not in play_cfg.curriculum
    assert (
        play_cfg.rewards["ball_landing_reward"].params["maximum_full_reward_net_height"]
        == 2.0
    )


def test_distilllinear_env_uses_five_meter_net_reward_without_analytic_planner() -> (
    None
):
    cfg = unitree_g1_phase_acceleration_deadline_tppo_distilllinear_env_cfg()
    landing_reward = cfg.rewards["ball_landing_reward"]

    assert landing_reward.params["maximum_full_reward_net_height"] == 5.0
    assert landing_reward.params["target_out_speed"] == 7.0
    assert landing_reward.params["out_speed_std"] == 5.0
    assert not cfg.commands["motion"].analytic_strike_planner_enabled
    play_cfg = unitree_g1_phase_acceleration_deadline_tppo_distilllinear_env_cfg(
        play=True
    )
    assert (
        play_cfg.rewards["ball_landing_reward"].params["maximum_full_reward_net_height"]
        == 5.0
    )
    assert play_cfg.rewards["ball_landing_reward"].params["target_out_speed"] == 7.0
    assert play_cfg.rewards["ball_landing_reward"].params["out_speed_std"] == 5.0
    assert not play_cfg.commands["motion"].analytic_strike_planner_enabled


def test_distilllinear_alive_variants_only_add_survival_and_speed_override() -> None:
    base_cfg = unitree_g1_phase_acceleration_deadline_tppo_distilllinear_env_cfg()
    alive_cfg = (
        unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_env_cfg()
    )
    speed5_cfg = (
        unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_speed5_env_cfg()
    )

    assert "alive_reward" not in base_cfg.rewards
    assert alive_cfg.rewards["alive_reward"].func is mdp.is_alive
    assert alive_cfg.rewards["alive_reward"].weight == 1.0
    assert alive_cfg.rewards["ball_landing_reward"].params["target_out_speed"] == 7.0
    assert alive_cfg.rewards["ball_landing_reward"].params["out_speed_std"] == 5.0
    assert speed5_cfg.rewards["alive_reward"].func is mdp.is_alive
    assert speed5_cfg.rewards["alive_reward"].weight == 1.0
    assert speed5_cfg.rewards["ball_landing_reward"].params["target_out_speed"] == 5.0
    assert speed5_cfg.rewards["ball_landing_reward"].params["out_speed_std"] == 3.0
    assert not speed5_cfg.commands["motion"].analytic_strike_planner_enabled


def test_ball_launch_task_preserves_floor01_contract_and_fixes_strike_target() -> None:
    base_cfg = unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_env_cfg()
    cfg = unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_ball_launch_env_cfg()
    motion_cfg = cfg.commands["motion"]

    assert tuple(cfg.observations) == tuple(base_cfg.observations)
    for group_name in cfg.observations:
        assert tuple(cfg.observations[group_name].terms) == tuple(
            base_cfg.observations[group_name].terms
        )
    assert tuple(cfg.actions) == tuple(base_cfg.actions)
    assert tuple(cfg.rewards) == tuple(base_cfg.rewards)
    assert motion_cfg.target_pos_std_scale == 0.0
    assert motion_cfg.incoming_ball_launch_enabled
    assert motion_cfg.incoming_ball_initial_position == (11.93, -0.098, 1.2)
    assert motion_cfg.incoming_ball_initial_velocity == (
        -8.5,
        0.7594238934967334,
        4.171773616594168,
    )
    assert math.isclose(motion_cfg.incoming_ball_flight_time_s, 1.641330776)
    assert not motion_cfg.incoming_ball_trajectory_drives_deadline
    assert motion_cfg.incoming_ball_strike_target_position_frame0 is None
    assert motion_cfg.incoming_ball_position_noise_std == (0.0, 0.0, 0.0)
    assert motion_cfg.incoming_ball_velocity_noise_std == (0.0, 0.0, 0.0)
    assert motion_cfg.incoming_ball_max_strike_deviation == 0.0
    assert "sample_target_std" not in cfg.curriculum
    assert not base_cfg.commands["motion"].incoming_ball_launch_enabled


def test_alive_student_no_global_root_only_changes_student_observations() -> None:
    base_cfg = unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_env_cfg()
    cfg = unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_student_no_global_root_env_cfg()

    assert tuple(cfg.observations["student"].terms) == tuple(
        name
        for name in base_cfg.observations["student"].terms
        if name != "global_root_pos"
    )
    assert "global_root_pos" not in cfg.observations["student"].terms
    assert "global_root_pos" in cfg.observations["critic"].terms
    assert tuple(cfg.observations["actor"].terms) == tuple(
        base_cfg.observations["actor"].terms
    )
    assert tuple(cfg.observations["critic"].terms) == tuple(
        base_cfg.observations["critic"].terms
    )
    assert tuple(cfg.actions) == tuple(base_cfg.actions)
    assert tuple(cfg.rewards) == tuple(base_cfg.rewards)
    assert tuple(cfg.terminations or {}) == tuple(base_cfg.terminations or {})
    assert tuple(cfg.curriculum or {}) == tuple(base_cfg.curriculum or {})


def test_launch_distill_hides_teacher_strike_labels_from_student() -> None:
    base_cfg = unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_ball_launch_env_cfg()
    cfg = unitree_g1_tennis_launch_distill_env_cfg()
    play_cfg = unitree_g1_tennis_launch_distill_env_cfg(play=True)
    base_student = base_cfg.observations["student"]
    student = cfg.observations["student"]
    critic = cfg.observations["critic"]
    motion_cfg = cfg.commands["motion"]

    expected_student_terms = tuple(
        term_name
        for term_name in base_student.terms
        if term_name not in {"task_goal", "time_remaining", "global_root_pos"}
    ) + ("sweet_spot_position",)
    assert tuple(student.terms) == expected_student_terms
    assert "task_goal" not in student.terms
    assert "time_remaining" not in student.terms
    assert "global_root_pos" not in student.terms
    assert "task_goal" in critic.terms
    assert "time_remaining" in critic.terms
    assert "global_root_pos" in critic.terms
    physics_dr = cfg.events["tennis_physics_domain_randomization"]
    assert physics_dr.mode == "startup"
    assert physics_dr.func is randomize_tennis_physics
    assert physics_dr.params["cfg"] == STANDARD_TENNIS_DOMAIN_RANDOMIZATION
    assert "tennis_physics_domain_randomization" not in play_cfg.events
    assert "command" in cfg.observations["actor"].terms
    assert (
        motion_cfg.incoming_ball_strike_target_position_frame0
        == LAUNCH_DISTILL_STRIKE_TARGET_FRAME0
    )
    assert motion_cfg.incoming_ball_trajectory_drives_deadline
    assert math.isclose(
        motion_cfg.incoming_ball_flight_time_s, LAUNCH_DISTILL_FLIGHT_TIME_S
    )
    assert motion_cfg.incoming_ball_position_noise_std == (0.05, 0.05, 0.0)
    assert motion_cfg.incoming_ball_velocity_noise_std == (0.0, 0.0, 0.0)
    assert math.isclose(motion_cfg.incoming_ball_max_strike_deviation, 0.15)
    for term_name in ("ball_position", "ball_linear_velocity"):
        assert base_student.terms[term_name].history_length == 0
        assert student.terms[term_name].func is TennisBallStridedHistory
        assert student.terms[term_name].history_length == 0
        assert student.terms[term_name].params["sample_lags"] == (20, 15, 10, 5, 0)
        assert student.terms[term_name].params["buffer_length"] == 25
        assert critic.terms[term_name].history_length == 0
    assert isinstance(student.terms["ball_position"].noise, GaussianNoiseCfg)
    assert student.terms["ball_position"].noise.std == 0.05
    assert isinstance(student.terms["ball_linear_velocity"].noise, GaussianNoiseCfg)
    assert student.terms["ball_linear_velocity"].noise.std == 0.2
    assert student.terms["sweet_spot_position"].func is tennis_sweet_spot_position_b
    assert isinstance(student.terms["sweet_spot_position"].noise, GaussianNoiseCfg)
    assert student.terms["sweet_spot_position"].noise.std == 0.05
    assert isinstance(
        student.terms["sweet_spot_linear_velocity"].noise, GaussianNoiseCfg
    )
    assert student.terms["sweet_spot_linear_velocity"].noise.std == 0.2
    assert critic.terms["sweet_spot_position"].func is tennis_sweet_spot_position_b
    assert critic.terms["sweet_spot_position"].noise is None
    assert "racket_to_live_ball_reward" not in base_cfg.rewards
    racket_ball_reward = cfg.rewards["racket_to_live_ball_reward"]
    assert racket_ball_reward.func is tennis_racket_ball_distance_reward
    assert racket_ball_reward.weight == 10.0
    assert racket_ball_reward.params["distance_std"] == 0.20
    assert racket_ball_reward.params["time_std_s"] == 0.05
    assert racket_ball_reward.params["window_half_width_s"] == 0.05
    for term_name in set(student.terms) - {
        "ball_position",
        "ball_linear_velocity",
        "sweet_spot_position",
    }:
        assert (
            student.terms[term_name].history_length
            == base_student.terms[term_name].history_length
        )


def test_launch_distill_goal_time_adds_only_deployable_strike_fields() -> None:
    base_cfg = unitree_g1_tennis_launch_distill_env_cfg()
    cfg = unitree_g1_tennis_launch_distill_goal_time_env_cfg()
    play_cfg = unitree_g1_tennis_launch_distill_goal_time_env_cfg(play=True)
    student = cfg.observations["student"]

    assert base_cfg.commands["motion"].auto_chain_motion is True
    assert "motion_complete" not in (base_cfg.terminations or {})
    assert cfg.commands["motion"].auto_chain_motion is False
    assert play_cfg.commands["motion"].auto_chain_motion is False
    assert cfg.terminations["motion_complete"].func is mdp.phase_motion_complete
    assert cfg.terminations["motion_complete"].params == {"command_name": "motion"}
    assert play_cfg.terminations["motion_complete"].func is mdp.phase_motion_complete

    assert tuple(student.terms) == tuple(base_cfg.observations["student"].terms) + (
        "strike_target_position",
        "time_remaining",
    )
    assert "task_goal" not in student.terms
    assert (
        student.terms["strike_target_position"].func is motion_strike_target_position_b
    )
    assert student.terms["strike_target_position"].params == {
        "command_name": "motion",
        "target_index": 0,
    }
    assert student.terms["time_remaining"].func is motion_time_remaining
    assert student.terms["time_remaining"].params == {"command_name": "motion"}
    assert "tennis_physics_domain_randomization" in cfg.events
    assert "tennis_physics_domain_randomization" not in play_cfg.events


def test_motion_strike_target_position_uses_current_robot_anchor_frame() -> None:
    command = object.__new__(mdp.MultiTargetMotionCommand)
    command.target_position_w = torch.tensor([[[2.0, 3.0, 1.0]]])
    command.robot_anchor_body_index = 0
    command.robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pos_w=torch.tensor([[[1.0, 1.0, 0.0]]]),
            body_link_quat_w=torch.tensor(
                [[[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]]]
            ),
        )
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )

    position_b = motion_strike_target_position_b(env, "motion")

    torch.testing.assert_close(position_b, torch.tensor([[2.0, -1.0, 1.0]]))


def test_incoming_ball_override_replaces_only_position_target() -> None:
    original = MotionCfg(
        name="test",
        sub_targets=[
            MotionGoalCfg(goal_type="position"),
            MotionGoalCfg(
                goal_type="velocity",
                target_vel_mean={"x": 1.0, "y": 2.0, "z": 3.0},
            ),
        ],
    )

    overridden = override_single_motion_strike_position(
        [original], LAUNCH_DISTILL_STRIKE_TARGET_FRAME0
    )

    assert overridden[0].sub_targets[0].target_pos_mean == dict(
        zip(("x", "y", "z"), LAUNCH_DISTILL_STRIKE_TARGET_FRAME0, strict=True)
    )
    assert overridden[0].sub_targets[0].target_pos_frame == "reference_anchor_frame0"
    assert overridden[0].sub_targets[1].target_vel_mean == {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
    }
    assert original.sub_targets[0].target_pos_mean == {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
    }


def test_gather_strided_history_samples_every_five_steps() -> None:
    from mjlab.utils.buffers import CircularBuffer

    buffer = CircularBuffer(max_len=25, batch_size=1, device="cpu")
    for step in range(25):
        buffer.append(torch.tensor([[float(step), 0.0, 0.0]]))

    history = gather_strided_history(buffer, (20, 15, 10, 5, 0))
    assert history.shape == (1, 15)
    torch.testing.assert_close(
        history.reshape(1, 5, 3)[0, :, 0],
        torch.tensor([4.0, 9.0, 14.0, 19.0, 24.0]),
    )


def test_analytic_strike_task_uses_derived_ball_and_racket_goals() -> None:
    cfg = unitree_g1_phase_acceleration_deadline_tppo_analytic_strike_distillation_env_cfg()
    motion_cfg = cfg.commands["motion"]

    assert motion_cfg.analytic_strike_planner_enabled
    assert motion_cfg.analytic_ball_preferred_speed == 10.0
    assert motion_cfg.analytic_ball_maximum_speed == 20.0
    assert motion_cfg.analytic_racket_effective_restitution == 0.7
    assert motion_cfg.analytic_maximum_net_height == 5.0
    magnitude_reward = cfg.rewards["target_velocity_magnitude_reward"]
    assert magnitude_reward.weight == 1.0
    assert magnitude_reward.params["std"] == 3.0
    landing_reward = cfg.rewards["ball_landing_reward"]
    assert landing_reward.params["use_analytic_target_out_speed"]
    assert landing_reward.params["maximum_full_reward_net_height"] == 5.0
    assert landing_reward.params["out_speed_std"] == 10.0


def test_analytic_low_arc_lands_on_target_and_respects_five_meter_limit() -> None:
    strike = torch.tensor([[0.3193, 0.9376, 1.0884]])
    landing = torch.tensor([[12.0, 0.0, 0.0335]])
    velocity, feasible, net_height = solve_low_arc_tennis_velocity(
        strike,
        landing,
        preferred_speed=10.0,
        maximum_speed=20.0,
        net_x_w=torch.tensor([5.6]),
        net_y_w=torch.tensor([0.0]),
        net_half_width=5.485,
        minimum_net_height=0.914 + 0.0335,
        maximum_net_height=5.0,
    )

    horizontal_offset = landing[:, :2] - strike[:, :2]
    flight_time = torch.linalg.vector_norm(horizontal_offset, dim=-1) / (
        torch.linalg.vector_norm(velocity[:, :2], dim=-1)
    )
    reconstructed_landing = strike + velocity * flight_time[:, None]
    reconstructed_landing[:, 2] -= 0.5 * 9.81 * torch.square(flight_time)

    assert feasible.all()
    torch.testing.assert_close(reconstructed_landing, landing, atol=2.0e-5, rtol=1.0e-5)
    assert 10.0 <= torch.linalg.vector_norm(velocity, dim=-1).item() <= 20.0
    assert 0.914 + 0.0335 <= net_height.item() <= 5.0


def test_inverse_racket_goal_and_orientation_match_ball_impulse() -> None:
    incoming = torch.zeros(1, 3)
    outgoing = torch.tensor([[8.0, -1.0, 4.0]])
    racket_velocity, normal = inverse_racket_velocity_and_normal(
        incoming,
        outgoing,
        effective_restitution=0.7,
    )
    expected_normal = outgoing / torch.linalg.vector_norm(outgoing, dim=-1)[:, None]
    expected_racket_speed = torch.linalg.vector_norm(outgoing, dim=-1) / 1.7

    torch.testing.assert_close(normal, expected_normal)
    torch.testing.assert_close(
        torch.linalg.vector_norm(racket_velocity, dim=-1), expected_racket_speed
    )
    aligned_quaternion = align_quaternion_axis_to_vector(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        normal,
    )
    torch.testing.assert_close(
        quat_apply(aligned_quaternion, torch.tensor([[0.0, 0.0, 1.0]])),
        normal,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_direction_alignment_reward_has_sixty_degree_full_reward_cone() -> None:
    angles = torch.deg2rad(torch.tensor([0.0, 60.0, 90.0, 180.0]))
    source = torch.tensor([[1.0, 0.0, 0.0]]).expand(4, -1)
    target = torch.stack(
        [torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=-1
    )

    reward = direction_alignment_reward_with_tolerance(
        source,
        target,
        std=0.5,
        tolerance_degrees=60.0,
    )

    torch.testing.assert_close(reward[:2], torch.ones(2))
    assert 0.0 < reward[3] < reward[2] < 1.0


def test_direction_alignment_reward_rejects_zero_velocity() -> None:
    reward = direction_alignment_reward_with_tolerance(
        torch.zeros(1, 3),
        torch.tensor([[1.0, 0.0, 0.0]]),
        std=0.5,
        tolerance_degrees=60.0,
    )
    torch.testing.assert_close(reward, torch.zeros(1))


def test_racket_incoming_velocity_alignment_uses_motion_template_sign() -> None:
    incoming_velocity = torch.tensor([[-4.0, 0.0, 0.0]])
    forehand_template_axis = torch.tensor([[1.0, 0.0, 0.0]])
    backhand_template_axis = torch.tensor([[-1.0, 0.0, 0.0]])

    forehand_score = tennis_racket_incoming_velocity_alignment_score(
        torch.tensor([[1.0, 0.0, 0.0]]),
        forehand_template_axis,
        incoming_velocity,
        std=0.5,
        tolerance_degrees=60.0,
    )
    backhand_score = tennis_racket_incoming_velocity_alignment_score(
        torch.tensor([[-1.0, 0.0, 0.0]]),
        backhand_template_axis,
        incoming_velocity,
        std=0.5,
        tolerance_degrees=60.0,
    )
    wrong_forehand_face_score = tennis_racket_incoming_velocity_alignment_score(
        torch.tensor([[-1.0, 0.0, 0.0]]),
        forehand_template_axis,
        incoming_velocity,
        std=0.5,
        tolerance_degrees=60.0,
    )

    torch.testing.assert_close(forehand_score, torch.ones(1))
    torch.testing.assert_close(backhand_score, torch.ones(1))
    assert wrong_forehand_face_score.item() < 0.1


def test_racket_orientation_window_spans_before_and_after_strike() -> None:
    active = incoming_ball_racket_orientation_active(
        torch.tensor([-0.1, -0.101, 0.0, 0.1, 0.101, 0.05, 0.0]),
        torch.tensor([0.5, 1.0, 0.49, 1.0, 1.0, 0.0, 1.0]),
        torch.tensor([False, False, False, True, True, True, True]),
        is_paused=torch.tensor([False, False, False, False, False, False, True]),
        window_before_s=0.1,
        window_after_s=0.1,
        minimum_ball_speed=0.5,
    )

    assert active.tolist() == [True, False, False, True, False, True, False]


def test_forehand_backhand_hit_metrics_record_each_attempt_once() -> None:
    command = object.__new__(PhaseAwareMultiTargetMotionCommand)
    command._stroke_attempt_active = torch.tensor([True, True, False])
    command._stroke_outcome_recorded = torch.zeros(3, dtype=torch.bool)
    command.trajectory_match_valid = torch.ones(3, dtype=torch.bool)
    command.strike_time_error_s = torch.tensor([0.1, 0.1, 0.1])
    command.which_motion = torch.tensor([0, 1, 0])
    command.ball_has_been_struck = torch.tensor([True, False, True])
    command._motion_is_forehand_t = torch.tensor([True, False])
    command._motion_is_backhand_t = torch.tensor([False, True])
    command.cfg = SimpleNamespace(racket_orientation_reward_window_after_s=0.1)
    command.metrics = {}
    for side in ("forehand", "backhand"):
        command.metrics[f"{side}_strike_attempts"] = torch.zeros(3)
        command.metrics[f"{side}_hits"] = torch.zeros(3)
        command.metrics[f"{side}_hit_success_rate"] = torch.zeros(3)

    env_ids = torch.arange(3)
    command._record_stroke_outcomes(env_ids)
    command._record_stroke_outcomes(env_ids)

    assert command.metrics["forehand_strike_attempts"].tolist() == [1.0, 0.0, 0.0]
    assert command.metrics["forehand_hits"].tolist() == [1.0, 0.0, 0.0]
    assert command.metrics["backhand_strike_attempts"].tolist() == [0.0, 1.0, 0.0]
    assert command.metrics["backhand_hits"].tolist() == [0.0, 0.0, 0.0]


def test_unmatched_trajectory_is_not_counted_as_a_strike_attempt() -> None:
    command = object.__new__(PhaseAwareMultiTargetMotionCommand)
    command._stroke_attempt_active = torch.tensor([True])
    command._stroke_outcome_recorded = torch.zeros(1, dtype=torch.bool)
    command.trajectory_match_valid = torch.tensor([False])
    command.strike_time_error_s = torch.tensor([1.0])
    command.which_motion = torch.tensor([0])
    command.ball_has_been_struck = torch.tensor([False])
    command._motion_is_forehand_t = torch.tensor([True])
    command._motion_is_backhand_t = torch.tensor([False])
    command.cfg = SimpleNamespace(racket_orientation_reward_window_after_s=0.1)
    command.metrics = {}
    for side in ("forehand", "backhand"):
        command.metrics[f"{side}_strike_attempts"] = torch.zeros(1)
        command.metrics[f"{side}_hits"] = torch.zeros(1)
        command.metrics[f"{side}_hit_success_rate"] = torch.zeros(1)

    command._record_stroke_outcomes(torch.arange(1), force=True)

    assert command.metrics["forehand_strike_attempts"].item() == 0.0
    assert command.metrics["backhand_strike_attempts"].item() == 0.0


def test_landing_target_std_curriculum_changes_only_at_stage_boundaries() -> None:
    stage_steps = (0, 12000, 24000, 36000, 48000)
    stage_stds = (
        (0.0, 0.0, 0.0),
        (0.25, 0.25, 0.0),
        (0.5, 0.5, 0.0),
        (0.75, 0.75, 0.0),
        (1.0, 1.0, 0.0),
    )

    assert staged_landing_target_std(0, stage_steps, stage_stds) == (0, stage_stds[0])
    assert staged_landing_target_std(11999, stage_steps, stage_stds) == (
        0,
        stage_stds[0],
    )
    assert staged_landing_target_std(12000, stage_steps, stage_stds) == (
        1,
        stage_stds[1],
    )
    assert staged_landing_target_std(47999, stage_steps, stage_stds) == (
        3,
        stage_stds[3],
    )
    assert staged_landing_target_std(48000, stage_steps, stage_stds) == (
        4,
        stage_stds[4],
    )


def test_sample_target_std_curriculum_changes_only_at_stage_boundaries() -> None:
    stage_steps = (0, 12000, 24000, 36000, 48000)
    stage_scales = (0.0, 0.25, 0.5, 0.75, 1.0)

    assert staged_target_pos_std_scale(0, stage_steps, stage_scales) == (0, 0.0)
    assert staged_target_pos_std_scale(11999, stage_steps, stage_scales) == (0, 0.0)
    assert staged_target_pos_std_scale(12000, stage_steps, stage_scales) == (
        1,
        0.25,
    )
    assert staged_target_pos_std_scale(47999, stage_steps, stage_scales) == (
        3,
        0.75,
    )
    assert staged_target_pos_std_scale(48000, stage_steps, stage_scales) == (4, 1.0)


def test_global_root_pos_observation_returns_current_world_position() -> None:
    root_pos_w = torch.tensor([[0.5, -0.25, 0.78], [1.0, 2.0, 0.81]])
    command = SimpleNamespace(robot_anchor_pos_w=root_pos_w)
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )

    torch.testing.assert_close(
        robot_global_root_pos_w(env, command_name="motion"),
        root_pos_w,
    )


def test_projected_gravity_observation_uses_current_robot_orientation() -> None:
    half_sqrt_two = math.sqrt(0.5)
    root_quat_w = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [half_sqrt_two, half_sqrt_two, 0.0, 0.0],
        ]
    )
    gravity_vec_w = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    command = SimpleNamespace(
        robot_anchor_quat_w=root_quat_w,
        robot=SimpleNamespace(data=SimpleNamespace(gravity_vec_w=gravity_vec_w)),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )

    torch.testing.assert_close(
        robot_projected_gravity_b(env, command_name="motion"),
        torch.tensor([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_startup_heading_observation_uses_only_relative_yaw() -> None:
    half_sqrt_two = math.sqrt(0.5)
    command = SimpleNamespace(
        startup_anchor_yaw_w=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [half_sqrt_two, 0.0, 0.0, half_sqrt_two],
            ]
        ),
        robot_anchor_quat_w=torch.tensor(
            [
                [half_sqrt_two, 0.0, 0.0, half_sqrt_two],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )

    torch.testing.assert_close(
        tennis_startup_heading_b(env, command_name="motion"),
        torch.tensor([[0.0, -1.0], [0.0, 1.0]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_startup_landing_target_observation_does_not_follow_current_root() -> None:
    command = SimpleNamespace(
        cfg=SimpleNamespace(
            landing_target_enabled=True,
            landing_target_frame="startup",
        ),
        target_landing_position_startup=torch.tensor(
            [[7.0, 0.5, 0.0], [6.0, -0.25, 0.0]]
        ),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )

    torch.testing.assert_close(
        tennis_landing_target_b(env, command_name="motion"),
        torch.tensor([[7.0, 0.5], [6.0, -0.25]]),
    )


def test_startup_ground_target_transform_uses_initial_xy_yaw_and_ground_z() -> None:
    half_sqrt_two = math.sqrt(0.5)
    target_startup = torch.tensor([[7.0, 0.0, 0.0]])
    target_w = transform_startup_frame_ground_target(
        target_startup,
        startup_anchor_pos_w=torch.tensor([[10.0, 20.0, 0.8]]),
        startup_anchor_yaw_w=torch.tensor([[half_sqrt_two, 0.0, 0.0, half_sqrt_two]]),
        env_origin_w=torch.tensor([[8.0, 15.0, 0.1]]),
    )

    torch.testing.assert_close(
        target_w,
        torch.tensor([[10.0, 27.0, 0.1]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_live_tennis_observations_use_robot_anchor_frame() -> None:
    half_sqrt_two = math.sqrt(0.5)
    root_quat_w = torch.tensor([[half_sqrt_two, 0.0, 0.0, half_sqrt_two]])
    source_position_w = torch.tensor([[[1.0, 3.0, 1.0]]])
    source_velocity_w = torch.tensor([[[1.0, 0.0, 0.0]]])
    command = SimpleNamespace(
        robot_anchor_pos_w=torch.tensor([[1.0, 2.0, 0.5]]),
        robot_anchor_quat_w=root_quat_w,
        get_source_pos_w=lambda: source_position_w,
        get_source_lin_vel_w=lambda: source_velocity_w,
    )
    ball = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[2.0, 2.0, 0.5]]),
            root_link_lin_vel_w=torch.tensor([[0.0, 2.0, 0.0]]),
        )
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
        scene={"tennis_ball": ball},
    )

    torch.testing.assert_close(
        tennis_ball_position_b(env, command_name="motion"),
        torch.tensor([[0.0, -1.0, 0.0]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        tennis_ball_linear_velocity_b(env, command_name="motion"),
        torch.tensor([[2.0, 0.0, 0.0]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        tennis_sweet_spot_position_b(env, command_name="motion"),
        torch.tensor([[1.0, 0.0, 0.5]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        tennis_sweet_spot_linear_velocity_b(env, command_name="motion"),
        torch.tensor([[0.0, -1.0, 0.0]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_reference_motion_state_contains_only_joint_position_and_velocity() -> None:
    joint_pos = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    joint_vel = joint_pos + 10.0
    command = SimpleNamespace(joint_pos=joint_pos, joint_vel=joint_vel)
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )

    torch.testing.assert_close(
        motion_reference_state(env, command_name="motion"),
        torch.cat([joint_pos, joint_vel], dim=-1),
    )


def test_first_bounce_reward_is_full_inside_ring_and_one_shot() -> None:
    command = SimpleNamespace(
        cfg=SimpleNamespace(landing_target_enabled=True),
        ball_has_been_struck=torch.zeros(2, dtype=torch.bool),
        ball_landing_recorded=torch.zeros(2, dtype=torch.bool),
        ball_previous_vertical_velocity=torch.full((2,), -1.0),
        ball_landing_position_w=torch.zeros(2, 3),
        target_landing_position_w=torch.tensor([[12.0, 0.0, 0.0], [12.0, 0.0, 0.0]]),
        metrics={"error_ball_landing": torch.zeros(2)},
    )
    ball = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[12.3, 0.0, 0.04], [14.0, 0.0, 0.04]]),
            root_link_lin_vel_w=torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        )
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
        scene={"tennis_ball": ball},
    )

    reward = tennis_ball_first_bounce_target_reward(
        env,
        command_name="motion",
        std=1.0,
        target_radius=0.5,
    )
    torch.testing.assert_close(reward[0], torch.tensor(1.0))
    torch.testing.assert_close(reward[1], torch.tensor(math.exp(-((2.0 - 0.5) ** 2))))
    assert command.ball_landing_recorded.all()

    repeated_reward = tennis_ball_first_bounce_target_reward(
        env,
        command_name="motion",
        std=1.0,
        target_radius=0.5,
    )
    torch.testing.assert_close(repeated_reward, torch.zeros(2))


def test_ballistic_landing_prediction_matches_closed_form() -> None:
    position = torch.tensor([[0.0, 0.0, 1.0]])
    velocity = torch.tensor([[10.0, 0.0, 3.0]])

    landing, flight_time = ballistic_first_landing_position(
        position,
        velocity,
        gravity_magnitude=9.81,
        ground_contact_height=0.0335,
    )

    expected_time = (3.0 + math.sqrt(3.0**2 + 2.0 * 9.81 * (1.0 - 0.0335))) / 9.81
    torch.testing.assert_close(flight_time, torch.tensor([expected_time]))
    torch.testing.assert_close(
        landing,
        torch.tensor([[10.0 * expected_time, 0.0, 0.0335]]),
    )


def test_net_clearance_is_full_through_two_meters_then_decays() -> None:
    heights = torch.tensor([1.0, 1.5, 2.0, 2.25, 2.5])
    score = tennis_net_clearance_score(
        heights,
        torch.ones(5, dtype=torch.bool),
        maximum_full_reward_height=2.0,
        excess_height_std=0.25,
    )

    torch.testing.assert_close(
        score,
        torch.tensor([1.0, 1.0, 1.0, math.exp(-1.0), math.exp(-4.0)]),
    )
    invalid_score = tennis_net_clearance_score(
        torch.tensor([1.5]),
        torch.zeros(1, dtype=torch.bool),
        maximum_full_reward_height=2.0,
        excess_height_std=0.25,
    )
    torch.testing.assert_close(invalid_score, torch.zeros(1))


def test_ball_out_speed_score_is_one_sided_and_saturates_at_target() -> None:
    speed = torch.tensor([0.0, 0.5, 2.0, 5.0, 10.0, 12.0])
    score = tennis_ball_out_speed_score(speed, target_speed=10.0, std=10.0)

    torch.testing.assert_close(
        score,
        torch.tensor(
            [
                math.exp(-1.0),
                math.exp(-0.9025),
                math.exp(-0.64),
                math.exp(-0.25),
                1.0,
                1.0,
            ]
        ),
    )

    per_env_target = torch.tensor([10.0, 12.0])
    torch.testing.assert_close(
        tennis_ball_out_speed_score(
            torch.tensor([12.0, 10.0]), target_speed=per_env_target, std=10.0
        ),
        torch.tensor([1.0, math.exp(-0.04)]),
    )


def test_ball_target_projected_speed_uses_only_forward_horizontal_velocity() -> None:
    velocity_xy = torch.tensor([[3.0, 4.0], [-3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])
    target_offset_xy = torch.tensor([[10.0, 0.0], [10.0, 0.0], [0.0, 10.0], [0.0, 0.0]])

    torch.testing.assert_close(
        tennis_ball_target_projected_speed(velocity_xy, target_offset_xy),
        torch.tensor([3.0, 0.0, 4.0, 0.0]),
    )


def test_floor03_ball_speed_reward_aligns_to_landing_target() -> None:
    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_env_cfg()

    assert cfg.rewards is not None
    assert (
        cfg.rewards["ball_landing_reward"].params["align_out_speed_to_landing_target"]
        is True
    )
    assert cfg.rewards["action_rate_l2"].weight == -0.04
    assert cfg.rewards["ball_landing_reward"].weight == 100.0
    assert cfg.rewards["net_clearance_reward"].weight == 100.0
    assert cfg.rewards["ball_out_speed_reward"].weight == 100.0


def test_rigid_point_velocity_includes_parent_angular_velocity() -> None:
    velocity = rigid_point_linear_velocity(
        parent_linear_velocity_w=torch.tensor([1.0, 0.0, 0.0]),
        parent_angular_velocity_w=torch.tensor([0.0, 0.0, 2.0]),
        parent_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        point_offset_parent=torch.tensor([0.5, 0.0, 0.0]),
    )

    torch.testing.assert_close(velocity, torch.tensor([1.0, 1.0, 0.0]))


def test_reference_sweet_spot_speed_uses_motion_specific_signed_direction() -> None:
    reference_velocity = torch.tensor(
        [[8.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [8.0, 0.0, 0.0]]
    )
    source_velocity = torch.tensor(
        [[6.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [-6.0, 0.0, 0.0]]
    )

    torch.testing.assert_close(
        tennis_reference_sweet_spot_velocity_score(
            source_velocity,
            reference_velocity,
            target_speed_scale=0.75,
        ),
        torch.tensor([1.0, 1.0, 0.0]),
    )


def test_reference_sweet_spot_velocity_reward_gates_direction_time_and_proximity() -> (
    None
):
    source_position = torch.tensor(
        [[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0]]]
    )
    source_velocity = torch.tensor(
        [[[6.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]], [[-3.0, 0.0, 0.0]]]
    )
    reference_velocity = torch.tensor(
        [[[8.0, 0.0, 0.0]], [[-4.0, 0.0, 0.0]], [[-4.0, 0.0, 0.0]]]
    )
    command = SimpleNamespace(
        get_source_pos_w=lambda: source_position,
        get_source_lin_vel_w=lambda: source_velocity,
        get_reference_source_strike_lin_vel_w=lambda: reference_velocity,
        _target_pos_reward_weights_t=torch.ones(1, 1),
        which_motion=torch.zeros(3, dtype=torch.long),
        time_remaining=torch.zeros(3),
        ball_has_been_struck=torch.zeros(3, dtype=torch.bool),
        metrics={},
    )
    ball = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1),
            root_link_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0]]).expand(3, -1),
        )
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
        scene={"tennis_ball": ball},
    )

    reward = tennis_reference_sweet_spot_velocity_reward(
        env,
        command_name="motion",
        target_speed_scale=0.75,
        contact_window_s=0.1,
        proximity_std=0.2,
    )

    torch.testing.assert_close(reward[:2], torch.tensor([1.0, 0.0]))
    assert reward[2].item() < 1.0e-10
    torch.testing.assert_close(
        command.metrics["racket_sweet_spot_projected_speed"],
        torch.tensor([6.0, -3.0, 3.0]),
    )


def test_intent_floor005_sweet_speed_task_is_an_isolated_reward_variant() -> None:
    baseline = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_env_cfg()
    variant = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed20_env_cfg()

    assert baseline.rewards is not None
    assert variant.rewards is not None
    assert "reference_sweet_spot_velocity_reward" not in baseline.rewards
    reward = variant.rewards["reference_sweet_spot_velocity_reward"]
    assert reward.weight == 20.0
    assert reward.params["target_speed_scale"] == 0.75
    assert reward.params["contact_window_s"] == 0.1
    assert reward.params["proximity_std"] == 0.2


def test_intent_floor005_racket_ground_task_is_an_isolated_contact_variant() -> None:
    baseline = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg()
    variant = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_racket_ground10_env_cfg()

    assert baseline.rewards is not None
    assert variant.rewards is not None
    assert "racket_ground_collision" not in baseline.rewards
    reward = variant.rewards["racket_ground_collision"]
    assert reward.weight == -10.0
    assert reward.params == {
        "sensor_name": "racket_ground_contact",
        "force_threshold": 10.0,
    }

    baseline_sensor_names = {sensor.name for sensor in baseline.scene.sensors}
    assert "racket_ground_contact" not in baseline_sensor_names
    sensor = next(
        sensor
        for sensor in variant.scene.sensors
        if sensor.name == "racket_ground_contact"
    )
    assert sensor.primary == ContactMatch(
        mode="geom",
        pattern="racket_ball_collision",
        entity="robot",
    )
    assert sensor.secondary == ContactMatch(mode="geom", pattern="terrain")
    assert sensor.fields == ("found", "force")
    assert sensor.reduce == "maxforce"
    assert sensor.history_length == variant.decimation == 8


def test_intent_floor005_sweet_speed_racket_ground_task_combines_both_rewards() -> None:
    variant = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_env_cfg()

    assert variant.rewards is not None
    assert variant.rewards["reference_sweet_spot_velocity_reward"].weight == 50.0
    assert variant.rewards["racket_ground_collision"].weight == -10.0
    sensor = next(
        sensor
        for sensor in variant.scene.sensors
        if sensor.name == "racket_ground_contact"
    )
    assert sensor.history_length == variant.decimation == 8


def test_ball_xy_direction_score_rewards_alignment_and_rejects_zero_speed() -> None:
    angles = torch.deg2rad(torch.tensor([0.0, 60.0, 90.0, 180.0]))
    velocity_xy = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    target_xy = torch.tensor([[1.0, 0.0]]).expand(4, -1)

    score = tennis_ball_xy_direction_score(velocity_xy, target_xy, std=0.5)

    torch.testing.assert_close(
        score,
        torch.tensor([1.0, math.exp(-1.0), math.exp(-4.0), math.exp(-16.0)]),
    )
    torch.testing.assert_close(
        tennis_ball_xy_direction_score(torch.zeros(1, 2), target_xy[:1], std=0.5),
        torch.zeros(1),
    )


def test_dense_racket_ball_score_is_spatially_and_temporally_focused() -> None:
    score = tennis_racket_ball_distance_score(
        torch.tensor([0.0, 0.2, 0.0]),
        torch.tensor([0.0, 0.0, 0.2]),
        distance_std=0.2,
        time_std_s=0.2,
    )

    torch.testing.assert_close(
        score,
        torch.tensor([1.0, math.exp(-1.0), math.exp(-1.0)]),
    )


def test_dense_racket_ball_score_is_zero_outside_hard_window() -> None:
    score = tennis_racket_ball_distance_score(
        torch.zeros(5),
        torch.tensor([-0.051, -0.05, 0.0, 0.05, 0.051]),
        distance_std=0.2,
        time_std_s=0.05,
        window_half_width_s=0.05,
    )

    torch.testing.assert_close(
        score,
        torch.tensor([0.0, math.exp(-1.0), 1.0, math.exp(-1.0), 0.0]),
    )


def test_strike_event_requires_a_nearby_horizontal_velocity_change() -> None:
    initialized = torch.ones(4, dtype=torch.bool)
    previous_velocity = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.06, 0.0, 0.0],
        ]
    )
    current_velocity = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    previous_offset = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ]
    )
    current_offset = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [-0.3, 0.0, 0.0],
            [-0.3, 0.0, 0.0],
        ]
    )

    event = tennis_ball_strike_event(
        current_velocity,
        previous_velocity,
        current_offset,
        previous_offset,
        initialized,
        speed_change_threshold=0.05,
        proximity_threshold=0.25,
    )

    torch.testing.assert_close(event, torch.tensor([False, False, True, False]))


def test_strike_timing_window_is_symmetric_around_deadline() -> None:
    active = tennis_strike_time_window_active(
        torch.tensor([-0.051, -0.05, 0.0, 0.05, 0.051]),
        contact_window_s=0.05,
    )

    torch.testing.assert_close(
        active,
        torch.tensor([False, True, True, True, False]),
    )


def test_contact_prediction_requires_an_approaching_nearby_trajectory() -> None:
    ball_position = torch.tensor(
        [[0.10, 0.0, 1.0], [0.10, 0.0, 1.0], [0.10, 0.30, 1.0]]
    )
    ball_velocity = torch.zeros(3, 3)
    sweet_position = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1)
    sweet_velocity = torch.tensor([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    predicted = tennis_ball_predicted_contact(
        ball_position,
        ball_velocity,
        sweet_position,
        sweet_velocity,
        prediction_horizon_s=0.02,
        proximity_threshold=0.25,
    )

    torch.testing.assert_close(predicted, torch.tensor([True, False, False]))


def _make_ball_controller_stub():
    qpos = torch.zeros(1, 7)
    qvel = torch.zeros(1, 6)
    root_position = torch.tensor([[0.0, 0.0, 1.0]])
    root_velocity = torch.zeros(1, 3)
    ball = SimpleNamespace(
        data=SimpleNamespace(
            indexing=SimpleNamespace(
                free_joint_q_adr=slice(0, 7),
                free_joint_v_adr=slice(0, 6),
            ),
            data=SimpleNamespace(qpos=qpos, qvel=qvel),
            root_link_pos_w=root_position,
            root_link_lin_vel_w=root_velocity,
        )
    )
    source_position = torch.tensor([[[1.0, 0.0, 1.0]]])
    source_velocity = torch.zeros(1, 1, 3)
    motion = SimpleNamespace(
        time_remaining=torch.tensor([1.0]),
        contact_time=torch.tensor([1.0]),
        which_motion=torch.zeros(1, dtype=torch.long),
        motion_configs=[
            SimpleNamespace(
                sub_targets=[
                    SimpleNamespace(
                        goal_type="position",
                        target_phase_start=0.4,
                        target_phase_end=0.6,
                    )
                ]
            )
        ],
        _time_step_totals=torch.tensor([100]),
        cfg=SimpleNamespace(
            motion_files=["motion.npz"],
            contact_reward_window_s=0.05,
        ),
        target_position_w=torch.tensor([[[0.0, 0.0, 1.0]]]),
        get_source_pos_w=lambda: source_position,
        get_source_lin_vel_w=lambda: source_velocity,
    )
    env = SimpleNamespace(
        scene={"tennis_ball": ball},
        command_manager=SimpleNamespace(get_term=lambda _: motion),
        device="cpu",
        num_envs=1,
        step_dt=0.02,
    )
    controller = TennisBallTargetController(env, release_lead_s=0.02)
    return (
        controller,
        motion,
        source_position,
        source_velocity,
        root_velocity,
        qvel,
    )


def _make_incoming_ball_controller_stub(
    *,
    trajectory_drives_deadline: bool = False,
    position_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocity_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_strike_deviation: float = 0.0,
):
    qpos = torch.zeros(1, 7)
    qvel = torch.ones(1, 6)
    ball = SimpleNamespace(
        data=SimpleNamespace(
            indexing=SimpleNamespace(
                free_joint_q_adr=slice(0, 7),
                free_joint_v_adr=slice(0, 6),
            ),
            data=SimpleNamespace(qpos=qpos, qvel=qvel),
        )
    )
    motion = SimpleNamespace(
        time_remaining=torch.tensor([2.0]),
        contact_time=torch.tensor([2.0]),
        which_motion=torch.zeros(1, dtype=torch.long),
        cfg=SimpleNamespace(
            motion_files=["ep_0000.npz"],
            incoming_ball_launch_enabled=True,
            incoming_ball_initial_position=(11.93, -0.098, 1.2),
            incoming_ball_initial_velocity=(-8.5, 0.75, 4.17),
            incoming_ball_initial_angular_velocity=(0.0, 0.0, 0.0),
            incoming_ball_flight_time_s=1.641330776,
            incoming_ball_trajectory_drives_deadline=trajectory_drives_deadline,
            incoming_ball_position_noise_std=position_noise_std,
            incoming_ball_velocity_noise_std=velocity_noise_std,
            incoming_ball_max_strike_deviation=max_strike_deviation,
        ),
    )

    class FakeScene(dict):
        pass

    scene = FakeScene(tennis_ball=ball)
    scene.env_origins = torch.tensor([[10.0, 20.0, 0.0]])
    env = SimpleNamespace(
        scene=scene,
        command_manager=SimpleNamespace(get_term=lambda _: motion),
        device="cpu",
        num_envs=1,
        step_dt=0.02,
    )
    return TennisIncomingBallController(env), motion, qpos, qvel


def test_incoming_ball_controller_holds_launches_and_rearms_after_reset() -> None:
    controller, motion, qpos, qvel = _make_incoming_ball_controller_stub()

    controller.before_step()
    torch.testing.assert_close(qpos[:, :3], torch.tensor([[21.93, 19.902, 1.2]]))
    torch.testing.assert_close(qpos[:, 3:], torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(qvel, torch.zeros_like(qvel))
    assert not controller._launched.item()

    motion.time_remaining[:] = 1.64
    controller.before_step()
    torch.testing.assert_close(qvel, torch.tensor([[-8.5, 0.75, 4.17, 0.0, 0.0, 0.0]]))
    assert controller._launched.item()

    controller.after_step(torch.tensor([True]))
    motion.time_remaining[:] = LAUNCH_DISTILL_FLIGHT_TIME_S
    controller.before_step()
    torch.testing.assert_close(qvel, torch.tensor([[-8.5, 0.75, 4.17, 0.0, 0.0, 0.0]]))
    assert controller._launched.item()

    qpos[:, :3] = torch.tensor([[1.0, 2.0, 3.0]])
    qvel[:] = 2.0
    motion.time_remaining[:] = 1.62
    controller.before_step()
    torch.testing.assert_close(qpos[:, :3], torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(qvel, torch.full_like(qvel, 2.0))

    controller.after_step(torch.tensor([True]))
    motion.time_remaining[:] = 2.0
    controller.before_step()
    torch.testing.assert_close(qpos[:, :3], torch.tensor([[21.93, 19.902, 1.2]]))
    torch.testing.assert_close(qvel, torch.zeros_like(qvel))
    assert not controller._launched.item()


def test_incoming_ball_controller_launches_immediately_for_trajectory_deadline() -> (
    None
):
    controller, motion, _, qvel = _make_incoming_ball_controller_stub(
        trajectory_drives_deadline=True
    )
    motion.contact_time[:] = LAUNCH_DISTILL_FLIGHT_TIME_S
    motion.time_remaining[:] = LAUNCH_DISTILL_FLIGHT_TIME_S

    controller.before_step()

    torch.testing.assert_close(qvel, torch.tensor([[-8.5, 0.75, 4.17, 0.0, 0.0, 0.0]]))
    assert controller._launched.item()


def test_incoming_ball_controller_rearms_when_chain_changes_with_same_motion() -> None:
    controller, motion, _, qvel = _make_incoming_ball_controller_stub(
        trajectory_drives_deadline=True
    )
    motion.motion_chain_count = torch.zeros(1, dtype=torch.long)
    controller._previous_motion_chain_count.zero_()
    controller.before_step()
    assert controller._launched.item()

    qvel.zero_()
    motion.motion_chain_count += 1
    controller.before_step()

    torch.testing.assert_close(qvel, torch.tensor([[-8.5, 0.75, 4.17, 0.0, 0.0, 0.0]]))
    assert controller._launched.item()


def test_incoming_ball_controller_bounds_strike_translation() -> None:
    torch.manual_seed(0)
    controller, _, qpos, _ = _make_incoming_ball_controller_stub(
        trajectory_drives_deadline=True,
        position_noise_std=(1.0, 1.0, 0.0),
        max_strike_deviation=0.15,
    )

    controller.before_step()

    translation = qpos[0, :3] - torch.tensor([21.93, 19.902, 1.2])
    assert torch.linalg.vector_norm(translation).item() <= 0.15 + 1.0e-6
    assert translation[2].item() == 0.0


def test_incoming_ball_trajectory_sets_phase_deadline() -> None:
    command = object.__new__(mdp.PhaseAwareMultiTargetMotionCommand)
    command._env = SimpleNamespace(device="cpu")
    command.which_motion = torch.tensor([0, 0], dtype=torch.long)
    command._nominal_contact_times = torch.tensor([3.26])
    command.contact_time = torch.zeros(2)
    command.time_remaining = torch.zeros(2)
    command.strike_time_error_s = torch.zeros(2)
    command.cfg = SimpleNamespace(
        max_contact_speedup=2.0,
        contact_time_step_s=0.1,
        incoming_ball_launch_enabled=True,
        incoming_ball_trajectory_drives_deadline=True,
        incoming_ball_flight_time_s=LAUNCH_DISTILL_FLIGHT_TIME_S,
    )

    command._sample_contact_times(torch.tensor([0, 1], dtype=torch.long))

    expected = torch.full((2,), LAUNCH_DISTILL_FLIGHT_TIME_S)
    torch.testing.assert_close(command.contact_time, expected)
    torch.testing.assert_close(command.time_remaining, expected)
    torch.testing.assert_close(command.strike_time_error_s, -expected)


def test_ball_controller_preserves_only_contact_window_impulses() -> None:
    controller, motion, source_position, _, root_velocity, qvel = (
        _make_ball_controller_stub()
    )

    controller.before_step()
    source_position[:] = torch.tensor([[[0.0, 0.0, 1.0]]])
    root_velocity[:] = torch.tensor([[1.0, 0.0, 0.0]])
    controller.after_step()
    assert not controller._released.item()

    motion.time_remaining[:] = 0.04
    source_position[:] = torch.tensor([[[1.0, 0.0, 1.0]]])
    root_velocity.zero_()
    controller.before_step()
    source_position[:] = torch.tensor([[[0.0, 0.0, 1.0]]])
    root_velocity[:] = torch.tensor([[1.0, 0.0, 0.0]])
    controller.after_step()
    assert controller._released.item()

    qvel[:, :3] = root_velocity
    motion.time_remaining[:] = 0.02
    controller.before_step()
    torch.testing.assert_close(qvel[:, :3], root_velocity)


def test_ball_controller_prediction_releases_only_inside_contact_window() -> None:
    controller, motion, _, source_velocity, _, _ = _make_ball_controller_stub()
    source_velocity[:] = torch.tensor([[[-100.0, 0.0, 0.0]]])

    controller.before_step()
    assert not controller._released.item()

    motion.time_remaining[:] = 0.04
    controller.before_step()
    assert controller._released.item()


def test_post_strike_heading_reward_is_delayed_and_faces_landing_target() -> None:
    command = SimpleNamespace(
        cfg=SimpleNamespace(landing_target_enabled=True),
        target_landing_position_w=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        robot_anchor_pos_w=torch.zeros(3, 3),
        robot_anchor_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3),
        ball_has_been_struck=torch.ones(3, dtype=torch.bool),
        ball_post_strike_elapsed_s=torch.tensor([0.30, 0.30, 0.29]),
        metrics={
            "post_strike_heading_error_degrees": torch.zeros(3),
            "post_strike_heading_active": torch.zeros(3),
        },
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )

    reward = tennis_post_strike_heading_to_landing_target_reward(
        env,
        command_name="motion",
        recovery_delay_s=0.3,
        tolerance_degrees=15.0,
        std_degrees=30.0,
    )

    expected = torch.tensor([1.0, math.exp(-6.25), 0.0])
    torch.testing.assert_close(reward, expected)
    torch.testing.assert_close(
        command.metrics["post_strike_heading_error_degrees"],
        torch.tensor([0.0, 90.0, 0.0]),
    )
    torch.testing.assert_close(
        command.metrics["post_strike_heading_active"],
        torch.tensor([1.0, 1.0, 0.0]),
    )


def test_post_strike_heading_reward_faces_startup_x_without_world_position() -> None:
    half_sqrt_two = math.sqrt(0.5)
    command = SimpleNamespace(
        startup_anchor_yaw_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3),
        robot_anchor_quat_w=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [half_sqrt_two, 0.0, 0.0, half_sqrt_two],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ),
        ball_has_been_struck=torch.ones(3, dtype=torch.bool),
        ball_post_strike_elapsed_s=torch.tensor([0.30, 0.30, 0.29]),
        metrics={
            "post_strike_heading_error_degrees": torch.zeros(3),
            "post_strike_heading_active": torch.zeros(3),
        },
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )

    reward = tennis_post_strike_heading_to_startup_x_reward(
        env,
        command_name="motion",
        recovery_delay_s=0.3,
        tolerance_degrees=15.0,
        std_degrees=30.0,
    )

    torch.testing.assert_close(
        reward,
        torch.tensor([1.0, math.exp(-6.25), 0.0]),
    )
    torch.testing.assert_close(
        command.metrics["post_strike_heading_error_degrees"],
        torch.tensor([0.0, 90.0, 0.0]),
    )


def test_predicted_landing_reward_waits_two_steps_and_checks_net() -> None:
    position = torch.tensor([[0.0, 0.0, 1.0]])
    velocity = torch.tensor([[10.0, 0.0, 3.0]])
    target, _ = ballistic_first_landing_position(
        position,
        velocity,
        gravity_magnitude=9.81,
        ground_contact_height=0.0335,
    )
    command = SimpleNamespace(
        cfg=SimpleNamespace(
            landing_target_enabled=True,
            contact_reward_window_s=0.05,
        ),
        time_remaining=torch.tensor([1.0]),
        ball_has_been_struck=torch.zeros(1, dtype=torch.bool),
        ball_landing_recorded=torch.zeros(1, dtype=torch.bool),
        ball_landing_position_w=torch.zeros(1, 3),
        ball_post_strike_steps=torch.zeros(1, dtype=torch.long),
        ball_previous_linear_velocity_w=torch.zeros(1, 3),
        ball_previous_racket_offset_w=torch.zeros(1, 3),
        ball_strike_state_initialized=torch.zeros(1, dtype=torch.bool),
        ball_hit_reward=torch.zeros(1),
        ball_direction_reward=torch.zeros(1),
        ball_net_clearance_reward=torch.zeros(1),
        ball_out_speed_reward=torch.zeros(1),
        target_landing_position_w=target,
        metrics={
            "error_ball_landing": torch.zeros(1),
            "ball_landing_prediction_valid": torch.zeros(1),
            "ball_net_crossing_height": torch.zeros(1),
            "ball_out_speed": torch.zeros(1),
            "ball_target_projected_speed": torch.zeros(1),
            "ball_direction_error_degrees": torch.zeros(1),
        },
        get_source_pos_w=lambda: position[:, None, :],
    )
    ball = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=position,
            root_link_lin_vel_w=torch.zeros_like(velocity),
        )
    )

    class SceneStub(dict):
        env_origins = torch.zeros(1, 3)

    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
        scene=SceneStub(tennis_ball=ball),
    )
    reward_args = {
        "command_name": "motion",
        "std": 1.0,
        "target_radius": 0.5,
        "prediction_delay_steps": 2,
    }

    # The first sample initializes the incoming-ball baseline and cannot be a strike.
    torch.testing.assert_close(
        tennis_ball_predicted_landing_target_reward(env, **reward_args),
        torch.zeros(1),
    )
    ball.data.root_link_lin_vel_w = velocity
    # A valid impulse outside the timing window must not become a strike or reward.
    torch.testing.assert_close(
        tennis_ball_predicted_landing_target_reward(env, **reward_args),
        torch.zeros(1),
    )
    torch.testing.assert_close(
        tennis_ball_hit_reward(env, command_name="motion"),
        torch.zeros(1),
    )
    assert not command.ball_has_been_struck.item()

    # Re-enter the timing window from the held zero-velocity state, then strike.
    command.time_remaining.zero_()
    ball.data.root_link_lin_vel_w = torch.zeros_like(velocity)
    torch.testing.assert_close(
        tennis_ball_predicted_landing_target_reward(env, **reward_args),
        torch.zeros(1),
    )
    ball.data.root_link_lin_vel_w = velocity
    torch.testing.assert_close(
        tennis_ball_predicted_landing_target_reward(env, **reward_args),
        torch.zeros(1),
    )
    torch.testing.assert_close(
        tennis_ball_hit_reward(env, command_name="motion"),
        torch.ones(1),
    )
    torch.testing.assert_close(
        tennis_ball_hit_reward(env, command_name="motion"),
        torch.zeros(1),
    )
    torch.testing.assert_close(
        tennis_ball_predicted_landing_target_reward(env, **reward_args),
        torch.zeros(1),
    )
    torch.testing.assert_close(
        tennis_ball_predicted_landing_target_reward(env, **reward_args),
        torch.ones(1),
    )
    torch.testing.assert_close(
        tennis_ball_direction_reward(env, command_name="motion"),
        torch.ones(1),
    )
    torch.testing.assert_close(
        tennis_ball_direction_reward(env, command_name="motion"),
        torch.zeros(1),
    )
    torch.testing.assert_close(
        command.metrics["ball_direction_error_degrees"],
        torch.zeros(1),
    )
    assert command.ball_landing_recorded.all()
    torch.testing.assert_close(
        command.metrics["ball_landing_prediction_valid"], torch.ones(1)
    )
    torch.testing.assert_close(command.ball_landing_position_w, target)
    torch.testing.assert_close(
        tennis_ball_net_clearance_reward(env, command_name="motion"),
        torch.ones(1),
    )
    torch.testing.assert_close(
        tennis_ball_net_clearance_reward(env, command_name="motion"),
        torch.zeros(1),
    )
    expected_speed_score = tennis_ball_out_speed_score(
        torch.linalg.vector_norm(velocity, dim=-1),
        target_speed=10.0,
        std=10.0,
    )
    torch.testing.assert_close(
        tennis_ball_out_speed_reward(env, command_name="motion"),
        expected_speed_score,
    )
    torch.testing.assert_close(
        tennis_ball_out_speed_reward(env, command_name="motion"),
        torch.zeros(1),
    )
    torch.testing.assert_close(
        command.metrics["ball_out_speed"],
        torch.linalg.vector_norm(velocity, dim=-1),
    )
    torch.testing.assert_close(
        command.metrics["ball_target_projected_speed"],
        torch.linalg.vector_norm(velocity[:, :2], dim=-1),
    )
    torch.testing.assert_close(
        tennis_ball_predicted_landing_target_reward(env, **reward_args),
        torch.zeros(1),
    )


def test_predicted_landing_reward_rejects_a_net_collision() -> None:
    position = torch.tensor([[0.0, 0.0, 1.0]])
    velocity = torch.tensor([[10.0, 0.0, 0.0]])
    target, _ = ballistic_first_landing_position(
        position,
        velocity,
        gravity_magnitude=9.81,
        ground_contact_height=0.0335,
    )
    command = SimpleNamespace(
        cfg=SimpleNamespace(
            landing_target_enabled=True,
            contact_reward_window_s=0.05,
        ),
        time_remaining=torch.zeros(1),
        ball_has_been_struck=torch.zeros(1, dtype=torch.bool),
        ball_landing_recorded=torch.zeros(1, dtype=torch.bool),
        ball_landing_position_w=torch.zeros(1, 3),
        ball_post_strike_steps=torch.zeros(1, dtype=torch.long),
        ball_previous_linear_velocity_w=torch.zeros(1, 3),
        ball_previous_racket_offset_w=torch.zeros(1, 3),
        ball_strike_state_initialized=torch.zeros(1, dtype=torch.bool),
        ball_hit_reward=torch.zeros(1),
        ball_direction_reward=torch.zeros(1),
        ball_net_clearance_reward=torch.zeros(1),
        ball_out_speed_reward=torch.zeros(1),
        target_landing_position_w=target,
        metrics={
            "error_ball_landing": torch.zeros(1),
            "ball_landing_prediction_valid": torch.zeros(1),
            "ball_net_crossing_height": torch.zeros(1),
            "ball_out_speed": torch.zeros(1),
            "ball_target_projected_speed": torch.zeros(1),
            "ball_direction_error_degrees": torch.zeros(1),
        },
        get_source_pos_w=lambda: position[:, None, :],
    )
    ball = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=position,
            root_link_lin_vel_w=torch.zeros_like(velocity),
        )
    )

    class SceneStub(dict):
        env_origins = torch.zeros(1, 3)

    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
        scene=SceneStub(tennis_ball=ball),
    )
    torch.testing.assert_close(
        tennis_ball_predicted_landing_target_reward(
            env,
            command_name="motion",
            std=1.0,
            target_radius=0.5,
            prediction_delay_steps=0,
        ),
        torch.zeros(1),
    )
    ball.data.root_link_lin_vel_w = velocity
    reward = tennis_ball_predicted_landing_target_reward(
        env,
        command_name="motion",
        std=1.0,
        target_radius=0.5,
        prediction_delay_steps=0,
    )

    torch.testing.assert_close(reward, torch.zeros(1))
    assert command.ball_landing_recorded.all()
    torch.testing.assert_close(
        command.metrics["ball_landing_prediction_valid"], torch.zeros(1)
    )
