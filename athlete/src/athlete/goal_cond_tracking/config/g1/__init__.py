from mjlab.tasks.registry import register_mjlab_task

from athlete.goal_cond_tracking.rl import MotionTrackingOnPolicyRunner
from athlete.goal_cond_tracking.rl.landing_curriculum_runner import LandingCurriculumOnPolicyRunner
from .full_flight import TASK_ID as FULL_FLIGHT_TASK_ID, m14_11_full_flight_env_cfg
from .full_flight_foot_force import (
    TASK_ID as FULL_FLIGHT_FOOT_FORCE_TASK_ID,
    m14_11_full_flight_foot_force_env_cfg,
)
from .full_flight_return_home import (
    TASK_ID as FULL_FLIGHT_RETURN_HOME_TASK_ID,
    m14_11_full_flight_return_home_env_cfg,
)

from .env_cfgs import (
    unitree_g1_multi_target_tracking_env_cfg,
    unitree_g1_phase_acceleration_deadline_root_pos_obs_tracking_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_analytic_strike_distillation_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_distillation_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_ball_launch_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_speed5_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_student_no_global_root_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_distilllinear_env_cfg,
    unitree_g1_phase_acceleration_deadline_tppo_landing_distillation_env_cfg,
    unitree_g1_phase_acceleration_root_pos_obs_tracking_env_cfg,
    unitree_g1_phase_aware_root_pos_obs_tracking_env_cfg,
    unitree_g1_phase_aware_tracking_env_cfg,
    unitree_g1_tennis_launch_distill_env_cfg,
    unitree_g1_tennis_launch_distill_goal_time_env_cfg,
    unitree_g1_tennis_launch_distill_torch_match_200_env_cfg,
    unitree_g1_tennis_launch_distill_warp_match_200_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_legacy_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_racket_ground10_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed20_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_radial20_env_cfg,
    unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_radial20_ball_occlusion_env_cfg,
    unitree_g1_tennis_small_court_intent_transformer_ball_observation_robust_env_cfg,
    unitree_g1_tennis_small_court_intent_transformer_ball_observation_robust_smooth_slow_env_cfg,
    unitree_g1_tennis_small_court_intent_transformer_failure_trajectory05_env_cfg,
    unitree_g1_tennis_small_court_intent_transformer_frozen_intent_epoch3_action_rate020_sweet_speed_scale03_env_cfg,
    unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_env_cfg,
    unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_pelvis_torso_tilt50_env_cfg,
    unitree_g1_tennis_small_court_landing5_pelvis_torso_tilt50_ball_dr150_env_cfg,
    unitree_g1_tennis_small_court_landing_std_curriculum150_env_cfg,
    unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_env_cfg,
    unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_global_root_env_cfg,
    unitree_g1_tennis_small_court_global_root_sweet_fk_env_cfg,
    unitree_g1_tennis_small_court_no_global_root_sweet_fk_env_cfg,
    unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg,
    unitree_g1_tennis_small_court_global_root_actuator_robust_env_cfg,
    unitree_g1_tennis_small_court_global_root_actuator_robust_half_env_cfg,
    unitree_g1_tennis_small_court_relative_sweet_ball_dr225_env_cfg,
    unitree_g1_tennis_small_court_relative_sweet_foot_force_env_cfg,
    unitree_g1_tennis_small_court_intent_transformer_sweet_speed50_racket_ground10_env_cfg,
)
from .rl_cfg import (
    unitree_g1_tracking_ppo_runner_cfg,
    unitree_g1_tracking_tppo_distillation_runner_cfg,
    unitree_g1_tracking_tppo_distilllinear_floor03_runner_cfg,
    unitree_g1_tracking_tppo_distilllinear_floor05_runner_cfg,
    unitree_g1_tracking_tppo_distilllinear_runner_cfg,
    unitree_g1_tracking_tppo_intent_reference_distill_runner_cfg,
    unitree_g1_tracking_tppo_intent_reference_latent_ball_observation_robust_runner_cfg,
    unitree_g1_tracking_tppo_intent_reference_latent_distill_runner_cfg,
    unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill00_runner_cfg,
    unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg,
    unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_runner_cfg,
    unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_runner_cfg,
    unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill01_frozen_intent_epoch3_runner_cfg,
    unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill01_runner_cfg,
    unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill03_runner_cfg,
    unitree_g1_tracking_tppo_intent_transformer_runner_cfg,
    unitree_g1_tracking_tppo_ppo_distill03_runner_cfg,
    unitree_g1_tracking_tppo_ppo_only_runner_cfg,
    unitree_g1_tracking_tppo_pure_distillation_runner_cfg,
)

register_mjlab_task(
    task_id="Mjlab-MultiTarget-Tracking-Flat-Unitree-G1",
    env_cfg=unitree_g1_multi_target_tracking_env_cfg(),
    play_env_cfg=unitree_g1_multi_target_tracking_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-PhaseAware-MultiTarget-Tracking-Flat-Unitree-G1",
    env_cfg=unitree_g1_phase_aware_tracking_env_cfg(),
    play_env_cfg=unitree_g1_phase_aware_tracking_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-PhaseAware-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs",
    env_cfg=unitree_g1_phase_aware_root_pos_obs_tracking_env_cfg(),
    play_env_cfg=unitree_g1_phase_aware_root_pos_obs_tracking_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-PhaseAccel-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs",
    env_cfg=unitree_g1_phase_acceleration_root_pos_obs_tracking_env_cfg(),
    play_env_cfg=unitree_g1_phase_acceleration_root_pos_obs_tracking_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-PhaseAccel-DeadlineProjection-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs",
    env_cfg=unitree_g1_phase_acceleration_deadline_root_pos_obs_tracking_env_cfg(),
    play_env_cfg=unitree_g1_phase_acceleration_deadline_root_pos_obs_tracking_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-PhaseAccel-DeadlineProjection-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs-TPPO-Distillation",
    env_cfg=unitree_g1_phase_acceleration_deadline_tppo_distillation_env_cfg(),
    play_env_cfg=unitree_g1_phase_acceleration_deadline_tppo_distillation_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_distillation_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-PhaseAccel-DeadlineProjection-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs-TPPO-Distillation-Landing",
    env_cfg=unitree_g1_phase_acceleration_deadline_tppo_landing_distillation_env_cfg(),
    play_env_cfg=unitree_g1_phase_acceleration_deadline_tppo_landing_distillation_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_distillation_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-PhaseAccel-DeadlineProjection-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs-TPPO-Distillation-Landing-DistillLinear",
    env_cfg=unitree_g1_phase_acceleration_deadline_tppo_distilllinear_env_cfg(),
    play_env_cfg=unitree_g1_phase_acceleration_deadline_tppo_distilllinear_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_distilllinear_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-PhaseAccel-DeadlineProjection-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs-TPPO-Distillation-Landing-DistillLinear-Alive",
    env_cfg=unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_env_cfg(),
    play_env_cfg=unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_distilllinear_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-PhaseAccel-DeadlineProjection-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs-TPPO-Distillation-Landing-DistillLinear-Alive-Student-No-Global-Root",
    env_cfg=unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_student_no_global_root_env_cfg(),
    play_env_cfg=unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_student_no_global_root_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_distilllinear_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-PhaseAccel-DeadlineProjection-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs-TPPO-Distillation-Landing-DistillLinear-Alive-BallLaunch",
    env_cfg=unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_ball_launch_env_cfg(),
    play_env_cfg=unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_ball_launch_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_distilllinear_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_tppo_distilllinear_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-TorchMatch-200-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_torch_match_200_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_torch_match_200_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_tppo_distilllinear_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpMatch-200-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_match_200_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_match_200_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_tppo_distilllinear_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_distilllinear_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_distilllinear_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-DistillFloor03-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_distilllinear_floor03_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-DistillFloor05-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_distilllinear_floor05_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RacketOrientation-PureDistill-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_pure_distillation_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RacketOrientation-PPOOnly-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_ppo_only_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RacketOrientation-PPOPlusDistill03-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_floor03_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_ppo_distill03_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-PPOOnly-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_ppo_only_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-PPOPlusDistill03-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_ppo_distill03_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-IntentTransformer-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_transformer_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-IntentRefDistill-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_distill_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-IntentRefLatentDistill-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_distill_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-IntentRefLatent-PPOPlusDistill03-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill03_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-IntentRefLatent-PPOPlusDistill01-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill01_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-IntentRefLatent-PPOPlusDistill005-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-IntentRefLatent-PPOPlusDistill005-RacketGround10-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_racket_ground10_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_racket_ground10_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-SweetSpeed50-RacketGround10-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_intent_transformer_sweet_speed50_racket_ground10_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_intent_transformer_sweet_speed50_racket_ground10_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-BallObsRobust-SweetSpeed50-RacketGround10-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_intent_transformer_ball_observation_robust_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_intent_transformer_ball_observation_robust_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ball_observation_robust_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-BallObsRobust-ActionRate008-Speed25-Landing5-SweetSpeed50-RacketGround10-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_intent_transformer_ball_observation_robust_smooth_slow_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_intent_transformer_ball_observation_robust_smooth_slow_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ball_observation_robust_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-FailureTraj05-ActionRate008-Speed25-Landing5-SweetSpeed50-RacketGround10-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_intent_transformer_failure_trajectory05_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_intent_transformer_failure_trajectory05_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-FrozenIntent-SweetSpeed50-RacketGround10-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_intent_transformer_sweet_speed50_racket_ground10_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_intent_transformer_sweet_speed50_racket_ground10_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-FrozenIntent-Epoch3-ActionRate020-SweetSpeed50-Scale03-RacketGround10-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_intent_transformer_frozen_intent_epoch3_action_rate020_sweet_speed_scale03_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_intent_transformer_frozen_intent_epoch3_action_rate020_sweet_speed_scale03_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-FrozenIntent-Epoch3-ActionRate020-SweetSpeed50-Scale03-RacketGround10-Landing5-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-FrozenIntent-Epoch3-ActionRate020-SweetSpeed50-Scale03-RacketGround10-Landing5-PelvisTorsoTilt50-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_pelvis_torso_tilt50_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_pelvis_torso_tilt50_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill01-FrozenIntent-Epoch3-ActionRate020-SweetSpeed50-Scale03-RacketGround10-Landing5-PelvisTorsoTilt50-BallDR150-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_landing5_pelvis_torso_tilt50_ball_dr150_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_landing5_pelvis_torso_tilt50_ball_dr150_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill01_frozen_intent_epoch3_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-FrozenIntent-Epoch3-ActionRate020-SweetSpeed50-Scale03-RacketGround10-Landing5-PelvisTorsoTilt50-LandingStd150-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_landing_std_curriculum150_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_landing_std_curriculum150_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
    runner_cls=LandingCurriculumOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-FrozenIntent-Epoch3-ActionRate020-SweetSpeed50-Scale03-RacketGround10-Landing5-PelvisTorsoTilt50-LandingStd150-TorsoMass10-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
    runner_cls=LandingCurriculumOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-FrozenIntent-Epoch3-ActionRate020-SweetSpeed50-Scale03-RacketGround10-Landing5-PelvisTorsoTilt50-LandingStd150-TorsoMass10-GlobalRoot-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_global_root_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_global_root_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
    runner_cls=LandingCurriculumOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-FrozenIntent-Epoch3-ActionRate020-SweetSpeed50-Scale03-RacketGround10-Landing5-PelvisTorsoTilt50-LandingStd150-TorsoMass10-GlobalRoot-SweetFK-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_global_root_sweet_fk_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_global_root_sweet_fk_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
    runner_cls=LandingCurriculumOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-FrozenIntent-Epoch3-ActionRate020-SweetSpeed50-Scale03-RacketGround10-Landing5-PelvisTorsoTilt50-LandingStd150-TorsoMass10-NoGlobalRoot-SweetFK-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_no_global_root_sweet_fk_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_no_global_root_sweet_fk_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
    runner_cls=LandingCurriculumOnPolicyRunner,
)

for _variant, _factory in (
    ("BallDR225", unitree_g1_tennis_small_court_relative_sweet_ball_dr225_env_cfg),
    ("FootForce20", unitree_g1_tennis_small_court_relative_sweet_foot_force_env_cfg),
):
    register_mjlab_task(
        task_id=f"Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-NoGlobalRoot-RelativeSweet-BallDelay-{_variant}-Unitree-G1",
        env_cfg=_factory(),
        play_env_cfg=_factory(play=True),
        rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
        runner_cls=LandingCurriculumOnPolicyRunner,
    )

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-GlobalRoot-RootLanding-BallDelay30Std15-PDGains20-ActDelay5To15-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_global_root_actuator_robust_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_global_root_actuator_robust_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
    runner_cls=LandingCurriculumOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-GlobalRoot-RootLanding-BallDelay30Std7p5-PDGains10-ActDelay7p5To12p5-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_global_root_actuator_robust_half_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_global_root_actuator_robust_half_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
    runner_cls=LandingCurriculumOnPolicyRunner,
)

from .measurement_env_cfgs import measurement_env_cfg, measurement_runner_cfg

for _history, _noise_label, _noise in ((8, '5', 0.05), (12, '5', 0.05), (8, '7p5', 0.075), (12, '7p5', 0.075)):
    register_mjlab_task(
        task_id=f"Mjlab-Tennis-SmallCourt-M14_11-Measurement-Hist{_history}-Noise{_noise_label}-Unitree-G1",
        env_cfg=measurement_env_cfg(history_steps=_history, position_noise_std=_noise),
        play_env_cfg=measurement_env_cfg(play=True, history_steps=_history, position_noise_std=_noise),
        rl_cfg=measurement_runner_cfg(history_steps=_history),
        runner_cls=LandingCurriculumOnPolicyRunner,
    )

register_mjlab_task(
    task_id=FULL_FLIGHT_TASK_ID,
    env_cfg=m14_11_full_flight_env_cfg(),
    play_env_cfg=m14_11_full_flight_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
    runner_cls=LandingCurriculumOnPolicyRunner,
)

register_mjlab_task(
    task_id=FULL_FLIGHT_RETURN_HOME_TASK_ID,
    env_cfg=m14_11_full_flight_return_home_env_cfg(),
    play_env_cfg=m14_11_full_flight_return_home_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
    runner_cls=LandingCurriculumOnPolicyRunner,
)

register_mjlab_task(
    task_id=FULL_FLIGHT_FOOT_FORCE_TASK_ID,
    env_cfg=m14_11_full_flight_foot_force_env_cfg(),
    play_env_cfg=m14_11_full_flight_foot_force_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
    runner_cls=LandingCurriculumOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-GlobalRoot-RelativeSweet-BallDelay-RootLanding-Unitree-G1",
    env_cfg=unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg(),
    play_env_cfg=unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg(),
    runner_cls=LandingCurriculumOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-IntentRefLatent-PPOPlusDistill005-SweetSpeed20-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed20_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed20_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-IntentRefLatent-PPOPlusDistill005-SweetSpeed50-RacketGround10-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-IntentRefLatent-PPOPlusDistill005-SweetSpeed50-RacketGround10-FailureTraj05-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-IntentRefLatent-PPOPlusDistill005-SweetSpeed50-RacketGround10-FailureTraj05-Omni20-Overhead20-Radial20-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_radial20_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_radial20_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-IntentRefLatent-PPOPlusDistill005-SweetSpeed50-RacketGround10-FailureTraj05-Omni20-Overhead20-Radial20-BallOcclusionMixed-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_radial20_ball_occlusion_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_radial20_ball_occlusion_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ball_observation_robust_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Racket0-RootTilt-IntentRefLatent-PPOPlusDistill00-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill00_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-NoTracking-Legacy131-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_legacy_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_legacy_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_distilllinear_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tennis-Launch-Distill-Goal-Time-Unitree-G1",
    env_cfg=unitree_g1_tennis_launch_distill_goal_time_env_cfg(),
    play_env_cfg=unitree_g1_tennis_launch_distill_goal_time_env_cfg(play=True),
    rl_cfg=unitree_g1_tracking_tppo_distilllinear_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-PhaseAccel-DeadlineProjection-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs-TPPO-Distillation-Landing-DistillLinear-Alive-Speed5Std3",
    env_cfg=unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_speed5_env_cfg(),
    play_env_cfg=unitree_g1_phase_acceleration_deadline_tppo_distilllinear_alive_speed5_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_distilllinear_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-PhaseAccel-DeadlineProjection-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs-TPPO-Distillation-AnalyticStrike",
    env_cfg=unitree_g1_phase_acceleration_deadline_tppo_analytic_strike_distillation_env_cfg(),
    play_env_cfg=unitree_g1_phase_acceleration_deadline_tppo_analytic_strike_distillation_env_cfg(
        play=True
    ),
    rl_cfg=unitree_g1_tracking_tppo_distillation_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
