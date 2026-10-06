from dataclasses import asdict

import pytest

from athlete.goal_cond_tracking.config.g1.env_cfgs import (
    unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_pelvis_torso_tilt50_env_cfg,
    unitree_g1_tennis_small_court_landing5_pelvis_torso_tilt50_ball_dr150_env_cfg,
)
from athlete.goal_cond_tracking.config.g1.rl_cfg import (
    unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill01_frozen_intent_epoch3_runner_cfg,
)


def _physics_dr(cfg):
    return cfg.events["tennis_physics_domain_randomization"].params["cfg"]


def test_ball_dr150_expands_only_ball_physics_ranges() -> None:
    baseline = unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_pelvis_torso_tilt50_env_cfg()
    variant = unitree_g1_tennis_small_court_landing5_pelvis_torso_tilt50_ball_dr150_env_cfg()
    before = asdict(_physics_dr(baseline))
    after = asdict(_physics_dr(variant))

    assert after.pop("ball_mass_kg") == pytest.approx((0.05515, 0.06025))
    assert after.pop("court_restitution") == pytest.approx((0.6775, 0.8275))
    assert after.pop("ground_tangent_speed_retention") == pytest.approx(
        (0.7125, 0.9375)
    )
    assert after.pop("drag_coefficient") == pytest.approx((0.475, 0.7))
    for name in (
        "ball_mass_kg",
        "court_restitution",
        "ground_tangent_speed_retention",
        "drag_coefficient",
    ):
        before.pop(name)
    assert after == before


def test_ball_dr150_play_keeps_nominal_physics() -> None:
    baseline = unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_pelvis_torso_tilt50_env_cfg(
        play=True
    )
    variant = unitree_g1_tennis_small_court_landing5_pelvis_torso_tilt50_ball_dr150_env_cfg(
        play=True
    )
    assert _physics_dr(variant) == _physics_dr(baseline)


def test_distill01_frozen_intent_epoch3_runner() -> None:
    cfg = unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill01_frozen_intent_epoch3_runner_cfg()
    assert cfg.algorithm.using_ppo is True
    assert cfg.algorithm.teacher_act_prob == 0.0
    assert cfg.algorithm.distillation_loss_coef == pytest.approx(0.1)
    assert cfg.algorithm.distillation_min_coef == pytest.approx(0.1)
    assert cfg.algorithm.distillation_update_times_scale is None
    assert cfg.algorithm.freeze_intent_encoders is True
    assert cfg.algorithm.num_learning_epochs == 3
    assert cfg.algorithm.intent_reference_loss_coef == 0.0
    assert cfg.algorithm.intent_teacher_action_loss_coef == 0.0
    assert cfg.algorithm.intent_latent_loss_coef == 0.0
