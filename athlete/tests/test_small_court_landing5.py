from dataclasses import asdict

import pytest

from athlete.goal_cond_tracking.config.g1.env_cfgs import (
    unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_env_cfg,
    unitree_g1_tennis_small_court_intent_transformer_frozen_intent_epoch3_action_rate020_sweet_speed_scale03_env_cfg,
)


@pytest.mark.parametrize("play", [False, True])
def test_landing5_changes_only_goal_and_keeps_m11_9(play):
    original = unitree_g1_tennis_small_court_intent_transformer_frozen_intent_epoch3_action_rate020_sweet_speed_scale03_env_cfg(play=play)
    cfg = unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_env_cfg(play=play)
    before = asdict(original.commands["motion"])
    after = asdict(cfg.commands["motion"])
    assert before["landing_target_mean"] == (6.0, 0.0, 0.0)
    assert after["landing_target_mean"] == (5.0, 0.0, 0.0)
    assert after["landing_target_std"] == (0.0, 0.0, 0.0)
    after["landing_target_mean"] = before["landing_target_mean"]
    assert before == after
    assert cfg.rewards == original.rewards
    assert cfg.observations == original.observations
    assert cfg.terminations == original.terminations
    assert cfg.sim == original.sim
    assert cfg.decimation == original.decimation
