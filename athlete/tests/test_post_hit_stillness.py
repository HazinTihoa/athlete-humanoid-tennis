from dataclasses import asdict

import pytest
import torch

from athlete.goal_cond_tracking.config.g1.full_flight import m14_11_full_flight_env_cfg
from athlete.goal_cond_tracking.config.g1.full_flight_return_home import m14_11_full_flight_return_home_env_cfg
from athlete.goal_cond_tracking.mdp.post_hit_stillness import PostHitStillnessReward
from test_full_flight import tracker_fixture, observe


@pytest.mark.parametrize("play", [False, True])
def test_non_return_home_base_adds_only_stillness_reward(play):
    from test_global_root_ablation import normalize
    cfg = m14_11_full_flight_env_cfg(play)
    term = cfg.rewards.pop("post_hit_stillness")
    assert term.func is PostHitStillnessReward and term.weight == 1.
    assert "return_home_speed" not in cfg.rewards
    assert normalize(asdict(cfg)) == normalize(asdict(m14_11_full_flight_env_cfg(play, post_hit_stillness=False)))
    assert "post_hit_stillness" not in m14_11_full_flight_return_home_env_cfg(play).rewards


def setup_reward():
    tracker, command, env = tracker_fixture()
    command.robot_anchor_pos_w[:] = torch.tensor([[1., -2., .8]])
    command.robot_anchor_lin_vel_w = torch.zeros(1, 3)
    env.reset_terminated = torch.zeros(1, dtype=torch.bool)
    cfg = m14_11_full_flight_env_cfg().rewards["post_hit_stillness"]
    return PostHitStillnessReward(cfg, env), tracker, command, env, cfg.params


def test_actual_hit_starts_reward_immediately_without_landing_or_delay():
    reward, tracker, c, env, params = setup_reward()
    assert reward(env, **params).item() == 0
    observe(tracker, [1., 0., 1.], hit=True)
    assert tracker.hit_age.item() == 0
    assert not c.ball_landing_recorded.item()
    torch.testing.assert_close(tracker.hit_root_position_w, torch.tensor([[1., -2., .8]]))
    assert reward(env, **params).item() == 1


def test_speed_and_drift_each_reduce_reward_and_hit_target_never_follows_robot():
    reward, tracker, c, env, params = setup_reward()
    observe(tracker, [1., 0., 1.], hit=True)
    c.robot_anchor_lin_vel_w[0, 0] = params["speed_std_m_s"]
    assert reward(env, **params).item() == pytest.approx(torch.exp(torch.tensor(-1.)).item())
    c.robot_anchor_pos_w[0, 0] += params["position_std_m"]
    # Repeated racket contacts cannot move the original first-hit root target.
    observe(tracker, [1., 0., 1.], hit=True)
    assert tracker.hit_root_position_w[0, 0].item() == 1
    assert reward(env, **params).item() == pytest.approx(torch.exp(torch.tensor(-2.)).item())
    c.robot_anchor_lin_vel_w.zero_()
    assert reward(env, **params).item() == pytest.approx(torch.exp(torch.tensor(-1.)).item())


def test_fall_new_serve_and_unlaunched_state_cannot_keep_previous_hit_credit():
    reward, tracker, c, env, params = setup_reward()
    observe(tracker, [1., 0., 1.], hit=True)
    env.reset_terminated[:] = True
    assert reward(env, **params).item() == 0
    env.reset_terminated[:] = False
    tracker.active[:] = False
    assert reward(env, **params).item() == 0
    tracker.reset(torch.tensor([0]))
    tracker.active[:] = True
    assert reward(env, **params).item() == 0
    assert not torch.count_nonzero(tracker.hit_root_position_w)
    c.robot_anchor_pos_w[0, 0] = 3.
    observe(tracker, [1., 0., 1.], hit=True)
    assert reward(env, **params).item() == 1
    assert tracker.hit_root_position_w[0, 0].item() == 3


def test_world_translation_does_not_change_reward_and_z_is_unconstrained():
    reward, tracker, c, env, params = setup_reward()
    observe(tracker, [1., 0., 1.], hit=True)
    offset = torch.tensor([[100., -60., 0.]])
    c.robot_anchor_pos_w += offset
    tracker.hit_root_position_w += offset
    c.robot_anchor_pos_w[:, 2] += .05
    c.robot_anchor_lin_vel_w[:, 2] = .3
    assert reward(env, **params).item() == 1


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_reward_scales_rejected(value):
    _, _, _, env, _ = setup_reward()
    cfg = m14_11_full_flight_env_cfg().rewards["post_hit_stillness"]
    cfg.params["speed_std_m_s"] = value
    with pytest.raises(ValueError):
        PostHitStillnessReward(cfg, env)
