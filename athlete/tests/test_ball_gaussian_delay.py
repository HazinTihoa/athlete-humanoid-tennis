from dataclasses import asdict
from types import SimpleNamespace

import mjlab
import pytest
import torch

from athlete.goal_cond_tracking.mdp.ball_delay import (
    GaussianBallDelay, shared_gaussian_ball_delay,
)


def fake_env(n=2):
    data = SimpleNamespace(
        root_link_pos_w=torch.zeros(n, 3), root_link_lin_vel_w=torch.zeros(n, 3)
    )
    return SimpleNamespace(
        step_dt=0.02, num_envs=n, device="cpu", common_step_counter=0,
        scene={"tennis_ball": SimpleNamespace(data=data)},
    )


def test_fractional_delay_cached_and_partial_reset():
    env = fake_env()
    state = GaussianBallDelay(env, "tennis_ball", 0.03, 0, 0.1)
    state.update(env)
    for step in range(1, 6):
        env.common_step_counter = step
        env.scene["tennis_ball"].data.root_link_pos_w[:] = step
        env.scene["tennis_ball"].data.root_link_lin_vel_w[:] = step * 2
        output = state.update(env)
        assert output is state.update(env)
    torch.testing.assert_close(output[0], torch.full((2, 3), 3.5))
    torch.testing.assert_close(output[1], torch.full((2, 3), 7.0))
    env.scene["tennis_ball"].data.root_link_pos_w[0] = 100
    cursor = state.cursor
    state.reset(torch.tensor([0]))
    output = state.update(env)
    assert state.cursor == cursor
    torch.testing.assert_close(output[0][0], torch.full((3,), 100.0))
    torch.testing.assert_close(output[0][1], torch.full((3,), 3.5))


def test_gaussian_distribution_and_episode_hold():
    torch.manual_seed(19)
    env = fake_env(100000)
    state = GaussianBallDelay(env, "tennis_ball", 0.02, 0.01, 0.05)
    state.update(env)
    delays = state.delay_s.clone()
    assert abs(delays.mean().item() - 0.020085) < 0.00015
    assert abs(delays.std().item() - 0.0098) < 0.00015
    assert delays.min() >= 0 and delays.max() <= 0.05
    assert (delays == 0.05).any()
    env.common_step_counter += 1
    state.update(env)
    torch.testing.assert_close(state.delay_s, delays)
    state.reset(torch.tensor([0, 1]))
    state.update(env)
    torch.testing.assert_close(state.delay_s[2:], delays[2:])
    assert not torch.equal(state.delay_s[:2], delays[:2])


def test_student_paths_share_same_state_and_reject_conflicting_delays():
    env = fake_env()
    params = dict(ball_entity_name="tennis_ball", delay_mean_s=0.02,
                  delay_std_s=0.01, delay_max_s=0.1)
    a = shared_gaussian_ball_delay(params, env)
    assert a is shared_gaussian_ball_delay(params, env)
    with pytest.raises(ValueError):
        shared_gaussian_ball_delay({**params, "delay_mean_s": 0.04}, env)
    assert shared_gaussian_ball_delay({}, env) is None


def test_half_range_gaussian_delay_and_minimum_signature():
    torch.manual_seed(21)
    env = fake_env(100000)
    params = dict(ball_entity_name="tennis_ball", delay_mean_s=.03,
                  delay_std_s=.0075, delay_min_s=.015, delay_max_s=.045)
    state = shared_gaussian_ball_delay(params, env)
    state.update(env)
    assert state.delay_s.min() >= .015 and state.delay_s.max() <= .045
    assert abs(state.delay_s.mean().item() - .03) < .0001
    assert abs(state.delay_s.std().item() - .007196) < .0001
    with pytest.raises(ValueError):
        shared_gaussian_ball_delay({**params, "delay_min_s": 0.}, env)
    with pytest.raises(ValueError):
        GaussianBallDelay(env, "tennis_ball", .03, .0075, .045, min_s=.04)


@pytest.mark.parametrize("play", [False, True])
def test_m14_10_changes_only_student_xyz_ball_delay_and_relative_sweet_velocity(play):
    from athlete.goal_cond_tracking.config.g1.env_cfgs import (
        unitree_g1_tennis_small_court_global_root_sweet_fk_env_cfg as base,
        unitree_g1_tennis_small_court_no_global_root_sweet_fk_env_cfg as delayed,
    )
    from test_global_root_ablation import normalize
    a, b = base(play), delayed(play)
    del a.observations["student"].terms["global_root_pos"]
    assert "global_root_pos" not in b.observations["student"].terms
    assert b.observations["critic"].terms["global_root_pos"].noise is None
    for group, terms in (("student", ("ball_position", "ball_linear_velocity")),
                         ("intent_ball_history", ("position", "linear_velocity"))):
        for name in terms:
            term = b.observations[group].terms[name]
            assert term.params.pop("delay_mean_s") == 0.020
            assert term.params.pop("delay_std_s") == 0.010
            assert term.params.pop("delay_max_s") == 0.050
    from athlete.goal_cond_tracking.mdp.observations import (
        tennis_sweet_spot_relative_linear_velocity_b,
    )
    for group in ("student", "critic"):
        term = b.observations[group].terms["sweet_spot_linear_velocity"]
        original = a.observations[group].terms["sweet_spot_linear_velocity"]
        assert term.func is tennis_sweet_spot_relative_linear_velocity_b
        assert term.params == {"command_name": "motion", "source_index": 0}
        assert term.noise == original.noise
        term.func, term.params = original.func, original.params
    assert b.rewards["reference_sweet_spot_velocity_reward"].params.pop("relative_to_pelvis") is True
    assert normalize(asdict(a)) == normalize(asdict(b))


def test_both_history_paths_use_delayed_ball_with_current_root_transform():
    from athlete.goal_cond_tracking.mdp.observations import (
        TennisBallStridedHistory, TennisBallCurrentAnchorStridedHistory,
    )
    env = fake_env()
    command = SimpleNamespace(robot_anchor_pos_w=torch.zeros(2, 3),
                              robot_anchor_quat_w=torch.tensor([[1., 0, 0, 0]] * 2))
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    params = dict(command_name="motion", ball_entity_name="tennis_ball",
                  sample_lags=(1, 0), buffer_length=2,
                  delay_mean_s=0.03, delay_std_s=0., delay_max_s=0.1)
    terms = []
    for cls in (TennisBallStridedHistory, TennisBallCurrentAnchorStridedHistory):
        for quantity in ("position", "linear_velocity"):
            config = SimpleNamespace(params={**params, "quantity": quantity})
            terms.append((cls(config, env), config.params))
    assert all(term.perception is terms[0][0].perception for term, _ in terms)
    for step in range(6):
        env.common_step_counter = step
        env.scene["tennis_ball"].data.root_link_pos_w[:] = step
        env.scene["tennis_ball"].data.root_link_lin_vel_w[:] = step * 2
        results = [term(env, **params) for term, params in terms]
    for k in (0, 2):
        torch.testing.assert_close(results[k][:, -3:], torch.full((2, 3), 3.5))
        torch.testing.assert_close(results[k+1][:, -3:], torch.full((2, 3), 7.))
    command.robot_anchor_pos_w[:] = 1
    # z_S re-expresses every stored world measurement in the current root frame.
    latest = terms[2][0](env, **terms[2][1])
    torch.testing.assert_close(latest[:, -3:], torch.full((2, 3), 2.5))


def test_shared_perception_holds_world_measurement_during_mixed_occlusion():
    from athlete.goal_cond_tracking.mdp.observations import (
        _TennisBallPerceptionState,
    )

    env = fake_env(10000)
    state = _TennisBallPerceptionState(
        env,
        ball_entity_name="tennis_ball",
        latency_min_steps=0,
        latency_max_steps=0,
        dropout_start_probability=1.0,
        dropout_duration_min_steps=1,
        dropout_duration_max_steps=3,
        dropout_duration_modes=((0.25, 1, 1), (0.75, 3, 3)),
        position_noise_std=0.0,
        velocity_noise_std=0.0,
        control_dt_s=0.02,
    )
    torch.manual_seed(14)
    state.update(env)
    sampled_duration = state.dropout_remaining + 1
    assert set(sampled_duration.tolist()) <= {1, 3}
    assert abs((sampled_duration == 1).float().mean().item() - 0.25) < 0.02

    state.dropout_duration_modes = ((1.0, 3, 3),)
    state.dropout_remaining.zero_()
    state.initialized.zero_()
    state.reset()
    env.scene["tennis_ball"].data.root_link_pos_w[:] = 2.0
    state.update(env)
    env.common_step_counter += 1
    env.scene["tennis_ball"].data.root_link_pos_w[:] = 9.0
    held_position, _, valid, age = state.update(env)
    torch.testing.assert_close(held_position, torch.full((10000, 3), 2.0))
    assert not valid.any()
    torch.testing.assert_close(age, torch.full((10000,), 0.02))


def test_radial20_occlusion_comparison_uses_latency_and_four_hold_durations():
    from athlete.goal_cond_tracking.config.g1.env_cfgs import (
        unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_radial20_ball_occlusion_env_cfg,
    )
    from athlete.goal_cond_tracking.config.g1.rl_cfg import (
        unitree_g1_tracking_tppo_intent_reference_latent_ball_observation_robust_runner_cfg,
    )

    cfg = unitree_g1_tennis_launch_distill_warp_root_directed_cache0_200_no_tracking_racket0_root_tilt_intent_transformer_sweet_speed50_racket_ground10_failure_trajectory05_radial20_ball_occlusion_env_cfg()
    params = cfg.observations["intent_ball_history"].terms["position"].params
    assert params["perception_latency_max_steps"] == 3
    assert params["perception_dropout_start_probability"] == 0.01
    assert params["perception_dropout_duration_modes"] == (
        (0.25, 1, 1), (0.50, 2, 5), (0.20, 8, 15), (0.05, 25, 50)
    )
    assert params["position_encoding"] == {
        "linear_radius_m": 10.0, "limit_radius_m": 20.0
    }
    assert "valid" in cfg.observations["intent_ball_history"].terms
    assert "age" in cfg.observations["intent_ball_history"].terms
    assert unitree_g1_tracking_tppo_intent_reference_latent_ball_observation_robust_runner_cfg().actor.ball_token_dim == 8
