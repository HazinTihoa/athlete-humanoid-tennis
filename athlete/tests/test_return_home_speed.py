from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from athlete.goal_cond_tracking.config.g1.full_flight import m14_11_full_flight_env_cfg
from athlete.goal_cond_tracking.config.g1.full_flight_return_home import m14_11_full_flight_return_home_env_cfg
from athlete.goal_cond_tracking.mdp.return_home_speed import ReturnHomeSpeedReward


@pytest.mark.parametrize("play", [False, True])
def test_return_home_preserves_its_reward_and_single_foot_disturbance(play):
    from test_global_root_ablation import normalize
    base = m14_11_full_flight_env_cfg(play, post_hit_stillness=False)
    cfg = m14_11_full_flight_return_home_env_cfg(play)
    term = cfg.rewards.pop("return_home_speed")
    assert "post_hit_stillness" not in cfg.rewards
    assert term.func is ReturnHomeSpeedReward and term.weight == 1.0
    assert "foot_force" not in base.events
    if not play:
        from athlete.goal_cond_tracking.mdp.disturbances import apply_single_body_impulse

        assert cfg.episode_length_s == 10.0
        assert base.episode_length_s == 20.0
        base.episode_length_s = 10.0
        for name, baseline_std, comparison_std in (
            ("sweet_spot_position", 0.05, 0.10),
            ("sweet_spot_linear_velocity", 0.20, 0.40),
        ):
            assert cfg.observations["student"].terms[name].noise.std == comparison_std
            assert base.observations["student"].terms[name].noise.std == baseline_std
            base.observations["student"].terms[name].noise.std = comparison_std
        event = cfg.events.pop("foot_force")
        assert event.func is apply_single_body_impulse
        assert event.mode == "step"
        assert event.params["asset_cfg"].name == "robot"
        assert event.params["asset_cfg"].body_names == (
            "left_ankle_roll_link", "right_ankle_roll_link",
        )
        assert event.params["force_range"] == (-20.0, 20.0)
        assert event.params["torque_range"] == (0.0, 0.0)
        assert event.params["duration_s"] == (0.1, 0.2)
        assert event.params["cooldown_s"] == (2.0, 4.0)
    assert normalize(asdict(cfg)) == normalize(asdict(base))


def setup_reward(n=1):
    command = SimpleNamespace(
        motion_chain_count=torch.ones(n, dtype=torch.long), metrics={},
        robot_anchor_pos_w=torch.tensor([[1.25, 0., .8]]).repeat(n, 1),
        startup_anchor_pos_w=torch.tensor([[0., 0., .8]]).repeat(n, 1),
        robot_anchor_lin_vel_w=torch.zeros(n, 3),
        startup_anchor_yaw_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
        ball_has_been_struck=torch.ones(n, dtype=torch.bool),
        trajectory_match_valid=torch.ones(n, dtype=torch.bool),
        phase_hold=torch.zeros(n, dtype=torch.bool),
        strike_time_error_s=torch.zeros(n), failure_trajectory_time_remaining=torch.ones(n),
    )
    ball = torch.tensor([[4., 0., .8]]).repeat(n, 1)
    tracker = SimpleNamespace(active=torch.ones(n, dtype=torch.bool),
                              hit_age=torch.full((n,), .3), ball_position=lambda: ball)
    env = SimpleNamespace(num_envs=n, device="cpu", step_dt=.02,
                          command_manager=SimpleNamespace(get_term=lambda _: command),
                          tennis_full_flight=tracker, reset_terminated=torch.zeros(n, dtype=torch.bool))
    cfg = m14_11_full_flight_return_home_env_cfg().rewards["return_home_speed"]
    reward = ReturnHomeSpeedReward(cfg, env)
    return reward, env, command, cfg.params, ball


def test_faster_recovery_has_higher_integrated_reward_for_same_path():
    totals = []
    for steps in (25, 50, 100):
        term, env, c, params, _ = setup_reward()
        assert term(env, **params).item() == 0
        total = 0
        for i in range(1, steps + 1):
            c.robot_anchor_pos_w[:, 0] = 1.25 - i / steps
            c.robot_anchor_lin_vel_w[:, 0] = -1 / (steps * env.step_dt) if i < steps else 0
            total += term(env, **params).item() * env.step_dt
        assert term.arrived.item()
        assert term.arrival_time.item() == pytest.approx(steps * env.step_dt, abs=1e-5)
        assert term(env, **params).item() == 0  # Only one arrival payment.
        totals.append(total)
    assert totals[0] > totals[1] > totals[2] > 0


def test_retreat_cannot_farm_progress_and_arrival_requires_slow_speed():
    term, env, c, params, _ = setup_reward()
    term(env, **params)
    c.robot_anchor_pos_w[:, 0] = .75
    assert term(env, **params).item() > 0
    c.robot_anchor_pos_w[:, 0] = 1.0
    assert term(env, **params).item() == 0
    c.robot_anchor_pos_w[:, 0] = .75
    assert term(env, **params).item() == 0
    c.robot_anchor_pos_w[:, 0] = .2
    c.robot_anchor_lin_vel_w[:, 0] = 1.0
    assert term(env, **params).item() > 0  # New inward progress only.
    assert not term.arrived.item()
    assert term(env, **params).item() == 0
    c.robot_anchor_lin_vel_w.zero_()
    assert term(env, **params).item() > 0
    assert term.arrived.item()


def test_starting_at_home_never_rewards_leaving_then_returning():
    term, env, c, params, _ = setup_reward()
    c.robot_anchor_pos_w[:, 0] = .1
    term(env, **params)
    for x in (1.25, .5, .1):
        c.robot_anchor_pos_w[:, 0] = x
        assert term(env, **params).item() == 0
    assert not term.eligible.item()


@pytest.mark.parametrize("reason", ["hit", "miss_behind", "miss_timeout", "failure", "failed_but_hit"])
def test_recovery_starts_only_after_rally_gate(reason):
    term, env, c, params, ball = setup_reward()
    c.ball_has_been_struck[:] = False
    env.tennis_full_flight.hit_age.zero_()
    if reason.startswith("miss"):
        c.strike_time_error_s[:] = .29
        if reason == "miss_behind":
            ball[:, 0] = 0
    elif reason in ("failure", "failed_but_hit"):
        c.trajectory_match_valid[:] = False
    if reason in ("hit", "failed_but_hit"):
        c.ball_has_been_struck[:] = True
        env.tennis_full_flight.hit_age[:] = .29
    term(env, **params)
    assert not term.started.item()
    if reason in ("hit", "failed_but_hit"):
        env.tennis_full_flight.hit_age[:] = .3
    elif reason == "miss_behind":
        c.strike_time_error_s[:] = .3
    elif reason == "miss_timeout":
        c.strike_time_error_s[:] = .8
    else:
        c.failure_trajectory_time_remaining.zero_()
    term(env, **params)
    assert term.started.item()
    assert term.reason.item() == {"hit": 1, "failed_but_hit": 1, "miss_behind": 2,
                                 "miss_timeout": 2, "failure": 3}[reason]


def test_new_serve_partial_reset_and_fall_do_not_leak_reward():
    term, env, c, params, _ = setup_reward(2)
    term(env, **params)
    c.robot_anchor_pos_w[:, 0] = .75
    env.reset_terminated[0] = True
    result = term(env, **params)
    assert result[0] == 0 and result[1] > 0
    term.reset(torch.tensor([0]))
    assert not term.started[0] and term.started[1]
    env.reset_terminated.zero_()
    c.motion_chain_count[1] += 1
    c.ball_has_been_struck[:] = False
    c.strike_time_error_s.zero_()
    c.robot_anchor_pos_w[:, 0] = .1
    assert torch.count_nonzero(term(env, **params)) == 0
    assert not term.started.any()
    assert c.startup_anchor_pos_w[:, 0].eq(0).all()


def test_unlaunched_ball_never_starts_recovery():
    term, env, _, params, _ = setup_reward()
    env.tennis_full_flight.active[:] = False
    assert term(env, **params).item() == 0 and not term.started.item()
