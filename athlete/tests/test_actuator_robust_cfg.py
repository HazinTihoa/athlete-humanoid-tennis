from dataclasses import asdict

import mjlab
import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

from mjlab.utils.buffers.delay_buffer import DelayBuffer
from test_global_root_ablation import normalize
from athlete.goal_cond_tracking.config.g1.env_cfgs import (
    unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg as base,
    unitree_g1_tennis_small_court_global_root_actuator_robust_env_cfg as variant,
    unitree_g1_tennis_small_court_global_root_actuator_robust_half_env_cfg as half,
)


@pytest.mark.parametrize("play", [False, True])
def test_only_requested_changes(play):
    a, b = base(play), variant(play)
    for group, names in (("student", ("ball_position", "ball_linear_velocity")),
                         ("intent_ball_history", ("position", "linear_velocity"))):
        for name in names:
            params = b.observations[group].terms[name].params
            assert (params["delay_mean_s"],params["delay_std_s"],params["delay_max_s"]) == (.03,.015,.06)
            params.update(a.observations[group].terms[name].params)
    if not play:
        event = b.events.pop("actuator_pd_gains")
        assert event.mode == "startup"
        assert event.params["kp_range"] == event.params["kd_range"] == (.8,1.2)
        assert event.params["operation"] == "scale"
        for old, new in zip(a.scene.entities["robot"].articulation.actuators,
                            b.scene.entities["robot"].articulation.actuators):
            assert (new.delay_min_lag,new.delay_max_lag,new.delay_update_period) == (2,6,400)
            assert not new.delay_per_env_phase
            for attr in ("delay_min_lag","delay_max_lag","delay_update_period","delay_hold_prob","delay_per_env_phase"):
                setattr(new,attr,getattr(old,attr))
    assert normalize(asdict(a)) == normalize(asdict(b))


def test_lags_start_in_range_hold_then_resample_and_reset():
    torch.manual_seed(73)
    buffer = DelayBuffer(min_lag=2,max_lag=6,batch_size=1000,device="cpu",
                         update_period=400,per_env_phase=False)
    for step in range(401):
        buffer.append(torch.full((1000,1),float(step)))
        output = buffer.compute().squeeze(-1)
        if step == 0:
            initial = buffer.current_lags.clone()
            assert initial.min() >= 2 and initial.max() <= 6
            assert abs(initial.float().mean().item()-4) < .15
        if 6 <= step < 400:
            torch.testing.assert_close(buffer.current_lags,initial)
            torch.testing.assert_close(output,step-initial.float())
    assert torch.any(buffer.current_lags != initial)
    buffer.reset(torch.tensor([0]))
    buffer.append(torch.full((1000,1),999.))
    output = buffer.compute()
    assert 2 <= buffer.current_lags[0] <= 6
    assert output[0].item() == 999.  # No previous episode history.


@pytest.mark.parametrize("play", [False, True])
def test_half_width_changes_only_new_random_ranges(play):
    a, b = variant(play), half(play)
    for group, names in (("student", ("ball_position", "ball_linear_velocity")),
                         ("intent_ball_history", ("position", "linear_velocity"))):
        for name in names:
            params = b.observations[group].terms[name].params
            assert (params["delay_mean_s"], params["delay_std_s"],
                    params.pop("delay_min_s"), params["delay_max_s"]) == (.03, .0075, .015, .045)
            params.update(a.observations[group].terms[name].params)
    if not play:
        event = b.events["actuator_pd_gains"]
        assert event.params["kp_range"] == event.params["kd_range"] == (.9, 1.1)
        event.params.update(a.events["actuator_pd_gains"].params)
        for actuator in b.scene.entities["robot"].articulation.actuators:
            assert (actuator.delay_min_lag, actuator.delay_max_lag) == (3, 5)
            actuator.delay_min_lag, actuator.delay_max_lag = 2, 6
    assert normalize(asdict(a)) == normalize(asdict(b))


def test_registered_tasks_do_not_share_mutable_actuator_settings():
    prefix = "Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-GlobalRoot-"
    old = load_env_cfg(prefix + "RelativeSweet-BallDelay-RootLanding-Unitree-G1")
    wide_id = prefix + "RootLanding-BallDelay30Std15-PDGains20-ActDelay5To15-Unitree-G1"
    narrow_id = prefix + "RootLanding-BallDelay30Std7p5-PDGains10-ActDelay7p5To12p5-Unitree-G1"
    wide, narrow = load_env_cfg(wide_id), load_env_cfg(narrow_id)
    play = load_env_cfg(wide_id, play=True)
    configs = [(old, (0, 0)), (wide, (2, 6)), (narrow, (3, 5)), (play, (0, 0))]
    for cfg, expected in configs:
        for act in cfg.scene.entities["robot"].articulation.actuators:
            assert (act.delay_min_lag, act.delay_max_lag) == expected
    baseline = base()
    v, h = variant(), half()
    assert v.scene.entities["robot"].articulation.actuators[0] is not h.scene.entities["robot"].articulation.actuators[0]
    assert baseline.scene.entities["robot"].articulation.actuators[0].delay_max_lag == 0
    assert v.scene.entities["robot"].articulation.actuators[0].delay_max_lag == 6
