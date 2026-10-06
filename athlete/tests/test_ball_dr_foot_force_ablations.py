from dataclasses import asdict
from types import SimpleNamespace

import mjlab
import pytest
import torch

from mjlab.envs.mdp.events import apply_body_impulse
from test_global_root_ablation import normalize
from athlete.goal_cond_tracking.config.g1.env_cfgs import (
    unitree_g1_tennis_small_court_no_global_root_sweet_fk_env_cfg as base,
    unitree_g1_tennis_small_court_relative_sweet_ball_dr225_env_cfg as ball,
    unitree_g1_tennis_small_court_relative_sweet_foot_force_env_cfg as feet,
)


@pytest.mark.parametrize("play", [False, True])
def test_ball_changes_only_four_ranges(play):
    a, b = base(play), ball(play)
    if not play:
        event = "tennis_physics_domain_randomization"
        original = a.events[event].params["cfg"]
        sampled = b.events[event].params["cfg"]
        for name in ("ball_mass_kg", "court_restitution",
                     "ground_tangent_speed_retention", "drag_coefficient"):
            low, high = getattr(original, name)
            actual_low, actual_high = getattr(sampled, name)
            assert actual_high - actual_low == pytest.approx(1.5 * (high - low))
            assert actual_high + actual_low == pytest.approx(high + low)
        assert sampled.racket_restitution == original.racket_restitution
        assert 0.5 <= sampled.court_restitution[0] < sampled.court_restitution[1] <= 0.924
        assert 0 < sampled.ground_tangent_speed_retention[0] < sampled.ground_tangent_speed_retention[1] < 1
        b.events[event].params["cfg"] = original
    assert normalize(asdict(a)) == normalize(asdict(b))


@pytest.mark.parametrize("play", [False, True])
def test_feet_changes_only_one_event(play):
    a, b = base(play), feet(play)
    if not play:
        event = b.events.pop("foot_force")
        assert event.func is apply_body_impulse
        assert event.params["force_range"] == (-20., 20.)
        assert event.params["asset_cfg"].body_names == ("left_ankle_roll_link", "right_ankle_roll_link")
    assert normalize(asdict(a)) == normalize(asdict(b))


def test_impulse_is_bounded_expires_and_clears_on_reset():
    class Asset:
        num_bodies = 2
        force = torch.zeros(2, 2, 3)

        def write_external_wrench_to_sim(self, force, torque, env_ids, body_ids):
            assert body_ids == [0, 1]
            assert torch.count_nonzero(torque) == 0
            self.force[env_ids] = force

    asset = Asset()
    env = SimpleNamespace(scene={"robot": asset}, num_envs=2, device="cpu", step_dt=.02)
    params = dict(asset_cfg=SimpleNamespace(name="robot", body_ids=[0, 1]),
                  force_range=(-20., 20.), torque_range=(0., 0.),
                  duration_s=(.1, .1), cooldown_s=(2., 4.))
    event = apply_body_impulse(SimpleNamespace(params=params), env)
    event(env, None, **params)
    assert torch.count_nonzero(asset.force) == 0
    event._interval_time_left.zero_()
    event(env, None, **params)
    assert torch.count_nonzero(asset.force) > 0
    assert asset.force.abs().max() <= 20
    for _ in range(7):
        event(env, None, **params)
    assert torch.count_nonzero(asset.force) == 0
    event._interval_time_left.zero_()
    event(env, None, **params)
    event.reset(torch.tensor([0]))
    assert torch.count_nonzero(asset.force[0]) == 0
    assert torch.count_nonzero(asset.force[1]) > 0
