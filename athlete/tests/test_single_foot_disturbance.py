from types import SimpleNamespace

import torch

from athlete.goal_cond_tracking.mdp.disturbances import apply_single_body_impulse


def make_event(num_envs=32):
    class Asset:
        num_bodies = 5

        def __init__(self):
            self.wrench = torch.zeros(num_envs, self.num_bodies, 6)
            self.wrench[:, 0] = 7.0  # An unrelated body's force must survive.

        def write_external_wrench_to_sim(self, forces, torques, env_ids, body_ids):
            assert body_ids == [1, 3]
            # Check every write, not just the final state seen by physics.
            active_bodies = (forces.abs().sum(-1) + torques.abs().sum(-1)) > 0
            assert torch.all(active_bodies.sum(-1) <= 1)
            self.wrench[env_ids[:, None], body_ids, :3] = forces
            self.wrench[env_ids[:, None], body_ids, 3:] = torques

    asset = Asset()
    env = SimpleNamespace(scene={"robot": asset}, num_envs=num_envs, device="cpu", step_dt=.02)
    params = dict(
        asset_cfg=SimpleNamespace(name="robot", body_ids=[1, 3]),
        force_range=(-20., 20.), torque_range=(0., 0.),
        duration_s=(.1, .2), cooldown_s=(2., 4.),
    )
    event = apply_single_body_impulse(SimpleNamespace(params=params), env)
    return event, env, params, asset


def test_single_foot_pulses_never_overlap_and_hold_the_selected_foot():
    with torch.random.fork_rng():
        torch.manual_seed(73)
        event, env, params, asset = make_event()
        selections = torch.zeros(2, dtype=torch.long)
        idle_steps = torch.zeros(env.num_envs, dtype=torch.long)
        for _ in range(650):
            before = asset.wrench[:, [1, 3]].clone()
            was_active = before.abs().sum((-1, -2)) > 0
            idle_steps[~was_active] += 1
            event(env, None, **params)
            after = asset.wrench[:, [1, 3]]
            loaded = after[..., :3].norm(dim=-1) > 0
            active = loaded.any(-1)
            assert torch.all(loaded.sum(-1) <= 1)
            assert after[..., :3].abs().max() <= 20
            assert torch.count_nonzero(after[..., 3:]) == 0
            assert torch.all(asset.wrench[:, 0] == 7)
            assert torch.count_nonzero(asset.wrench[:, [2, 4]]) == 0
            staying = was_active & active
            torch.testing.assert_close(after[staying], before[staying])
            started = (~was_active) & active
            # Starts follow a full cooldown (allow one control-step rounding).
            assert torch.all(idle_steps[started] >= 99)
            assert torch.all(idle_steps[started] <= 201)
            selections += loaded[started].sum(0)
            idle_steps[active] = 0
        assert torch.all(selections > 0), "Both feet must be eligible for selection"


def test_expiry_and_partial_reset_clear_only_the_affected_environment():
    event, env, params, asset = make_event(3)
    event(env, None, **params)
    assert torch.count_nonzero(asset.wrench[:, [1, 3]]) == 0
    event._interval_time_left[:2] = 0
    event(env, None, **params)
    other = asset.wrench[1].clone()
    assert torch.count_nonzero(other[[1, 3]]) > 0
    event.reset(torch.tensor([0]))
    assert torch.count_nonzero(asset.wrench[0, [1, 3]]) == 0
    torch.testing.assert_close(asset.wrench[1], other)
    assert torch.all((event._interval_time_left[[0, 2]] >= 1.96))
    for _ in range(12):
        event(env, None, **params)
    assert torch.count_nonzero(asset.wrench[:, [1, 3]]) == 0
    event._interval_time_left.zero_()
    event(env, None, **params)
    assert torch.all((asset.wrench[:, [1, 3], :3].norm(dim=-1) > 0).sum(-1) == 1)
    event.reset()
    assert torch.count_nonzero(asset.wrench[:, [1, 3]]) == 0
    assert torch.all(asset.wrench[:, 0] == 7)
