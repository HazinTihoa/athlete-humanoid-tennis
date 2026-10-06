from dataclasses import asdict
from types import SimpleNamespace

import mjlab
import pytest
import torch

from athlete.goal_cond_tracking.mdp.observations import tennis_landing_target_b


def test_target_world_position_stays_fixed_but_observation_follows_current_root():
    command = SimpleNamespace(
        cfg=SimpleNamespace(landing_target_enabled=True, landing_target_frame="startup"),
        target_landing_position_startup=torch.tensor([[5., 0, 0]]),
        target_landing_position_w=torch.tensor([[5., 0, 0]]),
        robot_anchor_pos_w=torch.tensor([[1., 0, 1.]]),
        robot_anchor_quat_w=torch.tensor([[1., 0, 0, 0]]),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    torch.testing.assert_close(tennis_landing_target_b(env, "motion"), torch.tensor([[5., 0]]))
    torch.testing.assert_close(tennis_landing_target_b(env, "motion", True), torch.tensor([[4., 0]]))
    command.robot_anchor_quat_w[:] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
    torch.testing.assert_close(tennis_landing_target_b(env, "motion", True), torch.tensor([[0., -4.]]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(command.target_landing_position_w, torch.tensor([[5., 0, 0]]))


@pytest.mark.parametrize("play", [False, True])
def test_ablation_only_adds_actor_xyz_and_changes_two_landing_observations(play):
    from athlete.goal_cond_tracking.config.g1.env_cfgs import (
        unitree_g1_tennis_small_court_no_global_root_sweet_fk_env_cfg as base,
        unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg as variant,
    )
    from test_global_root_ablation import normalize
    a, b = base(play), variant(play)
    term = b.observations["student"].terms.pop("global_root_pos")
    assert term.noise is None if play else term.noise.std == 0.05
    for group in ("student", "critic"):
        assert b.observations[group].terms["landing_target"].params.pop("current_root_frame") is True
    assert normalize(asdict(a)) == normalize(asdict(b))
