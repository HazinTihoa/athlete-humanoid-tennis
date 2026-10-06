import importlib.util
from dataclasses import asdict
from pathlib import Path

import mjlab
import torch
import pytest

from athlete.goal_cond_tracking.config.g1.env_cfgs import (
    unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_env_cfg,
    unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_global_root_env_cfg,
)


def normalize(value):
    if callable(value):
        return value.__module__, value.__qualname__
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(normalize(item) for item in value)
    return value


@pytest.mark.parametrize("play", [False, True])
def test_global_root_is_only_actor_addition_and_critic_keeps_clean_xyz(play):
    baseline = unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_env_cfg(play)
    variant = unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_global_root_env_cfg(play)
    actor = variant.observations["student"]
    assert list(actor.terms)[-1] == "global_root_pos"
    term = actor.terms.pop("global_root_pos")
    assert term.noise is None if play else term.noise.std == 0.05
    assert variant.commands["motion"].anchor_body_name == "pelvis"
    assert variant.observations["critic"].terms["global_root_pos"].noise is None
    assert normalize(asdict(variant)) == normalize(asdict(baseline))


def test_input_migration_preserves_weights_normalization_and_optimizer():
    path = Path(__file__).resolve().parents[2] / "scripts/expand_intent_global_root_checkpoint.py"
    spec = importlib.util.spec_from_file_location("migrate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(42)
    w = torch.randn(512, 261)
    source = {
        "actor_state_dict": {
            "mlp.0.weight": w,
            "obs_normalizer._mean": torch.randn(1, 133),
            "obs_normalizer._var": torch.ones(1, 133),
            "obs_normalizer._std": torch.ones(1, 133),
        },
        "optimizer_state_dict": {"state": {0: {"exp_avg": w.clone(), "exp_avg_sq": w.square()}}},
    }
    result = module.expand_checkpoint(source)
    expanded_w = result["actor_state_dict"]["mlp.0.weight"]
    x = torch.randn(8, 261)
    x_new = torch.cat((x[:, :133], torch.randn(8, 3), x[:, 133:]), dim=-1)
    torch.testing.assert_close(x @ w.T, x_new @ expanded_w.T, atol=2e-5, rtol=2e-5)
    assert source["actor_state_dict"]["mlp.0.weight"].shape == (512, 261)
    assert expanded_w[:, 133:136].count_nonzero() == 0
    torch.testing.assert_close(expanded_w[:, 136:], w[:, 133:])
    assert result["optimizer_state_dict"]["state"][0]["exp_avg"].shape == expanded_w.shape
