from dataclasses import asdict

from test_global_root_ablation import normalize

from athlete.goal_cond_tracking.config.g1.full_flight import m14_11_full_flight_env_cfg
from athlete.goal_cond_tracking.config.g1.full_flight_foot_force import (
    m14_11_full_flight_foot_force_env_cfg,
)
from athlete.goal_cond_tracking.mdp.disturbances import apply_single_body_impulse


def test_foot_force_is_the_only_training_environment_change():
    base = m14_11_full_flight_env_cfg()
    variant = m14_11_full_flight_foot_force_env_cfg()
    force = variant.events.pop("foot_force")
    assert normalize(asdict(variant)) == normalize(asdict(base))
    assert force.func is apply_single_body_impulse
    assert force.mode == "step"
    assert force.params["asset_cfg"].body_names == (
        "left_ankle_roll_link", "right_ankle_roll_link"
    )
    assert force.params["force_range"] == (-20.0, 20.0)
    assert force.params["torque_range"] == (0.0, 0.0)
    assert force.params["duration_s"] == (0.1, 0.2)
    assert force.params["cooldown_s"] == (2.0, 4.0)


def test_play_retains_unperturbed_full_flight_evaluation():
    assert normalize(asdict(m14_11_full_flight_foot_force_env_cfg(play=True))) == normalize(
        asdict(m14_11_full_flight_env_cfg(play=True))
    )
