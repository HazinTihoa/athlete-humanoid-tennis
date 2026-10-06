import math
from dataclasses import asdict, replace
from typing import Any

import pytest
from mjlab.envs.mdp import dr

from athlete.goal_cond_tracking.config.g1.env_cfgs import (
    unitree_g1_tennis_small_court_landing5_pelvis_torso_tilt50_ball_dr150_env_cfg,
    unitree_g1_tennis_small_court_landing_std_curriculum150_env_cfg,
    unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_env_cfg,
)


def _normalize_config(value: Any) -> Any:
    if callable(value):
        return (value.__module__, value.__qualname__)
    if isinstance(value, dict):
        return {key: _normalize_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_normalize_config(item) for item in value)
    return value


def test_torso_mass10_and_ball_dr150_leave_other_settings_unchanged() -> None:
    baseline = unitree_g1_tennis_small_court_landing_std_curriculum150_env_cfg()
    variant = unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_env_cfg()

    event_names = list(variant.events)
    assert event_names.index("torso_mass_inertia") < event_names.index("base_com")
    event = variant.events.pop("torso_mass_inertia")
    assert event.mode == "startup"
    assert event.func is dr.pseudo_inertia
    assert event.params["asset_cfg"].name == "robot"
    assert event.params["asset_cfg"].body_names == ("torso_link",)
    assert event.params["alpha_range"] == pytest.approx(
        (0.5 * math.log(0.9), 0.5 * math.log(1.1))
    )
    physics_event = "tennis_physics_domain_randomization"
    m14_6 = unitree_g1_tennis_small_court_landing5_pelvis_torso_tilt50_ball_dr150_env_cfg()
    actual = variant.events[physics_event].params["cfg"]
    historical = m14_6.events[physics_event].params["cfg"]
    assert actual.racket_restitution == pytest.approx((0.505, 0.715))
    assert actual.court_restitution == pytest.approx((0.6775, 0.8275))
    assert historical.racket_restitution == pytest.approx((0.54, 0.68))
    assert replace(actual, racket_restitution=historical.racket_restitution) == historical
    variant.events[physics_event].params["cfg"] = baseline.events[physics_event].params["cfg"]
    assert _normalize_config(asdict(variant)) == _normalize_config(asdict(baseline))


def test_torso_mass10_play_keeps_nominal_inertia() -> None:
    baseline = unitree_g1_tennis_small_court_landing_std_curriculum150_env_cfg(
        play=True
    )
    variant = unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_env_cfg(
        play=True
    )
    assert "torso_mass_inertia" not in variant.events
    assert _normalize_config(asdict(variant)) == _normalize_config(asdict(baseline))
