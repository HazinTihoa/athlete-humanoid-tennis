from types import SimpleNamespace

import pytest
import torch

from athlete.goal_cond_tracking.config.g1.env_cfgs import (
    unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_env_cfg,
    unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_pelvis_torso_tilt50_env_cfg,
)
from athlete.goal_cond_tracking.mdp.terminations import (
    bad_pelvis_and_torso_tilt,
)


def _quat(axis: str, degrees: float) -> torch.Tensor:
  radians = torch.deg2rad(torch.tensor(degrees)) / 2.0
  quat = torch.tensor([torch.cos(radians), 0.0, 0.0, 0.0])
  quat[{"x": 1, "y": 2, "z": 3}[axis]] = torch.sin(radians)
  return quat


def _env(pelvis: torch.Tensor, torso: torch.Tensor) -> SimpleNamespace:
  robot = SimpleNamespace(
    body_names=("pelvis", "torso_link"),
    data=SimpleNamespace(
      body_link_quat_w=torch.stack([pelvis, torso]).unsqueeze(0),
      gravity_vec_w=torch.tensor([[0.0, 0.0, -1.0]]),
    ),
  )
  command = SimpleNamespace(robot=robot)
  return SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda _: command),
  )


@pytest.mark.parametrize(
  ("pelvis_degrees", "torso_degrees", "expected"),
  [
    (49.9, 80.0, False),
    (80.0, 49.9, False),
    (50.0, 50.0, False),
    (50.1, 50.1, True),
    (90.0, 90.0, True),
  ],
)
def test_pelvis_and_torso_tilt_requires_both_bodies(
  pelvis_degrees: float, torso_degrees: float, expected: bool
) -> None:
  result = bad_pelvis_and_torso_tilt(
    _env(_quat("x", pelvis_degrees), _quat("y", torso_degrees)),
    command_name="motion",
    max_tilt_degrees=50.0,
  )
  assert bool(result.item()) is expected


def test_pelvis_and_torso_tilt_ignores_yaw() -> None:
  result = bad_pelvis_and_torso_tilt(
    _env(_quat("z", 180.0), _quat("z", 90.0)),
    command_name="motion",
    max_tilt_degrees=50.0,
  )
  assert bool(result.item()) is False


def test_pelvis_torso_tilt_task_changes_only_tilt_termination() -> None:
  baseline = unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_env_cfg()
  variant = unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_pelvis_torso_tilt50_env_cfg()
  assert baseline.terminations is not None
  assert variant.terminations is not None
  assert set(variant.terminations) == (set(baseline.terminations) - {"root_tilt"}) | {
    "pelvis_torso_tilt"
  }
  assert variant.terminations["pelvis_torso_tilt"].func is bad_pelvis_and_torso_tilt
  assert variant.terminations["pelvis_torso_tilt"].params == {
    "command_name": "motion",
    "max_tilt_degrees": 50.0,
    "pelvis_body_name": "pelvis",
    "torso_body_name": "torso_link",
  }
