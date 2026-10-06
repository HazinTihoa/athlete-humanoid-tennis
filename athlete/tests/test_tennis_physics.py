"""Tests for the shared tennis-ball physics contract."""

from __future__ import annotations

import numpy as np
import torch
from athlete.scripts.tennis_physics import (
  STANDARD_TENNIS_PHYSICS,
  contact_damping_for_restitution,
  tennis_ball_aerodynamic_wrench_numpy,
  tennis_ball_aerodynamic_wrench_torch,
)


def test_standard_ball_is_within_itf_type_two_mass_and_diameter_ranges() -> None:
  ball = STANDARD_TENNIS_PHYSICS.ball

  assert 0.0560 <= ball.mass_kg <= 0.0594
  assert 0.0654 <= ball.diameter_m <= 0.0686


def test_default_restitution_uses_the_calibrated_hard_court_mapping() -> None:
  damping = contact_damping_for_restitution(
    STANDARD_TENNIS_PHYSICS.court.restitution
  )

  assert np.isclose(damping, 0.0745, atol=1.0e-4)


def test_numpy_and_torch_aerodynamic_wrenches_match() -> None:
  linear_velocity = np.array([[8.0, -2.0, 1.5], [-3.0, 4.0, -0.5]])
  angular_velocity = np.array([[5.0, 80.0, -10.0], [10.0, -30.0, 25.0]])
  numpy_force, numpy_torque = tennis_ball_aerodynamic_wrench_numpy(
    linear_velocity, angular_velocity
  )
  torch_force, torch_torque = tennis_ball_aerodynamic_wrench_torch(
    torch.tensor(linear_velocity, dtype=torch.float64),
    torch.tensor(angular_velocity, dtype=torch.float64),
  )

  np.testing.assert_allclose(torch_force.numpy(), numpy_force)
  np.testing.assert_allclose(torch_torque.numpy(), numpy_torque)


def test_aerodynamic_drag_and_torque_oppose_motion() -> None:
  force, torque = tennis_ball_aerodynamic_wrench_numpy(
    np.array([8.0, 0.0, 0.0]),
    np.array([0.0, 80.0, 0.0]),
  )

  assert force[0] < 0.0
  assert force[1] == 0.0
  assert force[2] == 0.0
  assert torque[1] < 0.0


def test_stationary_ball_has_no_aerodynamic_wrench() -> None:
  force, torque = tennis_ball_aerodynamic_wrench_numpy(
    np.zeros(3), np.zeros(3)
  )

  np.testing.assert_array_equal(force, np.zeros(3))
  np.testing.assert_array_equal(torque, np.zeros(3))
