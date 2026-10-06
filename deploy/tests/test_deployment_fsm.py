"""Tests for the backend-rate DeploymentFSM shared by Sim and Real."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

DEPLOY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_DIR))

from utils.deployment_fsm import DeploymentFSM

JOINT_COUNT = 29


def make_fsm(*, home_duration_s: float = 3.0) -> DeploymentFSM:
    return DeploymentFSM(
        default_joint_position=np.linspace(-0.2, 0.2, JOINT_COUNT),
        home_kp=np.full(JOINT_COUNT, 40.0),
        home_kd=np.full(JOINT_COUNT, 2.0),
        home_duration_s=home_duration_s,
        command_timeout_s=0.1,
        damp_kd=3.0,
    )


def policy_command() -> np.ndarray:
    return np.concatenate(
        (
            np.linspace(-0.3, 0.3, JOINT_COUNT),
            np.zeros(JOINT_COUNT),
            np.full(JOINT_COUNT, 35.0),
            np.full(JOINT_COUNT, 1.5),
            np.zeros(JOINT_COUNT),
        )
    )


class DeploymentFSMTest(unittest.TestCase):
    def test_starts_in_damp(self) -> None:
        fsm = make_fsm()
        state = fsm.step(np.zeros(JOINT_COUNT), now_s=0.0)
        self.assertEqual(state, "damp")
        np.testing.assert_array_equal(fsm.kp, 0.0)
        np.testing.assert_array_equal(fsm.kd, 3.0)

    def test_home_interpolates_at_backend_steps(self) -> None:
        fsm = make_fsm(home_duration_s=2.0)
        start = np.linspace(0.4, -0.4, JOINT_COUNT)
        fsm.request("home")
        self.assertEqual(fsm.step(start, now_s=1.0), "home")
        fsm.step(start, now_s=2.0)
        expected = 0.5 * (start + fsm.default_joint_position)
        np.testing.assert_allclose(fsm.q_target, expected)
        np.testing.assert_allclose(fsm.kp, 20.0)
        self.assertFalse(fsm.home_complete)

        fsm.step(start, now_s=3.0)
        np.testing.assert_allclose(fsm.q_target, fsm.default_joint_position)
        self.assertTrue(fsm.home_complete)

    def test_control_requires_completed_home_and_fresh_policy_command(self) -> None:
        fsm = make_fsm(home_duration_s=1.0)
        position = np.zeros(JOINT_COUNT)
        fsm.request("home")
        fsm.step(position, now_s=0.0)
        fsm.request("control")
        self.assertEqual(fsm.step(position, now_s=0.5), "home")

        fsm.step(position, now_s=1.0)
        fsm.request("control")
        self.assertEqual(fsm.step(position, now_s=1.001), "control")
        np.testing.assert_array_equal(fsm.kp, 0.0)
        np.testing.assert_array_equal(fsm.kd, 3.0)

        command = policy_command()
        fsm.update_policy_command(command, now_s=1.01)
        self.assertEqual(fsm.step(position, now_s=1.011), "control")
        np.testing.assert_allclose(fsm.q_target, command[:JOINT_COUNT])
        np.testing.assert_allclose(fsm.kp, command[2 * JOINT_COUNT : 3 * JOINT_COUNT])

    def test_command_timeout_latches_until_explicit_damp(self) -> None:
        fsm = make_fsm(home_duration_s=0.01)
        position = np.zeros(JOINT_COUNT)
        fsm.request("home")
        fsm.step(position, now_s=0.0)
        fsm.step(position, now_s=0.01)
        fsm.request("control")
        fsm.step(position, now_s=0.011)
        fsm.update_policy_command(policy_command(), now_s=0.012)
        fsm.step(position, now_s=0.013)

        self.assertEqual(fsm.step(position, now_s=0.113), "damp")
        self.assertEqual(fsm.fault_reason, "policy_command_timeout")
        fsm.request("home")
        self.assertEqual(fsm.step(position, now_s=0.114), "damp")

        fsm.request("damp")
        fsm.step(position, now_s=0.115)
        self.assertIsNone(fsm.fault_reason)
        fsm.request("home")
        self.assertEqual(fsm.step(position, now_s=0.116), "home")

    def test_hard_fault_cannot_be_cleared(self) -> None:
        fsm = make_fsm()
        position = np.zeros(JOINT_COUNT)
        fsm.force_damp("tilt_limit", now_s=1.0, hard=True)
        fsm.request("damp")
        fsm.step(position, now_s=1.01)
        fsm.request("home")
        self.assertEqual(fsm.step(position, now_s=1.02), "damp")
        self.assertEqual(fsm.fault_reason, "tilt_limit")

    def test_step_cost_is_well_below_500hz_budget(self) -> None:
        fsm = make_fsm()
        position = np.zeros(JOINT_COUNT)
        samples = 20_000
        start = time.perf_counter()
        for index in range(samples):
            fsm.step(position, now_s=index * 0.002)
        average_s = (time.perf_counter() - start) / samples
        self.assertLess(average_s, 0.0002)


if __name__ == "__main__":
    unittest.main()
