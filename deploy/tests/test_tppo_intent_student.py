"""Contracts for legacy and robust Intent Student observation builders."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "deploy"
sys.path.insert(0, str(DEPLOY_DIR))

from utils.tppo_intent_student import (  # noqa: E402
    INTENT_OBSERVATION_DIM,
    INTENT_ROBUST_OBSERVATION_DIM,
    INTENT_ROBUST_OBSERVATION_SLICES,
    IntentObservationBuilder,
)
from utils.tppo_student import ACTION_DIM, BallState, RobotState  # noqa: E402


class IntentObservationBuilderTest(unittest.TestCase):
    def _states(self) -> tuple[RobotState, BallState]:
        robot = RobotState(
            root_position_w=np.zeros(3),
            root_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            base_angular_velocity_b=np.zeros(3),
            joint_position=np.zeros(ACTION_DIM),
            joint_velocity=np.zeros(ACTION_DIM),
            sweet_spot_velocity_w=np.zeros(3),
        )
        ball = BallState(
            position_w=np.array([2.0, 0.5, 0.8]),
            velocity_w=np.array([-4.0, 0.0, 1.0]),
        )
        return robot, ball

    def _build(self, robust: bool) -> np.ndarray:
        builder = IntentObservationBuilder(
            default_joint_position=np.zeros(ACTION_DIM),
            landing_target_startup=np.array([6.0, 0.0]),
            include_ball_observation_status=robust,
        )
        builder.reset(np.array([1.0, 0.0, 0.0, 0.0]))
        robot, ball = self._states()
        return builder.build_observation(
            robot=robot,
            ball=ball,
            projected_gravity_b=np.array([0.0, 0.0, -1.0]),
            last_action=np.zeros(ACTION_DIM),
            sweet_spot_position_w=np.array([0.2, 0.1, 1.0]),
            ball_observation_valid=False,
            ball_observation_age_s=0.08,
        )

    def test_legacy_contract_stays_628d(self) -> None:
        self.assertEqual(self._build(robust=False).shape, (INTENT_OBSERVATION_DIM,))

    def test_global_xyz_precedes_histories_without_recentring(self) -> None:
        builders = [IntentObservationBuilder(
            default_joint_position=np.zeros(ACTION_DIM),
            landing_target_startup=np.array([5.0, 0.0]),
            include_global_root_pos=global_root,
        ) for global_root in (False, True)]
        robot, ball = self._states()
        robot.root_position_w[:] = [1.0, -2.0, 0.76]
        outputs = []
        for builder in builders:
            builder.reset(np.array([1.0, 0.0, 0.0, 0.0]))
            outputs.append(builder.build_observation(
                robot, ball, np.array([0.0, 0.0, -1.0]),
                np.zeros(ACTION_DIM), np.array([1.2, -1.8, 1.0]),
            ))
        old, new = outputs
        self.assertEqual(new.shape, (631,))
        np.testing.assert_allclose(new[:133], old[:133])
        np.testing.assert_allclose(new[133:136], robot.root_position_w)
        np.testing.assert_allclose(new[136:], old[133:])

    def test_robust_contract_adds_validity_and_age_history(self) -> None:
        observation = self._build(robust=True)
        self.assertEqual(observation.shape, (INTENT_ROBUST_OBSERVATION_DIM,))
        np.testing.assert_array_equal(
            observation[
                INTENT_ROBUST_OBSERVATION_SLICES[
                    "ball_observation_valid_history"
                ]
            ],
            np.zeros(5),
        )
        np.testing.assert_allclose(
            observation[
                INTENT_ROBUST_OBSERVATION_SLICES["ball_observation_age_history"]
            ],
            np.full(5, 0.08),
        )

    def test_long_history_keeps_current_observation_contract(self) -> None:
        builders = [IntentObservationBuilder(
            default_joint_position=np.zeros(ACTION_DIM),
            landing_target_startup=np.array([5.0, 0.0]),
            include_global_root_pos=True,
            buffer_length=steps * 5,
            history_lags=tuple(range((steps - 1) * 5, -1, -5)),
        ) for steps in (5, 8, 12)]
        robot, ball = self._states()
        for builder in builders:
            builder.reset(np.array([1.0, 0.0, 0.0, 0.0]))
        for step in range(70):
            ball.position_w[0] = step * .03
            outputs = [builder.build_observation(
                robot, ball, np.array([0.0, 0.0, -1.0]),
                np.zeros(ACTION_DIM), np.array([.2, .1, 1.0]),
            ) for builder in builders]
            for output, size in zip(outputs, (631, 928, 1324)):
                self.assertEqual(output.shape, (size,))
                np.testing.assert_allclose(output[:136], outputs[0][:136])


if __name__ == "__main__":
    unittest.main()
