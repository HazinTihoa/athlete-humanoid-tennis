import json
import math
import tempfile
import unittest
from pathlib import Path

import mjlab
import numpy as np
import torch

from athlete.goal_cond_tracking.mdp.commands import (
    MotionCfg,
    MotionGoalCfg,
    reference_frame0_alignment,
    transform_reference_frame0_target,
)
from athlete.motion_sets.motion_set import MotionSet


class ReferenceFrame0TargetTest(unittest.TestCase):
    def test_target_is_reconstructed_directly_from_frame0(self) -> None:
        target_local = torch.tensor([[0.4, -1.1, 0.7]])
        reference_pos_0 = torch.tensor([[2.0, -3.0, 0.8]])
        reference_quat_0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        actual = transform_reference_frame0_target(
            target_local,
            reference_pos_0,
            reference_quat_0,
        )

        torch.testing.assert_close(actual, reference_pos_0 + target_local)

    def test_frame0_alignment_uses_robot_position_and_yaw(self) -> None:
        reference_yaw = 0.8
        robot_yaw = -0.4
        reference_quat_0 = torch.tensor(
            [[
                math.cos(reference_yaw / 2.0),
                0.0,
                0.0,
                math.sin(reference_yaw / 2.0),
            ]]
        )
        robot_pos = torch.tensor([[3.0, -2.0, 0.82]])
        robot_quat = torch.tensor(
            [[
                math.cos(robot_yaw / 2.0),
                0.0,
                0.0,
                math.sin(robot_yaw / 2.0),
            ]]
        )

        aligned_pos, align_yaw = reference_frame0_alignment(
            reference_quat_0,
            robot_pos,
            robot_quat,
        )

        expected_delta = robot_yaw - reference_yaw
        expected_yaw = torch.tensor(
            [[
                math.cos(expected_delta / 2.0),
                0.0,
                0.0,
                math.sin(expected_delta / 2.0),
            ]]
        )
        torch.testing.assert_close(aligned_pos, robot_pos)
        torch.testing.assert_close(align_yaw, expected_yaw)

    def test_target_uses_frame0_anchor_rotation(self) -> None:
        yaw = 0.5
        target_local = torch.tensor([[1.2, -0.3, 0.6]])
        reference_pos_0 = torch.tensor([[0.5, -0.4, 0.8]])
        reference_quat_0 = torch.tensor(
            [[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]]
        )

        actual = transform_reference_frame0_target(
            target_local,
            reference_pos_0,
            reference_quat_0,
        )

        local = target_local[0]
        expected = torch.tensor(
            [[
                reference_pos_0[0, 0]
                + math.cos(yaw) * local[0]
                - math.sin(yaw) * local[1],
                reference_pos_0[0, 1]
                + math.sin(yaw) * local[0]
                + math.cos(yaw) * local[1],
                reference_pos_0[0, 2] + local[2],
            ]]
        )
        torch.testing.assert_close(actual, expected)

    def test_dataset_ball_local_is_marked_as_frame0(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            motion_file = root / "ep_0000.npz"
            np.savez(motion_file, joint_pos=np.zeros((2, 1), dtype=np.float32))
            motion_file.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "strike_frame": 1,
                        "ball_local": [0.2, -0.4, 0.8],
                        "clip": "fh_example.npz",
                    }
                )
            )
            config = root / "motion.toml"
            config.write_text(
                f"""
[registry]
train_prefix = ""

[dataset]
glob = "{motion_file}"
forehand_template = "forehand"
backhand_template = "backhand"
target_pos_std = {{ x = 0.2, y = 0.2, z = 0.2 }}
"""
            )
            library = {
                "forehand": MotionCfg(
                    name="forehand",
                    sub_targets=[
                        MotionGoalCfg(goal_type="position", source_link="target")
                    ],
                ),
                "backhand": MotionCfg(
                    name="backhand",
                    sub_targets=[
                        MotionGoalCfg(goal_type="position", source_link="target")
                    ],
                ),
            }

            motion_cfg = MotionSet.from_toml(config).local_motion_cfgs(library)[0]
            position = motion_cfg.sub_targets[0]

            self.assertEqual(position.target_pos_frame, "reference_anchor_frame0")
            self.assertEqual(
                position.target_pos_mean,
                {"x": 0.2, "y": -0.4, "z": 0.8},
            )
            self.assertEqual(
                position.target_pos_std,
                {"x": 0.2, "y": 0.2, "z": 0.2},
            )

if __name__ == "__main__":
    unittest.main()
