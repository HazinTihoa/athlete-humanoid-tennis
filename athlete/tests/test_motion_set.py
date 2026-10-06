import json
from pathlib import Path

import numpy as np

from athlete.motion_sets.motion_set import MotionSet


def test_dataset_can_disable_velocity_and_orientation_sampling(
    tmp_path: Path,
) -> None:
    motion_path = tmp_path / "ep_0000.npz"
    np.savez(motion_path, joint_pos=np.zeros((3, 29), dtype=np.float32))
    motion_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "strike_frame": 1,
                "ball_local": [0.1, -0.2, 0.3],
                "clip": "fh_test_g1.npz",
                "sampling_weight": 0.125,
            }
        )
    )
    config_path = tmp_path / "motions.toml"
    config_path.write_text(
        "\n".join(
            (
                "[registry]",
                'train_prefix = ""',
                "",
                "[dataset]",
                f'glob = "{tmp_path}/ep_*.npz"',
                'forehand_template = "collected_forehand"',
                'backhand_template = "collected_backhand"',
                "target_pos_std = { x = 0.2, y = 0.3, z = 0.3 }",
                "target_vel_std = { x = 0.0, y = 0.0, z = 0.0 }",
                (
                    "target_orientation_std = "
                    "{ roll = 0.0, pitch = 0.0, yaw = 0.0 }"
                ),
            )
        )
        + "\n"
    )

    motion_cfgs = MotionSet.from_toml(config_path).local_motion_cfgs()
    assert motion_cfgs is not None
    assert motion_cfgs[0].sampling_weight == 0.125
    goal_cfgs = {
        goal.goal_type: goal for goal in motion_cfgs[0].sub_targets
    }

    assert goal_cfgs["position"].target_pos_std == {
        "x": 0.2,
        "y": 0.3,
        "z": 0.3,
    }
    assert goal_cfgs["velocity"].target_vel_std == {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
    }
    assert goal_cfgs["orientation"].target_orientation_std == {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }
