from types import SimpleNamespace
from unittest.mock import Mock

import mjlab  # Register tasks before importing their command types.
import numpy as np
import torch

from athlete.goal_cond_tracking.mdp.commands import (
    MultiTargetMotionCommand,
)


def test_root_frame_uses_actual_position_and_full_rotation():
    command = SimpleNamespace(
        cfg=SimpleNamespace(viz=SimpleNamespace(show_root_frame=True)),
        robot=SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[1.0, 2.0, 3.0]]),
            root_link_quat_w=torch.tensor([[2**-0.5, 2**-0.5, 0.0, 0.0]]),
        )),
    )
    visualizer = Mock()
    MultiTargetMotionCommand._add_root_frame_vis(command, visualizer, 0)
    args = visualizer.add_frame.call_args.kwargs
    np.testing.assert_allclose(args["position"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        args["rotation_matrix"], [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-6
    )
    assert args["scale"] == 0.3
    assert args["label"] == "robot_root_0"


def test_root_frame_disabled_does_not_read_robot_state():
    command = SimpleNamespace(
        cfg=SimpleNamespace(viz=SimpleNamespace(show_root_frame=False))
    )
    visualizer = Mock()
    MultiTargetMotionCommand._add_root_frame_vis(command, visualizer, 0)
    visualizer.add_frame.assert_not_called()
