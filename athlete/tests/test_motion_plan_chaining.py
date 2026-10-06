from types import SimpleNamespace

import pytest
import torch

from athlete.goal_cond_tracking.mdp.commands import (
    MotionResamplePlan,
    MultiTargetMotionCommand,
)
from athlete.goal_cond_tracking.mdp.phase_commands import (
    PhaseAccelerationMultiTargetMotionCommand,
)


@pytest.mark.parametrize("failure_probability", [0.0, 0.05])
def test_chain_applies_motion_target_deadline_atomically(failure_probability):
    command = object.__new__(MultiTargetMotionCommand)
    command.cfg = SimpleNamespace(
        incoming_ball_failure_trajectory_probability=failure_probability,
        incoming_ball_torch_match_enabled=True,
        max_contact_speedup=2.0,
    )
    command.which_motion = torch.zeros(3, dtype=torch.long)
    command.time_steps = torch.full((3,), 93, dtype=torch.long)
    command._time_step_totals = torch.tensor([95, 150])
    command.motion_loaders = [None, None]
    command.max_subtargets = 1
    command.motion_chain_count = torch.zeros(3, dtype=torch.long)
    command.target_position_w = torch.zeros(3, 1, 3)
    command.planned_contact_time = torch.zeros(3)
    command.planned_target_index = torch.zeros(3, dtype=torch.long)
    command.planned_launch_position_w = torch.zeros(3, 3)
    command.planned_launch_linear_velocity_w = torch.zeros(3, 3)
    command.planned_launch_angular_velocity_w = torch.zeros(3, 3)
    command.trajectory_match_valid = torch.ones(3, dtype=torch.bool)
    command.metrics = {
        key: torch.zeros(3)
        for key in (
            "motion_chain_count", "trajectory_match_distance",
            "trajectory_match_valid", "failure_trajectory",
            "trajectory_match_attempt", "trajectory_prediction_error",
        )
    }
    command._record_stroke_outcomes = lambda *args, **kwargs: None
    command._update_ghost_alignment = lambda env_ids: None
    command._sample_targets = lambda env_ids: None
    command._apply_analytic_strike_targets = lambda env_ids: None
    aligned = []
    command._align_reference_frame0_to_robot = lambda ids: aligned.extend(ids.tolist())

    # The nearest next motion allows 2.21 s; the preceding one only allows 1.88 s.
    plan = MotionResamplePlan(
        motion_ids=torch.tensor([1, 1]),
        target_indices=torch.zeros(2, dtype=torch.long),
        target_positions_w=torch.tensor([[1.0, 2.0, 0.8], [2.0, 1.0, 0.7]]),
        contact_times=torch.tensor([2.21, 2.31]),
        launch_positions_w=torch.ones(2, 3),
        launch_linear_velocities_w=torch.ones(2, 3),
        launch_angular_velocities_w=torch.zeros(2, 3),
        match_distances=torch.tensor([0.3, 0.1]),
        valid=torch.tensor([False, True]),
        attempts=torch.ones(2, dtype=torch.long),
    )
    command._motion_plan_provider = SimpleNamespace(
        plan_for_motion_chain=lambda env_ids: plan
    )
    env_ids = torch.tensor([2, 0])
    command._sample_next_motion(env_ids)

    active = torch.tensor([0]) if failure_probability else env_ids
    assert aligned == active.tolist()
    assert command.which_motion[0].item() == 1
    assert command.which_motion[1].item() == 0
    assert command.which_motion[2].item() == (0 if failure_probability else 1)
    assert command.time_steps[2].item() == (94 if failure_probability else 0)
    torch.testing.assert_close(command.target_position_w[0, 0], plan.target_positions_w[1])
    torch.testing.assert_close(
        command.target_position_w[2, 0],
        torch.zeros(3) if failure_probability else plan.target_positions_w[0],
    )
    assert command.metrics["failure_trajectory"][2].item() == bool(failure_probability)
    assert not command.trajectory_match_valid[2].item()
    torch.testing.assert_close(command.planned_contact_time[env_ids], plan.contact_times)

    command._nominal_contact_times = torch.tensor([1.88, 3.0])
    command.contact_time = torch.zeros(3)
    command.time_remaining = torch.zeros(3)
    command.strike_time_error_s = torch.zeros(3)
    PhaseAccelerationMultiTargetMotionCommand._sample_contact_times(command, active)
    torch.testing.assert_close(command.contact_time[active], command.planned_contact_time[active])

    # Genuine invalid deadlines still fail; the fix must not clamp their times.
    command.planned_contact_time[0] = 3.5
    with pytest.raises(ValueError, match="outside the motion deadline interval"):
        PhaseAccelerationMultiTargetMotionCommand._sample_contact_times(command, torch.tensor([0]))
