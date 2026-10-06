"""Boundary cases for the trajectory eligibility used by matching experiments."""

import pytest
import torch

from athlete.scripts.matching_reachability import trajectory_reachability


def evaluate(points, bounces, **kwargs):
    return trajectory_reachability(
        torch.tensor([points], dtype=torch.float64),
        torch.tensor([bounces]),
        **kwargs,
    )


def test_first_bounce_outside_disk_can_enter_later():
    result = evaluate([[4.0, 0, 0.03], [2.5, 0, 0.8], [1.5, 0, 0.03]], [1, 1, 2])
    assert result.first_bounce_radius.item() == 4.0
    assert result.within_radius.item()
    assert result.has_height.item()
    assert result.region_max_height.item() == 0.8


def test_apex_outside_disk_does_not_qualify_low_inside_segment():
    result = evaluate([[4.0, 0, 1.8], [2.5, 0, 0.49], [1.5, 0, 0.03]], [1, 1, 2])
    assert result.within_radius.item()
    assert not result.has_height.item()
    assert result.region_max_height.item() == 0.49


def test_second_bounce_and_later_height_cannot_qualify():
    result = evaluate([[2.5, 0, 0.4], [1.5, 0, 0.8], [1.0, 0, 1.2]], [1, 2, 3])
    assert not result.has_height.item()
    assert result.region_max_height.item() == 0.4


def test_radius_and_height_boundaries_are_inclusive():
    result = evaluate([[3.0, 0, 0.5]], [1])
    assert result.has_height.item()
    assert result.region_min_xy.item() == 3.0
    assert result.region_max_height.item() == 0.5


def test_pre_second_scope_can_use_pre_first_bounce_sample():
    points, bounces = [[2.0, 0, 0.6], [2.1, 0, 0.03], [2.2, 0, 0.4]], [0, 1, 1]
    assert not evaluate(points, bounces).has_height.item()
    assert evaluate(points, bounces, bounce_scope="pre_second").has_height.item()


def test_xy_distance_is_horizontal_and_relative_to_robot():
    result = evaluate([[13.0, -2, 2.0]], [1], root_positions_xy=torch.tensor([10.0, -2.0]))
    assert result.has_height.item()
    assert result.region_min_xy.item() == 3.0


def test_no_valid_phase_or_no_entry_diagnostics():
    no_phase = evaluate([[0.0, 0, 1.0]], [0])
    assert not no_phase.within_radius.item()
    assert torch.isneginf(no_phase.region_max_height).item()
    assert torch.isposinf(no_phase.region_min_xy).item()
    assert torch.isnan(no_phase.first_bounce_radius).item()
    outside = evaluate([[3.01, 0, 2.0]], [1])
    assert not outside.within_radius.item()
    assert not outside.has_height.item()


@pytest.mark.parametrize("kwargs", [
    {"reach_radius": 0}, {"minimum_reach_height": -1}, {"bounce_scope": "post_second"},
])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        evaluate([[0.0, 0, 1.0]], [1], **kwargs)
