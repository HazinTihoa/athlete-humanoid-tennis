"""Contract tests for Sim-only MuJoCo interactive perturbation."""

from __future__ import annotations

import contextlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np
import yaml

DEPLOY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_DIR))

try:
    from simulation.simulation_node.simulation_tppo_student import (
        TppoStudentSimulationNode,
    )
except ImportError as exc:
    TppoStudentSimulationNode = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class _FakeLogger:
    def info(self, _message: str) -> None:
        pass


class _FakeViewer:
    def __init__(self) -> None:
        self.perturb = types.SimpleNamespace(
            select=0,
            localpos=np.ones(3),
        )
        self.opt = types.SimpleNamespace(
            flags=np.zeros(mujoco.mjtVisFlag.mjNVISFLAG, dtype=bool)
        )

    def lock(self):
        return contextlib.nullcontext()


@unittest.skipIf(
    IMPORT_ERROR is not None, f"deploy environment unavailable: {IMPORT_ERROR}"
)
class TppoSimInteractionTest(unittest.TestCase):
    @staticmethod
    def _incoming_ball_sampler() -> TppoStudentSimulationNode:
        node = object.__new__(TppoStudentSimulationNode)
        node.incoming_ball_rng = np.random.default_rng(20260906)
        node.launch_position_min_w = np.array([5.5, -1.5, 0.6])
        node.launch_position_max_w = np.array([6.5, 1.5, 1.2])
        node.launch_linear_velocity_min_w = np.array([-4.2, -1.25, 3.0])
        node.launch_linear_velocity_max_w = np.array([-2.8, 1.25, 5.5])
        node.launch_angular_velocity_min_w = np.zeros(3)
        node.launch_angular_velocity_max_w = np.zeros(3)
        node.failure_trajectory_probability = 0.5
        node.failure_trajectory_no_net_fraction = 0.5
        node.failure_trajectory_xy_distance_threshold_m = 3.0
        node.failure_trajectory_sampling_attempts = 24
        node.failure_net_crossing_height_range_m = np.array([1.5, 3.5])
        node.failure_maximum_initial_speed_m_s = 7.0
        node.court_net_x_m = 3.5
        node.court_net_height_m = 0.914
        node.court_net_half_width_m = 2.5
        node.ball_radius_m = 0.0335
        node.maximum_ball_flight_time_s = 4.0
        node.sim_dt = 0.0025
        node.wind_w = np.zeros(3)
        node.air_density_kg_m3 = 1.225
        node.drag_coefficient = 0.55
        node.dynamic_viscosity_pa_s = 1.81e-5
        node.physics_sample = types.SimpleNamespace(
            ball_mass_kg=0.0577,
            court_restitution=0.75,
            ground_tangent_speed_retention=0.825,
        )
        return node

    def test_small_court_config_enables_half_failure_trajectories(self) -> None:
        config_path = DEPLOY_DIR / "configs/g1_tppo_student_intentref_smallcourt.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        incoming_ball = config["simulation"]["incoming_ball"]

        self.assertEqual(incoming_ball["failure_trajectory_probability"], 0.5)
        self.assertEqual(incoming_ball["failure_trajectory_no_net_fraction"], 0.5)
        self.assertEqual(
            incoming_ball["failure_trajectory_xy_distance_threshold_m"], 3.0
        )

    def test_failure_sampler_realizes_half_failure_trajectories(self) -> None:
        node = self._incoming_ball_sampler()
        counts = {"normal": 0, "no_net": 0, "too_far": 0}

        for _ in range(100):
            position, velocity, _, trajectory_type, attempts = (
                node._sample_incoming_ball_state(np.zeros(2))
            )
            self.assertEqual(
                node._predict_incoming_trajectory_type(
                    position, velocity, np.zeros(2)
                ),
                trajectory_type,
            )
            self.assertLessEqual(attempts, node.failure_trajectory_sampling_attempts)
            counts[trajectory_type] += 1

        failure_count = counts["no_net"] + counts["too_far"]
        self.assertGreaterEqual(failure_count, 35)
        self.assertLessEqual(failure_count, 65)
        self.assertGreater(counts["no_net"], 0)
        self.assertGreater(counts["too_far"], 0)

    def test_policy_tick_is_emitted_every_eight_physics_steps(self) -> None:
        node = object.__new__(TppoStudentSimulationNode)
        node.sim_dt = 0.0025
        node.sensor_publish_dt = 0.01
        node.control_dt = 0.02
        node.sensor_publish_accumulator = 0.0
        node.control_publish_accumulator = 0.0
        current_step = [0]
        state_steps: list[int] = []
        control_steps: list[int] = []
        node._publish_state = lambda: state_steps.append(current_step[0])
        node._publish_control_tick = lambda: control_steps.append(current_step[0])

        for step in range(1, 25):
            current_step[0] = step
            node._publish_due_state_and_control()

        self.assertEqual(state_steps, [4, 8, 12, 16, 20, 24])
        self.assertEqual(control_steps, [8, 16, 24])

    def test_interactive_mode_preselects_ball_and_force_visualization(self) -> None:
        node = object.__new__(TppoStudentSimulationNode)
        node.viewer = _FakeViewer()
        node.ball_body_id = 17
        node.get_logger = lambda: _FakeLogger()

        node._configure_interactive_perturbation()

        self.assertEqual(node.viewer.perturb.select, 17)
        np.testing.assert_array_equal(node.viewer.perturb.localpos, 0.0)
        self.assertTrue(
            node.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE]
        )
        self.assertTrue(node.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ])

    def test_gui_perturbation_is_applied_to_mujoco_data(self) -> None:
        node = object.__new__(TppoStudentSimulationNode)
        node.interactive_perturbation = True
        node.viewer = _FakeViewer()
        node.model = object()
        node.data = object()

        with (
            mock.patch(
                "simulation.simulation_node.simulation_tppo_student."
                "mujoco.mjv_applyPerturbPose"
            ) as apply_pose,
            mock.patch(
                "simulation.simulation_node.simulation_tppo_student."
                "mujoco.mjv_applyPerturbForce"
            ) as apply_force,
        ):
            node._apply_interactive_perturbation()

        apply_pose.assert_called_once_with(
            node.model, node.data, node.viewer.perturb, 0
        )
        apply_force.assert_called_once_with(
            node.model, node.data, node.viewer.perturb
        )

    def test_aerodynamics_adds_to_existing_ball_perturbation(self) -> None:
        node = object.__new__(TppoStudentSimulationNode)
        node.ball_body_id = 1
        node.ball_dof_address = 0
        node.ball_radius_m = 0.0335
        node.air_density_kg_m3 = 1.225
        node.dynamic_viscosity_pa_s = 0.0000181
        node.wind_w = np.zeros(3)
        node.drag_coefficient = 0.55
        node.magnus_coefficient = 0.0
        node.angular_drag_coefficient = 0.01
        xfrc_applied = np.zeros((2, 6), dtype=np.float64)
        xfrc_applied[1, 0] = 2.0
        node.data = types.SimpleNamespace(
            qvel=np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            xfrc_applied=xfrc_applied,
        )

        node._apply_ball_aerodynamics()

        self.assertGreater(node.data.xfrc_applied[1, 0], 0.0)
        self.assertLess(node.data.xfrc_applied[1, 0], 2.0)


if __name__ == "__main__":
    unittest.main()
