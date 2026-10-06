"""Viewer stalls must not advance the simulation command watchdog."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.simulation_node.simulation_tppo_student import TppoStudentSimulationNode
from utils.deployment_fsm import DeploymentFSM


class SimulationFsmClockTest(unittest.TestCase):
    def make_node(self):
        node = object.__new__(TppoStudentSimulationNode)
        node.data = SimpleNamespace(time=1.0)
        node.deployment_fsm = DeploymentFSM(
            default_joint_position=np.zeros(29), home_kp=np.ones(29) * 40,
            home_kd=np.ones(29) * 2, home_duration_s=.1,
            command_timeout_s=.1,
        )
        fsm = node.deployment_fsm
        fsm.request('home')
        fsm.step(np.zeros(29), now_s=0.)
        fsm.step(np.zeros(29), now_s=1.)
        fsm.request('control')
        fsm.step(np.zeros(29), now_s=1.)
        return node

    def test_command_timestamp_uses_sim_time_and_real_gap_still_expires(self):
        node = self.make_node()
        command = np.concatenate([np.zeros(58), np.ones(29)*40,
                                  np.ones(29)*2, np.zeros(29)])
        with patch('time.monotonic', return_value=123456.):
            node._command_callback(SimpleNamespace(data=command))
        fsm = node.deployment_fsm
        self.assertEqual(fsm._last_policy_command_s, 1.)
        # Arbitrarily long wall stall does not consume the simulation budget.
        with patch('time.monotonic', return_value=123999.):
            self.assertEqual(fsm.step(np.zeros(29), now_s=1.02), 'control')
        # Missing five simulation control ticks must still cause damping.
        self.assertEqual(fsm.step(np.zeros(29), now_s=1.101), 'damp')
        self.assertEqual(fsm.fault_reason, 'policy_command_timeout')

    def test_stale_pre_control_command_cannot_bypass_watchdog(self):
        node = self.make_node()
        node.data.time = .9
        node._command_callback(SimpleNamespace(data=np.zeros(145)))
        fsm = node.deployment_fsm
        self.assertEqual(fsm.step(np.zeros(29), now_s=1.101), 'damp')
        self.assertEqual(fsm.fault_reason, 'policy_command_timeout')
