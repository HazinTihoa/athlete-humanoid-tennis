"""Tests for real-robot FSM and command-timeout safety behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_DIR))

from utils.finite_state_machine import FiniteStateMachine  # noqa: E402
from utils.joystick_ros import JoystickNode  # noqa: E402
from utils.joystick_utils import JoystickState  # noqa: E402
from utils.safety import (  # noqa: E402
    UNITREE_REMOTE_A,
    UNITREE_REMOTE_B,
    UNITREE_REMOTE_X,
    CommandWatchdog,
    unitree_remote_fsm_transition,
)


class DeploySafetyTest(unittest.TestCase):
    def test_unitree_remote_requires_damp_home_control_sequence(self) -> None:
        state = unitree_remote_fsm_transition("damp", UNITREE_REMOTE_X, 0)
        self.assertEqual(state, "damp")
        state = unitree_remote_fsm_transition("damp", UNITREE_REMOTE_A, 0)
        self.assertEqual(state, "home")
        state = unitree_remote_fsm_transition("home", UNITREE_REMOTE_X, 0)
        self.assertEqual(state, "control")
        state = unitree_remote_fsm_transition(
            "control", UNITREE_REMOTE_B | UNITREE_REMOTE_X, 0
        )
        self.assertEqual(state, "damp")

    def test_joystick_disconnect_publishes_damp_and_zero_command(self) -> None:
        class Publisher:
            def __init__(self) -> None:
                self.messages = []

            def publish(self, message) -> None:
                self.messages.append(message)

        class DisconnectedJoystick:
            pass

        node = DisconnectedJoystick()
        node._last_joy_time = 0.0
        node._joy_timeout = 0.2
        node.is_connected = 1.0
        node.joystick_state = JoystickState()
        node.fsm = FiniteStateMachine()
        node.fsm.state = "control"
        node.fsm_pub = Publisher()
        node.command_pub = Publisher()

        JoystickNode.publish_command(node)

        self.assertEqual(node.is_connected, 0.0)
        self.assertEqual(node.fsm.state, "damp")
        self.assertEqual(node.fsm_pub.messages[-1].data, "damp")
        self.assertEqual(list(node.command_pub.messages[-1].data), [0.0] * 7)

    def test_forced_damp_requires_home_sequence_again(self) -> None:
        fsm = FiniteStateMachine()
        self.assertEqual(fsm.step(JoystickState(LB=1)), "damp")
        self.assertEqual(fsm.step(JoystickState(A=1)), "home")
        self.assertEqual(fsm.step(JoystickState(LMB=1)), "control")
        self.assertEqual(fsm.force_damp(), "damp")
        self.assertEqual(fsm.step(JoystickState(LMB=1)), "damp")
        self.assertEqual(fsm.step(JoystickState(A=1)), "home")
        self.assertEqual(fsm.step(JoystickState(LMB=1)), "control")

    def test_watchdog_waits_for_new_control_command_and_latches_timeout(self) -> None:
        watchdog = CommandWatchdog(timeout_s=0.1)
        watchdog.record_command(now_s=0.0)
        watchdog.update_fsm("control", now_s=1.0)

        waiting = watchdog.evaluate(now_s=1.05)
        self.assertTrue(waiting.force_damp)
        self.assertFalse(waiting.fault_latched)

        watchdog.record_command(now_s=1.06)
        active = watchdog.evaluate(now_s=1.07)
        self.assertFalse(active.force_damp)

        timed_out = watchdog.evaluate(now_s=1.17)
        self.assertTrue(timed_out.force_damp)
        self.assertTrue(timed_out.fault_just_latched)

        watchdog.record_command(now_s=1.18)
        still_latched = watchdog.evaluate(now_s=1.18)
        self.assertTrue(still_latched.force_damp)
        self.assertTrue(still_latched.fault_latched)

        self.assertTrue(watchdog.update_fsm("damp", now_s=1.19))
        self.assertFalse(watchdog.evaluate(now_s=1.20).force_damp)

    def test_watchdog_latches_when_first_control_command_never_arrives(self) -> None:
        watchdog = CommandWatchdog(timeout_s=0.1)
        watchdog.update_fsm("control", now_s=2.0)
        result = watchdog.evaluate(now_s=2.101)
        self.assertTrue(result.force_damp)
        self.assertTrue(result.fault_latched)
        self.assertTrue(result.fault_just_latched)


if __name__ == "__main__":
    unittest.main()
