"""Contract tests for the Intent TPPO real-robot hardware adapter."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

DEPLOY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_DIR))
sys.path.insert(0, str(DEPLOY_DIR / "hardware" / "hardware_node"))

try:
    import rclpy
    from hardware_athlete import ControlNode
    from std_msgs.msg import Float32MultiArray, String
except ImportError as exc:  # The Unitree wrapper exists only in the deploy venv.
    rclpy = None
    ControlNode = None
    Float32MultiArray = None
    String = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from utils.safety import UNITREE_REMOTE_Y  # noqa: E402


@unittest.skipIf(
    IMPORT_ERROR is not None, f"deploy environment unavailable: {IMPORT_ERROR}"
)
class TppoRealHardwareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rclpy.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self) -> None:
        self.node = ControlNode(
            "g1_tppo_student_intentref.yaml",
            enable_low_level_commands=False,
            use_unitree_remote_fsm=True,
        )

    def tearDown(self) -> None:
        self.node.destroy_node()

    def test_intent_config_requires_no_legacy_motion_fields(self) -> None:
        for field in ("contact_phase", "contact_duration", "motion_paths", "goals"):
            self.assertNotIn(field, self.node.config)
        for attribute in (
            "motion_frame",
            "motion_idx",
            "goals_pub",
            "which_motion_pub",
        ):
            self.assertFalse(hasattr(self.node, attribute))

    def test_hardware_adapter_initializes_without_real_sdk_connection(self) -> None:
        fake_robot = object()
        with mock.patch(
            "hardware_athlete.unitree_interface.UnitreeInterface.create_g1",
            return_value=fake_robot,
        ) as create_g1:
            self.node.network = "test-interface"
            self.node.Init()

        create_g1.assert_called_once_with("test-interface")
        self.assertIs(self.node.robot, fake_robot)
        self.assertTrue(hasattr(self.node, "pelvis_imu_state_pub"))
        self.assertTrue(hasattr(self.node, "joint_state_pub"))
        self.assertTrue(hasattr(self.node, "command_sub"))
        self.assertTrue(hasattr(self.node, "fsm_request_sub"))
        self.assertTrue(hasattr(self.node, "fsm_state_pub"))

    def test_145_dimension_student_command_is_accepted(self) -> None:
        command = np.arange(145, dtype=np.float32)
        message = Float32MultiArray()
        message.data = command.tolist()

        fsm = self.node.deployment_fsm
        fsm.home_duration_s = 0.01
        position = np.zeros(29)
        fsm.request("home")
        fsm.step(position, now_s=1.0)
        fsm.step(position, now_s=1.01)
        fsm.request("control")
        fsm.step(position, now_s=1.011)
        with mock.patch("hardware_athlete.time.monotonic", return_value=1.012):
            self.node.command_callback(message)
        fsm.step(position, now_s=1.013)

        np.testing.assert_allclose(fsm.q_target, command[0:29])
        np.testing.assert_allclose(fsm.dq_target, command[29:58])
        np.testing.assert_allclose(fsm.kp, command[58:87])
        np.testing.assert_allclose(fsm.kd, command[87:116])
        np.testing.assert_allclose(fsm.tau_ff, command[116:145])

    def test_fsm_request_is_forwarded_to_shared_state_machine(self) -> None:
        message = String()
        message.data = "home"
        self.node.fsm_request_callback(message)
        self.assertEqual(
            self.node.deployment_fsm.step(np.zeros(29), now_s=1.0), "home"
        )

    def test_remote_y_requests_root_calibration_in_home(self) -> None:
        class Publisher:
            def __init__(self) -> None:
                self.messages = []

            def publish(self, message) -> None:
                self.messages.append(message)

        class Remote:
            keys = UNITREE_REMOTE_Y

        class FakeRobot:
            @staticmethod
            def read_wireless_controller():
                return Remote()

        self.node.robot = FakeRobot()
        self.node.root_calibration_request_pub = Publisher()
        self.node.deployment_fsm.request("home")
        self.node.deployment_fsm.step(np.zeros(29), now_s=1.0)
        self.node._update_unitree_remote_fsm()

        self.assertEqual(
            self.node.root_calibration_request_pub.messages[-1].data,
            "calibrate",
        )

    def test_lowcmd_loop_sends_shared_fsm_output(self) -> None:
        low_state = types.SimpleNamespace(
            motor=types.SimpleNamespace(
                q=np.zeros(29), dq=np.zeros(29), tau_est=np.zeros(29)
            ),
            imu=types.SimpleNamespace(
                rpy=np.zeros(3),
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
                omega=np.zeros(3),
                accel=np.zeros(3),
            ),
        )

        class FakeRobot:
            def __init__(self) -> None:
                self.last_command = None

            def read_low_state(self):
                return low_state

            def create_zero_command(self):
                return types.SimpleNamespace()

            def write_low_command(self, command) -> None:
                self.last_command = command

        robot = FakeRobot()
        self.node.robot = robot
        self.node.LowCmdWrite()

        self.assertIsNotNone(robot.last_command)
        np.testing.assert_allclose(robot.last_command.q_target, 0.0)
        np.testing.assert_allclose(robot.last_command.kp, 0.0)
        np.testing.assert_allclose(robot.last_command.kd, self.node.damp_kd)


if __name__ == "__main__":
    unittest.main()
