##
#
# Run joystick commands (using ROS2 joy_node)
#
##

import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray, String

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.joystick_utils import JoystickState, rosjoy_to_joystick_state
from utils.finite_state_machine import FiniteStateMachine


############################################################################
# COMMAND NODE
############################################################################

class JoystickNode(Node):
    """
    Use joystick to publish commands to the simulation and control nodes.
    """

    def __init__(self):
        super().__init__('joystick_node')

        self.deadzone = 0.05
        self.joystick_state = JoystickState()
        self.init_joystick()

        self.fsm = FiniteStateMachine()

        self.command_pub = self.create_publisher(Float32MultiArray, 'deploy_robot/joystick', 10)
        self.fsm_pub = self.create_publisher(String, 'deploy_robot/fsm', 10)

        joystick_dt = 0.02
        self.command_timer = self.create_timer(joystick_dt, self.publish_command)

        print("Joystick node initialized.")

    def init_joystick(self):
        self.joy_process = subprocess.Popen(
            [
                'ros2', 'run', 'joy', 'joy_node',
                '--ros-args',
                '-r', 'joy:=/deploy_robot/joy',
                '-p', f'deadzone:={self.deadzone}',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("Launched joy_node as a subprocess.")

        self.joy_sub = self.create_subscription(Joy, 'deploy_robot/joy', self.joy_callback, 10)

        self.joy_msg = None
        self.is_connected = 0.0
        print("No joystick found. Waiting for connection...")
        while self.joy_msg is None:
            rclpy.spin_once(self, timeout_sec=0.01)

        self._joy_timeout = 0.2
        self._last_joy_time = time.time()

    def joy_callback(self, msg: Joy):
        self.joy_msg = msg
        self._last_joy_time = time.time()
        if self.is_connected == 0.0:
            print("Joystick connected.")
            self.is_connected = 1.0

    def publish_command(self):
        if time.time() - self._last_joy_time > self._joy_timeout:
            if self.is_connected == 1.0:
                print("Joystick disconnected.")
            self.is_connected = 0.0
            self.joystick_state = JoystickState()

        if self.is_connected == 0.0:
            fsm_msg = String()
            fsm_msg.data = self.fsm.force_damp()
            self.fsm_pub.publish(fsm_msg)
            cmd_msg = Float32MultiArray()
            cmd_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.command_pub.publish(cmd_msg)
            return

        self.joystick_state = rosjoy_to_joystick_state(self.joy_msg)

        fsm_state = self.fsm.step(self.joystick_state)
        fsm_msg = String()
        fsm_msg.data = fsm_state
        self.fsm_pub.publish(fsm_msg)

        # layout: [LS_X, LS_Y, RS_X, RS_Y, B, X, Y]
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [
            self.joystick_state.LS_X,           # data[0]
            self.joystick_state.LS_Y,           # data[1]
            self.joystick_state.RS_X,           # data[2]
            self.joystick_state.RS_Y,           # data[3]
            float(self.joystick_state.B),       # data[4]: launch / trigger
            float(self.joystick_state.X),       # data[5]: reset
            float(self.joystick_state.Y),       # data[6]: otherhand offset toggle
        ]
        self.command_pub.publish(cmd_msg)

    def destroy_node(self):
        if self.joy_process is not None:
            self.joy_process.terminate()
            self.joy_process.wait()
        super().destroy_node()


############################################################################
# MAIN FUNCTION
############################################################################

def main():
    rclpy.init()

    joystick_node = JoystickNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(joystick_node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        joystick_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print("Joystick shutdown complete.")


if __name__ == "__main__":
    main()
