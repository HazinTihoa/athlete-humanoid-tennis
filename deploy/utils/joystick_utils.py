##
#
# Joystick math functions. Tested with XBOX controllers.
#
##

# standard imports
from dataclasses import dataclass

#######################################################################
# JOYSTICK STATE
#######################################################################

# joystick state dataclass
@dataclass
class JoystickState:
    
    # Buttons
    A: int = 0
    B: int = 0
    X: int = 0
    Y: int = 0
    LB: int = 0
    RB: int = 0
    LMB: int = 0   # left middle button
    RMB: int = 0   # right middle button
    LS: int = 0    # left stick press
    RS: int = 0    # right stick press
    XBOX: int = 0  # XBOX button in the middle of the controller

    # Axes
    LS_X: float = 0.0
    LS_Y: float = 0.0
    RS_X: float = 0.0
    RS_Y: float = 0.0
    LT: float = 0.0
    RT: float = 0.0
    L_DPAD: float = 0.0  # D-PAD left
    R_DPAD: float = 0.0  # D-PAD right
    U_DPAD: float = 0.0  # D-PAD up
    D_DPAD: float = 0.0  # D-PAD down

#######################################################################
# Helper Functions
#######################################################################

# pygame joystick to JoystickState conversion
def pygame_to_joystick_state(joystick):

    state = JoystickState()

    # joysticks
    state.LS_X = -joystick.get_axis(0) # invert x-axis
    state.LS_Y = -joystick.get_axis(1) # invert y-axis
    state.RS_X = -joystick.get_axis(3) # invert x-axis
    state.RS_Y = -joystick.get_axis(4) # invert y-axis

    # triggers
    state.LT = 0.5 * joystick.get_axis(2) + 0.5
    state.RT = 0.5 * joystick.get_axis(5) + 0.5

    # D-PAD
    DPAD = joystick.get_hat(0)
    
    DPAD_X = DPAD[0]
    if DPAD_X <= -0.5:
        state.L_DPAD = 1.0
        state.R_DPAD = 0.0
    elif DPAD_X >= 0.5:
        state.L_DPAD = 0.0
        state.R_DPAD = 1.0
    else:
        state.L_DPAD = 0.0
        state.R_DPAD = 0.0

    DPAD_Y = DPAD[1]
    if DPAD_Y <= -0.5:
        state.U_DPAD = 0.0
        state.D_DPAD = 1.0
    elif DPAD_Y >= 0.5:
        state.U_DPAD = 1.0
        state.D_DPAD = 0.0
    else:
        state.U_DPAD = 0.0
        state.D_DPAD = 0.0

    # buttons
    state.A = joystick.get_button(0)
    state.B = joystick.get_button(1)
    state.X = joystick.get_button(2)
    state.Y = joystick.get_button(3)
    state.LB = joystick.get_button(4)
    state.RB = joystick.get_button(5)
    state.LMB = joystick.get_button(6)
    state.RMB = joystick.get_button(7)
    state.XBOX = joystick.get_button(8)
    state.LS = joystick.get_button(9)
    state.RS = joystick.get_button(10)

    return state


# ROS Joy message to JoystickState conversion
def rosjoy_to_joystick_state(joy_msg):

    state = JoystickState()

    # joysticks
    state.LS_X = joy_msg.axes[0]
    state.LS_Y = joy_msg.axes[1]
    state.RS_X = joy_msg.axes[3]
    state.RS_Y = joy_msg.axes[4]

    # triggers
    state.LT = -0.5 * joy_msg.axes[2] + 0.5
    state.RT = -0.5 * joy_msg.axes[5] + 0.5

    # D-PAD axes
    DPAD_X = joy_msg.axes[6]
    if DPAD_X == 1.0:
        state.L_DPAD = 1.0
        state.R_DPAD = 0.0
    elif DPAD_X == -1.0:
        state.L_DPAD = 0.0
        state.R_DPAD = 1.0
    else:
        state.L_DPAD = 0.0
        state.R_DPAD = 0.0

    DPAD_Y = joy_msg.axes[7]
    if DPAD_Y == -1.0:
        state.U_DPAD = 0.0
        state.D_DPAD = 1.0
    elif DPAD_Y == 1.0:
        state.U_DPAD = 1.0
        state.D_DPAD = 0.0
    else:
        state.U_DPAD = 0.0
        state.D_DPAD = 0.0

    # buttons
    state.A = joy_msg.buttons[0]
    state.B = joy_msg.buttons[1]
    state.X = joy_msg.buttons[2]
    state.Y = joy_msg.buttons[3]
    state.LB = joy_msg.buttons[4]
    state.RB = joy_msg.buttons[5]
    state.LMB = joy_msg.buttons[6]
    state.RMB = joy_msg.buttons[7]
    state.XBOX = joy_msg.buttons[8]
    state.LS = joy_msg.buttons[9]
    state.RS = joy_msg.buttons[10]

    return state