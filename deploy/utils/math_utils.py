##
#
# Helper math functions.
#
##

import numpy as np


############################################################################
# ROTATION HELPERS
############################################################################

def quat_to_rpy(q):
    """Convert quaternion to roll, pitch, yaw. Format: [w, x, y, z]."""
    w, x, y, z = q
    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def quat_conjugate(q):
    """Quaternion conjugate. Format: [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_multiply(q1, q2):
    """Multiply two quaternions. Format: [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def quat_to_rotation_matrix(q):
        """Convert quaternion to 3x3 rotation matrix. q format: [w, x, y, z]"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ], dtype=np.float32)


def quat_to_rot6d(q):
    """Convert quaternion to 6D rotation encoding (first two columns of rotation matrix)."""
    rot = quat_to_rotation_matrix(q)
    return np.array([
        rot[0,0], 
        rot[0,1],
        rot[1,0], 
        rot[1,1],
        rot[2,0], 
        rot[2,1]
    ], dtype=np.float32)


def yaw_quat(q):
    """Extract the yaw-only quaternion from q=[w,x,y,z]. Matches mjlab/isaaclab yaw_quat."""
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float32)



def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw to quaternion [w, x, y, z]."""
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)
