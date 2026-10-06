"""Planar frame calibration helpers for Mocap deployment data."""

from __future__ import annotations

import math

import numpy as np


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return the active rotation matrix represented by an XYZW quaternion."""
    x, y, z, w = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_xyzw_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Compose two active XYZW quaternion rotations."""
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    result = np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def planar_calibration_from_root_pose(
    root_position_w: np.ndarray,
    root_quaternion_xyzw: np.ndarray,
    target_xy_w: np.ndarray,
    target_yaw_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a yaw/XY transform that maps the current root to the target pose."""
    root_rotation = quaternion_xyzw_to_matrix(root_quaternion_xyzw)
    root_yaw = math.atan2(root_rotation[1, 0], root_rotation[0, 0])
    yaw_correction = target_yaw_rad - root_yaw
    cos_yaw = math.cos(yaw_correction)
    sin_yaw = math.sin(yaw_correction)
    rotation = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    translation = np.zeros(3, dtype=np.float64)
    translation[:2] = target_xy_w - (rotation @ root_position_w)[:2]
    half_yaw = 0.5 * yaw_correction
    quaternion = np.array(
        [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)], dtype=np.float64
    )
    return rotation, translation, quaternion
