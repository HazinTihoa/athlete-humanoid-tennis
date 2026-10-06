"""MuJoCo FK helpers shared by Mocap calibration tools."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


ACTION_DIM = 29


def parse_joint_names(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        names = tuple(name.strip() for name in value.split(",") if name.strip())
    elif isinstance(value, (list, tuple)):
        names = tuple(str(name).strip() for name in value if str(name).strip())
    else:
        raise TypeError("joint_names must be a comma-separated string or a list")
    if len(names) != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM} joint names, got {len(names)}")
    return names


class SweetSpotForwardKinematics:
    """Evaluate the deployment sweet-spot site at measured joint positions."""

    def __init__(
        self,
        xml_path: Path,
        joint_names: tuple[str, ...],
        site_name: str,
        position_offset_b: np.ndarray | None = None,
    ) -> None:
        self.xml_path = xml_path
        self.joint_names = joint_names
        self.site_name = site_name
        self.position_offset_b = np.zeros(3, dtype=np.float64)
        if position_offset_b is not None:
            offset = np.asarray(position_offset_b, dtype=np.float64)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError("position_offset_b must contain three finite values")
            self.position_offset_b[:] = offset
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        root_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        if root_joint_id < 0:
            raise ValueError("Kinematics XML has no floating_base_joint")
        self.root_qpos_address = int(self.model.jnt_qposadr[root_joint_id])
        self.root_body_id = int(self.model.jnt_bodyid[root_joint_id])

        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if self.site_id < 0:
            raise ValueError(f"Kinematics XML has no site {site_name!r}")

        joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in joint_names
            ],
            dtype=np.int32,
        )
        if np.any(joint_ids < 0):
            missing = [
                name
                for name, joint_id in zip(joint_names, joint_ids)
                if joint_id < 0
            ]
            raise ValueError(f"Kinematics XML is missing joints: {missing}")
        self.joint_qpos_addresses = self.model.jnt_qposadr[joint_ids].copy()

    def evaluate(
        self,
        root_position_w: np.ndarray,
        root_quaternion_xyzw: np.ndarray,
        joint_position: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:] = self.model.qpos0
        qadr = self.root_qpos_address
        self.data.qpos[qadr : qadr + 3] = root_position_w
        x, y, z, w = root_quaternion_xyzw
        self.data.qpos[qadr + 3 : qadr + 7] = (w, x, y, z)
        self.data.qpos[self.joint_qpos_addresses] = joint_position
        mujoco.mj_forward(self.model, self.data)
        root_rotation_wb = self.data.xmat[self.root_body_id].reshape(3, 3)
        position_w = (
            self.data.site_xpos[self.site_id]
            + root_rotation_wb @ self.position_offset_b
        ).copy()
        rotation_w = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return position_w, rotation_w
