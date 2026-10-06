"""Training-independent runtime for TPPO tennis Student policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import onnxruntime as ort


ACTION_DIM = 29
OBSERVATION_DIM = 119
NO_GLOBAL_ROOT_OBSERVATION_DIM = 116

INTENT_OBSERVATION_DIM = 628
INTENT_GLOBAL_ROOT_OBSERVATION_DIM = 631
INTENT_ROBUST_OBSERVATION_DIM = 638
INTENT_MEASUREMENT_OBSERVATION_DIMS = frozenset({928, 1324})
SUPPORTED_OBSERVATION_DIMS = frozenset(
    {
        OBSERVATION_DIM,
        NO_GLOBAL_ROOT_OBSERVATION_DIM,
        INTENT_OBSERVATION_DIM,
        INTENT_GLOBAL_ROOT_OBSERVATION_DIM,
        INTENT_ROBUST_OBSERVATION_DIM,
    } | INTENT_MEASUREMENT_OBSERVATION_DIMS
)
TASK_GOAL_MESSAGE_DIM = 13
SUPPORTED_AUXILIARY_INPUTS = frozenset({"which_motion", "time_step"})

OBSERVATION_SLICES: Mapping[str, slice] = {
    "task_goal": slice(0, 11),
    "time_remaining": slice(11, 12),
    "base_ang_vel": slice(12, 15),
    "joint_pos": slice(15, 44),
    "joint_vel": slice(44, 73),
    "last_action": slice(73, 102),
    "landing_target": slice(102, 104),
    "ball_position": slice(104, 107),
    "ball_velocity": slice(107, 110),
    "sweet_spot_velocity": slice(110, 113),
    "global_root_pos": slice(113, 116),
    "projected_gravity": slice(116, 119),
}

NO_GLOBAL_ROOT_OBSERVATION_SLICES: Mapping[str, slice] = {
    name: value
    for name, value in OBSERVATION_SLICES.items()
    if name != "global_root_pos"
}
NO_GLOBAL_ROOT_OBSERVATION_SLICES = {
    **NO_GLOBAL_ROOT_OBSERVATION_SLICES,
    "projected_gravity": slice(113, 116),
}


def _vector(value: np.ndarray | list[float], size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain {size} finite values, got {vector.shape}.")
    return vector


def normalize_quaternion_wxyz(quaternion: np.ndarray | list[float]) -> np.ndarray:
    quat = _vector(quaternion, 4, "quaternion")
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-8:
        raise ValueError("Quaternion norm is zero.")
    return quat / norm


def quaternion_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quat = normalize_quaternion_wxyz(quaternion)
    return quat * np.array([1.0, -1.0, -1.0, -1.0])


def quaternion_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = normalize_quaternion_wxyz(lhs)
    rw, rx, ry, rz = normalize_quaternion_wxyz(rhs)
    return normalize_quaternion_wxyz(
        np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ]
        )
    )


def quaternion_to_rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class RobotState:
    root_position_w: np.ndarray
    root_orientation_wxyz: np.ndarray
    base_angular_velocity_b: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    sweet_spot_velocity_w: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_position_w", _vector(self.root_position_w, 3, "root_position_w"))
        object.__setattr__(
            self,
            "root_orientation_wxyz",
            normalize_quaternion_wxyz(self.root_orientation_wxyz),
        )
        object.__setattr__(
            self,
            "base_angular_velocity_b",
            _vector(self.base_angular_velocity_b, 3, "base_angular_velocity_b"),
        )
        object.__setattr__(self, "joint_position", _vector(self.joint_position, ACTION_DIM, "joint_position"))
        object.__setattr__(self, "joint_velocity", _vector(self.joint_velocity, ACTION_DIM, "joint_velocity"))
        object.__setattr__(
            self,
            "sweet_spot_velocity_w",
            _vector(self.sweet_spot_velocity_w, 3, "sweet_spot_velocity_w"),
        )


@dataclass(frozen=True)
class BallState:
    position_w: np.ndarray
    velocity_w: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_w", _vector(self.position_w, 3, "ball_position_w"))
        object.__setattr__(self, "velocity_w", _vector(self.velocity_w, 3, "ball_velocity_w"))


@dataclass(frozen=True)
class TaskGoal:
    strike_position_w: np.ndarray
    target_racket_velocity_w: np.ndarray
    target_racket_orientation_wxyz: np.ndarray
    contact_time_s: float
    landing_target_position_w: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "strike_position_w", _vector(self.strike_position_w, 3, "strike_position_w"))
        object.__setattr__(
            self,
            "target_racket_velocity_w",
            _vector(
                self.target_racket_velocity_w, 3, "target_racket_velocity_w"
            ),
        )
        object.__setattr__(
            self,
            "target_racket_orientation_wxyz",
            normalize_quaternion_wxyz(self.target_racket_orientation_wxyz),
        )
        if not np.isfinite(self.contact_time_s) or self.contact_time_s < 0.0:
            raise ValueError("contact_time_s must be finite and non-negative.")
        object.__setattr__(
            self,
            "landing_target_position_w",
            _vector(self.landing_target_position_w, 3, "landing_target_position_w"),
        )

    def to_message(self) -> np.ndarray:
        return np.concatenate(
            [
                self.strike_position_w,
                self.target_racket_velocity_w,
                self.target_racket_orientation_wxyz,
                np.array([self.contact_time_s]),
                self.landing_target_position_w[:2],
            ]
        ).astype(np.float32)

    @classmethod
    def from_message(cls, data: np.ndarray | list[float]) -> "TaskGoal":
        values = _vector(data, TASK_GOAL_MESSAGE_DIM, "task_goal_message")
        return cls(
            strike_position_w=values[0:3],
            target_racket_velocity_w=values[3:6],
            target_racket_orientation_wxyz=values[6:10],
            contact_time_s=float(values[10]),
            landing_target_position_w=np.array([values[11], values[12], 0.0]),
        )


def _parse_float_metadata(metadata: Mapping[str, str], key: str) -> np.ndarray:
    value = metadata.get(key)
    if value is None:
        raise ValueError(f"Student ONNX is missing metadata {key!r}.")
    return np.asarray([float(item) for item in value.split(",")], dtype=np.float64)


class StudentOnnxPolicy:
    """ONNX Runtime adapter that exposes only the deployable Student actor."""

    def __init__(self, policy_path: str | Path) -> None:
        self.path = Path(policy_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Student ONNX not found: {self.path}")
        providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in ort.get_available_providers()
        ]
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        self.inputs = {value.name: value for value in self.session.get_inputs()}
        unsupported_inputs = set(self.inputs) - {"obs"} - SUPPORTED_AUXILIARY_INPUTS
        if unsupported_inputs:
            raise ValueError(
                f"Student ONNX has unsupported inputs: {sorted(unsupported_inputs)}."
            )
        shape = self.inputs.get("obs").shape if "obs" in self.inputs else None
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[0] != 1
            or not isinstance(shape[1], int)
            or shape[1] not in SUPPORTED_OBSERVATION_DIMS
        ):
            expected = sorted(SUPPORTED_OBSERVATION_DIMS)
            raise ValueError(
                f"Student ONNX must have obs [1, dim] with dim in {expected}, "
                f"got {shape}."
            )

        self.observation_dim = int(shape[1])
        self.is_intent_policy = self.observation_dim in {
            INTENT_OBSERVATION_DIM,
            INTENT_GLOBAL_ROOT_OBSERVATION_DIM,
            INTENT_ROBUST_OBSERVATION_DIM,
        } | INTENT_MEASUREMENT_OBSERVATION_DIMS
        self.includes_ball_observation_status = (
            self.observation_dim == INTENT_ROBUST_OBSERVATION_DIM
        )
        self.includes_global_root_pos = self.observation_dim in {
            OBSERVATION_DIM, INTENT_GLOBAL_ROOT_OBSERVATION_DIM,
        } | INTENT_MEASUREMENT_OBSERVATION_DIMS

        output_names = [value.name for value in self.session.get_outputs()]
        if "actions" not in output_names:
            raise ValueError(f"Student ONNX has no 'actions' output: {output_names}")
        action_info = next(value for value in self.session.get_outputs() if value.name == "actions")
        if action_info.shape != [1, ACTION_DIM]:
            raise ValueError(f"Student ONNX actions must be [1, {ACTION_DIM}], got {action_info.shape}.")

        self.metadata = self.session.get_modelmeta().custom_metadata_map
        joint_names = self.metadata.get("joint_names")
        if not joint_names:
            raise ValueError("Student ONNX is missing joint_names metadata.")
        self.joint_names = tuple(item.strip() for item in joint_names.split(","))
        if len(self.joint_names) != ACTION_DIM:
            raise ValueError(f"Student ONNX has {len(self.joint_names)} joints, expected {ACTION_DIM}.")
        self.default_joint_position = _parse_float_metadata(self.metadata, "default_joint_pos")
        self.action_scale = _parse_float_metadata(self.metadata, "action_scale")
        self.joint_stiffness = _parse_float_metadata(self.metadata, "joint_stiffness")
        self.joint_damping = _parse_float_metadata(self.metadata, "joint_damping")
        for name, values in (
            ("default_joint_pos", self.default_joint_position),
            ("action_scale", self.action_scale),
            ("joint_stiffness", self.joint_stiffness),
            ("joint_damping", self.joint_damping),
        ):
            if values.shape != (ACTION_DIM,):
                raise ValueError(f"ONNX metadata {name!r} must have {ACTION_DIM} values.")

    def infer(self, observation: np.ndarray) -> np.ndarray:
        observation = _vector(
            observation, self.observation_dim, "student_observation"
        )
        feeds: dict[str, np.ndarray] = {"obs": observation.astype(np.float32)[None, :]}
        for name, value in self.inputs.items():
            if name == "obs":
                continue
            shape = [dimension if isinstance(dimension, int) else 1 for dimension in value.shape]
            feeds[name] = np.zeros(shape, dtype=np.float32)
        action = np.asarray(self.session.run(["actions"], feeds)[0], dtype=np.float64).reshape(-1)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid Student action: {action.shape}")
        return action


class StudentPolicyController:
    """Shared policy API used by both simulation and real-robot ROS adapters."""

    def __init__(self, policy_path: str | Path) -> None:
        self.policy = StudentOnnxPolicy(policy_path)
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float64)

    def reset(self) -> None:
        self.last_action.fill(0.0)

    def build_observation(
        self,
        robot: RobotState,
        ball: BallState,
        goal: TaskGoal,
        time_remaining_s: float,
        global_root_position: np.ndarray | None = None,
        projected_gravity_b: np.ndarray | None = None,
    ) -> np.ndarray:
        if not np.isfinite(time_remaining_s):
            raise ValueError("time_remaining_s must be finite.")
        rotation_wb = quaternion_to_rotation_matrix_wxyz(robot.root_orientation_wxyz)
        rotation_bw = rotation_wb.T
        target_position_b = rotation_bw @ (goal.strike_position_w - robot.root_position_w)
        target_orientation_b = quaternion_multiply_wxyz(
            quaternion_conjugate_wxyz(robot.root_orientation_wxyz),
            goal.target_racket_orientation_wxyz,
        )
        landing_target_b = rotation_bw @ (
            goal.landing_target_position_w - robot.root_position_w
        )
        ball_position_b = rotation_bw @ (ball.position_w - robot.root_position_w)
        ball_velocity_b = rotation_bw @ ball.velocity_w
        sweet_spot_velocity_b = rotation_bw @ robot.sweet_spot_velocity_w
        if projected_gravity_b is None:
            projected_gravity = rotation_bw @ np.array([0.0, 0.0, -1.0])
        else:
            projected_gravity = _vector(
                projected_gravity_b, 3, "projected_gravity_b"
            )
        observation_terms = [
            target_position_b,
            # The motion command's velocity target is the racket sweet-point
            # velocity in world coordinates. It is not the outgoing ball speed.
            goal.target_racket_velocity_w,
            target_orientation_b,
            np.array([goal.contact_time_s]),
            np.array([max(0.0, time_remaining_s)]),
            robot.base_angular_velocity_b,
            robot.joint_position - self.policy.default_joint_position,
            robot.joint_velocity,
            self.last_action,
            landing_target_b[:2],
            ball_position_b,
            ball_velocity_b,
            sweet_spot_velocity_b,
        ]
        if self.policy.includes_global_root_pos:
            root_position = (
                robot.root_position_w
                if global_root_position is None
                else _vector(global_root_position, 3, "global_root_position")
            )
            observation_terms.append(root_position)
        observation_terms.append(projected_gravity)

        observation = np.concatenate(observation_terms).astype(np.float32)
        if (
            observation.shape != (self.policy.observation_dim,)
            or not np.isfinite(observation).all()
        ):
            raise RuntimeError(f"Invalid Student observation: {observation.shape}")
        return observation

    def step(
        self,
        robot: RobotState,
        ball: BallState,
        goal: TaskGoal,
        time_remaining_s: float,
        global_root_position: np.ndarray | None = None,
        projected_gravity_b: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observation = self.build_observation(
            robot,
            ball,
            goal,
            time_remaining_s,
            global_root_position,
            projected_gravity_b,
        )
        action = self.policy.infer(observation)
        self.last_action[:] = action
        joint_target = self.policy.default_joint_position + self.policy.action_scale * action
        return joint_target, action.copy(), observation


class SweetSpotVelocityEstimator:
    """Estimate collider-center site velocity using deployment MuJoCo FK."""

    def __init__(
        self,
        robot_xml: str | Path,
        joint_names: tuple[str, ...],
        site_name: str = "racket_sweet_spot",
        smoothing: float = 0.35,
        position_offset_b: np.ndarray | None = None,
    ) -> None:
        import mujoco

        if not 0.0 < smoothing <= 1.0:
            raise ValueError("sweet-spot velocity smoothing must be in (0, 1].")
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(robot_xml).resolve()))
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Kinematics XML has no site {site_name!r}.")
        joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in joint_names
            ],
            dtype=np.int32,
        )
        if np.any(joint_ids < 0):
            missing = [name for name, joint_id in zip(joint_names, joint_ids) if joint_id < 0]
            raise ValueError(f"Kinematics XML is missing policy joints: {missing}")
        self.joint_qpos_addresses = self.model.jnt_qposadr[joint_ids].copy()
        self.joint_dof_addresses = self.model.jnt_dofadr[joint_ids].copy()
        root_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        if root_joint_id < 0:
            raise ValueError("Kinematics XML has no floating_base_joint.")
        self.root_body_id = int(self.model.jnt_bodyid[root_joint_id])
        self.position_offset_b = np.zeros(3, dtype=np.float64)
        if position_offset_b is not None:
            self.position_offset_b[:] = _vector(
                position_offset_b, 3, "sweet_spot_position_offset_b"
            )
        self.smoothing = smoothing
        self._previous_position: np.ndarray | None = None
        self._previous_time_s: float | None = None
        self._position = np.zeros(3, dtype=np.float64)
        self._velocity = np.zeros(3, dtype=np.float64)

    @property
    def position_w(self) -> np.ndarray:
        return self._position.copy()

    def reset(self) -> None:
        self._previous_position = None
        self._previous_time_s = None
        self._position.fill(0.0)
        self._velocity.fill(0.0)

    def relative_velocity_w(self, joint_velocity: np.ndarray) -> np.ndarray:
        """Instantaneous root-relative velocity, expressed in world axes.

        Zero floating-base columns remove both root translation and omega cross r.
        Call update first so the Jacobian uses the current measured joint pose.
        """
        jacobian = np.zeros((3, self.model.nv))
        self._mujoco.mj_jacSite(self.model, self.data, jacobian, None, self.site_id)
        return jacobian[:, self.joint_dof_addresses] @ _vector(
            joint_velocity, ACTION_DIM, "joint_velocity"
        )

    def update(
        self,
        root_position_w: np.ndarray,
        root_orientation_wxyz: np.ndarray,
        joint_position: np.ndarray,
        time_s: float,
    ) -> np.ndarray:
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite.")
        self.data.qpos[:3] = _vector(root_position_w, 3, "root_position_w")
        self.data.qpos[3:7] = normalize_quaternion_wxyz(root_orientation_wxyz)
        self.data.qpos[self.joint_qpos_addresses] = _vector(
            joint_position, ACTION_DIM, "joint_position"
        )
        self._mujoco.mj_forward(self.model, self.data)
        root_rotation_wb = self.data.xmat[self.root_body_id].reshape(3, 3)
        position = (
            self.data.site_xpos[self.site_id]
            + root_rotation_wb @ self.position_offset_b
        ).copy()
        self._position[:] = position
        if self._previous_position is not None and self._previous_time_s is not None:
            dt = time_s - self._previous_time_s
            if 1.0e-4 <= dt <= 0.2:
                raw_velocity = (position - self._previous_position) / dt
                self._velocity = (
                    self.smoothing * raw_velocity
                    + (1.0 - self.smoothing) * self._velocity
                )
        self._previous_position = position
        self._previous_time_s = time_s
        return self._velocity.copy()
