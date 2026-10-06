"""Observation builder for 628-D, global-root 631-D and robust 638-D Students."""

from __future__ import annotations

from collections import deque

import numpy as np

from utils.tppo_student import (
    ACTION_DIM,
    BallState,
    RobotState,
    _vector,
    quaternion_to_rotation_matrix_wxyz,
)


INTENT_CURRENT_OBSERVATION_DIM = 133
INTENT_STATE_TOKEN_DIM = 93
INTENT_STATE_HISTORY_STEPS = 5
INTENT_BALL_TOKEN_DIM = 6
INTENT_ROBUST_BALL_TOKEN_DIM = 8
INTENT_BALL_HISTORY_STEPS = 5
INTENT_HISTORY_BUFFER_LENGTH = 25
INTENT_HISTORY_LAGS = (20, 15, 10, 5, 0)

INTENT_OBSERVATION_DIM = (
    INTENT_CURRENT_OBSERVATION_DIM
    + INTENT_STATE_HISTORY_STEPS * INTENT_STATE_TOKEN_DIM
    + INTENT_BALL_HISTORY_STEPS * INTENT_BALL_TOKEN_DIM
)
INTENT_ROBUST_OBSERVATION_DIM = (
    INTENT_CURRENT_OBSERVATION_DIM
    + INTENT_STATE_HISTORY_STEPS * INTENT_STATE_TOKEN_DIM
    + INTENT_BALL_HISTORY_STEPS * INTENT_ROBUST_BALL_TOKEN_DIM
)

INTENT_OBSERVATION_SLICES = {
    "base_ang_vel": slice(0, 3),
    "joint_pos": slice(3, 32),
    "joint_vel": slice(32, 61),
    "last_action": slice(61, 90),
    "landing_target": slice(90, 92),
    "ball_position": slice(92, 107),
    "ball_velocity": slice(107, 122),
    "sweet_spot_velocity": slice(122, 125),
    "projected_gravity": slice(125, 128),
    "sweet_spot_position": slice(128, 131),
    "startup_heading": slice(131, 133),
    "state_history": slice(133, 598),
    "intent_ball_history": slice(598, 628),
}
INTENT_ROBUST_OBSERVATION_SLICES = {
    **INTENT_OBSERVATION_SLICES,
    "intent_ball_history": slice(598, 628),
    "ball_observation_valid_history": slice(628, 633),
    "ball_observation_age_history": slice(633, 638),
}
INTENT_GLOBAL_ROOT_OBSERVATION_SLICES = {
    **INTENT_OBSERVATION_SLICES,
    "global_root_pos": slice(133, 136),
    "state_history": slice(136, 601),
    "intent_ball_history": slice(601, 631),
}


def gather_strided_history(
    history: deque[np.ndarray],
    sample_lags: tuple[int, ...],
) -> np.ndarray:
    """Match the training CircularBuffer order and initial backfill."""
    if not history:
        raise RuntimeError("Cannot gather an empty observation history.")

    values = tuple(history)
    newest = len(values) - 1
    return np.concatenate(
        [values[max(0, newest - lag)] for lag in sample_lags]
    )


class IntentObservationBuilder:
    """Maintain policy-step history for an Intent observation."""

    def __init__(
        self,
        default_joint_position: np.ndarray,
        landing_target_startup: np.ndarray,
        *,
        buffer_length: int = INTENT_HISTORY_BUFFER_LENGTH,
        history_lags: tuple[int, ...] = INTENT_HISTORY_LAGS,
        include_ball_observation_status: bool = False,
        include_global_root_pos: bool = False,
        landing_target_observation_frame: str = "startup",
    ) -> None:
        self.default_joint_position = _vector(
            default_joint_position, ACTION_DIM, "default_joint_position"
        )
        self.landing_target_startup = _vector(
            landing_target_startup, 2, "landing_target_startup"
        )
        self.history_lags = tuple(int(lag) for lag in history_lags)
        self.include_ball_observation_status = bool(include_ball_observation_status)
        self.include_global_root_pos = bool(include_global_root_pos)
        if landing_target_observation_frame not in {"startup", "current_root"}:
            raise ValueError("Invalid landing_target_observation_frame")
        self.landing_target_observation_frame = landing_target_observation_frame
        self.landing_target_w = np.zeros(3)
        if self.include_global_root_pos and self.include_ball_observation_status:
            raise ValueError("Combined global-root and robust Intent input is not supported.")

        if len(self.history_lags) not in (5, 8, 12):
            raise ValueError("Intent history_lags must contain 5, 8 or 12 values.")
        if len(self.history_lags) != 5 and (not include_global_root_pos or include_ball_observation_status):
            raise ValueError("Long history is supported only for M14-11 measurement policies.")
        if min(self.history_lags) < 0:
            raise ValueError("Intent history lags must be non-negative.")
        if max(self.history_lags) >= buffer_length:
            raise ValueError("Intent history lag exceeds buffer length.")

        self.state_history: deque[np.ndarray] = deque(maxlen=buffer_length)
        self.ball_position_history_b: deque[np.ndarray] = deque(maxlen=buffer_length)
        self.ball_velocity_history_b: deque[np.ndarray] = deque(maxlen=buffer_length)
        self.ball_position_history_w: deque[np.ndarray] = deque(maxlen=buffer_length)
        self.ball_velocity_history_w: deque[np.ndarray] = deque(maxlen=buffer_length)
        self.ball_valid_history: deque[np.ndarray] = deque(maxlen=buffer_length)
        self.ball_age_history: deque[np.ndarray] = deque(maxlen=buffer_length)

        self.startup_forward_w = np.array([1.0, 0.0, 0.0])
        self.initialized = False

    def reset(self, startup_orientation_wxyz: np.ndarray,
              startup_position_w: np.ndarray | None = None) -> None:
        startup_rotation_wb = quaternion_to_rotation_matrix_wxyz(
            startup_orientation_wxyz
        )
        startup_forward_w = startup_rotation_wb[:, 0].copy()
        startup_forward_w[2] = 0.0

        norm = float(np.linalg.norm(startup_forward_w[:2]))
        if norm < 1.0e-8:
            raise ValueError("Invalid startup heading.")

        self.startup_forward_w = startup_forward_w / norm
        forward = self.startup_forward_w
        left = np.array([-forward[1], forward[0], 0.0])
        origin = np.zeros(3) if startup_position_w is None else _vector(
            startup_position_w, 3, "startup_position_w"
        )
        self.landing_target_w = (
            origin + forward * self.landing_target_startup[0]
            + left * self.landing_target_startup[1]
        )
        self.landing_target_w[2] = 0.0

        self.state_history.clear()
        self.ball_position_history_b.clear()
        self.ball_velocity_history_b.clear()
        self.ball_position_history_w.clear()
        self.ball_velocity_history_w.clear()
        self.ball_valid_history.clear()
        self.ball_age_history.clear()
        self.initialized = True

    def reset_ball_history(self) -> None:
        """Discard observations from the previous physical ball track."""
        self.ball_position_history_b.clear()
        self.ball_velocity_history_b.clear()
        self.ball_position_history_w.clear()
        self.ball_velocity_history_w.clear()
        self.ball_valid_history.clear()
        self.ball_age_history.clear()

    def _record_history(
        self,
        robot: RobotState,
        ball: BallState,
        projected_gravity_b: np.ndarray,
        last_action: np.ndarray,
        ball_observation_valid: bool,
        ball_observation_age_s: float,
    ) -> None:
        if not self.initialized:
            raise RuntimeError("Intent observation builder must be reset first.")

        rotation_wb = quaternion_to_rotation_matrix_wxyz(
            robot.root_orientation_wxyz
        )
        rotation_bw = rotation_wb.T
        gravity_b = _vector(projected_gravity_b, 3, "projected_gravity_b")
        action = _vector(last_action, ACTION_DIM, "last_action")
        joint_pos_rel = robot.joint_position - self.default_joint_position

        state_token = np.concatenate(
            [
                robot.base_angular_velocity_b,
                gravity_b,
                joint_pos_rel,
                robot.joint_velocity,
                action,
            ]
        )
        if state_token.shape != (INTENT_STATE_TOKEN_DIM,):
            raise RuntimeError(f"Invalid state token: {state_token.shape}")

        self.state_history.append(state_token)
        self.ball_position_history_b.append(
            rotation_bw @ (ball.position_w - robot.root_position_w)
        )
        self.ball_velocity_history_b.append(rotation_bw @ ball.velocity_w)
        self.ball_position_history_w.append(ball.position_w.copy())
        self.ball_velocity_history_w.append(ball.velocity_w.copy())
        self.ball_valid_history.append(
            np.array([float(ball_observation_valid)], dtype=np.float64)
        )
        self.ball_age_history.append(
            np.array([float(ball_observation_age_s)], dtype=np.float64)
        )

    def _startup_heading_b(self, rotation_wb: np.ndarray) -> np.ndarray:
        """Startup +X expressed in the current root yaw frame."""
        yaw = float(np.arctan2(rotation_wb[1, 0], rotation_wb[0, 0]))
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_yaw_bw = np.array(
            [
                [cos_yaw, sin_yaw, 0.0],
                [-sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return (rotation_yaw_bw @ self.startup_forward_w)[:2]

    def build_observation(
        self,
        robot: RobotState,
        ball: BallState,
        projected_gravity_b: np.ndarray,
        last_action: np.ndarray,
        sweet_spot_position_w: np.ndarray,
        ball_observation_valid: bool = True,
        ball_observation_age_s: float = 0.0,
        sweet_spot_relative_velocity_b: np.ndarray | None = None,
    ) -> np.ndarray:
        gravity_b = _vector(projected_gravity_b, 3, "projected_gravity_b")
        action = _vector(last_action, ACTION_DIM, "last_action")
        sweet_position_w = _vector(
            sweet_spot_position_w, 3, "sweet_spot_position_w"
        )

        if ball_observation_age_s < 0.0 or not np.isfinite(ball_observation_age_s):
            raise ValueError("ball_observation_age_s must be finite and non-negative.")
        self._record_history(
            robot,
            ball,
            gravity_b,
            action,
            bool(ball_observation_valid),
            float(ball_observation_age_s),
        )

        rotation_wb = quaternion_to_rotation_matrix_wxyz(
            robot.root_orientation_wxyz
        )
        rotation_bw = rotation_wb.T
        joint_pos_rel = robot.joint_position - self.default_joint_position

        ball_position_history_b = gather_strided_history(
            self.ball_position_history_b, INTENT_HISTORY_LAGS
        )
        ball_velocity_history_b = gather_strided_history(
            self.ball_velocity_history_b, INTENT_HISTORY_LAGS
        )

        sweet_spot_position_b = rotation_bw @ (
            sweet_position_w - robot.root_position_w
        )
        sweet_spot_velocity_b = rotation_bw @ robot.sweet_spot_velocity_w
        if sweet_spot_relative_velocity_b is not None:
            sweet_spot_velocity_b = _vector(
                sweet_spot_relative_velocity_b, 3, "sweet_spot_relative_velocity_b"
            )
        landing_target = self.landing_target_startup
        if self.landing_target_observation_frame == "current_root":
            landing_target = (rotation_bw @ (self.landing_target_w - robot.root_position_w))[:2]

        current_student = np.concatenate(
            [
                robot.base_angular_velocity_b,
                joint_pos_rel,
                robot.joint_velocity,
                action,
                landing_target,
                ball_position_history_b,
                ball_velocity_history_b,
                sweet_spot_velocity_b,
                gravity_b,
                sweet_spot_position_b,
                self._startup_heading_b(rotation_wb),
            ]
        )

        if self.include_global_root_pos:
            # Training appends world XYZ to current Student, before both histories.
            current_student = np.concatenate([current_student, robot.root_position_w])

        state_history = gather_strided_history(
            self.state_history, self.history_lags
        )

        positions_w = gather_strided_history(
            self.ball_position_history_w, self.history_lags
        ).reshape(len(self.history_lags), 3)
        velocities_w = gather_strided_history(
            self.ball_velocity_history_w, self.history_lags
        ).reshape(len(self.history_lags), 3)

        intent_ball_history = np.concatenate(
            [
                (rotation_bw @ (positions_w - robot.root_position_w).T)
                .T.reshape(-1),
                (rotation_bw @ velocities_w.T).T.reshape(-1),
            ]
        )
        if self.include_ball_observation_status:
            intent_ball_history = np.concatenate(
                [
                    intent_ball_history,
                    gather_strided_history(
                        self.ball_valid_history, self.history_lags
                    ),
                    gather_strided_history(
                        self.ball_age_history, self.history_lags
                    ),
                ]
            )

        observation = np.concatenate(
            [current_student, state_history, intent_ball_history]
        ).astype(np.float32)

        expected_dim = (
            INTENT_ROBUST_OBSERVATION_DIM
            if self.include_ball_observation_status
            else INTENT_OBSERVATION_DIM
        )
        if self.include_global_root_pos:
            expected_dim += 3
        expected_dim += (len(self.history_lags) - 5) * 99
        if observation.shape != (expected_dim,):
            raise RuntimeError(f"Invalid Intent observation: {observation.shape}")
        if not np.isfinite(observation).all():
            raise RuntimeError("Intent observation contains NaN or Inf.")
        return observation
