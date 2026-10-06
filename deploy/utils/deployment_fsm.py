"""Backend-rate deployment FSM shared by MuJoCo and Unitree execution."""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence

import numpy as np

FSM_STATES = ("damp", "home", "control")
COMMAND_PARTS = 5


class DeploymentFSM:
    """Generate damp, home, or policy commands without per-step allocations."""

    def __init__(
        self,
        *,
        default_joint_position: Sequence[float],
        home_kp: Sequence[float],
        home_kd: Sequence[float],
        home_duration_s: float,
        command_timeout_s: float,
        damp_kd: float = 3.0,
    ) -> None:
        self.default_joint_position = self._vector(
            default_joint_position, "default_joint_position"
        )
        self.joint_count = int(self.default_joint_position.size)
        self.home_kp = self._vector(home_kp, "home_kp")
        self.home_kd = self._vector(home_kd, "home_kd")
        if (
            self.home_kp.size != self.joint_count
            or self.home_kd.size != self.joint_count
        ):
            raise ValueError("home gains must match the joint count")
        if not math.isfinite(home_duration_s) or home_duration_s <= 0.0:
            raise ValueError("home_duration_s must be positive")
        if not math.isfinite(command_timeout_s) or command_timeout_s <= 0.0:
            raise ValueError("command_timeout_s must be positive")
        if not math.isfinite(damp_kd) or damp_kd < 0.0:
            raise ValueError("damp_kd must be finite and non-negative")

        self.home_duration_s = float(home_duration_s)
        self.command_timeout_s = float(command_timeout_s)
        self.damp_kd = float(damp_kd)
        self._lock = threading.Lock()

        self._state = "damp"
        self._pending_request: str | None = None
        self._state_entry_s = 0.0
        self._clock_initialized = False
        self._home_complete = False
        self._home_start_position = self.default_joint_position.copy()
        self._last_policy_command_s: float | None = None
        self._control_entry_s: float | None = None
        self._recoverable_fault = False
        self._hard_fault = False
        self._fault_reason: str | None = None

        self._policy_q = self.default_joint_position.copy()
        self._policy_dq = np.zeros(self.joint_count, dtype=np.float64)
        self._policy_kp = self.home_kp.copy()
        self._policy_kd = self.home_kd.copy()
        self._policy_tau = np.zeros(self.joint_count, dtype=np.float64)

        self.q_target = np.zeros(self.joint_count, dtype=np.float64)
        self.dq_target = np.zeros(self.joint_count, dtype=np.float64)
        self.kp = np.zeros(self.joint_count, dtype=np.float64)
        self.kd = np.full(self.joint_count, self.damp_kd, dtype=np.float64)
        self.tau_ff = np.zeros(self.joint_count, dtype=np.float64)

    @staticmethod
    def _vector(value: Sequence[float], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite one-dimensional vector")
        return array.copy()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def state_entry_s(self) -> float:
        with self._lock:
            return self._state_entry_s

    @property
    def home_complete(self) -> bool:
        with self._lock:
            return self._home_complete

    @property
    def fault_reason(self) -> str | None:
        with self._lock:
            return self._fault_reason

    def request(self, requested_state: str) -> None:
        """Queue an operator request; damp has priority over all other requests."""
        if requested_state not in FSM_STATES:
            raise ValueError(f"unsupported FSM request: {requested_state}")
        with self._lock:
            if requested_state == "damp" or self._pending_request != "damp":
                self._pending_request = requested_state

    def update_policy_command(self, command: Sequence[float], *, now_s: float) -> None:
        """Store the latest 145-D style command from the 50 Hz policy process."""
        now = float(now_s)
        command_array = np.asarray(command, dtype=np.float64)
        expected = COMMAND_PARTS * self.joint_count
        if command_array.shape != (expected,):
            raise ValueError(f"policy command must contain {expected} values")
        if not np.isfinite(command_array).all() or not math.isfinite(now):
            raise ValueError("policy command and timestamp must be finite")
        n = self.joint_count
        with self._lock:
            np.copyto(self._policy_q, command_array[0:n])
            np.copyto(self._policy_dq, command_array[n : 2 * n])
            np.copyto(self._policy_kp, command_array[2 * n : 3 * n])
            np.copyto(self._policy_kd, command_array[3 * n : 4 * n])
            np.copyto(self._policy_tau, command_array[4 * n : 5 * n])
            self._last_policy_command_s = now

    def force_damp(self, reason: str, *, now_s: float, hard: bool = False) -> None:
        """Immediately force damp; hard faults cannot be cleared without restart."""
        if not reason:
            raise ValueError("forced damp requires a reason")
        with self._lock:
            if not self._clock_initialized:
                self._state_entry_s = float(now_s)
                self._clock_initialized = True
            self._hard_fault |= bool(hard)
            self._recoverable_fault |= not hard
            self._fault_reason = str(reason)
            self._pending_request = None
            self._transition_to_damp(float(now_s))
            self._write_damp_command()

    def step(self, joint_position: Sequence[float], *, now_s: float) -> str:
        """Advance the FSM once at the backend execution frequency."""
        now = float(now_s)
        position = np.asarray(joint_position, dtype=np.float64)
        if position.shape != (self.joint_count,) or not np.isfinite(position).all():
            raise ValueError("joint_position has an invalid shape or value")
        if not math.isfinite(now):
            raise ValueError("FSM time must be finite")

        with self._lock:
            if not self._clock_initialized:
                self._state_entry_s = now
                self._clock_initialized = True
            self._apply_pending_request(position, now)

            if self._hard_fault:
                self._transition_to_damp(now)

            if self._state == "damp":
                self._write_damp_command()
            elif self._state == "home":
                self._write_home_command(now)
            else:
                self._write_control_command(now)
            return self._state

    def _apply_pending_request(self, position: np.ndarray, now: float) -> None:
        request = self._pending_request
        self._pending_request = None
        if request is None:
            return
        if request == "damp":
            if not self._hard_fault:
                self._recoverable_fault = False
                self._fault_reason = None
            self._transition_to_damp(now)
            return
        if self._hard_fault or self._recoverable_fault:
            return
        if request == "home" and self._state == "damp":
            self._state = "home"
            self._state_entry_s = now
            self._home_complete = False
            np.copyto(self._home_start_position, position)
            return
        if request == "control" and self._state == "home" and self._home_complete:
            self._state = "control"
            self._state_entry_s = now
            self._control_entry_s = now

    def _transition_to_damp(self, now: float) -> None:
        if self._state != "damp":
            self._state = "damp"
            self._state_entry_s = now
        self._home_complete = False
        self._control_entry_s = None

    def _write_damp_command(self) -> None:
        self.q_target.fill(0.0)
        self.dq_target.fill(0.0)
        self.kp.fill(0.0)
        self.kd.fill(self.damp_kd)
        self.tau_ff.fill(0.0)

    def _write_home_command(self, now: float) -> None:
        ratio = min(max((now - self._state_entry_s) / self.home_duration_s, 0.0), 1.0)
        np.subtract(
            self.default_joint_position,
            self._home_start_position,
            out=self.q_target,
        )
        self.q_target *= ratio
        self.q_target += self._home_start_position
        self.dq_target.fill(0.0)
        np.multiply(self.home_kp, ratio, out=self.kp)
        np.multiply(self.home_kd, ratio, out=self.kd)
        self.tau_ff.fill(0.0)
        self._home_complete = ratio >= 1.0

    def _write_control_command(self, now: float) -> None:
        command_is_fresh = (
            self._control_entry_s is not None
            and self._last_policy_command_s is not None
            and self._last_policy_command_s >= self._control_entry_s
            and now - self._last_policy_command_s <= self.command_timeout_s
        )
        if command_is_fresh:
            np.copyto(self.q_target, self._policy_q)
            np.copyto(self.dq_target, self._policy_dq)
            np.copyto(self.kp, self._policy_kp)
            np.copyto(self.kd, self._policy_kd)
            np.copyto(self.tau_ff, self._policy_tau)
            return

        control_age = now - float(self._control_entry_s)
        if control_age >= self.command_timeout_s:
            self._recoverable_fault = True
            self._fault_reason = "policy_command_timeout"
            self._transition_to_damp(now)
        self._write_damp_command()
