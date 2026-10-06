"""Thread-safe deployment safety helpers."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


UNITREE_REMOTE_A = 0x0100
UNITREE_REMOTE_B = 0x0200
UNITREE_REMOTE_X = 0x0400
UNITREE_REMOTE_Y = 0x0800


def unitree_remote_fsm_transition(
    current_state: str,
    keys: int,
    previous_keys: int,
) -> str:
    """Map G1 wireless-controller button edges to the deployment FSM."""
    if current_state not in ("init", "damp", "home", "control"):
        raise ValueError(f"unsupported FSM state: {current_state}")
    keys = int(keys) & 0xFFFF
    previous_keys = int(previous_keys) & 0xFFFF
    if keys & UNITREE_REMOTE_B:
        return "damp"
    pressed = keys & ~previous_keys
    if pressed & UNITREE_REMOTE_A and (current_state == "damp" or current_state == "control"):
        return "home"
    if pressed & UNITREE_REMOTE_X and current_state == "home":
        return "control"
    return current_state


@dataclass(frozen=True)
class CommandWatchdogResult:
    """Decision returned to the hardware command loop."""

    force_damp: bool
    fault_latched: bool
    fault_just_latched: bool
    command_age_s: float | None


class CommandWatchdog:
    """Require a fresh command after every transition into control.

    During the initial grace period the hardware remains in damping. If no
    command arrives before the timeout, or an active stream becomes stale, the
    fault latches. A latched fault is cleared only by an explicit ``damp`` FSM
    message, forcing the operator to repeat the home/control sequence.
    """

    def __init__(self, timeout_s: float) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("command timeout must be positive")
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._fsm_state = "init"
        self._control_entry_s: float | None = None
        self._last_command_s: float | None = None
        self._fault_latched = False

    def update_fsm(self, state: str, now_s: float | None = None) -> bool:
        """Record an FSM request and return True when ``damp`` clears a fault."""
        if state not in ("init", "damp", "home", "control"):
            raise ValueError(f"unsupported FSM state: {state}")
        now = time.monotonic() if now_s is None else float(now_s)
        with self._lock:
            if state != self._fsm_state:
                self._fsm_state = state
                self._control_entry_s = now if state == "control" else None
            cleared = state == "damp" and self._fault_latched
            if cleared:
                self._fault_latched = False
            return cleared

    def record_command(self, now_s: float | None = None) -> None:
        now = time.monotonic() if now_s is None else float(now_s)
        with self._lock:
            self._last_command_s = now

    def evaluate(self, now_s: float | None = None) -> CommandWatchdogResult:
        now = time.monotonic() if now_s is None else float(now_s)
        with self._lock:
            command_age = (
                None
                if self._last_command_s is None
                else max(0.0, now - self._last_command_s)
            )
            if self._fsm_state != "control":
                return CommandWatchdogResult(False, self._fault_latched, False, command_age)
            if self._fault_latched:
                return CommandWatchdogResult(True, True, False, command_age)

            entry = self._control_entry_s
            has_fresh_control_command = (
                entry is not None
                and self._last_command_s is not None
                and self._last_command_s >= entry
                and command_age is not None
                and command_age <= self.timeout_s
            )
            if has_fresh_control_command:
                return CommandWatchdogResult(False, False, False, command_age)

            grace_expired = entry is None or now - entry >= self.timeout_s
            if grace_expired:
                self._fault_latched = True
                return CommandWatchdogResult(True, True, True, command_age)
            return CommandWatchdogResult(True, False, False, command_age)
