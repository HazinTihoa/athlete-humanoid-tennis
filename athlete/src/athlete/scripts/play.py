"""Script to play RL agent with RSL-RL."""

import os
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mujoco
import torch
import tyro
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from athlete.goal_cond_tracking.mdp import MultiTargetMotionCommandCfg
from athlete.goal_cond_tracking.mdp.phase_actions import (
  PhaseAccelerationAction,
  PhaseResidualAction,
)
from athlete.goal_cond_tracking.mdp.phase_commands import (
  PhaseAccelerationMultiTargetMotionCommand,
  PhaseAwareMultiTargetMotionCommand,
)
from athlete.scripts.fast_play_env import FastPlayManagerBasedRlEnv
from athlete.scripts.onnx_policy import OnnxPlayPolicy
from athlete.scripts.tennis_court_spec import (
  RACKET_COLLISION_HALF_SIZE_M,
  RACKET_COLLISION_POS_WRIST_M,
  RACKET_COLLISION_QUAT_WXYZ,
)
from athlete.scripts.tennis_scene import (
  add_racket_ball_collision,
  install_tennis_ball_aerodynamics,
  install_tennis_ball_controller,
  tennis_ball_spec,
)


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  motion_config: Path | None = None
  """Path to a motion set TOML. Resolves eval registry and robot XML when provided."""
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  onnx_policy_file: Path | None = None
  """Use an exported ONNX policy instead of loading an RSL-RL checkpoint."""
  motion_file: str | None = None
  num_envs: int | None = None
  seed: int | None = None
  """Optional deterministic environment seed for reproducible Play comparisons."""
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  ghost_reference_world: bool = False
  """Align reference frame 0 to the robot in XYZ/yaw, then preserve its trajectory."""
  physical_ball: bool = False
  """Add a play-only physical tennis ball at the sampled position target."""
  ball_release_lead_s: float = 0.02
  """Release the held ball this many seconds before the sampled strike deadline."""
  phase_plot: bool = True
  """Open a live phase-state plot for phase-aware tasks with the native viewer."""
  phase_plot_history_s: float = 10.0
  """Maximum simulation-time history retained by the live phase plot."""
  phase_debug: bool = False
  """Print per-frame contact-time and periodic phase-rate diagnostics."""
  fast_play: bool = True
  """Use an inference-only env step that skips critic, rewards, and log metrics."""
  tppo_phase_source: Literal["teacher", "zero"] = "teacher"
  """Source for TPPO environment-only phase actions during Play."""
  print_student_goal: bool = False
  """Print the sampled Student task goal and time remaining at startup."""
  ball_impact_debug: bool = False
  """Print per-impact ball and racket velocity diagnostics without changing physics."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _tennis_ball_spec() -> mujoco.MjSpec:
  return tennis_ball_spec()


def _add_racket_ball_collision(spec_fn):
  """Wrap a robot spec factory with an invisible racket-head collision geom."""

  def build_spec() -> mujoco.MjSpec:
    spec = spec_fn()
    wrist = spec.body("right_wrist_yaw_link")
    if wrist is None:
      raise ValueError("Physical ball requires body right_wrist_yaw_link.")
    wrist.add_geom(
      name="racket_ball_collision",
      type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
      pos=RACKET_COLLISION_POS_WRIST_M,
      # Local Z is aligned with the compiled visual mesh face normal.
      quat=RACKET_COLLISION_QUAT_WXYZ,
      size=RACKET_COLLISION_HALF_SIZE_M,
      rgba=[0.0, 0.0, 0.0, 0.0],
      density=0.0,
      contype=1,
      conaffinity=1,
    )
    return spec

  return build_spec


class _PlayTennisBallController:
  """Hold the ball at the sampled target, then release it before impact."""

  def __init__(self, env: ManagerBasedRlEnv, release_lead_s: float) -> None:
    self.env = env
    self.ball = env.scene["tennis_ball"]
    self.motion = env.command_manager.get_term("motion")
    self.release_lead_s = release_lead_s
    self.release_lead_steps = max(0, round(release_lead_s / env.step_dt))
    self._deadline_aware = hasattr(self.motion, "time_remaining") and hasattr(
      self.motion, "contact_time"
    )
    self._previous_time_steps = torch.full_like(self.motion.time_steps, -1)
    self._previous_time_remaining = (
      torch.full_like(self.motion.time_remaining, torch.inf)
      if self._deadline_aware
      else None
    )
    self._previous_motion_ids = torch.full_like(self.motion.which_motion, -1)
    self._released = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def before_step(self) -> None:
    time_steps = self.motion.time_steps
    motion_ids = self.motion.which_motion
    if self._deadline_aware:
      time_remaining = self.motion.time_remaining
      assert self._previous_time_remaining is not None
      restarted = (
        time_remaining > self._previous_time_remaining + 0.5 * self.env.step_dt
      ) | (motion_ids != self._previous_motion_ids)
    else:
      time_remaining = None
      restarted = (time_steps < self._previous_time_steps) | (
        motion_ids != self._previous_motion_ids
      )
    self._released[restarted] = False
    restarted_env_ids = set(torch.where(restarted)[0].tolist())

    target_indices = []
    release_steps = []
    for env_id, motion_id in enumerate(motion_ids.tolist()):
      cfg = self.motion.motion_configs[motion_id]
      target_index = next(
        i for i, target in enumerate(cfg.sub_targets) if target.goal_type == "position"
      )
      target_indices.append(target_index)
      total = int(self.motion._time_step_totals[motion_id].item())
      target_cfg = cfg.sub_targets[target_index]
      if env_id in restarted_env_ids:
        strike_phase = (
          target_cfg.target_phase_start + target_cfg.target_phase_end
        ) / 2.0
        strike_frame = round(strike_phase * (total - 1))
        contact_time_s = (
          float(self.motion.contact_time[env_id].item())
          if self._deadline_aware
          else strike_frame * self.env.step_dt
        )
        motion_file = Path(self.motion.cfg.motion_files[motion_id]).name
        print(
          f"[PLAY SAMPLE] env={env_id} motion_id={motion_id} "
          f"file={motion_file} strike_frame={strike_frame} "
          f"nominal_strike_time={contact_time_s:.3f}s total_frames={total}",
          flush=True,
        )
      if not self._deadline_aware:
        release_steps.append(
          max(
            0,
            round(target_cfg.target_phase_start * (total - 1))
            - self.release_lead_steps,
          )
        )

    target_indices_t = torch.tensor(target_indices, device=self.env.device)
    if self._deadline_aware:
      assert time_remaining is not None
      self._released |= time_remaining <= self.release_lead_s
    else:
      release_steps_t = torch.tensor(release_steps, device=self.env.device)
      self._released |= time_steps >= release_steps_t
    held = ~self._released
    if torch.any(held):
      env_ids = torch.where(held)[0]
      target_pos = self.motion.target_position_w[env_ids, target_indices_t[env_ids]]
      state = torch.zeros((len(env_ids), 13), device=self.env.device)
      state[:, :3] = target_pos
      state[:, 3] = 1.0
      self.ball.write_root_state_to_sim(state, env_ids=env_ids)

    self._previous_time_steps.copy_(time_steps)
    if self._deadline_aware:
      assert self._previous_time_remaining is not None
      assert time_remaining is not None
      self._previous_time_remaining.copy_(time_remaining)
    self._previous_motion_ids.copy_(motion_ids)


class _BallImpactDebugController:
  """Report single-step ball velocity changes near the racket sweet point."""

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self.env = env
    self.ball = env.scene["tennis_ball"]
    self.robot = env.scene["robot"]
    self.motion = env.command_manager.get_term("motion")
    wrist_ids, _ = self.robot.find_bodies(
      "right_wrist_yaw_link", preserve_order=True
    )
    if len(wrist_ids) != 1:
      raise ValueError("Ball impact diagnostics require right_wrist_yaw_link.")
    self._wrist_body_id = wrist_ids[0]
    self._ball_position_before = torch.empty((0, 3), device=env.device)
    self._ball_velocity_before = torch.empty((0, 3), device=env.device)
    self._sweet_position_before = torch.empty((0, 3), device=env.device)
    self._sweet_velocity_before = torch.empty((0, 3), device=env.device)
    self._wrist_position_before = torch.empty((0, 3), device=env.device)
    self._wrist_linear_velocity_before = torch.empty((0, 3), device=env.device)
    self._wrist_angular_velocity_before = torch.empty((0, 3), device=env.device)
    self._time_remaining_before = torch.empty(0, device=env.device)
    self._event_count = 0

  def before_step(self) -> None:
    self._ball_position_before = self.ball.data.root_link_pos_w.clone()
    self._ball_velocity_before = self.ball.data.root_link_lin_vel_w.clone()
    self._sweet_position_before = self.motion.get_source_pos_w()[:, 0].clone()
    self._sweet_velocity_before = self.motion.get_source_lin_vel_w()[:, 0].clone()
    self._wrist_position_before = self.robot.data.body_link_pos_w[
      :, self._wrist_body_id
    ].clone()
    self._wrist_linear_velocity_before = self.robot.data.body_link_lin_vel_w[
      :, self._wrist_body_id
    ].clone()
    self._wrist_angular_velocity_before = self.robot.data.body_link_ang_vel_w[
      :, self._wrist_body_id
    ].clone()
    self._time_remaining_before = self.motion.time_remaining.clone()

  def after_step(self) -> None:
    ball_position_after = self.ball.data.root_link_pos_w
    ball_velocity_after = self.ball.data.root_link_lin_vel_w
    sweet_position_after = self.motion.get_source_pos_w()[:, 0]
    sweet_velocity_after = self.motion.get_source_lin_vel_w()[:, 0]

    velocity_change = torch.linalg.vector_norm(
      ball_velocity_after - self._ball_velocity_before, dim=-1
    )
    distance_before = torch.linalg.vector_norm(
      self._ball_position_before - self._sweet_position_before, dim=-1
    )
    distance_after = torch.linalg.vector_norm(
      ball_position_after - sweet_position_after, dim=-1
    )
    near_racket = torch.minimum(distance_before, distance_after) < 0.30
    impact = (velocity_change > 0.50) & near_racket

    for env_id in torch.where(impact)[0].tolist():
      self._event_count += 1
      ball_speed_before = torch.linalg.vector_norm(
        self._ball_velocity_before[env_id]
      ).item()
      ball_speed_after = torch.linalg.vector_norm(
        ball_velocity_after[env_id]
      ).item()
      sweet_speed = max(
        torch.linalg.vector_norm(self._sweet_velocity_before[env_id]).item(),
        torch.linalg.vector_norm(sweet_velocity_after[env_id]).item(),
      )
      relative_speed = torch.linalg.vector_norm(
        self._ball_velocity_before[env_id]
        - self._sweet_velocity_before[env_id]
      ).item()
      racket_point_velocity = self._wrist_linear_velocity_before[env_id] + torch.cross(
        self._wrist_angular_velocity_before[env_id],
        self._ball_position_before[env_id] - self._wrist_position_before[env_id],
        dim=-1,
      )
      racket_point_speed = torch.linalg.vector_norm(racket_point_velocity).item()
      wrist_angular_speed = torch.linalg.vector_norm(
        self._wrist_angular_velocity_before[env_id]
      ).item()
      min_distance = min(
        distance_before[env_id].item(), distance_after[env_id].item()
      )
      print(
        f"[BALL IMPACT DEBUG] event={self._event_count} env={env_id} "
        f"time_remaining={self._time_remaining_before[env_id].item():+.3f}s "
        f"distance={min_distance:.3f}m "
        f"ball_speed={ball_speed_before:.3f}->{ball_speed_after:.3f}m/s "
        f"delta_v={velocity_change[env_id].item():.3f}m/s "
        f"sweet_speed={sweet_speed:.3f}m/s "
        f"racket_point_speed={racket_point_speed:.3f}m/s "
        f"wrist_angular_speed={wrist_angular_speed:.3f}rad/s "
        f"relative_speed={relative_speed:.3f}m/s",
        flush=True,
      )


class _PhaseDebugController:
  """Report sampled timing and the policy-controlled phase rate during Play."""

  def __init__(self, env: ManagerBasedRlEnv, interval_s: float, env_id: int) -> None:
    if interval_s <= 0.0:
      raise ValueError("phase_debug_interval_s must be greater than zero.")
    if not 0 <= env_id < env.num_envs:
      raise ValueError(
        f"phase_debug_env_id must be in [0, {env.num_envs}), got {env_id}."
      )

    motion = env.command_manager.get_term("motion")
    if not isinstance(motion, PhaseAwareMultiTargetMotionCommand):
      raise TypeError("Phase debugging requires a phase-aware tracking task.")
    if isinstance(motion, PhaseAccelerationMultiTargetMotionCommand):
      phase_action = env.action_manager.get_term("phase_acceleration")
      if not isinstance(phase_action, PhaseAccelerationAction):
        raise TypeError("Phase debugging requires the phase-acceleration action.")
    else:
      phase_action = env.action_manager.get_term("phase_rate")
      if not isinstance(phase_action, PhaseResidualAction):
        raise TypeError("Phase debugging requires the phase-rate residual action.")

    self.env = env
    self.motion = motion
    self.phase_action = phase_action
    self.env_id = env_id
    self.interval_steps = max(1, round(interval_s / env.step_dt))
    self._steps_since_report = 0
    self._previous_episode_step = int(env.episode_length_buf[env_id].item())
    self._previous_phase = float(motion.phase[env_id].item())
    self._print_sample()

  def _print_sample(self) -> None:
    env_id = self.env_id
    motion_id = int(self.motion.which_motion[env_id].item())
    motion_file = Path(self.motion.cfg.motion_files[motion_id]).name
    print(
      f"[CONTACT TIME SAMPLE] env={env_id} motion_id={motion_id} file={motion_file} "
      f"sampled_contact_time={self.motion.contact_time[env_id].item():.3f}s "
      f"strike_phase={self.motion.strike_phase[env_id].item():.6f}",
      flush=True,
    )

  def after_step(self) -> None:
    env_id = self.env_id
    episode_step = int(self.env.episode_length_buf[env_id].item())
    phase = float(self.motion.phase[env_id].item())
    print(
      f"[CONTACT TIME FRAME] env={env_id} episode_step={episode_step} "
      f"sampled_contact_time={self.motion.contact_time[env_id].item():.3f}s "
      f"time_remaining={self.motion.time_remaining[env_id].item():.3f}s",
      flush=True,
    )
    restarted = (
      episode_step < self._previous_episode_step
      or phase < self._previous_phase - 1.0e-6
    )

    if restarted:
      self._steps_since_report = 0
      self._print_sample()
    else:
      self._steps_since_report += 1
      if self._steps_since_report >= self.interval_steps:
        prefix = (
          f"[PHASE RATE] env={env_id} episode_step={episode_step} "
          f"phase={phase:.6f} "
          f"elapsed_time="
          f"{(self.motion.contact_time[env_id] - self.motion.time_remaining[env_id]).item():.3f}s "
          f"time_remaining={self.motion.time_remaining[env_id].item():.3f}s "
        )
        if isinstance(self.motion, PhaseAccelerationMultiTargetMotionCommand):
          print(
            prefix
            + f"policy_acceleration={self.phase_action.raw_action[env_id, 0].item():+.6f} "
            + f"phase_acceleration={self.motion.phase_acceleration[env_id].item():+.6f} "
            + f"required_rate={self.motion.required_phase_rate[env_id].item():.6f} "
            + f"safe_max_rate={self.motion.safe_max_phase_rate[env_id].item():.6f} "
            + f"phase_rate={self.motion.phase_rate[env_id].item():.6f} "
            + f"projection={bool(self.motion.deadline_projection_active[env_id].item())}",
            flush=True,
          )
        else:
          print(
            prefix
            + f"policy_residual={self.phase_action.raw_action[env_id, 0].item():+.6f} "
            + f"schedule_rate={self.motion.phase_rate_schedule[env_id].item():.6f} "
            + f"phase_rate={self.motion.phase_rate[env_id].item():.6f}",
            flush=True,
          )
        self._steps_since_report = 0

    self._previous_episode_step = episode_step
    self._previous_phase = phase


class _PhasePlotController:
  """Plot env-0 phase state against simulation time during native Play."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    *,
    env_id: int,
    history_s: float,
    update_interval_s: float = 0.05,
  ) -> None:
    if history_s <= 0.0:
      raise ValueError("phase_plot_history_s must be greater than zero.")
    if update_interval_s <= 0.0:
      raise ValueError("Phase plot update interval must be greater than zero.")
    if not 0 <= env_id < env.num_envs:
      raise ValueError(f"Phase plot env_id {env_id} is out of range.")

    motion = env.command_manager.get_term("motion")
    if not isinstance(motion, PhaseAwareMultiTargetMotionCommand):
      raise TypeError("Phase plotting requires a phase-aware tracking task.")

    import matplotlib.pyplot as plt

    self.env = env
    self.motion = motion
    self.env_id = env_id
    self._plt = plt
    self._update_interval_s = update_interval_s
    self._steps_per_draw = max(1, round(update_interval_s / env.step_dt))
    self._steps_since_draw = 0
    self._episode_index = 0
    self._previous_episode_step = int(env.episode_length_buf[env_id].item())
    self._previous_phase = float(motion.phase[env_id].item())
    self._previous_rate = float(motion.phase_rate[env_id].item())

    max_points = max(2, round(history_s / env.step_dt) + 1)
    self._history: dict[str, deque[float]] = {
      name: deque(maxlen=max_points)
      for name in (
        "time",
        "phase",
        "phase_rate",
        "phase_acceleration",
        "time_remaining",
        "contact_time",
      )
    }

    plt.ion()
    figure, axes = plt.subplots(4, 1, figsize=(9.2, 8.4), sharex=True)
    self._figure = figure
    self._axes = tuple(axes)
    manager = getattr(figure.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
      manager.set_window_title("ATHLETE Phase State")
    window = getattr(manager, "window", None)
    if window is not None and hasattr(window, "wm_geometry"):
      window.wm_geometry("+20+60")

    self._lines = {
      "phase": axes[0].plot([], [], color="#2563eb", label="phase")[0],
      "phase_rate": axes[1].plot(
        [], [], color="#16a34a", label="phase rate [1/s]"
      )[0],
      "phase_acceleration": axes[2].plot(
        [], [], color="#ea580c", label="phase acceleration [1/s^2]"
      )[0],
      "time_remaining": axes[3].plot(
        [], [], color="#dc2626", label="time remaining [s]"
      )[0],
      "contact_time": axes[3].plot(
        [], [], color="#111827", linestyle="--", label="contact time [s]"
      )[0],
    }
    self._deadline_lines = [
      axis.axvline(0.0, color="#7c3aed", linestyle=":", linewidth=1.2)
      for axis in axes
    ]

    axes[0].set_ylabel("phase")
    axes[1].set_ylabel("rate")
    axes[2].set_ylabel("acc")
    axes[3].set_ylabel("time [s]")
    axes[3].set_xlabel("simulation time since reset [s]")
    for axis in axes:
      axis.grid(True, alpha=0.25)
      axis.legend(loc="upper right")
    axes[0].set_ylim(-0.05, 1.05)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.show(block=False)

    self._append_sample()
    self._redraw()

  def _phase_acceleration(self, phase_rate: float) -> float:
    acceleration = getattr(self.motion, "phase_acceleration", None)
    if acceleration is not None:
      return float(acceleration[self.env_id].item())
    return (phase_rate - self._previous_rate) / self.env.step_dt

  def _append_sample(self) -> None:
    env_id = self.env_id
    episode_step = int(self.env.episode_length_buf[env_id].item())
    phase_rate = float(self.motion.phase_rate[env_id].item())
    sample = {
      "time": episode_step * self.env.step_dt,
      "phase": float(self.motion.phase[env_id].item()),
      "phase_rate": phase_rate,
      "phase_acceleration": self._phase_acceleration(phase_rate),
      "time_remaining": float(self.motion.time_remaining[env_id].item()),
      "contact_time": float(self.motion.contact_time[env_id].item()),
    }
    for name, value in sample.items():
      self._history[name].append(value)
    self._previous_rate = phase_rate

  def _redraw(self) -> None:
    if not self._plt.fignum_exists(self._figure.number):
      return

    time = list(self._history["time"])
    for name, line in self._lines.items():
      line.set_data(time, list(self._history[name]))

    contact_time = self._history["contact_time"][-1]
    for line in self._deadline_lines:
      line.set_xdata([contact_time, contact_time])

    time_min = min(time[0], contact_time)
    time_max = max(time[-1], contact_time, time_min + self.env.step_dt)
    padding = max(0.05, 0.03 * (time_max - time_min))
    self._axes[-1].set_xlim(max(0.0, time_min - padding), time_max + padding)

    max_rate = max(abs(value) for value in self._history["phase_rate"])
    self._axes[1].set_ylim(-0.02, max(0.25, 1.15 * max_rate))
    max_acceleration = max(
      abs(value) for value in self._history["phase_acceleration"]
    )
    acceleration_limit = max(0.25, 1.15 * max_acceleration)
    self._axes[2].set_ylim(-acceleration_limit, acceleration_limit)
    max_time = max(
      max(self._history["time_remaining"]),
      max(self._history["contact_time"]),
    )
    self._axes[3].set_ylim(-0.02, max(0.1, 1.08 * max_time))

    self._figure.suptitle(
      f"Phase state | env {self.env_id} | episode {self._episode_index} | "
      f"dt={self.env.step_dt:.3f}s"
    )
    self._figure.canvas.draw_idle()
    self._figure.canvas.flush_events()

  def after_step(self) -> None:
    if not self._plt.fignum_exists(self._figure.number):
      return

    env_id = self.env_id
    episode_step = int(self.env.episode_length_buf[env_id].item())
    phase = float(self.motion.phase[env_id].item())
    restarted = (
      episode_step < self._previous_episode_step
      or phase < self._previous_phase - 1.0e-6
    )
    if restarted:
      for values in self._history.values():
        values.clear()
      self._episode_index += 1
      self._steps_since_draw = self._steps_per_draw

    self._append_sample()
    self._steps_since_draw += 1
    if self._steps_since_draw >= self._steps_per_draw:
      self._redraw()
      self._steps_since_draw = 0

    self._previous_episode_step = episode_step
    self._previous_phase = phase

  def close(self) -> None:
    if self._plt.fignum_exists(self._figure.number):
      self._plt.close(self._figure)


def build_play_session(
  task_id: str,
  cfg: PlayConfig,
  *,
  install_physical_ball_controller: bool = True,
  env_cfg_hook: Callable[[ManagerBasedRlEnvCfg], None] | None = None,
):
  """Build the play environment and inference policy without starting a viewer."""
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  if cfg.seed is not None:
    env_cfg.seed = cfg.seed

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE
  if cfg.onnx_policy_file is not None and DUMMY_MODE:
    raise ValueError("--onnx-policy-file requires --agent trained.")
  if cfg.onnx_policy_file is not None and cfg.checkpoint_file is not None:
    raise ValueError(
      "--onnx-policy-file and --checkpoint-file are mutually exclusive."
    )

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.

  is_multi_target_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MultiTargetMotionCommandCfg
  )

  if is_multi_target_task and cfg._demo_mode:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MultiTargetMotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_multi_target_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MultiTargetMotionCommandCfg)
    if cfg.ghost_reference_world:
      motion_cmd.viz.mode = "ghost"
      motion_cmd.viz.ghost_root_mode = "initial_aligned"
      motion_cmd.sampling_mode = "start"
      print("[INFO]: Ghost frame 0 aligned to robot XYZ/yaw; trajectory preserved")
    resolved_registry_name: str | None = None
    if cfg.motion_config is not None:
      from athlete.motion_sets.motion_set import (
        MotionSet,
        filter_motion_cmd_cfg,
        set_motion_cmd_cfgs,
      )

      motion_set = MotionSet.from_toml(cfg.motion_config)
      resolved_registry_name = motion_set.train_registry()
      if motion_set.robot_xml is not None:
        import mujoco

        _xml = motion_set.robot_xml
        robot_entity = env_cfg.scene.entities.get("robot")
        if robot_entity is not None:
          robot_entity.spec_fn = lambda: mujoco.MjSpec.from_file(_xml)
        print(f"[INFO]: Robot XML from motion config: {_xml}")

      local_motion_files = motion_set.local_motion_files
      local_motion_cfgs = motion_set.local_motion_cfgs()
      if local_motion_files is not None and local_motion_cfgs is not None:
        motion_cmd.motion_files = local_motion_files
        set_motion_cmd_cfgs(motion_cmd, local_motion_cfgs)
        print(f"[INFO]: Motion files from motion config: {local_motion_files}")
    else:
      motion_set = None

    if motion_cmd.motion_files:
      # Already resolved (e.g. by the tracker block above).
      pass
    elif cfg.motion_file is not None:
      # Comma-separated local paths.
      files = [f.strip() for f in cfg.motion_file.split(",")]
      for f in files:
        if not Path(f).exists():
          raise FileNotFoundError(f"Motion file not found: {f}")
      motion_cmd.motion_files = files
      print(f"[INFO]: Using local motion files: {files}")
    elif DUMMY_MODE:
      if not resolved_registry_name:
        raise ValueError(
          "Multi-target tracking tasks require either:\n"
          "  --motion-file f1.npz,f2.npz (comma-separated local files)\n"
          "  --motion-config path/to/config.toml (resolves eval registry from TOML)"
        )
      import wandb

      api = wandb.Api()
      registry_names = [r.strip() for r in resolved_registry_name.split(",")]
      motion_files: list[str] = []
      motion_names: list[str] = []
      for rn in registry_names:
        if ":" not in rn:
          rn = rn + ":latest"
        artifact = api.artifact(rn)
        motion_files.append(str(Path(artifact.download()) / "motion.npz"))
        motion_names.append(rn.split("/")[-1].split(":")[0])
        print(f"[INFO]: Downloaded motion: {rn} -> {motion_files[-1]}")
      motion_cmd.motion_files = motion_files
      if motion_set is not None:
        from athlete.motion_sets.motion_set import (
          filter_motion_cmd_cfg,
        )

        filter_motion_cmd_cfg(motion_cmd, motion_set.enabled_names)
    else:
      # Trained mode: resolve motion artifacts from the W&B training run.
      import wandb

      api = wandb.Api()
      if resolved_registry_name:
        registry_names = [r.strip() for r in resolved_registry_name.split(",")]
        motion_files = []
        motion_names = []
        for rn in registry_names:
          if ":" not in rn:
            rn = rn + ":latest"
          artifact = api.artifact(rn)
          motion_files.append(str(Path(artifact.download()) / "motion.npz"))
          motion_names.append(rn.split("/")[-1].split(":")[0])
          print(f"[INFO]: Downloaded motion: {rn} -> {motion_files[-1]}")
        motion_cmd.motion_files = motion_files
        if motion_set is not None:
          from athlete.motion_sets.motion_set import (
            filter_motion_cmd_cfg,
          )

          filter_motion_cmd_cfg(motion_cmd, motion_set.enabled_names)
      elif cfg.wandb_run_path is not None:
        wandb_run = api.run(str(cfg.wandb_run_path))
        arts = [a for a in wandb_run.used_artifacts() if a.type == "motions"]
        if not arts:
          raise RuntimeError("No motion artifacts found in the run.")
        motion_files = []
        for art in arts:
          motion_files.append(str(Path(art.download()) / "motion.npz"))
          print(f"[INFO]: Downloaded motion artifact: {art.name} -> {motion_files[-1]}")
        motion_cmd.motion_files = motion_files
      else:
        raise ValueError(
          "Multi-target tracking tasks require motion files. Provide one of:\n"
          "  --motion-file f1.npz,f2.npz (comma-separated local files)\n"
          "  --motion-config path/to/config.toml (resolves eval registry from TOML)\n"
          "  --wandb-run-path entity/project/run_id (resolve from training run)"
        )

  log_dir: Path | None = None
  resume_path: Path | None = None
  onnx_policy_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.onnx_policy_file is not None:
      onnx_policy_path = cfg.onnx_policy_file.expanduser().resolve()
      if not onnx_policy_path.is_file():
        raise FileNotFoundError(f"ONNX policy file not found: {onnx_policy_path}")
      print(f"[INFO]: Loading ONNX policy: {onnx_policy_path.name}")
      log_dir = onnx_policy_path.parent
    elif cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    if resume_path is not None:
      log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  physical_ball_enabled = "tennis_ball" in env_cfg.scene.entities
  if physical_ball_enabled:
    robot_entity = env_cfg.scene.entities.get("robot")
    if robot_entity is None:
      raise ValueError("Physical tennis task requires a robot scene entity.")
    robot_entity.spec_fn = add_racket_ball_collision(robot_entity.spec_fn)
  elif cfg.physical_ball:
    robot_entity = env_cfg.scene.entities.get("robot")
    if robot_entity is None:
      raise ValueError("Physical ball requires a robot scene entity.")
    robot_entity.spec_fn = _add_racket_ball_collision(robot_entity.spec_fn)
    env_cfg.scene.entities["tennis_ball"] = EntityCfg(spec_fn=_tennis_ball_spec)
    env_cfg.sim.nconmax = max(env_cfg.sim.nconmax, 128)
    env_cfg.sim.njmax = max(env_cfg.sim.njmax, 512)
    physical_ball_enabled = True
    print("[INFO]: Play-only physical tennis ball enabled")
  if env_cfg_hook is not None:
    env_cfg_hook(env_cfg)
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = FastPlayManagerBasedRlEnv(
    cfg=env_cfg, device=device, render_mode=render_mode
  )

  if physical_ball_enabled and install_physical_ball_controller:
    if hasattr(env.command_manager.get_term("motion"), "time_remaining"):
      install_tennis_ball_controller(
        env,
        cfg.ball_release_lead_s,
        verbose_samples=True,
      )
    else:
      install_tennis_ball_aerodynamics(env)
      ball_controller = _PlayTennisBallController(env, cfg.ball_release_lead_s)
      base_step = env.step

      def step_with_physical_ball(action: torch.Tensor, _base_step=base_step):
        ball_controller.before_step()
        result = _base_step(action)
        after_step = getattr(ball_controller, "after_step", None)
        if after_step is not None:
          after_step(result[2] | result[3])
        return result

      env.step = step_with_physical_ball  # type: ignore[method-assign]
  elif physical_ball_enabled:
    install_tennis_ball_aerodynamics(env)

  if cfg.ball_impact_debug:
    if not physical_ball_enabled:
      raise ValueError("--ball-impact-debug requires a physical tennis ball.")
    impact_debug = _BallImpactDebugController(env)
    base_step = env.step

    def step_with_ball_impact_debug(
      action: torch.Tensor, _base_step=base_step
    ):
      impact_debug.before_step()
      result = _base_step(action)
      impact_debug.after_step()
      return result

    env.step = step_with_ball_impact_debug  # type: ignore[method-assign]
    print("[INFO]: Ball impact diagnostics enabled")

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  elif onnx_policy_path is not None:
    if cfg.tppo_phase_source == "zero":
      policy = OnnxPlayPolicy(
        onnx_policy_path,
        env,
        observation_group="student",
        zero_pad_actions=1,
      )
    else:
      policy = OnnxPlayPolicy(onnx_policy_path, env)
    print(
      f"[INFO]: ONNX Runtime providers: {', '.join(policy.providers)}"
    )
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    checkpoint_load_cfg = {"actor": True}
    if cfg.tppo_phase_source == "zero":
      checkpoint_load_cfg["teacher"] = False
    runner.load(
      str(resume_path),
      load_cfg=checkpoint_load_cfg,
      strict=True,
      map_location=device,
    )
    environment_policy_fn = getattr(runner.alg, "get_environment_policy", None)
    if environment_policy_fn is not None:
      runner.alg.eval_mode()
      policy = environment_policy_fn(
        phase_source=cfg.tppo_phase_source
      ).to(device)
    else:
      if cfg.tppo_phase_source != "teacher":
        raise ValueError(
          "--tppo-phase-source is only supported by a TPPO environment policy."
        )
      policy = runner.get_inference_policy(device=device)

  if cfg.fast_play:
    policy_obs_groups = tuple(getattr(policy, "obs_groups", ("actor",)))
    env.unwrapped.enable_fast_play(policy_obs_groups)

  motion = (
    env.unwrapped.command_manager.get_term("motion")
    if "motion" in env.unwrapped.command_manager.active_terms
    else None
  )
  if cfg.print_student_goal:
    observations = env.get_observations()
    if "student" not in observations:
      raise ValueError(
        "--print-student-goal requires an active 'student' observation group."
      )
    student_obs = observations["student"][0]
    if student_obs.shape[-1] < 12:
      raise ValueError(
        "Student observation is too small to contain the 11-D task goal and "
        f"time remaining: {tuple(student_obs.shape)}."
      )
    print(
      "[STUDENT GOAL] task_goal_11="
      f"{student_obs[:11].detach().cpu().tolist()} "
      f"time_remaining={float(student_obs[11].item()):.6f}s",
      flush=True,
    )
  if cfg.phase_debug and isinstance(motion, PhaseAwareMultiTargetMotionCommand):
    phase_debug = _PhaseDebugController(env.unwrapped, interval_s=0.1, env_id=0)
    base_step = env.step

    def step_with_phase_debug(action: torch.Tensor, _base_step=base_step):
      result = _base_step(action)
      phase_debug.after_step()
      return result

    env.step = step_with_phase_debug  # type: ignore[method-assign]

  return env, policy


def run_play(task_id: str, cfg: PlayConfig):
  env, policy = build_play_session(task_id, cfg)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  phase_plot: _PhasePlotController | None = None
  motion = (
    env.unwrapped.command_manager.get_term("motion")
    if "motion" in env.unwrapped.command_manager.active_terms
    else None
  )
  if cfg.phase_plot and resolved_viewer == "native":
    if isinstance(motion, PhaseAwareMultiTargetMotionCommand):
      phase_plot = _PhasePlotController(
        env.unwrapped, env_id=0, history_s=cfg.phase_plot_history_s
      )
      base_step = env.step

      def step_with_phase_plot(action: torch.Tensor, _base_step=base_step):
        result = _base_step(action)
        assert phase_plot is not None
        phase_plot.after_step()
        return result

      env.step = step_with_phase_plot  # type: ignore[method-assign]
      print("[INFO]: Live phase plot enabled for env 0")
    else:
      print("[WARN]: --phase-plot ignored because this task is not phase-aware")

  try:
    if resolved_viewer == "native":
      NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
      ViserPlayViewer(env, policy).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    if phase_plot is not None:
      phase_plot.close()
    env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
