"""Inference-only environment step path used by Play scripts."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandManager, NullCommandManager


def _compute_commands_without_metrics(
  manager: CommandManager | NullCommandManager, dt: float
) -> None:
  """Advance and resample commands without computing logging-only metrics."""
  for name in manager.active_terms:
    term = manager.get_term(name)
    term.time_left -= dt
    resample_env_ids = (term.time_left <= 0.0).nonzero().flatten()
    if len(resample_env_ids) > 0:
      term._resample(resample_env_ids)
    term._update_command()


class FastPlayManagerBasedRlEnv(ManagerBasedRlEnv):
  """Manager environment with an inference-only fast step that can be enabled late.

  The regular path remains active during initial reset and runner construction so
  RSL-RL can inspect actor and critic observation shapes. Play enables the fast path
  only after the checkpoint has been loaded.
  """

  fast_play_enabled: bool = False
  fast_play_observation_groups: tuple[str, ...] = ("actor",)

  def enable_fast_play(
    self, observation_groups: tuple[str, ...] = ("actor",)
  ) -> None:
    observation_groups = tuple(dict.fromkeys(observation_groups))
    if not observation_groups:
      raise ValueError("Fast Play requires at least one observation group.")
    missing_groups = tuple(
      group
      for group in observation_groups
      if group not in self.observation_manager.active_terms
    )
    if missing_groups:
      raise ValueError(
        f"Fast Play observation groups are not active: {missing_groups}."
      )
    self.fast_play_observation_groups = observation_groups
    self.fast_play_enabled = True
    print(
      "[INFO]: Fast Play enabled: observation groups "
      f"{observation_groups}; reward and logging metrics skipped"
    )

  def step(self, action: torch.Tensor):
    if not self.fast_play_enabled:
      return super().step(action)

    if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
      pending_ids = self._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
      raise RuntimeError(
        f"Environments {pending_ids.cpu().tolist()} must be reset via "
        "reset(env_ids=...) before calling step() again when auto_reset=False."
      )

    self.extras["log"] = {}
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)

    self.episode_length_buf += 1
    self.common_step_counter += 1

    # Terminations and resets affect Play behavior, so they remain enabled.
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.recorder_manager.record_pre_reset(reset_env_ids)
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()

    self.sim.forward()
    _compute_commands_without_metrics(self.command_manager, self.step_dt)

    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = {
      group: self.observation_manager.compute_group(group, update_history=True)
      for group in self.fast_play_observation_groups
    }
    # Keep get_observations() on the same inference result until the next step.
    self.observation_manager._obs_buffer = self.obs_buf

    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.recorder_manager.record_post_reset(reset_env_ids)
    elif len(reset_env_ids) > 0:
      self._manual_reset_pending[reset_env_ids] = True
    self.recorder_manager.record_post_step()

    reward = self.reward_manager._reward_buf
    reward.zero_()
    return (
      self.obs_buf,
      reward,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )
