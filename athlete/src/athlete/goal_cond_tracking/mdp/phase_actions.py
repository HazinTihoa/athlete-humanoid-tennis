from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from .phase_commands import PhaseAwareMultiTargetMotionCommand
from .phase_commands import PhaseAccelerationMultiTargetMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class PhaseResidualAction(ActionTerm):
  """Route one policy action to the phase-aware command generator."""

  cfg: PhaseResidualActionCfg

  def __init__(self, cfg: PhaseResidualActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    command = env.command_manager.get_term(cfg.command_name)
    if not isinstance(command, PhaseAwareMultiTargetMotionCommand):
      raise TypeError(
        f"Command {cfg.command_name!r} must be phase-aware, got {type(command).__name__}."
      )
    self.command = cast(PhaseAwareMultiTargetMotionCommand, command)
    self._raw_action = torch.zeros(self.num_envs, 1, device=self.device)

  @property
  def action_dim(self) -> int:
    return 1

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_action

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_action[:] = actions
    self.command.set_phase_residual(actions[:, 0])

  def apply_actions(self) -> None:
    pass

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_action[env_ids] = 0.0
    self.command.phase_residual[env_ids] = 0.0


@dataclass(kw_only=True)
class PhaseResidualActionCfg(ActionTermCfg):
  command_name: str = "motion"

  def build(self, env: ManagerBasedRlEnv) -> PhaseResidualAction:
    return PhaseResidualAction(self, env)


class PhaseAccelerationAction(ActionTerm):
  """Route one policy action to the phase-acceleration controller."""

  cfg: PhaseAccelerationActionCfg

  def __init__(self, cfg: PhaseAccelerationActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    command = env.command_manager.get_term(cfg.command_name)
    if not isinstance(command, PhaseAccelerationMultiTargetMotionCommand):
      raise TypeError(
        f"Command {cfg.command_name!r} must use phase acceleration, "
        f"got {type(command).__name__}."
      )
    self.command = cast(PhaseAccelerationMultiTargetMotionCommand, command)
    self._raw_action = torch.zeros(self.num_envs, 1, device=self.device)

  @property
  def action_dim(self) -> int:
    return 1

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_action

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_action[:] = actions
    self.command.set_phase_acceleration_action(actions[:, 0])

  def apply_actions(self) -> None:
    pass

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_action[env_ids] = 0.0
    self.command.phase_acceleration_action[env_ids] = 0.0


@dataclass(kw_only=True)
class PhaseAccelerationActionCfg(ActionTermCfg):
  command_name: str = "motion"

  def build(self, env: ManagerBasedRlEnv) -> PhaseAccelerationAction:
    return PhaseAccelerationAction(self, env)
