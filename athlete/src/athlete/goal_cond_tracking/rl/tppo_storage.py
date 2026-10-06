"""Rollout storage for TPPO teacher labels and DAgger control masks.

Adapted from ``ActionLabelRollout`` in Instinct-RL:
https://github.com/project-instinct/instinct_rl
Instinct-RL is licensed under CC BY-NC 4.0. This adaptation targets the
RSL-RL 5 storage API used by this project and is for non-commercial use.
License: https://creativecommons.org/licenses/by-nc/4.0/
"""

from __future__ import annotations

from collections.abc import Generator

import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


class TppoRolloutStorage(RolloutStorage):
  """PPO rollout storage augmented with teacher actions and control masks."""

  class Transition(RolloutStorage.Transition):
    def __init__(self) -> None:
      super().__init__()
      self.student_control_mask: torch.Tensor | None = None

  def __init__(
    self,
    num_envs: int,
    num_transitions_per_env: int,
    obs: TensorDict,
    actions_shape: tuple[int, ...] | list[int],
    device: str = "cpu",
  ) -> None:
    super().__init__(
      "rl",
      num_envs,
      num_transitions_per_env,
      obs,
      actions_shape,
      device,
    )
    self.privileged_actions = torch.zeros(
      num_transitions_per_env,
      num_envs,
      *actions_shape,
      device=self.device,
    )
    self.student_control_masks = torch.zeros(
      num_transitions_per_env,
      num_envs,
      1,
      dtype=torch.bool,
      device=self.device,
    )

  def add_transition(self, transition: Transition) -> None:
    """Store TPPO fields at the same cursor as the base PPO transition."""
    if transition.privileged_actions is None:
      raise ValueError("TPPO transition is missing teacher action labels.")
    if transition.student_control_mask is None:
      raise ValueError("TPPO transition is missing the student control mask.")

    step = self.step
    self.privileged_actions[step].copy_(transition.privileged_actions)
    self.student_control_masks[step].copy_(
      transition.student_control_mask.reshape(self.num_envs, 1)
    )
    super().add_transition(transition)

  def mini_batch_generator(
    self, num_mini_batches: int, num_epochs: int = 8
  ) -> Generator[RolloutStorage.Batch, None, None]:
    """Yield shuffled PPO batches with aligned TPPO fields."""
    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(
      num_mini_batches * mini_batch_size,
      requires_grad=False,
      device=self.device,
    )

    observations = self.observations.flatten(0, 1)
    actions = self.actions.flatten(0, 1)
    values = self.values.flatten(0, 1)
    returns = self.returns.flatten(0, 1)
    old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
    advantages = self.advantages.flatten(0, 1)
    old_distribution_params = tuple(
      parameter.flatten(0, 1)
      for parameter in self.distribution_params  # type: ignore[union-attr]
    )
    privileged_actions = self.privileged_actions.flatten(0, 1)
    student_control_masks = self.student_control_masks.flatten(0, 1)
    dones = self.dones.flatten(0, 1)

    for _ in range(num_epochs):
      for mini_batch_index in range(num_mini_batches):
        start = mini_batch_index * mini_batch_size
        stop = (mini_batch_index + 1) * mini_batch_size
        batch_idx = indices[start:stop]
        batch = RolloutStorage.Batch(
          observations=observations[batch_idx],  # type: ignore[arg-type]
          actions=actions[batch_idx],
          values=values[batch_idx],
          advantages=advantages[batch_idx],
          returns=returns[batch_idx],
          old_actions_log_prob=old_actions_log_prob[batch_idx],
          old_distribution_params=tuple(
            parameter[batch_idx] for parameter in old_distribution_params
          ),
          privileged_actions=privileged_actions[batch_idx],
          dones=dones[batch_idx],
        )
        batch.student_control_mask = student_control_masks[batch_idx]
        yield batch

  def recurrent_mini_batch_generator(
    self, num_mini_batches: int, num_epochs: int = 8
  ) -> Generator[RolloutStorage.Batch, None, None]:
    """Attach aligned TPPO fields to the base recurrent PPO batches."""
    base_generator = super().recurrent_mini_batch_generator(
      num_mini_batches, num_epochs
    )
    mini_batch_size = self.num_envs // num_mini_batches

    for _ in range(num_epochs):
      for mini_batch_index in range(num_mini_batches):
        start = mini_batch_index * mini_batch_size
        stop = (mini_batch_index + 1) * mini_batch_size
        batch = next(base_generator)
        batch.privileged_actions = self.privileged_actions[:, start:stop]
        batch.dones = self.dones[:, start:stop]
        batch.student_control_mask = self.student_control_masks[:, start:stop]
        yield batch
