from __future__ import annotations

import torch

from athlete.scripts.fast_play_env import (
  _compute_commands_without_metrics,
)


class _FakeCommandTerm:
  def __init__(self) -> None:
    self.time_left = torch.tensor([0.01, 0.03])
    self.resampled: list[torch.Tensor] = []
    self.update_count = 0

  def _resample(self, env_ids: torch.Tensor) -> None:
    self.resampled.append(env_ids.clone())
    self.time_left[env_ids] = 1.0

  def _update_command(self) -> None:
    self.update_count += 1


class _FakeCommandManager:
  active_terms = ["motion"]

  def __init__(self, term: _FakeCommandTerm) -> None:
    self.term = term

  def get_term(self, name: str) -> _FakeCommandTerm:
    assert name == "motion"
    return self.term


def test_compute_commands_without_metrics_preserves_resampling_and_update() -> None:
  term = _FakeCommandTerm()
  manager = _FakeCommandManager(term)

  _compute_commands_without_metrics(manager, dt=0.02)  # type: ignore[arg-type]

  assert len(term.resampled) == 1
  torch.testing.assert_close(term.resampled[0], torch.tensor([0]))
  torch.testing.assert_close(term.time_left, torch.tensor([1.0, 0.01]))
  assert term.update_count == 1
