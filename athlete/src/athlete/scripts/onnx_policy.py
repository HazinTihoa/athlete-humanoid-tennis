"""ONNX Runtime inference adapter for ATHLETE play environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


class OnnxPlayPolicy:
  """Run an exported ATHLETE actor from an ONNX model."""

  _SUPPORTED_INPUTS = frozenset({"obs", "which_motion", "time_step"})

  def __init__(
    self,
    policy_path: str | Path,
    env: Any,
    *,
    providers: Sequence[str] | None = None,
    observation_group: str = "actor",
    zero_pad_actions: int = 0,
  ) -> None:
    try:
      import onnxruntime as ort
    except ImportError as exc:
      raise RuntimeError(
        "ONNX policy playback requires onnxruntime. Run `uv sync` from the "
        "repository root."
      ) from exc

    path = Path(policy_path).expanduser().resolve()
    if not path.is_file():
      raise FileNotFoundError(f"ONNX policy file not found: {path}")

    available_providers = ort.get_available_providers()
    if providers is None:
      providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available_providers
      ]
    if not providers:
      raise RuntimeError(
        "ONNX Runtime has no usable execution provider. Available providers: "
        f"{available_providers}"
      )

    self._session = ort.InferenceSession(str(path), providers=list(providers))
    self._inputs = {value.name: value for value in self._session.get_inputs()}
    unknown_inputs = set(self._inputs) - self._SUPPORTED_INPUTS
    if "obs" not in self._inputs or unknown_inputs:
      raise ValueError(
        "Unsupported ONNX policy inputs. Expected `obs` with optional "
        f"`which_motion`/`time_step`, got {tuple(self._inputs)}."
      )

    output_names = [value.name for value in self._session.get_outputs()]
    self._action_output = "actions" if "actions" in output_names else output_names[0]
    self._num_envs = int(env.num_envs)
    self._num_actions = int(env.num_actions)
    self._device = torch.device(env.device)
    self._observation_group = observation_group
    self._zero_pad_actions = int(zero_pad_actions)
    if not 0 <= self._zero_pad_actions < self._num_actions:
      raise ValueError(
        "zero_pad_actions must be non-negative and smaller than the environment "
        f"action dimension, got {self._zero_pad_actions} and {self._num_actions}."
      )
    self._policy_action_dim = self._num_actions - self._zero_pad_actions
    self._motion = None
    if {"which_motion", "time_step"} & set(self._inputs):
      manager = env.unwrapped.command_manager
      if "motion" not in manager.active_terms:
        raise ValueError(
          "ONNX policy requires motion indices, but the environment has no active "
          "`motion` command."
        )
      self._motion = manager.get_term("motion")

    self._validate_static_shape("obs", self._num_envs, axis=0)
    self._validate_action_shape()
    self.policy_path = path
    self.providers = tuple(self._session.get_providers())

  @property
  def obs_groups(self) -> tuple[str, ...]:
    return (self._observation_group,)

  def _validate_static_shape(
    self, input_name: str, expected: int | None, *, axis: int
  ) -> None:
    shape = self._inputs[input_name].shape
    if len(shape) <= axis:
      raise ValueError(f"ONNX input {input_name!r} has invalid shape {shape}.")
    dimension = shape[axis]
    if expected is not None and isinstance(dimension, int) and dimension != expected:
      raise ValueError(
        f"ONNX input {input_name!r} expects dimension {dimension} on axis {axis}, "
        f"but the environment provides {expected}."
      )

  def _validate_action_shape(self) -> None:
    output = next(
      value
      for value in self._session.get_outputs()
      if value.name == self._action_output
    )
    if output.shape and isinstance(output.shape[-1], int):
      if output.shape[-1] != self._policy_action_dim:
        raise ValueError(
          f"ONNX policy outputs {output.shape[-1]} actions, but the environment "
          f"adapter expects {self._policy_action_dim} before zero padding."
        )

  def _policy_observation(self, observations: Any) -> torch.Tensor:
    if isinstance(observations, torch.Tensor):
      return observations
    try:
      policy_obs = observations[self._observation_group]
    except (KeyError, TypeError) as exc:
      raise TypeError(
        "ONNX policy expected a tensor or an observation mapping containing "
        f"the {self._observation_group!r} group."
      ) from exc
    if not isinstance(policy_obs, torch.Tensor):
      raise TypeError(
        "Policy observation must be a torch.Tensor, got "
        f"{type(policy_obs)}."
      )
    return policy_obs

  @staticmethod
  def _as_float32(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(
      tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    )

  def __call__(self, observations: Any) -> torch.Tensor:
    policy_obs = self._policy_observation(observations)
    if policy_obs.ndim != 2 or policy_obs.shape[0] != self._num_envs:
      raise ValueError(
        f"Policy observation must have shape ({self._num_envs}, N), got "
        f"{tuple(policy_obs.shape)}."
      )

    expected_obs_dim = self._inputs["obs"].shape[-1]
    if (
      isinstance(expected_obs_dim, int)
      and policy_obs.shape[-1] != expected_obs_dim
    ):
      raise ValueError(
        f"ONNX policy expects {expected_obs_dim} observations from group "
        f"{self._observation_group!r}, but the environment produced "
        f"{policy_obs.shape[-1]}."
      )

    feeds = {"obs": self._as_float32(policy_obs)}
    if self._motion is not None:
      if "which_motion" in self._inputs:
        feeds["which_motion"] = self._as_float32(
          self._motion.which_motion.reshape(self._num_envs, 1)
        )
      if "time_step" in self._inputs:
        feeds["time_step"] = self._as_float32(
          self._motion.time_steps.reshape(self._num_envs, 1)
        )

    actions = np.asarray(
      self._session.run([self._action_output], feeds)[0], dtype=np.float32
    )
    if actions.ndim == 1:
      actions = actions.reshape(1, -1)
    expected_shape = (self._num_envs, self._policy_action_dim)
    if actions.shape != expected_shape:
      raise RuntimeError(
        f"ONNX policy returned actions with shape {actions.shape}; expected "
        f"{expected_shape}."
      )
    if not np.isfinite(actions).all():
      raise RuntimeError("ONNX policy returned NaN or Inf actions.")
    if self._zero_pad_actions:
      actions = np.concatenate(
        [
          actions,
          np.zeros(
            (self._num_envs, self._zero_pad_actions), dtype=np.float32
          ),
        ],
        axis=-1,
      )
    return torch.from_numpy(actions).to(self._device)
