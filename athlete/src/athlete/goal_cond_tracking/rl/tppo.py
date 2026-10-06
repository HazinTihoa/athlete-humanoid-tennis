"""Teacher-guided PPO and DAgger distillation for RSL-RL 5.

The TPPO behavior is adapted from Instinct-RL's TPPO implementation:
https://github.com/project-instinct/instinct_rl
Instinct-RL is licensed under CC BY-NC 4.0. This adaptation changes the model,
storage, checkpoint, and runner interfaces to match this project's RSL-RL 5
backend and is for non-commercial use.
License: https://creativecommons.org/licenses/by-nc/4.0/
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups
from tensordict import TensorDict
from torch import nn

from .tppo_storage import TppoRolloutStorage


def _decay_schedule(name: str, scale: int, step: int) -> float:
  """Evaluate one of Instinct-RL's teacher/coefficient decay schedules."""
  if scale <= 0:
    raise ValueError("update_times_scale must be positive.")
  if name == "linear":
    return max(0.0, 1.0 - step / scale)
  if name == "exp":
    return max(0.0, (1.0 - 1.0 / scale) ** step)
  if name == "tanh":
    return max(0.0, 0.5 * (1.0 - math.tanh((step - scale) / scale)))
  raise ValueError(f"Unknown TPPO schedule '{name}'. Expected linear, exp, or tanh.")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """Compute a stable mean that is zero when no samples are selected."""
  while values.ndim > mask.ndim and values.shape[-1] == 1:
    values = values.squeeze(-1)
  while mask.ndim > values.ndim and mask.shape[-1] == 1:
    mask = mask.squeeze(-1)
  mask = torch.broadcast_to(mask.to(dtype=values.dtype), values.shape)
  return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _inference_obs_groups(model: MLPModel) -> tuple[str, ...]:
  """Return the current and auxiliary observation groups needed at inference."""
  return tuple(getattr(model, "inference_obs_groups", model.obs_groups))


class StudentWithTeacherPhasePolicy(nn.Module):
  """Run Student joints while sourcing hidden phase controls from Teacher."""

  def __init__(
    self,
    student: MLPModel,
    teacher: MLPModel,
    student_action_dim: int,
    env_action_dim: int,
  ) -> None:
    super().__init__()
    self.student = student
    self.teacher = teacher
    self.student_action_dim = student_action_dim
    self.env_action_dim = env_action_dim

  @property
  def obs_groups(self) -> tuple[str, ...]:
    """Observation groups required by both inference policies."""
    return tuple(
      dict.fromkeys(
        (
          *_inference_obs_groups(self.student),
          *_inference_obs_groups(self.teacher),
        )
      )
    )

  def forward(self, obs: TensorDict) -> torch.Tensor:
    student_actions = self.student(obs)
    teacher_actions = self.teacher(obs)
    if student_actions.shape[-1] != self.student_action_dim:
      raise ValueError(
        f"Student produced {student_actions.shape[-1]} actions, expected "
        f"{self.student_action_dim}."
      )
    if teacher_actions.shape[-1] != self.env_action_dim:
      raise ValueError(
        f"Teacher produced {teacher_actions.shape[-1]} actions, expected "
        f"{self.env_action_dim}."
      )
    return torch.cat(
      [student_actions, teacher_actions[..., self.student_action_dim :]], dim=-1
    )

  def reset(self, dones: torch.Tensor | None = None) -> None:
    self.student.reset(dones)
    self.teacher.reset(dones)


class StudentWithZeroPhasePolicy(nn.Module):
  """Run Student joints while sending zero for non-Student environment actions."""

  def __init__(
    self,
    student: MLPModel,
    student_action_dim: int,
    env_action_dim: int,
  ) -> None:
    super().__init__()
    self.student = student
    self.student_action_dim = student_action_dim
    self.env_action_dim = env_action_dim

  @property
  def obs_groups(self) -> tuple[str, ...]:
    return _inference_obs_groups(self.student)

  def forward(self, obs: TensorDict) -> torch.Tensor:
    student_actions = self.student(obs)
    if student_actions.shape[-1] != self.student_action_dim:
      raise ValueError(
        f"Student produced {student_actions.shape[-1]} actions, expected "
        f"{self.student_action_dim}."
      )
    phase_actions = torch.zeros(
      *student_actions.shape[:-1],
      self.env_action_dim - self.student_action_dim,
      dtype=student_actions.dtype,
      device=student_actions.device,
    )
    return torch.cat([student_actions, phase_actions], dim=-1)

  def reset(self, dones: torch.Tensor | None = None) -> None:
    self.student.reset(dones)


class TeacherStudentTrainingPolicy(nn.Module):
  """Mirror TPPO's current DAgger action mixing in a training debug rollout."""

  def __init__(
    self,
    student: MLPModel,
    teacher: MLPModel,
    student_action_dim: int,
    env_action_dim: int,
    teacher_action_probability: float,
  ) -> None:
    super().__init__()
    self.student = student
    self.teacher = teacher
    self.student_action_dim = student_action_dim
    self.env_action_dim = env_action_dim
    self.register_buffer(
      "teacher_action_probability",
      torch.tensor(float(teacher_action_probability)),
    )
    self.register_buffer(
      "_use_teacher_mask",
      torch.empty(0, dtype=torch.bool),
      persistent=False,
    )

  @property
  def obs_groups(self) -> tuple[str, ...]:
    return tuple(
      dict.fromkeys(
        (
          *_inference_obs_groups(self.student),
          *_inference_obs_groups(self.teacher),
        )
      )
    )

  def _resample_mask(self, num_envs: int) -> None:
    probability = float(self.teacher_action_probability.item())
    self._use_teacher_mask = (
      torch.rand(num_envs, device=self.teacher_action_probability.device) < probability
    )

  def forward(self, obs: TensorDict) -> torch.Tensor:
    student_actions = self.student(obs)
    teacher_actions = self.teacher(obs)
    if student_actions.shape[-1] != self.student_action_dim:
      raise ValueError(
        f"Student produced {student_actions.shape[-1]} actions, expected "
        f"{self.student_action_dim}."
      )
    if teacher_actions.shape[-1] != self.env_action_dim:
      raise ValueError(
        f"Teacher produced {teacher_actions.shape[-1]} actions, expected "
        f"{self.env_action_dim}."
      )

    if self._use_teacher_mask.shape[0] != student_actions.shape[0]:
      self._resample_mask(student_actions.shape[0])
    action_mask = self._use_teacher_mask.reshape(
      self._use_teacher_mask.shape[0],
      *([1] * (student_actions.ndim - 1)),
    )
    joint_actions = torch.where(
      action_mask,
      teacher_actions[..., : self.student_action_dim],
      student_actions,
    )
    return torch.cat(
      [joint_actions, teacher_actions[..., self.student_action_dim :]], dim=-1
    )

  def reset(self, dones: torch.Tensor | None = None) -> None:
    self.student.reset(dones)
    self.teacher.reset(dones)
    if self._use_teacher_mask.numel() == 0:
      return
    if dones is None:
      self._resample_mask(self._use_teacher_mask.shape[0])
      return
    done_mask = dones.reshape(-1).bool()
    count = int(done_mask.sum().item())
    if count > 0:
      probability = float(self.teacher_action_probability.item())
      self._use_teacher_mask[done_mask] = (
        torch.rand(count, device=self.teacher_action_probability.device) < probability
      )


class TPPO(PPO):
  """PPO with a frozen teacher, fixed-quota DAgger mixing, and distillation."""

  teacher: MLPModel

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    teacher: MLPModel,
    storage: TppoRolloutStorage,
    teacher_checkpoint: str | None = None,
    teacher_act_prob: float | str = "exp",
    update_times_scale: int = 100,
    using_ppo: bool = True,
    distillation_loss_coef: float | str = 1.0,
    distillation_update_times_scale: int | None = None,
    distillation_min_coef: float = 0.0,
    distill_target: str = "real",
    action_labels_from_sample: bool = False,
    strict_teacher_checkpoint: bool = True,
    student_action_dim: int = 29,
    env_action_dim: int | None = None,
    **kwargs: Any,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)
    self.teacher = teacher.to(self.device)
    self._raw_teacher = self.teacher
    self.teacher.requires_grad_(False)
    self.teacher.eval()

    self.teacher_act_prob = teacher_act_prob
    self.update_times_scale = update_times_scale
    self.using_ppo = using_ppo
    self.distillation_loss_coef = distillation_loss_coef
    self.distillation_update_times_scale = (
      update_times_scale
      if distillation_update_times_scale is None
      else distillation_update_times_scale
    )
    self.distillation_min_coef = float(distillation_min_coef)
    self.distill_target = distill_target
    self.action_labels_from_sample = action_labels_from_sample
    self.strict_teacher_checkpoint = strict_teacher_checkpoint
    self.student_action_dim = int(student_action_dim)
    self.env_action_dim = int(
      env_action_dim if env_action_dim is not None else student_action_dim
    )
    if not 0 < self.student_action_dim <= self.env_action_dim:
      raise ValueError(
        "student_action_dim must be positive and no larger than env_action_dim, "
        f"got {self.student_action_dim} and {self.env_action_dim}."
      )
    self.update_count = 0
    self.teacher_loaded = False
    self.teacher_checkpoint: str | None = None
    self.use_teacher_act_mask: torch.Tensor | None = None
    self.last_role_switch_count = 0
    self.transition = TppoRolloutStorage.Transition()

    self._validate_schedule(self.teacher_act_prob, "teacher_act_prob")
    self._validate_schedule(self.distillation_loss_coef, "distillation_loss_coef")
    if not 0.0 <= self.distillation_min_coef <= 1.0:
      raise ValueError("distillation_min_coef must be in [0, 1].")
    if teacher_checkpoint is not None:
      self.load_teacher(teacher_checkpoint)

  @staticmethod
  def _validate_schedule(value: float | str, field_name: str) -> None:
    if isinstance(value, str) and value not in {"linear", "exp", "tanh"}:
      raise ValueError(
        f"{field_name} must be a float or one of linear, exp, tanh; got {value!r}."
      )

  def _scheduled_value(
    self, value: float | str, update_times_scale: int | None = None
  ) -> float:
    if isinstance(value, str):
      scale = (
        self.update_times_scale if update_times_scale is None else update_times_scale
      )
      return _decay_schedule(value, scale, self.update_count)
    return float(value)

  @property
  def teacher_action_probability(self) -> float:
    return min(1.0, max(0.0, self._scheduled_value(self.teacher_act_prob)))

  @property
  def current_distillation_loss_coef(self) -> float:
    scheduled = max(
      0.0,
      self._scheduled_value(
        self.distillation_loss_coef, self.distillation_update_times_scale
      ),
    )
    if not isinstance(self.distillation_loss_coef, str):
      return scheduled
    return self.distillation_min_coef + (1.0 - self.distillation_min_coef) * scheduled

  @staticmethod
  def _resolve_teacher_checkpoint(path: str) -> Path:
    checkpoint = Path(path).expanduser()
    if checkpoint.is_file():
      return checkpoint
    if not checkpoint.is_dir():
      raise FileNotFoundError(f"Teacher checkpoint does not exist: {checkpoint}")

    candidates = list(checkpoint.glob("model_*.pt"))
    if not candidates:
      raise FileNotFoundError(
        f"No model_*.pt teacher checkpoint found in: {checkpoint}"
      )

    def iteration(model_path: Path) -> int:
      match = re.fullmatch(r"model_(\d+)\.pt", model_path.name)
      return int(match.group(1)) if match else -1

    return max(
      candidates, key=lambda model_path: (iteration(model_path), model_path.name)
    )

  @staticmethod
  def _extract_teacher_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if "teacher_state_dict" in checkpoint:
      return checkpoint["teacher_state_dict"]
    if "actor_state_dict" in checkpoint:
      state_dict = dict(checkpoint["actor_state_dict"])
    elif "model_state_dict" in checkpoint:
      state_dict = {}
      for key, value in checkpoint["model_state_dict"].items():
        if key.startswith("actor."):
          state_dict[key.replace("actor.", "mlp.", 1)] = value
        elif key.startswith("actor_obs_normalizer."):
          state_dict[key.replace("actor_obs_normalizer.", "obs_normalizer.", 1)] = value
        elif key in {"std", "log_std"}:
          state_dict[key] = value
    elif checkpoint and all(
      isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
      state_dict = dict(checkpoint)
    else:
      raise KeyError(
        "Teacher checkpoint has no teacher_state_dict, actor_state_dict, "
        "or legacy model_state_dict."
      )

    if "std" in state_dict:
      state_dict["distribution.std_param"] = state_dict.pop("std")
    if "log_std" in state_dict:
      state_dict["distribution.log_std_param"] = state_dict.pop("log_std")
    return state_dict

  def load_teacher_state_dict(
    self, state_dict: dict[str, Any], strict: bool | None = None
  ) -> None:
    """Load and freeze Teacher parameters from an already extracted state dict."""
    if strict is None:
      strict = self.strict_teacher_checkpoint
    self._raw_teacher.load_state_dict(state_dict, strict=strict)
    self._raw_teacher.requires_grad_(False)
    self._raw_teacher.eval()
    self.teacher_loaded = True

  def load_teacher(self, path: str) -> Path:
    """Load a Teacher actor from a PPO checkpoint file or run directory."""
    checkpoint_path = self._resolve_teacher_checkpoint(path)
    checkpoint = torch.load(
      checkpoint_path,
      map_location="cpu",
      weights_only=False,
    )
    state_dict = self._extract_teacher_state_dict(checkpoint)
    self.load_teacher_state_dict(state_dict)
    self.teacher_checkpoint = str(checkpoint_path)
    print(f"[TPPO] Loaded Teacher actor from {checkpoint_path}")
    return checkpoint_path

  def get_teacher_actions(self, obs: TensorDict) -> torch.Tensor:
    """Compute frozen Teacher labels from the configured Teacher observation set."""
    if not self.teacher_loaded:
      raise RuntimeError(
        "TPPO cannot collect rollouts before a Teacher is loaded. Set "
        "agent.algorithm.teacher_checkpoint or resume from a TPPO checkpoint."
      )
    with torch.inference_mode():
      return self.teacher(
        obs,
        stochastic_output=self.action_labels_from_sample,
      ).detach()

  def _desired_teacher_count(self, num_envs: int) -> int:
    """Convert the scheduled probability into an exact per-rank environment quota."""
    if num_envs <= 0:
      raise ValueError(f"num_envs must be positive, got {num_envs}.")
    return min(
      num_envs,
      max(0, math.floor(self.teacher_action_probability * num_envs + 0.5)),
    )

  def prepare_rollout(self, num_envs: int) -> torch.Tensor:
    """Apply the current Teacher quota and return envs whose role changed.

    The returned mask is empty on first initialization because all environments
    have already been reset by the vector wrapper. On later calls, the runner
    resets only environments that move between the Teacher and Student pools.
    """
    desired_count = self._desired_teacher_count(num_envs)
    if self.use_teacher_act_mask is None or self.use_teacher_act_mask.shape != (
      num_envs,
    ):
      teacher_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
      if desired_count > 0:
        selected = torch.randperm(num_envs, device=self.device)[:desired_count]
        teacher_mask[selected] = True
      self.use_teacher_act_mask = teacher_mask
      self.last_role_switch_count = 0
      return torch.zeros_like(teacher_mask)

    teacher_mask = self.use_teacher_act_mask
    current_count = int(teacher_mask.sum().item())
    role_switch_mask = torch.zeros_like(teacher_mask)
    if current_count > desired_count:
      candidates = teacher_mask.nonzero(as_tuple=False).squeeze(-1)
      count = current_count - desired_count
      selected = candidates[torch.randperm(current_count, device=self.device)[:count]]
      teacher_mask[selected] = False
      role_switch_mask[selected] = True
    elif current_count < desired_count:
      candidates = (~teacher_mask).nonzero(as_tuple=False).squeeze(-1)
      count = desired_count - current_count
      selected = candidates[
        torch.randperm(num_envs - current_count, device=self.device)[:count]
      ]
      teacher_mask[selected] = True
      role_switch_mask[selected] = True

    self.last_role_switch_count = int(role_switch_mask.sum().item())
    return role_switch_mask

  def reset_controller_states(self, reset_mask: torch.Tensor) -> None:
    """Reset recurrent policy state after role-switch environments are reset."""
    reset_mask = reset_mask.to(device=self.device, dtype=torch.bool)
    self.actor.reset(reset_mask)
    self.critic.reset(reset_mask)
    self.teacher.reset(reset_mask)

  def _teacher_student_labels(self, teacher_actions: torch.Tensor) -> torch.Tensor:
    if teacher_actions.shape[-1] != self.env_action_dim:
      raise ValueError(
        f"Teacher produced {teacher_actions.shape[-1]} actions, expected "
        f"{self.env_action_dim}."
      )
    return teacher_actions[..., : self.student_action_dim]

  def _compose_environment_actions(
    self,
    joint_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
  ) -> torch.Tensor:
    if joint_actions.shape[-1] != self.student_action_dim:
      raise ValueError(
        f"Student produced {joint_actions.shape[-1]} actions, expected "
        f"{self.student_action_dim}."
      )
    return torch.cat(
      [joint_actions, teacher_actions[..., self.student_action_dim :]], dim=-1
    )

  def act(self, obs: TensorDict) -> torch.Tensor:
    """Label every state and execute either Teacher or Student per environment."""
    student_actions = super().act(obs)
    if not self.using_ppo:
      # Pure distillation does not optimize the Gaussian exploration std. Execute
      # the deterministic mean so Student-controlled rollouts are not corrupted
      # by a permanently initialized std=1 distribution.
      student_actions = self.actor.output_mean.detach()
      self.transition.actions = student_actions
    teacher_actions = self.get_teacher_actions(obs).to(
      device=student_actions.device,
      dtype=student_actions.dtype,
    )
    teacher_labels = self._teacher_student_labels(teacher_actions)
    if teacher_labels.shape != student_actions.shape:
      raise ValueError(
        "Teacher joint labels and Student actions differ: "
        f"{tuple(teacher_labels.shape)} != {tuple(student_actions.shape)}"
      )
    if (
      self.use_teacher_act_mask is None
      or self.use_teacher_act_mask.shape[0] != student_actions.shape[0]
    ):
      self.prepare_rollout(student_actions.shape[0])

    teacher_mask = self.use_teacher_act_mask
    action_mask = teacher_mask.reshape(
      teacher_mask.shape[0], *([1] * (student_actions.ndim - 1))
    )
    joint_actions = torch.where(action_mask, teacher_labels, student_actions)
    actions = self._compose_environment_actions(joint_actions, teacher_actions)

    self.transition.privileged_actions = teacher_labels
    self.transition.student_control_mask = (~teacher_mask).reshape(-1, 1)
    return actions

  def get_environment_policy(
    self, phase_source: Literal["teacher", "zero"] = "teacher"
  ) -> nn.Module:
    """Return deterministic environment actions using the selected phase source."""
    if phase_source == "zero":
      policy = StudentWithZeroPhasePolicy(
        self._raw_actor,
        self.student_action_dim,
        self.env_action_dim,
      )
      policy.eval()
      return policy
    if phase_source != "teacher":
      raise ValueError(f"Unknown TPPO phase source: {phase_source!r}.")
    if not self.teacher_loaded:
      raise RuntimeError("Cannot use Teacher phase before Teacher is loaded.")
    policy = StudentWithTeacherPhasePolicy(
      self._raw_actor,
      self._raw_teacher,
      self.student_action_dim,
      self.env_action_dim,
    )
    policy.eval()
    return policy

  def get_training_debug_policy(self) -> nn.Module:
    """Return a debug policy using the current episodic DAgger mix."""
    if not self.teacher_loaded:
      raise RuntimeError("Cannot build TPPO training debug policy without Teacher.")
    policy = TeacherStudentTrainingPolicy(
      self._raw_actor,
      self._raw_teacher,
      self.student_action_dim,
      self.env_action_dim,
      self.teacher_action_probability,
    )
    policy.eval()
    return policy

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    """Store the transition while retaining each env fixed controller role."""
    super().process_env_step(obs, rewards, dones, extras)
    self.teacher.reset(dones)

  def compute_returns(self, obs: TensorDict) -> None:
    if self.using_ppo:
      super().compute_returns(obs)

  def compute_distillation_loss(
    self, student_actions: torch.Tensor, teacher_actions: torch.Tensor
  ) -> torch.Tensor:
    """Return per-sample Student-to-Teacher action loss."""
    if self.distill_target == "real":
      return torch.linalg.vector_norm(student_actions - teacher_actions, dim=-1)
    if self.distill_target == "mse_sum":
      return F.mse_loss(student_actions, teacher_actions, reduction="none").sum(dim=-1)
    if self.distill_target == "l1":
      return torch.linalg.vector_norm(student_actions - teacher_actions, ord=1, dim=-1)
    if self.distill_target in {"tanh", "scaled_tanh"}:
      epsilon = torch.finfo(student_actions.dtype).eps
      student_unit = ((student_actions + 1.0) * 0.5).clamp(epsilon, 1.0 - epsilon)
      teacher_unit = ((teacher_actions + 1.0) * 0.5).clamp(epsilon, 1.0 - epsilon)
      binary_cross_entropy = F.binary_cross_entropy(
        student_unit,
        teacher_unit,
        reduction="none",
      ).mean(dim=-1)
      if self.distill_target == "tanh":
        return 2.0 * binary_cross_entropy
      l1_distance = torch.linalg.vector_norm(
        student_actions - teacher_actions, ord=1, dim=-1
      )
      return 2.0 * binary_cross_entropy * l1_distance / student_actions.shape[-1]
    if self.distill_target == "max_log_prob":
      return -self.actor.get_output_log_prob(teacher_actions)
    if self.distill_target == "kl":
      raise NotImplementedError("TPPO distill_target='kl' is not implemented.")
    raise ValueError(f"Unknown TPPO distill_target: {self.distill_target}")

  def compute_auxiliary_distillation_loss(
    self,
    batch: RolloutStorage.Batch,
    student_action_mean: torch.Tensor,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return optional task-specific losses without changing standard TPPO.

    Subclasses can add encoder supervision here. The base implementation is a
    true no-op, so existing TPPO task configurations keep their prior loss.
    """
    zero = student_action_mean.sum() * 0.0
    return zero, {}

  def _augment_tppo_batch(
    self, batch: RolloutStorage.Batch, original_batch_size: int
  ) -> None:
    """Apply the configured action symmetry to Teacher labels and masks."""
    if not self.symmetry or not self.symmetry.use_data_augmentation:
      return
    _, privileged_actions = self.symmetry.data_augmentation_func(
      env=self.symmetry.env,
      obs=None,
      actions=batch.privileged_actions,
    )
    batch.privileged_actions = privileged_actions
    num_augmentations = int(batch.observations.batch_size[0] / original_batch_size)
    batch.student_control_mask = batch.student_control_mask.repeat(
      num_augmentations, *([1] * (batch.student_control_mask.ndim - 1))
    )

  def _compute_critic_values(self, batch: RolloutStorage.Batch) -> torch.Tensor:
    """Evaluate Critic values, allowing task-specific latent reuse."""
    return self.critic(
      batch.observations,
      masks=batch.masks,
      hidden_state=batch.hidden_states[1],
    )

  def update(self) -> dict[str, float]:
    """Optimize Student PPO and distillation losses while keeping Teacher frozen."""
    totals = {
      "value": 0.0,
      "surrogate": 0.0,
      "entropy": 0.0,
      "distillation": 0.0,
    }
    mean_rnd_loss = 0.0 if self.rnd and self.using_ppo else None
    mean_symmetry_loss = 0.0 if self.symmetry else None

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    num_updates = 0
    for batch in generator:
      num_updates += 1
      original_batch_size = batch.observations.batch_size[0]
      original_student_mask = batch.student_control_mask

      if self.using_ppo and self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1.0e-8
          )

      if self.symmetry:
        self.symmetry.augment_batch(batch, original_batch_size)
        self._augment_tppo_batch(batch, original_batch_size)

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      student_action_mean = self.actor.output_mean
      distillation_loss = self.compute_distillation_loss(
        student_action_mean,
        batch.privileged_actions,
      ).mean()
      loss = self.current_distillation_loss_coef * distillation_loss
      auxiliary_loss, auxiliary_metrics = self.compute_auxiliary_distillation_loss(
        batch,
        student_action_mean,
      )
      loss = loss + auxiliary_loss
      for name, value in auxiliary_metrics.items():
        totals.setdefault(name, 0.0)
        totals[name] += value.detach().item()

      zero = loss.detach() * 0.0
      surrogate_loss = zero
      value_loss = zero
      entropy_loss = zero
      rnd_loss = None

      if self.using_ppo:
        actions_log_prob = self.actor.get_output_log_prob(batch.actions)
        values = self._compute_critic_values(batch)
        distribution_params = tuple(
          parameter[:original_batch_size]
          for parameter in self.actor.output_distribution_params
        )
        entropy = self.actor.output_entropy[:original_batch_size]

        if self.desired_kl is not None and self.schedule == "adaptive":
          with torch.inference_mode():
            kl = self.actor.get_kl_divergence(
              batch.old_distribution_params, distribution_params
            )
            kl_mean = torch.mean(kl)
            if self.is_multi_gpu:
              torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
              kl_mean /= self.gpu_world_size
            if self.gpu_global_rank == 0:
              if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
              elif 0.0 < kl_mean < self.desired_kl / 2.0:
                self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
            if self.is_multi_gpu:
              learning_rate = torch.tensor(self.learning_rate, device=self.device)
              torch.distributed.broadcast(learning_rate, src=0)
              self.learning_rate = learning_rate.item()
            for parameter_group in self.optimizer.param_groups:
              parameter_group["lr"] = self.learning_rate

        ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob.squeeze(-1))
        advantages = batch.advantages.squeeze(-1)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(
          ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = _masked_mean(
          torch.maximum(surrogate, surrogate_clipped),
          batch.student_control_mask,
        )

        if self.use_clipped_value_loss:
          value_clipped = batch.values + (values - batch.values).clamp(
            -self.clip_param, self.clip_param
          )
          value_losses = (values - batch.returns).pow(2)
          value_losses_clipped = (value_clipped - batch.returns).pow(2)
          value_loss = torch.maximum(value_losses, value_losses_clipped).mean()
        else:
          value_loss = (batch.returns - values).pow(2).mean()

        entropy_loss = _masked_mean(entropy, original_student_mask)
        loss = (
          loss
          + surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy_loss
        )
        if self.rnd:
          rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size])

      symmetry_loss = None
      if self.symmetry:
        symmetry_loss = self.symmetry.compute_loss(
          self.actor, batch, original_batch_size
        )
        if self.symmetry.use_mirror_loss:
          loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd and rnd_loss is not None:
        self.rnd.optimizer.zero_grad()
        rnd_loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.rnd and rnd_loss is not None:
        self.rnd.optimizer.step()

      totals["value"] += value_loss.item()
      totals["surrogate"] += surrogate_loss.item()
      totals["entropy"] += entropy_loss.item()
      totals["distillation"] += distillation_loss.item()
      if mean_rnd_loss is not None and rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None and symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    if num_updates == 0:
      raise RuntimeError("TPPO update received no mini-batches.")

    loss_dict = {name: total / num_updates for name, total in totals.items()}
    loss_dict["teacher_action_probability"] = self.teacher_action_probability
    loss_dict["teacher_control_fraction"] = 1.0 - float(
      self.storage.student_control_masks.float().mean().item()
    )
    loss_dict["teacher_quota_fraction"] = (
      float(self.use_teacher_act_mask.float().mean().item())
      if self.use_teacher_act_mask is not None
      else 0.0
    )
    loss_dict["controller_role_switches"] = float(self.last_role_switch_count)
    loss_dict["distillation_coefficient"] = self.current_distillation_loss_coef
    if mean_rnd_loss is not None:
      loss_dict["rnd"] = mean_rnd_loss / num_updates
    if mean_symmetry_loss is not None:
      loss_dict["symmetry"] = mean_symmetry_loss / num_updates

    self.storage.clear()
    self.update_count += 1
    return loss_dict

  def train_mode(self) -> None:
    super().train_mode()
    self.teacher.eval()

  def eval_mode(self) -> None:
    super().eval_mode()
    self.teacher.eval()

  def save(self) -> dict[str, Any]:
    saved_dict = super().save()
    saved_dict["teacher_state_dict"] = self._raw_teacher.state_dict()
    saved_dict["tppo_update_count"] = self.update_count
    saved_dict["teacher_checkpoint"] = self.teacher_checkpoint
    return saved_dict

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    should_load_teacher = load_cfg is None or load_cfg.get("teacher", True)
    if should_load_teacher and "teacher_state_dict" in loaded_dict:
      self.load_teacher_state_dict(loaded_dict["teacher_state_dict"], strict)
      self.teacher_checkpoint = loaded_dict.get("teacher_checkpoint")
    if load_cfg is None or load_cfg.get("iteration", False):
      self.update_count = int(loaded_dict.get("tppo_update_count", 0))
    return load_iteration

  def compile(self, mode: str | None = None) -> None:
    self.actor = compile_model(self._raw_actor, mode)  # type: ignore[assignment]
    self.critic = compile_model(self._raw_critic, mode)  # type: ignore[assignment]

  def broadcast_parameters(self) -> None:
    model_parameters = [
      self._raw_actor.state_dict(),
      self._raw_critic.state_dict(),
      self._raw_teacher.state_dict(),
    ]
    if self.rnd:
      model_parameters.append(self.rnd.predictor.state_dict())
    torch.distributed.broadcast_object_list(model_parameters, src=0)
    self._raw_actor.load_state_dict(model_parameters[0])
    self._raw_critic.load_state_dict(model_parameters[1])
    self.load_teacher_state_dict(model_parameters[2])
    if self.rnd:
      self.rnd.predictor.load_state_dict(model_parameters[3])

  @staticmethod
  def _clean_model_cfg(model_cfg: dict[str, Any]) -> dict[str, Any]:
    for optional_key in ("cnn_cfg", "distribution_cfg"):
      if model_cfg.get(optional_key) is None:
        model_cfg.pop(optional_key, None)
    if model_cfg.get("rnn_type") is None:
      for recurrent_key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
        model_cfg.pop(recurrent_key, None)
    return model_cfg

  @staticmethod
  def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> TPPO:
    """Construct Student, critic, frozen Teacher, storage, and TPPO."""
    algorithm_class: type[TPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))
    actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
    critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))
    teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))

    default_sets = ["actor", "critic", "teacher"]
    if cfg["algorithm"].get("rnd_cfg") is not None:
      default_sets.append("rnd_state")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

    student_action_dim = int(cfg["algorithm"]["student_action_dim"])
    if not 0 < student_action_dim < env.num_actions:
      raise ValueError(
        "TPPO distillation requires Student actions to be a strict prefix of "
        f"the environment actions, got {student_action_dim} of {env.num_actions}."
      )

    actor = actor_class(
      obs,
      cfg["obs_groups"],
      "actor",
      student_action_dim,
      **TPPO._clean_model_cfg(cfg["actor"]),
    ).to(device)
    print(f"Student Actor Model: {actor}")
    if cfg["algorithm"].pop("share_cnn_encoders", None):
      cfg["critic"]["cnns"] = actor.cnns  # type: ignore[attr-defined]
    critic = critic_class(
      obs,
      cfg["obs_groups"],
      "critic",
      1,
      **TPPO._clean_model_cfg(cfg["critic"]),
    ).to(device)
    print(f"Critic Model: {critic}")
    teacher = teacher_class(
      obs,
      cfg["obs_groups"],
      "teacher",
      env.num_actions,
      **TPPO._clean_model_cfg(cfg["teacher"]),
    ).to(device)
    print(f"Teacher Actor Model: {teacher}")

    storage = TppoRolloutStorage(
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      [student_action_dim],
      device,
    )
    algorithm = algorithm_class(
      actor,
      critic,
      teacher,
      storage,
      device=device,
      env_action_dim=env.num_actions,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg["multi_gpu"],
    )
    algorithm.compile(cfg.get("torch_compile_mode"))
    return algorithm
