"""TPPO extension for privileged-reference temporal intent distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from rsl_rl.storage import RolloutStorage

from .intent_transformer import IntentTransformerActor, IntentTransformerCritic
from .tppo import TPPO


class IntentTPPO(TPPO):
  """TPPO with privileged reference reconstruction and latent alignment.

  Rollout actions and the PPO branch use only ``z_S``. ``z_T`` is anchored by
  future-reference reconstruction. Optional privileged action supervision is
  kept for legacy experiments, while new tasks disable it so PPO can depart
  from frozen Teacher actions.
  """

  def __init__(
    self,
    *args,
    intent_reference_loss_coef: float = 0.0,
    intent_teacher_action_loss_coef: float = 1.0,
    intent_latent_loss_coef: float = 0.05,
    freeze_intent_encoders: bool = False,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    if not isinstance(self._raw_actor, IntentTransformerActor):
      raise TypeError(
        "IntentTPPO requires IntentTransformerActor; "
        f"got {type(self._raw_actor).__name__}."
      )
    if not isinstance(self._raw_critic, IntentTransformerCritic):
      raise TypeError(
        "IntentTPPO requires IntentTransformerCritic; "
        f"got {type(self._raw_critic).__name__}."
    )
    self._raw_critic.attach_intent_actor(self._raw_actor)
    self.intent_reference_loss_coef = float(intent_reference_loss_coef)
    self.intent_teacher_action_loss_coef = float(intent_teacher_action_loss_coef)
    self.intent_latent_loss_coef = float(intent_latent_loss_coef)
    self.freeze_intent_encoders = bool(freeze_intent_encoders)
    if self.intent_reference_loss_coef < 0.0:
      raise ValueError("intent_reference_loss_coef must be non-negative.")
    if self.intent_teacher_action_loss_coef < 0.0:
      raise ValueError("intent_teacher_action_loss_coef must be non-negative.")
    if self.intent_latent_loss_coef < 0.0:
      raise ValueError("intent_latent_loss_coef must be non-negative.")
    if self.freeze_intent_encoders:
      if any(
        coefficient != 0.0
        for coefficient in (
          self.intent_reference_loss_coef,
          self.intent_teacher_action_loss_coef,
          self.intent_latent_loss_coef,
        )
      ):
        raise ValueError(
          "Frozen intent encoders require all intent auxiliary coefficients to be zero."
        )
      self.intent_actor.freeze_intent_encoders()
    self._critic_teacher_intent: torch.Tensor | None = None

  @property
  def intent_actor(self) -> IntentTransformerActor:
    if not isinstance(self._raw_actor, IntentTransformerActor):
      raise TypeError("IntentTPPO actor unexpectedly changed type.")
    return self._raw_actor

  @property
  def intent_critic(self) -> IntentTransformerCritic:
    if not isinstance(self._raw_critic, IntentTransformerCritic):
      raise TypeError("IntentTPPO critic unexpectedly changed type.")
    return self._raw_critic

  def compute_auxiliary_distillation_loss(
    self,
    batch: RolloutStorage.Batch,
    student_action_mean: torch.Tensor,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    z_s = self.intent_actor.consume_latest_student_intent()
    if self.freeze_intent_encoders:
      with torch.no_grad():
        z_t = self.intent_actor.teacher_intent(batch.observations)
      self._critic_teacher_intent = z_t if self.using_ppo else None
      zero = student_action_mean.sum() * 0.0
      return zero, {
        "intent_reference_reconstruction": zero.detach(),
        "intent_latent": zero.detach(),
      }
    z_t = self.intent_actor.teacher_intent(batch.observations)
    self._critic_teacher_intent = z_t.detach() if self.using_ppo else None
    reference_prediction, reference_target = (
      self.intent_actor.reconstruct_future_reference(batch.observations, z_t)
    )
    reference_loss = F.smooth_l1_loss(reference_prediction, reference_target)
    latent_loss = F.smooth_l1_loss(z_s, z_t.detach())
    loss = self.intent_reference_loss_coef * reference_loss
    loss = loss + self.intent_latent_loss_coef * latent_loss
    metrics = {
      "intent_reference_reconstruction": reference_loss,
      "intent_latent": latent_loss,
    }
    if self.intent_teacher_action_loss_coef > 0.0:
      teacher_intent_actions = self.intent_actor.teacher_intent_action(
        batch.observations,
        z_t=z_t,
      )
      teacher_intent_action_loss = self.compute_distillation_loss(
        teacher_intent_actions,
        batch.privileged_actions,
      ).mean()
      loss = loss + (
        self.current_distillation_loss_coef
        * self.intent_teacher_action_loss_coef
        * teacher_intent_action_loss
      )
      metrics["intent_teacher_action_distillation"] = teacher_intent_action_loss
    return loss, metrics

  def _compute_critic_values(self, batch: RolloutStorage.Batch) -> torch.Tensor:
    """Reuse z_T from the auxiliary losses for the immediately following Critic."""
    if self._critic_teacher_intent is None:
      raise RuntimeError(
        "Intent Critic requires z_T from the current mini-batch auxiliary loss."
      )
    z_t = self._critic_teacher_intent
    self._critic_teacher_intent = None
    return self.intent_critic.forward_with_teacher_intent(batch.observations, z_t)

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    """Load legacy 6-D ball-token checkpoints into the robust 8-D model."""
    saved_weight = loaded_dict.get("actor_state_dict", {}).get("ball_input.weight")
    upgrades_ball_tokens = (
      self.intent_actor.ball_token_dim == 8
      and saved_weight is not None
      and saved_weight.shape[-1] == 6
    )
    if upgrades_ball_tokens and load_cfg is None:
      # The Actor hook expands the affected tensors. Optimizer moments still
      # have the legacy shapes, so restart only the optimizer state.
      load_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": True,
        "rnd": True,
        "teacher": True,
      }
      print(
        "[INFO]: Upgrading Intent ball tokens 6D -> 8D; "
        "loading model/iteration and restarting optimizer state."
      )
    return super().load(loaded_dict, load_cfg, strict)
