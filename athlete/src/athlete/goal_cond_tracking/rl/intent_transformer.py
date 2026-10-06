"""Privileged-reference and deployable temporal intent encoders for TPPO."""

from __future__ import annotations

import copy
import weakref

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from tensordict import TensorDict
from torch import nn


class IntentTransformerActor(MLPModel):
  """Student actor conditioned on a deployable temporal intent latent.

  The normal actor forward path always consumes ``z_S``. During training,
  ``z_T`` receives privileged future-reference reconstruction supervision and
  can supervise ``z_S`` with a stop-gradient latent loss. This keeps the
  inference contract free of future reference motion.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    *,
    state_history_group: str = "intent_state_history",
    ball_history_group: str = "intent_ball_history",
    reference_future_group: str = "intent_reference_future",
    state_history_steps: int = 5,
    state_token_dim: int = 93,
    ball_history_steps: int = 5,
    ball_token_dim: int = 6,
    reference_steps: int = 12,
    reference_token_dim: int = 58,
    intent_latent_dim: int = 128,
    transformer_heads: int = 4,
    transformer_layers: int = 2,
    transformer_ff_dim: int = 256,
    **kwargs,
  ) -> None:
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    if intent_latent_dim % transformer_heads != 0:
      raise ValueError("intent_latent_dim must be divisible by transformer_heads.")

    self.state_history_group = state_history_group
    self.ball_history_group = ball_history_group
    self.reference_future_group = reference_future_group
    self.state_history_steps = int(state_history_steps)
    self.state_token_dim = int(state_token_dim)
    self.ball_history_steps = int(ball_history_steps)
    self.ball_token_dim = int(ball_token_dim)
    self.reference_steps = int(reference_steps)
    self.reference_token_dim = int(reference_token_dim)
    self.intent_latent_dim = int(intent_latent_dim)
    self._latest_student_intent: torch.Tensor | None = None
    self._intent_encoders_frozen = False

    self._validate_group_dim(
      obs, self.state_history_group, self.state_history_steps * self.state_token_dim
    )
    self._validate_group_dim(
      obs, self.ball_history_group, self.ball_history_steps * self.ball_token_dim
    )
    self._validate_group_dim(
      obs,
      self.reference_future_group,
      self.reference_steps * self.reference_token_dim,
    )

    self.state_input = nn.Linear(self.state_token_dim, self.intent_latent_dim)
    self.ball_input = nn.Linear(self.ball_token_dim, self.intent_latent_dim)
    self.reference_input = nn.Linear(
      self.reference_token_dim, self.intent_latent_dim
    )
    # Statistics are shared over temporal positions, while each feature type
    # keeps its own physical units and distribution.
    self.state_history_normalizer = EmpiricalNormalization(self.state_token_dim)
    self.ball_history_normalizer = EmpiricalNormalization(self.ball_token_dim)
    self.reference_future_normalizer = EmpiricalNormalization(
      self.reference_token_dim
    )
    self.student_position = nn.Parameter(
      torch.zeros(1, self.state_history_steps, self.intent_latent_dim)
    )
    self.reference_position = nn.Parameter(
      torch.zeros(1, self.reference_steps, self.intent_latent_dim)
    )
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=self.intent_latent_dim,
      nhead=transformer_heads,
      dim_feedforward=transformer_ff_dim,
      activation="gelu",
      dropout=0.0,
      batch_first=True,
      norm_first=False,
    )
    self.student_temporal_encoder = nn.TransformerEncoder(
      encoder_layer, num_layers=transformer_layers
    )
    teacher_state_layer = nn.TransformerEncoderLayer(
      d_model=self.intent_latent_dim,
      nhead=transformer_heads,
      dim_feedforward=transformer_ff_dim,
      activation="gelu",
      dropout=0.0,
      batch_first=True,
      norm_first=False,
    )
    self.teacher_state_encoder = nn.TransformerEncoder(
      teacher_state_layer, num_layers=transformer_layers
    )
    self.reference_cross_attention = nn.MultiheadAttention(
      self.intent_latent_dim,
      transformer_heads,
      batch_first=True,
    )
    self.student_intent_norm = nn.LayerNorm(self.intent_latent_dim)
    self.teacher_intent_norm = nn.LayerNorm(self.intent_latent_dim)
    self.reference_decoder = nn.Sequential(
      nn.Linear(self.intent_latent_dim, transformer_ff_dim),
      nn.GELU(),
      nn.Linear(
        transformer_ff_dim, self.reference_steps * self.reference_token_dim
      ),
    )

    mlp_output_dim = (
      self.distribution.input_dim if self.distribution is not None else output_dim
    )
    hidden_dims = tuple(
      layer.out_features
      for layer in self.mlp
      if isinstance(layer, nn.Linear)
    )[:-1]
    activation = self._activation_name()
    self.mlp = MLP(
      self.obs_dim + self.intent_latent_dim,
      mlp_output_dim,
      hidden_dims,
      activation,
    )
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)
    self._init_transformer_weights()

  @staticmethod
  def _validate_group_dim(obs: TensorDict, group: str, expected_dim: int) -> None:
    if group not in obs:
      raise KeyError(f"IntentTransformerActor requires observation group {group!r}.")
    actual_dim = int(obs[group].shape[-1])
    if actual_dim != expected_dim:
      raise ValueError(
        f"Observation group {group!r} has {actual_dim} dims, expected {expected_dim}."
      )

  def _activation_name(self) -> str:
    for module in self.mlp:
      if isinstance(module, nn.ELU):
        return "elu"
      if isinstance(module, nn.ReLU):
        return "relu"
      if isinstance(module, nn.LeakyReLU):
        return "leaky_relu"
    raise ValueError("IntentTransformerActor supports only ELU/ReLU/LeakyReLU MLPs.")

  def _init_transformer_weights(self) -> None:
    for module in (
      self.state_input,
      self.ball_input,
      self.reference_input,
    ):
      nn.init.xavier_uniform_(module.weight)
    nn.init.zeros_(module.bias)
    nn.init.normal_(self.student_position, std=0.02)
    nn.init.normal_(self.reference_position, std=0.02)
    for module in self.reference_decoder:
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)

  @property
  def inference_obs_groups(self) -> tuple[str, ...]:
    """Return current and temporal groups needed by the deployable z_S path.

    ``obs_groups`` must remain limited to the current Student group because
    ``MLPModel`` uses it to build the 133-D MLP input. Fast Play needs these
    additional groups only for the temporal encoder.
    """
    return tuple(
      dict.fromkeys(
        (*self.obs_groups, self.state_history_group, self.ball_history_group)
      )
    )

  def _state_tokens(self, obs: TensorDict) -> torch.Tensor:
    states = obs[self.state_history_group].reshape(
      -1, self.state_history_steps, self.state_token_dim
    )
    states = self.state_history_normalizer(states)
    return self.state_input(states) + self.student_position

  def _ball_tokens(self, obs: TensorDict) -> torch.Tensor:
    if self.ball_history_steps != self.state_history_steps:
      raise ValueError("Student state and ball history must use the same token count.")
    ball = self.unpack_ball_history(
      obs[self.ball_history_group], self.ball_history_steps, self.ball_token_dim
    )
    ball = self.ball_history_normalizer(ball)
    return self.ball_input(ball)

  @staticmethod
  def unpack_ball_history(
    values: torch.Tensor, history_steps: int, token_dim: int
  ) -> torch.Tensor:
    """Convert observation-manager layout into chronological ball tokens.

    The observation group concatenates terms, so its flat layout is all
    positions followed by velocities and optional visibility/age values. It is
    not directly a sequence of per-time-step tokens.
    """
    if token_dim not in {6, 8}:
      raise ValueError(
        "Ball token layout requires position(3) + velocity(3), optionally "
        "+ valid(1) + age(1)."
      )
    split = history_steps * 3
    positions = values[..., :split].reshape(-1, history_steps, 3)
    velocities = values[..., split : 2 * split].reshape(-1, history_steps, 3)
    token_parts = [positions, velocities]
    if token_dim == 8:
      status_start = 2 * split
      valid = values[..., status_start : status_start + history_steps].reshape(
        -1, history_steps, 1
      )
      age = values[..., status_start + history_steps :].reshape(
        -1, history_steps, 1
      )
      token_parts.extend((valid, age))
    return torch.cat(token_parts, dim=-1)

  def _load_from_state_dict(
    self,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
  ) -> None:
    """Upgrade a 6-D ball-token checkpoint when initializing the 8-D model."""
    weight_key = f"{prefix}ball_input.weight"
    weight = state_dict.get(weight_key)
    if (
      self.ball_token_dim == 8
      and weight is not None
      and tuple(weight.shape) == (self.intent_latent_dim, 6)
    ):
      upgraded = weight.new_zeros(self.intent_latent_dim, 8)
      upgraded[:, :6] = weight
      state_dict[weight_key] = upgraded
      for suffix, fill_value in (
        ("_mean", 0.0),
        ("_var", 1.0),
        ("_std", 1.0),
      ):
        key = f"{prefix}ball_history_normalizer.{suffix}"
        value = state_dict.get(key)
        if value is None or value.shape[-1] != 6:
          continue
        expanded = value.new_full((*value.shape[:-1], 8), fill_value)
        expanded[..., :6] = value
        state_dict[key] = expanded
    super()._load_from_state_dict(
      state_dict,
      prefix,
      local_metadata,
      strict,
      missing_keys,
      unexpected_keys,
      error_msgs,
    )

  def student_intent(self, obs: TensorDict) -> torch.Tensor:
    """Encode only the deployment-time ball and robot state histories."""
    tokens = self._state_tokens(obs) + self._ball_tokens(obs)
    encoded = self.student_temporal_encoder(tokens)
    return self.student_intent_norm(encoded[:, -1])

  def teacher_intent(self, obs: TensorDict) -> torch.Tensor:
    """Encode privileged future reference conditioned on state-history query."""
    state_tokens = self.teacher_state_encoder(self._state_tokens(obs))
    reference_tokens = obs[self.reference_future_group].reshape(
      -1, self.reference_steps, self.reference_token_dim
    )
    reference_tokens = self.reference_future_normalizer(reference_tokens)
    reference_tokens = self.reference_input(reference_tokens) + self.reference_position
    attended, _ = self.reference_cross_attention(
      state_tokens[:, -1:], reference_tokens, reference_tokens, need_weights=False
    )
    return self.teacher_intent_norm(attended[:, 0])

  def normalized_future_reference(self, obs: TensorDict) -> torch.Tensor:
    """Return the normalized 12 x 58 Teacher-only reference target."""
    reference = obs[self.reference_future_group].reshape(
      -1, self.reference_steps, self.reference_token_dim
    )
    return self.reference_future_normalizer(reference).reshape(
      reference.shape[0], -1
    )

  def reconstruct_future_reference(
    self, obs: TensorDict, z_t: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode ``z_T`` back into normalized future reference joint states."""
    if z_t is None:
      z_t = self.teacher_intent(obs)
    prediction = self.reference_decoder(z_t)
    with torch.no_grad():
      target = self.normalized_future_reference(obs)
    return prediction, target

  def intent_latents(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the deployable ``z_S`` and privileged ``z_T`` latents."""
    return self.student_intent(obs), self.teacher_intent(obs)

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    del masks, hidden_state
    student_obs = super().get_latent(obs)
    z_s = self.student_intent(obs)
    self._latest_student_intent = z_s
    return torch.cat([student_obs, z_s], dim=-1)

  def consume_latest_student_intent(self) -> torch.Tensor:
    """Consume z_S from the immediately preceding Actor forward pass."""
    if self._latest_student_intent is None:
      raise RuntimeError(
        "No cached z_S is available; run the Intent Actor before auxiliary loss."
      )
    z_s = self._latest_student_intent
    self._latest_student_intent = None
    return z_s

  def freeze_intent_encoders(self) -> None:
    """Freeze deployment and privileged intent representations in-place."""
    for module in (
      self.state_input,
      self.ball_input,
      self.reference_input,
      self.student_temporal_encoder,
      self.teacher_state_encoder,
      self.reference_cross_attention,
      self.student_intent_norm,
      self.teacher_intent_norm,
      self.reference_decoder,
    ):
      module.requires_grad_(False)
    self.student_position.requires_grad_(False)
    self.reference_position.requires_grad_(False)
    self._intent_encoders_frozen = True

  def teacher_intent_action(
    self, obs: TensorDict, z_t: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Evaluate the shared action head using privileged ``z_T`` during training."""
    student_obs = super().get_latent(obs)
    if z_t is None:
      z_t = self.teacher_intent(obs)
    mlp_output = self.mlp(torch.cat([student_obs, z_t], dim=-1))
    if self.distribution is not None:
      return self.distribution.deterministic_output(mlp_output)
    return mlp_output

  def update_normalization(self, obs: TensorDict) -> None:
    """Update current-observation and per-token temporal statistics."""
    super().update_normalization(obs)
    if self._intent_encoders_frozen:
      return
    self.state_history_normalizer.update(
      obs[self.state_history_group].reshape(-1, self.state_token_dim)
    )
    self.ball_history_normalizer.update(
      self.unpack_ball_history(
        obs[self.ball_history_group],
        self.ball_history_steps,
        self.ball_token_dim,
      ).reshape(-1, self.ball_token_dim)
    )
    self.reference_future_normalizer.update(
      obs[self.reference_future_group].reshape(-1, self.reference_token_dim)
    )

  def as_onnx(self, verbose: bool) -> nn.Module:
    """Export the deployable z_S branch without reference-motion inputs."""
    return _OnnxIntentTransformerActor(self, verbose)


class IntentTransformerCritic(MLPModel):
  """Privileged critic conditioned on the Actor's stopped-gradient ``z_T``.

  The reference encoder remains owned by the Actor. The Critic receives its
  exact privileged latent but cannot update it through the value loss, keeping
  the policy's intent representation trained only by its action/latent losses.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    *,
    intent_latent_dim: int = 128,
    **kwargs,
  ) -> None:
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    self.intent_latent_dim = int(intent_latent_dim)
    hidden_dims = tuple(
      layer.out_features
      for layer in self.mlp
      if isinstance(layer, nn.Linear)
    )[:-1]
    self.mlp = MLP(
      self.obs_dim + self.intent_latent_dim,
      output_dim,
      hidden_dims,
      self._activation_name(),
    )
    self._intent_actor_ref: weakref.ReferenceType[IntentTransformerActor] | None = None

  def _activation_name(self) -> str:
    for module in self.mlp:
      if isinstance(module, nn.ELU):
        return "elu"
      if isinstance(module, nn.ReLU):
        return "relu"
      if isinstance(module, nn.LeakyReLU):
        return "leaky_relu"
    raise ValueError("IntentTransformerCritic supports only ELU/ReLU/LeakyReLU MLPs.")

  def attach_intent_actor(self, actor: IntentTransformerActor) -> None:
    """Attach without registering Actor parameters in the Critic optimizer."""
    if actor.intent_latent_dim != self.intent_latent_dim:
      raise ValueError(
        "Actor/Critic intent dimensions differ: "
        f"{actor.intent_latent_dim} != {self.intent_latent_dim}."
      )
    self._intent_actor_ref = weakref.ref(actor)

  def _intent_actor(self) -> IntentTransformerActor:
    if self._intent_actor_ref is None:
      raise RuntimeError("IntentTransformerCritic has no attached IntentTransformerActor.")
    actor = self._intent_actor_ref()
    if actor is None:
      raise RuntimeError("Attached IntentTransformerActor no longer exists.")
    return actor

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    del masks, hidden_state
    critic_obs = super().get_latent(obs)
    with torch.no_grad():
      z_t = self._intent_actor().teacher_intent(obs)
    return torch.cat([critic_obs, z_t], dim=-1)

  def forward_with_teacher_intent(
    self,
    obs: TensorDict,
    z_t: torch.Tensor,
  ) -> torch.Tensor:
    """Evaluate value with an already-computed stopped-gradient Teacher intent."""
    critic_obs = super().get_latent(obs)
    if z_t.shape[:-1] != critic_obs.shape[:-1]:
      raise ValueError(
        "Teacher intent and Critic observation batch shapes differ: "
        f"{tuple(z_t.shape[:-1])} != {tuple(critic_obs.shape[:-1])}."
      )
    if z_t.shape[-1] != self.intent_latent_dim:
      raise ValueError(
        f"Teacher intent has {z_t.shape[-1]} dims, expected {self.intent_latent_dim}."
      )
    return self.mlp(torch.cat([critic_obs, z_t.detach()], dim=-1))


class _OnnxIntentTransformerActor(nn.Module):
  """Single-input ONNX wrapper for the Student-only temporal policy."""

  is_recurrent: bool = False

  def __init__(self, model: IntentTransformerActor, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.state_input = copy.deepcopy(model.state_input)
    self.ball_input = copy.deepcopy(model.ball_input)
    self.state_history_normalizer = copy.deepcopy(model.state_history_normalizer)
    self.ball_history_normalizer = copy.deepcopy(model.ball_history_normalizer)
    self.student_position = copy.deepcopy(model.student_position)
    self.student_temporal_encoder = copy.deepcopy(model.student_temporal_encoder)
    self.student_intent_norm = copy.deepcopy(model.student_intent_norm)
    self.mlp = copy.deepcopy(model.mlp)
    self.state_history_steps = model.state_history_steps
    self.state_token_dim = model.state_token_dim
    self.ball_history_steps = model.ball_history_steps
    self.ball_token_dim = model.ball_token_dim
    self.student_obs_dim = model.obs_dim
    self.input_size = (
      self.student_obs_dim
      + self.state_history_steps * self.state_token_dim
      + self.ball_history_steps * self.ball_token_dim
    )
    self.deterministic_output = (
      model.distribution.as_deterministic_output_module()
      if model.distribution is not None
      else nn.Identity()
    )
    # Frozen training encoders otherwise select PyTorch's fused inference
    # kernel, which the legacy ONNX exporter cannot represent. These are
    # independent export copies; enabling gradients does not unfreeze training.
    self.requires_grad_(True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    student = self.obs_normalizer(x[:, : self.student_obs_dim])
    cursor = self.student_obs_dim
    state_end = cursor + self.state_history_steps * self.state_token_dim
    states = x[:, cursor:state_end].reshape(
      -1, self.state_history_steps, self.state_token_dim
    )
    ball = IntentTransformerActor.unpack_ball_history(
      x[:, state_end:], self.ball_history_steps, self.ball_token_dim
    )
    tokens = self.state_input(self.state_history_normalizer(states))
    tokens = tokens + self.student_position
    tokens = tokens + self.ball_input(self.ball_history_normalizer(ball))
    z_s = self.student_intent_norm(self.student_temporal_encoder(tokens)[:, -1])
    return self.deterministic_output(self.mlp(torch.cat([student, z_s], dim=-1)))

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
