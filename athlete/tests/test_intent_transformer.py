from __future__ import annotations

import torch
from athlete.goal_cond_tracking.rl.intent_transformer import (
  IntentTransformerActor,
  IntentTransformerCritic,
)
from athlete.goal_cond_tracking.rl.tppo import (
  StudentWithTeacherPhasePolicy,
  StudentWithZeroPhasePolicy,
)
from tensordict import TensorDict
from torch import nn


def _intent_observations(num_envs: int, ball_token_dim: int = 6) -> TensorDict:
  return TensorDict(
    {
      "student": torch.randn(num_envs, 133),
      "intent_state_history": torch.randn(num_envs, 5 * 93),
      "intent_ball_history": torch.randn(num_envs, 5 * ball_token_dim),
      "intent_reference_future": torch.randn(num_envs, 12 * 58),
    },
    batch_size=[num_envs],
  )


def test_intent_transformer_trains_both_latents_and_exports_student_branch() -> None:
  torch.manual_seed(3)
  observations = _intent_observations(3)
  actor = IntentTransformerActor(
    observations,
    {"actor": ["student"]},
    "actor",
    output_dim=29,
    hidden_dims=(32,),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )

  student_actions = actor(observations, stochastic_output=True)
  teacher_intent_actions = actor.teacher_intent_action(observations)
  z_s, z_t = actor.intent_latents(observations)
  reference_prediction, reference_target = actor.reconstruct_future_reference(
    observations, z_t
  )
  loss = student_actions.square().mean()
  loss = loss + teacher_intent_actions.square().mean()
  loss = loss + (z_s - z_t.detach()).square().mean()
  loss = loss + (reference_prediction - reference_target).square().mean()
  loss.backward()

  assert actor.student_temporal_encoder.layers[0].self_attn.in_proj_weight.grad is not None
  assert actor.teacher_state_encoder.layers[0].self_attn.in_proj_weight.grad is not None
  assert actor.reference_cross_attention.in_proj_weight.grad is not None
  assert actor.reference_decoder[0].weight.grad is not None
  assert reference_prediction.shape == reference_target.shape == (3, 12 * 58)

  assert tuple(actor.obs_groups) == ("student",)
  assert actor.inference_obs_groups == (
    "student",
    "intent_state_history",
    "intent_ball_history",
  )
  zero_phase_policy = StudentWithZeroPhasePolicy(actor, 29, 30)
  assert zero_phase_policy.obs_groups == actor.inference_obs_groups

  teacher = nn.Identity()
  teacher.obs_groups = ("actor",)  # type: ignore[attr-defined]
  teacher_phase_policy = StudentWithTeacherPhasePolicy(actor, teacher, 29, 30)
  assert teacher_phase_policy.obs_groups == (
    "student",
    "intent_state_history",
    "intent_ball_history",
    "actor",
  )

  actor.update_normalization(observations)
  assert actor.state_history_normalizer.count.item() == 3 * 5
  assert actor.ball_history_normalizer.count.item() == 3 * 5
  assert actor.reference_future_normalizer.count.item() == 3 * 12

  actor.eval()
  onnx_model = actor.as_onnx(verbose=False)
  flattened_student_input = torch.cat(
    [
      observations["student"],
      observations["intent_state_history"],
      observations["intent_ball_history"],
    ],
    dim=-1,
  )
  torch.testing.assert_close(actor(observations), onnx_model(flattened_student_input))


def test_ball_history_is_repacked_into_position_velocity_tokens() -> None:
  positions = torch.tensor(
    [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]]
  )
  velocities = positions + 100.0
  packed_manager_output = torch.cat(
    [positions.reshape(1, -1), velocities.reshape(1, -1)], dim=-1
  )

  tokens = IntentTransformerActor.unpack_ball_history(
    packed_manager_output, history_steps=5, token_dim=6
  )
  torch.testing.assert_close(tokens, torch.cat([positions, velocities], dim=-1))


def test_robust_ball_history_adds_validity_and_measurement_age() -> None:
  positions = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
  velocities = positions + 100.0
  validity = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0]])
  age = torch.tensor([[0.02, 0.04, 0.06, 0.08, 0.0]])
  packed = torch.cat(
    [
      positions.reshape(1, -1),
      velocities.reshape(1, -1),
      validity,
      age,
    ],
    dim=-1,
  )

  tokens = IntentTransformerActor.unpack_ball_history(
    packed, history_steps=5, token_dim=8
  )
  expected = torch.cat(
    [positions, velocities, validity[..., None], age[..., None]], dim=-1
  )
  torch.testing.assert_close(tokens, expected)


def test_robust_actor_loads_legacy_ball_encoder_and_exports_638d() -> None:
  legacy_observations = _intent_observations(2)
  robust_observations = _intent_observations(2, ball_token_dim=8)
  def actor_kwargs() -> dict:
    return {
      "obs_groups": {"actor": ["student"]},
      "obs_set": "actor",
      "output_dim": 29,
      "hidden_dims": (32,),
      "activation": "elu",
      "obs_normalization": True,
      "distribution_cfg": {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    }

  legacy_actor = IntentTransformerActor(legacy_observations, **actor_kwargs())
  robust_actor = IntentTransformerActor(
    robust_observations, ball_token_dim=8, **actor_kwargs()
  )

  legacy_weight = legacy_actor.ball_input.weight.detach().clone()
  robust_actor.load_state_dict(legacy_actor.state_dict(), strict=True)
  torch.testing.assert_close(robust_actor.ball_input.weight[:, :6], legacy_weight)
  torch.testing.assert_close(
    robust_actor.ball_input.weight[:, 6:],
    torch.zeros_like(robust_actor.ball_input.weight[:, 6:]),
  )

  robust_actor.eval()
  onnx_model = robust_actor.as_onnx(verbose=False)
  robust_input = torch.cat(
    [
      robust_observations["student"],
      robust_observations["intent_state_history"],
      robust_observations["intent_ball_history"],
    ],
    dim=-1,
  )
  assert robust_input.shape[-1] == 638
  torch.testing.assert_close(
    robust_actor(robust_observations), onnx_model(robust_input)
  )


def test_freeze_intent_encoders_stops_gradients_and_normalizer_updates() -> None:
  observations = _intent_observations(2)
  actor = IntentTransformerActor(
    observations,
    {"actor": ["student"]},
    "actor",
    output_dim=29,
    hidden_dims=(32,),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  actor.freeze_intent_encoders()
  actor.update_normalization(observations)
  # ``Distribution.sample()`` intentionally detaches from the policy mean;
  # use the deterministic path when checking Actor-head gradients.
  actions = actor(observations)
  z_t = actor.teacher_intent(observations)
  actions.square().mean().backward()

  assert actor.mlp[0].weight.grad is not None
  assert actor.student_temporal_encoder.layers[0].self_attn.in_proj_weight.grad is None
  assert actor.reference_cross_attention.in_proj_weight.grad is None
  assert z_t.requires_grad is False
  assert actor.state_history_normalizer.count.item() == 0
  assert actor.ball_history_normalizer.count.item() == 0
  assert actor.reference_future_normalizer.count.item() == 0


def test_critic_receives_stopped_gradient_actor_teacher_intent() -> None:
  observations = _intent_observations(2)
  observations["critic"] = torch.randn(2, 325)
  obs_groups = {"actor": ["student"], "critic": ["critic"]}
  actor = IntentTransformerActor(
    observations,
    obs_groups,
    "actor",
    output_dim=29,
    hidden_dims=(32,),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  critic = IntentTransformerCritic(
    observations,
    obs_groups,
    "critic",
    output_dim=1,
    hidden_dims=(32,),
    activation="elu",
    obs_normalization=True,
  )
  critic.attach_intent_actor(actor)
  value = critic(observations)
  assert value.shape == (2, 1)
  assert critic.mlp[0].in_features == 325 + 128

  value.square().mean().backward()
  assert critic.mlp[0].weight.grad is not None
  assert actor.reference_cross_attention.in_proj_weight.grad is None


def test_actor_and_critic_reuse_current_minibatch_intents() -> None:
  observations = _intent_observations(2)
  observations["critic"] = torch.randn(2, 325)
  obs_groups = {"actor": ["student"], "critic": ["critic"]}
  actor = IntentTransformerActor(
    observations,
    obs_groups,
    "actor",
    output_dim=29,
    hidden_dims=(32,),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  critic = IntentTransformerCritic(
    observations,
    obs_groups,
    "critic",
    output_dim=1,
    hidden_dims=(32,),
    activation="elu",
    obs_normalization=True,
  )
  critic.attach_intent_actor(actor)

  call_counts = {"student": 0, "teacher": 0}

  def count_student(*_args) -> None:
    call_counts["student"] += 1

  def count_teacher(*_args) -> None:
    call_counts["teacher"] += 1

  student_hook = actor.student_temporal_encoder.register_forward_hook(count_student)
  teacher_hook = actor.teacher_state_encoder.register_forward_hook(count_teacher)
  actor(observations, stochastic_output=True)
  z_s = actor.consume_latest_student_intent()
  z_t = actor.teacher_intent(observations)
  value = critic.forward_with_teacher_intent(observations, z_t)
  student_hook.remove()
  teacher_hook.remove()
  loss = value.square().mean() + (z_s - z_t.detach()).square().mean()
  loss.backward()

  assert z_s.shape == z_t.shape == (2, 128)
  assert call_counts == {"student": 1, "teacher": 1}
  assert critic.mlp[0].weight.grad is not None
  assert actor.student_temporal_encoder.layers[0].self_attn.in_proj_weight.grad is not None
  assert actor.reference_cross_attention.in_proj_weight.grad is None
  try:
    actor.consume_latest_student_intent()
  except RuntimeError as error:
    assert "No cached z_S" in str(error)
  else:
    raise AssertionError("z_S cache must be single-use.")
