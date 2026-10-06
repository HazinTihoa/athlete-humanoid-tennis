from __future__ import annotations

import math
from dataclasses import asdict

import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from athlete.goal_cond_tracking.config.g1.env_cfgs import (
  unitree_g1_phase_acceleration_deadline_tppo_distillation_env_cfg,
)
from athlete.goal_cond_tracking.config.g1.rl_cfg import (
  unitree_g1_tracking_ppo_runner_cfg,
  unitree_g1_tracking_tppo_distillation_runner_cfg,
  unitree_g1_tracking_tppo_distilllinear_floor03_runner_cfg,
  unitree_g1_tracking_tppo_distilllinear_floor05_runner_cfg,
  unitree_g1_tracking_tppo_distilllinear_runner_cfg,
  unitree_g1_tracking_tppo_ppo_distill03_runner_cfg,
  unitree_g1_tracking_tppo_ppo_only_runner_cfg,
  unitree_g1_tracking_tppo_pure_distillation_runner_cfg,
)
from athlete.goal_cond_tracking.rl.runner import MotionTrackingOnPolicyRunner
from athlete.goal_cond_tracking.rl.tppo import TPPO
from athlete.goal_cond_tracking.rl.tppo_storage import (
  TppoRolloutStorage,
)
from tensordict import TensorDict


def _observations(num_envs: int) -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(num_envs, 3),
      "critic": torch.randn(num_envs, 5),
      "student": torch.randn(num_envs, 3),
    },
    batch_size=[num_envs],
  )


def test_tppo_storage_keeps_teacher_labels_and_masks_aligned() -> None:
  num_envs = 3
  storage = TppoRolloutStorage(
    num_envs=num_envs,
    num_transitions_per_env=2,
    obs=_observations(num_envs),
    actions_shape=[2],
  )

  for step in range(2):
    transition = TppoRolloutStorage.Transition()
    transition.observations = _observations(num_envs)
    transition.actions = torch.full((num_envs, 2), float(step))
    transition.rewards = torch.zeros(num_envs)
    transition.dones = torch.zeros(num_envs, dtype=torch.bool)
    transition.values = torch.zeros(num_envs, 1)
    transition.actions_log_prob = torch.zeros(num_envs)
    transition.distribution_params = (
      torch.zeros(num_envs, 2),
      torch.ones(num_envs, 2),
    )
    transition.privileged_actions = transition.actions + 10.0
    transition.student_control_mask = torch.full(
      (num_envs, 1), step == 1, dtype=torch.bool
    )
    storage.add_transition(transition)

  batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
  torch.testing.assert_close(batch.privileged_actions, batch.actions + 10.0)
  torch.testing.assert_close(
    batch.student_control_mask.squeeze(-1),
    batch.actions[:, 0].bool(),
  )


class _FakeEnv:
  num_envs = 4
  num_actions = 3
  step_dt = 0.02

  @property
  def unwrapped(self) -> _FakeEnv:
    return self


class _FakeQuotaAlgorithm:
  def __init__(self) -> None:
    self.reset_mask: torch.Tensor | None = None

  def prepare_rollout(self, num_envs: int) -> torch.Tensor:
    assert num_envs == _FakeEnv.num_envs
    return torch.tensor([False, True, False, True])

  def reset_controller_states(self, reset_mask: torch.Tensor) -> None:
    self.reset_mask = reset_mask.clone()


class _FakeManagerEnv:
  def __init__(self) -> None:
    self.reset_env_ids: torch.Tensor | None = None

  def reset(self, *, env_ids: torch.Tensor) -> None:
    assert torch.is_inference_mode_enabled()
    self.reset_env_ids = env_ids.clone()


class _FakeVecEnv:
  num_envs = _FakeEnv.num_envs
  device = torch.device("cpu")

  def __init__(self) -> None:
    self.unwrapped = _FakeManagerEnv()
    self.observations = _observations(self.num_envs)

  def get_observations(self) -> TensorDict:
    return self.observations


class _FakeLogger:
  def __init__(self) -> None:
    self.cur_reward_sum = torch.ones(_FakeEnv.num_envs, 1)
    self.cur_episode_length = torch.ones(_FakeEnv.num_envs, 1)


def _small_tppo() -> TPPO:
  runner_cfg = unitree_g1_tracking_tppo_distillation_runner_cfg()
  runner_cfg.num_steps_per_env = 2
  runner_cfg.actor.hidden_dims = (8,)
  runner_cfg.critic.hidden_dims = (8,)
  runner_cfg.teacher.hidden_dims = (8,)
  runner_cfg.algorithm.num_learning_epochs = 1
  runner_cfg.algorithm.num_mini_batches = 1
  runner_cfg.algorithm.schedule = "fixed"
  runner_cfg.algorithm.teacher_act_prob = 1.0
  runner_cfg.algorithm.student_action_dim = 2

  cfg = asdict(runner_cfg)
  cfg["multi_gpu"] = None
  cfg["torch_compile_mode"] = None
  algorithm = TPPO.construct_algorithm(
    _observations(_FakeEnv.num_envs),
    _FakeEnv(),
    cfg,
    "cpu",
  )
  algorithm.load_teacher_state_dict(algorithm.teacher.state_dict())
  return algorithm


def test_tppo_rollout_update_freezes_teacher_and_exports_student() -> None:
  torch.manual_seed(7)
  algorithm = _small_tppo()
  teacher_before = {
    name: parameter.detach().clone()
    for name, parameter in algorithm.teacher.state_dict().items()
  }
  algorithm.train_mode()
  assert not algorithm.teacher.training
  assert algorithm.get_policy() is algorithm._raw_actor

  observations = _observations(_FakeEnv.num_envs)
  for _ in range(2):
    expected_teacher_actions = algorithm.get_teacher_actions(observations)
    actions = algorithm.act(observations)
    torch.testing.assert_close(actions, expected_teacher_actions)
    torch.testing.assert_close(
      algorithm.transition.privileged_actions,
      expected_teacher_actions[:, :2],
    )
    assert algorithm.transition.actions.shape == (_FakeEnv.num_envs, 2)
    assert not algorithm.transition.student_control_mask.any()
    observations = _observations(_FakeEnv.num_envs)
    algorithm.process_env_step(
      observations,
      rewards=torch.ones(_FakeEnv.num_envs),
      dones=torch.zeros(_FakeEnv.num_envs, dtype=torch.bool),
      extras={},
    )

  algorithm.compute_returns(observations)
  losses = algorithm.update()

  assert losses["distillation"] >= 0.0
  assert losses["teacher_control_fraction"] == 1.0
  assert losses["surrogate"] == 0.0
  for name, parameter in algorithm.teacher.state_dict().items():
    torch.testing.assert_close(parameter, teacher_before[name])
  assert "teacher_state_dict" in algorithm.save()

  environment_policy = algorithm.get_environment_policy()
  assert environment_policy.obs_groups == ("student", "actor")
  observations = _observations(_FakeEnv.num_envs)
  environment_actions = environment_policy(observations)
  torch.testing.assert_close(
    environment_actions[:, :2], algorithm.get_policy()(observations)
  )
  torch.testing.assert_close(
    environment_actions[:, 2:], algorithm.get_teacher_actions(observations)[:, 2:]
  )

  zero_phase_policy = algorithm.get_environment_policy(phase_source="zero")
  assert zero_phase_policy.obs_groups == ("student",)
  zero_phase_actions = zero_phase_policy(observations)
  torch.testing.assert_close(
    zero_phase_actions[:, :2], algorithm.get_policy()(observations)
  )
  torch.testing.assert_close(
    zero_phase_actions[:, 2:], torch.zeros(_FakeEnv.num_envs, 1)
  )

  training_debug_policy = algorithm.get_training_debug_policy()
  training_debug_actions = training_debug_policy(observations)
  torch.testing.assert_close(
    training_debug_actions, algorithm.get_teacher_actions(observations)
  )

  algorithm.teacher_act_prob = 0.0
  student_debug_policy = algorithm.get_training_debug_policy()
  student_debug_actions = student_debug_policy(observations)
  torch.testing.assert_close(
    student_debug_actions[:, :2], algorithm.get_policy()(observations)
  )
  torch.testing.assert_close(
    student_debug_actions[:, 2:], algorithm.get_teacher_actions(observations)[:, 2:]
  )


def test_tppo_fixed_quota_survives_episode_resets() -> None:
  torch.manual_seed(11)
  algorithm = _small_tppo()
  algorithm.teacher_act_prob = 0.5

  initial_switches = algorithm.prepare_rollout(_FakeEnv.num_envs)
  assert not initial_switches.any()
  assert algorithm.use_teacher_act_mask is not None
  assert int(algorithm.use_teacher_act_mask.sum().item()) == 2
  initial_roles = algorithm.use_teacher_act_mask.clone()

  observations = _observations(_FakeEnv.num_envs)
  algorithm.act(observations)
  algorithm.process_env_step(
    observations,
    rewards=torch.ones(_FakeEnv.num_envs),
    dones=torch.ones(_FakeEnv.num_envs, dtype=torch.bool),
    extras={},
  )
  torch.testing.assert_close(algorithm.use_teacher_act_mask, initial_roles)

  algorithm.teacher_act_prob = 0.25
  role_switches = algorithm.prepare_rollout(_FakeEnv.num_envs)
  assert int(role_switches.sum().item()) == 1
  assert int(algorithm.use_teacher_act_mask.sum().item()) == 1
  assert torch.all(initial_roles[role_switches])
  assert algorithm.last_role_switch_count == 1


def test_tppo_rollout_control_fraction_matches_fixed_quota() -> None:
  torch.manual_seed(13)
  algorithm = _small_tppo()
  algorithm.teacher_act_prob = 0.5
  algorithm.prepare_rollout(_FakeEnv.num_envs)

  observations = _observations(_FakeEnv.num_envs)
  for _ in range(2):
    algorithm.act(observations)
    observations = _observations(_FakeEnv.num_envs)
    algorithm.process_env_step(
      observations,
      rewards=torch.ones(_FakeEnv.num_envs),
      dones=torch.zeros(_FakeEnv.num_envs, dtype=torch.bool),
      extras={},
    )

  algorithm.compute_returns(observations)
  losses = algorithm.update()
  assert losses["teacher_action_probability"] == 0.5
  assert losses["teacher_quota_fraction"] == 0.5
  assert losses["teacher_control_fraction"] == 0.5
  assert losses["controller_role_switches"] == 0.0


def test_pure_distillation_executes_deterministic_student_mean() -> None:
  torch.manual_seed(17)
  algorithm = _small_tppo()
  algorithm.using_ppo = False
  algorithm.teacher_act_prob = 0.0
  observations = _observations(_FakeEnv.num_envs)

  environment_actions = algorithm.act(observations)
  student_mean = algorithm.actor.output_mean.detach()

  torch.testing.assert_close(environment_actions[:, :2], student_mean)
  torch.testing.assert_close(algorithm.transition.actions, student_mean)
  assert not algorithm.use_teacher_act_mask.any()


def test_tppo_runner_resets_role_switch_envs_and_refreshes_obs() -> None:
  runner = object.__new__(MotionTrackingOnPolicyRunner)
  runner.alg = _FakeQuotaAlgorithm()
  runner.env = _FakeVecEnv()
  runner.logger = _FakeLogger()
  runner.device = "cpu"

  old_observations = _observations(_FakeEnv.num_envs)
  observations = runner.prepare_rollout(old_observations)
  assert observations is not old_observations
  torch.testing.assert_close(
    observations["student"], runner.env.observations["student"]
  )
  assert runner.env.unwrapped.reset_env_ids is not None
  torch.testing.assert_close(runner.env.unwrapped.reset_env_ids, torch.tensor([1, 3]))
  assert runner.alg.reset_mask is not None
  torch.testing.assert_close(
    runner.alg.reset_mask, torch.tensor([False, True, False, True])
  )
  torch.testing.assert_close(
    runner.logger.cur_reward_sum.squeeze(-1), torch.tensor([1.0, 0.0, 1.0, 0.0])
  )
  torch.testing.assert_close(
    runner.logger.cur_episode_length.squeeze(-1),
    torch.tensor([1.0, 0.0, 1.0, 0.0]),
  )


def test_tppo_config_does_not_replace_existing_ppo_backend() -> None:
  ppo_cfg = unitree_g1_tracking_ppo_runner_cfg()
  tppo_cfg = unitree_g1_tracking_tppo_distillation_runner_cfg()

  assert ppo_cfg.algorithm.class_name == "PPO"
  assert tppo_cfg.algorithm.class_name.endswith(".TPPO")
  assert tppo_cfg.obs_groups["actor"] == ("student",)
  assert tppo_cfg.obs_groups["teacher"] == ("actor",)
  assert tppo_cfg.algorithm.student_action_dim == 29
  assert tppo_cfg.algorithm.teacher_act_prob == "linear"
  assert tppo_cfg.algorithm.update_times_scale == 1000
  assert tppo_cfg.algorithm.distillation_loss_coef == 1.0
  assert tppo_cfg.algorithm.distillation_update_times_scale is None

  distilllinear_cfg = unitree_g1_tracking_tppo_distilllinear_runner_cfg()
  assert distilllinear_cfg.algorithm.distillation_loss_coef == "linear"
  assert distilllinear_cfg.algorithm.distillation_update_times_scale == 3000
  assert distilllinear_cfg.algorithm.distillation_min_coef == 0.0

  floor03_cfg = unitree_g1_tracking_tppo_distilllinear_floor03_runner_cfg()
  assert floor03_cfg.algorithm.distillation_loss_coef == "linear"
  assert floor03_cfg.algorithm.distillation_update_times_scale == 3000
  assert floor03_cfg.algorithm.distillation_min_coef == 0.3

  floor05_cfg = unitree_g1_tracking_tppo_distilllinear_floor05_runner_cfg()
  assert floor05_cfg.algorithm.distillation_loss_coef == "linear"
  assert floor05_cfg.algorithm.distillation_update_times_scale == 3000
  assert floor05_cfg.algorithm.distillation_min_coef == 0.5

  pure_cfg = unitree_g1_tracking_tppo_pure_distillation_runner_cfg()
  assert pure_cfg.algorithm.using_ppo is False
  assert pure_cfg.algorithm.teacher_act_prob == "linear"
  assert pure_cfg.algorithm.update_times_scale == 3000
  assert pure_cfg.algorithm.distillation_loss_coef == 1.0
  assert pure_cfg.algorithm.distillation_update_times_scale is None
  assert pure_cfg.algorithm.distillation_min_coef == 1.0

  ppo_only_cfg = unitree_g1_tracking_tppo_ppo_only_runner_cfg()
  assert ppo_only_cfg.algorithm.using_ppo is True
  assert ppo_only_cfg.algorithm.teacher_act_prob == 0.0
  assert ppo_only_cfg.algorithm.distillation_loss_coef == 0.0
  assert ppo_only_cfg.algorithm.distillation_update_times_scale is None
  assert ppo_only_cfg.algorithm.distillation_min_coef == 0.0

  ppo_distill03_cfg = unitree_g1_tracking_tppo_ppo_distill03_runner_cfg()
  assert ppo_distill03_cfg.algorithm.using_ppo is True
  assert ppo_distill03_cfg.algorithm.teacher_act_prob == 0.0
  assert ppo_distill03_cfg.algorithm.distillation_loss_coef == 0.3
  assert ppo_distill03_cfg.algorithm.distillation_update_times_scale is None
  assert ppo_distill03_cfg.algorithm.distillation_min_coef == 0.3


def test_tppo_distillation_schedule_has_an_independent_time_scale() -> None:
  algorithm = _small_tppo()
  algorithm.teacher_act_prob = "linear"
  algorithm.update_times_scale = 1000
  algorithm.distillation_loss_coef = "linear"
  algorithm.distillation_update_times_scale = 1500

  algorithm.update_count = 0
  assert algorithm.teacher_action_probability == 1.0
  assert algorithm.current_distillation_loss_coef == 1.0

  algorithm.update_count = 750
  assert algorithm.teacher_action_probability == 0.25
  assert algorithm.current_distillation_loss_coef == 0.5

  algorithm.update_count = 1500
  assert algorithm.teacher_action_probability == 0.0
  assert algorithm.current_distillation_loss_coef == 0.0


def test_no_tracking_racket0_floor03_task_registers_floor03_runner() -> None:
  cfg = load_rl_cfg(
    "Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-"
    "NoTracking-Racket0-DistillFloor03-Unitree-G1"
  )

  assert cfg.algorithm.distillation_loss_coef == "linear"
  assert cfg.algorithm.distillation_update_times_scale == 3000
  assert cfg.algorithm.distillation_min_coef == 0.3


def test_no_tracking_racket0_floor05_task_registers_floor05_runner() -> None:
  cfg = load_rl_cfg(
    "Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-"
    "NoTracking-Racket0-DistillFloor05-Unitree-G1"
  )

  assert cfg.algorithm.distillation_loss_coef == "linear"
  assert cfg.algorithm.distillation_update_times_scale == 3000
  assert cfg.algorithm.distillation_min_coef == 0.5


def test_racket_orientation_pure_distill_task_registers_pure_runner() -> None:
  cfg = load_rl_cfg(
    "Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-"
    "NoTracking-Racket0-RacketOrientation-PureDistill-Unitree-G1"
  )

  assert cfg.algorithm.using_ppo is False
  assert cfg.algorithm.teacher_act_prob == "linear"
  assert cfg.algorithm.update_times_scale == 3000
  assert cfg.algorithm.distillation_loss_coef == 1.0
  assert cfg.algorithm.distillation_min_coef == 1.0


def test_racket_orientation_ppo_only_task_registers_ppo_only_runner() -> None:
  cfg = load_rl_cfg(
    "Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-"
    "NoTracking-Racket0-RacketOrientation-PPOOnly-Unitree-G1"
  )

  assert cfg.algorithm.using_ppo is True
  assert cfg.algorithm.teacher_act_prob == 0.0
  assert cfg.algorithm.distillation_loss_coef == 0.0
  assert cfg.algorithm.distillation_min_coef == 0.0


def test_racket_orientation_ppo_distill03_task_registers_fixed_runner() -> None:
  cfg = load_rl_cfg(
    "Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-"
    "NoTracking-Racket0-RacketOrientation-PPOPlusDistill03-Unitree-G1"
  )

  assert cfg.algorithm.using_ppo is True
  assert cfg.algorithm.teacher_act_prob == 0.0
  assert cfg.algorithm.distillation_loss_coef == 0.3
  assert cfg.algorithm.distillation_min_coef == 0.3


def test_root_tilt_ppo_only_task_uses_no_reference_terminations() -> None:
  cfg = load_env_cfg(
    "Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-"
    "NoTracking-Racket0-RootTilt-PPOOnly-Unitree-G1"
  )

  assert cfg.terminations is not None
  assert set(cfg.terminations) == {"time_out", "root_tilt"}
  assert cfg.terminations["root_tilt"].params == {
    "command_name": "motion",
    "max_tilt_degrees": 45.0,
  }


def test_root_tilt_ppo_distill03_task_uses_fixed_distillation() -> None:
  cfg = load_rl_cfg(
    "Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-"
    "NoTracking-Racket0-RootTilt-PPOPlusDistill03-Unitree-G1"
  )

  assert cfg.algorithm.using_ppo is True
  assert cfg.algorithm.teacher_act_prob == 0.0
  assert cfg.algorithm.distillation_loss_coef == 0.3
  assert cfg.algorithm.distillation_update_times_scale is None
  assert cfg.algorithm.distillation_min_coef == 0.3


def test_intent_reference_latent_ppo_distill005_task_uses_fixed_distillation() -> None:
  cfg = load_rl_cfg(
    "Mjlab-Tennis-Launch-Distill-WarpRootDirected-Cache0-200-"
    "NoTracking-Racket0-RootTilt-IntentRefLatent-"
    "PPOPlusDistill005-Unitree-G1"
  )

  assert cfg.algorithm.using_ppo is True
  assert cfg.algorithm.teacher_act_prob == 0.0
  assert cfg.algorithm.distillation_loss_coef == 0.05
  assert cfg.algorithm.distillation_update_times_scale is None
  assert cfg.algorithm.distillation_min_coef == 0.05


def test_small_court_intent_task_uses_fixed_distillation() -> None:
  cfg = load_rl_cfg(
    "Mjlab-Tennis-SmallCourt-IntentRefLatent-"
    "PPOPlusDistill005-SweetSpeed50-RacketGround10-Unitree-G1"
  )

  assert cfg.algorithm.using_ppo is True
  assert cfg.algorithm.teacher_act_prob == 0.0
  assert cfg.algorithm.distillation_loss_coef == 0.05
  assert cfg.algorithm.distillation_min_coef == 0.05


def test_small_court_frozen_intent_task_disables_intent_updates() -> None:
  cfg = load_rl_cfg(
    "Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-"
    "FrozenIntent-SweetSpeed50-RacketGround10-Unitree-G1"
  )

  assert cfg.algorithm.using_ppo is True
  assert cfg.algorithm.distillation_loss_coef == 0.05
  assert cfg.algorithm.freeze_intent_encoders is True
  assert cfg.algorithm.intent_reference_loss_coef == 0.0
  assert cfg.algorithm.intent_teacher_action_loss_coef == 0.0
  assert cfg.algorithm.intent_latent_loss_coef == 0.0


def test_small_court_frozen_intent_epoch3_ablation_parameters() -> None:
  task_id = (
    "Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-"
    "FrozenIntent-Epoch3-ActionRate020-SweetSpeed50-Scale03-"
    "RacketGround10-Unitree-G1"
  )
  rl_cfg = load_rl_cfg(task_id)
  env_cfg = load_env_cfg(task_id)

  assert rl_cfg.algorithm.num_learning_epochs == 3
  assert rl_cfg.algorithm.freeze_intent_encoders is True
  assert rl_cfg.algorithm.distillation_loss_coef == 0.05
  assert env_cfg.rewards["action_rate_l2"].weight == -0.2
  assert (
    env_cfg.rewards["reference_sweet_spot_velocity_reward"].params["target_speed_scale"]
    == 0.3
  )


def test_tppo_distillation_schedule_can_decay_to_nonzero_floor() -> None:
  algorithm = _small_tppo()
  algorithm.distillation_loss_coef = "linear"
  algorithm.distillation_update_times_scale = 3000
  algorithm.distillation_min_coef = 0.2

  algorithm.update_count = 0
  assert algorithm.current_distillation_loss_coef == 1.0
  algorithm.update_count = 1500
  assert math.isclose(algorithm.current_distillation_loss_coef, 0.6)
  algorithm.update_count = 3000
  assert algorithm.current_distillation_loss_coef == 0.2
  algorithm.update_count = 10_000
  assert algorithm.current_distillation_loss_coef == 0.2


def test_tppo_distillation_config_separates_student_and_teacher_obs() -> None:
  env_cfg = unitree_g1_phase_acceleration_deadline_tppo_distillation_env_cfg()
  runner_cfg = unitree_g1_tracking_tppo_distillation_runner_cfg()

  assert env_cfg.observations["student"] is not env_cfg.observations["actor"]
  assert (
    env_cfg.observations["student"].terms is not env_cfg.observations["actor"].terms
  )
  assert tuple(env_cfg.observations["student"].terms) == (
    "task_goal",
    "time_remaining",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  assert "command" in env_cfg.observations["actor"].terms
  assert "phase_state" in env_cfg.observations["actor"].terms
  assert set(env_cfg.rewards) == {
    "action_rate_l2",
    "joint_limit",
    "self_collisions",
    "target_position_reward",
    "target_orientation_reward",
    "target_velocity_reward",
  }
  assert runner_cfg.obs_groups == {
    "actor": ("student",),
    "critic": ("critic",),
    "teacher": ("actor",),
  }
