"""RL configuration for Unitree G1 tracking task."""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class RslRlTppoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Configuration for the Instinct-RL-derived TPPO backend."""

  class_name: str = "athlete.goal_cond_tracking.rl.tppo.TPPO"
  teacher_checkpoint: str | None = None
  teacher_act_prob: float | Literal["linear", "exp", "tanh"] = "linear"
  update_times_scale: int = 1000
  using_ppo: bool = True
  distillation_loss_coef: float | Literal["linear", "exp", "tanh"] = 1.0
  distillation_update_times_scale: int | None = None
  distillation_min_coef: float = 0.0
  distill_target: Literal[
    "real", "mse_sum", "l1", "tanh", "scaled_tanh", "max_log_prob", "kl"
  ] = "real"
  action_labels_from_sample: bool = False
  strict_teacher_checkpoint: bool = True
  student_action_dim: int = 29


@dataclass
class RslRlIntentTppoAlgorithmCfg(RslRlTppoAlgorithmCfg):
  """TPPO configuration for privileged/reference temporal intent training."""

  class_name: str = "athlete.goal_cond_tracking.rl.intent_tppo.IntentTPPO"
  intent_reference_loss_coef: float = 0.0
  intent_teacher_action_loss_coef: float = 1.0
  intent_latent_loss_coef: float = 0.05
  freeze_intent_encoders: bool = False


@dataclass
class RslRlIntentTransformerModelCfg(RslRlModelCfg):
  """Actor configuration for the Student-only temporal intent encoder."""

  class_name: str = (
    "athlete.goal_cond_tracking.rl.intent_transformer.IntentTransformerActor"
  )
  state_history_group: str = "intent_state_history"
  ball_history_group: str = "intent_ball_history"
  reference_future_group: str = "intent_reference_future"
  state_history_steps: int = 5
  state_token_dim: int = 93
  ball_history_steps: int = 5
  ball_token_dim: int = 6
  reference_steps: int = 12
  reference_token_dim: int = 58
  intent_latent_dim: int = 128
  transformer_heads: int = 4
  transformer_layers: int = 2
  transformer_ff_dim: int = 256


@dataclass
class RslRlIntentTransformerCriticModelCfg(RslRlModelCfg):
  """Critic configuration receiving the Actor-owned privileged z_T."""

  class_name: str = "athlete.goal_cond_tracking.rl.intent_transformer.IntentTransformerCritic"
  intent_latent_dim: int = 128


@dataclass
class RslRlTppoRunnerCfg(RslRlOnPolicyRunnerCfg):
  """RSL-RL runner configuration containing a separate frozen Teacher."""

  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "actor": ("student",),
      "critic": ("critic",),
      "teacher": ("actor",),
    }
  )
  teacher: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  algorithm: RslRlTppoAlgorithmCfg = field(default_factory=RslRlTppoAlgorithmCfg)


def unitree_g1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 tracking task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_tracking_tppo_distillation_runner_cfg() -> RslRlTppoRunnerCfg:
  """Create TPPO with independent Student, Teacher, and Critic observations."""
  return RslRlTppoRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    teacher=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    algorithm=RslRlTppoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_tracking_tppo",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_tracking_tppo_distilllinear_runner_cfg() -> RslRlTppoRunnerCfg:
  """Create TPPO with a 3000-update linear distillation schedule."""
  cfg = unitree_g1_tracking_tppo_distillation_runner_cfg()
  cfg.algorithm.distillation_loss_coef = "linear"
  cfg.algorithm.distillation_update_times_scale = 3000
  cfg.algorithm.distillation_min_coef = 0.0
  return cfg


def unitree_g1_tracking_tppo_distilllinear_floor03_runner_cfg() -> RslRlTppoRunnerCfg:
  """Create TPPO whose distillation coefficient decays from 1.0 to 0.3."""
  cfg = unitree_g1_tracking_tppo_distilllinear_runner_cfg()
  cfg.algorithm.distillation_min_coef = 0.3
  return cfg


def unitree_g1_tracking_tppo_distilllinear_floor05_runner_cfg() -> RslRlTppoRunnerCfg:
  """Create TPPO whose distillation coefficient decays from 1.0 to 0.5."""
  cfg = unitree_g1_tracking_tppo_distilllinear_runner_cfg()
  cfg.algorithm.distillation_min_coef = 0.5
  return cfg


def unitree_g1_tracking_tppo_pure_distillation_runner_cfg() -> RslRlTppoRunnerCfg:
  """Create deterministic DAgger distillation without PPO updates."""
  cfg = unitree_g1_tracking_tppo_distillation_runner_cfg()
  cfg.algorithm.using_ppo = False
  cfg.algorithm.teacher_act_prob = "linear"
  cfg.algorithm.update_times_scale = 3000
  cfg.algorithm.distillation_loss_coef = 1.0
  cfg.algorithm.distillation_update_times_scale = None
  cfg.algorithm.distillation_min_coef = 1.0
  return cfg


def _unitree_g1_tracking_tppo_intent_transformer_runner_cfg(
  *,
  intent_reference_loss_coef: float,
  intent_teacher_action_loss_coef: float,
  intent_latent_loss_coef: float,
) -> RslRlTppoRunnerCfg:
  """Build an isolated pure-distillation config for intent encoder ablations."""
  cfg = unitree_g1_tracking_tppo_pure_distillation_runner_cfg()
  cfg.actor = RslRlIntentTransformerModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  cfg.critic = RslRlIntentTransformerCriticModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
  )
  cfg.algorithm = RslRlIntentTppoAlgorithmCfg(
    value_loss_coef=cfg.algorithm.value_loss_coef,
    use_clipped_value_loss=cfg.algorithm.use_clipped_value_loss,
    clip_param=cfg.algorithm.clip_param,
    entropy_coef=cfg.algorithm.entropy_coef,
    num_learning_epochs=cfg.algorithm.num_learning_epochs,
    num_mini_batches=cfg.algorithm.num_mini_batches,
    learning_rate=cfg.algorithm.learning_rate,
    schedule=cfg.algorithm.schedule,
    gamma=cfg.algorithm.gamma,
    lam=cfg.algorithm.lam,
    desired_kl=cfg.algorithm.desired_kl,
    max_grad_norm=cfg.algorithm.max_grad_norm,
    teacher_checkpoint=cfg.algorithm.teacher_checkpoint,
    teacher_act_prob=cfg.algorithm.teacher_act_prob,
    update_times_scale=cfg.algorithm.update_times_scale,
    using_ppo=cfg.algorithm.using_ppo,
    distillation_loss_coef=cfg.algorithm.distillation_loss_coef,
    distillation_update_times_scale=cfg.algorithm.distillation_update_times_scale,
    distillation_min_coef=cfg.algorithm.distillation_min_coef,
    distill_target=cfg.algorithm.distill_target,
    action_labels_from_sample=cfg.algorithm.action_labels_from_sample,
    strict_teacher_checkpoint=cfg.algorithm.strict_teacher_checkpoint,
    student_action_dim=cfg.algorithm.student_action_dim,
    intent_reference_loss_coef=intent_reference_loss_coef,
    intent_teacher_action_loss_coef=intent_teacher_action_loss_coef,
    intent_latent_loss_coef=intent_latent_loss_coef,
  )
  return cfg


def unitree_g1_tracking_tppo_intent_transformer_runner_cfg() -> RslRlTppoRunnerCfg:
  """Preserve the original IntentTransformer pure-distillation experiment."""
  return _unitree_g1_tracking_tppo_intent_transformer_runner_cfg(
    intent_reference_loss_coef=0.0,
    intent_teacher_action_loss_coef=1.0,
    intent_latent_loss_coef=0.05,
  )


def unitree_g1_tracking_tppo_intent_reference_distill_runner_cfg() -> (
  RslRlTppoRunnerCfg
):
  """Train z_T reference reconstruction and Student action distillation only."""
  return _unitree_g1_tracking_tppo_intent_transformer_runner_cfg(
    intent_reference_loss_coef=1.0,
    intent_teacher_action_loss_coef=0.0,
    intent_latent_loss_coef=0.0,
  )


def unitree_g1_tracking_tppo_intent_reference_latent_distill_runner_cfg() -> (
  RslRlTppoRunnerCfg
):
  """Add z_S-to-z_T latent alignment to the reference-distillation baseline."""
  return _unitree_g1_tracking_tppo_intent_transformer_runner_cfg(
    intent_reference_loss_coef=1.0,
    intent_teacher_action_loss_coef=0.0,
    intent_latent_loss_coef=0.05,
  )


def unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill03_runner_cfg() -> (
  RslRlTppoRunnerCfg
):
  """Fine-tune the reference/latent Intent Student with PPO and fixed distill 0.3."""
  cfg = unitree_g1_tracking_tppo_intent_reference_latent_distill_runner_cfg()
  cfg.algorithm.using_ppo = True
  cfg.algorithm.teacher_act_prob = 0.0
  cfg.algorithm.distillation_loss_coef = 0.3
  cfg.algorithm.distillation_update_times_scale = None
  cfg.algorithm.distillation_min_coef = 0.3
  return cfg


def unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill01_runner_cfg() -> (
  RslRlTppoRunnerCfg
):
  """Fine-tune the reference/latent Intent Student with PPO and fixed distill 0.1."""
  cfg = unitree_g1_tracking_tppo_intent_reference_latent_distill_runner_cfg()
  cfg.algorithm.using_ppo = True
  cfg.algorithm.teacher_act_prob = 0.0
  cfg.algorithm.distillation_loss_coef = 0.1
  cfg.algorithm.distillation_update_times_scale = None
  cfg.algorithm.distillation_min_coef = 0.1
  return cfg


def unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_runner_cfg() -> (
  RslRlTppoRunnerCfg
):
  """Fine-tune the reference/latent Intent Student with PPO and fixed distill 0.05."""
  cfg = unitree_g1_tracking_tppo_intent_reference_latent_distill_runner_cfg()
  cfg.algorithm.using_ppo = True
  cfg.algorithm.teacher_act_prob = 0.0
  cfg.algorithm.distillation_loss_coef = 0.05
  cfg.algorithm.distillation_update_times_scale = None
  cfg.algorithm.distillation_min_coef = 0.05
  return cfg


def unitree_g1_tracking_tppo_intent_reference_latent_ball_observation_robust_runner_cfg() -> (
  RslRlTppoRunnerCfg
):
  """Use visibility and measurement age in each deployable ball-history token."""
  cfg = unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_runner_cfg()
  if not isinstance(cfg.actor, RslRlIntentTransformerModelCfg):
    raise TypeError("Robust ball observation requires IntentTransformerActor.")
  cfg.actor.ball_token_dim = 8
  return cfg


def unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_runner_cfg() -> (
  RslRlTppoRunnerCfg
):
  """Fine-tune only the policy/value heads with fixed z_S and z_T encoders."""
  cfg = unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_runner_cfg()
  cfg.algorithm.intent_reference_loss_coef = 0.0
  cfg.algorithm.intent_teacher_action_loss_coef = 0.0
  cfg.algorithm.intent_latent_loss_coef = 0.0
  cfg.algorithm.freeze_intent_encoders = True
  return cfg


def unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg() -> (
  RslRlTppoRunnerCfg
):
  """Use three PPO learning epochs while keeping both Intent encoders frozen."""
  cfg = unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_runner_cfg()
  cfg.algorithm.num_learning_epochs = 3
  return cfg


def unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill01_frozen_intent_epoch3_runner_cfg() -> (
  RslRlTppoRunnerCfg
):
  """Use fixed distill 0.1 and three PPO epochs with frozen Intent encoders."""
  cfg = unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill01_runner_cfg()
  cfg.algorithm.intent_reference_loss_coef = 0.0
  cfg.algorithm.intent_teacher_action_loss_coef = 0.0
  cfg.algorithm.intent_latent_loss_coef = 0.0
  cfg.algorithm.freeze_intent_encoders = True
  cfg.algorithm.num_learning_epochs = 3
  return cfg


def unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill00_runner_cfg() -> (
  RslRlTppoRunnerCfg
):
  """Fine-tune the reference/latent Intent Student with PPO and no action distill."""
  cfg = unitree_g1_tracking_tppo_intent_reference_latent_distill_runner_cfg()
  cfg.algorithm.using_ppo = True
  cfg.algorithm.teacher_act_prob = 0.0
  cfg.algorithm.distillation_loss_coef = 0.0
  cfg.algorithm.distillation_update_times_scale = None
  cfg.algorithm.distillation_min_coef = 0.0
  return cfg


def unitree_g1_tracking_tppo_ppo_only_runner_cfg() -> RslRlTppoRunnerCfg:
  """Create PPO-only Student fine-tuning from a distilled TPPO checkpoint."""
  cfg = unitree_g1_tracking_tppo_distillation_runner_cfg()
  cfg.algorithm.using_ppo = True
  cfg.algorithm.teacher_act_prob = 0.0
  cfg.algorithm.distillation_loss_coef = 0.0
  cfg.algorithm.distillation_update_times_scale = None
  cfg.algorithm.distillation_min_coef = 0.0
  return cfg


def unitree_g1_tracking_tppo_ppo_distill03_runner_cfg() -> RslRlTppoRunnerCfg:
  """Create PPO fine-tuning with a fixed 0.3 Teacher distillation loss."""
  cfg = unitree_g1_tracking_tppo_distillation_runner_cfg()
  cfg.algorithm.using_ppo = True
  cfg.algorithm.teacher_act_prob = 0.0
  cfg.algorithm.distillation_loss_coef = 0.3
  cfg.algorithm.distillation_update_times_scale = None
  cfg.algorithm.distillation_min_coef = 0.3
  return cfg
