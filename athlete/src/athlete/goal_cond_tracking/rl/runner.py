import os
import time
from pathlib import Path
from typing import cast

import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.utils import check_nan
from torch import nn

import wandb
from athlete.goal_cond_tracking.mdp import MultiTargetMotionCommand


class _OnnxMotionModel(nn.Module):
  """ONNX-exportable model that wraps the policy and bundles motion reference data."""

  def __init__(self, actor, motion):
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)
    self.register_buffer("joint_pos", motion.joint_pos.to("cpu"))
    self.register_buffer("joint_vel", motion.joint_vel.to("cpu"))
    self.register_buffer("body_pos_w", motion.body_pos_w.to("cpu"))
    self.register_buffer("body_quat_w", motion.body_quat_w.to("cpu"))
    self.register_buffer("body_lin_vel_w", motion.body_lin_vel_w.to("cpu"))
    self.register_buffer("body_ang_vel_w", motion.body_ang_vel_w.to("cpu"))
    self.time_step_total: int = self.joint_pos.shape[0]  # type: ignore[index]

  def forward(self, x, time_step):
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    return (
      self.policy(x),
      self.joint_pos[time_step_clamped],  # type: ignore[index]
      self.joint_vel[time_step_clamped],  # type: ignore[index]
      self.body_pos_w[time_step_clamped],  # type: ignore[index]
      self.body_quat_w[time_step_clamped],  # type: ignore[index]
      self.body_lin_vel_w[time_step_clamped],  # type: ignore[index]
      self.body_ang_vel_w[time_step_clamped],  # type: ignore[index]
    )


class _OnnxMultiTargetMotionModel(nn.Module):
  """ONNX-exportable model for multi-target motion tracking.

  Stores stacked motion data for all motions. Takes ``which_motion`` and
  ``time_step`` to index into the correct motion's reference data.
  """

  def __init__(self, actor, cmd: MultiTargetMotionCommand):
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)
    # Stacked motion tensors: (num_motions, max_timesteps, ...)
    self.register_buffer("joint_pos", cmd._stacked_joint_pos.to("cpu"))
    self.register_buffer("joint_vel", cmd._stacked_joint_vel.to("cpu"))
    self.register_buffer("body_pos_w", cmd._stacked_body_pos_w.to("cpu"))
    self.register_buffer("body_quat_w", cmd._stacked_body_quat_w.to("cpu"))
    self.register_buffer("body_lin_vel_w", cmd._stacked_body_lin_vel_w.to("cpu"))
    self.register_buffer("body_ang_vel_w", cmd._stacked_body_ang_vel_w.to("cpu"))
    self.register_buffer("time_step_totals", cmd._time_step_totals.to("cpu").long())
    self.num_motions: int = len(cmd.motion_loaders)

  def forward(self, x, which_motion, time_step):
    which_motion_clamped = torch.clamp(
      which_motion.long().squeeze(-1), max=self.num_motions - 1
    )
    per_motion_max = self.time_step_totals[which_motion_clamped]  # type: ignore[index]
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=per_motion_max - 1
    )
    return (
      self.policy(x),
      self.joint_pos[which_motion_clamped, time_step_clamped],  # type: ignore[index]
      self.joint_vel[which_motion_clamped, time_step_clamped],  # type: ignore[index]
      self.body_pos_w[which_motion_clamped, time_step_clamped],  # type: ignore[index]
      self.body_quat_w[which_motion_clamped, time_step_clamped],  # type: ignore[index]
      self.body_lin_vel_w[which_motion_clamped, time_step_clamped],  # type: ignore[index]
      self.body_ang_vel_w[which_motion_clamped, time_step_clamped],  # type: ignore[index]
    )


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  def prepare_rollout(self, obs):
    """Apply an algorithm controller quota and reset role-switch environments."""
    prepare_fn = getattr(self.alg, "prepare_rollout", None)
    if prepare_fn is None:
      return obs

    role_switch_mask = prepare_fn(self.env.num_envs)
    if not isinstance(role_switch_mask, torch.Tensor):
      raise TypeError("prepare_rollout() must return a torch.Tensor mask.")
    if role_switch_mask.shape != (self.env.num_envs,):
      raise ValueError(
        "Controller role-switch mask must have shape "
        f"({self.env.num_envs},), got {tuple(role_switch_mask.shape)}."
      )
    role_switch_mask = role_switch_mask.to(dtype=torch.bool)
    if not torch.any(role_switch_mask):
      return obs

    env_ids = role_switch_mask.nonzero(as_tuple=False).squeeze(-1)
    with torch.inference_mode():
      self.env.unwrapped.reset(env_ids=env_ids.to(self.env.device))
      reset_states_fn = getattr(self.alg, "reset_controller_states", None)
      if reset_states_fn is not None:
        reset_states_fn(role_switch_mask)
      logger = getattr(self, "logger", None)
      if logger is not None:
        logger_env_ids = env_ids.to(self.device)
        for buffer_name in (
          "cur_reward_sum",
          "cur_episode_length",
          "cur_ereward_sum",
          "cur_ireward_sum",
        ):
          buffer = getattr(logger, buffer_name, None)
          if buffer is not None:
            buffer[logger_env_ids] = 0
      return self.env.get_observations().to(self.device)

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    """Run TPPO with rollout-boundary quota preparation; preserve PPO upstream."""
    if not hasattr(self.alg, "prepare_rollout"):
      super().learn(num_learning_iterations, init_at_random_ep_len)
      return

    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      obs = self.prepare_rollout(obs)
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic_rewards = (
            self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
          )
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

        stop = time.time()
        collect_time = stop - start
        start = stop
        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()
      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
      )
      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

    if self.logger.writer is not None:
      self.save(
        os.path.join(
          self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"
        )
      )
      self.logger.stop_logging_writer()

  def _export_multi_target_onnx(
    self, path: str, filename: str, verbose: bool = False
  ) -> None:
    cmd = cast(
      MultiTargetMotionCommand,
      self.env.unwrapped.command_manager.get_term("motion"),
    )
    model = _OnnxMultiTargetMotionModel(self.alg.get_policy(), cmd)
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    which_motion = torch.zeros(1, 1)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
      model,
      (obs, which_motion, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs", "which_motion", "time_step"],
      output_names=[
        "actions",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ],
      dynamic_axes={},
      dynamo=False,
    )

  def _is_multi_target(self) -> bool:
    cmd = self.env.unwrapped.command_manager.get_term("motion")
    return isinstance(cmd, MultiTargetMotionCommand)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    os.makedirs(path, exist_ok=True)
    self._export_multi_target_onnx(path, filename, verbose)

  def save(self, path: str, infos=None):
    super().save(path, infos)
    checkpoint_path = Path(path)
    policy_path = str(checkpoint_path.parent) + os.sep
    filename = checkpoint_path.parent.name + ".onnx"
    try:
      self.export_policy_to_onnx(policy_path, filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      observations = self.env.unwrapped.cfg.observations
      if 'intent_ball_history' in observations:
        term = observations['intent_ball_history'].terms['position']
        position_encoding = term.params.get('position_encoding')
        if position_encoding is not None:
          student_group = observations.get('student')
          student_term = (
            None if student_group is None
            else student_group.terms.get('ball_position')
          )
          if student_term is None or student_term.params.get('position_encoding') != position_encoding:
            raise ValueError(
              "Student and intent ball position encodings must match for ONNX export."
            )
          metadata.update({
            'ball_position_encoding': 'radial_tanh_v1',
            'ball_position_linear_radius_m': str(position_encoding['linear_radius_m']),
            'ball_position_limit_radius_m': str(position_encoding['limit_radius_m']),
          })
        if term.params.get('measurement_perception') is not None:
          import json
          metadata['measurement_perception'] = json.dumps(term.params['measurement_perception'])
          metadata['intent_history_lags'] = list(term.params['sample_lags'])
          metadata['ball_estimator'] = 'natnet_quadratic7_min5_fallback2_blend055_v1'
          metadata['measurement_rate_hz'] = '50'
      if self._is_multi_target():
        cmd = cast(
          MultiTargetMotionCommand,
          self.env.unwrapped.command_manager.get_term("motion"),
        )
        metadata.update(
          {
            "command_type": "multi_target",
            "anchor_body_name": cmd.cfg.anchor_body_name,
            "body_names": list(cmd.cfg.body_names),
            "num_motions": len(cmd.motion_loaders),
            "source_link_names": [
              [st.source_link for st in mc.sub_targets] for mc in cmd.motion_configs
            ],
            "source_link_types": [
              [st.source_type for st in mc.sub_targets] for mc in cmd.motion_configs
            ],
            "target_phase_starts": [
              [st.target_phase_start for st in mc.sub_targets]
              for mc in cmd.motion_configs
            ],
            "target_phase_ends": [
              [st.target_phase_end for st in mc.sub_targets]
              for mc in cmd.motion_configs
            ],
            "time_step_totals": cmd._time_step_totals.cpu().tolist(),
          }
        )
      else:
        motion_term = cast(
          MultiTargetMotionCommand,
          self.env.unwrapped.command_manager.get_term("motion"),
        )
        metadata.update(
          {
            "anchor_body_name": motion_term.cfg.anchor_body_name,
            "body_names": list(motion_term.cfg.body_names),
          }
        )
      attach_metadata_to_onnx(os.path.join(policy_path, filename), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
        if self.registry_name is not None:
          for rn in self.registry_name.split(","):
            rn = rn.strip()
            if not rn:
              continue
            # Build registry artifact reference for use_artifact.
            # Input may be "org/wandb-registry-collection/name[:alias]".
            # use_artifact needs "wandb-registry-collection/name:alias".
            parts = rn.split("/")
            if len(parts) >= 3:
              # Full registry path — take last two components.
              name_part = parts[-2] + "/" + parts[-1]
            else:
              name_part = parts[-1]
            if ":" not in name_part.split("/")[-1]:
              name_part = name_part + ":latest"
            try:
              wandb.run.use_artifact(name_part)  # type: ignore
            except Exception as e:
              print(f"[WARN] Could not link artifact '{name_part}' to run: {e}")
          self.registry_name = None
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
