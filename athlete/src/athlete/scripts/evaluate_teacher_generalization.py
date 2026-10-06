"""Evaluate a PPO Teacher on an equally sampled local motion set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from mjlab.tasks.registry import list_tasks
from athlete.goal_cond_tracking.mdp.commands import (
  MultiTargetMotionCommand,
  MultiTargetMotionCommandCfg,
)
from athlete.goal_cond_tracking.mdp.phase_commands import (
  PhaseAwareMultiTargetMotionCommand,
)
from athlete.motion_sets.motion_set import MotionSet
from athlete.scripts.play import PlayConfig, build_play_session


@dataclass(frozen=True)
class TeacherGeneralizationConfig:
  motion_config: Path
  checkpoint_file: Path
  episodes_per_motion: int = 2
  max_steps: int = 500
  device: str | None = None
  seed: int = 42
  target_success_threshold_m: float = 0.15
  output_file: Path | None = None


def _mean_over_valid(values: torch.Tensor, valid: torch.Tensor) -> float:
  selected = values[valid]
  return float(selected.mean().item()) if len(selected) > 0 else float("nan")


@torch.no_grad()
def run_evaluation(task_id: str, cfg: TeacherGeneralizationConfig) -> dict:
  if cfg.episodes_per_motion <= 0:
    raise ValueError("episodes_per_motion must be positive")
  if cfg.max_steps <= 0:
    raise ValueError("max_steps must be positive")

  motion_set = MotionSet.from_toml(cfg.motion_config)
  motion_count = len(motion_set.enabled_names)
  if motion_count == 0:
    raise ValueError("motion_config contains no enabled motions")
  num_envs = motion_count * cfg.episodes_per_motion

  def configure_clean_evaluation(env_cfg) -> None:
    motion_cfg = env_cfg.commands["motion"]
    if not isinstance(motion_cfg, MultiTargetMotionCommandCfg):
      raise TypeError("Selected task does not use MultiTargetMotionCommandCfg")
    motion_cfg.auto_chain_motion = False
    motion_cfg.target_pos_std_scale = 0.0
    motion_cfg.pose_range = {}
    motion_cfg.velocity_range = {}
    motion_cfg.joint_position_range = (0.0, 0.0)
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, 10.0)

  play_cfg = PlayConfig(
    motion_config=cfg.motion_config,
    checkpoint_file=str(cfg.checkpoint_file),
    num_envs=num_envs,
    seed=cfg.seed,
    device=cfg.device,
    phase_plot=False,
    fast_play=True,
  )
  env, policy = build_play_session(
    task_id,
    play_cfg,
    install_physical_ball_controller=False,
    env_cfg_hook=configure_clean_evaluation,
  )
  command = env.unwrapped.command_manager.get_term("motion")
  if not isinstance(command, MultiTargetMotionCommand):
    env.close()
    raise TypeError("Active motion command has an unexpected type")
  if not isinstance(command, PhaseAwareMultiTargetMotionCommand):
    env.close()
    raise TypeError("Teacher generalization evaluation requires a phase-aware task")

  initial_motion_ids = command.which_motion.clone()
  counts = torch.bincount(initial_motion_ids, minlength=motion_count)
  if not torch.all(counts == cfg.episodes_per_motion):
    env.close()
    raise RuntimeError(
      "Motion sampler did not allocate equal episodes: "
      f"min={int(counts.min())}, max={int(counts.max())}"
    )

  device = env.device
  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  completed = torch.zeros_like(active)
  terminated = torch.zeros_like(active)
  elapsed_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
  metric_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
  body_pos_error_sum = torch.zeros(num_envs, device=device)
  joint_pos_error_sum = torch.zeros(num_envs, device=device)
  target_error_sum = torch.zeros(num_envs, device=device)
  contact_target_error = torch.full((num_envs,), torch.nan, device=device)

  obs = env.get_observations()
  for _ in range(cfg.max_steps):
    if not torch.any(active):
      break

    body_pos_error = torch.linalg.vector_norm(
      command.body_pos_relative_w - command.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    joint_pos_error = torch.linalg.vector_norm(
      command.joint_pos - command.robot_joint_pos, dim=-1
    )
    target_error = torch.linalg.vector_norm(
      command.get_source_pos_w() - command.target_position_w, dim=-1
    )[:, 0]
    body_pos_error_sum[active] += body_pos_error[active]
    joint_pos_error_sum[active] += joint_pos_error[active]
    target_error_sum[active] += target_error[active]
    metric_steps[active] += 1

    at_contact = (
      active
      & torch.isnan(contact_target_error)
      & (command.time_remaining <= command.cfg.contact_reward_window_s)
    )
    contact_target_error[at_contact] = target_error[at_contact]

    motion_complete = active & (command.phase >= 1.0 - 1.0e-6)
    completed[motion_complete] = True
    active[motion_complete] = False
    if not torch.any(active):
      break

    actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    elapsed_steps[active] += 1
    newly_terminated = active & dones.bool()
    terminated[newly_terminated] = True
    active[newly_terminated] = False

  timed_out = active.clone()
  active[:] = False
  valid_steps = metric_steps.clamp(min=1)
  body_pos_error = body_pos_error_sum / valid_steps
  joint_pos_error = joint_pos_error_sum / valid_steps
  target_error = target_error_sum / valid_steps
  contact_valid = torch.isfinite(contact_target_error)
  contact_success = contact_valid & (
    contact_target_error <= cfg.target_success_threshold_m
  )

  per_motion_completion = torch.zeros(motion_count, device=device)
  per_motion_contact_success = torch.zeros(motion_count, device=device)
  for motion_id in range(motion_count):
    selected = initial_motion_ids == motion_id
    per_motion_completion[motion_id] = completed[selected].float().mean()
    valid_selected = selected & contact_valid
    per_motion_contact_success[motion_id] = (
      contact_success[valid_selected].float().mean()
      if torch.any(valid_selected)
      else torch.nan
    )

  results = {
    "task_id": task_id,
    "motion_config": str(cfg.motion_config.resolve()),
    "checkpoint_file": str(cfg.checkpoint_file.resolve()),
    "motion_count": motion_count,
    "episodes_per_motion": cfg.episodes_per_motion,
    "episodes": num_envs,
    "completion_rate": float(completed.float().mean().item()),
    "early_termination_rate": float(terminated.float().mean().item()),
    "evaluation_timeout_rate": float(timed_out.float().mean().item()),
    "mean_body_pos_error_m": float(body_pos_error.mean().item()),
    "mean_joint_pos_error_l2_rad": float(joint_pos_error.mean().item()),
    "mean_target_error_m": float(target_error.mean().item()),
    "contact_observed_rate": float(contact_valid.float().mean().item()),
    "mean_contact_target_error_m": _mean_over_valid(
      contact_target_error, contact_valid
    ),
    "contact_target_success_rate": float(contact_success.float().mean().item()),
    "per_motion_completion_p10": float(
      torch.quantile(per_motion_completion, 0.10).item()
    ),
    "per_motion_completion_median": float(
      torch.quantile(per_motion_completion, 0.50).item()
    ),
    "per_motion_contact_success_p10": float(
      torch.nanquantile(per_motion_contact_success, 0.10).item()
    ),
    "per_motion_contact_success_median": float(
      torch.nanquantile(per_motion_contact_success, 0.50).item()
    ),
  }
  env.close()

  print(json.dumps(results, indent=2, ensure_ascii=False))
  if cfg.output_file is not None:
    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_file.write_text(
      json.dumps(results, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
    )
  return results


def main() -> None:
  import mjlab.tasks

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    TeacherGeneralizationConfig,
    args=remaining_args,
    config=mjlab.TYRO_FLAGS,
  )
  run_evaluation(chosen_task, cfg)


if __name__ == "__main__":
  main()
