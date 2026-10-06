"""Script to train RL agent with RSL-RL."""

import copy
import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from athlete.goal_cond_tracking.mdp import MultiTargetMotionCommandCfg
from athlete.scripts.tennis_scene import (
  DEFAULT_BALL_RELEASE_LEAD_S,
  add_racket_ball_collision,
  install_tennis_ball_controller,
)


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_config: Path | None = None
  """Path to a motion set TOML. Drives registry and robot XML."""
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  smoke_test: bool = False
  """Store this run under logs/smoke_tests instead of the formal training logs."""
  resume_checkpoint: Path | None = None
  """Explicit local checkpoint to resume, including across formal/smoke log roots."""
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  debug_viewer: Literal["none", "auto", "native", "viser"] = "none"
  """Open a live viewer against a separate debug rollout while training."""
  debug_viewer_num_envs: int = 1
  """Number of environments to use in the debug viewer rollout."""

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  registry_name: str | None = None

  # Check if this is a tracking task by checking for motion command.

  is_multi_target_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MultiTargetMotionCommandCfg
  )
  has_physical_tennis = "tennis_ball" in cfg.env.scene.entities

  if is_multi_target_task:
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MultiTargetMotionCommandCfg)

    # Load motion set from TOML when --motion-config is provided.
    motion_set = None
    if cfg.motion_config is not None:
      import mujoco
      from athlete.motion_sets.motion_set import (
        MotionSet,
        filter_motion_cmd_cfg,
        set_motion_cmd_cfgs,
      )
      motion_set = MotionSet.from_toml(cfg.motion_config)

      if motion_set.robot_xml is not None:
        _xml = motion_set.robot_xml
        robot_entity = cfg.env.scene.entities.get("robot")
        if robot_entity is not None:
          robot_spec_fn = lambda: mujoco.MjSpec.from_file(_xml)
          if has_physical_tennis:
            robot_spec_fn = add_racket_ball_collision(robot_spec_fn)
          robot_entity.spec_fn = robot_spec_fn
        print(f"[INFO] Robot XML from motion config: {_xml}")

      registry_name = motion_set.train_registry()
      local_motion_files = motion_set.local_motion_files
      local_motion_cfgs = motion_set.local_motion_cfgs()
      if local_motion_files is not None and local_motion_cfgs is not None:
        motion_cmd.motion_files = local_motion_files
        set_motion_cmd_cfgs(motion_cmd, local_motion_cfgs)
        print(f"[INFO] Motion files from motion config: {local_motion_files}")

    if motion_cmd.motion_files and all(
      Path(f).exists() for f in motion_cmd.motion_files
    ):
      print(f"[INFO] Using local motion files: {motion_cmd.motion_files}")
    elif registry_name:
      registry_names = [r.strip() for r in registry_name.split(",")]
      import wandb

      api = wandb.Api()
      motion_files: list[str] = []
      motion_names: list[str] = []
      for rn in registry_names:
        if ":" not in rn:
          rn = rn + ":latest"
        artifact = api.artifact(rn)
        motion_files.append(str(Path(artifact.download()) / "motion.npz"))
        motion_names.append(rn.split("/")[-1].split(":")[0])
        print(f"[INFO] Downloaded motion: {rn} -> {motion_files[-1]}")
      motion_cmd.motion_files = motion_files
      filter_motion_cmd_cfg(motion_cmd, motion_set.enabled_names if motion_set else motion_names)
    else:
      raise ValueError(
        "For multi-target tracking tasks, provide either:\n"
        "  --motion-config path/to/config.toml (reads registry + robot XML from TOML)\n"
        "  --env.commands.motion.motion-files '[f1.npz,f2.npz]' (local files)"
      )

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )
  if has_physical_tennis:
    install_tennis_ball_controller(env, DEFAULT_BALL_RELEASE_LEAD_S)

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.resume_checkpoint is not None:
    resume_path = cfg.resume_checkpoint.expanduser().resolve(strict=True)
  elif cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  if is_multi_target_task:
    runner_kwargs["registry_name"] = registry_name

  # Write config files before runner creation, since the runner mutates agent_cfg
  # in-place (e.g., injecting non-serializable objects).
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  if cfg.debug_viewer != "none" and runner.is_distributed:
    raise ValueError("debug_viewer is not supported for distributed training.")

  if cfg.debug_viewer == "none":
    runner.learn(
      num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
    )
    env.close()
    return

  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  if cfg.debug_viewer == "auto":
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.debug_viewer
  print(f"[INFO] Debug viewer: {resolved_viewer} (live)")

  debug_env_cfg = copy.deepcopy(cfg.env)
  if resolved_viewer == "viser":
    # ViewerConfig uses MuJoCo's negative elevation for an above-ground camera;
    # mjviser's spherical camera offset uses positive elevation for that view.
    debug_env_cfg.viewer.elevation = -debug_env_cfg.viewer.elevation
  debug_env_cfg.scene.num_envs = cfg.debug_viewer_num_envs
  terrain = getattr(debug_env_cfg.scene, "terrain", None)
  if terrain is not None and hasattr(terrain, "num_envs"):
    terrain.num_envs = cfg.debug_viewer_num_envs

  # Match the play-time debug behavior so the window reflects the actual motion setup.
  if is_multi_target_task:
    debug_motion_cmd = debug_env_cfg.commands["motion"]
    assert isinstance(debug_motion_cmd, MultiTargetMotionCommandCfg)
    if cfg.motion_config is not None:
      from athlete.motion_sets.motion_set import (
        MotionSet,
        filter_motion_cmd_cfg,
        set_motion_cmd_cfgs,
      )

      motion_set = MotionSet.from_toml(cfg.motion_config)
      if motion_set.robot_xml is not None:
        import mujoco

        _xml = motion_set.robot_xml
        robot_entity = debug_env_cfg.scene.entities.get("robot")
        if robot_entity is not None:
          robot_spec_fn = lambda: mujoco.MjSpec.from_file(_xml)
          if has_physical_tennis:
            robot_spec_fn = add_racket_ball_collision(robot_spec_fn)
          robot_entity.spec_fn = robot_spec_fn
      local_motion_files = motion_set.local_motion_files
      local_motion_cfgs = motion_set.local_motion_cfgs()
      if local_motion_files is not None and local_motion_cfgs is not None:
        debug_motion_cmd.motion_files = local_motion_files
        set_motion_cmd_cfgs(debug_motion_cmd, local_motion_cfgs)

  debug_env = ManagerBasedRlEnv(cfg=debug_env_cfg, device=device, render_mode=None)
  if has_physical_tennis:
    install_tennis_ball_controller(
      debug_env,
      DEFAULT_BALL_RELEASE_LEAD_S,
      verbose_samples=True,
    )
  debug_env = RslRlVecEnvWrapper(debug_env, clip_actions=cfg.agent.clip_actions)
  training_debug_policy_fn = getattr(
    runner.alg, "get_training_debug_policy", None
  )
  environment_policy_fn = getattr(runner.alg, "get_environment_policy", None)

  def _source_debug_policy():
    if training_debug_policy_fn is not None:
      return training_debug_policy_fn()
    if environment_policy_fn is not None:
      return environment_policy_fn()
    return runner.alg.get_policy()

  debug_policy = copy.deepcopy(_source_debug_policy()).to(device)
  debug_policy.eval()
  teacher_probability = getattr(debug_policy, "teacher_action_probability", None)
  if teacher_probability is not None:
    print(
      "[INFO] Debug viewer TPPO Teacher control probability: "
      f"{float(teacher_probability.item()):.3f}"
    )
  debug_policy_lock = threading.Lock()
  stop_debug = threading.Event()

  class _PolicyMirror:
    def __call__(self, obs):
      with debug_policy_lock:
        return debug_policy(obs)

    def reset(self, dones: torch.Tensor | None = None) -> None:
      reset_fn = getattr(debug_policy, "reset", None)
      if reset_fn is not None:
        with debug_policy_lock:
          reset_fn(dones)

  viewer_policy = _PolicyMirror()
  original_debug_step = debug_env.step

  def _step_debug_env(actions):
    result = original_debug_step(actions)
    dones = result[2]
    if torch.any(dones):
      viewer_policy.reset(dones)
    return result

  debug_env.step = _step_debug_env

  def _sync_debug_policy() -> None:
    with debug_policy_lock:
      debug_policy.load_state_dict(_source_debug_policy().state_dict())
      debug_policy.eval()

  def _run_debug_viewer() -> None:
    try:
      if resolved_viewer == "native":
        viewer = NativeMujocoViewer(debug_env, viewer_policy)
      elif resolved_viewer == "viser":
        viewer = ViserPlayViewer(debug_env, viewer_policy)
      else:
        raise RuntimeError(f"Unsupported debug viewer backend: {resolved_viewer}")
      _sync_debug_policy()
      viewer.run(catch_sigint=False)
    finally:
      stop_debug.set()
      debug_env.close()

  debug_thread = threading.Thread(target=_run_debug_viewer, name="train-debug-viewer", daemon=True)
  debug_thread.start()

  try:
    runner.env.episode_length_buf = torch.randint_like(
      runner.env.episode_length_buf, high=int(runner.env.max_episode_length)
    )
    obs = runner.env.get_observations().to(device)
    start_it = runner.current_learning_iteration
    total_it = start_it + cfg.agent.max_iterations
    runner.alg.train_mode()

    if runner.is_distributed:
      print(f"Synchronizing parameters for rank {runner.gpu_global_rank}...")
      runner.alg.broadcast_parameters()

    runner.logger.init_logging_writer()

    for it in range(start_it, total_it):
      prepare_rollout_fn = getattr(runner, "prepare_rollout", None)
      if prepare_rollout_fn is not None:
        obs = prepare_rollout_fn(obs)
      start = time.time()
      with torch.inference_mode():
        for _ in range(cfg.agent.num_steps_per_env):
          actions = runner.alg.act(obs)
          obs, rewards, dones, extras = runner.env.step(actions.to(runner.env.device))
          if runner.cfg.get("check_for_nan", True):
            from rsl_rl.utils import check_nan

            check_nan(obs, rewards, dones)
          obs, rewards, dones = (
            obs.to(runner.device),
            rewards.to(runner.device),
            dones.to(runner.device),
          )
          runner.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic_rewards = (
            runner.alg.intrinsic_rewards if runner.cfg["algorithm"]["rnd_cfg"] else None
          )
          runner.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

        stop = time.time()
        collect_time = stop - start
        start = stop
        runner.alg.compute_returns(obs)

      loss_dict = runner.alg.update()
      _sync_debug_policy()

      stop = time.time()
      learn_time = stop - start
      runner.current_learning_iteration = it

      runner.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=runner.alg.learning_rate,
        action_std=runner.alg.get_policy().output_std,
        rnd_weight=runner.alg.rnd.weight if runner.cfg["algorithm"]["rnd_cfg"] else None,
      )

      if runner.logger.writer is not None and it % runner.cfg["save_interval"] == 0:
        runner.save(os.path.join(runner.logger.log_dir, f"model_{it}.pt"))

    if runner.logger.writer is not None:
      runner.save(
        os.path.join(runner.logger.log_dir, f"model_{runner.current_learning_iteration}.pt")
      )
      runner.logger.stop_logging_writer()
  finally:
    stop_debug.set()
    try:
      debug_env.close()
    except Exception:
      pass
    debug_thread.join(timeout=5.0)

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  # Local logging is the default. Set WANDB_MODE=online/offline explicitly to opt in.
  os.environ.setdefault("WANDB_MODE", "disabled")
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_category = "smoke_tests" if args.smoke_test else "rsl_rl"
  log_root_path = Path("logs") / log_category / args.agent.experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name
  if args.smoke_test:
    print(f"[INFO] Smoke-test outputs will be stored under: {log_root_path}")

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
