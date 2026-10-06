"""Entry point identical to `play` but always uses the viser web viewer.

Motion artifacts are resolved via the train registry (csv_to_npz project) rather
than the eval registry (org-level wandb-registry-motions), so project-level W&B
access is sufficient — no org membership required.

The resolved files and filter are injected into the env config before run_play
creates the environment, so the sampling-weight tensor stays consistent with the
number of loaded motion files.
"""

import dataclasses
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import tyro

from mjlab.tasks.registry import list_tasks, load_rl_cfg
from athlete.scripts.play import PlayConfig, run_play

if TYPE_CHECKING:
    from athlete.motion_sets.motion_set import MotionSet


def _resolve_motion_files_via_train_registry(
    motion_config: Path,
) -> tuple[list[str], "MotionSet"]:
    """Download motion artifacts via train_prefix; return (file_paths, motion_set)."""
    import wandb
    from athlete.motion_sets.motion_set import MotionSet

    motion_set = MotionSet.from_toml(motion_config)
    api = wandb.Api()
    motion_files: list[str] = []
    for rn in motion_set.train_registry().split(","):
        rn = rn.strip()
        if ":" not in rn:
            rn = rn + ":latest"
        artifact = api.artifact(rn)
        motion_files.append(str(Path(artifact.download()) / "motion.npz"))
        print(f"[INFO]: Downloaded motion (train registry): {rn} -> {motion_files[-1]}")
    return motion_files, motion_set


@contextmanager
def _pre_resolve_env_motions(task_id: str, motion_files: list[str], motion_set: "MotionSet"):
    """Temporarily patch load_env_cfg so the motion command is pre-filtered and
    pre-populated before run_play constructs the environment.

    This ensures the sampling-weight tensor (built from all motions in motion_lib)
    is filtered down to match the number of loaded motion files, which play.py
    omits when the motion_file path is taken instead of the registry path.

    play.py binds load_env_cfg via `from mjlab.tasks.registry import load_env_cfg`,
    so patching only the registry module attribute is not enough — we must also
    update the name in play.py's module namespace.
    """
    import athlete.scripts.play as _play_module
    from mjlab.tasks import registry as _task_registry

    original = _task_registry.load_env_cfg

    def _patched(tid, play=False):
        cfg = original(tid, play=play)
        if tid == task_id and "motion" in cfg.commands:
            motion_cmd = cfg.commands["motion"]
            from athlete.motion_sets.motion_set import filter_motion_cmd_cfg
            filter_motion_cmd_cfg(motion_cmd, motion_set.enabled_names)
            motion_cmd.motion_files = motion_files
        return cfg

    _task_registry.load_env_cfg = _patched
    _play_module.load_env_cfg = _patched
    try:
        yield
    finally:
        _task_registry.load_env_cfg = original
        _play_module.load_env_cfg = original


def main():
    import mjlab  # noqa: F401
    import mjlab.tasks  # noqa: F401 — populates task registry

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    agent_cfg = load_rl_cfg(chosen_task)
    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(viewer="viser"),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    del remaining_args, agent_cfg

    # Guarantee viser regardless of any --viewer flag the user passed.
    args = dataclasses.replace(args, viewer="viser")

    if args.motion_config is not None and args.motion_file is None:
        # Resolve motion files from W&B and pre-inject into the env config so the
        # motion-command sampling weights stay consistent with the loaded files.
        motion_files, motion_set = _resolve_motion_files_via_train_registry(args.motion_config)
        with _pre_resolve_env_motions(chosen_task, motion_files, motion_set):
            run_play(chosen_task, args)
    else:
        run_play(chosen_task, args)


if __name__ == "__main__":
    main()
