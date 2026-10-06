"""Like play_viser.py but overlays transparent ellipsoid blobs for every static
position goal in the motion library.

Each blob is centred at target_pos_mean (in the robot's anchor/root frame) and
has semi-axes equal to target_pos_std.  Blobs update every frame so they stay
attached to the robot's anchor body as it moves.

Motion files are resolved via the train registry (csv_to_npz project) so only
project-level W&B access is required.
"""

import dataclasses
import sys
from contextlib import contextmanager

import numpy as np
import tyro
from scipy.spatial.transform import Rotation

from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab.viewer import ViserPlayViewer
from athlete.scripts.play import PlayConfig, run_play
from athlete.scripts.play_viser import (
    _pre_resolve_env_motions,
    _resolve_motion_files_via_train_registry,
)

# 8 visually distinct colours (R, G, B, A) — alpha 0.25 gives transparency.
_COLORS: list[tuple[float, float, float, float]] = [
    (0.90, 0.20, 0.20, 0.25),  # red
    (0.20, 0.55, 0.90, 0.25),  # blue
    (0.20, 0.80, 0.30, 0.25),  # green
    (0.90, 0.75, 0.10, 0.25),  # yellow
    (0.70, 0.25, 0.90, 0.25),  # purple
    (0.95, 0.50, 0.10, 0.25),  # orange
    (0.10, 0.85, 0.85, 0.25),  # cyan
    (0.95, 0.30, 0.65, 0.25),  # pink
]


class MotionRangesViewer(ViserPlayViewer):
    """ViserPlayViewer that draws transparent ellipsoids for motion position goals."""

    def setup(self) -> None:
        super().setup()
        self._motion_goal_data = self._extract_motion_goals()

    def _extract_motion_goals(self) -> list[tuple[np.ndarray, np.ndarray, tuple]]:
        """Return (mean_anchor, std_anchor, color) for every static position goal.

        Reads from cmd.cfg.motion_target_cfgs, which has already been filtered to
        the motions enabled by the --motion-config .toml file.
        """
        try:
            cmd = self.env.unwrapped.command_manager.get_term("motion")
        except (AttributeError, KeyError):
            return []

        result: list[tuple[np.ndarray, np.ndarray, tuple]] = []
        for i, motion_cfg in enumerate(cmd.cfg.motion_target_cfgs):
            color = _COLORS[i % len(_COLORS)]
            for st in motion_cfg.sub_targets:
                if st.goal_type != "position" or st.target_link is not None:
                    continue
                mean = np.array([st.target_pos_mean.get(k, 0.0) for k in "xyz"],
                                dtype=np.float32)
                std = np.maximum(
                    np.array([st.target_pos_std.get(k, 0.0) for k in "xyz"],
                             dtype=np.float32),
                    0.02,  # 2 cm floor so blobs are always visible
                )
                result.append((mean, std, color))

        print(f"[INFO] MotionRangesViewer: {len(result)} position-goal blobs from {len(cmd.cfg.motion_target_cfgs)} motions.")
        return result

    def _queue_debug_visualizers(self) -> None:
        super()._queue_debug_visualizers()
        if self._motion_goal_data:
            with self._sim_lock:
                self._queue_motion_range_blobs()

    def _queue_motion_range_blobs(self) -> None:
        env_idx = self._scene.env_idx
        try:
            cmd = self.env.unwrapped.command_manager.get_term("motion")
        except (AttributeError, KeyError):
            return

        anchor_pos = cmd.robot_anchor_pos_w[env_idx].cpu().numpy()
        w, x, y, z = cmd.robot_anchor_quat_w[env_idx].cpu().numpy()
        R = Rotation.from_quat([x, y, z, w]).as_matrix().astype(np.float32)

        for mean_a, std_a, color in self._motion_goal_data:
            center = (R @ mean_a + anchor_pos).astype(np.float32)
            self._scene.add_ellipsoid(center, std_a, R, color)


@contextmanager
def _patch_viser_viewer():
    """Temporarily replace ViserPlayViewer with MotionRangesViewer in both namespaces."""
    import mjlab.viewer as _viewer_mod
    import athlete.scripts.play as _play_mod

    _viewer_mod.ViserPlayViewer = MotionRangesViewer  # type: ignore[attr-defined]
    _play_mod.ViserPlayViewer = MotionRangesViewer
    try:
        yield
    finally:
        _viewer_mod.ViserPlayViewer = ViserPlayViewer  # type: ignore[attr-defined]
        _play_mod.ViserPlayViewer = ViserPlayViewer


def main():
    import mjlab  # noqa: F401
    import mjlab.tasks  # noqa: F401  — populates task registry

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

    with _patch_viser_viewer():
        if args.motion_config is not None and args.motion_file is None:
            motion_files, motion_set = _resolve_motion_files_via_train_registry(
                args.motion_config
            )
            with _pre_resolve_env_motions(chosen_task, motion_files, motion_set):
                run_play(chosen_task, args)
        else:
            run_play(chosen_task, args)


if __name__ == "__main__":
    main()
