"""Motion set loader — reads a TOML config and produces registry strings and MotionCfg lists."""

from __future__ import annotations

import copy
from glob import glob as expand_glob
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from athlete.goal_cond_tracking.mdp import MotionCfg

_MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_ISAACLAB_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
_ISAACLAB_TO_MUJOCO = np.array(
    [_ISAACLAB_JOINT_NAMES.index(name) for name in _MUJOCO_JOINT_NAMES],
    dtype=np.int64,
)

# Paths in [robot] sections are relative to the repo root (athlete/).
# This file lives at src/athlete/motion_sets/motion_set.py,
# so parents[4] is the repo root for editable installs.
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _detect_joint_order(model, joint_pos: np.ndarray) -> str:
    n_joints = min(joint_pos.shape[1], model.njnt - 1)
    jnt_lo = np.zeros(n_joints)
    jnt_hi = np.zeros(n_joints)
    for i in range(n_joints):
        jid = i + 1
        if model.jnt_limited[jid]:
            jnt_lo[i] = model.jnt_range[jid, 0]
            jnt_hi[i] = model.jnt_range[jid, 1]
        else:
            jnt_lo[i] = -1e6
            jnt_hi[i] = 1e6

    vals_mj = joint_pos[:, :n_joints]
    violations_mj = np.sum((vals_mj < jnt_lo - 0.02) | (vals_mj > jnt_hi + 0.02))

    vals_isaac = joint_pos[:, _ISAACLAB_TO_MUJOCO[:n_joints]]
    violations_isaac = np.sum(
        (vals_isaac < jnt_lo - 0.02) | (vals_isaac > jnt_hi + 0.02)
    )
    return "mujoco" if violations_mj <= violations_isaac else "isaaclab"


@dataclass(frozen=True)
class MotionEntry:
    """One motion entry from a motion-set TOML file."""

    name: str
    enabled: bool = True
    file: str | None = None
    strike_frame: int | None = None
    target_from_strike_site: str | None = None
    target_anchor_body: str = "pelvis"
    target_pos_offset: dict[str, float] = field(default_factory=dict)
    target_pos_mean: dict[str, float] | None = None
    target_pos_std: dict[str, float] | None = None
    target_vel_std: dict[str, float] | None = None
    target_orientation_std: dict[str, float] | None = None
    sampling_weight: float | None = None


@dataclass
class MotionSet:
    """Loaded motion set from a TOML config file."""

    train_prefix: str
    _entries: list[MotionEntry]
    _base_dir: Path = field(default_factory=Path)
    robot_xml: str | None = field(default=None)
    """Absolute path to the robot XML, resolved from [robot].xml in the TOML."""

    @classmethod
    def from_toml(cls, path: str | Path) -> "MotionSet":
        config_path = Path(path).resolve()
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        registry = data.get("registry", {})
        entries = [
            MotionEntry(
                name=m["name"],
                enabled=m.get("enabled", True),
                file=m.get("file"),
                strike_frame=m.get("strike_frame"),
                target_from_strike_site=m.get("target_from_strike_site"),
                target_anchor_body=m.get("target_anchor_body", "pelvis"),
                target_pos_offset={
                    key: float(value)
                    for key, value in m.get("target_pos_offset", {}).items()
                },
                target_pos_mean=(
                    {
                        key: float(value)
                        for key, value in m["target_pos_mean"].items()
                    }
                    if "target_pos_mean" in m
                    else None
                ),
                target_pos_std=(
                    {
                        key: float(value)
                        for key, value in m["target_pos_std"].items()
                    }
                    if "target_pos_std" in m
                    else None
                ),
                target_vel_std=(
                    {
                        key: float(value)
                        for key, value in m["target_vel_std"].items()
                    }
                    if "target_vel_std" in m
                    else None
                ),
                target_orientation_std=(
                    {
                        key: float(value)
                        for key, value in m["target_orientation_std"].items()
                    }
                    if "target_orientation_std" in m
                    else None
                ),
                sampling_weight=(
                    float(m["sampling_weight"])
                    if "sampling_weight" in m
                    else None
                ),
            )
            for m in data.get("motions", [])
        ]
        dataset = data.get("dataset")
        if dataset is not None:
            raw_pattern = dataset.get("glob")
            if not raw_pattern:
                raise ValueError("[dataset].glob is required")
            pattern_path = Path(raw_pattern).expanduser()
            if pattern_path.is_absolute():
                pattern = str(pattern_path)
            else:
                config_pattern = config_path.parent / pattern_path
                repo_pattern = _REPO_ROOT / pattern_path
                pattern = str(
                    config_pattern
                    if expand_glob(str(config_pattern))
                    else repo_pattern
                )
            motion_paths = [Path(p).resolve() for p in sorted(expand_glob(pattern))]
            if not motion_paths:
                raise FileNotFoundError(f"Dataset glob matched no files: {pattern}")

            forehand_template = dataset.get("forehand_template", "collected_forehand")
            backhand_template = dataset.get("backhand_template", "collected_backhand")
            sampling_weight = float(dataset.get("sampling_weight", 1.0))
            target_pos_std = (
                {
                    key: float(value)
                    for key, value in dataset["target_pos_std"].items()
                }
                if "target_pos_std" in dataset
                else None
            )
            target_vel_std = (
                {
                    key: float(value)
                    for key, value in dataset["target_vel_std"].items()
                }
                if "target_vel_std" in dataset
                else None
            )
            target_orientation_std = (
                {
                    key: float(value)
                    for key, value in dataset["target_orientation_std"].items()
                }
                if "target_orientation_std" in dataset
                else None
            )
            for motion_path in motion_paths:
                sidecar = motion_path.with_suffix(".json")
                if not sidecar.exists():
                    raise FileNotFoundError(f"Missing dataset sidecar: {sidecar}")
                with open(sidecar, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                strike_frame = int(meta["strike_frame"])
                ball_local = meta.get("ball_local")
                if not isinstance(ball_local, list) or len(ball_local) != 3:
                    raise ValueError(f"Invalid ball_local in {sidecar}")
                clip = str(meta.get("clip", "")).lower()
                if clip.startswith("fh_"):
                    template = forehand_template
                elif clip.startswith("bh_"):
                    template = backhand_template
                else:
                    raise ValueError(f"Cannot infer forehand/backhand from {sidecar}: {clip!r}")
                entries.append(
                    MotionEntry(
                        name=template,
                        file=str(motion_path),
                        strike_frame=strike_frame,
                        target_pos_mean={
                            "x": float(ball_local[0]),
                            "y": float(ball_local[1]),
                            "z": float(ball_local[2]),
                        },
                        target_pos_std=target_pos_std,
                        target_vel_std=target_vel_std,
                        target_orientation_std=target_orientation_std,
                        sampling_weight=float(
                            meta.get("sampling_weight", sampling_weight)
                        ),
                    )
                )
        robot_xml_raw = data.get("robot", {}).get("xml")
        robot_xml = None
        if robot_xml_raw is not None:
            p = Path(robot_xml_raw)
            robot_xml = str((_REPO_ROOT / p).resolve()) if not p.is_absolute() else str(p)
        loaded_entries: list[MotionEntry] = []
        for entry in entries:
            if entry.file is None or entry.strike_frame is not None:
                loaded_entries.append(entry)
                continue
            try:
                resolved_file = Path(entry.file)
                if not resolved_file.is_absolute():
                    candidates = (config_path.parent / resolved_file, _REPO_ROOT / resolved_file)
                    resolved_file = next(
                        (candidate for candidate in candidates if candidate.exists()),
                        candidates[-1],
                    )
                sidecar = resolved_file.with_suffix(".json")
                if sidecar.exists():
                    with open(sidecar, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    strike_frame = meta.get("strike_frame")
                    if strike_frame is not None:
                        entry = MotionEntry(
                            name=entry.name,
                            enabled=entry.enabled,
                            file=entry.file,
                            strike_frame=int(strike_frame),
                            target_from_strike_site=entry.target_from_strike_site,
                            target_anchor_body=entry.target_anchor_body,
                            target_pos_offset=entry.target_pos_offset,
                            target_pos_mean=entry.target_pos_mean,
                            target_pos_std=entry.target_pos_std,
                            target_vel_std=entry.target_vel_std,
                            target_orientation_std=entry.target_orientation_std,
                            sampling_weight=entry.sampling_weight,
                        )
            except Exception:
                # Leave the entry unchanged if metadata is missing or malformed.
                pass
            loaded_entries.append(entry)

        return cls(
            train_prefix=registry.get("train_prefix", ""),
            _entries=loaded_entries,
            _base_dir=config_path.parent,
            robot_xml=robot_xml,
        )

    @property
    def enabled_names(self) -> list[str]:
        return [entry.name for entry in self._entries if entry.enabled]

    @property
    def local_motion_files(self) -> list[str] | None:
        """Return local motion files for enabled entries, or None for registry-backed sets."""
        enabled_entries = [entry for entry in self._entries if entry.enabled]
        if not any(entry.file for entry in enabled_entries):
            return None

        missing_files = [entry.name for entry in enabled_entries if entry.file is None]
        if missing_files:
            raise ValueError(
                "Motion config mixes local files and registry entries. Missing file for: "
                f"{missing_files}"
            )

        return [
            self._resolve_motion_file(entry.file)
            for entry in enabled_entries
            if entry.file is not None
        ]

    def local_motion_cfgs(self, lib: "dict[str, MotionCfg] | None" = None) -> "list[MotionCfg] | None":
        """Return motion configs for enabled entries, phase-aligned to local metadata.

        If a sidecar ``.json`` exists next to a local motion file and provides
        ``strike_frame``, the motion's phase windows are shifted so that the
        center of the first sub-target aligns with that frame.
        """
        enabled_entries = [entry for entry in self._entries if entry.enabled]
        if not any(entry.file for entry in enabled_entries):
            return None

        if lib is None:
            from athlete.motion_sets.motion_lib import MOTION_LIB

            lib = MOTION_LIB

        missing_files = [entry.name for entry in enabled_entries if entry.file is None]
        if missing_files:
            raise ValueError(
                "Motion config mixes local files and registry entries. Missing file for: "
                f"{missing_files}"
            )

        cfgs = []
        for entry in enabled_entries:
            if entry.name not in lib:
                raise ValueError(f"Motion names not found in library: {entry.name}")
            motion_cfg = copy.deepcopy(lib[entry.name])
            if entry.sampling_weight is not None:
                motion_cfg.sampling_weight = entry.sampling_weight
            if entry.strike_frame is not None and entry.file is not None:
                motion_file = Path(self._resolve_motion_file(entry.file))
                with np.load(motion_file) as data:
                    total_frames = int(data["joint_pos"].shape[0])
                if total_frames > 1 and motion_cfg.sub_targets:
                    strike_phase = entry.strike_frame / float(total_frames - 1)
                    self._shift_motion_cfg_to_phase(motion_cfg, strike_phase)
                if entry.target_from_strike_site is not None:
                    target_pos = self._target_pos_from_strike_site(
                        motion_file,
                        entry.strike_frame,
                        entry.target_from_strike_site,
                        entry.target_anchor_body,
                    )
                    target_pos = {
                        axis: value + entry.target_pos_offset.get(axis, 0.0)
                        for axis, value in target_pos.items()
                    }
                    for st in motion_cfg.sub_targets:
                        if st.goal_type == "position":
                            st.target_pos_mean = target_pos
                            st.target_pos_frame = "reference_anchor_frame0"
                    print(
                        "[INFO] Target from strike site "
                        f"{entry.target_from_strike_site}@{entry.strike_frame}: {target_pos}"
                    )
                if entry.target_pos_mean is not None:
                    for st in motion_cfg.sub_targets:
                        if st.goal_type == "position":
                            st.target_pos_mean = dict(entry.target_pos_mean)
                            st.target_pos_frame = "reference_anchor_frame0"
                if entry.target_pos_std is not None:
                    for st in motion_cfg.sub_targets:
                        if st.goal_type == "position":
                            st.target_pos_std = dict(entry.target_pos_std)
                if entry.target_vel_std is not None:
                    for st in motion_cfg.sub_targets:
                        if st.goal_type == "velocity":
                            st.target_vel_std = dict(entry.target_vel_std)
                if entry.target_orientation_std is not None:
                    for st in motion_cfg.sub_targets:
                        if st.goal_type == "orientation":
                            st.target_orientation_std = dict(
                                entry.target_orientation_std
                            )
            cfgs.append(motion_cfg)
        return cfgs

    def _resolve_motion_file(self, raw_file: str) -> str:
        path = Path(raw_file)
        if path.is_absolute():
            resolved = path
        else:
            candidates = (self._base_dir / path, _REPO_ROOT / path)
            resolved = next(
                (candidate for candidate in candidates if candidate.exists()),
                candidates[-1],
            )

        if not resolved.exists():
            raise FileNotFoundError(
                f"Motion file not found: {raw_file} (resolved to {resolved})"
            )
        return str(resolved.resolve())

    @staticmethod
    def _shift_motion_cfg_to_phase(motion_cfg: "MotionCfg", target_center: float) -> None:
        if not motion_cfg.sub_targets:
            return
        base_start = motion_cfg.sub_targets[0].target_phase_start
        base_end = motion_cfg.sub_targets[0].target_phase_end
        current_center = (base_start + base_end) / 2.0
        delta = target_center - current_center

        for st in motion_cfg.sub_targets:
            duration = max(0.0, st.target_phase_end - st.target_phase_start)
            center = (st.target_phase_start + st.target_phase_end) / 2.0 + delta
            start = max(0.0, min(1.0 - duration, center - duration / 2.0))
            st.target_phase_start = start
            st.target_phase_end = min(1.0, start + duration)

    def _target_pos_from_strike_site(
        self,
        motion_file: Path,
        strike_frame: int,
        site_name: str,
        anchor_body_name: str,
    ) -> dict[str, float]:
        if self.robot_xml is None:
            raise ValueError(
                "target_from_strike_site requires [robot].xml so MuJoCo can "
                "forward the strike frame."
            )

        import mujoco

        model = mujoco.MjModel.from_xml_path(self.robot_xml)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Site not found in robot XML: {site_name}")
        anchor_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, anchor_body_name
        )
        if anchor_body_id < 0:
            raise ValueError(f"Anchor body not found in robot XML: {anchor_body_name}")

        with np.load(motion_file) as data_npz:
            joint_pos = data_npz["joint_pos"].astype(np.float64)
            body_pos_w = data_npz["body_pos_w"].astype(np.float64)
            body_quat_w = data_npz["body_quat_w"].astype(np.float64)

        frame = max(0, min(int(strike_frame), joint_pos.shape[0] - 1))
        joint_order = _detect_joint_order(model, joint_pos.astype(np.float32))

        def joint_frame_at(frame_id: int) -> np.ndarray:
            values = joint_pos[frame_id].copy()
            if joint_order == "isaaclab":
                values = values[_ISAACLAB_TO_MUJOCO]
            return values

        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        data.qpos[:3] = body_pos_w[frame, 0]
        data.qpos[3:7] = body_quat_w[frame, 0]
        strike_joint_frame = joint_frame_at(frame)
        data.qpos[7 : 7 + len(strike_joint_frame)] = strike_joint_frame
        mujoco.mj_forward(model, data)
        site_pos_w = data.site_xpos[site_id].copy()

        # Preserve root displacement by expressing the strike site in the
        # first-frame anchor coordinates, where each episode starts.
        data.qpos[:] = model.qpos0
        data.qpos[:3] = body_pos_w[0, 0]
        data.qpos[3:7] = body_quat_w[0, 0]
        first_joint_frame = joint_frame_at(0)
        data.qpos[7 : 7 + len(first_joint_frame)] = first_joint_frame
        mujoco.mj_forward(model, data)
        anchor_pos_w = data.xpos[anchor_body_id].copy()
        anchor_rot_w = data.xmat[anchor_body_id].reshape(3, 3).copy()
        target_pos_b = anchor_rot_w.T @ (site_pos_w - anchor_pos_w)

        return {
            "x": float(target_pos_b[0]),
            "y": float(target_pos_b[1]),
            "z": float(target_pos_b[2]),
        }

    def train_registry(self) -> str:
        if not self.train_prefix:
            return ",".join(self.enabled_names)
        return ",".join(f"{self.train_prefix}/{name}" for name in self.enabled_names)

    def motion_cfgs(self, lib: "dict[str, MotionCfg] | None" = None) -> "list[MotionCfg]":
        """Return MotionCfg entries for the enabled motions, in config order.

        Args:
            lib: Motion library to look up specs from. Defaults to MOTION_LIB.
        """
        if lib is None:
            from athlete.motion_sets.motion_lib import MOTION_LIB
            lib = MOTION_LIB
        missing = [n for n in self.enabled_names if n not in lib]
        if missing:
            raise ValueError(f"Motion names not found in library: {missing}")
        return [lib[name] for name in self.enabled_names]


def filter_motion_cmd_cfg(motion_cmd_cfg: "MultiTargetMotionCommandCfg", motion_names: list[str]) -> None:
    """Trim motion_target_cfgs and motion_sampling_weights to match a subset of motions.

    When training or evaluating with fewer motions than the full library, the env cfg
    still holds all entries from MOTION_LIB. This aligns the two so _sample_motion_ids
    doesn't crash on a size mismatch between weights and loaded files.
    """
    import copy

    from athlete.motion_sets.motion_lib import MOTION_LIB
    from athlete.goal_cond_tracking.mdp import MultiTargetMotionCommandCfg as _Cfg

    assert isinstance(motion_cmd_cfg, _Cfg)
    full_cfgs = {m.name: m for m in motion_cmd_cfg.motion_target_cfgs if m.name}
    subset = []
    for name in motion_names:
        if name in full_cfgs:
            subset.append(copy.deepcopy(full_cfgs[name]))
        elif name in MOTION_LIB:
            subset.append(copy.deepcopy(MOTION_LIB[name]))
        else:
            raise ValueError(f"No motion config found for '{name}'. Add it to motion_lib.py.")

    motion_cmd_cfg.motion_target_cfgs = subset
    motion_cmd_cfg.motion_sampling_weights = [m.sampling_weight for m in subset]


def set_motion_cmd_cfgs(
    motion_cmd_cfg: "MultiTargetMotionCommandCfg", motion_cfgs: "list[MotionCfg]"
) -> None:
    """Replace the motion config list with an explicit subset."""
    import copy

    from athlete.goal_cond_tracking.mdp import MultiTargetMotionCommandCfg as _Cfg

    assert isinstance(motion_cmd_cfg, _Cfg)
    motion_cmd_cfg.motion_target_cfgs = [copy.deepcopy(cfg) for cfg in motion_cfgs]
    motion_cmd_cfg.motion_sampling_weights = [cfg.sampling_weight for cfg in motion_cfgs]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Print the --registry-name string for a motion set config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train registry string (default)
  uv run motion-set motion_sets/all_motions.toml

  # Eval registry string
  uv run motion-set motion_sets/all_motions.toml --eval

  # List enabled motion names
  uv run motion-set motion_sets/all_motions.toml --list
""",
    )
    parser.add_argument("config", type=Path, help="Path to motion set TOML config")
    parser.add_argument("--list", action="store_true", help="Print enabled motion names, one per line")
    args = parser.parse_args()

    ms = MotionSet.from_toml(args.config)

    if args.list:
        for name in ms.enabled_names:
            print(name)
    else:
        print(ms.train_registry())


if __name__ == "__main__":
    main()
