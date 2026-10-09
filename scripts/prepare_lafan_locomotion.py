#!/usr/bin/env python3
"""Convert locally obtained, G1-retargeted LAFAN1 CSV to TGMM NPZ files.

The input CSVs use the 30 Hz G1 format published by the LAFAN1 Retargeting
Dataset: root position, root quaternion (xyzw), then 29 MuJoCo-order joints.
Motion files are third-party data and are deliberately excluded from Git.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_matching.build_official_mm_db import (
    DEFAULT_G1_XML,
    ISAACLAB_G1_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    MUJOCO_JOINT_NAMES,
)


def slerp(q0: np.ndarray, q1: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where((dot < 0.0)[:, None], -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    safe = sin_theta > 1e-6
    left = np.where(safe, np.sin((1.0 - fraction) * theta) / np.maximum(sin_theta, 1e-12), 1.0 - fraction)
    right = np.where(safe, np.sin(fraction * theta) / np.maximum(sin_theta, 1e-12), fraction)
    q = left[:, None] * q0 + right[:, None] * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def angular_velocity(quat: np.ndarray, fps: float) -> np.ndarray:
    """World-frame angular velocity from central quaternion differences."""
    first = np.concatenate((quat[:1], quat[:-1]), axis=0)
    last = np.concatenate((quat[1:], quat[-1:]), axis=0)
    inverse = first.copy()
    inverse[..., 1:] *= -1.0
    w = last[..., 0] * inverse[..., 0] - np.sum(last[..., 1:] * inverse[..., 1:], axis=-1)
    v = (
        last[..., :1] * inverse[..., 1:]
        + inverse[..., :1] * last[..., 1:]
        + np.cross(last[..., 1:], inverse[..., 1:])
    )
    v = np.where((w < 0.0)[..., None], -v, v)
    w = np.abs(w)
    norm = np.linalg.norm(v, axis=-1)
    angle = 2.0 * np.arctan2(norm, w)
    scale = np.where(norm > 1e-10, angle / np.maximum(norm, 1e-10), 2.0)
    velocity = v * scale[..., None] * (fps / 2.0)
    velocity[0] *= 2.0
    velocity[-1] *= 2.0
    return velocity.astype(np.float32)


def convert(csv_path: Path, npz_path: Path, model: mujoco.MjModel, body_ids: np.ndarray,
            input_fps: int, output_fps: int) -> int:
    csv = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    if csv.ndim != 2 or csv.shape[1] != 36 or csv.shape[0] < 2 or not np.isfinite(csv).all():
        raise ValueError(f"{csv_path}: expected at least two finite rows of 36 G1 values")

    times = np.arange(0.0, (len(csv) - 1) / input_fps, 1.0 / output_fps, dtype=np.float32)
    source_index = times * input_fps
    i0 = np.floor(source_index).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(csv) - 1)
    fraction = source_index - i0
    alpha = fraction[:, None]

    root_pos = csv[i0, :3] * (1.0 - alpha) + csv[i1, :3] * alpha
    source_quat = csv[:, [6, 3, 4, 5]]  # xyzw -> MuJoCo wxyz
    root_quat = slerp(source_quat[i0], source_quat[i1], fraction)
    joint_mj = csv[i0, 7:] * (1.0 - alpha) + csv[i1, 7:] * alpha
    isaac_from_mj = [MUJOCO_JOINT_NAMES.index(name) for name in ISAACLAB_JOINT_NAMES]
    joint_pos = joint_mj[:, isaac_from_mj].astype(np.float32)
    joint_vel = np.gradient(joint_pos, axis=0).astype(np.float32) * output_fps

    body_pos = np.empty((len(times), len(body_ids), 3), dtype=np.float32)
    body_quat = np.empty((len(times), len(body_ids), 4), dtype=np.float32)
    data = mujoco.MjData(model)
    for frame in range(len(times)):
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        data.qpos[7:36] = joint_mj[frame]
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[body_ids]
        body_quat[frame] = data.xquat[body_ids]

    body_lin_vel = np.gradient(body_pos, axis=0).astype(np.float32) * output_fps
    body_ang_vel = angular_velocity(body_quat, output_fps)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        fps=np.asarray([output_fps], dtype=np.int64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
    )
    return len(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with G1 run/walk CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/lafan_npz_data/runandwalk"))
    parser.add_argument("--input-fps", type=int, default=30)
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument("--g1-xml", type=Path, default=Path(DEFAULT_G1_XML))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.input_fps <= 0 or args.output_fps <= 0:
        parser.error("frame rates must be positive")

    csv_files = sorted((*args.input_dir.glob("run*.csv"), *args.input_dir.glob("walk*.csv")))
    if not csv_files:
        parser.error(f"no run/walk CSV files found in {args.input_dir}")
    model = mujoco.MjModel.from_xml_path(str(args.g1_xml))
    body_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ISAACLAB_G1_BODY_NAMES],
        dtype=np.int32,
    )
    if np.any(body_ids < 0):
        parser.error(f"{args.g1_xml} is missing required G1 bodies")

    for csv_path in csv_files:
        npz_path = args.output_dir / f"{csv_path.stem}.npz"
        if npz_path.exists() and not args.overwrite:
            print(f"skip {npz_path} (use --overwrite to replace)")
            continue
        frames = convert(csv_path, npz_path, model, body_ids, args.input_fps, args.output_fps)
        print(f"wrote {npz_path}: {frames} frames @ {args.output_fps} fps")


if __name__ == "__main__":
    main()
