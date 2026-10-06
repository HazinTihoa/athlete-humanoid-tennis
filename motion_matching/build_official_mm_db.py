#!/usr/bin/env python3
"""Build current MuJoCo motion-matching resources from G1 NPZ files.

The generated directory contains:
  - database.bin
  - features.bin
  - g1_state.npz
  - metadata.json

It targets the 33-dimensional feature layout used by tennis_official_mm_mj.py:
the original 27 official features plus right-hand local position and velocity.
Pass --strike-dir to append tennis forehand/backhand clips and write strike
metadata used by the runtime entry planner.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


ISAACLAB_G1_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

MUJOCO_JOINT_NAMES = [
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
]

ISAACLAB_JOINT_NAMES = [
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
]

ISAACLAB_TO_MUJOCO = np.asarray([ISAACLAB_JOINT_NAMES.index(name) for name in MUJOCO_JOINT_NAMES], dtype=np.int32)

ISAACLAB_G1_PARENTS = np.asarray(
    [
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
    ],
    dtype=np.int32,
)

BODY_NAMES = ["Entity"] + ISAACLAB_G1_BODY_NAMES
BODY_PARENTS = np.asarray(
    [-1] + [0 if parent < 0 else int(parent) + 1 for parent in ISAACLAB_G1_PARENTS],
    dtype=np.int32,
)

ROOT_INDEX = 0
PELVIS_INDEX = 1
LEFT_FOOT_INDEX = 19
RIGHT_FOOT_INDEX = 20
RIGHT_HAND_INDEX = BODY_NAMES.index("right_wrist_yaw_link")
TRAJECTORY_OFFSETS = (20, 40, 60)
BASE_FEATURE_DIM = 27
FEATURE_DIM = BASE_FEATURE_DIM + 6
RIGHT_HAND_POSITION_SLICE = slice(BASE_FEATURE_DIM, BASE_FEATURE_DIM + 3)
RIGHT_HAND_VELOCITY_SLICE = slice(BASE_FEATURE_DIM + 3, FEATURE_DIM)
DEFAULT_G1_XML = str(Path(__file__).resolve().parents[1] / "robots/replay_unitree_description/mjcf/g1.xml")
RIGHT_WRIST_BODY_NAME = "right_wrist_yaw_link"
RACKET_OFFSET_Z_UP = np.asarray([0.38, 0.01, 0.27], dtype=np.float32)


def z_up_to_internal_y_up(v: np.ndarray) -> np.ndarray:
    """Convert G1/MuJoCo z-up vectors to the demo's internal y-up vectors."""
    return np.stack([v[..., 0], v[..., 2], v[..., 1]], axis=-1).astype(np.float32)


def yaw_from_quat_z_up(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(np.float32)


def quat_from_internal_yaw(angle: np.ndarray) -> np.ndarray:
    half = 0.5 * angle
    q = np.zeros(angle.shape + (4,), dtype=np.float32)
    q[..., 0] = np.cos(half)
    q[..., 2] = np.sin(half)
    return q


def quat_rotate_z_up(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    t = 2.0 * np.cross(q[..., 1:], v)
    return (v + q[..., 0:1] * t + np.cross(q[..., 1:], t)).astype(np.float32)


def quat_inv(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    t = 2.0 * np.cross(q[..., 1:], v)
    return (v + q[..., 0:1] * t + np.cross(q[..., 1:], t)).astype(np.float32)


def quat_inv_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_rotate(quat_inv(q), v)


def quat_identity(shape: tuple[int, ...]) -> np.ndarray:
    q = np.zeros(shape + (4,), dtype=np.float32)
    q[..., 0] = 1.0
    return q


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return (np.gradient(values.astype(np.float32), axis=0) * float(fps)).astype(np.float32)


def write_array1d(f, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    f.write(struct.pack("i", arr.shape[0]))
    f.write(arr.tobytes())


def write_array2d(f, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    f.write(struct.pack("ii", arr.shape[0], arr.shape[1]))
    f.write(arr.tobytes())


def normalize_feature_group(
    features: np.ndarray,
    features_offset: np.ndarray,
    features_scale: np.ndarray,
    offset: int,
    size: int,
    weight: float,
) -> int:
    values = features[:, offset : offset + size]
    mean = values.mean(axis=0)
    std = np.sqrt(((values - mean) ** 2).mean(axis=0)).mean()
    if not np.isfinite(std) or std <= 1e-8:
        std = 1.0
    scale = std / weight
    features_offset[offset : offset + size] = mean
    features_scale[offset : offset + size] = scale
    features[:, offset : offset + size] = (values - mean) / scale
    return offset + size


def trajectory_index(frame: int, offset: int, start: int, stop: int) -> int:
    return int(np.clip(frame + offset, start, stop - 1))


def compute_fk_body_state(
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    joint_pos: np.ndarray,
    fps: float,
    g1_xml: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(g1_xml))
    data = mujoco.MjData(model)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ISAACLAB_G1_BODY_NAMES]
    if any(body_id < 0 for body_id in body_ids):
        missing = [name for name, body_id in zip(ISAACLAB_G1_BODY_NAMES, body_ids) if body_id < 0]
        raise ValueError(f"{g1_xml} missing bodies: {missing}")
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_WRIST_BODY_NAME)
    if wrist_id < 0:
        raise ValueError(f"{g1_xml} missing body {RIGHT_WRIST_BODY_NAME}")

    nframes = int(root_pos_w.shape[0])
    body_pos = np.zeros((nframes, len(ISAACLAB_G1_BODY_NAMES), 3), dtype=np.float32)
    body_quat = np.zeros((nframes, len(ISAACLAB_G1_BODY_NAMES), 4), dtype=np.float32)
    racket_pos = np.zeros((nframes, 3), dtype=np.float32)
    joint_pos_mj = joint_pos[:, ISAACLAB_TO_MUJOCO]
    for frame in range(nframes):
        data.qpos[:3] = root_pos_w[frame].astype(np.float64)
        data.qpos[3:7] = root_quat_w[frame].astype(np.float64)
        data.qpos[7:36] = joint_pos_mj[frame].astype(np.float64)
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[body_ids].astype(np.float32)
        body_quat[frame] = data.xquat[body_ids].astype(np.float32)
        wrist_xmat = data.xmat[wrist_id].reshape(3, 3).astype(np.float32)
        racket_pos[frame] = data.xpos[wrist_id].astype(np.float32) + wrist_xmat @ RACKET_OFFSET_Z_UP

    body_vel = finite_difference(body_pos, fps)
    body_ang_vel = np.zeros_like(body_vel, dtype=np.float32)
    return body_pos, body_quat, body_vel, body_ang_vel, racket_pos


def build_clip(
    npz_path: Path,
    clip_type: str = "locomotion",
    strike_frame: int = -1,
    g1_xml: Path | None = None,
) -> dict[str, np.ndarray | int | float | str]:
    with np.load(npz_path) as data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        body_pos_z = data["body_pos_w"].astype(np.float32)
        body_quat_z = data["body_quat_w"].astype(np.float32)
        body_vel_z = data["body_lin_vel_w"].astype(np.float32)
        body_ang_vel_z = data["body_ang_vel_w"].astype(np.float32)
        joint_pos = data["joint_pos"].astype(np.float32)
        joint_vel = data["joint_vel"].astype(np.float32)

    if joint_pos.shape[1] != len(ISAACLAB_JOINT_NAMES):
        raise ValueError(f"{npz_path} has {joint_pos.shape[1]} joints; expected {len(ISAACLAB_JOINT_NAMES)}")

    racket_pos_z = np.zeros((body_pos_z.shape[0], 3), dtype=np.float32)
    if body_pos_z.shape[1] != len(ISAACLAB_G1_BODY_NAMES):
        if g1_xml is None:
            raise ValueError(f"{npz_path} has {body_pos_z.shape[1]} bodies; pass --g1-xml for FK remapping")
        body_pos_z, body_quat_z, body_vel_z, body_ang_vel_z, racket_pos_z = compute_fk_body_state(
            body_pos_z[:, 0],
            body_quat_z[:, 0],
            joint_pos,
            fps,
            g1_xml,
        )

    nframes = body_pos_z.shape[0]
    body_pos = z_up_to_internal_y_up(body_pos_z)
    body_vel = z_up_to_internal_y_up(body_vel_z)
    racket_pos = z_up_to_internal_y_up(racket_pos_z)

    yaw_z = yaw_from_quat_z_up(body_quat_z[:, 0])
    internal_yaw = (0.5 * math.pi - yaw_z).astype(np.float32)
    root_rot = quat_from_internal_yaw(internal_yaw)
    yaw_unwrapped = np.unwrap(internal_yaw.astype(np.float64)).astype(np.float32)
    yaw_rate = np.gradient(yaw_unwrapped) * fps
    root_ang_vel = np.zeros((nframes, 3), dtype=np.float32)
    root_ang_vel[:, 1] = yaw_rate

    left_h = body_pos[:, LEFT_FOOT_INDEX - 1, 1]
    right_h = body_pos[:, RIGHT_FOOT_INDEX - 1, 1]
    ground_height = float(min(left_h.min(), right_h.min()))

    world_pos = np.zeros((nframes, len(BODY_NAMES), 3), dtype=np.float32)
    world_vel = np.zeros_like(world_pos)
    world_pos[:, 0, 0] = body_pos[:, 0, 0]
    world_pos[:, 0, 1] = ground_height
    world_pos[:, 0, 2] = body_pos[:, 0, 2]
    world_pos[:, 1:] = body_pos
    world_vel[:, 0] = body_vel[:, 0]
    world_vel[:, 0, 1] = 0.0
    world_vel[:, 1:] = body_vel

    local_pos = np.zeros_like(world_pos, dtype=np.float32)
    local_vel = np.zeros_like(world_vel, dtype=np.float32)
    local_rot = quat_identity((nframes, len(BODY_NAMES)))
    local_ang_vel = np.zeros_like(world_vel, dtype=np.float32)

    local_pos[:, ROOT_INDEX] = world_pos[:, ROOT_INDEX]
    local_vel[:, ROOT_INDEX] = world_vel[:, ROOT_INDEX]
    local_rot[:, ROOT_INDEX] = root_rot
    local_ang_vel[:, ROOT_INDEX] = root_ang_vel

    for i, parent in enumerate(BODY_PARENTS):
        if parent < 0:
            continue
        delta = world_pos[:, i] - world_pos[:, parent]
        local_pos[:, i] = quat_inv_rotate(root_rot, delta)
        rel_vel = world_vel[:, i] - world_vel[:, parent] - np.cross(root_ang_vel, delta)
        local_vel[:, i] = quat_inv_rotate(root_rot, rel_vel)

    features_raw = np.zeros((nframes, FEATURE_DIM), dtype=np.float32)
    root_pos = world_pos[:, ROOT_INDEX]

    features_raw[:, 0:3] = quat_inv_rotate(root_rot, world_pos[:, LEFT_FOOT_INDEX] - root_pos)
    features_raw[:, 3:6] = quat_inv_rotate(root_rot, world_pos[:, RIGHT_FOOT_INDEX] - root_pos)
    features_raw[:, 6:9] = quat_inv_rotate(root_rot, world_vel[:, LEFT_FOOT_INDEX])
    features_raw[:, 9:12] = quat_inv_rotate(root_rot, world_vel[:, RIGHT_FOOT_INDEX])
    features_raw[:, 12:15] = quat_inv_rotate(root_rot, world_vel[:, PELVIS_INDEX])

    # Tennis query extension: racket-hand pose relative to the virtual root and
    # hand velocity relative to root translation, both in the facing frame. The
    # root-relative velocity captures the arm swing instead of duplicating the
    # already-present root velocity feature.
    features_raw[:, RIGHT_HAND_POSITION_SLICE] = quat_inv_rotate(
        root_rot, world_pos[:, RIGHT_HAND_INDEX] - root_pos
    )
    features_raw[:, RIGHT_HAND_VELOCITY_SLICE] = quat_inv_rotate(
        root_rot, world_vel[:, RIGHT_HAND_INDEX] - world_vel[:, ROOT_INDEX]
    )

    frames = np.arange(nframes)
    forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    for j, offset in enumerate(TRAJECTORY_OFFSETS):
        future = np.clip(frames + offset, 0, nframes - 1)
        local_delta = quat_inv_rotate(root_rot, root_pos[future] - root_pos)
        features_raw[:, 15 + 2 * j] = local_delta[:, 0]
        features_raw[:, 16 + 2 * j] = local_delta[:, 2]

        future_forward = quat_rotate(root_rot[future], np.broadcast_to(forward, (nframes, 3)))
        local_forward = quat_inv_rotate(root_rot, future_forward)
        features_raw[:, 21 + 2 * j] = local_forward[:, 0]
        features_raw[:, 22 + 2 * j] = local_forward[:, 2]

    left_speed = np.linalg.norm(world_vel[:, LEFT_FOOT_INDEX], axis=1)
    right_speed = np.linalg.norm(world_vel[:, RIGHT_FOOT_INDEX], axis=1)
    contact_states = np.stack(
        [
            (world_pos[:, LEFT_FOOT_INDEX, 1] < ground_height + 0.08) & (left_speed < 1.0),
            (world_pos[:, RIGHT_FOOT_INDEX, 1] < ground_height + 0.08) & (right_speed < 1.0),
        ],
        axis=1,
    ).astype(np.uint8)

    return {
        "fps": fps,
        "local_pos": local_pos,
        "local_vel": local_vel,
        "local_rot": local_rot,
        "local_ang_vel": local_ang_vel,
        "features_raw": features_raw,
        "contact_states": contact_states,
        "clip_type": clip_type,
        "strike_frame": int(strike_frame),
        "racket_pos_internal": racket_pos,
        "g1_root_pos_w": body_pos_z[:, 0],
        "g1_root_quat_w": body_quat_z[:, 0],
        "g1_root_lin_vel_w": body_vel_z[:, 0],
        "g1_root_ang_vel_w": body_ang_vel_z[:, 0],
        "g1_joint_pos": joint_pos,
        "g1_joint_vel": joint_vel,
    }


def write_database(
    output_dir: Path,
    bone_positions: np.ndarray,
    bone_velocities: np.ndarray,
    bone_rotations: np.ndarray,
    bone_angular_velocities: np.ndarray,
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    contact_states: np.ndarray,
) -> None:
    with open(output_dir / "database.bin", "wb") as f:
        write_array2d(f, bone_positions.astype(np.float32))
        write_array2d(f, bone_velocities.astype(np.float32))
        write_array2d(f, bone_rotations.astype(np.float32))
        write_array2d(f, bone_angular_velocities.astype(np.float32))
        write_array1d(f, BODY_PARENTS.astype(np.int32))
        write_array1d(f, range_starts.astype(np.int32))
        write_array1d(f, range_stops.astype(np.int32))
        write_array2d(f, contact_states.astype(np.uint8))


def write_features(
    output_dir: Path,
    features_raw: np.ndarray,
    foot_position_weight: float,
    foot_velocity_weight: float,
    hip_velocity_weight: float,
    trajectory_position_weight: float,
    trajectory_direction_weight: float,
    hand_position_weight: float,
    hand_velocity_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = features_raw.astype(np.float32, copy=True)
    features_offset = np.zeros(features.shape[1], dtype=np.float32)
    features_scale = np.ones(features.shape[1], dtype=np.float32)

    offset = 0
    offset = normalize_feature_group(features, features_offset, features_scale, offset, 3, foot_position_weight)
    offset = normalize_feature_group(features, features_offset, features_scale, offset, 3, foot_position_weight)
    offset = normalize_feature_group(features, features_offset, features_scale, offset, 3, foot_velocity_weight)
    offset = normalize_feature_group(features, features_offset, features_scale, offset, 3, foot_velocity_weight)
    offset = normalize_feature_group(features, features_offset, features_scale, offset, 3, hip_velocity_weight)
    offset = normalize_feature_group(features, features_offset, features_scale, offset, 6, trajectory_position_weight)
    offset = normalize_feature_group(features, features_offset, features_scale, offset, 6, trajectory_direction_weight)
    offset = normalize_feature_group(features, features_offset, features_scale, offset, 3, hand_position_weight)
    offset = normalize_feature_group(features, features_offset, features_scale, offset, 3, hand_velocity_weight)
    assert offset == FEATURE_DIM

    with open(output_dir / "features.bin", "wb") as f:
        write_array2d(f, features)
        write_array1d(f, features_offset)
        write_array1d(f, features_scale)

    return features, features_offset, features_scale


def write_g1_state(output_dir: Path, clips: list[dict[str, np.ndarray | int | float]]) -> None:
    np.savez_compressed(
        output_dir / "g1_state.npz",
        root_pos_w=np.concatenate([clip["g1_root_pos_w"] for clip in clips], axis=0).astype(np.float32),
        root_quat_w=np.concatenate([clip["g1_root_quat_w"] for clip in clips], axis=0).astype(np.float32),
        root_lin_vel_w=np.concatenate([clip["g1_root_lin_vel_w"] for clip in clips], axis=0).astype(np.float32),
        root_ang_vel_w=np.concatenate([clip["g1_root_ang_vel_w"] for clip in clips], axis=0).astype(np.float32),
        joint_pos=np.concatenate([clip["g1_joint_pos"] for clip in clips], axis=0).astype(np.float32),
        joint_vel=np.concatenate([clip["g1_joint_vel"] for clip in clips], axis=0).astype(np.float32),
        isaaclab_joint_names=np.asarray(ISAACLAB_JOINT_NAMES),
        mujoco_joint_names=np.asarray(MUJOCO_JOINT_NAMES),
    )


def write_strike_metadata(output_dir: Path, clips: list[dict[str, np.ndarray | int | float | str]]) -> None:
    np.savez_compressed(
        output_dir / "strike_metadata.npz",
        racket_pos_internal=np.concatenate([clip["racket_pos_internal"] for clip in clips], axis=0).astype(np.float32),
    )


def strike_frame_for(npz_path: Path) -> int:
    json_path = npz_path.with_suffix(".json")
    if not json_path.exists():
        return -1
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("strike_frame", -1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/lafan_npz_data/runandwalk"),
        help="folder containing G1 run/walk NPZ files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("motion_db/official_tennis_runwalk_hand33"),
        help="directory to write database.bin/features.bin/metadata.json",
    )
    parser.add_argument(
        "--strike-dir",
        type=Path,
        default=Path("data/djvkovic_npz_data_v5"),
        help="optional tennis strike NPZ folder; clips with *_g1.json strike_frame annotations are appended",
    )
    parser.add_argument(
        "--g1-xml",
        type=Path,
        default=Path(DEFAULT_G1_XML),
        help="G1 MJCF used to remap 34-body tennis clips to the 30-body runtime order",
    )
    parser.add_argument("--max-files", type=int, default=0, help="limit files for a quick test; 0 means all")
    parser.add_argument("--foot-position-weight", type=float, default=0.75)
    parser.add_argument("--foot-velocity-weight", type=float, default=1.0)
    parser.add_argument("--hip-velocity-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-position-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-direction-weight", type=float, default=1.5)
    parser.add_argument("--hand-position-weight", type=float, default=0.5)
    parser.add_argument("--hand-velocity-weight", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_files = sorted(args.input_dir.glob("*.npz"))
    if args.max_files > 0:
        npz_files = npz_files[: args.max_files]
    if not npz_files:
        raise FileNotFoundError(f"no .npz files found in {args.input_dir}")

    strike_files: list[Path] = []
    if args.strike_dir is not None:
        strike_files = [path for path in sorted(args.strike_dir.glob("*.npz")) if strike_frame_for(path) >= 0]
        if args.max_files > 0:
            strike_files = strike_files[: args.max_files]
        if not strike_files:
            raise FileNotFoundError(f"no annotated strike .npz files found in {args.strike_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    range_starts = []
    range_stops = []
    fps_values = []
    frame_offset = 0
    for path in npz_files:
        clip = build_clip(path, clip_type="locomotion", strike_frame=-1, g1_xml=args.g1_xml)
        nframes = int(clip["local_pos"].shape[0])
        clips.append(clip)
        range_starts.append(frame_offset)
        frame_offset += nframes
        range_stops.append(frame_offset)
        fps_values.append(float(clip["fps"]))
        print(f"{path.name}: {nframes} frames @ {float(clip['fps']):g} fps")

    for path in strike_files:
        sf = strike_frame_for(path)
        clip = build_clip(path, clip_type="strike", strike_frame=sf, g1_xml=args.g1_xml)
        nframes = int(clip["local_pos"].shape[0])
        clips.append(clip)
        range_starts.append(frame_offset)
        frame_offset += nframes
        range_stops.append(frame_offset)
        fps_values.append(float(clip["fps"]))
        print(f"{path.name}: {nframes} frames @ {float(clip['fps']):g} fps strike_frame={sf}")

    fps_unique = sorted(set(fps_values))
    if len(fps_unique) != 1:
        raise ValueError(f"mixed FPS values are not supported: {fps_unique}")

    bone_positions = np.concatenate([clip["local_pos"] for clip in clips], axis=0)
    bone_velocities = np.concatenate([clip["local_vel"] for clip in clips], axis=0)
    bone_rotations = np.concatenate([clip["local_rot"] for clip in clips], axis=0)
    bone_angular_velocities = np.concatenate([clip["local_ang_vel"] for clip in clips], axis=0)
    features_raw = np.concatenate([clip["features_raw"] for clip in clips], axis=0)
    contact_states = np.concatenate([clip["contact_states"] for clip in clips], axis=0)
    range_starts_arr = np.asarray(range_starts, dtype=np.int32)
    range_stops_arr = np.asarray(range_stops, dtype=np.int32)

    write_database(
        args.output_dir,
        bone_positions,
        bone_velocities,
        bone_rotations,
        bone_angular_velocities,
        range_starts_arr,
        range_stops_arr,
        contact_states,
    )
    features, features_offset, features_scale = write_features(
        args.output_dir,
        features_raw,
        args.foot_position_weight,
        args.foot_velocity_weight,
        args.hip_velocity_weight,
        args.trajectory_position_weight,
        args.trajectory_direction_weight,
        args.hand_position_weight,
        args.hand_velocity_weight,
    )
    write_g1_state(args.output_dir, clips)
    write_strike_metadata(args.output_dir, clips)

    source_paths = npz_files + strike_files

    metadata = {
        "source": "G1 runandwalk NPZ + optional tennis strike NPZ, IsaacLab BFS joint order",
        "source_input_dir": str(args.input_dir),
        "source_strike_dir": None if args.strike_dir is None else str(args.strike_dir),
        "source_files": [path.name for path in source_paths],
        "fps": fps_unique[0],
        "num_frames": int(bone_positions.shape[0]),
        "num_bodies": len(BODY_NAMES),
        "feature_dim": FEATURE_DIM,
        "feature_schema_version": 2,
        "g1_state": "g1_state.npz",
        "strike_metadata": "strike_metadata.npz",
        "g1_state_coordinate_system": "mujoco_z_up_wxyz",
        "g1_joint_order": "isaaclab_bfs",
        "mujoco_joint_order": MUJOCO_JOINT_NAMES,
        "isaaclab_joint_order": ISAACLAB_JOINT_NAMES,
        "bone_names": BODY_NAMES,
        "root_index": ROOT_INDEX,
        "pelvis_index": PELVIS_INDEX,
        "left_foot_index": LEFT_FOOT_INDEX,
        "right_foot_index": RIGHT_FOOT_INDEX,
        "right_hand_index": RIGHT_HAND_INDEX,
        "foot_indices": [15, LEFT_FOOT_INDEX, 16, RIGHT_FOOT_INDEX],
        "feature_layout": [
            "left_foot_position_local_xyz",
            "right_foot_position_local_xyz",
            "left_foot_velocity_local_xyz",
            "right_foot_velocity_local_xyz",
            "root_velocity_local_xyz",
            "future_root_positions_local_xz_at_20_40_60",
            "future_root_directions_local_xz_at_20_40_60",
            "right_hand_position_local_xyz",
            "right_hand_velocity_root_relative_local_xyz",
        ],
        "feature_weights": {
            "foot_position": args.foot_position_weight,
            "foot_velocity": args.foot_velocity_weight,
            "hip_velocity": args.hip_velocity_weight,
            "trajectory_position": args.trajectory_position_weight,
            "trajectory_direction": args.trajectory_direction_weight,
            "hand_position": args.hand_position_weight,
            "hand_velocity": args.hand_velocity_weight,
        },
        "ranges": [
            {
                "file": path.name,
                "start": int(start),
                "stop": int(stop),
                "type": str(clip["clip_type"]),
                "strike_frame": int(clip["strike_frame"]),
                "side": "fh" if path.name.lower().startswith("fh_") else "bh" if path.name.lower().startswith("bh_") else "",
            }
            for path, clip, start, stop in zip(source_paths, clips, range_starts_arr, range_stops_arr)
        ],
        "racket_offset_z_up": RACKET_OFFSET_Z_UP.tolist(),
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"wrote {args.output_dir / 'database.bin'}")
    print(f"wrote {args.output_dir / 'features.bin'}")
    print(f"wrote {args.output_dir / 'g1_state.npz'}")
    print(f"wrote {args.output_dir / 'strike_metadata.npz'}")
    print(f"wrote {args.output_dir / 'metadata.json'}")
    print(
        f"summary: frames={bone_positions.shape[0]} bodies={bone_positions.shape[1]} "
        f"features={features.shape[1]} fps={fps_unique[0]:g}"
    )
    print(
        "feature scale range: "
        f"min={float(features_scale.min()):.6g} max={float(features_scale.max()):.6g}; "
        f"offset mean={float(features_offset.mean()):.6g}"
    )


if __name__ == "__main__":
    main()
