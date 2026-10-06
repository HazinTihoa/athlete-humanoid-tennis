#!/usr/bin/env python3
"""Extract the mocap ball trajectory from a recorded ROS 2 bag and replay it in MuJoCo.

Two stages, both in this file:

  extract  Read `/rigid_bodies` (raw OptiTrack measurement of the ball rigid body)
           and `/ball/pose` (the bridge's filtered/predicted output) out of a
           `ros2 bag` directory and save them as one .npz.

  replay   Load the deployment model (robot + court + ball, identical to
           `bash run_tppo.sh sim`) and visualize only the recorded raw mocap
           and Bridge ball streams, plus an offline measurement-first candidate.
           No robot state or policy topics are needed.
           A separate Tk window scrubs the time; the MuJoCo window only renders.

Why `/rigid_bodies` and not `/ball/pose`: `/ball/pose` is
`BallStateEstimator.estimate()` output (natnet_policy_bridge.py:465-481), i.e. a
predictor-corrector state propagated forward by up to `maximum_prediction_s`
(natnet_bridge_state.py:254-270). During an occlusion it is pure prediction, not
a measurement. `/rigid_bodies` is the untouched OptiTrack input.

The extract stage needs the `rosbags` package (not in requirements-jazzy.txt):

    source deploy/setup_deploy.zsh && pip install rosbags

Usage:

    python estimation/mocap/replay_bag_ball.py \
        recordings/2026-09-08_21-09-30_real_g1_tppo_student_m14_9
    python estimation/mocap/replay_bag_ball.py <bag> --extract-only
    python estimation/mocap/replay_bag_ball.py --replay-only --npz <file.npz>
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import math
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


DEPLOY_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = DEPLOY_DIR.parent
sys.path.insert(0, str(DEPLOY_DIR))

from utils.tppo_simulation_physics import build_training_aligned_model  # noqa: E402
from utils.natnet_bridge_state import BallEstimatorConfig, MeasurementFirstBallStream  # noqa: E402


IDENTITY = np.eye(3, dtype=np.float64).reshape(-1)
RAW_COLOR = np.array([1.0, 0.48, 0.05, 0.95], dtype=np.float32)
BRIDGE_COLOR = np.array([0.08, 0.55, 1.0, 0.95], dtype=np.float32)
DIRECT_COLOR = np.array([0.15, 0.9, 0.35, 0.95], dtype=np.float32)
TRAIL_POINT_RADIUS_M = 0.009

BALL_JOINT = "tennis_ball_freejoint"
SPEED_CHOICES = ("0.1", "0.25", "0.5", "1", "2")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def load_config(config: str | Path) -> dict:
    path = Path(config).expanduser()
    if not path.is_absolute():
        candidate = DEPLOY_DIR / "configs" / path
        path = candidate if candidate.exists() else REPO_ROOT / path
    path = path.resolve()
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise TypeError(f"Deployment config must be a mapping: {path}")
    return loaded


def repo_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


# --------------------------------------------------------------------------- #
# Extract
# --------------------------------------------------------------------------- #


def _load_typestore(msg_dir: Path):
    """ROS 2 Jazzy typestore with mocap_msgs registered from the .msg sources."""
    from rosbags.typesys import Stores, get_typestore
    from rosbags.typesys.msg import get_types_from_msg

    typestore = get_typestore(Stores.ROS2_JAZZY)
    types: dict = {}
    for name in ("Marker", "RigidBody", "RigidBodies"):
        source = msg_dir / f"{name}.msg"
        if not source.is_file():
            raise FileNotFoundError(f"Missing message definition: {source}")
        types.update(get_types_from_msg(source.read_text(), f"mocap_msgs/msg/{name}"))
    typestore.register(types)
    return typestore


def _default_msg_dir() -> Path | None:
    home = Path.home()
    # setup_deploy.zsh sources $HOME/optitrack_companion_ws, so that is first.
    for base in (
        home / "optitrack_companion_ws" / "src" / "mocap_msgs" / "msg",
        home / "mocap4ros2_optitrack_demos",
    ):
        if (base / "RigidBodies.msg").is_file():
            return base
        if base.is_dir():
            for found in sorted(base.rglob("mocap_msgs/msg/RigidBodies.msg")):
                return found.parent
    return None


def fit_planar_transform(
    source_pose: np.ndarray, target_pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the Bridge yaw/XY calibration from paired Pelvis poses.

    Position-only fitting cannot recover yaw while the robot is stationary.
    The Bridge calibration is a yaw about Z plus an XY translation, so derive
    yaw from the paired orientations and translation from paired positions.
    """
    if source_pose.ndim != 2 or source_pose.shape[1] != 7:
        raise ValueError("source_pose must have shape (N, 7)")
    if target_pose.shape != source_pose.shape:
        raise ValueError("target_pose must have the same shape as source_pose")

    source_rotation = Rotation.from_quat(source_pose[:, 3:])
    target_rotation = Rotation.from_quat(target_pose[:, 3:])
    relative = target_rotation * source_rotation.inv()
    matrices = relative.as_matrix()
    yaws = np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0])
    center = math.atan2(float(np.sin(yaws).mean()), float(np.cos(yaws).mean()))
    yaw = center + float(np.median(np.angle(np.exp(1j * (yaws - center)))))
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    rotation = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    translations = target_pose[:, :3] - source_pose[:, :3] @ rotation.T
    translation = np.median(translations, axis=0)
    # The live Bridge deliberately preserves the mocap floor/Z coordinate.
    translation[2] = 0.0
    return rotation, translation


def _match_by_time(
    query_t: np.ndarray, reference_t: np.ndarray, tolerance_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-reference match. Returns (query indices, reference indices)."""
    indices = np.searchsorted(reference_t, query_t)
    indices = np.clip(indices, 1, reference_t.size - 1)
    left = reference_t[indices - 1]
    right = reference_t[indices]
    pick_left = np.abs(query_t - left) <= np.abs(right - query_t)
    indices = indices - pick_left.astype(np.intp)
    gaps = np.abs(query_t - reference_t[indices])
    keep = np.flatnonzero(gaps <= tolerance_s)
    return keep, indices[keep]


def align_raw_positions_to_bridge_frame(
    raw_time: np.ndarray,
    raw_position: np.ndarray,
    pelvis_raw_time: np.ndarray,
    pelvis_raw_pose: np.ndarray,
    pelvis_bridge_time: np.ndarray,
    pelvis_bridge_pose: np.ndarray,
    calibration_requests: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Apply the Bridge frame calibration active at each raw ball sample."""
    aligned = raw_position.copy()
    requests = np.unique(np.sort(calibration_requests))
    starts = np.concatenate(([-np.inf], requests))
    ends = np.concatenate((requests, [np.inf]))
    segments: list[dict] = []
    previous_rotation = np.eye(3, dtype=np.float64)
    previous_translation = np.zeros(3, dtype=np.float64)

    for start, end in zip(starts, ends):
        target_ids = np.flatnonzero(
            (pelvis_bridge_time >= start) & (pelvis_bridge_time < end)
        )
        query_ids, source_ids = _match_by_time(
            pelvis_bridge_time[target_ids], pelvis_raw_time, 0.05
        )
        target_ids = target_ids[query_ids]

        if target_ids.size >= 10:
            rotation, translation = fit_planar_transform(
                pelvis_raw_pose[source_ids], pelvis_bridge_pose[target_ids]
            )
            predicted = pelvis_raw_pose[source_ids, :3] @ rotation.T + translation
            residual = float(
                np.linalg.norm(
                    predicted - pelvis_bridge_pose[target_ids, :3], axis=1
                ).mean()
            )
            previous_rotation = rotation
            previous_translation = translation
        else:
            rotation = previous_rotation
            translation = previous_translation
            residual = float("nan")

        chosen = (raw_time >= start) & (raw_time < end)
        aligned[chosen] = raw_position[chosen] @ rotation.T + translation
        segments.append(
            {
                "start": float(start),
                "end": float(end),
                "yaw_rad": float(math.atan2(rotation[1, 0], rotation[0, 0])),
                "translation": translation.copy(),
                "pairs": int(target_ids.size),
                "residual_m": residual,
            }
        )
    return aligned, segments


def extract(bag_path: Path, config: dict, msg_dir: Path | None) -> dict:
    from rosbags.highlevel import AnyReader

    if msg_dir is None:
        msg_dir = _default_msg_dir()
    if msg_dir is None:
        raise SystemExit(
            "Could not locate mocap_msgs message definitions; pass --mocap-msg-dir."
        )
    typestore = _load_typestore(Path(msg_dir))
    bridge_config = config["mocap_bridge"]
    raw_topic = str(bridge_config.get("input_topic", "/rigid_bodies"))
    ball_name = str(bridge_config["ball_name"])
    pelvis_name = str(bridge_config["pelvis_name"])

    try:
        reader = AnyReader([bag_path], default_typestore=typestore)
    except TypeError:
        reader = AnyReader([bag_path])

    raw_t: list[float] = []
    raw_source_t: list[float] = []
    raw_pos: list[tuple[float, float, float]] = []
    raw_error: list[float] = []
    raw_valid: list[bool] = []
    raw_pose_valid: list[bool] = []
    filtered_t: list[float] = []
    filtered_pos: list[tuple[float, float, float]] = []
    filtered_velocity_t: list[float] = []
    filtered_velocity: list[tuple[float, float, float]] = []
    pelvis_raw_t: list[float] = []
    pelvis_raw_pose: list[tuple[float, ...]] = []
    pelvis_cal_t: list[float] = []
    pelvis_cal_pose: list[tuple[float, ...]] = []
    calibration_requests: list[float] = []

    recorded_topics = {
        raw_topic,
        "/g1_pelvis/pose",
        "/ball/pose",
        "/ball/velocity",
        "/deploy_robot/root_calibration_request",
    }

    with reader:
        for connection, _timestamp, rawdata in reader.messages():
            if connection.topic not in recorded_topics:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            # Recorder receive times provide a common playback clock. They
            # approximate event arrival, not exact Bridge callback execution.
            seconds = float(_timestamp) * 1.0e-9

            if connection.topic == "/deploy_robot/root_calibration_request":
                if message.data == "calibrate":
                    calibration_requests.append(seconds)
                continue

            if connection.topic == "/ball/pose":
                filtered_t.append(seconds)
                point = message.pose.position
                filtered_pos.append((float(point.x), float(point.y), float(point.z)))
                continue

            if connection.topic == "/ball/velocity":
                filtered_velocity_t.append(seconds)
                vector = message.vector
                filtered_velocity.append((float(vector.x), float(vector.y), float(vector.z)))
                continue

            if connection.topic == "/g1_pelvis/pose":
                pelvis_cal_t.append(seconds)
                point, quat = message.pose.position, message.pose.orientation
                pelvis_cal_pose.append(
                    (
                        float(point.x), float(point.y), float(point.z),
                        float(quat.x), float(quat.y), float(quat.z), float(quat.w),
                    )
                )
                continue

            body = next((b for b in message.rigidbodies if b.rigid_body_name == ball_name), None)
            raw_t.append(seconds)
            stamp = message.header.stamp
            raw_source_t.append(float(stamp.sec) + float(stamp.nanosec) * 1.0e-9)
            if body is None:
                raw_pos.append((float("nan"),) * 3)
                raw_error.append(float("inf"))
                raw_valid.append(False)
                raw_pose_valid.append(False)
            else:
                point, quat = body.pose.position, body.pose.orientation
                raw_pos.append((float(point.x), float(point.y), float(point.z)))
                raw_error.append(float(body.mean_error))
                raw_valid.append(bool(body.tracking_valid))
                quaternion = np.array([quat.x, quat.y, quat.z, quat.w])
                raw_pose_valid.append(bool(np.isfinite(quaternion).all() and np.linalg.norm(quaternion) > 1.0e-8))

            for body in message.rigidbodies:
                if body.rigid_body_name == pelvis_name and body.tracking_valid:
                    pelvis_raw_t.append(seconds)
                    point, quat = body.pose.position, body.pose.orientation
                    pelvis_raw_pose.append(
                        (
                            float(point.x), float(point.y), float(point.z),
                            float(quat.x), float(quat.y), float(quat.z), float(quat.w),
                        )
                    )

    if not raw_t or not np.isfinite(raw_pos).any():
        raise SystemExit(
            f"No rigid body named {ball_name!r} on /rigid_bodies in {bag_path}"
        )

    raw_receive_t = np.asarray(raw_t, dtype=np.float64)
    order = np.argsort(raw_receive_t, kind="stable")
    raw_receive_t = raw_receive_t[order]
    raw_pos_w = np.asarray(raw_pos, dtype=np.float64)[order]
    raw_mean_error = np.asarray(raw_error, dtype=np.float64)[order]
    raw_tracking_valid = np.asarray(raw_valid, dtype=bool)[order]
    raw_source_t_s = np.asarray(raw_source_t, dtype=np.float64)[order]
    raw_pose_valid_array = np.asarray(raw_pose_valid, dtype=bool)[order]
    raw_pos_mocap_w = raw_pos_w.copy()

    # Both series share the raw clock so the filtered ghost lines up with the
    # measurement it came from; /ball/pose only appears once the estimator has
    # initialised, which is later than the first /rigid_bodies message.
    origin_s = float(raw_receive_t[0])

    # /rigid_bodies is the raw OptiTrack world frame, but /g1_pelvis/pose and
    # /ball/pose are published after the bridge's planar root calibration
    # (natnet_policy_bridge.py:261-262), which is a yaw about Z plus an XY shift.
    # The pelvis is recorded in both frames, so recover that transform from it
    # and move the raw ball into the same frame as the filtered series.
    calibration_segments: list[dict] = []
    if pelvis_raw_t and pelvis_cal_t:
        raw_pelvis_t = np.asarray(pelvis_raw_t, dtype=np.float64)
        raw_pelvis_pose = np.asarray(pelvis_raw_pose, dtype=np.float64)
        order = np.argsort(raw_pelvis_t, kind="stable")
        raw_pelvis_t = raw_pelvis_t[order]
        raw_pelvis_pose = raw_pelvis_pose[order]

        cal_t = np.asarray(pelvis_cal_t, dtype=np.float64)
        cal_pose = np.asarray(pelvis_cal_pose, dtype=np.float64)
        order = np.argsort(cal_t, kind="stable")
        cal_t = cal_t[order]
        cal_pose = cal_pose[order]

        raw_pos_w, calibration_segments = align_raw_positions_to_bridge_frame(
            raw_receive_t,
            raw_pos_w,
            raw_pelvis_t,
            raw_pelvis_pose,
            cal_t,
            cal_pose,
            np.asarray(calibration_requests, dtype=np.float64),
        )

    raw_t_s = raw_receive_t - origin_s
    raw_source_t_s -= origin_s

    filtered_t_s = np.asarray(filtered_t, dtype=np.float64)
    filtered_pos_w = np.asarray(filtered_pos, dtype=np.float64).reshape(-1, 3)
    if filtered_t_s.size:
        order = np.argsort(filtered_t_s, kind="stable")
        filtered_t_s = filtered_t_s[order] - origin_s
        filtered_pos_w = filtered_pos_w[order]

    velocity_t = np.asarray(filtered_velocity_t, dtype=np.float64)
    velocity_w = np.asarray(filtered_velocity, dtype=np.float64).reshape(-1, 3)
    velocity_order = np.argsort(velocity_t, kind="stable")
    velocity_t = velocity_t[velocity_order] - origin_s
    velocity_w = velocity_w[velocity_order]

    final_calibration = calibration_segments[-1] if calibration_segments else {
        "yaw_rad": 0.0,
        "translation": np.zeros(3, dtype=np.float64),
        "pairs": 0,
        "residual_m": float("nan"),
    }
    return {
        "source": str(bag_path),
        "ball_name": ball_name,
        "raw_t_s": raw_t_s,
        "raw_source_t_s": raw_source_t_s,
        "raw_pos_mocap_w": raw_pos_mocap_w,
        "raw_pos_w": raw_pos_w,
        "raw_pose_valid": raw_pose_valid_array,
        "raw_mean_error": raw_mean_error,
        "raw_tracking_valid": raw_tracking_valid,
        "filtered_t_s": filtered_t_s,
        "filtered_pos_w": filtered_pos_w,
        "filtered_velocity_t_s": velocity_t,
        "filtered_velocity_w": velocity_w,
        "calibration_yaw_rad": final_calibration["yaw_rad"],
        "calibration_translation_w": final_calibration["translation"],
        "calibration_pairs": sum(s["pairs"] for s in calibration_segments),
        "calibration_residual_m": final_calibration["residual_m"],
        "calibration_segment_starts_s": np.asarray(
            [s["start"] - origin_s for s in calibration_segments], dtype=np.float64
        ),
        "calibration_segment_yaws_rad": np.asarray(
            [s["yaw_rad"] for s in calibration_segments], dtype=np.float64
        ),
        "calibration_segment_translations_w": np.asarray(
            [s["translation"] for s in calibration_segments], dtype=np.float64
        ).reshape(-1, 3),
        "calibration_segment_pairs": np.asarray(
            [s["pairs"] for s in calibration_segments], dtype=np.int64
        ),
        "calibration_segment_residuals_m": np.asarray(
            [s["residual_m"] for s in calibration_segments], dtype=np.float64
        ),
    }


# --------------------------------------------------------------------------- #
# Offline measurement-first comparison
# --------------------------------------------------------------------------- #


def _direct_estimator_config(config: dict) -> BallEstimatorConfig:
    bridge = config.get("mocap_bridge", {})
    physics = config.get("tennis_ball_physics", {})
    aliases = {
        "tangent_speed_retention": "ground_tangent_speed_retention",
        "short_dropout_s": "ball_short_dropout_s",
        "maximum_prediction_s": "ball_maximum_prediction_s",
        "reassociation_timeout_s": "ball_reassociation_timeout_s",
        "static_speed_threshold_mps": "ball_static_speed_threshold_mps",
        "static_hold_s": "ball_static_hold_s",
        "maximum_speed_mps": "ball_maximum_speed_mps",
        "velocity_correction_gain": "ball_velocity_correction_gain",
        "integration_dt_s": "ball_integration_dt_s",
    }
    values = {}
    for field in fields(BallEstimatorConfig):
        key = aliases.get(field.name, field.name)
        if key in bridge:
            values[field.name] = bridge[key]
        elif field.name in physics:
            values[field.name] = physics[field.name]
    values["position_correction_gain"] = 1.0
    return BallEstimatorConfig(**values)


def _frame_transforms(dataset: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.asarray(dataset.get("calibration_segment_starts_s", []), dtype=float)
    yaws = np.asarray(dataset.get("calibration_segment_yaws_rad", []), dtype=float)
    shifts = np.asarray(dataset.get("calibration_segment_translations_w", []), dtype=float).reshape(-1, 3)
    if not starts.size:
        return np.array([-np.inf]), np.eye(3)[None], np.zeros((1, 3))
    if len(starts) != len(yaws) or len(starts) != len(shifts):
        raise ValueError("Calibration arrays have inconsistent lengths")
    return starts, Rotation.from_euler("z", yaws).as_matrix().reshape(-1, 3, 3), shifts


def simulate_measurement_first(
    dataset: dict,
    config: dict,
    *,
    timer_hz: float = 120.0,
    dropout_after_s: float = 0.0,
) -> dict:
    """Causally replay arrivals and a fallback timer, without ROS publishing.

    Reception time schedules callbacks; source time fits physical velocity.
    Valid measurements win timer ties. Normal output uses the exact measured
    position at its source timestamp, with no extrapolation to arrival time.
    """
    if not math.isfinite(timer_hz) or timer_hz <= 0:
        raise ValueError("timer_hz must be finite and positive")
    if not math.isfinite(dropout_after_s) or dropout_after_s < 0:
        raise ValueError("dropout_after_s must be finite and non-negative")
    times = np.asarray(dataset["raw_t_s"], dtype=float)
    source_times = np.asarray(dataset.get("raw_source_t_s", times), dtype=float)
    aligned_positions = np.asarray(dataset["raw_pos_w"], dtype=float)
    # A ball position is not a six-DoF rigid-body pose. Shape deformation can
    # increase mean_error (or spoil orientation) while tracking_valid remains
    # true. Do not silently replace these measured positions with predictions.
    valid = np.asarray(dataset["raw_tracking_valid"], dtype=bool).copy()
    valid &= np.isfinite(aligned_positions).all(axis=1) & np.isfinite(source_times)
    if not times.size or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Raw receive times must be nonempty, finite and sorted")

    starts, rotations, translations = _frame_transforms(dataset)

    def frame_at(t: float) -> tuple[np.ndarray, np.ndarray]:
        index = int(np.searchsorted(starts, t, side="right")) - 1
        if index < 0:
            return np.eye(3), np.zeros(3)
        return rotations[index], translations[index]

    if "raw_pos_mocap_w" in dataset:
        positions = np.asarray(dataset["raw_pos_mocap_w"], dtype=float)
    else:
        # Older exported NPZs only stored calibrated positions.
        positions = np.empty_like(aligned_positions)
        for i, t in enumerate(times):
            rotation, translation = frame_at(t)
            positions[i] = rotation.T @ (aligned_positions[i] - translation)

    stream = MeasurementFirstBallStream(_direct_estimator_config(config))
    estimator = stream.estimator
    rows: list[tuple] = []
    last_arrival = -np.inf
    last_publish = -np.inf
    last_index = -1
    period = 1.0 / timer_hz
    raw_index = 0
    old_times = np.asarray(dataset.get("filtered_t_s", []), dtype=float)
    end = max(float(times[-1]), float(old_times[-1]) if old_times.size else times[-1])

    def emit(t, position, velocity, state_time, mode, measured_index):
        nonlocal last_publish
        rotation, translation = frame_at(t)
        # Preserve exact raw coordinates for measurement output, including at
        # calibration boundaries. Only predicted positions need transforming.
        output_position = aligned_positions[measured_index].copy() if measured_index >= 0 else rotation @ position + translation
        rows.append((t, output_position, rotation @ velocity, mode,
                     max(0.0, t - source_times[last_index]), state_time,
                     estimator.track_epoch, measured_index))
        last_publish = t

    ticks = times[0] + np.arange(int(np.floor((end - times[0]) / period)) + 1) * period
    for tick in ticks:
        while raw_index < len(times) and times[raw_index] <= tick + 1.0e-9:
            i = raw_index
            raw_index += 1
            if not valid[i]:
                continue
            if estimator.last_valid_time_s is not None and source_times[i] <= estimator.last_valid_time_s:
                continue
            stream.observe(positions[i], float(source_times[i]))
            last_arrival = float(times[i])
            last_index = i
            emit(last_arrival, positions[i], estimator.velocity_w,
                 source_times[i], "measured", i)

        sample = stream.prediction_tick(float(tick))
        if sample is None or tick - last_arrival <= dropout_after_s:
            continue
        if tick - last_publish < period - 1.0e-9:
            continue
        mode = estimator.mode(float(tick))
        emit(float(tick), sample.position_w, sample.velocity_w, tick, mode, -1)

    # A final arrival can lie after the last regular tick; preserve that callback
    # without advancing the timer beyond the recorded interval.
    while raw_index < len(times):
        i = raw_index
        raw_index += 1
        if not valid[i] or (estimator.last_valid_time_s is not None and source_times[i] <= estimator.last_valid_time_s):
            continue
        stream.observe(positions[i], float(source_times[i]))
        last_index = i
        emit(times[i], positions[i], estimator.velocity_w, source_times[i], "measured", i)

    result = {
        "direct_t_s": np.array([r[0] for r in rows]),
        "direct_pos_w": np.array([r[1] for r in rows]).reshape(-1, 3),
        "direct_velocity_w": np.array([r[2] for r in rows]).reshape(-1, 3),
        "direct_mode": np.array([r[3] for r in rows], dtype="U20"),
        "direct_measurement_age_s": np.array([r[4] for r in rows]),
        "direct_state_t_s": np.array([r[5] for r in rows]),
        "direct_epoch": np.array([r[6] for r in rows], dtype=np.int64),
        "direct_raw_index": np.array([r[7] for r in rows], dtype=np.int64),
        "direct_timer_hz": timer_hz,
        "direct_dropout_after_s": dropout_after_s,
    }
    return result


def audit_measurement_first(dataset: dict) -> dict:
    """Audit every tracked position, including frames NOT selected for output.

    Compare the last published state at each raw arrival and over the combined
    event timeline. This catches dropped measured frames followed by prediction,
    which an equality check on selected direct_raw_index rows cannot detect.
    """
    times = np.asarray(dataset["raw_t_s"])
    positions = np.asarray(dataset["raw_pos_w"])
    available = np.asarray(dataset["raw_tracking_valid"], dtype=bool) & np.isfinite(positions).all(axis=1)
    output_times = np.asarray(dataset["direct_t_s"])
    ids = np.asarray(dataset["direct_raw_index"])
    forwarded = np.zeros(len(times), dtype=bool)
    forwarded[ids[ids >= 0]] = True
    result = {
        "tracked_position_frames": int(available.sum()),
        "unforwarded_tracked_frames": int((available & ~forwarded).sum()),
        "mismatched_tracked_frames": 0,
        "arrival_position_error_max_m": float("nan"),
        "fresh_timeline_mismatches": 0,
        "fresh_timeline_error_max_m": float("nan"),
    }
    if not len(output_times):
        return result
    rows = np.searchsorted(output_times, times, side="right") - 1
    compare = available & (rows >= 0)
    if compare.any():
        error = np.linalg.norm(positions[compare] - dataset["direct_pos_w"][rows[compare]], axis=1)
        result["mismatched_tracked_frames"] = int((error > 1e-9).sum())
        result["arrival_position_error_max_m"] = float(error.max())
    # All piecewise-constant state changes plus dense viewer-like probes. A raw
    # measurement is fresh until the configured missing-frame timeout, not forever.
    probes = np.unique(np.concatenate((times, output_times, np.arange(times[0], times[-1], 1 / 240))))
    raw_rows = np.searchsorted(times, probes, side="right") - 1
    out_rows = np.searchsorted(output_times, probes, side="right") - 1
    fresh = (raw_rows >= 0) & (out_rows >= 0)
    fresh_window = max(1.0 / dataset["direct_timer_hz"], dataset["direct_dropout_after_s"])
    fresh &= available[raw_rows] & (probes - times[raw_rows] <= fresh_window)
    if fresh.any():
        error = np.linalg.norm(positions[raw_rows[fresh]] - dataset["direct_pos_w"][out_rows[fresh]], axis=1)
        result["fresh_timeline_mismatches"] = int((error > 1e-9).sum())
        result["fresh_timeline_error_max_m"] = float(error.max())
    return result


def report_direct(dataset: dict) -> None:
    index = dataset["direct_raw_index"]
    measured = index >= 0
    delta = dataset["direct_pos_w"][measured] - dataset["raw_pos_w"][index[measured]]
    max_error = float(np.linalg.norm(delta, axis=1).max()) if len(delta) else float("nan")
    print(f"New scheme  : {int(measured.sum())} measured + {int((~measured).sum())} timer fill samples; "
          f"{dataset['direct_timer_hz']:g} Hz timer, {dataset['direct_dropout_after_s'] * 1000:g} ms dropout tolerance")
    print(f"Direct check: maximum measured-position difference {max_error:.9f} m")
    audit = audit_measurement_first(dataset)
    print(f"ALL tracked : {audit['tracked_position_frames']} position frames, "
          f"{audit['unforwarded_tracked_frames']} unforwarded, "
          f"{audit['mismatched_tracked_frames']} mismatched at arrival, "
          f"max error={audit['arrival_position_error_max_m']:.9f} m")
    print(f"Fresh state : {audit['fresh_timeline_mismatches']} timeline mismatches, "
          f"max error={audit['fresh_timeline_error_max_m']:.9f} m")
    if len(index):
        print(f"Initialized : t={dataset['direct_t_s'][0]:.3f}s; pre-bag estimator history is not available")
        modes, counts = np.unique(dataset["direct_mode"], return_counts=True)
        print("New modes   : " + ", ".join(f"{mode}={count}" for mode, count in zip(modes, counts)))
    print("Comparison  : orange=raw, blue=recorded Bridge, green=offline measurement-first; "
          "event timing follows bag receive timestamps, not a live ROS benchmark")


# --------------------------------------------------------------------------- #
# Playback clock
# --------------------------------------------------------------------------- #


class TimeController:
    """Playback clock shared by the viewer loop and the Tk scrubber.

    Locked because the Tk callbacks run on the UI thread while `advance()` is
    called from the MuJoCo loop.
    """

    def __init__(self, duration_s: float, speed: float) -> None:
        self.duration_s = max(float(duration_s), 1.0e-3)
        self._lock = threading.Lock()
        self._t = 0.0
        self._paused = False
        self._speed = float(speed)
        self._scrub: float | None = None
        self._quit = False

    @property
    def t(self) -> float:
        with self._lock:
            return self._t

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def speed(self) -> float:
        with self._lock:
            return self._speed

    @property
    def should_quit(self) -> bool:
        with self._lock:
            return self._quit

    def advance(self, wall_dt_s: float) -> float:
        with self._lock:
            if self._scrub is not None:
                self._t = self._scrub
                self._scrub = None
            elif not self._paused:
                self._t += wall_dt_s * self._speed
                if self._t >= self.duration_s:
                    self._t = 0.0
            self._t = min(max(self._t, 0.0), self.duration_s)
            return self._t

    def scrub_to(self, value: float) -> None:
        with self._lock:
            self._scrub = min(max(float(value), 0.0), self.duration_s)

    def nudge(self, delta_s: float) -> None:
        with self._lock:
            self._paused = True
            self._scrub = min(
                max(self._t + float(delta_s), 0.0), self.duration_s
            )

    def toggle_paused(self) -> None:
        with self._lock:
            self._paused = not self._paused

    def set_speed(self, value: float) -> None:
        with self._lock:
            self._speed = float(value)

    def restart(self) -> None:
        with self._lock:
            self._t = 0.0
            self._scrub = None

    def request_quit(self) -> None:
        with self._lock:
            self._quit = True


def launch_time_ui(
    controller: TimeController,
    frame_dt_s: float,
    replay: "TrajectoryReplay",
) -> threading.Thread | None:
    """Open the scrubber and return its thread so shutdown can join it."""
    try:
        import tkinter as tk
    except ImportError:
        print("tkinter is unavailable; scrub with the MuJoCo window keys instead.")
        return None

    def loop() -> None:
        root = tk.Tk()
        root.title("Ball trajectory time")
        root.resizable(False, False)

        value = tk.DoubleVar(value=0.0)
        dragging = [False]

        scale = tk.Scale(
            root,
            from_=0.0,
            to=controller.duration_s,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            length=620,
            label="t (s)",
            variable=value,
            command=lambda raw: controller.scrub_to(float(raw)),
        )
        scale.pack(padx=12, pady=(8, 2))
        scale.bind("<ButtonPress-1>", lambda _e: dragging.__setitem__(0, True))
        scale.bind("<ButtonRelease-1>", lambda _e: dragging.__setitem__(0, False))

        play_label = tk.StringVar(value="Pause")

        def toggle() -> None:
            controller.toggle_paused()
            play_label.set("Play" if controller.paused else "Pause")

        row = tk.Frame(root)
        row.pack(pady=(0, 4))
        for text, command in (
            ("<< -1s", lambda: controller.nudge(-1.0)),
            ("-0.1s", lambda: controller.nudge(-0.1)),
            ("< frame", lambda: controller.nudge(-frame_dt_s)),
            (None, None),
            ("Pause", toggle),
            (None, None),
            ("frame >", lambda: controller.nudge(frame_dt_s)),
            ("+0.1s", lambda: controller.nudge(0.1)),
            ("+1s >>", lambda: controller.nudge(1.0)),
        ):
            if text is None:
                tk.Label(row, text="  ").pack(side=tk.LEFT)
                continue
            if text == "Pause":
                tk.Button(row, textvariable=play_label, width=8,
                          command=command).pack(side=tk.LEFT, padx=2)
            else:
                tk.Button(row, text=text, width=8,
                          command=command).pack(side=tk.LEFT, padx=2)

        speed_row = tk.Frame(root)
        speed_row.pack(pady=(0, 4))
        tk.Label(speed_row, text="Speed  ").pack(side=tk.LEFT)
        speed_var = tk.StringVar(value=f"{controller.speed:g}")
        for choice in SPEED_CHOICES:
            tk.Radiobutton(
                speed_row,
                text=f"{choice}x",
                value=choice,
                variable=speed_var,
                command=lambda v=choice: controller.set_speed(float(v)),
            ).pack(side=tk.LEFT)

        stream_controls = tk.Frame(root)
        stream_controls.pack(pady=(0, 6))
        variables = []
        for row_index, (label, color, ball_attr, trail_attr, available) in enumerate((
            ("Raw mocap", "#b35500", "show_raw", "show_raw_trail", True),
            ("Recorded Bridge", "#146db5", "show_filtered", "show_bridge_trail", replay.has_filtered),
            ("Measurement-first", "#158139", "show_direct", "show_direct_trail", replay.has_direct),
        )):
            tk.Label(stream_controls, text=label, fg=color, width=22, anchor="w").grid(row=row_index, column=0)
            for column, (title, attribute) in enumerate((("Ball", ball_attr), ("Trail", trail_attr)), start=1):
                variable = tk.BooleanVar(value=getattr(replay, attribute))
                variables.append(variable)
                tk.Checkbutton(
                    stream_controls, text=title, variable=variable,
                    state=tk.NORMAL if available else tk.DISABLED,
                    command=lambda v=variable, a=attribute: setattr(replay, a, bool(v.get())),
                ).grid(row=row_index, column=column)

        status = tk.Label(root, text="", anchor="center")
        status.pack(pady=(0, 8))

        def sync() -> None:
            if controller.should_quit:
                root.destroy()
                return
            if not dragging[0]:
                value.set(round(controller.t, 3))
            status.config(
                text=f"t = {controller.t:6.2f} / {controller.duration_s:.2f} s"
                     f"{'   [PAUSED]' if controller.paused else ''}"
                     f"\n{replay.direct_status(controller.t)}"
            )
            root.after(50, sync)

        root.protocol("WM_DELETE_WINDOW", root.destroy)
        root.after(50, sync)
        root.mainloop()

    def ui_worker() -> None:
        import gc

        try:
            loop()
        finally:
            # Tk callback closures can form cycles. Finalize their Variables on
            # the Tk owner thread, not during main-thread interpreter shutdown.
            gc.collect()

    thread = threading.Thread(target=ui_worker, daemon=True)
    thread.start()
    return thread


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


def add_sphere(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    radius: float,
    color: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        IDENTITY,
        color,
    )
    scene.ngeom += 1


def trail_points(
    times: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    timestamp: float,
    duration_s: float,
) -> np.ndarray:
    """Show actual samples in the history window, with no interpolated links."""
    left = int(np.searchsorted(times, timestamp - duration_s))
    right = int(np.searchsorted(times, timestamp, side="right"))
    selected_values = values[left:right, :3]
    selected_valid = valid[left:right] & np.isfinite(selected_values).all(axis=1)
    indices = np.flatnonzero(selected_valid)
    if indices.size > 400:
        indices = indices[
            np.linspace(0, indices.size - 1, 400, dtype=int)
        ]
    return selected_values[indices]

def build_model(config: dict) -> mujoco.MjModel:
    simulation = config["simulation"]
    court = simulation.get("court", {})
    model, _sample = build_training_aligned_model(
        robot_xml_path=repo_path(config["simulation_xml_path"]),
        court_xml_path=repo_path(config["tennis_court_xml_path"]),
        joint_names=tuple(config["joint_names"].split(",")),
        physics_dt=float(simulation["physics_dt"]),
        nominal_ball_physics=config["tennis_ball_physics"],
        randomization_config=simulation.get("physics_domain_randomization", {}),
        court_net_x_m=float(court.get("net_x_m", 5.6)),
        court_net_half_width_m=float(court.get("net_half_width_m", 5.485)),
    )
    return model


def set_home_pose(model: mujoco.MjModel, data: mujoco.MjData, config: dict) -> None:
    joint_names = tuple(config["joint_names"].split(","))
    for name, value in zip(joint_names, config.get("default_joint_pos", [])):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)


class TrajectoryReplay:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dataset: dict,
        *,
        ball_radius_m: float,
        speed: float,
        trail_seconds: float,
        show_filtered: bool,
        show_direct: bool = True,
    ) -> None:
        self.model = model
        self.data = data
        self.ball_radius_m = float(ball_radius_m)
        self.trail_seconds = float(trail_seconds)
        self.show_filtered = show_filtered
        self.show_raw = True
        self.show_direct = show_direct
        self.show_raw_trail = True

        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BALL_JOINT)
        if joint_id < 0:
            raise ValueError(f"Model has no {BALL_JOINT} joint")
        self.qpos_address = int(model.jnt_qposadr[joint_id])
        self.qvel_address = int(model.jnt_dofadr[joint_id])

        mask = np.asarray(dataset["raw_tracking_valid"], dtype=bool).copy()
        mask &= np.isfinite(dataset["raw_pos_w"]).all(axis=1)
        self.raw_t_s = np.asarray(dataset["raw_t_s"], dtype=np.float64)
        self.raw_pos_w = np.asarray(dataset["raw_pos_w"], dtype=np.float64)
        self.raw_valid = mask
        self.n_dropped = int(self.raw_t_s.size - int(mask.sum()))
        if np.count_nonzero(mask) < 2:
            raise ValueError("Trajectory needs at least two valid samples")

        self.filtered_t_s = np.asarray(dataset["filtered_t_s"], dtype=np.float64)
        self.filtered_pos_w = np.asarray(dataset["filtered_pos_w"], dtype=np.float64)
        self.filtered_valid = np.isfinite(self.filtered_pos_w).all(axis=1)
        self.has_filtered = (
            np.count_nonzero(self.filtered_valid) >= 2
            and self.filtered_pos_w.shape[0] == self.filtered_t_s.size
        )
        self.show_bridge_trail = self.has_filtered and show_filtered
        self.filtered_velocity_t_s = np.asarray(dataset.get("filtered_velocity_t_s", []))
        self.filtered_velocity_w = np.asarray(dataset.get("filtered_velocity_w", [])).reshape(-1, 3)
        self.direct_t_s = np.asarray(dataset.get("direct_t_s", []))
        self.direct_pos_w = np.asarray(dataset.get("direct_pos_w", [])).reshape(-1, 3)
        self.direct_velocity_w = np.asarray(dataset.get("direct_velocity_w", [])).reshape(-1, 3)
        self.direct_mode = np.asarray(dataset.get("direct_mode", []))
        self.direct_age_s = np.asarray(dataset.get("direct_measurement_age_s", []))
        self.direct_valid = np.isfinite(self.direct_pos_w).all(axis=1)
        self.has_direct = bool(len(self.direct_t_s))
        self.show_direct_trail = self.has_direct and show_direct

        self.duration_s = float(
            max(
                self.raw_t_s[-1],
                self.filtered_t_s[-1] if self.filtered_t_s.size else 0.0,
                self.direct_t_s[-1] if self.has_direct else 0.0,
            )
        )
        valid_raw_times = self.raw_t_s[self.raw_valid]
        self.sample_dt_s = float(np.median(np.diff(valid_raw_times)))
        self.controller = TimeController(self.duration_s, speed)

        ball_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "tennis_ball"
        )
        if ball_body_id >= 0:
            ball_geoms = np.flatnonzero(model.geom_bodyid == ball_body_id)
            model.geom_rgba[ball_geoms, 3] = 0.0

    # -- sampling ----------------------------------------------------------- #

    @staticmethod
    def _sample_fresh(
        times: np.ndarray,
        values: np.ndarray,
        valid: np.ndarray,
        t: float,
        max_age_s: float,
    ) -> np.ndarray | None:
        index = int(np.searchsorted(times, t, side="right")) - 1
        if index < 0 or not valid[index] or t - times[index] > max_age_s:
            return None
        return values[index].copy()

    def sample_raw(self, t: float) -> np.ndarray | None:
        return self._sample_fresh(
            self.raw_t_s,
            self.raw_pos_w,
            self.raw_valid,
            t,
            max(0.025, 2.5 * self.sample_dt_s),
        )

    def sample_filtered(self, t: float) -> np.ndarray | None:
        return self._sample_fresh(
            self.filtered_t_s,
            self.filtered_pos_w,
            self.filtered_valid,
            t,
            0.05,
        )

    def sample_direct(self, t: float) -> np.ndarray | None:
        return self._sample_fresh(
            self.direct_t_s, self.direct_pos_w, self.direct_valid, t, 0.05,
        )

    def direct_status(self, t: float) -> str:
        index = int(np.searchsorted(self.direct_t_s, t, side="right")) - 1
        if index < 0:
            return "New scheme: waiting for first valid measurement"
        velocity = self.direct_velocity_w[index]
        age_ms = 1000.0 * (self.direct_age_s[index] + t - self.direct_t_s[index])
        status = (f"New: {self.direct_mode[index]} | last measurement age {age_ms:.1f} ms\n"
                  f"New velocity W: [{velocity[0]:+.2f}, {velocity[1]:+.2f}, {velocity[2]:+.2f}] m/s")
        old_velocity = self._sample_fresh(
            self.filtered_velocity_t_s, self.filtered_velocity_w,
            np.isfinite(self.filtered_velocity_w).all(axis=1), t, 0.05,
        )
        if old_velocity is not None:
            status += f"\nOld velocity W: [{old_velocity[0]:+.2f}, {old_velocity[1]:+.2f}, {old_velocity[2]:+.2f}] m/s"
        return status

    def place_ball(self, position: np.ndarray) -> None:
        self.data.qpos[self.qpos_address : self.qpos_address + 3] = position
        self.data.qpos[self.qpos_address + 3 : self.qpos_address + 7] = (
            1.0, 0.0, 0.0, 0.0
        )
        self.data.qvel[self.qvel_address : self.qvel_address + 6] = 0.0

    # -- rendering ---------------------------------------------------------- #

    def key_callback(self, keycode: int) -> None:
        if keycode == ord(" "):
            self.controller.toggle_paused()
        elif keycode in (ord("R"), ord("r")):
            self.controller.restart()
        elif keycode in (ord("Q"), ord("q")):
            self.controller.request_quit()

    def draw(self, scene: mujoco.MjvScene, t: float, *, clear: bool = True) -> None:
        if clear:
            scene.ngeom = 0
        for show, times, positions, valid, color in (
            (self.show_raw_trail, self.raw_t_s, self.raw_pos_w, self.raw_valid, RAW_COLOR),
            (self.has_filtered and self.show_bridge_trail, self.filtered_t_s, self.filtered_pos_w, self.filtered_valid, BRIDGE_COLOR),
            (self.has_direct and self.show_direct_trail, self.direct_t_s, self.direct_pos_w, self.direct_valid, DIRECT_COLOR),
        ):
            if show:
                for point in trail_points(times, positions, valid, t, self.trail_seconds):
                    add_sphere(scene, point, TRAIL_POINT_RADIUS_M, color)
        raw_position = self.sample_raw(t)
        if self.show_raw and raw_position is not None:
            add_sphere(scene, raw_position, self.ball_radius_m, RAW_COLOR)
        if self.has_filtered:
            bridge_position = self.sample_filtered(t)
            if self.show_filtered and bridge_position is not None:
                add_sphere(
                    scene, bridge_position, self.ball_radius_m, BRIDGE_COLOR
                )
        if self.has_direct:
            direct_position = self.sample_direct(t)
            if self.show_direct and direct_position is not None:
                add_sphere(scene, direct_position, self.ball_radius_m, DIRECT_COLOR)

    def run(self, *, with_ui: bool) -> None:
        ui_thread = launch_time_ui(self.controller, self.sample_dt_s, self) if with_ui else None
        try:
            self._run_viewer()
        finally:
            self.controller.request_quit()
            if ui_thread is not None:
                ui_thread.join(timeout=2.0)

    def _run_viewer(self) -> None:
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.cam.lookat[:] = (1.5, 0.0, 0.7)
            viewer.cam.distance = 7.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0

            wall_last = time.monotonic()
            while viewer.is_running():
                if self.controller.should_quit:
                    break

                wall_now = time.monotonic()
                t = self.controller.advance(wall_now - wall_last)
                wall_last = wall_now

                position = self.sample_raw(t)
                self.place_ball(
                    position if position is not None else np.array([0.0, 0.0, -10.0])
                )
                mujoco.mj_forward(self.model, self.data)
                with viewer.lock():
                    self.draw(viewer.user_scn, t)
                position_text = (
                    f"x={position[0]:+.2f} y={position[1]:+.2f} z={position[2]:+.2f} m"
                    if position is not None
                    else "not tracking"
                )
                viewer.set_texts(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        (
                            f"t = {t:6.2f} / {self.duration_s:.2f} s"
                            f"   x{self.controller.speed:g}"
                            f"{'   [PAUSED]' if self.controller.paused else ''}\n"
                            f"raw ball: {position_text}\n"
                            f"{self.direct_status(t)}\n"
                            "Space: play/pause   R: restart   Q: quit"
                        ),
                        (
                            f"Raw mocap: orange ({self.raw_t_s.size} samples, "
                            f"{self.n_dropped} dropped)\n"
                            + (
                                f"Bridge /ball/pose: blue ({self.filtered_t_s.size} samples)\n"
                                if self.has_filtered
                                else ""
                            )
                            + "New measurement-first: green\n"
                            + "Measured green/orange positions coincide at each raw arrival"
                        ),
                    )
                )
                viewer.sync()
                time.sleep(0.002)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the mocap ball trajectory from a bag and replay it."
    )
    parser.add_argument(
        "bag",
        nargs="?",
        type=Path,
        help="ros2 bag directory (defaults to the newest recording).",
    )
    parser.add_argument("--config", default="g1_tppo_student_m14_9.yaml")
    parser.add_argument(
        "--npz",
        type=Path,
        help="Input (--replay-only) or output .npz path. "
        "Default: <bag>/ball_trajectory.npz",
    )
    parser.add_argument("--mocap-msg-dir", type=Path)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--direct-timer-hz", type=float, default=120.0,
                        help="Offline prediction fill timer frequency (Hz).")
    parser.add_argument("--direct-dropout-ms", type=float, default=0.0,
                        help="Optional extra dropout gate for offline comparisons; 0 matches deploy's per-tick selection.")
    parser.add_argument("--show-direct", action=argparse.BooleanOptionalAction,
                        default=True, help="Draw the offline measurement-first comparison.")
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=2.0,
        help="History window for trajectory scatter points in seconds (default: 2).",
    )
    parser.add_argument(
        "--show-filtered",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the Bridge /ball/pose stream (default: enabled).",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Do not open the Tk scrubber window (use the viewer keys).",
    )
    return parser.parse_args()


def report_separation(dataset: dict) -> None:
    """Print raw vs filtered ball positions side by side.

    This is the frame-mismatch smoke test: once both series share the bridge's
    calibrated frame the separation should be centimetres, not metres.
    """
    raw_t = np.asarray(dataset["raw_t_s"], dtype=np.float64)
    raw_p = np.asarray(dataset["raw_pos_w"], dtype=np.float64)
    valid = np.asarray(dataset["raw_tracking_valid"], dtype=bool).copy()
    valid &= np.isfinite(raw_p).all(axis=1)
    raw_t, raw_p = raw_t[valid], raw_p[valid]
    fil_t = np.asarray(dataset["filtered_t_s"], dtype=np.float64)
    fil_p = np.asarray(dataset["filtered_pos_w"], dtype=np.float64).reshape(-1, 3)
    if raw_t.size < 2 or fil_t.size < 2:
        return

    low = max(float(raw_t[0]), float(fil_t[0]))
    high = min(float(raw_t[-1]), float(fil_t[-1]))
    if high <= low:
        print("Separation : raw and filtered time ranges do not overlap")
        return

    query_ids, raw_ids = _match_by_time(fil_t, raw_t, 0.025)
    if query_ids.size == 0:
        print("Separation : no raw/Bridge samples are within 25 ms")
        return
    raw_at = raw_p[raw_ids]
    filtered_at = fil_p[query_ids]
    gap = np.linalg.norm(raw_at - filtered_at, axis=1)
    print(
        f"Separation : mean {gap.mean():.3f} m, median {np.median(gap):.3f} m, "
        f"max {gap.max():.3f} m"
    )
    for t, raw, filtered, distance in list(
        zip(fil_t[query_ids], raw_at, filtered_at, gap)
    )[:5]:
        print(
            f"   t={t:6.2f}s  raw=({raw[0]:+7.2f},{raw[1]:+7.2f},{raw[2]:+7.2f})"
            f"  filt=({filtered[0]:+7.2f},{filtered[1]:+7.2f},{filtered[2]:+7.2f})"
            f"  d={distance:6.3f} m"
        )
    if gap.mean() > 1.0:
        print(
            "   WARNING: raw and filtered are metres apart; the frame calibration\n"
            "   was probably not recovered (check the pelvis pair count above)."
        )


def newest_bag() -> Path:
    candidates = sorted(
        (path for path in (DEPLOY_DIR / "recordings").iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if not candidates:
        raise SystemExit("No recordings found under deploy/recordings")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.replay_only:
        if args.npz is None:
            raise SystemExit("--replay-only requires --npz")
        dataset = dict(np.load(args.npz, allow_pickle=True))
        dataset.update(simulate_measurement_first(
            dataset, config, timer_hz=args.direct_timer_hz,
            dropout_after_s=args.direct_dropout_ms * 0.001,
        ))
    else:
        bag_path = (args.bag or newest_bag()).expanduser().resolve()
        if not bag_path.is_dir():
            raise SystemExit(f"Not a ros2 bag directory: {bag_path}")
        dataset = extract(bag_path, config, args.mocap_msg_dir)
        dataset.update(simulate_measurement_first(
            dataset, config, timer_hz=args.direct_timer_hz,
            dropout_after_s=args.direct_dropout_ms * 0.001,
        ))

        out_path = (args.npz or (bag_path / "ball_trajectory.npz")).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **dataset)
        valid = int(np.count_nonzero(dataset["raw_tracking_valid"]))
        print(f"Bag         : {dataset['source']}")
        print(f"Ball body   : {dataset['ball_name']}")
        print(
            f"Raw samples : {dataset['raw_t_s'].size} "
            f"({valid} tracking_valid, {dataset['raw_t_s'].size - valid} dropped)"
        )
        print(f"Filtered    : {dataset['filtered_t_s'].size} samples on /ball/pose")
        print(
            f"Span        : {dataset['raw_t_s'][0]:.2f} s -> "
            f"{dataset['raw_t_s'][-1]:.2f} s"
        )
        pairs = int(dataset["calibration_pairs"])
        if pairs:
            print(
                f"Calibration : yaw={math.degrees(dataset['calibration_yaw_rad']):+7.2f} deg, "
                f"t=[{dataset['calibration_translation_w'][0]:+.3f}, "
                f"{dataset['calibration_translation_w'][1]:+.3f}, "
                f"{dataset['calibration_translation_w'][2]:+.3f}] m, "
                f"{pairs} pelvis pairs, "
                f"residual {dataset['calibration_residual_m']:.4f} m"
            )
        else:
            print(
                "Calibration : NOT recovered - no usable G1_pelvis pairs, so raw "
                "stays in the mocap frame; alignment with /ball/pose is unverified"
            )
        print(f"Wrote       : {out_path}")
        report_separation(dataset)
        if args.extract_only:
            report_direct(dataset)
            return

    report_direct(dataset)
    model = build_model(config)
    data = mujoco.MjData(model)
    set_home_pose(model, data, config)
    if dataset["raw_t_s"].size == 0:
        raise SystemExit("Dataset has no ball samples")

    TrajectoryReplay(
        model,
        data,
        dataset,
        ball_radius_m=float(config["tennis_ball_physics"]["radius_m"]),
        speed=args.speed,
        trail_seconds=args.trail_seconds,
        show_filtered=args.show_filtered,
        show_direct=args.show_direct,
    ).run(with_ui=not args.no_ui)


if __name__ == "__main__":
    main()
