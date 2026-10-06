#!/usr/bin/env python3
"""Collect static Mocap/FK correspondences for Pelvis-frame calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from mocap_msgs.msg import RigidBodies
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


DEPLOY_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEPLOY_DIR))

from utils.calibration_fk import ACTION_DIM, SweetSpotForwardKinematics, parse_joint_names
from utils.mocap_frame_calibration import quaternion_xyzw_to_matrix
from utils.tppo_control_node import load_deploy_config, resolve_deploy_path


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def pose_to_arrays(pose) -> tuple[np.ndarray, np.ndarray] | None:
    position = np.array(
        [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
    )
    quaternion = np.array(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        return None
    if norm < 1.0e-8:
        return None
    return position, quaternion / norm


class CalibrationCollector(Node):
    """Synchronize latest static robot and Mocap states at a fixed sample rate."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        sample_rate_hz: float,
        freshness_s: float,
    ) -> None:
        super().__init__("collect_fk_calibration_data")
        self.config, self.config_path = load_deploy_config(config_path)
        bridge = self.config.get("mocap_bridge", {})
        if not isinstance(bridge, dict):
            raise TypeError("mocap_bridge must be a mapping")

        self.raw_topic = str(bridge.get("input_topic", "/rigid_bodies"))
        self.pelvis_name = str(bridge.get("pelvis_name", "G1_pelvis"))
        self.ball_name = str(bridge.get("ball_name", "Ball"))
        self.pelvis_max_mean_error_m = float(
            bridge.get("pelvis_max_mean_error_m", 0.005)
        )
        self.ball_max_mean_error_m = float(
            bridge.get("ball_max_mean_error_m", 0.010)
        )
        self.sample_rate_hz = float(sample_rate_hz)
        self.freshness_s = float(freshness_s)
        if self.sample_rate_hz <= 0.0 or self.freshness_s <= 0.0:
            raise ValueError("sample_rate_hz and freshness_s must be positive")

        joint_names = parse_joint_names(self.config["joint_names"])
        kinematics_xml = resolve_deploy_path(
            self.config["kinematics_xml_path"],
            config_dir=self.config_path.parent,
        )
        self.site_name = str(self.config.get("sweet_spot_site", "racket_sweet_spot"))
        self.sweet_spot_position_offset_b = np.asarray(
            self.config.get("sweet_spot_position_offset_b", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        self.fk = SweetSpotForwardKinematics(
            kinematics_xml,
            joint_names,
            self.site_name,
            position_offset_b=self.sweet_spot_position_offset_b,
        )
        ball_physics = self.config.get("tennis_ball_physics", {})
        self.ball_radius_m = float(ball_physics.get("radius_m", 0.0335))

        self._lock = threading.Lock()
        self._latest: dict[str, dict] = {}
        self._active_pose_index: int | None = None
        self._active_pose_label = ""
        self._active_records: list[dict] = []
        self._skipped_samples = 0

        self.create_subscription(
            RigidBodies, self.raw_topic, self._raw_mocap_callback, 10
        )
        self.create_subscription(
            PoseStamped, "/g1_pelvis/pose", self._calibrated_pelvis_callback, 10
        )
        self.create_subscription(
            PoseStamped, "/ball/pose", self._calibrated_ball_callback, 10
        )
        self.create_subscription(
            Float32MultiArray,
            "/deploy_robot/joint_state",
            self._joint_state_callback,
            10,
        )
        self.create_timer(1.0 / self.sample_rate_hz, self._sample_callback)
        self.get_logger().info(
            "FK calibration collector ready: "
            f"raw={self.raw_topic} rate={self.sample_rate_hz:.1f}Hz "
            f"freshness={1000.0 * self.freshness_s:.0f}ms "
            f"xml={kinematics_xml} site={self.site_name}"
        )

    def _raw_mocap_callback(self, message: RigidBodies) -> None:
        bodies = {body.rigid_body_name: body for body in message.rigidbodies}
        pelvis = bodies.get(self.pelvis_name)
        ball = bodies.get(self.ball_name)
        if pelvis is None or ball is None:
            return
        pelvis_pose = pose_to_arrays(pelvis.pose)
        ball_pose = pose_to_arrays(ball.pose)
        if pelvis_pose is None or ball_pose is None:
            return
        now = time.monotonic()
        with self._lock:
            self._latest["raw"] = {
                "arrival_monotonic_s": now,
                "source_stamp_s": stamp_to_seconds(message.header.stamp),
                "frame_number": int(message.frame_number),
                "pelvis_position_w": pelvis_pose[0],
                "pelvis_quaternion_xyzw": pelvis_pose[1],
                "pelvis_tracking_valid": bool(pelvis.tracking_valid),
                "pelvis_mean_error_m": float(pelvis.mean_error),
                "ball_position_w": ball_pose[0],
                "ball_quaternion_xyzw": ball_pose[1],
                "ball_tracking_valid": bool(ball.tracking_valid),
                "ball_mean_error_m": float(ball.mean_error),
            }

    def _calibrated_pelvis_callback(self, message: PoseStamped) -> None:
        pose = pose_to_arrays(message.pose)
        if pose is None:
            return
        with self._lock:
            self._latest["calibrated_pelvis"] = {
                "arrival_monotonic_s": time.monotonic(),
                "source_stamp_s": stamp_to_seconds(message.header.stamp),
                "position_w": pose[0],
                "quaternion_xyzw": pose[1],
            }

    def _calibrated_ball_callback(self, message: PoseStamped) -> None:
        pose = pose_to_arrays(message.pose)
        if pose is None:
            return
        with self._lock:
            self._latest["calibrated_ball"] = {
                "arrival_monotonic_s": time.monotonic(),
                "source_stamp_s": stamp_to_seconds(message.header.stamp),
                "position_w": pose[0],
                "quaternion_xyzw": pose[1],
            }

    def _joint_state_callback(self, message: Float32MultiArray) -> None:
        values = np.asarray(message.data, dtype=np.float64)
        if values.shape[0] < 2 * ACTION_DIM or not np.isfinite(values).all():
            return
        with self._lock:
            self._latest["joint_state"] = {
                "arrival_monotonic_s": time.monotonic(),
                "position": values[:ACTION_DIM].copy(),
                "velocity": values[ACTION_DIM : 2 * ACTION_DIM].copy(),
            }

    def readiness(self) -> list[str]:
        now = time.monotonic()
        with self._lock:
            latest = dict(self._latest)
        missing: list[str] = []
        for name in ("raw", "calibrated_pelvis", "calibrated_ball", "joint_state"):
            state = latest.get(name)
            if state is None:
                missing.append(name)
            elif now - float(state["arrival_monotonic_s"]) > self.freshness_s:
                missing.append(f"{name}(stale)")
        raw = latest.get("raw")
        if raw is not None:
            if not raw["pelvis_tracking_valid"]:
                missing.append("pelvis_tracking_invalid")
            if (
                not math.isfinite(raw["pelvis_mean_error_m"])
                or raw["pelvis_mean_error_m"] > self.pelvis_max_mean_error_m
            ):
                missing.append("pelvis_mean_error")
            if not raw["ball_tracking_valid"]:
                missing.append("ball_tracking_invalid")
            if (
                not math.isfinite(raw["ball_mean_error_m"])
                or raw["ball_mean_error_m"] > self.ball_max_mean_error_m
            ):
                missing.append("ball_mean_error")
        return missing

    def start_capture(self, pose_index: int, pose_label: str) -> None:
        with self._lock:
            if self._active_pose_index is not None:
                raise RuntimeError("A pose capture is already active")
            self._active_pose_index = int(pose_index)
            self._active_pose_label = pose_label
            self._active_records = []
            self._skipped_samples = 0

    def stop_capture(self) -> tuple[list[dict], int]:
        with self._lock:
            records = self._active_records
            skipped = self._skipped_samples
            self._active_pose_index = None
            self._active_pose_label = ""
            self._active_records = []
            self._skipped_samples = 0
        return records, skipped

    def pop_latest_capture_record(self) -> dict | None:
        """Return only the newest sample so live monitoring cannot accumulate data."""
        with self._lock:
            if not self._active_records:
                return None
            record = self._active_records[-1]
            self._active_records.clear()
        return record

    def _sample_callback(self) -> None:
        with self._lock:
            if self._active_pose_index is None:
                return
            pose_index = self._active_pose_index
            pose_label = self._active_pose_label
            latest = dict(self._latest)

        if self.readiness():
            with self._lock:
                self._skipped_samples += 1
            return

        raw = latest["raw"]
        pelvis = latest["calibrated_pelvis"]
        ball = latest["calibrated_ball"]
        joints = latest["joint_state"]

        root_position_w = pelvis["position_w"]
        root_quaternion_xyzw = pelvis["quaternion_xyzw"]
        rotation_wb = quaternion_xyzw_to_matrix(root_quaternion_xyzw)
        rotation_bw = rotation_wb.T
        ball_position_w = ball["position_w"]
        ball_position_b = rotation_bw @ (ball_position_w - root_position_w)

        sweet_position_w, sweet_rotation_w = self.fk.evaluate(
            root_position_w,
            root_quaternion_xyzw,
            joints["position"],
        )
        sweet_position_b = rotation_bw @ (sweet_position_w - root_position_w)
        sweet_rotation_b = rotation_bw @ sweet_rotation_w
        racket_normal_b = sweet_rotation_b[:, 2]
        expected_ball_plus_b = (
            sweet_position_b + self.ball_radius_m * racket_normal_b
        )
        expected_ball_minus_b = (
            sweet_position_b - self.ball_radius_m * racket_normal_b
        )

        raw_rotation_wb = quaternion_xyzw_to_matrix(
            raw["pelvis_quaternion_xyzw"]
        )
        raw_ball_position_b = raw_rotation_wb.T @ (
            raw["ball_position_w"] - raw["pelvis_position_w"]
        )

        record = {
            "pose_index": pose_index,
            "pose_label": pose_label,
            "capture_monotonic_s": time.monotonic(),
            "raw_frame_number": raw["frame_number"],
            "raw_source_stamp_s": raw["source_stamp_s"],
            "raw_pelvis_position_w": raw["pelvis_position_w"].copy(),
            "raw_pelvis_quaternion_xyzw": raw[
                "pelvis_quaternion_xyzw"
            ].copy(),
            "raw_pelvis_mean_error_m": raw["pelvis_mean_error_m"],
            "raw_pelvis_tracking_valid": raw["pelvis_tracking_valid"],
            "raw_ball_position_w": raw["ball_position_w"].copy(),
            "raw_ball_quaternion_xyzw": raw["ball_quaternion_xyzw"].copy(),
            "raw_ball_mean_error_m": raw["ball_mean_error_m"],
            "raw_ball_tracking_valid": raw["ball_tracking_valid"],
            "raw_ball_position_b": raw_ball_position_b,
            "calibrated_pelvis_source_stamp_s": pelvis["source_stamp_s"],
            "calibrated_pelvis_position_w": root_position_w.copy(),
            "calibrated_pelvis_quaternion_xyzw": root_quaternion_xyzw.copy(),
            "calibrated_ball_source_stamp_s": ball["source_stamp_s"],
            "calibrated_ball_position_w": ball_position_w.copy(),
            "calibrated_ball_quaternion_xyzw": ball[
                "quaternion_xyzw"
            ].copy(),
            "joint_position": joints["position"].copy(),
            "joint_velocity": joints["velocity"].copy(),
            "ball_position_b": ball_position_b,
            "sweet_spot_position_w": sweet_position_w,
            "sweet_spot_position_b": sweet_position_b,
            "sweet_spot_rotation_b": sweet_rotation_b.reshape(-1),
            "racket_normal_b": racket_normal_b,
            "expected_ball_plus_b": expected_ball_plus_b,
            "expected_ball_minus_b": expected_ball_minus_b,
            "direct_error_b": ball_position_b - sweet_position_b,
            "contact_plus_error_b": ball_position_b - expected_ball_plus_b,
            "contact_minus_error_b": ball_position_b - expected_ball_minus_b,
        }
        with self._lock:
            if self._active_pose_index == pose_index:
                self._active_records.append(record)


def quaternion_mean_xyzw(values: np.ndarray) -> np.ndarray:
    reference = values[0]
    aligned = values.copy()
    signs = np.sign(aligned @ reference)
    signs[signs == 0.0] = 1.0
    aligned *= signs[:, None]
    mean = aligned.mean(axis=0)
    return mean / np.linalg.norm(mean)


def vector_stats(records: list[dict], key: str) -> dict[str, list[float]]:
    values = np.stack([np.asarray(record[key], dtype=np.float64) for record in records])
    return {
        "mean": values.mean(axis=0).round(8).tolist(),
        "std": values.std(axis=0).round(8).tolist(),
    }


def norm_stats(records: list[dict], key: str) -> dict[str, float]:
    values = np.stack([np.asarray(record[key], dtype=np.float64) for record in records])
    norms = np.linalg.norm(values, axis=1)
    return {
        "mean": round(float(norms.mean()), 8),
        "std": round(float(norms.std()), 8),
        "min": round(float(norms.min()), 8),
        "max": round(float(norms.max()), 8),
    }


def summarize_pose(records: list[dict], skipped_samples: int) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty pose")
    plus = norm_stats(records, "contact_plus_error_b")
    minus = norm_stats(records, "contact_minus_error_b")
    plus_vector = vector_stats(records, "contact_plus_error_b")
    minus_vector = vector_stats(records, "contact_minus_error_b")
    best_contact_side = "+normal" if plus["mean"] <= minus["mean"] else "-normal"
    best_contact_norm = plus if best_contact_side == "+normal" else minus
    best_contact_vector = (
        plus_vector if best_contact_side == "+normal" else minus_vector
    )
    pelvis_quaternions = np.stack(
        [record["calibrated_pelvis_quaternion_xyzw"] for record in records]
    )
    joint_positions = np.stack([record["joint_position"] for record in records])
    return {
        "pose_index": int(records[0]["pose_index"]),
        "pose_label": str(records[0]["pose_label"]),
        "sample_count": len(records),
        "skipped_samples": int(skipped_samples),
        "calibrated_pelvis_position_w": vector_stats(
            records, "calibrated_pelvis_position_w"
        ),
        "calibrated_pelvis_quaternion_xyzw_mean": quaternion_mean_xyzw(
            pelvis_quaternions
        ).round(8).tolist(),
        "measured_ball_position_b": vector_stats(records, "ball_position_b"),
        "fk_sweet_spot_position_b": vector_stats(
            records, "sweet_spot_position_b"
        ),
        "racket_normal_b": vector_stats(records, "racket_normal_b"),
        "direct_error_b": vector_stats(records, "direct_error_b"),
        "direct_error_norm_m": norm_stats(records, "direct_error_b"),
        "contact_plus_error_b": plus_vector,
        "contact_plus_error_norm_m": plus,
        "contact_minus_error_b": minus_vector,
        "contact_minus_error_norm_m": minus,
        "best_contact_side": best_contact_side,
        "best_contact_error_b": best_contact_vector,
        "best_contact_error_norm_m": best_contact_norm,
        "joint_position_mean_rad": joint_positions.mean(axis=0).round(8).tolist(),
        "joint_position_std_max_rad": round(
            float(joint_positions.std(axis=0).max()), 8
        ),
        "raw_pelvis_mean_error_m": round(
            float(np.mean([record["raw_pelvis_mean_error_m"] for record in records])),
            8,
        ),
        "raw_ball_mean_error_m": round(
            float(np.mean([record["raw_ball_mean_error_m"] for record in records])),
            8,
        ),
    }


def save_dataset(
    output_path: Path,
    config_path: Path,
    fk: SweetSpotForwardKinematics,
    ball_radius_m: float,
    records: list[dict],
    summaries: list[dict],
) -> tuple[Path, Path]:
    if not records:
        raise ValueError("No valid calibration records were collected")
    output_path = output_path.expanduser().resolve()
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".json")

    payload: dict[str, np.ndarray] = {}
    for key in records[0]:
        values = [record[key] for record in records]
        if key == "pose_label":
            payload[key] = np.asarray(values, dtype=np.str_)
        else:
            payload[key] = np.asarray(values)

    metadata = {
        "created_unix_s": time.time(),
        "config_path": str(config_path),
        "kinematics_xml_path": str(fk.xml_path),
        "joint_names": list(fk.joint_names),
        "sweet_spot_site": fk.site_name,
        "sweet_spot_position_offset_b": fk.position_offset_b.round(8).tolist(),
        "ball_radius_m": ball_radius_m,
        "coordinate_convention": "+X forward, +Y left, +Z up",
        "contact_model": "ball center = sweet spot +/- radius * site local Z axis",
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=True), dtype=np.str_
    )
    np.savez_compressed(output_path, **payload)

    document = {"metadata": metadata, "poses": summaries}
    json_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path, json_path


def wait_until_ready(node: CalibrationCollector, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_report = 0.0
    while time.monotonic() < deadline:
        missing = node.readiness()
        if not missing:
            return
        now = time.monotonic()
        if now - last_report >= 1.0:
            print(f"[WAIT] Missing or stale: {', '.join(missing)}")
            last_report = now
        time.sleep(0.05)
    raise TimeoutError(
        f"Inputs were not ready within {timeout_s:.1f}s: {', '.join(node.readiness())}"
    )


def default_output_path() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return DEPLOY_DIR / "calibration_data" / f"pelvis_fk_{timestamp}.npz"


def format_vector(value: np.ndarray) -> str:
    return np.array2string(
        np.asarray(value, dtype=np.float64),
        precision=3,
        suppress_small=True,
        floatmode="fixed",
    )


def print_live_record(record: dict) -> None:
    plus_error = np.asarray(record["contact_plus_error_b"], dtype=np.float64)
    minus_error = np.asarray(record["contact_minus_error_b"], dtype=np.float64)
    if np.linalg.norm(plus_error) <= np.linalg.norm(minus_error):
        contact_side = "+normal"
        contact_error = plus_error
    else:
        contact_side = "-normal"
        contact_error = minus_error

    direct_error = np.asarray(record["direct_error_b"], dtype=np.float64)
    print(
        "[LIVE] "
        f"pelvis_w={format_vector(record['calibrated_pelvis_position_w'])}  "
        f"ball_w={format_vector(record['calibrated_ball_position_w'])}  "
        f"sweet_w={format_vector(record['sweet_spot_position_w'])}\n"
        "       "
        f"ball_b={format_vector(record['ball_position_b'])}  "
        f"sweet_b={format_vector(record['sweet_spot_position_b'])}\n"
        "       "
        f"err_b(ball-sweet)={format_vector(direct_error)}  "
        f"err={np.linalg.norm(direct_error):.3f}m  "
        f"contact_err_b={format_vector(contact_error)}  "
        f"contact_err={np.linalg.norm(contact_error):.3f}m  "
        f"side={contact_side}",
        flush=True,
    )


def run_live_monitor(node: CalibrationCollector, print_rate_hz: float) -> None:
    print(
        "Live FK error monitor started. Values use +X forward, +Y left, +Z up. "
        "Press Ctrl+C to stop."
    )
    node.start_capture(0, "live")
    period_s = 1.0 / print_rate_hz
    try:
        while True:
            time.sleep(period_s)
            record = node.pop_latest_capture_record()
            if record is not None:
                print_live_record(record)
    finally:
        node.stop_capture()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively collect static Ball/Pelvis/joint correspondences and "
            "evaluate the deployment MuJoCo sweet-spot FK."
        )
    )
    parser.add_argument(
        "--config",
        default="g1_tppo_student_m14_9.yaml",
        help="Deployment YAML path or name.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--poses", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--sample-rate-hz", type=float, default=50.0)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Continuously print current Mocap/FK positions and errors without saving.",
    )
    parser.add_argument(
        "--print-rate-hz",
        type=float,
        default=5.0,
        help="Terminal update rate used with --live.",
    )
    parser.add_argument("--freshness-s", type=float, default=0.10)
    parser.add_argument("--startup-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--labels",
        default="",
        help="Optional comma-separated pose labels; overrides --poses count.",
    )
    args = parser.parse_args()
    if (
        args.poses <= 0
        or args.duration_s <= 0.0
        or args.startup_timeout_s <= 0.0
        or args.print_rate_hz <= 0.0
    ):
        parser.error(
            "poses, duration-s, startup-timeout-s and print-rate-hz must be positive"
        )
    return args


def main() -> None:
    args = parse_args()
    labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())
    if not labels:
        labels = tuple(f"pose_{index:03d}" for index in range(args.poses))

    rclpy.init()
    node = CalibrationCollector(
        args.config,
        sample_rate_hz=args.sample_rate_hz,
        freshness_s=args.freshness_s,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    all_records: list[dict] = []
    summaries: list[dict] = []
    interrupted = False
    try:
        print("Waiting for calibrated Pelvis, Ball and 29-DoF joint state...")
        wait_until_ready(node, args.startup_timeout_s)
        if args.live:
            run_live_monitor(node, args.print_rate_hz)
            return
        print("Inputs ready. Keep the robot static during each capture.")
        for pose_index, pose_label in enumerate(labels):
            input(
                f"[{pose_index + 1}/{len(labels)}] Hold the ball against the "
                f"same side of the racket at the sweet spot for {pose_label!r}, "
                "then press Enter: "
            )
            wait_until_ready(node, args.startup_timeout_s)
            node.start_capture(pose_index, pose_label)
            print(f"[CAPTURE] {pose_label}: {args.duration_s:.2f}s")
            time.sleep(args.duration_s)
            records, skipped = node.stop_capture()
            if not records:
                raise RuntimeError(
                    f"No valid samples captured for {pose_label}; skipped={skipped}"
                )
            summary = summarize_pose(records, skipped)
            all_records.extend(records)
            summaries.append(summary)
            print(
                f"[OK] {pose_label}: samples={len(records)} skipped={skipped} "
                f"ball_b={np.asarray(summary['measured_ball_position_b']['mean']).round(3)} "
                f"sweet_b={np.asarray(summary['fk_sweet_spot_position_b']['mean']).round(3)} "
                f"ball_minus_sweet_b={np.asarray(summary['direct_error_b']['mean']).round(3)} "
                f"center_distance={summary['direct_error_norm_m']['mean']:.3f}m "
                f"contact_error_b={np.asarray(summary['best_contact_error_b']['mean']).round(3)} "
                f"contact_error={summary['best_contact_error_norm_m']['mean']:.3f}m "
                f"contact_side={summary['best_contact_side']}"
            )
    except KeyboardInterrupt:
        interrupted = True
        if args.live:
            print("\nLive monitor stopped.")
        else:
            print("\nCapture interrupted; saving completed poses.")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)

    if args.live:
        return
    if not all_records:
        raise RuntimeError("No completed pose captures to save")
    output = args.output if args.output is not None else default_output_path()
    npz_path, json_path = save_dataset(
        output,
        node.config_path,
        node.fk,
        node.ball_radius_m,
        all_records,
        summaries,
    )
    print(f"Saved raw samples: {npz_path}")
    print(f"Saved summary:     {json_path}")
    if interrupted:
        print(f"Completed poses: {len(summaries)}/{len(labels)}")


if __name__ == "__main__":
    main()
