"""Generate fixed-contact-time variants of a reference motion."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


_POSITION_KEYS = ("joint_pos", "body_pos_w")
_REQUIRED_KEYS = (
  "fps",
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)
_ISAACLAB_G1_34_BODY_NAMES = (
  "pelvis",
  "left_hip_pitch_link",
  "pelvis_contour_link",
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
  "head_link",
  "left_shoulder_pitch_link",
  "logo_link",
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
  "racket_link",
)


def discrete_contact_times(
  min_time_s: float,
  max_time_s: float,
  interval_s: float,
) -> tuple[float, ...]:
  """Return an interval-aligned grid that never exceeds ``max_time_s``."""
  if min_time_s <= 0.0:
    raise ValueError("min_time_s must be positive")
  if interval_s <= 0.0:
    raise ValueError("interval_s must be positive")
  if max_time_s < min_time_s:
    raise ValueError(
      f"Empty contact-time range: [{min_time_s}, {max_time_s}]"
    )

  count = math.floor((max_time_s - min_time_s) / interval_s + 1.0e-9)
  return tuple(round(min_time_s + i * interval_s, 10) for i in range(count + 1))


def _linear_sample(values: np.ndarray, source_frames: np.ndarray) -> np.ndarray:
  lower = np.floor(source_frames).astype(np.int64)
  upper = np.minimum(lower + 1, values.shape[0] - 1)
  weight_shape = (source_frames.shape[0],) + (1,) * (values.ndim - 1)
  weight = (source_frames - lower).reshape(weight_shape)
  sampled = (1.0 - weight) * values[lower] + weight * values[upper]
  return sampled.astype(values.dtype, copy=False)


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
  norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
  return quaternions / np.maximum(norms, 1.0e-12)


def _slerp_sample(quaternions: np.ndarray, source_frames: np.ndarray) -> np.ndarray:
  lower = np.floor(source_frames).astype(np.int64)
  upper = np.minimum(lower + 1, quaternions.shape[0] - 1)
  q0 = _normalize_quaternions(quaternions[lower].astype(np.float64))
  q1 = _normalize_quaternions(quaternions[upper].astype(np.float64))

  dot = np.sum(q0 * q1, axis=-1, keepdims=True)
  q1 = np.where(dot < 0.0, -q1, q1)
  dot = np.clip(np.abs(dot), 0.0, 1.0)

  weight_shape = (source_frames.shape[0],) + (1,) * (quaternions.ndim - 1)
  weight = (source_frames - lower).reshape(weight_shape)
  theta = np.arccos(dot)
  sin_theta = np.sin(theta)
  near = sin_theta < 1.0e-7
  safe_sin_theta = np.where(near, 1.0, sin_theta)
  sampled = (
    np.sin((1.0 - weight) * theta) / safe_sin_theta * q0
    + np.sin(weight * theta) / safe_sin_theta * q1
  )
  linear = (1.0 - weight) * q0 + weight * q1
  sampled = np.where(near, linear, sampled)
  return _normalize_quaternions(sampled).astype(quaternions.dtype, copy=False)


def _differentiate(values: np.ndarray, dt: float) -> np.ndarray:
  edge_order = 2 if values.shape[0] >= 3 else 1
  derivative = np.gradient(
    values.astype(np.float64), dt, axis=0, edge_order=edge_order
  )
  return np.asarray(derivative)


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
  lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
  rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
  return np.stack(
    (
      lw * rw - lx * rx - ly * ry - lz * rz,
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
    ),
    axis=-1,
  )


def _rotation_vector_between(start: np.ndarray, end: np.ndarray) -> np.ndarray:
  conjugate = start.copy()
  conjugate[..., 1:] *= -1.0
  delta = _normalize_quaternions(_quat_multiply(end, conjugate))
  delta = np.where(delta[..., :1] < 0.0, -delta, delta)
  vector = delta[..., 1:]
  vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
  angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[..., :1], 0.0, 1.0))
  scale = np.full_like(vector_norm, 2.0)
  np.divide(angle, vector_norm, out=scale, where=vector_norm > 1.0e-10)
  return vector * scale


def _angular_velocity_world(quaternions: np.ndarray, dt: float) -> np.ndarray:
  quaternions = _normalize_quaternions(quaternions.astype(np.float64))
  velocity = np.zeros(quaternions.shape[:-1] + (3,), dtype=np.float64)
  if quaternions.shape[0] < 2:
    return velocity
  velocity[0] = _rotation_vector_between(quaternions[0], quaternions[1]) / dt
  velocity[-1] = _rotation_vector_between(quaternions[-2], quaternions[-1]) / dt
  if quaternions.shape[0] > 2:
    velocity[1:-1] = (
      _rotation_vector_between(quaternions[:-2], quaternions[2:]) / (2.0 * dt)
    )
  return velocity


def resample_motion_arrays(
  arrays: Mapping[str, np.ndarray],
  source_strike_frame: int,
  target_strike_frame: int,
) -> tuple[dict[str, np.ndarray], float]:
  """Time-warp a full motion while mapping the strike to a target frame."""
  missing = [key for key in _REQUIRED_KEYS if key not in arrays]
  if missing:
    raise KeyError(f"Motion is missing required arrays: {missing}")

  source_frames_count = int(arrays["joint_pos"].shape[0])
  if not 0 < source_strike_frame < source_frames_count:
    raise ValueError("source_strike_frame must be inside the source motion")
  if target_strike_frame <= 0:
    raise ValueError("target_strike_frame must be positive")

  fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
  if fps <= 0.0:
    raise ValueError("Motion fps must be positive")

  speed_scale = source_strike_frame / float(target_strike_frame)
  last_source_frame = float(source_frames_count - 1)
  output_steps = math.floor(last_source_frame / speed_scale + 1.0e-9)
  source_frames = np.arange(output_steps + 1, dtype=np.float64) * speed_scale
  source_frames = np.minimum(source_frames, last_source_frame)
  if last_source_frame - source_frames[-1] > 1.0e-8:
    source_frames = np.concatenate((source_frames, [last_source_frame]))

  if not np.isclose(
    source_frames[target_strike_frame], source_strike_frame, atol=1.0e-7
  ):
    raise RuntimeError("Resampled strike frame does not map to the source strike")

  result: dict[str, np.ndarray] = {
    key: np.array(value, copy=True)
    for key, value in arrays.items()
    if key not in _REQUIRED_KEYS
  }
  result["fps"] = np.array(arrays["fps"], copy=True)
  for key in _POSITION_KEYS:
    result[key] = _linear_sample(arrays[key], source_frames)
  result["body_quat_w"] = _slerp_sample(arrays["body_quat_w"], source_frames)

  dt = 1.0 / fps
  result["joint_vel"] = _differentiate(result["joint_pos"], dt).astype(
    arrays["joint_vel"].dtype
  )
  result["body_lin_vel_w"] = _differentiate(result["body_pos_w"], dt).astype(
    arrays["body_lin_vel_w"].dtype
  )
  result["body_ang_vel_w"] = _angular_velocity_world(
    result["body_quat_w"], dt
  ).astype(arrays["body_ang_vel_w"].dtype)
  return result, speed_scale


def generate_time_warp_variants(
  source_npz: str | Path,
  output_dir: str | Path,
  *,
  min_contact_time_s: float = 1.0,
  interval_s: float = 0.1,
  body_names: tuple[str, ...] | None = None,
  write_manifest: bool = True,
) -> dict[str, object]:
  """Generate discrete contact-time variants and return their manifest."""
  source_path = Path(source_npz).expanduser().resolve()
  sidecar_path = source_path.with_suffix(".json")
  if not source_path.exists():
    raise FileNotFoundError(source_path)
  if not sidecar_path.exists():
    raise FileNotFoundError(sidecar_path)

  with np.load(source_path) as loaded:
    arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
  if body_names is not None:
    body_count = int(arrays["body_pos_w"].shape[1])
    if len(body_names) != body_count:
      raise ValueError(
        f"Declared body layout has {len(body_names)} names, expected {body_count}"
      )
    arrays["body_names"] = np.asarray(body_names)
  metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
  source_strike_frame = int(metadata["strike_frame"])
  fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
  nominal_contact_time_s = source_strike_frame / fps
  contact_times = discrete_contact_times(
    min_contact_time_s, nominal_contact_time_s, interval_s
  )

  destination = Path(output_dir).expanduser().resolve()
  destination.mkdir(parents=True, exist_ok=True)
  variants: list[dict[str, object]] = []
  for contact_time_s in contact_times:
    target_strike_frame = round(contact_time_s * fps)
    if not math.isclose(target_strike_frame / fps, contact_time_s, abs_tol=1.0e-9):
      raise ValueError(
        f"Contact time {contact_time_s}s is not representable at {fps:g} Hz"
      )
    resampled, speed_scale = resample_motion_arrays(
      arrays, source_strike_frame, target_strike_frame
    )
    milliseconds = round(contact_time_s * 1000.0)
    stem = f"{source_path.stem}_t{milliseconds:04d}ms"
    output_npz = destination / f"{stem}.npz"
    output_json = destination / f"{stem}.json"
    np.savez_compressed(output_npz, **resampled)

    variant_metadata = dict(metadata)
    variant_metadata.update(
      {
        "strike_frame": target_strike_frame,
        "frames": int(resampled["joint_pos"].shape[0]),
        "source_motion": source_path.name,
        "source_strike_frame": source_strike_frame,
        "target_contact_time_s": contact_time_s,
        "reference_speed_scale": speed_scale,
        "sampling_weight": 1.0 / len(contact_times),
      }
    )
    output_json.write_text(
      json.dumps(variant_metadata, indent=2) + "\n", encoding="utf-8"
    )
    variants.append(
      {
        "npz": output_npz.name,
        "json": output_json.name,
        "contact_time_s": contact_time_s,
        "strike_frame": target_strike_frame,
        "frames": int(resampled["joint_pos"].shape[0]),
        "reference_speed_scale": speed_scale,
      }
    )

  manifest: dict[str, object] = {
    "source_motion": str(source_path),
    "source_strike_frame": source_strike_frame,
    "fps": fps,
    "nominal_contact_time_s": nominal_contact_time_s,
    "requested_min_contact_time_s": min_contact_time_s,
    "requested_max_contact_time_s": nominal_contact_time_s,
    "interval_s": interval_s,
    "actual_contact_times_s": list(contact_times),
    "variants": variants,
  }
  if write_manifest:
    (destination / "manifest.json").write_text(
      json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
  return manifest


def generate_time_warp_dataset(
  source_pattern: str,
  output_dir: str | Path,
  *,
  min_contact_time_s: float = 1.0,
  interval_s: float = 0.1,
  body_names: tuple[str, ...] | None = None,
) -> dict[str, object]:
  """Generate time-warp variants for every motion matching a glob pattern."""
  source_paths = tuple(Path(path) for path in sorted(glob.glob(source_pattern)))
  if not source_paths:
    raise FileNotFoundError(f"Source glob matched no files: {source_pattern}")

  destination = Path(output_dir).expanduser().resolve()
  destination.mkdir(parents=True, exist_ok=True)
  source_manifests: list[dict[str, object]] = []
  variant_count = 0
  short_sources: list[str] = []
  for source_path in source_paths:
    with np.load(source_path) as loaded:
      fps = float(np.asarray(loaded["fps"]).reshape(-1)[0])
    metadata = json.loads(
      source_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    nominal_contact_time_s = int(metadata["strike_frame"]) / fps
    source_min_time_s = min(min_contact_time_s, nominal_contact_time_s)
    if nominal_contact_time_s < min_contact_time_s:
      short_sources.append(source_path.name)

    manifest = generate_time_warp_variants(
      source_path,
      destination,
      min_contact_time_s=source_min_time_s,
      interval_s=interval_s,
      body_names=body_names,
      write_manifest=False,
    )
    source_manifests.append(manifest)
    variant_count += len(manifest["variants"])

  dataset_manifest: dict[str, object] = {
    "source_pattern": source_pattern,
    "source_count": len(source_paths),
    "variant_count": variant_count,
    "min_contact_time_s": min_contact_time_s,
    "interval_s": interval_s,
    "short_sources_kept_at_nominal_time": short_sources,
    "sources": source_manifests,
  }
  (destination / "dataset_manifest.json").write_text(
    json.dumps(dataset_manifest, indent=2) + "\n", encoding="utf-8"
  )
  return dataset_manifest


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Generate full-motion variants with discretized strike times."
  )
  parser.add_argument("source_npz", type=Path, nargs="?")
  parser.add_argument(
    "--source-glob",
    help="Generate one flat motion set from every NPZ matching this pattern.",
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--min-contact-time", type=float, default=1.0)
  parser.add_argument("--interval", type=float, default=0.1)
  parser.add_argument(
    "--body-layout",
    choices=("none", "isaaclab-g1-34"),
    default="none",
    help="Optional explicit NPZ body-column layout metadata.",
  )
  args = parser.parse_args()

  if (args.source_npz is None) == (args.source_glob is None):
    parser.error("provide exactly one of source_npz or --source-glob")

  body_names = (
    _ISAACLAB_G1_34_BODY_NAMES
    if args.body_layout == "isaaclab-g1-34"
    else None
  )

  if args.source_glob is not None:
    manifest = generate_time_warp_dataset(
      args.source_glob,
      args.output_dir,
      min_contact_time_s=args.min_contact_time,
      interval_s=args.interval,
      body_names=body_names,
    )
    summary = {
      "output_dir": str(args.output_dir.resolve()),
      "source_count": manifest["source_count"],
      "variant_count": manifest["variant_count"],
      "short_sources_kept_at_nominal_time": manifest[
        "short_sources_kept_at_nominal_time"
      ],
    }
  else:
    manifest = generate_time_warp_variants(
      args.source_npz,
      args.output_dir,
      min_contact_time_s=args.min_contact_time,
      interval_s=args.interval,
      body_names=body_names,
    )
    summary = {
      "output_dir": str(args.output_dir.resolve()),
      "variant_count": len(manifest["variants"]),
      "actual_contact_times_s": manifest["actual_contact_times_s"],
    }
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
