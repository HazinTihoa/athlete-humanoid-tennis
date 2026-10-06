"""Run a deployable Student ONNX policy in a standalone MuJoCo simulation."""

from __future__ import annotations

import glob
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

import mujoco
import mujoco.viewer
import numpy as np
import tomllib
import torch
import tyro
from athlete.goal_cond_tracking.torch_tennis_planner import (
  TorchTennisTrajectoryBatch,
  classify_tennis_failure_trajectories,
  match_tennis_trajectories_to_motions_torch,
  retarget_tennis_launches_to_miss_roots_torch,
  sample_root_directed_tennis_launches_torch,
)
from athlete.goal_cond_tracking.warp_tennis_planner import (
  simulate_tennis_trajectories_warp_fused,
)
from athlete.scripts.tennis_court_spec import (
  COURT_NET_X,
  RACKET_COLLISION_HALF_SIZE_M,
  RACKET_COLLISION_POS_WRIST_M,
  RACKET_COLLISION_QUAT_WXYZ,
  RACKET_NOMINAL_COM_WRIST_M,
  RACKET_NOMINAL_INERTIA_KG_M2,
  RACKET_NOMINAL_MASS_KG,
  attach_tennis_court_spec,
)
from athlete.scripts.tennis_physics import (
  STANDARD_TENNIS_DOMAIN_RANDOMIZATION,
  STANDARD_TENNIS_PHYSICS,
  contact_damping_for_restitution,
  tennis_ball_aerodynamic_wrench_numpy,
  tennis_ball_aerodynamic_wrench_torch,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MOTION_CONFIG = Path(
  "athlete/src/athlete/motion_sets/"
  "motion_train_configs/rollout_1000epis_ep_0000_phase.toml"
)
LEGACY_OBSERVATION_DIM = 102
LANDING_OBSERVATION_DIM = 119
INTENT_CURRENT_OBSERVATION_DIM = 133
INTENT_STATE_TOKEN_DIM = 93
INTENT_STATE_HISTORY_STEPS = 5
INTENT_BALL_TOKEN_DIM = 6
INTENT_ROBUST_BALL_TOKEN_DIM = 8
INTENT_BALL_HISTORY_STEPS = 5
INTENT_HISTORY_BUFFER_LENGTH = 25
INTENT_HISTORY_LAGS = (20, 15, 10, 5, 0)
INTENT_OBSERVATION_DIM = (
  INTENT_CURRENT_OBSERVATION_DIM
  + INTENT_STATE_HISTORY_STEPS * INTENT_STATE_TOKEN_DIM
  + INTENT_BALL_HISTORY_STEPS * INTENT_BALL_TOKEN_DIM
)
INTENT_ROBUST_OBSERVATION_DIM = (
  INTENT_CURRENT_OBSERVATION_DIM
  + INTENT_STATE_HISTORY_STEPS * INTENT_STATE_TOKEN_DIM
  + INTENT_BALL_HISTORY_STEPS * INTENT_ROBUST_BALL_TOKEN_DIM
)
INTENT_GLOBAL_ROOT_OBSERVATION_DIM = INTENT_OBSERVATION_DIM + 3
INTENT_OBSERVATION_DIMS = (
  INTENT_OBSERVATION_DIM, INTENT_ROBUST_OBSERVATION_DIM,
  INTENT_GLOBAL_ROOT_OBSERVATION_DIM,
)
SUPPORTED_OBSERVATION_DIMS = (
  LEGACY_OBSERVATION_DIM,
  LANDING_OBSERVATION_DIM,
  *INTENT_OBSERVATION_DIMS,
)
ACTION_DIM = 29
ISAACLAB_JOINT_NAMES = (
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
TARGET_GOAL_DIM = 11
CONTACT_TIME_STEP_S = 0.1
MAX_CONTACT_SPEEDUP = 2.0
UI_FONT_SCALE = 3.0
PHYSICS = STANDARD_TENNIS_PHYSICS
INCOMING_HORIZONTAL_ANGLE_HALF_WIDTH_DEG = 15.0
INCOMING_NET_CROSSING_HEIGHT_RANGE_M = (1.5, 3.5)
INCOMING_MAXIMUM_INITIAL_SPEED_M_S = 7.0
INCOMING_TRAJECTORY_DT = 0.01
INCOMING_TRAJECTORY_HORIZON_S = 4.0
INCOMING_MATCH_MAXIMUM_DISTANCE_M = 0.20
INCOMING_MATCH_ATTEMPTS = 8
INCOMING_MATCH_RETRY_ROUNDS = 3
INCOMING_MATCH_ADAPTIVE_BATCH_SIZE = 2
INCOMING_MATCH_HIERARCHICAL_TOP_K = 32
INCOMING_MATCH_HIERARCHICAL_NEIGHBOR_RADIUS = 1
INCOMING_MATCH_DEADLINE_OFFSET_S = 0.01
MANUAL_SHOOTING_FINITE_DIFFERENCE_M_S = 0.05
MANUAL_SHOOTING_MAX_CORRECTION_M_S = 3.0
MANUAL_BOUNCE_POSITION_TOLERANCE_M = 0.03
MANUAL_BOUNCE_TIME_TOLERANCE_S = 0.02
MANUAL_TIME_SAMPLE_ATTEMPTS = 16
STANDALONE_RACKET_RESTITUTION = (
  sum(STANDARD_TENNIS_DOMAIN_RANDOMIZATION.racket_restitution) / 2.0
)
STANDALONE_RACKET_CONTACT_DAMPING = contact_damping_for_restitution(
  STANDALONE_RACKET_RESTITUTION
)


@dataclass(frozen=True)
class _CourtProfile:
  name: Literal["standard", "small"]
  net_x_m: float
  net_half_width_m: float
  effective_half_length_m: float
  effective_singles_half_width_m: float
  launch_position_min: tuple[float, float, float]
  launch_position_max: tuple[float, float, float]
  horizontal_speed_range_m_s: tuple[float, float]
  landing_target_xy: tuple[float, float]
  viewer_lookat: tuple[float, float, float]
  viewer_distance: float

  @property
  def robot_half_baseline_x_m(self) -> float:
    return self.net_x_m - self.effective_half_length_m


STANDARD_COURT_PROFILE = _CourtProfile(
  name="standard",
  net_x_m=COURT_NET_X,
  net_half_width_m=PHYSICS.court.net_half_width_m,
  effective_half_length_m=11.885,
  effective_singles_half_width_m=4.115,
  launch_position_min=(8.0, -2.0, 0.5),
  launch_position_max=(10.0, 2.0, 1.3),
  horizontal_speed_range_m_s=(3.5, 5.25),
  landing_target_xy=(7.0, 0.0),
  viewer_lookat=(COURT_NET_X + 0.4, 0.0, 0.55),
  viewer_distance=21.5,
)
SMALL_COURT_PROFILE = _CourtProfile(
  name="small",
  net_x_m=3.5,
  net_half_width_m=2.5,
  effective_half_length_m=5.0,
  effective_singles_half_width_m=2.75,
  launch_position_min=(5.5, -1.5, 0.6),
  launch_position_max=(6.5, 1.5, 1.2),
  horizontal_speed_range_m_s=(2.8, 4.2),
  landing_target_xy=(6.0, 0.0),
  viewer_lookat=(3.9, 0.0, 0.55),
  viewer_distance=14.0,
)


def _resolve_court_profile(name: Literal["standard", "small"]) -> _CourtProfile:
  if name == "standard":
    return STANDARD_COURT_PROFILE
  if name == "small":
    return SMALL_COURT_PROFILE
  raise ValueError(f"Unsupported court preset: {name!r}")


@dataclass(frozen=True)
class StudentOnnxPlayConfig:
  """Configuration for standalone Student ONNX playback."""

  onnx_policy_file: Path
  motion_config: Path = DEFAULT_MOTION_CONFIG
  viewer: Literal["native"] = "native"
  court_preset: Literal["standard", "small"] = "standard"
  seed: int = 42
  physics_dt: float = 0.0025
  control_dt: float = 0.02
  ball_release_lead_s: float = 0.02
  horizontal_speed_threshold_mps: float = 0.15
  on_time_tolerance_s: float = 0.10
  timing_plot: bool = False
  timing_plot_fps: float = 5.0
  render_fps: float = 30.0
  realtime: bool = True
  landing_target_x: float | None = None
  landing_target_y: float | None = None
  landing_target_std_x: float = 0.0
  landing_target_std_y: float = 0.0
  # Lower bound in court/world X; condition the Gaussian, not the startup X.
  landing_target_min_x: float | None = None
  show_landing_target: bool = True
  show_sweet_point: bool = True
  sweet_spot_state_source: Literal["simulation", "fk"] = "simulation"
  """Use fk for policies trained on deployment-style position differences."""
  sweet_spot_velocity_smoothing: float = 0.35
  continuous_strikes: bool = True
  manual_mode: bool = False
  manual_mouse_select: bool = False
  manual_launch_x: float = 9.0
  manual_launch_y: float = 0.0
  manual_launch_z: float = 1.0
  manual_bounce_x: float = 3.5
  manual_bounce_y: float = 0.0
  manual_bounce_time_s: float = 1.2
  manual_random_bounce_time: bool = False
  manual_bounce_time_min_s: float = 0.9
  manual_bounce_time_max_s: float = 1.5
  manual_solver_iterations: int = 5
  manual_max_initial_speed_mps: float = 15.0
  ball_observation_latency_min_steps: int = 0
  ball_observation_latency_max_steps: int = 3
  ball_observation_dropout_start_probability: float = 0.03
  ball_observation_dropout_duration_min_steps: int = 2
  ball_observation_dropout_duration_max_steps: int = 5
  ball_observation_position_noise_std: float = 0.05
  ball_observation_velocity_noise_std: float = 0.05
  failure_trajectory_probability: float = 0.0
  """Probability of an intentional failed launch after the first valid ball."""
  failure_trajectory_no_net_fraction: float = 0.5
  """Fraction of failed launches selected from trajectories that miss the net."""
  failure_trajectory_xy_distance_threshold_m: float = 3.0
  """Minimum planar distance from G1 for an over-net failure trajectory."""


@dataclass(frozen=True)
class _GoalDistribution:
  source_file: Path
  clip: str
  frames: int
  strike_frame: int
  fps: float
  position_mean: np.ndarray
  aligned_position_mean: np.ndarray
  position_std: np.ndarray
  velocity_mean: np.ndarray
  velocity_std: np.ndarray
  orientation_rpy_mean: np.ndarray
  orientation_rpy_std: np.ndarray
  sampling_weight: float
  initial_anchor_position_w: np.ndarray
  initial_anchor_quaternion_w: np.ndarray
  initial_anchor_linear_velocity_w: np.ndarray
  initial_anchor_angular_velocity_w: np.ndarray
  initial_joint_position: np.ndarray
  initial_joint_velocity: np.ndarray


@dataclass(frozen=True)
class _GoalSample:
  distribution: _GoalDistribution
  position_local: np.ndarray
  position_world: np.ndarray
  velocity_world: np.ndarray
  orientation_world: np.ndarray
  contact_time_s: float


@dataclass(frozen=True)
class _IncomingBallPlan:
  distribution: _GoalDistribution
  launch_position_w: np.ndarray
  launch_velocity_w: np.ndarray
  target_position_w: np.ndarray
  contact_time_s: float
  match_distance_m: float
  valid: bool
  attempts: int
  motion_matched: bool = True
  failure_type: Literal["no_net", "too_far"] | None = None
  first_bounce_position_w: np.ndarray | None = None
  first_bounce_time_s: float | None = None


@dataclass(frozen=True)
class _StepReport:
  elapsed_s: float
  sampled_strike_time_s: float
  horizontal_speed_mps: float
  actual_hit_time_s: float | None
  released: bool
  episode_complete: bool


def _is_robot_half_singles_point(
  x_w: float,
  y_w: float,
  court: _CourtProfile = STANDARD_COURT_PROFILE,
) -> bool:
  """Return whether a court/world XY point is on G1's singles half."""
  return (
    court.robot_half_baseline_x_m <= x_w <= court.net_x_m
    and abs(y_w) <= court.effective_singles_half_width_m
  )


def _sample_manual_bounce_time(
  cfg: StudentOnnxPlayConfig, rng: np.random.Generator
) -> float:
  if not cfg.manual_random_bounce_time:
    return cfg.manual_bounce_time_s
  return float(rng.uniform(cfg.manual_bounce_time_min_s, cfg.manual_bounce_time_max_s))


def _resolve_config_path(path: Path, config_dir: Path) -> Path:
  path = path.expanduser()
  if path.is_absolute():
    return path.resolve()
  config_relative = (config_dir / path).resolve()
  if config_relative.exists():
    return config_relative
  return (REPO_ROOT / path).resolve()


def _xyz(
  mapping: dict[str, Any] | None, default: tuple[float, float, float]
) -> np.ndarray:
  mapping = mapping or {}
  return np.array(
    [
      float(mapping.get("x", default[0])),
      float(mapping.get("y", default[1])),
      float(mapping.get("z", default[2])),
    ],
    dtype=np.float64,
  )


def _rpy(
  mapping: dict[str, Any] | None, default: tuple[float, float, float]
) -> np.ndarray:
  mapping = mapping or {}
  return np.array(
    [
      float(mapping.get("roll", default[0])),
      float(mapping.get("pitch", default[1])),
      float(mapping.get("yaw", default[2])),
    ],
    dtype=np.float64,
  )


def _motion_joint_state_in_model_order(
  model: mujoco.MjModel,
  joint_position: np.ndarray,
  joint_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Resolve rollout IsaacLab-vs-MuJoCo joint order using model limits."""
  model_joint_names = tuple(
    str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
    for joint_id in range(1, ACTION_DIM + 1)
  )
  isaaclab_to_model = np.asarray(
    [ISAACLAB_JOINT_NAMES.index(name) for name in model_joint_names],
    dtype=np.int64,
  )
  lower = np.empty(ACTION_DIM, dtype=np.float64)
  upper = np.empty(ACTION_DIM, dtype=np.float64)
  for index, joint_id in enumerate(range(1, ACTION_DIM + 1)):
    if model.jnt_limited[joint_id]:
      lower[index], upper[index] = model.jnt_range[joint_id]
    else:
      lower[index], upper[index] = -1.0e6, 1.0e6

  direct = joint_position[:, :ACTION_DIM]
  mapped = joint_position[:, isaaclab_to_model]
  direct_violations = np.count_nonzero(
    (direct < lower - 0.02) | (direct > upper + 0.02)
  )
  mapped_violations = np.count_nonzero(
    (mapped < lower - 0.02) | (mapped > upper + 0.02)
  )
  if mapped_violations < direct_violations:
    return mapped, joint_velocity[:, isaaclab_to_model]
  return direct, joint_velocity[:, :ACTION_DIM]


def _load_goal_distributions(
  motion_config: Path,
) -> tuple[Path, list[_GoalDistribution]]:
  config_path = motion_config.expanduser().resolve()
  with config_path.open("rb") as file:
    config = tomllib.load(file)

  dataset = config.get("dataset")
  if not isinstance(dataset, dict):
    raise TypeError(
      "Standalone Student Play currently requires a motion TOML with [dataset]."
    )
  raw_glob = dataset.get("glob")
  if not raw_glob:
    raise ValueError("[dataset].glob is required for goal sampling.")
  pattern_path = Path(str(raw_glob)).expanduser()
  if pattern_path.is_absolute():
    pattern = str(pattern_path)
  else:
    config_pattern = str((config_path.parent / pattern_path).resolve())
    repo_pattern = str((REPO_ROOT / pattern_path).resolve())
    pattern = config_pattern if glob.glob(config_pattern) else repo_pattern
  source_files = [Path(path).resolve() for path in sorted(glob.glob(pattern))]
  if not source_files:
    raise FileNotFoundError(f"Dataset glob matched no files: {pattern}")

  robot_xml_raw = config.get("robot", {}).get("xml")
  if not robot_xml_raw:
    raise ValueError("[robot].xml is required for standalone MuJoCo Play.")
  robot_xml = _resolve_config_path(Path(str(robot_xml_raw)), config_path.parent)
  if not robot_xml.is_file():
    raise FileNotFoundError(f"Robot XML not found: {robot_xml}")
  robot_model = mujoco.MjModel.from_xml_path(str(robot_xml))

  position_std = _xyz(dataset.get("target_pos_std"), (0.2, 0.3, 0.3))
  velocity_std = _xyz(dataset.get("target_vel_std"), (0.0, 0.0, 0.0))
  orientation_std = _rpy(dataset.get("target_orientation_std"), (0.0, 0.0, 0.0))
  default_weight = float(dataset.get("sampling_weight", 1.0))
  velocity_pitch = float(math.atan2(1.0, 0.1))
  distributions: list[_GoalDistribution] = []

  for source_file in source_files:
    sidecar = source_file.with_suffix(".json")
    if not sidecar.is_file():
      raise FileNotFoundError(f"Missing dataset sidecar: {sidecar}")
    with sidecar.open("r", encoding="utf-8") as file:
      metadata = json.load(file)

    ball_local = metadata.get("ball_local")
    if not isinstance(ball_local, list) or len(ball_local) != 3:
      raise ValueError(f"Invalid ball_local in {sidecar}")
    clip = str(metadata.get("clip", "")).lower()
    if clip.startswith("fh_"):
      orientation_mean = np.array([0.0, velocity_pitch, 0.0], dtype=np.float64)
    elif clip.startswith("bh_"):
      orientation_mean = np.array(
        [0.0, math.pi - velocity_pitch, math.pi], dtype=np.float64
      )
    else:
      raise ValueError(f"Cannot infer forehand/backhand from {sidecar}: {clip!r}")

    frames = int(metadata["frames"])
    strike_frame = int(metadata["strike_frame"])
    if frames < 2 or not 0 <= strike_frame < frames:
      raise ValueError(
        f"Invalid frames/strike_frame in {sidecar}: {frames}/{strike_frame}"
      )
    with np.load(source_file) as motion:
      joint_position, joint_velocity = _motion_joint_state_in_model_order(
        robot_model,
        np.asarray(motion["joint_pos"], dtype=np.float64),
        np.asarray(motion["joint_vel"], dtype=np.float64),
      )
      reference_position_0 = np.asarray(motion["body_pos_w"][0, 0], dtype=np.float64)
      reference_quat_0 = np.asarray(motion["body_quat_w"][0, 0], dtype=np.float64)
      reference_linear_velocity_0 = np.asarray(
        motion["body_lin_vel_w"][0, 0], dtype=np.float64
      )
      reference_angular_velocity_0 = np.asarray(
        motion["body_ang_vel_w"][0, 0], dtype=np.float64
      )
      reference_joint_position_0 = np.asarray(
        joint_position[0], dtype=np.float64
      )
      reference_joint_velocity_0 = np.asarray(
        joint_velocity[0], dtype=np.float64
      )
      fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    reference_yaw = math.atan2(
      2.0
      * (
        reference_quat_0[0] * reference_quat_0[3]
        + reference_quat_0[1] * reference_quat_0[2]
      ),
      1.0
      - 2.0
      * (
        reference_quat_0[2] * reference_quat_0[2]
        + reference_quat_0[3] * reference_quat_0[3]
      ),
    )
    aligned_reference_quat = _quat_mul(
      _quat_from_rpy(np.array([0.0, 0.0, -reference_yaw])),
      reference_quat_0,
    )
    aligned_anchor_position = np.array(
      [0.0, 0.0, reference_position_0[2]], dtype=np.float64
    )
    aligned_position_mean = aligned_anchor_position + _quat_apply(
      aligned_reference_quat, np.asarray(ball_local, dtype=np.float64)
    )
    distributions.append(
      _GoalDistribution(
        source_file=source_file,
        clip=clip,
        frames=frames,
        strike_frame=strike_frame,
        fps=fps,
        position_mean=np.asarray(ball_local, dtype=np.float64),
        aligned_position_mean=aligned_position_mean,
        position_std=position_std.copy(),
        velocity_mean=np.array([1.0, 0.0, 0.1], dtype=np.float64),
        velocity_std=velocity_std.copy(),
        orientation_rpy_mean=orientation_mean,
        orientation_rpy_std=orientation_std.copy(),
        sampling_weight=float(metadata.get("sampling_weight", default_weight)),
        initial_anchor_position_w=reference_position_0.copy(),
        initial_anchor_quaternion_w=reference_quat_0.copy(),
        initial_anchor_linear_velocity_w=reference_linear_velocity_0.copy(),
        initial_anchor_angular_velocity_w=reference_angular_velocity_0.copy(),
        initial_joint_position=reference_joint_position_0.copy(),
        initial_joint_velocity=reference_joint_velocity_0.copy(),
      )
    )

  if not any(distribution.sampling_weight > 0.0 for distribution in distributions):
    raise ValueError("At least one goal distribution must have positive weight.")
  return robot_xml, distributions


def _quat_mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
  lw, lx, ly, lz = lhs
  rw, rx, ry, rz = rhs
  return np.array(
    [
      lw * rw - lx * rx - ly * ry - lz * rz,
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
    ],
    dtype=np.float64,
  )


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
  result = quaternion.copy()
  result[1:] *= -1.0
  return result


def _quat_apply(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
  quaternion = quaternion / np.linalg.norm(quaternion)
  imaginary = quaternion[1:]
  return (
    vector
    + 2.0 * np.cross(imaginary, np.cross(imaginary, vector))
    + 2.0 * quaternion[0] * np.cross(imaginary, vector)
  )


def _quat_from_rpy(rpy: np.ndarray) -> np.ndarray:
  roll, pitch, yaw = rpy * 0.5
  cr, sr = math.cos(roll), math.sin(roll)
  cp, sp = math.cos(pitch), math.sin(pitch)
  cy, sy = math.cos(yaw), math.sin(yaw)
  quaternion = np.array(
    [
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ],
    dtype=np.float64,
  )
  return quaternion / np.linalg.norm(quaternion)


def _parse_float_metadata(metadata: dict[str, str], key: str) -> np.ndarray:
  value = metadata.get(key)
  if value is None:
    raise ValueError(f"Student ONNX is missing required metadata: {key}")
  return np.array([float(item) for item in value.split(",")], dtype=np.float64)


class _StudentOnnxPolicy:
  """Minimal ONNX Runtime adapter with no RSL-RL or task dependencies."""

  def __init__(self, policy_path: Path) -> None:
    try:
      import onnxruntime as ort
    except ImportError as exc:
      raise RuntimeError("Run `uv sync` to install onnxruntime.") from exc

    path = policy_path.expanduser().resolve()
    if not path.is_file():
      raise FileNotFoundError(f"Student ONNX not found: {path}")
    providers = [
      provider
      for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
      if provider in ort.get_available_providers()
    ]
    self.session = ort.InferenceSession(str(path), providers=providers)
    self.inputs = {value.name: value for value in self.session.get_inputs()}
    if "obs" not in self.inputs:
      raise ValueError(f"Student ONNX has no 'obs' input: {tuple(self.inputs)}")
    obs_shape = self.inputs["obs"].shape
    if not isinstance(obs_shape[-1], int):
      raise TypeError(f"Student ONNX observation dimension must be static: {obs_shape}")
    self.observation_dim = obs_shape[-1]
    if self.observation_dim not in SUPPORTED_OBSERVATION_DIMS:
      raise ValueError(
        f"Student ONNX expects {self.observation_dim} observations; supported "
        f"dimensions are {SUPPORTED_OBSERVATION_DIMS}."
      )
    output_names = [value.name for value in self.session.get_outputs()]
    self.action_output = "actions" if "actions" in output_names else output_names[0]
    action_info = next(
      output
      for output in self.session.get_outputs()
      if output.name == self.action_output
    )
    if isinstance(action_info.shape[-1], int) and action_info.shape[-1] != ACTION_DIM:
      raise ValueError(
        f"Student ONNX outputs {action_info.shape[-1]} actions, expected {ACTION_DIM}."
      )
    self.metadata = self.session.get_modelmeta().custom_metadata_map
    self.providers = tuple(self.session.get_providers())

  def __call__(self, observation: np.ndarray) -> np.ndarray:
    if observation.shape != (self.observation_dim,):
      raise ValueError(
        f"Student observation must have shape ({self.observation_dim},), "
        f"got {observation.shape}."
      )
    feeds = {"obs": observation.astype(np.float32, copy=False)[None, :]}
    for name, value in self.inputs.items():
      if name == "obs":
        continue
      shape = [
        dimension if isinstance(dimension, int) else 1 for dimension in value.shape
      ]
      feeds[name] = np.zeros(shape, dtype=np.float32)
    action = np.asarray(
      self.session.run([self.action_output], feeds)[0], dtype=np.float64
    ).reshape(-1)
    if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
      raise RuntimeError(f"Invalid Student ONNX action: shape={action.shape}")
    return action


class _RootDirectedIncomingBallPlanner:
  """Single-environment form of the training WarpRootDirected planner."""

  def __init__(
    self,
    distributions: list[_GoalDistribution],
    *,
    seed: int,
    court: _CourtProfile = STANDARD_COURT_PROFILE,
  ) -> None:
    if not distributions:
      raise ValueError("Incoming-ball planning requires motion targets.")
    self.distributions = distributions
    self.court = court
    self.device = torch.device("cpu")
    self.dtype = torch.float32
    torch.manual_seed(seed)
    self.position_min = torch.tensor(
      court.launch_position_min, dtype=self.dtype, device=self.device
    )
    self.position_max = torch.tensor(
      court.launch_position_max, dtype=self.dtype, device=self.device
    )
    self.nominal_contact_times = torch.tensor(
      [item.strike_frame / item.fps for item in distributions],
      dtype=self.dtype,
      device=self.device,
    )
    self.minimum_contact_times = self.nominal_contact_times / MAX_CONTACT_SPEEDUP
    self.tangent_retention = float(
      sum(STANDARD_TENNIS_DOMAIN_RANDOMIZATION.ground_tangent_speed_retention) / 2.0
    )

  @staticmethod
  def _aligned_motion_targets_w(
    distributions: list[_GoalDistribution],
    pelvis_pos_w: np.ndarray,
    pelvis_rotation_w: np.ndarray,
  ) -> np.ndarray:
    """Align frame-0 motion targets to current root XY/yaw, not live root height."""
    yaw = math.atan2(pelvis_rotation_w[1, 0], pelvis_rotation_w[0, 0])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    yaw_rotation_w = np.array(
      [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
      dtype=np.float64,
    )
    planar_anchor_w = np.array(
      [pelvis_pos_w[0], pelvis_pos_w[1], 0.0], dtype=np.float64
    )
    return np.stack(
      [
        planar_anchor_w + yaw_rotation_w @ item.aligned_position_mean
        for item in distributions
      ]
    )

  def _simulate_trajectories(
    self,
    initial_positions: torch.Tensor,
    initial_velocities: torch.Tensor,
  ) -> TorchTennisTrajectoryBatch:
    count = len(initial_positions)
    return simulate_tennis_trajectories_warp_fused(
      initial_positions.contiguous(),
      initial_velocities.contiguous(),
      ball_mass_kg=torch.full(
        (count,), PHYSICS.ball.mass_kg, dtype=self.dtype, device=self.device
      ),
      court_restitution=torch.full(
        (count,),
        PHYSICS.court.restitution,
        dtype=self.dtype,
        device=self.device,
      ),
      tangent_speed_retention=torch.full(
        (count,), self.tangent_retention, dtype=self.dtype, device=self.device
      ),
      drag_coefficient=torch.full(
        (count,),
        PHYSICS.ball.drag_coefficient,
        dtype=self.dtype,
        device=self.device,
      ),
      dt=INCOMING_TRAJECTORY_DT,
      horizon_s=INCOMING_TRAJECTORY_HORIZON_S,
      net_x=self.court.net_x_m,
      net_half_width=self.court.net_half_width_m,
    )

  def _match_trajectories(
    self,
    trajectories: TorchTennisTrajectoryBatch,
    motion_targets_w: torch.Tensor,
  ) -> Any:
    return match_tennis_trajectories_to_motions_torch(
      trajectories,
      motion_targets_w,
      self.minimum_contact_times,
      self.nominal_contact_times,
      maximum_distance=INCOMING_MATCH_MAXIMUM_DISTANCE_M,
      motion_chunk_size=len(self.distributions),
      contact_time_offset_s=INCOMING_MATCH_DEADLINE_OFFSET_S,
      hierarchical_top_k=min(
        INCOMING_MATCH_HIERARCHICAL_TOP_K, len(self.distributions)
      ),
      hierarchical_coarse_neighbor_radius=(INCOMING_MATCH_HIERARCHICAL_NEIGHBOR_RADIUS),
    )

  @staticmethod
  def _first_bounce_measurements(
    trajectories: TorchTennisTrajectoryBatch,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover continuous first-impact position/time from the pre-impact state."""
    bounce_mask = trajectories.bounce_counts >= 1
    valid = bounce_mask.any(dim=1)
    first_indices = bounce_mask.to(torch.int8).argmax(dim=1)
    previous_indices = (first_indices - 1).clamp_min(0)
    rows = torch.arange(len(first_indices), device=trajectories.positions.device)
    previous_position = trajectories.positions[rows, previous_indices]
    previous_velocity = trajectories.velocities[rows, previous_indices]

    drag_force, _ = tennis_ball_aerodynamic_wrench_torch(
      previous_velocity,
      torch.zeros_like(previous_velocity),
    )
    gravity = torch.tensor(
      (0.0, 0.0, -9.81),
      dtype=previous_velocity.dtype,
      device=previous_velocity.device,
    )
    acceleration = gravity + drag_force / PHYSICS.ball.mass_kg
    next_velocity = previous_velocity + acceleration * INCOMING_TRAJECTORY_DT
    unconstrained_next_position = (
      previous_position + next_velocity * INCOMING_TRAJECTORY_DT
    )
    denominator = (
      previous_position[:, 2] - unconstrained_next_position[:, 2]
    ).clamp_min(1.0e-7)
    fraction = ((previous_position[:, 2] - PHYSICS.ball.radius_m) / denominator).clamp(
      0.0, 1.0
    )
    impact_position = (
      previous_position
      + (unconstrained_next_position - previous_position) * fraction[:, None]
    )
    impact_time = (
      trajectories.times[previous_indices] + fraction * INCOMING_TRAJECTORY_DT
    )
    return impact_position, impact_time, valid

  @torch.no_grad()
  def sample_manual(
    self,
    pelvis_pos_w: np.ndarray,
    pelvis_rotation_w: np.ndarray,
    *,
    launch_position_w: np.ndarray,
    bounce_position_xy_w: np.ndarray,
    bounce_time_s: float,
    solver_iterations: int,
    maximum_initial_speed_m_s: float,
  ) -> _IncomingBallPlan:
    """Shoot at a requested first bounce without imposing a motion match."""
    _ = pelvis_rotation_w
    ground_z = PHYSICS.ball.radius_m
    displacement = np.array(
      [
        bounce_position_xy_w[0] - launch_position_w[0],
        bounce_position_xy_w[1] - launch_position_w[1],
        ground_z - launch_position_w[2],
      ],
      dtype=np.float64,
    )
    velocity = displacement / bounce_time_s
    velocity[2] += 0.5 * 9.81 * bounce_time_s
    velocity = torch.tensor(velocity, dtype=self.dtype, device=self.device)
    position = torch.tensor(launch_position_w, dtype=self.dtype, device=self.device)
    target = torch.tensor(
      (bounce_position_xy_w[0], bounce_position_xy_w[1], bounce_time_s),
      dtype=self.dtype,
      device=self.device,
    )
    finite_difference = MANUAL_SHOOTING_FINITE_DIFFERENCE_M_S

    for _ in range(solver_iterations):
      candidate_velocities = velocity.repeat(4, 1)
      candidate_velocities[1:, :] += (
        torch.eye(3, dtype=self.dtype, device=self.device) * finite_difference
      )
      trajectories = self._simulate_trajectories(
        position.repeat(4, 1), candidate_velocities
      )
      impact_positions, impact_times, valid = self._first_bounce_measurements(
        trajectories
      )
      if not bool(valid.all().item()):
        raise RuntimeError(
          "Manual launch solver did not produce a first bounce within the "
          f"{INCOMING_TRAJECTORY_HORIZON_S:.1f}s planning horizon."
        )
      measurements = torch.cat((impact_positions[:, :2], impact_times[:, None]), dim=1)
      residual = measurements[0] - target
      jacobian = (measurements[1:] - measurements[0]).T / finite_difference
      correction = torch.linalg.lstsq(jacobian, -residual).solution
      correction_norm = torch.linalg.vector_norm(correction)
      if correction_norm > MANUAL_SHOOTING_MAX_CORRECTION_M_S:
        correction *= MANUAL_SHOOTING_MAX_CORRECTION_M_S / correction_norm
      velocity += correction

    final_trajectories = self._simulate_trajectories(position[None], velocity[None])
    impact_positions, impact_times, valid_bounce = self._first_bounce_measurements(
      final_trajectories
    )
    impact_position = impact_positions[0]
    impact_time = impact_times[0]
    position_error = torch.linalg.vector_norm(impact_position[:2] - target[:2]).item()
    time_error = abs(float(impact_time.item()) - bounce_time_s)
    speed = float(torch.linalg.vector_norm(velocity).item())
    if not bool(valid_bounce[0].item()):
      raise RuntimeError("Manual launch solver produced no first bounce.")
    if speed > maximum_initial_speed_m_s:
      raise RuntimeError(
        f"Manual launch requires {speed:.3f}m/s, exceeding "
        f"manual_max_initial_speed_mps={maximum_initial_speed_m_s:.3f}."
      )
    if (
      position_error > MANUAL_BOUNCE_POSITION_TOLERANCE_M
      or time_error > MANUAL_BOUNCE_TIME_TOLERANCE_S
    ):
      raise RuntimeError(
        "Manual launch shooting did not converge: "
        f"position_error={position_error:.3f}m, time_error={time_error:.3f}s."
      )

    positions = final_trajectories.positions[0]
    net_x = self.court.net_x_m
    crosses_net = (positions[:-1, 0] > net_x) & (positions[1:, 0] <= net_x)
    crossing_indices = crosses_net.nonzero(as_tuple=False).flatten()
    if len(crossing_indices) == 0:
      raise RuntimeError("Manual launch does not cross the tennis-net plane.")
    crossing_index = int(crossing_indices[0].item())
    before = positions[crossing_index]
    after = positions[crossing_index + 1]
    alpha = (before[0] - net_x) / (before[0] - after[0]).clamp_min(1.0e-7)
    crossing_position = before + alpha * (after - before)
    clears_net = (
      abs(float(crossing_position[1].item())) <= self.court.net_half_width_m
      and float(crossing_position[2].item())
      > PHYSICS.court.net_height_m + PHYSICS.ball.radius_m
      and bool(final_trajectories.net_cleared[0].item())
    )
    if not clears_net:
      raise RuntimeError(
        "Manual launch does not clear the physical net: "
        f"crossing={crossing_position.cpu().numpy().round(3).tolist()}m, "
        f"required_center_height>"
        f"{PHYSICS.court.net_height_m + PHYSICS.ball.radius_m:.3f}m."
      )

    # The Intent Student does not observe a motion goal or contact time. Keep a
    # closest-approach timestamp only for UI and episode bookkeeping; it does
    # not accept or reject a manual launch.
    post_bounce_indices = (
      (final_trajectories.bounce_counts[0] == 1).nonzero(as_tuple=False).flatten()
    )
    if len(post_bounce_indices) == 0:
      raise RuntimeError("Manual launch has no post-first-bounce trajectory segment.")
    post_bounce_positions = positions[post_bounce_indices]
    pelvis_position = torch.as_tensor(
      pelvis_pos_w, dtype=self.dtype, device=self.device
    )
    closest_offset = torch.linalg.vector_norm(
      post_bounce_positions - pelvis_position, dim=-1
    ).argmin()
    closest_index = int(post_bounce_indices[closest_offset].item())
    closest_position = positions[closest_index]
    closest_time = float(final_trajectories.times[closest_index].item())
    return _IncomingBallPlan(
      # Retained only for existing episode metadata. Manual Intent inference
      # does not consume this distribution's motion goal.
      distribution=self.distributions[0],
      launch_position_w=position.cpu().numpy().astype(np.float64),
      launch_velocity_w=velocity.cpu().numpy().astype(np.float64),
      target_position_w=closest_position.cpu().numpy().astype(np.float64),
      contact_time_s=closest_time,
      match_distance_m=math.nan,
      valid=True,
      attempts=solver_iterations,
      motion_matched=False,
      first_bounce_position_w=(impact_position.cpu().numpy().astype(np.float64)),
      first_bounce_time_s=float(impact_time.item()),
    )

  @torch.no_grad()
  def sample(
    self, pelvis_pos_w: np.ndarray, pelvis_rotation_w: np.ndarray
  ) -> _IncomingBallPlan:
    motion_targets_w = torch.tensor(
      self._aligned_motion_targets_w(
        self.distributions, pelvis_pos_w, pelvis_rotation_w
      ),
      dtype=self.dtype,
      device=self.device,
    )
    root_positions_xy = torch.tensor(
      pelvis_pos_w[None, :2], dtype=self.dtype, device=self.device
    )
    best_distance = math.inf
    best: (
      tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, float, bool] | None
    ) = None
    attempt_offset = 0
    total_attempts = INCOMING_MATCH_ATTEMPTS * INCOMING_MATCH_RETRY_ROUNDS

    while attempt_offset < total_attempts:
      attempt_count = min(
        INCOMING_MATCH_ADAPTIVE_BATCH_SIZE, total_attempts - attempt_offset
      )
      initial_position, initial_velocity = sample_root_directed_tennis_launches_torch(
        root_positions_xy.expand(attempt_count, -1),
        launch_position_min=self.position_min,
        launch_position_max=self.position_max,
        horizontal_speed_range_m_s=self.court.horizontal_speed_range_m_s,
        horizontal_angle_half_width_deg=INCOMING_HORIZONTAL_ANGLE_HALF_WIDTH_DEG,
        net_crossing_height_range_m=INCOMING_NET_CROSSING_HEIGHT_RANGE_M,
        maximum_initial_speed_m_s=INCOMING_MAXIMUM_INITIAL_SPEED_M_S,
        net_x_m=self.court.net_x_m,
      )
      trajectories = self._simulate_trajectories(initial_position, initial_velocity)
      match = self._match_trajectories(trajectories, motion_targets_w)
      selected = int(torch.argmin(match.distances).item())
      selected_distance = float(match.distances[selected].item())
      selected_valid = bool(match.valid[selected].item())
      if selected_distance < best_distance:
        best_distance = selected_distance
        best = (
          selected,
          int(match.motion_ids[selected].item()),
          initial_position[selected].clone(),
          initial_velocity[selected].clone(),
          match.target_positions[selected].clone(),
          float(match.contact_times[selected].item()),
          selected_valid,
        )
        best_attempt = attempt_offset + selected + 1
      attempt_offset += attempt_count
      if best is not None and best[-1]:
        break

    if best is None or not math.isfinite(best_distance):
      raise RuntimeError(
        "WarpRootDirected planning found no post-bounce motion intercept."
      )
    (
      _,
      motion_id,
      launch_position,
      launch_velocity,
      target_position,
      contact_time,
      valid,
    ) = best
    return _IncomingBallPlan(
      distribution=self.distributions[motion_id],
      launch_position_w=launch_position.cpu().numpy().astype(np.float64),
      launch_velocity_w=launch_velocity.cpu().numpy().astype(np.float64),
      target_position_w=target_position.cpu().numpy().astype(np.float64),
      contact_time_s=contact_time,
      match_distance_m=best_distance,
      valid=valid,
      attempts=best_attempt,
    )

  @torch.no_grad()
  def sample_failure(
    self,
    pelvis_pos_w: np.ndarray,
    *,
    no_net_fraction: float,
    xy_distance_threshold_m: float,
  ) -> _IncomingBallPlan:
    """Sample one intentional no-net or over-net-but-far trajectory."""
    root_positions_xy = torch.tensor(
      pelvis_pos_w[None, :2], dtype=self.dtype, device=self.device
    )
    prefer_no_net = bool(torch.rand((), device=self.device) < no_net_fraction)
    attempts_used = 0

    for _ in range(INCOMING_MATCH_RETRY_ROUNDS):
      attempt_count = INCOMING_MATCH_ATTEMPTS
      roots = root_positions_xy.expand(attempt_count, -1)
      initial_position, initial_velocity = sample_root_directed_tennis_launches_torch(
        roots,
        launch_position_min=self.position_min,
        launch_position_max=self.position_max,
        horizontal_speed_range_m_s=self.court.horizontal_speed_range_m_s,
        horizontal_angle_half_width_deg=INCOMING_HORIZONTAL_ANGLE_HALF_WIDTH_DEG,
        net_crossing_height_range_m=INCOMING_NET_CROSSING_HEIGHT_RANGE_M,
        maximum_initial_speed_m_s=INCOMING_MAXIMUM_INITIAL_SPEED_M_S,
        net_x_m=self.court.net_x_m,
      )

      no_net_count = round(attempt_count * no_net_fraction)
      if no_net_fraction > 0.0:
        no_net_count = max(1, no_net_count)
      if no_net_fraction < 1.0:
        no_net_count = min(attempt_count - 1, no_net_count)
      no_net_candidates = (
        torch.arange(attempt_count, device=self.device) < no_net_count
      )
      far_candidates = ~no_net_candidates

      if torch.any(far_candidates):
        initial_velocity[far_candidates] = (
          retarget_tennis_launches_to_miss_roots_torch(
            initial_position[far_candidates],
            initial_velocity[far_candidates],
            roots[far_candidates],
            minimum_xy_distance_m=xy_distance_threshold_m,
            net_crossing_height_range_m=INCOMING_NET_CROSSING_HEIGHT_RANGE_M,
            maximum_initial_speed_m_s=INCOMING_MAXIMUM_INITIAL_SPEED_M_S,
            net_x_m=self.court.net_x_m,
          )
        )
      if torch.any(no_net_candidates):
        no_net_position = initial_position[no_net_candidates]
        no_net_velocity = initial_velocity[no_net_candidates]
        time_to_net = (
          (self.court.net_x_m - no_net_position[:, 0])
          / no_net_velocity[:, 0].clamp(max=-1.0e-6)
        ).clamp_min(1.0e-6)
        crossing_height = 0.5 * PHYSICS.court.net_height_m
        required_vz = (
          crossing_height
          - no_net_position[:, 2]
          + 0.5 * 9.81 * time_to_net.square()
        ) / time_to_net
        horizontal_speed_sq = torch.sum(no_net_velocity[:, :2].square(), dim=1)
        vertical_speed_cap = torch.sqrt(
          (INCOMING_MAXIMUM_INITIAL_SPEED_M_S**2 - horizontal_speed_sq).clamp_min(
            0.0
          )
        )
        initial_velocity[no_net_candidates, 2] = required_vz.clamp(
          -vertical_speed_cap, vertical_speed_cap
        )

      trajectories = self._simulate_trajectories(initial_position, initial_velocity)
      failure = classify_tennis_failure_trajectories(
        trajectories,
        roots,
        trajectory_xy_distance_threshold_m=xy_distance_threshold_m,
        net_x=self.court.net_x_m,
        net_half_width=self.court.net_half_width_m,
      )
      preferred = (
        failure.did_not_clear_net if prefer_no_net else failure.trajectory_too_far
      )
      eligible = preferred if torch.any(preferred) else failure.failure
      eligible_ids = eligible.nonzero(as_tuple=False).flatten()
      attempts_used += attempt_count
      if len(eligible_ids) == 0:
        continue

      selected = int(
        eligible_ids[torch.randint(len(eligible_ids), (), device=self.device)].item()
      )
      failure_type: Literal["no_net", "too_far"] = (
        "no_net" if bool(failure.did_not_clear_net[selected].item()) else "too_far"
      )
      bounce_count = trajectories.bounce_counts[selected]
      second_bounce_ids = (bounce_count >= 2).nonzero(as_tuple=False).flatten()
      if len(second_bounce_ids) > 0:
        end_index = int(second_bounce_ids[0].item())
      else:
        end_index = len(trajectories.times) - 1
      contact_time = float(trajectories.times[end_index].item())
      target_position = failure.closest_trajectory_positions[selected]
      if not torch.all(torch.isfinite(target_position)):
        target_position = torch.tensor(
          pelvis_pos_w, dtype=self.dtype, device=self.device
        )
      return _IncomingBallPlan(
        distribution=self.distributions[0],
        launch_position_w=initial_position[selected].cpu().numpy().astype(np.float64),
        launch_velocity_w=initial_velocity[selected].cpu().numpy().astype(np.float64),
        target_position_w=target_position.cpu().numpy().astype(np.float64),
        contact_time_s=contact_time,
        match_distance_m=float(
          failure.closest_trajectory_distances[selected].item()
        ),
        valid=False,
        attempts=attempts_used,
        motion_matched=False,
        failure_type=failure_type,
      )

    requested = "no_net" if prefer_no_net else "too_far"
    raise RuntimeError(
      f"Could not sample a {requested} failure trajectory after "
      f"{attempts_used} attempts."
    )


def _build_standalone_model(
  robot_xml: Path,
  physics_dt: float,
  joint_names: tuple[str, ...],
  joint_stiffness: np.ndarray,
  joint_damping: np.ndarray,
  court: _CourtProfile = STANDARD_COURT_PROFILE,
) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(robot_xml))
  if spec.actuators:
    raise ValueError(
      "Standalone robot XML must not contain actuators; they are reconstructed "
      "from Student ONNX metadata."
    )

  # Compile once to resolve inherited joint effort/range defaults from the XML.
  passive_model = spec.compile()
  for name, stiffness, damping in zip(
    joint_names, joint_stiffness, joint_damping, strict=True
  ):
    joint_id = mujoco.mj_name2id(passive_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
      raise ValueError(f"Robot XML is missing policy joint: {name}")
    effort_limit = float(np.max(np.abs(passive_model.jnt_actfrcrange[joint_id])))
    joint_range = passive_model.jnt_range[joint_id]
    actuator = spec.add_actuator(name=name, target=name)
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.set_to_position(kp=float(stiffness), kv=float(damping))
    if effort_limit > 0.0:
      actuator.forcelimited = True
      actuator.forcerange = [-effort_limit, effort_limit]
      actuator.ctrllimited = True
      actuator.ctrlrange = [
        float(joint_range[0] - effort_limit / stiffness),
        float(joint_range[1] + effort_limit / stiffness),
      ]

  spec.worldbody.add_body(name="terrain").add_geom(
    name="terrain",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0.0, 0.0, 0.01],
    rgba=[0.0, 0.0, 0.0, 0.0],
    contype=1,
    conaffinity=1,
  )
  wrist = spec.body("right_wrist_yaw_link")
  if wrist is None:
    raise ValueError("Robot XML has no right_wrist_yaw_link for racket collision.")
  inertia_body = wrist.add_body(name="racket_inertia")
  inertia_body.explicitinertial = True
  inertia_body.mass = RACKET_NOMINAL_MASS_KG
  inertia_body.ipos = RACKET_NOMINAL_COM_WRIST_M
  inertia_body.inertia = RACKET_NOMINAL_INERTIA_KG_M2
  wrist.add_geom(
    name="racket_ball_collision",
    type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    pos=RACKET_COLLISION_POS_WRIST_M,
    quat=RACKET_COLLISION_QUAT_WXYZ,
    size=RACKET_COLLISION_HALF_SIZE_M,
    rgba=[1.0, 0.2, 0.05, 0.35],
    density=0.0,
    contype=1,
    conaffinity=1,
  )
  robot_geom_names = tuple(
    geom.name
    for geom in spec.geoms
    if geom.name != "terrain" and (geom.contype != 0 or geom.conaffinity != 0)
  )
  ball = spec.worldbody.add_body(name="tennis_ball")
  ball.add_joint(name="tennis_ball_freejoint", type=mujoco.mjtJoint.mjJNT_FREE)
  ball.add_geom(
    name="tennis_ball_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[PHYSICS.ball.radius_m, 0.0, 0.0],
    mass=PHYSICS.ball.mass_kg,
    friction=PHYSICS.ball.geom_friction,
    solref=PHYSICS.ball.geom_solref,
    solimp=PHYSICS.ball.geom_solimp,
    rgba=PHYSICS.ball.rgba,
    contype=1,
    conaffinity=1,
  )
  launch_min = np.asarray(court.launch_position_min, dtype=np.float64)
  launch_max = np.asarray(court.launch_position_max, dtype=np.float64)
  spec.worldbody.add_geom(
    name="incoming_ball_launch_region",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=((launch_min + launch_max) * 0.5).tolist(),
    size=((launch_max - launch_min) * 0.5).tolist(),
    rgba=[1.0, 0.82, 0.0, 0.16],
    contype=0,
    conaffinity=0,
  )
  spec.worldbody.add_geom(
    name="landing_target_marker",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    pos=[0.0, 0.0, 0.012],
    size=[0.5, 0.008, 0.0],
    rgba=[0.1, 0.95, 0.25, 0.28],
    contype=0,
    conaffinity=0,
  )
  spec.worldbody.add_geom(
    name="landing_target_center",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    pos=[0.0, 0.0, 0.022],
    size=[0.06, 0.012, 0.0],
    rgba=[0.05, 1.0, 0.2, 0.9],
    contype=0,
    conaffinity=0,
  )
  spec.worldbody.add_geom(
    name="manual_first_bounce_marker",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    pos=[0.0, 0.0, 0.012],
    size=[0.28, 0.01, 0.0],
    rgba=[1.0, 0.82, 0.0, 0.0],
    contype=0,
    conaffinity=0,
  )

  spec.option.timestep = physics_dt
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.iterations = 10
  spec.option.ls_iterations = 20
  spec.nconmax = max(spec.nconmax, 200)
  spec.njmax = max(spec.njmax, 500)
  attach_tennis_court_spec(
    spec,
    mode="physical",
    court_contact_damping=contact_damping_for_restitution(PHYSICS.court.restitution),
    net_x_m=court.net_x_m,
    net_half_width_m=court.net_half_width_m,
    racket_contact_damping=STANDALONE_RACKET_CONTACT_DAMPING,
    ball_geom_name="tennis_ball_geom",
    racket_geom_name="racket_ball_collision",
    robot_geom_names=robot_geom_names,
  )

  model = spec.compile()
  return model


class _SweetSpotFKEstimator:
  """Joint FK with a once-per-control-step world velocity difference and EMA."""

  def __init__(self, model: mujoco.MjModel, site_id: int, smoothing: float):
    if not 0.0 < smoothing <= 1.0:
      raise ValueError("sweet_spot_velocity_smoothing must lie in (0, 1].")
    self.model = model
    self.data = mujoco.MjData(model)
    self.site_id = site_id
    self.smoothing = smoothing
    self.reset()

  def reset(self) -> None:
    self.previous_time = None
    self.position = np.zeros(3)
    self.velocity = np.zeros(3)

  def update(self, qpos: np.ndarray, time_s: float) -> tuple[np.ndarray, np.ndarray]:
    if self.previous_time == time_s:
      return self.position.copy(), self.velocity.copy()
    self.data.qpos[:] = qpos
    mujoco.mj_kinematics(self.model, self.data)
    position = self.data.site_xpos[self.site_id].copy()
    if self.previous_time is not None:
      dt = time_s - self.previous_time
      if 1.0e-4 <= dt <= 0.2:
        raw = (position - self.position) / dt
        self.velocity = self.smoothing * raw + (1 - self.smoothing) * self.velocity
    self.position = position
    self.previous_time = time_s
    return self.position.copy(), self.velocity.copy()


class _StandaloneStudentSimulation:
  """Raw MuJoCo Student control loop, isolated from the training task."""

  def __init__(self, cfg: StudentOnnxPlayConfig) -> None:
    if cfg.physics_dt <= 0.0 or cfg.control_dt <= 0.0:
      raise ValueError("physics_dt and control_dt must be positive.")
    decimation = round(cfg.control_dt / cfg.physics_dt)
    if decimation < 1 or not math.isclose(
      decimation * cfg.physics_dt, cfg.control_dt, abs_tol=1.0e-9
    ):
      raise ValueError("control_dt must be an integer multiple of physics_dt.")
    if cfg.ball_release_lead_s < 0.0:
      raise ValueError("ball_release_lead_s must be non-negative.")
    if cfg.landing_target_std_x < 0.0 or cfg.landing_target_std_y < 0.0:
      raise ValueError("Landing-target standard deviations must be non-negative.")
    if not 0.0 <= cfg.failure_trajectory_probability <= 1.0:
      raise ValueError("failure_trajectory_probability must lie in [0, 1].")
    if not 0.0 <= cfg.failure_trajectory_no_net_fraction <= 1.0:
      raise ValueError("failure_trajectory_no_net_fraction must lie in [0, 1].")
    if cfg.failure_trajectory_xy_distance_threshold_m <= 0.0:
      raise ValueError("failure_trajectory_xy_distance_threshold_m must be positive.")
    landing_target_values = (
      cfg.landing_target_x,
      cfg.landing_target_y,
      cfg.landing_target_min_x,
    )
    if not all(
      value is None or math.isfinite(value) for value in landing_target_values
    ):
      raise ValueError("Landing-target coordinates must be finite when provided.")
    manual_values = (
      cfg.manual_launch_x,
      cfg.manual_launch_y,
      cfg.manual_launch_z,
      cfg.manual_bounce_x,
      cfg.manual_bounce_y,
      cfg.manual_bounce_time_s,
      cfg.manual_bounce_time_min_s,
      cfg.manual_bounce_time_max_s,
      cfg.manual_max_initial_speed_mps,
    )
    if not all(math.isfinite(value) for value in manual_values):
      raise ValueError("Manual launch parameters must be finite.")
    if cfg.manual_launch_z <= PHYSICS.ball.radius_m:
      raise ValueError("manual_launch_z must be above the court surface.")
    if not 0.0 < cfg.manual_bounce_time_s < INCOMING_TRAJECTORY_HORIZON_S:
      raise ValueError(
        "manual_bounce_time_s must lie inside the trajectory planning horizon."
      )
    if not (
      0.0
      < cfg.manual_bounce_time_min_s
      <= cfg.manual_bounce_time_max_s
      < INCOMING_TRAJECTORY_HORIZON_S
    ):
      raise ValueError(
        "manual_bounce_time_min_s/max_s must define a positive ordered range "
        "inside the trajectory planning horizon."
      )
    if cfg.manual_solver_iterations <= 0:
      raise ValueError("manual_solver_iterations must be positive.")
    if cfg.manual_max_initial_speed_mps <= 0.0:
      raise ValueError("manual_max_initial_speed_mps must be positive.")
    if cfg.manual_mouse_select:
      if not cfg.manual_mode:
        raise ValueError("manual_mouse_select requires manual_mode.")
      court = _resolve_court_profile(cfg.court_preset)
      if not _is_robot_half_singles_point(
        cfg.manual_bounce_x, cfg.manual_bounce_y, court
      ):
        raise ValueError(
          "manual_mouse_select requires its initial bounce point inside the "
          "robot-side singles court."
        )

    self.cfg = cfg
    self.court = _resolve_court_profile(cfg.court_preset)
    self.decimation = decimation
    self.rng = np.random.default_rng(cfg.seed)
    self.policy = _StudentOnnxPolicy(cfg.onnx_policy_file)
    metadata = self.policy.metadata
    joint_names_raw = metadata.get("joint_names")
    if not joint_names_raw:
      raise ValueError("Student ONNX is missing joint_names metadata.")
    self.joint_names = tuple(name.strip() for name in joint_names_raw.split(","))
    if len(self.joint_names) != ACTION_DIM:
      raise ValueError(
        f"Expected {ACTION_DIM} policy joints, got {len(self.joint_names)}."
      )
    self.default_joint_pos = _parse_float_metadata(metadata, "default_joint_pos")
    self.action_scale = _parse_float_metadata(metadata, "action_scale")
    self.joint_stiffness = _parse_float_metadata(metadata, "joint_stiffness")
    self.joint_damping = _parse_float_metadata(metadata, "joint_damping")
    metadata_arrays = (
      self.default_joint_pos,
      self.action_scale,
      self.joint_stiffness,
      self.joint_damping,
    )
    if any(values.shape != (ACTION_DIM,) for values in metadata_arrays):
      raise ValueError("Student ONNX joint metadata fields must contain 29 values.")

    robot_xml, self.goal_distributions = _load_goal_distributions(cfg.motion_config)
    self.incoming_ball_planner = (
      _RootDirectedIncomingBallPlanner(
        self.goal_distributions, seed=cfg.seed, court=self.court
      )
      if self.policy.observation_dim in INTENT_OBSERVATION_DIMS
      else None
    )
    if cfg.manual_mode and self.incoming_ball_planner is None:
      raise ValueError(
        "manual_mode requires an Intent Student ONNX incoming-ball planner."
      )
    self.model = _build_standalone_model(
      robot_xml,
      cfg.physics_dt,
      self.joint_names,
      self.joint_stiffness,
      self.joint_damping,
      self.court,
    )
    self.data = mujoco.MjData(self.model)

    self.joint_qpos_adr = np.array(
      [
        self.model.jnt_qposadr[
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in self.joint_names
      ],
      dtype=np.int32,
    )
    self.joint_dof_adr = np.array(
      [
        self.model.jnt_dofadr[
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in self.joint_names
      ],
      dtype=np.int32,
    )
    self.actuator_ids = np.array(
      [
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in self.joint_names
      ],
      dtype=np.int32,
    )
    if np.any(self.actuator_ids < 0):
      raise ValueError("Robot model does not actuate every policy joint.")

    self.pelvis_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    self.imu_sensor_id = self._required_id(mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    self.sweet_spot_site_id = self._required_id(
      mujoco.mjtObj.mjOBJ_SITE, "racket_sweet_spot"
    )
    # Replace the XML's larger yellow marker with the labeled viewer-only marker.
    self.model.site_rgba[self.sweet_spot_site_id, 3] = 0.0
    self._sweet_spot_fk = (
      _SweetSpotFKEstimator(
        self.model, self.sweet_spot_site_id, cfg.sweet_spot_velocity_smoothing
      )
      if cfg.sweet_spot_state_source == "fk" else None
    )
    self.ball_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "tennis_ball")
    self.ball_joint_id = self._required_id(
      mujoco.mjtObj.mjOBJ_JOINT, "tennis_ball_freejoint"
    )
    self.ball_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "tennis_ball_geom")
    self.racket_geom_id = self._required_id(
      mujoco.mjtObj.mjOBJ_GEOM, "racket_ball_collision"
    )
    self.manual_bounce_marker_geom_id = self._required_id(
      mujoco.mjtObj.mjOBJ_GEOM, "manual_first_bounce_marker"
    )
    self.landing_target_marker_geom_id = self._required_id(
      mujoco.mjtObj.mjOBJ_GEOM, "landing_target_marker"
    )
    self.landing_target_center_geom_id = self._required_id(
      mujoco.mjtObj.mjOBJ_GEOM, "landing_target_center"
    )
    self.ball_qpos_adr = int(self.model.jnt_qposadr[self.ball_joint_id])
    self.ball_dof_adr = int(self.model.jnt_dofadr[self.ball_joint_id])

    self.last_action = np.zeros(ACTION_DIM, dtype=np.float64)
    self.elapsed_s = 0.0
    self.actual_hit_time_s: float | None = None
    self.ball_released = False
    self.manual_launch_waiting = False
    self.episode_index = -1
    self.goal_sample: _GoalSample
    self.incoming_ball_plan: _IncomingBallPlan | None = None
    self._planned_launch_velocity_w = np.zeros(3, dtype=np.float64)
    self.manual_bounce_position_xy_w = np.array(
      (cfg.manual_bounce_x, cfg.manual_bounce_y), dtype=np.float64
    )
    self.current_manual_bounce_time_s = cfg.manual_bounce_time_s
    self.landing_target_position_w = np.zeros(3, dtype=np.float64)
    self.landing_target_position_startup = np.zeros(3, dtype=np.float64)
    self.episode_duration_s = 0.0
    self._state_history: deque[np.ndarray] = deque(maxlen=INTENT_HISTORY_BUFFER_LENGTH)
    self._ball_position_history_b: deque[np.ndarray] = deque(
      maxlen=INTENT_HISTORY_BUFFER_LENGTH
    )
    self._ball_velocity_history_b: deque[np.ndarray] = deque(
      maxlen=INTENT_HISTORY_BUFFER_LENGTH
    )
    self._ball_position_history_w: deque[np.ndarray] = deque(
      maxlen=INTENT_HISTORY_BUFFER_LENGTH
    )
    self._ball_velocity_history_w: deque[np.ndarray] = deque(
      maxlen=INTENT_HISTORY_BUFFER_LENGTH
    )
    raw_ball_buffer_length = max(1, cfg.ball_observation_latency_max_steps + 1)
    self._raw_ball_position_history_w: deque[np.ndarray] = deque(
      maxlen=raw_ball_buffer_length
    )
    self._raw_ball_velocity_history_w: deque[np.ndarray] = deque(
      maxlen=raw_ball_buffer_length
    )
    self._ball_valid_history: deque[np.ndarray] = deque(
      maxlen=INTENT_HISTORY_BUFFER_LENGTH
    )
    self._ball_age_history: deque[np.ndarray] = deque(
      maxlen=INTENT_HISTORY_BUFFER_LENGTH
    )
    self._perceived_ball_position_w: np.ndarray | None = None
    self._perceived_ball_velocity_w: np.ndarray | None = None
    self._ball_observation_age_s = 0.0
    self._ball_dropout_remaining_steps = 0
    self._last_history_time: float | None = None
    self._ground_impact_pending = False
    self._ground_impact_age = 0
    self._ground_impact_velocity_w = np.zeros(3, dtype=np.float64)
    self._viewer_model_dirty = True
    self._update_manual_bounce_marker()
    self.reset()

  def _update_manual_bounce_marker(self) -> None:
    self.model.geom_pos[self.manual_bounce_marker_geom_id, :2] = (
      self.manual_bounce_position_xy_w
    )
    self.model.geom_pos[self.manual_bounce_marker_geom_id, 2] = 0.012
    self.model.geom_rgba[self.manual_bounce_marker_geom_id, 3] = (
      0.42 if self.cfg.manual_mode else 0.0
    )
    self._viewer_model_dirty = True

  def _update_landing_target_marker(self) -> None:
    visible = self.cfg.show_landing_target and self.policy.observation_dim in (
      LANDING_OBSERVATION_DIM,
      *INTENT_OBSERVATION_DIMS,
    )
    for geom_id, z_w, alpha in (
      (self.landing_target_marker_geom_id, 0.012, 0.28),
      (self.landing_target_center_geom_id, 0.022, 0.9),
    ):
      self.model.geom_pos[geom_id, :2] = self.landing_target_position_w[:2]
      self.model.geom_pos[geom_id, 2] = z_w
      self.model.geom_rgba[geom_id, 3] = alpha if visible else 0.0
    self._viewer_model_dirty = True

  def sync_viewer(self, viewer: mujoco.viewer.Handle) -> None:
    with viewer.lock():
      _draw_sweet_point(
        viewer.user_scn,
        self.data.site_xpos[self.sweet_spot_site_id],
        visible=self.cfg.show_sweet_point,
      )
    # State-only sync omits model edits such as moving world-attached markers.
    viewer.sync(state_only=not self._viewer_model_dirty)
    self._viewer_model_dirty = False

  def set_manual_bounce_target(self, x_w: float, y_w: float) -> None:
    if not self.cfg.manual_mode:
      raise RuntimeError("Manual bounce targets require manual_mode.")
    if not _is_robot_half_singles_point(x_w, y_w, self.court):
      raise ValueError(
        f"Manual bounce point ({x_w:.3f}, {y_w:.3f}) is outside the "
        "robot-side singles court."
      )
    self.manual_bounce_position_xy_w[:] = (x_w, y_w)
    self._update_manual_bounce_marker()
    mujoco.mj_forward(self.model, self.data)

  def set_manual_launch_waiting(self, waiting: bool) -> None:
    """Keep Student inference active while holding the next ball at launch."""
    if not self.cfg.manual_mode:
      raise RuntimeError("Manual launch waiting requires manual_mode.")
    self.manual_launch_waiting = waiting
    if not waiting:
      self.model.geom_contype[self.ball_geom_id] = 1
      self.model.geom_conaffinity[self.ball_geom_id] = 1
      return

    self.elapsed_s = 0.0
    self.actual_hit_time_s = None
    self.ball_released = False
    self.model.geom_contype[self.ball_geom_id] = 0
    self.model.geom_conaffinity[self.ball_geom_id] = 0
    self._set_ball_state(
      np.array(
        (
          self.cfg.manual_launch_x,
          self.cfg.manual_launch_y,
          self.cfg.manual_launch_z,
        ),
        dtype=np.float64,
      )
    )
    mujoco.mj_forward(self.model, self.data)

  def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(self.model, object_type, name)
    if object_id < 0:
      raise ValueError(f"MuJoCo model is missing {object_type.name}: {name}")
    return object_id

  def _sample_distribution(self) -> _GoalDistribution:
    weights = np.array(
      [max(0.0, item.sampling_weight) for item in self.goal_distributions],
      dtype=np.float64,
    )
    probabilities = weights / weights.sum()
    index = int(self.rng.choice(len(self.goal_distributions), p=probabilities))
    return self.goal_distributions[index]

  def _sample_contact_time(self, distribution: _GoalDistribution) -> float:
    nominal = distribution.strike_frame / distribution.fps
    minimum = nominal / MAX_CONTACT_SPEEDUP
    count = math.ceil((nominal - minimum) / CONTACT_TIME_STEP_S) + 1
    sample_index = int(self.rng.integers(0, count))
    return min(minimum + sample_index * CONTACT_TIME_STEP_S, nominal)

  def _set_startup_anchor(
    self, position_w: np.ndarray, rotation_w: np.ndarray
  ) -> None:
    yaw = math.atan2(rotation_w[1, 0], rotation_w[0, 0])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    self.startup_anchor_pos_w = position_w.copy()
    self.startup_anchor_yaw_rotation_w = np.array(
      [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )

  def _landing_target_to_world(self, target_startup: np.ndarray) -> np.ndarray:
    target_w = (
      self.startup_anchor_pos_w
      + self.startup_anchor_yaw_rotation_w @ target_startup
    )
    # Landing targets lie on the court, not at pelvis height.
    target_w[2] = target_startup[2]
    return target_w

  def _sample_landing_target(self) -> np.ndarray:
    target_x = (
      self.court.landing_target_xy[0]
      if self.cfg.landing_target_x is None
      else self.cfg.landing_target_x
    )
    target_y = (
      self.court.landing_target_xy[1]
      if self.cfg.landing_target_y is None
      else self.cfg.landing_target_y
    )
    mean = np.array([target_x, target_y, 0.0], dtype=np.float64)
    std = np.array(
      [self.cfg.landing_target_std_x, self.cfg.landing_target_std_y]
    )
    sample = mean.copy()
    sample[:2] += self.rng.normal(size=2) * std
    minimum_x = self.cfg.landing_target_min_x
    if minimum_x is None:
      return sample

    sample_w = self._landing_target_to_world(sample)
    mean_w = self._landing_target_to_world(mean)
    rotation = self.startup_anchor_yaw_rotation_w[:2, :2]
    variance_x = float(np.sum(rotation[0] ** 2 * std**2))
    covariance_xy = float(np.sum(rotation[0] * rotation[1] * std**2))
    std_x = math.sqrt(variance_x)
    alpha = (minimum_x - mean_w[0]) / max(std_x, 1.0e-12)
    cdf_lower = 0.5 * (1.0 + math.erf(alpha / math.sqrt(2.0)))
    probability = cdf_lower + (1.0 - cdf_lower) * self.rng.random()
    eps = np.finfo(np.float64).eps
    quantile = NormalDist().inv_cdf(float(np.clip(probability, eps, 1.0 - eps)))
    sampled_x_w = max(mean_w[0] + std_x * quantile, minimum_x)
    # Match training's world-X truncated Gaussian and conditional-Y residual.
    sample_w[1] += covariance_xy / max(variance_x, 1.0e-12) * (
      sampled_x_w - sample_w[0]
    )
    sample_w[0] = sampled_x_w
    sample = self.startup_anchor_yaw_rotation_w.T @ (
      sample_w - self.startup_anchor_pos_w
    )
    sample[2] = 0.0
    return sample

  def _log_landing_target(self, context: str) -> None:
    if self.policy.observation_dim not in (
      LANDING_OBSERVATION_DIM,
      *INTENT_OBSERVATION_DIMS,
    ):
      return
    print(
      f"[LANDING TARGET SAMPLE] {context} "
      f"startup_frame={np.round(self.landing_target_position_startup[:2], 4).tolist()} "
      f"world={np.round(self.landing_target_position_w[:2], 4).tolist()} "
      f"std=({self.cfg.landing_target_std_x:.3f}, "
      f"{self.cfg.landing_target_std_y:.3f})m "
      f"min_world_x={self.cfg.landing_target_min_x}",
      flush=True,
    )

  def _sample_goal(
    self, distribution: _GoalDistribution | None = None
  ) -> _GoalSample:
    if distribution is None:
      distribution = self._sample_distribution()
    position_local = (
      distribution.position_mean + self.rng.normal(size=3) * distribution.position_std
    )
    velocity_local = (
      distribution.velocity_mean + self.rng.normal(size=3) * distribution.velocity_std
    )
    orientation_rpy = (
      distribution.orientation_rpy_mean
      + self.rng.normal(size=3) * distribution.orientation_rpy_std
    )

    pelvis_pos = self.data.xpos[self.pelvis_body_id].copy()
    pelvis_rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3).copy()
    pelvis_quat = self.data.xquat[self.pelvis_body_id].copy()
    position_world = pelvis_pos + pelvis_rotation @ position_local
    velocity_world = pelvis_rotation @ velocity_local
    orientation_world = _quat_mul(pelvis_quat, _quat_from_rpy(orientation_rpy))
    return _GoalSample(
      distribution=distribution,
      position_local=position_local,
      position_world=position_world,
      velocity_world=velocity_world,
      orientation_world=orientation_world,
      contact_time_s=self._sample_contact_time(distribution),
    )

  def _set_robot_state_from_motion_frame0(
    self, distribution: _GoalDistribution
  ) -> None:
    """Match the training reset while aligning frame-0 XY/yaw to world zero."""
    if distribution.initial_joint_position.shape != (ACTION_DIM,) or (
      distribution.initial_joint_velocity.shape != (ACTION_DIM,)
    ):
      raise ValueError(
        f"{distribution.source_file} must contain {ACTION_DIM} frame-0 joints."
      )

    reference_quat = distribution.initial_anchor_quaternion_w
    reference_yaw = math.atan2(
      2.0
      * (
        reference_quat[0] * reference_quat[3]
        + reference_quat[1] * reference_quat[2]
      ),
      1.0
      - 2.0
      * (
        reference_quat[2] * reference_quat[2]
        + reference_quat[3] * reference_quat[3]
      ),
    )
    yaw_alignment = _quat_from_rpy(
      np.array([0.0, 0.0, -reference_yaw], dtype=np.float64)
    )
    self.data.qpos[:3] = (
      0.0,
      0.0,
      distribution.initial_anchor_position_w[2],
    )
    self.data.qpos[3:7] = _quat_mul(yaw_alignment, reference_quat)
    self.data.qpos[self.joint_qpos_adr] = distribution.initial_joint_position
    self.data.qvel[:3] = _quat_apply(
      yaw_alignment, distribution.initial_anchor_linear_velocity_w
    )
    self.data.qvel[3:6] = _quat_apply(
      yaw_alignment, distribution.initial_anchor_angular_velocity_w
    )
    self.data.qvel[self.joint_dof_adr] = distribution.initial_joint_velocity

  def _set_ball_state(
    self, position: np.ndarray, linear_velocity: np.ndarray | None = None
  ) -> None:
    self.data.xfrc_applied[self.ball_body_id] = 0.0
    self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 3] = position
    self.data.qpos[self.ball_qpos_adr + 3 : self.ball_qpos_adr + 7] = (
      1.0,
      0.0,
      0.0,
      0.0,
    )
    self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 6] = 0.0
    if linear_velocity is not None:
      self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 3] = linear_velocity

  def _goal_from_incoming_plan(
    self,
    plan: _IncomingBallPlan,
    pelvis_pos: np.ndarray,
    pelvis_rotation: np.ndarray,
    pelvis_quat: np.ndarray,
  ) -> _GoalSample:
    distribution = plan.distribution
    orientation_rpy = (
      distribution.orientation_rpy_mean
      + self.rng.normal(size=3) * distribution.orientation_rpy_std
    )
    return _GoalSample(
      distribution=distribution,
      position_local=pelvis_rotation.T @ (plan.target_position_w - pelvis_pos),
      position_world=plan.target_position_w.copy(),
      velocity_world=pelvis_rotation @ distribution.velocity_mean,
      orientation_world=_quat_mul(pelvis_quat, _quat_from_rpy(orientation_rpy)),
      contact_time_s=plan.contact_time_s,
    )

  def _sample_incoming_plan(
    self,
    pelvis_pos_w: np.ndarray,
    pelvis_rotation_w: np.ndarray,
  ) -> _IncomingBallPlan:
    if self.incoming_ball_planner is None:
      raise RuntimeError("Incoming-ball planner is not available.")
    if not self.cfg.manual_mode:
      if (
        self.episode_index >= 0
        and self.rng.random() < self.cfg.failure_trajectory_probability
      ):
        return self.incoming_ball_planner.sample_failure(
          pelvis_pos_w,
          no_net_fraction=self.cfg.failure_trajectory_no_net_fraction,
          xy_distance_threshold_m=(
            self.cfg.failure_trajectory_xy_distance_threshold_m
          ),
        )
      return self.incoming_ball_planner.sample(pelvis_pos_w, pelvis_rotation_w)
    attempts = MANUAL_TIME_SAMPLE_ATTEMPTS if self.cfg.manual_random_bounce_time else 1
    last_error: RuntimeError | None = None
    for _ in range(attempts):
      bounce_time_s = _sample_manual_bounce_time(self.cfg, self.rng)
      try:
        plan = self.incoming_ball_planner.sample_manual(
          pelvis_pos_w,
          pelvis_rotation_w,
          launch_position_w=np.array(
            (
              self.cfg.manual_launch_x,
              self.cfg.manual_launch_y,
              self.cfg.manual_launch_z,
            ),
            dtype=np.float64,
          ),
          bounce_position_xy_w=np.array(
            self.manual_bounce_position_xy_w,
            dtype=np.float64,
          ),
          bounce_time_s=bounce_time_s,
          solver_iterations=self.cfg.manual_solver_iterations,
          maximum_initial_speed_m_s=self.cfg.manual_max_initial_speed_mps,
        )
      except RuntimeError as error:
        last_error = error
        continue
      self.current_manual_bounce_time_s = bounce_time_s
      return plan
    raise RuntimeError(
      f"No valid manual launch after {attempts} bounce-time sample(s): {last_error}"
    )

  def _log_incoming_plan(self, context: str) -> None:
    plan = self.incoming_ball_plan
    if plan is None:
      return
    if plan.failure_type is not None:
      print(
        f"[FAILURE BALL SAMPLE] {context} type={plan.failure_type} "
        f"p0={np.round(plan.launch_position_w, 3).tolist()} "
        f"v0={np.round(plan.launch_velocity_w, 3).tolist()} "
        f"min_xy_distance={plan.match_distance_m:.3f}m "
        f"duration={plan.contact_time_s:.3f}s attempts={plan.attempts}",
        flush=True,
      )
      return
    if plan.motion_matched:
      print(
        f"[INCOMING BALL SAMPLE] {context} "
        f"file={plan.distribution.source_file.name} "
        f"contact_time={plan.contact_time_s:.3f}s "
        f"p0={np.round(plan.launch_position_w, 3).tolist()} "
        f"v0={np.round(plan.launch_velocity_w, 3).tolist()} "
        f"target={np.round(plan.target_position_w, 3).tolist()} "
        f"match_error={plan.match_distance_m:.3f}m "
        f"valid={plan.valid} attempts={plan.attempts}",
        flush=True,
      )
      return

    print(
      f"[MANUAL INCOMING BALL] {context} motion_match=disabled "
      f"timing_constraint=disabled "
      f"p0={np.round(plan.launch_position_w, 3).tolist()} "
      f"v0={np.round(plan.launch_velocity_w, 3).tolist()} "
      f"closest_approach={np.round(plan.target_position_w, 3).tolist()} "
      f"closest_approach_time={plan.contact_time_s:.3f}s "
      f"attempts={plan.attempts}",
      flush=True,
    )
    if plan.first_bounce_position_w is not None:
      print(
        "[MANUAL FIRST BOUNCE] "
        f"requested={np.round(self.manual_bounce_position_xy_w, 3).tolist()}m "
        f"at {self.current_manual_bounce_time_s:.3f}s "
        f"solved={np.round(plan.first_bounce_position_w[:2], 3).tolist()}m "
        f"at {plan.first_bounce_time_s:.3f}s",
        flush=True,
      )

  def reset(self) -> None:
    mujoco.mj_resetData(self.model, self.data)
    if self._sweet_spot_fk is not None:
      self._sweet_spot_fk.reset()
    # Establish the common origin used by the trajectory planner. The selected
    # motion frame-0 state is applied immediately after planning.
    self.data.qpos[:7] = (0.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0)
    self.data.qpos[self.joint_qpos_adr] = self.default_joint_pos
    self.data.qvel[:] = 0.0
    mujoco.mj_forward(self.model, self.data)
    if self.incoming_ball_planner is None:
      self.incoming_ball_plan = None
      distribution = self._sample_distribution()
    else:
      provisional_pos = self.data.xpos[self.pelvis_body_id].copy()
      provisional_rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3).copy()
      self.incoming_ball_plan = self._sample_incoming_plan(
        provisional_pos, provisional_rotation
      )
      distribution = self.incoming_ball_plan.distribution

    self._set_robot_state_from_motion_frame0(distribution)
    mujoco.mj_forward(self.model, self.data)
    startup_pos = self.data.xpos[self.pelvis_body_id].copy()
    startup_rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3).copy()
    startup_quat = self.data.xquat[self.pelvis_body_id].copy()
    self._set_startup_anchor(startup_pos, startup_rotation)

    if self.incoming_ball_plan is None:
      self.goal_sample = self._sample_goal(distribution)
      initial_ball_position = self.goal_sample.position_world
      self._planned_launch_velocity_w.fill(0.0)
    else:
      self.goal_sample = self._goal_from_incoming_plan(
        self.incoming_ball_plan,
        startup_pos,
        startup_rotation,
        startup_quat,
      )
      initial_ball_position = self.incoming_ball_plan.launch_position_w
      self._planned_launch_velocity_w[:] = self.incoming_ball_plan.launch_velocity_w
    if self.policy.observation_dim in (
      LANDING_OBSERVATION_DIM,
      *INTENT_OBSERVATION_DIMS,
    ):
      self.landing_target_position_startup = self._sample_landing_target()
      self.landing_target_position_w = self._landing_target_to_world(
        self.landing_target_position_startup
      )
    self._update_landing_target_marker()
    self._set_ball_state(initial_ball_position)
    if self.incoming_ball_plan is None:
      self.model.geom_contype[self.ball_geom_id] = 0
      self.model.geom_conaffinity[self.ball_geom_id] = 0
    else:
      self.model.geom_contype[self.ball_geom_id] = 1
      self.model.geom_conaffinity[self.ball_geom_id] = 1
    self.data.ctrl[self.actuator_ids] = distribution.initial_joint_position
    self.last_action.fill(0.0)
    self.elapsed_s = 0.0
    self.actual_hit_time_s = None
    self.ball_released = False
    self._ground_impact_pending = False
    self._ground_impact_age = 0
    self._ground_impact_velocity_w.fill(0.0)
    self._state_history.clear()
    self._ball_position_history_b.clear()
    self._ball_velocity_history_b.clear()
    self._ball_position_history_w.clear()
    self._ball_velocity_history_w.clear()
    self._raw_ball_position_history_w.clear()
    self._raw_ball_velocity_history_w.clear()
    self._ball_valid_history.clear()
    self._ball_age_history.clear()
    self._perceived_ball_position_w = None
    self._perceived_ball_velocity_w = None
    self._ball_observation_age_s = 0.0
    self._ball_dropout_remaining_steps = 0
    self._last_history_time = None
    self.episode_duration_s = (
      self.goal_sample.contact_time_s
      if self.incoming_ball_plan is not None
      and self.incoming_ball_plan.failure_type is not None
      else max(
        self.goal_sample.distribution.frames / self.goal_sample.distribution.fps,
        self.goal_sample.contact_time_s + 1.5,
      )
    )
    self.episode_index += 1
    mujoco.mj_forward(self.model, self.data)
    if self.incoming_ball_plan is None:
      print(
        f"[GOAL SAMPLE] episode={self.episode_index} "
        f"file={self.goal_sample.distribution.source_file.name} "
        f"clip={self.goal_sample.distribution.clip} "
        f"position_local={np.round(self.goal_sample.position_local, 4).tolist()} "
        f"velocity_world={np.round(self.goal_sample.velocity_world, 4).tolist()} "
        f"landing_target={np.round(self.landing_target_position_w[:2], 4).tolist()} "
        f"contact_time={self.goal_sample.contact_time_s:.3f}s",
        flush=True,
      )
    else:
      self._log_incoming_plan(f"episode={self.episode_index}")
      self._log_landing_target(f"episode={self.episode_index}")

  def start_next_strike_cycle(self) -> None:
    """Plan another incoming ball while preserving the physical G1 state."""
    if self.incoming_ball_planner is None:
      raise RuntimeError("Continuous strikes require the incoming-ball planner.")

    mujoco.mj_forward(self.model, self.data)
    pelvis_pos = self.data.xpos[self.pelvis_body_id].copy()
    pelvis_rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3).copy()
    pelvis_quat = self.data.xquat[self.pelvis_body_id].copy()
    self.incoming_ball_plan = self._sample_incoming_plan(pelvis_pos, pelvis_rotation)
    self.goal_sample = self._goal_from_incoming_plan(
      self.incoming_ball_plan,
      pelvis_pos,
      pelvis_rotation,
      pelvis_quat,
    )
    self._planned_launch_velocity_w[:] = self.incoming_ball_plan.launch_velocity_w
    self.landing_target_position_startup = self._sample_landing_target()
    self.landing_target_position_w = self._landing_target_to_world(
      self.landing_target_position_startup
    )
    self._update_landing_target_marker()

    self._set_ball_state(self.incoming_ball_plan.launch_position_w)
    self.model.geom_contype[self.ball_geom_id] = 1
    self.model.geom_conaffinity[self.ball_geom_id] = 1
    self.elapsed_s = 0.0
    self.actual_hit_time_s = None
    self.ball_released = False
    self._ground_impact_pending = False
    self._ground_impact_age = 0
    self._ground_impact_velocity_w.fill(0.0)
    self.episode_duration_s = (
      self.goal_sample.contact_time_s
      if self.incoming_ball_plan.failure_type is not None
      else max(
        self.goal_sample.distribution.frames / self.goal_sample.distribution.fps,
        self.goal_sample.contact_time_s + 1.5,
      )
    )
    self.episode_index += 1
    mujoco.mj_forward(self.model, self.data)

    self._log_incoming_plan(
      f"cycle={self.episode_index} continuous=True "
      f"root_xy={np.round(pelvis_pos[:2], 3).tolist()}"
    )
    self._log_landing_target(
      f"cycle={self.episode_index} continuous=True"
    )

  @property
  def time_remaining_s(self) -> float:
    return max(0.0, self.goal_sample.contact_time_s - self.elapsed_s)

  @property
  def ball_speed_mps(self) -> float:
    return float(
      np.linalg.norm(self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 3])
    )

  def _object_linear_velocity_w(
    self, object_type: mujoco.mjtObj, object_id: int
  ) -> np.ndarray:
    velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
      self.model,
      self.data,
      object_type,
      object_id,
      velocity,
      0,
    )
    return velocity[3:].copy()

  @staticmethod
  def _gather_strided_history(
    history: deque[np.ndarray], sample_lags: tuple[int, ...]
  ) -> np.ndarray:
    """Match MJLab CircularBuffer lag lookup and first-sample backfill."""
    if not history:
      raise RuntimeError("Cannot gather an empty observation history.")
    values = tuple(history)
    newest = len(values) - 1
    return np.concatenate([values[max(0, newest - lag)] for lag in sample_lags])

  @staticmethod
  def _projected_gravity_b(
    pelvis_rotation: np.ndarray, gravity_w: np.ndarray
  ) -> np.ndarray:
    gravity_norm = np.linalg.norm(gravity_w)
    if gravity_norm <= 0.0:
      raise RuntimeError("MuJoCo gravity must be non-zero.")
    return pelvis_rotation.T @ (gravity_w / gravity_norm)

  @staticmethod
  def _startup_heading_b(pelvis_rotation: np.ndarray) -> np.ndarray:
    """Episode-start +X expressed in the current pelvis yaw frame."""
    yaw = math.atan2(pelvis_rotation[1, 0], pelvis_rotation[0, 0])
    return np.array([math.cos(yaw), -math.sin(yaw)], dtype=np.float64)

  def _perceived_ball_state(
    self,
    true_position_w: np.ndarray,
    true_velocity_w: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Apply the robust policy's transport latency and hold-last dropout model."""
    self._raw_ball_position_history_w.append(true_position_w.copy())
    self._raw_ball_velocity_history_w.append(true_velocity_w.copy())
    if self.policy.observation_dim != INTENT_ROBUST_OBSERVATION_DIM:
      return true_position_w, true_velocity_w, True, 0.0

    latency_steps = int(
      self.rng.integers(
        self.cfg.ball_observation_latency_min_steps,
        self.cfg.ball_observation_latency_max_steps + 1,
      )
    )
    raw_positions = tuple(self._raw_ball_position_history_w)
    raw_velocities = tuple(self._raw_ball_velocity_history_w)
    delayed_index = max(0, len(raw_positions) - 1 - latency_steps)
    measured_position_w = raw_positions[delayed_index].copy()
    measured_velocity_w = raw_velocities[delayed_index].copy()
    measured_position_w += self.rng.normal(
      0.0, self.cfg.ball_observation_position_noise_std, size=3
    )
    measured_velocity_w += self.rng.normal(
      0.0, self.cfg.ball_observation_velocity_noise_std, size=3
    )

    if (
      self._ball_dropout_remaining_steps == 0
      and self.rng.random()
      < self.cfg.ball_observation_dropout_start_probability
    ):
      self._ball_dropout_remaining_steps = int(
        self.rng.integers(
          self.cfg.ball_observation_dropout_duration_min_steps,
          self.cfg.ball_observation_dropout_duration_max_steps + 1,
        )
      )

    dropped = self._ball_dropout_remaining_steps > 0
    if (
      not dropped
      or self._perceived_ball_position_w is None
      or self._perceived_ball_velocity_w is None
    ):
      self._perceived_ball_position_w = measured_position_w
      self._perceived_ball_velocity_w = measured_velocity_w
      self._ball_observation_age_s = latency_steps * self.cfg.control_dt
    else:
      self._ball_observation_age_s += self.cfg.control_dt
    if dropped:
      self._ball_dropout_remaining_steps -= 1
    return (
      self._perceived_ball_position_w.copy(),
      self._perceived_ball_velocity_w.copy(),
      not dropped,
      self._ball_observation_age_s,
    )

  def _record_intent_history(
    self,
    pelvis_pos: np.ndarray,
    pelvis_rotation: np.ndarray,
    base_ang_vel: np.ndarray,
    joint_pos_rel: np.ndarray,
    joint_vel_rel: np.ndarray,
    projected_gravity_b: np.ndarray,
  ) -> None:
    """Record one control-step sample using the training observation layouts."""
    history_time = float(self.data.time)
    if self._last_history_time is not None and math.isclose(
      history_time, self._last_history_time, abs_tol=1.0e-12
    ):
      return

    true_ball_position_w = self.data.xpos[self.ball_body_id].copy()
    true_ball_velocity_w = self._object_linear_velocity_w(
      mujoco.mjtObj.mjOBJ_BODY, self.ball_body_id
    )
    ball_position_w, ball_velocity_w, ball_valid, ball_age_s = (
      self._perceived_ball_state(true_ball_position_w, true_ball_velocity_w)
    )
    self._state_history.append(
      np.concatenate(
        [
          base_ang_vel,
          projected_gravity_b,
          joint_pos_rel,
          joint_vel_rel,
          self.last_action,
        ]
      )
    )
    self._ball_position_history_b.append(
      pelvis_rotation.T @ (ball_position_w - pelvis_pos)
    )
    self._ball_velocity_history_b.append(pelvis_rotation.T @ ball_velocity_w)
    self._ball_position_history_w.append(ball_position_w)
    self._ball_velocity_history_w.append(ball_velocity_w)
    self._ball_valid_history.append(np.array([float(ball_valid)], dtype=np.float64))
    self._ball_age_history.append(np.array([ball_age_s], dtype=np.float64))
    self._last_history_time = history_time

  def _intent_observation(
    self,
    pelvis_pos: np.ndarray,
    pelvis_rotation: np.ndarray,
    base_ang_vel: np.ndarray,
    joint_pos_rel: np.ndarray,
    joint_vel_rel: np.ndarray,
  ) -> np.ndarray:
    projected_gravity_b = self._projected_gravity_b(
      pelvis_rotation, self.model.opt.gravity
    )
    self._record_intent_history(
      pelvis_pos,
      pelvis_rotation,
      base_ang_vel,
      joint_pos_rel,
      joint_vel_rel,
      projected_gravity_b,
    )

    ball_position_history_b = self._gather_strided_history(
      self._ball_position_history_b, INTENT_HISTORY_LAGS
    )
    ball_velocity_history_b = self._gather_strided_history(
      self._ball_velocity_history_b, INTENT_HISTORY_LAGS
    )
    sweet_spot_position_b = pelvis_rotation.T @ (
      self.data.site_xpos[self.sweet_spot_site_id] - pelvis_pos
    )
    sweet_spot_velocity_b = pelvis_rotation.T @ self._object_linear_velocity_w(
      mujoco.mjtObj.mjOBJ_SITE, self.sweet_spot_site_id
    )
    if self._sweet_spot_fk is not None:
      fk_qpos = self.data.qpos.copy()
      # Match the pose/joint measurements available to the deployment controller.
      fk_qpos[:3] = pelvis_pos
      fk_qpos[3:7] = self.data.xquat[self.pelvis_body_id]
      position_w, velocity_w = self._sweet_spot_fk.update(fk_qpos, float(self.data.time))
      sweet_spot_position_b = pelvis_rotation.T @ (position_w - pelvis_pos)
      sweet_spot_velocity_b = pelvis_rotation.T @ velocity_w

    # Preserve the ObservationManager insertion order of the 133-D Student group.
    current_student = np.concatenate(
      [
        base_ang_vel,
        joint_pos_rel,
        joint_vel_rel,
        self.last_action,
        self.landing_target_position_startup[:2],
        ball_position_history_b,
        ball_velocity_history_b,
        sweet_spot_velocity_b,
        projected_gravity_b,
        sweet_spot_position_b,
        self._startup_heading_b(pelvis_rotation),
      ]
    )
    current_dim = INTENT_CURRENT_OBSERVATION_DIM
    if self.policy.observation_dim == INTENT_GLOBAL_ROOT_OBSERVATION_DIM:
      current_student = np.concatenate([current_student, pelvis_pos])
      current_dim += 3
    state_history = self._gather_strided_history(
      self._state_history, INTENT_HISTORY_LAGS
    )

    selected_positions_w = self._gather_strided_history(
      self._ball_position_history_w, INTENT_HISTORY_LAGS
    ).reshape(INTENT_BALL_HISTORY_STEPS, 3)
    selected_velocities_w = self._gather_strided_history(
      self._ball_velocity_history_w, INTENT_HISTORY_LAGS
    ).reshape(INTENT_BALL_HISTORY_STEPS, 3)
    intent_positions_b = (
      pelvis_rotation.T @ (selected_positions_w - pelvis_pos).T
    ).T.reshape(-1)
    intent_velocities_b = (pelvis_rotation.T @ selected_velocities_w.T).T.reshape(-1)
    intent_ball_parts = [intent_positions_b, intent_velocities_b]
    ball_token_dim = INTENT_BALL_TOKEN_DIM
    if self.policy.observation_dim == INTENT_ROBUST_OBSERVATION_DIM:
      intent_ball_parts.extend(
        (
          self._gather_strided_history(
            self._ball_valid_history, INTENT_HISTORY_LAGS
          ),
          self._gather_strided_history(self._ball_age_history, INTENT_HISTORY_LAGS),
        )
      )
      ball_token_dim = INTENT_ROBUST_BALL_TOKEN_DIM
    intent_ball_history = np.concatenate(intent_ball_parts)

    if current_student.shape != (current_dim,):
      raise RuntimeError(
        f"Invalid current Intent Student observation: {current_student.shape}"
      )
    if state_history.shape != (INTENT_STATE_HISTORY_STEPS * INTENT_STATE_TOKEN_DIM,):
      raise RuntimeError(f"Invalid Intent state history: {state_history.shape}")
    if intent_ball_history.shape != (
      INTENT_BALL_HISTORY_STEPS * ball_token_dim,
    ):
      raise RuntimeError(f"Invalid Intent ball history: {intent_ball_history.shape}")
    return np.concatenate([current_student, state_history, intent_ball_history])

  def observation(self) -> np.ndarray:
    pelvis_pos = self.data.xpos[self.pelvis_body_id]
    pelvis_rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
    pelvis_quat = self.data.xquat[self.pelvis_body_id]
    target_position_b = pelvis_rotation.T @ (
      self.goal_sample.position_world - pelvis_pos
    )
    target_orientation_b = _quat_mul(
      _quat_conjugate(pelvis_quat), self.goal_sample.orientation_world
    )
    task_goal = np.concatenate(
      [
        target_position_b,
        self.goal_sample.velocity_world,
        target_orientation_b,
        np.array([self.goal_sample.contact_time_s]),
      ]
    )
    if task_goal.shape != (TARGET_GOAL_DIM,):
      raise RuntimeError(f"Invalid task goal shape: {task_goal.shape}")

    sensor_adr = int(self.model.sensor_adr[self.imu_sensor_id])
    sensor_dim = int(self.model.sensor_dim[self.imu_sensor_id])
    base_ang_vel = self.data.sensordata[sensor_adr : sensor_adr + sensor_dim]
    joint_pos_rel = self.data.qpos[self.joint_qpos_adr] - self.default_joint_pos
    joint_vel_rel = self.data.qvel[self.joint_dof_adr]
    if self.policy.observation_dim in INTENT_OBSERVATION_DIMS:
      observation = self._intent_observation(
        pelvis_pos,
        pelvis_rotation,
        base_ang_vel,
        joint_pos_rel,
        joint_vel_rel,
      ).astype(np.float32)
      if (
        observation.shape != (self.policy.observation_dim,)
        or not np.isfinite(observation).all()
      ):
        raise RuntimeError(f"Invalid Intent Student observation: {observation.shape}")
      return observation

    observation_terms = [
      task_goal,
      np.array([self.time_remaining_s]),
      base_ang_vel,
      joint_pos_rel,
      joint_vel_rel,
      self.last_action,
    ]
    if self.policy.observation_dim == LANDING_OBSERVATION_DIM:
      landing_target_b = pelvis_rotation.T @ (
        self.landing_target_position_w - pelvis_pos
      )
      ball_position_b = pelvis_rotation.T @ (
        self.data.xpos[self.ball_body_id] - pelvis_pos
      )
      ball_velocity_b = pelvis_rotation.T @ self._object_linear_velocity_w(
        mujoco.mjtObj.mjOBJ_BODY, self.ball_body_id
      )
      sweet_spot_velocity_b = pelvis_rotation.T @ self._object_linear_velocity_w(
        mujoco.mjtObj.mjOBJ_SITE, self.sweet_spot_site_id
      )
      projected_gravity_b = self._projected_gravity_b(
        pelvis_rotation, self.model.opt.gravity
      )
      observation_terms.extend(
        [
          landing_target_b[:2],
          ball_position_b,
          ball_velocity_b,
          sweet_spot_velocity_b,
          pelvis_pos,
          projected_gravity_b,
        ]
      )

    observation = np.concatenate(observation_terms).astype(np.float32)
    if (
      observation.shape != (self.policy.observation_dim,)
      or not np.isfinite(observation).all()
    ):
      raise RuntimeError(f"Invalid Student observation: {observation.shape}")
    return observation

  def _apply_ball_aerodynamics(self) -> None:
    velocity = self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 6]
    force_w, torque_w = tennis_ball_aerodynamic_wrench_numpy(
      velocity[:3], velocity[3:6]
    )
    self.data.xfrc_applied[self.ball_body_id, :3] = force_w
    self.data.xfrc_applied[self.ball_body_id, 3:6] = torque_w

  def _apply_explicit_ground_rebound(self) -> None:
    """Mirror the nominal play-mode rebound controller used by training."""
    velocity = self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 3]
    position_z = float(self.data.xpos[self.ball_body_id, 2])
    ground_z = PHYSICS.ball.radius_m

    if self._ground_impact_pending:
      self._ground_impact_age += 1
    elif velocity[2] < -0.05 and position_z <= ground_z + 0.03:
      self._ground_impact_pending = True
      self._ground_impact_age = 0
      self._ground_impact_velocity_w[:] = velocity
      height_to_ground = max(position_z - ground_z, 0.0)
      self._ground_impact_velocity_w[2] = -math.sqrt(
        float(velocity[2] ** 2) + 2.0 * 9.81 * height_to_ground
      )

    if self._ground_impact_pending and position_z <= ground_z + 0.002:
      self.data.qpos[self.ball_qpos_adr + 2] = ground_z + 1.0e-4
      self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 2] = (
        self._ground_impact_velocity_w[:2]
        * self.incoming_ball_planner.tangent_retention
      )
      self.data.qvel[self.ball_dof_adr + 2] = (
        -self._ground_impact_velocity_w[2] * PHYSICS.court.restitution
      )
      self._ground_impact_pending = False
      self._ground_impact_age = 0
    elif self._ground_impact_age > 20:
      self._ground_impact_pending = False
      self._ground_impact_age = 0

  def _ball_racket_contact(self) -> bool:
    for contact_index in range(self.data.ncon):
      contact = self.data.contact[contact_index]
      if (
        contact.geom1 == self.ball_geom_id and contact.geom2 == self.racket_geom_id
      ) or (
        contact.geom1 == self.racket_geom_id and contact.geom2 == self.ball_geom_id
      ):
        return True
    return False

  def step(self) -> _StepReport:
    if self.manual_launch_waiting:
      action = self.policy(self.observation())
      self.data.ctrl[self.actuator_ids] = (
        self.default_joint_pos + self.action_scale * action
      )
      self.last_action[:] = action
      launch_position = np.array(
        (
          self.cfg.manual_launch_x,
          self.cfg.manual_launch_y,
          self.cfg.manual_launch_z,
        ),
        dtype=np.float64,
      )
      for _ in range(self.decimation):
        self._set_ball_state(launch_position)
        mujoco.mj_step(self.model, self.data)
      self._set_ball_state(launch_position)
      mujoco.mj_forward(self.model, self.data)
      return _StepReport(
        elapsed_s=0.0,
        sampled_strike_time_s=self.goal_sample.contact_time_s,
        horizontal_speed_mps=0.0,
        actual_hit_time_s=None,
        released=False,
        episode_complete=False,
      )

    if (
      self.incoming_ball_plan is None
      and not self.ball_released
      and self.time_remaining_s <= self.cfg.ball_release_lead_s
    ):
      self.ball_released = True
      self.model.geom_contype[self.ball_geom_id] = 1
      self.model.geom_conaffinity[self.ball_geom_id] = 1
      self._set_ball_state(self.goal_sample.position_world)
      print(
        f"[BALL RELEASE] elapsed={self.elapsed_s:.3f}s "
        f"time_remaining={self.time_remaining_s:.3f}s",
        flush=True,
      )

    action = self.policy(self.observation())
    self.data.ctrl[self.actuator_ids] = (
      self.default_joint_pos + self.action_scale * action
    )
    self.last_action[:] = action

    if self.incoming_ball_plan is not None and not self.ball_released:
      self._set_ball_state(
        self.incoming_ball_plan.launch_position_w,
        self._planned_launch_velocity_w,
      )
      self.ball_released = True
      if self.incoming_ball_plan.motion_matched:
        print(
          f"[INCOMING BALL LAUNCH] elapsed={self.elapsed_s:.3f}s "
          f"time_remaining={self.time_remaining_s:.3f}s",
          flush=True,
        )
      else:
        print(
          f"[INCOMING BALL LAUNCH] elapsed={self.elapsed_s:.3f}s "
          "timing_constraint=disabled",
          flush=True,
        )

    racket_contact = False
    for _ in range(self.decimation):
      if not self.ball_released:
        self._set_ball_state(self.goal_sample.position_world)
      self._apply_ball_aerodynamics()
      mujoco.mj_step(self.model, self.data)
      if self.incoming_ball_plan is not None:
        self._apply_explicit_ground_rebound()
        racket_contact |= self._ball_racket_contact()

    self.elapsed_s += self.cfg.control_dt
    horizontal_speed = float(
      np.linalg.norm(self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 2])
    )
    strike_detected = (
      racket_contact
      if self.incoming_ball_plan is not None
      else horizontal_speed >= self.cfg.horizontal_speed_threshold_mps
    )
    if self.ball_released and self.actual_hit_time_s is None and strike_detected:
      self.actual_hit_time_s = self.elapsed_s
      if (
        self.incoming_ball_plan is not None
        and not self.incoming_ball_plan.motion_matched
      ):
        print(
          "[STRIKE] timing_constraint=disabled "
          f"actual={self.actual_hit_time_s:.3f}s "
          f"horizontal_speed={horizontal_speed:.3f}m/s",
          flush=True,
        )
      else:
        timing_error = self.actual_hit_time_s - self.goal_sample.contact_time_s
        status = (
          "ON TIME"
          if abs(timing_error) <= self.cfg.on_time_tolerance_s
          else "EARLY"
          if timing_error < 0.0
          else "LATE"
        )
        print(
          f"[STRIKE TIMING] status={status} "
          f"sampled={self.goal_sample.contact_time_s:.3f}s "
          f"actual={self.actual_hit_time_s:.3f}s error={timing_error:+.3f}s "
          f"horizontal_speed={horizontal_speed:.3f}m/s",
          flush=True,
        )

    return _StepReport(
      elapsed_s=self.elapsed_s,
      sampled_strike_time_s=self.goal_sample.contact_time_s,
      horizontal_speed_mps=horizontal_speed,
      actual_hit_time_s=self.actual_hit_time_s,
      released=self.ball_released,
      episode_complete=self.elapsed_s + 1.0e-9 >= self.episode_duration_s,
    )


class _StrikeTimingPlot:
  """Live timeline for sampled strike time and detected physical-ball motion."""

  def __init__(self, cfg: StudentOnnxPlayConfig) -> None:
    import matplotlib.pyplot as plt

    self.cfg = cfg
    self._plt = plt
    self._history_time: deque[float] = deque(maxlen=5000)
    self._history_speed: deque[float] = deque(maxlen=5000)
    self._last_actual_time: float | None = None
    self._steps_since_draw = 0
    self._steps_per_draw = max(1, round(1.0 / (cfg.timing_plot_fps * cfg.control_dt)))

    plt.ion()
    figure, axes = plt.subplots(2, 1, figsize=(16.0, 9.5), sharex=True)
    self.figure = figure
    self.timeline_axis, self.speed_axis = axes
    manager = getattr(figure.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
      manager.set_window_title("Student Strike Timing")

    self.timeline_axis.axhline(0.0, color="#9ca3af", linewidth=1.0)
    self.target_line = self.timeline_axis.axvline(
      0.0,
      color="#2563eb",
      linewidth=2.2,
      label=(
        "closest approach (diagnostic)" if cfg.manual_mode else "sampled strike time"
      ),
    )
    self.actual_line = self.timeline_axis.axvline(
      float("nan"),
      color="#f97316",
      linewidth=2.2,
      label="detected strike",
    )
    self.current_line = self.timeline_axis.axvline(
      0.0, color="#4b5563", linestyle=":", linewidth=1.2, label="current time"
    )
    self.status_text = self.timeline_axis.text(
      0.02,
      0.84,
      "",
      transform=self.timeline_axis.transAxes,
      fontsize=11 * UI_FONT_SCALE,
      fontweight="bold",
      va="top",
    )
    self.speed_line = self.speed_axis.plot(
      [], [], color="#f97316", linewidth=1.6, label="horizontal speed"
    )[0]
    self.speed_axis.axhline(
      cfg.horizontal_speed_threshold_mps,
      color="#dc2626",
      linestyle="--",
      linewidth=1.2,
      label=f"threshold ({cfg.horizontal_speed_threshold_mps:.2f} m/s)",
    )
    self.tolerance_patch = None
    self.timeline_axis.set_ylim(-0.6, 0.6)
    self.timeline_axis.set_yticks([])
    label_font_size = 10 * UI_FONT_SCALE
    self.timeline_axis.set_ylabel("events", fontsize=label_font_size)
    self.speed_axis.set_ylabel("speed [m/s]", fontsize=label_font_size)
    self.speed_axis.set_xlabel("time since reset [s]", fontsize=label_font_size)
    for axis in axes:
      axis.grid(True, alpha=0.22)
      axis.tick_params(axis="both", labelsize=10 * UI_FONT_SCALE)
      axis.legend(loc="upper right", fontsize=10 * UI_FONT_SCALE)
    figure.tight_layout()
    plt.show(block=False)

  def reset(self, simulation: _StandaloneStudentSimulation) -> None:
    target_time = simulation.goal_sample.contact_time_s
    self._history_time.clear()
    self._history_speed.clear()
    self._history_time.append(0.0)
    self._history_speed.append(0.0)
    self._last_actual_time = None
    self.target_line.set_xdata([target_time, target_time])
    self.actual_line.set_xdata([float("nan"), float("nan")])
    self.current_line.set_xdata([0.0, 0.0])
    if self.tolerance_patch is not None:
      self.tolerance_patch.remove()
    if self.cfg.manual_mode:
      self.status_text.set_text(
        f"TIMING CONSTRAINT DISABLED | closest approach={target_time:.3f}s"
      )
    else:
      self.tolerance_patch = self.timeline_axis.axvspan(
        max(0.0, target_time - self.cfg.on_time_tolerance_s),
        target_time + self.cfg.on_time_tolerance_s,
        color="#2563eb",
        alpha=0.12,
      )
      self.status_text.set_text(
        f"WAITING | target={target_time:.3f}s "
        f"tolerance=+/-{self.cfg.on_time_tolerance_s:.3f}s"
      )
    self.status_text.set_color("#4b5563")
    self._redraw(target_time, 0.0)

  def update(self, report: _StepReport) -> None:
    self._history_time.append(report.elapsed_s)
    self._history_speed.append(report.horizontal_speed_mps)
    if (
      report.actual_hit_time_s is not None
      and report.actual_hit_time_s != self._last_actual_time
    ):
      self._last_actual_time = report.actual_hit_time_s
      if self.cfg.manual_mode:
        self.actual_line.set_xdata([report.actual_hit_time_s, report.actual_hit_time_s])
        self.status_text.set_text(
          f"HIT | timing constraint disabled | actual={report.actual_hit_time_s:.3f}s"
        )
        self.status_text.set_color("#16a34a")
        return
      timing_error = report.actual_hit_time_s - report.sampled_strike_time_s
      on_time = abs(timing_error) <= self.cfg.on_time_tolerance_s
      status = "ON TIME" if on_time else "EARLY" if timing_error < 0.0 else "LATE"
      self.actual_line.set_xdata([report.actual_hit_time_s, report.actual_hit_time_s])
      self.status_text.set_text(
        f"{status} | target={report.sampled_strike_time_s:.3f}s "
        f"actual={report.actual_hit_time_s:.3f}s error={timing_error:+.3f}s"
      )
      self.status_text.set_color("#16a34a" if on_time else "#dc2626")
    elif not self.cfg.manual_mode and (
      report.actual_hit_time_s is None
      and report.elapsed_s > report.sampled_strike_time_s + self.cfg.on_time_tolerance_s
    ):
      self.status_text.set_text(
        f"NO HIT YET | target={report.sampled_strike_time_s:.3f}s "
        f"current={report.elapsed_s:.3f}s"
      )
      self.status_text.set_color("#dc2626")

    self._steps_since_draw += 1
    if self._steps_since_draw >= self._steps_per_draw:
      self._redraw(report.sampled_strike_time_s, report.elapsed_s)
      self._steps_since_draw = 0

  def _redraw(self, target_time: float, elapsed_time: float) -> None:
    if not self._plt.fignum_exists(self.figure.number):
      return
    times = list(self._history_time)
    speeds = list(self._history_speed)
    self.speed_line.set_data(times, speeds)
    self.current_line.set_xdata([elapsed_time, elapsed_time])
    time_max = max(target_time + 0.75, elapsed_time + 0.25, 1.0)
    self.speed_axis.set_xlim(0.0, time_max)
    speed_max = max(max(speeds, default=0.0), self.cfg.horizontal_speed_threshold_mps)
    self.speed_axis.set_ylim(0.0, max(0.5, speed_max * 1.15))
    self.figure.canvas.draw()
    self.figure.canvas.flush_events()

  def close(self) -> None:
    if self._plt.fignum_exists(self.figure.number):
      self._plt.close(self.figure)


class _ManualBounceSelector:
  """Top-down mouse picker for the incoming ball's first bounce."""

  def __init__(self, cfg: StudentOnnxPlayConfig, court: _CourtProfile) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    self._plt = plt
    self.court = court
    self._pending_selection: np.ndarray | None = None
    plt.ion()
    self.figure, self.axis = plt.subplots(figsize=(15.0, 6.5))
    manager = getattr(self.figure.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
      manager.set_window_title("Incoming Ball First-Bounce Selector")

    far_baseline_x = court.net_x_m + court.effective_half_length_m
    court_width = 2.0 * court.effective_singles_half_width_m
    self.axis.add_patch(
      Rectangle(
        (
          court.robot_half_baseline_x_m,
          -court.effective_singles_half_width_m,
        ),
        2.0 * court.effective_half_length_m,
        court_width,
        facecolor="#155e75",
        edgecolor="#f8fafc",
        linewidth=2.0,
        alpha=0.85,
      )
    )
    self.axis.add_patch(
      Rectangle(
        (
          court.robot_half_baseline_x_m,
          -court.effective_singles_half_width_m,
        ),
        court.effective_half_length_m,
        court_width,
        facecolor="#facc15",
        edgecolor="none",
        alpha=0.18,
      )
    )
    self.axis.axvline(court.net_x_m, color="#f8fafc", linewidth=3.0, label="net")
    self.axis.scatter(
      [0.0], [0.0], marker="^", s=180, color="#22c55e", label="G1 start"
    )
    self.axis.scatter(
      [cfg.manual_launch_x],
      [cfg.manual_launch_y],
      marker="o",
      s=120,
      color="#60a5fa",
      label="launch",
    )
    self.selection_marker = self.axis.scatter(
      [cfg.manual_bounce_x],
      [cfg.manual_bounce_y],
      marker="x",
      s=240,
      linewidths=3.0,
      color="#facc15",
      label="selected first bounce",
    )
    self.status_text = self.axis.text(
      0.01,
      1.02,
      "Click inside the yellow robot-side singles court to launch",
      transform=self.axis.transAxes,
      fontsize=14,
      fontweight="bold",
      color="#334155",
    )
    self.axis.set_xlim(court.robot_half_baseline_x_m - 0.5, far_baseline_x + 0.5)
    self.axis.set_ylim(
      -court.effective_singles_half_width_m - 0.7,
      court.effective_singles_half_width_m + 0.7,
    )
    self.axis.set_aspect("equal", adjustable="box")
    self.axis.set_xlabel("world/court x [m]")
    self.axis.set_ylabel("world/court y [m]")
    self.axis.grid(True, alpha=0.18)
    self.axis.legend(loc="upper right")
    self.figure.tight_layout()
    self.figure.canvas.mpl_connect("button_press_event", self._on_click)
    plt.show(block=False)

  def _on_click(self, event: Any) -> None:
    if event.inaxes is not self.axis or event.xdata is None or event.ydata is None:
      return
    x_w = float(event.xdata)
    y_w = float(event.ydata)
    if not _is_robot_half_singles_point(x_w, y_w, self.court):
      self.set_status("Rejected: select inside the yellow robot-side court", error=True)
      return
    self._pending_selection = np.array((x_w, y_w), dtype=np.float64)

  @property
  def is_open(self) -> bool:
    return bool(self._plt.fignum_exists(self.figure.number))

  def poll(self) -> None:
    if self.is_open:
      self.figure.canvas.flush_events()

  def consume_selection(self) -> np.ndarray | None:
    selection = self._pending_selection
    self._pending_selection = None
    return selection

  def show_selection(self, selection: np.ndarray) -> None:
    self.selection_marker.set_offsets(selection.reshape(1, 2))
    self.figure.canvas.draw()

  def set_status(self, message: str, *, error: bool = False) -> None:
    self.status_text.set_text(message)
    self.status_text.set_color("#dc2626" if error else "#334155")
    self.figure.canvas.draw()

  def close(self) -> None:
    if self.is_open:
      self._plt.close(self.figure)


def _draw_sweet_point(
  scene: mujoco.MjvScene, position_w: np.ndarray, *, visible: bool
) -> None:
  # Viewer-only geometry: never part of contacts, mass, or policy observations.
  scene.ngeom = 0
  if not visible or scene.maxgeom < 1:
    return
  marker = scene.geoms[0]
  mujoco.mjv_initGeom(
    marker,
    mujoco.mjtGeom.mjGEOM_SPHERE,
    np.array([0.008, 0.008, 0.008]),
    position_w,
    np.eye(3).reshape(-1),
    np.array([1.0, 0.0, 0.8, 1.0], dtype=np.float32),
  )
  marker.label = "sweet point"
  scene.ngeom = 1


class _ViewerStatusOverlay:
  """Mirror the native MJLab play status overlay for this standalone viewer."""

  def __init__(self, control_dt: float, realtime: bool) -> None:
    self.control_dt = control_dt
    self.realtime = realtime
    self.step_count = 0
    self._stats_frames = 0
    self._stats_steps = 0
    self._stats_last_time = time.perf_counter()
    self._fps = 0.0
    self._actual_realtime = 0.0
    self._capped = False
    self._last_submit_time = 0.0

  def record_step(self) -> None:
    self.step_count += 1
    self._stats_steps += 1

  def record_frame(self) -> None:
    self._stats_frames += 1

  def mark_capped(self) -> None:
    self._capped = True

  def update(
    self, viewer: mujoco.viewer.Handle, *, ball_speed_mps: float = 0.0
  ) -> None:
    now = time.perf_counter()
    elapsed = now - self._stats_last_time
    if elapsed >= 0.5:
      self._fps = self._stats_frames / elapsed
      steps_per_second = self._stats_steps / elapsed
      self._actual_realtime = steps_per_second * self.control_dt
      self._stats_frames = 0
      self._stats_steps = 0
      self._stats_last_time = now

    if self._last_submit_time and now - self._last_submit_time < 0.5:
      return

    capped = " [CAPPED]" if self._capped else ""
    text_1 = "Env\nStep\nStatus\nSpeed\nBall speed\nTarget RT\nActual RT"
    text_2 = (
      f"1/1\n"
      f"{self.step_count}\n"
      f"RUNNING{capped}\n"
      f"1x\n"
      f"{ball_speed_mps:.2f} m/s ({ball_speed_mps * 3.6:.1f} km/h)\n"
      f"1.00x\n"
      f"{self._actual_realtime:.2f}x ({self._fps:.0f} FPS)"
    )
    overlay = (
      mujoco.mjtFontScale.mjFONTSCALE_150.value,
      mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
      text_1,
      text_2,
    )
    viewer.set_texts(overlay)
    self._last_submit_time = now
    self._capped = False


def run_student_onnx(cfg: StudentOnnxPlayConfig) -> None:
  if cfg.render_fps <= 0.0:
    raise ValueError("render_fps must be greater than zero.")
  if cfg.timing_plot and cfg.timing_plot_fps <= 0.0:
    raise ValueError("timing_plot_fps must be greater than zero.")
  if (
    cfg.ball_observation_latency_min_steps < 0
    or cfg.ball_observation_latency_max_steps
    < cfg.ball_observation_latency_min_steps
  ):
    raise ValueError("Ball observation latency steps must be an ordered range.")
  if not 0.0 <= cfg.ball_observation_dropout_start_probability <= 1.0:
    raise ValueError("Ball observation dropout probability must be in [0, 1].")
  if (
    cfg.ball_observation_dropout_duration_min_steps < 1
    or cfg.ball_observation_dropout_duration_max_steps
    < cfg.ball_observation_dropout_duration_min_steps
  ):
    raise ValueError("Ball observation dropout duration must be a positive range.")
  if (
    cfg.ball_observation_position_noise_std < 0.0
    or cfg.ball_observation_velocity_noise_std < 0.0
  ):
    raise ValueError("Ball observation noise standard deviations must be non-negative.")

  simulation = _StandaloneStudentSimulation(cfg)
  intent_mode = simulation.policy.observation_dim in INTENT_OBSERVATION_DIMS
  simulation.model.vis.global_.fovy = 45.0 if intent_mode else 55.0
  timing_plot = _StrikeTimingPlot(cfg) if cfg.timing_plot else None
  manual_selector = (
    _ManualBounceSelector(cfg, simulation.court) if cfg.manual_mouse_select else None
  )
  waiting_for_manual_selection = manual_selector is not None
  if waiting_for_manual_selection:
    simulation.set_manual_launch_waiting(True)
  if timing_plot is not None:
    timing_plot.reset(simulation)

  print(
    "[INFO]: Standalone deployment simulation: no task registry, Teacher, "
    "reference motion, ghost, phase command, or RSL-RL runner"
  )
  print(
    f"[INFO]: ONNX providers={simulation.policy.providers} "
    f"obs={simulation.policy.observation_dim} action={ACTION_DIM}"
  )
  if intent_mode:
    print(
      "[INFO]: Student-only court/controller: "
      f"preset={simulation.court.name} "
      f"net_x={simulation.court.net_x_m:.3f}m "
      f"net_width={2.0 * simulation.court.net_half_width_m:.3f}m, "
      "physical court XML, WarpRootDirected cache=0, Warp fused rollout, "
      "200-motion match, nominal play physics"
    )
    if simulation.policy.observation_dim == INTENT_ROBUST_OBSERVATION_DIM:
      print(
        "[INFO]: Robust ball perception enabled: "
        f"latency={cfg.ball_observation_latency_min_steps}.."
        f"{cfg.ball_observation_latency_max_steps} control steps, "
        f"dropout_start_probability="
        f"{cfg.ball_observation_dropout_start_probability:.3f}, "
        f"dropout_duration={cfg.ball_observation_dropout_duration_min_steps}.."
        f"{cfg.ball_observation_dropout_duration_max_steps} control steps"
      )
    if cfg.manual_mode:
      print(
        "[INFO]: Manual incoming-ball mode: launch and first-bounce coordinates "
        "use the fixed court/world frame; initial velocity is solved with Warp "
        "shooting; motion matching and motion timing constraints are disabled"
      )
      if cfg.manual_random_bounce_time:
        print(
          "[INFO]: Manual first-bounce time: uniform random sampling in "
          f"[{cfg.manual_bounce_time_min_s:.3f}, "
          f"{cfg.manual_bounce_time_max_s:.3f}]s"
        )
      else:
        print(
          f"[INFO]: Manual first-bounce time: fixed at {cfg.manual_bounce_time_s:.3f}s"
        )
      if manual_selector is not None:
        print(
          "[INFO]: Waiting for a first-bounce click. Each accepted click launches "
          "one ball; Student inference and G1 control remain active while waiting."
        )
    if cfg.continuous_strikes:
      print(
        "[INFO]: Continuous strikes enabled: preserving G1 state, simulation "
        "time, last action, and observation history between incoming balls"
      )
    if cfg.failure_trajectory_probability > 0.0:
      print(
        "[INFO]: Student-only failure trajectories enabled after the first "
        f"valid ball: probability={cfg.failure_trajectory_probability:.3f}, "
        f"no_net_fraction={cfg.failure_trajectory_no_net_fraction:.3f}, "
        "over-net failure threshold="
        f"{cfg.failure_trajectory_xy_distance_threshold_m:.3f}m"
      )
  with mujoco.viewer.launch_passive(
    simulation.model,
    simulation.data,
    show_left_ui=False,
    show_right_ui=False,
  ) as viewer:
    torso_body_id = mujoco.mj_name2id(
      simulation.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
    )
    viewer.cam.fixedcamid = -1
    if intent_mode:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
      viewer.cam.trackbodyid = -1
      viewer.cam.azimuth = 135.0
      viewer.cam.elevation = -30.0
      viewer.cam.distance = simulation.court.viewer_distance
      viewer.cam.lookat[:] = simulation.court.viewer_lookat
    else:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      viewer.cam.trackbodyid = torso_body_id
      viewer.cam.azimuth = 120.0
      viewer.cam.elevation = -5.0
      viewer.cam.distance = 2.8
      viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SCLINERTIA] = 1
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    status_overlay = _ViewerStatusOverlay(cfg.control_dt, cfg.realtime)
    render_period = 1.0 / cfg.render_fps
    last_tick = time.perf_counter()
    next_render = last_tick
    sim_budget = cfg.control_dt

    try:
      while viewer.is_running():
        if manual_selector is not None:
          manual_selector.poll()
          if not manual_selector.is_open:
            print("[INFO]: Manual bounce selector closed; stopping play.")
            break
          selection = manual_selector.consume_selection()
          if selection is not None:
            previous_selection = simulation.manual_bounce_position_xy_w.copy()
            try:
              simulation.set_manual_bounce_target(*selection)
              simulation.start_next_strike_cycle()
            except (RuntimeError, ValueError) as error:
              simulation.set_manual_bounce_target(*previous_selection)
              manual_selector.show_selection(previous_selection)
              manual_selector.set_status(f"Rejected: {error}", error=True)
              print(f"[MANUAL LAUNCH REJECTED] {error}", flush=True)
            else:
              waiting_for_manual_selection = False
              simulation.set_manual_launch_waiting(False)
              manual_selector.show_selection(selection)
              manual_selector.set_status(
                "Ball launched; waiting for this cycle to finish"
              )
              if timing_plot is not None:
                timing_plot.reset(simulation)

        now = time.perf_counter()
        if cfg.realtime:
          sim_budget += now - last_tick
        else:
          sim_budget = cfg.control_dt
        last_tick = now

        steps_this_tick = 0
        while sim_budget + 1.0e-12 >= cfg.control_dt and steps_this_tick < 8:
          report = simulation.step()
          status_overlay.record_step()
          if timing_plot is not None and not waiting_for_manual_selection:
            timing_plot.update(report)
          if report.episode_complete:
            if manual_selector is not None:
              waiting_for_manual_selection = True
              simulation.set_manual_launch_waiting(True)
              manual_selector.set_status(
                "Cycle complete; click another robot-side first-bounce point"
              )
            elif intent_mode and cfg.continuous_strikes:
              simulation.start_next_strike_cycle()
            else:
              simulation.reset()
            if timing_plot is not None and not waiting_for_manual_selection:
              timing_plot.reset(simulation)
          sim_budget -= cfg.control_dt
          steps_this_tick += 1

        if cfg.realtime and sim_budget >= cfg.control_dt:
          status_overlay.mark_capped()
          sim_budget = 0.0

        rendered = False
        now = time.perf_counter()
        if now >= next_render:
          status_overlay.record_frame()
          status_overlay.update(viewer, ball_speed_mps=simulation.ball_speed_mps)
          simulation.sync_viewer(viewer)
          rendered = True
          next_render += render_period
          if next_render < now:
            next_render = now + render_period

        if steps_this_tick == 0 and not rendered:
          time.sleep(0.001)
    except KeyboardInterrupt:
      print("\n[INFO]: Student ONNX Play stopped")
    finally:
      if timing_plot is not None:
        timing_plot.close()
      if manual_selector is not None:
        manual_selector.close()


def main() -> None:
  cfg = tyro.cli(StudentOnnxPlayConfig)
  run_student_onnx(cfg)


if __name__ == "__main__":
  main()
