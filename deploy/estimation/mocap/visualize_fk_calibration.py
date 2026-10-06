#!/usr/bin/env python3
"""Visualize collected Mocap ball positions in the training MuJoCo model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml


DEPLOY_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = DEPLOY_DIR.parent
sys.path.insert(0, str(DEPLOY_DIR))

from utils.calibration_fk import SweetSpotForwardKinematics, parse_joint_names


IDENTITY = np.eye(3, dtype=np.float64).reshape(-1)
RED = np.array([1.0, 0.08, 0.08, 1.0], dtype=np.float32)
GREEN = np.array([0.08, 1.0, 0.08, 1.0], dtype=np.float32)
BLUE = np.array([0.08, 0.35, 1.0, 1.0], dtype=np.float32)
YELLOW = np.array([1.0, 0.82, 0.05, 0.92], dtype=np.float32)
CYAN = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
CONTACT_GREEN = np.array([0.15, 1.0, 0.3, 0.55], dtype=np.float32)
MAGENTA = np.array([1.0, 0.1, 0.75, 1.0], dtype=np.float32)
WHITE = np.array([1.0, 1.0, 1.0, 0.9], dtype=np.float32)


def load_config(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        candidate = DEPLOY_DIR / "configs" / path
        path = candidate if candidate.exists() else REPO_ROOT / path
    path = path.resolve()
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"Deployment config must be a mapping: {path}")
    return config, path


def resolve_path(path: str | Path, *, config_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    relative_to_config = (config_dir / candidate).resolve()
    if relative_to_config.exists():
        return relative_to_config
    return (REPO_ROOT / candidate).resolve()


@dataclass(frozen=True)
class PoseReplay:
    index: int
    label: str
    joint_position: np.ndarray
    measured_ball_b: np.ndarray


def load_pose_replays(data_path: Path) -> list[PoseReplay]:
    dataset = np.load(data_path)
    required = {"pose_index", "pose_label", "joint_position", "ball_position_b"}
    missing = sorted(required.difference(dataset.files))
    if missing:
        raise KeyError(f"Calibration dataset is missing: {missing}")

    replays: list[PoseReplay] = []
    for pose_index in np.unique(dataset["pose_index"]):
        mask = dataset["pose_index"] == pose_index
        first = int(np.flatnonzero(mask)[0])
        replays.append(
            PoseReplay(
                index=int(pose_index),
                label=str(dataset["pose_label"][first]),
                joint_position=dataset["joint_position"][mask].mean(axis=0),
                measured_ball_b=dataset["ball_position_b"][mask].mean(axis=0),
            )
        )
    if not replays:
        raise ValueError(f"Calibration dataset has no poses: {data_path}")
    return replays

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


def add_connector(
    scene: mujoco.MjvScene,
    geom_type: mujoco.mjtGeom,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    color: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.zeros(3, dtype=np.float64),
        np.asarray(start, dtype=np.float64),
        IDENTITY,
        color,
    )
    mujoco.mjv_connector(
        geom,
        geom_type,
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    geom.rgba[:] = color
    scene.ngeom += 1


def add_arrow(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    color: np.ndarray,
) -> None:
    add_connector(
        scene,
        mujoco.mjtGeom.mjGEOM_ARROW,
        start,
        end,
        width,
        color,
    )


def add_line(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    color: np.ndarray,
) -> None:
    add_connector(
        scene,
        mujoco.mjtGeom.mjGEOM_LINE,
        start,
        end,
        width,
        color,
    )


class CalibrationViewer:
    def __init__(
        self,
        config_path: str | Path,
        data_path: Path,
        *,
        axis_length_m: float,
    ) -> None:
        self.config, self.config_path = load_config(config_path)
        self.data_path = data_path.expanduser().resolve()
        self.replays = load_pose_replays(self.data_path)
        self.axis_length_m = float(axis_length_m)
        if self.axis_length_m <= 0.0:
            raise ValueError("axis_length_m must be positive")

        xml_path = resolve_path(
            self.config["kinematics_xml_path"],
            config_dir=self.config_path.parent,
        )
        self.joint_names = parse_joint_names(self.config["joint_names"])
        self.site_name = str(self.config.get("sweet_spot_site", "racket_sweet_spot"))
        self.fk = SweetSpotForwardKinematics(
            xml_path,
            self.joint_names,
            self.site_name,
        )
        self.model = self.fk.model
        self.data = self.fk.data
        self.root_height_m = float(self.config.get("default_base_pos", [0, 0, 0.76])[2])
        self.ball_radius_m = float(
            self.config.get("tennis_ball_physics", {}).get("radius_m", 0.0335)
        )
        self.pelvis_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        if self.pelvis_body_id < 0:
            raise ValueError("Training XML has no pelvis body")

        self._state_lock = threading.Lock()
        self._requested_index = 0
        self._quit_requested = False

    def request_pose_delta(self, delta: int) -> None:
        with self._state_lock:
            self._requested_index = (
                self._requested_index + delta
            ) % len(self.replays)

    def key_callback(self, keycode: int) -> None:
        if keycode in (ord("N"), ord("n"), 262):
            self.request_pose_delta(1)
        elif keycode in (ord("P"), ord("p"), 263):
            self.request_pose_delta(-1)
        elif keycode in (ord("Q"), ord("q")):
            with self._state_lock:
                self._quit_requested = True

    def requested_state(self) -> tuple[int, bool]:
        with self._state_lock:
            return self._requested_index, self._quit_requested

    def set_pose(self, replay: PoseReplay) -> dict[str, np.ndarray | float]:
        root_position_w = np.array([0.0, 0.0, self.root_height_m])
        root_quaternion_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        sweet_position_w, sweet_rotation_w = self.fk.evaluate(
            root_position_w,
            root_quaternion_xyzw,
            replay.joint_position,
        )
        pelvis_position_w = self.data.xpos[self.pelvis_body_id].copy()
        pelvis_rotation_w = self.data.xmat[self.pelvis_body_id].reshape(3, 3).copy()
        sweet_position_b = pelvis_rotation_w.T @ (
            sweet_position_w - pelvis_position_w
        )
        sweet_rotation_b = pelvis_rotation_w.T @ sweet_rotation_w
        racket_normal_b = sweet_rotation_b[:, 2]
        measured_ball_w = (
            pelvis_position_w + pelvis_rotation_w @ replay.measured_ball_b
        )
        expected_ball_b = (
            sweet_position_b - self.ball_radius_m * racket_normal_b
        )
        expected_ball_w = pelvis_position_w + pelvis_rotation_w @ expected_ball_b
        error_m = float(np.linalg.norm(replay.measured_ball_b - expected_ball_b))
        return {
            "pelvis_position_w": pelvis_position_w,
            "pelvis_rotation_w": pelvis_rotation_w,
            "sweet_position_w": sweet_position_w,
            "racket_normal_w": sweet_rotation_w[:, 2],
            "measured_ball_w": measured_ball_w,
            "expected_ball_w": expected_ball_w,
            "expected_ball_b": expected_ball_b,
            "error_m": error_m,
        }

    def draw(self, scene: mujoco.MjvScene, values: dict[str, np.ndarray | float]) -> None:
        scene.ngeom = 0
        pelvis_position_w = np.asarray(values["pelvis_position_w"])
        pelvis_rotation_w = np.asarray(values["pelvis_rotation_w"])
        for axis, color in enumerate((RED, GREEN, BLUE)):
            add_arrow(
                scene,
                pelvis_position_w,
                pelvis_position_w
                + self.axis_length_m * pelvis_rotation_w[:, axis],
                0.018,
                color,
            )

        measured_ball_w = np.asarray(values["measured_ball_w"])
        expected_ball_w = np.asarray(values["expected_ball_w"])
        sweet_position_w = np.asarray(values["sweet_position_w"])
        racket_normal_w = np.asarray(values["racket_normal_w"])
        add_sphere(scene, measured_ball_w, self.ball_radius_m, YELLOW)
        add_sphere(scene, expected_ball_w, self.ball_radius_m, CONTACT_GREEN)
        add_sphere(scene, sweet_position_w, 0.016, CYAN)
        add_line(scene, expected_ball_w, measured_ball_w, 3.0, WHITE)
        add_arrow(
            scene,
            sweet_position_w,
            sweet_position_w - 0.18 * racket_normal_w,
            0.012,
            MAGENTA,
        )

    def print_report(self) -> None:
        print(f"Training XML: {self.fk.xml_path}")
        print(f"Calibration data: {self.data_path}")
        print("Pose                       measured_ball_b            expected_ball_b            error")
        for replay in self.replays:
            values = self.set_pose(replay)
            expected = np.asarray(values["expected_ball_b"])
            print(
                f"{replay.label:26s} "
                f"{np.array2string(replay.measured_ball_b, precision=3):26s} "
                f"{np.array2string(expected, precision=3):26s} "
                f"{float(values['error_m']):.3f}m"
            )

    def run(self) -> None:
        self.print_report()
        current_index = -1
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.75)
            viewer.cam.distance = 2.7
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -12.0
            while viewer.is_running():
                requested_index, quit_requested = self.requested_state()
                if quit_requested:
                    break
                if requested_index != current_index:
                    replay = self.replays[requested_index]
                    with viewer.lock():
                        values = self.set_pose(replay)
                        self.draw(viewer.user_scn, values)
                    viewer.set_texts(
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            (
                                f"Pose {requested_index + 1}/{len(self.replays)}: "
                                f"{replay.label}\n"
                                f"Contact-center error: {float(values['error_m']):.3f} m\n"
                                "N/Right: next   P/Left: previous   Q: quit"
                            ),
                            (
                                "Pelvis axes: X red, Y green, Z blue\n"
                                "Mocap ball: yellow\n"
                                "FK sweet point: cyan\n"
                                "Expected ball center: transparent green\n"
                                "Racket -normal: magenta"
                            ),
                        )
                    )
                    print(
                        f"[POSE] {replay.label}: contact-center error="
                        f"{float(values['error_m']):.3f}m"
                    )
                    current_index = requested_index
                viewer.sync()
                time.sleep(0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show collected Mocap/FK calibration poses in the training XML."
    )
    parser.add_argument(
        "--config",
        default="g1_tppo_student_m14_9.yaml",
        help="Deployment YAML path or name; its kinematics_xml_path is used.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEPLOY_DIR / "calibration_data" / "m14_9_pelvis_fk.npz",
    )
    parser.add_argument("--axis-length-m", type=float, default=0.30)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualizer = CalibrationViewer(
        args.config,
        args.data,
        axis_length_m=args.axis_length_m,
    )
    if args.report_only:
        visualizer.print_report()
    else:
        visualizer.run()


if __name__ == "__main__":
    main()
