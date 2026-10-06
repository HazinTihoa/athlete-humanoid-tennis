"""Tests for the static Mocap/FK calibration collector."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "deploy"
SCRIPT_PATH = DEPLOY_DIR / "estimation" / "mocap" / "collect_fk_calibration_data.py"
sys.path.insert(0, str(DEPLOY_DIR))

SPEC = importlib.util.spec_from_file_location("collect_fk_calibration_data", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class CalibrationCollectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = DEPLOY_DIR / "configs" / "g1_tppo_student_m14_9.yaml"
        cls.config = yaml.safe_load(cls.config_path.read_text(encoding="utf-8"))
        cls.joint_names = collector.parse_joint_names(cls.config["joint_names"])
        cls.xml_path = REPO_ROOT / cls.config["kinematics_xml_path"]

    def test_fk_local_position_is_independent_of_world_root_pose(self) -> None:
        fk = collector.SweetSpotForwardKinematics(
            self.xml_path,
            self.joint_names,
            self.config["sweet_spot_site"],
        )
        joints = np.asarray(self.config["default_joint_pos"], dtype=np.float64)
        root_a = np.array([0.0, 0.0, 0.76])
        quat_a = np.array([0.0, 0.0, 0.0, 1.0])
        position_a, _ = fk.evaluate(root_a, quat_a, joints)

        yaw = 0.7
        root_b = np.array([1.2, -0.8, 0.76])
        quat_b = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)])
        position_b, _ = fk.evaluate(root_b, quat_b, joints)
        rotation_b = collector.quaternion_xyzw_to_matrix(quat_b)

        local_a = position_a - root_a
        local_b = rotation_b.T @ (position_b - root_b)
        np.testing.assert_allclose(local_a, local_b, atol=1.0e-9)

    def test_fk_applies_position_offset_in_root_frame(self) -> None:
        offset_b = np.array([0.0372, 0.0038, 0.0453])
        fk_without_offset = collector.SweetSpotForwardKinematics(
            self.xml_path,
            self.joint_names,
            self.config["sweet_spot_site"],
        )
        fk_with_offset = collector.SweetSpotForwardKinematics(
            self.xml_path,
            self.joint_names,
            self.config["sweet_spot_site"],
            position_offset_b=offset_b,
        )
        joints = np.asarray(self.config["default_joint_pos"], dtype=np.float64)
        yaw = 0.7
        root = np.array([1.2, -0.8, 0.76])
        quaternion_xyzw = np.array(
            [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)]
        )
        position_without_offset, _ = fk_without_offset.evaluate(
            root, quaternion_xyzw, joints
        )
        position_with_offset, _ = fk_with_offset.evaluate(
            root, quaternion_xyzw, joints
        )
        rotation_wb = collector.quaternion_xyzw_to_matrix(quaternion_xyzw)

        np.testing.assert_allclose(
            position_with_offset - position_without_offset,
            rotation_wb @ offset_b,
            atol=1.0e-9,
        )

    def test_summary_selects_matching_racket_normal_side(self) -> None:
        sweet = np.array([0.4, -0.2, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        ball = sweet + 0.0335 * normal
        records = []
        for index in range(4):
            records.append(
                {
                    "pose_index": 0,
                    "pose_label": "contact",
                    "calibrated_pelvis_position_w": np.array([0.0, 0.0, 0.76]),
                    "calibrated_pelvis_quaternion_xyzw": np.array(
                        [0.0, 0.0, 0.0, 1.0]
                    ),
                    "ball_position_b": ball,
                    "sweet_spot_position_b": sweet,
                    "racket_normal_b": normal,
                    "direct_error_b": ball - sweet,
                    "contact_plus_error_b": np.zeros(3),
                    "contact_minus_error_b": 2.0 * 0.0335 * normal,
                    "joint_position": np.full(29, 0.01 * index),
                    "raw_pelvis_mean_error_m": 0.001,
                    "raw_ball_mean_error_m": 0.001,
                }
            )

        summary = collector.summarize_pose(records, skipped_samples=1)
        self.assertEqual(summary["best_contact_side"], "+normal")
        self.assertEqual(summary["sample_count"], 4)
        self.assertAlmostEqual(summary["contact_plus_error_norm_m"]["mean"], 0.0)
        self.assertAlmostEqual(summary["direct_error_norm_m"]["mean"], 0.0335)


if __name__ == "__main__":
    unittest.main()
