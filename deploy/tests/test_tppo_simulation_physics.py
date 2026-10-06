"""Regression tests for the training-aligned TPPO MuJoCo simulation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "deploy"
sys.path.insert(0, str(DEPLOY_DIR))

from utils.tppo_simulation_physics import (
    ROBOT_TERRAIN_COLLISION_BIT,
    ExplicitGroundRebound,
    build_training_aligned_model,
    contact_damping_for_restitution,
    _apply_physics_sample,
    sample_tennis_physics,
)


class TppoSimulationPhysicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config_path = DEPLOY_DIR / "configs" / "g1_tppo_student_intentref.yaml"
        cls.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cls.model, cls.sample = build_training_aligned_model(
            robot_xml_path=REPO_ROOT / cls.config["simulation_xml_path"],
            court_xml_path=REPO_ROOT / cls.config["tennis_court_xml_path"],
            joint_names=tuple(cls.config["joint_names"].split(",")),
            physics_dt=float(cls.config["simulation"]["physics_dt"]),
            nominal_ball_physics=cls.config["tennis_ball_physics"],
            randomization_config=cls.config["simulation"][
                "physics_domain_randomization"
            ],
            court_net_x_m=float(cls.config["simulation"]["court"]["net_x_m"]),
            court_net_half_width_m=float(
                cls.config["simulation"]["court"]["net_half_width_m"]
            ),
            launch_position_min_w=cls.config["simulation"]["incoming_ball"][
                "position_min_w"
            ],
            launch_position_max_w=cls.config["simulation"]["incoming_ball"][
                "position_max_w"
            ],
        )

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        return mujoco.mj_name2id(self.model, object_type, name)

    def test_uses_training_robot_joint_contract(self) -> None:
        for name in ("waist_roll_joint", "waist_pitch_joint"):
            joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
            self.assertGreaterEqual(joint_id, 0)
            np.testing.assert_allclose(
                self.model.jnt_range[joint_id], (-0.52, 0.52), atol=1.0e-12
            )
            dof_address = self.model.jnt_dofadr[joint_id]
            self.assertAlmostEqual(self.model.dof_frictionloss[dof_address], 0.1)
        self.assertEqual(self.model.nu, 29)

    def test_optional_torso_mass_scales_inertia_without_changing_com(self) -> None:
        import copy

        model = copy.copy(self.model)
        body = model.body("torso_link").id
        mass = float(model.body_mass[body])
        inertia = model.body_inertia[body].copy()
        com = model.body_ipos[body].copy()
        sample = sample_tennis_physics(
            {"enabled": True, "torso_mass_scale": [1.1, 1.1],
             "torso_com_offset_x_m": [0, 0], "torso_com_offset_y_m": [0, 0],
             "torso_com_offset_z_m": [0, 0]}, self.config["tennis_ball_physics"],
        )
        _apply_physics_sample(model, sample)
        self.assertAlmostEqual(model.body_mass[body], mass * 1.1)
        np.testing.assert_allclose(model.body_inertia[body], inertia * 1.1)
        np.testing.assert_allclose(model.body_ipos[body], com)

    def test_has_one_robot_ground_and_no_legacy_floor(self) -> None:
        self.assertEqual(self._id(mujoco.mjtObj.mjOBJ_GEOM, "floor"), -1)
        terrain_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        ball_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "tennis_ball_geom")
        court_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "tennis_court/court_surface")
        self.assertGreaterEqual(terrain_id, 0)
        self.assertEqual(
            int(self.model.geom_contype[terrain_id]), ROBOT_TERRAIN_COLLISION_BIT
        )
        self.assertEqual(int(self.model.geom_conaffinity[terrain_id]), 0)
        self.assertEqual(
            int(self.model.geom_contype[terrain_id])
            & int(self.model.geom_conaffinity[ball_id]),
            0,
        )
        self.assertEqual(int(self.model.geom_contype[court_id]), 0)
        self.assertEqual(int(self.model.geom_conaffinity[court_id]), 0)

    def test_racket_inertia_and_contact_match_sample(self) -> None:
        racket_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "racket_inertia")
        self.assertGreaterEqual(racket_id, 0)
        self.assertAlmostEqual(
            float(self.model.body_mass[racket_id]), self.sample.racket_mass_kg
        )
        expected_com = np.array((0.38, 0.01, 0.27)) + self.sample.racket_com_offset_m
        np.testing.assert_allclose(self.model.body_ipos[racket_id], expected_com)

        collider_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "racket_ball_collision")
        np.testing.assert_allclose(self.model.geom_size[collider_id], (0.12, 0.17, 0.012))
        np.testing.assert_allclose(self.model.geom_pos[collider_id], (0.38, -0.003, 0.27))
        site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "racket_sweet_spot")
        np.testing.assert_allclose(self.model.site_pos[site_id], self.model.geom_pos[collider_id])
        np.testing.assert_allclose(self.model.site_quat[site_id], self.model.geom_quat[collider_id])

        pair_id = self._id(mujoco.mjtObj.mjOBJ_PAIR, "tennis_ball_racket")
        self.assertAlmostEqual(
            float(self.model.pair_solref[pair_id, 1]),
            contact_damping_for_restitution(self.sample.racket_restitution),
        )

    def test_court_contact_and_visual_scene_match_training_assets(self) -> None:
        self.assertGreaterEqual(
            self._id(mujoco.mjtObj.mjOBJ_GEOM, "tennis_court/surround"), 0
        )
        self.assertGreaterEqual(
            self._id(mujoco.mjtObj.mjOBJ_GEOM, "tennis_court/net_collision"), 0
        )
        self.assertGreaterEqual(
            self._id(mujoco.mjtObj.mjOBJ_GEOM, "tennis_court/baseline_near"), 0
        )
        self.assertGreaterEqual(self.model.nlight, 2)
        pair_id = self._id(mujoco.mjtObj.mjOBJ_PAIR, "tennis_ball_court")
        self.assertAlmostEqual(
            float(self.model.pair_solref[pair_id, 1]),
            contact_damping_for_restitution(self.sample.court_restitution),
        )

        net_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "tennis_court/net_collision")
        self.assertAlmostEqual(
            float(self.model.geom_pos[net_id, 0]),
            float(self.config["simulation"]["court"]["net_x_m"]),
        )
        self.assertAlmostEqual(
            float(self.model.geom_size[net_id, 1]),
            float(self.config["simulation"]["court"]["net_half_width_m"]),
        )

        region_id = self._id(
            mujoco.mjtObj.mjOBJ_GEOM, "incoming_ball_launch_region"
        )
        launch_min = np.asarray(
            self.config["simulation"]["incoming_ball"]["position_min_w"]
        )
        launch_max = np.asarray(
            self.config["simulation"]["incoming_ball"]["position_max_w"]
        )
        np.testing.assert_allclose(
            self.model.geom_pos[region_id], 0.5 * (launch_min + launch_max)
        )
        np.testing.assert_allclose(
            self.model.geom_size[region_id], 0.5 * (launch_max - launch_min)
        )

    def test_domain_sample_stays_inside_training_ranges(self) -> None:
        ranges = self.config["simulation"]["physics_domain_randomization"]
        checks = {
            "ball_mass_kg": self.sample.ball_mass_kg,
            "court_restitution": self.sample.court_restitution,
            "ground_tangent_speed_retention": (
                self.sample.ground_tangent_speed_retention
            ),
            "drag_coefficient": self.sample.drag_coefficient,
            "racket_mass_kg": self.sample.racket_mass_kg,
            "racket_restitution": self.sample.racket_restitution,
            "foot_sliding_friction": self.sample.foot_sliding_friction,
        }
        for name, value in checks.items():
            self.assertGreaterEqual(value, ranges[name][0], name)
            self.assertLessEqual(value, ranges[name][1], name)

    def test_small_court_config_controls_net_and_launch_region(self) -> None:
        config_path = (
            DEPLOY_DIR
            / "configs"
            / "g1_tppo_student_intentref_smallcourt.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        simulation = config["simulation"]
        model, _ = build_training_aligned_model(
            robot_xml_path=REPO_ROOT / config["simulation_xml_path"],
            court_xml_path=REPO_ROOT / config["tennis_court_xml_path"],
            joint_names=tuple(config["joint_names"].split(",")),
            physics_dt=float(simulation["physics_dt"]),
            nominal_ball_physics=config["tennis_ball_physics"],
            randomization_config=simulation["physics_domain_randomization"],
            court_net_x_m=float(simulation["court"]["net_x_m"]),
            court_net_half_width_m=float(
                simulation["court"]["net_half_width_m"]
            ),
            launch_position_min_w=simulation["incoming_ball"]["position_min_w"],
            launch_position_max_w=simulation["incoming_ball"]["position_max_w"],
        )

        net_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "tennis_court/net_collision"
        )
        region_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "incoming_ball_launch_region"
        )
        self.assertAlmostEqual(float(model.geom_pos[net_id, 0]), 3.5)
        self.assertAlmostEqual(float(model.geom_size[net_id, 1]), 2.5)
        np.testing.assert_allclose(model.geom_pos[region_id], (6.0, 0.0, 0.9))
        np.testing.assert_allclose(model.geom_size[region_id], (0.5, 1.5, 0.3))
        self.assertEqual(config["landing_target_startup"], [6.0, 0.0])

    def test_explicit_ground_rebound_sets_sampled_velocity(self) -> None:
        data = mujoco.MjData(self.model)
        base_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        base_qpos = self.model.jnt_qposadr[base_joint_id]
        data.qpos[base_qpos : base_qpos + 7] = (0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0)

        ball_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "tennis_ball_freejoint")
        ball_qpos = self.model.jnt_qposadr[ball_joint_id]
        ball_dof = self.model.jnt_dofadr[ball_joint_id]
        data.qpos[ball_qpos : ball_qpos + 7] = (
            1.0,
            0.0,
            0.5,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        data.qvel[ball_dof : ball_dof + 6] = (1.0, 0.5, -2.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, data)

        rebound = ExplicitGroundRebound(
            self.model,
            ball_radius_m=float(self.config["tennis_ball_physics"]["radius_m"]),
            court_restitution=self.sample.court_restitution,
            tangent_speed_retention=self.sample.ground_tangent_speed_retention,
        )
        bounced = False
        for _ in range(300):
            mujoco.mj_step(self.model, data)
            if rebound.after_step(self.model, data):
                bounced = True
                break

        self.assertTrue(bounced)
        self.assertGreater(data.qvel[ball_dof + 2], 0.0)
        self.assertAlmostEqual(
            data.qvel[ball_dof], self.sample.ground_tangent_speed_retention
        )
        self.assertAlmostEqual(
            data.qvel[ball_dof + 1],
            0.5 * self.sample.ground_tangent_speed_retention,
        )


if __name__ == "__main__":
    unittest.main()
