"""M14-11 observation parity, preserving the older M14-9 defaults."""
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'deploy'))
from utils.tppo_intent_student import IntentObservationBuilder
from utils.tppo_student import BallState, RobotState, SweetSpotVelocityEstimator


class M14ElevenTests(unittest.TestCase):
    def test_relative_velocity_removes_translation_and_rotation(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT/'robots/replay_unitree_description/mjcf/g1.xml'))
        names = tuple(model.joint(i).name for i in range(1, model.njnt))
        estimator = SweetSpotVelocityEstimator(ROOT/'robots/replay_unitree_description/mjcf/g1.xml', names)
        data = mujoco.MjData(model)
        rng = np.random.default_rng(11)
        site = model.site('racket_sweet_spot').id
        root = model.body('pelvis').id
        for step in range(20):
            data.qpos[:3] = rng.normal(size=3)
            quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
            data.qpos[3:7] = quat
            data.qpos[estimator.joint_qpos_addresses] = rng.normal(0, .1, 29)
            data.qvel[:] = rng.normal(size=model.nv)
            mujoco.mj_forward(model, data)
            jp = np.zeros((3,model.nv)); jb = jp.copy(); jr = jp.copy()
            mujoco.mj_jacSite(model,data,jp,None,site)
            mujoco.mj_jacBody(model,data,jb,jr,root)
            expected = (jp-jb)@data.qvel - np.cross(jr@data.qvel,data.site_xpos[site]-data.xpos[root])
            estimator.update(data.qpos[:3],quat,data.qpos[estimator.joint_qpos_addresses],step*.02)
            actual = estimator.relative_velocity_w(data.qvel[estimator.joint_dof_addresses])
            np.testing.assert_allclose(actual,expected,atol=1e-10)
            np.testing.assert_allclose(estimator.relative_velocity_w(np.zeros(29)),0,atol=1e-10)

    def test_current_root_landing_uses_full_rotation_and_ground_height(self):
        builder = IntentObservationBuilder(np.zeros(29),np.array([5.,0.]),
            include_global_root_pos=True,landing_target_observation_frame='current_root')
        builder.reset(np.array([1.,0,0,0]),np.array([0.,0,.76]))
        quat = np.array([np.cos(.3),0,np.sin(.3),0])
        robot = RobotState(np.array([1.,.5,.8]),quat,np.zeros(3),np.zeros(29),np.zeros(29),np.array([9.,9.,9.]))
        ball = BallState(np.array([2.,0,1.]),np.zeros(3))
        obs = builder.build_observation(robot,ball,np.array([0.,0,-1.]),np.zeros(29),np.array([1.,.5,1.]),
            sweet_spot_relative_velocity_b=np.array([1.,2.,3.]))
        from utils.tppo_student import quaternion_to_rotation_matrix_wxyz
        expected = quaternion_to_rotation_matrix_wxyz(quat).T@(np.array([5.,0,0])-robot.root_position_w)
        np.testing.assert_allclose(obs[90:92],expected[:2],atol=1e-6)
        np.testing.assert_allclose(obs[122:125],[1,2,3])
        np.testing.assert_allclose(obs[133:136],robot.root_position_w)
        self.assertEqual(obs.shape,(631,))


if __name__ == '__main__':
    unittest.main()
