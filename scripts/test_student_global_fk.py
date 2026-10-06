"""Standalone compatibility checks without starting ROS or a viewer."""
import unittest
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy'))
from utils.tppo_student import SweetSpotVelocityEstimator
from athlete.scripts.play_student_onnx import (
    _SweetSpotFKEstimator, INTENT_GLOBAL_ROOT_OBSERVATION_DIM,
    SUPPORTED_OBSERVATION_DIMS, StudentOnnxPlayConfig,
)


class FKTests(unittest.TestCase):
    def setUp(self):
        self.xml = Path('robots/replay_unitree_description/mjcf/g1.xml').resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        self.names = tuple(self.model.joint(i).name for i in range(1, self.model.njnt))
        self.ids = np.array([self.model.joint(n).qposadr[0] for n in self.names])

    def test_fk_matches_deploy_and_repeated_reads_and_reset(self):
        a = _SweetSpotFKEstimator(self.model, self.model.site('racket_sweet_spot').id, .35)
        b = SweetSpotVelocityEstimator(self.xml, self.names, smoothing=.35)
        rng = np.random.default_rng(7)
        qpos = self.model.qpos0.copy()
        for step in range(15):
            qpos[:3] = [step*.001, 0, .76]
            qpos[self.ids] += rng.normal(0, .005, len(self.ids))
            p, v = a.update(qpos, step*.02)
            vb = b.update(qpos[:3], qpos[3:7], qpos[self.ids], step*.02)
            np.testing.assert_allclose(p, b.position_w, atol=1e-10)
            np.testing.assert_allclose(v, vb, atol=1e-10)
            np.testing.assert_array_equal(a.update(qpos, step*.02)[1], v)
        a.reset()
        np.testing.assert_array_equal(a.update(qpos, 0)[1], np.zeros(3))

    def test_dimensions_and_legacy_default(self):
        self.assertEqual(INTENT_GLOBAL_ROOT_OBSERVATION_DIM, 631)
        self.assertIn(631, SUPPORTED_OBSERVATION_DIMS)
        self.assertEqual(StudentOnnxPlayConfig(onnx_policy_file=Path('a.onnx')).sweet_spot_state_source, 'simulation')


if __name__ == '__main__':
    unittest.main()
