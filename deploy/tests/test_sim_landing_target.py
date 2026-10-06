import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.landing_target import LandingTargetSampler
from utils.tppo_intent_student import IntentObservationBuilder
from utils.tppo_student import BallState, RobotState


class LandingSamplerTest(unittest.TestCase):
    def test_ring_matches_physical_target_without_collision_geoms(self):
        from simulation.simulation_node.simulation_tppo_student import TppoStudentSimulationNode
        node = object.__new__(TppoStudentSimulationNode)
        scene = mujoco.MjvScene(mujoco.MjModel.from_xml_string("<mujoco/>"), maxgeom=100)
        node.viewer = SimpleNamespace(user_scn=scene, lock=nullcontext)
        node.landing_target_w = np.array([5.6, -.3, 0])
        node.landing_ring_radius = .5
        node._draw_landing_target()
        self.assertEqual(scene.ngeom, 64)
        centers = np.array([g.pos for g in scene.geoms[:64]])
        np.testing.assert_allclose(centers.mean(axis=0), [5.6,-.3,.025],atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(centers[:,:2]-[5.6,-.3],axis=1),
                                   .5*np.cos(np.pi/64),atol=1e-5)

    def test_training_distribution_and_net_clamp(self):
        sampler = LandingTargetSampler([5, 0], [.50005, .50005], [0,0,.76,1,0,0,0],3.75,11)
        samples = np.array([sampler.sample() for _ in range(10000)])
        self.assertTrue(np.all(samples[:, 0] >= 3.75))
        self.assertAlmostEqual(samples[:, 0].mean(),5,delta=.02)
        self.assertAlmostEqual(samples[:, 1].std(),.50005,delta=.015)
        np.testing.assert_array_equal(samples[:,2],0)

    def test_startup_transform_and_reproducibility(self):
        args = ([5,0],[0,0],[1,2,.76,2**-.5,0,0,2**-.5],3.75)
        np.testing.assert_allclose(LandingTargetSampler(*args).sample(),[3.75,7,0])
        args = ([5,0],[.5,.5],[0,0,.76,1,0,0,0],3.75,11)
        a,b = LandingTargetSampler(*args),LandingTargetSampler(*args)
        np.testing.assert_equal(a.sample(),b.sample())

    def test_sampled_world_target_reaches_actor_without_moving_with_root(self):
        builder = IntentObservationBuilder(np.zeros(29), np.array([5.,0]),
            include_global_root_pos=True, landing_target_observation_frame='current_root')
        builder.reset(np.array([1.,0,0,0]),np.array([0.,0,.76]))
        target = np.array([4.,1.,0])
        builder.landing_target_w[:] = target
        robot = RobotState(np.array([1.,0,.76]), np.array([1.,0,0,0]),
                           np.zeros(3),np.zeros(29),np.zeros(29),np.zeros(3))
        def observe():
            return builder.build_observation(robot,BallState(np.array([2.,0,1]),np.zeros(3)),
                np.array([0.,0,-1]),np.zeros(29),np.array([1.,0,1]))[90:92]
        np.testing.assert_allclose(observe(),[3,1])
        robot.root_position_w[0] = 2
        np.testing.assert_allclose(observe(),[2,1])
        np.testing.assert_array_equal(builder.landing_target_w,target)
        builder.landing_target_w[:] = [6.,-1,0]
        np.testing.assert_allclose(observe(),[4,-1])
