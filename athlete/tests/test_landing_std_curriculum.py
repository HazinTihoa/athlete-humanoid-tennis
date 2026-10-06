import math
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_inv

from athlete.goal_cond_tracking.config.g1.env_cfgs import (
  unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_pelvis_torso_tilt50_env_cfg,
  unitree_g1_tennis_small_court_landing_std_curriculum150_env_cfg,
)
from athlete.goal_cond_tracking.mdp.commands import (
  linear_landing_target_std,
  transform_startup_frame_ground_target,
  truncate_landing_targets_at_net,
)
from athlete.goal_cond_tracking.rl.landing_curriculum_runner import (
  LandingCurriculumOnPolicyRunner,
)
from athlete.goal_cond_tracking.rl.runner import MotionTrackingOnPolicyRunner


class LandingStdCurriculumTests(unittest.TestCase):
  def test_schedule_and_isolated_configuration(self):
    for steps, expected in ((0, 0.0), (360000, 0.75), (720000, 1.5), (1200000, 1.5)):
      _, std = linear_landing_target_std(steps, 720000, (0, 0, 0), (1.5, 1.5, 0))
      self.assertEqual(std, (expected, expected, 0))
    before = unitree_g1_tennis_small_court_frozen_intent_epoch3_landing5_pelvis_torso_tilt50_env_cfg()
    after = unitree_g1_tennis_small_court_landing_std_curriculum150_env_cfg()
    self.assertEqual(after.rewards, before.rewards)
    self.assertEqual(after.observations, before.observations)
    self.assertEqual(after.terminations, before.terminations)
    self.assertEqual(after.sim, before.sim)
    self.assertEqual(after.decimation, before.decimation)
    command = asdict(after.commands["motion"])
    original = asdict(before.commands["motion"])
    for key in ("landing_target_std_final", "landing_target_std_ramp_steps", "landing_target_net_margin_m"):
      command[key] = original[key]
    self.assertEqual(command, original)
    self.assertEqual(after.commands["motion"].landing_target_std_final, (1.5, 1.5, 0))
    play = unitree_g1_tennis_small_court_landing_std_curriculum150_env_cfg(play=True)
    self.assertEqual(play.commands["motion"].landing_target_std, (1.5, 1.5, 0))
    self.assertIsNone(play.commands["motion"].landing_target_std_final)

  def test_truncated_gaussian_distribution(self):
    torch.manual_seed(17)
    n = 100000
    mean = torch.tensor([[5.0, 0.0, 0.0]]).repeat(n, 1)
    samples = mean + torch.randn(n, 3) * torch.tensor([1.5, 1.5, 0.0])
    lower = torch.full((n,), 3.75)
    result = truncate_landing_targets_at_net(
      samples, mean, torch.full((n,), 2.25), torch.zeros(n), lower,
    )
    self.assertTrue(torch.isfinite(result).all())
    self.assertTrue((result[:, 0] >= lower).all())
    self.assertTrue(torch.equal(result[:, 1:], samples[:, 1:]))
    alpha = (3.75 - 5.0) / 1.5
    tail = 0.5 * math.erfc(alpha / math.sqrt(2))
    expected_mean = 5 + 1.5 * math.exp(-alpha * alpha / 2) / math.sqrt(2 * math.pi) / tail
    self.assertAlmostEqual(result[:, 0].mean().item(), expected_mean, delta=0.025)
    self.assertLess((result[:, 0] == lower).float().mean().item(), 0.001)

  def test_world_net_bound_with_rotated_startup_and_multiple_envs(self):
    n = 4096
    torch.manual_seed(23)
    yaw = torch.linspace(-1.5, 1.5, n)
    quat = torch.stack((torch.cos(yaw / 2), torch.zeros(n), torch.zeros(n), torch.sin(yaw / 2)), -1)
    origin = torch.randn(n, 3) * 20
    anchor = origin + torch.tensor([0.1, 0.2, 0.8])
    mean = torch.tensor([[5.0, 0.0, 0.0]]).expand(n, -1)
    std = torch.tensor([1.5, 0.7, 0.0])
    samples = mean + torch.randn(n, 3) * std
    samples_w = transform_startup_frame_ground_target(samples, anchor, quat, origin)
    mean_w = transform_startup_frame_ground_target(mean, anchor, quat, origin)
    rotation = matrix_from_quat(quat)[:, :2, :2]
    var_x = (rotation[:, 0].square() * std[:2].square()).sum(-1)
    cov_xy = (rotation[:, 0] * rotation[:, 1] * std[:2].square()).sum(-1)
    result = truncate_landing_targets_at_net(samples_w, mean_w, var_x, cov_xy, origin[:, 0] + 3.75)
    self.assertTrue((result[:, 0] >= origin[:, 0] + 3.75).all())
    self.assertTrue(torch.equal(result[:, 2], origin[:, 2]))
    startup = quat_apply(quat_inv(quat), result - anchor)
    startup[:, 2] = 0
    restored = transform_startup_frame_ground_target(startup, anchor, quat, origin)
    torch.testing.assert_close(restored, result, atol=2e-5, rtol=1e-5)

  def test_zero_std_stays_at_mean(self):
    mean = torch.tensor([[5.0, 0.0, 0.0]])
    result = truncate_landing_targets_at_net(mean, mean, torch.zeros(1), torch.zeros(1), torch.tensor([3.75]))
    self.assertTrue(torch.equal(result, mean))

  def test_curriculum_resume_and_new_stage_initialization(self):
    command = SimpleNamespace(
      cfg=SimpleNamespace(landing_target_std_final=(1.5, 1.5, 0.0)),
      landing_target_curriculum_step_offset=0,
      update_landing_target_std_curriculum=lambda: None,
    )
    env = SimpleNamespace(common_step_counter=396000, command_manager=SimpleNamespace(get_term=lambda _: command))
    runner = object.__new__(LandingCurriculumOnPolicyRunner)
    runner.env = SimpleNamespace(unwrapped=env)
    with patch.object(MotionTrackingOnPolicyRunner, "load", return_value={"env_state": {}}):
      runner.load("pure_distill.pt")
    self.assertEqual(env.common_step_counter + command.landing_target_curriculum_step_offset, 0)
    env.common_step_counter += 120000
    with patch.object(MotionTrackingOnPolicyRunner, "save") as save:
      runner.save("curriculum.pt")
      saved_infos = save.call_args.args[1]
    self.assertEqual(saved_infos["landing_target_curriculum_steps"], 120000)
    env.common_step_counter = 516000
    with patch.object(MotionTrackingOnPolicyRunner, "load", return_value=saved_infos):
      runner.load("curriculum.pt")
    self.assertEqual(env.common_step_counter + command.landing_target_curriculum_step_offset, 120000)


if __name__ == "__main__":
  unittest.main()
