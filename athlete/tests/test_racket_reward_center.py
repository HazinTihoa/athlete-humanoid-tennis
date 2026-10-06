from types import SimpleNamespace
from pathlib import Path

import mjlab  # Initialize task registration before importing reward modules.
import mujoco
import numpy as np
import pytest
import torch

from athlete.goal_cond_tracking.mdp.rewards import (
  _racket_reward_source_state,
  all_motions_target_position_error_exp,
)


def make_command():
  positions = torch.tensor([[[0.38, 0.01, 0.27], [1.0, 2.0, 3.0]]])
  velocities = torch.zeros_like(positions)
  data = SimpleNamespace(
    indexing=SimpleNamespace(geom_ids=torch.tensor([1])),
    data=SimpleNamespace(
      geom_xpos=torch.tensor([[[9.0, 9.0, 9.0], [0.38, -0.003, 0.27]]])
    ),
    site_ang_vel_w=torch.tensor([[[0.0, 0.0, 2.0]]]),
  )
  return SimpleNamespace(
    robot=SimpleNamespace(
      geom_names=("racket_ball_collision",),
      site_names=("racket_sweet_spot",), data=data,
    ),
    which_motion=torch.tensor([0]),
    _source_is_site_t=torch.tensor([[True, False], [False, True]]),
    _source_body_indices_t=torch.tensor([[0, 0], [0, 0]]),
    get_source_pos_w=lambda: positions,
    get_source_lin_vel_w=lambda: velocities,
  )


def test_center_replaces_only_racket_reward_source_without_mutating_observation():
  command = make_command()
  old_position = command.get_source_pos_w().clone()
  result = _racket_reward_source_state(command)
  torch.testing.assert_close(result[0, 0], torch.tensor([0.38, -0.003, 0.27]))
  torch.testing.assert_close(result[0, 1], old_position[0, 1])
  torch.testing.assert_close(command.get_source_pos_w(), old_position)
  # Subsequent calls read live geometry, not a cached world position.
  command.robot.data.data.geom_xpos[0, 1, 0] += 1.0
  assert _racket_reward_source_state(command)[0, 0, 0] == pytest.approx(1.38)
  command.which_motion[:] = 1
  torch.testing.assert_close(_racket_reward_source_state(command)[0, 0], old_position[0, 0])


def test_center_velocity_includes_angular_offset():
  command = make_command()
  result = _racket_reward_source_state(command, velocity=True)
  torch.testing.assert_close(result[0, 0], torch.tensor([0.026, 0.0, 0.0]))
  torch.testing.assert_close(result[0, 1], torch.zeros(3))
  torch.testing.assert_close(command.get_source_lin_vel_w(), torch.zeros_like(result))


def test_non_racket_robot_retains_original_reward_source():
  command = make_command()
  command.robot.geom_names = ()
  assert _racket_reward_source_state(command) is command.get_source_pos_w()
  assert _racket_reward_source_state(command, velocity=True) is command.get_source_lin_vel_w()


def test_position_reward_is_maximized_at_collider_center():
  command = make_command()
  command.target_position_w = _racket_reward_source_state(command).clone()
  command.normalized_phase = torch.tensor([0.5])
  command._target_phase_starts_t = torch.tensor([[0.0, 0.0]])
  command._target_phase_ends_t = torch.tensor([[1.0, 1.0]])
  command._target_pos_reward_weights_t = torch.tensor([[1.0, 0.0]])
  command._target_ori_reward_weights_t = torch.zeros(1, 2)
  command._target_vel_reward_weights_t = torch.zeros(1, 2)
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
  score = all_motions_target_position_error_exp(env, "motion", std=0.3)
  torch.testing.assert_close(score, torch.ones(1))
  command.target_position_w = command.get_source_pos_w().clone()
  assert all_motions_target_position_error_exp(env, "motion", std=0.3).item() < 1.0


@pytest.mark.parametrize("kind", ["train", "play", "student", "deploy"])
def test_all_models_share_collider_center_site_pose_and_velocity(kind):
  from athlete.scripts.tennis_scene import add_racket_ball_collision
  from athlete.scripts.play import _add_racket_ball_collision
  from athlete.scripts.play_student_onnx import _build_standalone_model

  root = Path(__file__).resolve().parents[2]
  xml = root / "robots/replay_unitree_description/mjcf/g1.xml"
  if kind == "student":
    model = _build_standalone_model(xml, 0.0025, (), np.zeros(0), np.zeros(0))
  elif kind == "deploy":
    model = mujoco.MjModel.from_xml_path(
      str(root / "robots/deployment/g1_27dof_deploy_tppo_tennis.xml")
    )
  else:
    wrapper = add_racket_ball_collision if kind == "train" else _add_racket_ball_collision
    model = wrapper(lambda: mujoco.MjSpec.from_file(str(xml)))().compile()
  site = model.site("racket_sweet_spot")
  geom = model.geom("racket_ball_collision")
  np.testing.assert_allclose(site.pos, [0.38, -0.003, 0.27])
  np.testing.assert_allclose(site.pos, geom.pos)
  np.testing.assert_allclose(site.quat, geom.quat)
  assert site.bodyid == geom.bodyid
  data = mujoco.MjData(model)
  rng = np.random.default_rng(42)
  for _ in range(3):
    data.qvel[:] = rng.normal(0, 0.2, model.nv)
    mujoco.mj_integratePos(model, data.qpos, data.qvel, 0.1)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.site_xpos[site.id], data.geom_xpos[geom.id])
    np.testing.assert_allclose(data.site_xmat[site.id], data.geom_xmat[geom.id])
    site_vel, geom_vel = np.zeros(6), np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site.id, site_vel, 0)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom.id, geom_vel, 0)
    np.testing.assert_allclose(site_vel, geom_vel, atol=1e-10)


def test_tuner_moves_site_with_collider():
  from athlete.scripts.tune_racket_collider import _build_model, apply_collider_settings

  root = Path(__file__).resolve().parents[2]
  model, geom_id = _build_model(root / "robots/replay_unitree_description/mjcf/g1.xml")
  data = mujoco.MjData(model)
  apply_collider_settings(model, data, geom_id, [.12, .17, .012], [.38, -.02, .27])
  site = model.site("racket_sweet_spot")
  np.testing.assert_allclose(data.site_xpos[site.id], data.geom_xpos[geom_id])


def test_expanded_restitution_tensor_matches_scalar_without_clamping():
  from athlete.scripts.tennis_scene import _contact_damping_from_restitution_tensor
  from athlete.scripts.tennis_physics import contact_damping_for_restitution

  values = torch.linspace(0.505, 0.715, 100, dtype=torch.float64)
  tensor_result = _contact_damping_from_restitution_tensor(values)
  scalar_result = torch.tensor([contact_damping_for_restitution(float(v)) for v in values], dtype=torch.float64)
  torch.testing.assert_close(tensor_result, scalar_result)
  assert (tensor_result > 0).all()
  assert (tensor_result[:-1] > tensor_result[1:]).all()
  assert tensor_result[0] > contact_damping_for_restitution(0.5373)
  for value in (0.49, float("nan"), 1.0):
    with pytest.raises(ValueError):
      _contact_damping_from_restitution_tensor(torch.tensor([value]))
