from types import SimpleNamespace

import mjlab
import mujoco
import torch

from athlete.goal_cond_tracking.mdp.commands import MultiTargetMotionCommand
from athlete.goal_cond_tracking.mdp.observations import (
    tennis_sweet_spot_relative_linear_velocity_b,
)
from athlete.goal_cond_tracking.mdp.rewards import (
    tennis_reference_sweet_spot_velocity_reward,
)


def test_sim_observation_and_reward_remove_root_translation_and_rotation():
    position = torch.tensor([[[1., 0, 0]]] * 4)
    # First sample is rigidly attached to a moving, rotating Pelvis.
    relative = torch.tensor([[[0., 0, 0]], [[0., 1.2, 0]],
                             [[0., -1.2, 0]], [[0., -1.2, 0]]])
    source_velocity = relative + torch.tensor([1., 2, 0])
    targets = torch.tensor([[[0., 4, 0]], [[0., 4, 0]],
                            [[0., -4, 0]], [[0., 4, 0]]])
    command = SimpleNamespace(
        get_source_pos_w=lambda: position,
        get_source_lin_vel_w=lambda: source_velocity,
        get_reference_source_strike_relative_lin_vel_b=lambda: targets,
        robot_anchor_pos_w=torch.zeros(4, 3),
        robot_anchor_quat_w=torch.tensor([[1., 0, 0, 0]] * 4),
        robot_anchor_lin_vel_w=torch.tensor([[1., 0, 0]] * 4),
        robot_anchor_ang_vel_w=torch.tensor([[0., 0, 2.]] * 4),
        _target_pos_reward_weights_t=torch.ones(1, 1),
        which_motion=torch.zeros(4, dtype=torch.long), time_remaining=torch.zeros(4),
        ball_has_been_struck=torch.zeros(4, dtype=torch.bool), metrics={},
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command),
        scene={"tennis_ball": SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=position[:, 0], root_link_lin_vel_w=torch.ones(4, 3)
        ))},
    )
    torch.testing.assert_close(
        tennis_sweet_spot_relative_linear_velocity_b(env, "motion"), relative[:, 0]
    )
    reward = tennis_reference_sweet_spot_velocity_reward(
        env, "motion", relative_to_pelvis=True, target_speed_scale=0.3
    )
    torch.testing.assert_close(reward, torch.tensor([0., 1, 1, 0]))
    source_velocity[:] += torch.tensor([3., 0, 0])
    # No history, smoothing, or control-clock advance is needed for fresh velocity.
    torch.testing.assert_close(
        tennis_sweet_spot_relative_linear_velocity_b(env, "motion"),
        relative[:, 0] + torch.tensor([3., 0, 0]),
    )


def test_reference_target_uses_its_own_strike_frame_pelvis_and_signed_direction():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="robot/pelvis"><body name="robot/wrist">
        <site name="robot/racket_sweet_spot" pos="1 0 0"/>
      </body></body></worldbody></mujoco>''')
    pos = torch.zeros(2, 3, 2, 3)
    vel = torch.zeros_like(pos)
    ang = torch.zeros_like(pos)
    quat = torch.zeros(2, 3, 2, 4)
    quat[..., 0] = 1
    pos[:, 2, 0] = torch.tensor([10., 2, 3])
    pos[:, 2, 1] = torch.tensor([10., 2.5, 3])
    quat[:, 2] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
    vel[:, 2, 0] = torch.tensor([2., 0, 0])
    vel[:, 2, 1] = torch.tensor([[0.5, 4, 0], [0.5, -4, 0]])
    ang[:, 2, :, 2] = 3
    subtarget = SimpleNamespace(source_link="racket_sweet_spot", source_type="site",
                                target_phase_start=1., target_phase_end=1.)
    command = SimpleNamespace(
        motion_configs=[SimpleNamespace(sub_targets=[subtarget])] * 2,
        motion_loaders=[SimpleNamespace(time_step_total=3)] * 2,
        max_subtargets=1, device="cpu", motion_anchor_body_index=0,
        cfg=SimpleNamespace(body_names=("pelvis", "wrist")),
        _env=SimpleNamespace(sim=SimpleNamespace(mj_model=model)),
        _stacked_body_pos_w=pos, _stacked_body_quat_w=quat,
        _stacked_body_lin_vel_w=vel, _stacked_body_ang_vel_w=ang,
    )
    build = MultiTargetMotionCommand._build_reference_source_strike_lin_velocities
    relative = build(command, relative_to_pelvis=True)
    torch.testing.assert_close(relative[:, 0], torch.tensor([[4., 0, 0], [-4., 0, 0]]), atol=1e-5, rtol=1e-5)
    absolute = build(command)
    torch.testing.assert_close(absolute[:, 0], torch.tensor([[-2.5, 4, 0], [-2.5, -4, 0]]), atol=1e-5, rtol=1e-5)
