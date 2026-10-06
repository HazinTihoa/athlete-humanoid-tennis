from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from athlete.goal_cond_tracking.mdp.racket_fk import (
    FKDifferenceState,
    RacketJointFK,
    TennisSweetSpotFK,
)


def test_joint_fk_matches_mujoco_with_permuted_joint_order():
    path = (
        Path(__file__).resolve().parents[2]
        / "robots/replay_unitree_description/mjcf/g1.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(path))
    joints = [
        j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ][::-1]
    names = [model.joint(j).name for j in joints]
    fk = RacketJointFK(
        model,
        names,
        root_name="pelvis",
        site_name="racket_sweet_spot",
        device="cpu",
        dtype=torch.float64,
    )
    rng = np.random.default_rng(42)
    states, expected = [], []
    data = mujoco.MjData(model)
    for _ in range(80):
        data.qpos[:3] = rng.normal(size=3)
        quat = rng.normal(size=4)
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qpos[model.jnt_qposadr[joints]] = rng.uniform(-0.8, 0.8, len(joints))
        mujoco.mj_forward(model, data)
        pelvis = data.body("pelvis")
        expected.append(
            pelvis.xmat.reshape(3, 3).T
            @ (data.site("racket_sweet_spot").xpos - pelvis.xpos)
        )
        states.append(data.qpos[model.jnt_qposadr[joints]].copy())
    actual = fk.position_b(torch.tensor(np.array(states)))
    np.testing.assert_allclose(actual.numpy(), np.array(expected), atol=1e-12)


def test_difference_matches_deploy_filter_and_partial_resets():
    state = FKDifferenceState(2, "cpu", torch.float64)
    p = torch.zeros(2, 3, dtype=torch.float64)
    assert state.update(p, 0, 0.02).count_nonzero() == 0
    p = p + 0.02
    torch.testing.assert_close(state.update(p, 1, 0.02), torch.full_like(p, 0.35))
    torch.testing.assert_close(state.update(p, 1, 0.02), torch.full_like(p, 0.35))
    p = p + 0.02
    torch.testing.assert_close(state.update(p, 2, 0.02), torch.full_like(p, 0.5775))
    state.reset(torch.tensor([0]))
    p[0] += 100  # A teleported reset must not enter the velocity estimate.
    v = state.update(p, 2, 0.02)
    assert v[0].count_nonzero() == 0
    torch.testing.assert_close(v[1], torch.full_like(v[1], 0.5775))
    v = state.update(p + 0.02, 3, 0.02)
    torch.testing.assert_close(v[0], torch.full_like(v[0], 0.35))
    torch.testing.assert_close(v[1], torch.full_like(v[1], 0.725375))


@pytest.mark.parametrize("play", [False, True])
def test_new_config_only_replaces_student_and_critic_sweet_observations(play):
    from athlete.goal_cond_tracking.config.g1.env_cfgs import (
        unitree_g1_tennis_small_court_global_root_sweet_fk_env_cfg as variant,
    )
    from athlete.goal_cond_tracking.config.g1.env_cfgs import (
        unitree_g1_tennis_small_court_landing_std_curriculum150_torso_mass10_global_root_env_cfg as base,
    )
    from test_global_root_ablation import normalize

    original, changed = base(play), variant(play)
    for group in ("student", "critic"):
        for name in ("sweet_spot_position", "sweet_spot_linear_velocity"):
            term = changed.observations[group].terms[name]
            prior = original.observations[group].terms[name]
            assert term.func is TennisSweetSpotFK
            assert term.params["smoothing"] == 0.35
            assert term.noise == prior.noise
            term.func, term.params = prior.func, prior.params
    assert normalize(asdict(changed)) == normalize(asdict(original))


@pytest.mark.parametrize("play", [False, True])
def test_no_global_task_keeps_fk_position_but_uses_instantaneous_relative_velocity(play):
    from athlete.goal_cond_tracking.config.g1.env_cfgs import (
        unitree_g1_tennis_small_court_global_root_sweet_fk_env_cfg as base,
        unitree_g1_tennis_small_court_no_global_root_sweet_fk_env_cfg as variant,
    )
    original, changed = base(play), variant(play)
    assert "global_root_pos" not in changed.observations["student"].terms
    assert changed.observations["critic"].terms["global_root_pos"].noise is None
    from athlete.goal_cond_tracking.mdp.observations import (
        tennis_sweet_spot_relative_linear_velocity_b,
    )
    for group in ("student", "critic"):
        position = changed.observations[group].terms["sweet_spot_position"]
        velocity = changed.observations[group].terms["sweet_spot_linear_velocity"]
        assert position == original.observations[group].terms["sweet_spot_position"]
        assert velocity.func is tennis_sweet_spot_relative_linear_velocity_b
        assert "smoothing" not in velocity.params
