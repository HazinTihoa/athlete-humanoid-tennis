import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mjlab
import numpy as np
import pytest
import torch

from athlete.goal_cond_tracking.mdp.ball_measurement import BatchedBallEstimator, MeasurementBallPerception
from athlete.goal_cond_tracking.config.g1.measurement_env_cfgs import measurement_env_cfg, measurement_runner_cfg

ROOT = Path(__file__).resolve().parents[2]


def reference_module():
    import sys
    spec = importlib.util.spec_from_file_location('measurement_reference', ROOT / 'deploy/utils/natnet_bridge_state.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
@pytest.mark.parametrize('dropout', [False, True])
def test_estimator_matches_deploy_on_measured_trajectory(dropout, dtype):
    ref = reference_module()
    cfg = ref.BallEstimatorConfig(position_correction_gain=1.0)
    scalar = ref.BallStateEstimator(cfg)
    batched = BatchedBallEstimator(1, 'cpu', dtype=dtype)
    p, v = np.array([6., 0., 1.]), np.array([-4., .3, 5.])
    rng = np.random.default_rng(9)
    for step in range(200):
        if step:
            p, v, _ = ref.propagate_ball_state(p, v, .02, cfg)
        # The estimator sees measured positions, never p/v physics truth.
        measured = p + rng.normal(0, .005, 3)
        valid = not (dropout and step % 31 in (11, 12, 13))
        if valid:
            scalar.observe(measured, step * .02)
            expected = scalar.position_w.copy(), scalar.velocity_w.copy()
        else:
            expected = scalar.estimate(step * .02)
        bp, bv = batched.update(torch.as_tensor(measured[None], dtype=dtype), torch.tensor([valid]))
        tol = 2e-7 if dtype == torch.float64 else 2e-4
        np.testing.assert_allclose(bp[0].numpy(), expected[0], atol=tol, err_msg=f'position step={step}')
        np.testing.assert_allclose(bv[0].numpy(), expected[1], atol=tol * 10, err_msg=f'velocity step={step}')


def test_shared_measurement_partial_reset_and_same_step_cache():
    class Scene(dict):
        env_origins = torch.zeros(2, 3)
    scene = Scene(ball=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.ones(2, 3))))
    env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.02, scene=scene, common_step_counter=0, extras={})
    cfg = measurement_env_cfg().observations['student'].terms['ball_position'].params['measurement_perception']
    state = MeasurementBallPerception(env, 'ball', cfg)
    for step in range(20):
        env.common_step_counter = step
        scene['ball'].data.root_link_pos_w[:, 0] += .03
        out = state.update(env)
        assert state.update(env) is out
    previous = tuple(value[1].clone() for value in out)
    state.remaining[1] = 4
    state.reset(torch.tensor([0]))
    out = state.update(env)
    for value, before in zip(out, previous):
        torch.testing.assert_close(value[1], before)
    assert state.remaining[1] == 4


def test_partial_reset_does_not_advance_other_environment():
    batch = BatchedBallEstimator(2, 'cpu')
    for step in range(20):
        batch.update(torch.tensor([[step * .03, 0., 1.]] * 2), torch.ones(2, dtype=torch.bool))
    before = {k: v[1].clone() for k, v in vars(batch).items() if isinstance(v, torch.Tensor)}
    batch.reset(torch.tensor([0]))
    batch.update(torch.tensor([[9., 0., 1.], [0., 0., 0.]]), torch.ones(2, dtype=torch.bool), torch.tensor([True, False]))
    for key, value in before.items():
        torch.testing.assert_close(getattr(batch, key)[1], value)


@pytest.mark.parametrize('steps,std', [(8,.05), (12,.05), (8,.075), (12,.075)])
def test_factorial_only_new_tasks(steps, std):
    from athlete.goal_cond_tracking.config.g1.env_cfgs import unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg
    from dataclasses import asdict
    from test_global_root_ablation import normalize
    before = normalize(asdict(unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg()))
    cfg = measurement_env_cfg(history_steps=steps, position_noise_std=std)
    assert normalize(asdict(unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg())) == before
    for group, names in [('student', ['ball_position', 'ball_linear_velocity']), ('intent_ball_history', ['position', 'linear_velocity'])]:
        for name in names:
            term = cfg.observations[group].terms[name]
            assert term.params['measurement_perception']['position_noise_std'] == std / 10
            assert term.params['measurement_perception']['position_bias_std'] == std
            assert term.noise is None
    assert len(cfg.observations['intent_state_history'].terms['state'].params['sample_lags']) == steps
    assert measurement_runner_cfg(history_steps=steps).actor.state_history_steps == steps
    assert cfg.commands['motion'].incoming_ball_failure_overhead_fraction == .5
    assert normalize(asdict(cfg.rewards['action_rate_l2'])) == before['rewards']['action_rate_l2']


def test_history_migration_preserves_other_parameters():
    spec = importlib.util.spec_from_file_location('migration', ROOT / 'scripts/expand_intent_history_checkpoint.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    source = dict(actor_state_dict={'student_position': torch.randn(1,5,128), 'mlp.0.weight':torch.randn(512,264)}, optimizer_state_dict={'state':{}}, infos={})
    migrated = m.expand_history(source, 12)
    torch.testing.assert_close(migrated['actor_state_dict']['student_position'][:, -5:], source['actor_state_dict']['student_position'])
    torch.testing.assert_close(migrated['actor_state_dict']['mlp.0.weight'], source['actor_state_dict']['mlp.0.weight'])


def test_overhead_candidate_classifier():
    from athlete.goal_cond_tracking.overhead_launch import sample_overhead_candidates, select_overhead_candidates
    from athlete.goal_cond_tracking.torch_tennis_planner import simulate_tennis_trajectories_torch
    torch.manual_seed(11)
    roots = torch.zeros(64, 2)
    p, v = sample_overhead_candidates(roots, torch.tensor([5.5,-1.5,.6]), torch.tensor([6.5,1.5,1.2]), (8.,12.), (2.2,2.8))
    batch = simulate_tennis_trajectories_torch(p, v, dt=.01, horizon_s=4., net_x=3.5, net_half_width=2.5,
        ball_mass_kg=torch.full((64,), .0577), court_restitution=torch.full((64,), .745),
        tangent_speed_retention=torch.full((64,), .825), drag_coefficient=torch.full((64,), .55))
    good = select_overhead_candidates(batch, roots)
    assert good.float().mean() > .5
    assert (v[:, :2].norm(dim=-1) >= 8).all()
