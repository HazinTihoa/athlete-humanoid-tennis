from dataclasses import asdict, fields
from types import SimpleNamespace

import pytest
import torch

from athlete.goal_cond_tracking.config.g1.full_flight import m14_11_full_flight_env_cfg
from athlete.goal_cond_tracking.config.g1.env_cfgs import unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg
from athlete.goal_cond_tracking.mdp.full_flight import (
    FullFlightMotionCommand, PhysicalRallyTracker, actual_landing_reward, full_flight_time_out,
)


@pytest.mark.parametrize("play", [False, True])
def test_exact_m14_11_base_with_only_declared_changes(play):
    from test_global_root_ablation import normalize
    base = unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg(play)
    cfg = m14_11_full_flight_env_cfg(play)
    assert cfg.episode_length_s == (base.episode_length_s if play else 20.0)
    expected_observations = asdict(base)["observations"]
    if not play:
        assert cfg.observations["student"].enable_corruption
        for name, expected_std in (
            ("sweet_spot_position", 0.05), ("sweet_spot_linear_velocity", 0.20),
        ):
            original_std = base.observations["student"].terms[name].noise.std
            assert cfg.observations["student"].terms[name].noise.std == expected_std
            assert expected_std == original_std
    # All observation properties and groups match m14-11, including
    # critic, ball, joint velocity, and the play-mode corruption setting.
    assert normalize(expected_observations) == normalize(asdict(cfg)["observations"])
    assert cfg.commands["motion"].landing_target_std == (0, 0, 0)
    assert cfg.commands["motion"].landing_target_std_final is None
    assert "landing_target_std" not in cfg.curriculum
    allowed = {"landing_target_std", "landing_target_std_final",
               "incoming_ball_failure_trajectory_probability",
               "incoming_ball_failure_random_direction_fraction",
               "incoming_ball_failure_overhead_fraction"}
    for f in fields(base.commands["motion"]):
        if f.name not in allowed:
            assert normalize(getattr(base.commands["motion"], f.name)) == normalize(getattr(cfg.commands["motion"], f.name)), f.name
    for key, term in base.rewards.items():
        assert term.weight == cfg.rewards[key].weight
        assert term.params == cfg.rewards[key].params
        if key != "ball_landing_reward":
            assert term.func is cfg.rewards[key].func
    assert cfg.rewards["ball_landing_reward"].weight == 100
    assert cfg.rewards["ball_landing_reward"].params["std"] == 1.0
    assert set(cfg.rewards) - set(base.rewards) == {"torso_upright", "post_hit_stillness"}
    assert cfg.rewards["post_hit_stillness"].weight == 1.0
    assert cfg.rewards["torso_upright"].weight == 1.0
    assert cfg.rewards["torso_upright"].params == {
        "command_name": "motion", "body_name": "torso_link", "std_degrees": 30.0,
    }
    assert "foot_force" not in cfg.events
    assert normalize(asdict(base)["events"]) == normalize(asdict(cfg)["events"])


@pytest.mark.parametrize("axis", ["x", "y"])
def test_torso_upright_reward_uses_world_tilt_and_leaves_yaw_free(axis):
    from mjlab.utils.lab_api.math import quat_mul

    def quat(axis, degrees):
        half = torch.deg2rad(torch.tensor(degrees, dtype=torch.float64)) / 2
        q = torch.zeros(len(degrees), 4, dtype=torch.float64)
        q[:, 0] = half.cos()
        q[:, {"x": 1, "y": 2, "z": 3}[axis]] = half.sin()
        return q

    angles = [0, 10, 30, 60, 90, 180]
    torso = quat(axis, angles)
    pelvis = quat("x", [45] * len(angles))
    robot = SimpleNamespace(
        body_names=("pelvis", "torso_link"),
        data=SimpleNamespace(
            body_link_quat_w=torch.stack([pelvis, torso], dim=1),
            gravity_vec_w=torch.tensor([[0., 0., -9.81]], dtype=torch.float64).expand(len(angles), -1),
        ),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(robot=robot)))
    term = m14_11_full_flight_env_cfg().rewards["torso_upright"]
    score = term.func(env, **term.params)
    assert score[0].item() == 1.0  # Tilted pelvis, upright torso: full reward.
    assert torch.all(score[:-1] > score[1:])
    assert score[2].item() == pytest.approx(torch.exp(torch.tensor(-1.)).item())
    # Changing heading or pelvis pose leaves the same actual torso tilt score.
    robot.data.body_link_quat_w[:, 1] = quat_mul(quat("z", [75] * len(angles)), torso)
    robot.data.body_link_quat_w[:, 0] = quat("x", [0] * len(angles))
    torch.testing.assert_close(term.func(env, **term.params), score)
    robot.data.gravity_vec_w = robot.data.gravity_vec_w / 9.81
    torch.testing.assert_close(term.func(env, **term.params), score)


def tracker_fixture():
    cfg = m14_11_full_flight_env_cfg()
    c = SimpleNamespace(cfg=cfg.commands["motion"], metrics={},
                        motion_chain_count=torch.tensor([2]),
                        robot_anchor_pos_w=torch.zeros(1, 3),
                        target_landing_position_w=torch.tensor([[5., 0., 0.]]))
    for name in ("ball_has_been_struck", "ball_landing_recorded"):
        setattr(c, name, torch.zeros(1, dtype=torch.bool))
    c.ball_landing_position_w = torch.zeros(1, 3)
    for name in ("ball_hit_reward", "ball_direction_reward", "ball_net_clearance_reward",
                 "ball_out_speed_reward", "ball_post_strike_steps", "ball_post_strike_elapsed_s"):
        setattr(c, name, torch.zeros(1))
    for name in ("error_ball_landing", "ball_landing_prediction_valid", "ball_net_crossing_height",
                 "ball_out_speed", "ball_target_projected_speed", "ball_direction_error_degrees"):
        c.metrics[name] = torch.zeros(1)
    env = SimpleNamespace(cfg=cfg, num_envs=1, device="cpu", step_dt=0.02,
                          scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
                          command_manager=SimpleNamespace(get_term=lambda _: c),
                          episode_length_buf=torch.tensor([1001]), max_episode_length=1000)
    t = PhysicalRallyTracker(env, None)
    env.tennis_full_flight = t
    t.active[:] = True
    return t, c, env


def observe(t, position, *, hit=False, ground=False, speed=(4., 0., 0.), dt=0.0025):
    p = torch.tensor([position], dtype=torch.float32)
    t.observe(p, torch.tensor([speed], dtype=torch.float32), torch.tensor([hit]),
              torch.tensor([ground]), p, dt)


def test_incoming_bounce_never_scores_then_actual_first_return_landing_scores_once():
    t, c, env = tracker_fixture()
    observe(t, [2., 0., 0.], ground=True)
    assert not c.ball_landing_recorded.item()
    observe(t, [1., 0., 1.], hit=True)
    observe(t, [3., 0., 1.5])
    observe(t, [4., 0., 1.3])
    assert not c.ball_landing_recorded.item()
    assert actual_landing_reward(env, "motion").item() == 0
    observe(t, [5.2, 0., 0.], ground=True)
    assert c.ball_landing_recorded.item()
    assert actual_landing_reward(env, "motion").item() == 1
    assert actual_landing_reward(env, "motion").item() == 0
    observe(t, [6., 0., 1.])
    observe(t, [7., 0., 0.], ground=True)
    assert actual_landing_reward(env, "motion").item() == 0
    assert c.ball_landing_position_w[0, 0].item() == pytest.approx(5.2)
    assert not t.done.item()  # First landing is not a ball respawn.


def test_long_return_ends_past_radius_without_waiting_for_landing():
    t, c, env = tracker_fixture()
    observe(t, [1., 0., 1.], hit=True)
    observe(t, [21., 0., 1.5])
    assert t.done.item()
    assert full_flight_time_out(env).item()
    assert not c.ball_landing_recorded.item()
    assert actual_landing_reward(env, "motion").item() == 0
    c.motion_chain_count[:] = 1
    assert not full_flight_time_out(env).item()  # A long opening serve must allow failure sampling.


def test_twenty_second_timeout_waits_for_ball_completion_and_two_serves():
    t, c, env = tracker_fixture()
    assert env.max_episode_length * env.step_dt == 20.0
    c.motion_chain_count[:] = 2
    t.done[:] = True
    env.episode_length_buf[:] = 999
    assert not full_flight_time_out(env).item()
    env.episode_length_buf[:] = 1000
    assert full_flight_time_out(env).item()
    t.done[:] = False
    assert not full_flight_time_out(env).item()
    t.done[:] = True
    c.motion_chain_count[:] = 1
    assert not full_flight_time_out(env).item()


def test_failed_net_still_records_actual_landing_but_no_score_and_reset_clears():
    t, c, env = tracker_fixture()
    observe(t, [1., 0., 1.], hit=True)
    observe(t, [3., 0., .1])
    observe(t, [4., 0., .1])
    observe(t, [5., 0., 0.], ground=True)
    assert c.ball_landing_recorded.item()
    assert c.metrics["ball_actual_landing_observed"].item() == 1
    assert actual_landing_reward(env, "motion").item() == 0
    t.reset(torch.tensor([0]))
    assert not c.ball_has_been_struck.item()
    assert not c.ball_landing_recorded.item()
    assert not t.net_seen.item()


def test_grounded_racket_contact_resolves_without_crediting_incoming_bounce():
    t, c, env = tracker_fixture()
    observe(t, [1., 0., 0.], ground=True)
    observe(t, [1., 0., 0.], ground=True, hit=True)
    assert not c.ball_landing_recorded.item()
    observe(t, [1.01, 0., 0.], ground=True)
    assert c.ball_landing_recorded.item()
    assert actual_landing_reward(env, "motion").item() == 0


@pytest.mark.parametrize("hit", [False, True])
@pytest.mark.parametrize("ground", [False, True])
def test_ball_stops_immediately_at_threshold_regardless_of_hit_or_contact(hit, ground):
    t, c, env = tracker_fixture()
    observe(t, [0., 0., 1.], hit=hit, ground=ground, speed=(.1, 0., 0.))
    assert t.done.item()
    assert not c.ball_landing_recorded.item()
    assert actual_landing_reward(env, "motion").item() == 0


def test_recorded_low_speed_ball_ends_with_stale_ground_contact():
    """Reproduce the recorded position, speed magnitudes and stale contact age."""
    t, c, _ = tracker_fixture()
    c.robot_anchor_pos_w[:] = torch.tensor([[.08983, .54016, .75287]])
    t.ground_contact_age[:] = .8125
    t.stop_velocity_initialized[:] = True
    t.stop_velocity[:] = torch.tensor([[.00756, 0., 0.]])
    observe(t, [-8.40980, .61190, .03367], speed=(.02782, 0., 0.))
    assert t.done.item()
    assert t.ground_contact_age.item() > .05


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("direction", [-1, 1])
def test_radius_uses_current_root_distance_and_strict_greater_than_boundary(axis, direction):
    t, c, _ = tracker_fixture()
    c.robot_anchor_pos_w[:] = torch.tensor([[10., -3., 1.]])
    position = c.robot_anchor_pos_w[0].clone()
    position[axis] += direction * 20.
    observe(t, position.tolist())
    assert not t.done.item()
    position[axis] += direction * .001
    observe(t, position.tolist())
    assert t.done.item()


def test_unlaunched_ball_does_not_end_before_its_first_physical_sample():
    t, _, _ = tracker_fixture()
    t.active[:] = False
    observe(t, [0., 0., 0.], speed=(0., 0., 0.))
    assert not t.done.item()


def test_resting_contact_gaps_do_not_prevent_next_serve():
    t, _, _ = tracker_fixture()
    for i in range(110):
        observe(t, [1., 0., .0335], ground=(i % 4 == 0), speed=(.01, 0., .06 * (-1)**i))
    assert t.done.item()


def test_ball_above_speed_threshold_is_not_stopped():
    t, _, _ = tracker_fixture()
    for i in range(200):
        observe(t, [1. + i * .0003, 0., .0335], ground=True, speed=(.101, 0., 0.))
    assert not t.done.item()


def test_new_serve_does_not_reuse_the_last_balls_stop_filter():
    t, _, _ = tracker_fixture()
    observe(t, [1., 0., 0.], ground=True, speed=(.05, 0., 0.))
    assert t.done.item()
    t.reset(torch.tensor([0])); t.active[:] = True
    observe(t, [1., 0., 0.], ground=True, speed=(.2, 0., 0.))
    assert not t.done.item()
    assert t.stop_velocity[0, 0].item() == pytest.approx(.2)


def test_completed_ball_bypasses_remaining_animation_failure_timer_and_batch(monkeypatch):
    from athlete.goal_cond_tracking.mdp.phase_commands import PhaseAccelerationMultiTargetMotionCommand
    c = FullFlightMotionCommand.__new__(FullFlightMotionCommand)
    c._env = SimpleNamespace(tennis_full_flight=SimpleNamespace(done=torch.tensor([True, False, True])))
    c.auto_chain_motion = torch.tensor([True, True, False])
    c.phase = torch.tensor([.4, .4, .4])
    c.phase_rate = torch.ones(3)
    c.failure_trajectory_time_remaining = torch.full((3,), 4.)
    c.between_motion_pause_time = torch.zeros(3)
    c._sampled_pause_lengths = torch.full((3,), .5)
    c._motion_plan_batch_step = 0
    c._motion_plan_batch_interval_steps = 8
    called = []
    monkeypatch.setattr(PhaseAccelerationMultiTargetMotionCommand, "_update_command", lambda self: called.append(True))
    c._update_command()
    assert called == [True]
    torch.testing.assert_close(c.phase, torch.tensor([1., .4, .4]))
    torch.testing.assert_close(c.failure_trajectory_time_remaining, torch.tensor([0., 4., 4.]))
    assert torch.isinf(c.between_motion_pause_time[0])
    assert c._sampled_pause_lengths[0].item() == 0
    assert c._motion_plan_batch_step == 7


def test_motion_end_cannot_replace_airborne_ball(monkeypatch):
    from athlete.goal_cond_tracking.mdp.phase_commands import PhaseAccelerationMultiTargetMotionCommand
    seen = []
    monkeypatch.setattr(PhaseAccelerationMultiTargetMotionCommand, "_sample_next_motion", lambda self, ids: seen.extend(ids.tolist()))
    c = FullFlightMotionCommand.__new__(FullFlightMotionCommand)
    t = SimpleNamespace(done=torch.tensor([False, True]), reset=lambda ids: None)
    c._env = SimpleNamespace(tennis_full_flight=t)
    c._sampled_pause_lengths = torch.tensor([float("inf"), 0.])
    c._sample_next_motion(torch.tensor([0, 1]))
    assert seen == [1]
    assert c._sampled_pause_lengths[0].item() == 0


@pytest.mark.parametrize("x, sensor_prefix", [(5., "full_flight_court"), (25., "full_flight_surround")])
def test_native_mujoco_ground_contact_position_is_the_reward_source(x, sensor_prefix):
    """Drop an actual MuJoCo ball, including outside the marked court."""
    import mujoco
    import numpy as np
    from mjlab.scene import Scene

    cfg = m14_11_full_flight_env_cfg(play=True)
    model = Scene(cfg.scene, device="cpu").compile()
    model.opt.timestep = cfg.sim.mujoco.timestep
    data = mujoco.MjData(model)
    ball_body = model.geom("tennis_ball/tennis_ball_geom").bodyid[0]
    ball_joint = model.body_jntadr[ball_body]
    qadr, vadr = model.jnt_qposadr[ball_joint], model.jnt_dofadr[ball_joint]
    data.qpos[qadr:qadr + 3] = [x, 0., 1.]
    data.qvel[vadr:vadr + 3] = [0., 0., -1.]

    def sensor(field):
        names = [model.sensor(i).name for i in range(model.nsensor)
                 if model.sensor(i).name.startswith(sensor_prefix + "_")
                 and model.sensor(i).name.endswith("_" + field)]
        assert len(names) == 1
        return data.sensor(names[0]).data

    t, c, env = tracker_fixture()
    # Isolate physical landing measurement here, including the surround.
    # The separate radius test verifies the default 20m completion rule.
    c.cfg.full_flight_radius_m = 30.
    c.ball_has_been_struck[:] = True  # Isolate the landing stage of a return.
    t.net_valid[:] = True
    for _ in range(500):
        mujoco.mj_step(model, data)
        found = sensor("found")[0] > 0 and np.linalg.norm(sensor("force")) > 1e-6
        contact = torch.tensor(sensor("pos").copy(), dtype=torch.float32)[None]
        t.observe(torch.tensor(data.qpos[qadr:qadr + 3].copy(), dtype=torch.float32)[None],
                  torch.tensor(data.qvel[vadr:vadr + 3].copy(), dtype=torch.float32)[None],
                  torch.tensor([False]), torch.tensor([found]), contact, model.opt.timestep)
        if found:
            torch.testing.assert_close(c.ball_landing_position_w, contact)
            assert c.metrics["error_ball_landing"].item() == pytest.approx(abs(x - 5), abs=.01)
            assert actual_landing_reward(env, "motion").item() == pytest.approx(
                np.exp(-max(abs(x - 5) - .5, 0)**2), abs=1e-6)
            break
        assert not c.ball_landing_recorded.item()
        assert actual_landing_reward(env, "motion").item() == 0
    else:
        pytest.fail("Physical ball never reached the ground sensor")
