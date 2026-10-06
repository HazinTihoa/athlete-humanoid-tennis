"""M14-11 observation-robustness factorial; old tasks remain unchanged."""

from .env_cfgs import unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg
from .rl_cfg import unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg


def measurement_env_cfg(play=False, *, history_steps=8, position_noise_std=0.05):
    cfg = unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg(play)
    if history_steps not in (8, 12) or position_noise_std not in (0.05, 0.075):
        raise ValueError('Expected the 8/12 x 5/7.5cm factorial.')
    perception = dict(
        delay_mean_s=0.020, delay_std_s=0.010, delay_max_s=0.050,
        position_noise_std=position_noise_std / 10.0,
        position_bias_std=position_noise_std,
        drop_probability=0.10, burst_start_probability=0.01,
    )
    for group_name, names in (
        ('student', ('ball_position', 'ball_linear_velocity')),
        ('intent_ball_history', ('position', 'linear_velocity')),
    ):
        for name in names:
            term = cfg.observations[group_name].terms[name]
            for key in ('delay_mean_s', 'delay_std_s', 'delay_max_s', 'delay_min_s'):
                term.params.pop(key, None)
            term.params['measurement_perception'] = dict(perception)
            # Corrupt each physical packet once, not each history readout.
            term.noise = None
    lags = tuple(range((history_steps - 1) * 5, -1, -5))
    for group_name in ('intent_state_history', 'intent_ball_history'):
        for term in cfg.observations[group_name].terms.values():
            term.params.update(sample_lags=lags, buffer_length=history_steps * 5)
    motion = cfg.commands['motion']
    motion.incoming_ball_failure_trajectory_probability = 0.05
    motion.incoming_ball_failure_overhead_fraction = 0.50
    # Of the other half, equal no-net/far samples. First ball still must match.
    motion.incoming_ball_failure_no_net_fraction = 0.50
    motion.incoming_ball_failure_overhead_speed_range = (8.0, 12.0)
    motion.incoming_ball_failure_overhead_height_range = (2.2, 2.8)
    return cfg


def measurement_runner_cfg(*, history_steps=8):
    cfg = unitree_g1_tracking_tppo_intent_reference_latent_ppo_distill005_frozen_intent_epoch3_runner_cfg()
    cfg.actor.state_history_steps = history_steps
    cfg.actor.ball_history_steps = history_steps
    return cfg
