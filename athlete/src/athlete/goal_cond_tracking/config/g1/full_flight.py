"""m14-11 continuation, with complete ball flights and measured landing rewards."""

from dataclasses import fields

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ...mdp.full_flight import FullFlightMotionCommandCfg, actual_landing_reward, full_flight_time_out
from ...mdp.rewards import torso_upright_reward
from ...mdp.post_hit_stillness import PostHitStillnessReward
from .env_cfgs import unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg

TASK_ID = "Mjlab-Tennis-SmallCourt-M14-11-FullFlight-ActualLanding-Std0-Unitree-G1"


def m14_11_full_flight_env_cfg(play=False, *, post_hit_stillness=True):
    cfg = unitree_g1_tennis_small_court_global_root_realtime_landing_env_cfg(play)
    if not play:
        cfg.episode_length_s = 20.0
    original = cfg.commands["motion"]
    motion = FullFlightMotionCommandCfg(**{
        f.name: getattr(original, f.name) for f in fields(original) if f.init
    })
    cfg.commands["motion"] = motion
    motion.landing_target_std = (0.0, 0.0, 0.0)
    motion.landing_target_std_final = None
    cfg.curriculum.pop("landing_target_std", None)
    motion.incoming_ball_failure_trajectory_probability = 0.05
    # Candidate headings are not failure labels: the existing planner performs
    # trajectory/reachability checks, preserving reachable candidates as hits.
    motion.incoming_ball_failure_random_direction_fraction = 0.2
    motion.incoming_ball_failure_overhead_fraction = 0.2
    cfg.rewards["ball_landing_reward"].func = actual_landing_reward
    cfg.rewards["torso_upright"] = RewardTermCfg(
        func=torso_upright_reward,
        weight=1.0,
        params={"command_name": "motion", "body_name": "torso_link", "std_degrees": 30.0},
    )
    if post_hit_stillness:
        cfg.rewards["post_hit_stillness"] = RewardTermCfg(
            func=PostHitStillnessReward,
            weight=1.0,
            params={
                "command_name": "motion",
                "position_std_m": 0.25,
                "speed_std_m_s": 0.30,
            },
        )
    cfg.terminations["time_out"].func = full_flight_time_out
    for name, secondary in (
        ("court", ContactMatch(mode="geom", pattern="tennis_court/court_surface")),
        ("surround", ContactMatch(mode="geom", pattern="tennis_court/surround")),
        ("racket", ContactMatch(mode="geom", pattern="racket_ball_collision", entity="robot")),
    ):
        cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
            name=f"full_flight_{name}",
            primary=ContactMatch(mode="geom", pattern="tennis_ball_geom", entity="tennis_ball"),
            secondary=secondary, fields=("found", "force", "pos"),
            reduce="maxforce", num_slots=1, secondary_policy="error",
        ))
    return cfg
