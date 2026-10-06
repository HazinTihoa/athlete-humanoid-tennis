"""m14-11 FullFlight return-home variant, retaining its single-foot disturbance."""

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import GaussianNoiseCfg

from ...mdp.disturbances import apply_single_body_impulse
from ...mdp.return_home_speed import ReturnHomeSpeedReward
from .full_flight import m14_11_full_flight_env_cfg

TASK_ID = "Mjlab-Tennis-SmallCourt-M14-11-FullFlight-ActualLanding-Std0-ReturnHomeSpeed-Unitree-G1"


def m14_11_full_flight_return_home_env_cfg(play=False):
    # Stay-at-hit and return-home have different position targets; the user
    # requested stillness only for the non-return-home FullFlight task.
    cfg = m14_11_full_flight_env_cfg(play, post_hit_stillness=False)
    # Foot disturbances were removed only from the non-return-home base.
    # Preserve this existing comparison configuration's disturbance settings.
    if not play:
        # Keep this existing comparison's nominal episode length unchanged.
        cfg.episode_length_s = 10.0
        # Preserve this comparison's original doubled racket observation noise.
        cfg.observations["student"].terms["sweet_spot_position"].noise = GaussianNoiseCfg(
            mean=0.0, std=0.10,
        )
        cfg.observations["student"].terms["sweet_spot_linear_velocity"].noise = GaussianNoiseCfg(
            mean=0.0, std=0.40,
        )
        cfg.events["foot_force"] = EventTermCfg(
            func=apply_single_body_impulse,
            mode="step",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
                ),
                "force_range": (-20.0, 20.0),
                "torque_range": (0.0, 0.0),
                "duration_s": (0.1, 0.2),
                "cooldown_s": (2.0, 4.0),
            },
        )
    cfg.rewards["return_home_speed"] = RewardTermCfg(
        func=ReturnHomeSpeedReward,
        weight=1.0,
        params={
            "command_name": "motion",
            "time_scale_s": 2.0,
            "arrival_radius_m": 0.25,
            "arrival_speed_m_s": 0.3,
            "arrival_bonus": 0.5,
            "followthrough_s": 0.3,
            "miss_grace_s": 0.3,
            "miss_timeout_s": 0.8,
            "behind_margin_m": 0.3,
        },
    )
    return cfg
