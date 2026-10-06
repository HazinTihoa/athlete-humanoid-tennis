"""FullFlight continuation with mutually exclusive left/right foot force pulses."""

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...mdp.disturbances import apply_single_body_impulse
from .full_flight import m14_11_full_flight_env_cfg

TASK_ID = "Mjlab-Tennis-SmallCourt-M14-11-FullFlight-ActualLanding-Std0-FootForce20-Unitree-G1"


def m14_11_full_flight_foot_force_env_cfg(play=False):
    cfg = m14_11_full_flight_env_cfg(play)
    if not play:
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
    return cfg
