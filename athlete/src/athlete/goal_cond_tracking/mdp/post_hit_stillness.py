"""Reward staying at the physical strike location with little root translation."""

import math

import torch


class PostHitStillnessReward:
    """Immediate post-contact XY position/speed reward until the next serve.

    The physical rally tracker records the root at the first racket contact.
    No dwell, follow-through delay, landing gate or new termination is added.
    """

    def __init__(self, cfg, env):
        self.command = env.command_manager.get_term(cfg.params["command_name"])
        for name in ("position_std_m", "speed_std_m_s"):
            value = cfg.params[name]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("active", "distance_m", "speed_m_s", "score"):
            self.command.metrics[f"post_hit_stillness_{name}"] = torch.zeros(
                env.num_envs, device=env.device
            )

    def __call__(self, env, command_name: str, position_std_m: float,
                 speed_std_m_s: float):
        del command_name
        c, tracker = self.command, env.tennis_full_flight
        active = tracker.active & c.ball_has_been_struck & ~env.reset_terminated
        distance = (c.robot_anchor_pos_w[:, :2] - tracker.hit_root_position_w[:, :2]).norm(dim=-1)
        speed = c.robot_anchor_lin_vel_w[:, :2].norm(dim=-1)
        score = torch.exp(-(distance / position_std_m).square() - (speed / speed_std_m_s).square())
        reward = torch.where(active, score, 0.0)
        values = {
            "active": active,
            "distance_m": torch.where(active, distance, 0.0),
            "speed_m_s": torch.where(active, speed, 0.0),
            "score": reward,
        }
        for name, value in values.items():
            c.metrics[f"post_hit_stillness_{name}"].copy_(value)
        return reward
