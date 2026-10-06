"""Reward prompt, settled recovery without modifying physical rally timing."""

import math

import torch
from mjlab.utils.lab_api.math import matrix_from_quat


class ReturnHomeSpeedReward:
    """Time-discounted new inward progress and one settled arrival per rally.

    Progress is measured against the best distance so far, so retreating cannot
    earn the same progress twice. Dividing by dt gives a progress-speed reward;
    exp(-elapsed/tau) also makes the integrated reward favor faster recovery.
    """

    def __init__(self, cfg, env):
        self.command = env.command_manager.get_term(cfg.params["command_name"])
        p = cfg.params
        for name in ("time_scale_s", "arrival_radius_m", "arrival_speed_m_s"):
            if not math.isfinite(p[name]) or p[name] <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("arrival_bonus", "followthrough_s", "miss_grace_s",
                     "miss_timeout_s", "behind_margin_m"):
            if not math.isfinite(p[name]) or p[name] < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if p["miss_timeout_s"] < p["miss_grace_s"]:
            raise ValueError("miss_timeout_s must be at least miss_grace_s")
        n, device = env.num_envs, env.device
        self.round = torch.full((n,), -1, dtype=torch.long, device=device)
        self.started = torch.zeros(n, dtype=torch.bool, device=device)
        self.eligible = torch.zeros_like(self.started)
        self.arrived = torch.zeros_like(self.started)
        self.elapsed_s = torch.zeros(n, device=device)
        self.best_distance = torch.zeros_like(self.elapsed_s)
        self.initial_distance = torch.zeros_like(self.elapsed_s)
        self.arrival_time = torch.zeros_like(self.elapsed_s)
        self.reason = torch.zeros_like(self.round)
        for name in ("active", "distance_m", "progress_speed_m_s", "elapsed_s",
                     "arrived", "arrival_time_s", "initial_distance_m", "hit", "miss", "failure"):
            self.command.metrics[f"return_home_{name}"] = torch.zeros(n, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.round[ids] = -1
        for value in (self.started, self.eligible, self.arrived, self.elapsed_s,
                      self.best_distance, self.initial_distance, self.arrival_time, self.reason):
            value[ids] = 0

    def __call__(self, env, command_name: str, time_scale_s: float,
                 arrival_radius_m: float, arrival_speed_m_s: float,
                 arrival_bonus: float, followthrough_s: float,
                 miss_grace_s: float, miss_timeout_s: float, behind_margin_m: float):
        del command_name
        c, tracker, dt = self.command, env.tennis_full_flight, env.step_dt
        # A new serve resets scoring, but home remains the episode-start pelvis
        # position. This term neither pauses a serve nor changes any observation.
        changed = self.round != c.motion_chain_count
        self.reset(changed)
        self.round.copy_(c.motion_chain_count)
        distance = (c.robot_anchor_pos_w[:, :2] - c.startup_anchor_pos_w[:, :2]).norm(dim=-1)
        speed = c.robot_anchor_lin_vel_w[:, :2].norm(dim=-1)

        hit = c.ball_has_been_struck & (tracker.hit_age >= followthrough_s)
        forward = matrix_from_quat(c.startup_anchor_yaw_w)[:, :2, 0]
        delta = tracker.ball_position()[:, :2] - c.robot_anchor_pos_w[:, :2]
        behind = (delta * forward).sum(-1) <= -behind_margin_m
        overdue = c.strike_time_error_s
        missed = (c.trajectory_match_valid & ~c.phase_hold & ~c.ball_has_been_struck
                  & (overdue >= miss_grace_s) & (behind | (overdue >= miss_timeout_s)))
        failed = (~c.trajectory_match_valid & ~c.ball_has_been_struck
                  & (c.failure_trajectory_time_remaining <= 0))
        ready = tracker.active & (hit | missed | failed)
        newly_started = ready & ~self.started
        self.elapsed_s += self.started * (~self.arrived) * dt
        self.elapsed_s[newly_started] = 0
        self.best_distance[newly_started] = distance[newly_started].clamp_min(arrival_radius_m)
        self.initial_distance[newly_started] = distance[newly_started]
        self.eligible |= newly_started & (distance > arrival_radius_m)
        self.started |= newly_started
        reason = hit.long() + 2 * missed.long() + 3 * failed.long()
        self.reason[newly_started] = reason[newly_started]

        active = self.started & self.eligible & ~self.arrived & ~env.reset_terminated
        bounded_distance = distance.clamp_min(arrival_radius_m)
        progress = (self.best_distance - bounded_distance).clamp_min(0) * active
        arrived = active & (distance <= arrival_radius_m) & (speed <= arrival_speed_m_s)
        discount = torch.exp(-self.elapsed_s / time_scale_s)
        reward = (progress + arrived * arrival_bonus) * discount / dt
        self.best_distance.copy_(torch.where(
            active, torch.minimum(self.best_distance, bounded_distance), self.best_distance
        ))
        self.arrived |= arrived
        self.arrival_time[arrived] = self.elapsed_s[arrived]

        values = {
            "active": active & ~self.arrived, "distance_m": distance,
            "progress_speed_m_s": progress / dt, "elapsed_s": self.elapsed_s,
            "arrived": self.arrived, "arrival_time_s": self.arrival_time,
            "initial_distance_m": self.initial_distance,
            "hit": self.reason == 1, "miss": self.reason == 2, "failure": self.reason == 3,
        }
        for name, value in values.items():
            c.metrics[f"return_home_{name}"].copy_(value)
        return reward
