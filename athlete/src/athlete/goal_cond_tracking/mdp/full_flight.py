"""Opt-in physical rally lifetime and contact-based landing scoring for m14-11."""

from dataclasses import dataclass
from collections import deque
import math

import torch

from .phase_commands import (
    PhaseAccelerationMultiTargetMotionCommand,
    PhaseAccelerationMultiTargetMotionCommandCfg,
)
from .rewards import (
    tennis_ball_out_speed_score,
    tennis_ball_target_projected_speed,
    tennis_ball_xy_direction_score,
    tennis_net_clearance_score,
)


class FullFlightMotionCommand(PhaseAccelerationMultiTargetMotionCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        tracker = getattr(self._env, "tennis_full_flight", None)
        if tracker is not None:
            tracker.reset(env_ids)

    def _sample_next_motion(self, env_ids):
        tracker = getattr(self._env, "tennis_full_flight", None)
        if tracker is None:
            raise RuntimeError("Full-flight task requires install_tennis_ball_controller().")
        # Failed plans normally hold indefinitely once their planner timer expires.
        # Retry chaining on subsequent control steps while physical flight continues.
        blocked = env_ids[~tracker.done[env_ids]]
        self._sampled_pause_lengths[blocked] = 0.0
        ready = env_ids[tracker.done[env_ids]]
        if len(ready):
            super()._sample_next_motion(ready)
            tracker.reset(ready)

    def _update_command(self):
        tracker = getattr(self._env, "tennis_full_flight", None)
        if tracker is not None:
            ready = tracker.done & self.auto_chain_motion
            if torch.any(ready):
                # Finish through the normal phase/pause transition so its
                # reference pose and next observations are initialized together.
                # A finished physical ball need not wait for a failure timer,
                # the animation, the between-motion pause or a planning batch.
                self.phase[ready] = 1.0
                self.phase_rate[ready] = 0.0
                self.failure_trajectory_time_remaining[ready] = 0.0
                self.between_motion_pause_time[ready] = torch.inf
                self._sampled_pause_lengths[ready] = 0.0
                self._motion_plan_batch_step = self._motion_plan_batch_interval_steps - 1
        super()._update_command()

    def _debug_vis_impl(self, visualizer):
        super()._debug_vis_impl(visualizer)
        tracker = getattr(self._env, "tennis_full_flight", None)
        for batch in visualizer.get_env_indices(self.num_envs):
            if tracker is not None and batch in tracker.trails:
                points = list(tracker.trails[batch])
                for i, (start, _) in enumerate(points[:-1]):
                    end, after_hit = points[i + 1]
                    visualizer.add_cylinder(
                        start=start, end=end, radius=0.008,
                        color=(1.0, 0.45, 0.05, 0.8) if after_hit else (0.0, 0.8, 1.0, 0.8),
                        label=f"physical_ball_trail_{batch}_{i}",
                    )
            visualizer.add_sphere(
                center=self.target_landing_position_w[batch].cpu().numpy(),
                radius=0.10, color=(0.1, 1.0, 0.1, 0.9),
                label=f"fixed_landing_target_{batch}",
            )
            if self.ball_landing_recorded[batch]:
                visualizer.add_sphere(
                    center=self.ball_landing_position_w[batch].cpu().numpy(),
                    radius=0.10, color=(1.0, 0.1, 1.0, 0.9),
                    label=f"actual_first_landing_{batch}",
                )


@dataclass(kw_only=True)
class FullFlightMotionCommandCfg(PhaseAccelerationMultiTargetMotionCommandCfg):
    full_flight_radius_m: float = 20.0
    """End the rally when its distance from the current root exceeds this radius."""
    full_flight_stop_speed_m_s: float = 0.1
    """End immediately at or below this filtered speed, regardless of contact."""
    full_flight_stop_velocity_tau_s: float = 0.05
    """Filter only the stop detector, never the ball state or policy observation."""
    full_flight_min_serves_per_episode: int = 2
    """Allow a chained (possibly failed) serve before the ordinary time limit."""

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.incoming_ball_failure_trajectory_probability <= 1:
            raise ValueError("Failure trajectory probability must be in [0, 1].")
        if min(self.full_flight_radius_m, self.full_flight_stop_speed_m_s,
               self.full_flight_stop_velocity_tau_s) <= 0:
            raise ValueError("Full-flight radius, stop speed and filter tau must be positive.")
        if self.full_flight_min_serves_per_episode < 1:
            raise ValueError("Minimum serves per episode must be at least one.")

    def build(self, env):
        return FullFlightMotionCommand(self, env)


def actual_landing_reward(env, command_name: str, **score_params):
    """Consume the first physical post-hit ground contact, never a prediction.

    score_params retain m14-11's scoring constants in the saved reward config.
    The substep tracker uses them when the contact occurs.
    """
    tracker = env.tennis_full_flight
    reward = tracker.pending_landing.clone()
    tracker.pending_landing.zero_()
    return reward


def full_flight_time_out(env):
    """Defer the ordinary time limit until this ball finishes; retain fall resets."""
    command = env.tennis_full_flight.command
    # An initial serve is always reachable. If it consumes the ordinary time limit,
    # resetting immediately would starve the requested failure distribution.
    enough_serves = command.motion_chain_count >= command.cfg.full_flight_min_serves_per_episode
    return (env.episode_length_buf >= env.max_episode_length) & env.tennis_full_flight.done & enough_serves


class PhysicalRallyTracker:
    """Observe every physics substep; never write physical ball state.

    Contacts are native MuJoCo contact sensors, including the court surround.
    A pre-hit bounce cannot score. The first bounce after a racket contact is
    latched even if it occurs entirely between two policy steps.
    """

    def __init__(self, env, controller, *, verbose=False):
        self.env, self.controller, self.verbose = env, controller, verbose
        self.command = env.command_manager.get_term("motion")
        self.params = env.cfg.rewards["ball_landing_reward"].params
        n, device = env.num_envs, env.device
        self.active = torch.zeros(n, dtype=torch.bool, device=device)
        self.done = torch.zeros_like(self.active)
        self.ground_before = torch.zeros_like(self.active)
        self.net_seen = torch.zeros_like(self.active)
        self.net_valid = torch.zeros_like(self.active)
        self.speed_recorded = torch.zeros_like(self.active)
        self.hit_age = torch.zeros(n, device=device)
        self.hit_root_position_w = torch.zeros(n, 3, device=device)
        self.ground_contact_age = torch.full((n,), torch.inf, device=device)
        self.stop_velocity = torch.zeros(n, 3, device=device)
        self.stop_velocity_initialized = torch.zeros_like(self.active)
        self.pending_landing = torch.zeros(n, device=device)
        self.previous_position = torch.zeros(n, 3, device=device)
        # Only viewer environments record a trail; formal training allocates none.
        self.trails = {i: deque(maxlen=400) for i in range(n)} if verbose else {}
        self.control_steps = 0
        for name in ("ball_actual_landing_valid", "ball_actual_landing_observed",
                     "ball_actual_landing_score", "ball_full_flight_done"):
            self.command.metrics[name] = torch.zeros(n, device=device)
        print(f"[FULL FLIGHT] base=m14-11; landing=actual post-hit ground contact; "
              f"target_mean={self.command.cfg.landing_target_mean}; "
              f"target_std={self.command.cfg.landing_target_std}; "
              f"landing_weight={env.cfg.rewards['ball_landing_reward'].weight}; "
              f"failure_probability={self.command.cfg.incoming_ball_failure_trajectory_probability}; "
              f"end when distance>{self.command.cfg.full_flight_radius_m}m OR "
              f"filtered_speed<={self.command.cfg.full_flight_stop_speed_m_s}m/s", flush=True)

    def reset(self, env_ids):
        self.ground_contact_age[env_ids] = torch.inf
        self.stop_velocity[env_ids] = 0.0
        self.stop_velocity_initialized[env_ids] = False
        if self.trails:
            for index in env_ids.tolist():
                self.trails[index].clear()
        for value in (self.active, self.done, self.ground_before, self.net_seen,
                      self.net_valid, self.speed_recorded, self.hit_age, self.hit_root_position_w,
                      self.pending_landing):
            value[env_ids] = 0
        # Failed plans skip _sample_targets(), so clear their previous rally too.
        c = self.command
        for name in ("ball_has_been_struck", "ball_landing_recorded",
                     "ball_landing_position_w", "ball_hit_reward", "ball_direction_reward",
                     "ball_net_clearance_reward", "ball_out_speed_reward",
                     "ball_post_strike_steps", "ball_post_strike_elapsed_s"):
            getattr(c, name)[env_ids] = 0
        for name in ("error_ball_landing", "ball_landing_prediction_valid",
                     "ball_actual_landing_valid", "ball_actual_landing_observed",
                     "ball_actual_landing_score", "ball_full_flight_done",
                     "ball_net_crossing_height", "ball_out_speed",
                     "ball_target_projected_speed", "ball_direction_error_degrees"):
            c.metrics[name][env_ids] = 0

    def begin_control_step(self):
        launched = self.controller._launched
        new = launched & ~self.active
        self.previous_position[new] = self.ball_position()[new]
        self.active |= launched
        if self.trails and self.control_steps % 5 == 0:
            positions = self.ball_position().detach().cpu().numpy()
            hit = self.command.ball_has_been_struck.tolist()
            for i, trail in self.trails.items():
                trail.append((positions[i].copy(), hit[i]))
        if self.verbose and self.control_steps % 250 == 0:
            ball = self.controller.ball
            speeds = ball.data.data.qvel[:, ball.data.indexing.free_joint_v_adr][:, :3].norm(dim=-1)
            for i in self.trails:
                print(f"[FULL FLIGHT STATE] env={i} control_step={self.control_steps} "
                      f"ball={self.ball_position()[i].tolist()} speed={speeds[i].item():.4f}m/s "
                      f"ground={self.ground_before[i].item()} filtered_speed={self.stop_velocity[i].norm().item():.4f}m/s "
                      f"done={self.done[i].item()}", flush=True)
        self.control_steps += 1

    def ball_position(self):
        ball = self.controller.ball
        return ball.data.data.qpos[:, ball.data.indexing.free_joint_q_adr][:, :3]

    def update_substep(self, dt):
        ball = self.controller.ball
        velocity = ball.data.data.qvel[:, ball.data.indexing.free_joint_v_adr][:, :3]
        court = self.env.scene["full_flight_court"].data
        surround = self.env.scene["full_flight_surround"].data
        racket = self.env.scene["full_flight_racket"].data

        def touching(data):
            return (data.found[:, 0] > 0) & (data.force[:, 0].norm(dim=-1) > 1e-6)

        court_contact, surround_contact = touching(court), touching(surround)
        ground = court_contact | surround_contact
        contact_position = torch.where(court_contact[:, None], court.pos[:, 0], surround.pos[:, 0])
        self.observe(self.ball_position(), velocity, touching(racket), ground,
                     contact_position, dt)

    def observe(self, position, velocity, racket, ground, contact_position, dt):
        """Batched event logic, also exercised by deterministic unit tests."""
        c, p = self.command, self.params
        running = self.active & ~self.done
        already_hit = c.ball_has_been_struck.clone()
        hit = running & racket & ~already_hit
        c.ball_has_been_struck |= hit
        self.hit_root_position_w[hit] = c.robot_anchor_pos_w[hit]
        c.ball_hit_reward[hit] = 1.0
        c.ball_post_strike_elapsed_s[hit] = 0.0
        self.hit_age += already_hit * dt
        self.hit_age[hit] = 0

        # Physical crossing, interpolated across a single 2.5ms physics substep.
        net_x = self.env.scene.env_origins[:, 0] + p["net_x"]
        dx = position[:, 0] - self.previous_position[:, 0]
        crossing = (running & c.ball_has_been_struck & ~c.ball_landing_recorded
                    & ~self.net_seen & (self.previous_position[:, 0] < net_x)
                    & (position[:, 0] >= net_x) & (dx > 0))
        fraction = ((net_x - self.previous_position[:, 0]) / dx.clamp_min(1e-8)).clamp(0, 1)
        at_net = self.previous_position + fraction[:, None] * (position - self.previous_position)
        height = at_net[:, 2] - self.env.scene.env_origins[:, 2]
        valid = ((height > p.get("net_height", 0.914) + p.get("ball_radius", 0.0335))
                 & ((at_net[:, 1] - self.env.scene.env_origins[:, 1]).abs()
                    <= p["net_half_width"] + p.get("ball_radius", 0.0335)))
        self.net_seen |= crossing
        self.net_valid |= crossing & valid
        net_score = tennis_net_clearance_score(
            height, valid, maximum_full_reward_height=p["maximum_full_reward_net_height"],
            excess_height_std=p.get("excess_net_height_std", 0.25))
        c.ball_net_clearance_reward[crossing] = net_score[crossing]
        c.metrics["ball_net_crossing_height"][crossing] = height[crossing]

        # Preserve the baseline's post-impact settling delay for speed scoring.
        speed_ready = (running & already_hit & ~self.speed_recorded
                       & (self.hit_age >= p.get("prediction_delay_steps", 2) * self.env.step_dt))
        offset = c.target_landing_position_w[:, :2] - position[:, :2]
        direction = tennis_ball_xy_direction_score(velocity[:, :2], offset, std=p.get("ball_direction_std", 0.5))
        speed = velocity.norm(dim=-1)
        projected = tennis_ball_target_projected_speed(velocity[:, :2], offset)
        aligned = p.get("align_out_speed_to_landing_target", False)
        target_speed = c.target_ball_out_speed if p.get("use_analytic_target_out_speed", False) else p["target_out_speed"]
        score = tennis_ball_out_speed_score(projected if aligned else speed,
                                            target_speed=target_speed, std=p["out_speed_std"])
        if aligned:
            score *= direction
        c.ball_out_speed_reward[speed_ready] = score[speed_ready]
        c.ball_direction_reward[speed_ready] = direction[speed_ready]
        c.metrics["ball_out_speed"][speed_ready] = speed[speed_ready]
        c.metrics["ball_target_projected_speed"][speed_ready] = projected[speed_ready]
        cosine = (velocity[:, :2] * offset).sum(-1) / (velocity[:, :2].norm(dim=-1) * offset.norm(dim=-1)).clamp_min(1e-6)
        c.metrics["ball_direction_error_degrees"][speed_ready] = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))[speed_ready]
        self.speed_recorded |= speed_ready

        # Include a grounded ball struck by the racket: its continuing ground
        # contact after impact also resolves the landing stage (with no net
        # credit). Waiting for a new contact edge could strand that rally.
        landed = running & already_hit & ~c.ball_landing_recorded & ground
        distance = (contact_position[:, :2] - c.target_landing_position_w[:, :2]).norm(dim=-1)
        landing_score = torch.exp(-((distance - p.get("target_radius", 0.5)).clamp_min(0) / p["std"]).square())
        awarded = landing_score * self.net_valid
        self.pending_landing[landed] = awarded[landed]
        c.ball_landing_recorded |= landed
        c.ball_landing_position_w[landed] = contact_position[landed]
        c.metrics["error_ball_landing"][landed] = distance[landed]
        c.metrics["ball_actual_landing_observed"][landed] = 1.0
        c.metrics["ball_actual_landing_valid"][landed] = self.net_valid[landed].float()
        c.metrics["ball_actual_landing_score"][landed] = awarded[landed]

        # Contact age is diagnostic only; it must not gate the next serve.
        # The signed-velocity filter affects only lifetime detection, never
        # the physical ball state, policy observations or landing measurement.
        self.ground_contact_age = torch.where(ground, 0.0, self.ground_contact_age + dt)
        # Seed from the first physical sample, not zero: otherwise a
        # zero-initialized filter could incorrectly stop a newly launched ball.
        initialize = running & ~self.stop_velocity_initialized
        self.stop_velocity[initialize] = velocity[initialize]
        self.stop_velocity_initialized |= initialize
        self.stop_velocity.lerp_(velocity, 1.0 - math.exp(-dt / c.cfg.full_flight_stop_velocity_tau_s))
        stopped = self.stop_velocity.norm(dim=-1) <= c.cfg.full_flight_stop_speed_m_s
        outside = (position - c.robot_anchor_pos_w).norm(dim=-1) > c.cfg.full_flight_radius_m
        # These are the only two completion conditions for a launched ball.
        # Neither ground contact nor an outstanding landing measurement blocks
        # completion. Unobserved landings receive no invented/predicted reward.
        completed = running & (outside | stopped)
        self.done |= completed
        c.metrics["ball_full_flight_done"].copy_(self.done.float())
        self.previous_position.copy_(position)
        self.ground_before.copy_(ground)
        if self.verbose:
            for index in torch.where(hit)[0].tolist():
                print(f"[RACKET CONTACT] env={index} world_xyz={position[index].tolist()}", flush=True)
            for index in torch.where(landed)[0].tolist():
                print(f"[ACTUAL LANDING] env={index} world_xyz={contact_position[index].tolist()} "
                      f"target={c.target_landing_position_w[index].tolist()} "
                      f"xy_error={distance[index].item():.3f}m net_valid={self.net_valid[index].item()} "
                      f"score={awarded[index].item():.6g}", flush=True)
            for index in torch.where(completed)[0].tolist():
                print(f"[FULL FLIGHT END] env={index} reason={'radius' if outside[index] else 'stopped'} "
                      f"hit={c.ball_has_been_struck[index].item()} "
                      f"actual_landing={c.ball_landing_recorded[index].item()}", flush=True)


def install_full_flight_tracker(env, controller, *, verbose=False):
    tracker = PhysicalRallyTracker(env, controller, verbose=verbose)
    env.tennis_full_flight = tracker
    original_update = env.scene.update

    def update(dt):
        original_update(dt)
        tracker.update_substep(dt)

    env.scene.update = update
    return tracker
