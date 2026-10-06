"""Fast failure launches: aim over G1 rather than at an arbitrary far bounce."""

import torch


def sample_overhead_candidates(roots_xy, position_min, position_max, speed_range, height_range):
    if not (0 < speed_range[0] <= speed_range[1] and 2.0 <= height_range[0] <= height_range[1]):
        raise ValueError('Invalid overhead launch ranges.')
    n = len(roots_xy)
    p = position_min + torch.rand(n, 3, device=roots_xy.device, dtype=roots_xy.dtype) * (position_max - position_min)
    direction = roots_xy - p[:, :2]
    distance = direction.norm(dim=-1).clamp_min(0.01)
    speed = torch.empty_like(distance).uniform_(*speed_range)
    time = distance / speed
    height = torch.empty_like(distance).uniform_(*height_range)
    v = torch.zeros_like(p)
    v[:, :2] = direction / distance[:, None] * speed[:, None]
    v[:, 2] = (height - p[:, 2] + 0.5 * 9.81 * time.square()) / time
    return p, v


def select_overhead_candidates(trajectories, roots_xy):
    p, v = trajectories.positions, trajectories.velocities
    before, after = p[:, :-1], p[:, 1:]
    crosses = (before[..., 0] > roots_xy[:, None, 0]) & (after[..., 0] <= roots_xy[:, None, 0])
    alpha = ((before[..., 0] - roots_xy[:, None, 0]) / (before[..., 0] - after[..., 0]).clamp_min(1e-6)).clamp(0, 1)
    crossing = before + alpha[..., None] * (after - before)
    speed = torch.lerp(v[:, :-1], v[:, 1:], alpha[..., None]).norm(dim=-1)
    passes_overhead = (crosses & (trajectories.bounce_counts[:, 1:] == 0)
                       & ((crossing[..., 1] - roots_xy[:, None, 1]).abs() < 0.30)
                       & (crossing[..., 2] >= 2.0) & (crossing[..., 2] <= 3.2)
                       & (speed >= 6.0))
    # The ball must keep moving behind the root before first bounce; no
    # post-bounce return to the robot's reachable horizontal neighborhood.
    later_bounces = trajectories.bounce_counts >= 1
    distance_after_bounce = (p[..., :2] - roots_xy[:, None]).norm(dim=-1)
    far_bounce = (~later_bounces | (distance_after_bounce > 1.5)).all(dim=1)
    return passes_overhead.any(dim=1) & trajectories.net_cleared & far_bounce
