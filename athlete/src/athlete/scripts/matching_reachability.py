"""Geometric eligibility of incoming trajectories, independent of motion matching.

The height test is evaluated only at samples inside the robot's horizontal
reach disk and the selected bounce phase. It never uses a whole-flight apex.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ReachabilityDiagnostics:
    within_radius: torch.Tensor
    has_height: torch.Tensor
    region_max_height: torch.Tensor
    region_min_xy: torch.Tensor
    first_bounce_radius: torch.Tensor
    has_first_bounce: torch.Tensor


def trajectory_reachability(
    positions: torch.Tensor,
    bounce_counts: torch.Tensor,
    *,
    reach_radius: float = 3.0,
    minimum_reach_height: float = 0.5,
    bounce_scope: str = "post_first",
    root_positions_xy: torch.Tensor | None = None,
) -> ReachabilityDiagnostics:
    """Return reach-disk diagnostics at the existing trajectory sample times.

    ``post_first`` includes bounce count 1; ``pre_second`` includes counts 0
    and 1. Distance and height thresholds are inclusive. ``region_min_xy`` is
    the closest horizontal approach during that bounce phase (even if the
    trajectory never enters the disk). No valid phase yields +inf distance;
    no sample inside the disk yields -inf maximum height. Missing first
    bounce yields NaN first-bounce radius.
    """
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (N, T, 3)")
    if positions.shape[1] == 0 or bounce_counts.shape != positions.shape[:2]:
        raise ValueError("bounce_counts must match a nonempty trajectory time axis")
    if not positions.is_floating_point():
        raise ValueError("positions must be floating point")
    if not math.isfinite(reach_radius) or reach_radius <= 0:
        raise ValueError("reach_radius must be finite and positive")
    if not math.isfinite(minimum_reach_height) or minimum_reach_height < 0:
        raise ValueError("minimum_reach_height must be finite and nonnegative")
    if bounce_scope not in ("post_first", "pre_second"):
        raise ValueError("bounce_scope must be post_first or pre_second")
    n = positions.shape[0]
    if root_positions_xy is None:
        roots = positions.new_zeros((n, 2))
    else:
        roots = torch.as_tensor(root_positions_xy, device=positions.device, dtype=positions.dtype)
        if roots.shape == (2,):
            roots = roots.expand(n, 2)
        if roots.shape != (n, 2):
            raise ValueError("root_positions_xy must have shape (2,) or (N, 2)")

    finite = torch.isfinite(positions).all(dim=-1)
    phase = bounce_counts == 1 if bounce_scope == "post_first" else ((bounce_counts >= 0) & (bounce_counts < 2))
    phase = phase & finite
    xy_distance = torch.linalg.vector_norm(positions[..., :2] - roots[:, None, :], dim=-1)
    region = phase & (xy_distance <= reach_radius)
    within_radius = region.any(dim=1)
    region_max_height = positions[..., 2].masked_fill(~region, -torch.inf).max(dim=1).values
    has_height = within_radius & (region_max_height >= minimum_reach_height)
    region_min_xy = xy_distance.masked_fill(~phase, torch.inf).min(dim=1).values

    first = bounce_counts >= 1
    has_first_bounce = first.any(dim=1)
    first_indices = first.to(torch.int64).argmax(dim=1)
    first_bounce_radius = xy_distance[torch.arange(n, device=positions.device), first_indices]
    first_bounce_radius = first_bounce_radius.masked_fill(~has_first_bounce, torch.nan)
    return ReachabilityDiagnostics(
        within_radius=within_radius,
        has_height=has_height,
        region_max_height=region_max_height,
        region_min_xy=region_min_xy,
        first_bounce_radius=first_bounce_radius,
        has_first_bounce=has_first_bounce,
    )
