"""Shared physical contract for tennis-ball collection, play, and training.

The air model is deliberately implemented as an external wrench on the tennis
ball rather than MuJoCo's global fluid option. A non-zero global air density
would otherwise apply default fluid drag to every dynamic G1 link.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import numpy as np
import torch


@dataclass(frozen=True)
class TennisAirPhysicsCfg:
    """Still-air properties and ball-only aerodynamic coefficients."""

    density_kg_m3: float = 1.225
    dynamic_viscosity_pa_s: float = 1.81e-5
    wind_w: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class TennisBallPhysicsCfg:
    """ITF Type 2 ball geometry, contact material, and aerodynamic constants."""

    mass_kg: float = 0.0577
    radius_m: float = 0.0335
    rgba: tuple[float, float, float, float] = (0.75, 0.90, 0.08, 1.0)
    geom_friction: tuple[float, float, float] = (0.55, 0.02, 0.002)
    geom_solref: tuple[float, float] = (0.020, 0.300)
    geom_solimp: tuple[float, float, float, float, float] = (
        0.93,
        0.99,
        0.001,
        0.5,
        2.0,
    )
    drag_coefficient: float = 0.55
    magnus_coefficient: float = 0.0
    angular_drag_coefficient: float = 0.01

    @property
    def diameter_m(self) -> float:
        return 2.0 * self.radius_m

    @property
    def cross_section_m2(self) -> float:
        return pi * self.radius_m**2

    @property
    def volume_m3(self) -> float:
        return (4.0 / 3.0) * pi * self.radius_m**3

    @property
    def fluid_angular_inertia_m5(self) -> float:
        """Sphere term used by MJWarp's ellipsoid angular-fluid model."""
        return (8.0 / 15.0) * pi * self.radius_m**5


@dataclass(frozen=True)
class TennisCourtPhysicsCfg:
    """Hard-court geometry and ball contact response."""

    net_x_m: float = 5.6
    net_height_m: float = 0.914
    net_half_width_m: float = 5.485
    far_service_line_x_m: float = 12.0
    restitution: float = 0.745
    court_solref_time_constant_s: float = 0.065
    court_solimp: tuple[float, float, float, float, float] = (
        0.925,
        0.985,
        0.0015,
        0.5,
        2.0,
    )
    court_friction: tuple[float, float, float, float, float] = (
        0.62,
        0.62,
        0.02,
        0.002,
        0.002,
    )
    surround_friction: tuple[float, float, float, float, float] = (
        0.8,
        0.8,
        0.02,
        0.002,
        0.002,
    )
    net_solref: tuple[float, float] = (0.013, 0.2275)
    net_solimp: tuple[float, float, float, float, float] = (
        0.925,
        0.985,
        0.0015,
        0.5,
        2.0,
    )
    net_friction: tuple[float, float, float, float, float] = (
        0.55,
        0.55,
        0.02,
        0.002,
        0.002,
    )
    racket_solref: tuple[float, float] = (0.020, 0.300)
    racket_friction: tuple[float, float, float, float, float] = (
        0.55,
        0.55,
        0.02,
        0.002,
        0.002,
    )
    restitution_samples: tuple[float, ...] = (
        0.5373,
        0.6120,
        0.6368,
        0.6500,
        0.6850,
        0.7593,
        0.8389,
        0.9240,
    )
    contact_damping_samples: tuple[float, ...] = (
        0.120,
        0.105,
        0.100,
        0.095,
        0.085,
        0.072,
        0.060,
        0.056,
    )


@dataclass(frozen=True)
class TennisDomainRandomizationCfg:
    """Restricted train-time randomization for incoming-ball distillation."""

    ball_mass_kg: tuple[float, float] = (0.0560, 0.0594)
    court_restitution: tuple[float, float] = (0.70, 0.80)
    ground_tangent_speed_retention: tuple[float, float] = (0.75, 0.90)
    drag_coefficient: tuple[float, float] = (0.50, 0.65)
    racket_mass_kg: tuple[float, float] = (0.27, 0.33)
    racket_restitution: tuple[float, float] = (0.54, 0.68)
    racket_com_offset_x_m: tuple[float, float] = (-0.015, 0.015)
    racket_com_offset_y_m: tuple[float, float] = (-0.010, 0.010)
    racket_com_offset_z_m: tuple[float, float] = (-0.015, 0.015)
    explicit_ground_rebound: bool = False


@dataclass(frozen=True)
class TennisPhysicsCfg:
    """Single source of truth for all tennis-ball physical parameters."""

    ball: TennisBallPhysicsCfg = field(default_factory=TennisBallPhysicsCfg)
    court: TennisCourtPhysicsCfg = field(default_factory=TennisCourtPhysicsCfg)
    air: TennisAirPhysicsCfg = field(default_factory=TennisAirPhysicsCfg)


STANDARD_TENNIS_PHYSICS = TennisPhysicsCfg()
STANDARD_TENNIS_DOMAIN_RANDOMIZATION = TennisDomainRandomizationCfg()
# Below the measured table, use only a bounded linear extension for DR.
MIN_SUPPORTED_RESTITUTION = 0.5


def contact_damping_for_restitution(
    restitution: float,
    *,
    physics: TennisPhysicsCfg = STANDARD_TENNIS_PHYSICS,
) -> float:
    """Map restitution to damping; 0.5..0.5373 uses uncalibrated extrapolation."""
    if not np.isfinite(restitution):
        raise ValueError("restitution must be finite")
    samples = np.asarray(physics.court.restitution_samples, dtype=np.float64)
    damping = np.asarray(physics.court.contact_damping_samples, dtype=np.float64)
    lower = MIN_SUPPORTED_RESTITUTION
    upper = float(samples[-1])
    if not lower <= restitution <= upper:
        raise ValueError(
            f"restitution must be in the supported range [{lower}, {upper}]"
        )
    if restitution < samples[0]:
        slope = (damping[1] - damping[0]) / (samples[1] - samples[0])
        return float(damping[0] + (restitution - samples[0]) * slope)
    return float(np.interp(restitution, samples, damping))


def tennis_ball_aerodynamic_wrench_numpy(
    linear_velocity_w: np.ndarray,
    angular_velocity_w: np.ndarray,
    *,
    physics: TennisPhysicsCfg = STANDARD_TENNIS_PHYSICS,
    drag_coefficient: float | np.ndarray | None = None,
    magnus_coefficient: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ball-only drag/Magnus force and angular-drag torque in world frame."""
    velocity = np.asarray(linear_velocity_w, dtype=np.float64)
    angular_velocity = np.asarray(angular_velocity_w, dtype=np.float64)
    wind = np.asarray(physics.air.wind_w, dtype=np.float64)
    relative_velocity = velocity - wind
    speed = np.linalg.vector_norm(relative_velocity, axis=-1, keepdims=True)
    angular_speed = np.linalg.vector_norm(angular_velocity, axis=-1, keepdims=True)
    ball = physics.ball
    air = physics.air
    drag = ball.drag_coefficient if drag_coefficient is None else drag_coefficient
    magnus_scale = (
        ball.magnus_coefficient if magnus_coefficient is None else magnus_coefficient
    )
    drag = np.asarray(drag, dtype=velocity.dtype)
    magnus_scale = np.asarray(magnus_scale, dtype=velocity.dtype)
    if drag.ndim == velocity.ndim - 1:
        drag = drag[..., None]
    if magnus_scale.ndim == velocity.ndim - 1:
        magnus_scale = magnus_scale[..., None]
    linear_drag = (
        -(
            0.5 * air.density_kg_m3 * drag * ball.cross_section_m2 * speed
            + 3.0 * pi * ball.diameter_m * air.dynamic_viscosity_pa_s
        )
        * relative_velocity
    )
    magnus = (
        air.density_kg_m3
        * ball.volume_m3
        * magnus_scale
        * np.cross(angular_velocity, relative_velocity)
    )
    angular_drag = (
        -(
            air.density_kg_m3
            * ball.angular_drag_coefficient
            * ball.fluid_angular_inertia_m5
            * angular_speed
            + pi * ball.diameter_m**3 * air.dynamic_viscosity_pa_s
        )
        * angular_velocity
    )
    return linear_drag + magnus, angular_drag


def tennis_ball_aerodynamic_wrench_torch(
    linear_velocity_w: torch.Tensor,
    angular_velocity_w: torch.Tensor,
    *,
    physics: TennisPhysicsCfg = STANDARD_TENNIS_PHYSICS,
    drag_coefficient: float | torch.Tensor | None = None,
    magnus_coefficient: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch equivalent of :func:`tennis_ball_aerodynamic_wrench_numpy`."""
    wind = torch.tensor(
        physics.air.wind_w,
        dtype=linear_velocity_w.dtype,
        device=linear_velocity_w.device,
    )
    relative_velocity = linear_velocity_w - wind
    speed = torch.linalg.vector_norm(relative_velocity, dim=-1, keepdim=True)
    angular_speed = torch.linalg.vector_norm(angular_velocity_w, dim=-1, keepdim=True)
    ball = physics.ball
    air = physics.air
    drag = ball.drag_coefficient if drag_coefficient is None else drag_coefficient
    magnus_scale = (
        ball.magnus_coefficient if magnus_coefficient is None else magnus_coefficient
    )
    drag = torch.as_tensor(
        drag, dtype=linear_velocity_w.dtype, device=linear_velocity_w.device
    )
    magnus_scale = torch.as_tensor(
        magnus_scale,
        dtype=linear_velocity_w.dtype,
        device=linear_velocity_w.device,
    )
    if drag.ndim == linear_velocity_w.ndim - 1:
        drag = drag.unsqueeze(-1)
    if magnus_scale.ndim == linear_velocity_w.ndim - 1:
        magnus_scale = magnus_scale.unsqueeze(-1)
    linear_drag = (
        -(
            0.5 * air.density_kg_m3 * drag * ball.cross_section_m2 * speed
            + 3.0 * pi * ball.diameter_m * air.dynamic_viscosity_pa_s
        )
        * relative_velocity
    )
    magnus = (
        air.density_kg_m3
        * ball.volume_m3
        * magnus_scale
        * torch.linalg.cross(angular_velocity_w, relative_velocity)
    )
    angular_drag = (
        -(
            air.density_kg_m3
            * ball.angular_drag_coefficient
            * ball.fluid_angular_inertia_m5
            * angular_speed
            + pi * ball.diameter_m**3 * air.dynamic_viscosity_pa_s
        )
        * angular_velocity_w
    )
    return linear_drag + magnus, angular_drag
