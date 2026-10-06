"""Finite-duration external disturbances for tennis training."""

import torch
from mjlab.envs.mdp.events import apply_body_impulse
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform


class apply_single_body_impulse(apply_body_impulse):
    """Push one randomly selected body per environment, never overlapping bodies.

    Reuse mjlab's timers, reset cleanup and force-arrow visualization. Each pulse
    selects one body uniformly and holds its world-frame wrench at the CoM until
    expiry. The shared cooldown prevents another body from starting mid-pulse.
    """

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        force_range: tuple[float, float],
        torque_range: tuple[float, float],
        duration_s: tuple[float, float],
        cooldown_s: tuple[float, float],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        del env, env_ids, cooldown_s, asset_cfg
        # Only elapsed idle time consumes the cooldown. An expiring pulse gets
        # a fresh, full cooldown before either body can be selected again.
        self._interval_time_left[~self._active] -= self._step_dt
        self._time_remaining[self._active] -= self._step_dt
        expired_ids = (self._active & (self._time_remaining <= 0)).nonzero(
            as_tuple=False
        ).flatten()
        if expired_ids.numel():
            zeros = torch.zeros(
                (len(expired_ids), self._num_bodies, 3), device=self._device
            )
            self._asset.write_external_wrench_to_sim(
                zeros, zeros, env_ids=expired_ids, body_ids=self._body_ids
            )
            self._active[expired_ids] = False
            self._time_remaining[expired_ids] = 0.0
            self._interval_time_left[expired_ids] = self._sample_cooldown(len(expired_ids))

        trigger_ids = ((~self._active) & (self._interval_time_left <= 0)).nonzero(
            as_tuple=False
        ).flatten()
        n = len(trigger_ids)
        if not n:
            return

        selected = torch.randint(self._num_bodies, (n,), device=self._device)
        rows = torch.arange(n, device=self._device)
        forces = torch.zeros((n, self._num_bodies, 3), device=self._device)
        torques = torch.zeros_like(forces)
        forces[rows, selected] = sample_uniform(*force_range, (n, 3), self._device)
        torques[rows, selected] = sample_uniform(*torque_range, (n, 3), self._device)
        # The other foot is explicitly zero at the same write; no physics step
        # can see a pulse on both feet, including when the selected side changes.
        self._asset.write_external_wrench_to_sim(
            forces, torques, env_ids=trigger_ids, body_ids=self._body_ids
        )
        self._time_remaining[trigger_ids] = sample_uniform(
            *duration_s, (n,), self._device
        )
        self._active[trigger_ids] = True
