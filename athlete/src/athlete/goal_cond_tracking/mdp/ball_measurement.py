"""Batched, causal counterpart of deploy.utils.natnet_bridge_state.

Only position packets enter the estimator. Synthetic predictions never enter
its seven-valid-packet derivative fit. No ROS or simulator velocity is needed.
"""

import math

import torch


@torch.jit.script
def predict_step(p: torch.Tensor, v: torch.Tensor, dt: float):
    bounced = torch.zeros_like(p[:, 0], dtype=torch.bool)
    p = p.clone()
    p[:, 2] = p[:, 2].clamp_min(0.0335)
    steps = int(math.ceil(dt / 0.0025))
    h = dt / steps
    for _ in range(steps):
        drag = 0.5 * 1.225 * 0.55 * math.pi * 0.0335**2 / 0.0577
        acc = -drag * torch.linalg.vector_norm(v, dim=-1, keepdim=True) * v
        acc[:, 2] -= 9.81
        vn = v + acc * h
        pn = p + 0.5 * (v + vn) * h
        hit = (pn[:, 2] < 0.0335) & (vn[:, 2] < 0.0)
        pn[:, 2] = torch.where(hit, torch.full_like(pn[:, 2], 0.0335), pn[:, 2])
        vn[:, :2] *= torch.where(hit, 0.825, 1.0)[:, None]
        vn[:, 2] *= torch.where(hit, -0.745, 1.0)
        bounced |= hit
        p, v = pn, vn
    return p, v, bounced


class BatchedBallEstimator:
    """Same 7-point quadratic derivative, 2-point fallback and 0.55 blend as deploy.

    Timestamps are relative to each environment, not wall time. ``update`` runs
    at a fixed packet opportunity period; accepted packet timestamps may have
    gaps. Prediction uses nominal physics, never domain-randomized truth.
    """

    def __init__(self, count, device, dt=0.02, dtype=torch.float32):
        self.dt = dt
        self.p = torch.zeros(count, 3, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.p)
        self.times = torch.zeros(count, 7, device=device, dtype=dtype)
        self.positions = torch.zeros(count, 7, 3, device=device, dtype=dtype)
        self.count = torch.zeros(count, device=device, dtype=torch.long)
        self.age = torch.zeros(count, device=device, dtype=dtype)
        self.time = torch.zeros_like(self.age)
        self.static_age = torch.zeros_like(self.age)
        self.static = torch.ones(count, device=device, dtype=torch.bool)
        self.initialized = torch.zeros_like(self.static)
        self.bounced = torch.zeros_like(self.static)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.initialized[ids] = False

    @torch.no_grad()
    def update(self, measurement, valid, active=None):
        saved = None if active is None else {
            key: value.clone() for key, value in vars(self).items() if isinstance(value, torch.Tensor)
        }
        fresh = ~self.initialized
        self.p[fresh] = measurement[fresh]
        self.v[fresh] = 0.0
        self.count[fresh] = 0
        self.age[fresh] = 0.0
        self.time[fresh] = 0.0
        self.static_age[fresh] = 0.0
        self.static[fresh] = True
        self.bounced[fresh] = False
        self.initialized[:] = True
        self.time += self.dt
        self.age += self.dt
        pp, pv, bounce = predict_step(self.p, self.v, self.dt)
        self.p = torch.where(self.static[:, None], self.p, pp)
        self.v = torch.where(self.static[:, None], torch.zeros_like(pv), pv)
        self.bounced |= bounce & ~self.static
        valid = valid | fresh
        reset_track = valid & (self.age > 0.50)
        clear_fit = valid & ((self.age > 0.10 + 1e-6) | self.bounced | reset_track)
        self.count[clear_fit] = 0
        self.v[reset_track] = 0.0
        self.static[reset_track] = True
        self.static_age[reset_track] = 0.0

        self.positions = torch.where(
            valid[:, None, None],
            torch.cat((self.positions[:, 1:], measurement[:, None]), dim=1),
            self.positions,
        )
        self.times = torch.where(
            valid[:, None], torch.cat((self.times[:, 1:], self.time[:, None]), dim=1), self.times
        )
        self.count = (self.count + valid.long()).clamp(max=7)
        mask = torch.arange(7, device=measurement.device)[None] >= 7 - self.count[:, None]
        relative_t = self.times - self.times[:, -1:]
        scale = torch.where(mask, relative_t.abs(), 0.0).amax(dim=1).clamp_min(self.dt)
        t = relative_t / scale[:, None]
        design = torch.stack((torch.ones_like(t), t, t.square()), dim=-1)
        design *= mask[:, :, None]
        lhs = design.transpose(1, 2) @ design
        rhs = design.transpose(1, 2) @ self.positions
        enough = self.count >= 5
        lhs = torch.where(enough[:, None, None], lhs, torch.eye(3, device=lhs.device, dtype=lhs.dtype))
        coeff, info = torch.linalg.solve_ex(lhs, rhs)
        fit = coeff[:, 1] / scale[:, None]
        good_fit = enough & (info == 0) & torch.isfinite(fit).all(dim=-1) & (fit.norm(dim=-1) <= 40.0)
        rough_dt = (self.times[:, -1] - self.times[:, -2]).clamp_min(1e-6)
        rough = (self.positions[:, -1] - self.positions[:, -2]) / rough_dt[:, None]
        rough_ok = (self.count >= 2) & (rough.norm(dim=-1) <= 40.0) & torch.isfinite(rough).all(dim=-1)
        fitted = torch.where(good_fit[:, None], fit, rough)
        use_fit = valid & (good_fit | rough_ok)
        self.v = torch.where(use_fit[:, None], 0.45 * self.v + 0.55 * fitted, self.v)
        self.p = torch.where(valid[:, None], measurement, self.p)
        moving = self.v.norm(dim=-1) > 0.15
        self.static_age = torch.where(valid & moving, 0.0, self.static_age + self.dt)
        self.static = torch.where(valid & moving, False, self.static)
        self.static |= valid & ~moving & (self.static_age >= 0.15)
        self.v = torch.where((valid & self.static)[:, None], 0.0, self.v)
        self.age = torch.where(valid, 0.0, self.age)
        self.bounced &= ~valid
        if saved is not None:
            for key, old in saved.items():
                value = getattr(self, key)
                mask_shape = (len(active),) + (1,) * (value.ndim - 1)
                setattr(self, key, torch.where(active.reshape(mask_shape), value, old))
        return self.p, self.v


class MeasurementBallPerception:
    """Shared delayed/noisy packet stream for current and latent Student inputs."""

    def __init__(self, env, ball_entity_name, cfg):
        self.cfg = dict(cfg)
        self.ball_entity_name = ball_entity_name
        self.dt = float(env.step_dt)
        if not math.isclose(self.dt, 0.02):
            raise ValueError("Measurement experiments use a 50 Hz packet stream.")
        self.length = math.ceil(cfg['delay_max_s'] / self.dt) + 1
        self.history = torch.zeros(self.length, env.num_envs, 3, device=env.device)
        self.estimator = BatchedBallEstimator(env.num_envs, env.device, self.dt)
        self.delay = torch.zeros(env.num_envs, device=env.device)
        self.bias = torch.zeros(env.num_envs, 3, device=env.device)
        self.remaining = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.ids = torch.arange(env.num_envs, device=env.device)
        self.fresh = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.cursor = -1
        self.last_step = None
        self.output = None
        self.packet_valid = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.fresh[ids] = True
        self.estimator.reset(ids)
        self.output = None

    @torch.no_grad()
    def update(self, env):
        step = int(env.common_step_counter)
        if self.last_step == step and self.output is not None:
            return self.output
        position = env.scene[self.ball_entity_name].data.root_link_pos_w - env.scene.env_origins
        if step != self.last_step:
            self.cursor = (self.cursor + 1) % self.length
            self.history[self.cursor] = position
        same_step = step == self.last_step
        previous_remaining = self.remaining.clone() if same_step else None
        self.history[:, self.fresh] = position[self.fresh]
        self.delay[self.fresh] = (
            self.cfg['delay_mean_s'] + self.cfg['delay_std_s'] * torch.randn_like(self.delay[self.fresh])
        ).clamp(0.0, self.cfg['delay_max_s'])
        self.remaining[self.fresh] = 0
        self.bias[self.fresh] = self.cfg['position_bias_std'] * torch.randn_like(self.bias[self.fresh])
        lag = self.delay / self.dt
        recent = lag.floor().long()
        older = (recent + 1).clamp_max(self.length - 1)
        a = self.history[(self.cursor - recent) % self.length, self.ids]
        b = self.history[(self.cursor - older) % self.length, self.ids]
        measured = a + (lag - recent)[:, None] * (b - a)
        measured += self.bias + self.cfg['position_noise_std'] * torch.randn_like(measured)
        burst = (self.remaining == 0) & (torch.rand_like(self.delay) < self.cfg['burst_start_probability'])
        self.remaining = torch.where(burst, torch.randint(2, 6, self.remaining.shape, device=env.device), self.remaining)
        dropped = (torch.rand_like(self.delay) < self.cfg['drop_probability']) | (self.remaining > 0)
        dropped &= ~self.fresh
        self.remaining = (self.remaining - 1).clamp_min(0)
        if same_step:
            self.remaining = torch.where(self.fresh, self.remaining, previous_remaining)
            dropped = torch.where(self.fresh, dropped, ~self.packet_valid)
        p, v = self.estimator.update(measured, ~dropped, self.fresh if same_step else None)
        self.fresh[:] = False
        self.last_step = step
        self.output = (p + env.scene.env_origins, v, ~dropped, self.estimator.age + self.delay)
        self.packet_valid = ~dropped
        env.extras.setdefault('log', {}).update({
            'Perception/drop_fraction': dropped.float().mean(),
            'Perception/packet_age_s': self.estimator.age.mean(),
            'Perception/estimated_speed': v.norm(dim=-1).mean(),
        })
        return self.output


def shared_measurement_ball_perception(params, env):
    cfg = params.get('measurement_perception')
    if cfg is None:
        return None
    if params.get('delay_mean_s') is not None or params.get('use_shared_perception', False):
        raise ValueError('Measurement packets cannot be combined with another perception path.')
    state = getattr(env, '_athlete_measurement_ball_perception', None)
    if state is None:
        state = MeasurementBallPerception(env, params['ball_entity_name'], cfg)
        env._athlete_measurement_ball_perception = state
    elif state.cfg != cfg or state.ball_entity_name != params['ball_entity_name']:
        raise ValueError('All Student ball inputs must share identical measurement settings.')
    return state
