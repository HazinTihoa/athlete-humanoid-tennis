"""Episode-wise Gaussian latency shared by the two Student ball input paths."""

import math

import torch


class GaussianBallDelay:
    def __init__(self, env, ball_entity_name, mean_s, std_s, max_s, min_s=0.0):
        dt = float(env.step_dt)
        if not all(math.isfinite(x) for x in (mean_s, std_s, min_s, max_s, dt)):
            raise ValueError("Ball delay parameters must be finite.")
        if dt <= 0 or std_s < 0 or not 0 <= min_s <= mean_s <= max_s or max_s <= 0:
            raise ValueError("Invalid Gaussian ball delay range or timestep.")
        self.signature = (ball_entity_name, mean_s, std_s, max_s, dt, min_s)
        self.min_s = min_s
        self.mean_s, self.std_s, self.max_s, self.dt = mean_s, std_s, max_s, dt
        self.ball_entity_name = ball_entity_name
        self.length = math.ceil(max_s / dt) + 1
        self.history = torch.zeros(self.length, env.num_envs, 6, device=env.device)
        self.delay_s = torch.zeros(env.num_envs, device=env.device)
        self.initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.env_ids = torch.arange(env.num_envs, device=env.device)
        self.cursor = -1
        self.last_step = None
        self.output = None
        self.needs_seed = True

    def reset(self, env_ids=None):
        indices = slice(None) if env_ids is None else env_ids
        self.initialized[indices] = False
        self.output = None
        self.needs_seed = True

    def update(self, env):
        step = int(env.common_step_counter)
        if step == self.last_step and self.output is not None:
            return self.output
        ball = env.scene[self.ball_entity_name].data
        state = torch.cat((ball.root_link_pos_w, ball.root_link_lin_vel_w), dim=-1)
        if step != self.last_step:
            self.cursor = (self.cursor + 1) % self.length
            self.history[self.cursor] = state
            self.last_step = step

        # Seed only reset environments: no samples may leak across episodes.
        if self.needs_seed:
            fresh = ~self.initialized
            self.history[:, fresh] = state[fresh]
            self.delay_s[fresh] = (
                self.mean_s + self.std_s * torch.randn_like(self.delay_s[fresh])
            ).clamp(self.min_s, self.max_s)
            self.initialized[:] = True
            self.needs_seed = False

        lag = self.delay_s / self.dt
        recent = lag.floor().long()
        older = (recent + 1).clamp(max=self.length - 1)
        weight = (lag - recent)[:, None]
        a = self.history[(self.cursor - recent) % self.length, self.env_ids]
        b = self.history[(self.cursor - older) % self.length, self.env_ids]
        measured = a + weight * (b - a)
        self.output = (measured[:, :3], measured[:, 3:], self.initialized, self.delay_s)
        return self.output


def shared_gaussian_ball_delay(params, env):
    if params.get("measurement_perception") is not None:
        from .ball_measurement import shared_measurement_ball_perception
        return shared_measurement_ball_perception(params, env)
    mean = params.get("delay_mean_s")
    if mean is None:
        return None
    if params.get("use_shared_perception", False):
        raise ValueError("Gaussian ball latency cannot be combined with legacy perception.")
    signature = (
        params["ball_entity_name"], float(mean),
        float(params["delay_std_s"]), float(params["delay_max_s"]), float(env.step_dt),
        float(params.get("delay_min_s", 0.0)),
    )
    state = getattr(env, "_athlete_gaussian_ball_delay", None)
    if state is None:
        state = GaussianBallDelay(env, *signature[:4], min_s=signature[-1])
        env._athlete_gaussian_ball_delay = state
    elif state.signature != signature:
        raise ValueError("Student ball inputs must share the same Gaussian delay.")
    return state
