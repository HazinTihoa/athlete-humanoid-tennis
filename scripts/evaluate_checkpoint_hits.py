"""Evaluate actual net returns, or explicitly request legacy timing counters."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = 'Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill01-FrozenIntent-Epoch3-ActionRate020-SweetSpeed50-Scale03-RacketGround10-Landing5-PelvisTorsoTilt50-BallDR150-Unitree-G1'


class NetReturnTracker:
    def __init__(self, num_envs, device):
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.hit = torch.zeros_like(self.active)
        self.landed = torch.zeros_like(self.active)
        self.success = torch.zeros_like(self.active)
        self.forehand = torch.zeros_like(self.active)
        self.previous = torch.zeros((num_envs, 3), device=device)
        self.counts = {s: dict(attempts=0, hits=0, contacts=0) for s in ('forehand', 'backhand')}

    def finish(self, mask):
        for name, side in [('forehand', self.forehand), ('backhand', ~self.forehand)]:
            selected = mask & self.active & side
            self.counts[name]['attempts'] += int(selected.sum())
            self.counts[name]['hits'] += int((selected & self.success).sum())
            self.counts[name]['contacts'] += int((selected & self.hit).sum())
        self.active[mask] = False

    def start(self, mask, forehand, position):
        self.active[mask] = True
        self.forehand[mask] = forehand[mask]
        self.hit[mask] = False
        self.landed[mask] = False
        self.success[mask] = False
        self.previous[mask] = position[mask]

    def observe(self, position, contact, grounded):
        self.hit |= contact & self.active
        dx = position[:, 0] - self.previous[:, 0]
        fraction = ((3.5 - self.previous[:, 0]) / dx.clamp(min=1e-9)).clamp(0, 1)
        crossing = self.previous + fraction[:, None] * (position - self.previous)
        passed = ((self.previous[:, 0] < 3.5) & (position[:, 0] >= 3.5)
                  & (crossing[:, 2] > .914 + .0335)
                  & (crossing[:, 1].abs() < 2.5 - .0335))
        self.success |= self.active & self.hit & ~self.landed & passed
        self.landed |= self.active & self.hit & grounded
        self.previous.copy_(position)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--attempts', type=int, default=1000)
    p.add_argument('--stochastic', action='store_true')
    p.add_argument('--seed', type=int, default=20260908)
    p.add_argument('--metric', choices=['net', 'timing'], default='net')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from athlete.scripts import play
    from mjlab.tasks.registry import load_env_cfg

    # Use the train variant, not Play's simplified/no-randomization variant.
    original_loader = play.load_env_cfg
    play.load_env_cfg = lambda task_id, **kw: load_env_cfg(task_id, play=False)
    original_installer = play.install_tennis_ball_controller
    controllers = []

    def install(*a, **kw):
        kw['verbose_samples'] = False
        controller = original_installer(*a, **kw)
        controllers.append(controller)
        return controller

    play.install_tennis_ball_controller = install

    def hook(cfg):
        cfg.scene.terrain.num_envs = args.num_envs
        cfg.commands['motion'].incoming_ball_failure_trajectory_probability = 0.0
        cfg.commands['motion'].viz.mode = 'ghost'
        if args.metric == 'net':
            from mjlab.sensor import ContactMatch, ContactSensorCfg
            cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
                name='eval_ball_racket',
                primary=ContactMatch(mode='geom', pattern='tennis_ball_geom', entity='tennis_ball'),
                secondary=ContactMatch(mode='geom', pattern='racket_ball_collision', entity='robot'),
                fields=('found',), reduce='maxforce', num_slots=1))
        robot = cfg.scene.entities['robot']
        original_spec = robot.spec_fn

        def training_spec():
            spec = original_spec()
            geom = spec.geom('racket_ball_collision')
            # Restore the collider with which model_38500 was trained. Only this
            # evaluation instance is changed; the current smaller collider stays intact.
            geom.pos = (0.38, 0.01, 0.27)
            geom.size = (0.14, 0.20, 0.012)
            return spec

        robot.spec_fn = training_spec

    try:
        env, policy = play.build_play_session(TASK, play.PlayConfig(
            checkpoint_file=str(args.checkpoint), num_envs=args.num_envs,
            motion_config=ROOT / 'athlete/src/athlete/motion_sets/motion_train_configs/rollout_1000epis_relabelled_first200_phase.toml',
            seed=args.seed, device='cuda:0', physical_ball=True,
            no_terminations=False, fast_play=False, phase_plot=False), env_cfg_hook=hook)
    finally:
        play.load_env_cfg = original_loader
        play.install_tennis_ball_controller = original_installer

    m = env.unwrapped.command_manager.get_term('motion')
    counts = {side: {'attempts': 0, 'hits': 0} for side in ('forehand', 'backhand')}
    original_record = m._record_stroke_outcomes

    def record(ids, *, force=False):
        before = {s: [m.metrics[f'{s}_{k}'].sum().clone() for k in ('strike_attempts', 'hits')]
                  for s in counts}
        original_record(ids, force=force)
        for s in counts:
            counts[s]['attempts'] += int((m.metrics[f'{s}_strike_attempts'].sum()-before[s][0]).item())
            counts[s]['hits'] += int((m.metrics[f'{s}_hits'].sum()-before[s][1]).item())

    if args.metric == 'timing':
        m._record_stroke_outcomes = record
    else:
        raw = env.unwrapped
        tracker = NetReturnTracker(args.num_envs, raw.device)
        counts = tracker.counts
        controller = controllers[0]
        original_before = controller.before_step

        def before():
            restarted = (controller._force_restart
                | (m.time_remaining > controller._previous_time_remaining + .5 * raw.step_dt)
                | (m.which_motion != controller._previous_motion_ids)
                | (m.motion_chain_count != controller._previous_motion_chain_count))
            tracker.finish(restarted)
            original_before()
            selected = restarted & controller._launched & m.trajectory_match_valid
            ids = m.which_motion
            known = m._motion_is_forehand_t[ids] | m._motion_is_backhand_t[ids]
            if torch.any(selected & ~known):
                raise RuntimeError('Matched motion has no hand label')
            tracker.start(selected, m._motion_is_forehand_t[ids],
                          controller._launch_pose[:, :3] - raw.scene.env_origins)

        controller.before_step = before
        original_update = raw.scene.update

        def update(*a, **kw):
            result = original_update(*a, **kw)
            position = raw.scene['tennis_ball'].data.root_link_pos_w - raw.scene.env_origins
            contact = raw.scene['eval_ball_racket'].data.found.reshape(args.num_envs, -1).gt(0).any(-1)
            grounded = (position[:, 2] <= .0335 + .002) & (position[:, 2] <= tracker.previous[:, 2])
            tracker.observe(position, contact, grounded)
            return result

        raw.scene.update = update
    started = time.monotonic()
    name = ('net_' if args.metric == 'net' else '') + ('stochastic' if args.stochastic else 'deterministic')
    obs = env.get_observations()
    total_resets = 0
    try:
        with torch.inference_mode():
            for step in range(20000):
                if args.stochastic:
                    student = policy.student(obs, stochastic_output=True)
                    teacher = policy.teacher(obs)
                    action = torch.cat([student, teacher[..., student.shape[-1]:]], dim=-1)
                else:
                    action = policy(obs)
                obs, reward, dones, extras = env.step(action)
                total_resets += int(dones.sum())
                if step % 100 == 0:
                    print('EFFECTIVE_HITS', name, step, counts, flush=True)
                if sum(c['attempts'] for c in counts.values()) >= args.attempts:
                    break
            else:
                raise RuntimeError('Not enough completed valid attempts')
        result = dict(checkpoint=str(args.checkpoint.resolve()),
            sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            mode=name, task=TASK, num_envs=args.num_envs, control_steps=step+1,
            counts=counts, resets=total_resets, elapsed_s=time.monotonic()-started,
            failure_probability=0, physics='training DR150', collider_size=[.14,.20,.012],
            collider_pos=[.38,.01,.27], early_terminations=True,
            metric=args.metric,
            hits_field=('successful_net_returns' if args.metric == 'net' else 'legacy_timing_hits'),
            accounting=('Physical racket contact then actual forward net crossing before first landing; no time gate; denominator=completed launched matched balls'
                        if args.metric == 'net' else 'Sum actual increments of task _record_stroke_outcomes; no forced final settlement'))
        for c in counts.values():
            c['rate'] = c['hits']/c['attempts'] if c['attempts'] else None
        (args.out / f'{name}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
