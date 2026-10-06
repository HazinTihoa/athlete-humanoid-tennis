"""Extend only frozen positional embeddings; preserve actor, critic and Adam."""

import argparse
import copy
from pathlib import Path

import torch


def expand_history(source, steps):
    result = copy.deepcopy(source)
    actor = result['actor_state_dict']
    old = actor['student_position']
    if old.shape != (1, 5, 128) or steps not in (8, 12):
        raise ValueError('Expected m14-11 5-token checkpoint and 8/12 target steps.')
    # Same 100ms sample spacing: retain the newest five embeddings exactly,
    # prepend the oldest embedding for newly available, older history.
    actor['student_position'] = torch.cat((old[:, :1].expand(-1, steps - 5, -1), old), dim=1).clone()
    for state in result['optimizer_state_dict']['state'].values():
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            value = state.get(name)
            if isinstance(value, torch.Tensor) and value.shape == old.shape:
                state[name] = torch.cat((torch.zeros(1, steps - 5, 128, dtype=value.dtype), value), dim=1)
    result.setdefault('infos', {})['history_migration'] = dict(
        old_steps=5, new_steps=steps, stride=5, newest_embeddings='unchanged',
        older_embeddings='repeat oldest', encoders='frozen',
    )
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state = torch.load(args.source, map_location='cpu', weights_only=False)
    torch.save(expand_history(state, args.steps), args.output)
    print(f'Saved {args.output}: history={args.steps}; source iteration={state["iter"]}')
