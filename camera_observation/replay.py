"""Reproduce every held-out camera result from saved original-flight traces."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .live import load_camera_run
from .pipeline import PROTECTED, evaluate_one, write_json
from .simulation import ROOT, digest


def replay(directory, output):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError('Replay evidence requires a new output file')
    torch.set_num_threads(1)
    policy, config, manifest = load_camera_run(directory)
    original = json.loads((directory/'episodes.json').read_text())
    inventory = json.loads((directory/'trace_hashes.json').read_text())
    if manifest['protected_sha256'] != {name: digest(ROOT/name) for name in PROTECTED}:
        raise ValueError('Protected simulator/asset hash mismatch')
    checked = 0
    for index, seed in enumerate(manifest['splits']['heldout']):
        path = directory/'traces/heldout'/f'{seed}.npz'
        if digest(path) != inventory[path.relative_to(directory).as_posix()]:
            raise ValueError('Original-flight trace hash mismatch')
        with np.load(path, allow_pickle=False) as saved:
            positions = saved['positions']
            trace = dict(seed=seed, frames=saved['frames'],
                         termination_reason=original['learned'][index]['termination_reason'],
                         flight_path_m=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
                         minimum_sampled_altitude_m=float(positions[:, 2].min()))
        for mode in ('learned', 'periodic', 'wait'):
            expected = original[mode][index]
            if expected['seed'] != seed or evaluate_one(trace, policy, config, mode) != expected:
                raise AssertionError(f'Camera replay differs for {mode}, seed {seed}')
            checked += 1
    result = dict(identical=True, heldout_seeds=len(manifest['splits']['heldout']),
                  policy_episode_replays=checked, protected_simulation_unchanged=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    replay(args.run, args.output)
