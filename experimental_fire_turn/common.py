"""Immutable artifacts and strict split contracts."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from turning_experiment import sha256

SOURCE_FILES = ('hybrid_ppo.py', 'fighter_jet_env.py', 'projectile_motion.py',
                'updated_chase_jet_sim_v3.py', 'turning_experiment.py', 'turning_telemetry.py')


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def reserve(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def require_fresh_run_target(path):
    root = Path('experiments/turning_improvement').resolve()
    target = Path(path).resolve()
    if target.parent != root or target.exists():
        raise ValueError('Output must be a NEW direct child of experiments/turning_improvement')


def source_hashes():
    files = list(SOURCE_FILES) + [p.as_posix() for p in Path('experimental_fire_turn').glob('*.py')]
    return {name: sha256(name) for name in files}


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_manifests(directory):
    directory = Path(directory)
    index = json.loads((directory/'index.json').read_text())
    if index.get('schema') != 'fire-head-manifests-v1':
        raise ValueError('Not a fresh fire-head manifest set')
    result = {}
    ids = set()
    scenario_fingerprints = set()
    seeds = set()
    for split in ('training', 'validation', 'sealed_test'):
        path = directory/(split+'.json')
        if sha256(path) != index['hashes'][path.name]:
            raise ValueError('Manifest integrity mismatch')
        data = json.loads(path.read_text())
        if data['split'] != split or 'labels' in data:
            raise ValueError('Invalid split or exposed labels')
        if data['seed'] in seeds:
            raise ValueError('Split seeds overlap')
        seeds.add(data['seed'])
        for case in data['cases']:
            fingerprint = digest(case['scenario'])
            if case['episode_id'] in ids or fingerprint in scenario_fingerprints:
                raise ValueError('Duplicate scenario/episode across splits')
            ids.add(case['episode_id'])
            scenario_fingerprints.add(fingerprint)
        result[split] = data
    return index, result


def config_dict(config):
    return json.loads(json.dumps(asdict(config)))
