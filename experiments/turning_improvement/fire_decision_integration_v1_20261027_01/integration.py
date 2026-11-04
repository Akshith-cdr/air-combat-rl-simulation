"""Opt-in fire-only composition. No PPO, environment or normalization imports."""

import hashlib
import importlib
import json
import math
from numbers import Real
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = 'experiments/turning_improvement/'
FIT = PREFIX + 'fire_decision_conjunction_v1_run01/'
SOURCE = PREFIX + 'A_seed62001_run01/final/'
IMPLEMENTATION = PREFIX + 'fire_decision_conjunction_v1_20261027_01/'
MANIFEST = PREFIX + 'fire_head_manifests_20261026_01/validation.json'
# Original artifacts.json remains the historical research inventory.
PINS_SHA256 = '9ee9ff5365acc456c02f2eecaf9b2f8c4abb178e0ebd4f4861de6312fdc6cd88'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    """Reject ambiguous duplicate JSON keys and nonfinite JSON literals."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('Nonfinite JSON literal: ' + value)

    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                      parse_constant=invalid)


def verified_inputs(root=ROOT, pins_path=HERE / 'runtime_artifacts_500m.json',
                    pins_sha256=PINS_SHA256):
    """Hash and consume the same bytes; never open any sealed material."""
    pins_raw = Path(pins_path).read_bytes()
    if digest(pins_raw) != pins_sha256:
        raise ValueError('Integration pin inventory hash mismatch')
    pins = decode(pins_raw)
    if set(pins) != {'schema', 'sha256'} or pins['schema'] != 'conjunction-integration-pins-v1':
        raise ValueError('Invalid pin inventory')
    hashes = pins['sha256']
    required = {FIT + n for n in ('qualified_model.json', 'result.json', 'integrity.json', 'plan.json')}
    required |= {SOURCE + n for n in ('candidate.zip', 'candidate_vecnormalize.pkl', 'bundle.json')}
    required |= {IMPLEMENTATION + n for n in ('model.py', 'metrics.py', 'run.py')}
    required.add(MANIFEST)
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise ValueError('Missing required artifact pins')
    root = Path(root).resolve()
    buffers = {}
    for name, expected in sorted(hashes.items()):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or 'sealed' in name.lower():
            raise ValueError('Unauthorized pinned path')
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('Invalid SHA-256 pin')
        raw = path.read_bytes()
        if digest(raw) != expected:
            raise ValueError('Pinned input hash mismatch: ' + name)
        buffers[name] = raw
    return hashes, buffers


def _finite_vector(values, size):
    if len(values) != size or any(
        isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
        for v in values
    ):
        raise ValueError('Expected a finite one-dimensional vector of length ' + str(size))


class FireDecisionAdapter:
    """Default identity; active mode replaces only index 4 of a five-action vector.

    Input is ONE already-VecNormalized 19-vector. This adapter never normalizes
    observations, calls the launch gate, or supplies any flight control.
    """

    def __init__(self):
        self._model = None
        self.mode = 'policy'
        self.input_hashes = {}

    @classmethod
    def create(cls, mode='policy', *, root=ROOT, pins_path=HERE / 'runtime_artifacts_500m.json',
               pins_sha256=PINS_SHA256):
        instance = cls()
        if mode == 'policy':
            return instance
        if mode != 'conjunction-v1':
            raise ValueError('Explicit policy or conjunction-v1 mode required')
        hashes, buffers = verified_inputs(root, pins_path, pins_sha256)
        if (Path(root) / FIT / 'error.json').exists():
            raise ValueError('Fit has an error marker')
        source_pair = {name: hashes[SOURCE + name] for name in
                       ('candidate.zip', 'candidate_vecnormalize.pkl')}
        result = decode(buffers[FIT + 'result.json'])
        artifact = decode(buffers[FIT + 'qualified_model.json'])
        integrity = decode(buffers[FIT + 'integrity.json'])
        if result.get('status') != 'fit_completed' or result.get('qualified') is not True:
            raise ValueError('Fit is not qualified')
        if result.get('model_parameters') != artifact.get('model'):
            raise ValueError('Artifact/result model mismatch')
        for name in ('result.json', 'qualified_model.json'):
            if integrity['output_sha256'][name] != hashes[FIT + name]:
                raise ValueError('Fit completion hash mismatch')
        # Use the existing source-pair contract on a private snapshot of the
        # verified buffers. Reopening the original files cannot alter parsing.
        module = importlib.import_module(
            'experiments.turning_improvement.fire_decision_conjunction_v1_20261027_01.run')
        with tempfile.TemporaryDirectory(prefix='verified-conjunction-') as directory:
            stage = Path(directory)
            for name in ('qualified_model.json', 'result.json', 'integrity.json'):
                with (stage / name).open('xb') as stream:
                    stream.write(buffers[FIT + name])
            model = module.load_qualified(stage / 'qualified_model.json',
                                          hashes[FIT + 'qualified_model.json'], source_pair)
        instance._model = model
        instance.mode = mode
        instance.input_hashes = dict(hashes)
        return instance

    def apply(self, normalized_observation, action):
        if self.mode == 'policy':
            return action
        _finite_vector(normalized_observation, 19)
        _finite_vector(action, 5)
        # Preserve ndarray dtype, sign bits and exact bytes of indices 0..3.
        output = action.copy() if hasattr(action, 'copy') else list(action)
        output[4] = float(self._model.request(normalized_observation))
        return output

    def fire_probability(self, normalized_observation):
        if self.mode != 'conjunction-v1':
            raise ValueError('No conjunction probability in default policy mode')
        _finite_vector(normalized_observation, 19)
        z = self._model.logit(normalized_observation)
        return 1. / (1. + math.exp(-z)) if z >= 0 else math.exp(z) / (1. + math.exp(z))
