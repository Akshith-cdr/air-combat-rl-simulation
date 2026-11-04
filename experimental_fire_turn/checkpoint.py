"""Composed fire-head bundles: immutable source weights plus a head delta."""

import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch

from turning_experiment import ExperimentConfig, load_pair, read_bundle, sha256
from .common import source_hashes, write_json
from .losses import ALLOWED, freeze_policy


def frozen_state_hash(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        if name in ALLOWED: continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()+str(array.shape).encode()+str(array.dtype).encode()+array.tobytes())
    return digest.hexdigest()


def normalization_hash(norm):
    digest = hashlib.sha256()
    for value in (norm.obs_rms.mean, norm.obs_rms.var, norm.obs_rms.count):
        array = np.asarray(value)
        digest.update(str(array.shape).encode()+str(array.dtype).encode()+array.tobytes())
    digest.update(repr((norm.norm_obs, norm.epsilon, norm.clip_obs, norm.norm_reward)).encode())
    return digest.hexdigest()


def load_source(source):
    source = Path(source)
    metadata = read_bundle(source)
    config = ExperimentConfig(**metadata['config'])
    model, norm = load_pair(source/'candidate.zip', source/'candidate_vecnormalize.pkl', config, device='cpu')
    freeze_policy(model.policy)
    norm.training = False
    return model, norm, metadata


def save_candidate(output, source, policy, norm, supervised_updates=0, optimizer_state=None, dataset_hash=None):
    output, source = Path(output), Path(source)
    if output.exists() or output.resolve().is_relative_to(source.resolve()):
        raise FileExistsError('Cannot overwrite/nest inside source bundle')
    if type(supervised_updates) is not int or supervised_updates < 0:
        raise ValueError('Explicit nonnegative supervised update count required')
    if supervised_updates and (optimizer_state is None or not dataset_hash):
        raise ValueError('Updated candidates require head optimizer state and dataset lineage')
    if optimizer_state is not None:
        ids = [p for group in optimizer_state['param_groups'] for p in group['params']]
        if len(ids) != 2 or len(set(ids)) != 2 or not set(optimizer_state['state']).issubset(ids):
            raise ValueError('Optimizer must contain only the two fire-head parameters')
    freeze_map = {name: parameter.requires_grad for name, parameter in policy.named_parameters()}
    if {name for name, flag in freeze_map.items() if flag} != set(ALLOWED) or norm.training:
        raise ValueError('Policy/normalization freeze contract violated')
    reference, reference_norm, source_metadata = load_source(source)
    try:
        frozen_hash = frozen_state_hash(reference.policy)
        norm_hash = normalization_hash(reference_norm)
        if frozen_state_hash(policy) != frozen_hash or normalization_hash(norm) != norm_hash:
            raise ValueError('Flight/critic/buffer or normalization state differs from source')
        if supervised_updates == 0 and any(not torch.equal(value.detach().cpu(), reference.policy.fire_net.state_dict()[key].detach().cpu())
                                          for key, value in policy.fire_net.state_dict().items()):
            raise ValueError('A zero-update candidate must retain the original fire head')
    finally:
        reference_norm.close()
    before = {name: sha256(source/name) for name in ('candidate.zip', 'candidate_vecnormalize.pkl', 'bundle.json')}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent/('.fire-head-'+uuid.uuid4().hex)
    stage.mkdir()
    shutil.copyfile(source/'candidate.zip', stage/'source_policy.zip')
    shutil.copyfile(source/'candidate_vecnormalize.pkl', stage/'candidate_vecnormalize.pkl')
    torch.save({key: value.detach().cpu().clone() for key, value in policy.fire_net.state_dict().items()}, stage/'fire_head.pt')
    files = ['source_policy.zip', 'candidate_vecnormalize.pkl', 'fire_head.pt']
    if optimizer_state is not None:
        torch.save(optimizer_state, stage/'head_optimizer.pt')
        files.append('head_optimizer.pt')
    metadata = {'schema': 'fire-head-candidate-v1', 'source_bundle': str(source.resolve()),
        'source_bundle_sha256': before['bundle.json'], 'source_pair_hashes': source_metadata['hashes'],
        'source_config': json.loads(json.dumps(source_metadata['config'])), 'source_environment_timesteps': reference.num_timesteps,
        'environment_steps_added': 0, 'supervised_updates': supervised_updates, 'dataset_hash': dataset_hash,
        'trainable_parameters': list(ALLOWED), 'freeze_map': freeze_map,
        'frozen_state_sha256': frozen_hash, 'normalization_sha256': norm_hash,
        'deterministic_fire_threshold': .5, 'source_hashes': source_hashes(),
        'optimizer': 'Embedded source PPO optimizer discarded on load; head state separate, if present',
        'hashes': {name: sha256(stage/name) for name in files}}
    write_json(stage/'candidate.json', metadata)
    if before != {name: sha256(source/name) for name in before}:
        raise RuntimeError('Source artifacts changed')
    stage.rename(output)
    return metadata


def load_candidate(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'candidate.json').read_text())
    if metadata.get('schema') != 'fire-head-candidate-v1' or metadata['source_hashes'] != source_hashes():
        raise ValueError('Candidate schema/source version mismatch')
    if (metadata['source_pair_hashes']['candidate.zip'] != metadata['hashes'].get('source_policy.zip') or
            metadata['source_pair_hashes']['candidate_vecnormalize.pkl'] != metadata['hashes'].get('candidate_vecnormalize.pkl')):
        raise ValueError('Copied source pair does not match lineage')
    for name in ('source_policy.zip', 'candidate_vecnormalize.pkl', 'fire_head.pt'):
        if sha256(directory/name) != metadata['hashes'].get(name):
            raise ValueError('Candidate pair/head integrity mismatch')
    for name, value in metadata['hashes'].items():
        if name not in ('source_policy.zip', 'candidate_vecnormalize.pkl', 'fire_head.pt', 'head_optimizer.pt'):
            raise ValueError('Unknown bundle member')
        if sha256(directory/name) != value: raise ValueError('Bundle integrity mismatch')
    config = ExperimentConfig(**metadata['source_config'])
    model, norm = load_pair(directory/'source_policy.zip', directory/'candidate_vecnormalize.pkl', config, device='cpu')
    try:
        if frozen_state_hash(model.policy) != metadata['frozen_state_sha256'] or normalization_hash(norm) != metadata['normalization_sha256']:
            raise ValueError('Frozen state/normalization fingerprint mismatch')
        head = torch.load(directory/'fire_head.pt', map_location='cpu', weights_only=True)
        if set(head) != {'weight', 'bias'} or any(not torch.isfinite(value).all() for value in head.values()):
            raise ValueError('Invalid fire head')
        model.policy.fire_net.load_state_dict(head, strict=True)
        freeze = freeze_policy(model.policy)
        if freeze != metadata['freeze_map'] or metadata['trainable_parameters'] != list(ALLOWED) or metadata['deterministic_fire_threshold'] != .5:
            raise ValueError('Freeze/threshold contract mismatch')
        norm.training = False
        return model, norm, metadata
    except BaseException:
        norm.close()
        raise
