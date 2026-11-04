"""Paired checkpoint bundles; saving is not a model update."""

from dataclasses import asdict
import json
import platform
from pathlib import Path
import uuid

import numpy as np
import stable_baselines3
import torch

from turning_experiment import sha256, verify_baseline

SOURCES=('fighter_jet_env.py','updated_chase_jet_sim_v3.py','hybrid_ppo.py',
         'projectile_motion.py','turning_experiment.py','turning_checkpoints.py','train_turning_ppo.py')


def source_hashes():
    return {name:sha256(name) for name in SOURCES}


def save_bundle(destination,model,norm,config,source_timesteps,parent=None,kind='periodic'):
    destination=Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    verify_baseline()
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.parent/('.bundle-'+uuid.uuid4().hex)
    temporary.mkdir()
    # Only the final rename publishes a complete pair. On interruption, ignore
    # hidden staging directories and use a previously completed bundle.
    model.save(str(temporary/'candidate.zip'))
    norm.save(str(temporary/'candidate_vecnormalize.pkl'))
    metadata={'schema':1,'kind':kind,'config':asdict(config),
              'source_timesteps':source_timesteps,'num_timesteps':int(model.num_timesteps),
              'actual_additional_timesteps':int(model.num_timesteps-source_timesteps),
              'additional_timesteps_since_baseline':int(model.num_timesteps-100352),
              'optimizer_state_entries':len(model.policy.optimizer.state),
              'parent_bundle':str(parent) if parent else None,'baseline_hashes':verify_baseline(),
              'source_hashes':source_hashes(),'python':platform.python_version(),
              'sb3':stable_baselines3.__version__,'torch':torch.__version__,'numpy':np.__version__,
              'simulation_dt_seconds':norm.venv.envs[0].unwrapped.simulation_dt,
              'guidance_config':(asdict(norm.venv.envs[0].unwrapped.sim.guidance_config)
                                 if norm.venv.envs[0].unwrapped.sim.guidance_config else None),
              'continuation':'optimizer-restored fine-tuning; fresh rollout/environment, not exact replay',
              'hashes':{name:sha256(temporary/name) for name in ('candidate.zip','candidate_vecnormalize.pkl')}}
    (temporary/'bundle.json').write_text(json.dumps(metadata,indent=2)+'\n')
    temporary.rename(destination)
    return metadata


def validate_resume(metadata,config):
    actual=json.loads(json.dumps(asdict(config)))
    previous=dict(metadata['config'])
    for name in ('total_timesteps','device','checkpoint_every'):
        actual.pop(name,None)
        previous.pop(name,None)
    if actual!=previous:
        raise ValueError('Resume configuration changes scenario/reward/PPO/normalization semantics')
    if metadata['source_hashes']!=source_hashes():
        raise ValueError('Resume source version differs from checkpoint bundle')
    if metadata['sb3']!=stable_baselines3.__version__ or metadata['torch']!=torch.__version__:
        raise ValueError('Resume requires matching SB3 and Torch versions')
    if metadata['python']!=platform.python_version() or metadata['numpy']!=np.__version__:
        raise ValueError('Resume requires matching Python and NumPy versions')
    if metadata['baseline_hashes']!=verify_baseline():
        raise ValueError('Resume baseline provenance mismatch')
