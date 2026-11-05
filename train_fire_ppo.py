"""Stage 7: learn the fire decision with PPO under the 500-1000 m launch envelope.

Starts from the Stage 6 flight policy (experiments/flight_policy/
flight_v2_seed63011_run03/final). Stage 6 flew with a fixed fire classifier fitted
when missiles launched at ~100 m, so it never fires inside the current envelope.
Stage 7 replaces that classifier with the policy's own Bernoulli fire head and
trains it with PPO:

* Launch rule (weapon_config.py, enforced by the environment and simulator):
  a fire request launches the single missile only if 500 m <= range <= 1000 m,
  cos(ATA) > 0.95, the missile is still mounted and the target is alive.
* Rewards: +5 for a launch, +100 for a hit, -1 for a rejected request while the
  missile is still mounted (wrong range or alignment). Requests made after the
  missile has left are ignored (no penalty): the policy cannot observe whether
  its missile is gone, so penalising them would teach it the wrong thing.
* An episode ends on a hit, a chase-boundary exit, the time limit, or when the
  missile is wasted (launched and expired without a hit).
* By default the flight network is frozen (``train_flight: false``): only the
  fire head and the value function learn, so the aircraft flies exactly as in
  Stage 6. The fire head is re-initialised (p(fire) = initial_fire_probability
  everywhere) so the decision is learned from scratch by RL.

The Stage 1-6 historical launch rules (training_rules.py) are NOT applied here.

Usage:
    python train_fire_ppo.py --run-id stage7_fire
    python train_fire_ppo.py --resume experiments/fire_policy/stage7_fire/checkpoints/step_000000524288
"""

import argparse
from collections import deque
import csv
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import stable_baselines3
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import weapon_config
from experimental_flight_ppo_v1 import TrainingFlightEnv
from fighter_jet_env import FIRE_COS_ATA_GATE, INVALID_FIRE_PENALTY
from learned_fire_policy import LearnedFirePolicy
from turning_experiment import ExperimentConfig

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'configs/fire_ppo_stage7.json'
DEFAULT_RUN_ROOT = ROOT / 'experiments/fire_policy'
SCHEMA = 'stage7-learned-fire-ppo-v1'
BUNDLE_SCHEMA = 'stage7-fire-ppo-bundle-v1'
MODEL_OPTIONS = ('n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma', 'gae_lambda',
                 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm', 'target_kl', 'device')
EPISODE_COLUMNS = ('timesteps', 'episode', 'family', 'terminal_reason', 'hit', 'launches',
                   'launch_range_m', 'fire_requests', 'rejected_requests', 'penalized_rejections',
                   'length_steps', 'reward')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if cfg.get('schema') != SCHEMA:
        raise ValueError(f'Expected a {SCHEMA} configuration: {path}')
    for name in ('total_timesteps', 'n_steps', 'checkpoint_every', 'curriculum_start_steps', 'seed'):
        if type(cfg[name]) is not int or cfg[name] < 0:
            raise ValueError(f'{name} must be a non-negative integer')
    if cfg['total_timesteps'] % cfg['n_steps'] or cfg['checkpoint_every'] % cfg['n_steps']:
        raise ValueError('total_timesteps and checkpoint_every must be multiples of n_steps')
    if not 0 < cfg['initial_fire_probability'] < 1:
        raise ValueError('initial_fire_probability must be in (0, 1)')
    if type(cfg['train_flight']) is not bool:
        raise ValueError('train_flight must be true or false')
    return cfg


def check_launch_envelope():
    """Stage 7 trains under the current envelope, never the Stage 1-6 rules."""
    if (weapon_config.MIN_MISSILE_LAUNCH_DISTANCE_M != 500.0
            or weapon_config.MAX_MISSILE_LAUNCH_DISTANCE_M != 1000.0
            or weapon_config.DEFAULT_MISSILE_LAUNCH_CEILING_M != 1000.0):
        raise RuntimeError('Stage 7 requires the 500-1000 m launch envelope in weapon_config.py')


def environment_config(cfg):
    return ExperimentConfig(
        variant='B', seed=cfg['seed'], total_timesteps=cfg['total_timesteps'], gamma=cfg['gamma'],
        bearing_progress_weight=cfg['bearing_progress_weight'], checkpoint_every=cfg['checkpoint_every'],
        max_episode_steps=cfg['max_episode_steps'],
        launch_range_m=weapon_config.MAX_MISSILE_LAUNCH_DISTANCE_M)


class FireTrainingEnv(TrainingFlightEnv):
    """Stage 6 training environment with the fire decision's rewards kept."""

    REFUND_INVALID_FIRE_PENALTY = False

    def reset_tracking(self):
        super().reset_tracking()
        self.first_launch_range_m = math.nan
        self._stage7_return = 0.0
        self.penalized_rejections = 0

    def step(self, action):
        missile_mounted = bool(self._missile_available())
        obs, reward, terminated, truncated, info = super().step(action)
        weapon, parts = info['weapon'], info['reward_breakdown']
        gate = self._observer.record
        # A request is penalised whenever the launch geometry is wrong. It is free
        # only when the geometry was valid and the sole reason for rejection is
        # that the missile has already left (which the policy cannot observe).
        geometry_valid = (weapon_config.launch_distance_allowed(gate['range_m'])
                          and gate['range_m'] <= self.launch_range_gate
                          and gate['velocity_cosATA'] > FIRE_COS_ATA_GATE)
        free = weapon['invalid_fire_attempt'] and not missile_mounted and geometry_valid
        refund = -INVALID_FIRE_PENALTY if free else 0.0
        if refund:
            for key in ('weapon', 'base_total', 'total'):
                parts[key] += refund
        parts['request_after_launch_refund'] = refund
        self.penalized_rejections += int(weapon['invalid_fire_attempt'] and not free)
        # Fire-decision reward only. The flight is frozen, so the Stage 6 flight
        # shaping (closure, alignment, geometry, bearing; about +1 per step while
        # chasing) cannot be improved and would reward delaying the kill, since a
        # hit ends the episode and the shaping stream with it.
        reward = parts['weapon'] + parts['time'] + parts['terminal']
        parts['stage7_reward'] = reward
        self._stage7_return += reward
        if gate['launched'] and math.isnan(self.first_launch_range_m):
            self.first_launch_range_m = float(gate['range_m'])
        # The single missile is gone without a hit: the engagement has failed.
        if (not terminated and not self.sim.target_destroyed
                and not self._missile_available() and not self.sim.active_missiles):
            terminated, truncated = True, False
            info['terminal_reason'] = 'missile_wasted'
        if terminated or truncated:
            info['episode'] = dict(r=self._stage7_return, l=self._episode_step, length=self._episode_step,
                                   terminated=terminated, truncated=truncated)
            info['first_launch_range_m'] = self.first_launch_range_m
            info['penalized_rejections'] = self.penalized_rejections
        return obs, reward, terminated, truncated, info


def make_environment(cfg, seed, completed_steps):
    source = ROOT / cfg['source']
    curriculum_steps = cfg['curriculum_start_steps'] + completed_steps
    vector = DummyVecEnv([lambda: FireTrainingEnv(environment_config(cfg), seed, curriculum_steps)])
    try:
        norm = VecNormalize.load(str(source / 'candidate_vecnormalize.pkl'), vector)
        norm.action_space = vector.action_space
        # Observation statistics stay frozen at the Stage 5/6 values.
        norm.training = False
        norm.norm_reward = False
        if norm.obs_rms.mean.shape != (19,) or not norm.norm_obs:
            raise ValueError('Expected 19-dimensional observation normalization')
        norm.seed(seed)
        return norm
    except BaseException:
        vector.close()
        raise


def model_options(cfg):
    return {name: cfg[name] for name in MODEL_OPTIONS}


def initial_model(cfg, norm, seed):
    """Stage 6 weights with a re-initialised fire head and a fresh optimizer."""
    source = ROOT / cfg['source']
    model = PPO(LearnedFirePolicy, norm, seed=seed, **model_options(cfg),
                policy_kwargs=dict(train_flight=cfg['train_flight'],
                                   optimizer_kwargs={'eps': cfg['adam_epsilon']}))
    # Read tensors only: the Stage 6 fixed fire classifier is not instantiated.
    _, parameters, _ = load_from_zip_file(str(source / 'candidate.zip'), device='cpu')
    result = model.policy.load_state_dict(parameters['policy'], strict=False)
    # Stage 6 has no fire network; everything else must load exactly.
    if result.unexpected_keys or any(not key.startswith('fire_mlp.') for key in result.missing_keys):
        raise ValueError(f'Stage 6 weights do not match the Stage 7 policy: {result}')
    p0 = cfg['initial_fire_probability']
    with th.no_grad():
        th.nn.init.orthogonal_(model.policy.fire_net.weight, gain=0.01)
        model.policy.fire_net.bias.fill_(math.log(p0 / (1 - p0)))
    # Fresh optimizer over the trainable tensors (state loaded in place above).
    model.policy.optimizer = model.policy.optimizer_class(
        [p for p in model.policy.parameters() if p.requires_grad],
        lr=model.policy.optimizer.param_groups[0]['lr'], **model.policy.optimizer_kwargs)
    return model, parameters['policy']


def check_flight_preserved(model, source_state):
    """Frozen flight tensors must equal Stage 6 bit-for-bit."""
    if model.policy.train_flight:
        return
    current = model.policy.state_dict()
    for name in model.policy.frozen_parameter_names():
        if not th.equal(current[name], source_state[name]):
            raise RuntimeError(f'Frozen Stage 6 flight tensor changed: {name}')


def save_bundle(directory, model, cfg, run, kind):
    directory = Path(directory)
    source = ROOT / cfg['source']
    temporary = directory.with_name('.' + directory.name + '.tmp')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save(temporary / 'candidate.zip')
    shutil.copyfile(source / 'candidate_vecnormalize.pkl', temporary / 'candidate_vecnormalize.pkl')
    write_json(temporary / 'bundle.json', dict(
        schema=BUNDLE_SCHEMA, kind=kind, steps=int(model.num_timesteps),
        hashes={name: sha256(temporary / name) for name in ('candidate.zip', 'candidate_vecnormalize.pkl')},
        config_sha256=sha256(run / 'config.json'), source=cfg['source'],
        source_hashes={name: sha256(source / name) for name in ('candidate.zip', 'candidate_vecnormalize.pkl')},
        train_flight=cfg['train_flight'], frozen_parameters=model.policy.frozen_parameter_names(),
        fire_authority='PPO Bernoulli fire head (deterministic evaluation: fire when p >= 0.5)',
        launch_envelope_m=[weapon_config.MIN_MISSILE_LAUNCH_DISTANCE_M,
                           weapon_config.MAX_MISSILE_LAUNCH_DISTANCE_M]))
    if directory.exists():
        shutil.rmtree(directory)
    os.replace(temporary, directory)
    return directory


class Stage7Logs(BaseCallback):
    """Episode CSV, rolling engagement statistics and periodic checkpoints."""

    def __init__(self, run, cfg, source_state, budget, window=200):
        super().__init__()
        self.run, self.cfg, self.source_state, self.budget = run, cfg, source_state, budget
        self.recent = deque(maxlen=window)
        self.started = None
        self.start_steps = 0
        path = run / 'episodes.csv'
        exists = path.exists()
        self.file = path.open('a', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=EPISODE_COLUMNS)
        if not exists:
            self.writer.writeheader()
        self.episodes = 0
        if exists:
            with path.open(encoding='utf-8') as stream:
                self.episodes = max(sum(1 for _ in stream) - 1, 0)

    def _on_training_start(self):
        self.started = time.time()
        self.start_steps = self.num_timesteps

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if not done:
                continue
            metrics = info['flight_metrics']
            self.episodes += 1
            row = dict(timesteps=self.num_timesteps, episode=self.episodes,
                       family=info['flight_curriculum']['family'],
                       terminal_reason=info.get('terminal_reason') or 'terminated',
                       hit=int(metrics['missile_hits'] > 0), launches=metrics['launches'],
                       launch_range_m=info.get('first_launch_range_m', math.nan),
                       fire_requests=metrics['fire_requests'], rejected_requests=metrics['rejected_requests'],
                       penalized_rejections=info['penalized_rejections'],
                       length_steps=info['episode']['l'], reward=info['episode']['r'])
            self.writer.writerow(row)
            self.recent.append(row)
        if self.num_timesteps % self.cfg['checkpoint_every'] == 0:
            self.file.flush()
            check_flight_preserved(self.model, self.source_state)
            save_bundle(self.run / 'checkpoints' / f'step_{self.num_timesteps:012d}',
                        self.model, self.cfg, self.run, 'periodic')
        return True

    def summary(self):
        rows = list(self.recent)
        if not rows:
            return {}
        launched = [r for r in rows if r['launches']]
        return dict(
            episodes=len(rows), hit_rate=float(np.mean([r['hit'] for r in rows])),
            launch_rate=float(np.mean([bool(r['launches']) for r in rows])),
            hit_rate_given_launch=float(np.mean([r['hit'] for r in launched])) if launched else math.nan,
            mean_launch_range_m=float(np.mean([r['launch_range_m'] for r in launched])) if launched else math.nan,
            mean_penalized_rejections=float(np.mean([r['penalized_rejections'] for r in rows])),
            mean_episode_steps=float(np.mean([r['length_steps'] for r in rows])))

    def _on_rollout_end(self):
        self.file.flush()
        for key, value in self.summary().items():
            self.logger.record(f'stage7/{key}', value)
        done = self.num_timesteps - self.start_steps
        if done > 0:
            rate = done / max(time.time() - self.started, 1e-9)
            remaining = self.budget - self.num_timesteps
            self.logger.record('stage7/eta_minutes', remaining / rate / 60)

    def close(self):
        self.file.close()


def versions():
    return dict(python=platform.python_version(), stable_baselines3=stable_baselines3.__version__,
                torch=th.__version__, numpy=np.__version__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-id', help='New run directory name under --run-root')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument('--resume', type=Path, help='Checkpoint directory (…/checkpoints/step_N) to continue')
    parser.add_argument('--total-timesteps', type=int, help='Override the budget (e.g. 4096 for a smoke test)')
    parser.add_argument('--checkpoint-every', type=int, help='Override the checkpoint interval')
    args = parser.parse_args(argv)
    if bool(args.run_id) == bool(args.resume):
        parser.error('Give exactly one of --run-id (new run) or --resume (continue a run)')
    check_launch_envelope()

    if args.resume:
        checkpoint = args.resume.resolve()
        run = checkpoint.parents[1]
        cfg = load_config(run / 'config.json')
        bundle = json.loads((checkpoint / 'bundle.json').read_text(encoding='utf-8'))
        if bundle.get('schema') != BUNDLE_SCHEMA or bundle['config_sha256'] != sha256(run / 'config.json'):
            raise ValueError('Checkpoint does not belong to this Stage 7 run')
        for name, expected in bundle['hashes'].items():
            if sha256(checkpoint / name) != expected:
                raise ValueError(f'Checkpoint file hash mismatch: {name}')
        completed = int(bundle['steps'])
    else:
        if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
            parser.error('run-id may use letters, digits, underscores and hyphens only')
        run = args.run_root.resolve() / args.run_id
        if run.exists():
            parser.error(f'Run directory already exists: {run} (use --resume to continue it)')
        cfg = load_config(args.config)
        run.mkdir(parents=True)
        shutil.copyfile(args.config, run / 'config.json')
        completed = 0
    if args.total_timesteps is not None:
        cfg['total_timesteps'] = args.total_timesteps
    if args.checkpoint_every is not None:
        cfg['checkpoint_every'] = args.checkpoint_every
    budget = cfg['total_timesteps']
    if budget % cfg['n_steps'] or cfg['checkpoint_every'] % cfg['n_steps'] or not completed < budget:
        raise ValueError('Budget/checkpoint interval must be multiples of n_steps, beyond the completed steps')
    source = ROOT / cfg['source']
    source_state = load_from_zip_file(str(source / 'candidate.zip'), device='cpu')[1]['policy']

    segment = completed // cfg['n_steps']
    seed = cfg['seed'] + segment
    write_json(run / ('run.json' if not completed else f'resume_{completed:012d}.json'), dict(
        schema=SCHEMA, seed=seed, started_from_steps=completed, budget=budget, versions=versions(),
        source=cfg['source'], source_hashes={n: sha256(source / n) for n in
                                             ('candidate.zip', 'candidate_vecnormalize.pkl')},
        launch_envelope_m=[weapon_config.MIN_MISSILE_LAUNCH_DISTANCE_M,
                           weapon_config.MAX_MISSILE_LAUNCH_DISTANCE_M],
        command=sys.argv))
    print(f'[stage 7] run: {run}')
    print(f'[stage 7] launch envelope: {weapon_config.MIN_MISSILE_LAUNCH_DISTANCE_M:g} m <= range <= '
          f'{weapon_config.MAX_MISSILE_LAUNCH_DISTANCE_M:g} m, cos(ATA) > 0.95')
    print(f'[stage 7] fire decision: learned PPO Bernoulli head; flight '
          f'{"trainable" if cfg["train_flight"] else "frozen at Stage 6"}')

    norm = callback = None
    try:
        norm = make_environment(cfg, seed, completed)
        if args.resume:
            model = PPO.load(str(checkpoint / 'candidate.zip'), env=norm, device='cpu')
            if not isinstance(model.policy, LearnedFirePolicy):
                raise TypeError('Checkpoint is not a Stage 7 policy')
            model.set_random_seed(seed)
        else:
            model, _ = initial_model(cfg, norm, seed)
            save_bundle(run / 'initial', model, cfg, run, 'initial')
        check_flight_preserved(model, source_state)
        model.set_logger(configure(str(run / 'logs' / f'segment_{segment:06d}'), ['stdout', 'csv']))
        callback = Stage7Logs(run, cfg, source_state, budget)
        model.learn(total_timesteps=budget - completed, reset_num_timesteps=False, callback=callback)
        check_flight_preserved(model, source_state)
        final = save_bundle(run / 'final', model, cfg, run, 'final')
        write_json(run / 'result.json', dict(status='complete', training_steps=int(model.num_timesteps),
                   final_bundle=str(final.relative_to(run)), last_episodes=callback.summary()))
        print(f'[stage 7] complete: {final / "candidate.zip"}')
        print(f'[stage 7] last {len(callback.recent)} training episodes: {callback.summary()}')
    except BaseException as exc:
        write_json(run / 'error.json', dict(status='incomplete', error=repr(exc),
                   steps=int(getattr(locals().get('model'), 'num_timesteps', completed))))
        raise
    finally:
        if callback is not None:
            callback.close()
        if norm is not None:
            norm.close()


if __name__ == '__main__':
    main()
