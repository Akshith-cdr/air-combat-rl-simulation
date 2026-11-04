"""Finite, opt-in flight PPO. Importing this module never starts a simulator."""

import argparse
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid
import zipfile

import numpy as np
import stable_baselines3
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from experimental_flight_ppo_v1 import (
    BUDGET, FixedConjunctionFlightPolicy, TrainingFlightEnv, transfer_initialization,
)
from train_turning_ppo import ExperimentLogs
from turning_experiment import ExperimentConfig, sample_scenario, sha256, verify_baseline
from turning_telemetry import paired_bearing_metrics
from experimental_fire_turn.checkpoint import normalization_hash
from experiments.turning_improvement.fire_decision_integration_v1_20261027_01 import integration as fire
from training_rules import apply_training_launch_rules

ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / 'experiments/flight_policy'
CONFIG = ROOT / 'configs/flight_ppo_v1_seed63011.json'
CONFIG_SHA = '66893f974af82265615fd0a92cd759e9f587256ac924c65830d46eebb238d2a8'
CONFIG_V2 = ROOT / 'configs/flight_ppo_v2_seed63011.json'
CONFIG_V2_SHA = 'd7c19bbc3a80edbdec06c4f1f6cb57e4366a81791eaad5c2e952b5da7819df0e'
APPROVED_CONFIGS = {CONFIG_SHA: BUDGET, CONFIG_V2_SHA: 2_097_152}
DEFAULT_SOURCE = ROOT / fire.SOURCE
# Stage 5 bundle the flight policy is initialized from. Defaults to the supplied
# Stage 5 result; main() replaces it with --source (e.g. a Stage 5 run you trained).
SOURCE = DEFAULT_SOURCE
SOURCE_STEPS = 602112  # Timestep counter of SOURCE; main() reads it from the checkpoint.
CODE = ('experimental_flight_ppo_v1.py', 'train_flight_ppo.py', 'train_turning_ppo.py',
        'evaluate_turning_ppo.py', 'turning_experiment.py', 'turning_checkpoints.py',
        'turning_telemetry.py', 'fighter_jet_env.py', 'hybrid_ppo.py',
        'updated_chase_jet_sim_v3.py', 'projectile_motion.py', 'radar.py')


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def read_json(path, expected=None):
    raw = Path(path).read_bytes()
    if expected is not None and fire.digest(raw) != expected:
        raise ValueError('JSON input hash mismatch: ' + str(path))
    return fire.decode(raw)


def config(path=CONFIG):
    raw = Path(path).read_bytes()
    expected_budget = APPROVED_CONFIGS.get(fire.digest(raw))
    if expected_budget is None:
        raise ValueError('JSON input hash mismatch: ' + str(path))
    cfg = fire.decode(raw)
    if training_budget(cfg) != expected_budget:
        raise ValueError('Pinned configuration budget mismatch')
    return cfg


def training_budget(cfg):
    budget = cfg['total_timesteps']
    if type(budget) is not int or budget not in APPROVED_CONFIGS.values():
        raise ValueError('Unapproved finite training budget')
    return budget


def pins():
    original, _ = fire.verified_inputs()
    configurations = {}
    for path, expected in ((CONFIG, CONFIG_SHA), (CONFIG_V2, CONFIG_V2_SHA)):
        if sha256(path) != expected:
            raise ValueError('Protected configuration hash mismatch: ' + str(path))
        configurations[path.relative_to(ROOT).as_posix()] = expected
    result = {**original, **verify_baseline(), **{n: sha256(ROOT / n) for n in CODE},
              **configurations}
    if SOURCE.resolve() != DEFAULT_SOURCE.resolve():
        result.update({f'stage5_source/{n}': sha256(SOURCE / n)
                       for n in ('candidate.zip', 'candidate_vecnormalize.pkl')})
    return result


def source_statistics_sha256():
    return sha256(SOURCE / 'candidate_vecnormalize.pkl')


def checkpoint_timesteps(path):
    with zipfile.ZipFile(Path(path) / 'candidate.zip') as archive:
        return int(json.loads(archive.read('data'))['num_timesteps'])


def versions():
    return dict(python=platform.python_version(), sb3=stable_baselines3.__version__,
                torch=th.__version__, numpy=np.__version__)


def core_config(cfg):
    return ExperimentConfig(variant='B', seed=cfg['seed'], total_timesteps=training_budget(cfg),
        gamma=cfg['gamma'], bearing_progress_weight=cfg['bearing_progress_weight'],
        checkpoint_every=cfg['checkpoint_every'])


def prepare_protocol(cfg):
    from evaluate_turning_ppo import held_out_manifest
    protocol = {}
    fingerprints, seeds = set(), {cfg['seed']}
    for split in ('development', 'final'):
        spec = cfg[split]
        cases = []
        for i, scenario in enumerate(held_out_manifest(spec['pairs'], spec['geometry_seed'])):
            fingerprint = fire.digest(json.dumps(scenario, sort_keys=True, separators=(',', ':')).encode())
            env_seed, policy_seed = spec['environment_seed'] + i, spec['policy_seed'] + i
            if fingerprint in fingerprints or {env_seed, policy_seed} & seeds:
                raise ValueError('Development/final initial-state or seed overlap')
            fingerprints.add(fingerprint)
            seeds.update((env_seed, policy_seed))
            cases.append(dict(episode_id=f'{split}-{i:04d}', pair_id=i // 2, scenario=scenario,
                              scenario_sha256=fingerprint, environment_seed=env_seed, policy_seed=policy_seed))
        protocol[split] = cases
    return protocol


def reservations(protocol):
    cases = protocol['development'] + protocol['final']
    return ([c['scenario_sha256'] for c in cases],
            [c[k] for c in cases for k in ('environment_seed', 'policy_seed')])


def model_options(cfg):
    return {k: cfg[k] for k in ('n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma',
        'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm', 'target_kl', 'device')}


def remaining_steps(completed, cfg=None):
    budget = BUDGET if cfg is None else training_budget(cfg)
    rollout = 2048 if cfg is None else cfg['n_steps']
    if type(completed) is not int or not 0 <= completed < budget or completed % rollout:
        raise ValueError('Only complete rollouts inside the selected finite budget can resume')
    return budget - completed


def make_norm(cfg, protocol, seed, steps=0):
    reserved, seeds = reservations(protocol)
    vector = DummyVecEnv([lambda: TrainingFlightEnv(core_config(cfg), seed, steps, reserved, seeds)])
    try:
        norm = VecNormalize.load(str(SOURCE / 'candidate_vecnormalize.pkl'), vector)
        norm.action_space = vector.action_space
        norm.training = False
        norm.norm_reward = False
        if norm.obs_rms.mean.shape != (19,) or not norm.norm_obs:
            raise ValueError('Expected frozen Candidate A 19-dimensional normalization')
        norm.seed(seed)
        return norm
    except BaseException:
        vector.close()
        raise


def fire_tensor_hash(policy):
    return fire.digest(b''.join(p.detach().cpu().contiguous().numpy().tobytes()
                              for p in policy.fire_net.parameters()))


def assert_invariants(model, norm, state):
    if pins() != state['pins'] or versions() != state['versions']:
        raise ValueError('Source/code/version lineage drift')
    if norm.training or norm.norm_reward or normalization_hash(norm) != state['normalization_hash']:
        raise ValueError('Frozen normalization drift')
    if fire_tensor_hash(model.policy) != state['fire_tensor_hash']:
        raise ValueError('Frozen unused fire tensors changed')
    if not isinstance(model.policy, FixedConjunctionFlightPolicy):
        raise TypeError('Not the fixed-conjunction flight policy')
    cfg = state['config']
    if any(getattr(model, key) != cfg[key] for key in ('n_steps', 'batch_size', 'n_epochs', 'learning_rate',
            'gamma', 'gae_lambda', 'ent_coef', 'vf_coef', 'max_grad_norm', 'target_kl')):
        raise ValueError('PPO configuration drift')
    if model.clip_range(1.) != cfg['clip_range'] or not model.normalize_advantage:
        raise ValueError('PPO clipping/advantage configuration drift')
    if model.n_envs != 1 or model.device.type != 'cpu' or model.clip_range_vf is not None:
        raise ValueError('Unsupported environment/device/value-clipping configuration')
    if any(group['eps'] != cfg['adam_epsilon'] or group['lr'] != cfg['learning_rate']
           for group in model.policy.optimizer.param_groups):
        raise ValueError('Adam configuration drift')
    trainable = {id(p) for p in model.policy.parameters() if p.requires_grad}
    optimized = [id(p) for group in model.policy.optimizer.param_groups for p in group['params']]
    if len(optimized) != len(set(optimized)) or set(optimized) != trainable or any(
            p.requires_grad for p in model.policy.fire_net.parameters()):
        raise ValueError('Optimizer/frozen-parameter mapping changed')
    for parameter in model.policy.parameters():
        if not th.isfinite(parameter).all():
            raise FloatingPointError('Nonfinite policy state')


def save_bundle(destination, model, state, kind):
    """Publish one complete pair by rename; never mutate a published bundle."""
    destination = Path(destination).resolve()
    if not destination.is_relative_to(RUN_ROOT.resolve()):
        raise ValueError('Checkpoint output must remain inside the exclusive flight run root')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / ('.flight-bundle-' + uuid.uuid4().hex)
    stage.mkdir(exist_ok=False)
    try:
        model.save(stage / 'candidate.zip')
        with zipfile.ZipFile(stage / 'candidate.zip') as archive:
            require_state = (state['smoke_counters']['optimizer_completed_steps'] > 0
                             if kind == 'smoke' else model.num_timesteps > 0)
            validate_saved_optimizer(archive, require_state)
        # Statistics are frozen. Preserve original bytes, not transient returns.
        statistics = (SOURCE / 'candidate_vecnormalize.pkl').read_bytes()
        if fire.digest(statistics) != source_statistics_sha256():
            raise ValueError('Source statistics changed before checkpoint publication')
        with (stage / 'candidate_vecnormalize.pkl').open('xb') as stream:
            stream.write(statistics)
        metadata = dict(schema='flight-ppo-smoke-v1' if kind == 'smoke' else 'flight-ppo-bundle-v1',
            kind=kind, steps=int(model.num_timesteps),
            source_steps=SOURCE_STEPS, optimizer=('smoke-only Adam; not resumable' if kind == 'smoke'
                else 'fresh flight Adam at initialization; restored on resume'),
            state=state, hashes={n: sha256(stage / n) for n in ('candidate.zip', 'candidate_vecnormalize.pkl')})
        write_json(stage / 'bundle.json', metadata)
        stage.rename(destination)
    except BaseException:
        # Failed private staging is retained as evidence, never resumable.
        raise
    return destination


def publish_copy(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not destination.is_relative_to(RUN_ROOT.resolve()) or destination.is_relative_to(source):
        raise ValueError('Invalid/nested best-checkpoint destination')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / ('.best-bundle-' + uuid.uuid4().hex)
    shutil.copytree(source, stage)
    for name in ('candidate.zip', 'candidate_vecnormalize.pkl', 'bundle.json'):
        if sha256(stage / name) != sha256(Path(source) / name):
            raise ValueError('Best-checkpoint copy changed')
    stage.rename(destination)


def validate_saved_optimizer(archive, require_state):
    weights = th.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
    if any(not isinstance(value, th.Tensor) or not th.isfinite(value).all() for value in weights.values()):
        raise ValueError('Nonfinite/malformed saved policy tensors')
    optimizer = th.load(io.BytesIO(archive.read('policy.optimizer.pth')), map_location='cpu', weights_only=True)
    parameters = [value for name, value in weights.items() if not name.startswith('fire_net.')]
    groups = optimizer['param_groups']
    if len(groups) != 1 or groups[0]['params'] != list(range(len(parameters))):
        raise ValueError('Saved optimizer parameter mapping includes wrong/frozen tensors')
    states = optimizer['state']
    if not states and not require_state:
        return
    if set(states) != set(range(len(parameters))):
        raise ValueError('Saved optimizer state is incomplete')
    for index, parameter in enumerate(parameters):
        moment = states[index]
        if set(moment) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Invalid Adam state fields')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = moment[key]
            if value.shape != parameter.shape or value.dtype != parameter.dtype or not th.isfinite(value).all():
                raise ValueError('Invalid Adam moment shape/dtype/finiteness')
        if (not th.isfinite(moment['step']).all() or moment['step'].numel() != 1 or moment['step'] < 0
                or (moment['exp_avg_sq'] < 0).any()):
            raise ValueError('Invalid Adam step/second moment')


def read_bundle(path, cfg, expected_pins, resume=False):
    path = Path(path).resolve()
    if not path.is_relative_to(RUN_ROOT.resolve()) or 'sealed' in str(path).lower() or path.name.startswith('.'):
        raise ValueError('Unauthorized/incomplete checkpoint location')
    data = read_json(path / 'bundle.json')
    if data['schema'] != 'flight-ppo-bundle-v1' or data['state']['config'] != cfg:
        raise ValueError('Flight bundle configuration mismatch')
    if data['state']['pins'] != expected_pins or data['state']['versions'] != versions():
        raise ValueError('Flight bundle lineage/version mismatch')
    required = {'steps', 'segment', 'development', 'reference_bundle', 'reference_result', 'selected',
                'selected_source', 'normalization_hash', 'fire_tensor_hash', 'protocol_sha256'}
    if not required.issubset(data['state']):
        raise ValueError('Incomplete flight training metadata')
    budget = training_budget(cfg)
    if type(data['steps']) is not int or not 0 <= data['steps'] <= budget or data['steps'] % cfg['n_steps']:
        raise ValueError('Invalid completed training budget')
    if data['source_steps'] != SOURCE_STEPS or data['state']['steps'] != data['steps']:
        raise ValueError('Training/source counters disagree')
    for name, expected in data['hashes'].items():
        if name not in ('candidate.zip', 'candidate_vecnormalize.pkl') or sha256(path / name) != expected:
            raise ValueError('Checkpoint/statistics hash mismatch')
    if set(data['hashes']) != {'candidate.zip', 'candidate_vecnormalize.pkl'}:
        raise ValueError('Incomplete checkpoint pair')
    if data['hashes']['candidate_vecnormalize.pkl'] != source_statistics_sha256():
        raise ValueError('Statistics are not the exact Stage 5 source bytes')
    with zipfile.ZipFile(path / 'candidate.zip') as archive:
        if 'policy.optimizer.pth' not in archive.namelist():
            raise ValueError('Optimizer state missing')
        header = json.loads(archive.read('data'))
        if header['num_timesteps'] != data['steps']:
            raise ValueError('Checkpoint counter disagrees with metadata')
        if any(header[key] != cfg[key] for key in ('n_steps', 'batch_size', 'n_epochs', 'learning_rate',
                'gamma', 'gae_lambda', 'ent_coef', 'vf_coef', 'max_grad_norm', 'target_kl')):
            raise ValueError('Checkpoint PPO configuration mismatch')
        if header['policy_kwargs'] != dict(enable_conjunction_v1=True, optimizer_kwargs={'eps': cfg['adam_epsilon']}):
            raise ValueError('Checkpoint fixed-fire policy configuration mismatch')
        if header['n_envs'] != 1 or header['clip_range_vf'] is not None or not header['normalize_advantage']:
            raise ValueError('Checkpoint rollout/value-clipping configuration mismatch')
        validate_saved_optimizer(archive, data['steps'] > 0)
    if resume and (data['kind'] not in ('periodic', 'final') or data['steps'] == budget):
        raise ValueError('Resume requires a committed non-final-budget periodic/final checkpoint')
    if resume:
        steps = [item['step'] for item in data['state']['development']]
        if steps != list(range(cfg['validation_every'], data['steps'] + 1, cfg['validation_every'])):
            raise ValueError('Resume development schedule is incomplete or duplicated')
        if not data['state'].get('reference_result'):
            raise ValueError('Resume reference validation is missing')
    return data


def record(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def load_record(reference):
    path = Path(reference['path']).resolve()
    if not path.is_relative_to(RUN_ROOT.resolve()) or 'sealed' in str(path).lower():
        raise ValueError('Unauthorized saved development record')
    data = read_json(path, reference['sha256'])
    if (path.parent / 'error.json').exists() or data.get('status') != 'complete':
        raise ValueError('Incomplete development result')
    integrity = read_json(path.parent / 'integrity.json')
    if integrity['result_sha256'] != reference['sha256']:
        raise ValueError('Development completion hash mismatch')
    return data


def choose_best(reference, candidates, common_steps=500):
    """All monitored checkpoints share one horizon; final outcomes never enter."""
    if not candidates:
        return None
    data = [reference] + [item['data'] for item in candidates]
    if any(len(d['episodes']) != 10 or len(d['traces']) != 10 for d in data):
        raise ValueError('Development requires exactly ten complete cases')
    for d in data:
        if [r['episode_id'] for r in d['episodes']] != [r['episode_id'] for r in reference['episodes']]:
            raise ValueError('Development case identity/order mismatch')
        if [r['initial_sha256'] for r in d['episodes']] != [r['initial_sha256'] for r in reference['episodes']]:
            raise ValueError('Development initial-state fingerprint mismatch')
        if any(not np.isfinite([r['steps'], r['longest_same_side_large_bank_s'], r['opposing_bank_yaw_fraction']]).all()
               for r in d['episodes']) or any(not np.isfinite(row['bearing_error_deg']) for trace in d['traces'] for row in trace):
            raise FloatingPointError('Nonfinite development metric')
    horizon = min(common_steps, *(len(t) for d in data for t in d['traces']))
    if horizon <= 0:
        raise ValueError('Empty development trace')
    def metrics(d):
        rows = d['episodes']
        traces = {('baseline', 'deterministic', i): t for i, t in enumerate(reference['traces'])}
        traces.update({('candidate', 'deterministic', i): t for i, t in enumerate(d['traces'])})
        bearing = paired_bearing_metrics(traces, horizon)['deterministic']
        return dict(hits=sum(r['episode_success'] for r in rows),
            safety=sum(r['safety_termination'] for r in rows),
            invalid=sum(r['invalid_actions'] for r in rows), violations=sum(r['gate_violations'] for r in rows),
            completion=float(np.mean([r['steps'] if r['episode_success'] else 2000 for r in rows])),
            bearing=bearing['mean']['candidate']['common_horizon_mean_absolute_bearing_deg'],
            bank=float(np.mean([r['longest_same_side_large_bank_s'] for r in rows])),
            max_bank=max(r['longest_same_side_large_bank_s'] for r in rows),
            opposition=float(np.mean([r['opposing_bank_yaw_fraction'] for r in rows])),
            bearing_comparison=bearing)
    base = metrics(reference)
    def ranking(m):
        return (m['hits'], -m['completion'], -m['bearing'], -m['bank'], -m['opposition'])
    best = None
    for item in candidates:
        m = metrics(item['data'])
        eligible = (m['invalid'] == m['violations'] == 0 and m['safety'] <= base['safety']
                    and m['hits'] >= base['hits'] and m['bearing'] <= base['bearing']
                    and m['bank'] <= base['bank'] and m['opposition'] <= base['opposition'])
        if eligible and ranking(m) > ranking(base) and (best is None or ranking(m) > ranking(best['metrics'])):
            best = dict(step=item['step'], bundle=item['bundle'], metrics=m,
                        reference_metrics=base, common_horizon_steps=horizon)
    return best


def launch_worker(run, checkpoint, output, final=False):
    subprocess.run([sys.executable, '-B', str(ROOT / 'train_flight_ppo.py'),
        '--final-worker' if final else '--development-worker', '--run-dir', str(run),
        '--checkpoint', str(checkpoint), '--output', str(output), '--enable-conjunction-v1',
        '--source', str(SOURCE)],
        cwd=ROOT, check=True)
    reference = record(Path(output) / 'result.json')
    load_record(reference)
    return reference


def worker(args):
    """Reuse existing episode metrics, isolated from the learner's environment."""
    run = Path(args.run_dir).resolve()
    if not run.is_relative_to(RUN_ROOT.resolve()):
        raise ValueError('Worker run is outside flight output root')
    run_info = read_json(run / 'run.json')
    cfg = config(run / 'config.json')
    expected = pins()
    if run_info['pins'] != expected:
        raise ValueError('Worker input lineage drift')
    if (run_info.get('config_sha256') != sha256(run / 'config.json')
            or run_info.get('additional_step_budget') != training_budget(cfg)):
        raise ValueError('Worker configuration/budget lineage mismatch')
    protocol = read_json(run / 'protocol.json', run_info['protocol_sha256'])
    checkpoint = Path(args.checkpoint).resolve()
    meta = read_bundle(checkpoint, cfg, expected)
    if meta['state']['protocol_sha256'] != run_info['protocol_sha256']:
        raise ValueError('Worker checkpoint/protocol lineage mismatch')
    output = Path(args.output).resolve()
    expected_parent = run / ('final_evaluation' if args.final_worker else 'development')
    if output.parent != expected_parent or output.exists():
        raise FileExistsError('Worker requires an exclusive run-local output')
    if args.final_worker:
        completion = read_json(run / 'result.json')
        if not completion.get('selected') or Path(completion['selected']).resolve() != checkpoint:
            raise ValueError('Final evaluation requires the selected completed-run checkpoint')
    output.mkdir(parents=True, exist_ok=False)
    norm = None
    try:
        from evaluate_turning_ppo import episode
        final = args.final_worker
        cases = protocol['final' if final else 'development']
        bundles = [checkpoint] if not final else [Path(run_info['reference_bundle']), checkpoint]
        groups = []
        for bundle in bundles:
            bundle_meta = read_bundle(bundle, cfg, expected)
            norm = make_norm(cfg, protocol, cfg['seed'])
            model = PPO.load(bundle / 'candidate.zip', env=norm, device='cpu')
            assert_invariants(model, norm, bundle_meta['state'])
            rows, traces = [], []
            for mode in (('deterministic', 'sampled') if final else ('deterministic',)):
                for case in cases:
                    row, trace = episode(model, norm, case['scenario'], case['environment_seed'],
                        case['policy_seed'], mode == 'deterministic', cfg['max_episode_steps'])
                    raw = norm.venv.envs[0].unwrapped
                    initial = dict(raw.last_initial_state, policy_seed=case['policy_seed'])
                    row.update(raw.episode_metrics, episode_id=case['episode_id'], pair_id=case['pair_id'],
                               mode=mode, initial_state=initial,
                               initial_sha256=fire.digest(json.dumps(initial, sort_keys=True, separators=(',', ':')).encode()))
                    rows.append(row)
                    traces.append(trace)
                    assert_invariants(model, norm, bundle_meta['state'])
            groups.append(dict(bundle=str(bundle), episodes=rows, traces=traces))
            norm.close()
            norm = None
        if final:
            if any(a['initial_sha256'] != b['initial_sha256'] for a, b in zip(groups[0]['episodes'], groups[1]['episodes'])):
                raise ValueError('Final reference/candidate initial states differ')
            result = dict(status='complete', groups=groups, cases=20, mirror_pairs=10,
                          note='Descriptive small-sample paired evaluation; no significance or general reliability claim')
        else:
            result = dict(status='complete', **groups[0])
        if pins() != expected:
            raise ValueError('Worker inputs changed')
        write_json(output / 'result.json', result)
        write_json(output / 'integrity.json', dict(result_sha256=sha256(output / 'result.json'), pins=expected))
    except BaseException as exc:
        write_json(output / 'error.json', dict(status='incomplete', error=repr(exc)))
        raise
    finally:
        if norm is not None:
            norm.close()


class FlightLogs(ExperimentLogs):
    def __init__(self, run, cfg, state):
        super().__init__(run, core_config(cfg), 0, None)
        self.cfg, self.state = cfg, state
        self.last_save = state['steps']
        self.best_source = state.get('selected_source')
        self.flight_file = (run / 'flight_episodes.jsonl').open('x', encoding='utf-8', newline='\n')

    def _on_step(self):
        if self.model.num_timesteps > training_budget(self.cfg):
            raise RuntimeError('Training budget exceeded')
        if self.training_env.training or any(not np.isfinite(v) for info in self.locals['infos'] for v in info['reward_breakdown'].values()):
            raise FloatingPointError('Invalid normalization/reward state')
        for info in self.locals['infos']:
            if 'episode' in info:
                self.flight_file.write(json.dumps(dict(training_steps=int(self.model.num_timesteps),
                    metrics=info['flight_metrics'], curriculum=info['flight_curriculum'],
                    termination=info.get('terminal_reason'), episode=info['episode']), allow_nan=False) + '\n')
                self.flight_file.flush()
                for key, value in info['flight_metrics'].items():
                    if value is not None:
                        self.logger.record_mean('flight/episode_' + key, float(value))
            self.logger.record('flight/curriculum_stage', info['flight_curriculum']['stage'])
        return super()._on_step()

    def completed_update(self):
        step = int(self.model.num_timesteps)
        assert_invariants(self.model, self.training_env, self.state)
        for key, value in self.logger.name_to_value.items():
            if key.startswith('train/') and isinstance(value, (float, int, np.number)) and not np.isfinite(value):
                raise FloatingPointError('Nonfinite PPO metric: ' + key)
        if step == self.last_save or step % self.cfg['checkpoint_every']:
            return
        self.flush_metrics()
        self.state['steps'] = step
        self.state['curriculum'] = self.training_env.venv.envs[0].unwrapped.curriculum.state()
        if step % self.cfg['validation_every'] == 0:
            candidate = save_bundle(self.run / 'validation_inputs' / f'step_{step:012d}', self.model, self.state, 'validation_input')
            result = launch_worker(self.run, candidate, self.run / 'development' / f'step_{step:012d}')
            self.state['development'].append(dict(step=step, bundle=str(candidate), result=result))
            reference = load_record(self.state['reference_result'])
            candidates = [{**item, 'data': load_record(item['result'])} for item in self.state['development']]
            selected = choose_best(reference, candidates, self.cfg['common_horizon_steps'])
            write_json(self.run / 'development' / f'selection_{step:012d}.json', selected)
            self.state['selected_source'] = selected['bundle'] if selected else None
            if selected and selected['bundle'] != self.best_source:
                source = Path(selected['bundle'])
                destination = self.run / 'best' / f'selection_step_{step:012d}'
                read_bundle(source, self.cfg, self.state['pins'])
                publish_copy(source, destination)
                self.state['selected'] = str(destination)
            elif not selected:
                self.state['selected'] = None
            self.best_source = self.state['selected_source']
        save_bundle(self.run / 'checkpoints' / f'step_{step:012d}', self.model, self.state, 'periodic')
        self.last_save = step

    def _on_rollout_start(self):
        self.completed_update()  # Previous rollout has already been optimized.

    def _on_training_end(self):
        self.completed_update()  # Final optimization has no next rollout-start hook.
        self.flush_metrics()


def train(args):
    cfg = config(args.config)
    budget = training_budget(cfg)
    expected = pins()
    if Path(args.source).resolve() != SOURCE.resolve():
        raise ValueError('This experiment is initialized only from protected Candidate A')
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        raise ValueError('Invalid exclusive run ID')
    run = RUN_ROOT / args.run_id
    if run.exists():
        raise FileExistsError(run)
    protocol = prepare_protocol(cfg)
    previous = read_bundle(args.resume, cfg, expected, resume=True) if args.resume else None
    if previous:
        state = previous['state']
        read_bundle(state['reference_bundle'], cfg, expected)
        load_record(state['reference_result'])
        for item in state['development']:
            read_bundle(item['bundle'], cfg, expected)
            load_record(item['result'])
        if state['selected']:
            read_bundle(state['selected'], cfg, expected)
        protocol_raw = (json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        if fire.digest(protocol_raw) != state['protocol_sha256']:
            raise ValueError('Resume protocol changed')
    completed = previous['steps'] if previous else 0
    remaining = remaining_steps(completed, cfg)
    segment = previous['state']['segment'] + 1 if previous else 0
    seed = cfg['seed'] + segment
    if seed in reservations(protocol)[1]:
        raise ValueError('Segment seed collides with reserved evaluation seeds')
    # Reservation and immutable plan publication precede environment creation.
    run.mkdir(parents=True, exist_ok=False)
    with (run / 'config.json').open('xb') as stream:
        stream.write(Path(args.config).read_bytes())
    if config(run / 'config.json') != cfg:
        raise ValueError('Configuration changed before run publication')
    write_json(run / 'protocol.json', protocol)
    reference_bundle = previous['state']['reference_bundle'] if previous else str(run / 'initial')
    write_json(run / 'run.json', dict(pins=expected, protocol_sha256=sha256(run / 'protocol.json'),
               reference_bundle=reference_bundle, segment=segment, seed=seed,
               config_sha256=sha256(run / 'config.json'), additional_step_budget=budget,
               final_evaluation='Pending a separate explicit --final-worker command, never automatic'))
    norm = callback = None
    try:
        norm = make_norm(cfg, protocol, seed, completed)
        if previous:
            model = PPO.load(Path(args.resume) / 'candidate.zip', env=norm, device='cpu')
            state = dict(previous['state'])
            if state['protocol_sha256'] != sha256(run / 'protocol.json'):
                raise ValueError('Resume protocol changed')
        else:
            source = PPO.load(SOURCE / 'candidate.zip', device='cpu')
            if int(source.num_timesteps) != SOURCE_STEPS:
                raise ValueError('Unexpected source counter')
            model = PPO(FixedConjunctionFlightPolicy, norm, seed=seed, **model_options(cfg),
                        policy_kwargs=dict(enable_conjunction_v1=True, optimizer_kwargs={'eps': cfg['adam_epsilon']}))
            transfer_initialization(source.policy, model.policy)
            # Fixed synthetic observations, no rollout or policy evaluation data.
            obs = th.zeros((2, 19))
            with th.no_grad():
                old = source.policy.get_distribution(obs).gaussian
                new = model.policy.get_distribution(obs).distribution
                if not th.equal(old.mean, new.mean) or not th.equal(old.stddev, new.stddev):
                    raise ValueError('Initial Gaussian outputs differ from Candidate A')
            state = dict(config=cfg, pins=expected, versions=versions(), steps=0, development=[],
                         reference_bundle=reference_bundle, reference_result=None, selected=None,
                         selected_source=None, normalization_hash=normalization_hash(norm),
                         fire_tensor_hash=fire_tensor_hash(model.policy),
                         protocol_sha256=sha256(run / 'protocol.json'))
        state.update(segment=segment, steps=completed)
        model.set_random_seed(seed)
        model.set_logger(configure(str(run / 'logs'), ['stdout', 'csv']))
        assert_invariants(model, norm, state)
        save_bundle(run / 'initial', model, state, 'initial')
        if not previous:
            state['reference_result'] = launch_worker(run, run / 'initial', run / 'development' / 'step_000000000000')
        else:
            load_record(state['reference_result'])
            for item in state['development']:
                read_bundle(item['bundle'], cfg, expected)
                load_record(item['result'])
        callback = FlightLogs(run, cfg, state)
        model.learn(total_timesteps=remaining, reset_num_timesteps=False, callback=callback)
        if model.num_timesteps != budget:
            raise RuntimeError('Run ended before the declared budget without a completed result')
        assert_invariants(model, norm, state)
        save_bundle(run / 'final', model, state, 'final')
        write_json(run / 'result.json', dict(status='complete', training_steps=budget,
                   source_steps=SOURCE_STEPS, weight_lineage_steps=SOURCE_STEPS + budget,
                   selected=state['selected'], development=state['development'],
                   final_evaluation_pending=bool(state['selected']), no_production_promotion=True))
    except BaseException as exc:
        write_json(run / 'error.json', dict(status='incomplete', error=repr(exc)))
        raise
    finally:
        if callback is not None:
            callback.file.close()
            callback.flight_file.close()
        if norm is not None:
            norm.close()
        if pins() != expected:
            raise ValueError('Protected inputs changed')


SMOKE_DEFAULT_STEPS = 64
SMOKE_MAX_STEPS = 256


class OptimizerStepBudget:
    """Reserve the only slot BEFORE Adam can mutate parameters; never retry."""
    def __init__(self, optimizer, parameters):
        self.parameters = list(parameters)
        self.reserved = self.completed = self.changed = self.blocked = 0
        self.before = None
        self.before_update = None
        self.handles = [optimizer.register_step_pre_hook(self.before_step),
                        optimizer.register_step_post_hook(self.after_step)]

    def before_step(self, optimizer, args, kwargs):
        if self.reserved >= 1:
            self.blocked += 1
            raise RuntimeError('Smoke optimizer limit: a second step is forbidden before parameter mutation')
        for parameter in self.parameters:
            if not th.isfinite(parameter).all() or (parameter.grad is not None and not th.isfinite(parameter.grad).all()):
                raise FloatingPointError('Nonfinite smoke parameter/gradient before optimizer step')
        if self.before_update is not None:
            self.before_update()
        self.before = [p.detach().clone() for p in self.parameters]
        self.reserved += 1

    def after_step(self, optimizer, args, kwargs):
        self.completed += 1
        self.changed += int(self.current_changed())
        if any(not th.isfinite(p).all() for p in self.parameters):
            raise FloatingPointError('Nonfinite smoke parameter after optimizer step')

    def current_changed(self):
        return self.before is not None and any(not th.equal(old, p.detach())
                                               for old, p in zip(self.before, self.parameters))

    def counts(self):
        return dict(optimizer_slots_reserved=self.reserved, optimizer_completed_steps=self.completed,
                    optimizer_parameter_updates=self.changed, optimizer_blocked_steps=self.blocked,
                    parameter_change_since_reserved_step=bool(self.current_changed()))

    def close(self):
        for handle in self.handles:
            handle.remove()


class SmokeFlightEnv(TrainingFlightEnv):
    """One constructor reset, one bounded episode, no curriculum or autoreset."""
    def __init__(self, cfg, limit):
        self.smoke_limit = limit
        self.smoke_seed = cfg['seed']
        self.smoke_resets = self.smoke_steps = self.smoke_completed_steps = 0
        self.smoke_done = False
        self.smoke_scenario = sample_scenario(np.random.default_rng(self.smoke_seed), family='ahead')
        environment_config = replace(core_config(cfg), total_timesteps=limit, max_episode_steps=limit)
        super().__init__(environment_config, self.smoke_seed)

    def reset(self, *, seed=None, options=None):
        if self.smoke_resets:
            raise RuntimeError('Smoke episode limit: a second reset is forbidden')
        if options is not None or seed not in (None, self.smoke_seed):
            raise ValueError('Smoke accepts only its registered initial scenario and seed')
        self.smoke_resets += 1  # Reserve before any real reset, including failure.
        obs, info = super().reset(seed=self.smoke_seed,
                                 options={'turning_scenario': asdict(self.smoke_scenario)})
        self.smoke_observation = obs.copy()
        return obs, info

    def step(self, action):
        if self.smoke_resets != 1 or self.smoke_done or self.smoke_steps >= self.smoke_limit:
            raise RuntimeError('Smoke episode/step limit reached; no simulator step performed')
        self.smoke_steps += 1
        try:
            result = super().step(action)
        except BaseException:
            self.smoke_done = True
            raise
        self.smoke_completed_steps += 1
        self.smoke_done = bool(result[2] or result[3])
        return result


def smoke_limits(args, cfg):
    limit = SMOKE_DEFAULT_STEPS if args.smoke_steps is None else args.smoke_steps
    if type(limit) is not int or not 2 <= limit <= min(SMOKE_MAX_STEPS, cfg['max_episode_steps']):
        raise ValueError('Smoke step cap must be 2..256 and within the episode limit')
    if any((args.run_id, args.resume, args.run_dir, args.checkpoint)) or args.output is None:
        raise ValueError('Smoke requires --output and forbids training/resume/worker arguments')
    if Path(args.source).resolve() != SOURCE.resolve():
        raise ValueError('Smoke initialization must be protected Candidate A')
    output = Path(args.output).resolve()
    if output.parent != RUN_ROOT.resolve() or not output.name.startswith('smoke_'):
        raise ValueError('Smoke output must be a new smoke_* directory directly under flight_policy')
    if output.exists():
        raise FileExistsError(output)
    return limit, output


def equal_tensor_state(left, right):
    if isinstance(left, th.Tensor):
        return isinstance(right, th.Tensor) and th.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_tensor_state(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_tensor_state(a, b) for a, b in zip(left, right))
    return left == right


def smoke_configuration(base, limit):
    return dict(base, schema='fixed-conjunction-flight-smoke-v1', total_timesteps=limit,
                max_episode_steps=limit, n_steps=limit, batch_size=limit, n_epochs=1)


def checkpoint_has_no_environment(path):
    with zipfile.ZipFile(path) as archive:
        header = json.loads(archive.read('data'))
    if header.get('env') is not None:
        raise ValueError('Smoke refuses checkpoints containing an environment')


def smoke_loss_audit(model, transitions, buffer):
    """Read-only full-batch check of the installed SB3 PPO scalar loss.

    Called at the optimizer boundary. The real update still uses PPO.train().
    No second optimization loop or custom gradient is used.
    """
    obs = th.as_tensor(np.concatenate([t[0] for t in transitions]), device=model.device)
    actions = th.as_tensor(np.concatenate([t[1] for t in transitions]), device=model.device)
    returns = th.as_tensor(buffer.returns.reshape(-1), device=model.device)
    advantages = th.as_tensor(buffer.advantages.reshape(-1), device=model.device)
    old_logp = th.cat([t[4] for t in transitions])
    with th.no_grad():
        values, logp, entropy = model.policy.evaluate_actions(obs, actions)
        if model.normalize_advantage:
            deviation = advantages.std()
            if not th.isfinite(deviation):
                raise FloatingPointError('Nonfinite smoke advantage standard deviation')
            advantages = (advantages - advantages.mean()) / (deviation + 1e-8)
        ratio = th.exp(logp - old_logp)
        clip = model.clip_range(model._current_progress_remaining)
        policy_loss = -th.min(advantages * ratio, advantages * th.clamp(ratio, 1 - clip, 1 + clip)).mean()
        value_loss = (returns - values.flatten()).square().mean()
        entropy_loss = -entropy.mean()
        loss = policy_loss + model.vf_coef * value_loss + model.ent_coef * entropy_loss
        terms = dict(policy_loss=policy_loss, value_loss=value_loss, entropy_loss=entropy_loss, total_loss=loss)
        if any(not th.isfinite(v).all() for v in (returns, advantages, ratio, *terms.values())):
            raise FloatingPointError('Nonfinite smoke PPO loss before optimizer update')
        return {k: float(v) for k, v in terms.items()}


def smoke_test(args):
    """Single episode + zero/one guarded SB3 update. Never calls learn/workers."""
    base = config(args.config)
    limit, output = smoke_limits(args, base)
    expected = pins()
    checkpoint_has_no_environment(SOURCE / 'candidate.zip')
    smoke_cfg = smoke_configuration(base, limit)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', dict(mode='smoke-only', config=smoke_cfg,
        full_training_config_sha256=sha256(args.config), pins=expected, versions=versions(),
        max_episodes=1, max_optimizer_steps=1, max_environment_steps=limit,
        evaluation_suites=False, curriculum_progression=False, full_training=False))
    raw = norm = guard = None
    try:
        # Retain a reference for cleanup/counters even if the constructor fails.
        raw = SmokeFlightEnv.__new__(SmokeFlightEnv)
        raw.__init__(base, limit)  # Its constructor performs the only reset.
        vector = DummyVecEnv([lambda: raw])
        norm = VecNormalize.load(str(SOURCE / 'candidate_vecnormalize.pkl'), vector)
        norm.action_space, norm.training, norm.norm_reward = vector.action_space, False, False
        source = PPO.load(SOURCE / 'candidate.zip', device='cpu')
        if int(source.num_timesteps) != SOURCE_STEPS:
            raise ValueError('Unexpected smoke source checkpoint')
        model = PPO(FixedConjunctionFlightPolicy, norm, seed=base['seed'], **model_options(smoke_cfg),
                    policy_kwargs=dict(enable_conjunction_v1=True, optimizer_kwargs={'eps': base['adam_epsilon']}))
        transfer_initialization(source.policy, model.policy)
        model.set_random_seed(base['seed'])
        model.set_logger(configure(str(output / 'logs'), ['csv']))
        state = dict(config=smoke_cfg, pins=expected, versions=versions(), steps=0,
                     normalization_hash=normalization_hash(norm), fire_tensor_hash=fire_tensor_hash(model.policy))
        assert_invariants(model, norm, state)
        guard = OptimizerStepBudget(model.policy.optimizer,
                                    [p for p in model.policy.parameters() if p.requires_grad])
        observation = raw.smoke_observation.copy()
        transitions = []
        loss_audit = {}
        terminated = truncated = False
        info = {}
        with (output / 'episode.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for step in range(1, limit + 1):
                normalized = norm.normalize_obs(observation[None, :])
                if not np.isfinite(normalized).all():
                    raise FloatingPointError('Nonfinite smoke normalized observation')
                tensor, _ = model.policy.obs_to_tensor(normalized)
                with th.no_grad():
                    action, value, logp = model.policy(tensor)
                    distribution = model.policy.action_dist
                    gaussian_logp = distribution.distribution.log_prob(action[:, :4]).sum(-1)
                    entropy = distribution.entropy()
                    gaussian_entropy = distribution.distribution.entropy().sum(-1)
                fixed = float(model.policy.fire_adapter._model.request(normalized[0]))
                if (not th.equal(logp, gaussian_logp) or not th.equal(entropy, gaussian_entropy)
                        or float(action[0, 4]) != fixed or not th.isfinite(entropy).all()):
                    raise RuntimeError('Smoke fixed-fire/Gaussian-only objective mismatch')
                selected = action.cpu().numpy()
                applied = np.clip(selected[0], raw.action_space.low, raw.action_space.high)
                observation, reward, terminated, truncated, info = raw.step(applied)
                model.num_timesteps += 1
                transitions.append((normalized.copy(), selected.copy(), float(reward), value, logp))
                stream.write(json.dumps(dict(step=step, normalized_observation=normalized[0].tolist(),
                    selected_action=selected[0].tolist(), applied_action=applied.tolist(),
                    fixed_fire=fixed, gaussian_log_probability=float(logp[0]), gaussian_entropy=float(entropy[0]),
                    reward=float(reward), terminated=bool(terminated), truncated=bool(truncated),
                    gate=raw._observer.record, weapon=info['weapon'], reward_components=info['reward_breakdown']),
                    sort_keys=True, allow_nan=False) + '\n')
                stream.flush()
                if normalization_hash(norm) != state['normalization_hash']:
                    raise ValueError('Smoke normalization changed')
                if terminated or truncated:
                    break  # Never step or reset after the first episode ends.
        count = len(transitions)
        if raw.smoke_resets != 1 or not raw.smoke_done or count != raw.smoke_completed_steps:
            raise RuntimeError('Smoke episode did not end consistently within its cap')
        if count >= 2:
            # One actual minibatch in one epoch, even for an early termination.
            model.n_steps = model.batch_size = count
            smoke_cfg.update(n_steps=count, batch_size=count)
            buffer = RolloutBuffer(count, model.observation_space, model.action_space,
                                  device='cpu', gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=1)
            if truncated and not terminated:
                with th.no_grad():
                    terminal_tensor, _ = model.policy.obs_to_tensor(norm.normalize_obs(observation[None, :]))
                    bootstrap = float(model.policy.predict_values(terminal_tensor)[0, 0])
                if not np.isfinite(bootstrap):
                    raise FloatingPointError('Nonfinite smoke truncation bootstrap')
                last = transitions[-1]
                transitions[-1] = (last[0], last[1], last[2] + model.gamma * bootstrap, last[3], last[4])
            for index, (obs, act, reward, value, logp) in enumerate(transitions):
                buffer.add(obs, act, np.array([reward], np.float32), np.array([index == 0]), value, logp)
            buffer.compute_returns_and_advantage(th.zeros(1), np.ones(1, dtype=bool))
            if not np.isfinite(buffer.returns).all() or not np.isfinite(buffer.advantages).all():
                raise FloatingPointError('Nonfinite smoke return/advantage')
            model.rollout_buffer = buffer
            guard.before_update = lambda: loss_audit.update(smoke_loss_audit(model, transitions, buffer))
            model.train()  # Public SB3 loss; optimizer boundary guard permits at most one step.
        losses = {k: float(v) for k, v in model.logger.name_to_value.items()
                  if k.startswith('train/') and isinstance(v, (float, int, np.number))}
        if any(not np.isfinite(v) for v in losses.values()):
            raise FloatingPointError('Nonfinite smoke PPO loss/metric')
        if guard.completed > 1 or guard.reserved > 1 or guard.blocked:
            raise RuntimeError('Smoke optimizer budget violation')
        if any(p.grad is not None for p in model.policy.fire_net.parameters()):
            raise RuntimeError('Smoke frozen fire tensors received gradients')
        assert_invariants(model, norm, state)
        state.update(steps=count, smoke_counters=guard.counts())
        checkpoint = save_bundle(output / 'checkpoint', model, state, 'smoke')
        metadata = read_json(checkpoint / 'bundle.json')
        for name, expected_hash in metadata['hashes'].items():
            if sha256(checkpoint / name) != expected_hash:
                raise ValueError('Smoke checkpoint hash mismatch')
        checkpoint_has_no_environment(checkpoint / 'candidate.zip')
        reloaded = PPO.load(checkpoint / 'candidate.zip', device='cpu')  # No environment construction.
        reloaded_norm = VecNormalize.load(str(checkpoint / 'candidate_vecnormalize.pkl'), vector)
        reloaded_norm.action_space, reloaded_norm.training, reloaded_norm.norm_reward = vector.action_space, False, False
        assert_invariants(reloaded, reloaded_norm, state)
        if (reloaded.get_env() is not None or raw.smoke_resets != 1
                or not equal_tensor_state(model.policy.state_dict(), reloaded.policy.state_dict())
                or not equal_tensor_state(model.policy.optimizer.state_dict(), reloaded.policy.optimizer.state_dict())):
            raise ValueError('Smoke checkpoint/optimizer reload mismatch or unexpected environment')
        result = dict(status='complete', mode='smoke-only', configured_step_cap=limit,
            episode_steps=count, episode_resets=raw.smoke_resets, episode_completed=True,
            terminated=bool(terminated), truncated=bool(truncated), terminal_reason=info.get('terminal_reason'),
            optimizer=guard.counts(), optimizer_skip_reason='fewer than two transitions' if count < 2 else None,
            optimizer_integration_exercised=bool(guard.completed),
            optimizer_parameter_change_observed=bool(guard.changed),
            losses=losses, loss_audit_before_step=loss_audit,
            gaussian_only_log_probability_and_entropy=True, fire_gradients_absent=True,
            finite_observations_actions_rewards=True, finite_losses=True if count >= 2 else None,
            finite_gradients_checked_before_step=bool(guard.reserved),
            checkpoint_reload_equal=True, optimizer_reload_equal=True, normalization_unchanged=True,
            statistics_sha256=metadata['hashes']['candidate_vecnormalize.pkl'], episode_metrics=raw.episode_metrics,
            input_pins=expected, checkpoint_hashes=metadata['hashes'],
            no_evaluation_suites=True, no_curriculum_progression=True, no_full_training=True)
        write_json(output / 'result.json', result)
        write_json(output / 'integrity.json', dict(pins=expected, output_sha256={
            p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}))
        return result
    except BaseException as exc:
        write_json(output / 'error.json', dict(status='incomplete', error=repr(exc),
            episode_reset_attempts=getattr(raw, 'smoke_resets', 0),
            episode_steps_reserved=getattr(raw, 'smoke_steps', 0),
            episode_steps_completed=getattr(raw, 'smoke_completed_steps', 0),
            optimizer=guard.counts() if guard else None))
        raise
    finally:
        if guard is not None:
            guard.close()
        if norm is not None:
            norm.close()
        elif raw is not None and getattr(raw, 'sim', None) is not None:
            raw.close()
        if pins() != expected:
            raise ValueError('Protected inputs changed during smoke test')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE,
                        help='Stage 5 bundle directory (default: the supplied Stage 5 result)')
    parser.add_argument('--run-id')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--enable-conjunction-v1', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--smoke-test', action='store_true')
    group.add_argument('--development-worker', action='store_true')
    group.add_argument('--final-worker', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke-steps', type=int)
    args = parser.parse_args(argv)
    if not args.enable_conjunction_v1:
        raise PermissionError('Explicit --enable-conjunction-v1 is required before any environment initialization')
    apply_training_launch_rules()
    global SOURCE, SOURCE_STEPS
    SOURCE = Path(args.source).resolve()
    SOURCE_STEPS = checkpoint_timesteps(SOURCE)
    if SOURCE != DEFAULT_SOURCE.resolve():
        print(f'NOTE: initializing from Stage 5 bundle {SOURCE}; the result is bit-identical to '
              'flight_v2_seed63011_run03 only if that bundle has the same weights as the supplied one.')
    if args.smoke_test:
        return smoke_test(args)
    if args.smoke_steps is not None:
        raise ValueError('--smoke-steps is only supported with --smoke-test')
    if args.development_worker or args.final_worker:
        if not all((args.run_dir, args.checkpoint, args.output)):
            raise ValueError('Worker requires run directory, checkpoint and exclusive output')
        return worker(args)
    if not args.run_id:
        raise ValueError('An exclusive --run-id is required')
    return train(args)


if __name__ == '__main__':
    main()
