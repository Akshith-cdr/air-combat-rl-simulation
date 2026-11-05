"""Train/evaluate a separate camera-only PPO using original-simulator flights."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .simulation import FrozenFlightController, ROOT, collect_trace, digest
from .task import CameraConfig, CameraObservationEnv


PROTECTED = ('fighter_jet_env.py', 'updated_chase_jet_sim_v3.py', 'hybrid_ppo.py',
             'experimental_flight_ppo_v1.py', 'projectile_motion.py', 'weapon_config.py',
             'radar.py', 'missile_vfx.py', 'aim-9_missile.glb', 'enemy.glb', 'ba_f-15ex_eagle_ii.glb')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def wilson(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    p, denominator = successes/count, 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0.0, center-half), min(1.0, center+half)]


def bootstrap_mean(values, seed=90423):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), ci95=list(map(float, np.quantile(draws, [.025, .975]))),
                episodes=len(values))


def evaluate_one(trace, model, config, mode='learned'):
    env = CameraObservationEnv([trace], config)
    observation, _ = env.reset(seed=trace['seed'])
    while True:
        if mode == 'learned':
            action = int(model.predict(observation, deterministic=True)[0])
        elif mode == 'periodic':
            # Same exposure budget, evenly spaced in time, no geometry rule.
            interval = max(1, math.ceil(config.horizon/config.capture_budget))
            action = int(env.state.step_index % interval == 0)
        elif mode == 'wait':
            action = 0
        else:
            raise ValueError('Unknown camera baseline')
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return dict(seed=trace['seed'], **info['camera_metrics'],
                flight_path_m=trace['flight_path_m'],
                minimum_sampled_altitude_m=trace['minimum_sampled_altitude_m'])


def summarize(rows):
    count = len(rows)
    acquired = sum(r['acquired'] for r in rows)
    return dict(episodes=count, acquired_episodes=acquired,
                acquisition_rate=acquired/count, acquisition_rate_ci95=wilson(acquired, count),
                no_visible_opportunity_episodes=sum(r['visible_steps'] == 0 for r in rows),
                visible_tracking_coverage=bootstrap_mean([r['visible_tracking_coverage'] for r in rows
                                                         if r['visible_tracking_coverage'] is not None]),
                overall_track_availability=bootstrap_mean([r['overall_track_availability'] for r in rows]),
                captures=bootstrap_mean([r['captures'] for r in rows]),
                unusable_requests=bootstrap_mean([r['unusable_requests'] for r in rows]),
                first_acquisition_s_conditional=bootstrap_mean([r['first_acquisition_s'] for r in rows
                                                               if r['first_acquisition_s'] is not None]),
                mean_capture_quality_conditional=bootstrap_mean([r['mean_capture_quality'] for r in rows
                                                                  if r['mean_capture_quality'] is not None]),
                episode_return=bootstrap_mean([r['episode_return'] for r in rows]),
                termination_reasons=dict(Counter(r['termination_reason'] for r in rows)))


def run(output, timesteps=65536, train_count=64, validation_count=16, test_count=64):
    if min(timesteps, train_count, validation_count, test_count) <= 0:
        raise ValueError('Positive training steps and split sizes required')
    output = Path(output).resolve()
    # Exclusive run directories prevent accidental replacement of any checkpoint.
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = CameraConfig()
    seeds = dict(training=list(range(110000, 110000+train_count)),
                 validation=list(range(210000, 210000+validation_count)),
                 heldout=list(range(310000, 310000+test_count)))
    if len(set(sum(seeds.values(), []))) != sum(map(len, seeds.values())):
        raise ValueError('Camera splits must have disjoint seeds')
    protected = {name: digest(ROOT/name) for name in PROTECTED}
    controller = FrozenFlightController()
    manifest = dict(schema='camera-observation-v1', camera_config=asdict(config), splits=seeds,
                    flight_checkpoint_sha256=controller.hashes, protected_sha256=protected,
                    training=dict(algorithm='PPO', timesteps=timesteps, seed=90421, device='cpu',
                                  learning_rate=0.0003, n_steps=1024, batch_size=128, n_epochs=10,
                                  gamma=0.98, ent_coef=0.005, net_arch=[32, 32]),
                    confidence_intervals='95% Wilson for acquisition; seed-cluster bootstrap 5000 draws for means',
                    task='camera observation scheduling; no flight or weapon actions',
                    source_sha256={p.relative_to(ROOT).as_posix(): digest(p)
                                   for p in Path(__file__).parent.glob('*.py')})
    write_json(output/'manifest.json', manifest)
    traces = {}
    started = time.monotonic()
    for split, split_seeds in seeds.items():
        traces[split] = []
        directory = output/'traces'/split
        directory.mkdir(parents=True)
        for index, seed in enumerate(split_seeds):
            trace = collect_trace(controller, seed, config)
            traces[split].append(trace)
            np.savez_compressed(directory/f'{seed}.npz', frames=trace['frames'],
                                positions=trace['positions'], controls=trace['controls'])
            if index % 8 == 0:
                print(f'{split}: {index+1}/{len(split_seeds)} original flights recorded', flush=True)
    controller.verify()
    print('Training camera-only PPO; frozen-flight traces reused exactly.', flush=True)
    vector = DummyVecEnv([lambda: CameraObservationEnv(traces['training'], config)])
    try:
        model = PPO('MlpPolicy', vector, seed=90421, device='cpu', learning_rate=.0003,
                    n_steps=1024, batch_size=128, n_epochs=10, gamma=.98, ent_coef=.005,
                    policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])))
        model.learn(total_timesteps=timesteps)
        model.save(str(output/'camera_policy.zip'))
    finally:
        vector.close()
    validation = [evaluate_one(trace, model, config) for trace in traces['validation']]
    # Fixed training budget, no selection or retuning using held-out episodes.
    results = {mode: [evaluate_one(trace, model, config, mode) for trace in traces['heldout']]
               for mode in ('learned', 'periodic', 'wait')}
    paired = {}
    for baseline in ('periodic', 'wait'):
        values = [a['overall_track_availability']-b['overall_track_availability']
                  for a, b in zip(results['learned'], results[baseline])]
        visible_values = [a['visible_tracking_coverage']-b['visible_tracking_coverage']
                          for a, b in zip(results['learned'], results[baseline])
                          if a['visible_tracking_coverage'] is not None]
        paired[baseline] = dict(overall_availability_delta=bootstrap_mean(values),
                                visible_coverage_delta=bootstrap_mean(visible_values))
    final_protected = {name: digest(ROOT/name) for name in PROTECTED}
    if protected != final_protected:
        raise RuntimeError('Protected original simulation code or assets changed')
    controller.verify()
    # Store all per-seed evidence; CI resampling units are whole flight seeds.
    write_json(output/'episodes.json', results)
    write_json(output/'validation_episodes.json', validation)
    trace_inventory = {p.relative_to(output).as_posix(): digest(p) for p in (output/'traces').rglob('*.npz')}
    write_json(output/'trace_hashes.json', trace_inventory)
    report = dict(status='trained_and_evaluated', total_timesteps=int(model.num_timesteps),
                  elapsed_seconds=time.monotonic()-started, validation=summarize(validation),
                  heldout={mode: summarize(rows) for mode, rows in results.items()},
                  paired_learned_minus_baseline=paired, original_artifacts_unchanged=True,
                  camera_policy_sha256=digest(output/'camera_policy.zip'))
    write_json(output/'results.json', report)
    print(json.dumps(report['heldout']['learned'], indent=2), flush=True)
    print(f'Camera run saved: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timesteps', type=int, default=65536)
    parser.add_argument('--train-count', type=int, default=64)
    parser.add_argument('--validation-count', type=int, default=16)
    parser.add_argument('--test-count', type=int, default=64)
    args = parser.parse_args()
    run(args.output, args.timesteps, args.train_count, args.validation_count, args.test_count)


if __name__ == '__main__':
    main()
