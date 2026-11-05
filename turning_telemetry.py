"""Opt-in observation of policy output and the unchanged production fire gate."""

from collections import Counter
import hashlib
from types import SimpleNamespace

import numpy as np


def actor_state(actor):
    return {
        'position_m': list(actor.position),
        'velocity_m_s': list(actor.velocity),
        'hpr_deg': [actor.yaw, actor.pitch, actor.roll],
        'angular_rates_deg_s': [actor.yaw_rate, actor.pitch_rate, actor.roll_rate],
        'current_speed_m_s': actor.current_speed,
        'target_speed_m_s': actor.target_speed,
    }


def initial_state(raw, observation, simulator_seed, policy_seed):
    return {
        'environment_reset_seed': simulator_seed, 'policy_seed': policy_seed,
        'simulator_rng_sha256': hashlib.sha256(repr(raw.sim.rng.getstate()).encode()).hexdigest(),
        'simulation_dt_seconds': raw.simulation_dt,
        'player': actor_state(raw.sim.player), 'target': actor_state(raw.sim.enemy),
        'raw_observation': observation.tolist(),
        'missile_available': bool(raw._missile_available()),
        'target_destroyed': bool(raw.sim.target_destroyed),
        'game_over': bool(raw.sim.game_over),
    }


def policy_snapshot(model):
    """Read the distribution used by predict(), without inference or sampling."""
    distribution = model.policy.action_dist
    return {
        'fire_probability': float(distribution.bernoulli.probs[0, 0].detach().cpu()),
        'gaussian_means': distribution.gaussian.mean[0].detach().cpu().tolist(),
        'gaussian_stds': distribution.gaussian.stddev[0].detach().cpu().tolist(),
    }


class _BooleanProbe:
    def __init__(self, value, name, checks, inverted=False):
        self.value, self.name, self.checks, self.inverted = value, name, checks, inverted

    def __bool__(self):
        passed = not self.value if self.inverted else self.value
        self.checks.append({'prerequisite': self.name, 'satisfied': bool(passed)})
        return bool(self.value)


class _NumberProbe(float):
    def __new__(cls, value, name, checks):
        instance = super().__new__(cls, value)
        instance.name, instance.checks = name, checks
        return instance

    def _compare(self, threshold, passed):
        self.checks.append({'prerequisite': self.name, 'satisfied': bool(passed),
                            'value': float(self), 'threshold': float(threshold)})
        return passed

    def __lt__(self, other):
        return self._compare(other, float(self) < other)

    def __gt__(self, other):
        return self._compare(other, float(self) > other)

    def __le__(self, other):
        return self._compare(other, float(self) <= other)

    def __ge__(self, other):
        self.checks.append({'prerequisite': 'range_at_or_above_minimum',
                            'satisfied': float(self) >= other,
                            'value': float(self), 'threshold': float(other)})
        return float(self) >= other


def probe_gate(method, raw, current_range, current_cos_ata):
    """Execute the production predicate on inert data, never the live launcher.

    Bit 1 here asks a counterfactual eligibility question on a separate stub.
    No policy action, live missile, simulator state or RNG is changed. The
    comparison probes expose the first short-circuit failure and thresholds
    directly from that predicate rather than copying its conditional formula.
    """
    checks = []
    prerequisites = {
        'target_destroyed': bool(getattr(raw.sim, 'target_destroyed', False)),
        'game_over': bool(raw.sim.game_over),
        'missile_available': bool(raw._missile_available()),
    }
    shadow_sim = SimpleNamespace(
        target_destroyed=_BooleanProbe(prerequisites['target_destroyed'], 'target_alive', checks, True),
        game_over=_BooleanProbe(prerequisites['game_over'], 'simulation_running', checks, True),
        active_missiles=[],
    )
    shadow_sim.fire_missile = lambda: shadow_sim.active_missiles.append(None)
    shadow = SimpleNamespace(
        sim=shadow_sim, fire_range_gate=raw.fire_range_gate,
        launch_range_gate=getattr(raw, 'launch_range_gate', raw.fire_range_gate),
        _missile_available=lambda: _BooleanProbe(prerequisites['missile_available'],
                                                'missile_available', checks),
        _distance_to_target=lambda: _NumberProbe(current_range, 'range_below_gate', checks),
    )
    result = method.__func__(shadow, 1.0,
                             _NumberProbe(current_range, 'range_below_gate', checks),
                             _NumberProbe(current_cos_ata, 'cosATA_above_gate', checks))
    failed = next((check['prerequisite'] for check in checks if not check['satisfied']), None)
    return {
        **prerequisites, 'range_m': float(current_range), 'velocity_cosATA': float(current_cos_ata),
        'launch_ceiling_m': getattr(raw, 'launch_range_gate', raw.fire_range_gate),
        'eligible': bool(result[1]), 'first_failed_prerequisite': failed,
        'evaluated_prerequisites': checks,
        'eligibility_source': method.__func__.__qualname__ + ' on inert snapshot',
    }


class GateObserver:
    """Call real control setter/gate once, recording their actual inputs/results."""
    def __init__(self, raw):
        self.raw = raw
        self.record = None
        self.applied = None
        self.clipped = None
        self.originals = {}

    def __enter__(self):
        raw = self.raw
        gate, setter = raw._process_hybrid_fire, raw._set_player_controls
        for name in ('_process_hybrid_fire', '_set_player_controls'):
            self.originals[name] = (name in raw.__dict__, raw.__dict__.get(name))

        def observed_setter(action):
            result = setter(action)
            self.clipped = [float(x) for x in action]
            player = raw.sim.player
            self.applied = [float(getattr(player, name)) for name in
                            ('pitch_input', 'roll_input', 'yaw_input', 'speed_input')] + [float(action[4])]
            return result

        def observed_gate(fire_bit, current_range, current_cos_ata):
            record = probe_gate(gate, raw, current_range, current_cos_ata)
            record['player'] = actor_state(raw.sim.player)
            record['target'] = actor_state(raw.sim.enemy)
            result = gate(fire_bit, current_range, current_cos_ata)
            request, launched, rejected = map(bool, result)
            record.update(fire_request=request, launched=launched, rejected=rejected)
            record['rejection_reason'] = (
                (record['first_failed_prerequisite'] or 'launch_not_created') if rejected else None)
            self.record = record
            return result

        raw._process_hybrid_fire = observed_gate
        raw._set_player_controls = observed_setter
        return self

    def __exit__(self, *exception):
        for name, (existed, value) in self.originals.items():
            if existed:
                setattr(self.raw, name, value)
            else:
                delattr(self.raw, name)


def summarize_firing(records):
    counts = Counter()
    rejections = Counter()
    in_window = window_request = False
    probabilities = []
    for row in records:
        gate = row['gate']
        counts['total_steps'] += 1
        counts['fire_requests'] += int(gate['fire_request'])
        counts['launches'] += int(gate['launched'])
        if gate['rejected']:
            rejections[gate['rejection_reason']] += 1
        if gate['eligible']:
            counts['eligible_steps'] += 1
            counts['eligible_steps_with_request' if gate['fire_request'] else 'eligible_steps_without_request'] += 1
            probabilities.append(row['policy_distribution']['fire_probability'])
            counts['eligible_steps_probability_below_half' if probabilities[-1] < .5
                   else 'eligible_steps_probability_at_least_half'] += 1
            if not in_window:
                counts['eligible_windows'] += 1
            in_window = True
            window_request = window_request or gate['fire_request']
        elif in_window:
            counts['eligible_windows_with_request' if window_request else 'eligible_windows_without_request'] += 1
            in_window = window_request = False
    if in_window:
        counts['eligible_windows_with_request' if window_request else 'eligible_windows_without_request'] += 1
    keys = ('total_steps', 'fire_requests', 'launches', 'eligible_steps', 'eligible_steps_with_request',
            'eligible_steps_without_request', 'eligible_windows', 'eligible_windows_with_request',
            'eligible_windows_without_request')
    keys += ('eligible_steps_probability_below_half', 'eligible_steps_probability_at_least_half')
    return {**{key: counts[key] for key in keys}, 'rejections_by_reason': dict(rejections),
            'eligible_fire_probability_mean': float(np.mean(probabilities)) if probabilities else None,
            'eligible_fire_probability_max': max(probabilities) if probabilities else None}


def aggregate_firing(episodes):
    keys = ('total_steps', 'fire_requests', 'launches', 'eligible_steps', 'eligible_steps_with_request',
            'eligible_steps_without_request', 'eligible_windows', 'eligible_windows_with_request',
            'eligible_windows_without_request', 'eligible_steps_probability_below_half',
            'eligible_steps_probability_at_least_half', 'hits', 'misses', 'expiries', 'pending_missiles')
    result = {key: sum(row[key] for row in episodes) for key in keys}
    for name in ('rejections_by_reason', 'terminal_reasons'):
        counts = Counter()
        for row in episodes:
            counts.update(row[name])
        result[name] = dict(counts)
    eligible = result['eligible_steps']
    result['eligible_fire_probability_mean'] = (
        sum(row['eligible_fire_probability_mean'] * row['eligible_steps']
            for row in episodes if row['eligible_steps']) / eligible if eligible else None)
    maxima = [row['eligible_fire_probability_max'] for row in episodes if row['eligible_steps']]
    result['eligible_fire_probability_max'] = max(maxima) if maxima else None
    return result


def paired_bearing_metrics(traces, requested_common_steps, simulation_dt=.05):
    """Offline prefixes of stored traces; never advance either environment."""
    result = {}
    for mode in ('deterministic', 'sampled'):
        indices = sorted({index for policy, item_mode, index in traces if item_mode == mode})
        if not indices:
            continue
        common = min(requested_common_steps, *(len(traces[policy, mode, index])
                                               for index in indices for policy in ('baseline', 'candidate')))
        pairs = []
        for index in indices:
            duration_matched = min(len(traces[policy, mode, index]) for policy in ('baseline', 'candidate'))
            metrics = {}
            for policy in ('baseline', 'candidate'):
                errors = [abs((row['bearing_error_deg'] + 180) % 360 - 180)
                          for row in traces[policy, mode, index]]
                metrics[policy] = {
                    'duration_matched_mean_absolute_bearing_deg': float(np.mean(errors[:duration_matched])),
                    'common_horizon_mean_absolute_bearing_deg': float(np.mean(errors[:common])),
                    'duration_matched_accumulated_absolute_bearing_deg_s': float(
                        np.sum(errors[:duration_matched]) * simulation_dt),
                    'common_horizon_accumulated_absolute_bearing_deg_s': float(
                        np.sum(errors[:common]) * simulation_dt),
                }
            pairs.append({'case': index, 'duration_matched_steps': duration_matched,
                          'duration_matched_seconds': duration_matched * simulation_dt, **metrics})
        result[mode] = {'common_horizon_steps': common, 'common_horizon_seconds': common * simulation_dt,
                        'requested_common_horizon_steps': requested_common_steps,
                        'pairs': pairs, 'mean': {
                            policy: {key: float(np.mean([pair[policy][key] for pair in pairs]))
                                     for key in pairs[0][policy]}
                            for policy in ('baseline', 'candidate')}}
    return result
