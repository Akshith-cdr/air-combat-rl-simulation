"""Opt-in flight-only PPO, retaining the existing 19-observation/5-action API."""

from dataclasses import asdict
import itertools
import json
import numpy as np
import torch as th
from stable_baselines3.common.distributions import DiagGaussianDistribution

from hybrid_ppo import HybridActorCriticPolicy
from turning_experiment import TurningEnv, sample_scenario
from turning_telemetry import GateObserver, initial_state
from fighter_jet_env import CLOSURE_SCALE, FIRE_COS_ATA_GATE, W_CLOSURE, W_ALIGNMENT
from experiments.turning_improvement.fire_decision_integration_v1_20261027_01.integration import (
    FireDecisionAdapter, digest,
)

BUDGET = 1_048_576
HEADINGS = (0, 60, 120, 180, 240, 300)
FAMILIES = ('ahead', 'left', 'right', 'behind_left', 'behind_right')


def finite_tensor(value):
    if not th.isfinite(value).all():
        raise FloatingPointError('Nonfinite flight tensor/gradient')
    return value


class FixedFireGaussian(DiagGaussianDistribution):
    """Four-dimensional density times an observed-state-dependent point mass."""
    def __init__(self, means, log_std, fire):
        super().__init__(4)
        super().proba_distribution(finite_tensor(means), finite_tensor(log_std))
        self.fire = fire.detach()

    def log_prob(self, actions):
        finite_tensor(actions)
        if actions.shape != (len(self.fire), 5) or not th.equal(actions[:, 4:5], self.fire):
            raise ValueError('Rollout fire lane differs from the fixed decision for these observations')
        return super().log_prob(actions[:, :4])

    def sample(self):
        return th.cat((super().sample(), self.fire), dim=-1)

    def mode(self):
        return th.cat((super().mode(), self.fire), dim=-1)


class FixedConjunctionFlightPolicy(HybridActorCriticPolicy):
    """Same weight shapes as Candidate A; its Bernoulli head is unused/frozen."""
    def __init__(self, *args, enable_conjunction_v1=False, **kwargs):
        if enable_conjunction_v1 is not True:
            raise ValueError('Fixed conjunction requires explicit opt-in')
        self.fire_adapter = FireDecisionAdapter.create('conjunction-v1')
        super().__init__(*args, **kwargs)
        if self.observation_space.shape != (19,) or self.action_space.shape != (5,) or not self.share_features_extractor:
            raise ValueError('Expected Candidate A shared 19/5 architecture')
        np.testing.assert_array_equal(self.action_space.low, [-1, -1, -1, -1, 0])
        np.testing.assert_array_equal(self.action_space.high, [1, 1, 1, 1, 1])
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.register_hook(finite_tensor)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.fire_net.requires_grad_(False)
        self.optimizer = self.optimizer_class(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr_schedule(1), **self.optimizer_kwargs)
        self.optimizer.register_step_pre_hook(self._before_step)
        self.optimizer.register_step_post_hook(self._after_step)

    def _before_step(self, optimizer, args, kwargs):
        for name, parameter in self.named_parameters():
            if name.startswith('fire_net.') and parameter.grad is not None:
                raise RuntimeError('Frozen fire parameter received a gradient')
            if parameter.grad is not None:
                finite_tensor(parameter.grad)

    def _after_step(self, optimizer, args, kwargs):
        for parameter in self.parameters():
            finite_tensor(parameter)
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, th.Tensor):
                    finite_tensor(value)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), 'enable_conjunction_v1': True}

    def _latents(self, obs):
        finite_tensor(obs)
        if obs.ndim != 2 or obs.shape[1] != 19:
            raise ValueError('Expected a batch of normalized 19-vectors')
        return self.mlp_extractor(self.extract_features(obs))

    def _distribution(self, obs, latent_pi):
        # Detached, already normalized observations; no second normalization,
        # Torch approximation of the reviewed classifier, or Bernoulli sample.
        fire = [[float(self.fire_adapter._model.request(row))]
                for row in obs.detach().cpu().numpy()]
        self.action_dist = FixedFireGaussian(self.action_net(latent_pi), self.log_std,
                                             th.tensor(fire, dtype=obs.dtype, device=obs.device))
        return self.action_dist

    def _get_action_dist_from_latent(self, latent_pi):
        raise RuntimeError('Normalized observations are required for the fixed fire lane')

    def get_distribution(self, obs):
        latent_pi, _ = self._latents(obs)
        return self._distribution(obs, latent_pi)

    def forward(self, obs, deterministic=False):
        pi, vf = self._latents(obs)
        distribution = self._distribution(obs, pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, finite_tensor(self.value_net(vf)), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        pi, vf = self._latents(obs)
        distribution = self._distribution(obs, pi)
        return finite_tensor(self.value_net(vf)), distribution.log_prob(actions), distribution.entropy()


def transfer_initialization(source_policy, target_policy):
    """Strict weight transfer, not optimizer restoration or fine-tuning here."""
    state = source_policy.state_dict()
    target_policy.load_state_dict(state, strict=True)
    if target_policy.optimizer.state:
        raise ValueError('Initialization requires a fresh optimizer')
    if any(p.requires_grad for p in target_policy.fire_net.parameters()):
        raise ValueError('Fire tensors must remain frozen')


class MirroredCurriculum:
    def __init__(self, seed, reserved_fingerprints=()):
        self.rng = np.random.default_rng(seed)
        self.reserved = set(reserved_fingerprints)
        self.stage = 0
        self.deck = []
        self.pending = None
        self.pair_id = -1

    def next(self, steps):
        if self.pending is not None:
            result, self.pending = self.pending, None
            return result
        stage = 1 if steps < 262144 else 2 if steps < 524288 else 3
        if stage != self.stage:
            self.stage, self.deck = stage, []
        if not self.deck:
            families = FAMILIES[:1] if stage == 1 else FAMILIES[:3] if stage == 2 else FAMILIES
            self.deck = list(itertools.product(families, HEADINGS))
            self.rng.shuffle(self.deck)
        family, heading = self.deck.pop()
        scenario = sample_scenario(self.rng, (heading,), family)
        for value in (scenario, scenario.mirror()):
            fingerprint = digest(json.dumps(asdict(value), sort_keys=True, separators=(',', ':')).encode())
            if fingerprint in self.reserved:
                raise ValueError('Training scenario overlaps a reserved development/final initial state')
        self.pair_id += 1
        self.pending = scenario.mirror()
        return scenario

    def state(self):
        return dict(stage=self.stage, pair_id=self.pair_id, deck=self.deck,
                    pending=asdict(self.pending) if self.pending else None,
                    rng_state=self.rng.bit_generator.state)


def corrected_components(components, previous_range, current_range, cos_ata, dt):
    if not np.isfinite([previous_range, current_range, cos_ata, dt]).all() or dt <= 0:
        raise FloatingPointError('Invalid reward geometry/timestep')
    result = dict(components)
    closure = float(np.clip((previous_range - current_range) / (CLOSURE_SCALE * dt), -1, 1))
    alignment = max(0., closure) * max(0., float(np.clip((cos_ata - FIRE_COS_ATA_GATE) /
                                                       (1 - FIRE_COS_ATA_GATE), -1, 1)))
    delta = W_CLOSURE * (closure - result['closure']) + W_ALIGNMENT * (alignment - result['alignment'])
    result.update(closure=closure, alignment=alignment,
                  engagement=result['engagement'] + delta, total=result['total'] + delta)
    return result


class TrainingFlightEnv(TurningEnv):
    """Existing physics/gate; isolated reward correction and reset curriculum."""
    def __init__(self, config, seed, steps=0, reserved_fingerprints=(), reserved_seeds=()):
        self.training_steps = steps
        self.curriculum_seed = seed
        self.reserved_fingerprints = tuple(reserved_fingerprints)
        self.reserved_seeds = set(reserved_seeds)
        self.curriculum = MirroredCurriculum(seed, reserved_fingerprints)
        self.seed_rng = np.random.default_rng(seed)
        self._observer = None
        super().__init__(config)
        original = self.sim.sanitize_actor
        def checked_sanitize(actor):
            values = [*actor.position, *actor.velocity] + [getattr(actor, key) for key in
                ('roll', 'pitch', 'yaw', 'roll_rate', 'pitch_rate', 'yaw_rate',
                 'current_speed', 'target_speed', 'airspeed')]
            if not np.isfinite(values).all():
                raise FloatingPointError('Nonfinite state detected before simulator repair')
            return original(actor)
        self.sim.sanitize_actor = checked_sanitize
        self._observer = GateObserver(self)
        self._observer.__enter__()

    def reset(self, *, seed=None, options=None):
        explicit = (options or {}).get('turning_scenario')
        if explicit is None:
            if seed is not None:
                self.curriculum = MirroredCurriculum(seed, self.reserved_fingerprints)
                self.seed_rng = np.random.default_rng(seed)
            scenario = self.curriculum.next(self.training_steps)
            seed = seed if seed is not None else int(self.seed_rng.integers(0, 2**32))
            if seed in self.reserved_seeds:
                raise ValueError('Training reset seed overlaps development/final reservation')
            options = {'turning_scenario': asdict(scenario)}
        obs, info = super().reset(seed=seed, options=options)
        if not np.isfinite(obs).all():
            raise FloatingPointError('Nonfinite reset observation')
        self.reset_tracking()
        self.last_initial_state = initial_state(self, obs, seed, 0)
        return obs, info

    def reset_tracking(self):
        self.episode_metrics = dict(invalid_actions=0, gate_violations=0, numerical_repairs=0,
            target_destroyed=False, steps_to_hit=None, launches=0, fire_requests=0, rejected_requests=0,
            missile_hits=0, survived_proxy=True, mean_abs_bank_deg=0., longest_bank_s=0.,
            opposing_fraction=0., path_length_m=0., accumulated_bearing_deg_s=0.)
        self._bank_sum = self._opposed_steps = self._bank_run = 0
        self._bank_sign = 0
        self._previous_position = np.array(self.sim.player.position, dtype=float)

    def _compute_reward_components(self, previous_range, previous_cos, obs, dt, weapon_reward=0.):
        base = super()._compute_reward_components(previous_range, previous_cos, obs, dt, weapon_reward)
        return corrected_components(base, previous_range, float(obs[10]), float(obs[11]), dt)

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (5,) or not np.isfinite(action).all() or action[4] not in (0., 1.):
            self.episode_metrics['invalid_actions'] += 1
            raise FloatingPointError('Invalid flight action; no simulator step performed')
        self._observer.record = None
        obs, reward, terminated, truncated, info = super().step(action)
        self.training_steps += 1
        if not np.isfinite(obs).all() or not np.isfinite(reward):
            raise FloatingPointError('Nonfinite flight observation/reward')
        gate = self._observer.record
        if gate is None or (gate['launched'] and not (gate['fire_request'] and gate['eligible'])):
            self.episode_metrics['gate_violations'] += 1
            raise RuntimeError('Production gate observation/authorization failure')
        removed = float(info['weapon']['invalid_fire_attempt'])
        reward += removed
        self._episode_reward += removed
        parts = info['reward_breakdown']
        parts['weapon'] += removed
        parts['base_total'] += removed
        parts['total'] += removed
        parts['invalid_fire_penalty_removed'] = removed
        if 'episode' in info:
            info['episode']['r'] = self._episode_reward
        self.episode_metrics['target_destroyed'] = bool(self.sim.target_destroyed)
        metrics = self.episode_metrics
        for key, source in (('launches', 'valid_fire_attempt'), ('fire_requests', 'fire_request'),
                            ('rejected_requests', 'invalid_fire_attempt'), ('missile_hits', 'missile_hit_this_step')):
            metrics[key] += int(info['weapon'][source])
        player = self.sim.player
        self._bank_sum += abs(player.roll)
        sign = int(np.sign(player.roll)) if abs(player.roll) > 45 else 0
        self._bank_run = self._bank_run + 1 if sign and sign == self._bank_sign else int(bool(sign))
        self._bank_sign = sign
        self._opposed_steps += int(abs(player.roll) > 15 and player.roll * player.yaw_input > 0 and abs(player.yaw_input) > .05)
        metrics['mean_abs_bank_deg'] = self._bank_sum / self._episode_step
        metrics['longest_bank_s'] = max(metrics['longest_bank_s'], self._bank_run * self.simulation_dt)
        metrics['opposing_fraction'] = self._opposed_steps / self._episode_step
        position = np.array(player.position, dtype=float)
        metrics['path_length_m'] += float(np.linalg.norm(position - self._previous_position))
        self._previous_position = position
        metrics['accumulated_bearing_deg_s'] += abs((float(obs[15]) + 180) % 360 - 180) * self.simulation_dt
        metrics['survived_proxy'] = not bool(player.crashed)
        if info['weapon']['missile_hit_this_step'] and self.episode_metrics['steps_to_hit'] is None:
            self.episode_metrics['steps_to_hit'] = self._episode_step
        info['flight_metrics'] = dict(self.episode_metrics)
        info['flight_curriculum'] = dict(stage=self.curriculum.stage, pair_id=self.curriculum.pair_id,
                                         family=self.current_scenario.family,
                                         heading=self.current_scenario.player_heading)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._observer is not None:
            self._observer.__exit__()
            self._observer = None
        super().close()
