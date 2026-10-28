import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fighter_jet_env import FighterJetEnv


MODEL_PATH = Path("models/ppo_ucav_100k_19d_5a_weapon_fire_reward.zip")
EPISODES = 20
# Evaluation-only guard: the environment intentionally has no MDP truncation
# limit, so cap reporting episodes without changing environment dynamics.
MAX_EVALUATION_STEPS = 3000
INTERCEPTION_THRESHOLDS = (1000.0, 500.0, 250.0, 100.0)
FIRE_SIGNAL_DISTANCE_BANDS = (2000.0, 1500.0, 1000.0, 750.0, 500.0, 250.0, 100.0)


# Future evaluation-only randomized-enemy challenge interface. These immutable
# records are intentionally not consumed by the legacy evaluator yet: the
# environment does not support applying challenge scenarios at reset time.
@dataclass(frozen=True)
class EnemyChallengeScenario:
    scenario_id: str
    seed: int
    enemy_initial_position: tuple[float, float, float]
    enemy_initial_speed: float
    enemy_initial_heading: float
    enemy_initial_altitude: float


@dataclass(frozen=True)
class RandomizedEnemyChallengeConfig:
    enabled: bool = False
    scenario_manifest: tuple[EnemyChallengeScenario, ...] = ()


# Frozen Phase 1 manifest. Positions are absolute world coordinates; the UCAV
# remains fixed at (0, 0, 1000), so these cover lateral offsets of +/-1200 m,
# forward separations of 2200..4800 m, and altitudes of 700..2200 m. Explicit
# values, rather than seed-derived values, keep the challenge stable if random
# number generation or draw order changes later.
FROZEN_ENEMY_CHALLENGE_MANIFEST = (
    EnemyChallengeScenario("REC-001", 41001, (-1200.0, 2200.0, 700.0), 140.0, 0.0, 700.0),
    EnemyChallengeScenario("REC-002", 41002, (-800.0, 2600.0, 900.0), 155.0, 30.0, 900.0),
    EnemyChallengeScenario("REC-003", 41003, (-400.0, 3000.0, 1100.0), 170.0, 60.0, 1100.0),
    EnemyChallengeScenario("REC-004", 41004, (0.0, 3400.0, 1300.0), 185.0, 90.0, 1300.0),
    EnemyChallengeScenario("REC-005", 41005, (400.0, 3800.0, 1500.0), 200.0, 120.0, 1500.0),
    EnemyChallengeScenario("REC-006", 41006, (800.0, 4200.0, 1700.0), 215.0, 150.0, 1700.0),
    EnemyChallengeScenario("REC-007", 41007, (1200.0, 4600.0, 1900.0), 230.0, 180.0, 1900.0),
    EnemyChallengeScenario("REC-008", 41008, (-1000.0, 4800.0, 2200.0), 240.0, 210.0, 2200.0),
    EnemyChallengeScenario("REC-009", 41009, (-600.0, 4400.0, 2000.0), 225.0, 240.0, 2000.0),
    EnemyChallengeScenario("REC-010", 41010, (-200.0, 4000.0, 1800.0), 210.0, 270.0, 1800.0),
    EnemyChallengeScenario("REC-011", 41011, (200.0, 3600.0, 1600.0), 195.0, 300.0, 1600.0),
    EnemyChallengeScenario("REC-012", 41012, (600.0, 3200.0, 1400.0), 180.0, 330.0, 1400.0),
    EnemyChallengeScenario("REC-013", 41013, (1000.0, 2800.0, 1200.0), 165.0, 15.0, 1200.0),
    EnemyChallengeScenario("REC-014", 41014, (-1100.0, 2400.0, 1000.0), 150.0, 45.0, 1000.0),
    EnemyChallengeScenario("REC-015", 41015, (-700.0, 3300.0, 800.0), 175.0, 75.0, 800.0),
    EnemyChallengeScenario("REC-016", 41016, (-300.0, 3700.0, 1050.0), 205.0, 105.0, 1050.0),
    EnemyChallengeScenario("REC-017", 41017, (100.0, 4100.0, 1350.0), 235.0, 135.0, 1350.0),
    EnemyChallengeScenario("REC-018", 41018, (500.0, 4500.0, 1650.0), 220.0, 165.0, 1650.0),
    EnemyChallengeScenario("REC-019", 41019, (900.0, 4700.0, 1950.0), 190.0, 195.0, 1950.0),
    EnemyChallengeScenario("REC-020", 41020, (1100.0, 4300.0, 2150.0), 160.0, 225.0, 2150.0),
    EnemyChallengeScenario("REC-021", 41021, (-900.0, 3900.0, 1850.0), 145.0, 255.0, 1850.0),
    EnemyChallengeScenario("REC-022", 41022, (-500.0, 3500.0, 1550.0), 230.0, 285.0, 1550.0),
    EnemyChallengeScenario("REC-023", 41023, (300.0, 3100.0, 1250.0), 200.0, 315.0, 1250.0),
    EnemyChallengeScenario("REC-024", 41024, (700.0, 2700.0, 950.0), 170.0, 345.0, 950.0),
)

RANDOMIZED_ENEMY_CHALLENGE = RandomizedEnemyChallengeConfig(
    enabled=False,
    scenario_manifest=FROZEN_ENEMY_CHALLENGE_MANIFEST,
)


def normalization_path(model_path):
    return model_path.with_name(f"{model_path.stem}_vecnormalize.pkl")


def make_evaluation_env(model_path):
    stats_path = normalization_path(model_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization statistics not found: {stats_path}")

    env = VecNormalize.load(str(stats_path), DummyVecEnv([FighterJetEnv]))
    env.training = False
    env.norm_reward = False
    return env


def _firing_geometry_record(raw_observation, step, time_s):
    if (
        float(raw_observation[10]) < 100.0
        and float(raw_observation[11]) > 0.95
    ):
        return {
            "step": int(step),
            "time_s": float(time_s),
            "distance": float(raw_observation[10]),
            "cos_ata": float(raw_observation[11]),
        }
    return None


def _invalid_fire_reasons(context):
    reasons = []
    if not context["target_alive"]:
        reasons.append("target_destroyed")
    if context["game_over_at_request"]:
        reasons.append("simulation_game_over")
    if not context["missile_available"]:
        reasons.append("missile_unavailable")
    if context["distance"] >= 100.0:
        reasons.append("range_not_below_100m")
    if context["cos_ata"] <= 0.95:
        reasons.append("cos_ata_not_above_0.95")
    return reasons


def _interception_state(raw_observation, action_values, step, time_s):
    return {
        "step": int(step),
        "time_s": float(time_s),
        "distance": float(raw_observation[10]),
        "cos_ata": float(raw_observation[11]),
        "d_dot": float(raw_observation[14]),
        "airspeed": float(raw_observation[0]),
        "speed_action": float(action_values[3]),
    }


def _update_interception_diagnostic(
    minimum_state, threshold_states, raw_observation, action_values, step, time_s
):
    state = _interception_state(raw_observation, action_values, step, time_s)
    if minimum_state is None or state["distance"] < minimum_state["distance"]:
        minimum_state = state
    for threshold in INTERCEPTION_THRESHOLDS:
        if state["distance"] < threshold:
            closest = threshold_states[threshold]
            if closest is None or state["distance"] < closest["distance"]:
                threshold_states[threshold] = state
    return minimum_state


def _fire_signal_statistics(samples):
    if not samples:
        return None
    signals = np.asarray([sample["fire_signal"] for sample in samples], dtype=np.float64)
    return {
        "steps": len(samples),
        "mean": float(np.mean(signals)),
        "maximum": float(np.max(signals)),
        "minimum": float(np.min(signals)),
        "positive_percent": float(100.0 * np.mean(signals > 0.0)),
        "above_half_percent": float(100.0 * np.mean(signals > 0.5)),
    }


def _maximum_fire_signal(samples):
    return None if not samples else float(max(sample["fire_signal"] for sample in samples))


def _fire_action_distribution(policy, observation):
    """Return the un-clipped PPO Gaussian statistics for fire action index 4."""
    observation_tensor, _ = policy.policy.obs_to_tensor(observation)
    with th.no_grad():
        distribution = policy.policy.get_distribution(observation_tensor).distribution
        fire_mean = float(distribution.mean[..., 4].detach().cpu().item())
        fire_std = float(distribution.stddev[..., 4].detach().cpu().item())

    # For X ~ Normal(mean, std), P(X > 0) = Phi(mean / std).
    fire_probability = 0.5 * (
        1.0 + math.erf(fire_mean / (fire_std * math.sqrt(2.0)))
    )
    return {
        "fire_mean": fire_mean,
        "fire_std": fire_std,
        "fire_probability_positive": float(fire_probability),
    }


def _geometry_duration_result(
    episode_length,
    simulation_time_s,
    valid_steps,
    first_entry,
    last_exit,
    interval_count,
    completed_interval_lengths,
    current_interval_steps,
    minimum_distance,
    maximum_cos_ata,
):
    """Finalize passive valid-geometry duration counters for one episode."""
    interval_lengths = list(completed_interval_lengths)
    final_exit = last_exit
    if current_interval_steps > 0:
        interval_lengths.append(current_interval_steps)
        final_exit = {
            "step": int(episode_length),
            "time_s": float(simulation_time_s),
        }
    return {
        "valid_steps": int(valid_steps),
        "valid_percentage": (
            100.0 * float(valid_steps) / float(episode_length)
            if episode_length > 0
            else 0.0
        ),
        "first_entry": first_entry,
        "last_exit": final_exit,
        "interval_count": int(interval_count),
        "longest_interval_steps": max(interval_lengths, default=0),
        "minimum_distance": minimum_distance,
        "maximum_cos_ata": maximum_cos_ata,
    }


def _validate_challenge_config(config):
    if not config.enabled:
        raise ValueError("Randomized enemy challenge configuration is not enabled")
    if not 20 <= len(config.scenario_manifest) <= 50:
        raise ValueError("Challenge manifest must contain between 20 and 50 scenarios")

    scenario_ids = [scenario.scenario_id for scenario in config.scenario_manifest]
    seeds = [scenario.seed for scenario in config.scenario_manifest]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Challenge scenario IDs must be unique")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Challenge scenario seeds must be unique")

    for scenario in config.scenario_manifest:
        x, y, z = scenario.enemy_initial_position
        if not -1200.0 <= x <= 1200.0 or not 2200.0 <= y <= 4800.0:
            raise ValueError(f"{scenario.scenario_id}: initial position is outside challenge bounds")
        if z != scenario.enemy_initial_altitude:
            raise ValueError(f"{scenario.scenario_id}: position z and altitude differ")
        if not 700.0 <= scenario.enemy_initial_altitude <= 2200.0:
            raise ValueError(f"{scenario.scenario_id}: altitude is outside challenge bounds")
        if not 140.0 <= scenario.enemy_initial_speed <= 240.0:
            raise ValueError(f"{scenario.scenario_id}: speed is outside challenge bounds")
        if not 0.0 <= scenario.enemy_initial_heading < 360.0:
            raise ValueError(f"{scenario.scenario_id}: heading must be in [0, 360)")


def _challenge_reset_options(scenario):
    return {
        "enemy_challenge_scenario": {
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "enemy_initial_position": scenario.enemy_initial_position,
            "enemy_initial_speed": scenario.enemy_initial_speed,
            "enemy_initial_heading": scenario.enemy_initial_heading,
            "enemy_initial_altitude": scenario.enemy_initial_altitude,
        }
    }


def _print_challenge_initial_conditions(episode_number, scenario):
    print(
        f"Challenge episode {episode_number}: scenario_id={scenario.scenario_id}, "
        f"seed={scenario.seed}, position={scenario.enemy_initial_position}, "
        f"speed={scenario.enemy_initial_speed:.1f} m/s, "
        f"heading={scenario.enemy_initial_heading:.1f} deg, "
        f"altitude={scenario.enemy_initial_altitude:.1f} m"
    )


def _attach_challenge_result(result, scenario):
    samples = result["fire_signal_samples"]
    maximum_cos_ata = max(
        [result["cos_ata"], *(sample["cos_ata"] for sample in samples)]
    )
    result.update(
        {
            "scenario_id": scenario.scenario_id,
            "scenario_seed": scenario.seed,
            "enemy_initial_position": scenario.enemy_initial_position,
            "enemy_initial_speed": scenario.enemy_initial_speed,
            "enemy_initial_heading": scenario.enemy_initial_heading,
            "enemy_initial_altitude": scenario.enemy_initial_altitude,
            "minimum_range": result["minimum_interception_state"]["distance"],
            "maximum_cos_ata": maximum_cos_ata,
            "timeout": result["terminal_reason"] == "evaluation_step_limit",
        }
    )


def evaluate_policy(
    env, policy, seeds, deterministic, stochastic_fire_only=False,
    collect_fire_distribution=False, challenge_config=None,
    print_challenge_scenarios=False,
):
    results = []
    challenge_manifest = ()
    if challenge_config is not None and challenge_config.enabled:
        _validate_challenge_config(challenge_config)
        challenge_manifest = challenge_config.scenario_manifest
        seeds = tuple(seeds)
        if len(seeds) != len(challenge_manifest):
            raise ValueError("Challenge seeds and scenario manifest must have equal lengths")

    for episode_number, seed in enumerate(seeds, start=1):
        scenario = challenge_manifest[episode_number - 1] if challenge_manifest else None
        if scenario is not None:
            if seed != scenario.seed:
                raise ValueError(
                    f"{scenario.scenario_id}: evaluation seed does not match manifest seed"
                )
            env.set_options(_challenge_reset_options(scenario))
            if print_challenge_scenarios:
                _print_challenge_initial_conditions(episode_number, scenario)
        env.seed(seed)
        observation = env.reset()
        total_reward = 0.0
        episode_length = 0
        fire_attempts = 0
        valid_fire_attempts = 0
        invalid_fire_attempts = 0
        missile_hits = 0
        fire_events = []
        first_firing_geometry = None
        simulation_time_s = 0.0
        raw_env = env.venv.envs[0].unwrapped
        minimum_interception_state = None
        threshold_closest_states = {threshold: None for threshold in INTERCEPTION_THRESHOLDS}
        last_action_values = None
        fire_signal_samples = []
        fire_distribution_samples = []
        geometry_valid_steps = 0
        geometry_first_entry = None
        geometry_last_exit = None
        geometry_interval_count = 0
        geometry_completed_interval_lengths = []
        geometry_current_interval_steps = 0
        geometry_minimum_distance = None
        geometry_maximum_cos_ata = None

        while episode_length < MAX_EVALUATION_STEPS:
            raw_observation = env.get_original_obs()[0].copy()
            current_distance = float(raw_observation[10])
            current_cos_ata = float(raw_observation[11])
            valid_geometry = current_distance < 100.0 and current_cos_ata > 0.95
            if valid_geometry:
                geometry_valid_steps += 1
                if geometry_current_interval_steps == 0:
                    geometry_interval_count += 1
                    if geometry_first_entry is None:
                        geometry_first_entry = {
                            "step": int(episode_length),
                            "time_s": float(simulation_time_s),
                        }
                geometry_current_interval_steps += 1
                geometry_minimum_distance = (
                    current_distance
                    if geometry_minimum_distance is None
                    else min(geometry_minimum_distance, current_distance)
                )
                geometry_maximum_cos_ata = (
                    current_cos_ata
                    if geometry_maximum_cos_ata is None
                    else max(geometry_maximum_cos_ata, current_cos_ata)
                )
            elif geometry_current_interval_steps > 0:
                geometry_completed_interval_lengths.append(
                    geometry_current_interval_steps
                )
                geometry_current_interval_steps = 0
                geometry_last_exit = {
                    "step": int(episode_length),
                    "time_s": float(simulation_time_s),
                }
            if first_firing_geometry is None:
                first_firing_geometry = _firing_geometry_record(
                    raw_observation, episode_length, simulation_time_s
                )
            if (
                collect_fire_distribution
                and float(raw_observation[10]) < 100.0
                and float(raw_observation[11]) > 0.95
                and not bool(getattr(raw_env.sim, "target_destroyed", False))
                and not bool(raw_env.sim.game_over)
            ):
                fire_distribution_samples.append(
                    {
                        "step": episode_length,
                        "time_s": simulation_time_s,
                        "distance": float(raw_observation[10]),
                        "cos_ata": float(raw_observation[11]),
                        **_fire_action_distribution(policy, observation),
                    }
                )
            if policy is None:
                action = np.asarray([env.action_space.sample()], dtype=np.float32)
            elif stochastic_fire_only:
                deterministic_action, _ = policy.predict(
                    observation, deterministic=True
                )
                stochastic_action, _ = policy.predict(
                    observation, deterministic=False
                )
                action = np.asarray(deterministic_action, dtype=np.float32).copy()
                action[..., 4] = np.asarray(stochastic_action)[..., 4]
            else:
                action, _ = policy.predict(observation, deterministic=deterministic)

            raw_action_values = np.asarray(action).reshape(-1)
            action_values = np.asarray(action, dtype=np.float32).reshape(-1)
            fire_signal_samples.append(
                {
                    "step": episode_length,
                    "time_s": simulation_time_s,
                    "distance": float(raw_observation[10]),
                    "cos_ata": float(raw_observation[11]),
                    "d_dot": float(raw_observation[14]),
                    "fire_signal": float(raw_action_values[4]),
                }
            )
            minimum_interception_state = _update_interception_diagnostic(
                minimum_interception_state,
                threshold_closest_states,
                raw_observation,
                action_values,
                episode_length,
                simulation_time_s,
            )
            last_action_values = action_values.copy()
            fire_context = {
                "step": episode_length,
                "time_s": simulation_time_s,
                "distance": float(raw_observation[10]),
                "cos_ata": float(raw_observation[11]),
                "fire_signal": float(action_values[4]),
                "target_alive": not bool(getattr(raw_env.sim, "target_destroyed", False)),
                "missile_available": bool(raw_env._missile_available()),
                "game_over_at_request": bool(raw_env.sim.game_over),
            }
            observation, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            episode_length += 1
            weapon_info = info[0].get("weapon", {})
            fire_attempts += int(bool(weapon_info.get("fire_request", False)))
            valid_fire_attempts += int(bool(weapon_info.get("valid_fire_attempt", False)))
            invalid_fire_attempts += int(bool(weapon_info.get("invalid_fire_attempt", False)))
            missile_hits += int(bool(weapon_info.get("missile_hit_this_step", False)))
            simulation_time_s += float(info[0].get("dt", 0.0))

            if bool(weapon_info.get("fire_request", False)):
                invalid_reasons = _invalid_fire_reasons(fire_context)
                invalid = bool(weapon_info.get("invalid_fire_attempt", False))
                fire_events.append(
                    {
                        **fire_context,
                        "valid": bool(weapon_info.get("valid_fire_attempt", False)),
                        "invalid": invalid,
                        "missile_hit": bool(weapon_info.get("missile_hit_this_step", False)),
                        "done_after_step": bool(done[0]),
                        "invalid_reasons": (
                            invalid_reasons
                            if invalid_reasons
                            else (["launch_failed_after_validation"] if invalid else [])
                        ),
                    }
                )

            if bool(done[0]):
                terminal_reason = info[0].get("terminal_reason", "")
                terminal_observation = info[0].get("terminal_observation")
                if terminal_observation is not None:
                    original_observation = env.unnormalize_obs(terminal_observation)
                else:
                    original_observation = env.get_original_obs()[0]
                if first_firing_geometry is None:
                    first_firing_geometry = _firing_geometry_record(
                        original_observation, episode_length, simulation_time_s
                    )
                minimum_interception_state = _update_interception_diagnostic(
                    minimum_interception_state,
                    threshold_closest_states,
                    original_observation,
                    last_action_values,
                    episode_length,
                    simulation_time_s,
                )
                successful = terminal_reason == "missile_hit"
                boundary_failure = "CHASE BOUNDARY" in terminal_reason
                if successful:
                    result = "Success"
                elif boundary_failure:
                    result = "Boundary"
                else:
                    result = "Other/Timeout"

                geometry_duration = _geometry_duration_result(
                    episode_length,
                    simulation_time_s,
                    geometry_valid_steps,
                    geometry_first_entry,
                    geometry_last_exit,
                    geometry_interval_count,
                    geometry_completed_interval_lengths,
                    geometry_current_interval_steps,
                    geometry_minimum_distance,
                    geometry_maximum_cos_ata,
                )

                results.append(
                    {
                        "episode": episode_number,
                        "reward": total_reward,
                        "length": episode_length,
                        "result": result,
                        "success": successful,
                        "boundary": boundary_failure,
                        "distance": float(original_observation[10]),
                        "cos_ata": float(original_observation[11]),
                        "d_dot": float(original_observation[14]),
                        "terminal_reason": terminal_reason,
                        "fire_attempts": fire_attempts,
                        "valid_fire_attempts": valid_fire_attempts,
                        "invalid_fire_attempts": invalid_fire_attempts,
                        "missile_hits": missile_hits,
                        "fire_events": fire_events,
                        "entered_firing_geometry": first_firing_geometry is not None,
                        "first_firing_geometry": first_firing_geometry,
                        "minimum_interception_state": minimum_interception_state,
                        "threshold_closest_states": threshold_closest_states,
                        "fire_signal_samples": fire_signal_samples,
                        "fire_distribution_samples": fire_distribution_samples,
                        "geometry_duration": geometry_duration,
                    }
                )
                if scenario is not None:
                    _attach_challenge_result(results[-1], scenario)
                break

        else:
            original_observation = env.get_original_obs()[0]
            if first_firing_geometry is None:
                first_firing_geometry = _firing_geometry_record(
                    original_observation, episode_length, simulation_time_s
                )
            minimum_interception_state = _update_interception_diagnostic(
                minimum_interception_state,
                threshold_closest_states,
                original_observation,
                last_action_values,
                episode_length,
                simulation_time_s,
            )
            successful = False
            terminal_reason = "evaluation_step_limit"
            geometry_duration = _geometry_duration_result(
                episode_length,
                simulation_time_s,
                geometry_valid_steps,
                geometry_first_entry,
                geometry_last_exit,
                geometry_interval_count,
                geometry_completed_interval_lengths,
                geometry_current_interval_steps,
                geometry_minimum_distance,
                geometry_maximum_cos_ata,
            )
            results.append(
                {
                    "episode": episode_number,
                    "reward": total_reward,
                    "length": episode_length,
                    "result": "Success" if successful else "Other/Timeout",
                    "success": successful,
                    "boundary": False,
                    "distance": float(original_observation[10]),
                    "cos_ata": float(original_observation[11]),
                    "d_dot": float(original_observation[14]),
                    "terminal_reason": terminal_reason,
                    "fire_attempts": fire_attempts,
                    "valid_fire_attempts": valid_fire_attempts,
                    "invalid_fire_attempts": invalid_fire_attempts,
                    "missile_hits": missile_hits,
                    "fire_events": fire_events,
                    "entered_firing_geometry": first_firing_geometry is not None,
                    "first_firing_geometry": first_firing_geometry,
                    "minimum_interception_state": minimum_interception_state,
                    "threshold_closest_states": threshold_closest_states,
                    "fire_signal_samples": fire_signal_samples,
                    "fire_distribution_samples": fire_distribution_samples,
                    "geometry_duration": geometry_duration,
                }
            )
            if scenario is not None:
                _attach_challenge_result(results[-1], scenario)

    return results


def print_episode_table(title, results):
    print(f"\n{title}")
    print(
        "Episode | Reward | Length | Result | Final Distance | Final cos(ATA) | "
        "Final d_dot | Fire | Valid | Invalid | Hits"
    )
    for result in results:
        print(
            f"{result['episode']} | {result['reward']:.3f} | {result['length']} | "
            f"{result['result']} | {result['distance']:.3f} | "
            f"{result['cos_ata']:.6f} | {result['d_dot']:.3f} | "
            f"{result['fire_attempts']} | {result['valid_fire_attempts']} | "
            f"{result['invalid_fire_attempts']} | {result['missile_hits']}"
        )


def print_challenge_episode_table(title, results):
    print(f"\n{title}")
    print(
        "Scenario | Success | Hits | Valid | Invalid | Boundary | Timeout | "
        "Minimum range | Maximum cos(ATA) | Length | Final range | "
        "Final cos(ATA) | Reward"
    )
    for result in results:
        print(
            f"{result['scenario_id']} | {result['success']} | "
            f"{result['missile_hits']} | {result['valid_fire_attempts']} | "
            f"{result['invalid_fire_attempts']} | {result['boundary']} | "
            f"{result['timeout']} | {result['minimum_range']:.3f} | "
            f"{result['maximum_cos_ata']:.6f} | {result['length']} | "
            f"{result['distance']:.3f} | {result['cos_ata']:.6f} | "
            f"{result['reward']:.3f}"
        )
    print(
        "Challenge totals: "
        f"successes={sum(result['success'] for result in results)}/{len(results)}, "
        f"missile_hits={sum(result['missile_hits'] for result in results)}, "
        f"valid_fire_attempts={sum(result['valid_fire_attempts'] for result in results)}, "
        f"invalid_fire_attempts={sum(result['invalid_fire_attempts'] for result in results)}, "
        f"boundary_failures={sum(result['boundary'] for result in results)}, "
        f"timeouts={sum(result['timeout'] for result in results)}"
    )


def print_firing_geometry_table(title, results):
    print(f"\n{title}")
    print("Episode | Entered geometry | First step/time | Valid fires | Diagnostic")
    for result in results:
        first = result["first_firing_geometry"]
        if first is None:
            first_text = "-"
            diagnostic = "never entered firing geometry"
        else:
            first_text = f"{first['step']}/{first['time_s']:.3f}s"
            diagnostic = (
                "entered geometry but never produced a valid fire"
                if result["valid_fire_attempts"] == 0
                else "valid fire produced"
            )
        print(
            f"{result['episode']} | {result['entered_firing_geometry']} | "
            f"{first_text} | {result['valid_fire_attempts']} | {diagnostic}"
        )


def print_fire_attempt_table(title, results):
    print(f"\n{title}")
    print(
        "Episode | Step | Time (s) | Distance | cos(ATA) | Fire signal | "
        "Target alive | Missile available | Game-over | Done | Result | Invalid reason(s)"
    )
    emitted = False
    for result in results:
        for event in result["fire_events"]:
            emitted = True
            status = "valid" if event["valid"] else "invalid"
            reasons = ", ".join(event["invalid_reasons"]) or "-"
            print(
                f"{result['episode']} | {event['step']} | {event['time_s']:.3f} | "
                f"{event['distance']:.3f} | {event['cos_ata']:.6f} | "
                f"{event['fire_signal']:.6f} | {event['target_alive']} | "
                f"{event['missile_available']} | {event['game_over_at_request']} | "
                f"{event['done_after_step']} | {status} | {reasons}"
            )
    if not emitted:
        print("No rising-edge fire attempts.")


def _format_interception_state(state):
    if state is None:
        return "not crossed"
    return (
        f"d={state['distance']:.3f}, cos={state['cos_ata']:.6f}, "
        f"d_dot={state['d_dot']:.3f}, V={state['airspeed']:.3f}, "
        f"speed_action={state['speed_action']:.6f}"
    )


def print_interception_diagnostic(title, results):
    print(f"\n{title}")
    for result in results:
        minimum = result["minimum_interception_state"]
        print(
            f"Episode {result['episode']} minimum: "
            f"{_format_interception_state(minimum)}"
        )
        for threshold in INTERCEPTION_THRESHOLDS:
            print(
                f"  d < {threshold:.0f} m: "
                f"{_format_interception_state(result['threshold_closest_states'][threshold])}"
            )
        print(f"  firing geometry entered: {result['entered_firing_geometry']}")

    minimum_states = [result["minimum_interception_state"] for result in results]
    print("Aggregate close-range interception metrics")
    for threshold in INTERCEPTION_THRESHOLDS:
        crossed = sum(
            result["threshold_closest_states"][threshold] is not None
            for result in results
        )
        print(f"  Episodes crossing d < {threshold:.0f} m: {crossed}/{len(results)}")
    print(
        "  Episodes satisfying firing geometry: "
        f"{sum(result['entered_firing_geometry'] for result in results)}/{len(results)}"
    )
    print(
        "  Average minimum distance: "
        f"{np.mean([state['distance'] for state in minimum_states]):.3f}"
    )
    print(
        "  Average cos(ATA) at minimum distance: "
        f"{np.mean([state['cos_ata'] for state in minimum_states]):.6f}"
    )
    print(
        "  Average d_dot at minimum distance: "
        f"{np.mean([state['d_dot'] for state in minimum_states]):.3f}"
    )
    print(
        "  Average airspeed at minimum distance: "
        f"{np.mean([state['airspeed'] for state in minimum_states]):.3f}"
    )
    print(
        "  Average speed action at minimum distance: "
        f"{np.mean([state['speed_action'] for state in minimum_states]):.6f}"
    )


def print_fire_signal_diagnostic(title, results):
    print(f"\n{title}")
    print(
        "Episode | Max fire | First fire > 0 (time, distance, cosATA) | "
        "Max <1000 | <500 | <250 | <100 | Max valid geometry"
    )
    for result in results:
        samples = result["fire_signal_samples"]
        first_positive = next(
            (sample for sample in samples if sample["fire_signal"] > 0.0), None
        )
        if first_positive is None:
            first_text = "none"
        else:
            first_text = (
                f"{first_positive['time_s']:.3f}s, {first_positive['distance']:.3f}m, "
                f"{first_positive['cos_ata']:.6f}"
            )
        band_maxima = [
            _maximum_fire_signal(
                [sample for sample in samples if sample["distance"] < threshold]
            )
            for threshold in (1000.0, 500.0, 250.0, 100.0)
        ]
        geometry_maximum = _maximum_fire_signal(
            [
                sample
                for sample in samples
                if sample["distance"] < 100.0 and sample["cos_ata"] > 0.95
            ]
        )
        format_maximum = lambda value: "N/A" if value is None else f"{value:.6f}"
        print(
            f"{result['episode']} | {_maximum_fire_signal(samples):.6f} | {first_text} | "
            f"{format_maximum(band_maxima[0])} | {format_maximum(band_maxima[1])} | "
            f"{format_maximum(band_maxima[2])} | {format_maximum(band_maxima[3])} | "
            f"{format_maximum(geometry_maximum)}"
        )

    all_samples = [sample for result in results for sample in result["fire_signal_samples"]]
    print("Aggregate fire-signal statistics")
    print("Band | Steps | Mean | Max | % > 0 | % > 0.5")
    for threshold in FIRE_SIGNAL_DISTANCE_BANDS:
        stats = _fire_signal_statistics(
            [sample for sample in all_samples if sample["distance"] < threshold]
        )
        if stats is None:
            print(f"d < {threshold:.0f} m | 0 | N/A | N/A | N/A | N/A")
        else:
            print(
                f"d < {threshold:.0f} m | {stats['steps']} | {stats['mean']:.6f} | "
                f"{stats['maximum']:.6f} | {stats['positive_percent']:.2f}% | "
                f"{stats['above_half_percent']:.2f}%"
            )
    geometry_stats = _fire_signal_statistics(
        [
            sample
            for sample in all_samples
            if sample["distance"] < 100.0 and sample["cos_ata"] > 0.95
        ]
    )
    if geometry_stats is None:
        print("valid firing geometry | 0 | N/A | N/A | N/A | N/A")
    else:
        print(
            "valid firing geometry | "
            f"{geometry_stats['steps']} | {geometry_stats['mean']:.6f} | "
            f"{geometry_stats['maximum']:.6f} | {geometry_stats['positive_percent']:.2f}% | "
            f"{geometry_stats['above_half_percent']:.2f}%"
        )


def _fire_signal_geometry_bins(samples):
    """Return mutually exclusive distance/engagement samples for raw fire actions."""
    return (
        ("d >= 1000 m", [sample for sample in samples if sample["distance"] >= 1000.0]),
        (
            "500 <= d < 1000 m",
            [sample for sample in samples if 500.0 <= sample["distance"] < 1000.0],
        ),
        (
            "250 <= d < 500 m",
            [sample for sample in samples if 250.0 <= sample["distance"] < 500.0],
        ),
        (
            "100 <= d < 250 m",
            [sample for sample in samples if 100.0 <= sample["distance"] < 250.0],
        ),
        (
            "d < 100 m, cos(ATA) <= 0.95",
            [
                sample
                for sample in samples
                if sample["distance"] < 100.0 and sample["cos_ata"] <= 0.95
            ],
        ),
        (
            "d < 100 m, cos(ATA) > 0.95 (valid geometry)",
            [
                sample
                for sample in samples
                if sample["distance"] < 100.0 and sample["cos_ata"] > 0.95
            ],
        ),
    )


def _format_fire_signal_bin(label, samples):
    stats = _fire_signal_statistics(samples)
    if stats is None:
        return f"  {label} | 0 | N/A | N/A | N/A | N/A | N/A"
    return (
        f"  {label} | {stats['steps']} | {stats['mean']:.6f} | "
        f"{stats['maximum']:.6f} | {stats['minimum']:.6f} | "
        f"{stats['positive_percent']:.2f}% | {stats['above_half_percent']:.2f}%"
    )


def print_fire_signal_by_geometry_bin(title, results):
    """Print raw policy fire-signal behavior before environment action processing."""
    print(f"\n{title}")
    print("Per-episode bins: Bin | Steps | Mean | Max | Min | % > 0 | % > 0.5")
    for result in results:
        samples = result["fire_signal_samples"]
        bins = _fire_signal_geometry_bins(samples)
        print(f"Episode {result['episode']}")
        for label, bin_samples in bins:
            print(_format_fire_signal_bin(label, bin_samples))

        under_100_samples = [sample for sample in samples if sample["distance"] < 100.0]
        valid_geometry_samples = bins[-1][1]
        max_under_100 = _maximum_fire_signal(under_100_samples)
        max_valid_sample = (
            max(valid_geometry_samples, key=lambda sample: sample["fire_signal"])
            if valid_geometry_samples
            else None
        )
        max_valid = (
            None if max_valid_sample is None else float(max_valid_sample["fire_signal"])
        )
        any_valid_positive = any(
            sample["fire_signal"] > 0.0 for sample in valid_geometry_samples
        )
        any_valid_above_half = any(
            sample["fire_signal"] > 0.5 for sample in valid_geometry_samples
        )
        max_under_100_text = "N/A" if max_under_100 is None else f"{max_under_100:.6f}"
        if max_valid_sample is None:
            max_valid_text = "N/A"
            max_valid_state = "N/A"
        else:
            max_valid_text = f"{max_valid:.6f}"
            max_valid_state = (
                f"d={max_valid_sample['distance']:.3f} m, "
                f"cos(ATA)={max_valid_sample['cos_ata']:.6f}"
            )
        print(
            "  Summary | max fire d < 100 m: "
            f"{max_under_100_text} | max fire valid geometry: {max_valid_text} "
            f"at {max_valid_state} | any > 0: {any_valid_positive} | "
            f"any > 0.5: {any_valid_above_half}"
        )

    all_samples = [sample for result in results for sample in result["fire_signal_samples"]]
    print("Aggregate bins: Bin | Steps | Mean | Max | Min | % > 0 | % > 0.5")
    for label, bin_samples in _fire_signal_geometry_bins(all_samples):
        print(_format_fire_signal_bin(label, bin_samples))


def print_fire_policy_distribution_diagnostic(title, results):
    """Summarize the PPO Gaussian fire output at live valid-geometry states."""
    # This is intentionally unconditionally called from main after PPO rollout.
    # Flush the heading so it remains visible even when terminal output is captured.
    print(f"\n{title}", flush=True)
    print("Episode | Valid-geometry states | Mean fire mean | Mean fire std | Mean P(fire > 0)")
    episodes_with_geometry = 0
    for result in results:
        samples = result["fire_distribution_samples"]
        if not samples:
            continue
        episodes_with_geometry += 1
        means = np.asarray([sample["fire_mean"] for sample in samples], dtype=np.float64)
        stds = np.asarray([sample["fire_std"] for sample in samples], dtype=np.float64)
        probabilities = np.asarray(
            [sample["fire_probability_positive"] for sample in samples],
            dtype=np.float64,
        )
        print(
            f"{result['episode']} | {len(samples)} | {np.mean(means):.6f} | "
            f"{np.mean(stds):.6f} | {np.mean(probabilities):.6f}"
        )

    all_samples = [
        sample for result in results for sample in result["fire_distribution_samples"]
    ]
    print("Aggregate valid-firing-geometry distribution")
    if not all_samples:
        print("Total valid-geometry states: 0")
        print("Episodes containing valid-geometry states: 0")
        print("Mean/min/max fire-action mean: N/A / N/A / N/A")
        print("Mean/min/max fire-action standard deviation: N/A / N/A / N/A")
        print("Mean/min/max estimated P(fire_signal > 0): N/A / N/A / N/A")
        for threshold in (0.1, 0.25, 0.5, 0.75):
            print(
                f"Percentage with P(fire_signal > 0) >= {threshold:.2f}: N/A (0/0)"
            )
        return

    means = np.asarray([sample["fire_mean"] for sample in all_samples], dtype=np.float64)
    stds = np.asarray([sample["fire_std"] for sample in all_samples], dtype=np.float64)
    probabilities = np.asarray(
        [sample["fire_probability_positive"] for sample in all_samples],
        dtype=np.float64,
    )
    total = len(all_samples)
    print(f"Total valid-geometry states: {total}")
    print(f"Episodes containing valid-geometry states: {episodes_with_geometry}")
    print(f"Mean fire-action mean: {np.mean(means):.6f}")
    print(f"Min fire-action mean: {np.min(means):.6f}")
    print(f"Max fire-action mean: {np.max(means):.6f}")
    print(f"Mean fire-action standard deviation: {np.mean(stds):.6f}")
    print(f"Min fire-action standard deviation: {np.min(stds):.6f}")
    print(f"Max fire-action standard deviation: {np.max(stds):.6f}")
    print(f"Mean estimated P(fire_signal > 0): {np.mean(probabilities):.6f}")
    print(f"Min estimated P(fire_signal > 0): {np.min(probabilities):.6f}")
    print(f"Max estimated P(fire_signal > 0): {np.max(probabilities):.6f}")
    for threshold in (0.1, 0.25, 0.5, 0.75):
        count = int(np.count_nonzero(probabilities >= threshold))
        print(
            f"Percentage with P(fire_signal > 0) >= {threshold:.2f}: "
            f"{100.0 * count / total:.2f}% ({count}/{total})"
        )


def _format_geometry_transition(transition):
    if transition is None:
        return "N/A"
    return f"{transition['step']}/{transition['time_s']:.3f}s"


def print_geometry_duration_diagnostic(title, results):
    """Report time spent in d < 100 m and cos(ATA) > 0.95."""
    print(f"\n{title}")
    print(
        "Episode | Valid steps | Valid % | First entry | Last exit | "
        "Number of intervals | Longest interval | Min distance | Max cosATA"
    )
    for result in results:
        duration = result["geometry_duration"]
        minimum_distance = (
            "N/A"
            if duration["minimum_distance"] is None
            else f"{duration['minimum_distance']:.3f}"
        )
        maximum_cos_ata = (
            "N/A"
            if duration["maximum_cos_ata"] is None
            else f"{duration['maximum_cos_ata']:.6f}"
        )
        print(
            f"{result['episode']} | {duration['valid_steps']} | "
            f"{duration['valid_percentage']:.3f}% | "
            f"{_format_geometry_transition(duration['first_entry'])} | "
            f"{_format_geometry_transition(duration['last_exit'])} | "
            f"{duration['interval_count']} | "
            f"{duration['longest_interval_steps']} | "
            f"{minimum_distance} | {maximum_cos_ata}"
        )

    durations = [result["geometry_duration"] for result in results]
    longest_intervals = np.asarray(
        [duration["longest_interval_steps"] for duration in durations],
        dtype=np.float64,
    )
    minimum_distances = [
        duration["minimum_distance"]
        for duration in durations
        if duration["minimum_distance"] is not None
    ]
    maximum_cos_ata = [
        duration["maximum_cos_ata"]
        for duration in durations
        if duration["maximum_cos_ata"] is not None
    ]
    print("Aggregate valid-firing-geometry duration")
    print(f"Total valid-geometry steps: {sum(d['valid_steps'] for d in durations)}")
    print(
        "Mean valid-geometry steps per episode: "
        f"{np.mean([d['valid_steps'] for d in durations]):.3f}"
    )
    print(
        "Mean valid-geometry percentage: "
        f"{np.mean([d['valid_percentage'] for d in durations]):.6f}%"
    )
    print(
        "Mean number of intervals: "
        f"{np.mean([d['interval_count'] for d in durations]):.3f}"
    )
    print(f"Mean longest interval: {np.mean(longest_intervals):.3f} steps")
    print(
        "Minimum/maximum longest interval: "
        f"{int(np.min(longest_intervals))} / {int(np.max(longest_intervals))} steps"
    )
    print(
        "Mean minimum distance within valid geometry: "
        + (
            f"{np.mean(minimum_distances):.3f} m"
            if minimum_distances
            else "N/A"
        )
    )
    print(
        "Mean maximum cos(ATA) within valid geometry: "
        + (f"{np.mean(maximum_cos_ata):.6f}" if maximum_cos_ata else "N/A")
    )


def summarize(results):
    rewards = np.array([result["reward"] for result in results], dtype=np.float64)
    lengths = np.array([result["length"] for result in results], dtype=np.float64)
    distances = np.array([result["distance"] for result in results], dtype=np.float64)
    cos_ata = np.array([result["cos_ata"] for result in results], dtype=np.float64)
    d_dot = np.array([result["d_dot"] for result in results], dtype=np.float64)
    fire_attempts = np.array([result["fire_attempts"] for result in results], dtype=np.int64)
    valid_fire_attempts = np.array(
        [result["valid_fire_attempts"] for result in results], dtype=np.int64
    )
    invalid_fire_attempts = np.array(
        [result["invalid_fire_attempts"] for result in results], dtype=np.int64
    )
    missile_hits = np.array([result["missile_hits"] for result in results], dtype=np.int64)
    successes = sum(result["success"] for result in results)
    boundaries = sum(result["boundary"] for result in results)

    return {
        "episodes": len(results),
        "successes": successes,
        "boundaries": boundaries,
        "other": len(results) - successes - boundaries,
        "success_rate": 100.0 * successes / len(results),
        "rewards": rewards,
        "lengths": lengths,
        "distances": distances,
        "cos_ata": cos_ata,
        "d_dot": d_dot,
        "fire_attempts": fire_attempts,
        "valid_fire_attempts": valid_fire_attempts,
        "invalid_fire_attempts": invalid_fire_attempts,
        "missile_hits": missile_hits,
    }


def print_summary(title, summary):
    print(f"\n{title}")
    print(f"Episodes evaluated: {summary['episodes']}")
    print(f"Successful engagements: {summary['successes']}/{summary['episodes']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Boundary failures: {summary['boundaries']}/{summary['episodes']}")
    print(f"Other/timeout endings: {summary['other']}/{summary['episodes']}")
    print(f"Mean episode reward: {np.mean(summary['rewards']):.3f}")
    print(f"Median episode reward: {np.median(summary['rewards']):.3f}")
    print(f"Best episode reward: {np.max(summary['rewards']):.3f}")
    print(f"Worst episode reward: {np.min(summary['rewards']):.3f}")
    print(f"Mean episode length: {np.mean(summary['lengths']):.3f}")
    print(f"Median episode length: {np.median(summary['lengths']):.3f}")
    print(f"Mean final distance: {np.mean(summary['distances']):.3f}")
    print(f"Best final distance: {np.min(summary['distances']):.3f}")
    print(f"Mean final cos(ATA): {np.mean(summary['cos_ata']):.6f}")
    print(f"Best final cos(ATA): {np.max(summary['cos_ata']):.6f}")
    print(f"Mean final d_dot: {np.mean(summary['d_dot']):.3f}")
    print(f"Total fire attempts: {np.sum(summary['fire_attempts'])}")
    print(f"Valid fire attempts: {np.sum(summary['valid_fire_attempts'])}")
    print(f"Invalid fire attempts: {np.sum(summary['invalid_fire_attempts'])}")
    print(f"Missile hits: {np.sum(summary['missile_hits'])}")
    print(
        "Episodes with at least one valid fire: "
        f"{np.count_nonzero(summary['valid_fire_attempts'])}/{summary['episodes']}"
    )
    print(
        "Episodes with a missile hit: "
        f"{np.count_nonzero(summary['missile_hits'])}/{summary['episodes']}"
    )


def main(
    model_path=MODEL_PATH,
    stochastic=False,
    stochastic_fire_only=False,
    randomized_enemy=False,
):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    stats_path = normalization_path(model_path)
    print(f"Evaluating model: {model_path}")
    print(f"VecNormalize statistics: {stats_path}")
    action_mode = (
        "stochastic-fire-only"
        if stochastic_fire_only
        else ("stochastic" if stochastic else "deterministic")
    )
    print(f"Policy action mode: {action_mode}")

    env = make_evaluation_env(model_path)
    try:
        model = PPO.load(str(model_path), env=env, device="cpu")
        if randomized_enemy:
            challenge_config = RandomizedEnemyChallengeConfig(
                enabled=True,
                scenario_manifest=RANDOMIZED_ENEMY_CHALLENGE.scenario_manifest,
            )
            challenge_seeds = tuple(
                scenario.seed for scenario in challenge_config.scenario_manifest
            )
            ppo_results = evaluate_policy(
                env,
                model,
                challenge_seeds,
                deterministic=not stochastic,
                stochastic_fire_only=stochastic_fire_only,
                collect_fire_distribution=True,
                challenge_config=challenge_config,
                print_challenge_scenarios=True,
            )
            random_results = evaluate_policy(
                env,
                None,
                challenge_seeds,
                deterministic=False,
                challenge_config=challenge_config,
            )
        else:
            ppo_results = evaluate_policy(
                env,
                model,
                range(1001, 1001 + EPISODES),
                deterministic=not stochastic,
                stochastic_fire_only=stochastic_fire_only,
                collect_fire_distribution=True,
            )
            random_results = evaluate_policy(
                env,
                None,
                range(2001, 2001 + EPISODES),
                deterministic=False,
            )
    finally:
        env.venv.envs[0].sim.destroy()
        env.close()

    ppo_summary = summarize(ppo_results)
    random_summary = summarize(random_results)

    if randomized_enemy:
        print_challenge_episode_table("PPO RANDOMIZED-ENEMY CHALLENGE", ppo_results)
    print_episode_table("PPO EPISODES", ppo_results)
    print_firing_geometry_table("PPO FIRING GEOMETRY", ppo_results)
    print_fire_attempt_table("PPO FIRE ATTEMPTS", ppo_results)
    print_interception_diagnostic("PPO CLOSE-RANGE INTERCEPTION DIAGNOSTIC", ppo_results)
    print_geometry_duration_diagnostic(
        "VALID FIRING GEOMETRY DURATION DIAGNOSTIC", ppo_results
    )
    print_fire_signal_diagnostic("PPO FIRE-SIGNAL DIAGNOSTIC", ppo_results)
    print_fire_signal_by_geometry_bin("PPO FIRE-SIGNAL BY GEOMETRY BIN", ppo_results)
    print_fire_policy_distribution_diagnostic(
        "FIRE POLICY DISTRIBUTION DIAGNOSTIC", ppo_results
    )
    print_summary("PPO EVALUATION RESULTS", ppo_summary)
    if randomized_enemy:
        print_challenge_episode_table(
            "RANDOM BASELINE RANDOMIZED-ENEMY CHALLENGE", random_results
        )
    print_episode_table("RANDOM BASELINE EPISODES", random_results)
    print_firing_geometry_table("RANDOM FIRING GEOMETRY", random_results)
    print_fire_attempt_table("RANDOM FIRE ATTEMPTS", random_results)
    print_summary("RANDOM BASELINE RESULTS", random_summary)

    print("\nPPO vs RANDOM")
    print("Metric                 PPO        Random")
    print("------------------------------------------")
    print(f"Success rate           {ppo_summary['success_rate']:.1f}%      {random_summary['success_rate']:.1f}%")
    print(f"Mean reward            {np.mean(ppo_summary['rewards']):.3f}   {np.mean(random_summary['rewards']):.3f}")
    print(f"Mean episode length    {np.mean(ppo_summary['lengths']):.3f}    {np.mean(random_summary['lengths']):.3f}")
    print(f"Mean final distance    {np.mean(ppo_summary['distances']):.3f}   {np.mean(random_summary['distances']):.3f}")
    print(f"Mean final cos(ATA)    {np.mean(ppo_summary['cos_ata']):.6f}   {np.mean(random_summary['cos_ata']):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PPO model over 20 episodes.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--randomized-enemy",
        action="store_true",
        help="Run the frozen randomized-enemy challenge manifest.",
    )
    action_mode_group = parser.add_mutually_exclusive_group()
    action_mode_group.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample PPO actions instead of using deterministic policy means.",
    )
    action_mode_group.add_argument(
        "--stochastic-fire-only",
        action="store_true",
        help="Use deterministic flight controls and a stochastic PPO fire action.",
    )
    args = parser.parse_args()
    main(
        args.model_path,
        stochastic=args.stochastic,
        stochastic_fire_only=args.stochastic_fire_only,
        randomized_enemy=args.randomized_enemy,
    )
