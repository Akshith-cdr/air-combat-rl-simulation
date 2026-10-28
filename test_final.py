"""Final randomized-scenario evaluation for the trained hybrid PPO policy.

This module is evaluation-only.  It samples and records an explicit scenario
manifest, executes the hybrid policy stochastically, and writes detailed and
aggregate CSV files without changing the environment or checkpoint.

Event-time and valid-fire-range fields are written as NaN when the associated
event never occurs.  The long ``maximum_fire_action...`` field records the
maximum Bernoulli fire probability exposed by the hybrid policy.
"""

from __future__ import annotations

import argparse
import csv
import math
import secrets
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fighter_jet_env import FighterJetEnv
from hybrid_ppo import HybridActorCriticPolicy


DEFAULT_MODEL_PATH = Path(
    "models/ppo_ucav_100k_19d_hybrid_fire_randomized_training.zip"
)
DEFAULT_VECNORMALIZE_PATH = Path(
    "models/ppo_ucav_100k_19d_hybrid_fire_randomized_training_vecnormalize.pkl"
)
DEFAULT_SCENARIOS = 50
DEFAULT_MAX_STEPS = 3000

DETAIL_COLUMNS = [
    "scenario_id",
    "scenario_seed",
    "ucav_initial_x",
    "ucav_initial_y",
    "ucav_initial_z",
    "ucav_initial_speed",
    "ucav_initial_heading",
    "enemy_initial_x",
    "enemy_initial_y",
    "enemy_initial_z",
    "enemy_initial_speed",
    "enemy_initial_heading",
    "initial_3d_distance",
    "success",
    "terminal_reason",
    "missile_hit",
    "boundary_failure",
    "timeout",
    "episode_length_steps",
    "episode_time_seconds",
    "total_reward",
    "minimum_3d_distance_m",
    "minimum_horizontal_distance_m",
    "minimum_vertical_distance_m",
    "maximum_cosATA",
    "minimum_range_at_valid_fire",
    "maximum_fire_action_or_fire_probability_if_available",
    "number_of_fire_attempts",
    "number_of_valid_fire_attempts",
    "number_of_invalid_fire_attempts",
    "missile_launch_count",
    "missile_hit_count",
    "time_to_min_distance_seconds",
    "time_to_first_valid_fire_seconds",
    "time_to_missile_launch_seconds",
    "time_to_missile_hit_seconds",
    "final_3d_distance_m",
    "final_cosATA",
    "final_d_dot",
    "final_speed",
]

SUMMARY_COLUMNS = [
    "master_seed",
    "policy_seed",
    "model_path",
    "vecnormalize_path",
    "policy_mode",
    "total_scenarios",
    "successful_engagements",
    "success_rate",
    "missile_hits",
    "missile_hit_rate",
    "boundary_failures",
    "timeouts",
    "mean_episode_length_steps",
    "mean_episode_time_seconds",
    "mean_total_reward",
    "mean_minimum_3d_distance_m",
    "median_minimum_3d_distance_m",
    "best_minimum_3d_distance_m",
    "worst_minimum_3d_distance_m",
    "mean_maximum_cosATA",
    "mean_initial_separation_m",
    "mean_time_to_missile_launch_seconds",
    "mean_time_to_missile_hit_seconds",
    "valid_fire_rate",
    "invalid_fire_rate",
]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("seed must be a non-negative integer")
    return parsed


def _derived_seed(master_seed: int, stream: int) -> int:
    """Derive a stable uint32 seed for one independent random stream."""
    sequence = np.random.SeedSequence(master_seed, spawn_key=(stream,))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def generate_scenarios(count: int, master_seed: int) -> list[dict[str, Any]]:
    """Build reproducible scenarios from per-scenario independent seeds."""
    seed_rng = np.random.default_rng(_derived_seed(master_seed, 0))
    used_seeds: set[int] = set()
    scenarios = []

    for index in range(1, count + 1):
        scenario_seed = int(seed_rng.integers(0, 2**32, dtype=np.uint32))
        while scenario_seed in used_seeds:
            scenario_seed = int(seed_rng.integers(0, 2**32, dtype=np.uint32))
        used_seeds.add(scenario_seed)

        rng = np.random.default_rng(scenario_seed)
        enemy_x = float(rng.uniform(-1200.0, 1200.0))
        enemy_y = float(rng.uniform(2200.0, 4800.0))
        enemy_z = float(rng.uniform(700.0, 2200.0))
        enemy_speed = float(rng.uniform(140.0, 240.0))
        enemy_heading = float(rng.uniform(0.0, 345.0))
        scenarios.append(
            {
                "scenario_id": f"FINAL-{index:06d}",
                "seed": scenario_seed,
                "enemy_initial_position": (enemy_x, enemy_y, enemy_z),
                "enemy_initial_speed": enemy_speed,
                "enemy_initial_heading": enemy_heading,
                "enemy_initial_altitude": enemy_z,
            }
        )
    return scenarios


def make_env(stats_path: Path) -> VecNormalize:
    env = VecNormalize.load(
        str(stats_path),
        DummyVecEnv([lambda: FighterJetEnv(hybrid_fire=True)]),
    )
    env.training = False
    env.norm_reward = False
    return env


def _raw_after_step(env: VecNormalize, done: bool, info: dict[str, Any]) -> np.ndarray:
    """Get the state just produced, avoiding DummyVecEnv's terminal auto-reset."""
    if done and info.get("terminal_observation") is not None:
        terminal = np.asarray(info["terminal_observation"], dtype=np.float32)
        return np.asarray(env.unnormalize_obs(terminal), dtype=np.float64)
    return np.asarray(env.get_original_obs()[0], dtype=np.float64).copy()


def _geometry(raw_observation: np.ndarray) -> tuple[float, float, float, float]:
    dx, dy, dz = (float(value) for value in raw_observation[7:10])
    return (
        math.sqrt(dx * dx + dy * dy + dz * dz),
        math.hypot(dx, dy),
        abs(dz),
        float(raw_observation[11]),
    )


def _fire_probability(model: PPO, observation: np.ndarray) -> float:
    observation_tensor, _ = model.policy.obs_to_tensor(observation)
    with th.no_grad():
        distribution = model.policy.get_distribution(observation_tensor)
        bernoulli = getattr(distribution, "bernoulli", None)
        if bernoulli is None:
            return math.nan
        return float(bernoulli.probs.detach().cpu().reshape(-1)[0].item())


def _actual_initial_state(env: VecNormalize) -> dict[str, float]:
    base_env = env.venv.envs[0]
    player = base_env.sim.player
    enemy = base_env.sim.enemy
    relative = enemy.position - player.position
    return {
        "ucav_initial_x": float(player.position.x),
        "ucav_initial_y": float(player.position.y),
        "ucav_initial_z": float(player.position.z),
        "ucav_initial_speed": float(player.velocity.length()),
        "ucav_initial_heading": float(player.yaw),
        "enemy_initial_x": float(enemy.position.x),
        "enemy_initial_y": float(enemy.position.y),
        "enemy_initial_z": float(enemy.position.z),
        "enemy_initial_speed": float(enemy.velocity.length()),
        "enemy_initial_heading": float(enemy.yaw),
        "initial_3d_distance": float(relative.length()),
    }


def evaluate_scenario(
    env: VecNormalize,
    model: PPO,
    scenario: dict[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    env.set_options({"enemy_challenge_scenario": scenario})
    env.seed(int(scenario["seed"]))
    observation = env.reset()

    # Capture these directly from simulator objects immediately after reset and
    # before the first action changes the episode state.
    result: dict[str, Any] = {
        "scenario_id": scenario["scenario_id"],
        "scenario_seed": int(scenario["seed"]),
        **_actual_initial_state(env),
    }
    raw = np.asarray(env.get_original_obs()[0], dtype=np.float64).copy()
    distance_3d, horizontal, vertical, cos_ata = _geometry(raw)

    total_reward = 0.0
    elapsed_time = 0.0
    minimum_3d = distance_3d
    minimum_horizontal = horizontal
    minimum_vertical = vertical
    maximum_cos_ata = cos_ata
    maximum_fire_probability = math.nan
    minimum_range_at_valid_fire = math.nan
    time_to_min_distance = 0.0
    time_to_first_valid_fire = math.nan
    time_to_missile_launch = math.nan
    time_to_missile_hit = math.nan
    fire_attempts = 0
    valid_fire_attempts = 0
    invalid_fire_attempts = 0
    missile_launches = 0
    missile_hits = 0
    terminal_reason = "evaluation_step_limit"
    episode_length = 0

    for episode_length in range(1, max_steps + 1):
        pre_step_raw = raw
        pre_step_distance = float(pre_step_raw[10])
        probability = _fire_probability(model, observation)
        if math.isnan(maximum_fire_probability) or probability > maximum_fire_probability:
            maximum_fire_probability = probability

        action, _ = model.predict(observation, deterministic=False)
        fire_values = np.asarray(action)[..., 4]
        if not np.all((fire_values == 0.0) | (fire_values == 1.0)):
            raise ValueError("Hybrid policy produced a non-binary fire action")

        observation, reward, done_array, infos = env.step(action)
        done = bool(done_array[0])
        info = infos[0]
        dt = float(info.get("dt", math.nan))
        if math.isfinite(dt):
            elapsed_time += dt
        total_reward += float(reward[0])

        weapon = info.get("weapon", {})
        fire_request = bool(weapon.get("fire_request", False))
        valid_fire = bool(weapon.get("valid_fire_attempt", False))
        invalid_fire = bool(weapon.get("invalid_fire_attempt", False))
        hit_this_step = bool(weapon.get("missile_hit_this_step", False))
        fire_attempts += int(fire_request)
        valid_fire_attempts += int(valid_fire)
        invalid_fire_attempts += int(invalid_fire)

        # In the current environment a valid fire attempt is true only when
        # fire_missile actually appended a projectile to active_missiles.
        if valid_fire:
            missile_launches += 1
            minimum_range_at_valid_fire = (
                pre_step_distance
                if math.isnan(minimum_range_at_valid_fire)
                else min(minimum_range_at_valid_fire, pre_step_distance)
            )
            if math.isnan(time_to_first_valid_fire):
                time_to_first_valid_fire = elapsed_time
            if math.isnan(time_to_missile_launch):
                time_to_missile_launch = elapsed_time
        if hit_this_step:
            missile_hits += 1
            if math.isnan(time_to_missile_hit):
                time_to_missile_hit = elapsed_time

        raw = _raw_after_step(env, done, info)
        distance_3d, horizontal, vertical, cos_ata = _geometry(raw)
        if distance_3d < minimum_3d:
            minimum_3d = distance_3d
            time_to_min_distance = elapsed_time
        minimum_horizontal = min(minimum_horizontal, horizontal)
        minimum_vertical = min(minimum_vertical, vertical)
        maximum_cos_ata = max(maximum_cos_ata, cos_ata)

        if done:
            terminal_reason = str(info.get("terminal_reason", "") or "terminated")
            break

    timeout = terminal_reason == "evaluation_step_limit"
    boundary_failure = "CHASE BOUNDARY" in terminal_reason.upper()
    success = terminal_reason == "missile_hit" and missile_hits > 0

    result.update(
        {
            "success": success,
            "terminal_reason": terminal_reason,
            "missile_hit": missile_hits > 0,
            "boundary_failure": boundary_failure,
            "timeout": timeout,
            "episode_length_steps": episode_length,
            "episode_time_seconds": elapsed_time,
            "total_reward": total_reward,
            "minimum_3d_distance_m": minimum_3d,
            "minimum_horizontal_distance_m": minimum_horizontal,
            "minimum_vertical_distance_m": minimum_vertical,
            "maximum_cosATA": maximum_cos_ata,
            "minimum_range_at_valid_fire": minimum_range_at_valid_fire,
            "maximum_fire_action_or_fire_probability_if_available": maximum_fire_probability,
            "number_of_fire_attempts": fire_attempts,
            "number_of_valid_fire_attempts": valid_fire_attempts,
            "number_of_invalid_fire_attempts": invalid_fire_attempts,
            "missile_launch_count": missile_launches,
            "missile_hit_count": missile_hits,
            "time_to_min_distance_seconds": time_to_min_distance,
            "time_to_first_valid_fire_seconds": time_to_first_valid_fire,
            "time_to_missile_launch_seconds": time_to_missile_launch,
            "time_to_missile_hit_seconds": time_to_missile_hit,
            "final_3d_distance_m": distance_3d,
            "final_cosATA": cos_ata,
            "final_d_dot": float(raw[14]),
            "final_speed": float(raw[0]),
        }
    )
    return result


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else math.nan


def summarize(
    results: list[dict[str, Any]],
    master_seed: int,
    policy_seed: int,
    model_path: Path,
    stats_path: Path,
) -> dict[str, Any]:
    count = len(results)
    successes = sum(bool(row["success"]) for row in results)
    scenarios_with_hits = sum(bool(row["missile_hit"]) for row in results)
    total_hits = sum(int(row["missile_hit_count"]) for row in results)
    valid = sum(int(row["number_of_valid_fire_attempts"]) for row in results)
    invalid = sum(int(row["number_of_invalid_fire_attempts"]) for row in results)
    attempts = valid + invalid
    minimum_distances = [float(row["minimum_3d_distance_m"]) for row in results]

    return {
        "master_seed": master_seed,
        "policy_seed": policy_seed,
        "model_path": str(model_path),
        "vecnormalize_path": str(stats_path),
        "policy_mode": "stochastic",
        "total_scenarios": count,
        "successful_engagements": successes,
        "success_rate": successes / count,
        "missile_hits": total_hits,
        "missile_hit_rate": scenarios_with_hits / count,
        "boundary_failures": sum(bool(row["boundary_failure"]) for row in results),
        "timeouts": sum(bool(row["timeout"]) for row in results),
        "mean_episode_length_steps": _mean([row["episode_length_steps"] for row in results]),
        "mean_episode_time_seconds": _mean([row["episode_time_seconds"] for row in results]),
        "mean_total_reward": _mean([row["total_reward"] for row in results]),
        "mean_minimum_3d_distance_m": _mean(minimum_distances),
        "median_minimum_3d_distance_m": float(np.median(minimum_distances)),
        "best_minimum_3d_distance_m": min(minimum_distances),
        "worst_minimum_3d_distance_m": max(minimum_distances),
        "mean_maximum_cosATA": _mean([row["maximum_cosATA"] for row in results]),
        "mean_initial_separation_m": _mean([row["initial_3d_distance"] for row in results]),
        "mean_time_to_missile_launch_seconds": _mean([row["time_to_missile_launch_seconds"] for row in results]),
        "mean_time_to_missile_hit_seconds": _mean([row["time_to_missile_hit_seconds"] for row in results]),
        "valid_fire_rate": valid / attempts if attempts else math.nan,
        "invalid_fire_rate": invalid / attempts if attempts else math.nan,
    }


def choose_output_paths(output_dir: Path, count: int, master_seed: int) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    copy_number = 1
    while True:
        tag = f"{count}_seed_{master_seed}{suffix}"
        details = output_dir / f"final_eval_{tag}.csv"
        summary = output_dir / f"final_summary_{tag}.csv"
        if not details.exists() and not summary.exists():
            return details, summary
        copy_number += 1
        suffix = f"_run_{copy_number}"


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    # Exclusive creation protects against a race or an accidental overwrite.
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_model(model: PPO) -> None:
    if not isinstance(model.policy, HybridActorCriticPolicy):
        raise TypeError("The supplied checkpoint is not a hybrid PPO policy")
    if model.observation_space.shape != (19,) or model.action_space.shape != (5,):
        raise ValueError("Hybrid evaluation requires 19 observations and 5 actions")
    if float(model.action_space.low[4]) != 0.0:
        raise ValueError("The supplied model does not use binary-fire transport bounds")


def run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    missing = [
        path
        for path in (args.model_path, args.vecnormalize_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing evaluation input: " + ", ".join(map(str, missing)))

    master_seed = args.seed if args.seed is not None else secrets.randbits(64)
    policy_seed = args.policy_seed if args.policy_seed is not None else _derived_seed(master_seed, 1)
    scenarios = generate_scenarios(args.scenarios, master_seed)

    print("Final evaluation")
    print(f"Scenarios: {args.scenarios}")
    print(f"Master seed: {master_seed}")
    print(f"Policy seed: {policy_seed}")
    print("Policy mode: stochastic")

    env = make_env(args.vecnormalize_path)
    results: list[dict[str, Any]] = []
    try:
        model = PPO.load(str(args.model_path), env=env, device="cpu")
        validate_model(model)
        # This controls policy sampling. Each environment reset below is seeded
        # independently with its scenario seed.
        model.set_random_seed(policy_seed)

        progress_every = 1 if args.scenarios <= 100 else 10
        for index, scenario in enumerate(scenarios, start=1):
            if index == 1 or index == args.scenarios or index % progress_every == 0:
                print(f"Scenario {index}/{args.scenarios}")
            results.append(evaluate_scenario(env, model, scenario, args.max_steps))
    finally:
        try:
            env.venv.envs[0].sim.destroy()
        finally:
            env.close()

    summary = summarize(
        results,
        master_seed,
        policy_seed,
        args.model_path,
        args.vecnormalize_path,
    )
    detail_path, summary_path = choose_output_paths(Path("tests"), args.scenarios, master_seed)
    write_csv(detail_path, DETAIL_COLUMNS, results)
    try:
        write_csv(summary_path, SUMMARY_COLUMNS, [summary])
    except Exception:
        # Keep the detailed data if summary writing fails; never delete output.
        raise

    print("\n" + "=" * 50)
    print("FINAL HYBRID PPO EVALUATION")
    print("=" * 50)
    print(f"Scenarios: {args.scenarios}")
    print(f"Master seed: {master_seed}")
    print(f"Policy seed: {policy_seed}")
    print("Policy mode: stochastic\n")
    print(f"Successful missile engagements: {summary['successful_engagements']}/{args.scenarios}")
    print(f"Success rate: {100.0 * summary['success_rate']:.2f}%\n")
    print(f"Missile hits: {summary['missile_hits']}")
    print(f"Boundary failures: {summary['boundary_failures']}")
    print(f"Timeouts: {summary['timeouts']}\n")
    print(f"Mean minimum 3-D distance: {summary['mean_minimum_3d_distance_m']:.3f} m")
    print(f"Mean maximum cosATA: {summary['mean_maximum_cosATA']:.6f}")
    print(f"Mean episode time: {summary['mean_episode_time_seconds']:.6f} s\n")
    print(f"Detailed results:\n{detail_path}")
    print(f"\nSummary:\n{summary_path}")
    print("=" * 50)
    print(
        "Note: simulator dt comes from Panda3D's wall clock; identical seeds reproduce "
        "the manifest and random streams, but do not guarantee numerically identical rollouts."
    )
    return detail_path, summary_path, summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the hybrid PPO fighter policy on fresh randomized scenarios."
    )
    parser.add_argument("--scenarios", type=_positive_int, default=DEFAULT_SCENARIOS)
    parser.add_argument("--seed", type=_nonnegative_int, default=None, help="Master scenario-generation seed.")
    parser.add_argument("--policy-seed", type=_nonnegative_int, default=None, help="Independent stochastic-policy seed (derived from --seed by default).")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vecnormalize-path", type=Path, default=DEFAULT_VECNORMALIZE_PATH)
    parser.add_argument("--max-steps", type=_positive_int, default=DEFAULT_MAX_STEPS)
    return parser


if __name__ == "__main__":
    run(build_argument_parser().parse_args())
