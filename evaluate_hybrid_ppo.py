"""Evaluate hybrid-fire PPO on the frozen 24-scenario enemy challenge."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evaluate_ppo import EPISODES, FROZEN_ENEMY_CHALLENGE_MANIFEST, MAX_EVALUATION_STEPS
from fighter_jet_env import FighterJetEnv
from hybrid_ppo import HybridActorCriticPolicy


DEFAULT_MODEL = Path(
    "models/ppo_ucav_100k_19d_hybrid_fire_randomized_training.zip"
)
LEGACY_EVALUATION_SEEDS = tuple(range(1001, 1001 + EPISODES))


def normalization_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_vecnormalize.pkl")


def make_env(stats_path: Path):
    env = VecNormalize.load(
        str(stats_path),
        DummyVecEnv([lambda: FighterJetEnv(hybrid_fire=True)]),
    )
    env.training = False
    env.norm_reward = False
    return env


def evaluation_cases(legacy: bool):
    if legacy:
        return tuple(
            (f"LEGACY-{seed}", seed, None) for seed in LEGACY_EVALUATION_SEEDS
        )
    return tuple(
        (scenario.scenario_id, scenario.seed, scenario)
        for scenario in FROZEN_ENEMY_CHALLENGE_MANIFEST
    )


def evaluate(model_path: Path, deterministic: bool, legacy: bool = False) -> list[dict]:
    stats_path = normalization_path(model_path)
    missing = [str(path) for path in (model_path, stats_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing hybrid evaluation input: " + ", ".join(missing))

    env = make_env(stats_path)
    results = []
    try:
        model = PPO.load(str(model_path), env=env, device="cpu")
        if not isinstance(model.policy, HybridActorCriticPolicy):
            raise TypeError("The supplied checkpoint is not a hybrid PPO policy")
        if model.observation_space.shape != (19,) or model.action_space.shape != (5,):
            raise ValueError("Hybrid evaluation requires 19 observations and 5 actions")
        if float(model.action_space.low[4]) != 0.0:
            raise ValueError("The supplied model does not use binary-fire transport bounds")

        for evaluation_id, seed, scenario in evaluation_cases(legacy):
            if scenario is None:
                print(f"{evaluation_id}: seed={seed}, original fixed enemy initialization")
            else:
                print(
                    f"{scenario.scenario_id}: seed={scenario.seed}, "
                    f"position={scenario.enemy_initial_position}, "
                    f"speed={scenario.enemy_initial_speed:.1f}, "
                    f"heading={scenario.enemy_initial_heading:.1f}, "
                    f"altitude={scenario.enemy_initial_altitude:.1f}"
                )
                env.set_options({"enemy_challenge_scenario": asdict(scenario)})
            env.seed(seed)
            observation = env.reset()
            raw_observation = env.get_original_obs()[0].copy()
            total_reward = 0.0
            minimum_distance = float(raw_observation[10])
            maximum_cos_ata = float(raw_observation[11])
            valid_fire_attempts = 0
            invalid_fire_attempts = 0
            missile_hits = 0
            terminal_reason = "evaluation_step_limit"
            final_observation = raw_observation

            for episode_length in range(1, MAX_EVALUATION_STEPS + 1):
                action, _ = model.predict(observation, deterministic=deterministic)
                fire_values = np.asarray(action)[..., 4]
                if not np.all((fire_values == 0.0) | (fire_values == 1.0)):
                    raise ValueError("Hybrid policy produced a non-binary fire action")

                observation, reward, done, infos = env.step(action)
                total_reward += float(reward[0])
                info = infos[0]
                weapon = info.get("weapon", {})
                valid_fire_attempts += int(bool(weapon.get("valid_fire_attempt", False)))
                invalid_fire_attempts += int(bool(weapon.get("invalid_fire_attempt", False)))
                missile_hits += int(bool(weapon.get("missile_hit_this_step", False)))

                if bool(done[0]):
                    terminal_reason = info.get("terminal_reason", "") or "terminated"
                    terminal_observation = info.get("terminal_observation")
                    final_observation = (
                        env.unnormalize_obs(terminal_observation)
                        if terminal_observation is not None
                        else env.get_original_obs()[0].copy()
                    )
                else:
                    final_observation = env.get_original_obs()[0].copy()

                minimum_distance = min(minimum_distance, float(final_observation[10]))
                maximum_cos_ata = max(maximum_cos_ata, float(final_observation[11]))
                if bool(done[0]):
                    break

            results.append(
                {
                    "scenario_id": evaluation_id,
                    "seed": seed,
                    "success": terminal_reason == "missile_hit",
                    "missile_hits": missile_hits,
                    "valid_fire_attempts": valid_fire_attempts,
                    "invalid_fire_attempts": invalid_fire_attempts,
                    "minimum_distance": minimum_distance,
                    "maximum_cos_ata": maximum_cos_ata,
                    "final_distance": float(final_observation[10]),
                    "final_cos_ata": float(final_observation[11]),
                    "episode_length": episode_length,
                    "terminal_reason": terminal_reason,
                    "reward": total_reward,
                }
            )
    finally:
        env.venv.envs[0].sim.destroy()
        env.close()
    return results


def print_results(results: list[dict]) -> None:
    print(
        "\nScenario | Success | Hits | Valid | Invalid | Min range | Max cosATA | "
        "Final range | Final cosATA | Length | Terminal reason | Reward"
    )
    for result in results:
        print(
            f"{result['scenario_id']} | {result['success']} | "
            f"{result['missile_hits']} | {result['valid_fire_attempts']} | "
            f"{result['invalid_fire_attempts']} | {result['minimum_distance']:.3f} | "
            f"{result['maximum_cos_ata']:.6f} | {result['final_distance']:.3f} | "
            f"{result['final_cos_ata']:.6f} | {result['episode_length']} | "
            f"{result['terminal_reason']} | {result['reward']:.3f}"
        )
    print(
        "\nTotals: "
        f"successes={sum(item['success'] for item in results)}/{len(results)}, "
        f"missile_hits={sum(item['missile_hits'] for item in results)}, "
        f"valid_fire_attempts={sum(item['valid_fire_attempts'] for item in results)}, "
        f"invalid_fire_attempts={sum(item['invalid_fire_attempts'] for item in results)}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid PPO on the frozen 24-scenario challenge."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Run the original fixed-start 20-episode evaluation (seeds 1001-1020).",
    )
    action_mode = parser.add_mutually_exclusive_group()
    action_mode.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Use Gaussian means and Bernoulli mode.",
    )
    action_mode.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample the Gaussian and Bernoulli components (default).",
    )
    parser.set_defaults(deterministic=False)
    return parser


if __name__ == "__main__":
    arguments = build_argument_parser().parse_args()
    print(
        "Policy action mode: "
        + ("deterministic" if arguments.deterministic else "stochastic")
    )
    print(
        "Evaluation suite: "
        + ("legacy fixed-start seeds 1001-1020" if arguments.legacy else "frozen REC-001 to REC-024")
    )
    evaluation_results = evaluate(
        arguments.model_path,
        deterministic=arguments.deterministic,
        legacy=arguments.legacy,
    )
    print_results(evaluation_results)
