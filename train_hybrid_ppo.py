"""Separate training entry point for randomized-enemy hybrid-fire PPO."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fighter_jet_env import FighterJetEnv
from hybrid_ppo import HybridActorCriticPolicy


DEFAULT_TIMESTEPS = 100_000
DEFAULT_INIT_MODEL = Path(
    "models/ppo_ucav_100k_19d_hybrid_fire_randomized_transfer_init.zip"
)
DEFAULT_INIT_STATS = Path(
    "models/ppo_ucav_100k_19d_hybrid_fire_randomized_transfer_init_vecnormalize.pkl"
)
DEFAULT_OUTPUT_MODEL = Path(
    "models/ppo_ucav_100k_19d_hybrid_fire_randomized_training.zip"
)


def normalization_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_vecnormalize.pkl")


def make_hybrid_training_env(stats_path: Path):
    """Load observation statistics without retaining legacy action-space metadata."""
    vector_env = DummyVecEnv(
        [
            lambda: FighterJetEnv(
                randomized_enemy_training=True,
                hybrid_fire=True,
            )
        ]
    )
    env = VecNormalize.load(str(stats_path), vector_env)
    # VecNormalize pickles its old action_space. The statistics originated from a
    # legacy Box(5) environment, so restore the space from the new wrapped env.
    env.action_space = vector_env.action_space
    env.training = True
    env.norm_reward = False
    return env


def train(
    timesteps: int,
    init_model_path: Path,
    init_stats_path: Path,
    output_model_path: Path,
) -> None:
    if timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    missing = [
        str(path)
        for path in (init_model_path, init_stats_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing hybrid initialization: " + ", ".join(missing))

    output_stats_path = normalization_path(output_model_path)
    if output_model_path.exists() or output_stats_path.exists():
        raise FileExistsError(
            "Refusing to overwrite hybrid training output: "
            f"{output_model_path} or {output_stats_path}"
        )

    env = make_hybrid_training_env(init_stats_path)
    try:
        if env.observation_space.shape != (19,) or env.action_space.shape != (5,):
            raise ValueError("Hybrid training requires 19 observations and 5 actions")
        if not (
            (env.action_space.low == [-1.0, -1.0, -1.0, -1.0, 0.0]).all()
            and (env.action_space.high == [1.0, 1.0, 1.0, 1.0, 1.0]).all()
        ):
            raise ValueError("Hybrid fire transport must be bounded to [0, 1]")

        model = PPO.load(
            str(init_model_path),
            env=env,
            device="cpu",
            # The already-created transfer checkpoint inherited legacy bounds
            # from VecNormalize metadata. Its hybrid policy parameters are valid;
            # use the explicitly verified hybrid environment space when loading.
            custom_objects={"action_space": env.action_space},
        )
        if not isinstance(model.policy, HybridActorCriticPolicy):
            raise TypeError("Initialization checkpoint is not a hybrid PPO policy")
        model.learn(total_timesteps=timesteps)
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output_model_path))
        env.save(str(output_stats_path))
    finally:
        env.venv.envs[0].sim.destroy()
        env.close()

    print(f"Saved trained hybrid model: {output_model_path}")
    print(f"Saved matching VecNormalize statistics: {output_stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the separate hybrid-fire PPO.")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--init-model", type=Path, default=DEFAULT_INIT_MODEL)
    parser.add_argument("--init-vecnormalize", type=Path, default=DEFAULT_INIT_STATS)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_OUTPUT_MODEL)
    arguments = parser.parse_args()
    train(
        arguments.timesteps,
        arguments.init_model,
        arguments.init_vecnormalize,
        arguments.output_model,
    )
