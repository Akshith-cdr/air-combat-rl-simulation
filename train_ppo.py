import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fighter_jet_env import FighterJetEnv


DEFAULT_TOTAL_TIMESTEPS = 5000
DEFAULT_MODEL_PATH = "models/ppo_ucav_smoke_test"
DEFAULT_RANDOMIZED_ENEMY_TIMESTEPS = 100000
DEFAULT_RANDOMIZED_ENEMY_MODEL_PATH = (
    "models/ppo_ucav_100k_19d_5a_weapon_randomized_training"
)
DEFAULT_RANDOMIZED_ENEMY_INIT_MODEL = Path(
    "models/ppo_ucav_100k_19d_5a_weapon_fire_reward.zip"
)
DEFAULT_RANDOMIZED_ENEMY_INIT_STATS = Path(
    "models/ppo_ucav_100k_19d_5a_weapon_fire_reward_vecnormalize.pkl"
)
DEFAULT_TRANSFER_SOURCE_MODEL = Path("models/ppo_ucav_100k_18d_normalized.zip")
DEFAULT_TRANSFER_SOURCE_STATS = Path(
    "models/ppo_ucav_100k_18d_normalized_vecnormalize.pkl"
)
DEFAULT_TRANSFER_TARGET_MODEL = Path(
    "models/ppo_ucav_18d4a_to_19d5a_weapon_transfer_init"
)
TRANSFER_SEED = 123
FIRE_ACTION_BIAS = -1.0


def normalization_path(model_path):
    return model_path.with_name(f"{model_path.stem}_vecnormalize.pkl")


def make_vec_env(randomized_enemy=False):
    env_factory = (
        (lambda: FighterJetEnv(randomized_enemy_training=True))
        if randomized_enemy
        else FighterJetEnv
    )
    return VecNormalize(
        DummyVecEnv([env_factory]),
        norm_obs=True,
        norm_reward=False,
    )


def make_vec_env_from_stats(stats_path, randomized_enemy=False):
    env_factory = (
        (lambda: FighterJetEnv(randomized_enemy_training=True))
        if randomized_enemy
        else FighterJetEnv
    )
    env = VecNormalize.load(str(stats_path), DummyVecEnv([env_factory]))
    env.training = True
    env.norm_reward = False
    return env


def _load_normalization_statistics(stats_path, expected_observation_dim=18):
    """Read source normalization statistics without attaching them to the new env."""
    with Path(stats_path).open("rb") as source_file:
        source_stats = pickle.load(source_file)
    if not source_stats.norm_obs or source_stats.norm_reward:
        raise ValueError("Source VecNormalize must use norm_obs=True and norm_reward=False")
    if source_stats.obs_rms.mean.shape != (expected_observation_dim,):
        raise ValueError(
            "Unexpected VecNormalize observation dimension: "
            f"{source_stats.obs_rms.mean.shape}"
        )
    return source_stats


def _transfer_policy_parameters(source_model, target_model):
    """Transfer compatible 18-D/4-action weights into a 19-D/5-action policy."""
    source_policy = source_model.policy
    target_policy = target_model.policy

    source_actor_first = source_policy.mlp_extractor.policy_net[0]
    source_critic_first = source_policy.mlp_extractor.value_net[0]
    target_actor_first = target_policy.mlp_extractor.policy_net[0]
    target_critic_first = target_policy.mlp_extractor.value_net[0]

    expected_shapes = {
        "source actor first": tuple(source_actor_first.weight.shape) == (64, 18),
        "source critic first": tuple(source_critic_first.weight.shape) == (64, 18),
        "target actor first": tuple(target_actor_first.weight.shape) == (64, 19),
        "target critic first": tuple(target_critic_first.weight.shape) == (64, 19),
        "source action head": tuple(source_policy.action_net.weight.shape) == (4, 64),
        "target action head": tuple(target_policy.action_net.weight.shape) == (5, 64),
    }
    failed = [name for name, valid in expected_shapes.items() if not valid]
    if failed:
        raise ValueError(f"Unexpected policy architecture for transfer: {failed}")

    with torch.no_grad():
        target_actor_first.weight[:, :18].copy_(source_actor_first.weight)
        target_actor_first.weight[:, 18].zero_()
        target_actor_first.bias.copy_(source_actor_first.bias)
        target_critic_first.weight[:, :18].copy_(source_critic_first.weight)
        target_critic_first.weight[:, 18].zero_()
        target_critic_first.bias.copy_(source_critic_first.bias)

        target_policy.mlp_extractor.policy_net[2].load_state_dict(
            source_policy.mlp_extractor.policy_net[2].state_dict()
        )
        target_policy.mlp_extractor.value_net[2].load_state_dict(
            source_policy.mlp_extractor.value_net[2].state_dict()
        )
        target_policy.value_net.load_state_dict(source_policy.value_net.state_dict())

        target_policy.action_net.weight[:4].copy_(source_policy.action_net.weight)
        target_policy.action_net.bias[:4].copy_(source_policy.action_net.bias)
        target_policy.action_net.weight[4].zero_()
        target_policy.action_net.bias[4].fill_(FIRE_ACTION_BIAS)
        target_policy.log_std[:4].copy_(source_policy.log_std)
        # Leave target_policy.log_std[4] at the fresh PPO initialization.


def _transfer_observation_normalization(source_stats, target_env):
    """Expand 18-D source statistics while preserving every legacy feature exactly."""
    target_rms = target_env.obs_rms
    if target_rms.mean.shape != (19,):
        raise ValueError(f"Expected 19-D target normalization, got {target_rms.mean.shape}")

    target_rms.mean[:18] = source_stats.obs_rms.mean
    target_rms.var[:18] = source_stats.obs_rms.var
    target_rms.count = source_stats.obs_rms.count

    # target_speed is in m/s and shares the same 0..800 physical range as airspeed.
    # Reusing the learned airspeed RMS gives it a finite, non-saturating initial scale.
    target_rms.mean[18] = source_stats.obs_rms.mean[0]
    target_rms.var[18] = source_stats.obs_rms.var[0]


def create_transfer_initialized_model(
    source_model_path=DEFAULT_TRANSFER_SOURCE_MODEL,
    source_stats_path=DEFAULT_TRANSFER_SOURCE_STATS,
    target_model_path=DEFAULT_TRANSFER_TARGET_MODEL,
):
    """Create a 19-D/5-action checkpoint without PPO learning or old-state reuse."""
    source_model_path = Path(source_model_path)
    source_stats_path = Path(source_stats_path)
    target_model_path = Path(target_model_path)
    if not source_model_path.exists() or not source_stats_path.exists():
        raise FileNotFoundError("Transfer source model or VecNormalize statistics are missing")

    np.random.seed(TRANSFER_SEED)
    torch.manual_seed(TRANSFER_SEED)
    source_model = PPO.load(str(source_model_path), device="cpu")
    source_stats = _load_normalization_statistics(source_stats_path)
    target_env = make_vec_env()
    try:
        target_env.seed(TRANSFER_SEED)
        target_model = PPO("MlpPolicy", target_env, verbose=1, device="cpu")
        _transfer_policy_parameters(source_model, target_model)
        _transfer_observation_normalization(source_stats, target_env)

        target_model_path.parent.mkdir(parents=True, exist_ok=True)
        target_model.save(str(target_model_path))
        target_env.save(str(normalization_path(target_model_path)))
    finally:
        target_env.venv.envs[0].sim.destroy()
        target_env.close()


def validate_transfer_initialized_model(
    source_model_path=DEFAULT_TRANSFER_SOURCE_MODEL,
    source_stats_path=DEFAULT_TRANSFER_SOURCE_STATS,
    target_model_path=DEFAULT_TRANSFER_TARGET_MODEL,
):
    """Validate all required transfer invariants without stepping or learning."""
    source_model = PPO.load(str(source_model_path), device="cpu")
    target_model = PPO.load(str(target_model_path), device="cpu")
    source_stats = _load_normalization_statistics(source_stats_path)
    target_stats_path = normalization_path(Path(target_model_path))
    target_stats = _load_normalization_statistics(target_stats_path, 19)

    source_policy = source_model.policy
    target_policy = target_model.policy
    assert target_model.observation_space.shape == (19,)
    assert target_model.action_space.shape == (5,)
    assert torch.equal(
        target_policy.mlp_extractor.policy_net[0].weight[:, :18],
        source_policy.mlp_extractor.policy_net[0].weight,
    )
    assert torch.equal(
        target_policy.mlp_extractor.policy_net[0].bias,
        source_policy.mlp_extractor.policy_net[0].bias,
    )
    assert torch.equal(
        target_policy.mlp_extractor.value_net[0].weight[:, :18],
        source_policy.mlp_extractor.value_net[0].weight,
    )
    assert torch.equal(
        target_policy.mlp_extractor.value_net[0].bias,
        source_policy.mlp_extractor.value_net[0].bias,
    )
    assert torch.count_nonzero(target_policy.mlp_extractor.policy_net[0].weight[:, 18]) == 0
    assert torch.count_nonzero(target_policy.mlp_extractor.value_net[0].weight[:, 18]) == 0
    assert target_policy.mlp_extractor.policy_net[2].state_dict().keys() == (
        source_policy.mlp_extractor.policy_net[2].state_dict().keys()
    )
    assert target_policy.mlp_extractor.value_net[2].state_dict().keys() == (
        source_policy.mlp_extractor.value_net[2].state_dict().keys()
    )
    for key, value in source_policy.mlp_extractor.policy_net[2].state_dict().items():
        assert torch.equal(target_policy.mlp_extractor.policy_net[2].state_dict()[key], value)
    for key, value in source_policy.mlp_extractor.value_net[2].state_dict().items():
        assert torch.equal(target_policy.mlp_extractor.value_net[2].state_dict()[key], value)
    for key, value in source_policy.value_net.state_dict().items():
        assert torch.equal(target_policy.value_net.state_dict()[key], value)
    assert torch.equal(target_policy.action_net.weight[:4], source_policy.action_net.weight)
    assert torch.equal(target_policy.action_net.bias[:4], source_policy.action_net.bias)
    assert torch.count_nonzero(target_policy.action_net.weight[4]) == 0
    assert float(target_policy.action_net.bias[4].detach()) < 0.0
    assert torch.equal(target_policy.log_std[:4], source_policy.log_std)
    assert target_stats.obs_rms.mean.shape == (19,)
    assert target_stats.obs_rms.var.shape == (19,)
    assert np.array_equal(target_stats.obs_rms.mean[:18], source_stats.obs_rms.mean)
    assert np.array_equal(target_stats.obs_rms.var[:18], source_stats.obs_rms.var)
    assert target_stats.obs_rms.count == source_stats.obs_rms.count
    assert target_stats.norm_obs and not target_stats.norm_reward
    assert target_stats.clip_obs == 10.0
    return {
        "observation_shape": target_model.observation_space.shape,
        "action_shape": target_model.action_space.shape,
        "fire_bias": float(target_policy.action_net.bias[4].detach()),
        "fire_log_std": float(target_policy.log_std[4].detach()),
        "target_speed_mean": float(target_stats.obs_rms.mean[18]),
        "target_speed_var": float(target_stats.obs_rms.var[18]),
    }


def main(
    total_timesteps,
    model_path,
    init_model_path=None,
    init_vecnormalize_path=None,
    randomized_enemy=False,
):
    model_path = Path(model_path)
    if (init_model_path is None) != (init_vecnormalize_path is None):
        raise ValueError("Both init model and init VecNormalize paths must be supplied together")
    if randomized_enemy:
        model_archive_path = (
            model_path if model_path.suffix == ".zip" else Path(f"{model_path}.zip")
        )
        stats_output_path = normalization_path(model_path)
        existing_outputs = [
            path for path in (model_archive_path, stats_output_path) if path.exists()
        ]
        if existing_outputs:
            raise FileExistsError(
                "Randomized-enemy training will not overwrite existing outputs: "
                + ", ".join(str(path) for path in existing_outputs)
            )
    env = (
        make_vec_env(randomized_enemy=randomized_enemy)
        if init_model_path is None
        else make_vec_env_from_stats(
            init_vecnormalize_path, randomized_enemy=randomized_enemy
        )
    )
    try:
        env.seed(123)
        observation = env.reset()
        if observation.shape != (1, 19):
            raise ValueError(f"Expected observation shape (1, 19), got {observation.shape}")
        if env.action_space.shape != (5,):
            raise ValueError(f"Expected action shape (5,), got {env.action_space.shape}")

        model = (
            PPO("MlpPolicy", env, verbose=1, device="cpu")
            if init_model_path is None
            else PPO.load(str(init_model_path), env=env, device="cpu")
        )
        model.learn(total_timesteps=total_timesteps)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        env.save(str(normalization_path(model_path)))
    finally:
        env.venv.envs[0].sim.destroy()
        env.close()

    if not Path(f"{model_path}.zip").exists():
        raise FileNotFoundError(f"Saved model was not found at {model_path}.zip")
    if not normalization_path(model_path).exists():
        raise FileNotFoundError(
            f"Saved normalization statistics were not found at {normalization_path(model_path)}"
        )

    verification_factory = (
        (lambda: FighterJetEnv(randomized_enemy_training=True))
        if randomized_enemy
        else FighterJetEnv
    )
    verification_env = VecNormalize.load(
        str(normalization_path(model_path)),
        DummyVecEnv([verification_factory]),
    )
    verification_env.training = False
    verification_env.norm_reward = False
    try:
        model = PPO.load(str(model_path), env=verification_env, device="cpu")
        verification_env.seed(123)
        observation = verification_env.reset()
        for step_index in range(10):
            action, _ = model.predict(observation, deterministic=True)
            if action.shape != (1, 5):
                raise ValueError(f"Expected predicted action shape (1, 5), got {action.shape}")
            if not np.isfinite(action).all():
                raise ValueError("Predicted action contains non-finite values")
            if not np.all((action >= -1.0) & (action <= 1.0)):
                raise ValueError("Predicted action is outside [-1, 1]")

            observation, reward, done, _ = verification_env.step(action)
            if observation.shape != (1, 19):
                raise ValueError(f"Step {step_index + 1}: invalid observation shape {observation.shape}")
            if not np.isfinite(observation).all() or not np.isfinite(reward):
                raise ValueError(f"Step {step_index + 1}: non-finite observation or reward")
            if done.shape != (1,):
                raise TypeError(f"Step {step_index + 1}: invalid done shape {done.shape}")
    finally:
        verification_env.venv.envs[0].sim.destroy()
        verification_env.close()

    print(f"PPO smoke test passed; model saved to {model_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a minimal PPO training smoke test.")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--model-path")
    parser.add_argument("--init-model-path", type=Path)
    parser.add_argument("--init-vecnormalize-path", type=Path)
    parser.add_argument(
        "--randomized-enemy",
        action="store_true",
        help="Train with continuously randomized enemy initial conditions.",
    )
    parser.add_argument("--create-transfer-init", action="store_true")
    parser.add_argument("--transfer-source-model", type=Path, default=DEFAULT_TRANSFER_SOURCE_MODEL)
    parser.add_argument("--transfer-source-vecnormalize", type=Path, default=DEFAULT_TRANSFER_SOURCE_STATS)
    parser.add_argument("--transfer-target-model", type=Path, default=DEFAULT_TRANSFER_TARGET_MODEL)
    args = parser.parse_args()
    total_timesteps = args.timesteps
    if total_timesteps is None:
        total_timesteps = (
            DEFAULT_RANDOMIZED_ENEMY_TIMESTEPS
            if args.randomized_enemy
            else DEFAULT_TOTAL_TIMESTEPS
        )
    model_path = args.model_path
    if model_path is None:
        model_path = (
            DEFAULT_RANDOMIZED_ENEMY_MODEL_PATH
            if args.randomized_enemy
            else DEFAULT_MODEL_PATH
        )
    init_model_path = args.init_model_path
    init_vecnormalize_path = args.init_vecnormalize_path
    if (
        args.randomized_enemy
        and init_model_path is None
        and init_vecnormalize_path is None
    ):
        init_model_path = DEFAULT_RANDOMIZED_ENEMY_INIT_MODEL
        init_vecnormalize_path = DEFAULT_RANDOMIZED_ENEMY_INIT_STATS
    if args.create_transfer_init:
        if args.randomized_enemy:
            parser.error("--randomized-enemy cannot be combined with --create-transfer-init")
        create_transfer_initialized_model(
            args.transfer_source_model,
            args.transfer_source_vecnormalize,
            args.transfer_target_model,
        )
        result = validate_transfer_initialized_model(
            args.transfer_source_model,
            args.transfer_source_vecnormalize,
            args.transfer_target_model,
        )
        print(f"Transfer initialization passed: {result}")
    else:
        main(
            total_timesteps,
            model_path,
            init_model_path,
            init_vecnormalize_path,
            randomized_enemy=args.randomized_enemy,
        )
