"""Create an untrained hybrid PPO checkpoint from the legacy randomized model."""

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fighter_jet_env import FighterJetEnv
from hybrid_ppo import HybridActorCriticPolicy


SOURCE_MODEL = Path("models/ppo_ucav_100k_19d_5a_weapon_randomized_training.zip")
SOURCE_STATS = Path(
    "models/ppo_ucav_100k_19d_5a_weapon_randomized_training_vecnormalize.pkl"
)
TARGET_MODEL = Path(
    "models/ppo_ucav_100k_19d_hybrid_fire_randomized_transfer_init.zip"
)
TRANSFER_SEED = 123


def normalization_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_vecnormalize.pkl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_hybrid_env_from_stats(stats_path: Path):
    vector_env = DummyVecEnv([lambda: FighterJetEnv(hybrid_fire=True)])
    env = VecNormalize.load(str(stats_path), vector_env)
    # VecNormalize restores the source wrapper's legacy action space along with
    # its observation statistics. The wrapped hybrid environment is authoritative.
    env.action_space = vector_env.action_space
    env.training = False
    env.norm_reward = False
    return env


def transfer_legacy_policy(source_model: PPO, target_model: PPO) -> None:
    """Copy every compatible parameter except the legacy Gaussian fire output."""
    source = source_model.policy
    target = target_model.policy
    expected_shapes = {
        "source action mean": tuple(source.action_net.weight.shape) == (5, 64),
        "source log_std": tuple(source.log_std.shape) == (5,),
        "target action mean": tuple(target.action_net.weight.shape) == (4, 64),
        "target log_std": tuple(target.log_std.shape) == (4,),
        "target fire logit": tuple(target.fire_net.weight.shape) == (1, 64),
    }
    failures = [name for name, valid in expected_shapes.items() if not valid]
    if failures:
        raise ValueError(f"Unexpected transfer architecture: {failures}")

    with th.no_grad():
        target.features_extractor.load_state_dict(source.features_extractor.state_dict())
        target.mlp_extractor.policy_net.load_state_dict(
            source.mlp_extractor.policy_net.state_dict()
        )
        target.mlp_extractor.value_net.load_state_dict(
            source.mlp_extractor.value_net.state_dict()
        )
        target.value_net.load_state_dict(source.value_net.state_dict())
        target.action_net.weight.copy_(source.action_net.weight[:4])
        target.action_net.bias.copy_(source.action_net.bias[:4])
        target.log_std.copy_(source.log_std[:4])
        # target.fire_net deliberately retains its fresh orthogonal initialization.


def create_transfer_checkpoint(
    source_model_path: Path = SOURCE_MODEL,
    source_stats_path: Path = SOURCE_STATS,
    target_model_path: Path = TARGET_MODEL,
) -> None:
    paths = (source_model_path, source_stats_path)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing transfer source(s): " + ", ".join(missing))

    target_stats_path = normalization_path(target_model_path)
    if target_model_path.exists() or target_stats_path.exists():
        raise FileExistsError(
            "Refusing to overwrite hybrid initialization output: "
            f"{target_model_path} or {target_stats_path}"
        )

    source_hashes = (sha256(source_model_path), sha256(source_stats_path))
    np.random.seed(TRANSFER_SEED)
    th.manual_seed(TRANSFER_SEED)
    source_model = PPO.load(str(source_model_path), device="cpu")
    env = make_hybrid_env_from_stats(source_stats_path)
    try:
        target_model = PPO(
            HybridActorCriticPolicy,
            env,
            verbose=1,
            device="cpu",
            seed=TRANSFER_SEED,
        )
        transfer_legacy_policy(source_model, target_model)
        target_model_path.parent.mkdir(parents=True, exist_ok=True)
        target_model.save(str(target_model_path))
        shutil.copy2(source_stats_path, target_stats_path)
    finally:
        env.venv.envs[0].sim.destroy()
        env.close()

    if source_hashes != (sha256(source_model_path), sha256(source_stats_path)):
        raise RuntimeError("A transfer source file changed unexpectedly")
    print(f"Created hybrid transfer initialization: {target_model_path}")
    print(f"Copied unchanged VecNormalize statistics: {target_stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a hybrid PPO checkpoint without training."
    )
    parser.add_argument("--source-model", type=Path, default=SOURCE_MODEL)
    parser.add_argument("--source-vecnormalize", type=Path, default=SOURCE_STATS)
    parser.add_argument("--target-model", type=Path, default=TARGET_MODEL)
    arguments = parser.parse_args()
    create_transfer_checkpoint(
        arguments.source_model,
        arguments.source_vecnormalize,
        arguments.target_model,
    )
