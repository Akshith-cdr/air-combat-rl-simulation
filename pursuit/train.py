"""Train the pursuit policy with PPO from scratch: one run, one seed.

    python -m pursuit.train --seed 1 --out runs/train/seed1

There is no curriculum, no warm start and no earlier stage: the run starts
from randomly initialised weights in the final task (same launch gate,
collision check and reward as evaluation). Every --eval-every steps the
current policy is saved and flown deterministically on the 48-episode
development suite; that gives the learning curve (hit rate vs. steps).

Output directory
  run.json                 provenance (git commit, clean tree, versions),
                           task config, PPO hyperparameters, timings
  model.zip, vecnormalize.pkl      final policy
  checkpoints/<steps>/     snapshots used for the learning curve
  learning_curve.csv       steps, dev hit rate, dev outcome counts
  monitor/*.monitor.csv    every training episode: return, length, outcome
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pursuit import provenance

HYPERPARAMETERS = {
    "total_timesteps": 1_500_000,
    "n_envs": 3,
    "n_steps": 2048,            # per environment
    "batch_size": 256,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "net_arch": [64, 64],
    "normalize_observations": True,
    "normalize_rewards": True,
    "clip_obs": 10.0,
    "eval_every": 98_304,       # 16 rollouts of 3 x 2048 steps
    "eval_suite": "dev",
}


def make_env(seed, rank, monitor_dir):
    def thunk():
        from pursuit.env import PursuitEnv
        env = PursuitEnv(train_seed=seed * 1000 + rank)
        env = Monitor(env, str(monitor_dir / f"env{rank}"), info_keywords=("outcome",))
        env.reset(seed=seed * 1000 + rank)
        return env
    return thunk


class LearningCurve(BaseCallback):
    """Save a snapshot and fly it on the development suite every `every` steps."""

    def __init__(self, out, every, suite):
        super().__init__()
        self.out, self.every, self.suite = Path(out), every, suite
        self.next_eval = 0
        self.rows = []

    def _on_training_start(self):
        self._evaluate()                          # step 0: untrained policy

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self._evaluate()
        return True

    def _evaluate(self):
        from pursuit.evaluate import evaluate
        directory = self.out / "checkpoints" / f"{self.num_timesteps:08d}"
        save_policy(self.model, directory)
        start = time.time()
        _, summary = evaluate({"controller": "ppo", "model": str(directory),
                               "out": str(directory / "dev_eval")}, self.suite, workers=1)
        row = {"timesteps": self.num_timesteps, "dev_episodes": summary["episodes"],
               "dev_hits": summary["hits"], "dev_hit_rate": summary["hit_rate"],
               "dev_outcomes": json.dumps(summary["outcomes"], sort_keys=True),
               "eval_seconds": round(time.time() - start, 1)}
        self.rows.append(row)
        with open(self.out / "learning_curve.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"[curve] {self.num_timesteps:>9d} steps: dev {summary['hits']}/{summary['episodes']} "
              f"{summary['outcomes']}", flush=True)
        while self.next_eval <= self.num_timesteps:
            self.next_eval += self.every


def save_policy(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "model.zip")
    model.get_vec_normalize_env().save(str(directory / "vecnormalize.pkl"))
    from pursuit.baselines import export_policy
    export_policy(directory)


def train(seed, out, hp, prov):
    out = Path(out)
    (out / "monitor").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    set_random_seed(seed)
    # All environments of a run share one process (see baselines.export_policy for why).
    venv = DummyVecEnv([make_env(seed, rank, out / "monitor") for rank in range(hp["n_envs"])])
    venv.seed(seed * 1000)
    venv = VecNormalize(venv, norm_obs=hp["normalize_observations"],
                        norm_reward=hp["normalize_rewards"], clip_obs=hp["clip_obs"], gamma=hp["gamma"])
    model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0,
                n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
                learning_rate=hp["learning_rate"], gamma=hp["gamma"], gae_lambda=hp["gae_lambda"],
                clip_range=hp["clip_range"], ent_coef=hp["ent_coef"], vf_coef=hp["vf_coef"],
                max_grad_norm=hp["max_grad_norm"],
                policy_kwargs={"net_arch": hp["net_arch"]})
    run = {"seed": seed, "hyperparameters": hp, "provenance": prov,
           "reportable": not prov["git"]["dirty"], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    provenance.write_json(out / "run.json", run)
    start = time.time()
    try:
        model.learn(total_timesteps=hp["total_timesteps"],
                    callback=LearningCurve(out, hp["eval_every"], hp["eval_suite"]))
        save_policy(model, out)
    finally:
        venv.close()
    run.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), wall_seconds=round(time.time() - start),
               num_timesteps=model.num_timesteps,
               final_sha256={name: provenance.sha256(out / name)
                             for name in ("model.zip", "vecnormalize.pkl", "policy_numpy.npz")})
    provenance.write_json(out / "run.json", run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, help="override total_timesteps (pilot runs only)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="train from uncommitted code (run is marked non-reportable)")
    args = parser.parse_args(argv)
    prov = provenance.record(args.allow_dirty)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} already exists and is not empty")
    hp = dict(HYPERPARAMETERS)
    if args.steps:
        hp["total_timesteps"] = args.steps
    train(args.seed, args.out, hp, prov)


if __name__ == "__main__":
    main()
