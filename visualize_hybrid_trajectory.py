"""Record and plot one successful hybrid-PPO challenge trajectory."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evaluate_ppo import FROZEN_ENEMY_CHALLENGE_MANIFEST, MAX_EVALUATION_STEPS
from fighter_jet_env import FighterJetEnv
from hybrid_ppo import HybridActorCriticPolicy


DEFAULT_MODEL = Path(
    "models/ppo_ucav_100k_19d_hybrid_fire_randomized_training.zip"
)
DEFAULT_SCENARIO = "REC-003"
DEFAULT_OUTPUT_DIR = Path("visualizations_hybrid_REC003")
PLOT_DPI = 300


class PostStepCapture(gym.Wrapper):
    """Retain terminal positions before DummyVecEnv performs its automatic reset."""

    def __init__(self, env):
        super().__init__(env)
        self.last_transition = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        player = self.env.sim.player
        enemy = self.env.sim.enemy
        self.last_transition = {
            "observation": np.asarray(observation, dtype=np.float32).copy(),
            "ucav_position": (
                float(player.position.x),
                float(player.position.y),
                float(player.position.z),
            ),
            "enemy_position": (
                float(enemy.position.x),
                float(enemy.position.y),
                float(enemy.position.z),
            ),
            "ucav_airspeed": float(player.airspeed),
            "target_speed": float(enemy.airspeed),
        }
        return observation, reward, terminated, truncated, info


def normalization_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_vecnormalize.pkl")


def scenario_by_id(scenario_id: str):
    matches = [
        scenario
        for scenario in FROZEN_ENEMY_CHALLENGE_MANIFEST
        if scenario.scenario_id == scenario_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate challenge scenario: {scenario_id}")
    return matches[0]


def make_environment(stats_path: Path):
    vector_env = DummyVecEnv(
        [lambda: PostStepCapture(FighterJetEnv(hybrid_fire=True))]
    )
    env = VecNormalize.load(str(stats_path), vector_env)
    env.action_space = vector_env.action_space
    env.training = False
    env.norm_reward = False
    return env


def _initial_record(raw_env, raw_observation):
    player = raw_env.sim.player
    enemy = raw_env.sim.enemy
    return {
        "step": 0,
        "time_s": 0.0,
        "ucav_x": float(player.position.x),
        "ucav_y": float(player.position.y),
        "ucav_z": float(player.position.z),
        "enemy_x": float(enemy.position.x),
        "enemy_y": float(enemy.position.y),
        "enemy_z": float(enemy.position.z),
        "range_m": float(raw_observation[10]),
        "cos_ata": float(raw_observation[11]),
        "ucav_airspeed_mps": float(player.airspeed),
        "target_speed_mps": float(enemy.airspeed),
        "action_pitch": 0.0,
        "action_roll": 0.0,
        "action_yaw": 0.0,
        "action_speed": 0.0,
        "fire_action": 0,
        "inside_firing_gate": bool(
            raw_observation[10] < 100.0 and raw_observation[11] > 0.95
        ),
        "fire_request": False,
        "valid_fire_attempt": False,
        "invalid_fire_attempt": False,
        "missile_launch": False,
        "missile_hit": False,
        "terminal_reason": "",
    }


def record_successful_rollout(
    model_path: Path,
    scenario,
    simulation_seed: int,
    deterministic: bool,
    policy_seed: int,
    max_steps: int,
):
    stats_path = normalization_path(model_path)
    missing = [str(path) for path in (model_path, stats_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing visualization input: " + ", ".join(missing))

    np.random.seed(policy_seed)
    th.manual_seed(policy_seed)
    env = make_environment(stats_path)
    try:
        model = PPO.load(str(model_path), env=env, device="cpu")
        if not isinstance(model.policy, HybridActorCriticPolicy):
            raise TypeError("The supplied model is not a hybrid PPO policy")
        if model.observation_space.shape != (19,) or model.action_space.shape != (5,):
            raise ValueError("Expected a 19-D observation and flat 5-D hybrid action")
        if tuple(model.policy.action_net.weight.shape) != (4, 64):
            raise ValueError("Hybrid policy must have four Gaussian action means")
        if tuple(model.policy.log_std.shape) != (4,):
            raise ValueError("Hybrid policy must have four Gaussian log standard deviations")
        if tuple(model.policy.fire_net.weight.shape) != (1, 64):
            raise ValueError("Hybrid policy must have one Bernoulli fire logit")

        reset_scenario = asdict(scenario)
        reset_scenario["seed"] = int(simulation_seed)
        env.set_options({"enemy_challenge_scenario": reset_scenario})
        env.seed(simulation_seed)
        observation = env.reset()
        capture = env.venv.envs[0]
        raw_env = capture.unwrapped
        raw_observation = env.get_original_obs()[0].copy()
        records = [_initial_record(raw_env, raw_observation)]
        time_s = 0.0
        terminal_reason = "visualization_step_limit"

        for step in range(1, max_steps + 1):
            action, _ = model.predict(observation, deterministic=deterministic)
            action_values = np.asarray(action, dtype=np.float32).reshape(-1)
            if action_values.shape != (5,):
                raise ValueError(f"Unexpected hybrid action shape: {action_values.shape}")
            if action_values[4] not in (0.0, 1.0):
                raise ValueError(f"Non-binary hybrid fire action: {action_values[4]}")

            observation, _, done, infos = env.step(action)
            info = infos[0]
            weapon = info.get("weapon", {})
            transition = capture.last_transition
            if transition is None:
                raise RuntimeError("Post-step state capture was not populated")
            raw_observation = transition["observation"]
            time_s += float(info.get("dt", 0.0))
            valid_fire = bool(weapon.get("valid_fire_attempt", False))
            missile_hit = bool(weapon.get("missile_hit_this_step", False))
            record = {
                "step": step,
                "time_s": time_s,
                "ucav_x": transition["ucav_position"][0],
                "ucav_y": transition["ucav_position"][1],
                "ucav_z": transition["ucav_position"][2],
                "enemy_x": transition["enemy_position"][0],
                "enemy_y": transition["enemy_position"][1],
                "enemy_z": transition["enemy_position"][2],
                "range_m": float(raw_observation[10]),
                "cos_ata": float(raw_observation[11]),
                "ucav_airspeed_mps": transition["ucav_airspeed"],
                "target_speed_mps": transition["target_speed"],
                "action_pitch": float(action_values[0]),
                "action_roll": float(action_values[1]),
                "action_yaw": float(action_values[2]),
                "action_speed": float(action_values[3]),
                "fire_action": int(action_values[4]),
                "inside_firing_gate": bool(
                    raw_observation[10] < 100.0 and raw_observation[11] > 0.95
                ),
                "fire_request": bool(weapon.get("fire_request", False)),
                "valid_fire_attempt": valid_fire,
                "invalid_fire_attempt": bool(
                    weapon.get("invalid_fire_attempt", False)
                ),
                "missile_launch": valid_fire,
                "missile_hit": missile_hit,
                "terminal_reason": str(info.get("terminal_reason", "")),
            }
            records.append(record)
            if bool(done[0]):
                terminal_reason = record["terminal_reason"] or "environment_terminated"
                break

        if terminal_reason != "missile_hit":
            raise RuntimeError(
                f"The recorded stochastic rollout did not hit the target: {terminal_reason}"
            )
        return records, terminal_reason
    finally:
        env.venv.envs[0].unwrapped.sim.destroy()
        env.close()


def write_csv(records, path: Path):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _plot_values(records):
    numeric_keys = (
        "time_s", "ucav_x", "ucav_y", "ucav_z", "enemy_x", "enemy_y",
        "enemy_z", "range_m", "cos_ata", "ucav_airspeed_mps",
        "target_speed_mps", "fire_action",
    )
    return {
        key: np.asarray([record[key] for record in records], dtype=float)
        for key in numeric_keys
    }


def _event_index(records, key):
    return np.flatnonzero([bool(record[key]) for record in records])


def _mark_event_lines(axis, records, values):
    launch = _event_index(records, "missile_launch")
    hit = _event_index(records, "missile_hit")
    if launch.size:
        axis.axvline(
            values["time_s"][launch[0]], color="#1597e5", linestyle="--",
            linewidth=1.8, label="Missile launch",
        )
    if hit.size:
        axis.axvline(
            values["time_s"][hit[0]], color="#7b2cbf", linestyle=":",
            linewidth=2.2, label="Missile hit",
        )


def create_plots(records, output_dir: Path, scenario_id: str):
    values = _plot_values(records)
    launch = _event_index(records, "missile_launch")
    hit = _event_index(records, "missile_hit")
    prefix = f"{scenario_id}_hybrid"
    plt.rcParams.update(
        {"font.size": 11, "axes.titlesize": 15, "axes.labelsize": 12}
    )

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(values["ucav_x"], values["ucav_y"], values["ucav_z"],
              color="#1565c0", linewidth=2.2, label="UCAV")
    axis.plot(values["enemy_x"], values["enemy_y"], values["enemy_z"],
              color="#d84315", linewidth=2.2, label="Enemy")
    axis.scatter(*[values[key][0] for key in ("ucav_x", "ucav_y", "ucav_z")],
                 color="#1565c0", marker="o", s=85, label="UCAV start")
    axis.scatter(*[values[key][0] for key in ("enemy_x", "enemy_y", "enemy_z")],
                 color="#d84315", marker="o", s=85, label="Enemy start")
    if launch.size:
        index = launch[0]
        axis.scatter(*[values[key][index] for key in ("ucav_x", "ucav_y", "ucav_z")],
                     color="#00a6fb", marker="^", s=130, label="Missile launch")
    if hit.size:
        index = hit[0]
        axis.scatter(*[values[key][index] for key in ("enemy_x", "enemy_y", "enemy_z")],
                     color="#7b2cbf", marker="*", s=220, label="Missile hit")
    spans = [max(np.ptp(values[key]), 1.0) for key in ("ucav_x", "ucav_y", "ucav_z")]
    axis.set_box_aspect(spans)
    axis.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Altitude (m)",
             title=f"Hybrid PPO engagement trajectory — {scenario_id}")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    fig.savefig(output_dir / f"{prefix}_trajectory_3d.png", dpi=PLOT_DPI)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    axis.plot(values["time_s"], values["range_m"], color="#173f5f", linewidth=2)
    axis.axhline(100.0, color="#d62828", linestyle="--", label="Fire range: 100 m")
    _mark_event_lines(axis, records, values)
    axis.set(xlabel="Time (s)", ylabel="Range (m)", title="Target range versus time")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / f"{prefix}_range_vs_time.png", dpi=PLOT_DPI)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    axis.plot(values["time_s"], values["cos_ata"], color="#2a9d8f", linewidth=2)
    axis.axhline(0.95, color="#d62828", linestyle="--", label="Fire alignment: 0.95")
    _mark_event_lines(axis, records, values)
    axis.set(xlabel="Time (s)", ylabel="cos(ATA)", ylim=(-1.05, 1.05),
             title="Attack-to-target alignment versus time")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / f"{prefix}_cosATA_vs_time.png", dpi=PLOT_DPI)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    axis.plot(values["time_s"], values["ucav_airspeed_mps"],
              color="#1565c0", linewidth=2, label="UCAV airspeed")
    axis.plot(values["time_s"], values["target_speed_mps"],
              color="#d84315", linewidth=2, label="Enemy speed")
    _mark_event_lines(axis, records, values)
    axis.set(xlabel="Time (s)", ylabel="Speed (m/s)", title="Aircraft speed versus time")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / f"{prefix}_speed_vs_time.png", dpi=PLOT_DPI)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    axis.step(values["time_s"], values["fire_action"], where="post",
              color="#263238", linewidth=1.3, label="Binary fire action")
    gate = np.asarray([record["inside_firing_gate"] for record in records], dtype=float)
    axis.fill_between(values["time_s"], 0, gate, step="post", color="#f4d35e",
                      alpha=0.28, label="Inside firing gate")
    event_specs = (
        ("valid_fire_attempt", 1.10, "#00a6fb", "^", "Valid fire / launch"),
        ("invalid_fire_attempt", 1.20, "#ef233c", "x", "Invalid fire"),
        ("missile_hit", 1.30, "#7b2cbf", "*", "Missile hit"),
    )
    for key, level, color, marker, label in event_specs:
        indices = _event_index(records, key)
        if indices.size:
            axis.scatter(values["time_s"][indices], np.full(indices.size, level),
                         color=color, marker=marker, s=80 if marker != "*" else 160,
                         label=label, zorder=4)
    axis.set(xlabel="Time (s)", ylabel="Fire / event state", ylim=(-0.08, 1.42),
             yticks=[0, 1], title="Hybrid weapon decisions and engagement events")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, loc="upper right")
    fig.savefig(output_dir / f"{prefix}_weapon_events_vs_time.png", dpi=PLOT_DPI)
    plt.close(fig)


def save_outputs(records, output_dir: Path, scenario_id: str):
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / f"{scenario_id}_hybrid_trajectory.csv"
    write_csv(records, csv_path)
    create_plots(records, output_dir, scenario_id)
    expected = [csv_path, *output_dir.glob("*.png")]
    if len(expected) != 6 or not all(path.is_file() for path in expected):
        raise RuntimeError("Expected one CSV and five PNG visualization files")
    return csv_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--seed", type=int, help="Override the scenario simulation seed (default: manifest seed)."
    )
    parser.add_argument("--policy-seed", type=int, default=41003)
    parser.add_argument("--max-steps", type=int, default=MAX_EVALUATION_STEPS)
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use Gaussian means and Bernoulli mode instead of stochastic sampling.",
    )
    return parser


def main():
    arguments = build_parser().parse_args()
    if arguments.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {arguments.output_dir}"
        )
    scenario = scenario_by_id(arguments.scenario)
    simulation_seed = scenario.seed if arguments.seed is None else arguments.seed
    mode = "deterministic" if arguments.deterministic else "stochastic"
    print(f"Model: {arguments.model_path}")
    print(f"Scenario: {scenario.scenario_id}, simulation seed: {simulation_seed}")
    print(f"Policy action mode: {mode}, policy sampling seed: {arguments.policy_seed}")
    records, terminal_reason = record_successful_rollout(
        arguments.model_path,
        scenario,
        simulation_seed,
        arguments.deterministic,
        arguments.policy_seed,
        arguments.max_steps,
    )
    csv_path = save_outputs(records, arguments.output_dir, scenario.scenario_id)
    launch = next(record for record in records if record["missile_launch"])
    hit = next(record for record in records if record["missile_hit"])
    minimum_range = min(record["range_m"] for record in records)
    maximum_cos_ata = max(record["cos_ata"] for record in records)
    print(f"Output directory: {arguments.output_dir}")
    print(f"CSV: {csv_path}")
    print(f"Terminal reason: {terminal_reason}")
    print(f"Missile launch: step {launch['step']}, t={launch['time_s']:.3f} s")
    print(f"Missile hit: step {hit['step']}, t={hit['time_s']:.3f} s")
    print(f"Minimum range: {minimum_range:.3f} m")
    print(f"Maximum cosATA: {maximum_cos_ata:.6f}")


if __name__ == "__main__":
    main()
