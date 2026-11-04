"""Plot one engagement (3D flight paths + range over time) of a trained flight policy.

Runs the same episode as the evaluation, headless, for a given scenario seed
(the scenario_seed column of final_eval_1000.csv):

    python plot_engagement.py --seed 4016774293
    python plot_engagement.py --seed 4016774293 --model-path experiments/flight_policy/<run-id>/final/candidate.zip

Writes results/plots/engagement_<seed>.png (or --output-dir).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

import evaluate_visualize_hybrid_combat as evaluation  # noqa: E402


def record_positions(raw_env):
    """Record aircraft and missile positions after every simulator step."""
    samples = []
    original_step = raw_env.step

    def position(actor):
        return [float(actor.position[i]) for i in range(3)]

    def sample():
        sim = raw_env.sim
        missile = sim.active_missiles[0]["guided"].position if sim.active_missiles else None
        samples.append(dict(player=position(sim.player), enemy=position(sim.enemy),
                            missile=[float(missile[i]) for i in range(3)] if missile is not None else None))

    def step(action):
        result = original_step(action)
        sample()
        return result

    raw_env.step = step
    return samples, sample


def plot(samples, row, output):
    dt = evaluation.DEFAULT_SIMULATION_DT
    t = [i * dt for i in range(len(samples))]
    player = list(zip(*[s["player"] for s in samples]))
    enemy = list(zip(*[s["enemy"] for s in samples]))
    missile = [s["missile"] for s in samples if s["missile"] is not None]
    rng = [sum((a - b) ** 2 for a, b in zip(s["player"], s["enemy"])) ** 0.5 for s in samples]

    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*player, color="#2563eb", label="UCAV (PPO policy)")
    ax.plot(*enemy, color="#dc2626", label="Enemy")
    if missile:
        ax.plot(*zip(*missile), color="#f59e0b", linewidth=2, label="Missile")
        ax.scatter(*missile[-1], color="black", marker="x", s=60, label="Impact" if row["success"] else "Missile end")
    ax.scatter(*[c[0] for c in player], color="#2563eb", s=25)
    ax.scatter(*[c[0] for c in enemy], color="#dc2626", s=25)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("altitude (m)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("3D flight paths", fontsize=10)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, rng, color="#334155")
    for key, label, color in (("time_to_missile_launch_seconds", "launch", "#f59e0b"),
                              ("time_to_missile_hit_seconds", "hit", "#16a34a")):
        value = float(row[key])
        if value == value:  # not NaN
            ax2.axvline(value, color=color, linestyle="--", label=f"{label} ({value:.1f} s)")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("UCAV-to-enemy range (m)")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    ax2.set_title("Range to target", fontsize=10)

    fig.suptitle(f"{row['scenario_id']}  seed {row['scenario_seed']}  →  {row['terminal_reason']}", fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=4016774293, help="scenario_seed from final_eval_1000.csv")
    parser.add_argument("--model-path", type=Path, default=evaluation.MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    evaluation.select_model(args.model_path)
    evaluation.verify_inputs()
    environment = evaluation.make_environment(evaluation.VECNORMALIZE_PATH)
    try:
        model = PPO.load(str(evaluation.MODEL_PATH), env=environment, device="cpu")
        evaluation.validate_model(model, environment)
        raw_env = getattr(environment.venv.envs[0], "unwrapped", environment.venv.envs[0])
        samples, sample = record_positions(raw_env)
        original_reset = environment.reset

        def reset():
            observation = original_reset()
            samples.clear()
            sample()  # initial positions
            return observation

        environment.reset = reset
        row = evaluation.evaluate_episode(environment, model, 1, args.seed)
    finally:
        raw_env = getattr(environment.venv.envs[0], "unwrapped", environment.venv.envs[0])
        raw_env.sim.destroy()
        environment.close()

    output = args.output_dir / f"engagement_{args.seed}.png"
    plot(samples, row, output)
    print(f"{row['scenario_id']} seed {args.seed}: {row['terminal_reason']}, "
          f"launch at {float(row['time_to_missile_launch_seconds']):.2f} s, "
          f"range at launch {float(row['minimum_range_at_valid_fire']):.0f} m")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
