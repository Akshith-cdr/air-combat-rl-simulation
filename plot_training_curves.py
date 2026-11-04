"""Plot training curves of a Stage 5 or Stage 6 run from its logs/progress.csv.

    python plot_training_curves.py experiments/flight_policy/flight_v2_seed63011_run03
    python plot_training_curves.py experiments/turning_improvement/A_seed62001_run01

Writes <run>/training_curves.png (or --output).
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PANELS = (
    ("rollout/ep_rew_mean", "Mean episode reward"),
    ("rollout/ep_len_mean", "Mean episode length (steps)"),
    ("flight/episode_target_destroyed", "Target destroyed (fraction of episodes)"),
    ("train/explained_variance", "Value-function explained variance"),
)


def load(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    return rows


def series(rows, key):
    xs, ys = [], []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        y = float(value)
        if math.isfinite(y):
            xs.append(float(row["time/total_timesteps"]) / 1e6)
            ys.append(y)
    return xs, ys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Run directory containing logs/progress.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load(args.run / "logs" / "progress.csv")
    panels = [(key, title) for key, title in PANELS if series(rows, key)[0]]

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.6 * len(panels)), sharex=True)
    axes = [axes] if len(panels) == 1 else axes
    for ax, (key, title) in zip(axes, panels):
        xs, ys = series(rows, key)
        ax.plot(xs, ys, color="#93c5fd", linewidth=0.8, label="per update")
        window = max(1, len(ys) // 40)
        smooth = [sum(ys[max(0, i - window + 1):i + 1]) / len(ys[max(0, i - window + 1):i + 1])
                  for i in range(len(ys))]
        ax.plot(xs, smooth, color="#1d4ed8", linewidth=1.6, label=f"moving average ({window})")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[-1].set_xlabel("Training timesteps (millions)")
    fig.suptitle(f"Training curves: {args.run.name}", fontsize=12)
    fig.tight_layout()
    output = args.output or args.run / "training_curves.png"
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
