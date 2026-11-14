"""Aggregate seeds and controllers into the paper's numbers and figures.

    python -m pursuit.report --runs runs/train --evals runs/eval --out runs/report

Expects
  runs/train/seed<N>/learning_curve.csv                  (from pursuit.train)
  runs/eval/<controller>_<suite>/summary.json, episodes.csv  (from pursuit.evaluate;
      PPO seeds are named ppo_seed<N>_<suite>)
Writes
  learning_curves.png / .csv    dev hit rate vs. steps: each seed, mean and min-max
  comparison.csv                hit rate, CI, launch time/range, collisions per controller
  sectors.csv                   hit rate per bearing sector per controller (sweep suite)
"""

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def learning_curves(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = {}
    for path in sorted(Path(runs).glob("seed*/learning_curve.csv")):
        rows = read_csv(path)
        curves[path.parent.name] = {int(r["timesteps"]): float(r["dev_hit_rate"]) for r in rows}
    if not curves:
        return
    steps = sorted(set.intersection(*(set(c) for c in curves.values())))
    table = []
    for s in steps:
        values = [c[s] for c in curves.values()]
        table.append({"timesteps": s, "mean": st.mean(values), "min": min(values), "max": max(values),
                      "std": st.pstdev(values), **{name: c[s] for name, c in curves.items()}})
    with open(out / "learning_curves.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = [r["timesteps"] / 1e6 for r in table]
    for name, c in curves.items():
        ax.plot(x, [100 * c[s] for s in steps], lw=0.8, alpha=0.5, label=name)
    ax.fill_between(x, [100 * r["min"] for r in table], [100 * r["max"] for r in table], alpha=0.15, color="k")
    ax.plot(x, [100 * r["mean"] for r in table], color="k", lw=2, label=f"mean of {len(curves)} seeds")
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel("development hit rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=200)


def comparison(evals, out):
    rows, sectors = [], defaultdict(dict)
    for path in sorted(Path(evals).glob("*/summary.json")):
        summary = json.loads(path.read_text())
        episodes = read_csv(path.parent / "episodes.csv")
        launched = [e for e in episodes if e["launched"] == "True"]
        rows.append({
            "run": path.parent.name, "controller": summary["controller"], "suite": summary["suite"],
            "reportable": summary["reportable"], "episodes": summary["episodes"], "hits": summary["hits"],
            "hit_rate": summary["hit_rate"], "ci_low": summary["ci95"][0], "ci_high": summary["ci95"][1],
            "collisions": summary["outcomes"].get("collision", 0),
            "median_launch_time_s": st.median(float(e["launch_time_s"]) for e in launched) if launched else "",
            "median_launch_range_m": st.median(float(e["launch_range_m"]) for e in launched) if launched else "",
            "outcomes": json.dumps(summary["outcomes"], sort_keys=True)})
        if summary["suite"] == "sweep":
            for sector, value in summary["by_bearing_sector"].items():
                sectors[int(sector)][path.parent.name] = value["hit_rate"]
    if rows:
        with open(out / "comparison.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if sectors:
        names = sorted({n for v in sectors.values() for n in v})
        with open(out / "sectors.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["bearing_sector_deg", *names])
            for sector in sorted(sectors):
                writer.writerow([sector, *(sectors[sector].get(n, "") for n in names)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs/train"))
    parser.add_argument("--evals", type=Path, default=Path("runs/eval"))
    parser.add_argument("--out", type=Path, default=Path("runs/report"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    learning_curves(args.runs, args.out)
    comparison(args.evals, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
