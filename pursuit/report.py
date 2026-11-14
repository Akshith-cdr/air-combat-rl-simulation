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
  paired_tests.csv              exact McNemar test, each PPO seed vs each baseline, same episodes
  sweep_curves.csv              full-sweep hit rate vs. steps (from pursuit.sweep_curves)
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


def mcnemar_p(b, c):
    """Exact two-sided McNemar test on the discordant pairs b, c."""
    from math import comb
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_tests(evals, out):
    """PPO seed vs baseline on identical episodes (same scenario and seed).

    ppo_only = episodes PPO hit and the baseline missed; base_only = the reverse.
    """
    rows = []
    for suite in ("reviewed_six", "sweep"):
        for baseline in ("lead_pursuit", "pure_pursuit"):
            base_path = Path(evals) / f"{baseline}_{suite}" / "episodes.csv"
            if not base_path.exists():
                continue
            base = {(e["scenario_id"], e["seed"]): e["success"] == "True" for e in read_csv(base_path)}
            for path in sorted(Path(evals).glob(f"ppo_seed*_{suite}/episodes.csv")):
                ppo = {(e["scenario_id"], e["seed"]): e["success"] == "True" for e in read_csv(path)}
                if set(ppo) != set(base):
                    raise ValueError(f"{path} and {base_path} do not contain the same episodes")
                ppo_only = sum(ppo[k] and not base[k] for k in ppo)
                base_only = sum(base[k] and not ppo[k] for k in ppo)
                rows.append({"suite": suite, "baseline": baseline, "ppo_run": path.parent.name,
                             "episodes": len(ppo), "ppo_only_hits": ppo_only, "baseline_only_hits": base_only,
                             "mcnemar_p": mcnemar_p(ppo_only, base_only)})
    if rows:
        with open(out / "paired_tests.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def sweep_curves(evals, out):
    """Mean / min / max over seeds of the full-sweep learning curves."""
    files = sorted((Path(evals) / "curves").glob("seed*.csv"))
    if not files:
        return
    curves = {f.stem: {int(r["timesteps"]): float(r["hit_rate"]) for r in read_csv(f)} for f in files}
    steps = sorted(set.intersection(*(set(c) for c in curves.values())))
    table = [{"timesteps": s, "mean": st.mean(c[s] for c in curves.values()),
              "min": min(c[s] for c in curves.values()), "max": max(c[s] for c in curves.values()),
              **{name: c[s] for name, c in curves.items()}} for s in steps]
    with open(out / "sweep_curves.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs/final/train"))
    parser.add_argument("--evals", type=Path, default=Path("runs/final/eval"))
    parser.add_argument("--out", type=Path, default=Path("runs/final/report"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    learning_curves(args.runs, args.out)
    comparison(args.evals, args.out)
    paired_tests(args.evals, args.out)
    sweep_curves(args.evals, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
