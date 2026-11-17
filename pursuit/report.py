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
        lm = summary.get("launch_metrics") or {}
        by_class = lm.get("by_aspect_class", {})
        rows.append({
            "run": path.parent.name, "controller": summary["controller"], "suite": summary["suite"],
            "reportable": summary["reportable"], "episodes": summary["episodes"], "hits": summary["hits"],
            "hit_rate": summary["hit_rate"], "ci_low": summary["ci95"][0], "ci_high": summary["ci95"][1],
            "collisions": summary["outcomes"].get("collision", 0),
            "launches": lm.get("launches", 0),
            "median_launch_time_s": lm.get("launch_time_s", {}).get("median", ""),
            "median_launch_range_m": lm.get("launch_range_m", {}).get("median", ""),
            "median_launch_aspect_deg": lm.get("launch_aspect_deg", {}).get("median", ""),
            "favourable_share": lm.get("favourable_share", ""),
            **{f"launches_{c}": by_class.get(c, {}).get("launches", 0) for c in ("head_on", "beam", "tail")},
            **{f"hit_rate_{c}": by_class.get(c, {}).get("hit_rate", "") for c in ("head_on", "beam", "tail")},
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


def sign_test_p(wins, losses):
    """Exact two-sided sign test (ties dropped); same formula as McNemar."""
    return mcnemar_p(wins, losses)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def _episodes(path):
    return {(e["scenario_id"], e["seed"]): e for e in read_csv(path)}


def paired_row(suite, a_name, a_path, b_name, b_path):
    """A vs B on identical episodes (same scenario and simulator seed).

    a_only = episodes A hit and B missed; b_only = the reverse (McNemar).
    Launch time: over episodes where both launched, how often A launched earlier
    (sign test) and the median of the per-episode difference A - B.
    """
    a, b = _episodes(a_path), _episodes(b_path)
    if set(a) != set(b):
        raise ValueError(f"{a_path} and {b_path} do not contain the same episodes")
    hit = lambda e: e["success"] == "True"
    a_only = sum(hit(a[k]) and not hit(b[k]) for k in a)
    b_only = sum(hit(b[k]) and not hit(a[k]) for k in a)
    both = [k for k in a if a[k]["launched"] == "True" and b[k]["launched"] == "True"]
    diffs = [float(a[k]["launch_time_s"]) - float(b[k]["launch_time_s"]) for k in both]
    earlier, later = sum(d < 0 for d in diffs), sum(d > 0 for d in diffs)
    return {"suite": suite, "a": a_name, "b": b_name, "episodes": len(a),
            "a_hits": sum(hit(e) for e in a.values()), "b_hits": sum(hit(e) for e in b.values()),
            "a_only_hits": a_only, "b_only_hits": b_only, "mcnemar_p": mcnemar_p(a_only, b_only),
            "both_launched": len(both), "a_launch_earlier": earlier, "a_launch_later": later,
            "median_launch_time_diff_s": st.median(diffs) if diffs else "",
            "launch_time_sign_p": sign_test_p(earlier, later)}


def paired_tests(evals, out):
    """Every PPO seed vs every baseline (M5), and the speed-swap controls (M4).

    Families for the Holm correction: all PPO-vs-baseline tests of one suite.
    """
    from pursuit.baselines import BASELINES
    evals = Path(evals)
    rows = []
    for suite in ("reviewed_six", "sweep", "sweep_alt"):
        family = []
        for baseline in BASELINES:
            base_path = evals / f"{baseline}_{suite}" / "episodes.csv"
            if not base_path.exists():
                continue
            for path in sorted(evals.glob(f"ppo_seed*_{suite}/episodes.csv")):
                family.append(paired_row(suite, path.parent.name[:-len(suite) - 1], path, baseline, base_path))
        for row, p in zip(family, holm([r["mcnemar_p"] for r in family])):
            row["family"], row["mcnemar_p_holm"] = "ppo_vs_baseline", p
        rows += family
        # M4: PPO with the rule's throttle vs PPO; the rule's steering with PPO's throttle vs the rule.
        for path in sorted(evals.glob(f"ppo_rule_speed_seed*_{suite}/episodes.csv")):
            seed = path.parent.name[len("ppo_rule_speed_"):-len(suite) - 1]
            ppo = evals / f"ppo_{seed}_{suite}" / "episodes.csv"
            if ppo.exists():
                rows.append({**paired_row(suite, f"ppo_rule_speed_{seed}", path, f"ppo_{seed}", ppo),
                             "family": "speed_swap"})
        for path in sorted(evals.glob(f"rule_steer_ppo_speed_seed*_{suite}/episodes.csv")):
            seed = path.parent.name[len("rule_steer_ppo_speed_"):-len(suite) - 1]
            rule = json.loads((path.parent / "summary.json").read_text())["rule_baseline"]
            base = evals / f"{rule}_{suite}" / "episodes.csv"
            if base.exists():
                rows.append({**paired_row(suite, f"rule_steer_ppo_speed_{seed}", path, rule, base),
                             "family": "speed_swap"})
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / "paired_tests.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
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
