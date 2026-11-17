"""Evaluate one controller on one scenario suite.

    python -m pursuit.evaluate --controller pure_pursuit --suite sweep --out runs/eval/pure_sweep
    python -m pursuit.evaluate --controller ppo --model runs/train/seed1/final --suite reviewed_six ...

Writes
  episodes.csv          one row per episode (initial geometry + outcome)
  summary.json          hit rate with 95% Wilson CI, outcome counts, per-sector
                        and per-target-heading tables, provenance
  failures/<id>.csv     full time series of every episode that was not a hit
"""

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pursuit import provenance, scenarios

CONTROLLERS = ("ppo", "pure_pursuit", "lead_pursuit", "pn_pursuit", "ppo_rule_speed",
               "rule_steer_ppo_speed")


def wilson(successes, n, z=1.959964):
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def make_controller(spec):
    from pursuit import baselines
    if spec["controller"] == "ppo":
        return baselines.PPOController(spec["model"])
    if spec["controller"] in baselines.SWAPS:
        return baselines.make_swap(spec["controller"], spec["model"], spec.get("baseline"))
    gains = baselines.Gains(**spec["gains"]) if spec.get("gains") else None
    return baselines.make(spec["controller"], gains)


def run_chunk(spec, chunk):
    from pursuit.env import PursuitEnv
    controller = make_controller(spec)
    env = PursuitEnv(record_trajectory=True)
    rows = []
    try:
        for scenario in chunk:
            obs, _ = env.reset(options={"scenario": scenario})
            controller.reset()
            done = False
            while not done:
                obs, _, terminated, truncated, info = env.step(controller.act(env, obs))
                done = terminated or truncated
            row = {**scenario.as_dict(), **info["episode_summary"]}
            if not row["success"] and spec.get("save_failures", True):
                write_rows(Path(spec["out"]) / "failures" / f"{scenario.scenario_id}_{scenario.seed}.csv",
                           env.trajectory)
            rows.append(row)
    finally:
        env.close()
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    result = {}
    for value in sorted(groups):
        group = groups[value]
        hits = sum(r["success"] for r in group)
        low, high = wilson(hits, len(group))
        result[str(value)] = {"episodes": len(group), "hits": hits, "hit_rate": hits / len(group),
                              "ci95": [low, high],
                              "outcomes": dict(Counter(r["outcome"] for r in group))}
    return result


# Target aspect at launch (pursuit.env.launch_geometry): 0 = head-on, 180 = tail.
# "Favourable" = within 45 deg of head-on or of the tail, i.e. not a beam (crossing) shot.
FAVOURABLE_ASPECT_DEG = 45.0


def aspect_class(aspect_deg):
    if aspect_deg <= FAVOURABLE_ASPECT_DEG:
        return "head_on"
    if aspect_deg >= 180.0 - FAVOURABLE_ASPECT_DEG:
        return "tail"
    return "beam"


def launch_metrics(rows):
    """Where and when each controller launches (reviewer comment M6)."""
    import statistics as st
    launched = [r for r in rows if r["launched"]]
    if not launched:
        return {"launches": 0}
    times = [float(r["launch_time_s"]) for r in launched]
    ranges = [float(r["launch_range_m"]) for r in launched]
    aspects = [float(r["launch_aspect_deg"]) for r in launched]
    classes = Counter(aspect_class(a) for a in aspects)
    favourable = classes["head_on"] + classes["tail"]
    by_class = {}
    for name in ("head_on", "beam", "tail"):
        group = [r for r in launched if aspect_class(float(r["launch_aspect_deg"])) == name]
        hits = sum(r["success"] for r in group)
        by_class[name] = {"launches": len(group), "hits": hits,
                          "hit_rate": hits / len(group) if group else math.nan}
    q = lambda v, p: float(np.percentile(v, p))
    return {"launches": len(launched),
            "launch_time_s": {"median": st.median(times), "mean": st.mean(times),
                              "p25": q(times, 25), "p75": q(times, 75)},
            "launch_range_m": {"median": st.median(ranges), "mean": st.mean(ranges)},
            "launch_aspect_deg": {"median": st.median(aspects), "mean": st.mean(aspects)},
            "favourable_launches": favourable,
            "favourable_share": favourable / len(launched),
            "favourable_share_ci95": list(wilson(favourable, len(launched))),
            "by_aspect_class": by_class}


def summarize(rows):
    hits = sum(r["success"] for r in rows)
    low, high = wilson(hits, len(rows))
    launched = [r for r in rows if r["launched"]]
    return {"episodes": len(rows), "hits": hits, "hit_rate": hits / len(rows), "ci95": [low, high],
            "launches": len(launched), "launch_metrics": launch_metrics(rows),
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "by_bearing_sector": table(rows, "bearing_sector"),
            "by_target_relative_heading": table(rows, "target_relative_heading"),
            "by_scenario_family": table(rows, "scenario_family")}


def evaluate(spec, suite_name, workers=None, scenario_list=None):
    """Run a suite in parallel. Returns (rows, summary). Pure function of its inputs."""
    items = scenario_list if scenario_list is not None else scenarios.suite(suite_name)
    workers = max(1, min(workers or os.cpu_count() - 1, len(items)))
    chunks = [items[i::workers] for i in range(workers)]
    if workers == 1:
        rows = run_chunk(spec, items)
    else:
        with ProcessPoolExecutor(workers) as pool:
            rows = [row for part in pool.map(run_chunk, [spec] * workers, chunks) for row in part]
    order = {(s.scenario_id, s.seed): i for i, s in enumerate(items)}
    rows.sort(key=lambda r: order[(r["scenario_id"], r["seed"])])
    for row in rows:
        row["scenario_family"] = row["scenario_id"].split("-")[0] + (
            "-" + row["scenario_id"].split("-")[1] if row["scenario_id"].startswith("REC") else "")
    return rows, summarize(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", required=True, choices=CONTROLLERS)
    parser.add_argument("--model", help="training output directory holding model.zip + vecnormalize.pkl "
                                        "(ppo and the two speed-swap controllers)")
    parser.add_argument("--suite", required=True, choices=sorted(scenarios.SUITES))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="run from uncommitted code (result is marked non-reportable)")
    args = parser.parse_args(argv)
    from pursuit import baselines
    needs_model = args.controller == "ppo" or args.controller in baselines.SWAPS
    if needs_model and not args.model:
        parser.error(f"--model is required for --controller {args.controller}")
    prov = provenance.record(args.allow_dirty)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} already exists and is not empty")
    if needs_model and not (Path(args.model) / "policy_numpy.npz").exists():
        baselines.export_policy(args.model)
    spec = {"controller": args.controller, "model": args.model, "out": str(args.out)}
    rule = None
    if args.controller in baselines.SWAPS:
        rule = spec["baseline"] = baselines.best_baseline()
    elif args.controller != "ppo":
        rule = args.controller
    rows, summary = evaluate(spec, args.suite, args.workers)
    write_rows(args.out / "episodes.csv", rows)
    inputs = {}
    if args.model:
        inputs = {name: provenance.sha256(Path(args.model) / name)
                  for name in ("model.zip", "vecnormalize.pkl", "policy_numpy.npz")}
    gains = None
    if rule:
        from dataclasses import asdict
        gains = asdict(baselines.load_gains(rule))
    provenance.write_json(args.out / "summary.json", {
        "controller": args.controller, "model": args.model, "model_sha256": inputs,
        "rule_baseline": rule, "baseline_gains": gains,
        "suite": args.suite, "reportable": not prov["git"]["dirty"], "provenance": prov, **summary})
    print(f"{args.controller} on {args.suite}: {summary['hits']}/{summary['episodes']} hits "
          f"({100 * summary['hit_rate']:.1f}%, 95% CI {100 * summary['ci95'][0]:.1f}-"
          f"{100 * summary['ci95'][1]:.1f}%)  outcomes={summary['outcomes']}")


if __name__ == "__main__":
    main()
