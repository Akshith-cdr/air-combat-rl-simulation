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

from pursuit import provenance, scenarios


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


def summarize(rows):
    hits = sum(r["success"] for r in rows)
    low, high = wilson(hits, len(rows))
    launched = [r for r in rows if r["launched"]]
    return {"episodes": len(rows), "hits": hits, "hit_rate": hits / len(rows), "ci95": [low, high],
            "launches": len(launched), "outcomes": dict(Counter(r["outcome"] for r in rows)),
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
    parser.add_argument("--controller", required=True, choices=["ppo", "pure_pursuit", "lead_pursuit"])
    parser.add_argument("--model", help="training output directory holding model.zip + vecnormalize.pkl")
    parser.add_argument("--suite", required=True, choices=sorted(scenarios.SUITES))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="run from uncommitted code (result is marked non-reportable)")
    args = parser.parse_args(argv)
    if args.controller == "ppo" and not args.model:
        parser.error("--model is required for --controller ppo")
    prov = provenance.record(args.allow_dirty)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} already exists and is not empty")
    if args.controller == "ppo" and not (Path(args.model) / "policy_numpy.npz").exists():
        from pursuit.baselines import export_policy
        export_policy(args.model)
    spec = {"controller": args.controller, "model": args.model, "out": str(args.out)}
    rows, summary = evaluate(spec, args.suite, args.workers)
    write_rows(args.out / "episodes.csv", rows)
    inputs = {}
    if args.model:
        inputs = {name: provenance.sha256(Path(args.model) / name)
                  for name in ("model.zip", "vecnormalize.pkl", "policy_numpy.npz")}
    gains = None
    if args.controller != "ppo":
        from dataclasses import asdict
        from pursuit.baselines import load_gains
        gains = asdict(load_gains(args.controller))
    provenance.write_json(args.out / "summary.json", {
        "controller": args.controller, "model": args.model, "model_sha256": inputs, "baseline_gains": gains,
        "suite": args.suite, "reportable": not prov["git"]["dirty"], "provenance": prov, **summary})
    print(f"{args.controller} on {args.suite}: {summary['hits']}/{summary['episodes']} hits "
          f"({100 * summary['hit_rate']:.1f}%, 95% CI {100 * summary['ci95'][0]:.1f}-"
          f"{100 * summary['ci95'][1]:.1f}%)  outcomes={summary['outcomes']}")


if __name__ == "__main__":
    main()
