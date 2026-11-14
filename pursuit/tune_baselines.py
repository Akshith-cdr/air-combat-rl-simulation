"""Choose each baseline's gains by grid search on the tuning suite.

    python -m pursuit.tune_baselines          # writes pursuit/baseline_gains.json

The tuning suite (288 episodes) has the sweep's geometries but its own seeds,
so the test suites are never used to pick gains. Each candidate is ranked by
hit rate, then by fewer collisions, then by shorter median time to launch.
Every candidate's result is kept in runs/tuning/<controller>.csv.
"""

import argparse
import csv
import itertools
import json
import statistics as st
from dataclasses import asdict, replace
from pathlib import Path

from pursuit import provenance
from pursuit.baselines import GAINS_FILE, Gains
from pursuit.evaluate import evaluate
from pursuit.scenarios import tuning_suite

GRID = {
    "bank_per_deg": (1.0, 1.5, 2.5),
    "yaw_deg_per_unit": (10.0, 20.0, 40.0),
    "hold_range_m": (600.0, 750.0, 900.0),
    "speed_per_m": (0.05, 0.1, 0.2),
}


def score(rows):
    hits = sum(r["success"] for r in rows)
    collisions = sum(r["outcome"] == "collision" for r in rows)
    launched = [r["launch_time_s"] for r in rows if r["launched"]]
    median_launch = st.median(launched) if launched else float("inf")
    return hits, collisions, median_launch


def tune(controller, out, workers):
    items = tuning_suite()
    results = []
    for values in itertools.product(*GRID.values()):
        gains = replace(Gains(), **dict(zip(GRID, values)))
        spec = {"controller": controller, "gains": asdict(gains), "out": str(out), "save_failures": False}
        rows, _ = evaluate(spec, "tune", workers, scenario_list=items)
        hits, collisions, launch = score(rows)
        results.append({**dict(zip(GRID, values)), "episodes": len(rows), "hits": hits,
                        "collisions": collisions, "median_launch_s": launch})
        print(f"{controller} {dict(zip(GRID, values))}: {hits}/{len(rows)} hits, "
              f"{collisions} collisions, launch {launch:.1f} s", flush=True)
    results.sort(key=lambda r: (-r["hits"], r["collisions"], r["median_launch_s"]))
    with open(out / f"{controller}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    best = results[0]
    return replace(Gains(), **{k: best[k] for k in GRID}), best, results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("runs/tuning"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    prov = provenance.record(args.allow_dirty)
    args.out.mkdir(parents=True, exist_ok=True)
    table = {}
    for controller in ("pure_pursuit", "lead_pursuit"):
        gains, best, results = tune(controller, args.out, args.workers)
        default = next(r for r in results if all(r[k] == getattr(Gains(), k) for k in GRID))
        table[controller] = {"gains": asdict(gains), "tuning_hits": best["hits"],
                             "tuning_collisions": best["collisions"],
                             "default_gains_hits": default["hits"], "episodes": best["episodes"],
                             "candidates": len(results)}
    table["_provenance"] = {"git_commit": prov["git"]["commit"], "suite": "tune",
                            "grid": {k: list(v) for k, v in GRID.items()}}
    GAINS_FILE.write_text(json.dumps(table, indent=2, sort_keys=True))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
