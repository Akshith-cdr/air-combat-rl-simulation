"""Explain every failed episode, and relate misses to the launch geometry.

    python -m pursuit.failures --evals runs/eval --out runs/report

Categories (first that applies)
  collision        our aircraft came within the collision radius of the target
                   (before or after launch; the 'launched' column says which)
  ground           our aircraft went below z = 0
  no_launch        the episode ended otherwise without the aircraft reaching
                   the aligned launch band (timeout or boundary)
  missile_overshoot  the missile passed within NEAR_MISS_M of the target but
                   outside the 50 m hit radius, then could not turn back
  missile_outrun   the missile never came within NEAR_MISS_M (target out-turned
                   or outran it)

Writes
  failures.csv         one row per failed episode: controller, category, geometry,
                       missile closest approach and when it happened
  failure_summary.csv  counts per controller and category
  aspect_table.csv     launches, hits and hit rate per launch-aspect bin per controller
  collision_sensitivity.csv  episodes closer than 10/15/20/30 m to the target, per run
"""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pursuit import config as C

NEAR_MISS_M = 150.0
ASPECT_BINS = ((0, 45, "head-on 0-45"), (45, 90, "45-90"), (90, 135, "90-135"), (135, 180.01, "tail 135-180"))


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def missile_track(path):
    """Time of the missile's closest sampled approach, measured from launch."""
    rows = read_csv(path)
    t = np.array([num(r["t"]) for r in rows])
    m = np.array([[num(r["m_x"]), num(r["m_y"]), num(r["m_z"])] for r in rows])
    g = np.array([[num(r["t_x"]), num(r["t_y"]), num(r["t_z"])] for r in rows])
    gap = np.linalg.norm(m - g, axis=1)
    if np.all(np.isnan(gap)):
        return math.nan
    first = int(np.argmax(~np.isnan(gap)))
    i = int(np.nanargmin(gap))
    return float(t[i] - t[first])


def category(row):
    if row["outcome"] in ("collision", "ground"):
        return row["outcome"]                     # before or after launch; see 'launched'
    if row["launched"] != "True":
        return "no_launch"
    closest = num(row.get("missile_closest_approach_m"))
    return "missile_overshoot" if closest < NEAR_MISS_M else "missile_outrun"


def controller_name(directory):
    name = directory.name
    for suite in ("_reviewed_six", "_sweep"):
        name = name.replace(suite, "")
    return name


def aspect_bin(aspect):
    for low, high, label in ASPECT_BINS:
        if low <= aspect < high:
            return label
    return "unknown"


def analyse(evals, out):
    failures, summary = [], defaultdict(Counter)
    aspect = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for directory in sorted(Path(evals).iterdir()):
        if not (directory / "episodes.csv").exists():
            continue
        suite = "sweep" if directory.name.endswith("_sweep") else "reviewed_six"
        group = "ppo" if directory.name.startswith("ppo_") else controller_name(directory)
        for row in read_csv(directory / "episodes.csv"):
            if row["launched"] == "True":
                cell = aspect[(group, suite)][aspect_bin(num(row["launch_aspect_deg"]))]
                cell[0] += 1
                cell[1] += row["success"] == "True"
            if row["success"] == "True":
                continue
            kind = category(row)
            summary[(directory.name, suite)][kind] += 1
            trajectory = directory / "failures" / f"{row['scenario_id']}_{row['seed']}.csv"
            failures.append({
                "run": directory.name, "suite": suite, "scenario_id": row["scenario_id"], "seed": row["seed"],
                "category": kind, "outcome": row["outcome"], "launched": row["launched"],
                "bearing_sector": row["bearing_sector"], "target_relative_heading": row["target_relative_heading"],
                "range_m": row["range_m"], "launch_range_m": row["launch_range_m"],
                "launch_aspect_deg": row["launch_aspect_deg"],
                "launch_closing_speed_mps": row["launch_closing_speed_mps"],
                "missile_closest_approach_m": row["missile_closest_approach_m"],
                "missile_closest_after_launch_s": missile_track(trajectory) if trajectory.exists() else "",
                "aircraft_min_distance_m": row["min_distance_m"],
                "trajectory_file": str(trajectory.relative_to(Path(evals))) if trajectory.exists() else ""})
    write(out / "failures.csv", failures)
    kinds = ["no_launch", "collision", "ground", "missile_overshoot", "missile_outrun"]
    write(out / "failure_summary.csv", [{"run": run, "suite": suite, **{k: counts.get(k, 0) for k in kinds},
                                         "total": sum(counts.values())}
                                        for (run, suite), counts in sorted(summary.items())])
    rows = []
    for (group, suite), bins in sorted(aspect.items()):
        for _, _, label in ASPECT_BINS:
            launches, hits = bins[label]
            rows.append({"controller": group, "suite": suite, "launch_aspect": label, "launches": launches,
                         "hits": hits, "hit_rate": hits / launches if launches else ""})
    write(out / "aspect_table.csv", rows)
    write(out / "collision_sensitivity.csv", collision_sensitivity(evals))
    return failures


SENSITIVITY_RADII_M = (10, 15, 20, 30)


def collision_sensitivity(evals):
    """Episodes whose closest aircraft-target distance falls below each radius.

    Only 10 m ends an episode in the simulator; the larger radii show how the
    collision count would change with a larger (more realistic) aircraft size.
    """
    rows = []
    for directory in sorted(Path(evals).iterdir()):
        if not (directory / "episodes.csv").exists():
            continue
        episodes = read_csv(directory / "episodes.csv")
        distances = [num(e["min_distance_m"]) for e in episodes]
        rows.append({"run": directory.name, "episodes": len(episodes),
                     **{f"below_{r}m": sum(d < r for d in distances) for r in SENSITIVITY_RADII_M}})
    return rows


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evals", type=Path, default=Path("runs/eval"))
    parser.add_argument("--out", type=Path, default=Path("runs/report"))
    args = parser.parse_args(argv)
    failures = analyse(args.evals, args.out)
    print(f"{len(failures)} failed episodes classified -> {args.out}")
    print(Counter(f["category"] for f in failures))


if __name__ == "__main__":
    main()
