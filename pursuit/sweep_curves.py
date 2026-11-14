"""Learning curves on the full all-round sweep (not only the 48-episode dev set).

    python -m pursuit.sweep_curves --runs runs/train --out runs/eval/curves

Every saved checkpoint of every seed is flown on all 1,008 sweep episodes.
The development set saturates early, so this is what shows whether later
training still improves the policy. Writes one CSV per seed:
timesteps, episodes, hits, hit_rate, collisions.
"""

import argparse
import csv
from pathlib import Path

from pursuit import provenance
from pursuit.evaluate import evaluate
from pursuit.scenarios import sweep_suite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs/train"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    prov = provenance.record(args.allow_dirty)
    args.out.mkdir(parents=True, exist_ok=True)
    items = sweep_suite()
    for seed_dir in sorted(args.runs.glob("seed*")):
        path = args.out / f"{seed_dir.name}.csv"
        if path.exists():
            continue
        rows = []
        for checkpoint in sorted((seed_dir / "checkpoints").iterdir()):
            spec = {"controller": "ppo", "model": str(checkpoint), "out": str(args.out), "save_failures": False}
            episodes, summary = evaluate(spec, "sweep", args.workers, scenario_list=items)
            rows.append({"timesteps": int(checkpoint.name), "episodes": summary["episodes"],
                         "hits": summary["hits"], "hit_rate": summary["hit_rate"],
                         "collisions": summary["outcomes"].get("collision", 0)})
            print(f"{seed_dir.name} {int(checkpoint.name):>9d}: {summary['hits']}/{summary['episodes']}", flush=True)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    provenance.write_json(args.out / "provenance.json", prov)


if __name__ == "__main__":
    main()
