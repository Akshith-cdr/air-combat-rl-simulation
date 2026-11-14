"""Reproduce every number in the paper from a clean commit.

Two steps, each from a clean (committed) tree:

    python -m pursuit.tune_baselines --out runs/final/tuning   # writes pursuit/baseline_gains.json
    (commit pursuit/baseline_gains.json)
    python -m pursuit.run_all                                  # everything else

run_all, with --root runs/final by default:
  1. train seeds 1..5 (pursuit.train), each in its own process   -> train/seed<N>/
  2. evaluate the 5 final policies and both tuned baselines on both
     test suites                                                  -> eval/<controller>_<suite>/
  3. fly every training checkpoint on the full sweep              -> eval/curves/
  4. aggregate (pursuit.report) and explain failures (pursuit.failures) -> report/
Then build the paper's figures and numbers with  python paper/make_figures.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

from pursuit import provenance
from pursuit.baselines import GAINS_FILE

SEEDS = (1, 2, 3, 4, 5)
SUITES = ("reviewed_six", "sweep")
BASELINES = ("pure_pursuit", "lead_pursuit")


def run(args):
    print(">", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("runs/final"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    provenance.require_clean_tree()
    if not GAINS_FILE.exists():
        raise SystemExit(f"{GAINS_FILE} is missing: run pursuit.tune_baselines and commit its output first")
    train_dir, eval_dir, report = args.root / "train", args.root / "eval", args.root / "report"
    if not args.skip_train:
        args.root.mkdir(parents=True, exist_ok=True)
        jobs = []
        for seed in SEEDS:
            log = open(args.root / f"train_seed{seed}.log", "w")
            jobs.append(subprocess.Popen([sys.executable, "-m", "pursuit.train", "--seed", str(seed),
                                          "--out", str(train_dir / f"seed{seed}")],
                                         stdout=log, stderr=subprocess.STDOUT))
        failed = [seed for seed, job in zip(SEEDS, jobs) if job.wait() != 0]
        if failed:
            raise SystemExit(f"training failed for seeds {failed}; see {args.root}/train_seed*.log")
    for suite in SUITES:
        for name in BASELINES:
            run(["pursuit.evaluate", "--controller", name, "--suite", suite,
                 "--out", str(eval_dir / f"{name}_{suite}"), "--workers", str(args.workers)])
        for seed in SEEDS:
            run(["pursuit.evaluate", "--controller", "ppo", "--model", str(train_dir / f"seed{seed}"),
                 "--suite", suite, "--out", str(eval_dir / f"ppo_seed{seed}_{suite}"),
                 "--workers", str(args.workers)])
    run(["pursuit.sweep_curves", "--runs", str(train_dir), "--out", str(eval_dir / "curves"),
         "--workers", str(args.workers)])
    run(["pursuit.report", "--runs", str(train_dir), "--evals", str(eval_dir), "--out", str(report)])
    run(["pursuit.failures", "--evals", str(eval_dir), "--out", str(report)])


if __name__ == "__main__":
    main()
