"""Reproduce every number in the paper from a clean commit.

    python -m pursuit.run_all                # 5 seeds in parallel, then all evaluations, then report
    python -m pursuit.run_all --skip-train   # only re-run evaluations and report

Steps
  1. train seeds 1..5 (pursuit.train), each in its own process
  2. evaluate the 5 final policies and both baselines on both suites
  3. aggregate (pursuit.report) into runs/report
"""

import argparse
import subprocess
import sys
from pathlib import Path

from pursuit import provenance

SEEDS = (1, 2, 3, 4, 5)
SUITES = ("reviewed_six", "sweep")
BASELINES = ("pure_pursuit", "lead_pursuit")


def run(args):
    print(">", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    provenance.require_clean_tree()
    train_dir, eval_dir = args.root / "train", args.root / "eval"
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
    run(["pursuit.report", "--runs", str(train_dir), "--evals", str(eval_dir), "--out", str(args.root / "report")])


if __name__ == "__main__":
    main()
