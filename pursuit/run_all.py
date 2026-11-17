"""Reproduce every number in the paper from a clean commit.

The tuned baseline gains (pursuit/baseline_gains.json) are part of the
repository. They were made, from a clean tree, with

    python -m pursuit.tune_baselines --out runs/final/tuning

so on a fresh clone one command reproduces everything else:

    python -m pursuit.run_all                  # writes runs/reproduction/

  1. train seeds 1..5 (pursuit.train), each in its own process   -> train/seed<N>/
  2. evaluate the 5 final policies and the 3 tuned baselines on the
     3 test suites                                                -> eval/<controller>_<suite>/
  3. speed-swap controls (M4) with each seed on the two sweeps    -> eval/<swap>_seed<N>_<suite>/
  4. aggregate (pursuit.report) and explain failures (pursuit.failures) -> report/
  (--sweep-curves also flies every training checkpoint on the full sweep -> eval/curves/)
Then build the paper's figures and numbers with
    python paper/make_figures.py --root runs/reproduction
The paper's own results are in runs/final (made with --root runs/final).
"""

import argparse
import subprocess
import sys
from pathlib import Path

from pursuit import provenance
from pursuit.baselines import BASELINES, GAINS_FILE, SWAPS

SEEDS = (1, 2, 3, 4, 5)
SUITES = ("reviewed_six", "sweep", "sweep_alt")
SWAP_SUITES = ("sweep", "sweep_alt")


def run(args):
    print(">", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("runs/reproduction"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--sweep-curves", action="store_true",
                        help="also fly every checkpoint on the full sweep (about 80,000 episodes)")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)

    def evaluate(controller, suite, out, model=None):
        if (out / "summary.json").exists():
            print(f"skip {out} (done)", flush=True)
            return
        run(["pursuit.evaluate", "--controller", controller, "--suite", suite, "--out", str(out),
             "--workers", str(args.workers), *(["--model", str(model)] if model else [])])

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
            evaluate(name, suite, eval_dir / f"{name}_{suite}")
        for seed in SEEDS:
            evaluate("ppo", suite, eval_dir / f"ppo_seed{seed}_{suite}", train_dir / f"seed{seed}")
    for name in BASELINES:                       # reference lines for the learning curves
        evaluate(name, "dev", eval_dir / f"{name}_dev")
    for suite in SWAP_SUITES:
        for swap in SWAPS:
            for seed in SEEDS:
                evaluate(swap, suite, eval_dir / f"{swap}_seed{seed}_{suite}", train_dir / f"seed{seed}")
    if args.sweep_curves:
        run(["pursuit.sweep_curves", "--runs", str(train_dir), "--out", str(eval_dir / "curves"),
             "--workers", str(args.workers)])
    run(["pursuit.report", "--runs", str(train_dir), "--evals", str(eval_dir), "--out", str(report)])
    run(["pursuit.failures", "--evals", str(eval_dir), "--out", str(report)])


if __name__ == "__main__":
    main()
