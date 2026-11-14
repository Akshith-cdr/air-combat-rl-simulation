"""Build every figure, table and number in the paper from the run outputs.

    python paper/make_figures.py --runs runs --eval eval_v2

Reads   runs/train/seed*/            (run.json, learning_curve.csv)
        runs/<eval>/*/               (episodes.csv, summary.json, failures/*.csv)
        runs/report_<eval>/          (from pursuit.report and pursuit.failures)
Writes  paper/figures/*.pdf
        paper/generated/results.tex  (\newcommand macros used in the text)
        paper/generated/tab_*.tex    (table bodies)
Nothing in the paper is typed in by hand.
"""

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE.parent))        # repo root, for pursuit.scenarios
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
                     "pdf.fonttype": 42})
COLORS = {"ppo": "#1f4e9c", "pure_pursuit": "#c0392b", "lead_pursuit": "#d68910"}
LABELS = {"ppo": "PPO (5 seeds)", "pure_pursuit": "Pure pursuit", "lead_pursuit": "Lead pursuit"}
SEEDS = (1, 2, 3, 4, 5)
COL_W, FULL_W = 3.5, 7.16


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


class Data:
    def __init__(self, runs, eval_name):
        self.runs, self.eval = Path(runs), Path(runs) / eval_name
        self.report = Path(runs) / f"report_{eval_name}"
        self.episodes, self.summary = {}, {}
        for directory in sorted(self.eval.iterdir()):
            if (directory / "summary.json").exists():
                self.episodes[directory.name] = read_csv(directory / "episodes.csv")
                self.summary[directory.name] = json.loads((directory / "summary.json").read_text())

    def runs_of(self, controller, suite):
        if controller == "ppo":
            return [f"ppo_seed{s}_{suite}" for s in SEEDS]
        return [f"{controller}_{suite}"]

    def rates(self, controller, suite):
        return [self.summary[r]["hit_rate"] for r in self.runs_of(controller, suite)]

    def pooled(self, controller, suite):
        rows = [e for r in self.runs_of(controller, suite) for e in self.episodes[r]]
        return rows


# ---------------------------------------------------------------- macros
def fmt(x, digits=1):
    return f"{x:.{digits}f}"


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}"


def build_macros(d):
    m = {}
    for suite, tag in (("reviewed_six", "Six"), ("sweep", "Sweep")):
        ppo = [100 * r for r in d.rates("ppo", suite)]
        m[f"Ppo{tag}Mean"] = fmt(st.mean(ppo), 2)
        m[f"Ppo{tag}Sd"] = fmt(st.stdev(ppo), 2)
        m[f"Ppo{tag}Min"] = fmt(min(ppo))
        m[f"Ppo{tag}Max"] = fmt(max(ppo))
        m[f"Ppo{tag}Episodes"] = str(d.summary[d.runs_of("ppo", suite)[0]]["episodes"])
        for name, key in (("pure_pursuit", "Pure"), ("lead_pursuit", "Lead")):
            s = d.summary[f"{name}_{suite}"]
            m[f"{key}{tag}Rate"] = pct(s["hit_rate"])
            m[f"{key}{tag}Hits"] = str(s["hits"])
            m[f"{key}{tag}Ci"] = f"{pct(s['ci95'][0])}--{pct(s['ci95'][1])}"
            m[f"{key}{tag}Collisions"] = str(s["outcomes"].get("collision", 0))
        coll = [d.summary[r]["outcomes"].get("collision", 0) for r in d.runs_of("ppo", suite)]
        m[f"Ppo{tag}CollisionsTotal"] = str(sum(coll))
        m[f"Ppo{tag}CollisionsMax"] = str(max(coll))
        for controller, key in (("ppo", "Ppo"), ("pure_pursuit", "Pure"), ("lead_pursuit", "Lead")):
            launched = [e for e in d.pooled(controller, suite) if e["launched"] == "True"]
            m[f"{key}{tag}LaunchTime"] = fmt(st.median(float(e["launch_time_s"]) for e in launched))
            m[f"{key}{tag}LaunchRange"] = fmt(st.median(float(e["launch_range_m"]) for e in launched), 0)
            m[f"{key}{tag}LaunchRate"] = pct(len(launched) / len(d.pooled(controller, suite)))
    # Learning curves.
    curves = read_csv(d.report / "learning_curves.csv")
    first90 = next(int(r["timesteps"]) for r in curves if float(r["mean"]) >= 0.9)
    all100 = next(int(r["timesteps"]) for r in curves if float(r["min"]) >= 1.0)
    m["CurveFirstNinety"] = f"{first90 / 1e3:.0f}k"
    m["CurveAllHundred"] = f"{all100 / 1e6:.2f}M"
    m["CurveStepZeroMax"] = pct(float(curves[0]["max"]), 0)
    runs = [json.loads((d.runs / "train" / f"seed{s}" / "run.json").read_text()) for s in SEEDS]
    m["TrainSteps"] = f"{runs[0]['num_timesteps']:,}".replace(",", "{,}")
    m["TrainHours"] = fmt(st.mean(r["wall_seconds"] for r in runs) / 3600)
    m["TrainCommit"] = runs[0]["provenance"]["git"]["commit"][:7]
    m["EvalCommit"] = next(iter(d.summary.values()))["provenance"]["git"]["commit"][:7]
    # Failures.
    failures = read_csv(d.report / "failures.csv")
    over = [f for f in failures if f["category"] == "missile_overshoot"]
    m["FailTotal"] = str(len(failures))
    m["FailOvershoot"] = str(len(over))
    m["FailCollision"] = str(sum(f["category"] == "collision" for f in failures))
    m["FailOther"] = str(sum(f["category"] not in ("missile_overshoot", "collision") for f in failures))
    closest = [float(f["missile_closest_approach_m"]) for f in over]
    m["OvershootClosestMedian"] = fmt(st.median(closest), 0)
    m["OvershootClosestMax"] = fmt(max(closest), 0)
    when = [float(f["missile_closest_after_launch_s"]) for f in over if f["missile_closest_after_launch_s"]]
    m["OvershootTimeMedian"] = fmt(st.median(when))
    m["OvershootTimeMax"] = fmt(max(when))
    return m


def write_macros(m, path):
    lines = ["% Generated by paper/make_figures.py -- do not edit."]
    for key, value in sorted(m.items()):
        lines.append(f"\\newcommand{{\\R{key}}}{{{value}}}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- tables
def table_comparison(d, path):
    rows = []
    for controller in ("ppo", "lead_pursuit", "pure_pursuit"):
        cells = [LABELS[controller]]
        for suite in ("reviewed_six", "sweep"):
            rates = [100 * r for r in d.rates(controller, suite)]
            if controller == "ppo":
                cells.append(f"{st.mean(rates):.2f} $\\pm$ {st.stdev(rates):.2f} ({min(rates):.1f}--{max(rates):.1f})")
            else:
                s = d.summary[f"{controller}_{suite}"]
                cells.append(f"{100 * s['hit_rate']:.1f} ({100 * s['ci95'][0]:.1f}--{100 * s['ci95'][1]:.1f})")
        sweep = d.pooled(controller, "sweep")
        launched = [e for e in sweep if e["launched"] == "True"]
        n = len(d.runs_of(controller, "sweep"))
        coll = sum(e["outcome"] == "collision" for e in sweep)
        cells.append(f"{coll / n:.1f}" if n > 1 else str(coll))
        cells.append(f"{st.median(float(e['launch_time_s']) for e in launched):.1f}")
        cells.append(f"{st.median(float(e['launch_range_m']) for e in launched):.0f}")
        rows.append(" & ".join(cells) + r" \\")
    path.write_text("\n".join(rows) + "\n")


def table_scenarios(d, path):
    """Table III: the reviewed six geometries with target heading, per controller."""
    from pursuit.scenarios import REVIEWED_SIX
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for controller in ("ppo", "lead_pursuit", "pure_pursuit"):
        for e in d.pooled(controller, "reviewed_six"):
            cell = by[e["scenario_id"]][controller]
            cell[0] += e["success"] == "True"
            cell[1] += 1
    rows = []
    for name, (x, y, z), speed, heading in REVIEWED_SIX:
        bearing = math.degrees(math.atan2(x, y))
        d0 = math.sqrt(x * x + y * y + (z - 1000.0) ** 2)          # 3D initial range
        cells = [name, f"$({x / 1e3:.1f}, {y / 1e3:.1f}, {z / 1e3:.1f})$", f"{d0 / 1e3:.2f}",
                 f"${round(bearing):d}$", f"{heading:.0f}", f"{speed:.0f}"]
        for controller in ("ppo", "lead_pursuit", "pure_pursuit"):
            k, n = by[name][controller]
            cells.append(f"{100 * k / n:.1f}")
        rows.append(" & ".join(cells) + r" \\")
    path.write_text("\n".join(rows) + "\n")


def table_failures(d, path):
    order = ("missile_overshoot", "collision", "no_launch", "ground")
    summary = read_csv(d.report / "failure_summary.csv")
    rows = []
    for controller in ("ppo", "lead_pursuit", "pure_pursuit"):
        cells = [LABELS[controller]]
        for suite in ("reviewed_six", "sweep"):
            names = set(d.runs_of(controller, suite))
            part = [r for r in summary if r["run"] in names]
            n = len(names)
            for kind in ("missile_overshoot", "collision"):
                total = sum(int(r[kind]) for r in part)
                cells.append(f"{total / n:.1f}" if n > 1 else str(total))
            other = sum(int(r[k]) for r in part for k in ("no_launch", "ground", "missile_outrun"))
            cells.append(f"{other / n:.1f}" if n > 1 else str(other))
        rows.append(" & ".join(cells) + r" \\")
    path.write_text("\n".join(rows) + "\n")


# ---------------------------------------------------------------- figures
def fig_learning(d, path):
    curves = read_csv(d.report / "learning_curves.csv")
    x = np.array([int(r["timesteps"]) for r in curves]) / 1e6
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for s in SEEDS:
        ax.plot(x, [100 * float(r[f"seed{s}"]) for r in curves], lw=0.7, alpha=0.6, label=f"seed {s}")
    ax.fill_between(x, [100 * float(r["min"]) for r in curves], [100 * float(r["max"]) for r in curves],
                    color="k", alpha=0.10, lw=0)
    ax.plot(x, [100 * float(r["mean"]) for r in curves], color="k", lw=1.6, label="mean")
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel("dev. hit rate (%)")
    ax.set_ylim(-2, 102)
    ax.set_xlim(0, x.max())
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(ncol=3, loc="lower right", frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_comparison(d, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.2), gridspec_kw={"width_ratios": [1, 1.6]})
    suites = (("reviewed_six", "six reviewed geometries"), ("sweep", "all-round sweep"))
    for i, (suite, label) in enumerate(suites):
        for j, controller in enumerate(("ppo", "lead_pursuit", "pure_pursuit")):
            xpos = i * 4 + j
            color = COLORS[controller]
            if controller == "ppo":
                rates = [100 * r for r in d.rates(controller, suite)]
                a.scatter([xpos] * len(rates), rates, s=10, color=color, alpha=0.7, zorder=3)
                a.plot([xpos - 0.3, xpos + 0.3], [st.mean(rates)] * 2, color=color, lw=2)
            else:
                s = d.summary[f"{controller}_{suite}"]
                lo, hi = 100 * s["ci95"][0], 100 * s["ci95"][1]
                a.errorbar(xpos, 100 * s["hit_rate"], yerr=[[100 * s["hit_rate"] - lo], [hi - 100 * s["hit_rate"]]],
                           fmt="o", color=color, ms=4, capsize=2, lw=1)
    a.set_xticks([1, 5])
    a.set_xticklabels([s[1] for s in suites])
    a.set_ylabel("hit rate (%)")
    a.set_ylim(95, 100.2)
    a.set_title("(a) overall hit rate (axis starts at 95%)")
    a.grid(axis="y", alpha=0.3, lw=0.4)
    handles = [plt.Line2D([], [], color=COLORS[c], marker="o", ls="", ms=4, label=LABELS[c])
               for c in ("ppo", "lead_pursuit", "pure_pursuit")]
    a.legend(handles=handles, loc="lower left", frameon=False)
    # (b) per bearing sector, sweep.
    sectors = list(range(0, 360, 30))
    for controller in ("ppo", "lead_pursuit", "pure_pursuit"):
        per_run = []
        for r in d.runs_of(controller, "sweep"):
            table = d.summary[r]["by_bearing_sector"]
            per_run.append([100 * table[str(s)]["hit_rate"] for s in sectors])
        per_run = np.array(per_run)
        b.plot(sectors, per_run.mean(0), marker="o", ms=3, lw=1.2, color=COLORS[controller],
               label=LABELS[controller])
        if len(per_run) > 1:
            b.fill_between(sectors, per_run.min(0), per_run.max(0), color=COLORS[controller], alpha=0.15, lw=0)
    b.set_xticks(sectors)
    b.set_xlabel("initial target bearing (deg clockwise from our nose; 180 = behind)")
    b.set_ylabel("hit rate (%)")
    b.set_ylim(85, 100.5)
    b.set_title("(b) sweep, by initial bearing (band: PPO seed range)")
    b.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_aspect(d, path):
    table = read_csv(d.report / "aspect_table.csv")
    bins = ["head-on 0-45", "45-90", "90-135", "tail 135-180"]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    width = 0.26
    for j, controller in enumerate(("ppo", "lead_pursuit", "pure_pursuit")):
        launches = hits = None
        rates, errs = [], [[], []]
        for label in bins:
            cells = [r for r in table if r["controller"] == controller and r["launch_aspect"] == label]
            n = sum(int(r["launches"]) for r in cells)
            k = sum(int(r["hits"]) for r in cells)
            p = 100 * k / n
            lo, hi = wilson(k, n)
            rates.append(p)
            errs[0].append(p - 100 * lo)
            errs[1].append(100 * hi - p)
        xs = np.arange(len(bins)) + (j - 1) * width
        ax.bar(xs, rates, width, color=COLORS[controller], label=LABELS[controller], yerr=errs,
               error_kw={"lw": 0.6, "capsize": 1.5})
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(["0-45\n(head-on)", "45-90", "90-135", "135-180\n(tail)"])
    ax.set_xlabel("target aspect at launch (deg)")
    ax.set_ylabel("hit rate (%)")
    ax.set_ylim(90, 100.8)
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    ax.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_overshoot(d, path):
    """Top-down view of one typical overshoot miss (median closest approach, PPO)."""
    failures = [f for f in read_csv(d.report / "failures.csv")
                if f["category"] == "missile_overshoot" and f["run"].startswith("ppo") and f["suite"] == "sweep"]
    failures.sort(key=lambda f: float(f["missile_closest_approach_m"]))
    pick = failures[len(failures) // 2]
    rows = read_csv(d.eval / pick["trajectory_file"])
    num = lambda r, k: float(r[k])
    t = np.array([num(r, "t") for r in rows])
    P = np.array([[num(r, "p_x"), num(r, "p_y")] for r in rows])
    T = np.array([[num(r, "t_x"), num(r, "t_y")] for r in rows])
    M = np.array([[num(r, "m_x"), num(r, "m_y")] for r in rows])
    launched = np.array([r["launched"] == "True" for r in rows])
    li = int(np.argmax(launched))
    gap = np.linalg.norm(np.array([[num(r, "m_x"), num(r, "m_y"), num(r, "m_z")] for r in rows])
                         - np.array([[num(r, "t_x"), num(r, "t_y"), num(r, "t_z")] for r in rows]), axis=1)
    ci = int(np.nanargmin(gap))
    end = min(len(t) - 1, ci + int(6 / 0.05))
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 2.0), gridspec_kw={"width_ratios": [1.3, 1]})
    a.plot(P[:end, 0] / 1e3, P[:end, 1] / 1e3, color=COLORS["ppo"], lw=1, label="our aircraft")
    a.plot(T[:end, 0] / 1e3, T[:end, 1] / 1e3, color="0.3", lw=1, label="target")
    a.plot(M[li:end, 0] / 1e3, M[li:end, 1] / 1e3, color="#c0392b", lw=1, ls="--", label="missile")
    a.scatter(*P[li] / 1e3, s=12, color=COLORS["ppo"], zorder=3)
    a.scatter(*M[ci] / 1e3, s=14, marker="x", color="#c0392b", zorder=3)
    a.set_aspect("equal", "datalim")
    a.set_xlabel("x (km)")
    a.set_ylabel("y (km)")
    a.legend(frameon=False, loc="best", fontsize=6)
    a.set_title("(a) top view")
    a.grid(alpha=0.3, lw=0.4)
    tt = t[li:end] - t[li]
    b.plot(tt, gap[li:end], color="#c0392b", lw=1)
    b.axhline(50, color="k", ls=":", lw=0.8)
    b.text(tt[-1], 55, "50 m hit radius", ha="right", va="bottom", fontsize=6)
    b.set_yscale("log")
    b.set_xlabel("time after launch (s)")
    b.set_ylabel("miss distance (m)")
    b.set_title("(b) miss distance")
    b.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pick


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--eval", default="eval_v2")
    args = parser.parse_args(argv)
    d = Data(args.runs, args.eval)
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "generated").mkdir(exist_ok=True)
    macros = build_macros(d)
    fig_learning(d, HERE / "figures/fig_learning.pdf")
    fig_comparison(d, HERE / "figures/fig_comparison.pdf")
    fig_aspect(d, HERE / "figures/fig_aspect.pdf")
    pick = fig_overshoot(d, HERE / "figures/fig_overshoot.pdf")
    macros.update(OvershootExampleId=pick["scenario_id"].replace("_", r"\_"),
                  OvershootExampleRun=pick["run"].replace("_", r"\_"),
                  OvershootExampleClosest=fmt(float(pick["missile_closest_approach_m"]), 0),
                  OvershootExampleAspect=fmt(float(pick["launch_aspect_deg"]), 0))
    write_macros(macros, HERE / "generated/results.tex")
    table_comparison(d, HERE / "generated/tab_comparison.tex")
    table_scenarios(d, HERE / "generated/tab_scenarios.tex")
    table_failures(d, HERE / "generated/tab_failures.tex")
    print("figures and tables written to", HERE)


if __name__ == "__main__":
    main()
