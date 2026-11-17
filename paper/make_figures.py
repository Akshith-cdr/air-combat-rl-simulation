"""Build every figure, table and number in the paper from the run outputs.

    python paper/make_figures.py                 # --root runs/final

Reads   <root>/train/seed*/          (run.json, learning_curve.csv)
        <root>/eval/*/               (episodes.csv, summary.json, failures/*.csv)
        <root>/report/               (from pursuit.report and pursuit.failures)
        pursuit/baseline_gains.json  (from pursuit.tune_baselines)
Writes  paper/figures/*.pdf
        paper/generated/results.tex  (\\newcommand macros used in the text)
        paper/generated/tab_*.tex    (table bodies)
Nothing in the paper is typed in by hand. Every percentage is printed with
one decimal place, every time in seconds with one decimal, ranges and angles
as whole numbers.
"""

import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))        # repo root, for pursuit.*
from pursuit.evaluate import aspect_class  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
                     "pdf.fonttype": 42})
CONTROLLERS = ("ppo", "pn_pursuit", "lead_pursuit", "pure_pursuit")
BASELINES = CONTROLLERS[1:]
COLORS = {"ppo": "#1f4e9c", "pure_pursuit": "#c0392b", "lead_pursuit": "#d68910", "pn_pursuit": "#1e8449",
          "ppo_rule_speed": "#7d3c98", "rule_steer_ppo_speed": "#5d6d7e"}
LABELS = {"ppo": "PPO (5 seeds)", "pure_pursuit": "Pure pursuit", "lead_pursuit": "Lead-then-point",
          "pn_pursuit": "PN-then-point"}
KEYS = {"ppo": "Ppo", "pure_pursuit": "Pure", "lead_pursuit": "Lead", "pn_pursuit": "Pn",
        "ppo_rule_speed": "SwapPpoSteer", "rule_steer_ppo_speed": "SwapRuleSteer"}
SUITES = (("reviewed_six", "Six"), ("sweep", "Sweep"), ("sweep_alt", "Alt"))
SWEEPS = ("sweep", "sweep_alt")
SEEDS = (1, 2, 3, 4, 5)
COL_W, FULL_W = 3.5, 7.16


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def wilson(k, n, z=1.959964):
    if n == 0:
        return math.nan, math.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def fmt(x, digits=1):
    return f"{x:.{digits}f}"


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}"


class Data:
    """Reads <root>/train, <root>/eval and <root>/report as written by pursuit.run_all."""

    def __init__(self, root):
        self.runs, self.eval, self.report = Path(root), Path(root) / "eval", Path(root) / "report"
        self.episodes, self.summary = {}, {}
        for directory in sorted(self.eval.iterdir()):
            if (directory / "summary.json").exists():
                self.episodes[directory.name] = read_csv(directory / "episodes.csv")
                self.summary[directory.name] = json.loads((directory / "summary.json").read_text())
        self.rule = next((s["rule_baseline"] for name, s in self.summary.items()
                          if s["controller"] == "rule_steer_ppo_speed"), None)

    def runs_of(self, controller, suite):
        if controller in ("ppo", "ppo_rule_speed", "rule_steer_ppo_speed"):
            return [f"{controller}_seed{s}_{suite}" for s in SEEDS]
        return [f"{controller}_{suite}"]

    def rates(self, controller, suite):
        return [self.summary[r]["hit_rate"] for r in self.runs_of(controller, suite)]

    def pooled(self, controller, suites):
        suites = (suites,) if isinstance(suites, str) else suites
        return [e for s in suites for r in self.runs_of(controller, s) for e in self.episodes[r]]

    def n_runs(self, controller):
        return len(self.runs_of(controller, "sweep"))


def launched(rows):
    return [e for e in rows if e["launched"] == "True"]


def launch_stats(rows):
    """Median time to launch, median aspect, favourable (head-on or tail) share of launches."""
    ls = launched(rows)
    aspects = [float(e["launch_aspect_deg"]) for e in ls]
    fav = sum(aspect_class(a) != "beam" for a in aspects)
    return {"launches": len(ls), "launch_share": len(ls) / len(rows),
            "time": st.median(float(e["launch_time_s"]) for e in ls),
            "time_p25": float(np.percentile([float(e["launch_time_s"]) for e in ls], 25)),
            "time_p75": float(np.percentile([float(e["launch_time_s"]) for e in ls], 75)),
            "range": st.median(float(e["launch_range_m"]) for e in ls),
            "aspect": st.median(aspects), "fav": fav / len(ls), "fav_ci": wilson(fav, len(ls)),
            "beam": sum(aspect_class(a) == "beam" for a in aspects),
            "head_on": sum(aspect_class(a) == "head_on" for a in aspects),
            "tail": sum(aspect_class(a) == "tail" for a in aspects)}


def hit_cell(d, controller, suite):
    """'mean +- sd (min--max)' over seeds, or 'rate (CI)' for a single run; one decimal."""
    rates = [100 * r for r in d.rates(controller, suite)]
    if len(rates) > 1:
        return f"{st.mean(rates):.1f} $\\pm$ {st.stdev(rates):.1f} ({min(rates):.1f}--{max(rates):.1f})"
    s = d.summary[d.runs_of(controller, suite)[0]]
    return f"{100 * s['hit_rate']:.1f} ({100 * s['ci95'][0]:.1f}--{100 * s['ci95'][1]:.1f})"


# ---------------------------------------------------------------- macros
def build_macros(d):
    m = {}
    for controller in CONTROLLERS:
        key = KEYS[controller]
        for suite, tag in SUITES:
            rates = [100 * r for r in d.rates(controller, suite)]
            runs = d.runs_of(controller, suite)
            m[f"{key}{tag}Rate"] = fmt(st.mean(rates))
            if len(rates) > 1:
                m[f"{key}{tag}Sd"] = fmt(st.stdev(rates))
                m[f"{key}{tag}Min"] = fmt(min(rates))
                m[f"{key}{tag}Max"] = fmt(max(rates))
            else:
                s = d.summary[runs[0]]
                m[f"{key}{tag}Ci"] = f"{pct(s['ci95'][0])}--{pct(s['ci95'][1])}"
            m[f"{key}{tag}Episodes"] = str(d.summary[runs[0]]["episodes"])
            coll = [d.summary[r]["outcomes"].get("collision", 0) for r in runs]
            m[f"{key}{tag}Collisions"] = fmt(st.mean(coll)) if len(coll) > 1 else str(coll[0])
            ls = launch_stats(d.pooled(controller, suite))
            m[f"{key}{tag}LaunchTime"] = fmt(ls["time"])
            m[f"{key}{tag}LaunchRange"] = fmt(ls["range"], 0)
            m[f"{key}{tag}LaunchAspect"] = fmt(ls["aspect"], 0)
            m[f"{key}{tag}Fav"] = pct(ls["fav"])
            m[f"{key}{tag}LaunchShare"] = pct(ls["launch_share"])
        ls = launch_stats(d.pooled(controller, SWEEPS))
        m[f"{key}SweepsLaunchTime"] = fmt(ls["time"])
        m[f"{key}SweepsLaunchTimeIqr"] = f"{ls['time_p25']:.1f}--{ls['time_p75']:.1f}"
        m[f"{key}SweepsLaunchAspect"] = fmt(ls["aspect"], 0)
        m[f"{key}SweepsFav"] = pct(ls["fav"])
        m[f"{key}SweepsFavCi"] = f"{pct(ls['fav_ci'][0])}--{pct(ls['fav_ci'][1])}"
        m[f"{key}SweepsBeamShare"] = pct(ls["beam"] / ls["launches"])
    # Margins (PPO mean minus baseline).
    for suite, tag in SUITES:
        ppo = [100 * r for r in d.rates("ppo", suite)]
        for b in BASELINES:
            base = 100 * d.summary[f"{b}_{suite}"]["hit_rate"]
            m[f"Margin{tag}{KEYS[b]}Mean"] = f"{st.mean(ppo) - base:+.1f}"
            m[f"Margin{tag}{KEYS[b]}Min"] = f"{min(ppo) - base:+.1f}"
            m[f"Margin{tag}{KEYS[b]}Max"] = f"{max(ppo) - base:+.1f}"
        best = max(BASELINES, key=lambda b: d.summary[f"{b}_{suite}"]["hit_rate"])
        m[f"Best{tag}Baseline"] = LABELS[best].lower() if best != "pn_pursuit" else "PN-then-point"
        m[f"Best{tag}BaselineRate"] = pct(d.summary[f"{best}_{suite}"]["hit_rate"])
    # Paired tests (M5): seeds significantly better / worse after Holm correction.
    paired = read_csv(d.report / "paired_tests.csv")
    for suite, tag in SUITES:
        for b in BASELINES:
            rows = [r for r in paired if r["suite"] == suite and r["b"] == b and r["family"] == "ppo_vs_baseline"]
            sig = lambda r, col: float(r[col]) < 0.05
            for col, suffix in (("mcnemar_p_holm", ""), ("mcnemar_p", "Raw")):
                better = [r for r in rows if sig(r, col) and int(r["a_only_hits"]) > int(r["b_only_hits"])]
                worse = [r for r in rows if sig(r, col) and int(r["a_only_hits"]) < int(r["b_only_hits"])]
                m[f"Sig{tag}{KEYS[b]}Better{suffix}"] = str(len(better))
                m[f"Sig{tag}{KEYS[b]}Worse{suffix}"] = str(len(worse))
            earlier = [r for r in rows if float(r["launch_time_sign_p"]) < 0.05
                       and int(r["a_launch_earlier"]) > int(r["a_launch_later"])]
            m[f"Earlier{tag}{KEYS[b]}"] = str(len(earlier))
            diffs = [float(r["median_launch_time_diff_s"]) for r in rows if r["median_launch_time_diff_s"]]
            if diffs:
                m[f"LaunchDiff{tag}{KEYS[b]}"] = f"{st.mean(diffs):+.1f}"
    m["PairedTests"] = str(sum(r["family"] == "ppo_vs_baseline" for r in paired))
    # Speed swap (M4).
    if d.rule:
        m["SwapRule"] = LABELS[d.rule] if d.rule != "pn_pursuit" else "PN-then-point"
        for kind in ("ppo_rule_speed", "rule_steer_ppo_speed"):
            key = KEYS[kind]
            for suite, tag in SUITES[1:]:
                rates = [100 * r for r in d.rates(kind, suite)]
                m[f"{key}{tag}Rate"] = fmt(st.mean(rates))
                m[f"{key}{tag}Min"] = fmt(min(rates))
                m[f"{key}{tag}Max"] = fmt(max(rates))
            ls = launch_stats(d.pooled(kind, SWEEPS))
            m[f"{key}SweepsLaunchTime"] = fmt(ls["time"])
            m[f"{key}SweepsFav"] = pct(ls["fav"])
            m[f"{key}SweepsLaunchAspect"] = fmt(ls["aspect"], 0)
            for suite, tag in SUITES[1:]:
                rows = [r for r in paired if r["family"] == "speed_swap" and r["suite"] == suite
                        and r["a"].startswith(kind)]
                m[f"{key}{tag}SigBetter"] = str(sum(float(r["mcnemar_p"]) < .05 and int(r["a_only_hits"]) > int(r["b_only_hits"]) for r in rows))
                m[f"{key}{tag}SigWorse"] = str(sum(float(r["mcnemar_p"]) < .05 and int(r["a_only_hits"]) < int(r["b_only_hits"]) for r in rows))
    # Learning curves on the development set.
    curves = read_csv(d.report / "learning_curves.csv")
    m["DevEpisodes"] = str(d.summary[f"{BASELINES[0]}_dev"]["episodes"]) if f"{BASELINES[0]}_dev" in d.summary else "72"
    m["CurveStepZeroMax"] = pct(float(curves[0]["max"]))
    first90 = next((int(r["timesteps"]) for r in curves if float(r["mean"]) >= 0.9), None)
    m["CurveFirstNinety"] = f"{first90 / 1e6:.2f}M" if first90 else "never"
    best = max(curves, key=lambda r: float(r["mean"]))
    m["CurveBestMean"] = pct(float(best["mean"]))
    m["CurveBestSteps"] = f"{int(best['timesteps']) / 1e6:.2f}M"
    m["CurveFinalMean"] = pct(float(curves[-1]["mean"]))
    m["CurveFinalMin"] = pct(float(curves[-1]["min"]))
    m["CurveFinalMax"] = pct(float(curves[-1]["max"]))
    m["CurveAnyHundred"] = str(sum(any(float(r[f"seed{s}"]) >= 1.0 for r in curves) for s in SEEDS))
    for b in BASELINES:
        if f"{b}_dev" in d.summary:
            m[f"{KEYS[b]}DevRate"] = pct(d.summary[f"{b}_dev"]["hit_rate"])
    runs = [json.loads((d.runs / "train" / f"seed{s}" / "run.json").read_text()) for s in SEEDS]
    m["TrainSteps"] = f"{runs[0]['num_timesteps']:,}".replace(",", "{,}")
    m["TrainHours"] = fmt(st.mean(r["wall_seconds"] for r in runs) / 3600)
    m["TrainCommit"] = runs[0]["provenance"]["git"]["commit"][:7]
    m["EvalCommit"] = next(iter(d.summary.values()))["provenance"]["git"]["commit"][:7]
    # Failures (all three test suites, all controllers incl. every PPO seed; no swaps).
    failures = [f for f in read_csv(d.report / "failures.csv")
                if not f["run"].startswith(("ppo_rule_speed", "rule_steer"))]
    over = [f for f in failures if f["category"] == "missile_overshoot"]
    m["FailTotal"] = str(len(failures))
    m["FailOvershoot"] = str(len(over))
    m["FailCollision"] = str(sum(f["category"] == "collision" for f in failures))
    m["FailGround"] = str(sum(f["category"] == "ground" for f in failures))
    m["FailOutrun"] = str(sum(f["category"] == "missile_outrun" for f in failures))
    m["FailNoLaunch"] = str(sum(f["category"] == "no_launch" for f in failures))
    m["FailOther"] = str(sum(f["category"] not in ("missile_overshoot", "collision") for f in failures))
    m["FailEpisodes"] = f"{sum(len(d.episodes[r]) for c in CONTROLLERS for s, _ in SUITES for r in d.runs_of(c, s)):,}".replace(",", "{,}")
    if over:
        closest = [float(f["missile_closest_approach_m"]) for f in over]
        m["OvershootClosestMedian"] = fmt(st.median(closest), 0)
        m["OvershootClosestMax"] = fmt(max(closest), 0)
        when = [float(f["missile_closest_after_launch_s"]) for f in over if f["missile_closest_after_launch_s"]]
        m["OvershootTimeMedian"] = fmt(st.median(when))
        m["OvershootTimeMax"] = fmt(max(when))
        m["OvershootBeamShare"] = pct(sum(aspect_class(float(f["launch_aspect_deg"])) == "beam" for f in over) / len(over))
    for controller in CONTROLLERS:
        prefix = "ppo_seed" if controller == "ppo" else controller
        mine = [f for f in failures if f["category"] == "collision" and f["run"].startswith(prefix)]
        key = KEYS[controller]
        m[f"{key}CollAfterLaunch"] = str(sum(f["launched"] == "True" for f in mine))
        m[f"{key}CollBeforeLaunch"] = str(sum(f["launched"] != "True" for f in mine))
        m[f"{key}CollTotal"] = str(len(mine))
    # Hit rate per launch-aspect class, pooled over the two sweeps.
    for controller in CONTROLLERS:
        ls = launched(d.pooled(controller, SWEEPS))
        for cls in ("head_on", "beam", "tail"):
            group = [e for e in ls if aspect_class(float(e["launch_aspect_deg"])) == cls]
            hits = sum(e["success"] == "True" for e in group)
            tag = {"head_on": "HeadOn", "beam": "Beam", "tail": "Tail"}[cls]
            m[f"{KEYS[controller]}{tag}Hit"] = pct(hits / len(group)) if group else "--"
            m[f"{KEYS[controller]}{tag}Launches"] = f"{len(group):,}".replace(",", "{,}")
    # Collision sensitivity: episodes closer than 20 / 30 m, per run, over the three suites.
    sens = {r["run"]: r for r in read_csv(d.report / "collision_sensitivity.csv")}
    for controller in CONTROLLERS:
        for radius in (10, 20, 30):
            total = sum(int(sens[r][f"below_{radius}m"]) for s, _ in SUITES for r in d.runs_of(controller, s))
            n = d.n_runs(controller)
            m[f"{KEYS[controller]}Close{radius}"] = fmt(total / n) if n > 1 else str(total)
    # Tuned baseline gains.
    gains = json.loads((HERE.parent / "pursuit" / "baseline_gains.json").read_text())
    for b in BASELINES:
        g, key = gains[b]["gains"], KEYS[b]
        m[f"{key}GainBank"] = f"{g['bank_per_deg']:g}"
        m[f"{key}GainYaw"] = f"{g['yaw_deg_per_unit']:g}"
        m[f"{key}GainNav"] = f"{g['nav_gain']:g}"
        m[f"{key}GainHold"] = f"{g['hold_range_m']:g}"
        m[f"{key}GainSpeed"] = f"{g['speed_per_m']:g}"
        m[f"{key}TuneHits"] = str(gains[b]["tuning_hits"])
        m[f"{key}TuneDefaultHits"] = str(gains[b]["default_gains_hits"])
    m["TuneEpisodes"] = str(gains[BASELINES[0]]["episodes"])
    m["TuneCandidates"] = str(gains[BASELINES[0]]["candidates"])
    return m


def write_macros(m, path):
    lines = ["% Generated by paper/make_figures.py -- do not edit."]
    for key, value in sorted(m.items()):
        lines.append(f"\\newcommand{{\\R{key}}}{{{value}}}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- tables
def table_comparison(d, path):
    """Table II: hit rate on the three suites, and the launch metrics (M6) on the two sweeps."""
    rows = []
    for controller in CONTROLLERS:
        cells = [LABELS[controller]] + [hit_cell(d, controller, s) for s, _ in SUITES]
        ls = launch_stats(d.pooled(controller, SWEEPS))
        cells += [f"{ls['time']:.1f}", f"{ls['aspect']:.0f}", f"{100 * ls['fav']:.1f}"]
        coll = sum(e["outcome"] == "collision" for s, _ in SUITES for e in d.pooled(controller, s))
        n = d.n_runs(controller)
        cells.append(f"{coll / n:.1f}" if n > 1 else str(coll))
        rows.append(" & ".join(cells) + r" \\")
    path.write_text("\n".join(rows) + "\n")


def table_paired(d, path):
    """PPO seeds significantly better / worse than each baseline (exact McNemar, Holm), per suite."""
    paired = read_csv(d.report / "paired_tests.csv")
    rows = []
    for b in BASELINES:
        cells = [LABELS[b]]
        for suite, _ in SUITES:
            part = [r for r in paired if r["suite"] == suite and r["b"] == b and r["family"] == "ppo_vs_baseline"]
            better = sum(float(r["mcnemar_p_holm"]) < .05 and int(r["a_only_hits"]) > int(r["b_only_hits"]) for r in part)
            worse = sum(float(r["mcnemar_p_holm"]) < .05 and int(r["a_only_hits"]) < int(r["b_only_hits"]) for r in part)
            disc = [f"{r['a_only_hits']}/{r['b_only_hits']}" for r in part]
            cells.append(f"{better}/{worse}")
            cells.append(r"\scriptsize " + ", ".join(disc))
        rows.append(" & ".join(cells) + r" \\")
    path.write_text("\n".join(rows) + "\n")


def table_swap(d, path):
    """M4: which controller supplies steering and which the throttle."""
    if not d.rule:
        path.write_text("")
        return
    rule = LABELS[d.rule]
    entries = (("ppo", "PPO", "PPO"), ("ppo_rule_speed", "PPO", f"rule"),
               ("rule_steer_ppo_speed", rule, "PPO"), (d.rule, rule, "rule"))
    rows = []
    for controller, steer, speed in entries:
        cells = [steer, speed]
        for suite in SWEEPS:
            rates = [100 * r for r in d.rates(controller, suite)]
            cells.append(f"{st.mean(rates):.1f}" + (f" ({min(rates):.1f}--{max(rates):.1f})" if len(rates) > 1 else ""))
        ls = launch_stats(d.pooled(controller, SWEEPS))
        cells += [f"{ls['time']:.1f}", f"{100 * ls['fav']:.1f}"]
        rows.append(" & ".join(cells) + r" \\")
    path.write_text("\n".join(rows) + "\n")


def table_scenarios(d, path):
    """The reviewed six geometries, hit rate per controller (PPO pooled over seeds)."""
    from pursuit.scenarios import REVIEWED_SIX
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for controller in CONTROLLERS:
        for e in d.pooled(controller, "reviewed_six"):
            cell = by[e["scenario_id"]][controller]
            cell[0] += e["success"] == "True"
            cell[1] += 1
    rows = []
    for name, (x, y, z), speed, heading in REVIEWED_SIX:
        bearing = math.degrees(math.atan2(x, y))
        d0 = math.sqrt(x * x + y * y + (z - 1000.0) ** 2)
        cells = [name, f"$({x / 1e3:.1f}, {y / 1e3:.1f}, {z / 1e3:.1f})$", f"{d0 / 1e3:.1f}",
                 f"${round(bearing):d}$", f"{heading:.0f}", f"{speed:.0f}"]
        for controller in CONTROLLERS:
            k, n = by[name][controller]
            cells.append(f"{100 * k / n:.1f}")
        rows.append(" & ".join(cells) + r" \\")
    path.write_text("\n".join(rows) + "\n")


def table_failures(d, path):
    """Failures per run over the three test suites, by category."""
    failures = read_csv(d.report / "failures.csv")
    rows = []
    for controller in CONTROLLERS:
        names = {r for s, _ in SUITES for r in d.runs_of(controller, s)}
        mine = [f for f in failures if f["run"] in names]
        n = d.n_runs(controller)
        count = lambda pred: sum(1 for f in mine if pred(f)) / n
        cells = [LABELS[controller],
                 count(lambda f: f["category"] == "missile_overshoot"),
                 count(lambda f: f["category"] == "collision" and f["launched"] != "True"),
                 count(lambda f: f["category"] == "collision" and f["launched"] == "True"),
                 count(lambda f: f["category"] not in ("missile_overshoot", "collision"))]
        rows.append(" & ".join([cells[0]] + [f"{c:.1f}" for c in cells[1:]]) + r" \\")
    path.write_text("\n".join(rows) + "\n")


# ---------------------------------------------------------------- figures
def fig_learning(d, path):
    """Hard development set (and the full sweep, if pursuit.sweep_curves was run)."""
    panels = [("learning_curves.csv", "dev")]
    if (d.report / "sweep_curves.csv").exists():
        panels.append(("sweep_curves.csv", "sweep"))
    fig, axes = plt.subplots(1, len(panels), figsize=(COL_W, 2.0), sharex=True, squeeze=False)
    for ax, (name, suite) in zip(axes[0], panels):
        curves = read_csv(d.report / name)
        x = np.array([int(r["timesteps"]) for r in curves]) / 1e6
        for s in SEEDS:
            ax.plot(x, [100 * float(r[f"seed{s}"]) for r in curves], lw=0.6, alpha=0.55)
        ax.fill_between(x, [100 * float(r["min"]) for r in curves], [100 * float(r["max"]) for r in curves],
                        color="k", alpha=0.10, lw=0)
        ax.plot(x, [100 * float(r["mean"]) for r in curves], color="k", lw=1.4, label="PPO mean")
        for b in BASELINES:
            if f"{b}_{suite}" in d.summary:
                ax.axhline(100 * d.summary[f"{b}_{suite}"]["hit_rate"], color=COLORS[b], ls="--", lw=0.9,
                           label=LABELS[b])
        ax.set_ylim(0, 102)
        ax.legend(loc="lower right", frameon=False, fontsize=5.5)
        ax.set_title("development set" if suite == "dev" else "full sweep")
        ax.set_xlabel("training steps (millions)")
        ax.set_xlim(0, x.max())
        ax.grid(alpha=0.3, lw=0.4)
    axes[0][0].set_ylabel("hit rate (%)")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_comparison(d, path):
    """(a) hit rate on the three suites; (b) sweep hit rate per initial bearing. Returns axis floors."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.2), gridspec_kw={"width_ratios": [1.2, 1.6]})
    values = []
    for i, (suite, _) in enumerate(SUITES):
        for j, controller in enumerate(CONTROLLERS):
            xpos = i * 5 + j
            color = COLORS[controller]
            if controller == "ppo":
                rates = [100 * r for r in d.rates(controller, suite)]
                a.scatter([xpos] * len(rates), rates, s=10, color=color, alpha=0.7, zorder=3)
                a.plot([xpos - 0.3, xpos + 0.3], [st.mean(rates)] * 2, color=color, lw=2)
                values += rates
            else:
                s = d.summary[f"{controller}_{suite}"]
                lo, hi = 100 * s["ci95"][0], 100 * s["ci95"][1]
                a.errorbar(xpos, 100 * s["hit_rate"], yerr=[[max(0.0, 100 * s["hit_rate"] - lo)],
                                                            [max(0.0, hi - 100 * s["hit_rate"])]],
                           fmt="o", color=color, ms=4, capsize=2, lw=1)
                values.append(lo)
    floor_a = math.floor(min(values) - 0.5)
    a.set_xticks([1.5, 6.5, 11.5])
    a.set_xticklabels(["six geometries", "sweep", "sweep, altitude offsets"])
    a.set_ylabel("hit rate (%)")
    a.set_ylim(floor_a, 100.3)
    a.set_title(f"(a) overall hit rate (axis starts at {floor_a}%)")
    a.grid(axis="y", alpha=0.3, lw=0.4)
    handles = [plt.Line2D([], [], color=COLORS[c], marker="o", ls="", ms=4, label=LABELS[c]) for c in CONTROLLERS]
    a.legend(handles=handles, loc="lower left", frameon=False, ncol=2, fontsize=6)
    sectors = list(range(0, 360, 30))
    lows = []
    for controller in CONTROLLERS:
        per_run = np.array([[100 * d.summary[r]["by_bearing_sector"][str(s)]["hit_rate"] for s in sectors]
                            for r in d.runs_of(controller, "sweep")])
        b.plot(sectors, per_run.mean(0), marker="o", ms=3, lw=1.2, color=COLORS[controller], label=LABELS[controller])
        if len(per_run) > 1:
            b.fill_between(sectors, per_run.min(0), per_run.max(0), color=COLORS[controller], alpha=0.15, lw=0)
        lows.append(per_run.min())
    floor_b = int(5 * math.floor((min(lows) - 1) / 5))
    b.set_xticks(sectors)
    b.set_xlabel("initial target bearing (deg clockwise from our nose; 180 = behind)")
    b.set_ylabel("hit rate (%)")
    b.set_ylim(floor_b, 100.5)
    b.set_title(f"(b) sweep, by initial bearing (axis starts at {floor_b}%)")
    b.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return floor_a, floor_b


def fig_aspect(d, path):
    """(a) where each controller launches: share of launches per aspect bin;
    (b) hits per launch in each bin, with the number of launches printed on every bar."""
    table = read_csv(d.report / "aspect_table.csv")
    bins = ["head-on 0-45", "45-90", "90-135", "tail 135-180"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.6), sharex=True)
    width = 0.2
    floor = 100.0
    for j, controller in enumerate(CONTROLLERS):
        n_runs = d.n_runs(controller)
        counts, rates, errs = [], [], [[], []]
        for label in bins:
            cells = [r for r in table if r["controller"] == controller and r["launch_aspect"] == label
                     and r["suite"] in SWEEPS]
            n = sum(int(r["launches"]) for r in cells)
            k = sum(int(r["hits"]) for r in cells)
            counts.append(n)
            p = 100 * k / n if n else math.nan
            lo, hi = wilson(k, n)
            rates.append(p)
            errs[0].append(max(0.0, p - 100 * lo) if n else 0)
            errs[1].append(max(0.0, 100 * hi - p) if n else 0)
            if n:
                floor = min(floor, 100 * lo)
        xs = np.arange(len(bins)) + (j - 1.5) * width
        share = 100 * np.array(counts) / max(1, sum(counts))
        a.bar(xs, share, width, color=COLORS[controller], label=LABELS[controller])
        b.bar(xs, rates, width, color=COLORS[controller], yerr=errs, error_kw={"lw": 0.6, "capsize": 1.5})
        for x, n in zip(xs, counts):
            label = f"{n / n_runs:.0f}" if n_runs > 1 else str(n)
            b.text(x, 100.4, label, ha="center", va="bottom", fontsize=4.5, rotation=90)
    a.set_ylabel("share of launches (%)")
    a.set_title("(a) target aspect at launch (two sweeps)")
    a.grid(axis="y", alpha=0.3, lw=0.4)
    a.legend(loc="upper center", ncol=2, frameon=False, fontsize=5.5)
    floor = int(5 * math.floor((floor - 1) / 5))
    b.set_ylim(floor, 103.5)
    b.set_ylabel("hits per launch (%)")
    b.set_title(f"(b) hits per launch (axis starts at {floor}%; numbers: launches per run)")
    b.set_xticks(range(len(bins)))
    b.set_xticklabels(["0-45\n(head-on)", "45-90", "90-135", "135-180\n(tail)"])
    b.set_xlabel("target aspect at launch (deg)")
    b.grid(axis="y", alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return floor


def fig_overshoot(d, path):
    """Top-down view of one typical overshoot miss (median closest approach, PPO, sweep)."""
    failures = [f for f in read_csv(d.report / "failures.csv")
                if f["category"] == "missile_overshoot" and f["run"].startswith("ppo_seed") and f["suite"] == "sweep"]
    if not failures:
        failures = [f for f in read_csv(d.report / "failures.csv") if f["category"] == "missile_overshoot"]
    failures.sort(key=lambda f: float(f["missile_closest_approach_m"]))
    pick = failures[len(failures) // 2]
    rows = read_csv(d.eval / pick["trajectory_file"])
    num = lambda r, k: float(r[k])
    t = np.array([num(r, "t") for r in rows])
    P = np.array([[num(r, "p_x"), num(r, "p_y")] for r in rows])
    T = np.array([[num(r, "t_x"), num(r, "t_y")] for r in rows])
    M = np.array([[num(r, "m_x"), num(r, "m_y")] for r in rows])
    li = int(np.argmax(np.array([r["launched"] == "True" for r in rows])))
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
    parser.add_argument("--root", type=Path, default=Path("runs/final"),
                        help="output root of pursuit.run_all (holds train/, eval/, report/)")
    args = parser.parse_args(argv)
    d = Data(args.root)
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "generated").mkdir(exist_ok=True)
    macros = build_macros(d)
    fig_learning(d, HERE / "figures/fig_learning.pdf")
    floor_a, floor_b = fig_comparison(d, HERE / "figures/fig_comparison.pdf")
    floor_aspect = fig_aspect(d, HERE / "figures/fig_aspect.pdf")
    pick = fig_overshoot(d, HERE / "figures/fig_overshoot.pdf")
    macros.update(FigCompFloor=str(floor_a), FigBearingFloor=str(floor_b), FigAspectFloor=str(floor_aspect),
                  OvershootExampleId=pick["scenario_id"].replace("_", r"\_"),
                  OvershootExampleRun=pick["run"].replace("_", r"\_"),
                  OvershootExampleClosest=fmt(float(pick["missile_closest_approach_m"]), 0),
                  OvershootExampleAspect=fmt(float(pick["launch_aspect_deg"]), 0))
    write_macros(macros, HERE / "generated/results.tex")
    table_comparison(d, HERE / "generated/tab_comparison.tex")
    table_paired(d, HERE / "generated/tab_paired.tex")
    table_swap(d, HERE / "generated/tab_swap.tex")
    table_scenarios(d, HERE / "generated/tab_scenarios.tex")
    table_failures(d, HERE / "generated/tab_failures.tex")
    print("figures and tables written to", HERE)


if __name__ == "__main__":
    main()
