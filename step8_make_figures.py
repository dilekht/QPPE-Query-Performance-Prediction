#!/usr/bin/env python3
"""
QPPE Rebuild - Step 8: Paper Figures
=====================================
Generates the paper's figures into ./figures/ (PNG 300dpi + PDF).

  DB-DRIVEN (computed fresh from the qppe database):
    F1  believed-vs-actual: optimizer cost ratio vs measured slowdown
    F2  oracle gap per benchmark (never-steer vs oracle)
    F3  slowdown distribution by hint set (JOB)
    F4  materially-severe base rate by hint set and benchmark

  EMBEDDED (transcribed from experiment outputs; provenance in
  comments - audit against the step outputs before submission):
    F5  sample-size floor n_min(alpha, delta)
    F6  TPC-H certified live run, per-query default vs gated (Step 7i)
    F7  decision overhead per benchmark (Steps 6b / 7i)
    F8  calibration constructions: guarantee violations (Steps 5/7f/7i)

Requires: matplotlib (py -m pip install matplotlib if missing),
plus step6_live_loop.py and step7c_job_analysis.py in the folder.

Usage:
    py step8_make_figures.py --user postgres --password 12345
"""

import argparse
import importlib.util
import math
import os
import pathlib

import numpy as np
import pandas as pd

RATIO_EPS = 1e-9

# ======================================================================
# EMBEDDED RESULTS - transcribed from experiment outputs.
# Every entry cites its source step. Audit before submission.
# ======================================================================
EMBEDDED = {
    # Step 7i, TPC-H live table (fresh v100+ queries)
    "tpch_live": [
        # (query, default_ms, chosen_ms, steered)
        ("Q1",  2889,  2889, False), ("Q5a",  648,   723, True),
        ("Q5b",  736,   736, False), ("Q6",   868,   868, False),
        ("Q7a", 1676,  1008, True),  ("Q7b", 1657,   976, True),
        ("Q8",   381,   381, False), ("Q9",  2591,  2564, True),
        ("Q10", 1464,   695, True),  ("Q12",  916,   769, True),
        ("Q13", 3115,  1541, True),  ("Q14",  378,   378, False),
        ("Q17", 2199,   441, True),  ("Q18", 14945, 12878, True),
        ("Q19",  251,   251, False), ("Q21", 1538,  1538, False),
        ("Q22",  331,   331, False),
    ],
    # Step 6b / Step 7i overhead measurements (ms, per-query averages)
    "overhead": {
        "TPC-H (steady state)": {"gen": 20.5, "inf": 5.2},     # Step 6b
        "TPC-H (7i, incl. warm-up)": {"gen": 118.0, "inf": 7.3},  # Step 7i approx split
        "JOB (7i)": {"gen": 925.0, "inf": 19.5},                # Step 7i approx split
    },
    # Guarantee-violation counts across 20 seeds, by construction
    "violations": [
        # (label, violations, total, valid_construction)
        ("candidate-level\nTPC-H (Step 5 E1)", 1, 20, False),
        ("candidate-level\nJOB (Step 7f E1)", 7, 20, False),
        ("policy-level, query units\nTPC-H (Step 7i live)", 0, 1, True),
        ("policy-level, query units\nJOB (Step 7i live: abstained)", 0, 1, True),
    ],
}


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, str(pathlib.Path(__file__).parent / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def savefig(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"),
                    dpi=300, bbox_inches="tight")
    print(f"  wrote figures/{name}.png + .pdf")


def load_candidates(args, s6, s7c):
    frames = []
    for label, bname, vmax in [("TPC-H", "tpch_sf1", 100),
                               ("JOB", "imdb", None)]:
        c = s7c.load_benchmark(args, s6, bname, variant_max=vmax)
        c = c.reset_index(drop=True)
        c["benchmark"] = label
        c["slowdown"] = c["median_ms"] / (c["def_median_ms"] + RATIO_EPS)
        frames.append(c)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--log-db", default="qppe")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--theta-sev", type=float, default=2.0)
    parser.add_argument("--floor-ms", type=float, default=1000.0)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "figure.facecolor": "white"})

    s6 = load_module("s6", "step6_live_loop.py")
    s7c = load_module("s7c", "step7c_job_analysis.py")

    os.makedirs(args.outdir, exist_ok=True)
    print("QPPE Step 8 - generating figures")
    print("=" * 60)

    cand = load_candidates(args, s6, s7c)
    # recover hint set name from one-hot columns
    hs_cols = [c for c in cand.columns if c.startswith("hs_")]
    cand["hint_set"] = cand[hs_cols].idxmax(axis=1).str[3:]
    added = cand["median_ms"] - cand["def_median_ms"]
    cand["is_severe"] = (((cand["slowdown"] > args.theta_sev) &
                          (added > args.floor_ms)) |
                         cand["censored"]).astype(int)

    # ---------------- F1: believed vs actual ----------------
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for label, marker, color in [("TPC-H", "o", "#4878b0"),
                                 ("JOB", "^", "#d1615d")]:
        g = cand[cand.benchmark == label]
        ax.scatter(g["cost_ratio"].clip(1e-2, 1e4),
                   g["slowdown"].clip(1e-2, 1e4),
                   s=8, alpha=0.35, marker=marker, label=label,
                   color=color, linewidths=0)
    ax.axhline(1, color="gray", lw=0.6, ls="--")
    ax.axvline(1, color="gray", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("optimizer-believed cost ratio (candidate / default)")
    ax.set_ylabel("measured slowdown (candidate / default)")
    ax.set_title("Optimizer belief vs. measured reality")
    ax.legend(frameon=False)
    savefig(fig, args.outdir, "f1_believed_vs_actual")
    plt.close(fig)

    # quadrant statistics for the caption
    q = cand[(cand.cost_ratio > 1) & (cand.slowdown < 0.9)]
    print(f"  F1 caption stat: {len(q)} candidates "
          f"({len(q)/len(cand):.0%}) are believed-worse but measured-faster")

    # ---------------- F2: oracle gap ----------------
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    labels, never, oracle = [], [], []
    for label, g in cand.groupby("benchmark"):
        per_q = g.groupby("query_id").agg(
            d=("def_median_ms", "first"))
        best = g.groupby("query_id").apply(
            lambda x: min(x["def_median_ms"].iloc[0], x["median_ms"].min()))
        labels.append(label)
        never.append(per_q["d"].sum() / 1000)
        oracle.append(best.sum() / 1000)
    x = np.arange(len(labels))
    ax.bar(x - 0.18, never, 0.36, label="PostgreSQL default", color="#777777")
    ax.bar(x + 0.18, oracle, 0.36, label="oracle (best candidate)",
           color="#4878b0")
    for i in range(len(labels)):
        gap = (never[i] - oracle[i]) / never[i] * 100
        ax.text(x[i], max(never[i], oracle[i]) * 1.02, f"gap {gap:.0f}%",
                ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("total workload time (s)")
    ax.set_title("Headroom left by the default optimizer")
    ax.legend(frameon=False)
    savefig(fig, args.outdir, "f2_oracle_gap")
    plt.close(fig)

    # ---------------- F3: slowdown by hint set (JOB) ----------------
    job = cand[cand.benchmark == "JOB"]
    order = (job.groupby("hint_set")["slowdown"].median()
             .sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    data = [np.log10(job[job.hint_set == h]["slowdown"].clip(1e-2, 1e3))
            for h in order]
    bp = ax.boxplot(data, labels=order, showfliers=True,
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.axhline(math.log10(args.theta_sev), color="#d1615d", lw=0.8, ls=":",
               label=f"severe threshold ({args.theta_sev}x)")
    ax.set_ylabel("log10 slowdown vs default")
    ax.set_title("JOB: slowdown distribution by hint set")
    ax.legend(frameon=False, loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    savefig(fig, args.outdir, "f3_job_slowdown_by_hintset")
    plt.close(fig)

    # ---------------- F4: severe base rates ----------------
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    hs_order = (cand.groupby("hint_set")["is_severe"].mean()
                .sort_values().index.tolist())
    width = 0.38
    x = np.arange(len(hs_order))
    for i, (label, color) in enumerate([("TPC-H", "#4878b0"),
                                        ("JOB", "#d1615d")]):
        g = cand[cand.benchmark == label]
        rates = [g[g.hint_set == h]["is_severe"].mean() if
                 (g.hint_set == h).any() else 0 for h in hs_order]
        ax.bar(x + (i - 0.5) * width, rates, width, label=label, color=color)
    ax.set_xticks(x, hs_order)
    ax.set_ylabel("materially-severe rate")
    ax.set_title("Severe-regression base rate by hint set")
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    savefig(fig, args.outdir, "f4_severe_base_rates")
    plt.close(fig)

    # ---------------- F5: sample-size floor (analytic) ----------------
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    alphas = np.linspace(0.02, 0.30, 200)
    for delta, ls in [(0.05, "-"), (0.10, "--"), (0.20, ":")]:
        n = np.ceil(np.log(delta) / np.log(1 - alphas))
        ax.plot(alphas, n, ls, color="#333333", label=f"delta = {delta}")
    ax.scatter([0.10], [22], color="#d1615d", zorder=5)
    ax.annotate("(0.10, 0.10) -> 22 queries", (0.10, 22),
                textcoords="offset points", xytext=(8, 8), fontsize=8)
    ax.set_xlabel("target severe rate alpha")
    ax.set_ylabel("clean steered calibration queries required")
    ax.set_ylim(0, 120)
    ax.set_title("Sample-size floor of certified steering")
    ax.legend(frameon=False)
    savefig(fig, args.outdir, "f5_sample_size_floor")
    plt.close(fig)

    # ---------------- F6: TPC-H certified live (embedded) ----------------
    live = EMBEDDED["tpch_live"]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(live))
    d = [r[1] / 1000 for r in live]
    c = [r[2] / 1000 for r in live]
    steered = [r[3] for r in live]
    ax.bar(x - 0.19, d, 0.38, label="default", color="#777777")
    ax.bar(x + 0.19, c, 0.38, label="gated policy",
           color=["#4878b0" if s else "#bbbbbb" for s in steered])
    ax.set_yscale("log")
    ax.set_xticks(x, [r[0] for r in live])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("runtime (s, log scale)")
    ax.set_title("TPC-H certified live run: 9/17 steered, 0 severe, "
                 "+15.0% net (Step 7i)")
    ax.legend(frameon=False)
    savefig(fig, args.outdir, "f6_tpch_certified_live")
    plt.close(fig)

    # ---------------- F7: decision overhead (embedded) ----------------
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    labels = list(EMBEDDED["overhead"].keys())
    gen = [EMBEDDED["overhead"][k]["gen"] for k in labels]
    inf = [EMBEDDED["overhead"][k]["inf"] for k in labels]
    x = np.arange(len(labels))
    ax.bar(x, gen, 0.5, label="candidate generation (12x EXPLAIN)",
           color="#4878b0")
    ax.bar(x, inf, 0.5, bottom=gen, label="featurize + inference",
           color="#d1615d")
    ax.set_xticks(x, labels)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("decision overhead per query (ms)")
    ax.set_title("Measured decision overhead")
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, args.outdir, "f7_overhead")
    plt.close(fig)

    # ---------------- F8: constructions comparison (embedded) --------
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    v = EMBEDDED["violations"][:2]  # the two 20-seed candidate-level rows
    labels = [r[0] for r in v]
    rates = [r[1] / r[2] for r in v]
    x = np.arange(len(labels))
    ax.bar(x, rates, 0.5, color=["#88a8cc", "#d1615d"])
    ax.axhline(0.10, color="#333333", lw=0.9, ls="--",
               label="expected at delta = 0.10")
    for i, r in enumerate(v):
        ax.text(x[i], rates[i] + 0.01, f"{r[1]}/{r[2]} seeds",
                ha="center", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("fraction of seeds violating alpha")
    ax.set_title("Candidate-level calibration is anti-conservative\n"
                 "(policy-level query-unit construction replaces it)")
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, args.outdir, "f8_construction_violations")
    plt.close(fig)

    print("\nAll 8 figures written to ./figures/")
    print("F1-F4 computed from the database; F5 analytic; F6-F8 from the")
    print("EMBEDDED block (provenance in comments) - audit before use.")


if __name__ == "__main__":
    main()
