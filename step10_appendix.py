#!/usr/bin/env python3
"""
QPPE Rebuild - Step 10: Statistical Appendix Generator
=======================================================
Writes appendix.md: every number quoted in the paper, either
RECOMPUTED from primary data (qppe PostgreSQL DB; DuckDB/MySQL corpus
pickles) or EMBEDDED with step-level provenance.

Recomputed here (verification pass - compare against step outputs):
  A. corpus statistics per engine x benchmark
  B. materially-severe / mild / win label counts
  C. workload bounds (default vs oracle)
  D. cross-conformal policy-level calibration (t*, n steered, k severe)
  E. sample-size floor table (analytic)

Embedded with provenance (transcribed from step outputs):
  F. live-run summaries (Steps 7i, 9a, 9b)
  G. E1 / E2 / E3 summaries (Steps 5, 7c, 7f)

Requires: step6_live_loop.py, step7c_job_analysis.py,
step9a_duckdb_port.py, step9b_mysql_port.py, and the corpus pickles
qppe_duckdb_corpus.pkl / qppe_mysql_corpus.pkl in the same folder.

Usage:
    py step10_appendix.py --user postgres --password 12345
Runtime: ~5 minutes (recomputes 4 calibrations).
"""

import argparse
import importlib.util
import math
import pathlib
import pickle

import numpy as np
import pandas as pd

RATIO_EPS = 1e-9

EMBEDDED = {
    "live": [
        # (engine/benchmark, steers, of, severe, mild, net_pct, ovh_ms, source)
        ("PostgreSQL / TPC-H", 9, 17, 0, 0, +15.0, "26-125", "Step 7i"),
        ("PostgreSQL / JOB", 0, 15, 0, 0, -17.5, "944", "Step 7i"),
        ("DuckDB / TPC-H", 0, 17, 0, 0, -50.2, "55", "Step 9a"),
        ("MySQL / TPC-H", 0, 17, 0, 0, -1.2, "62", "Step 9b"),
    ],
    "e1_tpch_step5": {
        "seeds": 20, "alpha": 0.10, "reg_rate_mean": 0.021,
        "violations": "1/20", "speedup_mean": 9.6, "capture_mean": 26.8,
        "source": "Step 5 (theta=1.2 event, candidate-level)"},
    "e1_job_step7f": {
        "seeds": 20, "alpha": 0.10, "severe_rate_mean": 0.087,
        "violations": "7/20 (ANTI-CONSERVATIVE - motivates Sec 4.3)",
        "speedup_mean": 22.8,
        "source": "Step 7f (materially-severe, candidate-level)"},
    "e2": [
        ("PG/TPC-H template shift", "rate 0.30 vs alpha 0.10 "
         "(33 steers, 10 reg)", "Step 5"),
        ("PG/JOB family shift", "rate 0.32 vs alpha 0.10 "
         "(25 steers, 8 severe)", "Step 7f"),
    ],
    "e3": ("TPC-H-trained gate on all 113 JOB queries: 11 steers, "
           "0 regressions, +0.1% speedup - safety transfers, utility "
           "does not", "Step 7c"),
}


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, str(pathlib.Path(__file__).parent / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def new_model():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=8, random_state=42)


def cp_upper(k, n, delta):
    from scipy.stats import beta
    if n == 0 or k == n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def add_labels(cand, args):
    cand = cand.copy()
    slow = cand["median_ms"] / (cand["def_median_ms"] + RATIO_EPS)
    added = cand["median_ms"] - cand["def_median_ms"]
    cand["is_severe"] = (((slow > args.theta_sev) &
                          (added > args.floor_ms)) |
                         cand["censored"]).astype(int)
    cand["is_mild"] = ((slow > 1.2) & ~(
        (slow > args.theta_sev) & (added > args.floor_ms)) &
        ~cand["censored"]).astype(int)
    cand["is_win"] = ((slow < args.theta_win) & ~cand["censored"]).astype(int)
    return cand


def feats_of(cand):
    return [c for c in cand.columns
            if c not in ("template", "query_id", "median_ms",
                         "def_median_ms", "censored", "is_severe",
                         "is_mild", "is_win", "p_severe", "p_win")]


def calibrate(cand, args):
    """Cross-conformal policy-level calibration (verification pass)."""
    feats = feats_of(cand)
    cand = cand.reset_index(drop=True).copy()
    rng = np.random.default_rng(11)
    qids = rng.permutation(cand["query_id"].unique())
    folds = np.array_split(qids, 5)
    cand["p_severe"] = np.nan
    cand["p_win"] = np.nan
    for fold_q in folds:
        te = cand.query_id.isin(set(fold_q))
        ms_ = new_model()
        ms_.fit(cand.loc[~te, feats], cand.loc[~te, "is_severe"])
        cand.loc[te, "p_severe"] = ms_.predict_proba(
            cand.loc[te, feats])[:, 1]
        mw_ = new_model()
        mw_.fit(cand.loc[~te, feats], cand.loc[~te, "is_win"])
        cand.loc[te, "p_win"] = mw_.predict_proba(cand.loc[te, feats])[:, 1]
    grid = np.unique(np.quantile(cand["p_severe"].values,
                                 np.linspace(0.05, 0.95, 30)))
    t_star, bn, bk = 0.0, 0, 0
    for t in grid:
        n, k = 0, 0
        for qid, g in cand.groupby("query_id"):
            cert = g[(g["p_severe"] < t) & (g["p_win"] > args.tau_win)]
            if cert.empty:
                continue
            top = cert.loc[cert["p_win"].idxmax()]
            n += 1
            s = top["median_ms"] / (top["def_median_ms"] + RATIO_EPS)
            if ((s > args.theta_sev and
                 top["median_ms"] - top["def_median_ms"] > args.floor_ms)
                    or top["censored"]):
                k += 1
        if n > 0 and cp_upper(k, n, args.delta) <= args.alpha:
            t_star, bn, bk = float(t), n, k
    return t_star, bn, bk


def pickle_to_frame(pkl_file, build_row):
    with open(pkl_file, "rb") as f:
        blob = pickle.load(f)
    records = blob["records"]
    hint_names = (list(blob["hint_sets"])
                  if not isinstance(blob["hint_sets"], list)
                  else blob["hint_sets"])
    df = pd.DataFrame(records)
    defaults = (df[df.hint_set == "default"]
                .set_index("query_id")[["median_ms", "features"]]
                .rename(columns={"median_ms": "def_ms",
                                 "features": "def_features"}))
    canddf = df[df.hint_set != "default"].join(defaults, on="query_id")
    rows = []
    for _, r in canddf.iterrows():
        row = build_row(r["features"], r["def_features"], r["hint_set"],
                        hint_names)
        row.update(template=r["template"], query_id=r["query_id"],
                   median_ms=r["median_ms"], def_median_ms=r["def_ms"],
                   censored=bool(r["censored"]))
        rows.append(row)
    return pd.DataFrame(rows)


def corpus_block(name, cand, args, out):
    cand = add_labels(cand, args)
    n_q = cand["query_id"].nunique()
    td, to_ = 0.0, 0.0
    for qid, g in cand.groupby("query_id"):
        d = g["def_median_ms"].iloc[0]
        td += d
        to_ += min(d, g["median_ms"].min())
    t_star, bn, bk = calibrate(cand, args)
    out.append(f"### {name}\n")
    out.append(f"| metric | value |\n|---|---|")
    out.append(f"| queries with candidates | {n_q} |")
    out.append(f"| candidates | {len(cand)} |")
    out.append(f"| censored (timeout) | {int(cand.censored.sum())} |")
    out.append(f"| materially severe | {int(cand.is_severe.sum())} "
               f"({cand.is_severe.mean():.0%}) |")
    out.append(f"| mild (1.2x..severe) | {int(cand.is_mild.sum())} |")
    out.append(f"| wins (<0.9x) | {int(cand.is_win.sum())} "
               f"({cand.is_win.mean():.0%}) |")
    out.append(f"| default workload | {td/1000:.1f} s |")
    out.append(f"| oracle workload | {to_/1000:.1f} s |")
    out.append(f"| oracle headroom | {(td-to_)/td:.0%} |")
    out.append(f"| cross-conformal t* (recomputed) | {t_star:.3f} |")
    out.append(f"| calibration record | {bn} steered, {bk} severe |")
    out.append("")
    return t_star


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--log-db", default="qppe")
    parser.add_argument("--theta-sev", type=float, default=2.0)
    parser.add_argument("--floor-ms", type=float, default=1000.0)
    parser.add_argument("--theta-win", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--tau-win", type=float, default=0.5)
    args = parser.parse_args()

    s6 = load_module("s6", "step6_live_loop.py")
    s7c = load_module("s7c", "step7c_job_analysis.py")

    print("QPPE Step 10 - generating appendix.md")
    out = []
    out.append("# Statistical Appendix (generated by step10_appendix.py)\n")
    out.append(f"Certified event: slowdown > {args.theta_sev}x AND added "
               f"> {args.floor_ms:.0f} ms, or timeout. "
               f"alpha={args.alpha}, delta={args.delta}, "
               f"tau_win={args.tau_win}.\n")

    # ---- A-D: recomputed corpus blocks ----
    out.append("## A. Corpus statistics, labels, bounds, calibration "
               "(RECOMPUTED)\n")
    print("  PostgreSQL / TPC-H ...")
    tpch = s7c.load_benchmark(args, s6, "tpch_sf1", variant_max=100)
    corpus_block("PostgreSQL / TPC-H SF1", tpch, args, out)

    print("  PostgreSQL / JOB ...")
    job = s7c.load_benchmark(args, s6, "imdb")
    corpus_block("PostgreSQL / JOB (IMDB)", job, args, out)

    if pathlib.Path("qppe_duckdb_corpus.pkl").exists():
        print("  DuckDB / TPC-H ...")
        s9a = load_module("s9a", "step9a_duckdb_port.py")
        duck = pickle_to_frame("qppe_duckdb_corpus.pkl", s9a.build_row)
        corpus_block("DuckDB / TPC-H SF1", duck, args, out)
    else:
        out.append("### DuckDB - corpus pickle not found, skipped\n")

    if pathlib.Path("qppe_mysql_corpus.pkl").exists():
        print("  MySQL / TPC-H ...")
        s9b = load_module("s9b", "step9b_mysql_port.py")
        my = pickle_to_frame("qppe_mysql_corpus.pkl", s9b.build_row)
        corpus_block("MySQL / TPC-H SF1", my, args, out)
    else:
        out.append("### MySQL - corpus pickle not found, skipped\n")

    # ---- E: sample-size floor ----
    out.append("## E. Sample-size floor (ANALYTIC)\n")
    out.append("Clean steered calibration queries required for a single "
               "Clopper-Pearson certificate with zero observed failures: "
               "n_min = ceil(ln(delta) / ln(1 - alpha))\n")
    out.append("| alpha | delta=0.05 | delta=0.10 | delta=0.20 |")
    out.append("|---|---|---|---|")
    for a in [0.05, 0.10, 0.15, 0.20]:
        row = [str(math.ceil(math.log(d) / math.log(1 - a)))
               for d in [0.05, 0.10, 0.20]]
        out.append(f"| {a} | " + " | ".join(row) + " |")
    out.append("")

    # ---- F: live runs (embedded) ----
    out.append("## F. Live fresh-query runs (EMBEDDED - provenance cited)\n")
    out.append("| engine / benchmark | steers | severe | mild | net % "
               "(ovh incl.) | overhead ms | source |")
    out.append("|---|---|---|---|---|---|---|")
    for (nm, st, of, sv, mi, net, ovh, src) in EMBEDDED["live"]:
        out.append(f"| {nm} | {st}/{of} | {sv} | {mi} | {net:+.1f} | "
                   f"{ovh} | {src} |")
    out.append("")

    # ---- G: E1/E2/E3 (embedded) ----
    out.append("## G. Split-based evaluations (EMBEDDED)\n")
    e = EMBEDDED["e1_tpch_step5"]
    out.append(f"**E1 PG/TPC-H** ({e['source']}): realized rate "
               f"{e['reg_rate_mean']} vs alpha {e['alpha']}, violations "
               f"{e['violations']}, speedup {e['speedup_mean']}%, capture "
               f"{e['capture_mean']}%.\n")
    e = EMBEDDED["e1_job_step7f"]
    out.append(f"**E1 PG/JOB** ({e['source']}): severe rate "
               f"{e['severe_rate_mean']}, violations {e['violations']}, "
               f"speedup {e['speedup_mean']}%. These numbers demonstrate "
               f"the anti-conservativeness of candidate-level calibration "
               f"and are NOT quotable as certified results.\n")
    out.append("**E2 shift boundaries:**\n")
    for name, val, src in EMBEDDED["e2"]:
        out.append(f"- {name}: {val} ({src})")
    out.append("")
    out.append(f"**E3 transfer:** {EMBEDDED['e3'][0]} "
               f"({EMBEDDED['e3'][1]}).\n")

    with open("appendix.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("Wrote appendix.md - diff the recomputed blocks against the "
          "step outputs as a final audit.")


if __name__ == "__main__":
    main()
