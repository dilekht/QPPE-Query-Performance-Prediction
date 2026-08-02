#!/usr/bin/env python3
"""
QPPE Rebuild - Step 7e: Severity-Aware Conformal Gate
======================================================
Step 7d diagnosis: the theta=1.2 regression label mixes unpredictable
noise-level slowdowns with catastrophic ones; one mislabeled-in-spirit
borderline case caps the CP clean prefix at 2 and seals the gate.

Redesign - the CERTIFIED event is now SEVERE regression:
    severe  := slowdown > theta_sev (default 2.0) OR censored
    mild    := theta_reg < slowdown <= theta_sev   (reported, not certified)
Policy: certify candidates with p_severe < t* (pooled CP, alpha, delta),
steer to the certified candidate with highest p_win IF p_win > tau_win.

Outputs:
  - diagnostics for the severe head (AUC, clean prefix)
  - honest E1 (20 random splits): guaranteed severe rate vs alpha,
    plus unguaranteed mild rate, speedup, capture
  - honest E2 (33 family folds): the shift boundary for the new gate

Usage:
    py step7e_severity_gate.py --user postgres --password 12345
Runtime: ~5-10 minutes, no query execution.
"""

import argparse
import importlib.util
import pathlib

import numpy as np
import pandas as pd

RATIO_EPS = 1e-9


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, str(pathlib.Path(__file__).parent / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_labels(cand, theta_reg, theta_sev, theta_win):
    cand = cand.copy()
    cand["slowdown"] = cand["median_ms"] / (cand["def_median_ms"] + RATIO_EPS)
    cand["is_severe"] = ((cand["slowdown"] > theta_sev) |
                         cand["censored"]).astype(int)
    cand["is_mild"] = ((cand["slowdown"] > theta_reg) &
                       (cand["slowdown"] <= theta_sev) &
                       ~cand["censored"]).astype(int)
    cand["is_win"] = ((cand["slowdown"] < theta_win) &
                      ~cand["censored"]).astype(int)
    return cand


def fit_heads(train, feats):
    from sklearn.ensemble import GradientBoostingClassifier
    heads = {}
    for lab in ["is_severe", "is_win"]:
        m = GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        m.fit(train[feats], train[lab])
        heads[lab] = m
    return heads


def simulate(test, t_star, tau_win, theta_reg, theta_sev):
    rows = []
    for qid, g in test.groupby("query_id"):
        d_ms = g["def_median_ms"].iloc[0]
        best = min(d_ms, g["median_ms"].min())
        cert = g[(g["p_severe"] < t_star) & (g["p_win"] > tau_win)]
        steered, chosen = False, d_ms
        if not cert.empty:
            top = cert.loc[cert["p_win"].idxmax()]
            steered, chosen = True, top["median_ms"]
        slow = chosen / d_ms if d_ms > 0 else 1.0
        rows.append(dict(default_ms=d_ms, oracle_ms=best, policy_ms=chosen,
                         steered=steered,
                         severe=steered and slow > theta_sev,
                         mild=steered and theta_reg < slow <= theta_sev))
    P = pd.DataFrame(rows)
    td, tp_, to_ = P.default_ms.sum(), P.policy_ms.sum(), P.oracle_ms.sum()
    n_st = int(P.steered.sum())
    n_sv = int(P.severe.sum())
    n_mi = int(P.mild.sum())
    return dict(n_steer=n_st, n_severe=n_sv, n_mild=n_mi,
                severe_rate=n_sv / n_st if n_st else 0.0,
                mild_rate=n_mi / n_st if n_st else 0.0,
                speedup_pct=(td - tp_) / td * 100,
                capture_pct=(td - tp_) / (td - to_) * 100 if td > to_ else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--log-db", default="qppe")
    parser.add_argument("--theta-reg", type=float, default=1.2)
    parser.add_argument("--theta-sev", type=float, default=2.0)
    parser.add_argument("--theta-win", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--tau-win", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    from sklearn.metrics import roc_auc_score

    s6 = load_module("s6", "step6_live_loop.py")
    s7c = load_module("s7c", "step7c_job_analysis.py")

    print("QPPE Step 7e - Severity-Aware Conformal Gate (JOB)")
    print("=" * 72)
    print(f"Certified event: slowdown > {args.theta_sev}x or timeout | "
          f"alpha={args.alpha}, delta={args.delta}, tau_win={args.tau_win}")

    raw = s7c.load_benchmark(args, s6, "imdb")
    job = add_labels(raw, args.theta_reg, args.theta_sev, args.theta_win)
    job = job.reset_index(drop=True)
    feats = [c for c in job.columns
             if c not in ("template", "query_id", "median_ms", "def_median_ms",
                          "censored", "slowdown", "is_severe", "is_mild",
                          "is_win")]
    print(f"Candidates: {len(job)} | severe: {int(job.is_severe.sum())} | "
          f"mild: {int(job.is_mild.sum())} | wins: {int(job.is_win.sum())}")

    # ------------------------------------------------------------
    # diagnostics on the severe head (out-of-fold, 5 folds)
    # ------------------------------------------------------------
    from sklearn.ensemble import GradientBoostingClassifier
    rng = np.random.default_rng(7)
    qids_all = rng.permutation(job["query_id"].unique())
    folds = np.array_split(qids_all, 5)
    job["oof_severe"] = np.nan
    for fold_q in folds:
        te = job.query_id.isin(set(fold_q))
        m = GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        m.fit(job.loc[~te, feats], job.loc[~te, "is_severe"])
        job.loc[te, "oof_severe"] = m.predict_proba(job.loc[te, feats])[:, 1]

    auc = roc_auc_score(job["is_severe"], job["oof_severe"])
    order = np.argsort(job["oof_severe"].values)
    y_sorted = job["is_severe"].values[order]
    prefix = int(np.argmax(y_sorted == 1)) if y_sorted.any() else len(y_sorted)
    print(f"\nDIAGNOSTICS: severe-head AUC {auc:.3f} | clean lowest-risk "
          f"prefix {prefix} (needs >= 22)")
    print("[Step 7d reference with theta=1.2 label: AUC 0.875, prefix 2]")

    # ------------------------------------------------------------
    # E1: honest stationary evaluation
    # ------------------------------------------------------------
    print(f"\nE1 - STATIONARY ({args.seeds} random splits, 60/20/20)")
    qids = job["query_id"].unique()
    rows = []
    for seed in range(args.seeds):
        r_ = np.random.default_rng(seed)
        perm = r_.permutation(qids)
        n = len(perm)
        tr_q = set(perm[:int(0.6 * n)])
        ca_q = set(perm[int(0.6 * n):int(0.8 * n)])
        te_q = set(perm[int(0.8 * n):])
        train = job[job.query_id.isin(tr_q)]
        cal = job[job.query_id.isin(ca_q)].copy()
        test = job[job.query_id.isin(te_q)].copy()

        heads = fit_heads(train, feats)
        cal["p_severe"] = heads["is_severe"].predict_proba(cal[feats])[:, 1]
        t_star = s6.calibrate_threshold(cal["p_severe"].values,
                                        cal["is_severe"].values,
                                        args.alpha, args.delta)
        test["p_severe"] = heads["is_severe"].predict_proba(test[feats])[:, 1]
        test["p_win"] = heads["is_win"].predict_proba(test[feats])[:, 1]
        r = simulate(test, t_star, args.tau_win, args.theta_reg, args.theta_sev)
        r["t_star"] = t_star
        rows.append(r)

    E1 = pd.DataFrame(rows)
    print(f"{'metric':<30}{'mean':>10}{'std':>9}{'min':>9}{'max':>9}")
    print("-" * 67)
    for col, name in [("t_star", "calibrated t*"),
                      ("n_steer", "steered (of ~23)"),
                      ("severe_rate", "GUARANTEED severe rate"),
                      ("mild_rate", "mild rate (not guaranteed)"),
                      ("speedup_pct", "workload speedup %"),
                      ("capture_pct", "oracle capture %")]:
        print(f"{name:<30}{E1[col].mean():>10.3f}{E1[col].std():>9.3f}"
              f"{E1[col].min():>9.3f}{E1[col].max():>9.3f}")
    print(f"Splits with severe rate > alpha: "
          f"{int((E1['severe_rate'] > args.alpha).sum())}/{args.seeds}")

    # ------------------------------------------------------------
    # E2: family shift
    # ------------------------------------------------------------
    print("\nE2 - FAMILY SHIFT (leave-one-family-out, 33 folds)")
    agg = dict(steer=0, severe=0, mild=0)
    td, tp_, to_ = 0.0, 0.0, 0.0
    bad = []
    for fam in sorted(job["template"].unique(), key=int):
        rest = job[job.template != fam]
        test = job[job.template == fam].copy()
        rest_q = rest["query_id"].unique()
        r_ = np.random.default_rng(0)
        perm = r_.permutation(rest_q)
        ca_q = set(perm[:int(0.25 * len(perm))])
        train = rest[~rest.query_id.isin(ca_q)]
        cal = rest[rest.query_id.isin(ca_q)].copy()

        heads = fit_heads(train, feats)
        cal["p_severe"] = heads["is_severe"].predict_proba(cal[feats])[:, 1]
        t_star = s6.calibrate_threshold(cal["p_severe"].values,
                                        cal["is_severe"].values,
                                        args.alpha, args.delta)
        test["p_severe"] = heads["is_severe"].predict_proba(test[feats])[:, 1]
        test["p_win"] = heads["is_win"].predict_proba(test[feats])[:, 1]
        r = simulate(test, t_star, args.tau_win, args.theta_reg, args.theta_sev)
        agg["steer"] += r["n_steer"]
        agg["severe"] += r["n_severe"]
        agg["mild"] += r["n_mild"]
        # accumulate absolute times for aggregate speedup
        g = test.groupby("query_id").first()
        td_f = g["def_median_ms"].sum()
        td += td_f
        tp_ += td_f * (1 - r["speedup_pct"] / 100)
        to_ += test.groupby("query_id").apply(
            lambda x: min(x["def_median_ms"].iloc[0], x["median_ms"].min())).sum()
        if r["n_severe"] > 0:
            bad.append((fam, r["n_steer"], r["n_severe"]))

    sr = agg["severe"] / agg["steer"] if agg["steer"] else 0.0
    mr = agg["mild"] / agg["steer"] if agg["steer"] else 0.0
    print(f"AGGREGATE: {agg['steer']} steers | severe {agg['severe']} "
          f"(rate {sr:.2f} vs alpha {args.alpha}) | mild {agg['mild']} "
          f"(rate {mr:.2f}) | speedup {(td-tp_)/td*100:.1f}% | "
          f"capture {(td-tp_)/(td-to_)*100 if td > to_ else 0:.0f}%")
    if bad:
        print("Families with severe events: " +
              ", ".join(f"{f} ({sv}/{st})" for f, st, sv in bad))

    print("\nDone. Paste the full output back.")


if __name__ == "__main__":
    main()
