#!/usr/bin/env python3
"""
QPPE Rebuild - Step 7g: Policy-Level Conformal Calibration (Query Units)
=========================================================================
Step 7f's E1 violated its own guarantee (7/20 splits > alpha). Two
structural causes: candidate-level calibration breaks exchangeability
(candidates cluster within queries) and ignores selection bias (the
policy picks argmax p_win among certified, not a random certified
candidate).

Fix: calibrate the END-TO-END POLICY with QUERIES as the exchangeable
unit. For each threshold t (fixed ascending sequence, sequential
testing - no multiplicity correction needed):
    apply the full policy to each calibration query with OUT-OF-FOLD
    scores; the observed event is "steered AND materially severe";
    accept t while the Clopper-Pearson upper bound on
    P(severe | steered) stays <= alpha; t* = last accepted.

Cross-fitted calibration (4 folds over the train pool, ~79 queries)
provides enough query-level units; deployed heads are trained on the
full pool; evaluation uses untouched test queries.

Usage:
    py step7g_policy_calibration.py --user postgres --password 12345
Runtime: ~10-15 minutes (many model fits), no query execution.
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


def severe_label(cand, theta_sev, floor_ms):
    slow = cand["median_ms"] / (cand["def_median_ms"] + RATIO_EPS)
    added = cand["median_ms"] - cand["def_median_ms"]
    return (((slow > theta_sev) & (added > floor_ms)) |
            cand["censored"]).astype(int)


def new_model():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=8, random_state=42)


def oof_scores_pool(pool, feats, n_folds=4, seed=11):
    """Out-of-fold p_severe and p_win over the pool (folds by query)."""
    rng = np.random.default_rng(seed)
    qids = rng.permutation(pool["query_id"].unique())
    folds = np.array_split(qids, n_folds)
    ps = pd.Series(np.nan, index=pool.index)
    pw = pd.Series(np.nan, index=pool.index)
    for fold_q in folds:
        te = pool.query_id.isin(set(fold_q))
        ms = new_model()
        ms.fit(pool.loc[~te, feats], pool.loc[~te, "is_severe"])
        ps.loc[te] = ms.predict_proba(pool.loc[te, feats])[:, 1]
        mw = new_model()
        mw.fit(pool.loc[~te, feats], pool.loc[~te, "is_win"])
        pw.loc[te] = mw.predict_proba(pool.loc[te, feats])[:, 1]
    return ps, pw


def policy_outcomes(df, t, tau_win, theta_sev, floor_ms):
    """Per-query: (steered?, severe?) under threshold t."""
    outcomes = []
    for qid, g in df.groupby("query_id"):
        d_ms = g["def_median_ms"].iloc[0]
        cert = g[(g["p_severe"] < t) & (g["p_win"] > tau_win)]
        if cert.empty:
            outcomes.append((False, False, d_ms, d_ms,
                             min(d_ms, g["median_ms"].min())))
            continue
        top = cert.loc[cert["p_win"].idxmax()]
        c_ms = top["median_ms"]
        slow = c_ms / d_ms if d_ms > 0 else 1.0
        severe = (slow > theta_sev and (c_ms - d_ms) > floor_ms) or \
                 bool(top["censored"])
        outcomes.append((True, severe, d_ms, c_ms,
                         min(d_ms, g["median_ms"].min())))
    return outcomes


def calibrate_policy_threshold(cal_df, alpha, delta, tau_win,
                               theta_sev, floor_ms, cp_upper, n_grid=40):
    """Fixed ascending threshold sequence; last t whose CP bound passes."""
    grid = np.unique(np.quantile(cal_df["p_severe"].values,
                                 np.linspace(0.02, 0.98, n_grid)))
    t_star = 0.0
    for t in grid:
        out = policy_outcomes(cal_df, t, tau_win, theta_sev, floor_ms)
        n = sum(1 for s, _, *_ in out if s)
        k = sum(1 for s, sv, *_ in out if s and sv)
        if n == 0:
            continue
        if cp_upper(k, n, delta) <= alpha:
            t_star = float(t)
        else:
            break  # fixed-sequence testing: stop at first failure
    return t_star


def summarize(outcomes):
    n_st = sum(1 for s, *_ in outcomes if s)
    n_sv = sum(1 for s, sv, *_ in outcomes if s and sv)
    td = sum(o[2] for o in outcomes)
    tp_ = sum(o[3] for o in outcomes)
    to_ = sum(o[4] for o in outcomes)
    return dict(n_steer=n_st, n_severe=n_sv,
                severe_rate=n_sv / n_st if n_st else 0.0,
                speedup_pct=(td - tp_) / td * 100,
                capture_pct=(td - tp_) / (td - to_) * 100 if td > to_ else 0.0)


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
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    s6 = load_module("s6", "step6_live_loop.py")
    s7c = load_module("s7c", "step7c_job_analysis.py")

    print("QPPE Step 7g - Policy-Level Calibration, Query Units (JOB)")
    print("=" * 72)
    print(f"Certified event: steered query suffers slowdown > "
          f"{args.theta_sev}x AND added > {args.floor_ms:.0f} ms (or timeout)")
    print(f"alpha={args.alpha}, delta={args.delta}, tau_win={args.tau_win}")

    job = s7c.load_benchmark(args, s6, "imdb").reset_index(drop=True)
    job["slowdown"] = job["median_ms"] / (job["def_median_ms"] + RATIO_EPS)
    job["is_severe"] = severe_label(job, args.theta_sev, args.floor_ms)
    job["is_win"] = ((job["slowdown"] < args.theta_win) &
                     ~job["censored"]).astype(int)
    feats = [c for c in job.columns
             if c not in ("template", "query_id", "median_ms", "def_median_ms",
                          "censored", "slowdown", "is_severe", "is_win")]
    print(f"Candidates: {len(job)} | severe: {int(job.is_severe.sum())}")

    # ------------------------------------------------------------
    # E1: 70/30 train-pool/test, policy calibrated on OOF pool outcomes
    # ------------------------------------------------------------
    print(f"\nE1 - STATIONARY ({args.seeds} seeds, 70% pool / 30% test)")
    qids = job["query_id"].unique()
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(qids)
        n = len(perm)
        pool_q = set(perm[:int(0.7 * n)])
        test_q = set(perm[int(0.7 * n):])
        pool = job[job.query_id.isin(pool_q)].copy()
        test = job[job.query_id.isin(test_q)].copy()

        ps, pw = oof_scores_pool(pool, feats)
        pool["p_severe"] = ps
        pool["p_win"] = pw
        t_star = calibrate_policy_threshold(
            pool, args.alpha, args.delta, args.tau_win,
            args.theta_sev, args.floor_ms, s6.clopper_pearson_upper)

        ms = new_model()
        ms.fit(pool[feats], pool["is_severe"])
        mw = new_model()
        mw.fit(pool[feats], pool["is_win"])
        test["p_severe"] = ms.predict_proba(test[feats])[:, 1]
        test["p_win"] = mw.predict_proba(test[feats])[:, 1]
        out = policy_outcomes(test, t_star, args.tau_win,
                              args.theta_sev, args.floor_ms)
        r = summarize(out)
        r["t_star"] = t_star
        rows.append(r)

    E1 = pd.DataFrame(rows)
    print(f"{'metric':<30}{'mean':>10}{'std':>9}{'min':>9}{'max':>9}")
    print("-" * 67)
    for col, name in [("t_star", "calibrated t*"),
                      ("n_steer", "steered (of ~34)"),
                      ("severe_rate", "GUARANTEED severe rate"),
                      ("speedup_pct", "workload speedup %"),
                      ("capture_pct", "oracle capture %")]:
        print(f"{name:<30}{E1[col].mean():>10.3f}{E1[col].std():>9.3f}"
              f"{E1[col].min():>9.3f}{E1[col].max():>9.3f}")
    n_viol = int((E1["severe_rate"] > args.alpha).sum())
    print(f"Seeds with severe rate > alpha: {n_viol}/{args.seeds} "
          f"(expected ~{int(args.delta * args.seeds)} at delta="
          f"{args.delta}; Step 7f had 7/20)")

    # ------------------------------------------------------------
    # E2: family shift with the same construction
    # ------------------------------------------------------------
    print("\nE2 - FAMILY SHIFT (33 folds, policy-level calibration)")
    agg_st, agg_sv = 0, 0
    td, tp_, to_ = 0.0, 0.0, 0.0
    bad = []
    for fam in sorted(job["template"].unique(), key=int):
        pool = job[job.template != fam].copy()
        test = job[job.template == fam].copy()
        ps, pw = oof_scores_pool(pool, feats)
        pool["p_severe"] = ps
        pool["p_win"] = pw
        t_star = calibrate_policy_threshold(
            pool, args.alpha, args.delta, args.tau_win,
            args.theta_sev, args.floor_ms, s6.clopper_pearson_upper)
        ms = new_model()
        ms.fit(pool[feats], pool["is_severe"])
        mw = new_model()
        mw.fit(pool[feats], pool["is_win"])
        test["p_severe"] = ms.predict_proba(test[feats])[:, 1]
        test["p_win"] = mw.predict_proba(test[feats])[:, 1]
        out = policy_outcomes(test, t_star, args.tau_win,
                              args.theta_sev, args.floor_ms)
        r = summarize(out)
        agg_st += r["n_steer"]
        agg_sv += r["n_severe"]
        td += sum(o[2] for o in out)
        tp_ += sum(o[3] for o in out)
        to_ += sum(o[4] for o in out)
        if r["n_severe"] > 0:
            bad.append((fam, r["n_steer"], r["n_severe"]))
    rate = agg_sv / agg_st if agg_st else 0.0
    print(f"AGGREGATE: {agg_st} steers | severe {agg_sv} "
          f"(rate {rate:.2f} vs alpha {args.alpha}) | speedup "
          f"{(td-tp_)/td*100:.1f}% | capture "
          f"{(td-tp_)/(td-to_)*100 if td > to_ else 0:.0f}%")
    if bad:
        print("Families with severe events: " +
              ", ".join(f"{f} ({sv}/{st})" for f, st, sv in bad))

    print("\nDone. Paste the full output back.")


if __name__ == "__main__":
    main()
