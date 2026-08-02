#!/usr/bin/env python3
"""
QPPE Rebuild - Step 7f: Materially-Severe Gate (Relative x Absolute)
=====================================================================
Step 7e's prefix blockers are hypothesized to be tiny-absolute-time
queries where a 2x slowdown means milliseconds of damage. This step:

  1. EVIDENCE: prints the lowest-scored "severe" candidates under the
     7e label, with absolute times - the blockers, named.
  2. TRANSPARENCY: a label-sensitivity grid (theta_sev x damage floor)
     showing AUC and clean prefix for each definition.
  3. PRE-DECLARED EVALUATION: honest E1 (20 splits) and E2 (33 family
     folds) at the default definition only:
         severe := slowdown > 2x AND added_ms > 1000, OR censored.

Usage:
    py step7f_material_severity.py --user postgres --password 12345
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


def severe_label(cand, theta_sev, floor_ms):
    slow = cand["median_ms"] / (cand["def_median_ms"] + RATIO_EPS)
    added = cand["median_ms"] - cand["def_median_ms"]
    return (((slow > theta_sev) & (added > floor_ms)) |
            cand["censored"]).astype(int)


def oof(cand, feats, y, n_folds=5, seed=7):
    from sklearn.ensemble import GradientBoostingClassifier
    rng = np.random.default_rng(seed)
    qids = rng.permutation(cand["query_id"].unique())
    folds = np.array_split(qids, n_folds)
    out = pd.Series(np.nan, index=cand.index)
    for fold_q in folds:
        te = cand.query_id.isin(set(fold_q))
        m = GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        m.fit(cand.loc[~te, feats], y[~te])
        out.loc[te] = m.predict_proba(cand.loc[te, feats])[:, 1]
    return out


def clean_prefix(scores, y):
    order = np.argsort(scores)
    ys = np.asarray(y)[order]
    return int(np.argmax(ys == 1)) if ys.any() else len(ys)


def fit_heads(train, feats, sev_col):
    from sklearn.ensemble import GradientBoostingClassifier
    heads = {}
    for lab in [sev_col, "is_win"]:
        m = GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        m.fit(train[feats], train[lab])
        heads[lab] = m
    return heads


def simulate(test, t_star, tau_win, theta_reg, theta_sev, floor_ms):
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
        added = chosen - d_ms
        rows.append(dict(default_ms=d_ms, oracle_ms=best, policy_ms=chosen,
                         steered=steered,
                         severe=steered and slow > theta_sev and added > floor_ms,
                         mild=steered and slow > theta_reg and not
                              (slow > theta_sev and added > floor_ms)))
    P = pd.DataFrame(rows)
    td, tp_, to_ = P.default_ms.sum(), P.policy_ms.sum(), P.oracle_ms.sum()
    n_st = int(P.steered.sum())
    return dict(n_steer=n_st, n_severe=int(P.severe.sum()),
                n_mild=int(P.mild.sum()),
                severe_rate=int(P.severe.sum()) / n_st if n_st else 0.0,
                mild_rate=int(P.mild.sum()) / n_st if n_st else 0.0,
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
    parser.add_argument("--floor-ms", type=float, default=1000.0)
    parser.add_argument("--theta-win", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--tau-win", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    from sklearn.metrics import roc_auc_score

    s6 = load_module("s6", "step6_live_loop.py")
    s7c = load_module("s7c", "step7c_job_analysis.py")

    print("QPPE Step 7f - Materially-Severe Gate (JOB)")
    print("=" * 72)

    job = s7c.load_benchmark(args, s6, "imdb").reset_index(drop=True)
    job["slowdown"] = job["median_ms"] / (job["def_median_ms"] + RATIO_EPS)
    job["is_win"] = ((job["slowdown"] < args.theta_win) &
                     ~job["censored"]).astype(int)
    feats = [c for c in job.columns
             if c not in ("template", "query_id", "median_ms", "def_median_ms",
                          "censored", "slowdown", "is_win")]

    # ------------------------------------------------------------
    # 1. evidence: the 7e prefix blockers, named
    # ------------------------------------------------------------
    y_old = severe_label(job, args.theta_sev, 0.0)   # 7e label (no floor)
    p_old = oof(job, feats, y_old)
    print("\nEVIDENCE - lowest-scored 'severe' candidates under the "
          "7e label (no damage floor):")
    blockers = job[y_old == 1].copy()
    blockers["p"] = p_old[y_old == 1]
    blockers = blockers.nsmallest(10, "p")
    print(f"{'family':<8}{'p_severe':>9}{'default ms':>12}{'cand ms':>10}"
          f"{'slowdown':>10}{'added ms':>10}")
    print("-" * 59)
    for _, r in blockers.iterrows():
        print(f"{r['template']:<8}{r['p']:>9.3f}{r['def_median_ms']:>12.1f}"
              f"{r['median_ms']:>10.1f}{r['slowdown']:>10.1f}"
              f"{r['median_ms'] - r['def_median_ms']:>10.1f}")

    # ------------------------------------------------------------
    # 2. transparency: label-sensitivity grid
    # ------------------------------------------------------------
    print("\nLABEL-SENSITIVITY GRID (out-of-fold AUC / clean prefix)")
    print(f"{'theta_sev':>10}{'floor ms':>10}{'severe n':>10}{'AUC':>7}"
          f"{'prefix':>8}")
    print("-" * 45)
    for ts in [2.0, 3.0]:
        for fl in [0.0, 500.0, 1000.0]:
            y = severe_label(job, ts, fl)
            if y.sum() < 20:
                continue
            p = oof(job, feats, y)
            auc = roc_auc_score(y, p)
            pref = clean_prefix(p.values, y.values)
            mark = "  <- pre-declared" if (ts == args.theta_sev and
                                           fl == args.floor_ms) else ""
            print(f"{ts:>10.1f}{fl:>10.0f}{int(y.sum()):>10}{auc:>7.3f}"
                  f"{pref:>8}{mark}")

    # ------------------------------------------------------------
    # 3. honest E1 + E2 at the pre-declared definition
    # ------------------------------------------------------------
    job["is_severe"] = severe_label(job, args.theta_sev, args.floor_ms)
    print(f"\nPre-declared definition: slowdown > {args.theta_sev}x AND "
          f"added > {args.floor_ms:.0f} ms, OR censored "
          f"({int(job.is_severe.sum())} severe candidates)")

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
        heads = fit_heads(train, feats, "is_severe")
        cal["p_severe"] = heads["is_severe"].predict_proba(cal[feats])[:, 1]
        t_star = s6.calibrate_threshold(cal["p_severe"].values,
                                        cal["is_severe"].values,
                                        args.alpha, args.delta)
        test["p_severe"] = heads["is_severe"].predict_proba(test[feats])[:, 1]
        test["p_win"] = heads["is_win"].predict_proba(test[feats])[:, 1]
        r = simulate(test, t_star, args.tau_win, args.theta_reg,
                     args.theta_sev, args.floor_ms)
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

    print("\nE2 - FAMILY SHIFT (33 folds)")
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
        heads = fit_heads(train, feats, "is_severe")
        cal["p_severe"] = heads["is_severe"].predict_proba(cal[feats])[:, 1]
        t_star = s6.calibrate_threshold(cal["p_severe"].values,
                                        cal["is_severe"].values,
                                        args.alpha, args.delta)
        test["p_severe"] = heads["is_severe"].predict_proba(test[feats])[:, 1]
        test["p_win"] = heads["is_win"].predict_proba(test[feats])[:, 1]
        r = simulate(test, t_star, args.tau_win, args.theta_reg,
                     args.theta_sev, args.floor_ms)
        agg["steer"] += r["n_steer"]
        agg["severe"] += r["n_severe"]
        agg["mild"] += r["n_mild"]
        g = test.groupby("query_id").first()
        td_f = g["def_median_ms"].sum()
        td += td_f
        tp_ += td_f * (1 - r["speedup_pct"] / 100)
        to_ += test.groupby("query_id").apply(
            lambda x: min(x["def_median_ms"].iloc[0],
                          x["median_ms"].min())).sum()
        if r["n_severe"] > 0:
            bad.append((fam, r["n_steer"], r["n_severe"]))
    sr = agg["severe"] / agg["steer"] if agg["steer"] else 0.0
    print(f"AGGREGATE: {agg['steer']} steers | severe {agg['severe']} "
          f"(rate {sr:.2f} vs alpha {args.alpha}) | mild {agg['mild']} | "
          f"speedup {(td-tp_)/td*100:.1f}% | "
          f"capture {(td-tp_)/(td-to_)*100 if td > to_ else 0:.0f}%")
    if bad:
        print("Families with severe events: " +
              ", ".join(f"{f} ({sv}/{st})" for f, st, sv in bad))

    print("\nDone. Paste the full output back.")


if __name__ == "__main__":
    main()
