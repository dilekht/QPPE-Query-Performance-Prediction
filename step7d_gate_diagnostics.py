#!/usr/bin/env python3
"""
QPPE Rebuild - Step 7d: Why the Gate Sealed on JOB (Diagnostics + Remedies)
============================================================================
Step 7c: t* = 0 on all JOB-internal calibrations. Base rate is 70%
regressions; the CP bound needs a clean lowest-risk prefix of >= 22.
This step separates "bad ranking" from "harsh bound", then tests two
remedies:

  1. DIAGNOSTICS: 5-fold out-of-fold risk scores over all 850 JOB
     candidates -> AUC, clean-prefix length, per-class score quantiles,
     per-hint-set regression base rates.
  2. ALPHA/DELTA SWEEP: where does the pooled gate first open?
  3. MONDRIAN CALIBRATION: per-hint-set thresholds t*_g - benign hint
     sets can certify even when the pooled bound cannot.
  4. HONEST RE-VALIDATION: the best viable configuration is re-run with
     the E1 split protocol (20 seeds, 60/20/20) - those are the numbers
     that go in the paper.

Requires step6_live_loop.py, step3c_expand_corpus.py, and
step7c_job_analysis.py in the same folder.

Usage:
    py step7d_gate_diagnostics.py --user postgres --password 12345
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


def oof_scores(cand, feats, label_col, n_folds=5, seed=7):
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
        m.fit(cand.loc[~te, feats], cand.loc[~te, label_col])
        out.loc[te] = m.predict_proba(cand.loc[te, feats])[:, 1]
    return out


def mondrian_thresholds(cal, alpha, delta, cp_upper):
    """Per-hint-set CP thresholds. cal needs hint_set, p_risk, is_regression."""
    ts = {}
    for hs, g in cal.groupby("hint_set"):
        p = g["p_risk"].values
        y = g["is_regression"].values
        order = np.argsort(p)
        ps, ys = p[order], y[order]
        cum = np.cumsum(ys)
        best = 0.0
        for i in range(len(ps)):
            if cp_upper(int(cum[i]), i + 1, delta) <= alpha:
                best = float(ps[i]) + 1e-12
        ts[hs] = best
    return ts


def simulate(test, theta_reg, t_global=None, t_by_hs=None, tau_win=None):
    rows = []
    for qid, g in test.groupby("query_id"):
        d_ms = g["def_median_ms"].iloc[0]
        best = min(d_ms, g["median_ms"].min())
        if t_by_hs is not None:
            cert = g[g.apply(lambda r: r["p_risk"] < t_by_hs.get(r["hint_set"], 0.0),
                             axis=1)]
        else:
            cert = g[g["p_risk"] < t_global]
        if tau_win is not None and not cert.empty:
            cert = cert[cert["p_win"] > tau_win]
        steered, chosen = False, d_ms
        if not cert.empty:
            top = cert.loc[cert["p_win"].idxmax()]
            steered, chosen = True, top["median_ms"]
        rows.append(dict(default_ms=d_ms, oracle_ms=best, policy_ms=chosen,
                         steered=steered,
                         regressed=steered and chosen > theta_reg * d_ms))
    P = pd.DataFrame(rows)
    td, tp_, to_ = P.default_ms.sum(), P.policy_ms.sum(), P.oracle_ms.sum()
    n_st, n_rg = int(P.steered.sum()), int(P.regressed.sum())
    return dict(n_steer=n_st, n_reg=n_rg,
                reg_rate=n_rg / n_st if n_st else 0.0,
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
    parser.add_argument("--theta-win", type=float, default=0.9)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier

    s6 = load_module("s6", "step6_live_loop.py")
    s7c = load_module("s7c", "step7c_job_analysis.py")

    print("QPPE Step 7d - Gate Diagnostics & Remedies (JOB)")
    print("=" * 72)

    job = s7c.label(s7c.load_benchmark(args, s6, "imdb"),
                    args.theta_reg, args.theta_win)
    job = job.reset_index(drop=True)
    # recover hint_set from the one-hot columns for Mondrian grouping
    hs_cols = [c for c in job.columns if c.startswith("hs_")]
    job["hint_set"] = job[hs_cols].idxmax(axis=1).str[3:]
    feats = [c for c in job.columns
             if c not in ("template", "query_id", "median_ms", "def_median_ms",
                          "censored", "is_regression", "is_win", "hint_set")]

    # ------------------------------------------------------------
    # 1. diagnostics
    # ------------------------------------------------------------
    job["p_risk"] = oof_scores(job, feats, "is_regression")
    job["p_win"] = oof_scores(job, feats, "is_win")

    auc = roc_auc_score(job["is_regression"], job["p_risk"])
    order = np.argsort(job["p_risk"].values)
    y_sorted = job["is_regression"].values[order]
    prefix = int(np.argmax(y_sorted == 1)) if y_sorted.any() else len(y_sorted)
    q_safe = job.loc[job.is_regression == 0, "p_risk"].quantile([.1, .5, .9])
    q_reg = job.loc[job.is_regression == 1, "p_risk"].quantile([.1, .5, .9])
    print(f"\nDIAGNOSTICS (out-of-fold, {len(job)} candidates)")
    print(f"  risk-head AUC: {auc:.3f}")
    print(f"  clean lowest-risk prefix: {prefix} (pooled CP at alpha=0.1, "
          f"delta=0.1 needs >= 22)")
    print(f"  p_risk quantiles  safe: 10%={q_safe.iloc[0]:.3f} "
          f"50%={q_safe.iloc[1]:.3f} 90%={q_safe.iloc[2]:.3f}")
    print(f"                    regr: 10%={q_reg.iloc[0]:.3f} "
          f"50%={q_reg.iloc[1]:.3f} 90%={q_reg.iloc[2]:.3f}")

    print(f"\n  per-hint-set regression base rates:")
    print(f"  {'hint set':<22}{'n':>5}{'regr':>6}{'rate':>7}")
    for hs, g in job.groupby("hint_set"):
        print(f"  {hs:<22}{len(g):>5}{int(g.is_regression.sum()):>6}"
              f"{g.is_regression.mean():>7.2f}")

    # ------------------------------------------------------------
    # 2+3. alpha/delta sweep, pooled vs Mondrian (exploratory, OOF)
    # ------------------------------------------------------------
    print("\nEXPLORATORY SWEEP (calibrate & evaluate on OOF scores - "
          "optimistic; final numbers come from E1 below)")
    print(f"{'scheme':<10}{'alpha':>6}{'delta':>6}{'steers':>8}{'regs':>6}"
          f"{'rate':>7}{'speedup%':>10}{'capture%':>10}")
    print("-" * 63)
    best_cfg = None
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.30]:
        for delta in [0.10, 0.20]:
            t_g = s6.calibrate_threshold(job["p_risk"].values,
                                         job["is_regression"].values,
                                         alpha, delta)
            r = simulate(job, args.theta_reg, t_global=t_g)
            print(f"{'pooled':<10}{alpha:>6.2f}{delta:>6.2f}{r['n_steer']:>8}"
                  f"{r['n_reg']:>6}{r['reg_rate']:>7.2f}"
                  f"{r['speedup_pct']:>10.1f}{r['capture_pct']:>10.0f}")
            ts = mondrian_thresholds(job, alpha, delta, s6.clopper_pearson_upper)
            r = simulate(job, args.theta_reg, t_by_hs=ts)
            print(f"{'mondrian':<10}{alpha:>6.2f}{delta:>6.2f}{r['n_steer']:>8}"
                  f"{r['n_reg']:>6}{r['reg_rate']:>7.2f}"
                  f"{r['speedup_pct']:>10.1f}{r['capture_pct']:>10.0f}")
            if r["reg_rate"] <= alpha and (
                    best_cfg is None or r["capture_pct"] > best_cfg[2]["capture_pct"]):
                best_cfg = ("mondrian", (alpha, delta), r)

    if best_cfg is None:
        print("\nNo configuration achieved its target rate even in the "
              "exploratory setting. The honest conclusion is that the "
              "current risk head cannot support a certified gate on JOB.")
        return

    scheme, (alpha, delta), _ = best_cfg
    print(f"\nBest exploratory config: {scheme}, alpha={alpha}, delta={delta}")

    # ------------------------------------------------------------
    # 4. honest E1 re-validation at that config
    # ------------------------------------------------------------
    print(f"\nHONEST E1 RE-VALIDATION ({args.seeds} splits, 60/20/20, "
          f"{scheme}, alpha={alpha}, delta={delta})")
    qids = job["query_id"].unique()
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(qids)
        n = len(perm)
        tr_q = set(perm[:int(0.6 * n)])
        ca_q = set(perm[int(0.6 * n):int(0.8 * n)])
        te_q = set(perm[int(0.8 * n):])
        train = job[job.query_id.isin(tr_q)]
        cal = job[job.query_id.isin(ca_q)].copy()
        test = job[job.query_id.isin(te_q)].copy()

        heads = {}
        for lab in ["is_regression", "is_win"]:
            m = GradientBoostingClassifier(
                n_estimators=250, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=8, random_state=42)
            m.fit(train[feats], train[lab])
            heads[lab] = m
        for part in (cal, test):
            part["p_risk"] = heads["is_regression"].predict_proba(part[feats])[:, 1]
            part["p_win"] = heads["is_win"].predict_proba(part[feats])[:, 1]

        ts = mondrian_thresholds(cal, alpha, delta, s6.clopper_pearson_upper)
        r = simulate(test, args.theta_reg, t_by_hs=ts)
        rows.append(r)

    E1 = pd.DataFrame(rows)
    print(f"{'metric':<26}{'mean':>10}{'std':>9}{'min':>9}{'max':>9}")
    print("-" * 63)
    for col, name in [("n_steer", "steered (of ~23)"),
                      ("reg_rate", "realized reg rate"),
                      ("speedup_pct", "workload speedup %"),
                      ("capture_pct", "oracle capture %")]:
        print(f"{name:<26}{E1[col].mean():>10.3f}{E1[col].std():>9.3f}"
              f"{E1[col].min():>9.3f}{E1[col].max():>9.3f}")
    print(f"Splits exceeding alpha={alpha}: "
          f"{int((E1['reg_rate'] > alpha).sum())}/{args.seeds}")
    print("\nDone. Paste the full output back.")


if __name__ == "__main__":
    main()
