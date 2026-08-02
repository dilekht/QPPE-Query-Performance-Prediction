#!/usr/bin/env python3
"""
QPPE Rebuild - Step 7c: JOB Headline Analysis (+ Cross-Benchmark Transfer)
===========================================================================
Runs the full evaluation suite on the JOB corpus:

  E1  STATIONARY: 20 random query-level train/cal/test splits with the
      cross-conformal-style gate (train 60% / calibrate 20% / test 20%).
  E2  FAMILY SHIFT: leave-one-family-out (33 folds), calibration from
      the remaining families.
  E3  CROSS-BENCHMARK TRANSFER: heads trained + calibrated ONLY on the
      TPC-H corpus, tested cold on all of JOB. (Idea #3 from our
      original brainstorm, measured for free.)

Requires step6_live_loop.py and step3c_expand_corpus.py in the folder.

Usage:
    py step7c_job_analysis.py --user postgres --password 12345
Runtime: ~5-10 minutes (about 110 model fits, no query execution).
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


def load_benchmark(args, s6, benchmark, variant_max=None):
    """Candidate dataset for one benchmark, features as in steps 4e/5/6."""
    import psycopg2
    conn = psycopg2.connect(dbname=args.log_db, user=args.user,
                            password=args.password, host=args.host,
                            port=args.port)
    variant_clause = f"AND q.variant < {variant_max}" if variant_max else ""
    df = pd.read_sql(f"""
        WITH timings AS (
            SELECT e.plan_id,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY e.exec_ms) AS median_ms,
                   bool_or(e.timed_out) AS censored
            FROM executions e WHERE NOT e.is_warmup GROUP BY e.plan_id
        )
        SELECT q.template, q.variant, q.query_id, h.name AS hint_set,
               p.plan_id, p.plan_json, t.median_ms, t.censored
        FROM timings t
        JOIN plans p ON p.plan_id = t.plan_id
        JOIN hint_sets h ON h.hint_set_id = p.hint_set_id
        JOIN queries q ON q.query_id = p.query_id
        WHERE q.benchmark = '{benchmark}' {variant_clause}
        ORDER BY q.query_id, p.plan_id;
    """, conn)
    conn.close()

    flatd = [s6.flat_features(pj[0]["Plan"]) for pj in df["plan_json"]]
    noded = [s6.node_features(pj[0]["Plan"]) for pj in df["plan_json"]]
    by_qid = {}
    for i in range(len(df)):
        by_qid.setdefault(df.loc[i, "query_id"], {})[df.loc[i, "hint_set"]] = i

    rows = []
    for qid, hmap in by_qid.items():
        if "default" not in hmap:
            continue
        di = hmap["default"]
        for hs, i in hmap.items():
            if hs == "default":
                continue
            r = s6.build_feature_row(flatd[i], noded[i],
                                     flatd[di], noded[di], hs)
            r["template"] = df.loc[i, "template"]
            r["query_id"] = qid
            r["median_ms"] = df.loc[i, "median_ms"]
            r["def_median_ms"] = df.loc[di, "median_ms"]
            r["censored"] = bool(df.loc[i, "censored"])
            rows.append(r)
    return pd.DataFrame(rows)


def label(cand, theta_reg, theta_win):
    cand = cand.copy()
    slow = cand["median_ms"] / (cand["def_median_ms"] + RATIO_EPS)
    cand["is_regression"] = ((slow > theta_reg) | cand["censored"]).astype(int)
    cand["is_win"] = ((slow < theta_win) & ~cand["censored"]).astype(int)
    return cand


def fit_heads(train, feats):
    from sklearn.ensemble import GradientBoostingClassifier
    heads = {}
    for lab in ["is_regression", "is_win"]:
        m = GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        m.fit(train[feats], train[lab])
        heads[lab] = m
    return heads


def gate_and_policy(s6, heads, feats, cal, test, alpha, delta, theta_reg):
    cal = cal.copy()
    test = test.copy()
    cal_p = heads["is_regression"].predict_proba(cal[feats])[:, 1]
    t_star = s6.calibrate_threshold(cal_p, cal["is_regression"].values,
                                    alpha, delta)
    test["p_risk"] = heads["is_regression"].predict_proba(test[feats])[:, 1]
    test["p_win"] = heads["is_win"].predict_proba(test[feats])[:, 1]

    rows = []
    for qid, g in test.groupby("query_id"):
        d_ms = g["def_median_ms"].iloc[0]
        best = min(d_ms, g["median_ms"].min())
        cert = g[g["p_risk"] < t_star]
        steered, chosen = False, d_ms
        if not cert.empty:
            top = cert.loc[cert["p_win"].idxmax()]
            steered, chosen = True, top["median_ms"]
        rows.append(dict(default_ms=d_ms, oracle_ms=best, policy_ms=chosen,
                         steered=steered,
                         regressed=steered and chosen > theta_reg * d_ms,
                         template=g["template"].iloc[0]))
    P = pd.DataFrame(rows)
    tot_def, tot_pol, tot_or = P.default_ms.sum(), P.policy_ms.sum(), P.oracle_ms.sum()
    gain = tot_def - tot_or
    n_st, n_rg = int(P.steered.sum()), int(P.regressed.sum())
    return dict(t_star=t_star, n_steer=n_st, n_reg=n_rg,
                reg_rate=n_rg / n_st if n_st else 0.0,
                speedup_pct=(tot_def - tot_pol) / tot_def * 100,
                capture_pct=(tot_def - tot_pol) / gain * 100 if gain > 0 else 0.0,
                P=P)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--log-db", default="qppe")
    parser.add_argument("--theta-reg", type=float, default=1.2)
    parser.add_argument("--theta-win", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    s6 = load_module("s6", "step6_live_loop.py")

    print("QPPE Step 7c - JOB Analysis")
    print("=" * 72)

    job = label(load_benchmark(args, s6, "imdb"), args.theta_reg, args.theta_win)
    feats = [c for c in job.columns
             if c not in ("template", "query_id", "median_ms", "def_median_ms",
                          "censored", "is_regression", "is_win")]
    n_cens = int(job["censored"].sum())
    print(f"JOB candidates: {len(job)} ({n_cens} censored) | families: "
          f"{job['template'].nunique()} | queries: {job['query_id'].nunique()}")
    print(f"Regressions: {int(job.is_regression.sum())} | "
          f"wins: {int(job.is_win.sum())}")

    tot_def, tot_or = 0.0, 0.0
    for qid, g in job.groupby("query_id"):
        d = g["def_median_ms"].iloc[0]
        tot_def += d
        tot_or += min(d, g["median_ms"].min())
    print(f"Workload bounds: never-steer {tot_def/1000:.1f} s | oracle "
          f"{tot_or/1000:.1f} s (possible improvement "
          f"{(tot_def-tot_or)/tot_def:.0%})")

    # ------------------------------------------------------------
    # E1: stationary
    # ------------------------------------------------------------
    print(f"\nE1 - STATIONARY ({args.seeds} random splits, 60/20/20)")
    qids = job["query_id"].unique()
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(qids)
        n = len(perm)
        tr_q = set(perm[:int(0.6 * n)])
        ca_q = set(perm[int(0.6 * n):int(0.8 * n)])
        te_q = set(perm[int(0.8 * n):])
        heads = fit_heads(job[job.query_id.isin(tr_q)], feats)
        r = gate_and_policy(s6, heads, feats,
                            job[job.query_id.isin(ca_q)],
                            job[job.query_id.isin(te_q)],
                            args.alpha, args.delta, args.theta_reg)
        rows.append({k: v for k, v in r.items() if k != "P"})
    E1 = pd.DataFrame(rows)
    print(f"{'metric':<26}{'mean':>10}{'std':>9}{'min':>9}{'max':>9}")
    print("-" * 63)
    for col, name in [("t_star", "calibrated t*"),
                      ("n_steer", "steered (of ~23)"),
                      ("reg_rate", "realized reg rate"),
                      ("speedup_pct", "workload speedup %"),
                      ("capture_pct", "oracle capture %")]:
        print(f"{name:<26}{E1[col].mean():>10.3f}{E1[col].std():>9.3f}"
              f"{E1[col].min():>9.3f}{E1[col].max():>9.3f}")
    print(f"Splits exceeding alpha: "
          f"{int((E1['reg_rate'] > args.alpha).sum())}/{args.seeds}")

    # ------------------------------------------------------------
    # E2: family shift (33 folds)
    # ------------------------------------------------------------
    print("\nE2 - FAMILY SHIFT (leave-one-family-out)")
    agg_st, agg_rg = 0, 0
    td, tp, to_ = 0.0, 0.0, 0.0
    worst = []
    for fam in sorted(job["template"].unique(), key=int):
        rest = job[job.template != fam]
        test = job[job.template == fam]
        rest_q = rest["query_id"].unique()
        rng = np.random.default_rng(0)
        perm = rng.permutation(rest_q)
        ca_q = set(perm[:int(0.25 * len(perm))])
        heads = fit_heads(rest[~rest.query_id.isin(ca_q)], feats)
        r = gate_and_policy(s6, heads, feats, rest[rest.query_id.isin(ca_q)],
                            test, args.alpha, args.delta, args.theta_reg)
        agg_st += r["n_steer"]
        agg_rg += r["n_reg"]
        td += r["P"].default_ms.sum()
        tp += r["P"].policy_ms.sum()
        to_ += r["P"].oracle_ms.sum()
        if r["n_reg"] > 0:
            worst.append((fam, r["n_steer"], r["n_reg"]))
    rate = agg_rg / agg_st if agg_st else 0.0
    print(f"AGGREGATE: {agg_st} steers, {agg_rg} regressions "
          f"(rate {rate:.2f} vs alpha {args.alpha}) | speedup "
          f"{(td-tp)/td*100:.1f}% | capture "
          f"{(td-tp)/(td-to_)*100 if td > to_ else 0:.0f}%")
    if worst:
        print("Families with regressions: " +
              ", ".join(f"{f} ({r}/{s})" for f, s, r in worst))

    # ------------------------------------------------------------
    # E3: cross-benchmark transfer TPC-H -> JOB
    # ------------------------------------------------------------
    print("\nE3 - CROSS-BENCHMARK TRANSFER (train+calibrate on TPC-H, "
          "test cold on all JOB)")
    tpch = label(load_benchmark(args, s6, "tpch_sf1", variant_max=100),
                 args.theta_reg, args.theta_win)
    # align features (same construction => same columns)
    tpch_q = tpch["query_id"].unique()
    rng = np.random.default_rng(0)
    perm = rng.permutation(tpch_q)
    ca_q = set(perm[:int(0.25 * len(perm))])
    heads = fit_heads(tpch[~tpch.query_id.isin(ca_q)], feats)
    r = gate_and_policy(s6, heads, feats, tpch[tpch.query_id.isin(ca_q)],
                        job, args.alpha, args.delta, args.theta_reg)
    print(f"t* = {r['t_star']:.3f} | steered {r['n_steer']}/113 JOB queries | "
          f"regressions {r['n_reg']} (rate {r['reg_rate']:.2f}) | "
          f"speedup {r['speedup_pct']:.1f}% | capture {r['capture_pct']:.0f}%")

    print("\nDone. Paste the full output back.")


if __name__ == "__main__":
    main()
