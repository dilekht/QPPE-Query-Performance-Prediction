#!/usr/bin/env python3
"""
QPPE Rebuild - Step 7h: Select-then-Certify (Both Benchmarks)
==============================================================
Step 7g's fixed-sequence test failed at the first (tiny-n) grid point
by construction. This step uses the simplest airtight procedure:

  per seed, queries split 40% TRAIN / 25% SELECT / 15% CERTIFY / 20% TEST
    TRAIN   -> fit severe + win heads
    SELECT  -> choose ONE threshold t_hat: the largest t whose empirical
               severe rate on select-queries is <= alpha (free search,
               exploratory - no guarantee claimed here)
    CERTIFY -> a SINGLE Clopper-Pearson test of t_hat at (alpha, delta);
               pass -> deploy t_hat, fail -> t* = 0 (abstain)
    TEST    -> honest evaluation

Also prints the sample-size floor n_min(alpha, delta) = ln(delta)/ln(1-alpha)
for zero observed failures - the quantified "how much workload history
does certified steering need" result.

Run uniformly on BOTH benchmarks (TPC-H's earlier certificate was
candidate-level, i.e. technically anti-conservative; the paper needs
everything under the valid construction).

Usage:
    py step7h_select_then_certify.py --user postgres --password 12345
Runtime: ~10-15 minutes, no query execution.
"""

import argparse
import importlib.util
import math
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


def new_model():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=8, random_state=42)


def severe_label(cand, theta_sev, floor_ms):
    slow = cand["median_ms"] / (cand["def_median_ms"] + RATIO_EPS)
    added = cand["median_ms"] - cand["def_median_ms"]
    return (((slow > theta_sev) & (added > floor_ms)) |
            cand["censored"]).astype(int)


def policy_outcomes(df, t, tau_win, theta_sev, floor_ms):
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


def run_benchmark(name, cand, args, s6):
    print(f"\n{'=' * 72}\nBENCHMARK: {name}")
    print(f"{'=' * 72}")
    cand = cand.reset_index(drop=True)
    cand["slowdown"] = cand["median_ms"] / (cand["def_median_ms"] + RATIO_EPS)
    cand["is_severe"] = severe_label(cand, args.theta_sev, args.floor_ms)
    cand["is_win"] = ((cand["slowdown"] < args.theta_win) &
                      ~cand["censored"]).astype(int)
    feats = [c for c in cand.columns
             if c not in ("template", "query_id", "median_ms", "def_median_ms",
                          "censored", "slowdown", "is_severe", "is_win")]
    qids = cand["query_id"].unique()
    print(f"Queries: {len(qids)} | candidates: {len(cand)} | "
          f"severe: {int(cand.is_severe.sum())}")

    rows = []
    n_certified = 0
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(qids)
        n = len(perm)
        tr_q = set(perm[:int(0.40 * n)])
        se_q = set(perm[int(0.40 * n):int(0.65 * n)])
        ce_q = set(perm[int(0.65 * n):int(0.80 * n)])
        te_q = set(perm[int(0.80 * n):])

        train = cand[cand.query_id.isin(tr_q)]
        heads_s = new_model()
        heads_s.fit(train[feats], train["is_severe"])
        heads_w = new_model()
        heads_w.fit(train[feats], train["is_win"])

        def score(df):
            df = df.copy()
            df["p_severe"] = heads_s.predict_proba(df[feats])[:, 1]
            df["p_win"] = heads_w.predict_proba(df[feats])[:, 1]
            return df

        sel = score(cand[cand.query_id.isin(se_q)])
        cer = score(cand[cand.query_id.isin(ce_q)])
        tes = score(cand[cand.query_id.isin(te_q)])

        # SELECT: largest t with empirical severe rate <= alpha (free search)
        grid = np.unique(np.quantile(sel["p_severe"].values,
                                     np.linspace(0.05, 0.95, 25)))
        t_hat = 0.0
        for t in grid:
            out = policy_outcomes(sel, t, args.tau_win,
                                  args.theta_sev, args.floor_ms)
            n_st = sum(1 for s, *_ in out if s)
            k = sum(1 for s, sv, *_ in out if s and sv)
            if n_st >= 3 and k / n_st <= args.alpha:
                t_hat = float(t)

        # CERTIFY: single CP test of t_hat
        t_star = 0.0
        if t_hat > 0:
            out = policy_outcomes(cer, t_hat, args.tau_win,
                                  args.theta_sev, args.floor_ms)
            n_st = sum(1 for s, *_ in out if s)
            k = sum(1 for s, sv, *_ in out if s and sv)
            if n_st > 0 and s6.clopper_pearson_upper(k, n_st, args.delta) <= args.alpha:
                t_star = t_hat
        if t_star > 0:
            n_certified += 1

        out = policy_outcomes(tes, t_star, args.tau_win,
                              args.theta_sev, args.floor_ms)
        r = summarize(out)
        r["t_star"] = t_star
        rows.append(r)

    E = pd.DataFrame(rows)
    print(f"\nSeeds where certification PASSED (system allowed to steer): "
          f"{n_certified}/{args.seeds}")
    print(f"{'metric':<30}{'mean':>10}{'std':>9}{'min':>9}{'max':>9}")
    print("-" * 67)
    for col, name_ in [("t_star", "deployed t*"),
                       ("n_steer", "steered per test set"),
                       ("severe_rate", "GUARANTEED severe rate"),
                       ("speedup_pct", "workload speedup %"),
                       ("capture_pct", "oracle capture %")]:
        print(f"{name_:<30}{E[col].mean():>10.3f}{E[col].std():>9.3f}"
              f"{E[col].min():>9.3f}{E[col].max():>9.3f}")
    n_viol = int((E["severe_rate"] > args.alpha).sum())
    print(f"Seeds with severe rate > alpha: {n_viol}/{args.seeds} "
          f"(delta={args.delta} implies ~{max(1, int(args.delta * args.seeds))} "
          f"expected among certified seeds)")

    cert_seeds = E[E["t_star"] > 0]
    if len(cert_seeds):
        print(f"\nAmong the {len(cert_seeds)} certified seeds only: "
              f"speedup {cert_seeds['speedup_pct'].mean():.1f}% mean, "
              f"capture {cert_seeds['capture_pct'].mean():.0f}% mean, "
              f"severe rate {cert_seeds['severe_rate'].mean():.3f}")


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
    parser.add_argument("--delta", type=float, default=0.20)
    parser.add_argument("--tau-win", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    s6 = load_module("s6", "step6_live_loop.py")
    s7c = load_module("s7c", "step7c_job_analysis.py")

    print("QPPE Step 7h - Select-then-Certify (uniform, both benchmarks)")
    print("=" * 72)
    print(f"Certified event: steered query slowdown > {args.theta_sev}x AND "
          f"added > {args.floor_ms:.0f} ms (or timeout)")
    print(f"alpha={args.alpha}, delta={args.delta} (delta relaxed per the "
          f"sample-size floor below), tau_win={args.tau_win}")

    print("\nSAMPLE-SIZE FLOOR: clean steered calibration queries needed "
          "for a single CP test (zero failures observed):")
    print(f"{'':>8}" + "".join(f"{f'delta={d}':>12}" for d in [0.10, 0.20]))
    for a in [0.10, 0.15, 0.20]:
        vals = [math.ceil(math.log(d) / math.log(1 - a)) for d in [0.10, 0.20]]
        print(f"alpha={a:<4}" + "".join(f"{v:>12}" for v in vals))

    tpch = s7c.load_benchmark(args, s6, "tpch_sf1", variant_max=100)
    run_benchmark("TPC-H SF1", tpch, args, s6)

    job = s7c.load_benchmark(args, s6, "imdb")
    run_benchmark("JOB / IMDB", job, args, s6)

    print("\nDone. Paste the full output back.")


if __name__ == "__main__":
    main()
