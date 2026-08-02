#!/usr/bin/env python3
"""
QPPE Rebuild - Step 6b: Cross-Conformal Calibration + Live Re-Run
==================================================================
Step 6's single 25% calibration split gave t* = 0: with only 132
calibration points, the Clopper-Pearson bound needs >= 22 consecutive
regression-free lowest-risk candidates, and one unlucky ranking vetoes
everything.

Fix (standard cross-conformal / CV+ practice):
  - 5-fold split over QUERIES; each fold's candidates are risk-scored
    by heads trained on the other 4 folds (out-of-fold scores)
  - t* is calibrated on ALL 532 out-of-fold scores
  - the DEPLOYED heads are then trained on 100% of the corpus

Then the identical live experiment on the same 17 fresh queries.

Requires step6_live_loop.py and step3c_expand_corpus.py in the same
folder (imports their machinery).

Usage:
    py step6b_cross_conformal.py --user postgres --password 12345
"""

import argparse
import importlib.util
import json
import pathlib
import time

import numpy as np
import pandas as pd


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, str(pathlib.Path(__file__).parent / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--bench-db", default="tpch_sf1")
    parser.add_argument("--log-db", default="qppe")
    parser.add_argument("--theta-reg", type=float, default=1.2)
    parser.add_argument("--theta-win", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    import psycopg2
    from sklearn.ensemble import GradientBoostingClassifier

    s6 = load_module("s6", "step6_live_loop.py")
    s3c = load_module("s3c", "step3c_expand_corpus.py")
    TEMPLATES = s3c.TEMPLATES
    RATIO_EPS = 1e-9

    print("QPPE Step 6b - Cross-Conformal Calibration")
    print("=" * 72)

    # ------------------------------------------------------------
    # corpus + labels (identical to step 6)
    # ------------------------------------------------------------
    corpus = s6.load_corpus(args)
    corpus["is_regression"] = ((corpus["median_ms"] /
                                (corpus["def_median_ms"] + RATIO_EPS)
                                > args.theta_reg) | corpus["censored"]).astype(int)
    corpus["is_win"] = ((corpus["median_ms"] /
                         (corpus["def_median_ms"] + RATIO_EPS)
                         < args.theta_win) & ~corpus["censored"]).astype(int)
    feature_cols = [c for c in corpus.columns
                    if c not in ("template", "query_id", "median_ms",
                                 "def_median_ms", "censored",
                                 "is_regression", "is_win")]

    # ------------------------------------------------------------
    # out-of-fold risk scores over the WHOLE corpus
    # ------------------------------------------------------------
    rng = np.random.default_rng(7)
    qids = rng.permutation(corpus["query_id"].unique())
    folds = np.array_split(qids, args.folds)
    corpus = corpus.reset_index(drop=True)
    corpus["oof_risk"] = np.nan

    def new_model():
        return GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42)

    for i, fold_q in enumerate(folds):
        te = corpus.query_id.isin(set(fold_q))
        tr = ~te
        m = new_model()
        m.fit(corpus.loc[tr, feature_cols], corpus.loc[tr, "is_regression"])
        corpus.loc[te, "oof_risk"] = m.predict_proba(
            corpus.loc[te, feature_cols])[:, 1]
    print(f"Out-of-fold risk scores computed for {len(corpus)} candidates "
          f"({args.folds} folds over queries).")

    # calibration diagnostics: how long is the clean lowest-risk prefix?
    order = np.argsort(corpus["oof_risk"].values)
    y_sorted = corpus["is_regression"].values[order]
    first_reg = int(np.argmax(y_sorted == 1)) if y_sorted.any() else len(y_sorted)
    print(f"Lowest-risk prefix before first regression: {first_reg} candidates "
          f"(single-split Step 6 needed >= 22 and did not have them)")

    t_star = s6.calibrate_threshold(corpus["oof_risk"].values,
                                    corpus["is_regression"].values,
                                    args.alpha, args.delta)
    n_cert = int((corpus["oof_risk"] < t_star).sum())
    k_cert = int(corpus.loc[corpus["oof_risk"] < t_star, "is_regression"].sum())
    print(f"Cross-conformal t* = {t_star:.3f}  "
          f"(certifies {n_cert} calibration candidates, {k_cert} regressions "
          f"among them, empirical rate "
          f"{k_cert/max(n_cert,1):.3f})")

    if t_star <= 0:
        print("\nt* is still 0 - the corpus cannot support the guarantee at "
              f"alpha={args.alpha}, delta={args.delta}.")
        print("Options: --alpha 0.15, --delta 0.2, or expand the corpus.")
        return

    # ------------------------------------------------------------
    # deploy: heads on 100% of corpus, then the identical live loop
    # ------------------------------------------------------------
    heads = {}
    for label in ["is_regression", "is_win"]:
        m = new_model()
        m.fit(corpus[feature_cols], corpus[label])
        heads[label] = m

    bench = psycopg2.connect(dbname=args.bench_db, user=args.user,
                             password=args.password, host=args.host, port=args.port)
    bench.autocommit = True
    bcur = bench.cursor()

    results = []
    print(f"\n{'query':<9}{'decision':<22}{'p_risk':>7}{'default ms':>11}"
          f"{'chosen ms':>10}{'overhead ms':>12}{'result':>11}")
    print("-" * 84)
    for template, variant, params in s6.FRESH:
        sql = TEMPLATES[template].format(**params)

        t0 = time.perf_counter()
        plans = {}
        for hs_name, gucs in s6.HINT_SETS.items():
            bcur.execute("BEGIN;")
            try:
                for guc, val in gucs.items():
                    bcur.execute(f"SET LOCAL {guc} = {val};")
                bcur.execute(f"EXPLAIN (FORMAT JSON) {sql};")
                plan = bcur.fetchone()[0]
                if isinstance(plan, str):
                    plan = json.loads(plan)
                plans[hs_name] = plan[0]["Plan"]
            finally:
                bcur.execute("ROLLBACK;")
        gen_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        def_flat = s6.flat_features(plans["default"])
        def_node = s6.node_features(plans["default"])
        rows, names = [], []
        for hs_name, root in plans.items():
            if hs_name == "default":
                continue
            rows.append(s6.build_feature_row(
                s6.flat_features(root), s6.node_features(root),
                def_flat, def_node, hs_name))
            names.append(hs_name)
        F = pd.DataFrame(rows)[feature_cols]
        p_risk = heads["is_regression"].predict_proba(F)[:, 1]
        p_win = heads["is_win"].predict_proba(F)[:, 1]
        inf_ms = (time.perf_counter() - t0) * 1000

        certified = [(names[i], p_risk[i], p_win[i])
                     for i in range(len(names)) if p_risk[i] < t_star]
        if certified:
            chosen_hs, chosen_risk, _ = max(certified, key=lambda x: x[2])
            decision = f"steer:{chosen_hs}"
        else:
            chosen_hs, chosen_risk = "default", float("nan")
            decision = "keep default"

        d_ms, _ = s6.measure(bcur, sql, {}, 120000)
        timeout = min(max(3 * d_ms, 10000), 120000)
        if chosen_hs == "default":
            c_ms = d_ms
        else:
            c_ms, _ = s6.measure(bcur, sql, s6.HINT_SETS[chosen_hs], timeout)

        regressed = (chosen_hs != "default") and (c_ms > args.theta_reg * d_ms)
        if chosen_hs == "default":
            result = "-"
        elif regressed:
            result = "REGRESSED"
        elif c_ms < d_ms:
            result = f"won {d_ms/c_ms:.1f}x"
        else:
            result = "~neutral"
        results.append(dict(query=f"{template}v{variant}",
                            default_ms=d_ms, chosen_ms=c_ms,
                            gen_ms=gen_ms, inf_ms=inf_ms,
                            steered=chosen_hs != "default",
                            regressed=regressed))
        print(f"{template + 'v' + str(variant):<9}{decision:<22}"
              f"{chosen_risk:>7.3f}{d_ms:>11.0f}{c_ms:>10.0f}"
              f"{gen_ms + inf_ms:>12.1f}{result:>11}")

    R = pd.DataFrame(results)
    tot_def = R.default_ms.sum()
    tot_pol = R.chosen_ms.sum() + R.gen_ms.sum() + R.inf_ms.sum()
    n_st, n_rg = int(R.steered.sum()), int(R.regressed.sum())
    print("-" * 84)
    print(f"\nLIVE RESULTS ({len(R)} fresh queries, cross-conformal t* = "
          f"{t_star:.3f})")
    print(f"  steered: {n_st} | regressions: {n_rg} "
          f"(rate {n_rg/max(n_st,1):.2f} vs alpha {args.alpha})")
    print(f"  workload: default {tot_def/1000:.1f}s -> gated "
          f"{tot_pol/1000:.1f}s ({(tot_def-tot_pol)/tot_def*100:+.1f}%, "
          f"overhead included)")
    print(f"  decision overhead: {R.gen_ms.mean():.1f} + {R.inf_ms.mean():.1f} "
          f"= {R.gen_ms.mean()+R.inf_ms.mean():.1f} ms avg per query")
    print("\nPaste the full output back.")


if __name__ == "__main__":
    main()
