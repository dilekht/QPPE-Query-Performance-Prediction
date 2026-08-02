#!/usr/bin/env python3
"""
QPPE Rebuild - Step 4b: Scale-Invariant Model + Policy Evaluation
==================================================================
Two fixes over Step 4:

FIX 1 - Features: ratio/delta features ONLY. No raw magnitudes
        (est_cost, est_rows, ...) that let the model memorize
        template scale instead of learning plan structure.

FIX 2 - Evaluation: policy-level, not just classification.
        Under leave-one-template-out predictions we simulate:
          NEVER-STEER   always run the default (= what the
                        degenerate cost baseline amounts to)
          ML-STEER      override default with the candidate whose
                        predicted regression probability is lowest,
                        but only if that probability < tau
          ORACLE        per query, the best plan in hindsight
        Metrics: total workload time, captured share of the oracle's
        possible gain, and regressions actually incurred.

Usage:
    py step4b_model_v2.py --user postgres --password 12345
    py step4b_model_v2.py --user postgres --password 12345 --tau 0.3
"""

import argparse
import pickle

import numpy as np
import pandas as pd

RATIO_EPS = 1e-9

RAW = ["est_cost", "est_rows", "num_nodes", "max_depth", "num_relations",
       "num_joins", "n_nestloop", "n_hashjoin", "n_mergejoin",
       "n_seqscan", "n_indexscan", "n_bitmapscan", "n_sort", "n_agg",
       "n_gather", "n_parallel_nodes", "n_memoize", "n_material"]


def load_dataset(args):
    import psycopg2
    conn = psycopg2.connect(dbname=args.log_db, user=args.user,
                            password=args.password, host=args.host, port=args.port)
    df = pd.read_sql("""
        SELECT q.template, q.variant, q.query_id,
               h.name AS hint_set, p.plan_id,
               p.features, t.median_ms
        FROM plan_timings t
        JOIN plans p      ON p.plan_id = t.plan_id
        JOIN hint_sets h  ON h.hint_set_id = p.hint_set_id
        JOIN queries q    ON q.query_id = p.query_id
        ORDER BY q.query_id, p.plan_id;
    """, conn)
    conn.close()

    feats = pd.json_normalize(df["features"])
    df = pd.concat([df.drop(columns=["features"]), feats], axis=1)

    defaults = (df[df["hint_set"] == "default"]
                .set_index("query_id")[["median_ms"] + RAW]
                .add_prefix("def_"))
    df = df.join(defaults, on="query_id")

    cand = df[df["hint_set"] != "default"].copy()
    cand["slowdown"] = cand["median_ms"] / (cand["def_median_ms"] + RATIO_EPS)
    cand["is_regression"] = (cand["slowdown"] > args.theta).astype(int)

    # ---- STRICTLY scale-invariant features ----
    cand["cost_ratio"] = cand["est_cost"] / (cand["def_est_cost"] + RATIO_EPS)
    cand["log_cost_ratio"] = np.log(cand["cost_ratio"].clip(lower=1e-6))
    cand["rows_ratio"] = (cand["est_rows"] + 1) / (cand["def_est_rows"] + 1)
    cand["log_rows_ratio"] = np.log(cand["rows_ratio"].clip(lower=1e-6))
    cand["nodes_ratio"] = cand["num_nodes"] / (cand["def_num_nodes"] + RATIO_EPS)
    cand["depth_delta"] = cand["max_depth"] - cand["def_max_depth"]
    cand["nestloop_delta"] = cand["n_nestloop"] - cand["def_n_nestloop"]
    cand["hashjoin_delta"] = cand["n_hashjoin"] - cand["def_n_hashjoin"]
    cand["mergejoin_delta"] = cand["n_mergejoin"] - cand["def_n_mergejoin"]
    cand["seqscan_delta"] = cand["n_seqscan"] - cand["def_n_seqscan"]
    cand["indexscan_delta"] = cand["n_indexscan"] - cand["def_n_indexscan"]
    cand["bitmapscan_delta"] = cand["n_bitmapscan"] - cand["def_n_bitmapscan"]
    cand["sort_delta"] = cand["n_sort"] - cand["def_n_sort"]
    cand["material_delta"] = cand["n_material"] - cand["def_n_material"]
    cand["memoize_delta"] = cand["n_memoize"] - cand["def_n_memoize"]
    cand["gather_delta"] = cand["n_gather"] - cand["def_n_gather"]
    # share features: composition of the candidate itself (scale-free)
    tot_joins = cand["num_joins"].clip(lower=1)
    cand["nestloop_share"] = cand["n_nestloop"] / tot_joins
    cand["hashjoin_share"] = cand["n_hashjoin"] / tot_joins
    cand["mergejoin_share"] = cand["n_mergejoin"] / tot_joins

    features = ["cost_ratio", "log_cost_ratio", "rows_ratio", "log_rows_ratio",
                "nodes_ratio", "depth_delta", "nestloop_delta", "hashjoin_delta",
                "mergejoin_delta", "seqscan_delta", "indexscan_delta",
                "bitmapscan_delta", "sort_delta", "material_delta",
                "memoize_delta", "gather_delta",
                "nestloop_share", "hashjoin_share", "mergejoin_share"]
    return cand, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--log-db", default="qppe")
    parser.add_argument("--theta", type=float, default=1.2)
    parser.add_argument("--tau", type=float, default=0.5,
                        help="steer only if predicted regression prob < tau")
    parser.add_argument("--model-out", default="qppe_regression_model_v2.pkl")
    args = parser.parse_args()

    from sklearn.ensemble import GradientBoostingClassifier

    cand, features = load_dataset(args)

    print("QPPE Step 4b - Scale-Invariant Model + Policy Evaluation")
    print("=" * 72)
    print(f"Samples: {len(cand)} | features: {len(features)} (all scale-invariant)"
          f" | theta={args.theta} | tau={args.tau}")

    # ------------------------------------------------------------
    # LOTO predictions (probabilities, kept per row)
    # ------------------------------------------------------------
    cand = cand.reset_index(drop=True)
    cand["p_reg"] = np.nan
    templates = sorted(cand["template"].unique())

    for held_out in templates:
        tr = cand["template"] != held_out
        te = ~tr
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        model.fit(cand.loc[tr, features], cand.loc[tr, "is_regression"])
        cand.loc[te, "p_reg"] = model.predict_proba(cand.loc[te, features])[:, 1]

    # classification view (threshold 0.5) for comparability with Step 4
    pred = (cand["p_reg"] >= 0.5).astype(int)
    tp = int(((cand.is_regression == 1) & (pred == 1)).sum())
    fp = int(((cand.is_regression == 0) & (pred == 1)).sum())
    fn = int(((cand.is_regression == 1) & (pred == 0)).sum())
    missed = cand.loc[(cand.is_regression == 1) & (pred == 0), "slowdown"]
    print("\nClassification (LOTO, threshold 0.5):")
    print(f"  precision {tp/(tp+fp):.1%}  recall {tp/(tp+fn):.1%}  "
          f"missed {fn} (worst {missed.max() if len(missed) else 1:.1f}x)  "
          f"false alarms {fp}")
    print("  [Step 4 with raw features was: precision 60.4%, recall 67.7%, "
          "missed 32 (worst 4.1x), false alarms 44]")

    # ------------------------------------------------------------
    # POLICY EVALUATION (the headline)
    # ------------------------------------------------------------
    print("\nPOLICY EVALUATION (per query, using LOTO probabilities)")
    print("=" * 72)

    rows = []
    for qid, g in cand.groupby("query_id"):
        d_ms = g["def_median_ms"].iloc[0]
        template = g["template"].iloc[0]
        variant = g["variant"].iloc[0]

        # oracle: best of default and all candidates
        best_ms = min(d_ms, g["median_ms"].min())

        # ML policy: steer to lowest-risk candidate if risk < tau
        gmin = g.loc[g["p_reg"].idxmin()]
        if gmin["p_reg"] < args.tau:
            ml_ms = gmin["median_ms"]
            steered = True
            regressed = ml_ms > args.theta * d_ms
        else:
            ml_ms = d_ms
            steered = False
            regressed = False

        rows.append(dict(template=template, variant=variant,
                         default_ms=d_ms, oracle_ms=best_ms, ml_ms=ml_ms,
                         steered=steered, regressed=regressed))

    P = pd.DataFrame(rows)
    tot_def = P["default_ms"].sum()
    tot_oracle = P["oracle_ms"].sum()
    tot_ml = P["ml_ms"].sum()
    possible_gain = tot_def - tot_oracle
    captured = tot_def - tot_ml

    print(f"{'Policy':<14}{'total workload':>16}{'vs default':>12}"
          f"{'steered':>9}{'regressions':>13}")
    print("-" * 64)
    print(f"{'never-steer':<14}{tot_def/1000:>13.1f} s{'0.0%':>12}"
          f"{0:>9}{0:>13}")
    print(f"{'ML-steer':<14}{tot_ml/1000:>13.1f} s"
          f"{-(captured/tot_def)*100:>11.1f}%"
          f"{int(P.steered.sum()):>9}{int(P.regressed.sum()):>13}")
    print(f"{'oracle':<14}{tot_oracle/1000:>13.1f} s"
          f"{-(possible_gain/tot_def)*100:>11.1f}%"
          f"{'-':>9}{'-':>13}")
    if possible_gain > 0:
        print(f"\nML policy captures {captured/possible_gain:.0%} of the "
              f"oracle's possible improvement, with {int(P.regressed.sum())} "
              f"regression(s) incurred.")

    print("\nPer-query policy outcome:")
    print(f"{'query':<9}{'default ms':>11}{'ML ms':>10}{'oracle ms':>11}"
          f"{'action':>10}{'result':>12}")
    print("-" * 65)
    for _, r in P.sort_values(["template", "variant"]).iterrows():
        action = "steer" if r.steered else "keep"
        if not r.steered:
            result = "-"
        elif r.regressed:
            result = "REGRESSED"
        elif r.ml_ms < r.default_ms:
            result = f"won {r.default_ms/r.ml_ms:.1f}x"
        else:
            result = "~neutral"
        print(f"{r.template + 'v' + str(r.variant):<9}{r.default_ms:>11.0f}"
              f"{r.ml_ms:>10.0f}{r.oracle_ms:>11.0f}{action:>10}{result:>12}")

    # ------------------------------------------------------------
    # final model on all data
    # ------------------------------------------------------------
    final = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=8, random_state=42)
    final.fit(cand[features], cand["is_regression"])
    imp = sorted(zip(features, final.feature_importances_), key=lambda x: -x[1])
    print("\nTop 10 feature importances (scale-invariant model):")
    for name, v in imp[:10]:
        print(f"  {name:<20} {v:.3f}")

    with open(args.model_out, "wb") as f:
        pickle.dump({"model": final, "feature_cols": features,
                     "theta": args.theta}, f)
    print(f"\nSaved to {args.model_out}. Paste the full output back.")


if __name__ == "__main__":
    main()
