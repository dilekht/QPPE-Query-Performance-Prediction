#!/usr/bin/env python3
"""
QPPE Rebuild - Step 4: Labeling + Regression-Detection Model
=============================================================
Builds the training set from the qppe log database:
  - one sample per (query, candidate plan), default plans excluded
  - label: REGRESSION if candidate_median > THETA x default_median
  - features: candidate plan features + default plan features + ratios
    (ratio features are the scale-invariant core)

Trains a GradientBoostingClassifier and evaluates it the honest way:
  LEAVE-ONE-TEMPLATE-OUT cross-validation (train on 9 templates,
  test on the unseen 10th). Compares against a cost-ratio-only
  baseline, which the ML model must beat to justify its existence.

Saves the final model (trained on all data) for Step 5 (conformal).

Usage:
    py step4_label_and_train.py --user postgres --password 12345
    py step4_label_and_train.py --user postgres --password 12345 --theta 1.2
"""

import argparse
import pickle

import numpy as np
import pandas as pd


RATIO_EPS = 1e-9

# candidate-plan features used directly
PLAN_FEATURES = [
    "est_cost", "est_rows", "num_nodes", "max_depth", "num_relations",
    "num_joins", "n_nestloop", "n_hashjoin", "n_mergejoin",
    "n_seqscan", "n_indexscan", "n_bitmapscan", "n_sort", "n_agg",
    "n_gather", "n_parallel_nodes", "n_memoize", "n_material",
]


def load_dataset(args):
    import psycopg2
    conn = psycopg2.connect(dbname=args.log_db, user=args.user,
                            password=args.password, host=args.host, port=args.port)
    df = pd.read_sql("""
        SELECT q.template, q.variant, q.query_id,
               h.name AS hint_set, p.plan_id, p.plan_hash,
               p.features, t.median_ms
        FROM plan_timings t
        JOIN plans p      ON p.plan_id = t.plan_id
        JOIN hint_sets h  ON h.hint_set_id = p.hint_set_id
        JOIN queries q    ON q.query_id = p.query_id
        ORDER BY q.query_id, p.plan_id;
    """, conn)
    conn.close()

    # expand JSONB features into columns
    feats = pd.json_normalize(df["features"])
    df = pd.concat([df.drop(columns=["features"]), feats], axis=1)

    # default reference per query
    defaults = (df[df["hint_set"] == "default"]
                .set_index("query_id")[["median_ms"] + PLAN_FEATURES]
                .add_prefix("def_"))
    df = df.join(defaults, on="query_id")

    # candidates only (the model never judges the default itself;
    # the default is the thing we protect)
    cand = df[df["hint_set"] != "default"].copy()

    # relative outcome and label
    cand["slowdown"] = cand["median_ms"] / (cand["def_median_ms"] + RATIO_EPS)
    cand["is_regression"] = (cand["slowdown"] > args.theta).astype(int)

    # ---- scale-invariant ratio features (the core of portability) ----
    cand["cost_ratio"] = cand["est_cost"] / (cand["def_est_cost"] + RATIO_EPS)
    cand["rows_ratio"] = (cand["est_rows"] + 1) / (cand["def_est_rows"] + 1)
    cand["nodes_ratio"] = cand["num_nodes"] / (cand["def_num_nodes"] + RATIO_EPS)
    cand["depth_delta"] = cand["max_depth"] - cand["def_max_depth"]
    cand["nestloop_delta"] = cand["n_nestloop"] - cand["def_n_nestloop"]
    cand["hashjoin_delta"] = cand["n_hashjoin"] - cand["def_n_hashjoin"]
    cand["mergejoin_delta"] = cand["n_mergejoin"] - cand["def_n_mergejoin"]
    cand["seqscan_delta"] = cand["n_seqscan"] - cand["def_n_seqscan"]
    cand["indexscan_delta"] = cand["n_indexscan"] - cand["def_n_indexscan"]
    cand["sort_delta"] = cand["n_sort"] - cand["def_n_sort"]
    cand["log_cost_ratio"] = np.log(cand["cost_ratio"].clip(lower=1e-6))

    ratio_features = ["cost_ratio", "log_cost_ratio", "rows_ratio", "nodes_ratio",
                      "depth_delta", "nestloop_delta", "hashjoin_delta",
                      "mergejoin_delta", "seqscan_delta", "indexscan_delta",
                      "sort_delta"]
    feature_cols = ratio_features + PLAN_FEATURES
    return cand, feature_cols


def evaluate(y_true, y_pred, slowdowns, label):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    # the two errors that matter for deployment:
    missed = slowdowns[(y_true == 1) & (y_pred == 0)]  # disasters let through
    worst_missed = missed.max() if len(missed) else 1.0
    print(f"  {label:<26} precision {prec:5.1%}  recall {rec:5.1%}  "
          f"| missed regressions: {len(missed):>2} (worst {worst_missed:.1f}x) "
          f"| false alarms: {fp:>2}")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "missed_worst": worst_missed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--log-db", default="qppe")
    parser.add_argument("--theta", type=float, default=1.2,
                        help="regression threshold: candidate is a regression "
                             "if slower than theta x default")
    parser.add_argument("--model-out", default="qppe_regression_model.pkl")
    args = parser.parse_args()

    from sklearn.ensemble import GradientBoostingClassifier

    cand, feature_cols = load_dataset(args)

    print("QPPE Step 4 - Labeling & Regression Detector")
    print("=" * 70)
    print(f"Candidate samples: {len(cand)}  (theta = {args.theta})")
    print(f"Features: {len(feature_cols)}")

    dist = cand.groupby("template")["is_regression"].agg(["count", "sum"])
    print(f"\n{'Template':<10}{'samples':>9}{'regressions':>13}{'safe':>7}")
    print("-" * 39)
    for t, row in dist.iterrows():
        print(f"{t:<10}{int(row['count']):>9}{int(row['sum']):>13}"
              f"{int(row['count'] - row['sum']):>7}")
    total_reg = int(cand["is_regression"].sum())
    print(f"{'TOTAL':<10}{len(cand):>9}{total_reg:>13}{len(cand) - total_reg:>7}")

    # ------------------------------------------------------------
    # Leave-one-template-out cross-validation
    # ------------------------------------------------------------
    print("\nLEAVE-ONE-TEMPLATE-OUT EVALUATION")
    print("=" * 70)

    templates = sorted(cand["template"].unique())
    y_all, pred_ml_all, pred_cost_all, slow_all = [], [], [], []

    for held_out in templates:
        train = cand[cand["template"] != held_out]
        test = cand[cand["template"] == held_out]
        if test.empty:
            continue

        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42)
        model.fit(train[feature_cols], train["is_regression"])

        pred_ml = model.predict(test[feature_cols])
        # baseline: optimizer's own belief - flag regression if the
        # candidate's estimated cost exceeds the default's
        pred_cost = (test["cost_ratio"] > 1.0).astype(int).values

        y_all.append(test["is_regression"].values)
        pred_ml_all.append(pred_ml)
        pred_cost_all.append(pred_cost)
        slow_all.append(test["slowdown"].values)

    y = np.concatenate(y_all)
    pml = np.concatenate(pred_ml_all)
    pcost = np.concatenate(pred_cost_all)
    slow = np.concatenate(slow_all)

    print(f"Pooled over {len(templates)} held-out templates "
          f"({len(y)} test predictions):\n")
    r_ml = evaluate(y, pml, slow, "ML model (GBM)")
    r_cost = evaluate(y, pcost, slow, "Baseline (cost ratio > 1)")

    verdict = ("ML model beats the cost-only baseline"
               if (r_ml["fn"] < r_cost["fn"] or
                   (r_ml["fn"] == r_cost["fn"] and r_ml["fp"] < r_cost["fp"]))
               else "ML model does NOT beat the cost-only baseline yet")
    print(f"\n  -> {verdict}")

    # per-template detail for the ML model
    print("\nPer-template detail (ML model):")
    print(f"{'held out':<10}{'n':>4}{'reg':>5}{'caught':>8}{'missed':>8}{'false+':>8}")
    print("-" * 43)
    idx = 0
    for t, yt, pt in zip(templates, y_all, pred_ml_all):
        n = len(yt)
        reg = int(yt.sum())
        caught = int(((yt == 1) & (pt == 1)).sum())
        missed = int(((yt == 1) & (pt == 0)).sum())
        fpos = int(((yt == 0) & (pt == 1)).sum())
        print(f"{t:<10}{n:>4}{reg:>5}{caught:>8}{missed:>8}{fpos:>8}")

    # ------------------------------------------------------------
    # Final model on ALL data + feature importances -> disk
    # ------------------------------------------------------------
    final = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42)
    final.fit(cand[feature_cols], cand["is_regression"])

    imp = sorted(zip(feature_cols, final.feature_importances_),
                 key=lambda x: -x[1])
    print("\nTop 10 feature importances (final model):")
    for name, v in imp[:10]:
        print(f"  {name:<20} {v:.3f}")

    with open(args.model_out, "wb") as f:
        pickle.dump({"model": final, "feature_cols": feature_cols,
                     "theta": args.theta}, f)
    print(f"\nFinal model saved to {args.model_out}")
    print("Paste the full output back.")


if __name__ == "__main__":
    main()
