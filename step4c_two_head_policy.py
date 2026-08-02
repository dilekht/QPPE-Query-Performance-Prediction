#!/usr/bin/env python3
"""
QPPE Rebuild - Step 4c: Two-Head Policy (Risk + Win)
=====================================================
Step 4b's lesson: a risk model alone steers into neutral plans and
occasional disasters. A steering decision needs BOTH:
    RISK head: P(slowdown > theta_reg)   - the brake
    WIN head:  P(slowdown < theta_win)   - the engine

Policy(tau_risk, tau_win):
    eligible = candidates with p_risk < tau_risk
    best     = eligible candidate with highest p_win
    steer to best iff p_win(best) > tau_win, else keep default.

Evaluated with leave-one-template-out probabilities across a grid of
thresholds, so we see the whole conservativeness/aggressiveness
trade-off, then a per-query breakdown at a chosen operating point.

Usage:
    py step4c_two_head_policy.py --user postgres --password 12345
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


def loto_probs(cand, features, label_col):
    """Leave-one-template-out probabilities for a binary label."""
    from sklearn.ensemble import GradientBoostingClassifier
    out = pd.Series(np.nan, index=cand.index)
    for held_out in sorted(cand["template"].unique()):
        tr = cand["template"] != held_out
        te = ~tr
        if cand.loc[tr, label_col].nunique() < 2:
            out.loc[te] = float(cand.loc[tr, label_col].mode()[0])
            continue
        m = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        m.fit(cand.loc[tr, features], cand.loc[tr, label_col])
        out.loc[te] = m.predict_proba(cand.loc[te, features])[:, 1]
    return out


def simulate(cand, tau_risk, tau_win, theta_reg):
    """Apply the two-head policy per query; return summary + rows."""
    rows = []
    for qid, g in cand.groupby("query_id"):
        d_ms = g["def_median_ms"].iloc[0]
        best_ms = min(d_ms, g["median_ms"].min())
        eligible = g[g["p_risk"] < tau_risk]
        steered, regressed, chosen_ms = False, False, d_ms
        if not eligible.empty:
            top = eligible.loc[eligible["p_win"].idxmax()]
            if top["p_win"] > tau_win:
                steered = True
                chosen_ms = top["median_ms"]
                regressed = chosen_ms > theta_reg * d_ms
        rows.append(dict(template=g["template"].iloc[0],
                         variant=g["variant"].iloc[0],
                         default_ms=d_ms, oracle_ms=best_ms,
                         policy_ms=chosen_ms, steered=steered,
                         regressed=regressed))
    P = pd.DataFrame(rows)
    tot_def, tot_pol, tot_or = (P["default_ms"].sum(),
                                P["policy_ms"].sum(), P["oracle_ms"].sum())
    gain_possible = tot_def - tot_or
    gain_captured = tot_def - tot_pol
    return {
        "P": P,
        "workload_s": tot_pol / 1000,
        "delta_pct": -(gain_captured / tot_def) * 100,
        "capture_pct": (gain_captured / gain_possible * 100) if gain_possible > 0 else 0.0,
        "n_steer": int(P.steered.sum()),
        "n_reg": int(P.regressed.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--log-db", default="qppe")
    parser.add_argument("--theta-reg", type=float, default=1.2)
    parser.add_argument("--theta-win", type=float, default=0.9)
    parser.add_argument("--op-risk", type=float, default=0.3,
                        help="operating point tau_risk for the detail table")
    parser.add_argument("--op-win", type=float, default=0.6,
                        help="operating point tau_win for the detail table")
    parser.add_argument("--model-out", default="qppe_two_head_model.pkl")
    args = parser.parse_args()

    cand, features = load_dataset(args)
    cand = cand.reset_index(drop=True)
    cand["is_regression"] = (cand["slowdown"] > args.theta_reg).astype(int)
    cand["is_win"] = (cand["slowdown"] < args.theta_win).astype(int)

    print("QPPE Step 4c - Two-Head Policy (Risk + Win)")
    print("=" * 72)
    print(f"Samples: {len(cand)} | regressions: {int(cand.is_regression.sum())} "
          f"| wins: {int(cand.is_win.sum())} "
          f"(theta_reg={args.theta_reg}, theta_win={args.theta_win})")

    cand["p_risk"] = loto_probs(cand, features, "is_regression")
    cand["p_win"] = loto_probs(cand, features, "is_win")

    # ------------------------------------------------------------
    # Threshold grid: the conservativeness trade-off
    # ------------------------------------------------------------
    print("\nPOLICY GRID (LOTO, whole workload = 60.1 s never-steer, "
          "33.8 s oracle)")
    print(f"{'tau_risk':>9}{'tau_win':>9}{'workload s':>12}{'vs default':>12}"
          f"{'capture':>9}{'steers':>8}{'regs':>6}")
    print("-" * 65)
    for tr_ in [0.2, 0.3, 0.5]:
        for tw_ in [0.5, 0.6, 0.7]:
            r = simulate(cand, tr_, tw_, args.theta_reg)
            print(f"{tr_:>9.1f}{tw_:>9.1f}{r['workload_s']:>12.1f}"
                  f"{r['delta_pct']:>11.1f}%{r['capture_pct']:>8.0f}%"
                  f"{r['n_steer']:>8}{r['n_reg']:>6}")

    # ------------------------------------------------------------
    # Detail at the chosen operating point
    # ------------------------------------------------------------
    r = simulate(cand, args.op_risk, args.op_win, args.theta_reg)
    print(f"\nOPERATING POINT tau_risk={args.op_risk}, tau_win={args.op_win}: "
          f"{r['workload_s']:.1f}s ({r['delta_pct']:+.1f}%), "
          f"capture {r['capture_pct']:.0f}%, "
          f"{r['n_steer']} steers, {r['n_reg']} regressions")
    print(f"\n{'query':<9}{'default ms':>11}{'policy ms':>11}{'oracle ms':>11}"
          f"{'action':>9}{'result':>12}")
    print("-" * 65)
    for _, row in r["P"].sort_values(["template", "variant"]).iterrows():
        action = "steer" if row.steered else "keep"
        if not row.steered:
            result = "-"
        elif row.regressed:
            result = "REGRESSED"
        elif row.policy_ms < row.default_ms:
            result = f"won {row.default_ms/row.policy_ms:.1f}x"
        else:
            result = "~neutral"
        print(f"{row.template + 'v' + str(row.variant):<9}"
              f"{row.default_ms:>11.0f}{row.policy_ms:>11.0f}"
              f"{row.oracle_ms:>11.0f}{action:>9}{result:>12}")

    # ------------------------------------------------------------
    # Save both heads trained on all data
    # ------------------------------------------------------------
    from sklearn.ensemble import GradientBoostingClassifier
    heads = {}
    for label in ["is_regression", "is_win"]:
        m = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        m.fit(cand[features], cand[label])
        heads[label] = m
    with open(args.model_out, "wb") as f:
        pickle.dump({"risk_model": heads["is_regression"],
                     "win_model": heads["is_win"],
                     "feature_cols": features,
                     "theta_reg": args.theta_reg,
                     "theta_win": args.theta_win}, f)
    print(f"\nBoth heads saved to {args.model_out}. Paste the output back.")


if __name__ == "__main__":
    main()
