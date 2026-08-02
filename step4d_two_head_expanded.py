#!/usr/bin/env python3
"""
QPPE Rebuild - Step 4d: Two-Head Policy on the Expanded Corpus
===============================================================
Identical method to Step 4c, two changes:

  1. CENSORED PLANS INCLUDED: timed-out plans enter the dataset with
     their timeout as a LOWER BOUND on runtime and is_regression = 1.
     (Step 4c's data query silently dropped them.)
  2. Corpus: 20 templates / 105 query instances / ~590 candidates.

LOTO is now over 20 folds. If cross-template win patterns
(e.g. no_hashjoin on Q7/Q13/Q17) are learnable, this is where it shows.

Usage:
    py step4d_two_head_expanded.py --user postgres --password 12345
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
    # median over ALL measured runs incl. censored; censored flag carried
    df = pd.read_sql("""
        WITH timings AS (
            SELECT e.plan_id,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY e.exec_ms) AS median_ms,
                   bool_or(e.timed_out) AS censored
            FROM executions e
            WHERE NOT e.is_warmup
            GROUP BY e.plan_id
        )
        SELECT q.template, q.variant, q.query_id,
               h.name AS hint_set, p.plan_id,
               p.features, t.median_ms, t.censored
        FROM timings t
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
                chosen_ms = top["median_ms"]  # censored = timeout lower bound
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
    return {"P": P, "workload_s": tot_pol / 1000,
            "delta_pct": -(gain_captured / tot_def) * 100,
            "capture_pct": (gain_captured / gain_possible * 100) if gain_possible > 0 else 0.0,
            "n_steer": int(P.steered.sum()), "n_reg": int(P.regressed.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--log-db", default="qppe")
    parser.add_argument("--theta-reg", type=float, default=1.2)
    parser.add_argument("--theta-win", type=float, default=0.9)
    parser.add_argument("--op-risk", type=float, default=0.3)
    parser.add_argument("--op-win", type=float, default=0.6)
    parser.add_argument("--model-out", default="qppe_two_head_model_v2.pkl")
    args = parser.parse_args()

    cand, features = load_dataset(args)
    cand = cand.reset_index(drop=True)
    cand["is_regression"] = ((cand["slowdown"] > args.theta_reg) |
                             cand["censored"]).astype(int)
    cand["is_win"] = ((cand["slowdown"] < args.theta_win) &
                      (~cand["censored"])).astype(int)

    n_cens = int(cand["censored"].sum())
    print("QPPE Step 4d - Two-Head Policy, Expanded Corpus")
    print("=" * 72)
    print(f"Samples: {len(cand)} ({n_cens} censored) | templates: "
          f"{cand['template'].nunique()} | query instances: "
          f"{cand['query_id'].nunique()}")
    print(f"Regressions: {int(cand.is_regression.sum())} | "
          f"wins: {int(cand.is_win.sum())} "
          f"(theta_reg={args.theta_reg}, theta_win={args.theta_win})")

    cand["p_risk"] = loto_probs(cand, features, "is_regression")
    cand["p_win"] = loto_probs(cand, features, "is_win")

    # classification summary for both heads
    for label, pcol, name in [("is_regression", "p_risk", "RISK head"),
                              ("is_win", "p_win", "WIN head")]:
        pred = (cand[pcol] >= 0.5).astype(int)
        y = cand[label]
        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"\n{name} (LOTO, 0.5): precision {prec:.1%}  recall {rec:.1%}  "
              f"(TP {tp} / FP {fp} / FN {fn})")

    tot_def = 0.0
    tot_or = 0.0
    for qid, g in cand.groupby("query_id"):
        d = g["def_median_ms"].iloc[0]
        tot_def += d
        tot_or += min(d, g["median_ms"].min())
    print(f"\nWorkload bounds: never-steer {tot_def/1000:.1f} s | "
          f"oracle {tot_or/1000:.1f} s "
          f"(possible improvement {(tot_def-tot_or)/tot_def:.0%})")

    print("\nPOLICY GRID (LOTO)")
    print(f"{'tau_risk':>9}{'tau_win':>9}{'workload s':>12}{'vs default':>12}"
          f"{'capture':>9}{'steers':>8}{'regs':>6}")
    print("-" * 65)
    for tr_ in [0.1, 0.2, 0.3, 0.5]:
        for tw_ in [0.5, 0.6, 0.7]:
            r = simulate(cand, tr_, tw_, args.theta_reg)
            print(f"{tr_:>9.1f}{tw_:>9.1f}{r['workload_s']:>12.1f}"
                  f"{r['delta_pct']:>11.1f}%{r['capture_pct']:>8.0f}%"
                  f"{r['n_steer']:>8}{r['n_reg']:>6}")

    r = simulate(cand, args.op_risk, args.op_win, args.theta_reg)
    print(f"\nOPERATING POINT tau_risk={args.op_risk}, tau_win={args.op_win}: "
          f"{r['workload_s']:.1f}s ({r['delta_pct']:+.1f}%), capture "
          f"{r['capture_pct']:.0f}%, {r['n_steer']} steers, "
          f"{r['n_reg']} regressions")

    P = r["P"]
    acted = P[P.steered]
    print(f"\nSteered queries only ({len(acted)}):")
    print(f"{'query':<9}{'default ms':>11}{'policy ms':>11}{'oracle ms':>11}"
          f"{'result':>12}")
    print("-" * 55)
    for _, row in acted.sort_values(["template", "variant"]).iterrows():
        if row.regressed:
            result = "REGRESSED"
        elif row.policy_ms < row.default_ms:
            result = f"won {row.default_ms/row.policy_ms:.1f}x"
        else:
            result = "~neutral"
        print(f"{row.template + 'v' + str(row.variant):<9}"
              f"{row.default_ms:>11.0f}{row.policy_ms:>11.0f}"
              f"{row.oracle_ms:>11.0f}{result:>12}")

    kept = P[~P.steered]
    missed_wins = kept[kept.oracle_ms < 0.7 * kept.default_ms]
    print(f"\nKept-default queries with a >1.4x win left on the table: "
          f"{len(missed_wins)}")
    for _, row in missed_wins.sort_values("default_ms", ascending=False).head(8).iterrows():
        print(f"  {row.template}v{row.variant}: default {row.default_ms:.0f} ms, "
              f"oracle {row.oracle_ms:.0f} ms "
              f"({row.default_ms/row.oracle_ms:.1f}x available)")

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
    print(f"\nSaved to {args.model_out}. Paste the full output back.")


if __name__ == "__main__":
    main()
