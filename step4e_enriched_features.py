#!/usr/bin/env python3
"""
QPPE Rebuild - Step 4e: Node-Level Plan Features
=================================================
Step 4d showed the win head cannot distinguish "no_hashjoin wins here"
(Q7/Q13/Q17) from "no_hashjoin loses here" (Q9/Q3) - flat count-deltas
describe the MOVE but not the SITUATION. This step extracts node-level
features from the stored plan JSON (no re-execution needed):

  per plan:  log_max_nl_inner_rows   biggest nested-loop inner side
             log_max_hash_build_rows biggest hash build side
             log_max_sort_rows       biggest sort input
             max_self_cost_share     cost concentration in one node
             log_max_node_rows       biggest intermediate result
             log_blowup              max intermediate / final rows
  plus each as a DELTA vs the default plan's value.

Two configurations evaluated side by side:
  A) node features + Step 4d features
  B) A + hint-set identity (one-hot)     [decision-time-legitimate]

Usage:
    py step4e_enriched_features.py --user postgres --password 12345
"""

import argparse
import math
import pickle

import numpy as np
import pandas as pd

RATIO_EPS = 1e-9

RAW = ["est_cost", "est_rows", "num_nodes", "max_depth", "num_relations",
       "num_joins", "n_nestloop", "n_hashjoin", "n_mergejoin",
       "n_seqscan", "n_indexscan", "n_bitmapscan", "n_sort", "n_agg",
       "n_gather", "n_parallel_nodes", "n_memoize", "n_material"]

NODE_FEATS = ["log_max_nl_inner_rows", "log_max_hash_build_rows",
              "log_max_sort_rows", "max_self_cost_share",
              "log_max_node_rows", "log_blowup"]


# ----------------------------------------------------------------
# node-level extraction from EXPLAIN (FORMAT JSON)
# ----------------------------------------------------------------
def node_features(root):
    max_nl_inner = 0.0
    max_hash_build = 0.0
    max_sort_rows = 0.0
    max_rows = float(root.get("Plan Rows", 0) or 0)
    max_self_cost = 0.0
    total_cost = float(root.get("Total Cost", 0) or 0)

    def visit(node):
        nonlocal max_nl_inner, max_hash_build, max_sort_rows, max_rows, max_self_cost
        rows = float(node.get("Plan Rows", 0) or 0)
        max_rows = max(max_rows, rows)
        children = node.get("Plans", [])
        self_cost = float(node.get("Total Cost", 0) or 0) - sum(
            float(c.get("Total Cost", 0) or 0) for c in children)
        max_self_cost = max(max_self_cost, max(self_cost, 0.0))

        nt = node.get("Node Type")
        if nt == "Nested Loop" and len(children) >= 2:
            # inner side is the second child in EXPLAIN output
            max_nl_inner = max(max_nl_inner,
                               float(children[1].get("Plan Rows", 0) or 0))
        if nt == "Hash Join":
            for c in children:
                if c.get("Node Type") == "Hash":
                    h = c.get("Plans", [{}])
                    build_rows = float((h[0] if h else {}).get("Plan Rows", 0) or 0)
                    max_hash_build = max(max_hash_build, build_rows)
        if nt in ("Sort", "Incremental Sort") and children:
            max_sort_rows = max(max_sort_rows,
                                float(children[0].get("Plan Rows", 0) or 0))
        for c in children:
            visit(c)

    visit(root)
    root_rows = max(float(root.get("Plan Rows", 0) or 0), 1.0)
    return {
        "log_max_nl_inner_rows": math.log1p(max_nl_inner),
        "log_max_hash_build_rows": math.log1p(max_hash_build),
        "log_max_sort_rows": math.log1p(max_sort_rows),
        "max_self_cost_share": (max_self_cost / total_cost) if total_cost > 0 else 0.0,
        "log_max_node_rows": math.log1p(max_rows),
        "log_blowup": math.log1p(max_rows / root_rows),
    }


def load_dataset(args):
    import psycopg2
    conn = psycopg2.connect(dbname=args.log_db, user=args.user,
                            password=args.password, host=args.host, port=args.port)
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
               p.features, p.plan_json, t.median_ms, t.censored
        FROM timings t
        JOIN plans p      ON p.plan_id = t.plan_id
        JOIN hint_sets h  ON h.hint_set_id = p.hint_set_id
        JOIN queries q    ON q.query_id = p.query_id
        ORDER BY q.query_id, p.plan_id;
    """, conn)
    conn.close()

    flat = pd.json_normalize(df["features"])
    node = pd.DataFrame([node_features(pj[0]["Plan"]) for pj in df["plan_json"]])
    df = pd.concat([df.drop(columns=["features", "plan_json"]), flat, node], axis=1)

    defaults = (df[df["hint_set"] == "default"]
                .set_index("query_id")[["median_ms"] + RAW + NODE_FEATS]
                .add_prefix("def_"))
    df = df.join(defaults, on="query_id")

    cand = df[df["hint_set"] != "default"].copy()
    cand["slowdown"] = cand["median_ms"] / (cand["def_median_ms"] + RATIO_EPS)

    # Step 4d feature block
    cand["cost_ratio"] = cand["est_cost"] / (cand["def_est_cost"] + RATIO_EPS)
    cand["log_cost_ratio"] = np.log(cand["cost_ratio"].clip(lower=1e-6))
    cand["rows_ratio"] = (cand["est_rows"] + 1) / (cand["def_est_rows"] + 1)
    cand["log_rows_ratio"] = np.log(cand["rows_ratio"].clip(lower=1e-6))
    cand["nodes_ratio"] = cand["num_nodes"] / (cand["def_num_nodes"] + RATIO_EPS)
    cand["depth_delta"] = cand["max_depth"] - cand["def_max_depth"]
    for k in ["nestloop", "hashjoin", "mergejoin"]:
        cand[f"{k}_delta"] = cand[f"n_{k}"] - cand[f"def_n_{k}"]
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

    base_features = ["cost_ratio", "log_cost_ratio", "rows_ratio", "log_rows_ratio",
                     "nodes_ratio", "depth_delta", "nestloop_delta", "hashjoin_delta",
                     "mergejoin_delta", "seqscan_delta", "indexscan_delta",
                     "bitmapscan_delta", "sort_delta", "material_delta",
                     "memoize_delta", "gather_delta",
                     "nestloop_share", "hashjoin_share", "mergejoin_share"]

    # node features of the candidate + deltas vs default
    for nf in NODE_FEATS:
        cand[f"{nf}_delta"] = cand[nf] - cand[f"def_{nf}"]
    node_block = NODE_FEATS + [f"{nf}_delta" for nf in NODE_FEATS] \
        + [f"def_{nf}" for nf in NODE_FEATS]

    # hint-set one-hot
    hs_dummies = pd.get_dummies(cand["hint_set"], prefix="hs")
    cand = pd.concat([cand, hs_dummies], axis=1)
    hs_block = list(hs_dummies.columns)

    feats_A = base_features + node_block
    feats_B = feats_A + hs_block
    return cand, feats_A, feats_B


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
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        m.fit(cand.loc[tr, features], cand.loc[tr, label_col])
        out.loc[te] = m.predict_proba(cand.loc[te, features])[:, 1]
    return out


def head_metrics(cand, pcol, label):
    pred = (cand[pcol] >= 0.5).astype(int)
    y = cand[label]
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return prec, rec, tp, fp, fn


def simulate(cand, prisk, pwin, tau_risk, tau_win, theta_reg):
    rows = []
    for qid, g in cand.groupby("query_id"):
        d_ms = g["def_median_ms"].iloc[0]
        best_ms = min(d_ms, g["median_ms"].min())
        eligible = g[g[prisk] < tau_risk]
        steered, regressed, chosen_ms = False, False, d_ms
        if not eligible.empty:
            top = eligible.loc[eligible[pwin].idxmax()]
            if top[pwin] > tau_win:
                steered = True
                chosen_ms = top["median_ms"]
                regressed = chosen_ms > theta_reg * d_ms
        rows.append(dict(template=g["template"].iloc[0], variant=g["variant"].iloc[0],
                         default_ms=d_ms, oracle_ms=best_ms,
                         policy_ms=chosen_ms, steered=steered, regressed=regressed))
    P = pd.DataFrame(rows)
    tot_def, tot_pol, tot_or = P.default_ms.sum(), P.policy_ms.sum(), P.oracle_ms.sum()
    gain_possible = tot_def - tot_or
    return {"P": P, "workload_s": tot_pol / 1000,
            "capture_pct": ((tot_def - tot_pol) / gain_possible * 100)
                           if gain_possible > 0 else 0.0,
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
    parser.add_argument("--model-out", default="qppe_two_head_model_v3.pkl")
    args = parser.parse_args()

    cand, feats_A, feats_B = load_dataset(args)
    cand = cand.reset_index(drop=True)
    cand["is_regression"] = ((cand["slowdown"] > args.theta_reg) |
                             cand["censored"]).astype(int)
    cand["is_win"] = ((cand["slowdown"] < args.theta_win) &
                      (~cand["censored"])).astype(int)

    print("QPPE Step 4e - Node-Level Features")
    print("=" * 72)
    print(f"Samples: {len(cand)} | templates: {cand['template'].nunique()} | "
          f"regressions {int(cand.is_regression.sum())} | "
          f"wins {int(cand.is_win.sum())}")

    configs = [("A: +node features", feats_A),
               ("B: +node +hint-set id", feats_B)]

    results = {}
    for name, feats in configs:
        rk = f"p_risk_{name[0]}"
        wk = f"p_win_{name[0]}"
        cand[rk] = loto_probs(cand, feats, "is_regression")
        cand[wk] = loto_probs(cand, feats, "is_win")
        rprec, rrec, *_ = head_metrics(cand, rk, "is_regression")
        wprec, wrec, wtp, wfp, wfn = head_metrics(cand, wk, "is_win")
        print(f"\nConfig {name} ({len(feats)} features)")
        print(f"  RISK head: precision {rprec:.1%}  recall {rrec:.1%}")
        print(f"  WIN head:  precision {wprec:.1%}  recall {wrec:.1%}  "
              f"(TP {wtp} / FP {wfp} / FN {wfn})")
        print(f"    [4d reference - RISK: 72.8%/79.1%, WIN: 36.0%/17.6%]")
        results[name[0]] = (rk, wk)

    print("\nPOLICY GRID (LOTO)")
    print(f"{'config':>7}{'tau_risk':>9}{'tau_win':>9}{'workload s':>12}"
          f"{'capture':>9}{'steers':>8}{'regs':>6}")
    print("-" * 62)
    for cfg, (rk, wk) in results.items():
        for tr_ in [0.2, 0.3, 0.5]:
            for tw_ in [0.5, 0.6, 0.7]:
                r = simulate(cand, rk, wk, tr_, tw_, args.theta_reg)
                print(f"{cfg:>7}{tr_:>9.1f}{tw_:>9.1f}{r['workload_s']:>12.1f}"
                      f"{r['capture_pct']:>8.0f}%{r['n_steer']:>8}{r['n_reg']:>6}")

    # best config detail: pick grid point with max capture s.t. regs <= 1,
    # else max capture overall
    best = None
    for cfg, (rk, wk) in results.items():
        for tr_ in [0.2, 0.3, 0.5]:
            for tw_ in [0.5, 0.6, 0.7]:
                r = simulate(cand, rk, wk, tr_, tw_, args.theta_reg)
                key = (r["n_reg"] <= 1, r["capture_pct"])
                if best is None or key > best[0]:
                    best = (key, cfg, tr_, tw_, r)
    _, cfg, tr_, tw_, r = best
    print(f"\nBEST OPERATING POINT: config {cfg}, tau_risk={tr_}, tau_win={tw_}")
    print(f"  workload {r['workload_s']:.1f}s | capture {r['capture_pct']:.0f}% | "
          f"{r['n_steer']} steers | {r['n_reg']} regressions")
    acted = r["P"][r["P"].steered]
    print(f"\n{'query':<9}{'default ms':>11}{'policy ms':>11}{'result':>12}")
    print("-" * 45)
    for _, row in acted.sort_values(["template", "variant"]).iterrows():
        if row.regressed:
            res = "REGRESSED"
        elif row.policy_ms < row.default_ms:
            res = f"won {row.default_ms/row.policy_ms:.1f}x"
        else:
            res = "~neutral"
        print(f"{row.template + 'v' + str(row.variant):<9}"
              f"{row.default_ms:>11.0f}{row.policy_ms:>11.0f}{res:>12}")

    # save config-B heads trained on all data (richest, decision-time legal)
    from sklearn.ensemble import GradientBoostingClassifier
    heads = {}
    for label in ["is_regression", "is_win"]:
        m = GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        m.fit(cand[feats_B], cand[label])
        heads[label] = m
    with open(args.model_out, "wb") as f:
        pickle.dump({"risk_model": heads["is_regression"],
                     "win_model": heads["is_win"],
                     "feature_cols": feats_B,
                     "theta_reg": args.theta_reg,
                     "theta_win": args.theta_win}, f)
    print(f"\nSaved config-B heads to {args.model_out}. "
          f"Paste the full output back.")


if __name__ == "__main__":
    main()
