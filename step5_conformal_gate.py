#!/usr/bin/env python3
"""
QPPE Rebuild - Step 5: Conformal Safety Gate
=============================================
The paper's core mechanism. Instead of a hand-tuned tau_risk, the
steering threshold is CALIBRATED to a statistical guarantee:

  t* = largest t such that, on a held-out calibration set, the
       Clopper-Pearson upper (1-delta) confidence bound on
       P(regression | p_risk < t) is <= alpha.

Policy: among candidates certified safe (p_risk < t*), steer to the
one with highest p_win; if none certified, keep the default.
Guarantee (under exchangeability): at most an alpha fraction of
steered queries regress, with confidence 1-delta.

Evaluations:
  E1  stationary workload - random query-level train/cal/test splits,
      repeated over seeds. Exchangeability holds; the guarantee
      should verify empirically.
  E2  template shift - calibrate on 19 templates, test on the 20th.
      Exchangeability violated by construction; we MEASURE the
      degradation. Both results go in the paper.

Usage:
    py step5_conformal_gate.py --user postgres --password 12345
    py step5_conformal_gate.py --user postgres --password 12345 --alpha 0.1
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


def node_features(root):
    max_nl_inner = max_hash_build = max_sort_rows = max_self_cost = 0.0
    max_rows = float(root.get("Plan Rows", 0) or 0)
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
            max_nl_inner = max(max_nl_inner,
                               float(children[1].get("Plan Rows", 0) or 0))
        if nt == "Hash Join":
            for c in children:
                if c.get("Node Type") == "Hash":
                    h = c.get("Plans", [{}])
                    max_hash_build = max(max_hash_build,
                                         float((h[0] if h else {}).get("Plan Rows", 0) or 0))
        if nt in ("Sort", "Incremental Sort") and children:
            max_sort_rows = max(max_sort_rows,
                                float(children[0].get("Plan Rows", 0) or 0))
        for c in children:
            visit(c)

    visit(root)
    root_rows = max(float(root.get("Plan Rows", 0) or 0), 1.0)
    return {"log_max_nl_inner_rows": math.log1p(max_nl_inner),
            "log_max_hash_build_rows": math.log1p(max_hash_build),
            "log_max_sort_rows": math.log1p(max_sort_rows),
            "max_self_cost_share": (max_self_cost / total_cost) if total_cost > 0 else 0.0,
            "log_max_node_rows": math.log1p(max_rows),
            "log_blowup": math.log1p(max_rows / root_rows)}


def load_dataset(args):
    import psycopg2
    conn = psycopg2.connect(dbname=args.log_db, user=args.user,
                            password=args.password, host=args.host, port=args.port)
    df = pd.read_sql("""
        WITH timings AS (
            SELECT e.plan_id,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY e.exec_ms) AS median_ms,
                   bool_or(e.timed_out) AS censored
            FROM executions e WHERE NOT e.is_warmup GROUP BY e.plan_id
        )
        SELECT q.template, q.variant, q.query_id, h.name AS hint_set,
               p.plan_id, p.features, p.plan_json, t.median_ms, t.censored
        FROM timings t
        JOIN plans p ON p.plan_id = t.plan_id
        JOIN hint_sets h ON h.hint_set_id = p.hint_set_id
        JOIN queries q ON q.query_id = p.query_id
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

    base = ["cost_ratio", "log_cost_ratio", "rows_ratio", "log_rows_ratio",
            "nodes_ratio", "depth_delta", "nestloop_delta", "hashjoin_delta",
            "mergejoin_delta", "seqscan_delta", "indexscan_delta",
            "bitmapscan_delta", "sort_delta", "material_delta",
            "memoize_delta", "gather_delta",
            "nestloop_share", "hashjoin_share", "mergejoin_share"]
    for nf in NODE_FEATS:
        cand[f"{nf}_delta"] = cand[nf] - cand[f"def_{nf}"]
    node_block = NODE_FEATS + [f"{nf}_delta" for nf in NODE_FEATS] \
        + [f"def_{nf}" for nf in NODE_FEATS]
    hs = pd.get_dummies(cand["hint_set"], prefix="hs")
    cand = pd.concat([cand, hs], axis=1)
    features = base + node_block + list(hs.columns)
    return cand.reset_index(drop=True), features


def fit_heads(train, features):
    from sklearn.ensemble import GradientBoostingClassifier
    heads = {}
    for label in ["is_regression", "is_win"]:
        m = GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        m.fit(train[features], train[label])
        heads[label] = m
    return heads


def clopper_pearson_upper(k, n, delta):
    """Upper (1-delta) confidence bound on a binomial proportion."""
    from scipy.stats import beta
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def calibrate_threshold(cal_probs, cal_labels, alpha, delta):
    """Largest t with CP upper bound on P(regression | p_risk < t) <= alpha."""
    order = np.argsort(cal_probs)
    p_sorted = cal_probs[order]
    y_sorted = cal_labels[order]
    best_t = 0.0  # certify nothing by default
    cum_reg = np.cumsum(y_sorted)
    for i in range(len(p_sorted)):
        n = i + 1
        k = int(cum_reg[i])
        if clopper_pearson_upper(k, n, delta) <= alpha:
            best_t = float(p_sorted[i]) + 1e-12
    return best_t


def apply_policy(test, t_star, theta_reg):
    rows = []
    for qid, g in test.groupby("query_id"):
        d_ms = g["def_median_ms"].iloc[0]
        best_ms = min(d_ms, g["median_ms"].min())
        certified = g[g["p_risk"] < t_star]
        steered, regressed, chosen = False, False, d_ms
        if not certified.empty:
            top = certified.loc[certified["p_win"].idxmax()]
            steered = True
            chosen = top["median_ms"]
            regressed = chosen > theta_reg * d_ms
        rows.append(dict(default_ms=d_ms, oracle_ms=best_ms, policy_ms=chosen,
                         steered=steered, regressed=regressed,
                         template=g["template"].iloc[0]))
    return pd.DataFrame(rows)


def summarize(P):
    tot_def, tot_pol, tot_or = P.default_ms.sum(), P.policy_ms.sum(), P.oracle_ms.sum()
    gain = tot_def - tot_or
    n_st = int(P.steered.sum())
    n_rg = int(P.regressed.sum())
    return {"capture": ((tot_def - tot_pol) / gain * 100) if gain > 0 else 0.0,
            "speedup_pct": (tot_def - tot_pol) / tot_def * 100,
            "n_steer": n_st, "n_reg": n_rg,
            "reg_rate": (n_rg / n_st) if n_st else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--log-db", default="qppe")
    parser.add_argument("--theta-reg", type=float, default=1.2)
    parser.add_argument("--theta-win", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="target max regression rate among steered queries")
    parser.add_argument("--delta", type=float, default=0.10,
                        help="confidence parameter for the CP bound")
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    cand, features = load_dataset(args)
    cand["is_regression"] = ((cand["slowdown"] > args.theta_reg) |
                             cand["censored"]).astype(int)
    cand["is_win"] = ((cand["slowdown"] < args.theta_win) &
                      (~cand["censored"])).astype(int)

    print("QPPE Step 5 - Conformal Safety Gate")
    print("=" * 72)
    print(f"Samples: {len(cand)} | alpha = {args.alpha} | delta = {args.delta}")

    qids = cand["query_id"].unique()

    # ------------------------------------------------------------
    # E1: stationary workload (random query-level splits)
    # ------------------------------------------------------------
    print(f"\nE1 - STATIONARY WORKLOAD ({args.seeds} random splits, "
          f"queries 60/20/20 train/cal/test)")
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(qids)
        n = len(perm)
        tr_q = set(perm[:int(0.6 * n)])
        ca_q = set(perm[int(0.6 * n):int(0.8 * n)])
        te_q = set(perm[int(0.8 * n):])
        train = cand[cand.query_id.isin(tr_q)]
        cal = cand[cand.query_id.isin(ca_q)].copy()
        test = cand[cand.query_id.isin(te_q)].copy()

        heads = fit_heads(train, features)
        cal["p_risk"] = heads["is_regression"].predict_proba(cal[features])[:, 1]
        t_star = calibrate_threshold(cal["p_risk"].values,
                                     cal["is_regression"].values,
                                     args.alpha, args.delta)
        test["p_risk"] = heads["is_regression"].predict_proba(test[features])[:, 1]
        test["p_win"] = heads["is_win"].predict_proba(test[features])[:, 1]
        P = apply_policy(test, t_star, args.theta_reg)
        s = summarize(P)
        s["t_star"] = t_star
        rows.append(s)

    E1 = pd.DataFrame(rows)
    print(f"{'metric':<28}{'mean':>10}{'std':>9}{'min':>9}{'max':>9}")
    print("-" * 65)
    for col, name in [("t_star", "calibrated t*"),
                      ("n_steer", "steered queries (of ~18)"),
                      ("reg_rate", "realized regression rate"),
                      ("speedup_pct", "workload speedup %"),
                      ("capture", "oracle-gain capture %")]:
        print(f"{name:<28}{E1[col].mean():>10.3f}{E1[col].std():>9.3f}"
              f"{E1[col].min():>9.3f}{E1[col].max():>9.3f}")
    n_viol = int((E1["reg_rate"] > args.alpha).sum())
    print(f"\nSplits where realized regression rate exceeded alpha={args.alpha}: "
          f"{n_viol}/{args.seeds}")
    print("(the guarantee is on the CP upper bound, so occasional small-sample")
    print(" excursions are expected; systematic violation is not)")

    # ------------------------------------------------------------
    # E2: template shift (calibrate on 19 templates, test the 20th)
    # ------------------------------------------------------------
    print(f"\nE2 - TEMPLATE SHIFT (leave-one-template-out, "
          f"calibration from the 19 training templates)")
    print(f"{'held out':<10}{'t*':>7}{'steers':>8}{'regs':>6}{'reg rate':>10}"
          f"{'speedup %':>11}")
    print("-" * 55)
    agg_steer, agg_reg = 0, 0
    tot_def_all, tot_pol_all, tot_or_all = 0.0, 0.0, 0.0
    for held_out in sorted(cand["template"].unique()):
        rest = cand[cand.template != held_out]
        test = cand[cand.template == held_out].copy()
        rest_q = rest["query_id"].unique()
        rng = np.random.default_rng(0)
        perm = rng.permutation(rest_q)
        ca_q = set(perm[:int(0.25 * len(perm))])
        train = rest[~rest.query_id.isin(ca_q)]
        cal = rest[rest.query_id.isin(ca_q)].copy()

        heads = fit_heads(train, features)
        cal["p_risk"] = heads["is_regression"].predict_proba(cal[features])[:, 1]
        t_star = calibrate_threshold(cal["p_risk"].values,
                                     cal["is_regression"].values,
                                     args.alpha, args.delta)
        test["p_risk"] = heads["is_regression"].predict_proba(test[features])[:, 1]
        test["p_win"] = heads["is_win"].predict_proba(test[features])[:, 1]
        P = apply_policy(test, t_star, args.theta_reg)
        s = summarize(P)
        agg_steer += s["n_steer"]
        agg_reg += s["n_reg"]
        tot_def_all += P.default_ms.sum()
        tot_pol_all += P.policy_ms.sum()
        tot_or_all += P.oracle_ms.sum()
        print(f"{held_out:<10}{t_star:>7.3f}{s['n_steer']:>8}{s['n_reg']:>6}"
              f"{s['reg_rate']:>10.2f}{s['speedup_pct']:>11.1f}")

    rate = agg_reg / agg_steer if agg_steer else 0.0
    speed = (tot_def_all - tot_pol_all) / tot_def_all * 100
    gain = tot_def_all - tot_or_all
    cap = (tot_def_all - tot_pol_all) / gain * 100 if gain > 0 else 0.0
    print("-" * 55)
    print(f"AGGREGATE: {agg_steer} steers, {agg_reg} regressions "
          f"(rate {rate:.2f} vs alpha {args.alpha}) | "
          f"speedup {speed:.1f}% | capture {cap:.0f}%")

    print("\nInterpretation guide: E1 rate <= alpha validates the guarantee")
    print("under exchangeability; E2 rate quantifies degradation under")
    print("template shift. Both numbers are paper results.")

    with open("qppe_conformal_gate.pkl", "wb") as f:
        pickle.dump({"alpha": args.alpha, "delta": args.delta,
                     "features": features}, f)
    print("\nDone. Paste the full output back.")


if __name__ == "__main__":
    main()
