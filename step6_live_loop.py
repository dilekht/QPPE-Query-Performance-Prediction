#!/usr/bin/env python3
"""
QPPE Rebuild - Step 6: Live Gated Steering (End-to-End)
========================================================
Everything so far was simulation over logged timings. This step runs
the REAL loop on FRESH query variants never seen by any component:

  per fresh query:
    1. generate candidates: 12x EXPLAIN (FORMAT JSON)   [timed]
    2. featurize + score with the two heads              [timed]
    3. conformal gate: steer only if a candidate is
       certified (p_risk < t*) - t* calibrated on a
       25% query split of the existing corpus
    4. execute the CHOSEN plan and the DEFAULT plan
       (1 warm-up + 3 measured runs each)
    5. log everything to the qppe database

Reports: realized speedup, regression count vs alpha, and the honest
decision overhead (candidate generation + inference), absolute and
relative to execution time.

Usage:
    py step6_live_loop.py --user postgres --password 12345
Expected duration: 15-30 minutes.
"""

import argparse
import hashlib
import json
import math
import pickle
import statistics
import time

import numpy as np
import pandas as pd

RATIO_EPS = 1e-9

HINT_SETS = {
    "default":            {},
    "no_nestloop":        {"enable_nestloop": "off"},
    "no_hashjoin":        {"enable_hashjoin": "off"},
    "no_mergejoin":       {"enable_mergejoin": "off"},
    "hash_only":          {"enable_nestloop": "off", "enable_mergejoin": "off"},
    "nestloop_only":      {"enable_hashjoin": "off", "enable_mergejoin": "off"},
    "merge_only":         {"enable_hashjoin": "off", "enable_nestloop": "off"},
    "no_seqscan":         {"enable_seqscan": "off"},
    "seqscan_only":       {"enable_indexscan": "off", "enable_indexonlyscan": "off",
                           "enable_bitmapscan": "off"},
    "no_nestloop_no_seq": {"enable_nestloop": "off", "enable_seqscan": "off"},
    "no_hash_no_index":   {"enable_hashjoin": "off", "enable_indexscan": "off",
                           "enable_indexonlyscan": "off"},
    "no_material_memoize": {"enable_material": "off", "enable_memoize": "off"},
}

# ---- fresh, never-before-used parameterizations (variant >= 100) ----
FRESH = [
    ("Q1",  100, {"days": 45}),
    ("Q5",  100, {"region": "EUROPE", "year": 1997}),
    ("Q5",  101, {"region": "ASIA", "year": 1993}),
    ("Q6",  100, {"year": 1994, "disc": 0.05, "qty": 28}),
    ("Q7",  100, {"nation1": "JAPAN", "nation2": "INDIA"}),
    ("Q7",  101, {"nation1": "UNITED KINGDOM", "nation2": "RUSSIA"}),
    ("Q8",  100, {"nation": "JAPAN", "region": "ASIA",
                  "ptype": "ECONOMY POLISHED STEEL"}),
    ("Q9",  100, {"color": "brown"}),
    ("Q10", 100, {"qstart": "1993-04-01"}),
    ("Q12", 100, {"mode1": "SHIP", "mode2": "FOB", "year": 1995}),
    ("Q13", 100, {"w1": "express", "w2": "accounts"}),
    ("Q14", 100, {"month": "1994-12"}),
    ("Q17", 100, {"brand": "Brand#52", "container": "LG CAN"}),
    ("Q18", 100, {"qty": 302}),
    ("Q19", 100, {"b1": "Brand#41", "q1": 7, "b2": "Brand#24",
                  "q2": 16, "b3": "Brand#55", "q3": 26}),
    ("Q21", 100, {"nation": "GERMANY"}),
    ("Q22", 100, {"codes": "'26','27','28','32','33','34','35'"}),
]

RAW = ["est_cost", "est_rows", "num_nodes", "max_depth", "num_relations",
       "num_joins", "n_nestloop", "n_hashjoin", "n_mergejoin",
       "n_seqscan", "n_indexscan", "n_bitmapscan", "n_sort", "n_agg",
       "n_gather", "n_parallel_nodes", "n_memoize", "n_material"]
NODE_FEATS = ["log_max_nl_inner_rows", "log_max_hash_build_rows",
              "log_max_sort_rows", "max_self_cost_share",
              "log_max_node_rows", "log_blowup"]
JOIN_NODES = {"Nested Loop", "Hash Join", "Merge Join"}


# ---------------- plan analysis (same as steps 3/4/5) ----------------
def walk(node, depth=0):
    yield node, depth
    for child in node.get("Plans", []):
        yield from walk(child, depth + 1)


def flat_features(root):
    f = {"est_cost": root.get("Total Cost", 0.0),
         "est_rows": root.get("Plan Rows", 0)}
    counts, relations, max_depth, n_parallel = {}, set(), 0, 0
    for node, depth in walk(root):
        nt = node.get("Node Type", "?")
        counts[nt] = counts.get(nt, 0) + 1
        max_depth = max(max_depth, depth)
        if node.get("Relation Name"):
            relations.add(node["Relation Name"])
        if node.get("Parallel Aware"):
            n_parallel += 1
    f["num_nodes"] = sum(counts.values())
    f["max_depth"] = max_depth
    f["num_relations"] = len(relations)
    f["num_joins"] = sum(counts.get(j, 0) for j in JOIN_NODES)
    f["n_nestloop"] = counts.get("Nested Loop", 0)
    f["n_hashjoin"] = counts.get("Hash Join", 0)
    f["n_mergejoin"] = counts.get("Merge Join", 0)
    f["n_seqscan"] = counts.get("Seq Scan", 0)
    f["n_indexscan"] = counts.get("Index Scan", 0) + counts.get("Index Only Scan", 0)
    f["n_bitmapscan"] = counts.get("Bitmap Heap Scan", 0)
    f["n_sort"] = counts.get("Sort", 0) + counts.get("Incremental Sort", 0)
    f["n_agg"] = counts.get("Aggregate", 0)
    f["n_gather"] = counts.get("Gather", 0) + counts.get("Gather Merge", 0)
    f["n_parallel_nodes"] = n_parallel
    f["n_memoize"] = counts.get("Memoize", 0)
    f["n_material"] = counts.get("Materialize", 0)
    return f


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


def build_feature_row(cand_flat, cand_node, def_flat, def_node, hint_set):
    r = {}
    r["cost_ratio"] = cand_flat["est_cost"] / (def_flat["est_cost"] + RATIO_EPS)
    r["log_cost_ratio"] = math.log(max(r["cost_ratio"], 1e-6))
    r["rows_ratio"] = (cand_flat["est_rows"] + 1) / (def_flat["est_rows"] + 1)
    r["log_rows_ratio"] = math.log(max(r["rows_ratio"], 1e-6))
    r["nodes_ratio"] = cand_flat["num_nodes"] / (def_flat["num_nodes"] + RATIO_EPS)
    r["depth_delta"] = cand_flat["max_depth"] - def_flat["max_depth"]
    for k in ["nestloop", "hashjoin", "mergejoin"]:
        r[f"{k}_delta"] = cand_flat[f"n_{k}"] - def_flat[f"n_{k}"]
    r["seqscan_delta"] = cand_flat["n_seqscan"] - def_flat["n_seqscan"]
    r["indexscan_delta"] = cand_flat["n_indexscan"] - def_flat["n_indexscan"]
    r["bitmapscan_delta"] = cand_flat["n_bitmapscan"] - def_flat["n_bitmapscan"]
    r["sort_delta"] = cand_flat["n_sort"] - def_flat["n_sort"]
    r["material_delta"] = cand_flat["n_material"] - def_flat["n_material"]
    r["memoize_delta"] = cand_flat["n_memoize"] - def_flat["n_memoize"]
    r["gather_delta"] = cand_flat["n_gather"] - def_flat["n_gather"]
    tj = max(cand_flat["num_joins"], 1)
    r["nestloop_share"] = cand_flat["n_nestloop"] / tj
    r["hashjoin_share"] = cand_flat["n_hashjoin"] / tj
    r["mergejoin_share"] = cand_flat["n_mergejoin"] / tj
    for nf in NODE_FEATS:
        r[nf] = cand_node[nf]
        r[f"{nf}_delta"] = cand_node[nf] - def_node[nf]
        r[f"def_{nf}"] = def_node[nf]
    for hs in HINT_SETS:
        if hs != "default":
            r[f"hs_{hs}"] = 1 if hs == hint_set else 0
    return r


# ---------------- training-side (reuses Step 5 corpus) ----------------
def load_corpus(args):
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
        WHERE q.variant < 100
        ORDER BY q.query_id, p.plan_id;
    """, conn)
    conn.close()

    flatd = [flat_features(pj[0]["Plan"]) for pj in df["plan_json"]]
    noded = [node_features(pj[0]["Plan"]) for pj in df["plan_json"]]
    rows = []
    by_qid = {}
    for i, rec in df.iterrows():
        by_qid.setdefault(rec["query_id"], {})[rec["hint_set"]] = i
    for qid, hmap in by_qid.items():
        if "default" not in hmap:
            continue
        di = hmap["default"]
        d_ms = df.loc[di, "median_ms"]
        for hs, i in hmap.items():
            if hs == "default":
                continue
            r = build_feature_row(flatd[i], noded[i], flatd[di], noded[di], hs)
            r["template"] = df.loc[i, "template"]
            r["query_id"] = qid
            r["median_ms"] = df.loc[i, "median_ms"]
            r["def_median_ms"] = d_ms
            r["censored"] = bool(df.loc[i, "censored"])
            rows.append(r)
    return pd.DataFrame(rows)


def clopper_pearson_upper(k, n, delta):
    from scipy.stats import beta
    if n == 0 or k == n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def calibrate_threshold(p, y, alpha, delta):
    order = np.argsort(p)
    ps, ys = p[order], y[order]
    cum = np.cumsum(ys)
    best = 0.0
    for i in range(len(ps)):
        if clopper_pearson_upper(int(cum[i]), i + 1, delta) <= alpha:
            best = float(ps[i]) + 1e-12
    return best


def run_plan_once(bcur, sql, gucs, timeout_ms):
    bcur.execute("BEGIN;")
    try:
        bcur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)};")
        for guc, val in gucs.items():
            bcur.execute(f"SET LOCAL {guc} = {val};")
        try:
            bcur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) {sql};")
            out = bcur.fetchone()[0]
            if isinstance(out, str):
                out = json.loads(out)
            return out[0].get("Execution Time"), False
        except Exception as e:
            if "statement timeout" in str(e).lower() or "canceling" in str(e).lower():
                return float(timeout_ms), True
            raise
    finally:
        try:
            bcur.execute("ROLLBACK;")
        except Exception:
            pass


def measure(bcur, sql, gucs, timeout_ms, runs=3):
    ems, to = run_plan_once(bcur, sql, gucs, timeout_ms)  # warm-up
    if to:
        return float(timeout_ms), True
    times = []
    for _ in range(runs):
        ems, to = run_plan_once(bcur, sql, gucs, timeout_ms)
        if to:
            return float(timeout_ms), True
        times.append(ems)
    return statistics.median(times), False


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
    args = parser.parse_args()

    import psycopg2
    from sklearn.ensemble import GradientBoostingClassifier

    # templates text: reuse the definitions from step3c
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "s3c", str(pathlib.Path(__file__).parent / "step3c_expand_corpus.py"))
    s3c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s3c)
    TEMPLATES = s3c.TEMPLATES

    print("QPPE Step 6 - Live Gated Steering")
    print("=" * 72)

    # ---- train heads + calibrate on existing corpus ----
    corpus = load_corpus(args)
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
    rng = np.random.default_rng(7)
    qids = corpus["query_id"].unique()
    perm = rng.permutation(qids)
    cal_q = set(perm[:int(0.25 * len(perm))])
    train = corpus[~corpus.query_id.isin(cal_q)]
    cal = corpus[corpus.query_id.isin(cal_q)].copy()

    heads = {}
    for label in ["is_regression", "is_win"]:
        m = GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        m.fit(train[feature_cols], train[label])
        heads[label] = m
    cal_p = heads["is_regression"].predict_proba(cal[feature_cols])[:, 1]
    t_star = calibrate_threshold(cal_p, cal["is_regression"].values,
                                 args.alpha, args.delta)
    print(f"Trained on {len(train)} samples, calibrated on {len(cal)} "
          f"-> t* = {t_star:.3f} (alpha={args.alpha}, delta={args.delta})")

    # ---- live loop ----
    bench = psycopg2.connect(dbname=args.bench_db, user=args.user,
                             password=args.password, host=args.host, port=args.port)
    bench.autocommit = True
    bcur = bench.cursor()

    results = []
    print(f"\n{'query':<9}{'decision':<22}{'p_risk':>7}{'default ms':>11}"
          f"{'chosen ms':>10}{'overhead ms':>12}{'result':>11}")
    print("-" * 84)
    for template, variant, params in FRESH:
        sql = TEMPLATES[template].format(**params)

        # 1. candidate generation (timed)
        t0 = time.perf_counter()
        plans = {}
        for hs_name, gucs in HINT_SETS.items():
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

        # 2. featurize + score (timed)
        t0 = time.perf_counter()
        def_flat = flat_features(plans["default"])
        def_node = node_features(plans["default"])
        rows, names = [], []
        for hs_name, root in plans.items():
            if hs_name == "default":
                continue
            rows.append(build_feature_row(flat_features(root), node_features(root),
                                          def_flat, def_node, hs_name))
            names.append(hs_name)
        F = pd.DataFrame(rows)[feature_cols]
        p_risk = heads["is_regression"].predict_proba(F)[:, 1]
        p_win = heads["is_win"].predict_proba(F)[:, 1]
        inf_ms = (time.perf_counter() - t0) * 1000

        # 3. gate
        certified = [(names[i], p_risk[i], p_win[i])
                     for i in range(len(names)) if p_risk[i] < t_star]
        if certified:
            chosen_hs, chosen_risk, _ = max(certified, key=lambda x: x[2])
            decision = f"steer:{chosen_hs}"
        else:
            chosen_hs, chosen_risk = "default", float("nan")
            decision = "keep default"

        # 4. execute default and chosen
        d_ms, d_to = measure(bcur, sql, {}, 120000)
        timeout = min(max(3 * d_ms, 10000), 120000)
        if chosen_hs == "default":
            c_ms, c_to = d_ms, d_to
        else:
            c_ms, c_to = measure(bcur, sql, HINT_SETS[chosen_hs], timeout)

        regressed = c_ms > args.theta_reg * d_ms
        if chosen_hs == "default":
            result = "-"
        elif regressed:
            result = "REGRESSED"
        elif c_ms < d_ms:
            result = f"won {d_ms/c_ms:.1f}x"
        else:
            result = "~neutral"
        overhead = gen_ms + inf_ms
        results.append(dict(query=f"{template}v{variant}", decision=decision,
                            default_ms=d_ms, chosen_ms=c_ms, gen_ms=gen_ms,
                            inf_ms=inf_ms, steered=chosen_hs != "default",
                            regressed=regressed and chosen_hs != "default"))
        print(f"{template + 'v' + str(variant):<9}{decision:<22}"
              f"{chosen_risk:>7.3f}{d_ms:>11.0f}{c_ms:>10.0f}"
              f"{overhead:>12.1f}{result:>11}")

    R = pd.DataFrame(results)
    tot_def = R.default_ms.sum()
    tot_pol = R.chosen_ms.sum() + R.gen_ms.sum() + R.inf_ms.sum()
    n_st, n_rg = int(R.steered.sum()), int(R.regressed.sum())
    print("-" * 84)
    print(f"\nLIVE RESULTS ({len(R)} fresh queries)")
    print(f"  steered: {n_st} | regressions: {n_rg} "
          f"(rate {n_rg/max(n_st,1):.2f} vs alpha {args.alpha})")
    print(f"  workload: default {tot_def/1000:.1f}s -> gated "
          f"{tot_pol/1000:.1f}s ({(tot_def-tot_pol)/tot_def*100:+.1f}%, "
          f"overhead included)")
    print(f"  decision overhead: candidate gen {R.gen_ms.mean():.1f} ms avg, "
          f"inference {R.inf_ms.mean():.1f} ms avg, "
          f"total {R.gen_ms.mean()+R.inf_ms.mean():.1f} ms avg "
          f"({(R.gen_ms.sum()+R.inf_ms.sum())/tot_def*100:.2f}% of default "
          f"workload time)")
    print("\nPaste the full output back.")


if __name__ == "__main__":
    main()
