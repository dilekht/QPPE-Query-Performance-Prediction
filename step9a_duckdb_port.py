#!/usr/bin/env python3
"""
QPPE Rebuild - Step 9a: DuckDB Port (Portability Chapter, Part 1)
==================================================================
Ports the full QPPE pipeline to DuckDB with the same experimental
protocol as PostgreSQL:

  Phase 1  SETUP + CORPUS
    - builds TPC-H SF1 with DuckDB's built-in dbgen (no external data)
    - steering knobs discovered DYNAMICALLY from duckdb_optimizers():
      each hint set disables one (or two) optimizer passes via
      SET disabled_optimizers - the DuckDB analogue of enable_* GUCs
    - collects plans (EXPLAIN FORMAT JSON) for the SAME 105 query
      instances as the PostgreSQL corpus (imported from step3c)
    - executes distinct plans: warm-up + 3 runs, watchdog timeout
    - corpus persisted to qppe_duckdb_corpus.pkl (resumable)

  Phase 2  GATE + LIVE
    - identical labels (materially-severe: >2x AND >1s, or timeout)
    - policy-level cross-conformal calibration over all queries
    - live gated loop on the SAME 17 fresh variants as Step 6/7i

Adapter notes (the portability contribution):
  - DuckDB exposes no plan cost; est_cost := sum of node estimated
    cardinalities (documented proxy)
  - node-type mapping: HASH_JOIN->hashjoin, PIECEWISE_MERGE_JOIN->
    mergejoin, NESTED_LOOP/BLOCKWISE_NL/CROSS_PRODUCT->nestloop,
    TABLE/SEQ_SCAN->seqscan, ORDER_BY/TOP_N->sort, *GROUP_BY/
    UNGROUPED_AGGREGATE->agg

Usage:
    py -m pip install duckdb
    py step9a_duckdb_port.py            (both phases)
    py step9a_duckdb_port.py --phase 1  (corpus only)
Expected duration: phase 1 ~20-50 min, phase 2 ~5-10 min.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import pickle
import re
import statistics
import threading
import time

import numpy as np
import pandas as pd

RATIO_EPS = 1e-9
DB_FILE = "qppe_duckdb_tpch.duckdb"
CORPUS_FILE = "qppe_duckdb_corpus.pkl"

# optimizer passes we try to disable, in priority order (intersected
# with what duckdb_optimizers() actually reports in this version)
PREFERRED_KNOBS = [
    "join_order", "statistics_propagation", "filter_pushdown",
    "build_side_probe_side", "top_n", "in_clause", "filter_pullup",
    "join_filter_pushdown", "late_materialization",
    "compressed_materialization", "deliminator",
]
PAIR_KNOBS = [("join_order", "filter_pushdown"),
              ("statistics_propagation", "build_side_probe_side")]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, str(pathlib.Path(__file__).parent / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------- DuckDB plan adapter ----------------
def _extra(node):
    ei = node.get("extra_info", {})
    if isinstance(ei, str):
        return {"_raw": ei}
    return ei or {}


def _est_card(node):
    ei = _extra(node)
    v = ei.get("Estimated Cardinality")
    if v is None and "_raw" in ei:
        m = re.search(r"EC[:=]?\s*(\d+)", ei["_raw"])
        v = m.group(1) if m else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def walk(node, depth=0):
    yield node, depth
    for child in node.get("children", []):
        yield from walk(child, depth + 1)


NESTLOOP = {"NESTED_LOOP_JOIN", "BLOCKWISE_NL_JOIN", "CROSS_PRODUCT"}
HASHJOIN = {"HASH_JOIN"}
MERGEJOIN = {"PIECEWISE_MERGE_JOIN", "MERGE_JOIN"}
SEQSCAN = {"TABLE_SCAN", "SEQ_SCAN"}
SORT = {"ORDER_BY", "TOP_N"}
AGG = {"HASH_GROUP_BY", "PERFECT_HASH_GROUP_BY", "UNGROUPED_AGGREGATE",
       "AGGREGATE"}


def duck_features(root):
    counts, relations, max_depth = {}, set(), 0
    total_card, max_card = 0.0, 0.0
    max_nl_inner = 0.0
    for node, depth in walk(root):
        name = (node.get("name") or "?").strip().upper()
        counts[name] = counts.get(name, 0) + 1
        max_depth = max(max_depth, depth)
        card = _est_card(node)
        total_card += card
        max_card = max(max_card, card)
        ei = _extra(node)
        tbl = ei.get("Table")
        if tbl:
            relations.add(str(tbl))
        if name in NESTLOOP:
            ch = node.get("children", [])
            if len(ch) >= 2:
                max_nl_inner = max(max_nl_inner, _est_card(ch[1]))
    n_nl = sum(counts.get(k, 0) for k in NESTLOOP)
    n_hj = sum(counts.get(k, 0) for k in HASHJOIN)
    n_mj = sum(counts.get(k, 0) for k in MERGEJOIN)
    root_card = max(_est_card(root), 1.0)
    return {
        "est_cost": total_card,          # documented proxy
        "est_rows": _est_card(root),
        "num_nodes": sum(counts.values()),
        "max_depth": max_depth,
        "num_relations": len(relations),
        "num_joins": n_nl + n_hj + n_mj,
        "n_nestloop": n_nl, "n_hashjoin": n_hj, "n_mergejoin": n_mj,
        "n_seqscan": sum(counts.get(k, 0) for k in SEQSCAN),
        "n_indexscan": counts.get("INDEX_SCAN", 0),
        "n_sort": sum(counts.get(k, 0) for k in SORT),
        "n_agg": sum(counts.get(k, 0) for k in AGG),
        "n_filter": counts.get("FILTER", 0),
        "log_max_node_rows": math.log1p(max_card),
        "log_blowup": math.log1p(max_card / root_card),
        "log_max_nl_inner_rows": math.log1p(max_nl_inner),
    }


def duck_signature(node):
    name = (node.get("name") or "?").strip().upper()
    tbl = _extra(node).get("Table")
    return [[name, str(tbl) if tbl else None],
            [duck_signature(c) for c in node.get("children", [])]]


def duck_plan_hash(root):
    return hashlib.sha256(
        json.dumps(duck_signature(root), sort_keys=True).encode()
    ).hexdigest()[:16]


def get_plan_root(con, sql, disabled):
    con.execute(f"SET disabled_optimizers='{disabled}';")
    try:
        rows = con.execute(f"EXPLAIN (FORMAT json) {sql}").fetchall()
    finally:
        con.execute("SET disabled_optimizers='';")
    payload = rows[0][-1]
    obj = json.loads(payload)
    if isinstance(obj, list):
        obj = obj[0]
    return obj


def timed_run(con, sql, disabled, timeout_s):
    """Execute with watchdog interrupt; returns (ms, timed_out)."""
    con.execute(f"SET disabled_optimizers='{disabled}';")
    box = {"to": False}

    def kill():
        box["to"] = True
        try:
            con.interrupt()
        except Exception:
            pass

    timer = threading.Timer(timeout_s, kill)
    timer.start()
    t0 = time.perf_counter()
    try:
        con.execute(sql).fetchall()
        ms = (time.perf_counter() - t0) * 1000
        return ms, False
    except Exception:
        if box["to"]:
            return timeout_s * 1000, True
        raise
    finally:
        timer.cancel()
        con.execute("SET disabled_optimizers='';")


def measure(con, sql, disabled, timeout_s, runs=3):
    ms, to = timed_run(con, sql, disabled, timeout_s)  # warm-up
    if to:
        return timeout_s * 1000, True
    vals = []
    for _ in range(runs):
        ms, to = timed_run(con, sql, disabled, timeout_s)
        if to:
            return timeout_s * 1000, True
        vals.append(ms)
    return statistics.median(vals), False


# ---------------- feature rows (same construction as PG side) --------
def build_row(cand_f, def_f, hint_set, all_hint_names):
    r = {}
    r["cost_ratio"] = cand_f["est_cost"] / (def_f["est_cost"] + RATIO_EPS)
    r["log_cost_ratio"] = math.log(max(r["cost_ratio"], 1e-6))
    r["rows_ratio"] = (cand_f["est_rows"] + 1) / (def_f["est_rows"] + 1)
    r["log_rows_ratio"] = math.log(max(r["rows_ratio"], 1e-6))
    r["nodes_ratio"] = cand_f["num_nodes"] / (def_f["num_nodes"] + RATIO_EPS)
    r["depth_delta"] = cand_f["max_depth"] - def_f["max_depth"]
    for k in ["nestloop", "hashjoin", "mergejoin", "seqscan", "indexscan",
              "sort", "agg", "filter"]:
        r[f"{k}_delta"] = cand_f[f"n_{k}"] - def_f[f"n_{k}"]
    tj = max(cand_f["num_joins"], 1)
    r["nestloop_share"] = cand_f["n_nestloop"] / tj
    r["hashjoin_share"] = cand_f["n_hashjoin"] / tj
    r["mergejoin_share"] = cand_f["n_mergejoin"] / tj
    for nf in ["log_max_node_rows", "log_blowup", "log_max_nl_inner_rows"]:
        r[nf] = cand_f[nf]
        r[f"{nf}_delta"] = cand_f[nf] - def_f[nf]
        r[f"def_{nf}"] = def_f[nf]
    for hs in all_hint_names:
        if hs != "default":
            r[f"hs_{hs}"] = 1 if hs == hint_set else 0
    return r


def new_model():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=8, random_state=42)


def clopper_pearson_upper(k, n, delta):
    from scipy.stats import beta
    if n == 0 or k == n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--sf", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--theta-sev", type=float, default=2.0)
    parser.add_argument("--floor-ms", type=float, default=1000.0)
    parser.add_argument("--theta-win", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--tau-win", type=float, default=0.5)
    parser.add_argument("--timeout-cap-s", type=float, default=60.0)
    args = parser.parse_args()

    import duckdb

    s3c = load_module("s3c", "step3c_expand_corpus.py")
    s6 = load_module("s6", "step6_live_loop.py")

    print("QPPE Step 9a - DuckDB Port")
    print("=" * 72)
    print(f"DuckDB version: {duckdb.__version__}")

    con = duckdb.connect(DB_FILE)
    con.execute(f"SET threads={args.threads};")

    # ---------------- setup ----------------
    tables = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    if "lineitem" not in tables:
        print(f"Building TPC-H SF{args.sf} via dbgen extension...")
        con.execute("INSTALL tpch; LOAD tpch;")
        con.execute(f"CALL dbgen(sf={args.sf});")
    n_li = con.execute("SELECT count(*) FROM lineitem").fetchone()[0]
    print(f"lineitem rows: {n_li:,}")

    # ---------------- dynamic hint sets ----------------
    avail = {r[0] for r in con.execute(
        "SELECT name FROM duckdb_optimizers()").fetchall()}
    hint_sets = {"default": ""}
    for k in PREFERRED_KNOBS:
        if k in avail and len(hint_sets) < 11:
            hint_sets[f"no_{k}"] = k
    for a, b in PAIR_KNOBS:
        if a in avail and b in avail and len(hint_sets) < 13:
            hint_sets[f"no_{a}__{b}"] = f"{a},{b}"
    print(f"Hint sets ({len(hint_sets)}): {', '.join(hint_sets)}")

    # ---------------- phase 1: corpus ----------------
    if args.phase in (0, 1) and not pathlib.Path(CORPUS_FILE).exists():
        print("\nPHASE 1 - corpus collection + execution")
        instances = []
        for template, variants in s3c.PARAMS.items():
            for vi, params in enumerate(variants):
                instances.append((template, vi,
                                  s3c.TEMPLATES[template].format(**params)))
        print(f"Query instances: {len(instances)}")

        records = []
        n_err, n_to, t0 = 0, 0, time.time()
        for qi, (template, vi, sql) in enumerate(instances):
            plans = {}
            for hs_name, disabled in hint_sets.items():
                try:
                    root = get_plan_root(con, sql, disabled)
                    plans[hs_name] = root
                except Exception as e:
                    n_err += 1
                    if n_err <= 5:
                        print(f"  plan error {template}v{vi} [{hs_name}]: "
                              f"{str(e)[:80]}")
            if "default" not in plans:
                continue
            # dedup by structural hash, keep default first
            seen, todo = {}, []
            for hs_name, root in plans.items():
                h = duck_plan_hash(root)
                if h not in seen:
                    seen[h] = hs_name
                    todo.append((hs_name, root))
            # execute
            d_ms, d_to = measure(con, sql, hint_sets["default"],
                                 args.timeout_cap_s)
            timeout_s = min(max(3 * d_ms / 1000, 5.0), args.timeout_cap_s)
            for hs_name, root in todo:
                if hs_name == "default":
                    ms, to = d_ms, d_to
                else:
                    ms, to = measure(con, sql, hint_sets[hs_name], timeout_s)
                if to:
                    n_to += 1
                records.append(dict(
                    template=template, variant=vi,
                    query_id=f"{template}v{vi}", hint_set=hs_name,
                    features=duck_features(root), median_ms=ms, censored=to))
            if (qi + 1) % 15 == 0:
                el = time.time() - t0
                print(f"  {qi+1}/{len(instances)} queries "
                      f"({el/60:.0f} min elapsed)")
        with open(CORPUS_FILE, "wb") as f:
            pickle.dump({"records": records, "hint_sets": hint_sets}, f)
        print(f"Corpus saved: {len(records)} measured plans, "
              f"{n_to} timeouts, {n_err} plan errors, "
              f"{(time.time()-t0)/60:.1f} min")
    elif pathlib.Path(CORPUS_FILE).exists():
        print(f"\nCorpus file {CORPUS_FILE} found - skipping phase 1")

    if args.phase == 1:
        print("Phase 1 done. Rerun with --phase 2 (or no flag).")
        return

    # ---------------- phase 2: gate + live ----------------
    print("\nPHASE 2 - labels, calibration, live loop")
    with open(CORPUS_FILE, "rb") as f:
        blob = pickle.load(f)
    records, hint_sets = blob["records"], blob["hint_sets"]
    hint_names = list(hint_sets.keys())

    df = pd.DataFrame(records)
    defaults = (df[df.hint_set == "default"]
                .set_index("query_id")[["median_ms", "features"]]
                .rename(columns={"median_ms": "def_ms",
                                 "features": "def_features"}))
    cand = df[df.hint_set != "default"].join(defaults, on="query_id")
    rows = []
    for _, r in cand.iterrows():
        row = build_row(r["features"], r["def_features"], r["hint_set"],
                        hint_names)
        row.update(template=r["template"], query_id=r["query_id"],
                   median_ms=r["median_ms"], def_median_ms=r["def_ms"],
                   censored=bool(r["censored"]))
        rows.append(row)
    cand = pd.DataFrame(rows)
    slow = cand["median_ms"] / (cand["def_median_ms"] + RATIO_EPS)
    added = cand["median_ms"] - cand["def_median_ms"]
    cand["is_severe"] = (((slow > args.theta_sev) &
                          (added > args.floor_ms)) |
                         cand["censored"]).astype(int)
    cand["is_win"] = ((slow < args.theta_win) & ~cand["censored"]).astype(int)
    feats = [c for c in cand.columns
             if c not in ("template", "query_id", "median_ms",
                          "def_median_ms", "censored", "is_severe", "is_win")]
    print(f"Candidates: {len(cand)} | severe: {int(cand.is_severe.sum())} | "
          f"wins: {int(cand.is_win.sum())} | queries: "
          f"{cand['query_id'].nunique()}")

    # oracle bound
    td, to_ = 0.0, 0.0
    for qid, g in cand.groupby("query_id"):
        d = g["def_median_ms"].iloc[0]
        td += d
        to_ += min(d, g["median_ms"].min())
    print(f"Workload bounds: default {td/1000:.1f}s | oracle {to_/1000:.1f}s "
          f"(possible improvement {(td-to_)/td:.0%})")

    # cross-conformal policy-level calibration
    rng = np.random.default_rng(11)
    qids = rng.permutation(cand["query_id"].unique())
    folds = np.array_split(qids, 5)
    cand = cand.reset_index(drop=True)
    cand["p_severe"] = np.nan
    cand["p_win"] = np.nan
    for fold_q in folds:
        te = cand.query_id.isin(set(fold_q))
        ms_ = new_model()
        ms_.fit(cand.loc[~te, feats], cand.loc[~te, "is_severe"])
        cand.loc[te, "p_severe"] = ms_.predict_proba(cand.loc[te, feats])[:, 1]
        mw_ = new_model()
        mw_.fit(cand.loc[~te, feats], cand.loc[~te, "is_win"])
        cand.loc[te, "p_win"] = mw_.predict_proba(cand.loc[te, feats])[:, 1]

    grid = np.unique(np.quantile(cand["p_severe"].values,
                                 np.linspace(0.05, 0.95, 30)))
    t_star, bn, bk = 0.0, 0, 0
    for t in grid:
        n, k = 0, 0
        for qid, g in cand.groupby("query_id"):
            cert = g[(g["p_severe"] < t) & (g["p_win"] > args.tau_win)]
            if cert.empty:
                continue
            top = cert.loc[cert["p_win"].idxmax()]
            n += 1
            s = top["median_ms"] / (top["def_median_ms"] + RATIO_EPS)
            if ((s > args.theta_sev and
                 top["median_ms"] - top["def_median_ms"] > args.floor_ms)
                    or top["censored"]):
                k += 1
        if n > 0 and clopper_pearson_upper(k, n, args.delta) <= args.alpha:
            t_star, bn, bk = float(t), n, k
    print(f"Cross-conformal t* = {t_star:.3f} "
          f"(calibration: {bn} steered, {bk} severe)")

    heads_s = new_model()
    heads_s.fit(cand[feats], cand["is_severe"])
    heads_w = new_model()
    heads_w.fit(cand[feats], cand["is_win"])

    # live loop on the same 17 fresh variants
    print(f"\nLIVE - DuckDB fresh ({len(s6.FRESH)} queries, t*={t_star:.3f})")
    print(f"{'query':<10}{'decision':<28}{'default ms':>11}{'chosen ms':>10}"
          f"{'ovh ms':>8}{'result':>12}")
    print("-" * 81)
    results = []
    for template, variant, params in s6.FRESH:
        sql = s3c.TEMPLATES[template].format(**params)
        t0 = time.perf_counter()
        plans = {}
        try:
            for hs_name, disabled in hint_sets.items():
                plans[hs_name] = get_plan_root(con, sql, disabled)
        except Exception as e:
            print(f"{template}v{variant:<4} planning failed: {str(e)[:60]}")
            continue
        gen_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        def_f = duck_features(plans["default"])
        frows, names = [], []
        for hs_name, root in plans.items():
            if hs_name == "default":
                continue
            frows.append(build_row(duck_features(root), def_f, hs_name,
                                   hint_names))
            names.append(hs_name)
        F = pd.DataFrame(frows)[feats]
        p_sev = heads_s.predict_proba(F)[:, 1]
        p_win = heads_w.predict_proba(F)[:, 1]
        inf_ms = (time.perf_counter() - t0) * 1000

        cert = [(names[i], p_sev[i], p_win[i]) for i in range(len(names))
                if p_sev[i] < t_star and p_win[i] > args.tau_win]
        chosen = max(cert, key=lambda x: x[2])[0] if cert else "default"
        decision = f"steer:{chosen}" if cert else "keep default"

        d_ms, _ = measure(con, sql, hint_sets["default"], args.timeout_cap_s)
        if chosen == "default":
            c_ms = d_ms
        else:
            c_ms, _ = measure(con, sql, hint_sets[chosen],
                              min(max(3 * d_ms / 1000, 5.0),
                                  args.timeout_cap_s))
        slow = c_ms / d_ms if d_ms > 0 else 1.0
        severe = (chosen != "default" and slow > args.theta_sev
                  and c_ms - d_ms > args.floor_ms)
        result = ("-" if chosen == "default" else
                  "SEVERE" if severe else
                  "mild regr" if slow > 1.2 else
                  f"won {d_ms/c_ms:.1f}x" if c_ms < d_ms else "~neutral")
        results.append(dict(default_ms=d_ms, chosen_ms=c_ms, gen_ms=gen_ms,
                            inf_ms=inf_ms, steered=chosen != "default",
                            severe=severe))
        print(f"{template + 'v' + str(variant):<10}{decision:<28}"
              f"{d_ms:>11.0f}{c_ms:>10.0f}{gen_ms + inf_ms:>8.1f}"
              f"{result:>12}")

    R = pd.DataFrame(results)
    tot_def = R.default_ms.sum()
    tot_pol = R.chosen_ms.sum() + R.gen_ms.sum() + R.inf_ms.sum()
    n_st, n_sv = int(R.steered.sum()), int(R.severe.sum())
    print("-" * 81)
    print(f"DuckDB LIVE: steered {n_st}/{len(R)} | SEVERE {n_sv} "
          f"(rate {n_sv/max(n_st,1):.2f} vs alpha {args.alpha}) | "
          f"workload {tot_def/1000:.2f}s -> {tot_pol/1000:.2f}s "
          f"({(tot_def-tot_pol)/tot_def*100:+.1f}%, overhead incl.) | "
          f"overhead {R.gen_ms.mean()+R.inf_ms.mean():.1f} ms avg")
    print("\nDone. Paste the full output back.")


if __name__ == "__main__":
    main()
