#!/usr/bin/env python3
"""
QPPE Rebuild - Step 9b: MySQL Port (Portability Chapter, Part 2)
=================================================================
Same protocol, third engine.

  Phase 1  SETUP + CORPUS
    - connects to MySQL 8, raises innodb_buffer_pool_size to 4GB
      ONLINE (SET PERSIST - no restart needed), enables local_infile
    - loads TPC-H SF1 by EXPORTING from your existing PostgreSQL
      tpch_sf1 database (COPY -> CSV -> LOAD DATA LOCAL INFILE);
      documented index set (PKs + FK indexes)
    - knob layer: SET SESSION optimizer_switch flags discovered
      dynamically from @@optimizer_switch, plus two join-search knobs
      (optimizer_prune_level / optimizer_search_depth)
    - SQL dialect adapter: interval-literal translation (the only
      known incompatibility); every instance EXPLAIN-validated
    - corpus: same 89 instances, plan dedup, warm-up + 3 runs,
      timeouts via max_execution_time; CHECKPOINTED every 5 queries
      to qppe_mysql_corpus.pkl (Ctrl+C safe, resumable)

  Phase 2  GATE + LIVE
    - identical labels, policy-level cross-conformal calibration,
      live loop on the same 17 fresh variants

Adapter notes (for the paper):
  - est_cost := query_block cost_info.query_cost (MySQL native)
  - access_type ALL -> seqscan; ref/eq_ref/range/index -> indexscan
  - "using_join_buffer": "hash join" -> hashjoin;
    "Block Nested Loop" -> nestloop; default join nesting -> nestloop
  - plan hash from (table, access_type, key, join_buffer) tree

Usage:
    py -m pip install pymysql
    py step9b_mysql_port.py --password <MYSQL_ROOT_PW> --pg-password 12345
    py step9b_mysql_port.py --password <PW> --pg-password 12345 --phase 1
Expected duration: data load ~10-20 min (first run only);
corpus 1.5-4 h (MySQL is slow on TPC-H - run overnight, resumable);
phase 2 ~10 min.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import pickle
import re
import statistics
import tempfile
import time

import numpy as np
import pandas as pd

RATIO_EPS = 1e-9
CORPUS_FILE = "qppe_mysql_corpus.pkl"

PREFERRED_SWITCHES = [
    "block_nested_loop", "batched_key_access", "mrr", "mrr_cost_based",
    "index_merge", "semijoin", "materialization",
    "subquery_materialization_cost_based", "derived_merge",
    "condition_fanout_filter", "prefer_ordering_index",
    "index_condition_pushdown",
]

TABLES = {
    "region": """CREATE TABLE region (
        r_regionkey INT NOT NULL, r_name CHAR(25), r_comment VARCHAR(152),
        PRIMARY KEY (r_regionkey)) ENGINE=InnoDB""",
    "nation": """CREATE TABLE nation (
        n_nationkey INT NOT NULL, n_name CHAR(25), n_regionkey INT,
        n_comment VARCHAR(152), PRIMARY KEY (n_nationkey),
        KEY idx_n_regionkey (n_regionkey)) ENGINE=InnoDB""",
    "supplier": """CREATE TABLE supplier (
        s_suppkey INT NOT NULL, s_name CHAR(25), s_address VARCHAR(40),
        s_nationkey INT, s_phone CHAR(15), s_acctbal DECIMAL(15,2),
        s_comment VARCHAR(101), PRIMARY KEY (s_suppkey),
        KEY idx_s_nationkey (s_nationkey)) ENGINE=InnoDB""",
    "customer": """CREATE TABLE customer (
        c_custkey INT NOT NULL, c_name VARCHAR(25), c_address VARCHAR(40),
        c_nationkey INT, c_phone CHAR(15), c_acctbal DECIMAL(15,2),
        c_mktsegment CHAR(10), c_comment VARCHAR(117),
        PRIMARY KEY (c_custkey),
        KEY idx_c_nationkey (c_nationkey)) ENGINE=InnoDB""",
    "part": """CREATE TABLE part (
        p_partkey INT NOT NULL, p_name VARCHAR(55), p_mfgr CHAR(25),
        p_brand CHAR(10), p_type VARCHAR(25), p_size INT,
        p_container CHAR(10), p_retailprice DECIMAL(15,2),
        p_comment VARCHAR(23), PRIMARY KEY (p_partkey)) ENGINE=InnoDB""",
    "partsupp": """CREATE TABLE partsupp (
        ps_partkey INT NOT NULL, ps_suppkey INT NOT NULL,
        ps_availqty INT, ps_supplycost DECIMAL(15,2),
        ps_comment VARCHAR(199),
        PRIMARY KEY (ps_partkey, ps_suppkey),
        KEY idx_ps_suppkey (ps_suppkey)) ENGINE=InnoDB""",
    "orders": """CREATE TABLE orders (
        o_orderkey INT NOT NULL, o_custkey INT, o_orderstatus CHAR(1),
        o_totalprice DECIMAL(15,2), o_orderdate DATE,
        o_orderpriority CHAR(15), o_clerk CHAR(15), o_shippriority INT,
        o_comment VARCHAR(79), PRIMARY KEY (o_orderkey),
        KEY idx_o_custkey (o_custkey),
        KEY idx_o_orderdate (o_orderdate)) ENGINE=InnoDB""",
    "lineitem": """CREATE TABLE lineitem (
        l_orderkey INT NOT NULL, l_partkey INT, l_suppkey INT,
        l_linenumber INT NOT NULL, l_quantity DECIMAL(15,2),
        l_extendedprice DECIMAL(15,2), l_discount DECIMAL(15,2),
        l_tax DECIMAL(15,2), l_returnflag CHAR(1), l_linestatus CHAR(1),
        l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE,
        l_shipinstruct CHAR(25), l_shipmode CHAR(10),
        l_comment VARCHAR(44),
        PRIMARY KEY (l_orderkey, l_linenumber),
        KEY idx_l_partkey (l_partkey),
        KEY idx_l_suppkey (l_suppkey),
        KEY idx_l_shipdate (l_shipdate)) ENGINE=InnoDB""",
}
LOAD_ORDER = ["region", "nation", "supplier", "customer", "part",
              "partsupp", "orders", "lineitem"]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, str(pathlib.Path(__file__).parent / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def translate_sql(sql):
    """PG -> MySQL dialect. Known delta: interval string literals."""
    def repl(m):
        n, unit = m.group(1), m.group(2).upper().rstrip("S")
        return f"INTERVAL {n} {unit}"
    sql = re.sub(r"interval\s+'(\d+)\s+(years?|months?|days?)'",
                 repl, sql, flags=re.IGNORECASE)
    return sql


# ---------------- MySQL plan adapter ----------------
def _walk_json(obj, depth=0):
    if isinstance(obj, dict):
        yield obj, depth
        for v in obj.values():
            yield from _walk_json(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v, depth)


def mysql_features(plan):
    qb = plan.get("query_block", plan)
    cost = 0.0
    try:
        cost = float(qb.get("cost_info", {}).get("query_cost", 0))
    except (TypeError, ValueError):
        pass
    n_tables = n_full = n_index = n_hash = n_bnl = 0
    n_sort = n_agg = n_mat = n_sub = 0
    max_rows = 0.0
    total_rows = 0.0
    max_depth = 0
    relations = set()
    for node, depth in _walk_json(plan):
        max_depth = max(max_depth, depth)
        if "table_name" in node:
            n_tables += 1
            relations.add(node["table_name"])
            at = node.get("access_type", "")
            if at == "ALL":
                n_full += 1
            elif at in ("ref", "eq_ref", "range", "index", "const",
                        "fulltext", "index_merge", "unique_subquery"):
                n_index += 1
            jb = str(node.get("using_join_buffer", "")).lower()
            if "hash" in jb:
                n_hash += 1
            elif "nested" in jb or "bnl" in jb:
                n_bnl += 1
            try:
                r = float(node.get("rows_examined_per_scan", 0) or 0)
                max_rows = max(max_rows, r)
                total_rows += r
            except (TypeError, ValueError):
                pass
        if "ordering_operation" in node:
            n_sort += 1
        if "grouping_operation" in node:
            n_agg += 1
        if "materialized_from_subquery" in node:
            n_mat += 1
        if "attached_subqueries" in node or "subqueries" in node:
            n_sub += 1
    n_joins = max(n_tables - 1, 0)
    root_rows = 1.0
    for node, _ in _walk_json(plan):
        if "rows_produced_per_join" in node:
            try:
                root_rows = max(root_rows,
                                float(node["rows_produced_per_join"]))
            except (TypeError, ValueError):
                pass
            break
    return {
        "est_cost": cost,
        "est_rows": root_rows,
        "num_nodes": n_tables + n_sort + n_agg + n_mat,
        "max_depth": max_depth,
        "num_relations": len(relations),
        "num_joins": n_joins,
        "n_nestloop": max(n_joins - n_hash, 0),   # MySQL default = NL
        "n_hashjoin": n_hash,
        "n_mergejoin": 0,                          # MySQL has none
        "n_seqscan": n_full,
        "n_indexscan": n_index,
        "n_sort": n_sort,
        "n_agg": n_agg,
        "n_filter": n_sub,
        "log_max_node_rows": math.log1p(max_rows),
        "log_blowup": math.log1p(max_rows / max(root_rows, 1.0)),
        "log_max_nl_inner_rows": math.log1p(total_rows),
    }


def mysql_signature(obj):
    if isinstance(obj, dict):
        sig = []
        if "table_name" in obj:
            sig = [obj.get("table_name"), obj.get("access_type"),
                   obj.get("key"), str(obj.get("using_join_buffer", ""))]
        children = [mysql_signature(v) for k, v in sorted(obj.items())
                    if isinstance(v, (dict, list))]
        return [sig, [c for c in children if c != [[], []]]]
    if isinstance(obj, list):
        return [[], [mysql_signature(v) for v in obj]]
    return [[], []]


def mysql_plan_hash(plan):
    return hashlib.sha256(
        json.dumps(mysql_signature(plan), sort_keys=True).encode()
    ).hexdigest()[:16]


def apply_knob(cur, knob):
    kind, value = knob
    if kind == "switch":
        cur.execute(f"SET SESSION optimizer_switch='{value}';")
    elif kind == "var":
        cur.execute(f"SET SESSION {value};")


def reset_knobs(cur):
    cur.execute("SET SESSION optimizer_switch=DEFAULT;")
    cur.execute("SET SESSION optimizer_prune_level=DEFAULT;")
    cur.execute("SET SESSION optimizer_search_depth=DEFAULT;")


def get_plan(cur, sql, knob):
    try:
        apply_knob(cur, knob)
        cur.execute(f"EXPLAIN FORMAT=JSON {sql}")
        row = cur.fetchone()
        return json.loads(row[0])
    finally:
        reset_knobs(cur)


def timed_run(cur, sql, knob, timeout_ms):
    try:
        apply_knob(cur, knob)
        cur.execute(f"SET SESSION max_execution_time={int(timeout_ms)};")
        t0 = time.perf_counter()
        try:
            cur.execute(sql)
            cur.fetchall()
            return (time.perf_counter() - t0) * 1000, False
        except Exception as e:
            if "3024" in str(e) or "maximum statement execution time" in str(e).lower():
                return float(timeout_ms), True
            raise
    finally:
        cur.execute("SET SESSION max_execution_time=0;")
        reset_knobs(cur)


def measure(cur, sql, knob, timeout_ms, runs=3):
    ms, to = timed_run(cur, sql, knob, timeout_ms)   # warm-up
    if to:
        return float(timeout_ms), True
    vals = []
    for _ in range(runs):
        ms, to = timed_run(cur, sql, knob, timeout_ms)
        if to:
            return float(timeout_ms), True
        vals.append(ms)
    return statistics.median(vals), False


def build_row(cand_f, def_f, hint_set, all_hint_names):
    r = {}
    r["cost_ratio"] = cand_f["est_cost"] / (def_f["est_cost"] + RATIO_EPS)
    r["log_cost_ratio"] = math.log(max(r["cost_ratio"], 1e-6))
    r["rows_ratio"] = (cand_f["est_rows"] + 1) / (def_f["est_rows"] + 1)
    r["log_rows_ratio"] = math.log(max(r["rows_ratio"], 1e-6))
    r["nodes_ratio"] = cand_f["num_nodes"] / (def_f["num_nodes"] + RATIO_EPS)
    r["depth_delta"] = cand_f["max_depth"] - def_f["max_depth"]
    for k in ["nestloop", "hashjoin", "seqscan", "indexscan", "sort",
              "agg", "filter"]:
        r[f"{k}_delta"] = cand_f[f"n_{k}"] - def_f[f"n_{k}"]
    tj = max(cand_f["num_joins"], 1)
    r["nestloop_share"] = cand_f["n_nestloop"] / tj
    r["hashjoin_share"] = cand_f["n_hashjoin"] / tj
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


def export_and_load(mysql_conn, args):
    import psycopg2
    print("Exporting TPC-H from PostgreSQL and loading into MySQL...")
    pg = psycopg2.connect(dbname="tpch_sf1", user=args.pg_user,
                          password=args.pg_password, host=args.pg_host,
                          port=args.pg_port)
    pcur = pg.cursor()
    mcur = mysql_conn.cursor()
    mcur.execute("SET SESSION unique_checks=0;")
    mcur.execute("SET SESSION foreign_key_checks=0;")
    tmpdir = tempfile.mkdtemp(prefix="qppe_mysql_")
    for t in LOAD_ORDER:
        t0 = time.time()
        mcur.execute(f"DROP TABLE IF EXISTS {t};")
        mcur.execute(TABLES[t])
        path = os.path.join(tmpdir, f"{t}.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            pcur.copy_expert(
                f"COPY {t} TO STDOUT WITH (FORMAT csv, NULL '\\N')", f)
        path_sql = path.replace("\\", "/")
        mcur.execute(f"""
            LOAD DATA LOCAL INFILE '{path_sql}' INTO TABLE {t}
            FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
            LINES TERMINATED BY '\\n';
        """)
        mysql_conn.commit()
        mcur.execute(f"SELECT count(*) FROM {t};")
        n = mcur.fetchone()[0]
        os.remove(path)
        print(f"  {t:<10} {n:>10,} rows in {time.time()-t0:.1f}s")
    mcur.execute("ANALYZE TABLE " + ", ".join(LOAD_ORDER) + ";")
    mcur.fetchall()
    pcur.close(); pg.close()
    os.rmdir(tmpdir)
    print("Load complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", required=True, help="MySQL root pw")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--pg-password", default="postgres")
    parser.add_argument("--pg-host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--buffer-pool-gb", type=float, default=4.0)
    parser.add_argument("--theta-sev", type=float, default=2.0)
    parser.add_argument("--floor-ms", type=float, default=1000.0)
    parser.add_argument("--theta-win", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--tau-win", type=float, default=0.5)
    parser.add_argument("--timeout-cap-s", type=float, default=120.0)
    args = parser.parse_args()

    import pymysql
    s3c = load_module("s3c", "step3c_expand_corpus.py")
    s6 = load_module("s6", "step6_live_loop.py")

    print("QPPE Step 9b - MySQL Port")
    print("=" * 72)

    conn = pymysql.connect(host=args.host, port=args.port, user=args.user,
                           password=args.password, local_infile=True,
                           autocommit=True)
    cur = conn.cursor()
    cur.execute("SELECT VERSION();")
    print(f"MySQL version: {cur.fetchone()[0]}")

    # buffer pool (online resize, persisted)
    cur.execute("SELECT @@innodb_buffer_pool_size;")
    bp = int(cur.fetchone()[0])
    target = int(args.buffer_pool_gb * (1 << 30))
    if bp < target:
        print(f"Raising innodb_buffer_pool_size "
              f"{bp/(1<<30):.1f}GB -> {args.buffer_pool_gb:.1f}GB (online)")
        cur.execute(f"SET PERSIST innodb_buffer_pool_size={target};")
        time.sleep(3)
    cur.execute("SET PERSIST local_infile=1;")

    cur.execute("CREATE DATABASE IF NOT EXISTS tpch;")
    cur.execute("USE tpch;")
    cur.execute("SHOW TABLES;")
    have = {r[0] for r in cur.fetchall()}
    if "lineitem" not in have:
        export_and_load(conn, args)
        cur.execute("USE tpch;")
    cur.execute("SELECT count(*) FROM lineitem;")
    print(f"lineitem rows: {cur.fetchone()[0]:,}")

    # ---------------- dynamic knob layer ----------------
    cur.execute("SELECT @@optimizer_switch;")
    avail = {kv.split("=")[0] for kv in cur.fetchone()[0].split(",")}
    hint_sets = {"default": ("switch", "default")}
    for sw in PREFERRED_SWITCHES:
        if sw in avail and len(hint_sets) < 11:
            state = "on" if sw in ("batched_key_access",) else "off"
            hint_sets[f"{sw}_{state}"] = ("switch", f"{sw}={state}")
    hint_sets["exhaustive_search"] = ("var", "optimizer_prune_level=0")
    hint_sets["greedy_depth1"] = ("var", "optimizer_search_depth=1")
    print(f"Knobs ({len(hint_sets)}): {', '.join(hint_sets)}")

    # instances, dialect-translated + validated
    instances = []
    n_bad = 0
    for template, variants in s3c.PARAMS.items():
        for vi, params in enumerate(variants):
            sql = translate_sql(s3c.TEMPLATES[template].format(**params))
            try:
                cur.execute(f"EXPLAIN FORMAT=JSON {sql}")
                cur.fetchall()
                instances.append((template, vi, sql))
            except Exception as e:
                n_bad += 1
                print(f"  DIALECT FAILURE {template}v{vi}: {str(e)[:90]}")
    print(f"Instances valid: {len(instances)} ({n_bad} dialect failures - "
          f"report these for patching)")

    # ---------------- phase 1: corpus (checkpointed) ----------------
    if args.phase in (0, 1):
        done_records, done_q = [], set()
        if pathlib.Path(CORPUS_FILE).exists():
            with open(CORPUS_FILE, "rb") as f:
                blob = pickle.load(f)
            done_records = blob["records"]
            done_q = {r["query_id"] for r in done_records}
            print(f"\nResuming: {len(done_q)} queries already measured")
        todo = [x for x in instances if f"{x[0]}v{x[1]}" not in done_q]
        print(f"\nPHASE 1 - corpus: {len(todo)} queries to run "
              f"(checkpoint every 5)")
        t0, n_to = time.time(), 0
        for qi, (template, vi, sql) in enumerate(todo):
            qid = f"{template}v{vi}"
            plans = {}
            for hs_name, knob in hint_sets.items():
                try:
                    plans[hs_name] = get_plan(cur, sql, knob)
                except Exception as e:
                    print(f"  plan error {qid} [{hs_name}]: {str(e)[:70]}")
            if "default" not in plans:
                continue
            seen, todo_plans = {}, []
            for hs_name, plan in plans.items():
                h = mysql_plan_hash(plan)
                if h not in seen:
                    seen[h] = hs_name
                    todo_plans.append((hs_name, plan))
            d_ms, d_to = measure(cur, sql, hint_sets["default"],
                                 args.timeout_cap_s * 1000)
            timeout_ms = min(max(3 * d_ms, 10000), args.timeout_cap_s * 1000)
            for hs_name, plan in todo_plans:
                if hs_name == "default":
                    ms, to = d_ms, d_to
                else:
                    ms, to = measure(cur, sql, hint_sets[hs_name], timeout_ms)
                if to:
                    n_to += 1
                done_records.append(dict(
                    template=template, variant=vi, query_id=qid,
                    hint_set=hs_name, features=mysql_features(plan),
                    median_ms=ms, censored=to))
            if (qi + 1) % 5 == 0:
                with open(CORPUS_FILE, "wb") as f:
                    pickle.dump({"records": done_records,
                                 "hint_sets": list(hint_sets)}, f)
                el = time.time() - t0
                eta = el / (qi + 1) * (len(todo) - qi - 1)
                print(f"  {qi+1}/{len(todo)}  elapsed {el/60:.0f} min  "
                      f"ETA {eta/60:.0f} min  (timeouts so far: {n_to})")
        with open(CORPUS_FILE, "wb") as f:
            pickle.dump({"records": done_records,
                         "hint_sets": list(hint_sets)}, f)
        print(f"Corpus saved: {len(done_records)} measured plans, "
              f"{n_to} timeouts this session")

    if args.phase == 1:
        print("Phase 1 done. Rerun with --phase 2 (or no flag).")
        return

    # ---------------- phase 2: gate + live ----------------
    print("\nPHASE 2 - labels, calibration, live loop")
    with open(CORPUS_FILE, "rb") as f:
        blob = pickle.load(f)
    records, hint_names = blob["records"], blob["hint_sets"]

    df = pd.DataFrame(records)
    defaults = (df[df.hint_set == "default"]
                .set_index("query_id")[["median_ms", "features"]]
                .rename(columns={"median_ms": "def_ms",
                                 "features": "def_features"}))
    canddf = df[df.hint_set != "default"].join(defaults, on="query_id")
    rows = []
    for _, r in canddf.iterrows():
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

    td, to_ = 0.0, 0.0
    for qid, g in cand.groupby("query_id"):
        d = g["def_median_ms"].iloc[0]
        td += d
        to_ += min(d, g["median_ms"].min())
    print(f"Workload bounds: default {td/1000:.1f}s | oracle {to_/1000:.1f}s "
          f"(possible improvement {(td-to_)/td:.0%})")

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

    knob_map = dict(zip(hint_names,
                        [("switch", "default")] * len(hint_names)))
    # rebuild live knob map identically to phase-1 construction
    cur.execute("SELECT @@optimizer_switch;")
    avail = {kv.split("=")[0] for kv in cur.fetchone()[0].split(",")}
    live_knobs = {"default": ("switch", "default")}
    for sw in PREFERRED_SWITCHES:
        if sw in avail and len(live_knobs) < 11:
            state = "on" if sw in ("batched_key_access",) else "off"
            live_knobs[f"{sw}_{state}"] = ("switch", f"{sw}={state}")
    live_knobs["exhaustive_search"] = ("var", "optimizer_prune_level=0")
    live_knobs["greedy_depth1"] = ("var", "optimizer_search_depth=1")

    print(f"\nLIVE - MySQL fresh ({len(s6.FRESH)} queries, t*={t_star:.3f})")
    print(f"{'query':<10}{'decision':<32}{'default ms':>11}{'chosen ms':>10}"
          f"{'ovh ms':>8}{'result':>12}")
    print("-" * 85)
    results = []
    for template, variant, params in s6.FRESH:
        sql = translate_sql(s3c.TEMPLATES[template].format(**params))
        t0 = time.perf_counter()
        plans = {}
        try:
            for hs_name, knob in live_knobs.items():
                plans[hs_name] = get_plan(cur, sql, knob)
        except Exception as e:
            print(f"{template}v{variant:<4} planning failed: {str(e)[:60]}")
            continue
        gen_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        def_f = mysql_features(plans["default"])
        frows, names = [], []
        for hs_name, plan in plans.items():
            if hs_name == "default":
                continue
            frows.append(build_row(mysql_features(plan), def_f, hs_name,
                                   hint_names))
            names.append(hs_name)
        F = pd.DataFrame(frows).reindex(columns=feats, fill_value=0)
        p_sev = heads_s.predict_proba(F)[:, 1]
        p_win = heads_w.predict_proba(F)[:, 1]
        inf_ms = (time.perf_counter() - t0) * 1000

        cert = [(names[i], p_sev[i], p_win[i]) for i in range(len(names))
                if p_sev[i] < t_star and p_win[i] > args.tau_win]
        chosen = max(cert, key=lambda x: x[2])[0] if cert else "default"
        decision = f"steer:{chosen}" if cert else "keep default"

        d_ms, _ = measure(cur, sql, live_knobs["default"],
                          args.timeout_cap_s * 1000)
        if chosen == "default":
            c_ms = d_ms
        else:
            c_ms, _ = measure(cur, sql, live_knobs[chosen],
                              min(max(3 * d_ms, 10000),
                                  args.timeout_cap_s * 1000))
        slowq = c_ms / d_ms if d_ms > 0 else 1.0
        severe = (chosen != "default" and slowq > args.theta_sev
                  and c_ms - d_ms > args.floor_ms)
        result = ("-" if chosen == "default" else
                  "SEVERE" if severe else
                  "mild regr" if slowq > 1.2 else
                  f"won {d_ms/c_ms:.1f}x" if c_ms < d_ms else "~neutral")
        results.append(dict(default_ms=d_ms, chosen_ms=c_ms, gen_ms=gen_ms,
                            inf_ms=inf_ms, steered=chosen != "default",
                            severe=severe))
        print(f"{template + 'v' + str(variant):<10}{decision:<32}"
              f"{d_ms:>11.0f}{c_ms:>10.0f}{gen_ms + inf_ms:>8.1f}"
              f"{result:>12}")

    R = pd.DataFrame(results)
    tot_def = R.default_ms.sum()
    tot_pol = R.chosen_ms.sum() + R.gen_ms.sum() + R.inf_ms.sum()
    n_st, n_sv = int(R.steered.sum()), int(R.severe.sum())
    print("-" * 85)
    print(f"MySQL LIVE: steered {n_st}/{len(R)} | SEVERE {n_sv} "
          f"(rate {n_sv/max(n_st,1):.2f} vs alpha {args.alpha}) | "
          f"workload {tot_def/1000:.1f}s -> {tot_pol/1000:.1f}s "
          f"({(tot_def-tot_pol)/tot_def*100:+.1f}%, overhead incl.) | "
          f"overhead {R.gen_ms.mean()+R.inf_ms.mean():.1f} ms avg")
    print("\nDone. Paste the full output back.")


if __name__ == "__main__":
    main()
