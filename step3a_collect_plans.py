#!/usr/bin/env python3
"""
QPPE Rebuild - Step 3a: Hint Sets, Plan Collection & Diversity Report
======================================================================
For every (query variant x hint set) pair:
  BEGIN; SET LOCAL <gucs>; EXPLAIN (FORMAT JSON) <query>; ROLLBACK;
Extracts features from the JSON plan tree, computes a structural hash,
stores everything in the qppe log database, and reports how many
DISTINCT plans each query actually has (the steering space).

No queries are executed yet - this is planning only (fast, ~1 min).

Usage:
    py step3a_collect_plans.py --user postgres --password 12345
"""

import argparse
import hashlib
import json
import time

# ================================================================
# 1. HINT SETS - the steering space
#    Each hint set is a dict of GUCs applied via SET LOCAL.
#    'default' is PostgreSQL untouched: the baseline we protect.
# ================================================================
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

# ================================================================
# 2. QUERY TEMPLATES - TPC-H subset with parameter variants
#    {p} placeholders substituted from PARAMS lists.
# ================================================================
TEMPLATES = {
"Q1": """
SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty,
       sum(l_extendedprice) AS sum_base_price,
       sum(l_extendedprice*(1-l_discount)) AS sum_disc_price,
       avg(l_quantity) AS avg_qty, count(*) AS count_order
FROM lineitem
WHERE l_shipdate <= date '1998-12-01' - interval '{days} days'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus""",

"Q3": """
SELECT l_orderkey, sum(l_extendedprice*(1-l_discount)) AS revenue,
       o_orderdate, o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = '{segment}' AND c_custkey = o_custkey
  AND l_orderkey = o_orderkey AND o_orderdate < date '1995-03-15'
  AND l_shipdate > date '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate LIMIT 10""",

"Q5": """
SELECT n_name, sum(l_extendedprice*(1-l_discount)) AS revenue
FROM customer, orders, lineitem, supplier, nation, region
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
  AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey
  AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
  AND r_name = '{region}' AND o_orderdate >= date '{year}-01-01'
  AND o_orderdate < date '{year}-01-01' + interval '1 year'
GROUP BY n_name ORDER BY revenue DESC""",

"Q6": """
SELECT sum(l_extendedprice*l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= date '{year}-01-01'
  AND l_shipdate < date '{year}-01-01' + interval '1 year'
  AND l_discount BETWEEN {disc} - 0.01 AND {disc} + 0.01
  AND l_quantity < {qty}""",

"Q7": """
SELECT supp_nation, cust_nation, l_year, sum(volume) AS revenue
FROM (SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation,
             extract(year FROM l_shipdate) AS l_year,
             l_extendedprice*(1-l_discount) AS volume
      FROM supplier, lineitem, orders, customer, nation n1, nation n2
      WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey
        AND c_custkey = o_custkey AND s_nationkey = n1.n_nationkey
        AND c_nationkey = n2.n_nationkey
        AND ((n1.n_name = '{nation1}' AND n2.n_name = '{nation2}')
          OR (n1.n_name = '{nation2}' AND n2.n_name = '{nation1}'))
        AND l_shipdate BETWEEN date '1995-01-01' AND date '1996-12-31'
     ) AS shipping
GROUP BY supp_nation, cust_nation, l_year
ORDER BY supp_nation, cust_nation, l_year""",

"Q8": """
SELECT o_year, sum(CASE WHEN nation = '{nation}' THEN volume ELSE 0 END)/sum(volume) AS mkt_share
FROM (SELECT extract(year FROM o_orderdate) AS o_year,
             l_extendedprice*(1-l_discount) AS volume, n2.n_name AS nation
      FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
      WHERE p_partkey = l_partkey AND s_suppkey = l_suppkey
        AND l_orderkey = o_orderkey AND o_custkey = c_custkey
        AND c_nationkey = n1.n_nationkey AND n1.n_regionkey = r_regionkey
        AND r_name = '{region}' AND s_nationkey = n2.n_nationkey
        AND o_orderdate BETWEEN date '1995-01-01' AND date '1996-12-31'
        AND p_type = '{ptype}'
     ) AS all_nations
GROUP BY o_year ORDER BY o_year""",

"Q10": """
SELECT c_custkey, c_name, sum(l_extendedprice*(1-l_discount)) AS revenue,
       c_acctbal, n_name, c_address, c_phone
FROM customer, orders, lineitem, nation
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
  AND o_orderdate >= date '{qstart}'
  AND o_orderdate < date '{qstart}' + interval '3 months'
  AND l_returnflag = 'R' AND c_nationkey = n_nationkey
GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address
ORDER BY revenue DESC LIMIT 20""",

"Q12": """
SELECT l_shipmode,
       sum(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_line_count,
       sum(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_line_count
FROM orders, lineitem
WHERE o_orderkey = l_orderkey AND l_shipmode IN ('{mode1}', '{mode2}')
  AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate
  AND l_receiptdate >= date '{year}-01-01'
  AND l_receiptdate < date '{year}-01-01' + interval '1 year'
GROUP BY l_shipmode ORDER BY l_shipmode""",

"Q14": """
SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%%' THEN l_extendedprice*(1-l_discount)
                         ELSE 0 END) / sum(l_extendedprice*(1-l_discount)) AS promo_revenue
FROM lineitem, part
WHERE l_partkey = p_partkey AND l_shipdate >= date '{month}-01'
  AND l_shipdate < date '{month}-01' + interval '1 month'""",

"Q18": """
SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity)
FROM customer, orders, lineitem
WHERE o_orderkey IN (SELECT l_orderkey FROM lineitem
                     GROUP BY l_orderkey HAVING sum(l_quantity) > {qty})
  AND c_custkey = o_custkey AND o_orderkey = l_orderkey
GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
ORDER BY o_totalprice DESC, o_orderdate LIMIT 100""",
}

PARAMS = {
    "Q1":  [{"days": 60}, {"days": 90}, {"days": 120}],
    "Q3":  [{"segment": "BUILDING"}, {"segment": "AUTOMOBILE"}, {"segment": "MACHINERY"}],
    "Q5":  [{"region": "ASIA", "year": 1994}, {"region": "EUROPE", "year": 1995},
            {"region": "AMERICA", "year": 1996}],
    "Q6":  [{"year": 1994, "disc": 0.06, "qty": 24}, {"year": 1995, "disc": 0.05, "qty": 25},
            {"year": 1996, "disc": 0.07, "qty": 24}],
    "Q7":  [{"nation1": "FRANCE", "nation2": "GERMANY"},
            {"nation1": "CHINA", "nation2": "JAPAN"},
            {"nation1": "UNITED STATES", "nation2": "CANADA"}],
    "Q8":  [{"nation": "BRAZIL", "region": "AMERICA", "ptype": "ECONOMY ANODIZED STEEL"},
            {"nation": "CHINA", "region": "ASIA", "ptype": "SMALL PLATED COPPER"},
            {"nation": "GERMANY", "region": "EUROPE", "ptype": "STANDARD BRUSHED NICKEL"}],
    "Q10": [{"qstart": "1993-10-01"}, {"qstart": "1994-01-01"}, {"qstart": "1994-07-01"}],
    "Q12": [{"mode1": "MAIL", "mode2": "SHIP", "year": 1994},
            {"mode1": "AIR", "mode2": "TRUCK", "year": 1995},
            {"mode1": "RAIL", "mode2": "FOB", "year": 1996}],
    "Q14": [{"month": "1995-09"}, {"month": "1994-03"}, {"month": "1996-06"}],
    "Q18": [{"qty": 300}, {"qty": 312}, {"qty": 315}],
}

JOIN_NODES = {"Nested Loop", "Hash Join", "Merge Join"}
SCAN_NODES = {"Seq Scan", "Index Scan", "Index Only Scan",
              "Bitmap Heap Scan", "Bitmap Index Scan", "Tid Scan"}


# ================================================================
# 3. PLAN TREE ANALYSIS
# ================================================================
def walk(node, depth=0):
    """Yield (node, depth) for every node in the plan tree, including subplans."""
    yield node, depth
    for child in node.get("Plans", []):
        yield from walk(child, depth + 1)


def structural_signature(node):
    """Canonical structure: node types + relations + join types, no costs.
    Two plans with the same signature are the same plan."""
    sig = [node.get("Node Type"),
           node.get("Relation Name"),
           node.get("Index Name"),
           node.get("Join Type"),
           node.get("Parent Relationship"),
           node.get("Strategy")]
    children = [structural_signature(c) for c in node.get("Plans", [])]
    return [sig, children]


def plan_hash(root):
    canon = json.dumps(structural_signature(root), sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def extract_features(root):
    """Feature vector from the JSON plan tree. Far richer than the old
    regex approach: join types and scan types are now real features."""
    f = {
        "est_cost": root.get("Total Cost", 0.0),
        "est_startup_cost": root.get("Startup Cost", 0.0),
        "est_rows": root.get("Plan Rows", 0),
        "est_width": root.get("Plan Width", 0),
    }
    counts = {}
    relations = set()
    max_depth = 0
    n_parallel = 0
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


# ================================================================
# 4. COLLECTION
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--bench-db", default="tpch_sf1")
    parser.add_argument("--log-db", default="qppe")
    args = parser.parse_args()

    import psycopg2
    from psycopg2.extras import Json

    bench = psycopg2.connect(dbname=args.bench_db, user=args.user,
                             password=args.password, host=args.host, port=args.port)
    bench.autocommit = True
    bcur = bench.cursor()

    logc = psycopg2.connect(dbname=args.log_db, user=args.user,
                            password=args.password, host=args.host, port=args.port)
    logc.autocommit = True
    lcur = logc.cursor()

    # sanity: confirm frozen config
    bcur.execute("SHOW shared_buffers;")
    sb = bcur.fetchone()[0]
    bcur.execute("SHOW jit;")
    jit = bcur.fetchone()[0]
    print(f"Config check: shared_buffers={sb}, jit={jit}")
    if jit != "off":
        print("WARNING: jit is not off - did PostgreSQL restart with new config?")

    # register hint sets
    for name, gucs in HINT_SETS.items():
        lcur.execute("""
            INSERT INTO hint_sets (name, gucs) VALUES (%s, %s)
            ON CONFLICT (name) DO UPDATE SET gucs = EXCLUDED.gucs
            RETURNING hint_set_id;
        """, (name, Json(gucs)))
    lcur.execute("SELECT name, hint_set_id FROM hint_sets;")
    hs_ids = dict(lcur.fetchall())
    print(f"Registered {len(hs_ids)} hint sets.")

    t0 = time.time()
    total_pairs = 0
    for template, variants in PARAMS.items():
        for vi, params in enumerate(variants):
            sql = TEMPLATES[template].format(**params)
            lcur.execute("""
                INSERT INTO queries (template, variant, sql_text, params, benchmark)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (template, variant, benchmark)
                DO UPDATE SET sql_text = EXCLUDED.sql_text
                RETURNING query_id;
            """, (template, vi, sql, Json(params), args.bench_db))
            query_id = lcur.fetchone()[0]

            for hs_name, gucs in HINT_SETS.items():
                bcur.execute("BEGIN;")
                try:
                    for guc, val in gucs.items():
                        bcur.execute(f"SET LOCAL {guc} = {val};")
                    bcur.execute(f"EXPLAIN (FORMAT JSON) {sql};")
                    plan = bcur.fetchone()[0]
                    if isinstance(plan, str):
                        plan = json.loads(plan)
                    root = plan[0]["Plan"]
                finally:
                    bcur.execute("ROLLBACK;")

                feats = extract_features(root)
                ph = plan_hash(root)
                lcur.execute("""
                    INSERT INTO plans (query_id, hint_set_id, plan_json, plan_hash,
                                       est_cost, est_rows, features)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (query_id, hint_set_id) DO UPDATE
                    SET plan_json = EXCLUDED.plan_json,
                        plan_hash = EXCLUDED.plan_hash,
                        est_cost = EXCLUDED.est_cost,
                        est_rows = EXCLUDED.est_rows,
                        features = EXCLUDED.features;
                """, (query_id, hs_ids[hs_name], Json(plan), ph,
                      feats["est_cost"], feats["est_rows"], Json(feats)))
                total_pairs += 1

    print(f"Collected {total_pairs} (query, hint set) plans in {time.time()-t0:.1f}s.")

    # ============================================================
    # 5. DIVERSITY REPORT
    # ============================================================
    print("\nPLAN DIVERSITY REPORT")
    print("=" * 76)
    print(f"{'Query':<10}{'variants':>9}{'plans':>7}{'distinct':>10}"
          f"{'cost min':>12}{'cost max':>12}{'max/min':>9}")
    print("-" * 76)
    lcur.execute("""
        SELECT q.template,
               count(DISTINCT q.query_id)  AS variants,
               count(*)                     AS plans,
               count(DISTINCT p.plan_hash)  AS distinct_plans,
               min(p.est_cost)              AS cmin,
               max(p.est_cost)              AS cmax
        FROM plans p JOIN queries q ON q.query_id = p.query_id
        GROUP BY q.template ORDER BY q.template;
    """)
    for template, variants, plans, distinct, cmin, cmax in lcur.fetchall():
        ratio = (cmax / cmin) if cmin and cmin > 0 else float("nan")
        print(f"{template:<10}{variants:>9}{plans:>7}{distinct:>10}"
              f"{cmin:>12.0f}{cmax:>12.0f}{ratio:>9.1f}")

    # distinct plans per individual query variant (what 3b will execute)
    lcur.execute("""
        SELECT count(*) FROM (
            SELECT DISTINCT query_id, plan_hash FROM plans
        ) d;
    """)
    n_exec = lcur.fetchone()[0]
    lcur.execute("SELECT count(*) FROM plans;")
    n_all = lcur.fetchone()[0]
    print("-" * 76)
    print(f"Total (query, hint set) pairs: {n_all}")
    print(f"Distinct (query, plan) pairs Step 3b must execute: {n_exec}")
    print(f"Deduplication saves: {100 * (1 - n_exec / n_all):.0f}% of execution work")

    bcur.close(); bench.close()
    lcur.close(); logc.close()
    print("\nDone. Paste the full output back.")


if __name__ == "__main__":
    main()
