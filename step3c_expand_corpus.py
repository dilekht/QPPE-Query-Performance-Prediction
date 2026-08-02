#!/usr/bin/env python3
"""
QPPE Rebuild - Step 3c: Corpus Expansion (Templates + Variants)
================================================================
Adds to the existing corpus (Step 3a data is kept, upserts are safe):
  - 10 NEW templates: Q2, Q4, Q9, Q11, Q13, Q16, Q17, Q19, Q21, Q22
    (correlated subqueries, EXISTS/NOT EXISTS, outer joins, OR-heavy
     predicates - the query shapes the model has never seen)
  - 3 additional variants for each of the 10 existing templates

After this, run step3b_execute_plans.py again: it is resumable and
will execute only the NEW distinct plans.

Usage:
    py step3c_expand_corpus.py --user postgres --password 12345
"""

import argparse
import hashlib
import json
import time

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

TEMPLATES = {
# ---------------- existing 10 (unchanged text) ----------------
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

# ---------------- NEW templates ----------------
"Q2": """
SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
FROM part, supplier, partsupp, nation, region
WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
  AND p_size = {size} AND p_type LIKE '%{ptype}'
  AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
  AND r_name = '{region}'
  AND ps_supplycost = (SELECT min(ps_supplycost)
                       FROM partsupp, supplier, nation, region
                       WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
                         AND s_nationkey = n_nationkey
                         AND n_regionkey = r_regionkey AND r_name = '{region}')
ORDER BY s_acctbal DESC, n_name, s_name, p_partkey LIMIT 100""",

"Q4": """
SELECT o_orderpriority, count(*) AS order_count
FROM orders
WHERE o_orderdate >= date '{qstart}'
  AND o_orderdate < date '{qstart}' + interval '3 months'
  AND EXISTS (SELECT * FROM lineitem
              WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate)
GROUP BY o_orderpriority ORDER BY o_orderpriority""",

"Q9": """
SELECT nation, o_year, sum(amount) AS sum_profit
FROM (SELECT n_name AS nation, extract(year FROM o_orderdate) AS o_year,
             l_extendedprice*(1-l_discount) - ps_supplycost*l_quantity AS amount
      FROM part, supplier, lineitem, partsupp, orders, nation
      WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey
        AND ps_partkey = l_partkey AND p_partkey = l_partkey
        AND o_orderkey = l_orderkey AND s_nationkey = n_nationkey
        AND p_name LIKE '%{color}%'
     ) AS profit
GROUP BY nation, o_year ORDER BY nation, o_year DESC""",

"Q11": """
SELECT ps_partkey, sum(ps_supplycost*ps_availqty) AS value
FROM partsupp, supplier, nation
WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = '{nation}'
GROUP BY ps_partkey
HAVING sum(ps_supplycost*ps_availqty) >
       (SELECT sum(ps_supplycost*ps_availqty) * {frac}
        FROM partsupp, supplier, nation
        WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey
          AND n_name = '{nation}')
ORDER BY value DESC""",

"Q13": """
SELECT c_count, count(*) AS custdist
FROM (SELECT c_custkey, count(o_orderkey) AS c_count
      FROM customer LEFT OUTER JOIN orders
           ON c_custkey = o_custkey
          AND o_comment NOT LIKE '%{w1}%{w2}%'
      GROUP BY c_custkey) AS c_orders
GROUP BY c_count ORDER BY custdist DESC, c_count DESC""",

"Q16": """
SELECT p_brand, p_type, p_size, count(DISTINCT ps_suppkey) AS supplier_cnt
FROM partsupp, part
WHERE p_partkey = ps_partkey AND p_brand <> '{brand}'
  AND p_type NOT LIKE '{ptype}%'
  AND p_size IN ({sizes})
  AND ps_suppkey NOT IN (SELECT s_suppkey FROM supplier
                         WHERE s_comment LIKE '%Customer%Complaints%')
GROUP BY p_brand, p_type, p_size
ORDER BY supplier_cnt DESC, p_brand, p_type, p_size""",

"Q17": """
SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem, part
WHERE p_partkey = l_partkey AND p_brand = '{brand}'
  AND p_container = '{container}'
  AND l_quantity < (SELECT 0.2 * avg(l_quantity)
                    FROM lineitem WHERE l_partkey = p_partkey)""",

"Q19": """
SELECT sum(l_extendedprice*(1-l_discount)) AS revenue
FROM lineitem, part
WHERE (p_partkey = l_partkey AND p_brand = '{b1}'
   AND p_container IN ('SM CASE','SM BOX','SM PACK','SM PKG')
   AND l_quantity >= {q1} AND l_quantity <= {q1} + 10
   AND p_size BETWEEN 1 AND 5
   AND l_shipmode IN ('AIR','AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON')
   OR (p_partkey = l_partkey AND p_brand = '{b2}'
   AND p_container IN ('MED BAG','MED BOX','MED PKG','MED PACK')
   AND l_quantity >= {q2} AND l_quantity <= {q2} + 10
   AND p_size BETWEEN 1 AND 10
   AND l_shipmode IN ('AIR','AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON')
   OR (p_partkey = l_partkey AND p_brand = '{b3}'
   AND p_container IN ('LG CASE','LG BOX','LG PACK','LG PKG')
   AND l_quantity >= {q3} AND l_quantity <= {q3} + 10
   AND p_size BETWEEN 1 AND 15
   AND l_shipmode IN ('AIR','AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON')""",

"Q21": """
SELECT s_name, count(*) AS numwait
FROM supplier, lineitem l1, orders, nation
WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey
  AND o_orderstatus = 'F' AND l1.l_receiptdate > l1.l_commitdate
  AND EXISTS (SELECT * FROM lineitem l2
              WHERE l2.l_orderkey = l1.l_orderkey
                AND l2.l_suppkey <> l1.l_suppkey)
  AND NOT EXISTS (SELECT * FROM lineitem l3
                  WHERE l3.l_orderkey = l1.l_orderkey
                    AND l3.l_suppkey <> l1.l_suppkey
                    AND l3.l_receiptdate > l3.l_commitdate)
  AND s_nationkey = n_nationkey AND n_name = '{nation}'
GROUP BY s_name ORDER BY numwait DESC, s_name LIMIT 100""",

"Q22": """
SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal
FROM (SELECT substring(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal
      FROM customer
      WHERE substring(c_phone FROM 1 FOR 2) IN ({codes})
        AND c_acctbal > (SELECT avg(c_acctbal) FROM customer
                         WHERE c_acctbal > 0.00
                           AND substring(c_phone FROM 1 FOR 2) IN ({codes}))
        AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)
     ) AS custsale
GROUP BY cntrycode ORDER BY cntrycode""",
}

PARAMS = {
    # existing templates: variants 0-2 unchanged (upsert keeps them),
    # variants 3-5 are new
    "Q1":  [{"days": 60}, {"days": 90}, {"days": 120},
            {"days": 75}, {"days": 100}, {"days": 110}],
    "Q3":  [{"segment": "BUILDING"}, {"segment": "AUTOMOBILE"}, {"segment": "MACHINERY"},
            {"segment": "FURNITURE"}, {"segment": "HOUSEHOLD"}],
    "Q5":  [{"region": "ASIA", "year": 1994}, {"region": "EUROPE", "year": 1995},
            {"region": "AMERICA", "year": 1996},
            {"region": "MIDDLE EAST", "year": 1997}, {"region": "AFRICA", "year": 1993},
            {"region": "ASIA", "year": 1995}],
    "Q6":  [{"year": 1994, "disc": 0.06, "qty": 24}, {"year": 1995, "disc": 0.05, "qty": 25},
            {"year": 1996, "disc": 0.07, "qty": 24},
            {"year": 1993, "disc": 0.04, "qty": 24}, {"year": 1997, "disc": 0.08, "qty": 25},
            {"year": 1995, "disc": 0.06, "qty": 30}],
    "Q7":  [{"nation1": "FRANCE", "nation2": "GERMANY"},
            {"nation1": "CHINA", "nation2": "JAPAN"},
            {"nation1": "UNITED STATES", "nation2": "CANADA"},
            {"nation1": "BRAZIL", "nation2": "ARGENTINA"},
            {"nation1": "INDIA", "nation2": "INDONESIA"},
            {"nation1": "RUSSIA", "nation2": "ROMANIA"}],
    "Q8":  [{"nation": "BRAZIL", "region": "AMERICA", "ptype": "ECONOMY ANODIZED STEEL"},
            {"nation": "CHINA", "region": "ASIA", "ptype": "SMALL PLATED COPPER"},
            {"nation": "GERMANY", "region": "EUROPE", "ptype": "STANDARD BRUSHED NICKEL"},
            {"nation": "INDIA", "region": "ASIA", "ptype": "PROMO BURNISHED TIN"},
            {"nation": "FRANCE", "region": "EUROPE", "ptype": "LARGE POLISHED BRASS"},
            {"nation": "PERU", "region": "AMERICA", "ptype": "MEDIUM ANODIZED COPPER"}],
    "Q10": [{"qstart": "1993-10-01"}, {"qstart": "1994-01-01"}, {"qstart": "1994-07-01"},
            {"qstart": "1993-01-01"}, {"qstart": "1994-10-01"}, {"qstart": "1995-01-01"}],
    "Q12": [{"mode1": "MAIL", "mode2": "SHIP", "year": 1994},
            {"mode1": "AIR", "mode2": "TRUCK", "year": 1995},
            {"mode1": "RAIL", "mode2": "FOB", "year": 1996},
            {"mode1": "SHIP", "mode2": "RAIL", "year": 1993},
            {"mode1": "TRUCK", "mode2": "MAIL", "year": 1997},
            {"mode1": "REG AIR", "mode2": "FOB", "year": 1994}],
    "Q14": [{"month": "1995-09"}, {"month": "1994-03"}, {"month": "1996-06"},
            {"month": "1993-07"}, {"month": "1997-01"}, {"month": "1995-03"}],
    "Q18": [{"qty": 300}, {"qty": 312}, {"qty": 315},
            {"qty": 305}, {"qty": 308}, {"qty": 310}],

    # new templates: 3 variants each
    "Q2":  [{"size": 15, "ptype": "BRASS", "region": "EUROPE"},
            {"size": 25, "ptype": "STEEL", "region": "ASIA"},
            {"size": 35, "ptype": "COPPER", "region": "AMERICA"}],
    "Q4":  [{"qstart": "1993-07-01"}, {"qstart": "1995-01-01"}, {"qstart": "1996-04-01"}],
    "Q9":  [{"color": "green"}, {"color": "red"}, {"color": "blue"}],
    "Q11": [{"nation": "GERMANY", "frac": 0.0001},
            {"nation": "FRANCE", "frac": 0.0001},
            {"nation": "CHINA", "frac": 0.0001}],
    "Q13": [{"w1": "special", "w2": "requests"},
            {"w1": "pending", "w2": "deposits"},
            {"w1": "unusual", "w2": "packages"}],
    "Q16": [{"brand": "Brand#45", "ptype": "MEDIUM POLISHED",
             "sizes": "49, 14, 23, 45, 19, 3, 36, 9"},
            {"brand": "Brand#23", "ptype": "SMALL BRUSHED",
             "sizes": "12, 17, 25, 31, 39, 42, 5, 8"},
            {"brand": "Brand#34", "ptype": "LARGE ANODIZED",
             "sizes": "10, 20, 30, 40, 15, 26, 37, 48"}],
    "Q17": [{"brand": "Brand#23", "container": "MED BOX"},
            {"brand": "Brand#12", "container": "JUMBO PKG"},
            {"brand": "Brand#34", "container": "SM CASE"}],
    "Q19": [{"b1": "Brand#12", "q1": 1, "b2": "Brand#23", "q2": 10, "b3": "Brand#34", "q3": 20},
            {"b1": "Brand#21", "q1": 5, "b2": "Brand#13", "q2": 14, "b3": "Brand#42", "q3": 24},
            {"b1": "Brand#33", "q1": 3, "b2": "Brand#43", "q2": 12, "b3": "Brand#15", "q3": 22}],
    "Q21": [{"nation": "SAUDI ARABIA"}, {"nation": "FRANCE"}, {"nation": "UNITED STATES"}],
    "Q22": [{"codes": "'13','31','23','29','30','18','17'"},
            {"codes": "'20','40','60','10','11','12','14'"},
            {"codes": "'15','16','19','21','22','24','25'"}],
}

JOIN_NODES = {"Nested Loop", "Hash Join", "Merge Join"}


def walk(node, depth=0):
    yield node, depth
    for child in node.get("Plans", []):
        yield from walk(child, depth + 1)


def structural_signature(node):
    sig = [node.get("Node Type"), node.get("Relation Name"),
           node.get("Index Name"), node.get("Join Type"),
           node.get("Parent Relationship"), node.get("Strategy")]
    children = [structural_signature(c) for c in node.get("Plans", [])]
    return [sig, children]


def plan_hash(root):
    canon = json.dumps(structural_signature(root), sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def extract_features(root):
    f = {"est_cost": root.get("Total Cost", 0.0),
         "est_startup_cost": root.get("Startup Cost", 0.0),
         "est_rows": root.get("Plan Rows", 0),
         "est_width": root.get("Plan Width", 0)}
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

    for name, gucs in HINT_SETS.items():
        lcur.execute("""
            INSERT INTO hint_sets (name, gucs) VALUES (%s, %s)
            ON CONFLICT (name) DO UPDATE SET gucs = EXCLUDED.gucs;
        """, (name, Json(gucs)))
    lcur.execute("SELECT name, hint_set_id FROM hint_sets;")
    hs_ids = dict(lcur.fetchall())

    t0 = time.time()
    n_pairs, n_errors = 0, 0
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
                except Exception as e:
                    n_errors += 1
                    print(f"ERROR planning {template}v{vi} [{hs_name}]: "
                          f"{str(e).strip()[:100]}")
                    bcur.execute("ROLLBACK;")
                    continue
                finally:
                    try:
                        bcur.execute("ROLLBACK;")
                    except Exception:
                        pass

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
                n_pairs += 1

    print(f"\nCollected/updated {n_pairs} (query, hint set) plans "
          f"in {time.time()-t0:.1f}s ({n_errors} planning errors)")

    print("\nPLAN DIVERSITY REPORT (full corpus)")
    print("=" * 76)
    print(f"{'Query':<10}{'variants':>9}{'plans':>7}{'distinct':>10}"
          f"{'cost min':>12}{'cost max':>12}{'max/min':>9}")
    print("-" * 76)
    lcur.execute("""
        SELECT q.template, count(DISTINCT q.query_id), count(*),
               count(DISTINCT p.plan_hash), min(p.est_cost), max(p.est_cost)
        FROM plans p JOIN queries q ON q.query_id = p.query_id
        GROUP BY q.template ORDER BY q.template;
    """)
    for template, variants, plans, distinct, cmin, cmax in lcur.fetchall():
        ratio = (cmax / cmin) if cmin and cmin > 0 else float("nan")
        print(f"{template:<10}{variants:>9}{plans:>7}{distinct:>10}"
              f"{cmin:>12.0f}{cmax:>12.0f}{ratio:>9.1f}")

    lcur.execute("SELECT count(*) FROM (SELECT DISTINCT query_id, plan_hash FROM plans) d;")
    n_exec = lcur.fetchone()[0]
    lcur.execute("""
        SELECT count(*) FROM (SELECT DISTINCT query_id, plan_hash FROM plans) d
        WHERE NOT EXISTS (
            SELECT 1 FROM executions e JOIN plans p ON p.plan_id = e.plan_id
            WHERE p.query_id = d.query_id AND p.plan_hash = d.plan_hash
              AND NOT e.is_warmup);
    """)
    n_new = lcur.fetchone()[0]
    print("-" * 76)
    print(f"Distinct (query, plan) pairs total: {n_exec}")
    print(f"NEW pairs for step3b to execute:    {n_new}")
    print("\nNow run:  py step3b_execute_plans.py --user ... --password ...")
    print("(resumable - it will skip everything already measured)")

    bcur.close(); bench.close()
    lcur.close(); logc.close()


if __name__ == "__main__":
    main()
