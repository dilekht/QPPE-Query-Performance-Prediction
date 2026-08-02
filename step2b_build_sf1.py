#!/usr/bin/env python3
"""
QPPE Rebuild - Step 2b: Build TPC-H SF1 in PostgreSQL
======================================================
Generates official-spec TPC-H data with DuckDB's built-in dbgen,
bulk-loads it into the (empty) tpch_sf1 database via COPY,
creates indexes + foreign keys, and runs ANALYZE.

Requirements:
    pip install duckdb

Usage:
    py step2b_build_sf1.py --user postgres --password 12345
    py step2b_build_sf1.py --user postgres --password 12345 --sf 1 --db tpch_sf1

Expected duration: ~5-15 minutes depending on disk speed.
Disk needed: ~2.5 GB temporarily (CSV exports), ~1.5 GB final DB.
"""

import argparse
import io
import os
import tempfile
import time

DDL = """
DROP TABLE IF EXISTS lineitem, orders, partsupp, part, customer, supplier, nation, region CASCADE;

CREATE TABLE region (
    r_regionkey  INT NOT NULL,
    r_name       CHAR(25) NOT NULL,
    r_comment    VARCHAR(152)
);
CREATE TABLE nation (
    n_nationkey  INT NOT NULL,
    n_name       CHAR(25) NOT NULL,
    n_regionkey  INT NOT NULL,
    n_comment    VARCHAR(152)
);
CREATE TABLE supplier (
    s_suppkey    INT NOT NULL,
    s_name       CHAR(25) NOT NULL,
    s_address    VARCHAR(40) NOT NULL,
    s_nationkey  INT NOT NULL,
    s_phone      CHAR(15) NOT NULL,
    s_acctbal    NUMERIC(15,2) NOT NULL,
    s_comment    VARCHAR(101) NOT NULL
);
CREATE TABLE customer (
    c_custkey    INT NOT NULL,
    c_name       VARCHAR(25) NOT NULL,
    c_address    VARCHAR(40) NOT NULL,
    c_nationkey  INT NOT NULL,
    c_phone      CHAR(15) NOT NULL,
    c_acctbal    NUMERIC(15,2) NOT NULL,
    c_mktsegment CHAR(10) NOT NULL,
    c_comment    VARCHAR(117) NOT NULL
);
CREATE TABLE part (
    p_partkey     INT NOT NULL,
    p_name        VARCHAR(55) NOT NULL,
    p_mfgr        CHAR(25) NOT NULL,
    p_brand       CHAR(10) NOT NULL,
    p_type        VARCHAR(25) NOT NULL,
    p_size        INT NOT NULL,
    p_container   CHAR(10) NOT NULL,
    p_retailprice NUMERIC(15,2) NOT NULL,
    p_comment     VARCHAR(23) NOT NULL
);
CREATE TABLE partsupp (
    ps_partkey    INT NOT NULL,
    ps_suppkey    INT NOT NULL,
    ps_availqty   INT NOT NULL,
    ps_supplycost NUMERIC(15,2) NOT NULL,
    ps_comment    VARCHAR(199) NOT NULL
);
CREATE TABLE orders (
    o_orderkey      BIGINT NOT NULL,
    o_custkey       INT NOT NULL,
    o_orderstatus   CHAR(1) NOT NULL,
    o_totalprice    NUMERIC(15,2) NOT NULL,
    o_orderdate     DATE NOT NULL,
    o_orderpriority CHAR(15) NOT NULL,
    o_clerk         CHAR(15) NOT NULL,
    o_shippriority  INT NOT NULL,
    o_comment       VARCHAR(79) NOT NULL
);
CREATE TABLE lineitem (
    l_orderkey      BIGINT NOT NULL,
    l_partkey       INT NOT NULL,
    l_suppkey       INT NOT NULL,
    l_linenumber    INT NOT NULL,
    l_quantity      NUMERIC(15,2) NOT NULL,
    l_extendedprice NUMERIC(15,2) NOT NULL,
    l_discount      NUMERIC(15,2) NOT NULL,
    l_tax           NUMERIC(15,2) NOT NULL,
    l_returnflag    CHAR(1) NOT NULL,
    l_linestatus    CHAR(1) NOT NULL,
    l_shipdate      DATE NOT NULL,
    l_commitdate    DATE NOT NULL,
    l_receiptdate   DATE NOT NULL,
    l_shipinstruct  CHAR(25) NOT NULL,
    l_shipmode      CHAR(10) NOT NULL,
    l_comment       VARCHAR(44) NOT NULL
);
"""

INDEXES = """
ALTER TABLE region   ADD PRIMARY KEY (r_regionkey);
ALTER TABLE nation   ADD PRIMARY KEY (n_nationkey);
ALTER TABLE supplier ADD PRIMARY KEY (s_suppkey);
ALTER TABLE customer ADD PRIMARY KEY (c_custkey);
ALTER TABLE part     ADD PRIMARY KEY (p_partkey);
ALTER TABLE partsupp ADD PRIMARY KEY (ps_partkey, ps_suppkey);
ALTER TABLE orders   ADD PRIMARY KEY (o_orderkey);
ALTER TABLE lineitem ADD PRIMARY KEY (l_orderkey, l_linenumber);

CREATE INDEX idx_nation_regionkey   ON nation (n_regionkey);
CREATE INDEX idx_supplier_nationkey ON supplier (s_nationkey);
CREATE INDEX idx_customer_nationkey ON customer (c_nationkey);
CREATE INDEX idx_customer_mktsegment ON customer (c_mktsegment);
CREATE INDEX idx_partsupp_suppkey   ON partsupp (ps_suppkey);
CREATE INDEX idx_orders_custkey     ON orders (o_custkey);
CREATE INDEX idx_orders_orderdate   ON orders (o_orderdate);
CREATE INDEX idx_lineitem_partkey   ON lineitem (l_partkey);
CREATE INDEX idx_lineitem_suppkey   ON lineitem (l_suppkey);
CREATE INDEX idx_lineitem_shipdate  ON lineitem (l_shipdate);
CREATE INDEX idx_lineitem_receiptdate ON lineitem (l_receiptdate);
CREATE INDEX idx_lineitem_partkey_suppkey ON lineitem (l_partkey, l_suppkey);

ALTER TABLE nation   ADD FOREIGN KEY (n_regionkey) REFERENCES region (r_regionkey);
ALTER TABLE supplier ADD FOREIGN KEY (s_nationkey) REFERENCES nation (n_nationkey);
ALTER TABLE customer ADD FOREIGN KEY (c_nationkey) REFERENCES nation (n_nationkey);
ALTER TABLE partsupp ADD FOREIGN KEY (ps_partkey) REFERENCES part (p_partkey);
ALTER TABLE partsupp ADD FOREIGN KEY (ps_suppkey) REFERENCES supplier (s_suppkey);
ALTER TABLE orders   ADD FOREIGN KEY (o_custkey) REFERENCES customer (c_custkey);
ALTER TABLE lineitem ADD FOREIGN KEY (l_orderkey) REFERENCES orders (o_orderkey);
ALTER TABLE lineitem ADD FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp (ps_partkey, ps_suppkey);
"""

TABLES = ["region", "nation", "supplier", "customer",
          "part", "partsupp", "orders", "lineitem"]

EXPECTED_SF1 = {"region": 5, "nation": 25, "supplier": 10_000, "customer": 150_000,
                "part": 200_000, "partsupp": 800_000, "orders": 1_500_000,
                "lineitem": 6_001_215}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--db", default="tpch_sf1")
    parser.add_argument("--sf", type=float, default=1.0, help="TPC-H scale factor")
    args = parser.parse_args()

    import duckdb
    import psycopg2

    t_start = time.time()

    # ---------------- 1. Generate data with DuckDB ----------------
    log(f"Generating TPC-H SF{args.sf} data with DuckDB (in temp storage)...")
    tmpdir = tempfile.mkdtemp(prefix="tpch_gen_")
    duck = duckdb.connect(os.path.join(tmpdir, "gen.duckdb"))
    duck.execute("INSTALL tpch; LOAD tpch;")
    duck.execute(f"CALL dbgen(sf={args.sf});")
    log(f"Generation done ({time.time()-t_start:.0f}s). Exporting to CSV...")

    csv_paths = {}
    for t in TABLES:
        path = os.path.join(tmpdir, f"{t}.csv").replace("\\", "/")
        duck.execute(f"COPY {t} TO '{path}' (FORMAT CSV, HEADER FALSE, DELIMITER '|');")
        csv_paths[t] = path
        size_mb = os.path.getsize(path) / 1e6
        log(f"  exported {t:<10} {size_mb:>8.1f} MB")
    duck.close()

    # ---------------- 2. Create schema in PostgreSQL ----------------
    log(f"Creating schema in database '{args.db}'...")
    conn = psycopg2.connect(dbname=args.db, user=args.user, password=args.password,
                            host=args.host, port=args.port)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(DDL)

    # Speed up bulk load for this session
    cur.execute("SET maintenance_work_mem = '512MB';")

    # ---------------- 3. Bulk load via COPY ----------------
    for t in TABLES:
        t0 = time.time()
        with open(csv_paths[t], "r", encoding="utf-8") as f:
            cur.copy_expert(
                f"COPY {t} FROM STDIN WITH (FORMAT csv, DELIMITER '|')", f)
        cur.execute(f"SELECT count(*) FROM {t};")
        n = cur.fetchone()[0]
        expected = int(EXPECTED_SF1[t] * args.sf) if args.sf != 1.0 else EXPECTED_SF1[t]
        flag = "OK" if (args.sf != 1.0 or n == expected) else f"MISMATCH (expected {expected:,})"
        log(f"  loaded {t:<10} {n:>12,} rows in {time.time()-t0:.1f}s  [{flag}]")

    # ---------------- 4. Indexes + constraints ----------------
    log("Creating primary keys, indexes, and foreign keys (this is the slow part)...")
    for stmt in [s.strip() for s in INDEXES.split(";") if s.strip()]:
        t0 = time.time()
        cur.execute(stmt + ";")
        name = stmt.split()[2] if stmt.startswith("CREATE") else stmt.split()[2]
        log(f"  {stmt.split()[0]} {name:<38} {time.time()-t0:.1f}s")

    # ---------------- 5. ANALYZE ----------------
    log("Running ANALYZE (statistics for the optimizer)...")
    cur.execute("ANALYZE;")

    # ---------------- 6. Verify ----------------
    cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()));")
    log(f"Final database size: {cur.fetchone()[0]}")

    # Quick sanity query with timing
    t0 = time.time()
    cur.execute("""
        SELECT l_returnflag, l_linestatus, count(*), sum(l_quantity)
        FROM lineitem
        WHERE l_shipdate <= date '1998-12-01' - interval '90 days'
        GROUP BY l_returnflag, l_linestatus;
    """)
    rows = cur.fetchall()
    log(f"Sanity check (TPC-H Q1 core): {len(rows)} groups in {time.time()-t0:.2f}s")

    cur.close()
    conn.close()

    # ---------------- 7. Cleanup CSVs ----------------
    for p in csv_paths.values():
        try:
            os.remove(p)
        except OSError:
            pass
    log(f"ALL DONE in {time.time()-t_start:.0f}s total. Temp files cleaned.")
    log("Paste this output back for verification.")


if __name__ == "__main__":
    main()
