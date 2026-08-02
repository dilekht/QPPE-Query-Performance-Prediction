#!/usr/bin/env python3
"""
QPPE Rebuild - Step 1b: Remaining Environment Checks (fixed)
Runs only the checks that were skipped after the transaction bug.

Usage:
    py step1b_remaining_checks.py --db postgres --user postgres --password 12345
"""

import argparse
import json
import time

def report(name, ok, detail=""):
    status = "OK " if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ->  {detail}" if detail else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="postgres")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    args = parser.parse_args()

    import psycopg2
    conn = psycopg2.connect(dbname=args.db, user=args.user, password=args.password,
                            host=args.host, port=args.port)
    conn.autocommit = True  # stay in autocommit; manage transactions manually
    cur = conn.cursor()

    print("QPPE Step 1b - Remaining Checks")
    print("=" * 60)

    # --- SET LOCAL scoping (fixed: manual BEGIN/COMMIT under autocommit) ---
    try:
        cur.execute("BEGIN;")
        cur.execute("SET LOCAL enable_hashjoin = off;")
        cur.execute("SHOW enable_hashjoin;")
        inside = cur.fetchone()[0]
        cur.execute("COMMIT;")
        cur.execute("SHOW enable_hashjoin;")
        outside = cur.fetchone()[0]
        ok = (inside == "off" and outside == "on")
        report("SET LOCAL scoping", ok, f"inside tx: {inside}, after commit: {outside}")
    except Exception as e:
        try:
            cur.execute("ROLLBACK;")
        except Exception:
            pass
        report("SET LOCAL scoping", False, str(e).strip())

    # --- EXPLAIN (FORMAT JSON) ---
    try:
        cur.execute("EXPLAIN (FORMAT JSON) SELECT 1;")
        plan = cur.fetchone()[0]
        if isinstance(plan, str):
            plan = json.loads(plan)
        node = plan[0]["Plan"]
        ok = all(k in node for k in ["Node Type", "Total Cost", "Plan Rows"])
        report("EXPLAIN (FORMAT JSON)", ok,
               f"root: {node.get('Node Type')}, cost={node.get('Total Cost')}")
    except Exception as e:
        report("EXPLAIN (FORMAT JSON)", False, str(e).strip())

    # --- EXPLAIN ANALYZE timing ---
    try:
        cur.execute("EXPLAIN (ANALYZE, FORMAT JSON, TIMING ON) SELECT count(*) FROM pg_class;")
        plan = cur.fetchone()[0]
        if isinstance(plan, str):
            plan = json.loads(plan)
        exec_time = plan[0].get("Execution Time")
        report("EXPLAIN ANALYZE timing", exec_time is not None,
               f"Execution Time: {exec_time} ms")
    except Exception as e:
        report("EXPLAIN ANALYZE timing", False, str(e).strip())

    # --- pg_stat_statements availability (optional) ---
    try:
        cur.execute("SELECT count(*) FROM pg_available_extensions WHERE name='pg_stat_statements';")
        report("pg_stat_statements available", cur.fetchone()[0] > 0, "optional")
    except Exception as e:
        report("pg_stat_statements available", False, str(e).strip())

    # --- Existing databases ---
    cur.execute("SELECT datname FROM pg_database WHERE NOT datistemplate ORDER BY datname;")
    dbs = [r[0] for r in cur.fetchall()]
    report("Databases present", True, ", ".join(dbs))
    for candidate in ["tpch", "imdb", "job"]:
        if candidate in dbs:
            report(f"Benchmark DB '{candidate}' found", True, "we can reuse it")

    # --- OS-level info PostgreSQL sees ---
    cur.execute("SELECT version();")
    report("Full version string", True, cur.fetchone()[0])

    cur.close()
    conn.close()

    # --- Client timing resolution ---
    samples = []
    for _ in range(2000):
        t0 = time.perf_counter()
        t1 = time.perf_counter()
        if t1 - t0 > 0:
            samples.append(t1 - t0)
    if samples:
        res_us = min(samples) * 1e6
        report("perf_counter resolution", res_us < 10, f"~{res_us:.2f} microseconds")
    else:
        report("perf_counter resolution", True, "sub-measurable (excellent)")

    print("=" * 60)
    print("Done. Paste this output back.")


if __name__ == "__main__":
    main()
