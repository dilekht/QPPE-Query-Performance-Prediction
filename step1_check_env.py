#!/usr/bin/env python3
"""
QPPE Rebuild - Step 1: Environment Verification
================================================
Checks that your system can support the full project pipeline.

Usage:
    python step1_check_env.py --db postgres --user postgres --password YOURPASS

Paste the FULL output back for analysis.
"""

import sys
import platform
import argparse
import time
import json

RESULTS = []

def report(name, ok, detail=""):
    status = "OK " if ok else "FAIL"
    line = f"[{status}] {name}" + (f"  ->  {detail}" if detail else "")
    RESULTS.append((ok, line))
    print(line)

def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_python():
    section("1. Python Environment")
    v = sys.version_info
    report("Python version", v >= (3, 9), f"{v.major}.{v.minor}.{v.micro}")
    report("Platform", True, f"{platform.system()} {platform.release()} ({platform.machine()})")

    packages = {
        "psycopg2": "psycopg2-binary",
        "sklearn": "scikit-learn",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
    }
    for module, pipname in packages.items():
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "?")
            report(f"Package: {pipname}", True, f"v{ver}")
        except ImportError:
            report(f"Package: {pipname}", False, f"install with: pip install {pipname}")

    # Optional but needed for Step 5 (conformal prediction)
    try:
        import mapie  # noqa
        report("Package: mapie (conformal)", True, f"v{mapie.__version__}")
    except ImportError:
        report("Package: mapie (conformal)", False,
               "optional for now; Step 5 needs it: pip install mapie")


def check_postgres(args):
    section("2. PostgreSQL Connectivity & Capabilities")
    try:
        import psycopg2
    except ImportError:
        report("psycopg2 available", False, "cannot run DB checks")
        return

    conn_params = dict(dbname=args.db, user=args.user,
                       password=args.password, host=args.host, port=args.port)
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        report("Connection", True, f"{args.host}:{args.port}/{args.db}")
    except Exception as e:
        report("Connection", False, str(e).strip())
        return

    cur = conn.cursor()

    # Version
    cur.execute("SHOW server_version;")
    version = cur.fetchone()[0]
    major = int(version.split(".")[0])
    report("Server version", major >= 12, f"PostgreSQL {version}")

    # Key config values
    for guc in ["shared_buffers", "work_mem", "effective_cache_size",
                "random_page_cost", "max_parallel_workers_per_gather",
                "jit", "track_io_timing"]:
        try:
            cur.execute(f"SHOW {guc};")
            report(f"Config: {guc}", True, cur.fetchone()[0])
        except Exception as e:
            report(f"Config: {guc}", False, str(e).strip())

    # Plan-steering GUCs we will use as "hint set" knobs
    steering_gucs = ["enable_hashjoin", "enable_mergejoin", "enable_nestloop",
                     "enable_seqscan", "enable_indexscan", "enable_indexonlyscan",
                     "enable_bitmapscan", "enable_sort", "enable_hashagg",
                     "enable_material", "enable_memoize", "enable_gathermerge",
                     "enable_parallel_hash"]
    available = []
    for guc in steering_gucs:
        try:
            cur.execute(f"SHOW {guc};")
            cur.fetchone()
            available.append(guc)
        except Exception:
            pass
    report("Steering GUCs available", len(available) >= 9,
           f"{len(available)}/{len(steering_gucs)}: {', '.join(available)}")

    # SET LOCAL inside a transaction (core steering mechanism)
    try:
        conn.autocommit = False
        cur.execute("BEGIN;")
        cur.execute("SET LOCAL enable_hashjoin = off;")
        cur.execute("SHOW enable_hashjoin;")
        inside = cur.fetchone()[0]
        cur.execute("COMMIT;")
        cur.execute("SHOW enable_hashjoin;")
        outside = cur.fetchone()[0]
        conn.autocommit = True
        ok = (inside == "off" and outside == "on")
        report("SET LOCAL scoping", ok, f"inside tx: {inside}, after commit: {outside}")
    except Exception as e:
        conn.autocommit = True
        report("SET LOCAL scoping", False, str(e).strip())

    # EXPLAIN (FORMAT JSON) - feature extraction depends on this
    try:
        cur.execute("EXPLAIN (FORMAT JSON) SELECT 1;")
        plan = cur.fetchone()[0]
        if isinstance(plan, str):
            plan = json.loads(plan)
        node = plan[0]["Plan"]
        keys = ["Node Type", "Total Cost", "Plan Rows"]
        ok = all(k in node for k in keys)
        report("EXPLAIN (FORMAT JSON)", ok,
               f"root node: {node.get('Node Type')}, cost={node.get('Total Cost')}")
    except Exception as e:
        report("EXPLAIN (FORMAT JSON)", False, str(e).strip())

    # EXPLAIN ANALYZE with timing
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

    # pg_stat_statements (useful later, not required)
    try:
        cur.execute("SELECT count(*) FROM pg_available_extensions WHERE name='pg_stat_statements';")
        avail = cur.fetchone()[0] > 0
        report("pg_stat_statements available", avail,
               "optional; helps workload analysis later")
    except Exception:
        report("pg_stat_statements available", False, "optional")

    # Existing benchmark databases
    cur.execute("SELECT datname FROM pg_database WHERE NOT datistemplate ORDER BY datname;")
    dbs = [r[0] for r in cur.fetchall()]
    report("Databases present", True, ", ".join(dbs))
    for candidate in ["tpch", "imdb", "job"]:
        if candidate in dbs:
            report(f"Benchmark DB '{candidate}' found", True, "we can reuse it")

    cur.close()
    conn.close()


def check_timing_resolution():
    section("3. Client-Side Timing Resolution")
    samples = []
    for _ in range(2000):
        t0 = time.perf_counter()
        t1 = time.perf_counter()
        d = t1 - t0
        if d > 0:
            samples.append(d)
    if samples:
        res_us = min(samples) * 1e6
        report("perf_counter resolution", res_us < 10, f"~{res_us:.2f} microseconds")
    else:
        report("perf_counter resolution", True, "sub-measurable (excellent)")


def main():
    parser = argparse.ArgumentParser(description="QPPE Step 1: environment check")
    parser.add_argument("--db", default="postgres")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    args = parser.parse_args()

    print("QPPE Rebuild - Environment Verification Report")
    print(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    check_python()
    check_postgres(args)
    check_timing_resolution()

    section("SUMMARY")
    fails = [line for ok, line in RESULTS if not ok]
    print(f"Total checks: {len(RESULTS)}   Passed: {len(RESULTS) - len(fails)}   Failed: {len(fails)}")
    if fails:
        print("\nFailed checks:")
        for line in fails:
            print("  " + line)
    else:
        print("\nAll checks passed. Ready for Step 2 (benchmark setup).")


if __name__ == "__main__":
    main()
