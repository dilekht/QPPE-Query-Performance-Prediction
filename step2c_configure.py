#!/usr/bin/env python3
"""
QPPE Rebuild - Step 2c: Server Configuration + Experiment Log Schema
=====================================================================
1. Detects total system RAM and recommends PostgreSQL settings for
   reproducible benchmarking.
2. With --apply, writes them via ALTER SYSTEM (shared_buffers needs a
   PostgreSQL service restart afterwards).
3. Creates the 'qppe' database with the experiment-log schema that
   every later step (harness, labeling, model, evaluation) writes to
   and reads from.

Usage:
    py step2c_configure.py --user postgres --password 12345           # dry run: show only
    py step2c_configure.py --user postgres --password 12345 --apply   # apply settings + create schema
"""

import argparse
import ctypes
import sys


def get_total_ram_gb():
    """Detect total physical RAM on Windows (falls back to psutil elsewhere)."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            return None


LOG_SCHEMA = """
-- ============================================================
-- QPPE Experiment Log Schema (the backbone of the project)
-- Every candidate plan we ever generate and every execution we
-- ever time lands here. Training data, evaluation results, and
-- paper figures are all derived from these tables.
-- ============================================================

CREATE TABLE IF NOT EXISTS hint_sets (
    hint_set_id   SERIAL PRIMARY KEY,
    name          TEXT UNIQUE NOT NULL,
    gucs          JSONB NOT NULL,          -- e.g. {"enable_hashjoin":"off"}
    description   TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS queries (
    query_id      SERIAL PRIMARY KEY,
    template      TEXT NOT NULL,           -- e.g. 'Q3'
    variant       INT NOT NULL DEFAULT 0,  -- parameter variant number
    sql_text      TEXT NOT NULL,
    params        JSONB,                   -- the substituted parameter values
    benchmark     TEXT NOT NULL DEFAULT 'tpch_sf1',
    created_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE (template, variant, benchmark)
);

CREATE TABLE IF NOT EXISTS plans (
    plan_id       SERIAL PRIMARY KEY,
    query_id      INT NOT NULL REFERENCES queries(query_id),
    hint_set_id   INT NOT NULL REFERENCES hint_sets(hint_set_id),
    plan_json     JSONB NOT NULL,          -- full EXPLAIN (FORMAT JSON) output
    plan_hash     TEXT NOT NULL,           -- structural hash: detects when two
                                           -- hint sets yield the SAME plan
    est_cost      DOUBLE PRECISION,
    est_rows      DOUBLE PRECISION,
    features      JSONB,                   -- extracted feature vector
    created_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE (query_id, hint_set_id)
);
CREATE INDEX IF NOT EXISTS idx_plans_query ON plans(query_id);
CREATE INDEX IF NOT EXISTS idx_plans_hash  ON plans(plan_hash);

CREATE TABLE IF NOT EXISTS executions (
    execution_id  SERIAL PRIMARY KEY,
    plan_id       INT NOT NULL REFERENCES plans(plan_id),
    run_index     INT NOT NULL,            -- 0 = warm-up, 1..N = measured
    exec_ms       DOUBLE PRECISION NOT NULL,
    planning_ms   DOUBLE PRECISION,
    is_warmup     BOOLEAN NOT NULL DEFAULT FALSE,
    timed_out     BOOLEAN NOT NULL DEFAULT FALSE,
    server_config JSONB,                   -- snapshot of relevant GUCs
    executed_at   TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_exec_plan ON executions(plan_id);

-- Convenience view: median measured time per plan
CREATE OR REPLACE VIEW plan_timings AS
SELECT p.plan_id, p.query_id, p.hint_set_id,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY e.exec_ms) AS median_ms,
       min(e.exec_ms) AS min_ms,
       max(e.exec_ms) AS max_ms,
       count(*) AS n_runs
FROM plans p
JOIN executions e ON e.plan_id = p.plan_id
WHERE NOT e.is_warmup AND NOT e.timed_out
GROUP BY p.plan_id, p.query_id, p.hint_set_id;
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--apply", action="store_true",
                        help="actually apply settings and create the log schema")
    args = parser.parse_args()

    import psycopg2

    ram = get_total_ram_gb()
    print("QPPE Step 2c - Configuration")
    print("=" * 60)
    if ram:
        print(f"Detected total RAM: {ram:.1f} GB")
    else:
        print("Could not detect RAM; assuming 16 GB")
        ram = 16.0

    # --- Recommendations for a benchmarking workstation ---
    # shared_buffers: 25% of RAM, capped at 8GB (diminishing returns beyond)
    shared_buffers_gb = min(max(int(ram * 0.25), 1), 8)
    eff_cache_gb = int(ram * 0.5)
    recommendations = {
        "shared_buffers": f"{shared_buffers_gb}GB",      # needs restart
        "effective_cache_size": f"{eff_cache_gb}GB",
        "work_mem": "64MB",
        "maintenance_work_mem": "512MB",
        "jit": "off",                 # removes compilation latency spikes
        "track_io_timing": "on",      # enables I/O timing in EXPLAIN ANALYZE
        "max_parallel_workers_per_gather": "2",  # keep, but fixed & documented
    }

    conn = psycopg2.connect(dbname="postgres", user=args.user, password=args.password,
                            host=args.host, port=args.port)
    conn.autocommit = True
    cur = conn.cursor()

    print(f"\n{'Setting':<34}{'Current':>14}{'Recommended':>14}")
    print("-" * 62)
    needs_restart = False
    for guc, target in recommendations.items():
        cur.execute(f"SHOW {guc};")
        current = cur.fetchone()[0]
        mark = "" if current == target else "  <- change"
        print(f"{guc:<34}{current:>14}{target:>14}{mark}")
        if guc == "shared_buffers" and current != target:
            needs_restart = True

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write these settings")
        print("and create the experiment-log database.")
        cur.close(); conn.close()
        return

    # --- Apply via ALTER SYSTEM ---
    print("\nApplying settings via ALTER SYSTEM...")
    for guc, target in recommendations.items():
        cur.execute(f"ALTER SYSTEM SET {guc} = '{target}';")
        print(f"  set {guc} = {target}")
    cur.execute("SELECT pg_reload_conf();")
    print("  configuration reloaded")

    # --- Create qppe log database ---
    cur.execute("SELECT 1 FROM pg_database WHERE datname = 'qppe';")
    if not cur.fetchone():
        cur.execute("CREATE DATABASE qppe;")
        print("  created database 'qppe'")
    else:
        print("  database 'qppe' already exists")
    cur.close(); conn.close()

    conn = psycopg2.connect(dbname="qppe", user=args.user, password=args.password,
                            host=args.host, port=args.port)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(LOG_SCHEMA)
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' ORDER BY table_name;
    """)
    tables = [r[0] for r in cur.fetchall()]
    print(f"  log schema created: {', '.join(tables)}")
    cur.close(); conn.close()

    print("\n" + "=" * 60)
    if needs_restart:
        print("IMPORTANT: shared_buffers requires a PostgreSQL RESTART.")
        print("Open Command Prompt AS ADMINISTRATOR and run:")
        print("    net stop postgresql-x64-18")
        print("    net start postgresql-x64-18")
        print("(or restart the service from services.msc)")
        print("\nThen verify with:")
        print(f"    py step2c_configure.py --user {args.user} --password **** ")
        print("(dry run should show shared_buffers matching the recommendation)")
    else:
        print("No restart needed. Ready for Step 3 (candidate-plan harness).")


if __name__ == "__main__":
    main()
