#!/usr/bin/env python3
"""
QPPE Rebuild - Step 2: Inspect Existing TPC-H Databases
========================================================
Verifies scale factor, row counts, indexes, primary/foreign keys,
and statistics freshness of the tpch / tpch_sf1 databases.

Usage:
    py step2_inspect_tpch.py --user postgres --password 12345
"""

import argparse

TPCH_TABLES = ["region", "nation", "supplier", "customer",
               "part", "partsupp", "orders", "lineitem"]

# Expected row counts at SF1 (for scale detection)
SF1_ROWS = {"region": 5, "nation": 25, "supplier": 10_000, "customer": 150_000,
            "part": 200_000, "partsupp": 800_000, "orders": 1_500_000,
            "lineitem": 6_001_215}


def inspect_database(dbname, args):
    import psycopg2
    print("\n" + "=" * 70)
    print(f"DATABASE: {dbname}")
    print("=" * 70)
    try:
        conn = psycopg2.connect(dbname=dbname, user=args.user,
                                password=args.password, host=args.host, port=args.port)
    except Exception as e:
        print(f"  Cannot connect: {e}")
        return
    conn.autocommit = True
    cur = conn.cursor()

    # Database size
    cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()));")
    print(f"Total size: {cur.fetchone()[0]}")

    # Tables present (any schema, exclude system)
    cur.execute("""
        SELECT schemaname, tablename FROM pg_tables
        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY schemaname, tablename;
    """)
    tables = cur.fetchall()
    if not tables:
        print("No user tables found.")
        cur.close(); conn.close()
        return

    print(f"\n{'Table':<22}{'Rows (est.)':>14}{'Size':>12}{'Indexes':>9}{'Last analyze':>22}")
    print("-" * 79)

    sf_estimates = []
    for schema, table in tables:
        fq = f'"{schema}"."{table}"'
        cur.execute("""
            SELECT reltuples::bigint FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = %s;
        """, (schema, table))
        row = cur.fetchone()
        est_rows = row[0] if row else -1

        cur.execute(f"SELECT pg_size_pretty(pg_total_relation_size(%s));", (fq.replace('"',''),))
        try:
            size = cur.fetchone()[0]
        except Exception:
            size = "?"

        cur.execute("""
            SELECT count(*) FROM pg_indexes
            WHERE schemaname = %s AND tablename = %s;
        """, (schema, table))
        n_idx = cur.fetchone()[0]

        cur.execute("""
            SELECT COALESCE(last_analyze, last_autoanalyze)
            FROM pg_stat_user_tables
            WHERE schemaname = %s AND relname = %s;
        """, (schema, table))
        row = cur.fetchone()
        last_an = str(row[0])[:19] if row and row[0] else "NEVER"

        print(f"{table:<22}{est_rows:>14,}{size:>12}{n_idx:>9}{last_an:>22}")

        if table in SF1_ROWS and SF1_ROWS[table] >= 10_000 and est_rows > 0:
            sf_estimates.append(est_rows / SF1_ROWS[table])

    if sf_estimates:
        sf = sum(sf_estimates) / len(sf_estimates)
        print(f"\nEstimated scale factor: ~SF{sf:.2f}")

    # Index details
    print("\nIndexes:")
    cur.execute("""
        SELECT tablename, indexname, indexdef FROM pg_indexes
        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY tablename, indexname;
    """)
    idx = cur.fetchall()
    if idx:
        for t, name, definition in idx:
            # print compactly: table, index name, and the column part
            cols = definition.split("(", 1)[-1].rstrip(")") if "(" in definition else "?"
            kind = "UNIQUE" if "UNIQUE" in definition else "btree "
            print(f"  {t:<18} {name:<34} [{kind}] ({cols})")
    else:
        print("  NONE - this matters; we will need to create them.")

    # Primary and foreign keys
    cur.execute("""
        SELECT conrelid::regclass::text, contype, count(*)
        FROM pg_constraint
        WHERE contype IN ('p','f') AND connamespace NOT IN
            (SELECT oid FROM pg_namespace WHERE nspname IN ('pg_catalog','information_schema'))
        GROUP BY 1, 2 ORDER BY 1, 2;
    """)
    cons = cur.fetchall()
    print("\nConstraints (p=primary key, f=foreign key):")
    if cons:
        for table, ctype, n in cons:
            print(f"  {table:<20} {ctype} x{n}")
    else:
        print("  NONE")

    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--databases", nargs="+", default=["tpch", "tpch_sf1"])
    args = parser.parse_args()

    print("QPPE Step 2 - TPC-H Database Inspection")
    for db in args.databases:
        inspect_database(db, args)
    print("\nDone. Paste the full output back.")


if __name__ == "__main__":
    main()
