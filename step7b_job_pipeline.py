#!/usr/bin/env python3
"""
QPPE Rebuild - Step 7b: JOB Plan Collection + Execution
========================================================
Ports the whole pipeline to the Join Order Benchmark:

  PHASE 1 (fast): read the 113 .sql files from job_queries/,
    register them (template = family number, variant = letter),
    collect plans under all 12 hint sets, print diversity report.
  PHASE 2 (long, RESUMABLE): execute every distinct (query, plan)
    pair - 1 warm-up + 3 measured runs, default first per query,
    adaptive timeout (3x default median, floor 10s, cap 120s).

All rows are scoped to benchmark='imdb'; the TPC-H corpus in the
same qppe database is untouched.

Requires step3c_expand_corpus.py in the same folder (plan feature
extraction is imported from it).

Usage:
    py step7b_job_pipeline.py --user postgres --password 12345
    py step7b_job_pipeline.py --user postgres --password 12345 --phase 1
Expected duration: phase 1 ~2 min; phase 2 possibly several hours
(run overnight; Ctrl+C + rerun resumes).
"""

import argparse
import glob
import importlib.util
import json
import os
import pathlib
import re
import statistics
import time


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, str(pathlib.Path(__file__).parent / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_query_files(qdir):
    files = []
    for path in glob.glob(os.path.join(qdir, "**", "*.sql"), recursive=True):
        base = os.path.basename(path)
        if re.fullmatch(r"\d+[a-z]\.sql", base):
            files.append(path)
    return sorted(files, key=lambda p: (
        int(re.match(r"(\d+)", os.path.basename(p)).group(1)),
        os.path.basename(p)))


def run_plan_once(bcur, sql, gucs, timeout_ms):
    bcur.execute("BEGIN;")
    try:
        bcur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)};")
        for guc, val in gucs.items():
            bcur.execute(f"SET LOCAL {guc} = {val};")
        try:
            bcur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) {sql}")
            out = bcur.fetchone()[0]
            if isinstance(out, str):
                out = json.loads(out)
            return out[0].get("Execution Time"), out[0].get("Planning Time"), False
        except Exception as e:
            msg = str(e).lower()
            if "statement timeout" in msg or "canceling" in msg:
                return float(timeout_ms), None, True
            raise
    finally:
        try:
            bcur.execute("ROLLBACK;")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--bench-db", default="imdb")
    parser.add_argument("--log-db", default="qppe")
    parser.add_argument("--queries-dir", default="job_queries")
    parser.add_argument("--phase", type=int, default=0,
                        help="1 = collection only, 2 = execution only, "
                             "0 = both")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-floor-s", type=float, default=10.0)
    parser.add_argument("--timeout-cap-s", type=float, default=120.0)
    args = parser.parse_args()

    import psycopg2
    from psycopg2.extras import Json

    s3c = load_module("s3c", "step3c_expand_corpus.py")
    HINT_SETS = s3c.HINT_SETS

    bench = psycopg2.connect(dbname=args.bench_db, user=args.user,
                             password=args.password, host=args.host,
                             port=args.port)
    bench.autocommit = True
    bcur = bench.cursor()
    logc = psycopg2.connect(dbname=args.log_db, user=args.user,
                            password=args.password, host=args.host,
                            port=args.port)
    logc.autocommit = True
    lcur = logc.cursor()

    for guc in ["shared_buffers", "jit", "work_mem"]:
        bcur.execute(f"SHOW {guc};")
        print(f"Config: {guc} = {bcur.fetchone()[0]}")

    # ============================================================
    # PHASE 1: register queries + collect plans
    # ============================================================
    if args.phase in (0, 1):
        files = find_query_files(args.queries_dir)
        if not files:
            print(f"\nNo JOB .sql files found under '{args.queries_dir}'.")
            print("Extract them first:  mkdir job_queries && "
                  "tar -xf job.tgz -C job_queries")
            return
        print(f"\nPHASE 1: found {len(files)} JOB query files")

        lcur.execute("SELECT name, hint_set_id FROM hint_sets;")
        hs_ids = dict(lcur.fetchall())

        t0 = time.time()
        n_pairs, n_errors = 0, 0
        for path in files:
            base = os.path.basename(path)[:-4]           # e.g. '17a'
            family = re.match(r"(\d+)", base).group(1)   # '17'
            letter = base[len(family):]                  # 'a'
            variant = ord(letter) - ord("a")
            with open(path, "r", encoding="utf-8") as f:
                sql = f.read().strip().rstrip(";").strip()

            lcur.execute("""
                INSERT INTO queries (template, variant, sql_text, params, benchmark)
                VALUES (%s, %s, %s, %s, 'imdb')
                ON CONFLICT (template, variant, benchmark)
                DO UPDATE SET sql_text = EXCLUDED.sql_text
                RETURNING query_id;
            """, (family, variant, sql, Json({"job_query": base})))
            query_id = lcur.fetchone()[0]

            for hs_name, gucs in HINT_SETS.items():
                bcur.execute("BEGIN;")
                try:
                    for guc, val in gucs.items():
                        bcur.execute(f"SET LOCAL {guc} = {val};")
                    bcur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
                    plan = bcur.fetchone()[0]
                    if isinstance(plan, str):
                        plan = json.loads(plan)
                    root = plan[0]["Plan"]
                except Exception as e:
                    n_errors += 1
                    print(f"  ERROR planning {base} [{hs_name}]: "
                          f"{str(e).strip()[:90]}")
                    continue
                finally:
                    try:
                        bcur.execute("ROLLBACK;")
                    except Exception:
                        pass

                feats = s3c.extract_features(root)
                ph = s3c.plan_hash(root)
                lcur.execute("""
                    INSERT INTO plans (query_id, hint_set_id, plan_json,
                                       plan_hash, est_cost, est_rows, features)
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

        print(f"Collected {n_pairs} (query, hint set) plans in "
              f"{time.time()-t0:.1f}s ({n_errors} errors)")

        print("\nJOB PLAN DIVERSITY (by family)")
        print(f"{'family':<8}{'queries':>8}{'distinct':>10}{'cost max/min':>14}")
        print("-" * 40)
        lcur.execute("""
            SELECT q.template, count(DISTINCT q.query_id),
                   count(DISTINCT p.plan_hash),
                   max(p.est_cost) / greatest(min(p.est_cost), 1e-9)
            FROM plans p JOIN queries q ON q.query_id = p.query_id
            WHERE q.benchmark = 'imdb'
            GROUP BY q.template
            ORDER BY q.template::int;
        """)
        for family, nq, nd, ratio in lcur.fetchall():
            print(f"{family:<8}{nq:>8}{nd:>10}{ratio:>14.1f}")
        lcur.execute("""
            SELECT count(*) FROM (
                SELECT DISTINCT p.query_id, p.plan_hash
                FROM plans p JOIN queries q ON q.query_id = p.query_id
                WHERE q.benchmark = 'imdb') d;
        """)
        print(f"\nDistinct (query, plan) pairs to execute: {lcur.fetchone()[0]}")

    if args.phase == 1:
        print("\nPhase 1 done. Rerun with --phase 2 (or no flag) to execute.")
        return

    # ============================================================
    # PHASE 2: execute distinct plans (resumable)
    # ============================================================
    print("\nPHASE 2: execution (resumable, Ctrl+C safe)")
    snapshot = {}
    for guc in ["shared_buffers", "work_mem", "jit"]:
        bcur.execute(f"SHOW {guc};")
        snapshot[guc] = bcur.fetchone()[0]

    lcur.execute("""
        WITH ranked AS (
            SELECT p.plan_id, p.query_id, p.plan_hash, h.name AS hint_name,
                   h.gucs,
                   row_number() OVER (
                       PARTITION BY p.query_id, p.plan_hash
                       ORDER BY (h.name = 'default') DESC, p.plan_id) AS rn
            FROM plans p
            JOIN hint_sets h ON h.hint_set_id = p.hint_set_id
            JOIN queries q ON q.query_id = p.query_id
            WHERE q.benchmark = 'imdb'
        )
        SELECT r.plan_id, r.query_id, q.template, q.variant, q.sql_text,
               r.hint_name, r.gucs
        FROM ranked r JOIN queries q ON q.query_id = r.query_id
        WHERE r.rn = 1
        ORDER BY r.query_id, (r.hint_name = 'default') DESC, r.plan_id;
    """)
    worklist = lcur.fetchall()
    lcur.execute("SELECT DISTINCT plan_id FROM executions WHERE NOT is_warmup;")
    done = {r[0] for r in lcur.fetchall()}
    todo = [w for w in worklist if w[0] not in done]
    print(f"Distinct plans: {len(worklist)} | done: "
          f"{len(worklist) - len(todo)} | to run: {len(todo)}")

    default_median = {}
    lcur.execute("""
        SELECT p.query_id, t.median_ms
        FROM plan_timings t
        JOIN plans p ON p.plan_id = t.plan_id
        JOIN hint_sets h ON h.hint_set_id = p.hint_set_id
        JOIN queries q ON q.query_id = p.query_id
        WHERE h.name = 'default' AND q.benchmark = 'imdb';
    """)
    for qid, med in lcur.fetchall():
        default_median[qid] = med

    from psycopg2.extras import Json as PJson
    t_start = time.time()
    n_done, n_timeouts = 0, 0
    for plan_id, query_id, template, variant, sql, hint_name, gucs in todo:
        sql = sql.strip().rstrip(";")
        base = default_median.get(query_id)
        if base is not None:
            timeout_ms = min(max(3 * base, args.timeout_floor_s * 1000),
                             args.timeout_cap_s * 1000)
        else:
            timeout_ms = args.timeout_cap_s * 1000

        label = f"{template}{chr(ord('a')+variant)} [{hint_name}] plan {plan_id}"
        ems, pms, to = run_plan_once(bcur, sql, gucs, timeout_ms)
        lcur.execute("""
            INSERT INTO executions (plan_id, run_index, exec_ms, planning_ms,
                                    is_warmup, timed_out, server_config)
            VALUES (%s, 0, %s, %s, TRUE, %s, %s);
        """, (plan_id, ems, pms, to, PJson(snapshot)))
        if to:
            lcur.execute("""
                INSERT INTO executions (plan_id, run_index, exec_ms,
                                        is_warmup, timed_out, server_config)
                VALUES (%s, 1, %s, FALSE, TRUE, %s);
            """, (plan_id, ems, PJson(snapshot)))
            n_timeouts += 1
            n_done += 1
            print(f"  {label:<40} TIMEOUT at {timeout_ms/1000:.0f}s (censored)")
            continue

        times = []
        for i in range(1, args.runs + 1):
            ems, pms, to = run_plan_once(bcur, sql, gucs, timeout_ms)
            lcur.execute("""
                INSERT INTO executions (plan_id, run_index, exec_ms,
                                        planning_ms, is_warmup, timed_out,
                                        server_config)
                VALUES (%s, %s, %s, %s, FALSE, %s, %s);
            """, (plan_id, i, ems, pms, to, PJson(snapshot)))
            if to:
                n_timeouts += 1
                break
            times.append(ems)
        if times:
            med = statistics.median(times)
            if hint_name == "default":
                default_median[query_id] = med
            print(f"  {label:<40} median {med:>10.1f} ms (n={len(times)})")
        n_done += 1
        if n_done % 25 == 0:
            el = time.time() - t_start
            eta = el / n_done * (len(todo) - n_done)
            print(f"--- {n_done}/{len(todo)}  elapsed {el/60:.0f} min  "
                  f"ETA {eta/60:.0f} min ---")

    print(f"\nExecuted {n_done} plans in {(time.time()-t_start)/60:.1f} min "
          f"({n_timeouts} timeouts)")
    lcur.execute("""
        SELECT count(DISTINCT t.plan_id),
               sum(CASE WHEN t.n_runs > 0 THEN 0 ELSE 1 END)
        FROM plan_timings t
        JOIN queries q ON q.query_id = t.query_id
        WHERE q.benchmark = 'imdb';
    """)
    print("Summary follows in the analysis step; paste the tail of this "
          "output (last ~50 lines) plus any TIMEOUT lines.")


if __name__ == "__main__":
    main()
