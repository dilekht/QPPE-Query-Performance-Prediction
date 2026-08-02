#!/usr/bin/env python3
"""
QPPE Rebuild - Step 3b: Execute Distinct Plans (Training Corpus)
=================================================================
Executes every DISTINCT (query, plan) pair recorded by Step 3a:
  - 1 warm-up run (discarded) + N measured runs per plan
  - default plan first per query -> establishes the baseline and an
    ADAPTIVE TIMEOUT (3x default median, floor 10s, cap 120s) so a
    catastrophic steered plan cannot stall the whole collection
  - timings come from EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF):
    server-side Execution Time, no result-transfer noise
  - RESUMABLE: already-executed plans are skipped, so you can stop
    (Ctrl+C) and restart at any time

Usage:
    py step3b_execute_plans.py --user postgres --password 12345
    py step3b_execute_plans.py --user postgres --password 12345 --runs 3

Close other heavy applications while this runs.
"""

import argparse
import json
import statistics
import time


def get_conns(args):
    import psycopg2
    bench = psycopg2.connect(dbname=args.bench_db, user=args.user,
                             password=args.password, host=args.host, port=args.port)
    bench.autocommit = True
    logc = psycopg2.connect(dbname=args.log_db, user=args.user,
                            password=args.password, host=args.host, port=args.port)
    logc.autocommit = True
    return bench, logc


def run_plan_once(bcur, sql, gucs, timeout_ms):
    """One execution under SET LOCAL gucs. Returns (exec_ms, planning_ms, timed_out)."""
    bcur.execute("BEGIN;")
    try:
        bcur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)};")
        for guc, val in gucs.items():
            bcur.execute(f"SET LOCAL {guc} = {val};")
        try:
            bcur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) {sql};")
            out = bcur.fetchone()[0]
            if isinstance(out, str):
                out = json.loads(out)
            return out[0].get("Execution Time"), out[0].get("Planning Time"), False
        except Exception as e:
            if "statement timeout" in str(e).lower() or "canceling statement" in str(e).lower():
                return float(timeout_ms), None, True
            raise
    finally:
        try:
            bcur.execute("ROLLBACK;")
        except Exception:
            pass


def record(lcur, plan_id, run_index, exec_ms, planning_ms, is_warmup, timed_out, config_snapshot):
    from psycopg2.extras import Json
    lcur.execute("""
        INSERT INTO executions (plan_id, run_index, exec_ms, planning_ms,
                                is_warmup, timed_out, server_config)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
    """, (plan_id, run_index, exec_ms, planning_ms, is_warmup, timed_out, Json(config_snapshot)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--bench-db", default="tpch_sf1")
    parser.add_argument("--log-db", default="qppe")
    parser.add_argument("--runs", type=int, default=3, help="measured runs per plan")
    parser.add_argument("--timeout-floor-s", type=float, default=10.0)
    parser.add_argument("--timeout-cap-s", type=float, default=120.0)
    args = parser.parse_args()

    bench, logc = get_conns(args)
    bcur = bench.cursor()
    lcur = logc.cursor()

    # frozen-config snapshot stored with every execution
    snapshot = {}
    for guc in ["shared_buffers", "work_mem", "effective_cache_size",
                "random_page_cost", "jit", "max_parallel_workers_per_gather"]:
        bcur.execute(f"SHOW {guc};")
        snapshot[guc] = bcur.fetchone()[0]
    print("Config snapshot:", snapshot)
    if snapshot.get("jit") != "off" or snapshot.get("shared_buffers") != "7GB":
        print("WARNING: config differs from the frozen Step 2c setup!")

    # ------------------------------------------------------------
    # Work list: one representative hint set per DISTINCT plan.
    # Prefer 'default' as representative when it produced that plan.
    # ------------------------------------------------------------
    lcur.execute("""
        WITH ranked AS (
            SELECT p.plan_id, p.query_id, p.plan_hash, p.hint_set_id,
                   h.name AS hint_name, h.gucs,
                   row_number() OVER (
                       PARTITION BY p.query_id, p.plan_hash
                       ORDER BY (h.name = 'default') DESC, p.plan_id
                   ) AS rn
            FROM plans p JOIN hint_sets h ON h.hint_set_id = p.hint_set_id
        )
        SELECT r.plan_id, r.query_id, q.template, q.variant, q.sql_text,
               r.hint_name, r.gucs
        FROM ranked r JOIN queries q ON q.query_id = r.query_id
        WHERE r.rn = 1
        ORDER BY r.query_id, (r.hint_name = 'default') DESC, r.plan_id;
    """)
    worklist = lcur.fetchall()

    # resume: skip plans that already have measured runs
    lcur.execute("SELECT DISTINCT plan_id FROM executions WHERE NOT is_warmup;")
    done = {r[0] for r in lcur.fetchall()}

    todo = [w for w in worklist if w[0] not in done]
    print(f"Distinct plans total: {len(worklist)} | already done: {len(done)} | to run: {len(todo)}")
    if not todo:
        print("Nothing to do - Step 3b already complete.")
        return

    # ------------------------------------------------------------
    # Execution loop, grouped by query so the default goes first
    # and sets the adaptive timeout for its siblings.
    # ------------------------------------------------------------
    default_median = {}   # query_id -> median default exec_ms (for timeout)

    # preload defaults that were already measured in a previous session
    lcur.execute("""
        SELECT p.query_id, t.median_ms
        FROM plan_timings t
        JOIN plans p ON p.plan_id = t.plan_id
        JOIN hint_sets h ON h.hint_set_id = p.hint_set_id
        WHERE h.name = 'default';
    """)
    for qid, med in lcur.fetchall():
        default_median[qid] = med

    t_start = time.time()
    n_done = 0
    n_timeouts = 0

    for plan_id, query_id, template, variant, sql, hint_name, gucs in todo:
        # timeout: 3x default median, clamped
        base = default_median.get(query_id)
        if base is not None:
            timeout_ms = min(max(3 * base, args.timeout_floor_s * 1000),
                             args.timeout_cap_s * 1000)
        else:
            timeout_ms = args.timeout_cap_s * 1000  # default not measured yet

        label = f"{template}v{variant} [{hint_name}] plan {plan_id}"
        times = []

        # warm-up (run_index 0)
        ems, pms, to = run_plan_once(bcur, sql, gucs, timeout_ms)
        record(lcur, plan_id, 0, ems, pms, True, to, snapshot)

        if to:
            # a plan that cannot finish one warm-up within timeout is a
            # confirmed disaster: record it once as measured+censored, move on
            record(lcur, plan_id, 1, ems, None, False, True, snapshot)
            n_timeouts += 1
            n_done += 1
            print(f"  {label:<44} TIMEOUT at {timeout_ms/1000:.0f}s (censored)")
            continue

        for i in range(1, args.runs + 1):
            ems, pms, to = run_plan_once(bcur, sql, gucs, timeout_ms)
            record(lcur, plan_id, i, ems, pms, False, to, snapshot)
            if to:
                n_timeouts += 1
                break
            times.append(ems)

        if times:
            med = statistics.median(times)
            if hint_name == "default":
                default_median[query_id] = med
            spread = (max(times) - min(times)) / med * 100 if med > 0 else 0
            print(f"  {label:<44} median {med:>10.1f} ms  "
                  f"(n={len(times)}, spread {spread:.0f}%)")
        n_done += 1

        if n_done % 20 == 0:
            elapsed = time.time() - t_start
            rate = elapsed / n_done
            eta = rate * (len(todo) - n_done)
            print(f"--- progress {n_done}/{len(todo)}  elapsed {elapsed/60:.1f} min  "
                  f"ETA {eta/60:.1f} min ---")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Executed {n_done} plans in {(time.time()-t_start)/60:.1f} min "
          f"({n_timeouts} timeouts/censored)")

    lcur.execute("""
        SELECT q.template,
               count(DISTINCT t.plan_id)                 AS plans,
               min(t.median_ms)                          AS best_ms,
               max(t.median_ms)                          AS worst_ms
        FROM plan_timings t JOIN queries q ON q.query_id = t.query_id
        GROUP BY q.template ORDER BY q.template;
    """)
    print(f"\n{'Query':<8}{'plans':>7}{'best ms':>12}{'worst ms':>12}{'worst/best':>12}")
    print("-" * 51)
    for template, plans, best, worst in lcur.fetchall():
        ratio = worst / best if best and best > 0 else float("nan")
        print(f"{template:<8}{plans:>7}{best:>12.1f}{worst:>12.1f}{ratio:>12.1f}")

    print("\nDone. Paste the full output back (truncate the per-plan lines")
    print("if too long, but keep the summary table and any TIMEOUT lines).")


if __name__ == "__main__":
    main()
