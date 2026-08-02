#!/usr/bin/env python3
"""
QPPE Rebuild - Step 7a: JOB (Join Order Benchmark) Setup
=========================================================
Downloads the IMDB dataset (~1.2 GB) and the 113 JOB queries,
creates the 21-table schema in a new 'imdb' database, bulk-loads
via COPY, builds primary keys + foreign-key indexes, and ANALYZEs.

Disk needed: ~1.2 GB download + ~6 GB extracted/loaded (temp files
are cleaned afterwards). Expected duration: 15-40 minutes depending
on connection and disk.

Usage:
    py step7a_setup_job.py --user postgres --password 12345
    # if the default mirror is down:
    py step7a_setup_job.py --user postgres --password 12345 --imdb-url <url>
    # if you already downloaded imdb.tgz manually:
    py step7a_setup_job.py --user postgres --password 12345 --imdb-tgz C:/path/imdb.tgz
"""

import argparse
import io
import os
import tarfile
import tempfile
import time
import urllib.request

IMDB_URL = "https://event.cwi.nl/da/job/imdb.tgz"
QUERIES_URL = ("https://codeload.github.com/gregrahn/join-order-benchmark"
               "/tar.gz/refs/heads/master")

DDL = """
DROP TABLE IF EXISTS aka_name, aka_title, cast_info, char_name,
  comp_cast_type, company_name, company_type, complete_cast, info_type,
  keyword, kind_type, link_type, movie_companies, movie_info,
  movie_info_idx, movie_keyword, movie_link, name, person_info,
  role_type, title CASCADE;

CREATE TABLE aka_name (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    name text NOT NULL,
    imdb_index character varying(12),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);
CREATE TABLE aka_title (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    title text NOT NULL,
    imdb_index character varying(12),
    kind_id integer NOT NULL,
    production_year integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note text,
    md5sum character varying(32)
);
CREATE TABLE cast_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    movie_id integer NOT NULL,
    person_role_id integer,
    note text,
    nr_order integer,
    role_id integer NOT NULL
);
CREATE TABLE char_name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    imdb_index character varying(12),
    imdb_id integer,
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);
CREATE TABLE comp_cast_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL
);
CREATE TABLE company_name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    country_code character varying(255),
    imdb_id integer,
    name_pcode_nf character varying(5),
    name_pcode_sf character varying(5),
    md5sum character varying(32)
);
CREATE TABLE company_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL
);
CREATE TABLE complete_cast (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer,
    subject_id integer NOT NULL,
    status_id integer NOT NULL
);
CREATE TABLE info_type (
    id integer NOT NULL PRIMARY KEY,
    info character varying(32) NOT NULL
);
CREATE TABLE keyword (
    id integer NOT NULL PRIMARY KEY,
    keyword text NOT NULL,
    phonetic_code character varying(5)
);
CREATE TABLE kind_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(15) NOT NULL
);
CREATE TABLE link_type (
    id integer NOT NULL PRIMARY KEY,
    link character varying(32) NOT NULL
);
CREATE TABLE movie_companies (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    company_id integer NOT NULL,
    company_type_id integer NOT NULL,
    note text
);
CREATE TABLE movie_info (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);
CREATE TABLE movie_info_idx (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);
CREATE TABLE movie_keyword (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    keyword_id integer NOT NULL
);
CREATE TABLE movie_link (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    linked_movie_id integer NOT NULL,
    link_type_id integer NOT NULL
);
CREATE TABLE name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    imdb_index character varying(12),
    imdb_id integer,
    gender character varying(1),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);
CREATE TABLE person_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);
CREATE TABLE role_type (
    id integer NOT NULL PRIMARY KEY,
    role character varying(32) NOT NULL
);
CREATE TABLE title (
    id integer NOT NULL PRIMARY KEY,
    title text NOT NULL,
    imdb_index character varying(12),
    kind_id integer NOT NULL,
    production_year integer,
    imdb_id integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years character varying(49),
    md5sum character varying(32)
);
"""

FK_INDEXES = """
CREATE INDEX company_id_movie_companies ON movie_companies(company_id);
CREATE INDEX company_type_id_movie_companies ON movie_companies(company_type_id);
CREATE INDEX info_type_id_movie_info_idx ON movie_info_idx(info_type_id);
CREATE INDEX info_type_id_movie_info ON movie_info(info_type_id);
CREATE INDEX info_type_id_person_info ON person_info(info_type_id);
CREATE INDEX keyword_id_movie_keyword ON movie_keyword(keyword_id);
CREATE INDEX kind_id_aka_title ON aka_title(kind_id);
CREATE INDEX kind_id_title ON title(kind_id);
CREATE INDEX linked_movie_id_movie_link ON movie_link(linked_movie_id);
CREATE INDEX link_type_id_movie_link ON movie_link(link_type_id);
CREATE INDEX movie_id_aka_title ON aka_title(movie_id);
CREATE INDEX movie_id_cast_info ON cast_info(movie_id);
CREATE INDEX movie_id_complete_cast ON complete_cast(movie_id);
CREATE INDEX movie_id_movie_companies ON movie_companies(movie_id);
CREATE INDEX movie_id_movie_info_idx ON movie_info_idx(movie_id);
CREATE INDEX movie_id_movie_keyword ON movie_keyword(movie_id);
CREATE INDEX movie_id_movie_link ON movie_link(movie_id);
CREATE INDEX movie_id_movie_info ON movie_info(movie_id);
CREATE INDEX person_id_aka_name ON aka_name(person_id);
CREATE INDEX person_id_cast_info ON cast_info(person_id);
CREATE INDEX person_id_person_info ON person_info(person_id);
CREATE INDEX person_role_id_cast_info ON cast_info(person_role_id);
CREATE INDEX role_id_cast_info ON cast_info(role_id);
"""

TABLES = ["aka_name", "aka_title", "cast_info", "char_name",
          "comp_cast_type", "company_name", "company_type", "complete_cast",
          "info_type", "keyword", "kind_type", "link_type",
          "movie_companies", "movie_info", "movie_info_idx", "movie_keyword",
          "movie_link", "name", "person_info", "role_type", "title"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(url, dest):
    log(f"Downloading {url}")
    log("(this is ~1.2 GB for the dataset; progress dots every 50 MB)")
    done = [0]

    def hook(blocks, bs, total):
        done[0] += bs
        if done[0] % (50 * 1024 * 1024) < bs:
            print(".", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()
    log(f"Saved to {dest} ({os.path.getsize(dest)/1e9:.2f} GB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--db", default="imdb")
    parser.add_argument("--imdb-url", default=IMDB_URL)
    parser.add_argument("--imdb-tgz", default=None,
                        help="path to an already-downloaded imdb.tgz")
    parser.add_argument("--queries-dir", default="job_queries",
                        help="where to store the 113 JOB .sql files")
    args = parser.parse_args()

    import psycopg2

    t_start = time.time()
    tmpdir = tempfile.mkdtemp(prefix="job_setup_")

    # ---------------- 1. dataset ----------------
    tgz_path = args.imdb_tgz or os.path.join(tmpdir, "imdb.tgz")
    if not (args.imdb_tgz and os.path.exists(tgz_path)):
        try:
            download(args.imdb_url, tgz_path)
        except Exception as e:
            log(f"DOWNLOAD FAILED: {e}")
            log("Download imdb.tgz manually (search: 'JOB imdb.tgz cwi') and")
            log("re-run with --imdb-tgz <path>. Stopping.")
            return

    log("Extracting CSVs...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(tmpdir)
    csv_dir = tmpdir
    # csvs may be at top level or in a subfolder
    for root, _, files in os.walk(tmpdir):
        if "title.csv" in files:
            csv_dir = root
            break
    log(f"CSV directory: {csv_dir}")

    # ---------------- 2. queries ----------------
    os.makedirs(args.queries_dir, exist_ok=True)
    try:
        qtar_path = os.path.join(tmpdir, "job_queries.tgz")
        download(QUERIES_URL, qtar_path)
        n_q = 0
        with tarfile.open(qtar_path, "r:gz") as tar:
            for member in tar.getmembers():
                base = os.path.basename(member.name)
                if (base.endswith(".sql") and base[0].isdigit()):
                    content = tar.extractfile(member).read()
                    with open(os.path.join(args.queries_dir, base), "wb") as f:
                        f.write(content)
                    n_q += 1
        log(f"Saved {n_q} JOB query files to {args.queries_dir}/")
    except Exception as e:
        log(f"Query download failed ({e}) - you can fetch the "
            f"join-order-benchmark repo manually later; the data load "
            f"continues.")

    # ---------------- 3. database ----------------
    admin = psycopg2.connect(dbname="postgres", user=args.user,
                             password=args.password, host=args.host,
                             port=args.port)
    admin.autocommit = True
    acur = admin.cursor()
    acur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (args.db,))
    if not acur.fetchone():
        acur.execute(f'CREATE DATABASE "{args.db}";')
        log(f"Created database '{args.db}'")
    acur.close(); admin.close()

    conn = psycopg2.connect(dbname=args.db, user=args.user,
                            password=args.password, host=args.host,
                            port=args.port)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET maintenance_work_mem = '512MB';")
    log("Creating schema (21 tables)...")
    cur.execute(DDL)

    # ---------------- 4. load ----------------
    # JOB csvs use backslash-escaped quotes: FORMAT csv, ESCAPE '\'
    for t in TABLES:
        path = os.path.join(csv_dir, f"{t}.csv")
        if not os.path.exists(path):
            log(f"  MISSING {path} - skipped (report this)")
            continue
        t0 = time.time()
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            cur.copy_expert(
                f"COPY {t} FROM STDIN WITH (FORMAT csv, DELIMITER ',', "
                f"QUOTE '\"', ESCAPE '\\', NULL '')", f)
        cur.execute(f"SELECT count(*) FROM {t};")
        n = cur.fetchone()[0]
        log(f"  loaded {t:<18} {n:>12,} rows in {time.time()-t0:.1f}s")

    # ---------------- 5. indexes + stats ----------------
    log("Creating foreign-key indexes...")
    for stmt in [s.strip() for s in FK_INDEXES.split(";") if s.strip()]:
        t0 = time.time()
        cur.execute(stmt + ";")
        log(f"  {stmt.split()[2]:<38} {time.time()-t0:.1f}s")
    log("Running ANALYZE...")
    cur.execute("ANALYZE;")

    cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()));")
    log(f"Final database size: {cur.fetchone()[0]}")

    # sanity: run JOB query 1a shape
    t0 = time.time()
    cur.execute("""
        SELECT MIN(mc.note), MIN(t.title), MIN(t.production_year)
        FROM company_type ct, info_type it, movie_companies mc,
             movie_info_idx mi_idx, title t
        WHERE ct.kind = 'production companies' AND it.info = 'top 250 rank'
          AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
          AND (mc.note LIKE '%(co-production)%' OR mc.note LIKE '%(presents)%')
          AND ct.id = mc.company_type_id AND t.id = mc.movie_id
          AND t.id = mi_idx.movie_id AND mc.movie_id = mi_idx.movie_id
          AND it.id = mi_idx.info_type_id;
    """)
    log(f"Sanity check (JOB 1a): {cur.fetchone()} in {time.time()-t0:.2f}s")

    cur.close(); conn.close()

    # cleanup temp (keep queries dir)
    log("Cleaning temp files...")
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    log(f"ALL DONE in {(time.time()-t_start)/60:.1f} min.")
    log("Paste the full output back.")


if __name__ == "__main__":
    main()
