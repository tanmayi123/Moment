"""
Momento ML DAG Pipeline
Stages: decompose → score → aggregate

BQ layout (per-run staging, mirroring data_processing_dag):
    {BQ_PROJECT}.moments_staging_{run_id}.decompositions
    {BQ_PROJECT}.moments_staging_{run_id}.scoring_runs
    {BQ_PROJECT}.moments_staging_{run_id}.scored_pairs_staging
    {BQ_PROJECT}.moments_staging_{run_id}.compatibility_results

Final destination:
    {BQ_PROJECT}.new_moments_processed.compatibility_results

XCom usage: tasks push/pull only BQ table IDs (tiny strings), never raw records.

Flow:
    decompose → score → aggregate → upload

Can be imported as a library (stage_* functions) or run as an Airflow DAG.
Run standalone: python momento_dag.py
"""

import json
import itertools
import logging
import os
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps

import pandas as pd

# ════════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════════

logger = logging.getLogger("momento.ml")

LOGS_DIR = os.environ.get("LOGS_DIR", "/opt/airflow/logs")
os.makedirs(LOGS_DIR, exist_ok=True)

def _make_handler(path, level, filter_fn=None):
    h = logging.FileHandler(path)
    h.setLevel(level)
    if filter_fn:
        h.addFilter(filter_fn)
    h.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    return h

logger.addHandler(_make_handler(os.path.join(LOGS_DIR, "ml_INFO.log"),    logging.INFO,    lambda r: r.levelno == logging.INFO))
logger.addHandler(_make_handler(os.path.join(LOGS_DIR, "ml_WARNING.log"), logging.WARNING, lambda r: r.levelno == logging.WARNING))
logger.addHandler(_make_handler(os.path.join(LOGS_DIR, "ml_ERROR.log"),   logging.ERROR))
logger.addHandler(_make_handler(os.path.join(LOGS_DIR, "ml_ALL.log"),     logging.DEBUG))


# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════

BQ_PROJECT       = os.environ.get("GOOGLE_CLOUD_PROJECT", "moment-486719")
BQ_FINAL_DATASET = "new_moments_processed"

# How many decompositions to accumulate before flushing to BQ.
# Lower = more BQ writes but more crash-resilient.
# Higher = fewer writes but more work lost on failure.
DECOMP_BATCH_SIZE = int(os.environ.get("DECOMP_BATCH_SIZE", "5"))

def _ensure_ml_path():
    """Add ML_SCRIPTS_DIR to sys.path. Call at the top of every task function."""
    import sys
    ml_dir = os.environ.get("ML_SCRIPTS_DIR", "/opt/airflow/ml_pipeline/scripts")
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
        logger.debug(f"Added to sys.path: {ml_dir}")


# ════════════════════════════════════════════════════════════════
#  BIGQUERY HELPERS
# ════════════════════════════════════════════════════════════════

def _bq_client():
    from google.cloud import bigquery
    return bigquery.Client(project=BQ_PROJECT)

def staging_dataset(run_id: str) -> str:
    safe = (run_id.replace("-", "_").replace(":", "_")
                  .replace("+", "_").replace(".", "_").replace("T", "_"))
    return f"moments_staging_{safe}"[:1024]

def bq_table_id(run_id: str, table: str) -> str:
    return f"{BQ_PROJECT}.{staging_dataset(run_id)}.{table}"

def ensure_staging_dataset(run_id: str):
    from google.cloud import bigquery
    client = _bq_client()
    ds_id  = f"{BQ_PROJECT}.{staging_dataset(run_id)}"
    ds     = bigquery.Dataset(ds_id)
    ds.location = "US"
    client.create_dataset(ds, exists_ok=True)
    logger.debug(f"Staging dataset ready: {ds_id}")

def bq_write(df: pd.DataFrame, table_id: str):
    from google.cloud import bigquery
    job = _bq_client().load_table_from_dataframe(
        df, table_id,
        job_config=bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            autodetect=True,
        ),
    )
    job.result()
    logger.debug(f"Wrote {len(df)} rows → {table_id}")

def bq_append(df: pd.DataFrame, table_id: str):
    """Append rows to an existing BQ table without truncating it."""
    from google.cloud import bigquery
    job = _bq_client().load_table_from_dataframe(
        df, table_id,
        job_config=bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            autodetect=True,
        ),
    )
    job.result()
    logger.debug(f"Appended {len(df)} rows → {table_id}")

def bq_read(table_id: str) -> pd.DataFrame:
    df = _bq_client().query(f"SELECT * FROM `{table_id}`").to_dataframe()
    logger.debug(f"Read {len(df)} rows ← {table_id}")
    return df

def bq_copy_table(src: str, dst: str):
    from google.cloud import bigquery
    job = _bq_client().copy_table(
        src, dst,
        job_config=bigquery.CopyJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ),
    )
    job.result()
    logger.debug(f"Copied {src} → {dst}")


# ════════════════════════════════════════════════════════════════
#  DECORATOR
# ════════════════════════════════════════════════════════════════

def log_task_execution(task_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = datetime.now()
            logger.info("=" * 70)
            logger.info(f"🚀 Starting task: {task_name}  [{start:%Y-%m-%d %H:%M:%S}]")
            try:
                result = func(*args, **kwargs)
                dur = (datetime.now() - start).total_seconds()
                if dur > 300:
                    logger.warning(f"⚠️  {task_name} took {dur:.1f}s (>5 min)")
                logger.info(f"✅ Completed: {task_name}  [{dur:.2f}s]")
                logger.info("=" * 70)
                return result
            except Exception as e:
                dur = (datetime.now() - start).total_seconds()
                logger.error(f"❌ Failed: {task_name} — {type(e).__name__}: {e}  [{dur:.2f}s]")
                logger.exception("Full traceback:")
                raise
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def _fmt(d: dict) -> str:
    return (f"user={d.get('user_id','?')} "
            f"passage={d.get('passage_id','?')} "
            f"book={d.get('book_id','?')}")

def _pair_tag(uid_a: str, uid_b: str, passage_id: str) -> str:
    return f"{uid_a}×{uid_b} / {passage_id}"

def _load_moments(run_id: str, books: list | None, passage_ids: list | None) -> list[dict]:
    """Read moments from the data pipeline's final BQ table and apply filters."""
    src = f"{BQ_PROJECT}.{BQ_FINAL_DATASET}.moments_processed"
    logger.info(f"Loading moments from {src}")
    df = bq_read(src)
    logger.info(f"Loaded {len(df)} moment(s) from BQ")
    if books:
        df = df[df["book_id"].isin(books)]
        logger.info(f"After book filter ({books}): {len(df)} moment(s)")
    if passage_ids:
        df = df[df["passage_id"].isin(passage_ids)]
        logger.info(f"After passage filter ({passage_ids}): {len(df)} moment(s)")
    return df.to_dict("records")


def _flush_decomp_batch(batch: list[dict], cache_tid: str, is_first_flush: bool) -> None:
    """
    Write a batch of decompositions to BQ.
    First flush uses WRITE_TRUNCATE to reset the table for this run;
    subsequent flushes append so earlier batches are not lost.
    """
    if not batch:
        return
    df = pd.DataFrame(batch)
    if is_first_flush:
        bq_write(df, cache_tid)          # WRITE_TRUNCATE — clears stale data
        logger.info("[decompose] initial flush: wrote %d row(s) → %s", len(batch), cache_tid)
    else:
        bq_append(df, cache_tid)         # WRITE_APPEND — preserve earlier batches
        logger.info("[decompose] batch flush: appended %d row(s) → %s", len(batch), cache_tid)


# ════════════════════════════════════════════════════════════════
#  STAGE 1 — DECOMPOSE  (batched BQ writes)
# ════════════════════════════════════════════════════════════════

def stage_decompose(moments: list[dict], run_id: str,
                    batch_size: int = DECOMP_BATCH_SIZE) -> dict[tuple, dict]:
    """
    Decompose each moment into weighted sub-claims.

    Checks the run-scoped staging table for cached results first so that
    a re-run after a partial failure skips already-completed moments.

    New decompositions are flushed to BQ every `batch_size` completions
    (default 5) so that progress is saved incrementally — a crash mid-run
    loses at most `batch_size` decompositions rather than everything.

    Returns dict keyed by (user_id, passage_id, book_id).
    """
    _ensure_ml_path()
    from decomposing_agent import run_decomposer

    logger.info("── Stage 1: Decompose ── %d moment(s)  batch_size=%d",
                len(moments), batch_size)

    cache_tid = bq_table_id(run_id, "decompositions")

    # ── Load cache — check final table first, then staging ───────────────────
    # Priority: new_moments_processed (cross-run) > staging (this run only).
    # This means a moment decomposed in any previous run is never re-run.
    existing: dict[tuple, dict] = {}

    final_tid = f"{BQ_PROJECT}.{BQ_FINAL_DATASET}.decompositions"
    for label, tid in [("final", final_tid), ("staging", cache_tid)]:
        try:
            df = bq_read(tid)
            found = {
                (r["user_id"], r["passage_id"], r["book_id"]): r
                for r in df.to_dict("records")
                if "error" not in r
            }
            # staging can add to final but never overwrite it
            merged = {**found, **existing} if label == "staging" else found
            new_keys = len(merged) - len(existing)
            logger.info("[decompose] cache %s: %d total  %d new key(s)",
                        label, len(found), new_keys)
            existing = merged
        except Exception:
            logger.info("[decompose] cache %s: not found or empty — skipping", label)

    logger.info("[decompose] combined cache: %d valid decomposition(s)", len(existing))

    results:    dict[tuple, dict] = dict(existing)   # pre-populate with cached
    batch:      list[dict]        = []
    is_first_flush                = True              # first write truncates, rest append
    n_hit = n_run = n_fail = 0

    # ── Process moments ───────────────────────────────────────────────────────
    todo = [m for m in moments
            if (m["user_id"], m["passage_id"], m["book_id"]) not in existing]
    cached_count = len(moments) - len(todo)
    n_hit = cached_count
    logger.info("[decompose] %d cache hits  %d to run", cached_count, len(todo))

    for i, m in enumerate(todo, 1):
        key = (m["user_id"], m["passage_id"], m["book_id"])
        logger.info("[decompose] [%d/%d] running  %s", i, len(todo), _fmt(m))

        try:
            decomp = run_decomposer(m)
        except Exception:
            logger.error("[decompose] EXCEPTION  %s\n%s", _fmt(m), traceback.format_exc())
            n_fail += 1
            continue

        if "error" in decomp:
            logger.warning("[decompose] FAILED  %s — %s", _fmt(m), decomp["error"])
            if decomp.get("raw"):
                logger.debug("[decompose] raw: %s", str(decomp["raw"])[:400])
            n_fail += 1
            continue

        sc_count = len(decomp.get("subclaims", []))
        logger.info("[decompose] OK  %s — %d subclaim(s)", _fmt(m), sc_count)
        logger.debug("[decompose] subclaim ids: %s",
                     [s["id"] for s in decomp.get("subclaims", [])])

        results[key] = decomp
        batch.append(decomp)
        n_run += 1

        # ── Flush batch to BQ ─────────────────────────────────────────────────
        if len(batch) >= batch_size:
            _flush_decomp_batch(batch, cache_tid, is_first_flush)
            is_first_flush = False
            batch = []

    # ── Final flush for any remainder ─────────────────────────────────────────
    if batch:
        _flush_decomp_batch(batch, cache_tid, is_first_flush)

    logger.info("[decompose] summary — hit=%d  ran=%d  failed=%d  total=%d",
                n_hit, n_run, n_fail, len(moments))

    return results


# ════════════════════════════════════════════════════════════════
#  STAGE 2 — SCORE
# ════════════════════════════════════════════════════════════════

def stage_score(
    pairs: list[tuple[dict, dict]],
    decompositions: dict[tuple, dict],
    run_id: str,
) -> list[dict]:
    """
    Score each (moment_a, moment_b) pair via the compatibility scorer.
    Caches scoring runs in BQ staging. Returns list of scored dicts
    (with _decomp_a / _decomp_b attached for the aggregate stage).
    """
    _ensure_ml_path()
    from compatibility_agent import _build_scorer_prompt, _call_scorer

    logger.info("── Stage 2: Score ── %d pair(s) received", len(pairs))

    score_cache_tid = bq_table_id(run_id, "scoring_runs")
    existing_scores: dict[tuple, dict] = {}
    try:
        score_df = bq_read(score_cache_tid)
        existing_scores = {
            (r["user_a_id"], r["user_b_id"], r["passage_id"]): r
            for r in score_df.to_dict("records")
        }
        logger.info("[score] BQ cache loaded: %d existing scoring run(s)", len(existing_scores))
    except Exception:
        logger.info("[score] No existing scoring cache (first run or table absent)")

    results:        list[dict] = []
    new_score_rows: list[dict] = []
    n_hit = n_run = n_fail = n_skip = 0

    for m_a, m_b in pairs:
        uid_a, uid_b  = m_a["user_id"], m_b["user_id"]
        passage_id    = m_a["passage_id"]
        book_id       = m_a["book_id"]
        tag           = _pair_tag(uid_a, uid_b, passage_id)
        key_a         = (uid_a, passage_id, book_id)
        key_b         = (uid_b, passage_id, book_id)
        score_key     = (uid_a, uid_b, passage_id)
        score_key_rev = (uid_b, uid_a, passage_id)

        decomp_a = decompositions.get(key_a)
        decomp_b = decompositions.get(key_b)

        if not decomp_a or not decomp_b:
            missing = [u for u, d in [(uid_a, decomp_a), (uid_b, decomp_b)] if not d]
            logger.warning("[score] SKIP  %s — missing decomposition(s) for: %s",
                           tag, ", ".join(missing))
            n_skip += 1
            continue

        if score_key in existing_scores or score_key_rev in existing_scores:
            cached = existing_scores.get(score_key) or existing_scores.get(score_key_rev)
            logger.info("[score] cache hit  %s", tag)
            results.append({**cached["scoring"],
                            "user_a": uid_a, "user_b": uid_b,
                            "book_id": book_id, "passage_id": passage_id,
                            "_decomp_a": decomp_a, "_decomp_b": decomp_b})
            n_hit += 1
            continue

        logger.info("[score] running    %s", tag)
        logger.debug("[score] decomp_a subclaims=%d  decomp_b subclaims=%d",
                     len(decomp_a.get("subclaims", [])), len(decomp_b.get("subclaims", [])))

        try:
            prompt  = _build_scorer_prompt(decomp_a, decomp_b)
            scoring = _call_scorer(prompt)
        except Exception:
            logger.error("[score] EXCEPTION  %s\n%s", tag, traceback.format_exc())
            n_fail += 1
            continue

        new_score_rows.append({
            "user_a_id":  uid_a,
            "user_b_id":  uid_b,
            "passage_id": passage_id,
            "timestamp":  datetime.utcnow().isoformat(),
            "scoring":    json.dumps(scoring),
        })

        if "error" in scoring:
            logger.warning("[score] FAILED     %s — %s", tag, scoring["error"])
            if scoring.get("raw"):
                logger.debug("[score] raw scorer output: %s", str(scoring["raw"])[:400])
            n_fail += 1
            continue

        n_matched   = len(scoring.get("matched_pairs", []))
        n_unmatched = (len(scoring.get("unmatched_a", [])) +
                       len(scoring.get("unmatched_b", [])))
        logger.info("[score] OK  %s — %d matched  %d unmatched", tag, n_matched, n_unmatched)
        logger.debug("[score] matched pair ids: %s",
                     [(p["a_id"], p["b_id"]) for p in scoring.get("matched_pairs", [])])

        results.append({**scoring,
                        "user_a": uid_a, "user_b": uid_b,
                        "book_id": book_id, "passage_id": passage_id,
                        "_decomp_a": decomp_a, "_decomp_b": decomp_b})
        n_run += 1

    logger.info("[score] summary — hit=%d  ran=%d  failed=%d  skipped=%d  total=%d",
                n_hit, n_run, n_fail, n_skip, len(pairs))

    if new_score_rows:
        all_rows = list(existing_scores.values()) + new_score_rows
        bq_write(pd.DataFrame(all_rows), score_cache_tid)
        logger.info("[score] BQ write complete — %d row(s) → %s",
                    len(new_score_rows), score_cache_tid)

    return results


# ════════════════════════════════════════════════════════════════
#  STAGE 3 — AGGREGATE
# ════════════════════════════════════════════════════════════════

def stage_aggregate(scored_pairs: list[dict], run_id: str) -> list[dict]:
    """
    Aggregate scored pairs into final R/C/D compatibility results.
    Writes results to BQ staging.
    """
    _ensure_ml_path()
    from aggregator import aggregate

    logger.info("── Stage 3: Aggregate ── %d scored pair(s) received", len(scored_pairs))

    final_results: list[dict] = []
    n_ok = n_fail = 0

    for s in scored_pairs:
        uid_a    = s["user_a"]
        uid_b    = s["user_b"]
        tag      = _pair_tag(uid_a, uid_b, s["passage_id"])
        decomp_a = s.pop("_decomp_a")
        decomp_b = s.pop("_decomp_b")

        logger.debug("[aggregate] processing %s  matched_pairs=%d",
                     tag, len(s.get("matched_pairs", [])))

        try:
            result = aggregate({"reader_a": decomp_a, "reader_b": decomp_b}, s)
        except Exception:
            logger.error("[aggregate] EXCEPTION  %s\n%s", tag, traceback.format_exc())
            n_fail += 1
            continue

        result.update({
            "user_a":    uid_a,
            "user_b":    uid_b,
            "book_id":   s["book_id"],
            "timestamp": datetime.utcnow().isoformat(),
        })

        logger.info(
            "[aggregate] OK  %s — "
            "think=%s(R=%d C=%d D=%d)  feel=%s(R=%d C=%d D=%d)  "
            "conf=%.2f  matches=%d",
            tag,
            result["dominant_think"],
            result["think"]["R"], result["think"]["C"], result["think"]["D"],
            result["dominant_feel"],
            result["feel"]["R"],  result["feel"]["C"],  result["feel"]["D"],
            result["confidence"], result["match_count"],
        )
        final_results.append(result)
        n_ok += 1

    logger.info("[aggregate] summary — ok=%d  failed=%d  total=%d",
                n_ok, n_fail, len(scored_pairs))

    if final_results:
        staging_tid = bq_table_id(run_id, "compatibility_results")
        bq_write(pd.DataFrame(final_results), staging_tid)
        logger.info("[aggregate] wrote %d result(s) → %s", len(final_results), staging_tid)
    else:
        logger.warning("[aggregate] no results to write — all pairs failed")

    return final_results


# ════════════════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATOR  (standalone / library use)
# ════════════════════════════════════════════════════════════════

def run_pipeline(
    books:       list[str] | None = None,
    passage_ids: list[str] | None = None,
    run_id:      str | None = None,
    batch_size:  int = DECOMP_BATCH_SIZE,
) -> list[dict]:
    run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger.info("════ run_pipeline start  run_id=%s  batch_size=%d ════",
                run_id, batch_size)
    logger.info("filters — books=%s  passage_ids=%s", books, passage_ids)

    ensure_staging_dataset(run_id)

    moments = _load_moments(run_id, books, passage_ids)
    if not moments:
        logger.error("No moments found after filtering — aborting")
        return []

    decompositions = stage_decompose(moments, run_id, batch_size=batch_size)

    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for m in moments:
        grouped[(m["book_id"], m["passage_id"])][m["user_id"]] = m

    all_pairs = [
        (m_a, m_b)
        for user_map in grouped.values()
        for m_a, m_b in itertools.combinations(user_map.values(), 2)
    ]
    logger.info("Built %d unique pair(s) across %d (book, passage) group(s)",
                len(all_pairs), len(grouped))

    scored_pairs = stage_score(all_pairs, decompositions, run_id)
    final        = stage_aggregate(scored_pairs, run_id)

    logger.info("════ run_pipeline complete  run_id=%s — %d result(s) ════",
                run_id, len(final))
    return final


# ════════════════════════════════════════════════════════════════
#  AIRFLOW TASK CALLABLES
# ════════════════════════════════════════════════════════════════

@log_task_execution("ML Path Debug")
def task_debug_path(**context):
    """Temporary task — remove once imports are confirmed working."""
    import sys
    ml_dir = os.environ.get("ML_SCRIPTS_DIR", "/opt/airflow/scripts")
    logger.info(f"ML_SCRIPTS_DIR env = {ml_dir}")
    logger.info(f"sys.path = {sys.path}")
    try:
        files = os.listdir(ml_dir)
        logger.info(f"Files in {ml_dir}: {files}")
    except FileNotFoundError:
        logger.error(f"❌ Directory does not exist: {ml_dir}")
    except Exception as e:
        logger.error(f"❌ Could not list {ml_dir}: {e}")
    try:
        import importlib.util
        spec = importlib.util.find_spec("decomposing_agent")
        logger.info(f"decomposing_agent spec: {spec}")
    except Exception as e:
        logger.error(f"find_spec failed: {e}")


@log_task_execution("ML Decompose")
def task_decompose(**context):
    ti      = context["task_instance"]
    run_id  = context["run_id"]
    conf    = context["dag_run"].conf or {}
    books       = conf.get("books")
    passage_ids = conf.get("passage_ids")
    batch_size  = int(conf.get("batch_size", DECOMP_BATCH_SIZE))

    logger.info(f"[airflow][decompose] run_id={run_id}  books={books}  "
                f"passage_ids={passage_ids}  batch_size={batch_size}")

    ensure_staging_dataset(run_id)
    moments = _load_moments(run_id, books, passage_ids)

    if not moments:
        raise ValueError("[ml_dag][decompose] No moments found after filtering")
    logger.info(f"[airflow][decompose] {len(moments)} moment(s) to decompose")

    stage_decompose(moments, run_id, batch_size=batch_size)

    moments_tid = bq_table_id(run_id, "moments_processed")
    bq_write(pd.DataFrame(moments), moments_tid)
    logger.info(f"[airflow][decompose] moments snapshot → {moments_tid}")

    ti.xcom_push(key="moments_tid",     value=moments_tid)
    ti.xcom_push(key="decomp_tid",      value=bq_table_id(run_id, "decompositions"))
    ti.xcom_push(key="moment_count",    value=len(moments))


@log_task_execution("ML Score")
def task_score(**context):
    ti     = context["task_instance"]
    run_id = context["run_id"]

    moments_tid = ti.xcom_pull(task_ids="decompose", key="moments_tid")
    decomp_tid  = ti.xcom_pull(task_ids="decompose", key="decomp_tid")
    logger.info(f"[airflow][score] moments_tid={moments_tid}  decomp_tid={decomp_tid}")

    moments = bq_read(moments_tid).to_dict("records")
    decompositions = {
        (r["user_id"], r["passage_id"], r["book_id"]): r
        for r in bq_read(decomp_tid).to_dict("records")
        if "error" not in r
    }
    logger.info(f"[airflow][score] {len(moments)} moments  "
                f"{len(decompositions)} decompositions loaded")

    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for m in moments:
        grouped[(m["book_id"], m["passage_id"])][m["user_id"]] = m

    pairs = [
        (m_a, m_b)
        for user_map in grouped.values()
        for m_a, m_b in itertools.combinations(user_map.values(), 2)
    ]
    logger.info(f"[airflow][score] {len(pairs)} pair(s) across {len(grouped)} group(s)")

    scored = stage_score(pairs, decompositions, run_id)
    logger.info(f"[airflow][score] {len(scored)} pair(s) scored")

    staging_rows = []
    for s in scored:
        row = {k: v for k, v in s.items() if not k.startswith("_")}
        row["decomp_a"] = json.dumps(s["_decomp_a"])
        row["decomp_b"] = json.dumps(s["_decomp_b"])
        staging_rows.append(row)

    staging_tid = bq_table_id(run_id, "scored_pairs_staging")
    bq_write(pd.DataFrame(staging_rows), staging_tid)
    logger.info(f"[airflow][score] staged {len(staging_rows)} row(s) → {staging_tid}")

    ti.xcom_push(key="scored_staging_tid", value=staging_tid)
    ti.xcom_push(key="scored_count",       value=len(staging_rows))


@log_task_execution("ML Aggregate")
def task_aggregate(**context):
    ti     = context["task_instance"]
    run_id = context["run_id"]

    staging_tid  = ti.xcom_pull(task_ids="score", key="scored_staging_tid")
    scored_count = ti.xcom_pull(task_ids="score", key="scored_count") or "?"
    logger.info(f"[airflow][aggregate] staging_tid={staging_tid}  expected={scored_count}")

    staging_df = bq_read(staging_tid)
    logger.info(f"[airflow][aggregate] loaded {len(staging_df)} row(s) from staging")

    if staging_df.empty:
        raise ValueError("[ml_dag][aggregate] scored_pairs_staging is empty")

    scored_pairs = []
    for row in staging_df.to_dict("records"):
        row["_decomp_a"] = json.loads(row.pop("decomp_a"))
        row["_decomp_b"] = json.loads(row.pop("decomp_b"))
        scored_pairs.append(row)

    results = stage_aggregate(scored_pairs, run_id)
    logger.info(f"[airflow][aggregate] {len(results)} result(s) written to staging")

    ti.xcom_push(key="results_tid",   value=bq_table_id(run_id, "compatibility_results"))
    ti.xcom_push(key="results_count", value=len(results))


@log_task_execution("ML Upload to Final BQ")
def task_upload(**context):
    ti     = context["task_instance"]
    run_id = context["run_id"]

    # ── Table map: staging table → final destination ──────────────────────────
    tables = {
        "decompositions":        "decompositions",
        "compatibility_results": "compatibility_results",
    }

    copied  = []
    errors  = []

    for staging_table, final_table in tables.items():
        src = bq_table_id(run_id, staging_table)
        dst = f"{BQ_PROJECT}.{BQ_FINAL_DATASET}.{final_table}"
        logger.info(f"[airflow][upload] {src} → {dst}")
        try:
            bq_copy_table(src, dst)
            logger.info(f"[airflow][upload] ✓ copied {staging_table} → {dst}")
            copied.append(final_table)
        except Exception as e:
            logger.error(f"[airflow][upload] ❌ failed to copy {staging_table}: "
                         f"{type(e).__name__}: {e}")
            errors.append(staging_table)

    logger.info(f"[airflow][upload] done — copied={copied}  errors={errors}")
    if errors:
        raise RuntimeError(f"Upload failed for: {errors}")

    ti.xcom_push(key="copied_tables", value=copied)


# ════════════════════════════════════════════════════════════════
#  AIRFLOW DAG DEFINITION
# ════════════════════════════════════════════════════════════════

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    default_args = {
        "owner":             "momento",
        "depends_on_past":   False,
        "start_date":        datetime(2025, 2, 10),
        "email_on_failure":  True,
        "email_on_retry":    False,
        "retries":           1,
        "retry_delay":       timedelta(minutes=2),
        "execution_timeout": timedelta(minutes=30),
    }

    with DAG(
        dag_id="ml_dag",
        default_args=default_args,
        description="Momento ML pipeline: decompose → score → aggregate → upload",
        schedule=None,
        catchup=False,
        max_active_runs=1,
        tags=["momento", "ml", "compatibility"],
    ) as dag:

        t_debug     = PythonOperator(task_id="debug_path", python_callable=task_debug_path)
        t_decompose = PythonOperator(task_id="decompose",  python_callable=task_decompose)
        t_score     = PythonOperator(task_id="score",      python_callable=task_score)
        t_aggregate = PythonOperator(task_id="aggregate",  python_callable=task_aggregate)
        t_upload    = PythonOperator(task_id="upload",     python_callable=task_upload)

        t_debug >> t_decompose >> t_score >> t_aggregate >> t_upload

except ImportError:
    dag = None


# ════════════════════════════════════════════════════════════════
#  STANDALONE ENTRYPOINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_pipeline(
        books=["Frankenstein", "Pride and Prejudice", "The Great Gatsby"],
        passage_ids=["passage_1", "passage_2", "passage_3"],
    )
    print(json.dumps(results, indent=2))