"""
main.py — MOMENT FastAPI Data Pipeline
=======================================
Incremental pipeline using BQ watermark:
  POST /pipeline/run
    1. Read watermark from BQ (last processed created_at)
    2. Cloud SQL → fetch only NEW moments (created_at > watermark)
    3. Preprocess → write to BQ
    4. BQ moments_processed → compatibility agent → write to BQ
    5. Update watermark in BQ
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional
from itertools import combinations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import bigquery

from cloudsql_loader import CloudSQLLoader
from preprocessor_fastapi import preprocess_all
from bq_writer import write_to_bq
from tools import get_moments_for_passage
from compatibility_agent import run_compatibility_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MOMENT Data Pipeline",
    description="Incremental: Cloud SQL (new moments only) → Preprocess → BQ → Compatibility → BQ",
    version="3.0.0",
)

BQ_PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT", "moment-486719")
BQ_DATASET  = os.environ.get("BQ_DATASET", "new_moments_processed")
WATERMARK_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.pipeline_watermark"

_last_run: dict = {}


# ── Watermark helpers ─────────────────────────────────────────────────────────

def _get_watermark() -> Optional[str]:
    """Read last processed timestamp from BQ. Returns ISO string or None."""
    client = bigquery.Client(project=BQ_PROJECT)
    try:
        rows = list(client.query(
            f"SELECT last_processed_at FROM `{WATERMARK_TABLE}` ORDER BY last_processed_at DESC LIMIT 1"
        ).result())
        if rows:
            val = rows[0]["last_processed_at"]
            return val.isoformat() if hasattr(val, 'isoformat') else str(val)
        return None
    except Exception as e:
        logger.warning(f"Could not read watermark (table may not exist yet): {e}")
        return None


def _set_watermark(ts: str):
    """Upsert the watermark timestamp into BQ."""
    client = bigquery.Client(project=BQ_PROJECT)
    try:
        # Create table if not exists
        client.query(f"""
            CREATE TABLE IF NOT EXISTS `{WATERMARK_TABLE}` (
                last_processed_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """).result()

        # Delete old row and insert new
        client.query(f"DELETE FROM `{WATERMARK_TABLE}` WHERE TRUE").result()
        client.query(f"""
            INSERT INTO `{WATERMARK_TABLE}` (last_processed_at, updated_at)
            VALUES (TIMESTAMP('{ts}'), CURRENT_TIMESTAMP())
        """).result()
        logger.info(f"Watermark updated to {ts}")
    except Exception as e:
        logger.error(f"Failed to update watermark: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/pipeline/watermark")
def get_watermark():
    """Check current watermark (last processed timestamp)."""
    wm = _get_watermark()
    return {"watermark": wm or "none — full run will execute"}


@app.post("/pipeline/run")
def run_pipeline(full: bool = False):
    """
    Incremental pipeline — only processes moments newer than last run.
    Pass ?full=true to reprocess everything from scratch.
    """
    global _last_run
    start = datetime.utcnow()
    logger.info("=" * 60)

    # ── Read watermark ────────────────────────────────────────────
    watermark = None if full else _get_watermark()
    if watermark:
        logger.info(f"Incremental run — processing moments since {watermark}")
    else:
        logger.info("Full run — processing all moments")

    # ── Step 1: Load from Cloud SQL ───────────────────────────────
    logger.info("Step 1/4: Loading from Cloud SQL...")
    try:
        loader = CloudSQLLoader(since=watermark)
        loader.run()
        dfs = loader.get_dataframes()
    except Exception as e:
        logger.error(f"Cloud SQL load failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cloud SQL load failed: {str(e)}")

    new_count = len(dfs["interpretations_train"])
    if new_count == 0:
        logger.info("No new moments since last run — pipeline skipped")
        return {
            "status":        "skipped",
            "reason":        "no new moments since last run",
            "watermark":     watermark,
            "duration_sec":  round((datetime.utcnow() - start).total_seconds(), 2),
        }

    logger.info(f"  {new_count} new moments to process")

    # ── Step 2: Preprocess ────────────────────────────────────────
    logger.info("Step 2/4: Preprocessing...")
    try:
        moments, books, users = preprocess_all(
            interpretations_df=dfs["interpretations_train"],
            passages_df=dfs["passage_details_new"],
            users_df=dfs["user_details_new"],
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    valid_moments = [m for m in moments if m.get("is_valid", False)]
    logger.info(f"  {len(moments)} processed ({len(valid_moments)} valid)")

    # ── Step 3: Write to BQ ───────────────────────────────────────
    logger.info("Step 3/4: Writing to BQ...")
    try:
        bq_tables = write_to_bq(moments, books, users)
    except Exception as e:
        logger.error(f"BQ write failed: {e}")
        raise HTTPException(status_code=500, detail=f"BQ write failed: {str(e)}")

    # ── Step 4: Compatibility ─────────────────────────────────────
    logger.info("Step 4/4: Running compatibility agent...")
    compat_runs   = 0
    compat_errors = 0

    # For each new moment, run compatibility against all existing moments
    # on the same passage (fetched from BQ moments_processed)
    for m in valid_moments:
        user_id    = str(m.get("user_id", ""))
        passage_id = str(m.get("passage_id", ""))
        book_id    = str(m.get("book_id", ""))
        interp     = m.get("cleaned_interpretation", "")

        if not user_id or not passage_id or not interp:
            continue

        # Get all OTHER users with moments on same passage from BQ
        try:
            others = get_moments_for_passage(book_id, passage_id, exclude_user_id=user_id)
        except Exception as e:
            logger.warning(f"  Could not fetch passage moments: {e}")
            continue

        if not others:
            logger.info(f"  No other users on passage {passage_id} yet")
            continue

        for other in others:
            other_user_id = str(other["user_id"])
            other_text    = other.get("cleaned_interpretation", "")
            if not other_text:
                continue
            try:
                result = run_compatibility_agent(
                    user_a_id=user_id,
                    user_b_id=other_user_id,
                    book_id=book_id,
                    moment_a={"passage_id": passage_id, "cleaned_interpretation": interp},
                    moment_b={"passage_id": passage_id, "cleaned_interpretation": other_text},
                )
                if "error" in result:
                    logger.warning(f"  Compat error {user_id}×{other_user_id}: {result['error']}")
                    compat_errors += 1
                else:
                    compat_runs += 1
                    logger.info(f"  ✓ {user_id}×{other_user_id}: {result.get('dominant_think')} ({result.get('confidence')})")
            except Exception as e:
                logger.error(f"  Exception {user_id}×{other_user_id}: {e}")
                compat_errors += 1

    # ── Update watermark ──────────────────────────────────────────
    new_watermark = start.isoformat()
    _set_watermark(new_watermark)

    duration = (datetime.utcnow() - start).total_seconds()
    logger.info(f"Pipeline complete in {duration:.2f}s")
    logger.info("=" * 60)

    result = {
        "status":        "success",
        "timestamp":     start.isoformat(),
        "watermark":     new_watermark,
        "moments_count": len(moments),
        "books_count":   len(books),
        "users_count":   len(users),
        "valid_moments": len(valid_moments),
        "bq_tables":     bq_tables,
        "compat_runs":   compat_runs,
        "compat_errors": compat_errors,
        "duration_sec":  round(duration, 2),
    }
    _last_run = result
    return result


@app.get("/pipeline/status")
def pipeline_status():
    if not _last_run:
        return {"status": "no_runs_yet"}
    return _last_run