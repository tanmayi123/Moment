"""
bq_writer.py — Write processed data to BigQuery
=================================================
All tables → WRITE_APPEND (always add new, never delete existing)
Books → deduplicated against existing BQ rows before writing
"""

import logging
import os
from typing import List

import pandas as pd
from google.cloud import bigquery

logger = logging.getLogger(__name__)

BQ_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "moment-486719")
BQ_DATASET = os.environ.get("BQ_DATASET", "new_moments_processed")

TABLE_MAP = {
    "moments":  "moments_processed",
    "passages": "passages_processed",
    "books":    "books_processed",
    "users":    "users_processed",
}


def _get_client():
    return bigquery.Client(project=BQ_PROJECT)


def _write(client, df: pd.DataFrame, table_name: str) -> str:
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{table_name}"
    job = client.load_table_from_dataframe(
        df, table_id,
        job_config=bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            autodetect=True,
        )
    )
    job.result()
    logger.info(f"  ✓ {len(df)} rows appended → {table_id}")
    return table_id


def write_to_bq(
    moments:  List[dict],
    passages: List[dict],
    books:    List[dict],
    users:    List[dict],
) -> List[str]:
    client  = _get_client()
    written = []

    # ── Deduplicate books against existing BQ rows ─────────────
    try:
        existing = client.query(
            f"SELECT DISTINCT book_id FROM `{BQ_PROJECT}.{BQ_DATASET}.books_processed`"
        ).result()
        existing_ids = {row.book_id for row in existing}
        books = [b for b in books if b["book_id"] not in existing_ids]
        logger.info(f"  Books after dedup: {len(books)} new")
    except Exception as e:
        logger.warning(f"  Could not deduplicate books (writing all): {e}")

    for data, key in [
        (moments,  "moments"),
        (passages, "passages"),
        (books,    "books"),
        (users,    "users"),
    ]:
        if not data:
            logger.warning(f"  Skipping {key} — empty")
            continue
        df       = pd.DataFrame(data)
        table_id = _write(client, df, TABLE_MAP[key])
        written.append(table_id)

    logger.info(f"BQ write complete — {len(written)} tables")
    return written