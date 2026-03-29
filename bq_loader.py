"""
bq_loader.py
============
MOMENT Data Pipeline — BigQuery Integration

Handles ALL BigQuery operations for the pipeline:

  1. load_raw_from_bq()     — replaces GCS JSON reads in data_acquisition.py
                              Pulls raw interpretations, passages, and user data
                              from BQ staging tables directly into memory as dicts.

  2. write_processed_to_bq() — replaces write_outputs() local JSON writes in preprocessor.py
                               Upserts processed moments, books, and users into
                               their respective BQ tables after pipeline processing.

  3. upload_reports_to_bq()  — stores TFDV schema stats + validation + bias reports
                               in a BQ reports table so every run is queryable.

  4. get_bq_client()         — shared authenticated BigQuery client (lazy singleton).

Dataset layout in BigQuery (project: moment-486719):
  moment-486719.moment_raw.interpretations      ← raw user interpretations
  moment-486719.moment_raw.passages             ← raw passage details
  moment-486719.moment_raw.user_data            ← raw character/user data
  moment-486719.moment_processed.moments        ← processed moments (main ML table)
  moment-486719.moment_processed.books          ← processed book/passage records
  moment-486719.moment_processed.users          ← processed user profiles
  moment-486719.moment_reports.pipeline_runs    ← per-run pipeline stats & reports

All writes use WRITE_TRUNCATE by default (full refresh each pipeline run),
which is safe for this batch pipeline. Pass write_disposition='WRITE_APPEND'
to append incrementally instead.

Error handling:
  - All BQ operations are wrapped in try/except.
  - On failure: logs error, returns empty list / False.
  - Pipeline continues using local JSON fallback if BQ is unreachable.
  - Schema mismatches are auto-healed via autodetect=True on initial load.

Author: MOMENT Group 23 | IE7374 MLOps | Northeastern University
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from google.cloud import bigquery 
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ─── Project / Dataset constants ────────────────────────────────────────────

PROJECT_ID   = os.environ.get("MOMENT_GCP_PROJECTID", "")

# Raw staging tables (written by acquisition, read by preprocessing)
RAW_DATASET  = "moments_raw"
RAW_INTERP_TABLE   = "interpretations_raw"
RAW_PASSAGES_TABLE = "passage_details_new"
RAW_USERS_TABLE    = "user_details_new"

# Processed tables (written by preprocessing / post-pipeline)
PROC_DATASET        = "moment_processed"
PROC_MOMENTS_TABLE  = "moments"
PROC_PASSAGES_TABLE    = "passages"
PROC_USERS_TABLE    = "users"

# Reports table (written by schema_stats + validation + notify)
REPORTS_DATASET      = "moment_reports"
REPORTS_RUNS_TABLE   = "pipeline_runs"

# ─── Lazy BQ client singleton ────────────────────────────────────────────────

_bq_client = None


def get_bq_client():
    """
    Return a cached BigQuery client.

    Uses Application Default Credentials (ADC) — works inside GCP (Vertex AI,
    Cloud Composer / Airflow) and locally when `gcloud auth application-default
    login` has been run.

    Returns:
        google.cloud.bigquery.Client or None if unavailable.
    """
    global _bq_client
    if _bq_client is None:
        try:
            _bq_client = bigquery.Client(project=PROJECT_ID)
            logger.info(f"[bq_loader] BigQuery client initialized for project={PROJECT_ID}")
        except Exception as exc:
            logger.error(f"[bq_loader] Failed to initialize BigQuery client: {exc}")
            return None
    return _bq_client


# ─── Helper: ensure dataset exists ──────────────────────────────────────────

def _ensure_dataset(client, dataset_id: str, location: str = "US") -> bool:
    """Create BQ dataset if it does not already exist."""
    from google.cloud import bigquery  # type: ignore
    dataset_ref = f"{PROJECT_ID}.{dataset_id}"
    try:
        client.get_dataset(dataset_ref)
        return True
    except Exception:
        try:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = location
            client.create_dataset(dataset, exists_ok=True)
            logger.info(f"[bq_loader] Created dataset {dataset_ref}")
            return True
        except Exception as exc:
            logger.error(f"[bq_loader] Could not create dataset {dataset_ref}: {exc}")
            return False


# ─── Helper: flatten nested dicts/lists for BQ ──────────────────────────────

def _flatten_record(record: dict) -> dict:
    """
    Flatten a single record so it is BQ-compatible:
      - nested dicts  → JSON string
      - lists         → comma-joined string (for simple lists) or JSON string
      - None          → kept as None (BQ handles nullable columns)
    """
    flat = {}
    for k, v in record.items():
        if isinstance(v, dict):
            flat[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, list):
            if all(isinstance(i, str) for i in v):
                flat[k] = ", ".join(v)
            else:
                flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = v
    return flat


def _records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert a list of pipeline records to a BQ-ready DataFrame."""
    if not records:
        return pd.DataFrame()
    flat = [_flatten_record(r) for r in records]
    df = pd.DataFrame(flat)
    # BQ does not support column names with dots — replace with underscores
    df.columns = [c.replace(".", "_") for c in df.columns]
    return df


# ─── 1. Load raw data from BigQuery ─────────────────────────────────────────

def load_raw_from_bq(client, dataset: str, table_name:str) -> dict[str, list[dict]]:
    """
    Pull raw input tables from BigQuery into memory as lists of dicts.

    Replaces the GCS → local file download step in data_acquisition.py.
    The pipeline reads directly from BQ staging tables, which are pre-populated
    from GCS via the existing upload workflow (or manually loaded once).

    Returns:
        {
          "interpretations": [...],   # list of raw interpretation dicts
          "passages":        [...],   # list of passage dicts
          "user_data":       [...],   # list of user/character dicts
        }
        Any missing table returns an empty list; pipeline falls back to local files.
    """
    if client is None:
        logger.warning("[bq_loader] BQ client unavailable — returning empty raw data")
        return {"interpretations": [], "passages": [], "user_data": []}

    results: dict[str, list[dict]] = {}
    full_table=f"`{PROJECT_ID}.{dataset}.{table_name}`"

    try:
        query = f"SELECT * FROM {full_table}"
        logger.info(f"[bq_loader] Querying {full_table}...")
        df = client.query(query).to_dataframe()
        results = df.to_dict(orient="records")
        logger.info(f"[bq_loader] Loaded {len(results)} rows from {full_table}")
    except Exception as exc:
        logger.warning(f"[bq_loader] Could not read {full_table}: {exc}")
        results = []
    

    return results


# ─── 3. Write processed data to BigQuery ────────────────────────────────────

def write_processed_to_bq(client,
    records: list[dict],
    table_name:   list[dict],
    write_disposition: str = "WRITE_TRUNCATE",
) -> bool:
    """
    Write the three processed datasets (moments, books, users) to BigQuery.

    Called after preprocessor.write_outputs() so both local JSON files AND BQ
    are always in sync. If BQ write fails, local JSON is still intact.

    Args:
        moments:           list of processed moment records
        passages:             list of processed passage records
        users:             list of processed user profile records
        write_disposition: "WRITE_TRUNCATE" (default, full refresh per run)
                           or "WRITE_APPEND" (incremental)

    Returns:
        True if all three tables were written successfully.
    """
    if client is None:
        logger.warning("[bq_loader] BQ client unavailable — skipping processed write")
        return False

    _ensure_dataset(client, PROC_DATASET)

    all_success = True
    full_table = f"{PROJECT_ID}.{PROC_DATASET}.{table_name}"
    try:
        if not records:
            logger.warning(f"[bq_loader] No {table_name} records to write — skipping {full_table}")

        df = _records_to_dataframe(records)

        from google.cloud.bigquery import LoadJobConfig, WriteDisposition  # type: ignore

        wd = (WriteDisposition.WRITE_TRUNCATE
            if write_disposition == "WRITE_TRUNCATE"
            else WriteDisposition.WRITE_APPEND)

        job_config = LoadJobConfig(
                write_disposition=wd,
                autodetect=True,
        )
        job = client.load_table_from_dataframe(df, full_table, job_config=job_config)
        job.result()  # block until done
        logger.info(f"[bq_loader] Wrote {len(df)} {table_name} rows → {full_table}")

    except Exception as exc:
        logger.error(f"[bq_loader] Failed to write {table_name} → {full_table}: {exc}")
        all_success = False

    return all_success


# ─── Standalone smoke-test ───────────────────────────────────────────────────
import json
from pathlib import Path

def read_json_records(file_path: str | Path) -> list[dict]:
    """
    Read a JSON file and return records as a list of dicts,
    ready to pass into write_processed_to_bq().

    Handles both:
      - A JSON array:  [{...}, {...}, ...]
      - A JSON object: {"key": [{...}, ...]}  → pass `root_key` to extract nested list

    Args:
        file_path: Path to the JSON file.

    Returns:
        List of record dicts.

    Raises:
        ValueError: If the JSON structure isn't a list or extractable dict.
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # If there's a single top-level key whose value is a list, unwrap it
        list_values = {k: v for k, v in data.items() if isinstance(v, list)}
        if len(list_values) == 1:
            key, records = next(iter(list_values.items()))
            return records
        raise ValueError(
            f"JSON object has multiple list-valued keys: {list(list_values)}. "
            "Specify which to use by slicing the dict before passing in."
        )

    raise ValueError(f"Expected a JSON array or object, got {type(data).__name__}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = get_bq_client()
    if client:
        print(f"[bq_loader] Connected to BigQuery project: {PROJECT_ID}")
        print(f"[bq_loader] Raw dataset:       {RAW_DATASET}")
        print(f"[bq_loader] Processed dataset: {PROC_DATASET}")
        print(f"[bq_loader] Reports dataset:   {REPORTS_DATASET}")
        loaded_files=load_raw_from_bq(client, RAW_DATASET,RAW_PASSAGES_TABLE)
        print(loaded_files[:1])
        records=read_json_records("data/processed/decompositions.json")
        status=write_processed_to_bq(client,records, "decompositions")

    else:
        print("[bq_loader] Could not connect to BigQuery — check credentials")
