"""
cloudsql_loader.py — MOMENT Cloud SQL Data Acquisition
"""

import logging
import os
import yaml
import pandas as pd
from datetime import datetime

logger = logging.getLogger('moment_pipeline')

DEFAULT_INSTANCE = os.environ.get('INSTANCE_CONNECTION_NAME', 'moment-486719:us-central1:moment-db')
DEFAULT_DB_NAME  = os.environ.get('CLOUDSQL_DB',   'momento')
DEFAULT_DB_USER  = os.environ.get('CLOUDSQL_USER',  'momento_admin')
DEFAULT_DB_PASS  = os.environ.get('CLOUDSQL_PASS',  '')


class CloudSQLLoader:

    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataframes = None

        csql = self.config.get("cloudsql", {})
        self.instance = csql.get("instance_connection_name") or DEFAULT_INSTANCE
        self.db_name  = csql.get("db_name") or DEFAULT_DB_NAME
        self.db_user  = csql.get("db_user") or DEFAULT_DB_USER
        self.db_pass  = csql.get("db_pass") or DEFAULT_DB_PASS

        self._engine = None
        logger.info(f"CloudSQLLoader initialised — instance={self.instance}, db={self.db_name}")

    def run(self):
        logger.info("Starting Cloud SQL data acquisition")

        # Load moments first — books and users are filtered based on today's moments
        moments_df = self._load_moments()

        self.dataframes = {
            "moments_raw": moments_df,
            "books_raw":   self._load_books(moments_df),
            "users_raw":   self._load_users(moments_df),
        }
        total_rows = sum(len(df) for df in self.dataframes.values())
        logger.info(f"Acquisition complete: {len(self.dataframes)} tables, {total_rows} total rows")
        return {
            "timestamp":  self.timestamp,
            "num_files":  len(self.dataframes),
            "total_rows": total_rows,
            "files":      list(self.dataframes.keys()),
            "source":     "cloudsql",
        }

    def get_dataframes(self):
        if self.dataframes is None:
            raise ValueError("No data loaded — call run() first.")
        return self.dataframes

    def _load_moments(self) -> pd.DataFrame:
        """Only load moments created TODAY (midnight to midnight)."""
        query = """
            SELECT
                m.id::text                              AS moment_id,
                m.user_id::text                         AS user_id,
                b.id::text                              AS book_id,
                m.passage_key                           AS passage_id,
                m.passage                               AS passage,
                m.chapter,
                m.page_num,
                m.interpretation,
                COALESCE(
                    array_length(
                        regexp_split_to_array(trim(m.interpretation), '\\s+'),
                        1
                    ), 0
                )                                       AS word_count,
                m.created_at,
                m.interpretation_updated_at
            FROM public.moments m
            LEFT JOIN public.books b ON b.id = m.book_id
            WHERE m.is_deleted = FALSE
              AND m.interpretation IS NOT NULL
              AND trim(m.interpretation) <> ''
              AND DATE(m.created_at) = CURRENT_DATE
            ORDER BY m.created_at
        """
        df = self._run_query(query)
        logger.info(f"  -> {len(df)} moments (today)")
        return df

    def _load_books(self, moments_df: pd.DataFrame) -> pd.DataFrame:
        """Only load books linked to today's moments."""
        if moments_df.empty:
            logger.info("  -> 0 books (no moments today)")
            return pd.DataFrame()

        book_ids = moments_df['book_id'].dropna().unique().tolist()
        ids_str  = ", ".join(f"'{bid}'" for bid in book_ids)

        query = f"""
            SELECT
                b.id::text      AS book_id,
                b.title         AS book_title,
                b.author        AS book_author,
                b.year,
                b.gutenberg_id,
                b.cover_url,
                b.opening_passage,
                b.created_at,
                gc.epub_url,
                gc.text_url,
                gc.fetched_at   AS cache_fetched_at
            FROM public.books b
            LEFT JOIN public.gutenberg_book_cache gc ON gc.book_id = b.id
            WHERE b.id::text IN ({ids_str})
            ORDER BY b.title
        """
        df = self._run_query(query)
        logger.info(f"  -> {len(df)} books (linked to today's moments)")
        return df

    def _load_users(self, moments_df: pd.DataFrame) -> pd.DataFrame:
        """Only load users linked to today's moments."""
        if moments_df.empty:
            logger.info("  -> 0 users (no moments today)")
            return pd.DataFrame()

        user_ids = moments_df['user_id'].dropna().unique().tolist()
        ids_str  = ", ".join(f"'{uid}'" for uid in user_ids)

        query = f"""
            SELECT
                id::text                        AS user_id,
                firebase_uid,
                first_name,
                last_name,
                email,
                readername,
                bio,
                gender,
                photo_url,
                dark_mode,
                moments_layout_mode,
                passage_first,
                last_read_book_id::text         AS last_read_book_id,
                onboarding_complete,
                consent_given,
                consent_at,
                created_at,
                last_login_at,
                last_hero_gut_id,
                guide_book_gut_id,
                reading_state,
                last_captured_type,
                last_captured_shelf_id::text    AS last_captured_shelf_id
            FROM public.users
            WHERE id::text IN ({ids_str})
            ORDER BY created_at
        """
        df = self._run_query(query)
        logger.info(f"  -> {len(df)} users (linked to today's moments)")
        return df

    def _get_engine(self):
        if self._engine is not None:
            return self._engine

        from google.cloud.sql.connector import Connector
        from sqlalchemy import create_engine

        connector = Connector()

        def _get_conn():
            return connector.connect(
                self.instance,
                "pg8000",
                user=self.db_user,
                password=self.db_pass,
                db=self.db_name,
                ip_type="PRIVATE",
            )

        self._engine = create_engine("postgresql+pg8000://", creator=_get_conn)
        logger.info("SQLAlchemy engine created via Cloud SQL Connector")
        return self._engine

    def _run_query(self, query: str) -> pd.DataFrame:
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Cloud SQL query failed: {e}")
            raise