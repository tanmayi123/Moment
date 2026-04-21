# Moment

> **Read. Moments. Worth. Sharing.**
> IE7374 · MLOps · Group 23 · Northeastern University

> ⚠️ Note: This repository is a fork of the original group project repository. It has been forked to showcase my contributions and to continue independent development without affecting the original project.

> 🔗 Original Repository: [View Main Project Repository](https://github.com/jyothssena/Moment)


Moment is a private reading platform that uses machine learning to match intellectually compatible readers. Users capture book passages as visual "moments," write personal reflections, and are quietly matched with readers who think and feel similarly about literature — without performative social media posting.

This repository contains the **full MLOps pipeline** across four integrated components: a data pipeline (Airflow + TFDV), an agent-based compatibility model (Gemini 2.5 Flash), a FastAPI production pipeline (Cloud SQL → BigQuery → Cloud Run), and a monitoring stack (Google Cloud Monitoring + Grafana).

---

## Table of Contents

1. [Quick Start — Clone & Run](#1-quick-start--clone--run)
2. [Repository Structure](#2-repository-structure)
3. [Dataset Overview](#3-dataset-overview)
4. [System Architecture](#4-system-architecture)
5. [Environment Setup](#5-environment-setup)
6. [Data Pipeline (Airflow)](#6-data-pipeline-airflow)
6. [Data Pipeline (Airflow)](#6-data-pipeline-airflow)
7. [Compatibility Model](#7-compatibility-model)
8. [FastAPI Production Pipeline](#8-fastapi-production-pipeline)
9. [Monitoring & Alerting](#9-monitoring--alerting)
10. [CI/CD Pipeline](#10-cicd-pipeline)
11. [Experiment Tracking](#11-experiment-tracking)
12. [Model Validation & Bias Detection](#12-model-validation--bias-detection)
13. [Rankings — Bradley-Terry Model](#13-rankings--bradley-terry-model)
14. [Containerization](#14-containerization)
15. [Data Versioning with DVC](#15-data-versioning-with-dvc)
16. [Testing](#16-testing)
17. [Sensitivity Analysis](#17-sensitivity-analysis)
18. [Data Drift — Current Status](#18-data-drift--current-status)
19. [Evaluation Criteria Coverage](#19-evaluation-criteria-coverage)

---

## 1. Quick Start — Clone & Run

### Clone the Repository

```bash
git clone https://github.com/jyothssena/Moment.git
cd Moment
```

### Option A — Run the FastAPI Production Pipeline

This is the main pipeline used in production. It requires GCP credentials and a running Cloud SQL instance.

```bash
cd fastapi_pipeline
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set environment variables (copy and fill in your values):

```bash
export GEMINI_API_KEY_MOMENT=<your-gemini-api-key>
export GOOGLE_CLOUD_PROJECT=moment-486719
export INSTANCE_CONNECTION_NAME=moment-486719:us-central1:moment-db
export CLOUDSQL_DB=momento
export CLOUDSQL_USER=momento_admin
export CLOUDSQL_PASS=<your-db-password>
export BQ_DATASET=new_moments_processed
```

Start the server:

```bash
uvicorn main:app --reload --port 8080
```

Trigger the pipeline:

```bash
curl -X POST http://localhost:8080/pipeline/run
```

Interactive API docs at `http://localhost:8080/docs`.

---

### Option B — Run the Airflow Data Pipeline (Local)

This runs the preprocessing + validation + BQ upload pipeline locally using Docker.

```bash
# From the repo root
docker-compose up --build
```

Open `http://localhost:8080` → Airflow UI → trigger `moment_team_pipeline_MULTILEVEL`.

---

### Option C — Run the Compatibility Model Offline (No GCP)

To test the Gemini agents against the existing synthetic dataset without any GCP services:

```bash
# From the repo root
pip install -r requirements.txt
export GEMINI_API_KEY_MOMENT=<your-gemini-api-key>

# Replay existing results into MLflow (no Gemini calls)
python experiment_tracking/run_experiment.py --replay

# Run live against the synthetic dataset
python experiment_tracking/run_experiment.py
```

---

### Option D — Run Tests

```bash
pip install pytest
pytest data_pipeline/tests/ -v
```

---

## 2. Repository Structure

```
Moment/
│
├── fastapi_pipeline/                    ← Production pipeline (main)
│   ├── main.py                          ← FastAPI app: /pipeline/run, /feedback, /rankings
│   ├── cloudsql_loader.py               ← Cloud SQL → DataFrames (today's moments only)
│   ├── preprocessor_fastapi.py          ← Text cleaning, validation, quality scoring
│   ├── bq_writer.py                     ← Write processed data to BigQuery
│   ├── compatibility_agent.py           ← Compatibility scoring via Gemini
│   ├── decomposing_agent.py             ← Sub-claim decomposition via Gemini
│   ├── aggregator.py                    ← R/C/D score aggregation (passage, book, profile)
│   ├── run_rankings.py                  ← Bradley-Terry ranking model
│   ├── tools.py                         ← BQ query helpers and caching
│   ├── metrics.py                       ← Google Cloud Monitoring custom metrics
│   ├── requirements.txt                 ← Production dependencies
│   └── Dockerfile                       ← Cloud Run container
│
├── data_pipeline/                       ← Airflow data pipeline
│   ├── airflow/dags/                    ← DAG definitions
│   │   ├── data_pipeline_dag.py         ← Main pipeline DAG
│   │   ├── tests_dag.py                 ← Tests DAG
│   │   ├── inference_pipeline_dag.py    ← Inference DAG
│   │   └── bq_loader.py                 ← BQ loading DAG
│   ├── config/                          ← Pipeline configuration (YAML)
│   │   ├── config.yaml
│   │   ├── preprocessing_config.yaml
│   │   └── schema.yaml                  ← TFDV schema definition
│   └── scripts/                         ← Pipeline stages
│       ├── data_acquisition.py
│       ├── preprocessor.py
│       ├── validation.py
│       ├── generate_schema_stats.py     ← TFDV schema + statistics
│       ├── bias_detection.py
│       └── anomalies.py
│
├── cicd_pipeline/                       ← Model deployment scripts
│   ├── validate_model.py                ← Metrics gate (blocks deployment on regression)
│   ├── bias_detection.py                ← Multi-slice bias analysis
│   ├── model_interface.py               ← Unified model API
│   ├── notifications.py                 ← Slack alert system
│   └── rollback.py                      ← Threshold-based rollback
│
├── models/                              ← Agent model code (reference / offline use)
│   ├── main.py
│   ├── compatibility_agent.py
│   ├── decomposing_agent.py
│   ├── aggregator.py
│   ├── run_rankings.py
│   └── tools.py
│
├── monitoring/                          ← Observability stack
│   ├── setup_alerts.py                  ← Alertmanager alert rules
│   └── grafana/
│       ├── dashboards/moment_pipeline.json
│       └── provisioning/
│           ├── dashboards/dashboard.yml
│           └── datasources/prometheus.yml
│
├── experiment_tracking/                 ← MLflow experiment logging
│   ├── run_experiment.py                ← Replay or live mode
│   ├── MLflow_logger.py
│   └── config.yaml
│
├── data/                                ← Central data repository
│   ├── raw/                             ← Source PDFs, JSON, CSV
│   │   ├── pdfs/                        ← 9 passage PDFs (3 books × 3 passages)
│   │   └── csvs_jsons/                  ← Synthetic interpretation data
│   ├── processed/                       ← DVC-tracked processed data
│   ├── reports/                         ← TFDV, bias, and anomaly reports
│   ├── schemas/                         ← TFDV schema .pbtxt files
│   └── bias_results/
│
├── docs/                                ← Visual documentation (pipeline diagrams, MLflow screenshots)
├── tests/                               ← Integration tests
├── credentials/                         ← GCS service account (NOT committed to git)
├── .github/workflows/                   ← GitHub Actions (cicd.yml, deploy.yml)
├── dvc.yaml                             ← DVC pipeline stage definitions
├── docker-compose.yaml                  ← Airflow + Postgres for local dev
└── requirements.txt                     ← Root-level dependencies
```

---

## 3. Dataset Overview


| Property | Value |
|---|---|
| Final dataset | `data/raw/csvs_jsons/all_interpretations_450_FINAL_NO_BIAS.json` |
| Total interpretations | 450 |
| Character personas | 50 |
| Books | 3 — *Frankenstein*, *Pride & Prejudice*, *The Great Gatsby* |
| Passages per book | 3 (9 total) |
| Format | JSON |

### Source PDFs

| Book | Author | Year |
|---|---|---|
| *Frankenstein* | Mary Shelley | 1818 |
| *Pride & Prejudice* | Jane Austen | 1813 |
| *The Great Gatsby* | F. Scott Fitzgerald | 1925 |

All books are public domain (pre-1928), sourced from [Project Gutenberg](https://www.gutenberg.org/).

### Data Fields

Each interpretation record contains: `user_id`, `book_id`, `passage_id`, `interpretation` (the raw moment text), `word_count`, `character_name`, `gender`, `readership_type`, `created_at`. The bias-free version was produced by running the data through the bias detection pipeline and removing records that over-represented any demographic slice.

---

## 3.1 Application Layer - Frontend

Location: frontend/

The Moment frontend is a JavaScript + JSX application served via Nginx, built without a framework bundler — all JSX is compiled via a custom build script (scripts/build-v64.js) into src/app.compiled.js. The app uses Firebase Authentication for sign-in (email/password and Google OAuth) and communicates with the backend API using Bearer token headers.

The UI is structured around four panels navigated via a rotating cube metaphor (CubeHint):

Read (src/features/read/) — renders book text fetched from Project Gutenberg. Users drag-select a passage to "snip" it, triggering a SnipOverlay that captures the highlighted text as a new moment.

Moments (src/features/moments/) — displays the user's saved moments grouped by book. Each moment can have an interpretation written against it. Supports two layout modes (clip-by-books, grid) and passage-first or interpretation-first display ordering.

Worth (src/features/worth/) — shows compatible readers fetched from the backend, displaying Think/Feel R/C/D breakdowns per match. Pulls from /worth/matches, /worth/rankings, /worth/book-compatibility, and /worth/profile-compatibility endpoints and renders the scores as bar charts via charting.jsx.

Sharing (src/features/sharing/) — a close-readers feed showing waves, whispers, and shared moments between users the current reader has connected with.

The app also includes a full onboarding flow (OnboardingStageTrack, ReaderOnboardingOverlay, ConsentScreen) and a profile drawer (ProfileDrawer) with dark mode support. All authentication state is managed in MomentApp.jsx via Firebase's onAuthStateChanged.

Containerized via Dockerfile.frontend and served behind Nginx (nginx.conf) for production deployment.

## 3.2 Application Layer - Backend


The backend is a FastAPI application (api/main.py, version 1.0.0) backed by an async PostgreSQL connection via SQLAlchemy (asyncpg driver). It handles all user-facing data operations and acts as the bridge between the frontend, Cloud SQL, and the ML pipeline.

Database schema (schema.sql): PostgreSQL with tables for users, books, moments, reading_signatures, close_readers, waves, whispers, and shared_moments. UUIDs are used as primary keys throughout, and firebase_uid is the foreign key that links Firebase Auth users to the database.

Routes:

POST /users / GET /users/me / PATCH /users/me/preferences — user creation, profile retrieval, and preference updates (dark mode, reading state, onboarding progress, shelf management). Supports both email/password and Google OAuth sign-in flows.

GET /moments / POST /moments / PATCH /moments/{id} / DELETE /moments/{id} — full CRUD for moments. On every POST /moments, a passage_key is computed (SHA-256 of book_id + passage[:200]) for stable cross-user passage matching, and the ML pipeline is triggered via a fire-and-forget async POST to POST /pipeline/run — the moment is saved regardless of whether the pipeline call succeeds.

GET /worth/matches — fetches compatibility results from BigQuery (compatibility_results table), joining against users_processed for reader names, then enriches with book titles from Cloud SQL.

GET /worth/rankings — proxies to GET /rankings/{user_id} on the FastAPI ML pipeline (Cloud Run), triggering a live BT refit.

GET /worth/book-compatibility / GET /worth/profile-compatibility — reads from book_compatibility and profile_compatibility BigQuery tables for aggregated match scores.

GET /sharing/close-readers / POST /sharing/wave / GET /sharing/whispers / POST /sharing/shared-moments — manages the social layer between connected readers.

Authentication (api/auth.py) verifies Firebase ID tokens on every request via firebase_admin. Migrations run automatically at startup via the lifespan context manager, adding any missing columns to the users table without requiring a manual migration step

## 4. System Architecture

The system has two parallel pipelines that feed into each other, plus a monitoring layer:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AIRFLOW DATA PIPELINE (daily batch)              │
│  acquire_data → bias_detection → preprocessing →                   │
│  validation + schema_stats (parallel) → upload_to_gcs → notify     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ processed data → BQ
┌────────────────────────────────▼────────────────────────────────────┐
│                  FASTAPI PRODUCTION PIPELINE (Cloud Run)            │
│                                                                      │
│  POST /pipeline/run                                                  │
│    1. CloudSQLLoader    → fetch today's moments (DATE = CURRENT)    │
│    2. preprocessor_fastapi → clean, validate, quality score         │
│    3. bq_writer         → write to BigQuery (4 tables, APPEND)      │
│    4. [background] compatibility_agent × decomposing_agent          │
│       → Gemini 2.5 Flash → R/C/D scores → BQ                       │
│    5. [background] run_rankings → Bradley-Terry refit → BQ          │
│                                                                      │
│  POST /feedback         → log pairwise comparison                   │
│  GET  /rankings/{user}  → top-k matched readers (BT blend)          │
│  POST /admin/retrain-trigger → alert webhook → full BT refit        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ metrics push every 60s
┌────────────────────────────────▼────────────────────────────────────┐
│              MONITORING (Google Cloud Monitoring + Grafana)          │
│  metrics.py → daemon thread → GCM time series + Alertmanager        │
│  Alerts fire → /admin/retrain-trigger → background refit            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Environment Setup

### Prerequisites

- Python 3.11+
- Docker + Docker Compose (for local Airflow)
- Google Cloud SDK (`gcloud`) authenticated to your project
- A GCP project with BigQuery, Cloud SQL (PostgreSQL), Cloud Run, and Cloud Monitoring APIs enabled

### FastAPI Pipeline (Production)

```bash
cd fastapi_pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file or export these variables:

```bash
export GEMINI_API_KEY_MOMENT=<your-gemini-api-key>
export GOOGLE_CLOUD_PROJECT=moment-486719
export INSTANCE_CONNECTION_NAME=moment-486719:us-central1:moment-db
export CLOUDSQL_DB=momento
export CLOUDSQL_USER=momento_admin
export CLOUDSQL_PASS=<your-password>
export BQ_DATASET=new_moments_processed
```

Run locally:

```bash
uvicorn main:app --reload --port 8080
```

The interactive FastAPI docs will be available at `http://localhost:8080/docs`.

### Airflow Data Pipeline (Local Dev)

```bash
docker-compose up --build
```

Navigate to `http://localhost:8080` for the Airflow UI. Trigger the `moment_team_pipeline_MULTILEVEL` DAG manually or let it run on schedule.

### Credentials

GCS service account JSON should be placed at `credentials/gcs-service-account.json`. It is excluded from git via `.gitignore`. Never commit credentials to the repository.

---

## 6. Data Pipeline (Airflow)

**Location:** `data_pipeline/`

The Airflow DAG runs daily and handles all preprocessing, validation, and upload steps before data lands in BigQuery. `validation` and `schema_stats` run in parallel after `preprocessing` — a bottleneck identified by analyzing the Gantt chart (see below).

```
acquire_data → bias_detection → preprocessing → validation  ──┐
                                              → schema_stats ──┴→ upload_to_gcs → notify
```

### Stage Details

**`acquire_data`** (`data_acquisition.py`)

Pulls raw interpretation records from source. In production this connects to the PostgreSQL Cloud SQL instance. In development it reads from `data/raw/csvs_jsons/`. Calls `data/character_extraction.py` and `data/data_extraction.py` to generate the 450-record synthetic dataset from the 50 character personas across 9 passages.

**`bias_detection`** (`bias_detection.py`)

Runs before preprocessing to flag representational imbalances early. Slices the data by `gender` and `readership_type` and computes confidence distributions per slice. Alerts if the confidence gap between any two slices exceeds the configured threshold. Output goes to `data/bias_results/`.

**`preprocessing`** (`preprocessor.py`)

Applies the same cleaning logic as `preprocessor_fastapi.py`: Unicode normalisation, email scrubbing, whitespace collapsing, word count computation, Flesch readability scoring via `textstat`, and language detection via `langdetect`. Outputs a cleaned DataFrame with quality metrics attached.

**`validation`** (`validation.py`)

Runs TFDV schema validation against `data_pipeline/config/schema.yaml`. Checks field presence, type constraints, and distribution ranges. Any anomalies are written to `data/reports/validation_report.json` and trigger an alert notification.

**`schema_stats`** (`generate_schema_stats.py`)

Uses TFDV to compute statistics on the preprocessed dataset — feature counts, distributions, missing value rates. Saves output to `data/reports/schema_stats.json` and updates the TFDV schema `.pbtxt` files in `data/schemas/`.

**`upload_to_gcs`**

Uploads the validated data, reports, and schema files to the configured GCS bucket. Retried up to 3 times on failure (visible in the Airflow logs under `attempt=1.log`, `attempt=2.log`, `attempt=3.log`).

**`notify`** (`notifications.py`)

Sends a pipeline completion notification via Slack. If any anomalies were detected during validation, the notification includes a summary of the specific issues flagged.

### Gantt Optimization

The parallel execution of `validation` and `schema_stats` after `preprocessing` was identified as the key optimisation — both steps depend only on the preprocessed data and have no dependency on each other, so they can safely run concurrently. This reduced the total DAG runtime by the duration of whichever is shorter.

![Main Pipeline Graph](docs/main_pipeline_graph.png)

![Main Pipeline Gantt](docs/main_pipeline_gantt.png)

![Tests Pipeline Graph](docs/tests_pipeline_graph.png)

![Tests Pipeline Gantt](docs/tests_pipeline_gantt.png)

![Airflow DAGs Overview](docs/airflow_dags.png)

---

## 7. Compatibility Model

**Location:** `fastapi_pipeline/` (production) · `models/` (reference / offline)

The compatibility model uses **Gemini 2.5 Flash** to decompose reader interpretations into weighted sub-claims, then scores pairs of readers across two completely independent dimensions: **Think** (intellectual alignment) and **Feel** (emotional alignment). The final output is R/C/D (Resonate / Contradict / Diverge) percentages with a confidence score.

### Stage 1 — Decomposition (`decomposing_agent.py`)

Each reader moment is sent to Gemini with a structured system prompt that instructs it to:

1. Identify 2–4 distinct intellectual sub-claims from the moment text
2. Anchor each sub-claim to a direct quote from the reader's own written words (not the passage text)
3. Assign a weight proportional to how many words the reader spent on that sub-claim (weights must sum to 1.0, minimum 0.10 per sub-claim)
4. Assign one of six emotional modes to each sub-claim: `prosecutorial`, `philosophical`, `empathetic`, `observational`, `aesthetic`, `self-referential`

**Protected sub-claims rule:** If the reader directly quotes a specific phrase from the passage (words in quotation marks in their moment), that phrase must anchor its own sub-claim and cannot be merged into another — even if its weight would fall below 0.10. The minimum weight in this case is forced to 0.10 with other weights adjusted proportionally.

**Validation:** The agent validates that all required keys are present (`passage_id`, `user_id`, `subclaims`), that each sub-claim has `id`, `claim`, `quote`, `weight`, and `emotional_mode`, and that weights sum to within ±0.02 of 1.0. On any failure it returns a structured error dict rather than a partial result, and the error is counted in `gemini_errors_total`.

```json
{
  "passage_id": "frankenstein_passage_1",
  "user_id": "user_123",
  "book_id": "frankenstein",
  "subclaims": [
    {
      "id": "1",
      "claim": "Victor's obsession with creation stems from a desire to bypass natural limits",
      "quote": "I had worked hard for nearly two years",
      "weight": 0.55,
      "emotional_mode": "prosecutorial"
    },
    {
      "id": "2",
      "claim": "The creature's ugliness reveals Victor's failure to consider consequence",
      "quote": "breathless horror and disgust",
      "weight": 0.45,
      "emotional_mode": "philosophical"
    }
  ]
}
```

### Stage 2 — Scoring (`compatibility_agent.py`)

The compatibility scorer receives the two decompositions and follows a two-step process:

**Step 1 — Map sub-claims:** Match each Reader A sub-claim to the closest Reader B candidate. Two sub-claims match if they respond to the same passage phrase or the same specific moment, even if worded differently. Each sub-claim can be matched at most once. Any B sub-claims not matched become `unmatched_b`.

**Step 2 — Score each matched pair with 10 booleans (5 THINK, 5 FEEL, scored completely independently):**

THINK questions:
- T1. Same subject? (always true for matched pairs)
- T2. Do their positions point the same direction?
- T3. Are their analytical lenses compatible?
- T4. Are their conclusions mutually exclusive?
- T5. Would Reader A agree with Reader B's claim?

FEEL questions:
- F1. Same emotional subject? (always true for matched pairs)
- F2. Do they use the same emotional mode label?
- F3. Do they share the same emotional trigger?
- F4. Do they describe the same emotional experience?
- F5. Would Reader A recognise Reader B's emotional response as valid?

**Unmatched sub-claims:** Flagged as `divergence: true` if Reader B engages the same subject area but not the same specific phrase or moment. Flagged as `divergence: false` if Reader B has nothing on this subject at all. Divergent unmatched sub-claims penalise the score more heavily than absent ones.

**Critical design constraint:** Think and Feel scores are computed independently. The system prompt explicitly warns: *"Two readers can share a conclusion but feel differently. Two readers can disagree intellectually but share the same emotional response. Never let Think scores contaminate Feel scores."*

### Stage 3 — Aggregation (`aggregator.py`)

Raw boolean arrays are converted to R/C/D fractions and aggregated using weighted averages:

```python
THINK_ADDS_R = [True, True, True, False, True]   # T4 reversed — mutual exclusivity adds C not R
FEEL_ADDS_R  = [True, True, True, True,  True]

def score_from_bools(qs: list[bool], adds_r: list[bool]) -> tuple[float, float]:
    R = sum(1 for q, r in zip(qs, adds_r) if q == r) / 5
    return R, 1 - R

# Weight for matched pairs: conservative — use the lower of the two sub-claim weights
w = min(weight_a, weight_b)

# Unmatched sub-claims (divergence penalty):
d = 1.0 if subclaim.get('divergence') else 0.3
```

After aggregating all weighted sums, the final R/C/D is rounded to integers summing to 100, then confidence is computed from three independent signals:

```python
depth    = min(wc_a, wc_b) / max(wc_a, wc_b)          # relative passage length similarity
coverage = len(matched) / max(n_sc_a, n_sc_b)           # fraction of sub-claims matched
dominance = (max(think.values()) + max(feel.values())) / 200  # decisiveness of leading verdict

conf = min(0.95, max(0.20, round(0.30*depth + 0.40*coverage + 0.30*dominance, 2)))
```

Coverage has the highest weight (0.40) because a high match count is the strongest signal of genuine intellectual engagement between two readers.

The aggregator also computes **book-level** aggregation (`aggregate_book_level`) by averaging R/C/D across all passages of a book for a reader pair, and **profile-level** aggregation (`aggregate_profile_level`) by averaging book-level results across all books.

**Final output structure:**
```json
{
  "think": {"R": 60, "C": 30, "D": 10},
  "feel":  {"R": 45, "C": 40, "D": 15},
  "dominant_think": "resonate",
  "dominant_feel":  "resonate",
  "confidence": 0.74,
  "match_count": 3,
  "passage_id": "frankenstein_passage_1",
  "book_id": "frankenstein",
  "user_a": "user_123",
  "user_b": "user_456",
  "timestamp": "2026-04-20T10:30:00"
}
```

### Caching Strategy

All decompositions and scoring runs are cached in BigQuery to avoid redundant Gemini API calls. Before running the decomposer, `get_decomposition(user_id, passage_id, book_id)` checks BQ. Before running the scorer, `get_scoring(user_a_id, user_b_id, passage_id, book_id)` checks BQ. Before running the full compatibility agent, `get_compat_run(user_a_id, user_b_id, book_id, passage_id)` checks BQ. This means re-running the pipeline for any previously seen passage pair costs zero Gemini API calls. Cache hits are tracked by `decomp_cache_hits_total` and `compat_cache_hits_total` in Cloud Monitoring.

### Hyperparameter Configuration

Temperature is fixed at `0.1` across all agents. This is deliberately low to maximise reproducibility — we want deterministic sub-claim decompositions and consistent boolean scoring, not creative variation across runs.

```python
response = _gemini_client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        temperature=0.1,
    ),
    contents=user_message,
)
```

---

## 8. FastAPI Production Pipeline

**Location:** `fastapi_pipeline/main.py`

The production pipeline runs on **Google Cloud Run**, triggered daily by Cloud Scheduler. It is versioned at `6.0.0` and handles the complete daily flow from raw Cloud SQL data through preprocessing, BigQuery writes, compatibility scoring, and ranked results.

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/pipeline/run` | Trigger full daily pipeline — Steps 1–3 synchronous, compat + rankings in background |
| `POST` | `/feedback` | Log a pairwise comparison (feeds Bradley-Terry model), triggers BT refit in background |
| `GET` | `/rankings/{user_id}` | Get top-k ranked matches, refits BT synchronously to ensure freshness |
| `GET` | `/health` | Returns `{"status": "ok", "timestamp": ...}` |
| `GET` | `/pipeline/status` | Returns the full result dict from the last pipeline run |
| `POST` | `/admin/retrain-trigger` | Alertmanager webhook receiver — queues a full BT refit across all users in background |

### Daily Pipeline Flow (`POST /pipeline/run`)

Steps 1–3 run synchronously and block until complete, returning a result dict. Steps 4–5 are queued as FastAPI `BackgroundTasks` — one background task per valid moment.

**Step 1 — Load from Cloud SQL** (`CloudSQLLoader`)

Connects to Cloud SQL via the Cloud SQL Python Connector (private IP, `pg8000` driver, SQLAlchemy engine). Fetches only moments where `DATE(created_at) = CURRENT_DATE` and `is_deleted = FALSE` and `interpretation IS NOT NULL`. Then fetches only the books and users linked to those moments (no full table scans). Moments from previous days are never reprocessed — they are already in BQ.

**Step 2 — Preprocess** (`preprocess_all`)

Each moment passes through five sequential quality gates. A moment that fails any gate is dropped from the valid set and its outcome is incremented in the `moments_processed` Cloud Monitoring counter:

```
has_prompt_injection() → skipped_injection    (regex patterns for injected instructions)
validate_text()        → skipped_invalid      (word count 10–600, quality score ≥ 0.5, English, no gibberish)
detect_issues()        → skipped_pii          (email addresses, SSN patterns)
detect_issues()        → skipped_profanity    (profanity ratio threshold)
detect_issues()        → skipped_spam         (caps ratio, repetitive chars/words)
→ PASSED               → valid
```

For each valid moment, the preprocessor also computes and stores: `word_count`, `char_count`, `sentence_count`, `avg_word_length`, `avg_sentence_length`, `readability_score` (Flesch via `textstat`), `quality_score`, `language` (via `langdetect`). All of these are written to BigQuery and used as data drift monitoring signals.

**Step 3 — Write to BigQuery** (`write_to_bq`)

All four DataFrames (moments, passages, books, users) are written to BigQuery with `WRITE_APPEND`. Books are deduplicated against existing BQ rows before writing — if a `book_id` already exists in `books_processed`, it is skipped. After each successful write, the `bq_writes_total` monitoring counter is incremented per table. Any exception increments `bq_write_errors_total`.

After all BQ writes, distribution gauges are pushed to Cloud Monitoring to establish per-run drift baselines:

```python
_update_distribution_gauges(moments, [], run_id)
# Computes and sets via Cloud Monitoring:
# valid_ratio, word_count_mean, word_count_p50, word_count_p95
# quality_score_mean, quality_score_p10, readability_mean
```

**Step 4 — Compatibility (background)** (`_run_batch_compatibility`)

For each valid moment, a background task is queued that: fetches all existing BQ users who have a moment on the same passage (excluding the current user), then calls `run_compatibility_agent()` for each pair. The agent checks the BQ cache first; only on a cache miss does it call Gemini. After all pairs for a user are processed, `refit_user()` is called immediately to update that user's rankings.

**Step 5 — Rankings (background)**

Called at the end of each background compatibility batch. Bradley-Terry rankings are refit and written to the `rankings` BQ table. `GET /rankings/{user_id}` also triggers a synchronous refit on every call to guarantee the returned results include all the latest compatibility runs.

### Pipeline Response

```json
{
  "status": "success",
  "timestamp": "2026-04-20T10:00:00",
  "moments_count": 47,
  "passages_count": 12,
  "books_count": 3,
  "users_count": 31,
  "valid_moments": 44,
  "bq_tables": ["moments_processed", "passages_processed", "books_processed", "users_processed"],
  "compat_queued": 44,
  "duration_sec": 8.42
}
```

If no moments were created today, the pipeline returns `{"status": "skipped", "reason": "no moments created today"}` without writing anything to BQ.

### BigQuery Tables

| Table | Content |
|---|---|
| `moments_processed` | Cleaned interpretations with all quality metrics |
| `passages_processed` | Deduplicated passage text keyed by `passage_id` |
| `books_processed` | Book metadata, deduplicated — only new books are inserted |
| `users_processed` | User profile data |
| `compatibility_results` | R/C/D scores per reader pair per passage |
| `decompositions` | Cached sub-claim decompositions (BQ cache layer) |
| `scoring_runs` | Cached compatibility scoring outputs (BQ cache layer) |
| `comparisons` | Pairwise feedback from `POST /feedback` (feeds BT model) |
| `conversations` | Engagement scores per match (BT comparison weights) |
| `rankings` | BT-ranked matches per user per passage, with blend scores |
| `book_compatibility` | Book-level aggregated R/C/D per reader pair |
| `profile_compatibility` | Profile-level aggregated R/C/D per reader pair |

---

## 9. Monitoring & Alerting

**Location:** `fastapi_pipeline/metrics.py` · `monitoring/`

### Google Cloud Monitoring Integration

`metrics.py` is a complete drop-in replacement for `prometheus_client` that pushes custom metrics to **Google Cloud Monitoring (GCM)**. It provides `_Counter`, `_Gauge`, and `_Histogram` classes with the same `.inc()`, `.set()`, `.observe()`, and `.labels()` API as Prometheus — so every call site in the codebase requires zero changes to work with either backend.

All metric values accumulate in memory using thread-safe dicts. A daemon thread pushes the accumulated values to GCM every 60 seconds:

```python
threading.Thread(target=_background_push, daemon=True, name="gcm-push").start()
```

At the end of every pipeline run, `push_metrics_now()` forces an immediate flush before Cloud Run may shut down the instance:

```python
push_metrics_now()   # respects a 30s minimum interval to avoid GCM rate limit errors
```

On Cloud Run, GCP auth uses the service account attached to the instance. Locally, it uses ADC (`gcloud auth application-default login`). Credential failures are caught silently — the pipeline is never affected.

`_Histogram` pushes both **mean** and **P95** as separate GAUGE time series, since GCM does not have a native histogram type. This is sufficient for all dashboard and alerting use cases.

### Metric Categories

**Pipeline health:**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `pipeline_runs_total` | Counter | `status` (success/skipped/error/retrain_triggered) | Total pipeline run outcomes |
| `pipeline_duration_seconds` | Histogram | `phase` (full/preprocess/bq_write) | Timed duration per phase using `time.monotonic()` |
| `pipeline_bq_write_errors_total` | Counter | `table` | BQ write failures per table |
| `pipeline_bq_writes_total` | Counter | `table` | Successful BQ writes per table |

**Preprocessing / data drift:**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `pipeline_moments_processed_total` | Counter | `outcome` | Per-moment outcome (valid, skipped_injection, skipped_pii, skipped_profanity, skipped_spam, skipped_invalid) |
| `valid_ratio` | Gauge | `pipeline_run_id` | Fraction of moments passing all quality gates per run |
| `word_count_mean` / `p50` / `p95` | Gauge | `pipeline_run_id` | Interpretation word count distribution per run |
| `quality_score_mean` / `p10` | Gauge | `pipeline_run_id` | Quality score distribution per run |
| `readability_mean` | Gauge | `pipeline_run_id` | Mean Flesch readability score per run |

These per-run distribution gauges are the primary **data drift signal** — a sustained shift in `valid_ratio` or `word_count_p95` over multiple runs indicates a change in user writing behaviour that should trigger a pipeline review.

**Compatibility / model decay:**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `compat_runs_total` | Counter | `outcome` (success/cached/error_decomp/error_scoring/error_agg) | Agent run outcomes |
| `compat_confidence` | Histogram | `dominant_think` (resonate/contradict/diverge) | Confidence distribution (mean + P95 per verdict type) |
| `compat_confidence_gauge` | Gauge | — | Most recently observed confidence score |
| `compat_think_ratio` | Gauge | `verdict` | Cumulative think-dimension verdict distribution |
| `compat_feel_ratio` | Gauge | `verdict` | Cumulative feel-dimension verdict distribution |
| `gemini_latency_seconds` | Histogram | `call_type` (decompose/score) | End-to-end Gemini API call latency |
| `gemini_errors_total` | Counter | `call_type`, `error_type` (json_parse/missing_keys/exception) | Gemini API failures |
| `decomp_cache_hits_total` | Counter | — | Decomposition BQ cache hits (saved Gemini calls) |
| `compat_cache_hits_total` | Counter | — | Full compat result BQ cache hits |

Sustained low values across the `compat_confidence` histogram is the primary **model decay signal** — it indicates the model is failing to produce decisive verdicts, which should trigger `CompatConfidenceCritical` and queue a retraining run.

**Bradley-Terry rankings:**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `bt_refits_total` | Counter | `outcome` (success/no_data/error) | BT refit outcomes |
| `bt_refit_duration_seconds` | Histogram | — | Wall-clock time to refit BT and write rankings for one user |
| `bt_n_comparisons` | Histogram | — | Number of comparisons available at each refit |
| `bt_blend_weight_bt` | Gauge | — | BT contribution to the blend score for the most recently refit user |
| `rankings_written_total` | Counter | — | Ranking rows successfully inserted to BQ |

`bt_blend_weight_bt` near zero across many users signals a cold-start problem — users are not engaging with the feedback mechanism and the ranker is falling back to raw confidence ordering.

### Grafana Dashboard

`monitoring/grafana/dashboards/moment_pipeline.json` is a pre-built Grafana dashboard with panels for all the above metrics. It is provisioned automatically via `monitoring/grafana/provisioning/dashboards/dashboard.yml` and reads from the Cloud Monitoring data source configured in `monitoring/grafana/provisioning/datasources/prometheus.yml`.

### Alerting

`monitoring/setup_alerts.py` configures Alertmanager rules. When alerts fire (e.g. `CompatConfidenceCritical` when P95 confidence drops below threshold, or `ValidMomentRatioCritical` when `valid_ratio` drops significantly), the Alertmanager POSTs the alert payload to `/admin/retrain-trigger`:

```python
@app.post("/admin/retrain-trigger")
async def retrain_trigger(payload: dict, background_tasks: BackgroundTasks):
    alert_name = payload.get("commonLabels", {}).get("alertname", "unknown")
    status     = payload.get("status", "unknown")
    logger.warning(f"[Retrain] alert={alert_name} status={status} — queuing full refit")
    pipeline_runs.labels("retrain_triggered").inc()
    from run_rankings import main as run_full_refit
    background_tasks.add_task(run_full_refit)
    return {"status": "retraining_queued", "trigger": alert_name}
```

In production, this background task should be replaced with a Cloud Tasks or Pub/Sub publish to decouple the refit from the web request lifecycle.

---

## 10. CI/CD Pipeline

**Files:** `.github/workflows/cicd.yml` · `.github/workflows/deploy.yml`

The pipeline triggers on any push to `main` that touches model or infrastructure files (`compatibility_agent.py`, `decomposing_agent.py`, `validate_model.py`, `Dockerfile`, `requirements.txt`, `tests/**`, and others). It can also be triggered manually via `workflow_dispatch`. It enforces a strict no-regression policy — each stage must pass before the next begins, and every stage failure sends a Slack notification.

![CI/CD Pipeline](docs/cicd_pipeline.jpeg)

### Stage 1 — Tests (`test` job)

```yaml
- name: Run full test suite
  run: pytest tests/ -v --tb=short
```

Runs all unit and integration tests. On failure, `notify_pipeline_failure('test', 'pytest suite failed')` is called and all downstream stages are blocked. The test job uses pip caching keyed on `requirements.txt` to avoid reinstalling dependencies on every run.

### Stage 2 — Validate & Bias Check (`validate` job, needs: `test`)

```yaml
- run: python run_validation_set.py --output validation_results.json
- run: python validate_model.py validation_results.json
- run: python bias_detection.py validation_results.json
```

Runs the full agent pipeline on a held-out validation set. `validate_model.py` checks that `mean_confidence` and `schema_pass_rate` are above thresholds. `bias_detection.py` checks that confidence gaps between demographic slices are within limits. The `validation_results.json` artifact is uploaded to GitHub Actions storage for use in the rollback check stage. On failure, the notification includes the specific failing metrics and thresholds.

### Stage 3 — Build & Push to Artifact Registry (`build-and-push` job, needs: `validate`, only on `main`)

Builds three Docker images and pushes them to GCP Artifact Registry tagged with both the immutable Git SHA and `latest`:

```bash
for agent in decompose-agent compat-agent profile-agent; do
  docker build -t $REGISTRY/${agent}:${{ github.sha }} -t $REGISTRY/${agent}:latest -f Dockerfile .
  docker push $REGISTRY/${agent}:${{ github.sha }}
  docker push $REGISTRY/${agent}:latest
done
```

### Stage 4 — Rollback Check (`rollback-check` job, needs: `build-and-push`, only on `main`)

Downloads the validation artifact from Stage 2, computes metrics, fetches the current production baseline from GCS (`metrics_baseline.json`), and compares:

```python
ROLLBACK_THRESHOLDS = {
    "mean_confidence":  0.05,   # Rollback if new model drops by more than 0.05
    "schema_pass_rate": 0.03,   # Rollback if new model drops by more than 3%
}
```

If regression is detected, `production-stable` images are re-tagged to `production` (reverting to the last known-good build) and `notify_rollback(reason, rolled_back_to)` is called. No deployment occurs.

### Stage 5 — Deploy to Vertex AI (`deploy` job, needs: `rollback-check`, only on `main`)

Runs `deploy.py` which deploys agents to **Vertex AI Agent Engine**. On success: the deployed images are tagged `production-stable` for future rollback reference, and the new metrics file is saved to GCS as the updated baseline:

```bash
gcloud storage cp new_metrics.json gs://moment-agent-data/metrics_baseline.json
```

Slack notifications are sent for both success (`notify_deployment_success(sha, metrics)`, `notify_training_complete(sha, metrics)`) and failure (`notify_pipeline_failure('deploy', ...)`).

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | Deployment region (e.g. `us-central1`) |
| `GCP_SA_KEY` | Service account JSON key for GCP authentication |
| `GCS_BUCKET` | GCS bucket name for baseline metrics storage |
| `GEMINI_API_KEY_MOMENT` | Gemini API key used in validation set run |
| `SLACK_WEBHOOK_URL` | Slack incoming webhook URL for all notifications |

---

## 11. Experiment Tracking

**Tool:** MLflow · **Location:** `experiment_tracking/`

Every compatibility pair produces a **parent MLflow run** with two nested **child runs** (one per reader decomposition), giving full traceability from raw moment text through sub-claim decomposition to the final compatibility verdict:

```
parent run  →  Emma Chen × Marcus Williams | Frankenstein / passage_1
    ├── child run  →  decomp_reader_a_Emma Chen
    └── child run  →  decomp_reader_b_Marcus Williams
```

### Parent Run Logs

- **params:** `user_a`, `user_b`, `book_id`, `passage_id`, `model_name` (`gemini-2.5-flash`), `temperature` (0.1), `prompt_version`
- **metrics:** `confidence`, `match_count`, `think_R`, `think_C`, `think_D`, `feel_R`, `feel_C`, `feel_D`
- **tags:** `dominant_think`, `dominant_feel`, `route` (display/discard from `route_compatibility_result()`), `verdict`
- **artifact:** full result JSON with all scores and metadata

### Child Run (per decomposition) Logs

- **params:** `user_id`, `passage_id`, `book_id`, `reader_label` (a or b)
- **metrics:** `subclaim_count`, `weight_entropy` (Shannon entropy of weight distribution), `weight_min`, `weight_max`
- **tags:** `emotional_modes` (comma-separated list of all modes used), `dominant_mode`
- **artifact:** full decomposition JSON

### Screenshots

![MLflow Runs](docs/mlflow_runs1.jpeg)

![MLflow Runs](docs/mlflow_runs2.jpeg)

![MLflow Runs](docs/mlflow_runs3.jpeg)

![MLflow Runs](docs/mlflow_runs4.jpeg)

### Running

```bash
pip install mlflow pyyaml

# Replay mode — logs existing JSON files from data/processed/ to MLflow. Zero Gemini calls.
python experiment_tracking/run_experiment.py --replay

# Live mode — runs the full pipeline through Gemini agents and logs all results.
python experiment_tracking/run_experiment.py

# Launch MLflow UI
mlflow ui --port 5000
```

---

## 12. Model Validation & Bias Detection

**Location:** `cicd_pipeline/`

### Model Validation (`validate_model.py`)

Runs a fixed held-out validation set through the full compatibility agent pipeline and evaluates the following quality gates. All must pass or deployment is blocked and a Slack notification is sent:

- **mean_confidence** — average confidence score across all validation pairs. Must not drop more than 0.05 below the stored baseline.
- **schema_pass_rate** — fraction of agent outputs that conform to the expected output schema (all required keys present, valid R/C/D structure, confidence in [0, 1]). Must not drop more than 3% below baseline.
- **error_rate** — fraction of pairs where the agent returned an error dict. Must not increase from baseline.

```python
def run_validation_gate(results: list) -> dict:
    metrics  = compute_validation_metrics(results)
    failures = []
    if metrics["mean_confidence"] < THRESHOLDS["min_confidence"]:
        failures.append(f"mean_confidence {metrics['mean_confidence']:.3f} below threshold")
    if metrics["schema_pass_rate"] < THRESHOLDS["min_schema_pass_rate"]:
        failures.append(f"schema_pass_rate {metrics['schema_pass_rate']:.3f} below threshold")
    return {"passed": len(failures) == 0, "metrics": metrics, "failures": failures}
```

On gate failure, `notify_validation_failure(metrics, failures)` sends a detailed Slack alert with the specific failing metrics, the current values, and the thresholds they failed to meet.

### Bias Detection

Bias detection runs at two distinct points in the system:

**1. Data pipeline** (`data_pipeline/scripts/bias_detection.py`)

Runs before preprocessing on the raw dataset. Slices data by `gender` and `readership_type` and checks that each slice is represented proportionally. Computes per-slice statistics (count, mean word count, mean quality score) and flags imbalances. Output is written to `data/bias_results/bias_report_FINAL.md`. The `notify` DAG task includes a summary if any imbalances are detected.

**2. Model pipeline** (`cicd_pipeline/bias_detection.py`)

Runs as CI/CD Stage 2 on the validation set outputs. Checks that `confidence` scores don't systematically differ across demographic slices. Groups results by `gender` and `readership_type` of the two readers involved, then computes the mean confidence per slice. If the gap between any two slices exceeds the configured threshold, the build fails and the gate message includes which slices diverged and by how much.

The final bias-free dataset at `data/raw/csvs_jsons/all_interpretations_450_FINAL_NO_BIAS.json` was produced by iteratively running this pipeline on earlier dataset versions and removing records that caused demographic imbalance until all slice gaps were below threshold.

---

## 13. Rankings — Bradley-Terry Model

**Location:** `fastapi_pipeline/run_rankings.py`

Reader compatibility pairs are ranked using a **Bradley-Terry (BT) probability model** fit from implicit pairwise comparisons. When a user views one match over another, that comparison is recorded in BigQuery via `POST /feedback`. Engagement scores from the `conversations` table are used as comparison weights — a comparison from a deeply-engaged session counts more than a passing one.

### Model Fitting

The BT model is fit by minimising the negative log-likelihood of the observed win/loss outcomes across all comparisons:

```python
def neg_log_likelihood(log_scores):
    scores = np.exp(log_scores)
    loss = 0.0
    for (winner, loser), weight in zip(comparisons_list, weights):
        sw, sl = scores[idx[winner]], scores[idx[loser]]
        loss -= weight * np.log(sw / (sw + sl) + 1e-10)
    return loss

result = minimize(neg_log_likelihood, x0=np.zeros(n), method="L-BFGS-B")
scores = np.exp(result.x)
scores /= scores.sum()   # normalise to sum to 1.0
```

Fitting is done at the user level when ≥ 5 comparisons are available. Below this threshold, the **global BT model** (fit on all users' comparisons combined) is used as a cold-start fallback — ensuring new users get ranked results immediately even before they've given any feedback.

### Blend Scoring

The final rank score blends the BT probability with the raw compatibility confidence score. The BT weight grows linearly with the number of comparisons available, capping at 0.70:

```python
def blend_weights(n_comparisons: int) -> tuple[float, float]:
    bt_weight   = min(0.7, 0.1 + (n_comparisons / 50) * 0.6)
    conf_weight = round(1.0 - bt_weight, 2)
    return conf_weight, round(bt_weight, 2)

blend_score = conf_weight * confidence + bt_weight * bt_score_normalised
```

For a new user with 0 comparisons: `(conf=1.0, bt=0.0)` — pure confidence ordering.
For an active user with 50+ comparisons: `(conf=0.3, bt=0.7)` — BT-dominant ordering.

### Aggregation Levels

Rankings and compatibility scores are computed and stored at three levels:

- **Passage-level:** R/C/D scores for a specific reader pair on a specific passage — the atomic unit
- **Book-level** (`aggregate_book_level`): Averaged R/C/D across all passages in a book for a reader pair. Includes a `verdict` field (overall dominant dimension) and `passage_count`
- **Profile-level** (`aggregate_profile_level`): Averaged R/C/D across all books for a reader pair. Includes `book_count` and an overall `verdict`

All three levels are stored in BigQuery (`compatibility_results`, `book_compatibility`, `profile_compatibility`) and updated on every `refit_user()` call.

### Streaming Buffer Handling

BigQuery streaming inserts (used for low-latency writes) go into a streaming buffer that blocks `DELETE` DML for up to 90 minutes. `refit_user()` handles this gracefully:

```python
try:
    client.query("DELETE FROM rankings WHERE user_id = ...").result()
except Exception as e:
    if "streaming buffer" in str(e).lower():
        print(f"[BT] streaming buffer active, skipping delete")
    else:
        raise
```

When the delete is skipped, `get_rankings()` deduplicates on read using `ROW_NUMBER() OVER (PARTITION BY user_id, book_id, passage_id ORDER BY generated_at DESC)`.

### Serving Rankings

```
GET /rankings/{user_id}?book_id=frankenstein&passage_id=passage_1&k=5
```

Returns top-k matches sorted by `blend_score`. Refits BT synchronously on every call. Each ranking row includes: `rank_position`, `run_id`, `match_user`, `verdict`, `confidence`, `bt_score`, `blend_score`, `conf_weight`, `bt_weight`, `n_comparisons`.

---

## 14. Containerization

**File:** `fastapi_pipeline/Dockerfile`

The production FastAPI pipeline is containerised for deployment to Google Cloud Run:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Three agent images are built and pushed to GCP Artifact Registry in CI/CD Stage 3: `decompose-agent`, `compat-agent`, and `profile-agent`. Each image is tagged with the Git SHA (immutable, for rollback), `latest` (current), and `production-stable` (last successfully deployed) to support the CI/CD rollback mechanism.

The `docker-compose.yaml` at the repo root runs a local Airflow + PostgreSQL stack for developing and testing the data pipeline without GCP access:

```bash
docker-compose up --build
# Airflow UI available at http://localhost:8080
# PostgreSQL available at localhost:5432
```

---

## 15. Data Versioning with DVC

**File:** `dvc.yaml`

All processed data in `data/processed/` is tracked with DVC, backed by a GCS bucket. DVC ensures that every version of the processed dataset is reproducible and that large data files are never committed to git.

```bash
# Pull the latest processed data from GCS
dvc pull

# After running the pipeline to generate new processed data:
dvc add data/processed/
dvc push

# Commit the updated .dvc pointer file to git
git add data/processed.dvc
git commit -m "Update processed data"
git push
```

The `dvc.yaml` file defines pipeline stages with explicit dependencies (inputs) and outputs. Running `dvc repro` will re-execute only the stages whose dependencies have changed since the last run, ensuring efficient reproduction of results.

---

## 16. Testing

**Locations:** `data_pipeline/tests/` · `tests/`

```bash
# Run all tests
pytest

# Run data pipeline tests only
pytest data_pipeline/tests/ -v

# Run with HTML coverage report
pytest --cov=fastapi_pipeline --cov-report=html
```

### Test Files

| Test file | What it covers |
|---|---|
| `test_acquisition.py` | Cloud SQL connector, `DATE(created_at) = CURRENT_DATE` filter, schema of returned DataFrames |
| `test_preprocessing.py` | `clean_text()`, `validate_text()`, `detect_issues()`, `calculate_metrics()` — edge cases including empty text, PII patterns, profanity ratios, non-English detection, gibberish detection |
| `test_validation.py` | TFDV schema validation — checks that known-bad records trigger anomaly flags and known-good records pass cleanly |
| `test_bias_detection.py` | Slice analysis — checks that artificially imbalanced distributions are correctly detected above the threshold and that balanced distributions pass |
| `test_schema_stats.py` | TFDV statistics generation — checks output JSON structure, required field coverage, and that statistics are computed correctly for categorical and numeric features |
| `test_pipeline.py` | End-to-end integration — runs the full pipeline from `CloudSQLLoader` through `write_to_bq` with a mock Cloud SQL fixture and BQ client |

### Tests DAG

The Airflow tests DAG (`data_pipeline/airflow/dags/tests_dag.py`) runs the pytest suite as an Airflow task so test results are visible in the Airflow UI alongside regular pipeline runs.

![Tests Pipeline Gantt](docs/tests_pipeline_gantt.png)

---

## 17. Sensitivity Analysis

**File:** `model_sensitivity_analysis.py`

Sensitivity analysis examines how much each component of the compatibility pipeline contributes to the final score, verifying that alignment is based on genuine intellectual depth rather than surface-level text similarity.

### Analyses Performed

**Sub-claim weight sensitivity:** How much does confidence change when individual sub-claim weights are perturbed? High sensitivity to the single highest-weight sub-claim indicates over-reliance on one claim, which could be gamed by users who write long interpretations with one dominant idea.

**Emotional mode distribution:** Does the distribution of emotional modes across a user's moments correlate with their compatibility scores? This checks whether the model inadvertently penalises readers who predominantly use a single mode (e.g. always philosophical) versus readers who use a variety.

**Word count sensitivity:** Does confidence correlate with interpretation length? A well-argued short moment should score as highly as a long one — the model should reward intellectual depth, not verbosity.

**Match count sensitivity:** How does confidence change as more sub-claims are matched versus unmatched? Given that coverage has the highest weight in the confidence formula (0.40), this relationship should be strongly monotonically increasing — a good sanity check.

**SHAP integration:** SHAP values are computed on the aggregated score output to identify which features contribute most to the `dominant_think` verdict. High SHAP values for shared textual anchors (matched sub-claims with high gate_confidence) and compatible Think positions confirm the model is rewarding genuine intellectual alignment rather than superficial text similarity.

---

## 18. Data Drift — Current Status

> **Note:** Full data drift monitoring is instrumented but not yet actively used in production decision-making. This is an intentional limitation of the current stage of the project.

The monitoring infrastructure in `metrics.py` and the Grafana dashboard already track per-run distribution signals — `valid_ratio`, `word_count_mean/p50/p95`, `quality_score_mean/p10`, and `readability_mean` — specifically designed to detect data drift over time. However, since the current dataset is **entirely synthetic** (450 interpretations generated from 50 character personas), these signals do not exhibit meaningful drift. Every pipeline run sees statistically similar data because it comes from the same controlled generation process.

Meaningful data drift monitoring will become active once the platform has a real user base. At that point, shifts in these metrics will reflect genuine changes in how users write — for example, interpretations getting shorter over time, quality scores dropping as the user base grows, or the valid_ratio falling if users start writing in non-English or testing the system with low-quality inputs. When that happens, sustained anomalies in any of these gauges will fire Alertmanager alerts, which route to the `/admin/retrain-trigger` endpoint and queue a full pipeline refit.

For now, the drift signals are collected and visible in the Grafana dashboard as a baseline record, so that when real user data begins flowing, there will be a historical baseline to compare against from day one.

---

## 19. Evaluation Criteria Coverage

### Data Pipeline

| # | Criterion | How Met | Key Files |
|---|---|---|---|
| 1 | Data Acquisition | Cloud SQL connector with today-only filtering, 3 linked table loads, mock-testable | `data_pipeline/scripts/data_acquisition.py`, `fastapi_pipeline/cloudsql_loader.py` |
| 2 | Preprocessing | 5 sequential quality gates, Unicode normalisation, readability + quality scoring, PII/profanity/spam scrubbing | `fastapi_pipeline/preprocessor_fastapi.py`, `data_pipeline/scripts/preprocessor.py` |
| 3 | Schema Validation (TFDV) | TFDV statistics generation, schema `.pbtxt` files, anomaly detection with report output | `data_pipeline/scripts/generate_schema_stats.py`, `data_pipeline/config/schema.yaml` |
| 4 | Anomaly Detection | Distribution-based anomaly detection with per-field alert generation | `data_pipeline/scripts/anomalies.py` |
| 5 | Bias Detection | Pre-preprocessing slice analysis by gender/readership; post-model CI/CD bias check | `data_pipeline/scripts/bias_detection.py`, `cicd_pipeline/bias_detection.py` |
| 6 | Airflow DAGs + Gantt | Multi-level DAG with parallel `validation` + `schema_stats` branches, Gantt-optimised | `data_pipeline/airflow/dags/` |
| 7 | DVC Versioning | GCS-backed data versioning, reproducible `dvc repro` pipeline | `dvc.yaml` |
| 8 | Unit Testing | 6 test modules — acquisition, preprocessing, validation, bias, schema stats, end-to-end | `data_pipeline/tests/` |

### Model Pipeline

| # | Criterion | How Met | Key Files |
|---|---|---|---|
| 1 | Model Documentation | Full 3-stage agent pipeline with system prompts, scoring formulas, aggregation logic, and output schemas | Sections 6–12 above |
| 2 | Automated Validation | Three-metric gate (confidence, schema pass rate, error rate) blocks deployment on regression | `cicd_pipeline/validate_model.py` |
| 3 | Bias Detection | Multi-slice confidence gap analysis on validation set outputs, integrated into CI/CD Stage 2 | `cicd_pipeline/bias_detection.py` |
| 4 | Explainability (SHAP) | Sub-claim weight perturbation analysis + SHAP values on aggregated compatibility scores | `model_sensitivity_analysis.py` |
| 5 | Experiment Tracking | Hierarchical MLflow runs (parent + 2 child runs per pair) with params, metrics, tags, and JSON artifacts | `experiment_tracking/` |
| 6 | Containerization | Three agent Docker images versioned by Git SHA, deployed to Vertex AI Agent Engine | `fastapi_pipeline/Dockerfile`, `cicd_pipeline/` |
| 7 | CI/CD | 5-stage GitHub Actions workflow: test → validate/bias → build/push → rollback check → deploy | `.github/workflows/cicd.yml` |
| 8 | Rollback & Alerts | Threshold-based regression detection, `production-stable` re-tagging, Slack notifications at every gate | `cicd_pipeline/rollback.py`, `cicd_pipeline/notifications.py` |
| 9 | Monitoring | GCM custom metrics with 4 categories (pipeline/preprocessing/compatibility/rankings), Grafana dashboard, Alertmanager → auto-retrain webhook | `fastapi_pipeline/metrics.py`, `monitoring/` |
| 10 | Production Pipeline | Daily FastAPI pipeline on Cloud Run with BQ caching, background compat scoring, BT rankings, and `/admin/retrain-trigger` webhook | `fastapi_pipeline/main.py` |

---

*IE7374 · MLOps · Group 23 · Northeastern University · April 2026*
