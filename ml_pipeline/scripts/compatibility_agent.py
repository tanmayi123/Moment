import json
import logging
import os
import textwrap
from datetime import datetime

import pandas as pd
from google import genai
from google.genai import types  # type: ignore

from tools import extract_json

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger("momento.compat")

# ── Config ────────────────────────────────────────────────────────────────────

BQ_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "moment-486719")

# ── BQ Helpers ────────────────────────────────────────────────────────────────

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

# ── Gemini client — imported from decomposing_agent to avoid re-init ──────────
# These are imported inside functions to avoid module-level import errors
# when sys.path isn't yet configured (e.g. during Airflow DAG parsing).

def _get_gemini():
    from decomposing_agent import _gemini_client, _GEMINI_MODEL, _get_response_text
    return _gemini_client, _GEMINI_MODEL, _get_response_text

# ── Scorer system prompt ──────────────────────────────────────────────────────

_COMPAT_SYSTEM_PROMPT = textwrap.dedent("""
You are the Moment Compatibility Scorer for Momento.
You will be given two decomposed reader profiles — sub-claims with
weights and emotional modes already assigned — along with the
original moment text for each reader.
Use the original moment texts as context when scoring matched pairs,
especially when judging whether positions point in the same direction
(Think Q2) or whether readers share the same emotional experience
(Feel Q4). The decomposed sub-claims define the structure.
The original texts provide the interpretive framing.
Map sub-claims across readers, then score each matched pair.
Output only the final JSON — no preamble, no markdown fences.

════════════════════════════════════════════════════════════
STEP 1 — MAP USING THREE GATES
════════════════════════════════════════════════════════════

For each sub-claim in A, find the best candidate in B.
All three gates must pass for a match.

GATE 1 — Shared subject
Both sub-claims make a claim about the same specific subject?
(same character, same named aspect, same event in the passage)

AUTOMATIC YES: if both reference the same specific phrase from
the passage text — different angles on the same phrase are Gate 2
and Gate 3 questions, not different subjects.

Note: character evasion ≠ authorial choice even if both about language
Note: pre-creation moment ≠ post-creation moment even if same character
YES → Gate 2   |   NO → UNMATCHED

GATE 2 — Shared textual anchor
At least one applies:
  a. Both reference the same quoted phrase from the passage
  b. Both describe the same specific moment or event in the passage
     (same action, same beat — even if described in different words
     or anchored to different passage phrases)
  c. Both sub-claims respond to the same specific character behaviour
     or authorial choice AND make claims that could directly agree or
     disagree with each other.
     Note: "same character" alone is not sufficient for 2c — must be
     the same specific behaviour or action of that character.
YES → Gate 3   |   NO → UNMATCHED

GATE 3 — Meaningful comparison
Formulate: "Is it true that [A's claim]?"
Would B's claim answer YES or NO?
YES or NO → MATCHED   |   Irrelevant → UNMATCHED

Each sub-claim can only be matched once.
Any B sub-claim not claimed by a matched A sub-claim → UNMATCHED.

════════════════════════════════════════════════════════════
STEP 2 — SCORE EACH MATCHED PAIR
════════════════════════════════════════════════════════════

For each matched pair, answer 5 Think questions then 5 Feel questions.
Each YES or NO maps to 1R or 1C as specified.
score = points / 5  (D = 0.0 for matched pairs)

THINK QUESTIONS:
  Q1. Same subject? Always YES for matched pairs → 1R
  Q2. Positions same direction? YES=1R  NO=1C
  Q3. Interpretive lenses compatible? YES=1R  NO=1C
  Q4. Conclusions mutually exclusive? YES=1C  NO=1R
  Q5. Would A agree with B's conclusion? YES=1R  NO=1C

FEEL QUESTIONS:
  Q1. Same emotional subject? Always YES for matched pairs → 1R
  Q2. Same emotional mode TYPE? Compare the assigned modes —
      same label → YES (1R)  |  different labels → NO (1C)
  Q3. Same specific trigger? YES=1R  NO=1C
  Q4. Same emotional experience? YES=1R  NO=1C
  Q5. Would A recognise B's response as valid? YES=1R  NO=1C

For UNMATCHED sub-claims:
  think: R=0.0  C=0.0  D=1.0
  feel:  R=0.0  C=0.0  D=1.0

════════════════════════════════════════════════════════════
GATE CONFIDENCE — assign for every matched pair
════════════════════════════════════════════════════════════

  1.00 = both gates passed clearly
  0.75 = one gate required interpretation
  0.50 = borderline match
  If zero matched pairs → omit gate_confidence entirely.

════════════════════════════════════════════════════════════
OUTPUT — return ONLY this JSON, no preamble, no markdown fences
════════════════════════════════════════════════════════════

{
  "passage_id": "<passage identifier>",
  "matched_pairs": [
    {
      "a_id": "<sub-claim id from A>",
      "b_id": "<sub-claim id from B>",
      "weight_a": <float>,
      "weight_b": <float>,
      "gate_confidence": <1.00|0.75|0.50>,
      "think": {"R": <float>, "C": <float>, "D": 0.0},
      "feel":  {"R": <float>, "C": <float>, "D": 0.0}
    }
  ],
  "unmatched_a": ["<sub-claim id>"],
  "unmatched_b": ["<sub-claim id>"]
}
""")

_SCORER_REQUIRED_KEYS = {"matched_pairs", "unmatched_a", "unmatched_b"}

# ── Scorer helpers ────────────────────────────────────────────────────────────

def _build_scorer_prompt(decomp_a: dict, decomp_b: dict) -> str:
    return textwrap.dedent(f"""
        Reader A:
        {json.dumps(decomp_a, indent=2)}

        Reader B:
        {json.dumps(decomp_b, indent=2)}

        Map and score these two decomposed moments.
        Return a raw JSON object.
    """)


def _call_scorer(prompt: str) -> dict:
    _gemini_client, _GEMINI_MODEL, _get_response_text = _get_gemini()

    response = _gemini_client.models.generate_content(
        model=_GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=_COMPAT_SYSTEM_PROMPT,
            temperature=0.1,
        ),
        contents=prompt,
    )
    result = extract_json(_get_response_text(response))

    if "error" in result:
        raw = _get_response_text(response) or ""
        logger.error(f"[CompatAgent] JSON extraction failed. raw={raw[:200]}")
        return {"error": "invalid JSON from Compatibility Scorer", "raw": raw}

    missing = _SCORER_REQUIRED_KEYS - result.keys()
    if missing:
        logger.warning(f"[CompatAgent] incomplete result, missing keys: {missing}")
        result["error"] = f"incomplete result, missing: {missing}"

    return result


def _save_scoring(user_a_id: str, user_b_id: str, passage_id: str,
                  scoring: dict, run_id: str) -> None:
    """Upsert raw scorer output into BQ staging scoring_runs table."""
    tid = bq_table_id(run_id, "scoring_runs")
    try:
        existing_df = bq_read(tid)
        rows = existing_df.to_dict("records")
    except Exception:
        rows = []

    # Remove previous run for this pair+passage
    rows = [r for r in rows
            if not (r["user_a_id"] == user_a_id
                    and r["user_b_id"] == user_b_id
                    and r["passage_id"] == passage_id)]
    rows.append({
        "user_a_id":  user_a_id,
        "user_b_id":  user_b_id,
        "passage_id": passage_id,
        "timestamp":  datetime.utcnow().isoformat(),
        "scoring":    json.dumps(scoring),
    })
    bq_write(pd.DataFrame(rows), tid)
    logger.info(f"[CompatAgent] scoring saved for {user_a_id}×{user_b_id} / {passage_id} → {tid}")


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _get_existing_compat_run(user_a_id: str, user_b_id: str,
                              book_id: str, passage_id: str,
                              run_id: str) -> dict | None:
    tid = bq_table_id(run_id, "compatibility_results")
    try:
        df = bq_read(tid)
    except Exception:
        return None

    runs = df.to_dict("records")
    matches = [
        r for r in runs
        if r.get("book_id") == book_id and r.get("passage_id") == passage_id
        and (
            (r.get("user_a") == user_a_id and r.get("user_b") == user_b_id)
            or
            (r.get("user_a") == user_b_id and r.get("user_b") == user_a_id)
        )
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda r: r.get("timestamp", ""), reverse=True)[0]


def route_compatibility_result(result: dict) -> str:
    return "discard" if "error" in result else "display"


# ── Main agent ────────────────────────────────────────────────────────────────

def run_compatibility_agent(user_a_id: str, user_b_id: str, book_id: str,
                             moment_a: dict, moment_b: dict,
                             run_id: str) -> dict:
    """
    Evaluate compatibility between two readers on the same passage.
    Decomposes each moment (or uses cache), scores matched pairs,
    then aggregates into final R/C/D percentages + confidence.
    """
    from decomposing_agent import run_decomposer
    from aggregator import aggregate

    logger.info(f"[CompatAgent] evaluating {user_a_id}×{user_b_id} book={book_id}")

    passage_id = moment_a.get("passage_id", moment_b.get("passage_id", "unknown"))

    # ── Check cache ───────────────────────────────────────────────────────────
    existing = _get_existing_compat_run(user_a_id, user_b_id, book_id, passage_id, run_id)
    if existing:
        route = route_compatibility_result(existing)
        logger.info(f"[CompatAgent] cached result found — verdict={existing.get('verdict')} "
                    f"confidence={existing.get('confidence')} route={route}")
        return existing if route != "discard" else {}

    # ── Decompose ─────────────────────────────────────────────────────────────
    decomp_tid = bq_table_id(run_id, "decompositions")
    try:
        decomp_df = bq_read(decomp_tid)
        decomp_cache = {
            (r["user_id"], r["passage_id"], r["book_id"]): r
            for r in decomp_df.to_dict("records")
            if "error" not in r
        }
    except Exception:
        decomp_cache = {}
        logger.debug("[CompatAgent] no decomposition cache found — running fresh")

    def _get_decomp(uid: str, moment: dict) -> dict:
        key = (uid, passage_id, book_id)
        if key in decomp_cache:
            logger.debug(f"[CompatAgent] decomp cache hit for {uid}")
            return decomp_cache[key]
        logger.info(f"[CompatAgent] decomposing {uid}")
        return run_decomposer(moment)

    decomp_a = _get_decomp(user_a_id, moment_a)
    decomp_b = _get_decomp(user_b_id, moment_b)

    if "error" in decomp_a or "error" in decomp_b:
        failed = user_a_id if "error" in decomp_a else user_b_id
        logger.error(f"[CompatAgent] decomposition failed for {failed}")
        return {"error": f"decomposition failed for {failed}",
                "user_a": user_a_id, "user_b": user_b_id}

    # ── Score ─────────────────────────────────────────────────────────────────
    prompt  = _build_scorer_prompt(decomp_a, decomp_b)
    scoring = _call_scorer(prompt)
    _save_scoring(user_a_id, user_b_id, passage_id, scoring, run_id)

    if "error" in scoring:
        logger.error(f"[CompatAgent] scoring failed: {scoring['error']}")
        return {"error": scoring["error"], "user_a": user_a_id, "user_b": user_b_id}

    # ── Aggregate ─────────────────────────────────────────────────────────────
    result = aggregate({"reader_a": decomp_a, "reader_b": decomp_b}, scoring)
    result.update({
        "user_a":    user_a_id,
        "user_b":    user_b_id,
        "book_id":   book_id,
        "timestamp": datetime.utcnow().isoformat(),
    })

    logger.info(
        f"[CompatAgent] done — dominant_think={result.get('dominant_think')} "
        f"dominant_feel={result.get('dominant_feel')} "
        f"confidence={result.get('confidence')} "
        f"for {user_a_id}×{user_b_id}"
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    results_tid = bq_table_id(run_id, "compatibility_results")
    try:
        existing_df = bq_read(results_tid)
        all_results = existing_df.to_dict("records")
    except Exception:
        all_results = []

    keyed = {(r["user_a"], r["user_b"], r["book_id"], r["passage_id"]): r
             for r in all_results}
    keyed[(user_a_id, user_b_id, book_id, passage_id)] = result
    bq_write(pd.DataFrame(list(keyed.values())), results_tid)
    logger.info(f"[CompatAgent] result persisted → {results_tid}")

    return result