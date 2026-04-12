import json
import textwrap
from datetime import datetime
from google import genai
from google.genai import types # type: ignore

from tools import (
    extract_json,
    get_decomposition,
    get_scoring,
    save_scoring,
    get_compat_run,
    log_compat_run,
)
from decomposing_agent import (
    _gemini_client,
    _GEMINI_MODEL,
    _get_response_text,
    run_decomposer,
)
from aggregator import aggregate

# ── Compatibility Agent ───────────────────────────────────────────────────────

_COMPAT_SYSTEM_PROMPT = """
You are the Moment Compatibility Scorer.
You receive pre-decomposed sub-claims from two readers (A and B) for the same
passage. Map them, score each pair, return JSON.

════════════════════════════════════════════════════════════
CRITICAL — THINK AND FEEL ARE INDEPENDENT
════════════════════════════════════════════════════════════

Score them separately. Two readers can share a conclusion but feel differently.
Two readers can disagree intellectually but share the same emotional response.
Never let Think scores contaminate Feel scores.

════════════════════════════════════════════════════════════
STEP 1 — MAP SUB-CLAIMS 
════════════════════════════════════════════════════════════

Match each A sub-claim to the best B candidate.
Two sub-claims match if they respond to the same passage phrase or the same
specific moment — even if worded differently.
If no genuine match exists, mark UNMATCHED.
Each sub-claim matched at most once. Unclaimed B sub-claims → UNMATCHED.

════════════════════════════════════════════════════════════
STEP 2 — SCORE EACH MATCHED PAIR
════════════════════════════════════════════════════════════

For each matched pair answer 10 booleans — 5 THINK then 5 FEEL.

THINK — T1. Same subject (always true) | T2. Positions same direction? |
  T3. Lenses compatible? | T4. Mutually exclusive conclusions? | T5. Would A agree with B?

FEEL  — F1. Same emotional subject (always true) | F2. Same mode label? |
  F3. Same trigger? | F4. Same experience? | F5. Would A recognise B's response as valid?

════════════════════════════════════════════════════════════
UNMATCHED SUB-CLAIMS
════════════════════════════════════════════════════════════

B engages the same subject but not the same specific phrase or moment → divergence: true
B has nothing on this subject                                         → divergence: false

════════════════════════════════════════════════════════════
OUTPUT — raw JSON only, no markdown, no preamble
════════════════════════════════════════════════════════════

{
  "passage_id": "<id>",
  "matched_pairs": [
    {
      "a_id": "<id>", "b_id": "<id>",
      "weight_a": <float>, "weight_b": <float>,
      "gate_confidence": 1.0 or 0.5,
      "think_q": [<bool>, <bool>, <bool>, <bool>, <bool>],
      "feel_q":  [<bool>, <bool>, <bool>, <bool>, <bool>]
    }
  ],
  "unmatched_a": [{"id": "<id>", "divergence": <bool>}],
  "unmatched_b": [{"id": "<id>", "divergence": <bool>}],
  "think_rationale": "<1-2 sentences>",
  "feel_rationale": "<1-2 sentences>"
}
"""

_SCORER_REQUIRED_KEYS = {"matched_pairs", "unmatched_a", "unmatched_b"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_or_run_decomposition(user_id: str, passage_id: str, book_id: str, moment_text: str) -> dict:
    """Return cached BQ decomposition or run the decomposer and save result."""
    cached = get_decomposition(user_id, passage_id, book_id)
    if cached:
        print(f"[CompatAgent] cached decomposition for {user_id} / {passage_id}")
        return cached
    return run_decomposer(user_id, passage_id, book_id, moment_text)


def _build_scorer_prompt(decomp_a: dict, decomp_b: dict) -> str:
    """Build the user-turn message for the scorer from two decompositions."""
    return textwrap.dedent(f"""
        Reader A:
        {json.dumps(decomp_a, indent=2)}

        Reader B:
        {json.dumps(decomp_b, indent=2)}

        Map and score these two decomposed moments.
        Return a raw JSON object.
    """)


def _call_scorer(prompt: str) -> dict:
    """Send decomposed profiles to the scorer and return parsed result."""
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
        print(f"[CompatAgent] JSON extraction failed. raw={raw[:200]}")
        return {"error": "invalid JSON from Compatibility Scorer", "raw": raw}

    missing = _SCORER_REQUIRED_KEYS - result.keys()
    if missing:
        print(f"[CompatAgent] incomplete result, missing keys: {missing}")
        result["error"] = f"incomplete result, missing: {missing}"

    return result


# ── Agent ─────────────────────────────────────────────────────────────────────

def run_compatibility_agent(user_a_id: str,
                             user_b_id: str,
                             book_id: str,
                             moment_a: dict,
                             moment_b: dict) -> dict:
    """
    Evaluate compatibility between two readers on the same passage.

    Decomposes each moment (or uses BQ cache), scores matched pairs,
    then aggregates into final R/C/D percentages + confidence.
    Logs result to BQ via tools.log_compat_run().

    Returns a dict with keys: think, feel, dominant_think, dominant_feel,
    verdict, confidence, user_a, user_b, book_id, timestamp.
    """
    print(f"[CompatAgent] evaluating {user_a_id} × {user_b_id} on book={book_id}")

    passage_id   = moment_a.get("passage_id", moment_b.get("passage_id", "unknown"))
    moment_a_txt = moment_a.get("cleaned_interpretation", moment_a.get("interpretation", moment_a.get("text", "")))
    moment_b_txt = moment_b.get("cleaned_interpretation", moment_b.get("interpretation", moment_b.get("text", "")))

    # ── 1. Check for existing compat run in BQ ────────────────────────────────
    existing = get_compat_run(user_a_id, user_b_id, book_id, passage_id)
    if existing:
        print(f"[CompatAgent] cached compat run for {user_a_id} × {user_b_id}")
        return existing

    # ── 2. Decompose ──────────────────────────────────────────────────────────
    decomp_a = _get_or_run_decomposition(user_a_id, passage_id, book_id, moment_a_txt)
    decomp_b = _get_or_run_decomposition(user_b_id, passage_id, book_id, moment_b_txt)

    if "error" in decomp_a or "error" in decomp_b:
        failed = user_a_id if "error" in decomp_a else user_b_id
        print(f"[CompatAgent] decomposition failed for {failed}")
        return {"error": f"decomposition failed for {failed}",
                "user_a": user_a_id, "user_b": user_b_id}

    # ── 3. Score ──────────────────────────────────────────────────────────────
    scoring = get_scoring(user_a_id, user_b_id, passage_id, book_id)
    if scoring:
        print(f"[CompatAgent] cached scoring for {user_a_id} × {user_b_id} / {passage_id}")
    else:
        prompt  = _build_scorer_prompt(decomp_a, decomp_b)
        scoring = _call_scorer(prompt)
        if "error" not in scoring:
            save_scoring(user_a_id, user_b_id, passage_id, book_id, scoring)

    if "error" in scoring:
        return {"error": scoring["error"],
                "user_a": user_a_id, "user_b": user_b_id}

    # ── 4. Aggregate ──────────────────────────────────────────────────────────
    result = aggregate(decomp_a, decomp_b, scoring, book_id, passage_id)
    if result is None:
        return {"error": "aggregation returned no result",
                "user_a": user_a_id, "user_b": user_b_id}

    print(f"[CompatAgent] confidence={result.get('confidence', 0.0)}")

    # ── 5. Attach metadata and log to BQ ─────────────────────────────────────
    result["user_a"]    = user_a_id
    result["user_b"]    = user_b_id
    result["book_id"]   = book_id
    result["timestamp"] = datetime.utcnow().isoformat()

    log_compat_run(result)

    print(
        f"[CompatAgent] dominant_think={result.get('dominant_think')} "
        f"dominant_feel={result.get('dominant_feel')} "
        f"confidence={result.get('confidence')} "
        f"for {user_a_id} × {user_b_id}"
    )
    return result


# ── Uncertainty Router ────────────────────────────────────────────────────────

def route_compatibility_result(result: dict) -> str:
    if "error" in result:
        return "discard"
    return "display"