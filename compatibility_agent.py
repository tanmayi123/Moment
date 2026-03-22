import json
import re
import os
import textwrap
from datetime import datetime
from google import genai
from google.genai import types

from tools import (
    COMPAT_LOG_FILE,
    _read_json_file,
    _write_json_file,
    get_user_interpretations,
    get_user_profile,
    count_new_moments,
    log_compatibility_run,
    extract_json,
    COMPAT_TOOLS,
)
from profile_agent import (
    _gemini_client,
    _GEMINI_MODEL,
    _get_response_text,
    run_profile_agent,
)

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD  = 0.75  # Pass 1 result accepted as-is above this
CONFIDENCE_FLOOR      = 0.50  # Pass 1 results below this are discarded immediately

# ── Compatibility Agent ───────────────────────────────────────────────────────

_COMPAT_SYSTEM_PROMPT = textwrap.dedent("""
    You are the Compatibility Investigator for Moment, a platform that matches
    readers of the same book based on how they intellectually engage with it.

    You will be given:
    - One moment (annotation) from User A
    - One moment (annotation) from User B
    - The book_id being compared
    - Optionally: full reader portraits for both users (provided when the
      initial moment-based assessment yields low confidence)
      
    DIMENSIONS
    THINK (how_they_read + interpretive_lens + central_preoccupation):

    resonance:     same level of text, compatible frameworks, aligned position
    contradiction: same level and framework, opposing positions on the same question
    divergence:    different levels OR incommensurable frameworks

    FEEL (what_moves_them + emotional_mode + self_referential):

    resonance:     similar emotional triggers, compatible emotional modes
    contradiction: prosecutorial paired with empathetic-victim
    divergence:    different triggers OR one self-referential and one not

    Your job is to investigate whether these two readers would have a meaningful
    intellectual connection. Use get_user_interpretations to examine their actual
    moments for this specific book before forming any verdict.
    
    Reasoning steps — follow in order:
    1. Read both portraits carefully. Are they fundamentally incompatible?
       If yes, set verdict to "no_match" and stop.

    2. If portraits are provided, use them to deepen your assessment —
       situate each moment within the broader reading style and thematic
       tendencies described in the portrait.

    3. Determine the nature of the connection:
       - resonance: similar emotional and intellectual stance toward the book
       - contradiction: opposite sides of the same dynamic (e.g. one identifies with
         character A, the other with character B in the same conflict)
       - divergence: genuinely different interpretive frameworks that could
         produce productive tension
       - no_match: no meaningful basis for connection

    4. Assign a confidence score (0.0–1.0) reflecting how strongly the evidence
       supports your verdict.

    5. Write an insight — 2-3 sentences a real user would see. Be specific.
       Reference the actual moments or portrait details. Do not be generic.
       Set to null if verdict is no_match.

    Return ONLY a raw JSON object — no markdown fences, no preamble.
    Keys: verdict, confidence, reasoning, insight.
""")


_COMPAT_REQUIRED_KEYS = {"think_dimension", "feel_dimension",
                         "verdict", "confidence", "reasoning", "insight"}
# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_portrait(user_id: str) -> dict:
    """
    Return a portrait for user_id. Uses cached portrait if fresh
    (fewer than 3 new moments since last update), otherwise rebuilds.
    """
    existing = get_user_profile(user_id)
    if existing:
        return existing
    return run_profile_agent(user_id)


def _build_prompt(user_a_id: str,
                  user_b_id: str,
                  book_id: str,
                  moment_a: dict,
                  moment_b: dict,
                  portrait_a: dict | None = None,
                  portrait_b: dict | None = None) -> str:
    """Construct the investigation prompt, optionally including portraits."""
    portrait_section = ""
    if portrait_a and portrait_b:
        portrait_section = textwrap.dedent(f"""
        The initial moment-only assessment had low confidence.
        Use the full reader portraits below to deepen your judgement.

        Portrait A:
        {json.dumps(portrait_a, indent=2)}

        Portrait B:
        {json.dumps(portrait_b, indent=2)}
        """)

    return textwrap.dedent(f"""
        Investigate compatibility between these two readers for book_id: {book_id}.

        User A ID: {user_a_id}
        Moment A:
        {json.dumps(moment_a, indent=2)}

        User B ID: {user_b_id}
        Moment B:
        {json.dumps(moment_b, indent=2)}
        {portrait_section}
        Follow the reasoning steps in your instructions.
        Return a raw JSON object.
    """)


def _call_agent(prompt: str) -> dict:
    """Send a prompt to the compatibility agent and return the parsed result."""
    chat = _gemini_client.chats.create(
        model=_GEMINI_MODEL,
        config=types.GenerateContentConfig(
            tools=COMPAT_TOOLS,
            temperature=0.2,
            system_instruction=_COMPAT_SYSTEM_PROMPT,
        )
    )
    response = chat.send_message(prompt)
    result   = extract_json(_get_response_text(response))

    if "error" in result:
        print(f"[CompatAgent] JSON extraction failed. raw={response.text[:200]}")
        return {"error": "invalid JSON from Compatibility Agent",
                "raw": response.text}

    missing = _COMPAT_REQUIRED_KEYS - result.keys()
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
    Evaluate intellectual compatibility between two readers of the same book.

    Pass 1 — moments only. If confidence < CONFIDENCE_THRESHOLD, fetches both
    reader portraits and runs a second, deeper Pass 2.

    Args:
        user_a_id: ID of the first user.
        user_b_id: ID of the second user.
        book_id:   The book being compared.
        moment_a:  A single moment/annotation dict for user A.
        moment_b:  A single moment/annotation dict for user B.

    Returns a dict with keys: verdict, confidence, reasoning, insight,
    user_a, user_b, book_id, timestamp, and pass (1 or 2).
    """
    print(f"[CompatAgent] evaluating {user_a_id} × {user_b_id} on book={book_id}")

    # ── Pass 1: moments only ──────────────────────────────────────────────────
    print(f"[CompatAgent] Pass 1 — moments only")
    prompt_1 = _build_prompt(user_a_id, user_b_id, book_id, moment_a, moment_b)
    result   = _call_agent(prompt_1)
    passes   = 1

    if "error" not in result:
        confidence = result.get("confidence", 0.0)
        print(f"[CompatAgent] Pass 1 confidence={confidence}")

        # ── Early exit: discard pairs below the floor ─────────────────────────
        if confidence < CONFIDENCE_FLOOR:
            print(f"[CompatAgent] confidence {confidence} < floor {CONFIDENCE_FLOOR} "
                  f"— discarding without Pass 2")
            result["verdict"] = "no_match"
            result["pass"]    = 1

        # ── Pass 2: augment with portraits for mid-range confidence ───────────
        elif confidence < CONFIDENCE_THRESHOLD:
            print(f"[CompatAgent] confidence {confidence} in [{CONFIDENCE_FLOOR}, {CONFIDENCE_THRESHOLD}) "
                  f"— fetching portraits for Pass 2")
            portrait_a = _ensure_portrait(user_a_id)
            portrait_b = _ensure_portrait(user_b_id)

            if "error" not in portrait_a and "error" not in portrait_b:
                prompt_2 = _build_prompt(
                    user_a_id, user_b_id, book_id,
                    moment_a, moment_b,
                    portrait_a, portrait_b,
                )
                result = _call_agent(prompt_2)
                passes = 2
                print(f"[CompatAgent] Pass 2 confidence={result.get('confidence')}")
            else:
                failed = user_a_id if "error" in portrait_a else user_b_id
                print(f"[CompatAgent] portrait fetch failed for {failed}, "
                      f"keeping Pass 1 result")

    # ── Attach metadata and log ───────────────────────────────────────────────
    result["user_a"]    = user_a_id
    result["user_b"]    = user_b_id
    result["book_id"]   = book_id
    result["pass"]      = passes
    result["timestamp"] = datetime.utcnow().isoformat()

    log_compatibility_run({
        "user_a":           user_a_id,
        "user_b":           user_b_id,
        "book_id":          book_id,
        "moment_a":       moment_a,
        "moment_b":       moment_b,
        "think_dimension":  result.get("think_dimension"),
        "feel_dimension":   result.get("feel_dimension"),
        "verdict":          result.get("verdict"),
        "confidence":       result.get("confidence"),
        "reasoning":        result.get("reasoning"),
        "insight":          result.get("insight"),
        "timestamp":        result["timestamp"],
    })


    print(
        f"[CompatAgent] verdict={result.get('verdict')} "
        f"confidence={result.get('confidence')} "
        f"pass={passes} "
        f"for {user_a_id} × {user_b_id}"
    )
    return result


# ── Uncertainty Router ────────────────────────────────────────────────────────

def route_compatibility_result(result: dict) -> str:
    """
    Decide what to do with a compatibility verdict.

    Returns one of:
      "display"     — high confidence match, show to users
      "deep_review" — low confidence or uncertain, flag for review
      "discard"     — no match or agent error
    """
    if "error" in result:
        return "discard"

    verdict    = result.get("verdict")
    confidence = result.get("confidence", 0.0)

    if verdict == "no_match":
        return "discard"

    if verdict == "uncertain":
        return "deep_review"

    if confidence >= CONFIDENCE_THRESHOLD:
        return "display"

    if confidence >= 0.5:
        return "deep_review"

    return "discard"


# ── Backwards compatibility aliases ──────────────────────────────────────────

run_compatibility_agent_v1 = run_compatibility_agent
run_compatibility_agent_v2 = run_compatibility_agent


# ── Entrypoint ────────────────────────────────────────────────────────────────

def _get_existing_compat_run(user_a_id: str,
                              user_b_id: str,
                              book_id: str) -> dict | None:
    """
    Return the most recent logged compatibility run for this pair and book,
    or None if it doesn't exist yet.
    Checks both orderings (A×B and B×A) since the pair is symmetric.
    """
    runs = _read_json_file(COMPAT_LOG_FILE, [])
    matches = [
        r for r in runs
        if r.get("book_id") == book_id
        and (
            (r.get("user_a") == user_a_id and r.get("user_b") == user_b_id)
            or
            (r.get("user_a") == user_b_id and r.get("user_b") == user_a_id)
        )
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda r: r.get("timestamp", ""), reverse=True)[0]


def run_compatibility_for_all(user_a_id: str,
                               book_id: str,
                               moments_map: dict[str, dict]) -> list[dict]:
    """
    Run the compatibility agent between user_a_id and every other user
    in moments_map for the given book.

    Args:
        user_a_id:   The anchor user.
        book_id:     The book being compared.
        moments_map: Dict mapping user_id → moment dict. Must include
                     user_a_id and all candidate user IDs.
                     e.g. {"user_a": {...}, "user_b": {...}, "user_c": {...}}

    Returns a list of results sorted by confidence descending,
    filtered to display/deep_review only (no_match discarded).
    """
    moment_a    = moments_map.get(user_a_id)
    other_users = [uid for uid in moments_map if uid != user_a_id]

    if not moment_a:
        print(f"[main] no moment provided for anchor user {user_a_id}")
        return []

    if not other_users:
        print(f"[main] no other users found in moments_map for book_id={book_id}")
        return []

    print(f"[main] found {len(other_users)} other readers of {book_id}")
    print(f"[main] running compatibility for {user_a_id} against all\n")

    results = []
    for user_b_id in other_users:
        moment_b = moments_map.get(user_b_id)
        if not moment_b:
            print(f"  {user_b_id} — skipped (no moment provided)")
            continue

        existing = _get_existing_compat_run(user_a_id, user_b_id, book_id)
        if existing:
            print(f"  {user_b_id}")
            print(f"    [cached] verdict={existing.get('verdict')}  "
                  f"confidence={existing.get('confidence')}")
            route = route_compatibility_result(existing)
            existing["route"] = route
            if route != "discard":
                results.append(existing)
            continue

        result = run_compatibility_agent(user_a_id, user_b_id, book_id,moment_a,moment_b)
        route  = route_compatibility_result(result)
        result["route"] = route

        print(f"  {user_b_id}")
        print(f"    verdict={result.get('verdict')}  "
              f"think={result.get('think_dimension')}  "
              f"feel={result.get('feel_dimension')}  "
              f"confidence={result.get('confidence')}  "
              f"route={route}")
        if route != "discard":
            print(f"    insight: {result.get('insight')}")
        print()

        if route != "discard":
            results.append(result)


    results.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)

    print(f"── Match pool for {user_a_id} ───────────────────────────────")
    print(f"   {len(results)} matches from {len(other_users)} candidates\n")
    for r in results:
        print(f"  [{r['route'].upper()}] {r['user_b']} "
              f"— {r['verdict']} "
              f"(think={r.get('think_dimension')} "
              f"feel={r.get('feel_dimension')} "
              f"conf={r.get('confidence')})")
        print(f"    {r.get('insight')}\n")

    return results


if __name__ == "__main__":
    USER_A  = "user_emma_chen_fd5e3def"
    BOOK_ID = "gutenberg_1342"

    # Caller is responsible for selecting and passing in one moment per user.
    moments_map = {
        "user_emma_chen_fd5e3def": {
            "user_id": "user_emma_chen_fd5e3def",
            "book_id": BOOK_ID,
            "text": "...",
            "timestamp": "2024-01-01T00:00:00"
        },
        # add other users here
    }

    matches = run_compatibility_for_all(USER_A, BOOK_ID, moments_map)

    print("\n── Full results (JSON) ──────────────────────────")
    print(json.dumps(matches, indent=2))
    
    
with open("data/processed/frankenstein_all_passages_final.json") as f:
    moments = json.load(f)

ANCHOR_USER = "Emma Chen"
PASSAGE_ID  = "passage_1"
BOOK_ID     = 1

# find emma's moment for this passage
anchor_moment = next(
    (m for m in moments
     if m["character_name"] == ANCHOR_USER and m["passage_id"] == PASSAGE_ID),
    None
)

if not anchor_moment:
    raise ValueError(f"No moment found for {ANCHOR_USER} on {PASSAGE_ID}")

# build moments_map: one moment per user for this passage
# if a user has multiple moments on the same passage, takes the first
moments_map = {ANCHOR_USER: anchor_moment}
for m in moments:
    uid = m["user_id"]
    if m["passage_id"] == PASSAGE_ID and uid not in moments_map and m['user_id']!=ANCHOR_USER:
        moments_map[uid] = m

results = run_compatibility_for_all(ANCHOR_USER, BOOK_ID, moments_map)