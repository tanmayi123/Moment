import json
import os
import time
from dotenv import load_dotenv
from google import genai
from collections import Counter

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── CONFIG — change this to process one book at a time ────────
BOOK_TO_PROCESS = "Pride and Prejudice"  # change to "Pride and Prejudice" or "The Great Gatsby" next
CHECKPOINT_FILE = f"output/verdicts_{BOOK_TO_PROCESS.replace(' ', '_')}.json"

# ── Load candidate pairs for this book only ───────────────────
with open("output/candidate_pairs.json", "r") as f:
    all_pairs = json.load(f)

pairs = [p for p in all_pairs if p["book_title"] == BOOK_TO_PROCESS]
print(f"Book: {BOOK_TO_PROCESS}")
print(f"Total pairs for this book: {len(pairs)}")

# ── Load checkpoint if exists (resume from where we stopped) ──
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        verdicts = json.load(f)
    already_done = {(v["user_a_id"], v["user_b_id"]) for v in verdicts}
    pairs = [p for p in pairs if (p["user_a_id"], p["user_b_id"]) not in already_done]
    print(f"Resuming — {len(already_done)} already done, {len(pairs)} remaining")
else:
    verdicts = []
    print("Starting fresh")

# ── Verdict prompt ────────────────────────────────────────────
def build_verdict_prompt(pair):
    a = pair["portrait_a"]
    b = pair["portrait_b"]
    return f"""You are comparing two readers of the same book to determine their intellectual and emotional compatibility.

Book: {pair["book_title"]}
Reader A: {pair["user_a_name"]}
Reader B: {pair["user_b_name"]}

READER A PORTRAIT:
- How they read: {a.get("how_they_read", "")}
- Interpretive lens: {a.get("interpretive_lens", "")}
- Central preoccupation: {a.get("central_preoccupation", "")}
- What moves them: {a.get("what_moves_them", "")}
- Emotional mode: {a.get("emotional_mode", "")}
- Self referential: {a.get("self_referential", "")}
- Reflection density: {a.get("reflection_density", "")}

READER B PORTRAIT:
- How they read: {b.get("how_they_read", "")}
- Interpretive lens: {b.get("interpretive_lens", "")}
- Central preoccupation: {b.get("central_preoccupation", "")}
- What moves them: {b.get("what_moves_them", "")}
- Emotional mode: {b.get("emotional_mode", "")}
- Self referential: {b.get("self_referential", "")}
- Reflection density: {b.get("reflection_density", "")}

CLASSIFICATION RULES:

THINK dimension (how_they_read + interpretive_lens + central_preoccupation):
- Resonance: same level of text, compatible frameworks, aligned position
- Contradiction: same level and framework but opposing positions on the same question
- Divergence: different levels OR incommensurable frameworks

FEEL dimension (what_moves_them + emotional_mode + self_referential):
- Resonance: similar emotional triggers, compatible emotional modes
- Contradiction: prosecutorial paired with empathetic-victim
- Divergence: different triggers OR one self-referential and one not

Return ONLY valid JSON, no markdown, no backticks:

{{
  "think_label": "Resonance or Contradiction or Divergence",
  "think_reasoning": "one sentence",
  "feel_label": "Resonance or Contradiction or Divergence",
  "feel_reasoning": "one sentence",
  "confidence": 0.0,
  "overall_summary": "one sentence"
}}"""

# ── Process ───────────────────────────────────────────────────
total = len(pairs)
failed = 0

for idx, pair in enumerate(pairs, 1):
    prompt = build_verdict_prompt(pair)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        verdict_fields = json.loads(raw)

        verdicts.append({
            "book_title": pair["book_title"],
            "user_a_id": pair["user_a_id"],
            "user_a_name": pair["user_a_name"],
            "user_b_id": pair["user_b_id"],
            "user_b_name": pair["user_b_name"],
            "think_label": verdict_fields.get("think_label"),
            "think_reasoning": verdict_fields.get("think_reasoning"),
            "feel_label": verdict_fields.get("feel_label"),
            "feel_reasoning": verdict_fields.get("feel_reasoning"),
            "confidence": verdict_fields.get("confidence"),
            "overall_summary": verdict_fields.get("overall_summary")
        })

    except Exception as e:
        failed += 1
        verdicts.append({
            "book_title": pair["book_title"],
            "user_a_id": pair["user_a_id"],
            "user_a_name": pair["user_a_name"],
            "user_b_id": pair["user_b_id"],
            "user_b_name": pair["user_b_name"],
            "think_label": None,
            "feel_label": None,
            "confidence": None,
            "overall_summary": None,
            "error": str(e)
        })

    # Save checkpoint every 50 pairs
    if idx % 50 == 0:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(verdicts, f, indent=2)
        print(f"[{idx}/{total}] Checkpoint saved — {failed} failed so far")

    time.sleep(0.3)

# ── Final save ────────────────────────────────────────────────
with open(CHECKPOINT_FILE, "w") as f:
    json.dump(verdicts, f, indent=2)

think_counts = Counter(v["think_label"] for v in verdicts if v["think_label"])
feel_counts  = Counter(v["feel_label"]  for v in verdicts if v["feel_label"])
success = sum(1 for v in verdicts if v["think_label"] is not None)

print(f"\n✅ Done! {len(verdicts)} verdicts saved to {CHECKPOINT_FILE}")
print(f"   Successful: {success} | Failed: {failed}")
print(f"\nThink distribution: {dict(think_counts)}")
print(f"Feel distribution:  {dict(feel_counts)}")