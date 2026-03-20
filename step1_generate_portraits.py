import json
import os
import time
from dotenv import load_dotenv
from google import genai

# ── Load API key ──────────────────────────────────────────────
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Load data ─────────────────────────────────────────────────
with open("data/all_interpretations_450_FINAL_NO_BIAS.json", "r") as f:
    interpretations = json.load(f)

with open("data/users_processed.json", "r") as f:
    users = json.load(f)

# Build a lookup: character_name → user_id
user_lookup = {u["character_name"]: u["user_id"] for u in users}

# ── Group interpretations by (character_name, book) ──────────
grouped = {}
for interp in interpretations:
    name = interp["character_name"]
    book = interp["book"]
    key = (name, book)
    if key not in grouped:
        grouped[key] = []
    grouped[key].append(interp)

print(f"Total (user, book) combinations: {len(grouped)}")

# ── Portrait generation prompt ────────────────────────────────
def build_prompt(character_name, book_title, interp_list):
    interpretations_text = ""
    for i, interp in enumerate(interp_list, 1):
        interpretations_text += f"\nPassage {interp['passage_id']}:\n{interp['interpretation']}\n"

    return f"""You are analysing how a reader engages with a book based on their written interpretations of passages.

Reader: {character_name}
Book: {book_title}

Here are all their interpretations:
{interpretations_text}

Based ONLY on these interpretations, produce a reader portrait with exactly these 6 fields.
Be specific and grounded — every claim must be evidenced by the actual text above.

Return ONLY valid JSON, no explanation, no markdown, no backticks:

{{
  "how_they_read": "one sentence: what level of text do they operate on — words/phrases, scene/atmosphere, character psychology, or theme/ideology",
  "interpretive_lens": "one sentence: what conceptual vocabulary do they use — psychological, philosophical, literary-critical, clinical, craft-focused, or personal-experiential",
  "central_preoccupation": "one sentence: the recurring question they ask of this book AND their position on it",
  "what_moves_them": "one sentence: the specific emotional content that produces their deepest responses",
  "emotional_mode": "one of: prosecutorial / empathetic-victim / empathetic-perpetrator / craft-observer — plus one sentence of evidence",
  "self_referential": "true or false — does this reader map the text to personal experience? Plus one sentence of evidence",
  "reflection_density": "low / medium / high — based on how many logical moves they make and whether they push past surface meaning"
}}"""

# ── Generate portraits ────────────────────────────────────────
portraits = []
total = len(grouped)

for idx, ((character_name, book_title), interp_list) in enumerate(grouped.items(), 1):
    print(f"[{idx}/{total}] Generating portrait: {character_name} — {book_title}")

    prompt = build_prompt(character_name, book_title, interp_list)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        raw = response.text.strip()

        # Clean up if Gemini wraps in markdown despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        portrait_fields = json.loads(raw)

        portrait = {
            "user_id": user_lookup.get(character_name, "unknown"),
            "character_name": character_name,
            "book_title": book_title,
            "portrait": portrait_fields,
            "moments_count": len(interp_list)
        }
        portraits.append(portrait)

    except Exception as e:
        print(f"  ERROR for {character_name} / {book_title}: {e}")
        portraits.append({
            "user_id": user_lookup.get(character_name, "unknown"),
            "character_name": character_name,
            "book_title": book_title,
            "portrait": None,
            "error": str(e),
            "moments_count": len(interp_list)
        })

    time.sleep(0.5)

# ── Save output ───────────────────────────────────────────────
os.makedirs("output", exist_ok=True)
with open("output/portraits.json", "w") as f:
    json.dump(portraits, f, indent=2)

print(f"\n✅ Done! {len(portraits)} portraits saved to output/portraits.json")

success = sum(1 for p in portraits if p["portrait"] is not None)
failed = len(portraits) - success
print(f"   Successful: {success}")
print(f"   Failed:     {failed}")

