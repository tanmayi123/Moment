import json
from collections import Counter

# ── Load Frankenstein verdicts ────────────────────────────────
with open("output/verdicts_Frankenstein.json", "r") as f:
    verdicts = json.load(f)

# ── Load portraits for additional features ────────────────────
with open("output/portraits.json", "r") as f:
    portraits = json.load(f)

# Build portrait lookup: (user_id, book_title) → portrait fields
portrait_lookup = {}
for p in portraits:
    key = (p["user_id"], p["book_title"])
    portrait_lookup[key] = p["portrait"]

# ── Build training labels ─────────────────────────────────────
training_data = []

for v in verdicts:
    # Skip failed verdicts
    if v["think_label"] is None or v["feel_label"] is None:
        continue

    # Get portrait features for both users
    portrait_a = portrait_lookup.get((v["user_a_id"], v["book_title"]), {})
    portrait_b = portrait_lookup.get((v["user_b_id"], v["book_title"]), {})

    training_data.append({
        # ── Identifiers ──
        "pair_id": f"{v['user_a_id']}_{v['user_b_id']}_{v['book_title'].replace(' ', '_')}",
        "book_title": v["book_title"],
        "user_a_id": v["user_a_id"],
        "user_a_name": v["user_a_name"],
        "user_b_id": v["user_b_id"],
        "user_b_name": v["user_b_name"],

        # ── Labels (what the model learns to predict) ──
        "think_label": v["think_label"],
        "feel_label": v["feel_label"],
        "confidence": v["confidence"],

        # ── Agent reasoning (for interpretability) ──
        "think_reasoning": v["think_reasoning"],
        "feel_reasoning": v["feel_reasoning"],
        "overall_summary": v["overall_summary"],

        # ── Portrait features user A ──
        "a_how_they_read": portrait_a.get("how_they_read", ""),
        "a_interpretive_lens": portrait_a.get("interpretive_lens", ""),
        "a_central_preoccupation": portrait_a.get("central_preoccupation", ""),
        "a_what_moves_them": portrait_a.get("what_moves_them", ""),
        "a_emotional_mode": portrait_a.get("emotional_mode", ""),
        "a_self_referential": portrait_a.get("self_referential", ""),
        "a_reflection_density": portrait_a.get("reflection_density", ""),

        # ── Portrait features user B ──
        "b_how_they_read": portrait_b.get("how_they_read", ""),
        "b_interpretive_lens": portrait_b.get("interpretive_lens", ""),
        "b_central_preoccupation": portrait_b.get("central_preoccupation", ""),
        "b_what_moves_them": portrait_b.get("what_moves_them", ""),
        "b_emotional_mode": portrait_b.get("emotional_mode", ""),
        "b_self_referential": portrait_b.get("self_referential", ""),
        "b_reflection_density": portrait_b.get("reflection_density", ""),
    })

# ── Save as JSON ──────────────────────────────────────────────
with open("output/training_labels.json", "w") as f:
    json.dump(training_data, f, indent=2)

# ── Save as CSV too (useful for model training) ───────────────
import csv
if training_data:
    with open("output/training_labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=training_data[0].keys())
        writer.writeheader()
        writer.writerows(training_data)

# ── Summary ───────────────────────────────────────────────────
print(f"✅ Training labels created!")
print(f"   Total pairs: {len(training_data)}")
print(f"   Saved to: output/training_labels.json")
print(f"             output/training_labels.csv")

think_counts = Counter(d["think_label"] for d in training_data)
feel_counts  = Counter(d["feel_label"]  for d in training_data)
conf_values  = [d["confidence"] for d in training_data if d["confidence"]]
avg_conf     = sum(conf_values) / len(conf_values) if conf_values else 0

print(f"\nThink distribution: {dict(think_counts)}")
print(f"Feel distribution:  {dict(feel_counts)}")
print(f"Average confidence: {avg_conf:.2f}")

print(f"\nFinal folder structure:")
print(f"  output/portraits.json          — 150 reader portraits")
print(f"  output/candidate_pairs.json    — 3675 candidate pairs")
print(f"  output/verdicts_Frankenstein.json — 1225 R/C/D verdicts")
print(f"  output/training_labels.json    — {len(training_data)} labeled training examples")
print(f"  output/training_labels.csv     — same, CSV format")