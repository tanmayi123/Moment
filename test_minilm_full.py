"""
test_minilm_full.py
===================
Complete test of all-MiniLM-L6-v2 resonance model
Includes: accuracy test + all moments + all pairs export

Run with: python3 test_minilm_full.py
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

sys.path.insert(0, 'data_pipeline/scripts')

from minilm_model import (
    get_embedder,
    create_pairs_from_moments,
    balance_pairs,
    evaluate_on_pairs,
    calculate_scores
)

print("=" * 60)
print("MOMENT - MiniLM Full Test + Export")
print("Model: all-MiniLM-L6-v2")
print("=" * 60)

# ─────────────────────────────────────────
# SETUP: Find data
# ─────────────────────────────────────────
moments_path = None
for path in [
    'data/processed/moments_processed.json',
    'data/moments_processed.json',
    'moments_rewritten.json'
]:
    if os.path.exists(path):
        moments_path = path
        break

if moments_path is None:
    print("ERROR: No moments file found")
    sys.exit(1)

print(f"\nUsing: {moments_path}")

# load raw moments
with open(moments_path, 'r') as f:
    moments = json.load(f)

df = pd.DataFrame(moments)
print(f"Loaded {len(df)} moments")

# find columns
text_col = next(
    (c for c in ['cleaned_interpretation', 'interpretation',
                 'text', 'content'] if c in df.columns),
    None
)
user_col = next(
    (c for c in ['user_id', 'user', 'reader_id'] if c in df.columns),
    None
)
book_col = next(
    (c for c in ['book_id', 'book', 'book_title'] if c in df.columns),
    None
)
passage_col = next(
    (c for c in ['passage_id', 'passage'] if c in df.columns),
    None
)

print(f"Text column:    {text_col}")
print(f"User column:    {user_col}")
print(f"Book column:    {book_col}")
print(f"Passage column: {passage_col}")

os.makedirs('data/reports', exist_ok=True)


# ─────────────────────────────────────────
# PART 1: Create pairs
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 1: Creating pairs")
print("=" * 60)

pairs_df = create_pairs_from_moments(moments_path)

if len(pairs_df) == 0:
    print("ERROR: No pairs created")
    sys.exit(1)

print(f"\nTotal pairs: {len(pairs_df)}")
print("Distribution:")
print(pairs_df['relationship'].value_counts())
print(f"\nSimilarity stats:")
print(f"  mean:  {pairs_df['similarity'].mean():.3f}")
print(f"  min:   {pairs_df['similarity'].min():.3f}")
print(f"  max:   {pairs_df['similarity'].max():.3f}")


# ─────────────────────────────────────────
# PART 2: Balance and split
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 2: Balancing and splitting")
print("=" * 60)

balanced_df = balance_pairs(pairs_df)

# fix label column for pandas 3.x
if 'label' not in balanced_df.columns:
    label_map = {"resonant": 0, "conversion": 1, "divergent": 2}
    balanced_df['label'] = balanced_df['relationship'].map(label_map)

print(f"Balanced: {len(balanced_df)} pairs")
print(balanced_df['relationship'].value_counts())

if len(balanced_df) < 9:
    print("ERROR: Not enough pairs to split")
    sys.exit(1)

unique_labels = balanced_df['label'].nunique()
if unique_labels < 3:
    print(f"WARNING: Only {unique_labels} classes")
    train_df, test_df = train_test_split(
        balanced_df,
        test_size=0.2,
        random_state=42
    )
else:
    train_df, test_df = train_test_split(
        balanced_df,
        test_size=0.2,
        stratify=balanced_df['label'],
        random_state=42
    )

print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")


# ─────────────────────────────────────────
# PART 3: Load model
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 3: Loading model")
print("=" * 60)

model = get_embedder()
print("Model loaded")


# ─────────────────────────────────────────
# PART 4: Accuracy test (circular baseline)
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 4: Accuracy test (circular baseline)")
print("=" * 60)

results = evaluate_on_pairs(model, test_df)

print(f"\nCircular Accuracy: {results['accuracy']:.3f}")
print("(Note: labels from same model = circular)")
print("\nClassification Report:")
report = results['report']
for label in ['resonant', 'conversion', 'divergent']:
    if label in report:
        print(f"  {label}:")
        print(f"    precision: {report[label]['precision']:.3f}")
        print(f"    recall:    {report[label]['recall']:.3f}")
        print(f"    f1-score:  {report[label]['f1-score']:.3f}")


# ─────────────────────────────────────────
# PART 5: Sample predictions
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 5: Sample predictions from test set")
print("=" * 60)

correct_count = 0
for i in range(min(5, len(test_df))):
    row     = test_df.iloc[i]
    scores  = calculate_scores(model, row['interp_a'], row['interp_b'])
    correct = row['relationship'] == scores['predicted_relationship']
    if correct:
        correct_count += 1

    print(f"\nPair {i+1}:")
    print(f"  User A:      {row['user_a']}")
    print(f"  User B:      {row['user_b']}")
    print(f"  Similarity:  {scores['similarity']:.3f}")
    print(f"  True:        {row['relationship']}")
    print(f"  Predicted:   {scores['predicted_relationship']}")
    print(f"  Resonance:   {scores['resonance_score']:.3f}")
    print(f"  Conversion:  {scores['conversion_score']:.3f}")
    print(f"  Divergence:  {scores['divergence_score']:.3f}")
    print(f"  Correct:     {'YES ✓' if correct else 'NO ✗'}")

print(f"\nSample: {correct_count}/5 correct")


# ─────────────────────────────────────────
# PART 6: Export all pairs with scores
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 6: Adding scores to all pairs")
print("=" * 60)

resonance_scores  = []
conversion_scores = []
divergence_scores = []
confidences       = []

print("Scoring all pairs...")
for _, row in pairs_df.iterrows():
    scores = calculate_scores(
        model,
        row['interp_a'],
        row['interp_b']
    )
    resonance_scores.append(scores['resonance_score'])
    conversion_scores.append(scores['conversion_score'])
    divergence_scores.append(scores['divergence_score'])
    confidences.append(scores['confidence'])

pairs_df['resonance_score']  = resonance_scores
pairs_df['conversion_score'] = conversion_scores
pairs_df['divergence_score'] = divergence_scores
pairs_df['confidence']       = confidences

print(f"Scored {len(pairs_df)} pairs")

# save pairs
pairs_df.to_json(
    'data/reports/all_pairs_scored.json',
    orient='records',
    indent=2
)
pairs_df.to_csv(
    'data/reports/all_pairs_scored.csv',
    index=False
)
print("Saved all_pairs_scored.json and .csv")


# ─────────────────────────────────────────
# PART 7: Top pairs
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 7: Top pairs by relationship type")
print("=" * 60)

print("\nTOP 5 MOST RESONANT:")
top_resonant = pairs_df.nlargest(5, 'resonance_score')
for _, row in top_resonant.iterrows():
    print(f"  {row['user_a']} + {row['user_b']}")
    print(f"  resonance: {row['resonance_score']:.3f}  similarity: {row['similarity']:.3f}")

print("\nTOP 5 MOST DIVERGENT:")
top_divergent = pairs_df.nlargest(5, 'divergence_score')
for _, row in top_divergent.iterrows():
    print(f"  {row['user_a']} + {row['user_b']}")
    print(f"  divergence: {row['divergence_score']:.3f}  similarity: {row['similarity']:.3f}")

print("\nTOP 5 BEST CONVERSION:")
top_conversion = pairs_df.nlargest(5, 'conversion_score')
for _, row in top_conversion.iterrows():
    print(f"  {row['user_a']} + {row['user_b']}")
    print(f"  conversion: {row['conversion_score']:.3f}  similarity: {row['similarity']:.3f}")


# ─────────────────────────────────────────
# PART 8: Save full report
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 8: Saving full report")
print("=" * 60)

full_report = {
    "model":             "all-MiniLM-L6-v2",
    "note":              "Circular accuracy - labels from same model. Real accuracy needs agent labels.",
    "circular_accuracy": float(results['accuracy']),
    "total_moments":     len(df),
    "total_pairs":       len(pairs_df),
    "balanced_pairs":    len(balanced_df),
    "train_size":        len(train_df),
    "test_size":         len(test_df),
    "distribution": {
        rel: int(count)
        for rel, count in pairs_df['relationship'].value_counts().items()
    },
    "similarity_stats": {
        "mean": float(pairs_df['similarity'].mean()),
        "min":  float(pairs_df['similarity'].min()),
        "max":  float(pairs_df['similarity'].max())
    },
    "top_resonant_pairs": [
        {
            "user_a":     row['user_a'],
            "user_b":     row['user_b'],
            "resonance":  float(row['resonance_score']),
            "similarity": float(row['similarity'])
        }
        for _, row in top_resonant.iterrows()
    ],
    "top_divergent_pairs": [
        {
            "user_a":     row['user_a'],
            "user_b":     row['user_b'],
            "divergence": float(row['divergence_score']),
            "similarity": float(row['similarity'])
        }
        for _, row in top_divergent.iterrows()
    ],
    "top_conversion_pairs": [
        {
            "user_a":     row['user_a'],
            "user_b":     row['user_b'],
            "conversion": float(row['conversion_score']),
            "similarity": float(row['similarity'])
        }
        for _, row in top_conversion.iterrows()
    ],
    "classification_report": results['report']
}

with open('data/reports/minilm_full_report.json', 'w') as f:
    json.dump(full_report, f, indent=2)

print("Saved data/reports/minilm_full_report.json")


# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL DONE")
print("=" * 60)
print(f"\nFiles saved:")
print(f"  data/reports/all_pairs_scored.json")
print(f"  data/reports/all_pairs_scored.csv   ← open in Excel")
print(f"  data/reports/minilm_full_report.json")
print(f"\nKey numbers for Tuesday:")
print(f"  Total moments:     {len(df)}")
print(f"  Total pairs:       {len(pairs_df)}")
print(f"  Circular accuracy: {results['accuracy']:.3f}")
print(f"  (Real accuracy needs agent labels - Phase 2)")
