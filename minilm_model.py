"""
minilm_model.py
===============
Resonance model using all-MiniLM-L6-v2
Fast local testing version
No structured attributes needed

Author: Greta | MOMENT Group 23
"""

import json
import os
import logging
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

LABELS = {
    0: "resonant",
    1: "conversion",
    2: "divergent"
}

# will be auto adjusted based on data
RESONANT_THRESHOLD   = 0.5
CONVERSION_THRESHOLD = 0.2


def get_embedder():
    """Load all-MiniLM-L6-v2 model"""
    from sentence_transformers import SentenceTransformer
    logger.info("Loading all-MiniLM-L6-v2 (90MB, fast)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


def embed_interpretation(model, text):
    """Embed a single interpretation"""
    embedding = model.encode(str(text)[:512])
    return embedding


def calculate_scores(model, interpretation_a, interpretation_b):
    """
    Main function called by pipeline for any new pair

    Input:  two raw interpretation texts
    Output: resonance / conversion / divergence scores
    """
    emb_a = embed_interpretation(model, interpretation_a)
    emb_b = embed_interpretation(model, interpretation_b)

    similarity = cosine_similarity(
        emb_a.reshape(1, -1),
        emb_b.reshape(1, -1)
    )[0][0]

    if similarity >= RESONANT_THRESHOLD:
        predicted = "resonant"
        label     = 0
        scores    = {
            "resonance_score":  float(similarity),
            "conversion_score": float((1 - similarity) * 0.6),
            "divergence_score": float((1 - similarity) * 0.4)
        }
    elif similarity >= CONVERSION_THRESHOLD:
        predicted = "conversion"
        label     = 1
        scores    = {
            "resonance_score":  float(similarity * 0.4),
            "conversion_score": float(similarity),
            "divergence_score": float((1 - similarity) * 0.5)
        }
    else:
        predicted = "divergent"
        label     = 2
        scores    = {
            "resonance_score":  float(similarity * 0.2),
            "conversion_score": float(similarity * 0.4),
            "divergence_score": float(1 - similarity)
        }

    return {
        **scores,
        "similarity":             float(similarity),
        "predicted_relationship": predicted,
        "label":                  label,
        "confidence":             float(max(scores.values()))
    }


def auto_adjust_thresholds(embeddings, sample_size=100):
    """
    Auto adjust thresholds based on actual data distribution
    Ensures roughly equal distribution across three classes
    """
    global RESONANT_THRESHOLD, CONVERSION_THRESHOLD

    sample_sims = []
    limit = min(sample_size, len(embeddings))

    for i in range(limit):
        for j in range(i+1, limit):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[j].reshape(1, -1)
            )[0][0]
            sample_sims.append(sim)

    if sample_sims:
        sorted_sims          = sorted(sample_sims)
        n                    = len(sorted_sims)
        RESONANT_THRESHOLD   = sorted_sims[int(n * 0.67)]
        CONVERSION_THRESHOLD = sorted_sims[int(n * 0.33)]

        logger.info(f"Auto-adjusted thresholds:")
        logger.info(f"  resonant:   >= {RESONANT_THRESHOLD:.3f}")
        logger.info(f"  conversion: >= {CONVERSION_THRESHOLD:.3f}")
        logger.info(f"  divergent:  <  {CONVERSION_THRESHOLD:.3f}")
        logger.info(f"  similarity mean: {np.mean(sample_sims):.3f}")

    return RESONANT_THRESHOLD, CONVERSION_THRESHOLD


def create_pairs_from_moments(moments_path):
    """
    Load moments and create labeled pairs
    Uses all-MiniLM-L6-v2 similarity to label pairs
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading moments from {moments_path}")

    with open(moments_path, 'r') as f:
        moments = json.load(f)

    df = pd.DataFrame(moments)
    logger.info(f"Loaded {len(df)} moments")
    logger.info(f"Columns: {df.columns.tolist()}")

    # find text column
    text_col = None
    for col in ['cleaned_interpretation', 'interpretation', 'text', 'content']:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(
            f"No text column found. Available: {df.columns.tolist()}"
        )

    logger.info(f"Using text column: {text_col}")

    # find other columns
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

    # load model
    logger.info("Loading all-MiniLM-L6-v2...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # embed all texts
    logger.info("Embedding all interpretations...")
    texts      = df[text_col].fillna('').tolist()
    embeddings = embedder.encode(
        [t[:512] for t in texts],
        show_progress_bar=True,
        convert_to_numpy=True
    )
    logger.info(f"Embedded {len(embeddings)} interpretations")

    # auto adjust thresholds
    auto_adjust_thresholds(embeddings)

    # group by book and passage
    if book_col and passage_col:
        groups = df.groupby([book_col, passage_col])
    elif book_col:
        groups = df.groupby(book_col)
    else:
        groups = [('all', df)]

    # create pairs
    pairs = []
    logger.info("Creating pairs...")

    for group_key, group_df in groups:
        indices = group_df.index.tolist()

        if len(indices) > 30:
            indices = indices[:30]

        for idx_a, idx_b in combinations(indices, 2):
            row_a = df.loc[idx_a]
            row_b = df.loc[idx_b]

            if user_col and row_a[user_col] == row_b[user_col]:
                continue

            sim = cosine_similarity(
                embeddings[idx_a].reshape(1, -1),
                embeddings[idx_b].reshape(1, -1)
            )[0][0]

            label, relationship = (
                (0, "resonant")   if sim >= RESONANT_THRESHOLD   else
                (1, "conversion") if sim >= CONVERSION_THRESHOLD else
                (2, "divergent")
            )

            pairs.append({
                'user_a':       str(row_a.get(user_col, idx_a)),
                'user_b':       str(row_b.get(user_col, idx_b)),
                'book_id':      str(row_a.get(book_col, 'unknown')),
                'passage_id':   str(row_a.get(passage_col, 'unknown')),
                'interp_a':     str(row_a[text_col])[:512],
                'interp_b':     str(row_b[text_col])[:512],
                'similarity':   float(sim),
                'label':        label,
                'relationship': relationship
            })

    pairs_df = pd.DataFrame(pairs)
    logger.info(f"Created {len(pairs_df)} pairs")

    if len(pairs_df) > 0:
        dist = pairs_df['relationship'].value_counts().to_dict()
        logger.info(f"Distribution: {dist}")

    return pairs_df


def balance_pairs(pairs_df):
    """Balance classes"""
    counts    = pairs_df['label'].value_counts()
    min_count = counts.min()
    logger.info(f"Balancing to {min_count} per class")

    balanced = pairs_df.groupby('label').apply(
        lambda x: x.sample(min(len(x), min_count), random_state=42)
    ).reset_index(drop=True)

    # fix for pandas 3.x dropping label column after groupby
    if 'label' not in balanced.columns:
        label_map = {"resonant": 0, "conversion": 1, "divergent": 2}
        balanced['label'] = balanced['relationship'].map(label_map)

    logger.info(f"Balanced: {len(balanced)} pairs")
    return balanced


def evaluate_on_pairs(model, pairs_df):
    """Evaluate model on labeled pairs"""
    logger.info(f"Evaluating on {len(pairs_df)} pairs")

    predictions = []
    true_labels = pairs_df['label'].tolist()

    for _, row in pairs_df.iterrows():
        result = calculate_scores(model, row['interp_a'], row['interp_b'])
        predictions.append(result['label'])

    accuracy = accuracy_score(true_labels, predictions)
    report   = classification_report(
        true_labels,
        predictions,
        target_names=["resonant", "conversion", "divergent"],
        output_dict=True
    )

    return {
        "accuracy":    float(accuracy),
        "report":      report,
        "predictions": predictions
    }
