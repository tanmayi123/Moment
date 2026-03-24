"""
model_validation.py
===================
Standalone validation for the MOMENT compatibility model.
Uses all_profile_scores_fixed.json (overall + by_book scores)
and ground_truth.json for evaluation.

Input format (all_profile_scores_fixed.json):
  {
    "pairs": [
      {
        "user_a": "Lily Chen",
        "user_b": "Rebecca Stone",
        "overall": {
          "think": {"R": 70, "C": 4, "D": 26},
          "feel":  {"R": 55, "C": 19, "D": 26},
          "dominant_think": "resonate",
          "dominant_feel":  "resonate",
          "confidence": 0.54
        },
        "by_book": {
          "Frankenstein": {
            "think": {"R": 27, "C": 16, "D": 57},
            "feel":  {"R": 26, "C": 17, "D": 57},
            "dominant_think": "diverge",
            "dominant_feel":  "diverge",
            "confidence": 0.44
          },
          ...
        }
      }
    ]
  }

Ground truth format (ground_truth.json):
  [
    {
      "user_a":         "Emma Chen",
      "user_b":         "Marcus Williams",
      "book_id":        "Frankenstein",
      "dominant_think": "resonate",
      "dominant_feel":  "contradict"
    },
    ...
  ]

Usage:
    python model_validation.py
    python model_validation.py --demo
    python model_validation.py --predictions data/reports/all_profile_scores_fixed.json --ground-truth data/ground_truth.json
"""

import argparse
import json
import logging
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
PREDICTIONS_FILE  = "data/reports/all_profile_scores_fixed.json"
GROUND_TRUTH_FILE = "data/ground_truth.json"
OUTPUT_DIR        = "data/validation_output"

LABEL_MAP   = {"resonate": 0, "contradict": 1, "diverge": 2}
CLASS_NAMES = ["resonate", "contradict", "diverge"]


# ============================================================
# HELPERS
# ============================================================
def to_label(val):
    """Normalise a dominant_think / dominant_feel string to int label."""
    return LABEL_MAP.get(str(val).lower().strip(), -1)


# ============================================================
# 1. LOAD PREDICTIONS
# ============================================================
def load_predictions(predictions_path):
    """
    Load all_profile_scores_fixed.json.
    Flattens overall + by_book into one row per (user_a, user_b, book).
    Returns a DataFrame with one row per pair-book combination.
    """
    logger.info("=" * 60)
    logger.info("LOADING PREDICTIONS")
    logger.info("=" * 60)

    with open(predictions_path) as f:
        data = json.load(f)

    # unwrap {generated_at, total_pairs, pairs: [...]} structure
    if isinstance(data, dict):
        data = data.get("pairs", [])

    rows = []
    for obj in data:
        user_a = str(obj.get("user_a", ""))
        user_b = str(obj.get("user_b", ""))

        def make_row(scores, book_name):
            think = scores.get("think", {})
            feel  = scores.get("feel",  {})
            return {
                "user_a":      user_a,
                "user_b":      user_b,
                "book_id":     book_name,

                # continuous scores (0–100) → convert to 0–1 for AUC
                "think_R":     think.get("R", 0) / 100,
                "think_C":     think.get("C", 0) / 100,
                "think_D":     think.get("D", 0) / 100,
                "feel_R":      feel.get("R",  0) / 100,
                "feel_C":      feel.get("C",  0) / 100,
                "feel_D":      feel.get("D",  0) / 100,

                # predicted labels
                "think_pred":  to_label(scores.get("dominant_think", "")),
                "feel_pred":   to_label(scores.get("dominant_feel",  "")),

                "confidence":  scores.get("confidence", 0.0),
            }

        # overall row
        if "overall" in obj:
            rows.append(make_row(obj["overall"], "overall"))

        # by_book rows
        for book_name, book_scores in obj.get("by_book", {}).items():
            rows.append(make_row(book_scores, book_name))

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} prediction rows from {predictions_path}")
    logger.info(f"Books: {', '.join(df['book_id'].unique())}")
    logger.info(f"Think pred distribution:\n{df['think_pred'].value_counts().to_string()}")
    logger.info(f"Feel  pred distribution:\n{df['feel_pred'].value_counts().to_string()}")
    return df


# ============================================================
# 2. LOAD GROUND TRUTH
# ============================================================
def load_ground_truth(ground_truth_path):
    """
    Load ground_truth.json.
    Expected fields: user_a, user_b, book_id, dominant_think, dominant_feel.
    Returns a DataFrame with true labels.
    """
    logger.info("=" * 60)
    logger.info("LOADING GROUND TRUTH")
    logger.info("=" * 60)

    with open(ground_truth_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} ground truth records from {ground_truth_path}")

    df["think_true"] = df["dominant_think"].apply(to_label)
    df["feel_true"]  = df["dominant_feel"].apply(to_label)

    # keep only needed columns
    df = df[["user_a", "user_b", "book_id", "think_true", "feel_true"]]
    return df


# ============================================================
# 3. MERGE PREDICTIONS + GROUND TRUTH
# ============================================================
def merge_predictions_and_truth(pred_df, truth_df):
    """
    Merge on (user_a, user_b, book_id).
    Returns merged DataFrame with both pred and true labels.
    """
    logger.info("=" * 60)
    logger.info("MERGING PREDICTIONS + GROUND TRUTH")
    logger.info("=" * 60)

    merged = pred_df.merge(
        truth_df,
        on=["user_a", "user_b", "book_id"],
        how="inner"
    )
    logger.info(f"Matched {len(merged)} pairs after merge")

    if len(merged) == 0:
        raise ValueError(
            "No pairs matched after merge — check that user_a, "
            "user_b, and book_id match between both files."
        )

    # drop rows with invalid labels
    merged = merged[
        (merged["think_pred"] != -1) &
        (merged["feel_pred"]  != -1) &
        (merged["think_true"] != -1) &
        (merged["feel_true"]  != -1)
    ]
    logger.info(f"Valid rows after filtering invalid labels: {len(merged)}")
    return merged


# ============================================================
# 4. HOLD-OUT SPLIT
# ============================================================
def make_holdout_split(df, test_size=0.2, random_state=42):
    """
    Stratified 80/20 split on think_true label.
    Returns train_df, test_df.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["think_true"],
        random_state=random_state
    )
    logger.info(f"Train: {len(train_df)} | Hold-out test: {len(test_df)}")
    logger.info(f"Test think_true distribution:\n"
                f"{test_df['think_true'].value_counts().to_string()}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ============================================================
# 5. COMPUTE METRICS
# ============================================================
def compute_metrics(y_true, y_pred, y_proba, axis_name):
    """
    Compute accuracy, precision, recall, F1, AUC for one axis.
    y_proba: array of shape (n, 3) with R/C/D probabilities.
    """
    logger.info(f"\n{'=' * 50}")
    logger.info(f"  {axis_name.upper()} AXIS — Performance Metrics")
    logger.info(f"{'=' * 50}")

    acc = accuracy_score(y_true, y_pred)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )
    _, _, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    # AUC — one vs rest
    try:
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        auc   = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    logger.info(f"\nAccuracy  : {acc:.4f}")
    logger.info(f"Macro F1  : {f1_mac:.4f}")
    logger.info(f"Macro AUC : {auc:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)}")

    return {
        "axis":      axis_name,
        "accuracy":  float(acc),
        "macro_f1":  float(f1_mac),
        "macro_auc": float(auc),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(p[i]),
                "recall":    float(r[i]),
                "f1":        float(f1[i])
            }
            for i in range(3)
        }
    }


# ============================================================
# 6. CONFIDENCE ANALYSIS
# ============================================================
def analyse_confidence(test_df, output_dir):
    """
    Check whether confidence score correlates with accuracy.
    High confidence pairs should be more accurate than low confidence ones.
    """
    logger.info("=" * 60)
    logger.info("CONFIDENCE ANALYSIS")
    logger.info("=" * 60)

    test_df = test_df.copy()
    test_df["conf_bin"] = pd.cut(
        test_df["confidence"],
        bins=[0, 0.5, 0.75, 1.0],
        labels=["low (0-0.5)", "medium (0.5-0.75)", "high (0.75-1.0)"]
    )

    results = []
    for bin_label, group in test_df.groupby("conf_bin", observed=True):
        if len(group) == 0:
            continue
        think_acc = accuracy_score(group["think_true"], group["think_pred"])
        feel_acc  = accuracy_score(group["feel_true"],  group["feel_pred"])
        results.append({
            "confidence_bin": bin_label,
            "count":          len(group),
            "think_accuracy": round(think_acc, 3),
            "feel_accuracy":  round(feel_acc,  3)
        })
        logger.info(f"  {bin_label}: n={len(group)} | "
                    f"THINK acc={think_acc:.3f} | FEEL acc={feel_acc:.3f}")

    return results


# ============================================================
# 8. SAVE RESULTS
# ============================================================
def save_results(metrics_think, metrics_feel, conf_analysis, output_dir):
    out = {
        "think":      metrics_think,
        "feel":       metrics_feel,
        "confidence": conf_analysis
    }
    path = os.path.join(output_dir, "validation_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved → {path}")


# ============================================================
# 9. DEMO MODE
# ============================================================
def generate_demo_data(n=200):
    """
    Generate fake model output + ground truth for testing.
    Remove this once you have real data.
    """
    logger.info("Running in DEMO MODE — using fake data")
    np.random.seed(42)

    records = []
    labels  = ["resonate", "contradict", "diverge"]

    for i in range(n):
        think_true = np.random.choice(labels)
        feel_true  = np.random.choice(labels)

        think_pred = think_true if np.random.rand() > 0.25 else np.random.choice(labels)
        feel_pred  = feel_true  if np.random.rand() > 0.25 else np.random.choice(labels)

        def make_scores(pred):
            base = {"resonate": 0, "contradict": 1, "diverge": 2}[pred]
            s    = np.random.dirichlet([5 if i == base else 1 for i in range(3)])
            return {"R": int(s[0]*100), "C": int(s[1]*100), "D": 100 - int(s[0]*100) - int(s[1]*100)}

        records.append({
            "user_a":        f"user_{i % 20}",
            "user_b":        f"user_{(i+1) % 20}",
            "book_id":       np.random.choice(["Frankenstein", "Pride and Prejudice", "The Great Gatsby"]),
            "think_R":       make_scores(think_pred)["R"] / 100,
            "think_C":       make_scores(think_pred)["C"] / 100,
            "think_D":       make_scores(think_pred)["D"] / 100,
            "feel_R":        make_scores(feel_pred)["R"]  / 100,
            "feel_C":        make_scores(feel_pred)["C"]  / 100,
            "feel_D":        make_scores(feel_pred)["D"]  / 100,
            "think_pred":    to_label(think_pred),
            "feel_pred":     to_label(feel_pred),
            "think_true":    to_label(think_true),
            "feel_true":     to_label(feel_true),
            "confidence":    round(np.random.uniform(0.4, 0.95), 2),
        })

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MOMENT model validation")
    parser.add_argument("--predictions",  default=PREDICTIONS_FILE,  help="Path to model output JSON")
    parser.add_argument("--ground-truth", default=GROUND_TRUTH_FILE, help="Path to ground truth JSON")
    parser.add_argument("--output",       default=OUTPUT_DIR,        help="Output directory")
    parser.add_argument("--test-size",    type=float, default=0.2,   help="Hold-out test size")
    parser.add_argument("--demo",         action="store_true",       help="Run with fake data")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MODEL VALIDATION — MOMENT PROJECT")
    logger.info("=" * 60)

    # ── Load data ──
    if args.demo:
        df = generate_demo_data()
        _, test_df = make_holdout_split(df, test_size=args.test_size)
    else:
        pred_df  = load_predictions(args.predictions)
        truth_df = load_ground_truth(args.ground_truth)
        merged   = merge_predictions_and_truth(pred_df, truth_df)
        _, test_df = make_holdout_split(merged, test_size=args.test_size)

    # ── THINK metrics ──
    think_proba   = test_df[["think_R", "think_C", "think_D"]].values
    metrics_think = compute_metrics(
        test_df["think_true"].values,
        test_df["think_pred"].values,
        think_proba,
        "think"
    )

    # ── FEEL metrics ──
    feel_proba   = test_df[["feel_R", "feel_C", "feel_D"]].values
    metrics_feel = compute_metrics(
        test_df["feel_true"].values,
        test_df["feel_pred"].values,
        feel_proba,
        "feel"
    )

    # ── Confidence analysis ──
    conf_analysis = analyse_confidence(test_df, args.output)

    # ── Save ──
    save_results(metrics_think, metrics_feel, conf_analysis, args.output)

    logger.info(f"\n✓ Validation complete. All outputs in: {args.output}/")


if __name__ == "__main__":
    main()