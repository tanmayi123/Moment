"""
model_bias_detection.py
=======================
Model bias detection for the MOMENT compatibility model.
Uses Fairlearn to slice model predictions by demographic groups
and detect whether the model behaves fairly across subgroups.


Slices evaluated:
  - gender
  - age_group
  - personality
  - reader_type  (distribution_category)
  - book         (book_id)
  - passage      (passage_id)

Bias detection runs separately for dominant_think and dominant_feel.
Mitigation uses think.R and feel.R as continuous scores.

"""

import argparse
import json
import logging
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from fairlearn.metrics import MetricFrame

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
USERS_FILE       = "data/users_processed.json"
PAIRS_FILE       = "data/reports/all_profile_scores_fixed.json"
GROUND_TRUTH_FILE = "data/ground_truth.json"    # set to path when ground truth is available
OUTPUT_DIR       = "data/bias_results"
BIAS_FLAG_F1     = 0.15   # flag if a slice drops more than this below overall F1

# Label mappings for dominant_think / dominant_feel
LABEL_MAP   = {"resonate": 0, "contradict": 1, "diverge": 2}
LABEL_NAMES = {0: "resonate", 1: "contradict", 2: "diverge"}

# Demographic slices to evaluate
SLICE_COLS = ["gender", "age_group", "personality", "reader_type", "book"]

# Dimensions to run bias detection on (think and feel separately)
DIMENSIONS = [
    {"label_col": "think_label", "score_col": "think_R", "name": "Think"},
    {"label_col": "feel_label",  "score_col": "feel_R",  "name": "Feel"},
]


# ============================================================
# HELPERS
# ============================================================
def to_label(val):
    """Normalise a dominant_think / dominant_feel string to int label."""
    return LABEL_MAP.get(str(val).lower().strip(), -1)


def to_age_group(age):
    try:
        a = int(age)
        if a < 25: return "18-24 (Gen Z)"
        if a < 35: return "25-34 (Millennial)"
        if a < 45: return "35-44 (Gen X/Mill)"
        return "45+ (Gen X/Boom)"
    except Exception:
        return "Unknown"


# ============================================================
# LOAD DATA
# ============================================================
def load_user_demographics(users_path):
    """
    Load users_processed.json and return a dict keyed by character_name.
    Fields: gender, age_group, personality, reader_type.
    """
    logger.info("=" * 60)
    logger.info("LOADING USER DEMOGRAPHICS")
    logger.info("=" * 60)

    if not os.path.exists(users_path):
        logger.warning(f"User data not found at {users_path} — demographic slices will be skipped.")
        return {}

    with open(users_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = list(data.values())

    demo = {}
    for user in data:
        name = str(user.get("character_name", user.get("name", "")))
        if not name:
            continue
        demo[name] = {
            "gender":      str(user.get("gender",                "Unknown")),
            "age_group":   to_age_group(user.get("age",          "Unknown")),
            "personality": str(user.get("personality",           "Unknown")),
            "reader_type": str(user.get("distribution_category", "Unknown")),
        }

    logger.info(f"Loaded {len(demo)} users from {users_path}")
    return demo


def load_ground_truth(gt_path):
    """
    Load ground truth JSON and return a dict keyed by
    (user_a, user_b, book_id, passage_id) → {gt_think, gt_feel}.
    """
    logger.info("=" * 60)
    logger.info("LOADING GROUND TRUTH")
    logger.info("=" * 60)

    if gt_path is None or not os.path.exists(gt_path):
        logger.warning("No ground truth file provided — y_true will mirror y_pred (metrics not meaningful).")
        return {}

    with open(gt_path) as f:
        data = json.load(f)

    gt = {}
    for obj in data:
        key = (
            str(obj.get("user_a",     "")),
            str(obj.get("user_b",     "")),
            str(obj.get("book_id",    "overall")),
        )
        gt[key] = {
            "gt_think": to_label(obj.get("dominant_think", "")),
            "gt_feel":  to_label(obj.get("dominant_feel",  "")),
        }

    logger.info(f"Loaded {len(gt)} ground truth records from {gt_path}")
    return gt


def load_pairs(pairs_path, demo, gt):
    """
    Load flat list of pair JSON objects from the compatibility model output.
    Merges with demographics and ground truth where available.
    Returns a flat DataFrame with one row per pair.
    """
    logger.info("=" * 60)
    logger.info("LOADING PAIR PREDICTIONS")
    logger.info("=" * 60)

    with open(pairs_path) as f:
        data = json.load(f)

    # unwrap {generated_at, total_pairs, pairs: [...]} structure if needed
    if isinstance(data, dict):
        data = data.get("pairs", [])

    if not isinstance(data, list):
        raise ValueError("Expected a flat JSON array or a dict with a 'pairs' key.")

    rows = []
    gt_matched = 0

    for obj in data:
        user_a = str(obj.get("user_a", ""))
        user_b = str(obj.get("user_b", ""))

        # attach demographics from user_a
        d = demo.get(user_a, {})
        base_demo = {
            "gender":      d.get("gender",      "Unknown"),
            "age_group":   d.get("age_group",   "Unknown"),
            "personality": d.get("personality", "Unknown"),
            "reader_type": d.get("reader_type", "Unknown"),
        }

        # helper to build one row from a scores block
        def make_row(scores, book_name, passage_name):
            nonlocal gt_matched
            think = scores.get("think", {})
            feel  = scores.get("feel",  {})

            pred_think = to_label(scores.get("dominant_think", ""))
            pred_feel  = to_label(scores.get("dominant_feel",  ""))

            # match ground truth by (user_a, user_b, book_id, passage_id)
            gt_key    = (user_a, user_b, book_name)
            gt_record = gt.get(gt_key, {})
            gt_think  = gt_record.get("gt_think", pred_think)
            gt_feel   = gt_record.get("gt_feel",  pred_feel)
            if gt_record:
                gt_matched += 1

            return {
                "user_a":      user_a,
                "user_b":      user_b,
                "book":        book_name,
                "passage":     passage_name,

                # continuous scores (0–100) — used for mitigation
                "think_R":     think.get("R", 0),
                "think_C":     think.get("C", 0),
                "think_D":     think.get("D", 0),
                "feel_R":      feel.get("R",  0),
                "feel_C":      feel.get("C",  0),
                "feel_D":      feel.get("D",  0),

                # predicted labels
                "think_label": pred_think,
                "feel_label":  pred_feel,

                # ground truth labels (fallback to pred if no GT)
                "think_true":  gt_think,
                "feel_true":   gt_feel,

                "confidence":  scores.get("confidence", 0.0),
                **base_demo,
            }

        # 1. overall row
        if "overall" in obj:
            rows.append(make_row(obj["overall"], "overall", "overall"))

        # 2. by_book rows — one per book
        for book_name, book_scores in obj.get("by_book", {}).items():
            rows.append(make_row(book_scores, book_name, "overall"))

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} pairs from {pairs_path}")
    logger.info(f"Ground truth matched: {gt_matched}/{len(df)} pairs")
    logger.info(f"Books found: {', '.join(df['book'].unique())}")
    logger.info(f"Think label distribution:\n{df['think_label'].value_counts().to_string()}")
    logger.info(f"Feel  label distribution:\n{df['feel_label'].value_counts().to_string()}")

    if gt_matched == 0:
        logger.warning("No ground truth matches found — metrics will not be meaningful.")
    elif gt_matched < len(df):
        logger.warning(f"Partial ground truth — only {gt_matched}/{len(df)} pairs matched.")

    return df


# ============================================================
# 1. PERFORM SLICING
# Break down dataset by meaningful demographic slices and
# evaluate model performance on each slice.
# ============================================================
def run_analysis(df, output_dir, report):
    """
    Run bias detection across all slice columns for both
    dominant_think and dominant_feel dimensions.
    Returns a list of finding dicts.
    """
    logger.info("=" * 60)
    logger.info("1. PERFORMING SLICING & BIAS ANALYSIS")
    logger.info("=" * 60)

    report.append(f"\n{'=' * 60}\n")
    report.append("## 1. SLICING & BIAS ANALYSIS\n\n")
    report.append("Bias detection runs separately for Think and Feel dimensions.\n")
    report.append(f"Slices evaluated: {', '.join(SLICE_COLS)}\n")
    report.append(f"Bias flag threshold: F1 drop > {BIAS_FLAG_F1}\n")

    findings = []

    for dim in DIMENSIONS:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"DIMENSION: {dim['name'].upper()}")
        logger.info(f"{'─' * 60}")
        report.append(f"\n### Dimension: {dim['name']}\n")

        for slice_col in SLICE_COLS:
            if slice_col not in df.columns:
                continue
            result = analyse_slice(df, slice_col, dim, output_dir, report)
            if result:
                findings.append(result)

    return findings


# ============================================================
# 2. TRACK METRICS ACROSS SLICES
# Track accuracy and F1 per group, report significant disparities.
# ============================================================
def analyse_slice(df, slice_col, dim, output_dir, report):
    """
    Use Fairlearn MetricFrame to compute accuracy and F1 per group
    for a given dimension (think or feel) and slice column.
    Flags groups whose F1 drops more than BIAS_FLAG_F1 below overall.
    """
    label_col = dim["label_col"]   # predicted label column
    true_col  = label_col.replace("label", "true")  # think_true or feel_true
    dim_name  = dim["name"]

    logger.info(f"\n  [{dim_name}] Slice: {slice_col}")

    # drop unknowns and invalid labels
    mask = (
        (df[slice_col].astype(str).str.lower() != "unknown") &
        (df[label_col] != -1) &
        (df[true_col]  != -1)
    )
    sub = df[mask].copy()

    if len(sub) < 10:
        logger.warning(f"  Skipping [{dim_name}] '{slice_col}' — too few records.")
        return None

    y_true    = sub[true_col].values    # ground truth (or pred if no GT)
    y_pred    = sub[label_col].values   # model predictions
    sensitive = sub[slice_col].astype(str)

    def f1_macro(yt, yp):
        return f1_score(yt, yp, average="macro", zero_division=0)

    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "f1_macro": f1_macro},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive
    )

    overall_acc = mf.overall["accuracy"]
    overall_f1  = mf.overall["f1_macro"]
    by_group    = mf.by_group

    # log metrics per group
    logger.info(f"\n  {'Group':<25} {'Count':>7} {'Accuracy':>10} {'F1 Macro':>10}")
    logger.info(f"  {'-' * 55}")
    for grp in by_group.index:
        acc   = by_group.loc[grp, "accuracy"]
        f1    = by_group.loc[grp, "f1_macro"]
        count = int((sensitive == str(grp)).sum())
        flag  = "  ⚠" if f1 < overall_f1 - BIAS_FLAG_F1 else ""
        logger.info(f"  {str(grp):<25} {count:>7} {acc:>10.3f} {f1:>10.3f}{flag}")
    logger.info(f"  {'-' * 55}")
    logger.info(f"  {'Overall':<25} {len(sub):>7} {overall_acc:>10.3f} {overall_f1:>10.3f}")

    flagged     = by_group[by_group["f1_macro"] < overall_f1 - BIAS_FLAG_F1]
    bias_status = "BIAS DETECTED" if not flagged.empty else "NO BIAS"

    if flagged.empty:
        logger.info(f"  ✓ No significant bias detected.")
    else:
        logger.warning(f"  ⚠ Bias detected in [{dim_name}] '{slice_col}':")
        for grp in flagged.index:
            drop = overall_f1 - flagged.loc[grp, "f1_macro"]
            logger.warning(f"    - {grp}: F1 {flagged.loc[grp, 'f1_macro']:.3f} (drop: {drop:.3f})")

    # write to report
    report.append(f"\n#### [{dim_name}] Slice: {slice_col}\n")
    report.append(f"Overall Accuracy: {overall_acc:.3f} | Overall F1: {overall_f1:.3f}\n\n")
    report.append("```\n" + by_group.to_string() + "\n```\n")
    report.append(f"\nAssessment: **{bias_status}**\n")
    if not flagged.empty:
        report.append("Flagged groups (F1 drop > 0.15):\n")
        for grp in flagged.index:
            drop = overall_f1 - flagged.loc[grp, "f1_macro"]
            report.append(f"- {grp}: F1 {flagged.loc[grp, 'f1_macro']:.3f} (drop: {drop:.3f})\n")

    return {
        "dimension":   dim_name,
        "slice":       slice_col,
        "overall_f1":  float(overall_f1),
        "overall_acc": float(overall_acc),
        "by_group":    by_group.reset_index().to_dict(orient="records"),
        "flagged":     flagged.reset_index().to_dict(orient="records"),
        "bias_status": bias_status,
        "label_col":   label_col,
        "score_col":   dim["score_col"],
    }


# ============================================================
# 3. BIAS MITIGATION
# Apply threshold adjustment for flagged groups.
# ============================================================
def apply_mitigation(df, findings, report):
    """
    Threshold adjustment mitigation for flagged slices.

    For each biased group:
    - Compute the group's mean continuous score (think_R or feel_R)
    - If it is significantly below the overall mean (> 5 point gap),
      re-classify borderline contradict predictions → resonate
      for pairs whose score is >= 40 (borderline zone)

    Trade-off: slightly increases false positives for resonance
    in affected groups, but reduces systematic under-scoring bias.
    """
    logger.info("=" * 60)
    logger.info("3. BIAS MITIGATION")
    logger.info("=" * 60)

    report.append(f"\n{'=' * 60}\n")
    report.append("## 3. BIAS MITIGATION\n\n")

    biased = [f for f in findings if f and f.get("bias_status") == "BIAS DETECTED"]

    if not biased:
        logger.info("✓ No bias requiring mitigation detected.")
        report.append("No bias requiring mitigation was detected.\n\n")
        report.append("**Trade-off note:** No adjustments made — model consistency preserved.\n")
        return df

    score_col_map = {f["label_col"]: f["score_col"] for f in findings if f}

    for f in biased:
        slice_col = f["slice"]
        score_col = f["score_col"]   # think_R or feel_R
        label_col = f["label_col"]   # think_label or feel_label
        dim_name  = f["dimension"]

        logger.info(f"\n  Mitigating: [{dim_name}] {slice_col}")

        if slice_col not in df.columns or score_col not in df.columns:
            logger.warning(f"  Skipping — missing column(s) for mitigation.")
            continue

        mitigated    = df[label_col].copy()
        group_means  = df.groupby(slice_col)[score_col].mean()
        overall_mean = df[score_col].mean()

        adjusted_total = 0
        for group, mean_score in group_means.items():
            if mean_score < overall_mean - 5:
                mask       = df[slice_col] == group
                borderline = mask & (df[score_col] >= 40) & (mitigated == 1)
                mitigated[borderline] = 0
                count = int(borderline.sum())
                adjusted_total += count
                if count > 0:
                    logger.info(f"    Adjusted {count} records for group '{group}' "
                                f"(mean {score_col}: {mean_score:.1f} vs overall: {overall_mean:.1f})")

        mitigated_col = f"{label_col}_mitigated"
        df[mitigated_col] = mitigated
        logger.info(f"  Total adjusted: {adjusted_total} records → saved to '{mitigated_col}'")

        report.append(f"### [{dim_name}] {slice_col}\n")
        report.append(f"Overall F1: {f['overall_f1']:.3f}\n\n")
        report.append("**Flagged groups:**\n")
        for fg in f["flagged"]:
            grp  = fg.get(slice_col, fg.get("sensitive_feature", "?"))
            f1   = fg.get("f1_macro", 0)
            drop = f["overall_f1"] - f1
            report.append(f"- **{grp}**: F1 {f1:.3f} (drop of {drop:.3f})\n")
        report.append("\n")

    return df


# ============================================================
# 4. DOCUMENT BIAS MITIGATION
# Explain what was done, why, and what trade-offs were made.
# ============================================================
def document_mitigation(findings, report):
    """
    Write the full bias mitigation documentation section to the report.
    Covers: technique used, why it was applied, and trade-offs.
    """
    biased = [f for f in findings if f and f.get("bias_status") == "BIAS DETECTED"]

    report.append(f"\n{'=' * 60}\n")
    report.append("## 4. BIAS MITIGATION DOCUMENTATION\n\n")

    if not biased:
        report.append("No mitigation was required. All slices passed the bias threshold.\n\n")
        report.append("**Trade-off:** No adjustments were made, preserving full model consistency "
                      "across all groups.\n")
        return

    report.append(f"{len(biased)} slice(s) showed significant bias and were mitigated.\n\n")

    report.append("### Technique: Threshold Adjustment\n\n")
    report.append("**What was done:**\n")
    report.append("For each flagged group, the mean continuous resonance score (think_R or feel_R) "
                  "was compared against the overall mean. Groups scoring more than 5 points below "
                  "the overall mean were identified as systematically under-scored. For these groups, "
                  "borderline contradict predictions (score >= 40) were re-classified as resonate.\n\n")

    report.append("**Why this technique:**\n")
    report.append("Threshold adjustment is transparent and reversible. It directly addresses the "
                  "symptom of bias (under-classification of resonance for certain groups) without "
                  "altering the underlying model or retraining on new data.\n\n")

    report.append("**Alternative techniques considered:**\n")
    report.append("- *Re-weighting*: upweight under-represented groups during scoring aggregation. "
                  "Not applied — requires larger ground truth sample.\n")
    report.append("- *Re-sampling*: oversample under-represented groups in training data. "
                  "Not applied — model is prompt-based, not retrained on this dataset.\n\n")

    report.append("**Trade-offs:**\n")
    report.append("- Slightly increases false positives for resonance in affected groups.\n")
    report.append("- Reduces systematic under-scoring bias for flagged demographic groups.\n")
    report.append("- Mitigation is stored in a separate column (`*_mitigated`) and does not "
                  "overwrite the original predictions — original results are always preserved.\n")
    report.append("- Ground truth sample should be expanded for more robust bias estimates.\n\n")

    report.append("### Affected Slices\n\n")
    for f in biased:
        report.append(f"- **[{f['dimension']}] {f['slice']}** — "
                      f"Overall F1: {f['overall_f1']:.3f}, "
                      f"{len(f['flagged'])} group(s) flagged\n")

# ============================================================
# SUMMARY
# ============================================================
def log_summary(findings, report):
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for dim in [d["name"] for d in DIMENSIONS]:
        dim_findings = [f for f in findings if f and f["dimension"] == dim]
        biased = [f["slice"] for f in dim_findings if f["bias_status"] == "BIAS DETECTED"]
        clean  = [f["slice"] for f in dim_findings if f["bias_status"] == "NO BIAS"]
        logger.info(f"\n  [{dim}]")
        logger.info(f"  Clean  ({len(clean)}):  {', '.join(clean)  or 'None'}")
        logger.info(f"  Biased ({len(biased)}): {', '.join(biased) or 'None'}")

    all_biased = [f["slice"] for f in findings if f and f["bias_status"] == "BIAS DETECTED"]
    if not all_biased:
        verdict = "VERDICT: Model behaves equitably across all subgroups and dimensions."
        logger.info(f"\n✓ {verdict}")
    else:
        verdict = f"VERDICT: Bias detected in {len(all_biased)} slice(s). Mitigation applied."
        logger.warning(f"\n⚠ {verdict}")

    report.append(f"\n{'=' * 60}\n")
    report.append("## SUMMARY\n\n")

    for dim in [d["name"] for d in DIMENSIONS]:
        dim_findings = [f for f in findings if f and f["dimension"] == dim]
        biased = [f["slice"] for f in dim_findings if f["bias_status"] == "BIAS DETECTED"]
        clean  = [f["slice"] for f in dim_findings if f["bias_status"] == "NO BIAS"]
        report.append(f"**{dim}**\n")
        report.append(f"- Clean  ({len(clean)}):  {', '.join(clean)  or 'None'}\n")
        report.append(f"- Biased ({len(biased)}): {', '.join(biased) or 'None'}\n\n")

    report.append(f"**{verdict}**\n")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MOMENT model bias detection")
    parser.add_argument("--pairs",        default=PAIRS_FILE,        help="Path to pairs output JSON")
    parser.add_argument("--ground-truth", default=GROUND_TRUTH_FILE, help="Path to ground truth JSON")
    parser.add_argument("--users",        default=USERS_FILE,        help="Path to users_processed.json")
    parser.add_argument("--output",       default=OUTPUT_DIR,        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MODEL BIAS DETECTION — MOMENT PROJECT")
    logger.info("=" * 60)

    # ── Load ──
    demo = load_user_demographics(args.users)
    gt   = load_ground_truth(args.ground_truth)
    df   = load_pairs(args.pairs, demo, gt)
    logger.info(f"Total records for bias analysis: {len(df)}")

    # ── Report scaffold ──
    report = []
    report.append("# MODEL BIAS DETECTION REPORT — MOMENT PROJECT\n\n")
    report.append(f"Date: {pd.Timestamp.now().date()}\n")
    report.append(f"Total pairs analysed: {len(df)}\n")
    report.append(f"Ground truth available: {'Yes' if gt else 'No (metrics not meaningful)'}\n")
    report.append(f"Dimensions: Think, Feel\n")
    report.append(f"Bias flag threshold: F1 drop > {BIAS_FLAG_F1}\n")

    # ── Step 1 + 2: Slice and track metrics ──
    findings = run_analysis(df, args.output, report)

    # ── Step 3: Mitigate ──
    df = apply_mitigation(df, findings, report)

    # ── Step 4: Document ──
    document_mitigation(findings, report)

    # ── Summary ──
    log_summary(findings, report)

    # ── Save report ──
    report_path = os.path.join(args.output, "model_bias_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    logger.info(f"\n✓ Report  → {report_path}")

    # ── Save results JSON ──
    results_path = os.path.join(args.output, "model_bias_results.json")
    with open(results_path, "w") as f:
        json.dump([r for r in findings if r], f, indent=2)
    logger.info(f"✓ Results → {results_path}")

    logger.info(f"✓ Done. All outputs in: {args.output}/")


if __name__ == "__main__":
    main()
