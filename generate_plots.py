import json
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from collections import Counter
import mlflow # type: ignore

with open("data/processed/compatibility_results.json") as f:
    results = json.load(f)

mlflow.set_experiment("moment_compatibility")

with mlflow.start_run(run_name="plots_threshold_0.82"):

    # ── Plot 1: Match type distribution bar chart ──
    types = Counter(r["match_type"] for r in results)
    labels = list(types.keys())
    values = list(types.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4C9BE8", "#E87C4C", "#4CE8A0", "#888888"])
    ax.set_title("Match Type Distribution", fontsize=14)
    ax.set_ylabel("Number of Pairs")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig("data/processed/match_type_distribution.png")
    mlflow.log_artifact("data/processed/match_type_distribution.png")
    plt.close()

    # ── Plot 2: Confidence score distribution by match type ──
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"resonance": "#4C9BE8", "mirror": "#E87C4C", "contradiction": "#4CE8A0"}
    for match_type, color in colors.items():
        scores = [r["confidence"] for r in results if r["match_type"] == match_type]
        if scores:
            ax.hist(scores, bins=15, alpha=0.6, label=match_type, color=color)
    ax.set_title("Confidence Score Distribution by Match Type", fontsize=14)
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/processed/confidence_distribution.png")
    mlflow.log_artifact("data/processed/confidence_distribution.png")
    plt.close()

    # ── Plot 3: Confidence by match type box plot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    data_by_type = [
        [r["confidence"] for r in results if r["match_type"] == t]
        for t in ["resonance", "mirror", "contradiction"]
    ]
    bp = ax.boxplot(data_by_type, labels=["resonance", "mirror", "contradiction"],
                    patch_artist=True)
    colors_list = ["#4C9BE8", "#E87C4C", "#4CE8A0"]
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Confidence Score Range by Match Type", fontsize=14)
    ax.set_ylabel("Confidence Score")
    plt.tight_layout()
    plt.savefig("data/processed/confidence_boxplot.png")
    mlflow.log_artifact("data/processed/confidence_boxplot.png")
    plt.close()

    print("All plots saved and logged to MLflow.")