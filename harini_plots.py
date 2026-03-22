import json
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import mlflow # type: ignore

RESULTS_FILE = "training_results.json"

with open(RESULTS_FILE) as f:
    results = json.load(f)

mlflow.set_experiment("moment_compatibility")

with mlflow.start_run(run_name="two_tower_plots_epochs30"):

    # ── Plot 1: F1 Score Comparison ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35

    final_f1  = [results["think_f1_weighted"], results["feel_f1_weighted"]]
    best_f1   = [results["best_think_f1"],     results["best_feel_f1"]]

    bars1 = ax.bar(x - width/2, final_f1, width, label="Final F1",
                   color=["#4C9BE8", "#4CE8A0"], alpha=0.85)
    bars2 = ax.bar(x + width/2, best_f1,  width, label="Best F1",
                   color=["#1a6bbf", "#1abf6b"], alpha=0.85)

    ax.set_title("Two-Tower Model — F1 Score Comparison", fontsize=14)
    ax.set_ylabel("F1 Score (Weighted)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Think Head\n(Intellectual)", "Feel Head\n(Emotional)"])
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.axhline(y=0.75, color="gray", linestyle="--", alpha=0.5, label="0.75 baseline")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("f1_comparison.png")
    mlflow.log_artifact("f1_comparison.png")
    plt.close()
    print("Plot 1 saved.")

    # ── Plot 2: NDCG Comparison ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    labels  = ["Think NDCG", "Feel NDCG"]
    values  = [results["think_ndcg"], results["feel_ndcg"]]
    colors  = ["#4C9BE8", "#4CE8A0"]

    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.4)
    ax.set_title("Two-Tower Model — NDCG Score by Head", fontsize=14)
    ax.set_ylabel("NDCG Score")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.4f}", ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig("ndcg_comparison.png")
    mlflow.log_artifact("ndcg_comparison.png")
    plt.close()
    print("Plot 2 saved.")

    # ── Plot 3: Train vs Test Split ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes  = [results["train_size"], results["test_size"]]
    labels = [f"Train\n{results['train_size']} samples",
              f"Test\n{results['test_size']} samples"]
    colors = ["#4C9BE8", "#E87C4C"]
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        explode=explode, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 12}
    )
    ax.set_title("Train / Test Split", fontsize=14)

    plt.tight_layout()
    plt.savefig("train_test_split.png")
    mlflow.log_artifact("train_test_split.png")
    plt.close()
    print("Plot 3 saved.")

    # ── Plot 4: All metrics summary ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    metrics = {
        "think_f1\n(weighted)":  results["think_f1_weighted"],
        "feel_f1\n(weighted)":   results["feel_f1_weighted"],
        "best_think\nf1":        results["best_think_f1"],
        "best_feel\nf1":         results["best_feel_f1"],
    }

    colors = ["#4C9BE8", "#4CE8A0", "#1a6bbf", "#1abf6b"]
    bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                  color=colors, alpha=0.85, width=0.5)

    ax.set_title("Two-Tower Model — Full Performance Summary", fontsize=14)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.75, color="gray", linestyle="--",
               alpha=0.5, label="0.75 reference line")
    ax.legend()

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.4f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("performance_summary.png")
    mlflow.log_artifact("performance_summary.png")
    plt.close()
    print("Plot 4 saved.")

    print("\nAll plots saved and logged to MLflow.")