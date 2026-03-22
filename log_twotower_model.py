import json
import torch # type: ignore
import mlflow # type: ignore
import mlflow.pytorch # type: ignore

MODEL_FILE    = "two_tower_model.pt"
RESULTS_FILE  = "training_results.json"
LABELS_FILE   = "training_labels.json"

with open(RESULTS_FILE) as f:
    results = json.load(f)

with open(LABELS_FILE) as f:
    labels = json.load(f)

state_dict = torch.load(MODEL_FILE, map_location="cpu")

mlflow.set_experiment("moment_compatibility")

with mlflow.start_run(run_name="two_tower_model_epochs30"):

    # Log parameters
    mlflow.log_params({
        "model_type":        "two_tower",
        "epochs":            results["epochs"],
        "train_size":        results["train_size"],
        "test_size":         results["test_size"],
        "total_samples":     results["train_size"] + results["test_size"],
        "train_test_split":  "80/20",
        "heads":             "think_head + feel_head",
        "towers":            "tower_a + tower_b",
    })

    # Log all metrics from training_results.json
    mlflow.log_metrics({
        "think_f1_weighted":  results["think_f1_weighted"],
        "feel_f1_weighted":   results["feel_f1_weighted"],
        "think_ndcg":         results["think_ndcg"],
        "feel_ndcg":          results["feel_ndcg"],
        "best_think_f1":      results["best_think_f1"],
        "best_feel_f1":       results["best_feel_f1"],
        "avg_f1_weighted":    (results["think_f1_weighted"] + results["feel_f1_weighted"]) / 2,
        "avg_best_f1":        (results["best_think_f1"] + results["best_feel_f1"]) / 2,
    })

    # Log the model file as artifact
    mlflow.log_artifact(MODEL_FILE)
    mlflow.log_artifact(RESULTS_FILE)
    mlflow.log_artifact(LABELS_FILE)

    print("Run logged successfully.")
    print(f"think_f1_weighted:  {results['think_f1_weighted']:.4f}")
    print(f"feel_f1_weighted:   {results['feel_f1_weighted']:.4f}")
    print(f"best_think_f1:      {results['best_think_f1']:.4f}")
    print(f"best_feel_f1:       {results['best_feel_f1']:.4f}")
    print(f"avg_f1_weighted:    {(results['think_f1_weighted'] + results['feel_f1_weighted']) / 2:.4f}")