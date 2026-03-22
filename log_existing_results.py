import json
import mlflow # type: ignore
from collections import Counter

OUTPUT_FILE = "data/processed/compatibility_results.json"

with open(OUTPUT_FILE) as f:
    results = json.load(f)

types = Counter(r["match_type"] for r in results)
confidences = [r["confidence"] for r in results]
avg_confidence = sum(confidences) / len(confidences)

avg_by_type = {}
for match_type in ["resonance", "mirror", "contradiction", "no_match"]:
    scores = [r["confidence"] for r in results if r["match_type"] == match_type]
    avg_by_type[match_type] = round(sum(scores) / len(scores), 4) if scores else 0

mlflow.set_experiment("moment_compatibility")

with mlflow.start_run(run_name="threshold_0.82_topk_20_existing"):

    mlflow.log_params({
        "similarity_threshold": 0.82,
        "faiss_top_k":          20,
        "embedding_model":      "hkunlp/instructor-xl",
        "prompt_version":       "v1",
        "total_moments":        450,
        "total_users":          50,
        "candidate_pairs":      393
    })

    mlflow.log_metrics({
        "total_results":                len(results),
        "resonance_count":              types.get("resonance", 0),
        "mirror_count":                 types.get("mirror", 0),
        "contradiction_count":          types.get("contradiction", 0),
        "no_match_count":               types.get("no_match", 0),
        "resonance_pct":                round(types.get("resonance", 0) / len(results) * 100, 2),
        "mirror_pct":                   round(types.get("mirror", 0) / len(results) * 100, 2),
        "contradiction_pct":            round(types.get("contradiction", 0) / len(results) * 100, 2),
        "avg_confidence":               round(avg_confidence, 4),
        "avg_confidence_resonance":     avg_by_type["resonance"],
        "avg_confidence_mirror":        avg_by_type["mirror"],
        "avg_confidence_contradiction": avg_by_type["contradiction"],
    })

    mlflow.log_artifact(OUTPUT_FILE)

print(f"Logged {len(results)} results to MLflow.")
print(f"Match types: {dict(types)}")
print(f"Avg confidence: {avg_confidence:.4f}")