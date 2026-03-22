# Experiment Tracking — MOMENT Two-Tower Model

## Tool
MLflow 3.10.1 — run locally, logged to `mlruns/`  
Experiment name: `moment_compatibility`  
Run name: `two_tower_model_epochs30`

---

## Run Overview

| Parameter | Value |
|---|---|
| model_type | two_tower |
| epochs | 30 |
| train_size | 978 |
| test_size | 245 |
| total_samples | 1223 |
| train_test_split | 80/20 |
| heads | think_head + feel_head |
| towers | tower_a + tower_b |

---

## Model Architecture

The two-tower model processes two readers independently through separate 
encoder networks (tower_a and tower_b), then passes the combined 
representation through two task-specific classification heads:

- **think_head** — classifies intellectual compatibility (how readers 
  analytically engage with text)
- **feel_head** — classifies emotional compatibility (how readers 
  emotionally respond to text)

Each tower encodes one reader's portrait embedding through two linear 
layers. The outputs are combined and passed to each head independently, 
producing separate compatibility verdicts for intellectual and emotional 
dimensions.

---

## Metrics

| Metric | Value |
|---|---|
| think_f1_weighted | 0.7871 |
| feel_f1_weighted | 0.7306 |
| think_ndcg | 3.4513 |
| feel_ndcg | 2.6671 |
| best_think_f1 | 0.8048 |
| best_feel_f1 | 0.7306 |
| avg_f1_weighted | 0.7589 |
| avg_best_f1 | 0.7677 |

---

## Visualizations

### F1 Score Comparison — Think Head vs Feel Head

![F1 Comparison](images/f1_comparison.png)

The think head (intellectual compatibility) outperforms the feel head 
(emotional compatibility) on both final and best F1 scores. 
think_f1_weighted reaches 0.7871 with a best of 0.8048, while 
feel_f1_weighted plateaus at 0.7306 — the final and best scores are 
identical, indicating the feel head converged early and did not improve 
further after its best checkpoint. Both heads exceed the 0.75 reference 
line for the think head, while the feel head sits just below it.

---

### Confidence Score Distribution by Match Type

![Confidence Distribution](images/confidence_distribution.png)

The confidence score distribution shows that resonance and mirror pairs 
are heavily concentrated in the 0.88–0.95 range, indicating high agent 
certainty on aligned and opposing reader pairs. Contradiction pairs show 
a wider, lower distribution — several fall below 0.80 — reflecting 
genuine agent uncertainty on structurally complex classifications.

---

### Confidence Score Range by Match Type

![Confidence Boxplot](images/confidence_boxplot.png)

The box plot confirms the confidence gap across match types:

- **Resonance**: median ~0.90, tight interquartile range, two low 
  outliers at 0.78 and 0.82
- **Mirror**: median ~0.85, wider spread, whisker reaching down to 0.78
- **Contradiction**: median ~0.80, widest spread, one outlier as low 
  as 0.72

The agent assigns systematically lower confidence to contradiction pairs. 
This is expected — contradiction matches require reasoning about 
productive tension between different interpretive frameworks, which is 
a harder judgment than detecting alignment (resonance) or opposition 
(mirror).

---

## Training Labels Schema

Each training label record contains the following fields:

| Field | Description |
|---|---|
| pair_id | Unique identifier for the reader pair |
| book_title | Book both readers read |
| user_a_id / user_b_id | Reader identifiers |
| user_a_name / user_b_name | Reader names |
| think_label | Intellectual compatibility label |
| feel_label | Emotional compatibility label |
| confidence | Label confidence score |
| think_reasoning | Agent reasoning for think classification |
| feel_reasoning | Agent reasoning for feel classification |
| overall_summary | Full compatibility summary |
| a_how_they_read | How reader A engages with text |
| a_interpretive_lens | Reader A's conceptual framework |
| a_central_preoccupation | Reader A's main thematic focus |
| a_what_moves_them | What emotionally affects reader A |
| a_emotional_mode | Reader A's emotional engagement style |
| b_how_they_read | How reader B engages with text |
| b_interpretive_lens | Reader B's conceptual framework |
| b_central_preoccupation | Reader B's main thematic focus |
| b_what_moves_them | What emotionally affects reader B |
| b_emotional_mode | Reader B's emotional engagement style |

Sample record from `training_labels.json`:
- pair: Emma Chen vs Marcus Williams on Frankenstein
- think_label: Divergence
- feel_label: Divergence
- confidence: 1.0
- Emma reads prosecutorially — diagnosing character flaws and 
  psychological failures
- Marcus reads empathetically — mapping Victor's experiences onto 
  his own life

---

## Key Findings

**Think head outperforms feel head.** Intellectual compatibility 
(think_f1: 0.7871) is easier to classify than emotional compatibility 
(feel_f1: 0.7306). This makes intuitive sense — intellectual engagement 
patterns are more consistent and structurally distinct in the training 
data, while emotional responses are more varied and harder to separate 
into clean categories.

**Feel head converged early.** The identical final and best feel_f1 
scores (0.7306) indicate the feel head reached its performance ceiling 
before epoch 30. Future experiments should test early stopping on the 
feel head or a higher learning rate specifically for that head.

**NDCG gap between heads.** think_ndcg (3.4513) is notably higher than 
feel_ndcg (2.6671), confirming the think head ranks compatible pairs 
more accurately than the feel head. This suggests the model is more 
reliable for intellectual matching than emotional matching at this 
training scale.

**Training data size.** 978 training samples and 245 test samples is a 
relatively small dataset for a neural model. Performance is expected to 
improve significantly as more reader pairs are logged through the 
instrumentation layer and added to training.

---

## Artifacts Logged

| File | Description |
|---|---|
| two_tower_model.pt | Trained PyTorch model weights |
| training_results.json | Final metrics from training run |
| training_labels.json | Full labeled dataset used for training |
| f1_comparison.png | F1 score comparison bar chart |
| confidence_distribution.png | Confidence histogram by match type |
| confidence_boxplot.png | Confidence range box plot by match type |

---

## Next Experiment

Planned runs:
- Vary epochs (20, 40) to check if feel head improves with more training
- Test separate learning rates for think_head and feel_head
- Expand training labels as instrumentation layer accumulates more data