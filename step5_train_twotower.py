import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── Device ────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load data ─────────────────────────────────────────────────
print("\nLoading data...")
with open("output/training_labels.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Total samples: {len(df)}")
print(f"Think distribution:\n{df['think_label'].value_counts()}")
print(f"\nFeel distribution:\n{df['feel_label'].value_counts()}")

# ── Embed portrait fields ─────────────────────────────────────
print("\nLoading sentence transformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_portrait_text(row, prefix):
    return " ".join([
        str(row.get(f"{prefix}how_they_read", "")),
        str(row.get(f"{prefix}interpretive_lens", "")),
        str(row.get(f"{prefix}central_preoccupation", "")),
        str(row.get(f"{prefix}what_moves_them", "")),
        str(row.get(f"{prefix}emotional_mode", "")),
        str(row.get(f"{prefix}self_referential", "")),
        str(row.get(f"{prefix}reflection_density", "")),
    ])

print("Embedding User A portraits...")
texts_a = [get_portrait_text(row, "a_") for _, row in df.iterrows()]
embeddings_a = embedder.encode(texts_a, show_progress_bar=True)

print("Embedding User B portraits...")
texts_b = [get_portrait_text(row, "b_") for _, row in df.iterrows()]
embeddings_b = embedder.encode(texts_b, show_progress_bar=True)

print(f"Embedding shape: {embeddings_a.shape}")

# ── Encode labels ─────────────────────────────────────────────
le_think = LabelEncoder()
le_feel  = LabelEncoder()

y_think = le_think.fit_transform(df["think_label"])
y_feel  = le_feel.fit_transform(df["feel_label"])

print(f"\nThink classes: {le_think.classes_}")
print(f"Feel classes:  {le_feel.classes_}")

# ── Train/test split ──────────────────────────────────────────
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y_think
)
print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")

# ── Dataset ───────────────────────────────────────────────────
class ReaderPairDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return {
            "emb_a":       torch.tensor(embeddings_a[idx], dtype=torch.float32),
            "emb_b":       torch.tensor(embeddings_b[idx], dtype=torch.float32),
            "think_label": torch.tensor(y_think[idx], dtype=torch.long),
            "feel_label":  torch.tensor(y_feel[idx],  dtype=torch.long),
        }

train_dataset = ReaderPairDataset(train_idx)
test_dataset  = ReaderPairDataset(test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# ── Two-Tower Model ───────────────────────────────────────────
class Tower(nn.Module):
    def __init__(self, input_dim=384, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TwoTowerModel(nn.Module):
    def __init__(self, embed_dim=384, tower_dim=128, num_think=3, num_feel=3):
        super().__init__()
        self.tower_a = Tower(embed_dim, tower_dim)
        self.tower_b = Tower(embed_dim, tower_dim)

        interaction_dim = tower_dim * 3

        self.think_head = nn.Sequential(
            nn.Linear(interaction_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_think)
        )

        self.feel_head = nn.Sequential(
            nn.Linear(interaction_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_feel)
        )

    def forward(self, emb_a, emb_b):
        repr_a = self.tower_a(emb_a)
        repr_b = self.tower_b(emb_b)

        interaction = torch.cat([
            repr_a,
            repr_b,
            torch.abs(repr_a - repr_b),
        ], dim=1)

        think_logits = self.think_head(interaction)
        feel_logits  = self.feel_head(interaction)

        return think_logits, feel_logits


# ── Class weights ─────────────────────────────────────────────
def get_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)

think_weights = get_class_weights(y_think[train_idx], len(le_think.classes_))
feel_weights  = get_class_weights(y_feel[train_idx],  len(le_feel.classes_))

print(f"\nThink class weights: {think_weights}")
print(f"Feel class weights:  {feel_weights}")

# ── Training ──────────────────────────────────────────────────
model     = TwoTowerModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

think_criterion = nn.CrossEntropyLoss(weight=think_weights)
feel_criterion  = nn.CrossEntropyLoss(weight=feel_weights)

EPOCHS = 30
print(f"\n{'='*60}")
print(f"Training Two-Tower Model for {EPOCHS} epochs")
print(f"{'='*60}")

best_think_f1 = 0
best_feel_f1  = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for batch in train_loader:
        emb_a  = batch["emb_a"].to(device)
        emb_b  = batch["emb_b"].to(device)
        t_true = batch["think_label"].to(device)
        f_true = batch["feel_label"].to(device)

        optimizer.zero_grad()
        think_logits, feel_logits = model(emb_a, emb_b)

        loss = think_criterion(think_logits, t_true) + feel_criterion(feel_logits, f_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 5 == 0:
        model.eval()
        all_think_true, all_think_pred = [], []
        all_feel_true,  all_feel_pred  = [], []

        with torch.no_grad():
            for batch in test_loader:
                emb_a  = batch["emb_a"].to(device)
                emb_b  = batch["emb_b"].to(device)
                t_true = batch["think_label"].numpy()
                f_true = batch["feel_label"].numpy()

                think_logits, feel_logits = model(emb_a, emb_b)
                t_pred = think_logits.argmax(dim=1).cpu().numpy()
                f_pred = feel_logits.argmax(dim=1).cpu().numpy()

                all_think_true.extend(t_true)
                all_think_pred.extend(t_pred)
                all_feel_true.extend(f_true)
                all_feel_pred.extend(f_pred)

        think_f1 = f1_score(all_think_true, all_think_pred, average="weighted")
        feel_f1  = f1_score(all_feel_true,  all_feel_pred,  average="weighted")

        if think_f1 > best_think_f1: best_think_f1 = think_f1
        if feel_f1  > best_feel_f1:  best_feel_f1  = feel_f1

        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_loader):.4f} | Think F1: {think_f1:.4f} | Feel F1: {feel_f1:.4f}")

# ── Final evaluation ──────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL VALIDATION RESULTS")
print(f"{'='*60}")

model.eval()
all_think_true, all_think_pred = [], []
all_feel_true,  all_feel_pred  = [], []

with torch.no_grad():
    for batch in test_loader:
        emb_a  = batch["emb_a"].to(device)
        emb_b  = batch["emb_b"].to(device)
        t_true = batch["think_label"].numpy()
        f_true = batch["feel_label"].numpy()

        think_logits, feel_logits = model(emb_a, emb_b)
        t_pred = think_logits.argmax(dim=1).cpu().numpy()
        f_pred = feel_logits.argmax(dim=1).cpu().numpy()

        all_think_true.extend(t_true)
        all_think_pred.extend(t_pred)
        all_feel_true.extend(f_true)
        all_feel_pred.extend(f_pred)

print("\n--- Think Label ---")
print(classification_report(
    all_think_true, all_think_pred,
    target_names=le_think.classes_
))

print("\n--- Feel Label ---")
print(classification_report(
    all_feel_true, all_feel_pred,
    target_names=le_feel.classes_
))

print("\n--- Think Confusion Matrix ---")
print(pd.DataFrame(
    confusion_matrix(all_think_true, all_think_pred),
    index=le_think.classes_,
    columns=le_think.classes_
))

print("\n--- Feel Confusion Matrix ---")
print(pd.DataFrame(
    confusion_matrix(all_feel_true, all_feel_pred),
    index=le_feel.classes_,
    columns=le_feel.classes_
))

# ── NDCG ─────────────────────────────────────────────────────
def compute_ndcg(true_labels, pred_probs, label_order):
    relevance_map = {label: len(label_order) - i for i, label in enumerate(label_order)}
    ndcg_scores = []

    for true, probs in zip(true_labels, pred_probs):
        true_name  = label_order[true] if true < len(label_order) else "Unknown"
        ideal_gain = relevance_map.get(true_name, 0)
        ranked_labels = np.argsort(probs)[::-1]
        dcg = 0
        for rank, label_idx in enumerate(ranked_labels):
            label_name = label_order[label_idx] if label_idx < len(label_order) else "Unknown"
            gain = relevance_map.get(label_name, 0)
            dcg += gain / np.log2(rank + 2)
        idcg = ideal_gain / np.log2(2) if ideal_gain > 0 else 1
        ndcg_scores.append(dcg / idcg)

    return np.mean(ndcg_scores)

model.eval()
all_think_probs, all_feel_probs = [], []
all_think_true2, all_feel_true2 = [], []

with torch.no_grad():
    for batch in test_loader:
        emb_a = batch["emb_a"].to(device)
        emb_b = batch["emb_b"].to(device)

        think_logits, feel_logits = model(emb_a, emb_b)
        think_probs = torch.softmax(think_logits, dim=1).cpu().numpy()
        feel_probs  = torch.softmax(feel_logits,  dim=1).cpu().numpy()

        all_think_probs.extend(think_probs)
        all_feel_probs.extend(feel_probs)
        all_think_true2.extend(batch["think_label"].numpy())
        all_feel_true2.extend(batch["feel_label"].numpy())

think_order = list(le_think.classes_)
feel_order  = list(le_feel.classes_)

think_ndcg = compute_ndcg(all_think_true2, all_think_probs, think_order)
feel_ndcg  = compute_ndcg(all_feel_true2,  all_feel_probs,  feel_order)

print(f"\n--- Ranking Metrics ---")
print(f"Think NDCG: {think_ndcg:.4f}")
print(f"Feel NDCG:  {feel_ndcg:.4f}")

# ── Bias detection ────────────────────────────────────────────
print(f"\n{'='*60}")
print("BIAS DETECTION — Slicing by reflection density")
print(f"{'='*60}")

test_df = df.iloc[test_idx].copy()
test_df["think_pred"] = le_think.inverse_transform(all_think_pred)
test_df["feel_pred"]  = le_feel.inverse_transform(all_feel_pred)
test_df["think_correct"] = test_df["think_pred"] == test_df["think_label"]
test_df["feel_correct"]  = test_df["feel_pred"]  == test_df["feel_label"]

def get_density(val):
    val = str(val).lower()
    if "high" in val:   return "high"
    if "medium" in val: return "medium"
    if "low" in val:    return "low"
    return "unknown"

test_df["density_a"] = test_df["a_reflection_density"].apply(get_density)

print("\nThink accuracy by reflection density (User A):")
print(test_df.groupby("density_a")["think_correct"].mean().round(3))
print("\nFeel accuracy by reflection density (User A):")
print(test_df.groupby("density_a")["feel_correct"].mean().round(3))

# ── NEW: Save per-pair R/C/D scores ──────────────────────────
print(f"\n{'='*60}")
print("SAVING PER-PAIR R/C/D SCORES")
print(f"{'='*60}")

think_classes = list(le_think.classes_)
feel_classes  = list(le_feel.classes_)

pair_scores = []
test_df_reset = df.iloc[test_idx].copy().reset_index(drop=True)

for i, (think_prob, feel_prob) in enumerate(zip(all_think_probs, all_feel_probs)):
    if i >= len(test_df_reset):
        break
    row = test_df_reset.iloc[i]

    pair_scores.append({
        "user_a_name": row["user_a_name"],
        "user_b_name": row["user_b_name"],
        "book_title":  row["book_title"],

        # ── Think ──
        "think_true_label":          row["think_label"],
        "think_predicted_label":     le_think.inverse_transform([np.argmax(think_prob)])[0],
        "think_resonance_pct":       round(float(think_prob[think_classes.index("Resonance")]) * 100, 1),
        "think_contradiction_pct":   round(float(think_prob[think_classes.index("Contradiction")]) * 100, 1),
        "think_divergence_pct":      round(float(think_prob[think_classes.index("Divergence")]) * 100, 1),

        # ── Feel ──
        "feel_true_label":           row["feel_label"],
        "feel_predicted_label":      le_feel.inverse_transform([np.argmax(feel_prob)])[0],
        "feel_resonance_pct":        round(float(feel_prob[feel_classes.index("Resonance")]) * 100, 1),
        "feel_contradiction_pct":    round(float(feel_prob[feel_classes.index("Contradiction")]) * 100, 1),
        "feel_divergence_pct":       round(float(feel_prob[feel_classes.index("Divergence")]) * 100, 1),
    })

with open("models/pair_scores.json", "w") as f:
    json.dump(pair_scores, f, indent=2)

# Print a few examples
print(f"\n✅ {len(pair_scores)} pair scores saved to models/pair_scores.json")
print("\nSample predictions:")
for p in pair_scores[:3]:
    print(f"\n  {p['user_a_name']} vs {p['user_b_name']}")
    print(f"  Think → Resonance {p['think_resonance_pct']}% | Contradiction {p['think_contradiction_pct']}% | Divergence {p['think_divergence_pct']}%  (true: {p['think_true_label']})")
    print(f"  Feel  → Resonance {p['feel_resonance_pct']}% | Contradiction {p['feel_contradiction_pct']}% | Divergence {p['feel_divergence_pct']}%  (true: {p['feel_true_label']})")

# ── Save models ───────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/two_tower_model.pt")

with open("models/think_label_encoder.pkl", "wb") as f:
    pickle.dump(le_think, f)
with open("models/feel_label_encoder.pkl", "wb") as f:
    pickle.dump(le_feel, f)

results = {
    "think_f1_weighted": f1_score(all_think_true, all_think_pred, average="weighted"),
    "feel_f1_weighted":  f1_score(all_feel_true,  all_feel_pred,  average="weighted"),
    "think_ndcg": think_ndcg,
    "feel_ndcg":  feel_ndcg,
    "best_think_f1": best_think_f1,
    "best_feel_f1":  best_feel_f1,
    "train_size": len(train_idx),
    "test_size":  len(test_idx),
    "epochs": EPOCHS,
}
with open("models/training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("✅ All models saved to models/")
print(f"   two_tower_model.pt")
print(f"   think_label_encoder.pkl")
print(f"   feel_label_encoder.pkl")
print(f"   training_results.json")
print(f"   pair_scores.json         ← NEW: R/C/D % per pair")
print(f"{'='*60}")