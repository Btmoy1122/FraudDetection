import json
import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve

TARGET_RECALL = 0.85
DEFAULT_THRESHOLD = 0.5


def select_threshold_by_recall(y_true, probs, target_recall=TARGET_RECALL):
    """Pick threshold with best precision while meeting recall target."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)

    # precision/recall have one extra point compared to thresholds.
    candidate_thresholds = thresholds
    candidate_precisions = precisions[:-1]
    candidate_recalls = recalls[:-1]

    candidates = []
    for threshold, precision, recall in zip(
        candidate_thresholds, candidate_precisions, candidate_recalls
    ):
        if recall >= target_recall:
            candidates.append((float(threshold), float(precision), float(recall)))

    if not candidates:
        return DEFAULT_THRESHOLD, []

    # Among thresholds that satisfy recall target, maximize precision.
    best_threshold, _, _ = max(candidates, key=lambda x: x[1])
    return best_threshold, candidates

# 1) Load data
df = pd.read_csv("creditcard.csv")

# 2) Define features/label
# Option A: V1..V28 + Amount (skip Time initially for simplicity)
feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
X = df[feature_cols]
y = df["Class"]

# 3) Train/test split (stratify preserves fraud ratio in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Train model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# 5) Evaluate
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"ROC-AUC: {auc:.4f}")

# Tune threshold using held-out test set.
selected_threshold, candidate_rows = select_threshold_by_recall(
    y_test, probs, TARGET_RECALL
)
print(f"Selected threshold: {selected_threshold:.4f} (target recall={TARGET_RECALL:.2f})")

if candidate_rows:
    # Show strongest candidates to make the choice explainable.
    top_candidates = sorted(candidate_rows, key=lambda x: x[1], reverse=True)[:5]
    print("Top threshold candidates (threshold, precision, recall):")
    for threshold, precision, recall in top_candidates:
        print(f"  {threshold:.4f}, {precision:.4f}, {recall:.4f}")
else:
    print(
        "No threshold met target recall; falling back to default "
        f"threshold={DEFAULT_THRESHOLD:.2f}"
    )

preds = (probs >= selected_threshold).astype(int)
print(classification_report(y_test, preds, digits=4))

# 6) Save artifacts
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(exist_ok=True)

model_path = artifacts_dir / "model_v1.pkl"
meta_path = artifacts_dir / "model_v1_meta.json"

joblib.dump(model, model_path)

metadata = {
    "model_version": "v1",
    "feature_cols": feature_cols,
    "threshold": selected_threshold,
    "algorithm": "logistic_regression"
}
meta_path.write_text(json.dumps(metadata, indent=2))

print(f"Saved model to: {model_path}")
print(f"Saved metadata to: {meta_path}")