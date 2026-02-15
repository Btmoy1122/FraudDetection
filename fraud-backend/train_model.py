import json
import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve

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

# Default threshold = 0.5 for first pass
preds = (probs >= 0.5).astype(int)
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
    "threshold": 0.5,
    "algorithm": "logistic_regression"
}
meta_path.write_text(json.dumps(metadata, indent=2))

print(f"Saved model to: {model_path}")
print(f"Saved metadata to: {meta_path}")