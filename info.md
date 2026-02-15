# Phase 2 Cheat Sheet (ML Integration)

## What we built
- Trained a fraud classifier with scikit-learn (`LogisticRegression`).
- Split data into 80% train / 20% test to evaluate on unseen data.
- Tuned decision threshold instead of blindly using `0.5`.
- Saved artifacts for serving:
  - `artifacts/model_v1.pkl`
  - `artifacts/model_v1_meta.json`
- Loaded model in FastAPI and added ML scoring endpoint.

## Core concepts to explain
- **Model**: learned function mapping features -> fraud probability.
- **Training**: `fit(X_train, y_train)` learns weights from labeled examples.
- **Testing**: evaluate on held-out test data to estimate generalization.
- **Threshold**: converts probability to decision (`DENY` vs `APPROVE`).
- **Offline vs Online**:
  - Offline = train/tune model
  - Online = use saved model for real-time scoring in API

## Why logistic regression
- Fast baseline, easy to interpret, strong for tabular data.
- Outputs probabilities (`predict_proba`) needed for thresholding.
- Good first production-style model before trying complex models.

## Why artifacts matter
- `.pkl`: serialized trained model (weights + config needed for inference).
- `.json`: metadata (feature order, threshold, version).
- Keeps inference reproducible and versioned.

## Important practical rules
- Training and inference must use same feature schema/order.
- Never commit large dataset/model binaries to GitHub (use ignore rules).
- Keep threshold in metadata, not hardcoded in endpoint logic.

## Interview one-liner
"I trained a logistic regression baseline offline on labeled fraud data, tuned threshold on held-out evaluation data, versioned model artifacts, and served low-latency online inference in FastAPI with explicit feature-contract and threshold control."