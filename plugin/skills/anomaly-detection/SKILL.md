---
description: >
  ML-based anomaly detection and model training for vibration signals using the
  predictive-maintenance-mcp server. Use this skill when the user says "anomaly
  detection", "train model", "detect anomalies", "outlier detection", "normal vs
  abnormal", "machine learning", "one-class SVM", "LOF", "local outlier factor",
  "train anomaly model", "predict anomalies", "PCA visualization", "clustering",
  or wants to build or use anomaly detection models on vibration data.
---

# Anomaly Detection

Train and use machine learning models to detect anomalous vibration patterns.
Supports OneClassSVM and Local Outlier Factor (LOF) algorithms with PCA
visualization for clustering analysis.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Workflow

### Training a New Model

#### Step 1 — Prepare Normal Baseline Data

Gather a set of signals representing NORMAL machine operation. Call
`list_stored_signals()` to check available signals, or load signals with
`load_signal(...)`.

Ask the user:
- Which signals represent normal/healthy operation?
- What model algorithm to use? (OneClassSVM recommended for small datasets, LOF
  for larger ones)

#### Step 2 — Extract Features

For each baseline signal, call `extract_features_from_signal(signal_id=...)` to
verify data quality. Signals with very high kurtosis or ISO Zone C/D should be
excluded from the normal baseline.

#### Step 3 — Train the Model

Call `train_anomaly_model(signal_ids=[...], model_type="oneclass_svm")`.

Parameters:
- **signal_ids**: list of signal IDs for normal baseline
- **model_type**: `"oneclass_svm"` or `"lof"` (Local Outlier Factor)
- **contamination**: expected fraction of anomalies in training data (default
  0.05, lower = stricter)

The model is saved to the models/ directory for future use.

#### Step 4 — Validate

Run `predict_anomalies(signal_ids=[...])` on the training signals themselves.
Most should be classified as normal (anomaly_score near 0). If too many are
flagged, increase contamination or review the baseline data.

### Predicting Anomalies on New Signals

#### Step 1 — Load Target Signals

Load the signals to analyze with `load_signal(...)`.

#### Step 2 — Run Prediction

Call `predict_anomalies(signal_ids=[...])`.

The tool returns for each signal:
- **anomaly_score**: distance from normal (higher = more anomalous)
- **is_anomaly**: boolean classification
- **confidence**: model confidence

#### Step 3 — Interpret Results

| Anomaly Score | Classification | Action |
|---|---|---|
| Low (near 0) | Normal | No action needed |
| Medium | Borderline | Monitor more frequently |
| High | Anomaly | Investigate with bearing-diagnosis or quick-screening |

#### Step 4 — Visualize with PCA

Call `generate_pca_visualization_report(signal_ids=[...])` to project all
signals into 2D space. Normal signals cluster together; anomalies appear as
outliers.

## Important Notes

- Training requires multiple signals from the SAME machine under normal
  conditions — mixing machines produces unreliable models
- Models are deterministic when random seeds are fixed
- Retraining is needed when machine operating conditions change significantly
- The anomaly model supplements, but does NOT replace, physics-based diagnosis
- All ML processing happens locally — no data leaves the machine
