---
name: anomaly-detection
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
Supports OneClassSVM and LocalOutlierFactor novelty detection with PCA
visualization. The models flag statistical outliers for the engineer to
investigate — they supplement, and never replace, physics-based diagnosis.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Workflow

### Training a New Model

#### Step 1 — Prepare the Healthy Baseline

Gather signals representing NORMAL machine operation. Check what is loaded
with `list_signals(scope="memory")`; browse files with
`list_signals(scope="disk")`. Load the training set in one atomic batch:

Call `load_signal(filepath=["real_train/baseline_1.csv", "real_train/baseline_2.csv"], signal_unit="g")`

The batch is fail-fast: if any file is missing or an id collides, ONE error
names the offending entries and nothing is loaded. For raw binary files
(`.bin`/`.raw`/`.dat`), also declare `sample_format` and `sampling_rate`
(broadcast to the whole batch), or provide companion `<stem>_metadata.json`
files — see the signal-management skill.

Ask the user:
- Which signals represent normal/healthy operation?
- Which algorithm? (`"OneClassSVM"` recommended for small datasets,
  `"LocalOutlierFactor"` for larger ones)

#### Step 2 — Screen the Baseline Quality

For each baseline signal, call `analyze_statistics(signal_id="<id>")`.
Signals with very high kurtosis or crest factor may already contain damage
and should be excluded from the healthy baseline — confirm with the user.

#### Step 3 — Train the Model

Call `train_anomaly_model(healthy_signal_ids=["real_train_baseline_1", "real_train_baseline_2"], model_type="OneClassSVM", model_name="pump_a_health")`

Parameters:
- **healthy_signal_ids**: signal_ids of the healthy baseline (training data)
- **model_type**: `"OneClassSVM"` or `"LocalOutlierFactor"`
- **model_name**: name for the saved model files (echoed in the result;
  pass the same name to predict_anomalies later)
- **fault_signal_ids** (optional): known-fault signal_ids used to tune the
  hyperparameters (semi-supervised mode), never to fit the model. The
  reported fault detection rate comes from the same signals, so it is not a
  held-out figure
- **healthy_validation_ids** (optional): held-out healthy signal_ids for the
  specificity check. Without them, the check runs only when
  fault_signal_ids are given, on the last 20% of the training windows
- **segment_duration** / **overlap_ratio**: feature-extraction segmentation
  (defaults 0.1 s / 0.5)

The model, scaler, and PCA transformer are saved to the models/ directory.

#### Step 4 — Validate

Run `predict_anomalies(signal_id="real_train_baseline_1", model_name="pump_a_health")`
on a healthy signal. Most segments should be classified normal (low anomaly
ratio). If too many are flagged, review the baseline data with the user.

### Predicting Anomalies on New Signals

#### Step 1 — Load the Target Signal

Call `load_signal(filepath="<file>", signal_unit="g")`.

#### Step 2 — Run Prediction

Call `predict_anomalies(signal_id="<id>", model_name="pump_a_health")`.

The output is bounded (no per-segment arrays): segment counts, anomaly
ratio, score percentiles, and up to 10 worst segments with their time
positions. If the model name is wrong, the error lists the models actually
on disk.

#### Step 3 — Interpret Results

| Anomaly ratio | Classification | Action |
|---|---|---|
| Near 0 | Normal | No action needed |
| Moderate | Borderline | Monitor more frequently |
| High | Anomalous | Investigate with bearing-diagnosis or quick-screening |

An anomaly score is a statistical distance from the healthy baseline — it
says "different", not "why". Always follow up with physics-based analysis
(envelope, characteristic frequencies) before any maintenance decision.

#### Step 4 — Visualize with PCA

Call `generate_pca_visualization_report(model_name="pump_a_health", test_signal_ids=["<id>"], true_labels={"<id>": "<label>"})`

to project the test segments into the model's PCA space. Healthy segments
cluster together; anomalies appear as outliers. The report segments each
signal with the model's training `segment_duration` and `overlap_ratio`:
pass them if they are not the defaults. `true_labels` is optional
annotation for the legend; take a label only from the user, never from a
file name or a signal id. The report accepts healthy, normal or baseline for
a healthy signal and faulty, fault, anomaly or anomalous for a faulty one.

## Important Notes

- Training requires signals from the SAME machine under normal conditions —
  mixing machines produces unreliable models
- A model trained at one sampling rate applies to signals at that rate
- Retraining is needed when operating conditions change significantly
- The anomaly model supplements, but does NOT replace, physics-based
  diagnosis and expert judgment
- All ML processing happens locally — no data leaves the machine
