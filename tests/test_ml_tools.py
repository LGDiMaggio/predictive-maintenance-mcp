"""
Tests for the machine-learning tools, through the production code.

Tests:
- Feature extraction (``extract_time_domain_features``) on known signals
- Segmentation (``plan_segments`` / ``segment_features``)
- Training and prediction (``train_anomaly_model`` / ``predict_anomalies``)
- Model persistence (the saved artefacts reproduce the prediction)
- Health labels (``health_label``)

Every random signal is seeded. Training and prediction run in a sandbox:
signals and models live under a temporary directory and every by-value import
of the model directory is patched, so nothing is read from or written to the
checkout.
"""

import asyncio
import json
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from conftest import patch_data_dir
from predictive_maintenance_mcp.decision_support import diagnosis_pipeline
from predictive_maintenance_mcp.decision_support.anomaly_labels import health_label
from predictive_maintenance_mcp.mcp_tools import diagnostics_tools, report_tools
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from predictive_maintenance_mcp.signal_processing.features import (
    extract_time_domain_features,
    plan_segments,
    segment_features,
)

FS = 10_000.0


def tones(seed: int, num_samples: int) -> np.ndarray:
    """Three tones with seeded phases plus seeded Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(num_samples) / FS
    phases = rng.uniform(0, 2 * np.pi, 3)
    return (
        0.5 * np.sin(2 * np.pi * 25 * t + phases[0])
        + 0.2 * np.sin(2 * np.pi * 50 * t + phases[1])
        + 0.1 * np.sin(2 * np.pi * 1200 * t + phases[2])
        + 0.3 * rng.standard_normal(num_samples)
    )


class TestFeatureExtraction:
    """Tests for extract_time_domain_features and the segmentation."""

    def test_feature_extraction_synthetic(self):
        """Seventeen finite features from a synthetic signal."""
        t = np.arange(int(FS)) / FS
        signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.default_rng(
            0
        ).standard_normal(t.size)

        features = extract_time_domain_features(signal)

        assert len(features) == 17
        for name, value in features.items():
            assert isinstance(value, float), name
            assert np.isfinite(value), name

    def test_feature_rms_calculation(self):
        features = extract_time_domain_features(np.array([1.0, -1.0, 1.0, -1.0]))
        assert features["rms"] == 1.0

    def test_feature_kurtosis(self):
        """Excess kurtosis is near 0 for Gaussian noise and rises with outliers."""
        rng = np.random.default_rng(1)
        gaussian = rng.standard_normal(10_000)
        with_outliers = np.concatenate([gaussian[:9_990], np.full(10, 10.0)])

        kurt = extract_time_domain_features(gaussian)["kurtosis"]
        kurt_outliers = extract_time_domain_features(with_outliers)["kurtosis"]

        assert abs(kurt) < 0.2
        assert kurt_outliers > kurt + 1.0

    def test_feature_crest_factor(self):
        """A sine over whole periods has a crest factor of sqrt(2)."""
        signal = np.sin(2 * np.pi * np.arange(1000) / 100)
        crest = extract_time_domain_features(signal)["crest_factor"]
        assert abs(crest - np.sqrt(2)) < 1e-9

    def test_feature_segmentation(self):
        """0.2 s segments with overlap 0.5 at 1 kHz: 200 samples, 100 apart."""
        signal = np.arange(1000, dtype=float)
        plan = plan_segments(1000.0, 0.2, 0.5)

        features = segment_features(signal, plan)

        assert (plan.length, plan.hop) == (200, 100)
        assert len(features) == (1000 - 200) // 100 + 1
        for k, row in enumerate(features):
            assert row["mean"] == np.mean(signal[100 * k : 100 * k + 200])

    def test_feature_extraction_real_data(self, sample_healthy_signal):
        """Test feature extraction on real bearing data."""
        signal = sample_healthy_signal[:10000]  # Use 10k samples for speed

        features = extract_time_domain_features(signal)

        # All features should be finite
        for key, value in features.items():
            assert np.isfinite(value), f"Feature {key} is not finite: {value}"

        # RMS should be positive
        assert features["rms"] > 0, "RMS should be positive"

        # Crest factor should be > 1 for real signals
        assert (
            features["crest_factor"] > 1.0
        ), f"Crest factor should be > 1, got {features['crest_factor']}"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Signals and models under a temporary directory."""
    signals, models = tmp_path / "signals", tmp_path / "models"
    signals.mkdir()
    models.mkdir()
    patch_data_dir(monkeypatch, signals)
    for module in (diagnostics_tools, diagnosis_pipeline, report_tools):
        monkeypatch.setattr(module, "MODELS_DIR", models)
    repo = get_repository()
    loaded: list[str] = []

    def load(signal_id: str, values: np.ndarray) -> str:
        path = signals / f"{signal_id}.csv"
        pd.DataFrame(values).to_csv(path, index=False, header=False)
        repo.load_signal(
            path.name,
            signal_id=signal_id,
            sampling_rate=FS,
            signal_unit="g",
            overwrite=True,
        )
        loaded.append(signal_id)
        return signal_id

    yield SimpleNamespace(load=load, models=models)
    for signal_id in loaded:
        repo.clear_signal(signal_id)


def train(**kwargs):
    return asyncio.run(diagnostics_tools.train_anomaly_model(ctx=AsyncMock(), **kwargs))


def predict(signal_id: str, model_name: str):
    return asyncio.run(
        diagnostics_tools.predict_anomalies(signal_id, model_name, ctx=AsyncMock())
    )


class TestMLTraining:
    """Tests for train_anomaly_model."""

    @pytest.mark.parametrize("model_type", ["OneClassSVM", "LocalOutlierFactor"])
    def test_train_and_predict(self, sandbox, model_type):
        """A training record scores fewer flagged windows than a record with a
        large added disturbance, and the label follows the ratio."""
        training = [
            sandbox.load(f"ml_h{seed}", tones(seed, 20_000)) for seed in (1, 2, 3)
        ]
        train(healthy_signal_ids=training, model_type=model_type, model_name="ml_model")
        healthy = training[0]
        disturbed = sandbox.load(
            "ml_disturbed",
            tones(51, 20_000) + 5.0 * np.random.default_rng(52).standard_normal(20_000),
        )

        healthy_result = predict(healthy, "ml_model")
        disturbed_result = predict(disturbed, "ml_model")

        for result in (healthy_result, disturbed_result):
            assert result.num_segments == (20_000 - 1_000) // 500 + 1
            assert result.anomaly_ratio == result.anomaly_count / result.num_segments
            assert result.overall_health == health_label(result.anomaly_ratio)
        assert disturbed_result.anomaly_ratio > healthy_result.anomaly_ratio

    def test_pca_dimensionality_reduction(self, sandbox):
        """PCA keeps at least the requested share of the variance."""
        training = [sandbox.load(f"ml_p{seed}", tones(seed, 20_000)) for seed in (4, 5)]
        result = train(healthy_signal_ids=training, model_name="ml_pca")
        assert result.num_features_original == 17
        assert result.num_features_pca < 17
        assert result.variance_explained >= 0.95

    def test_model_persistence(self, sandbox):
        """The saved artefacts, loaded back, reproduce predict_anomalies."""
        training = [sandbox.load(f"ml_s{seed}", tones(seed, 20_000)) for seed in (6, 7)]
        train(healthy_signal_ids=training, model_name="ml_saved")
        probe_values = tones(60, 20_000)
        probe = sandbox.load("ml_probe", probe_values)
        parts = {}
        for part in ("model", "scaler", "pca"):
            with open(sandbox.models / f"ml_saved_{part}.pkl", "rb") as handle:
                parts[part] = pickle.load(handle)  # a model this test just trained
        metadata = json.loads(
            (sandbox.models / "ml_saved_metadata.json").read_text(encoding="utf-8")
        )
        plan = plan_segments(
            metadata["sampling_rate"],
            metadata["segment_duration"],
            metadata["overlap_ratio"],
        )
        rows = pd.DataFrame(segment_features(probe_values, plan)).values
        flags = parts["model"].predict(
            parts["pca"].transform(parts["scaler"].transform(rows))
        )

        result = predict(probe, "ml_saved")

        assert result.num_segments == len(flags)
        assert result.anomaly_count == int(np.sum(flags == -1))

    def test_training_validation_split(
        self, sample_healthy_signal, sample_faulty_signal
    ):
        """The production features of both recordings are finite and share
        their names."""
        healthy = extract_time_domain_features(sample_healthy_signal[:50000])
        faulty = extract_time_domain_features(sample_faulty_signal[:50000])

        assert healthy.keys() == faulty.keys()
        assert all(np.isfinite(v) for v in [*healthy.values(), *faulty.values()])


class TestMLPrediction:
    """Tests for the health label predict_anomalies and the diagnosis share."""

    def test_health_status_classification(self):
        test_cases = [
            (0.05, "Healthy"),  # 5% anomalies
            (0.15, "Suspicious"),  # 15% anomalies
            (0.40, "Faulty"),  # 40% anomalies
        ]

        for anomaly_ratio, expected_status in test_cases:
            status = health_label(anomaly_ratio)
            assert (
                status == expected_status
            ), f"Ratio {anomaly_ratio}: expected {expected_status}, got {status}"
