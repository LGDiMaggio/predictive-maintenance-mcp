"""
Tests for Pydantic models (data contracts for MCP tool outputs).

Covers:
- Model instantiation with valid data
- JSON serialization round-trip (model_dump_json → model_validate_json)
- Optional fields (None values)
- Field types and constraints
"""

import pytest
from pydantic import ValidationError

from predictive_maintenance_mcp.models import (
    SpectralPeak,
    FFTResult,
    EnvelopeResult,
    StatisticalResult,
    SignalInfo,
    ISO20816Result,
    FeatureExtractionResult,
    AnomalyModelResult,
    AnomalyPredictionResult,
)


# ── SpectralPeak ───────────────────────────────────────────────────────────

class TestSpectralPeak:

    def test_creation(self):
        p = SpectralPeak(frequency_hz=120.5, magnitude=0.015, magnitude_db=5.2, note="1x")
        assert p.frequency_hz == 120.5
        assert p.note == "1x"

    def test_default_note_empty(self):
        p = SpectralPeak(frequency_hz=50, magnitude=0.1, magnitude_db=0.0)
        assert p.note == ""

    def test_json_roundtrip(self):
        p = SpectralPeak(frequency_hz=100, magnitude=0.05, magnitude_db=-3.0, note="2x")
        restored = SpectralPeak.model_validate_json(p.model_dump_json())
        assert restored.frequency_hz == p.frequency_hz
        assert restored.note == p.note


# ── FFTResult ──────────────────────────────────────────────────────────────

class TestFFTResult:

    @pytest.fixture
    def sample_fft_result(self):
        return FFTResult(
            top_peaks=[
                SpectralPeak(frequency_hz=50, magnitude=0.1, magnitude_db=0),
                SpectralPeak(frequency_hz=100, magnitude=0.05, magnitude_db=-6),
            ],
            peak_frequency=50.0,
            peak_magnitude=0.1,
            rms_spectral=0.02,
            total_bins=5000,
            freq_range_hz=[0.0, 5000.0],
            sampling_rate=10000.0,
            num_samples=10000,
            frequency_resolution=2.0,
        )

    def test_creation(self, sample_fft_result):
        assert sample_fft_result.peak_frequency == 50.0
        assert len(sample_fft_result.top_peaks) == 2

    def test_json_roundtrip(self, sample_fft_result):
        restored = FFTResult.model_validate_json(sample_fft_result.model_dump_json())
        assert restored.peak_frequency == 50.0
        assert len(restored.top_peaks) == 2
        assert restored.top_peaks[0].frequency_hz == 50.0

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            FFTResult(
                top_peaks=[],
                peak_frequency=50.0,
                # missing peak_magnitude, rms_spectral, etc.
            )


# ── EnvelopeResult ─────────────────────────────────────────────────────────

class TestEnvelopeResult:

    def test_creation(self):
        r = EnvelopeResult(
            num_samples=10000,
            sampling_rate=10000.0,
            filter_band=(500.0, 5000.0),
            peak_frequencies=[81.13, 162.26],
            peak_magnitudes=[0.05, 0.03],
            diagnosis="Possible outer race fault (BPFO harmonics detected).",
        )
        assert r.num_samples == 10000
        assert len(r.peak_frequencies) == 2

    def test_empty_preview_defaults(self):
        r = EnvelopeResult(
            num_samples=100, sampling_rate=1000.0,
            filter_band=(100.0, 500.0),
            peak_frequencies=[], peak_magnitudes=[],
            diagnosis="No significant peaks.",
        )
        assert r.spectrum_preview_freq == []
        assert r.spectrum_preview_mag == []


# ── StatisticalResult ──────────────────────────────────────────────────────

class TestStatisticalResult:

    def test_creation(self):
        r = StatisticalResult(
            rms=5.234, peak_to_peak=18.5, peak=9.25,
            crest_factor=1.77, kurtosis=3.0, skewness=0.01,
            mean=0.001, std_dev=5.234,
            detected_unit="g", unit_note="Acceleration (g)",
        )
        assert r.rms == 5.234
        assert r.detected_unit == "g"


# ── SignalInfo ─────────────────────────────────────────────────────────────

class TestSignalInfo:

    def test_creation(self):
        s = SignalInfo(filename="test.csv", path="/data/test.csv", size_bytes=1024)
        assert s.filename == "test.csv"

    def test_optional_num_samples(self):
        s = SignalInfo(filename="test.csv", path="/data/test.csv", size_bytes=512)
        assert s.num_samples is None

        s2 = SignalInfo(filename="test.csv", path="/data/test.csv", size_bytes=512, num_samples=10000)
        assert s2.num_samples == 10000


# ── ISO20816Result ─────────────────────────────────────────────────────────

class TestISO20816Result:

    def test_all_zones(self):
        """Create an ISO result for each zone (A/B/C/D)."""
        zones = [
            ("A", "Good", "green"),
            ("B", "Acceptable", "yellow"),
            ("C", "Unsatisfactory", "orange"),
            ("D", "Unacceptable", "red"),
        ]
        for zone, severity, color in zones:
            r = ISO20816Result(
                rms_velocity=3.0, machine_group=2, support_type="rigid",
                zone=zone, zone_description=f"Zone {zone}",
                severity_level=severity, color_code=color,
                boundary_ab=1.4, boundary_bc=2.8, boundary_cd=4.5,
                frequency_range="10 Hz - 1 kHz",
            )
            assert r.zone == zone
            assert r.color_code == color

    def test_optional_speed(self):
        r = ISO20816Result(
            rms_velocity=1.0, machine_group=1, support_type="flexible",
            zone="A", zone_description="Good", severity_level="Good",
            color_code="green", boundary_ab=1.4, boundary_bc=2.8,
            boundary_cd=4.5, frequency_range="10 Hz - 1 kHz",
        )
        assert r.operating_speed_rpm is None


# ── FeatureExtractionResult ────────────────────────────────────────────────

class TestFeatureExtractionResult:

    def test_creation(self):
        r = FeatureExtractionResult(
            num_segments=50, segment_length_samples=1024,
            segment_duration_s=0.1024, overlap_ratio=0.5,
            features_shape=[50, 8],
            feature_names=["rms", "peak", "crest_factor", "kurtosis",
                           "skewness", "std", "p2p", "shape_factor"],
            features_preview=[{"rms": 5.0, "peak": 10.0}],
        )
        assert r.num_segments == 50
        assert len(r.feature_names) == 8


# ── AnomalyModelResult ────────────────────────────────────────────────────

class TestAnomalyModelResult:

    def test_creation(self):
        r = AnomalyModelResult(
            model_type="OneClassSVM",
            num_training_samples=200,
            num_features_original=8,
            num_features_pca=5,
            variance_explained=0.95,
            model_params={"kernel": "rbf", "nu": 0.05},
            model_path="/models/model.pkl",
            scaler_path="/models/scaler.pkl",
            pca_path="/models/pca.pkl",
        )
        assert r.model_type == "OneClassSVM"

    def test_optional_validation_fields(self):
        r = AnomalyModelResult(
            model_type="LocalOutlierFactor",
            num_training_samples=100,
            num_features_original=8, num_features_pca=4,
            variance_explained=0.90,
            model_params={"n_neighbors": 20},
            model_path="m.pkl", scaler_path="s.pkl", pca_path="p.pkl",
        )
        assert r.validation_accuracy is None
        assert r.validation_details is None
        assert r.validation_metrics is None


# ── AnomalyPredictionResult ───────────────────────────────────────────────

class TestAnomalyPredictionResult:

    def test_healthy(self):
        r = AnomalyPredictionResult(
            num_segments=100, anomaly_count=2, anomaly_ratio=0.02,
            predictions=[1] * 98 + [-1] * 2,
            overall_health="Healthy", confidence="High",
        )
        assert r.overall_health == "Healthy"
        assert r.anomaly_ratio == 0.02

    def test_faulty(self):
        r = AnomalyPredictionResult(
            num_segments=100, anomaly_count=90, anomaly_ratio=0.90,
            predictions=[-1] * 90 + [1] * 10,
            overall_health="Faulty", confidence="High",
        )
        assert r.overall_health == "Faulty"

    def test_optional_scores(self):
        r = AnomalyPredictionResult(
            num_segments=10, anomaly_count=0, anomaly_ratio=0.0,
            predictions=[1] * 10,
            overall_health="Healthy", confidence="High",
        )
        assert r.anomaly_scores is None

    def test_json_roundtrip(self):
        r = AnomalyPredictionResult(
            num_segments=50, anomaly_count=5, anomaly_ratio=0.1,
            predictions=[1] * 45 + [-1] * 5,
            overall_health="Suspicious", confidence="Medium",
            anomaly_scores=[0.1] * 45 + [0.9] * 5,
        )
        restored = AnomalyPredictionResult.model_validate_json(r.model_dump_json())
        assert restored.anomaly_count == 5
        assert len(restored.anomaly_scores) == 50
