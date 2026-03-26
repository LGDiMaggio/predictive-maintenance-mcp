"""Tests for time-domain feature extraction, segmentation, and sampling rate resolution.

Covers:
- extract_time_domain_features: 17 statistical/shape features
- segment_and_extract_features: windowed segmentation with overlap
- resolve_sampling_rate: metadata fallback and strict/lenient modes
"""

import json

import numpy as np
import pytest

from predictive_maintenance_mcp.signal_processing.features import (
    extract_time_domain_features,
    resolve_sampling_rate,
    segment_and_extract_features,
)

EXPECTED_FEATURE_KEYS = {
    "mean",
    "std",
    "var",
    "mean_abs",
    "rms",
    "max",
    "min",
    "range",
    "skewness",
    "kurtosis",
    "shape_factor",
    "crest_factor",
    "impulse_factor",
    "clearance_factor",
    "power",
    "entropy",
    "zero_crossing_rate",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sine_signal():
    """Unit-amplitude 50 Hz sine wave, 10 kHz sampling, 1 second."""
    fs = 10_000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * t)
    return signal, fs


@pytest.fixture
def amplitude_sine():
    """Amplitude-5 sine wave for RMS verification."""
    fs = 10_000
    amplitude = 5.0
    t = np.linspace(0, 1.0, fs, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * 50 * t)
    return signal, amplitude


@pytest.fixture
def zero_signal():
    """All-zero signal (1000 samples)."""
    return np.zeros(1000)


@pytest.fixture
def constant_signal():
    """Constant-value signal (value = 3.0, 1000 samples)."""
    return np.full(1000, 3.0)


@pytest.fixture
def impulse_signal():
    """Single impulse at the centre of 1000 samples."""
    sig = np.zeros(1000)
    sig[500] = 100.0
    return sig


@pytest.fixture
def gaussian_noise():
    """Zero-mean unit-variance Gaussian noise (100 000 samples, fixed seed)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(100_000)


# ===================================================================
# extract_time_domain_features
# ===================================================================


class TestExtractTimeDomainFeatures:
    """Tests for extract_time_domain_features."""

    def test_returns_17_features(self, sine_signal):
        """All 17 expected feature keys are present."""
        signal, _ = sine_signal
        features = extract_time_domain_features(signal)
        assert set(features.keys()) == EXPECTED_FEATURE_KEYS
        assert len(features) == 17

    def test_all_values_are_float(self, sine_signal):
        """Every feature value must be a Python float."""
        signal, _ = sine_signal
        features = extract_time_domain_features(signal)
        for key, value in features.items():
            assert isinstance(value, float), f"{key} is {type(value)}, expected float"

    def test_sine_wave_rms(self, amplitude_sine):
        """RMS of A*sin(t) should be A / sqrt(2)."""
        signal, amplitude = amplitude_sine
        features = extract_time_domain_features(signal)
        expected_rms = amplitude / np.sqrt(2)
        assert features["rms"] == pytest.approx(expected_rms, rel=1e-3)

    def test_sine_wave_mean(self, sine_signal):
        """Mean of a full-period sine wave should be approximately zero."""
        signal, _ = sine_signal
        features = extract_time_domain_features(signal)
        assert features["mean"] == pytest.approx(0.0, abs=1e-6)

    def test_zero_signal(self, zero_signal):
        """All-zero signal must not raise and must produce values."""
        features = extract_time_domain_features(zero_signal)
        assert len(features) == 17
        # skewness and kurtosis may be NaN for zero signal (scipy behavior)
        nan_ok = {"skewness", "kurtosis"}
        for key, value in features.items():
            if key not in nan_ok:
                assert np.isfinite(value), f"{key} is not finite for zero signal"

    def test_zero_signal_specific_values(self, zero_signal):
        """Spot-check specific values on a zero signal."""
        features = extract_time_domain_features(zero_signal)
        assert features["mean"] == pytest.approx(0.0, abs=1e-12)
        assert features["std"] == pytest.approx(0.0, abs=1e-12)
        assert features["range"] == pytest.approx(0.0, abs=1e-12)
        assert features["power"] == pytest.approx(0.0, abs=1e-12)

    def test_constant_signal(self, constant_signal):
        """Constant signal: std should be ~0, kurtosis is edge case."""
        features = extract_time_domain_features(constant_signal)
        assert features["std"] == pytest.approx(0.0, abs=1e-12)
        assert features["var"] == pytest.approx(0.0, abs=1e-12)
        assert features["mean"] == pytest.approx(3.0, abs=1e-12)
        assert features["rms"] == pytest.approx(3.0, rel=1e-6)

    def test_impulse_signal_high_crest_factor(self, impulse_signal):
        """Impulse signal should have a high crest factor."""
        features = extract_time_domain_features(impulse_signal)
        # Crest factor = peak / rms.  For a single-spike signal this is large.
        assert features["crest_factor"] > 10.0

    def test_impulse_signal_high_kurtosis(self, impulse_signal):
        """Impulse signal should have high kurtosis (leptokurtic)."""
        features = extract_time_domain_features(impulse_signal)
        assert features["kurtosis"] > 50.0

    def test_gaussian_noise_kurtosis(self, gaussian_noise):
        """Fisher kurtosis of Gaussian noise should be near 0."""
        features = extract_time_domain_features(gaussian_noise)
        assert features["kurtosis"] == pytest.approx(0.0, abs=0.15)

    def test_known_range(self, sine_signal):
        """Range must equal max - min."""
        signal, _ = sine_signal
        features = extract_time_domain_features(signal)
        assert features["range"] == pytest.approx(
            features["max"] - features["min"], abs=1e-12
        )

    def test_positive_power(self, sine_signal):
        """Power (mean of x^2) must be non-negative."""
        signal, _ = sine_signal
        features = extract_time_domain_features(signal)
        assert features["power"] >= 0.0

    def test_positive_power_gaussian(self, gaussian_noise):
        """Power must also be positive for random noise."""
        features = extract_time_domain_features(gaussian_noise)
        assert features["power"] > 0.0

    def test_zero_crossing_rate_sine(self, sine_signal):
        """A 50 Hz sine in 10 kHz has ~100 zero crossings in 10 000 samples."""
        signal, fs = sine_signal
        features = extract_time_domain_features(signal)
        # 50 Hz sine crosses zero 100 times per second, rate = 100/10000 = 0.01
        assert features["zero_crossing_rate"] == pytest.approx(0.01, abs=0.002)

    def test_entropy_uniform_vs_peaked(self, gaussian_noise, impulse_signal):
        """Gaussian noise (spread) should have higher entropy than an impulse."""
        ent_noise = extract_time_domain_features(gaussian_noise)["entropy"]
        ent_impulse = extract_time_domain_features(impulse_signal)["entropy"]
        assert ent_noise > ent_impulse


# ===================================================================
# segment_and_extract_features
# ===================================================================


class TestSegmentAndExtractFeatures:
    """Tests for segment_and_extract_features."""

    def test_correct_segment_count_no_overlap(self):
        """1 s signal, 0.1 s segments, 0 overlap -> 10 segments."""
        fs = 1000
        signal = np.random.default_rng(0).standard_normal(fs)  # 1 second
        result = segment_and_extract_features(signal, fs, 0.1, 0.0)
        assert len(result) == 10

    def test_no_overlap_segments(self):
        """With 0 overlap, segments should tile without gaps."""
        fs = 1000
        signal = np.arange(fs, dtype=float)
        result = segment_and_extract_features(signal, fs, 0.25, 0.0)
        # 1 s / 0.25 s = 4 segments
        assert len(result) == 4

    def test_half_overlap_doubles_segments(self):
        """50 % overlap should roughly double the segment count."""
        fs = 1000
        signal = np.random.default_rng(1).standard_normal(fs)
        no_overlap = segment_and_extract_features(signal, fs, 0.1, 0.0)
        half_overlap = segment_and_extract_features(signal, fs, 0.1, 0.5)
        # With 50 % overlap, expect ~19 segments (vs 10 without).
        assert len(half_overlap) >= len(no_overlap) * 1.8

    def test_short_signal_returns_empty(self):
        """Signal shorter than one segment yields 0 segments."""
        fs = 1000
        signal = np.zeros(50)  # 50 ms at 1 kHz
        result = segment_and_extract_features(signal, fs, 0.1, 0.0)
        assert result == []

    def test_single_segment(self):
        """Signal exactly one segment long -> 1 segment."""
        fs = 1000
        segment_dur = 0.5
        signal = np.ones(int(fs * segment_dur))
        result = segment_and_extract_features(signal, fs, segment_dur, 0.0)
        assert len(result) == 1

    def test_each_segment_has_17_features(self):
        """Every segment dict must contain exactly 17 features."""
        fs = 1000
        signal = np.random.default_rng(2).standard_normal(fs)
        result = segment_and_extract_features(signal, fs, 0.2, 0.0)
        for seg_features in result:
            assert set(seg_features.keys()) == EXPECTED_FEATURE_KEYS


# ===================================================================
# resolve_sampling_rate
# ===================================================================


class TestResolveSamplingRate:
    """Tests for resolve_sampling_rate."""

    def test_provided_rate_used(self):
        """When sampling_rate is given, return it directly."""
        resolver = lambda f: None  # should never be called
        result = resolve_sampling_rate("any_file.csv", 48000.0, metadata_resolver=resolver)
        assert result == 48000.0

    def test_metadata_fallback(self, tmp_path):
        """When no rate is provided, read from the JSON metadata file."""
        meta_file = tmp_path / "signal_metadata.json"
        meta_file.write_text(json.dumps({"sampling_rate": 12000}))

        resolver = lambda _f: meta_file
        result = resolve_sampling_rate("signal.csv", None, metadata_resolver=resolver)
        assert result == 12000

    def test_strict_missing_raises_valueerror(self, tmp_path):
        """strict=True with no rate and no metadata should raise ValueError."""
        empty_meta = tmp_path / "empty_metadata.json"
        empty_meta.write_text(json.dumps({"description": "no rate here"}))

        resolver = lambda _f: empty_meta
        with pytest.raises(ValueError, match="No sampling rate found"):
            resolve_sampling_rate("signal.csv", None, strict=True, metadata_resolver=resolver)

    def test_strict_missing_file_raises_valueerror(self, tmp_path):
        """strict=True with nonexistent metadata file should raise ValueError."""
        nonexistent = tmp_path / "nonexistent.json"
        resolver = lambda _f: nonexistent
        with pytest.raises(ValueError, match="No sampling rate found"):
            resolve_sampling_rate("signal.csv", None, strict=True, metadata_resolver=resolver)

    def test_lenient_missing_returns_none(self, tmp_path):
        """strict=False with no rate available should return None."""
        nonexistent = tmp_path / "nonexistent.json"
        resolver = lambda _f: nonexistent
        result = resolve_sampling_rate(
            "signal.csv", None, strict=False, metadata_resolver=resolver
        )
        assert result is None

    def test_provided_overrides_metadata(self, tmp_path):
        """Explicitly provided rate takes priority over metadata file."""
        meta_file = tmp_path / "signal_metadata.json"
        meta_file.write_text(json.dumps({"sampling_rate": 12000}))

        resolver = lambda _f: meta_file
        result = resolve_sampling_rate("signal.csv", 44100.0, metadata_resolver=resolver)
        assert result == 44100.0

    def test_custom_metadata_resolver(self, tmp_path):
        """A custom resolver can point to an arbitrary metadata file."""
        custom_meta = tmp_path / "custom_meta.json"
        custom_meta.write_text(json.dumps({"sampling_rate": 96000}))

        resolver = lambda _f: custom_meta
        result = resolve_sampling_rate("irrelevant.csv", None, metadata_resolver=resolver)
        assert result == 96000
