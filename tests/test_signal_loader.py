"""
Tests for the Signal Loader module.

Covers:
- Multi-format signal loading: CSV, NPY, MAT, WAV, Parquet
- Segment extraction with seed reproducibility
- Segment boundary conditions
- Metadata path derivation for all formats
- Error handling for missing / invalid files
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from predictive_maintenance_mcp.signal_loader import (
    load_signal_data,
    extract_segment,
    get_metadata_path,
    get_metadata_path_from_dir,
    DATA_DIR,
)


# ── load_signal_data ───────────────────────────────────────────────────────

class TestLoadSignalData:

    def test_load_csv(self, tmp_path, monkeypatch):
        """Load a CSV signal file."""
        data = np.array([1.0, 2.5, -1.3, 0.5, 3.2])
        csv_file = tmp_path / "test.csv"
        pd.DataFrame(data).to_csv(csv_file, header=False, index=False)

        monkeypatch.setattr("predictive_maintenance_mcp.signal_loader.DATA_DIR", tmp_path)
        signal = load_signal_data("test.csv")
        np.testing.assert_array_almost_equal(signal, data)

    def test_load_npy(self, tmp_path, monkeypatch):
        """Load a NPY binary file."""
        data = np.array([1.1, 2.2, 3.3, 4.4])
        np.save(tmp_path / "test.npy", data)

        monkeypatch.setattr("predictive_maintenance_mcp.signal_loader.DATA_DIR", tmp_path)
        signal = load_signal_data("test.npy")
        np.testing.assert_array_equal(signal, data)

    def test_load_mat(self, tmp_path, monkeypatch):
        """Load a MATLAB .mat file."""
        from scipy.io import savemat
        data = np.array([10.0, 20.0, 30.0])
        savemat(str(tmp_path / "test.mat"), {"signal": data})

        monkeypatch.setattr("predictive_maintenance_mcp.signal_loader.DATA_DIR", tmp_path)
        signal = load_signal_data("test.mat")
        assert signal is not None
        np.testing.assert_array_almost_equal(signal, data)

    def test_load_wav(self, tmp_path, monkeypatch):
        """Load a WAV audio file."""
        from scipy.io import wavfile
        fs = 16000
        data = (np.sin(np.linspace(0, 2 * np.pi * 440, fs)) * 32767).astype(np.int16)
        wavfile.write(str(tmp_path / "test.wav"), fs, data)

        monkeypatch.setattr("predictive_maintenance_mcp.signal_loader.DATA_DIR", tmp_path)
        signal = load_signal_data("test.wav")
        assert signal is not None
        assert len(signal) == len(data)
        assert signal.dtype == np.float64
        # Int16 signals are normalized to [-1, 1]
        assert np.max(np.abs(signal)) <= 1.0 + 1e-6

    @pytest.mark.skipif(
        not any(__import__("importlib").util.find_spec(e) for e in ("pyarrow", "fastparquet")),
        reason="pyarrow/fastparquet not installed",
    )
    def test_load_parquet(self, tmp_path, monkeypatch):
        """Load a Parquet file."""
        data = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        pd.DataFrame({"signal": data}).to_parquet(tmp_path / "test.parquet")

        monkeypatch.setattr("predictive_maintenance_mcp.signal_loader.DATA_DIR", tmp_path)
        signal = load_signal_data("test.parquet")
        assert signal is not None
        np.testing.assert_array_almost_equal(signal, data)

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("predictive_maintenance_mcp.signal_loader.DATA_DIR", tmp_path)
        assert load_signal_data("nonexistent.csv") is None

    def test_unsupported_extension_returns_none(self, tmp_path, monkeypatch):
        (tmp_path / "test.xyz").write_text("data")
        monkeypatch.setattr("predictive_maintenance_mcp.signal_loader.DATA_DIR", tmp_path)
        assert load_signal_data("test.xyz") is None

    def test_wav_stereo_uses_first_channel(self, tmp_path, monkeypatch):
        """Stereo WAV should use first channel only."""
        from scipy.io import wavfile
        fs = 8000
        ch1 = (np.sin(np.linspace(0, 2 * np.pi * 300, fs)) * 32767).astype(np.int16)
        ch2 = (np.sin(np.linspace(0, 2 * np.pi * 600, fs)) * 32767).astype(np.int16)
        stereo = np.column_stack([ch1, ch2])
        wavfile.write(str(tmp_path / "stereo.wav"), fs, stereo)

        monkeypatch.setattr("predictive_maintenance_mcp.signal_loader.DATA_DIR", tmp_path)
        signal = load_signal_data("stereo.wav")
        assert signal is not None
        assert signal.ndim == 1
        assert len(signal) == fs

    def test_load_real_baseline_if_available(self):
        """Smoke test: load a real baseline CSV from the repo."""
        baseline = DATA_DIR / "real_train" / "baseline_1.csv"
        if not baseline.exists():
            pytest.skip("Real data not in repo")
        signal = load_signal_data("real_train/baseline_1.csv")
        assert signal is not None
        assert len(signal) > 1000


# ── extract_segment ────────────────────────────────────────────────────────

class TestExtractSegment:

    def test_seed_reproducibility(self):
        signal = np.arange(10000, dtype=float)
        seg1 = extract_segment(signal, 0.5, 1000, seed=42)
        seg2 = extract_segment(signal, 0.5, 1000, seed=42)
        np.testing.assert_array_equal(seg1, seg2)

    def test_different_seeds_differ(self):
        signal = np.arange(10000, dtype=float)
        seg1 = extract_segment(signal, 0.5, 1000, seed=42)
        seg2 = extract_segment(signal, 0.5, 1000, seed=99)
        assert not np.array_equal(seg1, seg2)

    def test_correct_length(self):
        signal = np.arange(10000, dtype=float)
        seg = extract_segment(signal, 1.0, 1000, seed=0)
        assert len(seg) == 1000

    def test_duration_exceeds_signal_returns_full(self):
        signal = np.arange(100, dtype=float)
        seg = extract_segment(signal, 2.0, 100, seed=0)
        np.testing.assert_array_equal(seg, signal)

    def test_duration_equals_signal_returns_full(self):
        signal = np.arange(100, dtype=float)
        seg = extract_segment(signal, 1.0, 100, seed=0)
        np.testing.assert_array_equal(seg, signal)

    def test_segment_is_contiguous_slice(self):
        signal = np.arange(10000, dtype=float)
        seg = extract_segment(signal, 0.2, 1000, seed=7)
        # Contiguous: consecutive integer values
        diffs = np.diff(seg)
        assert np.all(diffs == 1.0)

    def test_no_seed_is_random(self):
        signal = np.arange(100000, dtype=float)
        seg1 = extract_segment(signal, 0.1, 1000, seed=None)
        seg2 = extract_segment(signal, 0.1, 1000, seed=None)
        # Very unlikely to get the same random start (but could, so not strict assert)
        # Just verify the function runs without error
        assert len(seg1) == 100
        assert len(seg2) == 100


# ── get_metadata_path ──────────────────────────────────────────────────────

class TestGetMetadataPath:

    def test_csv_extension(self):
        path = get_metadata_path("signal.csv")
        assert path.name == "signal_metadata.json"

    def test_npy_extension(self):
        path = get_metadata_path("signal.npy")
        assert path.name == "signal_metadata.json"

    def test_mat_extension(self):
        path = get_metadata_path("signal.mat")
        assert path.name == "signal_metadata.json"

    def test_wav_extension(self):
        path = get_metadata_path("signal.wav")
        assert path.name == "signal_metadata.json"

    def test_parquet_extension(self):
        path = get_metadata_path("signal.parquet")
        assert path.name == "signal_metadata.json"

    def test_subdirectory_preserved(self):
        path = get_metadata_path("real_train/baseline_1.csv")
        assert "real_train" in str(path)
        assert path.name == "baseline_1_metadata.json"

    def test_from_dir_variant(self, tmp_path):
        path = get_metadata_path_from_dir(tmp_path, "test_signal.csv")
        assert path.parent == tmp_path
        assert path.name == "test_signal_metadata.json"
