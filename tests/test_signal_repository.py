"""Tests for SignalRepository (in-memory LRU cache)."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from predictive_maintenance_mcp.signal_repository import (
    SignalRepository,
    VALID_SIGNAL_UNITS,
    normalize_signal_unit,
)


@pytest.fixture
def repo():
    """Fresh repository with 1 MB max for testing eviction."""
    return SignalRepository(max_memory_bytes=1 * 1024 * 1024)


@pytest.fixture
def signal_file(tmp_path):
    """Create a small CSV signal file with metadata."""
    signal = np.sin(np.linspace(0, 10 * np.pi, 1000))
    csv_path = tmp_path / "test_sig.csv"
    pd.DataFrame(signal).to_csv(csv_path, index=False, header=False)

    meta = {"sampling_rate": 10000, "signal_unit": "g"}
    meta_path = tmp_path / "test_sig_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return csv_path


@pytest.fixture
def npy_file(tmp_path):
    """Create a NPY signal file."""
    signal = np.random.randn(5000).astype(np.float64)
    npy_path = tmp_path / "test_npy.npy"
    np.save(npy_path, signal)
    return npy_path


class TestLoadSignal:
    def test_load_csv(self, repo, signal_file):
        info = repo.load_signal(str(signal_file))
        assert info["signal_id"] == "test_sig"
        assert info["num_samples"] == 1000
        assert info["sampling_rate"] == 10000
        assert info["signal_unit"] == "g"

    def test_load_npy(self, repo, npy_file):
        info = repo.load_signal(str(npy_file))
        assert info["signal_id"] == "test_npy"
        assert info["num_samples"] == 5000

    def test_custom_signal_id(self, repo, signal_file):
        info = repo.load_signal(str(signal_file), signal_id="my_signal")
        assert info["signal_id"] == "my_signal"

    def test_override_sampling_rate(self, repo, signal_file):
        info = repo.load_signal(str(signal_file), sampling_rate=48000)
        assert info["sampling_rate"] == 48000

    def test_file_not_found(self, repo):
        with pytest.raises(FileNotFoundError):
            repo.load_signal("/nonexistent/path/signal.csv")

    def test_duplicate_id_replaces(self, repo, signal_file):
        repo.load_signal(str(signal_file), signal_id="dup")
        repo.load_signal(str(signal_file), signal_id="dup")
        assert repo.signal_count == 1


class TestSignalUnitDeclaration:
    """U5: signal units are DECLARED (param or metadata), never guessed."""

    def test_explicit_unit_overrides_metadata(self, repo, signal_file):
        """Precedence: explicitly declared > companion metadata ('g')."""
        info = repo.load_signal(str(signal_file), signal_unit="mm/s")
        assert info["signal_unit"] == "mm/s"

    def test_invalid_explicit_unit_raises(self, repo, signal_file):
        with pytest.raises(ValueError, match="signal_unit"):
            repo.load_signal(str(signal_file), signal_unit="furlongs")

    def test_unit_alias_normalized(self, repo, signal_file):
        """'m/s²' (superscript) normalizes to canonical 'm/s2'."""
        info = repo.load_signal(str(signal_file), signal_unit="m/s²")
        assert info["signal_unit"] == "m/s2"

    def test_unit_case_insensitive(self, repo, signal_file):
        info = repo.load_signal(str(signal_file), signal_unit="G")
        assert info["signal_unit"] == "g"

    def test_unrecognized_metadata_unit_treated_as_undeclared(self, repo, tmp_path):
        """Garbage metadata unit → None (undeclared), never coerced."""
        signal = np.random.randn(500)
        csv_path = tmp_path / "weird_unit.csv"
        pd.DataFrame(signal).to_csv(csv_path, index=False, header=False)
        with open(tmp_path / "weird_unit_metadata.json", "w") as f:
            json.dump({"sampling_rate": 1000, "signal_unit": "banana"}, f)
        info = repo.load_signal(str(csv_path))
        assert info["signal_unit"] is None

    def test_normalize_signal_unit_vocabulary(self):
        for unit in VALID_SIGNAL_UNITS:
            assert normalize_signal_unit(unit) == unit
        assert normalize_signal_unit(None) is None
        assert normalize_signal_unit("nonsense") is None
        assert normalize_signal_unit("mm/sec") == "mm/s"
        assert normalize_signal_unit("m/s^2") == "m/s2"


class TestGetSignal:
    def test_get_returns_array(self, repo, signal_file):
        repo.load_signal(str(signal_file), signal_id="s1")
        arr = repo.get_signal("s1")
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 1000

    def test_get_missing_raises(self, repo):
        with pytest.raises(KeyError, match="not found"):
            repo.get_signal("missing")


class TestListAndInfo:
    def test_list_signals(self, repo, signal_file, npy_file):
        repo.load_signal(str(signal_file), signal_id="a")
        repo.load_signal(str(npy_file), signal_id="b")
        sigs = repo.list_signals()
        assert len(sigs) == 2
        ids = {s["signal_id"] for s in sigs}
        assert ids == {"a", "b"}

    def test_get_info(self, repo, signal_file):
        repo.load_signal(str(signal_file), signal_id="info_test")
        info = repo.get_signal_info("info_test")
        assert info["signal_id"] == "info_test"
        assert info["num_samples"] == 1000

    def test_info_missing_raises(self, repo):
        with pytest.raises(KeyError):
            repo.get_signal_info("nonexistent")


class TestClear:
    def test_clear_signal(self, repo, signal_file):
        repo.load_signal(str(signal_file), signal_id="to_clear")
        assert repo.clear_signal("to_clear") is True
        assert repo.signal_count == 0

    def test_clear_missing(self, repo):
        assert repo.clear_signal("nope") is False

    def test_clear_all(self, repo, signal_file, npy_file):
        repo.load_signal(str(signal_file), signal_id="a")
        repo.load_signal(str(npy_file), signal_id="b")
        count = repo.clear_all()
        assert count == 2
        assert repo.signal_count == 0


class TestEdgeCases:
    def test_load_mat_direct(self, repo, tmp_path):
        """Test loading a .mat file via direct fallback loader."""
        from scipy.io import savemat
        data = np.random.randn(500)
        mat_path = tmp_path / "test_mat.mat"
        savemat(str(mat_path), {"vibration": data})
        info = repo.load_signal(str(mat_path))
        assert info["num_samples"] == 500

    def test_no_metadata_file(self, repo, tmp_path):
        """Load signal without companion metadata JSON."""
        npy_path = tmp_path / "no_meta.npy"
        np.save(npy_path, np.random.randn(100))
        info = repo.load_signal(str(npy_path))
        assert info["sampling_rate"] is None
        assert info["signal_unit"] is None

    def test_memory_tracking(self, repo, tmp_path):
        """current_memory_bytes tracks correctly."""
        path = tmp_path / "mem_test.npy"
        data = np.random.randn(1000)
        np.save(path, data)
        repo.load_signal(str(path), signal_id="m1")
        assert repo.current_memory_bytes == data.nbytes
        repo.clear_signal("m1")
        assert repo.current_memory_bytes == 0

    def test_duration_calculated(self, repo, tmp_path):
        """Duration should be calculated when sampling_rate known."""
        path = tmp_path / "dur.npy"
        np.save(path, np.random.randn(10000))
        info = repo.load_signal(str(path), sampling_rate=10000)
        assert info["duration_s"] == pytest.approx(1.0, abs=0.01)


class TestLRUEviction:
    def test_eviction_when_full(self, tmp_path):
        """With a tiny max (8 KB), loading large signals should evict oldest."""
        repo = SignalRepository(max_memory_bytes=8 * 1024)

        # Each signal ~8 KB (1000 float64 = 8000 bytes)
        for i in range(3):
            path = tmp_path / f"sig_{i}.npy"
            np.save(path, np.random.randn(1000))
            repo.load_signal(str(path), signal_id=f"s{i}")

        # Should have evicted s0 (oldest)
        assert repo.signal_count <= 2
        # Most recent should still be there
        assert repo.get_signal("s2") is not None

    def test_lru_access_updates_order(self, tmp_path):
        """Accessing a signal should move it to end of eviction queue."""
        # Use enough memory for 3 signals (each ~8KB), then loading 4th triggers eviction
        repo = SignalRepository(max_memory_bytes=28 * 1024)

        for i in range(3):
            path = tmp_path / f"sig_{i}.npy"
            np.save(path, np.random.randn(1000))
            repo.load_signal(str(path), signal_id=f"s{i}")

        assert repo.signal_count == 3

        # Access s0 to make it recently used
        repo.get_signal("s0")

        # Load another signal to trigger eviction
        path = tmp_path / "sig_3.npy"
        np.save(path, np.random.randn(1000))
        repo.load_signal(str(path), signal_id="s3")

        # s0 should survive because it was recently accessed
        # s1 (oldest untouched) should be evicted
        ids = {s["signal_id"] for s in repo.list_signals()}
        assert "s0" in ids, "Recently accessed signal should not be evicted"
        assert "s1" not in ids, "Oldest untouched signal should be evicted"
