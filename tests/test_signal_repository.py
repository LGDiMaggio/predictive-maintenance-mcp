"""Tests for SignalRepository (in-memory LRU cache)."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from predictive_maintenance_mcp.signal_repository import SignalRepository


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
