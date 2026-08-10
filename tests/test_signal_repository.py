"""Tests for SignalRepository (in-memory LRU cache)."""

import json
import numpy as np
import pandas as pd
import pytest

from conftest import write_raw_file
from predictive_maintenance_mcp.signal_acquisition.repository import (
    SignalRepository,
    VALID_SIGNAL_UNITS,
    get_repository,
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

    def test_zero_sampling_rate_rejected(self, repo, signal_file):
        """F10: an explicit sampling_rate of 0 is rejected before storing —
        a stored 0 would break downstream fftfreq(N, 1/rate)."""
        with pytest.raises(ValueError, match="sampling rate must be positive"):
            repo.load_signal(str(signal_file), sampling_rate=0)
        assert repo.signal_count == 0

    def test_negative_sampling_rate_rejected(self, repo, signal_file):
        """F10: a negative explicit sampling_rate is rejected before storing."""
        with pytest.raises(ValueError, match="sampling rate must be positive"):
            repo.load_signal(str(signal_file), sampling_rate=-1000)
        assert repo.signal_count == 0

    def test_file_not_found(self, repo):
        with pytest.raises(FileNotFoundError):
            repo.load_signal("/nonexistent/path/signal.csv")

    def test_duplicate_id_is_explicit_collision_error(self, repo, signal_file):
        """U8: re-loading the same id without overwrite=True raises — no
        silent replacement."""
        repo.load_signal(str(signal_file), signal_id="dup")
        with pytest.raises(ValueError, match="overwrite=True"):
            repo.load_signal(str(signal_file), signal_id="dup")
        assert repo.signal_count == 1

    def test_duplicate_id_with_overwrite_replaces(self, repo, signal_file):
        repo.load_signal(str(signal_file), signal_id="dup")
        info = repo.load_signal(str(signal_file), signal_id="dup", overwrite=True)
        assert info["signal_id"] == "dup"
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
        with pytest.raises(KeyError, match="load_signal"):
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


class TestCacheIsolation:
    """F6: returned info is a DEEP copy — mutating it never leaks into cache."""

    def test_mutating_returned_info_does_not_corrupt_cache(self, repo, signal_file):
        """Mutating nested state (shape list, source_metadata dict) on the
        returned dict must not affect a subsequent read."""
        repo.load_signal(str(signal_file), signal_id="iso1")
        info = repo.get_signal_info("iso1")
        # Mutate the mutable nested members the old shallow .copy() shared.
        info["shape"][0] = -999
        info["source_metadata"]["rpm"] = "tampered"
        fresh = repo.get_signal_info("iso1")
        assert fresh["shape"][0] == 1000
        assert "rpm" not in fresh["source_metadata"]

    def test_list_signals_entries_are_deep_copies(self, repo, signal_file):
        repo.load_signal(str(signal_file), signal_id="iso2")
        listed = repo.list_signals()[0]
        listed["shape"].append(42)
        listed["source_metadata"]["rpm"] = "tampered"
        fresh = repo.get_signal_info("iso2")
        assert fresh["shape"] == [1000]
        assert "rpm" not in fresh["source_metadata"]


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


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """A DATA_DIR with same-named files in two subfolders (audit 3.6)."""
    signals_dir = tmp_path / "data" / "signals"
    (signals_dir / "real_train").mkdir(parents=True)
    (signals_dir / "real_test").mkdir(parents=True)
    train = np.sin(np.linspace(0, 10, 1000))
    test = 2.0 * np.sin(np.linspace(0, 10, 1000))
    pd.DataFrame(train).to_csv(
        signals_dir / "real_train" / "baseline_1.csv", index=False, header=False
    )
    pd.DataFrame(test).to_csv(
        signals_dir / "real_test" / "baseline_1.csv", index=False, header=False
    )
    monkeypatch.setattr(
        "predictive_maintenance_mcp.signal_acquisition.repository.DATA_DIR",
        signals_dir,
    )
    monkeypatch.setattr(
        "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR",
        signals_dir,
    )
    return signals_dir


class TestDefaultSignalIds:
    """U8: default ids derive from the relative path — no stem collisions."""

    def test_same_stem_different_dirs_get_distinct_ids(self, repo, data_dir):
        i1 = repo.load_signal("real_train/baseline_1.csv")
        i2 = repo.load_signal("real_test/baseline_1.csv")
        assert i1["signal_id"] == "real_train_baseline_1"
        assert i2["signal_id"] == "real_test_baseline_1"
        ids = {s["signal_id"] for s in repo.list_signals()}
        assert ids == {"real_train_baseline_1", "real_test_baseline_1"}
        # The two entries hold DIFFERENT data (nothing was shadowed).
        a = repo.get_signal("real_train_baseline_1")
        b = repo.get_signal("real_test_baseline_1")
        assert not np.array_equal(a, b)

    def test_reload_same_path_collides_without_overwrite(self, repo, data_dir):
        repo.load_signal("real_train/baseline_1.csv")
        with pytest.raises(ValueError, match="overwrite=True"):
            repo.load_signal("real_train/baseline_1.csv")

    def test_absolute_path_outside_data_dir_loads_that_file(
        self, repo, data_dir, tmp_path
    ):
        """A same-named file INSIDE DATA_DIR must never silently win over
        the absolute path the caller asked for (old _load_direct fallback)."""
        outside = tmp_path / "baseline_1.csv"
        pd.DataFrame(np.full(50, 7.0)).to_csv(outside, index=False, header=False)
        # Decoy with the same name inside DATA_DIR
        pd.DataFrame(np.zeros(50)).to_csv(
            data_dir / "baseline_1.csv", index=False, header=False
        )

        info = repo.load_signal(str(outside))
        arr = repo.get_signal(info["signal_id"])
        assert len(arr) == 50
        assert float(arr[0]) == 7.0  # the outside file, not the decoy


class TestReadOnlyArrays:
    """U8: repository arrays are read-only views — tools cannot corrupt the cache."""

    def test_mutating_returned_array_raises(self, repo, signal_file):
        repo.load_signal(str(signal_file), signal_id="ro")
        arr = repo.get_signal("ro")
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 123.0

    def test_view_cannot_be_made_writeable(self, repo, signal_file):
        repo.load_signal(str(signal_file), signal_id="ro2")
        arr = repo.get_signal("ro2")
        with pytest.raises(ValueError):
            arr.setflags(write=True)

    def test_copy_is_still_usable(self, repo, signal_file):
        repo.load_signal(str(signal_file), signal_id="ro3")
        arr = repo.get_signal("ro3")
        copy = np.array(arr)
        copy[0] = 1.0  # copies are writeable as usual
        assert copy[0] == 1.0


class TestNotFoundMessage:
    """U8: the standard not-found message names the remedy and eviction."""

    def test_message_names_remedy_and_eviction(self, repo, signal_file):
        repo.load_signal(str(signal_file), signal_id="present")
        with pytest.raises(KeyError) as exc:
            repo.get_signal("ghost")
        msg = exc.value.args[0]
        assert "load_signal" in msg
        assert "list_signals" in msg
        assert "evicted" in msg
        assert "PMM_SIGNAL_CACHE_GB" in msg
        assert "present" in msg  # available ids listed

    def test_get_info_uses_same_message(self, repo):
        with pytest.raises(KeyError) as exc:
            repo.get_signal_info("ghost")
        assert "load_signal" in exc.value.args[0]


class TestBatchLoad:
    """U8: load_signals is fail-fast and atomic (all-or-nothing)."""

    @pytest.fixture
    def batch_files(self, data_dir):
        names = []
        for i in range(3):
            name = f"batch_{i}.csv"
            pd.DataFrame(np.random.randn(200)).to_csv(
                data_dir / name, index=False, header=False
            )
            names.append(name)
        return names

    def test_batch_loads_all(self, repo, data_dir, batch_files):
        infos = repo.load_signals(batch_files, sampling_rate=1000)
        assert len(infos) == 3
        assert [i["signal_id"] for i in infos] == ["batch_0", "batch_1", "batch_2"]
        assert repo.signal_count == 3
        assert all(i["sampling_rate"] == 1000 for i in infos)

    def test_batch_missing_file_loads_nothing(self, repo, data_dir, batch_files):
        files = batch_files[:1] + ["__missing__.csv"] + batch_files[1:]
        with pytest.raises(ValueError) as exc:
            repo.load_signals(files)
        msg = str(exc.value)
        assert "__missing__.csv" in msg
        assert "nothing was loaded" in msg
        assert repo.signal_count == 0  # atomic: no partial state

    def test_batch_empty_list_raises(self, repo, data_dir):
        with pytest.raises(ValueError, match="empty list"):
            repo.load_signals([])

    def test_batch_collision_with_existing_loads_nothing(
        self, repo, data_dir, batch_files
    ):
        repo.load_signal(batch_files[0])  # 'batch_0' now exists
        with pytest.raises(ValueError) as exc:
            repo.load_signals(batch_files)
        assert "batch_0" in str(exc.value)
        assert "overwrite=True" in str(exc.value)
        # Only the pre-existing signal remains — none of the batch landed.
        assert repo.signal_count == 1

    def test_batch_collision_with_overwrite_succeeds(self, repo, data_dir, batch_files):
        repo.load_signal(batch_files[0])
        infos = repo.load_signals(batch_files, overwrite=True)
        assert len(infos) == 3
        assert repo.signal_count == 3

    def test_batch_duplicate_ids_within_batch_rejected(self, repo, data_dir):
        pd.DataFrame(np.zeros(10)).to_csv(
            data_dir / "dupe.csv", index=False, header=False
        )
        with pytest.raises(ValueError, match="already taken"):
            repo.load_signals(["dupe.csv", "dupe.csv"])
        assert repo.signal_count == 0

    def test_batch_invalid_unit_rejected_upfront(self, repo, data_dir, batch_files):
        with pytest.raises(ValueError, match="signal_unit"):
            repo.load_signals(batch_files, signal_unit="furlongs")
        assert repo.signal_count == 0

    def test_batch_unreadable_file_after_valid_loads_nothing(
        self, repo, data_dir, batch_files
    ):
        """F7: a file that passes phase-1 existence but fails to READ in
        phase-2 aborts before the single locked insert runs — the store
        stays empty, so a batch is never left half-populated."""
        # Exists (so phase-1 passes) but is an unsupported format that the
        # loader cannot read (so _prepare_entry raises in phase-2).
        (data_dir / "corrupt.xyz").write_text("not a signal")
        files = [batch_files[0], "corrupt.xyz", batch_files[1]]
        with pytest.raises(ValueError):
            repo.load_signals(files, sampling_rate=1000)
        assert repo.signal_count == 0  # nothing from the batch landed


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


class TestRawBinaryLoad:
    """U2: raw decode parameters thread through the repository on both routes."""

    def test_datadir_relative_explicit_params(self, repo, data_dir):
        values = np.sin(np.linspace(0, 10, 400))
        write_raw_file(data_dir / "raw_sig.bin", values)
        info = repo.load_signal(
            "raw_sig.bin",
            sampling_rate=25600.0,
            signal_unit="g",
            sample_format="float32",
        )
        assert info["signal_id"] == "raw_sig"
        assert info["num_samples"] == 400
        assert info["sampling_rate"] == 25600.0
        assert info["signal_unit"] == "g"
        assert info["duration_s"] == pytest.approx(400 / 25600.0, abs=1e-4)
        # Provenance records the EFFECTIVE parameters (merge + defaults).
        assert info["raw_format"] == {
            "sample_format": "float32",
            "byte_order": "little",
            "n_channels": 1,
            "channel_index": 0,
            "header_offset": 0,
            "scale_factor": None,
        }
        arr = repo.get_signal("raw_sig")
        np.testing.assert_allclose(arr, values.astype(np.float32), rtol=1e-6)

    @pytest.mark.parametrize("ext", [".dat", ".raw"])
    def test_other_raw_extensions_dispatch(self, repo, data_dir, ext):
        """The historical '.dat listed but unloadable' bug stays fixed at the
        routing level: every RAW_EXTENSIONS suffix decodes, not just .bin."""
        values = np.linspace(0.0, 1.0, 64)
        name = f"other{ext}"
        write_raw_file(data_dir / name, values)
        info = repo.load_signal(name, sampling_rate=500, sample_format="float32")
        assert info["num_samples"] == 64
        assert info["raw_format"]["sample_format"] == "float32"
        np.testing.assert_allclose(
            repo.get_signal(info["signal_id"]),
            values.astype(np.float32),
            rtol=1e-6,
        )

    def test_multichannel_counts_are_per_channel(self, repo, data_dir):
        interleaved = np.empty(400, dtype=np.float32)
        interleaved[0::2] = 5.0
        interleaved[1::2] = 9.0
        write_raw_file(data_dir / "inter.bin", interleaved)
        info = repo.load_signal(
            "inter.bin",
            sampling_rate=1000,
            sample_format="float32",
            n_channels=2,
            channel_index=1,
        )
        assert info["signal_id"] == "inter_ch1"
        assert info["num_samples"] == 200  # per-channel, not the file total
        assert info["duration_s"] == pytest.approx(0.2, abs=1e-4)
        assert np.all(repo.get_signal("inter_ch1") == 9.0)

    def test_absolute_path_outside_data_dir_raw(self, repo, data_dir, tmp_path):
        """Outside-DATA_DIR route: the outside file wins, the decoy is not
        touched (raw dispatch lives in _load_array for both routes)."""
        outside = tmp_path / "baseline_raw.bin"
        write_raw_file(outside, np.full(50, 7.0))
        # Decoy with the same name inside DATA_DIR
        write_raw_file(data_dir / "baseline_raw.bin", np.zeros(50))

        info = repo.load_signal(
            str(outside), sampling_rate=1000, sample_format="float32"
        )
        assert info["signal_id"] == "baseline_raw"
        arr = repo.get_signal("baseline_raw")
        assert len(arr) == 50
        assert float(arr[0]) == 7.0  # the outside file, not the decoy
        assert info["raw_format"]["sample_format"] == "float32"

    def test_companion_metadata_supplies_all_raw_params(self, repo, data_dir):
        values = np.linspace(-1.0, 1.0, 128)
        write_raw_file(data_dir / "comp.bin", values)
        with open(data_dir / "comp_metadata.json", "w") as f:
            json.dump(
                {
                    "sampling_rate": 2000,
                    "sample_format": "float32",
                    "byte_order": "little",
                    "n_channels": 1,
                    "channel_index": 0,
                    "header_offset": 0,
                },
                f,
            )
        info = repo.load_signal("comp.bin")  # zero explicit raw params
        assert info["sampling_rate"] == 2000
        assert info["num_samples"] == 128
        assert info["raw_format"]["sample_format"] == "float32"

    def test_explicit_param_overrides_companion_field(self, repo, data_dir):
        values = np.arange(64, dtype=np.float32)
        write_raw_file(data_dir / "override.bin", values)
        with open(data_dir / "override_metadata.json", "w") as f:
            json.dump({"sampling_rate": 2000, "sample_format": "int16"}, f)
        info = repo.load_signal("override.bin", sample_format="float32")
        assert info["raw_format"]["sample_format"] == "float32"
        assert info["num_samples"] == 64  # decoded as float32, not int16
        np.testing.assert_allclose(repo.get_signal("override"), values)

    def test_partial_companion_refusal_names_only_missing(self, repo, data_dir):
        write_raw_file(data_dir / "partial.bin", np.zeros(16))
        with open(data_dir / "partial_metadata.json", "w") as f:
            json.dump({"sample_format": "float32"}, f)
        with pytest.raises(ValueError) as exc:
            repo.load_signal("partial.bin")
        msg = str(exc.value)
        assert "sampling_rate" in msg
        assert "sample_format" not in msg  # companion already supplied it
        assert repo.signal_count == 0


class TestRawMissingDeclarationRefusal:
    """AE1/R2: one refusal accumulates ALL missing raw declarations."""

    @pytest.fixture
    def bare_bin(self, data_dir):
        write_raw_file(data_dir / "bare.bin", np.zeros(32))
        return "bare.bin"

    def test_missing_sample_format_named(self, repo, data_dir, bare_bin):
        with pytest.raises(ValueError) as exc:
            repo.load_signal(bare_bin, sampling_rate=1000)
        msg = str(exc.value)
        assert "sample_format" in msg
        assert "sampling_rate" not in msg  # only the MISSING one is named
        assert repo.signal_count == 0

    def test_missing_sampling_rate_named(self, repo, data_dir, bare_bin):
        with pytest.raises(ValueError) as exc:
            repo.load_signal(bare_bin, sample_format="float32")
        msg = str(exc.value)
        assert "sampling_rate" in msg
        assert "sample_format" not in msg

    def test_dat_without_declaration_refused(self, repo, data_dir):
        """.dat routes to the raw refusal, not the old silent None failure."""
        write_raw_file(data_dir / "old_style.dat", np.zeros(24))
        with pytest.raises(ValueError) as exc:
            repo.load_signal("old_style.dat")
        msg = str(exc.value)
        assert "sample_format" in msg and "sampling_rate" in msg
        assert repo.signal_count == 0

    def test_both_missing_named_in_one_message(self, repo, data_dir, bare_bin):
        with pytest.raises(ValueError) as exc:
            repo.load_signal(bare_bin)
        msg = str(exc.value)
        assert "sample_format" in msg and "sampling_rate" in msg
        # Both remedies: the exact re-call and the companion alternative.
        assert "load_signal(filepath='bare.bin'" in msg
        assert "bare_metadata.json" in msg
        assert repo.signal_count == 0


class TestRawMultiChannelIds:
    """Decision 9: _ch<k> suffix only when the EFFECTIVE n_channels > 1."""

    @pytest.fixture
    def multi_bin(self, data_dir):
        interleaved = np.empty(200, dtype=np.float32)
        interleaved[0::2] = 5.0
        interleaved[1::2] = 9.0
        write_raw_file(data_dir / "multi.bin", interleaved)
        return "multi.bin"

    def test_two_channels_coexist_with_suffixed_ids(self, repo, data_dir, multi_bin):
        i0 = repo.load_signal(
            multi_bin,
            sampling_rate=1000,
            sample_format="float32",
            n_channels=2,
            channel_index=0,
        )
        i1 = repo.load_signal(
            multi_bin,
            sampling_rate=1000,
            sample_format="float32",
            n_channels=2,
            channel_index=1,
        )
        assert i0["signal_id"] == "multi_ch0"
        assert i1["signal_id"] == "multi_ch1"
        assert repo.signal_count == 2
        assert np.all(repo.get_signal("multi_ch0") == 5.0)
        assert np.all(repo.get_signal("multi_ch1") == 9.0)

    def test_same_channel_without_overwrite_collides(self, repo, data_dir, multi_bin):
        repo.load_signal(
            multi_bin,
            sampling_rate=1000,
            sample_format="float32",
            n_channels=2,
            channel_index=0,
        )
        with pytest.raises(ValueError, match="overwrite=True"):
            repo.load_signal(
                multi_bin,
                sampling_rate=1000,
                sample_format="float32",
                n_channels=2,
                channel_index=0,
            )
        assert repo.signal_count == 1

    def test_single_channel_id_unchanged(self, repo, data_dir):
        """n_channels == 1 keeps the pre-raw id (backward compatible)."""
        write_raw_file(data_dir / "mono.bin", np.zeros(10))
        info = repo.load_signal("mono.bin", sampling_rate=1000, sample_format="float32")
        assert info["signal_id"] == "mono"

    def test_overwrite_with_new_dtype_replaces_raw_format_fully(
        self, repo, data_dir, multi_bin
    ):
        repo.load_signal(
            multi_bin,
            sampling_rate=1000,
            sample_format="float32",
            n_channels=2,
            channel_index=0,
            scale_factor=2.0,
        )
        assert np.all(repo.get_signal("multi_ch0") == 10.0)  # 5.0 * 2.0
        info = repo.load_signal(
            multi_bin,
            sampling_rate=1000,
            sample_format="int16",
            n_channels=2,
            channel_index=0,
            overwrite=True,
        )
        # Fully replaced: new dtype AND no stale scale_factor left behind.
        assert info["raw_format"] == {
            "sample_format": "int16",
            "byte_order": "little",
            "n_channels": 2,
            "channel_index": 0,
            "header_offset": 0,
            "scale_factor": None,
        }
        assert repo.signal_count == 1


class TestRawBatchLoad:
    """Decision 11: raw params broadcast; fail-fast atomicity preserved."""

    def test_batch_broadcasts_raw_params(self, repo, data_dir):
        for name, fill in (("rb0.bin", 1.0), ("rb1.bin", 2.0)):
            write_raw_file(data_dir / name, np.full(20, fill))
        infos = repo.load_signals(
            ["rb0.bin", "rb1.bin"], sampling_rate=1000, sample_format="float32"
        )
        assert [i["signal_id"] for i in infos] == ["rb0", "rb1"]
        assert all(i["raw_format"]["sample_format"] == "float32" for i in infos)
        assert repo.signal_count == 2

    def test_batch_non_divisible_file_is_atomic(self, repo, data_dir):
        write_raw_file(data_dir / "good.bin", np.zeros(20))
        (data_dir / "bad.bin").write_bytes(b"\x00" * 7)  # 7 % 4 != 0
        with pytest.raises(ValueError, match="frames"):
            repo.load_signals(
                ["good.bin", "bad.bin"], sampling_rate=1000, sample_format="float32"
            )
        assert repo.signal_count == 0  # fail-fast atomic: nothing registered

    def test_batch_invalid_companion_value_gets_batch_framing(self, repo, data_dir):
        """An invalid companion value in a batch joins the accumulated
        batch-abort message (nothing loaded) instead of escaping as a lone
        error without the batch framing."""
        write_raw_file(data_dir / "okc.bin", np.zeros(8))
        write_raw_file(data_dir / "badc.bin", np.zeros(8))
        with open(data_dir / "badc_metadata.json", "w") as f:
            json.dump({"sampling_rate": 1000, "sample_format": "Float32"}, f)
        with pytest.raises(ValueError) as exc:
            repo.load_signals(["okc.bin", "badc.bin"], sampling_rate=1000)
        msg = str(exc.value)
        assert "Batch load aborted" in msg
        assert "Float32" in msg and "badc_metadata.json" in msg
        assert repo.signal_count == 0

    def test_batch_channel_derivation_agrees_with_single_route(self, repo, data_dir):
        interleaved = np.empty(40, dtype=np.float32)
        interleaved[0::2] = 3.0
        interleaved[1::2] = 4.0
        write_raw_file(data_dir / "pair.bin", interleaved)
        repo.load_signal(
            "pair.bin",
            sampling_rate=1000,
            sample_format="float32",
            n_channels=2,
            channel_index=0,
        )
        infos = repo.load_signals(
            ["pair.bin"],
            sampling_rate=1000,
            sample_format="float32",
            n_channels=2,
            channel_index=1,
        )
        assert infos[0]["signal_id"] == "pair_ch1"
        assert repo.signal_count == 2  # coexists with the earlier _ch0 load
        # The batch derivation site SEES the suffixed id as a collision too.
        with pytest.raises(ValueError, match="pair_ch0"):
            repo.load_signals(
                ["pair.bin"],
                sampling_rate=1000,
                sample_format="float32",
                n_channels=2,
                channel_index=0,
            )


class TestRawCompanionValidation:
    """Companion values are re-validated against the closed vocabularies."""

    def test_invalid_companion_sample_format_names_vocab_and_source(
        self, repo, data_dir
    ):
        write_raw_file(data_dir / "badcomp.bin", np.zeros(16))
        with open(data_dir / "badcomp_metadata.json", "w") as f:
            json.dump({"sampling_rate": 1000, "sample_format": "Float32"}, f)
        with pytest.raises(ValueError) as exc:
            repo.load_signal("badcomp.bin")
        msg = str(exc.value)
        assert "Float32" in msg  # the offending value
        assert "badcomp_metadata.json" in msg  # the companion source
        assert "float32" in msg and "int16" in msg  # the valid vocabulary

    def test_invalid_companion_byte_order_rejected(self, repo, data_dir):
        write_raw_file(data_dir / "bo.bin", np.zeros(16))
        with open(data_dir / "bo_metadata.json", "w") as f:
            json.dump(
                {
                    "sampling_rate": 1000,
                    "sample_format": "float32",
                    "byte_order": "big-endian",
                },
                f,
            )
        with pytest.raises(ValueError) as exc:
            repo.load_signal("bo.bin")
        msg = str(exc.value)
        assert "big-endian" in msg and "bo_metadata.json" in msg
        assert "little" in msg

    def test_invalid_companion_n_channels_type_rejected(self, repo, data_dir):
        """Wrong-typed companion ints raise typed errors, never TypeError."""
        write_raw_file(data_dir / "nc.bin", np.zeros(16))
        with open(data_dir / "nc_metadata.json", "w") as f:
            json.dump(
                {
                    "sampling_rate": 1000,
                    "sample_format": "float32",
                    "n_channels": "two",
                },
                f,
            )
        with pytest.raises(ValueError) as exc:
            repo.load_signal("nc.bin")
        assert "n_channels" in str(exc.value)
        assert "nc_metadata.json" in str(exc.value)

    def test_invalid_companion_scale_factor_type_rejected(self, repo, data_dir):
        """The scale_factor branch of the companion validator is exercised."""
        write_raw_file(data_dir / "sf.bin", np.zeros(16))
        with open(data_dir / "sf_metadata.json", "w") as f:
            json.dump(
                {
                    "sampling_rate": 1000,
                    "sample_format": "float32",
                    "scale_factor": "half",
                },
                f,
            )
        with pytest.raises(ValueError) as exc:
            repo.load_signal("sf.bin")
        assert "scale_factor" in str(exc.value)
        assert "sf_metadata.json" in str(exc.value)


class TestRawErrorContract:
    """Raw errors keep the two-track contract: typed, never swallowed."""

    def test_not_found_bin_message_form_matches_other_formats(self, repo, data_dir):
        with pytest.raises(FileNotFoundError) as exc_bin:
            repo.load_signal("ghost.bin", sampling_rate=1000, sample_format="float32")
        with pytest.raises(FileNotFoundError) as exc_csv:
            repo.load_signal("ghost.csv")
        form_bin = str(exc_bin.value).replace("ghost.bin", "<F>")
        form_csv = str(exc_csv.value).replace("ghost.csv", "<F>")
        assert form_bin == form_csv  # no format-specific existence oracle

    def test_sample_format_on_csv_is_contradiction(self, repo, signal_file):
        with pytest.raises(ValueError, match="contradicts"):
            repo.load_signal(str(signal_file), sample_format="float32")
        assert repo.signal_count == 0

    def test_raw_params_on_unknown_extension_is_unsupported_format(
        self, repo, data_dir
    ):
        """A `.xyz` file is NOT "self-describing" — declaring raw params
        for it is refused as an unsupported format naming the supported
        extensions, not as a contradiction with a header it doesn't have."""
        (data_dir / "mystery.xyz").write_bytes(b"\x00" * 8)
        with pytest.raises(ValueError) as exc:
            repo.load_signal("mystery.xyz", sample_format="float32")
        msg = str(exc.value)
        assert "not a supported" in msg
        assert ".csv" in msg and ".bin" in msg  # names SUPPORTED_EXTENSIONS
        assert "self-describing" not in msg
        assert repo.signal_count == 0

    def test_decoder_error_propagates_not_swallowed(self, repo, data_dir):
        (data_dir / "odd.bin").write_bytes(b"\x00" * 6)  # 6 % 4 != 0
        with pytest.raises(ValueError) as exc:
            repo.load_signal("odd.bin", sampling_rate=1000, sample_format="float32")
        # The decoder's arithmetic reaches the caller intact — never the
        # generic "Unable to load signal" swallow-shell message.
        msg = str(exc.value)
        assert "Unable to load signal" not in msg
        assert "remainder" in msg


class TestResolveSignalSamplingRate:
    """F10: resolve_signal refuses a stored non-positive sampling rate.

    A metadata-derived rate of 0 bypasses load_signal's explicit-param check
    (that guard only covers the sampling_rate argument), so it lands in the
    store and exercises resolve_signal's own guard / the StoredSignalInfo
    schema backstop. Either rejection surfaces as a ValueError (pydantic's
    ValidationError also subclasses ValueError).
    """

    def test_zero_metadata_rate_rejected_by_resolve_signal(self, tmp_path):
        from predictive_maintenance_mcp.mcp_tools._utils import resolve_signal

        signal = np.random.randn(500)
        csv_path = tmp_path / "zero_rate.csv"
        pd.DataFrame(signal).to_csv(csv_path, index=False, header=False)
        with open(tmp_path / "zero_rate_metadata.json", "w") as f:
            json.dump({"sampling_rate": 0, "signal_unit": "g"}, f)

        repo = get_repository()
        repo.clear_all()
        try:
            repo.load_signal(str(csv_path), signal_id="zero_rate")
            with pytest.raises(ValueError):
                resolve_signal("zero_rate")
        finally:
            repo.clear_all()
