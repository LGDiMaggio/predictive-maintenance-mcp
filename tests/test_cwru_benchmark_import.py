"""U3 tests: ``.mat`` channel extraction → opaque signals in the repository.

Everything runs on synthetic ``.mat`` files written with
``scipy.io.savemat`` into tmp_path — no network, no real CWRU data. The
blind protocol (origin acceptance example AE3) is covered at the
companion level: the written ``<opaque_id>_metadata.json`` carries
exactly the two declarations and ``get_signal_info().source_metadata``
carries no label-bearing key. Every refusal path asserts that nothing
was written — no ``.npy``, no companion, no repository entry.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from scipy.io import savemat

from benchmarks.cwru.importer import (
    SIGNAL_UNIT,
    import_record,
    import_records,
)
from benchmarks.cwru.records import LABEL_BEARING_KEYS, OpsRecord
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from predictive_maintenance_mcp.signal_processing.spectral import (
    compute_envelope_spectrum,
)

#: Known drive-end samples — the channel the importer must extract.
DE_SAMPLES = np.arange(64, dtype=np.float64)

#: Decoy fan-end channel: same length, unmistakably different values.
FE_DECOY = np.full(64, -7.0)


def _record(
    opaque_id: str = "cwru_001",
    file_id: int = 105,
    internal_mat_key: Optional[str] = "X105_DE_time",
    fs_hz: int = 12000,
) -> OpsRecord:
    return OpsRecord(
        opaque_id=opaque_id,
        file_id=file_id,
        url=(f"https://engineering.case.edu/sites/default/files/{file_id}.mat"),
        internal_mat_key=internal_mat_key,
        channel="DE",
        fs_hz=fs_hz,
        nominal_rpm=1797,
        load_hp=0,
        cache_filename=f"{file_id}.mat",
    )


def _write_mat(path: Path, variables: dict) -> Path:
    savemat(str(path), variables)
    return path


def _default_mat_variables() -> dict:
    """Declared DE channel, FE decoy, and RPM scalar — the CWRU layout."""
    return {
        "X105_DE_time": DE_SAMPLES.reshape(-1, 1),
        "X105_FE_time": FE_DECOY.reshape(-1, 1),
        "X105RPM": np.array([[1797.0]]),
    }


@pytest.fixture(autouse=True)
def repo():
    """The singleton repository, cleared before and after every test."""
    repository = get_repository()
    repository.clear_all()
    yield repository
    repository.clear_all()


@pytest.fixture
def signals_dir(tmp_path: Path) -> Path:
    """Converted-signal target dir — NOT pre-created, so refusal tests can
    assert the importer never even created it."""
    return tmp_path / "signals"


@pytest.fixture
def mat_file(tmp_path: Path) -> Path:
    return _write_mat(tmp_path / "105.mat", _default_mat_variables())


def _assert_nothing_written(signals_dir: Path, repo) -> None:
    """The fail-closed contract: no files, no directory, no repo entry."""
    assert not signals_dir.exists(), "refusal must not create the signals dir"
    assert repo.signal_count == 0, "refusal must not load anything"


# ---------------------------------------------------------------------------
# Happy path: the declared key — and only the declared key — is imported
# ---------------------------------------------------------------------------


class TestImportHappyPath:
    """Synthetic .mat with declared key, decoy channel, and RPM scalar."""

    def test_imports_only_the_declared_key(self, mat_file, signals_dir, repo):
        result = import_record(_record(), mat_path=mat_file, signals_dir=signals_dir)

        stored = np.asarray(repo.get_signal("cwru_001"))
        np.testing.assert_array_equal(stored, DE_SAMPLES)
        assert not np.array_equal(stored, FE_DECOY), "decoy channel leaked"

        on_disk = np.load(signals_dir / "cwru_001.npy")
        np.testing.assert_array_equal(on_disk, DE_SAMPLES)

        assert result["signal_id"] == "cwru_001"
        assert result["num_samples"] == 64
        assert result["sampling_rate"] == 12000
        assert result["signal_unit"] == "g"
        assert Path(str(result["npy_path"])) == signals_dir / "cwru_001.npy"

    def test_info_shows_opaque_id_declared_fs_and_unit(
        self, mat_file, signals_dir, repo
    ):
        import_record(_record(), mat_path=mat_file, signals_dir=signals_dir)
        info = repo.get_signal_info("cwru_001")
        assert info["signal_id"] == "cwru_001"
        assert info["sampling_rate"] == 12000
        assert info["signal_unit"] == "g"
        assert info["num_samples"] == 64
        assert info["filepath"].endswith("cwru_001.npy")

    @pytest.mark.parametrize("shape", [(-1, 1), (1, -1)], ids=["Nx1", "1xN"])
    def test_column_and_row_vectors_coerce_via_ravel(
        self, tmp_path, signals_dir, repo, shape
    ):
        mat = _write_mat(
            tmp_path / "105.mat", {"X105_DE_time": DE_SAMPLES.reshape(shape)}
        )
        import_record(_record(), mat_path=mat, signals_dir=signals_dir)
        stored = np.asarray(repo.get_signal("cwru_001"))
        assert stored.ndim == 1
        np.testing.assert_array_equal(stored, DE_SAMPLES)

    def test_channel_is_stored_as_float64(self, tmp_path, signals_dir, repo):
        mat = _write_mat(
            tmp_path / "105.mat",
            {"X105_DE_time": DE_SAMPLES.astype(np.float32).reshape(-1, 1)},
        )
        import_record(_record(), mat_path=mat, signals_dir=signals_dir)
        stored = np.asarray(repo.get_signal("cwru_001"))
        assert stored.dtype == np.float64
        assert np.load(signals_dir / "cwru_001.npy").dtype == np.float64


# ---------------------------------------------------------------------------
# Happy path (blindness, AE3): the companion carries ONLY the declarations
# ---------------------------------------------------------------------------


class TestCompanionBlindness:
    """The written companion and the surfaced metadata carry no label."""

    def test_companion_contains_exactly_the_two_declarations(
        self, mat_file, signals_dir, repo
    ):
        import_record(_record(), mat_path=mat_file, signals_dir=signals_dir)
        payload = json.loads(
            (signals_dir / "cwru_001_metadata.json").read_text(encoding="utf-8")
        )
        assert payload == {"sampling_rate": 12000, "signal_unit": SIGNAL_UNIT}

    def test_source_metadata_carries_no_label_bearing_keys(
        self, mat_file, signals_dir, repo
    ):
        import_record(_record(), mat_path=mat_file, signals_dir=signals_dir)
        source_metadata = repo.get_signal_info("cwru_001")["source_metadata"]
        # The repository surfaces the WHOLE companion here, so exact key
        # equality is the blindness guarantee, not just disjointness.
        assert set(source_metadata) == {"sampling_rate", "signal_unit"}
        assert LABEL_BEARING_KEYS.isdisjoint(source_metadata)


# ---------------------------------------------------------------------------
# Error path: declared key missing from the .mat
# ---------------------------------------------------------------------------


class TestMissingDeclaredKey:
    """Absent internal_mat_key refuses, naming record, key, candidates."""

    def test_missing_key_names_record_key_and_candidates(
        self, tmp_path, signals_dir, repo
    ):
        mat = _write_mat(
            tmp_path / "105.mat",
            {
                "X105_FE_time": FE_DECOY.reshape(-1, 1),
                "X105RPM": np.array([[1797.0]]),
            },
        )
        with pytest.raises(ValueError, match="X105_DE_time") as excinfo:
            import_record(_record(), mat_path=mat, signals_dir=signals_dir)
        message = str(excinfo.value)
        assert "cwru_001" in message
        assert "X105_FE_time" in message  # candidates named
        assert "X105RPM" in message
        assert "records_ops.json" in message  # remedy named
        _assert_nothing_written(signals_dir, repo)

    def test_missing_cache_file_names_download_remedy(
        self, tmp_path, signals_dir, repo
    ):
        with pytest.raises(ValueError, match="download") as excinfo:
            import_record(
                _record(),
                mat_path=tmp_path / "absent.mat",
                signals_dir=signals_dir,
            )
        assert "cwru_001" in str(excinfo.value)
        _assert_nothing_written(signals_dir, repo)

    def test_corrupt_mat_names_redownload_remedy(self, tmp_path, signals_dir, repo):
        garbage = tmp_path / "105.mat"
        garbage.write_bytes(b"this is not a MATLAB file")
        with pytest.raises(ValueError, match="cannot be read"):
            import_record(_record(), mat_path=garbage, signals_dir=signals_dir)
        _assert_nothing_written(signals_dir, repo)


# ---------------------------------------------------------------------------
# Error path: declared key present but not numeric / not 1-D-coercible
# ---------------------------------------------------------------------------


class TestNotCoercible:
    """Non-numeric and ambiguous-shape variables refuse, nothing written."""

    @pytest.mark.parametrize(
        "value, reason",
        [
            ("not a signal channel", "string array"),
            (np.arange(20, dtype=np.float64).reshape(10, 2), "2-column matrix"),
            (np.zeros((2, 3, 4)), "3-D array"),
            (np.empty((0, 1)), "empty column"),
            ((DE_SAMPLES + 1j * DE_SAMPLES).reshape(-1, 1), "complex array"),
        ],
        ids=["string", "multi_column", "three_d", "empty", "complex"],
    )
    def test_non_coercible_variable_refused(
        self, tmp_path, signals_dir, repo, value, reason
    ):
        mat = _write_mat(tmp_path / "105.mat", {"X105_DE_time": value})
        with pytest.raises(ValueError, match="not a 1-D-coercible") as excinfo:
            import_record(_record(), mat_path=mat, signals_dir=signals_dir)
        message = str(excinfo.value)
        assert "cwru_001" in message
        assert "X105_DE_time" in message
        _assert_nothing_written(signals_dir, repo)

    def test_green_counterpart_plain_vector_still_imports(
        self, tmp_path, signals_dir, repo
    ):
        """The refusals above are the guard, not the harness: the same
        plumbing with a clean 1-D vector imports fine."""
        mat = _write_mat(tmp_path / "105.mat", {"X105_DE_time": DE_SAMPLES})
        import_record(_record(), mat_path=mat, signals_dir=signals_dir)
        np.testing.assert_array_equal(
            np.asarray(repo.get_signal("cwru_001")), DE_SAMPLES
        )


# ---------------------------------------------------------------------------
# Error path: internal_mat_key null on the record
# ---------------------------------------------------------------------------


class TestNullInternalMatKey:
    """An unpinned key refuses before touching anything, naming freeze."""

    def test_null_key_names_freeze_remedy(self, mat_file, signals_dir, repo):
        record = _record(internal_mat_key=None)
        with pytest.raises(ValueError, match="internal_mat_key") as excinfo:
            import_record(record, mat_path=mat_file, signals_dir=signals_dir)
        message = str(excinfo.value)
        assert "cwru_001" in message
        assert "freeze" in message
        _assert_nothing_written(signals_dir, repo)


# ---------------------------------------------------------------------------
# Edge: re-import refuses without overwrite, replaces cleanly with it
# ---------------------------------------------------------------------------


class TestReimportOverwrite:
    """overwrite=False fails closed; overwrite=True replaces everything."""

    def test_reimport_without_overwrite_refuses_and_keeps_original(
        self, tmp_path, mat_file, signals_dir, repo
    ):
        import_record(_record(), mat_path=mat_file, signals_dir=signals_dir)
        replacement = _write_mat(
            tmp_path / "replacement.mat",
            {"X105_DE_time": (DE_SAMPLES * 2).reshape(-1, 1)},
        )
        with pytest.raises(ValueError, match="overwrite=True") as excinfo:
            import_record(_record(), mat_path=replacement, signals_dir=signals_dir)
        assert "cwru_001" in str(excinfo.value)
        # Previous import untouched, in memory and on disk.
        np.testing.assert_array_equal(
            np.asarray(repo.get_signal("cwru_001")), DE_SAMPLES
        )
        np.testing.assert_array_equal(np.load(signals_dir / "cwru_001.npy"), DE_SAMPLES)

    def test_reimport_with_overwrite_replaces_cleanly(
        self, tmp_path, mat_file, signals_dir, repo
    ):
        import_record(_record(), mat_path=mat_file, signals_dir=signals_dir)
        replacement = _write_mat(
            tmp_path / "replacement.mat",
            {"X105_DE_time": (DE_SAMPLES * 2).reshape(-1, 1)},
        )
        import_record(
            _record(), mat_path=replacement, signals_dir=signals_dir, overwrite=True
        )
        np.testing.assert_array_equal(
            np.asarray(repo.get_signal("cwru_001")), DE_SAMPLES * 2
        )
        np.testing.assert_array_equal(
            np.load(signals_dir / "cwru_001.npy"), DE_SAMPLES * 2
        )
        assert repo.signal_count == 1

    def test_existing_repository_id_alone_refuses_without_overwrite(
        self, mat_file, signals_dir, repo
    ):
        """The repository entry survives a disk cleanup (it is in-memory
        state): the refusal must fire on it too, not only on files."""
        import_record(_record(), mat_path=mat_file, signals_dir=signals_dir)
        (signals_dir / "cwru_001.npy").unlink()
        (signals_dir / "cwru_001_metadata.json").unlink()
        with pytest.raises(ValueError, match="repository entry"):
            import_record(_record(), mat_path=mat_file, signals_dir=signals_dir)
        # With overwrite the same call succeeds and restores the files.
        import_record(
            _record(), mat_path=mat_file, signals_dir=signals_dir, overwrite=True
        )
        assert (signals_dir / "cwru_001.npy").exists()
        assert (signals_dir / "cwru_001_metadata.json").exists()


# ---------------------------------------------------------------------------
# Batch convenience: import_records
# ---------------------------------------------------------------------------


class TestImportRecordsBatch:
    """The batch wrapper resolves per-record cache paths and fails fast."""

    def test_batch_imports_all_records_in_order(self, tmp_path, signals_dir, repo):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        _write_mat(cache_dir / "105.mat", _default_mat_variables())
        _write_mat(
            cache_dir / "118.mat",
            {"X118_DE_time": (DE_SAMPLES + 100.0).reshape(-1, 1)},
        )
        records = [
            _record(),
            _record(opaque_id="cwru_002", file_id=118, internal_mat_key="X118_DE_time"),
        ]
        results = import_records(records, cache_dir=cache_dir, signals_dir=signals_dir)
        assert [r["signal_id"] for r in results] == ["cwru_001", "cwru_002"]
        assert repo.signal_count == 2
        np.testing.assert_array_equal(
            np.asarray(repo.get_signal("cwru_002")), DE_SAMPLES + 100.0
        )

    def test_batch_fails_fast_naming_the_offending_record(
        self, tmp_path, signals_dir, repo
    ):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        _write_mat(cache_dir / "105.mat", _default_mat_variables())
        records = [
            _record(),
            _record(  # 118.mat deliberately absent from the cache
                opaque_id="cwru_002", file_id=118, internal_mat_key="X118_DE_time"
            ),
        ]
        with pytest.raises(ValueError, match="cwru_002"):
            import_records(records, cache_dir=cache_dir, signals_dir=signals_dir)
        # Fail-fast, not atomic: the record imported before the failure
        # stays imported (documented; `all` refuses to score on partial).
        assert repo.signal_count == 1
        assert repo.get_signal_info("cwru_001")["signal_id"] == "cwru_001"


# ---------------------------------------------------------------------------
# Integration: imported signal retrievable and analyzable end-to-end
# ---------------------------------------------------------------------------


class TestAnalysisIntegration:
    """The opaque signal feeds the real analysis path without friction."""

    def test_imported_signal_runs_through_envelope_spectrum(
        self, tmp_path, signals_dir, repo
    ):
        fs = 12000
        n = 8192
        t = np.arange(n) / fs
        rng = np.random.default_rng(7)
        # Amplitude-modulated 3 kHz carrier + light noise: a plausible
        # bearing-like synthetic, deterministic by seed.
        synthetic = np.sin(2 * np.pi * 3000.0 * t) * (
            1.0 + 0.5 * np.sin(2 * np.pi * 107.4 * t)
        ) + 0.01 * rng.standard_normal(n)
        mat = _write_mat(
            tmp_path / "105.mat", {"X105_DE_time": synthetic.reshape(-1, 1)}
        )

        import_record(_record(), mat_path=mat, signals_dir=signals_dir)

        stored = repo.get_signal("cwru_001")  # read-only repository view
        result = compute_envelope_spectrum(np.asarray(stored), float(fs))
        assert set(result) >= {"top_peaks", "diagnosis", "num_envelope_samples"}
        assert result["num_envelope_samples"] == n
        assert len(result["top_peaks"]) > 0
