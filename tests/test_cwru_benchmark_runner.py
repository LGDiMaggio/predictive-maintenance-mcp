"""U4 tests: the benchmark runner over synthetic imported records.

Everything runs on deterministic synthetic signals imported through the
REAL importer (scipy.io.savemat into tmp_path, no network, no real CWRU
data), using the repository-isolation pattern from
tests/test_cwru_benchmark_import.py. Coverage:

- outcome schema keyed by opaque id, mirroring the raw
  ``check_all_bearing_faults`` fields for a seeded BPFO fault;
- double-run byte-identical serialization (claim scoped to this
  environment) plus the red counterpart for the determinism check;
- the anomaly block is ALWAYS the constant excluded marker, even when a
  local model produced a result;
- the import-provenance tripwire refuses before any record runs;
- a record with no imported signal is recorded as missing, and the
  ``all``-style gate helper refuses to proceed to scoring;
- float canonicalization (9-decimal rounding, NumPy scalars, non-finite
  refusal) and the atomic newline-terminated outcomes write;
- the ``score`` subcommand raises NotImplementedError naming U5.

Generated outcomes always go to tmp_path — never to the real
benchmarks/cwru/results/ directory (committed only by the maintainer
run in a later unit).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

import predictive_maintenance_mcp
from benchmarks.cwru import runner
from benchmarks.cwru.__main__ import main
from benchmarks.cwru.importer import SIGNAL_UNIT, import_record
from benchmarks.cwru.records import OpsRecord
from predictive_maintenance_mcp.diagnostics.bearing_analyzer import (
    check_all_bearing_faults,
)
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

#: Sampling rate of the synthetic records (the CWRU fault-group rate).
FS = 12000

#: Nominal shaft speed (0 hp CWRU load point): shaft ~29.95 Hz.
RPM = 1797

#: CWRU-published BPFO factor for the 6205 drive-end bearing.
BPFO_FACTOR = 3.5848


def _bpfo_fault_signal() -> np.ndarray:
    """Seeded outer-race-fault synthetic: carrier modulated at BPFO.

    3 kHz carrier (inside the default 500-5000 Hz envelope band at this
    fs), amplitude-modulated at BPFO and its 2nd harmonic, plus seeded
    noise — deterministic by construction, detected by the BPFO check.
    """
    n = FS  # 1 second
    t = np.arange(n) / FS
    rng = np.random.default_rng(123)
    bpfo = BPFO_FACTOR * (RPM / 60.0)
    modulation = (
        1.0
        + 0.8 * np.sin(2 * np.pi * bpfo * t)
        + 0.3 * np.sin(2 * np.pi * 2 * bpfo * t)
    )
    fault: np.ndarray = np.sin(
        2 * np.pi * 3000.0 * t
    ) * modulation + 0.05 * rng.standard_normal(n)
    return fault


def _noise_signal() -> np.ndarray:
    """Seeded pure-noise synthetic (no seeded fault)."""
    rng = np.random.default_rng(456)
    noise: np.ndarray = 0.1 * rng.standard_normal(FS)
    return noise


def _record(opaque_id: str, file_id: int, internal_mat_key: str) -> OpsRecord:
    return OpsRecord(
        opaque_id=opaque_id,
        file_id=file_id,
        url=f"https://engineering.case.edu/sites/default/files/{file_id}.mat",
        internal_mat_key=internal_mat_key,
        channel="DE",
        fs_hz=FS,
        nominal_rpm=RPM,
        load_hp=0,
        cache_filename=f"{file_id}.mat",
    )


@pytest.fixture(autouse=True)
def repo():
    """The singleton repository, cleared before and after every test."""
    repository = get_repository()
    repository.clear_all()
    yield repository
    repository.clear_all()


@pytest.fixture
def imported(tmp_path: Path, repo):
    """Two synthetic records (BPFO fault + noise) through the real importer."""
    signals_dir = tmp_path / "signals"
    signals = {"cwru_001": _bpfo_fault_signal(), "cwru_002": _noise_signal()}
    records = (
        _record("cwru_001", 105, "X105_DE_time"),
        _record("cwru_002", 118, "X118_DE_time"),
    )
    for record, (opaque_id, signal) in zip(records, signals.items()):
        assert record.opaque_id == opaque_id
        mat_path = tmp_path / record.cache_filename
        key = record.internal_mat_key
        assert key is not None
        savemat(str(mat_path), {key: signal.reshape(-1, 1)})
        import_record(record, mat_path=mat_path, signals_dir=signals_dir)
    return {"records": records, "signals": signals}


# ---------------------------------------------------------------------------
# Happy path: schema-valid outcomes keyed by opaque id
# ---------------------------------------------------------------------------


class TestRunnerHappyPath:
    """Runner over 2 synthetic imported records produces valid outcomes."""

    def test_outcomes_keyed_by_opaque_id_with_full_schema(self, imported, repo):
        outcomes = runner.run_records(imported["records"], repository=repo)

        assert set(outcomes) == {"cwru_001", "cwru_002"}
        for opaque_id, outcome in outcomes.items():
            assert outcome["status"] == runner.OUTCOME_STATUS_OK
            assert outcome["signal_id"] == opaque_id
            assert set(outcome["bearing"]["fault_checks"]) == {
                "BPFO",
                "BPFI",
                "BSF",
                "FTF",
            }
            assert outcome["context"]["bearing_id"] == "6205"
            assert outcome["context"]["fs_hz"] == FS
            assert outcome["context"]["nominal_rpm"] == RPM
            assert outcome["context"]["signal_unit"] == SIGNAL_UNIT
            assert outcome["context"]["num_samples"] == FS
            assert "status" in outcome["iso_severity"]
            assert outcome["anomaly_detection"] == dict(runner.EXCLUDED_ANOMALY_MARKER)

    def test_outcome_subset_mirrors_check_all_bearing_faults(self, imported, repo):
        """Per-fault fields are copied verbatim from the analyzer output."""
        outcomes = runner.run_records(imported["records"], repository=repo)
        out = outcomes["cwru_001"]

        direct = check_all_bearing_faults(
            signal=imported["signals"]["cwru_001"],
            fs=float(FS),
            bearing_id="6205",
            rpm=float(RPM),
            signal_id="cwru_001",
        )

        # The seeded BPFO fault is detected by the BPFO check.
        bpfo = out["bearing"]["fault_checks"]["BPFO"]
        assert bpfo["detected"] is True
        assert bpfo["fault_type_canonical"] == "outer_race"
        assert bpfo["evidence_strength"] in {"moderate", "high"}
        assert bpfo["detected_frequency_hz"] is not None

        # Every fault check mirrors the raw analyzer fields exactly.
        for check in direct["fault_checks"]:
            mirrored = out["bearing"]["fault_checks"][check["fault_type"]]
            for field in (
                "fault_type_canonical",
                "expected_frequency_hz",
                "detected",
                "detected_frequency_hz",
                "magnitude",
                "deviation_pct",
                "harmonics_detected",
                "evidence_strength",
            ):
                assert mirrored[field] == check[field], field

        # most_likely carried as context, matching the analyzer summary.
        assert out["bearing"]["most_likely_fault"] == direct["most_likely_fault"]
        assert (
            out["bearing"]["most_likely_fault_canonical"]
            == direct["most_likely_fault_canonical"]
        )
        assert out["bearing"]["shaft_frequency_hz"] == direct["shaft_frequency_hz"]

    def test_marker_constant_has_the_documented_value(self):
        assert dict(runner.EXCLUDED_ANOMALY_MARKER) == {
            "status": "excluded_unversioned_models"
        }


# ---------------------------------------------------------------------------
# Happy path: determinism — double run, byte-identical serialization
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Double run over the same records serializes byte-identically."""

    def test_double_run_serializes_byte_identically(self, imported, repo):
        records = imported["records"]
        first = runner.serialize_outcomes(runner.run_records(records, repository=repo))
        second = runner.serialize_outcomes(runner.run_records(records, repository=repo))
        assert first == second
        assert first.endswith(b"\n")

    def test_check_determinism_passes_and_returns_outcomes(self, imported, repo):
        outcomes = runner.check_determinism(imported["records"], repository=repo)
        assert set(outcomes) == {"cwru_001", "cwru_002"}

    def test_check_determinism_red_on_divergent_runs(self, monkeypatch):
        """The check must actually fail when runs diverge, not just pass."""
        divergent = iter(
            [
                {"cwru_001": {"status": "ok", "value": 1.0}},
                {"cwru_001": {"status": "ok", "value": 2.0}},
            ]
        )
        monkeypatch.setattr(
            runner, "run_records", lambda records, repository=None: next(divergent)
        )
        with pytest.raises(ValueError, match="[Dd]eterminism"):
            runner.check_determinism((), repository=None)


# ---------------------------------------------------------------------------
# Happy path: anomaly block excluded even when a local model ran
# ---------------------------------------------------------------------------


class TestAnomalyExclusion:
    """Committed outcomes never bind to unversioned local models."""

    def test_marker_replaces_a_live_anomaly_result(self, imported, repo, monkeypatch):
        real = runner.diagnose_vibration

        def with_local_model(*args, **kwargs):
            result = real(*args, **kwargs)
            # Simulate models/*.pkl being present on this machine.
            result["anomaly_detection"] = {
                "anomaly_ratio": 0.42,
                "anomaly_count": 21,
                "num_segments": 50,
                "overall_health": "Faulty",
                "anomaly_score": -0.3,
            }
            return result

        monkeypatch.setattr(runner, "diagnose_vibration", with_local_model)
        outcomes = runner.run_records(imported["records"], repository=repo)

        for outcome in outcomes.values():
            assert outcome["anomaly_detection"] == dict(runner.EXCLUDED_ANOMALY_MARKER)
        payload = runner.serialize_outcomes(outcomes)
        assert b"excluded_unversioned_models" in payload
        assert b"anomaly_ratio" not in payload
        assert b"overall_health" not in payload


# ---------------------------------------------------------------------------
# Error path: import-provenance tripwire
# ---------------------------------------------------------------------------


class TestProvenanceTripwire:
    """A foreign-checkout import refuses before anything is measured."""

    def test_green_in_this_checkout(self):
        resolved = runner.assert_import_provenance()
        assert resolved == runner.EXPECTED_PACKAGE_INIT.resolve()

    def test_foreign_package_refused_before_any_record_runs(
        self, imported, repo, tmp_path, monkeypatch
    ):
        foreign = tmp_path / "other-checkout" / "src" / "__init__.py"
        monkeypatch.setattr(predictive_maintenance_mcp, "__file__", str(foreign))

        never_called = []
        monkeypatch.setattr(
            runner,
            "diagnose_vibration",
            lambda *a, **k: never_called.append(1),
        )

        with pytest.raises(ValueError, match="checkout") as excinfo:
            runner.run_records(imported["records"], repository=repo)
        message = str(excinfo.value)
        assert "other-checkout" in message
        assert "setup-worktree" in message  # remedy named
        assert never_called == [], "tripwire must fire before any record runs"

    def test_missing_dunder_file_refused(self, monkeypatch):
        monkeypatch.setattr(predictive_maintenance_mcp, "__file__", None)
        with pytest.raises(ValueError, match="tripwire"):
            runner.assert_import_provenance()


# ---------------------------------------------------------------------------
# Error path: missing imported signal + the fail-closed scoring gate
# ---------------------------------------------------------------------------


class TestMissingSignalAndGate:
    """A record with no imported signal is missing, and the gate refuses."""

    def test_missing_signal_recorded_without_aborting_the_batch(self, imported, repo):
        records = imported["records"] + (_record("cwru_003", 130, "X130_DE_time"),)
        outcomes = runner.run_records(records, repository=repo)

        assert set(outcomes) == {"cwru_001", "cwru_002", "cwru_003"}
        missing = outcomes["cwru_003"]
        assert missing["status"] == runner.OUTCOME_STATUS_MISSING_SIGNAL
        assert "cwru_003" in missing["error"]
        assert "import" in missing["error"]  # remedy named
        assert "bearing" not in missing  # no measurement fields
        # The other records still ran — distinguishable, not aborted.
        assert outcomes["cwru_001"]["status"] == runner.OUTCOME_STATUS_OK
        assert outcomes["cwru_002"]["status"] == runner.OUTCOME_STATUS_OK

    def test_gate_refuses_to_score_on_a_missing_record(self, imported, repo):
        records = imported["records"] + (_record("cwru_003", 130, "X130_DE_time"),)
        outcomes = runner.run_records(records, repository=repo)
        with pytest.raises(ValueError, match="cwru_003") as excinfo:
            runner.assert_outcomes_complete(outcomes, records)
        assert "scoring" in str(excinfo.value)

    def test_gate_refuses_an_outcome_with_no_record(self, imported, repo):
        outcomes = runner.run_records(imported["records"], repository=repo)
        outcomes["cwru_099"] = {"status": "ok"}
        with pytest.raises(ValueError, match="cwru_099"):
            runner.assert_outcomes_complete(outcomes, imported["records"])

    def test_gate_green_on_a_complete_ok_set(self, imported, repo):
        outcomes = runner.run_records(imported["records"], repository=repo)
        runner.assert_outcomes_complete(outcomes, imported["records"])  # no raise

    def test_duplicate_opaque_ids_refused(self, imported, repo):
        records = (imported["records"][0], imported["records"][0])
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            runner.run_records(records, repository=repo)


# ---------------------------------------------------------------------------
# Serialization: canonical floats, atomic newline-terminated write
# ---------------------------------------------------------------------------


class TestSerialization:
    """Deterministic serialization contract (documented in runner.py)."""

    def test_floats_rounded_to_nine_decimals_including_numpy(self):
        payload = {
            "cwru_x": {
                "status": "ok",
                "plain": 0.123456789123456,
                "third": np.float64(1.0) / np.float64(3.0),
                "count": np.int64(7),
                "flag": np.bool_(True),
            }
        }
        text = runner.serialize_outcomes(payload).decode("utf-8")
        assert "0.123456789" in text
        assert "0.123456789123456" not in text
        assert "0.333333333" in text
        parsed = json.loads(text)
        assert parsed["cwru_x"]["count"] == 7
        assert parsed["cwru_x"]["flag"] is True

    def test_non_finite_float_refused_naming_the_path(self):
        with pytest.raises(ValueError, match="cwru_x.bad"):
            runner.serialize_outcomes({"cwru_x": {"bad": float("nan")}})

    def test_unsupported_type_refused(self):
        with pytest.raises(ValueError, match="unsupported type"):
            runner.serialize_outcomes({"cwru_x": {"raw": np.arange(3)}})

    def test_write_outcomes_is_newline_terminated_sorted_json(
        self, imported, repo, tmp_path
    ):
        outcomes = runner.run_records(imported["records"], repository=repo)
        target = tmp_path / "results" / "outcomes.json"
        written = runner.write_outcomes(outcomes, target)

        assert written == target
        raw = target.read_bytes()
        assert raw.endswith(b"\n")
        assert not raw.endswith(b"\n\n")
        assert raw == runner.serialize_outcomes(outcomes)
        parsed = json.loads(raw.decode("utf-8"))
        assert list(parsed) == sorted(parsed)
        assert not target.with_name(target.name + ".part").exists()


# ---------------------------------------------------------------------------
# CLI: score is honestly unimplemented until U5
# ---------------------------------------------------------------------------


class TestScoreSubcommand:
    """`python -m benchmarks.cwru score` refuses, naming U5."""

    def test_score_raises_not_implemented_naming_u5(self):
        with pytest.raises(NotImplementedError, match="U5"):
            main(["score"])
