"""U6 executable guards: blindness, publication drift, CI wiring.

Every methodology claim gets a named executing check, and every guard is
shown to go red (a guard that can only pass is decoration). The blindness
legs of the plan (origin acceptance example AE3) map to files like this:

- Leg 1 -- ops allowlist + file-level key disjointness, with its red
  mutation: ``tests/test_cwru_benchmark_records.py`` (TestOpsBlindness,
  TestMutationGuard). Fully covered there; deliberately NOT duplicated
  here.
- Leg 2 -- static import/reference scan of the modules feeding the
  system under test: THIS file, :class:`TestStaticLabelBlindness`.
- Leg 3 -- companion content (the written ``<opaque_id>_metadata.json``
  and the surfaced ``source_metadata`` carry exactly the two
  declarations): ``tests/test_cwru_benchmark_import.py``
  (TestCompanionBlindness). Fully covered there; NOT duplicated here.
- Leg 4 -- SUT call-boundary guard on the actual ``diagnose_vibration``
  invocation, mutation-tested: THIS file, :class:`TestSutCallBoundary`.
- Determinism double-run (green AND red counterpart):
  ``tests/test_cwru_benchmark_runner.py`` (TestDeterminism). Fully
  covered there; NOT duplicated here.
- Publication drift guard with slot binding (AE4): THIS file,
  :class:`TestDriftGuardOnFixtures` (tmp-fixture matrix, mutation
  included) and :class:`TestDriftGuardOnTheRealRepo` (activates
  automatically the moment U7 publishes a section).
- CI wiring -- the Black format gate covers ``benchmarks/``: THIS file,
  :class:`TestFormatGateCoversBenchmarks` (blockingness of the job
  itself is asserted by ``tests/test_ci_gates.py``).

Everything runs on synthetic data written into tmp_path -- no network,
no real CWRU records, no writes to ``benchmarks/cwru/results/``.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import savemat

from benchmarks.cwru import drift_guard, runner, scorer
from benchmarks.cwru.importer import SIGNAL_UNIT, import_record
from benchmarks.cwru.records import OpsRecord
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "benchmarks" / "cwru"

#: Sampling rate / nominal speed of the synthetic records (CWRU 0 hp point).
FS = 12000
RPM = 1797

#: The blind protocol's id shape: sequential and free of fault semantics.
OPAQUE_ID_PATTERN = re.compile(r"^cwru_\d{3}$")


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
    """Two synthetic records through the REAL importer (savemat -> repo).

    Same fixture pattern as tests/test_cwru_benchmark_runner.py; seeded
    noise is enough here because this file guards the call boundary, not
    detection quality.
    """
    signals_dir = tmp_path / "signals"
    signals = {
        "cwru_001": 0.1 * np.random.default_rng(456).standard_normal(FS),
        "cwru_002": 0.1 * np.random.default_rng(789).standard_normal(FS),
    }
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
# Blindness leg 2: static import/reference scan of the blind modules
# ---------------------------------------------------------------------------

#: The modules upstream of the scorer: none may reference the label side.
BLIND_MODULE_NAMES = ("runner.py", "importer.py", "download.py", "__main__.py")

#: Direct label accessors: the function, the file, the path constant, and
#: the label-carrying model types. These names may appear ONLY in
#: records.py (which defines them) and scorer.py (the sole label reader).
#: ``__main__.py`` importing the ``scorer`` MODULE is allowed -- 'scorer'
#: is deliberately not in this tuple.
LABEL_ACCESS_TOKENS = (
    "label_view",
    "labels.json",
    "LABELS_PATH",
    "LabelRecord",
    "LabeledRecord",
)


def _label_references(source: str) -> set[str]:
    """Return every direct label-accessor token found in *source*."""
    return {token for token in LABEL_ACCESS_TOKENS if token in source}


def _module_source(name: str) -> str:
    return (PACKAGE_DIR / name).read_text(encoding="utf-8")


class TestStaticLabelBlindness:
    """No module feeding the SUT references the label side, verifiably."""

    def test_scan_inputs_actually_read(self):
        """Anchor: 'found nothing' must mean something (test_ci_gates
        pattern). An empty or unreadable source would pass the scan
        vacuously."""
        for name in BLIND_MODULE_NAMES + ("scorer.py", "records.py"):
            source = _module_source(name)
            assert len(source) > 500, f"{name} read suspiciously short"
            assert "import" in source, f"{name} does not even import"

    @pytest.mark.parametrize("name", BLIND_MODULE_NAMES)
    def test_blind_module_references_no_label_accessor(self, name):
        leaked = _label_references(_module_source(name))
        assert not leaked, (
            f"benchmarks/cwru/{name} references label accessor(s) "
            f"{sorted(leaked)} -- only scorer.py may touch the label "
            f"side; everything upstream must stay blind (AE3)."
        )

    def test_scorer_still_references_label_view(self):
        """Positive control: on a rename of the accessor the scan above
        would pass vacuously, so the guard requires the tokens to still
        exist where they belong."""
        scorer_source = _module_source("scorer.py")
        records_source = _module_source("records.py")
        assert "label_view" in scorer_source
        assert "def label_view" in records_source
        assert "labels.json" in records_source

    def test_main_wires_score_through_the_scorer_module(self):
        """The allowed path: __main__ imports the scorer MODULE (which is
        not a label accessor token) rather than any label accessor."""
        source = _module_source("__main__.py")
        assert "scorer" in source
        assert not _label_references(source)

    def test_the_scanner_finds_a_leak_when_one_exists(self):
        """The guard's own failing path: a leaking import and a direct
        file read are both found in synthetic sources."""
        leaking_import = "from benchmarks.cwru.records import label_view\n"
        assert _label_references(leaking_import) == {"label_view"}
        direct_read = 'json.load(open("labels.json"))\n'
        assert _label_references(direct_read) == {"labels.json"}
        via_constant = "from benchmarks.cwru.records import LABELS_PATH\n"
        assert _label_references(via_constant) == {"LABELS_PATH"}


# ---------------------------------------------------------------------------
# Blindness leg 4: SUT call-boundary guard (the boundary AE3 is defined on)
# ---------------------------------------------------------------------------

#: Exact keyword set the runner passes to diagnose_vibration. A new
#: argument must be reviewed for label leakage before being added here.
EXPECTED_SUT_KWARGS = frozenset(
    {
        "fs",
        "rpm",
        "signal_id",
        "bearing_id",
        "machine_group",
        "support_type",
        "signal_unit",
    }
)


def _assert_call_blind(args: tuple, kwargs: dict, record: OpsRecord) -> None:
    """Refuse a captured SUT invocation that could carry a label channel.

    Checks, in order: exactly one positional argument (the signal array);
    the exact keyword-name set; an opaque ``signal_id``; and no keyword
    value whose string form contains the record's CWRU ``file_id`` or
    ``cache_filename`` (the label-encoding names the blind protocol
    exists to keep out). The signal array itself is checked by type and,
    in the integration test, by exact sample equality -- its float
    samples are exempt from the substring scan because digit runs like
    '105' occur in sample values by chance.

    Raises:
        ValueError: Naming the offending argument and the leak.
    """
    if len(args) != 1 or not isinstance(args[0], np.ndarray):
        raise ValueError(
            f"SUT call must receive exactly the signal array positionally, "
            f"got {len(args)} positional argument(s) -- review the runner's "
            f"diagnose_vibration call."
        )
    if set(kwargs) != EXPECTED_SUT_KWARGS:
        unexpected = sorted(set(kwargs) - EXPECTED_SUT_KWARGS)
        missing = sorted(EXPECTED_SUT_KWARGS - set(kwargs))
        raise ValueError(
            f"SUT call keyword set diverged: unexpected {unexpected}, "
            f"missing {missing} -- a new argument must be reviewed for "
            f"label leakage and added to EXPECTED_SUT_KWARGS deliberately."
        )
    signal_id = str(kwargs["signal_id"])
    if not OPAQUE_ID_PATTERN.match(signal_id):
        raise ValueError(
            f"SUT call signal_id {signal_id!r} does not match "
            f"{OPAQUE_ID_PATTERN.pattern!r} -- only opaque sequential ids "
            f"may reach the system under test (AE3)."
        )
    fragments = (record.cache_filename, str(record.file_id))
    for name, value in kwargs.items():
        rendered = str(value)
        for fragment in fragments:
            if fragment in rendered:
                raise ValueError(
                    f"SUT call kwarg {name}={rendered!r} contains "
                    f"{fragment!r} (the record's CWRU identity) -- the "
                    f"system under test must never see file ids or cache "
                    f"filenames (AE3). Strip it from the runner's call."
                )


def _good_capture(record: OpsRecord) -> tuple[tuple, dict]:
    """A capture shaped exactly like the runner's real invocation."""
    return (
        (np.zeros(8),),
        {
            "fs": float(record.fs_hz),
            "rpm": float(record.nominal_rpm),
            "signal_id": record.opaque_id,
            "bearing_id": runner.BEARING_ID,
            "machine_group": runner.MACHINE_GROUP,
            "support_type": runner.SUPPORT_TYPE,
            "signal_unit": SIGNAL_UNIT,
        },
    )


class TestSutCallBoundary:
    """What actually crosses into diagnose_vibration is label-free."""

    def test_runner_invocations_pass_the_boundary_guard(
        self, imported, repo, monkeypatch
    ):
        """Intercept the real calls over 2 synthetic records: exact kwarg
        set, opaque signal_id, no file_id/cache_filename in any value,
        and the signal is exactly the imported samples."""
        captures = []
        real = runner.diagnose_vibration

        def capturing(*args, **kwargs):
            captures.append((args, kwargs))
            return real(*args, **kwargs)

        monkeypatch.setattr(runner, "diagnose_vibration", capturing)
        outcomes = runner.run_records(imported["records"], repository=repo)

        assert len(captures) == 2, "expected one SUT call per record"
        by_id = {kwargs["signal_id"]: (args, kwargs) for args, kwargs in captures}
        for record in imported["records"]:
            args, kwargs = by_id[record.opaque_id]
            _assert_call_blind(args, kwargs, record)  # must not raise
            np.testing.assert_array_equal(
                args[0], imported["signals"][record.opaque_id]
            )
        for outcome in outcomes.values():
            assert outcome["status"] == runner.OUTCOME_STATUS_OK

    def test_checker_green_on_a_well_formed_capture(self):
        """Green control: the mutation reds below are the guard tripping,
        not the harness."""
        record = _record("cwru_001", 105, "X105_DE_time")
        args, kwargs = _good_capture(record)
        _assert_call_blind(args, kwargs, record)  # must not raise

    def test_injected_leaking_kwarg_turns_the_guard_red(self):
        """Mutation: a wrapper that passes cache_filename through goes red."""
        record = _record("cwru_001", 105, "X105_DE_time")
        args, kwargs = _good_capture(record)
        kwargs["cache_filename"] = record.cache_filename
        with pytest.raises(ValueError, match="cache_filename"):
            _assert_call_blind(args, kwargs, record)

    def test_leaking_value_in_an_allowed_kwarg_turns_the_guard_red(self):
        """Mutation: the leak hides inside a legitimate kwarg's value."""
        record = _record("cwru_001", 105, "X105_DE_time")
        args, kwargs = _good_capture(record)
        kwargs["signal_unit"] = f"g ({record.cache_filename})"
        with pytest.raises(ValueError, match=re.escape("105.mat")):
            _assert_call_blind(args, kwargs, record)

    def test_file_id_substring_in_a_value_turns_the_guard_red(self):
        record = _record("cwru_001", 105, "X105_DE_time")
        args, kwargs = _good_capture(record)
        kwargs["support_type"] = "rigid-105"
        with pytest.raises(ValueError, match="support_type"):
            _assert_call_blind(args, kwargs, record)

    def test_non_opaque_signal_id_turns_the_guard_red(self):
        record = _record("cwru_001", 105, "X105_DE_time")
        args, kwargs = _good_capture(record)
        kwargs["signal_id"] = "OR007_at_6"
        with pytest.raises(ValueError, match="opaque"):
            _assert_call_blind(args, kwargs, record)


# ---------------------------------------------------------------------------
# Publication drift guard: slot binding on tmp fixtures (mutation included)
# ---------------------------------------------------------------------------

#: Small hand-written stand-in for results.json: every value type the
#: slot convention must render (int, float, str, bool, null, list index).
RESULTS_FIXTURE = {
    "metadata": {"git_describe": "v0.12.0-7-gabc1234"},
    "headline": {
        "strata_included": ["Y1", "Y2"],
        "n_records": 4,
        "frequency_detection": {"hits": 3, "total": 4, "rate": 0.75},
        "published": True,
        "note": None,
    },
}

#: A section whose slots all match RESULTS_FIXTURE (the green baseline).
MATCHING_SECTION = (
    "Frequency detection: "
    "<!-- slot: headline.frequency_detection.hits -->3<!-- /slot -->/"
    "<!-- slot: headline.frequency_detection.total -->4<!-- /slot --> ("
    "<!-- slot: headline.frequency_detection.rate pct1 -->75.0<!-- /slot -->%)\n"
    "Raw rate: <!-- slot: headline.frequency_detection.rate -->0.75<!-- /slot -->\n"
    "First stratum: <!-- slot: headline.strata_included.0 -->Y1<!-- /slot -->\n"
    "Published: <!-- slot: headline.published -->true<!-- /slot -->, "
    "note <!-- slot: headline.note -->null<!-- /slot -->\n"
    "Measured tree: <!-- slot: metadata.git_describe -->v0.12.0-7-gabc1234"
    "<!-- /slot -->\n"
)


def _write_results(tmp_path: Path) -> Path:
    results = tmp_path / "results.json"
    results.write_text(json.dumps(RESULTS_FIXTURE), encoding="utf-8")
    return results


def _write_document(tmp_path: Path, section_body: str, name: str = "README.md") -> Path:
    document = tmp_path / name
    document.write_text(
        "# Fixture readme\n\nProse before.\n\n"
        f"{drift_guard.SECTION_START}\n{section_body}{drift_guard.SECTION_END}\n"
        "\nProse after.\n",
        encoding="utf-8",
    )
    return document


class TestDriftGuardOnFixtures:
    """The full U6 test matrix for the slot-binding checker."""

    def test_matching_slots_are_green(self, tmp_path):
        document = _write_document(tmp_path, MATCHING_SECTION)
        verified = drift_guard.check_document(document, _write_results(tmp_path))
        assert verified == 8

    def test_one_altered_slot_turns_the_guard_red(self, tmp_path):
        """Mutation: a single stale value fails, named with expected and
        found text (set-membership checking would miss collisions)."""
        drifted = MATCHING_SECTION.replace(
            "hits -->3<!-- /slot -->", "hits -->2<!-- /slot -->"
        )
        assert drifted != MATCHING_SECTION
        document = _write_document(tmp_path, drifted)
        with pytest.raises(ValueError) as excinfo:
            drift_guard.check_document(document, _write_results(tmp_path))
        message = str(excinfo.value)
        assert "headline.frequency_detection.hits" in message
        assert "'2'" in message  # found
        assert "'3'" in message  # expected

    def test_drifted_pct_slot_turns_the_guard_red(self, tmp_path):
        drifted = MATCHING_SECTION.replace(
            "pct1 -->75.0<!-- /slot -->", "pct1 -->99.9<!-- /slot -->"
        )
        document = _write_document(tmp_path, drifted)
        with pytest.raises(ValueError, match="99.9"):
            drift_guard.check_document(document, _write_results(tmp_path))

    def test_slot_naming_a_missing_key_path_turns_the_guard_red(self, tmp_path):
        section = "<!-- slot: headline.no_such_key -->1<!-- /slot -->\n"
        document = _write_document(tmp_path, section)
        with pytest.raises(ValueError, match="no_such_key"):
            drift_guard.check_document(document, _write_results(tmp_path))

    def test_section_without_artifact_turns_the_guard_red(self, tmp_path):
        document = _write_document(tmp_path, MATCHING_SECTION)
        absent = tmp_path / "results.json"  # never written
        with pytest.raises(ValueError, match="results.json") as excinfo:
            drift_guard.check_document(document, absent)
        assert "benchmarks.cwru all" in str(excinfo.value)  # remedy named

    def test_artifact_without_section_is_green_prepublication(self, tmp_path):
        document = tmp_path / "README.md"
        document.write_text("# No benchmark section here\n", encoding="utf-8")
        assert drift_guard.check_document(document, _write_results(tmp_path)) == 0

    def test_absent_document_is_green(self, tmp_path):
        absent = tmp_path / "does-not-exist.md"
        assert drift_guard.check_document(absent, _write_results(tmp_path)) == 0

    def test_section_with_zero_slots_turns_the_guard_red(self, tmp_path):
        document = _write_document(tmp_path, "Accuracy is 99% (trust me).\n")
        with pytest.raises(ValueError, match="no slots"):
            drift_guard.check_document(document, _write_results(tmp_path))

    def test_malformed_slot_turns_the_guard_red(self, tmp_path):
        section = "<!-- slot: headline.n_records -->4\n"  # close marker missing
        document = _write_document(tmp_path, section)
        with pytest.raises(ValueError, match="malformed"):
            drift_guard.check_document(document, _write_results(tmp_path))

    def test_unbalanced_section_markers_turn_the_guard_red(self, tmp_path):
        document = tmp_path / "README.md"
        document.write_text(
            f"{drift_guard.SECTION_START}\nno end marker\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="marker"):
            drift_guard.check_document(document, _write_results(tmp_path))

    def test_duplicate_sections_turn_the_guard_red(self, tmp_path):
        document = tmp_path / "README.md"
        document.write_text(
            f"{drift_guard.SECTION_START}\n{drift_guard.SECTION_END}\n"
            f"{drift_guard.SECTION_START}\n{drift_guard.SECTION_END}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="exactly one"):
            drift_guard.check_document(document, _write_results(tmp_path))

    def test_slot_bound_to_a_non_scalar_turns_the_guard_red(self, tmp_path):
        section = "<!-- slot: headline -->stuff<!-- /slot -->\n"
        document = _write_document(tmp_path, section)
        with pytest.raises(ValueError, match="scalar"):
            drift_guard.check_document(document, _write_results(tmp_path))

    def test_pct_on_a_non_numeric_value_turns_the_guard_red(self, tmp_path):
        section = "<!-- slot: metadata.git_describe pct1 -->1.0<!-- /slot -->\n"
        document = _write_document(tmp_path, section)
        with pytest.raises(ValueError, match="numeric"):
            drift_guard.check_document(document, _write_results(tmp_path))

    def test_publication_sweep_reaches_every_document(self, tmp_path):
        """check_publication sweeps ALL documents: a drifted slot is
        caught whether it sits in the second document (proving the sweep
        does not stop after the first) or in the first."""
        results = _write_results(tmp_path)
        drifted_section = MATCHING_SECTION.replace(
            "hits -->3<!-- /slot -->", "hits -->2<!-- /slot -->"
        )
        assert drifted_section != MATCHING_SECTION
        clean = _write_document(tmp_path, MATCHING_SECTION, name="clean.md")
        drifted = _write_document(tmp_path, drifted_section, name="drifted.md")

        # Drift in the SECOND document of the sweep.
        with pytest.raises(ValueError) as excinfo:
            drift_guard.check_publication(
                document_paths=(clean, drifted), results_path=results
            )
        message = str(excinfo.value)
        assert "headline.frequency_detection.hits" in message  # slot named
        assert "drifted.md" in message  # offending document named

        # Drift in the FIRST document raises just the same.
        with pytest.raises(ValueError, match="headline.frequency_detection.hits"):
            drift_guard.check_publication(
                document_paths=(drifted, clean), results_path=results
            )

    def test_publication_sweep_green_reports_counts_per_document(self, tmp_path):
        """Green counterpart: a clean two-document sweep verifies every
        slot in BOTH documents and reports the per-document counts."""
        results = _write_results(tmp_path)
        first = _write_document(tmp_path, MATCHING_SECTION, name="one.md")
        second = _write_document(tmp_path, MATCHING_SECTION, name="two.md")
        counts = drift_guard.check_publication(
            document_paths=(first, second), results_path=results
        )
        assert counts == {str(first): 8, str(second): 8}


class TestDriftGuardOnTheRealRepo:
    """The guard runs against the REAL documents and artifact path, so it
    activates automatically the moment U7 publishes a section."""

    def test_publication_state_is_consistent_right_now(self):
        """Green today (pre-publication: no section anywhere) and stays
        the binding check after U7 publishes -- it must never be skipped
        or pinned to the current zero-slot state."""
        counts = drift_guard.check_publication()  # must not raise
        assert set(counts) == {str(path) for path in drift_guard.DEFAULT_DOCUMENT_PATHS}

    def test_default_paths_agree_with_the_rest_of_the_benchmark(self):
        """drift_guard stays stdlib-only, so its path constants are local
        duplicates -- pinned here against the canonical ones so the two
        derivations cannot drift apart silently."""
        assert drift_guard.DEFAULT_RESULTS_PATH == scorer.DEFAULT_RESULTS_PATH
        assert drift_guard.REPO_ROOT == runner.REPO_ROOT
        readme, methodology = drift_guard.DEFAULT_DOCUMENT_PATHS
        assert readme == drift_guard.REPO_ROOT / "README.md"
        assert readme.exists(), "the README the guard sweeps must exist"
        assert methodology.name == "benchmark-methodology.md"


# ---------------------------------------------------------------------------
# CI wiring: the Black format gate covers benchmarks/
# ---------------------------------------------------------------------------


class TestFormatGateCoversBenchmarks:
    """The workflow's Black check includes the benchmark package.

    Blockingness of the job (no continue-on-error) is asserted by
    tests/test_ci_gates.py; this test pins the SCOPE so benchmarks/
    cannot silently rot out of the format gate like bench_phase1.py
    rotted out of the import graph.
    """

    def test_black_check_scope_includes_benchmarks(self):
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text(
                encoding="utf-8"
            )
        )
        steps = workflow["jobs"]["format-check"]["steps"]
        assert steps, "format-check job parsed with no steps"
        script = "\n".join(step.get("run", "") for step in steps)
        assert "black --check" in script
        assert re.search(r"black --check[^\n]*\bbenchmarks\b", script), (
            "the Black gate no longer covers benchmarks/ -- restore "
            "'black --check src tests benchmarks' in tests.yml so the "
            "benchmark package stays format-gated like src and tests."
        )
