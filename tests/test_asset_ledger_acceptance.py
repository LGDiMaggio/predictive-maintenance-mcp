"""Acceptance test of the asset health ledger through the MCP tools (U8).

The plan's acceptance expectations AE1-AE9 (and the generic part of AE10)
are crossed end to end: every scene is driven ONLY through the registered
tools (``load_signal``, ``declare_measurement_point``,
``declare_healthy_baseline``, ``get_asset_history``, ``assess_asset_change``,
``diagnose_vibration``) and the deterministic synthetic sequence of
``tests/_synthetic_sequence.py`` (thirty CSV measurements with companions:
M01-M10 stable, M11 an isolated spike, M12-M20 stable, M21-M30 a progressive
ramp with BPFO modulation of a 6205 at 1800 rpm; ten companions without
``rpm``).

Scenes and the expectations they cover
--------------------------------------
* ``story`` fixture (one ledger, run once per module, a checkpoint per
  scene): the point declared once (setup) -> M01 loaded, history of one,
  ``diagnose_vibration`` without and with an explicit rpm (AE1) -> M02..M11
  loaded, assessment (AE8: unconfirmed single acquisition) -> M12 loaded,
  assessment (AE8: isolated episode) -> M13..M15 loaded, simulated restart
  (the in-memory repository emptied; every tool call opens its own
  ``LedgerStore`` anyway), M16..M30 loaded, history of thirty (AE2) ->
  assessment (AE8: persistent change with BPFO evidence; AE3: the rpm-less
  measurements participate qualified; AE9: the structure of the response)
  -> a measurement declared in mm/s loaded and assessed (AE4: recorded,
  excluded with the reason; AE6: the automatic window is never called
  healthy) -> a baseline declared on M01..M05 by "L. Rossi" and assessed
  (AE7).
* Independent tests on fresh directories: byte identity of two generations
  (R19); snapshots of a previous lineage injected in the raw ledger for
  M01..M20 and the algorithm version bumped before M21..M30 are loaded, so
  no lineage covers the whole set, then bounded re-processing (AE5); the
  same measurements as raw float32 files with companions, then the CSV twin
  of one of them (AE10, generic path); out-of-order loading; the same scene
  run twice with canonical payloads compared (determinism).
"""

import asyncio
import copy
import hashlib
import inspect
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import AsyncMock

import numpy as np
import pytest
from mcp.server.mcpserver import MCPServer

from _synthetic_sequence import (
    ACQUISITION_INTERVAL,
    ASSET_ID,
    BEARING_ID,
    DEFAULT_SEED,
    DIRECTION,
    FIRST_ACQUIRED_AT,
    INDICES_WITHOUT_RPM,
    NOMINAL_RPM,
    POINT_ID,
    SENSOR_ID,
    SEQUENCE_LENGTH,
    SIGNAL_UNIT,
    acquired_at,
    build_measurement_sequence,
    companion_metadata,
    companion_path,
    synthesize_measurement,
)
from conftest import patch_data_dir, write_raw_file
from predictive_maintenance_mcp.asset_ledger import snapshot as snapshot_module
from predictive_maintenance_mcp.asset_ledger.service import MAX_REPROCESS_PER_CALL
from predictive_maintenance_mcp.asset_ledger.snapshot import processing_id, snapshot_id
from predictive_maintenance_mcp.asset_ledger.store import (
    EVENT_HEALTH_SNAPSHOT_COMPUTED,
    EVENT_MEASUREMENT_RECORDED,
    LedgerStore,
    canonical_json,
    make_event,
)
from predictive_maintenance_mcp.config import get_ledger_dir
from predictive_maintenance_mcp.mcp_tools import (
    acquisition_tools,
    asset_tools,
    diagnostics_tools,
)
from predictive_maintenance_mcp.models import (
    AssetChangeAssessment,
    AssetHistoryResult,
    BaselineDeclarationResult,
    DiagnosisResult,
    MeasurementPointDeclarationResult,
    StoredSignalInfo,
)
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

#: Who declares the point in every scene.
DECLARER = "acceptance-test"

#: Who declares the healthy baseline (AE7): cited verbatim by the assessment.
BASELINE_DECLARER = "L. Rossi"

#: The tool's defaults, spelled out where a scene depends on them.
REFERENCE_WINDOW = 10
LAST_K = 5

#: Indices of the story's stages.
FIRST = 1
SPIKE = 11
RETURN = 12
RESTART_AFTER = 15
BASELINE_MEMBERS = range(1, 6)

#: The 31st measurement of the point, declared in a velocity unit (AE4).
EXTRA_UNIT_NAME = "extra_velocity.csv"
EXTRA_UNIT = "mm/s"

#: Subset re-exported as raw float32 files (AE10, generic path).
RAW_SUBSET = (5, 6, 7)

#: Load order of the out-of-order scene, then the rest of the first eleven.
OUT_OF_ORDER = (5, 3, 1)
OUT_OF_ORDER_REST = (2, 4, 6, 7, 8, 9, 10, 11)

#: How the three persistence rules of the assessment name themselves in the
#: criterion (consecutive run, k-of-n on the same side, significant drift).
RULE_PATTERNS = (
    r"\d+ consecutive acquisitions outside the band",
    r"\d+ of last \d+ acquisitions outside the band on the same side",
    r"significant drift \(p=",
)

#: Lineage of the snapshots injected for AE5: the shape of a processing_id
#: (family/algorithm version+16 hex of the policy) with a version and a
#: policy hash no current code produces, as a previous release would have
#: written them.
INJECTED_LINEAGE = "health_snapshot/0+deadbeefdeadbeef"

#: Keys dropped before two payloads of the same scene are compared: server
#: instants and host provenance are the only values two runs may legitimately
#: differ in. Nothing else is stripped.
VOLATILE_KEYS = frozenset({"recorded_at", "declared_at", "provenance"})

#: The exact re-processing call the tools name as the remedy.
REPROCESS_CALL = (
    f"assess_asset_change(asset_id={ASSET_ID!r}, "
    f"measurement_point_id={POINT_ID!r}, reprocess=True)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _instant(value: str) -> datetime:
    """An ISO 8601 string as an aware instant (``+01:00`` equals its UTC twin)."""
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None, value
    return parsed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digests(directory: Path) -> dict[str, str]:
    """Every file of *directory* by name, with the SHA-256 of its bytes."""
    return {p.name: _sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}


def _keys(tree: Any) -> set[str]:
    """Every dict key in a nested structure."""
    found: set[str] = set()
    if isinstance(tree, dict):
        for key, value in tree.items():
            found.add(key)
            found |= _keys(value)
    elif isinstance(tree, (list, tuple)):
        for item in tree:
            found |= _keys(item)
    return found


def _without_volatile(tree: Any, found: set[str]) -> Any:
    """*tree* without the ``VOLATILE_KEYS`` (the names met go to *found*)."""
    if isinstance(tree, dict):
        kept: dict[str, Any] = {}
        for key, value in tree.items():
            if key in VOLATILE_KEYS:
                found.add(key)
                continue
            kept[key] = _without_volatile(value, found)
        return kept
    if isinstance(tree, list):
        return [_without_volatile(item, found) for item in tree]
    return tree


def _one_sentence(text: str) -> bool:
    """One sentence: one line, ends with a period, no sentence break inside."""
    return text.endswith(".") and "\n" not in text and not re.search(r"[.!?]\s", text)


def sequence_name(index: int) -> str:
    """File name (and data-directory-relative location) of measurement *index*."""
    return f"seq_m{index:02d}.csv"


def _assert_well_formed(model: Any) -> dict[str, Any]:
    """The dump is JSON without NaN and names no key ``error`` or
    ``recommendations`` (a structured response, not a decision)."""
    dumped = model.model_dump()
    names = _keys(dumped)
    assert "error" not in names
    assert "recommendations" not in names
    json.dumps(dumped, allow_nan=False)
    return dumped


def _snapshot_lineage_counts(asset_id: str) -> dict[str, int]:
    """``health_snapshot_computed`` events per ``processing_id`` in the raw
    ledger of *asset_id* (what is on disk, not what a view selects)."""
    counts: dict[str, int] = {}
    for event in LedgerStore(get_ledger_dir()).read(asset_id).events:
        if event["event_type"] != EVENT_HEALTH_SNAPSHOT_COMPUTED:
            continue
        lineage = event["payload"]["processing"]["processing_id"]
        counts[lineage] = counts.get(lineage, 0) + 1
    return counts


def _inject_older_lineage(measurement_ids: Iterable[str], lineage: str) -> int:
    """Append, for every measurement, a ``health_snapshot_computed`` event
    copied from its latest snapshot in the raw ledger but filed under
    *lineage* (what a previous release would have left behind: the same
    numbers, another processing id, its own snapshot id). Returns the
    number of events appended."""
    store = LedgerStore(get_ledger_dir())
    latest: dict[str, dict[str, Any]] = {}
    for event in store.read(ASSET_ID).events:
        if event["event_type"] == EVENT_HEALTH_SNAPSHOT_COMPUTED:
            latest[event["payload"]["measurement_id"]] = event["payload"]
    appended = 0
    for measurement_id in measurement_ids:
        payload = copy.deepcopy(latest[measurement_id])
        payload["processing"] = {
            **payload["processing"],
            "processing_id": lineage,
            "algorithm_version": 0,
        }
        payload["snapshot_id"] = snapshot_id(
            measurement_id, lineage, payload["context_digest"]
        )
        event = make_event(EVENT_HEALTH_SNAPSHOT_COMPUTED, ASSET_ID, payload)
        assert store.append(ASSET_ID, event).appended
        appended += 1
    return appended


def _write_companion(path: Path, companion: dict[str, Any]) -> None:
    companion_path(path).write_text(
        json.dumps(companion, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _write_extra_unit_measurement(directory: Path) -> Path:
    """A measurement of the same point declared in ``mm/s`` (AE4).

    Distinct bytes (its own ``measurement_id``), its own acquisition slot
    one week after M30, rpm declared: the unit is the ONLY contradiction.
    """
    path = directory / EXTRA_UNIT_NAME
    values = synthesize_measurement(FIRST, seed=DEFAULT_SEED + 7919)
    with path.open("w", encoding="ascii", newline="\n") as fh:
        np.savetxt(fh, values, fmt="%.8f")
    companion = companion_metadata(FIRST)
    companion["signal_unit"] = EXTRA_UNIT
    companion["measurement"]["acquired_at"] = (
        FIRST_ACQUIRED_AT + SEQUENCE_LENGTH * ACQUISITION_INTERVAL
    ).isoformat()
    _write_companion(path, companion)
    return path


# ---------------------------------------------------------------------------
# Tool drivers (the only way the scenes touch the ledger)
# ---------------------------------------------------------------------------


async def _declare_point(tools, ctx) -> MeasurementPointDeclarationResult:
    return await tools["declare_measurement_point"](
        ctx=ctx,
        asset_id=ASSET_ID,
        measurement_point_id=POINT_ID,
        bearing_id=BEARING_ID,
        machine_group=2,
        support_type="rigid",
        expected_signal_unit=SIGNAL_UNIT,
        expected_direction=DIRECTION,
        nominal_rpm=NOMINAL_RPM,
        declared_by=DECLARER,
    )


async def _load(tools, ctx, sequence: list[Path], indices: Iterable[int]) -> list:
    """Batch-load the measurements *indices* (1-based) of the sequence."""
    names = [sequence[index - 1].name for index in indices]
    infos = await tools["load_signal"](ctx=ctx, filepath=names)
    assert isinstance(infos, list) and len(infos) == len(names)
    return infos


async def _assess(tools, ctx, **kwargs) -> AssetChangeAssessment:
    """Nothing but the asset and the point (AE9), unless a scene says more."""
    return await tools["assess_asset_change"](
        ctx=ctx, asset_id=ASSET_ID, measurement_point_id=POINT_ID, **kwargs
    )


async def _history(tools, ctx, **kwargs) -> AssetHistoryResult:
    return await tools["get_asset_history"](ctx=ctx, asset_id=ASSET_ID, **kwargs)


def _oldest_first(history: AssetHistoryResult) -> list[dict[str, Any]]:
    """The listed measurements in acquisition order (the tool lists newest
    first)."""
    assert history.status == "found", history.message
    return list(reversed(history.asset["measurements"]))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sequence(tmp_path_factory) -> list[Path]:
    """One generation of the thirty measurements, shared by the module."""
    return build_measurement_sequence(tmp_path_factory.mktemp("acceptance-sequence"))


@pytest.fixture(scope="module")
def tools():
    """The registered tools the scenes drive, by name."""
    server = MCPServer("test-asset-ledger-acceptance")
    acquisition_tools.register(server)
    asset_tools.register(server)
    diagnostics_tools.register(server)
    return {t.name: t.fn for t in server._tool_manager._tools.values()}


@pytest.fixture(scope="module")
def ctx():
    return AsyncMock()


@pytest.fixture
def clean_repo():
    repo = get_repository()
    repo.clear_all()
    yield repo
    repo.clear_all()


@pytest.fixture
def sequence_data_dir(sequence, monkeypatch, clean_repo, ledger_dir) -> Path:
    """The generated sequence as the data directory, an empty repository and
    the fresh ledger directory of the test."""
    directory = sequence[0].parent
    patch_data_dir(monkeypatch, directory)
    return directory


async def _run_story(tools, ctx, sequence: list[Path]) -> dict[str, Any]:
    """The progressive scenes, in order; a checkpoint after each."""
    checkpoints: dict[str, Any] = {"ids": {}, "infos": {}}
    ids: dict[int, str] = checkpoints["ids"]

    async def load(indices: Iterable[int]) -> list[StoredSignalInfo]:
        indices = list(indices)
        infos = await _load(tools, ctx, sequence, indices)
        for index, info in zip(indices, infos):
            ids[index] = info.measurement["measurement_id"]
            checkpoints["infos"][index] = info
        return infos

    # Setup: the point, declared once with the catalog bearing.
    declared = await _declare_point(tools, ctx)
    checkpoints["declaration"] = declared
    if declared.bearing_in_catalog is not True:
        raise AssertionError(
            f"bearing {BEARING_ID!r} is not in the verified catalog "
            f"(bearing_in_catalog={declared.bearing_in_catalog!r}); the story "
            f"needs the catalog route, and a second declaration with "
            f"fault_orders would qualify every measurement acquired before it "
            f"(point_declaration_changed)."
        )

    # AE1: the first measurement, its history, the diagnosis defaults.
    (first,) = await load([FIRST])
    checkpoints["history_m01"] = await _history(tools, ctx)
    checkpoints["diagnosis_default"] = await tools["diagnose_vibration"](
        ctx=ctx, signal_id=first.signal_id
    )
    checkpoints["diagnosis_explicit"] = await tools["diagnose_vibration"](
        ctx=ctx, signal_id=first.signal_id, rpm=NOMINAL_RPM
    )

    # AE8, first scene: ten stable then the spike.
    await load(range(FIRST + 1, SPIKE + 1))
    checkpoints["assess_m11"] = await _assess(tools, ctx)

    # AE8, second scene: the return inside the band.
    await load([RETURN])
    checkpoints["assess_m12"] = await _assess(tools, ctx)

    # AE2: a restart between M15 and M16. The in-memory repository is
    # emptied (what a server restart loses); the ledger on disk is the only
    # memory that survives, and every tool call opens its own LedgerStore.
    await load(range(RETURN + 1, RESTART_AFTER + 1))
    checkpoints["restart_cleared"] = get_repository().clear_all()
    await load(range(RESTART_AFTER + 1, SEQUENCE_LENGTH + 1))
    checkpoints["history_m30"] = await _history(
        tools, ctx, max_measurements=SEQUENCE_LENGTH
    )

    # AE8 third scene, AE3, AE9: the ramp, assessed with nothing but the
    # asset and the point.
    checkpoints["assess_m30"] = await _assess(tools, ctx)

    # AE4 (and AE6 wording): a measurement declared in mm/s.
    extra = _write_extra_unit_measurement(sequence[0].parent)
    checkpoints["extra_info"] = await tools["load_signal"](ctx=ctx, filepath=extra.name)
    checkpoints["assess_extra"] = await _assess(tools, ctx)

    # AE7: a declared healthy baseline on M01..M05.
    checkpoints["baseline"] = await tools["declare_healthy_baseline"](
        ctx=ctx,
        asset_id=ASSET_ID,
        measurement_point_id=POINT_ID,
        measurement_ids=[ids[index] for index in BASELINE_MEMBERS],
        declared_by=BASELINE_DECLARER,
    )
    checkpoints["assess_baseline"] = await _assess(tools, ctx)
    checkpoints["history_final"] = await _history(
        tools, ctx, max_measurements=SEQUENCE_LENGTH + 1
    )
    return checkpoints


@pytest.fixture(scope="module")
def story(tmp_path_factory, tools, ctx, sequence) -> dict[str, Any]:
    """The progressive scenes run once on their own ledger; the checkpoints.

    The ledger directory and the data directory are pinned for the duration
    of the run only (a module-scoped MonkeyPatch context), so the autouse
    per-test ledger of the other tests is untouched.
    """
    ledger_root = tmp_path_factory.mktemp("acceptance-ledger")
    repo = get_repository()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PMM_LEDGER_DIR", str(ledger_root))
        patch_data_dir(patch, sequence[0].parent)
        repo.clear_all()
        try:
            checkpoints = asyncio.run(_run_story(tools, ctx, sequence))
        finally:
            repo.clear_all()
    checkpoints["ledger_root"] = ledger_root
    return checkpoints


@pytest.fixture(scope="module")
def rms_by_id(story) -> dict[str, float]:
    """RMS of every recorded measurement, from the history's snapshot
    preview (the numbers the assessment's statistics are built on)."""
    return {
        entry["measurement_id"]: float(entry["indicators"]["rms"])
        for entry in story["history_final"].asset["measurements"]
    }


# ---------------------------------------------------------------------------
# Generator determinism (R19)
# ---------------------------------------------------------------------------


def test_two_generations_with_the_same_seed_are_byte_identical(tmp_path, sequence):
    first = build_measurement_sequence(tmp_path / "first")
    second = build_measurement_sequence(tmp_path / "second", seed=DEFAULT_SEED)

    assert [p.name for p in first] == [p.name for p in second]
    digests = _digests(first[0].parent)
    assert len(digests) == 2 * SEQUENCE_LENGTH  # a CSV and a companion each
    assert digests == _digests(second[0].parent)
    # The module's shared generation is the same bytes (the story may have
    # added its extra file next to them, hence the restriction by name).
    shared = _digests(sequence[0].parent)
    assert {name: shared[name] for name in digests} == digests


# ---------------------------------------------------------------------------
# The story: setup and AE1, AE2, AE3, AE4, AE6, AE7, AE8, AE9
# ---------------------------------------------------------------------------


def test_setup_the_point_is_declared_once_with_the_catalog_bearing(story):
    declared = story["declaration"]
    assert isinstance(declared, MeasurementPointDeclarationResult)
    assert declared.declaration_version == 1 and declared.appended is True
    assert declared.bearing_in_catalog is True
    declaration = declared.declaration
    assert declaration["bearing_id"] == BEARING_ID
    assert declaration["fault_orders"] is None
    assert declaration["machine_group"] == 2
    assert declaration["support_type"] == "rigid"
    assert declaration["expected_signal_unit"] == SIGNAL_UNIT
    assert declaration["expected_direction"] == DIRECTION
    assert declaration["nominal_rpm"] == float(NOMINAL_RPM)
    assert declaration["declared_by"] == DECLARER
    _assert_well_formed(declared)
    # Declared once: the whole story runs at declaration version 1, so no
    # measurement is ever qualified against a changed context.
    point = story["history_final"].asset["point_declarations"][POINT_ID]
    assert point["current"]["declaration_version"] == 1
    assert len(point["history"]) == 1


def test_ae1_the_first_measurement_is_in_the_history_and_diagnosable(story):
    """AE1: M01 loaded -> a history of one; its signal_id drives
    diagnose_vibration without an explicit rpm (the measurement's) and with
    one (explicit wins), the bearing and the ISO context from the point."""
    info = story["infos"][FIRST]
    assert isinstance(info, StoredSignalInfo)
    assert info.signal_id == "seq_m01"
    assert info.measurement["ledger_status"] == "recorded"
    assert info.measurement["snapshot_status"] == "complete"
    assert info.measurement["comparability"]["grade"] == "comparable"

    history = story["history_m01"]
    assert isinstance(history, AssetHistoryResult)
    assert history.status == "found" and history.truncated is False
    assert history.asset["asset_id"] == ASSET_ID
    assert history.asset["measurement_count"] == 1
    (entry,) = history.asset["measurements"]
    assert entry["measurement_id"] == story["ids"][FIRST]
    assert entry["measurement_point_id"] == POINT_ID
    assert _instant(entry["acquired_at"]) == _instant(acquired_at(FIRST))
    assert entry["signal_id"] == info.signal_id
    assert entry["location"] == sequence_name(FIRST)
    assert entry["rpm"] == NOMINAL_RPM
    assert entry["lineages"] == [processing_id()]
    assert entry["snapshot_count"] == 1

    default = story["diagnosis_default"]
    assert isinstance(default, DiagnosisResult)
    assert default.signal_id == info.signal_id
    assert default.rpm == float(NOMINAL_RPM)
    assert default.bearing_id == BEARING_ID
    assert default.machine_group == 2 and default.support_type == "rigid"
    assert default.parameter_sources == {
        "rpm": "measurement",
        "bearing_id": "point",
        "machine_group": "point",
        "support_type": "point",
    }
    assert default.bearing_faults is not None
    assert default.bearing_faults.bearing_id == BEARING_ID
    assert default.iso_severity.status == "assessed"

    explicit = story["diagnosis_explicit"]
    assert explicit.rpm == float(NOMINAL_RPM)
    assert explicit.parameter_sources["rpm"] == "explicit"
    assert explicit.parameter_sources["bearing_id"] == "point"
    assert explicit.fft_summary == default.fft_summary
    assert explicit.bearing_faults == default.bearing_faults


def test_ae2_the_history_after_a_restart_lists_all_thirty_in_order(story):
    """AE2: M02..M15 loaded, the repository emptied, M16..M30 loaded -> the
    history shows the thirty measurements in acquired_at order, each with
    its snapshot lineage; nothing was lost or recorded twice."""
    assert story["restart_cleared"] == RESTART_AFTER  # M01..M15 were in memory
    assert {
        story["infos"][index].measurement["ledger_status"]
        for index in range(1, SEQUENCE_LENGTH + 1)
    } == {"recorded"}

    history = story["history_m30"]
    assert history.status == "found" and history.truncated is False
    asset = history.asset
    assert asset["measurement_count"] == SEQUENCE_LENGTH
    entries = _oldest_first(history)
    assert [e["measurement_id"] for e in entries] == [
        story["ids"][index] for index in range(1, SEQUENCE_LENGTH + 1)
    ]
    instants = [_instant(e["acquired_at"]) for e in entries]
    assert instants == [_instant(acquired_at(i)) for i in range(1, SEQUENCE_LENGTH + 1)]
    assert instants == sorted(instants)
    for index, entry in enumerate(entries, start=1):
        assert entry["measurement_point_id"] == POINT_ID
        assert entry["location"] == sequence_name(index)
        assert entry["lineages"] == [processing_id()]
        assert entry["snapshot_count"] == 1
        assert entry["indicators"]["rms"] > 0
        assert entry["rpm"] == (None if index in INDICES_WITHOUT_RPM else NOMINAL_RPM)
    (point,) = asset["summary"]["points"]
    assert point["measurement_point_id"] == POINT_ID
    assert point["measurement_count"] == SEQUENCE_LENGTH
    assert point["latest_lineage"] == processing_id()
    assert point["baseline_declared"] is False
    assert asset["integrity"]["unreadable_records"] == 0
    assert asset["reattributed"] == []


def test_ae8_a_single_spike_is_an_unconfirmed_acquisition(story):
    """AE8 after M11 alone: never called degradation; a repeat is asked."""
    result = story["assess_m11"]
    assert isinstance(result, AssetChangeAssessment)
    assert result.status == "assessed"
    assert result.reference["measurement_ids"] == [
        story["ids"][index] for index in range(1, REFERENCE_WINDOW + 1)
    ]
    assert result.observed["acquisitions_assessed"] == 1
    assert result.observed["latest_measurement_id"] == story["ids"][SPIKE]

    assessed = result.assessed
    assert assessed["classification"] == "unconfirmed_single_acquisition"
    assert assessed["direction"] == "increase"
    assert assessed["criterion"]
    assert assessed["onset_measurement_id"] == story["ids"][SPIKE]
    assert "rms" in assessed["indicators_driving"]
    assert result.derived["per_indicator"]["rms"]["classification"] == (
        "unconfirmed_single_acquisition"
    )
    verification = result.suggested_verification
    assert verification is not None and _one_sentence(verification)
    assert "repeat" in verification.lower()
    assert "same operating conditions" in verification
    assert "unconfirmed single acquisition" in result.message


def test_ae8_a_return_inside_the_band_is_an_isolated_episode(story):
    """AE8 after M12: the spike is an isolated episode, nothing to verify."""
    result = story["assess_m12"]
    assert result.status == "assessed"
    assert result.observed["acquisitions_assessed"] == 2
    assert result.observed["latest_measurement_id"] == story["ids"][RETURN]
    assessed = result.assessed
    assert assessed["classification"] == "isolated_episode"
    assert assessed["criterion"]
    assert result.derived["per_indicator"]["rms"]["classification"] == (
        "isolated_episode"
    )
    assert result.derived["deltas"]["rms"]["exceeds"] is False
    assert result.suggested_verification is None
    assert "isolated episode" in result.message


def test_ae8_the_ramp_is_a_persistent_increase_with_bpfo_evidence(story):
    """AE8 after M30: persistent change, direction increase, the criterion
    names one of the three rules, the onset lies in the ramp (never the
    spike), BPFO evidence in at least 3 of the last 5."""
    result = story["assess_m30"]
    assert result.status == "assessed"
    assert (
        result.observed["acquisitions_assessed"] == SEQUENCE_LENGTH - REFERENCE_WINDOW
    )
    assessed = result.assessed
    assert assessed["classification"] == "persistent_change"
    assert assessed["direction"] == "increase"
    criterion = assessed["criterion"]
    assert any(re.search(rule, criterion) for rule in RULE_PATTERNS), criterion
    assert assessed["indicators_driving"]
    ramp_ids = {story["ids"][index] for index in range(21, SEQUENCE_LENGTH + 1)}
    assert assessed["onset_measurement_id"] in ramp_ids
    assert assessed["onset_measurement_id"] != story["ids"][SPIKE]
    assert _instant(assessed["onset_acquired_at"]) >= _instant(acquired_at(21))
    evidence = assessed["evidence"]["BPFO"]
    assert evidence["k"] == LAST_K
    assert evidence["present_in_last_k"] >= 3
    assert evidence["classification"] == "persistent_change"
    presence = result.observed["evidence_presence"]["BPFO"]
    assert presence["present_in_last_k"] == evidence["present_in_last_k"]
    assert sum(presence["presence"]) >= 3
    assert "persistent change (increase)" in result.message


def test_ae3_the_rpm_less_measurements_participate_as_qualified(story):
    """AE3: the ten companions without rpm are in the trend, graded
    qualified with rpm_not_declared (constant regime assumed), inside the
    reference window as well as among the acquisitions assessed."""
    result = story["assess_m30"]
    comparability = result.comparability
    assert comparability["qualified"] >= 10
    assert comparability["qualified"] == len(INDICES_WITHOUT_RPM)
    assert comparability["comparable"] == SEQUENCE_LENGTH - len(INDICES_WITHOUT_RPM)
    assert comparability["non_comparable"] == 0
    (entry,) = [
        q for q in comparability["qualifications"] if q["code"] == "rpm_not_declared"
    ]
    assert entry["count"] == len(INDICES_WITHOUT_RPM)
    assert "constant regime" in entry["detail"]

    # Every slot is evaluated: the reference plus the twenty after it.
    assert result.observed["slots_assessed"] == SEQUENCE_LENGTH
    assert (
        result.observed["acquisitions_assessed"] == SEQUENCE_LENGTH - REFERENCE_WINDOW
    )
    assert result.observed["reference_statistics"]["rms"]["n"] == REFERENCE_WINDOW
    in_window = sorted(i for i in INDICES_WITHOUT_RPM if i <= REFERENCE_WINDOW)
    assert result.reference["qualification_codes"] == {
        "rpm_not_declared": len(in_window)
    }
    assert "rpm_not_declared" in result.reference["message"]
    listed = {
        m["measurement_id"]: m["grade"]
        for m in result.observed["measurement_ids_assessed"]
    }
    last = range(SEQUENCE_LENGTH - LAST_K + 1, SEQUENCE_LENGTH + 1)
    assert set(listed) == {story["ids"][index] for index in last}
    for index in last:
        expected = "qualified" if index in INDICES_WITHOUT_RPM else "comparable"
        assert listed[story["ids"][index]] == expected, index


def test_ae4_an_incompatible_unit_is_recorded_but_excluded_with_the_reason(story):
    """AE4: the mm/s measurement is recorded in the ledger, excluded from
    the assessment as unit_incompatible and listed as excluded; the
    assessment of the point does not move."""
    info = story["extra_info"]
    assert isinstance(info, StoredSignalInfo)
    assert info.signal_unit == EXTRA_UNIT
    block = info.measurement
    assert block["ledger_status"] == "recorded"
    assert block["snapshot_status"] == "complete"
    assert block["comparability"]["grade"] == "non_comparable"
    codes = [q["code"] for q in block["comparability"]["qualifications"]]
    assert "unit_incompatible" in codes
    extra_id = block["measurement_id"]
    assert extra_id not in story["ids"].values()

    result = story["assess_extra"]
    assert result.status == "assessed"
    comparability = result.comparability
    assert comparability["non_comparable"] == 1
    assert comparability["qualified"] == len(INDICES_WITHOUT_RPM)
    (excluded,) = comparability["excluded"]
    assert excluded["measurement_id"] == extra_id
    assert "unit_incompatible" in excluded["reasons"]
    (reason,) = [
        q for q in comparability["qualifications"] if q["code"] == "unit_incompatible"
    ]
    assert reason["count"] == 1 and EXTRA_UNIT in reason["detail"]
    assert extra_id not in result.reference["measurement_ids"]
    assert extra_id not in {
        m["measurement_id"] for m in result.observed["measurement_ids_assessed"]
    }
    assert result.observed["latest_measurement_id"] == story["ids"][SEQUENCE_LENGTH]
    assert (
        result.observed["acquisitions_assessed"] == SEQUENCE_LENGTH - REFERENCE_WINDOW
    )
    assert (
        result.assessed["classification"]
        == story["assess_m30"].assessed["classification"]
    )
    assert result.observed["latest"] == story["assess_m30"].observed["latest"]

    # Recorded: the history lists it, graded against the point.
    (entry,) = [
        e
        for e in story["history_final"].asset["measurements"]
        if e["measurement_id"] == extra_id
    ]
    assert entry["comparability"]["grade"] == "non_comparable"
    assert "unit_incompatible" in entry["comparability"]["codes"]
    assert entry["indicators"]["unit"] == EXTRA_UNIT


def test_ae6_the_automatic_window_is_never_called_healthy(story, rms_by_id):
    """AE6: before any baseline, the reference is the automatic window, its
    health is not declared, the wording says so and the word "healthy"
    appears nowhere in the payload; its statistics are those of the first
    ten acquisitions."""
    for key in ("assess_m11", "assess_m12", "assess_m30", "assess_extra"):
        result = story[key]
        reference = result.reference
        assert reference["kind"] == "automatic_window", key
        assert reference["health_declared"] is False, key
        assert "not declared" in reference["message"], key
        assert "relative comparison" in reference["message"], key
        assert reference["baseline"] is None and reference["withdrawn_baseline"] is None
        assert "healthy" not in canonical_json(result.model_dump()).lower(), key

    result = story["assess_extra"]
    window = [story["ids"][index] for index in range(1, REFERENCE_WINDOW + 1)]
    assert result.reference["measurement_ids"] == window
    assert result.reference["count"] == REFERENCE_WINDOW
    assert result.reference["provisional"] is False
    assert result.reference["statistics_quality"] == "full"
    assert _instant(result.reference["acquired_from"]) == _instant(acquired_at(1))
    assert _instant(result.reference["acquired_to"]) == _instant(
        acquired_at(REFERENCE_WINDOW)
    )
    stats = result.observed["reference_statistics"]["rms"]
    assert stats["n"] == REFERENCE_WINDOW
    assert stats["mean"] == pytest.approx(np.mean([rms_by_id[m] for m in window]))
    assert stats["std"] == pytest.approx(np.std([rms_by_id[m] for m in window], ddof=1))


def test_ae7_a_declared_baseline_is_cited_with_its_declarer(story, rms_by_id):
    """AE7: after declare_healthy_baseline on M01..M05 by L. Rossi, the
    reference is the declared baseline, health declared, the declarer
    cited, the statistics computed on those five only."""
    baseline = story["baseline"]
    assert isinstance(baseline, BaselineDeclarationResult)
    members = [story["ids"][index] for index in BASELINE_MEMBERS]
    assert baseline.measurement_ids == members
    assert baseline.declared_by == BASELINE_DECLARER
    assert baseline.withdrawn is False and baseline.superseded_baseline_id is None

    result = story["assess_baseline"]
    assert result.status == "assessed"
    reference = result.reference
    assert reference["kind"] == "declared_baseline"
    assert reference["health_declared"] is True
    assert "declared healthy baseline" in reference["message"]
    assert BASELINE_DECLARER in reference["message"]
    assert reference["measurement_ids"] == members
    assert reference["count"] == len(members)
    assert reference["baseline"]["baseline_id"] == baseline.baseline_id
    assert reference["baseline"]["declared_by"] == BASELINE_DECLARER
    assert reference["baseline"]["declared_at"] == baseline.declared_at
    assert reference["baseline"]["members_used"] == len(members)
    assert reference["baseline"]["excluded"] == []
    stats = result.observed["reference_statistics"]["rms"]
    assert stats["n"] == len(members)
    assert stats["mean"] == pytest.approx(np.mean([rms_by_id[m] for m in members]))
    assert result.observed["acquisitions_assessed"] == SEQUENCE_LENGTH - len(members)
    assert BASELINE_DECLARER in result.message
    assert result.assessed["classification"] == "persistent_change"

    # The baseline supersedes the automatic window without deleting anything:
    # the history keeps the declaration and every measurement.
    history = story["history_final"]
    assert (
        history.asset["baselines"][POINT_ID]["current"]["baseline_id"]
        == baseline.baseline_id
    )
    assert history.asset["measurement_count"] == SEQUENCE_LENGTH + 1


def test_ae9_the_response_is_structured_and_needs_only_asset_and_point(story):
    """AE9: observed / derived / assessed blocks, at most one verification
    sentence, no recommendations, no error key; the call took nothing but
    the asset and the point."""
    result = story["assess_m30"]
    assert isinstance(result, AssetChangeAssessment)
    _assert_well_formed(result)
    assert result.reprocess is None

    observed = result.observed
    stats = observed["reference_statistics"]
    assert {"rms", "peak", "one_x", "envelope_BPFO"} <= set(stats)
    for name in ("rms", "envelope_BPFO"):
        assert {"mean", "std", "n", "band", "unit"} <= set(stats[name])
        band = stats[name]["band"]
        assert band["low"] < stats[name]["mean"] < band["high"]
        assert band["basis"]
    assert stats["rms"]["unit"] == SIGNAL_UNIT
    assert observed["latest"]["rms"] > stats["rms"]["band"]["high"]
    assert "envelope_BPFO" in observed["latest"]
    assert observed["latest_measurement_id"] == story["ids"][SEQUENCE_LENGTH]
    assert len(observed["measurement_ids_assessed"]) == LAST_K
    assert observed["evidence_presence"]["BPFO"]["k"] == LAST_K

    derived = result.derived
    assert {"deltas", "drift", "per_indicator", "exceedance_runs", "iso_change"} <= set(
        derived
    )
    assert derived["deltas"]["rms"]["latest"] > derived["deltas"]["rms"]["mean"]
    assert derived["deltas"]["rms"]["exceeds"] is True
    assert derived["deltas"]["rms"]["side"] == "above"
    assert derived["drift"]["rms"]["n"] == LAST_K
    assert derived["per_indicator"]["rms"]["criterion"]
    assert derived["iso_change"]["machine_group"] == 2

    assessed = result.assessed
    assert {
        "classification",
        "direction",
        "criterion",
        "indicators_driving",
        "onset_measurement_id",
        "evidence",
    } <= set(assessed)
    assert assessed["indicators_driving"]
    assert assessed["evidence"]["BPFO"]["present_in_last_k"] >= 3

    verification = result.suggested_verification
    assert verification is None or _one_sentence(verification)

    # Nothing beyond the asset and the point is required by the tool.
    params = inspect.signature(asset_tools.assess_asset_change).parameters
    required = [
        name
        for name, p in params.items()
        if p.default is inspect.Parameter.empty and name != "ctx"
    ]
    assert required == ["asset_id", "measurement_point_id"]


# ---------------------------------------------------------------------------
# AE5: processing lineages, on a fresh ledger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ae5_an_older_lineage_is_never_mixed_and_reprocessing_moves_on(
    tools, ctx, sequence_data_dir, sequence, monkeypatch
):
    """AE5: M01..M20 loaded, their snapshots copied into the raw ledger
    under an injected previous lineage, the algorithm version bumped (the
    current lineage changes), M21..M30 loaded on it -> no lineage covers the
    thirty slots: processing_not_homogeneous with the counts per lineage and
    the reprocess remedy, never a mixed trend; reprocess=True (ten per
    call, reference members first) -> assessed on the current lineage at
    the second call; every older snapshot stays in the raw ledger.

    M21..M30 are loaded AFTER the bump on purpose: with all thirty on the
    pre-bump lineage that lineage would still cover every evaluated slot
    and the tool would legitimately assess on it (is_current False).
    """
    await _declare_point(tools, ctx)
    previous = processing_id()  # the lineage of the release before the bump
    early = await _load(tools, ctx, sequence, range(1, 21))
    assert {info.measurement["processing_id"] for info in early} == {previous}
    ids = {
        index: info.measurement["measurement_id"] for index, info in enumerate(early, 1)
    }
    assert _inject_older_lineage([ids[i] for i in range(1, 21)], INJECTED_LINEAGE) == 20
    assert _snapshot_lineage_counts(ASSET_ID) == {INJECTED_LINEAGE: 20, previous: 20}

    monkeypatch.setattr(
        snapshot_module, "ALGORITHM_VERSION", snapshot_module.ALGORITHM_VERSION + 1
    )
    current = processing_id()
    assert current not in {previous, INJECTED_LINEAGE}
    late = await _load(tools, ctx, sequence, range(21, SEQUENCE_LENGTH + 1))
    assert {info.measurement["processing_id"] for info in late} == {current}
    ids.update(
        (index, info.measurement["measurement_id"])
        for index, info in enumerate(late, 21)
    )
    older = {INJECTED_LINEAGE: 20, previous: 20}
    assert _snapshot_lineage_counts(ASSET_ID) == {**older, current: 10}

    mixed = await _assess(tools, ctx)
    assert mixed.status == "processing_not_homogeneous"
    assert mixed.lineages == {**older, current: 10}
    assert mixed.evaluated_slots == SEQUENCE_LENGTH
    assert mixed.missing_for_current == 20
    assert mixed.current_processing_id == current
    assert mixed.assessed is None and mixed.observed is None and mixed.reprocess is None
    assert REPROCESS_CALL in mixed.remedy and "reprocess=True" in mixed.remedy
    assert mixed.reference["kind"] == "automatic_window"
    assert "No processing lineage covers" in mixed.message
    _assert_well_formed(mixed)

    calls: list[AssetChangeAssessment] = []
    for _ in range(3):
        result = await _assess(tools, ctx, reprocess=True)
        calls.append(result)
        if result.status == "assessed" and result.lineage["is_current"]:
            break
    assert [c.status for c in calls] == ["processing_not_homogeneous", "assessed"]
    first, second = calls

    block = first.reprocess
    assert block["processing_id"] == current
    assert block["stale"] == 20
    assert block["reprocessed"] == MAX_REPROCESS_PER_CALL == 10
    assert block["not_reprocessable"] == 0
    assert block["remaining"] == 10
    assert block["next_call"] == REPROCESS_CALL
    # Reference members first, in acquisition order, from the recorded location.
    assert [r["measurement_id"] for r in block["results"]] == [
        ids[i] for i in range(1, 11)
    ]
    assert [r["outcome"] for r in block["results"]] == ["reprocessed"] * 10
    assert [r["location_used"] for r in block["results"]] == [
        sequence_name(i) for i in range(1, 11)
    ]
    assert first.lineages == {**older, current: 20}

    block = second.reprocess
    assert block["reprocessed"] == 10 and block["remaining"] == 0
    assert block["next_call"] is None
    assert [r["measurement_id"] for r in block["results"]] == [
        ids[i] for i in range(20, 10, -1)
    ]
    assert second.lineage["processing_id"] == current
    assert second.lineage["is_current"] is True
    assert second.lineage["algorithm_version"] == snapshot_module.ALGORITHM_VERSION
    assert second.lineage["covered"] == SEQUENCE_LENGTH
    assert second.lineage["candidates"] == {**older, current: 30}
    assert second.lineage["missing_for_current"] == 0
    assert second.assessed["classification"] == "persistent_change"
    _assert_well_formed(second)

    # The older snapshots remain: re-processing appends, never deletes.
    counts = _snapshot_lineage_counts(ASSET_ID)
    assert counts == {**older, current: 30}
    events = LedgerStore(get_ledger_dir()).read(ASSET_ID).events
    assert sum(e["event_type"] == EVENT_MEASUREMENT_RECORDED for e in events) == 30

    # Idempotent: a third call recomputes nothing.
    third = await _assess(tools, ctx, reprocess=True)
    assert third.status == "assessed"
    assert third.reprocess["stale"] == 0 and third.reprocess["reprocessed"] == 0
    assert "nothing to reprocess" in third.reprocess["message"]
    assert _snapshot_lineage_counts(ASSET_ID) == counts


# ---------------------------------------------------------------------------
# AE10 (generic path): raw float32 files with companions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ae10_raw_files_with_companions_declare_the_same_measurements(
    tools, ctx, sandbox_data_dir, clean_repo, sequence, story
):
    """AE10, generic path: M05..M07 as float32 little-endian .bin files
    with a companion (sample_format plus the same measurement object) ->
    a history of three with the declarations the CSV path recorded (and
    the same indicators, up to the float32 quantization); the CSV twin of
    M05 then collapses into the same acquisition slot."""
    csv_path_entries = {
        entry["measurement_id"]: entry
        for entry in story["history_m30"].asset["measurements"]
    }
    raw_dir = sandbox_data_dir / "raw"
    raw_dir.mkdir()
    for index in RAW_SUBSET:
        path = raw_dir / f"seq_m{index:02d}.bin"
        write_raw_file(path, synthesize_measurement(index), "<f4")
        _write_companion(
            path, {**companion_metadata(index), "sample_format": "float32"}
        )

    await _declare_point(tools, ctx)
    infos = await tools["load_signal"](
        ctx=ctx, filepath=[f"raw/seq_m{index:02d}.bin" for index in RAW_SUBSET]
    )
    assert len(infos) == len(RAW_SUBSET)
    raw_ids: dict[int, str] = {}
    for index, info in zip(RAW_SUBSET, infos):
        assert info.raw_format["sample_format"] == "float32"
        assert info.raw_format["byte_order"] == "little"
        assert info.num_samples == synthesize_measurement(index).size
        assert info.sampling_rate == float(companion_metadata(index)["sampling_rate"])
        assert info.signal_unit == SIGNAL_UNIT
        assert info.measurement["ledger_status"] == "recorded"
        assert info.measurement["snapshot_status"] == "complete"
        raw_ids[index] = info.measurement["measurement_id"]

    history = await _history(tools, ctx)
    entries = _oldest_first(history)
    assert history.asset["measurement_count"] == len(RAW_SUBSET)
    for index, entry in zip(RAW_SUBSET, entries):
        declared = companion_metadata(index)["measurement"]
        assert entry["measurement_id"] == raw_ids[index]
        assert history.asset["asset_id"] == declared["asset_id"] == ASSET_ID
        assert entry["measurement_point_id"] == declared["measurement_point_id"]
        assert _instant(entry["acquired_at"]) == _instant(declared["acquired_at"])
        assert entry["sensor_id"] == declared["sensor_id"] == SENSOR_ID
        assert entry["direction"] == declared["direction"] == DIRECTION
        assert entry["rpm"] == declared.get("rpm")
        assert entry["location"] == f"raw/seq_m{index:02d}.bin"
        assert entry["lineages"] == [processing_id()]
        assert entry["comparability"]["grade"] == (
            "qualified" if index in INDICES_WITHOUT_RPM else "comparable"
        )
        # The CSV path recorded the same declaration for the same capture,
        # under another measurement_id (other bytes), with the same numbers.
        twin_entry = csv_path_entries[story["ids"][index]]
        assert twin_entry["measurement_id"] != raw_ids[index]
        for key in ("measurement_point_id", "sensor_id", "direction", "rpm"):
            assert entry[key] == twin_entry[key], key
        assert _instant(entry["acquired_at"]) == _instant(twin_entry["acquired_at"])
        assert entry["declaration_version"] == twin_entry["declaration_version"] == 1
        assert entry["comparability"] == twin_entry["comparability"]
        assert entry["indicators"]["unit"] == twin_entry["indicators"]["unit"]
        for name in ("rms", "peak", "crest_factor", "kurtosis"):
            assert entry["indicators"][name] == pytest.approx(
                twin_entry["indicators"][name], rel=1e-5
            ), name

    # The CSV twin of M05: same declaration, other bytes, same slot.
    csv_dir = sandbox_data_dir / "csv"
    csv_dir.mkdir()
    twin = RAW_SUBSET[0]
    source = sequence[twin - 1]
    shutil.copyfile(source, csv_dir / source.name)
    shutil.copyfile(companion_path(source), csv_dir / companion_path(source).name)
    csv_info = await tools["load_signal"](ctx=ctx, filepath=f"csv/{source.name}")
    assert csv_info.measurement["ledger_status"] == "recorded"
    csv_id = csv_info.measurement["measurement_id"]
    assert csv_id != raw_ids[twin]

    history = await _history(tools, ctx)
    assert history.asset["measurement_count"] == len(RAW_SUBSET) + 1
    twins = [
        e
        for e in history.asset["measurements"]
        if e["measurement_id"] in {csv_id, raw_ids[twin]}
    ]
    assert len(twins) == 2
    assert len({_instant(e["acquired_at"]) for e in twins}) == 1

    result = await _assess(tools, ctx)
    assert result.status in {"assessed", "insufficient_history"}, result.message
    _assert_well_formed(result)
    comparability = result.comparability
    collapsed = comparability["collapsed_duplicates"]
    codes = {q["code"] for q in comparability["qualifications"]}
    if collapsed:
        # One acquisition slot: the latest recorded (the CSV) represents it.
        (entry,) = collapsed
        assert {entry["measurement_id"], entry["kept_measurement_id"]} == {
            csv_id,
            raw_ids[twin],
        }
        assert entry["kept_measurement_id"] == csv_id
        assert "same acquisition slot" in entry["detail"]
        assert "timestamp_collision" not in codes
        # Three distinct slots after the collapse: a reference of three and
        # nothing after it to assess.
        assert result.status == "insufficient_history"
        assert result.available == len(RAW_SUBSET)
    else:
        assert "timestamp_collision" in codes


# ---------------------------------------------------------------------------
# Out-of-order loading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_out_of_order_loading_orders_the_history_by_acquired_at(
    tools, ctx, sequence_data_dir, sequence
):
    """M05, M03, M01 loaded in that order -> the history is M01, M03, M05;
    after M02, M04, M06..M11 the reference window is the first ten by
    acquired_at, whatever the load order."""
    await _declare_point(tools, ctx)
    ids: dict[int, str] = {}
    for index in OUT_OF_ORDER:
        (info,) = await _load(tools, ctx, sequence, [index])
        ids[index] = info.measurement["measurement_id"]

    entries = _oldest_first(await _history(tools, ctx))
    assert [e["measurement_id"] for e in entries] == [
        ids[i] for i in sorted(OUT_OF_ORDER)
    ]
    assert [_instant(e["acquired_at"]) for e in entries] == [
        _instant(acquired_at(i)) for i in sorted(OUT_OF_ORDER)
    ]

    infos = await _load(tools, ctx, sequence, OUT_OF_ORDER_REST)
    for index, info in zip(OUT_OF_ORDER_REST, infos):
        ids[index] = info.measurement["measurement_id"]
    assert sorted(ids) == list(range(1, SPIKE + 1))

    result = await _assess(tools, ctx)
    assert result.status == "assessed"
    assert result.reference["measurement_ids"] == [
        ids[i] for i in range(1, REFERENCE_WINDOW + 1)
    ]
    assert _instant(result.reference["acquired_from"]) == _instant(acquired_at(1))
    assert _instant(result.reference["acquired_to"]) == _instant(
        acquired_at(REFERENCE_WINDOW)
    )
    assert result.observed["latest_measurement_id"] == ids[SPIKE]
    assert result.assessed["classification"] == "unconfirmed_single_acquisition"


# ---------------------------------------------------------------------------
# Determinism: the same scene twice
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_same_scene_twice_gives_identical_canonical_payloads(
    tools, ctx, sequence_data_dir, sequence, tmp_path
):
    """Two fresh ledgers, the same declaration and the same thirty loads ->
    the assessment and the history serialize to the same canonical JSON
    once the volatile keys are dropped (``recorded_at``, ``declared_at``,
    ``provenance``: server instants and host provenance, nothing else)."""

    async def run(ledger_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("PMM_LEDGER_DIR", str(ledger_root))
            get_repository().clear_all()
            await _declare_point(tools, ctx)
            await _load(tools, ctx, sequence, range(1, SEQUENCE_LENGTH + 1))
            assessment = await _assess(tools, ctx)
            history = await _history(tools, ctx, max_measurements=SEQUENCE_LENGTH)
        assert assessment.status == "assessed"
        return assessment.model_dump(), history.model_dump()

    first = await run(tmp_path / "ledger-first")
    second = await run(tmp_path / "ledger-second")

    stripped: set[str] = set()
    for payload_first, payload_second in zip(first, second):
        assert canonical_json(
            _without_volatile(payload_first, stripped)
        ) == canonical_json(_without_volatile(payload_second, stripped))
    assert stripped <= VOLATILE_KEYS
    # The assessment itself is pure: it carries no volatile key at all, so
    # its raw canonical form is identical too.
    assert canonical_json(first[0]) == canonical_json(second[0])
