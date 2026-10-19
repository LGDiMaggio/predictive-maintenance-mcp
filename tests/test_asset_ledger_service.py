"""Tests for asset_ledger.service (U5): registration of loaded measurements.

The service is exercised against a REAL ``LedgerStore`` under ``tmp_path``
with signals from ``tests/_golden_signals.py`` and numpy, and info dicts
shaped like the repository's (identity blocks built by
``build_measurement_identity`` on real files, so digests and sizes are
real). These tests pin, from the plan's U5 scenarios:

- the effective declaration, the file block (relative POSIX location inside
  the data directory, absolute with a flag outside), the fingerprint and
  the changed keys;
- the three ledger outcomes (recorded, already_recorded, superseded with
  the changed keys), the snapshot rule (a new snapshot only when the
  resolved context changes: rpm and unit yes, timestamp and location no),
  the batch discipline (one read per asset), the restart (a new store sees
  the same history in acquired_at order);
- the error paths: an unusable ledger directory, a busy lock, a failed
  fsync after the bytes were written (retry is idempotent), a snapshot
  failure (retry appends exactly one snapshot);
- privacy (free-form companion keys never reach the ledger bytes) and the
  wrong-asset correction (reattribution without deleting anything);
- the outcome contract (keys, no ``error`` key anywhere);
- the re-processing of stale snapshots (U6): the priority order under the
  per-call limit (reference first, then the last K newest first), the
  assessment reachable within two calls, the hash-verified search of the
  file across every recorded location, the per-measurement refusals (file
  missing everywhere, content changed, append failure) and idempotence.
"""

import copy
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from _golden_signals import FS, golden_signals
from conftest import write_raw_file
from predictive_maintenance_mcp.asset_ledger import service
from predictive_maintenance_mcp.asset_ledger import store as store_module
from predictive_maintenance_mcp.asset_ledger.assessment import (
    AssessmentParams,
    assess_change,
)
from predictive_maintenance_mcp.asset_ledger.service import (
    DECLARATION_KEYS,
    LEDGER_STATUSES,
    LOAD_OUTCOME_KEYS,
    MAX_REPROCESS_PER_CALL,
    OUTCOME_KEYS,
    POINT_DECLARED_KEYS,
    POINT_FREE_TEXT_FIELDS,
    REPROCESS_OUTCOMES,
    SNAPSHOT_PAYLOAD_KEYS,
    SNAPSHOT_STATUSES,
    build_declaration,
    changed_keys,
    declaration_fingerprint,
    declare_healthy_baseline,
    declare_measurement_point,
    file_block,
    record_measurements,
    reprocess_stale_snapshots,
    resolve_point_context,
)
from predictive_maintenance_mcp.asset_ledger.snapshot import (
    SnapshotPolicy,
    processing_id,
)
from predictive_maintenance_mcp.asset_ledger.store import (
    EVENT_HEALTH_SNAPSHOT_COMPUTED,
    EVENT_MEASUREMENT_POINT_DECLARED,
    EVENT_MEASUREMENT_RECORDED,
    LedgerStore,
    make_event,
)
from predictive_maintenance_mcp.signal_acquisition.measurement import (
    MEASUREMENT_DECLARATION_KEYS,
    build_measurement_identity,
)

ASSET = "P-101"
OTHER_ASSET = "P-102"
POINT = "motor_de_h"
ACQUIRED_AT = "2026-08-20T13:42:00+02:00"
ACQUIRED_AT_UTC = "2026-08-20T11:42:00+00:00"
RPM = 1800

#: Fixed provenance: nothing host-dependent in the recorded snapshots.
PROVENANCE = {
    "platform": "service-test",
    "python_version": "0.0.0",
    "numpy_version": "0.0.0",
    "scipy_version": "0.0.0",
    "pipeline_version": "0.0.0",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_signal(path: Path, values: np.ndarray) -> None:
    """Write a headerless single-column CSV with LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as fh:
        np.savetxt(fh, values, fmt="%.8f")


def noise(seed: int, n: int = 10000) -> np.ndarray:
    return 0.1 * np.random.default_rng(seed).standard_normal(n)


def companion(
    *,
    asset: str = ASSET,
    point: str = POINT,
    acquired_at: str = ACQUIRED_AT,
    rpm: Optional[float] = RPM,
    direction: Optional[str] = "horizontal",
    sensor_id: Optional[str] = "ACC01",
    signal_unit: Optional[str] = "g",
    sampling_rate: Optional[float] = FS,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    measurement: dict[str, Any] = {
        "asset_id": asset,
        "measurement_point_id": point,
        "acquired_at": acquired_at,
    }
    if rpm is not None:
        measurement["rpm"] = rpm
    if direction is not None:
        measurement["direction"] = direction
    if sensor_id is not None:
        measurement["sensor_id"] = sensor_id
    payload: dict[str, Any] = {"measurement": measurement}
    if sampling_rate is not None:
        payload["sampling_rate"] = sampling_rate
    if signal_unit is not None:
        payload["signal_unit"] = signal_unit
    payload.update(extra or {})
    return payload


def make_info(
    path: Path,
    values: np.ndarray,
    *,
    signal_id: Optional[str] = None,
    raw_format: Optional[dict[str, Any]] = None,
    signal_unit: Optional[str] = "g",
    sampling_rate: Optional[float] = FS,
    **companion_kwargs: Any,
) -> dict[str, Any]:
    """A repository-shaped info dict for *path* (written if absent).

    The identity block is built exactly as the repository builds it; the
    unit and rate given here are the EFFECTIVE ones (an explicit
    ``signal_unit`` parameter of load_signal wins over the companion).
    """
    if not path.exists():
        write_signal(path, values)
    comp = companion(
        signal_unit=signal_unit, sampling_rate=sampling_rate, **companion_kwargs
    )
    identity = build_measurement_identity(
        comp, f"{path.stem}_metadata.json", signal_path=path, channel_index=0
    )
    return {
        "signal_id": signal_id or path.stem,
        "filepath": str(path),
        "load_timestamp": "2026-09-10T08:00:00+00:00",
        "shape": [int(values.size)],
        "num_samples": int(values.size),
        "sampling_rate": sampling_rate,
        "duration_s": None if sampling_rate is None else values.size / sampling_rate,
        "size_bytes": int(values.nbytes),
        "signal_unit": signal_unit,
        "source_metadata": comp,
        "raw_format": raw_format,
        "measurement": identity,
        "companion_warning": None,
    }


def point_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "measurement_point_id": POINT,
        "declaration_version": 1,
        "bearing_id": "6205",
        "fault_orders": None,
        "machine_group": 2,
        "support_type": "rigid",
        "machine_power_kw": None,
        "expected_signal_unit": "g",
        "expected_sensor_id": None,
        "expected_direction": "horizontal",
        "nominal_rpm": 1800.0,
        "declared_by": "test",
        "note": None,
        "changed": [],
    }
    payload.update(overrides)
    return payload


def declare_point(store: LedgerStore, asset: str = ASSET, **overrides: Any) -> None:
    payload = point_payload(**overrides)
    store.append(asset, make_event(EVENT_MEASUREMENT_POINT_DECLARED, asset, payload))


def events_of(store: LedgerStore, asset: str = ASSET) -> list[dict[str, Any]]:
    return store.read(asset).events


def payloads(store: LedgerStore, event_type: str, asset: str = ASSET) -> list[dict]:
    return [
        e["payload"] for e in events_of(store, asset) if e["event_type"] == event_type
    ]


def walk(tree: Any):
    """Yield every dict key in a nested structure."""
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield key
            yield from walk(value)
    elif isinstance(tree, (list, tuple)):
        for item in tree:
            yield from walk(item)


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "ledger"


@pytest.fixture
def store(root) -> LedgerStore:
    return LedgerStore(root)


@pytest.fixture
def data_dir(tmp_path) -> Path:
    directory = tmp_path / "data" / "signals"
    directory.mkdir(parents=True)
    return directory


@pytest.fixture
def signals() -> dict[str, np.ndarray]:
    return golden_signals()


def record(
    store: LedgerStore,
    data_dir: Path,
    infos: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return record_measurements(
        infos,
        arrays,
        store=store,
        data_dir=data_dir,
        provenance_overrides=PROVENANCE,
        **kwargs,
    )


def record_one(
    store: LedgerStore, data_dir: Path, info: dict[str, Any], values: np.ndarray
) -> dict[str, Any]:
    return record(store, data_dir, [info], {info["signal_id"]: values})[0]


#: A previous processing lineage (a policy that is not the default).
OLD_POLICY = SnapshotPolicy(tolerance_pct=2.0)

#: Weekly acquisitions from a fixed instant, like the synthetic sequence.
FIRST_ACQUIRED = datetime(2026, 1, 5, 9, 0, tzinfo=timezone(timedelta(hours=1)))

REPROCESS_CALL = (
    "assess_asset_change(asset_id='P-101', measurement_point_id='motor_de_h', "
    "reprocess=True)"
)


def acquired(index: int) -> str:
    return (FIRST_ACQUIRED + (index - 1) * timedelta(weeks=1)).isoformat()


def load_sequence(
    store: LedgerStore,
    data_dir: Path,
    count: int,
    *,
    policy: Optional[SnapshotPolicy] = OLD_POLICY,
) -> list[str]:
    """Record *count* weekly noise measurements ``m01.csv``.. and return their
    measurement ids in acquisition order (snapshots on *policy*'s lineage)."""
    infos: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for index in range(1, count + 1):
        values = noise(index, 5000)
        info = make_info(
            data_dir / f"m{index:02d}.csv", values, acquired_at=acquired(index)
        )
        infos.append(info)
        arrays[info["signal_id"]] = values
    outcomes = record(store, data_dir, infos, arrays, policy=policy)
    assert [o["ledger_status"] for o in outcomes] == ["recorded"] * count
    return [o["measurement_id"] for o in outcomes]


def reprocess(store: LedgerStore, data_dir: Path, **kwargs: Any) -> dict[str, Any]:
    return reprocess_stale_snapshots(
        ASSET,
        POINT,
        store=store,
        data_dir=data_dir,
        provenance_overrides=PROVENANCE,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Declaration, file block, fingerprint
# ---------------------------------------------------------------------------


class TestDeclarationAndFile:
    def test_build_declaration_keys_and_values(self, data_dir):
        info = make_info(data_dir / "m01.csv", noise(1), sampling_rate=10000)
        declaration = build_declaration(info)
        assert tuple(declaration) == DECLARATION_KEYS
        assert DECLARATION_KEYS[: len(MEASUREMENT_DECLARATION_KEYS)] == (
            MEASUREMENT_DECLARATION_KEYS
        )
        assert declaration["asset_id"] == ASSET
        assert declaration["acquired_at"] == ACQUIRED_AT_UTC
        assert declaration["rpm"] == 1800.0
        assert declaration["sampling_rate"] == 10000.0
        assert isinstance(declaration["sampling_rate"], float)
        assert declaration["signal_unit"] == "g"
        assert declaration["raw_format"] is None
        assert declaration["channel_index"] == 0
        assert "measurement_id" not in declaration
        assert "content_sha256" not in declaration

    def test_build_declaration_copies_the_raw_format(self, data_dir):
        raw = {"sample_format": "float32", "scale_factor": 0.5, "channel_index": 1}
        info = make_info(data_dir / "m01.bin", noise(1), raw_format=raw)
        declaration = build_declaration(info)
        assert declaration["raw_format"] == raw
        assert declaration["raw_format"] is not raw

    def test_build_declaration_refuses_a_signal_without_identity(self, data_dir):
        info = make_info(data_dir / "m01.csv", noise(1))
        info["measurement"] = None
        with pytest.raises(ValueError, match="measurement"):
            build_declaration(info)

    def test_file_block_inside_data_dir_is_relative_posix(self, data_dir):
        path = data_dir / "site_a" / "m01.csv"
        block = file_block(path, data_dir, content_sha256="ab" * 32, size_bytes=12)
        assert block == {
            "location": "site_a/m01.csv",
            "location_is_relative": True,
            "content_sha256": "ab" * 32,
            "size_bytes": 12,
        }

    def test_file_block_absolute_spelling_inside_is_relative(self, data_dir):
        spelled = os.path.join(str(data_dir), "sub", "..", "m01.csv")
        block = file_block(spelled, data_dir, content_sha256="0" * 64, size_bytes=1)
        assert block["location"] == "m01.csv"
        assert block["location_is_relative"] is True

    def test_file_block_outside_data_dir_is_absolute_with_flag(
        self, tmp_path, data_dir
    ):
        outside = tmp_path / "elsewhere" / "m01.csv"
        block = file_block(outside, data_dir, content_sha256="0" * 64, size_bytes=1)
        assert block["location"] == str(outside)
        assert block["location_is_relative"] is False

    def test_sibling_directory_is_not_inside(self, tmp_path):
        base = tmp_path / "signals"
        sibling = tmp_path / "signals_evil" / "m01.csv"
        block = file_block(sibling, base, content_sha256="0" * 64, size_bytes=1)
        assert block["location_is_relative"] is False

    def test_fingerprint_is_stable_and_sensitive(self, data_dir):
        declaration = build_declaration(make_info(data_dir / "m01.csv", noise(1)))
        same = declaration_fingerprint(dict(reversed(list(declaration.items()))), "a")
        assert declaration_fingerprint(declaration, "a") == same
        assert declaration_fingerprint(declaration, "a") != declaration_fingerprint(
            declaration, "b"
        )
        corrected = {**declaration, "rpm": 1500.0}
        assert declaration_fingerprint(corrected, "a") != declaration_fingerprint(
            declaration, "a"
        )

    def test_changed_keys(self, data_dir):
        declaration = build_declaration(make_info(data_dir / "m01.csv", noise(1)))
        assert changed_keys(declaration, "a", declaration, "a") == []
        assert changed_keys(declaration, "a", {**declaration, "rpm": 1.0}, "a") == [
            "rpm"
        ]
        assert changed_keys(declaration, "a", declaration, "b") == ["location"]
        assert changed_keys(declaration, "a", {**declaration, "rpm": 1.0}, "b") == [
            "location",
            "rpm",
        ]
        assert changed_keys(
            declaration, "a", {**declaration, "asset_id": OTHER_ASSET}, "a"
        ) == ["asset_id"]
        # A key present on one side only counts as changed.
        assert changed_keys({**declaration, "extra": 1}, "a", declaration, "a") == [
            "extra"
        ]

    def test_contract_constants(self):
        assert LEDGER_STATUSES == (
            "recorded",
            "already_recorded",
            "superseded",
            "not_recorded",
        )
        assert SNAPSHOT_STATUSES == ("complete", "partial", "failed", "skipped")
        assert set(LOAD_OUTCOME_KEYS) < set(OUTCOME_KEYS)
        assert "event_ids" not in LOAD_OUTCOME_KEYS
        assert "context" in SNAPSHOT_PAYLOAD_KEYS
        assert POINT_FREE_TEXT_FIELDS == (
            "bearing_id",
            "expected_sensor_id",
            "declared_by",
            "note",
        )
        assert set(POINT_FREE_TEXT_FIELDS) <= set(POINT_DECLARED_KEYS)


# ---------------------------------------------------------------------------
# Registration: happy paths
# ---------------------------------------------------------------------------


class TestRecordMeasurements:
    def test_first_load_is_recorded_with_a_partial_snapshot(
        self, store, data_dir, root
    ):
        """Covers AE1 (service side): no point declared yet, so the snapshot
        is partial (bearing and ISO blocks missing, each with a remedy)."""
        values = noise(1)
        info = make_info(data_dir / "site_a" / "m01.csv", values)
        outcome = record_one(store, data_dir, info, values)

        assert tuple(outcome) == OUTCOME_KEYS
        assert outcome["signal_id"] == "m01"
        assert outcome["asset_id"] == ASSET
        assert outcome["measurement_point_id"] == POINT
        assert outcome["measurement_id"] == info["measurement"]["measurement_id"]
        assert outcome["ledger_status"] == "recorded"
        assert outcome["reason"] is None
        assert outcome["changed"] == []
        assert outcome["reattributed_from"] is None
        assert outcome["declaration_version"] == 1
        assert outcome["snapshot_status"] == "partial"
        assert set(outcome["missing"]) == {"bearing", "iso"}
        for entry in outcome["missing"].values():
            assert entry["reason"] and entry["remedy"]
        assert outcome["processing_id"] == processing_id()
        assert len(outcome["snapshot_id"]) == 16
        assert len(outcome["context_digest"]) == 16
        assert outcome["comparability"]["grade"] == "comparable"
        assert outcome["comparability"]["qualifications"] == []
        assert len(outcome["event_ids"]) == 2
        assert "error" not in set(walk(outcome))

        events = events_of(store)
        assert [e["event_type"] for e in events] == [
            EVENT_MEASUREMENT_RECORDED,
            EVENT_HEALTH_SNAPSHOT_COMPUTED,
        ]
        assert [e["event_id"] for e in events] == outcome["event_ids"]
        recorded = events[0]["payload"]
        assert recorded["measurement_id"] == outcome["measurement_id"]
        assert recorded["measurement_point_id"] == POINT
        assert recorded["declaration_version"] == 1
        # Canonical JSON sorts keys on disk; the key SET is the contract.
        assert set(recorded["declaration"]) == set(DECLARATION_KEYS)
        assert recorded["declaration"]["asset_id"] == ASSET
        assert recorded["file"] == {
            "location": "site_a/m01.csv",
            "location_is_relative": True,
            "content_sha256": hashlib.sha256(
                (data_dir / "site_a" / "m01.csv").read_bytes()
            ).hexdigest(),
            "size_bytes": (data_dir / "site_a" / "m01.csv").stat().st_size,
        }
        assert recorded["signal_id"] == "m01"
        assert recorded["changed"] == []
        assert recorded["locations"] == ["site_a/m01.csv"]

        snapshot = events[1]["payload"]
        assert set(snapshot) == set(SNAPSHOT_PAYLOAD_KEYS)
        assert snapshot["snapshot_id"] == outcome["snapshot_id"]
        assert snapshot["processing"]["processing_id"] == outcome["processing_id"]
        assert snapshot["processing"]["provenance"] == PROVENANCE
        assert snapshot["context_digest"] == outcome["context_digest"]
        assert snapshot["context"]["rpm"] == 1800.0
        assert snapshot["context"]["rpm_source"] == "measurement"
        assert snapshot["point_declaration_version"] is None
        assert snapshot["one_x"] is not None
        assert snapshot["bearing"] is None
        assert snapshot["iso"] is None
        assert snapshot["missing"] == outcome["missing"]

        assert store.find_measurement_asset(outcome["measurement_id"]) == ASSET
        assert store.list_assets() == [ASSET]
        assert sorted(p.name for p in root.iterdir()) == [
            "P-101.jsonl",
            "P-101.jsonl.lock",
            "_measurements.jsonl",
            "_measurements.jsonl.lock",
        ]

    def test_point_declared_first_gives_a_complete_comparable_snapshot(
        self, store, data_dir, signals
    ):
        declare_point(store, expected_sensor_id="ACC01")
        values = signals["golden_bearing"]
        info = make_info(data_dir / "m01.csv", values)
        outcome = record_one(store, data_dir, info, values)

        assert outcome["ledger_status"] == "recorded"
        assert outcome["snapshot_status"] == "complete"
        assert outcome["missing"] == {}
        assert outcome["comparability"] == {"grade": "comparable", "qualifications": []}

        snapshot = payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)[0]
        assert snapshot["point_declaration_version"] == 1
        assert snapshot["context"]["bearing_id"] == "6205"
        assert snapshot["iso"]["direction"] == "horizontal"
        assert set(snapshot["bearing"]["labels"]) == {"BPFO", "BPFI", "BSF", "FTF"}
        assert snapshot["bearing"]["labels"]["BPFO"]["detected"] is True

    def test_restart_between_loads_keeps_the_history_in_acquired_order(
        self, root, data_dir
    ):
        """Covers AE2: a new store instance (a restarted server) sees the
        history; the later-acquired measurement loaded first still sorts
        second."""
        later = make_info(
            data_dir / "m02.csv", noise(2), acquired_at="2026-08-27T13:42:00+02:00"
        )
        first = record_one(LedgerStore(root), data_dir, later, noise(2))
        earlier = make_info(data_dir / "m01.csv", noise(1))
        second = record_one(LedgerStore(root), data_dir, earlier, noise(1))
        assert first["ledger_status"] == second["ledger_status"] == "recorded"

        view = LedgerStore(root).read_view(ASSET)
        assert view["ordered_measurement_ids"] == [
            earlier["measurement"]["measurement_id"],
            later["measurement"]["measurement_id"],
        ]
        for slot in view["measurements"].values():
            assert len(slot["snapshots"]) == 1
            assert len(slot["history"]) == 1

    def test_same_file_twice_is_already_recorded_with_the_same_id(
        self, store, data_dir
    ):
        values = noise(1)
        info = make_info(data_dir / "m01.csv", values)
        first = record_one(store, data_dir, info, values)
        again = record_one(store, data_dir, copy.deepcopy(info), values)

        assert again["ledger_status"] == "already_recorded"
        assert again["measurement_id"] == first["measurement_id"]
        assert again["declaration_version"] == 1
        assert again["changed"] == []
        assert again["event_ids"] == []
        assert again["snapshot_status"] == "partial"
        assert again["snapshot_id"] == first["snapshot_id"]
        assert len(payloads(store, EVENT_MEASUREMENT_RECORDED)) == 1
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 1

    def test_rpm_correction_supersedes_and_adds_a_snapshot(self, store, data_dir):
        values = noise(1)
        first = record_one(
            store, data_dir, make_info(data_dir / "m01.csv", values), values
        )
        corrected = make_info(data_dir / "m01.csv", values, rpm=1500)
        second = record_one(store, data_dir, corrected, values)

        assert second["ledger_status"] == "superseded"
        assert second["measurement_id"] == first["measurement_id"]
        assert second["changed"] == ["rpm"]
        assert second["declaration_version"] == 2
        assert second["snapshot_id"] != first["snapshot_id"]
        assert second["context_digest"] != first["context_digest"]
        assert second["processing_id"] == first["processing_id"]

        recorded = payloads(store, EVENT_MEASUREMENT_RECORDED)
        assert [p["declaration_version"] for p in recorded] == [1, 2]
        assert recorded[1]["changed"] == ["rpm"]
        assert recorded[1]["declaration"]["rpm"] == 1500.0
        snapshots = payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)
        assert [s["one_x"]["target_hz"] for s in snapshots] == [30.0, 25.0]
        view = store.read_view(ASSET)
        slot = view["measurements"][first["measurement_id"]]
        assert len(slot["history"]) == 2
        assert slot["current"]["declaration"]["rpm"] == 1500.0
        assert len(slot["snapshots"]) == 2  # the first one is kept

    def test_acquired_at_correction_supersedes_without_a_new_snapshot(
        self, store, data_dir
    ):
        values = noise(1)
        first = record_one(
            store, data_dir, make_info(data_dir / "m01.csv", values), values
        )
        corrected = make_info(
            data_dir / "m01.csv", values, acquired_at="2026-08-20T14:42:00+02:00"
        )
        second = record_one(store, data_dir, corrected, values)

        assert second["ledger_status"] == "superseded"
        assert second["changed"] == ["acquired_at"]
        assert second["snapshot_id"] == first["snapshot_id"]
        assert second["snapshot_status"] == "partial"
        assert len(second["event_ids"]) == 1
        view = store.read_view(ASSET)
        slot = view["measurements"][first["measurement_id"]]
        assert slot["current"]["declaration"]["acquired_at"] == (
            "2026-08-20T12:42:00+00:00"
        )
        assert len(slot["snapshots"]) == 1

    def test_moved_file_supersedes_with_location_only(self, store, data_dir):
        values = noise(1)
        original = data_dir / "m01.csv"
        first = record_one(store, data_dir, make_info(original, values), values)
        moved = data_dir / "archive" / "m01.csv"
        moved.parent.mkdir()
        shutil.move(str(original), str(moved))
        second = record_one(store, data_dir, make_info(moved, values), values)

        assert second["ledger_status"] == "superseded"
        assert second["changed"] == ["location"]
        assert second["measurement_id"] == first["measurement_id"]
        assert second["snapshot_id"] == first["snapshot_id"]
        recorded = payloads(store, EVENT_MEASUREMENT_RECORDED)
        assert recorded[1]["file"]["location"] == "archive/m01.csv"
        assert recorded[1]["locations"] == ["m01.csv", "archive/m01.csv"]
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 1

    def test_unit_change_supersedes_and_adds_a_snapshot(self, store, data_dir):
        values = noise(1)
        first = record_one(
            store, data_dir, make_info(data_dir / "m01.csv", values), values
        )
        redeclared = make_info(data_dir / "m01.csv", values, signal_unit="m/s2")
        second = record_one(store, data_dir, redeclared, values)

        assert second["ledger_status"] == "superseded"
        assert second["changed"] == ["signal_unit"]
        assert second["snapshot_id"] != first["snapshot_id"]
        snapshots = payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)
        assert [s["indicators"]["unit"] for s in snapshots] == ["g", "m/s2"]

    def test_two_files_in_the_same_slot_are_both_recorded(self, store, data_dir):
        """A timestamp collision is a query-time qualification: both
        measurements are recorded with distinct ids."""
        a = make_info(data_dir / "a.csv", noise(1))
        b = make_info(data_dir / "b.csv", noise(2))
        outcomes = record(store, data_dir, [a, b], {"a": noise(1), "b": noise(2)})
        assert [o["ledger_status"] for o in outcomes] == ["recorded", "recorded"]
        assert outcomes[0]["measurement_id"] != outcomes[1]["measurement_id"]
        view = store.read_view(ASSET)
        assert len(view["ordered_measurement_ids"]) == 2
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 2

    def test_axes_against_a_point_expecting_x(self, store, data_dir):
        declare_point(store, expected_direction="x")
        infos = [
            make_info(data_dir / f"axis_{axis}.csv", noise(i), direction=axis)
            for i, axis in enumerate(("x", "y", "z"), start=1)
        ]
        arrays = {info["signal_id"]: noise(i) for i, info in enumerate(infos, 1)}
        outcomes = record(store, data_dir, infos, arrays)
        assert [o["ledger_status"] for o in outcomes] == ["recorded"] * 3
        grades = {o["signal_id"]: o["comparability"] for o in outcomes}
        assert grades["axis_x"]["grade"] == "comparable"
        for axis in ("axis_y", "axis_z"):
            assert grades[axis]["grade"] == "non_comparable"
            codes = [q["code"] for q in grades[axis]["qualifications"]]
            assert codes == ["direction_mismatch"]
        assert len(store.read_view(ASSET)["ordered_measurement_ids"]) == 3

    def test_batch_reads_the_asset_and_the_index_once(
        self, store, data_dir, monkeypatch
    ):
        reads: list[str] = []
        original_read = LedgerStore.read
        index_reads: list[int] = []
        original_index = LedgerStore.read_index

        def spy_read(self, asset_id):
            reads.append(asset_id)
            return original_read(self, asset_id)

        def spy_index(self):
            index_reads.append(1)
            return original_index(self)

        monkeypatch.setattr(LedgerStore, "read", spy_read)
        monkeypatch.setattr(LedgerStore, "read_index", spy_index)

        infos = [
            make_info(
                data_dir / f"m{i:02d}.csv",
                noise(i),
                acquired_at=f"2026-08-{i:02d}T10:00:00+00:00",
            )
            for i in range(1, 6)
        ]
        arrays = {info["signal_id"]: noise(i) for i, info in enumerate(infos, 1)}
        outcomes = record(store, data_dir, infos, arrays)
        assert [o["ledger_status"] for o in outcomes] == ["recorded"] * 5
        assert [o["declaration_version"] for o in outcomes] == [1] * 5
        assert reads == [ASSET]
        assert len(index_reads) == 1
        assert len(payloads(store, EVENT_MEASUREMENT_RECORDED)) == 5
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 5

    def test_same_measurement_twice_in_one_batch_is_versioned_once_each(
        self, store, data_dir
    ):
        """Within a batch the second copy of the same bytes at another
        location is a supersession of the first (the view is kept current
        between signals, never re-read)."""
        values = noise(1)
        a = make_info(data_dir / "a.csv", values)
        b = make_info(data_dir / "copy" / "a.csv", values, signal_id="copy_a")
        outcomes = record(store, data_dir, [a, b], {"a": values, "copy_a": values})
        assert outcomes[0]["ledger_status"] == "recorded"
        assert outcomes[1]["ledger_status"] == "superseded"
        assert outcomes[1]["changed"] == ["location"]
        assert outcomes[1]["declaration_version"] == 2
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 1

    def test_policy_and_lineage_flow_into_the_outcome(self, store, data_dir):
        policy = SnapshotPolicy(tolerance_pct=2.0)
        values = noise(1)
        outcome = record(
            store,
            data_dir,
            [make_info(data_dir / "m01.csv", values)],
            {"m01": values},
            policy=policy,
        )[0]
        assert outcome["processing_id"] == processing_id(policy)
        assert outcome["processing_id"] != processing_id()
        snapshot = payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)[0]
        assert snapshot["processing"]["params"]["tolerance_pct"] == 2.0


# ---------------------------------------------------------------------------
# Snapshot skipped or failed, never raised
# ---------------------------------------------------------------------------


class TestSnapshotOutcomes:
    def test_without_sampling_rate_the_snapshot_is_skipped(self, store, data_dir):
        values = noise(1)
        info = make_info(data_dir / "m01.csv", values, sampling_rate=None)
        outcome = record_one(store, data_dir, info, values)
        assert outcome["ledger_status"] == "recorded"
        assert outcome["snapshot_status"] == "skipped"
        assert "sampling_rate" in outcome["reason"]
        assert outcome["snapshot_id"] is None
        assert outcome["comparability"]["grade"] == "comparable"
        assert payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED) == []
        # Declaring the rate later supersedes and computes the snapshot.
        declared = make_info(data_dir / "m01.csv", values, sampling_rate=FS)
        again = record_one(store, data_dir, declared, values)
        assert again["ledger_status"] == "superseded"
        assert again["changed"] == ["sampling_rate"]
        assert again["snapshot_status"] == "partial"
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 1

    def test_missing_signal_skips_the_snapshot_with_a_reason(self, store, data_dir):
        info = make_info(data_dir / "m01.csv", noise(1))
        outcome = record(store, data_dir, [info], {})[0]
        assert outcome["ledger_status"] == "recorded"
        assert outcome["snapshot_status"] == "skipped"
        assert "m01" in outcome["reason"]
        assert payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED) == []

    def test_signals_may_be_a_callable(self, store, data_dir):
        values = noise(1)
        info = make_info(data_dir / "m01.csv", values)
        asked: list[str] = []

        def source(signal_id: str) -> np.ndarray:
            asked.append(signal_id)
            return values

        outcome = record_measurements([info], source, store=store, data_dir=data_dir)[0]
        assert asked == ["m01"]
        assert outcome["snapshot_status"] == "partial"

    def test_snapshot_failure_keeps_the_measurement_and_the_retry_appends_one(
        self, store, data_dir
    ):
        values = noise(1)
        info = make_info(data_dir / "m01.csv", values)

        def boom(*args, **kwargs):
            raise RuntimeError("engine exploded")

        # A scoped patch: the autouse ledger sandbox must survive the undo.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(service, "compute_health_snapshot", boom)
            first = record_one(store, data_dir, info, values)
        assert first["ledger_status"] == "recorded"
        assert first["snapshot_status"] == "failed"
        assert "engine exploded" in first["reason"]
        assert first["snapshot_id"] is None
        assert first["missing"] == {}
        assert len(first["event_ids"]) == 1
        assert len(payloads(store, EVENT_MEASUREMENT_RECORDED)) == 1
        assert payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED) == []
        view = store.read_view(ASSET)
        assert view["measurements"][first["measurement_id"]]["snapshots"] == []

        again = record_one(store, data_dir, copy.deepcopy(info), values)
        assert again["ledger_status"] == "already_recorded"
        assert again["snapshot_status"] == "partial"
        assert len(again["event_ids"]) == 1
        assert len(payloads(store, EVENT_MEASUREMENT_RECORDED)) == 1
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 1

    def test_snapshot_append_failure_is_reported_as_failed(
        self, store, data_dir, monkeypatch
    ):
        values = noise(1)
        info = make_info(data_dir / "m01.csv", values)
        original = LedgerStore.append

        def failing_append(self, asset_id, event):
            if event["event_type"] == EVENT_HEALTH_SNAPSHOT_COMPUTED:
                raise store_module.LedgerWriteError("disk full while appending")
            return original(self, asset_id, event)

        monkeypatch.setattr(LedgerStore, "append", failing_append)
        outcome = record_one(store, data_dir, info, values)
        assert outcome["ledger_status"] == "recorded"
        assert outcome["snapshot_status"] == "failed"
        assert "disk full" in outcome["reason"]
        assert outcome["snapshot_id"] is not None  # computed, not persisted


# ---------------------------------------------------------------------------
# Ledger failures: not_recorded, never raised
# ---------------------------------------------------------------------------


class TestLedgerFailures:
    def test_unusable_ledger_directory_is_not_recorded(self, tmp_path, data_dir):
        blocker = tmp_path / "ledger"
        blocker.write_text("a file where the directory should be", encoding="utf-8")
        values = noise(1)
        info = make_info(data_dir / "m01.csv", values)
        outcome = record_one(LedgerStore(blocker), data_dir, info, values)
        assert outcome["ledger_status"] == "not_recorded"
        assert outcome["reason"]
        assert "PMM_LEDGER_DIR" in outcome["reason"]
        assert outcome["snapshot_status"] == "skipped"
        assert outcome["snapshot_id"] is None
        assert outcome["event_ids"] == []
        assert outcome["comparability"]["grade"] == "comparable"
        assert blocker.read_text(encoding="utf-8").startswith("a file")

    def test_busy_lock_is_not_recorded_then_recovers(self, root, data_dir):
        holder = LedgerStore(root)
        declare_point(holder)
        _, lock_path = holder._paths(ASSET)
        fd = holder._acquire_lock(lock_path, what="test holder")
        values = noise(1)
        info = make_info(data_dir / "m01.csv", values)
        try:
            contender = LedgerStore(root, lock_timeout=0.2)
            started = time.monotonic()
            outcome = record_one(contender, data_dir, info, values)
            assert time.monotonic() - started < 5.0
        finally:
            holder._release_lock(fd)
        assert outcome["ledger_status"] == "not_recorded"
        assert "lock" in outcome["reason"]
        assert outcome["snapshot_status"] == "skipped"
        # The point was readable, so the grade still saw it.
        assert outcome["comparability"]["grade"] == "comparable"
        assert len(events_of(LedgerStore(root))) == 1

        again = record_one(LedgerStore(root), data_dir, copy.deepcopy(info), values)
        assert again["ledger_status"] == "recorded"
        assert again["snapshot_status"] == "complete"

    def test_fsync_failure_after_the_bytes_then_idempotent_retry(
        self, store, root, data_dir
    ):
        values = noise(1)
        info = make_info(data_dir / "m01.csv", values)

        def failing_fsync(fd):
            raise OSError(5, "Input/output error")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(os, "fsync", failing_fsync)
            first = record_one(store, data_dir, info, values)

        assert first["ledger_status"] == "not_recorded"
        assert "Input/output error" in first["reason"]
        assert first["snapshot_status"] == "skipped"
        # The bytes reached the file before fsync failed.
        recorded = payloads(LedgerStore(root), EVENT_MEASUREMENT_RECORDED)
        assert len(recorded) == 1
        assert store.find_measurement_asset(first["measurement_id"]) is None

        again = record_one(LedgerStore(root), data_dir, copy.deepcopy(info), values)
        assert again["ledger_status"] == "already_recorded"
        assert again["snapshot_status"] == "partial"
        assert len(payloads(LedgerStore(root), EVENT_MEASUREMENT_RECORDED)) == 1
        assert len(payloads(LedgerStore(root), EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 1
        assert LedgerStore(root).find_measurement_asset(again["measurement_id"]) == (
            ASSET
        )

    def test_case_collision_with_an_existing_ledger_is_not_recorded(
        self, store, data_dir
    ):
        declare_point(store, asset="p-101", measurement_point_id=POINT)
        values = noise(1)
        info = make_info(data_dir / "m01.csv", values)  # declares "P-101"
        outcome = record_one(store, data_dir, info, values)
        assert outcome["ledger_status"] == "not_recorded"
        assert "letter case" in outcome["reason"]

    def test_infos_without_identity_are_a_caller_error(self, store, data_dir):
        info = make_info(data_dir / "m01.csv", noise(1))
        info["measurement"] = None
        with pytest.raises(ValueError, match="identity"):
            record(store, data_dir, [info], {"m01": noise(1)})

    def test_empty_batch_is_a_no_op(self, store, root, data_dir):
        assert record(store, data_dir, [], {}) == []
        assert not root.exists()


# ---------------------------------------------------------------------------
# Privacy and the wrong-asset correction
# ---------------------------------------------------------------------------


class TestPrivacyAndReattribution:
    def test_free_form_companion_keys_never_reach_the_ledger(
        self, store, root, data_dir
    ):
        values = noise(1)
        info = make_info(
            data_dir / "m01.csv",
            values,
            extra={"operator_name": "Mario Rossi", "shift_notes": "night shift"},
        )
        outcome = record_one(store, data_dir, info, values)
        assert outcome["ledger_status"] == "recorded"
        for path in root.iterdir():
            data = path.read_bytes()
            assert b"Mario Rossi" not in data
            assert b"operator_name" not in data
            assert b"night shift" not in data

    def test_wrong_asset_then_corrected(self, store, root, data_dir):
        """Loaded as P-102 by mistake, then re-loaded as P-101: recorded
        under P-101, P-102 keeps its bytes plus a superseding declaration,
        its view lists the id under ``reattributed``, the index says P-101."""
        values = noise(1)
        wrong = make_info(data_dir / "m01.csv", values, asset=OTHER_ASSET)
        first = record_one(store, data_dir, wrong, values)
        assert first["ledger_status"] == "recorded"
        assert store.find_measurement_asset(first["measurement_id"]) == OTHER_ASSET
        wrong_ledger = root / "P-102.jsonl"
        bytes_before = wrong_ledger.read_bytes()

        right = make_info(data_dir / "m01.csv", values, asset=ASSET)
        second = record_one(store, data_dir, right, values)
        assert second["ledger_status"] == "recorded"
        assert second["declaration_version"] == 1
        assert second["reattributed_from"] == OTHER_ASSET
        assert second["measurement_id"] == first["measurement_id"]
        assert len(second["event_ids"]) == 3  # recorded, supersession, snapshot

        assert wrong_ledger.exists()
        assert wrong_ledger.read_bytes().startswith(bytes_before)
        superseding = payloads(store, EVENT_MEASUREMENT_RECORDED, OTHER_ASSET)
        assert [p["declaration_version"] for p in superseding] == [1, 2]
        assert superseding[1]["changed"] == ["asset_id"]
        assert superseding[1]["declaration"]["asset_id"] == ASSET
        wrong_view = store.read_view(OTHER_ASSET)
        assert wrong_view["measurements"] == {}
        assert wrong_view["reattributed"] == [
            {"measurement_id": first["measurement_id"], "to_asset_id": ASSET}
        ]

        right_view = store.read_view(ASSET)
        assert right_view["ordered_measurement_ids"] == [first["measurement_id"]]
        assert (
            len(right_view["measurements"][first["measurement_id"]]["snapshots"]) == 1
        )
        assert store.find_measurement_asset(first["measurement_id"]) == ASSET
        assert store.list_assets() == [ASSET, OTHER_ASSET]

    def test_completed_reattribution_is_not_repeated_on_retry(self, store, data_dir):
        values = noise(1)
        wrong = make_info(data_dir / "m01.csv", values, asset=OTHER_ASSET)
        record_one(store, data_dir, wrong, values)
        right = make_info(data_dir / "m01.csv", values, asset=ASSET)
        record_one(store, data_dir, right, values)
        again = record_one(store, data_dir, copy.deepcopy(right), values)
        assert again["ledger_status"] == "already_recorded"
        assert again["reattributed_from"] is None  # the index already agrees
        assert again["event_ids"] == []
        assert len(payloads(store, EVENT_MEASUREMENT_RECORDED, OTHER_ASSET)) == 2

    def test_interrupted_reattribution_converges_on_retry(self, store, data_dir):
        """The index append fails after both ledgers were written: the load
        reports not_recorded; the retry finds the supersession already in
        place (appends nothing there), records nothing twice, and fixes the
        index."""
        values = noise(1)
        wrong = make_info(data_dir / "m01.csv", values, asset=OTHER_ASSET)
        first = record_one(store, data_dir, wrong, values)
        original = LedgerStore.append_index_entry

        def failing(self, *args, **kwargs):
            raise store_module.LedgerLockTimeout("index lock busy")

        right = make_info(data_dir / "m01.csv", values, asset=ASSET)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(LedgerStore, "append_index_entry", failing)
            interrupted = record_one(store, data_dir, right, values)
        assert interrupted["ledger_status"] == "not_recorded"
        assert "index lock busy" in interrupted["reason"]
        assert store.find_measurement_asset(first["measurement_id"]) == OTHER_ASSET
        assert len(payloads(store, EVENT_MEASUREMENT_RECORDED, OTHER_ASSET)) == 2

        retry = record_one(store, data_dir, copy.deepcopy(right), values)
        assert retry["ledger_status"] == "already_recorded"
        assert retry["reattributed_from"] == OTHER_ASSET
        assert retry["snapshot_status"] == "partial"
        assert len(payloads(store, EVENT_MEASUREMENT_RECORDED, ASSET)) == 1
        assert len(payloads(store, EVENT_MEASUREMENT_RECORDED, OTHER_ASSET)) == 2
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED, ASSET)) == 1
        assert store.find_measurement_asset(first["measurement_id"]) == ASSET
        assert LedgerStore.append_index_entry is original

    def test_moving_back_supersedes_both_ledgers(self, store, data_dir):
        """P-102 -> P-101 -> P-102: the third load supersedes the reattributed
        history of P-102 (version 3) and reattributes P-101 in turn."""
        values = noise(1)
        mid = None
        for asset in (OTHER_ASSET, ASSET, OTHER_ASSET):
            outcome = record_one(
                store,
                data_dir,
                make_info(data_dir / "m01.csv", values, asset=asset),
                values,
            )
            mid = outcome["measurement_id"]
        assert outcome["ledger_status"] == "superseded"
        assert outcome["changed"] == ["asset_id"]
        assert outcome["declaration_version"] == 3
        assert outcome["reattributed_from"] == ASSET
        assert [
            p["declaration_version"]
            for p in payloads(store, EVENT_MEASUREMENT_RECORDED, OTHER_ASSET)
        ] == [1, 2, 3]
        assert [
            p["declaration_version"]
            for p in payloads(store, EVENT_MEASUREMENT_RECORDED, ASSET)
        ] == [1, 2]
        assert store.read_view(ASSET)["reattributed"] == [
            {"measurement_id": mid, "to_asset_id": OTHER_ASSET}
        ]
        assert store.read_view(OTHER_ASSET)["ordered_measurement_ids"] == [mid]
        assert store.find_measurement_asset(mid) == OTHER_ASSET
        # One snapshot per ledger: the context did not change with the asset.
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED, OTHER_ASSET)) == 1
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED, ASSET)) == 1


# ---------------------------------------------------------------------------
# Point context for later units
# ---------------------------------------------------------------------------


class TestResolvePointContext:
    def test_none_without_ledger_or_declaration(self, store):
        assert resolve_point_context(store, ASSET, POINT) is None
        declare_point(store)
        assert resolve_point_context(store, ASSET, "other_point") is None

    def test_latest_declaration_is_returned_as_a_copy(self, store):
        declare_point(store)
        declare_point(
            store, declaration_version=2, machine_group=1, changed=["machine_group"]
        )
        context = resolve_point_context(store, ASSET, POINT)
        assert context["declaration_version"] == 2
        assert context["machine_group"] == 1
        context["machine_group"] = 99
        assert resolve_point_context(store, ASSET, POINT)["machine_group"] == 1

    def test_invalid_asset_id_is_refused(self, store):
        with pytest.raises(ValueError):
            resolve_point_context(store, "../x", POINT)


# ---------------------------------------------------------------------------
# Module purity
# ---------------------------------------------------------------------------


class TestModulePurity:
    def test_service_imports_neither_repository_nor_models_nor_mcp(self):
        import ast

        source = Path(service.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "repository" not in module, module
                assert module != "models" and not module.endswith(".models"), module
                assert "mcp" not in module.split("."), module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("mcp"), alias.name

    def test_package_reexports_the_service_api(self):
        import predictive_maintenance_mcp.asset_ledger as pkg

        for name in (
            "record_measurements",
            "resolve_point_context",
            "build_declaration",
            "file_block",
            "declaration_fingerprint",
            "changed_keys",
            "OUTCOME_KEYS",
            "LOAD_OUTCOME_KEYS",
        ):
            assert getattr(pkg, name) is getattr(service, name), name

    def test_outcomes_are_json_serializable(self, store, data_dir):
        values = noise(1)
        outcome = record_one(
            store, data_dir, make_info(data_dir / "m01.csv", values), values
        )
        json.dumps(outcome, allow_nan=False)


# ---------------------------------------------------------------------------
# Re-processing of stale snapshots (U6)
# ---------------------------------------------------------------------------


class TestReprocessStaleSnapshots:
    def test_contract_constants(self):
        assert MAX_REPROCESS_PER_CALL == 10
        assert REPROCESS_OUTCOMES == ("reprocessed", "up_to_date", "not_reprocessable")

    def test_twenty_five_stale_with_limit_ten_over_three_calls(self, store, data_dir):
        """Reference first, then the last K newest first, then the rest
        newest first; the assessment succeeds after every call because the
        old lineage still covers everything, and the old snapshots stay."""
        declare_point(store)
        ids = load_sequence(store, data_dir, 25)
        old, new = processing_id(OLD_POLICY), processing_id()

        before = assess_change(store.read_view(ASSET), ASSET, POINT)
        assert before["status"] == "assessed"
        assert before["lineage"]["processing_id"] == old
        assert before["lineage"]["is_current"] is False
        assert before["lineage"]["missing_for_current"] == 25

        first = reprocess(store, data_dir)
        assert first["processing_id"] == new
        assert first["stale"] == 25
        assert first["reprocessed"] == 10
        assert first["not_reprocessable"] == 0
        assert first["up_to_date"] == 0
        assert first["remaining"] == 15
        assert [r["measurement_id"] for r in first["results"]] == ids[:10]
        assert [r["outcome"] for r in first["results"]] == ["reprocessed"] * 10
        assert [r["location_used"] for r in first["results"]] == [
            f"m{i:02d}.csv" for i in range(1, 11)
        ]
        assert all(len(r["snapshot_id"]) == 16 for r in first["results"])
        assert first["next_call"] == REPROCESS_CALL
        assert REPROCESS_CALL in first["message"]
        assert "error" not in set(walk(first))
        json.dumps(first, allow_nan=False)

        second = reprocess(store, data_dir)
        assert second["reprocessed"] == 10
        assert second["remaining"] == 5
        # M25..M21 (the last K, newest first), then M20..M16.
        assert [r["measurement_id"] for r in second["results"]] == (
            ids[24:19:-1] + ids[19:14:-1]
        )
        after_two = assess_change(store.read_view(ASSET), ASSET, POINT)
        assert after_two["status"] == "assessed"
        assert after_two["lineage"]["candidates"] == {old: 25, new: 20}
        assert after_two["lineage"]["processing_id"] == old

        third = reprocess(store, data_dir)
        assert third["reprocessed"] == 5
        assert third["remaining"] == 0
        assert third["next_call"] is None
        assert [r["measurement_id"] for r in third["results"]] == ids[14:9:-1]
        final = assess_change(store.read_view(ASSET), ASSET, POINT)
        assert final["status"] == "assessed"
        assert final["lineage"]["processing_id"] == new
        assert final["lineage"]["is_current"] is True
        assert final["lineage"]["candidates"] == {old: 25, new: 25}
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 50

        again = reprocess(store, data_dir)
        assert again["stale"] == 0
        assert again["reprocessed"] == 0
        assert again["up_to_date"] == 25
        assert again["results"] == []
        assert again["next_call"] is None
        assert "nothing to reprocess" in again["message"]
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 50

    def test_baseline_members_come_first(self, store, data_dir):
        declare_point(store)
        ids = load_sequence(store, data_dir, 12)
        members = [ids[i] for i in (4, 6, 8)]
        payload = {
            "baseline_id": store_module.short_id(POINT, *sorted(members), "t"),
            "measurement_point_id": POINT,
            "measurement_ids": members,
            "members": [
                {
                    "measurement_id": m,
                    "declaration_version": 1,
                    "point_declaration_version": 1,
                }
                for m in members
            ],
            "declared_by": "L. Rossi",
            "note": None,
            "declared_at": "2026-04-01T00:00:00+00:00",
        }
        store.append(
            ASSET, make_event(store_module.EVENT_BASELINE_DECLARED, ASSET, payload)
        )
        result = reprocess(store, data_dir, limit=5)
        # The three members, then the two newest of the last K.
        assert [r["measurement_id"] for r in result["results"]] == members + [
            ids[11],
            ids[10],
        ]
        assert result["remaining"] == 7

    def test_file_missing_everywhere_is_not_reprocessable(self, store, data_dir):
        declare_point(store)
        load_sequence(store, data_dir, 1)
        (data_dir / "m01.csv").unlink()
        result = reprocess(store, data_dir)
        assert result["stale"] == 1
        assert result["reprocessed"] == 0
        assert result["not_reprocessable"] == 1
        assert result["remaining"] == 0
        assert result["next_call"] is None
        entry = result["results"][0]
        assert entry["outcome"] == "not_reprocessable"
        assert entry["location_used"] is None
        assert "file not found at m01.csv" in entry["reason"]
        assert entry["acquired_at"] == "2026-01-05T08:00:00+00:00"
        # The old snapshot is kept untouched.
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 1

    def test_changed_content_is_not_reprocessable(self, store, data_dir):
        declare_point(store)
        load_sequence(store, data_dir, 1)
        write_signal(data_dir / "m01.csv", noise(99, 5000))
        result = reprocess(store, data_dir)
        entry = result["results"][0]
        assert entry["outcome"] == "not_reprocessable"
        assert "content differs at m01.csv" in entry["reason"]
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 1

    def test_undeclared_sampling_rate_is_not_reprocessable(self, store, data_dir):
        """Recorded without a rate (the snapshot was skipped at load) and
        stale since the point was declared: the re-processing names the
        remedy per measurement instead of guessing a rate."""
        values = noise(1)
        info = make_info(data_dir / "m01.csv", values, sampling_rate=None)
        assert record_one(store, data_dir, info, values)["snapshot_status"] == "skipped"
        declare_point(store)
        result = reprocess(store, data_dir)
        assert result["stale"] == 1
        assert result["reprocessed"] == 0
        assert result["not_reprocessable"] == 1
        assert result["remaining"] == 0
        entry = result["results"][0]
        assert entry["outcome"] == "not_reprocessable"
        assert entry["location_used"] is None
        assert entry["snapshot_id"] is None
        assert "sampling_rate not declared" in entry["reason"]
        assert "re-load the file" in entry["reason"]
        assert payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED) == []

    def test_undecodable_declaration_is_not_reprocessable(
        self, store, data_dir, monkeypatch
    ):
        """The file is found and its hash verified, but the recorded
        declaration cannot decode it: reported per measurement with the
        decoder's reason, never raised, and the old snapshot stays."""
        declare_point(store)
        load_sequence(store, data_dir, 1)

        def refusing(path: Path, declaration: Any) -> np.ndarray:
            raise ValueError(
                "recorded raw_format declares no sample_format; re-load the file "
                "with the raw declaration (the new declaration supersedes this one)"
            )

        monkeypatch.setattr(service, "_decode", refusing)
        result = reprocess(store, data_dir)
        assert result["stale"] == 1
        assert result["reprocessed"] == 0
        assert result["not_reprocessable"] == 1
        assert result["remaining"] == 0
        entry = result["results"][0]
        assert entry["outcome"] == "not_reprocessable"
        assert entry["location_used"] == "m01.csv"
        assert entry["snapshot_id"] is None
        assert entry["reason"].startswith("snapshot not computed:")
        assert "sample_format" in entry["reason"]
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 1

    def record_unloadable(
        self, store: LedgerStore, data_dir: Path, path: Path, values: np.ndarray, **kw
    ) -> None:
        """Record *path* (already written, bytes the store would now refuse)
        with the clean *values* on the old lineage, so it is stale."""
        info = make_info(path, values, acquired_at=acquired(1), **kw)
        [outcome] = record(
            store, data_dir, [info], {info["signal_id"]: values}, policy=OLD_POLICY
        )
        assert outcome["ledger_status"] == "recorded"
        assert outcome["snapshot_status"] == "complete"

    def assert_refused_on_reread(
        self, store: LedgerStore, data_dir: Path, location: str
    ) -> str:
        result = reprocess(store, data_dir)
        assert (
            result["stale"],
            result["reprocessed"],
            result["not_reprocessable"],
        ) == (
            1,
            0,
            1,
        )
        entry = result["results"][0]
        assert entry["outcome"] == "not_reprocessable"
        assert entry["location_used"] == location
        assert entry["snapshot_id"] is None
        assert entry["reason"].startswith("snapshot not computed:")
        assert location in entry["reason"]
        # Only the snapshot computed at load time; nothing appended.
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 1
        return entry["reason"]

    def test_csv_with_an_internal_blank_line_is_not_reprocessable(
        self, store, data_dir
    ):
        """The CSV reader skips the empty line, which drops a sample and
        shifts every later one: the re-read refuses it and names the line."""
        declare_point(store)
        values = noise(1, 5000)
        path = data_dir / "m01.csv"
        write_signal(path, values)
        lines = path.read_bytes().split(b"\n")
        path.write_bytes(b"\n".join(lines[:10] + [b""] + lines[10:]))
        self.record_unloadable(store, data_dir, path, values)

        reason = self.assert_refused_on_reread(store, data_dir, "m01.csv")
        assert "empty line at line 11" in reason

    def test_npy_with_a_nan_is_not_reprocessable(self, store, data_dir):
        declare_point(store)
        values = noise(1, 5000)
        corrupt = values.copy()
        corrupt[42] = np.nan
        path = data_dir / "m01.npy"
        np.save(path, corrupt)
        self.record_unloadable(store, data_dir, path, values)

        reason = self.assert_refused_on_reread(store, data_dir, "m01.npy")
        assert "1 non-finite sample(s)" in reason
        assert "out of 5000" in reason
        assert "sample index 42" in reason

    def test_raw_file_decoding_to_infinity_names_the_raw_declaration(
        self, store, data_dir
    ):
        """A scale_factor large enough to overflow one sample: the remedy
        points at the recorded raw declaration, not only at the capture."""
        declare_point(store)
        values = noise(1, 5000)
        values[42] = 10.0  # 10 x 1e308 overflows float64; |noise| < 1 does not
        path = data_dir / "m01.bin"
        written = write_raw_file(path, values).astype(np.float64)
        raw = {
            "sample_format": "float32",
            "byte_order": "little",
            "n_channels": 1,
            "channel_index": 0,
            "header_offset": 0,
            "scale_factor": 1e308,
        }
        self.record_unloadable(store, data_dir, path, written, raw_format=raw)

        reason = self.assert_refused_on_reread(store, data_dir, "m01.bin")
        assert "1 non-finite sample(s)" in reason
        assert "sample index 42" in reason
        assert "scale_factor" in reason

    def test_collapsed_duplicates_are_reprocessed_too(self, store, data_dir):
        """Two exports of one capture (other bytes, same acquisition slot):
        the slot collapse keeps the latest recorded, but the point
        declaration counts both as stale, so the remedy it names must
        recompute both before that count returns to zero."""
        declare_point(store)
        arrays = {"m01": noise(1, 5000), "m01_again": noise(2, 5000)}
        infos = [
            make_info(data_dir / f"{name}.csv", values, acquired_at=acquired(1))
            for name, values in arrays.items()
        ]
        outcomes = record(store, data_dir, infos, arrays, policy=None)
        assert [o["ledger_status"] for o in outcomes] == ["recorded", "recorded"]
        duplicate, kept = (o["measurement_id"] for o in outcomes)

        def redeclare() -> dict[str, Any]:
            return declare_measurement_point(
                store=store,
                asset_id=ASSET,
                measurement_point_id=POINT,
                bearing_id="6205",
                machine_group=1,
                support_type="rigid",
                expected_signal_unit="g",
                expected_direction="horizontal",
                nominal_rpm=1800.0,
                declared_by="test",
            )

        redeclared = redeclare()
        assert redeclared["declaration_version"] == 2
        assert redeclared["changed"] == ["machine_group"]
        assert redeclared["measurements_with_stale_context"] == 2
        assert redeclared["remedy"] == REPROCESS_CALL

        # One measurement per call, so the collapsed tail needs a second call.
        first = reprocess(store, data_dir, limit=1)
        assert (first["stale"], first["reprocessed"], first["remaining"]) == (2, 1, 1)
        assert first["next_call"] == REPROCESS_CALL
        # The slot's representative first, the collapsed duplicate last.
        assert [r["measurement_id"] for r in first["results"]] == [kept]
        second = reprocess(store, data_dir, limit=1)
        assert (second["stale"], second["reprocessed"], second["remaining"]) == (
            1,
            1,
            0,
        )
        assert second["not_reprocessable"] == 0
        assert second["next_call"] is None
        assert [r["measurement_id"] for r in second["results"]] == [duplicate]

        again = redeclare()
        assert again["appended"] is False
        assert again["measurements_with_stale_context"] == 0
        assert again["remedy"] is None
        assert reprocess(store, data_dir)["stale"] == 0

    def test_deleted_temporary_copy_falls_back_to_the_original(
        self, store, data_dir, tmp_path
    ):
        declare_point(store)
        values = noise(1, 5000)
        original = data_dir / "m01.csv"
        first = record(
            store,
            data_dir,
            [make_info(original, values, acquired_at=acquired(1))],
            {"m01": values},
            policy=OLD_POLICY,
        )[0]
        temp = tmp_path / "scratch" / "m01.csv"
        temp.parent.mkdir()
        shutil.copy(original, temp)
        second = record(
            store,
            data_dir,
            [make_info(temp, values, acquired_at=acquired(1))],
            {"m01": values},
            policy=OLD_POLICY,
        )[0]
        assert second["ledger_status"] == "superseded"
        assert second["changed"] == ["location"]
        temp.unlink()

        result = reprocess(store, data_dir)
        entry = result["results"][0]
        assert entry["outcome"] == "reprocessed"
        assert entry["location_used"] == "m01.csv"
        lineages = store.read_view(ASSET)["measurements"][first["measurement_id"]][
            "snapshots_by_lineage"
        ]
        assert set(lineages) == {processing_id(OLD_POLICY), processing_id()}

    def test_raw_file_is_decoded_with_the_recorded_declaration(self, store, data_dir):
        declare_point(store)
        values = noise(1, 5000)
        path = data_dir / "m01.bin"
        written = write_raw_file(path, values)
        raw = {
            "sample_format": "float32",
            "byte_order": "little",
            "n_channels": 1,
            "channel_index": 0,
            "header_offset": 0,
            "scale_factor": None,
        }
        info = make_info(path, values, raw_format=raw, acquired_at=acquired(1))
        outcome = record(
            store,
            data_dir,
            [info],
            {"m01": written.astype(np.float64)},
            policy=OLD_POLICY,
        )[0]
        assert outcome["snapshot_status"] == "complete"

        result = reprocess(store, data_dir)
        assert result["results"][0]["outcome"] == "reprocessed"
        assert result["results"][0]["location_used"] == "m01.bin"
        lineages = store.read_view(ASSET)["measurements"][outcome["measurement_id"]][
            "snapshots_by_lineage"
        ]
        old = lineages[processing_id(OLD_POLICY)]
        new = lineages[processing_id()]
        assert new["indicators"]["rms"] == pytest.approx(old["indicators"]["rms"])
        assert new["iso"]["velocity_rms_mm_s"] == pytest.approx(
            old["iso"]["velocity_rms_mm_s"]
        )

    def test_later_point_declaration_makes_the_snapshot_stale(self, store, data_dir):
        """Loaded before the point was declared: the snapshot is on the
        current lineage but with the old context (no ISO block); the
        re-processing computes it with the current declaration."""
        [measurement_id] = load_sequence(store, data_dir, 1, policy=None)
        declare_point(store)
        result = reprocess(store, data_dir)
        assert result["stale"] == 1
        assert result["reprocessed"] == 1
        snapshots = store.read_view(ASSET)["measurements"][measurement_id]["snapshots"]
        assert len(snapshots) == 2
        assert snapshots[0]["iso"] is None
        assert snapshots[1]["iso"] is not None
        assert snapshots[1]["point_declaration_version"] == 1
        assert snapshots[0]["context_digest"] != snapshots[1]["context_digest"]
        assert reprocess(store, data_dir)["stale"] == 0

    def test_append_failure_is_reported_not_raised(self, store, data_dir, monkeypatch):
        declare_point(store)
        load_sequence(store, data_dir, 2)

        def failing(self, asset_id, event):
            raise store_module.LedgerWriteError("disk full while appending")

        monkeypatch.setattr(LedgerStore, "append", failing)
        result = reprocess(store, data_dir)
        assert result["reprocessed"] == 0
        assert result["not_reprocessable"] == 2
        for entry in result["results"]:
            assert entry["outcome"] == "not_reprocessable"
            assert "not appended" in entry["reason"]
            assert "disk full" in entry["reason"]
            assert entry["snapshot_id"] is not None
        assert result["remaining"] == 0

    def test_limit_is_validated_and_an_empty_point_has_nothing_to_do(
        self, store, data_dir
    ):
        with pytest.raises(ValueError, match="limit"):
            reprocess(store, data_dir, limit=0)
        declare_point(store)
        result = reprocess(store, data_dir)
        assert result["stale"] == 0
        assert result["results"] == []
        assert result["next_call"] is None
        assert "nothing to reprocess" in result["message"]

    def test_package_reexports_the_reprocessing_api(self):
        import predictive_maintenance_mcp.asset_ledger as pkg

        assert pkg.reprocess_stale_snapshots is reprocess_stale_snapshots
        assert pkg.MAX_REPROCESS_PER_CALL is MAX_REPROCESS_PER_CALL


# ---------------------------------------------------------------------------
# A declaration changed and changed back, end to end
# ---------------------------------------------------------------------------


def declare(
    store: LedgerStore, machine_group: int, support_type: str = "rigid"
) -> dict[str, Any]:
    """The point's declaration through the service, varying the ISO context."""
    return declare_measurement_point(
        store=store,
        asset_id=ASSET,
        measurement_point_id=POINT,
        bearing_id="6205",
        machine_group=machine_group,
        support_type=support_type,
        expected_signal_unit="g",
        expected_direction="horizontal",
        nominal_rpm=1800.0,
        declared_by="test",
    )


def assess_point(store: LedgerStore) -> dict[str, Any]:
    return assess_change(store.read_view(ASSET), ASSET, POINT)


def sequence_infos(
    data_dir: Path, count: int
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """The infos and arrays ``load_sequence`` records, for a re-load."""
    infos: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for index in range(1, count + 1):
        values = noise(index, 5000)
        info = make_info(
            data_dir / f"m{index:02d}.csv", values, acquired_at=acquired(index)
        )
        infos.append(info)
        arrays[info["signal_id"]] = values
    return infos, arrays


class TestRestoredDeclaration:
    """A, B, then A again through the real service: the snapshots A
    produced are already recorded, so nothing is appended, and the
    assessment, the point declaration and re-processing agree that nothing
    is stale."""

    def test_point_revert_reuses_the_recorded_snapshots(self, store, data_dir, root):
        """A, B (re-processed), then A: nothing is stale or written."""
        declare(store, 2)
        load_sequence(store, data_dir, 11, policy=None)
        assert declare(store, 1)["measurements_with_stale_context"] == 11
        assert reprocess(store, data_dir, limit=11)["reprocessed"] == 11
        restored = declare(store, 2)
        assert restored["declaration_version"] == 3
        assert restored["measurements_with_stale_context"] == 0
        assert restored["remedy"] is None

        ledger = root / f"{ASSET}{store_module.LEDGER_SUFFIX}"
        before = ledger.read_bytes()
        result = assess_point(store)
        again = reprocess(store, data_dir)
        assert (again["stale"], again["reprocessed"]) == (0, 0)
        assert "nothing to reprocess" in again["message"]
        assert ledger.read_bytes() == before

        assert result["status"] == "assessed"
        assert result["lineage"]["is_current"] is True
        assert result["lineage"]["stale_context"] == 0
        assert result["derived"]["iso_change"]["machine_group"] == 2
        assert REPROCESS_CALL not in result["message"]
        assert store_module.canonical_json(result) == store_module.canonical_json(
            assess_point(store)
        )

    def test_reload_under_b_then_revert(self, store, data_dir):
        """Re-loading unchanged files while B is declared
        appends B's snapshots at load time, with no re-processing call."""
        declare(store, 2)
        infos, arrays = sequence_infos(data_dir, 11)
        record(store, data_dir, infos, arrays)
        declare(store, 1)
        before = len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED))
        reloaded = record(store, data_dir, infos, arrays)
        assert {o["ledger_status"] for o in reloaded} == {"already_recorded"}
        assert len(payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)) == before + 11
        declare(store, 2)
        result = assess_point(store)
        assert result["lineage"]["stale_context"] == 0
        assert result["derived"]["iso_change"]["machine_group"] == 2

    def test_measurement_revert_reuses_the_recorded_snapshot(self, store, data_dir):
        """The latest measurement re-loaded with rpm 1500, then
        with 1800 again; the third load appends no snapshot and its
        snapshot_id is the one whose values the assessment uses."""
        declare(store, 2)
        load_sequence(store, data_dir, 10, policy=None)
        values = noise(11, 5000)
        path = data_dir / "m11.csv"
        first = record_one(
            store, data_dir, make_info(path, values, acquired_at=acquired(11)), values
        )
        changed = record_one(
            store,
            data_dir,
            make_info(path, values, acquired_at=acquired(11), rpm=1500),
            values,
        )
        assert changed["ledger_status"] == "superseded"
        assert changed["snapshot_id"] != first["snapshot_id"]
        restored = record_one(
            store, data_dir, make_info(path, values, acquired_at=acquired(11)), values
        )
        assert restored["ledger_status"] == "superseded"
        assert restored["snapshot_id"] == first["snapshot_id"]
        appended = {
            e["event_type"]
            for e in events_of(store)
            if e["event_id"] in set(restored["event_ids"])
        }
        assert appended == {EVENT_MEASUREMENT_RECORDED}

        by_id = {
            p["snapshot_id"]: p for p in payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)
        }
        one_x_a = by_id[first["snapshot_id"]]["one_x"]["amplitude"]
        one_x_b = by_id[changed["snapshot_id"]]["one_x"]["amplitude"]
        assert one_x_a != pytest.approx(one_x_b)
        result = assess_point(store)
        assert result["lineage"]["stale_context"] == 0
        assert result["observed"]["latest_measurement_id"] == first["measurement_id"]
        assert result["observed"]["latest"]["one_x"] == pytest.approx(one_x_a)

    def test_every_surface_agrees_on_what_is_stale(self, store, data_dir):
        """After each declaration of A, B, A, C, with re-processing only after
        B: the point declaration's stale count is the assessment's, and the
        assessment names the re-processing call exactly when that count is
        above zero."""
        declare(store, 2)
        load_sequence(store, data_dir, 11, policy=None)
        steps = ((2, "rigid", False), (1, "rigid", True), (2, "rigid", False))
        for group, support, reprocess_after in steps + ((2, "flexible", False),):
            declared = declare(store, group, support)
            result = assess_point(store)
            stale = result["lineage"]["stale_context"]
            assert declared["measurements_with_stale_context"] == stale
            assert (REPROCESS_CALL in result["message"]) == (stale > 0)
            if reprocess_after:
                reprocess(store, data_dir, limit=11)
        assert stale == 11


# ---------------------------------------------------------------------------
# Progress of a re-processing sequence: following next_call reaches an end
# ---------------------------------------------------------------------------


def call_cap(eligible: int, limit: int = MAX_REPROCESS_PER_CALL) -> int:
    """Calls a follower may need: one, one per *limit* eligible
    measurements, and one more that ends the sequence."""
    return 1 + -(-eligible // limit) + 1


def follow(
    store: LedgerStore, data_dir: Path, *, cap: int, **kwargs: Any
) -> list[dict[str, Any]]:
    """Make the calls a caller is told to make: repeat while ``next_call``
    is set, carrying ``next_cursor`` when the response gives one; fail once
    *cap* calls still name a next call."""
    responses: list[dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        extra = {} if cursor is None else {"cursor": cursor}
        response = reprocess(store, data_dir, **kwargs, **extra)
        responses.append(response)
        if response["next_call"] is None:
            return responses
        if len(responses) >= cap:
            pytest.fail(
                f"still naming a next call after {cap} call(s): {response['message']}"
            )
        cursor = response.get("next_cursor")


def stale_ids(store: LedgerStore) -> set[str]:
    """The measurements the point declaration counts as stale."""
    view = store.read_view(ASSET)
    return set(
        service._stale_measurement_ids(view, POINT, service._current_point(view, POINT))
    )


def reported_failures(responses: list[dict[str, Any]]) -> dict[str, str]:
    """Every measurement some response reported not reprocessable, with
    its reason."""
    return {
        entry["measurement_id"]: entry["reason"]
        for response in responses
        for entry in response["results"]
        if entry["outcome"] == "not_reprocessable"
    }


def same_size_change(path: Path) -> None:
    """Other bytes of the same length: the lines in reverse order."""
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(reversed(lines)))
    assert len(lines) > 1


def current_lineage_snapshots(store: LedgerStore) -> dict[str, int]:
    """Snapshot events per measurement on the current lineage."""
    counts: dict[str, int] = {}
    for payload in payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED):
        if payload["processing"]["processing_id"] == processing_id():
            counts[payload["measurement_id"]] = (
                counts.get(payload["measurement_id"], 0) + 1
            )
    return counts


class TestReprocessProgress:
    """Following next_call ends, and every eligible stale measurement is
    re-processed, however many measurements ahead of it cannot be."""

    def test_s1_missing_reference_files_do_not_starve_the_later_slots(
        self, store, data_dir
    ):
        """S1: a re-declaration makes 20 measurements stale; the files of
        the 12 reference members are gone, the 8 later ones are present."""
        declare(store, 2)
        ids = load_sequence(store, data_dir, 20, policy=None)
        declare(store, 1)
        for index in range(1, 13):
            (data_dir / f"m{index:02d}.csv").unlink()
        params = AssessmentParams(reference_measurements=12)

        responses = follow(store, data_dir, cap=call_cap(8), params=params)

        assert stale_ids(store) == set(ids[:12])
        failures = reported_failures(responses)
        assert set(failures) == set(ids[:12])
        assert all("file not found at" in reason for reason in failures.values())

    def test_s2_more_failures_than_the_limit_are_all_reported(self, store, data_dir):
        """S2: 15 stale measurements whose files were rewritten with other
        bytes of the same length; none can be re-processed."""
        declare_point(store)
        ids = load_sequence(store, data_dir, 15)
        for index in range(1, 16):
            same_size_change(data_dir / f"m{index:02d}.csv")

        responses = follow(store, data_dir, cap=call_cap(0))

        failures = reported_failures(responses)
        assert set(failures) == set(ids)
        assert all("content differs at" in reason for reason in failures.values())
        assert stale_ids(store) == set(ids)

    def test_s3_a_limit_of_one_reaches_the_eligible_measurements(self, store, data_dir):
        """S3: limit=1 and the first measurement in priority cannot be
        re-processed; no call repeats the previous one."""
        declare_point(store)
        ids = load_sequence(store, data_dir, 3)
        same_size_change(data_dir / "m01.csv")

        responses = follow(store, data_dir, cap=call_cap(2, limit=1), limit=1)

        assert stale_ids(store) == {ids[0]}
        dumps = [store_module.canonical_json(r) for r in responses]
        assert all(a != b for a, b in zip(dumps, dumps[1:]))

    def test_s6_few_failures_keep_todays_order_and_call_count(self, store, data_dir):
        """S6: 15 stale, two files missing among the first ten in priority.
        Characterization of today's outcome: the same measurements are
        re-processed in the same order, in no more than today's two calls,
        and each gets exactly one snapshot of the current lineage."""
        declare_point(store)
        ids = load_sequence(store, data_dir, 15)
        for index in (3, 7):
            (data_dir / f"m{index:02d}.csv").unlink()

        responses = follow(store, data_dir, cap=call_cap(13))

        reprocessed = [
            entry["measurement_id"]
            for response in responses
            for entry in response["results"]
            if entry["outcome"] == "reprocessed"
        ]
        priority = ids[:10] + ids[14:9:-1]
        assert reprocessed == [m for m in priority if m not in (ids[2], ids[6])]
        assert len(responses) <= 2
        assert stale_ids(store) == {ids[2], ids[6]}
        assert current_lineage_snapshots(store) == dict.fromkeys(reprocessed, 1)


# ---------------------------------------------------------------------------
# Cost tiers and reason codes
# ---------------------------------------------------------------------------

#: The reason-code table of the re-processing plan, in tier order.
REASON_CODE_TABLE = (
    "no_valid_sampling_rate",
    "declaration_unresolvable",
    "raw_format_incomplete",
    "no_file_recorded",
    "location_outside_data_dir",
    "file_missing",
    "content_differs",
    "file_inaccessible",
    "undecodable",
    "append_failed",
    "superseded_in_slot",
)


def supersede(
    store: LedgerStore,
    measurement_id: str,
    *,
    declaration: Optional[dict[str, Any]] = None,
    file: Optional[dict[str, Any]] = None,
    asset: str = ASSET,
    **changes: Any,
) -> None:
    """Append the next declaration version of a recorded measurement with
    some declaration, file or top-level values changed."""
    current = copy.deepcopy(
        store.read_view(asset)["measurements"][measurement_id]["current"]
    )
    current["declaration_version"] = int(current["declaration_version"]) + 1
    current["declaration"].update(declaration or {})
    current["file"].update(file or {})
    current.update(changes)
    store.append(asset, make_event(EVENT_MEASUREMENT_RECORDED, asset, current))


class _OsProxy:
    """The ``os`` module as the service sees it, with ``stat`` replaced."""

    def __init__(self, stat):
        self.stat = stat

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)


@pytest.fixture
def spy(monkeypatch) -> dict[str, list[str]]:
    """Record every file check and every costly step re-processing makes:
    containment checks and metadata calls by location or path, hashes and
    decodes by file name, and snapshot computations."""
    calls: dict[str, list[str]] = {
        "resolve": [],
        "stat": [],
        "digest": [],
        "decode": [],
        "compute": [],
    }
    real_resolve = service.safe_resolve
    real_digest = service.digest_file
    real_decode = service._decode
    real_compute = service.compute_health_snapshot

    def resolve(base: Path, location: str) -> Path:
        calls["resolve"].append(location)
        return real_resolve(base, location)

    def stat_of(path: Any, *args: Any, **kwargs: Any) -> Any:
        calls["stat"].append(Path(path).name)
        return os.stat(path, *args, **kwargs)

    def digest(path: Path) -> Any:
        calls["digest"].append(Path(path).name)
        return real_digest(path)

    def decode(path: Path, declaration: Any) -> np.ndarray:
        calls["decode"].append(Path(path).name)
        return real_decode(path, declaration)

    def compute(*args: Any, **kwargs: Any) -> Any:
        calls["compute"].append("snapshot")
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(service, "safe_resolve", resolve)
    monkeypatch.setattr(service, "os", _OsProxy(stat_of))
    monkeypatch.setattr(service, "digest_file", digest)
    monkeypatch.setattr(service, "_decode", decode)
    monkeypatch.setattr(service, "compute_health_snapshot", compute)
    return calls


def refuse_stat_of(monkeypatch, name: str, error: OSError) -> None:
    """The metadata call on the file named *name* raises *error*."""

    def stat_of(path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path).name == name:
            raise error
        return os.stat(path, *args, **kwargs)

    monkeypatch.setattr(service, "os", _OsProxy(stat_of))


def only(response: dict[str, Any], measurement_id: str) -> dict[str, Any]:
    [entry] = [e for e in response["results"] if e["measurement_id"] == measurement_id]
    return entry


def one_stale(store: LedgerStore, data_dir: Path) -> str:
    declare_point(store)
    [measurement_id] = load_sequence(store, data_dir, 1)
    return measurement_id


def _no_rate(store, data_dir, monkeypatch) -> str:
    measurement_id = one_stale(store, data_dir)
    supersede(store, measurement_id, declaration={"sampling_rate": None})
    return measurement_id


def _unresolvable(store, data_dir, monkeypatch) -> str:
    measurement_id = one_stale(store, data_dir)
    supersede(store, measurement_id, declaration={"direction": "diagonal"})
    return measurement_id


def _raw_incomplete(store, data_dir, monkeypatch) -> str:
    measurement_id = one_stale(store, data_dir)
    raw = {
        "sample_format": None,
        "byte_order": "little",
        "n_channels": 1,
        "channel_index": 0,
        "header_offset": 0,
        "scale_factor": None,
    }
    supersede(store, measurement_id, declaration={"raw_format": raw})
    return measurement_id


def _no_file(store, data_dir, monkeypatch) -> str:
    measurement_id = one_stale(store, data_dir)
    supersede(store, measurement_id, file={"content_sha256": None})
    return measurement_id


def _outside(store, data_dir, monkeypatch) -> str:
    measurement_id = one_stale(store, data_dir)
    location = "../outside/m01.csv"
    supersede(store, measurement_id, file={"location": location}, locations=[location])
    return measurement_id


def _missing(store, data_dir, monkeypatch) -> str:
    measurement_id = one_stale(store, data_dir)
    (data_dir / "m01.csv").unlink()
    return measurement_id


def _differs(store, data_dir, monkeypatch) -> str:
    measurement_id = one_stale(store, data_dir)
    write_signal(data_dir / "m01.csv", noise(99, 5001))
    return measurement_id


def _inaccessible(store, data_dir, monkeypatch) -> str:
    measurement_id = one_stale(store, data_dir)
    refuse_stat_of(monkeypatch, "m01.csv", PermissionError(13, "Permission denied"))
    return measurement_id


def _undecodable(store, data_dir, monkeypatch) -> str:
    measurement_id = one_stale(store, data_dir)

    def refusing(path: Path, declaration: Any) -> np.ndarray:
        raise ValueError(f"Recorded file '{path.name}' has 3 non-finite sample(s)")

    monkeypatch.setattr(service, "_decode", refusing)
    return measurement_id


def _append_failed(store, data_dir, monkeypatch) -> str:
    measurement_id = one_stale(store, data_dir)

    def failing(self, asset_id, event):
        raise store_module.LedgerWriteError("disk full while appending")

    monkeypatch.setattr(LedgerStore, "append", failing)
    return measurement_id


def _superseded(store, data_dir, monkeypatch) -> str:
    """Two exports of one capture: the earlier recorded one is collapsed
    into the slot of the later one, and its file is gone."""
    declare_point(store)
    arrays = {"m01": noise(1, 5000), "m01_again": noise(2, 5000)}
    infos = [
        make_info(data_dir / f"{name}.csv", values, acquired_at=acquired(1))
        for name, values in arrays.items()
    ]
    outcomes = record(store, data_dir, infos, arrays, policy=OLD_POLICY)
    (data_dir / "m01.csv").unlink()
    return outcomes[0]["measurement_id"]


#: One trigger per reason code: the setup, the reason substring, and
#: whether finding it spends an attempt.
TRIGGERS = {
    "no_valid_sampling_rate": (_no_rate, "sampling_rate not declared", False),
    "declaration_unresolvable": (_unresolvable, "does not resolve", False),
    "raw_format_incomplete": (_raw_incomplete, "declares no sample_format", False),
    "no_file_recorded": (_no_file, "no content hash recorded", False),
    "location_outside_data_dir": (
        _outside,
        "location escapes the data directory: ../outside/m01.csv",
        False,
    ),
    "file_missing": (_missing, "file not found at m01.csv", False),
    "content_differs": (_differs, "content differs at m01.csv", False),
    "file_inaccessible": (_inaccessible, "cannot read m01.csv", False),
    "undecodable": (_undecodable, "snapshot not computed:", True),
    "append_failed": (_append_failed, "snapshot computed but not appended:", True),
    "superseded_in_slot": (_superseded, "superseded in its acquisition slot by", False),
}


class TestReasonCodes:
    def test_the_vocabulary_is_the_plan_table(self):
        assert service.REPROCESS_REASON_CODES == REASON_CODE_TABLE
        assert set(TRIGGERS) == set(REASON_CODE_TABLE)
        for code in REASON_CODE_TABLE[:-1]:
            assert service.reprocess_action(code)
        assert service.reprocess_action("superseded_in_slot") is None

    @pytest.mark.parametrize("code", REASON_CODE_TABLE)
    def test_each_code_for_its_trigger(self, store, data_dir, monkeypatch, code):
        setup, substring, attempted = TRIGGERS[code]
        measurement_id = setup(store, data_dir, monkeypatch)

        def snapshots() -> int:
            return sum(
                p["measurement_id"] == measurement_id
                for p in payloads(store, EVENT_HEALTH_SNAPSHOT_COMPUTED)
            )

        before = snapshots()

        result = reprocess(store, data_dir)

        entry = only(result, measurement_id)
        assert entry["outcome"] == "not_reprocessable"
        assert entry["reason_code"] == code
        assert substring in entry["reason"]
        assert entry["attempted"] is attempted
        assert (entry["action"] is None) is (code == "superseded_in_slot")
        assert (entry["snapshot_id"] is not None) is (code == "append_failed")
        assert result["next_call"] is None
        assert snapshots() == before

    @pytest.mark.parametrize(
        ("code", "hint"),
        [
            ("no_valid_sampling_rate", "with a declared sampling rate"),
            ("raw_format_incomplete", "with the raw declaration"),
        ],
    )
    def test_view_reasons_keep_their_reload_hint(
        self, store, data_dir, monkeypatch, code, hint
    ):
        measurement_id = TRIGGERS[code][0](store, data_dir, monkeypatch)
        reason = only(reprocess(store, data_dir), measurement_id)["reason"]
        assert reason.endswith(
            f"; re-load the file {hint} (the new declaration supersedes this one)"
        )

    def test_a_directory_at_the_location_is_no_file(self, store, data_dir):
        measurement_id = one_stale(store, data_dir)
        path = data_dir / "m01.csv"
        path.unlink()
        path.mkdir()
        entry = only(reprocess(store, data_dir), measurement_id)
        assert entry["reason_code"] == "file_missing"
        assert entry["reason"] == "file not found at m01.csv"

    def test_no_location_recorded_at_all(self, store, data_dir):
        measurement_id = one_stale(store, data_dir)
        supersede(store, measurement_id, file={"location": None}, locations=[])
        entry = only(reprocess(store, data_dir), measurement_id)
        assert entry["reason_code"] == "no_file_recorded"
        assert entry["reason"] == "no file location recorded for the measurement"
        assert entry["location"] is None

    def test_a_data_directory_that_does_not_resolve(self, store, data_dir, monkeypatch):
        """Resolving the data directory fails (a link loop): each relative
        location reports it through its containment check, nothing raises."""
        measurement_id = one_stale(store, data_dir)
        real_resolve = Path.resolve

        def resolve(self, *args, **kwargs):
            if self == data_dir or data_dir in self.parents:
                raise RuntimeError(f"Symlink loop from {str(self)!r}")
            return real_resolve(self, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", resolve)
        entry = only(reprocess(store, data_dir), measurement_id)
        assert entry["reason_code"] == "file_inaccessible"
        assert entry["reason"].startswith("cannot read m01.csv: Symlink loop")

    def test_a_size_change_is_found_without_hashing(self, store, data_dir, spy):
        measurement_id = _differs(store, data_dir, None)
        spy["digest"].clear()
        entry = only(reprocess(store, data_dir), measurement_id)
        assert entry["reason_code"] == "content_differs"
        assert entry["reason"].startswith("content differs at m01.csv: ")
        assert entry["reason"].endswith(" were recorded")
        assert entry["attempted"] is False
        assert spy["digest"] == []

    @pytest.mark.parametrize("change", ["same_size", "other_size"])
    def test_s5_a_changed_file_is_the_users_decision(self, store, data_dir, change):
        """S5: the action asks the user to choose between restoring the
        original and loading the current file as a new recording, and
        suggests neither on its own."""
        measurement_id = one_stale(store, data_dir)
        path = data_dir / "m01.csv"
        if change == "same_size":
            same_size_change(path)
        else:
            write_signal(path, noise(99, 5001))
        entry = only(reprocess(store, data_dir), measurement_id)
        assert entry["reason_code"] == "content_differs"
        assert entry["action"].startswith(
            "a decision for the user, which the agent asks for and does not make: "
            "restore the original file at its recorded location, or load the "
            "current file with load_signal as a new recording"
        )
        assert entry["attempted"] is (change == "same_size")

    def test_an_equal_size_change_is_found_by_the_hash(self, store, data_dir, spy):
        measurement_id = one_stale(store, data_dir)
        same_size_change(data_dir / "m01.csv")
        result = reprocess(store, data_dir)
        entry = only(result, measurement_id)
        assert entry["reason_code"] == "content_differs"
        assert entry["reason"] == "content differs at m01.csv"
        assert entry["attempted"] is True
        assert spy["digest"] == ["m01.csv"]
        assert spy["decode"] == []

    def test_a_missing_recorded_size_falls_through_to_the_hash(self, store, data_dir):
        measurement_id = one_stale(store, data_dir)
        supersede(store, measurement_id, file={"size_bytes": None})
        entry = only(reprocess(store, data_dir), measurement_id)
        assert entry["outcome"] == "reprocessed"
        assert entry["location_used"] == "m01.csv"

    def two_locations(self, store, data_dir, tmp_path) -> tuple[str, Path, Path]:
        """m01 recorded in the data directory, then loaded again from a copy
        outside it (a supersession of the location): the copy is the most
        recent location."""
        declare_point(store)
        values = noise(1, 5000)
        original = data_dir / "m01.csv"
        info = make_info(original, values, acquired_at=acquired(1))
        [first] = record(store, data_dir, [info], {"m01": values}, policy=OLD_POLICY)
        copy_path = tmp_path / "scratch" / "m01.csv"
        copy_path.parent.mkdir()
        shutil.copy(original, copy_path)
        [second] = record(
            store,
            data_dir,
            [make_info(copy_path, values, acquired_at=acquired(1))],
            {"m01": values},
            policy=OLD_POLICY,
        )
        assert second["changed"] == ["location"]
        return first["measurement_id"], original, copy_path

    def test_the_most_recent_location_decides_the_code(self, store, data_dir, tmp_path):
        measurement_id, original, newest = self.two_locations(store, data_dir, tmp_path)
        newest.unlink()
        write_signal(original, noise(99, 5001))
        entry = only(reprocess(store, data_dir), measurement_id)
        assert entry["reason_code"] == "file_missing"
        assert entry["reason"].startswith(f"file not found at {newest}")
        assert "content differs at m01.csv" in entry["reason"]

        write_signal(newest, noise(98, 5001))
        original.unlink()
        entry = only(reprocess(store, data_dir), measurement_id)
        assert entry["reason_code"] == "content_differs"
        assert entry["reason"].startswith(f"content differs at {newest}")
        assert "file not found at m01.csv" in entry["reason"]

    def test_an_intact_older_copy_is_used_when_the_newest_changed(
        self, store, data_dir, tmp_path
    ):
        measurement_id, _, newest = self.two_locations(store, data_dir, tmp_path)
        same_size_change(newest)
        entry = only(reprocess(store, data_dir), measurement_id)
        assert entry["outcome"] == "reprocessed"
        assert entry["location_used"] == "m01.csv"
        assert entry["location"] == str(newest)

    def test_a_location_another_measurement_holds_is_not_this_file(
        self, store, data_dir
    ):
        """m01.csv was overwritten by another recording, which was loaded
        from it: the first measurement's file is gone, whatever its size."""
        declare_point(store)
        [first] = load_sequence(store, data_dir, 1)
        path = data_dir / "m01.csv"
        values = noise(50, 5000)
        write_signal(path, values)
        [second] = record(
            store,
            data_dir,
            [make_info(path, values, acquired_at=acquired(2))],
            {"m01": values},
            policy=OLD_POLICY,
        )
        result = reprocess(store, data_dir)
        entry = only(result, first)
        assert entry["reason_code"] == "file_missing"
        assert entry["reason"] == (
            "file not found at m01.csv: another recorded measurement is there now"
        )
        assert only(result, second["measurement_id"])["outcome"] == "reprocessed"

    def test_two_channels_of_one_file_both_reprocess(self, store, data_dir):
        """Two channels of one raw file share its location and content hash:
        neither takes the location from the other."""
        declare_point(store)
        frames = np.stack([noise(1, 5000), noise(2, 5000)], axis=1)
        path = data_dir / "two.bin"
        written = write_raw_file(path, frames.reshape(-1)).reshape(-1, 2)
        infos, arrays = [], {}
        for channel, sensor in ((0, "ACC01"), (1, "ACC02")):
            raw = {
                "sample_format": "float32",
                "byte_order": "little",
                "n_channels": 2,
                "channel_index": channel,
                "header_offset": 0,
                "scale_factor": None,
            }
            comp = companion(sensor_id=sensor, acquired_at=acquired(1))
            identity = build_measurement_identity(
                comp, "two_metadata.json", signal_path=path, channel_index=channel
            )
            values = written[:, channel].astype(np.float64)
            info = make_info(
                path, values, raw_format=raw, sensor_id=sensor, acquired_at=acquired(1)
            )
            info["measurement"] = identity
            info["signal_id"] = f"two_ch{channel}"
            infos.append(info)
            arrays[info["signal_id"]] = values
        outcomes = record(store, data_dir, infos, arrays, policy=OLD_POLICY)
        ids = [o["measurement_id"] for o in outcomes]
        assert len(set(ids)) == 2
        result = reprocess(store, data_dir)
        for measurement_id in ids:
            entry = only(result, measurement_id)
            assert entry["outcome"] == "reprocessed"
            assert entry["location_used"] == "two.bin"

    def test_view_checks_read_no_file(self, store, data_dir, monkeypatch, spy):
        """An infinite recorded rate (possible only in a ledger written before
        the rate was validated) and an unresolvable declaration are found
        from the ledger alone: no containment check, metadata call or hash."""
        declare_point(store)
        ids = load_sequence(store, data_dir, 2)
        supersede(store, ids[1], declaration={"direction": "diagonal"})
        view = store.read_view(ASSET)
        view["measurements"][ids[0]]["current"]["declaration"]["sampling_rate"] = float(
            "inf"
        )
        monkeypatch.setattr(
            LedgerStore, "read_view", lambda self, asset: copy.deepcopy(view)
        )
        result = reprocess(store, data_dir)
        assert [only(result, m)["reason_code"] for m in ids] == [
            "no_valid_sampling_rate",
            "declaration_unresolvable",
        ]
        assert "sampling_rate inf is not a positive finite number" in (
            only(result, ids[0])["reason"]
        )
        assert spy["resolve"] == spy["stat"] == spy["digest"] == []

    def test_ae12_a_permission_error_on_one_location(
        self, store, data_dir, monkeypatch
    ):
        declare_point(store)
        ids = load_sequence(store, data_dir, 3)
        refuse_stat_of(monkeypatch, "m02.csv", PermissionError(13, "Permission denied"))
        result = reprocess(store, data_dir)
        entry = only(result, ids[1])
        assert entry["reason_code"] == "file_inaccessible"
        assert (
            entry["reason"] == "cannot read m02.csv: PermissionError: Permission denied"
        )
        assert entry["action"].startswith("fix access to the file")
        assert [only(result, m)["outcome"] for m in (ids[0], ids[2])] == [
            "reprocessed",
            "reprocessed",
        ]

    def test_a_read_error_while_hashing(self, store, data_dir, monkeypatch):
        measurement_id = one_stale(store, data_dir)

        def unreadable(path: Path) -> Any:
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(service, "digest_file", unreadable)
        entry = only(reprocess(store, data_dir), measurement_id)
        assert entry["reason_code"] == "file_inaccessible"
        assert entry["attempted"] is True
        assert entry["reason"] == "cannot read m01.csv: OSError: Input/output error"

    def test_a_read_error_while_decoding(self, store, data_dir, monkeypatch):
        measurement_id = one_stale(store, data_dir)

        def unreadable(path: Path, declaration: Any) -> np.ndarray:
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(service, "_decode", unreadable)
        entry = only(reprocess(store, data_dir), measurement_id)
        assert entry["reason_code"] == "file_inaccessible"
        assert entry["reason"].startswith("snapshot not computed: PermissionError")

    def test_ae11_missing_files_of_baseline_members_restore_first(
        self, store, data_dir
    ):
        """S11: restoring keeps a member in the declared baseline; loading it
        from anywhere else removes it until the baseline is declared again.
        A non-member's missing file keeps the plain action."""
        declare_point(store)
        ids = load_sequence(store, data_dir, 12)
        members = ids[:3]
        declare_healthy_baseline(
            store=store,
            asset_id=ASSET,
            measurement_point_id=POINT,
            measurement_ids=members,
            declared_by="L. Rossi",
        )
        for index in (1, 2, 3, 12):
            (data_dir / f"m{index:02d}.csv").unlink()
        result = reprocess(store, data_dir)
        for measurement_id in members:
            action = only(result, measurement_id)["action"]
            assert "the only action that keeps it in the declared baseline" in action
            assert (
                "removes it from the baseline until the baseline is declared again"
                in action
            )
        plain = only(result, ids[11])["action"]
        assert "declared baseline" not in plain
        assert "load it with load_signal from where the user says it is now" in plain
        assert f"relative locations resolved under {data_dir.resolve()}" in plain

    def test_a_failing_duplicate_needs_no_action(self, store, data_dir):
        """AE18 on the service: the only stale leftover is a recording
        superseded in its acquisition slot, so the sequence is complete."""
        duplicate = _superseded(store, data_dir, None)
        result = reprocess(store, data_dir)
        assert only(result, duplicate)["reason_code"] == "superseded_in_slot"
        assert result["sequence_state"] == "complete"
        assert result["unresolved"] == 0
        assert result["superseded"] == 1
        assert result["groups"]["superseded"]["measurement_ids"] == [duplicate]
        assert result["needs_action"] == []
        assert result["next_call"] is None
        assert "the sequence is complete" in result["message"]
        assert "needs_action" not in result["message"]

    def test_the_declaration_names_the_call_only_when_one_can_act(
        self, store, data_dir
    ):
        """Stale measurements that fail the view checks, or are collapsed into
        another slot, are counted by reason; the remedy names the call only
        for one a call can act on."""
        values = noise(1)
        info = make_info(data_dir / "norate.csv", values, sampling_rate=None)
        assert record_one(store, data_dir, info, values)["snapshot_status"] == "skipped"
        declared = declare(store, 2)
        assert declared["measurements_with_stale_context"] == 1
        assert declared["remedy"] is None
        assert declared["message"].endswith(
            "; 1 recorded measurement(s) lack a snapshot computed with the current "
            "context and lineage; 1 needs an action before re-processing can "
            "compute it (1 no_valid_sampling_rate)."
        )
        load_sequence(store, data_dir, 2)
        redeclared = declare(store, 1)
        assert redeclared["measurements_with_stale_context"] == 3
        assert redeclared["remedy"] == REPROCESS_CALL
        assert f"; call {REPROCESS_CALL} to attempt the 2 that pass" in (
            redeclared["message"]
        )

    def test_an_already_recorded_snapshot_is_not_appended_again(
        self, store, data_dir, monkeypatch
    ):
        """A computed snapshot whose id the view already holds (written by
        another process under that id) counts as up to date."""
        measurement_id = one_stale(store, data_dir)
        recorded = next(
            iter(service._snapshot_ids(store.read_view(ASSET), measurement_id))
        )
        monkeypatch.setattr(service, "compute_snapshot_id", lambda *parts: recorded)
        before = len(events_of(store))
        result = reprocess(store, data_dir)
        entry = only(result, measurement_id)
        assert entry["outcome"] == "up_to_date"
        assert result["up_to_date"] == 1
        assert result["groups"]["up_to_date"]["measurement_ids"] == [measurement_id]
        assert len(events_of(store)) == before


# ---------------------------------------------------------------------------
# Sequences: continuation token, groups, state, bounds
# ---------------------------------------------------------------------------


def assert_partition(response: dict[str, Any], total: int) -> None:
    """The groups partition the point's measurements, with exact counts and
    capped, disjoint lists."""
    groups = response["groups"]
    assert set(groups) == {
        "reprocessed",
        "up_to_date",
        "not_reprocessable",
        "earlier_in_sequence",
        "superseded",
        "remaining",
    }
    assert sum(group["count"] for group in groups.values()) == total
    listed: list[str] = []
    for group in groups.values():
        assert len(group["measurement_ids"]) <= service.MAX_LISTED_ITEMS
        assert group["truncated"] is (group["count"] > service.MAX_LISTED_ITEMS)
        listed += group["measurement_ids"]
    assert len(listed) == len(set(listed))
    for key in ("reprocessed", "not_reprocessable", "remaining", "up_to_date"):
        assert groups[key]["count"] == response[key]
    assert groups["earlier_in_sequence"]["count"] == response["earlier_in_sequence"]
    assert groups["superseded"]["count"] == response["superseded"]
    assert response["unresolved"] == (
        response["not_reprocessable"] + response["earlier_in_sequence"]
    )
    assert response["sequence_state"] in service.SEQUENCE_STATES
    assert (response["next_call"] is None) is (response["remaining"] == 0)
    assert (response["sequence_state"] == "continue") is (response["remaining"] > 0)
    if response["remaining"] == 0:
        expected = "exhausted" if response["unresolved"] else "complete"
        assert response["sequence_state"] == expected
    if response["next_cursor"] is not None:
        assert len(response["next_cursor"]) < 160
        assert f"reprocess_cursor={response['next_cursor']!r}" in response["next_call"]
    assert "error" not in set(walk(response))
    json.dumps(response, allow_nan=False)


def s2_with_eligible(store: LedgerStore, data_dir: Path) -> list[str]:
    """S2 with 5 eligible measurements after the 15: twenty stale, the 15
    first in priority (the reference m01..m10 and the last K m20..m16)
    rewritten with other bytes of the same length."""
    declare_point(store)
    ids = load_sequence(store, data_dir, 20)
    for index in list(range(1, 11)) + list(range(16, 21)):
        same_size_change(data_dir / f"m{index:02d}.csv")
    return ids


def base30(store: LedgerStore, data_dir: Path) -> list[str]:
    """Thirty stale measurements: the reference (m01..m10) and the last K
    (m30..m26) files missing, the next ten in priority (m25..m16) rewritten
    with the same size, the last five (m15..m11) eligible. The first call
    ends after its tenth attempt, on m16."""
    declare_point(store)
    ids = load_sequence(store, data_dir, 30)
    for index in list(range(1, 11)) + list(range(26, 31)):
        (data_dir / f"m{index:02d}.csv").unlink()
    for index in range(16, 26):
        same_size_change(data_dir / f"m{index:02d}.csv")
    return ids


class TestReprocessSequences:
    def test_ae1_ae7_s1_ends_in_one_call_exhausted(self, store, data_dir):
        declare(store, 2)
        ids = load_sequence(store, data_dir, 20, policy=None)
        declare(store, 1)
        for index in range(1, 13):
            (data_dir / f"m{index:02d}.csv").unlink()
        params = AssessmentParams(reference_measurements=12)
        result = reprocess(store, data_dir, params=params)
        assert_partition(result, 20)
        assert result["reprocessed"] == 8
        assert result["not_reprocessable"] == 12
        assert result["remaining"] == 0
        assert result["unresolved"] == 12
        assert result["sequence_state"] == "exhausted"
        assert result["next_call"] is None and result["next_cursor"] is None
        failures = [e for e in result["results"] if e["outcome"] == "not_reprocessable"]
        assert [e["measurement_id"] for e in failures] == ids[:12]
        assert {e["reason_code"] for e in failures} == {"file_missing"}
        assert all(e["action"].startswith("check first") for e in failures)
        assert all(e["attempted"] is False for e in failures)
        [entry] = result["needs_action"]
        assert entry["reason_code"] == "file_missing"
        assert entry["count"] == 12
        assert entry["measurement_ids"] == ids[:12]
        assert result["message"].endswith(
            "; the sequence is exhausted: 12 measurement(s) stay stale until the "
            "action needs_action names is taken (12 file_missing: "
            + ", ".join(ids[:5])
            + " and more), and no further call can progress"
        )
        assert "assess_asset_change(" not in result["message"]

    def test_ae2_ae10_s2_two_calls_list_every_failure(self, store, data_dir):
        declare_point(store)
        ids = load_sequence(store, data_dir, 15)
        for index in range(1, 16):
            same_size_change(data_dir / f"m{index:02d}.csv")
        events = len(events_of(store))

        first = reprocess(store, data_dir)
        assert_partition(first, 15)
        assert first["not_reprocessable"] == 10 and first["remaining"] == 5
        assert first["sequence_state"] == "continue"
        token = first["next_cursor"]
        assert token is not None
        assert first["next_call"] == (
            "assess_asset_change(asset_id='P-101', measurement_point_id='motor_de_h', "
            f"reprocess=True, reprocess_cursor={token!r})"
        )

        second = reprocess(store, data_dir, cursor=token)
        assert_partition(second, 15)
        assert second["cursor"] == token
        assert second["restarted"] is None
        assert second["not_reprocessable"] == 5
        assert second["earlier_in_sequence"] == 10
        assert second["sequence_state"] == "exhausted"
        assert second["next_call"] is None
        detailed = second["earlier_failures"]
        assert sorted(e["measurement_id"] for e in detailed) == sorted(
            first["groups"]["not_reprocessable"]["measurement_ids"]
        )
        for entry in detailed:
            assert entry["reason_code"] == "content_differs"
            assert entry["action"].startswith("a decision for the user")
            assert entry["location"] == only(first, entry["measurement_id"])["location"]
        assert second["earlier_by_reason"] == {"content_differs": 10}
        assert second["earlier_not_detailed"] == 0
        [needs] = second["needs_action"]
        assert needs["count"] == 15 and len(needs["measurement_ids"]) == 15
        reported = {e["measurement_id"] for r in (first, second) for e in r["results"]}
        assert reported == set(ids)
        assert len(events_of(store)) == events

    def test_ae3_each_call_handles_new_measurements(self, store, data_dir):
        declare_point(store)
        load_sequence(store, data_dir, 3)
        same_size_change(data_dir / "m01.csv")
        responses = follow(store, data_dir, cap=call_cap(2, limit=1), limit=1)
        handled = [{e["measurement_id"] for e in r["results"]} for r in responses]
        assert len(responses) == 3
        assert all(handled)
        assert all(not (a & b) for a, b in zip(handled, handled[1:]))
        for response in responses:
            assert_partition(response, 3)

    def test_ae6_s6_carries_a_token_past_the_failures(self, store, data_dir):
        declare_point(store)
        load_sequence(store, data_dir, 15)
        for index in (3, 7):
            (data_dir / f"m{index:02d}.csv").unlink()
        first = reprocess(store, data_dir)
        assert first["reprocessed"] == 10 and first["remaining"] == 3
        assert first["next_cursor"] is not None
        second = reprocess(store, data_dir, cursor=first["next_cursor"])
        assert second["restarted"] is None
        assert second["reprocessed"] == 3
        assert second["sequence_state"] == "exhausted"

    def test_ae8_a_redeclaration_between_calls_restarts(self, store, data_dir):
        ids = s2_with_eligible(store, data_dir)
        first = reprocess(store, data_dir)
        declare(store, 1)
        second = reprocess(store, data_dir, cursor=first["next_cursor"])
        assert second["restarted"] == {
            "reason_code": "ledger_changed",
            "reason": service._RESTART_REASONS["ledger_changed"],
        }
        assert second["message"].startswith(
            "The continuation token was not used (it no longer matches the ledger"
        )
        assert second["earlier_in_sequence"] == 0
        assert second["earlier_failures"] == []
        assert second["results"][0]["measurement_id"] == ids[0]
        assert_partition(second, 20)

    def test_ae13_a_restored_declaration(self, store, data_dir):
        """A, B, A with every A snapshot recorded: nothing stale, nothing
        appended, complete. A revert declared while a sequence under B is
        in progress restarts it."""
        declare(store, 2)
        load_sequence(store, data_dir, 11, policy=None)
        declare(store, 1)
        assert reprocess(store, data_dir, limit=11)["reprocessed"] == 11
        declare(store, 2)
        before = len(events_of(store))
        result = reprocess(store, data_dir)
        assert (result["stale"], result["sequence_state"]) == (0, "complete")
        assert result["next_call"] is None
        assert len(events_of(store)) == before
        assert REPROCESS_CALL not in assess_point(store)["message"]

        declare(store, 1, "flexible")  # a context with no snapshot yet
        (data_dir / "m01.csv").unlink()
        first = reprocess(store, data_dir, limit=1)
        assert first["next_cursor"] is not None
        declare(store, 2)
        second = reprocess(store, data_dir, cursor=first["next_cursor"])
        assert second["restarted"]["reason_code"] == "ledger_changed"

    def test_ae14_an_interrupted_sequence(self, store, data_dir, root):
        declare_point(store)
        load_sequence(store, data_dir, 15)
        for index in range(1, 16):
            same_size_change(data_dir / f"m{index:02d}.csv")
        first = reprocess(store, data_dir)
        events = len(events_of(store))
        again = follow(store, data_dir, cap=call_cap(0))
        assert [r["cursor"] for r in again] == [None, again[0]["next_cursor"]]
        assert again[0]["results"] == first["results"]
        assert again[-1]["sequence_state"] == "exhausted"
        assert len(events_of(store)) == events
        restarted_server = reprocess_stale_snapshots(
            ASSET,
            POINT,
            store=LedgerStore(root),
            data_dir=data_dir,
            provenance_overrides=PROVENANCE,
            cursor=first["next_cursor"],
        )
        assert restarted_server["restarted"] is None
        assert restarted_server["earlier_in_sequence"] == 10
        assert restarted_server["sequence_state"] == "exhausted"

    def test_identical_calls_on_an_unchanged_ledger(self, store, data_dir):
        """R11: the exhausted ledger and the all-failures ledger answer two
        identical calls identically, and append nothing."""
        declare_point(store)
        load_sequence(store, data_dir, 15)
        for index in range(1, 16):
            same_size_change(data_dir / f"m{index:02d}.csv")
        events = len(events_of(store))
        first, second = reprocess(store, data_dir), reprocess(store, data_dir)
        assert store_module.canonical_json(first) == store_module.canonical_json(second)
        token = first["next_cursor"]
        end, again = (reprocess(store, data_dir, cursor=token) for _ in range(2))
        assert end["sequence_state"] == "exhausted"
        assert store_module.canonical_json(end) == store_module.canonical_json(again)
        assert len(events_of(store)) == events

    @pytest.mark.parametrize("scenario", ["s1", "s2"])
    def test_nothing_is_attempted_twice_in_a_sequence(
        self, store, data_dir, spy, scenario
    ):
        if scenario == "s1":
            declare_point(store)
            load_sequence(store, data_dir, 20)
            for index in range(1, 13):
                (data_dir / f"m{index:02d}.csv").unlink()
            eligible = 8
        else:
            s2_with_eligible(store, data_dir)
            eligible = 5
        for calls in spy.values():
            calls.clear()
        responses = follow(store, data_dir, cap=call_cap(eligible) + 1)
        assert responses[-1]["sequence_state"] == "exhausted"
        for key in ("digest", "decode"):
            assert len(spy[key]) == len(set(spy[key])), key
        assert len(spy["compute"]) == eligible

        # One restart (a call without the token) repeats each at most once.
        for calls in spy.values():
            calls.clear()
        follow(store, data_dir, cap=call_cap(eligible) + 1)
        follow(store, data_dir, cap=call_cap(eligible) + 1)
        for key in ("digest", "decode"):
            assert all(spy[key].count(name) <= 2 for name in spy[key]), key

    def test_the_file_check_bound(self, store, data_dir, monkeypatch, spy):
        """Each measurement's two relative locations cost four file checks:
        a call stops before a measurement that would exceed the bound, but
        its first measurement always runs in full."""
        declare_point(store)
        values = {i: noise(i, 5000) for i in (1, 2, 3)}
        for index, array in values.items():
            name = f"m{index:02d}.csv"
            for location in (data_dir / name, data_dir / "copy" / name):
                info = make_info(location, array, acquired_at=acquired(index))
                record(store, data_dir, [info], {name[:-4]: array}, policy=OLD_POLICY)
                location.unlink()
        monkeypatch.setattr(service, "MAX_FILE_CHECKS_PER_CALL", 5)
        first = reprocess(store, data_dir)
        assert [e["measurement_id"] for e in first["results"]] == [
            first["results"][0]["measurement_id"]
        ]
        assert len(spy["resolve"]) + len(spy["stat"]) == 4
        assert first["remaining"] == 2

        monkeypatch.setattr(service, "MAX_FILE_CHECKS_PER_CALL", 3)
        spy["resolve"].clear()
        spy["stat"].clear()
        second = reprocess(store, data_dir, cursor=first["next_cursor"])
        assert len(second["results"]) == 1
        assert len(spy["resolve"]) + len(spy["stat"]) == 4
        assert second["remaining"] == 1

    def test_ae18_only_superseded_leftovers_after_a_sequence(self, store, data_dir):
        duplicate = _superseded(store, data_dir, None)
        first = reprocess(store, data_dir, limit=1)
        assert first["sequence_state"] == "continue"
        assert first["next_cursor"] is None  # nothing failed yet
        second = reprocess(store, data_dir, limit=1)
        assert second["sequence_state"] == "complete"
        assert second["groups"]["superseded"]["measurement_ids"] == [duplicate]
        assert second["needs_action"] == []

    def test_a_malformed_token_is_refused_before_the_ledger_is_read(
        self, store, data_dir, monkeypatch
    ):
        def unread(self, asset_id):
            raise AssertionError("the ledger was read")

        monkeypatch.setattr(LedgerStore, "read_view", unread)
        valid = "r1." + "0" * 16 + "." + "0" * 8 + ".f.f1"
        for token, problem in (
            (valid[:-2], "does not have the shape"),
            (valid.replace(".f.", ".F."), "does not have the shape"),
            ("r2" + valid[2:], "does not have the shape"),
            (valid + "x" * 200, "more than the 200"),
            (valid.replace(".f.f1", ".ff.f1"), "its codes disagree with its counts"),
            (valid.replace(".f.f1", ".f.g1f1"), "its counts are not in code order"),
            (12345, "must be a string"),
        ):
            with pytest.raises(ValueError, match=problem) as refused:
                reprocess(store, data_dir, cursor=token)
            assert "pass the token exactly as next_call gives it" in str(refused.value)

    def test_the_token_stays_under_its_cap(self):
        counts = {code: 999_999 for code in REASON_CODE_TABLE}
        codes = [REASON_CODE_TABLE[i % 11] for i in range(50)]
        token = service._cursor_token("0123456789abcdef", "89abcdef", codes, counts)
        assert len(token) < 160
        parsed = service.parse_reprocess_cursor(token)
        assert parsed.codes == tuple(codes) and parsed.counts == counts

    def test_ae15_s9_thousands_of_missing_files(self, store, data_dir, spy):
        """S9: 2,000 stale measurements whose files are gone, each with six
        recorded locations, sit ahead of five eligible ones. Every call stays
        within both bounds and the 50-failure stop, lists each failure in
        full, and reaches measurements no earlier call reached."""
        declare_point(store)
        eligible = load_sequence(store, data_dir, 15)[10:]
        for index in range(1, 11):
            (data_dir / f"m{index:02d}.csv").unlink()
        template = store.read_view(ASSET)["measurements"][eligible[0]]["current"]
        lines: list[bytes] = []
        for index in range(16, 16 + 1990):
            payload = copy.deepcopy(template)
            payload["measurement_id"] = f"{index:016x}"
            payload["declaration"]["acquired_at"] = acquired(index)
            names = [f"gone/{index}_{k}.csv" for k in range(6)]
            payload["file"].update(
                location=names[-1], content_sha256=f"{index:064x}", size_bytes=10
            )
            payload["locations"] = names
            event = make_event(EVENT_MEASUREMENT_RECORDED, ASSET, payload)
            lines.append(store._encode_event(ASSET, event))
        # One write instead of 1,990 locked and synced appends.
        ledger = Path(store.root) / f"{ASSET}{store_module.LEDGER_SUFFIX}"
        with ledger.open("ab") as handle:
            handle.write(b"".join(lines))
        assert len(store.read_view(ASSET)["measurements"]) == 2005
        for calls in spy.values():
            calls.clear()

        responses: list[dict[str, Any]] = []
        cursor: Optional[str] = None
        reached: set[str] = set()
        while True:
            checks = len(spy["resolve"]) + len(spy["stat"])
            response = reprocess(store, data_dir, cursor=cursor)
            responses.append(response)
            spent = len(spy["resolve"]) + len(spy["stat"]) - checks
            assert spent <= service.MAX_FILE_CHECKS_PER_CALL
            assert sum(e["attempted"] for e in response["results"]) <= 10
            found = [
                e for e in response["results"] if e["outcome"] == "not_reprocessable"
            ]
            assert len(found) <= service.MAX_LISTED_ITEMS
            assert all(
                e["reason_code"] and e["action"] and e["location"] for e in found
            )
            handled = {e["measurement_id"] for e in response["results"]}
            assert handled and not (handled & reached)
            reached |= handled
            assert_partition(response, 2005)
            if response["next_call"] is None:
                break
            assert len(responses) < 60
            cursor = response["next_cursor"]
        assert stale_ids(store) == reached - set(eligible)
        assert responses[-1]["reprocessed"] == 5
        assert responses[-1]["earlier_not_detailed"] > 0
        assert responses[-1]["sequence_state"] == "exhausted"


class TestTokenValidity:
    """One test per row of the plan's token validity table."""

    def first_call(self, store, data_dir, ids_builder=s2_with_eligible):
        ids = ids_builder(store, data_dir)
        first = reprocess(store, data_dir)
        assert first["next_cursor"] is not None
        return ids, first

    def again(self, store, data_dir, first, **kwargs):
        return reprocess(store, data_dir, cursor=first["next_cursor"], **kwargs)

    def test_own_appends_continue(self, store, data_dir):
        _, first = self.first_call(store, data_dir)
        second = self.again(store, data_dir, first)
        assert second["restarted"] is None and second["earlier_in_sequence"] == 10

    def test_another_agent_after_the_last_continues(self, store, data_dir):
        ids, first = self.first_call(store, data_dir)
        reprocess(store, data_dir, limit=20)  # another agent
        second = self.again(store, data_dir, first)
        assert second["restarted"] is None
        assert set(ids[10:15]) <= set(second["groups"]["up_to_date"]["measurement_ids"])
        assert second["sequence_state"] == "exhausted"

    def test_another_agent_on_an_earlier_failure_restarts(self, store, data_dir):
        _, first = self.first_call(store, data_dir)
        same_size_change(data_dir / "m01.csv")  # the original restored
        assert reprocess(store, data_dir, limit=1)["reprocessed"] == 1
        assert self.again(store, data_dir, first)["restarted"] is not None

    def test_a_load_with_its_snapshot_continues(self, store, data_dir):
        ids, first = self.first_call(store, data_dir, base30)
        values = noise(31, 5000)
        info = make_info(data_dir / "m31.csv", values, acquired_at=acquired(31))
        record(store, data_dir, [info], {"m31": values})
        second = self.again(store, data_dir, first)
        assert second["restarted"] is None
        assert second["earlier_in_sequence"] == 25

    def test_a_backfill_shifting_the_reference_restarts(self, store, data_dir):
        _, first = self.first_call(store, data_dir)
        values = noise(40, 5000)
        early = (FIRST_ACQUIRED - timedelta(weeks=1)).isoformat()
        record(
            store,
            data_dir,
            [make_info(data_dir / "m00.csv", values, acquired_at=early)],
            {"m00": values},
        )
        assert self.again(store, data_dir, first)["restarted"] is not None

    def test_a_failing_load_before_the_last_restarts(self, store, data_dir):
        _, first = self.first_call(store, data_dir, base30)
        values = noise(31, 5000)
        info = make_info(
            data_dir / "m31.csv", values, acquired_at=acquired(31), sampling_rate=None
        )
        record(store, data_dir, [info], {"m31": values})
        assert self.again(store, data_dir, first)["restarted"] is not None

    def test_a_failing_load_after_the_last_continues(self, store, data_dir):
        _, first = self.first_call(store, data_dir, base30)
        values = noise(32, 5000)
        info = make_info(
            data_dir / "m32.csv",
            values,
            acquired_at=acquired(32),
            sampling_rate=None,
            direction="vertical",
        )
        [loaded] = record(store, data_dir, [info], {"m32": values})
        second = self.again(store, data_dir, first)
        assert second["restarted"] is None
        assert only(second, loaded["measurement_id"])["reason_code"] == (
            "no_valid_sampling_rate"
        )

    def test_a_superseded_earlier_failure_restarts(self, store, data_dir):
        ids, first = self.first_call(store, data_dir)
        supersede(store, ids[0], declaration={"rpm": 1500.0})
        assert self.again(store, data_dir, first)["restarted"] is not None

    def test_a_baseline_moving_a_stale_measurement_restarts(self, store, data_dir):
        ids, first = self.first_call(store, data_dir, base30)
        declare_healthy_baseline(
            store=store,
            asset_id=ASSET,
            measurement_point_id=POINT,
            measurement_ids=ids[10:13],
            declared_by="L. Rossi",
        )
        assert self.again(store, data_dir, first)["restarted"] is not None

    def test_a_baseline_moving_nothing_continues(self, store, data_dir):
        ids, first = self.first_call(store, data_dir)
        declare_healthy_baseline(
            store=store,
            asset_id=ASSET,
            measurement_point_id=POINT,
            measurement_ids=ids[:10],
            declared_by="L. Rossi",
        )
        assert self.again(store, data_dir, first)["restarted"] is None

    def test_other_reference_parameters_restart(self, store, data_dir):
        _, first = self.first_call(store, data_dir)
        for params in (
            AssessmentParams(last_k=6),
            AssessmentParams(reference_measurements=11),
        ):
            assert self.again(store, data_dir, first, params=params)["restarted"]

    def test_a_new_lineage_restarts(self, store, data_dir):
        _, first = self.first_call(store, data_dir)
        policy = SnapshotPolicy(tolerance_pct=3.0)
        assert self.again(store, data_dir, first, policy=policy)["restarted"]

    def test_the_last_measurement_moved_to_another_point_restarts(
        self, store, data_dir
    ):
        _, first = self.first_call(store, data_dir)
        last = first["results"][-1]["measurement_id"]
        supersede(store, last, measurement_point_id="motor_nde_v")
        second = self.again(store, data_dir, first)
        assert second["restarted"]["reason_code"] == "unknown_measurement"

    def test_a_restored_file_of_an_earlier_failure_continues(self, store, data_dir):
        ids, first = self.first_call(store, data_dir)
        same_size_change(data_dir / "m01.csv")
        second = self.again(store, data_dir, first)
        assert second["restarted"] is None
        assert ids[0] in second["groups"]["earlier_in_sequence"]["measurement_ids"]

    def test_a_server_restart_continues(self, store, data_dir, root):
        _, first = self.first_call(store, data_dir)
        second = reprocess_stale_snapshots(
            ASSET,
            POINT,
            store=LedgerStore(root),
            data_dir=data_dir,
            provenance_overrides=PROVENANCE,
            cursor=first["next_cursor"],
        )
        assert second["restarted"] is None

    def test_another_limit_continues(self, store, data_dir):
        _, first = self.first_call(store, data_dir)
        second = self.again(store, data_dir, first, limit=3)
        assert second["restarted"] is None and second["remaining"] == 7

    def test_a_crafted_token_that_matches_the_digest_restarts(
        self, store, data_dir, monkeypatch
    ):
        """The digest detects corruption, it is no security boundary: a token
        made to match it, whose codes and counts disagree with the ledger,
        restarts the sequence instead of raising."""
        _, first = self.first_call(store, data_dir)
        parsed = service.parse_reprocess_cursor(first["next_cursor"])
        monkeypatch.setattr(service, "content_hash", lambda payload: parsed.digest * 8)
        crafted = service._cursor_token(
            parsed.last, parsed.digest, ["file_missing"], {"file_missing": 1}
        )
        second = reprocess(store, data_dir, cursor=crafted)
        assert second["restarted"]["reason_code"] == "ledger_changed"

    def test_a_token_of_another_point_or_asset_restarts(self, store, data_dir):
        _, first = self.first_call(store, data_dir)
        for asset, point in ((ASSET, "motor_nde_v"), (OTHER_ASSET, POINT)):
            declare_point(store, asset=asset, measurement_point_id=point)
            other = reprocess_stale_snapshots(
                asset,
                point,
                store=store,
                data_dir=data_dir,
                cursor=first["next_cursor"],
            )
            assert other["restarted"]["reason_code"] == "unknown_measurement"


class TestManyFailures:
    """S14: more failures than a listing holds."""

    def test_ae17_every_failure_is_listed_by_the_call_that_finds_it(
        self, store, data_dir, monkeypatch
    ):
        declare_point(store)
        duplicates = {f"d{i}": noise(500 + i, 5000) for i in (1, 2, 3)}
        slots = {"d1": 120, "d2": 121, "d3": 122}
        record(
            store,
            data_dir,
            [
                make_info(data_dir / f"{n}.csv", v, acquired_at=acquired(slots[n] + 1))
                for n, v in duplicates.items()
            ],
            duplicates,
            policy=OLD_POLICY,
        )
        ids = load_sequence(store, data_dir, 125)
        order = service._reprocess_plan(
            store.read_view(ASSET), POINT, AssessmentParams()
        ).order
        name = {m: f"m{ids.index(m) + 1:02d}.csv" for m in ids}
        undecodable: set[str] = set()
        plan = (
            ["file_missing"] * 45
            + ["content_differs"] * 5
            + ["file_missing"] * 45
            + ["undecodable"] * 5
            + ["file_missing"] * 16
            + ["content_differs"] * 2
            + ["undecodable"] * 2
        )
        failing = order[:120]
        for measurement_id, code in zip(failing, plan):
            path = data_dir / name[measurement_id]
            if code == "file_missing":
                path.unlink()
            elif code == "content_differs":
                same_size_change(path)
            else:
                undecodable.add(path.name)
        for n in duplicates:
            (data_dir / f"{n}.csv").unlink()
        real_decode = service._decode

        def decode(path: Path, declaration: Any) -> np.ndarray:
            if path.name in undecodable:
                raise ValueError(f"Recorded file '{path.name}' no longer decodes")
            return real_decode(path, declaration)

        monkeypatch.setattr(service, "_decode", decode)
        events = len(events_of(store))

        responses = follow(store, data_dir, cap=4)
        assert len(responses) == 3
        found = [
            [e for e in r["results"] if e["outcome"] == "not_reprocessable"]
            for r in responses
        ]
        assert [len(f) for f in found] == [50, 50, 23]
        for entries in found:
            for entry in entries:
                assert entry["reason_code"] and entry["location"]
                assert entry["action"] or entry["reason_code"] == "superseded_in_slot"
        final = responses[-1]
        assert_partition(final, 128)
        assert final["reprocessed"] == 5
        assert final["not_reprocessable"] == 20
        assert final["superseded"] == 3
        assert final["earlier_in_sequence"] == 100
        assert len(final["earlier_failures"]) == 50
        assert final["earlier_not_detailed"] == 50
        assert final["earlier_by_reason"] == {
            "file_missing": 90,
            "content_differs": 5,
            "undecodable": 5,
        }
        assert {e["reason_code"]: e["count"] for e in final["needs_action"]} == {
            "file_missing": 106,
            "content_differs": 7,
            "undecodable": 7,
        }
        assert sum(len(e["measurement_ids"]) for e in final["needs_action"]) == 50
        assert "50 earlier failure(s) are not detailed in this response" in (
            final["message"]
        )
        assert all(len(r["next_cursor"] or "") < 160 for r in responses)
        assert len(events_of(store)) == events + 5

        # A new sequence without the earlier responses lists each again.
        again = follow(store, data_dir, cap=4)
        listed = {
            e["measurement_id"]
            for r in again
            for e in r["results"]
            if e["outcome"] == "not_reprocessable"
            and e["reason_code"] != "superseded_in_slot"
        }
        assert listed == set(failing)
        assert len(events_of(store)) == events + 5
