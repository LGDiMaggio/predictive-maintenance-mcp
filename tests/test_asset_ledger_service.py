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
- the outcome contract (keys, no ``error`` key anywhere).
"""

import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from _golden_signals import FS, golden_signals
from predictive_maintenance_mcp.asset_ledger import service
from predictive_maintenance_mcp.asset_ledger import store as store_module
from predictive_maintenance_mcp.asset_ledger.service import (
    DECLARATION_KEYS,
    LEDGER_STATUSES,
    LOAD_OUTCOME_KEYS,
    OUTCOME_KEYS,
    SNAPSHOT_PAYLOAD_KEYS,
    SNAPSHOT_STATUSES,
    build_declaration,
    changed_keys,
    declaration_fingerprint,
    file_block,
    record_measurements,
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
