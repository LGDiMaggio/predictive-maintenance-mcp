"""Tests for ``asset_ledger.store``: the append-only per-asset ledger.

Every property claimed about bytes on disk (delimiters, torn tails, locks,
offsets) is verified by executing against real files under ``tmp_path``,
with real child interpreters where a second process is the question. No
filesystem mocks; the only monkeypatch is a spy on the store's single read
primitive, used to prove that a versioned append reads only the delta.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from conftest import REPO_ROOT, SUBPROCESS_PIN

from predictive_maintenance_mcp.asset_ledger import store as store_module
from predictive_maintenance_mcp.asset_ledger.store import (
    ENVELOPE_KEYS,
    EVENT_BASELINE_DECLARED,
    EVENT_HEALTH_SNAPSHOT_COMPUTED,
    EVENT_MEASUREMENT_POINT_DECLARED,
    EVENT_MEASUREMENT_RECORDED,
    EVENT_TYPES,
    MAX_INTEGRITY_ISSUES,
    MEASUREMENT_INDEX_NAME,
    PRODUCER_NAME,
    SCHEMA_VERSION,
    AppendResult,
    LedgerError,
    LedgerLockTimeout,
    LedgerReadResult,
    LedgerStore,
    LedgerWriteError,
    build_asset_view,
    canonical_json,
    compute_event_id,
    content_hash,
    make_event,
    short_id,
)

ASSET = "P-101"
POINT = "de_h"
UTC = timezone.utc

#: Generous: the children import the whole package, and a cold OneDrive
#: checkout has been measured at over a minute for that alone.
CHILD_TIMEOUT = 300.0


# ---------------------------------------------------------------------------
# Event builders (payloads follow the module's key contracts)
# ---------------------------------------------------------------------------


def point_event(version: int, *, asset: str = ASSET, note=None, recorded_at=None):
    payload = {
        "measurement_point_id": POINT,
        "declaration_version": version,
        "bearing_id": "6205",
        "fault_orders": None,
        "machine_group": 2,
        "support_type": "rigid",
        "machine_power_kw": 22.0,
        "expected_signal_unit": "g",
        "expected_sensor_id": None,
        "expected_direction": "horizontal",
        "nominal_rpm": 1500.0,
        "declared_by": "test",
        "note": note,
        "changed": [] if version == 1 else ["note"],
    }
    return make_event(
        EVENT_MEASUREMENT_POINT_DECLARED, asset, payload, recorded_at=recorded_at
    )


def measurement_event(
    measurement_id: str,
    acquired_at: str,
    *,
    asset: str = ASSET,
    declared_asset=None,
    version: int = 1,
    changed=(),
    rpm=None,
):
    payload = {
        "measurement_id": measurement_id,
        "measurement_point_id": POINT,
        "declaration_version": version,
        "declaration": {
            "asset_id": declared_asset or asset,
            "measurement_point_id": POINT,
            "acquired_at": acquired_at,
            "timezone_declared": True,
            "timestamp_suspect": False,
            "rpm": rpm,
            "load": None,
            "operating_state": None,
            "sensor_id": None,
            "direction": "horizontal",
            "declared_by": None,
            "sampling_rate": 10000.0,
            "signal_unit": "g",
            "raw_format": None,
            "channel_index": 0,
        },
        "file": {
            "location": f"{measurement_id}.csv",
            "location_is_relative": True,
            "content_sha256": "0" * 64,
            "size_bytes": 10,
        },
        "signal_id": f"sig_{measurement_id}",
        "changed": list(changed),
        "locations": [f"{measurement_id}.csv"],
    }
    return make_event(EVENT_MEASUREMENT_RECORDED, asset, payload)


def snapshot_event(measurement_id: str, processing_id: str, *, asset: str = ASSET):
    payload = {
        "snapshot_id": short_id(measurement_id, processing_id, "ctx"),
        "measurement_id": measurement_id,
        "measurement_point_id": POINT,
        "processing": {
            "processing_id": processing_id,
            "algorithm_version": 1,
            "params": {"tolerance_pct": 2.0},
            "effective": {"fs": 10000.0},
            "provenance": {"numpy": "x"},
        },
        "context_digest": "ctx",
        "point_declaration_version": 1,
        "indicators": {"rms": 0.5},
        "one_x": None,
        "bearing": None,
        "iso": None,
        "missing": {},
    }
    return make_event(EVENT_HEALTH_SNAPSHOT_COMPUTED, asset, payload)


def baseline_event(measurement_ids, *, asset: str = ASSET, declared_at: str):
    payload = {
        "baseline_id": short_id(POINT, *sorted(measurement_ids), declared_at),
        "measurement_point_id": POINT,
        "measurement_ids": list(measurement_ids),
        "members": [
            {
                "measurement_id": mid,
                "declaration_version": 1,
                "point_declaration_version": 1,
            }
            for mid in measurement_ids
        ],
        "declared_by": "L. Rossi",
        "note": None,
        "declared_at": declared_at,
    }
    return make_event(EVENT_BASELINE_DECLARED, asset, payload)


def line_of(event) -> bytes:
    return canonical_json(event).encode("ascii") + b"\n"


def write_raw(path: Path, data: bytes) -> None:
    """Append bytes directly, bypassing the store (to plant corruption)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as fh:
        fh.write(data)


def retagged(event, **changes):
    """A hand-edited envelope with its id recomputed (foreign records)."""
    edited = dict(event)
    edited.update(changes)
    edited["event_id"] = compute_event_id(edited)
    return edited


def versions_in(result: LedgerReadResult) -> list:
    return [e["payload"]["declaration_version"] for e in result.events]


def view_without_bookkeeping(view: dict) -> dict:
    return {k: v for k, v in view.items() if k not in ("integrity", "end_offset")}


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "ledger"


@pytest.fixture
def store(root) -> LedgerStore:
    return LedgerStore(root)


# ---------------------------------------------------------------------------
# Canonical JSON and ids
# ---------------------------------------------------------------------------


class TestCanonicalIds:
    def test_canonical_json_is_compact_sorted_ascii(self):
        text = canonical_json({"b": [1, 2.5, None, True], "a": "\u2028\ré"})
        assert text == '{"a":"\\u2028\\r\\u00e9","b":[1,2.5,null,true]}'
        assert text.isascii()

    def test_canonical_json_refuses_nan(self):
        with pytest.raises(ValueError):
            canonical_json({"x": float("nan")})

    def test_foreign_types_are_refused_as_value_error_naming_the_type(self):
        # json reports these as TypeError; the snapshot unit hands the store
        # numpy results, so the refusal must be the package's ValueError and
        # must say which type to convert.
        import numpy as np

        for value in (np.float32(1.0), np.int64(1), {1, 2}, datetime(2026, 1, 1)):
            with pytest.raises(ValueError, match=type(value).__name__):
                canonical_json({"x": value})
        with pytest.raises(ValueError, match="float32"):
            make_event(EVENT_MEASUREMENT_POINT_DECLARED, ASSET, {"rms": np.float32(1)})
        assert canonical_json({"x": np.float64(0.5)}) == '{"x":0.5}'  # a float subclass

    def test_content_hash_is_sha256_of_canonical_json(self):
        obj = {"z": 1, "a": [3, 2]}
        expected = hashlib.sha256(canonical_json(obj).encode()).hexdigest()
        assert content_hash(obj) == expected
        assert content_hash({"a": [3, 2], "z": 1}) == expected  # key order irrelevant

    def test_short_id_is_first_16_hex_of_joined_parts(self):
        expected = hashlib.sha256(b"a:b:c").hexdigest()[:16]
        assert short_id("a", "b", "c") == expected
        with pytest.raises(ValueError):
            short_id()
        with pytest.raises(ValueError):
            short_id("a", 1)  # type: ignore[arg-type]

    def test_event_id_ignores_recorded_at_and_tracks_payload(self):
        first = point_event(
            1, note="alpha", recorded_at=datetime(2026, 1, 1, tzinfo=UTC)
        )
        retry = point_event(
            1, note="alpha", recorded_at=datetime(2026, 6, 1, tzinfo=UTC)
        )
        other = point_event(1, note="alphb")
        assert first["event_id"] == retry["event_id"]
        assert first["recorded_at"] != retry["recorded_at"]
        assert first["event_id"] != other["event_id"]
        assert len(first["event_id"]) == 64
        assert compute_event_id(first) == first["event_id"]

    def test_envelope_shape(self):
        event = point_event(1)
        assert tuple(event) == ENVELOPE_KEYS
        assert event["schema_version"] == SCHEMA_VERSION == 1
        assert event["producer"]["name"] == PRODUCER_NAME
        from predictive_maintenance_mcp import __version__

        assert event["producer"]["version"] == __version__
        assert event["recorded_at"].endswith("+00:00")
        datetime.fromisoformat(event["recorded_at"])

    def test_make_event_normalizes_payload_through_canonical_json(self):
        event = make_event(EVENT_BASELINE_DECLARED, ASSET, {"measurement_ids": ("a",)})
        assert event["payload"] == {"measurement_ids": ["a"]}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"event_type": "sensor_replaced", "asset_id": ASSET, "payload": {}},
            {"event_type": EVENT_TYPES[0], "asset_id": "../x", "payload": {}},
            {"event_type": EVENT_TYPES[0], "asset_id": ASSET, "payload": [1]},
            {"event_type": EVENT_TYPES[0], "asset_id": ASSET, "payload": {"x": 1e999}},
            {
                "event_type": EVENT_TYPES[0],
                "asset_id": ASSET,
                "payload": {},
                "recorded_at": datetime(2026, 1, 1),  # naive
            },
        ],
    )
    def test_make_event_refuses_bad_input(self, kwargs):
        with pytest.raises(ValueError):
            make_event(**kwargs)


# ---------------------------------------------------------------------------
# Contract surface the other units code against
# ---------------------------------------------------------------------------


class TestContractSurface:
    def test_constants_and_result_shapes(self):
        assert EVENT_MEASUREMENT_POINT_DECLARED == "measurement_point_declared"
        assert EVENT_MEASUREMENT_RECORDED == "measurement_recorded"
        assert EVENT_HEALTH_SNAPSHOT_COMPUTED == "health_snapshot_computed"
        assert EVENT_BASELINE_DECLARED == "baseline_declared"
        assert PRODUCER_NAME == "predictive-maintenance-mcp"
        assert AppendResult._fields == (
            "appended",
            "event_id",
            "duplicate",
            "offset_after",
        )
        assert LedgerReadResult._fields == ("events", "end_offset", "integrity")

    def test_package_reexports_the_store_api_unchanged(self):
        from predictive_maintenance_mcp import asset_ledger

        for name in (
            "SCHEMA_VERSION",
            "PRODUCER_NAME",
            "EVENT_MEASUREMENT_POINT_DECLARED",
            "EVENT_MEASUREMENT_RECORDED",
            "EVENT_HEALTH_SNAPSHOT_COMPUTED",
            "EVENT_BASELINE_DECLARED",
            "LedgerError",
            "LedgerLockTimeout",
            "LedgerWriteError",
            "AppendResult",
            "LedgerReadResult",
            "canonical_json",
            "content_hash",
            "short_id",
            "make_event",
            "LedgerStore",
            "build_asset_view",
        ):
            assert getattr(asset_ledger, name) is getattr(store_module, name), name

    def test_integrity_block_keys(self, store):
        integrity = store.read(ASSET).integrity
        assert set(integrity) == {
            "readable_records",
            "unreadable_records",
            "asset_id_mismatches",
            "duplicate_event_ids",
            "recorded_at_regressions",
            "unsupported_schema_versions",
            "issues",
            "issues_truncated",
        }
        assert integrity["issues_truncated"] is False


# ---------------------------------------------------------------------------
# Append and read
# ---------------------------------------------------------------------------


class TestAppendRead:
    def test_four_event_types_round_trip_in_file_order(self, store, root):
        events = [
            point_event(1),
            measurement_event("m1", "2026-01-01T00:00:00+00:00"),
            snapshot_event("m1", "health_snapshot/1+abc"),
            baseline_event(["m1"], declared_at="2026-01-02T00:00:00+00:00"),
        ]
        results = [store.append(ASSET, e) for e in events]
        assert all(isinstance(r, AppendResult) and r.appended for r in results)
        assert [r.event_id for r in results] == [e["event_id"] for e in events]

        raw = (root / "P-101.jsonl").read_bytes()
        assert raw.endswith(b"\n")
        assert raw.count(b"\n") == 4
        assert raw == b"".join(line_of(e) for e in events)
        assert results[-1].offset_after == len(raw)

        result = store.read(ASSET)
        assert result.events == events
        assert result.end_offset == len(raw)
        assert result.integrity["readable_records"] == 4
        assert result.integrity["unreadable_records"] == 0
        assert result.integrity["issues"] == []

    def test_missing_ledger_reads_empty_and_creates_nothing(self, store, root):
        result = store.read(ASSET)
        assert result == LedgerReadResult([], 0, result.integrity)
        assert result.integrity["readable_records"] == 0
        assert not root.exists()
        assert store.read_since(ASSET, 0).events == []
        assert not root.exists()

    def test_two_store_instances_see_the_same_view(self, root):
        writer = LedgerStore(root)
        writer.append(ASSET, point_event(1))
        writer.append(ASSET, measurement_event("m1", "2026-01-01T00:00:00+00:00"))
        reader = LedgerStore(root)
        assert reader.read_view(ASSET) == writer.read_view(ASSET)
        assert reader.read_view(ASSET)["end_offset"] == writer.read(ASSET).end_offset

    def test_root_created_on_first_append_only(self, store, root):
        assert not root.exists()
        store.list_assets()
        store.read(ASSET)
        assert not root.exists()
        store.append(ASSET, point_event(1))
        assert (root / "P-101.jsonl").is_file()
        assert (root / "P-101.jsonl.lock").is_file()

    def test_store_on_configured_ledger_dir_stays_in_the_sandbox(self, ledger_dir):
        from predictive_maintenance_mcp.config import get_ledger_dir

        store = LedgerStore(get_ledger_dir())
        store.append(ASSET, point_event(1))
        assert (ledger_dir / "P-101.jsonl").is_file()
        assert not (REPO_ROOT / "data" / "ledger" / "P-101.jsonl").exists()

    def test_append_versioned_none_appends_nothing(self, store, root):
        first = store.append(ASSET, point_event(1))
        result = store.append_versioned(ASSET, first.offset_after, lambda delta: None)
        assert result == AppendResult(False, "", False, first.offset_after)
        assert (root / "P-101.jsonl").stat().st_size == first.offset_after

    def test_append_versioned_absorbs_a_duplicate_seen_in_the_delta(self, store):
        store.append(ASSET, point_event(1))
        other = LedgerStore(store.root)
        second = point_event(2)
        other.append(ASSET, second)  # somebody else already wrote this exact event
        result = store.append_versioned(ASSET, 0, lambda delta: second)
        assert result.appended is False
        assert result.duplicate is True
        assert result.event_id == second["event_id"]
        assert versions_in(store.read(ASSET)) == [1, 2]


# ---------------------------------------------------------------------------
# Directory listing
# ---------------------------------------------------------------------------


class TestListAssets:
    def test_empty_directory_lists_nothing_and_creates_nothing(self, store, root):
        assert store.list_assets() == []
        assert store.integrity_report() == {"assets": [], "unlisted_files": []}
        assert not root.exists()
        root.mkdir()
        assert store.list_assets() == []
        assert list(root.iterdir()) == []

    def test_lists_only_files_whose_first_event_matches_the_stem(self, store, root):
        store.append(ASSET, point_event(1))
        store.append("Q-2", point_event(1, asset="Q-2"))
        # A copy of P-101's ledger under another name: events say P-101.
        (root / "P-101-DESKTOP.jsonl").write_bytes((root / "P-101.jsonl").read_bytes())
        (root / "Z-9.jsonl.lock").write_bytes(b"")  # lock without a ledger
        (root / "E-1.jsonl").write_bytes(b"")  # empty
        (root / "bad name.jsonl").write_bytes(line_of(point_event(1)))
        store.append_index_entry("m1", ASSET, "e", "2026-01-01T00:00:00+00:00")
        assert (root / "_measurements.jsonl").is_file()

        assert store.list_assets() == [ASSET, "Q-2"]
        assert store.integrity_report()["unlisted_files"] == [
            {"file": "E-1.jsonl", "reason": "no_readable_event"},
            {"file": "P-101-DESKTOP.jsonl", "reason": "asset_id_mismatch"},
            {"file": "bad name.jsonl", "reason": "invalid_stem"},
        ]
        assert MEASUREMENT_INDEX_NAME not in store.list_assets()

    def test_first_readable_event_decides_after_garbage(self, store, root):
        write_raw(
            root / "R-1.jsonl", b"garbage\n\n" + line_of(point_event(1, asset="R-1"))
        )
        assert store.list_assets() == ["R-1"]
        write_raw(
            root / "R-2.jsonl", b"garbage\n" + line_of(point_event(1, asset="R-1"))
        )
        assert store.list_assets() == ["R-1"]
        assert store.integrity_report()["unlisted_files"] == [
            {"file": "R-2.jsonl", "reason": "asset_id_mismatch"}
        ]


# ---------------------------------------------------------------------------
# Byte rules
# ---------------------------------------------------------------------------


class TestByteRules:
    def test_line_separators_and_carriage_returns_stay_one_record(self, store, root):
        note = "line\u2028sep\u2029para\u0085next\rret\nnl"
        event = point_event(1, note=note)
        store.append(ASSET, event)
        raw = (root / "P-101.jsonl").read_bytes()
        assert raw.count(b"\n") == 1 and raw.endswith(b"\n")
        assert b"\r" not in raw
        for literal in ("\u2028", "\u2029", "\u0085"):
            assert literal.encode("utf-8") not in raw
        result = store.read(ASSET)
        assert result.events == [event]
        assert result.events[0]["payload"]["note"] == note
        assert result.integrity["unreadable_records"] == 0

    def test_nul_tail_is_one_unreadable_record_and_bytes_stay(self, store, root):
        first = point_event(1)
        store.append(ASSET, first)
        path = root / "P-101.jsonl"
        clean_length = path.stat().st_size
        write_raw(path, b"\x00" * 4096)
        prefix_length = clean_length + 4096
        digest_before = hashlib.sha256(path.read_bytes()).hexdigest()

        torn = store.read(ASSET)
        assert torn.events == [first]
        assert torn.end_offset == clean_length  # the tail is not consumed
        assert torn.integrity["unreadable_records"] == 1
        assert torn.integrity["issues"] == [{"index": 1, "code": "no_terminator"}]

        second = point_event(2)
        store.append(ASSET, second)
        raw = path.read_bytes()
        assert hashlib.sha256(raw[:prefix_length]).hexdigest() == digest_before
        assert raw[prefix_length:] == b"\n" + line_of(second)

        result = store.read(ASSET)
        assert result.events == [first, second]
        assert result.integrity["unreadable_records"] == 1
        assert result.integrity["issues"] == [{"index": 1, "code": "invalid_json"}]
        delta = store.read_since(ASSET, torn.end_offset)
        assert delta.events == [second]
        assert delta.integrity["unreadable_records"] == 1
        assert delta.end_offset == result.end_offset == len(raw)

    def test_record_missing_only_its_newline_is_recovered_by_next_append(
        self, store, root
    ):
        first = point_event(1)
        path = root / "P-101.jsonl"
        write_raw(path, line_of(first)[:-1])
        before = store.read(ASSET)
        assert before.events == []
        assert before.end_offset == 0
        assert before.integrity["unreadable_records"] == 1

        second = point_event(2)
        store.append(ASSET, second)
        assert path.read_bytes() == line_of(first) + line_of(second)
        after = store.read(ASSET)
        assert after.events == [first, second]
        assert after.integrity["unreadable_records"] == 0
        assert after.integrity["issues"] == []

    def test_one_trailing_carriage_return_is_tolerated(self, store, root):
        event = point_event(1)
        write_raw(root / "P-101.jsonl", line_of(event)[:-1] + b"\r\n")
        result = store.read(ASSET)
        assert result.events == [event]
        assert result.integrity["issues"] == []


# ---------------------------------------------------------------------------
# Reader integrity
# ---------------------------------------------------------------------------


class TestReaderIntegrity:
    def test_invalid_json_mid_file_is_reported_by_index_and_later_lines_used(
        self, store, root
    ):
        first, second = point_event(1), point_event(2)
        write_raw(
            root / "P-101.jsonl", line_of(first) + b"{not json\n" + line_of(second)
        )
        result = store.read(ASSET)
        assert result.events == [first, second]
        assert result.integrity["unreadable_records"] == 1
        assert result.integrity["issues"] == [{"index": 1, "code": "invalid_json"}]

    @pytest.mark.parametrize(
        "raw, code",
        [
            (b"[1,2]\n", "not_object"),
            (b'{"a":1}\n', "envelope_incomplete"),
            (b"\n", "empty_record"),
            (b"\r\n", "empty_record"),
            (b"\xff\xfe\n", "invalid_json"),
        ],
    )
    def test_unreadable_shapes_are_classified_never_raised(
        self, store, root, raw, code
    ):
        event = point_event(1)
        write_raw(root / "P-101.jsonl", raw + line_of(event))
        result = store.read(ASSET)
        assert result.events == [event]
        assert result.integrity["unreadable_records"] == 1
        assert result.integrity["issues"] == [{"index": 0, "code": code}]

    def test_duplicated_line_leaves_the_view_unchanged(self, store, root):
        event = point_event(1)
        store.append(ASSET, event)
        view_before = store.read_view(ASSET)
        write_raw(root / "P-101.jsonl", line_of(event))
        result = store.read(ASSET)
        assert result.events == [event]
        assert result.integrity["duplicate_event_ids"] == 1
        assert result.integrity["issues"] == [
            {"index": 1, "code": "duplicate_event_id", "event_id": event["event_id"]}
        ]
        view_after = store.read_view(ASSET)
        assert view_without_bookkeeping(view_after) == view_without_bookkeeping(
            view_before
        )
        assert view_after["integrity"]["duplicate_event_ids"] == 1

    def test_unsupported_schema_version_is_excluded_and_counted(self, store, root):
        good = point_event(1)
        future = retagged(point_event(2), schema_version=99)
        write_raw(root / "P-101.jsonl", line_of(future) + line_of(good))
        result = store.read(ASSET)
        assert result.events == [good]
        assert result.integrity["unsupported_schema_versions"] == 1
        assert result.integrity["unreadable_records"] == 0
        assert result.integrity["issues"] == [
            {"index": 0, "code": "unsupported_schema_version", "schema_version": 99}
        ]
        view = store.read_view(ASSET)
        assert view["points"][POINT]["current"]["declaration_version"] == 1

    def test_unknown_event_type_is_kept_by_reader_and_ignored_by_view(
        self, store, root
    ):
        good = point_event(1)
        foreign = retagged(point_event(2), event_type="sensor_replaced")
        write_raw(root / "P-101.jsonl", line_of(good) + line_of(foreign))
        result = store.read(ASSET)
        assert result.events == [good, foreign]
        assert result.integrity["issues"] == []
        view = build_asset_view(ASSET, result.events, result.integrity)
        assert view["points"][POINT]["current"] == good["payload"]
        assert view["integrity"]["ignored_event_types"] == {"sensor_replaced": 1}
        assert view["event_count"] == 2

    def test_foreign_asset_events_are_excluded_and_listed(self, store, root):
        mine = point_event(1)
        theirs = point_event(1, asset="Q-2")
        write_raw(root / "P-101.jsonl", line_of(theirs) + line_of(mine))
        result = store.read(ASSET)
        assert result.events == [mine]
        assert result.integrity["asset_id_mismatches"] == 1
        assert result.integrity["issues"] == [
            {"index": 0, "code": "asset_id_mismatch", "found_asset_id": "Q-2"}
        ]

    def test_mismatch_entries_echo_only_a_grammatical_id(self, store, root):
        mine = point_event(1)
        bad_ids = ["../x", "x" * 150, "a b", "NUL"]
        foreign = b"".join(
            line_of(retagged(point_event(1), asset_id=b)) for b in bad_ids
        )
        write_raw(root / "P-101.jsonl", foreign + line_of(mine))
        result = store.read(ASSET)
        assert result.events == [mine]
        assert result.integrity["asset_id_mismatches"] == len(bad_ids)
        assert result.integrity["issues"] == [
            {"index": i, "code": "asset_id_mismatch", "found_asset_id": None}
            for i in range(len(bad_ids))
        ]
        listing = json.dumps(result.integrity)
        for bad in bad_ids:
            assert bad not in listing

    def test_issue_list_is_bounded_while_counters_stay_exact(self, store, root):
        good = point_event(1)
        garbage_lines = MAX_INTEGRITY_ISSUES + 50
        write_raw(root / "P-101.jsonl", b"{\n" * garbage_lines + line_of(good))
        result = store.read(ASSET)
        assert result.events == [good]
        assert result.integrity["unreadable_records"] == garbage_lines
        assert len(result.integrity["issues"]) == MAX_INTEGRITY_ISSUES
        assert result.integrity["issues_truncated"] is True
        assert result.integrity["issues"][-1] == {
            "index": MAX_INTEGRITY_ISSUES - 1,
            "code": "invalid_json",
        }
        assert store.read_view(ASSET)["points"][POINT]["current"] == good["payload"]

    def test_recorded_at_regression_is_informational(self, store):
        later = point_event(1, recorded_at=datetime(2026, 1, 2, tzinfo=UTC))
        earlier = point_event(2, recorded_at=datetime(2026, 1, 1, tzinfo=UTC))
        store.append(ASSET, later)
        store.append(ASSET, earlier)
        result = store.read(ASSET)
        assert result.events == [later, earlier]  # file order respected
        assert result.integrity["recorded_at_regressions"] == 1
        assert result.integrity["issues"] == [
            {
                "index": 1,
                "code": "recorded_at_regression",
                "event_id": earlier["event_id"],
            }
        ]
        assert versions_in(result) == [1, 2]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


BAD_IDS = [
    "../x",
    "..",
    "P-101/x",
    "/etc/x",
    "NUL",
    "com1.old",
    "-x",
    "P-101.",
    "a b",
    "",
]


class TestRefusals:
    @pytest.mark.parametrize("bad", BAD_IDS)
    def test_traversal_and_reserved_ids_are_refused_without_touching_disk(
        self, store, root, bad
    ):
        event = point_event(1)
        with pytest.raises(ValueError):
            store.append(bad, event)
        with pytest.raises(ValueError):
            store.read(bad)
        with pytest.raises(ValueError):
            store.read_since(bad, 0)
        with pytest.raises(ValueError):
            store.append_versioned(bad, 0, lambda delta: event)
        assert not root.exists()

    @pytest.mark.skipif(
        os.name != "nt", reason="backslash is a separator only on Windows"
    )
    def test_windows_backslash_traversal_is_refused(self, store, root):
        with pytest.raises(ValueError):
            store.append("..\\x", point_event(1))
        with pytest.raises(ValueError):
            store.read("..\\x")
        assert not root.exists()

    def test_refusal_message_is_a_closed_oracle(self, store, root):
        store.append("Q-777", point_event(1, asset="Q-777"))
        (root / "secret_note.jsonl").write_bytes(b"")
        for bad in ("../x", "NUL"):
            with pytest.raises(ValueError) as exc_info:
                store.read(bad)
            message = str(exc_info.value).replace(bad, "")
            assert "Q-777" not in message
            assert "secret_note" not in message
            assert "available" not in message
        # A case collision names the one colliding file (by design) and nothing else.
        with pytest.raises(ValueError) as exc_info:
            store.read("q-777")
        message = str(exc_info.value)
        assert "Q-777.jsonl" in message
        assert "secret_note" not in message
        assert "available" not in message

    def test_reserved_index_prefix_is_refused(self, store, root):
        for reserved in ("_measurements", "_x"):
            with pytest.raises(ValueError, match="reserved"):
                store.append(reserved, point_event(1))
            with pytest.raises(ValueError, match="reserved"):
                store.read(reserved)
        assert not root.exists()

    def test_case_collision_is_refused_on_every_os(self, store, root):
        store.append(ASSET, point_event(1))
        listing_before = sorted(os.listdir(root))
        size_before = (root / "P-101.jsonl").stat().st_size

        with pytest.raises(ValueError) as exc_info:
            store.append("p-101", point_event(1, asset="p-101"))
        assert "P-101.jsonl" in str(exc_info.value)
        with pytest.raises(ValueError):
            store.append_versioned(
                "p-101", 0, lambda delta: point_event(1, asset="p-101")
            )
        with pytest.raises(ValueError):
            store.read("p-101")

        assert (root / "P-101.jsonl").stat().st_size == size_before
        assert sorted(os.listdir(root)) == listing_before

    def test_event_for_another_asset_is_refused_before_any_write(self, store, root):
        with pytest.raises(ValueError, match="does not match"):
            store.append(ASSET, point_event(1, asset="Q-2"))
        assert not root.exists()

    def test_tampered_or_foreign_envelopes_are_refused(self, store, root):
        tampered = dict(point_event(1))
        tampered["payload"] = dict(tampered["payload"], note="edited")
        with pytest.raises(ValueError, match="event_id"):
            store.append(ASSET, tampered)
        with pytest.raises(ValueError, match="event_type"):
            store.append(ASSET, retagged(point_event(1), event_type="sensor_replaced"))
        with pytest.raises(ValueError, match="schema_version"):
            store.append(ASSET, retagged(point_event(1), schema_version=2))
        with pytest.raises(ValueError):
            store.append(ASSET, {"event_type": EVENT_TYPES[0]})
        with pytest.raises(ValueError):
            store.append(ASSET, "not an event")  # type: ignore[arg-type]
        assert not root.exists()

    def test_bad_offsets_are_refused(self, store):
        store.append(ASSET, point_event(1))
        for offset in (-1, True, 1.5, None):
            with pytest.raises(ValueError):
                store.read_since(ASSET, offset)  # type: ignore[arg-type]
            with pytest.raises(ValueError):
                store.append_versioned(ASSET, offset, lambda d: None)  # type: ignore[arg-type]

    def test_shrunken_ledger_is_detected_from_a_stale_offset(self, store, root):
        first = store.append(ASSET, point_event(1))
        with pytest.raises(LedgerError, match="truncated"):
            store.read_since(ASSET, first.offset_after + 1)
        with pytest.raises(LedgerError, match="truncated"):
            store.append_versioned(ASSET, first.offset_after + 1, lambda d: None)
        (root / "P-101.jsonl").unlink()  # outside the store; never done by it
        with pytest.raises(LedgerError, match="removed"):
            store.read_since(ASSET, first.offset_after)

    def test_unreadable_paths_raise_typed_errors_and_write_nothing(
        self, store, root, tmp_path
    ):
        root.mkdir()
        (root / "P-101.jsonl").mkdir()  # a directory where the ledger should be
        with pytest.raises(LedgerError, match="Cannot read"):
            store.read(ASSET)
        with pytest.raises(LedgerError, match="Cannot read"):
            store.read_view(ASSET)
        assert store.integrity_report()["unlisted_files"] == [
            {"file": "P-101.jsonl", "reason": "no_readable_event"}
        ]

        not_a_directory = tmp_path / "ledger.txt"
        not_a_directory.write_bytes(b"")
        broken = LedgerStore(not_a_directory)
        with pytest.raises(LedgerError, match="Cannot list"):
            broken.list_assets()
        with pytest.raises(LedgerError, match="Cannot list"):
            broken.append(ASSET, point_event(1))
        with pytest.raises(LedgerError, match="Cannot list"):
            broken.read(ASSET)
        assert not_a_directory.read_bytes() == b""

    def test_index_entry_validation(self, store, root):
        with pytest.raises(ValueError):
            store.append_index_entry("", ASSET, "e", "t")
        with pytest.raises(ValueError):
            store.append_index_entry("m1", "../x", "e", "t")
        with pytest.raises(ValueError):
            store.append_index_entry("m1", ASSET, None, "t")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            store.find_measurement_asset("")
        assert not root.exists()

    def test_error_hierarchy(self):
        assert issubclass(LedgerError, ValueError)
        assert issubclass(LedgerLockTimeout, LedgerError)
        assert issubclass(LedgerWriteError, LedgerError)


# ---------------------------------------------------------------------------
# Locking, within one process (runs on the whole matrix, no child interpreter)
# ---------------------------------------------------------------------------


class TestLockingInProcess:
    """A second open of the sidecar contends with the first on every OS
    (``flock`` is per open file description, ``msvcrt.locking`` per handle),
    so the timeout path can be exercised without a subprocess."""

    def test_busy_lock_times_out_within_the_timeout_then_recovers(self, root):
        holder = LedgerStore(root)
        holder.append(ASSET, point_event(1))
        path = root / "P-101.jsonl"
        bytes_before = path.read_bytes()
        _, lock_path = holder._paths(ASSET)
        fd = holder._acquire_lock(lock_path, what="test holder")
        try:
            timeout = 0.2
            contender = LedgerStore(root, lock_timeout=timeout)
            started = time.monotonic()
            with pytest.raises(LedgerLockTimeout, match="another process"):
                contender.append(ASSET, point_event(2))
            elapsed = time.monotonic() - started
            assert timeout - 0.02 <= elapsed < timeout + 2.0
            with pytest.raises(LedgerLockTimeout):
                contender.append_versioned(ASSET, 0, lambda delta: point_event(2))
            assert path.read_bytes() == bytes_before
            # The index has its own sidecar: a busy asset lock does not stop it.
            assert contender.append_index_entry("m1", ASSET, "e", "t").appended
        finally:
            holder._release_lock(fd)
        result = LedgerStore(root).append(ASSET, point_event(2))
        assert result.appended
        assert versions_in(LedgerStore(root).read(ASSET)) == [1, 2]

    def test_sidecar_is_never_replaced_or_truncated(self, root):
        store = LedgerStore(root)
        store.append(ASSET, point_event(1))
        lock = root / "P-101.jsonl.lock"
        before = lock.stat()
        store.append(ASSET, point_event(2))
        store.append_versioned(ASSET, 0, lambda delta: None)
        store.append_versioned(ASSET, 0, lambda delta: point_event(3))
        after = lock.stat()
        assert before.st_size == after.st_size == 0
        if before.st_ino:  # populated on NTFS, ext4 and APFS
            assert after.st_ino == before.st_ino
        assert versions_in(store.read(ASSET)) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Locking, across real processes
# ---------------------------------------------------------------------------


CHILD_HOLD_LOCK = SUBPROCESS_PIN + """
import sys, time
from pathlib import Path
from predictive_maintenance_mcp.asset_ledger.store import LedgerStore
root, marker, hold = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
store = LedgerStore(root)
_, lock = store._paths("P-101")
store._ensure_root()
fd = store._acquire_lock(lock, what="test holder")
marker.write_text("locked")
time.sleep(hold)
"""

CHILD_WRITER = SUBPROCESS_PIN + """
import sys
from pathlib import Path
from predictive_maintenance_mcp.asset_ledger.store import (
    LedgerStore, make_event, EVENT_MEASUREMENT_POINT_DECLARED,
)
root, count = Path(sys.argv[1]), int(sys.argv[2])
store = LedgerStore(root)
for i in range(count):
    event = make_event(
        EVENT_MEASUREMENT_POINT_DECLARED, "P-101",
        {"measurement_point_id": "de_h", "declaration_version": i + 1,
         "note": "x" * 200, "changed": []},
    )
    store.append("P-101", event)
"""

CHILD_VERSIONS = SUBPROCESS_PIN + """
import sys, time
from pathlib import Path
from predictive_maintenance_mcp.asset_ledger.store import (
    LedgerStore, make_event, build_asset_view, EVENT_MEASUREMENT_POINT_DECLARED,
)
root, who, count = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
barrier = Path(sys.argv[4])
(barrier / f"ready-{who}").write_text("1")
while not (barrier / "go").exists():
    time.sleep(0.002)

store = LedgerStore(root)
first = store.read("P-101")
events = list(first.events)
offset = first.end_offset
for _ in range(count):
    built = {}

    def build(delta):
        events.extend(delta)
        view = build_asset_view("P-101", events, {})
        current = view["points"].get("de_h", {}).get("current")
        version = (current["declaration_version"] if current else 0) + 1
        built["event"] = make_event(
            EVENT_MEASUREMENT_POINT_DECLARED, "P-101",
            {"measurement_point_id": "de_h", "declaration_version": version,
             "declared_by": who, "changed": [] if version == 1 else ["declared_by"]},
        )
        return built["event"]

    result = store.append_versioned("P-101", offset, build)
    assert result.appended, result
    events.append(built["event"])
    offset = result.offset_after
"""


def _spawn(tmp_path: Path, name: str, source: str, *args: str) -> subprocess.Popen:
    script = tmp_path / f"{name}.py"
    script.write_text(source, encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, str(script), *args],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for(path: Path, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + CHILD_TIMEOUT
    while not path.exists():
        if proc.poll() is not None:
            _, err = proc.communicate()
            raise AssertionError(f"child exited early ({proc.returncode}):\n{err}")
        if time.monotonic() > deadline:
            proc.kill()
            raise AssertionError(f"child never signalled {path.name}")
        time.sleep(0.01)


def _finish(proc: subprocess.Popen) -> None:
    out, err = proc.communicate(timeout=CHILD_TIMEOUT)
    assert proc.returncode == 0, f"child failed ({proc.returncode}):\n{out}\n{err}"


@pytest.mark.integration
class TestLockingAcrossProcesses:
    def test_lock_held_elsewhere_times_out_without_writing(self, tmp_path, root):
        LedgerStore(root).append(ASSET, point_event(1))
        path = root / "P-101.jsonl"
        bytes_before = path.read_bytes()
        marker = tmp_path / "locked"
        holder = _spawn(
            tmp_path, "holder", CHILD_HOLD_LOCK, str(root), str(marker), "600"
        )
        try:
            _wait_for(marker, holder)
            timeout = 0.5
            store = LedgerStore(root, lock_timeout=timeout)
            started = time.monotonic()
            with pytest.raises(LedgerLockTimeout) as exc_info:
                store.append(ASSET, point_event(2))
            elapsed = time.monotonic() - started
            assert timeout - 0.05 <= elapsed < timeout + 3.0
            assert "P-101.jsonl.lock" in str(exc_info.value)
            assert path.read_bytes() == bytes_before
            assert (root / "P-101.jsonl.lock").exists()  # never removed

            # The holder dies: the OS releases its lock, the next append succeeds.
            holder.kill()
            holder.wait(timeout=60)
            started = time.monotonic()
            result = LedgerStore(root).append(ASSET, point_event(2))
            assert result.appended
            assert time.monotonic() - started < 5.0
            assert versions_in(LedgerStore(root).read(ASSET)) == [1, 2]
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait(timeout=60)

    def test_reader_iterating_while_a_writer_appends_sees_no_corruption(
        self, tmp_path, root
    ):
        count = 300
        writer = _spawn(tmp_path, "writer", CHILD_WRITER, str(root), str(count))
        store = LedgerStore(root)
        observed: set = set()
        try:
            while writer.poll() is None:
                result = store.read(ASSET)  # must never raise
                for event in result.events:
                    assert compute_event_id(event) == event["event_id"]
                assert versions_in(result) == list(range(1, len(result.events) + 1))
                observed.add(len(result.events))
        finally:
            _finish(writer)
        final = store.read(ASSET)
        assert versions_in(final) == list(range(1, count + 1))
        assert final.integrity["unreadable_records"] == 0
        assert final.integrity["duplicate_event_ids"] == 0
        assert any(
            0 < n < count for n in observed
        ), f"the reader never overlapped the writer: {sorted(observed)}"

    def test_two_processes_assign_unique_monotone_versions(self, tmp_path, root):
        barrier = tmp_path / "barrier"
        barrier.mkdir()
        children = [
            _spawn(
                tmp_path,
                f"child_{who}",
                CHILD_VERSIONS,
                str(root),
                who,
                "20",
                str(barrier),
            )
            for who in ("A", "B")
        ]
        try:
            for who, child in zip(("A", "B"), children):
                _wait_for(barrier / f"ready-{who}", child)
            (barrier / "go").write_text("1")
            for child in children:
                _finish(child)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=60)

        result = LedgerStore(root).read(ASSET)
        assert versions_in(result) == list(range(1, 41))
        assert {e["payload"]["declared_by"] for e in result.events} == {"A", "B"}
        assert result.integrity["unreadable_records"] == 0
        assert result.integrity["duplicate_event_ids"] == 0
        assert (
            LedgerStore(root).read_view(ASSET)["points"][POINT]["current"][
                "declaration_version"
            ]
            == 40
        )


# ---------------------------------------------------------------------------
# Offsets: a versioned append reads only the delta
# ---------------------------------------------------------------------------


class TestOffsets:
    def test_append_versioned_reads_only_bytes_after_the_offset(
        self, store, root, monkeypatch
    ):
        store.append(ASSET, point_event(1))
        first = store.read(ASSET)
        offset = first.end_offset

        other = LedgerStore(root)
        other.append(ASSET, point_event(2))
        other.append(ASSET, point_event(3))
        size_before = (root / "P-101.jsonl").stat().st_size

        calls = []
        original = LedgerStore._read_segment

        def spy(self, path, start):
            data = original(self, path, start)
            calls.append((path.name, start, len(data)))
            return data

        monkeypatch.setattr(LedgerStore, "_read_segment", spy)

        delta_seen = []

        def build(delta):
            delta_seen.extend(delta)
            newest = max(
                e["payload"]["declaration_version"] for e in [*first.events, *delta]
            )
            return point_event(newest + 1)

        result = store.append_versioned(ASSET, offset, build)
        assert result.appended and not result.duplicate
        assert versions_in(LedgerReadResult(delta_seen, 0, {})) == [2, 3]
        assert calls == [("P-101.jsonl", offset, size_before - offset)]
        assert result.offset_after == (root / "P-101.jsonl").stat().st_size
        monkeypatch.undo()
        assert versions_in(store.read(ASSET)) == [1, 2, 3, 4]

    def test_offset_after_is_a_valid_read_since_offset(self, store):
        results = [store.append(ASSET, point_event(v)) for v in (1, 2, 3)]
        delta = store.read_since(ASSET, results[0].offset_after)
        assert versions_in(delta) == [2, 3]
        assert delta.end_offset == results[-1].offset_after
        assert store.read_since(ASSET, results[-1].offset_after).events == []


# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------


class TestView:
    def test_supersession_uses_the_latest_declaration_and_keeps_history(self, store):
        first = measurement_event("m1", "2026-01-01T00:00:00+00:00")
        second = measurement_event(
            "m1", "2026-01-01T00:00:00+00:00", version=2, changed=["rpm"], rpm=1480.0
        )
        store.append(ASSET, first)
        store.append(ASSET, second)
        view = store.read_view(ASSET)
        slot = view["measurements"]["m1"]
        assert slot["current"] == second["payload"]
        assert slot["history"] == [first["payload"], second["payload"]]
        assert slot["current"]["declaration"]["rpm"] == 1480.0
        assert view["ordered_measurement_ids"] == ["m1"]
        assert view["reattributed"] == []

    def test_baseline_withdrawal_leaves_no_current_but_keeps_history(self, store):
        declared = [
            baseline_event(["m1", "m2"], declared_at="2026-02-01T00:00:00+00:00"),
            baseline_event(["m1"], declared_at="2026-02-02T00:00:00+00:00"),
            baseline_event([], declared_at="2026-02-03T00:00:00+00:00"),
        ]
        for event in declared:
            store.append(ASSET, event)
        view = store.read_view(ASSET)
        slot = view["baselines"][POINT]
        assert slot["current"] is None
        assert [h["withdrawn"] for h in slot["history"]] == [False, False, True]
        assert [h["baseline_id"] for h in slot["history"]] == [
            e["payload"]["baseline_id"] for e in declared
        ]
        store.append(
            ASSET, baseline_event(["m2"], declared_at="2026-02-04T00:00:00+00:00")
        )
        assert store.read_view(ASSET)["baselines"][POINT]["current"][
            "measurement_ids"
        ] == ["m2"]

    def test_measurements_are_ordered_by_acquired_at_then_file_order(self, store):
        store.append(ASSET, measurement_event("m1", "2026-03-01T00:00:00+00:00"))
        store.append(ASSET, measurement_event("m2", "2026-01-01T00:00:00+00:00"))
        store.append(ASSET, measurement_event("m3", "2026-02-01T00:00:00+00:00"))
        store.append(ASSET, measurement_event("m4", "2026-01-01T00:00:00+00:00"))
        view = store.read_view(ASSET)
        assert view["ordered_measurement_ids"] == ["m2", "m4", "m3", "m1"]

    def test_measurements_are_ordered_by_instant_not_by_spelling(self, store):
        spellings = {
            "m1": "2026-01-01T03:00:00+02:00",  # 01:00Z, spelled last
            "m2": "2026-01-01T02:00:00+00:00",  # 02:00Z
            "m3": "2026-01-01T01:30:00",  # naive: ordered as UTC
            "m4": "2026-01-01T00:00:00Z",  # 00:00Z
            "m5": "not-a-date",  # unparsable: after every instant
        }
        for measurement_id, acquired_at in spellings.items():
            store.append(ASSET, measurement_event(measurement_id, acquired_at))
        no_instant = measurement_event("m6", "2026-01-01T00:00:00Z")
        payload = json.loads(json.dumps(no_instant["payload"]))
        del payload["declaration"]["acquired_at"]
        store.append(ASSET, retagged(no_instant, payload=payload))

        view = store.read_view(ASSET)
        assert view["ordered_measurement_ids"] == ["m4", "m1", "m3", "m2", "m5", "m6"]
        # The spellings alone would have put m1 last: the sort is by instant.
        by_spelling = sorted(k for k in spellings if k != "m5")
        assert sorted(by_spelling, key=spellings.get)[-1] == "m1"

    def test_reattribution_excludes_the_measurement_and_index_answers(self, store):
        other = "P-102"
        store.append(
            other, measurement_event("m1", "2026-01-01T00:00:00+00:00", asset=other)
        )
        moved = measurement_event(
            "m1",
            "2026-01-01T00:00:00+00:00",
            asset=other,
            declared_asset=ASSET,
            version=2,
            changed=["asset_id"],
        )
        store.append(other, moved)
        recorded_here = measurement_event("m1", "2026-01-01T00:00:00+00:00")
        store.append(ASSET, recorded_here)

        view = store.read_view(other)
        assert view["measurements"] == {}
        assert view["ordered_measurement_ids"] == []
        assert view["reattributed"] == [{"measurement_id": "m1", "to_asset_id": ASSET}]
        assert store.read_view(ASSET)["ordered_measurement_ids"] == ["m1"]

        assert store.find_measurement_asset("m1") is None
        store.append_index_entry("m1", other, "e1", "2026-01-01T00:00:00+00:00")
        store.append_index_entry(
            "m1", ASSET, recorded_here["event_id"], "2026-01-02T00:00:00+00:00"
        )
        assert store.find_measurement_asset("m1") == ASSET  # last entry wins
        assert store.read_index() == {"m1": ASSET}
        assert store.list_assets() == [ASSET, other]
        assert MEASUREMENT_INDEX_NAME not in store.list_assets()

    def test_index_reader_is_tolerant(self, store, root):
        path = root / "_measurements.jsonl"
        write_raw(path, b"garbage\n")
        write_raw(path, b'{"measurement_id":"m9","asset_id":"../x"}\n')
        write_raw(path, b'{"measurement_id":"m8"}\n')
        write_raw(path, b"[1]\n")
        store.append_index_entry("m1", ASSET, "e", "t")
        write_raw(path, b'{"measurement_id":"m2","asset_id":"Q-2"}')  # torn
        assert store.read_index() == {"m1": ASSET}
        assert store.find_measurement_asset("m9") is None

    def test_snapshots_grouped_by_measurement_and_lineage(self, store):
        store.append(ASSET, measurement_event("m1", "2026-01-01T00:00:00+00:00"))
        a1 = snapshot_event("m1", "health_snapshot/1+a")
        b1 = snapshot_event("m1", "health_snapshot/1+b")
        store.append(ASSET, a1)
        store.append(ASSET, b1)
        a2 = retagged(a1, payload=dict(a1["payload"], indicators={"rms": 0.7}))
        store.append(ASSET, a2)
        slot = store.read_view(ASSET)["measurements"]["m1"]
        assert slot["snapshots"] == [a1["payload"], b1["payload"], a2["payload"]]
        assert slot["snapshots_by_lineage"] == {
            "health_snapshot/1+a": a2["payload"],
            "health_snapshot/1+b": b1["payload"],
        }

    def test_orphan_snapshot_is_kept_but_never_ordered(self, store):
        store.append(ASSET, snapshot_event("m9", "health_snapshot/1+a"))
        view = store.read_view(ASSET)
        assert view["measurements"]["m9"]["current"] is None
        assert len(view["measurements"]["m9"]["snapshots"]) == 1
        assert view["ordered_measurement_ids"] == []

    def test_malformed_payloads_are_counted_not_raised(self, store, root):
        broken = make_event(EVENT_MEASUREMENT_POINT_DECLARED, ASSET, {"note": "no id"})
        store.append(ASSET, broken)
        store.append(ASSET, point_event(1))
        view = store.read_view(ASSET)
        assert view["integrity"]["malformed_payloads"] == 1
        assert view["points"][POINT]["current"]["declaration_version"] == 1
        assert view["event_count"] == 2

    def test_view_is_pure_and_deduplicates_a_merged_delta(self, store):
        events = [point_event(1), measurement_event("m1", "2026-01-01T00:00:00+00:00")]
        for event in events:
            store.append(ASSET, event)
        result = store.read(ASSET)
        once = build_asset_view(ASSET, result.events, result.integrity, end_offset=7)
        twice = build_asset_view(
            ASSET, result.events + result.events, result.integrity, end_offset=7
        )
        assert once == twice
        assert once["end_offset"] == 7
        assert once["event_count"] == 2
        assert once["integrity"]["ignored_event_types"] == {}
        assert result.integrity["issues"] == []  # the input was not mutated
        assert "ignored_event_types" not in result.integrity

    def test_view_of_an_absent_asset_is_empty(self, store):
        view = store.read_view("Q-0")
        assert view == {
            "asset_id": "Q-0",
            "event_count": 0,
            "points": {},
            "measurements": {},
            "ordered_measurement_ids": [],
            "reattributed": [],
            "baselines": {},
            "integrity": {
                **store.read("Q-0").integrity,
                "ignored_event_types": {},
                "malformed_payloads": 0,
            },
            "end_offset": 0,
        }
