"""Tests for imported measurements in the asset ledger.

Imports run through the real service (validation, record store, signal
store, ledger) on synthesized files in a sandboxed data directory; ledger
histories an older server would write are appended by hand through
``LedgerStore``.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import import_corpus as corpus
import numpy as np
import pytest

from predictive_maintenance_mcp import config
from predictive_maintenance_mcp.asset_ledger import service as ledger_service
from predictive_maintenance_mcp.asset_ledger.comparability import (
    assess_measurement_comparability,
    build_reference_context,
)
from predictive_maintenance_mcp.asset_ledger.service import (
    IMPORTED_DECLARATION_KEYS,
    build_declaration,
    declare_healthy_baseline,
    declare_measurement_point,
    record_measurements,
    reprocess_stale_snapshots,
)
from predictive_maintenance_mcp.asset_ledger.snapshot import context_digest
from predictive_maintenance_mcp.asset_ledger.store import (
    EVENT_HEALTH_SNAPSHOT_COMPUTED,
    EVENT_MEASUREMENT_RECORDED,
    LedgerStore,
    make_event,
)
from predictive_maintenance_mcp.signal_acquisition.import_config import (
    IMPORT_RAW_MARKER_SAMPLE_FORMAT,
    samples_digest,
)
from predictive_maintenance_mcp.signal_acquisition.loaders import load_raw_binary
from predictive_maintenance_mcp.signal_acquisition.repository import SignalRepository
from predictive_maintenance_mcp.signal_import import (
    SignalImportService,
    validate_import,
)

RATE = 25600
ASSET = "P-101"
OTHER = "P-202"
ACQUIRED = "2026-08-20T11:42:00+00:00"
DIRECTIONS = {1: "horizontal", 2: "vertical", 3: "axial", 4: "x"}


def rows(count: int, channels: int) -> list[str]:
    return [
        ",".join(
            [f"{row / RATE:.6f}"]
            + [f"{1000 * corpus.sample(row, c):.4f}" for c in range(channels)]
        )
        for row in range(count)
    ]


def channel(column: int, sensitivity: str = "100", asset: Optional[str] = None):
    binding: dict[str, Any] = {
        "measurement_point_id": "motor_de",
        "direction": DIRECTIONS[column],
    }
    if asset is not None:
        binding["asset_id"] = asset
    return {
        "column": column,
        "role": "vibration",
        "quantity": "acceleration",
        "unit": "g",
        "scaling": {
            "source_unit": "mV",
            "sensitivity": {
                "value": sensitivity,
                "unit": "mV/g",
                "source": "datasheet_nominal",
            },
            "dc_policy": "keep",
        },
        "binding": binding,
    }


def configuration(
    columns: tuple[int, ...] = (1, 2, 3),
    sensitivities: Optional[dict[int, str]] = None,
    asset: Optional[str] = None,
    acquired: str = ACQUIRED,
) -> dict[str, Any]:
    sensitivities = sensitivities or {}
    stated = {"basis": "user_stated"}
    bases: dict[str, Any] = {"time_base": stated, "defaults.acquired_at": stated}
    for column in columns:
        for part in ("unit", "scaling", "dc_policy", "binding"):
            bases[f"channels[{column}].{part}"] = stated
    return {
        "schema": "pmm-text-import-config/1",
        "interpretation": {
            "schema": "pmm-text-import/1",
            "dialect": {
                "encoding": "utf-8",
                "delimiter": ",",
                "decimal": ".",
                "header_line": 1,
            },
            "data_lines": {"first": 2},
            "time_base": {"source": "stated", "sampling_rate_hz": "25600"},
            "time_columns": [{"column": 0, "time_unit": "s"}],
            "defaults": {"asset_id": ASSET, "acquired_at": acquired},
            "channels": [
                channel(c, sensitivities.get(c, "100"), asset) for c in columns
            ],
        },
        "attestation": {"bases": bases, "overrides": {}},
    }


@pytest.fixture
def write(sandbox_data_dir):
    def _write(name: str, data: bytes) -> Path:
        path = sandbox_data_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    return _write


@pytest.fixture
def four_columns(write) -> Path:
    header = "time [s],a1 [mV],a2 [mV],a3 [mV],a4 [mV]"
    return write("run.csv", corpus.lines_to_bytes([header] + rows(400, 4)))


@pytest.fixture
def repo():
    return SignalRepository()


@pytest.fixture
def service(repo):
    return SignalImportService(repo)


def store() -> LedgerStore:
    return LedgerStore(config.get_ledger_dir())


def run_import(service, path: Path, conf: dict[str, Any], **kwargs):
    result = validate_import(str(path), conf, channel="python_api")
    assert result.outcome == "ready", [i.message for i in result.issues]
    return service.import_(
        str(path),
        conf,
        result.import_digest,
        channel="python_api",
        transport=None,
        **kwargs,
    )


def events(asset: str = ASSET) -> list[dict[str, Any]]:
    return store().read(asset).events


def current(asset: str, measurement_id: str) -> dict[str, Any]:
    return store().read_view(asset)["measurements"][measurement_id]["current"]


def by_column(outcome) -> dict[int, Any]:
    return {c.column: c for c in outcome.channels}


class TestRecording:
    def test_each_channel_recorded_with_its_own_interpretation(
        self, four_columns, service
    ):
        outcome = run_import(service, four_columns, configuration())
        assert [c.ledger_status for c in outcome.channels] == ["recorded"] * 3
        assert all(
            c.snapshot_status in ("complete", "partial") for c in outcome.channels
        )
        for item in outcome.channels:
            payload = current(ASSET, item.measurement_id)
            declaration = payload["declaration"]
            assert set(declaration) == set(IMPORTED_DECLARATION_KEYS)
            assert declaration["raw_format"] == {
                "sample_format": IMPORT_RAW_MARKER_SAMPLE_FORMAT
            }
            assert declaration["declared_by"] == "import"
            assert declaration["channel_index"] == item.column
            assert declaration["import_format"]["samples_digest"] == item.samples_digest
            assert payload["import_ref"] == {
                "import_id": outcome.import_id,
                "decoder_contract": "pmm-text-decoder/1",
            }

    def test_repeat_is_already_recorded_and_appends_nothing(
        self, four_columns, service
    ):
        run_import(service, four_columns, configuration())
        before = len(events())
        again = run_import(service, four_columns, configuration())
        assert again.status == "already_imported"
        assert [c.ledger_status for c in again.channels] == ["already_recorded"] * 3
        assert len(events()) == before

    def test_reconfirming_through_another_channel_changes_no_declaration(
        self, four_columns, service
    ):
        run_import(service, four_columns, configuration())
        before = len(events())
        conf = configuration()
        result = validate_import(
            str(four_columns), conf, channel="mcp_tool_argument", transport="stdio"
        )
        outcome = service.import_(
            str(four_columns),
            conf,
            result.import_digest,
            channel="mcp_tool_argument",
            transport="stdio",
        )
        assert [c.ledger_status for c in outcome.channels] == ["already_recorded"] * 3
        assert len(events()) == before

    def test_ae16_one_channel_corrected_then_one_added(self, four_columns, service):
        first = by_column(run_import(service, four_columns, configuration()))
        corrected = by_column(
            run_import(
                service,
                four_columns,
                configuration(sensitivities={3: "10.2"}),
                overwrite=True,
            )
        )
        assert corrected[1].ledger_status == "already_recorded"
        assert corrected[2].ledger_status == "already_recorded"
        assert corrected[3].ledger_status == "superseded"
        assert "import_format.scaling" in corrected[3].changed
        assert "import_format.samples_digest" in corrected[3].changed
        added = by_column(
            run_import(
                service,
                four_columns,
                configuration((1, 2, 3, 4), sensitivities={3: "10.2"}),
                overwrite=True,
            )
        )
        assert [added[c].ledger_status for c in (1, 2, 3)] == ["already_recorded"] * 3
        assert added[4].ledger_status == "recorded"
        assert first[1].measurement_id == added[1].measurement_id

    def test_ae8_corrected_sensitivity_supersedes_with_a_new_snapshot(
        self, write, service
    ):
        header = "time [s],a1 [mV],a2 [mV],a3 [mV]"
        ids = []
        paths = []
        for day in (20, 21, 22):
            path = write(
                f"day{day}.csv",
                corpus.lines_to_bytes([header] + rows(400 + day, 3)),
            )
            acquired = f"2026-08-{day}T11:42:00+00:00"
            outcome = run_import(service, path, configuration((3,), acquired=acquired))
            ids.append(outcome.channels[0].measurement_id)
            paths.append((path, acquired))
        declare_healthy_baseline(
            store=store(),
            asset_id=ASSET,
            measurement_point_id="motor_de",
            measurement_ids=ids,
            declared_by="test",
        )
        path, acquired = paths[0]
        corrected_conf = configuration(
            (3,), sensitivities={3: "10.2"}, acquired=acquired
        )
        warned = validate_import(str(path), corrected_conf, channel="python_api")
        [check] = warned.ledger_precheck
        assert check.measurement_status == "supersedes"
        assert check.baseline_member is True
        mid = ids[0]

        def snapshots() -> set[str]:
            return {
                e["payload"]["snapshot_id"]
                for e in events()
                if e["event_type"] == EVENT_HEALTH_SNAPSHOT_COMPUTED
                and e["payload"]["measurement_id"] == mid
            }

        before = snapshots()
        corrected = service.import_(
            str(path),
            corrected_conf,
            warned.import_digest,
            channel="python_api",
            transport=None,
            overwrite=True,
        )
        assert corrected.channels[0].ledger_status == "superseded"
        assert "import_format.scaling" in corrected.channels[0].changed
        assert len(snapshots() - before) == 1
        declaration = current(ASSET, mid)["declaration"]
        assert declaration["import_format"]["scaling"]["sensitivity"]["value"] == "10.2"

    def test_aba_restores_and_the_ledger_holds_three_versions(
        self, four_columns, service
    ):
        run_import(service, four_columns, configuration())
        run_import(
            service,
            four_columns,
            configuration(sensitivities={3: "10.2"}),
            overwrite=True,
        )
        conf = configuration()
        result = validate_import(str(four_columns), conf, channel="python_api")
        assert result.outcome == "ready"
        restored = [p for p in result.ledger_precheck if p.column == 3][0]
        assert restored.restores_superseded is True
        outcome = service.import_(
            str(four_columns),
            conf,
            result.import_digest,
            channel="python_api",
            transport=None,
            overwrite=True,
        )
        mid = by_column(outcome)[3].measurement_id
        assert by_column(outcome)[3].ledger_status == "superseded"
        slot = store().read_view(ASSET)["measurements"][mid]
        assert len(slot["history"]) == 3

    def test_snapshot_is_computed_from_the_handed_array_not_the_store(
        self, four_columns, service, repo, monkeypatch
    ):
        def poisoned(signal_id):  # pragma: no cover - must not be called
            raise AssertionError("the ledger looked the samples up in the store")

        monkeypatch.setattr(repo, "get_signal", poisoned)
        outcome = run_import(service, four_columns, configuration())
        assert all(
            c.snapshot_status in ("complete", "partial") for c in outcome.channels
        )

    def test_a_ledger_failure_on_one_channel_is_retried_by_the_repeat(
        self, four_columns, service, monkeypatch
    ):
        real = LedgerStore.append_versioned
        calls = {"n": 0}

        def flaky(self, asset_id, offset, build):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError(28, "No space left on device")
            return real(self, asset_id, offset, build)

        monkeypatch.setattr(LedgerStore, "append_versioned", flaky)
        first = by_column(run_import(service, four_columns, configuration()))
        assert first[1].ledger_status == "recorded"
        assert first[2].ledger_status == "not_recorded"
        assert "No space left" in first[2].reason
        assert first[3].ledger_status == "recorded"
        assert first[2].materialized is True
        monkeypatch.setattr(LedgerStore, "append_versioned", real)
        again = by_column(run_import(service, four_columns, configuration()))
        assert again[2].ledger_status == "recorded"
        assert again[1].ledger_status == "already_recorded"
        recorded = [
            e for e in events() if e["event_type"] == EVENT_MEASUREMENT_RECORDED
        ]
        assert len(recorded) == 3


class TestPlainNeverReplacesAnImport:
    def _plain_info(self, write, outcome, column: int, asset: str = ASSET):
        """The info load_signal would record for the same bytes and column."""
        mid = by_column(outcome)[column].measurement_id
        identity = current(ASSET, mid)["declaration"]
        measurement = {
            key: identity[key]
            for key in (
                "asset_id",
                "measurement_point_id",
                "acquired_at",
                "timezone_declared",
                "timestamp_suspect",
                "rpm",
                "load",
                "operating_state",
                "sensor_id",
                "direction",
            )
        }
        measurement.update(
            asset_id=asset,
            declared_by=None,
            measurement_id=mid,
            channel_index=column,
            content_sha256=current(ASSET, mid)["file"]["content_sha256"],
            size_bytes=current(ASSET, mid)["file"]["size_bytes"],
        )
        return {
            "signal_id": "plain",
            "filepath": str(config.DATA_DIR / "run.csv"),
            "sampling_rate": 25600.0,
            "signal_unit": "g",
            "raw_format": None,
            "measurement": measurement,
        }

    def test_ae12_a_plain_load_is_not_recorded_and_appends_no_snapshot(
        self, four_columns, service, write
    ):
        outcome = run_import(service, four_columns, configuration((1,)))
        before = len(events())
        info = self._plain_info(write, outcome, 1)
        [plain] = record_measurements(
            [info],
            {"plain": np.zeros(400)},
            store=store(),
            data_dir=config.DATA_DIR,
        )
        assert plain["ledger_status"] == "not_recorded"
        assert "import_signal_file" in plain["reason"]
        assert plain["snapshot_status"] == "skipped"
        assert len(events()) == before

    def test_a_plain_load_moving_it_to_another_asset_appends_nothing(
        self, four_columns, service, write
    ):
        outcome = run_import(service, four_columns, configuration((1,)))
        before = len(events())
        info = self._plain_info(write, outcome, 1, asset=OTHER)
        [plain] = record_measurements(
            [info], {"plain": np.zeros(400)}, store=store(), data_dir=config.DATA_DIR
        )
        assert plain["ledger_status"] == "not_recorded"
        assert len(events()) == before
        assert store().read(OTHER).events == []

    def test_an_import_supersedes_a_plain_declaration(self, four_columns, service):
        outcome = run_import(service, four_columns, configuration((1,)))
        mid = by_column(outcome)[1].measurement_id
        payload = current(ASSET, mid)
        plain = dict(payload["declaration"])
        plain.pop("import_format")
        plain["raw_format"] = None
        older = {**payload, "declaration": plain, "declaration_version": 2}
        older.pop("import_ref")
        store().append(ASSET, make_event(EVENT_MEASUREMENT_RECORDED, ASSET, older))
        again = run_import(service, four_columns, configuration((1,)))
        assert by_column(again)[1].ledger_status == "superseded"

    def test_an_import_moves_a_measurement_the_index_does_not_name(
        self, four_columns, service
    ):
        outcome = run_import(service, four_columns, configuration((1,)))
        mid = by_column(outcome)[1].measurement_id
        index = Path(config.get_ledger_dir()) / "_index.jsonl"
        # As after an index append that failed: the ledger holds it, the
        # index does not.
        if index.exists():
            index.write_bytes(b"")
        moved = run_import(
            service, four_columns, configuration((1,), asset=OTHER), overwrite=True
        )
        assert by_column(moved)[1].reattributed_from == ASSET
        assert current(OTHER, mid)["declaration"]["asset_id"] == OTHER
        assert {"measurement_id": mid, "to_asset_id": OTHER} in (
            store().read_view(ASSET)["reattributed"]
        )


class TestReplay:
    def _stale(self, outcome) -> None:
        """Declare the point so every snapshot of it is stale."""
        declare_measurement_point(
            store=store(),
            asset_id=ASSET,
            measurement_point_id="motor_de",
            machine_group=2,
            support_type="rigid",
            declared_by="test",
        )

    def test_ae9_column_3_replays_to_its_samples_digest(self, four_columns, service):
        outcome = run_import(service, four_columns, configuration((3,)))
        self._stale(outcome)
        block = reprocess_stale_snapshots(
            ASSET,
            "motor_de",
            store=store(),
            data_dir=config.DATA_DIR,
            import_roots=config.get_import_roots(),
        )
        [entry] = block["results"]
        assert entry["outcome"] == "reprocessed", entry

    def test_replay_works_with_the_record_directory_deleted(
        self, four_columns, service
    ):
        outcome = run_import(service, four_columns, configuration((3,)))
        shutil.rmtree(config.get_import_dir())
        self._stale(outcome)
        block = reprocess_stale_snapshots(
            ASSET, "motor_de", store=store(), data_dir=config.DATA_DIR
        )
        assert block["results"][0]["outcome"] == "reprocessed"

    def test_a_samples_digest_mismatch_is_not_reprocessable(
        self, four_columns, service, monkeypatch
    ):
        outcome = run_import(service, four_columns, configuration((3,)))
        self._stale(outcome)
        real = ledger_service._replay_interpretation

        def shifted(declaration):
            interpretation = real(declaration)
            interpretation["data_lines"] = {"first": 3, "last": None}
            return interpretation

        monkeypatch.setattr(ledger_service, "_replay_interpretation", shifted)
        block = reprocess_stale_snapshots(
            ASSET, "motor_de", store=store(), data_dir=config.DATA_DIR
        )
        [entry] = block["results"]
        assert entry["outcome"] == "not_reprocessable"
        assert entry["reason_code"] == "undecodable"
        assert "samples digest" in entry["reason"]
        assert entry["action_id"] == "undecodable+imported_measurement"

    def test_missing_or_changed_file_refused_before_decoding(
        self, four_columns, service, monkeypatch
    ):
        outcome = run_import(service, four_columns, configuration((3,)))
        self._stale(outcome)
        calls: list[str] = []
        monkeypatch.setattr(
            ledger_service,
            "_decode_import",
            lambda *a, **k: calls.append("decode"),
        )
        four_columns.write_bytes(
            four_columns.read_bytes().replace(b"0.0000", b"0.0001", 1)
        )
        block = reprocess_stale_snapshots(
            ASSET, "motor_de", store=store(), data_dir=config.DATA_DIR
        )
        assert block["results"][0]["reason_code"] == "content_differs"
        four_columns.unlink()
        block = reprocess_stale_snapshots(
            ASSET, "motor_de", store=store(), data_dir=config.DATA_DIR
        )
        assert block["results"][0]["reason_code"] == "file_missing"
        assert "import_signal_file" in json.dumps(block["needs_action"])
        assert calls == []

    def test_outside_the_import_roots_refused_without_any_stat(
        self, service, tmp_path, monkeypatch
    ):
        root = tmp_path / "exports"
        root.mkdir()
        path = root / "run.csv"
        header = "time [s],a1 [mV],a2 [mV],a3 [mV]"
        path.write_bytes(corpus.lines_to_bytes([header] + rows(400, 3)))
        monkeypatch.setenv("PMM_IMPORT_ROOTS", str(root))
        outcome = run_import(service, path, configuration((3,)))
        self._stale(outcome)
        monkeypatch.delenv("PMM_IMPORT_ROOTS")
        seen: list[str] = []
        real_stat = os.stat

        def watched(target, *args, **kwargs):
            if "exports" in str(target):
                seen.append(str(target))
            return real_stat(target, *args, **kwargs)

        monkeypatch.setattr(ledger_service.os, "stat", watched)
        block = reprocess_stale_snapshots(
            ASSET,
            "motor_de",
            store=store(),
            data_dir=config.DATA_DIR,
            import_roots=config.get_import_roots(),
        )
        [entry] = block["results"]
        assert entry["reason_code"] == "location_outside_data_dir"
        assert "PMM_IMPORT_ROOTS" in entry["reason"]
        assert seen == []

    def test_a_network_location_is_refused_without_any_stat(self, monkeypatch):
        seen: list[str] = []
        real_stat = os.stat
        monkeypatch.setattr(
            ledger_service.os,
            "stat",
            lambda target, *a, **k: seen.append(str(target))
            or real_stat(target, *a, **k),
        )
        passed, failed, checks = ledger_service._check_locations(
            ["//server/share/run.csv"], config.DATA_DIR, recorded_size=10
        )
        assert passed == []
        assert failed["//server/share/run.csv"][0] == "location_outside_data_dir"
        assert seen == []

    def test_supported_local_locations_reach_the_file_check(self, tmp_path):
        data_dir = (tmp_path / "data").resolve()
        data_dir.mkdir()
        (data_dir / "run.csv").write_bytes(b"0123456789")
        outside = tmp_path.resolve() / "elsewhere.csv"
        outside.write_bytes(b"0123456789")
        passed, failed, _ = ledger_service._check_locations(
            ["run.csv", str(outside)], data_dir, recorded_size=10
        )
        assert failed == {}
        assert [location for location, _ in passed] == ["run.csv", str(outside)]

    def test_import_then_plain_history_is_not_reprocessable(
        self, four_columns, service
    ):
        outcome = run_import(service, four_columns, configuration((3,)))
        mid = by_column(outcome)[3].measurement_id
        payload = current(ASSET, mid)
        plain = dict(payload["declaration"])
        plain.pop("import_format")
        plain["raw_format"] = None
        older = {**payload, "declaration": plain, "declaration_version": 2}
        older.pop("import_ref")
        # Written directly, as a server predating the import would.
        store().append(ASSET, make_event(EVENT_MEASUREMENT_RECORDED, ASSET, older))
        self._stale(outcome)
        block = reprocess_stale_snapshots(
            ASSET, "motor_de", store=store(), data_dir=config.DATA_DIR
        )
        [entry] = block["results"]
        assert entry["reason_code"] == "declaration_unresolvable"
        assert "predates the import" in entry["reason"]


class TestDigestsAndCompatibility:
    def test_context_digest_includes_import_format_only_when_present(self):
        context = {"signal_unit": "g", "direction": "horizontal"}
        plain = {"sampling_rate": 25600.0, "raw_format": None}
        imported = {**plain, "import_format": {"samples_digest": "a" * 64}}
        corrected = {**plain, "import_format": {"samples_digest": "b" * 64}}
        assert context_digest(context, plain) != context_digest(context, imported)
        assert context_digest(context, imported) != context_digest(context, corrected)

    def test_plain_declarations_keep_their_key_set(self):
        info = {
            "signal_id": "s",
            "sampling_rate": 1000.0,
            "signal_unit": "g",
            "raw_format": None,
            "measurement": {"asset_id": ASSET, "channel_index": 0},
        }
        assert "import_format" not in build_declaration(info)

    def test_the_marker_is_refused_by_the_raw_decoder_before_reading(self, tmp_path):
        missing = tmp_path / "never-created.bin"
        with pytest.raises(ValueError) as refused:
            load_raw_binary(missing, sample_format=IMPORT_RAW_MARKER_SAMPLE_FORMAT)
        assert not isinstance(refused.value, FileNotFoundError)
        assert not missing.exists()

    def test_imported_units_are_physical_and_a_recalibration_qualifies(self):
        def declaration(divisor: str) -> dict[str, Any]:
            return {
                "signal_unit": "g",
                "sampling_rate": 25600.0,
                "direction": "horizontal",
                "raw_format": {"sample_format": IMPORT_RAW_MARKER_SAMPLE_FORMAT},
                "import_format": {"scaling": {"divisor": divisor}},
            }

        reference = build_reference_context([declaration("100")])
        same = assess_measurement_comparability(declaration("100"), None, reference)
        changed = assess_measurement_comparability(declaration("10.2"), None, reference)
        codes = {q["code"] for q in changed["qualifications"]}
        assert "acquisition_setup_differs" in codes
        assert "amplitude_not_physical" not in {
            q["code"] for q in same["qualifications"]
        }
        assert "acquisition_setup_differs" not in {
            q["code"] for q in same["qualifications"]
        }


class TestRematerializeCrossCheck:
    def test_ledger_with_another_interpretation_is_reported(
        self, four_columns, service
    ):
        first = run_import(service, four_columns, configuration((3,)))
        run_import(
            service,
            four_columns,
            configuration((3,), sensitivities={3: "10.2"}),
            overwrite=True,
        )
        restored = SignalImportService(SignalRepository()).rematerialize(
            first.import_id
        )
        assert any("another interpretation" in n for n in restored.notices)

    def test_matching_ledger_adds_no_notice(self, four_columns, service):
        first = run_import(service, four_columns, configuration((3,)))
        restored = SignalImportService(SignalRepository()).rematerialize(
            first.import_id
        )
        assert restored.notices == []


def test_imported_samples_digest_matches_the_stored_array(four_columns, service, repo):
    outcome = run_import(service, four_columns, configuration((3,)))
    item = outcome.channels[0]
    assert samples_digest(repo.get_signal(item.signal_id)) == item.samples_digest


@pytest.mark.anyio
async def test_another_tool_answers_while_an_imported_file_replays(
    four_columns, service, monkeypatch
):
    """The reprocess branch of assess_asset_change runs on a worker thread."""
    import threading
    import time

    import anyio

    from predictive_maintenance_mcp.mcp_tools import asset_tools

    outcome = run_import(service, four_columns, configuration((3,)))
    TestReplay()._stale(outcome)
    real = ledger_service._decode_import
    started = threading.Event()

    def slow(*args, **kwargs):
        started.set()
        time.sleep(0.6)
        return real(*args, **kwargs)

    monkeypatch.setattr(ledger_service, "_decode_import", slow)
    order: list[str] = []

    async def assessment() -> None:
        await asset_tools.assess_asset_change(
            None, asset_id=ASSET, measurement_point_id="motor_de", reprocess=True
        )
        order.append("assessment")

    async def other_tool() -> None:
        while not started.is_set():
            await anyio.sleep(0.01)
        order.append("other")

    async with anyio.create_task_group() as group:
        group.start_soon(assessment)
        group.start_soon(other_tool)
    assert order == ["other", "assessment"]


class TestRemainingScenarios:
    def test_ae15_a_second_column_for_the_same_point_and_slot_is_never_ready(
        self, four_columns, service
    ):
        """Import column 3, then validate column 4 bound to the same point
        and acquisition slot: ready is refused until column 3 is rebound."""
        run_import(service, four_columns, configuration((3,)))
        conf = configuration((4,))
        conf["interpretation"]["channels"][0]["binding"]["direction"] = DIRECTIONS[3]
        result = validate_import(str(four_columns), conf, channel="python_api")
        assert result.outcome != "ready"
        [check] = result.ledger_precheck
        assert check.column_swap is not None
        assert result.import_digest is None

    def test_a_recorded_contract_replays_under_a_newer_current_contract(
        self, four_columns, service, monkeypatch
    ):
        outcome = run_import(service, four_columns, configuration((3,)))
        TestReplay()._stale(outcome)
        from predictive_maintenance_mcp.signal_acquisition import text_decoder

        asked: list[str] = []
        real = text_decoder.get_decoder

        def spy(contract: str):
            asked.append(contract)
            return real(contract)

        monkeypatch.setattr(ledger_service, "DECODER_CONTRACT", "pmm-text-decoder/2")
        monkeypatch.setattr(ledger_service, "get_decoder", spy)
        block = reprocess_stale_snapshots(
            ASSET, "motor_de", store=store(), data_dir=config.DATA_DIR
        )
        assert block["results"][0]["outcome"] == "reprocessed"
        assert asked == ["pmm-text-decoder/1"]

    def test_aba_reuses_the_first_snapshot_and_appends_none(
        self, four_columns, service
    ):
        first = by_column(run_import(service, four_columns, configuration()))
        mid = first[3].measurement_id

        def snapshot_ids() -> list[str]:
            return [
                e["payload"]["snapshot_id"]
                for e in events()
                if e["event_type"] == EVENT_HEALTH_SNAPSHOT_COMPUTED
                and e["payload"]["measurement_id"] == mid
            ]

        a = snapshot_ids()
        run_import(
            service,
            four_columns,
            configuration(sensitivities={3: "10.2"}),
            overwrite=True,
        )
        ab = snapshot_ids()
        run_import(service, four_columns, configuration(), overwrite=True)
        aba = snapshot_ids()
        assert len(a) == 1 and len(ab) == 2
        assert aba == ab
        assert (
            current(ASSET, mid)["declaration"]["import_format"]["scaling"][
                "sensitivity"
            ]["value"]
            == "100"
        )


def _linked(target: Path, link: Path) -> bool:
    """A directory link (a junction on Windows, a symlink elsewhere)."""
    try:
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(str(target), str(link))
        else:
            os.symlink(target, link, target_is_directory=True)
    except (OSError, AttributeError, NotImplementedError):
        return False
    return True


class TestReviewFindings:
    def test_an_import_root_reached_through_a_link_replays(
        self, service, tmp_path, monkeypatch
    ):
        real = tmp_path / "real_exports"
        real.mkdir()
        link = tmp_path / "linked_exports"
        if not _linked(real, link):
            pytest.skip("directory links are not available here")
        header = "time [s],a1 [mV],a2 [mV],a3 [mV]"
        (real / "run.csv").write_bytes(corpus.lines_to_bytes([header] + rows(400, 3)))
        monkeypatch.setenv("PMM_IMPORT_ROOTS", str(link))
        outcome = run_import(service, link / "run.csv", configuration((3,)))
        mid = outcome.channels[0].measurement_id
        assert current(ASSET, mid)["file"]["location"] == str(link / "run.csv")
        TestReplay()._stale(outcome)
        block = reprocess_stale_snapshots(
            ASSET,
            "motor_de",
            store=store(),
            data_dir=config.DATA_DIR,
            import_roots=config.get_import_roots(),
        )
        assert block["results"][0]["outcome"] == "reprocessed", block["results"]

    def test_ledger_results_name_the_import_not_the_path(
        self, service, tmp_path, monkeypatch
    ):
        root = tmp_path / "exports"
        root.mkdir()
        path = root / "run.csv"
        header = "time [s],a1 [mV],a2 [mV],a3 [mV]"
        path.write_bytes(corpus.lines_to_bytes([header] + rows(400, 3)))
        monkeypatch.setenv("PMM_IMPORT_ROOTS", str(root))
        # An explicit prefix: a path-derived signal id carries the folder name
        # on purpose, and this test looks for the path elsewhere.
        outcome = run_import(service, path, configuration((3,)), signal_id_prefix="p")
        shown = f"import {outcome.import_id[:12]} column 3"
        history = ledger_service.asset_history(store(), ASSET, max_measurements=5)
        assert json.dumps(history).count("exports") == 0
        assert shown in json.dumps(history)
        TestReplay()._stale(outcome)
        path.unlink()
        block = reprocess_stale_snapshots(
            ASSET,
            "motor_de",
            store=store(),
            data_dir=config.DATA_DIR,
            import_roots=config.get_import_roots(),
        )
        [entry] = block["results"]
        assert entry["reason_code"] == "file_missing"
        assert "exports" not in json.dumps(entry)
        assert shown in entry["reason"]

    @pytest.mark.anyio
    async def test_a_malformed_import_roots_setting_keeps_reprocessing(
        self, four_columns, service, monkeypatch
    ):
        from predictive_maintenance_mcp.mcp_tools import asset_tools

        outcome = run_import(service, four_columns, configuration((3,)))
        TestReplay()._stale(outcome)
        monkeypatch.setenv("PMM_IMPORT_ROOTS", os.pathsep)
        result = await asset_tools.assess_asset_change(
            None, asset_id=ASSET, measurement_point_id="motor_de", reprocess=True
        )
        assert result.reprocess["reprocessed"] == 1


class TestRemainingBranches:
    @pytest.mark.parametrize(
        "code",
        [
            "declaration_unresolvable",
            "location_outside_data_dir",
            "file_missing",
            "content_differs",
            "undecodable",
        ],
    )
    def test_actions_for_an_imported_measurement_name_the_import(self, code):
        from predictive_maintenance_mcp.asset_ledger.assessment import reprocess_action

        imported = reprocess_action(code, imported=True)
        assert imported != reprocess_action(code)
        assert "import" in imported
        assert "load_signal with" not in imported

    def test_other_codes_keep_the_plain_action(self):
        from predictive_maintenance_mcp.asset_ledger.assessment import reprocess_action

        assert reprocess_action("append_failed", imported=True) == reprocess_action(
            "append_failed"
        )

    def test_the_marker_without_an_interpretation_is_not_reprocessable(self):
        from predictive_maintenance_mcp.asset_ledger.snapshot import snapshot_blocker

        payload = {
            "declaration": {
                "sampling_rate": 25600.0,
                "signal_unit": "g",
                "direction": "horizontal",
                "raw_format": {"sample_format": IMPORT_RAW_MARKER_SAMPLE_FORMAT},
            },
            "file": {"location": "x.csv", "content_sha256": "a" * 64},
            "locations": ["x.csv"],
        }
        code, reason = snapshot_blocker(payload, None)
        assert code == "declaration_unresolvable"
        assert "first column" in reason

    def test_a_handed_array_with_another_digest_gets_no_snapshot(
        self, four_columns, service, repo
    ):
        outcome = run_import(service, four_columns, configuration((3,)))
        mid = outcome.channels[0].measurement_id
        payload = current(ASSET, mid)
        info = {
            "signal_id": "x",
            "filepath": str(four_columns),
            "sampling_rate": 25600.0,
            "signal_unit": "g",
            "measurement": {
                **{
                    k: payload["declaration"][k]
                    for k in (
                        "asset_id",
                        "measurement_point_id",
                        "acquired_at",
                        "timezone_declared",
                        "timestamp_suspect",
                        "rpm",
                        "load",
                        "operating_state",
                        "sensor_id",
                        "direction",
                        "declared_by",
                    )
                },
                "measurement_id": mid,
                "channel_index": 3,
                "content_sha256": payload["file"]["content_sha256"],
                "size_bytes": payload["file"]["size_bytes"],
            },
            "import_format": {
                **payload["declaration"]["import_format"],
                "data_lines": {"first": 3, "last": None},
            },
        }
        [result] = record_measurements(
            [info], {"x": np.zeros(400)}, store=store(), data_dir=config.DATA_DIR
        )
        assert result["snapshot_status"] == "failed"
        assert "samples digest" in result["reason"]

    @pytest.mark.parametrize("divisor", ["not-a-number", 12, "0", "inf"])
    def test_an_unreadable_divisor_has_no_multiplier(self, divisor):
        from predictive_maintenance_mcp.asset_ledger import comparability

        declaration = {"import_format": {"scaling": {"divisor": divisor}}}
        assert comparability._scale_factor_of(declaration) is None

    def test_a_file_the_open_refuses_is_reported_not_reprocessable(
        self, four_columns, service, monkeypatch
    ):
        from predictive_maintenance_mcp.path_safety import PathRefused

        outcome = run_import(service, four_columns, configuration((3,)))
        TestReplay()._stale(outcome)

        def refusing(*args, **kwargs):
            raise PathRefused("not_a_regular_file", "refused at open")

        monkeypatch.setattr(ledger_service, "open_import_file", refusing)
        block = reprocess_stale_snapshots(
            ASSET, "motor_de", store=store(), data_dir=config.DATA_DIR
        )
        [entry] = block["results"]
        assert entry["reason_code"] == "file_inaccessible"
