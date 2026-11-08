"""Tests for the signal import service: import, re-materialization, index.

Files are synthesized into a sandboxed data directory (or a sandboxed import
root) and go through the real path gate, strict decoder and validation; the
signal store is a fresh ``SignalRepository``; the record and ledger
directories are the autouse sandboxes of ``conftest``.
"""

import os
from pathlib import Path
from typing import Any, Optional

import import_corpus as corpus
import numpy as np
import pytest

from predictive_maintenance_mcp import config
from predictive_maintenance_mcp.file_text import FileText
from predictive_maintenance_mcp.signal_acquisition import (
    repository as repository_module,
)
from predictive_maintenance_mcp.signal_acquisition.import_config import samples_digest
from predictive_maintenance_mcp.signal_acquisition.repository import (
    BLANK_ROW_REMEDY,
    TEXT_TOKEN_REMEDY,
    SignalRepository,
)
from predictive_maintenance_mcp.signal_import import (
    ImportRefused,
    SignalImportService,
    validate_import,
)
from predictive_maintenance_mcp.signal_import.records import (
    ImportRecordMissing,
    ImportRecordStore,
)

RATE = 25600


def rows(count: int, channels: int = 1, sep: str = ",") -> list[str]:
    return [
        sep.join(
            [f"{row / RATE:.6f}"]
            + [f"{corpus.sample(row, c):.6f}" for c in range(channels)]
        )
        for row in range(count)
    ]


def interpretation(channels: tuple[int, ...] = (1,), **changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
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
        "channels": [
            {"column": c, "role": "vibration", "quantity": "acceleration", "unit": "g"}
            for c in channels
        ],
    }
    base.update(changes)
    return base


def configuration(
    interp: dict[str, Any], bases: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    return {
        "schema": "pmm-text-import-config/1",
        "interpretation": interp,
        "attestation": {"bases": bases or {}, "overrides": {}},
    }


def bases(channels: tuple[int, ...] = (1,)) -> dict[str, Any]:
    return {
        "time_base": {"basis": "user_stated"},
        **{f"channels[{c}].unit": {"basis": "user_stated"} for c in channels},
    }


def header(channels: tuple[int, ...]) -> str:
    return ",".join(["time [s]"] + [f"acc{c} [g]" for c in channels])


@pytest.fixture
def write(sandbox_data_dir):
    def _write(name: str, data: bytes) -> Path:
        path = sandbox_data_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    return _write


@pytest.fixture
def repo():
    return SignalRepository()


@pytest.fixture
def service(repo):
    return SignalImportService(repo)


def three_channel_file(write, name: str = "run.csv", count: int = 400) -> Path:
    columns = (1, 2, 3)
    return write(
        name, corpus.lines_to_bytes([header(columns)] + rows(count, channels=3))
    )


def plain_file(write, name: str = "other.csv") -> Path:
    """A headerless one-column file load_signal reads."""
    return write(
        name, corpus.lines_to_bytes([f"{corpus.sample(r, 5):.6f}" for r in range(300)])
    )


def ready(
    path: Path,
    conf: dict[str, Any],
    channel: str = "python_api",
    transport: Optional[str] = None,
):
    result = validate_import(str(path), conf, channel=channel, transport=transport)
    assert result.outcome == "ready", [i.message for i in result.issues]
    return result


def tree(*roots: Path) -> dict[str, bytes]:
    snapshot = {}
    for root in roots:
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    snapshot[str(path)] = path.read_bytes()
    return snapshot


def dirs() -> tuple[Path, Path]:
    return config.get_import_dir(), config.get_ledger_dir()


class TestImport:
    def test_three_channels_inserted_under_column_ids(self, write, repo, service):
        path = three_channel_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        outcome = service.import_(
            str(path), conf, result.import_digest, channel="python_api", transport=None
        )
        assert outcome.status == "imported"
        assert outcome.record_created is True
        assert [c.signal_id for c in outcome.channels] == ["run_c1", "run_c2", "run_c3"]
        assert all(c.materialized for c in outcome.channels)
        for column in (1, 2, 3):
            array = repo.get_signal(f"run_c{column}")
            expected = [corpus.sample(row, column - 1) for row in range(400)]
            assert np.allclose(array, expected, atol=5e-7)
            info = repo.get_signal_info(f"run_c{column}")
            assert info["sampling_rate"] == 25600.0
            assert info["signal_unit"] == "g"
            assert info["source_metadata"] == {}
            assert info["raw_format"] is None
            assert info["import_provenance"]["import_id"] == result.import_id
            assert info["import_provenance"]["column"] == column
            assert info["import_provenance"]["samples_digest"] == samples_digest(array)

    def test_record_holds_interpretation_attestation_location_outcome(
        self, write, service
    ):
        path = three_channel_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        service.import_(
            str(path), conf, result.import_digest, channel="python_api", transport=None
        )
        record = ImportRecordStore(config.get_import_dir()).read(result.import_id)
        kinds = [event.kind for event in record.events]
        assert kinds == ["attestation", "location", "runtime", "outcome"]
        attestation = record.events[0].payload
        assert attestation["confirmation_channel"] == "python_api"
        assert attestation["import_digest"] == result.import_digest
        outcome = record.events[-1].payload
        assert [c["signal_id"] for c in outcome["channels"]] == [
            "run_c1",
            "run_c2",
            "run_c3",
        ]
        assert [c["samples_digest"] for c in outcome["channels"]] == [
            c.samples_digest for c in result.channels
        ]

    def test_bad_channel_inserts_nothing_and_writes_no_record(
        self, write, repo, service
    ):
        lines = [header((1, 2, 3))] + rows(400, channels=3)
        lines[200] = lines[200].rsplit(",", 1)[0] + ",nan"
        path = write("bad.csv", corpus.lines_to_bytes(lines))
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        before = tree(*dirs())
        with pytest.raises(ImportRefused) as refused:
            service.import_(
                str(path), conf, "a" * 64, channel="python_api", transport=None
            )
        assert refused.value.code == "validate_again"
        assert "not ready to import" in str(refused.value)
        assert repo.signal_count == 0
        assert tree(*dirs()) == before

    def test_digest_mismatch_after_config_edit_is_validate_again(
        self, write, repo, service
    ):
        path = three_channel_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        edited = configuration(
            interpretation(
                (1, 2, 3), time_base={"source": "stated", "sampling_rate_hz": "25000"}
            ),
            bases((1, 2, 3)),
        )
        before = tree(*dirs())
        with pytest.raises(ImportRefused) as refused:
            service.import_(
                str(path),
                edited,
                result.import_digest,
                channel="python_api",
                transport=None,
            )
        assert refused.value.code == "validate_again"
        assert "validate the configuration again" in str(refused.value)
        assert repo.signal_count == 0
        assert tree(*dirs()) == before

    def test_one_byte_changed_after_validation_is_validate_again(
        self, write, repo, service
    ):
        path = three_channel_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        data = bytearray(path.read_bytes())
        index = data.rindex(b"1")
        data[index : index + 1] = b"2"
        path.write_bytes(bytes(data))
        before = tree(*dirs())
        with pytest.raises(ImportRefused) as refused:
            service.import_(
                str(path),
                conf,
                result.import_digest,
                channel="python_api",
                transport=None,
            )
        assert refused.value.code == "validate_again"
        assert "export in progress" in str(refused.value)
        assert repo.signal_count == 0
        assert tree(*dirs()) == before

    def test_digest_from_another_channel_is_validate_again(self, write, service):
        path = three_channel_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf, channel="python_api")
        with pytest.raises(ImportRefused) as refused:
            service.import_(
                str(path),
                conf,
                result.import_digest,
                channel="mcp_tool_argument",
                transport="stdio",
            )
        assert refused.value.code == "validate_again"

    def test_malformed_digest_refused_before_reading(self, write, service, monkeypatch):
        path = three_channel_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        with pytest.raises(ImportRefused) as refused:
            service.import_(
                str(path), conf, "XYZ", channel="python_api", transport=None
            )
        assert refused.value.code == "import_digest_malformed"

    def test_collision_refused_before_the_record_is_written(self, write, repo, service):
        path = three_channel_file(write)
        other = plain_file(write)
        repo.load_signal(str(other), signal_id="run_c2", sampling_rate=25600)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        before = tree(*dirs())
        with pytest.raises(ImportRefused) as refused:
            service.import_(
                str(path),
                conf,
                result.import_digest,
                channel="python_api",
                transport=None,
            )
        assert refused.value.code == "signal_id_taken"
        assert "run_c2" in str(refused.value)
        assert tree(*dirs()) == before
        assert repo.signal_count == 1

    def test_overwrite_replaces_and_refused_overwrite_keeps(self, write, repo, service):
        path = three_channel_file(write)
        other = plain_file(write)
        repo.load_signal(str(other), signal_id="run_c2", sampling_rate=25600)
        kept = repo.get_signal("run_c2").copy()
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        with pytest.raises(ImportRefused):
            service.import_(
                str(path),
                conf,
                result.import_digest,
                channel="python_api",
                transport=None,
            )
        assert np.array_equal(repo.get_signal("run_c2"), kept)
        outcome = service.import_(
            str(path),
            conf,
            result.import_digest,
            channel="python_api",
            transport=None,
            overwrite=True,
        )
        assert all(c.materialized for c in outcome.channels)
        assert repo.get_signal_info("run_c2")["import_provenance"]["column"] == 2

    def test_repeat_returns_already_imported_with_same_ids(self, write, repo, service):
        path = three_channel_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        first = service.import_(
            str(path), conf, result.import_digest, channel="python_api", transport=None
        )
        record_before = tree(config.get_import_dir())
        again = service.import_(
            str(path),
            conf,
            result.import_digest,
            channel="python_api",
            transport=None,
            signal_id_prefix="other",
        )
        assert again.status == "already_imported"
        assert again.record_created is False
        assert [c.signal_id for c in again.channels] == [
            c.signal_id for c in first.channels
        ]
        assert any("keep the signal ids" in notice for notice in again.notices)
        assert tree(config.get_import_dir()) == record_before

    def test_repeat_rematerializes_an_evicted_channel(self, write, repo, service):
        path = three_channel_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        service.import_(
            str(path), conf, result.import_digest, channel="python_api", transport=None
        )
        repo.clear_signal("run_c2")
        again = service.import_(
            str(path), conf, result.import_digest, channel="python_api", transport=None
        )
        assert again.status == "already_imported"
        assert repo.get_signal_info("run_c2")["import_provenance"]["column"] == 2

    def test_concurrent_load_after_the_record_reports_not_materialized(
        self, write, repo, service, monkeypatch
    ):
        path = three_channel_file(write)
        other = plain_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        original = SignalRepository.insert_decoded

        def racing(self, channels, *, overwrite=False):
            # Another load takes one id between the check and the insert.
            self.load_signal(str(other), signal_id="run_c3", sampling_rate=25600)
            return original(self, channels, overwrite=overwrite)

        monkeypatch.setattr(SignalRepository, "insert_decoded", racing)
        outcome = service.import_(
            str(path), conf, result.import_digest, channel="python_api", transport=None
        )
        assert not any(c.materialized for c in outcome.channels)
        assert all(c.signal_id is None for c in outcome.channels)
        assert outcome.notices
        record = ImportRecordStore(config.get_import_dir()).read(result.import_id)
        last = record.events[-1].payload
        assert all(c["signal_id"] is None for c in last["channels"])
        assert repo.get_signal_info("run_c3")["import_provenance"] is None
        assert "run_c1" not in [s["signal_id"] for s in repo.list_signals()]

    def test_prefix_forms_ids_and_path_that_cannot_form_an_id_needs_one(
        self, write, service
    ):
        path = write(
            "pump export.csv", corpus.lines_to_bytes([header((1,))] + rows(300))
        )
        conf = configuration(interpretation(), bases())
        result = ready(path, conf)
        with pytest.raises(ImportRefused) as refused:
            service.import_(
                str(path),
                conf,
                result.import_digest,
                channel="python_api",
                transport=None,
            )
        assert refused.value.code == "signal_id_invalid"
        assert "pump export" not in str(refused.value)
        outcome = service.import_(
            str(path),
            conf,
            result.import_digest,
            channel="python_api",
            transport=None,
            signal_id_prefix="pump",
        )
        assert outcome.channels[0].signal_id == "pump_c1"

    def test_malformed_prefix_refused(self, write, service):
        path = three_channel_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        with pytest.raises(ImportRefused) as refused:
            service.import_(
                str(path),
                conf,
                "a" * 64,
                channel="python_api",
                transport=None,
                signal_id_prefix="../x",
            )
        assert refused.value.code == "signal_id_invalid"

    def test_subfolder_ids_and_import_root_ids(
        self, write, service, tmp_path, monkeypatch
    ):
        nested = write(
            "site/a/run.csv", corpus.lines_to_bytes([header((1,))] + rows(300))
        )
        conf = configuration(interpretation(), bases())
        result = ready(nested, conf)
        outcome = service.import_(
            str(nested),
            conf,
            result.import_digest,
            channel="python_api",
            transport=None,
        )
        assert outcome.channels[0].signal_id == "site_a_run_c1"

        root = tmp_path / "exports"
        (root / "line2").mkdir(parents=True)
        outside = root / "line2" / "run.csv"
        # Other bytes: the same bytes would be the same import, whatever
        # their location.
        outside.write_bytes(corpus.lines_to_bytes([header((1,))] + rows(301)))
        monkeypatch.setenv("PMM_IMPORT_ROOTS", str(root))
        result = ready(outside, conf)
        outcome = service.import_(
            str(outside),
            conf,
            result.import_digest,
            channel="python_api",
            transport=None,
        )
        assert outcome.channels[0].signal_id == "exports_line2_run_c1"


class TestProvenance:
    def test_caller_asserted_bases_and_escaped_label(self, write, repo, service):
        path = write(
            "run.csv",
            corpus.lines_to_bytes(["time [s],accé [g]"] + rows(300)),
        )
        conf = configuration(interpretation(), bases())
        result = ready(path, conf, channel="mcp_tool_argument", transport="stdio")
        service.import_(
            str(path),
            conf,
            result.import_digest,
            channel="mcp_tool_argument",
            transport="stdio",
        )
        provenance = repo.get_signal_info("run_c1")["import_provenance"]
        assert provenance["caller_asserted"] is True
        assert provenance["confirmation_channel"] == "mcp_tool_argument"
        assert provenance["transport"] == "stdio"
        assert provenance["bases"] == {
            "channels[1].unit": "user_stated",
            "time_base": "user_stated",
        }
        # Marked file text: the escaped form FileText gives, never a bare
        # string.
        assert provenance["label"] == FileText.of("accé [g]", cap=120).as_dict()
        assert provenance["time_base_source"] == "stated"
        assert provenance["sampling_rate_hz"] == "25600"
        assert provenance["scaling"] is None
        assert provenance["decoder_contract"] == "pmm-text-decoder/1"

    def test_load_signal_info_carries_null_provenance(self, write, repo):
        info = repo.load_signal(str(plain_file(write)), sampling_rate=25600)
        assert info["import_provenance"] is None

    def test_not_found_message_names_import_rematerialization(self, repo):
        with pytest.raises(KeyError) as missing:
            repo.get_signal("nowhere")
        text = str(missing.value)
        assert "import_signal_file(import_id=...)" in text
        assert "get_signal_imports" in text


class TestRematerialize:
    def _imported(self, write, service, **kwargs):
        path = three_channel_file(write, **kwargs)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        outcome = service.import_(
            str(path), conf, result.import_digest, channel="python_api", transport=None
        )
        return path, outcome

    def test_new_repository_restores_without_confirmations(self, write, service):
        path, outcome = self._imported(write, service)
        fresh = SignalRepository()
        restored = SignalImportService(fresh).rematerialize(outcome.import_id)
        assert restored.status == "rematerialized"
        assert [c.signal_id for c in restored.channels] == [
            "run_c1",
            "run_c2",
            "run_c3",
        ]
        for channel, original in zip(restored.channels, outcome.channels):
            assert channel.samples_digest == original.samples_digest
            assert (
                samples_digest(fresh.get_signal(channel.signal_id))
                == original.samples_digest
            )
            info = fresh.get_signal_info(channel.signal_id)
            assert info["import_provenance"]["import_id"] == outcome.import_id

    def test_moved_file_refused_naming_recorded_locations(self, write, service):
        path, outcome = self._imported(write, service)
        path.rename(path.with_name("moved.csv"))
        with pytest.raises(ImportRefused) as refused:
            SignalImportService(SignalRepository()).rematerialize(outcome.import_id)
        assert refused.value.code == "source_unavailable"
        assert "run.csv" in str(refused.value)
        assert "file_not_found" in str(refused.value)

    def test_changed_bytes_refused_before_decoding(self, write, service, monkeypatch):
        path, outcome = self._imported(write, service)
        data = bytearray(path.read_bytes())
        data[-3:-2] = b"9" if data[-3:-2] != b"9" else b"8"
        path.write_bytes(bytes(data))

        def never(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("decoded a file whose bytes differ")

        monkeypatch.setattr(
            "predictive_maintenance_mcp.signal_import.service.get_decoder",
            lambda contract: never,
        )
        with pytest.raises(ImportRefused) as refused:
            SignalImportService(SignalRepository()).rematerialize(outcome.import_id)
        assert refused.value.code == "source_unavailable"
        assert "other bytes" in str(refused.value)

    def test_location_outside_current_roots_refused_without_stat(
        self, write, service, tmp_path, monkeypatch
    ):
        root = tmp_path / "exports"
        root.mkdir()
        outside = root / "run.csv"
        outside.write_bytes(corpus.lines_to_bytes([header((1,))] + rows(300)))
        monkeypatch.setenv("PMM_IMPORT_ROOTS", str(root))
        conf = configuration(interpretation(), bases())
        result = ready(outside, conf)
        outcome = service.import_(
            str(outside),
            conf,
            result.import_digest,
            channel="python_api",
            transport=None,
        )
        monkeypatch.delenv("PMM_IMPORT_ROOTS")
        calls: list[str] = []
        real_stat = os.stat

        def watched(target, *args, **kwargs):
            if "exports" in str(target):
                calls.append(str(target))
            return real_stat(target, *args, **kwargs)

        monkeypatch.setattr(os, "stat", watched)
        with pytest.raises(ImportRefused) as refused:
            SignalImportService(SignalRepository()).rematerialize(outcome.import_id)
        assert "path_outside_roots" in str(refused.value)
        assert calls == []

    def test_record_directory_deleted_refuses_naming_the_import(self, write, service):
        import shutil

        path, outcome = self._imported(write, service)
        shutil.rmtree(config.get_import_dir())
        with pytest.raises(ImportRecordMissing) as missing:
            SignalImportService(SignalRepository()).rematerialize(outcome.import_id)
        assert outcome.import_id[:12] in str(missing.value)

    def test_malformed_import_id_refused(self, service):
        with pytest.raises(ImportRefused) as refused:
            service.rematerialize("ABC")
        assert refused.value.code == "import_id_malformed"


class TestLoadSignalRemedies:
    def test_headered_csv_names_inspect_signal_file(self, write, repo):
        values = [f"{corpus.sample(row, 0):.6f}" for row in range(300)]
        path = write("headered.csv", corpus.lines_to_bytes(["acc [g]"] + values))
        with pytest.raises(ValueError) as refused:
            repo.load_signal(str(path), sampling_rate=25600)
        assert "inspect_signal_file" in str(refused.value)
        assert TEXT_TOKEN_REMEDY in str(refused.value)

    def test_nan_sample_keeps_the_shared_remedy(self, write, repo):
        lines = [f"{corpus.sample(row, 0):.6f}" for row in range(300)]
        lines[10] = "nan"
        path = write("nan.csv", corpus.lines_to_bytes(lines))
        with pytest.raises(ValueError) as refused:
            repo.load_signal(str(path), sampling_rate=25600)
        assert "inspect_signal_file" not in str(refused.value)

    def test_blank_line_after_a_preamble_names_inspect_signal_file(self, write, repo):
        lines = ["0.1", "", *[f"{corpus.sample(row, 0):.6f}" for row in range(300)]]
        path = write("pre.csv", corpus.lines_to_bytes(lines))
        with pytest.raises(ValueError) as refused:
            repo.load_signal(str(path), sampling_rate=25600)
        assert "inspect_signal_file" in str(refused.value)
        assert BLANK_ROW_REMEDY in str(refused.value)


def test_repository_module_keeps_one_freeze_path():
    """Both routes share the sample check and freeze helper."""
    source = Path(repository_module.__file__).read_text(encoding="utf-8")
    assert source.count("setflags(write=False)") == 1


class TestRematerializeRefusals:
    def _imported(self, write, service, name="run.csv"):
        path = three_channel_file(write, name=name)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        outcome = service.import_(
            str(path), conf, result.import_digest, channel="python_api", transport=None
        )
        return path, outcome

    def test_a_digest_that_no_longer_reproduces_stores_nothing(
        self, write, service, monkeypatch
    ):
        from predictive_maintenance_mcp.signal_acquisition.text_decoder import (
            decode_text,
        )
        from predictive_maintenance_mcp.signal_import import service as service_module

        path, outcome = self._imported(write, service)

        def drifting(file, interpretation, mode="full"):
            result = decode_text(file, interpretation, mode=mode)
            first = result.channels[0]
            changed = type(first)(
                column=first.column,
                samples=first.samples + 1.0,
                samples_digest="0" * 64,
                sample_count=first.sample_count,
            )
            return type(result)(
                **{**result.__dict__, "channels": (changed, *result.channels[1:])}
            )

        monkeypatch.setattr(service_module, "get_decoder", lambda contract: drifting)
        fresh = SignalRepository()
        with pytest.raises(ImportRefused) as refused:
            SignalImportService(fresh).rematerialize(outcome.import_id)
        assert refused.value.code == "determinism_failure"
        assert fresh.signal_count == 0

    def test_a_record_that_stored_nothing_cannot_be_restored(
        self, write, repo, service, monkeypatch
    ):
        path = three_channel_file(write)
        other = plain_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        original = SignalRepository.insert_decoded

        def racing(self, channels, *, overwrite=False):
            self.load_signal(str(other), signal_id="run_c1", sampling_rate=25600)
            return original(self, channels, overwrite=overwrite)

        monkeypatch.setattr(SignalRepository, "insert_decoded", racing)
        outcome = service.import_(
            str(path), conf, result.import_digest, channel="python_api", transport=None
        )
        with pytest.raises(ImportRefused) as refused:
            SignalImportService(SignalRepository()).rematerialize(outcome.import_id)
        assert refused.value.code == "record_incomplete"

    def test_an_id_held_by_another_signal_needs_overwrite(self, write, service):
        path, outcome = self._imported(write, service)
        fresh = SignalRepository()
        fresh.load_signal(
            str(plain_file(write)), signal_id="run_c2", sampling_rate=25600
        )
        with pytest.raises(ImportRefused) as refused:
            SignalImportService(fresh).rematerialize(outcome.import_id)
        assert refused.value.code == "signal_id_taken"
        assert fresh.signal_count == 1
        restored = SignalImportService(fresh).rematerialize(
            outcome.import_id, overwrite=True
        )
        assert all(c.materialized for c in restored.channels)

    def test_an_earlier_location_is_tried_when_the_latest_is_gone(self, write, service):
        path, outcome = self._imported(write, service)
        copy = path.with_name("copy.csv")
        copy.write_bytes(path.read_bytes())
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        again = ready(copy, conf)
        service.import_(
            str(copy), conf, again.import_digest, channel="python_api", transport=None
        )
        copy.unlink()
        restored = SignalImportService(SignalRepository()).rematerialize(
            outcome.import_id
        )
        assert restored.status == "rematerialized"
        assert restored.file.file_text == "run.csv"


class TestCollisionsAndRepeats:
    def test_a_concurrent_repeat_of_the_same_channel_counts_as_stored(
        self, write, repo, service, monkeypatch
    ):
        path = three_channel_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        original = SignalRepository.insert_decoded
        calls = {"n": 0}

        def twin(self, channels, *, overwrite=False):
            calls["n"] += 1
            if calls["n"] == 1:
                # A concurrent repeat of this import stores one channel first.
                original(self, [channels[0]], overwrite=False)
            return original(self, channels, overwrite=overwrite)

        monkeypatch.setattr(SignalRepository, "insert_decoded", twin)
        outcome = service.import_(
            str(path), conf, result.import_digest, channel="python_api", transport=None
        )
        assert all(c.materialized for c in outcome.channels)
        assert calls["n"] == 2
        assert sorted(i["signal_id"] for i in repo.list_signals()) == [
            "run_c1",
            "run_c2",
            "run_c3",
        ]

    def test_a_repeat_whose_first_ids_are_taken_offers_only_overwrite(
        self, write, repo, service
    ):
        path = three_channel_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        service.import_(
            str(path), conf, result.import_digest, channel="python_api", transport=None
        )
        repo.clear_signal("run_c2")
        repo.load_signal(
            str(plain_file(write)), signal_id="run_c2", sampling_rate=25600
        )
        with pytest.raises(ImportRefused) as refused:
            service.import_(
                str(path),
                conf,
                result.import_digest,
                channel="python_api",
                transport=None,
                signal_id_prefix="other",
            )
        assert refused.value.code == "signal_id_taken"
        assert "signal_id_prefix" not in str(refused.value)
        assert "overwrite=True" in str(refused.value)


class TestServiceEdges:
    def test_two_channels_with_one_id_are_refused(self, write, repo):
        from predictive_maintenance_mcp.signal_acquisition.repository import (
            DecodedChannel,
        )

        channel = DecodedChannel(
            signal_id="same",
            filepath=Path("x.csv"),
            samples=np.zeros(300),
            sampling_rate=25600.0,
            signal_unit="g",
            measurement=None,
            import_provenance={"column": 1},
        )
        with pytest.raises(ValueError) as refused:
            repo.insert_decoded([channel, channel])
        assert "share the signal_id" in str(refused.value)
        assert repo.signal_count == 0

    def test_injected_stores_are_used(self, write, repo, tmp_path):
        from predictive_maintenance_mcp.asset_ledger.store import LedgerStore

        records = ImportRecordStore(tmp_path / "own_imports")
        ledger = LedgerStore(tmp_path / "own_ledger")
        service = SignalImportService(repo, records=records, ledger=ledger)
        assert service.records is records and service.ledger is ledger
        path = three_channel_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        outcome = service.import_(
            str(path), conf, result.import_digest, channel="python_api", transport=None
        )
        assert records.read(outcome.import_id).import_id == outcome.import_id

    def test_a_grown_file_is_refused_by_size(self, write, service):
        path = three_channel_file(write)
        conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
        result = ready(path, conf)
        outcome = service.import_(
            str(path), conf, result.import_digest, channel="python_api", transport=None
        )
        path.write_bytes(path.read_bytes() + b"0.1,0.1,0.1,0.1\n")
        with pytest.raises(ImportRefused) as refused:
            SignalImportService(SignalRepository()).rematerialize(outcome.import_id)
        assert "has another size" in str(refused.value)

    def test_an_unreadable_ledger_is_reported_when_restoring(
        self, write, service, monkeypatch
    ):
        from predictive_maintenance_mcp.asset_ledger.store import (
            LedgerError,
            LedgerStore,
        )

        path = write("bound.csv", corpus.lines_to_bytes([header((1,))] + rows(300)))
        interp = interpretation(
            (1,),
            defaults={"asset_id": "P-101", "acquired_at": "2026-08-20T11:42:00+00:00"},
        )
        interp["channels"][0]["binding"] = {
            "measurement_point_id": "pt",
            "direction": "horizontal",
        }
        conf = configuration(
            interp,
            {
                **bases(),
                "defaults.acquired_at": {"basis": "user_stated"},
                "channels[1].binding": {"basis": "user_stated"},
            },
        )
        result = ready(path, conf)
        outcome = service.import_(
            str(path), conf, result.import_digest, channel="python_api", transport=None
        )

        def unreadable(self, asset_id):
            raise LedgerError("ledger unreadable")

        monkeypatch.setattr(LedgerStore, "read_view", unreadable)
        restored = SignalImportService(SignalRepository()).rematerialize(
            outcome.import_id
        )
        assert any("could not be read" in notice for notice in restored.notices)

    def test_an_uninstalled_package_records_an_unknown_version(self, monkeypatch):
        from importlib import metadata

        from predictive_maintenance_mcp.signal_import import service as service_module

        def missing(name):
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(service_module.metadata, "version", missing)
        assert service_module._package_version() == "unknown"


def test_an_attestation_edited_after_validation_is_validate_again(write, repo, service):
    """The digest binds the attestation too: a changed declared_by keeps the
    configuration ready but is refused."""
    path = three_channel_file(write)
    conf = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
    result = ready(path, conf)
    edited = configuration(interpretation((1, 2, 3)), bases((1, 2, 3)))
    edited["attestation"]["declared_by"] = "someone else"
    assert ready(path, edited).import_digest != result.import_digest
    with pytest.raises(ImportRefused) as refused:
        service.import_(
            str(path),
            edited,
            result.import_digest,
            channel="python_api",
            transport=None,
        )
    assert refused.value.code == "validate_again"
    assert repo.signal_count == 0
