"""Tests for ``signal_import.records``: the append-only import record store.

Every property about bytes on disk is checked against real files under
``tmp_path``; concurrency runs with real threads and real child processes
(the same matrix as the ledger store, Windows included).
"""

import ast
import errno
import inspect
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from conftest import REPO_ROOT, SUBPROCESS_PIN

from predictive_maintenance_mcp import config
from predictive_maintenance_mcp.asset_ledger.store import (
    LedgerLockTimeout,
    acquire_lock,
    release_lock,
)
from predictive_maintenance_mcp.signal_acquisition.import_config import (
    import_id as compute_import_id,
)
from predictive_maintenance_mcp.signal_import import records as records_module
from predictive_maintenance_mcp.signal_import.records import (
    RECORD_SCHEMA,
    RECORD_WORDING,
    ImportRecordError,
    ImportRecordStore,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SIZE = 4096
CHILD_TIMEOUT = 300.0


def interpretation(*columns: int, rate: str = "25600") -> dict:
    """A valid interpretation importing *columns* (default: column 1)."""
    return {
        "schema": "pmm-text-import/1",
        "dialect": {"encoding": "utf-8", "delimiter": ";", "decimal": ","},
        "data_lines": {"first": 9},
        "time_base": {"source": "stated", "sampling_rate_hz": rate},
        "channels": [
            {
                "column": column,
                "role": "vibration",
                "quantity": "acceleration",
                "unit": "g",
            }
            for column in (columns or (1,))
        ],
    }


@pytest.fixture
def store(tmp_path) -> ImportRecordStore:
    return ImportRecordStore(tmp_path / "imports")


@pytest.fixture
def clock(monkeypatch):
    """Deterministic, strictly increasing ``recorded_at`` values."""
    ticks = iter(range(10, 60))

    def now() -> str:
        return f"2026-11-05T10:00:{next(ticks):02d}.000000+00:00"

    monkeypatch.setattr(records_module, "utc_now_iso", now)


def record_lines(store: ImportRecordStore, import_id: str) -> list[bytes]:
    return (store.records_dir / f"{import_id}.jsonl").read_bytes().split(b"\n")


class TestInterpretationLine:
    def test_first_write_creates_line_one_with_only_identity_content(self, store):
        written = store.write_interpretation(SHA_A, SIZE, interpretation())
        assert written.created
        assert written.import_id == compute_import_id(SHA_A, SIZE, interpretation())
        lines = record_lines(store, written.import_id)
        assert lines[-1] == b""  # terminated
        first = json.loads(lines[0])
        assert set(first) == {"schema", "kind", "import_id", "source", "interpretation"}
        assert first["schema"] == RECORD_SCHEMA
        assert first["source"] == {"sha256": SHA_A, "size_bytes": SIZE}
        assert "recorded_at" not in first

    def test_same_interpretation_twice_keeps_one_line(self, store):
        first = store.write_interpretation(SHA_A, SIZE, interpretation())
        again = store.write_interpretation(SHA_A, SIZE, interpretation())
        assert (first.created, again.created) == (True, False)
        assert first.import_id == again.import_id
        assert len(record_lines(store, first.import_id)) == 2  # one line + b""

    def test_second_root_appends_a_location_instead_of_failing(self, store, clock):
        rid = store.write_interpretation(SHA_A, SIZE, interpretation()).import_id
        store.append_event(rid, "location", {"location": r"C:\exports\run.csv"})
        again = store.write_interpretation(SHA_A, SIZE, interpretation())
        assert not again.created
        store.append_event(rid, "location", {"location": "/mnt/b/run.csv"})
        record = store.read(rid)
        assert [e.payload["location"] for e in record.events] == [
            r"C:\exports\run.csv",
            "/mnt/b/run.csv",
        ]

    def test_second_attestation_is_appended(self, store, clock):
        rid = store.write_interpretation(SHA_A, SIZE, interpretation()).import_id
        store.append_event(rid, "attestation", {"channel": "python_api", "n": 1})
        store.append_event(rid, "attestation", {"channel": "mcp_tool_argument", "n": 2})
        record = store.read(rid)
        assert [(e.line, e.kind, e.payload["n"]) for e in record.events] == [
            (2, "attestation", 1),
            (3, "attestation", 2),
        ]
        assert record.unreadable_lines == ()

    def test_event_on_a_missing_record_is_refused_and_writes_nothing(self, store):
        rid = compute_import_id(SHA_A, SIZE, interpretation())
        store.records_dir.mkdir(parents=True)
        with pytest.raises(
            ImportRecordError, match="interpretation line is written first"
        ):
            store.append_event(rid, "attestation", {})
        assert not (store.records_dir / f"{rid}.jsonl").exists()

    def test_event_with_no_records_directory_creates_nothing(self, store):
        rid = compute_import_id(SHA_A, SIZE, interpretation())
        with pytest.raises(
            ImportRecordError, match="interpretation line is written first"
        ):
            store.append_event(rid, "attestation", {})
        assert not store.root.exists()

    def test_damaged_record_names_the_file_to_move_aside(self, store):
        rid = store.write_interpretation(SHA_A, SIZE, interpretation()).import_id
        path = store.records_dir / f"{rid}.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"25600"', b'"25000"'))
        with pytest.raises(ImportRecordError) as caught:
            store.write_interpretation(SHA_A, SIZE, interpretation())
        message = str(caught.value)
        assert f"records/{rid}.jsonl" in message and "PMM_IMPORT_DIR" in message
        assert "import the file again" not in message

    def test_unknown_event_kind_is_refused(self, store):
        rid = store.write_interpretation(SHA_A, SIZE, interpretation()).import_id
        with pytest.raises(ValueError, match="kind must be one of"):
            store.append_event(rid, "comment", {})


class TestCrashBetweenTemporaryWriteAndLink:
    def test_interrupted_link_leaves_no_line_one_and_the_retry_succeeds(
        self, store, monkeypatch
    ):
        rid = compute_import_id(SHA_A, SIZE, interpretation())

        def crash(*_args):
            raise RuntimeError("simulated crash before the link")

        monkeypatch.setattr(records_module.os, "link", crash)
        with pytest.raises(RuntimeError, match="simulated crash"):
            store.write_interpretation(SHA_A, SIZE, interpretation())
        assert not (store.records_dir / f"{rid}.jsonl").exists()
        monkeypatch.undo()
        written = store.write_interpretation(SHA_A, SIZE, interpretation())
        assert written.created and written.import_id == rid
        assert json.loads(record_lines(store, rid)[0])["import_id"] == rid

    def test_a_stray_temporary_file_from_a_dead_writer_stays_untouched(self, store):
        store.records_dir.mkdir(parents=True)
        stray = store.records_dir / ".tmp-deadwriter-1"
        stray.write_bytes(b"half a line")
        store.write_interpretation(SHA_A, SIZE, interpretation())
        assert stray.read_bytes() == b"half a line"
        leftovers = [
            p.name for p in store.records_dir.iterdir() if p.name.startswith(".tmp-")
        ]
        assert leftovers == [stray.name]


class TestReading:
    def test_edited_line_one_is_refused(self, store):
        rid = store.write_interpretation(SHA_A, SIZE, interpretation()).import_id
        path = store.records_dir / f"{rid}.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"25600"', b'"25000"'))
        with pytest.raises(ImportRecordError, match="does not match its import_id"):
            store.read(rid)
        with pytest.raises(ImportRecordError, match="does not match its import_id"):
            store.write_interpretation(SHA_A, SIZE, interpretation())

    def test_torn_last_line_is_one_unreadable_line_and_the_rest_reads(
        self, store, clock
    ):
        rid = store.write_interpretation(SHA_A, SIZE, interpretation()).import_id
        store.append_event(rid, "attestation", {"n": 1})
        path = store.records_dir / f"{rid}.jsonl"
        with open(path, "ab") as handle:
            handle.write(b'{"schema":"pmm-import-record/1","kind":"outc')
        record = store.read(rid)
        assert [e.payload for e in record.events] == [{"n": 1}]
        assert record.unreadable_lines == (3,)
        # The next append isolates the torn bytes on their own line.
        store.append_event(rid, "attestation", {"n": 2})
        record = store.read(rid)
        assert [e.payload for e in record.events] == [{"n": 1}, {"n": 2}]
        assert record.unreadable_lines == (3,)

    def test_missing_record_is_refused_with_a_remedy(self, store):
        rid = compute_import_id(SHA_A, SIZE, interpretation())
        with pytest.raises(ImportRecordError, match="List the records"):
            store.read(rid)

    @pytest.mark.parametrize(
        "bad", ["../ledger/x", "A" * 64, "a" * 63, "a" * 65, "", None, 7]
    )
    def test_malformed_import_id_is_refused_before_any_path_is_formed(
        self, store, monkeypatch, bad
    ):
        formed: list[object] = []
        monkeypatch.setattr(
            records_module, "safe_resolve", lambda *args: formed.append(args)
        )
        with pytest.raises(ImportRecordError, match="64 lowercase hexadecimal"):
            store.read(bad)
        with pytest.raises(ImportRecordError, match="64 lowercase hexadecimal"):
            store.append_event(bad, "attestation", {})
        assert formed == []


class TestIndex:
    def _import(self, store, sha, columns, rate, location, signals):
        rid = store.write_interpretation(
            sha, SIZE, interpretation(*columns, rate=rate)
        ).import_id
        store.append_event(rid, "attestation", {"channel": "python_api"})
        store.append_event(rid, "location", {"location": location})
        store.append_event(
            rid,
            "outcome",
            {
                "channels": [
                    {"column": c, "signal_id": s, "materialized": True}
                    for c, s in zip(columns, signals)
                ]
            },
        )
        return rid

    def test_newest_first_with_filters_and_supersedes(self, store, clock):
        first = self._import(
            store, SHA_A, (1, 2), "25600", r"C:\x\run.csv", ("run_c1", "run_c2")
        )
        other = self._import(store, SHA_B, (1,), "25600", "/y/other.txt", ("other_c1",))
        fix = self._import(store, SHA_A, (2,), "25000", r"C:\x\run.csv", ("run_c2",))

        entries, refused = store.list_records()
        assert refused == ()
        assert [e.import_id for e in entries] == [fix, other, first]
        by_id = {e.import_id: e for e in entries}
        assert by_id[fix].supersedes == (first,)
        assert by_id[first].supersedes == ()
        assert by_id[other].supersedes == ()
        assert by_id[first].file_name.file_text == "run.csv"
        assert by_id[first].columns == (1, 2)

        assert [e.import_id for e in store.list_records(signal_id="run_c2")[0]] == [
            fix,
            first,
        ]
        assert [e.import_id for e in store.list_records(file_name="other.txt")[0]] == [
            other
        ]
        assert [e.import_id for e in store.list_records(file_sha256=SHA_A)[0]] == [
            fix,
            first,
        ]

    def test_record_with_a_bad_first_line_is_left_out_and_named(self, store, clock):
        good = self._import(store, SHA_A, (1,), "25600", "a.csv", ("a_c1",))
        bad = self._import(store, SHA_B, (1,), "25600", "b.csv", ("b_c1",))
        path = store.records_dir / f"{bad}.jsonl"
        path.write_bytes(b"not json\n" + path.read_bytes())
        entries, refused = store.list_records()
        assert [e.import_id for e in entries] == [good]
        assert refused == (bad,)

    def test_listing_writes_nothing(self, store, clock):
        self._import(store, SHA_A, (1,), "25600", "a.csv", ("a_c1",))
        before = {p: p.read_bytes() for p in store.records_dir.iterdir()}
        store.list_records()
        store.list_records(signal_id="a_c1")
        assert {p: p.read_bytes() for p in store.records_dir.iterdir()} == before

    def test_empty_or_absent_directory_lists_nothing(self, store):
        assert store.list_records() == ([], ())


class TestMisshapenLaterLines:
    """A later line that parses but has the wrong shape is one unreadable
    line; it never stops the record, or the index, from being read."""

    @pytest.mark.parametrize(
        "line",
        [
            {"kind": "location", "recorded_at": "2026-11-05", "payload": {}},
            {"kind": "outcome", "payload": {"channels": None}},
            {"kind": "outcome", "payload": {"channels": 3}},
            {"kind": "outcome", "payload": {"channels": ["c1"]}},
            {"kind": "outcome", "payload": {"channels": [{"column": "1"}]}},
            {"kind": "location", "payload": {"location": 7}},
        ],
    )
    def test_misshapen_line_is_unreadable_and_the_index_still_lists(
        self, store, clock, line
    ):
        good = store.write_interpretation(SHA_A, SIZE, interpretation()).import_id
        store.append_event(good, "attestation", {"n": 1})
        other = store.write_interpretation(SHA_B, SIZE, interpretation()).import_id
        store.append_event(other, "attestation", {"n": 1})
        raw = {
            "schema": RECORD_SCHEMA,
            "import_id": other,
            "recorded_at": "2026-11-05T11:00:00.000000+00:00",
            **line,
        }
        with open(store.records_dir / f"{other}.jsonl", "ab") as handle:
            handle.write(json.dumps(raw).encode("ascii") + b"\n")
        assert store.read(other).unreadable_lines == (3,)
        entries, refused = store.list_records()
        assert sorted(e.import_id for e in entries) == sorted([good, other])
        assert refused == ()


class TestRecencyOrder:
    def test_restored_interpretation_comes_first_and_supersedes(self, store, clock):
        first = store.write_interpretation(SHA_A, SIZE, interpretation(1)).import_id
        store.append_event(first, "attestation", {"n": 1})
        second = store.write_interpretation(
            SHA_A, SIZE, interpretation(1, rate="25000")
        ).import_id
        store.append_event(second, "attestation", {"n": 2})
        again = store.write_interpretation(SHA_A, SIZE, interpretation(1))
        assert not again.created
        store.append_event(first, "attestation", {"n": 3})

        entries, _ = store.list_records()
        assert [e.import_id for e in entries] == [first, second]
        by_id = {e.import_id: e for e in entries}
        assert by_id[first].supersedes == (second,)
        assert by_id[second].supersedes == ()  # the first was used again after it
        assert by_id[first].recorded_at < by_id[first].last_recorded_at


class TestConcurrentWriters:
    def test_threads_writing_the_same_interpretation_leave_one_line(self, store):
        results: list[object] = []
        barrier = threading.Barrier(6)

        def write() -> None:
            barrier.wait()
            try:
                results.append(
                    ImportRecordStore(store.root).write_interpretation(
                        SHA_A, SIZE, interpretation()
                    )
                )
            except Exception as exc:  # noqa: BLE001 - reported below
                results.append(exc)

        threads = [threading.Thread(target=write) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert all(not isinstance(r, Exception) for r in results), results
        assert sorted(r.created for r in results) == [False] * 5 + [True]
        rid = results[0].import_id
        assert len(record_lines(store, rid)) == 2
        assert store.read(rid).events == ()

    def test_child_processes_share_one_record(self, store):
        script = SUBPROCESS_PIN + (
            "import sys\n"
            "from pathlib import Path\n"
            "from predictive_maintenance_mcp.signal_import.records import ImportRecordStore\n"
            "store = ImportRecordStore(Path(sys.argv[1]), lock_timeout=60)\n"
            "interp = {'schema': 'pmm-text-import/1',\n"
            "  'dialect': {'encoding': 'utf-8', 'delimiter': ';', 'decimal': ','},\n"
            "  'data_lines': {'first': 9},\n"
            "  'time_base': {'source': 'stated', 'sampling_rate_hz': '25600'},\n"
            "  'channels': [{'column': 1, 'role': 'vibration',\n"
            "                'quantity': 'acceleration', 'unit': 'g'}]}\n"
            f"rid = store.write_interpretation({SHA_A!r}, {SIZE}, interp).import_id\n"
            "for i in range(25):\n"
            "    store.append_event(rid, 'attestation', {'writer': sys.argv[2], 'i': i})\n"
        )
        children = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(store.root), name],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for name in ("one", "two", "three")
        ]
        for child in children:
            _, err = child.communicate(timeout=CHILD_TIMEOUT)
            assert child.returncode == 0, err.decode(errors="replace")
        rid = compute_import_id(SHA_A, SIZE, interpretation())
        record = store.read(rid)
        assert record.unreadable_lines == ()
        assert len(record.events) == 75
        assert {e.payload["writer"] for e in record.events} == {"one", "two", "three"}

    def test_lock_timeout_names_the_import_directory(self, store):
        rid = store.write_interpretation(SHA_A, SIZE, interpretation()).import_id
        _, _, lock = store._paths(rid)
        fd = acquire_lock(lock, what="holder", timeout=1, wording=RECORD_WORDING)
        try:
            with pytest.raises(LedgerLockTimeout) as caught:
                ImportRecordStore(store.root, lock_timeout=0.05).append_event(
                    rid, "attestation", {}
                )
        finally:
            release_lock(fd)
        message = str(caught.value)
        assert "import record lock" in message and "PMM_IMPORT_DIR" in message
        assert "ledger" not in message.lower()


class TestUnusableDirectory:
    def test_directory_that_cannot_be_created_refuses_with_a_remedy(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("a file, not a directory", encoding="utf-8")
        with pytest.raises(ImportRecordError, match="PMM_IMPORT_DIR"):
            ImportRecordStore(blocker / "imports").write_interpretation(
                SHA_A, SIZE, interpretation()
            )

    def test_filesystem_without_hard_links_refuses_with_a_remedy(
        self, store, monkeypatch
    ):
        def no_links(*_args):
            raise OSError(errno.EPERM, "Operation not permitted")

        monkeypatch.setattr(records_module.os, "link", no_links)
        with pytest.raises(ImportRecordError, match="hard links") as caught:
            store.write_interpretation(SHA_A, SIZE, interpretation())
        assert "PMM_IMPORT_DIR" in str(caught.value)
        assert [
            p.name for p in store.records_dir.iterdir() if not p.name.endswith(".lock")
        ] == []


class TestNoDeletion:
    def test_the_only_removal_is_the_writer_s_own_temporary_file(self):
        tree = ast.parse(inspect.getsource(records_module))
        removals = [
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr
            in {"unlink", "remove", "rmdir", "rmtree", "rename", "replace", "truncate"}
        ]
        assert removals == ["os.unlink"]
        source = inspect.getsource(records_module.ImportRecordStore._link_line_one)
        assert "os.unlink(temp)" in source


class TestConfigurationAndSandbox:
    def test_tests_never_write_under_the_checkout(self, tmp_path):
        configured = config.get_import_dir()
        assert configured == tmp_path / "imports"
        assert REPO_ROOT not in configured.parents
        assert "PMM_IMPORT_ROOTS" not in os.environ

    def test_default_is_the_project_data_directory(self, monkeypatch):
        monkeypatch.delenv("PMM_IMPORT_DIR")
        assert config.get_import_dir() == config.PROJECT_ROOT / "data" / "imports"
        monkeypatch.setenv("PMM_IMPORT_DIR", "   ")
        assert config.get_import_dir() == config.PROJECT_ROOT / "data" / "imports"


@pytest.fixture
def sandbox_server(tmp_path, monkeypatch):
    import predictive_maintenance_mcp.server as srv

    resources = tmp_path / "resources"
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path / "data" / "signals")
    monkeypatch.setattr(srv, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(srv, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(srv, "RESOURCES_DIR", resources)
    monkeypatch.setattr(srv, "CACHE_DIR", resources / "cache")
    for var in srv.SYNC_ROOT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return srv


def _import_warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and (
            "PMM_IMPORT_DIR" in record.getMessage()
            or "PMM_IMPORT_ROOTS" in record.getMessage()
        )
    ]


class TestStartupWarnings:
    def test_local_writable_directory_is_silent(
        self, sandbox_server, package_caplog, import_dir
    ):
        sandbox_server._setup_environment()
        assert import_dir.is_dir()
        assert _import_warnings(package_caplog) == []

    def test_not_writable(self, sandbox_server, monkeypatch, package_caplog):
        monkeypatch.setattr(sandbox_server, "_ledger_dir_writable", lambda path: False)
        sandbox_server._setup_environment()
        warnings = _import_warnings(package_caplog)
        assert len(warnings) == 1 and "is not writable" in warnings[0]

    def test_no_hard_links(self, sandbox_server, monkeypatch, package_caplog):
        monkeypatch.setattr(sandbox_server, "_supports_hard_links", lambda path: False)
        sandbox_server._setup_environment()
        warnings = _import_warnings(package_caplog)
        assert len(warnings) == 1 and "without hard links" in warnings[0]

    def test_hard_link_probe_is_real_and_leaves_nothing(self, sandbox_server, tmp_path):
        assert sandbox_server._supports_hard_links(tmp_path) is True
        assert list(tmp_path.iterdir()) == []

    def test_cloud_synced(self, sandbox_server, monkeypatch, tmp_path, package_caplog):
        monkeypatch.setenv("OneDrive", str(tmp_path))
        sandbox_server._setup_environment()
        warnings = _import_warnings(package_caplog)
        assert len(warnings) == 1 and "cloud-synced" in warnings[0]

    def test_whole_drive_and_home_folder_roots(
        self, sandbox_server, monkeypatch, package_caplog, tmp_path
    ):
        drive = Path(tmp_path.anchor)
        monkeypatch.setenv(
            "PMM_IMPORT_ROOTS", os.pathsep.join([str(drive), str(Path.home())])
        )
        sandbox_server._setup_environment()
        warnings = _import_warnings(package_caplog)
        assert len(warnings) == 2
        assert "is a whole drive" in warnings[0]
        assert "is the home folder" in warnings[1]

    def test_ordinary_export_root_is_silent(
        self, sandbox_server, monkeypatch, package_caplog, tmp_path
    ):
        monkeypatch.setenv("PMM_IMPORT_ROOTS", str(tmp_path / "exports"))
        sandbox_server._setup_environment()
        assert _import_warnings(package_caplog) == []

    def test_no_home_directory_does_not_stop_startup(
        self, sandbox_server, monkeypatch, package_caplog, tmp_path
    ):
        def no_home():
            raise RuntimeError("Could not determine home directory.")

        monkeypatch.setattr(sandbox_server.Path, "home", staticmethod(no_home))
        monkeypatch.setenv("PMM_IMPORT_ROOTS", str(Path(tmp_path.anchor)))
        sandbox_server._setup_environment()  # must not raise
        warnings = _import_warnings(package_caplog)
        assert len(warnings) == 1 and "is a whole drive" in warnings[0]

    def test_malformed_roots_are_reported(
        self, sandbox_server, monkeypatch, package_caplog
    ):
        monkeypatch.setenv("PMM_IMPORT_ROOTS", "relative/folder")
        sandbox_server._setup_environment()
        warnings = _import_warnings(package_caplog)
        assert len(warnings) == 1 and "signal imports will be refused" in warnings[0]
