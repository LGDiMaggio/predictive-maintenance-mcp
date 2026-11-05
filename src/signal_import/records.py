"""
The import record store: one append-only, content-addressed record per
interpretation of one file's bytes.

ISO 13374 Block 1 (data acquisition), provenance layer. Every import writes
its record here before any of its signals becomes visible, so an imported
signal's meaning can always be read back. Nothing here leaves the machine,
nothing here decodes a file, and nothing here imports MCP.

Layout: ``<import dir>/records/<import_id>.jsonl`` (``config.get_import_dir``
at the service boundary), one JSON object per line in the ledger's canonical
form (compact, key-sorted, ASCII-only), each line ending with the single byte
0x0A.

Line 1 holds only what the ``import_id`` determines: the source digest and
size and the canonical interpretation (``import_config``)::

    {"import_id": "<64 hex>", "interpretation": {...}, "kind": "interpretation",
     "schema": "pmm-import-record/1",
     "source": {"sha256": "<64 hex>", "size_bytes": 1234}}

so two writers of the same interpretation write the same bytes. It is written
under the record's sidecar lock into a temporary file, synced, and then
linked to its final name with a link that never replaces an existing file:
line 1 exists completely or not at all, and a crash in between leaves no
empty line 1. The writer removes its own temporary file, the only file this
module ever deletes. An existing line 1 counts as written when recomputing
the ``import_id`` from its content gives the record's name.

Later lines are events of :data:`EVENT_KINDS` (who confirmed what through
which channel, where the file was found, the outcome per channel, runtime
versions), each with the server time it was recorded::

    {"import_id": "...", "kind": "location", "payload": {...},
     "recorded_at": "2026-11-05T10:00:00.000000+00:00",
     "schema": "pmm-import-record/1"}

They are not chained to each other or to line 1: replay never reads them,
and an unkeyed chain would stop nobody who can write the directory. Their
payloads hold verbatim file evidence (paths, labels); a caller wraps any of
it in ``FileText`` before it reaches a result. The index reads two payload
keys: a ``location`` event's ``location`` (the path as found, a string) and
an ``outcome`` event's ``channels`` (a list of objects with an integer
``column`` and a ``signal_id`` string or null). A later line whose time has
no UTC offset, or whose payload has another shape, is read as unreadable.

The index lists records newest first by their latest later line, so a record
whose interpretation is used again (an earlier interpretation restored)
comes first again, and ``supersedes`` names, per shared column, the record
of the same bytes used most recently before it.

Reading reuses the ledger's byte-level splitter. A torn last line counts as
one unreadable line, an unparseable later line as another, and only an
invalid line 1 refuses the record. The store never deletes or rewrites a
record; a user may delete one, which leaves ledger replay unaffected.

Locking, appending and their failure messages are the ledger store's
primitives with this store's wording, so a failure names
``PMM_IMPORT_DIR``; the store raises the ledger's ``LedgerLockTimeout`` and
``LedgerWriteError`` for them, and :class:`ImportRecordError` for a record it
refuses.
"""

import errno
import json
import os
import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

from ..asset_ledger.store import (
    LOCK_TIMEOUT_SECONDS,
    StoreWording,
    acquire_lock,
    append_line,
    release_lock,
    split_records,
    utc_now_iso,
)
from ..canonical_json import canonical_json
from ..file_text import LABEL_CHARS, FileText
from ..path_safety import safe_resolve
from ..signal_acquisition.import_config import (
    import_id as compute_import_id,
)
from ..signal_acquisition.import_config import (
    is_import_id,
    normalize_interpretation,
)

__all__ = [
    "RECORD_SCHEMA",
    "RECORDS_DIR_NAME",
    "RECORD_SUFFIX",
    "RECORD_LOCK_SUFFIX",
    "EVENT_KINDS",
    "RECORD_WORDING",
    "ImportRecordError",
    "ImportRecordDamaged",
    "ImportRecordMissing",
    "RecordWrite",
    "RecordEvent",
    "ImportRecord",
    "RecordChannel",
    "ImportRecordIndexEntry",
    "ImportRecordStore",
]

#: Schema id written on every record line.
RECORD_SCHEMA = "pmm-import-record/1"

#: Subdirectory of the import directory that holds the records.
RECORDS_DIR_NAME = "records"

#: Suffix of a record file and of its sidecar lock.
RECORD_SUFFIX = ".jsonl"
RECORD_LOCK_SUFFIX = ".jsonl.lock"

#: Kinds of the lines after line 1.
EVENT_KINDS: tuple[str, ...] = ("attestation", "location", "outcome", "runtime")

#: The wording of this store's lock and append failures.
RECORD_WORDING = StoreWording(
    directory="the import record directory (PMM_IMPORT_DIR)",
    directory_short="the import record directory",
    lock="import record lock",
    torn=(
        "The torn tail will be read as one unreadable line; check free space "
        "and retry."
    ),
    failed="Check free space and permissions, then retry.",
)

_RECORD_NAME = re.compile(r"[0-9a-f]{64}" + re.escape(RECORD_SUFFIX))
_TEMP_PREFIX = ".tmp-"

#: ``os.link`` errors that mean "this filesystem cannot hold hard links".
_NO_LINK_ERRNOS = frozenset(
    code
    for code in (
        getattr(errno, "EPERM", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "EXDEV", None),
        getattr(errno, "EINVAL", None),
        getattr(errno, "ENOSYS", None),
        getattr(errno, "EMLINK", None),
    )
    if code is not None
)

_O_BINARY = getattr(os, "O_BINARY", 0)


class ImportRecordError(ValueError):
    """A record the store refuses to name, create, read or extend."""


class ImportRecordDamaged(ImportRecordError):
    """A record whose bytes were read and whose line 1 fails its check: the
    one refusal that tells the operator to move the record aside. A record
    that cannot be read at all raises :class:`ImportRecordError` itself."""


class ImportRecordMissing(ImportRecordError):
    """No record under that id: never written, or moved out of the import
    record directory."""


class RecordWrite(NamedTuple):
    """Result of :meth:`ImportRecordStore.write_interpretation`.

    Attributes:
        import_id: The interpretation's identity.
        created: True when this call wrote line 1; False when an identical
            line 1 already existed (another root, a retry, another process).
        path: The record file.
    """

    import_id: str
    created: bool
    path: Path


class RecordEvent(NamedTuple):
    """One readable line after line 1.

    Attributes:
        line: Physical line number in the record (2 or more).
        kind: One of :data:`EVENT_KINDS`.
        recorded_at: Server time the line was written (ISO 8601, UTC).
        payload: The event's content, verbatim (it may hold file evidence).
    """

    line: int
    kind: str
    recorded_at: str
    payload: dict[str, Any]


class ImportRecord(NamedTuple):
    """A record read back.

    Attributes:
        import_id: The interpretation's identity (verified against line 1).
        source_sha256: SHA-256 of the file bytes.
        size_bytes: Size of the file in bytes.
        interpretation: The canonical interpretation.
        events: The readable later lines, in file order.
        unreadable_lines: Physical line numbers of later lines that could not
            be read (a torn last line included).
    """

    import_id: str
    source_sha256: str
    size_bytes: int
    interpretation: dict[str, Any]
    events: tuple[RecordEvent, ...]
    unreadable_lines: tuple[int, ...]


class RecordChannel(NamedTuple):
    """One channel of a record's latest outcome."""

    column: int
    signal_id: Optional[str]


class ImportRecordIndexEntry(NamedTuple):
    """One entry of :meth:`ImportRecordStore.list_records`.

    Attributes:
        import_id: The interpretation's identity.
        file_name: Name of the file at its latest recorded location, as file
            text (None when no location was recorded).
        recorded_at: Time of the record's first later line (None when the
            record holds line 1 only).
        last_recorded_at: Time of its latest later line: the index is newest
            first by this time, so a record whose interpretation was used
            again (an earlier interpretation restored) comes first again.
        source_sha256: SHA-256 of the file bytes.
        columns: Physical columns the interpretation imports.
        channels: Column and signal id per channel of the latest outcome.
        supersedes: Records of the same bytes, last used before this one,
            that share a column: per shared column the one used most
            recently, oldest first.
        unreadable_lines: Count of later lines that could not be read.
    """

    import_id: str
    file_name: Optional[FileText]
    recorded_at: Optional[str]
    last_recorded_at: Optional[str]
    source_sha256: str
    columns: tuple[int, ...]
    channels: tuple[RecordChannel, ...]
    supersedes: tuple[str, ...]
    unreadable_lines: int


def _short(import_id: str) -> str:
    return f"import record {import_id[:12]}"


def _not_an_id() -> ImportRecordError:
    return ImportRecordError(
        "import_id must be exactly 64 lowercase hexadecimal characters, as "
        "validation and the record listing return it."
    )


def _line_one(source_sha256: str, size_bytes: int, interpretation: dict) -> dict:
    return {
        "schema": RECORD_SCHEMA,
        "kind": "interpretation",
        "import_id": compute_import_id(source_sha256, size_bytes, interpretation),
        "source": {"sha256": source_sha256, "size_bytes": size_bytes},
        "interpretation": interpretation,
    }


def _encode(obj: dict[str, Any]) -> bytes:
    return canonical_json(obj).encode("ascii") + b"\n"


def _decode(raw: bytes) -> Optional[dict[str, Any]]:
    if raw.endswith(b"\r"):
        raw = raw[:-1]
    try:
        obj = json.loads(raw.decode("ascii"))
    except (ValueError, RecursionError):  # RecursionError: nested too deep
        return None
    return obj if isinstance(obj, dict) else None


def _corrupt(import_id: str) -> ImportRecordDamaged:
    return ImportRecordDamaged(
        f"The first line of {_short(import_id)} does not match its import_id: "
        f"the record was edited or damaged outside the server, so it is not "
        f"read, and the same interpretation cannot be recorded again while it "
        f"is there. Ledger replay does not depend on it. Move "
        f"records/{import_id}.jsonl out of the import record directory "
        f"(PMM_IMPORT_DIR) to record it again; the server never deletes a "
        f"record."
    )


def _verified_line_one(import_id: str, raw: bytes) -> dict[str, Any]:
    """The decoded line 1, or :func:`_corrupt` when it fails its check."""
    obj = _decode(raw)
    try:
        if (
            obj is None
            or obj.get("schema") != RECORD_SCHEMA
            or obj.get("kind") != "interpretation"
            or obj.get("import_id") != import_id
            or not isinstance(obj.get("source"), dict)
            or set(obj) != {"schema", "kind", "import_id", "source", "interpretation"}
        ):
            raise ValueError
        source = obj["source"]
        recomputed = compute_import_id(
            source.get("sha256"), source.get("size_bytes"), obj["interpretation"]
        )
        canonical = normalize_interpretation(obj["interpretation"])
    except ValueError:
        raise _corrupt(import_id) from None
    if recomputed != import_id or canonical != obj["interpretation"]:
        raise _corrupt(import_id)
    return obj


def _well_formed_payload(kind: str, payload: dict[str, Any]) -> bool:
    """The payload keys the index reads have the shape it reads them in."""
    if kind == "location":
        return isinstance(payload.get("location", ""), str)
    if kind == "outcome":
        channels = payload.get("channels", [])
        return isinstance(channels, list) and all(
            isinstance(item, dict)
            and (item.get("column") is None or type(item.get("column")) is int)
            and (
                item.get("signal_id") is None or isinstance(item.get("signal_id"), str)
            )
            for item in channels
        )
    return True


def _event(index: int, raw: bytes, import_id: str) -> Optional[RecordEvent]:
    """The later line as an event, or None when it is unreadable (malformed,
    another record's, an unknown kind, a time without its UTC offset, or a
    payload the index cannot read)."""
    obj = _decode(raw)
    if (
        obj is None
        or obj.get("schema") != RECORD_SCHEMA
        or obj.get("import_id") != import_id
        or obj.get("kind") not in EVENT_KINDS
        or _instant(obj.get("recorded_at")) is None
        or not isinstance(obj.get("payload"), dict)
        or not _well_formed_payload(obj["kind"], obj["payload"])
    ):
        return None
    return RecordEvent(index + 1, obj["kind"], obj["recorded_at"], obj["payload"])


def _basename(location: object) -> Optional[str]:
    """The last segment of a recorded location (either separator), or None."""
    if not isinstance(location, str) or not location:
        return None
    return re.split(r"[\\/]", location)[-1]


def _instant(value: object) -> Optional[datetime]:
    """An ISO 8601 time with a UTC offset, or None (a naive time cannot be
    ordered against the server's own)."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


class ImportRecordStore:
    """Append-only import records under one directory.

    Two instances (or two processes) on the same directory see the same
    files: all coordination goes through the sidecar locks.

    Args:
        root: The import directory (``config.get_import_dir()`` at the
            service boundary). Not created here.
        lock_timeout: Seconds to wait for a busy record lock.
    """

    def __init__(
        self, root: Union[str, Path], *, lock_timeout: float = LOCK_TIMEOUT_SECONDS
    ) -> None:
        self.root = Path(root)
        self.lock_timeout = float(lock_timeout)

    @property
    def records_dir(self) -> Path:
        return self.root / RECORDS_DIR_NAME

    def _paths(self, import_id: object) -> tuple[str, Path, Path]:
        """Check the id's form BEFORE any path is built, then resolve."""
        if not isinstance(import_id, str) or not is_import_id(import_id):
            raise _not_an_id()
        record = safe_resolve(self.records_dir, f"{import_id}{RECORD_SUFFIX}")
        lock = safe_resolve(self.records_dir, f"{import_id}{RECORD_LOCK_SUFFIX}")
        return import_id, record, lock

    def _ensure_dir(self) -> None:
        try:
            self.records_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ImportRecordError(
                f"Cannot create the import record directory: "
                f"{exc.strerror or exc}. Point PMM_IMPORT_DIR at a writable "
                f"local directory."
            ) from exc

    def _read_bytes(self, import_id: str, record: Path) -> bytes:
        try:
            with open(record, "rb") as handle:
                return handle.read()
        except FileNotFoundError:
            raise ImportRecordMissing(
                f"There is no {_short(import_id)} in the import record "
                f"directory (PMM_IMPORT_DIR). List the records to find the "
                f"import_id, or validate and import the file again."
            ) from None
        except OSError as exc:
            raise ImportRecordError(
                f"Cannot read {_short(import_id)}: {exc.strerror or exc}. Check "
                f"that the import record directory (PMM_IMPORT_DIR) is readable."
            ) from exc

    @contextmanager
    def _locked(self, import_id: str, lock: Path) -> Iterator[None]:
        fd = acquire_lock(
            lock,
            what=f"{_short(import_id)} in PMM_IMPORT_DIR",
            timeout=self.lock_timeout,
            wording=RECORD_WORDING,
        )
        try:
            yield
        finally:
            release_lock(fd)

    def _load_line_one(
        self, import_id: str, record: Path
    ) -> tuple[list[bytes], bytes, dict[str, Any]]:
        """The record's segments and torn tail, and its verified line 1."""
        segments, tail, _ = split_records(self._read_bytes(import_id, record))
        if not segments:
            raise _corrupt(import_id)
        return segments, tail, _verified_line_one(import_id, segments[0])

    # -- writing -------------------------------------------------------------

    def write_interpretation(
        self, source_sha256: str, size_bytes: int, interpretation: dict[str, Any]
    ) -> RecordWrite:
        """Write line 1 of the record of *interpretation*, once.

        Args:
            source_sha256: SHA-256 of the file bytes, 64 lowercase hex.
            size_bytes: Size of the file in bytes.
            interpretation: The interpretation (normalized here).

        Returns:
            The :class:`RecordWrite`; ``created`` is False when an identical
            line 1 already existed.

        Raises:
            ValueError: From the interpretation normalizer or for a malformed
                digest or size.
            ImportRecordDamaged: An existing line 1 fails its check.
            ImportRecordError: The directory cannot be created, the
                filesystem cannot hold hard links, or an existing record
                cannot be read.
            LedgerLockTimeout, LedgerWriteError: From the shared lock.
        """
        canonical = normalize_interpretation(interpretation)
        first = _line_one(source_sha256, size_bytes, canonical)
        import_id, record, lock = self._paths(first["import_id"])
        line = _encode(first)
        self._ensure_dir()
        with self._locked(import_id, lock):
            if record.exists():
                self._load_line_one(import_id, record)
                return RecordWrite(import_id, False, record)
            created = self._link_line_one(import_id, record, line)
            return RecordWrite(import_id, created, record)

    def _link_line_one(self, import_id: str, record: Path, line: bytes) -> bool:
        temp = self.records_dir / (
            f"{_TEMP_PREFIX}{import_id[:16]}-{os.getpid()}-{secrets.token_hex(4)}"
        )
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_BINARY, 0o644)
            try:
                if os.write(fd, line) != len(line):
                    raise ImportRecordError(
                        f"Short write while creating {_short(import_id)}. "
                        f"{RECORD_WORDING.failed}"
                    )
                os.fsync(fd)
            finally:
                os.close(fd)
            try:
                os.link(temp, record)
            except FileExistsError:
                self._load_line_one(import_id, record)
                return False
            except OSError as exc:
                if exc.errno in _NO_LINK_ERRNOS or getattr(exc, "winerror", None) == 1:
                    raise ImportRecordError(
                        "The import record directory is on a filesystem that "
                        "cannot hold hard links (FAT, exFAT and some network "
                        "shares), which records need. Point PMM_IMPORT_DIR at a "
                        "local NTFS, APFS or ext4 directory."
                    ) from exc
                raise ImportRecordError(
                    f"Cannot create {_short(import_id)}: {exc.strerror or exc}. "
                    f"{RECORD_WORDING.failed}"
                ) from exc
            return True
        except OSError as exc:
            raise ImportRecordError(
                f"Cannot create {_short(import_id)}: {exc.strerror or exc}. Check "
                f"that the import record directory (PMM_IMPORT_DIR) is writable."
            ) from exc
        finally:
            try:
                os.unlink(temp)
            except OSError:
                pass  # never created, or busy: a stray temporary file is inert

    def append_event(self, import_id: str, kind: str, payload: dict[str, Any]) -> str:
        """Append one later line to an existing record.

        Args:
            import_id: The record's identity.
            kind: One of :data:`EVENT_KINDS`.
            payload: The event's content (canonical JSON; no floats that are
                not finite).

        Returns:
            The server time written as the line's ``recorded_at``.

        Raises:
            ImportRecordError: A malformed id, or no record with that id.
            ValueError: An unknown kind or a payload that is not a JSON
                object.
            LedgerLockTimeout, LedgerWriteError: From the shared primitives.
        """
        import_id, record, lock = self._paths(import_id)
        if kind not in EVENT_KINDS:
            raise ValueError(f"kind must be one of {EVENT_KINDS}, got {kind!r}.")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object (a dict).")
        recorded_at = utc_now_iso()
        line = _encode(
            {
                "schema": RECORD_SCHEMA,
                "kind": kind,
                "import_id": import_id,
                "recorded_at": recorded_at,
                "payload": payload,
            }
        )
        if not record.exists():  # checked again under the lock
            raise ImportRecordError(
                f"There is no {_short(import_id)} to extend: its "
                f"interpretation line is written first."
            )
        with self._locked(import_id, lock):
            if not record.exists():
                raise ImportRecordError(
                    f"There is no {_short(import_id)} to extend: its "
                    f"interpretation line is written first."
                )
            append_line(record, line, what=_short(import_id), wording=RECORD_WORDING)
        return recorded_at

    # -- reading -------------------------------------------------------------

    def read(self, import_id: str) -> ImportRecord:
        """Read one record back (no lock; readers never take one).

        Raises:
            ImportRecordDamaged: A line 1 that fails its check.
            ImportRecordMissing: No record under that id.
            ImportRecordError: A malformed id, or a record that cannot be
                read (an operating-system error, which may pass).
        """
        import_id, record, _ = self._paths(import_id)
        segments, tail, first = self._load_line_one(import_id, record)
        events: list[RecordEvent] = []
        unreadable: list[int] = []
        for index, raw in enumerate(segments[1:], start=1):
            event = _event(index, raw, import_id)
            if event is None:
                unreadable.append(index + 1)
            else:
                events.append(event)
        if tail:
            unreadable.append(len(segments) + 1)
        return ImportRecord(
            import_id=import_id,
            source_sha256=first["source"]["sha256"],
            size_bytes=first["source"]["size_bytes"],
            interpretation=first["interpretation"],
            events=tuple(events),
            unreadable_lines=tuple(unreadable),
        )

    def _record_names(self) -> list[str]:
        try:
            names = os.listdir(self.records_dir)
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise ImportRecordError(
                f"Cannot list the import record directory: {exc.strerror or exc}. "
                f"Point PMM_IMPORT_DIR at a readable directory."
            ) from exc
        return sorted(name for name in names if _RECORD_NAME.fullmatch(name))

    def list_records(
        self,
        *,
        file_sha256: Optional[str] = None,
        file_name: Optional[str] = None,
        signal_id: Optional[str] = None,
    ) -> tuple[list[ImportRecordIndexEntry], tuple[str, ...]]:
        """Index the records, newest first, optionally filtered.

        Args:
            file_sha256: Keep records of these file bytes.
            file_name: Keep records with a location of this file name (exact
                spelling).
            signal_id: Keep records whose latest outcome names this signal.

        Returns:
            ``(entries, refused)``: the index entries, newest first (records
            without any later line last), and the ids of records whose line
            1 failed its check, which the index leaves out. A record moved
            out of the directory while it is listed is left out silently.

        Raises:
            ImportRecordError: The directory, or one record in it, cannot be
                read (an operating-system error, which may pass); such a
                record is never reported as damaged.
        """
        entries: list[ImportRecordIndexEntry] = []
        locations: dict[str, list[str]] = {}
        refused: list[str] = []
        for name in self._record_names():
            import_id = name[: -len(RECORD_SUFFIX)]
            try:
                record = self.read(import_id)
            except ImportRecordDamaged:
                refused.append(import_id)
                continue
            except ImportRecordMissing:
                continue  # moved aside after the directory was listed
            names: list[str] = []
            channels: tuple[RecordChannel, ...] = ()
            for event in record.events:
                if event.kind == "location":
                    basename = _basename(event.payload.get("location"))
                    if basename is not None:
                        names.append(basename)
                elif event.kind == "outcome":
                    channels = tuple(
                        RecordChannel(item.get("column"), item.get("signal_id"))
                        for item in event.payload.get("channels", [])
                        if isinstance(item, dict)
                    )
            locations[import_id] = names
            entries.append(
                ImportRecordIndexEntry(
                    import_id=import_id,
                    file_name=(
                        FileText.of(names[-1], cap=LABEL_CHARS) if names else None
                    ),
                    recorded_at=(
                        record.events[0].recorded_at if record.events else None
                    ),
                    last_recorded_at=(
                        record.events[-1].recorded_at if record.events else None
                    ),
                    source_sha256=record.source_sha256,
                    columns=tuple(
                        channel["column"]
                        for channel in record.interpretation["channels"]
                    ),
                    channels=channels,
                    supersedes=(),
                    unreadable_lines=len(record.unreadable_lines),
                )
            )

        entries = _with_supersedes(entries)
        if file_sha256 is not None:
            entries = [e for e in entries if e.source_sha256 == file_sha256]
        if file_name is not None:
            entries = [e for e in entries if file_name in locations[e.import_id]]
        if signal_id is not None:
            entries = [
                e for e in entries if any(c.signal_id == signal_id for c in e.channels)
            ]
        instants = {e.import_id: _instant(e.last_recorded_at) for e in entries}
        dated = [e for e in entries if instants[e.import_id] is not None]
        undated = [e for e in entries if instants[e.import_id] is None]
        dated.sort(key=lambda e: (instants[e.import_id], e.import_id), reverse=True)
        undated.sort(key=lambda e: e.import_id)
        return dated + undated, tuple(refused)


def _with_supersedes(
    entries: list[ImportRecordIndexEntry],
) -> list[ImportRecordIndexEntry]:
    """Each entry with ``supersedes`` set: per column it shares with records
    of the same bytes last used before it, the one used most recently, oldest
    first. Only records of the same bytes are compared."""
    dated_by_source: dict[str, list[tuple[datetime, ImportRecordIndexEntry]]] = {}
    for entry in entries:
        when = _instant(entry.last_recorded_at)
        if when is not None:
            dated_by_source.setdefault(entry.source_sha256, []).append((when, entry))
    result: list[ImportRecordIndexEntry] = []
    for entry in entries:
        when = _instant(entry.last_recorded_at)
        if when is None:
            result.append(entry)
            continue
        latest: dict[int, tuple[datetime, str]] = {}
        for other_when, other in dated_by_source[entry.source_sha256]:
            if other.import_id == entry.import_id or other_when >= when:
                continue
            for column in set(entry.columns) & set(other.columns):
                if column not in latest or latest[column][0] < other_when:
                    latest[column] = (other_when, other.import_id)
        chosen = sorted(set(latest.values()))
        supersedes = tuple(dict.fromkeys(import_id for _, import_id in chosen))
        result.append(entry._replace(supersedes=supersedes))
    return result
