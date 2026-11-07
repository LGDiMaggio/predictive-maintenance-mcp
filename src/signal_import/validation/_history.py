"""
Read-only history checks of an import validation.

Earlier import records of the same bytes and every asset ledger (two
bounded scans), the ledger pre-check, the comparison with earlier
interpretations, and the fail-closed check of history that could not be
read, or was not read in full.

ISO 13374 Block 1 (data acquisition). Internal to
:mod:`predictive_maintenance_mcp.signal_import.validation`; nothing here
writes anywhere.
"""

import math
import os
import re
import stat
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping, Optional

from ...asset_ledger.store import (
    EVENT_MEASUREMENT_RECORDED,
    EVENT_TYPES,
    LEDGER_SUFFIX,
    LedgerError,
    LedgerReadResult,
    LedgerStore,
    _decode_record,
    build_asset_view,
    split_records,
)
from ...signal_acquisition.import_config import MAX_FIELDS
from ...signal_acquisition.measurement import (
    measurement_id_from_digest,
    validate_ledger_id,
)
from ...signal_acquisition.text_decoder import DecodeResult
from .. import statements as st
from ..contracts import ChannelPrecheck
from ..records import (
    RECORD_SUFFIX,
    ImportRecordDamaged,
    ImportRecordError,
    ImportRecordStore,
    _corrupt,
    _verified_line_one,
)
from ._common import _Run

_MIB = 1024 * 1024

#: Most import records one validation reads to find the earlier records of
#: the same bytes. A record's file digest is stored only inside the record,
#: so finding them means reading every record (about 2 ms each, measured):
#: a store holding more than this is not compared at all, and validation
#: reports the history as incomplete instead of taking it as empty.
#: Approved as a temporary safety limit of this version.
HISTORY_RECORD_LIMIT = 250

#: Most bytes of any one asset ledger or import record a validation reads:
#: 8 MiB. Enforced on the bytes actually read (a file that grows after it
#: was sized is never read past it), and checked first against the sizes in
#: the directory, so an oversized file is not read at all. Measured: an
#: 8 MiB ledger (about 6,500 measurement events) reads in about 0.25 s and
#: keeps about 40 MiB in memory. Approved 2026-11-06 as a temporary U8
#: safety limit of the history-validation operation.
HISTORY_FILE_BYTES_LIMIT = 8 * _MIB

#: Most bytes of asset ledgers and import records together one validation
#: reads: 32 MiB (ledger files that are not listed as asset ledgers count
#: too), enforced on the bytes actually read. Measured: about 32 MiB across
#: 250 ledgers reads in about 2.2 s warm (6.4 s on a first read after
#: writing) and keeps about 155 MiB in memory. Approved 2026-11-06 as a
#: temporary U8 safety limit.
HISTORY_TOTAL_BYTES_LIMIT = 32 * _MIB

#: Most ledger events one validation projects and checks, across every
#: asset ledger it reads: about 26,000 measurement events fill the byte
#: bound, and smaller events stop here first. Approved 2026-11-06 as a
#: temporary U8 safety limit.
HISTORY_EVENT_LIMIT = 30_000

_BOUND_NOTE = " This is a safety bound of this version, not damage to the history."

#: How history files are opened: read only, binary, and without blocking
#: where the platform has it, so a FIFO swapped in is never waited on.
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
_CHUNK = _MIB


class _NotRegular(Exception):
    """A history file that is not a regular file (a directory, FIFO,
    device or socket): never read."""


def _read_bounded(path: Any, limit: int) -> Optional[bytes]:
    """The bytes of the regular file *path*, or None when it holds more than
    *limit* bytes; never more than *limit* + 1 bytes are consumed.

    The type is checked on the opened handle, so a file replaced after it
    was listed is judged as what is actually read.

    Raises:
        _NotRegular: The path is not a regular file.
        FileNotFoundError: The file is gone.
        OSError: The file cannot be opened or read.
    """
    try:
        fd = os.open(path, _READ_FLAGS)
    except PermissionError:
        if os.path.isdir(path):  # where a directory cannot be opened at all
            raise _NotRegular(path) from None
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise _NotRegular(path)
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, _CHUNK))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    data = b"".join(chunks)
    return None if len(data) > limit else data


def _size_words(count: int) -> str:
    return f"{count // _MIB} MiB" if count % _MIB == 0 else f"{count} bytes"


@dataclass
class _Budget:
    """The bytes and events one validation may still read, shared by the
    record scan and the ledger scan, whichever reads first."""

    used_bytes: int = 0
    used_events: int = 0

    def remaining_bytes(self) -> int:
        return max(HISTORY_TOTAL_BYTES_LIMIT - self.used_bytes, 0)

    def take_bytes(self, count: int) -> None:
        self.used_bytes += count

    def take_events(self, count: int) -> bool:
        if self.used_events + count > HISTORY_EVENT_LIMIT:
            return False
        self.used_events += count
        return True

    def over_total(self) -> str:
        return (
            f"The ledger and import record directories hold more than the "
            f"{_size_words(HISTORY_TOTAL_BYTES_LIMIT)} one validation reads"
        )

    def read(self, path: Any) -> tuple[bytes, Optional[str]]:
        """``(bytes, None)`` of a regular history file within the bounds, or
        ``(b"", "file" | "total")`` when reading it would pass the per-file
        or the total bound (nothing past it is kept). Raises like
        :func:`_read_bounded`."""
        limit = min(HISTORY_FILE_BYTES_LIMIT, self.remaining_bytes())
        data = _read_bounded(path, limit)
        if data is None:
            return b"", "file" if limit == HISTORY_FILE_BYTES_LIMIT else "total"
        self.take_bytes(len(data))
        return data, None


def _file_entries(directory: Any, names: list[str]) -> list[tuple[str, int, bool]]:
    """``(name, size, regular)`` of each named file, from the directory
    alone; a file that disappears between listing and sizing counts as an
    empty regular file (the read finds it gone)."""
    entries = []
    for name in names:
        try:
            status = os.stat(os.path.join(directory, name))
        except FileNotFoundError:
            entries.append((name, 0, True))
            continue
        entries.append((name, status.st_size, stat.S_ISREG(status.st_mode)))
    return entries


def _over_total(entries: list[tuple[str, int, bool]], budget: _Budget) -> bool:
    return (
        sum(size for _n, size, regular in entries if regular) > budget.remaining_bytes()
    )


def _record_line_one(import_id: str, data: bytes) -> dict[str, Any]:
    """The verified line 1 of a record read as *data* (the same check
    :meth:`ImportRecordStore.read` applies).

    Raises:
        ImportRecordDamaged: A line 1 that fails its check.
    """
    segments, _tail, _end = split_records(data)
    if not segments:
        raise _corrupt(import_id)
    return _verified_line_one(import_id, segments[0])


def _ledger_stem(stem: object) -> bool:
    """Whether *stem* is a name the store gives an asset ledger."""
    try:
        validate_ledger_id(stem, kind="asset_id")
    except ValueError:
        return False
    return True


def _over_file(entries: list[tuple[str, int, bool]]) -> Optional[str]:
    return next(
        (
            name
            for name, size, regular in entries
            if regular and size > HISTORY_FILE_BYTES_LIMIT
        ),
        None,
    )


@dataclass
class _Prior:
    """Earlier records of the same bytes, read once and only when the store
    holds at most :data:`HISTORY_RECORD_LIMIT` records within the byte
    bounds. An unreadable store or record raises; a damaged one is listed in
    *refused*; neither ever reads as no records."""

    store: ImportRecordStore
    sha256: Optional[str]
    budget: _Budget = field(default_factory=_Budget)
    loaded: Optional[dict[str, dict[str, Any]]] = None
    refused: tuple[str, ...] = ()
    #: Why the records were not read, when a bound kept them unread.
    incomplete: Optional[str] = None

    def records(self) -> dict[str, dict[str, Any]]:
        if self.loaded is None:
            self.loaded = self._read() if self.sha256 is not None else {}
        return self.loaded

    def _file_bound(self, name: str) -> str:
        return (
            f"The import record {name[: -len(RECORD_SUFFIX)][:12]} holds more than "
            f"the {_size_words(HISTORY_FILE_BYTES_LIMIT)} one validation reads per "
            f"file"
        )

    def _read(self) -> dict[str, dict[str, Any]]:
        """Every record, each read once within the bounds; the ones of these
        bytes kept by their interpretation. A record moved aside after the
        directory was listed is left out, as the store's index does."""
        names = self.store._record_names()
        try:
            entries = _file_entries(self.store.records_dir, names)
        except OSError as exc:
            raise ImportRecordError(
                f"Cannot size the import records: {exc.strerror or exc}. Check that "
                f"the import record directory (PMM_IMPORT_DIR) is readable."
            ) from exc
        over = _over_file(entries)
        if len(names) > HISTORY_RECORD_LIMIT:
            self.incomplete = (
                f"The import record directory holds {len(names)} records, "
                f"more than the {HISTORY_RECORD_LIMIT} one validation reads"
            )
        elif over is not None:
            self.incomplete = self._file_bound(over)
        elif _over_total(entries, self.budget):
            self.incomplete = self.budget.over_total()
        if self.incomplete is not None:
            return {}
        loaded: dict[str, dict[str, Any]] = {}
        refused: list[str] = []
        for name, _size, regular in entries:
            import_id = name[: -len(RECORD_SUFFIX)]
            if not regular:
                refused.append(import_id)
                continue
            try:
                data, bound = self.budget.read(self.store.records_dir / name)
            except FileNotFoundError:
                continue
            except _NotRegular:
                refused.append(import_id)
                continue
            except OSError as exc:
                raise ImportRecordError(
                    f"Cannot read import {import_id[:12]}: {exc.strerror or exc}. "
                    f"Check that the import record directory (PMM_IMPORT_DIR) is "
                    f"readable."
                ) from exc
            if bound is not None:
                self.incomplete = (
                    self._file_bound(name)
                    if bound == "file"
                    else self.budget.over_total()
                )
                return {}
            try:
                first = _record_line_one(import_id, data)
            except ImportRecordDamaged:
                refused.append(import_id)
                continue
            if first["source"]["sha256"] == self.sha256:
                loaded[import_id] = first["interpretation"]
        self.refused = tuple(refused)
        return loaded

    def interpretation(self, import_id: Optional[str]) -> Optional[dict[str, Any]]:
        if import_id is None:
            return None
        return self.records().get(import_id)

    def unread(self, import_id: Optional[str]) -> bool:
        """Whether *import_id* may be a record this validation did not read
        (the store was over the limit, or the record is damaged): a cited
        record then cannot be compared, which is not a mismatch."""
        self.records()
        return self.incomplete is not None or import_id in self.refused

    def time_base_rate(self, import_id: Optional[str]) -> Optional[str]:
        found = self.interpretation(import_id)
        rate: Optional[str] = (
            None if found is None else found["time_base"]["sampling_rate_hz"]
        )
        return rate

    def channel(
        self, import_id: Optional[str], column: int
    ) -> Optional[dict[str, Any]]:
        found = self.interpretation(import_id)
        if found is None:
            return None
        for channel in found["channels"]:
            if channel["column"] == column:
                return dict(channel)
        return None


def _slot(declaration: Mapping[str, Any]) -> tuple[Any, ...]:
    instant = declaration.get("acquired_at")
    try:
        parsed: Any = (
            datetime.fromisoformat(instant) if isinstance(instant, str) else instant
        )
    except ValueError:
        parsed = instant
    return (parsed, declaration.get("direction"), declaration.get("sensor_id"))


def _range(import_format: Mapping[str, Any]) -> Optional[tuple[int, int]]:
    lines = _mapping(import_format.get("data_lines"))
    first = lines.get("first")
    count = import_format.get("sample_count")
    if not isinstance(first, int) or not isinstance(count, int):
        return None
    last = (
        lines.get("last") if isinstance(lines.get("last"), int) else first + count - 1
    )
    return first, last


def _mapping(value: Any) -> Mapping[str, Any]:
    """A ledger payload part as a mapping; anything else as an empty one, so
    an unexpected shape reads as nothing recorded for that part."""
    return value if isinstance(value, Mapping) else {}


def _as_float(value: Any) -> Optional[float]:
    """A recorded or configured number as a float; None when it is absent or
    unreadable (a hand-written ledger may hold anything)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _changed(recorded: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """The declaration keys a supersession would change. A recorded value
    that cannot be read counts as changed, never as equal."""
    changed: list[str] = []
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
        "sampling_rate",
        "signal_unit",
    ):
        if key == "sampling_rate":
            left, right = _as_float(recorded.get(key)), _as_float(new.get(key))
            same = left is not None and right is not None and left == right
        elif key in ("rpm", "load"):
            raw_left, raw_right = recorded.get(key), new.get(key)
            left, right = _as_float(raw_left), _as_float(raw_right)
            same = (raw_left is None and raw_right is None) or (
                left is not None and right is not None and left == right
            )
        elif key == "acquired_at":
            same = _slot(recorded)[0] == _slot(new)[0]
        else:
            same = recorded.get(key) == new.get(key)
        if not same:
            changed.append(key)
    old_format = recorded.get("import_format")
    new_format = new["import_format"]
    if not isinstance(old_format, Mapping):
        changed.append("import_format")
    else:
        changed.extend(
            f"import_format.{key}"
            for key in new_format
            if old_format.get(key) != new_format[key]
        )
    return changed


#: Most asset ledgers one validation reads. A measurement of these bytes may
#: be recorded under any asset, and the global measurement index can lag the
#: ledgers (it is appended after them, separately, and a failed append is
#: not undone), so the ledgers themselves are read (about 2.7 ms for a
#: ledger of 21 events, measured): a ledger directory holding more ledger
#: files than this is not compared at all, and validation reports the
#: history as incomplete. Approved as a temporary safety limit of this
#: version.
HISTORY_ASSET_LIMIT = 250


def _instant(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _record_damage(payload: Mapping[str, Any], index: int, sha256: str) -> bool:
    """Whether a recorded measurement of these bytes cannot be read in full,
    or disagrees with the id it is recorded under. The id derives from the
    file digest and the channel index, so both must be these; every field
    the pre-check compares must have the type it is written with."""
    declaration = payload.get("declaration")
    file_block = payload.get("file")
    if not isinstance(declaration, Mapping) or not isinstance(file_block, Mapping):
        return True
    channel_index = declaration.get("channel_index")
    if (
        file_block.get("content_sha256") != sha256
        or isinstance(channel_index, bool)
        or not isinstance(channel_index, int)
        or channel_index != index
    ):
        return True
    for key in ("asset_id", "measurement_point_id"):
        value = declaration.get(key)
        if not isinstance(value, str) or not value:
            return True
    if _instant(declaration.get("acquired_at")) is None:
        return True
    # Null where the recording service writes null for an unknown value.
    if any(
        not (declaration.get(key) is None or isinstance(declaration.get(key), str))
        for key in ("direction", "sensor_id", "operating_state", "signal_unit")
    ):
        return True
    if any(
        not isinstance(declaration.get(key), bool)
        for key in ("timezone_declared", "timestamp_suspect")
    ):
        return True
    if any(
        declaration.get(key) is not None and _as_float(declaration.get(key)) is None
        for key in ("sampling_rate", "rpm", "load")
    ):
        return True
    import_format = declaration.get("import_format")
    if import_format is not None:
        if not isinstance(import_format, Mapping):
            return True
        lines = import_format.get("data_lines")
        if lines is not None and not isinstance(lines, Mapping):
            return True
    return False


_INTEGRITY_COUNTERS = (
    "unreadable_records",
    "asset_id_mismatches",
    "unsupported_schema_versions",
    "malformed_payloads",
    # The store keeps the first event of an id and sets later ones aside
    # unread here, so what they record is unknown.
    "duplicate_event_ids",
)

#: The form of every measurement id (:func:`measurement_id_from_digest`).
_MEASUREMENT_ID = re.compile(r"[0-9a-f]{16}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _integrity_damage(integrity: Mapping[str, Any]) -> int:
    """Records a ledger read or projection set aside as unreadable: their
    measurement id, if any, is unknown, so any of them may be of these
    bytes."""
    return sum(
        count
        for key in _INTEGRITY_COUNTERS
        if isinstance(count := integrity.get(key), int)
    )


@dataclass
class _Views:
    """Every asset ledger, read once per validation into one snapshot, when
    the ledger directory holds at most :data:`HISTORY_ASSET_LIMIT` ledger
    files within the byte and event bounds; the pre-check and the comparison
    with earlier imports read the same views. The measurement index is not
    consulted: it can lag the ledgers. A ledger directory or ledger that
    cannot be read raises."""

    store: LedgerStore
    #: The measurement ids of these bytes and the channel index each derives
    #: from: one per possible column of the file (:data:`MAX_FIELDS`), since
    #: an earlier reading of the same bytes may have split them into more
    #: fields. A measurement id derives from the file bytes and the index.
    candidates: Mapping[str, int] = field(default_factory=dict)
    #: The digest of these bytes.
    sha256: Optional[str] = None
    budget: _Budget = field(default_factory=_Budget)
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Per asset read: records it set aside as unreadable, plus recorded
    #: measurements of these bytes that cannot be read in full or disagree
    #: with their measurement id, plus a bound point it cannot read.
    damaged: dict[str, int] = field(default_factory=dict)
    #: Ledger files with a ledger name that the store cannot list as an
    #: asset ledger and that hold something (an empty file holds nothing).
    unlisted: list[str] = field(default_factory=list)
    #: ``(measurement_id, from_asset, to_asset)`` of each measurement of
    #: these bytes a ledger records as moved, when following its moves never
    #: reaches an asset that holds it as current.
    dangling: list[tuple[str, str, str]] = field(default_factory=list)
    #: Whether a ledger file holds a record nested too deeply to decode.
    too_deep: bool = False
    #: Why the ledgers were not read, when a bound kept them unread.
    incomplete: Optional[str] = None

    def __post_init__(self) -> None:
        # A file whose name is not a ledger id was never written by the store
        # as an asset ledger, so it holds no history and is not read.
        names = sorted(
            name
            for name in self.store._list_root()
            if name.endswith(LEDGER_SUFFIX)
            and not name.startswith("_")
            and _ledger_stem(name[: -len(LEDGER_SUFFIX)])
        )
        try:
            entries = _file_entries(self.store.root, names)
        except OSError as exc:
            raise LedgerError(
                f"Cannot size the asset ledgers: {exc.strerror or exc}. Check that "
                f"the ledger directory (PMM_LEDGER_DIR) is readable."
            ) from exc
        over = _over_file(entries)
        if len(names) > HISTORY_ASSET_LIMIT:
            self.incomplete = (
                f"The ledger directory holds {len(names)} asset ledgers, more than "
                f"the {HISTORY_ASSET_LIMIT} one validation reads"
            )
        elif over is not None:
            self.incomplete = self._file_bound(over)
        elif _over_total(entries, self.budget):
            self.incomplete = self.budget.over_total()
        if self.incomplete is not None:
            return
        for name, _size, regular in entries:
            if not regular:
                self.unlisted.append(name)  # a directory, FIFO or device
                continue
            self._read(name)
            if self.incomplete is not None or self.too_deep:
                self.cache.clear()
                return
        self._check_moves()

    def _file_bound(self, name: str) -> str:
        return (
            f"The ledger file {name} holds more than the "
            f"{_size_words(HISTORY_FILE_BYTES_LIMIT)} one validation reads per file"
        )

    def _read(self, name: str) -> None:
        """Read one ledger file within the bounds and classify it as the
        store's directory report does: an asset ledger when its first
        readable record names its own asset, nothing when it holds nothing
        (an empty file a failed first append left), damage otherwise."""
        stem = name[: -len(LEDGER_SUFFIX)]
        try:
            data, bound = self.budget.read(self.store.root / name)
        except FileNotFoundError:
            return  # gone since it was listed: it holds nothing now
        except _NotRegular:
            self.unlisted.append(name)
            return
        except OSError as exc:
            raise LedgerError(
                f"Cannot read ledger {name}: {exc.strerror or exc}. Check that the "
                f"ledger directory (PMM_LEDGER_DIR) is readable and that {name} is "
                f"a regular file."
            ) from exc
        if bound is not None:
            self.incomplete = (
                self._file_bound(name) if bound == "file" else self.budget.over_total()
            )
            return
        try:
            segments, _tail, _end = split_records(data)
            first = next(
                (
                    obj
                    for obj, _code in map(_decode_record, segments)
                    if obj is not None
                ),
                None,
            )
            result = LedgerStore._parse_records(data, 0, stem)
        except RecursionError:
            self.too_deep = True
            return
        if first is None:
            if result.events or _integrity_damage(result.integrity):
                self.unlisted.append(name)
            return
        if first["asset_id"] != stem:
            self.unlisted.append(name)
            return
        self._add(stem, result)

    def _add(self, asset: str, result: LedgerReadResult) -> None:
        if not self.budget.take_events(len(result.events)):
            self.incomplete = (
                f"The asset ledgers hold more than the {HISTORY_EVENT_LIMIT} "
                f"events one validation reads"
            )
            return
        view = build_asset_view(
            asset, result.events, result.integrity, end_offset=result.end_offset
        )
        damage = sum(self._names_these_bytes(event) for event in result.events)
        self.damaged[asset] = (
            _integrity_damage(_mapping(view.get("integrity"))) + damage
        )
        self.cache[asset] = view

    def _check_moves(self) -> None:
        """Every recorded move of a measurement of these bytes, followed from
        asset to asset, must end at an asset that holds it as current; a
        chain that ends elsewhere, or loops, is incomplete history."""
        moves: dict[tuple[str, str], str] = {}
        for asset, view in self.cache.items():
            for moved in view["reattributed"]:
                moves[(asset, moved["measurement_id"])] = moved["to_asset_id"]
        for (asset, mid), target in sorted(moves.items()):
            if mid not in self.candidates:
                continue
            seen, current = {asset}, target
            while current not in self.holders(mid):
                if current in seen or (current, mid) not in moves:
                    self.dangling.append((mid, asset, target))
                    break
                seen.add(current)
                current = moves[(current, mid)]

    def of(self, asset: str) -> dict[str, Any]:
        """The view of *asset*, read with every ledger when the validation
        started; an empty one for an asset without a ledger, for one whose
        file is damaged (reported as such), or once a bound kept the history
        unread or a ledger could not be decoded (the outcome is invalid
        then). An asset id differing from an existing ledger only by letter
        case raises, as the store's read does."""
        if asset not in self.cache:
            self.store._paths(asset)
            self.store._refuse_case_collision(asset)
            return build_asset_view(asset, [], {})
        return self.cache[asset]

    def _names_these_bytes(self, event: Mapping[str, Any]) -> bool:
        """Whether *event* is damaged history that may concern these bytes.

        Every recorded measurement of these bytes, moved away or not, must
        be readable and agree with its id; one that records these bytes
        under another id disagrees with it; one whose id cannot be read is
        irrelevant only when its digest shows other bytes. An event the
        projection does not read (an unknown type) is damage when it names
        these bytes. Other events are irrelevant by their id and digest; a
        payload the projection cannot read is counted there."""
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            return False
        mid = payload.get("measurement_id")
        digest = _mapping(payload.get("file")).get("content_sha256")
        ours = self.sha256 is not None and digest == self.sha256
        kind = event.get("event_type")
        if kind == EVENT_MEASUREMENT_RECORDED:
            if not isinstance(mid, str):
                return False
            index = self.candidates.get(mid)
            if index is not None:
                return _record_damage(payload, index, self.sha256 or "")
            return ours or not (
                _MEASUREMENT_ID.fullmatch(mid)
                or (isinstance(digest, str) and _SHA256.fullmatch(digest))
            )
        if kind not in EVENT_TYPES:
            return ours or (isinstance(mid, str) and mid in self.candidates)
        return False

    def holders(self, measurement_id: str) -> list[str]:
        """The read assets that hold *measurement_id* as a current
        measurement (one, in a consistent ledger); a snapshot alone holds
        no measurement."""
        return sorted(
            asset
            for asset, view in self.cache.items()
            if _mapping(view["measurements"].get(measurement_id)).get("current")
            is not None
        )


def _precheck(
    run: _Run,
    decode: DecodeResult,
    identities: Mapping[int, Mapping[str, Any]],
    formats: Mapping[int, Mapping[str, Any]],
    views: "_Views",
) -> tuple[list[ChannelPrecheck], dict[int, Optional[dict[str, Any]]]]:
    """Asset, point, supersession, baseline and column-swap, read only."""
    checks: list[ChannelPrecheck] = []
    points: dict[int, Optional[dict[str, Any]]] = {}
    view_of = views.of
    channels = {c["column"]: c for c in run.interpretation["channels"]}
    unreadable_points: set[tuple[str, str]] = set()
    same_bytes: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for column, identity in identities.items():
        asset = identity["asset_id"]
        point_id = identity["measurement_point_id"]
        mid = identity["measurement_id"]
        view = view_of(asset)
        point = (view["points"].get(point_id) or {}).get("current")
        version = (point or {}).get("declaration_version")
        if point is not None and (
            isinstance(version, bool) or not isinstance(version, int) or version < 1
        ):
            # A bound point whose declaration version cannot be read: no
            # point value can be cited from it, and it is damaged history.
            if (asset, point_id) not in unreadable_points:
                unreadable_points.add((asset, point_id))
                views.damaged[asset] = views.damaged.get(asset, 0) + 1
            point = None
        points[column] = point
        holders = views.holders(mid)
        recorded_asset = holders[0] if holders else None
        status = "new"
        changed: list[str] = []
        member = False
        restores = False
        overlaps: Optional[bool] = None
        if recorded_asset is not None:
            slot = view_of(recorded_asset)["measurements"].get(mid)
            current = _mapping(slot).get("current")
            if isinstance(current, Mapping):
                declaration = _mapping(current.get("declaration"))
                new_declaration = {
                    **identity,
                    "sampling_rate": run.interpretation["time_base"][
                        "sampling_rate_hz"
                    ],
                    "signal_unit": channels[column]["unit"],
                    "import_format": formats[column],
                }
                changed = _changed(declaration, new_declaration)
                status = "supersedes" if changed else "already_recorded"
                recorded_format = declaration.get("import_format")
                if status == "supersedes" and isinstance(recorded_format, Mapping):
                    old, new = _range(recorded_format), _range(formats[column])
                    if old is not None and new is not None:
                        overlaps = old[0] <= new[1] and new[0] <= old[1]
                if status == "supersedes":
                    earlier = [
                        _mapping(_mapping(entry).get("declaration")).get(
                            "import_format"
                        )
                        for entry in (_mapping(slot).get("history") or [])[:-1]
                    ]
                    restores = formats[column] in earlier
                recorded_point = declaration.get("measurement_point_id")
                baseline = (
                    (
                        view_of(recorded_asset)["baselines"].get(recorded_point)
                        if isinstance(recorded_point, str)
                        else None
                    )
                    or {}
                ).get("current")
                member = _baseline_member(baseline, mid)
        if asset not in same_bytes:
            same_bytes[asset] = _same_bytes(view, decode.source_sha256)
        swap = _column_swap(run, identity, column, view, same_bytes[asset])
        checks.append(
            ChannelPrecheck(
                column=column,
                measurement_id=mid,
                asset_id=asset,
                asset_status="known" if view["event_count"] else "new",
                point_status="declared" if point is not None else "undeclared",
                point_declaration_version=(point or {}).get("declaration_version"),
                measurement_status=status,
                changed=changed,
                recorded_asset_id=recorded_asset,
                baseline_member=member,
                restores_superseded=restores,
                data_range_overlaps=overlaps,
                column_swap=swap,
            )
        )
        where = st.channel_words(column).capitalize()
        if status == "supersedes":
            note = ""
            if overlaps is False:
                note = (
                    " The new data line range does not overlap the recorded one, so "
                    "the correction describes other samples of the file."
                )
                run.issue(
                    "data_range_not_overlapping",
                    "notice",
                    f"{where}: the new data line range does not overlap the one "
                    f"recorded for measurement {mid}.",
                    field_path="data_lines",
                    column=column,
                )
            run.issue(
                "prior_import_differs",
                "notice",
                f"{where} supersedes the recorded declaration of measurement {mid}, "
                f"changing {changed}; the earlier versions stay in the ledger's "
                f"history.{note}",
                field_path=f"channels[{column}].binding",
                column=column,
            )
            if member:
                run.issue(
                    "supersedes_baseline_member",
                    "notice",
                    f"{where}: measurement {mid} is a member of its point's active "
                    f"baseline, which the assessment then scores with the "
                    f"corrected declaration.",
                    field_path=f"channels[{column}].binding",
                    column=column,
                )
        if restores:
            run.issue(
                "restores_superseded_interpretation",
                "notice",
                f"{where} restores an interpretation an earlier, later superseded "
                f"declaration of measurement {mid} used; the assessment scores the "
                f"snapshot that interpretation already produced. The import still "
                f"proceeds.",
                field_path=f"channels[{column}].binding",
                column=column,
            )
    return checks, points


def _baseline_member(baseline: Optional[Mapping[str, Any]], mid: str) -> bool:
    return baseline is not None and mid in baseline.get("measurement_ids", [])


def _index_words(value: Any) -> str:
    """A recorded channel index for a message: the integer, never ledger
    text."""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return "an unreadable index"


def _same_bytes(
    view: Mapping[str, Any], sha256: str
) -> list[tuple[str, Mapping[str, Any]]]:
    """``(measurement_id, current payload)`` of every current measurement of
    *view* that records these bytes."""
    found = []
    for mid, slot in view["measurements"].items():
        current = _mapping(_mapping(slot).get("current"))
        if current and _mapping(current.get("file")).get("content_sha256") == sha256:
            found.append((mid, current))
    return found


def _column_swap(
    run: _Run,
    identity: Mapping[str, Any],
    column: int,
    view: Mapping[str, Any],
    same_bytes: list[tuple[str, Mapping[str, Any]]],
) -> Optional[str]:
    """Another column of the same bytes recorded at this point and slot
    (*same_bytes*: the asset's current measurements of these bytes)."""
    point_id = identity["measurement_point_id"]
    for mid, current in same_bytes:
        if mid == identity["measurement_id"]:
            continue
        declaration = _mapping(current.get("declaration"))
        if (
            declaration.get("measurement_point_id") != point_id
            or declaration.get("channel_index") == column
            or _slot(declaration) != _slot(identity)
        ):
            continue
        other = _index_words(declaration.get("channel_index"))
        baseline = (view["baselines"].get(point_id) or {}).get("current")
        member = _baseline_member(baseline, mid)
        run.issue(
            "binding_collision",
            "invalid",
            f"{st.channel_words(column).capitalize()} would land in the slot of "
            f"measurement {mid}, recorded from index {other} of the same file at "
            f"the same point and acquisition slot"
            + (", a member of the point's active baseline" if member else "")
            + f". The two would collapse into one slot, and the earlier column "
            f"would win over this one. This version does not swap or rebind "
            f"channels within one import, so this binding is refused while "
            f"measurement {mid} holds the slot. The existing binding conflict "
            f"must be resolved outside this import, through an operator-controlled "
            f"correction, before validation can succeed.",
            field_path=f"channels[{column}].binding",
            column=column,
        )
        return str(mid)
    return None


def _prior_differs(
    run: _Run,
    decode: DecodeResult,
    views: "_Views",
    prior: _Prior,
    current_import_id: str,
    bound: set[str],
) -> None:
    """Earlier interpretations of the same bytes, from every ledger read and
    from the import records, that read them differently. A measurement a
    bound channel supersedes is reported by its pre-check already
    (*bound*)."""
    interpretation = run.interpretation
    file_level = {
        key: interpretation[key] for key in ("dialect", "data_lines", "time_base")
    }
    columns = {c["column"]: c for c in interpretation["channels"]}
    rate = Decimal(interpretation["time_base"]["sampling_rate_hz"])
    reported: set[str] = set(bound)
    for asset in sorted(views.cache):
        view = views.cache[asset]
        for mid, slot in view["measurements"].items():
            current = _mapping(_mapping(slot).get("current"))
            if (
                _mapping(current.get("file")).get("content_sha256")
                != decode.source_sha256
            ):
                continue
            declaration = _mapping(current.get("declaration"))
            recorded_format = declaration.get("import_format")
            differs = False
            recorded_rate = declaration.get("sampling_rate")
            recorded = _as_float(recorded_rate)
            if recorded_rate is not None and (
                recorded is None or Decimal(repr(recorded)) != rate
            ):
                differs = True
            if isinstance(recorded_format, Mapping):
                for key in ("dialect", "data_lines"):
                    if recorded_format.get(key) != file_level[key]:
                        differs = True
            if differs and mid not in reported:
                reported.add(mid)
                run.issue(
                    "prior_import_differs",
                    "notice",
                    f"The ledger records measurement {mid} of these same bytes "
                    f"(index {_index_words(declaration.get('channel_index'))}) with another "
                    f"sampling rate, dialect or data range than this configuration.",
                    field_path="time_base",
                )
    for earlier_id, earlier in prior.records().items():
        if earlier_id == current_import_id:
            continue
        differs = any(earlier[key] != file_level[key] for key in file_level)
        for channel in earlier["channels"]:
            mine = columns.get(channel["column"])
            if mine is not None and any(
                mine[key] != channel[key]
                for key in ("role", "quantity", "unit", "scaling")
            ):
                differs = True
        if differs:
            run.issue(
                "prior_import_differs",
                "notice",
                f"An earlier import of these same bytes (import "
                f"{earlier_id[:12]}) read them differently.",
            )


def _candidate_ids(decode: DecodeResult) -> dict[str, int]:
    """The measurement id of every possible channel index of these bytes, up
    to :data:`MAX_FIELDS`: every measurement ever recorded from them has one
    of these ids, whatever field count its reading split them into."""
    return {
        measurement_id_from_digest(decode.source_sha256, index): index
        for index in range(MAX_FIELDS)
    }


def _check_history(
    run: _Run,
    views: "_Views",
    prior: _Prior,
    candidates: Mapping[str, int],
) -> None:
    """History this validation needed but could not read, or did not read in
    full: ``invalid``, because no user confirmation can stand in for it (a
    storage-integrity problem, not a configuration one). Any of it may
    concern these bytes, and nothing that was read shows it does not.

    Runs after the pre-check (:func:`_precheck`), which adds the bound
    points it cannot read to *views.damaged*; it reads the records itself."""
    prior.records()  # read here too, whatever ran before
    remedy = (
        " A user confirmation cannot resolve this: the history must be readable "
        "before any configuration of these bytes can be ready."
    )
    unread = (
        " A user confirmation cannot resolve this: the history must be read in "
        "full before any configuration of these bytes can be ready."
    )
    if prior.incomplete is not None:
        run.issue(
            "history_incomplete",
            "invalid",
            f"{prior.incomplete}, so earlier imports of these bytes were not "
            f"compared.{_BOUND_NOTE}{unread}",
        )
    if views.incomplete is not None:
        run.issue(
            "history_incomplete",
            "invalid",
            f"{views.incomplete}, so earlier measurements of these bytes were not "
            f"compared.{_BOUND_NOTE}{unread}",
        )
    if prior.refused:
        shown = ", ".join(import_id[:12] for import_id in prior.refused[:5])
        more = f" and {len(prior.refused) - 5} more" if len(prior.refused) > 5 else ""
        run.issue(
            "history_unreadable",
            "invalid",
            f"{len(prior.refused)} import record(s) are damaged ({shown}{more}): "
            f"their interpretation line cannot be read, so whether they concern "
            f"these bytes is unknown. An operator must investigate each one and, "
            f"where appropriate, repair or quarantine it under a documented "
            f"recovery procedure that preserves its original bytes and an audit "
            f"trail. Deleting a damaged record is not an acceptable way to pass "
            f"this check.{remedy}",
        )
    if views.unlisted:
        shown = ", ".join(views.unlisted[:5])
        run.issue(
            "history_unreadable",
            "invalid",
            f"{len(views.unlisted)} file(s) in the ledger directory cannot be read "
            f"as an asset ledger ({shown}), so whether they record these bytes is "
            f"unknown.{remedy}",
        )
    for asset in sorted(views.damaged):
        count = views.damaged[asset]
        if count:
            run.issue(
                "history_unreadable",
                "invalid",
                f"The ledger of asset {asset} has {count} record(s) that cannot be "
                f"read in full, or that disagree with the measurement they "
                f"record, so whether they concern these bytes is unknown.{remedy}",
            )
    if views.too_deep:
        run.issue(
            "history_unreadable",
            "invalid",
            f"A file in the ledger directory holds a record nested too deeply to "
            f"read, so whether it concerns these bytes is unknown.{remedy}",
        )
    for mid, source, target in views.dangling:
        # Only a valid ledger id is quoted: the target is ledger text.
        named = f"asset {target}" if _ledger_stem(target) else "an invalid asset id"
        run.issue(
            "history_unreadable",
            "invalid",
            f"Measurement {mid} of these bytes is recorded by asset {source} as "
            f"moved to {named}, and following its moves reaches no asset that "
            f"holds it as current, so its authoritative asset cannot be "
            f"determined. An operator must investigate and reconcile the "
            f"conflicting records before validation can succeed.{remedy}",
        )
    for mid in sorted(candidates):
        holders = views.holders(mid)
        if len(holders) > 1:
            run.issue(
                "history_unreadable",
                "invalid",
                f"Measurement {mid} of these bytes is recorded as current under "
                f"assets {holders}, so its authoritative asset cannot be "
                f"determined. An operator must investigate and reconcile the "
                f"conflicting records before validation can succeed.{remedy}",
            )
