"""
Append-only asset ledger: event envelope, deterministic ids, canonical JSON,
:class:`LedgerStore` (locked append, tolerant read) and :func:`build_asset_view`.

ISO 13374 Block 3 (state detection), persistence layer. One JSONL file per
asset under the ledger directory (``config.get_ledger_dir()``), written only
by appending whole records and read by a tolerant binary reader that reports
every integrity problem instead of failing or hiding it. Nothing here leaves
the machine, nothing here computes a snapshot, nothing here imports MCP.

Event envelope (one JSON object per line, canonical form)::

    {
      "schema_version": 1,
      "event_id": "<sha256 hex of the envelope without event_id/recorded_at>",
      "event_type": "measurement_recorded",
      "recorded_at": "2026-08-20T11:42:00.123456+00:00",   # server clock, UTC
      "asset_id": "P-101",
      "producer": {"name": "predictive-maintenance-mcp", "version": "0.13.0"},
      "payload": {...}
    }

``event_id`` excludes ``recorded_at``, so an identical retry yields the same
id and the reader absorbs it as a duplicate. The order of truth is the order
in the file; ``recorded_at`` (server) and the declared ``acquired_at`` inside
payloads stay separate, and every history ordering uses ``acquired_at``.

Payload key contracts (documented here, interpreted by :func:`build_asset_view`
and the upper layers; the reader validates only the envelope):

``measurement_point_declared``
    ``measurement_point_id``, ``declaration_version`` (1-based, per point),
    ``bearing_id``, ``fault_orders``, ``machine_group``, ``support_type``,
    ``machine_power_kw``, ``expected_signal_unit``, ``expected_sensor_id``,
    ``expected_direction``, ``nominal_rpm``, ``declared_by``, ``note``,
    ``changed`` (keys changed vs the previous version; empty for version 1).

``measurement_recorded``
    ``measurement_id``, ``measurement_point_id``, ``declaration_version``,
    ``declaration`` ({``asset_id``, ``measurement_point_id``, ``acquired_at``,
    ``timezone_declared``, ``timestamp_suspect``, ``rpm``, ``load``,
    ``operating_state``, ``sensor_id``, ``direction``, ``declared_by``,
    ``sampling_rate``, ``signal_unit``, ``raw_format``, ``channel_index``}),
    ``file`` ({``location``, ``location_is_relative``, ``content_sha256``,
    ``size_bytes``}), ``signal_id``, ``changed`` (keys changed vs the previous
    declaration; a supersession that moves the measurement to another asset
    has ``changed == ["asset_id"]`` and ``declaration.asset_id`` naming the
    new asset), ``locations`` (every location ever declared, most recent
    last).

``health_snapshot_computed``
    ``snapshot_id``, ``measurement_id``, ``measurement_point_id``,
    ``processing`` ({``processing_id``, ``algorithm_version``, ``params``,
    ``effective``, ``provenance``}), ``context_digest``, ``context`` (the
    resolved inputs the digest hashes: rpm and its source, bearing or
    orders, group, support, power, unit, direction),
    ``point_declaration_version``, ``indicators``, ``one_x``, ``bearing``,
    ``iso``, ``missing``.

``baseline_declared``
    ``baseline_id``, ``measurement_point_id``, ``measurement_ids`` (an empty
    list withdraws the baseline), ``members`` ([{``measurement_id``,
    ``declaration_version``, ``point_declaration_version``}]),
    ``declared_by``, ``note``, ``declared_at``.

Byte rules
    The record delimiter is the single byte 0x0A. The writer serializes
    compact canonical JSON with ``ensure_ascii=True`` (a declared deviation
    from RFC 8785, so U+2028 / U+2029 / U+0085 / carriage returns never
    appear literally) and writes ``line + b"\\n"`` with ONE ``os.write`` on an
    unbuffered, binary, ``O_APPEND`` descriptor (opened ``O_RDWR`` rather than
    ``O_WRONLY`` only so the same descriptor can read the last byte of the
    file), checking the byte count and calling ``fsync``. If the file does
    not end with 0x0A (a torn tail from a crash, a NUL run from a filesystem
    hiccup), the writer prepends one 0x0A to that same write, so the torn
    bytes stay isolated as exactly one unreadable record. The reader opens
    in binary, splits on 0x0A only, tolerates one trailing 0x0D per line,
    never uses ``splitlines()``, and treats a final segment without
    terminator as unreadable and NOT consumed (``end_offset`` stops before
    it). Integrity entries carry a record index, a reason code and ids that
    pass the ledger grammar, never record content, and the listing is
    capped at ``MAX_INTEGRITY_ISSUES`` (the counters are always exact).

Locking
    A sidecar ``<asset_id>.jsonl.lock`` next to each ledger (and
    ``_measurements.jsonl.lock`` next to the index) is locked with
    ``fcntl.flock(LOCK_EX | LOCK_NB)`` on POSIX and ``msvcrt.locking(LK_NBLCK)``
    on one byte at offset 0 on Windows, polled every ~10 ms up to a timeout
    measured with ``time.monotonic`` (5 s by default) and refused as
    :class:`LedgerLockTimeout`. An OS lock dies with its process, so there is
    no stale-lock cleanup and no need for one. Readers take no lock.

Invariants
    * No code path rewrites, truncates, renames or deletes a ledger file.
    * The sidecar lock file is never deleted, truncated or replaced: on
      POSIX the lock lives on the inode, and recreating the file would hand
      two processes two different locks on "the same" name.
    * The lock is held from the delta read to the append in
      :meth:`LedgerStore.append_versioned` (version assignment), around the
      write alone in :meth:`LedgerStore.append`, and never during any
      computation: the store computes nothing.
    * A ledger id is validated by ``signal_acquisition.measurement.
      validate_ledger_id`` and every path is resolved through
      ``path_safety.safe_resolve``; ids starting with ``_`` are reserved for
      the index; an id that differs from an existing ledger file only by
      letter case is refused on every operating system (NTFS and APFS would
      silently append to the other file).
    * The ledger directory is created lazily on the first append, never at
      import or construction time.
"""

import errno
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Union

from ..path_safety import safe_resolve
from ..signal_acquisition.measurement import LEDGER_ID_GRAMMAR, validate_ledger_id

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

__all__ = [
    "SCHEMA_VERSION",
    "PRODUCER_NAME",
    "EVENT_MEASUREMENT_POINT_DECLARED",
    "EVENT_MEASUREMENT_RECORDED",
    "EVENT_HEALTH_SNAPSHOT_COMPUTED",
    "EVENT_BASELINE_DECLARED",
    "EVENT_TYPES",
    "ENVELOPE_KEYS",
    "LEDGER_SUFFIX",
    "LOCK_SUFFIX",
    "MEASUREMENT_INDEX_NAME",
    "LOCK_TIMEOUT_SECONDS",
    "LOCK_POLL_SECONDS",
    "MAX_INTEGRITY_ISSUES",
    "LedgerError",
    "LedgerLockTimeout",
    "LedgerWriteError",
    "AppendResult",
    "LedgerReadResult",
    "canonical_json",
    "content_hash",
    "short_id",
    "utc_now_iso",
    "compute_event_id",
    "make_event",
    "LedgerStore",
    "build_asset_view",
]

logger = logging.getLogger(__name__)

#: Schema version written in every envelope; the reader excludes other values.
SCHEMA_VERSION = 1

#: ``producer.name`` of every envelope written by this package.
PRODUCER_NAME = "predictive-maintenance-mcp"

EVENT_MEASUREMENT_POINT_DECLARED = "measurement_point_declared"
EVENT_MEASUREMENT_RECORDED = "measurement_recorded"
EVENT_HEALTH_SNAPSHOT_COMPUTED = "health_snapshot_computed"
EVENT_BASELINE_DECLARED = "baseline_declared"

#: Event types this version writes. The reader keeps unknown types untouched
#: (weak versioning: only additions, never renames); the writer refuses them
#: so a typo cannot silently produce an event every view ignores.
EVENT_TYPES: tuple[str, ...] = (
    EVENT_MEASUREMENT_POINT_DECLARED,
    EVENT_MEASUREMENT_RECORDED,
    EVENT_HEALTH_SNAPSHOT_COMPUTED,
    EVENT_BASELINE_DECLARED,
)

#: Keys every readable record must carry (the envelope, in canonical order).
ENVELOPE_KEYS: tuple[str, ...] = (
    "schema_version",
    "event_id",
    "event_type",
    "recorded_at",
    "asset_id",
    "producer",
    "payload",
)

#: Envelope keys hashed into ``event_id`` (everything but the id and the
#: server timestamp, so a retry of the same fact yields the same id).
_EVENT_ID_KEYS: tuple[str, ...] = (
    "schema_version",
    "event_type",
    "asset_id",
    "producer",
    "payload",
)

LEDGER_SUFFIX = ".jsonl"
LOCK_SUFFIX = ".jsonl.lock"

#: Stem of the global measurement index ``_measurements.jsonl``. The leading
#: underscore is not a valid ledger id, so the index can never collide with
#: an asset and ``list_assets()`` skips it.
MEASUREMENT_INDEX_NAME = "_measurements"
_RESERVED_PREFIX = "_"

#: Default lock wait, measured by this module with ``time.monotonic``.
LOCK_TIMEOUT_SECONDS = 5.0

#: Poll interval while the lock is busy.
LOCK_POLL_SECONDS = 0.01

#: Cap of the ``issues`` list of an integrity block. The counters stay exact
#: whatever the file contains; only the per-record listing is bounded, and
#: ``issues_truncated`` says when it was, so a ledger full of garbage cannot
#: turn a read into an unbounded list (the plan forbids unbounded lists).
MAX_INTEGRITY_ISSUES = 200

_DELIMITER = b"\n"
_O_BINARY = getattr(os, "O_BINARY", 0)

#: errno values that mean "somebody else holds the lock" (poll again) as
#: opposed to a real failure (raise at once). POSIX ``flock`` reports
#: EWOULDBLOCK (== EAGAIN); the Windows CRT reports EACCES, or EDEADLOCK.
_LOCK_BUSY_ERRNOS = frozenset(
    {
        errno.EACCES,
        errno.EAGAIN,
        errno.EWOULDBLOCK,
        errno.EDEADLK,
        getattr(errno, "EDEADLOCK", errno.EDEADLK),
    }
)

#: Reason codes of the integrity block (never the content of a record).
_CODE_EMPTY = "empty_record"
_CODE_INVALID_JSON = "invalid_json"
_CODE_NOT_OBJECT = "not_object"
_CODE_ENVELOPE = "envelope_incomplete"
_CODE_NO_TERMINATOR = "no_terminator"
_CODE_SCHEMA = "unsupported_schema_version"
_CODE_MISMATCH = "asset_id_mismatch"
_CODE_DUPLICATE = "duplicate_event_id"
_CODE_REGRESSION = "recorded_at_regression"

_UNREADABLE_CODES = frozenset(
    {
        _CODE_EMPTY,
        _CODE_INVALID_JSON,
        _CODE_NOT_OBJECT,
        _CODE_ENVELOPE,
        _CODE_NO_TERMINATOR,
    }
)


# ---------------------------------------------------------------------------
# Errors and results
# ---------------------------------------------------------------------------


class LedgerError(ValueError):
    """Base of every ledger refusal; messages state the problem and a remedy."""


class LedgerLockTimeout(LedgerError):
    """The sidecar lock stayed busy for the whole measured timeout."""


class LedgerWriteError(LedgerError):
    """The bytes could not be written (or synced) completely."""


class AppendResult(NamedTuple):
    """Outcome of one append.

    Attributes:
        appended: Whether a record was written.
        event_id: Id of the event handed to the store (empty string when the
            ``build_event`` callback of :meth:`LedgerStore.append_versioned`
            returned ``None``).
        duplicate: True when the store itself saw the same ``event_id`` among
            the records it read under the lock and therefore wrote nothing.
        offset_after: Byte length of the file after the call (a valid offset
            for :meth:`LedgerStore.read_since` and
            :meth:`LedgerStore.append_versioned`).
    """

    appended: bool
    event_id: str
    duplicate: bool
    offset_after: int


class LedgerReadResult(NamedTuple):
    """Outcome of one tolerant read.

    Attributes:
        events: Readable, supported, attributed, deduplicated events in file
            order (unknown ``event_type`` values are kept untouched).
        end_offset: Byte offset just after the last terminated record (an
            unterminated tail is reported but not consumed).
        integrity: Counts per problem class plus an ``issues`` list of
            ``{"index", "code", ...}`` entries (record ordinal within the
            bytes read, reason code, and ids where they help; never content),
            capped at ``MAX_INTEGRITY_ISSUES`` entries with
            ``issues_truncated`` set when the cap was hit.
    """

    events: list[dict[str, Any]]
    end_offset: int
    integrity: dict[str, Any]


# ---------------------------------------------------------------------------
# Canonical JSON and ids
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Serialize *obj* to compact, key-sorted, ASCII-only JSON.

    ``allow_nan=False`` refuses NaN/Infinity (a ValueError, before any byte
    touches the disk); ``ensure_ascii=True`` keeps line separators and every
    non-ASCII character escaped, so a record can never contain a literal
    0x0A, 0x0D, U+2028 or U+2029.

    Raises:
        ValueError: For non-finite floats or non-serializable values (``json``
            reports foreign types such as numpy scalars, sets and datetimes
            as ``TypeError``; a refused input is a ``ValueError`` everywhere
            in this package, and the message names the offending type).
    """
    try:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        )
    except TypeError as exc:
        raise ValueError(
            f"Cannot serialize to canonical JSON: {exc}. Convert numpy scalars "
            f"with .item(), sets to lists and datetimes to ISO 8601 strings "
            f"before building the event."
        ) from exc


def content_hash(obj: Any) -> str:
    """SHA-256 (64 hex chars) of the canonical JSON of *obj*."""
    return hashlib.sha256(canonical_json(obj).encode("ascii")).hexdigest()


def short_id(*parts: str) -> str:
    """First 16 hex chars of the SHA-256 of the ``":"``-joined string parts.

    Raises:
        ValueError: If no part is given or a part is not a string.
    """
    if not parts or any(not isinstance(part, str) for part in parts):
        raise ValueError(
            "short_id needs one or more string parts; convert numbers and "
            "lists to strings explicitly so the id is reproducible."
        )
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()[:16]


def utc_now_iso() -> str:
    """Current instant in UTC, ISO 8601 with microseconds and ``+00:00``."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _producer_version() -> str:
    """The package version, imported lazily.

    ``src/__init__.py`` binds ``__version__`` before importing the server,
    and this module is imported from within that chain; a module-level
    import would depend on that ordering (see ``advisory._build_provenance``).
    ``importlib.metadata`` is avoided on purpose: in a worktree it stamps the
    version of whichever checkout ran ``pip install -e .``.
    """
    from .. import __version__

    return str(__version__)


def compute_event_id(event: dict[str, Any]) -> str:
    """The deterministic id of an envelope: hash of everything but
    ``event_id`` and ``recorded_at``.

    Raises:
        ValueError: If the envelope lacks one of the hashed keys.
    """
    missing = [key for key in _EVENT_ID_KEYS if key not in event]
    if missing:
        raise ValueError(
            f"Cannot compute event_id: envelope lacks {missing}; build events "
            f"with make_event()."
        )
    return content_hash({key: event[key] for key in _EVENT_ID_KEYS})


def make_event(
    event_type: str,
    asset_id: str,
    payload: dict[str, Any],
    *,
    recorded_at: Optional[datetime] = None,
) -> dict[str, Any]:
    """Build a complete envelope around *payload*.

    The payload is round-tripped through canonical JSON, so the returned
    event holds exactly what the file will hold (tuples become lists, keys
    are strings) and ``event_id`` matches the bytes on disk.

    Args:
        event_type: One of ``EVENT_TYPES``.
        asset_id: The ledger the event belongs to (validated grammar).
        payload: JSON-serializable dict (see the module docstring for the
            keys of each event type).
        recorded_at: Server instant, timezone-aware; ``None`` means now.

    Returns:
        The envelope dict with the keys of ``ENVELOPE_KEYS``.

    Raises:
        ValueError: Unknown event type, invalid asset id, non-dict or
            non-serializable payload, naive ``recorded_at``.
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(
            f"Unknown event_type {event_type!r}: this version writes only "
            f"{list(EVENT_TYPES)}."
        )
    validate_ledger_id(asset_id, kind="asset_id")
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid payload for {event_type}: expected a JSON object, got "
            f"{type(payload).__name__}."
        )
    if recorded_at is None:
        recorded = utc_now_iso()
    else:
        if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
            raise ValueError(
                "recorded_at must be timezone-aware (use datetime.now("
                "timezone.utc)); a naive instant cannot be ordered."
            )
        recorded = recorded_at.astimezone(timezone.utc).isoformat(
            timespec="microseconds"
        )
    normalized_payload = json.loads(canonical_json(payload))
    event: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "",
        "event_type": event_type,
        "recorded_at": recorded,
        "asset_id": asset_id,
        "producer": {"name": PRODUCER_NAME, "version": _producer_version()},
        "payload": normalized_payload,
    }
    event["event_id"] = compute_event_id(event)
    return event


# ---------------------------------------------------------------------------
# Locking primitives
# ---------------------------------------------------------------------------


def _try_lock(fd: int) -> None:
    """Non-blocking exclusive lock; raises OSError when busy."""
    if sys.platform == "win32":
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    """Release the lock taken by :func:`_try_lock` (position never moved)."""
    if sys.platform == "win32":
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _empty_integrity() -> dict[str, Any]:
    return {
        "readable_records": 0,
        "unreadable_records": 0,
        "asset_id_mismatches": 0,
        "duplicate_event_ids": 0,
        "recorded_at_regressions": 0,
        "unsupported_schema_versions": 0,
        "issues": [],
        "issues_truncated": False,
    }


def _safe_id(value: Any) -> Optional[str]:
    """*value* when it is a grammatical ledger id, else ``None``.

    Integrity entries carry ids, never content: an ``asset_id`` found in a
    foreign record is echoed only if it could be a ledger id itself.
    """
    try:
        return validate_ledger_id(value, kind="asset_id")
    except ValueError:
        return None


def _parse_instant(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _decode_record(raw: bytes) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Decode one terminated segment into an envelope, or a reason code."""
    if raw.endswith(b"\r"):
        raw = raw[:-1]
    if not raw:
        return None, _CODE_EMPTY
    try:
        obj = json.loads(raw.decode("utf-8"))
    except ValueError:
        return None, _CODE_INVALID_JSON
    if not isinstance(obj, dict):
        return None, _CODE_NOT_OBJECT
    if (
        any(key not in obj for key in ENVELOPE_KEYS)
        or not isinstance(obj["event_id"], str)
        or not isinstance(obj["event_type"], str)
        or not isinstance(obj["asset_id"], str)
        or not isinstance(obj["payload"], dict)
    ):
        return None, _CODE_ENVELOPE
    return obj, None


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class LedgerStore:
    """Append-only JSONL ledgers, one per asset, under one directory.

    Two instances (or two processes) on the same directory see the same
    files: all coordination goes through the sidecar locks, none through
    memory.

    Args:
        root: The ledger directory (``config.get_ledger_dir()`` at the tool
            boundary). Not created here.
        lock_timeout: Seconds to wait for a busy sidecar lock before raising
            :class:`LedgerLockTimeout`.
    """

    def __init__(
        self, root: Union[str, Path], *, lock_timeout: float = LOCK_TIMEOUT_SECONDS
    ) -> None:
        self.root = Path(root)
        self.lock_timeout = float(lock_timeout)

    # -- paths ---------------------------------------------------------------

    def _paths(self, asset_id: object) -> tuple[Path, Path]:
        """Validate *asset_id* and resolve its ledger and lock paths."""
        if isinstance(asset_id, str) and asset_id.startswith(_RESERVED_PREFIX):
            raise ValueError(
                f"Invalid asset_id {asset_id!r}: names starting with '_' are "
                f"reserved for the ledger index. A ledger id must "
                f"{LEDGER_ID_GRAMMAR}."
            )
        validate_ledger_id(asset_id, kind="asset_id")
        ledger = safe_resolve(self.root, f"{asset_id}{LEDGER_SUFFIX}")
        lock = safe_resolve(self.root, f"{asset_id}{LOCK_SUFFIX}")
        return ledger, lock

    def _index_paths(self) -> tuple[Path, Path]:
        return (
            safe_resolve(self.root, f"{MEASUREMENT_INDEX_NAME}{LEDGER_SUFFIX}"),
            safe_resolve(self.root, f"{MEASUREMENT_INDEX_NAME}{LOCK_SUFFIX}"),
        )

    def _existing_ledger_name(self, asset_id: str) -> tuple[Optional[str], ...]:
        """Return ``(exact_name, case_variant_name)`` from the directory listing.

        A listing, not ``exists()``: on NTFS and APFS ``exists("p-101.jsonl")``
        answers True for ``P-101.jsonl``, which is precisely the collision to
        refuse.
        """
        wanted = f"{asset_id}{LEDGER_SUFFIX}"
        folded = wanted.lower()
        exact: Optional[str] = None
        variant: Optional[str] = None
        for name in self._list_root():
            if name == wanted:
                exact = name
            elif name.lower() == folded:
                variant = name
        return exact, variant

    def _list_root(self) -> list[str]:
        """Names in the ledger directory; an absent directory lists nothing.

        Raises:
            LedgerError: If the directory exists but cannot be listed (not a
                directory, no permission).
        """
        try:
            return os.listdir(self.root)
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise LedgerError(
                f"Cannot list the ledger directory: {exc.strerror or exc}. Point "
                f"PMM_LEDGER_DIR at a readable directory."
            ) from exc

    def _refuse_case_collision(self, asset_id: str) -> None:
        exact, variant = self._existing_ledger_name(asset_id)
        if exact is None and variant is not None:
            raise ValueError(
                f"asset_id {asset_id!r} differs only by letter case from the "
                f"existing ledger {variant!r}; asset ids are case-sensitive "
                f"declarations and one filesystem may not tell them apart. Use "
                f"the existing id exactly, or choose a distinct one."
            )

    def _ensure_root(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LedgerWriteError(
                f"Cannot create the ledger directory: {exc.strerror or exc}. "
                f"Point PMM_LEDGER_DIR at a writable local directory."
            ) from exc

    # -- locking -------------------------------------------------------------

    def _acquire_lock(self, lock_path: Path, *, what: str) -> int:
        """Open the sidecar and lock it, polling up to ``lock_timeout``."""
        try:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | _O_BINARY, 0o644)
        except OSError as exc:
            raise LedgerWriteError(
                f"Cannot open the lock file of {what}: {exc.strerror or exc}. "
                f"Check that the ledger directory (PMM_LEDGER_DIR) is writable."
            ) from exc
        deadline = time.monotonic() + self.lock_timeout
        while True:
            try:
                _try_lock(fd)
                return fd
            except OSError as exc:
                if exc.errno not in _LOCK_BUSY_ERRNOS:
                    os.close(fd)
                    raise LedgerWriteError(
                        f"Cannot lock {what}: {exc.strerror or exc}. Check that "
                        f"the ledger directory is on a filesystem that supports "
                        f"file locks."
                    ) from exc
            if time.monotonic() >= deadline:
                os.close(fd)
                raise LedgerLockTimeout(
                    f"Could not acquire the ledger lock of {what} within "
                    f"{self.lock_timeout:g} s: another process is writing it. "
                    f"Retry; if the wait persists, find the process holding "
                    f"{lock_path.name} (the lock is released when it exits)."
                )
            time.sleep(LOCK_POLL_SECONDS)

    @staticmethod
    def _release_lock(fd: int) -> None:
        try:
            _unlock(fd)
        finally:
            os.close(fd)

    # -- raw bytes -----------------------------------------------------------

    @staticmethod
    def _write_line(path: Path, line: bytes, *, what: str) -> int:
        """Append *line* (already terminated) as one write; return the new size.

        A file that does not end with 0x0A gets one 0x0A prepended INSIDE the
        same write, so the torn tail becomes exactly one unreadable record
        and this record starts on its own line.
        """
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | _O_BINARY
        try:
            fd = os.open(path, flags, 0o644)
        except OSError as exc:
            raise LedgerWriteError(
                f"Cannot open {what} for appending: {exc.strerror or exc}. "
                f"Check that the ledger directory (PMM_LEDGER_DIR) is writable."
            ) from exc
        try:
            size = os.lseek(fd, 0, os.SEEK_END)
            prefix = b""
            if size > 0:
                os.lseek(fd, size - 1, os.SEEK_SET)
                if os.read(fd, 1) != _DELIMITER:
                    prefix = _DELIMITER
            data = prefix + line
            written = os.write(fd, data)
            if written != len(data):
                raise LedgerWriteError(
                    f"Short write on {what}: {written} of {len(data)} bytes. "
                    f"The torn tail will be reported as one unreadable record; "
                    f"check free space and retry (the retry is idempotent)."
                )
            os.fsync(fd)
            return os.lseek(fd, 0, os.SEEK_END)
        except LedgerError:
            raise
        except OSError as exc:
            raise LedgerWriteError(
                f"Write to {what} failed: {exc.strerror or exc}. Check free "
                f"space and permissions, then retry (the retry is idempotent: "
                f"a duplicate is absorbed by the reader)."
            ) from exc
        finally:
            os.close(fd)

    def _read_segment(self, path: Path, offset: int) -> bytes:
        """Return the bytes of *path* from *offset* to the end.

        The single read primitive of the store (tests spy on it to prove
        that versioned appends read only the delta).

        Raises:
            LedgerError: If the file is shorter than *offset* (truncated or
                replaced outside the store: the caller's view is unusable),
                or cannot be read at all (permissions, a directory in place
                of the file).
        """
        try:
            with open(path, "rb") as fh:
                size = fh.seek(0, os.SEEK_END)
                if offset > size:
                    raise LedgerError(
                        f"Ledger {path.name} is {size} bytes long but the view "
                        f"was read at offset {offset}: the file was truncated or "
                        f"replaced outside the store. Re-read the asset before "
                        f"appending."
                    )
                fh.seek(offset)
                return fh.read()
        except FileNotFoundError:
            if offset > 0:
                raise LedgerError(
                    f"Ledger {path.name} no longer exists but the view was read "
                    f"at offset {offset}: the file was removed outside the "
                    f"store. Re-read the asset before appending."
                ) from None
            return b""
        except OSError as exc:
            raise LedgerError(
                f"Cannot read ledger {path.name}: {exc.strerror or exc}. Check "
                f"that the ledger directory (PMM_LEDGER_DIR) is readable and "
                f"that {path.name} is a regular file."
            ) from exc

    @staticmethod
    def _parse_records(
        data: bytes, base_offset: int, asset_id: str
    ) -> LedgerReadResult:
        """Split *data* on 0x0A and classify every segment (see module doc)."""
        events: list[dict[str, Any]] = []
        integrity = _empty_integrity()
        issues: list[dict[str, Any]] = integrity["issues"]
        seen: set[str] = set()
        last_recorded: Optional[datetime] = None

        segments = data.split(_DELIMITER)
        tail = segments.pop()
        end_offset = base_offset + len(data) - len(tail)

        def report(index: int, code: str, **extra: Any) -> None:
            if code in _UNREADABLE_CODES:
                integrity["unreadable_records"] += 1
            if len(issues) >= MAX_INTEGRITY_ISSUES:
                integrity["issues_truncated"] = True
                return
            entry: dict[str, Any] = {"index": index, "code": code}
            entry.update(extra)
            issues.append(entry)

        for index, raw in enumerate(segments):
            obj, code = _decode_record(raw)
            if obj is None:
                report(index, code or _CODE_INVALID_JSON)
                continue
            integrity["readable_records"] += 1
            if obj["schema_version"] != SCHEMA_VERSION:
                integrity["unsupported_schema_versions"] += 1
                version = obj["schema_version"]
                report(
                    index,
                    _CODE_SCHEMA,
                    schema_version=(
                        version
                        if isinstance(version, int) and not isinstance(version, bool)
                        else None
                    ),
                )
                continue
            if obj["asset_id"] != asset_id:
                integrity["asset_id_mismatches"] += 1
                report(index, _CODE_MISMATCH, found_asset_id=_safe_id(obj["asset_id"]))
                continue
            event_id = obj["event_id"]
            if event_id in seen:
                integrity["duplicate_event_ids"] += 1
                report(index, _CODE_DUPLICATE, event_id=event_id)
                continue
            seen.add(event_id)
            recorded = _parse_instant(obj["recorded_at"])
            if recorded is not None:
                if last_recorded is not None and recorded < last_recorded:
                    integrity["recorded_at_regressions"] += 1
                    report(index, _CODE_REGRESSION, event_id=event_id)
                last_recorded = recorded
            events.append(obj)

        if tail:
            report(len(segments), _CODE_NO_TERMINATOR)
        return LedgerReadResult(events, end_offset, integrity)

    def _read_from(self, path: Path, offset: int, asset_id: str) -> LedgerReadResult:
        data = self._read_segment(path, offset)
        result = self._parse_records(data, offset, asset_id)
        problems = result.integrity["unreadable_records"] + (
            result.integrity["asset_id_mismatches"]
            + result.integrity["unsupported_schema_versions"]
        )
        if problems:
            logger.warning(
                "Ledger %s: %d record(s) excluded on read (see integrity block)",
                path.name,
                problems,
            )
        return result

    # -- encoding ------------------------------------------------------------

    @staticmethod
    def _encode_event(asset_id: str, event: dict[str, Any]) -> bytes:
        """Validate the envelope against *asset_id* and return its line."""
        if not isinstance(event, dict):
            raise ValueError(
                f"Invalid event: expected the dict returned by make_event(), "
                f"got {type(event).__name__}."
            )
        missing = [key for key in ENVELOPE_KEYS if key not in event]
        if missing:
            raise ValueError(
                f"Invalid event: envelope lacks {missing}; build events with "
                f"make_event()."
            )
        if event["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Invalid event: schema_version {event['schema_version']!r} is "
                f"not {SCHEMA_VERSION}; build events with make_event()."
            )
        if event["event_type"] not in EVENT_TYPES:
            raise ValueError(
                f"Unknown event_type {event['event_type']!r}: this version "
                f"writes only {list(EVENT_TYPES)}."
            )
        if event["asset_id"] != asset_id:
            raise ValueError(
                f"Invalid event: asset_id {event['asset_id']!r} does not match "
                f"the ledger {asset_id!r} it is appended to; a reader would "
                f"exclude it as a mismatch. Build the event for the same asset."
            )
        if not isinstance(event["payload"], dict):
            raise ValueError("Invalid event: payload must be a JSON object.")
        if not isinstance(event["recorded_at"], str):
            raise ValueError("Invalid event: recorded_at must be an ISO 8601 string.")
        expected = compute_event_id(event)
        if event["event_id"] != expected:
            raise ValueError(
                "Invalid event: event_id does not match the envelope content; "
                "rebuild the event with make_event() instead of editing it."
            )
        return canonical_json(event).encode("ascii") + _DELIMITER

    # -- public API: asset ledgers ------------------------------------------

    def append(self, asset_id: str, event: dict[str, Any]) -> AppendResult:
        """Append one event under the sidecar lock (blind append).

        No duplicate scan is performed here: the reader absorbs a duplicate
        ``event_id`` (first copy wins). Use :meth:`append_versioned` when the
        decision to append depends on what the file already contains.

        Raises:
            ValueError: Invalid id, envelope, or a case collision with an
                existing ledger file.
            LedgerLockTimeout: The lock stayed busy for ``lock_timeout``.
            LedgerWriteError: The bytes could not be written or synced.
        """
        ledger, lock = self._paths(asset_id)
        line = self._encode_event(asset_id, event)
        self._refuse_case_collision(asset_id)
        self._ensure_root()
        what = f"asset {asset_id!r}"
        fd = self._acquire_lock(lock, what=what)
        try:
            offset_after = self._write_line(ledger, line, what=what)
        finally:
            self._release_lock(fd)
        logger.debug(
            "Ledger %s: appended %s (%d bytes, offset %d)",
            ledger.name,
            event["event_type"],
            len(line),
            offset_after,
        )
        return AppendResult(True, event["event_id"], False, offset_after)

    def append_versioned(
        self,
        asset_id: str,
        offset: int,
        build_event: Callable[[list[dict[str, Any]]], Optional[dict[str, Any]]],
    ) -> AppendResult:
        """Read the delta after *offset*, let the caller decide, append, all
        under one lock.

        The caller read the asset (``read()``) at *offset* and holds its own
        view. Under the lock the store reads only the bytes appended since,
        passes those delta events to *build_event* (the caller merges them
        into its view and decides), and appends the returned event; ``None``
        means nothing to append. Two processes assigning versions cannot
        both read version n and both append n + 1.

        Args:
            asset_id: The ledger.
            offset: ``end_offset`` of the caller's read (0 for a new asset).
            build_event: Callback receiving the delta events (file order).

        Returns:
            :class:`AppendResult`; ``duplicate`` is True when the returned
            event's id was already among the delta events (nothing written).

        Raises:
            ValueError, LedgerLockTimeout, LedgerWriteError: As
                :meth:`append`; :class:`LedgerError` if the file is shorter
                than *offset*.
        """
        ledger, lock = self._paths(asset_id)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(
                f"Invalid offset {offset!r}: pass the end_offset of a previous "
                f"read (0 for a new asset)."
            )
        self._refuse_case_collision(asset_id)
        self._ensure_root()
        what = f"asset {asset_id!r}"
        fd = self._acquire_lock(lock, what=what)
        try:
            delta = self._read_from(ledger, offset, asset_id)
            event = build_event(delta.events)
            if event is None:
                return AppendResult(False, "", False, delta.end_offset)
            line = self._encode_event(asset_id, event)
            if any(seen["event_id"] == event["event_id"] for seen in delta.events):
                return AppendResult(False, event["event_id"], True, delta.end_offset)
            offset_after = self._write_line(ledger, line, what=what)
        finally:
            self._release_lock(fd)
        logger.debug(
            "Ledger %s: appended %s after reading %d delta event(s)",
            ledger.name,
            event["event_type"],
            len(delta.events),
        )
        return AppendResult(True, event["event_id"], False, offset_after)

    def read(self, asset_id: str) -> LedgerReadResult:
        """Read the whole ledger of *asset_id* without taking a lock.

        A missing file yields an empty result with ``end_offset == 0``.

        Raises:
            ValueError: Invalid id, or an id that differs from an existing
                ledger file only by letter case.
        """
        ledger, _ = self._paths(asset_id)
        self._refuse_case_collision(asset_id)
        return self._read_from(ledger, 0, asset_id)

    def read_since(self, asset_id: str, offset: int) -> LedgerReadResult:
        """Read only the bytes after *offset* (no lock; see :meth:`read`).

        Record indices in ``integrity["issues"]`` are relative to the bytes
        read, and duplicates are detected within the delta only: the merge
        into the caller's view deduplicates across the boundary.
        """
        ledger, _ = self._paths(asset_id)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(
                f"Invalid offset {offset!r}: pass the end_offset of a previous "
                f"read (0 for a new asset)."
            )
        self._refuse_case_collision(asset_id)
        return self._read_from(ledger, offset, asset_id)

    def read_view(self, asset_id: str) -> dict[str, Any]:
        """``read`` followed by :func:`build_asset_view`, in one call."""
        result = self.read(asset_id)
        return build_asset_view(
            asset_id, result.events, result.integrity, end_offset=result.end_offset
        )

    # -- public API: directory --------------------------------------------

    def integrity_report(self) -> dict[str, Any]:
        """Directory-level report: listed assets and the files that are not.

        A ``.jsonl`` file is an asset ledger only if its stem is a valid
        ledger id and its first readable event carries that same
        ``asset_id``. Lock files and names starting with ``_`` are ignored.

        Returns:
            ``{"assets": [...], "unlisted_files": [{"file", "reason"}]}`` with
            reasons ``invalid_stem``, ``no_readable_event`` or
            ``asset_id_mismatch``. Nothing is created.
        """
        assets: list[str] = []
        unlisted: list[dict[str, str]] = []
        for name in sorted(self._list_root()):
            if not name.endswith(LEDGER_SUFFIX) or name.startswith(_RESERVED_PREFIX):
                continue
            stem = name[: -len(LEDGER_SUFFIX)]
            try:
                validate_ledger_id(stem, kind="asset_id")
            except ValueError:
                unlisted.append({"file": name, "reason": "invalid_stem"})
                continue
            first = self._first_readable_event(self.root / name)
            if first is None:
                unlisted.append({"file": name, "reason": "no_readable_event"})
            elif first["asset_id"] != stem:
                unlisted.append({"file": name, "reason": _CODE_MISMATCH})
            else:
                assets.append(stem)
        return {"assets": assets, "unlisted_files": unlisted}

    def list_assets(self) -> list[str]:
        """Sorted ids of the asset ledgers in the directory (see
        :meth:`integrity_report`); an absent directory lists nothing and
        creates nothing."""
        return list(self.integrity_report()["assets"])

    @staticmethod
    def _first_readable_event(path: Path) -> Optional[dict[str, Any]]:
        """Stream *path* until the first envelope parses (binary lines)."""
        try:
            with open(path, "rb") as fh:
                for raw in fh:
                    if not raw.endswith(_DELIMITER):
                        break
                    obj, _ = _decode_record(raw[:-1])
                    if obj is not None:
                        return obj
        except OSError:
            return None
        return None

    # -- public API: global measurement index -------------------------------

    def append_index_entry(
        self, measurement_id: str, asset_id: str, event_id: str, recorded_at: str
    ) -> AppendResult:
        """Append ``measurement_id -> asset_id`` to ``_measurements.jsonl``.

        Same lock discipline and byte rules as an asset ledger, under the
        index's own sidecar. Entries are plain objects with the four keys;
        the last entry for a measurement wins on read.

        Raises:
            ValueError: Invalid ids or non-string fields.
            LedgerLockTimeout, LedgerWriteError: As :meth:`append`.
        """
        if not isinstance(measurement_id, str) or not measurement_id.strip():
            raise ValueError(
                f"Invalid measurement_id {measurement_id!r}: pass the 16-hex id "
                f"of a recorded measurement."
            )
        validate_ledger_id(asset_id, kind="asset_id")
        if not isinstance(event_id, str) or not isinstance(recorded_at, str):
            raise ValueError(
                "Invalid index entry: event_id and recorded_at must be the "
                "strings of the measurement_recorded envelope."
            )
        entry = {
            "measurement_id": measurement_id,
            "asset_id": asset_id,
            "event_id": event_id,
            "recorded_at": recorded_at,
        }
        line = canonical_json(entry).encode("ascii") + _DELIMITER
        index, lock = self._index_paths()
        self._ensure_root()
        what = "the measurement index"
        fd = self._acquire_lock(lock, what=what)
        try:
            offset_after = self._write_line(index, line, what=what)
        finally:
            self._release_lock(fd)
        return AppendResult(True, event_id, False, offset_after)

    def read_index(self) -> dict[str, str]:
        """``{measurement_id: asset_id}`` from the index, last entry wins.

        Tolerant like the ledger reader: unreadable lines, entries without
        the two ids, and entries whose ``asset_id`` fails the grammar are
        skipped (and counted in a debug log), never raised.
        """
        index, _ = self._index_paths()
        data = self._read_segment(index, 0)
        mapping: dict[str, str] = {}
        skipped = 0
        segments = data.split(_DELIMITER)
        segments.pop()
        for raw in segments:
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            try:
                obj = json.loads(raw.decode("utf-8"))
            except ValueError:
                skipped += 1
                continue
            if not isinstance(obj, dict):
                skipped += 1
                continue
            measurement_id = obj.get("measurement_id")
            asset_id = obj.get("asset_id")
            if not isinstance(measurement_id, str) or not isinstance(asset_id, str):
                skipped += 1
                continue
            try:
                validate_ledger_id(asset_id, kind="asset_id")
            except ValueError:
                skipped += 1
                continue
            mapping[measurement_id] = asset_id
        if skipped:
            logger.debug("Measurement index: %d unreadable entry(ies) skipped", skipped)
        return mapping

    def find_measurement_asset(self, measurement_id: str) -> Optional[str]:
        """The asset a measurement was last indexed under, or ``None``."""
        if not isinstance(measurement_id, str) or not measurement_id.strip():
            raise ValueError(
                f"Invalid measurement_id {measurement_id!r}: pass the 16-hex id "
                f"of a recorded measurement."
            )
        return self.read_index().get(measurement_id)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _measurement_slot() -> dict[str, Any]:
    return {
        "history": [],
        "current": None,
        "snapshots": [],
        "snapshots_by_lineage": {},
        "lineage_positions": {},
    }


def build_asset_view(
    asset_id: str,
    events: list[dict[str, Any]],
    integrity: dict[str, Any],
    *,
    end_offset: int = 0,
) -> dict[str, Any]:
    """Project the events of one asset into its current state (pure).

    Events are deduplicated by ``event_id`` (first wins) and filtered on
    ``asset_id`` (foreign events are counted, never used), so a caller can
    pass the events of a read merged with the delta of a later
    ``read_since``. Payloads are interpreted by the key contracts of the
    module docstring; a payload lacking the id it is filed under is counted
    as malformed and skipped.

    Args:
        asset_id: The asset the events belong to.
        events: Envelopes in file order.
        integrity: The integrity block of the read (copied, then extended
            with ``ignored_event_types`` and ``malformed_payloads``).
        end_offset: The ``end_offset`` of the read, carried into the view so
            a versioned append can start from it.

    Returns:
        ``{asset_id, event_count, points, measurements,
        ordered_measurement_ids, reattributed, baselines, integrity,
        end_offset}`` where ``points[point_id] = {history, current}``
        (``current`` is the latest payload as recorded; every ``history``
        entry is a copy of its payload plus ``recorded_at``, the envelope's
        server instant, so a query can tell which measurements were
        acquired before the current declaration),
        ``measurements[measurement_id] = {history, current, snapshots,
        snapshots_by_lineage, lineage_positions}`` (``lineage_positions``
        maps a ``processing_id`` to the position, in the event order, of
        its latest snapshot, so lineages can be ranked by recency across
        measurements; measurements whose latest declaration names another
        asset are moved to ``reattributed`` as
        ``{measurement_id, to_asset_id}``), ``ordered_measurement_ids`` sorts
        the remaining measurements by ``current.declaration.acquired_at``
        (as an instant, naive strings as UTC; unparsable strings after every
        instant, missing values last) then file order, and
        ``baselines[point_id] = {history, current}``
        with ``current`` ``None`` after a withdrawal (empty
        ``measurement_ids``; every history entry carries ``withdrawn``).
    """
    view_integrity = dict(integrity)
    view_integrity.setdefault("asset_id_mismatches", 0)
    ignored: dict[str, int] = {}
    malformed = 0

    points: dict[str, dict[str, Any]] = {}
    measurements: dict[str, dict[str, Any]] = {}
    baselines: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    seen_ids: set[str] = set()
    event_count = 0

    for position, event in enumerate(events):
        event_id = event.get("event_id")
        if isinstance(event_id, str):
            if event_id in seen_ids:
                continue
            seen_ids.add(event_id)
        if event.get("asset_id") != asset_id:
            view_integrity["asset_id_mismatches"] += 1
            continue
        event_count += 1
        event_type = event.get("event_type")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            malformed += 1
            continue

        if event_type == EVENT_MEASUREMENT_POINT_DECLARED:
            point_id = payload.get("measurement_point_id")
            if not isinstance(point_id, str):
                malformed += 1
                continue
            slot = points.setdefault(point_id, {"history": [], "current": None})
            slot["history"].append({**payload, "recorded_at": event.get("recorded_at")})
            slot["current"] = payload

        elif event_type == EVENT_MEASUREMENT_RECORDED:
            measurement_id = payload.get("measurement_id")
            if not isinstance(measurement_id, str):
                malformed += 1
                continue
            slot = measurements.setdefault(measurement_id, _measurement_slot())
            first_seen.setdefault(measurement_id, position)
            slot["history"].append(payload)
            slot["current"] = payload

        elif event_type == EVENT_HEALTH_SNAPSHOT_COMPUTED:
            measurement_id = payload.get("measurement_id")
            processing = payload.get("processing")
            processing_id = (
                processing.get("processing_id")
                if isinstance(processing, dict)
                else None
            )
            if not isinstance(measurement_id, str) or not isinstance(
                processing_id, str
            ):
                malformed += 1
                continue
            slot = measurements.setdefault(measurement_id, _measurement_slot())
            slot["snapshots"].append(payload)
            slot["snapshots_by_lineage"][processing_id] = payload
            slot["lineage_positions"][processing_id] = position

        elif event_type == EVENT_BASELINE_DECLARED:
            point_id = payload.get("measurement_point_id")
            members = payload.get("measurement_ids")
            if not isinstance(point_id, str) or not isinstance(members, list):
                malformed += 1
                continue
            slot = baselines.setdefault(point_id, {"history": [], "current": None})
            entry = dict(payload)
            entry["withdrawn"] = not members
            slot["history"].append(entry)
            slot["current"] = None if not members else payload

        else:
            key = event_type if isinstance(event_type, str) else repr(event_type)
            ignored[key] = ignored.get(key, 0) + 1

    reattributed: list[dict[str, str]] = []
    for measurement_id in list(measurements):
        current = measurements[measurement_id]["current"]
        if current is None:
            continue
        declaration = current.get("declaration")
        declared_asset = (
            declaration.get("asset_id") if isinstance(declaration, dict) else None
        )
        if isinstance(declared_asset, str) and declared_asset != asset_id:
            reattributed.append(
                {"measurement_id": measurement_id, "to_asset_id": declared_asset}
            )
            del measurements[measurement_id]

    def order_key(measurement_id: str) -> tuple[int, float, str, int]:
        # Instants first (compared as instants: a declared offset or a naive
        # string ordered as UTC never reorders by spelling), then unparsable
        # strings in string order, then missing values; file order breaks ties.
        current = measurements[measurement_id]["current"]
        declaration = current.get("declaration")
        acquired = (
            declaration.get("acquired_at") if isinstance(declaration, dict) else None
        )
        position = first_seen.get(measurement_id, len(events))
        instant = _parse_instant(acquired)
        if instant is not None:
            return (0, instant.timestamp(), "", position)
        if isinstance(acquired, str):
            return (1, 0.0, acquired, position)
        return (2, 0.0, "", position)

    ordered = sorted(
        (mid for mid, slot in measurements.items() if slot["current"] is not None),
        key=order_key,
    )

    view_integrity["ignored_event_types"] = ignored
    view_integrity["malformed_payloads"] = malformed
    return {
        "asset_id": asset_id,
        "event_count": event_count,
        "points": points,
        "measurements": measurements,
        "ordered_measurement_ids": ordered,
        "reattributed": reattributed,
        "baselines": baselines,
        "integrity": view_integrity,
        "end_offset": end_offset,
    }
