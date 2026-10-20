"""
ISO 13374 Block 3 — State Detection (asset ledger service).

Orchestration shared by the tool modules, on the precedent of
``decision_support.diagnosis_pipeline``: a plain module inside the package
whose functions receive every dependency by argument. The :class:`LedgerStore`
and the data directory come from the caller (``config.get_ledger_dir()`` and
``config.DATA_DIR`` are read only at the tool boundary, at call time), so
nothing here binds a directory at import. No MCP, no ``models.py``, no
``signal_acquisition.repository``.

What ``load_signal`` does after a successful load, per signal that declares
an identity (:func:`record_measurements`; the asset view is read ONCE per
asset per batch and the global index once per batch):

1. The effective declaration (:func:`build_declaration`: the ``measurement``
   object plus ``sampling_rate``, ``signal_unit``, the effective
   ``raw_format`` and the decoded channel) and the file block
   (:func:`file_block`: location relative to the data directory with POSIX
   separators, or absolute with a flag; content digest; size) are
   fingerprinted against the latest declaration of the same
   ``measurement_id`` in the asset ledger: ``recorded`` (new for this
   asset), ``already_recorded`` (same fingerprint: no event) or
   ``superseded`` (a new ``measurement_recorded`` with
   ``declaration_version + 1`` naming the changed keys; a moved file is a
   supersession with ``changed == ["location"]``).
2. The ``measurement_recorded`` event is appended with
   :meth:`LedgerStore.append_versioned`: the events another process appended
   since the read are merged into the view BEFORE the decision, so two
   processes never assign the same version. If the global index says the
   id was recorded under ANOTHER asset (a mistyped ``asset_id`` later
   corrected), that ledger receives a superseding declaration naming the
   new asset, so its view lists the measurement under ``reattributed``
   without deleting anything; then the index entry is appended.
3. The health snapshot is computed from the signal, the effective
   declaration and the point's current declaration (or none) and appended
   as ``health_snapshot_computed`` only when its deterministic
   ``snapshot_id`` is not already in the view: a location-only or
   timestamp-only supersession yields the same id (nothing appended), an
   rpm correction a new one (the old snapshot stays).
4. Comparability is graded against the point alone (no reference context
   exists at load time) and reported, never stored.

A ledger failure never undoes a load: it is reported as
``ledger_status == "not_recorded"`` with the reason, and a retry is safe
because every event is deduplicated by content. A snapshot failure leaves
the measurement recorded (``snapshot_status == "failed"``); the next load
of the same file appends the missing snapshot and nothing else.

Re-processing (:func:`reprocess_stale_snapshots`, behind
``assess_asset_change(..., reprocess=True)``): a measurement of the point is
STALE when its view holds no snapshot with the current ``processing_id``
AND the ``context_digest`` the current declarations of the measurement and
of its point resolve to. One call of a re-processing sequence walks the
stale measurements in priority order: members of the active reference
(declared baseline or the first N slots), then the last K slots newest
first, then the rest newest first (the duplicates collapsed into another
acquisition slot last); the assessment moves onto the current lineage once
every evaluated slot carries it and no other lineage covering them was
recorded later, and until then it uses the older lineage that still covers
the whole set, or reports ``processing_not_homogeneous`` when none does.
Each measurement goes through three tiers. The view checks
(``snapshot.snapshot_blocker``, shared with the assessment) read the
ledger only. The file checks read metadata only: containment of each
relative location through ``safe_resolve`` (the data directory resolved
once per call; absolute locations as recorded) and one metadata call per
location ever declared, most recent first, with the size compared to the
recorded one; at most ``MAX_FILE_CHECKS_PER_CALL`` per call, a measurement's
checks run in full or not at all, and an error is a per-location outcome,
never a raise. The attempt verifies the content hash against the ledger
before a single sample is decoded, decodes with the recorded declaration
(refusing what the signal store refuses: an empty line between data lines,
a NaN, infinite or non-numeric sample), computes and appends; at most
``limit`` per call (``MAX_REPROCESS_PER_CALL`` for the tool). The call also
stops once it has found ``MAX_LISTED_ITEMS`` failures, so every failure is
listed in full by the call that finds it, with a code of
``REPROCESS_REASON_CODES`` and the ``action_id`` of the action that clears
it, whose text ``needs_action`` gives once (when no recorded location holds
the recorded content, the most recent one decides the code).
A stateless continuation token, which ``next_call`` carries only when an
earlier failure must be passed over, lets the next call start after the
last measurement handled; it is valid only while nothing that decides
which measurements it passed has changed, else the call restarts the
sequence and says so. The old snapshot is never touched and a failure
appends nothing: only derived events are appended, each deduplicated by its
deterministic ``snapshot_id``, so a repeated call on an unchanged ledger
returns the same response.

Declarations and queries (behind the four ledger tools):

* :func:`declare_measurement_point` validates the declared context of a
  point (ids by the ledger grammar, closed vocabularies, the companion's
  free-text rule, positive numbers) and appends a
  ``measurement_point_declared`` event with ``declaration_version`` =
  current + 1 under the versioned-append lock; a declaration identical to
  the current version appends nothing and returns the current version. The
  response names the keys that changed and how many recorded measurements
  of the point now lack a snapshot with the current context and lineage
  (the staleness test shared with the re-processing), with the exact
  re-processing call as the remedy when the view checks find one a call can
  act on that is not collapsed into another acquisition slot.
* :func:`declare_healthy_baseline` validates that every id is a recorded
  measurement of the point, that no two share an acquisition slot, and that
  each is comparable or qualified against the point and against the
  context built from the members themselves (the revalidation the
  assessment performs at query time), then appends a ``baseline_declared``
  event; an empty list with a note withdraws the active baseline.
* :func:`asset_index` reads at most ``max_assets`` ledgers and summarizes
  each (points, counts, latest lineage, baseline present, integrity);
  :func:`asset_history` reads ONE ledger and returns its summary, the last
  N measurements newest first with an indicator preview, the point
  declarations, the baselines, the reattributions and the integrity block.
"""

import logging
import math
import os
import re
import stat
import string
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np

from ..diagnostics.bearing_catalog import lookup_bearing
from ..path_safety import safe_resolve
from ..signal_acquisition.loaders import (
    RAW_SAMPLES_REMEDY,
    TEXT_EXTENSIONS,
    check_text_blank_rows,
    load_raw_binary,
    load_self_describing,
)
from ..signal_acquisition.measurement import (
    MEASUREMENT_DECLARATION_KEYS,
    MEASUREMENT_ID_HEX_CHARS,
    UNIT_FAMILIES,
    digest_file,
    normalize_direction,
    validate_free_text,
    validate_ledger_id,
)
from ..signal_validation import validate_finite_samples
from .assessment import (
    MAX_LISTED_ITEMS,
    MIN_REFERENCE_MEASUREMENTS,
    AssessmentParams,
    ReprocessCoupling,
    _baseline_ids,
    _known_points,
    collect_point_slots,
    current_snapshot_id_of,
    reprocess_action,
    reprocess_call,
)
from .comparability import assess_measurement_comparability, build_reference_context
from .snapshot import (
    BEARING_LABELS,
    VIEW_BLOCKER_CODES,
    SnapshotPolicy,
    compute_health_snapshot,
    processing_id as compute_processing_id,
    recorded_locations,
    snapshot_blocker,
    snapshot_id as compute_snapshot_id,
)
from .store import (
    EVENT_BASELINE_DECLARED,
    EVENT_HEALTH_SNAPSHOT_COMPUTED,
    EVENT_MEASUREMENT_POINT_DECLARED,
    EVENT_MEASUREMENT_RECORDED,
    LEDGER_SUFFIX,
    LedgerError,
    LedgerReadResult,
    LedgerStore,
    build_asset_view,
    canonical_json,
    content_hash,
    make_event,
    short_id,
    utc_now_iso,
)

__all__ = [
    "DECLARATION_KEYS",
    "LEDGER_STATUSES",
    "SNAPSHOT_STATUSES",
    "OUTCOME_KEYS",
    "LOAD_OUTCOME_KEYS",
    "SNAPSHOT_PAYLOAD_KEYS",
    "MAX_REPROCESS_PER_CALL",
    "MAX_FILE_CHECKS_PER_CALL",
    "MAX_CURSOR_CHARS",
    "REPROCESS_OUTCOMES",
    "REPROCESS_REASON_CODES",
    "SEQUENCE_STATES",
    "POINT_DECLARATION_KEYS",
    "POINT_DECLARED_KEYS",
    "POINT_FREE_TEXT_FIELDS",
    "POINT_SIGNAL_UNITS",
    "VALID_MACHINE_GROUPS",
    "VALID_SUPPORT_TYPES",
    "BASELINE_PAYLOAD_KEYS",
    "MAX_BASELINE_MEMBERS",
    "SignalSource",
    "build_declaration",
    "file_block",
    "declaration_fingerprint",
    "changed_keys",
    "record_measurements",
    "resolve_point_context",
    "reprocess_call",
    "parse_reprocess_cursor",
    "reprocess_stale_snapshots",
    "reprocess_for_assessment",
    "declare_measurement_point",
    "declare_healthy_baseline",
    "asset_index",
    "asset_history",
]

logger = logging.getLogger(__name__)

#: Stale measurements ATTEMPTED per :func:`reprocess_stale_snapshots` call
#: (an attempt hashes the recorded locations until one matches, then decodes,
#: computes and appends). Ten keeps one tool call bounded: ten decodes, ten
#: snapshots.
MAX_REPROCESS_PER_CALL = 10

#: File checks per call: each containment check of a relative location and
#: each metadata call, which read no file content. A call with thousands of
#: missing or inaccessible files still ends within this bound.
MAX_FILE_CHECKS_PER_CALL = 500

#: Per-measurement outcomes of a re-processing call.
REPROCESS_OUTCOMES: tuple[str, ...] = ("reprocessed", "up_to_date", "not_reprocessable")

#: Why a measurement is not reprocessable, closed vocabulary, in the order
#: of the tier that finds it: the view checks (``snapshot_blocker``, free),
#: the file checks (metadata only), the attempt; then the code of a
#: recording superseded in its acquisition slot, whatever its own reason.
REPROCESS_REASON_CODES: tuple[str, ...] = VIEW_BLOCKER_CODES + (
    "location_outside_data_dir",
    "file_missing",
    "content_differs",
    "file_inaccessible",
    "undecodable",
    "append_failed",
    "superseded_in_slot",
)

#: Where a re-processing sequence stands after a call: a next call can reach
#: more, every stale measurement that represents an acquisition slot is
#: current, or no further call can progress while some stay stale.
SEQUENCE_STATES: tuple[str, ...] = ("continue", "complete", "exhausted")

#: Longest continuation token accepted; a token stays under 160 characters
#: whatever the number of failures it carries.
MAX_CURSOR_CHARS = 200

#: Version prefix of the token format: a token of another version is
#: refused as malformed.
_CURSOR_VERSION = "r1"

#: Hex characters of the token's digest.
_CURSOR_DIGEST_CHARS = 8

#: One letter per reason code in a token.
_CODE_LETTERS: dict[str, str] = dict(
    zip(REPROCESS_REASON_CODES, string.ascii_lowercase)
)
_LETTER_CODES: dict[str, str] = {letter: code for code, letter in _CODE_LETTERS.items()}

#: A code letter, and one count of a token: a letter and 1 to 999999.
_CODE_LETTER = f"[{''.join(_CODE_LETTERS.values())}]"
_COUNT_ENTRY = f"{_CODE_LETTER}[1-9][0-9]{{0,5}}"
_CURSOR_COUNT = re.compile(f"({_CODE_LETTER})([1-9][0-9]{{0,5}})")

#: ``r1.<last measurement>.<digest>.<codes of the first 50 earlier failures
#: by id>.<count per code over every earlier failure>``.
_CURSOR_PATTERN = re.compile(
    rf"{_CURSOR_VERSION}\.([0-9a-f]{{{MEASUREMENT_ID_HEX_CHARS}}})"
    rf"\.([0-9a-f]{{{_CURSOR_DIGEST_CHARS}}})"
    rf"\.({_CODE_LETTER}{{1,{MAX_LISTED_ITEMS}}})"
    rf"\.((?:{_COUNT_ENTRY}){{1,{len(REPROCESS_REASON_CODES)}}})"
)

#: Locations a not-reprocessable reason names before "and N more".
_MAX_REASON_LOCATIONS = 3

#: Windows error of a metadata call on a name the file system cannot hold.
_ERROR_INVALID_NAME = 123

#: Measurement ids a message names per reason code.
_MAX_MESSAGE_IDS = 5

#: Keys of a ``measurement_point_declared`` payload (the store's contract),
#: in canonical order. ``declaration_version`` and ``changed`` are assigned
#: by :func:`declare_measurement_point`; every other key is declared.
POINT_DECLARATION_KEYS: tuple[str, ...] = (
    "measurement_point_id",
    "declaration_version",
    "bearing_id",
    "fault_orders",
    "machine_group",
    "support_type",
    "machine_power_kw",
    "expected_signal_unit",
    "expected_sensor_id",
    "expected_direction",
    "nominal_rpm",
    "declared_by",
    "note",
    "changed",
)

#: The declared part of a point payload: what two versions are compared on
#: (an identical re-declaration appends nothing).
POINT_DECLARED_KEYS: tuple[str, ...] = tuple(
    key
    for key in POINT_DECLARATION_KEYS
    if key not in ("declaration_version", "changed")
)

#: Free-text fields of a point declaration, under the companion's rule
#: (``validate_free_text``: bounded, one line, never interpreted).
POINT_FREE_TEXT_FIELDS: tuple[str, ...] = (
    "bearing_id",
    "expected_sensor_id",
    "declared_by",
    "note",
)

#: Units a point may expect: the union of the leaf module's unit families,
#: a partition of the repository's ``VALID_SIGNAL_UNITS`` asserted by test
#: (the repository is never imported by this package).
POINT_SIGNAL_UNITS: tuple[str, ...] = tuple(
    unit for units in UNIT_FAMILIES.values() for unit in units
)

#: ISO 20816-3 machine groups and support types a point may declare.
VALID_MACHINE_GROUPS: tuple[int, ...] = (1, 2)
VALID_SUPPORT_TYPES: tuple[str, ...] = ("rigid", "flexible")

#: Keys of a ``baseline_declared`` payload (the store's contract).
BASELINE_PAYLOAD_KEYS: tuple[str, ...] = (
    "baseline_id",
    "measurement_point_id",
    "measurement_ids",
    "members",
    "declared_by",
    "note",
    "declared_at",
)

#: Largest declared baseline accepted (lists stay bounded everywhere).
MAX_BASELINE_MEMBERS = 100

#: Raw decode parameters a recorded ``raw_format`` block may carry, as the
#: decoder's keyword arguments (``sample_format`` is required by it).
_RAW_DECODE_KEYS: tuple[str, ...] = (
    "sample_format",
    "byte_order",
    "n_channels",
    "channel_index",
    "header_offset",
    "scale_factor",
)

#: Keys of the effective declaration recorded in a ``measurement_recorded``
#: payload: the normalized ``measurement`` object (identity minus the two
#: file-derived keys) plus what the signal was loaded with.
DECLARATION_KEYS: tuple[str, ...] = (
    *MEASUREMENT_DECLARATION_KEYS,
    "sampling_rate",
    "signal_unit",
    "raw_format",
    "channel_index",
)

#: The four ledger outcomes of a load.
LEDGER_STATUSES: tuple[str, ...] = (
    "recorded",
    "already_recorded",
    "superseded",
    "not_recorded",
)

#: The four snapshot outcomes of a load: ``partial`` is a snapshot with a
#: non-empty ``missing`` block; ``skipped`` means no snapshot was attempted
#: (ledger failure, undeclared sampling rate, signal not available).
SNAPSHOT_STATUSES: tuple[str, ...] = ("complete", "partial", "failed", "skipped")

#: Keys of every outcome dict returned by :func:`record_measurements`.
OUTCOME_KEYS: tuple[str, ...] = (
    "signal_id",
    "asset_id",
    "measurement_point_id",
    "measurement_id",
    "ledger_status",
    "reason",
    "changed",
    "reattributed_from",
    "declaration_version",
    "snapshot_status",
    "snapshot_id",
    "processing_id",
    "context_digest",
    "comparability",
    "missing",
    "event_ids",
)

#: The outcome fields ``load_signal`` copies into the ``measurement`` block
#: of the ``StoredSignalInfo`` it RETURNS (the repository's cached block
#: keeps the identity only).
LOAD_OUTCOME_KEYS: tuple[str, ...] = (
    "ledger_status",
    "reason",
    "changed",
    "reattributed_from",
    "declaration_version",
    "snapshot_status",
    "snapshot_id",
    "processing_id",
    "comparability",
    "missing",
)

#: Keys of a ``health_snapshot_computed`` payload written here (the store's
#: contract plus the resolved ``context`` block, additive).
SNAPSHOT_PAYLOAD_KEYS: tuple[str, ...] = (
    "snapshot_id",
    "measurement_id",
    "measurement_point_id",
    "processing",
    "context_digest",
    "context",
    "point_declaration_version",
    "indicators",
    "one_x",
    "bearing",
    "iso",
    "missing",
)

#: Where the signal arrays come from: a mapping ``{signal_id: array}`` or a
#: callable ``signal_id -> array`` (``SignalRepository.get_signal`` at the
#: tool boundary). A missing id (``KeyError``/``LookupError``) skips the
#: snapshot with a reason.
SignalSource = Union[Mapping[str, np.ndarray], Callable[[str], np.ndarray]]

#: Exceptions that turn a ledger step into ``not_recorded`` (never raised to
#: the caller: the load already succeeded). ``LedgerError`` is a
#: ``ValueError``; the store also raises plain ``ValueError`` for refusals
#: such as a letter-case collision with an existing ledger.
_LEDGER_FAILURES: tuple[type[BaseException], ...] = (ValueError, OSError)


# ---------------------------------------------------------------------------
# Effective declaration, file block, fingerprint
# ---------------------------------------------------------------------------


def build_declaration(info: Mapping[str, Any]) -> dict[str, Any]:
    """The effective declaration of a loaded signal with an identity block.

    Args:
        info: A ``StoredSignalInfo``-shaped dict (the repository's info dict
            or ``StoredSignalInfo.model_dump()``) whose ``measurement`` key
            holds the identity block.

    Returns:
        A dict with exactly :data:`DECLARATION_KEYS`, in that order: the
        normalized ``measurement`` fields, ``sampling_rate`` (float or
        None), ``signal_unit`` (canonical or None), ``raw_format`` (a copy
        of the effective raw decode parameters, or None for self-describing
        formats) and ``channel_index`` (the decoded channel, 0 by default).

    Raises:
        ValueError: If *info* carries no identity block (a signal loaded
            without a ``measurement`` object has nothing to record).
    """
    identity = info.get("measurement")
    if not isinstance(identity, Mapping):
        raise ValueError(
            f"Signal {info.get('signal_id')!r} declares no measurement identity "
            f"(StoredSignalInfo.measurement is None): only signals whose "
            f'companion carries a "measurement" object are recorded in the '
            f"asset ledger."
        )
    declaration: dict[str, Any] = {
        key: identity.get(key) for key in MEASUREMENT_DECLARATION_KEYS
    }
    rate = info.get("sampling_rate")
    declaration["sampling_rate"] = None if rate is None else float(rate)
    declaration["signal_unit"] = info.get("signal_unit")
    raw_format = info.get("raw_format")
    declaration["raw_format"] = (
        dict(raw_format) if isinstance(raw_format, Mapping) else None
    )
    channel = identity.get("channel_index")
    declaration["channel_index"] = 0 if channel is None else int(channel)
    return declaration


def _relative_location(path: Path, base: Path) -> Optional[str]:
    """POSIX-form path of *path* under *base*, or None when not contained.

    Lexical only (no filesystem access): first on the paths as given, with
    ``..`` segments collapsed, then made absolute against the working
    directory, so an absolute spelling of a file inside the data directory
    records the same relative location as its bare name. Containment uses
    :meth:`pathlib.Path.is_relative_to`, the idiom of ``path_safety`` (a
    ``startswith`` check would accept a sibling ``signals_evil``).
    """
    candidates = (
        (Path(os.path.normpath(path)), Path(os.path.normpath(base))),
        (Path(os.path.abspath(path)), Path(os.path.abspath(base))),
    )
    for candidate, root in candidates:
        if candidate.is_relative_to(root):
            return candidate.relative_to(root).as_posix()
    return None


def file_block(
    filepath: Union[str, Path],
    data_dir: Union[str, Path],
    *,
    content_sha256: str,
    size_bytes: int,
) -> dict[str, Any]:
    """The ``file`` block of a ``measurement_recorded`` payload.

    Args:
        filepath: The loaded file (``StoredSignalInfo.filepath``).
        data_dir: The data directory the location is made relative to
            (``config.DATA_DIR`` read at call time by the caller).
        content_sha256: Full hex digest of the file bytes (from the identity
            block, hashed once at load time).
        size_bytes: Size of the file in bytes.

    Returns:
        ``{"location", "location_is_relative", "content_sha256",
        "size_bytes"}``: the location relative to *data_dir* with ``/``
        separators on every platform when the file lies under it, else the
        absolute path as given with ``location_is_relative`` False.
    """
    path = Path(filepath)
    relative = _relative_location(path, Path(data_dir))
    return {
        "location": str(path) if relative is None else relative,
        "location_is_relative": relative is not None,
        "content_sha256": str(content_sha256),
        "size_bytes": int(size_bytes),
    }


def declaration_fingerprint(
    declaration: Mapping[str, Any], location: Optional[str]
) -> str:
    """Content hash of the canonical effective declaration plus the location.

    Two loads with the same fingerprint are the same declaration
    (``already_recorded``); any difference, including a moved file, is a
    supersession. Key order is irrelevant (canonical JSON).

    Raises:
        ValueError: If a value is not JSON-serializable (a caller bug).
    """
    return content_hash({"declaration": dict(declaration), "location": location})


def changed_keys(
    previous_declaration: Mapping[str, Any],
    previous_location: Optional[str],
    new_declaration: Mapping[str, Any],
    new_location: Optional[str],
) -> list[str]:
    """Sorted keys whose values differ between two effective declarations.

    Values are compared in canonical JSON form (the same form the
    fingerprint hashes, so an empty list here means an equal fingerprint).
    ``"location"`` is listed when the file moved and ``"asset_id"`` when the
    measurement moved to another asset.
    """
    keys = set(previous_declaration) | set(new_declaration)
    changed = {
        key
        for key in keys
        if canonical_json(previous_declaration.get(key))
        != canonical_json(new_declaration.get(key))
    }
    if previous_location != new_location:
        changed.add("location")
    return sorted(changed)


# ---------------------------------------------------------------------------
# Per-batch state
# ---------------------------------------------------------------------------


class _AssetState:
    """The events of one asset, read once per batch, and its view.

    The view is rebuilt (pure, cheap) after every append and after every
    delta merged under the versioned-append lock, so later signals of the
    same batch see earlier ones and a concurrent writer is never missed.
    ``end_offset`` advances only after a versioned append, whose delta read
    guarantees the event list covers the file up to that offset; a blind
    append (snapshot) does not advance it, so the next versioned append
    re-reads the few bytes since and the deduplication absorbs them.
    """

    def __init__(self, asset_id: str, result: LedgerReadResult) -> None:
        self.asset_id = asset_id
        self.events: list[dict[str, Any]] = list(result.events)
        self.integrity = result.integrity
        self.end_offset = result.end_offset
        self.view = self._build()

    def _build(self) -> dict[str, Any]:
        return build_asset_view(
            self.asset_id, self.events, self.integrity, end_offset=self.end_offset
        )

    def merge(self, delta: list[dict[str, Any]]) -> None:
        """Merge the events another process appended since the read."""
        if delta:
            self.events.extend(delta)
            self.view = self._build()

    def appended_versioned(self, event: dict[str, Any], offset_after: int) -> None:
        self.events.append(event)
        self.end_offset = offset_after
        self.view = self._build()

    def appended_blind(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        self.view = self._build()

    def latest_recorded(
        self, measurement_id: str
    ) -> tuple[Optional[dict[str, Any]], int]:
        """Latest ``measurement_recorded`` ENVELOPE of *measurement_id* in this
        ledger and the number of its recorded versions.

        Scans the events (deduplicated by ``event_id``, first wins, filtered
        on ``asset_id`` like the view) rather than the view: a measurement
        whose latest declaration names another asset is dropped from the
        view's ``measurements`` but its history is still here, and a load
        that brings it back must supersede that history, not restart it.
        """
        seen: set[str] = set()
        latest: Optional[dict[str, Any]] = None
        count = 0
        for event in self.events:
            event_id = event.get("event_id")
            if isinstance(event_id, str):
                if event_id in seen:
                    continue
                seen.add(event_id)
            if (
                event.get("event_type") != EVENT_MEASUREMENT_RECORDED
                or event.get("asset_id") != self.asset_id
            ):
                continue
            payload = event.get("payload")
            if isinstance(payload, dict) and payload.get("measurement_id") == (
                measurement_id
            ):
                latest = event
                count += 1
        return latest, count


def _current_point(
    view: Optional[Mapping[str, Any]], point_id: str
) -> Optional[dict[str, Any]]:
    """The current declaration payload of a point in a view, or None."""
    if view is None:
        return None
    slot = view.get("points", {}).get(point_id)
    if not isinstance(slot, dict):
        return None
    current = slot.get("current")
    return dict(current) if isinstance(current, dict) else None


def _snapshot_ids(view: Mapping[str, Any], measurement_id: str) -> set[str]:
    slot = view.get("measurements", {}).get(measurement_id)
    if not isinstance(slot, dict):
        return set()
    return {
        snapshot["snapshot_id"]
        for snapshot in slot.get("snapshots", [])
        if isinstance(snapshot, dict) and isinstance(snapshot.get("snapshot_id"), str)
    }


def _snapshot_payload(
    snapshot_id: str,
    measurement_id: str,
    point_id: str,
    snapshot: Mapping[str, Any],
    point: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """The ``health_snapshot_computed`` payload (exactly
    :data:`SNAPSHOT_PAYLOAD_KEYS`), shared by the load path and the
    re-processing path so the two can never drift."""
    return {
        "snapshot_id": snapshot_id,
        "measurement_id": measurement_id,
        "measurement_point_id": point_id,
        "processing": snapshot["processing"],
        "context_digest": str(snapshot["context_digest"]),
        "context": snapshot["context"],
        "point_declaration_version": (
            None if point is None else point.get("declaration_version")
        ),
        "indicators": snapshot["indicators"],
        "one_x": snapshot["one_x"],
        "bearing": snapshot["bearing"],
        "iso": snapshot["iso"],
        "missing": snapshot["missing"],
    }


def _extend_locations(previous: Any, location: str) -> list[str]:
    """Every location ever declared, most recent last, without a repeat."""
    locations = [str(item) for item in previous] if isinstance(previous, list) else []
    if not locations or locations[-1] != location:
        locations.append(location)
    return locations


def _measurement_payload(
    measurement_id: str,
    point_id: str,
    version: int,
    declaration: Mapping[str, Any],
    file_info: Mapping[str, Any],
    signal_id: str,
    changed: list[str],
    locations: list[str],
) -> dict[str, Any]:
    return {
        "measurement_id": measurement_id,
        "measurement_point_id": point_id,
        "declaration_version": version,
        "declaration": dict(declaration),
        "file": dict(file_info),
        "signal_id": signal_id,
        "changed": list(changed),
        "locations": list(locations),
    }


def _plan(
    state: _AssetState,
    measurement_id: str,
    declaration: Mapping[str, Any],
    file_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Decide the ledger outcome of one declaration against the current
    state (called under the versioned-append lock, after the delta merge).

    Returns:
        ``{"status", "version", "changed", "locations", "previous"}`` where
        ``previous`` is the latest recorded envelope (None for a new id).
    """
    latest, count = state.latest_recorded(measurement_id)
    location = str(file_info["location"])
    if latest is None:
        return {
            "status": "recorded",
            "version": 1,
            "changed": [],
            "locations": [location],
            "previous": None,
        }
    previous = latest["payload"]
    previous_declaration = previous.get("declaration")
    if not isinstance(previous_declaration, dict):
        previous_declaration = {}
    previous_file = previous.get("file")
    previous_location = (
        previous_file.get("location") if isinstance(previous_file, dict) else None
    )
    previous_version = previous.get("declaration_version")
    known_version = (
        previous_version
        if isinstance(previous_version, int) and not isinstance(previous_version, bool)
        else 0
    )
    if declaration_fingerprint(
        previous_declaration, previous_location
    ) == declaration_fingerprint(declaration, location):
        return {
            "status": "already_recorded",
            "version": max(known_version, count),
            "changed": [],
            "locations": _extend_locations(previous.get("locations"), location),
            "previous": latest,
        }
    return {
        "status": "superseded",
        "version": max(known_version, count) + 1,
        "changed": changed_keys(
            previous_declaration, previous_location, declaration, location
        ),
        "locations": _extend_locations(previous.get("locations"), location),
        "previous": latest,
    }


def _new_outcome(
    info: Mapping[str, Any], identity: Mapping[str, Any], processing: str
) -> dict[str, Any]:
    return {
        "signal_id": str(info.get("signal_id")),
        "asset_id": identity["asset_id"],
        "measurement_point_id": identity["measurement_point_id"],
        "measurement_id": identity["measurement_id"],
        "ledger_status": "not_recorded",
        "reason": None,
        "changed": [],
        "reattributed_from": None,
        "declaration_version": None,
        "snapshot_status": "skipped",
        "snapshot_id": None,
        "processing_id": processing,
        "context_digest": None,
        "comparability": {"grade": None, "qualifications": []},
        "missing": {},
        "event_ids": [],
    }


def _add_reason(outcome: dict[str, Any], text: str) -> None:
    outcome["reason"] = (
        text if outcome["reason"] is None else (f"{outcome['reason']}; {text}")
    )


def _describe(exc: BaseException) -> str:
    """One-line description of a failure (``strerror`` for bare OS errors)."""
    if isinstance(exc, OSError) and not isinstance(exc, LedgerError):
        detail = exc.strerror or str(exc)
        return f"{type(exc).__name__}: {detail}"
    text = str(exc)
    return text if text else type(exc).__name__


class _Batch:
    """One :func:`record_measurements` call: store, directories, caches."""

    def __init__(
        self,
        store: LedgerStore,
        data_dir: Path,
        signals: SignalSource,
        policy: Optional[SnapshotPolicy],
        provenance_overrides: Optional[Mapping[str, str]],
    ) -> None:
        self.store = store
        self.data_dir = data_dir
        self.signals = signals
        self.policy = policy
        self.provenance_overrides = provenance_overrides
        self.processing_id = compute_processing_id(policy)
        self.states: dict[str, _AssetState] = {}
        self.index: Optional[dict[str, str]] = None

    # -- caches (one read per asset, one per index, per batch) ---------------

    def state_for(self, asset_id: str) -> _AssetState:
        state = self.states.get(asset_id)
        if state is None:
            state = _AssetState(asset_id, self.store.read(asset_id))
            self.states[asset_id] = state
        return state

    def index_for(self) -> dict[str, str]:
        if self.index is None:
            self.index = self.store.read_index()
        return self.index

    def signal_for(self, signal_id: str) -> np.ndarray:
        if callable(self.signals):
            return self.signals(signal_id)
        return self.signals[signal_id]

    # -- one signal ----------------------------------------------------------

    def record(self, info: Mapping[str, Any]) -> dict[str, Any]:
        declaration = build_declaration(info)
        identity = info["measurement"]
        outcome = _new_outcome(info, identity, self.processing_id)
        signal_id = outcome["signal_id"]
        asset_id = outcome["asset_id"]
        measurement_id = outcome["measurement_id"]

        state: Optional[_AssetState] = None
        file_info: Optional[dict[str, Any]] = None
        try:
            file_info = self._file_block(info, identity)
            state = self._record_declaration(outcome, declaration, file_info, signal_id)
        except _LEDGER_FAILURES as exc:
            outcome["ledger_status"] = "not_recorded"
            _add_reason(outcome, _describe(exc))
            logger.warning(
                "Measurement %s of %s (signal %r) not recorded in the asset "
                "ledger: %s",
                measurement_id,
                asset_id,
                signal_id,
                _describe(exc),
            )
            state = self.states.get(asset_id)
        else:
            logger.info(
                "Measurement %s of %s (signal %r): %s, declaration version %s",
                measurement_id,
                asset_id,
                signal_id,
                outcome["ledger_status"],
                outcome["declaration_version"],
            )

        view = None if state is None else state.view
        point = _current_point(view, outcome["measurement_point_id"])
        if outcome["ledger_status"] != "not_recorded" and state is not None:
            self._snapshot(outcome, state, point, declaration, signal_id)

        measurement: dict[str, Any] = {
            "measurement_id": measurement_id,
            "declaration": declaration,
        }
        if file_info is not None:
            measurement["file"] = file_info
        assessment = assess_measurement_comparability(measurement, point)
        outcome["comparability"] = {
            "grade": assessment["grade"],
            "qualifications": assessment["qualifications"],
        }
        return outcome

    def _file_block(
        self, info: Mapping[str, Any], identity: Mapping[str, Any]
    ) -> dict[str, Any]:
        digest = identity.get("content_sha256")
        size = identity.get("size_bytes")
        if not isinstance(digest, str) or not digest or size is None:
            # An identity block built before the digest was kept on it:
            # hash the file now (the only second read of the file).
            digest, size = digest_file(Path(str(info["filepath"])))
        return file_block(
            str(info["filepath"]),
            self.data_dir,
            content_sha256=digest,
            size_bytes=int(size),
        )

    def _record_declaration(
        self,
        outcome: dict[str, Any],
        declaration: Mapping[str, Any],
        file_info: dict[str, Any],
        signal_id: str,
    ) -> _AssetState:
        asset_id = outcome["asset_id"]
        point_id = outcome["measurement_point_id"]
        measurement_id = outcome["measurement_id"]
        state = self.state_for(asset_id)
        index = self.index_for()
        decided: dict[str, Any] = {}

        def build_event(delta: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
            state.merge(delta)
            plan = _plan(state, measurement_id, declaration, file_info)
            decided.update(plan)
            if plan["status"] == "already_recorded":
                return None
            payload = _measurement_payload(
                measurement_id,
                point_id,
                plan["version"],
                declaration,
                file_info,
                signal_id,
                plan["changed"],
                plan["locations"],
            )
            event = make_event(EVENT_MEASUREMENT_RECORDED, asset_id, payload)
            decided["event"] = event
            return event

        result = self.store.append_versioned(asset_id, state.end_offset, build_event)
        event = decided.get("event")
        if result.appended and event is not None:
            state.appended_versioned(event, result.offset_after)
            outcome["event_ids"].append(event["event_id"])
            recorded: Optional[dict[str, Any]] = event
        else:
            # Nothing written: the same declaration (or, with ``duplicate``,
            # the identical event appended by another process meanwhile).
            state.end_offset = result.offset_after
            decided["status"] = "already_recorded"
            decided["changed"] = []
            recorded, _ = state.latest_recorded(measurement_id)
        outcome["ledger_status"] = decided["status"]
        outcome["declaration_version"] = decided["version"]
        outcome["changed"] = list(decided["changed"])

        other = index.get(measurement_id)
        if other is not None and other != asset_id:
            self._reattribute(outcome, other, declaration, file_info, signal_id)

        if (result.appended or index.get(measurement_id) != asset_id) and (
            recorded is not None
        ):
            self.store.append_index_entry(
                measurement_id,
                asset_id,
                str(recorded["event_id"]),
                str(recorded["recorded_at"]),
            )
            index[measurement_id] = asset_id
        return state

    def _reattribute(
        self,
        outcome: dict[str, Any],
        other: str,
        declaration: Mapping[str, Any],
        file_info: dict[str, Any],
        signal_id: str,
    ) -> None:
        """Supersede the measurement in the ledger of the asset it was
        recorded under by mistake: a declaration naming the new asset, so
        that ledger's view lists it under ``reattributed``."""
        asset_id = outcome["asset_id"]
        point_id = outcome["measurement_point_id"]
        measurement_id = outcome["measurement_id"]
        other_state = self.state_for(other)
        decided: dict[str, Any] = {}

        def build_event(delta: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
            other_state.merge(delta)
            latest, count = other_state.latest_recorded(measurement_id)
            if latest is None:
                return None  # a stale index entry: nothing to supersede
            decided["held"] = True
            previous = latest["payload"]
            previous_declaration = previous.get("declaration")
            if not isinstance(previous_declaration, dict):
                previous_declaration = {}
            if previous_declaration.get("asset_id") == asset_id:
                return None  # already reattributed (a retry)
            previous_file = previous.get("file")
            previous_location = (
                previous_file.get("location")
                if isinstance(previous_file, dict)
                else None
            )
            previous_version = previous.get("declaration_version")
            known_version = (
                previous_version
                if isinstance(previous_version, int)
                and not isinstance(previous_version, bool)
                else 0
            )
            changed = changed_keys(
                previous_declaration,
                previous_location,
                declaration,
                str(file_info["location"]),
            )
            if "asset_id" not in changed:
                changed = sorted({*changed, "asset_id"})
            payload = _measurement_payload(
                measurement_id,
                point_id,
                max(known_version, count) + 1,
                declaration,
                file_info,
                signal_id,
                changed,
                _extend_locations(
                    previous.get("locations"), str(file_info["location"])
                ),
            )
            event = make_event(EVENT_MEASUREMENT_RECORDED, other, payload)
            decided["event"] = event
            return event

        result = self.store.append_versioned(other, other_state.end_offset, build_event)
        event = decided.get("event")
        if result.appended and event is not None:
            other_state.appended_versioned(event, result.offset_after)
            outcome["event_ids"].append(event["event_id"])
        else:
            other_state.end_offset = result.offset_after
        if decided.get("held"):
            outcome["reattributed_from"] = other
            logger.info(
                "Measurement %s reattributed from %s to %s",
                measurement_id,
                other,
                asset_id,
            )

    def _snapshot(
        self,
        outcome: dict[str, Any],
        state: _AssetState,
        point: Optional[dict[str, Any]],
        declaration: Mapping[str, Any],
        signal_id: str,
    ) -> None:
        asset_id = outcome["asset_id"]
        measurement_id = outcome["measurement_id"]
        fs = declaration.get("sampling_rate")
        if fs is None or not fs > 0:
            outcome["snapshot_status"] = "skipped"
            _add_reason(
                outcome,
                "snapshot skipped: sampling_rate not declared; re-load the file "
                "with load_signal(sampling_rate=..., overwrite=True) or declare "
                "sampling_rate in the companion (the new declaration supersedes "
                "this one and the snapshot is computed then)",
            )
            return
        try:
            signal = np.asarray(self.signal_for(signal_id))
        except LookupError:
            outcome["snapshot_status"] = "skipped"
            _add_reason(
                outcome,
                f"snapshot skipped: signal {signal_id!r} is not available (evicted "
                f"from the repository cache?); re-load the file to compute it",
            )
            return
        try:
            snapshot = compute_health_snapshot(
                signal,
                float(fs),
                declaration=declaration,
                point=point,
                policy=self.policy,
                provenance_overrides=self.provenance_overrides,
            )
        except Exception as exc:  # reported, never raised: the load succeeded
            outcome["snapshot_status"] = "failed"
            _add_reason(outcome, f"snapshot failed: {_describe(exc)}")
            logger.warning(
                "Health snapshot of measurement %s (signal %r) failed: %s",
                measurement_id,
                signal_id,
                _describe(exc),
                exc_info=True,
            )
            return

        processing = str(snapshot["processing"]["processing_id"])
        digest = str(snapshot["context_digest"])
        snapshot_id = compute_snapshot_id(measurement_id, processing, digest)
        outcome["snapshot_id"] = snapshot_id
        outcome["processing_id"] = processing
        outcome["context_digest"] = digest
        outcome["missing"] = dict(snapshot["missing"])
        outcome["snapshot_status"] = "partial" if snapshot["missing"] else "complete"
        if snapshot_id in _snapshot_ids(state.view, measurement_id):
            return

        payload = _snapshot_payload(
            snapshot_id,
            measurement_id,
            outcome["measurement_point_id"],
            snapshot,
            point,
        )
        try:
            event = make_event(EVENT_HEALTH_SNAPSHOT_COMPUTED, asset_id, payload)
            result = self.store.append(asset_id, event)
        except _LEDGER_FAILURES as exc:
            outcome["snapshot_status"] = "failed"
            _add_reason(
                outcome, f"snapshot computed but not appended: {_describe(exc)}"
            )
            logger.warning(
                "Health snapshot %s of measurement %s not appended: %s",
                snapshot_id,
                measurement_id,
                _describe(exc),
            )
            return
        if result.appended:
            state.appended_blind(event)
            outcome["event_ids"].append(event["event_id"])


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def record_measurements(
    infos: Sequence[Mapping[str, Any]],
    signals: SignalSource,
    *,
    store: LedgerStore,
    data_dir: Union[str, Path],
    policy: Optional[SnapshotPolicy] = None,
    provenance_overrides: Optional[Mapping[str, str]] = None,
) -> list[dict[str, Any]]:
    """Record loaded signals in the asset ledger and derive their snapshots.

    See the module docstring for the flow. Every info must carry an
    identity block (``measurement``); the caller filters the signals loaded
    without one. The asset view is read once per asset and the global
    index once per call, whatever the batch size.

    Args:
        infos: ``StoredSignalInfo``-shaped dicts of the loaded signals, in
            load order.
        signals: The arrays, as a mapping ``{signal_id: array}`` or a
            callable ``signal_id -> array``.
        store: The ledger store (``LedgerStore(get_ledger_dir())`` at the
            tool boundary).
        data_dir: The data directory file locations are made relative to
            (``config.DATA_DIR`` read at call time).
        policy: Snapshot policy; None means the defaults.
        provenance_overrides: See ``snapshot.collect_provenance``.

    Returns:
        One outcome dict per info, in order, with exactly
        :data:`OUTCOME_KEYS`::

            {
              "signal_id", "asset_id", "measurement_point_id", "measurement_id",
              "ledger_status":   "recorded" | "already_recorded" | "superseded"
                                 | "not_recorded",
              "reason":          None or the reason of a non-nominal status,
              "changed":         keys that differ from the previous declaration,
              "reattributed_from": asset the measurement was recorded under
                                 by mistake, or None,
              "declaration_version": int or None,
              "snapshot_status": "complete" | "partial" | "failed" | "skipped",
              "snapshot_id", "processing_id", "context_digest",
              "comparability":   {"grade", "qualifications"} against the point,
              "missing":         {block: {"reason", "remedy"}} of the snapshot,
              "event_ids":       ids of the events appended by this call,
            }

    Raises:
        ValueError: If an info carries no identity block (a caller bug).
            Ledger and snapshot failures never raise: they are reported in
            the outcome.
    """
    batch = _Batch(store, Path(data_dir), signals, policy, provenance_overrides)
    return [batch.record(info) for info in infos]


def resolve_point_context(
    store: LedgerStore, asset_id: str, measurement_point_id: str
) -> Optional[dict[str, Any]]:
    """The current declaration of a measurement point, or None.

    For the tools that default their diagnostic parameters to the declared
    context of the point: the payload of the latest
    ``measurement_point_declared`` event of *asset_id* for
    *measurement_point_id* (``bearing_id``, ``fault_orders``,
    ``machine_group``, ``support_type``, ``machine_power_kw``,
    ``nominal_rpm``, ``expected_*``, ``declaration_version``, ...).

    Args:
        store: The ledger store.
        asset_id: The asset (validated by the store).
        measurement_point_id: The point.

    Returns:
        A copy of the current declaration payload, or None when the asset
        has no ledger or the point was never declared.

    Raises:
        ValueError: An invalid ``asset_id`` or a letter-case collision with
            an existing ledger; ``LedgerError`` when the ledger cannot be
            read.
    """
    return _current_point(store.read_view(asset_id), measurement_point_id)


# ---------------------------------------------------------------------------
# Re-processing of stale snapshots
# ---------------------------------------------------------------------------


def _snapshot_is_current(
    view: Mapping[str, Any],
    measurement_id: str,
    declaration: Mapping[str, Any],
    point: Optional[Mapping[str, Any]],
    processing: str,
) -> bool:
    """Whether the view holds the snapshot the current context and lineage
    would produce for the measurement (the staleness test shared by the
    re-processing and the point declaration). A malformed declaration is
    not current: the re-processing reports it."""
    try:
        expected = current_snapshot_id_of(
            dict(declaration),
            None if point is None else dict(point),
            processing=processing,
            measurement_id=measurement_id,
        )
    except ValueError:
        return False
    return expected in _snapshot_ids(view, measurement_id)


class _ReprocessPlan(NamedTuple):
    """The measurements of a point in re-processing priority, with what the
    groups and the actions read from the same staging."""

    order: list[str]
    kept_by_duplicate: dict[str, str]
    baseline_members: frozenset[str]


def _reprocess_plan(
    view: Mapping[str, Any],
    measurement_point_id: str,
    params: AssessmentParams,
) -> _ReprocessPlan:
    """Measurement ids of the point in re-processing priority: members of
    the active reference (declared baseline, else the first N usable slots
    by ``acquired_at``), then the last K slots newest first, then the rest
    newest first, then the non-comparable ones newest first (they are
    re-processed last: a corrected declaration may make them usable), then
    the duplicates collapsed into another slot newest first (they never
    represent a slot, but they are measurements of the point and the point
    declaration counts them as stale, so the sequence reaches them). Also
    the measurement representing the slot of each collapsed duplicate and
    the members of the active declared baseline."""
    staged = collect_point_slots(dict(view), measurement_point_id, params=params)
    usable = staged["usable"]
    head = staged["reference"]["slots"] or usable[: params.reference_measurements]
    ordered: list[str] = []
    seen: set[str] = set()

    def add(slots: Sequence[Mapping[str, Any]]) -> None:
        for slot in slots:
            measurement_id = str(slot["measurement_id"])
            if measurement_id not in seen:
                seen.add(measurement_id)
                ordered.append(measurement_id)

    add(head)
    add(list(reversed(usable[-params.last_k :])))
    add(list(reversed(usable)))
    add(list(reversed(staged["slots"])))
    add(list(reversed(staged["collapsed"])))
    kept = {
        str(entry["measurement_id"]): str(entry["kept_measurement_id"])
        for entry in staged["collapsed"]
    }
    return _ReprocessPlan(
        ordered, kept, _baseline_ids(dict(view), measurement_point_id)
    )


def _current_locations(view: Mapping[str, Any]) -> dict[str, set[str]]:
    """The recorded content hashes of the asset's measurements, by the
    location each one records as its current one."""
    found: dict[str, set[str]] = {}
    for slot in (view.get("measurements") or {}).values():
        current = slot.get("current") if isinstance(slot, dict) else None
        file_info = current.get("file") if isinstance(current, dict) else None
        if (
            isinstance(file_info, dict)
            and file_info.get("location") is not None
            and file_info.get("content_sha256")
        ):
            found.setdefault(str(file_info["location"]), set()).add(
                str(file_info["content_sha256"])
            )
    return found


def _resolved_dir(directory: Path) -> Path:
    """*directory* resolved once per call (as given when that fails: each
    relative location then reports the failure through its containment
    check)."""
    try:
        return directory.resolve()
    except (OSError, RuntimeError):
        return directory


#: A location's outcome: the reason code and the clause of the reason.
_LocationOutcome = tuple[str, str]


def _file_checks_needed(locations: Sequence[str]) -> int:
    """The file checks :func:`_check_locations` spends at most on
    *locations*: a containment check and a metadata call for a relative
    location, a metadata call for an absolute one."""
    return sum(1 if Path(location).is_absolute() else 2 for location in locations)


def _missing(location: str) -> _LocationOutcome:
    return ("file_missing", f"file not found at {location}")


def _inaccessible(location: str, exc: BaseException) -> _LocationOutcome:
    return ("file_inaccessible", f"cannot read {location}: {_describe(exc)}")


def _check_locations(
    locations: Sequence[str],
    data_dir: Path,
    *,
    recorded_size: Optional[int],
) -> tuple[list[tuple[str, Path]], dict[str, _LocationOutcome], int]:
    """The file checks of one measurement: containment of each relative
    location and one metadata call per location, most recent first, with
    the size compared to the recorded one. No file content is read.

    Returns:
        ``(passed, failed, checks)``: the ``(location, path)`` pairs whose
        file exists with the recorded size (or no size is recorded), most
        recent first; the outcome of every other location; the file checks
        spent.
    """
    passed: list[tuple[str, Path]] = []
    failed: dict[str, _LocationOutcome] = {}
    checks = 0
    for location in locations:
        candidate = Path(location)
        if not candidate.is_absolute():
            checks += 1
            try:
                candidate = safe_resolve(data_dir, location)
            except ValueError:
                failed[location] = (
                    "location_outside_data_dir",
                    f"location escapes the data directory: {location}",
                )
                continue
            except (OSError, RuntimeError) as exc:
                failed[location] = _inaccessible(location, exc)
                continue
        checks += 1
        try:
            info = os.stat(candidate)
        except (FileNotFoundError, NotADirectoryError, ValueError):
            failed[location] = _missing(location)
            continue
        except (OSError, RuntimeError) as exc:
            # A name Windows cannot hold is no file, not an access problem.
            invalid_name = getattr(exc, "winerror", None) == _ERROR_INVALID_NAME
            failed[location] = (
                _missing(location) if invalid_name else _inaccessible(location, exc)
            )
            continue
        if not stat.S_ISREG(info.st_mode):
            failed[location] = _missing(location)
            continue
        if recorded_size is not None and info.st_size != recorded_size:
            failed[location] = (
                "content_differs",
                f"content differs at {location}: {info.st_size} bytes where "
                f"{recorded_size} were recorded",
            )
            continue
        passed.append((location, candidate))
    return passed, failed, checks


def _verify_content(
    passed: Sequence[tuple[str, Path]], expected: str
) -> tuple[Optional[tuple[str, Path]], dict[str, _LocationOutcome]]:
    """Hash the locations that passed the file checks, most recent first,
    until one holds the recorded content."""
    failed: dict[str, _LocationOutcome] = {}
    for location, path in passed:
        try:
            digest, _ = digest_file(path)
        except (OSError, RuntimeError) as exc:
            failed[location] = _inaccessible(location, exc)
            continue
        if digest != expected:
            failed[location] = ("content_differs", f"content differs at {location}")
            continue
        return (location, path), failed
    return None, failed


def _located_failure(
    locations: Sequence[str],
    failed: Mapping[str, _LocationOutcome],
    taken: Callable[[], frozenset[str]],
) -> tuple[str, str]:
    """The code and reason of a measurement no recorded location holds: the
    most recent location decides the code, and the reason lists the
    outcome of each location, most recent first.

    Other content at a location that another measurement with a different
    recorded content hash records as its current one (*taken*, read from
    the ledger) is that measurement's file: this measurement's file is
    missing there, there is no changed file to decide about.
    """
    reused = taken()
    outcomes: dict[str, _LocationOutcome] = {
        location: (
            (
                "file_missing",
                f"file not found at {location}: another recorded measurement "
                f"is there now",
            )
            if failed[location][0] == "content_differs" and location in reused
            else failed[location]
        )
        for location in locations
    }
    clauses = [outcomes[location][1] for location in locations]
    reason = "; ".join(clauses[:_MAX_REASON_LOCATIONS])
    if len(clauses) > _MAX_REASON_LOCATIONS:
        reason += f"; and {len(clauses) - _MAX_REASON_LOCATIONS} more location(s)"
    return outcomes[locations[0]][0], reason


#: The re-load hint the reason of two view-check codes ends with.
_VIEW_REASON_SUFFIX = {
    "no_valid_sampling_rate": (
        "; re-load the file with a declared sampling rate (the new declaration "
        "supersedes this one)"
    ),
    "raw_format_incomplete": (
        "; re-load the file with the raw declaration (the new declaration "
        "supersedes this one)"
    ),
}


def _decode(path: Path, declaration: Mapping[str, Any]) -> np.ndarray:
    """Decode the verified file with the recorded declaration.

    The re-read refuses what the signal store refuses at load time (an empty
    line between two data lines of a ``.csv``/``.txt`` file; a NaN,
    infinite or non-numeric sample), because a file recorded before the
    store refused it, or decoded with a declaration that does not fit it,
    must not become a snapshot.

    Raises:
        ValueError: A raw declaration without ``sample_format``, a decoder
            refusal, an unsupported / unreadable self-describing file, an
            empty line between data lines, or a NaN, infinite or non-numeric
            decoded sample.
        OSError: From the decoders.
    """
    source = f"Recorded file '{path.name}'"
    raw_format = declaration.get("raw_format")
    if isinstance(raw_format, dict):
        kwargs = {
            key: raw_format[key]
            for key in _RAW_DECODE_KEYS
            if key in raw_format
            and (raw_format[key] is not None or key == "scale_factor")
        }
        if kwargs.get("sample_format") is None:
            raise ValueError(
                "recorded raw_format declares no sample_format; re-load the file "
                "with the raw declaration (the new declaration supersedes this one)"
            )
        return validate_finite_samples(
            load_raw_binary(path, **kwargs), source=source, remedy=RAW_SAMPLES_REMEDY
        )
    if path.suffix in TEXT_EXTENSIONS:
        check_text_blank_rows(path)
    data = load_self_describing(path)
    if data is None:
        raise ValueError(
            f"unsupported or empty self-describing file {path.suffix!r}; re-load "
            f"the file as CSV, NPY or raw float32 with a companion"
        )
    return np.asarray(validate_finite_samples(data, source=source), dtype=np.float64)


# ---------------------------------------------------------------------------
# Continuation token of a re-processing sequence
# ---------------------------------------------------------------------------


class _Cursor(NamedTuple):
    """A parsed continuation token."""

    last: str
    digest: str
    codes: tuple[str, ...]
    counts: dict[str, int]


_CURSOR_REMEDY = (
    "pass the token exactly as next_call gives it, or leave reprocess_cursor "
    "out to start a new sequence"
)


def _malformed_cursor(cursor: str, problem: str) -> ValueError:
    return ValueError(
        f"reprocess_cursor {cursor!r} is not a well-formed continuation token: "
        f"{problem}; {_CURSOR_REMEDY}."
    )


def parse_reprocess_cursor(cursor: object) -> _Cursor:
    """Parse a continuation token, refusing a malformed one.

    Reads nothing but the token, so a caller refuses misuse before the
    ledger is read. Whether a well-formed token still matches the ledger is
    decided by the re-processing call itself (a mismatch restarts the
    sequence, it is not refused).

    Raises:
        ValueError: Not a string, longer than :data:`MAX_CURSOR_CHARS`,
            outside the token's alphabet or shape, or codes that disagree
            with the counts; the message names the problem and the fix.
    """
    if not isinstance(cursor, str):
        raise ValueError(
            f"reprocess_cursor must be a string, got {type(cursor).__name__}; "
            f"{_CURSOR_REMEDY}."
        )
    if len(cursor) > MAX_CURSOR_CHARS:
        raise ValueError(
            f"reprocess_cursor is {len(cursor)} characters long, more than the "
            f"{MAX_CURSOR_CHARS} a continuation token can have; {_CURSOR_REMEDY}."
        )
    match = _CURSOR_PATTERN.fullmatch(cursor)
    if match is None:
        raise _malformed_cursor(
            cursor,
            "it does not have the shape r1.<16 hex>.<8 hex>.<codes>.<counts>",
        )
    last, digest, letters, counted = match.groups()
    counts: dict[str, int] = {}
    previous = ""
    for letter, number in _CURSOR_COUNT.findall(counted):
        if letter <= previous:
            raise _malformed_cursor(cursor, "its counts are not in code order")
        previous = letter
        counts[_LETTER_CODES[letter]] = int(number)
    codes = tuple(_LETTER_CODES[letter] for letter in letters)
    if len(codes) != min(MAX_LISTED_ITEMS, sum(counts.values())) or any(
        codes.count(code) > counts.get(code, 0) for code in set(codes)
    ):
        raise _malformed_cursor(cursor, "its codes disagree with its counts")
    return _Cursor(last, digest, codes, counts)


def _cursor_token(
    last: str, digest: str, codes: Sequence[str], counts: Mapping[str, int]
) -> str:
    """Encode a continuation token (see :func:`parse_reprocess_cursor`)."""
    letters = "".join(_CODE_LETTERS[code] for code in codes)
    counted = "".join(
        f"{_CODE_LETTERS[code]}{counts[code]}"
        for code in REPROCESS_REASON_CODES
        if counts.get(code)
    )
    return f"{_CURSOR_VERSION}.{last}.{digest}.{letters}.{counted}"


#: Why a well-formed token restarted the sequence, by restart code.
_RESTART_REASONS = {
    "unknown_measurement": (
        "the measurement it continues after is no longer a measurement of this point"
    ),
    "ledger_changed": (
        "it no longer matches the ledger: the point, its declaration or "
        "lineage, reference_measurements, last_k, or the measurements the "
        "earlier calls handled changed since it was issued"
    ),
}


# ---------------------------------------------------------------------------
# One re-processing call
# ---------------------------------------------------------------------------


def _group(ids: Sequence[str]) -> dict[str, Any]:
    """A group of the partition: its exact count and its capped ids."""
    return {
        "count": len(ids),
        "measurement_ids": list(ids[:MAX_LISTED_ITEMS]),
        "truncated": len(ids) > MAX_LISTED_ITEMS,
    }


def _ids_text(ids: Sequence[str], count: int) -> str:
    listed = ", ".join(ids[:_MAX_MESSAGE_IDS])
    return listed + (" and more" if count > min(len(ids), _MAX_MESSAGE_IDS) else "")


def _run_reprocess(
    asset_id: str,
    measurement_point_id: str,
    *,
    store: LedgerStore,
    data_dir: Union[str, Path],
    limit: int,
    policy: Optional[SnapshotPolicy],
    provenance_overrides: Optional[Mapping[str, str]],
    params: AssessmentParams,
    cursor: Optional[str],
) -> tuple[dict[str, Any], ReprocessCoupling]:
    """One call of a re-processing sequence (see
    :func:`reprocess_stale_snapshots`), with what the assessment of the same
    call needs to know about it."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError(f"limit must be an integer >= 1, got {limit!r}.")
    token = None if cursor is None else parse_reprocess_cursor(cursor)
    view = store.read_view(asset_id)
    point = _current_point(view, measurement_point_id)
    current_processing = compute_processing_id(policy)
    data_root = _resolved_dir(Path(data_dir))
    plan = _reprocess_plan(view, measurement_point_id, params)
    measurements = view["measurements"]
    position = {measurement_id: i for i, measurement_id in enumerate(plan.order)}
    stale = [
        measurement_id
        for measurement_id in plan.order
        if not _snapshot_is_current(
            view,
            measurement_id,
            measurements[measurement_id]["current"].get("declaration") or {},
            point,
            current_processing,
        )
    ]
    stale_set = set(stale)
    point_version = _point_version(view, measurement_point_id)

    def digest_of(
        last: str,
        members: Sequence[str],
        codes: Sequence[str],
        counts: Mapping[str, int],
    ) -> str:
        """What decides which measurements a token has passed, hashed."""
        entries = [
            [
                member,
                measurements[member]["current"].get("declaration_version"),
                member in plan.kept_by_duplicate,
            ]
            for member in members
        ]
        return content_hash(
            [
                asset_id,
                measurement_point_id,
                current_processing,
                point_version,
                params.reference_measurements,
                params.last_k,
                last,
                entries,
                list(codes),
                sorted(counts.items()),
            ]
        )[:_CURSOR_DIGEST_CHARS]

    last: Optional[str] = None
    start = 0
    earlier: list[str] = []
    earlier_codes: dict[str, str] = {}
    earlier_counts: dict[str, int] = {}
    restarted: Optional[dict[str, str]] = None
    if token is not None:
        if token.last not in position:
            restart_code = "unknown_measurement"
        else:
            cut = position[token.last]
            members = sorted(m for m in stale if position[m] <= cut)
            if (
                members
                and sum(token.counts.values()) == len(members)
                and digest_of(token.last, members, token.codes, token.counts)
                == token.digest
            ):
                restart_code = None
                last, start, earlier = token.last, cut + 1, members
                earlier_codes = dict(zip(members, token.codes))
                earlier_counts = dict(token.counts)
            else:
                restart_code = "ledger_changed"
        if restart_code is not None:
            restarted = {
                "reason_code": restart_code,
                "reason": _RESTART_REASONS[restart_code],
            }

    results: list[dict[str, Any]] = []
    found: dict[str, str] = {}
    reprocessed_ids: list[str] = []
    late_current: list[str] = []
    attempts = 0
    file_checks = 0
    current_locations: dict[str, set[str]] = {}

    # The text of every action_id this response references, by reason code.
    forms: dict[str, dict[str, str]] = {}

    def form_of(code: str, measurement_id: str) -> Optional[str]:
        """The ``action_id`` of *code* for *measurement_id*, its text kept
        for ``needs_action``."""
        action_id, text = _action_form(
            code,
            member=measurement_id in plan.baseline_members,
            relative=_location_is_relative(measurements[measurement_id]["current"]),
            data_dir=str(data_root),
        )
        if action_id is not None and text is not None:
            forms.setdefault(code, {})[action_id] = text
        return action_id

    def taken_from(own_hash: str, locations: Sequence[str]) -> frozenset[str]:
        """The *locations* another measurement with a different recorded
        content hash records as its current one (the index is built once,
        on the first failure that needs it)."""
        if not current_locations:
            current_locations.update(_current_locations(view))
        return frozenset(
            location
            for location in locations
            if current_locations.get(location, set()) - {own_hash}
        )

    def fail(result: dict[str, Any], code: str, reason: str) -> None:
        """Report the measurement of *result* not reprocessable."""
        measurement_id = result["measurement_id"]
        kept = plan.kept_by_duplicate.get(measurement_id)
        if kept is not None:
            result["reason_code"] = found[measurement_id] = "superseded_in_slot"
            result["reason"] = (
                f"{reason}; superseded in its acquisition slot by {kept}, so it "
                f"never scores and needs no action"
            )
            return
        result["reason_code"] = found[measurement_id] = code
        result["reason"] = reason
        result["action_id"] = form_of(code, measurement_id)

    for measurement_id in plan.order[start:]:
        if measurement_id not in stale_set:
            continue
        if attempts >= limit or len(found) >= MAX_LISTED_ITEMS:
            break
        payload = measurements[measurement_id]["current"]
        declaration = _declaration_of(payload)
        locations = recorded_locations(payload)
        file_info = payload.get("file") if isinstance(payload.get("file"), dict) else {}
        blocker = snapshot_blocker(payload, point)
        if (
            blocker is None
            and results
            and _file_checks_needed(locations) > MAX_FILE_CHECKS_PER_CALL - file_checks
        ):
            break
        result: dict[str, Any] = {
            "measurement_id": measurement_id,
            "acquired_at": declaration.get("acquired_at"),
            "outcome": "not_reprocessable",
            "reason_code": None,
            "reason": None,
            "action_id": None,
            "location": next(iter(locations), None),
            "location_used": None,
            "attempted": False,
            "snapshot_id": None,
        }
        results.append(result)
        if blocker is not None:
            code, reason = blocker
            fail(result, code, reason + _VIEW_REASON_SUFFIX.get(code, ""))
            continue
        size = file_info.get("size_bytes")
        own_hash = str(file_info["content_sha256"])
        taken = partial(taken_from, own_hash, locations)
        passed, failed, checks = _check_locations(
            locations,
            data_root,
            recorded_size=(
                size if isinstance(size, int) and not isinstance(size, bool) else None
            ),
        )
        file_checks += checks
        if not passed:
            fail(result, *_located_failure(locations, failed, taken))
            continue
        attempts += 1
        result["attempted"] = True
        match, unmatched = _verify_content(passed, own_hash)
        if match is None:
            fail(result, *_located_failure(locations, {**failed, **unmatched}, taken))
            continue
        location, path = match
        result["location_used"] = location
        try:
            signal = _decode(path, declaration)
            snapshot = compute_health_snapshot(
                signal,
                float(declaration["sampling_rate"]),
                declaration=declaration,
                point=point,
                policy=policy,
                provenance_overrides=provenance_overrides,
            )
        except (ValueError, OSError) as exc:
            fail(
                result,
                "file_inaccessible" if isinstance(exc, OSError) else "undecodable",
                f"snapshot not computed: {_describe(exc)}",
            )
            logger.warning(
                "Re-processing of measurement %s (%s) failed: %s",
                measurement_id,
                location,
                _describe(exc),
            )
            continue
        snapshot_id = compute_snapshot_id(
            measurement_id,
            str(snapshot["processing"]["processing_id"]),
            str(snapshot["context_digest"]),
        )
        result["snapshot_id"] = snapshot_id
        if snapshot_id in _snapshot_ids(view, measurement_id):
            result["outcome"] = "up_to_date"
            result["reason"] = "a snapshot with this id was already recorded"
            late_current.append(measurement_id)
            continue
        event = make_event(
            EVENT_HEALTH_SNAPSHOT_COMPUTED,
            asset_id,
            _snapshot_payload(
                snapshot_id, measurement_id, measurement_point_id, snapshot, point
            ),
        )
        try:
            store.append(asset_id, event)
        except _LEDGER_FAILURES as exc:
            fail(
                result,
                "append_failed",
                f"snapshot computed but not appended: {_describe(exc)}",
            )
            logger.warning(
                "Re-processed snapshot %s of measurement %s not appended: %s",
                snapshot_id,
                measurement_id,
                _describe(exc),
            )
            continue
        result["outcome"] = "reprocessed"
        reprocessed_ids.append(measurement_id)
        logger.info(
            "Re-processed measurement %s of %s from %s (lineage %s)",
            measurement_id,
            asset_id,
            location,
            current_processing,
        )

    handled = [result["measurement_id"] for result in results]
    if handled:
        last = handled[-1]
    cut = -1 if last is None else position[last]
    remaining_ids = [m for m in stale if position[m] > cut]
    earlier_set = set(earlier)
    passed_ids = sorted(earlier_set | set(found))
    code_of = {**earlier_codes, **found}
    coded = passed_ids[:MAX_LISTED_ITEMS]
    counts = dict(earlier_counts)
    for code in found.values():
        counts[code] = counts.get(code, 0) + 1
    token_out: Optional[str] = None
    if remaining_ids and passed_ids and last is not None:
        codes = [code_of[m] for m in coded]
        token_out = _cursor_token(
            last, digest_of(last, passed_ids, codes, counts), codes, counts
        )
    next_call = (
        reprocess_call(asset_id, measurement_point_id, params=params, cursor=token_out)
        if remaining_ids
        else None
    )

    superseded = set(plan.kept_by_duplicate)
    failed_here = [m for m in handled if m in found and m not in superseded]
    earlier_ids = [m for m in plan.order if m in earlier_set and m not in superseded]
    superseded_ids = [
        m for m in plan.order if (m in earlier_set or m in found) and m in superseded
    ]
    current_ids = [m for m in plan.order if m not in stale_set or m in late_current]
    unresolved = len(failed_here) + len(earlier_ids)
    if remaining_ids:
        sequence_state = "continue"
    elif unresolved:
        sequence_state = "exhausted"
    else:
        sequence_state = "complete"

    detailed = [
        {
            "measurement_id": m,
            "acquired_at": _declaration_of(measurements[m]["current"]).get(
                "acquired_at"
            ),
            "reason_code": earlier_codes[m],
            "action_id": form_of(earlier_codes[m], m),
            "location": next(
                iter(recorded_locations(measurements[m]["current"])), None
            ),
        }
        for m in earlier[:MAX_LISTED_ITEMS]
        if m not in superseded
    ]
    earlier_by_reason = {
        code: earlier_counts[code]
        for code in REPROCESS_REASON_CODES
        if earlier_counts.get(code) and code != "superseded_in_slot"
    }
    not_detailed = len(earlier_ids) - len(detailed)

    totals: dict[str, int] = dict(earlier_by_reason)
    for m in failed_here:
        totals[found[m]] = totals.get(found[m], 0) + 1

    def actions_of(code: str) -> list[dict[str, Optional[str]]]:
        """Every form of *code*'s action a failure of this response
        references, else its plain form (no failure of *code* is detailed
        in this response)."""
        referenced = forms.get(code)
        if not referenced:
            action_id, text = _action_form(code, data_dir=str(data_root))
            return [{"action_id": action_id, "action": text}]
        return [{"action_id": a, "action": text} for a, text in referenced.items()]

    needs: dict[str, dict[str, Any]] = {
        code: {
            "reason_code": code,
            "count": totals[code],
            "actions": actions_of(code),
            "measurement_ids": [],
        }
        for code in REPROCESS_REASON_CODES
        if totals.get(code)
    }
    listed = [(m, found[m]) for m in failed_here] + [
        (entry["measurement_id"], entry["reason_code"]) for entry in detailed
    ]
    for m, code in listed[:MAX_LISTED_ITEMS]:
        needs[code]["measurement_ids"].append(m)
    needs_action = []
    for entry in needs.values():
        entry["measurement_ids_truncated"] = entry["count"] > len(
            entry["measurement_ids"]
        )
        needs_action.append(entry)

    if not stale:
        message = (
            f"nothing to reprocess: {len(current_ids)} snapshot(s) of "
            f"{measurement_point_id} already on lineage {current_processing} with "
            f"the current point declaration"
        )
    else:
        message = (
            f"{len(reprocessed_ids)} measurement(s) re-processed on lineage "
            f"{current_processing}, {len(failed_here)} not reprocessable"
        )
        if earlier_ids:
            message += (
                f", {len(earlier_ids)} found not reprocessable earlier in this "
                f"sequence"
            )
        if superseded_ids:
            message += (
                f", {len(superseded_ids)} superseded in their acquisition slot "
                f"(no action needed)"
            )
        message += f", {len(remaining_ids)} stale measurement(s) remaining"
        if sequence_state == "continue":
            message += f"; call {next_call} to continue"
        elif sequence_state == "complete":
            message += (
                "; the sequence is complete: every stale measurement that "
                "represents an acquisition slot now has a current snapshot"
            )
        else:
            message += (
                f"; the sequence is exhausted: {unresolved} measurement(s) stay "
                f"stale until the action needs_action names is taken ("
                + "; ".join(
                    f"{entry['count']} {code}: "
                    f"{_ids_text(entry['measurement_ids'], entry['count'])}"
                    for code, entry in needs.items()
                )
                + "), and no further call can progress"
            )
        if not_detailed:
            message += (
                f"; {not_detailed} earlier failure(s) are not detailed in this "
                f"response: the call that found each one listed it with its "
                f"reason, location and action_id, the text of its action under "
                f"needs_action, and without those responses one new sequence "
                f"(the call without reprocess_cursor) lists each again"
            )
    if restarted is not None:
        message = (
            f"The continuation token was not used ({restarted['reason']}), so "
            f"this call started a new sequence; " + message
        )

    block = {
        "asset_id": asset_id,
        "measurement_point_id": measurement_point_id,
        "processing_id": current_processing,
        "stale": len(stale),
        "reprocessed": len(reprocessed_ids),
        "not_reprocessable": len(failed_here),
        "up_to_date": len(current_ids),
        "remaining": len(remaining_ids),
        "unresolved": unresolved,
        "superseded": len(superseded_ids),
        "earlier_in_sequence": len(earlier_ids),
        "sequence_state": sequence_state,
        "results": results,
        "groups": {
            "reprocessed": _group(reprocessed_ids),
            "up_to_date": _group(current_ids),
            "not_reprocessable": _group(failed_here),
            "earlier_in_sequence": _group(earlier_ids),
            "superseded": _group(superseded_ids),
            "remaining": _group(remaining_ids),
        },
        "needs_action": needs_action,
        "earlier_failures": detailed,
        "earlier_by_reason": earlier_by_reason,
        "earlier_not_detailed": not_detailed,
        "cursor": cursor,
        "restarted": restarted,
        "next_cursor": token_out,
        "next_call": next_call,
        "message": message,
    }
    blocked: dict[str, Optional[str]] = {m: found[m] for m in failed_here}
    blocked.update((m, earlier_codes.get(m)) for m in earlier_ids)
    return block, ReprocessCoupling(blocked=blocked, next_call=next_call)


def _location_is_relative(payload: Mapping[str, Any]) -> bool:
    """Whether the most recent recorded location is relative to the data
    directory."""
    locations = recorded_locations(payload)
    return bool(locations) and not Path(locations[0]).is_absolute()


#: The qualifiers of an ``action_id``, for the forms of an action that
#: differ from its plain text.
_MEMBER_FORM = "baseline_member"
_RELATIVE_FORM = "relative_location"


def _action_form(
    code: str, *, member: bool = False, relative: bool = False, data_dir: str
) -> tuple[Optional[str], Optional[str]]:
    """The ``action_id`` and text of the action that clears *code* for one
    measurement, (None, None) for a code that needs no action.

    The id is the reason code, qualified by each form that changes the
    text: ``+baseline_member`` for a member of the declared baseline,
    ``+relative_location`` for a most recent location relative to the data
    directory. Failures that need the same text share the id, and
    ``needs_action`` gives that text once.
    """
    plain = reprocess_action(code, data_dir=data_dir)
    if plain is None:
        return None, None
    member = member and reprocess_action(code, member=True, data_dir=data_dir) != plain
    relative = (
        relative and reprocess_action(code, relative=True, data_dir=data_dir) != plain
    )
    qualifiers = [_MEMBER_FORM] * member + [_RELATIVE_FORM] * relative
    return "+".join([code, *qualifiers]), reprocess_action(
        code, member=member, relative=relative, data_dir=data_dir
    )


def reprocess_stale_snapshots(
    asset_id: str,
    measurement_point_id: str,
    *,
    store: LedgerStore,
    data_dir: Union[str, Path],
    limit: int = MAX_REPROCESS_PER_CALL,
    policy: Optional[SnapshotPolicy] = None,
    provenance_overrides: Optional[Mapping[str, str]] = None,
    params: AssessmentParams = AssessmentParams(),
    cursor: Optional[str] = None,
) -> dict[str, Any]:
    """One call of a re-processing sequence of one point.

    See the module docstring (Re-processing). The call walks the point's
    stale measurements in priority order from its start (the head, or right
    after the measurement a valid *cursor* names) and handles each one
    through three tiers: the view checks (free), the file checks (metadata
    only, at most :data:`MAX_FILE_CHECKS_PER_CALL` per call) and the attempt
    (hash, decode, compute, append; at most *limit* per call). It stops at
    the end of the list, at either bound, or once it has found
    ``MAX_LISTED_ITEMS`` failures, so every failure is listed in full in the
    response of the call that finds it. Per-measurement failures are
    reported, never raised, and append nothing.

    Args:
        asset_id: The asset (validated by the store).
        measurement_point_id: The point.
        store: The ledger store.
        data_dir: The data directory relative locations are resolved under
            (``config.DATA_DIR`` read at call time by the caller).
        limit: Maximum measurements ATTEMPTED in this call;
            ``MAX_REPROCESS_PER_CALL`` by default. Must be >= 1.
        policy: Snapshot policy defining the current lineage; None means
            the defaults.
        provenance_overrides: See ``snapshot.collect_provenance``.
        params: Assessment policy (reference size N and last K decide the
            priority order).
        cursor: The continuation token of the previous call of the
            sequence (its ``next_cursor``), or None to start a new one. A
            token that no longer matches the ledger restarts the sequence,
            and the response says so in ``restarted``.

    Returns:
        ``{asset_id, measurement_point_id, processing_id, stale,
        reprocessed, not_reprocessable, up_to_date, remaining, unresolved,
        superseded, earlier_in_sequence, sequence_state, results, groups,
        needs_action, earlier_failures, earlier_by_reason,
        earlier_not_detailed, cursor, restarted, next_cursor, next_call,
        message}``. ``groups`` partitions the point's measurements (each
        ``{count, measurement_ids, truncated}``, at most
        ``MAX_LISTED_ITEMS`` ids); ``results`` lists one ``{measurement_id,
        acquired_at, outcome, reason_code, reason, action_id, location,
        location_used, attempted, snapshot_id}`` per measurement attempted
        or found not reprocessable in this call (``outcome`` in
        :data:`REPROCESS_OUTCOMES`, ``reason_code`` in
        :data:`REPROCESS_REASON_CODES`); ``earlier_failures`` lists
        ``{measurement_id, acquired_at, reason_code, action_id, location}``
        per earlier failure detailed; ``needs_action`` holds one
        ``{reason_code, count, actions, measurement_ids,
        measurement_ids_truncated}`` per reason code, whose ``actions`` give
        the text of each ``action_id`` this response references once (see
        :func:`_action_form`); ``remaining`` counts the stale
        measurements after the last one this call handled, so ``next_call``
        (the exact call to make, with ``next_cursor`` as its
        ``reprocess_cursor`` when an earlier failure must be passed over) is
        None exactly when it is 0; ``sequence_state`` is in
        :data:`SEQUENCE_STATES`.

    Raises:
        ValueError: An invalid ``asset_id``, ``limit`` < 1, out-of-range
            *params* or a malformed *cursor* (refused before the ledger is
            read); ``LedgerError`` when the ledger cannot be read.
    """
    block, _ = _run_reprocess(
        asset_id,
        measurement_point_id,
        store=store,
        data_dir=data_dir,
        limit=limit,
        policy=policy,
        provenance_overrides=provenance_overrides,
        params=params,
        cursor=cursor,
    )
    return block


def reprocess_for_assessment(
    asset_id: str,
    measurement_point_id: str,
    *,
    store: LedgerStore,
    data_dir: Union[str, Path],
    params: AssessmentParams = AssessmentParams(),
    cursor: Optional[str] = None,
) -> tuple[dict[str, Any], ReprocessCoupling]:
    """:func:`reprocess_stale_snapshots` with the tool's limit, plus what
    the assessment of the same call needs (``assess_change``'s
    ``reprocess`` argument): every measurement the sequence found not
    reprocessable so far, with its code, and the block's ``next_call``."""
    return _run_reprocess(
        asset_id,
        measurement_point_id,
        store=store,
        data_dir=data_dir,
        limit=MAX_REPROCESS_PER_CALL,
        policy=None,
        provenance_overrides=None,
        params=params,
        cursor=cursor,
    )


# ---------------------------------------------------------------------------
# View helpers shared by the declarations and the queries
# ---------------------------------------------------------------------------


def _asset_known(view: Mapping[str, Any]) -> bool:
    """Whether the view holds anything at all (an unknown asset is empty)."""
    return bool(
        view.get("event_count") or view.get("points") or view.get("measurements")
    )


def _measurements_of_point(
    view: Mapping[str, Any], point_id: str
) -> list[tuple[str, dict[str, Any]]]:
    """``(measurement_id, current payload)`` of the point's measurements in
    the view's order (``acquired_at``, then file order)."""
    measurements = view.get("measurements") or {}
    found: list[tuple[str, dict[str, Any]]] = []
    for measurement_id in view.get("ordered_measurement_ids") or []:
        slot = measurements.get(measurement_id)
        current = slot.get("current") if isinstance(slot, dict) else None
        if isinstance(current, dict) and current.get("measurement_point_id") == (
            point_id
        ):
            found.append((str(measurement_id), current))
    return found


def _declaration_of(payload: Mapping[str, Any]) -> dict[str, Any]:
    declaration = payload.get("declaration")
    return dict(declaration) if isinstance(declaration, dict) else {}


def _acquired_key(declaration: Mapping[str, Any]) -> tuple[Any, ...]:
    """Ordering key of an acquisition instant (instants first, then
    unparsable strings, then missing), the view's rule."""
    acquired = declaration.get("acquired_at")
    if isinstance(acquired, str):
        try:
            parsed = datetime.fromisoformat(acquired)
        except ValueError:
            return (1, 0.0, acquired)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (0, parsed.timestamp(), "")
    return (2, 0.0, "")


def _point_version(view: Mapping[str, Any], point_id: str) -> int:
    """The current declaration version of a point (0 when never declared)."""
    slot = (view.get("points") or {}).get(point_id)
    if not isinstance(slot, dict):
        return 0
    current = slot.get("current") or {}
    version = current.get("declaration_version")
    if isinstance(version, int) and not isinstance(version, bool) and version > 0:
        return version
    return len(slot.get("history") or [])


def _stale_measurement_ids(
    view: Mapping[str, Any], point_id: str, point: Optional[Mapping[str, Any]]
) -> list[str]:
    """Measurements of the point lacking the snapshot the current lineage
    and *point* context would produce (what a re-processing would do)."""
    processing = compute_processing_id(None)
    return [
        measurement_id
        for measurement_id, current in _measurements_of_point(view, point_id)
        if not _snapshot_is_current(
            view, measurement_id, _declaration_of(current), point, processing
        )
    ]


def _ledger_bytes(store: LedgerStore, asset_id: str, view: Mapping[str, Any]) -> int:
    """Size of the ledger file on disk (the readable extent as a fallback)."""
    try:
        return os.path.getsize(Path(store.root) / f"{asset_id}{LEDGER_SUFFIX}")
    except OSError:
        end_offset = view.get("end_offset")
        return int(end_offset) if isinstance(end_offset, int) else 0


def _integrity_summary(integrity: Mapping[str, Any]) -> dict[str, Any]:
    """The integrity counters without the per-record listing."""
    summary = {key: value for key, value in integrity.items() if key != "issues"}
    issues = integrity.get("issues")
    summary["issue_count"] = len(issues) if isinstance(issues, list) else 0
    return summary


def _latest_lineage(
    view: Mapping[str, Any], measurement_ids: Sequence[str]
) -> Optional[str]:
    """The ``processing_id`` of the most recently recorded snapshot among
    the measurements, or None when none has a snapshot."""
    latest: Optional[str] = None
    latest_position = -1
    measurements = view.get("measurements") or {}
    for measurement_id in measurement_ids:
        slot = measurements.get(measurement_id) or {}
        for processing, position in (slot.get("lineage_positions") or {}).items():
            if isinstance(position, int) and position > latest_position:
                latest, latest_position = str(processing), position
    return latest


def _point_summaries(view: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    baselines = view.get("baselines") or {}
    for point_id in _known_points(dict(view)):
        records = _measurements_of_point(view, point_id)
        acquired = [
            _declaration_of(current).get("acquired_at") for _, current in records
        ]
        point = _current_point(view, point_id)
        baseline_slot = baselines.get(point_id) or {}
        summaries.append(
            {
                "measurement_point_id": point_id,
                "measurement_count": len(records),
                "first_acquired_at": acquired[0] if acquired else None,
                "last_acquired_at": acquired[-1] if acquired else None,
                "latest_lineage": _latest_lineage(view, [mid for mid, _ in records]),
                "baseline_declared": baseline_slot.get("current") is not None,
                "declaration_version": (
                    None if point is None else point.get("declaration_version")
                ),
            }
        )
    return summaries


def _asset_summary(
    store: LedgerStore, asset_id: str, view: Mapping[str, Any]
) -> dict[str, Any]:
    ordered = [str(mid) for mid in (view.get("ordered_measurement_ids") or [])]
    measurements = view.get("measurements") or {}

    def acquired_of(measurement_id: str) -> Any:
        slot = measurements.get(measurement_id) or {}
        return _declaration_of(slot.get("current") or {}).get("acquired_at")

    points = _point_summaries(view)
    return {
        "asset_id": asset_id,
        "points": points,
        "point_count": len(points),
        "measurement_count": len(ordered),
        "first_acquired_at": acquired_of(ordered[0]) if ordered else None,
        "last_acquired_at": acquired_of(ordered[-1]) if ordered else None,
        "reattributed_count": len(view.get("reattributed") or []),
        "event_count": int(view.get("event_count") or 0),
        "ledger_bytes": _ledger_bytes(store, asset_id, view),
        "integrity": _integrity_summary(view.get("integrity") or {}),
    }


# ---------------------------------------------------------------------------
# Declaration validation
# ---------------------------------------------------------------------------


def _positive(field: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"{field} must be a positive number, got {value!r}; omit it when unknown"
        )
    return float(value)


def _normalize_fault_orders(value: object) -> Optional[dict[str, float]]:
    """Canonical ``{label: order}`` with labels from ``BEARING_LABELS``."""
    if not isinstance(value, Mapping):
        raise ValueError(
            f"fault_orders must be a mapping {{label: order}} with labels among "
            f"{list(BEARING_LABELS)} and orders as multiples of the shaft "
            f"frequency (for example {{'BPFO': 3.58}}), got {type(value).__name__}"
        )
    if not value:
        return None
    orders: dict[str, float] = {}
    for label, order in value.items():
        canonical = label.upper() if isinstance(label, str) else None
        if canonical not in BEARING_LABELS:
            raise ValueError(
                f"fault_orders label {label!r} is not one of {list(BEARING_LABELS)}"
            )
        if canonical in orders:
            raise ValueError(f"fault_orders declares {canonical} more than once")
        orders[canonical] = _positive(f"fault_orders[{canonical!r}]", order)
    return {label: orders[label] for label in BEARING_LABELS if label in orders}


def _validate_machine_group(value: object) -> int:
    if isinstance(value, bool) or value not in VALID_MACHINE_GROUPS:
        raise ValueError(
            f"machine_group must be one of {list(VALID_MACHINE_GROUPS)} "
            f"(ISO 20816-3: 1 large machines, 2 medium machines), got {value!r}; "
            f"omit it when unknown"
        )
    return 1 if value == 1 else 2


def _validate_support_type(value: object) -> str:
    if not isinstance(value, str) or value.lower() not in VALID_SUPPORT_TYPES:
        raise ValueError(
            f"support_type must be one of {list(VALID_SUPPORT_TYPES)}, got "
            f"{value!r}; omit it when unknown"
        )
    return value.lower()


def _validate_unit(value: object) -> str:
    if not isinstance(value, str) or value not in POINT_SIGNAL_UNITS:
        raise ValueError(
            f"expected_signal_unit must be one of {list(POINT_SIGNAL_UNITS)} "
            f"('g'/'m/s2' acceleration, 'mm/s'/'m/s' velocity), got {value!r}; "
            f"omit it when unknown"
        )
    return value


def _normalize_point_context(
    measurement_point_id: str,
    *,
    bearing_id: Optional[str],
    fault_orders: Optional[Mapping[str, float]],
    machine_group: Optional[int],
    support_type: Optional[str],
    machine_power_kw: Optional[float],
    expected_signal_unit: Optional[str],
    expected_sensor_id: Optional[str],
    expected_direction: Optional[str],
    nominal_rpm: Optional[float],
    declared_by: Optional[str],
    note: Optional[str],
) -> dict[str, Any]:
    """The declared part of a point payload, validated; every problem is
    accumulated into ONE ``ValueError`` naming the field and the remedy."""
    problems: list[str] = []
    context: dict[str, Any] = {key: None for key in POINT_DECLARED_KEYS}
    context["measurement_point_id"] = measurement_point_id

    def attempt(field: str, value: Any, normalize: Callable[[Any], Any]) -> None:
        if value is None:
            return
        try:
            context[field] = normalize(value)
        except ValueError as exc:
            problems.append(str(exc))

    free_text = {
        "bearing_id": bearing_id,
        "expected_sensor_id": expected_sensor_id,
        "declared_by": declared_by,
        "note": note,
    }
    for field in POINT_FREE_TEXT_FIELDS:
        attempt(field, free_text[field], partial(validate_free_text, field))
    attempt("fault_orders", fault_orders, _normalize_fault_orders)
    attempt("machine_group", machine_group, _validate_machine_group)
    attempt("support_type", support_type, _validate_support_type)
    attempt(
        "machine_power_kw",
        machine_power_kw,
        lambda value: _positive("machine_power_kw", value),
    )
    attempt("expected_signal_unit", expected_signal_unit, _validate_unit)
    attempt("expected_direction", expected_direction, normalize_direction)
    attempt("nominal_rpm", nominal_rpm, lambda value: _positive("nominal_rpm", value))
    if problems:
        raise ValueError(
            f"Invalid declaration of measurement point {measurement_point_id!r}: "
            + "; ".join(problems)
            + ". Fix the named field(s) and declare again."
        )
    return context


def _declared_part(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in POINT_DECLARED_KEYS}


def _validate_measurement_ids(measurement_ids: object) -> list[str]:
    if isinstance(measurement_ids, (str, bytes)) or not isinstance(
        measurement_ids, Sequence
    ):
        raise ValueError(
            "measurement_ids must be a list of measurement ids (the 16-hex "
            "measurement_id values reported by load_signal and get_asset_history); "
            "an empty list withdraws the active baseline"
        )
    ids: list[str] = []
    for item in measurement_ids:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"measurement_ids contains {item!r}: every entry must be a non-empty "
                f"measurement_id string"
            )
        if item in ids:
            raise ValueError(f"measurement_ids lists {item!r} more than once")
        ids.append(item)
    if len(ids) > MAX_BASELINE_MEMBERS:
        raise ValueError(
            f"measurement_ids lists {len(ids)} measurements, over the "
            f"{MAX_BASELINE_MEMBERS}-member cap of a baseline"
        )
    return ids


def _comparability_refusals(
    records: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
    point: Optional[dict[str, Any]],
    context: Optional[dict[str, Any]],
) -> list[str]:
    refusals: list[str] = []
    for measurement_id in ids:
        assessment = assess_measurement_comparability(
            dict(records[measurement_id]), point, context
        )
        if assessment["grade"] == "non_comparable":
            reasons = "; ".join(
                f"{entry['code']}: {entry['detail']}"
                for entry in assessment["qualifications"]
            )
            refusals.append(f"{measurement_id} ({reasons})")
    return refusals


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def declare_measurement_point(
    *,
    store: LedgerStore,
    asset_id: str,
    measurement_point_id: str,
    bearing_id: Optional[str] = None,
    fault_orders: Optional[Mapping[str, float]] = None,
    machine_group: Optional[int] = None,
    support_type: Optional[str] = None,
    machine_power_kw: Optional[float] = None,
    expected_signal_unit: Optional[str] = None,
    expected_sensor_id: Optional[str] = None,
    expected_direction: Optional[str] = None,
    nominal_rpm: Optional[float] = None,
    declared_by: Optional[str] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    """Declare (or re-declare) the context of a measurement point.

    See the module docstring (Declarations). The event is appended under
    the versioned-append lock, so two processes never assign the same
    version; a declaration identical to the current version appends
    nothing and returns the current version.

    Args:
        store: The ledger store.
        asset_id: The asset (ledger grammar).
        measurement_point_id: The point (ledger grammar).
        bearing_id: Catalog designation of the bearing at the point, or None.
        fault_orders: ``{label: order}`` with labels among ``BEARING_LABELS``
            and orders in multiples of the shaft frequency, or None.
        machine_group: ISO 20816-3 group (1 or 2), or None (not declared).
        support_type: ``"rigid"`` or ``"flexible"``, or None.
        machine_power_kw: Rated power (positive), or None.
        expected_signal_unit: Canonical unit the point's measurements are
            expected in, or None.
        expected_sensor_id: Sensor expected at the point (free text), or
            None.
        expected_direction: Expected direction (``VALID_DIRECTIONS`` and
            its form aliases), or None.
        nominal_rpm: Design speed of the point (positive), or None.
        declared_by: Who declares (free text), or None.
        note: Free-text note, or None.

    Returns:
        ``{asset_id, measurement_point_id, declaration_version, appended,
        changed, previous_version, measurements_with_stale_context, remedy,
        declaration, event_id, bearing_in_catalog, message}``: ``changed``
        lists the declared keys that differ from the previous version
        (every declared key for version 1, empty when nothing changed),
        ``measurements_with_stale_context`` counts the recorded measurements
        of the point that lack a snapshot with the current context and
        lineage, ``remedy`` is the exact re-processing call when one of them
        passes the view checks and is not collapsed into another acquisition
        slot (else None, and the message counts them by reason),
        ``declaration`` is the payload as recorded
        (the current one when nothing was appended) and
        ``bearing_in_catalog`` says whether a declared ``bearing_id`` is in
        the verified catalog (None without a bearing).

    Raises:
        ValueError: Invalid ids or values (one message naming every
            problem and its remedy); ``LedgerError`` when the ledger cannot
            be read or written.
    """
    validate_ledger_id(asset_id, kind="asset_id")
    validate_ledger_id(measurement_point_id, kind="measurement_point_id")
    declared = _normalize_point_context(
        measurement_point_id,
        bearing_id=bearing_id,
        fault_orders=fault_orders,
        machine_group=machine_group,
        support_type=support_type,
        machine_power_kw=machine_power_kw,
        expected_signal_unit=expected_signal_unit,
        expected_sensor_id=expected_sensor_id,
        expected_direction=expected_direction,
        nominal_rpm=nominal_rpm,
        declared_by=declared_by,
        note=note,
    )
    state = _AssetState(asset_id, store.read(asset_id))
    decided: dict[str, Any] = {}

    def build_event(delta: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        state.merge(delta)
        current = _current_point(state.view, measurement_point_id)
        previous_version = _point_version(state.view, measurement_point_id)
        if current is not None and canonical_json(
            _declared_part(current)
        ) == canonical_json(declared):
            decided.update(
                version=previous_version,
                changed=[],
                previous_version=previous_version,
                payload=current,
            )
            return None
        if current is None:
            changed = sorted(
                key
                for key in POINT_DECLARED_KEYS
                if key != "measurement_point_id" and declared[key] is not None
            )
        else:
            changed = sorted(
                key
                for key in POINT_DECLARED_KEYS
                if canonical_json(current.get(key)) != canonical_json(declared[key])
            )
        payload: dict[str, Any] = {}
        for key in POINT_DECLARATION_KEYS:
            if key == "declaration_version":
                payload[key] = previous_version + 1
            elif key == "changed":
                payload[key] = changed
            else:
                payload[key] = declared[key]
        event = make_event(EVENT_MEASUREMENT_POINT_DECLARED, asset_id, payload)
        decided.update(
            version=previous_version + 1,
            changed=changed,
            previous_version=previous_version or None,
            payload=event["payload"],
            event=event,
        )
        return event

    result = store.append_versioned(asset_id, state.end_offset, build_event)
    event = decided.get("event")
    appended = bool(result.appended and event is not None)
    if appended:
        state.appended_versioned(event, result.offset_after)
    else:
        state.end_offset = result.offset_after
    point = _current_point(state.view, measurement_point_id) or dict(decided["payload"])
    version = int(decided["version"])
    changed = list(decided["changed"])
    stale = _stale_measurement_ids(state.view, measurement_point_id, point)
    collapsed = (
        set(
            _reprocess_plan(
                state.view, measurement_point_id, AssessmentParams()
            ).kept_by_duplicate
        )
        if stale
        else set()
    )
    blocked: dict[str, int] = {}
    collapsed_count = 0
    attemptable = 0
    for measurement_id in stale:
        if measurement_id in collapsed:
            collapsed_count += 1
            continue
        found = snapshot_blocker(
            state.view["measurements"][measurement_id]["current"], point
        )
        if found is None:
            attemptable += 1
        else:
            blocked[found[0]] = blocked.get(found[0], 0) + 1
    remedy = reprocess_call(asset_id, measurement_point_id) if attemptable else None

    in_catalog: Optional[bool] = None
    declared_bearing = point.get("bearing_id")
    if isinstance(declared_bearing, str) and declared_bearing:
        in_catalog = lookup_bearing(declared_bearing) is not None

    message = (
        f"Measurement point {measurement_point_id!r} of asset {asset_id!r} is at "
        f"declaration version {version}"
    )
    if appended:
        message += (
            f" (appended; changed: {', '.join(changed) if changed else 'nothing'})"
        )
    else:
        message += " (identical to the current declaration: nothing appended)"
    if stale:
        message += (
            f"; {len(stale)} recorded measurement(s) lack a snapshot computed "
            f"with the current context and lineage"
        )
        if remedy is not None:
            message += (
                f"; call {remedy} to attempt the {attemptable} that pass the "
                f"checks re-processing reads from the ledger"
            )
        if blocked:
            count = sum(blocked.values())
            message += (
                f"; {count} {'needs' if count == 1 else 'need'} an action before "
                f"re-processing can compute {'it' if count == 1 else 'them'} ("
                + ", ".join(f"{n} {code}" for code, n in blocked.items())
                + ")"
            )
        if collapsed_count:
            message += f"; {collapsed_count} " + (
                "is a recording collapsed into another acquisition slot, "
                "which never scores"
                if collapsed_count == 1
                else "are recordings collapsed into another acquisition "
                "slot, which never score"
            )
    if in_catalog is False:
        message += (
            f"; bearing {declared_bearing!r} is not in the verified catalog, so "
            f"the bearing block of every snapshot reports it as missing until the "
            f"catalog knows it or fault_orders are declared instead"
        )
    message += "."
    logger.info(
        "Measurement point %s/%s declared at version %s (%s)",
        asset_id,
        measurement_point_id,
        version,
        "appended" if appended else "unchanged",
    )
    return {
        "asset_id": asset_id,
        "measurement_point_id": measurement_point_id,
        "declaration_version": version,
        "appended": appended,
        "changed": changed,
        "previous_version": decided.get("previous_version"),
        "measurements_with_stale_context": len(stale),
        "remedy": remedy,
        "declaration": dict(point),
        "event_id": event["event_id"] if appended and event is not None else None,
        "bearing_in_catalog": in_catalog,
        "message": message,
    }


def declare_healthy_baseline(
    *,
    store: LedgerStore,
    asset_id: str,
    measurement_point_id: str,
    measurement_ids: Sequence[str],
    declared_by: str,
    note: Optional[str] = None,
    declared_at: Optional[datetime] = None,
) -> dict[str, Any]:
    """Declare which recorded measurements are the healthy reference of a
    point, or withdraw the active baseline with an empty list and a note.

    See the module docstring (Declarations). The event is a blind append:
    its ``baseline_id`` hashes the point, the sorted ids and the instant,
    so a duplicate is absorbed by the reader.

    Args:
        store: The ledger store.
        asset_id: The asset (ledger grammar); it must have a ledger.
        measurement_point_id: The point (ledger grammar).
        measurement_ids: Ids of recorded measurements of the point, at
            least ``MIN_REFERENCE_MEASUREMENTS`` and at most
            ``MAX_BASELINE_MEMBERS``; an empty list withdraws the active
            baseline (a *note* is then required).
        declared_by: Who declares (free text, required): the baseline is
            attributed to this string, never to the server.
        note: Free-text note, or None.
        declared_at: Timezone-aware instant to record; None means now.

    Returns:
        ``{asset_id, measurement_point_id, baseline_id, measurement_ids,
        members, declared_by, note, declared_at, superseded_baseline_id,
        withdrawn, event_id, message}`` with ``measurement_ids`` and
        ``members`` in acquisition order, each member recording the
        ``declaration_version`` of the measurement and the
        ``point_declaration_version`` it was validated against.

    Raises:
        ValueError: Invalid ids or free text, an asset without a ledger
            (naming the known assets), an id that is not a measurement of
            the point (naming the valid ids, capped), two ids in the same
            acquisition slot, a member that is not comparable or qualified
            (naming the reasons), a withdrawal without a note or without
            an active baseline; ``LedgerError`` on a ledger failure.
    """
    validate_ledger_id(asset_id, kind="asset_id")
    validate_ledger_id(measurement_point_id, kind="measurement_point_id")
    declarer = validate_free_text("declared_by", declared_by)
    note_text = None if note is None else validate_free_text("note", note)
    ids = _validate_measurement_ids(measurement_ids)
    if declared_at is not None and (
        declared_at.tzinfo is None or declared_at.utcoffset() is None
    ):
        raise ValueError(
            "declared_at must be timezone-aware (use datetime.now(timezone.utc))."
        )

    view = store.read_view(asset_id)
    if not _asset_known(view):
        raise ValueError(
            f"Asset {asset_id!r} has no ledger; known assets: "
            f"{store.list_assets() or 'none'}. Load a measurement whose companion "
            f'declares "asset_id": "{asset_id}" before declaring a baseline.'
        )
    baseline_slot = (view.get("baselines") or {}).get(measurement_point_id) or {}
    active = baseline_slot.get("current")
    active_id = active.get("baseline_id") if isinstance(active, dict) else None
    point = _current_point(view, measurement_point_id)
    point_version = None if point is None else point.get("declaration_version")

    members: list[dict[str, Any]] = []
    if not ids:
        if note_text is None:
            raise ValueError(
                "An empty measurement_ids withdraws the active baseline of the "
                "point: pass a note saying why, or list the measurement ids of "
                "the new baseline."
            )
        if active is None:
            raise ValueError(
                f"Point {measurement_point_id!r} of asset {asset_id!r} has no "
                f"active baseline to withdraw; declare one by listing at least "
                f"{MIN_REFERENCE_MEASUREMENTS} measurement ids of the point."
            )
    else:
        records = dict(_measurements_of_point(view, measurement_point_id))
        if not records:
            raise ValueError(
                f"Point {measurement_point_id!r} of asset {asset_id!r} has no "
                f"recorded measurement; known points: "
                f"{_known_points(view) or 'none'}. Load measurements of the point "
                f"before declaring its baseline."
            )
        unknown = [
            measurement_id for measurement_id in ids if measurement_id not in records
        ]
        if unknown:
            all_measurements = view.get("measurements") or {}
            elsewhere = []
            for measurement_id in unknown:
                other = (all_measurements.get(measurement_id) or {}).get("current")
                if isinstance(other, dict):
                    elsewhere.append(
                        f"{measurement_id} belongs to point "
                        f"{other.get('measurement_point_id')!r}"
                    )
            valid = list(records)
            raise ValueError(
                f"measurement_ids {unknown} are not measurements of point "
                f"{measurement_point_id!r} of asset {asset_id!r}"
                + (f" ({'; '.join(elsewhere)})" if elsewhere else "")
                + f"; valid ids of the point: {valid[:MAX_LISTED_ITEMS]}"
                + (
                    f" (first {MAX_LISTED_ITEMS} of {len(valid)})"
                    if len(valid) > MAX_LISTED_ITEMS
                    else ""
                )
                + ". Use get_asset_history(asset_id=..., measurement_point_id=...) "
                "to list them."
            )
        if len(ids) < MIN_REFERENCE_MEASUREMENTS:
            raise ValueError(
                f"A baseline needs at least {MIN_REFERENCE_MEASUREMENTS} "
                f"measurements of the point, got {len(ids)}; list more ids (the "
                f"point has {len(records)} recorded measurement(s))."
            )
        slots: dict[tuple[Any, ...], str] = {}
        for measurement_id in ids:
            declaration = _declaration_of(records[measurement_id])
            key = (
                _acquired_key(declaration),
                declaration.get("direction"),
                declaration.get("sensor_id"),
            )
            if key in slots:
                raise ValueError(
                    f"measurement_ids {slots[key]} and {measurement_id} share the "
                    f"same acquisition slot (acquired_at, direction, sensor_id): "
                    f"the same capture exported twice counts once; keep one of them."
                )
            slots[key] = measurement_id
        refusals = _comparability_refusals(records, ids, point, None)
        if not refusals:
            context = build_reference_context(
                [dict(records[measurement_id]) for measurement_id in ids], point
            )
            refusals = _comparability_refusals(records, ids, point, context)
        if refusals:
            raise ValueError(
                f"Baseline refused: {len(refusals)} measurement(s) are not "
                f"comparable or qualified against the current declaration of "
                f"point {measurement_point_id!r} and the other members: "
                + "; ".join(refusals)
                + ". Leave them out, or correct the declarations (companion or "
                "declare_measurement_point) and re-load."
            )
        ids = sorted(ids, key=lambda mid: _acquired_key(_declaration_of(records[mid])))
        members = [
            {
                "measurement_id": measurement_id,
                "declaration_version": records[measurement_id].get(
                    "declaration_version"
                ),
                "point_declaration_version": point_version,
            }
            for measurement_id in ids
        ]

    instant = (
        utc_now_iso()
        if declared_at is None
        else declared_at.astimezone(timezone.utc).isoformat(timespec="microseconds")
    )
    baseline_id = short_id(measurement_point_id, *sorted(ids), instant)
    payload: dict[str, Any] = {
        "baseline_id": baseline_id,
        "measurement_point_id": measurement_point_id,
        "measurement_ids": list(ids),
        "members": members,
        "declared_by": declarer,
        "note": note_text,
        "declared_at": instant,
    }
    event = make_event(EVENT_BASELINE_DECLARED, asset_id, payload)
    store.append(asset_id, event)
    withdrawn = not ids
    if withdrawn:
        message = (
            f"Baseline {active_id} of point {measurement_point_id!r} of asset "
            f"{asset_id!r} withdrawn by {declarer} on {instant}; assessments fall "
            f"back to the automatic reference window and report the withdrawal."
        )
    else:
        message = (
            f"Baseline {baseline_id} declared for point {measurement_point_id!r} "
            f"of asset {asset_id!r} by {declarer} on {instant}: {len(ids)} "
            f"measurement(s) validated at point declaration version "
            f"{point_version}"
            + (f"; it supersedes baseline {active_id}" if active_id else "")
            + "."
        )
    logger.info(
        "Baseline %s of %s/%s %s by %s",
        baseline_id,
        asset_id,
        measurement_point_id,
        "withdrawn" if withdrawn else "declared",
        declarer,
    )
    return {
        "asset_id": asset_id,
        "measurement_point_id": measurement_point_id,
        "baseline_id": baseline_id,
        "measurement_ids": list(ids),
        "members": members,
        "declared_by": declarer,
        "note": note_text,
        "declared_at": instant,
        "superseded_baseline_id": active_id,
        "withdrawn": withdrawn,
        "event_id": event["event_id"],
        "message": message,
    }


# ---------------------------------------------------------------------------
# Queries: index and history
# ---------------------------------------------------------------------------


def asset_index(store: LedgerStore, *, max_assets: int) -> dict[str, Any]:
    """Summarize the assets of the ledger directory, at most *max_assets*.

    Args:
        store: The ledger store.
        max_assets: Ledgers read (and listed) at most; the ids beyond it
            are counted, never read.

    Returns:
        ``{assets, asset_count, truncated}`` where every asset entry is
        ``{asset_id, points, point_count, measurement_count,
        first_acquired_at, last_acquired_at, reattributed_count,
        event_count, ledger_bytes, integrity}`` and every point entry is
        ``{measurement_point_id, measurement_count, first_acquired_at,
        last_acquired_at, latest_lineage, baseline_declared,
        declaration_version}``.

    Raises:
        ValueError: ``max_assets`` < 1; ``LedgerError`` when a ledger
            cannot be read.
    """
    if (
        isinstance(max_assets, bool)
        or not isinstance(max_assets, int)
        or max_assets < 1
    ):
        raise ValueError(f"max_assets must be an integer >= 1, got {max_assets!r}.")
    all_assets = store.list_assets()
    listed = all_assets[:max_assets]
    return {
        "assets": [
            _asset_summary(store, asset_id, store.read_view(asset_id))
            for asset_id in listed
        ],
        "asset_count": len(all_assets),
        "truncated": len(all_assets) > max_assets,
    }


def _measurement_entry(
    view: Mapping[str, Any], measurement_id: str, current: Mapping[str, Any]
) -> dict[str, Any]:
    """One measurement of the history listing.

    The indicator preview comes from the last snapshot recorded for the
    measurement, whatever its lineage or context. After a restored
    declaration that can differ from the snapshot ``assess_change`` scores,
    which is the one the current declarations produce.
    """
    declaration = _declaration_of(current)
    file_info = current.get("file")
    slot = (view.get("measurements") or {}).get(measurement_id) or {}
    snapshots = list(slot.get("snapshots") or [])
    latest = snapshots[-1] if snapshots else None
    indicators: Optional[dict[str, Any]] = None
    if isinstance(latest, dict):
        values = latest.get("indicators") or {}
        indicators = {
            "rms": values.get("rms"),
            "peak": values.get("peak"),
            "crest_factor": values.get("crest_factor"),
            "kurtosis": values.get("kurtosis"),
            "unit": declaration.get("signal_unit"),
        }
    point_id = current.get("measurement_point_id")
    point = _current_point(view, point_id) if isinstance(point_id, str) else None
    assessment = assess_measurement_comparability(dict(current), point)
    return {
        "measurement_id": measurement_id,
        "measurement_point_id": point_id,
        "acquired_at": declaration.get("acquired_at"),
        "signal_id": current.get("signal_id"),
        "location": file_info.get("location") if isinstance(file_info, dict) else None,
        "declaration_version": current.get("declaration_version"),
        "rpm": declaration.get("rpm"),
        "direction": declaration.get("direction"),
        "sensor_id": declaration.get("sensor_id"),
        "lineages": sorted(
            str(key) for key in (slot.get("snapshots_by_lineage") or {})
        ),
        "snapshot_count": len(snapshots),
        "indicators": indicators,
        "comparability": {
            "grade": assessment["grade"],
            "codes": [str(entry["code"]) for entry in assessment["qualifications"]],
        },
    }


def _capped_history(slot: Mapping[str, Any]) -> dict[str, Any]:
    history = list(slot.get("history") or [])
    return {
        "current": slot.get("current"),
        "history": history[-MAX_LISTED_ITEMS:],
        "history_truncated": len(history) > MAX_LISTED_ITEMS,
    }


def asset_history(
    store: LedgerStore,
    asset_id: str,
    *,
    measurement_point_id: Optional[str] = None,
    max_measurements: int = 20,
) -> Optional[dict[str, Any]]:
    """The history of ONE asset, read from its ledger alone.

    Args:
        store: The ledger store.
        asset_id: The asset (ledger grammar).
        measurement_point_id: Restrict the measurements, declarations and
            baselines to one point, or None for the whole asset.
        max_measurements: Measurements listed at most, newest first.

    Returns:
        None when the asset has no ledger; else ``{asset_id, summary
        (measurement_count, point_count, first_acquired_at,
        last_acquired_at, points), measurement_point_id, point_found,
        known_points, measurements, measurement_count, truncated,
        point_declarations, baselines, reattributed, reattributed_truncated,
        integrity, event_count, ledger_bytes}``: every measurement entry
        carries ``{measurement_id, measurement_point_id, acquired_at,
        signal_id, location, declaration_version, rpm, direction,
        sensor_id, lineages, snapshot_count, indicators (rms, peak,
        crest_factor, kurtosis, unit from the last snapshot recorded, which
        after a restored declaration may not be the one the assessment
        scores; or None),
        comparability (grade and codes against the point)}``; the
        declarations and baselines map each selected point to ``{current,
        history, history_truncated}``.

    Raises:
        ValueError: Invalid ids or ``max_measurements`` < 1; ``LedgerError``
            when the ledger cannot be read.
    """
    validate_ledger_id(asset_id, kind="asset_id")
    if measurement_point_id is not None:
        validate_ledger_id(measurement_point_id, kind="measurement_point_id")
    if (
        isinstance(max_measurements, bool)
        or not isinstance(max_measurements, int)
        or max_measurements < 1
    ):
        raise ValueError(
            f"max_measurements must be an integer >= 1, got {max_measurements!r}."
        )
    view = store.read_view(asset_id)
    if not _asset_known(view):
        return None

    known_points = _known_points(view)
    point_found = measurement_point_id is None or measurement_point_id in known_points
    if measurement_point_id is None:
        selected_points = known_points
    else:
        selected_points = [measurement_point_id] if point_found else []
    summary = _asset_summary(store, asset_id, view)

    measurements = view.get("measurements") or {}
    ordered: list[tuple[str, dict[str, Any]]] = []
    for measurement_id in view.get("ordered_measurement_ids") or []:
        current = (measurements.get(measurement_id) or {}).get("current")
        if not isinstance(current, dict):
            continue
        if measurement_point_id is not None and (
            current.get("measurement_point_id") != measurement_point_id
        ):
            continue
        ordered.append((str(measurement_id), current))
    newest_first = list(reversed(ordered))[:max_measurements]

    points = view.get("points") or {}
    baselines = view.get("baselines") or {}
    reattributed = list(view.get("reattributed") or [])
    return {
        "asset_id": asset_id,
        "summary": {
            "measurement_count": summary["measurement_count"],
            "point_count": summary["point_count"],
            "first_acquired_at": summary["first_acquired_at"],
            "last_acquired_at": summary["last_acquired_at"],
            "points": [
                point
                for point in summary["points"]
                if point["measurement_point_id"] in selected_points
            ],
        },
        "measurement_point_id": measurement_point_id,
        "point_found": point_found,
        "known_points": known_points,
        "measurements": [
            _measurement_entry(view, measurement_id, current)
            for measurement_id, current in newest_first
        ],
        "measurement_count": len(ordered),
        "truncated": len(ordered) > max_measurements,
        "point_declarations": {
            point_id: _capped_history(points[point_id])
            for point_id in selected_points
            if isinstance(points.get(point_id), dict)
        },
        "baselines": {
            point_id: _capped_history(baselines[point_id])
            for point_id in selected_points
            if isinstance(baselines.get(point_id), dict)
        },
        "reattributed": reattributed[:MAX_LISTED_ITEMS],
        "reattributed_truncated": len(reattributed) > MAX_LISTED_ITEMS,
        "integrity": dict(view.get("integrity") or {}),
        "event_count": summary["event_count"],
        "ledger_bytes": summary["ledger_bytes"],
    }
