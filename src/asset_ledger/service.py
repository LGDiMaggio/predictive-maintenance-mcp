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
AND the ``context_digest`` of the current point declaration. Up to
``MAX_REPROCESS_PER_CALL`` stale measurements are re-processed per call, in
the order members of the active reference (declared baseline or the first N
slots), then the last K slots newest first, then the rest newest first, so
the reference and the last K slots carry the current lineage within two
calls whatever the length of the history (the assessment moves onto the
current lineage once EVERY evaluated slot carries it; until then it uses
the older lineage that still covers the whole set, or reports
``processing_not_homogeneous`` when none does). The file is searched in
every location ever declared for the measurement, most recent first
(relative locations under
the data directory, contained by ``safe_resolve``; absolute ones as
recorded), its content hash is verified against the ledger before a single
sample is decoded, and the location used is reported. A file missing
everywhere or changed is a per-measurement ``not_reprocessable`` with the
reason; the old snapshot is never touched. Only derived events are
appended, each deduplicated by its deterministic ``snapshot_id``, so a
repeated call is idempotent.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np

from ..path_safety import safe_resolve
from ..signal_acquisition.loaders import load_raw_binary, load_self_describing
from ..signal_acquisition.measurement import (
    MEASUREMENT_DECLARATION_KEYS,
    digest_file,
)
from .assessment import AssessmentParams, collect_point_slots, current_snapshot_id_of
from .comparability import assess_measurement_comparability
from .snapshot import (
    SnapshotPolicy,
    compute_health_snapshot,
    processing_id as compute_processing_id,
    snapshot_id as compute_snapshot_id,
)
from .store import (
    EVENT_HEALTH_SNAPSHOT_COMPUTED,
    EVENT_MEASUREMENT_RECORDED,
    LedgerError,
    LedgerReadResult,
    LedgerStore,
    build_asset_view,
    canonical_json,
    content_hash,
    make_event,
)

__all__ = [
    "DECLARATION_KEYS",
    "LEDGER_STATUSES",
    "SNAPSHOT_STATUSES",
    "OUTCOME_KEYS",
    "LOAD_OUTCOME_KEYS",
    "SNAPSHOT_PAYLOAD_KEYS",
    "MAX_REPROCESS_PER_CALL",
    "REPROCESS_OUTCOMES",
    "SignalSource",
    "build_declaration",
    "file_block",
    "declaration_fingerprint",
    "changed_keys",
    "record_measurements",
    "resolve_point_context",
    "reprocess_stale_snapshots",
]

logger = logging.getLogger(__name__)

#: Stale measurements re-processed per :func:`reprocess_stale_snapshots`
#: call. Ten keeps one tool call bounded (ten hashes, ten decodes, ten
#: snapshots) while reference plus last K fit in two calls.
MAX_REPROCESS_PER_CALL = 10

#: Per-measurement outcomes of a re-processing call.
REPROCESS_OUTCOMES: tuple[str, ...] = ("reprocessed", "up_to_date", "not_reprocessable")

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


def _reprocess_order(
    view: Mapping[str, Any],
    measurement_point_id: str,
    params: AssessmentParams,
) -> list[str]:
    """Measurement ids of the point in re-processing priority: members of
    the active reference (declared baseline, else the first N usable slots
    by ``acquired_at``), then the last K slots newest first, then the rest
    newest first, then the non-comparable ones newest first (they are
    re-processed last: a corrected declaration may make them usable)."""
    staged = collect_point_slots(dict(view), measurement_point_id, params=params)
    usable = staged["usable"]
    head = staged["reference"]["slots"] or usable[: params.reference_measurements]
    ordered: list[str] = []

    def add(slots: Sequence[Mapping[str, Any]]) -> None:
        for slot in slots:
            measurement_id = str(slot["measurement_id"])
            if measurement_id not in ordered:
                ordered.append(measurement_id)

    add(head)
    add(list(reversed(usable[-params.last_k :])))
    add(list(reversed(usable)))
    add(list(reversed(staged["slots"])))
    return ordered


def _candidate_locations(payload: Mapping[str, Any]) -> list[str]:
    """Every location ever declared, most recent first, without repeats."""
    locations = payload.get("locations")
    candidates = (
        [str(item) for item in locations] if isinstance(locations, list) else []
    )
    file_info = payload.get("file")
    if isinstance(file_info, dict) and file_info.get("location") is not None:
        candidates.append(str(file_info["location"]))
    ordered: list[str] = []
    for location in reversed(candidates):
        if location not in ordered:
            ordered.append(location)
    return ordered


def _locate_file(
    payload: Mapping[str, Any], data_dir: Path
) -> tuple[Optional[Path], Optional[str], list[str]]:
    """Find the measurement's file at one of its declared locations.

    Returns:
        ``(path, location, reasons)``: the first existing location (most
        recent first) whose content hash equals the recorded one, or
        ``(None, None, reasons)`` with one reason per location tried.
    """
    file_info = payload.get("file")
    expected = (
        str(file_info.get("content_sha256"))
        if isinstance(file_info, dict) and file_info.get("content_sha256")
        else None
    )
    reasons: list[str] = []
    for location in _candidate_locations(payload):
        candidate = Path(location)
        if not candidate.is_absolute():
            try:
                candidate = safe_resolve(data_dir, location)
            except ValueError:
                reasons.append(f"location escapes the data directory: {location}")
                continue
        if not candidate.is_file():
            reasons.append(f"file not found at {location}")
            continue
        if expected is None:
            reasons.append(
                f"no content hash recorded for {location}: the file cannot be "
                f"verified"
            )
            continue
        try:
            digest, _ = digest_file(candidate)
        except OSError as exc:
            reasons.append(f"cannot read {location}: {exc.strerror or exc}")
            continue
        if digest != expected:
            reasons.append(f"content differs at {location}")
            continue
        return candidate, location, reasons
    if not reasons:
        reasons.append("no file location recorded for the measurement")
    return None, None, reasons


def _decode(path: Path, declaration: Mapping[str, Any]) -> np.ndarray:
    """Decode the verified file with the recorded declaration.

    Raises:
        ValueError: A raw declaration without ``sample_format``, a decoder
            refusal, or an unsupported / unreadable self-describing file.
        OSError: From the decoders.
    """
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
        return load_raw_binary(path, **kwargs)
    data = load_self_describing(path)
    if data is None:
        raise ValueError(
            f"unsupported or empty self-describing file {path.suffix!r}; re-load "
            f"the file as CSV, NPY or raw float32 with a companion"
        )
    return np.asarray(data, dtype=np.float64)


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
) -> dict[str, Any]:
    """Re-process up to *limit* stale measurements of one point.

    See the module docstring (Re-processing). Idempotent: a second call on
    an unchanged ledger re-processes nothing. Per-measurement failures are
    reported, never raised.

    Args:
        asset_id: The asset (validated by the store).
        measurement_point_id: The point.
        store: The ledger store.
        data_dir: The data directory relative locations are resolved under
            (``config.DATA_DIR`` read at call time by the caller).
        limit: Maximum measurements ATTEMPTED (re-processed or not
            reprocessable) in this call; ``MAX_REPROCESS_PER_CALL`` by
            default. Must be >= 1.
        policy: Snapshot policy defining the current lineage; None means
            the defaults.
        provenance_overrides: See ``snapshot.collect_provenance``.
        params: Assessment policy (reference size N and last K decide the
            priority order).

    Returns:
        ``{asset_id, measurement_point_id, processing_id, stale,
        reprocessed, not_reprocessable, up_to_date, remaining, results,
        next_call, message}`` where ``results`` lists one
        ``{measurement_id, acquired_at, outcome, location_used, reason,
        snapshot_id}`` per measurement attempted (``outcome`` in
        :data:`REPROCESS_OUTCOMES`; ``up_to_date`` appears only when a
        snapshot with the expected id was appended meanwhile), ``remaining``
        counts the stale measurements not attempted, and ``next_call`` is
        the exact call to make when ``remaining > 0`` (else None).

    Raises:
        ValueError: An invalid ``asset_id``, ``limit`` < 1, or out-of-range
            *params*; ``LedgerError`` when the ledger cannot be read.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError(f"limit must be an integer >= 1, got {limit!r}.")
    view = store.read_view(asset_id)
    point = _current_point(view, measurement_point_id)
    current_processing = compute_processing_id(policy)
    base_dir = Path(data_dir)

    stale: list[str] = []
    up_to_date = 0
    for measurement_id in _reprocess_order(view, measurement_point_id, params):
        slot = view["measurements"][measurement_id]
        declaration = slot["current"].get("declaration") or {}
        try:
            expected = current_snapshot_id_of(
                declaration,
                point,
                processing=current_processing,
                measurement_id=measurement_id,
            )
        except ValueError:
            stale.append(measurement_id)  # reported by the attempt below
            continue
        if expected in _snapshot_ids(view, measurement_id):
            up_to_date += 1
        else:
            stale.append(measurement_id)

    results: list[dict[str, Any]] = []
    reprocessed = 0
    failed = 0
    for measurement_id in stale[:limit]:
        payload = view["measurements"][measurement_id]["current"]
        declaration = payload.get("declaration") or {}
        result: dict[str, Any] = {
            "measurement_id": measurement_id,
            "acquired_at": declaration.get("acquired_at"),
            "outcome": "not_reprocessable",
            "location_used": None,
            "reason": None,
            "snapshot_id": None,
        }
        results.append(result)
        fs = declaration.get("sampling_rate")
        if fs is None or not fs > 0:
            result["reason"] = (
                "sampling_rate not declared; re-load the file with a declared "
                "sampling rate (the new declaration supersedes this one)"
            )
            failed += 1
            continue
        path, location, reasons = _locate_file(payload, base_dir)
        if path is None:
            result["reason"] = "; ".join(reasons)
            failed += 1
            continue
        result["location_used"] = location
        try:
            signal = _decode(path, declaration)
            snapshot = compute_health_snapshot(
                signal,
                float(fs),
                declaration=declaration,
                point=point,
                policy=policy,
                provenance_overrides=provenance_overrides,
            )
        except (ValueError, OSError) as exc:
            result["reason"] = f"snapshot not computed: {_describe(exc)}"
            failed += 1
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
            up_to_date += 1
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
            result["reason"] = f"snapshot computed but not appended: {_describe(exc)}"
            failed += 1
            logger.warning(
                "Re-processed snapshot %s of measurement %s not appended: %s",
                snapshot_id,
                measurement_id,
                _describe(exc),
            )
            continue
        result["outcome"] = "reprocessed"
        reprocessed += 1
        logger.info(
            "Re-processed measurement %s of %s from %s (lineage %s)",
            measurement_id,
            asset_id,
            location,
            current_processing,
        )

    remaining = max(0, len(stale) - len(results))
    next_call = (
        None
        if remaining == 0
        else (
            f"assess_asset_change(asset_id={asset_id!r}, "
            f"measurement_point_id={measurement_point_id!r}, reprocess=True)"
        )
    )
    if not stale:
        message = (
            f"nothing to reprocess: {up_to_date} snapshot(s) of "
            f"{measurement_point_id} already on lineage {current_processing} with "
            f"the current point declaration"
        )
    else:
        message = (
            f"{reprocessed} measurement(s) re-processed on lineage "
            f"{current_processing}, {failed} not reprocessable, {remaining} stale "
            f"measurement(s) remaining"
            + (f"; call {next_call} to continue" if next_call else "")
        )
    return {
        "asset_id": asset_id,
        "measurement_point_id": measurement_point_id,
        "processing_id": current_processing,
        "stale": len(stale),
        "reprocessed": reprocessed,
        "not_reprocessable": failed,
        "up_to_date": up_to_date,
        "remaining": remaining,
        "results": results,
        "next_call": next_call,
        "message": message,
    }
