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
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import AbstractSet, Any, Mapping, Optional

from ...asset_ledger.store import (
    EVENT_MEASUREMENT_RECORDED,
    LEDGER_SUFFIX,
    LedgerStore,
    build_asset_view,
)
from ...signal_acquisition.measurement import measurement_id_from_digest
from ...signal_acquisition.text_decoder import DecodeResult
from .. import statements as st
from ..contracts import ChannelPrecheck
from ..records import ImportRecordStore
from ._common import _Run

#: Most import records one validation reads to find the earlier records of
#: the same bytes. A record's file digest is stored only inside the record,
#: so finding them means reading every record (about 2 ms each, measured):
#: a store holding more than this is not compared at all, and validation
#: reports the history as incomplete instead of taking it as empty.
#: PROPOSED value, pending the owner's approval.
HISTORY_RECORD_LIMIT = 250


@dataclass
class _Prior:
    """Earlier records of the same bytes, read once and only when the store
    holds at most :data:`HISTORY_RECORD_LIMIT` records. An unreadable store
    or record raises; a damaged one is listed in *refused*; neither ever
    reads as no records."""

    store: ImportRecordStore
    sha256: Optional[str]
    loaded: Optional[dict[str, dict[str, Any]]] = None
    refused: tuple[str, ...] = ()
    #: Records in the store when it held more than the limit (not read).
    incomplete: Optional[int] = None

    def records(self) -> dict[str, dict[str, Any]]:
        if self.loaded is None:
            loaded: dict[str, dict[str, Any]] = {}
            if self.sha256 is not None:
                held = len(self.store._record_names())
                if held > HISTORY_RECORD_LIMIT:
                    self.incomplete = held
                else:
                    entries, refused = self.store.list_records(file_sha256=self.sha256)
                    self.refused = refused
                    for entry in entries:
                        loaded[entry.import_id] = self.store.read(
                            entry.import_id
                        ).interpretation
            self.loaded = loaded
        return self.loaded

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
    lines = import_format.get("data_lines") or {}
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
    except (TypeError, ValueError):
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
#: ledger of 21 events, measured): a ledger directory holding more than this
#: is not compared at all, and validation reports the history as
#: incomplete. PROPOSED value, pending the owner's approval.
HISTORY_ASSET_LIMIT = 250


def _well_formed(payload: Any) -> bool:
    """A recorded measurement whose declaration and file block can be read,
    so whether it concerns these bytes is known."""
    if not isinstance(payload, Mapping):
        return False
    file_block = payload.get("file")
    return (
        isinstance(payload.get("declaration"), Mapping)
        and isinstance(file_block, Mapping)
        and isinstance(file_block.get("content_sha256"), str)
    )


_INTEGRITY_COUNTERS = (
    "unreadable_records",
    "asset_id_mismatches",
    "unsupported_schema_versions",
    "malformed_payloads",
)


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
    the ledger directory holds at most :data:`HISTORY_ASSET_LIMIT` of them;
    the pre-check and the comparison with earlier imports read the same
    views. The measurement index is not consulted: it can lag the ledgers.
    A ledger directory or ledger that cannot be read raises."""

    store: LedgerStore
    #: The measurement ids of these bytes, one per column of the file. A
    #: measurement id is derived from the file bytes and the column, so a
    #: recorded measurement under another id is not of these bytes.
    candidates: AbstractSet[str] = frozenset()
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Per asset read: records it set aside as unreadable, plus recorded
    #: measurements of these bytes whose declaration or file block cannot
    #: be read.
    damaged: dict[str, int] = field(default_factory=dict)
    #: Ledger files with a ledger name that the store cannot list as an
    #: asset ledger and that hold something (an empty file holds nothing).
    unlisted: list[str] = field(default_factory=list)
    #: Asset ledgers in the directory when it held more than the limit.
    incomplete: Optional[int] = None

    def __post_init__(self) -> None:
        report = self.store.integrity_report()
        for entry in report["unlisted_files"]:
            # A file whose name is not a ledger id was never written by the
            # store as an asset ledger, so it holds no history.
            if entry["reason"] == "invalid_stem":
                continue
            if entry["reason"] == "no_readable_event":
                # An empty file (a first append that failed after creating
                # it) holds nothing; anything else unreadable is damage.
                result = self.store.read(entry["file"][: -len(LEDGER_SUFFIX)])
                if not result.events and not _integrity_damage(result.integrity):
                    continue
            self.unlisted.append(entry["file"])
        assets = list(report["assets"])
        if len(assets) > HISTORY_ASSET_LIMIT:
            self.incomplete = len(assets)
            return
        for asset in assets:
            self.of(asset)

    def of(self, asset: str) -> dict[str, Any]:
        if asset not in self.cache:
            result = self.store.read(asset)
            view = build_asset_view(
                asset, result.events, result.integrity, end_offset=result.end_offset
            )
            # Every recorded measurement of these bytes, moved away or not,
            # must be readable; one of other bytes is irrelevant by its id.
            malformed = sum(
                event.get("event_type") == EVENT_MEASUREMENT_RECORDED
                and isinstance(payload := event.get("payload"), Mapping)
                and payload.get("measurement_id") in self.candidates
                and not _well_formed(payload)
                for event in result.events
            )
            self.damaged[asset] = (
                _integrity_damage(_mapping(view.get("integrity"))) + malformed
            )
            self.cache[asset] = view
        return self.cache[asset]

    def holders(self, measurement_id: str) -> list[str]:
        """The read assets that hold *measurement_id* as a current
        measurement (one, in a consistent ledger)."""
        return sorted(
            asset
            for asset, view in self.cache.items()
            if measurement_id in view["measurements"]
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
    for column, identity in identities.items():
        asset = identity["asset_id"]
        point_id = identity["measurement_point_id"]
        mid = identity["measurement_id"]
        view = view_of(asset)
        point = (view["points"].get(point_id) or {}).get("current")
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
                baseline = (
                    view_of(recorded_asset)["baselines"].get(
                        declaration.get("measurement_point_id")
                    )
                    or {}
                ).get("current")
                member = _baseline_member(baseline, mid)
        swap = _column_swap(run, decode, identity, column, view)
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


def _column_swap(
    run: _Run,
    decode: DecodeResult,
    identity: Mapping[str, Any],
    column: int,
    view: Mapping[str, Any],
) -> Optional[str]:
    """Another column of the same bytes recorded at this point and slot."""
    point_id = identity["measurement_point_id"]
    for mid, slot in view["measurements"].items():
        current = _mapping(_mapping(slot).get("current"))
        if not current or mid == identity["measurement_id"]:
            continue
        declaration = _mapping(current.get("declaration"))
        file = _mapping(current.get("file"))
        if (
            file.get("content_sha256") != decode.source_sha256
            or declaration.get("measurement_point_id") != point_id
            or declaration.get("channel_index") == column
            or _slot(declaration) != _slot(identity)
        ):
            continue
        other = declaration.get("channel_index")
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
                    f"(index {declaration.get('channel_index')}) with another "
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


def _candidate_ids(decode: DecodeResult) -> frozenset[str]:
    """The measurement id of each column of these bytes: every measurement
    ever recorded from them has one of these ids."""
    return frozenset(
        measurement_id_from_digest(decode.source_sha256, column)
        for column in range(decode.field_count)
    )


def _check_history(
    run: _Run,
    views: "_Views",
    prior: _Prior,
    candidates: AbstractSet[str],
) -> None:
    """History this validation needed but could not read, or did not read in
    full: ``invalid``, because no user confirmation can stand in for it (a
    storage-integrity problem, not a configuration one). Any of it may
    concern these bytes, and nothing that was read shows it does not."""
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
            f"The import record directory holds {prior.incomplete} records, more "
            f"than the {HISTORY_RECORD_LIMIT} one validation reads, so earlier "
            f"imports of these bytes were not compared.{unread}",
        )
    if views.incomplete is not None:
        run.issue(
            "history_incomplete",
            "invalid",
            f"The ledger directory holds {views.incomplete} asset ledgers, more "
            f"than the {HISTORY_ASSET_LIMIT} one validation reads, so earlier "
            f"measurements of these bytes were not compared.{unread}",
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
                f"read in full, so whether they concern these bytes is "
                f"unknown.{remedy}",
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
