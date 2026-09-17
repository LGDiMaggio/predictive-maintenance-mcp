"""
ISO 13374 Block 3 — State Detection (asset ledger assessment of change).

Turns the projected view of one asset (``store.build_asset_view``) and the
health snapshots it holds into a structured assessment of change for one
measurement point: acquisition slots, the reference used, the homogeneous
processing lineage, the variation of every amplitude indicator against the
reference band, the presence of bearing evidence, and a classification with
an explicit criterion. ISO 20816-3:2022, 6.3 (criterion II) judges a change
against a reference established in the same position and approximately the
same operating conditions, and calls a change beyond 25% of the B/C
boundary significant whatever the zone; ISO 13373-1:2002, 7.3.2 asks that a
25% jump "be confirmed by continued monitoring"; ISO 17359:2018, 8.4
requires a fault to be told from a change of load or speed. This module is
where those requirements become a reportable verdict.

Vocabulary of the assessment
    * **Acquisition slot**: (``acquired_at``, ``direction``, ``sensor_id``)
      of a measurement of the point. Several measurements in one slot (the
      same capture exported twice) collapse to one: the latest recorded wins
      (a member of the active baseline wins over a non-member), the others
      are listed under ``collapsed_duplicates``. Slots that share an instant
      but differ in direction or sensor carry ``timestamp_collision``.
    * **Usable slot**: graded ``comparable`` or ``qualified`` by the
      comparability rules; a ``non_comparable`` slot stays recorded, is
      excluded and listed with its reasons.
    * **Reference**: the active declared baseline (members revalidated) or
      the automatic window of the first N usable slots by ``acquired_at``
      (the earliest history: ISO 13373-1, 7.2.1 takes the baseline "when the
      operation is known to be acceptable and stable"; a window that skipped
      early qualified slots for later comparable ones would let a later
      episode into the reference). Qualified slots inside the window are
      reported with their codes, because a sensor change or a speed
      deviation inside the reference inflates sigma and must be said. With
      fewer than N + 1 usable slots and at least four, the reference is
      provisional (the first n - 1); with fewer than four, the history is
      insufficient. The automatic window is never called healthy
      (``health_declared`` False); only a declared baseline is.
    * **Lineage**: the ``processing_id`` used for every slot of the
      evaluated set (reference plus the post-reference slots inside the
      requested window). The most recent lineage that covers EVERY slot is
      used; none covering all is ``processing_not_homogeneous`` with the
      counts and the re-processing remedy, never a trend with holes.
    * **Band**: per amplitude indicator, mean +/- max(3 sigma, 25% of the
      mean) of the reference in the unit expected by the point (the floor
      implements note 3 of ISO 20816-3, 6.3 and protects a sigma near
      zero). For the ISO velocity, with machine group and support type
      known, a change beyond 25% of the B/C boundary is reported too.
    * **Classification** per indicator over the post-reference slots:
      ``persistent_change`` when the last >= 3 slots are outside the band on
      the same side, or at least 4 of the last max(K, 5) slots are outside
      on the same side (intermittent faults), or a significant drift
      (``analyze_trend``, p < 0.05) ends with the latest value outside the
      band; ``isolated_episode`` when an earlier exceedance returned inside
      and the latest is inside; ``unconfirmed_single_acquisition`` when only
      the latest is outside; ``no_change`` otherwise; ``sudden`` when a step
      between consecutive slots exceeds twice the band half-width. The
      overall classification is the most severe across the amplitude
      indicators and the bearing evidence persistence; crest factor and
      kurtosis are reported as support only. Direction is ``increase`` or
      ``decrease``, never "improvement".

Pure module: it works on the view and on snapshot payloads only. It never
reads a file, never touches a signal, never takes a lock, never reads the
clock. Two identical views give byte-identical canonical payloads.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np

from ..diagnostics.iso20816 import get_zone_boundaries
from ..prognostics.trend_analyzer import analyze_trend
from .comparability import (
    INFORMATIONAL_CODES,
    QUALIFICATION_CODES,
    ComparabilityThresholds,
    assess_measurement_comparability,
    build_reference_context,
    summarize_comparability,
)
from .snapshot import (
    BEARING_LABELS,
    context_digest,
    processing_id,
    resolve_context,
    snapshot_id,
)

__all__ = [
    "ASSESSMENT_STATUSES",
    "CLASSIFICATIONS",
    "REFERENCE_KINDS",
    "STATISTICS_QUALITIES",
    "AMPLITUDE_INDICATORS",
    "SUPPORT_INDICATORS",
    "ENVELOPE_INDICATOR_PREFIX",
    "EVIDENCE_INDICATOR_PREFIX",
    "BASELINE_QUALIFICATION_CODES",
    "MIN_REFERENCE_MEASUREMENTS",
    "MIN_USABLE_SLOTS",
    "MAX_LISTED_ITEMS",
    "AssessmentParams",
    "validate_params",
    "collect_point_slots",
    "current_snapshot_id_of",
    "assess_change",
]

logger = logging.getLogger(__name__)

#: The four outcomes of :func:`assess_change`, discriminated by ``status``.
ASSESSMENT_STATUSES: tuple[str, ...] = (
    "assessed",
    "not_found",
    "insufficient_history",
    "processing_not_homogeneous",
)

#: Classifications from the least to the most severe.
CLASSIFICATIONS: tuple[str, ...] = (
    "no_change",
    "isolated_episode",
    "unconfirmed_single_acquisition",
    "persistent_change",
)

REFERENCE_KINDS: tuple[str, ...] = ("automatic_window", "declared_baseline")

#: Statistical quality of a reference by its size: 3-5, 6-9, >= 10 slots.
STATISTICS_QUALITIES: tuple[str, ...] = ("relative_only", "provisional", "full")

#: Amplitude indicators read from every snapshot (plus one
#: ``envelope_<label>`` per bearing label present in the evaluated set).
AMPLITUDE_INDICATORS: tuple[str, ...] = ("rms", "peak", "one_x", "iso_velocity")

#: Reported as support, never classified.
SUPPORT_INDICATORS: tuple[str, ...] = ("crest_factor", "kurtosis")

ENVELOPE_INDICATOR_PREFIX = "envelope_"
EVIDENCE_INDICATOR_PREFIX = "evidence_"

#: Qualification codes produced here for the members of a declared
#: baseline (outside the comparability vocabulary, counted after it).
BASELINE_QUALIFICATION_CODES: tuple[str, ...] = (
    "baseline_member_superseded",
    "baseline_member_non_comparable",
)

#: Smallest reference (declared or automatic) the statistics accept.
MIN_REFERENCE_MEASUREMENTS = 3

#: Smallest usable history: a reference of three plus one slot to assess.
MIN_USABLE_SLOTS = MIN_REFERENCE_MEASUREMENTS + 1

#: Cap of the per-measurement listings (excluded, collapsed) in a payload.
MAX_LISTED_ITEMS = 50

_SEVERITY: dict[str, int] = {name: rank for rank, name in enumerate(CLASSIFICATIONS)}
_DIRECTION_OF_SIDE: dict[str, str] = {"above": "increase", "below": "decrease"}


class AssessmentParams(NamedTuple):
    """Policy of the assessment (all defaults are the plan's numbers).

    Attributes:
        reference_measurements: Size N of the automatic reference window
            (minimum ``MIN_REFERENCE_MEASUREMENTS``).
        last_k: Slots K listed for drill-down and scanned for the bearing
            evidence presence.
        persistence_consecutive: Consecutive out-of-band slots (same side)
            that make a change persistent.
        persistence_min_out_of_last: Out-of-band slots (same side) among the
            last max(K, 5) that make an intermittent change persistent.
        drift_alpha: Significance level of the drift regression.
        band_sigma: Sigma multiplier of the band half-width.
        band_relative_floor: Floor of the half-width as a fraction of the
            reference mean.
        iso_change_fraction: Fraction of the ISO B/C boundary beyond which a
            velocity change is reported as significant.
        acquired_since: ISO 8601 lower bound of the assessed post-reference
            slots (the reference is always used), or None.
        acquired_until: ISO 8601 upper bound, or None.
        thresholds: Comparability tolerances.
    """

    reference_measurements: int = 10
    last_k: int = 5
    persistence_consecutive: int = 3
    persistence_min_out_of_last: int = 4
    drift_alpha: float = 0.05
    band_sigma: float = 3.0
    band_relative_floor: float = 0.25
    iso_change_fraction: float = 0.25
    acquired_since: Optional[str] = None
    acquired_until: Optional[str] = None
    thresholds: ComparabilityThresholds = ComparabilityThresholds()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _instant(value: object) -> Optional[datetime]:
    """ISO 8601 string to an aware UTC datetime (naive read as UTC), or None."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _number(value: object) -> Optional[float]:
    """*value* as a finite float, or None (bools are not numbers here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive_int(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{name} must be an integer >= {minimum}, got {value!r}; pass a "
            f"value in range."
        )


def _fraction(name: str, value: object, *, low: float, high: float) -> None:
    number = _number(value)
    if number is None or not low < number <= high:
        raise ValueError(
            f"{name} must be a number in ({low:g}, {high:g}], got {value!r}."
        )


def validate_params(params: AssessmentParams) -> None:
    """Refuse an out-of-range policy with the limit in the message.

    Raises:
        ValueError: If *params* is not an :class:`AssessmentParams` or one
            of its fields is out of range (a caller bug: the tool boundary
            validates the user's values first).
    """
    if not isinstance(params, AssessmentParams):
        raise ValueError(
            f"params must be an AssessmentParams, got {type(params).__name__}."
        )
    _positive_int(
        "reference_measurements",
        params.reference_measurements,
        MIN_REFERENCE_MEASUREMENTS,
    )
    _positive_int("last_k", params.last_k, 1)
    _positive_int("persistence_consecutive", params.persistence_consecutive, 1)
    _positive_int("persistence_min_out_of_last", params.persistence_min_out_of_last, 1)
    _fraction("drift_alpha", params.drift_alpha, low=0.0, high=1.0)
    band_sigma = _number(params.band_sigma)
    if band_sigma is None or band_sigma <= 0:
        raise ValueError(f"band_sigma must be > 0, got {params.band_sigma!r}.")
    floor = _number(params.band_relative_floor)
    if floor is None or floor < 0:
        raise ValueError(
            f"band_relative_floor must be >= 0, got {params.band_relative_floor!r}."
        )
    _fraction("iso_change_fraction", params.iso_change_fraction, low=0.0, high=1.0)
    for name in ("acquired_since", "acquired_until"):
        bound = getattr(params, name)
        if bound is not None and _instant(bound) is None:
            raise ValueError(
                f"{name} must be an ISO 8601 instant such as "
                f"2026-01-05T09:00:00+01:00, got {bound!r}."
            )
    if not isinstance(params.thresholds, ComparabilityThresholds):
        raise ValueError("thresholds must be a ComparabilityThresholds.")


def _known_points(view: dict[str, Any]) -> list[str]:
    """Declared points plus the points the measurements name, sorted."""
    known: set[str] = {str(key) for key in (view.get("points") or {})}
    for slot in (view.get("measurements") or {}).values():
        current = slot.get("current") if isinstance(slot, dict) else None
        if isinstance(current, dict) and isinstance(
            current.get("measurement_point_id"), str
        ):
            known.add(current["measurement_point_id"])
    return sorted(known)


def _current_point(view: dict[str, Any], point_id: str) -> Optional[dict[str, Any]]:
    slot = (view.get("points") or {}).get(point_id)
    if not isinstance(slot, dict):
        return None
    current = slot.get("current")
    return current if isinstance(current, dict) else None


def _point_changed_at(view: dict[str, Any], point_id: str) -> Optional[str]:
    """``recorded_at`` of the latest declaration when the point was
    re-declared (version > 1), else None."""
    slot = (view.get("points") or {}).get(point_id)
    if not isinstance(slot, dict):
        return None
    history = slot.get("history") or []
    current = slot.get("current") or {}
    version = current.get("declaration_version")
    redeclared = len(history) > 1 or (
        isinstance(version, int) and not isinstance(version, bool) and version > 1
    )
    if not redeclared or not history:
        return None
    recorded = history[-1].get("recorded_at") if isinstance(history[-1], dict) else None
    return recorded if isinstance(recorded, str) else None


def current_snapshot_id_of(
    declaration: dict[str, Any],
    point: Optional[dict[str, Any]],
    *,
    processing: str,
    measurement_id: str,
) -> str:
    """The ``snapshot_id`` the current context and lineage would produce.

    The staleness test shared with the re-processing service: a measurement
    is up to date when its view holds a snapshot with this id, i.e. the
    current ``processing_id`` AND the ``context_digest`` of the current
    point declaration.

    Raises:
        ValueError: If the declaration is malformed (see
            ``snapshot.resolve_context``).
    """
    context = resolve_context(declaration, point)
    return snapshot_id(measurement_id, processing, context_digest(context, declaration))


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------


def _records_of_point(view: dict[str, Any], point_id: str) -> list[dict[str, Any]]:
    """Measurements of the point in the view's order (``acquired_at``, then
    file order), each with its snapshots."""
    measurements = view.get("measurements") or {}
    records: list[dict[str, Any]] = []
    for order, measurement_id in enumerate(view.get("ordered_measurement_ids") or []):
        slot = measurements.get(measurement_id)
        current = slot.get("current") if isinstance(slot, dict) else None
        if not isinstance(current, dict):
            continue
        if current.get("measurement_point_id") != point_id:
            continue
        declaration = current.get("declaration")
        if not isinstance(declaration, dict):
            declaration = {}
        records.append(
            {
                "measurement_id": str(measurement_id),
                "order": order,
                "payload": current,
                "declaration": declaration,
                "snapshots": list(slot.get("snapshots") or []),
                "snapshots_by_lineage": dict(slot.get("snapshots_by_lineage") or {}),
                "lineage_positions": dict(slot.get("lineage_positions") or {}),
            }
        )
    return records


def _time_key(declaration: dict[str, Any]) -> tuple[Any, ...]:
    """Sort/identity key of an acquisition instant (instants first, then
    unparsable strings, then missing)."""
    acquired = declaration.get("acquired_at")
    instant = _instant(acquired)
    if instant is not None:
        return (0, instant.timestamp(), "")
    if isinstance(acquired, str):
        return (1, 0.0, acquired)
    return (2, 0.0, "")


def _collapse_slots(
    records: Sequence[dict[str, Any]], preferred_ids: frozenset[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group the records by acquisition slot; one representative per slot.

    The latest recorded (view order) wins, except that a member of the
    active baseline (*preferred_ids*) wins over a non-member so a declared
    baseline is never silently replaced by a later duplicate export.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        declaration = record["declaration"]
        key = (
            _time_key(declaration),
            declaration.get("direction"),
            declaration.get("sensor_id"),
        )
        groups.setdefault(key, []).append(record)

    slots: list[dict[str, Any]] = []
    collapsed: list[dict[str, Any]] = []
    for key, members in groups.items():
        chosen = max(
            members,
            key=lambda item: (item["measurement_id"] in preferred_ids, item["order"]),
        )
        duplicates = [m["measurement_id"] for m in members if m is not chosen]
        for duplicate in duplicates:
            collapsed.append(
                {
                    "measurement_id": duplicate,
                    "kept_measurement_id": chosen["measurement_id"],
                    "acquired_at": chosen["declaration"].get("acquired_at"),
                    "detail": (
                        "same acquisition slot (acquired_at, direction, sensor_id): "
                        "the latest recorded measurement represents the slot"
                    ),
                }
            )
        declaration = chosen["declaration"]
        slots.append(
            {
                "measurement_id": chosen["measurement_id"],
                "record": chosen,
                "payload": chosen["payload"],
                "declaration": declaration,
                "acquired_at": declaration.get("acquired_at"),
                "instant": _instant(declaration.get("acquired_at")),
                "time_key": key[0],
                "direction": declaration.get("direction"),
                "sensor_id": declaration.get("sensor_id"),
                "order": min(m["order"] for m in members),
                "duplicates": duplicates,
                "collision": False,
                "assessment": None,
            }
        )
    slots.sort(key=lambda slot: (slot["time_key"], slot["order"]))
    by_time: dict[tuple[Any, ...], int] = {}
    for slot in slots:
        by_time[slot["time_key"]] = by_time.get(slot["time_key"], 0) + 1
    for slot in slots:
        slot["collision"] = by_time[slot["time_key"]] > 1
    return slots, collapsed


def _grade(
    slot: dict[str, Any],
    point: Optional[dict[str, Any]],
    context: Optional[dict[str, Any]],
    params: AssessmentParams,
    changed_at: Optional[str],
) -> dict[str, Any]:
    assessment = assess_measurement_comparability(
        slot["payload"],
        point,
        context,
        thresholds=params.thresholds,
        timestamp_collision=bool(slot["collision"]),
        point_declaration_changed_at=changed_at,
    )
    slot["assessment"] = assessment
    return assessment


def _grade_all(
    slots: Sequence[dict[str, Any]],
    point: Optional[dict[str, Any]],
    context: Optional[dict[str, Any]],
    params: AssessmentParams,
    changed_at: Optional[str],
) -> None:
    for slot in slots:
        _grade(slot, point, context, params, changed_at)


def _usable(slots: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in slots if s["assessment"]["grade"] != "non_comparable"]


def _ids(slots: Sequence[dict[str, Any]]) -> list[str]:
    return [str(s["measurement_id"]) for s in slots]


def _codes_of(slot: dict[str, Any]) -> list[str]:
    return [str(q["code"]) for q in slot["assessment"]["qualifications"]]


def _quality(size: int) -> str:
    if size >= 10:
        return "full"
    if size >= 6:
        return "provisional"
    return "relative_only"


def _active_baseline(view: dict[str, Any], point_id: str) -> dict[str, Any]:
    """``{"current": payload|None, "withdrawn": entry|None}`` of the point."""
    slot = (view.get("baselines") or {}).get(point_id)
    if not isinstance(slot, dict):
        return {"current": None, "withdrawn": None}
    current = slot.get("current")
    history = slot.get("history") or []
    withdrawn = None
    if current is None and history and isinstance(history[-1], dict):
        if history[-1].get("withdrawn"):
            withdrawn = history[-1]
    return {
        "current": current if isinstance(current, dict) else None,
        "withdrawn": withdrawn,
    }


def _baseline_members(
    baseline: dict[str, Any],
    slots: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Revalidate the members of a declared baseline against the slots.

    A member must still be a measurement of the point with the declaration
    version the baseline recorded (else ``baseline_member_superseded``) and
    usable (else ``baseline_member_non_comparable``).
    """
    by_id = {slot["measurement_id"]: slot for slot in slots}
    recorded_versions: dict[str, Any] = {}
    for member in baseline.get("members") or []:
        if isinstance(member, dict) and isinstance(member.get("measurement_id"), str):
            recorded_versions[member["measurement_id"]] = member.get(
                "declaration_version"
            )
    valid: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for member_id in baseline.get("measurement_ids") or []:
        member_id = str(member_id)
        slot = by_id.get(member_id)
        if slot is None:
            issues.append(
                {
                    "measurement_id": member_id,
                    "code": "baseline_member_superseded",
                    "detail": (
                        f"{member_id} is no longer a measurement of this point "
                        f"(re-declared on another point or asset, or collapsed "
                        f"into another slot)"
                    ),
                }
            )
            continue
        expected_version = recorded_versions.get(member_id)
        version = slot["payload"].get("declaration_version")
        if expected_version is not None and version != expected_version:
            issues.append(
                {
                    "measurement_id": member_id,
                    "code": "baseline_member_superseded",
                    "detail": (
                        f"{member_id} was validated at declaration version "
                        f"{expected_version} and is now at version {version}"
                    ),
                }
            )
            continue
        if slot["assessment"]["grade"] == "non_comparable":
            issues.append(
                {
                    "measurement_id": member_id,
                    "code": "baseline_member_non_comparable",
                    "detail": (
                        f"{member_id} is non_comparable against the current "
                        f"declarations: {', '.join(_codes_of(slot))}"
                    ),
                }
            )
            continue
        valid.append(slot)
    valid.sort(key=lambda slot: (slot["time_key"], slot["order"]))
    return valid, issues


def collect_point_slots(
    view: dict[str, Any],
    measurement_point_id: str,
    *,
    params: AssessmentParams = AssessmentParams(),
) -> dict[str, Any]:
    """Slots of the point, graded, with the reference selected (no lineage,
    no statistics): the stage shared with the re-processing service.

    Comparability is graded in two passes: first against the point alone
    (unit, direction and amplitude contradictions exclude), then against
    the reference context built from the reference candidates
    (``build_reference_context``); the candidates are refilled when one of
    them drops to ``non_comparable`` and the loop stops at a fixed point.

    Args:
        view: Output of ``store.build_asset_view``.
        measurement_point_id: The point.
        params: Policy (validated by the caller or here).

    Returns:
        A dict with ``found`` (the point is declared or has measurements),
        ``point`` (current declaration or None), ``point_declaration_changed_at``,
        ``slots`` (all slots in ``acquired_at`` order, each with
        ``measurement_id``, ``acquired_at``, ``assessment``, ``duplicates``,
        ``collision``), ``collapsed``, ``usable`` (slots not
        ``non_comparable``), ``reference`` (``kind``, ``slots``,
        ``provisional``, ``statistics_quality``, ``baseline``,
        ``withdrawn``, ``issues``, ``insufficient``), ``sequence`` (usable
        slots after the reference, unwindowed) and ``excluded_inside_span``
        (usable non-members inside a declared baseline's time span).
    """
    validate_params(params)
    point = _current_point(view, measurement_point_id)
    records = _records_of_point(view, measurement_point_id)
    found = point is not None or bool(records)
    changed_at = _point_changed_at(view, measurement_point_id)
    baseline_state = _active_baseline(view, measurement_point_id)
    baseline = baseline_state["current"]
    preferred = (
        frozenset(str(mid) for mid in (baseline.get("measurement_ids") or []))
        if baseline is not None
        else frozenset()
    )

    slots, collapsed = _collapse_slots(records, preferred)
    _grade_all(slots, point, None, params, changed_at)

    reference: dict[str, Any] = {
        "kind": "automatic_window",
        "slots": [],
        "provisional": False,
        "statistics_quality": None,
        "baseline": None,
        "withdrawn": baseline_state["withdrawn"],
        "issues": [],
        "insufficient": None,
    }
    sequence: list[dict[str, Any]] = []
    excluded_inside_span: list[dict[str, Any]] = []
    usable = _usable(slots)
    remedy_baseline = (
        "re-declare the baseline via declare_healthy_baseline with at least "
        f"{MIN_REFERENCE_MEASUREMENTS} measurements of the point that are "
        "comparable against the current declarations"
    )

    if baseline is not None:
        reference["kind"] = "declared_baseline"
        members, issues = _baseline_members(baseline, slots)
        for _ in range(len(slots) + 1):
            context = build_reference_context(
                [m["payload"] for m in members], point, params.thresholds
            )
            _grade_all(slots, point, context, params, changed_at)
            still, new_issues = _baseline_members(baseline, slots)
            if _ids(still) == _ids(members):
                issues = new_issues
                break
            members, issues = still, new_issues
        usable = _usable(slots)
        reference["issues"] = issues
        reference["baseline"] = {
            "baseline_id": baseline.get("baseline_id"),
            "declared_by": baseline.get("declared_by"),
            "declared_at": baseline.get("declared_at"),
            "note": baseline.get("note"),
            "members_declared": len(baseline.get("measurement_ids") or []),
            "members_used": len(members),
        }
        if len(members) < MIN_REFERENCE_MEASUREMENTS:
            reference["insufficient"] = {
                "available": len(members),
                "required": MIN_REFERENCE_MEASUREMENTS,
                "remedy": remedy_baseline,
                "message": (
                    f"declared baseline {baseline.get('baseline_id')} keeps "
                    f"{len(members)} valid member(s) of "
                    f"{len(baseline.get('measurement_ids') or [])} after "
                    f"revalidation ({', '.join(i['code'] for i in issues)}); "
                    f"at least {MIN_REFERENCE_MEASUREMENTS} are needed and the "
                    f"automatic window is not used in place of a declared "
                    f"baseline"
                ),
            }
        reference["slots"] = members
        if members:
            member_ids = set(_ids(members))
            last_key = members[-1]["time_key"]
            for slot in usable:
                if slot["measurement_id"] in member_ids:
                    continue
                if slot["time_key"] > last_key:
                    sequence.append(slot)
                else:
                    excluded_inside_span.append(slot)
    else:
        size = params.reference_measurements
        for _ in range(len(slots) + 1):
            candidates = usable[:size] if len(usable) > size else usable[:-1]
            context = build_reference_context(
                [c["payload"] for c in candidates], point, params.thresholds
            )
            _grade_all(slots, point, context, params, changed_at)
            still = _usable(slots)
            if _ids(still) == _ids(usable):
                break
            usable = still
        if len(usable) < MIN_USABLE_SLOTS:
            reference["insufficient"] = {
                "available": len(usable),
                "required": MIN_USABLE_SLOTS,
                "remedy": (
                    f"load at least {MIN_USABLE_SLOTS - len(usable)} more "
                    f"comparable measurement(s) of the point (a reference of "
                    f"{MIN_REFERENCE_MEASUREMENTS} plus one to assess), or "
                    f"declare a baseline via declare_healthy_baseline"
                ),
                "message": (
                    f"{len(usable)} usable acquisition slot(s) of "
                    f"{len(slots)}; at least {MIN_USABLE_SLOTS} are needed "
                    f"(reference of {MIN_REFERENCE_MEASUREMENTS} plus one to "
                    f"assess) and the reference window asks for "
                    f"{size}"
                ),
            }
            reference["slots"] = list(usable)
        elif len(usable) <= size:
            reference["provisional"] = True
            reference["slots"] = usable[:-1]
            sequence = usable[-1:]
        else:
            reference["slots"] = usable[:size]
            sequence = usable[size:]

    reference["statistics_quality"] = (
        _quality(len(reference["slots"])) if reference["slots"] else None
    )
    return {
        "found": found,
        "point": point,
        "point_declaration_changed_at": changed_at,
        "slots": slots,
        "collapsed": collapsed,
        "usable": usable,
        "reference": reference,
        "sequence": sequence,
        "excluded_inside_span": excluded_inside_span,
    }


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


def _select_lineage(evaluated: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The most recent lineage covering every evaluated slot, or None.

    Returns:
        ``{"processing_id", "counts", "recency"}``: the chosen id (None when
        no lineage covers all), the number of covered slots per lineage and
        the recency key used (position of the latest snapshot).
    """
    counts: dict[str, int] = {}
    recency: dict[str, int] = {}
    for slot in evaluated:
        record = slot["record"]
        positions = record["lineage_positions"]
        for index, snapshot in enumerate(record["snapshots"]):
            processing = (snapshot.get("processing") or {}).get("processing_id")
            if not isinstance(processing, str):
                continue
            position = positions.get(processing)
            recency[processing] = max(
                recency.get(processing, -1),
                int(position) if isinstance(position, int) else index,
            )
        for processing in record["snapshots_by_lineage"]:
            counts[processing] = counts.get(processing, 0) + 1
    total = len(evaluated)
    covering = [pid for pid, count in counts.items() if count == total]
    chosen = (
        max(covering, key=lambda pid: (recency.get(pid, -1), pid)) if covering else None
    )
    return {
        "processing_id": chosen,
        "counts": {pid: counts[pid] for pid in sorted(counts)},
        "recency": recency,
    }


# ---------------------------------------------------------------------------
# Indicators, bands, classification
# ---------------------------------------------------------------------------


def _scale(value: object, factor: float) -> Optional[float]:
    number = _number(value)
    return None if number is None else number * factor


def _slot_values(slot: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Indicator values of one snapshot, amplitudes converted to the
    expected unit with the slot's comparability conversion factor."""
    conversion = slot["assessment"].get("conversion")
    factor = float(conversion["factor"]) if isinstance(conversion, dict) else 1.0
    indicators = snapshot.get("indicators") or {}
    values: dict[str, Optional[float]] = {
        "rms": _scale(indicators.get("rms"), factor),
        "peak": _scale(indicators.get("peak"), factor),
    }
    one_x = snapshot.get("one_x")
    values["one_x"] = (
        _scale(one_x.get("amplitude"), factor) if isinstance(one_x, dict) else None
    )
    iso = snapshot.get("iso")
    values["iso_velocity"] = (
        _number(iso.get("velocity_rms_mm_s")) if isinstance(iso, dict) else None
    )
    bearing = snapshot.get("bearing")
    labels = bearing.get("labels") if isinstance(bearing, dict) else None
    if isinstance(labels, dict):
        for label, block in labels.items():
            if isinstance(block, dict):
                values[f"{ENVELOPE_INDICATOR_PREFIX}{label}"] = _scale(
                    block.get("envelope_amplitude"), factor
                )
    values["crest_factor"] = _number(indicators.get("crest_factor"))
    values["kurtosis"] = _number(indicators.get("kurtosis"))
    return values


def _indicator_order(names: set[str]) -> list[str]:
    """Canonical order: the four amplitudes, envelope labels (catalog order,
    then sorted), then the support indicators."""
    ordered = [name for name in AMPLITUDE_INDICATORS if name in names]
    envelope = [n for n in names if n.startswith(ENVELOPE_INDICATOR_PREFIX)]
    by_label = {n[len(ENVELOPE_INDICATOR_PREFIX) :]: n for n in envelope}
    for label in BEARING_LABELS:
        if label in by_label:
            ordered.append(by_label.pop(label))
    ordered.extend(by_label[label] for label in sorted(by_label))
    ordered.extend(name for name in SUPPORT_INDICATORS if name in names)
    return ordered


def _evidence_present(snapshot: dict[str, Any], label: str) -> bool:
    """Bearing evidence counts as present only with ``detected`` True AND a
    numeric ``magnitude`` (the analyzer can report a detection with no
    magnitude on an empty envelope)."""
    bearing = snapshot.get("bearing")
    labels = bearing.get("labels") if isinstance(bearing, dict) else None
    block = labels.get(label) if isinstance(labels, dict) else None
    if not isinstance(block, dict):
        return False
    return bool(block.get("detected")) and _number(block.get("magnitude")) is not None


def _evidence_strength(snapshot: dict[str, Any], label: str) -> Optional[str]:
    bearing = snapshot.get("bearing")
    labels = bearing.get("labels") if isinstance(bearing, dict) else None
    block = labels.get(label) if isinstance(labels, dict) else None
    if not isinstance(block, dict):
        return None
    strength = block.get("evidence_strength")
    return str(strength) if strength is not None else None


def _band(values: Sequence[float], params: AssessmentParams) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    sigma_half = params.band_sigma * std
    floor_half = params.band_relative_floor * abs(mean)
    half = max(sigma_half, floor_half)
    basis = (
        f"{params.band_sigma:g} sigma"
        if sigma_half >= floor_half and sigma_half > 0
        else f"{params.band_relative_floor * 100:g}% of the mean"
    )
    return {
        "mean": mean,
        "std": std,
        "n": int(array.size),
        "band": {
            "low": mean - half,
            "high": mean + half,
            "half_width": half,
            "basis": basis,
            "degenerate": half == 0.0,
        },
    }


def _band_text(params: AssessmentParams) -> str:
    return (
        f"mean +/- max({params.band_sigma:g} sigma, "
        f"{params.band_relative_floor * 100:g}%)"
    )


def _side(value: float, mean: float, half: float) -> Optional[str]:
    if value > mean + half:
        return "above"
    if value < mean - half:
        return "below"
    return None


def _classify_series(
    values: Sequence[float],
    stats: dict[str, Any],
    params: AssessmentParams,
    previous: Optional[float],
) -> dict[str, Any]:
    """Classify one indicator's post-reference series (see module doc)."""
    mean = float(stats["mean"])
    half = float(stats["band"]["half_width"])
    sides = [_side(v, mean, half) for v in values]
    last_side = sides[-1] if sides else None
    window = max(params.last_k, 5)
    band_text = _band_text(params)

    run = 0
    if last_side is not None:
        for side in reversed(sides):
            if side != last_side:
                break
            run += 1

    recent = sides[-window:]
    above = recent.count("above")
    below = recent.count("below")
    k_side = "above" if above >= below else "below"
    k_count = max(above, below)

    recent_values = [float(v) for v in values[-window:]]
    if len(recent_values) >= 3:
        trend = analyze_trend(recent_values, significance_level=params.drift_alpha)
    else:
        trend = {
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "trend_direction": "stable",
            "p_value": None,
        }
    p_value = _number(trend.get("p_value"))
    slope = float(trend.get("slope") or 0.0)
    drift_significant = p_value is not None and p_value < params.drift_alpha
    drift_agrees = last_side is not None and (
        (slope > 0 and last_side == "above") or (slope < 0 and last_side == "below")
    )

    consecutive = last_side is not None and run >= params.persistence_consecutive
    intermittent = k_count >= params.persistence_min_out_of_last
    drift = drift_significant and drift_agrees

    classification: str
    direction: Optional[str]
    criterion: str
    if consecutive or intermittent or drift:
        classification = "persistent_change"
        if consecutive:
            direction = _DIRECTION_OF_SIDE[str(last_side)]
            criterion = f"{run} consecutive acquisitions outside the band ({band_text})"
        elif intermittent:
            direction = _DIRECTION_OF_SIDE[k_side]
            criterion = (
                f"{k_count} of last {len(recent)} acquisitions outside the band "
                f"on the same side ({band_text})"
            )
        else:
            direction = _DIRECTION_OF_SIDE[str(last_side)]
            criterion = (
                f"significant drift (p={p_value:.2g}) over the last "
                f"{len(recent_values)} acquisitions with the latest value "
                f"outside the band ({band_text})"
            )
    elif last_side is not None:
        classification = "unconfirmed_single_acquisition"
        direction = _DIRECTION_OF_SIDE[str(last_side)]
        drift_text = "not computable" if p_value is None else f"p={p_value:.2g}"
        criterion = (
            f"latest acquisition outside the band ({band_text}) without "
            f"persistence: {run} consecutive, {k_count} of last {len(recent)} on "
            f"the same side, drift {drift_text}"
        )
    elif any(side is not None for side in sides):
        classification = "isolated_episode"
        last_out = [side for side in sides if side is not None][-1]
        direction = _DIRECTION_OF_SIDE[str(last_out)]
        criterion = (
            f"an earlier acquisition exceeded the band ({band_text}) and the "
            f"later ones returned inside it; the latest is inside"
        )
    else:
        classification = "no_change"
        direction = None
        criterion = f"every acquisition inside the band ({band_text})"

    chain = ([previous] if previous is not None else []) + [float(v) for v in values]
    sudden_at: Optional[int] = None
    for index in range(1, len(chain)):
        if abs(chain[index] - chain[index - 1]) > 2.0 * half:
            sudden_at = index - (1 if previous is not None else 0)
            break

    # Onset: the first slot of the episode that fired the verdict (the run
    # ending at the latest slot; the k-of-n window; the last episode that
    # returned inside), never an earlier episode that is over.
    def run_start(end: int) -> int:
        start = end
        while start > 0 and sides[start - 1] == sides[end]:
            start -= 1
        return start

    onset_index: Optional[int] = None
    last_index = len(sides) - 1
    if classification == "persistent_change" and intermittent and not consecutive:
        window_start = len(sides) - len(recent)
        for index in range(window_start, len(sides)):
            if sides[index] == k_side:
                onset_index = index
                break
    elif last_side is not None:
        onset_index = run_start(last_index)
    elif classification == "isolated_episode":
        last_out_index = max(i for i, side in enumerate(sides) if side is not None)
        onset_index = run_start(last_out_index)

    return {
        "classification": classification,
        "direction": direction,
        "criterion": criterion,
        "sudden": sudden_at is not None,
        "sudden_at_index": sudden_at,
        "onset_index": onset_index,
        "sides": sides,
        "exceedance_runs": [side is not None for side in sides[-params.last_k :]],
        "drift": {
            "slope": slope,
            "r_squared": float(trend.get("r_squared") or 0.0),
            "p_value": p_value,
            "trend_direction": str(trend.get("trend_direction") or "stable"),
            "n": len(recent_values),
            "significant": drift_significant,
        },
        "rules": {
            "consecutive": consecutive,
            "consecutive_run": run,
            "intermittent": intermittent,
            "intermittent_count": k_count,
            "intermittent_window": len(recent),
            "drift": drift,
        },
    }


def _classify_evidence(
    presence: Sequence[bool], params: AssessmentParams
) -> dict[str, Any]:
    """Classify one bearing label from its presence over the last K slots.

    ``onset_offset`` is the index (within *presence*) of the first slot of
    the last run of presence: where the current evidence started."""
    longest = 0
    run = 0
    onset_offset: Optional[int] = None
    for offset, present in enumerate(presence):
        if present:
            if run == 0:
                onset_offset = offset
            run += 1
        else:
            run = 0
        longest = max(longest, run)
    count = sum(1 for present in presence if present)
    persistent = longest >= params.persistence_consecutive
    latest = bool(presence[-1]) if presence else False
    if persistent:
        classification = "persistent_change"
        criterion = (
            f"bearing evidence present in {count} of the last {len(presence)} "
            f"acquisitions, {longest} consecutive"
        )
    elif latest:
        classification = "unconfirmed_single_acquisition"
        criterion = (
            f"bearing evidence present in the latest acquisition only "
            f"({count} of the last {len(presence)})"
        )
    elif count:
        classification = "isolated_episode"
        criterion = (
            f"bearing evidence present in {count} of the last {len(presence)} "
            f"acquisitions but not in the latest"
        )
    else:
        classification = "no_change"
        criterion = f"no bearing evidence in the last {len(presence)} acquisitions"
    return {
        "classification": classification,
        "criterion": criterion,
        "persistent": persistent,
        "present_in_last_k": count,
        "k": len(presence),
        "consecutive": longest,
        "latest_present": latest,
        "onset_offset": onset_offset,
    }


# ---------------------------------------------------------------------------
# Payload blocks
# ---------------------------------------------------------------------------


def _slot_ref(slot: dict[str, Any]) -> dict[str, Any]:
    file_block = slot["payload"].get("file")
    location = file_block.get("location") if isinstance(file_block, dict) else None
    return {
        "measurement_id": slot["measurement_id"],
        "acquired_at": slot["acquired_at"],
        "grade": slot["assessment"]["grade"],
        "location": location,
        "signal_id": slot["payload"].get("signal_id"),
    }


def _comparability_block(
    staged: dict[str, Any], baseline_issues: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    slots = staged["slots"]
    summary = summarize_comparability([s["assessment"] for s in slots])
    details: dict[str, str] = {}
    for slot in slots:
        for entry in slot["assessment"]["qualifications"]:
            details.setdefault(str(entry["code"]), str(entry["detail"]))
    qualifications = [
        {"code": code, "count": count, "detail": details.get(code, "")}
        for code, count in summary["codes"].items()
    ]
    for code in BASELINE_QUALIFICATION_CODES:
        matching = [i for i in baseline_issues if i["code"] == code]
        if matching:
            qualifications.append(
                {"code": code, "count": len(matching), "detail": matching[0]["detail"]}
            )
    excluded = [
        {
            "measurement_id": s["measurement_id"],
            "acquired_at": s["acquired_at"],
            "reasons": _codes_of(s),
        }
        for s in slots
        if s["assessment"]["grade"] == "non_comparable"
    ]
    collapsed = list(staged["collapsed"])
    return {
        "comparable": int(summary["grades"]["comparable"]),
        "qualified": int(summary["grades"]["qualified"]),
        "non_comparable": int(summary["grades"]["non_comparable"]),
        "qualifications": qualifications,
        "excluded": excluded[:MAX_LISTED_ITEMS],
        "excluded_truncated": len(excluded) > MAX_LISTED_ITEMS,
        "collapsed_duplicates": collapsed[:MAX_LISTED_ITEMS],
        "collapsed_truncated": len(collapsed) > MAX_LISTED_ITEMS,
    }


def _reference_block(
    staged: dict[str, Any],
    params: AssessmentParams,
    since: Optional[datetime],
    until: Optional[datetime],
) -> dict[str, Any]:
    reference = staged["reference"]
    slots = reference["slots"]
    codes: dict[str, int] = {}
    for slot in slots:
        for code in _codes_of(slot):
            if code not in INFORMATIONAL_CODES:
                codes[code] = codes.get(code, 0) + 1
    ordered_codes = {c: codes[c] for c in QUALIFICATION_CODES if c in codes}
    for code in sorted(codes):
        ordered_codes.setdefault(code, codes[code])

    outside = False
    for slot in slots:
        instant = slot["instant"]
        if since is not None and (instant is None or instant < since):
            outside = True
        if until is not None and (instant is None or instant > until):
            outside = True

    baseline = reference["baseline"]
    withdrawn = reference["withdrawn"]
    if reference["kind"] == "declared_baseline" and baseline is not None:
        health_declared = True
        message = (
            f"declared healthy baseline (declared by {baseline['declared_by']} on "
            f"{baseline['declared_at']})"
        )
        if baseline.get("note"):
            message += f"; note: {baseline['note']}"
        if reference["issues"]:
            message += (
                f"; {baseline['members_used']} of {baseline['members_declared']} "
                f"members used after revalidation"
            )
    else:
        health_declared = False
        message = (
            "health of the reference measurements not declared; relative "
            f"comparison against the first {len(slots)} acquisition slot(s) of "
            "the point"
        )
        if reference["provisional"]:
            message += (
                f" (provisional reference: fewer than "
                f"{params.reference_measurements + 1} usable slots)"
            )
        if withdrawn is not None:
            message += (
                f"; the previous baseline was withdrawn on "
                f"{withdrawn.get('declared_at')} by {withdrawn.get('declared_by')}"
            )
    if ordered_codes:
        message += "; qualified slots inside the reference: " + ", ".join(
            f"{code} ({count})" for code, count in ordered_codes.items()
        )
    if outside:
        message += (
            "; the reference lies outside the requested window and is used anyway"
        )

    return {
        "kind": reference["kind"],
        "health_declared": health_declared,
        "message": message,
        "measurement_ids": _ids(slots),
        "count": len(slots),
        "acquired_from": slots[0]["acquired_at"] if slots else None,
        "acquired_to": slots[-1]["acquired_at"] if slots else None,
        "provisional": bool(reference["provisional"]),
        "statistics_quality": reference["statistics_quality"],
        "qualification_codes": ordered_codes,
        "outside_window": outside,
        "baseline": (
            None
            if baseline is None
            else {**baseline, "excluded": list(reference["issues"])}
        ),
        "withdrawn_baseline": (
            None
            if withdrawn is None
            else {
                "baseline_id": withdrawn.get("baseline_id"),
                "declared_by": withdrawn.get("declared_by"),
                "declared_at": withdrawn.get("declared_at"),
            }
        ),
        "excluded_inside_span": _ids(staged["excluded_inside_span"])[:MAX_LISTED_ITEMS],
    }


def _not_found(
    asset_id: str,
    point_id: str,
    *,
    known_points: list[str],
    known_assets: list[str],
    asset_known: bool,
) -> dict[str, Any]:
    if asset_known:
        message = (
            f"Measurement point {point_id!r} is not declared for asset "
            f"{asset_id!r} and no measurement names it; known points: "
            f"{known_points or 'none'}."
        )
        suggestion = (
            "Use one of the known points, declare the point via "
            "declare_measurement_point, or load a measurement whose companion "
            f'declares "measurement_point_id": "{point_id}".'
        )
    else:
        message = (
            f"Asset {asset_id!r} has no ledger; known assets: "
            f"{known_assets or 'none'}."
        )
        suggestion = (
            "Use one of the known assets, or load a measurement whose companion "
            f'declares "asset_id": "{asset_id}".'
        )
    return {
        "status": "not_found",
        "asset_id": asset_id,
        "measurement_point_id": point_id,
        "suggestion": suggestion,
        "known_points": known_points,
        "known_assets": known_assets,
        "message": message,
    }


def _reprocess_call(asset_id: str, point_id: str) -> str:
    return (
        f"assess_asset_change(asset_id={asset_id!r}, "
        f"measurement_point_id={point_id!r}, reprocess=True)"
    )


def _verification(
    classification: str, coincides: Sequence[str], evidence_driven: bool
) -> Optional[str]:
    if classification == "no_change" or classification == "isolated_episode":
        return None
    if coincides:
        return (
            "Check whether the onset coincides with the "
            f"{', '.join(coincides)} qualification of that acquisition before "
            "attributing the change to the machine."
        )
    if classification == "unconfirmed_single_acquisition":
        return "Repeat the measurement in the same operating conditions."
    if evidence_driven:
        return (
            "Confirm the bearing evidence with diagnose_vibration on the latest "
            "measurement and on a reference measurement."
        )
    return (
        "Compare the spectrum of the latest measurement with a reference "
        "measurement (diagnose_vibration) to characterize the change."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def assess_change(
    view: dict[str, Any],
    asset_id: str,
    measurement_point_id: str,
    *,
    params: AssessmentParams = AssessmentParams(),
    known_assets: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Assess the change of one measurement point from the asset view.

    See the module docstring for the vocabulary. The function is pure and
    deterministic: the same view and parameters give the same payload.

    Args:
        view: Output of ``store.build_asset_view`` (or ``read_view``) for
            *asset_id*; an empty view means an unknown asset.
        asset_id: The asset (for the messages and the remedies).
        measurement_point_id: The point to assess.
        params: Policy; see :class:`AssessmentParams`.
        known_assets: The ids the ledger directory lists (reported on
            ``not_found``); None means unknown.

    Returns:
        A dict with ``status`` in :data:`ASSESSMENT_STATUSES` and:

        * ``not_found``: ``suggestion``, ``known_points``, ``known_assets``,
          ``message``.
        * ``insufficient_history``: ``available``, ``required``, ``remedy``,
          ``reference`` (the partial selection), ``comparability``,
          ``message``.
        * ``processing_not_homogeneous``: ``lineages`` ({processing_id:
          covered slots}), ``evaluated_slots``, ``missing_for_current``,
          ``current_processing_id``, ``remedy``, ``reference``,
          ``comparability``, ``message``.
        * ``assessed``: ``reference``, ``lineage``, ``observed``,
          ``derived``, ``assessed``, ``comparability``,
          ``suggested_verification`` (one sentence or None) and ``message``.

        No key is ever named ``error``.

    Raises:
        ValueError: If *params* is out of range or *view* belongs to another
            asset (caller bugs).
    """
    validate_params(params)
    view_asset = view.get("asset_id")
    if view_asset is not None and view_asset != asset_id:
        raise ValueError(
            f"The view belongs to asset {view_asset!r}, not {asset_id!r}; pass "
            f"the view read for the asset being assessed."
        )
    assets = sorted(str(a) for a in (known_assets or []))
    known_points = _known_points(view)
    asset_known = bool(
        view.get("event_count") or view.get("points") or view.get("measurements")
    )
    if not asset_known:
        return _not_found(
            asset_id,
            measurement_point_id,
            known_points=known_points,
            known_assets=assets,
            asset_known=False,
        )

    staged = collect_point_slots(view, measurement_point_id, params=params)
    if not staged["found"]:
        return _not_found(
            asset_id,
            measurement_point_id,
            known_points=known_points,
            known_assets=assets,
            asset_known=True,
        )

    since = _instant(params.acquired_since)
    until = _instant(params.acquired_until)
    reference = staged["reference"]
    comparability = _comparability_block(staged, reference["issues"])
    reference_block = _reference_block(staged, params, since, until)
    base: dict[str, Any] = {
        "asset_id": asset_id,
        "measurement_point_id": measurement_point_id,
    }

    insufficient = reference["insufficient"]
    if insufficient is not None:
        return {
            "status": "insufficient_history",
            **base,
            "available": int(insufficient["available"]),
            "required": int(insufficient["required"]),
            "remedy": insufficient["remedy"],
            "reference": reference_block,
            "comparability": comparability,
            "message": f"Insufficient history: {insufficient['message']}.",
        }

    sequence: list[dict[str, Any]] = []
    for slot in staged["sequence"]:
        instant = slot["instant"]
        if since is not None and (instant is None or instant < since):
            continue
        if until is not None and (instant is None or instant > until):
            continue
        sequence.append(slot)
    if not sequence:
        window = " and ".join(
            f"{name}={value}"
            for name, value in (
                ("acquired_since", params.acquired_since),
                ("acquired_until", params.acquired_until),
            )
            if value is not None
        )
        return {
            "status": "insufficient_history",
            **base,
            "available": 0,
            "required": 1,
            "remedy": (
                "widen the window (acquired_since / acquired_until) or load a "
                "measurement acquired after the reference"
            ),
            "reference": reference_block,
            "comparability": comparability,
            "message": (
                "Insufficient history: no usable acquisition after the reference"
                + (f" inside the requested window ({window})" if window else "")
                + "."
            ),
        }

    reference_slots: list[dict[str, Any]] = reference["slots"]
    evaluated = reference_slots + sequence
    lineage = _select_lineage(evaluated)
    current_processing = processing_id()
    missing_for_current = sum(
        1
        for slot in evaluated
        if current_processing not in slot["record"]["snapshots_by_lineage"]
    )
    chosen = lineage["processing_id"]
    if chosen is None:
        reference_missing = [
            s["measurement_id"]
            for s in reference_slots
            if current_processing not in s["record"]["snapshots_by_lineage"]
        ]
        text = (
            f"No processing lineage covers every one of the {len(evaluated)} "
            f"evaluated acquisition slots (reference plus post-reference): "
            f"{lineage['counts'] or 'no snapshot at all'}."
        )
        if reference["kind"] == "declared_baseline" and reference_missing:
            text += (
                f" {len(reference_missing)} member(s) of the declared baseline "
                f"lack a snapshot of the current lineage."
            )
        return {
            "status": "processing_not_homogeneous",
            **base,
            "lineages": lineage["counts"],
            "evaluated_slots": len(evaluated),
            "missing_for_current": missing_for_current,
            "current_processing_id": current_processing,
            "remedy": (
                f"call {_reprocess_call(asset_id, measurement_point_id)} to "
                f"re-process the missing snapshots with the current lineage "
                f"(idempotent, limited per call; repeat until remaining is 0)"
            ),
            "reference": reference_block,
            "comparability": comparability,
            "message": text,
        }

    snapshots = {
        s["measurement_id"]: s["record"]["snapshots_by_lineage"][chosen]
        for s in evaluated
    }
    point = staged["point"]
    stale_context = 0
    for slot in evaluated:
        try:
            expected = current_snapshot_id_of(
                slot["declaration"],
                point,
                processing=chosen,
                measurement_id=slot["measurement_id"],
            )
        except ValueError:
            continue
        if snapshots[slot["measurement_id"]].get("snapshot_id") != expected:
            stale_context += 1

    values = {
        s["measurement_id"]: _slot_values(s, snapshots[s["measurement_id"]])
        for s in evaluated
    }
    names: set[str] = set()
    for per_slot in values.values():
        names.update(per_slot)
    ordered_names = _indicator_order(names)
    available: list[str] = []
    unavailable: dict[str, Any] = {}
    for name in ordered_names:
        missing_in = [
            s["measurement_id"]
            for s in evaluated
            if values[s["measurement_id"]].get(name) is None
        ]
        if missing_in:
            reason = None
            block = (
                "one_x"
                if name == "one_x"
                else (
                    "iso"
                    if name == "iso_velocity"
                    else (
                        "bearing"
                        if name.startswith(ENVELOPE_INDICATOR_PREFIX)
                        else None
                    )
                )
            )
            if block is not None:
                entry = (snapshots[missing_in[0]].get("missing") or {}).get(block)
                if isinstance(entry, dict):
                    reason = entry.get("reason")
            unavailable[name] = {
                "missing_in": len(missing_in),
                "of": len(evaluated),
                "reason": reason,
            }
        else:
            available.append(name)

    conversion_target: Optional[str] = None
    if point is not None and point.get("expected_signal_unit") is not None:
        conversion_target = str(point["expected_signal_unit"])
    else:
        for slot in reference_slots:
            conversion = slot["assessment"].get("conversion")
            unit = (
                conversion["to"]
                if isinstance(conversion, dict)
                else slot["declaration"].get("signal_unit")
            )
            if unit is not None:
                conversion_target = str(unit)
                break

    def unit_of(name: str) -> Optional[str]:
        if name == "iso_velocity":
            return "mm/s"
        if name in SUPPORT_INDICATORS:
            return None
        return conversion_target

    reference_statistics: dict[str, Any] = {}
    per_indicator: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    exceedance_runs: dict[str, Any] = {}
    drift: dict[str, Any] = {}
    latest_slot = sequence[-1]
    latest: dict[str, Any] = {}
    for name in available:
        ref_values = [float(values[s["measurement_id"]][name]) for s in reference_slots]
        seq_values = [float(values[s["measurement_id"]][name]) for s in sequence]
        stats = _band(ref_values, params)
        stats["unit"] = unit_of(name)
        reference_statistics[name] = stats
        latest_value = seq_values[-1]
        latest[name] = latest_value
        mean = float(stats["mean"])
        delta_pct = None if mean == 0.0 else (latest_value - mean) / abs(mean) * 100.0
        if name in SUPPORT_INDICATORS:
            deltas[name] = {
                "latest": latest_value,
                "mean": mean,
                "delta_pct": delta_pct,
                "exceeds": None,
                "side": None,
            }
            continue
        verdict = _classify_series(seq_values, stats, params, ref_values[-1])
        per_indicator[name] = verdict
        deltas[name] = {
            "latest": latest_value,
            "mean": mean,
            "delta_pct": delta_pct,
            "exceeds": verdict["sides"][-1] is not None,
            "side": verdict["sides"][-1],
        }
        exceedance_runs[name] = verdict["exceedance_runs"]
        drift[name] = verdict["drift"]

    iso_change: Optional[dict[str, Any]] = None
    if "iso_velocity" in available:
        iso_block = snapshots[latest_slot["measurement_id"]].get("iso") or {}
        group = iso_block.get("machine_group")
        support = iso_block.get("support_type")
        if group is not None and support is not None:
            try:
                boundaries = get_zone_boundaries(group, support)
            except ValueError:
                boundaries = None
            if boundaries is not None:
                boundary_bc = float(boundaries[1])
                threshold = params.iso_change_fraction * boundary_bc
                delta = latest["iso_velocity"] - float(
                    reference_statistics["iso_velocity"]["mean"]
                )
                iso_change = {
                    "machine_group": group,
                    "support_type": support,
                    "boundary_bc_mm_s": boundary_bc,
                    "threshold_mm_s": threshold,
                    "delta_mm_s": delta,
                    "exceeded": abs(delta) > threshold,
                    "fraction": params.iso_change_fraction,
                }

    # Bearing evidence over the last K slots of the sequence.
    last_k_slots = sequence[-params.last_k :]
    labels: list[str] = []
    for slot in evaluated:
        bearing = snapshots[slot["measurement_id"]].get("bearing")
        blocks = bearing.get("labels") if isinstance(bearing, dict) else None
        if isinstance(blocks, dict):
            for label in blocks:
                if label not in labels:
                    labels.append(label)
    labels = [label for label in BEARING_LABELS if label in labels] + sorted(
        label for label in labels if label not in BEARING_LABELS
    )
    evidence_presence: dict[str, Any] = {}
    evidence_assessed: dict[str, Any] = {}
    evidence_onsets: dict[str, Optional[int]] = {}
    for label in labels:
        presence = [
            _evidence_present(snapshots[s["measurement_id"]], label)
            for s in last_k_slots
        ]
        verdict = _classify_evidence(presence, params)
        evidence_onsets[label] = verdict["onset_offset"]
        evidence_presence[label] = {
            "present_in_last_k": verdict["present_in_last_k"],
            "k": verdict["k"],
            "presence": presence,
            "latest_evidence_strength": _evidence_strength(
                snapshots[latest_slot["measurement_id"]], label
            ),
        }
        evidence_assessed[label] = {
            "classification": verdict["classification"],
            "persistent": verdict["persistent"],
            "consecutive": verdict["consecutive"],
            "present_in_last_k": verdict["present_in_last_k"],
            "k": verdict["k"],
            "criterion": verdict["criterion"],
        }

    # Overall: the most severe across amplitude indicators and evidence.
    candidates: list[tuple[str, str, str]] = [
        (name, verdict["classification"], verdict["criterion"])
        for name, verdict in per_indicator.items()
    ] + [
        (f"{EVIDENCE_INDICATOR_PREFIX}{label}", v["classification"], v["criterion"])
        for label, v in evidence_assessed.items()
    ]
    overall = "no_change"
    for _, classification, _ in candidates:
        if _SEVERITY[classification] > _SEVERITY[overall]:
            overall = classification
    driving = [
        name for name, cls, _ in candidates if cls == overall and overall != "no_change"
    ]
    if driving:
        criterion = "; ".join(
            f"{name}: {text}" for name, cls, text in candidates if cls == overall
        )
    else:
        criterion = (
            f"latest acquisition inside the band ({_band_text(params)}) of every "
            f"amplitude indicator and no bearing evidence persistence"
        )

    direction: Optional[str] = None
    sudden = False
    onset_slot: Optional[dict[str, Any]] = None
    evidence_driven = False
    if driving:
        first = driving[0]
        if first.startswith(EVIDENCE_INDICATOR_PREFIX):
            evidence_driven = True
            label = first[len(EVIDENCE_INDICATOR_PREFIX) :]
            offset = evidence_onsets.get(label)
            if offset is not None:
                onset_slot = last_k_slots[offset]
            amplitude_driving = [
                n for n in driving if not n.startswith(EVIDENCE_INDICATOR_PREFIX)
            ]
            if amplitude_driving:
                direction = per_indicator[amplitude_driving[0]]["direction"]
        else:
            verdict = per_indicator[first]
            direction = verdict["direction"]
            if verdict["onset_index"] is not None:
                onset_slot = sequence[int(verdict["onset_index"])]
        sudden = any(
            per_indicator[n]["sudden"]
            for n in driving
            if not n.startswith(EVIDENCE_INDICATOR_PREFIX)
        )
    coincides = (
        [c for c in _codes_of(onset_slot) if c not in INFORMATIONAL_CODES]
        if onset_slot is not None
        else []
    )

    reference_text = reference_block["message"]
    words = {
        "no_change": "no change",
        "isolated_episode": "isolated episode",
        "unconfirmed_single_acquisition": "unconfirmed single acquisition",
        "persistent_change": "persistent change",
    }
    headline = words[overall]
    if direction is not None:
        headline += f" ({direction})"
    message = (
        f"{headline} at {measurement_point_id} of {asset_id} over "
        f"{len(sequence)} post-reference acquisition(s): {criterion}. Reference: "
        f"{reference_text}. Lineage {chosen}"
        + (
            f" ({stale_context} snapshot(s) computed with a previous point "
            f"declaration; {_reprocess_call(asset_id, measurement_point_id)} "
            f"recomputes them)"
            if stale_context
            else ""
        )
        + "."
    )

    return {
        "status": "assessed",
        **base,
        "reference": reference_block,
        "lineage": {
            "processing_id": chosen,
            "algorithm_version": (
                snapshots[latest_slot["measurement_id"]].get("processing") or {}
            ).get("algorithm_version"),
            "covered": len(evaluated),
            "candidates": lineage["counts"],
            "current_processing_id": current_processing,
            "is_current": chosen == current_processing,
            "missing_for_current": missing_for_current,
            "stale_context": stale_context,
        },
        "observed": {
            "reference_statistics": reference_statistics,
            "latest": latest,
            "latest_measurement_id": latest_slot["measurement_id"],
            "latest_acquired_at": latest_slot["acquired_at"],
            "evidence_presence": evidence_presence,
            "indicators_unavailable": unavailable,
            "acquisitions_assessed": len(sequence),
            "slots_assessed": len(evaluated),
            "measurement_ids_assessed": [_slot_ref(s) for s in last_k_slots],
        },
        "derived": {
            "deltas": deltas,
            "exceedance_runs": exceedance_runs,
            "drift": drift,
            "iso_change": iso_change,
            "per_indicator": {
                name: {
                    "classification": v["classification"],
                    "direction": v["direction"],
                    "sudden": v["sudden"],
                    "criterion": v["criterion"],
                    "rules": v["rules"],
                }
                for name, v in per_indicator.items()
            },
        },
        "assessed": {
            "classification": overall,
            "direction": direction,
            "sudden": sudden,
            "criterion": criterion,
            "indicators_driving": driving,
            "onset_measurement_id": (
                None if onset_slot is None else onset_slot["measurement_id"]
            ),
            "onset_acquired_at": (
                None if onset_slot is None else onset_slot["acquired_at"]
            ),
            "onset_coincides_with": coincides,
            "evidence": evidence_assessed,
        },
        "comparability": comparability,
        "suggested_verification": _verification(overall, coincides, evidence_driven),
        "message": message,
    }
