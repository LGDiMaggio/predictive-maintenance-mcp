"""
ISO 13374 Block 3 — State Detection (asset ledger comparability).

Grades ONE measurement against the current declaration of its measurement
point and against the context of the reference measurements it will be
trended with. ISO 20816-3:2022, 6.3 judges a change against a reference
established "in the same position and orientation of the transducer and
approximately in the same operating conditions"; ISO 17359:2018, 8.2 and
8.4 bind a baseline to a predetermined operating condition and require a
fault to be told from a change of load or speed. This module is where
those requirements become reportable codes.

Rule of the module: **a contradiction excludes, an absence qualifies**.

- A declared value that CONTRADICTS the point or the reference (another
  unit family, another direction, a speed beyond tolerance, amplitudes that
  are not physical) makes the measurement ``non_comparable``: it stays
  recorded, it is excluded from the trend, and the reason is reported.
- A value that is simply NOT DECLARED (direction, rpm, timezone) or that
  differs in a way no model corrects (sensor, operating condition,
  acquisition setup) makes it ``qualified``: it enters the trend with the
  caveat attached.
- A same-family unit difference is ``comparable`` with the conversion
  factor recorded (informational qualification ``unit_converted``).

Pure functions: no file access, no MCP, no repository, no models. The unit
families come from the measurement contract
(``signal_acquisition.measurement.UNIT_FAMILIES``) and the conversion
constants from ``diagnostics.iso20816`` (single source, never redefined).
Consumers: the load-time registration grades against the point alone
(``assess_measurement_comparability(measurement, point)``); the query-time
assessment builds the reference context with :func:`build_reference_context`
and grades every measurement against point plus context. The grade is
computed at query time against the CURRENT declarations and reported at
load time as information; it is never stored as a fact.
"""

import logging
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np

from ..diagnostics.iso20816 import G_TO_M_S2, M_S_TO_MM_S
from ..signal_acquisition.measurement import UNIT_FAMILIES, unit_family

logger = logging.getLogger(__name__)

__all__ = [
    "COMPARABILITY_GRADES",
    "QUALIFICATION_CODES",
    "EXCLUDING_CODES",
    "INFORMATIONAL_CODES",
    "ComparabilityThresholds",
    "DEFAULT_THRESHOLDS",
    "unit_conversion_factor",
    "build_reference_context",
    "assess_measurement_comparability",
    "grade_of",
    "summarize_comparability",
]

#: The three grades, from best to worst.
COMPARABILITY_GRADES: tuple[str, ...] = ("comparable", "qualified", "non_comparable")

#: Closed vocabulary of qualification codes, grouped by concern in the order
#: the rules run (unit, direction, sensor, speed, operating condition,
#: acquisition setup, amplitude, timestamps, point declaration).
QUALIFICATION_CODES: tuple[str, ...] = (
    "unit_not_declared",
    "unit_incompatible",
    "unit_converted",
    "direction_mismatch",
    "direction_differs_from_reference",
    "direction_not_declared",
    "sensor_changed",
    "rpm_not_declared",
    "no_rpm_anchor",
    "rpm_deviation_qualified",
    "rpm_deviation_excluded",
    "operating_condition_differs",
    "acquisition_setup_differs",
    "amplitude_not_physical",
    "timezone_not_declared",
    "timestamp_suspect",
    "timestamp_collision",
    "point_declaration_changed",
)

#: Codes that make a measurement ``non_comparable`` (a contradiction).
EXCLUDING_CODES: frozenset[str] = frozenset(
    {
        "unit_not_declared",
        "unit_incompatible",
        "direction_mismatch",
        "direction_differs_from_reference",
        "rpm_deviation_excluded",
        "amplitude_not_physical",
    }
)

#: Codes that are reported but do NOT lower the grade.
INFORMATIONAL_CODES: frozenset[str] = frozenset({"unit_converted"})

#: Scale of every canonical unit relative to the base unit of its family
#: (m/s2 for acceleration, mm/s for velocity), so that
#: ``factor(from, to) = _UNIT_SCALE[from] / _UNIT_SCALE[to]``. Built from the
#: two ISO constants; the key set equals the union of ``UNIT_FAMILIES``
#: (asserted by test).
_UNIT_SCALE: dict[str, float] = {
    "g": G_TO_M_S2,
    "m/s2": 1.0,
    "mm/s": 1.0,
    "m/s": M_S_TO_MM_S,
}

#: Human labels of the anchor sources named in the details.
_SOURCE_LABELS: dict[str, str] = {
    "nominal_rpm": "nominal_rpm of the point",
    "expected_direction": "expected_direction of the point",
    "expected_sensor_id": "expected_sensor_id of the point",
    "expected_signal_unit": "expected_signal_unit of the point",
    "reference_median": "median of the reference measurements",
    "reference_majority": "majority of the reference measurements",
}


class ComparabilityThresholds(NamedTuple):
    """Tolerances of the comparability rules, all in percent.

    Attributes:
        rpm_comparable_pct: Speed deviation from the anchor up to which a
            measurement is comparable without qualification. ISO 20816-3
            only asks for "approximately the same" conditions; +-5% of
            speed is about +-10% of unbalance response, inside the 25%
            change band.
        rpm_qualified_pct: Deviation up to which the measurement is
            qualified (reported with the percentage); beyond it, excluded.
        sampling_rate_tolerance_pct: Relative difference from the reference
            median sampling rate tolerated before
            ``acquisition_setup_differs``.
        load_tolerance_pct: Relative difference from the reference median
            load tolerated before ``operating_condition_differs`` (the
            median of an even-sized reference can fall between declared
            values, so an exact comparison would flag every member).
    """

    rpm_comparable_pct: float = 5.0
    rpm_qualified_pct: float = 10.0
    sampling_rate_tolerance_pct: float = 1.0
    load_tolerance_pct: float = 5.0


#: The thresholds used when the caller passes none.
DEFAULT_THRESHOLDS = ComparabilityThresholds()


# ---------------------------------------------------------------------------
# Payload access and formatting helpers
# ---------------------------------------------------------------------------


def _declaration_of(measurement: dict[str, Any]) -> dict[str, Any]:
    """Return the ``declaration`` block of a payload, or the bare declaration.

    A ``measurement_recorded`` payload carries its declaration under the
    ``"declaration"`` key; a bare declaration dict (no such key) is used as
    is, so both the ledger service and the tools can call the rules.
    """
    declaration = measurement.get("declaration")
    if isinstance(declaration, dict):
        return declaration
    return measurement


def _location_of(measurement: dict[str, Any]) -> Optional[str]:
    """Return the declared file location (payload ``file.location`` first)."""
    file_block = measurement.get("file")
    if isinstance(file_block, dict) and file_block.get("location") is not None:
        return str(file_block["location"])
    for source in (measurement, _declaration_of(measurement)):
        for key in ("location", "filepath"):
            if source.get(key) is not None:
                return str(source[key])
    return None


def _measurement_id_of(measurement: dict[str, Any]) -> Optional[str]:
    """Return the measurement id from the payload or its declaration."""
    for source in (measurement, _declaration_of(measurement)):
        value = source.get("measurement_id")
        if value is not None:
            return str(value)
    return None


def _scale_factor_of(declaration: dict[str, Any]) -> Optional[float]:
    """Return the effective raw ``scale_factor``; None for self-describing files."""
    raw_format = declaration.get("raw_format")
    if isinstance(raw_format, dict):
        return _as_float(raw_format.get("scale_factor"))
    return None


def _as_float(value: object) -> Optional[float]:
    """Return *value* as a finite float, or None (bool is not a number here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _num(value: float) -> str:
    """Format a number for a detail: ``1480`` not ``1480.0``, ``26585.3``."""
    return f"{float(value):.10g}"


def _pct(value: float) -> str:
    """Format a signed percentage: ``+5.4%``, ``+100%``, ``-7.5%``."""
    text = f"{value:+.1f}"
    if text.endswith(".0"):
        text = text[:-2]
    return f"{text}%"


def _differs(value: float, reference: float, tolerance_pct: float) -> bool:
    """True when *value* is outside ``reference +- tolerance_pct``."""
    if reference == 0.0:
        return value != reference
    return abs(value - reference) / abs(reference) * 100.0 > tolerance_pct


def _majority(values: Sequence[Any]) -> Any:
    """Most frequent declared (non-None) value; ties go to the first in order."""
    declared = [value for value in values if value is not None]
    if not declared:
        return None
    counts = Counter(declared)
    best = max(counts.values())
    for value in declared:
        if counts[value] == best:
            return value
    return None  # pragma: no cover - every declared value has a count


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    """Median of the declared (non-None) values, or None."""
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return float(np.median(np.asarray(numbers, dtype=float)))


def _parse_instant(value: object) -> Optional[datetime]:
    """Parse an ISO 8601 string to an aware UTC datetime (naive = UTC)."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _describe_source(source: object) -> str:
    """Human label of an anchor source for the details."""
    return _SOURCE_LABELS.get(str(source), "the reference context")


def _family_name(unit: str) -> str:
    """Family of a unit for the details ("unknown family" when outside)."""
    return unit_family(unit) or "unknown family"


def _qualification(code: str, detail: str) -> dict[str, str]:
    """Build one qualification entry."""
    return {"code": code, "detail": detail}


def _expected(
    point: dict[str, Any],
    point_key: str,
    context: dict[str, Any],
    context_key: str,
) -> tuple[Any, Optional[str]]:
    """Resolve an anchor: the point's declaration wins, else the context's.

    Returns:
        ``(value, source_label)``; ``(None, None)`` when neither declares it.
    """
    value = point.get(point_key)
    if value is not None:
        return value, _describe_source(point_key)
    value = context.get(context_key)
    if value is not None:
        return value, _describe_source(context.get(f"{context_key}_source"))
    return None, None


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def unit_conversion_factor(
    from_unit: Optional[str], to_unit: Optional[str]
) -> Optional[float]:
    """Multiplier that converts amplitudes from one canonical unit to another.

    Args:
        from_unit: Canonical unit of the values ("g", "m/s2", "mm/s", "m/s").
        to_unit: Canonical unit wanted.

    Returns:
        1.0 for the same unit; ``G_TO_M_S2`` for g to m/s2 and its inverse
        back; ``M_S_TO_MM_S`` for m/s to mm/s and its inverse back; None
        across families (acceleration vs velocity) or when either unit is
        undeclared or unknown (no guessing).
    """
    if from_unit is None or to_unit is None:
        return None
    family = unit_family(from_unit)
    if family is None or family != unit_family(to_unit):
        return None
    if from_unit == to_unit:
        return 1.0
    return _UNIT_SCALE[from_unit] / _UNIT_SCALE[to_unit]


# ---------------------------------------------------------------------------
# Reference context
# ---------------------------------------------------------------------------


def _anchor(
    point_value: Any, point_key: str, reference_values: Sequence[Any]
) -> tuple[Any, Optional[str]]:
    """Point declaration when present, else the majority of the reference."""
    if point_value is not None:
        return point_value, point_key
    majority = _majority(reference_values)
    return majority, ("reference_majority" if majority is not None else None)


def build_reference_context(
    reference_measurements: Sequence[dict[str, Any]],
    point: Optional[dict[str, Any]] = None,
    thresholds: ComparabilityThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Build the context every measurement is graded against at query time.

    The caller chooses the reference measurements (the declared baseline, or
    the first N acquisition slots by ``acquired_at`` after the NON-rpm
    exclusions, per the two-pass anchor rule); this function only computes
    the anchors. The point's declaration wins over the reference wherever it
    declares a value.

    Args:
        reference_measurements: ``measurement_recorded`` payloads (or bare
            declarations) of the reference.
        point: Current ``measurement_point_declared`` payload, or None.
        thresholds: Tolerances, recorded in the context for reporting.

    Returns:
        A dict with the keys ``reference_count``, ``measurement_ids``,
        ``rpm_anchor`` / ``rpm_anchor_source`` ("nominal_rpm",
        "reference_median" or None), ``direction`` / ``direction_source``,
        ``sensor_id`` / ``sensor_id_source``, ``unit`` / ``unit_source``
        (sources: the point key, "reference_majority" or None),
        ``operating_state`` (majority), ``load`` (median), ``sampling_rate``
        (median, Hz), ``scale_factor`` (majority of the declared raw scale
        factors), ``point_declaration_version`` and ``thresholds``.
    """
    point_context = point or {}
    declarations = [_declaration_of(m) for m in reference_measurements]

    nominal_rpm = _as_float(point_context.get("nominal_rpm"))
    rpm_anchor_source: Optional[str]
    if nominal_rpm is not None and nominal_rpm > 0:
        rpm_anchor: Optional[float] = nominal_rpm
        rpm_anchor_source = "nominal_rpm"
    else:
        rpm_anchor = _median([_as_float(d.get("rpm")) for d in declarations])
        rpm_anchor_source = "reference_median" if rpm_anchor is not None else None

    direction, direction_source = _anchor(
        point_context.get("expected_direction"),
        "expected_direction",
        [d.get("direction") for d in declarations],
    )
    sensor_id, sensor_source = _anchor(
        point_context.get("expected_sensor_id"),
        "expected_sensor_id",
        [d.get("sensor_id") for d in declarations],
    )
    unit, unit_source = _anchor(
        point_context.get("expected_signal_unit"),
        "expected_signal_unit",
        [d.get("signal_unit") for d in declarations],
    )

    return {
        "reference_count": len(declarations),
        "measurement_ids": [_measurement_id_of(m) for m in reference_measurements],
        "rpm_anchor": rpm_anchor,
        "rpm_anchor_source": rpm_anchor_source,
        "direction": direction,
        "direction_source": direction_source,
        "sensor_id": sensor_id,
        "sensor_id_source": sensor_source,
        "unit": unit,
        "unit_source": unit_source,
        "operating_state": _majority([d.get("operating_state") for d in declarations]),
        "load": _median([_as_float(d.get("load")) for d in declarations]),
        "sampling_rate": _median(
            [_as_float(d.get("sampling_rate")) for d in declarations]
        ),
        "scale_factor": _majority([_scale_factor_of(d) for d in declarations]),
        "point_declaration_version": point_context.get("declaration_version"),
        "thresholds": dict(thresholds._asdict()),
    }


# ---------------------------------------------------------------------------
# Rules (one function per concern, in the order the codes are listed)
# ---------------------------------------------------------------------------


def _assess_unit(
    declaration: dict[str, Any],
    point: dict[str, Any],
    context: dict[str, Any],
    qualifications: list[dict[str, str]],
) -> Optional[dict[str, Any]]:
    """Unit rule: undeclared excludes, another family excludes, same family
    converts. Returns the conversion block when one applies."""
    unit = declaration.get("signal_unit")
    expected, source = _expected(point, "expected_signal_unit", context, "unit")
    if unit is None:
        target = expected if expected is not None else "|".join(_UNIT_SCALE)
        qualifications.append(
            _qualification(
                "unit_not_declared",
                f"signal_unit not declared: re-load with signal_unit={target} "
                f"(a new declaration supersedes the previous one); amplitude "
                f"indicators need a declared unit",
            )
        )
        return None
    if expected is None or expected == unit:
        return None
    factor = unit_conversion_factor(unit, expected)
    if factor is None:
        qualifications.append(
            _qualification(
                "unit_incompatible",
                f"{unit} vs {expected} ({_family_name(unit)} vs "
                f"{_family_name(expected)}, {source}): amplitudes in different "
                f"unit families are not comparable",
            )
        )
        return None
    qualifications.append(
        _qualification(
            "unit_converted",
            f"{unit} converted to {expected} ({source}), factor {factor:.6g}",
        )
    )
    return {"from": unit, "to": expected, "factor": factor}


def _assess_direction(
    declaration: dict[str, Any],
    point: dict[str, Any],
    context: dict[str, Any],
    qualifications: list[dict[str, str]],
) -> None:
    """Direction rule: undeclared qualifies; a different declared direction is
    another measurement point (against the point, else the reference)."""
    direction = declaration.get("direction")
    if direction is None:
        qualifications.append(
            _qualification(
                "direction_not_declared",
                "direction not declared: the measurement cannot be told from "
                "other axes of the same point; declare direction in the "
                "companion",
            )
        )
        return
    expected = point.get("expected_direction")
    if expected is not None:
        if direction != expected:
            qualifications.append(
                _qualification(
                    "direction_mismatch",
                    f"{direction} vs {expected} (expected_direction of the "
                    f"point): another direction is another measurement point; "
                    f"axis and machine directions are never mapped onto each "
                    f"other",
                )
            )
        return
    reference = context.get("direction")
    if reference is not None and direction != reference:
        qualifications.append(
            _qualification(
                "direction_differs_from_reference",
                f"{direction} vs {reference} "
                f"({_describe_source(context.get('direction_source'))}): the "
                f"point declares no expected_direction, so the direction of "
                f"the reference measurements is the anchor",
            )
        )


def _assess_sensor(
    declaration: dict[str, Any],
    point: dict[str, Any],
    context: dict[str, Any],
    qualifications: list[dict[str, str]],
) -> None:
    """Sensor rule: a different declared sensor qualifies (no calibration
    model); an undeclared sensor is not graded (no code exists for it)."""
    sensor = declaration.get("sensor_id")
    if sensor is None:
        return
    expected, source = _expected(point, "expected_sensor_id", context, "sensor_id")
    if expected is not None and sensor != expected:
        qualifications.append(
            _qualification(
                "sensor_changed",
                f"{sensor} vs {expected} ({source}): no calibration model "
                f"relates the two sensors, amplitude offsets between them are "
                f"not corrected",
            )
        )


def _assess_rpm(
    declaration: dict[str, Any],
    point: dict[str, Any],
    context: dict[str, Any],
    has_context: bool,
    thresholds: ComparabilityThresholds,
    qualifications: list[dict[str, str]],
) -> None:
    """Speed rule: undeclared qualifies ("constant regime"); a declared speed
    is graded against the anchor when one exists."""
    rpm = _as_float(declaration.get("rpm"))
    if rpm is None:
        qualifications.append(
            _qualification(
                "rpm_not_declared",
                "rpm not declared: trend assumes constant regime (declare rpm "
                "in the companion to grade speed deviations)",
            )
        )
        return

    anchor: Optional[float] = None
    source: Optional[str] = None
    if has_context:
        anchor = _as_float(context.get("rpm_anchor"))
        source = context.get("rpm_anchor_source")
    if anchor is None:
        anchor = _as_float(point.get("nominal_rpm"))
        source = "nominal_rpm" if anchor is not None else None
    if anchor is not None and anchor <= 0:
        anchor = None
    if anchor is None:
        if has_context:
            qualifications.append(
                _qualification(
                    "no_rpm_anchor",
                    f"rpm {_num(rpm)} declared but no rpm anchor: the point "
                    f"declares no nominal_rpm and the reference measurements "
                    f"declare no rpm",
                )
            )
        return

    deviation = (rpm - anchor) / anchor * 100.0
    magnitude = abs(deviation)
    if magnitude <= thresholds.rpm_comparable_pct:
        return
    if magnitude <= thresholds.rpm_qualified_pct:
        qualifications.append(
            _qualification(
                "rpm_deviation_qualified",
                f"{_num(rpm)} vs {_num(anchor)} rpm, {_pct(deviation)}, "
                f"tolerance ±{_num(thresholds.rpm_comparable_pct)}% (anchor: "
                f"{_describe_source(source)}; excluded beyond "
                f"±{_num(thresholds.rpm_qualified_pct)}%)",
            )
        )
        return
    qualifications.append(
        _qualification(
            "rpm_deviation_excluded",
            f"{_num(rpm)} vs {_num(anchor)} rpm, {_pct(deviation)}, "
            f"tolerance ±{_num(thresholds.rpm_qualified_pct)}% (anchor: "
            f"{_describe_source(source)})",
        )
    )


def _assess_operating_conditions(
    declaration: dict[str, Any],
    context: dict[str, Any],
    thresholds: ComparabilityThresholds,
    qualifications: list[dict[str, str]],
) -> None:
    """Operating-condition rule: a declared state or load that differs from
    the reference qualifies (ISO 17359 8.4: a change of load is not a
    fault)."""
    parts: list[str] = []
    state = declaration.get("operating_state")
    reference_state = context.get("operating_state")
    if state is not None and reference_state is not None and state != reference_state:
        parts.append(
            f"operating_state {state!r} vs {reference_state!r} (majority of the "
            f"reference measurements)"
        )
    load = _as_float(declaration.get("load"))
    reference_load = _as_float(context.get("load"))
    if (
        load is not None
        and reference_load is not None
        and _differs(load, reference_load, thresholds.load_tolerance_pct)
    ):
        parts.append(
            f"load {_num(load)} vs {_num(reference_load)} (median of the "
            f"reference measurements, tolerance "
            f"±{_num(thresholds.load_tolerance_pct)}%)"
        )
    if parts:
        qualifications.append(
            _qualification("operating_condition_differs", "; ".join(parts))
        )


def _assess_acquisition_setup(
    declaration: dict[str, Any],
    context: dict[str, Any],
    thresholds: ComparabilityThresholds,
    qualifications: list[dict[str, str]],
) -> None:
    """Acquisition-setup rule: a sampling rate outside tolerance or a raw
    scale factor different from the reference's qualifies, with both
    values in the detail."""
    parts: list[str] = []
    sampling_rate = _as_float(declaration.get("sampling_rate"))
    reference_rate = _as_float(context.get("sampling_rate"))
    if (
        sampling_rate is not None
        and reference_rate is not None
        and _differs(
            sampling_rate, reference_rate, thresholds.sampling_rate_tolerance_pct
        )
    ):
        parts.append(
            f"sampling_rate {_num(sampling_rate)} vs {_num(reference_rate)} Hz "
            f"(median of the reference measurements, tolerance "
            f"±{_num(thresholds.sampling_rate_tolerance_pct)}%)"
        )
    scale = _scale_factor_of(declaration)
    reference_scale = _as_float(context.get("scale_factor"))
    if scale is not None and reference_scale is not None and scale != reference_scale:
        parts.append(
            f"scale_factor {_num(scale)} vs {_num(reference_scale)} (majority of "
            f"the reference measurements)"
        )
    if parts:
        qualifications.append(
            _qualification(
                "acquisition_setup_differs",
                "; ".join(parts) + ": acquisition setup differs from the reference",
            )
        )


def _assess_amplitude(
    location: Optional[str],
    declaration: dict[str, Any],
    qualifications: list[dict[str, str]],
) -> bool:
    """Amplitude rule: a WAV file (the loader normalises integer PCM to
    [-1, 1]) or a raw integer file without ``scale_factor`` (ADC counts)
    has no physical amplitudes. Returns True when excluded."""
    if location is not None and location.lower().endswith(".wav"):
        qualifications.append(
            _qualification(
                "amplitude_not_physical",
                "WAV file: integer samples are normalised to [-1, 1] by the "
                "loader, so amplitudes are not physical; export the capture as "
                "CSV or raw float32 with a declared signal_unit",
            )
        )
        return True
    raw_format = declaration.get("raw_format")
    if isinstance(raw_format, dict):
        sample_format = str(raw_format.get("sample_format") or "").lower()
        if (
            sample_format.startswith(("int", "uint"))
            and raw_format.get("scale_factor") is None
        ):
            qualifications.append(
                _qualification(
                    "amplitude_not_physical",
                    f"raw {sample_format} samples without scale_factor: ADC "
                    f"counts, not physical units; re-load with scale_factor="
                    f"<physical units per count> (a new declaration supersedes "
                    f"the previous one)",
                )
            )
            return True
    return False


def _assess_timestamps(
    declaration: dict[str, Any],
    point: dict[str, Any],
    timestamp_collision: bool,
    point_declaration_changed_at: Optional[str],
    qualifications: list[dict[str, str]],
) -> None:
    """Timestamp rules: naive instant, implausible instant, slot collision and
    a point declaration that changed after the measurement all qualify."""
    acquired_at = declaration.get("acquired_at")
    if declaration.get("timezone_declared") is False:
        qualifications.append(
            _qualification(
                "timezone_not_declared",
                f"acquired_at {acquired_at} declared without a UTC offset: "
                f"ordered as UTC; declare the offset (or 'Z') in the companion",
            )
        )
    if declaration.get("timestamp_suspect"):
        qualifications.append(
            _qualification(
                "timestamp_suspect",
                f"acquired_at {acquired_at} is implausible (1970 epoch or more "
                f"than one day in the future): check the clock of the "
                f"acquisition device",
            )
        )
    if timestamp_collision:
        qualifications.append(
            _qualification(
                "timestamp_collision",
                f"another measurement of the same point declares the same "
                f"acquired_at {acquired_at}: same acquisition slot",
            )
        )
    if point_declaration_changed_at is None:
        return
    changed = _parse_instant(point_declaration_changed_at)
    acquired = _parse_instant(acquired_at)
    if changed is None or acquired is None:
        logger.debug(
            "point_declaration_changed not graded: unparsable instant "
            "(changed_at=%r, acquired_at=%r)",
            point_declaration_changed_at,
            acquired_at,
        )
        return
    if acquired < changed:
        version = point.get("declaration_version")
        suffix = f" (version {version})" if version is not None else ""
        qualifications.append(
            _qualification(
                "point_declaration_changed",
                f"acquired {acquired_at}, before the current point declaration "
                f"of {point_declaration_changed_at}{suffix}: graded against the "
                f"new context",
            )
        )


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


def grade_of(qualifications: Sequence[dict[str, str]]) -> str:
    """Grade implied by a list of qualifications.

    ``non_comparable`` if any code is in ``EXCLUDING_CODES``; else
    ``qualified`` if any code is not informational; else ``comparable``.
    """
    codes = {entry["code"] for entry in qualifications}
    if codes & EXCLUDING_CODES:
        return "non_comparable"
    if codes - INFORMATIONAL_CODES:
        return "qualified"
    return "comparable"


def assess_measurement_comparability(
    measurement: dict[str, Any],
    point: Optional[dict[str, Any]] = None,
    reference_context: Optional[dict[str, Any]] = None,
    *,
    thresholds: ComparabilityThresholds = DEFAULT_THRESHOLDS,
    timestamp_collision: bool = False,
    point_declaration_changed_at: Optional[str] = None,
) -> dict[str, Any]:
    """Grade one measurement against its point and the reference context.

    Without a point and a reference context only the intrinsic checks run:
    unit declared, direction declared, rpm declared, physical amplitudes,
    timestamps. With a point, its ``expected_*`` and ``nominal_rpm`` are
    the anchors; with a reference context (see
    :func:`build_reference_context`) the context fills what the point does
    not declare and adds the operating-condition and acquisition-setup
    comparisons.

    Args:
        measurement: A ``measurement_recorded`` payload (``declaration`` and
            ``file`` blocks) or a bare declaration dict.
        point: Current ``measurement_point_declared`` payload, or None.
        reference_context: Output of :func:`build_reference_context`, or
            None (load time: no reference yet).
        thresholds: Tolerances of the speed, load and sampling-rate rules.
        timestamp_collision: True when another measurement of the same
            point declares the same ``acquired_at`` (decided by the caller,
            which sees the whole history).
        point_declaration_changed_at: ISO 8601 instant of the current point
            declaration; a measurement acquired before it is qualified.

    Returns:
        ``{"grade", "qualifications", "conversion",
        "excluded_from_amplitude_trend"}``: the grade (one of
        ``COMPARABILITY_GRADES``); the qualifications, each
        ``{"code", "detail"}`` with the numbers in the detail, in rule
        order; the conversion ``{"from", "to", "factor"}`` to the expected
        unit when the families match and the units differ, else None; and
        whether the amplitude indicators are not physical
        (``amplitude_not_physical``; any other non-comparable grade already
        excludes the whole measurement).
    """
    declaration = _declaration_of(measurement)
    point_context = point or {}
    context = reference_context or {}
    qualifications: list[dict[str, str]] = []

    conversion = _assess_unit(declaration, point_context, context, qualifications)
    _assess_direction(declaration, point_context, context, qualifications)
    _assess_sensor(declaration, point_context, context, qualifications)
    _assess_rpm(
        declaration,
        point_context,
        context,
        reference_context is not None,
        thresholds,
        qualifications,
    )
    _assess_operating_conditions(declaration, context, thresholds, qualifications)
    _assess_acquisition_setup(declaration, context, thresholds, qualifications)
    not_physical = _assess_amplitude(
        _location_of(measurement), declaration, qualifications
    )
    _assess_timestamps(
        declaration,
        point_context,
        timestamp_collision,
        point_declaration_changed_at,
        qualifications,
    )

    return {
        "grade": grade_of(qualifications),
        "qualifications": qualifications,
        "conversion": conversion,
        "excluded_from_amplitude_trend": not_physical,
    }


def summarize_comparability(assessments: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count assessments per grade and qualifications per code.

    Args:
        assessments: Outputs of :func:`assess_measurement_comparability`.

    Returns:
        ``{"total", "grades", "codes"}``: ``grades`` always has the three
        grades (zero included); ``codes`` lists only the codes that occur,
        in ``QUALIFICATION_CODES`` order, followed by any code a caller
        added outside this vocabulary, sorted.

    Raises:
        ValueError: If an assessment carries a grade outside
            ``COMPARABILITY_GRADES``.
    """
    grades = {grade: 0 for grade in COMPARABILITY_GRADES}
    codes: Counter[str] = Counter()
    for assessment in assessments:
        grade = assessment.get("grade")
        if grade not in grades:
            raise ValueError(
                f"Unknown comparability grade {grade!r}: expected one of "
                f"{list(COMPARABILITY_GRADES)}."
            )
        grades[grade] += 1
        for entry in assessment.get("qualifications", ()):
            codes[entry["code"]] += 1
    ordered = {code: codes[code] for code in QUALIFICATION_CODES if codes[code]}
    for code in sorted(codes):
        if code not in ordered:
            ordered[code] = codes[code]
    return {"total": len(assessments), "grades": grades, "codes": ordered}
