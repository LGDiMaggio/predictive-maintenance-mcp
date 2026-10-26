"""
ISO 13374 Block 5 — Prognostic Assessment: indicator projection payload.

Assembles what the ``forecast_asset_indicator`` tool returns: the projection
(or the reason the history cannot support one), the history report, staleness
against the caller-supplied clock, warnings, provenance, and the fixed
server-authored texts. A projection whose band the quantiles cannot carry at
the level of the values is refused rather than rounded to zero width.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import numpy as np

from ..signal_validation import InvalidInputError
from .indicator_series import (
    MAX_HORIZON_DURATION,
    MAX_HORIZON_STEPS,
    Declaration,
    RegularGrid,
    duration_cap_steps,
    epoch_us,
    format_duration,
    format_instant,
    history_report,
    prepare_series,
)
from .reference_forecast import (
    MIN_DIFFERENCE_PAIRS,
    MIN_SEASONS_COVERED,
    QUANTILE_LEVELS,
    Z_P90,
    Insufficient,
    ReferenceForecast,
    reference_forecast,
)

#: Lineage of the numbers. Bump by hand whenever the same inputs would give
#: different numbers; the golden fixture trips otherwise.
FORECAST_FAMILY = "indicator_forecast"
FORECAST_ALGORITHM_VERSION = 1

#: Quantiles keep at least SIGNIFICANT_DIGITS significant digits, and more when
#: the band is narrow against the level, so that BAND_DIGITS digits of the
#: narrowest half-band survive rounding; a zero-width band would read as
#: certainty. MAX_SIGNIFICANT_DIGITS bounds the output size.
SIGNIFICANT_DIGITS = 6
BAND_DIGITS = 3
MAX_SIGNIFICANT_DIGITS = 10

#: Largest ratio of the projection's magnitude to its narrowest half-band the
#: quantiles can carry: up to it, MAX_SIGNIFICANT_DIGITS keep BAND_DIGITS digits
#: of the half-band, so rounding moves no quantile by more than a hundredth of
#: it and P10 < P50 < P90 survives. A projection beyond it is refused.
MAX_LEVEL_TO_BAND_RATIO = 10.0 ** (MAX_SIGNIFICANT_DIGITS - BAND_DIGITS)

#: A half-band within this many float64 spacings of the level has fewer than
#: BAND_DIGITS digits in the values themselves, so centering cannot restore it.
FLOAT_RESOLUTION_SPACINGS = 10**BAND_DIGITS

#: Byte budget of a tool result as the client receives it (the structured
#: content plus its text copy in one CallToolResult). The MCP tool measures each
#: result and refuses one above it; see prognostics_tools.client_result_bytes.
MAX_RESULT_BYTES = 32_000

STATUSES = ("forecast", "insufficient_history")
WARNING_CODES = (
    "stale_history",
    "observed_after_issued_at",
    "partial_last_bin",
    "isolated_origin",
    "interval_crosses_observed_sign",
)

INTERPRETATION = (
    "A projection of the indicator's future values from its own history. It is not a "
    "fault detection, a failure probability or a remaining-useful-life estimate, and "
    "it says nothing about the condition of the machine."
)

METHOD_NOTES = {
    "naive": (
        "Naive reference: the median repeats the last observed bin value at every step "
        "by construction. A flat median is a property of the method, not a prediction "
        "that the indicator will stay stable."
    ),
    "seasonal_naive": (
        "Seasonal naive reference: each step's median repeats the most recent observed "
        "bin at the same phase of the declared season, by construction. A repeating "
        "median is a property of the method, not a prediction that the pattern will "
        "continue."
    ),
}

INTERVAL_NOTE = (
    "P10 and P90 are nominal quantiles under the method's assumption of independent "
    "increments with no trend, estimated from the differences between the supplied "
    "values; their coverage on this series is not validated. Under a sustained trend, "
    "values tend to leave the band on the trend side more often as the horizon grows. "
    "On an indicator that fluctuates around a level, the band is wider than the data "
    "need. Values an export interpolated or held, rather than recorded or averaged, "
    "make the band narrower than the data warrant. The band is not a probability of "
    "crossing any limit and must not be compared with alarm or ISO zone boundaries."
)

_VALUES_MEANING = (
    "Values are the {aggregation} of {indicator} in each {resolution} bin, in {unit}, "
    "as stated in the call. P50 projects that bin value; P10 and P90 bound a nominal "
    "band around it."
)

_AGGREGATION_PHRASES = {
    "mean": "mean",
    "median": "median",
    "max": "maximum",
    "last": "last reading",
}

_NO_FILL = "Do not fill, interpolate or repeat values to create them."


def explain_insufficient(
    result: Insufficient, declaration: Declaration, max_supported: int
) -> tuple[str, str]:
    """Server-authored reason and remedy for an ``insufficient_history`` result."""
    code = result.reason_code
    # Consecutive observed bins that pass both seasonal checks at once, so that
    # following either seasonal remedy cannot run into the other refusal.
    seasonal_bins = max(
        MIN_SEASONS_COVERED * result.lag, result.lag + MIN_DIFFERENCE_PAIRS
    )
    if code == "too_few_difference_pairs":
        if result.backend == "naive":
            what = "pairs of adjacent observed bins"
            fix = (
                "Supply a longer history at this resolution, or a resolution at least "
                "as coarse as the measurement cadence so that consecutive bins are "
                "observed."
            )
        else:
            what = f"pairs of observed bins one season ({result.lag} steps) apart"
            fix = (
                f"Supply a longer history, at least {seasonal_bins} consecutive "
                f"observed bins for a season of {result.lag} steps, or use the naive "
                f"backend."
            )
        return (
            f"Only {result.available} {what} are available; the {result.backend} "
            f"reference needs at least {result.required} to estimate its band.",
            f"{fix} At least {result.required} such pairs are needed. {_NO_FILL}",
        )
    if code == "season_coverage_too_short":
        return (
            f"The history covers {result.available} bins; seasonal naive needs at least "
            f"{MIN_SEASONS_COVERED} full seasons, {result.required} bins at a season of "
            f"{result.lag} steps.",
            f"Supply at least {seasonal_bins} consecutive observed bins of history, or "
            f"use the naive backend.",
        )
    if code == "season_phase_unobserved":
        missing = result.required - result.available
        return (
            f"{missing} of the {result.required} requested steps fall on a phase of the "
            f"season with no observed bin in the history.",
            f"Supply history that covers every phase the horizon needs, or use the naive "
            f"backend. {_NO_FILL}",
        )
    if code == "no_variation":
        if result.backend == "seasonal_naive":
            return (
                "Every pair of bins one season apart is equal, so the series repeats "
                "exactly and the season-to-season spread is zero. That pattern can come "
                "from a replayed, synthetic or interpolated signal, or from a cycle that "
                "repeats exactly at the recorded resolution.",
                "Check the source of the values, or use the naive backend. A zero-width "
                "band would read as certainty, so none is produced.",
            )
        return (
            "Every pair of adjacent observed bins is equal, so the spread is zero. A "
            "flat series can come from a stuck sensor, a held or interpolated value, or "
            "an indicator that did not change at the recorded resolution; it is not "
            "evidence about the condition of the machine.",
            "Check the source of the values before projecting them. A zero-width band "
            "would read as certainty, so none is produced.",
        )
    assert code == "horizon_exceeds_history", code
    step = declaration.step_seconds
    return (
        f"The requested horizon of {result.required} steps "
        f"({format_duration(result.required * step)}) is longer than the history "
        f"allows: at most {max_supported} steps can be projected at resolution "
        f"{declaration.resolution}, because the forecast may not span more than the "
        f"observed history, {MAX_HORIZON_STEPS} steps or "
        f"{MAX_HORIZON_DURATION.days} days.",
        f"Call again with horizon_steps of at most {max_supported}, or supply a longer "
        f"history.",
    )


def forecast_indicator(
    *,
    observed_at: Sequence[Any],
    values: Sequence[Any],
    indicator: Any,
    unit: Any,
    resolution: Any,
    horizon_steps: Any,
    backend: str = "naive",
    aggregation: str = "mean",
    season_length_steps: Any = None,
    now: datetime,
    server_version: str,
) -> dict[str, Any]:
    """Project one indicator series and return the tool's payload as a dict.

    Raises:
        InvalidInputError: For every call error (see ``indicator_series``).
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    declaration, grid = prepare_series(
        observed_at=observed_at,
        values=values,
        indicator=indicator,
        unit=unit,
        resolution=resolution,
        horizon_steps=horizon_steps,
        backend=backend,
        aggregation=aggregation,
        season_length_steps=season_length_steps,
        now=now,
    )
    history = history_report(grid, declaration.lag)
    max_supported = _max_supported_horizon(grid, declaration)
    outcome = reference_forecast(
        grid.values,
        backend=declaration.backend,
        horizon_steps=declaration.horizon_steps,
        season_length_steps=declaration.season_length_steps,
        max_supported_horizon_steps=max_supported,
    )

    now_us = epoch_us(now)
    last_obs_us = int(grid.instants_us[-1])
    step_us = grid.step_us
    since = max(0, (now_us - last_obs_us) // step_us)
    already_past = int(
        min(
            max(0, (now_us - grid.next_bin_start_us) // step_us),
            declaration.horizon_steps,
        )
    )

    payload: dict[str, Any] = {
        "status": "forecast",
        "indicator": declaration.indicator,
        "unit": declaration.unit,
        "backend": declaration.backend,
        "resolution": declaration.resolution,
        "horizon_steps": declaration.horizon_steps,
        "issued_at": format_instant(now_us),
        "last_observed_at": history["last_observed_at"],
        "steps_since_last_observation": int(since),
        "forecast_steps_already_past": already_past,
        "forecast_start": None,
        "horizon_end": None,
        "p10": None,
        "p50": None,
        "p90": None,
        "anchors": None,
        "reason_code": None,
        "reason": None,
        "available": None,
        "required": None,
        "max_supported_horizon_steps": None,
        "remedy": None,
        "warnings": [],
        "history": history,
        "provenance": _provenance(grid, declaration, server_version),
        "interpretation": INTERPRETATION,
        "values_meaning": _VALUES_MEANING.format(
            aggregation=_AGGREGATION_PHRASES[declaration.aggregation],
            indicator=declaration.indicator,
            resolution=declaration.resolution,
            unit=declaration.unit,
        ),
        "method_note": None,
        "interval_note": None,
    }

    warnings: list[dict[str, str]] = []
    if last_obs_us > now_us:
        warnings.append(
            {
                "code": "observed_after_issued_at",
                "detail": (
                    f"The last observation ({history['last_observed_at']}) is after the "
                    f"server clock ({payload['issued_at']}); check the source clock and the "
                    f"server clock. The staleness counts are reported as zero."
                ),
            }
        )
    if history["last_bin_points"] < history["median_points_per_observed_bin"]:
        warnings.append(
            {
                "code": "partial_last_bin",
                "detail": (
                    f"The last bin holds {history['last_bin_points']} point(s) against a "
                    f"median of {history['median_points_per_observed_bin']:g} per observed "
                    f"bin, so the most recent bin value may rest on a partial bin."
                ),
            }
        )
    if history["empty_bins_before_last_observed"] >= declaration.horizon_steps:
        warnings.append(
            {
                "code": "isolated_origin",
                "detail": (
                    f"The last observed bin follows {history['empty_bins_before_last_observed']} "
                    f"empty bin(s), at least as many as the horizon, so the most recent "
                    f"value stands alone after a gap."
                ),
            }
        )

    if isinstance(outcome, Insufficient):
        reason, remedy = explain_insufficient(outcome, declaration, max_supported)
        payload.update(
            status="insufficient_history",
            reason_code=outcome.reason_code,
            reason=reason,
            available=int(outcome.available),
            required=int(outcome.required),
            max_supported_horizon_steps=int(max_supported),
            remedy=remedy,
            warnings=warnings,
        )
        return payload

    _refuse_unreportable_band(outcome, declaration.unit, grid.raw_values)
    payload.update(_projection(outcome, grid, declaration))
    if already_past > 0:
        if already_past >= declaration.horizon_steps:
            consequence = (
                "so every projected step is already in the past: this projection is "
                "not a view of the future."
            )
        else:
            consequence = f"so the first {already_past} projected step(s) are already in the past."
        warnings.insert(
            0,
            {
                "code": "stale_history",
                "detail": (
                    f"The last observation is {since} step(s) before the issue time, "
                    f"{consequence}"
                ),
            },
        )
    crossing = _sign_crossing(grid, outcome)
    if crossing:
        warnings.append({"code": "interval_crosses_observed_sign", "detail": crossing})
    payload["warnings"] = warnings
    payload["method_note"] = METHOD_NOTES[declaration.backend]
    payload["interval_note"] = INTERVAL_NOTE
    return payload


def _max_supported_horizon(grid: RegularGrid, declaration: Declaration) -> int:
    by_span = grid.observed_span_us // grid.step_us
    by_duration = duration_cap_steps(declaration.step_seconds)
    return int(min(by_span, MAX_HORIZON_STEPS, by_duration))


def _band_scale(outcome: ReferenceForecast) -> tuple[float, float]:
    """Narrowest half-band and largest quantile magnitude of a projection."""
    half = min(
        float(np.min(outcome.p90 - outcome.p50)),
        float(np.min(outcome.p50 - outcome.p10)),
    )
    level = float(np.max(np.abs(np.concatenate((outcome.p10, outcome.p90)))))
    return half, level


def _refuse_unreportable_band(
    outcome: ReferenceForecast, unit: str, raw_values: np.ndarray
) -> None:
    """Refuse a band the quantiles cannot carry at the level of the values.

    The emit check uses the half-band as computed in float64, which is what
    gets serialized. The diagnosis uses the method's own half-band
    (Z_P90 * sigma), which a far-away level can round to zero, and picks the
    remedy by cause, in this order: a seasonal pattern no constant can shift
    away; variation at the rounding limit of the values as sent; a median far
    from the values whose differences set the band; otherwise a level that
    subtracting the last observed value removes.

    Raises:
        InvalidInputError: When the narrowest half-band is zero or under
            1/MAX_LEVEL_TO_BAND_RATIO of the largest quantile magnitude.
    """
    half, level = _band_scale(outcome)
    if half > 0.0 and level <= MAX_LEVEL_TO_BAND_RATIO * half:
        return
    intrinsic = Z_P90 * outcome.sigma
    near = float(outcome.p50[int(np.argmax(np.abs(outcome.p50)))])
    typical = float(np.median(np.abs(raw_values)))
    pattern = float(np.max(outcome.p50) - np.min(outcome.p50))
    spacing = float(np.spacing(typical))
    carried = f"the quantiles carry at most {MAX_SIGNIFICANT_DIGITS} significant digits"
    if pattern / 2 > MAX_LEVEL_TO_BAND_RATIO * intrinsic:
        raise InvalidInputError(
            f"The projection's band is too narrow to report against its seasonal "
            f"pattern: its median spans {pattern:.6g} {unit} across the horizon "
            f"while the season-to-season spread gives a narrowest half-width of "
            f"{intrinsic:.3g} {unit}, and {carried}, so no constant subtracted "
            f"from the values can bring the band within reach. Seasons that repeat "
            f"this closely can come from a replayed, synthetic or interpolated "
            f"signal, or from a cycle that repeats almost exactly at the recorded "
            f"resolution; check the source of the values, or use the naive backend."
        )
    problem = (
        f"The projection's band is too narrow to report at the level of these "
        f"values: its narrowest half-width is {intrinsic:.3g} {unit} on values "
        f"near {near:.10g} {unit}"
    )
    if intrinsic < FLOAT_RESOLUTION_SPACINGS * spacing:
        raise InvalidInputError(
            f"{problem}, less than {FLOAT_RESOLUTION_SPACINGS:,} times the "
            f"floating-point spacing of the values ({spacing:.3g} {unit}). Fewer "
            f"than three digits of this variation survive in the values as sent, "
            f"so subtracting a constant from them would report a band that rests "
            f"partly on rounding: export the series again with an offset removed "
            f"at the source, before the values are converted to floating point, "
            f"for example as differences from a reference reading."
        )
    if typical <= MAX_LEVEL_TO_BAND_RATIO * intrinsic:
        raise InvalidInputError(
            f"The projection's band is too narrow to report at the level it "
            f"reaches: the band, with a narrowest half-width of {intrinsic:.3g} "
            f"{unit}, comes from values near {typical:.10g} {unit}, but its median "
            f"reaches {near:.10g} {unit}, far from them, and {carried}. A value "
            f"this far from the rest can be a unit change, a fill or sentinel "
            f"value, a counter reset or a genuine change of level; check the last "
            f"values at the source before projecting the series."
        )
    raise InvalidInputError(
        f"{problem}, less than one part in {MAX_LEVEL_TO_BAND_RATIO:,.0f} of "
        f"the level, and {carried}, so the band could round to zero width and "
        f"read as certainty. The values sent still resolve this variation: "
        f"subtract the last observed value exactly from every value, name it in "
        f"the indicator (for example 'energy minus 123456 kWh') and add it back "
        f"when quoting the quantiles. This removes the level the band is measured "
        f"against; with the seasonal_naive backend it may not be enough when the "
        f"seasonal pattern is large next to its season-to-season spread."
    )


def _significant_digits(outcome: ReferenceForecast) -> int:
    """Digits that keep BAND_DIGITS digits of the narrowest half-band.

    Called only after _refuse_unreportable_band, so the half-band is positive
    and the cap does not bind.
    """
    half, level = _band_scale(outcome)
    needed = math.ceil(math.log10(level / half)) + BAND_DIGITS
    return min(max(SIGNIFICANT_DIGITS, needed), MAX_SIGNIFICANT_DIGITS)


def _round(values: np.ndarray, digits: int) -> list[float]:
    return [float(f"{x:.{digits}g}") for x in values]


def _projection(
    outcome: ReferenceForecast, grid: RegularGrid, declaration: Declaration
) -> dict[str, Any]:
    horizon = declaration.horizon_steps
    step_us = grid.step_us
    first_bin_us = grid.next_bin_start_us
    digits = _significant_digits(outcome)
    p10, p50, p90 = (_round(q, digits) for q in (outcome.p10, outcome.p50, outcome.p90))
    anchor_steps = sorted(
        {1, round(horizon / 3), round(2 * horizon / 3), horizon} - {0}
    )
    anchors = [
        {
            "step": k,
            "bin_start": format_instant(first_bin_us + (k - 1) * step_us),
            "bin_end": format_instant(first_bin_us + k * step_us),
            "p10": p10[k - 1],
            "p50": p50[k - 1],
            "p90": p90[k - 1],
        }
        for k in anchor_steps
    ]
    return {
        "forecast_start": format_instant(first_bin_us),
        "horizon_end": format_instant(first_bin_us + horizon * step_us),
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "anchors": anchors,
    }


def _sign_crossing(grid: RegularGrid, outcome: ReferenceForecast) -> Optional[str]:
    raw = grid.raw_values
    if np.all(raw > 0) and np.min(outcome.p10) < 0:
        return (
            "Every observed value is positive, but P10 falls below zero: the band is the "
            "method's symmetric Gaussian band and does not respect the indicator's sign."
        )
    if np.all(raw < 0) and np.max(outcome.p90) > 0:
        return (
            "Every observed value is negative, but P90 rises above zero: the band is the "
            "method's symmetric Gaussian band and does not respect the indicator's sign."
        )
    return None


def _provenance(
    grid: RegularGrid, declaration: Declaration, server_version: str
) -> dict[str, Any]:
    effective = {**declaration.as_dict(), "quantile_levels": list(QUANTILE_LEVELS)}
    digest_input = {
        "family": FORECAST_FAMILY,
        "algorithm_version": FORECAST_ALGORITHM_VERSION,
        "observed_at": [format_instant(int(t)) for t in grid.instants_us],
        "values": [float(v) for v in grid.raw_values],
        "declaration": effective,
    }
    canonical = json.dumps(
        digest_input,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return {
        "family": FORECAST_FAMILY,
        "algorithm_version": FORECAST_ALGORITHM_VERSION,
        "effective_declaration": effective,
        "input_digest": hashlib.sha256(canonical.encode("ascii")).hexdigest()[:16],
        "server_version": server_version,
    }
