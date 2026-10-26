"""
ISO 13374 Block 5 — Prognostic Assessment: indicator series for projection.

Turns a caller-supplied indicator series (ISO 8601 instants and values, with a
declared unit and grid resolution) into a regular UTC grid, without
interpolating, filling or deleting anything. Empty bins stay empty; several
observations in one bin combine by the declared aggregation.

Every call error that follows from the call's own fields and timestamps is
refused here, in the caller's vocabulary, before any computation. Values too
large to aggregate or difference without overflow are refused later, where the
overflow happens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import numpy as np

from ..signal_acquisition.measurement import normalize_acquired_at, validate_free_text
from ..signal_validation import InvalidInputError, validate_finite_samples

BACKENDS = ("naive", "seasonal_naive")
AGGREGATIONS = ("mean", "median", "max", "last")

#: Provisional product limits: they bound the work and the output and are not
#: validated forecasting thresholds.
MAX_INPUT_POINTS = 2048
MAX_GRID_BINS = 100_000
MAX_HORIZON_STEPS = 256
MAX_HORIZON_DURATION = timedelta(days=30)

RESOLUTION_UNITS = {"s": 1, "min": 60, "h": 3600, "d": 86400, "w": 604800}
_RESOLUTION_RE = re.compile(rf"([1-9][0-9]*)({'|'.join(RESOLUTION_UNITS)})")

BIN_ALIGNMENT = "utc_epoch"

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ONE_US = timedelta(microseconds=1)
_US_PER_S = 1_000_000

_GAP_REMEDY = (
    "omit the instant and value of a missing observation from both lists; "
    "do not fill, interpolate or replace it with zero"
)


@dataclass(frozen=True)
class Declaration:
    """The caller's declaration after defaults, as the forecast uses it."""

    indicator: str
    unit: str
    resolution: str
    step_seconds: int
    aggregation: str
    horizon_steps: int
    backend: str
    season_length_steps: Optional[int]

    @property
    def lag(self) -> int:
        """Difference lag the backend uses: 1, or the season length."""
        if self.backend == "seasonal_naive" and self.season_length_steps:
            return self.season_length_steps
        return 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "indicator": self.indicator,
            "unit": self.unit,
            "resolution": self.resolution,
            "resolution_seconds": self.step_seconds,
            "aggregation": self.aggregation,
            "horizon_steps": self.horizon_steps,
            "backend": self.backend,
            "season_length_steps": self.season_length_steps,
        }


@dataclass(frozen=True)
class RegularGrid:
    """A series on its UTC grid. ``values`` holds NaN where a bin is empty."""

    start_us: int
    step_us: int
    values: np.ndarray
    counts: np.ndarray
    instants_us: np.ndarray
    raw_values: np.ndarray

    @property
    def bins(self) -> int:
        return int(self.values.size)

    @property
    def last_bin_start_us(self) -> int:
        return self.start_us + (self.bins - 1) * self.step_us

    @property
    def next_bin_start_us(self) -> int:
        """Start of the first bin after the history: projection step 1."""
        return self.start_us + self.bins * self.step_us

    @property
    def observed_span_us(self) -> int:
        """Last observed instant minus first observed instant."""
        return int(self.instants_us[-1] - self.instants_us[0])


def parse_resolution(text: object) -> int:
    """Return the grid step in seconds, or refuse a value outside the grammar."""
    match = _RESOLUTION_RE.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        raise InvalidInputError(_resolution_problem(text))
    return int(match.group(1)) * RESOLUTION_UNITS[match.group(2)]


def epoch_us(dt: datetime) -> int:
    """Whole microseconds since 1970-01-01T00:00Z of a timezone-aware instant."""
    return (dt - _EPOCH) // _ONE_US


def format_instant(us: int) -> str:
    """UTC ISO 8601 with ``Z``; microseconds only when non-zero."""
    dt = _EPOCH + timedelta(microseconds=int(us))
    return dt.isoformat().replace("+00:00", "Z")


def duration_cap_steps(step_seconds: int) -> int:
    """The most horizon steps of ``step_seconds`` that fit in the duration cap."""
    return int(MAX_HORIZON_DURATION.total_seconds() // step_seconds)


def format_duration(seconds: float) -> str:
    """Compact duration for messages, at most two units, such as '1 d 12 h'."""
    seconds = int(seconds)
    if seconds == 0:
        return "0 s"
    parts = []
    for label, size in (("d", 86400), ("h", 3600), ("min", 60), ("s", 1)):
        count, seconds = divmod(seconds, size)
        if count:
            parts.append(f"{count} {label}")
    return " ".join(parts[:2])


def prepare_series(
    *,
    observed_at: Sequence[Any],
    values: Sequence[Any],
    indicator: Any,
    unit: Any,
    resolution: Any,
    horizon_steps: Any,
    backend: str,
    aggregation: str,
    season_length_steps: Any,
    now: datetime,
) -> tuple[Declaration, RegularGrid]:
    """Validate a call and build the regular grid.

    Raises:
        InvalidInputError: For every call error, with the problem and the
            remedy in the caller's vocabulary.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    _check_sizes(observed_at, values)
    declaration, finite_values = _check_declaration(
        indicator=indicator,
        unit=unit,
        resolution=resolution,
        horizon_steps=horizon_steps,
        backend=backend,
        aggregation=aggregation,
        season_length_steps=season_length_steps,
        values=values,
    )
    instants_us = _parse_instants(observed_at, now)
    grid = _build_grid(instants_us, finite_values, declaration)
    return declaration, grid


def history_report(grid: RegularGrid, lag: int) -> dict[str, Any]:
    """Counts that let a caller check the transcription and the data quality."""
    v = grid.values
    observed = ~np.isnan(v)
    counts = grid.counts
    diffs = lag_differences(v, lag)
    zero_share = float(np.mean(diffs == 0.0)) if diffs.size else None
    observed_before_last = np.flatnonzero(counts[:-1])
    after_previous = (
        int(observed_before_last[-1]) + 1 if observed_before_last.size else 0
    )
    empty_before_last = counts.size - 1 - after_previous
    return {
        "raw_points": int(grid.raw_values.size),
        "first_observed_at": format_instant(int(grid.instants_us[0])),
        "last_observed_at": format_instant(int(grid.instants_us[-1])),
        "last_value": float(grid.raw_values[-1]),
        "min_value": float(np.min(grid.raw_values)),
        "max_value": float(np.max(grid.raw_values)),
        "grid_bins": grid.bins,
        "observed_bins": int(observed.sum()),
        "empty_bins": int((~observed).sum()),
        "bins_with_several_points": int((counts > 1).sum()),
        "last_bin_points": int(counts[-1]),
        "median_points_per_observed_bin": float(np.median(counts[counts > 0])),
        "empty_bins_before_last_observed": empty_before_last,
        "difference_lag": int(lag),
        "difference_pairs": int(diffs.size),
        "zero_difference_share": zero_share,
        "bin_alignment": BIN_ALIGNMENT,
    }


def lag_differences(values: np.ndarray, lag: int) -> np.ndarray:
    """Differences between bins exactly ``lag`` steps apart, both observed."""
    if lag < 1 or lag >= values.size:
        return np.empty(0)
    later, earlier = values[lag:], values[:-lag]
    both = ~np.isnan(later) & ~np.isnan(earlier)
    # An overflow becomes infinity here and is refused by the backend.
    with np.errstate(over="ignore", invalid="ignore"):
        diffs: np.ndarray = later[both] - earlier[both]
    return diffs


def _check_sizes(observed_at: Sequence[Any], values: Sequence[Any]) -> None:
    n_obs, n_val = len(observed_at), len(values)
    largest = max(n_obs, n_val)
    if largest > MAX_INPUT_POINTS:
        raise InvalidInputError(
            f"{largest} points exceed the limit of {MAX_INPUT_POINTS}: pass the "
            f"most recent {MAX_INPUT_POINTS} points, or aggregate the series at "
            f"the source to a coarser resolution."
        )
    if n_obs == 0 or n_val == 0:
        raise InvalidInputError(
            "observed_at and values must each hold at least one entry: pass the "
            "measured series, one value per instant."
        )
    if n_obs != n_val:
        raise InvalidInputError(
            f"observed_at has {n_obs} entries but values has {n_val}: pass one "
            f"value per instant, in the same order."
        )


def _check_declaration(
    *,
    indicator: Any,
    unit: Any,
    resolution: Any,
    horizon_steps: Any,
    backend: str,
    aggregation: str,
    season_length_steps: Any,
    values: Sequence[Any],
) -> tuple[Declaration, np.ndarray]:
    problems: list[str] = []

    for field, value, problem in (
        (
            "unit",
            unit,
            "unit must be declared: pass the indicator's unit as recorded at the "
            "source, for example 'mm/s' or 'degC'; it is never inferred, so ask "
            "the user when it is unknown.",
        ),
        (
            "indicator",
            indicator,
            "indicator must name the measured quantity, for example "
            "'velocity RMS' or 'bearing temperature'.",
        ),
    ):
        if not isinstance(value, str) or not value.strip():
            problems.append(problem)
            continue
        try:
            validate_free_text(field, value)
        except ValueError as exc:
            problems.append(str(exc))

    step_seconds: Optional[int] = None
    allowed: Optional[int] = None
    try:
        step_seconds = parse_resolution(resolution)
    except InvalidInputError as exc:
        problems.append(str(exc))
    else:
        allowed = duration_cap_steps(step_seconds)
        if allowed == 0:
            days = MAX_HORIZON_DURATION.days
            problems.append(
                f"resolution {resolution} is longer than the {days}-day horizon "
                f"limit, so not even one step fits: use a resolution of at most "
                f"{days}d, for example '4w' or '1d'."
            )

    horizon_ok = (
        isinstance(horizon_steps, int)
        and not isinstance(horizon_steps, bool)
        and 1 <= horizon_steps <= MAX_HORIZON_STEPS
    )
    if not horizon_ok:
        problems.append(
            f"horizon_steps must be an integer between 1 and {MAX_HORIZON_STEPS}, "
            f"got {horizon_steps!r}."
        )
    elif step_seconds is not None and allowed:
        if horizon_steps > allowed:
            problems.append(
                f"horizon_steps {horizon_steps} at resolution {resolution} spans "
                f"{format_duration(horizon_steps * step_seconds)}, over the "
                f"{MAX_HORIZON_DURATION.days}-day limit; at this resolution pass "
                f"at most {allowed} steps."
            )

    if backend not in BACKENDS:
        problems.append(f"backend must be one of {list(BACKENDS)}, got {backend!r}.")
    if aggregation not in AGGREGATIONS:
        problems.append(
            f"aggregation must be one of {list(AGGREGATIONS)}, got {aggregation!r}."
        )

    season: Optional[int] = None
    if backend == "seasonal_naive":
        if season_length_steps is None:
            problems.append(
                "season_length_steps is required for the seasonal_naive backend: "
                "declare the season the user states, in grid steps, for example 24 "
                "when the user says the pattern repeats daily at resolution '1h'; "
                "it is never inferred."
            )
        elif (
            not isinstance(season_length_steps, int)
            or isinstance(season_length_steps, bool)
            or season_length_steps < 2
        ):
            problems.append(
                f"season_length_steps must be an integer of at least 2, got "
                f"{season_length_steps!r}."
            )
        else:
            season = season_length_steps
    elif season_length_steps is not None:
        problems.append(
            "season_length_steps applies only to the seasonal_naive backend: omit "
            "it for the naive backend."
        )

    finite_values = np.empty(0)
    first_null = next((i for i, v in enumerate(values) if v is None), None)
    if first_null is not None:
        problems.append(
            f"values[{first_null}] is null: a missing observation is expressed by "
            f"omitting its instant and its value from both lists, never by null, "
            f"zero, filling or interpolation."
        )
    else:
        try:
            finite_values = np.asarray(
                validate_finite_samples(
                    list(values), source="values", remedy=_GAP_REMEDY
                ),
                dtype=float,
            )
        except InvalidInputError as exc:
            problems.append(str(exc))

    if problems:
        if len(problems) == 1:
            raise InvalidInputError(problems[0])
        numbered = " ".join(f"({i}) {p}" for i, p in enumerate(problems, start=1))
        raise InvalidInputError(f"The call has {len(problems)} problems: {numbered}")

    assert step_seconds is not None
    declaration = Declaration(
        indicator=indicator,
        unit=unit,
        resolution=resolution,
        step_seconds=step_seconds,
        aggregation=aggregation,
        horizon_steps=horizon_steps,
        backend=backend,
        season_length_steps=season,
    )
    return declaration, finite_values


def _parse_instants(observed_at: Sequence[Any], now: datetime) -> np.ndarray:
    instants = np.empty(len(observed_at), dtype=np.int64)
    for i, raw in enumerate(observed_at):
        try:
            iso_utc, declared, suspect = normalize_acquired_at(raw, now=now)
        except OverflowError:
            # Converting an instant at the edge of the datetime range to UTC
            # overflows; such an instant is outside the accepted range anyway.
            declared, suspect = True, True
        except ValueError:
            raise InvalidInputError(
                f"observed_at[{i}] = {raw!r} is not an ISO 8601 instant: write it "
                f"like '2026-10-24T08:00:00Z' or '2026-10-24T10:00:00+02:00'."
            ) from None
        if not declared:
            raise InvalidInputError(
                f"observed_at[{i}] = {raw!r} has no timezone: add 'Z' or the UTC "
                f"offset the source recorded; a timezone-naive instant is never "
                f"interpreted."
            )
        if suspect:
            raise InvalidInputError(
                f"observed_at[{i}] = {raw!r} lies before 1971, as an epoch or "
                f"spreadsheet zero date does, or more than one day after the server "
                f"clock ({format_instant(epoch_us(now))}): correct the timestamps "
                f"at the source."
            )
        instants[i] = epoch_us(datetime.fromisoformat(iso_utc))

    steps = np.diff(instants)
    if steps.size and np.all(steps < 0):
        raise InvalidInputError(
            f"observed_at runs newest first: observed_at[0] "
            f"({format_instant(int(instants[0]))}) is the latest instant. Reverse "
            f"both lists together, keeping each value with its instant, so the "
            f"instants increase."
        )
    bad = np.flatnonzero(steps <= 0)
    if bad.size:
        j = int(bad[0]) + 1
        raise InvalidInputError(
            f"observed_at[{j}] ({format_instant(int(instants[j]))}) is not after "
            f"observed_at[{j - 1}] ({format_instant(int(instants[j - 1]))}) once "
            f"both are converted to UTC: instants must strictly increase, so "
            f"remove the duplicate or correct the out-of-order row at the source "
            f"rather than sorting the series."
        )
    return instants


def _build_grid(
    instants_us: np.ndarray, values: np.ndarray, declaration: Declaration
) -> RegularGrid:
    step_us = declaration.step_seconds * _US_PER_S
    bins_of = instants_us // step_us
    start_bin = int(bins_of[0])
    n_bins = int(bins_of[-1]) - start_bin + 1
    if n_bins > MAX_GRID_BINS:
        gaps = np.diff(instants_us)
        k = int(np.argmax(gaps)) if gaps.size else 0
        span_s = (int(instants_us[-1]) - int(instants_us[0])) / _US_PER_S
        raise InvalidInputError(
            f"The series spans {format_duration(span_s)} at resolution "
            f"{declaration.resolution}, which needs {n_bins} bins, over the limit "
            f"of {MAX_GRID_BINS}. The largest gap is between observed_at[{k}] and "
            f"observed_at[{k + 1}] ({format_duration(int(gaps[k]) / _US_PER_S)}). "
            f"If that gap is not expected, check those instants for a typo; "
            f"otherwise use a coarser resolution or a shorter history."
        )

    index = (bins_of - start_bin).astype(np.int64)
    unique, starts, counts_per = np.unique(index, return_index=True, return_counts=True)
    agg = declaration.aggregation
    # An overflow becomes infinity here and is refused below.
    with np.errstate(over="ignore", invalid="ignore"):
        if agg == "mean":
            reduced = np.add.reduceat(values, starts) / counts_per
        elif agg == "max":
            reduced = np.maximum.reduceat(values, starts)
        elif agg == "last":
            reduced = values[starts + counts_per - 1]
        else:  # median
            # A single-point bin is its own median, so only shared bins need a call.
            reduced = values[starts].astype(float)
            for i in np.flatnonzero(counts_per > 1):
                a, c = starts[i], counts_per[i]
                reduced[i] = np.median(values[a : a + c])

    grid_values = np.full(n_bins, np.nan)
    grid_counts = np.zeros(n_bins, dtype=np.int64)
    grid_values[unique] = reduced
    grid_counts[unique] = counts_per
    if not np.all(np.isfinite(reduced)):
        raise InvalidInputError(
            "values are too large in magnitude to aggregate without overflow: "
            "check the unit and scale of the values."
        )
    return RegularGrid(
        start_us=start_bin * step_us,
        step_us=step_us,
        values=grid_values,
        counts=grid_counts,
        instants_us=instants_us,
        raw_values=values,
    )


def _resolution_problem(text: object) -> str:
    return (
        f"resolution {text!r} is not a grid step: write a positive whole number "
        f"followed by s, min, h, d or w, for example '10min' or '1h'."
    )
