"""
ISO 13374 Block 5 — Prognostic Assessment: statistical reference projections.

Two reference backends over a regular grid (NaN marks an empty bin), after
Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.),
section 5.5, table 5.2:

- **naive**: P50 is the last observed bin; the band is
  P50 +/- z(0.9) * sigma * sqrt(h), with sigma the uncentred RMS of the
  differences between grid-adjacent observed bins.
- **seasonal naive**: P50 at each step is the most recent observed bin at the
  same phase, ``k`` seasons back; the band uses sigma from same-phase pairs
  one season apart and widens with sqrt(k).

Uncentred sigma means the model assumes no drift. A drift only inflates sigma
by a constant factor while its offset grows linearly with the step, so under a
sustained trend actuals leave the band on the trend side more often as the
horizon grows.

When the history cannot support the band, a typed :class:`Insufficient` is
returned instead of a forecast. The horizon check runs last, so following its
remedy, a shorter horizon, cannot trip one of the earlier checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Optional, Union

import numpy as np

from ..signal_validation import InvalidInputError
from .indicator_series import lag_differences

QUANTILE_LEVELS = (0.1, 0.5, 0.9)
Z_P90 = NormalDist().inv_cdf(0.9)

#: Provisional technical minimum: it makes the band computable and does not
#: prove its calibration.
MIN_DIFFERENCE_PAIRS = 10

MIN_SEASONS_COVERED = 2

#: In the order the checks run.
REASON_CODES = (
    "too_few_difference_pairs",
    "season_coverage_too_short",
    "season_phase_unobserved",
    "no_variation",
    "horizon_exceeds_history",
)


@dataclass(frozen=True)
class Insufficient:
    """The history cannot support the requested projection."""

    reason_code: str
    backend: str
    available: int
    required: int
    lag: int


@dataclass(frozen=True)
class ReferenceForecast:
    """Quantiles per horizon step (index 0 is step 1)."""

    backend: str
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    seasons_back: Optional[np.ndarray]
    sigma: float
    difference_pairs: int
    lag: int


def reference_forecast(
    values: np.ndarray,
    *,
    backend: str,
    horizon_steps: int,
    season_length_steps: Optional[int],
    max_supported_horizon_steps: int,
) -> Union[ReferenceForecast, Insufficient]:
    """Project a regular grid with a reference backend.

    Args:
        values: The grid, oldest bin first; NaN marks an empty bin. The last
            bin is observed.
        backend: ``"naive"`` or ``"seasonal_naive"``.
        horizon_steps: Steps to project.
        season_length_steps: Season in grid steps (seasonal naive only).
        max_supported_horizon_steps: The most steps the observed history span
            and the caps allow.

    Raises:
        InvalidInputError: If the values are too large to difference.
    """
    if backend == "seasonal_naive":
        assert season_length_steps is not None
        return _seasonal_naive(
            values, horizon_steps, season_length_steps, max_supported_horizon_steps
        )
    return _naive(values, horizon_steps, max_supported_horizon_steps)


def _naive(
    values: np.ndarray, horizon: int, max_supported: int
) -> Union[ReferenceForecast, Insufficient]:
    diffs = lag_differences(values, 1)
    if diffs.size < MIN_DIFFERENCE_PAIRS:
        return Insufficient(
            reason_code="too_few_difference_pairs",
            backend="naive",
            available=diffs.size,
            required=MIN_DIFFERENCE_PAIRS,
            lag=1,
        )
    sigma = _uncentred_rms(diffs)
    if sigma == 0.0:
        return Insufficient(
            reason_code="no_variation", backend="naive", available=0, required=1, lag=1
        )
    if horizon > max_supported:
        return Insufficient(
            reason_code="horizon_exceeds_history",
            backend="naive",
            available=max_supported,
            required=horizon,
            lag=1,
        )

    observed = np.flatnonzero(~np.isnan(values))
    origin = float(values[observed[-1]])
    h = np.arange(1, horizon + 1, dtype=float)
    half = Z_P90 * sigma * np.sqrt(h)
    p50 = np.full(horizon, origin)
    return ReferenceForecast(
        backend="naive",
        p10=p50 - half,
        p50=p50,
        p90=p50 + half,
        seasons_back=None,
        sigma=sigma,
        difference_pairs=int(diffs.size),
        lag=1,
    )


def _seasonal_naive(
    values: np.ndarray, horizon: int, m: int, max_supported: int
) -> Union[ReferenceForecast, Insufficient]:
    diffs = lag_differences(values, m)
    if diffs.size < MIN_DIFFERENCE_PAIRS:
        return Insufficient(
            reason_code="too_few_difference_pairs",
            backend="seasonal_naive",
            available=diffs.size,
            required=MIN_DIFFERENCE_PAIRS,
            lag=m,
        )
    needed_bins = MIN_SEASONS_COVERED * m
    if values.size < needed_bins:
        return Insufficient(
            reason_code="season_coverage_too_short",
            backend="seasonal_naive",
            available=values.size,
            required=needed_bins,
            lag=m,
        )

    n = values.size
    sources = np.empty(horizon, dtype=np.int64)
    seasons_back = np.empty(horizon, dtype=np.int64)
    # Steps on the same phase start from the same bin, so each backward walk
    # over empty bins runs once per phase rather than once per step.
    source_of_start: dict[int, int] = {}
    missing = 0
    for step in range(1, horizon + 1):
        target = n + step - 1
        start = target - math.ceil(step / m) * m
        source = source_of_start.get(start)
        if source is None:
            source = start
            while source >= 0 and math.isnan(values[source]):
                source -= m
            source_of_start[start] = source
        if source < 0:
            missing += 1
            continue
        sources[step - 1] = source
        seasons_back[step - 1] = (target - source) // m
    if missing:
        return Insufficient(
            reason_code="season_phase_unobserved",
            backend="seasonal_naive",
            available=horizon - missing,
            required=horizon,
            lag=m,
        )

    sigma = _uncentred_rms(diffs)
    if sigma == 0.0:
        return Insufficient(
            reason_code="no_variation",
            backend="seasonal_naive",
            available=0,
            required=1,
            lag=m,
        )
    if horizon > max_supported:
        return Insufficient(
            reason_code="horizon_exceeds_history",
            backend="seasonal_naive",
            available=max_supported,
            required=horizon,
            lag=m,
        )

    p50 = values[sources].astype(float)
    half = Z_P90 * sigma * np.sqrt(seasons_back.astype(float))
    return ReferenceForecast(
        backend="seasonal_naive",
        p10=p50 - half,
        p50=p50,
        p90=p50 + half,
        seasons_back=seasons_back,
        sigma=sigma,
        difference_pairs=int(diffs.size),
        lag=m,
    )


def _uncentred_rms(diffs: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        sigma = float(np.sqrt(np.mean(diffs * diffs)))
    if not math.isfinite(sigma) or not np.all(np.isfinite(diffs)):
        raise InvalidInputError(
            "values are too large in magnitude to difference without overflow: "
            "check the unit and scale of the values."
        )
    return sigma
