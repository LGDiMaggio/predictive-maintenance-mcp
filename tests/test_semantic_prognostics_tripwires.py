"""Known defects of the RUL, trend and forecast tools, held as strict expected
failures.

Every tripwire states the behaviour a correction must reach as a predicate,
and the predicate admits each outcome a later decision could choose: a
project-authored refusal, a distinct status, a structural field, or a
specific statement that the current output does not contain. A control next
to each tripwire applies the same predicate to a corrected stand-in, so a
fix is shown to turn the tripwire into an unexpected pass.

The bases are the tools' own contracts (field descriptions, docstrings), the
consistency of one series' units, and, for the forecast note, the method's
own nominal coverage measured with an exact binomial interval. No threshold
or classification is introduced here.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta

from _semantic_guards import (
    Outcome,
    capture,
    flat,
    no_foreign_error,
    refused_or_accepted,
    require,
)
from predictive_maintenance_mcp.mcp_tools.acquisition_tools import load_signal
from predictive_maintenance_mcp.mcp_tools.prognostics_tools import estimate_rul
from predictive_maintenance_mcp.models import RULEstimationResult
from predictive_maintenance_mcp.prognostics import indicator_forecast
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from test_diagnostic_invariants import known_defect

FS = 10_000.0


def _rul(**kwargs) -> Outcome:
    return asyncio.run(capture(estimate_rul(AsyncMock(), **kwargs)))


def _field_description(name: str) -> str:
    return RULEstimationResult.model_fields[name].description or ""


@pytest.fixture
def load_csv(sandbox_data_dir):
    """Write a one-column CSV into the sandbox and load it with a declared
    rate and unit; every id is cleared afterwards."""
    repo = get_repository()
    ids: list[str] = []

    async def _load(signal_id, values, unit):
        path = sandbox_data_dir / f"{signal_id}.csv"
        pd.DataFrame(np.asarray(values, dtype=float)).to_csv(
            path, index=False, header=False
        )
        ids.append(signal_id)
        return await load_signal(
            AsyncMock(),
            str(path),
            signal_id=signal_id,
            sampling_rate=FS,
            signal_unit=unit,
            overwrite=True,
        )

    yield _load
    for sid in ids:
        repo.clear_signal(sid)


def _tone(rms: float, freq_hz: float = 50.0, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(FS * seconds)) / FS
    return rms * np.sqrt(2.0) * np.sin(2.0 * np.pi * freq_hz * t)


# ---------------------------------------------------------------------------
# Mixed units on the signal_ids route
# ---------------------------------------------------------------------------

UNIT_MISMATCH_RE = re.compile(r"signal_unit|different units|mixed units", re.IGNORECASE)


def mixed_units_handled(outcome: Outcome) -> bool:
    """A series whose recordings declare different units is refused, or the
    result names the units it combined (a field or a statement)."""

    def accept(result) -> bool:
        data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        unit_fields = [k for k in data if "unit" in k and k != "time_unit"]
        return bool(unit_fields) or bool(
            UNIT_MISMATCH_RE.search(str(data.get("message", "")))
        )

    return refused_or_accepted(outcome, accept)


class TestSignalIdsRoute:
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B2-F5",
            "estimate_rul reduces signal_ids recordings that declare different "
            "units (mm/s and g) into one series without refusing or saying so",
        ),
    )
    def test_mixed_declared_units_are_refused_or_stated(self, load_csv):
        async def scenario():
            await load_csv("rc0_vel_0", _tone(1.0), "mm/s")
            await load_csv("rc0_vel_1", _tone(1.5), "mm/s")
            await load_csv("rc0_acc_2", _tone(0.064), "g")
            await load_csv("rc0_vel_3", _tone(2.5), "mm/s")
            return await capture(
                estimate_rul(
                    AsyncMock(),
                    failure_threshold=4.5,
                    timestamps=[0.0, 7.0, 14.0, 21.0],
                    signal_ids=["rc0_vel_0", "rc0_vel_1", "rc0_acc_2", "rc0_vel_3"],
                    time_unit="days",
                )
            )

        outcome = asyncio.run(scenario())
        no_foreign_error(outcome, "estimate_rul")
        assert mixed_units_handled(outcome)

    def test_control_a_unit_statement_satisfies_the_predicate(self):
        stand_in = SimpleNamespace(
            model_dump=lambda: {
                "status": "estimated",
                "message": "The recordings use different units.",
            }
        )
        assert mixed_units_handled(Outcome(result=stand_in))
        assert not mixed_units_handled(
            Outcome(result=SimpleNamespace(model_dump=lambda: {"message": "ok"}))
        )


# ---------------------------------------------------------------------------
# Falling indicator with a low threshold
# ---------------------------------------------------------------------------

RISING_RE = re.compile(r"\brises?\b|\brising\b", re.IGNORECASE)


def falling_series_handled(outcome: Outcome) -> bool:
    """A falling series whose latest value sits above a low threshold is
    refused, gets a status other than an observed exceedance, or its message
    names the rising-indicator assumption."""

    def accept(result) -> bool:
        return result.status != "threshold_already_exceeded" or bool(
            RISING_RE.search(result.message)
        )

    return refused_or_accepted(outcome, accept)


class TestFallingIndicator:
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B2-F9",
            "a falling indicator above a low failure threshold is reported as "
            "an observed exceedance with 'Inspect the machine now', without "
            "naming the rising-indicator assumption",
        ),
    )
    def test_falling_series_is_not_reported_as_an_exceedance_silently(self):
        outcome = _rul(
            failure_threshold=2.0,
            timestamps=[0, 1, 2, 3],
            feature_values=[10.0, 9.0, 8.0, 7.0],
        )
        no_foreign_error(outcome, "estimate_rul")
        assert falling_series_handled(outcome)

    def test_control_a_distinct_status_or_the_assumption_satisfies_it(self):
        named = SimpleNamespace(
            status="threshold_already_exceeded",
            message="estimate_rul assumes a rising indicator.",
        )
        assert falling_series_handled(Outcome(result=named))
        assert falling_series_handled(
            Outcome(result=SimpleNamespace(status="no_degradation_trend", message=""))
        )
        silent = SimpleNamespace(
            status="threshold_already_exceeded", message="Inspect the machine now."
        )
        assert not falling_series_handled(Outcome(result=silent))


# ---------------------------------------------------------------------------
# Output fields that contradict their own descriptions
# ---------------------------------------------------------------------------


def feature_label_honest(result, supplied: bool) -> bool:
    return supplied or result.feature_name != "rms"


def kalman_r_squared_consistent(result, description: str) -> bool:
    says_none = "None for the kalman method" in flat(description)
    return not says_none or result.fit_r_squared is None


def _r_squared(y: np.ndarray, fitted: np.ndarray) -> float:
    return float(1.0 - np.sum((y - fitted) ** 2) / np.sum((y - np.mean(y)) ** 2))


def exponential_r_squared_consistent(result, t, y, description: str) -> bool:
    """When the description says the R-squared is on the observed data, the
    reported value is the R-squared of the fitted curve on the observed values."""
    if "on the observed data" not in flat(description) or result.fit_r_squared is None:
        return True
    slope, intercept = np.polyfit(t, np.log(y), 1)
    observed = _r_squared(y, np.exp(intercept + slope * t))
    return abs(result.fit_r_squared - observed) < 5e-4


EXP_T = np.arange(6, dtype=float)
EXP_Y = np.array([1.0, 1.25, 1.7, 2.1, 3.2, 3.6])


class TestFieldContracts:
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B2-F15-label",
            "estimate_rul echoes feature_name 'rms' for feature_values it did "
            "not compute and whose feature the caller never named",
        ),
    )
    def test_feature_values_are_not_labelled_rms_by_default(self):
        outcome = _rul(
            failure_threshold=90.0,
            timestamps=[0, 1, 2, 3],
            feature_values=[60.0, 65.0, 70.0, 75.0],
        )
        no_foreign_error(outcome, "estimate_rul")
        assert refused_or_accepted(
            outcome, lambda result: feature_label_honest(result, supplied=False)
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B2-F15-kalman-r2",
            "a kalman no_degradation_trend result reports fit_r_squared while "
            "the field description says it is None for the kalman method",
        ),
    )
    def test_kalman_result_honours_the_r_squared_description(self):
        outcome = _rul(
            failure_threshold=4.5,
            timestamps=[0, 1, 2, 3, 4, 5],
            feature_values=[2.0, 2.1, 1.9, 2.05, 1.95, 2.0],
            method="kalman",
        )
        no_foreign_error(outcome, "estimate_rul")
        description = _field_description("fit_r_squared")
        assert refused_or_accepted(
            outcome, lambda result: kalman_r_squared_consistent(result, description)
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B2-F10-r2",
            "the exponential method's fit_r_squared is computed on the "
            "logarithm of the values while its description says it is on "
            "the observed data",
        ),
    )
    def test_exponential_r_squared_matches_its_description(self):
        outcome = _rul(
            failure_threshold=10.0,
            timestamps=list(EXP_T),
            feature_values=list(EXP_Y),
            method="exponential",
        )
        require(
            outcome.error is None and outcome.result.status == "estimated",
            f"the exponential fit of this series is no longer estimated: {outcome}",
        )
        assert exponential_r_squared_consistent(
            outcome.result, EXP_T, EXP_Y, _field_description("fit_r_squared")
        )

    def test_controls_for_the_field_predicates(self):
        assert feature_label_honest(SimpleNamespace(feature_name=None), supplied=False)
        assert not feature_label_honest(
            SimpleNamespace(feature_name="rms"), supplied=False
        )
        assert kalman_r_squared_consistent(
            SimpleNamespace(fit_r_squared=None), "None for the kalman method."
        )
        assert kalman_r_squared_consistent(
            SimpleNamespace(fit_r_squared=0.4), "Goodness of fit."
        )
        assert not kalman_r_squared_consistent(
            SimpleNamespace(fit_r_squared=0.4), "None for the kalman method."
        )
        slope, intercept = np.polyfit(EXP_T, np.log(EXP_Y), 1)
        observed = _r_squared(EXP_Y, np.exp(intercept + slope * EXP_T))
        description = "R-squared of the fitted degradation curve on the observed data."
        assert exponential_r_squared_consistent(
            SimpleNamespace(fit_r_squared=observed), EXP_T, EXP_Y, description
        )
        assert exponential_r_squared_consistent(
            SimpleNamespace(fit_r_squared=0.1), EXP_T, EXP_Y, "On log values."
        )
        assert not exponential_r_squared_consistent(
            SimpleNamespace(fit_r_squared=observed - 0.05), EXP_T, EXP_Y, description
        )


# ---------------------------------------------------------------------------
# Forecast interval note versus measured coverage
# ---------------------------------------------------------------------------

#: The sentence of the interval note under test, as approved and shipped.
OVER_WIDE_SENTENCE = "On an indicator that fluctuates around a level, the band is wider than the data need."
NOMINAL_INSIDE = 0.80
REPLICATES = 400
HORIZON = 12
NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


def _hourly_instants(n: int) -> list[str]:
    end = NOW - timedelta(hours=1)
    start = end - (n - 1) * timedelta(hours=1)
    return [
        (start + i * timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        for i in range(n)
    ]


def upper_clopper_pearson(inside: int, n: int, alpha: float = 0.05) -> float:
    return (
        1.0 if inside >= n else float(beta.ppf(1 - alpha / 2, inside + 1, n - inside))
    )


def note_consistent_with_coverage(note: str, inside: int, n: int) -> bool:
    """The note may keep the over-wide sentence only if the band does not
    under-cover on an indicator that fluctuates around a level."""
    return (
        flat(OVER_WIDE_SENTENCE) not in flat(note)
        or upper_clopper_pearson(inside, n) >= NOMINAL_INSIDE
    )


def daily_cycle_inside_count() -> int:
    """Naive-backend coverage at a 12-hour horizon on a daily sine around a
    level (amplitude 3, noise sd 0.5, 200 hourly bins, origin phase drawn per
    replicate, seeded)."""
    observed_at = _hourly_instants(200)
    inside = 0
    for replicate in range(REPLICATES):
        rng = np.random.default_rng(3_002_000 + replicate)
        phase = int(rng.integers(0, 24))
        t = np.arange(200 + HORIZON)
        y = (
            10.0
            + 3.0 * np.sin(2.0 * np.pi * (t + phase) / 24.0)
            + rng.normal(0.0, 0.5, t.size)
        )
        result = indicator_forecast.forecast_indicator(
            observed_at=observed_at,
            values=[float(v) for v in y[:200]],
            now=NOW,
            server_version="semantic-test",
            indicator="velocity RMS",
            unit="mm/s",
            resolution="1h",
            horizon_steps=HORIZON,
            backend="naive",
            aggregation="mean",
            season_length_steps=None,
        )
        actual = y[200 + HORIZON - 1]
        inside += int(
            result["p10"][HORIZON - 1] <= actual <= result["p90"][HORIZON - 1]
        )
    return inside


class TestForecastIntervalNote:
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B3-F1",
            "the forecast interval note says the band is wider than needed on "
            "an indicator that fluctuates around a level, while the naive band "
            "covers well below its nominal 80 % on a daily cycle at a 12-hour "
            "horizon",
        ),
    )
    def test_interval_note_agrees_with_coverage_on_a_cyclic_indicator(self):
        inside = daily_cycle_inside_count()
        assert note_consistent_with_coverage(
            indicator_forecast.INTERVAL_NOTE, inside, REPLICATES
        ), (
            f"inside P10..P90: {inside}/{REPLICATES}, upper 95 % bound "
            f"{upper_clopper_pearson(inside, REPLICATES):.3f} < {NOMINAL_INSIDE}"
        )

    def test_control_the_predicate_accepts_either_correction(self):
        under = 186
        assert not note_consistent_with_coverage(OVER_WIDE_SENTENCE, under, REPLICATES)
        assert note_consistent_with_coverage("A two-sided note.", under, REPLICATES)
        assert note_consistent_with_coverage(OVER_WIDE_SENTENCE, 320, REPLICATES)
