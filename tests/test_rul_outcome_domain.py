"""The linear method of estimate_rul never reaches model_not_rising.

estimate_rul fits a model only after the slope test finds a statistically
significant increasing trend, and the linear method's fit is that same
regression: ``analyze_trend`` and ``fit_linear_degradation`` both run
``scipy.stats.linregress`` on the same values and timestamps. The test calls
a trend increasing only when the slope is positive, which is exactly when the
linear fit reports a crossing time. So every series that reaches the linear
fit gives a rising line, and the linear method carries no wording for a line
that does not rise. The exponential curve (fitted to the logarithm of the
values) and the Kalman filter's rate can still fail to rise after the test,
and keep their wording.

These tests cover the input domain the argument rests on, and pin what the
tool does if a linear fit that does not rise were ever forced on it: it
fails loudly instead of writing a message.
"""

import numpy as np
import pytest

from predictive_maintenance_mcp.mcp_tools import prognostics_tools
from predictive_maintenance_mcp.prognostics.rul_estimator import fit_linear_degradation
from predictive_maintenance_mcp.prognostics.trend_analyzer import analyze_trend


def _families(seed: int, count: int):
    """Seeded series over value scales, spacings, lengths and noise levels."""
    rng = np.random.default_rng(seed)
    for k in range(count):
        n = int(rng.integers(3, 40))
        scale = 10.0 ** rng.uniform(-8, 8)
        if k % 3 == 0:
            t = np.arange(n, dtype=float) * 10.0 ** rng.uniform(-3, 5)
        else:
            t = np.cumsum(rng.uniform(0.1, 10.0, n)) * 10.0 ** rng.uniform(-3, 5)
        slope = rng.normal(0.0, 1.0) * scale / t[-1]
        noise = rng.normal(0.0, scale * rng.uniform(0.0, 0.5), n)
        y = scale + slope * t + noise
        if k % 5 == 0:
            y[-1] -= abs(scale) * rng.uniform(0.1, 2.0)  # a last reading below the line
        yield [float(v) for v in y], [float(x) for x in t]


def _run(values, timestamps, threshold, method="linear"):
    """Call estimate_rul, which awaits nothing, without an event loop per call."""
    coroutine = prognostics_tools.estimate_rul(
        ctx=None,
        failure_threshold=threshold,
        timestamps=timestamps,
        feature_values=values,
        method=method,
    )
    with pytest.raises(StopIteration) as done:
        coroutine.send(None)
    return done.value.value


def test_a_series_that_passes_the_slope_test_gives_a_rising_line():
    passed = 0
    for values, timestamps in _families(seed=11, count=3000):
        trend = analyze_trend(
            values,
            timestamps=timestamps,
            significance_level=prognostics_tools.TREND_ALPHA,
        )
        if trend["trend_direction"] != "increasing":
            continue
        passed += 1
        fit = fit_linear_degradation(
            values, timestamps, max(values) + abs(max(values)) + 1
        )
        assert fit["slope"] == trend["slope"]
        assert fit["time_to_crossing"] is not None
    assert passed >= 500


def test_linear_estimate_rul_never_reports_a_line_that_does_not_rise():
    statuses = set()
    for values, timestamps in _families(seed=12, count=600):
        last, top = values[-1], max(values)
        for threshold in (
            last + 1e-9 * abs(last),
            last + 0.05 * abs(last),
            top + 3 * abs(top),
        ):
            if threshold <= last:
                continue
            statuses.add(_run(values, timestamps, threshold).status)
    assert "model_not_rising" not in statuses
    assert {"estimated", "model_threshold_crossed", "no_degradation_trend"} <= statuses


def test_a_forced_non_rising_linear_fit_fails_instead_of_writing_a_message(monkeypatch):
    values = [1.0 + 0.2 * i for i in range(12)]
    timestamps = [float(i) for i in range(12)]
    real_fit = prognostics_tools.fit_linear_degradation

    def non_rising_fit(series, times, threshold):
        fit = real_fit(series, times, threshold)
        return {**fit, "slope": 0.0, "crossing_time": None, "time_to_crossing": None}

    monkeypatch.setattr(prognostics_tools, "fit_linear_degradation", non_rising_fit)
    with pytest.raises(RuntimeError) as exc:
        _run(values, timestamps, threshold=6.0)
    assert str(exc.value) == (
        "estimate_rul has no wording for a linear model that does not rise: its fit "
        "is the slope test's own regression, so it rises whenever that test passes"
    )
