"""One outcome decision for the RUL estimators and the estimate_rul tool.

The estimators (``estimate_rul_linear``, ``estimate_rul_exponential``,
``kalman_rul_from_state``) and estimate_rul decide where a model stands
against the threshold at the latest measurement through the same helpers, so
they cannot drift apart. These tests hold that decision to the rules written
out below, independently of the code, and check on seeded synthetic families
per method that the tool's status, rul and message agree with the estimator
given the same series.

The rules: a rising fitted curve reports its own time to the threshold, and
it has crossed when that time is zero or negative; a curve that does not rise
has crossed when its value at the latest measurement is at or above the
threshold. The Kalman filter has crossed when its filtered level is at or
above the threshold, and rises when its filtered rate is positive. Only a
model that rises and has not crossed gives a time to the threshold.
"""

import numpy as np
import pytest

from predictive_maintenance_mcp.mcp_tools import prognostics_tools
from predictive_maintenance_mcp.prognostics.kalman_rul import (
    kalman_degradation_state,
    kalman_rul_from_state,
    kalman_threshold_outcome,
)
from predictive_maintenance_mcp.prognostics.rul_estimator import (
    estimate_rul_exponential,
    estimate_rul_linear,
    fit_exponential_degradation,
    fit_linear_degradation,
    fit_threshold_outcome,
)

NOT_RISING_WORDING = {
    "exponential": "the fitted exponential curve does not rise",
    "kalman": "the Kalman filter's rate estimate at the latest measurement is not positive",
}
MODEL_STATUSES = {"estimated", "model_threshold_crossed", "model_not_rising"}


def _expected_fit(fit, threshold):
    time_to_crossing = fit["time_to_crossing"]
    if time_to_crossing is not None:
        return time_to_crossing <= 0, True
    return fit["fitted_last_value"] >= threshold, False


def _expected_state(state, threshold):
    return state["final_level"] >= threshold, state["final_rate"] > 0


def _families(seed: int):
    """Seeded series: noisy rises, rises with a low last reading, rises that turn
    down, and rises with near-zero readings; all regularly spaced so that the
    Kalman method accepts them."""
    rng = np.random.default_rng(seed)
    for k in range(240):
        n = int(rng.integers(5, 20))
        t = np.arange(n, dtype=float) * [1.0, 24.0, 0.5][k % 3]
        x = np.arange(n, dtype=float)
        kind = k % 4
        y = 2.0 + rng.uniform(0.1, 0.8) * x + rng.normal(0.0, rng.uniform(0.0, 0.4), n)
        if kind == 1:
            y[-1] -= rng.uniform(0.5, 3.0)
        elif kind == 2:
            peak = int(rng.integers(n // 2, n - 1))
            y[peak + 1 :] = y[peak] - rng.uniform(0.5, 2.5) * (x[peak + 1 :] - peak)
        elif kind == 3:
            y = np.abs(y) + 1e-3
            y[int(rng.integers(0, n))] = 10.0 ** rng.uniform(-8, -1)
        yield [float(v) for v in y], [float(s) for s in t]


def _model(method, values, timestamps, threshold):
    """The fit or filtered state the method decides on, and the stated outcome."""
    if method == "kalman":
        state = kalman_degradation_state(
            values, sampling_interval=timestamps[1] - timestamps[0]
        )
        return state, state["final_level"], _expected_state(state, threshold)
    fit_series = (
        fit_linear_degradation if method == "linear" else fit_exponential_degradation
    )
    fit = fit_series(values, timestamps, threshold)
    return fit, fit["fitted_last_value"], _expected_fit(fit, threshold)


def _thresholds(method, values, timestamps):
    """Thresholds above the latest reading, including the model's own value at the
    latest measurement, where 'at or above' and 'above' part."""
    last, top = values[-1], max(values)
    out = [last + 1e-9 * abs(last), last * 1.05, top * 1.5, top * 10.0]
    _, modelled, _ = _model(method, values, timestamps, 1.0)
    if np.isfinite(modelled) and modelled > last:
        out.append(float(modelled))
    return [x for x in out if x > last]


def _estimator(method, values, timestamps, threshold):
    if method == "linear":
        return estimate_rul_linear(values, timestamps, threshold)
    if method == "exponential":
        return estimate_rul_exponential(values, timestamps, threshold)
    state = kalman_degradation_state(
        values, sampling_interval=timestamps[1] - timestamps[0]
    )
    return kalman_rul_from_state(state, threshold)


def _run(method, values, timestamps, threshold):
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


@pytest.mark.parametrize("method", ["linear", "exponential", "kalman"])
def test_the_decision_follows_the_stated_rules(method):
    boundary = 0
    for values, timestamps in _families(seed=101):
        if method == "exponential" and min(values) <= 0:
            continue
        for threshold in _thresholds(method, values, timestamps):
            model, modelled, expected = _model(method, values, timestamps, threshold)
            if method == "kalman":
                decided = kalman_threshold_outcome(model, threshold)
            else:
                decided = fit_threshold_outcome(model, threshold)
            assert (decided.crossed, decided.rising) == expected
            assert decided.estimable == (expected[1] and not expected[0])
            boundary += modelled == threshold
    assert boundary >= 50


@pytest.mark.parametrize("method", ["linear", "exponential", "kalman"])
def test_the_tool_agrees_with_the_estimator(method):
    seen = set()
    for values, timestamps in _families(seed=202):
        if method == "exponential" and min(values) <= 0:
            continue
        for threshold in _thresholds(method, values, timestamps):
            result = _run(method, values, timestamps, threshold)
            if result.status not in MODEL_STATUSES:
                continue
            seen.add(result.status)
            _, modelled, (crossed, rising) = _model(
                method, values, timestamps, threshold
            )
            estimate = _estimator(method, values, timestamps, threshold)
            if result.status == "estimated":
                assert not crossed and rising
                assert estimate is not None
                assert result.rul == round(estimate["rul"], 4)
                assert f"{estimate['rul']:g} hours after the latest measurement" in (
                    result.message
                )
                continue
            assert estimate is None
            assert result.rul is None
            if result.status == "model_threshold_crossed":
                assert crossed
                assert result.modelled_current_value == modelled
                assert (
                    f"places the indicator at {modelled:g} at the time"
                    in result.message
                )
            else:
                assert not crossed and not rising
                assert result.modelled_current_value is None
                assert (
                    f"but {NOT_RISING_WORDING[method]}, so the model" in result.message
                )
    expected = {"estimated", "model_threshold_crossed"}
    if method != "linear":
        expected.add("model_not_rising")
    assert seen == expected


def test_a_kalman_rate_that_is_not_a_number_gives_no_estimate_in_either():
    """Values near 1e150 measured 1e-160 apart overflow the filter's rate."""
    values = [1e150 * v for v in [1.0, 2.0, 3.0, 4.0001]]
    timestamps = [i * 1e-160 for i in range(4)]
    state = kalman_degradation_state(values, sampling_interval=1e-160)
    assert np.isnan(state["final_rate"])
    assert kalman_rul_from_state(state, failure_threshold=1e152) is None
    result = _run("kalman", values, timestamps, threshold=1e152)
    assert result.status == "model_not_rising"
    assert result.rul is None


def test_a_curve_that_does_not_rise_has_crossed_when_its_value_reaches_the_threshold():
    """The crossing of a non-rising fit is decided on its value at the latest
    measurement, at or above the threshold, so equality counts as crossed."""
    fit = {"time_to_crossing": None, "fitted_last_value": 2.0}
    assert fit_threshold_outcome(fit, failure_threshold=2.0) == (True, False)
    assert fit_threshold_outcome(fit, failure_threshold=2.5) == (False, False)


def test_a_non_rising_exponential_curve_above_the_threshold_is_a_model_crossing():
    """A significant rise with a near-zero last reading: the log-linear curve
    slopes down, yet its value at the latest measurement lies above the
    threshold while the reading itself lies below it."""
    values = [
        1.0278680023715885,
        4.938185827332011,
        6.970596334160333,
        9.654587195498069,
        13.307108334003962,
        16.53846603523093,
        18.028267604713193,
        22.601303448640714,
        23.35426567205625,
        28.40047967376482,
        31.637277033448235,
        34.043518971824476,
        0.0023593189985263077,
    ]
    timestamps = [float(i) for i in range(len(values))]
    threshold = 1.9634899689411653
    fit = fit_exponential_degradation(values, timestamps, threshold)
    assert fit["time_to_crossing"] is None
    assert fit["fitted_last_value"] >= threshold > values[-1]
    assert estimate_rul_exponential(values, timestamps, threshold) is None
    result = _run("exponential", values, timestamps, threshold)
    assert result.status == "model_threshold_crossed"
    assert result.modelled_current_value == fit["fitted_last_value"]
    assert "the fitted exponential curve places the indicator at" in result.message
