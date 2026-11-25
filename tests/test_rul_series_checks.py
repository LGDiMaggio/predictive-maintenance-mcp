"""estimate_rul refuses series it cannot treat as one indicator rising toward
its threshold, and says what an accepted series rests on.

Refused, with the server's own text:

- a series whose first measurement is already at or above failure_threshold;
- a signal_ids series whose recordings declare different units, assets,
  measurement points or measurement directions, or whose declared
  acquisition times do not follow the order of the timestamps.

Accepted series report extrapolation_ratio (status 'estimated' only), the
recordings behind the values (signal_ids route only) and no feature_name for
values the caller measured. The fit_r_squared description is checked against
what each status and method actually returns.

The expected texts below are this module's own copy of the approved wording,
not read from the code under test.
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from predictive_maintenance_mcp.mcp_tools import prognostics_tools
from predictive_maintenance_mcp.models import RULEstimationResult
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

REPO_ROOT = Path(__file__).resolve().parents[1]
FS = 2000.0
DAYS = [0.0, 7.0, 14.0, 21.0]

FIRST_AT_OR_ABOVE = (
    "estimate_rul assumes an indicator that rises toward failure_threshold, "
    "but the first measurement ({first}) is already at or above the threshold "
    "supplied in this call (failure_threshold = {thr}), so the series is "
    "refused and no time to the threshold is estimated. {latest} "
    "This can mean that the indicator falls as the condition worsens, which "
    "estimate_rul does not support, that the threshold does not suit this "
    "indicator or its unit, or that the indicator was already at or above the "
    "limit when the series began. The threshold is an indicator limit, not an "
    "observed component failure. Ask the user which way the indicator moves as "
    "the condition worsens and whether the threshold is given in the "
    "indicator's unit; do not negate, shift or reorder the values to get past "
    "this check."
)
LATEST_AT_OR_ABOVE = (
    "The latest measurement ({last}) is also at or above the threshold."
)
LATEST_BELOW = "The latest measurement ({last}) is below the threshold."
UNITS_DIFFER = (
    "The recordings in signal_ids declare different units: '{a}' (index {i}) "
    "declares {ua}, but '{b}' (index {j}) declares {ub}. estimate_rul converts "
    "no units, so their {feature} values cannot form one series. Ask the "
    "user which unit each recording was taken in: pass recordings declared in "
    "one unit, or reload a recording whose unit was declared wrongly with its "
    "correct signal_unit. Do not convert, drop or reorder recordings on your "
    "own."
)
ASSETS_DIFFER = (
    "The recordings in signal_ids are declared for different assets: '{a}' "
    "(index {i}) for asset '{A}', but '{b}' (index {j}) for asset '{B}'. A "
    "series for estimate_rul must hold repeated measurements of one machine. "
    "Ask the user which recordings belong to the series and pass only those; "
    "do not drop or reorder recordings on your own."
)
POINTS_DIFFER = (
    "The recordings in signal_ids are declared for different measurement "
    "points: '{a}' (index {i}) for point '{P}', but '{b}' (index {j}) for point "
    "'{Q}'. A series for estimate_rul must hold repeated measurements at one "
    "measurement point. Ask the user which recordings belong to the series and "
    "pass only those; do not drop or reorder recordings on your own."
)
DIRECTIONS_DIFFER = (
    "The recordings in signal_ids are declared with different measurement "
    "directions: '{a}' (index {i}) declares {da}, but '{b}' (index {j}) "
    "declares {db}. Another direction is another measurement point, so their "
    "values cannot form one series. Ask the user which recordings belong to "
    "the series and pass only those; do not drop or reorder recordings on your "
    "own."
)
ORDER_DISAGREES = (
    "The declared acquisition times of the recordings in signal_ids do not "
    "follow the order of timestamps: '{b}' (index {j}) was acquired at {tb}, "
    "which is not later than '{a}' (index {i}), acquired at {ta}, while "
    "timestamps place it later. Ask the user which is right, the timestamps or "
    "the acquisition times the recordings declare; do not reorder the series "
    "or change either on your own."
)


def _run(**kwargs):
    """Call estimate_rul, which awaits nothing, and return its result."""
    coroutine = prognostics_tools.estimate_rul(ctx=None, **kwargs)
    with pytest.raises(StopIteration) as done:
        coroutine.send(None)
    return done.value.value


def _refusal(**kwargs) -> str:
    with pytest.raises(ValueError) as exc:
        _run(**kwargs)
    return str(exc.value)


def _first_refusal(values, threshold, last_shown=None):
    last = values[-1]
    shown = last_shown if last_shown is not None else f"{last:g}"
    latest = (LATEST_AT_OR_ABOVE if last >= threshold else LATEST_BELOW).format(
        last=shown
    )
    return FIRST_AT_OR_ABOVE.format(
        first=f"{values[0]:g}", thr=f"{threshold:g}", latest=latest
    )


def _tone(rms: float) -> np.ndarray:
    t = np.arange(int(FS)) / FS
    return rms * math.sqrt(2.0) * np.sin(2.0 * math.pi * 50.0 * t)


def _measurement(asset, point, day, direction=None):
    declared = {
        "asset_id": asset,
        "measurement_point_id": point,
        "acquired_at": f"2026-10-{day:02d}T08:00:00+00:00",
    }
    if direction is not None:
        declared["direction"] = direction
    return declared


@pytest.fixture
def recordings(tmp_path):
    """Load one synthetic recording per (rms, unit, measurement) entry; return
    the ids, the stored arrays and the repository. Every id is cleared after."""
    repo = get_repository()
    loaded: list[str] = []

    def load(name: str, entries):
        ids, arrays = [], []
        for i, (rms, unit, declared) in enumerate(entries):
            data = _tone(rms)
            path = tmp_path / f"{name}_{i}.csv"
            np.savetxt(path, data, delimiter=",")
            companion = {"sampling_rate": FS}
            if unit is not None:
                companion["signal_unit"] = unit
            if declared is not None:
                companion["measurement"] = declared
            (tmp_path / f"{name}_{i}_metadata.json").write_text(json.dumps(companion))
            sid = f"rulchk_{name}_{i}"
            repo.load_signal(str(path), signal_id=sid, overwrite=True)
            loaded.append(sid)
            ids.append(sid)
            arrays.append(np.asarray(repo.get_signal(sid), dtype=float))
        return ids, arrays

    yield load, repo
    for sid in loaded:
        repo.clear_signal(sid)


def _rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(array**2)))


# ---------------------------------------------------------------------------
# First measurement at or above the threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["linear", "exponential", "kalman"])
@pytest.mark.parametrize(
    "values, threshold",
    [
        pytest.param([10.0, 9.0, 8.0, 7.0], 2.0, id="falling_above_a_low_threshold"),
        pytest.param([5.0, 6.0, 7.0], 4.0, id="rising_entirely_above"),
        pytest.param([10.0, 5.0, 1.0], 4.0, id="starts_above_ends_below"),
        pytest.param([4.5, 3.0, 2.0], 4.5, id="first_equal_to_the_threshold"),
        pytest.param([1.0, 1.2, 1.5], -1.0, id="negative_threshold"),
    ],
)
def test_a_series_starting_at_or_above_the_threshold_is_refused(
    values, threshold, method
):
    message = _refusal(
        failure_threshold=threshold,
        timestamps=[float(i) for i in range(len(values))],
        feature_values=values,
        method=method,
    )
    assert message == _first_refusal(values, threshold)
    assert "Inspect the machine now" not in message


def test_a_latest_value_that_prints_like_the_threshold_is_shown_exactly():
    values = [4.6, 4.4, 4.4999999]
    message = _refusal(
        failure_threshold=4.5, timestamps=[0.0, 1.0, 2.0], feature_values=values
    )
    assert message == _first_refusal(values, 4.5, last_shown="4.4999999")


def test_a_series_ending_below_the_threshold_states_no_exceedance():
    message = _refusal(
        failure_threshold=4.0,
        timestamps=[0.0, 1.0, 2.0],
        feature_values=[10.0, 5.0, 1.0],
    )
    assert "The latest measurement (1) is below the threshold." in message
    assert "also at or above" not in message


def test_a_first_value_just_below_the_threshold_keeps_its_outcome():
    threshold = math.nextafter(4.0, math.inf)
    result = _run(
        failure_threshold=threshold,
        timestamps=[0.0, 1.0, 2.0],
        feature_values=[4.0, 4.5, 5.0],
    )
    assert result.status == "threshold_already_exceeded"


def test_the_refusal_comes_before_the_exponential_positivity_rule():
    values = [5.0, -1.0, 3.0]
    message = _refusal(
        failure_threshold=4.0,
        timestamps=[0.0, 1.0, 2.0],
        feature_values=values,
        method="exponential",
    )
    assert message == _first_refusal(values, 4.0)


def test_the_refusal_applies_to_stored_recordings_after_their_checks(recordings):
    load, _ = recordings
    ids, arrays = load(
        "falling",
        [
            (rms, "mm/s", _measurement("P-101", "de_h", d))
            for rms, d in zip([4.0, 3.5, 3.0, 2.5], [1, 8, 15, 22])
        ],
    )
    values = [_rms(a) for a in arrays]
    message = _refusal(failure_threshold=3.2, timestamps=DAYS, signal_ids=ids)
    assert message == _first_refusal(values, 3.2)


# ---------------------------------------------------------------------------
# Recordings of one signal_ids series checked against each other
# ---------------------------------------------------------------------------


def test_different_declared_units_are_refused_before_any_recording_is_reduced(
    recordings, monkeypatch
):
    load, _ = recordings
    ids, _ = load(
        "units",
        [
            (1.0, "mm/s", None),
            (1.5, "mm/s", None),
            (0.064, "g", None),
            (2.5, "mm/s", None),
        ],
    )
    calls = []
    real = prognostics_tools.extract_time_domain_features

    def recording(array):
        calls.append(len(array))
        return real(array)

    monkeypatch.setattr(prognostics_tools, "extract_time_domain_features", recording)
    message = _refusal(failure_threshold=4.5, timestamps=DAYS, signal_ids=ids)
    assert message == UNITS_DIFFER.format(
        a=ids[0], i=0, ua="mm/s", b=ids[2], j=2, ub="g", feature="rms"
    )
    assert calls == []


def test_the_unit_check_comes_before_the_first_measurement_refusal(recordings):
    load, _ = recordings
    ids, _ = load(
        "units_first", [(5.0, "mm/s", None), (4.0, "g", None), (3.0, "mm/s", None)]
    )
    message = _refusal(
        failure_threshold=2.0, timestamps=[0.0, 7.0, 14.0], signal_ids=ids
    )
    assert message == UNITS_DIFFER.format(
        a=ids[0], i=0, ua="mm/s", b=ids[1], j=1, ub="g", feature="rms"
    )


def test_a_recording_that_declares_no_unit_is_not_compared(recordings):
    load, _ = recordings
    ids, _ = load(
        "unit_gap",
        [
            (1.0, "mm/s", None),
            (1.5, None, None),
            (2.0, "mm/s", None),
            (2.5, "mm/s", None),
        ],
    )
    result = _run(failure_threshold=4.5, timestamps=DAYS, signal_ids=ids)
    assert [m.signal_unit for m in result.measurements] == [
        "mm/s",
        None,
        "mm/s",
        "mm/s",
    ]


@pytest.mark.parametrize(
    "units, a, b",
    [
        pytest.param(["mm/s", None, "g"], 0, 2, id="undeclared_between"),
        pytest.param([None, "mm/s", "g"], 1, 2, id="undeclared_first"),
    ],
)
def test_units_are_compared_across_an_undeclared_recording(recordings, units, a, b):
    load, _ = recordings
    ids, _ = load(
        f"unit_skip_{a}", [(1.0 + 0.5 * k, u, None) for k, u in enumerate(units)]
    )
    message = _refusal(failure_threshold=4.5, timestamps=DAYS[:3], signal_ids=ids)
    assert message == UNITS_DIFFER.format(
        a=ids[a], i=a, ua=units[a], b=ids[b], j=b, ub=units[b], feature="rms"
    )


def test_a_leading_undeclared_recording_is_not_a_unit_of_its_own(recordings):
    load, _ = recordings
    ids, _ = load(
        "unit_skip_ok", [(1.0, None, None), (1.5, "mm/s", None), (2.0, "mm/s", None)]
    )
    result = _run(failure_threshold=4.5, timestamps=DAYS[:3], signal_ids=ids)
    assert [m.signal_unit for m in result.measurements] == [None, "mm/s", "mm/s"]


def test_different_declared_assets_are_refused(recordings):
    load, _ = recordings
    ids, _ = load(
        "assets",
        [
            (1.0, "mm/s", _measurement("P-101", "de_h", 1)),
            (1.5, "mm/s", _measurement("P-202", "de_h", 8)),
            (2.0, "mm/s", _measurement("P-101", "de_h", 15)),
        ],
    )
    message = _refusal(failure_threshold=4.5, timestamps=DAYS[:3], signal_ids=ids)
    assert message == ASSETS_DIFFER.format(
        a=ids[0], i=0, A="P-101", b=ids[1], j=1, B="P-202"
    )


def test_different_declared_points_are_refused(recordings):
    load, _ = recordings
    ids, _ = load(
        "points",
        [
            (1.0, "mm/s", _measurement("P-101", "de_h", 1)),
            (1.5, "mm/s", _measurement("P-101", "de_h", 8)),
            (2.0, "mm/s", _measurement("P-101", "nde_h", 15)),
        ],
    )
    message = _refusal(failure_threshold=4.5, timestamps=DAYS[:3], signal_ids=ids)
    assert message == POINTS_DIFFER.format(
        a=ids[0], i=0, P="de_h", b=ids[2], j=2, Q="nde_h"
    )


def test_different_declared_directions_at_one_point_are_refused(recordings):
    load, _ = recordings
    ids, _ = load(
        "directions",
        [
            (1.0, "mm/s", _measurement("P-101", "de", 1, "x")),
            (1.5, "mm/s", _measurement("P-101", "de", 8, "y")),
            (2.0, "mm/s", _measurement("P-101", "de", 15, "x")),
        ],
    )
    message = _refusal(failure_threshold=4.5, timestamps=DAYS[:3], signal_ids=ids)
    assert message == DIRECTIONS_DIFFER.format(
        a=ids[0], i=0, da="x", b=ids[1], j=1, db="y"
    )


@pytest.mark.parametrize(
    "days", [[15, 8, 22], [1, 8, 8]], ids=["reversed", "equal_instants"]
)
def test_an_acquisition_order_that_disagrees_with_the_timestamps_is_refused(
    recordings, days
):
    load, repo = recordings
    ids, _ = load(
        f"order_{days[1]}_{days[2]}",
        [
            (rms, "mm/s", _measurement("P-101", "de_h", d))
            for rms, d in zip([1.0, 1.5, 2.0], days)
        ],
    )
    instants = [repo.get_signal_info(s)["measurement"]["acquired_at"] for s in ids]
    message = _refusal(failure_threshold=4.5, timestamps=DAYS[:3], signal_ids=ids)
    bad = 1 if days[1] <= days[0] else 2
    assert message == ORDER_DISAGREES.format(
        a=ids[bad - 1],
        i=bad - 1,
        ta=instants[bad - 1],
        b=ids[bad],
        j=bad,
        tb=instants[bad],
    )


def test_a_recording_without_a_measurement_block_is_not_compared(recordings):
    load, _ = recordings
    ids, _ = load(
        "measurement_gap",
        [
            (1.0, "mm/s", _measurement("P-101", "de_h", 1)),
            (1.5, "mm/s", None),
            (2.0, "mm/s", _measurement("P-101", "de_h", 15)),
            (2.5, "mm/s", _measurement("P-101", "de_h", 22)),
        ],
    )
    assert (
        _run(failure_threshold=4.5, timestamps=DAYS, signal_ids=ids).status
        == "estimated"
    )


def test_acquisition_order_is_compared_across_an_undeclared_recording(recordings):
    load, repo = recordings
    ids, _ = load(
        "order_skip",
        [
            (1.0, "mm/s", _measurement("P-101", "de_h", 15)),
            (1.5, "mm/s", None),
            (2.0, "mm/s", _measurement("P-101", "de_h", 8)),
        ],
    )
    instants = [
        repo.get_signal_info(s)["measurement"]["acquired_at"] for s in (ids[0], ids[2])
    ]
    message = _refusal(failure_threshold=4.5, timestamps=DAYS[:3], signal_ids=ids)
    assert message == ORDER_DISAGREES.format(
        a=ids[0], i=0, ta=instants[0], b=ids[2], j=2, tb=instants[1]
    )


def test_a_recording_evicted_after_its_declarations_are_read_is_refused(
    recordings, monkeypatch
):
    load, repo = recordings
    ids, _ = load("evicted", [(rms, "mm/s", None) for rms in [1.0, 1.5, 2.0]])
    real = repo.get_signal
    calls = []

    def evicted(sid):
        calls.append(sid)
        if sid == ids[1]:
            raise KeyError(f"Signal '{sid}' is no longer stored")
        return real(sid)

    monkeypatch.setattr(repo, "get_signal", evicted)
    message = _refusal(failure_threshold=4.5, timestamps=DAYS[:3], signal_ids=ids)
    assert message == f"Signal '{ids[1]}' is no longer stored"
    assert calls == ids[:2]


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------

LINE = [1.0 + 0.2 * i for i in range(12)]
CROSSED = [6.0, 7.1, 8.0, 9.2, 10.4, 9.7]
KALMAN_TURNS_DOWN = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.0, 4.0]


def test_extrapolation_ratio_is_the_rul_over_the_observed_span():
    result = _run(
        failure_threshold=6.0,
        timestamps=[float(i) for i in range(12)],
        feature_values=LINE,
    )
    assert result.status == "estimated"
    assert result.extrapolation_ratio == float(f"{14.0 / 11.0:.6g}")


@pytest.mark.parametrize(
    "values, threshold, method",
    [
        ([0.5 * 1.2**i for i in range(10)], 20.0, "exponential"),
        (LINE, 6.0, "kalman"),
    ],
)
def test_extrapolation_ratio_is_the_rul_over_the_span_for_every_method(
    values, threshold, method
):
    timestamps = [float(i) for i in range(len(values))]
    result = _run(
        failure_threshold=threshold,
        timestamps=timestamps,
        feature_values=values,
        method=method,
    )
    assert result.status == "estimated"
    span = timestamps[-1] - timestamps[0]
    assert result.extrapolation_ratio == pytest.approx(result.rul / span, rel=1e-4)


@pytest.mark.parametrize(
    "values, threshold, method, status",
    [
        ([2.0] * 6, 4.5, "linear", "no_degradation_trend"),
        ([3.0, 4.0, 5.0], 4.5, "linear", "threshold_already_exceeded"),
        (CROSSED, 10.0, "linear", "model_threshold_crossed"),
        (KALMAN_TURNS_DOWN, 20.0, "kalman", "model_not_rising"),
    ],
)
def test_extrapolation_ratio_is_null_without_an_estimate(
    values, threshold, method, status
):
    result = _run(
        failure_threshold=threshold,
        timestamps=[float(i) for i in range(len(values))],
        feature_values=values,
        method=method,
    )
    assert result.status == status
    assert result.extrapolation_ratio is None


@pytest.mark.parametrize(
    "values, threshold, method",
    [(LINE, 6.0, "linear"), ([0.5 * 1.2**i for i in range(10)], 20.0, "exponential")],
)
def test_extrapolation_ratio_does_not_depend_on_the_time_unit_or_the_scale(
    values, threshold, method
):
    def ratio(step, scale):
        return _run(
            failure_threshold=threshold * scale,
            timestamps=[step * i for i in range(len(values))],
            feature_values=[scale * v for v in values],
            method=method,
        ).extrapolation_ratio

    days = ratio(1.0, 1.0)
    assert ratio(24.0, 1.0) == pytest.approx(days, rel=1e-5)
    assert ratio(1.0, 1000.0) == pytest.approx(days, rel=1e-5)


def test_an_estimate_far_beyond_the_span_is_not_capped():
    timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
    result = _run(
        failure_threshold=4.5,
        timestamps=timestamps,
        feature_values=[1.0 + 1e-6 * t for t in timestamps],
        time_unit="hours",
    )
    assert result.status == "estimated"
    assert result.extrapolation_ratio == pytest.approx(874999.0, rel=1e-5)
    assert "Caution" in result.message


def test_stored_recordings_are_echoed_with_their_values_and_units(recordings):
    load, _ = recordings
    ids, arrays = load(
        "echo",
        [
            (rms, "mm/s", _measurement("P-101", "de_h", d))
            for rms, d in zip([1.0, 1.5, 2.0, 2.5], [1, 8, 15, 22])
        ],
    )
    result = _run(failure_threshold=4.5, timestamps=DAYS, signal_ids=ids)
    assert [m.signal_id for m in result.measurements] == ids
    assert [m.value for m in result.measurements] == [round(_rms(a), 6) for a in arrays]
    assert [m.signal_unit for m in result.measurements] == ["mm/s"] * 4
    assert result.measurements[-1].value == result.current_value
    assert result.feature_name == "rms"


@pytest.mark.parametrize(
    "rms_values, threshold, method, status",
    [
        ([2.0] * 6, 4.5, "linear", "no_degradation_trend"),
        ([3.0, 4.0, 5.0], 4.5, "linear", "threshold_already_exceeded"),
        (CROSSED, 10.0, "linear", "model_threshold_crossed"),
        (KALMAN_TURNS_DOWN, 20.0, "kalman", "model_not_rising"),
    ],
)
def test_stored_recordings_are_echoed_at_every_status(
    recordings, rms_values, threshold, method, status
):
    load, _ = recordings
    ids, arrays = load(f"echo_{status}", [(rms, "mm/s", None) for rms in rms_values])
    result = _run(
        failure_threshold=threshold,
        timestamps=[float(i) for i in range(len(ids))],
        signal_ids=ids,
        method=method,
    )
    assert result.status == status
    assert [m.signal_id for m in result.measurements] == ids
    assert [m.value for m in result.measurements] == [round(_rms(a), 6) for a in arrays]
    assert [m.signal_unit for m in result.measurements] == ["mm/s"] * len(ids)


@pytest.mark.parametrize(
    "values, threshold",
    [(LINE, 6.0), ([2.0] * 6, 4.5), ([3.0, 4.0, 5.0], 4.5), (CROSSED, 10.0)],
)
def test_caller_values_carry_no_feature_name_and_no_echo(values, threshold):
    result = _run(
        failure_threshold=threshold,
        timestamps=[float(i) for i in range(len(values))],
        feature_values=values,
        feature_name="rms",
    )
    assert result.feature_name is None
    assert result.measurements is None


def test_a_requested_feature_is_named_on_the_stored_route(recordings):
    load, _ = recordings
    ids, _ = load("kurtosis", [(rms, "mm/s", None) for rms in [1.0, 1.5, 2.0, 2.5]])
    result = _run(
        failure_threshold=50.0, timestamps=DAYS, signal_ids=ids, feature_name="kurtosis"
    )
    assert result.feature_name == "kurtosis"


def _slope_r_squared(values, timestamps):
    return round(float(stats.linregress(timestamps, values).rvalue ** 2), 4)


def test_fit_r_squared_follows_its_description_for_each_status_and_method():
    t12 = [float(i) for i in range(12)]
    kalman_flat = [2.0, 2.1, 1.9, 2.05, 1.95, 2.0]
    t6 = [float(i) for i in range(6)]
    no_trend = _run(
        failure_threshold=4.5,
        timestamps=t6,
        feature_values=kalman_flat,
        method="kalman",
    )
    assert no_trend.status == "no_degradation_trend"
    assert no_trend.fit_r_squared == _slope_r_squared(kalman_flat, t6)

    kalman = _run(
        failure_threshold=6.0, timestamps=t12, feature_values=LINE, method="kalman"
    )
    assert kalman.status == "estimated" and kalman.fit_r_squared is None

    growth = [1.0, 1.25, 1.7, 2.1, 3.2, 3.6]
    exponential = _run(
        failure_threshold=10.0,
        timestamps=t6,
        feature_values=growth,
        method="exponential",
    )
    assert exponential.status == "estimated"
    assert exponential.fit_r_squared == _slope_r_squared(np.log(growth), t6)

    noisy_line = [1.0, 1.3, 1.35, 1.7, 1.75, 2.1]
    linear = _run(failure_threshold=4.0, timestamps=t6, feature_values=noisy_line)
    assert linear.status == "estimated"
    assert linear.fit_r_squared == _slope_r_squared(noisy_line, t6)

    exceeded = _run(
        failure_threshold=4.5,
        timestamps=[0.0, 1.0, 2.0],
        feature_values=[3.0, 4.0, 5.0],
    )
    assert exceeded.fit_r_squared is None


def test_the_result_schema_types_the_new_fields():
    schema = RULEstimationResult.model_json_schema()
    properties = schema["properties"]
    assert {"type": "null"} in properties["extrapolation_ratio"]["anyOf"]
    assert {"type": "number"} in properties["extrapolation_ratio"]["anyOf"]
    measurement_types = properties["measurements"]["anyOf"]
    assert {"type": "null"} in measurement_types
    array = next(t for t in measurement_types if t.get("type") == "array")
    assert array["items"] == {"$ref": "#/$defs/RULMeasurement"}
    assert {"type": "string"} in properties["feature_name"]["anyOf"]
    assert {"type": "null"} in properties["feature_name"]["anyOf"]
    assert "feature_name" in schema["required"]


# ---------------------------------------------------------------------------
# Model-facing text that states these limits
# ---------------------------------------------------------------------------


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_tool_description_states_the_threshold_and_series_limits():
    doc = _flat(prognostics_tools.estimate_rul.__doc__ or "")
    assert "standard choice" not in doc
    assert (
        doc.count(
            "is a threshold only for values comparable with that assessment: velocity "
            "RMS over its evaluation band, measured under comparable conditions."
        )
        == 1
    )
    assert (
        doc.count(
            "Only declared values are compared, so an accepted series is not thereby "
            "shown to come from one asset or point, or in acquisition order."
        )
        == 1
    )
    assert (
        doc.count(
            "A series whose first measurement is already at or above failure_threshold "
            "is refused"
        )
        == 1
    )


SKILL = REPO_ROOT / "plugin" / "skills" / "prognostics" / "SKILL.md"


@pytest.mark.parametrize(
    "sentence",
    [
        "otherwise `estimate_rul` refuses the series and names the recordings that disagree",
        "do not tell the user that an accepted series was confirmed as one machine, one "
        "point or in acquisition order",
        "A series whose first measurement is already at or above the threshold is refused",
        "Do not negate, shift or reorder the values.",
        "`extrapolation_ratio` (the RUL divided by the observation horizon)",
        "also report `measurements`",
    ],
)
def test_the_prognostics_skill_relays_the_checks_and_fields(sentence):
    assert _flat(sentence) in _flat(SKILL.read_text(encoding="utf-8"))


def test_the_architecture_guide_lists_every_prognostics_module():
    guide = (REPO_ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    block = guide.split("### Block 5", 1)[1].split("### Block 6", 1)[0]
    listed = set(re.findall(r"^\| `([a-z0-9_]+\.py)` \|", block, flags=re.MULTILINE))
    modules = {
        p.name
        for p in (REPO_ROOT / "src" / "prognostics").glob("*.py")
        if p.name != "__init__.py"
    }
    assert listed == modules
    assert "degradation forecasts" not in guide
