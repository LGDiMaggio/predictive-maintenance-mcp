"""Tests for the indicator projection assembly (forecast PR 1, U3).

Golden fixture ``tests/fixtures/indicator_forecast_golden.json`` holds the
numbers of one seeded naive and one seeded seasonal-naive projection, keyed by
``FORECAST_ALGORITHM_VERSION``. It trips when the numbers change without a
version bump. Regenerate it ONLY together with a version bump, in the locked
environment, from the worktree root::

    python -c "import tests.conftest; from tests.test_indicator_forecast import write_golden; write_golden()"
"""

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from predictive_maintenance_mcp.prognostics import indicator_forecast as f
from predictive_maintenance_mcp.prognostics import indicator_series as s
from predictive_maintenance_mcp.prognostics import reference_forecast as rf
from predictive_maintenance_mcp.signal_validation import InvalidInputError

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
GOLDEN = Path(__file__).parent / "fixtures" / "indicator_forecast_golden.json"
MODULES = (f, s, rf)


def _series(
    n=200, step=timedelta(hours=1), end=NOW - timedelta(hours=1), seed=7, offset="Z"
):
    rng = np.random.default_rng(seed)
    start = end - (n - 1) * step
    tz = timezone.utc if offset == "Z" else timezone(timedelta(hours=2))
    instants = []
    for i in range(n):
        t = (start + i * step).astimezone(tz)
        instants.append(t.isoformat().replace("+00:00", "Z"))
    values = list(np.round(np.cumsum(rng.normal(0, 0.2, n)) + 4.0, 6))
    return instants, values


def _forecast(observed_at, values, now=NOW, server_version="0.0.0-test", **kw):
    kwargs = dict(
        indicator="velocity RMS",
        unit="mm/s",
        resolution="1h",
        horizon_steps=24,
        backend="naive",
        aggregation="mean",
        season_length_steps=None,
    )
    kwargs.update(kw)
    return f.forecast_indicator(
        observed_at=observed_at,
        values=values,
        now=now,
        server_version=server_version,
        **kwargs,
    )


def _golden_payload():
    out = {}
    obs, vals = _series(n=120, seed=11)
    r = _forecast(obs, vals, horizon_steps=12)
    out["naive"] = {k: r[k] for k in ("p10", "p50", "p90")}
    rng = np.random.default_rng(12)
    base = np.sin(2 * np.pi * np.arange(24) / 24) * 3 + 10
    seasonal_vals = list(np.round(np.tile(base, 5) + rng.normal(0, 0.2, 120), 6))
    r = _forecast(
        obs,
        seasonal_vals,
        backend="seasonal_naive",
        season_length_steps=24,
        horizon_steps=30,
    )
    out["seasonal_naive"] = {k: r[k] for k in ("p10", "p50", "p90")}
    return out


def write_golden():
    data = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}
    data[str(f.FORECAST_ALGORITHM_VERSION)] = _golden_payload()
    GOLDEN.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class TestForecastPayload:
    def test_forecast_status_and_shape(self):
        obs, vals = _series()
        r = _forecast(obs, vals)
        assert r["status"] == "forecast"
        assert len(r["p10"]) == len(r["p50"]) == len(r["p90"]) == 24
        assert r["last_observed_at"] == "2026-10-24T11:00:00Z"
        assert r["forecast_start"] == "2026-10-24T12:00:00Z"
        assert r["horizon_end"] == "2026-10-25T12:00:00Z"
        assert r["method_note"] == f.METHOD_NOTES["naive"]
        assert r["interval_note"] == f.INTERVAL_NOTE
        assert r["interpretation"] == f.INTERPRETATION
        assert r["values_meaning"] == (
            "Values are the mean of velocity RMS in each 1h bin, in mm/s, as stated in "
            "the call. P50 projects that bin value; P10 and P90 bound a nominal band "
            "around it."
        )
        assert r["reason_code"] is None
        assert r["provenance"]["family"] == "indicator_forecast"
        assert r["provenance"]["algorithm_version"] == f.FORECAST_ALGORITHM_VERSION
        assert r["history"]["difference_lag"] == 1

    def test_anchors_carry_absolute_bins_and_match_columns(self):
        obs, vals = _series()
        r = _forecast(obs, vals)
        steps = [a["step"] for a in r["anchors"]]
        assert steps == [1, 8, 16, 24]
        for a in r["anchors"]:
            assert a["p50"] == r["p50"][a["step"] - 1]
        first = r["anchors"][0]
        assert (first["bin_start"], first["bin_end"]) == (
            "2026-10-24T12:00:00Z",
            "2026-10-24T13:00:00Z",
        )

    def test_offset_input_gives_utc_output(self):
        obs, vals = _series(offset="+02:00")
        r = _forecast(obs, vals)
        assert r["last_observed_at"].endswith("Z")
        assert r["anchors"][0]["bin_start"].endswith("Z")

    @pytest.mark.parametrize(
        "aggregation, phrase",
        [("max", "the maximum of"), ("last", "the last reading of")],
    )
    def test_values_meaning_names_the_aggregation(self, aggregation, phrase):
        obs, vals = _series()
        r = _forecast(obs, vals, aggregation=aggregation)
        assert r["values_meaning"].startswith(f"Values are {phrase} velocity RMS")

    @pytest.mark.parametrize(
        "horizon, steps",
        [(1, [1]), (2, [1, 2]), (3, [1, 2, 3]), (4, [1, 3, 4]), (5, [1, 2, 3, 5])],
    )
    def test_anchor_steps_at_short_horizons(self, horizon, steps):
        obs, vals = _series()
        r = _forecast(obs, vals, horizon_steps=horizon)
        assert [a["step"] for a in r["anchors"]] == steps
        assert r["anchors"][0]["bin_start"] == r["forecast_start"]
        assert r["anchors"][-1]["bin_end"] == r["horizon_end"]

    def test_values_are_rounded_to_six_significant_digits(self):
        obs, vals = _series()
        r = _forecast(obs, vals)
        assert all(float(f"{x:.6g}") == x for x in r["p10"] + r["p50"] + r["p90"])

    def test_a_narrow_band_on_a_large_level_keeps_its_width(self):
        """At six digits a band of 0.02 on a level near 1e5 would round to zero."""
        rng = np.random.default_rng(5)
        obs, _ = _series(n=48)
        vals = list(123456.7 + np.cumsum(rng.normal(0.0, 0.02, 48)))
        r = _forecast(obs, vals)
        assert r["status"] == "forecast"
        assert all(lo < hi for lo, hi in zip(r["p10"], r["p90"]))
        assert r["p50"][0] == pytest.approx(r["history"]["last_value"], abs=1e-3)
        assert r["p90"][0] - r["p10"][0] == pytest.approx(2 * rf.Z_P90 * 0.02, rel=0.5)

    def test_insufficient_history_payload(self):
        """Covers AE3: no projection columns, the history block is present."""
        obs, vals = _series(n=8)
        r = _forecast(obs, vals, horizon_steps=6)
        assert r["status"] == "insufficient_history"
        assert r["reason_code"] == "too_few_difference_pairs"
        assert (r["available"], r["required"]) == (7, 10)
        assert r["p10"] is None and r["anchors"] is None and r["method_note"] is None
        assert r["interpretation"] == f.INTERPRETATION
        assert "10" in r["remedy"] and "fill" in r["remedy"]
        assert r["history"]["raw_points"] == 8

    def test_max_supported_horizon_is_reported(self):
        obs, vals = _series(n=30)
        r = _forecast(obs, vals, horizon_steps=40)
        assert r["reason_code"] == "horizon_exceeds_history"
        assert r["max_supported_horizon_steps"] == 29
        assert "29" in r["remedy"]
        retry = _forecast(obs, vals, horizon_steps=29)
        assert retry["status"] == "forecast"

    @pytest.mark.parametrize("code", rf.REASON_CODES)
    def test_every_reason_has_a_reason_and_remedy(self, code):
        reason, remedy = f.explain_insufficient(
            rf.Insufficient(code, "naive", 3, 10, 1), f_decl(), max_supported=5
        )
        assert reason and remedy

    def test_no_variation_wording_follows_the_backend(self):
        naive_reason, _ = f.explain_insufficient(
            rf.Insufficient("no_variation", "naive", 0, 1, 1), f_decl(), 5
        )
        seasonal_reason, _ = f.explain_insufficient(
            rf.Insufficient("no_variation", "seasonal_naive", 0, 1, 24),
            f_decl("seasonal_naive"),
            5,
        )
        assert "stuck sensor" in naive_reason
        assert "did not change at the recorded resolution" in naive_reason
        assert "not evidence about the condition of the machine" in naive_reason
        assert "repeats exactly" in seasonal_reason and "stuck" not in seasonal_reason
        assert "repeats exactly at the recorded resolution" in seasonal_reason

    @pytest.mark.parametrize(
        "code, available, required, reason_part, remedy_part",
        [
            (
                "too_few_difference_pairs",
                6,
                10,
                "Only 6 pairs of observed bins one season (24 steps) apart",
                "at least 48 consecutive observed bins for a season of 24 steps",
            ),
            (
                "season_coverage_too_short",
                34,
                48,
                "covers 34 bins; seasonal naive needs at least 2 full seasons, 48 bins",
                "Supply at least 48 consecutive observed bins",
            ),
            (
                "season_phase_unobserved",
                23,
                24,
                "1 of the 24 requested steps fall on a phase",
                "covers every phase the horizon needs",
            ),
        ],
    )
    def test_seasonal_reasons_and_remedies(
        self, code, available, required, reason_part, remedy_part
    ):
        reason, remedy = f.explain_insufficient(
            rf.Insufficient(code, "seasonal_naive", available, required, 24),
            f_decl("seasonal_naive"),
            5,
        )
        assert reason_part in reason
        assert remedy_part in remedy

    def test_short_season_remedy_names_the_pair_minimum(self):
        """A season of 4 needs 14 bins for 10 pairs, more than its two seasons."""
        _, remedy = f.explain_insufficient(
            rf.Insufficient("season_coverage_too_short", "seasonal_naive", 6, 8, 4),
            f_decl("seasonal_naive"),
            5,
        )
        assert "at least 14 consecutive observed bins" in remedy

    def test_following_the_seasonal_pairs_remedy_gives_a_forecast(self):
        rng = np.random.default_rng(8)
        base = np.sin(2 * np.pi * np.arange(24) / 24) * 3 + 10
        obs, _ = _series(n=48)
        vals = list(np.tile(base, 2) + rng.normal(0, 0.2, 48))
        first = _forecast(
            obs[-30:],
            vals[-30:],
            backend="seasonal_naive",
            season_length_steps=24,
            horizon_steps=6,
        )
        assert first["reason_code"] == "too_few_difference_pairs"
        assert "at least 48 consecutive observed bins" in first["remedy"]
        retry = _forecast(
            obs, vals, backend="seasonal_naive", season_length_steps=24, horizon_steps=6
        )
        assert retry["status"] == "forecast"

    def test_seasonal_horizon_remedy_can_be_run_as_given(self):
        rng = np.random.default_rng(9)
        base = np.sin(2 * np.pi * np.arange(24) / 24) * 3 + 10
        obs, _ = _series(n=72)
        vals = list(np.tile(base, 3) + rng.normal(0, 0.2, 72))
        kw = dict(backend="seasonal_naive", season_length_steps=24)
        first = _forecast(obs, vals, horizon_steps=100, **kw)
        assert first["reason_code"] == "horizon_exceeds_history"
        retry = _forecast(
            obs, vals, horizon_steps=first["max_supported_horizon_steps"], **kw
        )
        assert retry["status"] == "forecast"

    @pytest.mark.parametrize("status", ["forecast", "insufficient_history"])
    def test_payload_keys_match_the_result_models(self, status):
        """model_validate drops unknown keys silently, so the two must agree."""
        from predictive_maintenance_mcp import models as m

        obs, vals = _series() if status == "forecast" else _series(n=8)
        r = _forecast(obs, vals, horizon_steps=6)
        assert r["status"] == status
        assert set(r) == set(m.IndicatorForecastResult.model_fields)
        assert set(r["history"]) == set(m.IndicatorForecastHistory.model_fields)
        assert set(r["provenance"]) == set(m.IndicatorForecastProvenance.model_fields)
        assert set(r["provenance"]["effective_declaration"]) == set(
            m.IndicatorForecastDeclaration.model_fields
        )
        for anchor in r["anchors"] or []:
            assert set(anchor) == set(m.IndicatorForecastAnchor.model_fields)


def f_decl(backend="naive"):
    return s.Declaration(
        indicator="velocity RMS",
        unit="mm/s",
        resolution="1h",
        step_seconds=3600,
        aggregation="mean",
        horizon_steps=6,
        backend=backend,
        season_length_steps=24 if backend == "seasonal_naive" else None,
    )


def _alternating(level, step, n=24, gap_to=None):
    """Hourly values alternating level and level + step, ending an hour before NOW.

    The uncentred RMS of the adjacent differences is ``step``, so the naive
    step-1 half-band is Z_P90 * step. ``gap_to`` appends one more value after
    a three-hour gap; a difference across a gap is not counted.
    """
    end = NOW - timedelta(hours=5 if gap_to is not None else 1)
    obs = [
        (end - (n - 1 - i) * timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        for i in range(n)
    ]
    vals = [level + (step if i % 2 else 0.0) for i in range(n)]
    if gap_to is not None:
        obs.append((NOW - timedelta(hours=1)).isoformat().replace("+00:00", "Z"))
        vals.append(gap_to)
    return obs, vals


def _outcome(level, half, horizon=256):
    """A naive projection with the given origin and step-1 half-band."""
    widen = np.sqrt(np.arange(1, horizon + 1, dtype=float))
    p50 = np.full(horizon, float(level))
    return rf.ReferenceForecast(
        "naive",
        p50 - half * widen,
        p50,
        p50 + half * widen,
        None,
        half / rf.Z_P90,
        100,
        1,
    )


def _seasonal_outcome(level, amplitude, half, m=24, horizon=96):
    """A seasonal projection: the median follows a sine, k seasons back."""
    steps = np.arange(horizon)
    p50 = level + amplitude * np.sin(2 * np.pi * steps / m)
    k = steps // m + 1
    widen = np.sqrt(k.astype(float))
    return rf.ReferenceForecast(
        "seasonal_naive",
        p50 - half * widen,
        p50,
        p50 + half * widen,
        k,
        half / rf.Z_P90,
        100,
        m,
    )


def _refusal(outcome, raw=None, unit="u"):
    raw = np.array([outcome.p50[-1]]) if raw is None else np.asarray(raw)
    with pytest.raises(InvalidInputError) as exc:
        f._refuse_unreportable_band(outcome, unit, raw)
    return str(exc.value)


def _serialized(outcome):
    digits = f._significant_digits(outcome)
    return digits, [
        f._round(q, digits) for q in (outcome.p10, outcome.p50, outcome.p90)
    ]


def _strictly_ordered(p10, p50, p90):
    return all(a < b < c for a, b, c in zip(p10, p50, p90))


class TestBandPrecision:
    """A band the quantiles cannot carry at the level of the values is refused.

    Up to MAX_LEVEL_TO_BAND_RATIO (1e7) the digits that rounding keeps hold
    BAND_DIGITS digits of the narrowest half-band, so no quantile moves by
    more than a hundredth of it and P10 < P50 < P90 survives serialization.
    """

    def test_the_threshold_is_derived_from_the_digit_constants(self):
        assert f.MAX_LEVEL_TO_BAND_RATIO == 1e7
        assert f.MAX_LEVEL_TO_BAND_RATIO == 10.0 ** (
            f.MAX_SIGNIFICANT_DIGITS - f.BAND_DIGITS
        )
        assert f.FLOAT_RESOLUTION_SPACINGS == 10**f.BAND_DIGITS

    @pytest.mark.parametrize("scale", [1e-300, 1e-9, 1.0, 1e15, 1e300])
    @pytest.mark.parametrize("sign", [1, -1])
    @pytest.mark.parametrize("ratio", [16.0, 1e3, 1e6, 9.99e6, 1e7 * (1 - 1e-6)])
    def test_serialized_quantiles_keep_their_order_and_width(self, scale, sign, ratio):
        """Ratio as the code defines it: largest |quantile| / narrowest half-band.

        Exactly 1e7 is tested at scale 1 in test_threshold_boundary, where the
        arithmetic is exact; at other scales it does not round back to 1e7.
        """
        half = scale
        outcome = _outcome(sign * (ratio - 16.0) * half, half)
        f._refuse_unreportable_band(outcome, "u", outcome.p50)
        digits, (p10, p50, p90) = _serialized(outcome)
        assert f.SIGNIFICANT_DIGITS <= digits <= f.MAX_SIGNIFICANT_DIGITS
        assert _strictly_ordered(p10, p50, p90)
        true = outcome.p90 - outcome.p10
        width = np.array(p90) - np.array(p10)
        assert np.max(np.abs(width - true) / true) <= 0.005
        for side in (np.array(p50) - np.array(p10), np.array(p90) - np.array(p50)):
            assert np.max(np.abs(side - true / 2) / (true / 2)) <= 0.01

    @pytest.mark.parametrize("sign", [1, -1])
    @pytest.mark.parametrize("ratio", [1e3, 1e6, 9.9e6])
    def test_seasonal_quantiles_keep_their_order_and_width(self, sign, ratio):
        """A median that changes per step, half-bands widening with k."""
        half = 1.0
        outcome = _seasonal_outcome(sign * (ratio / 2) * half, 5.0, half)
        f._refuse_unreportable_band(outcome, "u", outcome.p50)
        _, (p10, p50, p90) = _serialized(outcome)
        assert _strictly_ordered(p10, p50, p90)
        true = outcome.p90 - outcome.p10
        assert np.max(np.abs(np.array(p90) - np.array(p10) - true) / true) <= 0.005

    def test_threshold_boundary(self):
        at = _outcome(1e7 - 16.0, 1.0)  # ratio exactly 1e7
        f._refuse_unreportable_band(at, "u", at.p50)
        digits, quantiles = _serialized(at)
        assert digits == f.MAX_SIGNIFICANT_DIGITS
        assert _strictly_ordered(*quantiles)
        beyond = _outcome(1e7 - 16.0 + 0.5, 1.0)  # ratio 1e7 + 0.5
        assert "too narrow to report" in _refusal(beyond)

    @pytest.mark.parametrize("level", [1e9, -1e9])
    def test_bands_that_would_collapse_in_serialization_are_refused(self, level):
        """At ratio 1e12 ten digits would round the band to zero width."""
        outcome = _outcome(level, abs(level) / 1e12, horizon=1)
        p10, p90 = (
            f._round(q, f.MAX_SIGNIFICANT_DIGITS) for q in (outcome.p10, outcome.p90)
        )
        assert p10 == p90  # the premise: ten digits cannot carry this band
        msg = _refusal(outcome, unit="kWh")
        assert "The values sent still resolve this variation" in msg
        assert f"values near {level:.10g} kWh" in msg  # the sign is kept

    def test_a_band_collapsed_inside_float64_is_refused(self):
        """p50 +/- half rounds back to p50 while the method's half-band is 1."""
        outcome = _outcome(1e17, 1.0, horizon=4)
        assert np.all(outcome.p90 == outcome.p50)
        msg = _refusal(outcome)
        assert "narrowest half-width is 1 u on values near 1e+17 u" in msg
        assert "Fewer than three digits of this variation survive" in msg
        assert "export the series again" in msg

    def test_a_seasonal_pattern_no_constant_can_remove_is_refused(self):
        outcome = _seasonal_outcome(0.0, 1e6, 1e-3)
        msg = _refusal(outcome, raw=outcome.p50)
        assert "too narrow to report against its seasonal pattern" in msg
        assert "no constant subtracted from the values" in msg
        assert "use the naive backend" in msg

    def test_normal_band_end_to_end(self):
        obs, vals = _series()
        r = _forecast(obs, vals)
        assert r["status"] == "forecast"
        assert _strictly_ordered(r["p10"], r["p50"], r["p90"])

    def test_narrow_band_below_the_threshold_end_to_end(self):
        """Ratio about 5e6: reported, with the band to within half a percent."""
        step = 1e5 / 5e6 / rf.Z_P90
        obs, vals = _alternating(1e5, step)
        r = _forecast(obs, vals, horizon_steps=1)
        assert r["status"] == "forecast"
        assert _strictly_ordered(r["p10"], r["p50"], r["p90"])
        assert r["p90"][0] - r["p10"][0] == pytest.approx(
            2 * rf.Z_P90 * step, rel=0.005
        )

    @pytest.mark.parametrize("target, reported", [(0.99e7, True), (1.01e7, False)])
    def test_threshold_boundary_end_to_end(self, target, reported):
        step = 1e-3
        half = rf.Z_P90 * step
        obs, vals = _alternating(target * half - half - step, step)
        if reported:
            r = _forecast(obs, vals, horizon_steps=1)
            assert r["status"] == "forecast"
            assert _strictly_ordered(r["p10"], r["p50"], r["p90"])
        else:
            with pytest.raises(InvalidInputError, match="too narrow to report"):
                _forecast(obs, vals, horizon_steps=1)

    @pytest.mark.parametrize("sign", [1, -1])
    def test_large_offsets_end_to_end(self, sign):
        """A level of 1e9 is reported when the band is wide enough, else refused."""
        obs, vals = _alternating(sign * 1e9, 200.0)  # ratio about 3.9e6
        r = _forecast(obs, vals, horizon_steps=6)
        assert r["status"] == "forecast"
        assert _strictly_ordered(r["p10"], r["p50"], r["p90"])
        assert all(sign * x > 0 for x in r["p50"])
        obs, vals = _alternating(sign * 1e9, 1e-2)  # ratio about 7.8e10
        with pytest.raises(InvalidInputError) as exc:
            _forecast(obs, vals, horizon_steps=6, unit="kWh")
        msg = str(exc.value)
        assert f"values near {sign * 1e9:.10g} kWh" in msg
        assert "subtract the last observed value exactly" in msg
        assert "may not be enough" in msg

    def test_variation_at_the_float64_spacing_end_to_end(self):
        """Steps of one spacing at 1e17: the values carry under three digits."""
        spacing = float(np.spacing(1e17))
        obs, vals = _alternating(1e17, spacing)
        with pytest.raises(InvalidInputError) as exc:
            _forecast(obs, vals, horizon_steps=1)
        msg = str(exc.value)
        assert "Fewer than three digits of this variation survive" in msg
        assert "export the series again" in msg
        assert "The values sent still resolve" not in msg

    def test_band_collapsed_inside_float64_end_to_end(self):
        """A gap isolates a value of 1e17 from the small steps sigma comes from.

        The band collapses inside float64 (p50 +/- half == p50 at 1e17), but the
        values resolve the variation where it was measured: the cause named is
        the far-away value, not the rounding limit.
        """
        obs, vals = _alternating(1.0, 1e-3, gap_to=1e17)
        with pytest.raises(InvalidInputError) as exc:
            _forecast(obs, vals, horizon_steps=1)
        msg = str(exc.value)
        assert "comes from values near 1.001 mm/s, but its median reaches 1e+17" in msg
        assert "a unit change, a fill or sentinel value, a counter reset" in msg
        assert "Fewer than three digits" not in msg

    def test_seasonal_projection_end_to_end_keeps_its_order(self):
        rng = np.random.default_rng(13)
        base = np.sin(2 * np.pi * np.arange(24) / 24) * 3 + 1e5
        obs, _ = _series(n=96)
        vals = list(np.tile(base, 4) + rng.normal(0, 0.05, 96))
        r = _forecast(
            obs,
            vals,
            backend="seasonal_naive",
            season_length_steps=24,
            horizon_steps=48,
        )
        assert r["status"] == "forecast"
        assert len(set(r["p50"])) > 1
        assert _strictly_ordered(r["p10"], r["p50"], r["p90"])

    def test_seasonal_pattern_far_above_its_spread_end_to_end(self):
        rng = np.random.default_rng(14)
        base = np.sin(2 * np.pi * np.arange(24) / 24) * 1e6
        obs, _ = _series(n=96)
        vals = list(np.tile(base, 4) + rng.normal(0, 1e-3, 96))
        with pytest.raises(InvalidInputError, match="against its seasonal pattern"):
            _forecast(
                obs,
                vals,
                backend="seasonal_naive",
                season_length_steps=24,
                horizon_steps=6,
            )


class TestStaleness:
    def test_already_past_is_capped_at_the_horizon(self):
        obs, vals = _series(end=NOW - timedelta(hours=30))
        r = _forecast(obs, vals, horizon_steps=12)
        assert r["forecast_steps_already_past"] == 12
        stale = [w["detail"] for w in r["warnings"] if w["code"] == "stale_history"]
        assert "every projected step is already in the past" in stale[0]

    def test_stale_history(self):
        """Covers AE5: now 30 steps after the last instant."""
        obs, vals = _series(end=NOW - timedelta(hours=30))
        r = _forecast(obs, vals, horizon_steps=48)
        assert r["steps_since_last_observation"] == 30
        assert r["forecast_steps_already_past"] == 29
        assert r["warnings"][0]["code"] == "stale_history"
        assert "the first 29 projected step(s)" in r["warnings"][0]["detail"]

    def test_fresh_history_has_no_stale_warning(self):
        obs, vals = _series(end=NOW - timedelta(minutes=30))
        r = _forecast(obs, vals)
        assert r["forecast_steps_already_past"] == 0
        assert "stale_history" not in [w["code"] for w in r["warnings"]]

    def test_observation_after_the_issue_time(self):
        obs, vals = _series(end=NOW + timedelta(hours=3))
        r = _forecast(obs, vals)
        assert r["steps_since_last_observation"] == 0
        assert r["forecast_steps_already_past"] == 0
        assert "observed_after_issued_at" in [w["code"] for w in r["warnings"]]


class TestWarnings:
    def test_a_clean_regular_series_carries_no_warning(self):
        obs, vals = _series(end=NOW - timedelta(minutes=30))
        assert _forecast(obs, vals)["warnings"] == []

    def test_partial_last_bin(self):
        obs, vals = _series(
            n=121, step=timedelta(minutes=10), end=NOW - timedelta(hours=1)
        )
        r = _forecast(obs, vals, horizon_steps=6)
        assert r["history"]["last_bin_points"] == 1
        assert "partial_last_bin" in [w["code"] for w in r["warnings"]]

    def test_isolated_origin(self):
        obs, vals = _series(n=60)
        obs = obs[:50] + obs[-1:]
        vals = vals[:50] + vals[-1:]
        r = _forecast(obs, vals, horizon_steps=6)
        assert r["history"]["empty_bins_before_last_observed"] == 9
        assert "isolated_origin" in [w["code"] for w in r["warnings"]]

    def test_interval_crosses_observed_sign(self):
        rng = np.random.default_rng(3)
        obs, _ = _series(n=60)
        vals = list(np.round(0.2 + np.abs(rng.normal(0, 0.15, 60)), 6))
        r = _forecast(obs, vals, horizon_steps=24)
        assert min(r["p10"]) < 0
        assert "interval_crosses_observed_sign" in [w["code"] for w in r["warnings"]]

    def test_interval_crosses_observed_sign_for_a_negative_series(self):
        rng = np.random.default_rng(3)
        obs, _ = _series(n=60)
        vals = list(np.round(-0.2 - np.abs(rng.normal(0, 0.15, 60)), 6))
        r = _forecast(obs, vals, horizon_steps=24)
        assert max(r["p90"]) > 0
        details = [
            w["detail"]
            for w in r["warnings"]
            if w["code"] == "interval_crosses_observed_sign"
        ]
        assert "P90 rises above zero" in details[0]

    def test_a_mixed_sign_series_has_no_sign_warning(self):
        obs, vals = _series()
        vals = [v - 4.0 for v in vals]
        r = _forecast(obs, vals)
        assert min(r["history"]["min_value"], 0) < 0 < r["history"]["max_value"]
        assert "interval_crosses_observed_sign" not in [
            w["code"] for w in r["warnings"]
        ]

    def test_warning_codes_are_the_closed_vocabulary(self):
        assert f.WARNING_CODES == (
            "stale_history",
            "observed_after_issued_at",
            "partial_last_bin",
            "isolated_origin",
            "interval_crosses_observed_sign",
        )


class TestProvenance:
    def test_digest_ignores_now_and_server_version(self):
        obs, vals = _series()
        a = _forecast(obs, vals)
        b = _forecast(obs, vals, now=NOW + timedelta(hours=5), server_version="9.9.9")
        assert a["provenance"]["input_digest"] == b["provenance"]["input_digest"]
        assert a["p90"] == b["p90"]
        assert b["provenance"]["server_version"] == "9.9.9"

    @pytest.mark.parametrize(
        "change",
        [
            "value",
            "instant",
            "unit",
            "indicator",
            "resolution",
            "aggregation",
            "horizon",
        ],
    )
    def test_digest_changes_with_inputs(self, change):
        obs, vals = _series()
        base = _forecast(obs, vals)["provenance"]["input_digest"]
        kw = {}
        if change == "value":
            vals = vals[:-1] + [vals[-1] + 0.5]
        elif change == "instant":
            obs = obs[:-1] + [obs[-1].replace(":00:00Z", ":30:00Z")]
        elif change == "unit":
            kw["unit"] = "in/s"
        elif change == "indicator":
            kw["indicator"] = "acceleration RMS"
        elif change == "resolution":
            kw["resolution"] = "60min"
        elif change == "aggregation":
            kw["aggregation"] = "max"
        else:
            kw["horizon_steps"] = 12
        assert _forecast(obs, vals, **kw)["provenance"]["input_digest"] != base

    def test_digest_is_the_same_for_z_and_offset_instants(self):
        obs_z, vals = _series()
        obs_offset, _ = _series(offset="+02:00")
        assert obs_z != obs_offset
        assert (
            _forecast(obs_z, vals)["provenance"]["input_digest"]
            == _forecast(obs_offset, vals)["provenance"]["input_digest"]
        )

    def test_digest_changes_with_algorithm_version(self, monkeypatch):
        obs, vals = _series()
        base = _forecast(obs, vals)["provenance"]["input_digest"]
        monkeypatch.setattr(f, "FORECAST_ALGORITHM_VERSION", 99)
        assert _forecast(obs, vals)["provenance"]["input_digest"] != base

    def test_golden_numbers_for_the_current_algorithm_version(self):
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[
            str(f.FORECAST_ALGORITHM_VERSION)
        ]
        current = _golden_payload()
        for backend in ("naive", "seasonal_naive"):
            for q in ("p10", "p50", "p90"):
                np.testing.assert_allclose(
                    current[backend][q], golden[backend][q], rtol=1e-9
                )


# ---------------------------------------------------------------------------
# Static guards: wording and import isolation (each with a red path)
# ---------------------------------------------------------------------------

FORBIDDEN_PHRASES = (
    "probability of failure",
    "remaining useful life",
    "remaining-useful-life",
    "time to failure",
    "healthy",
    "stable machine",
    "confidence",
)

FORBIDDEN_IMPORT_PARTS = {
    "asset_ledger",
    "rul_estimator",
    "kalman_rul",
    "diagnostics",
    "decision_support",
    "models",
    "server",
    "mcp_tools",
    "mcp",
}


def _string_constants(source: str):
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
            ):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            yield node.value


def _forbidden_hits(source: str, exempt=()):
    hits = []
    for text in _string_constants(source):
        if text in exempt:
            continue
        lowered = text.lower()
        hits.extend(p for p in FORBIDDEN_PHRASES if p in lowered)
    return hits


def _forbidden_imports(source: str):
    hits = []
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [alias.name for alias in node.names]
        for name in names:
            if set(name.split(".")) & FORBIDDEN_IMPORT_PARTS:
                hits.append(name)
    return hits


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_no_server_authored_string_uses_forbidden_wording(module):
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert _forbidden_hits(source, exempt={f.INTERPRETATION}) == []


def test_forbidden_wording_guard_has_a_red_path():
    assert _forbidden_hits('X = "a time to failure estimate"') == ["time to failure"]


def test_interpretation_is_pinned():
    assert f.INTERPRETATION == (
        "A projection of the indicator's future values from its own history. It is not a "
        "fault detection, a failure probability or a remaining-useful-life estimate, and "
        "it says nothing about the condition of the machine."
    )


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_forecasting_modules_import_nothing_forbidden(module):
    """R17, checked on each module's own import statements.

    Side effects of the parent package initialisers are out of scope:
    ``src/__init__.py`` imports the server and ``src/prognostics/__init__.py``
    imports the RUL estimators.
    """
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert _forbidden_imports(source) == []


def test_import_guard_has_a_red_path():
    assert _forbidden_imports("from ..asset_ledger import store\nimport numpy") == [
        "asset_ledger"
    ]
