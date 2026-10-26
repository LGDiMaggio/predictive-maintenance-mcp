"""Tests for the reference backends and the sufficiency rules (forecast PR 1, U2).

Naive and seasonal naive follow Hyndman and Athanasopoulos, Forecasting:
Principles and Practice (3rd ed.), section 5.5, table 5.2, with sigma taken as
the uncentred RMS of grid-adjacent differences. The coverage screens keep the
nominal level, the measured rate and the out-of-model case apart.
"""

import math
from statistics import NormalDist

import numpy as np
import pytest
from scipy.stats import beta

from predictive_maintenance_mcp.prognostics import reference_forecast as rf
from predictive_maintenance_mcp.signal_validation import InvalidInputError

Z90 = NormalDist().inv_cdf(0.9)


def _run(values, horizon=6, backend="naive", season=None, max_supported=None):
    values = np.asarray(values, dtype=float)
    return rf.reference_forecast(
        values,
        backend=backend,
        horizon_steps=horizon,
        season_length_steps=season,
        max_supported_horizon_steps=(
            values.size if max_supported is None else max_supported
        ),
    )


def _clopper_pearson(k, n, alpha=0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


class TestNaive:
    def test_median_repeats_last_bin_and_band_grows_with_sqrt_h(self):
        """Covers AE1."""
        rng = np.random.default_rng(1)
        v = np.cumsum(rng.normal(0.0, 1.0, 300)) + 50.0
        out = _run(v, horizon=16)
        d = np.diff(v)
        sigma = math.sqrt(float(np.mean(d * d)))
        assert np.all(out.p50 == v[-1])
        for h in (1, 4, 16):
            assert out.p90[h - 1] - out.p50[h - 1] == pytest.approx(
                Z90 * sigma * math.sqrt(h)
            )
            assert out.p50[h - 1] - out.p10[h - 1] == pytest.approx(
                Z90 * sigma * math.sqrt(h)
            )
        assert out.difference_pairs == 299
        assert out.lag == 1

    def test_pair_across_a_gap_is_not_counted(self):
        a = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        gap = np.full(100, np.nan)
        b = np.array([50.0, 51.0, 50.0, 51.0, 50.0, 51.0])
        out = _run(np.concatenate([a, gap, b]), horizon=3)
        assert out.difference_pairs == 10
        assert out.sigma == pytest.approx(1.0)

    def test_quantiles_are_monotone(self):
        rng = np.random.default_rng(2)
        out = _run(np.cumsum(rng.normal(size=80)), horizon=30)
        assert np.all(out.p10 <= out.p50) and np.all(out.p50 <= out.p90)

    def test_too_few_pairs(self):
        """Covers AE3: eight observed adjacent bins give 7 pairs, 10 required."""
        out = _run([1.0, 2.0, 1.5, 3.0, 2.5, 2.0, 2.2, 2.9], horizon=6)
        assert isinstance(out, rf.Insufficient)
        assert out.reason_code == "too_few_difference_pairs"
        assert (out.available, out.required) == (7, rf.MIN_DIFFERENCE_PAIRS)
        assert rf.MIN_DIFFERENCE_PAIRS == 10

    def test_too_few_pairs_comes_before_the_horizon_check(self):
        out = _run(
            [1.0, 2.0, 1.5, 3.0, 2.5, 2.0, 2.2, 2.9], horizon=24, max_supported=7
        )
        assert out.reason_code == "too_few_difference_pairs"

    def test_constant_series_has_no_variation(self):
        """Covers AE4."""
        out = _run([3.2] * 50)
        assert out.reason_code == "no_variation"
        assert out.backend == "naive"

    def test_horizon_beyond_the_observed_span(self):
        rng = np.random.default_rng(3)
        out = _run(np.cumsum(rng.normal(size=40)), horizon=30, max_supported=20)
        assert out.reason_code == "horizon_exceeds_history"
        assert (out.available, out.required) == (20, 30)

    def test_overflowing_differences_are_refused(self):
        v = [1e308, -1e308] * 10
        with pytest.raises(InvalidInputError, match="too large in magnitude"):
            _run(v)


class TestSeasonalNaive:
    def _periodic(self, seasons=4, m=24, noise=0.0, seed=0):
        rng = np.random.default_rng(seed)
        base = np.sin(2 * np.pi * np.arange(m) / m) * 5.0 + 20.0
        return np.tile(base, seasons) + noise * rng.normal(size=seasons * m)

    def test_median_comes_from_the_source_bins(self):
        v = self._periodic(noise=0.3)
        out = _run(v, horizon=48, backend="seasonal_naive", season=24)
        assert np.allclose(out.p50[:24], v[-24:])
        assert np.allclose(out.p50[24:48], v[-24:])
        assert list(out.seasons_back[:24]) == [1] * 24
        assert list(out.seasons_back[24:48]) == [2] * 24
        d = v[24:] - v[:-24]
        sigma = math.sqrt(float(np.mean(d * d)))
        assert out.p90[30] - out.p50[30] == pytest.approx(Z90 * sigma * math.sqrt(2))

    def test_empty_source_phase_falls_back_one_season_and_widens(self):
        v = self._periodic(noise=0.3)
        v[-24 + 5] = np.nan
        out = _run(v, horizon=24, backend="seasonal_naive", season=24)
        assert out.seasons_back[5] == 2
        assert out.p50[5] == pytest.approx(v[-48 + 5])
        width_1 = out.p90[4] - out.p50[4]
        width_2 = out.p90[5] - out.p50[5]
        assert width_2 == pytest.approx(width_1 * math.sqrt(2))

    def test_unobserved_phase_needed_by_the_horizon(self):
        """Covers AE7: a phase empty in every season."""
        v = self._periodic(noise=0.3)
        v[7::24] = np.nan
        out = _run(v, horizon=24, backend="seasonal_naive", season=24)
        assert out.reason_code == "season_phase_unobserved"
        assert (out.available, out.required) == (23, 24)
        ok = _run(v, horizon=5, backend="seasonal_naive", season=24)
        assert isinstance(ok, rf.ReferenceForecast)

    def test_coverage_shorter_than_two_seasons(self):
        v = self._periodic(seasons=2, noise=0.3)[:34]
        out = _run(v, horizon=6, backend="seasonal_naive", season=24)
        assert out.reason_code == "season_coverage_too_short"
        assert (out.available, out.required) == (34, 48)

    def test_exactly_periodic_series_has_no_variation(self):
        v = self._periodic(noise=0.0)
        out = _run(v, horizon=6, backend="seasonal_naive", season=24)
        assert out.reason_code == "no_variation"
        assert out.backend == "seasonal_naive"

    def test_source_bins_match_a_direct_search(self):
        """The most recent observed same-phase bin, against a brute-force search."""
        rng = np.random.default_rng(7)
        checked = 0
        for m in (2, 3, 7, 24):
            v = self._periodic(seasons=30, m=m, noise=0.5, seed=m)
            v[rng.random(v.size) < 0.3] = np.nan
            v[-1] = 1.0
            n, horizon = v.size, 3 * m
            out = _run(v, horizon=horizon, backend="seasonal_naive", season=m)
            assert isinstance(out, rf.ReferenceForecast)
            for step in range(1, horizon + 1):
                for k in range(1, n):
                    j = n + step - 1 - k * m
                    if j < n and not np.isnan(v[j]):
                        break
                assert out.seasons_back[step - 1] == k
                assert out.p50[step - 1] == v[j]
                checked += 1
        assert checked == 3 * (2 + 3 + 7 + 24)

    def test_unobserved_phase_on_a_long_sparse_grid(self):
        """Every odd step lands on a phase empty across 99,999 bins."""
        v = np.full(99_999, np.nan)
        v[0:24:2] = np.arange(12, dtype=float) ** 1.5
        v[-1] = 7.0
        out = _run(v, horizon=256, backend="seasonal_naive", season=2)
        assert out.reason_code == "season_phase_unobserved"
        assert (out.available, out.required) == (128, 256)

    def test_no_variation_comes_before_the_horizon_check(self):
        v = self._periodic(noise=0.0)
        out = _run(v, horizon=30, backend="seasonal_naive", season=24, max_supported=10)
        assert out.reason_code == "no_variation"

    def test_seasonal_needs_ten_lag_pairs(self):
        m = 4
        v = self._periodic(seasons=3, m=m, noise=0.3)
        out = _run(v, horizon=2, backend="seasonal_naive", season=m)
        assert out.reason_code == "too_few_difference_pairs"
        assert (out.available, out.required) == (8, 10)


class TestCoverageScreens:
    """Measured rates on seeded families; never reseeded after the outcome."""

    def test_one_step_band_is_near_nominal_on_driftless_random_walks(self):
        """Screen: one family per seed index, 400 walks of 500 bins.

        Under the model (Gaussian increments, no drift) the share of one-step
        actuals inside P10..P90 should be 0.80. With 400 walks the
        Clopper-Pearson 95 % interval is about +/- 0.04 wide, so the screen
        detects a miscalibration of roughly 0.06 or more and is blind to
        smaller ones.
        """
        inside = 0
        n = 400
        for seed in range(n):
            rng = np.random.default_rng(10_000 + seed)
            walk = np.cumsum(rng.normal(size=501))
            out = _run(walk[:500], horizon=1)
            inside += int(out.p10[0] <= walk[500] <= out.p90[0])
        lo, hi = _clopper_pearson(inside, n)
        assert lo <= 0.80 <= hi, (inside, lo, hi)

    def test_out_of_model_drift_leaves_the_band_on_the_trend_side(self):
        """Out-of-model family: drift 0.2 sigma per step, horizon 24.

        The naive band assumes no trend, so the share of actuals above P90 at
        h = 24 is expected near 0.37, far above the nominal 0.10. The test
        records that limitation (R10); it asserts only that the measured share
        exceeds 0.10, with the Clopper-Pearson lower bound.
        """
        above = 0
        n = 300
        for seed in range(n):
            rng = np.random.default_rng(20_000 + seed)
            walk = np.cumsum(rng.normal(loc=0.2, scale=1.0, size=524))
            out = _run(walk[:500], horizon=24)
            above += int(walk[523] > out.p90[23])
        lo, _ = _clopper_pearson(above, n)
        assert lo > 0.10, (above, lo)


def test_literal_vocabularies():
    assert rf.REASON_CODES == (
        "too_few_difference_pairs",
        "season_coverage_too_short",
        "season_phase_unobserved",
        "no_variation",
        "horizon_exceeds_history",
    )
    assert rf.QUANTILE_LEVELS == (0.1, 0.5, 0.9)
    assert rf.MIN_SEASONS_COVERED == 2
